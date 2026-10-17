/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "akg/Conversion/VectorToPypto/VectorToPypto.h"

#include <iomanip>
#include <iterator>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>

#include "akg/Dialect/EmitPy/IR/EmitPy.h"
#include "akg/Dialect/EmitPy/IR/EmitPyOps.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "akg/Dialect/Pypto/IR/PyptoTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;  // NOLINT(build/namespaces)

namespace emitpy = mlir::emitpy;
namespace npuvector = mlir::npuvector;
namespace pypto = mlir::pypto;

namespace mlir {

#define GEN_PASS_DEF_CONVERTVECTORTOEMITPY
#include "akg/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PyPTO type mapping
//===----------------------------------------------------------------------===//

/// Maps an MLIR scalar element type onto the PyPTO dtype name, e.g.
/// f32 -> "pl.FP32", i32 -> "pl.INT32".
static std::optional<std::string> pyptoDtypeName(Type type) {
  if (isa<FloatType>(type)) {
    if (type.isF16()) {
      return std::string("pl.FP16");
    }
    if (type.isBF16()) {
      return std::string("pl.BF16");
    }
    if (type.isF32()) {
      return std::string("pl.FP32");
    }
    return std::nullopt;  // f64 and friends have no PyPTO dtype
  }
  if (auto intType = dyn_cast<IntegerType>(type)) {
    bool unsignedInt = intType.isUnsigned();
    switch (intType.getWidth()) {
    case 1:
      return std::string("pl.BOOL");
    case 8:
      return std::string(unsignedInt ? "pl.UINT8" : "pl.INT8");
    case 16:
      return std::string(unsignedInt ? "pl.UINT16" : "pl.INT16");
    case 32:
      return std::string(unsignedInt ? "pl.UINT32" : "pl.INT32");
    case 64:
      return std::string(unsignedInt ? "pl.UINT64" : "pl.INT64");
    default:
      return std::nullopt;
    }
  }
  if (isa<IndexType>(type)) {
    return std::string("pl.INT64");
  }
  return std::nullopt;
}

/// Renders a shape as a Python list literal, e.g. {4, 8} -> "[4, 8]".
static std::string shapeToPythonListText(ArrayRef<int64_t> shape) {
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << "[";
  llvm::interleaveComma(shape, rso);
  rso << "]";
  rso.flush();
  return text;
}

/// Builds the PyPTO annotation text for a shaped value, e.g.
/// "pl.Tile[[4, 8], pl.FP32]" or "pl.Tensor[[4, 8], pl.FP32]". Returns a null
/// Type for shapes PyPTO cannot express (dynamic dims, unsupported dtype).
static Type pyptoShapedOpaque(MLIRContext *ctx, ArrayRef<int64_t> shape,
                              Type elemType, StringRef wrapper) {
  auto dtype = pyptoDtypeName(elemType);
  if (!dtype) {
    return Type();
  }
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << wrapper << "[" << shapeToPythonListText(shape) << ", " << *dtype
      << "]";
  rso.flush();
  return emitpy::OpaqueType::get(ctx, text);
}

static Type pyptoShapedOpaque(MLIRContext *ctx, ShapedType type,
                              StringRef wrapper) {
  if (!type.hasStaticShape()) {
    return Type();
  }
  return pyptoShapedOpaque(ctx, type.getShape(), type.getElementType(),
                           wrapper);
}

/// Builds the PyPTO annotation text for a scalar, e.g. "pl.Scalar[pl.FP32]".
static Type pyptoScalarOpaque(MLIRContext *ctx, Type type) {
  auto dtype = pyptoDtypeName(type);
  if (!dtype) {
    return Type();
  }
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << "pl.Scalar[" << *dtype << "]";
  rso.flush();
  return emitpy::OpaqueType::get(ctx, text);
}

/// The type converter is deliberately the identity: the conversion
/// framework would otherwise materialize unrealized_conversion_cast ops for
/// every operand whose type maps to something else, and a full conversion
/// rejects those. Values inside the kernel keep their source types; only the
/// function signature is rewritten by hand afterwards, into opaque PyPTO
/// annotation types.
class VectorToEmitPyTypeConverter : public TypeConverter {
 public:
  explicit VectorToEmitPyTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
  }
};

/// Builds the opaque PyPTO annotation type for a kernel parameter. Kernel
/// parameters are memory buffers or scalars: every shaped parameter
/// (memref/tensor/vector/npuvector) crosses the kernel boundary as
/// `pl.Tensor[[...], dtype]` -- `pl.Tile` is not a legal parameter
/// annotation -- and scalars become `pl.Scalar[dtype]`. Tile-valued
/// parameters are loaded back into tiles at the entry (see the pass tail).
static Type pyptoAnnotation(MLIRContext *ctx, Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return pyptoShapedOpaque(ctx, shaped, "pl.Tensor");
  }
  if (isa<FloatType>(type) || isa<IntegerType>(type) ||
      isa<IndexType>(type)) {
    return pyptoScalarOpaque(ctx, type);
  }
  return Type();
}

//===----------------------------------------------------------------------===//
// Python text helpers
//===----------------------------------------------------------------------===//

/// Renders a floating-point value as Python source text.
static std::string floatToPythonText(APFloat value) {
  if (value.isInfinity()) {
    return value.isNegative() ? "float('-inf')" : "float('inf')";
  }
  if (value.isNaN()) {
    return "float('nan')";
  }
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10)
      << value.convertToDouble();
  std::string text = oss.str();
  if (text.find('.') == std::string::npos &&
      text.find('e') == std::string::npos &&
      text.find('E') == std::string::npos) {
    text += ".0";
  }
  return text;
}

/// Renders an integer value as Python source text.
static std::optional<std::string> intToPythonText(const APInt &value) {
  if (value.getSignificantBits() > 64) {
    return std::nullopt;
  }
  return std::to_string(value.getSExtValue());
}

/// Extracts the integer value of a constant index operand: either an
/// unconverted `arith.constant` or its already-converted `emitpy.literal`
/// form (patterns run in arbitrary order, so both can appear).
static std::optional<int64_t> constIndexValue(Value idx) {
  if (auto constOp = idx.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
    return std::nullopt;
  }
  if (auto literalOp = idx.getDefiningOp<emitpy::LiteralOp>()) {
    int64_t value;
    if (!llvm::to_integer(literalOp.getValue(), value)) {
      return std::nullopt;
    }
    return value;
  }
  return std::nullopt;
}

/// Extracts the integer value of a mixed static/dynamic bound or step, as
/// exposed by scf.forall: an IntegerAttr is the static form, a value (a
/// constant op) the dynamic one.
static std::optional<int64_t> mixedIndexValue(OpFoldResult bound) {
  if (bound.is<Attribute>()) {
    auto intAttr = dyn_cast<IntegerAttr>(bound.get<Attribute>());
    if (!intAttr) {
      return std::nullopt;
    }
    return intAttr.getInt();
  }
  return constIndexValue(bound.get<Value>());
}

/// PTOAS allocates tiles in whole 32-byte rows: a constant tile extent is
/// rounded up so `extent * elemBytes` is a multiple of 32. Only the
/// allocation is padded -- the valid shape still bounds the data actually
/// moved and computed, so tail iterations stay within the source buffer.
static int64_t alignExtentTo32B(int64_t extent, unsigned elemBytes) {
  if (elemBytes == 0 || extent <= 0) {
    return extent;
  }
  int64_t bytes = extent * static_cast<int64_t>(elemBytes);
  return llvm::alignTo(bytes, 32) / static_cast<int64_t>(elemBytes);
}

/// Renders constant index operands as a Python list literal, e.g.
/// "%c0, %c1" -> "[0, 1]". Returns nullopt when any index is not a constant.
static std::optional<std::string> constIndicesToText(ValueRange indices) {
  std::string text = "[";
  bool first = true;
  for (Value idx : indices) {
    auto value = constIndexValue(idx);
    if (!value) {
      return std::nullopt;
    }
    if (!first) {
      text += ", ";
    }
    first = false;
    text += std::to_string(*value);
  }
  text += "]";
  return text;
}

/// Builds the `[i0, i1, ...]` offsets argument for pl.load/pl.store/pl.write.
/// Constant-only offsets render directly as an opaque list literal; mixed or
/// dynamic offsets become a nested ArrayAttr whose elements are operand
/// references (IntegerAttr of index type), and the index values are appended
/// to `operands` so the emitter can print them by name. The Python emitter
/// prints such an array as `[<operand names and literals>]`, so e.g. a read
/// at a loop-dependent row emits as `pl.load(inputs, [row, 0], [4, 1024])`.
static Attribute buildOffsetsArg(OpBuilder &rewriter, Location loc,
                                 ValueRange indices,
                                 SmallVectorImpl<Value> &operands) {
  if (auto text = constIndicesToText(indices)) {
    return rewriter.getAttr<emitpy::OpaqueAttr>(*text);
  }
  SmallVector<Attribute> items;
  for (Value idx : indices) {
    operands.push_back(idx);
    items.push_back(rewriter.getIndexAttr(operands.size() - 1));
  }
  return rewriter.getArrayAttr(items);
}

/// Extracts the constant integer of a mixed static/dynamic entry, or nullopt
/// when the entry is a runtime value.
static std::optional<int64_t> staticIntValue(OpFoldResult value) {
  if (!value.is<Attribute>()) {
    return std::nullopt;
  }
  auto intAttr = dyn_cast<IntegerAttr>(value.get<Attribute>());
  if (!intAttr) {
    return std::nullopt;
  }
  return intAttr.getInt();
}

/// Builds a `[v0, v1, ...]` list argument from mixed static/dynamic entries,
/// the OpFoldResult counterpart of buildOffsetsArg: constants render inline
/// as an opaque list literal, dynamic values become operand references and
/// the values are appended to `operands` (the emitter prints such a list as
/// `[<operand names and literals>]`).
static Attribute buildMixedListArg(ConversionPatternRewriter &rewriter,
                                   ArrayRef<OpFoldResult> values,
                                   SmallVectorImpl<Value> &operands) {
  if (llvm::all_of(values, [](OpFoldResult value) {
        return staticIntValue(value).has_value();
      })) {
    std::string text = "[";
    bool first = true;
    for (OpFoldResult value : values) {
      if (!first) {
        text += ", ";
      }
      first = false;
      text += std::to_string(*staticIntValue(value));
    }
    text += "]";
    return rewriter.getAttr<emitpy::OpaqueAttr>(text);
  }
  SmallVector<Attribute> items;
  for (OpFoldResult value : values) {
    auto cst = staticIntValue(value);
    if (cst) {
      items.push_back(
          rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(*cst)));
      continue;
    }
    operands.push_back(value.get<Value>());
    items.push_back(rewriter.getIndexAttr(operands.size() - 1));
  }
  return rewriter.getArrayAttr(items);
}

/// Renders dropped-axis indices as a Python list literal, e.g. {0, 2} ->
/// "[0, 2]".
static std::string dimsToPythonText(ArrayRef<unsigned> dims) {
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << "[";
  llvm::interleaveComma(dims, rso);
  rso << "]";
  rso.flush();
  return text;
}

/// Replaces a view op with `pypto.slice`: the window is full-rank over the
/// source, with unit dims standing in for the axes a rank-reducing view
/// drops (an empty `droppedDims` keeps the op rank-preserving). Mixed
/// static/dynamic offsets are materialized as index constants, which the
/// main conversion turns into `emitpy.literal`; the sweep folds them back
/// into the `pl.slice` offsets list.
static void replaceWithSliceOp(Operation *op, Value source, Value result,
                               ArrayRef<int64_t> window,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<unsigned> droppedDims,
                               ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  SmallVector<Value> offsetValues;
  offsetValues.reserve(offsets.size());
  for (OpFoldResult offset : offsets) {
    if (auto cst = staticIntValue(offset)) {
      offsetValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, *cst));
    } else {
      offsetValues.push_back(offset.get<Value>());
    }
  }
  DenseI64ArrayAttr dropDimsAttr;
  if (!droppedDims.empty()) {
    dropDimsAttr = rewriter.getDenseI64ArrayAttr(
        llvm::to_vector_of<int64_t>(droppedDims));
  }
  // The replacement keeps the original result type: the type converter is
  // the identity, so a type-changing replacement would demand a
  // materialization at already-converted users and abort the full
  // conversion.
  rewriter.replaceOpWithNewOp<pypto::SliceOp>(
      op, result.getType(), source, window, ValueRange(offsetValues),
      dropDimsAttr);
}

/// Creates a scratch tile: `pl.create_tile(shape, dtype=...,
/// target_memory=pl.MemorySpace.Vec)`.
static Value createTmpTile(ConversionPatternRewriter &rewriter, Location loc,
                           ArrayRef<int64_t> shape, Type elemType) {
  auto dtype = pyptoDtypeName(elemType);
  assert(dtype && "caller validated the element type");
  std::string shapeText = shapeToPythonListText(shape);
  SmallVector<Attribute> args{
      rewriter.getAttr<emitpy::OpaqueAttr>(shapeText),
      rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
      rewriter.getAttr<emitpy::OpaqueAttr>("pl.MemorySpace.Vec")};
  SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                  rewriter.getStringAttr("dtype"),
                                  rewriter.getStringAttr("target_memory")};
  std::string annotation;
  llvm::raw_string_ostream rso(annotation);
  rso << "pl.Tile[" << shapeText << ", " << *dtype << "]";
  rso.flush();
  return rewriter
      .create<emitpy::CallOpaqueOp>(
          loc, emitpy::OpaqueType::get(rewriter.getContext(), annotation),
          "pl.create_tile", ValueRange{}, rewriter.getArrayAttr(args),
          rewriter.getArrayAttr(keywords))
      .getResult(0);
}

/// Views `input` as a `[rows, cols]` tile:
/// `pl.reshape(input, [rows, cols])` with a `pl.Tile[[rows, cols], dtype]`
/// result annotation.
static Value reshapeTileOp(ConversionPatternRewriter &rewriter, Location loc,
                           Value input, int64_t rows, int64_t cols,
                           Type elemType) {
  auto dtype = pyptoDtypeName(elemType);
  assert(dtype && "caller validated the element type");
  std::string annotation;
  llvm::raw_string_ostream rso(annotation);
  rso << "pl.Tile[[" << rows << ", " << cols << "], " << *dtype << "]";
  rso.flush();
  SmallVector<Attribute> reshapeArgs{
      rewriter.getIndexAttr(0),
      rewriter.getAttr<emitpy::OpaqueAttr>("[" + std::to_string(rows) +
                                           ", " + std::to_string(cols) +
                                           "]")};
  SmallVector<Attribute> reshapeKeywords{rewriter.getStringAttr(""),
                                         rewriter.getStringAttr("")};
  SmallVector<Value> reshapeOperands{input};
  return rewriter
      .create<emitpy::CallOpaqueOp>(
          loc, emitpy::OpaqueType::get(rewriter.getContext(), annotation),
          "pl.reshape", ValueRange(reshapeOperands),
          rewriter.getArrayAttr(reshapeArgs),
          rewriter.getArrayAttr(reshapeKeywords))
      .getResult(0);
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

/// Rewrites an operation into the matching pypto op, e.g.
/// `arith.addf %a, %b` -> `pypto.add %a, %b`.
/// Like OpToCallOpaquePattern below, the raw (non-adaptor) operands are
/// used on purpose: values already rewritten point at their replacements,
/// and anything else keeps its source type so the converter never
/// materializes casts. During the gradual migration the pypto op keeps the
/// source op's result type; the trailing pypto->emitpy sweep lowers it
/// into the `pl.*` call_opaque form.
template <typename SourceOp, typename PyptoOp>
class OpToPyptoOpPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<PyptoOp>(op, op->getResultTypes(),
                                         op->getOperands());
    return success();
  }
};

/// Rewrites signed min/max into the pypto op chosen by operand kind:
/// scalars go to `pypto.max`/`pypto.min` (PyPTO's scalar-only forms),
/// tiles keep `pypto.maximum`/`pypto.minimum` (elementwise, tile-only
/// forms).
template <typename SourceOp, typename ScalarOp, typename TileOp>
class MinMaxToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<VectorType, npuvector::NPUVectorType>(op.getLhs().getType())) {
      rewriter.replaceOpWithNewOp<TileOp>(op, op->getResultTypes(),
                                          op->getOperands());
    } else {
      rewriter.replaceOpWithNewOp<ScalarOp>(op, op->getResultTypes(),
                                            op->getOperands());
    }
    return success();
  }
};

/// Rewrites an operation into `emitpy.call_opaque` with the given callee and
/// all operands passed positionally, e.g.
/// `arith.addf %a, %b` -> `%r = pl.add(a, b)`.
/// The raw (non-adaptor) operands are used on purpose: values already
/// rewritten point at their replacements, and anything else keeps its source
/// type so the converter never materializes casts.
template <typename SourceOp>
class OpToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  OpToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context,
                        StringRef callee)
      : OpConversionPattern<SourceOp>(typeConverter, context), callee(callee) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResultTypes(), callee, op->getOperands());
    return success();
  }

 private:
  StringRef callee;
};

/// Rewrites signed min/max into a callee chosen by operand kind: scalars go
/// to `pl.min`/`pl.max` (PyPTO's scalar-only forms), tiles keep
/// `pl.minimum`/`pl.maximum` (elementwise, tile-only forms).
template <typename SourceOp>
class MinMaxToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  MinMaxToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context,
                            StringRef scalarCallee, StringRef tileCallee)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        scalarCallee(scalarCallee), tileCallee(tileCallee) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef callee = isa<VectorType, npuvector::NPUVectorType>(
                           op.getLhs().getType())
                           ? tileCallee
                           : scalarCallee;
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResultTypes(), callee, op->getOperands());
    return success();
  }

 private:
  StringRef scalarCallee;
  StringRef tileCallee;
};

/// `pl.cmp` comparison type codes: 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge.
static std::optional<unsigned> cmpTypeCodeFor(arith::CmpIPredicate predicate) {
  switch (predicate) {
  case arith::CmpIPredicate::eq:
    return 0u;
  case arith::CmpIPredicate::ne:
    return 1u;
  case arith::CmpIPredicate::slt:
  case arith::CmpIPredicate::ult:
    return 2u;
  case arith::CmpIPredicate::sle:
  case arith::CmpIPredicate::ule:
    return 3u;
  case arith::CmpIPredicate::sgt:
  case arith::CmpIPredicate::ugt:
    return 4u;
  case arith::CmpIPredicate::sge:
  case arith::CmpIPredicate::uge:
    return 5u;
  }
  return std::nullopt;
}

static std::optional<unsigned> cmpTypeCodeFor(arith::CmpFPredicate predicate) {
  switch (predicate) {
  case arith::CmpFPredicate::OEQ:
    return 0u;
  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return 1u;
  case arith::CmpFPredicate::OLT:
    return 2u;
  case arith::CmpFPredicate::OLE:
    return 3u;
  case arith::CmpFPredicate::OGT:
    return 4u;
  case arith::CmpFPredicate::OGE:
    return 5u;
  default:
    // NaN-sensitive unordered predicates have no pl.cmp counterpart.
    return std::nullopt;
  }
}

/// Scalar comparison pseudo-callees, indexed by the `pl.cmp` type code
/// (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge). The PyPTO runtime's `pl.cmp`
/// accepts Tensor/Tile operands only, so scalar comparisons (e.g. the
/// cmp/div/select chains left behind by the built-in affine floordiv
/// lowering) go through these pseudo-callees, which the Python translator
/// prints as infix comparisons that the PyPTO parser maps onto ir.lt etc.
static StringRef scalarCmpCalleeFor(unsigned cmpTypeCode) {
  static const StringRef callees[] = {"util_cmp_eq", "util_cmp_ne",
                                      "util_cmp_lt", "util_cmp_le",
                                      "util_cmp_gt", "util_cmp_ge"};
  return cmpTypeCode < std::size(callees) ? callees[cmpTypeCode] : StringRef();
}

/// Rewrites an integer/float comparison into `pypto.cmp`: the predicate
/// maps onto the matching PyPTO comparison kind, and the `pl.cmp` type
/// codes coincide with the enum values (0=eq .. 5=ge). Comparisons without
/// vector operands (scalars) become infix-comparison pseudo-callees
/// instead, since `pl.cmp` has no scalar overload.
template <typename SourceOp>
class CmpToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto code = cmpTypeCodeFor(op.getPredicate());
    if (!code) {
      return rewriter.notifyMatchFailure(
          op, "comparison predicate has no pl.cmp counterpart");
    }
    if (!llvm::any_of(op->getOperands(), [](Value operand) {
          return isa<VectorType, npuvector::NPUVectorType>(operand.getType());
        })) {
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
          op, op->getResult(0).getType(), scalarCmpCalleeFor(*code),
          op->getOperands());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::CmpOp>(
        op, op->getResult(0).getType(),
        static_cast<pypto::CmpPredicate>(*code), op->getOperands()[0],
        op->getOperands()[1]);
    return success();
  }
};

/// Rewrites a numeric cast into `pypto.cast`: the target dtype travels in
/// the result type, and the sweep re-derives the `pl.cast(v, pl.<dtype>)`
/// argument from it.
template <typename SourceOp>
class CastToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::CastOp>(op, resultType,
                                               op->getOperand(0));
    return success();
  }
};

/// Rewrites a bitcast into `pypto.reinterpret_view`: the target dtype
/// travels in the result type, and the sweep re-derives the
/// `pl.reinterpret_view(v, pl.<dtype>)` argument from it.
template <typename SourceOp>
class BitcastToPyptoPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    if (!pyptoDtypeName(getElementTypeOrSelf(resultType))) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::ReinterpretViewOp>(
        op, resultType, op->getOperand(0));
    return success();
  }
};

/// Rewrites `arith.constant` into EmitPy constant producers:
/// index -> `emitpy.literal`, splat tiles -> `pl.tile.full(shape, dtype, v)`,
/// scalars -> `emitpy.constant` with the Python literal as text. Scalar
/// literals stay Python values: PyPTO accepts them wherever a scalar operand
/// is expected.
class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
 public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    TypedAttr value = op.getValue();
    Type type = op.getType();

    // Index constants feed `emitpy.subscript` etc. and stay index-typed.
    if (isa<IndexType>(type)) {
      auto intAttr = dyn_cast<IntegerAttr>(value);
      if (!intAttr) {
        return rewriter.notifyMatchFailure(op, "unsupported index constant");
      }
      auto elem = intToPythonText(intAttr.getValue());
      if (!elem) {
        return rewriter.notifyMatchFailure(op, "unsupported index constant");
      }
      rewriter.replaceOpWithNewOp<emitpy::LiteralOp>(
          op, rewriter.getIndexType(), *elem);
      return success();
    }

    // Splat tile constants map onto `pl.tile.full(shape, dtype, value)`.
    if (auto dense = dyn_cast<DenseElementsAttr>(value)) {
      auto shaped = dyn_cast<ShapedType>(type);
      if (!shaped || !shaped.hasStaticShape() || !dense.isSplat()) {
        return rewriter.notifyMatchFailure(
            op, "only static splat tile constants map to pl.tile.full");
      }
      auto dtype = pyptoDtypeName(shaped.getElementType());
      if (!dtype) {
        return rewriter.notifyMatchFailure(op,
                                           "element type has no PyPTO dtype");
      }
      std::optional<std::string> elem;
      if (isa<FloatType>(shaped.getElementType())) {
        elem = floatToPythonText(dense.getSplatValue<APFloat>());
      } else {
        elem = intToPythonText(dense.getSplatValue<APInt>());
      }
      if (!elem) {
        return rewriter.notifyMatchFailure(op, "unsupported splat value");
      }
      rewriter.replaceOpWithNewOp<pypto::TileFullOp>(
          op, type, rewriter.getStringAttr(*elem), ValueRange{});
      return success();
    }

    std::optional<std::string> text;
    if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
      text = floatToPythonText(floatAttr.getValue());
    } else if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
      if (intAttr.getType().isInteger(1)) {
        text = intAttr.getValue().getBoolValue() ? std::string("True")
                                                 : std::string("False");
      } else {
        text = intToPythonText(intAttr.getValue());
      }
    }
    if (!text) {
      return rewriter.notifyMatchFailure(op, "unsupported constant value");
    }
    // Scalar constants carry their exact dtype (kept in the result type):
    // a bare Python literal would default to index/float64, and scalar ops
    // such as pl.write require the value dtype to match the tensor dtype.
    auto dtype = pyptoDtypeName(type);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "scalar type has no PyPTO dtype");
    }
    rewriter.replaceOpWithNewOp<pypto::ConstantOp>(op, type,
                                                   rewriter.getStringAttr(*text));
    return success();
  }
};

/// Rewrites `transfer_read` into
/// `%r = pl.load(src, [o...], [shape])`; offsets may be loop-dependent.
/// A dynamic `npuvector.transfer_read src[off] [sizes] [max], pad` instead
/// becomes `pl.load(src, [o...], shapes, valid_shape)`: `shapes` takes the
/// per-dimension upper bounds (`maxSizes`) so the tile is allocated at the
/// full extent, and `valid_shape` takes the runtime sizes so a tail tile
/// never reads or stores past the end of the source.
template <typename TransferReadOp>
class TransferReadPattern : public OpConversionPattern<TransferReadOp> {
 public:
  using OpConversionPattern<TransferReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransferReadOp op, typename TransferReadOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getMask()) {
      return rewriter.notifyMatchFailure(op, "masked reads are not supported");
    }
    auto shaped = dyn_cast<ShapedType>(op.getVector().getType());
    if (!shaped) {
      return rewriter.notifyMatchFailure(op,
                                         "read result must be a shaped type");
    }
    if constexpr (std::is_same_v<TransferReadOp,
                                 npuvector::TransferReadOp>) {
      if (!shaped.hasStaticShape()) {
        return rewriteDynamicRead(op, shaped, rewriter);
      }
    }
    if (!shaped.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "read result must be statically shaped");
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(op, op.getVector().getType(),
                                               op.getSource(), op.getIndices(),
                                               ValueRange{}, ValueRange{});
    return success();
  }

 private:
  /// Dynamic-extent read: one `dynamicSizes` value per `?` in the result
  /// shape. Static extents pass through unchanged; `maxSizes` (when present,
  /// same arity as `dynamicSizes`) names the tile's allocated extent.
  LogicalResult rewriteDynamicRead(npuvector::TransferReadOp op,
                                   ShapedType shaped,
                                   ConversionPatternRewriter &rewriter) const {
    unsigned numDynamic =
        llvm::count(shaped.getShape(), ShapedType::kDynamic);
    if (op.getDynamicSizes().size() != numDynamic) {
      return rewriter.notifyMatchFailure(
          op, "dynamic sizes must match the read's dynamic extents");
    }
    Location loc = op.getLoc();
    ValueRange maxSizes = op.getMaxSizes();
    // Per dimension: the tile's allocated extent, and the valid extent.
    SmallVector<Value> shapeValues;
    SmallVector<Value> validValues;
    unsigned dynIdx = 0;
    for (int64_t dim : shaped.getShape()) {
      if (dim == ShapedType::kDynamic) {
        Value size = op.getDynamicSizes()[dynIdx];
        validValues.push_back(size);
        Value alloc = dynIdx < maxSizes.size() ? maxSizes[dynIdx] : size;
        // PTOAS allocates whole 32-byte rows: pad a constant bound so the
        // allocation is addressable; the valid extent still bounds the
        // data actually moved.
        if (auto bound = constIndexValue(alloc)) {
          unsigned elemBytes =
              shaped.getElementType().getIntOrFloatBitWidth() / 8;
          alloc = rewriter.create<arith::ConstantIndexOp>(
              loc, alignExtentTo32B(*bound, elemBytes));
        }
        shapeValues.push_back(alloc);
        ++dynIdx;
        continue;
      }
      Value extent = rewriter.create<arith::ConstantIndexOp>(loc, dim);
      shapeValues.push_back(extent);
      validValues.push_back(extent);
    }
    rewriter.replaceOpWithNewOp<pypto::LoadOp>(
        op, op.getVector().getType(), op.getSource(), op.getIndices(),
        ValueRange(shapeValues), ValueRange(validValues));
    return success();
  }
};

/// Rewrites `transfer_write` into `pl.store(v, [o...], dest)`; offsets may
/// be loop-dependent. Device backends impose two extra rules:
/// - tile stores need a 2D tile and a matching-rank tensor, so rank-1
///   values are stored through a `[1, N]` reshape when the row is a whole
///   number of 32-byte units, and element-wise through `pl.write`
///   otherwise;
/// - a value that is just a broadcast scalar is written directly with
///   `pl.write(dest, [o...], scalar)` instead of materializing a tile.
template <typename TransferWriteOp>
class TransferWritePattern : public OpConversionPattern<TransferWriteOp> {
 public:
  using OpConversionPattern<TransferWriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransferWriteOp op, typename TransferWriteOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getMask()) {
      return rewriter.notifyMatchFailure(op, "masked writes are not supported");
    }
    Location loc = op.getLoc();

    // A broadcast scalar feeding only this write is written directly; the
    // broadcast (or its converted add-onto-carrier form) is left dead for
    // the post-conversion cleanup to drop.
    Value vector = op.getVector();
    if (Value scalar = scalarBroadcastSource(vector)) {
      writeScalar(rewriter, loc, op.getSource(), op.getIndices(), scalar);
      finishRewrite(rewriter, op);
      return success();
    }

    // A scalar value into a single-element destination: 0-D and [1] tiles
    // are the shape-normalized forms of scalars, which have no tile form on
    // the device -- write the value directly with pl.write, addressing
    // every destination dimension at 0.
    auto scalarDestType = dyn_cast<ShapedType>(op.getSource().getType());
    if (isa<FloatType, IntegerType, IndexType>(vector.getType()) &&
        scalarDestType && scalarDestType.hasStaticShape() &&
        scalarDestType.getNumElements() == 1) {
      writeScalarToSingleElement(rewriter, loc, op.getSource(), vector,
                                 scalarDestType);
      finishRewrite(rewriter, op);
      return success();
    }

    auto destType = dyn_cast<ShapedType>(op.getSource().getType());
    auto valueType = dyn_cast<ShapedType>(vector.getType());

    // Rank-1 values go through a [1, N] reshape: tile stores need a 2D
    // tile. The destination is either rank-1 (a row offset of 0 is
    // prepended) or rank-2 (the tile's own row offset is used directly).
    // The value extent may be dynamic; the read that allocated the tile
    // bounds N.
    if (valueType && destType && valueType.getRank() == 1 &&
        destType.hasStaticShape() &&
        (destType.getRank() == 1 || destType.getRank() == 2)) {
      if (failed(storeRankOneViaReshape(rewriter, loc, op, vector, destType,
                                        valueType))) {
        return failure();
      }
      finishRewrite(rewriter, op);
      return success();
    }

    rewriter.create<pypto::StoreOp>(loc, op.getVector(), op.getSource(),
                                    op.getIndices());
    finishRewrite(rewriter, op);
    return success();
  }

 private:
  /// If `vector` is a scalar broadcast feeding a single user -- either the
  /// still-unconverted broadcast op or its converted `pypto.add(carrier, s)`
  /// form -- returns the broadcast scalar.
  static Value scalarBroadcastSource(Value vector) {
    auto isScalar = [](Type type) {
      return isa<FloatType>(type) || isa<IntegerType>(type) ||
             isa<IndexType>(type);
    };
    Operation *def = vector.getDefiningOp();
    if (!def || !vector.hasOneUse()) {
      return Value();
    }
    if (auto bcast = dyn_cast<vector::BroadcastOp>(def)) {
      if (!isScalar(bcast.getSource().getType())) {
        return Value();
      }
      return bcast.getSource();
    }
    if (auto bcast = dyn_cast<npuvector::BroadcastOp>(def)) {
      if (!isScalar(bcast.getSource().getType())) {
        return Value();
      }
      return bcast.getSource();
    }
    if (auto add = dyn_cast<pypto::AddOp>(def)) {
      // The converted runtime-scalar broadcast: pypto.add(carrier, s)
      // with a statically-shaped pypto.tile_full carrier.
      Value scalar = add.getRhs();
      if (!isScalar(scalar.getType())) {
        return Value();
      }
      auto carrier = add.getLhs().getDefiningOp<pypto::TileFullOp>();
      if (!carrier || carrier->getNumOperands() != 0) {
        return Value();
      }
      return scalar;
    }
    return Value();
  }

  /// Reads one scalar element out of a tile: `pl.read(tile, [k])`.
  static Value scalarRead(ConversionPatternRewriter &rewriter, Location loc,
                          Value tile, Type elemType, int64_t index) {
    return rewriter
        .create<pypto::ReadOp>(loc, elemType, tile, ArrayRef<int64_t>{index})
        .getResult();
  }

  /// Writes one scalar into `dest` at `indices` with `pl.write`.
  static void writeScalar(ConversionPatternRewriter &rewriter, Location loc,
                          Value dest, ValueRange indices, Value scalar) {
    rewriter.create<pypto::WriteOp>(loc, dest, scalar, indices);
  }

  /// Writes a scalar into a single-element destination with `pl.write`,
  /// addressing every destination dimension at 0.
  static void writeScalarToSingleElement(ConversionPatternRewriter &rewriter,
                                         Location loc, Value dest, Value scalar,
                                         ShapedType destType) {
    SmallVector<Value> offsets;
    for (int64_t d = 0; d < destType.getRank(); ++d) {
      offsets.push_back(rewriter.create<emitpy::LiteralOp>(
          loc, rewriter.getIndexType(), "0"));
    }
    rewriter.create<pypto::WriteOp>(loc, dest, scalar, ValueRange(offsets));
  }

  /// The allocation bound of a dynamic rank-1 tile produced by an
  /// unconverted `transfer_read`: its first `maxSizes` entry, padded to
  /// whole 32-byte rows as the read's rewrite will do.
  static std::optional<int64_t> readAllocationBound(npuvector::TransferReadOp read, Value v) {
    ValueRange maxSizes = read.getMaxSizes();
    if (maxSizes.empty()) {
      return std::nullopt;
    }
    auto bound = constIndexValue(maxSizes[0]);
    if (!bound) {
      return std::nullopt;
    }
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      return alignExtentTo32B(*bound, elemBytes);
    }
    return bound;
  }

  /// The allocation bound carried by a rewritten dynamic read: the
  /// `pypto.load`'s padded-shape operand of the dynamic dimension.
  static std::optional<int64_t> loadAllocationBound(pypto::LoadOp load,
                                                    Value v) {
    if (load.getPaddedShapes().empty()) {
      return std::nullopt;
    }
    std::optional<int64_t> bound = constIndexValue(load.getPaddedShapes()[0]);
    if (!bound) {
      return std::nullopt;
    }
    // The padded extent already carries the padded allocation when the
    // read was rewritten first; padding again is a no-op then, and
    // repairs an unpadded bound otherwise.
    if (auto shaped = dyn_cast<ShapedType>(v.getType())) {
      unsigned elemBytes = shaped.getElementType().getIntOrFloatBitWidth() / 8;
      return alignExtentTo32B(*bound, elemBytes);
    }
    return bound;
  }

  /// Whether `op` is a converted element-wise tile op that keeps the
  /// producing read's allocation.
  static bool isElementWiseTileOp(Operation *op) {
    return isa<pypto::AddOp, pypto::SubOp, pypto::MulOp, pypto::DivOp,
               pypto::MaximumOp, pypto::MinimumOp>(op);
  }

  /// The statically-known allocated extent of a dynamic rank-1 tile: the
  /// tile was allocated by a `transfer_read` whose `maxSizes` bound its `?`
  /// extent, and element-wise users keep that allocation. Both the
  /// unconverted read and its `pypto.load` form are handled because patterns
  /// run in arbitrary order.
  static std::optional<int64_t> dynamicTileExtent(Value v,
                                                  unsigned depth = 0) {
    if (depth > 8) {
      return std::nullopt;
    }
    Operation *def = v.getDefiningOp();
    if (!def) {
      return std::nullopt;
    }
    auto tryOperands = [&](Operation *owner) -> std::optional<int64_t> {
      for (Value operand : owner->getOperands()) {
        if (!isa<ShapedType>(operand.getType())) {
          continue;
        }
        if (auto extent = dynamicTileExtent(operand, depth + 1)) {
          return extent;
        }
      }
      return std::nullopt;
    };
    if (auto read = dyn_cast<npuvector::TransferReadOp>(def)) {
      return readAllocationBound(read, v);
    }
    if (auto load = dyn_cast<pypto::LoadOp>(def)) {
      return loadAllocationBound(load, v);
    }
    // Converted element-wise tile ops keep the read's allocation.
    if (isElementWiseTileOp(def)) {
      return tryOperands(def);
    }
    // Unconverted element-wise ops keep the read's allocation.
    if (isa<arith::AddIOp, arith::AddFOp, arith::MulIOp, arith::MulFOp,
            arith::SubIOp, arith::SubFOp, arith::DivSIOp, arith::DivUIOp,
            arith::DivFOp, arith::MaxSIOp, arith::MinSIOp>(def)) {
      return tryOperands(def);
    }
    return std::nullopt;
  }

  /// Stores a rank-1 value through a `[1, N]` reshape: tile stores need a
  /// 2D tile. The destination is either rank-1 (a row offset of 0 is
  /// prepended) or rank-2 (the tile's own row offset is used directly).
  /// A dynamic extent resolves to the tile's allocated bound; the tile's
  /// valid shape still limits what is written, so tail iterations stay in
  /// bounds. Statically-sized sub-32-byte rows cannot hold a tile and fall
  /// back to element-wise `pl.write`s, which require a rank-1 destination
  /// and a constant base offset.
  static LogicalResult
  storeRankOneViaReshape(ConversionPatternRewriter &rewriter, Location loc,
                         TransferWriteOp op, Value vector, ShapedType destType,
                         ShapedType valueType) {
    Type elemType = valueType.getElementType();
    // A dynamic extent names no element count of its own: the tile was
    // allocated by the producing read, whose maxSizes bound it.
    std::optional<int64_t> dynamicExtent;
    int64_t n = valueType.getNumElements();
    if (n == ShapedType::kDynamic) {
      dynamicExtent = dynamicTileExtent(vector);
      if (!dynamicExtent) {
        return rewriter.notifyMatchFailure(
            op, "dynamic rank-1 store needs the tile's allocated extent");
      }
      n = *dynamicExtent;
    }
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (!dynamicExtent && n * elemBytes % 32 != 0) {
      if (destType.getRank() != 1) {
        return rewriter.notifyMatchFailure(
            op, "sub-32B rows cannot back a 2D tile store");
      }
      // Sub-32B rows cannot hold a tile at all: fall back to
      // element-wise scalar writes, which need constant offsets.
      auto base = constIndexValue(op.getIndices()[0]);
      if (!base) {
        return rewriter.notifyMatchFailure(
            op, "element-wise fallback needs a constant offset");
      }
      for (int64_t k = 0; k < n; ++k) {
        Value elem = scalarRead(rewriter, loc, vector, elemType, k);
        SmallVector<Value> elemIndices;
        Value offset = rewriter.create<emitpy::LiteralOp>(
            loc, rewriter.getIndexType(), std::to_string(*base + k));
        elemIndices.push_back(offset);
        writeScalar(rewriter, loc, op.getSource(), elemIndices, elem);
      }
      return success();
    }
    Value reshaped = reshapeTileOp(rewriter, loc, vector, 1, n, elemType);
    SmallVector<Value> offsetItems;
    Value storeDest = op.getSource();
    if (destType.getRank() == 1) {
      // Prepend the [1, N] row offset, and store through a [1, N] view of
      // the flat destination: PTOAS matches both the offset count and the
      // partition shape to the destination's rank, so a 2D tile needs a
      // 2D destination view.
      offsetItems.push_back(rewriter.create<emitpy::LiteralOp>(
          loc, rewriter.getIndexType(), "0"));
      Type viewType;
      if (auto memrefType = dyn_cast<MemRefType>(destType)) {
        viewType = MemRefType::get({1, memrefType.getNumElements()},
                                   memrefType.getElementType());
      } else {
        auto tensorType = cast<RankedTensorType>(destType);
        viewType = RankedTensorType::get({1, tensorType.getNumElements()},
                                         tensorType.getElementType());
      }
      SmallVector<Attribute> viewArgs{
          rewriter.getIndexAttr(0),
          rewriter.getAttr<emitpy::OpaqueAttr>(
              shapeToPythonListText(cast<ShapedType>(viewType).getShape()))};
      SmallVector<Attribute> viewKeywords{rewriter.getStringAttr(""),
                                          rewriter.getStringAttr("")};
      // The view is a tensor-level reshape: create it at function scope,
      // next to the destination's own view chain, not beside the store.
      // The PyPTO emitter hoists entry-level view ops above the device
      // scope; created inside it, the reshape would be materialized as a
      // tile op, which tile.store rejects as its destination.
      OpBuilder::InsertionGuard guard(rewriter);
      if (Operation *destDef = op.getSource().getDefiningOp()) {
        rewriter.setInsertionPointAfter(destDef);
      } else if (auto funcOp = op->template getParentOfType<func::FuncOp>()) {
        rewriter.setInsertionPointToStart(&funcOp.getBody().front());
      }
      storeDest = rewriter
                      .create<emitpy::CallOpaqueOp>(
                          loc, viewType, "pl.reshape",
                          ValueRange(op.getSource()),
                          rewriter.getArrayAttr(viewArgs),
                          rewriter.getArrayAttr(viewKeywords))
                      .getResult(0);
    }
    llvm::append_range(offsetItems, op.getIndices());
    SmallVector<Value> storeOperands{reshaped, storeDest};
    Attribute indexArg =
        buildOffsetsArg(rewriter, loc, offsetItems, storeOperands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), indexArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        loc, TypeRange{}, "pl.store", ValueRange(storeOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }

  /// Removes the rewritten write.
  static void finishRewrite(ConversionPatternRewriter &rewriter,
                            TransferWriteOp op) {
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      // Result-ful writes alias the destination buffer.
      rewriter.replaceOp(op, op.getSource());
    }
  }
};

/// Reduction kinds map onto the pypto reduction families; bitwise
/// combining has no PyPTO counterpart.
static std::optional<pypto::ReduceKind> reduceKind(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
    return pypto::ReduceKind::sum;
  case vector::CombiningKind::MUL:
    return pypto::ReduceKind::prod;
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    return pypto::ReduceKind::min;
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    return pypto::ReduceKind::max;
  default:
    return std::nullopt;
  }
}

/// Whether the combining function has a scalar-form PyPTO op:
/// `pl.minimum`/`pl.maximum` have no scalar form.
static bool hasScalarCombine(pypto::ReduceKind kind) {
  return kind == pypto::ReduceKind::sum || kind == pypto::ReduceKind::prod;
}

/// Reads one element out of a tile.
static Value readTileElement(ConversionPatternRewriter &rewriter, Location loc,
                             Type resultType, Value tile,
                             ArrayRef<int64_t> index) {
  return rewriter.create<pypto::ReadOp>(loc, resultType, tile, index)
      .getResult();
}

/// Combines two reduction partials with the scalar-form op; a null `acc`
/// (the first element) passes through unchanged.
static Value combineScalars(ConversionPatternRewriter &rewriter, Location loc,
                            Type resultType, pypto::ReduceKind kind, Value acc,
                            Value elem) {
  if (!acc) {
    return elem;
  }
  if (kind == pypto::ReduceKind::prod) {
    return rewriter.create<pypto::MulOp>(loc, resultType, acc, elem);
  }
  return rewriter.create<pypto::AddOp>(loc, resultType, acc, elem);
}

/// Renders flat element index `k` of a col-major tile with extents `dims`
/// as an index list, e.g. [1, 2].
static SmallVector<int64_t> flatIndexToList(ArrayRef<int64_t> dims, int64_t k) {
  SmallVector<int64_t> index;
  for (unsigned d = 0; d < dims.size(); ++d) {
    int64_t stride = 1;
    for (unsigned t = d + 1; t < dims.size(); ++t) {
      stride *= dims[t];
    }
    index.push_back((k / stride) % dims[d]);
  }
  return index;
}

/// Rewrites reductions onto the PyPTO reduction families:
/// - npuvector partial reductions reduce the last axis (`pl.row_*`) or the
///   first axis (`pl.col_*`) and consume a scratch tile;
/// - full reductions (and `vector.reduction`) fold to a scalar through
///   row_* then col_*, reading out the single surviving element. Rank-1
///   inputs are reshaped to [1, N] first because pl.row_* requires 2D.
template <typename ReductionOp>
class ReductionPattern : public OpConversionPattern<ReductionOp> {
 public:
  using OpConversionPattern<ReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReductionOp op, typename ReductionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<pypto::ReduceKind> kind = reduceKind(op.getKind());
    if (!kind) {
      return rewriter.notifyMatchFailure(
          op, "reduction kind has no PyPTO counterpart");
    }

    auto shaped = dyn_cast<ShapedType>(op.getVector().getType());
    if (!shaped || !shaped.hasStaticShape() || shaped.getRank() == 0) {
      return rewriter.notifyMatchFailure(
          op, "reduction input must be a statically shaped tile");
    }
    if (!pyptoDtypeName(shaped.getElementType())) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }

    // Partial reduction: a single reduced axis that must be the first or the
    // last one. Rank-1 reductions always sweep the only axis and fall
    // through to the full-reduction lowering below.
    if constexpr (std::is_same_v<ReductionOp, npuvector::ReductionOp>) {
      auto dims = op.getReductionDims();
      if (dims && !dims->empty() && shaped.getRank() >= 2) {
        return rewritePartialReduction(op, rewriter, *kind, shaped);
      }
    }

    return rewriteFullReduction(op, rewriter, *kind, shaped);
  }

 private:
  /// Reduces one axis in place: the last axis (`row`) or the first axis
  /// (`col`), consuming a scratch tile and folding the accumulator with
  /// an elementwise add.
  static LogicalResult
  rewritePartialReduction(npuvector::ReductionOp op,
                          ConversionPatternRewriter &rewriter,
                          pypto::ReduceKind kind, ShapedType shaped) {
    Location loc = op.getLoc();
    auto dims = op.getReductionDims();
    if (dims->size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-axis reductions are supported");
    }
    int64_t dim = (*dims)[0];
    unsigned rank = shaped.getRank();
    pypto::ReduceAxis axis;
    if (dim == static_cast<int64_t>(rank - 1)) {
      axis = pypto::ReduceAxis::row;
    } else if (dim == 0) {
      axis = pypto::ReduceAxis::col;
    } else {
      return rewriter.notifyMatchFailure(
          op, "only first/last axis reductions map onto pl.row_*/pl.col_*");
    }
    Type resultType = op->getResult(0).getType();
    Value tmp = createTmpTile(rewriter, loc, shaped.getShape(),
                              shaped.getElementType());
    Value reduced = rewriter
                        .create<pypto::ReductionOp>(loc, resultType, kind,
                                                    axis, op.getVector(), tmp)
                        .getResult();
    if (op.getAcc()) {
      reduced = rewriter.create<pypto::AddOp>(loc, resultType, reduced,
                                              op.getAcc());
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }

  /// Folds a full reduction to a scalar. Device backends cannot allocate
  /// [1, 1] tiles, so the final value is assembled from scalar reads: when
  /// the operand tiles are allocatable (extents a whole number of 32-byte
  /// units) the last axis is reduced on device (`pl.row_*`) and only the
  /// surviving elements are read out; otherwise every element is read
  /// directly. The elements are combined with scalar-form ops --
  /// `pl.minimum`/`pl.maximum` have no scalar form, so min/max full
  /// reductions are rejected.
  static LogicalResult
  rewriteFullReduction(ReductionOp op, ConversionPatternRewriter &rewriter,
                       pypto::ReduceKind kind, ShapedType shaped) {
    Location loc = op.getLoc();
    if (!hasScalarCombine(kind)) {
      return rewriter.notifyMatchFailure(
          op, "full min/max reductions have no scalar-form PyPTO lowering");
    }
    if (op.getAcc()) {
      return rewriter.notifyMatchFailure(
          op, "accumulator chaining is not supported for full reductions");
    }

    Type resultType = op->getResult(0).getType();
    Type elemType = shaped.getElementType();
    Value input = op.getVector();
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    // The device walks a col-major tile's contiguous (row) axis in 32-byte
    // steps, so the smallest allocatable row extent is the quantum below
    // (8 rows for a 4-byte dtype).
    unsigned alignRows = 32 / std::gcd(32u, elemBytes);
    int64_t rows = shaped.getRank() == 2 ? shaped.getDimSize(0) : 0;
    int64_t cols = shaped.getRank() == 2 ? shaped.getDimSize(1) : 0;

    Value acc;
    if (shaped.getRank() == 1 && shaped.getNumElements() > 1 &&
        shaped.getNumElements() % alignRows == 0 &&
        (shaped.getNumElements() / alignRows) * elemBytes % 32 == 0) {
      acc = reduceRankOneViaReshape(rewriter, loc, kind, shaped, resultType,
                                    input, alignRows);
    } else if (shaped.getRank() == 2 && cols > 1 &&
               rows * elemBytes % 32 == 0 && cols * elemBytes % 32 == 0) {
      acc = reduceViaDeviceRowReduce(rewriter, loc, kind, shaped, resultType,
                                     input, rows);
    } else {
      acc = reduceViaScalarChain(rewriter, loc, kind, shaped,
                                 resultType, input);
    }
    rewriter.replaceOp(op, acc);
    return success();
  }

  /// Rank-1 lowering: view the tile as [alignRows, N/alignRows] so the
  /// device reduces the last axis into an allocatable [alignRows, 1] tile
  /// (a [1, N] view would reduce to [1, 1], which the device cannot
  /// allocate), then read out and combine the surviving row partials.
  static Value
  reduceRankOneViaReshape(ConversionPatternRewriter &rewriter, Location loc,
                          pypto::ReduceKind kind, ShapedType shaped,
                          Type resultType, Value input, unsigned alignRows) {
    int64_t r = alignRows;
    int64_t c = shaped.getNumElements() / r;
    Value reshaped =
        reshapeTileOp(rewriter, loc, input, r, c, shaped.getElementType());
    Value rowTmp = createTmpTile(rewriter, loc, SmallVector<int64_t>{r, c},
                                 shaped.getElementType());
    Value row = rewriter
                    .create<pypto::ReductionOp>(
                        loc, resultType, kind, pypto::ReduceAxis::row,
                        reshaped, rowTmp)
                    .getResult();
    return combineRowPartials(rewriter, loc, kind, resultType, row, r);
  }

  /// Hybrid lowering: row-reduce to [rows, 1] on device, then combine the
  /// surviving `rows` scalars.
  static Value
  reduceViaDeviceRowReduce(ConversionPatternRewriter &rewriter, Location loc,
                           pypto::ReduceKind kind, ShapedType shaped,
                           Type resultType, Value input, int64_t rows) {
    Value rowTmp = createTmpTile(rewriter, loc, shaped.getShape(),
                                 shaped.getElementType());
    Value row = rewriter
                    .create<pypto::ReductionOp>(
                        loc, resultType, kind, pypto::ReduceAxis::row,
                        input, rowTmp)
                    .getResult();
    return combineRowPartials(rewriter, loc, kind, resultType, row, rows);
  }

  /// Reads the surviving `[i, 0]` partials out of a keepdim row reduction
  /// and combines them with the scalar-form op.
  static Value combineRowPartials(ConversionPatternRewriter &rewriter,
                                  Location loc, pypto::ReduceKind kind,
                                  Type resultType, Value row, int64_t rows) {
    Value acc;
    for (int64_t i = 0; i < rows; ++i) {
      Value elem = readTileElement(rewriter, loc, resultType, row,
                                   SmallVector<int64_t>{i, 0});
      acc = combineScalars(rewriter, loc, resultType, kind, acc, elem);
    }
    return acc;
  }

  /// Pure scalar chain: read and combine every element.
  static Value
  reduceViaScalarChain(ConversionPatternRewriter &rewriter, Location loc,
                       pypto::ReduceKind kind, ShapedType shaped,
                       Type resultType, Value input) {
    SmallVector<int64_t> dims(shaped.getShape());
    Value acc;
    for (int64_t k = 0, e = shaped.getNumElements(); k < e; ++k) {
      Value elem = readTileElement(rewriter, loc, resultType, input,
                                   flatIndexToList(dims, k));
      acc = combineScalars(rewriter, loc, resultType, kind, acc, elem);
    }
    return acc;
  }
};

template <typename BroadcastOp>
static FailureOr<SmallVector<Value>> buildBroadcastExtents(
    BroadcastOp op, ShapedType resultType, Location loc,
    ConversionPatternRewriter &rewriter) {
  if (resultType.hasStaticShape())
    return SmallVector<Value>{};
  if constexpr (std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
    unsigned numDynamic = llvm::count(resultType.getShape(), ShapedType::kDynamic);
    if (op.getDynamicSizes().size() != numDynamic)
      return rewriter.notifyMatchFailure(
          op, "dynamic sizes must match the broadcast's dynamic extents");
    ValueRange maxSizes = op.getMaxSizes();
    SmallVector<Value> shapeValues;
    unsigned dynIdx = 0;
    for (int64_t dim : resultType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        Value alloc = dynIdx < maxSizes.size()
                          ? maxSizes[dynIdx]
                          : op.getDynamicSizes()[dynIdx];
        // Pad a constant bound to whole 32-byte rows, matching the extent
        // a dynamic transfer_read allocates, so broadcast tiles stay
        // shape-compatible with the loads they combine with.
        if (auto bound = constIndexValue(alloc)) {
          unsigned elemBytes =
              resultType.getElementType().getIntOrFloatBitWidth() / 8;
          alloc = rewriter.create<arith::ConstantIndexOp>(
              loc, alignExtentTo32B(*bound, elemBytes));
        }
        shapeValues.push_back(alloc);
        ++dynIdx;
      } else {
        shapeValues.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, dim));
      }
    }
    return shapeValues;
  }
  return rewriter.notifyMatchFailure(
      op, "only statically-shaped broadcasts are supported");
}

/// Rewrites a scalar broadcast:
/// - constant scalars materialize directly as `pl.tile.full(shape, dtype, v)`;
/// - runtime scalars are broadcast by adding them onto a zero carrier tile:
///   `pl.add(pl.tile.full(shape, dtype, 0), scalar)`. `pl.expands` has no
///   codegen on current device backends, while the scalar form of `pl.add`
///   broadcasts for free.
/// Dynamic `!npuvector` results take their `maxSizes` upper bounds as the
/// tile shape, matching the extent a dynamic transfer_read allocates.
/// Tile-to-tile broadcasts have no PyPTO counterpart.
template <typename BroadcastOp>
class BroadcastPattern : public OpConversionPattern<BroadcastOp> {
 public:
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, typename BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "broadcast result must be shaped");
    }
    if constexpr (!std::is_same_v<BroadcastOp, npuvector::BroadcastOp>) {
      if (!resultType.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            op, "only statically-shaped broadcasts are supported");
      }
    }
    auto dtype = pyptoDtypeName(resultType.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    Type sourceType = op.getSource().getType();
    if (!isa<FloatType, IntegerType, IndexType>(sourceType)) {
      return rewriter.notifyMatchFailure(
          op, "only scalar broadcasts map onto pl.tile.full/pl.expands");
    }

    Location loc = op.getLoc();
    Type convertedResult = op->getResult(0).getType();

    // The tile's allocated extent: static dims stay in the result type;
    // dynamic dims take their `maxSizes` upper bound (constants fold into
    // index constants, runtime bounds stay operand values).
    FailureOr<SmallVector<Value>> extents =
        buildBroadcastExtents(op, resultType, loc, rewriter);
    if (failed(extents))
      return failure();

    std::optional<std::string> constText;
    if (auto constOp =
            op.getSource().template getDefiningOp<arith::ConstantOp>()) {
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        constText = floatToPythonText(floatAttr.getValue());
      } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        if (!intAttr.getType().isInteger(1)) {
          constText = intToPythonText(intAttr.getValue());
        }
      }
    }

    Value result;
    if (constText) {
      result = rewriter
                   .create<pypto::TileFullOp>(loc, convertedResult,
                                              rewriter.getStringAttr(*constText),
                                              ValueRange(*extents))
                   .getResult();
    } else {
      std::string zero = isa<FloatType>(sourceType) ? "0.0" : "0";
      Value carrier =
          rewriter
              .create<pypto::TileFullOp>(loc, convertedResult,
                                         rewriter.getStringAttr(zero),
                                         ValueRange(*extents))
              .getResult();
      result = rewriter.create<pypto::AddOp>(loc, convertedResult, carrier,
                                             op.getSource())
                   .getResult();
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites a transpose into `pypto.transpose`: PyPTO swaps exactly
/// two axes, so the permutation must be a single transposition (or the
/// identity, which folds away).
template <typename TransposeOp>
class TransposePattern : public OpConversionPattern<TransposeOp> {
 public:
  using OpConversionPattern<TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, typename TransposeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ArrayRef<int64_t> permutation = op.getPermutation();
    std::optional<std::pair<int64_t, int64_t>> swap;
    for (auto [axis, target] : llvm::enumerate(permutation)) {
      if (target == static_cast<int64_t>(axis)) {
        continue;
      }
      if (swap) {
        // The far end of the same transposition is expected; anything else
        // is a second, independent displacement.
        if (static_cast<int64_t>(axis) == swap->second &&
            target == swap->first) {
          continue;
        }
        return rewriter.notifyMatchFailure(
            op, "permutation is not a single axis swap");
      }
      if (target < 0 || target >= static_cast<int64_t>(permutation.size()) ||
          permutation[target] != static_cast<int64_t>(axis)) {
        return rewriter.notifyMatchFailure(
            op, "permutation is not a single axis swap");
      }
      swap = std::make_pair(static_cast<int64_t>(axis), target);
    }
    if (!swap) {
      // Identity permutation: the value passes through unchanged.
      rewriter.replaceOp(op, op.getVector());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(
        op, op->getResult(0).getType(), op.getVector(),
        static_cast<uint64_t>(swap->first),
        static_cast<uint64_t>(swap->second));
    return success();
  }
};

/// Rewrites a select into `pypto.select` over an inline scratch tile.
/// Constant scalar branches are materialized as `pl.tile.full`, since the
/// runtime select requires tile operands. Scalar selects (e.g. the
/// branchless floordiv chain left by the built-in affine lowering) map
/// onto the `util_select` pseudo-callee, which the translator prints as a
/// Python conditional expression.
template <typename SelectOp>
class SelectPattern : public OpConversionPattern<SelectOp> {
 public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, typename SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Operand order is (condition, trueValue, falseValue) in both the IR
    // and the pseudo-callee; the translator reorders for Python syntax.
    if (!isa<ShapedType>(op->getResult(0).getType())) {
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
          op, op->getResult(0).getType(), "util_select", op->getOperands());
      return success();
    }
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "select result must be a static tile");
    }
    auto dtype = pyptoDtypeName(resultType.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    if (!isa<ShapedType>(op.getCondition().getType())) {
      return rewriter.notifyMatchFailure(op, "condition must be a tile");
    }

    Location loc = op.getLoc();
    Type convertedResult = op->getResult(0).getType();
    std::string shapeText = shapeToPythonListText(resultType.getShape());

    // pl.sel takes tiles; constant scalar branches become full tiles.
    auto materializeBranch = [&](Value branch) -> Value {
      if (isa<ShapedType>(branch.getType())) {
        return branch;
      }
      auto constOp = branch.getDefiningOp<arith::ConstantOp>();
      if (!constOp) {
        return Value();
      }
      std::optional<std::string> text;
      if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
        text = floatToPythonText(floatAttr.getValue());
      } else if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
        if (!intAttr.getType().isInteger(1)) {
          text = intToPythonText(intAttr.getValue());
        }
      }
      if (!text) {
        return Value();
      }
      SmallVector<Attribute> args{
          rewriter.getAttr<emitpy::OpaqueAttr>(shapeText),
          rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
          rewriter.getAttr<emitpy::OpaqueAttr>(*text)};
      SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                      rewriter.getStringAttr(""),
                                      rewriter.getStringAttr("")};
      return rewriter
          .create<emitpy::CallOpaqueOp>(
              loc, convertedResult, "pl.tile.full", ValueRange{},
              rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords))
          .getResult(0);
    };

    Value trueBranch = materializeBranch(op.getTrueValue());
    Value falseBranch = materializeBranch(op.getFalseValue());
    if (!trueBranch || !falseBranch) {
      return rewriter.notifyMatchFailure(
          op, "select branches must be tiles or constant scalars");
    }

    Value tmp = createTmpTile(rewriter, loc, resultType.getShape(),
                              resultType.getElementType());
    // The scratch tile and the materialized constant branches stay as
    // call_opaque producers: `pypto.create_tile` would demand a TileType
    // result the intermediate layer does not carry yet. The select itself
    // becomes the structured pypto op; the sweep restores the positional
    // `pl.sel` call.
    rewriter.replaceOpWithNewOp<pypto::SelectOp>(
        op, convertedResult, op.getCondition(), trueBranch, falseBranch, tmp);
    return success();
  }
};

/// Rewrites `memref.collapse_shape` into `pypto.reshape`: the flattened
/// tensor view PyPTO loads and stores through. Only statically shaped
/// views are supported; the shape travels in the result type and the sweep
/// re-derives the `pl.reshape(src, [shape])` argument from it.
class CollapseShapePattern
    : public OpConversionPattern<memref::CollapseShapeOp> {
 public:
  using OpConversionPattern<memref::CollapseShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CollapseShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically shaped collapse views are supported");
    }
    // The replacement keeps the original result type: the type converter is
    // the identity, so a type-changing replacement would demand a
    // materialization at already-converted users (e.g. a live `pl.load`) and
    // abort the full conversion. The raw source operand is used on purpose:
    // signature rewriting happens by hand after the conversion, so the
    // memref stays as-is.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(),
                                                  op.getSrc());
    return success();
  }
};

/// Rewrites `memref.expand_shape` into `pypto.reshape`, the mirror of the
/// collapse handling. Only statically shaped views are supported.
class ExpandShapePattern : public OpConversionPattern<memref::ExpandShapeOp> {
 public:
  using OpConversionPattern<memref::ExpandShapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ExpandShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically shaped expand views are supported");
    }
    // Same materialization constraints as the collapse handling: the
    // replacement keeps the original result type.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(),
                                                  op.getSrc());
    return success();
  }
};

/// Rewrites `memref.transpose` into `pypto.transpose`: PyPTO swaps
/// exactly two axes, so the permutation must be a single transposition (or
/// the identity, which folds away). The memref permutation is an AffineMap
/// whose result dim i comes from source dim `getDimPosition(i)`, the same
/// convention as the vector transpose.
class MemrefTransposePattern
    : public OpConversionPattern<memref::TransposeOp> {
 public:
  using OpConversionPattern<memref::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    AffineMap permutation = op.getPermutation();
    SmallVector<int64_t> mapping(permutation.getNumResults());
    for (auto [axis, dim] : llvm::enumerate(mapping)) {
      dim = permutation.getDimPosition(axis);
    }
    std::optional<std::pair<int64_t, int64_t>> swap;
    for (auto [axis, target] : llvm::enumerate(mapping)) {
      if (target == static_cast<int64_t>(axis)) {
        continue;
      }
      if (swap) {
        // The far end of the same transposition is expected; anything else
        // is a second, independent displacement.
        if (static_cast<int64_t>(axis) == swap->second &&
            target == swap->first) {
          continue;
        }
        return rewriter.notifyMatchFailure(
            op, "permutation is not a single axis swap");
      }
      if (target < 0 || target >= static_cast<int64_t>(mapping.size()) ||
          mapping[target] != static_cast<int64_t>(axis)) {
        return rewriter.notifyMatchFailure(
            op, "permutation is not a single axis swap");
      }
      swap = std::make_pair(static_cast<int64_t>(axis), target);
    }
    if (!swap) {
      // Identity permutation: the value passes through unchanged.
      rewriter.replaceOp(op, op.getIn());
      return success();
    }
    rewriter.replaceOpWithNewOp<pypto::TransposeOp>(
        op, op.getResult().getType(), op.getIn(),
        static_cast<uint64_t>(swap->first),
        static_cast<uint64_t>(swap->second));
    return success();
  }
};

/// Rewrites `memref.reshape` into `pypto.reshape`, matching the
/// collapse/expand handling. Only statically shaped results are supported;
/// the shape operand itself becomes dead and is swept by eraseDeadOps.
class ReshapePattern : public OpConversionPattern<memref::ReshapeOp> {
 public:
  using OpConversionPattern<memref::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically shaped reshapings are supported");
    }
    // Same materialization constraints as the collapse handling: the
    // replacement keeps the original result type.
    rewriter.replaceOpWithNewOp<pypto::ReshapeOp>(op, op.getResult().getType(),
                                                  op.getSource());
    return success();
  }
};

/// Rewrites `memref.subview` into `pypto.slice`: the window is the
/// subview's sizes (full-rank over the source, with unit dims standing in
/// for the axes a rank-reducing view drops) and the offsets carry any
/// dynamic entries. Non-unit strides have no PyPTO lowering.
class SubviewPattern : public OpConversionPattern<memref::SubViewOp> {
 public:
  using OpConversionPattern<memref::SubViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::SubViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically shaped subviews are supported");
    }
    auto srcType = dyn_cast<ShapedType>(op.getSource().getType());
    if (!srcType) {
      return rewriter.notifyMatchFailure(op,
                                         "unranked sources are unsupported");
    }
    for (OpFoldResult stride : op.getMixedStrides()) {
      auto cst = staticIntValue(stride);
      if (!cst || *cst != 1) {
        return rewriter.notifyMatchFailure(
            op, "only unit-stride subviews are supported");
      }
    }
    SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
    SmallVector<int64_t> window(resultType.getShape().begin(),
                                resultType.getShape().end());
    SmallVector<unsigned> droppedDims;
    if (resultType.getRank() < srcType.getRank()) {
      // A rank-reducing view: recover the full-rank window from the sizes,
      // where the dropped axes appear as unit dims, and record them for the
      // drop_dims argument of pl.slice.
      SmallVector<int64_t> sizes;
      for (OpFoldResult size : op.getMixedSizes()) {
        auto cst = staticIntValue(size);
        if (!cst) {
          return rewriter.notifyMatchFailure(
              op, "rank-reducing subviews need static sizes");
        }
        sizes.push_back(*cst);
      }
      auto dropped = computeRankReductionMask(sizes, resultType.getShape());
      if (!dropped) {
        return rewriter.notifyMatchFailure(
            op, "cannot map the rank-reduced window onto pl.slice");
      }
      window = std::move(sizes);
      droppedDims.assign(dropped->begin(), dropped->end());
      llvm::sort(droppedDims);
    }
    replaceWithSliceOp(op, op.getSource(), op.getResult(), window, offsets,
                       droppedDims, rewriter);
    return success();
  }
};

/// Builds the full-rank slice coordinates for a reinterpret cast with a
/// static flat offset.
static LogicalResult buildStaticReinterpretWindow(
    memref::ReinterpretCastOp op, MemRefType srcType, MemRefType resultType,
    ArrayRef<int64_t> srcStrides, ConversionPatternRewriter &rewriter,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<int64_t> &window,
    SmallVectorImpl<unsigned> &droppedDims) {
  int64_t rem = *staticIntValue(op.getMixedOffsets().front());
  unsigned srcRank = srcType.getRank();
  unsigned resultRank = resultType.getRank();
  for (unsigned axis = 0; axis < srcRank; ++axis) {
    int64_t coord = rem / srcStrides[axis];
    rem %= srcStrides[axis];
    offsets.push_back(rewriter.getIndexAttr(coord));
    if (axis < srcRank - resultRank) {
      window.push_back(1);
      droppedDims.push_back(axis);
    } else {
      window.push_back(resultType.getDimSize(axis - (srcRank - resultRank)));
    }
  }
  if (rem != 0)
    return rewriter.notifyMatchFailure(
        op, "offset does not decompose into source coordinates");
  return success();
}

/// Builds the full-rank slice coordinates for a reinterpret cast with a
/// dynamic flat offset.
static LogicalResult buildDynamicReinterpretWindow(
    memref::ReinterpretCastOp op, MemRefType srcType, MemRefType resultType,
    ArrayRef<OpFoldResult> strides, ConversionPatternRewriter &rewriter,
    SmallVectorImpl<OpFoldResult> &offsets,
    SmallVectorImpl<int64_t> &window,
    SmallVectorImpl<unsigned> &droppedDims) {
  auto stride0 = staticIntValue(strides[0]);
  if (resultType.getRank() != 1 || !stride0 || *stride0 != 1)
    return rewriter.notifyMatchFailure(
        op, "dynamic offsets only map onto the last source axis");
  unsigned srcRank = srcType.getRank();
  OpFoldResult flatOffset = op.getMixedOffsets().front();
  for (unsigned axis = 0; axis < srcRank; ++axis) {
    if (axis + 1 == srcRank) {
      offsets.push_back(flatOffset);
      window.push_back(resultType.getDimSize(0));
    } else {
      offsets.push_back(rewriter.getIndexAttr(0));
      window.push_back(1);
      droppedDims.push_back(axis);
    }
  }
  return success();
}

/// Rewrites `memref.reinterpret_cast` into `pypto.slice` when the cast
/// re-windows a region of its source that follows the source's own
/// trailing strides: the flat element offset (counted from the source's
/// base, per the descriptor semantics of the op) decomposes into per-axis
/// coordinates. Dynamic offsets only map when the window runs along the
/// last source axis, where the flat offset is that axis's coordinate.
class ReinterpretCastPattern
    : public OpConversionPattern<memref::ReinterpretCastOp> {
 public:
  using OpConversionPattern<memref::ReinterpretCastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::ReinterpretCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto resultType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically shaped reinterpretations are supported");
    }
    if (!srcType) {
      return rewriter.notifyMatchFailure(op,
                                         "unranked sources are unsupported");
    }
    unsigned srcRank = srcType.getRank();
    unsigned resultRank = resultType.getRank();
    if (resultRank == 0 || resultRank > srcRank) {
      return rewriter.notifyMatchFailure(
          op, "result rank must fit inside the source");
    }
    SmallVector<int64_t> srcStrides;
    int64_t srcOffset = 0;
    if (failed(getStridesAndOffset(srcType, srcStrides, srcOffset)) ||
        srcOffset != 0) {
      return rewriter.notifyMatchFailure(
          op, "only zero-offset strided sources are supported");
    }
    // The result axes must walk the source's trailing axes one element at a
    // time for pl.slice to express the window.
    auto strides = op.getMixedStrides();
    for (unsigned axis = 0; axis < resultRank; ++axis) {
      auto cst = staticIntValue(strides[axis]);
      if (!cst || *cst != srcStrides[srcRank - resultRank + axis]) {
        return rewriter.notifyMatchFailure(
            op, "strides must match the source's trailing layout");
      }
    }
    SmallVector<OpFoldResult> offsets;
    SmallVector<int64_t> window;
    SmallVector<unsigned> droppedDims;
    if (staticIntValue(op.getMixedOffsets().front())) {
      if (failed(buildStaticReinterpretWindow(
              op, srcType, resultType, srcStrides, rewriter, offsets, window,
              droppedDims)))
        return failure();
    } else if (failed(buildDynamicReinterpretWindow(
                   op, srcType, resultType, strides, rewriter, offsets, window,
                   droppedDims))) {
      return failure();
    }
    replaceWithSliceOp(op, op.getSource(), op.getResult(), window, offsets,
                       droppedDims, rewriter);
    return success();
  }
};

/// Drops `memref.dealloc`: the buffers become kernel arguments whose
/// lifetime the runtime manages.
class DeallocPattern : public OpConversionPattern<memref::DeallocOp> {
 public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrites `scf.parallel` into an `emitpy.spmd_scope` multi-core launch:
/// a reduction-free single induction variable sweeping `[0, N)` with step 1
/// maps onto a PyPTO SPMD grid of N cores, the induction variable being the
/// executing core's block index.
class ScfParallelPattern : public OpConversionPattern<scf::ParallelOp> {
 public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumLoops() != 1 || !op.getInitVals().empty() ||
        op.getNumResults() != 0) {
      return rewriter.notifyMatchFailure(
          op, "only reduction-free single-induction parallel loops map onto "
              "an SPMD grid");
    }
    auto lb = constIndexValue(op.getLowerBound()[0]);
    auto ub = constIndexValue(op.getUpperBound()[0]);
    auto step = constIndexValue(op.getStep()[0]);
    if (!lb || !ub || !step || *lb != 0 || *step != 1) {
      return rewriter.notifyMatchFailure(
          op, "SPMD grids sweep [0, N) with step 1");
    }
    Block &body = op.getRegion().front();
    // Even a reduction-free scf.parallel ends in an (empty) scf.reduce.
    auto reduce = dyn_cast<scf::ReduceOp>(body.getTerminator());
    if (!reduce || reduce.getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op,
                                         "reductions have no PyPTO lowering");
    }
    auto spmd = rewriter.create<emitpy::SpmdScopeOp>(
        op.getLoc(), rewriter.getI64IntegerAttr(*ub));
    rewriter.inlineRegionBefore(op.getRegion(), spmd.getBody(),
                                spmd.getBody().end());
    rewriter.eraseOp(reduce);
    rewriter.replaceOp(op, spmd);
    return success();
  }
};

/// Rewrites `scf.forall` into the same `emitpy.spmd_scope` multi-core
/// launch as scf.parallel: a single induction variable sweeping `[0, N)`
/// with step 1 becomes the executing core's block index. forall stores its
/// bounds as mixed static/dynamic pairs, so both spellings of a constant
/// grid are accepted. The trailing in_parallel region only wraps tensor
/// merge writes (parallel_insert_slice), which have no PyPTO counterpart,
/// so it must be empty.
class ScfForallPattern : public OpConversionPattern<scf::ForallOp> {
 public:
  using OpConversionPattern<scf::ForallOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForallOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getRank() != 1 || !op.getOutputs().empty()) {
      return rewriter.notifyMatchFailure(op,
                                         "only single-induction forall loops without shared outputs map "
                                         "onto an SPMD grid");
    }
    auto lb = mixedIndexValue(op.getMixedLowerBound()[0]);
    auto ub = mixedIndexValue(op.getMixedUpperBound()[0]);
    auto step = mixedIndexValue(op.getMixedStep()[0]);
    if (!lb || !ub || !step || *lb != 0 || *step != 1 || *ub <= 0) {
      return rewriter.notifyMatchFailure(op, "SPMD grids are non-empty sweeps [0, N) with step 1");
    }
    Block &body = op.getRegion().front();
    auto inParallel = dyn_cast<scf::InParallelOp>(body.getTerminator());
    // The in_parallel region hosts the parallel merge writes; it must hold
    // nothing but its (possibly absent) empty block.
    if (!inParallel || (!inParallel.getRegion().empty() && !inParallel.getRegion().front().empty())) {
      return rewriter.notifyMatchFailure(op, "parallel merge writes have no PyPTO lowering");
    }
    auto spmd = rewriter.create<emitpy::SpmdScopeOp>(op.getLoc(), rewriter.getI64IntegerAttr(*ub));
    rewriter.inlineRegionBefore(op.getRegion(), spmd.getBody(), spmd.getBody().end());
    rewriter.eraseOp(inParallel);
    rewriter.replaceOp(op, spmd);
    return success();
  }
};

/// Rewrites `scf.for` into `emitpy.for` (emitted as
/// `for <iv> in pl.range(lb, ub, step)`): the per-core tile loop.
/// Loop-carried values are rejected: the generated kernels are straight-line
/// tile code with all results flowing through tile stores.
class ScfForPattern : public OpConversionPattern<scf::ForOp> {
 public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getInitArgs().empty()) {
      return rewriter.notifyMatchFailure(
          op, "loop-carried values have no PyPTO lowering");
    }
    Block &body = op.getRegion().front();
    auto yield = dyn_cast<scf::YieldOp>(body.getTerminator());
    if (!yield || yield.getNumOperands() != 0) {
      return rewriter.notifyMatchFailure(op, "unexpected loop terminator");
    }
    auto loop = rewriter.create<emitpy::ForOp>(
        op.getLoc(), op.getLowerBound(), op.getUpperBound(), op.getStep());
    rewriter.inlineRegionBefore(op.getRegion(), loop.getBody(),
                                loop.getBody().end());
    rewriter.eraseOp(yield);
    rewriter.replaceOp(op, loop);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function I/O rewrite (pre-conversion)
//===----------------------------------------------------------------------===//

/// Direction markers recorded by this pass and consumed by the Python
/// emitter when printing parameter annotations.
static constexpr StringLiteral kOutArgsAttr = "emitpy.out_args";
static constexpr StringLiteral kInOutArgsAttr = "emitpy.inout_args";

/// Marks a function as a PyPTO kernel; the Python emitter prints such
/// functions with an @pl.jit decorator, parameter annotations and a pl.at
/// scope around the body.
static constexpr StringLiteral kPyptoKernelAttr = "emitpy.pypto_kernel";

/// Returns the buffer a store into `value` ultimately writes when `value`
/// is a view chain (collapse_shape, subview, memory_space_cast, ...)
/// rooted in that buffer; returns `value` itself otherwise. The
/// bufferization routes stores through reshaped views of parameters, so
/// the write analysis has to look through them.
static Value stripViewChain(Value value) {
  while (Operation *defOp = value.getDefiningOp()) {
    if (auto view = dyn_cast<ViewLikeOpInterface>(defOp)) {
      value = view.getViewSource();
    } else if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(defOp)) {
      value = castOp.getSource();
    } else {
      break;
    }
  }
  return value;
}

static void markWriteDest(Value dest, DenseSet<unsigned> &writeDests) {
  dest = stripViewChain(dest);
  if (auto arg = dyn_cast<BlockArgument>(dest)) {
    writeDests.insert(arg.getArgNumber());
  }
}

/// Returns true when `use` stores into its operand, i.e. it is the
/// destination of a transfer_write.
static bool isStoreInto(OpOperand &use) {
  Operation *owner = use.getOwner();
  if (auto write = dyn_cast<vector::TransferWriteOp>(owner)) {
    return use.get() == write.getSource();
  }
  if (auto write = dyn_cast<npuvector::TransferWriteOp>(owner)) {
    return use.get() == write.getSource();
  }
  return false;
}

static bool viewChainOnlyWritten(Value buffer);

/// Returns true when `use` does not read its operand: it either stores
/// into it, or turns it into the buffer of another view chain that is
/// itself only written.
static bool useIsStoreOrWrittenView(OpOperand &use) {
  if (isStoreInto(use)) {
    return true;
  }
  Operation *owner = use.getOwner();
  if (auto view = dyn_cast<ViewLikeOpInterface>(owner)) {
    return use.get() == view.getViewSource() &&
           viewChainOnlyWritten(view.getOperation()->getResult(0));
  }
  if (auto castOp = dyn_cast<memref::MemorySpaceCastOp>(owner)) {
    return use.get() == castOp.getSource() &&
           viewChainOnlyWritten(castOp.getOperation()->getResult(0));
  }
  return false;
}

/// Returns true when every use of `buffer` writes into it, directly or
/// through further view chains: such a buffer is never read.
static bool viewChainOnlyWritten(Value buffer) {
  return llvm::all_of(buffer.getUses(),
                      [&](OpOperand &use) {
                        return useIsStoreOrWrittenView(use);
                      });
}

/// Returns true when `arg` is used by anything other than a store into it,
/// i.e. it is also read.
static bool argHasNonWriteUse(BlockArgument arg) {
  for (OpOperand &use : arg.getUses()) {
    if (!useIsStoreOrWrittenView(use)) {
      return true;
    }
  }
  return false;
}

/// Classifies written block arguments (step 1): arguments written by
/// transfer_write are outputs, and arguments that are additionally read
/// elsewhere become InOut instead.
static void collectWriteDestArgs(func::FuncOp funcOp, Block &entry,
                                 DenseSet<unsigned> &outArgs,
                                 DenseSet<unsigned> &inOutArgs) {
  DenseSet<unsigned> writeDests;
  funcOp.walk([&](Operation *op) {
    if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
      markWriteDest(write.getSource(), writeDests);
    } else if (auto write = dyn_cast<npuvector::TransferWriteOp>(op)) {
      markWriteDest(write.getSource(), writeDests);
    }
  });
  for (unsigned argNumber : writeDests) {
    BlockArgument arg = entry.getArgument(argNumber);
    if (argHasNonWriteUse(arg)) {
      inOutArgs.insert(argNumber);
    } else {
      outArgs.insert(argNumber);
    }
  }
}

/// Appends output parameters for allocations that only feed stores/returns
/// (step 2): read-back allocations become InOut instead.
static void promoteAllocsToOutputs(func::FuncOp funcOp, Block &entry,
                                   DenseSet<unsigned> &outArgs,
                                   DenseSet<unsigned> &inOutArgs) {
  SmallVector<Operation *> allocs;
  funcOp.walk([&](Operation *op) {
    if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      allocs.push_back(op);
    }
  });
  for (Operation *allocOp : allocs) {
    Value result = allocOp->getResult(0);
    bool ioOnly = true;
    bool hasRead = false;
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      if (isStoreInto(use)) {
        continue;
      }
      if (isa<func::ReturnOp, memref::DeallocOp>(owner)) {
        continue;
      }
      if (isa<vector::TransferReadOp, npuvector::TransferReadOp>(owner)) {
        hasRead = true;
        continue;
      }
      ioOnly = false;
      break;
    }
    if (!ioOnly) {
      continue;  // leaves the op for the conversion to reject
    }

    BlockArgument newArg = entry.addArgument(
        cast<MemRefType>(result.getType()), allocOp->getLoc());
    if (hasRead) {
      inOutArgs.insert(newArg.getArgNumber());
    } else {
      outArgs.insert(newArg.getArgNumber());
    }
    result.replaceAllUsesWith(newArg);
    allocOp->erase();
  }
}

/// Appends an output tensor for a computed tile return value and stores
/// the tile into it.
static LogicalResult storeTileReturn(OpBuilder &builder, Block &entry,
                                     DenseSet<unsigned> &outArgs,
                                     func::ReturnOp ret, Value operand,
                                     ShapedType shaped) {
  auto memrefType =
      MemRefType::get(shaped.getShape(), shaped.getElementType());
  BlockArgument newArg = entry.addArgument(memrefType, ret.getLoc());
  SmallVector<Value> zeros;
  for (int64_t i = 0; i < shaped.getRank(); ++i) {
    zeros.push_back(builder.create<arith::ConstantIndexOp>(ret.getLoc(), 0));
  }
  if (isa<npuvector::NPUVectorType>(operand.getType())) {
    builder.create<npuvector::TransferWriteOp>(
        ret.getLoc(), TypeRange{}, operand, newArg, zeros,
        /*mask=*/Value());
  } else {
    builder.create<vector::TransferWriteOp>(ret.getLoc(), operand, newArg,
                                            zeros);
  }
  outArgs.insert(newArg.getArgNumber());
  return success();
}

/// Appends an output tensor for a computed scalar return value: the scalar
/// is broadcast to vector<1> and stored.
static void storeScalarReturn(OpBuilder &builder, Block &entry,
                              DenseSet<unsigned> &outArgs,
                              func::ReturnOp ret, Value operand) {
  Type type = operand.getType();
  auto vecType = VectorType::get({1}, type);
  auto memrefType = MemRefType::get({1}, type);
  BlockArgument newArg = entry.addArgument(memrefType, ret.getLoc());
  Value broadcast =
      builder.create<vector::BroadcastOp>(ret.getLoc(), vecType, operand);
  Value zero = builder.create<arith::ConstantIndexOp>(ret.getLoc(), 0);
  SmallVector<Value> storeIndices{zero};
  builder.create<vector::TransferWriteOp>(ret.getLoc(), broadcast, newArg,
                                          storeIndices);
  outArgs.insert(newArg.getArgNumber());
}

/// Rewrites every `return` (step 3): computed values are stored into
/// appended output parameters and the return itself is emptied. PyPTO
/// kernels return their outputs by name; the Python emitter reconstructs
/// that from the recorded directions, so the IR keeps a plain `return`.
static LogicalResult rewriteReturns(OpBuilder &builder, Block &entry,
                                    ArrayRef<func::ReturnOp> returns,
                                    DenseSet<unsigned> &outArgs,
                                    DenseSet<unsigned> &inOutArgs) {
  for (func::ReturnOp ret : returns) {
    builder.setInsertionPoint(ret);
    for (Value operand : ret.getOperands()) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        // Returning an argument that is not an output still needs a
        // direction that permits both read and write.
        if (!outArgs.count(arg.getArgNumber())) {
          inOutArgs.insert(arg.getArgNumber());
        }
        continue;
      }
      Type type = operand.getType();
      auto shaped = dyn_cast<ShapedType>(type);
      if (shaped && shaped.hasStaticShape() &&
          (isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type))) {
        if (failed(storeTileReturn(builder, entry, outArgs, ret, operand,
                                   shaped))) {
          return failure();
        }
        continue;
      }
      if (isa<FloatType>(type) || isa<IntegerType>(type) ||
          isa<IndexType>(type)) {
        storeScalarReturn(builder, entry, outArgs, ret, operand);
        continue;
      }
      return ret.emitOpError("unsupported return value type for PyPTO");
    }
    ret.getOperandsMutable().clear();
  }
  return success();
}

/// Records the Out/InOut parameter directions for the emitter (step 5).
/// DenseI64ArrayAttr keeps its identity across the textual round-trip
/// (`array<i64: ...>`), unlike the generic ArrayAttr the emitter would not
/// find again.
static void recordArgDirections(func::FuncOp funcOp,
                                const DenseSet<unsigned> &outArgs,
                                const DenseSet<unsigned> &inOutArgs) {
  MLIRContext *ctx = funcOp.getContext();
  if (!outArgs.empty()) {
    SmallVector<int64_t> outList(outArgs.begin(), outArgs.end());
    llvm::sort(outList);
    funcOp->setAttr(kOutArgsAttr, DenseI64ArrayAttr::get(ctx, outList));
  }
  if (!inOutArgs.empty()) {
    SmallVector<int64_t> inOutList(inOutArgs.begin(), inOutArgs.end());
    llvm::sort(inOutList);
    funcOp->setAttr(kInOutArgsAttr, DenseI64ArrayAttr::get(ctx, inOutList));
  }
}

/// Rewrites the function so that every output flows through a `pl.Out`
/// parameter, mirroring the PyPTO kernel model:
/// - arguments written by `transfer_write` become outputs;
/// - allocations that only feed stores/returns become appended output
///   arguments (read-back allocations become `pl.InOut` instead);
/// - computed return values are stored into appended output arguments
///   (scalars through a broadcast to vector<1>);
/// - every `return` carries the output arguments, since a bare `return` is
///   not a legal PyPTO kernel tail.
static LogicalResult rewriteFunctionOutputs(func::FuncOp funcOp) {
  MLIRContext *ctx = funcOp.getContext();
  OpBuilder builder(ctx);
  Block &entry = funcOp.getBody().front();

  DenseSet<unsigned> outArgs, inOutArgs;

  // 1. Arguments written by transfer_write are outputs; arguments that are
  // additionally read elsewhere become InOut instead.
  collectWriteDestArgs(funcOp, entry, outArgs, inOutArgs);

  // 2. Allocations that only feed stores/returns become output arguments.
  promoteAllocsToOutputs(funcOp, entry, outArgs, inOutArgs);

  // 3. Rewrite returns: computed values are stored into appended output
  // parameters and the return itself is emptied.
  SmallVector<func::ReturnOp> returns;
  funcOp.walk([&](func::ReturnOp op) { returns.push_back(op); });
  if (failed(rewriteReturns(builder, entry, returns, outArgs, inOutArgs))) {
    return failure();
  }

  // 4. The kernel hands results back through its Out parameters only. The
  // input list is rebuilt from the block arguments: appended output
  // parameters are only visible there, not in the old function type.
  if (!returns.empty()) {
    SmallVector<Type> inputTypes;
    llvm::transform(entry.getArguments(), std::back_inserter(inputTypes),
                    [](BlockArgument arg) { return arg.getType(); });
    funcOp.setType(FunctionType::get(ctx, inputTypes, {}));
  }

  // 5. Record the directions for the emitter.
  recordArgDirections(funcOp, outArgs, inOutArgs);
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Makes sure the module enclosing `funcOp` imports pypto.language as `pl`
/// exactly once, so the generated callees (pl.*, ...) resolve.
static void insertPyptoImport(func::FuncOp funcOp) {
  ModuleOp module = funcOp->getParentOfType<ModuleOp>();
  if (!module) {
    return;
  }
  // Function passes run in parallel on sibling functions; the module level
  // import bookkeeping (lookup plus insertion) must be serialized.
  static std::mutex importMutex;
  std::lock_guard<std::mutex> lock(importMutex);
  for (Operation &op : module.getBody()->getOperations()) {
    auto importOp = dyn_cast<emitpy::ImportOp>(op);
    if (!importOp) {
      continue;
    }
    auto alias = importOp.getModuleAlias();
    if (importOp.getModuleName() == "pypto.language" && alias &&
        *alias == "pl") {
      return;
    }
  }
  OpBuilder builder(module);
  builder.setInsertionPointToStart(module.getBody());
  builder.create<emitpy::ImportOp>(module.getLoc(), "pypto.language",
                                   builder.getStringAttr("pl"),
                                   /*members_to_import=*/nullptr,
                                   /*member_aliases=*/nullptr,
                                   /*import_all=*/nullptr);
}

/// Returns true for statically shaped types holding at most one element
/// (0-D or size-1 tiles).
static bool isScalarTileType(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  return shaped && shaped.hasStaticShape() &&
         (shaped.getRank() == 0 || shaped.getNumElements() == 1);
}

/// Undoes the size-1 shape folds (memref.collapse_shape) that the shape
/// normalization inserts for scalar tensors: a 0-D buffer holds the same
/// single element as the [1] buffer it came from, so every use is reroute
/// to the original buffer and the 0-D reads/writes keep addressing it at
/// [0]. Doing this before the signature rewrite keeps the parameters and
/// the output analysis on the original buffers.
static void undoSizeOneShapeFolds(func::FuncOp funcOp) {
  SmallVector<memref::CollapseShapeOp> folds;
  funcOp.walk([&](memref::CollapseShapeOp fold) {
    auto srcType = dyn_cast<ShapedType>(fold.getSrc().getType());
    auto dstType = dyn_cast<ShapedType>(fold.getResult().getType());
    if (srcType && dstType && srcType.getNumElements() == 1 &&
        dstType.getNumElements() == 1) {
      folds.push_back(fold);
    }
  });
  for (memref::CollapseShapeOp fold : folds) {
    fold.getResult().replaceAllUsesWith(fold.getSrc());
    fold.erase();
  }
}

/// Lowers the 0-D (scalar) chains onto plain scalars before the dialect
/// conversion: the shape normalization folds size-1 tensors to 0-D tiles,
/// which have no device form. A 0-D read becomes pypto.read of the single
/// element, a 0-D broadcast is its source itself, any other op's results
/// adopt the scalar element type, and 0-D stores are left for the
/// transfer-write pattern (which writes scalars through pypto.write). Doing
/// this ahead of the conversion keeps the value types stable while the
/// patterns rewrite ops one by one.
static void lowerZeroDimChains(func::FuncOp funcOp) {
  SmallVector<Operation *> scalarOps;
  funcOp.walk([&](Operation *op) {
    if (llvm::any_of(op->getResultTypes(), isScalarTileType)) {
      scalarOps.push_back(op);
    }
  });
  OpBuilder builder(funcOp);
  for (Operation *op : scalarOps) {
    if (isa<npuvector::TransferReadOp, vector::TransferReadOp>(op)) {
      // %v = pypto.read src[0]: read the single element of the unfolded
      // [1] buffer as a scalar.
      Value source = op->getOperand(0);
      auto shaped = cast<ShapedType>(op->getResult(0).getType());
      builder.setInsertionPoint(op);
      Value read = builder
                       .create<pypto::ReadOp>(op->getLoc(),
                                              shaped.getElementType(), source,
                                              ArrayRef<int64_t>{0})
                       .getResult();
      op->getResult(0).replaceAllUsesWith(read);
      op->erase();
      continue;
    }
    if (isa<npuvector::BroadcastOp, vector::BroadcastOp>(op)) {
      // A 0-D broadcast of a scalar is the scalar itself.
      op->getResult(0).replaceAllUsesWith(op->getOperand(0));
      op->erase();
      continue;
    }
    for (OpResult res : op->getResults()) {
      if (auto shaped = dyn_cast<ShapedType>(res.getType());
          shaped && shaped.getRank() == 0) {
        res.setType(shaped.getElementType());
      }
    }
  }
}

/// Drops dead conversion by-products: the scalar-write fast path leaves
/// the materialized broadcast chain behind, and source padding constants
/// (e.g. transfer_read's fill value) never gain a use. Repeat to fixed
/// point so producer chains collapse. Result-less ops (stores, writes,
/// loops, returns) are kept unconditionally. The sweep covers the whole
/// function: loops carry dead by-products in their bodies too.
static void eraseDeadOps(func::FuncOp funcOp) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *> dead;
    funcOp.walk([&](Operation *op) {
      if (op->getNumResults() != 0 && op->use_empty()) {
        dead.push_back(op);
      }
    });
    for (Operation *op : dead) {
      op->erase();
      changed = true;
    }
  }
}

/// Loads one tile-valued parameter into a tile (`pl.load`) so every body
/// op keeps operating on tiles, and redirects every use except the load's
/// own operand.
static void loadTileParameter(OpBuilder &builder, BlockArgument arg,
                              ShapedType shaped) {
  SmallVector<Attribute> loadArgs{
      builder.getIndexAttr(0),
      builder.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(SmallVector<int64_t>(shaped.getRank(), 0))),
      builder.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(shaped.getShape()))};
  SmallVector<Attribute> loadKeywords{builder.getStringAttr(""),
                                      builder.getStringAttr(""),
                                      builder.getStringAttr("")};
  SmallVector<Value> loadOperands{arg};
  Value load = builder
                   .create<emitpy::CallOpaqueOp>(
                       arg.getLoc(), shaped, "pl.load",
                       ValueRange(loadOperands),
                       builder.getArrayAttr(loadArgs),
                       builder.getArrayAttr(loadKeywords))
                   .getResult(0);
  for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
    if (use.getOwner() == load.getDefiningOp()) {
      continue;
    }
    use.set(load);
  }
}

/// Rewrites the signature by hand: parameters become opaque PyPTO
/// annotations while the body keeps referencing the very same block
/// arguments, which is fine because emitpy ops take operands of any type
/// and the Python emitter prints names, not types.
static LogicalResult rewriteSignature(func::FuncOp funcOp,
                                      MLIRContext *ctx) {
  Block &entry = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
  // Tile stores need a 2D tensor of matching rank, so write-only rank-1
  // outputs whose row would be a whole number of 32-byte units are
  // presented as [1, N] tensors (the store pattern reshapes the tile
  // through the same view).
  DenseSet<unsigned> outOnlyArgs;
  if (auto outList = funcOp->getAttrOfType<DenseI64ArrayAttr>(kOutArgsAttr)) {
    for (int64_t argNo : outList.asArrayRef()) {
      outOnlyArgs.insert(static_cast<unsigned>(argNo));
    }
  }
  SmallVector<Type> newInputs;
  OpBuilder builder(ctx);
  builder.setInsertionPointToStart(&entry);
  for (auto [idx, type] : llvm::enumerate(funcType.getInputs())) {
    Type annotation;
    if (outOnlyArgs.contains(static_cast<unsigned>(idx))) {
      if (auto shaped = dyn_cast<ShapedType>(type);
          shaped && shaped.getRank() == 1 && shaped.hasStaticShape() &&
          shaped.getNumElements() *
                  (shaped.getElementType().getIntOrFloatBitWidth() / 8) %
              32 ==
          0) {
        annotation = pyptoShapedOpaque(
            ctx, SmallVector<int64_t>{1, shaped.getDimSize(0)},
            shaped.getElementType(), "pl.Tensor");
      }
    }
    if (!annotation) {
      annotation = pyptoAnnotation(ctx, type);
    }
    if (!annotation) {
      return funcOp->emitError("parameter ")
          << idx << " has no PyPTO annotation";
    }
    newInputs.push_back(annotation);
    BlockArgument arg = entry.getArgument(idx);
    arg.setType(annotation);
    // Tile-valued parameters cross the kernel boundary as tensors; load
    // them into tiles once so every body op keeps operating on tiles.
    if ((isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type)) &&
        !arg.use_empty()) {
      loadTileParameter(builder, arg, cast<ShapedType>(type));
    }
  }
  funcOp.setType(FunctionType::get(ctx, newInputs, {}));
  return success();
}

/// Strips the HACC annotations: they belong to the HIVM/bishengir
/// lowering this conversion replaces and would otherwise pin unregistered
/// dialects into the emitted IR.
static void stripHaccAttrs(func::FuncOp funcOp) {
  auto isHaccAttr = [](NamedAttribute attr) {
    return attr.getName().strref().starts_with("hacc.");
  };
  SmallVector<StringAttr> toRemove;
  for (NamedAttribute attr : funcOp->getAttrs()) {
    if (isHaccAttr(attr)) {
      toRemove.push_back(attr.getName());
    }
  }
  for (StringAttr name : toRemove) {
    funcOp->removeAttr(name);
  }
  for (unsigned i = 0; i < funcOp.getNumArguments(); ++i) {
    toRemove.clear();
    if (auto argAttrs = funcOp.getArgAttrDict(i)) {
      for (NamedAttribute attr : argAttrs.getValue()) {
        if (isHaccAttr(attr)) {
          toRemove.push_back(attr.getName());
        }
      }
    }
    for (StringAttr name : toRemove) {
      funcOp.removeArgAttr(i, name);
    }
  }
}

//===----------------------------------------------------------------------===//
// PyPTO -> EmitPy sweep
//===----------------------------------------------------------------------===//
// The patterns above rewrite tile computations into pypto ops that keep
// the source result types; this sweep lowers those intermediates back
// into the `pl.*` call_opaque forms the direct lowering produced, so the
// emitted IR (and Python) is unchanged.

/// Rewrites a positional pypto op into `emitpy.call_opaque`, e.g.
/// `pypto.add %a, %b` -> `%r = pl.add(a, b)`. The raw (non-adaptor)
/// operands and result types are used on purpose: values already
/// rewritten point at their replacements, and the source result types
/// keep the identity converter from materializing casts.
template <typename PyptoOp>
class PyptoToCallOpaquePattern : public OpConversionPattern<PyptoOp> {
 public:
  PyptoToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context,
                           StringRef callee)
      : OpConversionPattern<PyptoOp>(typeConverter, context), callee(callee) {}

  LogicalResult
  matchAndRewrite(PyptoOp op, typename PyptoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResultTypes(), callee, op->getOperands());
    return success();
  }

 private:
  StringRef callee;
};

/// Rewrites `pypto.constant` into `pl.const(value, dtype)`.
class PyptoConstantPattern : public OpConversionPattern<pypto::ConstantOp> {
 public:
  using OpConversionPattern<pypto::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = pyptoDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "constant type has no PyPTO dtype");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue()),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.const", ValueRange{},
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.tile_full` into `pl.tile.full(shape, dtype, value)`:
/// statically shaped results carry the shape in the result type; dynamic
/// results carry one extent operand per dimension (constants fold into
/// the list literal, runtime bounds become operand references, exactly
/// like offsets).
class PyptoTileFullPattern : public OpConversionPattern<pypto::TileFullOp> {
 public:
  using OpConversionPattern<pypto::TileFullOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::TileFullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shaped = dyn_cast<ShapedType>(op.getResult().getType());
    if (!shaped) {
      return rewriter.notifyMatchFailure(op,
                                         "tile_full result must be shaped");
    }
    auto dtype = pyptoDtypeName(shaped.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    SmallVector<Value> fullOperands;
    Attribute shapeArg;
    if (op.getExtents().empty()) {
      if (!shaped.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            op, "tile_full without extents must be statically shaped");
      }
      shapeArg = rewriter.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(shaped.getShape()));
    } else {
      shapeArg =
          buildOffsetsArg(rewriter, op.getLoc(), op.getExtents(), fullOperands);
    }
    SmallVector<Attribute> args{
        shapeArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue())};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.tile.full",
        ValueRange(fullOperands), rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.load` into `pl.load`: statically shaped reads take the
/// shape from the result type; dynamic reads pass the padded allocation
/// shapes and the valid extents, matching the direct lowering (offsets
/// fold into a list literal when constant, runtime values become operand
/// references appended after the source).
class PyptoLoadPattern : public OpConversionPattern<pypto::LoadOp> {
 public:
  using OpConversionPattern<pypto::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shaped = dyn_cast<ShapedType>(op.getResult().getType());
    if (!shaped) {
      return rewriter.notifyMatchFailure(op, "load result must be shaped");
    }
    SmallVector<Value> loadOperands{op.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, op.getLoc(), op.getOffsets(), loadOperands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg};
    if (op.getPaddedShapes().empty()) {
      if (!shaped.hasStaticShape()) {
        return rewriter.notifyMatchFailure(
            op, "load without padded shapes must be statically shaped");
      }
      args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(shaped.getShape())));
    } else {
      args.push_back(buildOffsetsArg(rewriter, op.getLoc(),
                                     op.getPaddedShapes(), loadOperands));
      args.push_back(buildOffsetsArg(rewriter, op.getLoc(),
                                     op.getValidExtents(), loadOperands));
    }
    SmallVector<Attribute> keywords(args.size(), rewriter.getStringAttr(""));
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.load", ValueRange(loadOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.store` into `pl.store(value, [offsets], dest)`.
class PyptoStorePattern : public OpConversionPattern<pypto::StoreOp> {
 public:
  using OpConversionPattern<pypto::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> storeOperands{op.getValue(), op.getDest()};
    Attribute indexArg = buildOffsetsArg(rewriter, op.getLoc(),
                                         op.getOffsets(), storeOperands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), indexArg,
                                rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, TypeRange{}, "pl.store", ValueRange(storeOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reduction` into `pl.row_<kind>(vector, scratch)` /
/// `pl.col_<kind>(vector, scratch)`: a keepdim in-place fold of the last
/// (row) or first (col) axis, working in the scratch tile.
class PyptoReductionPattern : public OpConversionPattern<pypto::ReductionOp> {
 public:
  using OpConversionPattern<pypto::ReductionOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef prefix =
        op.getAxis() == pypto::ReduceAxis::row ? "row_" : "col_";
    StringRef suffix;
    switch (op.getKind()) {
    case pypto::ReduceKind::sum:
      suffix = "sum";
      break;
    case pypto::ReduceKind::prod:
      suffix = "prod";
      break;
    case pypto::ReduceKind::min:
      suffix = "min";
      break;
    case pypto::ReduceKind::max:
      suffix = "max";
      break;
    }
    std::string callee = "pl.";
    callee += prefix;
    callee += suffix;
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), callee,
        ValueRange{op.getVector(), op.getScratch()});
    return success();
  }
};

/// Rewrites `pypto.read` into `pl.read(src, [indices])`: the constant
/// coordinates render as a Python index list.
class PyptoReadPattern : public OpConversionPattern<pypto::ReadOp> {
 public:
  using OpConversionPattern<pypto::ReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getIndices()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.read", op.getSource(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.write` into `pl.write(dest, [offsets], value)`: the
/// first-stage pattern materialized constant offsets as index literals
/// (now `emitpy.literal`); constants fold back into the list literal,
/// runtime values become operand references appended after the operands.
class PyptoWritePattern : public OpConversionPattern<pypto::WriteOp> {
 public:
  using OpConversionPattern<pypto::WriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::WriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> writeOperands{op.getDest(), op.getValue()};
    Attribute indexArg = buildOffsetsArg(rewriter, op.getLoc(),
                                         op.getOffsets(), writeOperands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), indexArg,
                                rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "pl.write", ValueRange(writeOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    rewriter.eraseOp(op);
    return success();
  }
};

/// Rewrites `pypto.slice` into `pl.slice(src, [window], [offsets],
/// drop_dims=[...])`: the first-stage pattern materialized static offsets
/// as index constants (now `emitpy.literal`); constants fold back into
/// the list literal, runtime values become operand references appended
/// after the source.
class PyptoSlicePattern : public OpConversionPattern<pypto::SliceOp> {
 public:
  using OpConversionPattern<pypto::SliceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{op.getSource()};
    SmallVector<OpFoldResult> offsets;
    for (Value idx : op.getOffsets()) {
      if (auto cst = constIndexValue(idx)) {
        offsets.push_back(rewriter.getIndexAttr(*cst));
      } else {
        offsets.push_back(OpFoldResult(idx));
      }
    }
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getWindow())),
        buildMixedListArg(rewriter, offsets, operands)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    if (auto dropDims = op.getDropDims()) {
      args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(
          dimsToPythonText(llvm::to_vector_of<unsigned>(*dropDims))));
      keywords.push_back(rewriter.getStringAttr("drop_dims"));
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.slice", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reshape` into `pl.reshape(src, [shape])`: the shape
/// travels in the result type.
class PyptoReshapePattern : public OpConversionPattern<pypto::ReshapeOp> {
 public:
  using OpConversionPattern<pypto::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto shaped = dyn_cast<ShapedType>(op.getResult().getType());
    if (!shaped || !shaped.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "reshape result must be statically shaped");
    }
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(shaped.getShape()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.reshape", op.getSource(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.transpose` into `pl.transpose(src, i, j)`: the swapped
/// axes render as inline integers.
class PyptoTransposePattern : public OpConversionPattern<pypto::TransposeOp> {
 public:
  using OpConversionPattern<pypto::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getI())),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getJ()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.transpose", op.getSource(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites a dtype-annotated conversion (`pypto.cast` -> `pl.cast`,
/// `pypto.reinterpret_view` -> `pl.reinterpret_view`) into
/// `<callee>(v, pl.<dtype>)`: the dtype travels in the result type.
template <typename PyptoOp>
class PyptoDtypeCastPattern : public OpConversionPattern<PyptoOp> {
 public:
  PyptoDtypeCastPattern(TypeConverter &typeConverter, MLIRContext *context,
                        StringRef callee)
      : OpConversionPattern<PyptoOp>(typeConverter, context), callee(callee) {}

  LogicalResult
  matchAndRewrite(PyptoOp op, typename PyptoOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype =
        pyptoDtypeName(getElementTypeOrSelf(op->getResult(0).getType()));
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "result type has no PyPTO dtype");
    }
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResult(0).getType(), callee, op->getOperands(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }

 private:
  StringRef callee;
};

/// Rewrites `pypto.cmp` into `pl.cmp(a, b, cmp_type=N)`: the enum values
/// coincide with the `pl.cmp` type codes (0=eq .. 5=ge).
class PyptoCmpPattern : public OpConversionPattern<pypto::CmpOp> {
 public:
  using OpConversionPattern<pypto::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(1),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            std::to_string(static_cast<unsigned>(op.getPredicate())))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("cmp_type")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.cmp", op->getOperands(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Collects the sweep patterns: one per pypto op the first-stage patterns
/// can produce.
static void populatePyptoSweepPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Data movement and constant producers.
  patterns.add<PyptoLoadPattern, PyptoStorePattern, PyptoConstantPattern,
               PyptoTileFullPattern>(typeConverter, ctx);

  // Reductions and scalar IO (third-batch switch).
  patterns.add<PyptoReductionPattern, PyptoReadPattern, PyptoWritePattern>(
      typeConverter, ctx);

  // Elementwise binaries. Bitwise ops carry Python's trailing underscore
  // where the runtime name is a keyword (`pl.and_`, `pl.or_`).
  patterns.add<PyptoToCallOpaquePattern<pypto::AddOp>>(typeConverter, ctx,
                                                       "pl.add");
  patterns.add<PyptoToCallOpaquePattern<pypto::SubOp>>(typeConverter, ctx,
                                                       "pl.sub");
  patterns.add<PyptoToCallOpaquePattern<pypto::MulOp>>(typeConverter, ctx,
                                                       "pl.mul");
  patterns.add<PyptoToCallOpaquePattern<pypto::DivOp>>(typeConverter, ctx,
                                                       "pl.div");
  patterns.add<PyptoToCallOpaquePattern<pypto::FmodOp>>(typeConverter, ctx,
                                                        "pl.fmod");
  patterns.add<PyptoToCallOpaquePattern<pypto::MaximumOp>>(typeConverter, ctx,
                                                           "pl.maximum");
  patterns.add<PyptoToCallOpaquePattern<pypto::MinimumOp>>(typeConverter, ctx,
                                                           "pl.minimum");
  patterns.add<PyptoToCallOpaquePattern<pypto::AndOp>>(typeConverter, ctx,
                                                       "pl.and_");
  patterns.add<PyptoToCallOpaquePattern<pypto::OrOp>>(typeConverter, ctx,
                                                      "pl.or_");
  patterns.add<PyptoToCallOpaquePattern<pypto::XorOp>>(typeConverter, ctx,
                                                       "pl.xor");
  patterns.add<PyptoToCallOpaquePattern<pypto::ShlOp>>(typeConverter, ctx,
                                                       "pl.shl");
  patterns.add<PyptoToCallOpaquePattern<pypto::ShrOp>>(typeConverter, ctx,
                                                       "pl.shr");

  // Elementwise unaries.
  patterns.add<PyptoToCallOpaquePattern<pypto::NegOp>>(typeConverter, ctx,
                                                       "pl.neg");
  patterns.add<PyptoToCallOpaquePattern<pypto::AbsOp>>(typeConverter, ctx,
                                                       "pl.abs");
  patterns.add<PyptoToCallOpaquePattern<pypto::ExpOp>>(typeConverter, ctx,
                                                       "pl.exp");
  patterns.add<PyptoToCallOpaquePattern<pypto::LogOp>>(typeConverter, ctx,
                                                       "pl.log");
  patterns.add<PyptoToCallOpaquePattern<pypto::SqrtOp>>(typeConverter, ctx,
                                                        "pl.sqrt");
  patterns.add<PyptoToCallOpaquePattern<pypto::RsqrtOp>>(typeConverter, ctx,
                                                         "pl.rsqrt");
  patterns.add<PyptoToCallOpaquePattern<pypto::SinOp>>(typeConverter, ctx,
                                                       "pl.sin");
  patterns.add<PyptoToCallOpaquePattern<pypto::CosOp>>(typeConverter, ctx,
                                                       "pl.cos");

  // Scalar min/max: PyPTO's scalar-only forms (tiles use maximum/minimum).
  patterns.add<PyptoToCallOpaquePattern<pypto::MaxOp>>(typeConverter, ctx,
                                                       "pl.max");
  patterns.add<PyptoToCallOpaquePattern<pypto::MinOp>>(typeConverter, ctx,
                                                       "pl.min");

  // Views, casts, comparisons and selects (second-batch switch).
  patterns.add<PyptoSlicePattern, PyptoReshapePattern, PyptoTransposePattern,
               PyptoCmpPattern>(typeConverter, ctx);
  patterns.add<PyptoDtypeCastPattern<pypto::CastOp>>(typeConverter, ctx,
                                                     "pl.cast");
  patterns.add<PyptoDtypeCastPattern<pypto::ReinterpretViewOp>>(
      typeConverter, ctx, "pl.reinterpret_view");
  patterns.add<PyptoToCallOpaquePattern<pypto::SelectOp>>(typeConverter, ctx,
                                                          "pl.sel");
}

struct ConvertVectorToEmitPyPass
    : impl::ConvertVectorToEmitPyBase<ConvertVectorToEmitPyPass> {
  using impl::ConvertVectorToEmitPyBase<
      ConvertVectorToEmitPyPass>::ConvertVectorToEmitPyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect,
                    pypto::PyptoDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();

    insertPyptoImport(funcOp);

    // Mark the function so the Python emitter renders it as a @pl.jit
    // kernel rather than a plain function.
    funcOp->setAttr(kPyptoKernelAttr, UnitAttr::get(ctx));

    undoSizeOneShapeFolds(funcOp);

    // Normalize the function I/O to the PyPTO Out-parameter model before
    // converting: the rewrite inserts npuvector/vector transfer ops that the
    // patterns below then lower.
    if (failed(rewriteFunctionOutputs(funcOp))) {
      signalPassFailure();
      return;
    }

    lowerZeroDimChains(funcOp);

    VectorToEmitPyTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<emitpy::EmitPyDialect>();
    // The first-stage patterns emit intermediate pypto ops (keeping the
    // source result types); they are legal during the main conversion and
    // the sweep below lowers them back into emitpy.
    target.addLegalDialect<pypto::PyptoDialect>();
    target.addIllegalDialect<npuvector::NPUVectorDialect, vector::VectorDialect,
                             arith::ArithDialect, math::MathDialect>();
    // Control flow maps onto SPMD launches and tile loops; any other scf
    // construct has no PyPTO counterpart. scf.forall lowers through the
    // same SPMD grid as scf.parallel.
    target.addIllegalDialect<scf::SCFDialect>();
    // Allocations are handled by rewriteFunctionOutputs; any survivor means
    // a use this backend cannot express. The view ops must become `pl.*`
    // tiles: the Python emitter has no memref form, so leaving them legal
    // lets them survive the conversion as raw memrefs.
    target.addIllegalOp<
        memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
        memref::LoadOp, memref::StoreOp, memref::CollapseShapeOp,
        memref::ExpandShapeOp, memref::TransposeOp, memref::ReshapeOp,
        memref::SubViewOp, memref::ReinterpretCastOp>();
    // The signature itself stays untouched by the conversion: retyping
    // parameters through the framework would materialize
    // unrealized_conversion_cast ops that a full conversion rejects. The
    // signature and block arguments are rewritten by hand below, once the
    // body is fully converted. Returns were emptied by the pre-pass rewrite
    // and calls are left as-is.
    target.addLegalOp<func::FuncOp, func::ReturnOp, func::CallOp>();

    RewritePatternSet patterns(ctx);
    vector_to_emitpy::populateVectorToEmitPyConversionPatterns(patterns,
                                                               typeConverter);

    if (failed(applyFullConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Debug escape hatch: stop after the first stage and keep the
    // intermediate pypto layer (source result types intact) in the output,
    // exactly as the sweep below receives it.
    if (emitPypto) {
      return;
    }

    // Lower the intermediate pypto ops back into the `pl.*` call_opaque
    // forms: the first-stage patterns produce pypto ops (keeping the
    // source result types), and this sweep restores the exact emitpy IR
    // the direct lowering emitted, so the emitted Python is unchanged.
    ConversionTarget sweepTarget(*ctx);
    sweepTarget.addLegalDialect<emitpy::EmitPyDialect>();
    sweepTarget.addIllegalDialect<pypto::PyptoDialect>();
    // The sweep only replaces pypto ops; the surrounding function shell
    // (already legalized by the main conversion above) must stay legal.
    sweepTarget.addLegalOp<func::FuncOp, func::ReturnOp, func::CallOp>();
    RewritePatternSet sweepPatterns(ctx);
    populatePyptoSweepPatterns(sweepPatterns, typeConverter);
    if (failed(applyFullConversion(funcOp, sweepTarget,
                                   std::move(sweepPatterns)))) {
      signalPassFailure();
      return;
    }

    eraseDeadOps(funcOp);

    if (failed(rewriteSignature(funcOp, ctx))) {
      signalPassFailure();
      return;
    }

    stripHaccAttrs(funcOp);
  }
};

}  // namespace

namespace vector_to_emitpy {

namespace {

/// Constants and elementwise float arithmetic.
static void populateFloatArithPatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Constants.
  patterns.add<ArithConstantPattern>(typeConverter, ctx);

  // Elementwise float arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddFOp, pypto::AddOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubFOp, pypto::SubOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulFOp, pypto::MulOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivFOp, pypto::DivOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::RemFOp, pypto::FmodOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::MaximumFOp, pypto::MaximumOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinimumFOp, pypto::MinimumOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::NegFOp, pypto::NegOp>>(typeConverter,
                                                                ctx);
}

/// Elementwise integer arithmetic.
static void populateIntArithPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Elementwise integer arithmetic.
  patterns.add<OpToPyptoOpPattern<arith::AddIOp, pypto::AddOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::SubIOp, pypto::SubOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::MulIOp, pypto::MulOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivSIOp, pypto::DivOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::DivUIOp, pypto::DivOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::RemSIOp, pypto::FmodOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::RemUIOp, pypto::FmodOp>>(
      typeConverter, ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MaxSIOp, pypto::MaxOp,
                                    pypto::MaximumOp>>(typeConverter, ctx);
  patterns.add<MinMaxToPyptoPattern<arith::MinSIOp, pypto::MinOp,
                                    pypto::MinimumOp>>(typeConverter, ctx);
  // Floor division maps onto Python's `//` operator: the translator prints
  // the `util_floordiv` pseudo-callee as an infix `(a // b)` expression.
  patterns.add<OpToCallOpaquePattern<arith::FloorDivSIOp>>(typeConverter, ctx,
                                                           "util_floordiv");
  patterns.add<OpToPyptoOpPattern<arith::MaxUIOp, pypto::MaximumOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::MinUIOp, pypto::MinimumOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::AndIOp, pypto::AndOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::OrIOp, pypto::OrOp>>(typeConverter,
                                                              ctx);
  patterns.add<OpToPyptoOpPattern<arith::XOrIOp, pypto::XorOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShLIOp, pypto::ShlOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRSIOp, pypto::ShrOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<arith::ShRUIOp, pypto::ShrOp>>(
      typeConverter, ctx);
}

/// Comparisons and select.
static void populateCmpSelectPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Comparisons and select.
  patterns.add<CmpToPyptoPattern<arith::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<arith::CmpFOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToPyptoPattern<npuvector::CmpFOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<arith::SelectOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<npuvector::SelectOp>>(typeConverter, ctx);
}

/// Numeric casts (pl.cast) and bitcasts (pl.reinterpret_view).
static void populateCastPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Numeric casts -> pl.cast; bitcasts -> pl.reinterpret_view.
  patterns.add<CastToPyptoPattern<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::ExtSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::ExtUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::TruncIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::UIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::FPToUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<arith::IndexCastUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncFOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::ExtUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::TruncIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::SIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::UIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToSIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::FPToUIOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastOp>>(typeConverter, ctx);
  patterns.add<CastToPyptoPattern<npuvector::IndexCastUIOp>>(typeConverter,
                                                            ctx);
  patterns.add<BitcastToPyptoPattern<arith::BitcastOp>>(typeConverter, ctx);
  patterns.add<BitcastToPyptoPattern<npuvector::BitcastOp>>(typeConverter, ctx);
}

/// Math dialect elementwise functions.
static void populateMathPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Math dialect elementwise functions with PyPTO counterparts.
  patterns.add<OpToPyptoOpPattern<math::AbsFOp, pypto::AbsOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<math::AbsIOp, pypto::AbsOp>>(typeConverter,
                                                               ctx);
  patterns.add<OpToPyptoOpPattern<math::ExpOp, pypto::ExpOp>>(typeConverter,
                                                              ctx);
  patterns.add<OpToPyptoOpPattern<math::LogOp, pypto::LogOp>>(typeConverter,
                                                              ctx);
  patterns.add<OpToPyptoOpPattern<math::SqrtOp, pypto::SqrtOp>>(typeConverter,
                                                                ctx);
  patterns.add<OpToPyptoOpPattern<math::RsqrtOp, pypto::RsqrtOp>>(
      typeConverter, ctx);
  patterns.add<OpToPyptoOpPattern<math::SinOp, pypto::SinOp>>(typeConverter,
                                                              ctx);
  patterns.add<OpToPyptoOpPattern<math::CosOp, pypto::CosOp>>(typeConverter,
                                                              ctx);
}

/// Tile ops, memref management, and control flow.
static void populateTileAndControlFlowPatterns(RewritePatternSet &patterns,
                                               TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // npuvector / vector tile ops.
  patterns.add<TransferReadPattern<npuvector::TransferReadOp>>(typeConverter,
                                                               ctx);
  patterns.add<TransferReadPattern<vector::TransferReadOp>>(typeConverter,
                                                            ctx);
  patterns.add<TransferWritePattern<npuvector::TransferWriteOp>>(typeConverter,
                                                                 ctx);
  patterns.add<TransferWritePattern<vector::TransferWriteOp>>(typeConverter,
                                                              ctx);
  patterns.add<ReductionPattern<npuvector::ReductionOp>>(typeConverter, ctx);
  patterns.add<ReductionPattern<vector::ReductionOp>>(typeConverter, ctx);
  patterns.add<BroadcastPattern<npuvector::BroadcastOp>>(typeConverter, ctx);
  patterns.add<BroadcastPattern<vector::BroadcastOp>>(typeConverter, ctx);
  patterns.add<TransposePattern<npuvector::TransposeOp>>(typeConverter, ctx);
  patterns.add<TransposePattern<vector::TransposeOp>>(typeConverter, ctx);

  // memref management ops.
  patterns.add<CollapseShapePattern>(typeConverter, ctx);
  patterns.add<ExpandShapePattern>(typeConverter, ctx);
  patterns.add<MemrefTransposePattern>(typeConverter, ctx);
  patterns.add<ReshapePattern>(typeConverter, ctx);
  patterns.add<SubviewPattern>(typeConverter, ctx);
  patterns.add<ReinterpretCastPattern>(typeConverter, ctx);
  patterns.add<DeallocPattern>(typeConverter, ctx);

  // Control flow: scf.parallel and scf.forall are the multi-core SPMD grid,
  // scf.for the per-core tile loop.
  patterns.add<ScfParallelPattern>(typeConverter, ctx);
  patterns.add<ScfForallPattern>(typeConverter, ctx);
  patterns.add<ScfForPattern>(typeConverter, ctx);
}

}  // namespace

void populateVectorToEmitPyConversionPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter) {
  populateFloatArithPatterns(patterns, typeConverter);
  populateIntArithPatterns(patterns, typeConverter);
  populateCmpSelectPatterns(patterns, typeConverter);
  populateCastPatterns(patterns, typeConverter);
  populateMathPatterns(patterns, typeConverter);
  populateTileAndControlFlowPatterns(patterns, typeConverter);

  // No func plumbing: signatures are rewritten by hand after the conversion
  // (see ConvertVectorToEmitPyPass::runOnOperation), and returns/calls are
  // legal as-is.
}

}  // namespace vector_to_emitpy

std::unique_ptr<Pass> createVectorToEmitPyConversionPass() {
  return std::make_unique<ConvertVectorToEmitPyPass>();
}

}  // namespace mlir
