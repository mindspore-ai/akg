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
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;  // NOLINT(build/namespaces)

namespace emitpy = mlir::emitpy;
namespace npuvector = mlir::npuvector;

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
static Attribute buildOffsetsArg(ConversionPatternRewriter &rewriter,
                                 Location loc, ValueRange indices,
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

/// Rewrites an integer/float comparison into `pl.cmp(a, b, cmp_type=N)`.
template <typename SourceOp>
class CmpToCallOpaquePattern : public OpConversionPattern<SourceOp> {
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
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getIndexAttr(1),
                                rewriter.getAttr<emitpy::OpaqueAttr>(
                                    std::to_string(*code))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("cmp_type")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResult(0).getType(),
        "pl.cmp", op->getOperands(), rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites a numeric cast into `pl.cast(v, pl.<dtype>)`.
template <typename SourceOp>
class CastToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    auto dtype = pyptoDtypeName(getElementTypeOrSelf(resultType));
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target type has no PyPTO dtype");
    }
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType,
        "pl.cast", op->getOperands(), rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites a bitcast into `pl.reinterpret_view(v, pl.<dtype>)`.
template <typename SourceOp>
class BitcastToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = op->getResult(0).getType();
    auto dtype = pyptoDtypeName(getElementTypeOrSelf(resultType));
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target type has no PyPTO dtype");
    }
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType,
        "pl.reinterpret_view", op->getOperands(),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
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
      SmallVector<Attribute> args{
          rewriter.getAttr<emitpy::OpaqueAttr>(
              shapeToPythonListText(shaped.getShape())),
          rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
          rewriter.getAttr<emitpy::OpaqueAttr>(*elem)};
      SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                      rewriter.getStringAttr(""),
                                      rewriter.getStringAttr("")};
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
          op, type, "pl.tile.full", ValueRange{},
          rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
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
    // Scalar constants carry their exact dtype through pl.const: a bare
    // Python literal would default to index/float64, and scalar ops such as
    // pl.write require the value dtype to match the tensor dtype.
    auto dtype = pyptoDtypeName(type);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "scalar type has no PyPTO dtype");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(*text),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, type, "pl.const", ValueRange{}, rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `transfer_read` into
/// `%r = pl.load(src, [o...], [shape])`; offsets may be loop-dependent.
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
    if (!shaped || !shaped.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "read result must be statically shaped");
    }
    SmallVector<Value> loadOperands{op.getSource()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, op.getLoc(),
                                           op.getIndices(), loadOperands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        offsetsArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(shaped.getShape()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getVector().getType(),
        "pl.load", ValueRange(loadOperands), rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
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
    if (valueType && destType && valueType.getRank() == 1 &&
        valueType.hasStaticShape() && destType.hasStaticShape() &&
        (destType.getRank() == 1 || destType.getRank() == 2)) {
      if (failed(storeRankOneViaReshape(rewriter, loc, op, vector, destType,
                                        valueType))) {
        return failure();
      }
      finishRewrite(rewriter, op);
      return success();
    }

    SmallVector<Value> operands{op.getVector(), op.getSource()};
    Attribute indexArg =
        buildOffsetsArg(rewriter, loc, op.getIndices(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), indexArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    // Note: build operand ranges explicitly; brace-initialized ValueRange
    // temporaries are not safe to pass through the OpBuilder call chain.
    rewriter.create<emitpy::CallOpaqueOp>(
        loc, TypeRange{}, "pl.store", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    finishRewrite(rewriter, op);
    return success();
  }

 private:
  /// If `vector` is a scalar broadcast feeding a single user -- either the
  /// still-unconverted broadcast op or its converted `pl.add(carrier, s)`
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
    if (auto call = dyn_cast<emitpy::CallOpaqueOp>(def)) {
      // The converted runtime-scalar broadcast: pl.add(carrier, scalar)
      // with a pl.tile.full carrier.
      if (call.getCallee() != "pl.add" || call->getNumOperands() != 2) {
        return Value();
      }
      Value scalar = call->getOperand(1);
      if (!isScalar(scalar.getType())) {
        return Value();
      }
      auto carrier = call->getOperand(0).getDefiningOp<emitpy::CallOpaqueOp>();
      if (!carrier || carrier.getCallee() != "pl.tile.full" ||
          carrier->getNumOperands() != 0) {
        return Value();
      }
      return scalar;
    }
    return Value();
  }

  /// Reads one scalar element out of a tile: `pl.read(tile, [k])`.
  static Value scalarRead(ConversionPatternRewriter &rewriter, Location loc,
                          Value tile, Type elemType,
                          const std::string &index) {
    SmallVector<Attribute> readArgs{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(index)};
    SmallVector<Attribute> readKeywords{rewriter.getStringAttr(""),
                                        rewriter.getStringAttr("")};
    SmallVector<Value> readOperands{tile};
    return rewriter
        .create<emitpy::CallOpaqueOp>(loc, elemType, "pl.read",
                                      ValueRange(readOperands),
                                      rewriter.getArrayAttr(readArgs),
                                      rewriter.getArrayAttr(readKeywords))
        .getResult(0);
  }

  /// Writes one scalar into `dest` at `indices` with `pl.write`.
  static void writeScalar(ConversionPatternRewriter &rewriter, Location loc,
                          Value dest, ValueRange indices, Value scalar) {
    SmallVector<Value> writeOperands{dest, scalar};
    Attribute indexArg =
        buildOffsetsArg(rewriter, loc, indices, writeOperands);
    SmallVector<Attribute> writeArgs{
        rewriter.getIndexAttr(0), indexArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> writeKeywords{rewriter.getStringAttr(""),
                                         rewriter.getStringAttr(""),
                                         rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        loc, TypeRange{}, "pl.write", ValueRange(writeOperands),
        rewriter.getArrayAttr(writeArgs),
        rewriter.getArrayAttr(writeKeywords));
  }

  /// Writes a scalar into a single-element destination with `pl.write`,
  /// addressing every destination dimension at 0.
  static void writeScalarToSingleElement(ConversionPatternRewriter &rewriter,
                                         Location loc, Value dest, Value scalar,
                                         ShapedType destType) {
    std::string zeros = "[0";
    for (int64_t d = 1; d < destType.getRank(); ++d) {
      zeros += ", 0";
    }
    zeros += "]";
    SmallVector<Value> writeOperands{dest, scalar};
    SmallVector<Attribute> writeArgs{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(zeros),
        rewriter.getIndexAttr(1)};
    SmallVector<Attribute> writeKeywords{rewriter.getStringAttr(""),
                                         rewriter.getStringAttr(""),
                                         rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        loc, TypeRange{}, "pl.write", ValueRange(writeOperands),
        rewriter.getArrayAttr(writeArgs),
        rewriter.getArrayAttr(writeKeywords));
  }

  /// Stores a rank-1 value through a `[1, N]` reshape: tile stores need a
  /// 2D tile. The destination is either rank-1 (a row offset of 0 is
  /// prepended) or rank-2 (the tile's own row offset is used directly).
  /// Sub-32-byte rows cannot hold a tile and fall back to element-wise
  /// `pl.write`s, which require a rank-1 destination and a constant base
  /// offset.
  static LogicalResult
  storeRankOneViaReshape(ConversionPatternRewriter &rewriter, Location loc,
                         TransferWriteOp op, Value vector, ShapedType destType,
                         ShapedType valueType) {
    Type elemType = valueType.getElementType();
    int64_t n = valueType.getNumElements();
    unsigned elemBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (n * elemBytes % 32 != 0) {
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
        Value elem = scalarRead(rewriter, loc, vector, elemType,
                                "[" + std::to_string(k) + "]");
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
    if (destType.getRank() == 1) {
      // Prepend the [1, N] row offset.
      offsetItems.push_back(rewriter.create<emitpy::LiteralOp>(
          loc, rewriter.getIndexType(), "0"));
    }
    llvm::append_range(offsetItems, op.getIndices());
    SmallVector<Value> storeOperands{reshaped, op.getSource()};
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

/// Reduction kinds map onto the pl.row_* / pl.col_* families; bitwise
/// combining has no PyPTO counterpart.
static StringRef reduceBaseName(vector::CombiningKind kind) {
  switch (kind) {
  case vector::CombiningKind::ADD:
    return "sum";
  case vector::CombiningKind::MUL:
    return "prod";
  case vector::CombiningKind::MINUI:
  case vector::CombiningKind::MINSI:
  case vector::CombiningKind::MINNUMF:
  case vector::CombiningKind::MINIMUMF:
    return "min";
  case vector::CombiningKind::MAXUI:
  case vector::CombiningKind::MAXSI:
  case vector::CombiningKind::MAXNUMF:
  case vector::CombiningKind::MAXIMUMF:
    return "max";
  default:
    return "";
  }
}

/// The scalar-form PyPTO op that combines the elements of a full
/// reduction; `pl.minimum`/`pl.maximum` have no scalar form.
static StringRef scalarCombineCallee(StringRef base) {
  if (base == "sum") {
    return "pl.add";
  }
  if (base == "prod") {
    return "pl.mul";
  }
  return StringRef();
}

/// Reads one element out of a tile: `pl.read(tile, [i, 0])`.
static Value readTileElement(ConversionPatternRewriter &rewriter, Location loc,
                             Type resultType, Value tile,
                             const std::string &index) {
  SmallVector<Attribute> readArgs{
      rewriter.getIndexAttr(0),
      rewriter.getAttr<emitpy::OpaqueAttr>(index)};
  SmallVector<Attribute> readKeywords{rewriter.getStringAttr(""),
                                      rewriter.getStringAttr("")};
  SmallVector<Value> readOperands{tile};
  return rewriter
      .create<emitpy::CallOpaqueOp>(loc, resultType, "pl.read",
                                    ValueRange(readOperands),
                                    rewriter.getArrayAttr(readArgs),
                                    rewriter.getArrayAttr(readKeywords))
      .getResult(0);
}

/// Combines two reduction partials with the scalar-form op; a null `acc`
/// (the first element) passes through unchanged.
static Value combineScalars(ConversionPatternRewriter &rewriter, Location loc,
                            Type resultType, StringRef callee, Value acc,
                            Value elem) {
  if (!acc) {
    return elem;
  }
  SmallVector<Value> combineOperands{acc, elem};
  return rewriter
      .create<emitpy::CallOpaqueOp>(loc, resultType, callee,
                                    ValueRange(combineOperands))
      .getResult(0);
}

/// Renders flat element index `k` of a col-major tile with extents `dims`
/// as a Python index list, e.g. `[1, 2]`.
static std::string flatIndexToList(ArrayRef<int64_t> dims, int64_t k) {
  std::string index = "[";
  for (unsigned d = 0; d < dims.size(); ++d) {
    int64_t stride = 1;
    for (unsigned t = d + 1; t < dims.size(); ++t) {
      stride *= dims[t];
    }
    if (d != 0) {
      index += ", ";
    }
    index += std::to_string((k / stride) % dims[d]);
  }
  index += "]";
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
    StringRef base = reduceBaseName(op.getKind());
    if (base.empty()) {
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
        return rewritePartialReduction(op, rewriter, base, shaped);
      }
    }

    return rewriteFullReduction(op, rewriter, base, shaped);
  }

 private:
  /// Reduces one axis in place: the last axis with `pl.row_*` or the first
  /// axis with `pl.col_*`, consuming a scratch tile and folding the
  /// accumulator with `pl.add`.
  static LogicalResult
  rewritePartialReduction(npuvector::ReductionOp op,
                          ConversionPatternRewriter &rewriter,
                          StringRef base, ShapedType shaped) {
    Location loc = op.getLoc();
    auto dims = op.getReductionDims();
    if (dims->size() != 1) {
      return rewriter.notifyMatchFailure(
          op, "only single-axis reductions are supported");
    }
    int64_t dim = (*dims)[0];
    unsigned rank = shaped.getRank();
    std::string callee;
    if (dim == static_cast<int64_t>(rank - 1)) {
      callee = "pl.row_" + base.str();
    } else if (dim == 0) {
      callee = "pl.col_" + base.str();
    } else {
      return rewriter.notifyMatchFailure(
          op, "only first/last axis reductions map onto pl.row_*/pl.col_*");
    }
    Type resultType = op->getResult(0).getType();
    Value tmp = createTmpTile(rewriter, loc, shaped.getShape(),
                              shaped.getElementType());
    SmallVector<Value> reduceOperands{op.getVector(), tmp};
    Value reduced = rewriter
                        .create<emitpy::CallOpaqueOp>(
                            loc, resultType, callee,
                            ValueRange(reduceOperands))
                        .getResult(0);
    if (op.getAcc()) {
      SmallVector<Value> accOperands{reduced, op.getAcc()};
      reduced = rewriter
                    .create<emitpy::CallOpaqueOp>(
                        loc, resultType, "pl.add",
                        ValueRange(accOperands))
                    .getResult(0);
    }
    rewriter.replaceOp(op, reduced);
    return success();
  }

  /// Folds a full reduction to a scalar. Device backends cannot allocate
  /// [1, 1] tiles, so the final value is assembled from scalar reads: when
  /// the operand tiles are allocatable (extents a whole number of 32-byte
  /// units) the last axis is reduced on device with `pl.row_*` and only the
  /// surviving elements are read out; otherwise every element is read
  /// directly. The elements are combined with scalar-form ops --
  /// `pl.minimum`/`pl.maximum` have no scalar form, so min/max full
  /// reductions are rejected.
  static LogicalResult
  rewriteFullReduction(ReductionOp op, ConversionPatternRewriter &rewriter,
                       StringRef base, ShapedType shaped) {
    Location loc = op.getLoc();
    StringRef scalarCallee = scalarCombineCallee(base);
    if (scalarCallee.empty()) {
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
      acc = reduceRankOneViaReshape(rewriter, loc, base, shaped, resultType,
                                    input, alignRows);
    } else if (shaped.getRank() == 2 && cols > 1 &&
               rows * elemBytes % 32 == 0 && cols * elemBytes % 32 == 0) {
      acc = reduceViaDeviceRowReduce(rewriter, loc, base, shaped, resultType,
                                     input, rows);
    } else {
      acc = reduceViaScalarChain(rewriter, loc, scalarCallee, shaped,
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
                          StringRef base, ShapedType shaped, Type resultType,
                          Value input, unsigned alignRows) {
    StringRef scalarCallee = scalarCombineCallee(base);
    int64_t r = alignRows;
    int64_t c = shaped.getNumElements() / r;
    Value reshaped =
        reshapeTileOp(rewriter, loc, input, r, c, shaped.getElementType());
    Value rowTmp = createTmpTile(rewriter, loc, SmallVector<int64_t>{r, c},
                                 shaped.getElementType());
    SmallVector<Value> rowOperands{reshaped, rowTmp};
    Value row = rewriter
                    .create<emitpy::CallOpaqueOp>(
                        loc, resultType, "pl.row_" + base.str(),
                        ValueRange(rowOperands))
                    .getResult(0);
    return combineRowPartials(rewriter, loc, scalarCallee, resultType, row, r);
  }

  /// Hybrid lowering: row-reduce to [rows, 1] on device, then combine the
  /// surviving `rows` scalars.
  static Value
  reduceViaDeviceRowReduce(ConversionPatternRewriter &rewriter, Location loc,
                           StringRef base, ShapedType shaped, Type resultType,
                           Value input, int64_t rows) {
    StringRef scalarCallee = scalarCombineCallee(base);
    Value rowTmp = createTmpTile(rewriter, loc, shaped.getShape(),
                                 shaped.getElementType());
    SmallVector<Value> rowOperands{input, rowTmp};
    Value row = rewriter
                    .create<emitpy::CallOpaqueOp>(
                        loc, resultType, "pl.row_" + base.str(),
                        ValueRange(rowOperands))
                    .getResult(0);
    return combineRowPartials(rewriter, loc, scalarCallee, resultType, row,
                              rows);
  }

  /// Reads the surviving `[i, 0]` partials out of a keepdim row reduction
  /// and combines them with the scalar-form op.
  static Value combineRowPartials(ConversionPatternRewriter &rewriter,
                                  Location loc, StringRef scalarCallee,
                                  Type resultType, Value row, int64_t rows) {
    Value acc;
    for (int64_t i = 0; i < rows; ++i) {
      Value elem = readTileElement(rewriter, loc, resultType, row,
                                   "[" + std::to_string(i) + ", 0]");
      acc = combineScalars(rewriter, loc, resultType, scalarCallee, acc, elem);
    }
    return acc;
  }

  /// Pure scalar chain: read and combine every element.
  static Value
  reduceViaScalarChain(ConversionPatternRewriter &rewriter, Location loc,
                       StringRef scalarCallee, ShapedType shaped,
                       Type resultType, Value input) {
    SmallVector<int64_t> dims(shaped.getShape());
    Value acc;
    for (int64_t k = 0, e = shaped.getNumElements(); k < e; ++k) {
      Value elem = readTileElement(rewriter, loc, resultType, input,
                                   flatIndexToList(dims, k));
      acc = combineScalars(rewriter, loc, resultType, scalarCallee, acc, elem);
    }
    return acc;
  }
};

/// Rewrites a scalar broadcast:
/// - constant scalars materialize directly as `pl.tile.full(shape, dtype, v)`;
/// - runtime scalars are broadcast by adding them onto a zero carrier tile:
///   `pl.add(pl.tile.full(shape, dtype, 0), scalar)`. `pl.expands` has no
///   codegen on current device backends, while the scalar form of `pl.add`
///   broadcasts for free.
/// Tile-to-tile broadcasts have no PyPTO counterpart.
template <typename BroadcastOp>
class BroadcastPattern : public OpConversionPattern<BroadcastOp> {
 public:
  using OpConversionPattern<BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BroadcastOp op, typename BroadcastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "only statically-shaped broadcasts are supported");
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
    std::string shapeText = shapeToPythonListText(resultType.getShape());
    Type convertedResult = op->getResult(0).getType();

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
      SmallVector<Attribute> args{
          rewriter.getAttr<emitpy::OpaqueAttr>(shapeText),
          rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
          rewriter.getAttr<emitpy::OpaqueAttr>(*constText)};
      SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                      rewriter.getStringAttr(""),
                                      rewriter.getStringAttr("")};
      result = rewriter
                   .create<emitpy::CallOpaqueOp>(
                       loc, convertedResult, "pl.tile.full", ValueRange{},
                       rewriter.getArrayAttr(args),
                       rewriter.getArrayAttr(keywords))
                   .getResult(0);
    } else {
      std::string zero = isa<FloatType>(sourceType) ? "0.0" : "0";
      SmallVector<Attribute> carrierArgs{
          rewriter.getAttr<emitpy::OpaqueAttr>(shapeText),
          rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
          rewriter.getAttr<emitpy::OpaqueAttr>(zero)};
      SmallVector<Attribute> carrierKeywords{rewriter.getStringAttr(""),
                                             rewriter.getStringAttr(""),
                                             rewriter.getStringAttr("")};
      Value carrier = rewriter
                          .create<emitpy::CallOpaqueOp>(
                              loc, convertedResult, "pl.tile.full",
                              ValueRange{}, rewriter.getArrayAttr(carrierArgs),
                              rewriter.getArrayAttr(carrierKeywords))
                          .getResult(0);
      SmallVector<Value> broadcastOperands{carrier, op.getSource()};
      result = rewriter.create<emitpy::CallOpaqueOp>(
                   loc, convertedResult, "pl.add",
                   ValueRange(broadcastOperands))
                   .getResult(0);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Rewrites a transpose into `pl.transpose(v, i, j)`: PyPTO swaps exactly
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
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(swap->first)),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(swap->second))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    SmallVector<Value> transposeOperands{op.getVector()};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResult(0).getType(),
        "pl.transpose", ValueRange(transposeOperands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites a select into `pl.sel(cond, lhs, rhs, tmp)` with an inline
/// scratch tile. Constant scalar branches are materialized as
/// `pl.tile.full`, since pl.sel requires tile operands.
template <typename SelectOp>
class SelectPattern : public OpConversionPattern<SelectOp> {
 public:
  using OpConversionPattern<SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SelectOp op, typename SelectOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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
    SmallVector<Value> selOperands{op.getCondition(), trueBranch, falseBranch,
                                   tmp};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, convertedResult, "pl.sel", ValueRange(selOperands));
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

static void markWriteDest(Value dest, DenseSet<unsigned> &writeDests) {
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

/// Returns true when `arg` is used by anything other than a store into it,
/// i.e. it is also read.
static bool argHasNonWriteUse(BlockArgument arg) {
  for (OpOperand &use : arg.getUses()) {
    if (!isStoreInto(use)) {
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
/// which have no device form. A 0-D read becomes pl.read of the single
/// element, a 0-D broadcast is its source itself, any other op's results
/// adopt the scalar element type, and 0-D stores are left for the
/// transfer-write pattern (which writes scalars through pl.write). Doing
/// this ahead of the conversion keeps the value types stable while the
/// patterns rewrite ops one by one.
static void lowerZeroDimChains(func::FuncOp funcOp, MLIRContext *ctx) {
  SmallVector<Operation *> scalarOps;
  funcOp.walk([&](Operation *op) {
    if (llvm::any_of(op->getResultTypes(), isScalarTileType)) {
      scalarOps.push_back(op);
    }
  });
  OpBuilder builder(funcOp);
  for (Operation *op : scalarOps) {
    if (isa<npuvector::TransferReadOp, vector::TransferReadOp>(op)) {
      // %v = pl.read(src, [0]): read the single element of the unfolded
      // [1] buffer as a scalar.
      Value source = op->getOperand(0);
      auto shaped = cast<ShapedType>(op->getResult(0).getType());
      builder.setInsertionPoint(op);
      SmallVector<Attribute> readArgs{
          builder.getIndexAttr(0),
          emitpy::OpaqueAttr::get(ctx, "[0]")};
      SmallVector<Attribute> readKeywords{builder.getStringAttr(""),
                                          builder.getStringAttr("")};
      auto call = builder.create<emitpy::CallOpaqueOp>(
          op->getLoc(), shaped.getElementType(), "pl.read",
          ValueRange{source}, builder.getArrayAttr(readArgs),
          builder.getArrayAttr(readKeywords));
      op->getResult(0).replaceAllUsesWith(call.getResult(0));
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

struct ConvertVectorToEmitPyPass
    : impl::ConvertVectorToEmitPyBase<ConvertVectorToEmitPyPass> {
  using impl::ConvertVectorToEmitPyBase<
      ConvertVectorToEmitPyPass>::ConvertVectorToEmitPyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect,
                    scf::SCFDialect>();
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

    lowerZeroDimChains(funcOp, ctx);

    VectorToEmitPyTypeConverter typeConverter(ctx);
    ConversionTarget target(*ctx);
    target.addLegalDialect<emitpy::EmitPyDialect>();
    target.addIllegalDialect<npuvector::NPUVectorDialect, vector::VectorDialect,
                             arith::ArithDialect, math::MathDialect>();
    // Control flow maps onto SPMD launches and tile loops; any other scf
    // construct has no PyPTO counterpart.
    target.addIllegalDialect<scf::SCFDialect>();
    // Allocations are handled by rewriteFunctionOutputs; any survivor means
    // a use this backend cannot express.
    target.addIllegalOp<memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
                        memref::LoadOp, memref::StoreOp>();
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
  patterns.add<OpToCallOpaquePattern<arith::AddFOp>>(typeConverter, ctx,
                                                     "pl.add");
  patterns.add<OpToCallOpaquePattern<arith::SubFOp>>(typeConverter, ctx,
                                                     "pl.sub");
  patterns.add<OpToCallOpaquePattern<arith::MulFOp>>(typeConverter, ctx,
                                                     "pl.mul");
  patterns.add<OpToCallOpaquePattern<arith::DivFOp>>(typeConverter, ctx,
                                                     "pl.div");
  patterns.add<OpToCallOpaquePattern<arith::RemFOp>>(typeConverter, ctx,
                                                     "pl.fmod");
  patterns.add<OpToCallOpaquePattern<arith::MaximumFOp>>(typeConverter, ctx,
                                                         "pl.maximum");
  patterns.add<OpToCallOpaquePattern<arith::MinimumFOp>>(typeConverter, ctx,
                                                         "pl.minimum");
  patterns.add<OpToCallOpaquePattern<arith::NegFOp>>(typeConverter, ctx,
                                                     "pl.neg");
}

/// Elementwise integer arithmetic.
static void populateIntArithPatterns(RewritePatternSet &patterns,
                                     TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Elementwise integer arithmetic.
  patterns.add<OpToCallOpaquePattern<arith::AddIOp>>(typeConverter, ctx,
                                                     "pl.add");
  patterns.add<OpToCallOpaquePattern<arith::SubIOp>>(typeConverter, ctx,
                                                     "pl.sub");
  patterns.add<OpToCallOpaquePattern<arith::MulIOp>>(typeConverter, ctx,
                                                     "pl.mul");
  patterns.add<OpToCallOpaquePattern<arith::DivSIOp>>(typeConverter, ctx,
                                                      "pl.div");
  patterns.add<OpToCallOpaquePattern<arith::DivUIOp>>(typeConverter, ctx,
                                                      "pl.div");
  patterns.add<OpToCallOpaquePattern<arith::RemSIOp>>(typeConverter, ctx,
                                                      "pl.fmod");
  patterns.add<OpToCallOpaquePattern<arith::RemUIOp>>(typeConverter, ctx,
                                                      "pl.fmod");
  patterns.add<OpToCallOpaquePattern<arith::MaxSIOp>>(typeConverter, ctx,
                                                      "pl.maximum");
  patterns.add<OpToCallOpaquePattern<arith::MinSIOp>>(typeConverter, ctx,
                                                      "pl.minimum");
  patterns.add<OpToCallOpaquePattern<arith::MaxUIOp>>(typeConverter, ctx,
                                                      "pl.maximum");
  patterns.add<OpToCallOpaquePattern<arith::MinUIOp>>(typeConverter, ctx,
                                                      "pl.minimum");
  patterns.add<OpToCallOpaquePattern<arith::AndIOp>>(typeConverter, ctx,
                                                     "pl.and_");
  patterns.add<OpToCallOpaquePattern<arith::OrIOp>>(typeConverter, ctx,
                                                    "pl.or_");
  patterns.add<OpToCallOpaquePattern<arith::XOrIOp>>(typeConverter, ctx,
                                                     "pl.xor");
  patterns.add<OpToCallOpaquePattern<arith::ShLIOp>>(typeConverter, ctx,
                                                     "pl.shl");
  patterns.add<OpToCallOpaquePattern<arith::ShRSIOp>>(typeConverter, ctx,
                                                      "pl.shr");
  patterns.add<OpToCallOpaquePattern<arith::ShRUIOp>>(typeConverter, ctx,
                                                      "pl.shr");
}

/// Comparisons and select.
static void populateCmpSelectPatterns(RewritePatternSet &patterns,
                                      TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Comparisons and select.
  patterns.add<CmpToCallOpaquePattern<arith::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToCallOpaquePattern<arith::CmpFOp>>(typeConverter, ctx);
  patterns.add<CmpToCallOpaquePattern<npuvector::CmpIOp>>(typeConverter, ctx);
  patterns.add<CmpToCallOpaquePattern<npuvector::CmpFOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<arith::SelectOp>>(typeConverter, ctx);
  patterns.add<SelectPattern<npuvector::SelectOp>>(typeConverter, ctx);
}

/// Numeric casts (pl.cast) and bitcasts (pl.reinterpret_view).
static void populateCastPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Numeric casts -> pl.cast; bitcasts -> pl.reinterpret_view.
  patterns.add<CastToCallOpaquePattern<arith::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::TruncFOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::ExtSIOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::ExtUIOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::TruncIOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::SIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::UIToFPOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::FPToSIOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::FPToUIOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::IndexCastOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<arith::IndexCastUIOp>>(typeConverter,
                                                             ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::ExtFOp>>(typeConverter, ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::TruncFOp>>(typeConverter,
                                                             ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::ExtSIOp>>(typeConverter,
                                                           ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::ExtUIOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::TruncIOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::SIToFPOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::UIToFPOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::FPToSIOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::FPToUIOp>>(typeConverter,
                                                            ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::IndexCastOp>>(typeConverter,
                                                               ctx);
  patterns.add<CastToCallOpaquePattern<npuvector::IndexCastUIOp>>(
      typeConverter, ctx);
  patterns.add<BitcastToCallOpaquePattern<arith::BitcastOp>>(typeConverter,
                                                             ctx);
  patterns.add<BitcastToCallOpaquePattern<npuvector::BitcastOp>>(typeConverter,
                                                                 ctx);
}

/// Math dialect elementwise functions.
static void populateMathPatterns(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  MLIRContext *ctx = patterns.getContext();

  // Math dialect elementwise functions with PyPTO counterparts.
  patterns.add<OpToCallOpaquePattern<math::AbsFOp>>(typeConverter, ctx,
                                                    "pl.abs");
  patterns.add<OpToCallOpaquePattern<math::AbsIOp>>(typeConverter, ctx,
                                                    "pl.abs");
  patterns.add<OpToCallOpaquePattern<math::ExpOp>>(typeConverter, ctx,
                                                   "pl.exp");
  patterns.add<OpToCallOpaquePattern<math::LogOp>>(typeConverter, ctx,
                                                   "pl.log");
  patterns.add<OpToCallOpaquePattern<math::SqrtOp>>(typeConverter, ctx,
                                                    "pl.sqrt");
  patterns.add<OpToCallOpaquePattern<math::RsqrtOp>>(typeConverter, ctx,
                                                     "pl.rsqrt");
  patterns.add<OpToCallOpaquePattern<math::SinOp>>(typeConverter, ctx,
                                                   "pl.sin");
  patterns.add<OpToCallOpaquePattern<math::CosOp>>(typeConverter, ctx,
                                                   "pl.cos");
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
  patterns.add<DeallocPattern>(typeConverter, ctx);

  // Control flow: scf.parallel is the multi-core SPMD grid, scf.for the
  // per-core tile loop.
  patterns.add<ScfParallelPattern>(typeConverter, ctx);
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
