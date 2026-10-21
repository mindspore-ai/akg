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

#include "akg/Conversion/PyptoToEmitPy/PyptoToEmitPy.h"

#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

#include "akg/Dialect/EmitPy/IR/EmitPyOps.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "akg/Dialect/Pypto/IR/PyptoTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;  // NOLINT(build/namespaces)

namespace emitpy = mlir::emitpy;
namespace npuvector = mlir::npuvector;
namespace pypto = mlir::pypto;

namespace mlir {

#define GEN_PASS_DEF_CONVERTPYPTOTOEMITPY
#include "akg/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// PyPTO type mapping
//===----------------------------------------------------------------------===//

/// Maps an MLIR scalar element type onto the PyPTO dtype name, e.g.
/// f32 -> "pl.FP32", i32 -> "pl.INT32". Mirrors the mapping in
/// VectorToPypto.cpp; kept local so the pypto lowering does not depend on
/// the vector-to-emitpy conversion library.
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

/// The dtype name carried by an op result: a tile's element type, a
/// container's element type (the pipeline form keeps vector/npuvector
/// result types), or the scalar type itself.
static std::optional<std::string> resultDtypeName(Type type) {
  if (auto tile = dyn_cast<pypto::TileType>(type)) {
    return pyptoDtypeName(tile.getElementType());
  }
  return pyptoDtypeName(getElementTypeOrSelf(type));
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

/// Renders an integer value as Python source text.
static std::optional<std::string> intToPythonText(const APInt &value) {
  if (value.getSignificantBits() > 64) {
    return std::nullopt;
  }
  return std::to_string(value.getSExtValue());
}

/// Renders a float value as Python source text. Mirrors the helper in
/// VectorToPypto.cpp; kept local so the pass stays self-contained.
static std::string floatToPythonText(APFloat value) {
  if (value.isInfinity()) {
    return value.isNegative() ? "float('-inf')" : "float('inf')";
  }
  if (value.isNaN()) {
    return "float('nan')";
  }
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<double>::max_digits10) << value.convertToDouble();
  std::string text = oss.str();
  if (text.find('.') == std::string::npos && text.find('e') == std::string::npos &&
      text.find('E') == std::string::npos) {
    text += ".0";
  }
  return text;
}

/// Renders a constant scalar operand as `pl.const(value, dtype)` source
/// text: the raw `arith.constant` has no emitpy printer, and a bare
/// Python literal would default to float64/int semantics instead of the
/// operand's dtype. Accepts every spelling the fixpoint may present —
/// arith.constant, pypto.constant, and the emitpy pl.const call the
/// constant pattern lowers it to. Non-constant scalars return nullopt.
static std::optional<std::string> constScalarText(Value value) {
  // A pypto.constant carries the Python literal text directly; the DSL
  // parser only keeps a pl.const call a compile-time constant when it is
  // written inline at the use site, so the emitpy form must spell it out
  // here rather than reference a bound pl.const statement.
  if (auto pyptoConst = value.getDefiningOp<pypto::ConstantOp>()) {
    auto dtype = resultDtypeName(value.getType());
    if (!dtype) {
      return std::nullopt;
    }
    return "pl.const(" + pyptoConst.getValue().str() + ", " + *dtype + ")";
  }
  // When the constant pattern has already run, the value is an emitpy
  // pl.const call; recover the literal from its opaque args.
  if (auto call = value.getDefiningOp<emitpy::CallOpaqueOp>();
      call && call.getCallee() == "pl.const" && call->getNumOperands() == 0) {
    auto args = call.getArgs();
    if (args && args->size() == 2 && isa<emitpy::OpaqueAttr>((*args)[0]) && isa<emitpy::OpaqueAttr>((*args)[1])) {
      return "pl.const(" + cast<emitpy::OpaqueAttr>((*args)[0]).getValue().str() + ", " +
             cast<emitpy::OpaqueAttr>((*args)[1]).getValue().str() + ")";
    }
    return std::nullopt;
  }
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return std::nullopt;
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
    return std::nullopt;
  }
  auto dtype = resultDtypeName(value.getType());
  if (!dtype) {
    return std::nullopt;
  }
  return "pl.const(" + *text + ", " + *dtype + ")";
}

/// Extracts the constant integer of an index operand, or nullopt when the
/// operand is a runtime value. Stage one keeps index constants as
/// `arith.constant`; this pass rewrites them into `emitpy.literal`, and
/// patterns run in arbitrary order, so both spellings are accepted.
static std::optional<int64_t> constIndexValue(Value idx) {
  if (auto constOp = idx.getDefiningOp<arith::ConstantOp>()) {
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr) {
      return std::nullopt;
    }
    return intAttr.getInt();
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

/// Builds the `[i0, i1, ...]` offsets argument for pl.load/pl.store.
/// Constant-only offsets render directly as an opaque list literal; dynamic
/// offsets become operand references (IntegerAttr of index type) and the
/// index values are appended to `operands`, matching the call_opaque
/// argument protocol used by the vector-to-emitpy lowering.
static Attribute buildOffsetsArg(ConversionPatternRewriter &rewriter, ValueRange offsets,
                                 SmallVectorImpl<Value> &operands) {
  bool allConstant = llvm::all_of(offsets, [](Value offset) { return constIndexValue(offset).has_value(); });
  if (allConstant) {
    std::string text = "[";
    bool first = true;
    for (Value offset : offsets) {
      if (!first) {
        text += ", ";
      }
      first = false;
      text += std::to_string(*constIndexValue(offset));
    }
    text += "]";
    return rewriter.getAttr<emitpy::OpaqueAttr>(text);
  }
  SmallVector<Attribute> items;
  for (Value offset : offsets) {
    operands.push_back(offset);
    items.push_back(rewriter.getIndexAttr(operands.size() - 1));
  }
  return rewriter.getArrayAttr(items);
}

/// The reduce-kind suffix shared by the `pl.row_*` / `pl.col_*` families.
static StringRef reduceKindSuffix(pypto::ReduceKind kind) {
  switch (kind) {
    case pypto::ReduceKind::sum:
      return "sum";
    case pypto::ReduceKind::prod:
      return "prod";
    case pypto::ReduceKind::min:
      return "min";
    case pypto::ReduceKind::max:
      return "max";
  }
  return "";
}

/// The axis prefix of the `pl.row_*` / `pl.col_*` families: `row` folds the
/// last axis, `col` the first one.
static StringRef reduceAxisPrefix(pypto::ReduceAxis axis) {
  switch (axis) {
    case pypto::ReduceAxis::row:
      return "row_";
    case pypto::ReduceAxis::col:
      return "col_";
  }
  return "";
}

/// Renders the opaque PyPTO annotation `wrapper[[shape], dtype]` for a
/// statically known shape, e.g. `pl.Tensor[[512], pl.FP32]`.
static Type pyptoShapedOpaque(MLIRContext *ctx, ArrayRef<int64_t> shape, Type elemType, StringRef wrapper) {
  auto dtype = pyptoDtypeName(elemType);
  if (!dtype) {
    return Type();
  }
  std::string text;
  llvm::raw_string_ostream rso(text);
  rso << wrapper << "[" << shapeToPythonListText(shape) << ", " << *dtype << "]";
  rso.flush();
  return emitpy::OpaqueType::get(ctx, text);
}

static Type pyptoShapedOpaque(MLIRContext *ctx, ShapedType type, StringRef wrapper) {
  if (!type.hasStaticShape()) {
    return Type();
  }
  return pyptoShapedOpaque(ctx, type.getShape(), type.getElementType(), wrapper);
}

/// Renders the opaque PyPTO annotation `pl.Scalar[dtype]`.
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

/// The PyPTO annotation of a function parameter: tensors carry
/// `pl.Tensor[[shape], dtype]`, scalars `pl.Scalar[dtype]`.
static Type pyptoAnnotation(MLIRContext *ctx, Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return pyptoShapedOpaque(ctx, shaped, "pl.Tensor");
  }
  if (isa<FloatType>(type) || isa<IntegerType>(type) || isa<IndexType>(type)) {
    return pyptoScalarOpaque(ctx, type);
  }
  return Type();
}

/// Converts `!pypto.tile<elem, [shape]>` into the opaque PyPTO annotation
/// `!emitpy.opaque<"pl.Tile[[shape], dtype]">`: this is the single place
/// where the tile model is encoded as a Python-side annotation string.
class PyptoToEmitPyTypeConverter : public TypeConverter {
 public:
  PyptoToEmitPyTypeConverter() {
    // Fallback first: the framework tries conversions in reverse order of
    // registration, so the identity fallback must be added before the
    // tile-specific one (mirrors NPUVectorToVector.cpp).
    addConversion([](Type type) { return type; });
    // Returns `std::nullopt` (not an empty Type) when the element dtype has
    // no annotation spelling, so the framework falls back to the identity
    // conversion above: a null return is a hard failure and would leave
    // e.g. `!pypto.tile<i1, ...>` (cmp results) unconvertible.
    addConversion([](pypto::TileType type) -> std::optional<Type> {
      // A `?` extent cannot be spelled in the annotation; dynamic tiles
      // keep their source type, which the Python translator accepts for
      // call_opaque results (as it already does for dynamic vector tiles).
      if (llvm::any_of(type.getShape(), [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
        return type;
      }
      auto dtype = pyptoDtypeName(type.getElementType());
      if (!dtype) {
        return std::nullopt;
      }
      std::string text;
      llvm::raw_string_ostream rso(text);
      rso << "pl.Tile[" << shapeToPythonListText(type.getShape()) << ", " << *dtype << "]";
      rso.flush();
      return emitpy::OpaqueType::get(type.getContext(), text);
    });
  }
};

//===----------------------------------------------------------------------===//
// Generic positional-call patterns
//===----------------------------------------------------------------------===//

/// Rewrites an op into `emitpy.call_opaque` with the given callee and all
/// operands passed positionally, e.g.
/// `pypto.add %a, %b` -> `%r = pl.add(a, b)`.
/// Covers the elementwise binaries/unaries and the scalar combines.
template <typename SourceOp>
class OpToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  OpToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context, StringRef callee)
      : OpConversionPattern<SourceOp>(typeConverter, context), callee(callee) {}

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, callee, adaptor.getOperands());
    return success();
  }

 private:
  StringRef callee;
};

/// Rewrites `pypto.select` into `pl.tile.select`: the runtime's composite
/// synthesizes the architecture-defined scratch tile, so no scratch is
/// materialized here and dynamic result shapes are legal. A constant
/// scalar branch rides inline as `pl.const(...)` text — the composite
/// materializes it with `tile.full`, which demands a literal, and the
/// raw `arith.constant` has no emitpy printer.
class SelectOpPattern : public OpConversionPattern<pypto::SelectOp> {
 public:
  using OpConversionPattern<pypto::SelectOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    auto isTile = [](Value value) { return isa<ShapedType, pypto::TileType>(value.getType()); };
    // Fast path: every operand tile-typed is a plain positional call,
    // byte-identical to the generic pattern.
    if (llvm::all_of(op->getOperands(), isTile)) {
      rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.tile.select", adaptor.getOperands());
      return success();
    }
    // Scalar-branch path: a constant branch inlines as pl.const(...) text —
    // the DSL parser keeps a pl.const a compile-time constant only when it
    // appears inline at the call site, so a referenced constant statement
    // would arrive as a runtime Var, which the composite cannot materialize.
    // Anything else is a runtime scalar, likewise rejected.
    SmallVector<Value> operands;
    SmallVector<Attribute> args;
    SmallVector<Operation *> inlinedDefs;
    auto converted = adaptor.getOperands();
    for (auto [value, convertedValue] : llvm::zip(op->getOperands(), converted)) {
      if (isTile(value)) {
        args.push_back(rewriter.getIndexAttr(operands.size()));
        operands.push_back(convertedValue);
        continue;
      }
      if (auto text = constScalarText(value)) {
        args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(*text));
        inlinedDefs.push_back(value.getDefiningOp());
        continue;
      }
      return rewriter.notifyMatchFailure(op, "scalar select branch must be a constant");
    }
    if (operands.size() < 2) {
      return rewriter.notifyMatchFailure(op, "select needs at least one tile branch");
    }
    SmallVector<Attribute> keywords(3, rewriter.getStringAttr(""));
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.tile.select", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    // Constants consumed inline as text leave their raw arith.constant
    // dead; drop them so the translator never has to print them.
    for (Operation *def : inlinedDefs) {
      if (def->use_empty()) {
        rewriter.eraseOp(def);
      }
    }
    return success();
  }
};

/// Callee for a scalar arith op: the operation's own name minus the
/// dialect prefix ("arith.addi" -> "addi"), so the emitpy IR spells scalar
/// math with the arith mnemonic itself instead of a parallel `util_*`
/// pseudo-callee family.
template <typename ArithOp>
static StringRef scalarArithCallee() {
  return ArithOp::getOperationName().split('.').second;
}

/// Registers one OpToCallOpaquePattern per scalar arith op, with the
/// callee derived from each op's name (no hand-spelled callee strings).
template <typename... ArithOps>
static void addScalarArithPatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  (patterns.add<OpToCallOpaquePattern<ArithOps>>(typeConverter, patterns.getContext(), scalarArithCallee<ArithOps>()),
   ...);
}

//===----------------------------------------------------------------------===//
// Stage-one survivors: scalar pseudo-callees and index constants
//===----------------------------------------------------------------------===//

/// Scalar comparison pseudo-callees, indexed by comparison kind
/// (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge) — the arith predicate mnemonics
/// with signed/unsigned collapsed. The PyPTO runtime's `pl.cmp`
/// accepts Tensor/Tile operands only, so scalar comparisons (e.g. the
/// cmp/div/select chains of the built-in affine floordiv lowering) go
/// through these pseudo-callees, which the Python translator prints as
/// infix comparisons that the PyPTO parser maps onto ir.lt etc.
static StringRef scalarCmpCalleeFor(unsigned cmpTypeCode) {
  static const StringRef callees[] = {"eq", "ne", "lt", "le", "gt", "ge"};
  return cmpTypeCode < std::size(callees) ? callees[cmpTypeCode] : StringRef();
}

/// Comparison kind codes shared with the pypto.cmp predicate values:
/// 0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge. Signed and unsigned integer
/// predicates collapse onto the same kind; the runtime treats index math
/// as signed.
static std::optional<unsigned> cmpTypeCodeFor(arith::CmpIPredicate p) {
  switch (p) {
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

static std::optional<unsigned> cmpTypeCodeFor(arith::CmpFPredicate p) {
  switch (p) {
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
      // NaN-sensitive unordered predicates have no infix counterpart.
      return std::nullopt;
  }
}

/// Rewrites a scalar `arith.cmpi`/`arith.cmpf` (kept by stage one) onto
/// the infix comparison pseudo-callees: `pl.cmp` has no scalar overload.
/// Tile comparisons were rewritten to `pypto.cmp` by stage one and lower
/// through CmpOpPattern instead.
template <typename SourceOp>
class ArithCmpPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto code = cmpTypeCodeFor(op.getPredicate());
    if (!code) {
      return rewriter.notifyMatchFailure(op, "comparison predicate has no infix counterpart");
    }
    StringRef callee = scalarCmpCalleeFor(*code);
    if (callee.empty()) {
      return rewriter.notifyMatchFailure(op, "unknown comparison kind");
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, op->getResult(0).getType(), callee, op->getOperands());
    return success();
  }
};

/// Rewrites the index constants kept by stage one (they feed
/// `emitpy.subscript` etc. and stay index-typed) into `emitpy.literal`.
class IndexConstantOpPattern : public OpConversionPattern<arith::ConstantOp> {
 public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<IndexType>(op.getType())) {
      return rewriter.notifyMatchFailure(op, "not an index constant");
    }
    auto intAttr = dyn_cast<IntegerAttr>(op.getValue());
    if (!intAttr) {
      return rewriter.notifyMatchFailure(op, "unsupported index constant");
    }
    auto text = intToPythonText(intAttr.getValue());
    if (!text) {
      return rewriter.notifyMatchFailure(op, "unsupported index constant");
    }
    rewriter.replaceOpWithNewOp<emitpy::LiteralOp>(op, rewriter.getIndexType(), *text);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Structured control flow
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.spmd` into `emitpy.spmd_scope` one-to-one: stage one
/// already resolved the multi-core grid (core count, body, block index),
/// so this stage only retargets the op onto its emitpy spelling.
class SpmdOpPattern : public OpConversionPattern<pypto::SpmdOp> {
 public:
  using OpConversionPattern<pypto::SpmdOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::SpmdOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto spmd = rewriter.create<emitpy::SpmdScopeOp>(op.getLoc(), op.getCoreNum());
    rewriter.inlineRegionBefore(op.getBody(), spmd.getBody(), spmd.getBody().end());
    rewriter.replaceOp(op, spmd);
    return success();
  }
};

/// Rewrites `pypto.range` into `emitpy.for` (emitted as
/// `for <iv> in pl.range(lb, ub, step)`): the per-core tile loop. Stage
/// one already resolved the loop, so this is a direct retargeting.
class RangeOpPattern : public OpConversionPattern<pypto::RangeOp> {
 public:
  using OpConversionPattern<pypto::RangeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::RangeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loop =
      rewriter.create<emitpy::ForOp>(op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep());
    rewriter.inlineRegionBefore(op.getBody(), loop.getBody(), loop.getBody().end());
    rewriter.replaceOp(op, loop);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant and allocation producers
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.constant` into `pl.const(value, dtype)`: the value is
/// the Python literal text; the dtype keeps scalars from defaulting to
/// Python int/float64 semantics.
class ConstantOpPattern : public OpConversionPattern<pypto::ConstantOp> {
 public:
  using OpConversionPattern<pypto::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "constant type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue()),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.const", ValueRange{},
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.tile_full` into `pl.tile.full(shape, dtype, value)`:
/// a statically shaped result carries the shape in its type; a dynamic
/// result carries one extent operand per dimension (constants fold into
/// the list literal, runtime bounds become operand references, exactly
/// like offsets).
class TileFullOpPattern : public OpConversionPattern<pypto::TileFullOp> {
 public:
  using OpConversionPattern<pypto::TileFullOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::TileFullOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // The result keeps the source type during the staged migration, so it
    // is either a `!pypto.tile<...>` (TileType is not a ShapedType) or
    // the original vector/npuvector type; both spell the element type
    // and, when static, the shape.
    Type sourceType = op.getResult().getType();
    Type elementType;
    SmallVector<int64_t> staticShape;
    bool hasStaticShape = false;
    if (auto tile = dyn_cast<pypto::TileType>(sourceType)) {
      elementType = tile.getElementType();
      if (!llvm::any_of(tile.getShape(), [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
        staticShape = SmallVector<int64_t>(tile.getShape());
        hasStaticShape = true;
      }
    } else if (auto shaped = dyn_cast<ShapedType>(sourceType)) {
      elementType = shaped.getElementType();
      if (shaped.hasStaticShape()) {
        staticShape = SmallVector<int64_t>(shaped.getShape());
        hasStaticShape = true;
      }
    } else {
      return rewriter.notifyMatchFailure(op, "tile_full result must be shaped");
    }
    auto dtype = pyptoDtypeName(elementType);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(sourceType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> fullOperands;
    Attribute shapeArg;
    if (adaptor.getOperands().empty()) {
      if (!hasStaticShape) {
        return rewriter.notifyMatchFailure(op, "tile_full without extents must be statically shaped");
      }
      shapeArg = rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(staticShape));
    } else {
      shapeArg = buildOffsetsArg(rewriter, adaptor.getOperands(), fullOperands);
    }
    SmallVector<Attribute> args{shapeArg, rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
                                rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue())};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.tile.full", ValueRange(fullOperands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.create_tile` into `pl.create_tile(shape, dtype=...,
/// target_memory=pl.MemorySpace.Vec)`: an uninitialized scratch tile in
/// vector-local memory.
class CreateTileOpPattern : public OpConversionPattern<pypto::CreateTileOp> {
 public:
  using OpConversionPattern<pypto::CreateTileOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::CreateTileOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto tile = llvm::cast<pypto::TileType>(op.getResult().getType());
    if (llvm::any_of(tile.getShape(), [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(op, "pl.create_tile requires a statically shaped result");
    }
    auto dtype = pyptoDtypeName(tile.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(tile);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(tile.getShape())),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
                                rewriter.getAttr<emitpy::OpaqueAttr>("pl.MemorySpace.Vec")};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("dtype"),
                                    rewriter.getStringAttr("target_memory")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.create_tile", ValueRange{},
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// View ops
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.slice` into `pl.slice(src, [window], [offsets],
/// drop_dims=[...])`: the window is full-rank over the source with unit
/// dims standing in for rank-reducing drops.
class SliceOpPattern : public OpConversionPattern<pypto::SliceOp> {
 public:
  using OpConversionPattern<pypto::SliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::SliceOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(op.getWindow())),
                                offsetsArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    if (auto dropDims = op.getDropDims()) {
      if (!dropDims->empty()) {
        args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(*dropDims)));
        keywords.push_back(rewriter.getStringAttr("drop_dims"));
      }
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.slice", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reshape` into `pl.reshape(src, [shape])`. Handles both
/// the spike tile form and the pipeline form, where the intermediate
/// layer keeps the source result type (vector/memref) until this stage.
/// Dynamically shaped results carry per-dimension `extents` (the op's own
/// model, mirroring `pypto.tile_full`); their shape list mixes literals
/// and operand references the same way load offsets do.
class ReshapeOpPattern : public OpConversionPattern<pypto::ReshapeOp> {
 public:
  using OpConversionPattern<pypto::ReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> shape;
    if (auto tile = dyn_cast<pypto::TileType>(op.getResult().getType())) {
      shape = llvm::to_vector(tile.getShape());
    } else if (auto shaped = dyn_cast<ShapedType>(op.getResult().getType())) {
      shape = llvm::to_vector(shaped.getShape());
    } else {
      return rewriter.notifyMatchFailure(op, "reshape result must be a shaped type");
    }
    bool dynamic = llvm::any_of(shape, [](int64_t dim) { return ShapedType::isDynamic(dim); });
    if (dynamic && adaptor.getExtents().size() != shape.size()) {
      return rewriter.notifyMatchFailure(op, "dynamic reshape needs one extent per result dimension");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute shapeArg;
    if (!dynamic) {
      shapeArg = rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(shape));
    } else {
      // The identity fallback keeps the npuvector/memref spelling; a `?`
      // extent cannot be spelled as an annotation, and the Python
      // translator accepts the source type for call_opaque results (as it
      // does for dynamic loads).
      shapeArg = buildOffsetsArg(rewriter, adaptor.getExtents(), operands);
    }
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), shapeArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.reshape", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.transpose` into `pl.transpose(src, i, j)`.
class TransposeOpPattern : public OpConversionPattern<pypto::TransposeOp> {
 public:
  using OpConversionPattern<pypto::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getI())),
                                rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getJ()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.transpose", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reinterpret_view` into
/// `pl.reinterpret_view(src, dtype)`: a bit-level reinterpretation of the
/// payload.
class ReinterpretViewOpPattern : public OpConversionPattern<pypto::ReinterpretViewOp> {
 public:
  using OpConversionPattern<pypto::ReinterpretViewOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::ReinterpretViewOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "view type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.reinterpret_view", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.cast` into `pl.cast(src, dtype)`.
class CastOpPattern : public OpConversionPattern<pypto::CastOp> {
 public:
  using OpConversionPattern<pypto::CastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op, "cast target has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.cast", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.cmp` into `pl.cmp(lhs, rhs, cmp_type=code)`: the enum
/// values already match the PyPTO comparison codes. A scalar rhs lowers
/// onto the tile-scalar `pl.tile.cmps` form instead; a constant scalar
/// rides inline as `pl.const(...)` text (the raw `arith.constant` has no
/// emitpy printer), while a runtime scalar stays an operand reference.
class CmpOpPattern : public OpConversionPattern<pypto::CmpOp> {
 public:
  using OpConversionPattern<pypto::CmpOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::CmpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    bool scalarRhs = !isa<ShapedType, pypto::TileType>(op.getRhs().getType());
    SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Attribute> args{
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(1),
      rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(static_cast<uint32_t>(op.getPredicate())))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("cmp_type")};
    if (scalarRhs) {
      if (auto text = constScalarText(op.getRhs())) {
        args[1] = rewriter.getAttr<emitpy::OpaqueAttr>(*text);
        operands.pop_back();
      }
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, scalarRhs ? "pl.tile.cmps" : "pl.cmp",
                                                      ValueRange(operands), rewriter.getArrayAttr(args),
                                                      rewriter.getArrayAttr(keywords));
    // A constant consumed inline as text leaves the raw arith.constant
    // dead; drop it so the translator never has to print it.
    if (auto constOp = op.getRhs().getDefiningOp<arith::ConstantOp>(); constOp && constOp->use_empty()) {
      rewriter.eraseOp(constOp);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reduction and scalar IO
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.reduction` into `pl.row_<kind>(vector, scratch)` /
/// `pl.col_<kind>(vector, scratch)`: a keepdim in-place fold of the last
/// (row) or first (col) axis, working in the scratch tile.
class ReductionOpPattern : public OpConversionPattern<pypto::ReductionOp> {
 public:
  using OpConversionPattern<pypto::ReductionOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::ReductionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    StringRef prefix = reduceAxisPrefix(op.getAxis());
    StringRef suffix = reduceKindSuffix(op.getKind());
    if (prefix.empty() || suffix.empty()) {
      return rewriter.notifyMatchFailure(op, "reduction kind/axis has no PyPTO counterpart");
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    std::string callee = "pl." + prefix.str() + suffix.str();
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, callee,
                                                      ValueRange{adaptor.getVector(), adaptor.getScratch()});
    return success();
  }
};

/// Rewrites `pypto.read` into `pl.read(src, [indices])`: the constant
/// coordinates render as a Python index list.
class ReadOpPattern : public OpConversionPattern<pypto::ReadOp> {
 public:
  using OpConversionPattern<pypto::ReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::ReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(op.getIndices()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.read", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.write` into `pl.write(dest, [offsets], value)`.
class WriteOpPattern : public OpConversionPattern<pypto::WriteOp> {
 public:
  using OpConversionPattern<pypto::WriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::WriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getDest(), adaptor.getValue()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(op.getLoc(), TypeRange{}, "pl.write", ValueRange(operands),
                                          rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile movement between tensor memory and cores
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.load` into `pl.load`. Statically shaped results emit
/// the 3-argument form `pl.load(src, [offsets], [shape])`; dynamic loads
/// emit the 4-argument form `pl.load(src, [offsets], [shapes], [valid])`
/// with the per-dimension allocation bounds and valid extents the op
/// carries. The dispatch follows the op's own model (paddedShapes
/// present), not the result type: pipeline-form loads keep the source
/// result type (vector/npuvector), only spike IR carries tile types.
class LoadOpPattern : public OpConversionPattern<pypto::LoadOp> {
 public:
  using OpConversionPattern<pypto::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getPaddedShapes().empty()) {
      return rewriteStaticRead(op, adaptor, rewriter);
    }
    return rewriteDynamicRead(op, adaptor, rewriter);
  }

 private:
  LogicalResult rewriteStaticRead(pypto::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    SmallVector<int64_t> shape;
    if (auto tile = dyn_cast<pypto::TileType>(op.getResult().getType())) {
      shape = llvm::to_vector(tile.getShape());
    } else if (auto shaped = dyn_cast<ShapedType>(op.getResult().getType())) {
      shape = llvm::to_vector(shaped.getShape());
    } else {
      return rewriter.notifyMatchFailure(op, "load result must be a shaped type");
    }
    if (llvm::any_of(shape, [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(op, "static load result must have a static shape");
    }
    // A rank-reduced load (source rank > tile rank) drops leading source
    // dimensions; the offsets still address every source dimension, and
    // pl.load requires offsets and shapes to share the source's coordinate
    // system -- pad the tile's shape with leading ones up to the source
    // rank.
    if (auto srcType = dyn_cast<ShapedType>(op.getSource().getType())) {
      if (srcType.getRank() > shape.size()) {
        shape.insert(shape.begin(), srcType.getRank() - shape.size(), 1);
      }
    }
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "tile element type has no PyPTO dtype");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg,
                                rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(shape))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, "pl.load", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }

  LogicalResult rewriteDynamicRead(pypto::LoadOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {
    ValueRange paddedShapes = adaptor.getPaddedShapes();
    ValueRange validExtents = adaptor.getValidExtents();
    if (validExtents.size() != paddedShapes.size()) {
      return rewriter.notifyMatchFailure(op, "dynamic load needs per-dimension padded/valid extents");
    }
    // A `?` extent cannot be spelled as an annotation; the call keeps the
    // source result type, which the Python translator accepts for
    // call_opaque results (as it does for dynamic vector tiles).
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    Attribute shapesArg = buildOffsetsArg(rewriter, paddedShapes, operands);
    Attribute validArg = buildOffsetsArg(rewriter, validExtents, operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg, shapesArg, validArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, op.getResult().getType(), "pl.load", ValueRange(operands),
                                                      rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.store` into `pl.store(t, [o...], dest)`; offsets may
/// be loop-dependent.
class StoreOpPattern : public OpConversionPattern<pypto::StoreOp> {
 public:
  using OpConversionPattern<pypto::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(pypto::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getValue(), adaptor.getDest()};
    Attribute offsetsArg = buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""), rewriter.getStringAttr(""), rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(op.getLoc(), TypeRange{}, "pl.store", ValueRange(operands),
                                          rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Kernel finalization (pipeline form)
//===----------------------------------------------------------------------===//

/// The emitpy.out_args attribute set by the vector-to-pypto stage: the
/// indices of parameters used in the PyPTO out-parameter model.
static constexpr StringLiteral kOutArgsAttr = "emitpy.out_args";

/// The emitpy.pypto_kernel attribute set by the vector-to-pypto stage: it
/// marks a function lowered through the two-stage pipeline, as opposed to
/// hand-written spike IR.
static constexpr StringLiteral kPyptoKernelAttr = "emitpy.pypto_kernel";

/// Erases all dead operations, including those made dead by erasing others.
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

/// Loads a tile parameter into the local (core) view: the entry block
/// argument becomes a `pl.load` of the annotated parameter.
static void loadTileParameter(OpBuilder &builder, BlockArgument arg, ShapedType shaped) {
  SmallVector<Attribute> loadArgs{
    builder.getIndexAttr(0),
    builder.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(SmallVector<int64_t>(shaped.getRank(), 0))),
    builder.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(shaped.getShape()))};
  SmallVector<Attribute> loadKeywords{builder.getStringAttr(""), builder.getStringAttr(""), builder.getStringAttr("")};
  SmallVector<Value> loadOperands{arg};
  Value load = builder
                 .create<emitpy::CallOpaqueOp>(arg.getLoc(), shaped, "pl.load", ValueRange(loadOperands),
                                               builder.getArrayAttr(loadArgs), builder.getArrayAttr(loadKeywords))
                 .getResult(0);
  for (OpOperand &use : llvm::make_early_inc_range(arg.getUses())) {
    if (use.getOwner() == load.getDefiningOp()) {
      continue;
    }
    use.set(load);
  }
}

/// Rewrites the kernel signature into the PyPTO out-parameter model:
/// every parameter becomes an opaque annotation, vector results are
/// turned into out parameters (already folded by the vector-to-pypto
/// stage), and tile parameters get an explicit `pl.load` prologue.
static LogicalResult rewriteSignature(func::FuncOp funcOp, MLIRContext *ctx) {
  Block &entry = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
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
          shaped.getNumElements() * (shaped.getElementType().getIntOrFloatBitWidth() / 8) % 32 == 0) {
        annotation =
          pyptoShapedOpaque(ctx, SmallVector<int64_t>{1, shaped.getDimSize(0)}, shaped.getElementType(), "pl.Tensor");
      }
    }
    if (!annotation) {
      annotation = pyptoAnnotation(ctx, type);
    }
    if (!annotation) {
      return funcOp->emitError("parameter ") << idx << " has no PyPTO annotation";
    }
    newInputs.push_back(annotation);
    BlockArgument arg = entry.getArgument(idx);
    arg.setType(annotation);
    if ((isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type)) && !arg.use_empty()) {
      loadTileParameter(builder, arg, cast<ShapedType>(type));
    }
  }
  funcOp.setType(FunctionType::get(ctx, newInputs, {}));
  return success();
}

/// Strips the hacc.* bookkeeping attributes the pipeline attached to the
/// kernel and its parameters.
static void stripHaccAttrs(func::FuncOp funcOp) {
  auto isHaccAttr = [](NamedAttribute attr) { return attr.getName().strref().starts_with("hacc."); };
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

/// Registers the pypto arithmetic, comparison and select ops that lower to
/// emitpy.call_opaque "pl.*" helper calls.
static void populateCallOpaquePatterns(RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<OpToCallOpaquePattern<pypto::AddOp>>(typeConverter, patterns.getContext(), "pl.add");
  patterns.add<OpToCallOpaquePattern<pypto::SubOp>>(typeConverter, patterns.getContext(), "pl.sub");
  patterns.add<OpToCallOpaquePattern<pypto::MulOp>>(typeConverter, patterns.getContext(), "pl.mul");
  patterns.add<OpToCallOpaquePattern<pypto::DivOp>>(typeConverter, patterns.getContext(), "pl.div");
  // Row-broadcast binaries: the [rows, 1] row vector operand broadcasts
  // across the tile's columns at runtime.
  patterns.add<OpToCallOpaquePattern<pypto::RowExpandAddOp>>(typeConverter, patterns.getContext(), "pl.row_expand_add");
  patterns.add<OpToCallOpaquePattern<pypto::RowExpandSubOp>>(typeConverter, patterns.getContext(), "pl.row_expand_sub");
  patterns.add<OpToCallOpaquePattern<pypto::RowExpandMulOp>>(typeConverter, patterns.getContext(), "pl.row_expand_mul");
  patterns.add<OpToCallOpaquePattern<pypto::RowExpandDivOp>>(typeConverter, patterns.getContext(), "pl.row_expand_div");
  patterns.add<OpToCallOpaquePattern<pypto::FmodOp>>(typeConverter, patterns.getContext(), "pl.fmod");
  patterns.add<OpToCallOpaquePattern<pypto::MaximumOp>>(typeConverter, patterns.getContext(), "pl.maximum");
  patterns.add<OpToCallOpaquePattern<pypto::MinimumOp>>(typeConverter, patterns.getContext(), "pl.minimum");
  patterns.add<OpToCallOpaquePattern<pypto::AndOp>>(typeConverter, patterns.getContext(), "pl.and_");
  patterns.add<OpToCallOpaquePattern<pypto::OrOp>>(typeConverter, patterns.getContext(), "pl.or_");
  patterns.add<OpToCallOpaquePattern<pypto::XorOp>>(typeConverter, patterns.getContext(), "pl.xor");
  patterns.add<OpToCallOpaquePattern<pypto::ShlOp>>(typeConverter, patterns.getContext(), "pl.shl");
  patterns.add<OpToCallOpaquePattern<pypto::ShrOp>>(typeConverter, patterns.getContext(), "pl.shr");
  patterns.add<OpToCallOpaquePattern<pypto::NegOp>>(typeConverter, patterns.getContext(), "pl.neg");
  patterns.add<OpToCallOpaquePattern<pypto::AbsOp>>(typeConverter, patterns.getContext(), "pl.abs");
  patterns.add<OpToCallOpaquePattern<pypto::ExpOp>>(typeConverter, patterns.getContext(), "pl.exp");
  patterns.add<OpToCallOpaquePattern<pypto::LogOp>>(typeConverter, patterns.getContext(), "pl.log");
  patterns.add<OpToCallOpaquePattern<pypto::SqrtOp>>(typeConverter, patterns.getContext(), "pl.sqrt");
  patterns.add<OpToCallOpaquePattern<pypto::RsqrtOp>>(typeConverter, patterns.getContext(), "pl.rsqrt");
  patterns.add<OpToCallOpaquePattern<pypto::SinOp>>(typeConverter, patterns.getContext(), "pl.sin");
  patterns.add<OpToCallOpaquePattern<pypto::CosOp>>(typeConverter, patterns.getContext(), "pl.cos");
  patterns.add<OpToCallOpaquePattern<pypto::MaxOp>>(typeConverter, patterns.getContext(), "pl.max");
  patterns.add<OpToCallOpaquePattern<pypto::MinOp>>(typeConverter, patterns.getContext(), "pl.min");
  patterns.add<SelectOpPattern>(typeConverter, patterns.getContext());
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertPyptoToEmitPyPass : impl::ConvertPyptoToEmitPyBase<ConvertPyptoToEmitPyPass> {
  using impl::ConvertPyptoToEmitPyBase<ConvertPyptoToEmitPyPass>::ConvertPyptoToEmitPyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    PyptoToEmitPyTypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addIllegalDialect<pypto::PyptoDialect>();
    target.addLegalDialect<emitpy::EmitPyDialect>();
    // Stage-one survivors lowered here: the scalar arith arithmetic kept
    // by stage one (pypto is a tile dialect) onto infix pseudo-callees
    // named after the arith op itself (addi, floordivsi, select, ...),
    // and the index constants feeding subscripts and
    // `emitpy.for` bounds onto `emitpy.literal`. Stage one already
    // rewrote all control flow into `pypto.spmd`/`pypto.range`, so this
    // pass is a straight pypto -> emitpy retargeting plus the scalar
    // arith lowering; any raw tile arith/scf reaching it means an input
    // stage one never saw, and fails loudly here.
    target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) { return !isa<IndexType>(op.getType()); });
    target.addIllegalOp<arith::CmpIOp, arith::CmpFOp, arith::SelectOp>();
    target.addIllegalDialect<scf::SCFDialect>();
    // Tile-typed block arguments must be legalized together with their
    // signature, otherwise the conversion leaves dangling materializations.
    target.addDynamicallyLegalOp<func::FuncOp>(
      [&](func::FuncOp op) { return typeConverter.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ConstantOpPattern, TileFullOpPattern, CreateTileOpPattern, SliceOpPattern, ReshapeOpPattern,
                 TransposeOpPattern, ReinterpretViewOpPattern, CastOpPattern, CmpOpPattern, ReductionOpPattern,
                 ReadOpPattern, WriteOpPattern, LoadOpPattern, StoreOpPattern>(typeConverter, patterns.getContext());
    patterns.add<IndexConstantOpPattern, SpmdOpPattern, RangeOpPattern>(typeConverter, patterns.getContext());
    // Scalar comparisons kept by stage one lower onto the eq/ne/lt/le/gt/ge
    // pseudo-callees (`pl.cmp` has no scalar overload).
    patterns.add<ArithCmpPattern<arith::CmpIOp>, ArithCmpPattern<arith::CmpFOp>>(typeConverter, patterns.getContext());
    // Scalar arithmetic kept by stage one lowers onto call_opaque ops
    // named after the arith op itself (addi, subf, divsi, floordivsi,
    // remsi, select, ...), which the translator prints as Python
    // expressions. The DSL's `/` maps integer operands onto divi-style
    // truncated division (result stays integer) and floats onto divf.
    // Scalar integer remainders survive stage one as arith ops
    // (`pl.fmod` is tile-only).
    addScalarArithPatterns<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
                           arith::DivFOp, arith::DivSIOp, arith::SelectOp, arith::FloorDivSIOp, arith::RemSIOp,
                           arith::RemUIOp>(patterns, typeConverter);
    populateCallOpaquePatterns(patterns, typeConverter);

    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Pipeline form: the vector-to-pypto stage marks the kernel and
    // normalizes the I/O to the PyPTO out-parameter model; finish the
    // signature rewrite and cleanup. Hand-written spike IR carries no
    // marker and is emitted as-is.
    if (funcOp->getAttr(kPyptoKernelAttr)) {
      eraseDeadOps(funcOp);
      if (failed(rewriteSignature(funcOp, &getContext()))) {
        signalPassFailure();
        return;
      }
      stripHaccAttrs(funcOp);
    }
  }
};

}  // namespace mlir

std::unique_ptr<Pass> mlir::createPyptoToEmitPyConversionPass() { return std::make_unique<ConvertPyptoToEmitPyPass>(); }
