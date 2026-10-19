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

#include <optional>
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

/// Builds the `[i0, i1, ...]` offsets argument for pl.load/pl.store.
/// Constant-only offsets render directly as an opaque list literal; dynamic
/// offsets become operand references (IntegerAttr of index type) and the
/// index values are appended to `operands`, matching the call_opaque
/// argument protocol used by the vector-to-emitpy lowering.
static Attribute buildOffsetsArg(ConversionPatternRewriter &rewriter,
                                 ValueRange offsets,
                                 SmallVectorImpl<Value> &operands) {
  bool allConstant = llvm::all_of(
      offsets, [](Value offset) { return constIndexValue(offset).has_value(); });
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
      if (llvm::any_of(type.getShape(), [](int64_t dim) {
            return ShapedType::isDynamic(dim);
          })) {
        return type;
      }
      auto dtype = pyptoDtypeName(type.getElementType());
      if (!dtype) {
        return std::nullopt;
      }
      std::string text;
      llvm::raw_string_ostream rso(text);
      rso << "pl.Tile[" << shapeToPythonListText(type.getShape()) << ", "
          << *dtype << "]";
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
/// Covers the elementwise binaries/unaries, the scalar combines and
/// `pypto.select` (whose scratch tile is an explicit operand, unlike the
/// vector-side select that has to materialize one).
template <typename SourceOp>
class OpToCallOpaquePattern : public OpConversionPattern<SourceOp> {
 public:
  OpToCallOpaquePattern(TypeConverter &typeConverter, MLIRContext *context,
                        StringRef callee)
      : OpConversionPattern<SourceOp>(typeConverter, context), callee(callee) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(op, resultType, callee,
                                                      adaptor.getOperands());
    return success();
  }

 private:
  StringRef callee;
};

//===----------------------------------------------------------------------===//
// Stage-one survivors: scalar arith forms and scf control flow
//===----------------------------------------------------------------------===//

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
/// accepts Tensor/Tile operands only, so the scalar comparisons kept by
/// stage one (e.g. the cmp/div/select chains of the built-in affine
/// floordiv lowering) go through these pseudo-callees, which the Python
/// translator prints as infix comparisons that the PyPTO parser maps onto
/// ir.lt etc.
static StringRef scalarCmpCalleeFor(unsigned cmpTypeCode) {
  static const StringRef callees[] = {"util_cmp_eq", "util_cmp_ne",
                                      "util_cmp_lt", "util_cmp_le",
                                      "util_cmp_gt", "util_cmp_ge"};
  return cmpTypeCode < std::size(callees) ? callees[cmpTypeCode] : StringRef();
}

/// Rewrites the scalar comparisons kept by stage one onto the infix
/// comparison pseudo-callees: `pl.cmp` has no scalar overload. Tile
/// comparisons must already be `pypto.cmp` by then.
template <typename SourceOp>
class ScalarCmpPattern : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (llvm::any_of(op->getOperands(), [](Value operand) {
          return isa<VectorType, npuvector::NPUVectorType>(operand.getType());
        })) {
      return rewriter.notifyMatchFailure(
          op, "tile comparisons must be pypto.cmp by stage one");
    }
    auto code = cmpTypeCodeFor(op.getPredicate());
    if (!code) {
      return rewriter.notifyMatchFailure(
          op, "comparison predicate has no util_cmp counterpart");
    }
    // Raw operands: values already rewritten (e.g. index constants turned
    // into emitpy.literal) resolve to their replacements.
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResult(0).getType(), scalarCmpCalleeFor(*code),
        op->getOperands());
    return success();
  }
};

/// Rewrites the scalar selects kept by stage one onto the `util_select`
/// pseudo-callee, which the translator prints as a Python conditional
/// expression. Operand order is (condition, trueValue, falseValue) in both
/// the IR and the pseudo-callee; the translator reorders for Python
/// syntax. Tile selects must already be `pypto.select` by then.
class ScalarSelectPattern : public OpConversionPattern<arith::SelectOp> {
 public:
  using OpConversionPattern<arith::SelectOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op->getResult(0).getType())) {
      return rewriter.notifyMatchFailure(
          op, "tile selects must be pypto.select by stage one");
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op->getResult(0).getType(), "util_select", op->getOperands());
    return success();
  }
};

/// Rewrites the index constants kept by stage one (they feed
/// `emitpy.subscript` etc. and stay index-typed) into `emitpy.literal`.
class IndexConstantOpPattern : public OpConversionPattern<arith::ConstantOp> {
 public:
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
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
    rewriter.replaceOpWithNewOp<emitpy::LiteralOp>(
        op, rewriter.getIndexType(), *text);
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
// Constant and allocation producers
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.constant` into `pl.const(value, dtype)`: the value is
/// the Python literal text; the dtype keeps scalars from defaulting to
/// Python int/float64 semantics.
class ConstantOpPattern : public OpConversionPattern<pypto::ConstantOp> {
 public:
  using OpConversionPattern<pypto::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "constant type has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue()),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.const", ValueRange{}, rewriter.getArrayAttr(args),
        rewriter.getArrayAttr(keywords));
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

  LogicalResult
  matchAndRewrite(pypto::TileFullOp op, OpAdaptor adaptor,
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
      if (!llvm::any_of(tile.getShape(), [](int64_t dim) {
            return ShapedType::isDynamic(dim);
          })) {
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
      return rewriter.notifyMatchFailure(op,
                                         "tile_full result must be shaped");
    }
    auto dtype = pyptoDtypeName(elementType);
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(sourceType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> fullOperands;
    Attribute shapeArg;
    if (adaptor.getOperands().empty()) {
      if (!hasStaticShape) {
        return rewriter.notifyMatchFailure(
            op, "tile_full without extents must be statically shaped");
      }
      shapeArg = rewriter.getAttr<emitpy::OpaqueAttr>(
          shapeToPythonListText(staticShape));
    } else {
      shapeArg =
          buildOffsetsArg(rewriter, adaptor.getOperands(), fullOperands);
    }
    SmallVector<Attribute> args{
        shapeArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
        rewriter.getAttr<emitpy::OpaqueAttr>(op.getValue())};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.tile.full", ValueRange(fullOperands),
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

  LogicalResult
  matchAndRewrite(pypto::CreateTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = llvm::cast<pypto::TileType>(op.getResult().getType());
    if (llvm::any_of(tile.getShape(),
                     [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(
          op, "pl.create_tile requires a statically shaped result");
    }
    auto dtype = pyptoDtypeName(tile.getElementType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "element type has no PyPTO dtype");
    }
    Type resultType = getTypeConverter()->convertType(tile);
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Attribute> args{
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(tile.getShape())),
        rewriter.getAttr<emitpy::OpaqueAttr>(*dtype),
        rewriter.getAttr<emitpy::OpaqueAttr>("pl.MemorySpace.Vec")};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("dtype"),
                                    rewriter.getStringAttr("target_memory")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.create_tile", ValueRange{},
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

  LogicalResult
  matchAndRewrite(pypto::SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getWindow())),
        offsetsArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    if (auto dropDims = op.getDropDims()) {
      if (!dropDims->empty()) {
        args.push_back(rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(*dropDims)));
        keywords.push_back(rewriter.getStringAttr("drop_dims"));
      }
    }
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.slice", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reshape` into `pl.reshape(src, [shape])`. Handles both
/// the spike tile form and the pipeline form, where the intermediate
/// layer keeps the source result type (vector/memref) until this stage.
class ReshapeOpPattern : public OpConversionPattern<pypto::ReshapeOp> {
 public:
  using OpConversionPattern<pypto::ReshapeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> shape;
    if (auto tile = dyn_cast<pypto::TileType>(op.getResult().getType())) {
      shape = llvm::to_vector(tile.getShape());
    } else if (auto shaped =
                   dyn_cast<ShapedType>(op.getResult().getType())) {
      shape = llvm::to_vector(shaped.getShape());
    } else {
      return rewriter.notifyMatchFailure(
          op, "reshape result must be a shaped type");
    }
    if (llvm::any_of(shape,
                     [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(
          op, "reshape result must be statically shaped");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(shape))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.reshape", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.transpose` into `pl.transpose(src, i, j)`.
class TransposeOpPattern : public OpConversionPattern<pypto::TransposeOp> {
 public:
  using OpConversionPattern<pypto::TransposeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getI())),
        rewriter.getAttr<emitpy::OpaqueAttr>(std::to_string(op.getJ()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.transpose", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.reinterpret_view` into
/// `pl.reinterpret_view(src, dtype)`: a bit-level reinterpretation of the
/// payload.
class ReinterpretViewOpPattern
    : public OpConversionPattern<pypto::ReinterpretViewOp> {
 public:
  using OpConversionPattern<
      pypto::ReinterpretViewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReinterpretViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "view type has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.reinterpret_view", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.cast` into `pl.cast(src, dtype)`.
class CastOpPattern : public OpConversionPattern<pypto::CastOp> {
 public:
  using OpConversionPattern<pypto::CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dtype = resultDtypeName(op.getResult().getType());
    if (!dtype) {
      return rewriter.notifyMatchFailure(op,
                                         "cast target has no PyPTO dtype");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{rewriter.getIndexAttr(0),
                                rewriter.getAttr<emitpy::OpaqueAttr>(*dtype)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.cast", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

/// Rewrites `pypto.cmp` into `pl.cmp(lhs, rhs, cmp_type=code)`: the enum
/// values already match the PyPTO comparison codes.
class CmpOpPattern : public OpConversionPattern<pypto::CmpOp> {
 public:
  using OpConversionPattern<pypto::CmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getLhs(), adaptor.getRhs()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getIndexAttr(1),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            std::to_string(static_cast<uint32_t>(op.getPredicate())))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("cmp_type")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.cmp", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
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

  LogicalResult
  matchAndRewrite(pypto::ReductionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef prefix = reduceAxisPrefix(op.getAxis());
    StringRef suffix = reduceKindSuffix(op.getKind());
    if (prefix.empty() || suffix.empty()) {
      return rewriter.notifyMatchFailure(
          op, "reduction kind/axis has no PyPTO counterpart");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    std::string callee = "pl." + prefix.str() + suffix.str();
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, callee,
        ValueRange{adaptor.getVector(), adaptor.getScratch()});
    return success();
  }
};

/// Rewrites `pypto.read` into `pl.read(src, [indices])`: the constant
/// coordinates render as a Python index list.
class ReadOpPattern : public OpConversionPattern<pypto::ReadOp> {
 public:
  using OpConversionPattern<pypto::ReadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::ReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "result type has no emitpy counterpart");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        rewriter.getAttr<emitpy::OpaqueAttr>(
            shapeToPythonListText(op.getIndices()))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.read", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.write` into `pl.write(dest, [offsets], value)`.
class WriteOpPattern : public OpConversionPattern<pypto::WriteOp> {
 public:
  using OpConversionPattern<pypto::WriteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::WriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getDest(), adaptor.getValue()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "pl.write", ValueRange(operands),
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

  LogicalResult
  matchAndRewrite(pypto::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getPaddedShapes().empty()) {
      return rewriteStaticRead(op, adaptor, rewriter);
    }
    return rewriteDynamicRead(op, adaptor, rewriter);
  }

 private:
  LogicalResult rewriteStaticRead(pypto::LoadOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const {
    SmallVector<int64_t> shape;
    if (auto tile = dyn_cast<pypto::TileType>(op.getResult().getType())) {
      shape = llvm::to_vector(tile.getShape());
    } else if (auto shaped =
                   dyn_cast<ShapedType>(op.getResult().getType())) {
      shape = llvm::to_vector(shaped.getShape());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "load result must be a shaped type");
    }
    if (llvm::any_of(shape,
                     [](int64_t dim) { return ShapedType::isDynamic(dim); })) {
      return rewriter.notifyMatchFailure(
          op, "static load result must have a static shape");
    }
    Type resultType =
        getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(
          op, "tile element type has no PyPTO dtype");
    }
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0),
        offsetsArg,
        rewriter.getAttr<emitpy::OpaqueAttr>(shapeToPythonListText(shape))};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, resultType, "pl.load", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }

  LogicalResult rewriteDynamicRead(pypto::LoadOp op, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    ValueRange paddedShapes = adaptor.getPaddedShapes();
    ValueRange validExtents = adaptor.getValidExtents();
    if (validExtents.size() != paddedShapes.size()) {
      return rewriter.notifyMatchFailure(
          op, "dynamic load needs per-dimension padded/valid extents");
    }
    // A `?` extent cannot be spelled as an annotation; the call keeps the
    // source result type, which the Python translator accepts for
    // call_opaque results (as it does for dynamic vector tiles).
    SmallVector<Value> operands{adaptor.getSource()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    Attribute shapesArg = buildOffsetsArg(rewriter, paddedShapes, operands);
    Attribute validArg = buildOffsetsArg(rewriter, validExtents, operands);
    SmallVector<Attribute> args{rewriter.getIndexAttr(0), offsetsArg,
                                shapesArg, validArg};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.replaceOpWithNewOp<emitpy::CallOpaqueOp>(
        op, op.getResult().getType(), "pl.load", ValueRange(operands),
        rewriter.getArrayAttr(args), rewriter.getArrayAttr(keywords));
    return success();
  }
};

/// Rewrites `pypto.store` into `pl.store(t, [o...], dest)`; offsets may
/// be loop-dependent.
class StoreOpPattern : public OpConversionPattern<pypto::StoreOp> {
 public:
  using OpConversionPattern<pypto::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(pypto::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands{adaptor.getValue(), adaptor.getDest()};
    Attribute offsetsArg =
        buildOffsetsArg(rewriter, adaptor.getOffsets(), operands);
    SmallVector<Attribute> args{
        rewriter.getIndexAttr(0), offsetsArg, rewriter.getIndexAttr(1)};
    SmallVector<Attribute> keywords{rewriter.getStringAttr(""),
                                    rewriter.getStringAttr(""),
                                    rewriter.getStringAttr("")};
    rewriter.create<emitpy::CallOpaqueOp>(
        op.getLoc(), TypeRange{}, "pl.store", ValueRange(operands),
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
          shaped.getNumElements() *
                  (shaped.getElementType().getIntOrFloatBitWidth() / 8) % 32 ==
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
      return funcOp->emitError("parameter ") << idx
                                             << " has no PyPTO annotation";
    }
    newInputs.push_back(annotation);
    BlockArgument arg = entry.getArgument(idx);
    arg.setType(annotation);
    if ((isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type)) &&
        !arg.use_empty()) {
      loadTileParameter(builder, arg, cast<ShapedType>(type));
    }
  }
  funcOp.setType(FunctionType::get(ctx, newInputs, {}));
  return success();
}

/// Strips the hacc.* bookkeeping attributes the pipeline attached to the
/// kernel and its parameters.
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

/// Registers the pypto arithmetic, comparison and select ops that lower to
/// emitpy.call_opaque "pl.*" helper calls.
static void populateCallOpaquePatterns(RewritePatternSet &patterns,
                                       TypeConverter &typeConverter) {
  patterns.add<OpToCallOpaquePattern<pypto::AddOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.add");
  patterns.add<OpToCallOpaquePattern<pypto::SubOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.sub");
  patterns.add<OpToCallOpaquePattern<pypto::MulOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.mul");
  patterns.add<OpToCallOpaquePattern<pypto::DivOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.div");
  patterns.add<OpToCallOpaquePattern<pypto::FmodOp>>(typeConverter,
                                                     patterns.getContext(),
                                                     "pl.fmod");
  patterns.add<OpToCallOpaquePattern<pypto::MaximumOp>>(typeConverter,
                                                        patterns.getContext(),
                                                        "pl.maximum");
  patterns.add<OpToCallOpaquePattern<pypto::MinimumOp>>(typeConverter,
                                                        patterns.getContext(),
                                                        "pl.minimum");
  patterns.add<OpToCallOpaquePattern<pypto::AndOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.and_");
  patterns.add<OpToCallOpaquePattern<pypto::OrOp>>(typeConverter,
                                                   patterns.getContext(),
                                                   "pl.or_");
  patterns.add<OpToCallOpaquePattern<pypto::XorOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.xor");
  patterns.add<OpToCallOpaquePattern<pypto::ShlOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.shl");
  patterns.add<OpToCallOpaquePattern<pypto::ShrOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.shr");
  patterns.add<OpToCallOpaquePattern<pypto::NegOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.neg");
  patterns.add<OpToCallOpaquePattern<pypto::AbsOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.abs");
  patterns.add<OpToCallOpaquePattern<pypto::ExpOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.exp");
  patterns.add<OpToCallOpaquePattern<pypto::LogOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.log");
  patterns.add<OpToCallOpaquePattern<pypto::SqrtOp>>(typeConverter,
                                                     patterns.getContext(),
                                                     "pl.sqrt");
  patterns.add<OpToCallOpaquePattern<pypto::RsqrtOp>>(typeConverter,
                                                      patterns.getContext(),
                                                      "pl.rsqrt");
  patterns.add<OpToCallOpaquePattern<pypto::SinOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.sin");
  patterns.add<OpToCallOpaquePattern<pypto::CosOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.cos");
  patterns.add<OpToCallOpaquePattern<pypto::MaxOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.max");
  patterns.add<OpToCallOpaquePattern<pypto::MinOp>>(typeConverter,
                                                    patterns.getContext(),
                                                    "pl.min");
  patterns.add<OpToCallOpaquePattern<pypto::SelectOp>>(typeConverter,
                                                       patterns.getContext(),
                                                       "pl.sel");
}

}  // namespace

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertPyptoToEmitPyPass
    : impl::ConvertPyptoToEmitPyBase<ConvertPyptoToEmitPyPass> {
  using impl::ConvertPyptoToEmitPyBase<
      ConvertPyptoToEmitPyPass>::ConvertPyptoToEmitPyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<emitpy::EmitPyDialect, func::FuncDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    PyptoToEmitPyTypeConverter typeConverter;
    ConversionTarget target(getContext());
    target.addIllegalDialect<pypto::PyptoDialect>();
    target.addLegalDialect<emitpy::EmitPyDialect>();
    // Stage-one survivors that this pass lowers onto the `util_*`
    // pseudo-callees and `emitpy.literal`: scalar compares and selects
    // (`pl.cmp`/`pl.sel` are tile-only), the floordivsi of the affine
    // floordiv chain, and index constants (tile compares still convert).
    // The dynamic predicates mirror stage one, keeping the tile forms that
    // never reach this pass legal so a mis-fed input fails loudly in the
    // pattern rather than being silently retained.
    target.addDynamicallyLegalOp<arith::CmpIOp, arith::CmpFOp>(
        [](Operation *op) {
          return llvm::any_of(op->getOperands(), [](Value operand) {
            return isa<VectorType, npuvector::NPUVectorType>(
                operand.getType());
          });
        });
    target.addDynamicallyLegalOp<arith::SelectOp>([](arith::SelectOp op) {
      return isa<ShapedType>(op->getResult(0).getType());
    });
    target.addDynamicallyLegalOp<arith::ConstantOp>(
        [](arith::ConstantOp op) { return !isa<IndexType>(op.getType()); });
    // Control flow: scf.parallel/scf.forall map onto SPMD launches and
    // scf.for onto the per-core tile loop; any other scf construct has no
    // PyPTO counterpart.
    target.addIllegalDialect<scf::SCFDialect>();
    // Tile-typed block arguments must be legalized together with their
    // signature, otherwise the conversion leaves dangling materializations.
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
      return typeConverter.isLegal(op);
    });

    RewritePatternSet patterns(&getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    patterns.add<ConstantOpPattern, TileFullOpPattern, CreateTileOpPattern,
                 SliceOpPattern, ReshapeOpPattern, TransposeOpPattern,
                 ReinterpretViewOpPattern, CastOpPattern, CmpOpPattern,
                 ReductionOpPattern, ReadOpPattern, WriteOpPattern,
                 LoadOpPattern, StoreOpPattern>(typeConverter,
                                                patterns.getContext());
    patterns.add<IndexConstantOpPattern, ScalarSelectPattern,
                 ScfParallelPattern, ScfForallPattern, ScfForPattern>(
        typeConverter, patterns.getContext());
    patterns.add<ScalarCmpPattern<arith::CmpIOp>,
                 ScalarCmpPattern<arith::CmpFOp>>(typeConverter,
                                                  patterns.getContext());
    patterns.add<OpToCallOpaquePattern<arith::FloorDivSIOp>>(
        typeConverter, patterns.getContext(), "util_floordiv");
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

std::unique_ptr<Pass> mlir::createPyptoToEmitPyConversionPass() {
  return std::make_unique<ConvertPyptoToEmitPyPass>();
}
