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

#include "mfusion/Conversion/Passes.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Dialect/Dvm/IR/Dvm.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#include <cmath>
#include <limits>

namespace mlir {
#define GEN_PASS_DEF_CONVERTMFUSETODVM
#include "mfusion/Conversion/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Macros for generating conversion patterns
//===----------------------------------------------------------------------===//

// Macro for unary operations: MfuseOp -> dvm::UnaryOp
// Uses getOperands() to handle different operand names
#define CONVERT_UNARY_OP(MfuseOp, DvmOpType)                                                      \
  struct Convert##MfuseOp : public OpConversionPattern<mfuse::MfuseOp> {                          \
    using OpConversionPattern::OpConversionPattern;                                               \
    LogicalResult matchAndRewrite(mfuse::MfuseOp op, OpAdaptor adaptor,                           \
                                  ConversionPatternRewriter &rewriter) const override {           \
      if (!isDvmOutlinedOp(op.getOperation())) {                                                  \
        return failure();                                                                         \
      }                                                                                           \
      auto operands = adaptor.getOperands();                                                      \
      auto attr = dvm::UnaryOpTypeAttr::get(getContext(), dvm::UnaryOpType::DvmOpType);           \
      rewriter.replaceOpWithNewOp<dvm::UnaryOp>(op, op.getResult().getType(), attr, operands[0]); \
      return success();                                                                           \
    }                                                                                             \
  };

static bool isDvmOutlinedFunc(func::FuncOp func) {
  if (!func || !func->hasAttr(mfusion_attrs::kOutlined)) {
    return false;
  }
  auto fusionTypeAttr = func->getAttrOfType<StringAttr>(mfusion_attrs::kFusionType);
  return fusionTypeAttr && fusionTypeAttr.getValue() == "dvm";
}

static bool isDvmOutlinedOp(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  return func && isDvmOutlinedFunc(func);
}

static bool hasScalarMarker(RankedTensorType type) {
  auto dictAttr = llvm::dyn_cast_or_null<DictionaryAttr>(type.getEncoding());
  return dictAttr && dictAttr.contains(mfuse::kScalarMarkerAttr);
}

static TypedAttr getZeroScalarAttr(Type elementType, OpBuilder &builder) {
  if (mlir::isa<FloatType>(elementType)) {
    return cast<TypedAttr>(builder.getFloatAttr(elementType, 0.0));
  }
  if (auto intType = mlir::dyn_cast<IntegerType>(elementType); intType && intType.getWidth() == 32) {
    return cast<TypedAttr>(builder.getIntegerAttr(elementType, 0));
  }
  return {};
}

static FailureOr<TypedAttr> extractTypedScalarAttr(DenseElementsAttr denseAttr, Type elementType, Location loc) {
  if (denseAttr.getElementType() != elementType) {
    return emitError(loc, "DVM scalar constant value element type ")
           << denseAttr.getElementType() << " does not match result element type " << elementType;
  }

  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    auto value = (*denseAttr.getValues<APInt>().begin());
    return cast<TypedAttr>(IntegerAttr::get(intType, value));
  }

  if (auto floatType = dyn_cast<FloatType>(elementType)) {
    APFloat value = (*denseAttr.getValues<APFloat>().begin());
    return cast<TypedAttr>(FloatAttr::get(floatType, value));
  }

  return emitError(loc, "unsupported DVM scalar constant element type: ") << elementType;
}

static FailureOr<TypedAttr> normalizeScalarConstantForDvmImpl(mfuse::ConstantOp op, RankedTensorType rankedType,
                                                              bool allowBool, StringRef errorContext) {
  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(op.getValue());
  if (!denseAttr) {
    return op->emitError() << errorContext << " scalar constant must be a DenseElementsAttr";
  }

  auto denseType = llvm::dyn_cast<RankedTensorType>(denseAttr.getType());
  if (!denseType || denseType.getRank() != 0 || denseAttr.getNumElements() != 1) {
    return op->emitError() << errorContext << " scalar constant must be a rank-0 tensor with one element";
  }

  auto elementType = rankedType.getElementType();
  if (elementType.isF32() || elementType.isF16() || elementType.isBF16() || elementType.isInteger(32)) {
    auto scalarAttr = extractTypedScalarAttr(denseAttr, elementType, op.getLoc());
    if (failed(scalarAttr)) return failure();
    return *scalarAttr;
  }

  if (allowBool && elementType.isInteger(1)) {
    bool value = (*denseAttr.getValues<APInt>().begin()).getBoolValue();
    auto i32Type = IntegerType::get(op.getContext(), 32);
    return cast<TypedAttr>(IntegerAttr::get(i32Type, value ? 1 : 0));
  }

  if (elementType.isF64()) {
    double value = (*denseAttr.getValues<APFloat>().begin()).convertToDouble();
    constexpr double kMaxFloat = static_cast<double>(std::numeric_limits<float>::max());
    if (!std::isfinite(value) || value < -kMaxFloat || value > kMaxFloat) {
      return op->emitError() << "cannot convert f64 scalar constant to f32 for " << errorContext
                             << ": value is not finite or is out of range";
    }
    auto f32Type = Float32Type::get(op.getContext());
    return cast<TypedAttr>(FloatAttr::get(f32Type, static_cast<float>(value)));
  }

  if (elementType.isInteger(64)) {
    int64_t value = (*denseAttr.getValues<APInt>().begin()).getSExtValue();
    if (value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max()) {
      return op->emitError() << "cannot convert i64 scalar constant to i32 for " << errorContext
                             << ": value is out of range";
    }
    auto i32Type = IntegerType::get(op.getContext(), 32);
    return cast<TypedAttr>(IntegerAttr::get(i32Type, static_cast<int32_t>(value)));
  }

  return op->emitError() << "unsupported " << errorContext << " scalar constant element type: " << elementType;
}

static FailureOr<TypedAttr> normalizeScalarConstantForDvm(mfuse::ConstantOp op, RankedTensorType rankedType) {
  // DVM binary scalar APIs support float, int32_t, Float16 and BFloat16.
  // double, int64_t and bool are normalized to f32/i32 when safe before
  // entering the DVM scalar ABI.
  return normalizeScalarConstantForDvmImpl(op, rankedType, /*allowBool=*/true, "DVM");
}

static FailureOr<dvm::DType> getDvmBroadcastDType(Type elementType, Location loc) {
  if (elementType.isInteger(1)) {
    return dvm::DType::Bool;
  }
  if (elementType.isF16()) {
    return dvm::DType::Float16;
  }
  if (elementType.isBF16()) {
    return dvm::DType::BFloat16;
  }
  if (elementType.isF32()) {
    return dvm::DType::Float32;
  }
  if (elementType.isInteger(32)) {
    return dvm::DType::Int32;
  }
  if (elementType.isInteger(64)) {
    return dvm::DType::Int64;
  }
  return emitError(loc, "unsupported DVM broadcast_scalar result element type: ") << elementType;
}

static FailureOr<TypedAttr> normalizeScalarConstantForDvmBroadcast(mfuse::ConstantOp op, RankedTensorType rankedType) {
  return normalizeScalarConstantForDvmImpl(op, rankedType, /*allowBool=*/true, "DVM broadcast");
}

static mfuse::ConstantOp getScalarConstant(Value value) {
  auto constantOp = value.getDefiningOp<mfuse::ConstantOp>();
  if (!constantOp) {
    return nullptr;
  }
  auto rankedType = dyn_cast<RankedTensorType>(constantOp.getResult().getType());
  if (!rankedType || !hasScalarMarker(rankedType)) {
    return nullptr;
  }
  return constantOp;
}

static LogicalResult convertBinaryOp(Operation *op, ValueRange originalOperands, ValueRange adaptorOperands,
                                     Type resultType, dvm::BinaryOpType dvmOpType,
                                     ConversionPatternRewriter &rewriter) {
  if (!isDvmOutlinedOp(op)) {
    return failure();
  }

  if (originalOperands.size() != 2 || adaptorOperands.size() != 2) {
    return op->emitError("DVM binary conversion expects exactly two operands");
  }

  auto lhsScalar = getScalarConstant(originalOperands[0]);
  auto rhsScalar = getScalarConstant(originalOperands[1]);
  auto attr = dvm::BinaryOpTypeAttr::get(op->getContext(), dvmOpType);

  if (lhsScalar && rhsScalar) {
    return op->emitError("DVM binary scalar conversion does not support scalar-scalar operands");
  }

  if (lhsScalar || rhsScalar) {
    auto scalarOp = lhsScalar ? lhsScalar : rhsScalar;
    auto rankedType = cast<RankedTensorType>(scalarOp.getResult().getType());
    auto normalized = normalizeScalarConstantForDvm(scalarOp, rankedType);
    if (failed(normalized)) return failure();

    bool scalarOnLhs = static_cast<bool>(lhsScalar);
    Value tensorOperand = lhsScalar ? adaptorOperands[1] : adaptorOperands[0];
    rewriter.replaceOpWithNewOp<dvm::BinaryScalarOp>(op, resultType, attr, tensorOperand, *normalized, scalarOnLhs);
    if (scalarOp.getResult().use_empty()) {
      rewriter.eraseOp(scalarOp);
    }
    return success();
  }

  rewriter.replaceOpWithNewOp<dvm::BinaryOp>(op, resultType, attr, adaptorOperands[0], adaptorOperands[1]);
  return success();
}

// Macro for binary operations: MfuseOp -> dvm::BinaryOp or dvm::BinaryScalarOp.
#define CONVERT_BINARY_OP(MfuseOp, DvmOpType)                                                                       \
  struct Convert##MfuseOp : public OpConversionPattern<mfuse::MfuseOp> {                                            \
    using OpConversionPattern::OpConversionPattern;                                                                 \
    LogicalResult matchAndRewrite(mfuse::MfuseOp op, OpAdaptor adaptor,                                             \
                                  ConversionPatternRewriter &rewriter) const override {                             \
      return convertBinaryOp(op.getOperation(), op->getOperands(), adaptor.getOperands(), op.getResult().getType(), \
                             dvm::BinaryOpType::DvmOpType, rewriter);                                               \
    }                                                                                                               \
  };

// Binary operations generated by macro
CONVERT_BINARY_OP(MulOp, Mul)
CONVERT_BINARY_OP(AddOp, Add)
CONVERT_BINARY_OP(SubOp, Sub)
CONVERT_BINARY_OP(DivOp, Div)
CONVERT_BINARY_OP(PowOp, Pow)
CONVERT_BINARY_OP(MaximumOp, Maximum)
CONVERT_BINARY_OP(MinimumOp, Minimum)
CONVERT_BINARY_OP(EqOp, Equal)
CONVERT_BINARY_OP(NeOp, NotEqual)
CONVERT_BINARY_OP(GtOp, Greater)
CONVERT_BINARY_OP(GeOp, GreaterEqual)
CONVERT_BINARY_OP(LtOp, Less)
CONVERT_BINARY_OP(LeOp, LessEqual)
CONVERT_BINARY_OP(LogicalAndOp, LogicalAnd)
CONVERT_BINARY_OP(LogicalOrOp, LogicalOr)
CONVERT_BINARY_OP(RealDivOp, Div)

// Unary operations generated by macro
CONVERT_UNARY_OP(AbsOp, Abs)
CONVERT_UNARY_OP(CeilOp, Ceil)
CONVERT_UNARY_OP(FloorOp, Floor)
CONVERT_UNARY_OP(TruncOp, Trunc)
CONVERT_UNARY_OP(ExpOp, Exp)
CONVERT_UNARY_OP(LogOp, Log)
CONVERT_UNARY_OP(ReciprocalOp, Reciprocal)
CONVERT_UNARY_OP(SqrtOp, Sqrt)
CONVERT_UNARY_OP(IsFiniteOp, IsFinite)
CONVERT_UNARY_OP(LogicalNotOp, LogicalNot)

// Hand-written conversion patterns for operations that need special handling

struct ConvertNegOp : public OpConversionPattern<mfuse::NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::NegOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    // Neg(x) -> Sub(0, x)
    auto input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto resultType = op.getResult().getType();

    auto zeroAttr = getZeroScalarAttr(elementType, rewriter);
    if (!zeroAttr) {
      return op->emitError("unsupported DVM scalar zero element type: ") << elementType;
    }

    auto subAttr = dvm::BinaryOpTypeAttr::get(getContext(), dvm::BinaryOpType::Sub);
    rewriter.replaceOpWithNewOp<dvm::BinaryScalarOp>(op, resultType, subAttr, input, zeroAttr,
                                                     /*scalarOnLhs=*/true);
    return success();
  }
};

struct ConvertRsqrtOp : public OpConversionPattern<mfuse::RsqrtOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::RsqrtOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    // Rsqrt(x) -> Reciprocal(Sqrt(x))
    auto sqrtAttr = dvm::UnaryOpTypeAttr::get(getContext(), dvm::UnaryOpType::Sqrt);
    auto sqrtResultType = op.getResult().getType();
    auto sqrtOp = rewriter.create<dvm::UnaryOp>(op.getLoc(), sqrtResultType, sqrtAttr, adaptor.getInput());
    auto recipAttr = dvm::UnaryOpTypeAttr::get(getContext(), dvm::UnaryOpType::Reciprocal);
    rewriter.replaceOpWithNewOp<dvm::UnaryOp>(op, sqrtResultType, recipAttr, sqrtOp.getResult());
    return success();
  }
};

struct ConvertSelectOp : public OpConversionPattern<mfuse::SelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::SelectOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    rewriter.replaceOpWithNewOp<dvm::SelectOp>(op, op.getResult().getType(), adaptor.getCondition(),
                                               adaptor.getOnTrue(), adaptor.getOnFalse());
    return success();
  }
};

struct ConvertCastOp : public OpConversionPattern<mfuse::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::CastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto resultType = op.getResult().getType();
    auto elementType = llvm::cast<RankedTensorType>(resultType).getElementType();
    dvm::DType dtype;
    if (mlir::isa<Float16Type>(elementType)) {
      dtype = dvm::DType::Float16;
    } else if (mlir::isa<BFloat16Type>(elementType)) {
      dtype = dvm::DType::BFloat16;
    } else if (mlir::isa<Float32Type>(elementType)) {
      dtype = dvm::DType::Float32;
    } else if (mlir::isa<IntegerType>(elementType)) {
      auto width = mlir::cast<IntegerType>(elementType).getWidth();
      if (width == 1) {
        dtype = dvm::DType::Bool;
      } else if (width == 32) {
        dtype = dvm::DType::Int32;
      } else {
        dtype = dvm::DType::Int64;
      }
    } else {
      return failure();
    }
    auto dtypeAttr = dvm::DTypeAttr::get(getContext(), dtype);
    rewriter.replaceOpWithNewOp<dvm::CastOp>(op, resultType, adaptor.getInput(), dtypeAttr);
    return success();
  }
};

struct ConvertBroadcastToOp : public OpConversionPattern<mfuse::BroadcastToOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::BroadcastToOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto shape = rewriter.getI64ArrayAttr(resultType.getShape());
    rewriter.replaceOpWithNewOp<dvm::BroadcastOp>(op, op.getResult().getType(), adaptor.getInput(), shape);
    return success();
  }
};

struct ConvertFullOp : public OpConversionPattern<mfuse::FullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::FullOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }

    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto shape = rewriter.getI64ArrayAttr(resultType.getShape());

    if (auto scalarOp = getScalarConstant(op.getFillValue())) {
      auto scalarType = cast<RankedTensorType>(scalarOp.getResult().getType());
      auto normalized = normalizeScalarConstantForDvmBroadcast(scalarOp, scalarType);
      if (failed(normalized)) return failure();

      auto dtype = getDvmBroadcastDType(resultType.getElementType(), op.getLoc());
      if (failed(dtype)) return failure();

      auto dtypeAttr = dvm::DTypeAttr::get(getContext(), *dtype);
      rewriter.replaceOpWithNewOp<dvm::BroadcastScalarOp>(op, op.getResult().getType(), *normalized, shape, dtypeAttr);
      if (scalarOp.getResult().use_empty()) {
        rewriter.eraseOp(scalarOp);
      }
      return success();
    }

    if (op.getFillValue().getDefiningOp<mfuse::ConstantOp>()) {
      return op->emitError(
        "DVM full conversion requires a scalar mfuse.constant fill value; "
        "non-scalar mfuse.constant cannot be lowered to dvm.broadcast_scalar");
    }

    rewriter.replaceOpWithNewOp<dvm::BroadcastOp>(op, op.getResult().getType(), adaptor.getFillValue(), shape);
    return success();
  }
};

struct ConvertReshapeOp : public OpConversionPattern<mfuse::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto resultType = llvm::cast<RankedTensorType>(op.getResult().getType());
    auto shape = rewriter.getI64ArrayAttr(resultType.getShape());
    rewriter.replaceOpWithNewOp<dvm::ReshapeOp>(op, op.getResult().getType(), adaptor.getInput(), shape);
    return success();
  }
};

struct ConvertReduceSumOp : public OpConversionPattern<mfuse::ReduceSumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::ReduceSumOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto dimsAttr = op.getDimensions();
    auto keepdimAttr = op.getKeepdimAttr();
    rewriter.replaceOpWithNewOp<dvm::ReduceOp>(op, op.getResult().getType(),
                                               dvm::ReduceOpTypeAttr::get(getContext(), dvm::ReduceOpType::Sum),
                                               adaptor.getInput(), dimsAttr, keepdimAttr);
    return success();
  }
};

struct ConvertReluOp : public OpConversionPattern<mfuse::ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    // Relu(x) -> Maximum(x, 0)
    auto input = adaptor.getInput();
    auto inputType = llvm::cast<RankedTensorType>(input.getType());
    auto elementType = inputType.getElementType();
    auto resultType = op.getResult().getType();

    auto zeroAttr = getZeroScalarAttr(elementType, rewriter);
    if (!zeroAttr) {
      return op->emitError("unsupported DVM scalar zero element type: ") << elementType;
    }

    auto maxAttr = dvm::BinaryOpTypeAttr::get(getContext(), dvm::BinaryOpType::Maximum);
    rewriter.replaceOpWithNewOp<dvm::BinaryScalarOp>(op, resultType, maxAttr, input, zeroAttr,
                                                     /*scalarOnLhs=*/false);
    return success();
  }
};

struct ConvertMatmulOp : public OpConversionPattern<mfuse::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    // Use bool version with default-constructed empty Value for optional bias
    rewriter.replaceOpWithNewOp<dvm::MatMulOp>(op, op.getResult().getType(), adaptor.getSelf(), adaptor.getOther(),
                                               op.getTransX1(), op.getTransX2(), mlir::Value());
    return success();
  }
};

struct ConvertBatchMatmulOp : public OpConversionPattern<mfuse::BatchMatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::BatchMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    // BatchMatmul maps directly to dvm::MatMul which supports batch dimensions
    rewriter.replaceOpWithNewOp<dvm::MatMulOp>(op, op.getResult().getType(), adaptor.getSelf(), adaptor.getMat2(),
                                               op.getTransposeAAttr(), op.getTransposeBAttr(), mlir::Value());
    return success();
  }
};

struct ConvertMatmulWithBiasOp : public OpConversionPattern<mfuse::MatmulWithBiasOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::MatmulWithBiasOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto transX1Attr = rewriter.getBoolAttr(op.getTransX1());
    auto transX2Attr = rewriter.getBoolAttr(op.getTransX2());
    rewriter.replaceOpWithNewOp<dvm::MatMulOp>(op, op.getResult().getType(), adaptor.getSelf(), adaptor.getOther(),
                                               transX1Attr, transX2Attr, adaptor.getBias());
    return success();
  }
};

struct ConvertGroupedMatmulOp : public OpConversionPattern<mfuse::GroupedMatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::GroupedMatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    dvm::GroupType groupType;
    switch (op.getGroupType()) {
      case 0:
        groupType = dvm::GroupType::Split_M;
        break;
      case 1:
        groupType = dvm::GroupType::Split_N;
        break;
      case 2:
        groupType = dvm::GroupType::Split_K;
        break;
      default:
        groupType = dvm::GroupType::Split_M;
        break;
    }
    auto transAAttr = rewriter.getBoolAttr(op.getTransposeA());
    auto transBAttr = rewriter.getBoolAttr(op.getTransposeB());
    auto groupTypeAttr = dvm::GroupTypeAttr::get(getContext(), groupType);
    rewriter.replaceOpWithNewOp<dvm::GroupedMatMulOp>(op, op.getResult().getType(), adaptor.getX(), adaptor.getWeight(),
                                                      transAAttr, transBAttr, adaptor.getBias(), adaptor.getGroupList(),
                                                      groupTypeAttr);
    return success();
  }
};

static void insertLoadStoreOps(ModuleOp module) {
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    if (!isDvmOutlinedFunc(func)) continue;

    if (!func.getBlocks().empty()) {
      Block &entryBlock = func.front();
      OpBuilder builder(&entryBlock, entryBlock.begin());
      for (auto arg : entryBlock.getArguments()) {
        if (mlir::isa<RankedTensorType>(arg.getType())) {
          auto loadOp = builder.create<dvm::LoadOp>(func.getLoc(), arg.getType(), arg);
          arg.replaceAllUsesExcept(loadOp.getResult(), loadOp);
        }
      }
    }

    func.walk([&](func::ReturnOp returnOp) {
      OpBuilder builder(returnOp);
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        Value operand = returnOp.getOperand(i);
        if (mlir::isa<RankedTensorType>(operand.getType())) {
          auto storeOp = builder.create<dvm::StoreOp>(returnOp.getLoc(), operand.getType(), operand);
          returnOp.setOperand(i, storeOp.getResult());
        }
      }
    });
  }
}

static LogicalResult cleanupAndVerifyScalarConstants(ModuleOp module) {
  SmallVector<mfuse::ConstantOp> constantsToErase;

  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal() || !isDvmOutlinedFunc(func)) {
      continue;
    }

    auto walkResult = func.walk([&](mfuse::ConstantOp op) {
      auto rankedType = dyn_cast<RankedTensorType>(op.getResult().getType());
      if (!rankedType || !hasScalarMarker(rankedType)) {
        op.emitError(
          "non-scalar constant is not supported in DVM conversion, "
          "only scalar constants (is_scalar=true) are supported");
        return WalkResult::interrupt();
      }

      if (!op.getResult().use_empty()) {
        Operation *user = *op.getResult().user_begin();
        user->emitError()
          << user->getName()
          << " does not support scalar constants yet; add scalar lowering support instead of using a rank-0 tensor "
             "constant";
        return WalkResult::interrupt();
      }

      constantsToErase.push_back(op);
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted()) {
      return failure();
    }
  }

  for (mfuse::ConstantOp op : constantsToErase) {
    op.erase();
  }
  return success();
}

}  // namespace

struct ConvertMfuseToDvmPass : public PassWrapper<ConvertMfuseToDvmPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "convert-mfuse-to-dvm"; }
  StringRef getDescription() const final { return "Convert outlined Mfuse subgraphs to DVM dialect operations"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::mfuse::MfuseDialect>();
    registry.insert<mlir::dvm::DvmDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    insertLoadStoreOps(module);

    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::dvm::DvmDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::tensor::TensorDialect>();
    target.addLegalDialect<mlir::mfuse::MfuseDialect>();

    // Binary operations
    target.addDynamicallyLegalOp<mlir::mfuse::MulOp>(
      [](mlir::mfuse::MulOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::AddOp>(
      [](mlir::mfuse::AddOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::SubOp>(
      [](mlir::mfuse::SubOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::DivOp>(
      [](mlir::mfuse::DivOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::PowOp>(
      [](mlir::mfuse::PowOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::MaximumOp>(
      [](mlir::mfuse::MaximumOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::MinimumOp>(
      [](mlir::mfuse::MinimumOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::EqOp>(
      [](mlir::mfuse::EqOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::NeOp>(
      [](mlir::mfuse::NeOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::GtOp>(
      [](mlir::mfuse::GtOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::GeOp>(
      [](mlir::mfuse::GeOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LtOp>(
      [](mlir::mfuse::LtOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LeOp>(
      [](mlir::mfuse::LeOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LogicalAndOp>(
      [](mlir::mfuse::LogicalAndOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LogicalOrOp>(
      [](mlir::mfuse::LogicalOrOp op) { return !isDvmOutlinedOp(op.getOperation()); });

    // Unary operations
    target.addDynamicallyLegalOp<mlir::mfuse::AbsOp>(
      [](mlir::mfuse::AbsOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::CeilOp>(
      [](mlir::mfuse::CeilOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::FloorOp>(
      [](mlir::mfuse::FloorOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::TruncOp>(
      [](mlir::mfuse::TruncOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::ExpOp>(
      [](mlir::mfuse::ExpOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LogOp>(
      [](mlir::mfuse::LogOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::ReciprocalOp>(
      [](mlir::mfuse::ReciprocalOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::SqrtOp>(
      [](mlir::mfuse::SqrtOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::IsFiniteOp>(
      [](mlir::mfuse::IsFiniteOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::LogicalNotOp>(
      [](mlir::mfuse::LogicalNotOp op) { return !isDvmOutlinedOp(op.getOperation()); });

    // Special operations
    target.addDynamicallyLegalOp<mlir::mfuse::NegOp>(
      [](mlir::mfuse::NegOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::RsqrtOp>(
      [](mlir::mfuse::RsqrtOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::SelectOp>(
      [](mlir::mfuse::SelectOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::CastOp>(
      [](mlir::mfuse::CastOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::BroadcastToOp>(
      [](mlir::mfuse::BroadcastToOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::FullOp>(
      [](mlir::mfuse::FullOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::ReshapeOp>(
      [](mlir::mfuse::ReshapeOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::ReduceSumOp>(
      [](mlir::mfuse::ReduceSumOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::ReluOp>(
      [](mlir::mfuse::ReluOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::RealDivOp>(
      [](mlir::mfuse::RealDivOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::MatmulOp>(
      [](mlir::mfuse::MatmulOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::MatmulWithBiasOp>(
      [](mlir::mfuse::MatmulWithBiasOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    // mfuse.grouped_matmul uses tensor-list operands/results, while
    // dvm.grouped_matmul takes single ranked tensors. Keep it illegal inside a
    // DVM outlined function until the list-to-tensor bridge semantics are
    // defined.
    // TODO: Enable conversion for a verified subset after unpacking
    // MfuseTensorList/MfuseOptionalTensorList operands safely.
    target.addDynamicallyLegalOp<mlir::mfuse::GroupedMatmulOp>(
      [](mlir::mfuse::GroupedMatmulOp op) { return !isDvmOutlinedOp(op.getOperation()); });
    target.addDynamicallyLegalOp<mlir::mfuse::BatchMatmulOp>(
      [](mlir::mfuse::BatchMatmulOp op) { return !isDvmOutlinedOp(op.getOperation()); });

    // Do not mark mfuse.constant dynamically illegal here. Scalar constants are
    // allowed to remain as temporary producers during conversion so binary
    // patterns can inline them into dvm.binary_scalar. A post-conversion cleanup
    // verifies that no mfuse.constant escapes into final DVM IR.

    // Operations not supported by DVM - mark as illegal in dvm outlined functions
    // target.addIllegalOp<mlir::mfuse::Conv2DOp>();
    // target.addIllegalOp<mlir::mfuse::Conv2DWithBiasOp>();
    // target.addIllegalOp<mlir::mfuse::PermuteOp>();
    // target.addIllegalOp<mlir::mfuse::SliceOp>();

    RewritePatternSet patterns(ctx);

    // Binary operations
    patterns.add<ConvertMulOp>(ctx);
    patterns.add<ConvertAddOp>(ctx);
    patterns.add<ConvertSubOp>(ctx);
    patterns.add<ConvertDivOp>(ctx);
    patterns.add<ConvertPowOp>(ctx);
    patterns.add<ConvertMaximumOp>(ctx);
    patterns.add<ConvertMinimumOp>(ctx);
    patterns.add<ConvertEqOp>(ctx);
    patterns.add<ConvertNeOp>(ctx);
    patterns.add<ConvertGtOp>(ctx);
    patterns.add<ConvertGeOp>(ctx);
    patterns.add<ConvertLtOp>(ctx);
    patterns.add<ConvertLeOp>(ctx);
    patterns.add<ConvertLogicalAndOp>(ctx);
    patterns.add<ConvertLogicalOrOp>(ctx);

    // Unary operations
    patterns.add<ConvertAbsOp>(ctx);
    patterns.add<ConvertCeilOp>(ctx);
    patterns.add<ConvertFloorOp>(ctx);
    patterns.add<ConvertTruncOp>(ctx);
    patterns.add<ConvertExpOp>(ctx);
    patterns.add<ConvertLogOp>(ctx);
    patterns.add<ConvertReciprocalOp>(ctx);
    patterns.add<ConvertSqrtOp>(ctx);
    patterns.add<ConvertIsFiniteOp>(ctx);
    patterns.add<ConvertLogicalNotOp>(ctx);

    // Special operations
    patterns.add<ConvertNegOp>(ctx);
    patterns.add<ConvertRsqrtOp>(ctx);
    patterns.add<ConvertSelectOp>(ctx);
    patterns.add<ConvertCastOp>(ctx);
    patterns.add<ConvertBroadcastToOp>(ctx);
    patterns.add<ConvertFullOp>(ctx);
    patterns.add<ConvertReshapeOp>(ctx);
    patterns.add<ConvertReduceSumOp>(ctx);
    patterns.add<ConvertReluOp>(ctx);
    patterns.add<ConvertRealDivOp>(ctx);
    patterns.add<ConvertMatmulOp>(ctx);
    patterns.add<ConvertBatchMatmulOp>(ctx);
    patterns.add<ConvertMatmulWithBiasOp>(ctx);
    // ConvertGroupedMatmulOp is intentionally not registered. See the dynamic
    // legality note above: converting list-typed mfuse.grouped_matmul operands
    // directly to dvm.grouped_matmul would create invalid DVM IR.
    // patterns.add<ConvertGroupedMatmulOp>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    if (failed(cleanupAndVerifyScalarConstants(module))) {
      signalPassFailure();
      return;
    }
  }
};

std::unique_ptr<Pass> createConvertMfuseToDvmPass() { return std::make_unique<ConvertMfuseToDvmPass>(); }

}  // namespace mlir
