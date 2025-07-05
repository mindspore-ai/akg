/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <iterator>
#include <numeric>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

using namespace mlir;
using namespace mlir::mindspore;

class ConvertMindSporeConcatOp : public OpRewritePattern<mindspore::ConcatOp> {
 public:
  using OpRewritePattern<mindspore::ConcatOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::ConcatOp op, PatternRewriter &rewriter) const final {
    IntegerAttr axisAttr = (op.getAxis() == std::nullopt)
                             ? cast<IntegerAttr>(rewriter.getZeroAttr(rewriter.getI64Type()))
                             : op.getAxisAttr();
    (void)rewriter.replaceOpWithNewOp<tosa::ConcatOp>(op, op.getType(), op.getInput(), axisAttr);
    return success();
  }
};

// mindspore::MulOp and mindspore::SquareOp
template <typename SourceOp>
class ConvertMindSporeMulOp : public OpRewritePattern<SourceOp> {
 public:
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const final {
    Value lhs = op.getInput1();
    Value rhs = op.getInput2();

    Operation *operation = op;
    auto resultTy = dyn_cast<ShapedType>(operation->getResult(0).getType());
    (void)rewriter.replaceOpWithNewOp<tosa::MulOp>(op, resultTy, lhs, rhs, 0);
    return success();
  }
};

// unary ops
template <typename SourceOp, typename TargetOp>
class ConvertMindSporeUnaryOp : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp mindsporeOp, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value opnd = adaptor.getInput();

    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    auto resultElemTy = resultTy.getElementType();
    if (!resultElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Only floating-point or integer datatype legalization supported");
    }
    auto unaryOp = rewriter.create<TargetOp>(op->getLoc(), resultTy, opnd);
    rewriter.replaceOp(op, unaryOp.getResult());
    return success();
  }
};

// binary ops
template <typename SourceOp, typename TargetOp>
class ConvertMindSporeBinaryOp : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp mindsporeOp, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    (void)adaptor;
    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    auto resultElemTy = resultTy.getElementType();
    if (!resultElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Only floating-point or integer datatype legalization supported");
    }
    auto binaryOp = rewriter.create<TargetOp>(op->getLoc(), resultTy, lhs, rhs);
    rewriter.replaceOp(op, binaryOp.getResult());
    return success();
  }
};

// reduce ops
template <typename SourceOp, typename TargetOp>
class ConvertMindSporeReduceOp : public OpConversionPattern<SourceOp> {
 public:
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp mindsporeOp, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    MLIRContext *context = rewriter.getContext();
    Operation *op = mindsporeOp;
    auto loc = op->getLoc();
    Value opnd = adaptor.getInput();
    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    auto resultElementTy = resultTy.getElementType();
    BoolAttr keepdims_attr = BoolAttr::get(context, false);
    if (adaptor.getKeepdimsAttr()) {
      keepdims_attr = adaptor.getKeepdimsAttr();
    }
    ShapedType input_shapes = cast<ShapedType>(adaptor.getInput().getType());
    llvm::SmallVector<int64_t> reduce_output_shape;
    (void)std::copy(input_shapes.getShape().begin(), input_shapes.getShape().end(),
                    std::back_inserter(reduce_output_shape));
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    auto sym_shape = analysis.getSymbolicShape(input_shapes);

    // all reduce
    bool is_all_reduce = true;
    llvm::SmallVector<int64_t> axes(adaptor.getAxis().begin(), adaptor.getAxis().end());
    for (size_t i = 0; i < input_shapes.getShape().size(); i++) {
      if (!llvm::is_contained(axes, int64_t(i)) && (int64_t(input_shapes.getShape()[i])) != 1) {
        is_all_reduce = false;
        break;
      }
    }
    if (is_all_reduce) {
      int64_t total_size =
        std::accumulate(input_shapes.getShape().begin(), input_shapes.getShape().end(), 1, std::multiplies<int64_t>());
      llvm::SmallVector<NamedAttribute> attrs_flat;
      (void)attrs_flat.emplace_back(
        NamedAttribute(StringAttr::get(context, "new_shape"), DenseI64ArrayAttr::get(context, total_size)));
      llvm::SmallVector<NamedAttribute> attrs_reduce;
      (void)attrs_reduce.emplace_back(
        NamedAttribute(StringAttr::get(context, "axis"), IntegerAttr::get(rewriter.getI64Type(), 0)));
      (void)attrs_reduce.emplace_back(NamedAttribute(StringAttr::get(context, "keepdims"), keepdims_attr));
      auto flat_tensor = RankedTensorType::get(total_size, resultElementTy);
      llvm::SmallVector<std::string> one_size;
      (void)one_size.emplace_back("1");
      if (sym_shape) {
        llvm::SmallVector<std::string> total_size_symbolic;
        (void)total_size_symbolic.emplace_back(std::to_string(total_size));
        flat_tensor = dyn_cast<RankedTensorType>(analysis.updateSymbolicShape(flat_tensor, total_size_symbolic));
      }
      auto flat_op = rewriter.create<mindspore::ReshapeOp>(loc, flat_tensor, opnd, attrs_flat);
      auto out_tensor = RankedTensorType::get(1, resultElementTy);
      auto reduce_op = rewriter.create<TargetOp>(loc, out_tensor, flat_op.getResult(), attrs_reduce);
      opnd = reduce_op.getResult();
      if (keepdims_attr.getValue()) {
        llvm::SmallVector<int64_t> new_shape;
        (void)std::copy(resultTy.getShape().begin(), resultTy.getShape().end(), std::back_inserter(new_shape));
        llvm::SmallVector<NamedAttribute> attr;
        (void)attr.emplace_back(NamedAttribute(StringAttr::get(context, "new_shape"),
                                               DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(new_shape))));
        auto reshape_op = rewriter.create<mindspore::ReshapeOp>(loc, resultTy, opnd, attr);
        opnd = reshape_op.getResult();
      }
      rewriter.replaceOp(op, opnd);
      return success();
    }

    // create one tosa.reduce operation for each axis
    for (int64_t i = 0; i < adaptor.getAxisAttr().size(); i++) {
      int64_t axis = (int64_t)adaptor.getAxisAttr()[(unsigned long)i];
      reduce_output_shape[(unsigned long)axis] = (int64_t)1;
      auto reduce_inter_tensor = RankedTensorType::get(reduce_output_shape, resultElementTy);
      llvm::SmallVector<NamedAttribute> attrs_once;
      (void)attrs_once.emplace_back(
        NamedAttribute(StringAttr::get(context, "axis"), IntegerAttr::get(rewriter.getI64Type(), axis)));
      (void)attrs_once.emplace_back(NamedAttribute(StringAttr::get(context, "keepdims"), keepdims_attr));
      if (sym_shape) {
        (*sym_shape)[(unsigned long)axis] = "1";
        reduce_inter_tensor =
          dyn_cast<RankedTensorType>(analysis.updateSymbolicShape(reduce_inter_tensor, *sym_shape));
      }
      auto reduce_op_once = rewriter.create<TargetOp>(loc, reduce_inter_tensor, opnd, attrs_once);
      opnd = reduce_op_once.getResult();
    }
    // if keep-dims = false, reshape the output
    if (!keepdims_attr.getValue()) {
      llvm::SmallVector<int64_t> new_shape;
      (void)std::copy(resultTy.getShape().begin(), resultTy.getShape().end(), std::back_inserter(new_shape));
      llvm::SmallVector<NamedAttribute> attrs_once;
      (void)attrs_once.emplace_back(NamedAttribute(StringAttr::get(context, "new_shape"),
                                                   DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(new_shape))));
      auto reshape_op = rewriter.create<mindspore::ReshapeOp>(loc, resultTy, opnd, attrs_once);
      opnd = reshape_op.getResult();
    }
    rewriter.replaceOp(op, opnd);

    return success();
  }
};

// specific for NotEqual ops lowering
template <typename SrcOp, typename DstOp>
class ConvertMindSporeNotBinaryOp : public OpConversionPattern<SrcOp> {
 public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using OpAdaptor = typename SrcOp::Adaptor;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    (void)adaptor;
    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    auto resultElemTy = resultTy.getElementType();
    if (!resultElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Only floating-point or integer datatype legalization supported");
    }
    auto resultOp = rewriter.create<DstOp>(op->getLoc(), resultTy, lhs, rhs);
    (void)rewriter.replaceOpWithNewOp<tosa::LogicalNotOp>(op, resultTy, resultOp.getResult());
    return success();
  }
};

// specific for select
template <typename SrcOp, typename DstOp>
class ConvertMindSporeSelectOp : public OpConversionPattern<SrcOp> {
 public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, Adaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value cond = op->getOperand(0);
    Value xVal = op->getOperand(1);
    Value yVal = op->getOperand(2);
    (void)adaptor;
    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    auto resultElemTy = resultTy.getElementType();
    if (!resultElemTy.isIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Only floating-point or integer datatype legalization supported");
    }
    auto resultOp = rewriter.create<DstOp>(op->getLoc(), resultTy, cond, xVal, yVal);
    rewriter.replaceOp(op, resultOp.getResult());
    return success();
  }
};

template <typename T>
std::optional<Value> getConstTensor(PatternRewriter &rewriter, Operation *op, const ArrayRef<T> vec,
                                    const ArrayRef<int64_t> shape) {
  int64_t elemNum = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
  if (vec.size() != (uint64_t)elemNum) {
    (void)op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto constType = RankedTensorType::get(shape, rewriter.getIntegerType(sizeof(T) * 8));
  auto constAttr = DenseElementsAttr::get(constType, vec);
  auto constOp = rewriter.create<tosa::ConstOp>(op->getLoc(), constType, constAttr);
  return constOp.getResult();
}

template <>
std::optional<Value> getConstTensor<float>(PatternRewriter &rewriter, Operation *op, const ArrayRef<float> vec,
                                           ArrayRef<int64_t> shape) {
  uint64_t elemNum = 1;
  for (int64_t a : shape) {
    elemNum *= (uint64_t)a;
  }

  if (vec.size() != elemNum) {
    (void)op->emitOpError("getConstTensor(): number of elements mismatch.");
    return std::nullopt;
  }

  auto constType = RankedTensorType::get(shape, rewriter.getF32Type());
  auto constAttr = DenseElementsAttr::get(constType, vec);

  auto constOp = rewriter.create<tosa::ConstOp>(op->getLoc(), constType, constAttr);
  return constOp.getResult();
}

template <typename SrcOp>
class ConvertMindSporePadOp : public OpConversionPattern<SrcOp> {
 public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, Adaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    auto padding = adaptor.getPadding();
    auto mode = adaptor.getMode();
    auto value = adaptor.getValue();
    Location loc = op->getLoc();
    Value inputX = adaptor.getInputX();
    if (!isa<RankedTensorType>(inputX.getType())) {
      return rewriter.notifyMatchFailure(op, "only support for ranked tensor");
    }
    auto inputTy = dyn_cast<RankedTensorType>(inputX.getType());
    auto inputElemTy = inputTy.getElementType();
    if (!value.has_value()) {
      // set the default value  zero;
      if (isa<IntegerType>(inputElemTy) || isa<FloatType>(inputElemTy)) {
        IntegerType i64Ty = rewriter.getI64Type();
        mindsporeOp.setValueAttr(rewriter.getIntegerAttr(i64Ty, 0));
      }
    }

    if (!mode.has_value()) {
      mindsporeOp.setModeAttr(rewriter.getStringAttr("constant"));
    }

    int64_t rank = inputTy.getRank();

    SmallVector<int64_t> padInts;
    (void)std::copy(padding.begin(), padding.end(), std::back_inserter(padInts));

    const uint32_t doubleSize = 2;
    uint64_t padRank = padInts.size() / doubleSize;
    if (padRank * doubleSize != padInts.size()) {
      return rewriter.notifyMatchFailure(op, "pad range size should be even");
    }

    if (rank < 0 || padRank > (uint64_t)rank) {
      return rewriter.notifyMatchFailure(op, "padding exceeds out tensor rank");
    }

    // Initialize all the tensor dim padding with 0;
    SmallVector<int64_t> lowPadding(rank, 0);
    SmallVector<int64_t> highPadding(rank, 0);
    for (unsigned int i = 0; i < padRank; i++) {
      lowPadding[(unsigned long)rank - i - 1] = padInts[i * doubleSize];
      highPadding[(unsigned long)rank - i - 1] = padInts[i * doubleSize + 1];
    }

    SmallVector<int64_t> paddingList;
    for (unsigned int i = 0; i < rank; i++) {
      paddingList.push_back(lowPadding[i]);
      paddingList.push_back(highPadding[i]);
    }

    DenseElementsAttr paddingAttr =
      DenseIntElementsAttr::get(RankedTensorType::get({rank, 2}, rewriter.getI64Type()), paddingList);

    const Value padList = rewriter.create<tosa::ConstOp>(loc, paddingAttr.getType(), paddingAttr);

    Value padTensor;
    IntegerAttr integerAttr = mindsporeOp.getValueAttr();
    int64_t padValue = integerAttr.getInt();
    if (failed(MindSporeScalarToTosaTensor(rewriter, op, padValue, padTensor, inputElemTy, {}))) {
      return rewriter.notifyMatchFailure(
        op, "Pad value needs to be a scalar constant for conversion to TOSA pad operation");
    }
    auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
    (void)rewriter.replaceOpWithNewOp<tosa::PadOp>(mindsporeOp, resultTy, inputX, padList, padTensor);
    return success();
  }

  template <typename T>
  static bool isInvalidRange(const int64_t &intValue) {
    return (intValue >= std::numeric_limits<T>::min()) && (intValue <= std::numeric_limits<T>::max());
  }

  LogicalResult MindSporeScalarToTosaTensor(ConversionPatternRewriter &rewriter, Operation *op, int64_t padScalarValue,
                                            Value &tosaTensor, const Type dtype,
                                            const llvm::ArrayRef<int64_t> dshape) const {
    uint32_t width32 = 32, width64 = 64;
    if (isa<FloatType>(dtype)) {
      float floatValue = static_cast<float>(padScalarValue);
      tosaTensor = getConstTensor<float>(rewriter, op, {floatValue}, dshape).value();
    } else if (auto intType = dyn_cast<IntegerType>(dtype)) {
      auto w = intType.getWidth();
      if (w != width32 && w != width64) {
        return rewriter.notifyMatchFailure(op, "only support 32 or 64 bits int");
      }

      if (w == width32) {
        if (isInvalidRange<int32_t>(padScalarValue)) {
          int32_t dVal = static_cast<int32_t>(padScalarValue);
          tosaTensor = getConstTensor<int32_t>(rewriter, op, {dVal}, dshape).value();
        } else {
          return rewriter.notifyMatchFailure(op, "value of scalar constant exceeds limits of destination type");
        }
      }

      if (w == width64) {
        if (!isInvalidRange<int64_t>(padScalarValue)) {
          return rewriter.notifyMatchFailure(op, "value of scalar constant exceeds limits of destination type");
        }
        int64_t dVal = static_cast<int64_t>(padScalarValue);
        tosaTensor = getConstTensor<int64_t>(rewriter, op, {dVal}, dshape).value();
      }
    }
    return success();
  }
};

struct ConvertMindSporeToTosaPass : public ConvertMindSporeToTosaBase<ConvertMindSporeToTosaPass> {
 public:
  ConvertMindSporeToTosaPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    FunctionOpInterface func = getOperation();

    target.addLegalDialect<tosa::TosaDialect>();
    target.addIllegalDialect<mindspore::MindSporeDialect>();
    target.addLegalOp<mindspore::ReshapeOp>();
    target.addLegalOp<mindspore::AssignOp>();
    target.addLegalOp<mindspore::AddNOp>();
    target.addLegalOp<mindspore::SqrtOp>();
    target.addLegalOp<mindspore::LessEqualOp>();
    target.addLegalOp<mindspore::LessOp>();
    target.addLegalOp<mindspore::DivOp>();
    target.addLegalOp<mindspore::SqrtOp>();
    target.addLegalOp<mindspore::CosOp>();
    target.addLegalOp<mindspore::SinOp>();
    target.addLegalOp<mindspore::AsinOp>();
    target.addLegalOp<mindspore::AsinhOp>();
    target.addLegalOp<mindspore::AcosOp>();
    target.addLegalOp<mindspore::AcoshOp>();
    target.addLegalOp<mindspore::AtanOp>();
    target.addLegalOp<mindspore::Atan2Op>();
    target.addLegalOp<mindspore::GatherOp>();
    target.addLegalOp<mindspore::SliceOp>();
    target.addLegalOp<mindspore::Strided_SliceOp>();
    target.addLegalOp<mindspore::SplitOp>();
    target.addLegalOp<mindspore::IsnanOp>();
    target.addLegalOp<mindspore::IsinfOp>();
    target.addLegalOp<mindspore::InplaceAssignOp>();
    target.addLegalOp<mindspore::ReduceAllOp>();
    target.addLegalOp<mindspore::ReduceAnyOp>();
    target.addLegalOp<mindspore::ReduceMinOp>();
    target.addLegalOp<mindspore::ReduceMaxOp>();
    target.addLegalOp<mindspore::ReduceSumOp>();
    target.addLegalOp<mindspore::ReduceProdOp>();
    target.addLegalOp<mindspore::ReduceAnyOp>();
    target.addLegalOp<mindspore::UnsortedSegmentSumOp>();
    target.addLegalOp<mindspore::GatherOp>();
    target.addLegalOp<mindspore::BroadcastToOp>();
    target.addLegalOp<mindspore::TileOp>();
    target.addLegalOp<mindspore::MatMulOp>();
    target.addLegalOp<mindspore::BatchMatMulOp>();
    target.addLegalOp<mindspore::ConstOp>();

    // clang-format off
    (void)patterns.add<
      ConvertMindSporeBinaryOp<mindspore::AddOp, tosa::AddOp>,
      ConvertMindSporeBinaryOp<mindspore::SubOp, tosa::SubOp>,
      ConvertMindSporeBinaryOp<mindspore::PowOp, tosa::PowOp>,
      ConvertMindSporeBinaryOp<mindspore::TransposeOp, tosa::TransposeOp>,
      ConvertMindSporeBinaryOp<mindspore::GreaterOp, tosa::GreaterOp>,
      ConvertMindSporeBinaryOp<mindspore::GreaterEqualOp, tosa::GreaterEqualOp>,
      ConvertMindSporeBinaryOp<mindspore::EqualOp, tosa::EqualOp>,
      ConvertMindSporeBinaryOp<mindspore::LogicalAndOp, tosa::LogicalAndOp>,
      ConvertMindSporeBinaryOp<mindspore::LogicalOrOp, tosa::LogicalOrOp>,
      ConvertMindSporeBinaryOp<mindspore::MaximumOp, tosa::MaximumOp>,
      ConvertMindSporeBinaryOp<mindspore::MinimumOp, tosa::MinimumOp>,
      ConvertMindSporeUnaryOp<mindspore::ExpOp, tosa::ExpOp>,
      ConvertMindSporeUnaryOp<mindspore::TanhOp, tosa::TanhOp>,
      ConvertMindSporeUnaryOp<mindspore::CastOp, tosa::CastOp>,
      ConvertMindSporeUnaryOp<mindspore::NegateOp, tosa::NegateOp>,
      ConvertMindSporeUnaryOp<mindspore::InvOp, tosa::ReciprocalOp>,
      ConvertMindSporeUnaryOp<mindspore::RsqrtOp, tosa::RsqrtOp>,
      ConvertMindSporeUnaryOp<mindspore::LogOp, tosa::LogOp>,
      ConvertMindSporeUnaryOp<mindspore::AbsOp, tosa::AbsOp>,
      ConvertMindSporeUnaryOp<mindspore::FloorOp, tosa::FloorOp>,
      ConvertMindSporeUnaryOp<mindspore::LogicalNotOp, tosa::LogicalNotOp>,
      ConvertMindSporeReduceOp<mindspore::ArgMaxOp, tosa::ArgMaxOp>,
      ConvertMindSporeNotBinaryOp<mindspore::NotEqualOp, tosa::EqualOp>,
      ConvertMindSporeSelectOp<mindspore::SelectOp, tosa::SelectOp>,
      ConvertMindSporeMulOp<mindspore::MulOp>,
      ConvertMindSporeMulOp<mindspore::SquareOp>,
      ConvertMindSporeConcatOp,
      ConvertMindSporePadOp<mindspore::PadOp>
    >(patterns.getContext());
    mlir::populateMindSporeLowerPattern(patterns);

    // clang-format on
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMindSporeToTosaPass() {
  return std::make_unique<ConvertMindSporeToTosaPass>();
}
