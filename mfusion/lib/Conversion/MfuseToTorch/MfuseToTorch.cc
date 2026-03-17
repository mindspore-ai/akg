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

#include "mfusion/Conversion/MfuseToTorch/MfuseToTorch.h"

#include <algorithm>
#include <iterator>
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Conversion/MfuseToTorch/MfuseMetaToTorch.h"
#include "mfusion/Conversion/MfuseToTorch/MfuseAclnnToTorch.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Support/Logging.h"
#include "mfusion/Conversion/PdllHelper.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "MfuseToTorch.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

void populateMfuseToTorchTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
    auto encoding = type.getEncoding();
    if (encoding) {
      auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
      if (dictAttr && dictAttr.contains(mlir::mfuse::kScalarMarkerAttr)) {
        // Check the value of is_scalar attribute
        auto scalarAttr = dictAttr.get(mlir::mfuse::kScalarMarkerAttr);
        if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(scalarAttr)) {
          std::string typeStr = strAttr.getValue().str();
          // Convert based on the original torch type in is_scalar
          if (typeStr == "!torch.int") {
            return TorchD::IntType::get(type.getContext());
          } else if (typeStr == "!torch.float") {
            return TorchD::FloatType::get(type.getContext());
          } else if (typeStr == "!torch.bool") {
            return TorchD::BoolType::get(type.getContext());
          }
        }
        // Fall back to original logic if is_scalar is not a string or doesn't match known torch types
        auto elementType = type.getElementType();
        if (mlir::isa<mlir::FloatType>(elementType)) {
          return TorchD::FloatType::get(type.getContext());
        }
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elementType)) {
          return TorchD::IntType::get(type.getContext());
        }
      }
    }

    llvm::SmallVector<int64_t> shape;
    auto sizes = type.getShape();
    shape.reserve(sizes.size());
    std::transform(sizes.begin(), sizes.end(), std::back_inserter(shape),
                   [](int64_t dim) { return dim == mlir::ShapedType::kDynamic ? TorchD::kUnknownSize : dim; });
    return TorchD::ValueTensorType::get(type.getContext(), llvm::ArrayRef<int64_t>(shape), type.getElementType());
  });
  converter.addConversion([](mlir::UnrankedTensorType type) -> mlir::Type {
    return TorchD::ValueTensorType::get(type.getContext(), std::nullopt, type.getElementType());
  });

  converter.addConversion(
    [](mlir::mfuse::StringType type) -> mlir::Type { return TorchD::StringType::get(type.getContext()); });
  converter.addConversion(
    [](mlir::mfuse::NoneType type) -> mlir::Type { return TorchD::NoneType::get(type.getContext()); });
  converter.addConversion([&](mlir::mfuse::ListType type) -> mlir::Type {
    return TorchD::ListType::get(type.getContext(), converter.convertType(type.getContainedType()));
  });
}

namespace {

// Helper function to convert a dense elements attribute to a Torch constant
mlir::Value convertDenseElementsAttrToTorchConstant(mlir::OpBuilder &builder, mlir::DenseElementsAttr denseAttr,
                                                    mlir::Location loc, mlir::Value input) {
  // Always check encoding
  auto encoding = mlir::dyn_cast<mlir::RankedTensorType>(input.getType()).getEncoding();
  if (!encoding) {
    return {};
  }
  auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
  if (!dictAttr || !dictAttr.contains(mlir::mfuse::kScalarMarkerAttr)) {
    return {};
  }

  // Check the value of is_scalar attribute
  auto scalarAttr = dictAttr.get(mlir::mfuse::kScalarMarkerAttr);
  if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(scalarAttr)) {
    std::string typeStr = strAttr.getValue().str();

    // Convert based on the original torch type in is_scalar
    if (typeStr == "!torch.int") {
      // For torch.int, ensure we convert to integer even if stored as float
      if (mlir::isa<mlir::FloatType>(denseAttr.getType().getElementType())) {
        double val = denseAttr.getSplatValue<mlir::APFloat>().convertToDouble();
        return builder.create<TorchD::ConstantIntOp>(loc, builder.getI64IntegerAttr(static_cast<int64_t>(val)));
      }
      int64_t val = denseAttr.getSplatValue<mlir::APInt>().getSExtValue();
      return builder.create<TorchD::ConstantIntOp>(loc, builder.getI64IntegerAttr(val));
    } else if (typeStr == "!torch.float") {
      // For torch.float, ensure we convert to float even if stored as integer
      if (mlir::isa<mlir::IntegerType>(denseAttr.getType().getElementType())) {
        int64_t val = denseAttr.getSplatValue<mlir::APInt>().getSExtValue();
        return builder.create<TorchD::ConstantFloatOp>(loc, builder.getF64FloatAttr(static_cast<double>(val)));
      }
      double val = denseAttr.getSplatValue<mlir::APFloat>().convertToDouble();
      return builder.create<TorchD::ConstantFloatOp>(loc, builder.getF64FloatAttr(val));
    } else if (typeStr == "!torch.bool") {
      auto boolValue = denseAttr.getSplatValue<mlir::APInt>().getBoolValue();
      return builder.create<TorchD::ConstantBoolOp>(loc, builder.getBoolAttr(boolValue));
    }
  }

  // Fall back to original logic if is_scalar is not a string or doesn't match known torch types
  auto tensorType = denseAttr.getType();
  mlir::Type elemType = tensorType.getElementType();

  if (mlir::isa<mlir::FloatType>(elemType)) {
    double val = denseAttr.getSplatValue<mlir::APFloat>().convertToDouble();
    return builder.create<TorchD::ConstantFloatOp>(loc, builder.getF64FloatAttr(val));
  }

  if (mlir::isa<mlir::IntegerType>(elemType)) {
    int64_t val = denseAttr.getSplatValue<mlir::APInt>().getSExtValue();
    return builder.create<TorchD::ConstantIntOp>(loc, builder.getI64IntegerAttr(val));
  }

  return {};
}

}  // namespace

class MfuseToTorchTypeConverter : public mlir::TypeConverter {
 private:
  static mlir::Value tryConvertConstant(mlir::OpBuilder &builder, mlir::Type toType, mlir::Value input,
                                        mlir::Location loc) {
    if (auto cst = input.getDefiningOp<mlir::mfuse::ConstantOp>()) {
      if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
        // Use the common helper function with encoding check
        return convertDenseElementsAttrToTorchConstant(builder, denseAttr, loc, input);
      }
    }
    return {};
  }

 public:
  MfuseToTorchTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    populateMfuseToTorchTypeConversions(*this);

    addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        mlir::Value input = inputs[0];
        // Try to convert to different types of Torch constants
        if (auto v = tryConvertConstant(builder, toType, input, loc)) return v;
        // Fall back to conversion cast if not a handled constant type
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });

    addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });
  }
};

class ConvertMfuseConstantToTorch : public mlir::OpConversionPattern<mlir::mfuse::ConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::ConstantOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = op.getLoc();
    mlir::Attribute value = op.getValue();
    mlir::Type resultType = op.getResult().getType();

    mlir::Type convertedType = getTypeConverter()->convertType(resultType);
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value);
    if (!denseAttr) {
      return rewriter.notifyMatchFailure(op, "value must be a dense elements attribute");
    }

    auto tensorType = denseAttr.getType();

    // Use the common helper function to convert tensors with scalar marker encoding
    if (auto torchCst = convertDenseElementsAttrToTorchConstant(rewriter, denseAttr, loc, op.getResult())) {
      rewriter.replaceOp(op, torchCst);
      return mlir::success();
    }

    llvm::SmallVector<int64_t> shape;
    auto sizes = mlir::dyn_cast<mlir::RankedTensorType>(resultType).getShape();
    shape.reserve(sizes.size());
    std::transform(sizes.begin(), sizes.end(), std::back_inserter(shape),
                   [](int64_t dim) { return dim == mlir::ShapedType::kDynamic ? TorchD::kUnknownSize : dim; });
    auto vtensorType = TorchD::ValueTensorType::get(rewriter.getContext(), shape, tensorType.getElementType());

    if (vtensorType) {
      auto torchTensor = rewriter.create<TorchD::ValueTensorLiteralOp>(loc, vtensorType, denseAttr);

      rewriter.replaceOp(op, torchTensor.getResult());
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "failed to convert tensor type");
  }
};

class ConvertAkgCallOp : public mlir::OpConversionPattern<mlir::mfuse::AkgCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::AkgCallOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }

    auto numInputs = adaptor.getOperands().size();
    auto numOutputs = op->getNumResults();
    std::string opName = llvm::formatv("torch.mfusion.akg_call__i{0}_o{1}", numInputs, numOutputs).str();

    mlir::OperationState subgraphState(op.getLoc(), "torch.constant.str");
    subgraphState.addAttribute("value", op.getSubgraphAttr());
    subgraphState.addTypes(TorchD::StringType::get(op.getContext()));
    mlir::Operation *subgraphConst = rewriter.create(subgraphState);
    mlir::Value subgraphValue = subgraphConst->getResult(0);

    mlir::OperationState state(op.getLoc(), "torch.operator");
    state.addOperands(adaptor.getOperands());
    state.addOperands(subgraphValue);
    state.addTypes(resultTypes);
    state.addAttribute("name", rewriter.getStringAttr(opName));
    state.addAttribute("mfusion.subgraph_mlir", op.getSubgraphMlirAttr());
    state.addAttribute("mfusion.is_dynamic", op.getIsDynamicAttr());

    mlir::Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

class ConvertBishengCallOp : public mlir::OpConversionPattern<mlir::mfuse::BishengCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::BishengCallOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }

    auto numInputs = adaptor.getOperands().size();
    auto numOutputs = op->getNumResults();
    std::string opName = llvm::formatv("torch.mfusion.bisheng_call__i{0}_o{1}", numInputs, numOutputs).str();

    mlir::OperationState subgraphState(op.getLoc(), "torch.constant.str");
    subgraphState.addAttribute("value", op.getSubgraphAttr());
    subgraphState.addTypes(TorchD::StringType::get(op.getContext()));
    mlir::Operation *subgraphConst = rewriter.create(subgraphState);
    mlir::Value subgraphValue = subgraphConst->getResult(0);

    mlir::OperationState state(op.getLoc(), "torch.operator");
    state.addOperands(adaptor.getOperands());
    state.addOperands(subgraphValue);
    state.addTypes(resultTypes);
    state.addAttribute("name", rewriter.getStringAttr(opName));
    state.addAttribute("mfusion.subgraph_mlir", op.getSubgraphMlirAttr());
    state.addAttribute("mfusion.is_dynamic", op.getIsDynamicAttr());

    mlir::Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

class ConvertDvmCallOp : public mlir::OpConversionPattern<mlir::mfuse::DvmCallOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(mlir::mfuse::DvmCallOp op, OpAdaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type, 4> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes))) {
      return rewriter.notifyMatchFailure(op, "failed to convert result types");
    }

    auto numInputs = adaptor.getOperands().size();
    auto numOutputs = op->getNumResults();
    std::string opName = llvm::formatv("torch.mfusion.dvm_call__i{0}_o{1}", numInputs, numOutputs).str();

    mlir::OperationState subgraphState(op.getLoc(), "torch.constant.str");
    subgraphState.addAttribute("value", op.getSubgraphAttr());
    subgraphState.addTypes(TorchD::StringType::get(op.getContext()));
    mlir::Operation *subgraphConst = rewriter.create(subgraphState);
    mlir::Value subgraphValue = subgraphConst->getResult(0);

    mlir::OperationState state(op.getLoc(), "torch.operator");
    state.addOperands(adaptor.getOperands());
    state.addOperands(subgraphValue);
    state.addTypes(resultTypes);
    state.addAttribute("name", rewriter.getStringAttr(opName));
    state.addAttribute("mfusion.subgraph_mlir", op.getSubgraphMlirAttr());
    state.addAttribute("mfusion.is_dynamic", op.getIsDynamicAttr());

    mlir::Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }
};

struct ConvertMfuseToTorchPass
    : public mlir::PassWrapper<ConvertMfuseToTorchPass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const final { return "convert-mfuse-to-torch"; }

  mlir::StringRef getDescription() const final { return "Convert Mfuse operations to Torch dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::mfuse::MfuseDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::pdl::PDLDialect>();
    registry.insert<mlir::pdl_interp::PDLInterpDialect>();
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    mlir::RewritePatternSet patternList(ctx);
    mlir::registerConversionPDLFunctions(patternList);
    mlir::registerPDLLHelperFunctions(patternList);
    populateGeneratedPDLLPatterns(patternList, mlir::PDLConversionConfig(&converter_));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patternList, converter_);
    mlir::populateMfuseMetaToTorchConversionPatterns(converter_, patternList);
    mlir::populateMfuseAclnnToTorchConversionPatterns(converter_, patternList);
    patternList.add<ConvertAkgCallOp, ConvertBishengCallOp, ConvertDvmCallOp>(converter_, ctx);
    patternList.add<ConvertMfuseConstantToTorch>(converter_, ctx);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    MLOG(DEBUG) << "convert-mfuse-to-torch pass start";
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<mlir::mfuse::MfuseDialect>();
    target.addLegalDialect<TorchD::TorchDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return converter_.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return converter_.isLegal(op.getOperandTypes()); });

    if (mlir::failed(mlir::applyPartialConversion(module, target, patterns_))) {
      signalPassFailure();
    }
    MLOG(DEBUG) << "convert-mfuse-to-torch pass end";
  }

  mlir::FrozenRewritePatternSet patterns_;
  MfuseToTorchTypeConverter converter_;
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertMfuseToTorchPass() { return std::make_unique<ConvertMfuseToTorchPass>(); }
}  // namespace mlir
