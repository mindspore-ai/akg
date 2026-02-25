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
#include "mfusion/Conversion/MfuseToTorch/MfuseMetaToTorch.h"
#include "mfusion/Conversion/MfuseToTorch/MfuseAclnnToTorch.h"

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
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Support/Logging.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "MfuseToTorch.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

void populateMfuseToTorchTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](mlir::RankedTensorType type) -> mlir::Type {
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

class MfuseToTorchTypeConverter : public mlir::TypeConverter {
 public:
  MfuseToTorchTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    populateMfuseToTorchTypeConversions(*this);

    addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });

    addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });
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

struct ConvertMfuseToTorchPass : public mlir::PassWrapper<ConvertMfuseToTorchPass, mlir::OperationPass<mlir::ModuleOp>> {
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
    populateGeneratedPDLLPatterns(patternList, mlir::PDLConversionConfig(&converter_));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patternList, converter_);
    mlir::populateMfuseMetaToTorchConversionPatterns(converter_, patternList);
    mlir::populateMfuseAclnnToTorchConversionPatterns(converter_, patternList);
    patternList.add<ConvertDvmCallOp>(converter_, ctx);

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
