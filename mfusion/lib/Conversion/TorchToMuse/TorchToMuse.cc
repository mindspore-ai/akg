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

#include "mfusion/Conversion/TorchToMuse/TorchToMuse.h"

#include <algorithm>
#include <iterator>
#include <numeric>

#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include "mfusion/Conversion/MuseTypeConverter.h"
#include "mfusion/Conversion/TorchToMuse/TorchAtenToMuse.h"
#include "mfusion/Conversion/TorchToMuse/TorchNpuToMuse.h"
#include "mfusion/Conversion/TorchToMuse/TorchArithToMuse.h"
#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "TorchToMuse.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

// Populate Torch-specific type conversions to Muse types
void populateTorchToMuseTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](TorchD::ValueTensorType type) -> mlir::Type {
    auto optionalSizes = type.getOptionalSizes();
    if (optionalSizes.has_value()) {
      // Normalize dynamic dims: Torch uses -1 / kUnknownSize, Muse uses ShapedType::kDynamic
      llvm::SmallVector<int64_t> shape;
      auto sizes = optionalSizes.value();
      shape.reserve(sizes.size());
      std::transform(sizes.begin(), sizes.end(), std::back_inserter(shape),
                     [](int64_t dim) { return dim < 0 ? mlir::ShapedType::kDynamic : dim; });
      return mlir::muse::TensorType::get(type.getContext(), shape, type.getOptionalDtype(), nullptr);
    } else {
      return mlir::muse::TensorType::get(type.getContext(), std::nullopt, type.getOptionalDtype(), nullptr);
    }
  });

  converter.addConversion(
    [](TorchD::IntType type) -> mlir::Type { return mlir::muse::I64Type::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::FloatType type) -> mlir::Type { return mlir::muse::F64Type::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::BoolType type) -> mlir::Type { return mlir::muse::BooleanType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::StringType type) -> mlir::Type { return mlir::muse::StringType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::DeviceType type) -> mlir::Type { return mlir::muse::StringType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::NoneType type) -> mlir::Type { return mlir::muse::NoneType::get(type.getContext()); });

  converter.addConversion([&](TorchD::ListType type) -> mlir::Type {
    return mlir::muse::ListType::get(type.getContext(), converter.convertType(type.getContainedType()));
  });
}

// TypeConverter for Torch to Muse conversion
class TorchToMuseTypeConverter : public mlir::TypeConverter {
 public:
  TorchToMuseTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    mlir::muse::populateMuseTypeConversions(*this);
    mlir::muse::populateMuseTypeMaterializations(*this);
    populateTorchToMuseTypeConversions(*this);
  }
};

// Pattern to remove torch_c conversion ops (they become identity/casts in Muse)
template <typename OpTy>
class TorchConversionOpToMusePattern : public mlir::OpConversionPattern<OpTy> {
 public:
  using mlir::OpConversionPattern<OpTy>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1) {
      return mlir::failure();
    }
    rewriter.replaceOp(op, adaptor.getOperand());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertTorchToMusePass : public mlir::PassWrapper<ConvertTorchToMusePass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const final { return "convert-torch-to-muse"; }

  mlir::StringRef getDescription() const final { return "Convert Torch operations to Muse dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::muse::MuseDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::pdl::PDLDialect>();
    registry.insert<mlir::pdl_interp::PDLInterpDialect>();
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    mlir::RewritePatternSet patternList(ctx);
    mlir::registerConversionPDLFunctions(patternList);
    populateGeneratedPDLLPatterns(patternList, mlir::PDLConversionConfig(&converter_));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patternList, converter_);

    // Aten ops
    mlir::populateAtenToMuseConversionPatterns(converter_, patternList);

    // Npu ops
    mlir::populateNpuToMuseConversionPatterns(converter_, patternList);

    // Integer arithmetic ops
    mlir::populateArithToMuseConversionPatterns(converter_, patternList);

    // TorchConversion ops
    patternList.add<TorchConversionOpToMusePattern<mlir::torch::TorchConversion::ToBuiltinTensorOp>,
                    TorchConversionOpToMusePattern<mlir::torch::TorchConversion::FromBuiltinTensorOp>>(converter_, ctx);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<TorchD::TorchDialect>();
    target.addIllegalOp<mlir::torch::TorchConversion::ToBuiltinTensorOp,
                        mlir::torch::TorchConversion::FromBuiltinTensorOp>();
    target.addLegalDialect<mlir::muse::MuseDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp op) { return converter_.isSignatureLegal(op.getFunctionType()); });
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
      [&](mlir::func::ReturnOp op) { return converter_.isLegal(op.getOperandTypes()); });

    if (mlir::failed(mlir::applyPartialConversion(module, target, patterns_))) {
      signalPassFailure();
    }
  }

  mlir::FrozenRewritePatternSet patterns_;
  TorchToMuseTypeConverter converter_;
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchToMusePass() { return std::make_unique<ConvertTorchToMusePass>(); }
}  // namespace mlir
