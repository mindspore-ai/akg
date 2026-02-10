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

#include "mfusion/Conversion/TorchToMfuse/TorchToMfuse.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/TorchUpstream.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Conversion/MfuseTypeConverter.h"
#include "mfusion/Conversion/TorchToMfuse/TorchAtenToMfuse.h"
#include "mfusion/Conversion/TorchToMfuse/TorchNpuToMfuse.h"
#include "mfusion/Conversion/PdllHelper.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"

#include "TorchToMfuse.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

// Populate Torch-specific type conversions to Mfuse types (built-in RankedTensorType/UnrankedTensorType)
void populateTorchToMfuseTypeConversions(mlir::TypeConverter &converter) {
  converter.addConversion([](TorchD::ValueTensorType type) -> mlir::Type { return type.toBuiltinTensor(); });

  converter.addConversion([](TorchD::IntType type) -> mlir::Type {
    auto elementType = mlir::IntegerType::get(type.getContext(), 64);
    return mlir::RankedTensorType::get({}, elementType);
  });

  converter.addConversion([](TorchD::FloatType type) -> mlir::Type {
    auto elementType = mlir::Float64Type::get(type.getContext());
    return mlir::RankedTensorType::get({}, elementType);
  });

  converter.addConversion([](TorchD::BoolType type) -> mlir::Type {
    auto elementType = mlir::IntegerType::get(type.getContext(), 1);
    return mlir::RankedTensorType::get({}, elementType);
  });

  converter.addConversion(
    [](TorchD::StringType type) -> mlir::Type { return mlir::mfuse::StringType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::DeviceType type) -> mlir::Type { return mlir::mfuse::StringType::get(type.getContext()); });

  converter.addConversion(
    [](TorchD::NoneType type) -> mlir::Type { return mlir::mfuse::NoneType::get(type.getContext()); });

  // Convert !torch.list<int> to tensor<?xi64> for shape-related operations
  // This enables dynamic shape support for reshape, broadcast_to, etc.
  converter.addConversion([&](TorchD::ListType type) -> mlir::Type {
    // For list<int> types (commonly used for shape parameters), convert to 1D tensor
    if (mlir::isa<TorchD::IntType>(type.getContainedType())) {
      return mlir::RankedTensorType::get({mlir::ShapedType::kDynamic}, mlir::IntegerType::get(type.getContext(), 64));
    }
    // For other list types, keep as Mfuse ListType
    return mlir::mfuse::ListType::get(type.getContext(), converter.convertType(type.getContainedType()));
  });
}

// TypeConverter for Torch to Mfuse conversion
class TorchToMfuseTypeConverter : public mlir::TypeConverter {
 public:
  TorchToMfuseTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    mlir::mfuse::populateMfuseTypeConversions(*this);
    mlir::mfuse::populateMfuseTypeMaterializations(*this);
    populateTorchToMfuseTypeConversions(*this);
    addTorchMaterializations();
  }

 private:
  template <typename OpTy>
  static mlir::Value tryConvertConstant(mlir::OpBuilder &builder, mlir::Type toType, mlir::Value input,
                                        mlir::Location loc) {
    if (auto op = input.getDefiningOp<OpTy>()) {
      if (auto ranked = toType.dyn_cast<mlir::RankedTensorType>()) {
        auto denseAttr = mlir::DenseElementsAttr::get(ranked, op.getValueAttr());
        return builder.create<mlir::arith::ConstantOp>(loc, ranked, denseAttr).getResult();
      }
    }
    return {};
  }

  void addTorchMaterializations() {
    // Torch -> builtin/mfuse materialization.
    addTargetMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        mlir::Value input = inputs[0];

        if (auto v = tryConvertConstant<TorchD::ConstantIntOp>(builder, toType, input, loc)) return v;
        if (auto v = tryConvertConstant<TorchD::ConstantFloatOp>(builder, toType, input, loc)) return v;
        if (auto v = tryConvertConstant<TorchD::ConstantBoolOp>(builder, toType, input, loc)) return v;

        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });

    // builtin/mfuse -> Torch materialization.
    addSourceMaterialization(
      [](mlir::OpBuilder &builder, mlir::Type toType, mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1) return {};
        return builder.create<mlir::UnrealizedConversionCastOp>(loc, toType, inputs).getResult(0);
      });
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ConvertTorchToMfusePass : public mlir::PassWrapper<ConvertTorchToMfusePass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const final { return "convert-torch-to-mfuse"; }

  mlir::StringRef getDescription() const final { return "Convert Torch operations to Mfuse dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::arith::ArithDialect>();
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
    // Aten ops
    mlir::populateAtenToMfuseConversionPatterns(converter_, patternList);

    // Npu ops
    mlir::populateNpuToMfuseConversionPatterns(converter_, patternList);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    MLOG(DEBUG) << "convert-torch-to-mfuse pass start";
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::mfuse::MfuseDialect>();
    target.addLegalOp<mlir::UnrealizedConversionCastOp>();
    target.addLegalOp<TorchD::BindSymbolicShapeOp>();
    target.addLegalOp<TorchD::SymbolicIntOp>();
    if (mlir::failed(mlir::applyPartialConversion(module, target, patterns_))) {
      signalPassFailure();
    }
    MLOG(DEBUG) << "convert-torch-to-mfuse pass end";
  }

  mlir::FrozenRewritePatternSet patterns_;
  TorchToMfuseTypeConverter converter_;
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchToMfusePass() { return std::make_unique<ConvertTorchToMfusePass>(); }
}  // namespace mlir
