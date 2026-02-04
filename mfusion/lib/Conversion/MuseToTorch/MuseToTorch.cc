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

#include "mfusion/Conversion/MuseToTorch/MuseToTorch.h"

#include <algorithm>
#include <iterator>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

#include "MuseToTorch.pdll.h.inc"

namespace {

namespace TorchD = mlir::torch::Torch;

/// Converts muse.aclnn.add -> torch.aten.add.Tensor, materializing alpha (rank-0 tensor) as Torch scalar.
class ConvertMuseAclnnAddToTorch : public mlir::OpConversionPattern<mlir::muse::AclnnAddOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::muse::AclnnAddOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value x = adaptor.getX();
    mlir::Value y = adaptor.getY();
    mlir::Value alpha = adaptor.getAlpha();
    // Adaptor may give type-converted operands; look through UnrealizedConversionCastOp to find the source constant.
    mlir::Value alphaForConst = alpha;
    if (auto cast = alpha.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getOperands().size() == 1) {
        alphaForConst = cast.getOperand(0);
      }
    }

    mlir::Value alphaScalar;
    if (auto cst = alphaForConst.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
      if (attr && attr.getType().hasRank() && attr.getType().getRank() == 0 &&
          mlir::isa<mlir::FloatType>(attr.getType().getElementType())) {
        double val = mlir::cast<mlir::FloatAttr>(attr.getValues<mlir::Attribute>()[0]).getValueAsDouble();
        mlir::FloatAttr valueAttr = rewriter.getFloatAttr(rewriter.getF64Type(), val);
        alphaScalar = rewriter.create<TorchD::ConstantFloatOp>(op.getLoc(), valueAttr);
      }
    }
    if (!alphaScalar) {
      return rewriter.notifyMatchFailure(op, "alpha must be a constant rank-0 float tensor");
    }

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();
    rewriter.replaceOpWithNewOp<TorchD::AtenAddTensorOp>(op, resultType, x, y, alphaScalar);
    return mlir::success();
  }
};

/// Converts muse.permute -> torch.aten.transpose.Int.
/// Handles permute operations that swap two dimensions (typically the last two).
class ConvertMusePermuteToTorch : public mlir::OpConversionPattern<mlir::muse::PermuteOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::muse::PermuteOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    auto permAttr = op.getPermAttr();
    if (!permAttr) {
      return rewriter.notifyMatchFailure(op, "perm attribute must be present");
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInput().getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op, "input must be ranked tensor");
    }

    int64_t rank = inputType.getRank();
    auto permValues = permAttr.getValue();
    if (permValues.size() != static_cast<size_t>(rank)) {
      return rewriter.notifyMatchFailure(op, "perm size must match input rank");
    }

    // Extract permutation values
    llvm::SmallVector<int64_t> perm;
    for (auto attr : permValues) {
      if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
        perm.push_back(intAttr.getInt());
      } else {
        return rewriter.notifyMatchFailure(op, "perm values must be integers");
      }
    }

    // Check if this is a simple two-dimension swap (identity except for swapping two dims).
    // Find the two dimensions that are swapped.
    int64_t dim0 = -1, dim1 = -1;
    for (int64_t i = 0; i < rank; ++i) {
      if (perm[i] != i) {
        if (dim0 == -1) {
          dim0 = i;
        } else if (dim1 == -1) {
          dim1 = i;
        } else {
          // More than two dimensions are swapped, cannot use transpose.Int
          return rewriter.notifyMatchFailure(op, "permute swaps more than two dimensions");
        }
      }
    }

    if (dim0 == -1 || dim1 == -1) {
      // Identity permutation, no conversion needed (should be eliminated by canonicalize)
      return rewriter.notifyMatchFailure(op, "identity permutation");
    }

    // Verify that perm[dim0] == dim1 and perm[dim1] == dim0 (swapped)
    if (perm[dim0] != dim1 || perm[dim1] != dim0) {
      return rewriter.notifyMatchFailure(op, "permute is not a simple two-dimension swap");
    }

    mlir::Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto dim0Attr = rewriter.getI64IntegerAttr(dim0);
    auto dim1Attr = rewriter.getI64IntegerAttr(dim1);
    auto dim0Const = rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), dim0Attr);
    auto dim1Const = rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), dim1Attr);

    rewriter.replaceOpWithNewOp<TorchD::AtenTransposeIntOp>(op, resultType, input, dim0Const, dim1Const);
    return mlir::success();
  }
};

/// Converts muse.reshape -> torch.aten.view.
/// Shape is a Value (1D tensor of i64); if constant, extract dims and build list for view.
class ConvertMuseReshapeToTorch : public mlir::OpConversionPattern<mlir::muse::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::muse::ReshapeOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value shapeVal = adaptor.getShape();
    llvm::SmallVector<mlir::Value> shapeValues;

    auto constOp = shapeVal.getDefiningOp<mlir::arith::ConstantOp>();
    if (constOp) {
      auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (denseAttr && denseAttr.getElementType().isInteger(64)) {
        for (auto apInt : denseAttr.getValues<mlir::APInt>()) {
          shapeValues.push_back(
              rewriter.create<TorchD::ConstantIntOp>(op.getLoc(), apInt.getSExtValue()));
        }
      }
    }
    if (shapeValues.empty()) {
      return rewriter.notifyMatchFailure(op, "shape must be a constant 1D i64 tensor");
    }

    auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op.getResult().getType());
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "result must be ranked tensor");
    }

    mlir::Type torchResultType = getTypeConverter()->convertType(resultType);
    if (!torchResultType) return mlir::failure();

    mlir::Value input = adaptor.getInput();
    auto listType = TorchD::ListType::get(op.getContext(), TorchD::IntType::get(op.getContext()));
    auto shapeList = rewriter.create<TorchD::PrimListConstructOp>(op.getLoc(), listType, shapeValues);
    rewriter.replaceOpWithNewOp<TorchD::AtenViewOp>(op, torchResultType, input, shapeList);
    return mlir::success();
  }
};

void populateMuseToTorchTypeConversions(mlir::TypeConverter &converter) {
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
    [](mlir::muse::StringType type) -> mlir::Type { return TorchD::StringType::get(type.getContext()); });
  converter.addConversion(
    [](mlir::muse::NoneType type) -> mlir::Type { return TorchD::NoneType::get(type.getContext()); });
  converter.addConversion([&](mlir::muse::ListType type) -> mlir::Type {
    return TorchD::ListType::get(type.getContext(), converter.convertType(type.getContainedType()));
  });
}

class MuseToTorchTypeConverter : public mlir::TypeConverter {
 public:
  MuseToTorchTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    populateMuseToTorchTypeConversions(*this);

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

struct ConvertMuseToTorchPass : public mlir::PassWrapper<ConvertMuseToTorchPass, mlir::OperationPass<mlir::ModuleOp>> {
  mlir::StringRef getArgument() const final { return "convert-muse-to-torch"; }

  mlir::StringRef getDescription() const final { return "Convert Muse operations to Torch dialect operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::muse::MuseDialect>();
    registry.insert<mlir::func::FuncDialect>();
    registry.insert<mlir::pdl::PDLDialect>();
    registry.insert<mlir::pdl_interp::PDLInterpDialect>();
  }

  mlir::LogicalResult initialize(mlir::MLIRContext *ctx) override {
    mlir::RewritePatternSet patternList(ctx);
    mlir::registerConversionPDLFunctions(patternList);
    patternList.add<ConvertMuseAclnnAddToTorch>(converter_, ctx);
    patternList.add<ConvertMusePermuteToTorch>(converter_, ctx);
    patternList.add<ConvertMuseReshapeToTorch>(converter_, ctx);
    populateGeneratedPDLLPatterns(patternList, mlir::PDLConversionConfig(&converter_));
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(patternList, converter_);

    patterns_ = std::move(patternList);
    return mlir::success();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mlir::ConversionTarget target(*ctx);
    target.addIllegalDialect<mlir::muse::MuseDialect>();
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
  }

  mlir::FrozenRewritePatternSet patterns_;
  MuseToTorchTypeConverter converter_;
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertMuseToTorchPass() { return std::make_unique<ConvertMuseToTorchPass>(); }
}  // namespace mlir
