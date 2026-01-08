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

#include "mfusion/Conversion/StandardToMuse/ArithToMuse.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Conversion/MuseTypeConverter.h"
#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"

namespace {

using mlir::cast;
using mlir::ConversionPatternRewriter;
using mlir::ConversionTarget;
using mlir::dyn_cast;
using mlir::failure;
using mlir::isa;
using mlir::LogicalResult;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpConversionPattern;
using mlir::OperationPass;
using mlir::Pass;
using mlir::PassWrapper;
using mlir::RewritePatternSet;
using mlir::StringRef;
using mlir::success;
using mlir::TypeConverter;

//===----------------------------------------------------------------------===//
// Arith to Muse Conversion Patterns
//===----------------------------------------------------------------------===//

// Convert arith.constant to muse.constant.xxx
struct ConvertArithConstantOp : public OpConversionPattern<mlir::arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto value = op.getValue();
    auto *ctx = rewriter.getContext();

    // Handle integer constants
    if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value)) {
      auto intType = cast<mlir::IntegerType>(intAttr.getType());
      unsigned width = intType.getWidth();

      if (width == 1) {
        auto boolAttr = rewriter.getBoolAttr(intAttr.getValue().getBoolValue());
        rewriter.replaceOpWithNewOp<mlir::muse::CreateBooleanOp>(op, mlir::muse::BooleanType::get(ctx), boolAttr);
        return success();
      }
      if (width <= 64) {
        auto i64Attr = rewriter.getI64IntegerAttr(intAttr.getValue().getSExtValue());
        rewriter.replaceOpWithNewOp<mlir::muse::CreateI64Op>(op, mlir::muse::I64Type::get(ctx), i64Attr);
        return success();
      }
      return failure();
    }

    // Handle float constants
    if (auto floatAttr = dyn_cast<mlir::FloatAttr>(value)) {
      auto floatType = floatAttr.getType();

      if (floatType.isF64()) {
        rewriter.replaceOpWithNewOp<mlir::muse::CreateF64Op>(op, mlir::muse::F64Type::get(ctx), floatAttr);
        return success();
      }
      if (floatType.isF16() || floatType.isBF16() || floatType.isF32()) {
        auto f64Attr = rewriter.getF64FloatAttr(floatAttr.getValue().convertToDouble());
        rewriter.replaceOpWithNewOp<mlir::muse::CreateF64Op>(op, mlir::muse::F64Type::get(ctx), f64Attr);
        return success();
      }
      return failure();
    }

    // Handle index type (treat as i64)
    if (isa<mlir::IndexType>(op.getType())) {
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(value)) {
        auto i64Attr = rewriter.getI64IntegerAttr(intAttr.getValue().getSExtValue());
        rewriter.replaceOpWithNewOp<mlir::muse::CreateI64Op>(op, mlir::muse::I64Type::get(ctx), i64Attr);
        return success();
      }
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct ConvertArithToMusePass : public PassWrapper<ConvertArithToMusePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertArithToMusePass)

  StringRef getArgument() const final { return "convert-arith-to-muse"; }

  StringRef getDescription() const final { return "Convert Arith constant operations to Muse constant operations"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::muse::MuseDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    TypeConverter typeConverter;
    typeConverter.addConversion([](mlir::Type type) { return type; });
    mlir::muse::populateMuseScalarTypeConversions(typeConverter);

    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::muse::MuseDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();

    RewritePatternSet patterns(ctx);
    mlir::muse::populateArithToMuseConversionPatterns(patterns, typeConverter, target);

    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

namespace mlir {
namespace muse {

void populateArithToMuseConversionPatterns(mlir::RewritePatternSet &patterns, mlir::TypeConverter &typeConverter,
                                           mlir::ConversionTarget &target) {
  // Mark arith.constant with scalar types as illegal
  target.addDynamicallyLegalOp<mlir::arith::ConstantOp>([](mlir::arith::ConstantOp op) {
    auto type = op.getType();
    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::IndexType>(type)) {
      return false;  // illegal, needs conversion
    }
    return true;  // legal (e.g., tensor constants)
  });

  patterns.add<ConvertArithConstantOp>(typeConverter, patterns.getContext());
}

}  // namespace muse

std::unique_ptr<Pass> createConvertArithToMusePass() { return std::make_unique<ConvertArithToMusePass>(); }

}  // namespace mlir
