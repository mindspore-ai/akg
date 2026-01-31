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
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mfusion/Dialect/Dvm/Dvm.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTMFUSETODVM
#include "mfusion/Conversion/Passes.h.inc"

namespace {

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

struct ConvertMulOp : public OpConversionPattern<mfuse::MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(mfuse::MulOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if (!isDvmOutlinedOp(op.getOperation())) {
      return failure();
    }
    auto attr = dvm::BinaryOpTypeAttr::get(getContext(), dvm::BinaryOpType::Mul);
    rewriter.replaceOpWithNewOp<dvm::BinaryOp>(op, op.getResult().getType(), attr, adaptor.getLhs(), adaptor.getRhs());
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
        if (arg.getType().isa<RankedTensorType>()) {
          auto loadOp = builder.create<dvm::LoadOp>(func.getLoc(), arg.getType(), arg);
          arg.replaceAllUsesExcept(loadOp.getResult(), loadOp);
        }
      }
    }

    func.walk([&](func::ReturnOp returnOp) {
      OpBuilder builder(returnOp);
      for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
        Value operand = returnOp.getOperand(i);
        if (operand.getType().isa<RankedTensorType>()) {
          auto storeOp = builder.create<dvm::StoreOp>(returnOp.getLoc(), operand.getType(), operand);
          returnOp.setOperand(i, storeOp.getResult());
        }
      }
    });
  }
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
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    insertLoadStoreOps(module);

    ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::dvm::DvmDialect>();
    target.addLegalDialect<mlir::func::FuncDialect>();
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<mlir::mfuse::MfuseDialect>();
    target.addDynamicallyLegalOp<mlir::mfuse::MulOp>([](mlir::mfuse::MulOp op) {
      return !isDvmOutlinedOp(op.getOperation());
    });

    RewritePatternSet patterns(ctx);
    patterns.add<ConvertMulOp>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createConvertMfuseToDvmPass() { return std::make_unique<ConvertMfuseToDvmPass>(); }

}  // namespace mlir
