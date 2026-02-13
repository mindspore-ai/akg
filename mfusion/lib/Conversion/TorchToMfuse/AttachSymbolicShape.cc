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

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "mfusion/Analysis/SymbolicShape/SymEngineAnalysis.h"
#include "mfusion/Conversion/Passes.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Utils/SymbolicShapeUtils.h"

namespace {
namespace TorchD = mlir::torch::Torch;

bool attachToCastResults(mlir::Operation *castOp, mlir::mfuse::SymbolicShapeAttr attr, unsigned exprCount) {
  bool updated = false;
  for (mlir::Value result : castOp->getResults()) {
    auto ranked = result.getType().dyn_cast<mlir::RankedTensorType>();
    if (!ranked) {
      continue;
    }
    if (ranked.getRank() != static_cast<int64_t>(exprCount)) {
      continue;
    }
    result.setType(mlir::mfuse::SymbolicShapeUtils::withSymbolicAttr(ranked, attr));
    updated = true;
  }
  return updated;
}

struct ConvertTorchSymbolToMfusePass
    : public mlir::PassWrapper<ConvertTorchSymbolToMfusePass, mlir::OperationPass<mlir::ModuleOp>> {
  ConvertTorchSymbolToMfusePass() = default;
  ConvertTorchSymbolToMfusePass(const ConvertTorchSymbolToMfusePass &pass) : PassWrapper(pass) {}

  Option<bool> removeAllBindOps{*this, "remove_all_bind_ops",
                                llvm::cl::desc("Remove all torch.bind_symbolic_shape ops"), llvm::cl::init(true)};

  mlir::StringRef getArgument() const final { return "convert-torch-symbol-to-mfuse"; }
  mlir::StringRef getDescription() const final {
    return "Attach symbolic shape attributes after Torch-to-Mfuse conversion";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::mfuse::MfuseDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mfusion::SymEngineAnalysis analysis;
    handleSymbolicIntOps(module, ctx);
    if (mlir::failed(handleBindSymbolicShapeOps(module, ctx, analysis))) {
      signalPassFailure();
    }
  }

 private:
  void handleSymbolicIntOps(mlir::ModuleOp module, mlir::MLIRContext *ctx) {
    llvm::SmallVector<TorchD::SymbolicIntOp> symIntOps;
    module.walk([&](TorchD::SymbolicIntOp op) { symIntOps.push_back(op); });

    llvm::DenseMap<mlir::func::FuncOp, llvm::SmallVector<mlir::NamedAttribute>> funcSymInfo;
    for (auto symOp : symIntOps) {
      auto func = symOp->getParentOfType<mlir::func::FuncOp>();
      if (!func) {
        continue;
      }
      auto nameAttr = mlir::StringAttr::get(ctx, symOp.getSymbolName());
      auto infoAttr = mlir::mfuse::SymbolInfoAttr::get(ctx, symOp.getMinVal(), symOp.getMaxVal());
      funcSymInfo[func].emplace_back(nameAttr, infoAttr);
    }

    for (auto &entry : funcSymInfo) {
      auto func = entry.first;
      auto existing = func->getAttrOfType<mlir::DictionaryAttr>("mfuse.syminfo");
      llvm::SmallVector<mlir::NamedAttribute> merged;
      if (existing) {
        merged.assign(existing.getValue().begin(), existing.getValue().end());
      }

      for (const auto &attr : entry.second) {
        auto it = std::find_if(merged.begin(), merged.end(),
                               [&](const mlir::NamedAttribute &item) { return item.getName() == attr.getName(); });
        if (it == merged.end()) {
          merged.push_back(attr);
        } else if (it->getValue() != attr.getValue()) {
          *it = attr;
        }
      }

      func->setAttr("mfuse.syminfo", mlir::DictionaryAttr::get(ctx, merged));
    }
  }

  mlir::LogicalResult handleBindSymbolicShapeOps(mlir::ModuleOp module, mlir::MLIRContext *ctx,
                                                 mfusion::SymEngineAnalysis &analysis) {
    llvm::SmallVector<TorchD::BindSymbolicShapeOp> bindOps;
    module.walk([&](TorchD::BindSymbolicShapeOp op) { bindOps.push_back(op); });

    llvm::SmallVector<mlir::Operation *> eraseOps;
    for (auto bindOp : bindOps) {
      auto affineMap = bindOp.getShapeExpressions().getValue();
      auto exprs = analysis.applyAffineMap(affineMap, bindOp.getShapeSymbols());
      if (mlir::failed(exprs)) {
        bindOp.emitError("unsupported symbolic dim source; expected torch.symbolic_int values");
        return mlir::failure();
      }

      llvm::SmallVector<mlir::Attribute> exprAttrs(exprs->size());
      std::transform(exprs->begin(), exprs->end(), exprAttrs.begin(),
                     [ctx](const auto &expr) { return mlir::StringAttr::get(ctx, expr->__str__()); });
      auto shapeAttr = mlir::mfuse::SymbolicShapeAttr::get(ctx, mlir::ArrayAttr::get(ctx, exprAttrs));
      unsigned exprCount = static_cast<unsigned>(exprs->size());

      // Case 1: Torch tensor used by casts to RankedTensorType.
      for (mlir::Operation *user : bindOp.getOperand().getUsers()) {
        auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
        if (!castOp) {
          continue;
        }
        attachToCastResults(castOp, shapeAttr, exprCount);
      }

      // Case 2: Mfuse op result cast to Torch tensor, then bound by bind_symbolic_shape.
      // after this case, the bind_symbolic_shape and the unrealized_conversion_cast is no longer needed.
      // unrealized_conversion_cast will be removed by the post conversion pipeline.
      bool remove_op = false;
      if (auto castOp = bindOp.getOperand().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        for (mlir::Value input : castOp.getInputs()) {
          auto ranked = input.getType().dyn_cast<mlir::RankedTensorType>();
          if (!ranked || ranked.getRank() != static_cast<int64_t>(exprCount)) {
            continue;
          }
          if (mlir::mfuse::SymbolicShapeUtils::attachToValue(input, shapeAttr)) {
            remove_op = true;
          }
        }
      }

      if (removeAllBindOps || remove_op) {
        eraseOps.push_back(bindOp);
      }
    }

    for (mlir::Operation *op : eraseOps) {
      op->erase();
    }
    return mlir::success();
  }
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchSymbolToMfusePass() {
  return std::make_unique<ConvertTorchSymbolToMfusePass>();
}
}  // namespace mlir
