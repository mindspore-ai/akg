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
#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"

namespace {
namespace TorchD = mlir::torch::Torch;

mlir::Attribute mergeEncoding(mlir::RankedTensorType type, mlir::muse::SymbolicShapeAttr attr) {
  auto encoding = type.getEncoding();
  if (!encoding || encoding.isa<mlir::muse::SymbolicShapeAttr>()) {
    return attr;
  }

  mlir::MLIRContext *ctx = type.getContext();
  auto symKey = mlir::StringAttr::get(ctx, "muse.symshape");
  if (auto dict = encoding.dyn_cast<mlir::DictionaryAttr>()) {
    auto existing = dict.get(symKey);
    if (existing == attr) {
      return dict;
    }

    llvm::SmallVector<mlir::NamedAttribute> entries;
    entries.reserve(dict.getValue().size() + 1);
    bool replaced = false;
    for (const auto &entry : dict.getValue()) {
      if (entry.getName() == symKey) {
        entries.emplace_back(symKey, attr);
        replaced = true;
        continue;
      }
      entries.push_back(entry);
    }
    if (!replaced) {
      entries.emplace_back(symKey, attr);
    }
    return mlir::DictionaryAttr::get(ctx, entries);
  }

  auto baseKey = mlir::StringAttr::get(ctx, "muse.encoding");
  return mlir::DictionaryAttr::get(ctx, {mlir::NamedAttribute(symKey, attr), mlir::NamedAttribute(baseKey, encoding)});
}

mlir::RankedTensorType withSymbolicAttr(mlir::RankedTensorType type, mlir::muse::SymbolicShapeAttr attr) {
  auto merged = mergeEncoding(type, attr);
  if (merged == type.getEncoding()) {
    return type;
  }
  return mlir::RankedTensorType::get(type.getShape(), type.getElementType(), merged);
}

bool attachToValue(mlir::Value value, mlir::muse::SymbolicShapeAttr attr) {
  auto ranked = value.getType().dyn_cast<mlir::RankedTensorType>();
  if (!ranked) {
    return false;
  }
  auto newType = withSymbolicAttr(ranked, attr);
  if (newType == ranked) {
    return true;
  }

  if (auto result = value.dyn_cast<mlir::OpResult>()) {
    result.setType(newType);
    return true;
  }
  if (auto arg = value.dyn_cast<mlir::BlockArgument>()) {
    arg.setType(newType);
    return true;
  }
  return false;
}

bool attachToCastResults(mlir::Operation *castOp, mlir::muse::SymbolicShapeAttr attr, unsigned exprCount) {
  bool updated = false;
  for (mlir::Value result : castOp->getResults()) {
    auto ranked = result.getType().dyn_cast<mlir::RankedTensorType>();
    if (!ranked) {
      continue;
    }
    if (ranked.getRank() != static_cast<int64_t>(exprCount)) {
      continue;
    }
    result.setType(withSymbolicAttr(ranked, attr));
    updated = true;
  }
  return updated;
}

struct ConvertTorchSymbolToMusePass
    : public mlir::PassWrapper<ConvertTorchSymbolToMusePass, mlir::OperationPass<mlir::ModuleOp>> {
  ConvertTorchSymbolToMusePass() = default;
  ConvertTorchSymbolToMusePass(const ConvertTorchSymbolToMusePass &pass) : PassWrapper(pass) {}

  Option<bool> removeAllBindOps{*this, "remove_all_bind_ops",
                                llvm::cl::desc("Remove all torch.bind_symbolic_shape ops"), llvm::cl::init(true)};

  mlir::StringRef getArgument() const final { return "convert-torch-symbol-to-muse"; }
  mlir::StringRef getDescription() const final {
    return "Attach symbolic shape attributes after Torch-to-Muse conversion";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
    registry.insert<mlir::muse::MuseDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = &getContext();

    mfusion::SymEngineAnalysis analysis;
    llvm::SmallVector<TorchD::BindSymbolicShapeOp> bindOps;
    module.walk([&](TorchD::BindSymbolicShapeOp op) { bindOps.push_back(op); });

    llvm::SmallVector<mlir::Operation *> eraseOps;
    for (auto bindOp : bindOps) {
      auto affineMap = bindOp.getShapeExpressions().getValue();
      auto exprs = analysis.applyAffineMap(affineMap, bindOp.getShapeSymbols());
      if (mlir::failed(exprs)) {
        bindOp.emitError("unsupported symbolic dim source; expected torch.symbolic_int values");
        signalPassFailure();
        return;
      }

      llvm::SmallVector<mlir::Attribute> exprAttrs(exprs->size());
      std::transform(exprs->begin(), exprs->end(), exprAttrs.begin(),
                     [ctx](const auto &expr) { return mlir::StringAttr::get(ctx, expr->__str__()); });
      auto shapeAttr = mlir::muse::SymbolicShapeAttr::get(ctx, mlir::ArrayAttr::get(ctx, exprAttrs));
      unsigned exprCount = static_cast<unsigned>(exprs->size());

      // Case 1: Torch tensor used by casts to RankedTensorType.
      for (mlir::Operation *user : bindOp.getOperand().getUsers()) {
        auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
        if (!castOp) {
          continue;
        }
        attachToCastResults(castOp, shapeAttr, exprCount);
      }

      // Case 2: Muse op result cast to Torch tensor, then bound by bind_symbolic_shape.
      // after this case, the bind_symbolic_shape and the unrealized_conversion_cast is no longer needed.
      // unrealized_conversion_cast will be removed by the post conversion pipeline.
      bool remove_op = false;
      if (auto castOp = bindOp.getOperand().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        for (mlir::Value input : castOp.getInputs()) {
          auto ranked = input.getType().dyn_cast<mlir::RankedTensorType>();
          if (!ranked || ranked.getRank() != static_cast<int64_t>(exprCount)) {
            continue;
          }
          if (attachToValue(input, shapeAttr)) {
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
  }
};
}  // namespace

namespace mlir {
std::unique_ptr<Pass> createConvertTorchSymbolToMusePass() { return std::make_unique<ConvertTorchSymbolToMusePass>(); }
}  // namespace mlir
