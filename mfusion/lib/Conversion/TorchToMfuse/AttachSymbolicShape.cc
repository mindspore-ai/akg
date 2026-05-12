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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"

namespace {
namespace TorchD = mlir::torch::Torch;

mlir::Attribute dropSymbolicShapeEncoding(mlir::RankedTensorType type) {
  auto encoding = type.getEncoding();
  if (!encoding) {
    return {};
  }
  if (mlir::mfuse::SymbolAttrUtils::isSymbolicShapeEncoding(encoding)) {
    return {};
  }

  auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(encoding);
  if (!dict) {
    return encoding;
  }

  mlir::MLIRContext *ctx = type.getContext();
  auto symKey = mlir::StringAttr::get(ctx, mlir::mfuse::SymbolAttrUtils::kSymShapeKey);
  auto baseKey = mlir::StringAttr::get(ctx, mlir::mfuse::SymbolAttrUtils::kBaseEncodingKey);
  llvm::SmallVector<mlir::NamedAttribute> entries;
  entries.reserve(dict.getValue().size());
  for (const auto &entry : dict.getValue()) {
    if (entry.getName() != symKey) {
      entries.push_back(entry);
    }
  }

  if (entries.empty()) {
    return {};
  }
  if (entries.size() == 1 && entries.front().getName() == baseKey) {
    return entries.front().getValue();
  }
  return mlir::DictionaryAttr::get(ctx, entries);
}

bool setValueType(mlir::Value value, mlir::RankedTensorType newType) {
  if (auto result = mlir::dyn_cast<mlir::OpResult>(value)) {
    if (result.getType() != newType) {
      result.setType(newType);
    }
    return true;
  }
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (arg.getType() != newType) {
      arg.setType(newType);
    }
    return true;
  }
  return false;
}

mlir::FailureOr<mlir::RankedTensorType> canonicalizeTypeWithSymbolicExprs(
  mlir::Operation *diagOp, mlir::RankedTensorType ranked, llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
  mlir::OpBuilder &builder, mfusion::SymEngineAnalysis &analysis) {
  if (ranked.getRank() != static_cast<int64_t>(exprs.size())) {
    diagOp->emitError("bind_symbolic_shape result rank does not match ranked tensor type rank");
    return mlir::failure();
  }

  llvm::SmallVector<int64_t> newShape(ranked.getShape().begin(), ranked.getShape().end());
  bool hasDynamicExpr = false;
  for (int64_t i = 0; i < ranked.getRank(); ++i) {
    auto maybeInt = analysis.tryExtractInt64(exprs[static_cast<size_t>(i)]);
    if (mlir::succeeded(maybeInt)) {
      if (!mlir::ShapedType::isDynamic(newShape[static_cast<size_t>(i)]) &&
          newShape[static_cast<size_t>(i)] != *maybeInt) {
        diagOp->emitError() << "static tensor dimension " << i << " is " << newShape[static_cast<size_t>(i)]
                            << " but bind_symbolic_shape expression is " << *maybeInt;
        return mlir::failure();
      }
      newShape[static_cast<size_t>(i)] = *maybeInt;
      continue;
    }

    if (!mlir::ShapedType::isDynamic(newShape[static_cast<size_t>(i)])) {
      diagOp->emitError() << "static tensor dimension " << i << " is " << newShape[static_cast<size_t>(i)]
                          << " but bind_symbolic_shape expression is non-constant: "
                          << exprs[static_cast<size_t>(i)]->__str__();
      return mlir::failure();
    }
    hasDynamicExpr = true;
  }

  auto baseEncoding = hasDynamicExpr ? ranked.getEncoding() : dropSymbolicShapeEncoding(ranked);
  auto baseType = mlir::RankedTensorType::get(newShape, ranked.getElementType(), baseEncoding);
  if (!hasDynamicExpr) {
    return baseType;
  }
  return mlir::mfuse::SymbolAttrUtils::withSymbolicAttr(
    baseType, mlir::mfuse::SymbolAttrUtils::createSymbolicShapeAttr(builder, exprs));
}

mlir::LogicalResult attachToCastResults(mlir::Operation *diagOp, mlir::Operation *castOp,
                                        llvm::ArrayRef<mlir::mfuse::SymbolAttrUtils::SymExpr> exprs,
                                        mlir::OpBuilder &builder, mfusion::SymEngineAnalysis &analysis) {
  for (mlir::Value result : castOp->getResults()) {
    auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!ranked) {
      continue;
    }
    if (ranked.getRank() != static_cast<int64_t>(exprs.size())) {
      continue;
    }
    auto canonicalType = canonicalizeTypeWithSymbolicExprs(diagOp, ranked, exprs, builder, analysis);
    if (mlir::failed(canonicalType)) {
      return mlir::failure();
    }
    result.setType(*canonicalType);
  }
  return mlir::success();
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
    mlir::OpBuilder builder(ctx);

    // Resolver for Torch dialect symbolic integers.
    mfusion::SymEngineAnalysis::SymbolNameResolver torchResolver = [](mlir::Value v) -> mlir::FailureOr<std::string> {
      if (auto symIntOp = v.getDefiningOp<TorchD::SymbolicIntOp>()) {
        return symIntOp.getSymbolName().str();
      }
      return mlir::failure();
    };

    llvm::SmallVector<mlir::Operation *> eraseOps;
    for (auto bindOp : bindOps) {
      auto affineMap = bindOp.getShapeExpressions().getValue();
      auto exprs = analysis.applyAffineMap(affineMap, bindOp.getShapeSymbols(), torchResolver);
      if (mlir::failed(exprs)) {
        bindOp.emitError("unsupported symbolic dim source; expected torch.symbolic_int values");
        return mlir::failure();
      }

      // Case 1: Torch tensor used by casts to RankedTensorType.
      for (mlir::Operation *user : bindOp.getOperand().getUsers()) {
        auto castOp = llvm::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
        if (!castOp) {
          continue;
        }
        if (mlir::failed(attachToCastResults(bindOp, castOp, *exprs, builder, analysis))) {
          return mlir::failure();
        }
      }

      // Case 2: Mfuse op result cast to Torch tensor, then bound by bind_symbolic_shape.
      // after this case, the bind_symbolic_shape and the unrealized_conversion_cast is no longer needed.
      // unrealized_conversion_cast will be removed by the post conversion pipeline.
      bool remove_op = false;
      if (auto castOp = bindOp.getOperand().getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        for (mlir::Value input : castOp.getInputs()) {
          auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
          if (!ranked || ranked.getRank() != static_cast<int64_t>(exprs->size())) {
            continue;
          }
          auto canonicalType = canonicalizeTypeWithSymbolicExprs(bindOp, ranked, *exprs, builder, analysis);
          if (mlir::failed(canonicalType)) {
            return mlir::failure();
          }
          if (setValueType(input, *canonicalType)) {
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
