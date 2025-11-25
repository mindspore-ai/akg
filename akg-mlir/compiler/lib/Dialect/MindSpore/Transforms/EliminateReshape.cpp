/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "akg/Dialect/MindSpore/Transforms/EliminateReshape.h"

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
#define GEN_PASS_DEF_ELIMINATERESHAPE
#define GEN_PASS_DECL_ELIMINATERESHAPE
#include "akg/Dialect/MindSpore/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace {
constexpr auto kVectorInitSize4 = 4;
llvm::SmallVector<std::string, kVectorInitSize4> RecordFuncArgSymbols(func::FuncOp funcOp) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  llvm::SmallVector<std::string, kVectorInitSize4> funcSymbols;
  for (BlockArgument arg : funcOp.getArguments()) {
    auto symbolShape = analysis.getSymbolicShape(arg.getType());
    if (!symbolShape) {
      continue;
    }
    for (auto symbol : (*symbolShape)) {
      if (!std::all_of(symbol.begin(), symbol.end(), [](char c) { return std::isdigit(c); })) {
        (void)funcSymbols.emplace_back(symbol);
      }
    }
  }
  funcOp.walk([&](func::ReturnOp op) {
    for (auto returnArg : op.getOperation()->getOperands()) {
      auto symbolShape = analysis.getSymbolicShape(returnArg.getType());
      if (!symbolShape) {
        continue;
      }
      for (auto symbol : (*symbolShape)) {
        if (!std::all_of(symbol.begin(), symbol.end(), [](char c) { return std::isdigit(c); })) {
          (void)funcSymbols.emplace_back(symbol);
        }
      }
    }
  });
  return funcSymbols;
}

void updateFuncTypes(func::FuncOp funcOp) {
  llvm::SmallVector<Type, kVectorInitSize4> newInTys;
  for (auto value : funcOp.getBody().front().getArguments()) {
    (void)newInTys.emplace_back(value.getType());
  }
  llvm::SmallVector<Type, kVectorInitSize4> newResTys;
  funcOp.walk([&](func::ReturnOp op) {
    for (auto value : op.getOperation()->getOperands()) {
      (void)newResTys.emplace_back(value.getType());
    }
  });
  auto newFuncTy = mlir::FunctionType::get(funcOp.getContext(), newInTys, newResTys);
  funcOp.setType(newFuncTy);
}

void preprocessReshape(func::FuncOp funcOp) {
  // func arg's ONLY user is ReshapeOp, then modify func arg's shape and erase ReshapeOp

  //  func.func @func_0(%arg0: tensor<16x39x8xf32>, %arg1: tensor<16x39xf32>) -> (tensor<16x39x8xf32>,
  //  tensor<16x312xf32>) attributes {...} {
  //   %0 = "mindspore.reshape"(%arg1) {new_shape = array<i64: 16, 39, 1>} : (tensor<16x39xf32>)
  //        -> tensor<16x39x1xf32>
  //   %1 = "tosa.mul"(%arg0, %0) : (tensor<16x39x8xf32>, tensor<16x39x1xf32>) -> tensor<16x39x8xf32>
  //    ...
  //  }
  //    ------------------>
  //  func.func @func_0(%arg0: tensor<16x39x8xf32>, %arg1: tensor<16x39x1xf32>) -> (tensor<16x39x8xf32>,
  //  tensor<16x312xf32>) attributes {...} {
  //   %0 = "tosa.mul"(%arg0, %arg1) : (tensor<16x39x8xf32>, tensor<16x39x1xf32>) -> tensor<16x39x8xf32>
  //    ...
  //  }
  akgglobal::ShapeAlignTool &tool = akgglobal::ShapeAlignTool::getInstance();
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  size_t argIdx = 0;
  for (BlockArgument arg : funcOp.getArguments()) {
    argIdx++;
    if (!arg.hasOneUse()) {
      continue;
    }
    for (Operation *userOp : arg.getUsers()) {
      if (!(isa<mindspore::ReshapeOp>(userOp) || isa<tosa::ReshapeOp>(userOp))) {
        continue;
      }
      // skip the arg which is the 2nd input of reshape (new_shape), then cannot be removed
      if (userOp->getNumOperands() == 2 && arg == userOp->getOperand(1)) {
        continue;
      }
      // skip the arg whose operands have implicit broadcast shown in needFixIndices
      auto oldNeedFixIndices = tool.getNeedFixIndice(argIdx - 1);
      if (std::any_of(oldNeedFixIndices.begin(), oldNeedFixIndices.end(),
                      [](int64_t needFix) { return (needFix == 1); })) {
        continue;
      }
      // update NeedFixIndices and ShapeInfo for dumpping correct device shapes
      auto newTy = userOp->getResult(0).getType();
      if (analysis.hasSymbolicShape(newTy)) {
        llvm::SmallVector<std::string, kVectorInitSize4> symbolShape = *analysis.getSymbolicShape(newTy);
        std::vector<std::string> newShape;
        (void)std::transform(symbolShape.begin(), symbolShape.end(), std::back_inserter(newShape),
                             [](std::string s) { return s; });
        tool.updateCurrShapeInfo(argIdx - 1, newShape);
      }
      SmallVector<int64_t> newNeedFixIndices{dyn_cast<ShapedType>(newTy).getRank(), 0};
      tool.recordNeedFixIndice(argIdx - 1, newNeedFixIndices);
      // erase ReshapeOp and replace its output with the func arg
      arg.setType(newTy);
      userOp->getResult(0).replaceAllUsesWith(userOp->getOperand(0));
      userOp->erase();
    }
  }

  // return arg's owner is ReshapeOp and ReshapeOp has ONLY owner, then replace the return arg with the value before
  // Reshape and erase ReshapeOp

  //  func.func @func_0(%arg0: tensor<16x39x8xf32>, %arg1: tensor<16x39x1xf32>) -> (tensor<16x39x8xf32>,
  //  tensor<16x312xf32>) attributes {...} {
  //   %0 = "tosa.mul"(%arg0, %arg1) : (tensor<16x39x8xf32>, tensor<16x39x1xf32>) -> tensor<16x39x8xf32>
  //   %1 = "mindspore.reshape"(%0) {new_shape = array<i64: -1, 312>} : (tensor<16x39x8xf32>) -> tensor<16x312xf32>
  //   "func.return"(%0, %1) : (tensor<16x39x8xf32>, tensor<16x312xf32>) -> ()
  //  }
  //    ------------------>
  //  func.func @func_0(%arg0: tensor<16x39x8xf32>, %arg1: tensor<16x39x1xf32>) -> (tensor<16x39x8xf32>,
  //  tensor<16x39x8xf32>) attributes {...} {
  //   %0 = "tosa.mul"(%arg0, %arg1) : (tensor<16x39x8xf32>, tensor<16x39x1xf32>) -> tensor<16x39x8xf32>
  //   "func.return"(%0, %0) : (tensor<16x39x8xf32>, tensor<16x39x8xf32>) -> ()
  //  }

  // record symbol in inputs and outputs
  auto originalFuncSymbols = RecordFuncArgSymbols(funcOp);

  // find ReshapeOp that can be eliminated
  funcOp.walk([&](func::ReturnOp op) {
    for (auto returnArg : op.getOperation()->getOperands()) {
      argIdx++;
      if (auto ownerOp = returnArg.getDefiningOp()) {
        if (!(isa<mindspore::ReshapeOp>(ownerOp) || isa<tosa::ReshapeOp>(ownerOp))) {
          continue;
        }
        if (!ownerOp->getResult(0).hasOneUse()) {
          continue;
        }
        auto reshapeInput = ownerOp->getOperand(0);
        auto newTy = reshapeInput.getType();
        auto symbolShape = analysis.getSymbolicShape(newTy);
        auto hasSymbolNotInOriginalFuncArgs = false;
        if (symbolShape) {
          for (auto symbol : (*symbolShape)) {
            // A Reshape op cannot be rmeoved if exposing a symbol not in the original inputs and outputs
            if (std::find(originalFuncSymbols.begin(), originalFuncSymbols.end(), symbol) ==
                originalFuncSymbols.end()) {
              hasSymbolNotInOriginalFuncArgs = true;
            }
          }
        }
        if (hasSymbolNotInOriginalFuncArgs) {
          continue;
        }
        auto reshapeOutput = ownerOp->getResult(0);
        auto oldTy = reshapeOutput.getType();
        tool.alignStaticShapeReconstruct(argIdx - 1, oldTy, newTy);
        reshapeOutput.replaceAllUsesWith(reshapeInput);
        ownerOp->erase();
      }
    }
  });

  updateFuncTypes(funcOp);
}

struct EliminateReshape : public impl::EliminateReshapeBase<EliminateReshape> {
  EliminateReshape() {}
  void runOnOperation() override { preprocessReshape(getOperation()); }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createEliminateReshapePass() {
  return std::make_unique<EliminateReshape>();
}
