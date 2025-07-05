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

#include "akg/Analysis/SymbolicShapeAnalysis.h"

#include <algorithm>
#include <iterator>
#include "akg/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#ifndef GEN_PASS_DECL_REMOVESYMBOLIC
#define GEN_PASS_DECL_REMOVESYMBOLIC
#ifndef GEN_PASS_DEF_REMOVESYMBOLIC
#define GEN_PASS_DEF_REMOVESYMBOLIC
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
}  // namespace mlir

using namespace mlir;

static Type RemoveTypeSymbolic(Type type) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (!analysis.hasSymbolicShape(type)) {
    return type;
  }
  return RankedTensorType::get(cast<ShapedType>(type).getShape(), cast<ShapedType>(type).getElementType());
}

static void RemoveFuncSymbolic(func::FuncOp &func) {
  for (auto value : func.getArguments()) {
    value.setType(RemoveTypeSymbolic(value.getType()));
  }
  llvm::SmallVector<Type, 4> newInTys;
  (void)std::transform(func.getArgumentTypes().begin(), func.getArgumentTypes().end(), std::back_inserter(newInTys),
                       [](const Type type) { return RemoveTypeSymbolic(type); });

  llvm::SmallVector<Type, 4> newResTys;
  (void)std::transform(func.getResultTypes().begin(), func.getResultTypes().end(), std::back_inserter(newResTys),
                       [](Type type) { return RemoveTypeSymbolic(type); });
  // update func type
  auto newFuncTy = mlir::FunctionType::get(func.getContext(), newInTys, newResTys);
  func.setType(newFuncTy);
}

namespace {
struct RemoveSymbolic : public impl::RemoveSymbolicBase<RemoveSymbolic> {
  void runOnOperation() override {
    (void)getOperation()->walk([&](Operation *op) {
      if (isa<func::FuncOp>(op)) {
        func::FuncOp func = dyn_cast<func::FuncOp>(op);
        RemoveFuncSymbolic(func);
      }
      for (mlir::Value opnd : op->getOperands()) {
        opnd.setType(RemoveTypeSymbolic(opnd.getType()));
      }
      for (mlir::Value resVal : op->getResults()) {
        resVal.setType(RemoveTypeSymbolic(resVal.getType()));
      }
      return WalkResult::advance();
    });
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> mlir::createSymbolicRemovalPass() { return std::make_unique<RemoveSymbolic>(); }
