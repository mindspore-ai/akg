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

#include "mfusion/Dialect/Mfuse/Transforms/Outlining/CopyFusedSubgraphs.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_COPYFUSEDSUBGRAPHS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

static std::string buildCopyName(StringRef baseName) { return (baseName + "_").str(); }

static std::string uniquifyName(StringRef baseName, SymbolTable &symbolTable) {
  std::string name = baseName.str();
  unsigned counter = 0;
  while (symbolTable.lookup(name)) {
    name = (baseName + "_" + std::to_string(++counter)).str();
  }
  return name;
}

struct CopyFusedSubgraphsPass : public impl::CopyFusedSubgraphsBase<CopyFusedSubgraphsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbolTable(module);

    SmallVector<func::FuncOp> outlinedFuncs;
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func->hasAttr(mfusion_attrs::kOutlined)) {
        outlinedFuncs.push_back(func);
      }
    }

    for (auto func : outlinedFuncs) {
      std::string desiredName = buildCopyName(func.getName());
      std::string uniqueName = uniquifyName(desiredName, symbolTable);

      auto cloned = func.clone();
      cloned.setName(uniqueName);
      cloned->removeAttr(mfusion_attrs::kOutlined);

      auto &ops = module.getBody()->getOperations();
      // Insert the cloned subgraph immediately after the original one for easier debugging
      ops.insert(std::next(func.getOperation()->getIterator()), cloned);
      symbolTable.insert(cloned);

      func->setAttr(mfusion_attrs::kCopiedSubgraph, StringAttr::get(module.getContext(), uniqueName));
    }
  }
};

}  // namespace
}  // namespace mfuse

std::unique_ptr<Pass> createCopyFusedSubgraphsPass() { return std::make_unique<mfuse::CopyFusedSubgraphsPass>(); }

}  // namespace mlir
