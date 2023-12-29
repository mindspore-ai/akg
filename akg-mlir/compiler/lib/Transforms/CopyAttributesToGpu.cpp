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

#include "akg/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include <string>
#include <unordered_set>
namespace mlir {
#ifndef GEN_PASS_DECL_COPYATTRIBUTESTOGPU
#define GEN_PASS_DECL_COPYATTRIBUTESTOGPU
#ifndef GEN_PASS_DEF_COPYATTRIBUTESTOGPU
#define GEN_PASS_DEF_COPYATTRIBUTESTOGPU
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Transforms/Passes.h.inc"
#endif
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "copy-attributes-to-gpu"

using namespace mlir;

namespace {

// ===----------------------------------------------------------------------===//
// CopyAttributesToGpuPass
// This pass will copy all the attributes of funcOp to GPUFuncOp
// ===----------------------------------------------------------------------===//

struct CopyAttributesToGpuPass : public CopyAttributesToGpuBase<CopyAttributesToGpuPass> {
 public:
  void runOnOperation() override;
  std::unordered_set<std::string> excludedKeys{"function_type", "sym_name"};
};

void CopyAttributesToGpuPass::runOnOperation() {
  Operation *gpuFuncOp = nullptr;
  Operation *funcOp = nullptr;
  getOperation()->walk([&](Operation *op) {
    if (isa<gpu::GPUFuncOp>(op)) {
      gpuFuncOp = op;
    } else if (isa<mlir::func::FuncOp>(op)) {
      funcOp = op;
    }
  });
  if (funcOp && gpuFuncOp) {
    auto attrs = funcOp->getAttrs();
    for (auto attr : attrs) {
      auto keyStr = attr.getName().dyn_cast<StringAttr>().getValue().str();
      if (excludedKeys.count(keyStr) != 0) {
        continue;
      }
      gpuFuncOp->setAttr(attr.getName(), attr.getValue());
    }
  }
}
}  // end anonymous namespace

std::unique_ptr<Pass> mlir::createCopyAttributesToGpuPass() { return std::make_unique<CopyAttributesToGpuPass>(); }
