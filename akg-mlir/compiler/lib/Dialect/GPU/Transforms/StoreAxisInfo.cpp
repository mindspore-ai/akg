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

#include "akg/Dialect/GPU/Transforms/StoreAxisInfo.h"
#include <deque>

#include "akg/Utils/AKGGlobalVars.hpp"

#include "llvm/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
#define GEN_PASS_DEF_STOREAXISINFO
#define GEN_PASS_DECL_STOREAXISINFO
#include "akg/Dialect/GPU/Passes.h.inc"
}  // namespace mlir

using namespace akgglobal;
namespace mlir {
namespace gpu {

namespace {
struct StoreAxisInfoPass : public impl::StoreAxisInfoBase<StoreAxisInfoPass> {
  StoreAxisInfoPass() {}
  void runOnOperation() override {
    std::deque<std::string> newTags;
    getOperation()->walk([&](Operation *op) {
      if (!isa<affine::AffineForOp, affine::AffineParallelOp>(op)) {
        return;
      }
      if (op->getAttr(kLoopTag)) {
        auto loopTag = op->getAttr(kLoopTag).dyn_cast<StringAttr>().getValue().str();
        newTags.push_front(loopTag);
      } else {
        newTags.push_front(kPlaceHolder);
      }
    });
    std::vector<std::string> newStructure;
    (void)std::transform(newTags.begin(), newTags.end(), std::back_inserter(newStructure),
                         [](std::string l) { return l; });
    GpuScheduleTool::getInstance().updateLoopStructure(newStructure);
  }
};
}  // namespace
}  // namespace gpu
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createStoreAxisInfoPass() {
  return std::make_unique<gpu::StoreAxisInfoPass>();
}
