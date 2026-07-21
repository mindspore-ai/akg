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

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmUtils.h"

#include <algorithm>
#include <iterator>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
namespace mfuse {
namespace layernorm_dvm {
namespace {

bool isAllowedBwdTagUser(Operation *user, const llvm::DenseSet<Operation *> &closure) {
  if (!user) {
    return false;
  }
  if (closure.contains(user)) {
    return true;
  }
  return isa<func::ReturnOp>(user);
}

bool opOnlyUsedByOpsInSet(Operation *op, const llvm::DenseSet<Operation *> &closure,
                          llvm::DenseSet<Operation *> *visited = nullptr) {
  if (!op) {
    return false;
  }
  llvm::DenseSet<Operation *> localVisited;
  if (!visited) {
    visited = &localVisited;
  }
  if (!visited->insert(op).second) {
    return true;
  }
  for (Value result : op->getResults()) {
    for (auto &use : result.getUses()) {
      Operation *user = use.getOwner();
      if (!isAllowedBwdTagUser(user, closure)) {
        return false;
      }
      if (closure.contains(user) && !opOnlyUsedByOpsInSet(user, closure, visited)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace

void collectExclusiveBackwardOps(llvm::ArrayRef<Operation *> ops, llvm::SmallVectorImpl<Operation *> &members) {
  llvm::DenseSet<Operation *> closure(ops.begin(), ops.end());
  std::copy_if(ops.begin(), ops.end(), std::back_inserter(members),
               [&](Operation *op) { return op && opOnlyUsedByOpsInSet(op, closure); });
}

bool isOpInFusedIsland(Operation *op) { return op && op->getParentOfType<FusedOp>() != nullptr; }

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir
