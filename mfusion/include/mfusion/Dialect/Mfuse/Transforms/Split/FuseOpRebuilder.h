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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPREBUILDER_H_
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPREBUILDER_H_

#include <memory>
#include <vector>
#include <unordered_map>
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"

namespace mlir {
namespace mfuse {
namespace split {
class Rebuilder {
 public:
  Rebuilder(mlir::mfuse::FusedOp fuseOp, const SplitSchemerPtr &splitSchemer,
            const DenseMap<Value, Value> &paramToMainGraphValueMap);
  ~Rebuilder() = default;

  void rebuild();

 private:
  void createFusedOps();
  void connectToMainGraph();
  void createFusedOpForGroup(const SmallVector<Operation *> &groupOps, const llvm::SetVector<Value> &groupInputs,
                             const llvm::SetVector<Value> &groupOutputs, const llvm::DenseSet<Operation *> &groupOpSet);

 private:
  func::FuncOp mainFuncOp_;
  mlir::mfuse::FusedOp fuseOp_;
  SplitSchemerPtr splitSchemer_;
  SmallVector<mlir::mfuse::FusedOp> fusedOpsNeedInline_;
  IRMapping mapping_;
  DenseMap<Value, Value> paramToMainGraphValueMap_;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_SPLIT_FUSEOPREBUILDER_H_
