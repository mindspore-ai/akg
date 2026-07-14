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

#ifndef MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMUTILS_H
#define MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMUTILS_H

#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace mfuse {
namespace layernorm_dvm {

/// Collect backward operations whose transitive users stay inside the matched closure.
void collectExclusiveBackwardOps(llvm::ArrayRef<Operation *> ops, llvm::SmallVectorImpl<Operation *> &members);

bool isOpInFusedIsland(Operation *op);

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMUTILS_H
