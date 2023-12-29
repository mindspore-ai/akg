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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_GPU_TRANSFORMS_GPUKERNELOUTLININGEXT_H_
#define COMPILER_INCLUDE_AKG_DIALECT_GPU_TRANSFORMS_GPUKERNELOUTLININGEXT_H_

#include <memory>
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Pass/Pass.h"

namespace llvm {
class TargetMachine;
class LLVMContext;
class Module;
}  // namespace llvm

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func

/// Replaces `gpu.launch` with `gpu.launch_func` by moving the region into
/// a separate kernel function.
std::unique_ptr<OperationPass<ModuleOp>> createGpuKernelOutliningExt(StringRef dataLayoutStr = StringRef());

}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_DIALECT_GPU_TRANSFORMS_GPUKERNELOUTLININGEXT_H_

