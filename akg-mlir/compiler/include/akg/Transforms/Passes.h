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

#ifndef COMPILER_INCLUDE_AKG_TRANSFORMS_PASSES_H_
#define COMPILER_INCLUDE_AKG_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>
#include "akg/Transforms/AKGFuncOutlining.h"
#include "akg/Transforms/AKGParallelLaunch.h"
#include "akg/Transforms/CopyElision.h"
#include "akg/Transforms/CopyRemoval.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createLoadGlobalConfigPass(const std::string &fileName);
std::unique_ptr<Pass> createLoadGlobalConfigPass();
std::unique_ptr<Pass> createDumpShapeInfoPass(const std::string &fileName);
std::unique_ptr<Pass> createDumpShapeInfoPass();
std::unique_ptr<Pass> createCopyAttributesToGpuPass();
std::unique_ptr<Pass> createPromoteTempBufferPass();
std::unique_ptr<Pass> createStoreLoadElimPass();
std::unique_ptr<Pass> createSymbolRemovalPass();
std::unique_ptr<Pass> createInferSymbolicShapesPass();
std::unique_ptr<Pass> createSymbolicRemovalPass();
std::unique_ptr<OperationPass<ModuleOp>> createAKGFuncOutliningPass();
std::unique_ptr<OperationPass<ModuleOp>> createAKGFuncOutliningPass(bool isMindSpore, bool isOutlining);
std::unique_ptr<OperationPass<ModuleOp>> createAKGParallelLaunchPass();
std::unique_ptr<OperationPass<ModuleOp>> createAKGParallelLaunchPass(bool isMindSpore, bool isOutlining);

namespace func {
class FuncOp;
}  // namespace func

/// Generate the code for registering transforms passes.
#ifndef GEN_PASS_REGISTRATION
#define GEN_PASS_REGISTRATION
#include "akg/Transforms/Passes.h.inc"
#endif

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_TRANSFORMS_PASSES_H_
