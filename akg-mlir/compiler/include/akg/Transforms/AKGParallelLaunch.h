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

#ifndef COMPILER_INCLUDE_AKG_TRANSFORMS_AKGPARALLELLAUNCH_H_
#define COMPILER_INCLUDE_AKG_TRANSFORMS_AKGPARALLELLAUNCH_H_

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {
class Pass;
class ModuleOp;
}  // namespace mlir

static constexpr const char *kParallelLaunchFunc = "AKGBackendParallelLaunch";
namespace mlir {
std::unique_ptr<OperationPass<ModuleOp>> createAKGParallelLaunchPass();
std::unique_ptr<OperationPass<ModuleOp>> createAKGParallelLaunchPass(bool isMindSpore, bool isOutlining);
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_TRANSFORMS_AKGPARALLELLAUNCH_H_
