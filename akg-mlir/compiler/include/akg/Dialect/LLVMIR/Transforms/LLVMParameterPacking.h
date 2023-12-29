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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_LLVMIR_TRANSFORMS_LLVMPARAMETERPACKING_H_
#define COMPILER_INCLUDE_AKG_DIALECT_LLVMIR_TRANSFORMS_LLVMPARAMETERPACKING_H_

#include <memory>

namespace mlir {
class Pass;
namespace LLVM {
// Creates a pass that packs pointer parameters to pointer arrays
std::unique_ptr<Pass> createParameterPackingPass();
std::unique_ptr<Pass> createParameterPackingPass(bool isMindSpore);
}  // namespace LLVM
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_LLVMIR_TRANSFORMS_LLVMPARAMETERPACKING_H_

