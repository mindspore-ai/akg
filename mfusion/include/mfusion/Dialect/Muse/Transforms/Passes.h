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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H
#define MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Pass declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass creators
//===----------------------------------------------------------------------===//

/// Create a pass to set device information for tensors in MUSE dialect.
std::unique_ptr<Pass> createSetTensorDevice();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"  // NOLINT(build/include)

}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_TRANSFORMS_PASSES_H
