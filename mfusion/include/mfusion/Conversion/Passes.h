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

#ifndef MFUSION_CONVERSION_PASSES_H
#define MFUSION_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

// Forward declarations
class Pass;

//===----------------------------------------------------------------------===//
// Pass declarations
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "mfusion/Conversion/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass creators
//===----------------------------------------------------------------------===//

/// Create a pass to convert Arith constant operations to Muse constant operations.
std::unique_ptr<Pass> createConvertArithToMusePass();

/// Create a pass to convert Torch operations to Muse dialect operations.
std::unique_ptr<Pass> createConvertTorchToMusePass();

/// Create a pass to convert Muse operations to Torch dialect operations.
std::unique_ptr<Pass> createConvertMuseToTorchPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "mfusion/Conversion/Passes.h.inc"  // NOLINT(build/include)

}  // namespace mlir

#endif  // MFUSION_CONVERSION_PASSES_H
