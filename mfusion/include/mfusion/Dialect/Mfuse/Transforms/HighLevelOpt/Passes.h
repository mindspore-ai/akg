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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mfuse {

/// Create a pass to reorder Cast and type-insensitive operations.
std::unique_ptr<Pass> createReorderOpsPass();

/// Create a pass to convert bfloat16 inputs to float32 for operations that don't support bfloat16.
std::unique_ptr<Pass> createConvertBFloat16Pass();

/// Create a pass to raise reduction precision from float16 to float32.
std::unique_ptr<Pass> createRaiseReductionPrecisionPass();

/// Create a pass to promote binary operations to ensure consistent input types.
std::unique_ptr<Pass> createPromoteBinaryOpsPass();

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_HIGH_LEVEL_OPT_PASSES_H
