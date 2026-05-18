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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_PASSES_CONV_FUSE_BIASADD_CONV_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_PASSES_CONV_FUSE_BIASADD_CONV_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace mfuse {

/// Fuse Conv2D (no bias) and Add (bias) into Conv2DWithBias.
/// Constraints: Conv2D must not already have bias; no dynamic shapes.
std::unique_ptr<Pass> createFuseBiasaddConvPass();

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_FUSION_PASSES_CONV_FUSE_BIASADD_CONV_H
