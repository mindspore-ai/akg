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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_GPU_PASSES_H_
#define COMPILER_INCLUDE_AKG_DIALECT_GPU_PASSES_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "akg/Dialect/GPU/Transforms/AKGGPUMapping.h"
#include "akg/Dialect/GPU/Transforms/GpuUseAllReduceWithAtomicReturn.h"
#include "akg/Dialect/GPU/Transforms/GetOrderMapBeforeAfterGpuOutlining.h"
#include "akg/Dialect/GPU/Transforms/GpuKernelOutliningExt.h"
#include "akg/Dialect/GPU/Transforms/StoreAxisInfo.h"
#include "akg/Dialect/GPU/Transforms/LoadAxisInfo.h"

namespace mlir {

/// Generate the code for registering transforms passes.
#define GEN_PASS_REGISTRATION
#include "akg/Dialect/GPU/Passes.h.inc"

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_GPU_PASSES_H_

