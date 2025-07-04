/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_
#include <limits>
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
}  // namespace func

namespace affine {
class AffineForOp;

/// Performs packing (or explicit copying) of accessed memref regions into
/// buffers in the specified faster memory space through either pointwise copies
/// or DMA operations.
std::unique_ptr<OperationPass<func::FuncOp>> createAKGAffineDataCopyGenerationPass(
  unsigned slowMemorySpace, unsigned fastMemorySpace, unsigned tagMemorySpace = 0, int minDmaTransferSize = 1024,
  uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max(), bool generateDmaArg = true,
  bool skipNonUnitStrideLoopsArg = false);
/// Overload relying on pass options for initialization.
std::unique_ptr<OperationPass<func::FuncOp>> createAKGAffineDataCopyGenerationPass();
}  // namespace affine
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_