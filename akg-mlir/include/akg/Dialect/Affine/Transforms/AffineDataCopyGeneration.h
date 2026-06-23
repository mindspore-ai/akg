/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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
#ifndef AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_
#define AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_
#include <limits>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace affine {

struct AKGDataCopyGenerationParams {
  unsigned slowMemorySpace = 0;
  unsigned fastMemorySpace = 0;
  unsigned tagMemorySpace = 0;
  int minDmaTransferSize = 1024;
  uint64_t fastMemCapacityBytes = std::numeric_limits<uint64_t>::max();
  bool generateDma = true;
  bool skipNonUnitStrideLoops = false;
};

std::unique_ptr<OperationPass<func::FuncOp>> createAKGAffineDataCopyGenerationPass(
  const AKGDataCopyGenerationParams &params);
std::unique_ptr<OperationPass<func::FuncOp>> createAKGAffineDataCopyGenerationPass();
}  // namespace affine
}  // namespace mlir
#endif  // AKG_DIALECT_AFFINE_TRANSFORMS_AFFINEDATACOPYGENERATION_H_
