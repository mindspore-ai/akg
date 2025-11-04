/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_VECTORTRANSFERTENSORIZE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_VECTORTRANSFERTENSORIZE_H_

#include <memory>
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace func {
class FuncOp;
}  // namespace func

namespace affine {

/// VectorTransferTensorize
/// ------------------------------------------------------------
/// This Pass performs three main transformations:
///  1. Upgrades scalar arith.constant used to populate vector.transfer_read
///     into tensor constants with matching shapes.
///  2. Replaces vector.transfer_read with bufferization.to_tensor.
///  3. Replaces vector.transfer_write with bufferization.to_memref,
///     and inserts a memref.copy to maintain explicit copy semantics for writes.
///
/// After the transformations, loops (or even the entire function) will no longer depend on vectors,
/// and can seamlessly connect to the subsequent pipeline in tensor/memref domain.
std::unique_ptr<OperationPass<func::FuncOp>>
createVectorTransferTensorizePass();

}  // namespace affine
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_VECTORTRANSFERTENSORIZE_H_
