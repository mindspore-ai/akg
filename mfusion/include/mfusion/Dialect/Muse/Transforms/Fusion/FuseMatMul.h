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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_FUSE_MAT_MUL_H
#define MFUSION_DIALECT_MUSE_TRANSFORMS_FUSE_MAT_MUL_H

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Fuse MatMul/Mm (float16) followed by Cast (float32) into MatMul with float32 result.
std::unique_ptr<Pass> createFuseMatMulCastPass();

/// Fuse MatMul/Mm with 1D inputs by inserting reshape (unsqueeze/squeeze).
std::unique_ptr<Pass> createFuseMatmulUnsqueezeSqueezePass();

/// Insert permute for MatMul/Mm inputs when inner axis is not 512-byte aligned; set trans_x1/trans_x2.
std::unique_ptr<Pass> createFuseMatmulTransposeWeightPass();

/// Fuse MatMul -> Reshape -> Add(bias) into MatMul(with bias) -> Reshape.
std::unique_ptr<Pass> createFuseMatmulReshapeBiasAddPass();

}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_TRANSFORMS_FUSE_MAT_MUL_H
