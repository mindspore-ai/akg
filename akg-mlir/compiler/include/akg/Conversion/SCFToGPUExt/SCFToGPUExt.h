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
//===- SCFToGPU.h - Convert loop nests to GPU kernels -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUEXT_H_
#define COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUEXT_H_

#include "mlir/Support/LLVM.h"

namespace mlir {
class ConversionTarget;
struct LogicalResult;
class MLIRContext;
class Value;
class Operation;
class RewritePatternSet;

namespace affine {
class AffineForOp;
}  // namespace affine

namespace scf {
class ForOp;
}  // namespace scf

/// Convert a perfect affine loop nest with the outermost loop identified by
/// `forOp` into a gpu::Launch operation.  Map `numBlockDims` outer loops to
/// GPU blocks and `numThreadDims` to GPU threads.  The bounds of the loops that
/// are mapped should be independent of the induction variables of the other
/// mapped loops.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.

// TODO: Consider removing this in favor of affine.for -> affine.parallel
// detection followed by an affine.parallel -> scf.parallel -> gpu.launch
// conversion
LogicalResult convertAffineLoopNestToGPULaunch(affine::AffineForOp forOp, unsigned numBlockDims,
                                               unsigned numThreadDims);

/// Adds the conversion pattern from `scf.parallel` to `gpu.launch` to the
/// provided pattern list.
void populateParallelLoopToGPUPatterns(RewritePatternSet &patterns);

/// Configures the rewrite target such that only `scf.parallel` operations that
/// are not rewritten by the provided patterns are legal.
void configureParallelLoopToGPULegality(ConversionTarget &target);

/// Clean up after applyPartialConversion/applyFullConversion call.
void finalizeParallelLoopToGPUConversion(Operation *op);

}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUEXT_H_
