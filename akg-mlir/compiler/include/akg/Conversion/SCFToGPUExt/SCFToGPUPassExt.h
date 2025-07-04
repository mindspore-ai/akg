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
//===- SCFToGPUPass.h - Pass converting loops to GPU kernels ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUPASSEXT_H_
#define COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUPASSEXT_H_
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"

#include <memory>

namespace mlir {
template <typename T>
class InterfacePass;
class Pass;

#define GEN_PASS_DECL_CONVERTAKGAFFINEFORTOGPU
#define GEN_PASS_DECL_CONVERTAKGPARALLELLOOPTOGPU
#include "akg/Conversion/Passes.h.inc"

/// Create a pass that converts loop nests into GPU kernels.  It considers
/// top-level affine.for operations as roots of loop nests and converts them to
/// the gpu.launch operations if possible.
///
/// No check on the size of the block or grid, or on the validity of
/// parallelization is performed, it is under the responsibility of the caller
/// to strip-mine the loops and to perform the dependence analysis before
/// calling the conversion.
std::unique_ptr<InterfacePass<FunctionOpInterface>> createAKGAffineForToGPUPass(unsigned numBlockDims,
                                                                                unsigned numThreadDims);
std::unique_ptr<InterfacePass<FunctionOpInterface>> createAKGAffineForToGPUPass();

/// Creates a pass that converts scf.parallel operations into a gpu.launch
/// operation. The mapping of loop dimensions to launch dimensions is derived
/// from mapping attributes. See ParallelToGpuLaunchLowering::matchAndRewrite
/// for a description of the used attributes.

}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_CONVERSION_SCFTOGPUEXT_SCFTOGPUPASSEXT_H_
