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

//===----------------------------------------------------------------------===//
//
// This file declares the transform passes for the TOSA Dialect in MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_PASSES_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_PASSES_H_

#include <memory>

#include "akg/Dialect/MindSpore/Transforms/AKGSplitGraph.h"
#include "akg/Dialect/MindSpore/Transforms/EliminateReshape.h"
#include "akg/Dialect/MindSpore/Transforms/MoveDownReductionOps.h"
#include "akg/Dialect/MindSpore/Transforms/RemoveRedundantReduce.h"

namespace mlir {

std::unique_ptr<mlir::Pass> createMakeDynamicBroadcastablePass();
std::unique_ptr<OperationPass<func::FuncOp>> createRemoveRedundantReducePass();
std::unique_ptr<OperationPass<func::FuncOp>> createMoveDownReductionOpsPass();
std::unique_ptr<OperationPass<func::FuncOp>> createEliminateReshapePass();

std::unique_ptr<mlir::Pass> createMakeDynamicBroadcastablePass(bool ignoreImplicitBroadcast);
#define GEN_PASS_REGISTRATION
#include "akg/Dialect/MindSpore/Passes.h.inc"
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_PASSES_H_
