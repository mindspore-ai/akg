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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SIMPLIFYSHAPE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SIMPLIFYSHAPE_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

#define GEN_PASS_DECL_SIMPLIFYSHAPE
#include "akg/Dialect/Affine/Passes.h.inc"

namespace mlir {
/// Create a pass that removes dimension 1 shape.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSimplifyShapePass();

/// Create a pass that removes dimension 1 shape.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSimplifyShapePass(bool keepArg);
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SIMPLIFYSHAPE_H_

