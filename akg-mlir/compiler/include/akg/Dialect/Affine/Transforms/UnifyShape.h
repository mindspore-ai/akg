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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_UNIFYSHAPE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_UNIFYSHAPE_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
}  // namespace mlir

#ifndef GEN_PASS_DECL_UNIFYSHAPE
#define GEN_PASS_DECL_UNIFYSHAPE
#include "akg/Dialect/Affine/Passes.h.inc"
#endif

namespace mlir {
/// Create a pass that removes reshape.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createUnifyShapePass();

/// Create a pass that removes reshape.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createUnifyShapePass(bool allowNonPolyhedralAccess, bool keepArg);
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_UNIFYSHAPE_H_
