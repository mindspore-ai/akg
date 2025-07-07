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
// This file declares the optimization passes for the Linalg Dialect in MLIR.
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_DIALECT_LINALG_PASSES_H_
#define COMPILER_INCLUDE_AKG_DIALECT_LINALG_PASSES_H_

#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Transforms/Bufferize.h"
#include "akg/Dialect/Linalg/Transforms/LinalgCopyBufferize.h"
#include "akg/Dialect/Linalg/Transforms/LinalgSimplify.h"
#include "akg/Dialect/Linalg/Transforms/MatchAndMarkReductionOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir {
const char TemplateFuncAttrName[] = "template_func";

namespace linalgExt {
struct TemplateOpFusionOptions {
  TemplateOpFusionOptions() = default;

  TemplateOpFusionOptions &enableOptReshapeByExpand() {
    optReshapeByExpand = true;
    return *this;
  }

  TemplateOpFusionOptions &enableOptReshapeByCollapse() {
    optReshapeByExpand = true;
    return *this;
  }

  bool optReshapeByExpand{false};
  bool optReshapeByCollapse{false};
};
}  // namespace linalgExt

/// Create a pass to convert named Linalg operations to Linalg templated
/// Implementation.
std::unique_ptr<OperationPass<ModuleOp>> createLinalgTemplatedPass(std::string templatePath = "");

/// Create a pass to lower templated Linalg operations to function call.
std::unique_ptr<OperationPass<func::FuncOp>> createLinalgLowerTemplateOpPass();
void populateLinalgTemplateOpLowerPatterns(RewritePatternSet &patterns);

std::unique_ptr<Pass> createLinalgElementwiseFusionExtPass();

/// Pattern to fuse `linalg.generic` -> `linalg.template` operations
/// when both operations are fusable elementwise operations.
void populateTemplateOpsFusionPatterns(RewritePatternSet &patterns);

#define GEN_PASS_REGISTRATION
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_LINALG_PASSES_H_
