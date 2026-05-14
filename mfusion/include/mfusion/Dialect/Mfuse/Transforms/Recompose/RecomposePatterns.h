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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_RECOMPOSE_RECOMPOSE_PATTERNS_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_RECOMPOSE_RECOMPOSE_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {

/// Populate the given pattern set with recompose patterns.
/// This function registers OpRewritePattern-based recompose patterns
/// that lower Mfuse operations to aclnn operations.
void registerRecomposePatterns(RewritePatternSet &patterns);

/// Lower Mfuse meta ops (e.g. mfuse.matmul / mfuse.matmul_with_bias) to aclnn.matmul / aclnn.mm
/// (matmul_with_bias also lower to aclnn.add for bias).
void registerMfuseMetaOpsToAclnnPatterns(RewritePatternSet &patterns);

}  // namespace mlir::mfuse

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_RECOMPOSE_RECOMPOSE_PATTERNS_H
