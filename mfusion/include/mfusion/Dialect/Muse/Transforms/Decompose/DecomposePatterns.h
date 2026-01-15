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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H
#define MFUSION_DIALECT_MUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::muse {

/// Populate the given pattern set with decompose patterns.
/// This function registers OpRewritePattern-based decompose patterns
/// that automatically match and rewrite GELU and Tanh operations.
void registerDecomposePatterns(RewritePatternSet &patterns);

void registerDecomposeMathOpPatterns(RewritePatternSet &patterns);

} // namespace mlir::muse

#endif // MFUSION_DIALECT_MUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H