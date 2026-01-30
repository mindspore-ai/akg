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

#include "mfusion/Dialect/Muse/Transforms/Decompose/DecomposePatterns.h"

#include "mfusion/Dialect/Muse/Muse.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::muse {
/// Populate the given pattern set with decompose patterns.
/// This function registers all available decompose patterns with a RewritePatternSet.
void registerDecomposePatterns(RewritePatternSet &patterns) {
  registerDecomposeMathOpPatterns(patterns);
}

}  // namespace mlir::muse
