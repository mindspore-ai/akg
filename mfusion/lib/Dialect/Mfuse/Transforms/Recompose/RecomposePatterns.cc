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

#include "mfusion/Dialect/Mfuse/Transforms/Recompose/RecomposePatterns.h"

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {
/// Populate the given pattern set with recompose patterns.
/// This function registers all available recompose patterns with a RewritePatternSet.
void registerRecomposePatterns(RewritePatternSet &patterns) {
  registerMfuseMetaOpsToAclnnPatterns(patterns);
}

}  // namespace mlir::mfuse
