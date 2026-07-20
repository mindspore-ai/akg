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

#include "mfusion/Dialect/Mfuse/Transforms/Decompose/DecomposePatterns.h"

#include <algorithm>
#include <string>
#include <vector>
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {
/// Normalize operation name for pattern matching
/// This function converts operation name to lowercase, removes underscores,
/// and removes any namespace prefixes
std::string normalizeOpName(const std::string &opName) {
  std::string normalized = opName;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  normalized.erase(std::remove(normalized.begin(), normalized.end(), '_'), normalized.end());
  normalized.erase(0, normalized.find_last_of('.') + 1);  // Remove any namespace prefix
  normalized.erase(0, normalized.find_last_of(':') + 1);  // Remove any other prefix
  return normalized;
}

/// Register patterns based on the provided op list
/// This function registers patterns from the given map based on the op list
void registerPatternsByOpList(RewritePatternSet &patterns, MLIRContext *ctx,
                              const std::map<std::string, PatternFunc> &patternMap,
                              const std::vector<std::string> &opList) {
  if (opList.empty()) {
    // Register all patterns
    for (const auto &[name, func] : patternMap) {
      func(patterns, ctx);
    }
  } else {
    // Register only patterns for operations in the list
    for (const auto &opName : opList) {
      std::string normalized = normalizeOpName(opName);
      if (patternMap.find(normalized) != patternMap.end()) {
        patternMap.at(normalized)(patterns, ctx);
      }
    }
  }
}

/// Register \p extraOpList patterns selectively on top of an already-populated set.
static void registerExtraDecomposePatterns(RewritePatternSet &patterns, DecomposePatternType patternType,
                                           const std::vector<std::string> &extraOpList) {
  if (extraOpList.empty()) {
    return;
  }
  switch (patternType) {
    case DecomposePatternType::ALL:
      registerAclnnDecomposePatterns(patterns, extraOpList);
      registerDecomposeMathOpPatterns(patterns, extraOpList, /*includeMatMulWithBiasByDefault=*/true);
      break;
    case DecomposePatternType::BEFORE_MANUAL_FUSION:
      registerAclnnDecomposePatterns(patterns, extraOpList);
      break;
    case DecomposePatternType::AFTER_MANUAL_FUSION:
      registerAclnnPostFusionDecomposePatterns(patterns, extraOpList);
      registerDecomposeMathOpPatterns(patterns, extraOpList, /*includeMatMulWithBiasByDefault=*/true);
      break;
    case DecomposePatternType::NONE:
      break;
  }
}

/// Populate the given pattern set with decompose patterns.
/// \p opList empty → pattern-type defaults; non-empty → only those ops.
/// \p extraOpList is always applied selectively on top (e.g. DVM AFTER defaults + matmul_with_bias).
void registerDecomposePatterns(RewritePatternSet &patterns, DecomposePatternType patternType,
                               const std::vector<std::string> &opList,
                               const std::vector<std::string> &extraOpList) {
  switch (patternType) {
    case DecomposePatternType::ALL:
      registerAclnnDecomposePatterns(patterns, opList);
      registerDecomposeMathOpPatterns(patterns, opList, /*includeMatMulWithBiasByDefault=*/true);
      break;
    case DecomposePatternType::BEFORE_MANUAL_FUSION:
      registerAclnnDecomposePatterns(patterns, opList);
      break;
    case DecomposePatternType::AFTER_MANUAL_FUSION:
      // Expand aclnn.var / var_mean while reduce_mean is still a meta op, then decompose reduce_mean.
      // Empty op-list omits matmul_with_bias (non-DVM → aten.addmm); DVM adds it via
      // extra-op-list=matmul_with_bias in the same AFTER stage.
      registerAclnnPostFusionDecomposePatterns(patterns, opList);
      registerDecomposeMathOpPatterns(patterns, opList, /*includeMatMulWithBiasByDefault=*/false);
      break;
    case DecomposePatternType::NONE:
      // No patterns to register
      break;
  }
  registerExtraDecomposePatterns(patterns, patternType, extraOpList);
}

}  // namespace mlir::mfuse
