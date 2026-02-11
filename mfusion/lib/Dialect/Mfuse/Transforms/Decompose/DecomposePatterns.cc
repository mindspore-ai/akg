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
#include "mfusion/Dialect/Mfuse/Mfuse.h"
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

/// Populate the given pattern set with decompose patterns.
/// This function registers decompose patterns based on the specified pattern type and op list.
void registerDecomposePatterns(RewritePatternSet &patterns, DecomposePatternType patternType,
                               const std::vector<std::string> &opList) {
  switch (patternType) {
    case DecomposePatternType::ALL:
      registerAclnnDecomposePatterns(patterns, opList);
      registerDecomposeMathOpPatterns(patterns, opList);
      break;
    case DecomposePatternType::BEFORE_MANUAL_FUSION:
      registerAclnnDecomposePatterns(patterns, opList);
      break;
    case DecomposePatternType::AFTER_MANUAL_FUSION:
      registerDecomposeMathOpPatterns(patterns, opList);
      break;
    case DecomposePatternType::NONE:
      // No patterns to register
      break;
  }
}

}  // namespace mlir::mfuse
