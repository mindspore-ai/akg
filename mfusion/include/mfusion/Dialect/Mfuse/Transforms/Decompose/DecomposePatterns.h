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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H

#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {

/// Decompose pattern types enum
enum class DecomposePatternType {
  NONE,                  // No patterns
  ALL,                   // All patterns
  BEFORE_MANUAL_FUSION,  // Patterns to apply before manual fusion
  AFTER_MANUAL_FUSION    // Patterns to apply after manual fusion
};

/// Pattern function type for registering decomposition patterns
using PatternFunc = void (*)(RewritePatternSet &, MLIRContext *);

/// Normalize operation name for pattern matching
/// This function converts operation name to lowercase, removes underscores,
/// and removes any namespace prefixes
/// \param opName The original operation name
/// \return The normalized operation name
std::string normalizeOpName(const std::string &opName);

/// Register patterns based on the provided op list
/// This function registers patterns from the given map based on the op list
/// \param patterns The pattern set to populate
/// \param ctx The MLIR context
/// \param patternMap Map of normalized operation names to their pattern registration functions
/// \param opList List of specific operations to register (empty means all)
void registerPatternsByOpList(RewritePatternSet &patterns, MLIRContext *ctx,
                              const std::map<std::string, PatternFunc> &patternMap,
                              const std::vector<std::string> &opList);

/// Populate the given pattern set with decompose patterns.
/// This function registers OpRewritePattern-based decompose patterns
/// that automatically match and rewrite operations based on the pattern type.
/// \param patterns The pattern set to populate
/// \param patternType The type of decompose patterns to register
/// \param opList Optional list of specific operations to decompose (empty means all)
void registerDecomposePatterns(RewritePatternSet &patterns,
                               DecomposePatternType patternType = DecomposePatternType::ALL,
                               const std::vector<std::string> &opList = {});

void registerDecomposeMathOpPatterns(RewritePatternSet &patterns, const std::vector<std::string> &opList = {});

void registerAclnnDecomposePatterns(RewritePatternSet &patterns, const std::vector<std::string> &opList = {});

}  // namespace mlir::mfuse

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_DECOMPOSE_DECOMPOSE_PATTERNS_H
