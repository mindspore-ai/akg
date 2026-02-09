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

#ifndef MFUSION_DIALECT_MUSE_TRANSFORMS_FUSION_FUSION_PASS_MACROS_H
#define MFUSION_DIALECT_MUSE_TRANSFORMS_FUSION_FUSION_PASS_MACROS_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace muse {

//===----------------------------------------------------------------------===//
// Fusion Pass Definition Macros
//===----------------------------------------------------------------------===//

/**
 * Macro to define a fusion pass with pattern-based rewriting and auto-registration.
 * This macro generates:
 * 1. Pass struct inheriting from the Base class
 * 2. runOnOperation() method that adds patterns and applies greedy rewrite
 * 3. create function for the pass
 */
#define DEFINE_MUSE_FUSION_PASS(PassName, ...)                                         \
  struct PassName##Pass : public impl::PassName##Base<PassName##Pass> {                \
    void runOnOperation() override {                                                   \
      RewritePatternSet patterns(&getContext());                                       \
      patterns.add<__VA_ARGS__>(&getContext());                                        \
      if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) { \
        signalPassFailure();                                                           \
      }                                                                                \
    }                                                                                  \
  };                                                                                   \
                                                                                       \
  std::unique_ptr<Pass> create##PassName##Pass() { return std::make_unique<PassName##Pass>(); }

}  // namespace muse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_TRANSFORMS_FUSION_FUSION_PASS_MACROS_H
