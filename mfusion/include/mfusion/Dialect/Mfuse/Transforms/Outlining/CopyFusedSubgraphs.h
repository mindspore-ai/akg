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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_COPY_FUSED_SUBGRAPHS_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_COPY_FUSED_SUBGRAPHS_H

#include <memory>

namespace mlir {
class Pass;

/// Create a pass to clone fused subgraphs for downstream shape inference in pytorch.
///
/// This pass clones each outlined fusion function and stores the clone's name
/// in the original function's `mfusion.copied_subgraph` attribute. The cloned
/// function will be converted to torch dialect and then to fx graph, serving
/// as the shape inference function for the DVM kernel (torch.operator).
///
/// Data flow:
/// - Original function: outlined -> DVM lowering -> serialized as subgraph_mlir
/// - Cloned function: remains in IR -> torch dialect -> fx graph for shape inference
std::unique_ptr<Pass> createCopyFusedSubgraphsPass();

}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_OUTLINING_COPY_FUSED_SUBGRAPHS_H
