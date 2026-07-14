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

#ifndef MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMMATERIALIZER_H
#define MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMMATERIALIZER_H

#include "mfusion/Analysis/DvmFusion/LayerNorm/LayerNormDvmPartitioner.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace mfuse {
class FusedOp;

namespace layernorm_dvm {

/// Materialize matcher-driven LayerNorm DVM islands. Tags only the mfuse.fused wrapper.
LogicalResult materializeLayerNormDvmPlan(PatternRewriter &rewriter, const LayerNormDvmPlan &plan);

void tagLayerNormDvmFusedOp(FusedOp fusedOp, llvm::StringRef groupId);

bool isLayerNormDvmMaterialized(Operation *op);

LogicalResult fuseLayerNormDvmForward(LayerNormDvmMatch &match, PatternRewriter &rewriter);
LogicalResult fuseLayerNormDvmBackwardGradDiv(const LayerNormDvmBwdMatch &match, PatternRewriter &rewriter);
LogicalResult fuseLayerNormDvmBackwardVector(const LayerNormDvmBwdMatch &match, PatternRewriter &rewriter);

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_LAYERNORMDVM_LAYERNORMDVMMATERIALIZER_H
