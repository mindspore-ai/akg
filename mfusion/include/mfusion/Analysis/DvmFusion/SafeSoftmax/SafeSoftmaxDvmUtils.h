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

#ifndef MFUSION_ANALYSIS_SAFESOFTMAXDVM_SAFESOFTMAXDVMUTILS_H
#define MFUSION_ANALYSIS_SAFESOFTMAXDVM_SAFESOFTMAXDVMUTILS_H

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace mfuse {
namespace safe_softmax_dvm {

struct SafeSoftmaxDvmMatch {
  bool matched = false;
  SelectOp select;
  DivOp softmaxDiv;
  ExpOp exp;
  ReduceSumOp reduceSum;
  SubOp centerSub;
  ReduceMaxOp reduceMax;
  FullOp full;
  /// Zero branch when reused from a prior safe-softmax mfuse.fused result #0.
  Value zeroBranch;
  llvm::SmallVector<Operation *, 16> memberOps;
};

/// Fused mfuse.softmax / torch.aten._softmax feeding a safe-softmax select.
struct FusedSafeSoftmaxCandidate {
  bool matched = false;
  SelectOp select;
  FullOp full;
  Value zeroBranch;
  Operation *softmaxOp = nullptr;
  Value softmaxOutput;
};

/// Decomposed softmax (sub/exp/reduce_sum/div) + broadcast masked select.
SafeSoftmaxDvmMatch matchSafeSoftmaxFromSelect(SelectOp selectOp);

/// Broadcast select with full(0) branch and mfuse.softmax or torch.aten._softmax.
FusedSafeSoftmaxCandidate matchFusedSoftmaxCandidateFromSelect(SelectOp selectOp);

/// Like matchFusedSoftmaxCandidateFromSelect but returns {} when select is already decomposed.
FusedSafeSoftmaxCandidate matchFusedSafeSoftmaxFromSelect(SelectOp selectOp);

/// Pre- or post-decompose safe-softmax broadcast masked select.
bool isSafeSoftmaxBroadcastSelect(SelectOp selectOp);

/// True when the op is mfuse.softmax.
bool isMfuseSoftmaxOp(Operation *op);

/// True when the op is torch.aten._softmax.
bool isAtenSoftmaxOp(Operation *op);

/// True when the op is mfuse.softmax or torch.aten._softmax.
bool isSoftmaxProducerOp(Operation *op);

/// True when a fused softmax feeds a safe-softmax broadcast select.
bool isSafeSoftmaxSoftmaxProducer(Operation *softmaxOp);

/// Scan module/func for safe-softmax candidate regions.
bool hasSafeSoftmaxCandidate(Operation *root);

void markSafeSoftmaxPipelineActive(ModuleOp module);

bool isSafeSoftmaxTagged(Operation *op);

/// True when value is result #0 of a tagged safe-softmax mfuse.fused (shared zero buffer).
bool isSafeSoftmaxZeroOutput(Value value);

/// Collect member ops for direct mfuse.fused materialization (includes reduce_max / reshape).
void collectSafeSoftmaxMemberOps(SafeSoftmaxDvmMatch &match);

/// Ops produced by Albert mask canonicalization.
/// Pattern A (eq-based): eq -> not -> reduce_any -> not -> select
/// Pattern B (ne-based): ne -> reduce_any -> not -> select  (ne absorbs eq+not)
struct AlbertMaskChainOps {
  Operation *eq = nullptr;
  Operation *eqNot = nullptr;
  Operation *ne = nullptr;
  Operation *reduceAny = nullptr;
  Operation *condNot = nullptr;
};

void appendAlbertMaskChainMemberOps(SafeSoftmaxDvmMatch &match, const AlbertMaskChainOps &chain);
void appendMemberOp(SafeSoftmaxDvmMatch &match, Operation *op);

}  // namespace safe_softmax_dvm
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SAFESOFTMAXDVM_SAFESOFTMAXDVMUTILS_H
