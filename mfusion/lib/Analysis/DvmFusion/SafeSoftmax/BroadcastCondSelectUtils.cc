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

#include "mfusion/Analysis/DvmFusion/SafeSoftmax/BroadcastCondSelectUtils.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace mfuse {
namespace broadcast_cond_select {
namespace {

bool isBoolElementType(Type type) {
  return type.isInteger(1) || type.isSignlessInteger(1);
}

bool isFloatElementType(Type type) { return type.isF16() || type.isF32() || type.isBF16(); }

bool condBroadcastsToOutput(RankedTensorType condType, RankedTensorType outType) {
  if (!condType.hasStaticShape() || !outType.hasStaticShape()) {
    return false;
  }
  int64_t condRank = condType.getRank();
  int64_t outRank = outType.getRank();
  if (condRank > outRank || condRank == 0) {
    return false;
  }
  for (int64_t i = 1; i <= condRank; ++i) {
    int64_t condDim = condType.getDimSize(condRank - i);
    int64_t outDim = outType.getDimSize(outRank - i);
    if (condDim != outDim && condDim != 1) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool isBroadcastConditionalSelect(SelectOp op) {
  auto condType = dyn_cast<RankedTensorType>(op.getCondition().getType());
  auto outType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!condType || !outType) {
    return false;
  }
  if (!isBoolElementType(condType.getElementType()) || !isFloatElementType(outType.getElementType())) {
    return false;
  }
  for (Value branch : {op.getOnTrue(), op.getOnFalse()}) {
    auto branchType = dyn_cast<RankedTensorType>(branch.getType());
    if (!branchType || !isFloatElementType(branchType.getElementType())) {
      return false;
    }
  }
  if (!condBroadcastsToOutput(condType, outType)) {
    return false;
  }
  int64_t condElems = condType.getNumElements();
  int64_t outElems = outType.getNumElements();
  if (condElems <= 0 || outElems <= 0 || condElems >= outElems) {
    return false;
  }
  // Require a material broadcast expansion; skip near-equal shapes.
  constexpr int64_t kMinBroadcastExpansion = 4;
  return outElems / condElems >= kMinBroadcastExpansion;
}

}  // namespace broadcast_cond_select
}  // namespace mfuse
}  // namespace mlir
