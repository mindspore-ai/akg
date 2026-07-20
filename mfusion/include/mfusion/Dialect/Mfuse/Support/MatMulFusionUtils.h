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

#ifndef MFUSION_DIALECT_MFUSE_SUPPORT_MATMUL_FUSION_UTILS_H
#define MFUSION_DIALECT_MFUSE_SUPPORT_MATMUL_FUSION_UTILS_H

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace mfuse {

/// Matmul input dtypes only support float16, float32, bfloat16.
inline bool isSupportedMatmulDtype(Type type) {
  auto elemType = dyn_cast<FloatType>(type);
  if (!elemType) {
    return false;
  }
  return isa<Float16Type>(elemType) || elemType.isF32() || isa<BFloat16Type>(elemType);
}

/// Returns true if the permute only swaps the last two dimensions:
/// input (..., a, b) -> output (..., b, a); batch dims unchanged.
inline bool isPermuteSwapLastTwoDims(PermuteOp permuteOp) {
  auto inputType = dyn_cast<RankedTensorType>(permuteOp.getInput().getType());
  if (!inputType) {
    return false;
  }
  int64_t rank = inputType.getRank();
  if (rank < 2) {
    return false;
  }
  auto permAttr = permuteOp.getPermAttr();
  if (!permAttr) {
    return false;
  }
  auto permValues = permAttr.getValue();
  if (permValues.size() != static_cast<size_t>(rank)) {
    return false;
  }
  int64_t lastIdx = rank - 1;
  int64_t secondLastIdx = rank - 2;
  for (int64_t i = 0; i < rank; ++i) {
    auto intAttr = dyn_cast<IntegerAttr>(permValues[i]);
    if (!intAttr) {
      return false;
    }
    int64_t p = intAttr.getInt();
    if (i < secondLastIdx) {
      if (p != i) {
        return false;
      }
    } else if (i == secondLastIdx) {
      if (p != lastIdx) {
        return false;
      }
    } else {
      if (p != secondLastIdx) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_SUPPORT_MATMUL_FUSION_UTILS_H
