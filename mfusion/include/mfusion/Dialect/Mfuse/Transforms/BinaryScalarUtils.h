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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_BINARY_SCALAR_UTILS_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_BINARY_SCALAR_UTILS_H

#include <optional>

#include "mfusion/Dialect/Mfuse/Analysis/BinaryOpCommonInfer.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::mfuse {

struct ScalarOperandInfo {
  Value scalar;
  Operation *numToTensor = nullptr;
};

inline bool isScalarConstant(Value value) {
  return value.getDefiningOp<ConstantOp>() && isScalarType(value.getType());
}

inline std::optional<ScalarOperandInfo> getRecoverableScalar(Value value) {
  if (isScalarConstant(value)) {
    return ScalarOperandInfo{value};
  }

  auto numToTensorOp = value.getDefiningOp<NumToTensorOp>();
  if (!numToTensorOp || !isScalarConstant(numToTensorOp.getValue())) {
    return std::nullopt;
  }
  return ScalarOperandInfo{numToTensorOp.getValue(), numToTensorOp.getOperation()};
}

inline void eraseDeadNumToTensor(PatternRewriter &rewriter,
                                 const std::optional<ScalarOperandInfo> &info) {
  if (info && info->numToTensor && info->numToTensor->use_empty()) {
    rewriter.eraseOp(info->numToTensor);
  }
}

}  // namespace mlir::mfuse

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_BINARY_SCALAR_UTILS_H
