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

#ifndef MFUSION_DIALECT_MFUSE_ANALYSIS_CONV_BIAS_ADD_INFER_H
#define MFUSION_DIALECT_MFUSE_ANALYSIS_CONV_BIAS_ADD_INFER_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

namespace mlir {
namespace mfuse {

/// Shape inference for aclnn.conv2d output + 1D NCHW channel bias **Add** only.
/// Not for Sub/Mul/etc. Call sites must gate on Add.
class ConvBiasAddInfer {
 public:
  static bool isAclnnConv2DResult(mlir::Value value) {
    mlir::Value src = value;
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast.getOperands().size() == 1) {
        src = cast.getOperand(0);
      }
    }
    return src.getDefiningOp<AclnnConv2DOp>() != nullptr;
  }

  /// True when any non-channel NCHW dim equals \p channels, so trailing-broadcast
  /// with bias [C] is also legal and channel-bias semantics are ambiguous.
  static bool hasAmbiguousChannelBiasShape(mlir::RankedTensorType nchwType, int64_t channels) {
    if (!nchwType || nchwType.getRank() != 4) {
      return true;
    }
    llvm::ArrayRef<int64_t> ndShape = nchwType.getShape();
    for (size_t i = 0; i < ndShape.size(); ++i) {
      if (i != 1 && ndShape[i] == channels) {
        return true;
      }
    }
    return false;
  }

  /// Ordered: \p nd must be conv2d result, \p bias must be 1D [C]. Does not swap operands.
  /// Requires matching element types (no promotion).
  static mlir::Type inferAddResultTypeOrdered(mlir::Value nd, mlir::Value bias) {
    if (!isAclnnConv2DResult(nd)) {
      return {};
    }
    auto ndType = mlir::dyn_cast<mlir::RankedTensorType>(nd.getType());
    auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
    if (!ndType || !biasType || biasType.getRank() != 1) {
      return {};
    }

    // Keep aclnn.add when mixed-dtype; avoid creating a generic Add that fuse will refuse.
    if (ndType.getElementType() != biasType.getElementType()) {
      return {};
    }

    llvm::ArrayRef<int64_t> ndShape = ndType.getShape();
    int64_t channels = biasType.getDimSize(0);
    if (ndShape.size() != 4 || ndShape[1] != channels) {
      return {};
    }
    if (hasAmbiguousChannelBiasShape(ndType, channels)) {
      return {};
    }

    return mlir::RankedTensorType::get(ndShape, ndType.getElementType());
  }

  /// Bidirectional helper for unit-alpha Add (operands may be swapped).
  /// Add-only: must not be used to infer Sub/other binary results.
  static mlir::Type inferAddResultType(mlir::Value lhs, mlir::Value rhs) {
    if (mlir::Type resultType = inferAddResultTypeOrdered(lhs, rhs)) {
      return resultType;
    }
    return inferAddResultTypeOrdered(rhs, lhs);
  }
};

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_ANALYSIS_CONV_BIAS_ADD_INFER_H
