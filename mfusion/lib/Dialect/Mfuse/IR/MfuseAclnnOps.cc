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

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace mfuse {

mlir::LogicalResult AclnnVarMeanOp::verify() {
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(getSelf().getType());
  if (!inputType || !inputType.hasRank()) {
    return emitOpError("input must be a ranked tensor");
  }

  int64_t rank = inputType.getRank();
  auto dimAttr = getDim();

  llvm::DenseSet<int64_t> seenDims;
  for (auto dimAttrVal : dimAttr.getValue()) {
    auto dim = mlir::cast<mlir::IntegerAttr>(dimAttrVal).getValue().getSExtValue();
    int64_t actualDim = dim < 0 ? rank + dim : dim;
    if (actualDim < 0) {
      return emitOpError("dimension out of range, got ") << dim;
    }
    if (actualDim >= rank) {
      return emitOpError("dimension out of range, got ") << dim << " for input rank " << rank;
    }
    if (!seenDims.insert(actualDim).second) {
      return emitOpError("duplicate reduction dimensions are not supported, got ") << dim;
    }
  }

  auto varianceOutType = mlir::dyn_cast<mlir::RankedTensorType>(getVarianceOut().getType());
  auto meanOutType = mlir::dyn_cast<mlir::RankedTensorType>(getMeanOut().getType());
  if (!varianceOutType || !meanOutType) {
    return emitOpError("results must be ranked tensors");
  }

  if (!mlir::isa<mlir::FloatType>(varianceOutType.getElementType()) ||
      !mlir::isa<mlir::FloatType>(meanOutType.getElementType())) {
    return emitOpError("result element types must be floating point");
  }

  auto correction = getCorrection();
  if (correction < 0) {
    return emitOpError("correction must be non-negative, got ") << correction;
  }

  return mlir::success();
}

}  // namespace mfuse
}  // namespace mlir
