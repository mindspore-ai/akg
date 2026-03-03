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

#ifndef MFUSION_DIALECT_MFUSE_ANALYSIS_REDUCE_OP_COMMON_INFER_H
#define MFUSION_DIALECT_MFUSE_ANALYSIS_REDUCE_OP_COMMON_INFER_H

#include <cstdint>
#include <type_traits>
#include <vector>

#include "llvm/ADT/DenseSet.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

#include "mfusion/Analysis/SymbolicShape/SymExprBuilder.h"
#include "mfusion/Dialect/Mfuse/Utils/SymbolAttrUtils.h"

namespace mlir {
namespace mfuse {

/// Common type/shape inference for reduce ops (e.g. ReduceSum).
class ReduceOpCommonInfer {
 public:
  /// Infers output shape from input shape, reduce dimensions and keepdim.
  /// - inputShape: input tensor shape (concrete int64_t or symbolic SymExpr).
  /// - dimensions: which dimensions to reduce.
  /// - keepdim: if true, reduced dimensions become 1 in output; otherwise they are removed.
  template <typename T>
  static std::vector<T> inferShape(const std::vector<T> &inputShape, mlir::ArrayAttr dimensions, bool keepdim) {
    llvm::DenseSet<size_t> reduceDims;
    for (auto dimAttr : dimensions.getValue()) {
      auto dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
      reduceDims.insert(static_cast<size_t>(dim));
    }

    std::vector<T> outShape;
    for (size_t i = 0; i < inputShape.size(); ++i) {
      if (reduceDims.contains(i)) {
        if (keepdim) {
          if constexpr (std::is_same_v<T, int64_t>) {
            outShape.push_back(1);
          } else {
            mfusion::SymExprBuilder symBuilder;
            outShape.push_back(symBuilder.makeInteger(1));
          }
        }
      } else {
        outShape.push_back(inputShape[i]);
      }
    }
    return outShape;
  }

  /// Infers result type for reduce ops (concrete shape).
  static mlir::Type inferResultType(mlir::Value input, mlir::ArrayAttr dimensions, mlir::BoolAttr keepdim,
                                    mlir::Type elementType) {
    auto inType = llvm::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inType) {
      return {};
    }
    std::vector<int64_t> inShapeVec(inType.getShape().begin(), inType.getShape().end());
    std::vector<int64_t> outShapeVec = inferShape(inShapeVec, dimensions, keepdim.getValue());
    return mlir::RankedTensorType::get(outShapeVec, elementType);
  }

  /// Infers symbolic shape attribute for reduce ops and returns updated result type.
  static mlir::FailureOr<mlir::Type> inferSymbolicShapes(mlir::OpBuilder &builder, const mlir::OperationState &state,
                                                         mlir::Type resultType) {
    if (state.operands.empty()) {
      return mlir::failure();
    }

    auto inType = state.operands[0].getType().dyn_cast<mlir::RankedTensorType>();
    auto outType = resultType.dyn_cast<mlir::RankedTensorType>();
    if (!outType || !inType) {
      return mlir::failure();
    }

    auto maybeInExprs = SymbolAttrUtils::getSymbolicShapeExprs(inType);
    if (mlir::failed(maybeInExprs)) {
      return mlir::failure();
    }
    std::vector<SymbolAttrUtils::SymExpr> inExprs(maybeInExprs->begin(), maybeInExprs->end());

    auto dimensions = state.attributes.get("dimensions").dyn_cast_or_null<mlir::ArrayAttr>();
    auto keepdimAttr = state.attributes.get("keepdim").dyn_cast_or_null<mlir::BoolAttr>();
    if (!dimensions || !keepdimAttr) {
      return mlir::failure();
    }
    std::vector<SymbolAttrUtils::SymExpr> outExprs = inferShape(inExprs, dimensions, keepdimAttr.getValue());
    return SymbolAttrUtils::withSymbolicAttr(outType, builder, outExprs);
  }
};

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_ANALYSIS_REDUCE_OP_COMMON_INFER_H
