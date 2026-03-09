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

#include "mfusion/Conversion/MfuseToTorch/Utils.h"

namespace mlir {

namespace TorchD = mlir::torch::Torch;

Value materializeConstValueToTorchScalar(Operation *op, Value value, ConversionPatternRewriter &rewriter) {
  Value valueForConst = value;
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getOperands().size() == 1) {
      valueForConst = cast.getOperand(0);
    }
  }

  if (auto cst = valueForConst.getDefiningOp<arith::ConstantOp>()) {
    auto attr = dyn_cast<DenseElementsAttr>(cst.getValue());
    if (attr && attr.getType().hasRank() && attr.getType().getRank() == 0) {
      auto elementType = attr.getType().getElementType();
      if (isa<FloatType>(elementType)) {
        auto floatValue = attr.getSplatValue<APFloat>().convertToDouble();
        return rewriter.create<TorchD::ConstantFloatOp>(op->getLoc(),
                                                        rewriter.getFloatAttr(rewriter.getF64Type(), floatValue));
      }
      if (isa<IntegerType>(elementType)) {
        auto intValue = attr.getSplatValue<APInt>().getSExtValue();
        return rewriter.create<TorchD::ConstantIntOp>(op->getLoc(), rewriter.getI64IntegerAttr(intValue));
      }
    }
  }
  return valueForConst;
}

}  // namespace mlir