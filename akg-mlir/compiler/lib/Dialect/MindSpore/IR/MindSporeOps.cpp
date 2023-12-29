/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

#include "akg/Dialect/MindSpore/IR/MindSporeOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::mindspore;

void ReshapeOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Type output, Value input,
                      const Value newShape) {
  build(odsBuilder, odsState, output, input, newShape, {});
}

void ReshapeOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Type output, Value input,
                      const DenseI64ArrayAttr newShape) {
  build(odsBuilder, odsState, output, input, {}, newShape);
}

LogicalResult ReshapeOp::verify() {
  if ((getNewShapeValue() == nullptr && getNewShapeAttr() != nullptr) ||
      (getNewShapeValue() != nullptr && getNewShapeAttr() == nullptr)) {
    return success();
  }
  return emitOpError("unexpected mindspore.rehape op error");
}

void BroadcastToOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Type output, Value input,
                          const Value newShape) {
  build(odsBuilder, odsState, output, input, newShape, {});
}

void BroadcastToOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Type output, Value input,
                          const DenseI64ArrayAttr newShape) {
  build(odsBuilder, odsState, output, input, {}, newShape);
}

LogicalResult BroadcastToOp::verify() {
  uint64_t newRankSize = getNewRankSize();
  if (getNewShapeValue() == nullptr && getNewShapeAttr() != nullptr) {
    llvm::ArrayRef<int64_t> newShape = *getNewShape();
    uint64_t inputRank = getInput().getType().cast<ShapedType>().getRank();
    for (uint64_t i = 0; i < newRankSize - inputRank; i++) {
      if (newShape[i] == -1) {
        return emitOpError(
          "if the first * dims of output shape have -1 in it, it implies this -1 iscorresponding to a non-existing dim "
          "so they' re not broadcastable.");
      }
    }
    return success();
  }
  if (getNewShapeValue() != nullptr && getNewShapeAttr() == nullptr) {
    return success();
  }
  return emitOpError("unexpected mindspore.broadcast_to op error");
}

uint64_t BroadcastToOp::getNewRankSize() {
  if (getNewShapeValue() != nullptr) {
    return getNewShapeValue().getType().cast<mlir::ShapedType>().getShape()[0];
  }
  if (getNewShapeAttr() != nullptr) {
    return (*getNewShape()).size();
  }
  return 0;
}

void ConcatOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, mlir::Type output, Value input) {
  build(odsBuilder, odsState, output, input, {});
}

void mlir::mindspore::MindSporeDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "akg/Dialect/MindSpore/IR/MindSporeOps.cpp.inc"
    >();
}

#ifndef GET_OP_CLASSES
#define GET_OP_CLASSES
#include "akg/Dialect/MindSpore/IR/MindSporeOps.cpp.inc"
#endif
