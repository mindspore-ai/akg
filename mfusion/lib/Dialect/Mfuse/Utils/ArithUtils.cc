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

#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"

#include <algorithm>
#include <cmath>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace mfuse {

namespace {
bool extractConstF64(Value v, double &outVal) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  auto attr = constOp.getValue();
  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr) {
    return false;
  }

  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }

  auto elementType = denseAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    auto floatVal = denseAttr.getSplatValue<APFloat>();
    outVal = floatVal.convertToDouble();
    return true;
  }
  return false;
}
}  // namespace

bool isSingleElementInt(Value v, int64_t x) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  // Check if it's a scalar (rank-0 tensor) or all dimensions are 1
  auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!denseAttr) {
    return false;
  }
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }

  // Check if it's a splat tensor (all elements are the same)
  if (!denseAttr.isSplat()) {
    return false;
  }
  if (!isa<IntegerType>(denseAttr.getElementType())) {
    return false;
  }
  return denseAttr.getSplatValue<APInt>().getSExtValue() == x;
}

// Check if tensor shape is empty (rank-0) or all dimensions are 1
bool isScalarOrSingleElement(RankedTensorType tensorType) {
  if (!tensorType) {
    return false;
  }

  // Check if rank is 0 (shape is empty)
  if (tensorType.getRank() == 0) {
    return true;
  }

  // Check if all dimensions are 1
  return !std::any_of(tensorType.getShape().begin(), tensorType.getShape().end(),
                      [](int64_t dim) { return dim != 1; });
}

bool hasDynamicShape(Type type) {
  auto ranked = dyn_cast<RankedTensorType>(type);
  if (!ranked) {
    return true;
  }
  return !ranked.hasStaticShape();
}

bool isSingleElementFloat(Value v, double x, double tolerance) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  auto attr = constOp.getValue();
  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr) {
    return false;
  }

  // Check if it's a scalar (rank-0 tensor) or all dimensions are 1
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }

  // Check if it's a splat tensor (all elements are the same)
  if (!denseAttr.isSplat()) {
    return false;
  }

  auto elementType = denseAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    auto floatVal = denseAttr.getSplatValue<APFloat>();
    return std::abs(floatVal.convertToDouble() - x) <= tolerance;
  }
  return false;
}

bool isConstOne(Value v, double tolerance) {
  auto constOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  auto attr = constOp.getValue();
  auto denseAttr = dyn_cast<DenseElementsAttr>(attr);
  if (!denseAttr) {
    return false;
  }

  // Check if it's a scalar (rank-0 tensor) or all dimensions are 1
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }

  // Check if it's a splat tensor (all elements are the same)
  if (!denseAttr.isSplat()) {
    return false;
  }

  auto elementType = denseAttr.getElementType();
  if (isa<FloatType>(elementType)) {
    auto floatVal = denseAttr.getSplatValue<APFloat>();
    return std::abs(floatVal.convertToDouble() - 1.0) <= tolerance;
  }
  if (isa<IntegerType>(elementType)) {
    auto intVal = denseAttr.getSplatValue<APInt>();
    return intVal.isOne();
  }
  return false;
}

bool isScalarMul(MulOp mulOp, double &scalarVal, Value &tensorOperand) {
  Value lhs = mulOp.getLhs();
  Value rhs = mulOp.getRhs();

  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());

  // Check if lhs is scalar or single-element (shape is empty or all dimensions are 1)
  if (lhsType && isScalarOrSingleElement(lhsType)) {
    if (extractConstF64(lhs, scalarVal)) {
      tensorOperand = rhs;
      return true;
    }
  }

  // Check if rhs is scalar or single-element, considering commutativity
  if (rhsType && isScalarOrSingleElement(rhsType)) {
    if (extractConstF64(rhs, scalarVal)) {
      tensorOperand = lhs;
      return true;
    }
  }

  return false;
}

}  // namespace mfuse
}  // namespace mlir
