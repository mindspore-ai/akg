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

#ifndef MFUSION_DIALECT_MUSE_UTILS_ARITH_UTILS_H
#define MFUSION_DIALECT_MUSE_UTILS_ARITH_UTILS_H

#include <cmath>

#include "mfusion/Dialect/Muse/Muse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace muse {

// Check if tensor shape is empty (rank-0) or all dimensions are 1
inline bool isScalarOrSingleElement(RankedTensorType tensorType) {
  if (!tensorType) {
    return false;
  }
  
  // Check if rank is 0 (shape is empty)
  if (tensorType.getRank() == 0) {
    return true;
  }
  
  // Check if all dimensions are 1
  for (int64_t dim : tensorType.getShape()) {
    if (dim != 1) {
      return false;
    }
  }
  return true;
}

// Extract constant value from a scalar or single-element tensor
// Returns true if the value is a constant scalar/single-element with the expected value
// Only supports float types (F64/F32)
inline bool extractConstF64(Value v, double &outVal) {
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

// Check if Value is a constant scalar or single-element integer tensor with value x
inline bool IsSingleElementInt(Value v, int64_t x) {
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
  if (isa<IntegerType>(elementType)) {
    auto intVal = denseAttr.getSplatValue<APInt>();
    return intVal.getSExtValue() == x;
  }
  return false;
}

// Check if Value is a constant scalar or single-element float tensor with value x
// Uses tolerance check for floating point comparison
inline bool isSingleElementFloat(Value v, double x, double tolerance = 1e-6) {
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

// Check if Value is a constant scalar or single-element tensor with value 1.0
// Supports both float and integer types
// For float types, uses tolerance check (1e-6) for floating point comparison
inline bool isConstOne(Value v, double tolerance = 1e-6) {
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

// Check if MulOp is a scalar multiplication (one operand is scalar or single-element tensor)
// Returns the scalar value and the tensor operand
// This function checks both operands to identify which one is scalar
inline bool isScalarMul(MulOp mulOp, double &scalarVal, Value &tensorOperand) {
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

}  // namespace muse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_UTILS_ARITH_UTILS_H
