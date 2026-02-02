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

#include "mfusion/Dialect/Muse/Muse.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

namespace mlir {
namespace muse {

// Check if Value is a constant scalar or single-element integer tensor with value x
bool IsSingleElementInt(Value v, int64_t x);

// Check if Value is a constant scalar or single-element float tensor with value x
// Uses tolerance check for floating point comparison
bool isSingleElementFloat(Value v, double x, double tolerance = 1e-6);

// Check if Value is a constant scalar or single-element tensor with value 1.0
// Supports both float and integer types
// For float types, uses tolerance check (1e-6) for floating point comparison
bool isConstOne(Value v, double tolerance = 1e-6);

// Check if MulOp is a scalar multiplication (one operand is scalar or single-element tensor)
// Returns the scalar value and the tensor operand
// This function checks both operands to identify which one is scalar
bool isScalarMul(MulOp mulOp, double &scalarVal, Value &tensorOperand);

// Match both operands of a binary operation with specific Op types, considering commutativity.
template <typename TargetLhsOpType, typename TargetRhsOpType>
inline bool matchCommutativeOperands(Value x, Value y, TargetLhsOpType &matchedLhsOp, TargetRhsOpType &matchedRhsOp) {
  // Try pattern: lhs matches TargetLhsOpType, rhs matches TargetRhsOpType
  if (auto xOp = x.getDefiningOp<TargetLhsOpType>()) {
    if (auto yOp = y.getDefiningOp<TargetRhsOpType>()) {
      matchedLhsOp = xOp;
      matchedRhsOp = yOp;
      return true;
    }
  }

  // Try pattern: lhs matches TargetRhsOpType, rhs matches TargetLhsOpType
  if (auto xOp = x.getDefiningOp<TargetRhsOpType>()) {
    if (auto yOp = y.getDefiningOp<TargetLhsOpType>()) {
      matchedLhsOp = yOp;
      matchedRhsOp = xOp;
      return true;
    }
  }

  return false;
}

}  // namespace muse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MUSE_UTILS_ARITH_UTILS_H
