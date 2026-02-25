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

#ifndef MFUSION_DIALECT_MFUSE_TRANSFORMS_COMPUTE_OP_BUILDER_H
#define MFUSION_DIALECT_MFUSE_TRANSFORMS_COMPUTE_OP_BUILDER_H

#include <type_traits>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"

namespace mlir {
namespace mfuse {
// Common constants
constexpr float kOne = 1.0f;      // 1
constexpr float kNegOne = -1.0f;  // -1
constexpr float kHalf = 0.5f;     // 0.5
constexpr float kTwo = 2.0f;      // 2
constexpr float kThree = 3.0f;    // 3

class Expr;

class ComputeOpBuilder {
 public:
  // ComputeOpBuilder constructor
  ComputeOpBuilder(PatternRewriter &rewriter, Location loc) : rewriter(rewriter), loc(loc) {}

  // ComputeOpBuilder methods
  Value add(Value a, Value b, Type resultType, Value alpha = nullptr) {
    if (alpha == nullptr) {
      return rewriter.create<AddOp>(loc, resultType, a, b);
    } else {
      auto mul_op = rewriter.create<MulOp>(loc, resultType, b, alpha);
      return rewriter.create<AddOp>(loc, resultType, a, mul_op);
    }
  }

  template <typename T>
  Value add(Value a, T scalar, Type resultType, Value alpha = nullptr) {
    auto b = createScalarRankTensor(scalar);
    return add(a, b, resultType, alpha);
  }

  Value cast(Value input, Type targetType) {
    auto tensorType = targetType.cast<RankedTensorType>();
    auto dtypeAttr = TypeAttr::get(tensorType.getElementType());
    return rewriter.create<CastOp>(loc, targetType, input, dtypeAttr);
  }

  Value exp(Value x, Type resultType) { return rewriter.create<ExpOp>(loc, resultType, x); }

  Value div(Value a, Value b, Type resultType) { return rewriter.create<DivOp>(loc, resultType, a, b); }

  template <typename T>
  Value div(Value a, T scalar, Type resultType) {
    auto b = createScalarRankTensor(scalar);
    return rewriter.create<DivOp>(loc, resultType, a, b);
  }

  Value mul(Value a, Value b, Type resultType) { return rewriter.create<MulOp>(loc, resultType, a, b); }

  template <typename T>
  Value mul(Value a, T scalar, Type resultType) {
    auto b = createScalarRankTensor(scalar);
    return rewriter.create<MulOp>(loc, resultType, a, b);
  }

  Value reciprocal(Value x, Type resultType) { return rewriter.create<ReciprocalOp>(loc, resultType, x); }

  Value sub(Value a, Value b, Type resultType, Value alpha = nullptr) {
    if (alpha == nullptr) {
      return rewriter.create<SubOp>(loc, resultType, a, b);
    } else {
      auto mul_op = rewriter.create<MulOp>(loc, resultType, b, alpha);
      return rewriter.create<SubOp>(loc, resultType, a, mul_op);
    }
  }

  template <typename T>
  Value sub(Value a, T scalar, Type resultType, Value alpha = nullptr) {
    auto b = createScalarRankTensor(scalar);
    return sub(a, b, resultType, alpha);
  }

  Value tanh(Value x, Type resultType) { return rewriter.create<AclnnTanhOp>(loc, resultType, x); }

  template <typename T>
  Value createScalarRankTensor(T value) {
    static_assert(std::is_arithmetic_v<T>, "Value must be arithmetic type");
    Type elementType;
    if constexpr (std::is_same_v<T, float>) {
      elementType = rewriter.getF32Type();
    } else if constexpr (std::is_same_v<T, double>) {
      elementType = rewriter.getF64Type();
    } else if constexpr (std::is_same_v<T, int32_t>) {
      elementType = rewriter.getI32Type();
    } else if constexpr (std::is_same_v<T, int64_t>) {
      elementType = rewriter.getI64Type();
    } else {
      return nullptr;
    }
    auto tensorType = RankedTensorType::get({}, elementType);
    auto denseAttr = DenseElementsAttr::get(tensorType, value);
    return rewriter.create<mlir::arith::ConstantOp>(loc, tensorType, denseAttr).getResult();
  }

  // Create Expr from Value
  Expr buildExpr(Value value, Type resultType);

 private:
  PatternRewriter &rewriter;
  Location loc;

  friend class Expr;
};

// Expr wrapper class to support operator overloading
class Expr {
 public:
  Expr(ComputeOpBuilder &builder, Value value, Type resultType)
      : builder(builder), value(value), resultType(resultType) {}

  // Implementations of Expr operators
  Expr operator*(const Expr &other) const {
    auto result = builder.mul(value, other.value, resultType);
    return Expr(builder, result, resultType);
  }

  Expr operator+(const Expr &other) const {
    auto result = builder.add(value, other.value, resultType);
    return Expr(builder, result, resultType);
  }

  Expr operator-(const Expr &other) const {
    auto result = builder.sub(value, other.value, resultType);
    return Expr(builder, result, resultType);
  }

  Expr operator/(const Expr &other) const {
    auto result = builder.div(value, other.value, resultType);
    return Expr(builder, result, resultType);
  }

  Value getValue() const { return value; }

  Type getResultType() const { return resultType; }

  ComputeOpBuilder &getBuilder() const { return builder; }

 private:
  ComputeOpBuilder &builder;
  Value value;
  Type resultType;

  friend class ComputeOpBuilder;
  template <typename T>
  friend Expr operator+(T scalar, const Expr &expr);
  template <typename T>
  friend Expr operator+(const Expr &expr, T scalar);
  template <typename T>
  friend Expr operator*(T scalar, const Expr &expr);
  template <typename T>
  friend Expr operator*(const Expr &expr, T scalar);
  template <typename T>
  friend Expr operator-(T scalar, const Expr &expr);
  template <typename T>
  friend Expr operator-(const Expr &expr, T scalar);
  template <typename T>
  friend Expr operator/(T scalar, const Expr &expr);
  template <typename T>
  friend Expr operator/(const Expr &expr, T scalar);
};

// Create Expr from Value
inline Expr ComputeOpBuilder::buildExpr(Value value, Type resultType) { return Expr(*this, value, resultType); }

// Global operator overloads for scalar operations
template <typename T>
Expr operator+(T scalar, const Expr &expr) {
  Value result = expr.builder.add(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator+(const Expr &expr, T scalar) {
  Value result = expr.builder.add(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator*(T scalar, const Expr &expr) {
  Value result = expr.builder.mul(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator*(const Expr &expr, T scalar) {
  Value result = expr.builder.mul(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator-(T scalar, const Expr &expr) {
  Value neg_tensor = expr.builder.mul(expr.value, kNegOne, expr.resultType);
  Value result = expr.builder.add(neg_tensor, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator-(const Expr &expr, T scalar) {
  Value result = expr.builder.sub(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator/(T scalar, const Expr &expr) {
  Value reciprocal_tensor = expr.builder.reciprocal(expr.value, expr.resultType);
  Value result = expr.builder.mul(reciprocal_tensor, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

template <typename T>
Expr operator/(const Expr &expr, T scalar) {
  Value result = expr.builder.div(expr.value, scalar, expr.resultType);
  return Expr(expr.builder, result, expr.resultType);
}

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_COMPUTE_OP_BUILDER_H
