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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

namespace mlir {
namespace mfuse {
// Common constants
constexpr float kOne = 1.0f;
constexpr float kNegOne = -1.0f;
constexpr float kHalf = 0.5f;
constexpr float kTwo = 2.0f;
constexpr float kThree = 3.0f;
constexpr float kMinTanhClampValue = -8.8f;
constexpr float kMaxTanhClampValue = 8.8f;

class Expr;

class ComputeOpBuilder {
 public:
  // ComputeOpBuilder constructor
  ComputeOpBuilder(PatternRewriter &rewriter, Location loc) : rewriter(rewriter), loc(loc) {}

  // ComputeOpBuilder methods
  Value add(Value a, Value b) { return rewriter.create<AddOp>(loc, a, b); }
  Value sub(Value a, Value b) { return rewriter.create<SubOp>(loc, a, b); }
  Value mul(Value a, Value b) { return rewriter.create<MulOp>(loc, a, b); }
  Value div(Value a, Value b) { return rewriter.create<DivOp>(loc, a, b); }

  // Scalar overloads using CRTP-like pattern to avoid code duplication
  template <typename T>
  Value add(Value a, T scalar) {
    return binOpWithScalar(a, scalar, [&](Value a, Value b) { return add(a, b); });
  }

  template <typename T>
  Value sub(Value a, T scalar) {
    return binOpWithScalar(a, scalar, [&](Value a, Value b) { return sub(a, b); });
  }

  template <typename T>
  Value mul(Value a, T scalar) {
    return binOpWithScalar(a, scalar, [&](Value a, Value b) { return mul(a, b); });
  }

  template <typename T>
  Value div(Value a, T scalar) {
    return binOpWithScalar(a, scalar, [&](Value a, Value b) { return div(a, b); });
  }

  // Other operations
  Value cast(Value input, Type dtype) { return rewriter.create<CastOp>(loc, input, dtype); }

  Value exp(Value x) { return rewriter.create<ExpOp>(loc, x); }

  Value reciprocal(Value x) { return rewriter.create<ReciprocalOp>(loc, x); }

  Value tanh(Value x) { return rewriter.create<AclnnTanhOp>(loc, x); }

  Value clamp(Value a, Value b, Value c) {
    auto resultType = a.getType();
    return rewriter.create<AclnnClampOp>(loc, resultType, a, b, c);
  }

  template <typename T>
  Value clamp(Value a, T b, T c) {
    auto resultType = a.getType();
    auto elemType = mlir::cast<RankedTensorType>(resultType).getElementType();
    auto value_b = createScalarRankTensor(b, elemType);
    auto value_c = createScalarRankTensor(c, elemType);
    return rewriter.create<AclnnClampOp>(loc, resultType, a, value_b, value_c);
  }

  template <typename T>
  Value createScalarRankTensor(T value, Type elementType) {
    static_assert(std::is_arithmetic_v<T>, "Value must be arithmetic type");
    auto tensorType = RankedTensorType::get({}, elementType);
    auto denseAttr = DenseElementsAttr::get(tensorType, value);
    return rewriter.create<mlir::mfuse::ConstantOp>(loc, tensorType, denseAttr).getResult();
  }

  // Create Expr from Value
  Expr buildExpr(Value value);

 private:
  // Helper function to handle binary operations with scalars
  template <typename T, typename OpFunc>
  Value binOpWithScalar(Value a, T scalar, OpFunc opFunc) {
    auto tensorType = mlir::cast<RankedTensorType>(a.getType());
    auto b = createScalarRankTensor(scalar, tensorType.getElementType());
    return opFunc(a, b);
  }

  PatternRewriter &rewriter;
  Location loc;

  friend class Expr;
};

// Expr wrapper class to support operator overloading
class Expr {
 public:
  Expr(ComputeOpBuilder &builder, Value value) : builder(builder), value(value) {}

  // Implementations of Expr operators
  Expr operator*(const Expr &other) const {
    auto result = builder.mul(value, other.value);
    return Expr(builder, result);
  }

  Expr operator+(const Expr &other) const {
    auto result = builder.add(value, other.value);
    return Expr(builder, result);
  }

  Expr operator-(const Expr &other) const {
    auto result = builder.sub(value, other.value);
    return Expr(builder, result);
  }

  Expr operator/(const Expr &other) const {
    auto result = builder.div(value, other.value);
    return Expr(builder, result);
  }

  Value getValue() const { return value; }

  ComputeOpBuilder &getBuilder() const { return builder; }

 private:
  ComputeOpBuilder &builder;
  Value value;

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
inline Expr ComputeOpBuilder::buildExpr(Value value) { return Expr(*this, value); }

// Global operator overloads for scalar operations
template <typename T>
Expr operator+(T scalar, const Expr &expr) {
  Value result = expr.builder.add(expr.value, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator+(const Expr &expr, T scalar) {
  Value result = expr.builder.add(expr.value, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator*(T scalar, const Expr &expr) {
  Value result = expr.builder.mul(expr.value, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator*(const Expr &expr, T scalar) {
  Value result = expr.builder.mul(expr.value, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator-(T scalar, const Expr &expr) {
  Value neg_tensor = expr.builder.mul(expr.value, kNegOne);
  Value result = expr.builder.add(neg_tensor, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator-(const Expr &expr, T scalar) {
  Value result = expr.builder.sub(expr.value, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator/(T scalar, const Expr &expr) {
  Value reciprocal_tensor = expr.builder.reciprocal(expr.value);
  Value result = expr.builder.mul(reciprocal_tensor, scalar);
  return Expr(expr.builder, result);
}

template <typename T>
Expr operator/(const Expr &expr, T scalar) {
  Value result = expr.builder.div(expr.value, scalar);
  return Expr(expr.builder, result);
}

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_TRANSFORMS_COMPUTE_OP_BUILDER_H
