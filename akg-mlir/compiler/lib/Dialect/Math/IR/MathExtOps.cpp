// ===-- MathExtOps.cpp - conversion from Math to libm calls ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//  MathExtOps.cpp   based on MathOps.cpp
// ===----------------------------------------------------------------------===//
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

#include <optional>
#include "akg/Dialect/Math/IR/MathExtOps.h"
#include "akg/Dialect/Math/IR/MathExtOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::mathExt;
using namespace mlir::arith;

namespace {
/// This class defines the interface for handling inlining with math
/// operations.
struct MathExtInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All operations within math ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final { return true; }
};
}  // namespace

/// Return the type of the same shape (scalar, vector or tensor) containing i1.
static Type getI1SameShape(const Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorType = dyn_cast<RankedTensorType>(type)) {
    return RankedTensorType::get(tensorType.getShape(), i1Type);
  }
  if (isa<UnrankedTensorType>(type)) {
    return UnrankedTensorType::get(i1Type);
  }
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return VectorType::get(vectorType.getShape(), i1Type, vectorType.getScalableDims());
  }
  return i1Type;
}

void mlir::mathExt::MathExtDialect::initialize() {
  addOperations<
#ifndef GET_OP_LIST
#define GET_OP_LIST
#include "akg/Dialect/Math/IR/MathExtOps.cpp.inc"
#endif
    >();
  addInterfaces<MathExtInlinerInterface>();
}

#ifndef GET_OP_CLASSES
#define GET_OP_CLASSES
#include "akg/Dialect/Math/IR/MathExtOps.cpp.inc"
#endif

// ===----------------------------------------------------------------------===//
// AsinOp folder
// ===----------------------------------------------------------------------===//

OpFoldResult mathExt::AsinOp::fold(FoldAdaptor adaptor) {
  const uint64_t width64 = 64, width32 = 32;
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
    switch (APFloat::getSizeInBits(a.getSemantics())) {
      case width64:
        return APFloat(asin(a.convertToDouble()));
      case width32:
        return APFloat(asinf(a.convertToFloat()));
      default:
        return {};
    }
  });
}

// ===----------------------------------------------------------------------===//
// AcosOp folder
// ===----------------------------------------------------------------------===//
OpFoldResult mathExt::AcosOp::fold(FoldAdaptor adaptor) {
  const uint64_t width64 = 64, width32 = 32;
  return constFoldUnaryOpConditional<FloatAttr>(adaptor.getOperands(), [](const APFloat &a) -> std::optional<APFloat> {
    switch (APFloat::getSizeInBits(a.getSemantics())) {
      case width64:
        return APFloat(acos(a.convertToDouble()));
      case width32:
        return APFloat(acosf(a.convertToFloat()));
      default:
        return {};
    }
  });
}

static Attribute getBoolAttribute(Type type, MLIRContext *ctx, bool value) {
  auto boolAttr = BoolAttr::get(ctx, value);
  ShapedType shapedType = llvm::dyn_cast_or_null<ShapedType>(type);
  if (!shapedType) {
    return boolAttr;
  }
  return DenseElementsAttr::get(shapedType, boolAttr);
}
