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

#ifndef MFUSION_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H
#define MFUSION_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mfusion {

//===----------------------------------------------------------------------===//
// Generic variadic conversion pattern template
//===----------------------------------------------------------------------===//

// Universal template for operations with arbitrary number of operands
// Usage: ConvertOp<SrcOp, DstOp, &SrcOp::Adaptor::getLhs, &SrcOp::Adaptor::getRhs>
//        ConvertOp<SrcOp, DstOp, &SrcOp::Adaptor::getSelf, &SrcOp::Adaptor::getOther, &SrcOp::Adaptor::getAlpha>
template <typename SrcOp, typename DstOp, auto... Getters>
struct ConvertOp : public mlir::OpConversionPattern<SrcOp> {
  using mlir::OpConversionPattern<SrcOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(SrcOp op, typename SrcOp::Adaptor adaptor,
                                      mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Type resultType = this->getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType) return mlir::failure();

    rewriter.replaceOpWithNewOp<DstOp>(op, resultType, (adaptor.*Getters)()...);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Utility macros for getter construction
//===----------------------------------------------------------------------===//

// Helper macro for concatenation
#define MFUSION_CONCAT_IMPL(x, y) x##y
#define MFUSION_CONCAT(x, y) MFUSION_CONCAT_IMPL(x, y)

// Convert parameter name to getter (e.g., Lhs -> getLhs)
#define MFUSION_GETTER(SrcOp, param) &SrcOp::Adaptor::MFUSION_CONCAT(get, param)

//===----------------------------------------------------------------------===//
// Simplified conversion macros
//===----------------------------------------------------------------------===//

// Universal conversion macro supporting 1-5 operands
// Usage: CONVERT_OP(SrcOp, DstOp, Lhs, Rhs) -> getLhs(), getRhs()
//        CONVERT_OP(SrcOp, DstOp, Self, Other, Alpha) -> getSelf(), getOther(), getAlpha()

#define MFUSION_CONVERT_1(SrcOp, DstOp, P1) ::mfusion::ConvertOp<SrcOp, DstOp, MFUSION_GETTER(SrcOp, P1)>

#define MFUSION_CONVERT_2(SrcOp, DstOp, P1, P2) \
  ::mfusion::ConvertOp<SrcOp, DstOp, MFUSION_GETTER(SrcOp, P1), MFUSION_GETTER(SrcOp, P2)>

#define MFUSION_CONVERT_3(SrcOp, DstOp, P1, P2, P3) \
  ::mfusion::ConvertOp<SrcOp, DstOp, MFUSION_GETTER(SrcOp, P1), MFUSION_GETTER(SrcOp, P2), MFUSION_GETTER(SrcOp, P3)>

#define MFUSION_CONVERT_4(SrcOp, DstOp, P1, P2, P3, P4)                                                               \
  ::mfusion::ConvertOp<SrcOp, DstOp, MFUSION_GETTER(SrcOp, P1), MFUSION_GETTER(SrcOp, P2), MFUSION_GETTER(SrcOp, P3), \
                       MFUSION_GETTER(SrcOp, P4)>

#define MFUSION_CONVERT_5(SrcOp, DstOp, P1, P2, P3, P4, P5)                                                           \
  ::mfusion::ConvertOp<SrcOp, DstOp, MFUSION_GETTER(SrcOp, P1), MFUSION_GETTER(SrcOp, P2), MFUSION_GETTER(SrcOp, P3), \
                       MFUSION_GETTER(SrcOp, P4), MFUSION_GETTER(SrcOp, P5)>

// Macro overloading based on number of arguments
#define MFUSION_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, NAME, ...) NAME

#define CONVERT_OP(...)                                                                                      \
  MFUSION_GET_MACRO(__VA_ARGS__, MFUSION_CONVERT_5, MFUSION_CONVERT_4, MFUSION_CONVERT_3, MFUSION_CONVERT_2, \
                    MFUSION_CONVERT_1)                                                                       \
  (__VA_ARGS__)

}  // namespace mfusion

#endif  // MFUSION_CONVERSION_CONVERSION_PATTERN_TEMPLATES_H
