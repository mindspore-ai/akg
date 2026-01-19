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

#include "mfusion/Conversion/TorchToMuse/TorchArithToMuse.h"

#include <numeric>
#include <unordered_map>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"
#include "mfusion/Dialect/Muse/Muse.h"

namespace mlir {
namespace {

using torch::Torch::AtenAddIntOp;
using torch::Torch::AtenFloordivIntOp;
using torch::Torch::AtenMulIntOp;
using torch::Torch::AtenRemainderIntOp;
using torch::Torch::AtenSubIntOp;
using torch::Torch::m_TorchConstantInt;

// Helper class to build AffineExpr from Torch Integer Operations
class AffineExprBuilder {
 public:
  AffineExprBuilder(MLIRContext *ctx, SmallVectorImpl<Value> &operands) : ctx(ctx), operands(operands) {}

  AffineExpr build(Value val) {
    // Check if the value is a constant integer
    int64_t constVal;
    if (matchPattern(val, m_TorchConstantInt(&constVal))) {
      return getAffineConstantExpr(constVal, ctx);
    }

    // Check if the value is defined by a supported Torch operation
    Operation *defOp = val.getDefiningOp();
    if (!defOp) {
      return getOrAddSymbol(val);
    }

    // Recursively build AffineExpr for supported operations
    if (auto op = dyn_cast<AtenAddIntOp>(defOp)) {
      return build(op.getOperand(0)) + build(op.getOperand(1));
    } else if (auto op = dyn_cast<AtenSubIntOp>(defOp)) {
      return build(op.getOperand(0)) - build(op.getOperand(1));
    } else if (auto op = dyn_cast<AtenMulIntOp>(defOp)) {
      return build(op.getOperand(0)) * build(op.getOperand(1));
    } else if (auto op = dyn_cast<AtenFloordivIntOp>(defOp)) {
      return build(op.getOperand(0)).floorDiv(build(op.getOperand(1)));
    } else if (auto op = dyn_cast<AtenRemainderIntOp>(defOp)) {
      return build(op.getOperand(0)) % build(op.getOperand(1));
    }

    // If not supported, treat as a symbolic leaf
    return getOrAddSymbol(val);
  }

 private:
  MLIRContext *ctx;
  SmallVectorImpl<Value> &operands;

  AffineExpr getOrAddSymbol(Value val) {
    // Check if the value is already in the operands list
    for (size_t i = 0; i < operands.size(); ++i) {
      if (operands[i] == val) {
        return getAffineSymbolExpr(i, ctx);
      }
    }
    // Add new operand
    operands.push_back(val);
    return getAffineSymbolExpr(operands.size() - 1, ctx);
  }
};

template <typename OpType>
struct TorchArithToMuseFusion : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 4> operands;
    AffineExprBuilder builder(rewriter.getContext(), operands);

    // Construct the expression by looking at the op's explicit behavior
    // and recursively building its operands.
    AffineExpr lhs = builder.build(op.getOperand(0));
    AffineExpr rhs = builder.build(op.getOperand(1));
    AffineExpr resultExpr;

    if (std::is_same<OpType, AtenAddIntOp>::value) {
      resultExpr = lhs + rhs;
    } else if (std::is_same<OpType, AtenSubIntOp>::value) {
      resultExpr = lhs - rhs;
    } else if (std::is_same<OpType, AtenMulIntOp>::value) {
      resultExpr = lhs * rhs;
    } else if (std::is_same<OpType, AtenFloordivIntOp>::value) {
      resultExpr = lhs.floorDiv(rhs);
    } else if (std::is_same<OpType, AtenRemainderIntOp>::value) {
      resultExpr = lhs % rhs;
    } else {
      return failure();
    }

    // Create the AffineMap: ()[s0, s1, ...] -> (expr)
    auto map = AffineMap::get(0, operands.size(), resultExpr);
    auto mapAttr = AffineMapAttr::get(map);

    // Convert operands to Muse types (muse.i64)
    SmallVector<Value, 4> convertedOperands;
    for (Value v : operands) {
      Value newV = rewriter.getRemappedValue(v);
      if (!newV) newV = v;

      Type targetType = this->getTypeConverter()->convertType(v.getType());
      if (!targetType) return failure();

      if (newV.getType() != targetType) {
        Value converted = rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), targetType, newV).getResult(0);
        convertedOperands.push_back(converted);
      } else {
        convertedOperands.push_back(newV);
      }
    }

    auto resultType = mlir::muse::I64Type::get(rewriter.getContext());
    rewriter.replaceOpWithNewOp<mlir::muse::EvalSymbolicExprOp>(op, resultType, convertedOperands, mapAttr);
    return success();
  }
};

}  // namespace

void populateArithToMuseConversionPatterns(TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TorchArithToMuseFusion<AtenAddIntOp>>(converter, patterns.getContext());
  patterns.add<TorchArithToMuseFusion<AtenSubIntOp>>(converter, patterns.getContext());
  patterns.add<TorchArithToMuseFusion<AtenMulIntOp>>(converter, patterns.getContext());
  patterns.add<TorchArithToMuseFusion<AtenFloordivIntOp>>(converter, patterns.getContext());
  patterns.add<TorchArithToMuseFusion<AtenRemainderIntOp>>(converter, patterns.getContext());
}

}  // namespace mlir
