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

#include "akg/Dialect/Vector/Transforms/VectorLegalizeType.h"

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_VECTORLEGALIZETYPE
#define GEN_PASS_DECL_VECTORLEGALIZETYPE
#include "akg/Dialect/Vector/Passes.h.inc"

#define DEBUG_TYPE "vector-legalize-type"

namespace {

// Convert arith.uitofp : i8 -> f32 into arith.extui : i8 -> i16 followed by arith.uitofp : i16 -> f32
struct UIToFPLegalizePattern : public OpRewritePattern<arith::UIToFPOp> {
  using OpRewritePattern<arith::UIToFPOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::UIToFPOp op, PatternRewriter &rewriter) const override {
    Type inType = op.getIn().getType();
    Type outType = op.getOut().getType();

    Type inElemType = inType;
    Type outElemType = outType;
    if (auto vecIn = dyn_cast<VectorType>(inType)) {
      inElemType = vecIn.getElementType();
    }
    if (auto vecOut = dyn_cast<VectorType>(outType)) {
      outElemType = vecOut.getElementType();
    }

    auto intIn = dyn_cast<IntegerType>(inElemType);
    if (!intIn || intIn.getWidth() != kI8BitWidth) {
      return failure();
    }
    auto floatOut = dyn_cast<FloatType>(outElemType);
    if (!floatOut || !floatOut.isF32()) {
      return failure();
    }

    Type i16Type = IntegerType::get(rewriter.getContext(), 16);
    Type midType = i16Type;
    if (auto vecIn = dyn_cast<VectorType>(inType)) {
      midType = VectorType::get(vecIn.getShape(), i16Type);
    }

    Value ext = rewriter.create<arith::ExtUIOp>(op.getLoc(), midType, op.getIn());
    rewriter.replaceOpWithNewOp<arith::UIToFPOp>(op, outType, ext);
    return success();
  }
};

class VectorLegalizeType : public mlir::vector::impl::VectorLegalizeTypeBase<VectorLegalizeType> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    for (auto func : module.getOps<func::FuncOp>()) {
      if (!func->hasAttr(kVectorFunctionAttr)) {
        continue;
      }

      RewritePatternSet patterns(context);
      patterns.add<UIToFPLegalizePattern>(context);

      GreedyRewriteConfig config;
      config.useTopDownTraversal = true;

      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createVectorLegalizeTypePass() {
  return std::make_unique<VectorLegalizeType>();
}
}  // namespace vector
}  // namespace mlir
