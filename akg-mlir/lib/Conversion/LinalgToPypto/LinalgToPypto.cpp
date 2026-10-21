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

#include "akg/Conversion/LinalgToPypto/LinalgToPypto.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOPYPTO
#include "akg/Conversion/Passes.h.inc"

namespace {
class MatmulOpPattern : public OpConversionPattern<linalg::MatmulOp> {
 public:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!op.hasPureTensorSemantics()) return failure();
    rewriter.replaceOpWithNewOp<pypto::MatmulOp>(op, op.getResultTypes(), adaptor.getInputs()[0],
                                                 adaptor.getInputs()[1], adaptor.getOutputs()[0]);
    return success();
  }
};

struct ConvertLinalgToPypto : impl::ConvertLinalgToPyptoBase<ConvertLinalgToPypto> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pypto::PyptoDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<pypto::PyptoDialect>();
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpPattern>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> createLinalgToPyptoConversionPass() { return std::make_unique<ConvertLinalgToPypto>(); }
}  // namespace mlir
