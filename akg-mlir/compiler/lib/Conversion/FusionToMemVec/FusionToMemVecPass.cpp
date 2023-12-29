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

#include "akg/Conversion/FusionToMemVec/FusionToMemVecPass.h"

#include "akg/Conversion/FusionToMemVec/FusionToMemVec.h"
#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
#ifndef GEN_PASS_DEF_CONVERTFUSIONTOMEMVEC
#define GEN_PASS_DEF_CONVERTFUSIONTOMEMVEC
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

using namespace mlir;

namespace {
class ConvertFusionToMemVecPass : public impl::ConvertFusionToMemVecBase<ConvertFusionToMemVecPass> {
  void runOnOperation() override;
};
}  // namespace

void ConvertFusionToMemVecPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<memref::MemRefDialect>();
  target.addLegalDialect<vector::VectorDialect>();
  target.addLegalDialect<arith::ArithDialect>();
  target.addIllegalDialect<fusion::FusionDialect>();

  RewritePatternSet patterns(context);
  populateFusionPatterns(patterns);

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  };
}

std::unique_ptr<OperationPass<>> mlir::createLowerFusionPass() { return std::make_unique<ConvertFusionToMemVecPass>(); }

