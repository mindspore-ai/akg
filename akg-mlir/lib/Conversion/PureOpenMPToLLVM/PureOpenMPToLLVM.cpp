/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Conversion/PureOpenMPToLLVM/PureOpenMPToLLVM.h"
#include "akg/Conversion/Passes.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

#include "mlir/IR/MLIRContext.h"

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DEF_PUREOPENMPTOLLVM
#define GEN_PASS_DEF_PUREOPENMPTOLLVM
#ifndef GEN_PASS_DECL_PUREOPENMPTOLLVM
#define GEN_PASS_DECL_PUREOPENMPTOLLVM
#include "akg/Conversion/Passes.h.inc"
#endif
#endif
}  // namespace mlir

namespace {
namespace cf = mlir::cf;
namespace impl = mlir::impl;
namespace omp = mlir::omp;
using mlir::applyPartialConversion;
using mlir::configureOpenMPToLLVMConversionLegality;
using mlir::failed;
using mlir::LLVMConversionTarget;
using mlir::LLVMTypeConverter;
using mlir::ModuleOp;
using mlir::OperationPass;
using mlir::populateOpenMPToLLVMConversionPatterns;
using mlir::RewritePatternSet;

class PureOpenMPToLLVM : public impl::PureOpenMPToLLVMBase<PureOpenMPToLLVM> {
 public:
  PureOpenMPToLLVM() = default;
  void runOnOperation() override;
};

void PureOpenMPToLLVM::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateOpenMPToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  target.addLegalOp<omp::TerminatorOp, omp::TaskyieldOp, omp::FlushOp, omp::BarrierOp, omp::TaskwaitOp>();
  configureOpenMPToLLVMConversionLegality(target, converter);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

namespace mlir {
std::unique_ptr<OperationPass<mlir::ModuleOp>> createPureOpenMPToLLVM() { return std::make_unique<PureOpenMPToLLVM>(); }
}  // namespace mlir
