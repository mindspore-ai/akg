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

#include "mfusion/Dialect/Mfuse/Mfuse.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"

namespace {

struct MfuseUtCreateReduceSumPass
    : public mlir::PassWrapper<MfuseUtCreateReduceSumPass, mlir::OperationPass<mlir::func::FuncOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MfuseUtCreateReduceSumPass)

  mlir::StringRef getArgument() const final { return "mfuse-ut-create-reduce-sum"; }
  mlir::StringRef getDescription() const final {
    return "UT-only pass: create mfuse.reduce_sum via C++ builder to trigger symbolic inference";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::mfuse::MfuseDialect>();
    registry.insert<mlir::func::FuncDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    if (func.getNumArguments() == 0) {
      return;
    }

    auto inputType = func.getArgument(0).getType().dyn_cast<mlir::RankedTensorType>();
    if (!inputType) {
      return;
    }

    auto dimsAttr = func->getAttrOfType<mlir::ArrayAttr>("mfuse.ut_reduce_sum_dims");
    auto keepdimAttr = func->getAttrOfType<mlir::BoolAttr>("mfuse.ut_reduce_sum_keepdim");
    if (!dimsAttr || !keepdimAttr) {
      return;
    }

    mlir::Block &entry = func.getBody().front();
    mlir::OpBuilder builder(func.getContext());
    builder.setInsertionPoint(entry.getTerminator());
    (void)builder.create<mlir::mfuse::ReduceSumOp>(func.getLoc(), func.getArgument(0), dimsAttr, keepdimAttr);
  }
};

static mlir::PassRegistration<MfuseUtCreateReduceSumPass> passReg;
}  // namespace
