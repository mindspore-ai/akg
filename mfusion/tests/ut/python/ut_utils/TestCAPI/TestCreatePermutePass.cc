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

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/TypeID.h"

namespace {

// UT-only pass that creates a single mfuse.permute using the C++ builder.
// The permutation is provided via a function attribute:
//   mfuse.ut_permute = [i64, i64, ...]
// The initial result type is intentionally dynamic on every axis so tests can
// verify that symbolic builder inference refines static permuted dimensions.
struct MfuseUtCreatePermutePass
    : public mlir::PassWrapper<MfuseUtCreatePermutePass, mlir::OperationPass<mlir::func::FuncOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MfuseUtCreatePermutePass)

  mlir::StringRef getArgument() const final { return "mfuse-ut-create-permute"; }
  mlir::StringRef getDescription() const final {
    return "UT-only pass: create mfuse.permute via C++ builder to trigger symbolic inference";
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

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(func.getArgument(0).getType());
    if (!inputType) {
      return;
    }

    auto permAttr = func->getAttrOfType<mlir::ArrayAttr>("mfuse.ut_permute");
    if (!permAttr || permAttr.size() != static_cast<size_t>(inputType.getRank())) {
      return;
    }

    llvm::SmallVector<int64_t, 4> outShape(static_cast<size_t>(inputType.getRank()), mlir::ShapedType::kDynamic);
    auto outType = mlir::RankedTensorType::get(outShape, inputType.getElementType());

    mlir::Block &entry = func.getBody().front();
    mlir::OpBuilder builder(func.getContext());
    builder.setInsertionPoint(entry.getTerminator());
    (void)builder.create<mlir::mfuse::PermuteOp>(func.getLoc(), outType, func.getArgument(0), permAttr);
  }
};

static mlir::PassRegistration<MfuseUtCreatePermutePass> passReg;

}  // namespace
