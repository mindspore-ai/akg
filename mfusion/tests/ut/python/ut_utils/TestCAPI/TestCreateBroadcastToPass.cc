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

// UT-only pass that creates a single mfuse.broadcast_to using the C++ builder.
// The target output shape is provided via a function attribute
//   mfuse.ut_broadcast_to_outshape = [i64, i64, ...]
// where -1 denotes a dynamic dimension.
struct MfuseUtCreateBroadcastToPass
    : public mlir::PassWrapper<MfuseUtCreateBroadcastToPass, mlir::OperationPass<mlir::func::FuncOp>> {
  // cppcheck-suppress unknownMacro
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MfuseUtCreateBroadcastToPass)

  mlir::StringRef getArgument() const final { return "mfuse-ut-create-broadcast-to"; }
  mlir::StringRef getDescription() const final {
    return "UT-only pass: create mfuse.broadcast_to via C++ builder to trigger symbolic inference and verifier";
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

    auto outShapeAttr = func->getAttrOfType<mlir::ArrayAttr>("mfuse.ut_broadcast_to_outshape");
    if (!outShapeAttr) {
      return;
    }

    llvm::SmallVector<int64_t, 4> outShape;
    outShape.reserve(outShapeAttr.size());
    for (mlir::Attribute attr : outShapeAttr) {
      auto intAttr = attr.dyn_cast<mlir::IntegerAttr>();
      if (!intAttr) {
        return;
      }
      int64_t value = intAttr.getInt();
      if (value == -1) {
        outShape.push_back(mlir::ShapedType::kDynamic);
      } else {
        outShape.push_back(value);
      }
    }

    auto outType = mlir::RankedTensorType::get(outShape, inputType.getElementType());

    mlir::Block &entry = func.getBody().front();
    mlir::OpBuilder builder(func.getContext());
    builder.setInsertionPoint(entry.getTerminator());
    (void)builder.create<mlir::mfuse::BroadcastToOp>(func.getLoc(), outType, func.getArgument(0));
  }
};

static mlir::PassRegistration<MfuseUtCreateBroadcastToPass> passReg;

}  // namespace
