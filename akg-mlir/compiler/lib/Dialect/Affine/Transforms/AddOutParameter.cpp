/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/AddOutParameter.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// HACC dialect enums/attrs
#include "bishengir/Dialect/HACC/IR/HACC.h"

#define DEBUG_TYPE "add-out-parameter"

namespace mlir {
#define GEN_PASS_DECL_ADDOUTPARAMETER
#define GEN_PASS_DEF_ADDOUTPARAMETER
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

namespace mlir::affine {

using hacc::InputIdxAttr;
using hacc::KernelArgType;
using hacc::KernelArgTypeAttr;
using hacc::OutputIdxAttr;

namespace {

static void setHaccIOArgAttrs(func::FuncOp f, unsigned nInputs, unsigned nOutputs, OpBuilder &builder) {
  auto *ctx = builder.getContext();

  for (unsigned i = 0; i < nInputs; ++i) {
    f.setArgAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
    f.setArgAttr(i, InputIdxAttr::name, InputIdxAttr::get(ctx, i));
  }

  for (unsigned i = 0; i < nOutputs; ++i) {
    unsigned argIdx = nInputs + i;
    f.setArgAttr(argIdx, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
    f.setArgAttr(argIdx, OutputIdxAttr::name, OutputIdxAttr::get(ctx, i));
  }
}

static LogicalResult rewriteReturnsAndAllocToUseOutParams(func::FuncOp func, unsigned origNumInputs,
                                                          unsigned origNumResults) {
  if (origNumResults == 0) return success();
  if (func.empty()) return success();

  Block &entry = func.front();
  unsigned totalArgs = entry.getNumArguments();
  if (totalArgs != origNumInputs + origNumResults) {
    func.emitError() << "entry block arg count mismatch after signature transform: expected "
                     << (origNumInputs + origNumResults) << " got " << totalArgs;
    return failure();
  }

  SmallVector<Value> outArgs;
  outArgs.reserve(origNumResults);
  for (unsigned i = 0; i < origNumResults; ++i) outArgs.push_back(entry.getArgument(origNumInputs + i));

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

  if (returns.empty()) return success();

  for (auto ret : returns) {
    auto retOperands = ret.getOperands();
    if (retOperands.size() != origNumResults) {
      ret.emitError() << "number of return values (" << retOperands.size()
                      << ") does not match number of function results (" << origNumResults << ")";
      return failure();
    }

    for (unsigned i = 0; i < origNumResults; ++i) {
      Value oldResVal = retOperands[i];
      Value outArg = outArgs[i];

      if (oldResVal.getType() != outArg.getType()) {
        ret.emitError() << "type mismatch between return value #" << i << " (" << oldResVal.getType()
                        << ") and out argument (" << outArg.getType() << ")";
        return failure();
      }

      if (oldResVal != outArg) oldResVal.replaceAllUsesWith(outArg);

      if (auto allocOp = oldResVal.getDefiningOp<memref::AllocOp>()) {
        if (allocOp->use_empty()) allocOp->erase();
      }
    }

    OpBuilder builder(ret);
    SmallVector<Value> newRetOperands(outArgs.begin(), outArgs.end());
    builder.create<func::ReturnOp>(ret.getLoc(), newRetOperands);
    ret.erase();
  }

  return success();
}

static LogicalResult transformFunc(func::FuncOp func, OpBuilder &builder) {
  auto funcTy = func.getFunctionType();
  auto *ctx = builder.getContext();

  SmallVector<Type> origInputs(funcTy.getInputs().begin(), funcTy.getInputs().end());
  SmallVector<Type> origResults(funcTy.getResults().begin(), funcTy.getResults().end());

  unsigned origNumInputs = origInputs.size();
  unsigned origNumResults = origResults.size();

  if (origNumResults == 0) {
    setHaccIOArgAttrs(func, origNumInputs, /*nOutputs=*/0, builder);
    return success();
  }

  SmallVector<Type> newInputs;
  newInputs.reserve(origNumInputs + origNumResults);
  newInputs.append(origInputs.begin(), origInputs.end());
  newInputs.append(origResults.begin(), origResults.end());

  SmallVector<Type> newResults(origResults.begin(), origResults.end());

  auto newFuncTy = FunctionType::get(ctx, newInputs, newResults);
  func.setFunctionType(newFuncTy);

  if (!func.empty()) {
    Block &entry = func.front();

    if (entry.getNumArguments() != origNumInputs) {
      func.emitError() << "unexpected number of entry block arguments before transform: got " << entry.getNumArguments()
                       << ", expected " << origNumInputs;
      return failure();
    }

    for (Type resTy : origResults) (void)entry.addArgument(resTy, func.getLoc());
  }

  setHaccIOArgAttrs(func, origNumInputs, origNumResults, builder);

  if (failed(rewriteReturnsAndAllocToUseOutParams(func, origNumInputs, origNumResults))) return failure();

  return success();
}

struct AddOutParameter : public mlir::impl::AddOutParameterBase<AddOutParameter> {
  AddOutParameter() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(module.getContext());

    SmallVector<func::FuncOp, 8> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });

    auto it = std::find_if(funcs.begin(), funcs.end(), [&](func::FuncOp f) {
      if (failed(transformFunc(f, builder))) {
        f.emitError("AddOutParameter pass failed for this function");
        return true;
      }
      return false;
    });

    if (it != funcs.end()) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace mlir::affine

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::affine::createAddOutParameterPass() {
  return std::make_unique<AddOutParameter>();
}
