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

#include "mfusion/Dialect/Mfuse/Transforms/Outlining/OutlineMfuseFusedSubgraphs.h"
#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_OUTLINEMFUSEFUSEDSUBGRAPHS
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

static func::FuncOp createOutlinedFunc(ModuleOp module, func::FuncOp parentFunc, mfuse::FusedOp fusedOp,
                                       unsigned index) {
  Location loc = fusedOp.getLoc();
  std::string funcName = (parentFunc.getName() + "_fused_" + std::to_string(index)).str();
  auto funcType = FunctionType::get(module.getContext(), fusedOp.getOperandTypes(), fusedOp.getResultTypes());

  OpBuilder moduleBuilder(module.getBodyRegion());
  auto outlinedFunc = func::FuncOp::create(loc, funcName, funcType);
  moduleBuilder.insert(outlinedFunc);
  outlinedFunc.setPrivate();
  outlinedFunc->setAttr(mfusion_attrs::kOutlined, UnitAttr::get(module.getContext()));
  if (auto fusionTypeAttr = fusedOp.getFusionTypeAttr()) {
    outlinedFunc->setAttr(mfusion_attrs::kFusionType, fusionTypeAttr);
  }

  Block *entryBlock = outlinedFunc.addEntryBlock();
  OpBuilder funcBuilder(entryBlock, entryBlock->begin());

  Block &fusedBody = fusedOp.getBody().front();
  IRMapping mapper;
  for (auto [oldArg, newArg] : llvm::zip(fusedBody.getArguments(), entryBlock->getArguments())) {
    mapper.map(oldArg, newArg);
  }

  bool createdReturn = false;
  for (Operation &op : fusedBody.getOperations()) {
    if (auto yieldOp = dyn_cast<mfuse::YieldOp>(&op)) {
      SmallVector<Value> returnValues;
      returnValues.reserve(yieldOp.getNumOperands());
      for (Value operand : yieldOp.getOperands()) {
        returnValues.push_back(mapper.lookup(operand));
      }
      funcBuilder.create<func::ReturnOp>(yieldOp.getLoc(), returnValues);
      createdReturn = true;
      continue;
    }

    funcBuilder.clone(op, mapper);
  }

  if (!createdReturn) {
    // In well-formed IR, this should never trigger:
    // - `mfuse.fused` is declared as a single-block region with an implicit
    //   terminator `mfuse.yield` (see `SingleBlockImplicitTerminator<"YieldOp">`),
    //   so the body must contain a `mfuse.yield` terminator.
    // - The clustering path that builds `mfuse.fused` also explicitly creates
    //   `mfuse.yield` for the fused body.
    //
    // Keep this as a defensive fallback to ensure the outlined `func.func` always
    // has a terminator even if upstream produces malformed/partially-constructed
    // `mfuse.fused` IR.
    funcBuilder.create<func::ReturnOp>(loc, ValueRange{});
  }

  return outlinedFunc;
}

static void replaceFusedOpWithCall(mfuse::FusedOp fusedOp, func::FuncOp outlinedFunc) {
  OpBuilder builder(fusedOp);
  auto callOp = builder.create<func::CallOp>(fusedOp.getLoc(), outlinedFunc, fusedOp.getOperands());
  fusedOp.replaceAllUsesWith(callOp.getResults());
  fusedOp.erase();
}

struct OutlineMfuseFusedSubgraphsPass : public impl::OutlineMfuseFusedSubgraphsBase<OutlineMfuseFusedSubgraphsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::StringMap<unsigned> funcClusterIndices;

    module.walk([&](func::FuncOp funcOp) {
      SmallVector<mfuse::FusedOp> fusedOps;
      funcOp.walk([&](mfuse::FusedOp fusedOp) { fusedOps.push_back(fusedOp); });

      for (mfuse::FusedOp fusedOp : fusedOps) {
        unsigned index = funcClusterIndices[funcOp.getName()]++;
        func::FuncOp outlinedFunc = createOutlinedFunc(module, funcOp, fusedOp, index);
        replaceFusedOpWithCall(fusedOp, outlinedFunc);
      }
    });
  }
};

}  // namespace
}  // namespace mfuse

std::unique_ptr<Pass> createOutlineMfuseFusedSubgraphsPass() {
  return std::make_unique<mfuse::OutlineMfuseFusedSubgraphsPass>();
}

}  // namespace mlir
