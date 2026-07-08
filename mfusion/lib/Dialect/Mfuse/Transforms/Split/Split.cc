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

#include "mfusion/Dialect/Mfuse/Transforms/Split/Split.h"

#include "mfusion/Analysis/Split/SplitModel.h"
#include "mfusion/Support/Logging.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/FuseOpSplitter.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"
#include "mfusion/Dialect/Dvm/IR/Dvm.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {

#define GEN_PASS_DECL_SPLIT
#define GEN_PASS_DEF_SPLIT
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace {

bool isHoistableFusedInputOp(Operation *op) { return isa<mfuse::ReshapeOp>(op); }

bool prunePassthroughFusedResults(FusedOp fusedOp) {
  Block &body = fusedOp.getBodyBlock();
  auto yieldOp = dyn_cast<YieldOp>(body.getTerminator());
  if (!yieldOp || fusedOp.getNumResults() == 0) {
    return false;
  }

  SmallVector<Value> replacements(fusedOp.getNumResults());
  SmallVector<Value> keptYieldValues;
  SmallVector<Type> keptResultTypes;

  for (auto [index, yielded] : llvm::enumerate(yieldOp.getValues())) {
    Value replacement;
    if (auto blockArg = dyn_cast<BlockArgument>(yielded)) {
      if (blockArg.getOwner() == &body && blockArg.getArgNumber() < fusedOp.getNumOperands()) {
        replacement = fusedOp.getOperand(blockArg.getArgNumber());
      }
    }

    if (replacement && replacement.getType() == fusedOp.getResult(index).getType()) {
      replacements[index] = replacement;
      continue;
    }

    keptYieldValues.push_back(yielded);
    keptResultTypes.push_back(fusedOp.getResult(index).getType());
  }

  if (keptYieldValues.size() == yieldOp.getNumOperands() || keptYieldValues.empty()) {
    return false;
  }

  OpBuilder builder(fusedOp);
  auto newFusedOp = builder.create<FusedOp>(fusedOp.getLoc(), keptResultTypes, fusedOp.getInputs(),
                                           fusedOp.getFusionTypeAttr(), fusedOp.getKernelNameAttr());
  newFusedOp->setAttrs(fusedOp->getAttrDictionary());
  newFusedOp.getBody().takeBody(fusedOp.getBody());

  auto newYieldOp = cast<YieldOp>(newFusedOp.getBodyBlock().getTerminator());
  newYieldOp->setOperands(keptYieldValues);

  unsigned keptIndex = 0;
  for (auto [index, oldResult] : llvm::enumerate(fusedOp.getResults())) {
    if (replacements[index]) {
      oldResult.replaceAllUsesWith(replacements[index]);
      continue;
    }
    oldResult.replaceAllUsesWith(newFusedOp.getResult(keptIndex++));
  }
  fusedOp.erase();
  return true;
}

void hoistFusedInputOps(FusedOp fusedOp) {
  Block &body = fusedOp.getBodyBlock();
  OpBuilder builder(fusedOp);

  for (auto [index, arg] : llvm::enumerate(body.getArguments())) {
    if (index >= fusedOp.getNumOperands()) {
      continue;
    }

    while (true) {
      Value externalInput = fusedOp.getOperand(index);
      if (!arg.hasOneUse()) {
        break;
      }

      Operation *innerOp = *arg.user_begin();
      if (!isHoistableFusedInputOp(innerOp) || innerOp->getNumOperands() != 1 || innerOp->getOperand(0) != arg ||
          innerOp->getNumResults() != 1) {
        break;
      }

      IRMapping mapping;
      mapping.map(arg, externalInput);
      builder.setInsertionPoint(fusedOp);
      Operation *outerOp = builder.clone(*innerOp, mapping);
      Value outerResult = outerOp->getResult(0);
      Value innerResult = innerOp->getResult(0);

      fusedOp->setOperand(index, outerResult);
      arg.setType(outerResult.getType());
      innerResult.replaceAllUsesWith(arg);
      innerOp->erase();
    }
  }
  prunePassthroughFusedResults(fusedOp);
}

void hoistFusedInputOps(func::FuncOp funcOp, llvm::StringRef fusionType) {
  SmallVector<FusedOp> fuseOps;
  funcOp.walk([&](FusedOp fuseOp) { fuseOps.push_back(fuseOp); });
  for (FusedOp fuseOp : fuseOps) {
    auto fusedOpType = fuseOp.getFusionType();
    if (!fusedOpType || fusedOpType.value() != fusionType) {
      continue;
    }
    hoistFusedInputOps(fuseOp);
  }
}

}  // namespace

struct SplitPass : public impl::SplitBase<SplitPass> {
  using SplitBase::SplitBase;

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    SmallVector<FusedOp> fuseOps;
    func_op.walk([&](FusedOp fuseOp) { fuseOps.push_back(fuseOp); });

    FuseOpSplitter fuse_op_splitter;
    for (FusedOp fuseOp : fuseOps) {
      MLOG(DEBUG) << "Try split fuseOp: " << fuseOp;
      fuse_op_splitter.trySplit(fuseOp, kernelGenerator);
    }
    hoistFusedInputOps(func_op, kernelGenerator);
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createSplitPass() { return std::make_unique<mfuse::SplitPass>(); }

}  // namespace mlir
