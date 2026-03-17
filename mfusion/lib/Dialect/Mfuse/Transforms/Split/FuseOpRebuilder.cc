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

#include "mfusion/Dialect/Mfuse/Transforms/Split/FuseOpRebuilder.h"

#include "mfusion/Dialect/Mfuse/Support/FusedOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "split-rebuilder"

namespace mlir {
namespace mfuse {
namespace split {

Rebuilder::Rebuilder(mlir::mfuse::FusedOp fuseOp, const SplitSchemerPtr &splitSchemer,
                     const DenseMap<Value, Value> &paramToMainGraphValueMap)
    : mainFuncOp_(fuseOp.getOperation()->getParentOfType<func::FuncOp>()),
      fuseOp_(fuseOp),
      splitSchemer_(splitSchemer),
      paramToMainGraphValueMap_(paramToMainGraphValueMap) {}

void Rebuilder::rebuild() {
  if (!fuseOp_ || !mainFuncOp_) {
    return;
  }
  createFusedOps();
  connectToMainGraph();
  fuseOp_.erase();
}

void Rebuilder::createFusedOps() {
  const auto &plan = splitSchemer_->getSplitPlan();
  for (size_t i = 0; i < plan.size(); ++i) {
    const auto &groupOps = plan[i];
    if (groupOps.empty() ||
        (groupOps.size() == 1 && (isa<mfuse::YieldOp>(groupOps.front()) || isa<mfuse::ConstantOp>(groupOps.front())))) {
      continue;
    }
    // Skip cases that need inline, already inlined all ops in this fusedOp
    if (splitSchemer_->needInline(i)) {
      continue;
    }
    SmallVector<Operation *> groupOpsWithConstant(groupOps.begin(), groupOps.end());
    auto constantsToCluster = collectConstantsToCluster(groupOps);
    LLVM_DEBUG(llvm::dbgs() << "groupOps size: " << groupOps.size()
                            << " constantsToCluster size: " << constantsToCluster.size() << "\n");
    std::copy(constantsToCluster.begin(), constantsToCluster.end(), std::back_inserter(groupOpsWithConstant));
    std::sort(groupOpsWithConstant.begin(), groupOpsWithConstant.end(),
              [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
    if (groupOpsWithConstant.size() == 1) {
      continue;
    }
    // Use hash set to improve lookup efficiency, map original ops to mapped ops
    llvm::DenseSet<Operation *> groupOpSet;
    for (auto op : groupOpsWithConstant) {
      groupOpSet.insert(op);
    }
    // Collect inputs and outputs of the subgraph
    llvm::SetVector<Value> groupInputs = findExternalInputs(groupOpsWithConstant, groupOpSet);
    llvm::SetVector<Value> groupOutputs = findExternalOutputs(groupOpsWithConstant, constantsToCluster, groupOpSet);
    // No external outputs means this cluster's results are not used
    if (groupOutputs.empty()) {
      continue;
    }
    createFusedOpForGroup(groupOpsWithConstant, groupInputs, groupOutputs, groupOpSet);
  }
}

void Rebuilder::createFusedOpForGroup(const SmallVector<Operation *> &groupOps,
                                      const llvm::SetVector<Value> &groupInputs,
                                      const llvm::SetVector<Value> &groupOutputs,
                                      const llvm::DenseSet<Operation *> &groupOpSet) {
  llvm::SmallVector<Type> resultTypes;
  std::transform(groupOutputs.begin(), groupOutputs.end(), std::back_inserter(resultTypes),
                 [](Value output) { return output.getType(); });

  OpBuilder mainBuilder(fuseOp_.getContext());
  Operation *insertPoint = findValidInsertPoint(groupOps, groupInputs, groupOutputs, groupOpSet);
  if (!insertPoint) {
    return;
  }
  mainBuilder.setInsertionPointAfter(insertPoint);

  auto fusedOp = mainBuilder.create<mfuse::FusedOp>(fuseOp_.getLoc(), resultTypes, groupInputs.getArrayRef(),
                                                    fuseOp_.getFusionTypeAttr(), nullptr);
  LLVM_DEBUG(llvm::dbgs() << "Create fusedOp: " << *fusedOp << "\n");
  Block *body = new Block();
  fusedOp.getBody().push_back(body);

  llvm::SmallVector<Location> argLocs;
  std::transform(groupInputs.begin(), groupInputs.end(), std::back_inserter(argLocs),
                 [](Value input) { return input.getLoc(); });
  body->addArguments(TypeRange(groupInputs.getArrayRef()), argLocs);

  IRMapping mapping;
  for (auto [input, arg] : llvm::zip(groupInputs, body->getArguments())) {
    mapping.map(input, arg);
  }
  mainBuilder.setInsertionPointToStart(body);
  for (Operation *op : groupOps) {
    mainBuilder.clone(*op, mapping);
  }
  SmallVector<Value> yieldVals;
  std::transform(groupOutputs.begin(), groupOutputs.end(), std::back_inserter(yieldVals),
                 [&mapping](Value v) { return mapping.lookup(v); });
  mainBuilder.create<mfuse::YieldOp>(fuseOp_.getLoc(), yieldVals);
  for (auto [old_output, new_result] : llvm::zip(groupOutputs, fusedOp.getResults())) {
    Value output = old_output;
    output.replaceAllUsesWith(new_result);
  }
  for (auto it = groupOps.rbegin(); it != groupOps.rend(); ++it) {
    if ((*it)->use_empty()) {
      (*it)->erase();
    }
  }
}

void Rebuilder::connectToMainGraph() {
  Block *fusedBody = &fuseOp_.getBodyBlock();
  for (auto [arg, operand] : llvm::zip(fusedBody->getArguments(), fuseOp_.getOperands())) {
    mapping_.map(arg, operand);
  }
  OpBuilder mainBuilder(fuseOp_.getContext());
  mainBuilder.setInsertionPoint(fuseOp_);

  for (Operation &op : *fusedBody) {
    if (isa<mfuse::YieldOp>(op)) {
      continue;
    }
    mainBuilder.clone(op, mapping_);
  }
  auto yieldOp = dyn_cast<mfuse::YieldOp>(fusedBody->getTerminator());
  if (!yieldOp) {
    return;
  }

  SmallVector<Value> newOutputs;
  std::transform(yieldOp.getOperands().begin(), yieldOp.getOperands().end(), std::back_inserter(newOutputs),
                 [this](Value innerVal) { return this->mapping_.lookup(innerVal); });
  for (auto [orig, newVal] : llvm::zip(fuseOp_.getResults(), newOutputs)) {
    orig.replaceAllUsesWith(newVal);
  }
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
