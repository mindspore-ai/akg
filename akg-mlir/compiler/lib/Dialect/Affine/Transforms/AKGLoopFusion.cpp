/**
 * Copyright 2035 Huawei Technologies Co., Ltd
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

 #include "akg/Dialect/Affine/Transforms/AKGLoopFusion.h"
 #include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
 #include "akg/Dialect/Affine/Analysis/AKGLoopFusionAnalyzer.h"
 #include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
 #include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
 #include "akg/Utils/AnalysisCommon.hpp"
 
 #include "llvm/ADT/DenseMap.h"
 #include "llvm/ADT/SetVector.h"
 #include "llvm/ADT/SmallVector.h"
 #include "llvm/Support/CommandLine.h"
 #include "llvm/Support/Debug.h"
 #include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
 #include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
 #include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
 #include "mlir/Dialect/Affine/Analysis/Utils.h"
 #include "mlir/Dialect/Affine/IR/AffineOps.h"
 #include "mlir/Dialect/Affine/LoopUtils.h"
 #include "mlir/Dialect/Arith/IR/Arith.h"
 #include "mlir/Dialect/Func/IR/FuncOps.h"
 #include "mlir/Dialect/MemRef/IR/MemRef.h"
 #include "mlir/Dialect/Vector/IR/VectorOps.h"
 #include "mlir/Dialect/Vector/Utils/VectorUtils.h"
 #include "mlir/IR/AffineExpr.h"
 #include "mlir/IR/AffineMap.h"
 #include "mlir/IR/Block.h"
 #include "mlir/IR/Builders.h"
 #include "mlir/IR/Value.h"

#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace mlir {
#define GEN_PASS_DEF_AKGLOOPFUSION
#define GEN_PASS_DECL_AKGLOOPFUSION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "akg-loop-fusion"

using namespace mlir;
using namespace llvm;
using namespace akg;

namespace {
struct AKGLoopFusion : public impl::AKGLoopFusionBase<AKGLoopFusion> {
  AKGLoopFusion() {}

  void runOnOperation() override;

 private:
  void runOnBlock(Block *block);
  void runPreProcess();
};
} // namespace

void AKGLoopFusion::runOnBlock(Block *block) {
  // build dependence graph
  auto dependenceGraph = MemRefDependenceGraphForFusion(block);
  if (!dependenceGraph.init()) {
    return;
  }
  dependenceGraph.dump();

  func::FuncOp funcOp = getOperation();
  FusionAnalyzer analyzer(dependenceGraph, funcOp);
  // Plan the fusion strategy based on analysis
  analyzer.plan();
  for (auto &fusion : analyzer.fusionPlans) {
    llvm::outs() << "Fusion group " << fusion.fusedGroup.from << " to " << fusion.fusedGroup.to << ", node "
                 << fusion.fusedBand.from << " to " << fusion.fusedBand.to << "\n";
  }

  if (analyzer.fusionPlans.empty()) {
    return;
  }

  FusionCodeGenHelper codegenerator = FusionCodeGenHelper(dependenceGraph);
  
  // Process each fusion plan
  for (auto &plan : analyzer.fusionPlans) {
    auto [srcId, dstId] = plan.fusedBand;
    // Get source and destination affine::AffineForOp operations from the dependence graph
    auto srcFor = dyn_cast<affine::AffineForOp>(dependenceGraph.getNode(srcId)->op);
    auto dstFor = dyn_cast<affine::AffineForOp>(dependenceGraph.getNode(dstId)->op);
    
    if (srcFor && dstFor) {
      if (plan.fusionType == "V") {
        // Vertical fusion: calculate loop depth for the destination loop
        unsigned dstLoopDepthTest = 0;
        dstFor.walk([&](affine::AffineForOp op) { dstLoopDepthTest++; });
        codegenerator.doVFuse(srcId, dstId, srcFor, dstFor, dstLoopDepthTest, dstLoopDepthTest);
      } else {
        // Horizontal fusion: fuse loops at the same nesting level
        codegenerator.doHFuse(srcId, dstId, srcFor, dstFor);
      }
    }
  }
}

/// Preprocessing step that performs loop interchange optimization for reduction operations.
/// Identifies nested loops with reduction axes and interchanges them to improve cache performance.
void AKGLoopFusion::runPreProcess() {
  func::FuncOp funcOp = getOperation();

  // The reduce axis sinks to the innermost layer.
  funcOp.walk([&](affine::AffineForOp inner) {
    if (auto outer = dyn_cast<affine::AffineForOp>(inner->getParentOp())) {
      if (CommonUtils::isReduceAxis(funcOp, inner->getParentOp())) {
        interchangeLoops(outer, inner);
      }
    }
  });

  funcOp.dump();
}

void AKGLoopFusion::runOnOperation() {
  auto funcOp = getOperation();
  
  runPreProcess();
  
  for (Region &region : funcOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      runOnBlock(&block);
    }
  }
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createAKGLoopFusionPass() {
  return std::make_unique<AKGLoopFusion>();
}
