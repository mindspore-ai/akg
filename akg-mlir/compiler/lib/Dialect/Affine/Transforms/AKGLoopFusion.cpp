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

#include "akg/Dialect/Affine/Transforms/AKGLoopFusion.h"

#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/Affine/Analysis/AKGLoopFusionAnalyzer.h"
#include "akg/Dialect/Affine/Analysis/AKGLoopFusionBuilder.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Analysis/SymbolicShapeAnalysis.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace mlir {
#define GEN_PASS_DEF_AKGLOOPFUSION
#define GEN_PASS_DECL_AKGLOOPFUSION
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "akg-loop-fusion"

namespace {
struct AKGLoopFusion : public mlir::impl::AKGLoopFusionBase<AKGLoopFusion> {
  AKGLoopFusion() {}

  void runOnOperation() override;

 private:
  void runOnBlock(mlir::Block *block);
  void runPreProcess();

  void replaceDimWithPrimes(mlir::func::FuncOp funcOp);
  void restoreDimFromPrimes(mlir::func::FuncOp funcOp);

  std::optional<llvm::SmallVector<std::string>> getSymShapeAttrFromValue(mlir::Value source);
  std::optional<int64_t> getConstantDimIndex(mlir::Value dimIndex);
  void collectDimOperationsFromLoops(mlir::func::FuncOp funcOp,
                                     llvm::StringMap<llvm::SmallVector<mlir::Value>> &axisToDimValues);

  // Map: prime constant -> representative dim value (first dim in the group)
  llvm::DenseMap<mlir::Value, mlir::Value> primeToDimMap;
  // Whether to print fusion information
  bool printFusionInfo{false};
  // Large primes for replacing dim values
  static constexpr int64_t kPrimes[] = {1000003, 1000033, 1000037, 1000039, 1000081,
                                        1000099, 1000117, 1000121, 1000133, 1000139};
  static constexpr size_t kNumPrimes = sizeof(kPrimes) / sizeof(kPrimes[0]);
};
}  // namespace

// Trace back through operations to find the SymShapeAttr for a Value.
// Handles block arguments, bufferization.to_memref, etc.
std::optional<llvm::SmallVector<std::string>> AKGLoopFusion::getSymShapeAttrFromValue(mlir::Value source) {
  // Get SymShapeAttr from a RankedTensorType.
  auto getSymShape = [](mlir::Type type) -> std::optional<llvm::SmallVector<std::string>> {
    // RankedTensorType stores SymShapeAttr in its encoding
    // MemRefType stores SymShapeAttr in its memorySpace
    if (mlir::isa<mlir::RankedTensorType, mlir::MemRefType>(type)) {
      return mlir::SymbolicShapeAnalysis::getInstance().getSymbolicShape(type);
    }
    return std::nullopt;
  };

  // Case 1: Direct type has SymShapeAttr
  if (auto symShape = getSymShape(source.getType())) {
    return symShape;
  }

  // Case 2: Block argument (e.g., function argument)
  if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(source)) {
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      mlir::Type argType = funcOp.getFunctionType().getInput(blockArg.getArgNumber());
      return getSymShape(argType);
    }
    return std::nullopt;
  }

  // Case 3: MemRef created from tensor via bufferization.to_memref
  if (auto *defOp = source.getDefiningOp()) {
    if (auto toMemref = mlir::dyn_cast<mlir::bufferization::ToMemrefOp>(defOp)) {
      mlir::Value tensor = toMemref.getTensor();
      if (auto symShape = getSymShape(tensor.getType())) {
        return symShape;
      }
      // Recursively trace back
      return getSymShapeAttrFromValue(tensor);
    }
  }

  return std::nullopt;
}

// Extract constant dimension index from a dim operation's index operand.
std::optional<int64_t> AKGLoopFusion::getConstantDimIndex(mlir::Value dimIndex) {
  if (auto constOp = dimIndex.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  if (auto constOp = dimIndex.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

// Collect dim operations from affine.for loop bounds and group them by axis.
void AKGLoopFusion::collectDimOperationsFromLoops(mlir::func::FuncOp funcOp,
                                                  llvm::StringMap<llvm::SmallVector<mlir::Value>> &axisToDimValues) {
  funcOp.walk([&](mlir::affine::AffineForOp forOp) {
    // Process a single operand from loop bounds
    auto processBoundOperand = [&](mlir::Value operand) {
      auto *defOp = operand.getDefiningOp();
      if (!defOp) {
        return;
      }

      // Check if operand is defined by memref.dim or tensor.dim
      bool isMemRefDim = mlir::isa<mlir::memref::DimOp>(defOp);
      bool isTensorDim = mlir::isa<mlir::tensor::DimOp>(defOp);
      if (!isMemRefDim && !isTensorDim) {
        return;
      }

      // Extract source and dimension index from dim operation
      mlir::Value source = defOp->getOperand(0);
      mlir::Value dimIndexValue = defOp->getOperand(1);

      auto constIndex = getConstantDimIndex(dimIndexValue);
      if (!constIndex.has_value()) return;  // Skip non-constant dimension index

      // Get symbolic shape and extract axis key (symbol name only)
      auto symShape = getSymShapeAttrFromValue(source);
      // Skip if no SymShapeAttr or index out of range
      if (!symShape.has_value() || *constIndex >= static_cast<int64_t>(symShape->size())) {
        return;
      }

      // Group by axis key: symbolic dimension symbol only
      std::string axisKey = (*symShape)[*constIndex];
      axisToDimValues[axisKey].push_back(operand);
    };

    // Process both upper and lower bound operands
    for (mlir::Value operand : forOp.getUpperBoundOperands()) {
      processBoundOperand(operand);
    }
    for (mlir::Value operand : forOp.getLowerBoundOperands()) {
      processBoundOperand(operand);
    }
  });
}

// Replace dynamic dim values with large prime constants before fusion.
// This allows the fusion analysis to treat loops with the same symbolic dimension as having identical bounds.
void AKGLoopFusion::replaceDimWithPrimes(mlir::func::FuncOp funcOp) {
  // Step 1: Collect dim operations and group by axis
  llvm::StringMap<llvm::SmallVector<mlir::Value>> axisToDimValues;
  collectDimOperationsFromLoops(funcOp, axisToDimValues);

  if (axisToDimValues.empty()) {
    return;
  }

  // Step 2: Replace each axis group with a unique prime constant
  mlir::OpBuilder builder(funcOp.getContext());
  size_t primeIndex = 0;

  for (auto &entry : axisToDimValues) {
    if (primeIndex >= kNumPrimes) {
      llvm::errs() << "Warning: Not enough primes for all dim axes\n";
      break;
    }

    llvm::StringRef axisKey = entry.getKey();
    auto &dimValues = entry.getValue();

    // Select first dim as representative (will be used when restoring)
    mlir::Value representativeDim = dimValues.front();
    builder.setInsertionPointAfterValue(representativeDim);
    auto primeConst = builder.create<mlir::arith::ConstantIndexOp>(representativeDim.getLoc(), kPrimes[primeIndex]);

    // Directly map prime constant to representative dim
    primeToDimMap[primeConst] = representativeDim;

    // Replace all dim values in this group with the same prime
    for (mlir::Value dimValue : dimValues) {
      dimValue.replaceAllUsesWith(primeConst);
    }

    primeIndex++;
  }
}

// Restore the original dim values after fusion is complete.
// All dim values in the same axis group are replaced with a single representative dim value.
void AKGLoopFusion::restoreDimFromPrimes(mlir::func::FuncOp funcOp) {
  // Replace each prime with its corresponding representative dim
  for (auto &[primeConst, representativeDim] : primeToDimMap) {
    primeConst.replaceAllUsesWith(representativeDim);

    // Clean up unused prime constant
    if (auto constOp = primeConst.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      if (constOp.use_empty()) {
        constOp.erase();
      }
    }
  }

  // Clear state for next run
  primeToDimMap.clear();
}

void AKGLoopFusion::runOnBlock(mlir::Block *block) {
  // build dependence graph
  auto dependenceGraph = mlir::akg::MemRefDependenceGraphForFusion(block);
  if (!dependenceGraph.init()) {
    return;
  }

  if (printFusionInfo) {
    dependenceGraph.dump();
  }

  mlir::func::FuncOp funcOp = getOperation();
  mlir::akg::FusionAnalyzer analyzer(dependenceGraph, funcOp);
  // Plan the fusion strategy based on analysis
  analyzer.plan();
  if (analyzer.fusionPlans.empty()) {
    return;
  }

  if (printFusionInfo) {
    analyzer.dump();
  }

  mlir::akg::FusionCodeGenHelper codegenerator = mlir::akg::FusionCodeGenHelper(dependenceGraph);

  // Process each fusion plan
  for (size_t i = 0; i < analyzer.fusionPlans.size(); ++i) {
    auto &plan = analyzer.fusionPlans[i];

    // Apply node alias resolution to get the actual current node IDs
    auto actualSrcId = codegenerator.getAliasId(plan.fusedBand.from);
    auto actualDstId = codegenerator.getAliasId(plan.fusedBand.to);

    // Skip if source node has been fused into destination (alias exists)
    if (actualSrcId == actualDstId) {
      continue;
    }

    // Check for conflicts with subsequent fusion plans
    bool hasConflict = false;
    for (size_t j = i + 1; j < analyzer.fusionPlans.size(); ++j) {
      auto &futurePlan = analyzer.fusionPlans[j];
      auto futureSrcId = codegenerator.getAliasId(futurePlan.fusedBand.from);
      auto futureDstId = codegenerator.getAliasId(futurePlan.fusedBand.to);

      // Check for bidirectional conflict: current (src->dst) conflicts with future (dst->src)
      if ((actualSrcId == futureDstId && actualDstId == futureSrcId) ||
          (actualSrcId == futureSrcId && actualDstId == futureDstId)) {
        hasConflict = true;
        break;
      }
    }

    if (hasConflict) {
      continue;
    }

    // Get source and destination affine::AffineForOp operations from the dependence graph
    auto srcFor = mlir::dyn_cast<mlir::affine::AffineForOp>(dependenceGraph.getNode(actualSrcId)->op);
    auto dstFor = mlir::dyn_cast<mlir::affine::AffineForOp>(dependenceGraph.getNode(actualDstId)->op);

    if (srcFor && dstFor) {
      if (plan.fusionType == "V") {
        // Vertical fusion: calculate loop depth for the destination loop
        codegenerator.doVFuse(actualSrcId, actualDstId, srcFor, dstFor);
      } else if (plan.fusionType == "H") {
        // Horizontal fusion: fuse loops at the same nesting level
        codegenerator.doHFuse(actualSrcId, actualDstId, srcFor, dstFor);
      } else {
        llvm::outs() << "Warning: Could not find valid operations for fusion plan: node " << plan.fusedBand.from
                     << " to " << plan.fusedBand.to << "\n";
      }
    } else {
      llvm::outs() << "Warning: Could not find valid operations for fusion plan: node " << plan.fusedBand.from << " to "
                   << plan.fusedBand.to << "\n";
    }
  }
}

/// Preprocessing step that performs loop interchange optimization for reduction operations.
/// Identifies nested loops with reduction axes and interchanges them to improve cache performance.
void AKGLoopFusion::runPreProcess() {
  mlir::func::FuncOp funcOp = getOperation();

  // The reduce axis sinks to the innermost layer.
  funcOp.walk([&](mlir::affine::AffineForOp inner) {
    if (auto outer = mlir::dyn_cast<mlir::affine::AffineForOp>(inner->getParentOp())) {
      if (mlir::CommonUtils::isReduceAxis(funcOp, inner->getParentOp())) {
        mlir::affine::interchangeLoops(outer, inner);
      }
    }
  });

  // Replace dim values with large primes before fusion
  replaceDimWithPrimes(funcOp);
}

void AKGLoopFusion::runOnOperation() {
  auto funcOp = getOperation();

  runPreProcess();

  for (mlir::Region &region : funcOp->getRegions()) {
    for (mlir::Block &block : region.getBlocks()) {
      runOnBlock(&block);
    }
  }

  // Restore original dim values after fusion
  restoreDimFromPrimes(funcOp);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGLoopFusionPass() {
  return std::make_unique<AKGLoopFusion>();
}
