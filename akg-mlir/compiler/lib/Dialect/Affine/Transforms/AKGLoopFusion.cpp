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

namespace mlir {
namespace {

struct AKGLoopFusion : public impl::AKGLoopFusionBase<AKGLoopFusion> {
  AKGLoopFusion() {}

  void runOnOperation() override;

 private:
  void runOnBlock(Block *block, OperatorTemplate &curOpTemplate);
  void runPreProcess();
  void runOnLoopFusion();
  void runPostProcess();

  void replaceDimWithPrimes(func::FuncOp funcOp);
  void restoreDimFromPrimes();
  void repairReductionLoopAttrs(func::FuncOp funcOp);
  void moveAllocBeforeAffineFor(func::FuncOp funcOp);

  std::optional<SmallVector<std::string>> getSymShapeAttrFromValue(Value source);
  std::optional<int64_t> getConstantDimIndex(Value dimIndex);
  SmallVector<int64_t> getDynamicDimIndicesOfValue(Value v);
  void collectDimOperationsFromLoops(func::FuncOp funcOp, llvm::StringMap<SmallVector<Value>> &axisToDimValues);

  // Map: prime constant -> representative dim value (first dim in the group)
  llvm::DenseMap<Value, Value> primeToDimMap;
  // Whether to print fusion information
  bool printFusionInfo{false};
  // Create new symbol
  static int64_t newSymbolCount;
  static constexpr auto NEW_SYMBOL = "si";
  // Large primes for replacing dim values
  static constexpr int64_t kPrimes[] = {1000003, 1000033, 1000037, 1000039, 1000081,
                                        1000099, 1000117, 1000121, 1000133, 1000139};
  static constexpr size_t kNumPrimes = sizeof(kPrimes) / sizeof(kPrimes[0]);
};

int64_t AKGLoopFusion::newSymbolCount = 0;

// Trace back through operations to find the SymShapeAttr for a Value.
// Handles block arguments, bufferization.to_memref, etc.
std::optional<SmallVector<std::string>> AKGLoopFusion::getSymShapeAttrFromValue(Value source) {
  // Get SymShapeAttr from a RankedTensorType.
  auto getSymShape = [](Type type) -> std::optional<SmallVector<std::string>> {
    // RankedTensorType stores SymShapeAttr in its encoding
    // MemRefType stores SymShapeAttr in its memorySpace
    if (isa<RankedTensorType, MemRefType>(type)) {
      return SymbolicShapeAnalysis::getInstance().getSymbolicShape(type);
    }
    return std::nullopt;
  };

  // Case 1: Direct type has SymShapeAttr
  if (auto symShape = getSymShape(source.getType())) {
    return symShape;
  }

  // Case 2: Block argument (e.g., function argument)
  if (auto blockArg = dyn_cast<BlockArgument>(source)) {
    if (auto funcOp = dyn_cast<func::FuncOp>(blockArg.getOwner()->getParentOp())) {
      Type argType = funcOp.getFunctionType().getInput(blockArg.getArgNumber());
      return getSymShape(argType);
    }
    return std::nullopt;
  }

  // Case 3: MemRef created from tensor via bufferization.to_memref
  if (auto *defOp = source.getDefiningOp()) {
    if (auto toMemref = dyn_cast<bufferization::ToMemrefOp>(defOp)) {
      Value tensor = toMemref.getTensor();
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
std::optional<int64_t> AKGLoopFusion::getConstantDimIndex(Value dimIndex) {
  if (auto constOp = dimIndex.getDefiningOp<arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  if (auto constOp = dimIndex.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

SmallVector<int64_t> AKGLoopFusion::getDynamicDimIndicesOfValue(Value v) {
  SmallVector<int64_t> dynDims;
  auto st = dyn_cast<ShapedType>(v.getType());
  if (!st || !st.hasRank()) return dynDims;

  auto shape = st.getShape();
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (shape[i] == ShapedType::kDynamic) dynDims.push_back(i);
  }
  return dynDims;
}

// Collect dim operations from affine.for loop bounds and group them by axis.
void AKGLoopFusion::collectDimOperationsFromLoops(func::FuncOp funcOp,
                                                  llvm::StringMap<SmallVector<Value>> &axisToDimValues) {
  funcOp.walk([&](affine::AffineForOp forOp) {
    // Process a single operand from loop bounds
    auto processBoundOperand = [&](Value operand) {
      auto *defOp = operand.getDefiningOp();

      if (defOp && isa<memref::DimOp>(defOp)) {
        // Extract source and dimension index from dim operation
        Value source = defOp->getOperand(0);
        Value dimIndexValue = defOp->getOperand(1);

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
      } else {
        Value useOp;
        int64_t userIndex;
        for (auto &use : defOp->getResult(0).getUses()) {
          Operation *userOp = use.getOwner();
          if (auto allocOp = dyn_cast<memref::AllocOp>(userOp)) {
            useOp = allocOp.getResult();
            userIndex = static_cast<int64_t>(use.getOperandNumber());
            break;
          }
        }

        if (!useOp) {
          std::string axisKey = std::string(NEW_SYMBOL) + std::to_string(newSymbolCount);
          ++newSymbolCount;
          axisToDimValues[axisKey].push_back(operand);
          return;
        }

        auto dynDims = getDynamicDimIndicesOfValue(useOp);
        if (dynDims.empty()) {
          llvm::errs() << "This op has no dynamic shape\n";
          return;
        }
        int64_t dynamicIndex = dynDims[userIndex];
        auto symShape = getSymShapeAttrFromValue(useOp);
        if (!symShape.has_value() || dynamicIndex >= static_cast<int64_t>(symShape->size())) {
          return;
        }

        std::string axisKey = (*symShape)[dynamicIndex];
        axisToDimValues[axisKey].push_back(operand);
      }
    };

    // Process both upper and lower bound operands
    for (Value operand : forOp.getUpperBoundOperands()) {
      processBoundOperand(operand);
    }
    for (Value operand : forOp.getLowerBoundOperands()) {
      processBoundOperand(operand);
    }
  });
}

// Replace dynamic dim values with large prime constants before fusion.
// This allows the fusion analysis to treat loops with the same symbolic dimension as having identical bounds.
void AKGLoopFusion::replaceDimWithPrimes(func::FuncOp funcOp) {
  // Step 1: Collect dim operations and group by axis
  llvm::StringMap<SmallVector<Value>> axisToDimValues;
  collectDimOperationsFromLoops(funcOp, axisToDimValues);

  if (axisToDimValues.empty()) {
    return;
  }

  // Step 2: Replace each axis group with a unique prime constant
  OpBuilder builder(funcOp.getContext());
  size_t primeIndex = 0;

  for (auto &entry : axisToDimValues) {
    if (primeIndex >= kNumPrimes) {
      llvm::errs() << "Warning: Not enough primes for all dim axes\n";
      break;
    }

    auto &dimValues = entry.getValue();
    // Select first dim as representative (will be used when restoring)
    Value earliestDim;
    for (Value dimVal : dimValues) {
      if (!earliestDim) {
        earliestDim = dimVal;
        continue;
      }

      // if dimVal is BlockArgument, getDefiningOp() is nullptr
      auto *op1 = earliestDim.getDefiningOp();
      if (!op1) break;

      auto *op2 = dimVal.getDefiningOp();
      if (!op2) {
        earliestDim = dimVal;
        break;
      }

      if (op1->getBlock() == op2->getBlock()) {
        if (op2->isBeforeInBlock(op1)) {
          earliestDim = dimVal;
        }
      } else {
        return;
      }
    }

    Value representativeDim = earliestDim ? earliestDim : dimValues.front();
    builder.setInsertionPointAfterValue(representativeDim);
    auto primeConst = builder.create<arith::ConstantIndexOp>(representativeDim.getLoc(), kPrimes[primeIndex]);

    // Directly map prime constant to representative dim
    primeToDimMap[primeConst] = representativeDim;

    // Replace all dim values in this group with the same prime
    for (Value dimValue : dimValues) {
      dimValue.replaceAllUsesWith(primeConst);
    }

    primeIndex++;
  }
}

// Restore the original dim values after fusion is complete.
// All dim values in the same axis group are replaced with a single representative dim value.
void AKGLoopFusion::restoreDimFromPrimes() {
  // Replace each prime with its corresponding representative dim
  for (auto &[primeConst, representativeDim] : primeToDimMap) {
    primeConst.replaceAllUsesWith(representativeDim);

    // Clean up unused prime constant
    if (auto constOp = primeConst.getDefiningOp<arith::ConstantIndexOp>()) {
      if (constOp.use_empty()) {
        constOp.erase();
      }
    }
  }

  // Clear state for next run
  primeToDimMap.clear();
}

void AKGLoopFusion::runOnBlock(Block *block, OperatorTemplate &curOpTemplate) {
  // build dependence graph
  auto dependenceGraph = akg::MemRefDependenceGraphForFusion(block);
  if (!dependenceGraph.init()) {
    return;
  }

  curOpTemplate = dependenceGraph.funcOperatorType;

  if (printFusionInfo) {
    dependenceGraph.dump();
  }

  func::FuncOp funcOp = getOperation();
  akg::FusionAnalyzer analyzer(dependenceGraph, funcOp);
  // Plan the fusion strategy based on analysis
  analyzer.plan();
  if (analyzer.fusionPlans.empty()) {
    return;
  }

  if (printFusionInfo) {
    analyzer.dump();
  }

  akg::FusionCodeGenHelper codegenerator = akg::FusionCodeGenHelper(dependenceGraph);

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
    auto srcFor = dyn_cast<affine::AffineForOp>(dependenceGraph.getNode(actualSrcId)->op);
    auto dstFor = dyn_cast<affine::AffineForOp>(dependenceGraph.getNode(actualDstId)->op);

    if (srcFor && dstFor) {
      if (plan.fusionType == "V") {
        // Vertical fusion: calculate loop depth for the destination loop
        codegenerator.doVFuse(actualSrcId, actualDstId, srcFor, dstFor, plan);
      } else if (plan.fusionType == "H") {
        // Horizontal fusion: fuse loops at the same nesting level
        codegenerator.doHFuse(actualSrcId, actualDstId, srcFor, dstFor, plan);
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

void AKGLoopFusion::moveAllocBeforeAffineFor(func::FuncOp funcOp) {
  Block &block = funcOp.getBody().front();

  Operation *firstAffineFor = nullptr;
  for (Operation &op : block) {
    if (isa<affine::AffineForOp>(&op)) {
      firstAffineFor = &op;
      break;
    }
  }

  if (!firstAffineFor) {
    llvm::errs() << "[AKG] moveAllocBeforeAffineFor: no top-level affine.for found in func " << funcOp.getName()
                 << "\n";
    return;
  }

  SmallVector<Operation *, 8> toMove;

  for (auto it = std::next(firstAffineFor->getIterator()), e = block.end(); it != e; ++it) {
    Operation *op = &*it;
    if (isa<memref::AllocOp, memref::SubViewOp, memref::ReshapeOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
            memref::ReinterpretCastOp, memref::MemorySpaceCastOp>(op)) {
      toMove.push_back(op);
    }
  }

  if (toMove.empty()) {
    llvm::errs() << "[AKG] no allocs to move after first top-level affine.for\n";
    return;
  }

  for (Operation *op : toMove) {
    op->moveBefore(firstAffineFor);
  }
}

/// Preprocessing step that performs loop interchange optimization for reduction operations.
/// Identifies nested loops with reduction axes and interchanges them to improve cache performance.
void AKGLoopFusion::runPreProcess() {
  func::FuncOp funcOp = getOperation();

  moveAllocBeforeAffineFor(funcOp);

  // Replace dim values with large primes before fusion
  replaceDimWithPrimes(funcOp);
}

void AKGLoopFusion::runOnOperation() {
  runPreProcess();
  runOnLoopFusion();
  runPostProcess();
}

void AKGLoopFusion::runPostProcess() {
  auto funcOp = getOperation();
  repairReductionLoopAttrs(funcOp);
  restoreDimFromPrimes();
}

// After fusion, a reduction loop body may have been merged into a non-reduction
// loop, causing the {reduction} attribute to be lost.
void AKGLoopFusion::repairReductionLoopAttrs(func::FuncOp funcOp) {
  if (CommonUtils::getOperatorType(funcOp) != OperatorTemplate::Reduction) {
    return;
  }

  funcOp.walk([&](Operation *op) {
    if (!op->hasAttr(kReductionTypeStr)) return;
    auto axesAttr = op->getAttrOfType<ArrayAttr>(kReductionAxesStr);
    if (!axesAttr) return;

    SmallVector<affine::AffineForOp, 4> enclosingLoops;
    affine::getAffineForIVs(*op, &enclosingLoops);

    for (auto axis : axesAttr) {
      auto idx = cast<IntegerAttr>(axis).getInt();
      if (idx < 0 || idx >= static_cast<int64_t>(enclosingLoops.size())) continue;
      auto forOp = enclosingLoops[idx];
      if (!forOp->hasAttr(kReductionLoopAttr)) {
        OpBuilder builder(forOp.getContext());
        forOp->setAttr(kReductionLoopAttr, builder.getUnitAttr());
      }
    }
  });
}

void AKGLoopFusion::runOnLoopFusion() {
  auto funcOp = getOperation();

  OperatorTemplate curOpTemplate = OperatorTemplate::Default;
  for (Region &region : funcOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      OperatorTemplate opTemplate;
      runOnBlock(&block, opTemplate);
      if (opTemplate > curOpTemplate) {
        curOpTemplate = opTemplate;
      }
    }
  }

  // modify func operatorType
  auto iter = operatorTemplateMap.find((int)curOpTemplate);
  if (iter == operatorTemplateMap.end()) {
    return;
  }
  OpBuilder builder(funcOp.getContext());
  Attribute opType = builder.getStringAttr(iter->second);
  funcOp->setAttr(kOperatorTypeStr, opType);
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createAKGLoopFusionPass() { return std::make_unique<AKGLoopFusion>(); }

}  // namespace mlir
