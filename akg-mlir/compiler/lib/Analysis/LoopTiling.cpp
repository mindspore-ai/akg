/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "akg/Analysis/LoopTiling.h"

#include <algorithm>
#include <climits>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <unordered_set>

#include "akg/Analysis/BufferAnalysis.h"
#include "akg/Analysis/AutoTiling.h"
#include "akg/Analysis/TilingSolver.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

using hacc::HACCFuncType;
using hacc::HACCFuncTypeAttr;
using hacc::KernelArgType;
using hacc::KernelArgTypeAttr;
using llvm::DenseMap;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::Twine;
using mlir::autotiling::buildModelGraph;
using mlir::autotiling::buildNpuModelGraph;
using mlir::autotiling::getHeuristicTilingSolver;
using mlir::autotiling::getTileSizeWithSolver;
using mlir::autotiling::parseIr;
using mlir::autotiling::TilingSolverPtr;
using mlir::autotiling::TilingTaskDesc;

namespace mlir {
namespace autotiling {

static constexpr const char *kTreeNodeIdAttr = "node_id";
static constexpr const char *kTreeLeafAttr = "leaf";
static constexpr const char *kInnerLoopAttr = "inner";
static constexpr const char *kDeleteLoopAttr = "delete";
static constexpr const char *kNotInnerDimensionBroadcastLoopAttr = "not_inner_dimension_broadcast";
static constexpr const char *kParallelAxisAttr = "parallel__axis";
static constexpr int64_t kDefaultNpuCoreNum = 48;

// Main tiling functions
static LogicalResult createTilingFuncDefault(func::FuncOp originalKernel, OpBuilder &builder, func::FuncOp &tilingFunc,
                                             bool isStaticShape);
static LogicalResult calculateTileSizesForBands(func::FuncOp funcOp, bool useAutoTiling,
                                                std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                                std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                size_t &levelToTile);
static LogicalResult applyTilingToLoop(mlir::scf::ForOp loop, ArrayRef<Value> tileSizeValues,
                                       ArrayRef<unsigned> tileSizesInt, OpBuilder &builder,
                                       std::map<int64_t, Value> &constantCache,
                                       mlir::scf::ForOp parallelMapLoop = mlir::scf::ForOp(),
                                       int64_t parallelUseCore = 0);

struct LeafBranchBandPlan {
  SmallVector<mlir::scf::ForOp, 6> representativeBand;
  SmallVector<mlir::scf::ForOp, 6> peerLeafLoops;
  unsigned representativeLeafDim = 0;
  bool hasLeafBranching = false;
};
static LogicalResult buildLeafBranchBandPlanForRoot(mlir::scf::ForOp rootLoop, LeafBranchBandPlan &plan);
static LogicalResult buildLeafBranchBandPlans(func::FuncOp funcOp, std::vector<LeafBranchBandPlan> &plans,
                                              bool &hasUnsupportedTreeShape);
static bool isInnermostScfLoop(mlir::scf::ForOp forOp);

static LogicalResult wrapFunctionBodyWithFor(func::FuncOp func, OpBuilder &builder);
// Helper struct for loop bounds
struct LoopBounds {
  Value lb;
  Value ub;
  Value step;
  ValueRange inits;
};

// Helper struct for dynamic axis mapping
struct DynamicAxisMapping {
  unsigned inputMemrefIndex;
  unsigned dimIndex;
  Value upperBound;
};

// Bundled state for one band's tile rewrite traversal.
// Lives only on the stack and references caller-owned containers.
struct TileRewriteContext {
  const llvm::DenseSet<mlir::Operation *> &escapeReduceLoops;
  mlir::scf::ForOp parallelMapLoop;
  int parallelDim = -1;
  int64_t parallelUseCore = 0;
  bool dropMappedOutermostFirstLevelIterArgs = false;
};

// Loop tiling core helpers
static LoopBounds createFirstLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value tilesize,
                                                 unsigned tilesizeInt, OpBuilder &builder,
                                                 std::map<int64_t, Value> &constantCache,
                                                 bool dropFirstLevelReductionIterArgs);
static LoopBounds createMiddleLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value curTilesize,
                                                  Value prevTilesize, mlir::scf::ForOp prevLoop, OpBuilder &builder,
                                                  std::map<int64_t, Value> &constantCache, bool escapeReduceIterArgs);
static LoopBounds createPointLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop,
                                        OpBuilder &builder, std::map<int64_t, Value> &constantCache);
static mlir::scf::ForOp replaceLoopWithNewBounds(mlir::scf::ForOp oldLoop, const LoopBounds &bounds, mlir::Location loc,
                                                 OpBuilder &builder);
static void processSingleTileLoop(int i, int j, int bandSize, int tileNum, MutableArrayRef<mlir::scf::ForOp> newLoops,
                                  ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                  ArrayRef<unsigned> tileSizesInt,
                                  SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                  mlir::Location loc, std::map<int64_t, Value> &constantCache,
                                  TileRewriteContext &ctx);
static void constructTiledIndexStatic(MutableArrayRef<mlir::scf::ForOp> newLoops, ArrayRef<mlir::scf::ForOp> band,
                                      ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                      OpBuilder &builder, std::map<int64_t, Value> &constantCache,
                                      TileRewriteContext &ctx);

// Dynamic axis mapping and tile size helpers
static std::vector<DynamicAxisMapping> buildDynamicAxisMappingForBand(ArrayRef<mlir::scf::ForOp> band,
                                                                      ArrayRef<unsigned> bandTileSizes,
                                                                      func::FuncOp originalKernel);
static Value emitDynamicTilePerDim(Location loc, OpBuilder &builder, Value dim, int constraintMax);
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                 Value &result);
static LogicalResult prepareTileSizesForStaticShape(func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                    const std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                    const std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                                    const std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                    std::vector<SmallVector<Value, 6>> &allTileSizeValues);
static LogicalResult prepareTileSizesFromMemref(func::FuncOp originalKernel,
                                                ArrayRef<SmallVector<mlir::scf::ForOp, 6>> bands, mlir::Location loc,
                                                OpBuilder &builder,
                                                std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                                std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt);
static LogicalResult prepareLeafBranchPlansForApply(func::FuncOp originalKernel, OpBuilder &builder,
                                                    std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                                    bool &shouldReturnEarly);
static LogicalResult prepareTileMetadataForApply(func::FuncOp originalKernel, OpBuilder &builder, mlir::Location loc,
                                                 bool isStaticShape,
                                                 std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                 std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt,
                                                 std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                 std::vector<SmallVector<Value, 6>> &allTileSizeValues);
static LogicalResult applySingleLinearBandDecoupled(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                    ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                                    ArrayRef<unsigned> tileSizesInt, int64_t &nextNodeId);
static LogicalResult applySingleLeafBranchBandDecoupled(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                        const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band,
                                                        ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                                        int64_t &nextNodeId);
static void buildLeafBranchTileSlices(
  const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
  ArrayRef<unsigned> tileSizesInt, unsigned bandSize, unsigned tileLevels, SmallVector<mlir::scf::ForOp, 6> &prefixBand,
  SmallVector<Value, 6> &prefixTileValues, SmallVector<unsigned, 6> &prefixTileSizesInt,
  SmallVector<Value, 6> &leafTileValues, SmallVector<unsigned, 6> &leafTileSizesInt);
static LogicalResult collectAndTagLeafBranchLoops(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                  const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band,
                                                  int64_t &nextNodeId, SmallVector<int64_t, 6> &branchLeafNodeIds);
static LogicalResult applyLeafBandsByNodeId(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                            ArrayRef<int64_t> branchLeafNodeIds, ArrayRef<Value> leafTileValues,
                                            ArrayRef<unsigned> leafTileSizesInt);
static LogicalResult applyAllBandsWithPlans(func::FuncOp originalKernel, OpBuilder &builder,
                                            const std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                            const std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                            const std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                            const std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt);
static void runApplyPostProcessing(func::FuncOp originalKernel, OpBuilder &builder);

// Tiling function creation helpers
static void buildTilingFunctionSignature(FunctionType origTy, MLIRContext *ctx, OpBuilder &builder,
                                         SmallVector<Type> &argTypes, SmallVector<Type> &resTypes,
                                         int64_t tilingStructMemrefSize);
static func::FuncOp createAndInitTilingFunc(func::FuncOp originalKernel, ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resTypes, OpBuilder &builder);
static LogicalResult storeTileSizesToMemref(func::FuncOp tilingFunc, func::FuncOp originalKernel,
                                            ArrayRef<SmallVector<mlir::scf::ForOp, 6>> bands,
                                            ArrayRef<SmallVector<unsigned, 6>> allBandTileSizes,
                                            ArrayRef<SmallVector<int, 6>> allBandConstraintMaxs,
                                            ArrayRef<std::vector<DynamicAxisMapping>> allBandDynamicMappings,
                                            OpBuilder &builder);
static void getOperandsTree(mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &ops,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &visited);
static mlir::Value cloneUpperBoundDefinition(mlir::Value upperBound, func::FuncOp originalKernel,
                                             func::FuncOp tilingFunc, OpBuilder &builder);

// Attribute marking helpers
static Value stripIndexLikeCasts(Value value);
static void markBandTransposeLoops(func::FuncOp funcOp, const LeafBranchBandPlan &plan);
static void preprocessLoopAttrsForTileCalculation(func::FuncOp funcOp, const LeafBranchBandPlan &plan);
static ReduceDirection getReduceType(mlir::scf::ForOp loop);
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder);
static void setBlockDimAttribute(func::FuncOp funcOp, OpBuilder &builder);

// Tail block creation
static LogicalResult createTailBlockForBodyStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                                  std::map<int64_t, Value> &constantCache);
static LogicalResult createTailBlockStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                           std::map<int64_t, Value> &constantCache);
static LogicalResult createTailBlockStaticImpl(mlir::scf::ForOp forOp, int64_t differenceUbAndLb, OpBuilder &builder,
                                               std::map<int64_t, Value> &constantCache);
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value dynamicBound, OpBuilder &builder,
                                                std::map<int64_t, Value> &constantCache);

// Loop structure helpers
static std::tuple<std::optional<int64_t>, mlir::Value, mlir::Value> getDifferenceUbAndLb(mlir::scf::ForOp forOp);
static std::optional<int64_t> getConstantIndexValue(mlir::Value value);
static bool isAncestorOp(mlir::Operation *maybeAncestor, mlir::Operation *op);
static mlir::LogicalResult verifyBandIsNestedChain(llvm::ArrayRef<mlir::scf::ForOp> band);

// Operation cloning helpers
static void collectChainComputeOps(llvm::ArrayRef<mlir::scf::ForOp> band,
                                   llvm::SmallVectorImpl<mlir::Operation *> &ops);
static mlir::LogicalResult cloneComputeIntoInnermostPointLoop(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                              llvm::ArrayRef<mlir::scf::ForOp> tiledLoops,
                                                              unsigned tileSizesNum, mlir::scf::ForOp rootScfForOp,
                                                              mlir::OpBuilder &builder, mlir::IRMapping &mapping);
static void splitOpsAroundChildLoop(mlir::scf::ForOp parent, mlir::Operation *childLoopOp,
                                    llvm::SmallVectorImpl<mlir::Operation *> &preOps,
                                    llvm::SmallVectorImpl<mlir::Operation *> &postOps);
static bool isReductionLoopWithIterArgs(mlir::scf::ForOp loop);
static bool isLoopResultEscapingBand(mlir::scf::ForOp loop, mlir::scf::ForOp bandRoot);
static unsigned getMiddleLevelLoopIndex(unsigned tileSizesNum, unsigned bandSize, unsigned dim);
static void initIVMapping(llvm::ArrayRef<mlir::scf::ForOp> band, llvm::ArrayRef<mlir::scf::ForOp> tiledLoops,
                          unsigned tileSizesNum, mlir::IRMapping &mapping);
static void updatePointLoopYieldsFromOriginalLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                   llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                   unsigned tileSizesNum, mlir::IRMapping &mapping,
                                                   mlir::OpBuilder &builder);
static bool isTransposePointLoop(mlir::scf::ForOp loop);
static bool isMultiVecPointLoop(mlir::scf::ForOp loop);
static void clearTransposeChain(mlir::scf::ForOp loop);
static bool sinkTransposePointLoopOnce(mlir::scf::ForOp pointLoop);
static void sinkTransposePointLoops(func::FuncOp funcOp);
static void sinkMultiVecPointLoops(func::FuncOp funcOp);
static void forwardIterArgsThroughWrapperLoops(llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                               mlir::OpBuilder &builder);
static LogicalResult collectReduceResultUserOps(mlir::scf::ForOp rootLoop, mlir::scf::ForOp userAnchorLoop,
                                                llvm::SmallSet<Operation *, 16> &opSet);
static void collectOpsInOrderAfter(Operation *startOp, const llvm::SmallSet<Operation *, 16> &opSet,
                                   SmallVectorImpl<Operation *> &opsInOrder);
static void moveOpsAfterLoop(mlir::scf::ForOp destLoop, ArrayRef<Operation *> ops, OpBuilder &builder);
static void replaceUsesInOps(ValueRange oldResults, ValueRange newResults, ArrayRef<Operation *> ops);
static LogicalResult sinkReduceLoopResultsToMiddleLevel(mlir::scf::ForOp rootLoop, mlir::scf::ForOp userAnchorLoop,
                                                        llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                        unsigned tileSizesNum, unsigned bandSize,
                                                        mlir::OpBuilder &builder);
static void cloneOpsToPointLoop(mlir::scf::ForOp pointLoop, llvm::ArrayRef<mlir::Operation *> ops, bool insertAtStart,
                                mlir::OpBuilder &builder, mlir::IRMapping &mapping);
static mlir::LogicalResult cloneNonPerfectChainIntoPointLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                              llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                              unsigned tileSizesNum, mlir::OpBuilder &builder,
                                                              mlir::IRMapping &mapping);
// Helper: collect all defining ops in the operand tree (post-order)
static void getOperandsTree(mlir::Operation *op, llvm::SmallVectorImpl<mlir::Operation *> &ops,
                            llvm::SmallPtrSetImpl<mlir::Operation *> &visited) {
  if (!op || !visited.insert(op).second) {
    return;
  }
  for (Value operand : op->getOperands()) {
    if (auto *def = operand.getDefiningOp()) {
      getOperandsTree(def, ops, visited);
    }
  }
  ops.push_back(op);
}

// Helper: clone the upper bound definition chain into tiling function
static mlir::Value cloneUpperBoundDefinition(mlir::Value upperBound, func::FuncOp originalKernel,
                                             func::FuncOp tilingFunc, OpBuilder &builder) {
  if (!upperBound) {
    return Value();
  }

  mlir::IRMapping mapping;

  // Map original kernel arguments to tiling function arguments by index.
  auto &origEntry = originalKernel.getBody().front();
  for (auto [origArg, tilingArg] : llvm::zip(origEntry.getArguments(), tilingFunc.getArguments())) {
    mapping.map(origArg, tilingArg);
  }

  // If upperBound is already an entry block argument, return mapped value.
  if (auto blockArg = mlir::dyn_cast<BlockArgument>(upperBound)) {
    if (blockArg.getOwner() == &origEntry) {
      return mapping.lookupOrDefault(upperBound);
    }
    llvm::dbgs() << "cloneUpperBoundDefinition: non-entry block argument, fallback\n";
    return Value();
  }

  auto *def = upperBound.getDefiningOp();
  if (!def) {
    llvm::dbgs() << "cloneUpperBoundDefinition: no defining op, fallback\n";
    return Value();
  }

  llvm::SmallVector<mlir::Operation *, 16> ops;
  llvm::SmallPtrSet<mlir::Operation *, 32> visited;
  getOperandsTree(def, ops, visited);

  auto isSupportedOp = [](mlir::Operation *op) -> bool {
    if (!op || op->getNumRegions() != 0) {
      return false;
    }
    return isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp, arith::IndexCastOp, arith::CmpIOp,
               arith::SelectOp, arith::AddIOp, arith::SubIOp, arith::MulIOp, memref::DimOp, memref::ExpandShapeOp,
               affine::AffineApplyOp, affine::AffineMinOp>(op);
  };

  for (auto *op : ops) {
    if (!isSupportedOp(op)) {
      llvm::dbgs() << "cloneUpperBoundDefinition: unsupported op " << op->getName() << ", fallback\n";
      return Value();
    }

    // Ensure operands are mappable.
    for (Value operand : op->getOperands()) {
      if (mapping.contains(operand)) {
        continue;
      }
      if (auto blockArg = mlir::dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() == &origEntry) {
          mapping.map(blockArg, tilingFunc.getArgument(blockArg.getArgNumber()));
          continue;
        }
        llvm::dbgs() << "cloneUpperBoundDefinition: non-entry block argument in chain, fallback\n";
        return Value();
      }
      llvm::dbgs() << "cloneUpperBoundDefinition: operand not mapped for op " << op->getName() << ", fallback\n";
      return Value();
    }

    Operation *cloned = builder.clone(*op, mapping);
    mapping.map(op, cloned);
  }

  if (auto mapped = mapping.lookupOrNull(upperBound)) {
    return mapped;
  }
  llvm::dbgs() << "cloneUpperBoundDefinition: upper bound not mapped, fallback\n";
  return Value();
}

static bool isNoSplitLoop(mlir::scf::ForOp loop) {
  return loop && (loop->hasAttr(kReductionLoopAttr) || loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr));
}

static bool isReductionLoop(mlir::scf::ForOp loop) { return loop && loop->hasAttr(kReductionLoopAttr); }

static bool valueDependsOnTarget(Value value, Value target, llvm::DenseSet<Value> &visitedValues,
                                 llvm::SmallPtrSetImpl<Operation *> &visitedOps) {
  if (value == target) {
    return true;
  }

  if (!visitedValues.insert(value).second) {
    return false;
  }

  auto blockArg = dyn_cast<BlockArgument>(value);
  if (blockArg) {
    return false;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp || !visitedOps.insert(defOp).second) {
    return false;
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    auto opResult = dyn_cast<OpResult>(value);
    if (opResult) {
      unsigned resultIdx = opResult.getResultNumber();
      auto dependsOnYieldOperand = [&](Region &region) {
        auto yieldOp = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yieldOp || resultIdx >= yieldOp.getNumOperands()) {
          return false;
        }
        return valueDependsOnTarget(yieldOp.getOperand(resultIdx), target, visitedValues, visitedOps);
      };
      if (dependsOnYieldOperand(ifOp.getThenRegion()) || dependsOnYieldOperand(ifOp.getElseRegion())) {
        return true;
      }
    }
  }

  return std::any_of(defOp->operand_begin(), defOp->operand_end(),
                     [&](Value operand) { return valueDependsOnTarget(operand, target, visitedValues, visitedOps); });
}

static bool valueDependsOnLoopIV(Value value, mlir::scf::ForOp loop) {
  if (!value || !loop) {
    return false;
  }

  llvm::DenseSet<Value> visitedValues;
  llvm::SmallPtrSet<Operation *, 32> visitedOps;
  return valueDependsOnTarget(value, loop.getInductionVar(), visitedValues, visitedOps);
}

static bool isInnerDimensionBroadcastLoop(mlir::scf::ForOp loop) {
  return loop
    .walk([&](memref::StoreOp store) -> WalkResult {
      if (store.getIndices().empty() || !valueDependsOnLoopIV(store.getIndices().back(), loop)) {
        return WalkResult::advance();
      }

      llvm::SmallVector<Value, 8> worklist{store.getValueToStore()};
      llvm::DenseSet<Value> visited;
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value).second) {
          continue;
        }

        if (auto load = value.getDefiningOp<memref::LoadOp>()) {
          if (llvm::none_of(load.getIndices(), [&](Value idx) { return valueDependsOnLoopIV(idx, loop); })) {
            return WalkResult::interrupt();
          }
          continue;
        }

        if (Operation *defOp = value.getDefiningOp()) {
          auto operands = defOp->getOperands();
          std::copy(operands.begin(), operands.end(), std::back_inserter(worklist));
        }
      }
      return WalkResult::advance();
    })
    .wasInterrupted();
}

static void setNotInnerDimensionBroadcastLoopAttr(func::FuncOp funcOp) {
  OpBuilder builder(funcOp.getContext());
  bool hasOuterInnerDimensionBroadcastLoop = false;
  funcOp.walk([&](mlir::scf::ForOp loop) {
    if (!loop->hasAttr(kBroadcastLoopAttr)) {
      if (loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr)) {
        loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
      }
      return;
    }
    if (isInnerDimensionBroadcastLoop(loop)) {
      hasOuterInnerDimensionBroadcastLoop = true;
    } else if (hasOuterInnerDimensionBroadcastLoop) {
      loop->removeAttr(kBroadcastLoopAttr);
    } else {
      loop->setAttr(kNotInnerDimensionBroadcastLoopAttr, builder.getUnitAttr());
    }
  });
  if (!hasOuterInnerDimensionBroadcastLoop) {
    funcOp.walk([&](mlir::scf::ForOp loop) {
      loop->removeAttr(kBroadcastLoopAttr);
      loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
    });
  }
}

static void clearNotInnerDimensionBroadcastLoopAttr(func::FuncOp funcOp) {
  funcOp.walk([&](mlir::scf::ForOp loop) {
    if (loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr)) {
      loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
    }
  });
}

static void clearAllBroadcastLoopAttrs(func::FuncOp funcOp) {
  funcOp.walk([](mlir::scf::ForOp loop) {
    loop->removeAttr(kBroadcastLoopAttr);
    loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
  });
}

static void preprocessLoopAttrsForTileCalculation(func::FuncOp funcOp, const LeafBranchBandPlan &plan) {
  if (plan.representativeBand.empty()) {
    return;
  }
  if (plan.hasLeafBranching || plan.representativeBand.size() <= 1) {
    clearAllBroadcastLoopAttrs(funcOp);
    markBandTransposeLoops(funcOp, plan);
    return;
  }

  mlir::scf::ForOp innerLoop = plan.representativeBand.back();
  if (isReductionLoop(innerLoop)) {
    clearAllBroadcastLoopAttrs(funcOp);
    markBandTransposeLoops(funcOp, plan);
    return;
  }

  markBandTransposeLoops(funcOp, plan);
  setNotInnerDimensionBroadcastLoopAttr(funcOp);
}

static void clearValuelessBroadcastLoopAttr(func::FuncOp funcOp) {
  funcOp.walk([](mlir::scf::ForOp loop) {
    if (!loop->getAttrOfType<UnitAttr>(kBroadcastLoopAttr)) {
      return;
    }
    loop->removeAttr(kBroadcastLoopAttr);
  });
}

// Helper function to get constant index value from Value
static std::optional<int64_t> getConstantIndexValue(mlir::Value value) {
  // BlockArgument (e.g., iter_args) has no defining op, getDefiningOp() returns nullptr
  if (!value || !value.getDefiningOp()) {
    return std::nullopt;
  }
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return constOp.value();
  }
  if (auto constOp = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
      return intAttr.getInt();
    }
  }
  return std::nullopt;
}

static int64_t getLoopExtent(mlir::scf::ForOp loop) {
  if (!loop) {
    return kVectorSize;
  }
  if (auto innerAttr = loop->getAttrOfType<IntegerAttr>(kInnerLoopAttr)) {
    return innerAttr.getInt();
  }

  std::optional<int64_t> lb = getConstantIndexValue(loop.getLowerBound());
  std::optional<int64_t> ub = getConstantIndexValue(loop.getUpperBound());
  if (lb && ub) {
    return *ub - *lb;
  }

  return kVectorSize;
}

static int64_t ceilDivPositive(int64_t lhs, int64_t rhs) {
  return (lhs <= 0 || rhs <= 0) ? 1 : (lhs + rhs - 1) / rhs;
}

static int64_t getNpuCoreNum(func::FuncOp funcOp) {
  if (auto archAttr = funcOp->getAttrOfType<StringAttr>("arch")) {
    uint32_t coreNum = akg::NpuInfo::getInstance(archAttr.getValue().str()).getCoreNumAiv();
    if (coreNum > 0) {
      return static_cast<int64_t>(coreNum);
    }
  }
  return kDefaultNpuCoreNum;
}

static bool getStaticTripCount(mlir::scf::ForOp loop, int64_t &tripCount) {
  auto [extent, dynamicUb, dynamicLb] = getDifferenceUbAndLb(loop);
  (void)dynamicUb;
  (void)dynamicLb;
  std::optional<int64_t> step = getConstantIndexValue(loop.getStep());
  if (!extent || !step || *extent <= 0 || *step <= 0) {
    return false;
  }
  tripCount = ceilDivPositive(*extent, *step);
  return true;
}

static bool isParallelCandidate(mlir::scf::ForOp loop, unsigned tileSize0, int64_t &parallelWork) {
  if (!loop || loop->hasAttr(kReductionLoopAttr) || loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr) ||
      tileSize0 == 0 || tileSize0 == static_cast<unsigned>(-1)) {
    return false;
  }
  int64_t tripCount = 0;
  if (!getStaticTripCount(loop, tripCount)) {
    return false;
  }
  parallelWork = ceilDivPositive(tripCount, static_cast<int64_t>(tileSize0));
  return parallelWork > 0;
}

static int markParallelAxis(func::FuncOp funcOp, ArrayRef<mlir::scf::ForOp> band, ArrayRef<unsigned> tileSizesInt,
                            unsigned tileLevels, OpBuilder &builder, int64_t &useCore, bool allowLastAxis = false) {
  useCore = -1;
  if (band.empty() || tileLevels == 0 || tileSizesInt.size() < band.size()) {
    return -1;
  }
  int64_t coreNum = getNpuCoreNum(funcOp);
  int bestDim = -1;
  int64_t bestWork = 0;
  int64_t bestDistance = std::numeric_limits<int64_t>::max();
  for (auto loop : band) {
    if (loop) loop->removeAttr(kParallelAxisAttr);
  }
  SmallVector<int64_t, 6> axisWorks(band.size(), 0);
  for (unsigned dim = 0; dim < band.size(); ++dim) {
    if (!allowLastAxis && band.size() > 1 && dim + 1 == band.size()) continue;
    int64_t parallelWork = 0;
    if (!isParallelCandidate(band[dim], tileSizesInt[dim], parallelWork)) {
      continue;
    }
    axisWorks[dim] = parallelWork;
  }
  for (unsigned dim = 0; dim < band.size(); ++dim) {
    if (axisWorks[dim] <= 0) {
      continue;
    }
    int64_t parallelWork = axisWorks[dim];
    int64_t distance = (parallelWork > coreNum) ? (parallelWork - coreNum) : (coreNum - parallelWork);
    if (distance < bestDistance || (distance == bestDistance && parallelWork > bestWork)) {
      bestDim = static_cast<int>(dim);
      bestWork = parallelWork;
      bestDistance = distance;
    }
  }
  if (bestDim < 0) {
    return -1;
  }
  useCore = std::min(coreNum, bestWork);
  band[bestDim]->setAttr(kParallelAxisAttr, builder.getUnitAttr());
  return bestDim;
}

// Helper function to calculate difference between upper and lower bounds
// Returns: (static_diff, dynamic_ub, dynamic_lb)
static std::tuple<std::optional<int64_t>, mlir::Value, mlir::Value> getDifferenceUbAndLb(mlir::scf::ForOp forOp) {
  auto lb = forOp.getLowerBound();
  auto ub = forOp.getUpperBound();

  // Try to get constant values
  auto lbConst = getConstantIndexValue(lb);
  auto ubConst = getConstantIndexValue(ub);

  if (lbConst.has_value() && ubConst.has_value()) {
    // Static case: return constant difference
    return std::make_tuple(ubConst.value() - lbConst.value(), mlir::Value(), mlir::Value());
  }

  // Dynamic case: return ub and lb, let caller create diff in the correct block
  return std::make_tuple(std::nullopt, ub, lb);
}

// Helper function to copy HACC IO attributes from original to destination function
static void copyHaccIOAttrsFrom(func::FuncOp orig, func::FuncOp dst) {
  if (std::optional<ArrayAttr> maybeArray = orig.getArgAttrs()) {
    ArrayAttr arr = *maybeArray;
    unsigned n = std::min<unsigned>(arr.size(), dst.getNumArguments());
    for (unsigned i = 0; i < n; ++i) {
      if (auto dict = dyn_cast_or_null<DictionaryAttr>(arr[i])) {
        SmallVector<NamedAttribute, 4> attrs;
        attrs.reserve(dict.size());
        std::copy(dict.begin(), dict.end(), std::back_inserter(attrs));
        for (const auto &na : attrs) {
          dst.setArgAttr(i, na.getName(), na.getValue());
        }
      }
    }
  }
}

// Helper function to set tiling key and data argument attributes
static void setTilingKeyAndDataArgAttrs(func::FuncOp func, unsigned keyIdx, unsigned tilingDataIdx, MLIRContext *ctx) {
  auto katName = StringAttr::get(ctx, KernelArgTypeAttr::name);

  auto setArgKernelKind = [&](unsigned idx, KernelArgType kind) {
    DictionaryAttr old = func.getArgAttrDict(idx);
    SmallVector<NamedAttribute> nas;
    if (old) {
      nas.reserve(old.size() + 1);
      std::copy(old.begin(), old.end(), std::back_inserter(nas));
    }
    auto kat = KernelArgTypeAttr::get(ctx, kind);

    bool replaced = false;
    for (auto &na : nas) {
      if (na.getName() == katName) {
        na = NamedAttribute(katName, kat);
        replaced = true;
        break;
      }
    }
    if (!replaced) nas.emplace_back(katName, kat);

    func.setArgAttrs(idx, DictionaryAttr::get(ctx, nas));
  };

  setArgKernelKind(keyIdx, KernelArgType::kTilingKey);
  setArgKernelKind(tilingDataIdx, KernelArgType::kTilingStruct);
}

// Helper function to trace the source of a dynamic upper bound
// Returns {argIndex, dimIndex} if the upper bound comes from memref.dim(arg, dim)
// Returns {-1, -1} if not found
static std::pair<int, int> traceDynamicUpperBound(Value upperBound, func::FuncOp func) {
  // Check if upperBound is directly from memref.dim
  if (auto dimOp = upperBound.getDefiningOp<memref::DimOp>()) {
    Value source = dimOp.getSource();
    Value dimIdx = dimOp.getIndex();

    // Check if source is a function argument
    if (auto blockArg = dyn_cast<BlockArgument>(source)) {
      if (blockArg.getOwner() == &func.getBody().front()) {
        unsigned argIndex = blockArg.getArgNumber();

        // Check if dimIdx is a constant
        if (auto constOp = dimIdx.getDefiningOp<arith::ConstantIndexOp>()) {
          int64_t dimValue = constOp.value();
          if (dimValue >= 0) {
            return {static_cast<int>(argIndex), static_cast<int>(dimValue)};
          }
        }
      }
    }
  }
  return {-1, -1};
}

static void collectDirectChildLoopsInOrder(mlir::scf::ForOp parent, SmallVectorImpl<mlir::scf::ForOp> &children) {
  children.clear();
  Block *body = parent.getBody();
  if (!body) {
    return;
  }
  for (Operation &op : body->without_terminator()) {
    if (auto childLoop = dyn_cast<mlir::scf::ForOp>(&op)) {
      children.push_back(childLoop);
    }
  }
}

static bool isLeafForLoop(mlir::scf::ForOp loop) {
  SmallVector<mlir::scf::ForOp, 4> children;
  collectDirectChildLoopsInOrder(loop, children);
  return children.empty();
}

static LogicalResult buildLeafBranchBandPlanForRoot(mlir::scf::ForOp rootLoop, LeafBranchBandPlan &plan) {
  if (!rootLoop) {
    return failure();
  }

  plan = LeafBranchBandPlan{};
  SmallVector<mlir::scf::ForOp, 6> linearPrefix;
  linearPrefix.push_back(rootLoop);
  mlir::scf::ForOp current = rootLoop;

  while (true) {
    SmallVector<mlir::scf::ForOp, 4> children;
    collectDirectChildLoopsInOrder(current, children);
    if (children.empty()) {
      plan.representativeBand = linearPrefix;
      plan.representativeLeafDim = static_cast<unsigned>(linearPrefix.size() - 1);
      plan.hasLeafBranching = false;
      return success();
    }

    if (children.size() == 1) {
      current = children.front();
      linearPrefix.push_back(current);
      continue;
    }

    if (!llvm::all_of(children, [](mlir::scf::ForOp loop) { return isLeafForLoop(loop); })) {
      return failure();
    }

    mlir::scf::ForOp representativeLeaf = children.front();
    plan.representativeBand = linearPrefix;
    plan.representativeLeafDim = static_cast<unsigned>(linearPrefix.size());
    plan.representativeBand.push_back(representativeLeaf);
    plan.hasLeafBranching = true;
    for (mlir::scf::ForOp loop : children) {
      if (loop != representativeLeaf) {
        plan.peerLeafLoops.push_back(loop);
      }
    }
    return success();
  }
}

static LogicalResult buildLeafBranchBandPlans(func::FuncOp funcOp, std::vector<LeafBranchBandPlan> &plans,
                                              bool &hasUnsupportedTreeShape) {
  plans.clear();
  hasUnsupportedTreeShape = false;

  Block *body = &funcOp.getBody().front();
  for (Operation &op : *body) {
    auto rootLoop = dyn_cast<mlir::scf::ForOp>(&op);
    if (!rootLoop) {
      continue;
    }

    LeafBranchBandPlan plan;
    if (failed(buildLeafBranchBandPlanForRoot(rootLoop, plan))) {
      hasUnsupportedTreeShape = true;
      plans.clear();
      return success();
    }
    plans.push_back(std::move(plan));
  }

  return success();
}

static void collectRepresentativeBands(ArrayRef<LeafBranchBandPlan> plans,
                                       std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands) {
  bands.clear();
  bands.reserve(plans.size());
  for (const LeafBranchBandPlan &plan : plans) {
    if (!plan.representativeBand.empty()) {
      bands.push_back(plan.representativeBand);
    }
  }
}

static LogicalResult calculateTileSizesForBands(func::FuncOp funcOp, bool useAutoTiling,
                                                std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                                std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                size_t &levelToTile) {
  allBandTileSizes.clear();
  allBandConstraintMaxs.clear();

  if (bandsToUse.empty()) {
    return success();
  }

  if (std::any_of(bandsToUse.begin(), bandsToUse.end(),
                  [&](const auto &band) { return failed(verifyBandIsNestedChain(band)); })) {
    funcOp.emitError("tile size solver requires nested-chain bands");
    return failure();
  }

  // Determine target (default to NPU)
  std::string target = mlir::kTargetNpu;  // "aicore"
  Attribute processAttr = funcOp->getAttr("process");
  if (processAttr) {
    if (auto stringAttr = dyn_cast<mlir::StringAttr>(processAttr)) {
      target = stringAttr.getValue().str();
    }
  }

  // Determine feature (default empty)
  std::string feature = "";
  Attribute computeCapabilityAttr = funcOp->getAttr("compute_capability");
  if (computeCapabilityAttr) {
    if (auto stringAttr = dyn_cast<mlir::StringAttr>(computeCapabilityAttr)) {
      feature = stringAttr.getValue().str();
    }
  }

  // Determine arch (default empty)
  std::string arch = "";
  Attribute archAttr = funcOp->getAttr("arch");
  if (archAttr) {
    if (auto stringAttr = dyn_cast<mlir::StringAttr>(archAttr)) {
      arch = stringAttr.getValue().str();
    }
  }

  // Check if dynamic shape (for autoTiling, not loop dynamic shape)
  bool isDynamicShape = akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0;

  if (!useAutoTiling) {
    // If not using auto-tiling, return empty (will be handled by caller)
    return success();
  }

  mlir::akg::BufferAnalysisOptions options;
  countMaxBuffer(funcOp, options);

  // Parse IR and build model graph
  auto initGraph = parseIr(funcOp, bandsToUse);
  initGraph->setHardware(target);
  initGraph->setFeature(feature);
  initGraph->setArch(arch);
  initGraph->setIsDynamicShape(isDynamicShape);
  initGraph->setTilingMode("auto");

  auto modelGraph = buildModelGraph(initGraph);
  auto solver = getHeuristicTilingSolver(modelGraph);
  levelToTile = modelGraph->levelToTile;

  // Calculate tile sizes for each band (with constraint max for dynamic shapes)
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    SmallVector<mlir::scf::ForOp, 6> curBand = bandsToUse[bandIdx];
    SmallVector<unsigned, 6> bandTileSizes;
    SmallVector<int, 6> bandConstraintMaxs;

    for (size_t level = 0; level < levelToTile; ++level) {
      getTileSizeWithSolver(solver, curBand, &bandTileSizes, &bandConstraintMaxs, TilingTaskDesc(bandIdx, level));
    }

    allBandTileSizes.push_back(bandTileSizes);
    allBandConstraintMaxs.push_back(bandConstraintMaxs);
  }

  // Print all band tile sizes with constraint max info
  llvm::errs() << "=== All Band Tile Sizes ===\n";
  for (size_t i = 0; i < allBandTileSizes.size(); ++i) {
    llvm::errs() << "Band " << i << " tile sizes: [";
    for (size_t j = 0; j < allBandTileSizes[i].size(); ++j) {
      if (j > 0) llvm::errs() << ", ";
      if (allBandTileSizes[i][j] == static_cast<unsigned>(-1)) {
        llvm::errs() << "dynamic(max=" << allBandConstraintMaxs[i][j] << ")";
      } else {
        llvm::errs() << allBandTileSizes[i][j];
      }
    }
    llvm::errs() << "]\n";
  }
  llvm::errs() << "==========================\n";

  return success();
}

// Helper function to get or create constant (for use in static functions)
static Value getOrCreateConstantStatic(Location loc, int64_t value, OpBuilder &builder,
                                       std::map<int64_t, Value> &constantCache) {
  auto it = constantCache.find(value);
  if (it != constantCache.end()) {
    if (auto constOp = it->second.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
      if (constOp.value() == value) {
        return it->second;
      }
    }
    constantCache.erase(it);
  }
  Value constVal = builder.create<mlir::arith::ConstantIndexOp>(loc, value);
  if (auto constOp = constVal.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    if (constOp.value() == value) {
      constantCache[value] = constVal;
    }
  }
  return constVal;
}

// Helper to recreate constant at current insertion point, or return original value
static Value recreateConstantOrSelf(Value v, OpBuilder &builder) {
  if (!v) return v;

  // If it's a constant, always recreate it at builder's current insertion point
  // This ensures constants are defined before they're used
  if (auto constOp = v.getDefiningOp<arith::ConstantIndexOp>()) {
    return builder.create<arith::ConstantIndexOp>(v.getLoc(), constOp.value());
  }
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    return builder.clone(*constOp)->getResult(0);
  }

  // For non-constant values, return as-is
  return v;
}

// Helper function to construct tiled loop structure
static void constructTiledLoopStatic(mlir::scf::ForOp rootScfForOp, unsigned width,
                                     MutableArrayRef<mlir::scf::ForOp> tiledLoops, OpBuilder &builder,
                                     std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = rootScfForOp.getLoc();
  mlir::Operation *topLoop = rootScfForOp.getOperation();

  // Get original init args - used for ALL wrapper loops during initial construction
  // The correct init args chaining will be established when bounds are replaced
  auto origInits = rootScfForOp.getInitArgs();
  bool hasIterArgs = !origInits.empty();

  // Create width number of nested loops (from innermost to outermost)
  for (unsigned i = 0; i < width; ++i) {
    builder.setInsertionPoint(topLoop);

    mlir::Value c0 = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    mlir::Value c1 = getOrCreateConstantStatic(loc, 1, builder, constantCache);

    // All loops use original init args during construction
    // This ensures SSA validity - origInits are defined outside all loops
    // The correct iter_args chaining is established in replaceLoopWithNewBounds
    ValueRange inits = hasIterArgs ? origInits : ValueRange{};

    // CRITICAL: Must provide explicit body builder to ensure yield terminator is created
    // Without this, scf.for may create an empty body without terminator
    auto wrapperLoop = builder.create<mlir::scf::ForOp>(
      loc, c0, c0, c1, inits,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value /*iv*/, mlir::ValueRange iterArgs) {
        // Create yield with iter args (pass through by default)
        nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc, iterArgs);
      });

    // Insert topLoop before the terminator in wrapperLoop's body
    auto *terminator = wrapperLoop.getBody()->getTerminator();
    mlir::Block::iterator insertLoc = terminator ? terminator->getIterator() : wrapperLoop.getBody()->end();
    wrapperLoop.getBody()->getOperations().splice(insertLoc, topLoop->getBlock()->getOperations(), topLoop);

    // CRITICAL: Update yield to use inner loop's results (for loops with iter_args)
    // This ensures value flow: inner.results -> outer.yield -> outer.results
    if (terminator && hasIterArgs && isa<mlir::scf::YieldOp>(terminator)) {
      if (isa<mlir::scf::ForOp>(topLoop)) {
        auto innerLoop = cast<mlir::scf::ForOp>(topLoop);
        auto results = innerLoop.getResults();
        if (!results.empty()) {
          terminator->setOperands(results);
        }
      }
    }

    tiledLoops[width - 1 - i] = wrapperLoop;
    topLoop = wrapperLoop.getOperation();
  }
}

// Build a map-for-to-forall wrapper loop directly in front of the band root and
// move the band root into its body. The wrapper carries the block-id iteration
// space `[0, useCore)`; downstream lowering replaces it with `get_block_idx`.
static mlir::scf::ForOp createParallelMapLoop(func::FuncOp funcOp, mlir::scf::ForOp bandRoot, int64_t useCore,
                                              OpBuilder &builder) {
  if (!bandRoot || useCore <= 0) {
    return mlir::scf::ForOp();
  }
  OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = bandRoot.getLoc();
  builder.setInsertionPoint(bandRoot);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value core = builder.create<arith::ConstantIndexOp>(loc, useCore);
  auto mapLoop = builder.create<mlir::scf::ForOp>(
    loc, c0, core, c1, ValueRange{},
    [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value, mlir::ValueRange) {
      nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc);
    });
  mapLoop->setAttr(kMapForToForallAttr, builder.getUnitAttr());
  funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(useCore));
  bandRoot->moveBefore(mapLoop.getBody()->getTerminator());
  return mapLoop;
}

// Wrapper-level no-split kind. Wrapper levels (FirstLevel/Middle) collapse the loop to a
// single iteration; the point level keeps the original range.
enum class NoSplitKind { FirstLevel, Middle, Point };

// Build the LoopBounds for a no-split origLoop at the requested level.
// `dropReductionInits` is honored only on wrapper levels (FirstLevel/Middle); the point level
// always keeps `origLoop.getInitArgs()` so per-iteration reduction semantics are preserved.
// `prevLoop` is only consulted for non-FirstLevel/non-reduction inits.
static LoopBounds buildNoSplitBounds(mlir::Location loc, mlir::scf::ForOp origLoop, mlir::scf::ForOp prevLoop,
                                     OpBuilder &builder, std::map<int64_t, Value> &constantCache, NoSplitKind kind,
                                     bool dropReductionInits) {
  LoopBounds bounds;
  if (kind == NoSplitKind::Point) {
    bounds.lb = recreateConstantOrSelf(origLoop.getLowerBound(), builder);
    bounds.ub = recreateConstantOrSelf(origLoop.getUpperBound(), builder);
    bounds.step = recreateConstantOrSelf(origLoop.getStep(), builder);
  } else {
    bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    bounds.ub = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  }
  if (isReductionLoop(origLoop)) {
    bounds.inits = (kind == NoSplitKind::Point || !dropReductionInits) ? ValueRange(origLoop.getInitArgs())
                                                                       : ValueRange{};
  } else {
    bounds.inits =
      (kind == NoSplitKind::FirstLevel) ? ValueRange(origLoop.getInitArgs()) : ValueRange(prevLoop.getResults());
  }
  return bounds;
}

// Helper: Create bounds for first level tile loop
static LoopBounds createFirstLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value tilesize,
                                                 unsigned tilesizeInt, OpBuilder &builder,
                                                 std::map<int64_t, Value> &constantCache,
                                                 bool dropFirstLevelReductionIterArgs) {
  if (isNoSplitLoop(origLoop)) {
    return buildNoSplitBounds(loc, origLoop, /*prevLoop=*/{}, builder, constantCache, NoSplitKind::FirstLevel,
                              dropFirstLevelReductionIterArgs);
  }

  LoopBounds bounds;
  Value origUb = origLoop.getUpperBound();

  // lb = 0
  bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);

  // Determine if this is a dynamic axis by checking origUb
  auto origUbConst = getConstantIndexValue(origUb);

  if (origUbConst && tilesizeInt != static_cast<unsigned>(-1) && tilesizeInt != 0) {
    // Static axis with known tilesize: compute ceildiv statically
    // Use tilesizeInt (original unsigned value) instead of trying to extract from Value
    int64_t numBlocks = (origUbConst.value() + tilesizeInt - 1) / tilesizeInt;
    bounds.ub = getOrCreateConstantStatic(loc, numBlocks, builder, constantCache);
  } else {
    // Dynamic axis: ub = kBlockDimSize (40)
    // tilesize should already be ceildiv(origUb, kernelsize)
    bounds.ub = getOrCreateConstantStatic(loc, kBlockDimSize, builder, constantCache);
  }

  // step = 1
  bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  bounds.inits = origLoop.getInitArgs();

  return bounds;
}

// Helper: Create bounds for middle level tile loop
static LoopBounds createMiddleLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value curTilesize,
                                                  Value prevTilesize, mlir::scf::ForOp prevLoop, OpBuilder &builder,
                                                  std::map<int64_t, Value> &constantCache, bool escapeReduceIterArgs) {
  if (isNoSplitLoop(origLoop)) {
    // Drop reduction inits on the middle level unless this loop's results escape and we want
    // sinkReduceLoopResultsToMiddleLevel to consume them.
    bool dropReductionInits = origLoop.getNumResults() == 0 || !escapeReduceIterArgs;
    return buildNoSplitBounds(loc, origLoop, prevLoop, builder, constantCache, NoSplitKind::Middle, dropReductionInits);
  }

  LoopBounds bounds;
  bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);

  // ub = ceildiv(previous tile size, current tile size)
  auto prevConst = getConstantIndexValue(prevTilesize);
  auto curConst = getConstantIndexValue(curTilesize);

  if (prevConst && curConst && curConst.value() != 0) {
    // Both are compile-time constants: compute ceildiv statically
    int64_t ubVal = (prevConst.value() + curConst.value() - 1) / curConst.value();
    bounds.ub = getOrCreateConstantStatic(loc, ubVal, builder, constantCache);
  } else {
    // curTilesize is dynamic: compute ceildiv dynamically
    auto ceildivMap =
      AffineMap::get(1, 1, builder.getAffineDimExpr(0).ceilDiv(builder.getAffineSymbolExpr(0)), builder.getContext());
    bounds.ub = builder.create<mlir::affine::AffineApplyOp>(
      loc, ceildivMap, ValueRange{recreateConstantOrSelf(prevTilesize, builder), curTilesize});
  }

  // step = 1
  bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);

  bounds.inits = prevLoop.getResults();
  return bounds;
}

// Helper: Create bounds for point loop
static LoopBounds createPointLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop,
                                        OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  if (isNoSplitLoop(origLoop)) {
    return buildNoSplitBounds(loc, origLoop, prevLoop, builder, constantCache, NoSplitKind::Point,
                              /*dropReductionInits=*/false);
  }

  LoopBounds bounds;
  Value origUb = origLoop.getUpperBound();
  Value clampedOrigUb = recreateConstantOrSelf(origUb, builder);
  SmallVector<Value, 8> operands;
  operands.reserve(levelInfo.size() * 2 + 1);
  for (const auto &[iv, _] : levelInfo) operands.push_back(iv);
  operands.push_back(clampedOrigUb);
  for (const auto &[_, tileSize] : levelInfo) operands.push_back(tileSize);

  auto levels = static_cast<unsigned>(levelInfo.size());
  AffineExpr lbExpr = builder.getAffineConstantExpr(0);
  AffineExpr origUbExpr = builder.getAffineDimExpr(levels);
  SmallVector<AffineExpr, 4> ubExprs{origUbExpr};
  for (unsigned k = 0; k < levels; ++k) {
    lbExpr = lbExpr + builder.getAffineDimExpr(k) * builder.getAffineSymbolExpr(k);
    ubExprs.push_back(lbExpr + builder.getAffineSymbolExpr(k));
  }

  SmallVector<AffineExpr, 2> lbExprs{lbExpr, origUbExpr};
  auto lbMap = AffineMap::get(levels + 1, levels, lbExprs, builder.getContext());
  bounds.lb = builder.create<mlir::affine::AffineMinOp>(loc, lbMap, operands);

  auto ubMap = AffineMap::get(levels + 1, levels, ubExprs, builder.getContext());
  bounds.ub = builder.create<mlir::affine::AffineMinOp>(loc, ubMap, operands);

  // step = 1
  bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  bounds.inits = prevLoop.getResults();

  return bounds;
}

// Helper: Replace loop with new bounds (create new loop, move body, replace old)
static mlir::scf::ForOp replaceLoopWithNewBounds(mlir::scf::ForOp oldLoop, const LoopBounds &bounds, mlir::Location loc,
                                                 OpBuilder &builder) {
  builder.setInsertionPoint(oldLoop);

  // Create new loop with explicit body builder to ensure yield is created
  auto newLoop = builder.create<mlir::scf::ForOp>(
    loc, bounds.lb, bounds.ub, bounds.step, bounds.inits,
    [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value /*iv*/, mlir::ValueRange iterArgs) {
      // Create default yield - will be updated later with correct operands
      nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc, iterArgs);
    });

  auto *oldBody = oldLoop.getBody();
  auto *newBody = newLoop.getBody();

  // CRITICAL: Replace uses of old IV and old region iter args BEFORE saving yield operands
  // This ensures that yield operands reference the new loop's values after replacement
  oldLoop.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());
  for (auto [oldArg, newArg] : llvm::zip(oldLoop.getRegionIterArgs(), newLoop.getRegionIterArgs())) {
    oldArg.replaceAllUsesWith(newArg);
  }
  if (newLoop.getRegionIterArgs().empty() && !oldLoop.getRegionIterArgs().empty()) {
    for (auto [oldArg, initArg] : llvm::zip(oldLoop.getRegionIterArgs(), oldLoop.getInitArgs())) {
      oldArg.replaceAllUsesWith(initArg);
    }
  }

  // Save old yield operands AFTER replacing uses
  // Now yield operands correctly reference new loop's regionIterArgs if they were using old ones
  SmallVector<Value> yieldOperands;
  if (auto oldYield = dyn_cast<mlir::scf::YieldOp>(oldBody->getTerminator())) {
    yieldOperands.assign(oldYield.getOperands().begin(), oldYield.getOperands().end());
  }

  // Remove the default yield created by body builder
  auto *newTerminator = newBody->getTerminator();
  if (newTerminator) {
    newTerminator->erase();
  }

  // Move body operations from old to new (excluding terminator)
  if (!oldBody->empty()) {
    Operation *oldTerminator = oldBody->getTerminator();
    if (oldTerminator) {
      // Move all operations except the terminator
      for (Operation &op : llvm::make_early_inc_range(*oldBody)) {
        if (&op == oldTerminator) break;
        op.moveBefore(newBody, newBody->end());
      }
    }
  }

  // Create new terminator with the saved operands.
  builder.setInsertionPointToEnd(newBody);
  if (newLoop.getNumResults() == 0) {
    builder.create<mlir::scf::YieldOp>(loc);
  } else if (yieldOperands.empty()) {
    auto regionIterArgs = newLoop.getRegionIterArgs();
    if (regionIterArgs.empty()) {
      builder.create<mlir::scf::YieldOp>(loc);
    } else {
      builder.create<mlir::scf::YieldOp>(loc, regionIterArgs);
    }
  } else {
    builder.create<mlir::scf::YieldOp>(loc, yieldOperands);
  }

  // Replace all uses of old loop results with new loop results
  if (oldLoop.getNumResults() == newLoop.getNumResults()) {
    oldLoop.replaceAllUsesWith(newLoop.getResults());
  }
  oldLoop.erase();

  return newLoop;
}

static bool buildLoopBoundsForTileLevel(int i, int j, int bandSize, int tileNum, int curTile, int lastTile,
                                        MutableArrayRef<mlir::scf::ForOp> newLoops, ArrayRef<mlir::scf::ForOp> band,
                                        ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                        const SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                        mlir::Location loc, mlir::OpBuilder &loopBuilder,
                                        std::map<int64_t, Value> &constantCache, bool escapeReduceIterArgs,
                                        TileRewriteContext &ctx, LoopBounds &bounds) {
  mlir::scf::ForOp origLoop = band[j];

  if (i == 0) {
    unsigned tilesizeInt =
      (curTile < static_cast<int>(tileSizesInt.size())) ? tileSizesInt[curTile] : static_cast<unsigned>(-1);
    bool dropFirstLevelReductionIterArgs = ctx.dropMappedOutermostFirstLevelIterArgs && j == 0;
    bounds = createFirstLevelTileLoopBounds(loc, origLoop, tileSizeValues[curTile], tilesizeInt, loopBuilder,
                                            constantCache, dropFirstLevelReductionIterArgs);
    if (ctx.parallelMapLoop && j == ctx.parallelDim && ctx.parallelUseCore > 0) {
      Value core = loopBuilder.create<arith::ConstantIndexOp>(loc, ctx.parallelUseCore);
      Value remaining = loopBuilder.create<arith::SubIOp>(loc, bounds.ub, ctx.parallelMapLoop.getInductionVar());
      bounds.ub = loopBuilder.create<arith::CeilDivSIOp>(loc, remaining, core);
    }
    return true;
  }

  if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) {
    return false;
  }
  mlir::scf::ForOp prevLoop = newLoops[lastTile];

  if (i == tileNum) {
    bounds = createPointLoopBounds(loc, origLoop, tileLevelInfo[j], prevLoop, loopBuilder, constantCache);
    return true;
  }

  bounds = createMiddleLevelTileLoopBounds(loc, origLoop, tileSizeValues[curTile], tileSizeValues[lastTile], prevLoop,
                                           loopBuilder, constantCache, escapeReduceIterArgs);
  return true;
}

static bool shouldPropagateTransposeAttrToTileLoop(int tileLevel, int tileNum, mlir::scf::ForOp origLoop) {
  if (!origLoop || !origLoop->hasAttr(kTransposeLoopAttr)) {
    return false;
  }
  return tileLevel == tileNum || origLoop->hasAttr(kTreeLeafAttr);
}

static bool shouldDeleteRedundantTileLoop(int i, int j, int bandSize, int tileNum, mlir::scf::ForOp origLoop,
                                          ArrayRef<unsigned> tileSizesInt) {
  if (i >= tileNum) return false;
  unsigned curTile = static_cast<unsigned>(i * bandSize + j);
  if (curTile >= tileSizesInt.size() || tileSizesInt[curTile] == static_cast<unsigned>(-1)) return false;
  if (i == 0) {
    std::optional<int64_t> lb = getConstantIndexValue(origLoop.getLowerBound());
    std::optional<int64_t> ub = getConstantIndexValue(origLoop.getUpperBound());
    return lb && ub && *ub - *lb == static_cast<int64_t>(tileSizesInt[curTile]);
  }
  unsigned prevTile = static_cast<unsigned>((i - 1) * bandSize + j);
  return prevTile < tileSizesInt.size() && tileSizesInt[prevTile] == tileSizesInt[curTile];
}

static void updateLoopAttrsForTileLoop(int i, int j, int bandSize, int tileNum, mlir::scf::ForOp origLoop,
                                       mlir::scf::ForOp newLoop, ArrayRef<unsigned> tileSizesInt,
                                       mlir::OpBuilder &builder) {
  if (!origLoop) {
    return;
  }

  if (shouldDeleteRedundantTileLoop(i, j, bandSize, tileNum, origLoop, tileSizesInt)) {
    newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
  }

  if (shouldPropagateTransposeAttrToTileLoop(i, tileNum, origLoop)) {
    newLoop->setAttr(kTransposeLoopAttr, builder.getUnitAttr());
  }
  if (origLoop->hasAttr(kParallelAxisAttr)) {
    newLoop->setAttr(kParallelAxisAttr, builder.getUnitAttr());
  }

  if (isReductionLoop(origLoop)) {
    ReduceDirection reduceType = getReduceType(origLoop);
    if (reduceType == ReduceDirection::Y) {
      // For reduce-y, remove all wrappers and keep only the point loop in final IR.
      if (i < tileNum) {
        newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
      }
      if (i == tileNum) {
        newLoop->setAttr(kReductionLoopAttr, builder.getI64IntegerAttr(kVectorSize));
      }
      return;
    }

    // For reduce-x/all, wrapper levels are single-iteration shells after no-split tiling.
    if (i < tileNum) {
      newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
    }

    // For inner point loop: preserve reduction attribute from original loop.
    if (i == tileNum && j == (bandSize - 1)) {
      newLoop->setAttr(kReductionLoopAttr, builder.getI64IntegerAttr(kVectorSize));
    }
    return;
  }

  if (origLoop->hasAttr(kNotInnerDimensionBroadcastLoopAttr) && i < tileNum) {
    newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
  }
}

static void copySemanticLoopAttrsToPointLoop(mlir::scf::ForOp origLoop, mlir::scf::ForOp pointLoop) {
  assert(origLoop && pointLoop && "copySemanticLoopAttrsToPointLoop: null loop");
  for (NamedAttribute attr : origLoop->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == kTreeNodeIdAttr || name == kTreeLeafAttr || name == kDeleteLoopAttr || name == kInnerLoopAttr ||
        name == kParallelAxisAttr) {
      continue;
    }
    pointLoop->setAttr(attr.getName(), attr.getValue());
  }
  pointLoop->setAttr(kInnerLoopAttr, UnitAttr::get(pointLoop.getContext()));
}

static void recordTileLevelInfoForLoop(int i, int j, int tileNum, int curTile, ArrayRef<Value> tileSizeValues,
                                        mlir::scf::ForOp newLoop,
                                        SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                        Value mappedIv = Value()) {
  if (i >= tileNum || curTile >= static_cast<int>(tileSizeValues.size())) {
    return;
  }

  tileLevelInfo[j].push_back({mappedIv ? mappedIv : newLoop.getInductionVar(), tileSizeValues[curTile]});
}

static bool tryFoldFirstTileLoopIntoParallelMapLoop(
  int i, int j, int tileNum, int curTile, MutableArrayRef<mlir::scf::ForOp> newLoops,
  ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
  SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo, TileRewriteContext &ctx) {
  if (!ctx.parallelMapLoop || i != 0 || j != ctx.parallelDim || ctx.parallelUseCore <= 0) {
    return false;
  }

  mlir::scf::ForOp loop = newLoops[curTile];
  if (!loop || loop.getNumResults() != 0 || ctx.parallelDim < 0) {
    return false;
  }

  unsigned parallelDim = static_cast<unsigned>(ctx.parallelDim);
  if (parallelDim >= band.size() || parallelDim >= tileSizesInt.size()) {
    return false;
  }

  int64_t parallelWork = 0;
  if (!isParallelCandidate(band[parallelDim], tileSizesInt[parallelDim], parallelWork) ||
      parallelWork > ctx.parallelUseCore) {
    return false;
  }

  loop.getInductionVar().replaceAllUsesWith(ctx.parallelMapLoop.getInductionVar());
  Block *body = loop.getBody();
  Operation *terminator = body->getTerminator();
  while (!body->empty() && &body->front() != terminator) {
    body->front().moveBefore(loop);
  }
  loop.erase();
  newLoops[curTile] = ctx.parallelMapLoop;
  recordTileLevelInfoForLoop(i, j, tileNum, curTile, tileSizeValues, ctx.parallelMapLoop, tileLevelInfo,
                             ctx.parallelMapLoop.getInductionVar());
  return true;
}

// Helper: Process a single tile loop (extracted to reduce cyclomatic complexity)
static void processSingleTileLoop(int i, int j, int bandSize, int tileNum, MutableArrayRef<mlir::scf::ForOp> newLoops,
                                  ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                  ArrayRef<unsigned> tileSizesInt,
                                  SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                  mlir::Location loc, std::map<int64_t, Value> &constantCache,
                                  TileRewriteContext &ctx) {
  int curTile = i * bandSize + j;
  int lastTile = curTile - bandSize;
  if (curTile >= static_cast<int>(newLoops.size())) {
    return;
  }

  mlir::scf::ForOp loop = newLoops[curTile];
  if (!loop) {
    return;
  }

  if (tryFoldFirstTileLoopIntoParallelMapLoop(i, j, tileNum, curTile, newLoops, band, tileSizeValues, tileSizesInt,
                                              tileLevelInfo, ctx)) {
    return;
  }

  mlir::scf::ForOp origLoop = band[j];
  mlir::OpBuilder loopBuilder(loop);
  bool escapeReduceIterArgs = origLoop && ctx.escapeReduceLoops.contains(origLoop.getOperation());

  LoopBounds bounds;
  if (!buildLoopBoundsForTileLevel(i, j, bandSize, tileNum, curTile, lastTile, newLoops, band, tileSizeValues,
                                   tileSizesInt, tileLevelInfo, loc, loopBuilder, constantCache, escapeReduceIterArgs,
                                   ctx, bounds)) {
    return;
  }

  mlir::scf::ForOp newLoop = replaceLoopWithNewBounds(loop, bounds, loc, loopBuilder);
  newLoops[curTile] = newLoop;
  Value mappedIv;
  if (ctx.parallelMapLoop && i == 0 && j == ctx.parallelDim && ctx.parallelUseCore > 0) {
    OpBuilder::InsertionGuard guard(loopBuilder);
    loopBuilder.setInsertionPointToStart(newLoop.getBody());
    Value core = loopBuilder.create<arith::ConstantIndexOp>(loc, ctx.parallelUseCore);
    Value scaled = loopBuilder.create<arith::MulIOp>(loc, newLoop.getInductionVar(), core);
    mappedIv = loopBuilder.create<arith::AddIOp>(loc, scaled, ctx.parallelMapLoop.getInductionVar());
  }

  if (i == tileNum) {
    copySemanticLoopAttrsToPointLoop(origLoop, newLoop);
    if (lastTile >= 0 && lastTile < static_cast<int>(tileSizesInt.size())) {
      unsigned pointVectorSize = tileSizesInt[lastTile];
      if (pointVectorSize != static_cast<unsigned>(-1)) {
        newLoop->setAttr(kInnerLoopAttr, loopBuilder.getI64IntegerAttr(pointVectorSize));
      }
    }
  }

  updateLoopAttrsForTileLoop(i, j, bandSize, tileNum, origLoop, newLoop, tileSizesInt, loopBuilder);
  recordTileLevelInfoForLoop(i, j, tileNum, curTile, tileSizeValues, newLoop, tileLevelInfo, mappedIv);
}

// Helper function to construct tiled index (bounds and steps) using tileSizeValues from memref
static void constructTiledIndexStatic(MutableArrayRef<mlir::scf::ForOp> newLoops, ArrayRef<mlir::scf::ForOp> band,
                                      ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                      OpBuilder &builder, std::map<int64_t, Value> &constantCache,
                                      TileRewriteContext &ctx) {
  int bandSize = static_cast<int>(band.size());
  if (bandSize == 0 || tileSizeValues.size() == 0) {
    return;
  }

  mlir::Location loc = band[0]->getLoc();
  int tileNum = static_cast<int>(tileSizeValues.size()) / bandSize;

  // Track tile level info for each dimension: {IV, tilesize}
  SmallVector<SmallVector<std::pair<Value, Value>, 4>, 4> tileLevelInfo(bandSize);

  // Process each tile level and dimension
  for (int i = 0; i <= tileNum; ++i) {
    for (int j = 0; j < bandSize; ++j) {
      processSingleTileLoop(i, j, bandSize, tileNum, newLoops, band, tileSizeValues, tileSizesInt, tileLevelInfo, loc,
                            constantCache, ctx);
    }
  }
}

// Static helper function to recursively create tail blocks for nested loops
static LogicalResult createTailBlockForBodyStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                                  std::map<int64_t, Value> &constantCache) {
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (auto bodyOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
      if (failed(createTailBlockStatic(bodyOp, builder, constantCache))) {
        return failure();
      }
    }
  }
  return success();
}

// Static helper function to create tail block for a single loop
static LogicalResult createTailBlockStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                           std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = forOp.getLoc();

  auto [staticDiff, dynamicUb, dynamicLb] = getDifferenceUbAndLb(forOp);

  if (staticDiff.has_value()) {
    return createTailBlockStaticImpl(forOp, staticDiff.value(), builder, constantCache);
  } else if (dynamicUb && dynamicLb) {
    builder.setInsertionPoint(forOp);
    mlir::Value dynamicDiff = builder.create<mlir::arith::SubIOp>(loc, dynamicUb, dynamicLb);
    return createTailBlockDynamicImpl(forOp, dynamicDiff, builder, constantCache);
  }

  builder.setInsertionPoint(forOp);
  return createTailBlockForBodyStatic(forOp, builder, constantCache);
}

// Replace `forOp` with a new scf.for using `newUb` (same lb/step/initArgs), moving body
// and forwarding results. Returns the newly created loop. `forOp` is erased.
static mlir::scf::ForOp rewriteForOpWithNewUb(mlir::scf::ForOp forOp, mlir::Value newUb, OpBuilder &builder) {
  mlir::Location loc = forOp.getLoc();
  builder.setInsertionPoint(forOp);
  auto newLoop =
    builder.create<mlir::scf::ForOp>(loc, forOp.getLowerBound(), newUb, forOp.getStep(), forOp.getInitArgs());
  newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(), forOp.getBody()->getOperations(),
                                            forOp.getBody()->begin(), std::prev(forOp.getBody()->end()));
  forOp.replaceAllUsesWith(newLoop.getResults());
  forOp.erase();
  return newLoop;
}

// Emit a continuation tail loop right after `mainLoop` covering [tailLb, tailUb) with step `tailStep`,
// cloning `mainLoop` body into it (IV remapped) and recursively splitting the tail body.
static LogicalResult emitTailContinuationLoop(mlir::scf::ForOp mainLoop, mlir::Value tailLb, mlir::Value tailUb,
                                              mlir::Value tailStep, OpBuilder &builder,
                                              std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = mainLoop.getLoc();
  builder.setInsertionPointAfter(mainLoop);
  auto tailLoop = builder.create<mlir::scf::ForOp>(loc, tailLb, tailUb, tailStep, mainLoop.getInitArgs());
  mlir::IRMapping mapping;
  mapping.map(mainLoop.getInductionVar(), tailLoop.getInductionVar());
  builder.setInsertionPointToStart(tailLoop.getBody());
  for (auto &op : mainLoop.getBody()->without_terminator()) {
    builder.clone(op, mapping);
  }
  return createTailBlockForBodyStatic(tailLoop, builder, constantCache);
}

// Static helper function to create tail block for static bounds
static LogicalResult createTailBlockStaticImpl(mlir::scf::ForOp forOp, int64_t differenceUbAndLb, OpBuilder &builder,
                                               std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = forOp.getLoc();
  mlir::Value origUb = forOp.getUpperBound();
  auto origStepOpt = getConstantIndexValue(forOp.getStep());
  if (!origStepOpt.has_value()) {
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }
  int64_t origStep = origStepOpt.value();
  int64_t tailSize = differenceUbAndLb % origStep;
  if (tailSize == 0) {
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }

  builder.setInsertionPoint(forOp);
  mlir::Value tailSizeVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
  mlir::Value newUb = builder.create<mlir::arith::SubIOp>(loc, origUb, tailSizeVal);
  int64_t newDifferenceUbAndLb = differenceUbAndLb - tailSize;

  // Branch A: tail-only — rewrite forOp to skip the tail and emit a single tail continuation.
  if (differenceUbAndLb < origStep && newDifferenceUbAndLb) {
    mlir::scf::ForOp mainLoop = rewriteForOpWithNewUb(forOp, newUb, builder);
    mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
    return emitTailContinuationLoop(mainLoop, newUb, origUb, tailStepVal, builder, constantCache);
  }

  // Branches B/C: process main body first.
  mlir::scf::ForOp currentLoop = (differenceUbAndLb >= origStep) ? rewriteForOpWithNewUb(forOp, newUb, builder) : forOp;
  if (failed(createTailBlockForBodyStatic(currentLoop, builder, constantCache))) {
    return failure();
  }
  bool isEqualToBlock = (newDifferenceUbAndLb == 0) || (newDifferenceUbAndLb == origStep && tailSize == origStep);
  if (differenceUbAndLb < origStep || !newDifferenceUbAndLb || isEqualToBlock) {
    return success();
  }
  mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
  return emitTailContinuationLoop(currentLoop, newUb, origUb, tailStepVal, builder, constantCache);
}

// Static helper function to create tail block for dynamic bounds
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value /*dynamicBound*/,
                                                OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = forOp.getLoc();
  mlir::Value origLb = forOp.getLowerBound();
  mlir::Value origUb = forOp.getUpperBound();
  auto origStepOpt = getConstantIndexValue(forOp.getStep());
  if (!origStepOpt.has_value()) {
    builder.setInsertionPoint(forOp);
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }
  int64_t origStep = origStepOpt.value();
  if (origStep == 1) {
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }

  builder.setInsertionPoint(forOp);
  mlir::Value recalculatedDynamicBound = builder.create<mlir::arith::SubIOp>(loc, origUb, origLb);
  mlir::Value stepVal = getOrCreateConstantStatic(loc, origStep, builder, constantCache);
  mlir::Value remainder = builder.create<mlir::arith::RemSIOp>(loc, recalculatedDynamicBound, stepVal);
  mlir::Value newUb = builder.create<mlir::arith::SubIOp>(loc, origUb, remainder);

  mlir::scf::ForOp currentLoop = rewriteForOpWithNewUb(forOp, newUb, builder);
  if (failed(createTailBlockForBodyStatic(currentLoop, builder, constantCache))) {
    return failure();
  }
  auto remainderConst = getConstantIndexValue(remainder);
  if (remainderConst && remainderConst.value() == 0) {
    return success();
  }
  builder.setInsertionPointAfter(currentLoop);
  mlir::Value tailStepValConst = getOrCreateConstantStatic(loc, origStep, builder, constantCache);
  mlir::Value tailDynamicBound = builder.create<mlir::arith::SubIOp>(loc, origUb, origLb);
  mlir::Value tailStepVal = builder.create<mlir::arith::RemSIOp>(loc, tailDynamicBound, tailStepValConst);
  return emitTailContinuationLoop(currentLoop, newUb, origUb, tailStepVal, builder, constantCache);
}

static llvm::DenseSet<mlir::Operation *> collectEscapingReductionLoops(ArrayRef<mlir::scf::ForOp> band) {
  llvm::DenseSet<mlir::Operation *> escapeReduceLoops;
  if (band.empty()) {
    return escapeReduceLoops;
  }

  mlir::scf::ForOp bandRoot = band.front();
  for (mlir::scf::ForOp loop : band) {
    if (isLoopResultEscapingBand(loop, bandRoot)) {
      escapeReduceLoops.insert(loop.getOperation());
    }
  }
  return escapeReduceLoops;
}

static LogicalResult cloneAndWireTiledLoops(ArrayRef<mlir::scf::ForOp> band,
                                            MutableArrayRef<mlir::scf::ForOp> tiledLoops, unsigned tileSizesNum,
                                            mlir::scf::ForOp rootScfForOp, OpBuilder &builder) {
  mlir::IRMapping mapping;
  if (failed(cloneNonPerfectChainIntoPointLoops(band, tiledLoops, tileSizesNum, builder, mapping))) {
    return failure();
  }
  if (failed(cloneComputeIntoInnermostPointLoop(band, tiledLoops, tileSizesNum, rootScfForOp, builder, mapping))) {
    return failure();
  }
  updatePointLoopYieldsFromOriginalLoops(band, tiledLoops, tileSizesNum, mapping, builder);
  forwardIterArgsThroughWrapperLoops(tiledLoops, builder);
  return success();
}

static LogicalResult finalizeRootLoopAfterTiling(mlir::scf::ForOp rootScfForOp,
                                                 MutableArrayRef<mlir::scf::ForOp> tiledLoops, unsigned tileSizesNum,
                                                 unsigned forNum,
                                                 const llvm::DenseSet<mlir::Operation *> &escapeReduceLoops,
                                                 mlir::scf::ForOp parallelMapLoop,
                                                 bool dropMappedOutermostFirstLevelIterArgs, OpBuilder &builder) {
  bool rootEscape = escapeReduceLoops.contains(rootScfForOp.getOperation());
  if (isReductionLoopWithIterArgs(rootScfForOp) && rootEscape) {
    if (dropMappedOutermostFirstLevelIterArgs) {
      mlir::scf::ForOp userAnchorLoop = parallelMapLoop ? parallelMapLoop : tiledLoops[0];
      if (failed(sinkReduceLoopResultsToMiddleLevel(rootScfForOp, userAnchorLoop, tiledLoops, tileSizesNum, forNum,
                                                    builder))) {
        return failure();
      }
    } else if (rootScfForOp.getNumResults() > 0 && tiledLoops[0]) {
      if (tiledLoops[0].getNumResults() == rootScfForOp.getNumResults()) {
        rootScfForOp.replaceAllUsesWith(tiledLoops[0].getResults());
      } else if (!rootScfForOp.use_empty()) {
        return failure();
      }
    } else if (!rootScfForOp.use_empty()) {
      return failure();
    }
  } else if (rootScfForOp.getNumResults() > 0 && tiledLoops[0]) {
    // Replace original loop's results with outermost tiled loop's results before erasing.
    if (tiledLoops[0].getNumResults() == rootScfForOp.getNumResults()) {
      rootScfForOp.replaceAllUsesWith(tiledLoops[0].getResults());
    } else if (!rootScfForOp.use_empty()) {
      return failure();
    }
  }

  rootScfForOp.erase();
  return success();
}

// Apply tiling to a single loop. The caller is responsible for creating the
// optional `parallelMapLoop` (via createParallelMapLoop) that already wraps the
// band root; `parallelUseCore > 0` must be paired with a non-null map loop and
// with `loop` carrying `kParallelAxisAttr`.
static LogicalResult applyTilingToLoop(mlir::scf::ForOp loop, ArrayRef<Value> tileSizeValues,
                                       ArrayRef<unsigned> tileSizesInt, OpBuilder &builder,
                                       std::map<int64_t, Value> &constantCache,
                                       mlir::scf::ForOp parallelMapLoop, int64_t parallelUseCore) {
  unsigned tileSizesNum = tileSizeValues.size();

  SmallVector<mlir::scf::ForOp, 1> band{loop};
  // The band root is "outermost" w.r.t. its tile nest if either it has no
  // surrounding scf.for at all, or its only surrounding scf.for is the
  // parallel map loop created by the caller.
  mlir::scf::ForOp parentFor = loop->getParentOfType<mlir::scf::ForOp>();
  unsigned width = tileSizesNum + 1;
  SmallVector<mlir::scf::ForOp, 6> tiledLoops(width);

  llvm::DenseSet<mlir::Operation *> escapeReduceLoops = collectEscapingReductionLoops(band);
  TileRewriteContext ctx{escapeReduceLoops, parallelMapLoop,
                         /*parallelDim=*/(parallelMapLoop && parallelUseCore > 0 && loop->hasAttr(kParallelAxisAttr))
                           ? 0
                           : -1,
                         parallelUseCore,
                         /*dropMappedOutermostFirstLevelIterArgs=*/!parentFor || parentFor == parallelMapLoop};

  constructTiledLoopStatic(loop, width, tiledLoops, builder, constantCache);

  // Replace all dummy loops first, before cloning operations, so IV mapping points to final loops.
  constructTiledIndexStatic(tiledLoops, band, tileSizeValues, tileSizesInt, builder, constantCache, ctx);

  if (failed(cloneAndWireTiledLoops(band, tiledLoops, tileSizesNum, loop, builder))) {
    return failure();
  }

  return finalizeRootLoopAfterTiling(loop, tiledLoops, tileSizesNum, /*forNum=*/1, escapeReduceLoops, parallelMapLoop,
                                     ctx.dropMappedOutermostFirstLevelIterArgs, builder);
}

static void clearTemporaryLoopIdentificationAttrs(func::FuncOp funcOp, bool clearPointLoopAttr = true) {
  funcOp.walk([&](mlir::scf::ForOp loop) {
    if (loop->hasAttr(kTreeNodeIdAttr)) {
      loop->removeAttr(kTreeNodeIdAttr);
    }
    if (loop->hasAttr(kTreeLeafAttr)) {
      loop->removeAttr(kTreeLeafAttr);
    }
    if (loop->hasAttr(kParallelAxisAttr)) {
      loop->removeAttr(kParallelAxisAttr);
    }
    if (clearPointLoopAttr && loop->hasAttr(kInnerLoopAttr)) {
      loop->removeAttr(kInnerLoopAttr);
    }
  });
}

// Emit an error on `funcOp` and clean temporary identification attrs in one go.
// Always returns `failure()` so call sites can `return emitTilingFailure(...)`.
static LogicalResult emitTilingFailure(func::FuncOp funcOp, const Twine &msg) {
  clearTemporaryLoopIdentificationAttrs(funcOp, /*clearPointLoopAttr=*/false);
  funcOp.emitError() << msg;
  return failure();
}

static LogicalResult findUniqueLoopByTreeNodeId(func::FuncOp funcOp, int64_t nodeId, mlir::scf::ForOp &loop) {
  loop = mlir::scf::ForOp();
  int64_t matchCount = 0;
  funcOp.walk([&](mlir::scf::ForOp candidate) {
    auto idAttr = candidate->getAttrOfType<IntegerAttr>(kTreeNodeIdAttr);
    if (!idAttr || idAttr.getInt() != nodeId) {
      return;
    }
    ++matchCount;
    loop = candidate;
  });
  return matchCount == 1 ? success() : failure();
}

// Pick the best parallel axis in `band` and wrap band[0] with a map-for-to-forall
// loop. When no axis is eligible but the band root can be wrapped, a degenerate
// `useCore=1` map loop is still created so that downstream sees `hacc.block_dim`.
// Returns a null ForOp only when the band root carries non-reduction iter_args
// (in which case parallel binding is unsafe).
static mlir::scf::ForOp markAndWrapParallelAxis(func::FuncOp funcOp, ArrayRef<mlir::scf::ForOp> band,
                                                 ArrayRef<unsigned> tileSizesInt, unsigned tileLevels,
                                                 OpBuilder &builder, int &parallelDim, int64_t &parallelUseCore,
                                                 bool allowLastAxis = false) {
  parallelDim = markParallelAxis(funcOp, band, tileSizesInt, tileLevels, builder, parallelUseCore, allowLastAxis);
  mlir::scf::ForOp root = band.front();
  if (root.getNumResults() != 0 && !isReductionLoopWithIterArgs(root)) {
    if (parallelDim >= 0) {
      band[parallelDim]->removeAttr(kParallelAxisAttr);
    }
    parallelDim = -1;
    parallelUseCore = 0;
    return mlir::scf::ForOp();
  }
  if (parallelDim < 0) {
    parallelUseCore = 1;
  }
  return createParallelMapLoop(funcOp, root, parallelUseCore, builder);
}

// Tile `band` axis by axis. Each axis is re-located by its tree-node id before
// tiling so that earlier axes' transformations remain valid. `parallelMapLoop`,
// `parallelDim`, `parallelUseCore` describe an already-created parallel binding
// (no-op when `parallelDim < 0`).
static LogicalResult applyDecoupledAxisTiling(func::FuncOp funcOp, OpBuilder &builder, StringRef errLabel,
                                              ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                              ArrayRef<unsigned> tileSizesInt, int parallelDim,
                                              int64_t parallelUseCore, mlir::scf::ForOp parallelMapLoop,
                                              int64_t &nextNodeId) {
  unsigned bandSize = band.size();
  unsigned tileLevels = tileSizeValues.size() / bandSize;

  SmallVector<int64_t, 6> axisNodeIds;
  axisNodeIds.reserve(bandSize);
  for (unsigned dim = 0; dim < bandSize; ++dim) {
    int64_t axisNodeId = nextNodeId++;
    band[dim]->setAttr(kTreeNodeIdAttr, builder.getI64IntegerAttr(axisNodeId));
    axisNodeIds.push_back(axisNodeId);
  }

  for (unsigned dim = 0; dim < bandSize; ++dim) {
    mlir::scf::ForOp activeAxisLoop;
    if (failed(findUniqueLoopByTreeNodeId(funcOp, axisNodeIds[dim], activeAxisLoop))) {
      return emitTilingFailure(funcOp, "failed to locate unique axis loop for " + Twine(errLabel) + " band");
    }
    activeAxisLoop->removeAttr(kParallelAxisAttr);
    if (static_cast<int>(dim) == parallelDim) {
      activeAxisLoop->setAttr(kParallelAxisAttr, builder.getUnitAttr());
    }

    SmallVector<Value, 6> axisTileValues;
    SmallVector<unsigned, 6> axisTileSizesInt;
    axisTileValues.reserve(tileLevels);
    axisTileSizesInt.reserve(tileLevels);
    for (unsigned level = 0; level < tileLevels; ++level) {
      size_t idx = static_cast<size_t>(level) * bandSize + dim;
      axisTileValues.push_back(tileSizeValues[idx]);
      axisTileSizesInt.push_back(tileSizesInt[idx]);
    }

    std::map<int64_t, Value> axisConstantCache;
    if (Operation *defOp = axisTileValues.back().getDefiningOp()) {
      builder.setInsertionPointAfter(defOp);
    }

    int64_t axisParallelUseCore = (static_cast<int>(dim) == parallelDim) ? parallelUseCore : 0;
    if (failed(applyTilingToLoop(activeAxisLoop, axisTileValues, axisTileSizesInt, builder, axisConstantCache,
                                 parallelMapLoop, axisParallelUseCore))) {
      return emitTilingFailure(funcOp, "failed to apply " + Twine(errLabel) + " axis tiling");
    }
  }
  return success();
}

// Helper function to check if a scf.for loop is innermost (no nested scf.for inside)
static bool isInnermostScfLoop(mlir::scf::ForOp forOp) {
  Block *body = forOp.getBody();
  if (!body) {
    return true;
  }

  for (Operation &op : body->without_terminator()) {
    if (isa<mlir::scf::ForOp>(op)) {
      return false;
    }
  }

  return true;
}

// Strip index cast chains so index comparison is stable across trivial cast rewrites.
static Value stripIndexLikeCasts(Value value) {
  Value cur = value;
  while (true) {
    if (auto castOp = cur.getDefiningOp<arith::IndexCastOp>()) {
      cur = castOp.getIn();
      continue;
    }
    if (auto castUIOp = cur.getDefiningOp<arith::IndexCastUIOp>()) {
      cur = castUIOp.getIn();
      continue;
    }
    return cur;
  }
}

struct TransposeOrderPairInfo {
  SmallVector<mlir::scf::ForOp, 4> lhsCommonOrder;
  SmallVector<mlir::scf::ForOp, 4> rhsCommonOrder;
  size_t commonPrefix = 0;
};

static bool intersectLoopOrdersForTranspose(ArrayRef<mlir::scf::ForOp> lhsOrder, ArrayRef<mlir::scf::ForOp> rhsOrder,
                                            SmallVector<mlir::scf::ForOp, 4> &lhsCommonOrder,
                                            SmallVector<mlir::scf::ForOp, 4> &rhsCommonOrder) {
  if (lhsOrder.size() < 2 || rhsOrder.size() < 2) {
    return false;
  }

  llvm::DenseSet<Operation *> rhsLoops;
  for (mlir::scf::ForOp loop : rhsOrder) {
    rhsLoops.insert(loop.getOperation());
  }

  lhsCommonOrder.clear();
  for (mlir::scf::ForOp loop : lhsOrder) {
    if (rhsLoops.contains(loop.getOperation())) {
      lhsCommonOrder.push_back(loop);
    }
  }

  llvm::DenseSet<Operation *> lhsLoops;
  for (mlir::scf::ForOp loop : lhsOrder) {
    lhsLoops.insert(loop.getOperation());
  }

  rhsCommonOrder.clear();
  for (mlir::scf::ForOp loop : rhsOrder) {
    if (lhsLoops.contains(loop.getOperation())) {
      rhsCommonOrder.push_back(loop);
    }
  }

  return lhsCommonOrder.size() >= 2 && lhsCommonOrder.size() == rhsCommonOrder.size();
}

static std::optional<TransposeOrderPairInfo> buildTransposeOrderPairAnalysis(ArrayRef<mlir::scf::ForOp> band,
                                                                             ArrayRef<mlir::scf::ForOp> lhsOrder,
                                                                             ArrayRef<mlir::scf::ForOp> rhsOrder,
                                                                             bool linearBand) {
  SmallVector<mlir::scf::ForOp, 4> lhsCommonOrder;
  SmallVector<mlir::scf::ForOp, 4> rhsCommonOrder;
  if (!intersectLoopOrdersForTranspose(lhsOrder, rhsOrder, lhsCommonOrder, rhsCommonOrder)) {
    return std::nullopt;
  }
  // Tree bands keep the strict rule: the innermost index must actually participate in the permutation,
  // otherwise downstream tree-aware tiling cannot use the transpose path safely.
  if (!linearBand && lhsCommonOrder.back() == rhsCommonOrder.back()) {
    return std::nullopt;
  }

  size_t commonPrefix = 0;
  while (commonPrefix < lhsCommonOrder.size() && lhsCommonOrder[commonPrefix] == rhsCommonOrder[commonPrefix]) {
    ++commonPrefix;
  }
  if (commonPrefix == lhsCommonOrder.size()) {
    return std::nullopt;
  }
  if (!std::is_permutation(lhsCommonOrder.begin() + commonPrefix, lhsCommonOrder.end(),
                           rhsCommonOrder.begin() + commonPrefix, rhsCommonOrder.end())) {
    return std::nullopt;
  }

  if (!linearBand &&
      std::find(lhsCommonOrder.begin() + commonPrefix, lhsCommonOrder.end(), band.back()) == lhsCommonOrder.end()) {
    return std::nullopt;
  }

  return TransposeOrderPairInfo{std::move(lhsCommonOrder), std::move(rhsCommonOrder), commonPrefix};
}

static void clearBandBroadcastAttrsForTranspose(ArrayRef<mlir::scf::ForOp> band) {
  for (mlir::scf::ForOp loop : band) {
    loop->removeAttr(kBroadcastLoopAttr);
    loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
  }
}

// Non-tree (single-chain) band: mark every loop from the outermost permuted loop down to band.back(),
// keeping a contiguous transpose suffix that applyBandTiling's transpose path requires.
static void markTransposeAttrsLinearBandSuffix(ArrayRef<mlir::scf::ForOp> band, const TransposeOrderPairInfo &info,
                                               UnitAttr transposeAttr) {
  const size_t commonPrefix = info.commonPrefix;
  const auto &lhsCommonOrder = info.lhsCommonOrder;
  const auto &rhsCommonOrder = info.rhsCommonOrder;

  llvm::DenseSet<Operation *> permutedLoops;
  for (auto it = lhsCommonOrder.begin() + commonPrefix; it != lhsCommonOrder.end(); ++it) {
    mlir::scf::ForOp loopOp = *it;
    permutedLoops.insert(loopOp.getOperation());
  }
  for (auto it = rhsCommonOrder.begin() + commonPrefix; it != rhsCommonOrder.end(); ++it) {
    mlir::scf::ForOp loopOp = *it;
    permutedLoops.insert(loopOp.getOperation());
  }
  size_t startIdx = band.size();
  for (size_t i = 0; i < band.size(); ++i) {
    mlir::scf::ForOp loop = band[i];
    if (permutedLoops.contains(loop.getOperation())) {
      startIdx = i;
      break;
    }
  }
  for (size_t i = startIdx; i < band.size(); ++i) {
    mlir::scf::ForOp loop = band[i];
    if (loop->hasAttr(kReductionLoopAttr) && !isLeafForLoop(loop)) {
      for (mlir::scf::ForOp bandLoop : band) {
        bandLoop->removeAttr(kTransposeLoopAttr);
      }
      return;
    }
  }
  for (size_t i = startIdx; i < band.size(); ++i) {
    mlir::scf::ForOp loop = band[i];
    loop->setAttr(kTransposeLoopAttr, transposeAttr);
  }
}

static void markTransposeAttrsTreeCommonSuffix(const TransposeOrderPairInfo &info, UnitAttr transposeAttr) {
  const size_t commonPrefix = info.commonPrefix;
  const auto &lhsCommonOrder = info.lhsCommonOrder;
  const auto &rhsCommonOrder = info.rhsCommonOrder;

  auto hasBlockingReduction = [&](ArrayRef<mlir::scf::ForOp> order) {
    for (auto it = order.begin() + commonPrefix; it != order.end(); ++it) {
      mlir::scf::ForOp loopOp = *it;
      if (loopOp->hasAttr(kReductionLoopAttr) && !isLeafForLoop(loopOp)) {
        return true;
      }
    }
    return false;
  };
  if (hasBlockingReduction(lhsCommonOrder) || hasBlockingReduction(rhsCommonOrder)) {
    return;
  }

  for (auto it = lhsCommonOrder.begin() + commonPrefix; it != lhsCommonOrder.end(); ++it) {
    mlir::scf::ForOp loopOp = *it;
    loopOp->setAttr(kTransposeLoopAttr, transposeAttr);
  }
  for (auto it = rhsCommonOrder.begin() + commonPrefix; it != rhsCommonOrder.end(); ++it) {
    mlir::scf::ForOp loopOp = *it;
    loopOp->setAttr(kTransposeLoopAttr, transposeAttr);
  }
}

static void applyTransposeLoopAttrsForOrderPair(ArrayRef<mlir::scf::ForOp> band, const TransposeOrderPairInfo &info,
                                                UnitAttr transposeAttr, bool linearBand) {
  clearBandBroadcastAttrsForTranspose(band);
  if (linearBand) {
    markTransposeAttrsLinearBandSuffix(band, info, transposeAttr);
    return;
  }
  markTransposeAttrsTreeCommonSuffix(info, transposeAttr);
}

static void markTransposeOrderPair(ArrayRef<mlir::scf::ForOp> band, ArrayRef<mlir::scf::ForOp> lhsOrder,
                                   ArrayRef<mlir::scf::ForOp> rhsOrder, UnitAttr transposeAttr, bool linearBand) {
  std::optional<TransposeOrderPairInfo> info = buildTransposeOrderPairAnalysis(band, lhsOrder, rhsOrder, linearBand);
  if (!info) {
    return;
  }
  applyTransposeLoopAttrsForOrderPair(band, *info, transposeAttr, linearBand);
}

static SmallVector<mlir::scf::ForOp, 4> extractMemrefLoopOrder(Operation *op, ArrayRef<mlir::scf::ForOp> band) {
  SmallVector<mlir::scf::ForOp, 4> order;
  auto appendIfMatch = [&](Value idx) {
    Value strippedIdx = stripIndexLikeCasts(idx);
    for (mlir::scf::ForOp loop : band) {
      if (stripIndexLikeCasts(loop.getInductionVar()) == strippedIdx) {
        order.push_back(loop);
        break;
      }
    }
  };

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    for (Value idx : loadOp.getIndices()) {
      appendIfMatch(idx);
    }
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    for (Value idx : storeOp.getIndices()) {
      appendIfMatch(idx);
    }
  }
  return order;
}

static void reconcileTransposeAttrsAfterBranchScan(ArrayRef<SmallVector<mlir::scf::ForOp, 6>> leafBands,
                                                   UnitAttr transposeAttr) {
  const bool hasTransposeLeaf = llvm::any_of(leafBands, [](const SmallVector<mlir::scf::ForOp, 6> &band) {
    return !band.empty() && band.back()->hasAttr(kTransposeLoopAttr);
  });
  for (ArrayRef<mlir::scf::ForOp> band : leafBands) {
    if (band.empty()) {
      continue;
    }
    if (hasTransposeLeaf) {
      band.back()->setAttr(kTransposeLoopAttr, transposeAttr);
      continue;
    }
    for (mlir::scf::ForOp loop : band) {
      loop->removeAttr(kTransposeLoopAttr);
    }
  }
}

static void markBandTransposeLoops(func::FuncOp funcOp, const LeafBranchBandPlan &plan) {
  SmallVector<SmallVector<mlir::scf::ForOp, 6>, 4> leafBands;
  if (!plan.representativeBand.empty()) {
    leafBands.push_back(plan.representativeBand);
    for (mlir::scf::ForOp peerLeaf : plan.peerLeafLoops) {
      SmallVector<mlir::scf::ForOp, 6> peerBand(plan.representativeBand.begin(), plan.representativeBand.end());
      peerBand.back() = peerLeaf;
      leafBands.push_back(std::move(peerBand));
    }
  }

  auto transposeAttr = UnitAttr::get(funcOp.getContext());
  const bool linearBand = !plan.hasLeafBranching;
  for (ArrayRef<mlir::scf::ForOp> band : leafBands) {
    if (band.empty()) {
      continue;
    }

    mlir::scf::ForOp leafLoop = band.back();
    SmallVector<Operation *, 8> memOps;
    leafLoop.walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op)) {
        memOps.push_back(op);
      }
    });

    for (size_t i = 0; i < memOps.size(); ++i) {
      SmallVector<mlir::scf::ForOp, 4> lhsOrder = extractMemrefLoopOrder(memOps[i], band);
      for (size_t j = i + 1; j < memOps.size(); ++j) {
        SmallVector<mlir::scf::ForOp, 4> rhsOrder = extractMemrefLoopOrder(memOps[j], band);
        markTransposeOrderPair(band, lhsOrder, rhsOrder, transposeAttr, linearBand);
      }
    }
  }

  if (plan.hasLeafBranching) {
    reconcileTransposeAttrsAfterBranchScan(leafBands, transposeAttr);
  }
}

// Detect reduction type by scanning all ops inside the loop.
static ReduceDirection getReduceType(mlir::scf::ForOp loop) {
  if (!loop) {
    return ReduceDirection::UNKNOWN;
  }

  ReduceDirection result = ReduceDirection::UNKNOWN;
  loop.getBody()->walk([&](Operation *op) {
    auto typeAttr = op->getAttrOfType<StringAttr>(kReductionTypeStr);
    if (!typeAttr) {
      return mlir::WalkResult::advance();
    }

    StringRef typeStr = typeAttr.getValue();
    if (typeStr == "all") {
      result = ReduceDirection::ALL;
      return mlir::WalkResult::interrupt();
    }
    if (typeStr == "y") {
      result = ReduceDirection::Y;
      return mlir::WalkResult::advance();
    }
    if (typeStr == "x" && result == ReduceDirection::UNKNOWN) {
      result = ReduceDirection::X;
    }
    return mlir::WalkResult::advance();
  });

  return result;
}

static bool hasPointLoopInAncestorChain(mlir::scf::ForOp loop) {
  for (mlir::scf::ForOp curLoop = loop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (curLoop->hasAttr(kInnerLoopAttr)) {
      return true;
    }
  }
  return false;
}

static bool isVectorSelectablePointLoop(mlir::scf::ForOp loop) {
  return loop && loop->hasAttr(kInnerLoopAttr) && !loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr);
}

static bool shouldSkipVectorAttrCandidate(mlir::scf::ForOp loop, bool restrictToPointLoops, bool skipDeleteLoops) {
  if (!loop) {
    return true;
  }
  if (restrictToPointLoops && !isVectorSelectablePointLoop(loop)) {
    return true;
  }
  if (skipDeleteLoops && loop->hasAttr(kDeleteLoopAttr)) {
    return true;
  }
  return false;
}

static mlir::scf::ForOp findVectorAttrTargetLoop(mlir::scf::ForOp startLoop, bool skipDeleteLoops) {
  bool restrictToPointLoops = hasPointLoopInAncestorChain(startLoop);

  for (mlir::scf::ForOp curLoop = startLoop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (shouldSkipVectorAttrCandidate(curLoop, restrictToPointLoops, skipDeleteLoops)) {
      continue;
    }
    return curLoop;
  }

  return mlir::scf::ForOp();
}

static void markTransposeLoopChainWithVectorAttr(mlir::scf::ForOp innermostLoop, OpBuilder &builder,
                                                 mlir::scf::ForOp stopLoop = mlir::scf::ForOp()) {
  for (mlir::scf::ForOp curLoop = innermostLoop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (shouldSkipVectorAttrCandidate(curLoop, /*restrictToPointLoops=*/false, /*skipDeleteLoops=*/true)) {
      if (curLoop == stopLoop) {
        break;
      }
      continue;
    }
    if (!stopLoop && !curLoop->hasAttr(kTransposeLoopAttr)) {
      break;
    }

    if (curLoop->hasAttr(kTransposeLoopAttr)) {
      curLoop->removeAttr(kTransposeLoopAttr);
    }
    if (curLoop->hasAttr(kReductionLoopAttr)) {
      curLoop->removeAttr(kReductionLoopAttr);
      ReduceDirection reduceType = getReduceType(curLoop);
      if (reduceType == ReduceDirection::Y) {
        curLoop->setAttr(kReductionYLoopAttr, builder.getUnitAttr());
      } else if (reduceType == ReduceDirection::ALL) {
        curLoop->setAttr(kReductionAllLoopAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
      } else {
        curLoop->setAttr(kReductionXLoopAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
      }
    } else {
      curLoop->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
    }
    if (curLoop == stopLoop) {
      break;
    }
  }
}

// Multi-dim vec analogue of `markTransposeLoopChainWithVectorAttr`. Walks the
// ancestor chain from `innermostLoop` upward, converting every loop tagged
// with `kMultiVecLoopAttr` into the appropriate vec / reduction marker. The
// strategy is responsible for *not* placing `kMultiVecLoopAttr` on innermost
// `reduce_y` loops, so this function only needs to handle reduction X / ALL
// alongside the normal vec case.
static void markMultiVecLoopChainWithVectorAttr(mlir::scf::ForOp innermostLoop, OpBuilder &builder) {
  for (mlir::scf::ForOp curLoop = innermostLoop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (shouldSkipVectorAttrCandidate(curLoop, /*restrictToPointLoops=*/false, /*skipDeleteLoops=*/true)) {
      continue;
    }
    if (!curLoop->hasAttr(kMultiVecLoopAttr)) {
      break;
    }
    curLoop->removeAttr(kMultiVecLoopAttr);
    if (curLoop->hasAttr(kReductionLoopAttr)) {
      curLoop->removeAttr(kReductionLoopAttr);
      ReduceDirection reduceType = getReduceType(curLoop);
      if (reduceType == ReduceDirection::ALL) {
        curLoop->setAttr(kReductionAllLoopAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
      } else {
        // reduce_y was banned by the strategy; treat the residual UNKNOWN /
        // X case as reduction along the contiguous dimension.
        curLoop->setAttr(kReductionXLoopAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
      }
    } else {
      curLoop->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
    }
  }
}

static bool hasReductionVectorAttr(mlir::scf::ForOp loop) {
  return loop && (loop->hasAttr(kReductionLoopAttr) || loop->hasAttr(kReductionXLoopAttr) ||
                  loop->hasAttr(kReductionYLoopAttr) || loop->hasAttr(kReductionAllLoopAttr));
}

static void markBroadcastLoopChainWithVectorAttr(mlir::scf::ForOp vectorTarget, OpBuilder &builder) {
  vectorTarget.walk([&](mlir::scf::ForOp loop) {
    if (!loop || loop == vectorTarget || hasReductionVectorAttr(loop)) {
      return;
    }
    loop->setAttr(kBroadcastLoopAttr, builder.getI64IntegerAttr(getLoopExtent(loop)));
  });
}

// Mark all innermost scf.for loops with vector attribute.
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder) {
  funcOp->walk([&](mlir::scf::ForOp forOp) {
    if (isInnermostScfLoop(forOp)) {
      auto reduceType = ReduceDirection::UNKNOWN;
      const bool hasReductionAttr = forOp->hasAttr(kReductionLoopAttr);
      // Multi-dim vec is consumed first: when an innermost point loop is part
      // of a `kMultiVecLoopAttr` chain, the whole chain owns the vec marking.
      // The strategy guarantees this branch is mutually exclusive with the
      // transpose / broadcast paths below, so the early-return is safe.
      if (forOp->hasAttr(kMultiVecLoopAttr)) {
        markMultiVecLoopChainWithVectorAttr(forOp, builder);
        return;
      }
      if (forOp->hasAttr(kTransposeLoopAttr)) {
        markTransposeLoopChainWithVectorAttr(forOp, builder);
        return;
      }
      // set vector attribute to innermost loops
      if (!hasReductionAttr) {
        for (mlir::scf::ForOp curLoop = forOp->getParentOfType<mlir::scf::ForOp>(); curLoop;
             curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
          if (curLoop->hasAttr(kTransposeLoopAttr)) {
            markTransposeLoopChainWithVectorAttr(forOp, builder, curLoop);
            return;
          }
        }
        if (mlir::scf::ForOp vectorTarget = findVectorAttrTargetLoop(forOp, /*skipDeleteLoops=*/false)) {
          if (vectorTarget->hasAttr(kBroadcastLoopAttr)) {
            markBroadcastLoopChainWithVectorAttr(vectorTarget, builder);
            vectorTarget->removeAttr(kBroadcastLoopAttr);
          }
          vectorTarget->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(vectorTarget)));
        }
      } else {
        forOp->removeAttr(kReductionLoopAttr);
        reduceType = getReduceType(forOp);

        if (reduceType == ReduceDirection::X) {
          forOp->setAttr(kReductionXLoopAttr,
                         builder.getI64IntegerAttr(std::min<int64_t>(getLoopExtent(forOp), kVectorSize)));
        } else if (reduceType == ReduceDirection::Y) {
          forOp->setAttr(kReductionYLoopAttr, builder.getUnitAttr());
        } else if (reduceType == ReduceDirection::ALL) {
          forOp->setAttr(kReductionAllLoopAttr,
                         builder.getI64IntegerAttr(std::min<int64_t>(getLoopExtent(forOp), kVectorSize)));
        }

        forOp = forOp->getParentOfType<mlir::scf::ForOp>();
      }

      // For reduce-y: clear reduction tags outward and set vector tag on the first non-reduction parent loop.
      if (reduceType == ReduceDirection::Y) {
        mlir::scf::ForOp curLoop = forOp;
        bool restrictToPointLoops = hasPointLoopInAncestorChain(curLoop);
        while (curLoop) {
          if (curLoop->hasAttr(kReductionLoopAttr)) {
            curLoop->removeAttr(kReductionLoopAttr);
            curLoop = curLoop->getParentOfType<mlir::scf::ForOp>();
            continue;
          }
          if (shouldSkipVectorAttrCandidate(curLoop, restrictToPointLoops, /*skipDeleteLoops=*/true)) {
            curLoop = curLoop->getParentOfType<mlir::scf::ForOp>();
            continue;
          }
          curLoop->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
          break;
        }
      }
    } else {
      if (forOp->hasAttr(kReductionLoopAttr)) {
        forOp->removeAttr(kReductionLoopAttr);
      }
    }
  });
}

// Set hacc.block_dim attribute to funcOp based on outermost tiled loop's upper bound
static void setBlockDimAttribute(func::FuncOp funcOp, OpBuilder &builder) {
  if (funcOp->hasAttr(kBlockDimAttr)) {
    return;
  }
  // Find the first outermost scf.for loop in the function body
  Block *body = &funcOp.getBody().front();
  mlir::scf::ForOp outermostLoop;

  for (Operation &op : *body) {
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(&op)) {
      outermostLoop = forOp;
      break;
    }
  }

  if (outermostLoop) {
    Value kernelnum = outermostLoop.getUpperBound();

    // Try to get constant value
    if (auto constantKernelnum = getConstantIndexValue(kernelnum)) {
      funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(constantKernelnum.value()));
    } else {
      // use kBlockDimSize as block dimension
      funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(kBlockDimSize));
    }
  }
}

// Inline loops marked with delete when they are guaranteed to execute exactly once.
static void inlineDeleteMarkedLoops(func::FuncOp funcOp, OpBuilder &builder) {
  SmallVector<mlir::scf::ForOp, 8> loopsToInline;

  funcOp->walk([&](mlir::scf::ForOp forOp) {
    if (!forOp->hasAttr(kDeleteLoopAttr)) {
      return;
    }
    loopsToInline.push_back(forOp);
  });

  // Inline these loops by moving their body operations to their parent block
  for (auto loop : loopsToInline) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(loop);

    Block *loopBody = loop.getBody();

    // Handle iter_args: replace region iter args with loop's init args
    for (auto [regionArg, initArg] : llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs())) {
      regionArg.replaceAllUsesWith(initArg);
    }

    // Replace loop IV with constant 0 (since it only executes once with iv=0)
    Value iv = loop.getInductionVar();
    Value c0 = builder.create<arith::ConstantIndexOp>(loop.getLoc(), 0);
    iv.replaceAllUsesWith(c0);

    // Move all operations from loop body to parent block (except terminator)
    SmallVector<Operation *, 16> opsToMove;
    std::transform(loopBody->without_terminator().begin(), loopBody->without_terminator().end(),
                   std::back_inserter(opsToMove), [](Operation &op) { return &op; });
    std::for_each(opsToMove.begin(), opsToMove.end(), [&](Operation *op) { op->moveBefore(loop); });

    // Handle yield operands: these become the loop's results
    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(loopBody->getTerminator());
    if (yieldOp) {
      loop.replaceAllUsesWith(yieldOp.getOperands());
    } else {
      loop.replaceAllUsesWith(ValueRange{});
    }

    // Erase the loop
    loop.erase();
  }
}

// Helper: Build tiling function signature (add tiling key and data memref args)
static void buildTilingFunctionSignature(FunctionType origTy, MLIRContext *ctx, OpBuilder &builder,
                                         SmallVector<Type> &argTypes, SmallVector<Type> &resTypes,
                                         int64_t tilingStructMemrefSize) {
  argTypes.assign(origTy.getInputs().begin(), origTy.getInputs().end());

  auto i64Ty = builder.getI64Type();
  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
  auto memrefTy = MemRefType::get({tilingStructMemrefSize}, i64Ty);

  argTypes.push_back(llvmPtrTy);
  argTypes.push_back(memrefTy);

  resTypes.clear();
}

// Helper: Create and initialize tiling function (set attrs, write strategy key)
static func::FuncOp createAndInitTilingFunc(func::FuncOp originalKernel, ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resTypes, OpBuilder &builder) {
  auto *ctx = builder.getContext();
  auto loc = originalKernel.getLoc();

  std::string baseName = originalKernel.getSymName().str();
  std::string name = baseName + "_00_tiling_function";

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(originalKernel);

  auto f = builder.create<func::FuncOp>(loc, name, FunctionType::get(ctx, argTypes, resTypes));
  f.addEntryBlock();

  copyHaccIOAttrsFrom(originalKernel, f);
  f->setAttr(hacc::HACCFuncTypeAttr::name, hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::HOST));

  unsigned numArgs = f.getNumArguments();
  unsigned keyIdx = numArgs - 2;
  unsigned tilingDataIdx = numArgs - 1;
  setTilingKeyAndDataArgAttrs(f, keyIdx, tilingDataIdx, ctx);

  // Write strategy index to tiling key
  OpBuilder b(&f.getBody().front(), f.getBody().front().end());
  Value tilingKey = f.getArgument(keyIdx);
  int64_t strategyIndex = getSelectedTilingStrategyIndex();
  Value strategyValue = b.create<arith::ConstantIntOp>(loc, strategyIndex, 64);
  b.create<LLVM::StoreOp>(loc, strategyValue, tilingKey);

  return f;
}

// Helper: Store tile sizes to memref
static LogicalResult storeTileSizesToMemref(func::FuncOp tilingFunc, func::FuncOp originalKernel,
                                            ArrayRef<SmallVector<mlir::scf::ForOp, 6>> bands,
                                            ArrayRef<SmallVector<unsigned, 6>> allBandTileSizes,
                                            ArrayRef<SmallVector<int, 6>> allBandConstraintMaxs,
                                            ArrayRef<std::vector<DynamicAxisMapping>> allBandDynamicMappings,
                                            OpBuilder &builder) {
  auto loc = tilingFunc.getLoc();
  unsigned numArgs = tilingFunc.getNumArguments();
  unsigned tilingDataIdx = numArgs - 1;

  OpBuilder b(&tilingFunc.getBody().front(), tilingFunc.getBody().front().end());
  Value dataMem = tilingFunc.getArgument(tilingDataIdx);
  auto i64Ty = builder.getI64Type();

  size_t memrefOffset = 0;

  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &bandConstraintMaxs = allBandConstraintMaxs[bandIdx];
    const auto &bandDynamicMapping = allBandDynamicMappings[bandIdx];
    size_t bandSize = bands[bandIdx].size();

    // Cache first level step values (as i64) for dynamic axes to avoid recomputation
    SmallVector<Value, 6> firstLevelStepI64Cache(bandSize, Value());

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, memrefOffset);
      unsigned tileSize = bandTileSizes[tileIdx];

      // Calculate level and dimIdx for current tile
      size_t level = tileIdx / bandSize;
      size_t dimIdx = tileIdx % bandSize;
      size_t firstLevelIdx = 0 * bandSize + dimIdx;

      if (tileSize == static_cast<unsigned>(-1)) {
        // Dynamic tile size: compute min(constraintMax, dim) and store
        // The constraintMax comes from TilingStrategy's constraint upper bound
        const auto &mapping = bandDynamicMapping[tileIdx];
        int constraintMax = (tileIdx < bandConstraintMaxs.size()) ? bandConstraintMaxs[tileIdx] : 0;
        Value dim;
        if (mapping.upperBound) {
          dim = cloneUpperBoundDefinition(mapping.upperBound, originalKernel, tilingFunc, b);
          if (!dim) {
            llvm::dbgs() << "storeTileSizesToMemref: cloneUpperBoundDefinition failed, fallback to memref.dim\n";
          }
        }

        if (!dim) {
          if (mapping.inputMemrefIndex == UINT_MAX || mapping.dimIndex == UINT_MAX) {
            tilingFunc.emitError("dynamic axis mapping not found for tile index " + std::to_string(tileIdx));
            return failure();
          }

          if (mapping.inputMemrefIndex >= tilingFunc.getNumArguments() - 2) {
            tilingFunc.emitError("invalid memref argument index " + std::to_string(mapping.inputMemrefIndex));
            return failure();
          }

          Value memrefArg = tilingFunc.getArgument(mapping.inputMemrefIndex);
          if (!isa<MemRefType>(memrefArg.getType())) {
            tilingFunc.emitError("argument at index " + std::to_string(mapping.inputMemrefIndex) + " is not a memref");
            return failure();
          }

          Value dimIndexVal = b.create<arith::ConstantIndexOp>(loc, mapping.dimIndex);
          dim = b.create<memref::DimOp>(loc, memrefArg, dimIndexVal);
        }

        Value step = emitDynamicTilePerDim(loc, b, dim, constraintMax);
        Value stepI64 = b.create<arith::IndexCastOp>(loc, i64Ty, step);
        b.create<memref::StoreOp>(loc, stepI64, dataMem, ValueRange{idx});

        // Cache first level step for later use
        if (level == 0) {
          firstLevelStepI64Cache[dimIdx] = stepI64;
        }
      } else {
        // if first level is dynamic, need to compute min of second level
        if (level >= 1 && bandTileSizes[firstLevelIdx] == static_cast<unsigned>(-1)) {
          // First level is dynamic, use cached value
          Value firstLevelStepI64 = firstLevelStepI64Cache[dimIdx];

          Value staticTileSizeI64 = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(tileSize), 64);
          Value staticTileSize = b.create<arith::IndexCastOp>(loc, builder.getIndexType(), staticTileSizeI64);

          Value firstLevelStep = b.create<arith::IndexCastOp>(loc, builder.getIndexType(), firstLevelStepI64);
          Value minVal = b.create<arith::MinSIOp>(loc, staticTileSize, firstLevelStep);
          Value stepI64 = b.create<arith::IndexCastOp>(loc, i64Ty, minVal);
          b.create<memref::StoreOp>(loc, stepI64, dataMem, ValueRange{idx});
        } else {
          // Static tile size: store constant
          Value val = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(tileSize), 64);
          b.create<memref::StoreOp>(loc, val, dataMem, ValueRange{idx});
        }
      }

      memrefOffset++;
    }
  }

  b.create<func::ReturnOp>(loc);
  return success();
}

// Create default tiling function
static LogicalResult createTilingFuncDefault(func::FuncOp originalKernel, OpBuilder &builder, func::FuncOp &tilingFunc,
                                             bool isStaticShape = false) {
  auto *mlirCtx = builder.getContext();
  auto loc = originalKernel.getLoc();
  auto origTy = originalKernel.getFunctionType();
  int64_t tilingStructMemrefSize = 1;

  std::vector<LeafBranchBandPlan> leafBranchPlans;
  bool hasUnsupportedTreeShape = false;
  if (failed(buildLeafBranchBandPlans(originalKernel, leafBranchPlans, hasUnsupportedTreeShape))) {
    return failure();
  }
  std::vector<SmallVector<mlir::scf::ForOp, 6>> bandsToUse;
  collectRepresentativeBands(leafBranchPlans, bandsToUse);

  // Step 1: Dynamic-shape path computes tile metadata to derive tiling struct size.
  std::vector<SmallVector<unsigned, 6>> allBandTileSizes;
  std::vector<SmallVector<int, 6>> allBandConstraintMaxs;
  if (!isStaticShape) {
    if (!hasUnsupportedTreeShape) {
      if (!leafBranchPlans.empty()) {
        preprocessLoopAttrsForTileCalculation(originalKernel, leafBranchPlans.front());
      }
      [[maybe_unused]] auto clearNotInnerBroadcastGuard =
        llvm::make_scope_exit([&] { clearNotInnerDimensionBroadcastLoopAttr(originalKernel); });
      (void)clearNotInnerBroadcastGuard;
      size_t levelToTile = 0;
      if (failed(calculateTileSizesForBands(originalKernel, true, bandsToUse, allBandTileSizes, allBandConstraintMaxs,
                                            levelToTile))) {
        return failure();
      }

      tilingStructMemrefSize = std::accumulate(allBandTileSizes.begin(), allBandTileSizes.end(), int64_t{0},
                                               [](int64_t acc, const SmallVector<unsigned, 6> &bandTileSizes) {
                                                 return acc + static_cast<int64_t>(bandTileSizes.size());
                                               });
      if (tilingStructMemrefSize <= 0) tilingStructMemrefSize = 1;
    } else {
      bandsToUse.clear();
    }
  }

  // Step 2: Build function signature
  SmallVector<Type> argTypes, resTypes;
  mlirCtx->getOrLoadDialect<LLVM::LLVMDialect>();
  buildTilingFunctionSignature(origTy, mlirCtx, builder, argTypes, resTypes, tilingStructMemrefSize);

  // Step 3: Create and initialize tiling function.
  func::FuncOp f = createAndInitTilingFunc(originalKernel, argTypes, resTypes, builder);

  // Step 4: Static-shape or no-band path doesn't need memref store in create stage.
  if (isStaticShape || bandsToUse.empty()) {
    OpBuilder b(&f.getBody().front(), f.getBody().front().end());
    b.create<func::ReturnOp>(loc);
    tilingFunc = f;
    return success();
  }

  // Step 5: Build dynamic axis mappings (reuse existing function)
  std::vector<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &curBand = bandsToUse[bandIdx];
    allBandDynamicMappings.push_back(buildDynamicAxisMappingForBand(curBand, bandTileSizes, originalKernel));
  }

  // Step 6: Store tile sizes to memref (with constraint max for dynamic shapes)
  if (failed(storeTileSizesToMemref(f, originalKernel, bandsToUse, allBandTileSizes, allBandConstraintMaxs,
                                    allBandDynamicMappings, builder))) {
    return failure();
  }

  tilingFunc = f;
  return success();
}

// Public interface functions
LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape) {
  func::FuncOp tilingFunc;
  if (failed(createTilingFuncDefault(originalKernel, builder, tilingFunc, isStaticShape))) {
    return failure();
  }
  // func::FuncOp tilingFuncMemAnalysis;
  // if (failed(createTilingFuncMemAnalysis(originalKernel, builder, tilingFuncMemAnalysis, isStaticShape))) {
  //   return failure();
  // }
  out[0] = tilingFunc;
  // out[1] = tilingFuncMemAnalysis;
  return success();
}

int64_t getSelectedTilingStrategyIndex() { return 0; }

// Helper: Build dynamic axis mapping for a single band
static std::vector<DynamicAxisMapping> buildDynamicAxisMappingForBand(ArrayRef<mlir::scf::ForOp> band,
                                                                      ArrayRef<unsigned> bandTileSizes,
                                                                      func::FuncOp originalKernel) {
  std::vector<DynamicAxisMapping> bandDynamicMapping;

  for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
    unsigned tileSize = bandTileSizes[tileIdx];
    if (tileSize == static_cast<unsigned>(-1)) {
      // Dynamic tile size: trace back to memref argument
      size_t axisIdx = tileIdx % band.size();
      mlir::scf::ForOp forOp = band[axisIdx];
      Value upperBound = forOp.getUpperBound();

      auto [argIndex, dimIndex] = traceDynamicUpperBound(upperBound, originalKernel);
      if (argIndex >= 0 && dimIndex >= 0) {
        bandDynamicMapping.push_back({static_cast<unsigned>(argIndex), static_cast<unsigned>(dimIndex), upperBound});
      } else {
        // Fallback: use first memref argument
        for (unsigned i = 0; i < originalKernel.getNumArguments(); ++i) {
          Value arg = originalKernel.getArgument(i);
          if (isa<MemRefType>(arg.getType())) {
            bandDynamicMapping.push_back({i, 0, upperBound});
            break;
          }
        }
      }
    } else {
      // Static tile size
      bandDynamicMapping.push_back({UINT_MAX, UINT_MAX, Value()});
    }
  }

  return bandDynamicMapping;
}

// Emit per-dim dynamic tile-size: min(constraintMax, dim) if constraintMax > 0,
// else ceildiv(dim, kBlockDimSize).
static Value emitDynamicTilePerDim(Location loc, OpBuilder &builder, Value dim, int constraintMax) {
  if (constraintMax > 0) {
    Value constraintMaxVal = builder.create<arith::ConstantIndexOp>(loc, constraintMax);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, constraintMaxVal, dim);
    return builder.create<arith::SelectOp>(loc, cmp, constraintMaxVal, dim);
  }
  Value c40 = builder.create<arith::ConstantIndexOp>(loc, kBlockDimSize);
  return builder.create<arith::CeilDivSIOp>(loc, dim, c40);
}

// Helper: Compute dynamic tile size value (ceildiv(dim, 40))
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                 Value &result) {
  if (mapping.inputMemrefIndex == UINT_MAX || mapping.dimIndex == UINT_MAX ||
      mapping.inputMemrefIndex >= originalKernel.getNumArguments()) {
    return failure();
  }
  Value memrefArg = originalKernel.getArgument(mapping.inputMemrefIndex);
  if (!isa<MemRefType>(memrefArg.getType())) {
    return failure();
  }
  Value dimIndexVal = builder.create<arith::ConstantIndexOp>(loc, mapping.dimIndex);
  Value dim = builder.create<memref::DimOp>(loc, memrefArg, dimIndexVal);
  result = emitDynamicTilePerDim(loc, builder, dim, constraintMax);
  return success();
}

// Helper: Prepare tile sizes for static shape
static LogicalResult prepareTileSizesForStaticShape(func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                    const std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                    const std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                                    const std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                    std::vector<SmallVector<Value, 6>> &allTileSizeValues) {
  if (bandsToUse.size() != allBandTileSizes.size() || bandsToUse.size() != allBandConstraintMaxs.size()) {
    originalKernel.emitError("inconsistent band metadata when preparing static tile values");
    return failure();
  }

  // Build dynamic axis mappings for all bands.
  std::vector<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &curBand = bandsToUse[bandIdx];
    allBandDynamicMappings.push_back(buildDynamicAxisMappingForBand(curBand, bandTileSizes, originalKernel));
  }

  // Compute tile size values (with constraint max for dynamic axes)
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &bandConstraintMaxs = allBandConstraintMaxs[bandIdx];
    const auto &bandDynamicMapping = allBandDynamicMappings[bandIdx];
    SmallVector<Value, 6> tileSizeValues;

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      unsigned tileSize = bandTileSizes[tileIdx];
      Value tileSizeValue;

      if (tileSize == static_cast<unsigned>(-1)) {
        // Dynamic tile size: use constraint upper bound
        const auto &mapping = bandDynamicMapping[tileIdx];
        int constraintMax = (tileIdx < bandConstraintMaxs.size()) ? bandConstraintMaxs[tileIdx] : 0;
        if (failed(computeDynamicTileSizeValue(mapping, constraintMax, originalKernel, loc, builder, tileSizeValue))) {
          originalKernel.emitError("Failed to compute dynamic tile size for band " + std::to_string(bandIdx) +
                                   " tile " + std::to_string(tileIdx));
          return failure();
        }
      } else {
        // Static tile size
        tileSizeValue = builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(tileSize));
      }

      tileSizeValues.push_back(tileSizeValue);
    }

    allTileSizeValues.push_back(tileSizeValues);
  }

  return success();
}

// Helper: Prepare tile sizes from memref (dynamic shape)
static LogicalResult prepareTileSizesFromMemref(func::FuncOp originalKernel,
                                                ArrayRef<SmallVector<mlir::scf::ForOp, 6>> bands, mlir::Location loc,
                                                OpBuilder &builder,
                                                std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                                std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt) {
  auto args = originalKernel.getArguments();
  if (args.empty()) {
    originalKernel.emitError("originalKernel must have at least one argument (tileSizesMemref)");
    return failure();
  }

  Value tileSizesMemref = args.back();
  auto memrefType = dyn_cast<MemRefType>(tileSizesMemref.getType());

  if (!memrefType || memrefType.getRank() != 1 || !memrefType.getElementType().isInteger(64)) {
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    os << tileSizesMemref.getType();
    originalKernel.emitError("Last argument (tileSizesMemref) must be memref<?xi64>, got: " + typeStr);
    return failure();
  }

  size_t memrefOffset = 0;

  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &band = bands[bandIdx];
    unsigned forNum = band.size();
    unsigned bandTileSizesCount = forNum * 2;

    SmallVector<Value, 6> tileSizeValues;
    SmallVector<unsigned, 6> tileSizesInt;

    for (unsigned i = 0; i < bandTileSizesCount; ++i) {
      Value idx = builder.create<arith::ConstantIndexOp>(loc, memrefOffset + i);
      Value loaded = builder.create<memref::LoadOp>(loc, tileSizesMemref, ValueRange{idx});
      Value tileSizeIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), loaded);
      tileSizeValues.push_back(tileSizeIndex);

      // Try to extract constant tilesize by looking at the defining op chain
      unsigned tileSizeIntValue = static_cast<unsigned>(-1);

      // Trace back through index_cast -> load to find the stored constant
      if (auto indexCastOp = tileSizeIndex.getDefiningOp<arith::IndexCastOp>()) {
        Value castSource = indexCastOp.getIn();
        if (castSource.getDefiningOp<memref::LoadOp>()) {
          // Can't directly get the stored value, but we can check the tiling function
          // For now, mark as dynamic
          tileSizeIntValue = static_cast<unsigned>(-1);
        }
      }

      tileSizesInt.push_back(tileSizeIntValue);
    }

    allTileSizeValues.push_back(tileSizeValues);
    allBandTileSizesInt.push_back(tileSizesInt);
    memrefOffset += bandTileSizesCount;
  }

  return success();
}

static LogicalResult wrapFunctionBodyWithFor(func::FuncOp func, OpBuilder &builder) {
  Location loc = func.getLoc();

  if (!func.getBody().hasOneBlock()) {
    func.emitError("wrapFunctionBodyWithFor currently supports only single-block functions");
    return failure();
  }

  Block &entryBlock = func.getBody().front();
  Operation *terminator = entryBlock.getTerminator();
  if (!terminator) {
    func.emitError("entry block has no terminator");
    return failure();
  }

  llvm::SmallVector<Operation *, 16> opsToMove;
  for (auto it = entryBlock.begin(); it != Block::iterator(terminator); ++it) {
    opsToMove.push_back(&*it);
  }
  builder.setInsertionPointToStart(&entryBlock);

  Value lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  scf::ForOp forOp = builder.create<scf::ForOp>(loc, lb, ub, step);
  Block *forBody = forOp.getBody();
  Operation *forTerminator = forBody->getTerminator();
  for (Operation *op : opsToMove) {
    op->moveBefore(forTerminator);
  }
  return success();
}

static LogicalResult prepareLeafBranchPlansForApply(func::FuncOp originalKernel, OpBuilder &builder,
                                                    std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                                    bool &shouldReturnEarly) {
  shouldReturnEarly = false;
  bool hasUnsupportedTreeShape = false;
  if (failed(buildLeafBranchBandPlans(originalKernel, leafBranchPlans, hasUnsupportedTreeShape))) {
    return failure();
  }

  // Unsupported tree shape keeps stage-1 skip behavior.
  if (hasUnsupportedTreeShape) {
    clearAllBroadcastLoopAttrs(originalKernel);
    markInnermostLoopsWithVectorAttr(originalKernel, builder);
    originalKernel->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(1));
    shouldReturnEarly = true;
    return success();
  }

  if (leafBranchPlans.empty()) {
    if (failed(wrapFunctionBodyWithFor(originalKernel, builder))) {
      return failure();
    }
    markInnermostLoopsWithVectorAttr(originalKernel, builder);

    if (failed(buildLeafBranchBandPlans(originalKernel, leafBranchPlans, hasUnsupportedTreeShape))) {
      return failure();
    }

    if (hasUnsupportedTreeShape) {
      clearAllBroadcastLoopAttrs(originalKernel);
      markInnermostLoopsWithVectorAttr(originalKernel, builder);
      originalKernel->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(1));
      shouldReturnEarly = true;
      return success();
    }
  }

  return success();
}

static LogicalResult prepareTileMetadataForApply(func::FuncOp originalKernel, OpBuilder &builder, mlir::Location loc,
                                                 bool isStaticShape,
                                                 std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                 std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt,
                                                 std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                                 std::vector<SmallVector<Value, 6>> &allTileSizeValues) {
  // Always calculate tile sizes first to get unsigned values (with constraint max for dynamic axes)
  size_t levelToTile = 0;
  if (failed(calculateTileSizesForBands(originalKernel, true, bandsToUse, allBandTileSizesInt, allBandConstraintMaxs,
                                        levelToTile))) {
    return failure();
  }

  // Then prepare Value representations based on shape type.
  allTileSizeValues.clear();
  allTileSizeValues.reserve(bandsToUse.size());

  if (isStaticShape) {
    // Static shape: materialize Value tile sizes from the already solved tile metadata.
    if (failed(prepareTileSizesForStaticShape(originalKernel, loc, builder, bandsToUse, allBandTileSizesInt,
                                              allBandConstraintMaxs, allTileSizeValues))) {
      return failure();
    }
    return success();
  }

  // Dynamic shape: load tile sizes from memref (but we already have unsigned values)
  std::vector<SmallVector<unsigned, 6>> tempTileSizes;  // Will be overwritten
  return prepareTileSizesFromMemref(originalKernel, bandsToUse, loc, builder, allTileSizeValues, tempTileSizes);
}

static LogicalResult applySingleLinearBandDecoupled(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                    ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                                    ArrayRef<unsigned> tileSizesInt, int64_t &nextNodeId) {
  if (band.empty() || tileSizeValues.size() != tileSizesInt.size() || tileSizeValues.size() % band.size() != 0) {
    return emitTilingFailure(originalKernel, "invalid linear-band tiling metadata for band " + Twine(bandIdx));
  }

  unsigned bandSize = static_cast<unsigned>(band.size());
  unsigned tileLevels = static_cast<unsigned>(tileSizeValues.size() / bandSize);
  if (tileLevels == 0) {
    return emitTilingFailure(originalKernel, "empty tile levels for linear band " + Twine(bandIdx));
  }

  int parallelDim = -1;
  int64_t parallelUseCore = 0;
  mlir::scf::ForOp parallelMapLoop =
    markAndWrapParallelAxis(originalKernel, band, tileSizesInt, tileLevels, builder, parallelDim, parallelUseCore);

  // Decouple linear chain tiling by axis (outer->inner):
  // i{j{k}} -> i1,i2,i3{j1,j2,j3{k1,k2,k3}}.
  if (failed(applyDecoupledAxisTiling(originalKernel, builder, "linear", band, tileSizeValues, tileSizesInt,
                                      parallelDim, parallelUseCore, parallelMapLoop, nextNodeId))) {
    return failure();
  }

  return success();
}

static void buildLeafBranchTileSlices(
  const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
  ArrayRef<unsigned> tileSizesInt, unsigned bandSize, unsigned tileLevels, SmallVector<mlir::scf::ForOp, 6> &prefixBand,
  SmallVector<Value, 6> &prefixTileValues, SmallVector<unsigned, 6> &prefixTileSizesInt,
  SmallVector<Value, 6> &leafTileValues, SmallVector<unsigned, 6> &leafTileSizesInt) {
  // Shared prefix axes are tiled first (e.g. i1,i2,i3), then each leaf axis is tiled independently.
  for (unsigned dim = 0; dim < plan.representativeLeafDim; ++dim) {
    prefixBand.push_back(band[dim]);
  }

  prefixTileValues.reserve(tileLevels * plan.representativeLeafDim);
  prefixTileSizesInt.reserve(tileLevels * plan.representativeLeafDim);
  leafTileValues.reserve(tileLevels);
  leafTileSizesInt.reserve(tileLevels);
  for (unsigned level = 0; level < tileLevels; ++level) {
    for (unsigned dim = 0; dim < plan.representativeLeafDim; ++dim) {
      size_t idx = static_cast<size_t>(level) * bandSize + dim;
      prefixTileValues.push_back(tileSizeValues[idx]);
      prefixTileSizesInt.push_back(tileSizesInt[idx]);
    }
    size_t leafIdx = static_cast<size_t>(level) * bandSize + plan.representativeLeafDim;
    leafTileValues.push_back(tileSizeValues[leafIdx]);
    leafTileSizesInt.push_back(tileSizesInt[leafIdx]);
  }
}

static LogicalResult collectAndTagLeafBranchLoops(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                  const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band,
                                                  int64_t &nextNodeId, SmallVector<int64_t, 6> &branchLeafNodeIds) {
  mlir::scf::ForOp representativeLeaf = band[plan.representativeLeafDim];
  mlir::scf::ForOp branchPoint = representativeLeaf->getParentOfType<mlir::scf::ForOp>();
  if (!branchPoint) {
    return emitTilingFailure(originalKernel, "failed to locate leaf-branch parent loop for band " + Twine(bandIdx));
  }

  SmallVector<mlir::scf::ForOp, 6> branchLeaves;
  collectDirectChildLoopsInOrder(branchPoint, branchLeaves);
  if (branchLeaves.size() < 2 ||
      !llvm::all_of(branchLeaves, [](mlir::scf::ForOp loop) { return isLeafForLoop(loop); }) ||
      llvm::find(branchLeaves, representativeLeaf) == branchLeaves.end() ||
      branchLeaves.size() != plan.peerLeafLoops.size() + 1) {
    return emitTilingFailure(originalKernel, "invalid leaf-branch structure for band " + Twine(bandIdx));
  }

  branchLeafNodeIds.clear();
  branchLeafNodeIds.reserve(branchLeaves.size());
  for (mlir::scf::ForOp leafLoop : branchLeaves) {
    int64_t nodeId = nextNodeId++;
    leafLoop->setAttr(kTreeNodeIdAttr, builder.getI64IntegerAttr(nodeId));
    leafLoop->setAttr(kTreeLeafAttr, builder.getUnitAttr());
    branchLeafNodeIds.push_back(nodeId);
  }
  return success();
}

static LogicalResult applyLeafBandsByNodeId(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                            ArrayRef<int64_t> branchLeafNodeIds, ArrayRef<Value> leafTileValues,
                                            ArrayRef<unsigned> leafTileSizesInt) {
  for (int64_t nodeId : branchLeafNodeIds) {
    mlir::scf::ForOp activeLeafLoop;
    if (failed(findUniqueLoopByTreeNodeId(originalKernel, nodeId, activeLeafLoop))) {
      return emitTilingFailure(originalKernel, "failed to locate unique leaf loop by temporary node id");
    }

    std::map<int64_t, Value> leafConstantCache;
    if (!leafTileValues.empty()) {
      if (Operation *defOp = leafTileValues.back().getDefiningOp()) {
        builder.setInsertionPointAfter(defOp);
      }
    }

    if (failed(applyTilingToLoop(activeLeafLoop, leafTileValues, leafTileSizesInt, builder, leafConstantCache))) {
      return emitTilingFailure(originalKernel, "Failed to apply leaf tiling for band " + Twine(bandIdx));
    }
  }
  return success();
}

static LogicalResult applySingleLeafBranchBandDecoupled(func::FuncOp originalKernel, OpBuilder &builder, size_t bandIdx,
                                                        const LeafBranchBandPlan &plan, ArrayRef<mlir::scf::ForOp> band,
                                                        ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                                        int64_t &nextNodeId) {
  if (band.empty() || plan.representativeLeafDim == 0 || plan.representativeLeafDim >= band.size() ||
      tileSizeValues.size() != tileSizesInt.size() || tileSizeValues.size() % band.size() != 0) {
    return emitTilingFailure(originalKernel, "invalid leaf-branch tiling metadata for band " + Twine(bandIdx));
  }

  unsigned bandSize = static_cast<unsigned>(band.size());
  unsigned tileLevels = static_cast<unsigned>(tileSizeValues.size() / bandSize);
  if (tileLevels == 0) {
    return emitTilingFailure(originalKernel, "empty tile levels for leaf-branch band " + Twine(bandIdx));
  }

  SmallVector<mlir::scf::ForOp, 6> prefixBand;
  SmallVector<Value, 6> prefixTileValues;
  SmallVector<unsigned, 6> prefixTileSizesInt;
  SmallVector<Value, 6> leafTileValues;
  SmallVector<unsigned, 6> leafTileSizesInt;
  buildLeafBranchTileSlices(plan, band, tileSizeValues, tileSizesInt, bandSize, tileLevels, prefixBand,
                            prefixTileValues, prefixTileSizesInt, leafTileValues, leafTileSizesInt);

  int prefixParallelDim = -1;
  int64_t prefixParallelUseCore = 0;
  // The shared prefix excludes the actual leaf/vector axis, so its last axis is still
  // eligible for block-level parallel mapping.
  mlir::scf::ForOp prefixParallelMapLoop = markAndWrapParallelAxis(
    originalKernel, prefixBand, prefixTileSizesInt, tileLevels, builder, prefixParallelDim, prefixParallelUseCore,
    /*allowLastAxis=*/true);

  SmallVector<int64_t, 6> branchLeafNodeIds;
  if (failed(
        collectAndTagLeafBranchLoops(originalKernel, builder, bandIdx, plan, band, nextNodeId, branchLeafNodeIds))) {
    return failure();
  }

  if (failed(applyDecoupledAxisTiling(originalKernel, builder, "shared-prefix", prefixBand, prefixTileValues,
                                      prefixTileSizesInt, prefixParallelDim, prefixParallelUseCore,
                                      prefixParallelMapLoop, nextNodeId))) {
    return failure();
  }

  return applyLeafBandsByNodeId(originalKernel, builder, bandIdx, branchLeafNodeIds, leafTileValues, leafTileSizesInt);
}

static LogicalResult applyAllBandsWithPlans(func::FuncOp originalKernel, OpBuilder &builder,
                                            const std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                            const std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                            const std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                            const std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt) {
  if (bandsToUse.size() != leafBranchPlans.size()) {
    originalKernel.emitError("representative bands and leaf-branch plans mismatch");
    return failure();
  }

  int64_t nextNodeId = 0;
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &band = bandsToUse[bandIdx];
    const auto &tileSizeValues = allTileSizeValues[bandIdx];
    const auto &tileSizesInt = allBandTileSizesInt[bandIdx];
    const LeafBranchBandPlan &plan = leafBranchPlans[bandIdx];

    LogicalResult status = plan.hasLeafBranching
                             ? applySingleLeafBranchBandDecoupled(originalKernel, builder, bandIdx, plan, band,
                                                                  tileSizeValues, tileSizesInt, nextNodeId)
                             : applySingleLinearBandDecoupled(originalKernel, builder, bandIdx, band, tileSizeValues,
                                                              tileSizesInt, nextNodeId);
    if (failed(status)) {
      return failure();
    }

    clearTemporaryLoopIdentificationAttrs(originalKernel, /*clearPointLoopAttr=*/false);
  }

  return success();
}

static void runApplyPostProcessing(func::FuncOp originalKernel, OpBuilder &builder) {
  // Step 5: Post-processing - mark attributes.
  clearTemporaryLoopIdentificationAttrs(originalKernel, /*clearPointLoopAttr=*/false);
  inlineDeleteMarkedLoops(originalKernel, builder);
  sinkTransposePointLoops(originalKernel);
  sinkMultiVecPointLoops(originalKernel);
  markInnermostLoopsWithVectorAttr(originalKernel, builder);
  clearNotInnerDimensionBroadcastLoopAttr(originalKernel);
  clearTemporaryLoopIdentificationAttrs(originalKernel);
  setBlockDimAttribute(originalKernel, builder);
  clearValuelessBroadcastLoopAttr(originalKernel);
}

LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape) {
  auto loc = originalKernel.getLoc();

  std::vector<LeafBranchBandPlan> leafBranchPlans;
  bool shouldReturnEarly = false;
  if (failed(prepareLeafBranchPlansForApply(originalKernel, builder, leafBranchPlans, shouldReturnEarly))) {
    return failure();
  }
  if (shouldReturnEarly) {
    return success();
  }

  std::vector<SmallVector<mlir::scf::ForOp, 6>> bandsToUse;
  collectRepresentativeBands(leafBranchPlans, bandsToUse);

  if (!leafBranchPlans.empty()) {
    preprocessLoopAttrsForTileCalculation(originalKernel, leafBranchPlans.front());
  }
  [[maybe_unused]] auto clearNotInnerBroadcastGuard =
    llvm::make_scope_exit([&] { clearNotInnerDimensionBroadcastLoopAttr(originalKernel); });
  (void)clearNotInnerBroadcastGuard;

  // Setup builder.
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &originalKernel.getBody().front();
  builder.setInsertionPointToStart(body);

  std::vector<SmallVector<unsigned, 6>> allBandTileSizesInt;
  std::vector<SmallVector<int, 6>> allBandConstraintMaxs;
  std::vector<SmallVector<Value, 6>> allTileSizeValues;
  if (failed(prepareTileMetadataForApply(originalKernel, builder, loc, isStaticShape, bandsToUse, allBandTileSizesInt,
                                         allBandConstraintMaxs, allTileSizeValues))) {
    return failure();
  }

  if (failed(applyAllBandsWithPlans(originalKernel, builder, leafBranchPlans, bandsToUse, allTileSizeValues,
                                    allBandTileSizesInt))) {
    return failure();
  }

  runApplyPostProcessing(originalKernel, builder);
  return success();
}

static bool isAncestorOp(mlir::Operation *maybeAncestor, mlir::Operation *op) {
  for (mlir::Operation *p = op->getParentOp(); p != nullptr; p = p->getParentOp()) {
    if (p == maybeAncestor) {
      return true;
    }
  }
  return false;
}

static mlir::LogicalResult verifyBandIsNestedChain(llvm::ArrayRef<mlir::scf::ForOp> band) {
  for (size_t i = 0; i + 1 < band.size(); ++i) {
    if (!band[i] || !band[i + 1]) {
      return mlir::failure();
    }
    mlir::scf::ForOp loopI = band[i];
    mlir::scf::ForOp loopI1 = band[i + 1];
    if (!isAncestorOp(loopI.getOperation(), loopI1.getOperation())) {
      // not a chain from outer to inner
      return mlir::failure();
    }
  }
  return mlir::success();
}

static void collectChainComputeOps(llvm::ArrayRef<mlir::scf::ForOp> band,
                                   llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  ops.clear();
  if (band.empty()) {
    return;
  }

  // NOTE: We only collect ops from the innermost layer!
  // Middle layer ops (pre/post) are already handled by cloneNonPerfectChainIntoPointLoops.
  // If we collect them here, they will be cloned twice, causing SSA dominance errors.

  // innermost layer: collect all ops in its body except terminator
  mlir::scf::ForOp innerLoop = band.back();
  mlir::Block *innerBody = innerLoop.getBody();
  std::transform(innerBody->without_terminator().begin(), innerBody->without_terminator().end(),
                 std::back_inserter(ops), [](mlir::Operation &op) { return &op; });
}
static mlir::LogicalResult cloneComputeIntoInnermostPointLoop(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                              llvm::ArrayRef<mlir::scf::ForOp> tiledLoops,
                                                              unsigned tileSizesNum, mlir::scf::ForOp rootScfForOp,
                                                              mlir::OpBuilder &builder, mlir::IRMapping &mapping) {
  unsigned forNum = band.size();
  if (tiledLoops.size() < tileSizesNum + forNum) return mlir::failure();

  // innermost point loop: last point (idx = tileSizesNum + forNum - 1)
  mlir::scf::ForOp innermostPoint = tiledLoops[tileSizesNum + forNum - 1];
  if (!innermostPoint) return mlir::failure();

  // 1) collect the list of ops to clone
  llvm::SmallVector<mlir::Operation *, 32> computeOps;
  collectChainComputeOps(band, computeOps);

  // 2) establish mapping: original band each layer iv -> new point each layer iv
  //    (plus region iter args / results for loops with iter_args).
  initIVMapping(band, tiledLoops, tileSizesNum, mapping);

  // 3) set insertion point: before rootScfForOp (root is still in some block of innermostPoint)
  mlir::OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPoint(rootScfForOp.getOperation());

  // 4) clone in order (mapping will automatically replace the cloned value with the subsequent op)
  for (mlir::Operation *op : computeOps) {
    builder.clone(*op, mapping);
  }

  return mlir::success();
}

static void splitOpsAroundChildLoop(mlir::scf::ForOp parent, mlir::Operation *childLoopOp,
                                    llvm::SmallVectorImpl<mlir::Operation *> &preOps,
                                    llvm::SmallVectorImpl<mlir::Operation *> &postOps) {
  preOps.clear();
  postOps.clear();

  bool seenChild = false;
  for (mlir::Operation &op : parent.getBody()->without_terminator()) {
    if (&op == childLoopOp) {  // child loop dont clone
      seenChild = true;
      continue;
    }
    if (!seenChild) {
      preOps.push_back(&op);
    } else {
      postOps.push_back(&op);
    }
  }
}

static bool isReductionLoopWithIterArgs(mlir::scf::ForOp loop) {
  return loop && loop->hasAttr(kReductionLoopAttr) && loop.getNumResults() > 0;
}

static bool isLoopResultEscapingBand(mlir::scf::ForOp loop, mlir::scf::ForOp bandRoot) {
  if (!isReductionLoopWithIterArgs(loop) || !bandRoot) {
    return false;
  }
  for (Value res : loop.getResults()) {
    for (Operation *user : res.getUsers()) {
      // Escapes if any user is outside the band root region.
      if (!bandRoot->isAncestor(user)) {
        return true;
      }
    }
  }
  return false;
}

static unsigned getMiddleLevelLoopIndex(unsigned tileSizesNum, unsigned bandSize, unsigned dim) {
  if (bandSize == 0) {
    return tileSizesNum + dim;
  }
  unsigned tileNum = tileSizesNum / bandSize;
  if (tileNum <= 1) {
    return tileSizesNum + dim;
  }
  return (tileNum - 1) * bandSize + dim;
}

// Helper function to initialize the induction variable and region iter args mapping
static void initIVMapping(llvm::ArrayRef<mlir::scf::ForOp> band, llvm::ArrayRef<mlir::scf::ForOp> tiledLoops,
                          unsigned tileSizesNum, mlir::IRMapping &mapping) {
  for (unsigned i = 0; i < band.size(); ++i) {
    mlir::scf::ForOp loop = band[i];

    // Map IVs to point loops. Reduction loops keep their original bounds at point level.
    unsigned pointIdx = tileSizesNum + i;
    unsigned targetIdx = pointIdx;

    mlir::scf::ForOp tiledLoop = tiledLoops[targetIdx];
    if (!loop || !tiledLoop) {
      continue;
    }

    // Map IV
    Value origIV = loop.getInductionVar();
    Value tiledIV = tiledLoop.getInductionVar();
    if (!mapping.contains(origIV)) {
      mapping.map(origIV, tiledIV);
    }

    // Map region iter args (for loops with iter_args)
    for (auto [origArg, tiledArg] : llvm::zip(loop.getRegionIterArgs(), tiledLoop.getRegionIterArgs())) {
      if (!mapping.contains(origArg)) {
        mapping.map(origArg, tiledArg);
      }
    }

    // Map loop results to corresponding tiled loop results.
    unsigned resultIdx = tileSizesNum + i;
    if (resultIdx < tiledLoops.size()) {
      mlir::scf::ForOp resultLoop = tiledLoops[resultIdx];
      if (resultLoop) {
        for (auto [origRes, tiledRes] : llvm::zip(loop.getResults(), resultLoop.getResults())) {
          if (!mapping.contains(origRes)) {
            mapping.map(origRes, tiledRes);
          }
        }
      }
    }
  }
}

static void replaceLoopYield(mlir::scf::ForOp loop, llvm::ArrayRef<mlir::Value> newOperands, mlir::OpBuilder &builder) {
  auto *body = loop.getBody();
  if (!body) {
    return;
  }

  auto *terminator = body->getTerminator();
  if (!terminator) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  // CRITICAL: Erase terminator first, then set insertion point to end
  // Setting insertion point to a terminator that will be erased causes issues
  terminator->erase();
  builder.setInsertionPointToEnd(body);
  builder.create<mlir::scf::YieldOp>(loop.getLoc(), newOperands);
}

static mlir::scf::ForOp getUniqueChildLoop(mlir::scf::ForOp loop) {
  mlir::scf::ForOp childLoop;
  for (Operation &op : loop.getBody()->without_terminator()) {
    auto childFor = dyn_cast<mlir::scf::ForOp>(op);
    if (!childFor) {
      continue;
    }
    if (childLoop) {
      return mlir::scf::ForOp();
    }
    childLoop = childFor;
  }
  return childLoop;
}

static bool isTransposePointLoop(mlir::scf::ForOp loop) {
  return loop && loop->hasAttr(kInnerLoopAttr) && loop->hasAttr(kTransposeLoopAttr);
}

static bool isMultiVecPointLoop(mlir::scf::ForOp loop) {
  return loop && loop->hasAttr(kInnerLoopAttr) && loop->hasAttr(kMultiVecLoopAttr);
}

// Drop the chain-marker attribute (e.g. `kTransposeLoopAttr` or
// `kMultiVecLoopAttr`) on `loop` and any nested point loops also part of the
// chain. Used to abort the sink for one chain when an op can't be safely
// hoisted across a downstream point loop.
static void clearPointLoopChain(mlir::scf::ForOp loop, llvm::StringRef chainAttr,
                                llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop) {
  for (mlir::scf::ForOp cur = loop; cur && isPointLoop(cur);) {
    mlir::scf::ForOp parentLoop = cur;
    cur->removeAttr(chainAttr);
    cur = mlir::scf::ForOp();
    for (mlir::scf::ForOp innerLoop = getUniqueChildLoop(parentLoop); innerLoop;
         innerLoop = getUniqueChildLoop(innerLoop)) {
      if (isPointLoop(innerLoop)) {
        cur = innerLoop;
        break;
      }
    }
  }
}

// Walk down `pointLoop`'s unique-child chain to locate the nearest matching
// inner point loop. If their bodies allow it (no iter_args / no side-effecting
// non-load ops in between), hoist the independent ops up past `pointLoop` and
// re-nest so `pointLoop` and the next point loop sit back-to-back. Returns
// `true` when one sink step actually happened, `false` otherwise.
static bool sinkPointLoopOnce(mlir::scf::ForOp pointLoop, llvm::StringRef chainAttr,
                              llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop) {
  mlir::scf::ForOp nextPointLoop;
  mlir::scf::ForOp nextPointParent;
  for (mlir::scf::ForOp loop = getUniqueChildLoop(pointLoop); loop; loop = getUniqueChildLoop(loop)) {
    if (isPointLoop(loop)) {
      nextPointLoop = loop;
      break;
    }
    nextPointParent = loop;
  }
  if (!nextPointLoop || !nextPointParent) {
    return false;
  }

  for (mlir::scf::ForOp loop : {pointLoop, nextPointParent, nextPointLoop}) {
    if (loop.getNumResults() != 0 || !loop.getRegionIterArgs().empty()) {
      return false;
    }
  }

  SmallVector<Operation *, 8> hoistOps;
  llvm::SmallDenseSet<Value, 8> dependentValues;
  dependentValues.insert(pointLoop.getInductionVar());
  for (Operation &op : pointLoop.getBody()->without_terminator()) {
    bool dependsOnPointLoop =
      llvm::any_of(op.getOperands(), [&](Value operand) { return dependentValues.contains(operand); });
    if (!dependsOnPointLoop) {
      hoistOps.push_back(&op);
      continue;
    }

    if (op.getNumRegions() != 0 || (!isa<memref::LoadOp>(op) && !mlir::isMemoryEffectFree(&op))) {
      clearPointLoopChain(pointLoop, chainAttr, isPointLoop);
      return false;
    }

    for (Value result : op.getResults()) {
      dependentValues.insert(result);
    }
  }
  if (hoistOps.empty()) {
    return false;
  }

  Operation *insertBefore = pointLoop.getOperation();
  for (Operation *op : hoistOps) {
    op->moveBefore(insertBefore);
  }

  nextPointLoop->moveBefore(pointLoop.getBody()->getTerminator());
  pointLoop->moveBefore(nextPointParent.getBody()->getTerminator());
  return true;
}

// Generic point-loop sink driver. `chainAttr` is the marker that identifies a
// chain (`kTransposeLoopAttr` for transpose, `kMultiVecLoopAttr` for
// multi-dim vectorization). `isPointLoop` recognises members of the chain.
static void sinkPointLoopsByPredicate(func::FuncOp funcOp, llvm::StringRef chainAttr,
                                      llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop) {
  SmallVector<mlir::scf::ForOp, 8> pointLoops;
  funcOp.walk([&](mlir::scf::ForOp loop) {
    if (isPointLoop(loop)) {
      pointLoops.push_back(loop);
    }
  });

  for (mlir::scf::ForOp pointLoop : pointLoops) {
    while (isPointLoop(pointLoop) && sinkPointLoopOnce(pointLoop, chainAttr, isPointLoop)) {
    }
  }
}

static void sinkTransposePointLoops(func::FuncOp funcOp) {
  sinkPointLoopsByPredicate(funcOp, kTransposeLoopAttr, isTransposePointLoop);
}

static void sinkMultiVecPointLoops(func::FuncOp funcOp) {
  sinkPointLoopsByPredicate(funcOp, kMultiVecLoopAttr, isMultiVecPointLoop);
}

static void updatePointLoopYieldsFromOriginalLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                   llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                   unsigned tileSizesNum, mlir::IRMapping &mapping,
                                                   mlir::OpBuilder &builder) {
  for (unsigned i = 0; i < band.size(); ++i) {
    mlir::scf::ForOp origLoop = band[i];
    if (!origLoop || origLoop.getNumResults() == 0) {
      continue;
    }

    // Update the point loop's yield; wrapper loops will forward results outward.
    unsigned targetIdx = tileSizesNum + i;

    mlir::scf::ForOp targetLoop = tiledLoops[targetIdx];
    auto *origBody = origLoop.getBody();

    if (!origBody || !targetLoop) {
      continue;
    }

    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(origBody->getTerminator());
    if (!yieldOp) {
      continue;
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(yieldOp.getNumOperands());
    std::transform(yieldOp.getOperands().begin(), yieldOp.getOperands().end(), std::back_inserter(newOperands),
                   [&mapping](Value operand) {
                     return (operand && mapping.contains(operand)) ? mapping.lookup(operand) : operand;
                   });

    if (newOperands.size() != targetLoop.getNumResults()) {
      continue;
    }

    replaceLoopYield(targetLoop, newOperands, builder);
  }
}

static void forwardIterArgsToChildLoop(mlir::scf::ForOp loop, mlir::OpBuilder &builder) {
  if (!loop || loop.getNumResults() == 0) {
    return;
  }

  mlir::scf::ForOp childLoop;
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
      if (childLoop) {
        return;  // More than one child loop.
      }
      childLoop = forOp;
      continue;
    }
    return;  // Non-loop op present, skip.
  }

  if (!childLoop || childLoop.getNumResults() != loop.getNumResults()) {
    return;
  }

  // Convert ResultRange to SmallVector<Value> for replaceLoopYield
  SmallVector<Value> childResults(childLoop.getResults().begin(), childLoop.getResults().end());
  replaceLoopYield(loop, childResults, builder);
}

static void forwardIterArgsThroughWrapperLoops(llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                               mlir::OpBuilder &builder) {
  for (mlir::scf::ForOp loop : tiledLoops) {
    forwardIterArgsToChildLoop(loop, builder);
  }
}

static LogicalResult collectReduceResultUserOps(mlir::scf::ForOp rootLoop, mlir::scf::ForOp userAnchorLoop,
                                                llvm::SmallSet<Operation *, 16> &opSet) {
  Block *sourceBlock = userAnchorLoop->getBlock();
  SmallVector<Value, 8> worklist(rootLoop.getResults().begin(), rootLoop.getResults().end());

  for (size_t i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];
    for (Operation *user : v.getUsers()) {
      if (user == rootLoop.getOperation()) {
        continue;
      }
      if (user->getBlock() != sourceBlock || user->isBeforeInBlock(userAnchorLoop) || isa<func::ReturnOp>(user)) {
        return failure();
      }
      if (opSet.insert(user).second) {
        worklist.append(user->result_begin(), user->result_end());
      }
    }
  }

  return success();
}

static void collectOpsInOrderAfter(Operation *startOp, const llvm::SmallSet<Operation *, 16> &opSet,
                                   SmallVectorImpl<Operation *> &opsInOrder) {
  for (Operation *op = startOp->getNextNode(); op != nullptr; op = op->getNextNode()) {
    if (opSet.count(op)) {
      opsInOrder.push_back(op);
    }
  }
}

static void moveOpsAfterLoop(mlir::scf::ForOp destLoop, ArrayRef<Operation *> ops, OpBuilder &builder) {
  Block *destBlock = destLoop->getBlock();

  OpBuilder::InsertionGuard guard(builder);
  auto insertIt = std::next(destLoop->getIterator());
  for (Operation *op : ops) {
    op->moveBefore(destBlock, insertIt);
    insertIt = std::next(op->getIterator());
  }
}

static void replaceUsesInOps(ValueRange oldResults, ValueRange newResults, ArrayRef<Operation *> ops) {
  for (auto [oldRes, newRes] : llvm::zip(oldResults, newResults)) {
    for (Operation *op : ops) {
      op->replaceUsesOfWith(oldRes, newRes);
    }
  }
}

static bool isTriviallyRecreatableConst(Operation *op) { return isa<arith::ConstantOp, arith::ConstantIndexOp>(op); }

static LogicalResult ensureNoExternalUsers(const llvm::SmallSet<Operation *, 16> &opSet) {
  for (Operation *op : opSet) {
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (!opSet.count(user)) {
          llvm::errs() << "Sink failed. External user found: " << *user << "\n";
          return failure();
        }
      }
    }
  }
  return success();
}

// Collect operand dependencies for ops to be moved. For constant-like defs after the outer loop,
// mark them for recreation instead of moving.
static LogicalResult collectOperandClosure(Operation *outerLoopOp, llvm::SmallSet<Operation *, 16> &opSet,
                                           llvm::SmallSet<Operation *, 16> &recreateOps,
                                           const llvm::DenseSet<Value> &ignoreValues) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *, 16> currentOps(opSet.begin(), opSet.end());
    for (Operation *op : currentOps) {
      for (Value operand : op->getOperands()) {
        if (ignoreValues.contains(operand)) {
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (!defOp) {
          continue;  // Block argument.
        }
        if (defOp->getBlock() != outerLoopOp->getBlock()) {
          return failure();
        }
        if (defOp->isBeforeInBlock(outerLoopOp)) {
          continue;  // Dominates moved ops from outer scope.
        }
        if (isTriviallyRecreatableConst(defOp)) {
          recreateOps.insert(defOp);
          continue;
        }
        if (opSet.insert(defOp).second) {
          changed = true;
        }
      }
    }
  }

  // Ensure moved ops won't leave behind users outside the move set.
  return ensureNoExternalUsers(opSet);
}

static void recreateConstantsForMovedOps(ArrayRef<Operation *> movedOps,
                                         const llvm::SmallSet<Operation *, 16> &recreateOps, mlir::OpBuilder &builder) {
  if (movedOps.empty() || recreateOps.empty()) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(movedOps.front());

  DenseMap<Value, Value> constMap;
  auto getRecreated = [&](Value v) -> Value {
    auto it = constMap.find(v);
    if (it != constMap.end()) {
      return it->second;
    }
    Value recreated = recreateConstantOrSelf(v, builder);
    constMap[v] = recreated;
    return recreated;
  };

  for (Operation *op : movedOps) {
    for (OpOperand &operand : op->getOpOperands()) {
      Value v = operand.get();
      Operation *defOp = v.getDefiningOp();
      if (!defOp || !recreateOps.count(defOp)) {
        continue;
      }
      operand.set(getRecreated(v));
    }
  }
}

static LogicalResult sinkReduceLoopResultsToMiddleLevel(mlir::scf::ForOp rootLoop, mlir::scf::ForOp userAnchorLoop,
                                                        llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                        unsigned tileSizesNum, unsigned bandSize,
                                                        mlir::OpBuilder &builder) {
  if (!isReductionLoopWithIterArgs(rootLoop)) {
    return success();
  }
  if (tiledLoops.empty()) {
    return failure();
  }

  if (!userAnchorLoop) {
    return failure();
  }

  unsigned middleIdx = getMiddleLevelLoopIndex(tileSizesNum, bandSize, 0);
  if (middleIdx >= tiledLoops.size() || !tiledLoops[middleIdx]) {
    return failure();
  }

  mlir::scf::ForOp middleLoop = tiledLoops[middleIdx];
  auto rootResults = rootLoop.getResults();
  if (rootResults.empty()) {
    return success();
  }
  if (rootResults.size() != middleLoop.getNumResults()) {
    return failure();
  }

  llvm::SmallSet<Operation *, 16> opSet;
  if (failed(collectReduceResultUserOps(rootLoop, userAnchorLoop, opSet))) {
    return failure();
  }

  if (opSet.empty()) {
    return success();
  }

  llvm::SmallSet<Operation *, 16> recreateOps;
  llvm::DenseSet<Value> ignoreValues;
  ignoreValues.insert(rootResults.begin(), rootResults.end());
  if (failed(collectOperandClosure(userAnchorLoop.getOperation(), opSet, recreateOps, ignoreValues))) {
    return failure();
  }

  SmallVector<Operation *, 16> opsInOrder;
  collectOpsInOrderAfter(userAnchorLoop.getOperation(), opSet, opsInOrder);
  moveOpsAfterLoop(middleLoop, opsInOrder, builder);
  recreateConstantsForMovedOps(opsInOrder, recreateOps, builder);
  replaceUsesInOps(rootResults, middleLoop.getResults(), opsInOrder);

  return rootLoop.use_empty() ? success() : failure();
}

// Helper function to clone ops to point loop
static void cloneOpsToPointLoop(mlir::scf::ForOp pointLoop, llvm::ArrayRef<mlir::Operation *> ops, bool insertAtStart,
                                mlir::OpBuilder &builder, mlir::IRMapping &mapping) {
  mlir::OpBuilder::InsertionGuard g(builder);

  if (insertAtStart) {
    builder.setInsertionPointToStart(pointLoop.getBody());
  } else {
    // If terminator exists, insert before it; otherwise insert at end
    auto *terminator = pointLoop.getBody()->getTerminator();
    if (terminator) {
      builder.setInsertionPoint(terminator);
    } else {
      builder.setInsertionPointToEnd(pointLoop.getBody());
    }
  }

  for (mlir::Operation *op : ops) {
    builder.clone(*op, mapping);
  }
}

// Helper function to clone non-perfect chain into point loops
static mlir::LogicalResult cloneNonPerfectChainIntoPointLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                              llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                              unsigned tileSizesNum, mlir::OpBuilder &builder,
                                                              mlir::IRMapping &mapping) {
  unsigned forNum = band.size();
  if (forNum == 0) {
    return mlir::success();
  }

  if (tiledLoops.size() < tileSizesNum + forNum) {
    return mlir::failure();
  }

  initIVMapping(band, tiledLoops, tileSizesNum, mapping);

  for (unsigned i = 0; i + 1 < forNum; ++i) {
    llvm::SmallVector<mlir::Operation *, 16> pre, post;
    mlir::scf::ForOp parentLoop = band[i];
    mlir::scf::ForOp childLoop = band[i + 1];
    splitOpsAroundChildLoop(parentLoop, childLoop.getOperation(), pre, post);

    // put into the parent loop of tile level 0 (index=i) by default
    mlir::scf::ForOp parentTile0 = tiledLoops[tileSizesNum + i];
    if (!parentTile0) {
      return mlir::failure();
    }

    // pre: put into the beginning of the body (before the child loop op)
    if (!pre.empty()) {
      cloneOpsToPointLoop(parentTile0, pre, /*insertAtStart=*/true, builder, mapping);
    }

    // post: put into the end of the body (before the yield)
    if (!post.empty()) {
      cloneOpsToPointLoop(parentTile0, post, /*insertAtStart=*/false, builder, mapping);
    }
  }

  return mlir::success();
}

}  // namespace autotiling
}  // namespace mlir
