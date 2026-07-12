/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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
#include <cmath>
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
#include "akg/Utils/GlobalVars.hpp"
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
#include "akg/Utils/Constants.h"

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
using mlir::hacc::HACCFuncType;
using mlir::hacc::HACCFuncTypeAttr;
using mlir::hacc::KernelArgType;
using mlir::hacc::KernelArgTypeAttr;

namespace mlir {
namespace autotiling {

static constexpr const char *kTreeNodeIdAttr = "node_id";
static constexpr const char *kTreeLeafAttr = "leaf";
static constexpr const char *kInnerLoopAttr = "inner";
static constexpr const char *kDeleteLoopAttr = "delete";
static constexpr const char *kNotInnerDimensionBroadcastLoopAttr = "not_inner_dimension_broadcast";
static constexpr const char *kParallelAxisAttr = "parallel__axis";
static constexpr int64_t kDefaultNpuCoreNum = 48;
static constexpr int64_t kDefaultTilingKey = 0;
static constexpr int64_t kTwoDimDynamicVectorTilingKey = 1;
static constexpr unsigned kTilingFuncReservedArgCount = 2;
static constexpr unsigned kTilingDataArgOffset = 1;

// Main tiling functions
struct CreateTilingFuncParams {
  mutable func::FuncOp originalKernel;
  OpBuilder &builder;
  bool isStaticShape;
  int64_t tilingKey;
  TilingMetadata *metadata;
};

static LogicalResult createTilingFuncDefault(const CreateTilingFuncParams &params, func::FuncOp &tilingFunc);
struct BandTilingOutput {
  std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> &bandsToUse;
  std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> &allBandTileSizes;
  std::vector<SmallVector<int, kSmallVectorSizeSix>> &allBandConstraintMaxs;
};

static LogicalResult calculateTileSizesForBands(func::FuncOp funcOp, bool useAutoTiling, BandTilingOutput output,
                                                size_t &levelToTile);
struct ApplyTilingParams {
  mutable mlir::scf::ForOp loop;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  OpBuilder &builder;
  std::map<int64_t, Value> &constantCache;
  mutable mlir::scf::ForOp parallelMapLoop;
  mutable mlir::scf::ForOp parallelTileLoop;
  Value parallelTileCoord;
  bool useRuntimeTileCounts;
};

static LogicalResult applyTilingToLoop(const ApplyTilingParams &params);

struct LeafBranchBandPlan {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> representativeBand;
  SmallVector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>, kSmallVectorSizeFour> branchBands;
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

// Bundled parameters for processSingleTileLoop to stay under readability-function-size_parameters limit.
struct TileLoopParams {
  int i;
  int j;
  int bandSize;
  int tileNum;
  MutableArrayRef<mlir::scf::ForOp> newLoops;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
};

// Bundled state for one band's tile rewrite traversal.
// Lives only on the stack and references caller-owned containers.
struct TileRewriteContext {
  const llvm::DenseSet<mlir::Operation *> &escapeReduceLoops;
  mlir::scf::ForOp parallelMapLoop;
  mlir::scf::ForOp parallelTileLoop;
  Value parallelTileCoord;
  int64_t dynamicTileCoreNum = kDefaultNpuCoreNum;
  bool useRuntimeFirstLevelTileCount = false;
  bool dropMappedOutermostFirstLevelIterArgs = false;
};

struct TileLoopCtx {
  std::map<int64_t, Value> &constantCache;
  const TileRewriteContext &ctx;
};

// Loop tiling core helpers
// Wrapper-level no-split kind. Wrapper levels (FirstLevel/Middle) collapse the loop to a
// single iteration; the point level keeps the original range.
enum class NoSplitKind { FirstLevel, Middle, Point };

// Bundled build context for createPointLoopBounds (no kernel needed).
struct BuildContext {
  mlir::Location loc;
  OpBuilder &builder;
  std::map<int64_t, Value> &constantCache;
};

// Bundled params for buildNoSplitBounds to stay under readability-function-size_parameters limit.
struct NoSplitBoundsParams {
  mutable mlir::scf::ForOp origLoop;
  mutable mlir::scf::ForOp prevLoop;
  NoSplitKind kind;
  bool dropReductionInits;
};

// Bundled params for createFirstLevelTileLoopBounds to stay under readability-function-size_parameters limit.
struct FirstLevelTileBoundsParams {
  mutable mlir::scf::ForOp origLoop;
  Value tilesize;
  unsigned tilesizeInt;
  bool dropFirstLevelReductionIterArgs;
  int64_t dynamicTileCoreNum;
  bool useRuntimeTileCount;
};

// Bundled params for createMiddleLevelTileLoopBounds to stay under readability-function-size_parameters limit.
struct MiddleLevelTileBoundsParams {
  mutable mlir::scf::ForOp origLoop;
  Value curTilesize;
  Value prevTilesize;
  mutable mlir::scf::ForOp prevLoop;
  bool escapeReduceIterArgs;
};

static LoopBounds buildNoSplitBounds(const BuildContext &bc, const NoSplitBoundsParams &params);
static LoopBounds createFirstLevelTileLoopBounds(const BuildContext &bc, const FirstLevelTileBoundsParams &params);
static LoopBounds createMiddleLevelTileLoopBounds(const BuildContext &bc, const MiddleLevelTileBoundsParams &params);

static LoopBounds createPointLoopBounds(const BuildContext &bc, mlir::scf::ForOp origLoop,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop);
static mlir::scf::ForOp replaceLoopWithNewBounds(mlir::scf::ForOp oldLoop, const LoopBounds &bounds, mlir::Location loc,
                                                 OpBuilder &builder);
static void processSingleTileLoop(
  const TileLoopParams &p, SmallVectorImpl<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>> &tileLevelInfo,
  mlir::Location loc, const TileLoopCtx &tc);
// Bundled loop data for constructTiledIndexStatic to stay under readability-function-size_parameters limit.
struct TileLoopData {
  MutableArrayRef<mlir::scf::ForOp> newLoops;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
};

static void constructTiledIndexStatic(const TileLoopData &data, OpBuilder &builder,
                                      std::map<int64_t, Value> &constantCache, const TileRewriteContext &ctx);

// Dynamic axis mapping and tile size helpers
static std::vector<DynamicAxisMapping> buildDynamicAxisMappingForBand(ArrayRef<mlir::scf::ForOp> band,
                                                                      ArrayRef<unsigned> bandTileSizes,
                                                                      func::FuncOp originalKernel);
static void initializeEmptyMultiVecMasks(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                         TilingMetadata &metadata);
static void captureMultiVecMasks(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                 TilingMetadata &metadata);
static bool buildTwoDimDynamicVectorMetadata(func::FuncOp originalKernel,
                                             ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bandsToUse,
                                             const TilingMetadata &baseMetadata, TilingMetadata &metadata);
struct DynamicTilePerDimParams {
  Location loc;
  OpBuilder &builder;
  Value dim;
  int constraintMax;
  int64_t dynamicTileCoreNum;
};

static Value emitDynamicTilePerDim(const DynamicTilePerDimParams &params);
struct KernelBuildContext {
  mutable func::FuncOp originalKernel;
  mlir::Location loc;
  OpBuilder &builder;
  std::map<int64_t, Value> &constantCache;
};
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 const KernelBuildContext &kbc, Value &result);
static LogicalResult prepareTileSizesForStaticShape(
  const KernelBuildContext &kbc, const BandTilingOutput &bandData,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues);
// Bundled params for prepareTileSizesFromMemref to stay under readability-function-size_parameters limit.
struct PrepareTileSizesFromMemrefParams {
  mutable func::FuncOp originalKernel;
  ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands;
  mlir::Location loc;
  OpBuilder &builder;
  ArrayRef<SmallVector<unsigned, kSmallVectorSizeSix>> allBandTileSizesInt;
};

static LogicalResult prepareTileSizesFromMemref(
  const PrepareTileSizesFromMemrefParams &params,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues);
static LogicalResult prepareLeafBranchPlansForApply(func::FuncOp originalKernel, OpBuilder &builder,
                                                    std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                                    bool isStaticShape, bool &shouldReturnEarly);
// Bundled params for prepareTileMetadataForApply to stay under readability-function-size_parameters limit.
struct PrepareTileMetadataParams {
  mutable func::FuncOp originalKernel;
  bool isStaticShape;
  BandTilingOutput bandOutput;
  const TilingMetadata *metadata;
};

static LogicalResult prepareTileMetadataForApply(
  const PrepareTileMetadataParams &params, const BuildContext &bc,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues);
// Bundled params for applySingleLinearBandDecoupled to stay under readability-function-size_parameters limit.
struct SingleLinearBandParams {
  mutable func::FuncOp originalKernel;
  OpBuilder &builder;
  size_t bandIdx;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  bool useRuntimeTileCounts;
};

static LogicalResult applySingleLinearBandDecoupled(const SingleLinearBandParams &params, int64_t &nextNodeId);
// Bundled params for applySingleLeafBranchBandDecoupled to stay under readability-function-size_parameters limit.
struct SingleLeafBranchBandParams {
  mutable func::FuncOp originalKernel;
  OpBuilder &builder;
  size_t bandIdx;
  const LeafBranchBandPlan &plan;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  bool useRuntimeTileCounts;
};

static LogicalResult applySingleLeafBranchBandDecoupled(const SingleLeafBranchBandParams &params, int64_t &nextNodeId);
// Bundled params for buildLeafBranchTileSlices to stay under readability-function-size_parameters limit.
struct LeafBranchTileSlicesInput {
  const LeafBranchBandPlan &plan;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  unsigned bandSize;
  unsigned tileLevels;
};
struct LeafBranchTileSlicesOutput {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> &prefixBand;
  SmallVector<Value, kSmallVectorSizeSix> &prefixTileValues;
  SmallVector<unsigned, kSmallVectorSizeSix> &prefixTileSizesInt;
};

static void buildLeafBranchTileSlices(const LeafBranchTileSlicesInput &in, const LeafBranchTileSlicesOutput &out);
// Bundled params for collectAndTagLeafBranchLoops to stay under readability-function-size_parameters limit.
struct CollectTagLeafBranchParams {
  mutable func::FuncOp originalKernel;
  OpBuilder &builder;
  size_t bandIdx;
  const LeafBranchBandPlan &plan;
  ArrayRef<mlir::scf::ForOp> band;
};

static LogicalResult collectAndTagLeafBranchLoops(const CollectTagLeafBranchParams &params, int64_t &nextNodeId,
                                                  SmallVector<int64_t, kSmallVectorSizeSix> &branchRootNodeIds);
// Bundled params for applyLeafBandsByNodeId to stay under readability-function-size_parameters limit.
struct ApplyLeafBandsParams {
  mutable func::FuncOp originalKernel;
  OpBuilder &builder;
  size_t bandIdx;
  const LeafBranchBandPlan &plan;
  ArrayRef<int64_t> branchRootNodeIds;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  unsigned bandSize;
  unsigned tileLevels;
  bool useRuntimeTileCounts;
  int64_t &nextNodeId;
};

static LogicalResult applyLeafBandsByNodeId(const ApplyLeafBandsParams &params);
// Bundled band apply data for applyAllBandsWithPlans.
struct BandApplyData {
  const std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> &bandsToUse;
  const std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues;
  const std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> &allBandTileSizesInt;
};

static LogicalResult applyAllBandsWithPlans(func::FuncOp originalKernel, OpBuilder &builder,
                                            const std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                            const BandApplyData &bandData, bool useRuntimeTileCounts);
static void runApplyPostProcessing(func::FuncOp originalKernel, OpBuilder &builder);

// Tiling function creation helpers
// Bundled params for buildTilingFunctionSignature to stay under readability-function-size_parameters limit.
struct TilingSignatureParams {
  FunctionType origTy;
  MLIRContext *ctx;
  OpBuilder &builder;
  int64_t tilingStructMemrefSize;
};

static void buildTilingFunctionSignature(const TilingSignatureParams &params, SmallVector<Type> &argTypes,
                                         SmallVector<Type> &resTypes);
static func::FuncOp createAndInitTilingFunc(func::FuncOp originalKernel, ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resTypes, OpBuilder &builder, int64_t tilingKey);
struct BandTilingData {
  ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands;
  ArrayRef<SmallVector<unsigned, kSmallVectorSizeSix>> allBandTileSizes;
  ArrayRef<SmallVector<int, kSmallVectorSizeSix>> allBandConstraintMaxs;
  ArrayRef<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
};
static LogicalResult storeTileSizesToMemref(func::FuncOp tilingFunc, func::FuncOp originalKernel,
                                            const BandTilingData &bandData, OpBuilder &builder);
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
static Value recreateConstantOrSelf(Value v, OpBuilder &builder);
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
static bool isMultiVecPointLoop(mlir::scf::ForOp loop);
static void sinkMultiVecPointLoops(func::FuncOp funcOp);
static void forwardIterArgsThroughWrapperLoops(llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                               mlir::OpBuilder &builder);
static LogicalResult collectReduceResultUserOps(mlir::scf::ForOp rootLoop, mlir::scf::ForOp userAnchorLoop,
                                                llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet);
static void collectOpsInOrderAfter(Operation *startOp,
                                   const llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet,
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
  if ((op == nullptr) || !visited.insert(op).second) {
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
    return {};
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
    llvm::dbgs() << "cloneUpperBoundDefinition: non-entry block argument\n";
    return {};
  }

  auto *def = upperBound.getDefiningOp();
  if (def == nullptr) {
    llvm::dbgs() << "cloneUpperBoundDefinition: no defining op\n";
    return {};
  }

  llvm::SmallVector<mlir::Operation *, kSmallVectorSizeSixteen> ops;
  llvm::SmallPtrSet<mlir::Operation *, kSmallVectorSizeThirtyTwo> visited;
  getOperandsTree(def, ops, visited);

  auto isSupportedOp = [](mlir::Operation *op) -> bool {
    if (!op || op->getNumRegions() != 0) {
      return false;
    }
    return isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp, arith::IndexCastOp,
               arith::IndexCastUIOp, arith::CmpIOp, arith::SelectOp, arith::AddIOp, arith::SubIOp, arith::MulIOp,
               arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
               arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, memref::DimOp, affine::AffineApplyOp,
               affine::AffineMinOp>(op);
  };

  for (auto *op : ops) {
    if (!isSupportedOp(op)) {
      llvm::dbgs() << "cloneUpperBoundDefinition: unsupported op " << op->getName() << "\n";
      return {};
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
        llvm::dbgs() << "cloneUpperBoundDefinition: non-entry block argument in chain\n";
        return {};
      }
      llvm::dbgs() << "cloneUpperBoundDefinition: operand not mapped for op " << op->getName() << "\n";
      return {};
    }

    Operation *cloned = builder.clone(*op, mapping);
    mapping.map(op, cloned);
  }

  if (auto mapped = mapping.lookupOrNull(upperBound)) {
    return mapped;
  }
  llvm::dbgs() << "cloneUpperBoundDefinition: upper bound not mapped\n";
  return {};
}

static bool isNoSplitLoop(mlir::scf::ForOp loop) {
  return loop && (loop->hasAttr(kReductionLoopAttr) || loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr));
}

static bool isReductionLoop(mlir::scf::ForOp loop) { return loop && loop->hasAttr(kReductionLoopAttr); }

[[maybe_unused]] static bool isInnerDimensionBroadcastLoop(mlir::scf::ForOp loop) {
  return loop
    .walk([&loop](memref::StoreOp store) -> WalkResult {
      if (store.getIndices().empty() || !CommonUtils::valueDependsOnLoopIV(store.getIndices().back(), loop)) {
        return WalkResult::advance();
      }

      llvm::SmallVector<Value, kSmallVectorSizeEight> worklist{store.getValueToStore()};
      llvm::DenseSet<Value> visited;
      while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!visited.insert(value).second) {
          continue;
        }

        if (auto load = value.getDefiningOp<memref::LoadOp>()) {
          if (llvm::none_of(load.getIndices(),
                            [&loop](Value idx) { return CommonUtils::valueDependsOnLoopIV(idx, loop); })) {
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
  funcOp.walk([&hasOuterInnerDimensionBroadcastLoop, &builder](mlir::scf::ForOp loop) {
    loop->removeAttr(kBroadcastLoopAttr);
    loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
  });
}

static void clearNotInnerDimensionBroadcastLoopAttr(func::FuncOp funcOp) {
  funcOp.walk([](mlir::scf::ForOp loop) {
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
    const bool hasNonLeafBranch =
      plan.hasLeafBranching &&
      llvm::any_of(plan.branchBands, [](const auto &branchBand) { return branchBand.size() != 1; });
    if (hasNonLeafBranch) {
      funcOp.walk([](mlir::scf::ForOp loop) { loop->removeAttr(kTransposeLoopAttr); });
      return;
    }
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
  if (!value || (value.getDefiningOp() == nullptr)) {
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

static int64_t ceilDivPositive(int64_t lhs, int64_t rhs) { return (lhs <= 0 || rhs <= 0) ? 1 : (lhs + rhs - 1) / rhs; }

static int64_t getNpuCoreNum(func::FuncOp funcOp) {
  if (!funcOp) {
    return kDefaultNpuCoreNum;
  }
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

static bool getStaticParallelTileWork(mlir::scf::ForOp loop, unsigned tileSize0, int64_t &parallelWork) {
  if (!loop || tileSize0 == 0 || tileSize0 == static_cast<unsigned>(-1)) {
    return false;
  }
  int64_t tripCount = 0;
  if (!getStaticTripCount(loop, tripCount)) {
    return false;
  }
  parallelWork = ceilDivPositive(tripCount, static_cast<int64_t>(tileSize0));
  return parallelWork > 0;
}

static bool isDynamicParallelCandidate(mlir::scf::ForOp loop, unsigned tileSize0) {
  if (!loop || tileSize0 == 0) {
    return false;
  }
  int64_t tripCount = 0;
  return tileSize0 == static_cast<unsigned>(-1) || !getStaticTripCount(loop, tripCount);
}

static bool isParallelCandidate(mlir::scf::ForOp loop, unsigned tileSize0, int64_t &parallelWork,
                                bool &isDynamicCandidate) {
  isDynamicCandidate = false;
  if (!loop || loop->hasAttr(kReductionLoopAttr) || loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr)) {
    return false;
  }
  if (isDynamicParallelCandidate(loop, tileSize0)) {
    isDynamicCandidate = true;
    parallelWork = 0;
    return true;
  }
  return getStaticParallelTileWork(loop, tileSize0, parallelWork);
}

static bool isStrictPerfectLoopEdge(mlir::scf::ForOp parent, mlir::scf::ForOp child) {
  if (!parent || !child) {
    return false;
  }
  bool sawChild = false;
  for (Operation &op : parent.getBody()->without_terminator()) {
    if (&op == child.getOperation()) {
      if (sawChild) {
        return false;
      }
      sawChild = true;
      continue;
    }
    return false;
  }
  return sawChild;
}

static void collectParallelPrefixDims(ArrayRef<mlir::scf::ForOp> band, ArrayRef<unsigned> tileSizesInt,
                                      SmallVectorImpl<unsigned> &parallelDims) {
  parallelDims.clear();
  if (band.empty() || tileSizesInt.size() < band.size()) {
    return;
  }

  for (unsigned dim = 0; dim < band.size(); ++dim) {
    int64_t parallelWork = 0;
    bool isDynamicCandidate = false;
    if (isReductionLoop(band[dim])) {
      break;
    }
    if (!isParallelCandidate(band[dim], tileSizesInt[dim], parallelWork, isDynamicCandidate)) {
      break;
    }
    bool hasNextPerfect = dim + 1 < band.size() && isStrictPerfectLoopEdge(band[dim], band[dim + 1]);
    if (!CommonUtils::loopIvFeedsIfCondition(band[dim])) {
      parallelDims.push_back(dim);
    }
    if (!hasNextPerfect) {
      break;
    }
  }
}

static Value emitLoopTripCountInCurrentFunc(Location loc, OpBuilder &builder, mlir::scf::ForOp loop) {
  int64_t staticTripCount = 0;
  if (getStaticTripCount(loop, staticTripCount)) {
    return builder.create<arith::ConstantIndexOp>(loc, staticTripCount);
  }

  Value extent = builder.create<arith::SubIOp>(loc, recreateConstantOrSelf(loop.getUpperBound(), builder),
                                               recreateConstantOrSelf(loop.getLowerBound(), builder));
  return builder.create<arith::CeilDivSIOp>(loc, extent, recreateConstantOrSelf(loop.getStep(), builder));
}

// Bundled params for emitLoopTripCountInTilingFunc to stay under readability-function-size_parameters limit.
struct TripCountEmitParams {
  mutable func::FuncOp tilingFunc;
  mutable func::FuncOp originalKernel;
  mutable mlir::scf::ForOp loop;
  OpBuilder &builder;
};

static LogicalResult emitLoopTripCountInTilingFunc(const TripCountEmitParams &params, Value &tripCount) {
  auto &tilingFunc = params.tilingFunc;
  auto &loop = params.loop;
  auto &builder = params.builder;
  const auto &originalKernel = params.originalKernel;
  int64_t staticTripCount = 0;
  if (getStaticTripCount(loop, staticTripCount)) {
    tripCount = builder.create<arith::ConstantIndexOp>(tilingFunc.getLoc(), staticTripCount);
    return success();
  }

  Value lb = cloneUpperBoundDefinition(loop.getLowerBound(), originalKernel, tilingFunc, builder);
  Value ub = cloneUpperBoundDefinition(loop.getUpperBound(), originalKernel, tilingFunc, builder);
  Value step = cloneUpperBoundDefinition(loop.getStep(), originalKernel, tilingFunc, builder);
  if (!lb || !ub || !step) {
    tilingFunc.emitError("failed to clone loop bounds when computing runtime parallel tile size");
    return failure();
  }

  Location loc = tilingFunc.getLoc();
  Value extent = builder.create<arith::SubIOp>(loc, ub, lb);
  tripCount = builder.create<arith::CeilDivSIOp>(loc, extent, step);
  return success();
}

static Value emitTileCountFromTileSize(Location loc, OpBuilder &builder, mlir::scf::ForOp loop, Value tileSize) {
  return builder.create<arith::CeilDivSIOp>(loc, emitLoopTripCountInCurrentFunc(loc, builder, loop), tileSize);
}

// Bundled params for computeParallelDimTileSize to stay under readability-function-size_parameters limit.
struct ParallelDimTileParams {
  mutable func::FuncOp tilingFunc;
  mutable func::FuncOp originalKernel;
  mutable mlir::scf::ForOp loop;
  OpBuilder &builder;
  Value coreNum;
  Value one;
};

static LogicalResult computeParallelDimTileSize(const ParallelDimTileParams &params, Value &producedTasks,
                                                Value &tileSize) {
  auto &tilingFunc = params.tilingFunc;
  const auto &originalKernel = params.originalKernel;
  const auto &loop = params.loop;
  auto &builder = params.builder;
  const auto &coreNum = params.coreNum;
  const auto &one = params.one;
  Location loc = tilingFunc.getLoc();
  Value tripCount;
  if (failed(emitLoopTripCountInTilingFunc({tilingFunc, originalKernel, loop, builder}, tripCount))) {
    return failure();
  }

  Value hasEnoughTasks = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, producedTasks, coreNum);
  Value remainingTarget = builder.create<arith::CeilDivSIOp>(loc, coreNum, producedTasks);
  Value smallTrip = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, tripCount, remainingTarget);
  Value splitTile = builder.create<arith::DivSIOp>(loc, tripCount, remainingTarget);
  splitTile = builder.create<arith::MaxSIOp>(loc, splitTile, one);
  Value activeTile = builder.create<arith::SelectOp>(loc, smallTrip, one, splitTile);
  tileSize = builder.create<arith::SelectOp>(loc, hasEnoughTasks, tripCount, activeTile);
  Value tileCount = builder.create<arith::CeilDivSIOp>(loc, tripCount, tileSize);
  producedTasks = builder.create<arith::MulIOp>(loc, producedTasks, tileCount);
  return success();
}

// Bundled params for emitParallelOuterTilesForBand to stay under readability-function-size_parameters limit.
struct ParallelOuterTileParams {
  mutable func::FuncOp tilingFunc;
  mutable func::FuncOp originalKernel;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<unsigned> parallelDims;
  OpBuilder &builder;
};

static LogicalResult emitParallelOuterTilesForBand(const ParallelOuterTileParams &params,
                                                   SmallVectorImpl<Value> &outerTileByDim) {
  auto &tilingFunc = params.tilingFunc;
  const auto &originalKernel = params.originalKernel;
  const auto &band = params.band;
  const auto &parallelDims = params.parallelDims;
  auto &builder = params.builder;
  outerTileByDim.assign(band.size(), Value());
  if (parallelDims.empty()) {
    return success();
  }

  Location loc = tilingFunc.getLoc();
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value coreNum = builder.create<arith::ConstantIndexOp>(loc, getNpuCoreNum(originalKernel));
  Value producedTasks = one;

  for (unsigned dim : parallelDims) {
    if (dim >= band.size()) {
      continue;
    }
    Value tileSize;
    if (failed(computeParallelDimTileSize({tilingFunc, originalKernel, band[dim], builder, coreNum, one}, producedTasks,
                                          tileSize))) {
      return failure();
    }
    outerTileByDim[dim] = tileSize;
  }
  return success();
}

static int64_t multiplyAndCapPositive(int64_t lhs, int64_t rhs) {
  if (lhs <= 0 || rhs <= 0) {
    return 1;
  }
  return (lhs > std::numeric_limits<int64_t>::max() / rhs) ? std::numeric_limits<int64_t>::max() : lhs * rhs;
}

static int64_t floorSqrtInt64(int64_t value) {
  if (value <= 0) {
    return 0;
  }
  int64_t root = static_cast<int64_t>(std::sqrt(static_cast<long double>(value)));
  while (multiplyAndCapPositive(root + 1, root + 1) <= value) {
    ++root;
  }
  while (multiplyAndCapPositive(root, root) > value) {
    --root;
  }
  return root;
}

std::string formatTilingKeySuffix(int64_t key) {
  std::string keyStr = std::to_string(key);
  return keyStr.size() == 1 ? "0" + keyStr : keyStr;
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
        SmallVector<NamedAttribute, kSmallVectorSizeFour> attrs;
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

  auto setArgKernelKind = [&func, &ctx, &katName](unsigned idx, KernelArgType kind) {
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
    if (!replaced) {
      nas.emplace_back(katName, kat);
    }

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
  if (body == nullptr) {
    return;
  }
  for (Operation &op : body->without_terminator()) {
    if (auto childLoop = dyn_cast<mlir::scf::ForOp>(&op)) {
      children.push_back(childLoop);
    }
  }
}

static bool isLeafForLoop(mlir::scf::ForOp loop) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> children;
  collectDirectChildLoopsInOrder(loop, children);
  return children.empty();
}

static LogicalResult collectSingleChain(mlir::scf::ForOp root, SmallVectorImpl<mlir::scf::ForOp> &chain) {
  chain.clear();
  for (mlir::scf::ForOp current = root; current;) {
    chain.push_back(current);
    SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> children;
    collectDirectChildLoopsInOrder(current, children);
    if (children.empty()) {
      return success();
    }
    if (children.size() != 1) {
      chain.clear();
      return failure();
    }
    current = children.front();
  }
  return failure();
}

static LogicalResult buildLeafBranchBandPlanForRoot(mlir::scf::ForOp rootLoop, LeafBranchBandPlan &plan) {
  if (!rootLoop) {
    return failure();
  }

  plan = LeafBranchBandPlan{};
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> linearPrefix;
  linearPrefix.push_back(rootLoop);
  mlir::scf::ForOp current = rootLoop;

  while (true) {
    SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> children;
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

    plan.branchBands.clear();
    for (mlir::scf::ForOp child : children) {
      SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> branchBand;
      if (failed(collectSingleChain(child, branchBand))) {
        return failure();
      }
      plan.branchBands.push_back(std::move(branchBand));
    }
    auto representativeIt =
      llvm::max_element(plan.branchBands, [](const auto &lhs, const auto &rhs) { return lhs.size() < rhs.size(); });
    if (representativeIt == plan.branchBands.end()) {
      return failure();
    }

    plan.representativeBand = linearPrefix;
    plan.representativeLeafDim = static_cast<unsigned>(linearPrefix.size());
    plan.representativeBand.append(representativeIt->begin(), representativeIt->end());
    plan.hasLeafBranching = true;
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
                                       std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> &bands) {
  bands.clear();
  bands.reserve(plans.size());
  for (const LeafBranchBandPlan &plan : plans) {
    if (!plan.representativeBand.empty()) {
      bands.push_back(plan.representativeBand);
    }
  }
}

static LogicalResult rejectDynamicMultiTopLevelBands(func::FuncOp funcOp, bool isStaticShape,
                                                     ArrayRef<LeafBranchBandPlan> plans) {
  if (isStaticShape || plans.size() <= 1) {
    return success();
  }
  funcOp.emitError("dynamic shape with multiple top-level bands is not supported by NPU auto tiling");
  return failure();
}

static void logAllBandTileSizes(const std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> &allBandTileSizes,
                                const std::vector<SmallVector<int, kSmallVectorSizeSix>> &allBandConstraintMaxs) {
  llvm::dbgs() << "=== All Band Tile Sizes ===\n";
  for (size_t i = 0; i < allBandTileSizes.size(); ++i) {
    llvm::dbgs() << "Band " << i << " tile sizes: [";
    for (size_t j = 0; j < allBandTileSizes[i].size(); ++j) {
      if (j > 0) {
        llvm::dbgs() << ", ";
      }
      if (allBandTileSizes[i][j] == static_cast<unsigned>(-1)) {
        int constraintMax =
          (i < allBandConstraintMaxs.size() && j < allBandConstraintMaxs[i].size()) ? allBandConstraintMaxs[i][j] : 0;
        llvm::dbgs() << "dynamic(max=" << constraintMax << ")";
      } else {
        llvm::dbgs() << allBandTileSizes[i][j];
      }
    }
    llvm::dbgs() << "]\n";
  }
  llvm::dbgs() << "==========================\n";
}

static LogicalResult calculateTileSizesForBands(func::FuncOp funcOp, bool useAutoTiling, BandTilingOutput output,
                                                size_t &levelToTile) {
  auto &bandsToUse = output.bandsToUse;
  auto &allBandTileSizes = output.allBandTileSizes;
  auto &allBandConstraintMaxs = output.allBandConstraintMaxs;
  allBandTileSizes.clear();
  allBandConstraintMaxs.clear();

  if (bandsToUse.empty()) {
    return success();
  }

  if (std::any_of(bandsToUse.begin(), bandsToUse.end(),
                  [](const auto &band) { return failed(verifyBandIsNestedChain(band)); })) {
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
    SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> curBand = bandsToUse[bandIdx];
    SmallVector<unsigned, kSmallVectorSizeSix> bandTileSizes;
    SmallVector<int, kSmallVectorSizeSix> bandConstraintMaxs;

    for (size_t level = 0; level < levelToTile; ++level) {
      getTileSizeWithSolver(solver, curBand, &bandTileSizes, &bandConstraintMaxs, TilingTaskDesc(bandIdx, level));
    }

    allBandTileSizes.push_back(bandTileSizes);
    allBandConstraintMaxs.push_back(bandConstraintMaxs);
  }

  logAllBandTileSizes(allBandTileSizes, allBandConstraintMaxs);

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
  if (!v) {
    return v;
  }

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
// Bundled params for constructTiledLoopStatic to stay under readability-function-size_parameters limit.
struct ConstructTiledLoopParams {
  mutable mlir::scf::ForOp rootScfForOp;
  unsigned width;
  MutableArrayRef<mlir::scf::ForOp> tiledLoops;
  OpBuilder &builder;
  std::map<int64_t, Value> &constantCache;
};

static void constructTiledLoopStatic(const ConstructTiledLoopParams &params) {
  auto &rootScfForOp = params.rootScfForOp;
  auto &builder = params.builder;
  auto &constantCache = params.constantCache;
  const auto &width = params.width;
  const auto &tiledLoops = params.tiledLoops;
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
      [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value /* iv */, mlir::ValueRange iterArgs) {
        // Create yield with iter args (pass through by default)
        nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc, iterArgs);
      });

    // Insert topLoop before the terminator in wrapperLoop's body
    auto *terminator = wrapperLoop.getBody()->getTerminator();
    mlir::Block::iterator insertLoc =
      (terminator != nullptr) ? terminator->getIterator() : wrapperLoop.getBody()->end();
    wrapperLoop.getBody()->getOperations().splice(insertLoc, topLoop->getBlock()->getOperations(), topLoop);

    // CRITICAL: Update yield to use inner loop's results (for loops with iter_args)
    // This ensures value flow: inner.results -> outer.yield -> outer.results
    if ((terminator != nullptr) && hasIterArgs && isa<mlir::scf::YieldOp>(terminator)) {
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
    return {};
  }
  OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = bandRoot.getLoc();
  builder.setInsertionPoint(bandRoot);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value core = builder.create<arith::ConstantIndexOp>(loc, useCore);
  auto mapLoop =
    builder.create<mlir::scf::ForOp>(loc, c0, core, c1, ValueRange{},
                                     [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value,
                                        mlir::ValueRange) { nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc); });
  mapLoop->setAttr(kMapForToForallAttr, builder.getUnitAttr());
  funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(useCore));
  bandRoot->moveBefore(mapLoop.getBody()->getTerminator());
  return mapLoop;
}

// Bundled params for buildParallelTileCoordMap to stay under readability-function-size_parameters limit.
struct ParallelTileCoordMapParams {
  ArrayRef<unsigned> parallelDims;
  ArrayRef<Value> tileCountsByDim;
  mutable mlir::scf::ForOp parallelMapLoop;
  mutable mlir::scf::ForOp parallelTileLoop;
  mutable mlir::scf::ForOp insertBeforeLoop;
  OpBuilder &builder;
  Value linearTaskId;
};

static void buildParallelTileCoordMap(const ParallelTileCoordMapParams &params,
                                      SmallVectorImpl<Value> &parallelTileCoordByDim) {
  const auto &parallelDims = params.parallelDims;
  const auto &tileCountsByDim = params.tileCountsByDim;
  mlir::scf::ForOp insertBeforeLoop = params.insertBeforeLoop;
  auto &parallelMapLoop = params.parallelMapLoop;
  auto &parallelTileLoop = params.parallelTileLoop;
  auto &builder = params.builder;
  Value remaining = params.linearTaskId;
  if ((!remaining && (!parallelMapLoop || !parallelTileLoop)) || !insertBeforeLoop || parallelDims.empty()) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = insertBeforeLoop.getLoc();
  builder.setInsertionPoint(insertBeforeLoop);
  if (!remaining) {
    AffineExpr taskLoopExpr = builder.getAffineDimExpr(0);
    AffineExpr mapLoopExpr = builder.getAffineDimExpr(1);
    auto taskIdMap = AffineMap::get(2, 0, taskLoopExpr + mapLoopExpr, builder.getContext());
    remaining = builder.create<affine::AffineApplyOp>(
      loc, taskIdMap, ValueRange{parallelTileLoop.getInductionVar(), parallelMapLoop.getInductionVar()});
  }
  AffineExpr remainingExpr = builder.getAffineDimExpr(0);
  AffineExpr tileCountExpr = builder.getAffineSymbolExpr(0);
  auto remMap = AffineMap::get(1, 1, remainingExpr % tileCountExpr, builder.getContext());
  auto divMap = AffineMap::get(1, 1, remainingExpr.floorDiv(tileCountExpr), builder.getContext());
  for (int64_t pos = static_cast<int64_t>(parallelDims.size()) - 1; pos >= 0; --pos) {
    unsigned dim = parallelDims[static_cast<size_t>(pos)];
    if (dim >= tileCountsByDim.size() || dim >= parallelTileCoordByDim.size()) {
      continue;
    }
    Value tileCount = tileCountsByDim[dim];
    if (!tileCount) {
      continue;
    }
    if (auto tileCountConst = getConstantIndexValue(tileCount); tileCountConst && *tileCountConst == 1) {
      parallelTileCoordByDim[dim] = builder.create<arith::ConstantIndexOp>(loc, 0);
      continue;
    }
    parallelTileCoordByDim[dim] = builder.create<affine::AffineApplyOp>(loc, remMap, ValueRange{remaining, tileCount});
    if (pos > 0) {
      remaining = builder.create<affine::AffineApplyOp>(loc, divMap, ValueRange{remaining, tileCount});
    }
  }
}

// Bundled params for createParallelTileLoop to stay under readability-function-size_parameters limit.
struct ParallelTileLoopParams {
  mutable mlir::scf::ForOp mapLoop;
  mutable mlir::scf::ForOp bandRoot;
  Value totalTiles;
  int64_t useCore;
  OpBuilder &builder;
};

static mlir::scf::ForOp createParallelTileLoop(const ParallelTileLoopParams &params) {
  auto &mapLoop = params.mapLoop;
  auto &bandRoot = params.bandRoot;
  const auto &totalTiles = params.totalTiles;
  const auto &useCore = params.useCore;
  auto &builder = params.builder;
  if (!mapLoop || !bandRoot || !totalTiles || useCore <= 0) {
    return {};
  }

  OpBuilder::InsertionGuard guard(builder);
  mlir::Location loc = bandRoot.getLoc();
  builder.setInsertionPoint(bandRoot);
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value step = builder.create<arith::ConstantIndexOp>(loc, useCore);
  AffineExpr taskNumExpr = builder.getAffineDimExpr(0);
  AffineExpr mapLoopExpr = builder.getAffineDimExpr(1);
  auto taskUbMap = AffineMap::get(2, 0, taskNumExpr - mapLoopExpr, builder.getContext());
  Value taskUb =
    builder.create<affine::AffineApplyOp>(loc, taskUbMap, ValueRange{totalTiles, mapLoop.getInductionVar()});
  auto tileLoop =
    builder.create<mlir::scf::ForOp>(loc, c0, taskUb, step, ValueRange{},
                                     [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value,
                                        mlir::ValueRange) { nestedBuilder.create<mlir::scf::YieldOp>(nestedLoc); });
  bandRoot->moveBefore(tileLoop.getBody()->getTerminator());
  return tileLoop;
}

// Bundled input params for createParallelMapAndTileLoop to stay under readability-function-size_parameters limit.
struct ParallelMapTileInput {
  mutable func::FuncOp funcOp;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  ArrayRef<unsigned> parallelDims;
  bool useRuntimeTileCounts;
  OpBuilder &builder;
};

// Bundled output params for createParallelMapAndTileLoop.
struct ParallelMapTileOutput {
  mlir::scf::ForOp &parallelMapLoop;
  mlir::scf::ForOp &parallelTileLoop;
  SmallVectorImpl<Value> &parallelTileCoordByDim;
  int64_t &parallelUseCore;
};

static void collectParallelTileCounts(const ParallelMapTileInput &in, int64_t coreNum, mlir::Location loc,
                                      SmallVectorImpl<Value> &tileCountsByDim, Value &totalTilesValue,
                                      int64_t &totalTiles) {
  const auto &band = in.band;
  const auto &tileSizeValues = in.tileSizeValues;
  const auto &tileSizesInt = in.tileSizesInt;
  const auto &parallelDims = in.parallelDims;
  const auto &useRuntimeTileCounts = in.useRuntimeTileCounts;
  auto &builder = in.builder;
  for (unsigned dim : parallelDims) {
    if (dim >= band.size() || dim >= tileSizesInt.size() || tileSizesInt[dim] == 0) {
      continue;
    }
    if (useRuntimeTileCounts) {
      if (dim >= tileSizeValues.size() || !tileSizeValues[dim]) {
        continue;
      }
      tileCountsByDim[dim] = emitTileCountFromTileSize(loc, builder, band[dim], tileSizeValues[dim]);
      totalTilesValue = builder.create<arith::MulIOp>(loc, totalTilesValue, tileCountsByDim[dim]);
    } else {
      int64_t tileCount = 0;
      if (tileSizesInt[dim] == static_cast<unsigned>(-1) ||
          !getStaticParallelTileWork(band[dim], tileSizesInt[dim], tileCount)) {
        tileCount = coreNum;
      }
      tileCountsByDim[dim] = builder.create<arith::ConstantIndexOp>(loc, std::max<int64_t>(tileCount, 1));
      totalTiles = multiplyAndCapPositive(totalTiles, std::max<int64_t>(tileCount, 1));
    }
    band[dim]->setAttr(kParallelAxisAttr, builder.getUnitAttr());
  }
}

static void createParallelMapAndTileLoop(const ParallelMapTileInput &in, ParallelMapTileOutput out) {
  const auto &funcOp = in.funcOp;
  const auto &band = in.band;
  const auto &parallelDims = in.parallelDims;
  const auto &useRuntimeTileCounts = in.useRuntimeTileCounts;
  auto &builder = in.builder;
  auto &parallelMapLoop = out.parallelMapLoop;
  auto &parallelTileLoop = out.parallelTileLoop;
  auto &parallelTileCoordByDim = out.parallelTileCoordByDim;
  auto &parallelUseCore = out.parallelUseCore;
  parallelMapLoop = mlir::scf::ForOp();
  parallelTileLoop = mlir::scf::ForOp();
  parallelUseCore = 0;
  parallelTileCoordByDim.assign(band.size(), Value());
  if (band.empty()) {
    return;
  }

  mlir::scf::ForOp root = band.front();
  if (root.getNumResults() != 0 && !isReductionLoopWithIterArgs(root)) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(root);

  int64_t coreNum = getNpuCoreNum(funcOp);
  mlir::Location loc = root.getLoc();
  SmallVector<Value, kSmallVectorSizeSix> tileCountsByDim(band.size(), Value());
  Value totalTilesValue;
  int64_t totalTiles = 1;
  if (useRuntimeTileCounts) {
    totalTilesValue = builder.create<arith::ConstantIndexOp>(loc, 1);
  }

  collectParallelTileCounts(in, coreNum, loc, tileCountsByDim, totalTilesValue, totalTiles);

  if (!useRuntimeTileCounts) {
    totalTilesValue = builder.create<arith::ConstantIndexOp>(loc, totalTiles);
  }
  parallelUseCore =
    parallelDims.empty() ? 1 : (useRuntimeTileCounts ? coreNum : std::min<int64_t>(coreNum, totalTiles));
  parallelMapLoop = createParallelMapLoop(funcOp, root, parallelUseCore, builder);
  if (!parallelMapLoop || parallelDims.empty()) {
    return;
  }

  if (!useRuntimeTileCounts && totalTiles <= parallelUseCore) {
    buildParallelTileCoordMap({parallelDims, tileCountsByDim, parallelMapLoop, parallelTileLoop, root, builder,
                               parallelMapLoop.getInductionVar()},
                              parallelTileCoordByDim);
    return;
  }
  parallelTileLoop = createParallelTileLoop({parallelMapLoop, root, totalTilesValue, parallelUseCore, builder});
  buildParallelTileCoordMap({parallelDims, tileCountsByDim, parallelMapLoop, parallelTileLoop, root, builder, Value()},
                            parallelTileCoordByDim);
}

// Build the LoopBounds for a no-split origLoop at the requested level.
// `dropReductionInits` is honored only on wrapper levels (FirstLevel/Middle); the point level
// always keeps `origLoop.getInitArgs()` so per-iteration reduction semantics are preserved.
// `prevLoop` is only consulted for non-FirstLevel/non-reduction inits.
static LoopBounds buildNoSplitBounds(const BuildContext &bc, const NoSplitBoundsParams &params) {
  auto &builder = bc.builder;
  auto &origLoop = params.origLoop;
  const auto &kind = params.kind;
  LoopBounds bounds;
  if (kind == NoSplitKind::Point) {
    bounds.lb = recreateConstantOrSelf(origLoop.getLowerBound(), builder);
    bounds.ub = recreateConstantOrSelf(origLoop.getUpperBound(), builder);
    bounds.step = recreateConstantOrSelf(origLoop.getStep(), builder);
  } else {
    const auto &loc = bc.loc;
    auto &constantCache = bc.constantCache;
    bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    bounds.ub = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  }
  if (isReductionLoop(origLoop)) {
    const auto &dropReductionInits = params.dropReductionInits;
    bounds.inits =
      (kind == NoSplitKind::Point || !dropReductionInits) ? ValueRange(origLoop.getInitArgs()) : ValueRange{};
  } else {
    auto &prevLoop = params.prevLoop;
    bounds.inits =
      (kind == NoSplitKind::FirstLevel) ? ValueRange(origLoop.getInitArgs()) : ValueRange(prevLoop.getResults());
  }
  return bounds;
}

// Helper: Create bounds for first level tile loop
static LoopBounds createFirstLevelTileLoopBounds(const BuildContext &bc, const FirstLevelTileBoundsParams &params) {
  const auto &loc = bc.loc;
  auto &builder = bc.builder;
  auto &constantCache = bc.constantCache;
  auto &origLoop = params.origLoop;
  const auto &tilesizeInt = params.tilesizeInt;
  const auto &dynamicTileCoreNum = params.dynamicTileCoreNum;
  const auto &useRuntimeTileCount = params.useRuntimeTileCount;
  if (isNoSplitLoop(origLoop)) {
    const auto &dropFirstLevelReductionIterArgs = params.dropFirstLevelReductionIterArgs;
    return buildNoSplitBounds(bc,
                              {origLoop, /*prevLoop=*/{}, NoSplitKind::FirstLevel, dropFirstLevelReductionIterArgs});
  }

  LoopBounds bounds;
  Value origUb = origLoop.getUpperBound();

  // lb = 0
  bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);

  // Determine if this is a dynamic axis by checking origUb
  auto origUbConst = getConstantIndexValue(origUb);

  if (useRuntimeTileCount) {
    const auto &tilesize = params.tilesize;
    bounds.ub = emitTileCountFromTileSize(loc, builder, origLoop, tilesize);
  } else if (origUbConst && tilesizeInt != static_cast<unsigned>(-1) && tilesizeInt != 0) {
    // Static axis with known tilesize: compute ceildiv statically
    // Use tilesizeInt (original unsigned value) instead of trying to extract from Value
    int64_t numBlocks = (origUbConst.value() + tilesizeInt - 1) / tilesizeInt;
    bounds.ub = getOrCreateConstantStatic(loc, numBlocks, builder, constantCache);
  } else {
    // Dynamic axis: tilesize is computed as ceildiv(origUb, coreNum).
    bounds.ub = getOrCreateConstantStatic(loc, dynamicTileCoreNum, builder, constantCache);
  }

  // step = 1
  bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  bounds.inits = origLoop.getInitArgs();

  return bounds;
}

// Helper: Create bounds for middle level tile loop
static LoopBounds createMiddleLevelTileLoopBounds(const BuildContext &bc, const MiddleLevelTileBoundsParams &params) {
  const auto &loc = bc.loc;
  auto &builder = bc.builder;
  auto &constantCache = bc.constantCache;
  auto &prevLoop = params.prevLoop;
  auto &origLoop = params.origLoop;
  const auto &curTilesize = params.curTilesize;
  const auto &prevTilesize = params.prevTilesize;
  if (isNoSplitLoop(origLoop)) {
    const auto &escapeReduceIterArgs = params.escapeReduceIterArgs;
    // Drop reduction inits on the middle level unless this loop's results escape and we want
    // sinkReduceLoopResultsToMiddleLevel to consume them.
    bool dropReductionInits = origLoop.getNumResults() == 0 || !escapeReduceIterArgs;
    return buildNoSplitBounds(bc, {origLoop, prevLoop, NoSplitKind::Middle, dropReductionInits});
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

static bool collectStaticFullPointTileSizes(mlir::scf::ForOp origLoop, ArrayRef<std::pair<Value, Value>> levelInfo,
                                            SmallVectorImpl<int64_t> &tileSizes) {
  tileSizes.clear();
  if (!origLoop || isReductionLoop(origLoop) || levelInfo.empty()) {
    return false;
  }

  std::optional<int64_t> origLb = getConstantIndexValue(origLoop.getLowerBound());
  std::optional<int64_t> origUb = getConstantIndexValue(origLoop.getUpperBound());
  std::optional<int64_t> origStep = getConstantIndexValue(origLoop.getStep());
  if (!origLb || !origUb || !origStep || *origLb != 0 || *origStep != 1 || *origUb <= *origLb) {
    return false;
  }

  tileSizes.reserve(levelInfo.size());
  for (const auto &entry : levelInfo) {
    const auto &tileSizeValue = entry.second;
    std::optional<int64_t> tileSize = getConstantIndexValue(tileSizeValue);
    if (!tileSize || *tileSize <= 0) {
      tileSizes.clear();
      return false;
    }
    tileSizes.push_back(*tileSize);
  }

  int64_t extent = *origUb - *origLb;
  if (extent % tileSizes.front() != 0) {
    return false;
  }
  for (size_t i = 1; i < tileSizes.size(); ++i) {
    if (tileSizes[i - 1] % tileSizes[i] != 0) {
      return false;
    }
  }
  return true;
}

// Bundled params for createFullTilePointBound to stay under readability-function-size_parameters limit.
struct FullTilePointBoundParams {
  mlir::Location loc;
  ArrayRef<std::pair<Value, Value>> levelInfo;
  ArrayRef<int64_t> tileSizes;
  int64_t offset;
  OpBuilder &builder;
};

static Value createFullTilePointBound(const FullTilePointBoundParams &params) {
  const auto &loc = params.loc;
  const auto &levelInfo = params.levelInfo;
  const auto &tileSizes = params.tileSizes;
  const auto &offset = params.offset;
  auto &builder = params.builder;
  MLIRContext *context = builder.getContext();
  SmallVector<Value, kSmallVectorSizeEight> operands;
  operands.reserve(levelInfo.size());
  AffineExpr expr = builder.getAffineConstantExpr(offset);
  for (auto [idx, entry] : llvm::enumerate(levelInfo)) {
    operands.push_back(entry.first);
    expr = expr + builder.getAffineDimExpr(static_cast<unsigned>(idx)) * tileSizes[idx];
  }
  auto map = AffineMap::get(static_cast<unsigned>(levelInfo.size()), /* symbolCount= */ 0, expr, context);
  return builder.create<mlir::affine::AffineApplyOp>(loc, map, operands);
}

// Helper: Create bounds for point loop
static LoopBounds createPointLoopBounds(const BuildContext &bc, mlir::scf::ForOp origLoop,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop) {
  const auto &loc = bc.loc;
  auto &builder = bc.builder;
  auto &constantCache = bc.constantCache;
  if (isNoSplitLoop(origLoop)) {
    return buildNoSplitBounds(bc, {origLoop, prevLoop, NoSplitKind::Point, /*dropReductionInits=*/false});
  }

  LoopBounds bounds;
  SmallVector<int64_t, kSmallVectorSizeFour> fullTileSizes;
  if (collectStaticFullPointTileSizes(origLoop, levelInfo, fullTileSizes)) {
    bounds.lb = createFullTilePointBound({loc, levelInfo, fullTileSizes, /*offset=*/0, builder});
    bounds.ub = createFullTilePointBound({loc, levelInfo, fullTileSizes, fullTileSizes.back(), builder});
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    bounds.inits = prevLoop.getResults();
    return bounds;
  }

  Value origUb = origLoop.getUpperBound();
  Value clampedOrigUb = recreateConstantOrSelf(origUb, builder);
  SmallVector<Value, kSmallVectorSizeEight> operands;
  operands.reserve(levelInfo.size() * kSmallVectorSizeTwo + kSmallVectorSizeOne);
  for (const auto &entry : levelInfo) {
    const auto &iv = entry.first;
    operands.push_back(iv);
  }
  operands.push_back(clampedOrigUb);
  for (const auto &entry : levelInfo) {
    const auto &tileSize = entry.second;
    operands.push_back(tileSize);
  }

  auto levels = static_cast<unsigned>(levelInfo.size());
  AffineExpr lbExpr = builder.getAffineConstantExpr(0);
  AffineExpr origUbExpr = builder.getAffineDimExpr(levels);
  SmallVector<AffineExpr, kSmallVectorSizeFour> ubExprs{origUbExpr};
  for (unsigned k = 0; k < levels; ++k) {
    lbExpr = lbExpr + builder.getAffineDimExpr(k) * builder.getAffineSymbolExpr(k);
    ubExprs.push_back(lbExpr + builder.getAffineSymbolExpr(k));
  }

  SmallVector<AffineExpr, kSmallVectorSizeTwo> lbExprs{lbExpr, origUbExpr};
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
    [](mlir::OpBuilder &nestedBuilder, mlir::Location nestedLoc, mlir::Value /* iv */, mlir::ValueRange iterArgs) {
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
  if (newTerminator != nullptr) {
    newTerminator->erase();
  }

  // Move body operations from old to new (excluding terminator)
  if (!oldBody->empty()) {
    Operation *oldTerminator = oldBody->getTerminator();
    if (oldTerminator != nullptr) {
      // Move all operations except the terminator
      for (Operation &op : llvm::make_early_inc_range(*oldBody)) {
        if (&op == oldTerminator) {
          break;
        }
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

// Bundled params for buildLoopBoundsForTileLevel to stay under readability-function-size_parameters limit.
struct TileLevelBoundsInput {
  int i;
  int j;
  int bandSize;
  int tileNum;
  int curTile;
  int lastTile;
  MutableArrayRef<mlir::scf::ForOp> newLoops;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  const SmallVectorImpl<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>> &tileLevelInfo;
  bool escapeReduceIterArgs;
};

static bool buildLoopBoundsForTileLevel(const BuildContext &bc, const TileLevelBoundsInput &in,
                                        const TileRewriteContext &ctx, LoopBounds &bounds) {
  const auto &i = in.i;
  const auto &j = in.j;
  const auto &tileNum = in.tileNum;
  const auto &curTile = in.curTile;
  const auto &lastTile = in.lastTile;
  const auto &newLoops = in.newLoops;
  const auto &band = in.band;
  const auto &tileSizeValues = in.tileSizeValues;
  const auto &escapeReduceIterArgs = in.escapeReduceIterArgs;
  const auto &loc = bc.loc;
  auto &loopBuilder = bc.builder;
  auto &constantCache = bc.constantCache;
  mlir::scf::ForOp origLoop = band[j];

  if (i == 0) {
    const auto &tileSizesInt = in.tileSizesInt;
    unsigned tilesizeInt =
      (curTile < static_cast<int>(tileSizesInt.size())) ? tileSizesInt[curTile] : static_cast<unsigned>(-1);
    bool dropFirstLevelReductionIterArgs = ctx.dropMappedOutermostFirstLevelIterArgs && j == 0;
    bounds =
      createFirstLevelTileLoopBounds(BuildContext{loc, loopBuilder, constantCache},
                                     {origLoop, tileSizeValues[curTile], tilesizeInt, dropFirstLevelReductionIterArgs,
                                      ctx.dynamicTileCoreNum, ctx.useRuntimeFirstLevelTileCount});
    return true;
  }

  if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) {
    return false;
  }
  mlir::scf::ForOp prevLoop = newLoops[lastTile];

  if (i == tileNum) {
    const auto &tileLevelInfo = in.tileLevelInfo;
    bounds = createPointLoopBounds(BuildContext{loc, loopBuilder, constantCache}, origLoop, tileLevelInfo[j], prevLoop);
    return true;
  }

  bounds = createMiddleLevelTileLoopBounds(
    BuildContext{loc, loopBuilder, constantCache},
    {origLoop, tileSizeValues[curTile], tileSizeValues[lastTile], prevLoop, escapeReduceIterArgs});
  return true;
}

static bool shouldPropagateTransposeAttrToTileLoop(int tileLevel, int tileNum, mlir::scf::ForOp origLoop) {
  if (!origLoop || !origLoop->hasAttr(kTransposeLoopAttr)) {
    return false;
  }
  return tileLevel == tileNum || origLoop->hasAttr(kTreeLeafAttr);
}

static bool shouldDeleteRedundantTileWrapper(const TileLoopParams &p, mlir::scf::ForOp origLoop) {
  int i = p.i;
  int j = p.j;
  int bandSize = p.bandSize;
  int tileNum = p.tileNum;
  const auto &tileSizesInt = p.tileSizesInt;
  if (i >= tileNum) {
    return false;
  }
  int curTile = i * bandSize + j;
  if (curTile < 0 || curTile >= static_cast<int>(tileSizesInt.size())) {
    return false;
  }
  unsigned curTileSize = tileSizesInt[curTile];
  if (curTileSize == 0 || curTileSize == static_cast<unsigned>(-1)) {
    return false;
  }

  if (i == 0) {
    int64_t tripCount = 0;
    return getStaticTripCount(origLoop, tripCount) && tripCount <= static_cast<int64_t>(curTileSize);
  }

  int lastTile = curTile - bandSize;
  if (lastTile < 0 || lastTile >= static_cast<int>(tileSizesInt.size())) {
    return false;
  }
  unsigned prevTileSize = tileSizesInt[lastTile];
  return prevTileSize != 0 && prevTileSize != static_cast<unsigned>(-1) && prevTileSize <= curTileSize;
}

// Bundled params for updateLoopAttrsForTileLoop to stay under readability-function-size_parameters limit.
struct UpdateLoopAttrsParams {
  int i;
  int j;
  int bandSize;
  int tileNum;
  mutable mlir::scf::ForOp origLoop;
  mutable mlir::scf::ForOp newLoop;
  ArrayRef<unsigned> tileSizesInt;
  mlir::OpBuilder &builder;
};

static void updateLoopAttrsForTileLoop(const UpdateLoopAttrsParams &params) {
  const auto &i = params.i;
  const auto &j = params.j;
  const auto &bandSize = params.bandSize;
  const auto &tileNum = params.tileNum;
  const auto &origLoop = params.origLoop;
  auto &newLoop = params.newLoop;
  const auto &tileSizesInt = params.tileSizesInt;
  auto &builder = params.builder;
  if (!origLoop) {
    return;
  }

  if (shouldPropagateTransposeAttrToTileLoop(i, tileNum, origLoop)) {
    newLoop->setAttr(kTransposeLoopAttr, builder.getUnitAttr());
  }
  if (origLoop->hasAttr(kParallelAxisAttr)) {
    newLoop->setAttr(kParallelAxisAttr, builder.getUnitAttr());
  }

  if (!isReductionLoop(origLoop) && !origLoop->hasAttr(kNotInnerDimensionBroadcastLoopAttr) &&
      shouldDeleteRedundantTileWrapper(TileLoopParams{i, j, bandSize, tileNum, {}, {}, {}, tileSizesInt}, origLoop)) {
    newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
  }

  if (isReductionLoop(origLoop)) {
    ReduceDirection reduceType = getReduceType(origLoop);
    if (reduceType == ReduceDirection::Y) {
      // For reduce-y, remove all wrappers and keep only the point loop in final IR.
      if (i < tileNum) {
        newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
      }
      if (i == tileNum) {
        newLoop->setAttr(kReductionLoopAttr, builder.getUnitAttr());
      }
      return;
    }

    // For reduce-x/all, wrapper levels are single-iteration shells after no-split tiling.
    if (i < tileNum) {
      newLoop->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
    }

    // For inner point loop: preserve reduction attribute from original loop.
    if (i == tileNum && j == (bandSize - 1)) {
      newLoop->setAttr(kReductionLoopAttr, builder.getUnitAttr());
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

// Bundled tile info context for recordTileLevelInfoForLoop.
struct TileInfoContext {
  int i;
  int j;
  int tileNum;
  int curTile;
  ArrayRef<Value> tileSizeValues;
  SmallVectorImpl<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>> &tileLevelInfo;
};

static void recordTileLevelInfoForLoop(const TileInfoContext &tic, mlir::scf::ForOp newLoop, Value mappedIv = Value()) {
  const auto &i = tic.i;
  const auto &j = tic.j;
  const auto &tileNum = tic.tileNum;
  auto &curTile = tic.curTile;
  auto &tileSizeValues = tic.tileSizeValues;
  auto &tileLevelInfo = tic.tileLevelInfo;
  if (i >= tileNum || curTile >= static_cast<int>(tileSizeValues.size())) {
    return;
  }

  tileLevelInfo[j].push_back({mappedIv ? mappedIv : newLoop.getInductionVar(), tileSizeValues[curTile]});
}

// Bundled tile loop context for moveFirstTileLoopBodyToParallelScope.
struct TileLoopContext {
  int i;
  int j;
  int tileNum;
  int curTile;
  MutableArrayRef<mlir::scf::ForOp> newLoops;
  ArrayRef<Value> tileSizeValues;
  SmallVectorImpl<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>> &tileLevelInfo;
};

static bool moveFirstTileLoopBodyToParallelScope(mlir::scf::ForOp loop, mlir::scf::ForOp parallelScopeLoop,
                                                 Value parallelTileCoord, const TileLoopContext &tlc) {
  const auto &i = tlc.i;
  const auto &j = tlc.j;
  const auto &tileNum = tlc.tileNum;
  const auto &curTile = tlc.curTile;
  auto &newLoops = tlc.newLoops;
  auto &tileLevelInfo = tlc.tileLevelInfo;
  const auto &tileSizeValues = tlc.tileSizeValues;
  if (!loop || !parallelScopeLoop || !parallelTileCoord || loop.getNumResults() != 0) {
    return false;
  }

  loop.getInductionVar().replaceAllUsesWith(parallelTileCoord);
  Block *body = loop.getBody();
  Operation *terminator = body->getTerminator();
  while (!body->empty() && &body->front() != terminator) {
    body->front().moveBefore(loop);
  }
  loop.erase();
  newLoops[curTile] = parallelScopeLoop;
  recordTileLevelInfoForLoop(TileInfoContext{i, j, tileNum, curTile, tileSizeValues, tileLevelInfo}, parallelScopeLoop,
                             parallelTileCoord);
  return true;
}

static bool tryMapFirstTileLoopToParallelWork(const TileLoopContext &tlc, const TileRewriteContext &ctx) {
  const auto &i = tlc.i;
  const auto &curTile = tlc.curTile;
  const auto &newLoops = tlc.newLoops;
  if (!ctx.parallelTileCoord || i != 0) {
    return false;
  }
  mlir::scf::ForOp parallelScopeLoop = (ctx.parallelTileLoop != nullptr) ? ctx.parallelTileLoop : ctx.parallelMapLoop;
  return moveFirstTileLoopBodyToParallelScope(newLoops[curTile], parallelScopeLoop, ctx.parallelTileCoord, tlc);
}

// Helper: Process a single tile loop (extracted to reduce cyclomatic complexity)
static void processSingleTileLoop(
  const TileLoopParams &p, SmallVectorImpl<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>> &tileLevelInfo,
  mlir::Location loc, const TileLoopCtx &tc) {
  auto &constantCache = tc.constantCache;
  auto &ctx = tc.ctx;
  int i = p.i;
  int j = p.j;
  int bandSize = p.bandSize;
  int tileNum = p.tileNum;
  auto &newLoops = p.newLoops;
  const auto &band = p.band;
  const auto &tileSizeValues = p.tileSizeValues;
  const auto &tileSizesInt = p.tileSizesInt;
  int curTile = i * bandSize + j;
  int lastTile = curTile - bandSize;
  if (curTile >= static_cast<int>(newLoops.size())) {
    return;
  }

  mlir::scf::ForOp loop = newLoops[curTile];
  if (!loop) {
    return;
  }

  if (tryMapFirstTileLoopToParallelWork(
        TileLoopContext{i, j, tileNum, curTile, newLoops, tileSizeValues, tileLevelInfo}, ctx)) {
    return;
  }

  mlir::scf::ForOp origLoop = band[j];
  mlir::OpBuilder loopBuilder(loop);
  bool escapeReduceIterArgs = origLoop && ctx.escapeReduceLoops.contains(origLoop.getOperation());

  LoopBounds bounds;
  if (!buildLoopBoundsForTileLevel(
        BuildContext{loc, loopBuilder, constantCache},
        TileLevelBoundsInput{i, j, bandSize, tileNum, curTile, lastTile, newLoops, band, tileSizeValues, tileSizesInt,
                             tileLevelInfo, escapeReduceIterArgs},
        ctx, bounds)) {
    return;
  }

  mlir::scf::ForOp newLoop = replaceLoopWithNewBounds(loop, bounds, loc, loopBuilder);
  newLoops[curTile] = newLoop;

  if (i == tileNum) {
    copySemanticLoopAttrsToPointLoop(origLoop, newLoop);
    if (lastTile >= 0 && lastTile < static_cast<int>(tileSizesInt.size())) {
      unsigned pointVectorSize = tileSizesInt[lastTile];
      if (pointVectorSize != static_cast<unsigned>(-1)) {
        newLoop->setAttr(kInnerLoopAttr, loopBuilder.getI64IntegerAttr(pointVectorSize));
      }
    }
  }

  updateLoopAttrsForTileLoop({i, j, bandSize, tileNum, origLoop, newLoop, tileSizesInt, loopBuilder});
  recordTileLevelInfoForLoop(TileInfoContext{i, j, tileNum, curTile, tileSizeValues, tileLevelInfo}, newLoop);
}

// Helper function to construct tiled index (bounds and steps) using tileSizeValues from memref
static void constructTiledIndexStatic(const TileLoopData &data, OpBuilder &builder,
                                      // cppcheck-suppress constParameter
                                      std::map<int64_t, Value> &constantCache, const TileRewriteContext &ctx) {
  auto &newLoops = data.newLoops;
  const auto &band = data.band;
  const auto &tileSizeValues = data.tileSizeValues;
  const auto &tileSizesInt = data.tileSizesInt;
  int bandSize = static_cast<int>(band.size());
  if (bandSize == 0 || tileSizeValues.empty()) {
    return;
  }

  mlir::Location loc = band[0]->getLoc();
  int tileNum = static_cast<int>(tileSizeValues.size()) / bandSize;

  // Track tile level info for each dimension: {IV, tilesize}
  SmallVector<SmallVector<std::pair<Value, Value>, kSmallVectorSizeFour>, kSmallVectorSizeFour> tileLevelInfo(bandSize);

  // Process each tile level and dimension
  for (int i = 0; i <= tileNum; ++i) {
    for (int j = 0; j < bandSize; ++j) {
      processSingleTileLoop(TileLoopParams{i, j, bandSize, tileNum, newLoops, band, tileSizeValues, tileSizesInt},
                            tileLevelInfo, loc, {constantCache, ctx});
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
  }
  if (dynamicUb && dynamicLb) {
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

// Bundled tail loop bounds to stay under readability-function-size_parameters limit.
struct TailLoopBounds {
  mlir::Value lb;
  mlir::Value ub;
  mlir::Value step;
};

// Emit a continuation tail loop right after `mainLoop` covering [tailLb, tailUb) with step `tailStep`,
// cloning `mainLoop` body into it (IV remapped) and recursively splitting the tail body.
static LogicalResult emitTailContinuationLoop(mlir::scf::ForOp mainLoop, const TailLoopBounds &bounds,
                                              OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = mainLoop.getLoc();
  builder.setInsertionPointAfter(mainLoop);
  auto tailLoop = builder.create<mlir::scf::ForOp>(loc, bounds.lb, bounds.ub, bounds.step, mainLoop.getInitArgs());
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
  if (differenceUbAndLb < origStep && (newDifferenceUbAndLb != 0)) {
    mlir::scf::ForOp mainLoop = rewriteForOpWithNewUb(forOp, newUb, builder);
    mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
    return emitTailContinuationLoop(mainLoop, TailLoopBounds{newUb, origUb, tailStepVal}, builder, constantCache);
  }

  // Branches B/C: process main body first.
  mlir::scf::ForOp currentLoop = (differenceUbAndLb >= origStep) ? rewriteForOpWithNewUb(forOp, newUb, builder) : forOp;
  if (failed(createTailBlockForBodyStatic(currentLoop, builder, constantCache))) {
    return failure();
  }
  bool isEqualToBlock = (newDifferenceUbAndLb == 0) || (newDifferenceUbAndLb == origStep && tailSize == origStep);
  if (differenceUbAndLb < origStep || (newDifferenceUbAndLb == 0) || isEqualToBlock) {
    return success();
  }
  mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
  return emitTailContinuationLoop(currentLoop, TailLoopBounds{newUb, origUb, tailStepVal}, builder, constantCache);
}

// Static helper function to create tail block for dynamic bounds
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value /* dynamicBound */,
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
  return emitTailContinuationLoop(currentLoop, TailLoopBounds{newUb, origUb, tailStepVal}, builder, constantCache);
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

// Bundled params for cloneAndWireTiledLoops to stay under readability-function-size_parameters limit.
struct CloneWireTiledLoopsParams {
  ArrayRef<mlir::scf::ForOp> band;
  MutableArrayRef<mlir::scf::ForOp> tiledLoops;
  unsigned tileSizesNum;
  mutable mlir::scf::ForOp rootScfForOp;
  OpBuilder &builder;
};

static LogicalResult cloneAndWireTiledLoops(const CloneWireTiledLoopsParams &params) {
  const auto &band = params.band;
  const auto &tiledLoops = params.tiledLoops;
  const auto &tileSizesNum = params.tileSizesNum;
  const auto &rootScfForOp = params.rootScfForOp;
  auto &builder = params.builder;
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

// Bundled params for finalizeRootLoopAfterTiling to stay under readability-function-size_parameters limit.
struct FinalizeRootLoopParams {
  mutable mlir::scf::ForOp rootScfForOp;
  MutableArrayRef<mlir::scf::ForOp> tiledLoops;
  unsigned tileSizesNum;
  unsigned forNum;
  const llvm::DenseSet<mlir::Operation *> &escapeReduceLoops;
  mutable mlir::scf::ForOp parallelMapLoop;
  bool dropMappedOutermostFirstLevelIterArgs;
  OpBuilder &builder;
};

static LogicalResult finalizeRootLoopAfterTiling(const FinalizeRootLoopParams &params) {
  auto &rootScfForOp = params.rootScfForOp;
  const auto &tiledLoops = params.tiledLoops;
  auto &escapeReduceLoops = params.escapeReduceLoops;
  bool rootEscape = escapeReduceLoops.contains(rootScfForOp.getOperation());
  if (isReductionLoopWithIterArgs(rootScfForOp) && rootEscape) {
    const auto &dropMappedOutermostFirstLevelIterArgs = params.dropMappedOutermostFirstLevelIterArgs;
    if (dropMappedOutermostFirstLevelIterArgs) {
      const auto &parallelMapLoop = params.parallelMapLoop;
      const auto &tileSizesNum = params.tileSizesNum;
      const auto &forNum = params.forNum;
      auto &builder = params.builder;
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
// optional `parallelMapLoop` that already wraps the band root. Multi-axis
// parallel dispatch additionally passes `parallelTileLoop` and this axis'
// `parallelTileCoord`, so the first tile wrapper can be folded into that
// precomputed tile coordinate.
static LogicalResult applyTilingToLoop(const ApplyTilingParams &params) {
  const auto &loop = params.loop;
  const auto &tileSizeValues = params.tileSizeValues;
  const auto &tileSizesInt = params.tileSizesInt;
  auto &builder = params.builder;
  auto &constantCache = params.constantCache;
  const auto &parallelMapLoop = params.parallelMapLoop;
  const auto &parallelTileLoop = params.parallelTileLoop;
  const auto &parallelTileCoord = params.parallelTileCoord;
  const auto &useRuntimeTileCounts = params.useRuntimeTileCounts;
  unsigned tileSizesNum = tileSizeValues.size();

  SmallVector<mlir::scf::ForOp, kSmallVectorSizeOne> band{loop};
  // The band root is "outermost" w.r.t. its tile nest if either it has no
  // surrounding scf.for at all, or its only surrounding scf.for is the
  // parallel map loop created by the caller.
  auto parentFor = loop->getParentOfType<mlir::scf::ForOp>();
  unsigned width = tileSizesNum + 1;
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> tiledLoops(width);

  llvm::DenseSet<mlir::Operation *> escapeReduceLoops = collectEscapingReductionLoops(band);
  func::FuncOp funcOp = loop->getParentOfType<func::FuncOp>();
  TileRewriteContext ctx{escapeReduceLoops,
                         parallelMapLoop,
                         parallelTileLoop,
                         parallelTileCoord,
                         getNpuCoreNum(funcOp),
                         useRuntimeTileCounts,
                         /* dropMappedOutermostFirstLevelIterArgs= */ !parentFor || parentFor == parallelMapLoop};
  constructTiledLoopStatic({loop, width, tiledLoops, builder, constantCache});

  // Replace all dummy loops first, before cloning operations, so IV mapping points to final loops.
  constructTiledIndexStatic(TileLoopData{tiledLoops, band, tileSizeValues, tileSizesInt}, builder, constantCache, ctx);

  if (failed(cloneAndWireTiledLoops({band, tiledLoops, tileSizesNum, loop, builder}))) {
    return failure();
  }

  return finalizeRootLoopAfterTiling({loop, tiledLoops, tileSizesNum, /*forNum=*/1, escapeReduceLoops, parallelMapLoop,
                                      ctx.dropMappedOutermostFirstLevelIterArgs, builder});
}

static void clearTemporaryLoopIdentificationAttrs(func::FuncOp funcOp, bool clearPointLoopAttr = true) {
  funcOp.walk([&clearPointLoopAttr](mlir::scf::ForOp loop) {
    if (loop->hasAttr(kTreeNodeIdAttr)) {
      loop->removeAttr(kTreeNodeIdAttr);
    }
    if (loop->hasAttr(kTreeLeafAttr)) {
      loop->removeAttr(kTreeLeafAttr);
    }
    if (clearPointLoopAttr && loop->hasAttr(kParallelAxisAttr)) {
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
  clearTemporaryLoopIdentificationAttrs(funcOp, /* clearPointLoopAttr= */ false);
  funcOp.emitError() << msg;
  return failure();
}

static LogicalResult findUniqueLoopByTreeNodeId(func::FuncOp funcOp, int64_t nodeId, mlir::scf::ForOp &loop) {
  loop = mlir::scf::ForOp();
  int64_t matchCount = 0;
  funcOp.walk([&nodeId, &matchCount, &loop](mlir::scf::ForOp candidate) {
    auto idAttr = candidate->getAttrOfType<IntegerAttr>(kTreeNodeIdAttr);
    if (!idAttr || idAttr.getInt() != nodeId) {
      return;
    }
    ++matchCount;
    loop = candidate;
  });
  return matchCount == 1 ? success() : failure();
}

// Tile `band` axis by axis. Each axis is re-located by its tree-node id before
// tiling so that earlier axes' transformations remain valid. `parallelTileCoordByDim`
// contains precomputed per-dimension tile coordinates for the unified parallel
// tile loop; empty entries mean the axis is not dispatched by that loop.
// Bundled params for applyDecoupledAxisTiling to stay under readability-function-size_parameters limit.
struct DecoupledAxisParams {
  mutable func::FuncOp funcOp;
  OpBuilder &builder;
  StringRef errLabel;
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<Value> tileSizeValues;
  ArrayRef<unsigned> tileSizesInt;
  mutable mlir::scf::ForOp parallelMapLoop;
  mutable mlir::scf::ForOp parallelTileLoop;
  ArrayRef<Value> parallelTileCoordByDim;
  bool useRuntimeTileCounts;
};

static LogicalResult applyDecoupledAxisTiling(const DecoupledAxisParams &params, int64_t &nextNodeId) {
  const auto &funcOp = params.funcOp;
  auto &builder = params.builder;
  const auto &errLabel = params.errLabel;
  const auto &band = params.band;
  const auto &tileSizeValues = params.tileSizeValues;
  const auto &tileSizesInt = params.tileSizesInt;
  const auto &parallelMapLoop = params.parallelMapLoop;
  const auto &parallelTileLoop = params.parallelTileLoop;
  const auto &parallelTileCoordByDim = params.parallelTileCoordByDim;
  const auto &useRuntimeTileCounts = params.useRuntimeTileCounts;
  unsigned bandSize = band.size();
  unsigned tileLevels = tileSizeValues.size() / bandSize;

  SmallVector<int64_t, kSmallVectorSizeSix> axisNodeIds;
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
    bool hasParallelTileCoord = dim < parallelTileCoordByDim.size() && parallelTileCoordByDim[dim];
    if (hasParallelTileCoord) {
      activeAxisLoop->setAttr(kParallelAxisAttr, builder.getUnitAttr());
    }

    SmallVector<Value, kSmallVectorSizeSix> axisTileValues;
    SmallVector<unsigned, kSmallVectorSizeSix> axisTileSizesInt;
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

    Value axisParallelTileCoord = hasParallelTileCoord ? parallelTileCoordByDim[dim] : Value();
    if (failed(applyTilingToLoop({activeAxisLoop, axisTileValues, axisTileSizesInt, builder, axisConstantCache,
                                  parallelMapLoop, parallelTileLoop, axisParallelTileCoord, useRuntimeTileCounts}))) {
      return emitTilingFailure(funcOp, "failed to apply " + Twine(errLabel) + " axis tiling");
    }
  }
  return success();
}

// Helper function to check if a scf.for loop is innermost (no nested scf.for inside)
static bool isInnermostScfLoop(mlir::scf::ForOp forOp) {
  Block *body = forOp.getBody();
  if (body == nullptr) {
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
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> lhsCommonOrder;
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> rhsCommonOrder;
};

static bool intersectLoopOrdersForTranspose(ArrayRef<mlir::scf::ForOp> lhsOrder, ArrayRef<mlir::scf::ForOp> rhsOrder,
                                            SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> &lhsCommonOrder,
                                            SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> &rhsCommonOrder) {
  if (lhsOrder.size() < kSmallVectorSizeTwo || rhsOrder.size() < kSmallVectorSizeTwo) {
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

  return lhsCommonOrder.size() >= kSmallVectorSizeTwo && lhsCommonOrder.size() == rhsCommonOrder.size();
}

static std::optional<TransposeOrderPairInfo> buildTransposeOrderPairAnalysis(ArrayRef<mlir::scf::ForOp> lhsOrder,
                                                                             ArrayRef<mlir::scf::ForOp> rhsOrder) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> lhsCommonOrder;
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> rhsCommonOrder;
  if (!intersectLoopOrdersForTranspose(lhsOrder, rhsOrder, lhsCommonOrder, rhsCommonOrder)) {
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

  return TransposeOrderPairInfo{std::move(lhsCommonOrder), std::move(rhsCommonOrder)};
}

static void clearBandBroadcastAttrsForTranspose(ArrayRef<mlir::scf::ForOp> band) {
  for (mlir::scf::ForOp loop : band) {
    loop->removeAttr(kBroadcastLoopAttr);
    loop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
  }
}

static void markActualTransposeLoops(ArrayRef<mlir::scf::ForOp> band, const TransposeOrderPairInfo &info,
                                     UnitAttr transposeAttr) {
  const auto &lhsCommonOrder = info.lhsCommonOrder;
  const auto &rhsCommonOrder = info.rhsCommonOrder;

  llvm::DenseSet<Operation *> permutedLoops;
  for (size_t i = 0; i < lhsCommonOrder.size(); ++i) {
    if (lhsCommonOrder[i] == rhsCommonOrder[i]) {
      continue;
    }
    mlir::scf::ForOp lhsLoop = lhsCommonOrder[i];
    mlir::scf::ForOp rhsLoop = rhsCommonOrder[i];
    permutedLoops.insert(lhsLoop.getOperation());
    permutedLoops.insert(rhsLoop.getOperation());
  }
  for (mlir::scf::ForOp loop : band) {
    if (permutedLoops.contains(loop.getOperation())) {
      loop->setAttr(kTransposeLoopAttr, transposeAttr);
    }
  }
}

// Bundled params for markTransposeOrderPair to stay under readability-function-size_parameters limit.
struct TransposeOrderPairParams {
  ArrayRef<mlir::scf::ForOp> band;
  ArrayRef<mlir::scf::ForOp> lhsOrder;
  ArrayRef<mlir::scf::ForOp> rhsOrder;
  UnitAttr transposeAttr;
};

static void markTransposeOrderPair(const TransposeOrderPairParams &params) {
  const auto &band = params.band;
  const auto &lhsOrder = params.lhsOrder;
  const auto &rhsOrder = params.rhsOrder;
  const auto &transposeAttr = params.transposeAttr;
  std::optional<TransposeOrderPairInfo> info = buildTransposeOrderPairAnalysis(lhsOrder, rhsOrder);
  if (!info) {
    return;
  }
  clearBandBroadcastAttrsForTranspose(band);
  markActualTransposeLoops(band, *info, transposeAttr);
}

static bool memOpsMayFormTransposePair(Operation *lhs, Operation *rhs) {
  auto isDerivedStore = [](memref::StoreOp store, memref::LoadOp load) {
    return CommonUtils::valueDependsOnTarget(store.getValueToStore(), load.getResult());
  };
  if (auto lhsLoad = dyn_cast<memref::LoadOp>(lhs)) {
    if (auto rhsStore = dyn_cast<memref::StoreOp>(rhs)) {
      return isDerivedStore(rhsStore, lhsLoad);
    }
  }
  if (auto rhsLoad = dyn_cast<memref::LoadOp>(rhs)) {
    if (auto lhsStore = dyn_cast<memref::StoreOp>(lhs)) {
      return isDerivedStore(lhsStore, rhsLoad);
    }
  }
  return true;
}

static SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> extractMemrefLoopOrder(Operation *op,
                                                                                  ArrayRef<mlir::scf::ForOp> band) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> order;
  auto appendIfMatch = [&band, &order](Value idx) {
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

static void reconcileTransposeAttrsAfterBranchScan(
  ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> leafBands, UnitAttr transposeAttr) {
  const bool hasTransposeLeaf =
    llvm::any_of(leafBands, [](const SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> &band) {
      return !band.empty() && band.back()->hasAttr(kTransposeLoopAttr);
    });
  if (!hasTransposeLeaf) {
    return;
  }
  for (ArrayRef<mlir::scf::ForOp> band : leafBands) {
    if (band.empty()) {
      continue;
    }
    band.back()->setAttr(kTransposeLoopAttr, transposeAttr);
  }
}

static void markBandTransposeLoops(func::FuncOp funcOp, const LeafBranchBandPlan &plan) {
  SmallVector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>, kSmallVectorSizeFour> leafBands;
  if (!plan.representativeBand.empty()) {
    if (!plan.hasLeafBranching) {
      leafBands.push_back(plan.representativeBand);
    } else {
      ArrayRef<mlir::scf::ForOp> prefix(plan.representativeBand.data(), plan.representativeLeafDim);
      for (const auto &branchBand : plan.branchBands) {
        SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> leafBand(prefix.begin(), prefix.end());
        leafBand.append(branchBand.begin(), branchBand.end());
        leafBands.push_back(std::move(leafBand));
      }
    }
  }
  for (ArrayRef<mlir::scf::ForOp> band : leafBands) {
    for (mlir::scf::ForOp loop : band) {
      loop->removeAttr(kTransposeLoopAttr);
    }
  }

  auto transposeAttr = UnitAttr::get(funcOp.getContext());
  const bool linearBand = !plan.hasLeafBranching;
  for (ArrayRef<mlir::scf::ForOp> band : leafBands) {
    if (band.empty()) {
      continue;
    }

    mlir::scf::ForOp leafLoop = band.back();
    mlir::scf::ForOp scanRoot = leafLoop;
    if (linearBand && band.size() > 1 && isReductionLoop(leafLoop)) {
      scanRoot = band[band.size() - kSmallVectorSizeTwo];
    }
    SmallVector<Operation *, kSmallVectorSizeEight> memOps;
    scanRoot.walk([&memOps](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op)) {
        memOps.push_back(op);
      }
    });

    for (size_t i = 0; i < memOps.size(); ++i) {
      SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> lhsOrder = extractMemrefLoopOrder(memOps[i], band);
      for (size_t j = i + 1; j < memOps.size(); ++j) {
        if (!memOpsMayFormTransposePair(memOps[i], memOps[j])) {
          continue;
        }
        SmallVector<mlir::scf::ForOp, kSmallVectorSizeFour> rhsOrder = extractMemrefLoopOrder(memOps[j], band);
        markTransposeOrderPair({band, lhsOrder, rhsOrder, transposeAttr});
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
  loop.getBody()->walk([&result](Operation *op) {
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

  return {};
}

static void markTransposeLoopChainWithVectorAttr(mlir::scf::ForOp innermostLoop, OpBuilder &builder,
                                                 mlir::scf::ForOp stopLoop = mlir::scf::ForOp()) {
  for (mlir::scf::ForOp curLoop = innermostLoop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (shouldSkipVectorAttrCandidate(curLoop, /* restrictToPointLoops= */ false, /* skipDeleteLoops= */ true)) {
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

static mlir::scf::ForOp findOutermostTransposeAncestor(mlir::scf::ForOp loop) {
  mlir::scf::ForOp outermost;
  for (mlir::scf::ForOp cur = loop->getParentOfType<mlir::scf::ForOp>(); cur;
       cur = cur->getParentOfType<mlir::scf::ForOp>()) {
    if (cur->hasAttr(kTransposeLoopAttr)) {
      outermost = cur;
    }
  }
  return outermost;
}

static bool hasMultiVecLoopInAncestorChain(mlir::scf::ForOp loop) {
  for (mlir::scf::ForOp cur = loop; cur; cur = cur->getParentOfType<mlir::scf::ForOp>()) {
    if (cur->hasAttr(kMultiVecLoopAttr)) {
      return true;
    }
  }
  return false;
}

// Consume sparse multi-vec markers without requiring every intervening point
// loop to participate. Markers are cleared after all leaf chains are visited so
// a shared leaf-branch prefix remains visible to every peer leaf.
static void markMultiVecLoopChainWithVectorAttr(mlir::scf::ForOp innermostLoop, OpBuilder &builder) {
  for (mlir::scf::ForOp curLoop = innermostLoop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    curLoop->removeAttr(kTransposeLoopAttr);
    if (shouldSkipVectorAttrCandidate(curLoop, /* restrictToPointLoops= */ false, /* skipDeleteLoops= */ true)) {
      continue;
    }
    if (!curLoop->hasAttr(kMultiVecLoopAttr)) {
      continue;
    }
    curLoop->removeAttr(kBroadcastLoopAttr);
    curLoop->removeAttr(kNotInnerDimensionBroadcastLoopAttr);
    if (curLoop->hasAttr(kVectorAttr) || curLoop->hasAttr(kReductionXLoopAttr) ||
        curLoop->hasAttr(kReductionYLoopAttr) || curLoop->hasAttr(kReductionAllLoopAttr)) {
      continue;
    }
    if (curLoop->hasAttr(kReductionLoopAttr)) {
      curLoop->removeAttr(kReductionLoopAttr);
      ReduceDirection reduceType = getReduceType(curLoop);
      if (reduceType == ReduceDirection::ALL) {
        curLoop->setAttr(kReductionAllLoopAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
      } else {
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

static bool isInlineableParallelNonVectorPointLoop(mlir::scf::ForOp loop) {
  if (!loop || !loop->hasAttr(kParallelAxisAttr) || loop.getNumResults() != 0 || !loop.getRegionIterArgs().empty()) {
    return false;
  }
  auto innerAttr = loop->getAttrOfType<IntegerAttr>(kInnerLoopAttr);
  if (!innerAttr || innerAttr.getInt() != 1) {
    return false;
  }
  // A clamped point loop may be empty on a tail tile, so its upper-bound guard
  // must be preserved instead of unconditionally executing the body once.
  if (loop.getUpperBound().getDefiningOp<affine::AffineMinOp>()) {
    return false;
  }
  return !loop->hasAttr(kVectorAttr) && !loop->hasAttr(kMultiVecLoopAttr) && !loop->hasAttr(kTransposeLoopAttr) &&
         !loop->hasAttr(kBroadcastLoopAttr) && !loop->hasAttr(kNotInnerDimensionBroadcastLoopAttr) &&
         !hasReductionVectorAttr(loop);
}

static void inlinePointLoopWithLowerBound(mlir::scf::ForOp loop, OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(loop);
  loop.getInductionVar().replaceAllUsesWith(loop.getLowerBound());

  SmallVector<Operation *, kSmallVectorSizeSixteen> opsToMove;
  std::transform(loop.getBody()->without_terminator().begin(), loop.getBody()->without_terminator().end(),
                 std::back_inserter(opsToMove), [](Operation &op) { return &op; });
  for (Operation *op : opsToMove) {
    op->moveBefore(loop);
  }
  loop.erase();
}

static void inlineParallelNonVectorPointLoops(func::FuncOp funcOp, OpBuilder &builder) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeEight> loopsToInline;
  funcOp.walk([&loopsToInline](mlir::scf::ForOp loop) {
    if (isInlineableParallelNonVectorPointLoop(loop)) {
      loopsToInline.push_back(loop);
    }
  });

  for (mlir::scf::ForOp loop : loopsToInline) {
    if (isInlineableParallelNonVectorPointLoop(loop)) {
      inlinePointLoopWithLowerBound(loop, builder);
    }
  }
}

static void markBroadcastLoopChainWithVectorAttr(mlir::scf::ForOp vectorTarget, OpBuilder &builder) {
  vectorTarget.walk([&vectorTarget, &builder](mlir::scf::ForOp loop) {
    if (!loop || loop == vectorTarget || hasReductionVectorAttr(loop)) {
      return;
    }
    loop->setAttr(kBroadcastLoopAttr, builder.getI64IntegerAttr(getLoopExtent(loop)));
  });
}

static void markReduceYParentWithVectorAttr(mlir::scf::ForOp loop, OpBuilder &builder) {
  bool restrictToPointLoops = hasPointLoopInAncestorChain(loop);
  for (mlir::scf::ForOp curLoop = loop; curLoop; curLoop = curLoop->getParentOfType<mlir::scf::ForOp>()) {
    if (curLoop->hasAttr(kReductionLoopAttr)) {
      curLoop->removeAttr(kReductionLoopAttr);
      continue;
    }
    if (shouldSkipVectorAttrCandidate(curLoop, restrictToPointLoops, /* skipDeleteLoops= */ true)) {
      continue;
    }
    curLoop->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(curLoop)));
    break;
  }
}

// Mark all innermost scf.for loops with vector attribute.
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder) {
  funcOp->walk([&builder](mlir::scf::ForOp forOp) {
    if (!isInnermostScfLoop(forOp)) {
      if (forOp->hasAttr(kReductionLoopAttr)) {
        forOp->removeAttr(kReductionLoopAttr);
      }
      return;
    }

    auto reduceType = ReduceDirection::UNKNOWN;
    const bool hasReductionAttr = forOp->hasAttr(kReductionLoopAttr);
    // Multi-dim vec is consumed first, including sparse marked ancestors.
    if (hasMultiVecLoopInAncestorChain(forOp)) {
      markMultiVecLoopChainWithVectorAttr(forOp, builder);
      return;
    }
    if (forOp->hasAttr(kTransposeLoopAttr)) {
      markTransposeLoopChainWithVectorAttr(forOp, builder);
      return;
    }
    // set vector attribute to innermost loops
    if (!hasReductionAttr) {
      if (mlir::scf::ForOp transposeAncestor = findOutermostTransposeAncestor(forOp)) {
        markTransposeLoopChainWithVectorAttr(forOp, builder, transposeAncestor);
        return;
      }
      if (mlir::scf::ForOp vectorTarget = findVectorAttrTargetLoop(forOp, /* skipDeleteLoops= */ false)) {
        if (vectorTarget->hasAttr(kBroadcastLoopAttr)) {
          markBroadcastLoopChainWithVectorAttr(vectorTarget, builder);
          vectorTarget->removeAttr(kBroadcastLoopAttr);
        }
        vectorTarget->setAttr(kVectorAttr, builder.getI64IntegerAttr(getLoopExtent(vectorTarget)));
      }
      return;
    }

    forOp->removeAttr(kReductionLoopAttr);
    reduceType = getReduceType(forOp);
    if (reduceType == ReduceDirection::X) {
      forOp->setAttr(kReductionXLoopAttr, builder.getI64IntegerAttr(getLoopExtent(forOp)));
    } else if (reduceType == ReduceDirection::Y) {
      forOp->setAttr(kReductionYLoopAttr, builder.getUnitAttr());
    } else if (reduceType == ReduceDirection::ALL) {
      forOp->setAttr(kReductionAllLoopAttr, builder.getI64IntegerAttr(getLoopExtent(forOp)));
    }

    if (reduceType == ReduceDirection::Y) {
      markReduceYParentWithVectorAttr(forOp->getParentOfType<mlir::scf::ForOp>(), builder);
    }
  });
  funcOp.walk([](mlir::scf::ForOp loop) {
    loop->removeAttr(kMultiVecLoopAttr);
    loop->removeAttr(kTransposeLoopAttr);
  });
}

static void applySkipTilingFallback(func::FuncOp funcOp, OpBuilder &builder) {
  clearAllBroadcastLoopAttrs(funcOp);
  markInnermostLoopsWithVectorAttr(funcOp, builder);
  funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(1));
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
      funcOp->setAttr(kBlockDimAttr, builder.getI64IntegerAttr(getNpuCoreNum(funcOp)));
    }
  }
}

// Inline loops marked with delete when they are guaranteed to execute exactly once.
static void inlineDeleteMarkedLoops(func::FuncOp funcOp, OpBuilder &builder) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeEight> loopsToInline;

  funcOp->walk([&loopsToInline](mlir::scf::ForOp forOp) {
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
    SmallVector<Operation *, kSmallVectorSizeSixteen> opsToMove;
    std::transform(loopBody->without_terminator().begin(), loopBody->without_terminator().end(),
                   std::back_inserter(opsToMove), [](Operation &op) { return &op; });
    std::for_each(opsToMove.begin(), opsToMove.end(), [&loop](Operation *op) { op->moveBefore(loop); });

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
static void buildTilingFunctionSignature(const TilingSignatureParams &params, SmallVector<Type> &argTypes,
                                         SmallVector<Type> &resTypes) {
  const auto &origTy = params.origTy;
  const auto &ctx = params.ctx;
  auto &builder = params.builder;
  const auto &tilingStructMemrefSize = params.tilingStructMemrefSize;
  argTypes.assign(origTy.getInputs().begin(), origTy.getInputs().end());

  auto i64Ty = builder.getI64Type();
  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
  auto memrefTy = MemRefType::get({tilingStructMemrefSize}, i64Ty);

  argTypes.push_back(llvmPtrTy);
  argTypes.push_back(memrefTy);

  resTypes.clear();
}

// Helper: Create and initialize tiling function (set attrs, write tiling key)
static func::FuncOp createAndInitTilingFunc(func::FuncOp originalKernel, ArrayRef<Type> argTypes,
                                            ArrayRef<Type> resTypes, OpBuilder &builder, int64_t tilingKey) {
  auto *ctx = builder.getContext();
  auto loc = originalKernel.getLoc();

  std::string baseName = originalKernel.getSymName().str();
  std::string name = baseName + "_" + formatTilingKeySuffix(tilingKey) + "_tiling_function";

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(originalKernel);

  auto f = builder.create<func::FuncOp>(loc, name, FunctionType::get(ctx, argTypes, resTypes));
  f.addEntryBlock();

  copyHaccIOAttrsFrom(originalKernel, f);
  f->setAttr(hacc::HACCFuncTypeAttr::name, hacc::HACCFuncTypeAttr::get(ctx, hacc::HACCFuncType::HOST));

  unsigned numArgs = f.getNumArguments();
  unsigned keyIdx = numArgs - kTilingFuncReservedArgCount;
  unsigned tilingDataIdx = numArgs - kTilingDataArgOffset;
  setTilingKeyAndDataArgAttrs(f, keyIdx, tilingDataIdx, ctx);

  // Write tiling key.
  OpBuilder b(&f.getBody().front(), f.getBody().front().end());
  Value tilingKeyPtr = f.getArgument(keyIdx);
  Value strategyValue = b.create<arith::ConstantIntOp>(loc, tilingKey, 64);
  b.create<LLVM::StoreOp>(loc, strategyValue, tilingKeyPtr);

  return f;
}

static void createTilingFuncReturn(func::FuncOp tilingFunc, bool writeDummyTilingData) {
  auto loc = tilingFunc.getLoc();
  OpBuilder b(&tilingFunc.getBody().front(), tilingFunc.getBody().front().end());
  if (writeDummyTilingData) {
    Value idx = b.create<arith::ConstantIndexOp>(loc, 0);
    Value val = b.create<arith::ConstantIntOp>(loc, 1, b.getI64Type());
    b.create<memref::StoreOp>(loc, val, tilingFunc.getArgument(tilingFunc.getNumArguments() - 1), ValueRange{idx});
  }
  b.create<func::ReturnOp>(loc);
}

// Helper: Store tile sizes to memref
static LogicalResult storeTileSizesToMemref(func::FuncOp tilingFunc, func::FuncOp originalKernel,
                                            const BandTilingData &bandData, OpBuilder &builder) {
  auto &bands = bandData.bands;
  const auto &allBandTileSizes = bandData.allBandTileSizes;
  const auto &allBandConstraintMaxs = bandData.allBandConstraintMaxs;
  const auto &allBandDynamicMappings = bandData.allBandDynamicMappings;
  auto loc = tilingFunc.getLoc();
  unsigned numArgs = tilingFunc.getNumArguments();
  unsigned tilingDataIdx = numArgs - 1;

  OpBuilder b(&tilingFunc.getBody().front(), tilingFunc.getBody().front().end());
  Value dataMem = tilingFunc.getArgument(tilingDataIdx);
  auto i64Ty = builder.getI64Type();
  int64_t dynamicTileCoreNum = getNpuCoreNum(originalKernel);

  size_t memrefOffset = 0;

  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &bandConstraintMaxs = allBandConstraintMaxs[bandIdx];
    const auto &bandDynamicMapping = allBandDynamicMappings[bandIdx];
    size_t bandSize = bands[bandIdx].size();

    SmallVector<unsigned, kSmallVectorSizeSix> parallelDims;
    collectParallelPrefixDims(bands[bandIdx], bandTileSizes, parallelDims);
    SmallVector<Value, kSmallVectorSizeSix> parallelOuterTiles;
    if (failed(emitParallelOuterTilesForBand({tilingFunc, originalKernel, bands[bandIdx], parallelDims, b},
                                             parallelOuterTiles))) {
      return failure();
    }

    // Cache runtime first-level step values (as i64) for later levels.
    SmallVector<Value, kSmallVectorSizeSix> firstLevelStepI64Cache(bandSize, Value());

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, memrefOffset);
      unsigned tileSize = bandTileSizes[tileIdx];

      // Calculate level and dimIdx for current tile
      size_t level = tileIdx / bandSize;
      size_t dimIdx = tileIdx % bandSize;

      if (level == 0 && dimIdx < parallelOuterTiles.size() && parallelOuterTiles[dimIdx]) {
        Value stepI64 = b.create<arith::IndexCastOp>(loc, i64Ty, parallelOuterTiles[dimIdx]);
        b.create<memref::StoreOp>(loc, stepI64, dataMem, ValueRange{idx});
        firstLevelStepI64Cache[dimIdx] = stepI64;
      } else if (tileSize == static_cast<unsigned>(-1)) {
        int constraintMax = (tileIdx < bandConstraintMaxs.size()) ? bandConstraintMaxs[tileIdx] : 0;
        Value step;
        if (level == 0) {
          if (failed(emitLoopTripCountInTilingFunc({tilingFunc, originalKernel, bands[bandIdx][dimIdx], b}, step))) {
            return failure();
          }
        } else {
          // Dynamic tile size: compute min(constraintMax, dim) and store. The
          // constraintMax comes from TilingStrategy's constraint upper bound.
          const auto &mapping = bandDynamicMapping[tileIdx];
          if (!mapping.upperBound) {
            tilingFunc.emitError("dynamic axis upper bound missing for tile index " + std::to_string(tileIdx));
            return failure();
          }

          Value dim = cloneUpperBoundDefinition(mapping.upperBound, originalKernel, tilingFunc, b);
          if (!dim) {
            tilingFunc.emitError("failed to clone dynamic axis upper bound for tile index " + std::to_string(tileIdx));
            return failure();
          }
          step = emitDynamicTilePerDim({loc, b, dim, constraintMax, dynamicTileCoreNum});
        }

        Value stepI64 = b.create<arith::IndexCastOp>(loc, i64Ty, step);
        b.create<memref::StoreOp>(loc, stepI64, dataMem, ValueRange{idx});

        // Cache first level step for later use
        if (level == 0) {
          firstLevelStepI64Cache[dimIdx] = stepI64;
        }
      } else {
        if (level >= 1 && firstLevelStepI64Cache[dimIdx]) {
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
static LogicalResult createTilingFuncDefault(const CreateTilingFuncParams &params, func::FuncOp &tilingFunc) {
  auto &originalKernel = params.originalKernel;
  auto &builder = params.builder;
  const auto &isStaticShape = params.isStaticShape;
  const auto &tilingKey = params.tilingKey;
  const auto &metadata = params.metadata;
  auto *mlirCtx = builder.getContext();
  auto origTy = originalKernel.getFunctionType();
  int64_t tilingStructMemrefSize = 1;
  bool hasPresetMetadata = metadata && !metadata->bandTileSizes.empty();
  if (metadata && !hasPresetMetadata) {
    metadata->clear();
  }

  std::vector<LeafBranchBandPlan> leafBranchPlans;
  bool hasUnsupportedTreeShape = false;
  if (failed(buildLeafBranchBandPlans(originalKernel, leafBranchPlans, hasUnsupportedTreeShape))) {
    return failure();
  }
  if (failed(rejectDynamicMultiTopLevelBands(originalKernel, isStaticShape, leafBranchPlans))) {
    return failure();
  }
  std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bandsToUse;
  collectRepresentativeBands(leafBranchPlans, bandsToUse);
  bool skipTiling = hasUnsupportedTreeShape || leafBranchPlans.size() > 1;

  // Step 1: Dynamic-shape path computes tile metadata to derive tiling struct size.
  std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> allBandTileSizes;
  std::vector<SmallVector<int, kSmallVectorSizeSix>> allBandConstraintMaxs;
  if (!isStaticShape) {
    if (!skipTiling) {
      if (!leafBranchPlans.empty()) {
        preprocessLoopAttrsForTileCalculation(originalKernel, leafBranchPlans.front());
      }
      [[maybe_unused]] auto clearNotInnerBroadcastGuard =
        llvm::make_scope_exit([&kernelRef = originalKernel] { clearNotInnerDimensionBroadcastLoopAttr(kernelRef); });
      (void)clearNotInnerBroadcastGuard;
      if (hasPresetMetadata) {
        allBandTileSizes = metadata->bandTileSizes;
        allBandConstraintMaxs = metadata->bandConstraintMaxs;
      } else {
        size_t levelToTile = 0;
        if (failed(calculateTileSizesForBands(originalKernel, true,
                                              BandTilingOutput{bandsToUse, allBandTileSizes, allBandConstraintMaxs},
                                              levelToTile))) {
          return failure();
        }
      }
      if (metadata && !hasPresetMetadata) {
        metadata->tilingKey = tilingKey;
        metadata->bandTileSizes = allBandTileSizes;
        metadata->bandConstraintMaxs = allBandConstraintMaxs;
        captureMultiVecMasks(bandsToUse, *metadata);
      }

      tilingStructMemrefSize =
        std::accumulate(allBandTileSizes.begin(), allBandTileSizes.end(), int64_t{0},
                        [](int64_t acc, const SmallVector<unsigned, kSmallVectorSizeSix> &bandTileSizes) {
                          return acc + static_cast<int64_t>(bandTileSizes.size());
                        });
      if (tilingStructMemrefSize <= 0) {
        tilingStructMemrefSize = 1;
      }
    } else {
      bandsToUse.clear();
    }
  }

  // Step 2: Build function signature
  SmallVector<Type> argTypes;
  SmallVector<Type> resTypes;
  mlirCtx->getOrLoadDialect<LLVM::LLVMDialect>();
  buildTilingFunctionSignature({origTy, mlirCtx, builder, tilingStructMemrefSize}, argTypes, resTypes);

  // Step 3: Create and initialize tiling function.
  func::FuncOp f = createAndInitTilingFunc(originalKernel, argTypes, resTypes, builder, tilingKey);

  // Step 4: Static-shape or no-band path doesn't need memref store in create stage.
  if (isStaticShape || bandsToUse.empty()) {
    createTilingFuncReturn(f, !isStaticShape);
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
  if (failed(storeTileSizesToMemref(
        f, originalKernel, BandTilingData{bandsToUse, allBandTileSizes, allBandConstraintMaxs, allBandDynamicMappings},
        builder))) {
    return failure();
  }

  tilingFunc = f;
  return success();
}

// Public interface functions
LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape) {
  return createTilingFunctions(originalKernel, builder, out, isStaticShape, static_cast<TilingMetadata *>(nullptr));
}

LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape,
                                    TilingMetadata *metadata) {
  out.clear();
  func::FuncOp tilingFunc;
  if (failed(
        createTilingFuncDefault({originalKernel, builder, isStaticShape, kDefaultTilingKey, metadata}, tilingFunc))) {
    return failure();
  }
  out[kDefaultTilingKey] = tilingFunc;
  return success();
}

LogicalResult createTilingFunctions(func::FuncOp originalKernel, OpBuilder &builder,
                                    DenseMap<int64_t, func::FuncOp> &out, bool isStaticShape,
                                    TilingMetadataMap *metadataByKey) {
  out.clear();
  if (metadataByKey) {
    metadataByKey->clear();
  }

  TilingMetadata key0Metadata;
  func::FuncOp tilingFunc;
  if (failed(createTilingFuncDefault(
        {originalKernel, builder, isStaticShape, kDefaultTilingKey, metadataByKey ? &key0Metadata : nullptr},
        tilingFunc))) {
    return failure();
  }
  out[kDefaultTilingKey] = tilingFunc;
  if (metadataByKey && !key0Metadata.empty()) {
    (*metadataByKey)[kDefaultTilingKey] = key0Metadata;
  }

  std::vector<LeafBranchBandPlan> plans;
  bool hasUnsupportedTreeShape = false;
  std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bandsToUse;
  if (!isStaticShape && metadataByKey &&
      succeeded(buildLeafBranchBandPlans(originalKernel, plans, hasUnsupportedTreeShape)) && !hasUnsupportedTreeShape) {
    collectRepresentativeBands(plans, bandsToUse);
  }

  TilingMetadata key1Metadata;
  if (!isStaticShape && metadataByKey &&
      buildTwoDimDynamicVectorMetadata(originalKernel, bandsToUse, key0Metadata, key1Metadata)) {
    func::FuncOp key1TilingFunc;
    if (failed(createTilingFuncDefault(
          {originalKernel, builder, isStaticShape, kTwoDimDynamicVectorTilingKey, &key1Metadata}, key1TilingFunc))) {
      return failure();
    }
    out[kTwoDimDynamicVectorTilingKey] = key1TilingFunc;
    (*metadataByKey)[kTwoDimDynamicVectorTilingKey] = key1Metadata;
  }
  return success();
}

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
        // Keep the real upper bound for create-side cloning. Do not invent a
        // memref.dim fallback; callers that need direct memref mapping should fail explicitly.
        bandDynamicMapping.push_back({UINT_MAX, UINT_MAX, upperBound});
      }
    } else {
      // Static tile size
      bandDynamicMapping.push_back({UINT_MAX, UINT_MAX, Value()});
    }
  }

  return bandDynamicMapping;
}

static void initializeEmptyMultiVecMasks(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                         TilingMetadata &metadata) {
  metadata.bandMultiVecAxisMasks.clear();
  metadata.bandMultiVecAxisMasks.reserve(bands.size());
  for (const auto &band : bands) {
    metadata.bandMultiVecAxisMasks.emplace_back(band.size(), static_cast<char>(false));
  }
}

static void captureMultiVecMasks(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                 TilingMetadata &metadata) {
  initializeEmptyMultiVecMasks(bands, metadata);
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    for (size_t axisIdx = 0; axisIdx < bands[bandIdx].size(); ++axisIdx) {
      metadata.bandMultiVecAxisMasks[bandIdx][axisIdx] =
        static_cast<char>(bands[bandIdx][axisIdx]->hasAttr(kMultiVecLoopAttr));
    }
  }
}

static bool isZeroBasedUnitStepLoop(mlir::scf::ForOp loop) {
  std::optional<int64_t> lb = getConstantIndexValue(loop.getLowerBound());
  std::optional<int64_t> step = getConstantIndexValue(loop.getStep());
  return lb && step && *lb == 0 && *step == 1;
}

static int64_t computeTwoDimDynamicVectorCap(func::FuncOp originalKernel, const DynamicAxisMapping &selectorMapping,
                                             int64_t singleAxisCap) {
  if (singleAxisCap <= 1 || selectorMapping.inputMemrefIndex >= originalKernel.getNumArguments()) {
    return 0;
  }
  auto memrefTy = dyn_cast<MemRefType>(originalKernel.getArgument(selectorMapping.inputMemrefIndex).getType());
  if (!memrefTy) {
    return 0;
  }
  int64_t alignUnit = std::max<int64_t>(akg::getBishengStrideAlignTargetForBits(akg::getElementBitWidth(memrefTy)), 1);
  int64_t root = floorSqrtInt64(singleAxisCap);
  int64_t cap = (root / alignUnit) * alignUnit;
  return (cap > 1 && multiplyAndCapPositive(cap, cap) <= singleAxisCap) ? cap : 0;
}

static bool buildTwoDimDynamicVectorMetadata(func::FuncOp originalKernel,
                                             ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bandsToUse,
                                             const TilingMetadata &baseMetadata, TilingMetadata &metadata) {
  if (bandsToUse.size() != 1 || baseMetadata.bandTileSizes.size() != 1 || baseMetadata.bandConstraintMaxs.size() != 1) {
    return false;
  }

  const auto &band = bandsToUse.front();
  const auto &baseTileSizes = baseMetadata.bandTileSizes.front();
  size_t bandSize = band.size();
  if (bandSize < kSmallVectorSizeTwo || baseTileSizes.size() != bandSize * kSmallVectorSizeTwo ||
      !isZeroBasedUnitStepLoop(band[bandSize - kSmallVectorSizeTwo]) || !isZeroBasedUnitStepLoop(band.back())) {
    return false;
  }
  if (std::any_of(band.begin(), band.end(), [](mlir::scf::ForOp loop) { return loop->hasAttr(kTransposeLoopAttr); })) {
    return false;
  }
  if (band.back()->hasAttr(kReductionLoopAttr) && getReduceType(band.back()) == ReduceDirection::Y) {
    return false;
  }

  unsigned dynamicTile = static_cast<unsigned>(-1);
  size_t outerDim0 = bandSize - kSmallVectorSizeTwo;
  size_t outerDim1 = bandSize - kSmallVectorSizeOne;
  size_t innerDim0 = bandSize + outerDim0;
  size_t innerDim1 = bandSize + outerDim1;
  if (baseTileSizes[outerDim0] != dynamicTile || baseTileSizes[outerDim1] != dynamicTile ||
      baseTileSizes[innerDim1] == dynamicTile) {
    return false;
  }

  std::vector<DynamicAxisMapping> mappings = buildDynamicAxisMappingForBand(band, baseTileSizes, originalKernel);
  if (mappings.size() != baseTileSizes.size()) {
    return false;
  }
  const DynamicAxisMapping &selector = mappings[outerDim1];
  if (selector.inputMemrefIndex == UINT_MAX || selector.dimIndex == UINT_MAX) {
    return false;
  }

  int64_t cap = computeTwoDimDynamicVectorCap(originalKernel, selector, baseTileSizes[innerDim1]);
  if (cap <= 1) {
    return false;
  }

  metadata = baseMetadata;
  metadata.tilingKey = kTwoDimDynamicVectorTilingKey;
  metadata.selectorInputIndex = selector.inputMemrefIndex;
  metadata.selectorDimIndex = selector.dimIndex;
  metadata.selectorLimit = cap;
  metadata.selectorTrueKey = kTwoDimDynamicVectorTilingKey;
  metadata.selectorFalseKey = kDefaultTilingKey;
  metadata.bandTileSizes[0][innerDim0] = static_cast<unsigned>(cap);
  metadata.bandTileSizes[0][innerDim1] = static_cast<unsigned>(cap);
  initializeEmptyMultiVecMasks(bandsToUse, metadata);
  metadata.bandMultiVecAxisMasks[0][outerDim0] = static_cast<char>(true);
  metadata.bandMultiVecAxisMasks[0][outerDim1] = static_cast<char>(true);
  return true;
}

// Emit per-dim dynamic tile-size: min(constraintMax, dim) if constraintMax > 0,
// else ceildiv(dim, dynamicTileCoreNum).
static Value emitDynamicTilePerDim(const DynamicTilePerDimParams &params) {
  const auto &loc = params.loc;
  auto &builder = params.builder;
  const auto &dim = params.dim;
  const auto &constraintMax = params.constraintMax;
  const auto &dynamicTileCoreNum = params.dynamicTileCoreNum;
  if (constraintMax > 0) {
    Value constraintMaxVal = builder.create<arith::ConstantIndexOp>(loc, constraintMax);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, constraintMaxVal, dim);
    return builder.create<arith::SelectOp>(loc, cmp, constraintMaxVal, dim);
  }
  Value coreNum = builder.create<arith::ConstantIndexOp>(loc, dynamicTileCoreNum);
  return builder.create<arith::CeilDivSIOp>(loc, dim, coreNum);
}

// Helper: Compute dynamic tile size value (ceildiv(dim, coreNum))
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 const KernelBuildContext &kbc, Value &result) {
  auto &originalKernel = kbc.originalKernel;
  const auto &loc = kbc.loc;
  auto &builder = kbc.builder;
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
  result = emitDynamicTilePerDim({loc, builder, dim, constraintMax, getNpuCoreNum(originalKernel)});
  return success();
}

// Helper: Prepare tile sizes for static shape
static LogicalResult prepareTileSizesForStaticShape(
  const KernelBuildContext &kbc, const BandTilingOutput &bandData,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues) {
  auto &originalKernel = kbc.originalKernel;
  const auto &loc = kbc.loc;
  auto &builder = kbc.builder;
  auto &constantCache = kbc.constantCache;
  const auto &bandsToUse = bandData.bandsToUse;
  const auto &allBandTileSizes = bandData.allBandTileSizes;
  const auto &allBandConstraintMaxs = bandData.allBandConstraintMaxs;
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
    SmallVector<Value, kSmallVectorSizeSix> tileSizeValues;

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      unsigned tileSize = bandTileSizes[tileIdx];
      Value tileSizeValue;

      if (tileSize == static_cast<unsigned>(-1)) {
        // Dynamic tile size: use constraint upper bound
        const auto &mapping = bandDynamicMapping[tileIdx];
        int constraintMax = (tileIdx < bandConstraintMaxs.size()) ? bandConstraintMaxs[tileIdx] : 0;
        if (failed(computeDynamicTileSizeValue(mapping, constraintMax,
                                               KernelBuildContext{originalKernel, loc, builder, constantCache},
                                               tileSizeValue))) {
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
static LogicalResult prepareTileSizesFromMemref(
  const PrepareTileSizesFromMemrefParams &params,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues) {
  auto &originalKernel = params.originalKernel;
  const auto &bands = params.bands;
  const auto &loc = params.loc;
  auto &builder = params.builder;
  const auto &allBandTileSizesInt = params.allBandTileSizesInt;
  auto args = originalKernel.getArguments();
  if (args.empty()) {
    originalKernel.emitError("originalKernel must have at least one argument (tileSizesMemref)");
    return failure();
  }

  Value tileSizesMemref = args.back();
  auto memrefType = dyn_cast<MemRefType>(tileSizesMemref.getType());
  if (!memrefType || memrefType.getRank() != 1 || !memrefType.getElementType().isInteger(kI64BitWidth)) {
    std::string typeStr;
    llvm::raw_string_ostream os(typeStr);
    os << tileSizesMemref.getType();
    originalKernel.emitError("Last argument (tileSizesMemref) must be memref<?xi64>, got: " + typeStr);
    return failure();
  }

  size_t memrefOffset = 0;

  if (bands.size() != allBandTileSizesInt.size()) {
    originalKernel.emitError("inconsistent band metadata when loading tile sizes from memref");
    return failure();
  }

  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &band = bands[bandIdx];
    const auto &bandTileSizes = allBandTileSizesInt[bandIdx];
    size_t bandSize = band.size();
    size_t bandTileSizesCount = bandTileSizes.size();
    if (bandSize == 0 || bandTileSizesCount == 0 || bandTileSizesCount % bandSize != 0) {
      originalKernel.emitError("invalid dynamic tile metadata for band " + std::to_string(bandIdx));
      return failure();
    }

    SmallVector<Value, kSmallVectorSizeSix> tileSizeValues;
    tileSizeValues.reserve(bandTileSizesCount);

    for (size_t i = 0; i < bandTileSizesCount; ++i) {
      Value idx = builder.create<arith::ConstantIndexOp>(loc, memrefOffset + i);
      Value loaded = builder.create<memref::LoadOp>(loc, tileSizesMemref, ValueRange{idx});
      Value tileSizeIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), loaded);
      tileSizeValues.push_back(tileSizeIndex);
    }

    allTileSizeValues.push_back(tileSizeValues);
    memrefOffset += bandTileSizesCount;
  }

  return success();
}

static bool hasValidMultiVecMaskLayout(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                       ArrayRef<SmallVector<char, kSmallVectorSizeSix>> masks) {
  if (masks.empty()) {
    return true;
  }
  if (masks.size() != bands.size()) {
    return false;
  }
  for (auto [band, mask] : llvm::zip_equal(bands, masks)) {
    if (mask.size() != band.size()) {
      return false;
    }
  }
  return true;
}

static void applyMetadataMultiVecMasks(ArrayRef<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bands,
                                       ArrayRef<SmallVector<char, kSmallVectorSizeSix>> masks) {
  if (masks.empty()) {
    return;
  }
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &band = bands[bandIdx];
    const auto &mask = masks[bandIdx];
    for (size_t loopIdx = 0; loopIdx < band.size(); ++loopIdx) {
      mlir::scf::ForOp loop = band[loopIdx];
      if (mask[loopIdx]) {
        loop->setAttr(kMultiVecLoopAttr, UnitAttr::get(loop.getContext()));
      } else {
        loop->removeAttr(kMultiVecLoopAttr);
      }
    }
  }
}

static LogicalResult wrapFunctionBodyWithFor(func::FuncOp func, OpBuilder &builder) {
  Location loc = func.getLoc();

  if (!func.getBody().hasOneBlock()) {
    func.emitError("wrapFunctionBodyWithFor currently supports only single-block functions");
    return failure();
  }

  Block &entryBlock = func.getBody().front();
  Operation *terminator = entryBlock.getTerminator();
  if (terminator == nullptr) {
    func.emitError("entry block has no terminator");
    return failure();
  }

  llvm::SmallVector<Operation *, kSmallVectorSizeSixteen> opsToMove;
  for (auto it = entryBlock.begin(); it != Block::iterator(terminator); ++it) {
    opsToMove.push_back(&*it);
  }
  builder.setInsertionPointToStart(&entryBlock);

  Value lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step);
  Block *forBody = forOp.getBody();
  Operation *forTerminator = forBody->getTerminator();
  for (Operation *op : opsToMove) {
    op->moveBefore(forTerminator);
  }
  return success();
}

static LogicalResult prepareLeafBranchPlansForApply(func::FuncOp originalKernel, OpBuilder &builder,
                                                    std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                                    bool isStaticShape, bool &shouldReturnEarly) {
  shouldReturnEarly = false;
  bool hasUnsupportedTreeShape = false;
  if (failed(buildLeafBranchBandPlans(originalKernel, leafBranchPlans, hasUnsupportedTreeShape))) {
    return failure();
  }
  if (failed(rejectDynamicMultiTopLevelBands(originalKernel, isStaticShape, leafBranchPlans))) {
    return failure();
  }

  // Remaining unsupported shapes and static multiple top-level bands keep stage-1 skip behavior.
  if (hasUnsupportedTreeShape || leafBranchPlans.size() > 1) {
    applySkipTilingFallback(originalKernel, builder);
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

    if (hasUnsupportedTreeShape || leafBranchPlans.size() > 1) {
      applySkipTilingFallback(originalKernel, builder);
      shouldReturnEarly = true;
      return success();
    }
  }

  return success();
}

static LogicalResult prepareTileMetadataForApply(
  const PrepareTileMetadataParams &params, const BuildContext &bc,
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> &allTileSizeValues) {
  auto &originalKernel = params.originalKernel;
  const auto &isStaticShape = params.isStaticShape;
  auto &bandOutput = params.bandOutput;
  const auto &metadata = params.metadata;
  auto &builder = bc.builder;
  const auto &loc = bc.loc;
  auto &bandsToUse = bandOutput.bandsToUse;
  auto &allBandTileSizesInt = bandOutput.allBandTileSizes;
  auto &allBandConstraintMaxs = bandOutput.allBandConstraintMaxs;
  bool usedCachedMetadata = false;
  if (!isStaticShape && metadata && !metadata->empty()) {
    if (metadata->bandTileSizes.size() == bandsToUse.size() &&
        metadata->bandConstraintMaxs.size() == bandsToUse.size() &&
        hasValidMultiVecMaskLayout(bandsToUse, metadata->bandMultiVecAxisMasks)) {
      usedCachedMetadata = true;
      for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
        size_t bandSize = bandsToUse[bandIdx].size();
        if (bandSize == 0 || metadata->bandTileSizes[bandIdx].empty() ||
            metadata->bandTileSizes[bandIdx].size() % bandSize != 0 ||
            metadata->bandTileSizes[bandIdx].size() != metadata->bandConstraintMaxs[bandIdx].size()) {
          usedCachedMetadata = false;
          break;
        }
      }
      if (usedCachedMetadata) {
        allBandTileSizesInt = metadata->bandTileSizes;
        allBandConstraintMaxs = metadata->bandConstraintMaxs;
        applyMetadataMultiVecMasks(bandsToUse, metadata->bandMultiVecAxisMasks);
      }
    }
  }

  if (!usedCachedMetadata) {
    // Always calculate tile sizes first to get unsigned values (with constraint max for dynamic axes)
    size_t levelToTile = 0;
    if (failed(calculateTileSizesForBands(originalKernel, true,
                                          BandTilingOutput{bandsToUse, allBandTileSizesInt, allBandConstraintMaxs},
                                          levelToTile))) {
      return failure();
    }
  }

  // Then prepare Value representations based on shape type.
  allTileSizeValues.clear();
  allTileSizeValues.reserve(bandsToUse.size());

  if (isStaticShape) {
    // Static shape: materialize Value tile sizes from the already solved tile metadata.
    if (failed(prepareTileSizesForStaticShape(KernelBuildContext{originalKernel, loc, builder, bc.constantCache},
                                              BandTilingOutput{bandsToUse, allBandTileSizesInt, allBandConstraintMaxs},
                                              allTileSizeValues))) {
      return failure();
    }
    return success();
  }

  // Dynamic shape: load tile sizes from memref (but we already have unsigned values)
  return prepareTileSizesFromMemref({originalKernel, bandsToUse, loc, builder, allBandTileSizesInt}, allTileSizeValues);
}

static LogicalResult applySingleLinearBandDecoupled(const SingleLinearBandParams &params, int64_t &nextNodeId) {
  const auto &originalKernel = params.originalKernel;
  auto &builder = params.builder;
  const auto &bandIdx = params.bandIdx;
  const auto &band = params.band;
  const auto &tileSizeValues = params.tileSizeValues;
  const auto &tileSizesInt = params.tileSizesInt;
  const auto &useRuntimeTileCounts = params.useRuntimeTileCounts;
  if (band.empty() || tileSizeValues.size() != tileSizesInt.size() || tileSizeValues.size() % band.size() != 0) {
    return emitTilingFailure(originalKernel, "invalid linear-band tiling metadata for band " + Twine(bandIdx));
  }

  auto bandSize = static_cast<unsigned>(band.size());
  auto tileLevels = static_cast<unsigned>(tileSizeValues.size() / bandSize);
  if (tileLevels == 0) {
    return emitTilingFailure(originalKernel, "empty tile levels for linear band " + Twine(bandIdx));
  }

  SmallVector<unsigned, kSmallVectorSizeSix> parallelDims;
  collectParallelPrefixDims(band, tileSizesInt, parallelDims);
  int64_t parallelUseCore = 0;
  mlir::scf::ForOp parallelMapLoop;
  mlir::scf::ForOp parallelTileLoop;
  SmallVector<Value, kSmallVectorSizeSix> parallelTileCoordByDim;
  createParallelMapAndTileLoop(
    {originalKernel, band, tileSizeValues, tileSizesInt, parallelDims, useRuntimeTileCounts, builder},
    {parallelMapLoop, parallelTileLoop, parallelTileCoordByDim, parallelUseCore});

  // Decouple linear chain tiling by axis (outer->inner):
  // i{j{k}} -> i1,i2,i3{j1,j2,j3{k1,k2,k3}}.
  if (failed(applyDecoupledAxisTiling({originalKernel, builder, "linear", band, tileSizeValues, tileSizesInt,
                                       parallelMapLoop, parallelTileLoop, parallelTileCoordByDim, useRuntimeTileCounts},
                                      nextNodeId))) {
    return failure();
  }

  return success();
}

static void buildLeafBranchTileSlices(const LeafBranchTileSlicesInput &in, const LeafBranchTileSlicesOutput &out) {
  auto &plan = in.plan;
  const auto &band = in.band;
  const auto &tileSizeValues = in.tileSizeValues;
  const auto &tileSizesInt = in.tileSizesInt;
  const auto &bandSize = in.bandSize;
  const auto &tileLevels = in.tileLevels;
  auto &prefixBand = out.prefixBand;
  auto &prefixTileValues = out.prefixTileValues;
  auto &prefixTileSizesInt = out.prefixTileSizesInt;
  // Shared prefix axes are tiled first (e.g. i1,i2,i3), then each leaf axis is tiled independently.
  for (unsigned dim = 0; dim < plan.representativeLeafDim; ++dim) {
    prefixBand.push_back(band[dim]);
    band[dim]->removeAttr(kMultiVecLoopAttr);
  }

  prefixTileValues.reserve(tileLevels * plan.representativeLeafDim);
  prefixTileSizesInt.reserve(tileLevels * plan.representativeLeafDim);
  for (unsigned level = 0; level < tileLevels; ++level) {
    for (unsigned dim = 0; dim < plan.representativeLeafDim; ++dim) {
      size_t idx = static_cast<size_t>(level) * bandSize + dim;
      prefixTileValues.push_back(tileSizeValues[idx]);
      prefixTileSizesInt.push_back(tileSizesInt[idx]);
    }
  }
}

static LogicalResult collectAndTagLeafBranchLoops(const CollectTagLeafBranchParams &params, int64_t &nextNodeId,
                                                  SmallVector<int64_t, kSmallVectorSizeSix> &branchRootNodeIds) {
  const auto &originalKernel = params.originalKernel;
  auto &builder = params.builder;
  const auto &bandIdx = params.bandIdx;
  auto &plan = params.plan;
  const auto &band = params.band;
  mlir::scf::ForOp representativeLeaf = band[plan.representativeLeafDim];
  auto branchPoint = representativeLeaf->getParentOfType<mlir::scf::ForOp>();
  if (!branchPoint) {
    return emitTilingFailure(originalKernel, "failed to locate leaf-branch parent loop for band " + Twine(bandIdx));
  }

  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> branchRoots;
  collectDirectChildLoopsInOrder(branchPoint, branchRoots);
  if (branchRoots.size() < kSmallVectorSizeTwo || branchRoots.size() != plan.branchBands.size() ||
      llvm::find(branchRoots, representativeLeaf) == branchRoots.end()) {
    return emitTilingFailure(originalKernel, "invalid leaf-branch structure for band " + Twine(bandIdx));
  }

  branchRootNodeIds.clear();
  branchRootNodeIds.reserve(branchRoots.size());
  for (auto [branchIdx, branchRoot] : llvm::enumerate(branchRoots)) {
    const auto &branchBand = plan.branchBands[branchIdx];
    for (auto [dim, branchLoop] : llvm::enumerate(branchBand)) {
      size_t representativeDim = plan.representativeLeafDim + dim;
      bool usesMultiVec = representativeDim < band.size() && band[representativeDim]->hasAttr(kMultiVecLoopAttr);
      if (usesMultiVec) {
        branchLoop->setAttr(kMultiVecLoopAttr, builder.getUnitAttr());
      } else {
        branchLoop->removeAttr(kMultiVecLoopAttr);
      }
    }
    int64_t nodeId = nextNodeId++;
    branchRoot->setAttr(kTreeNodeIdAttr, builder.getI64IntegerAttr(nodeId));
    branchBand.back()->setAttr(kTreeLeafAttr, builder.getUnitAttr());
    branchRootNodeIds.push_back(nodeId);
  }
  return success();
}

static void buildBranchTileValues(const ApplyLeafBandsParams &params, unsigned branchSize,
                                  SmallVectorImpl<Value> &branchTileValues,
                                  SmallVectorImpl<unsigned> &branchTileSizesInt) {
  branchTileValues.clear();
  branchTileSizesInt.clear();
  branchTileValues.reserve(params.tileLevels * branchSize);
  branchTileSizesInt.reserve(params.tileLevels * branchSize);
  for (unsigned level = 0; level < params.tileLevels; ++level) {
    for (unsigned dim = 0; dim < branchSize; ++dim) {
      size_t idx =
        static_cast<size_t>(level) * params.bandSize + params.plan.representativeLeafDim + static_cast<size_t>(dim);
      branchTileValues.push_back(params.tileSizeValues[idx]);
      branchTileSizesInt.push_back(params.tileSizesInt[idx]);
    }
  }
}

static LogicalResult applyLeafBandsByNodeId(const ApplyLeafBandsParams &params) {
  auto &originalKernel = params.originalKernel;
  auto &builder = params.builder;
  const auto &bandIdx = params.bandIdx;
  for (auto [branchIdx, nodeId] : llvm::enumerate(params.branchRootNodeIds)) {
    mlir::scf::ForOp activeRoot;
    if (failed(findUniqueLoopByTreeNodeId(originalKernel, nodeId, activeRoot))) {
      return emitTilingFailure(originalKernel, "failed to locate unique branch loop by temporary node id");
    }
    if (branchIdx >= params.plan.branchBands.size()) {
      return emitTilingFailure(originalKernel, "invalid branch metadata for band " + Twine(bandIdx));
    }
    SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> activeBand;
    if (failed(collectSingleChain(activeRoot, activeBand)) ||
        activeBand.size() != params.plan.branchBands[branchIdx].size()) {
      return emitTilingFailure(originalKernel, "failed to rebuild branch chain for band " + Twine(bandIdx));
    }

    SmallVector<Value, kSmallVectorSizeSix> branchTileValues;
    SmallVector<unsigned, kSmallVectorSizeSix> branchTileSizesInt;
    buildBranchTileValues(params, activeBand.size(), branchTileValues, branchTileSizesInt);
    if (failed(applyDecoupledAxisTiling({originalKernel,
                                         builder,
                                         "branch",
                                         activeBand,
                                         branchTileValues,
                                         branchTileSizesInt,
                                         mlir::scf::ForOp(),
                                         mlir::scf::ForOp(),
                                         {},
                                         params.useRuntimeTileCounts},
                                        params.nextNodeId))) {
      return emitTilingFailure(originalKernel, "Failed to apply branch tiling for band " + Twine(bandIdx));
    }
  }
  return success();
}

static LogicalResult applySingleLeafBranchBandDecoupled(const SingleLeafBranchBandParams &params, int64_t &nextNodeId) {
  const auto &originalKernel = params.originalKernel;
  auto &builder = params.builder;
  const auto &bandIdx = params.bandIdx;
  const auto &plan = params.plan;
  const auto &band = params.band;
  const auto &tileSizeValues = params.tileSizeValues;
  const auto &tileSizesInt = params.tileSizesInt;
  const auto &useRuntimeTileCounts = params.useRuntimeTileCounts;
  if (band.empty() || plan.representativeLeafDim == 0 || plan.representativeLeafDim >= band.size() ||
      tileSizeValues.size() != tileSizesInt.size() || tileSizeValues.size() % band.size() != 0) {
    return emitTilingFailure(originalKernel, "invalid leaf-branch tiling metadata for band " + Twine(bandIdx));
  }

  auto bandSize = static_cast<unsigned>(band.size());
  auto tileLevels = static_cast<unsigned>(tileSizeValues.size() / bandSize);
  if (tileLevels == 0) {
    return emitTilingFailure(originalKernel, "empty tile levels for leaf-branch band " + Twine(bandIdx));
  }

  SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix> prefixBand;
  SmallVector<Value, kSmallVectorSizeSix> prefixTileValues;
  SmallVector<unsigned, kSmallVectorSizeSix> prefixTileSizesInt;
  buildLeafBranchTileSlices({plan, band, tileSizeValues, tileSizesInt, bandSize, tileLevels},
                            {prefixBand, prefixTileValues, prefixTileSizesInt});

  int64_t prefixParallelUseCore = 0;
  SmallVector<unsigned, kSmallVectorSizeSix> prefixParallelDims;
  collectParallelPrefixDims(prefixBand, prefixTileSizesInt, prefixParallelDims);
  mlir::scf::ForOp prefixParallelMapLoop;
  mlir::scf::ForOp prefixParallelTileLoop;
  SmallVector<Value, kSmallVectorSizeSix> prefixParallelTileCoordByDim;
  createParallelMapAndTileLoop(
    {originalKernel, prefixBand, prefixTileValues, prefixTileSizesInt, prefixParallelDims, useRuntimeTileCounts,
     builder},
    {prefixParallelMapLoop, prefixParallelTileLoop, prefixParallelTileCoordByDim, prefixParallelUseCore});

  SmallVector<int64_t, kSmallVectorSizeSix> branchRootNodeIds;
  if (failed(
        collectAndTagLeafBranchLoops({originalKernel, builder, bandIdx, plan, band}, nextNodeId, branchRootNodeIds))) {
    return failure();
  }

  if (failed(applyDecoupledAxisTiling(
        {originalKernel, builder, "shared-prefix", prefixBand, prefixTileValues, prefixTileSizesInt,
         prefixParallelMapLoop, prefixParallelTileLoop, prefixParallelTileCoordByDim, useRuntimeTileCounts},
        nextNodeId))) {
    return failure();
  }

  return applyLeafBandsByNodeId({originalKernel, builder, bandIdx, plan, branchRootNodeIds, tileSizeValues,
                                 tileSizesInt, bandSize, tileLevels, useRuntimeTileCounts, nextNodeId});
}

static LogicalResult applyAllBandsWithPlans(func::FuncOp originalKernel, OpBuilder &builder,
                                            const std::vector<LeafBranchBandPlan> &leafBranchPlans,
                                            const BandApplyData &bandData, bool useRuntimeTileCounts) {
  const auto &bandsToUse = bandData.bandsToUse;
  const auto &allTileSizeValues = bandData.allTileSizeValues;
  const auto &allBandTileSizesInt = bandData.allBandTileSizesInt;
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

    LogicalResult status =
      plan.hasLeafBranching
        ? applySingleLeafBranchBandDecoupled(
            {originalKernel, builder, bandIdx, plan, band, tileSizeValues, tileSizesInt, useRuntimeTileCounts},
            nextNodeId)
        : applySingleLinearBandDecoupled(
            {originalKernel, builder, bandIdx, band, tileSizeValues, tileSizesInt, useRuntimeTileCounts}, nextNodeId);
    if (failed(status)) {
      return failure();
    }

    clearTemporaryLoopIdentificationAttrs(originalKernel, /* clearPointLoopAttr= */ false);
  }

  return success();
}

static void runApplyPostProcessing(func::FuncOp originalKernel, OpBuilder &builder) {
  // Step 5: Post-processing - mark attributes.
  clearTemporaryLoopIdentificationAttrs(originalKernel, /* clearPointLoopAttr= */ false);
  inlineDeleteMarkedLoops(originalKernel, builder);
  sinkMultiVecPointLoops(originalKernel);
  markInnermostLoopsWithVectorAttr(originalKernel, builder);
  inlineParallelNonVectorPointLoops(originalKernel, builder);
  clearNotInnerDimensionBroadcastLoopAttr(originalKernel);
  clearTemporaryLoopIdentificationAttrs(originalKernel);
  setBlockDimAttribute(originalKernel, builder);
  clearValuelessBroadcastLoopAttr(originalKernel);
}

LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape) {
  return applyTilingFromTilingFunc(originalKernel, builder, isStaticShape, nullptr);
}

LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape,
                                        const TilingMetadata *metadata) {
  auto loc = originalKernel.getLoc();

  std::vector<LeafBranchBandPlan> leafBranchPlans;
  bool shouldReturnEarly = false;
  if (failed(
        prepareLeafBranchPlansForApply(originalKernel, builder, leafBranchPlans, isStaticShape, shouldReturnEarly))) {
    return failure();
  }
  if (shouldReturnEarly) {
    return success();
  }

  std::vector<SmallVector<mlir::scf::ForOp, kSmallVectorSizeSix>> bandsToUse;
  collectRepresentativeBands(leafBranchPlans, bandsToUse);

  if (!leafBranchPlans.empty()) {
    preprocessLoopAttrsForTileCalculation(originalKernel, leafBranchPlans.front());
  }
  [[maybe_unused]] auto clearNotInnerBroadcastGuard =
    llvm::make_scope_exit([&originalKernel] { clearNotInnerDimensionBroadcastLoopAttr(originalKernel); });
  (void)clearNotInnerBroadcastGuard;

  // Setup builder.
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &originalKernel.getBody().front();
  builder.setInsertionPointToStart(body);

  std::vector<SmallVector<unsigned, kSmallVectorSizeSix>> allBandTileSizesInt;
  std::vector<SmallVector<int, kSmallVectorSizeSix>> allBandConstraintMaxs;
  std::vector<SmallVector<Value, kSmallVectorSizeSix>> allTileSizeValues;
  std::map<int64_t, Value> constantCache;
  if (failed(prepareTileMetadataForApply(
        {originalKernel, isStaticShape, BandTilingOutput{bandsToUse, allBandTileSizesInt, allBandConstraintMaxs},
         metadata},
        BuildContext{loc, builder, constantCache}, allTileSizeValues))) {
    return failure();
  }

  if (failed(applyAllBandsWithPlans(originalKernel, builder, leafBranchPlans,
                                    BandApplyData{bandsToUse, allTileSizeValues, allBandTileSizesInt},
                                    !isStaticShape))) {
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
    if ((band[i] == nullptr) || (band[i + 1] == nullptr)) {
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
  if (tiledLoops.size() < tileSizesNum + forNum) {
    return mlir::failure();
  }

  // innermost point loop: last point (idx = tileSizesNum + forNum - 1)
  mlir::scf::ForOp innermostPoint = tiledLoops[tileSizesNum + forNum - 1];
  if (!innermostPoint) {
    return mlir::failure();
  }

  // 1) collect the list of ops to clone
  llvm::SmallVector<mlir::Operation *, kSmallVectorSizeThirtyTwo> computeOps;
  collectChainComputeOps(band, computeOps);
  if (computeOps.empty()) {
    return mlir::success();
  }

  // 2) establish mapping: original band each layer iv -> new point each layer iv
  //    (plus region iter args / results for loops with iter_args).
  initIVMapping(band, tiledLoops, tileSizesNum, mapping);

  // 3) set insertion point.
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
    if (!loop) {
      continue;
    }

    // Map IV
    Value origIV = loop.getInductionVar();
    if (tiledLoop) {
      Value tiledIV = tiledLoop.getInductionVar();
      if (!mapping.contains(origIV)) {
        mapping.map(origIV, tiledIV);
      }
    } else {
      continue;
    }

    // Map region iter args (for loops with iter_args)
    if (tiledLoop) {
      for (auto [origArg, tiledArg] : llvm::zip(loop.getRegionIterArgs(), tiledLoop.getRegionIterArgs())) {
        if (!mapping.contains(origArg)) {
          mapping.map(origArg, tiledArg);
        }
      }
    }

    // Map loop results to corresponding tiled loop results.
    unsigned resultIdx = tileSizesNum + i;
    if (tiledLoop && resultIdx < tiledLoops.size()) {
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
  if (body == nullptr) {
    return;
  }

  auto *terminator = body->getTerminator();
  if (terminator == nullptr) {
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
      return {};
    }
    childLoop = childFor;
  }
  return childLoop;
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

static mlir::scf::ForOp findNextContiguousPointLoop(mlir::scf::ForOp pointLoop,
                                                    llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop,
                                                    mlir::scf::ForOp &nextPointParent) {
  for (mlir::scf::ForOp loop = getUniqueChildLoop(pointLoop); loop; loop = getUniqueChildLoop(loop)) {
    if (isPointLoop(loop)) {
      return loop;
    }
    if (loop->hasAttr(kInnerLoopAttr)) {
      return {};
    }
    nextPointParent = loop;
  }
  return {};
}

// Walk down `pointLoop`'s unique-child chain to locate the nearest matching
// inner point loop. If their bodies allow it (no iter_args / no side-effecting
// non-load ops in between), hoist the independent ops up past `pointLoop` and
// re-nest so `pointLoop` and the next point loop sit back-to-back. Returns
// `true` when one sink step actually happened, `false` otherwise.
static bool sinkPointLoopOnce(mlir::scf::ForOp pointLoop, llvm::StringRef chainAttr,
                              llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop) {
  mlir::scf::ForOp nextPointParent;
  mlir::scf::ForOp nextPointLoop = findNextContiguousPointLoop(pointLoop, isPointLoop, nextPointParent);
  if (!nextPointLoop || !nextPointParent) {
    return false;
  }

  for (mlir::scf::ForOp loop : {pointLoop, nextPointParent, nextPointLoop}) {
    if (loop.getNumResults() != 0 || !loop.getRegionIterArgs().empty()) {
      return false;
    }
  }

  SmallVector<Operation *, kSmallVectorSizeEight> hoistOps;
  llvm::SmallDenseSet<Value, kSmallVectorSizeEight> dependentValues;
  dependentValues.insert(pointLoop.getInductionVar());
  for (Operation &op : pointLoop.getBody()->without_terminator()) {
    bool dependsOnPointLoop =
      llvm::any_of(op.getOperands(), [&dependentValues](Value operand) { return dependentValues.contains(operand); });
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

// Sink only contiguous multi-vec point loops. A non-marked point loop is a
// semantic gap and must keep the original nesting order.
static void sinkPointLoopsByPredicate(func::FuncOp funcOp, llvm::StringRef chainAttr,
                                      llvm::function_ref<bool(mlir::scf::ForOp)> isPointLoop) {
  SmallVector<mlir::scf::ForOp, kSmallVectorSizeEight> pointLoops;
  funcOp.walk([&isPointLoop, &pointLoops](mlir::scf::ForOp loop) {
    if (isPointLoop(loop)) {
      pointLoops.push_back(loop);
    }
  });

  for (mlir::scf::ForOp pointLoop : pointLoops) {
    while (isPointLoop(pointLoop) && sinkPointLoopOnce(pointLoop, chainAttr, isPointLoop)) {
    }
  }
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

    if ((origBody == nullptr) || !targetLoop) {
      continue;
    }

    auto yieldOp = dyn_cast<mlir::scf::YieldOp>(origBody->getTerminator());
    if (!yieldOp) {
      continue;
    }

    SmallVector<Value> newOperands;
    newOperands.reserve(yieldOp.getNumOperands());
    std::transform(
      yieldOp.getOperands().begin(), yieldOp.getOperands().end(), std::back_inserter(newOperands),
      [&mapping](Value operand) { return (operand && mapping.contains(operand)) ? mapping.lookup(operand) : operand; });

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
                                                llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet) {
  Block *sourceBlock = userAnchorLoop->getBlock();
  SmallVector<Value, kSmallVectorSizeEight> worklist(rootLoop.getResults().begin(), rootLoop.getResults().end());

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

static void collectOpsInOrderAfter(Operation *startOp,
                                   const llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet,
                                   SmallVectorImpl<Operation *> &opsInOrder) {
  for (Operation *op = startOp->getNextNode(); op != nullptr; op = op->getNextNode()) {
    if (opSet.count(op) != 0u) {
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

static LogicalResult ensureNoExternalUsers(const llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet) {
  for (Operation *op : opSet) {
    for (Value res : op->getResults()) {
      for (Operation *user : res.getUsers()) {
        if (opSet.count(user) == 0u) {
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
static LogicalResult collectOperandClosure(Operation *outerLoopOp,
                                           llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &opSet,
                                           llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &recreateOps,
                                           const llvm::DenseSet<Value> &ignoreValues) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<Operation *, kSmallVectorSizeSixteen> currentOps(opSet.begin(), opSet.end());
    for (Operation *op : currentOps) {
      for (Value operand : op->getOperands()) {
        if (ignoreValues.contains(operand)) {
          continue;
        }
        Operation *defOp = operand.getDefiningOp();
        if (defOp == nullptr) {
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
                                         const llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> &recreateOps,
                                         mlir::OpBuilder &builder) {
  if (movedOps.empty() || recreateOps.empty()) {
    return;
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(movedOps.front());

  DenseMap<Value, Value> constMap;
  auto getRecreated = [&constMap, &builder](Value v) -> Value {
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
      if ((defOp == nullptr) || (recreateOps.count(defOp) == 0u)) {
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

  llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> opSet;
  if (failed(collectReduceResultUserOps(rootLoop, userAnchorLoop, opSet))) {
    return failure();
  }

  if (opSet.empty()) {
    return success();
  }

  llvm::SmallSet<Operation *, kSmallVectorSizeSixteen> recreateOps;
  llvm::DenseSet<Value> ignoreValues;
  ignoreValues.insert(rootResults.begin(), rootResults.end());
  if (failed(collectOperandClosure(userAnchorLoop.getOperation(), opSet, recreateOps, ignoreValues))) {
    return failure();
  }

  SmallVector<Operation *, kSmallVectorSizeSixteen> opsInOrder;
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
    if (terminator != nullptr) {
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
    llvm::SmallVector<mlir::Operation *, kSmallVectorSizeSixteen> pre;
    llvm::SmallVector<mlir::Operation *, kSmallVectorSizeSixteen> post;
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
      cloneOpsToPointLoop(parentTile0, pre, /* insertAtStart= */ true, builder, mapping);
    }

    // post: put into the end of the body (before the yield)
    if (!post.empty()) {
      cloneOpsToPointLoop(parentTile0, post, /* insertAtStart= */ false, builder, mapping);
    }
  }

  return mlir::success();
}

}  // namespace autotiling
}  // namespace mlir
