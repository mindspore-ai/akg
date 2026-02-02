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
#include <map>
#include <memory>
#include <unordered_set>

#include "akg/Analysis/BufferAnalysis.h"
#include "akg/Analysis/AutoTiling.h"
#include "akg/Analysis/TilingSolver.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
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
#include "mlir/Transforms/RegionUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

using hacc::HACCFuncType;
using hacc::HACCFuncTypeAttr;
using hacc::KernelArgType;
using hacc::KernelArgTypeAttr;
using llvm::DenseMap;
using llvm::SmallVector;
using llvm::SmallVectorImpl;
using mlir::autotiling::buildModelGraph;
using mlir::autotiling::buildNpuModelGraph;
using mlir::autotiling::getHeuristicTilingSolver;
using mlir::autotiling::getTileSizeWithSolver;
using mlir::autotiling::parseIr;
using mlir::autotiling::TilingSolverPtr;
using mlir::autotiling::TilingTaskDesc;

namespace mlir {
namespace autotiling {

// Main tiling functions
static LogicalResult createTilingFuncDefault(func::FuncOp originalKernel, OpBuilder &builder, func::FuncOp &tilingFunc,
                                             bool isStaticShape);
static LogicalResult calculateAllBandTileSizes(func::FuncOp funcOp, bool useAutoTiling,
                                               std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands,
                                               std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                               std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                               size_t &levelToTile);
static LogicalResult applyTilingToBand(ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                       ArrayRef<unsigned> tileSizesInt, OpBuilder &builder,
                                       std::map<int64_t, Value> &constantCache, func::FuncOp funcOp);
static LogicalResult collectBands(func::FuncOp funcOp, std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands);

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

// Loop tiling core helpers
static Value calculateOffsetForPointLoop(mlir::Location loc, ArrayRef<std::pair<Value, Value>> levelInfo,
                                         Value tilesize, OpBuilder &builder);
static LoopBounds createFirstLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value tilesize,
                                                 unsigned tilesizeInt, OpBuilder &builder,
                                                 std::map<int64_t, Value> &constantCache);
static LoopBounds createMiddleLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value curTilesize,
                                                  mlir::scf::ForOp prevLoop, mlir::scf::ForOp upperLoop,
                                                  OpBuilder &builder, std::map<int64_t, Value> &constantCache);
static LoopBounds createPointLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value lastTilesize,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop,
                                        OpBuilder &builder, std::map<int64_t, Value> &constantCache);
static mlir::scf::ForOp replaceLoopWithNewBounds(mlir::scf::ForOp oldLoop, const LoopBounds &bounds, mlir::Location loc,
                                                 OpBuilder &builder);
static void insertArgInnerCalculationInPointLoop(mlir::scf::ForOp newLoop, mlir::scf::ForOp origLoop,
                                                 ArrayRef<std::pair<Value, Value>> levelInfo, Value lastTilesize,
                                                 mlir::Location loc, int lastTile, ArrayRef<Value> tileSizeValues);
static void processSingleTileLoop(int i, int j, int bandSize, int tileNum, MutableArrayRef<mlir::scf::ForOp> newLoops,
                                  ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                  ArrayRef<unsigned> tileSizesInt,
                                  SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                  Value kernelsizeConstant, mlir::Location loc,
                                  std::map<int64_t, Value> &constantCache);
static void constructTiledIndexStatic(MutableArrayRef<mlir::scf::ForOp> newLoops, ArrayRef<mlir::scf::ForOp> band,
                                      ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                      OpBuilder &builder, std::map<int64_t, Value> &constantCache,
                                      mlir::IRMapping &mapping);

// Dynamic axis mapping and tile size helpers
static std::vector<DynamicAxisMapping> buildDynamicAxisMappingForBand(ArrayRef<mlir::scf::ForOp> band,
                                                                      ArrayRef<unsigned> bandTileSizes,
                                                                      func::FuncOp originalKernel);
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                 Value &result);
static LogicalResult prepareTileSizesForStaticShape(func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                    std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                    std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                                    std::vector<SmallVector<unsigned, 6>> &allBandTileSizesOut);
static LogicalResult prepareTileSizesFromMemref(func::FuncOp originalKernel,
                                                ArrayRef<SmallVector<mlir::scf::ForOp, 6>> bands, mlir::Location loc,
                                                OpBuilder &builder,
                                                std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                                std::vector<SmallVector<unsigned, 6>> &allBandTileSizesInt);

// Tiling function creation helpers
static void buildTilingFunctionSignature(FunctionType origTy, MLIRContext *ctx, OpBuilder &builder,
                                         SmallVector<Type> &argTypes, SmallVector<Type> &resTypes);
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
static bool isInnermostScfLoop(mlir::scf::ForOp forOp);
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder, int64_t vectorSize);
static void setBlockDimAttribute(func::FuncOp funcOp, OpBuilder &builder);
static void markOutermostLoopsForParallelMapping(func::FuncOp funcOp, OpBuilder &builder);

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
static std::optional<int64_t> getConstantStep(mlir::scf::ForOp forOp);
static void moveLoopBody(mlir::scf::ForOp src, mlir::scf::ForOp dest);
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
static unsigned getMiddleLevelLoopIndex(unsigned tileSizesNum, unsigned bandSize, unsigned dim);
static void initIVMapping(llvm::ArrayRef<mlir::scf::ForOp> band, llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                          unsigned tileSizesNum, mlir::IRMapping &mapping);
static void updatePointLoopYieldsFromOriginalLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                   llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                   unsigned tileSizesNum, mlir::IRMapping &mapping,
                                                   mlir::OpBuilder &builder);
static void forwardIterArgsThroughWrapperLoops(llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                               mlir::OpBuilder &builder);
static LogicalResult collectReduceResultUserOps(mlir::scf::ForOp rootLoop, mlir::scf::ForOp outerLoop,
                                                llvm::SmallSet<Operation *, 16> &opSet);
static void collectOpsInOrderAfter(Operation *startOp, const llvm::SmallSet<Operation *, 16> &opSet,
                                   SmallVectorImpl<Operation *> &opsInOrder);
static void moveOpsAfterLoop(mlir::scf::ForOp destLoop, ArrayRef<Operation *> ops, OpBuilder &builder);
static void replaceUsesInOps(ValueRange oldResults, ValueRange newResults, ArrayRef<Operation *> ops);
static LogicalResult sinkReduceLoopResultsToMiddleLevel(mlir::scf::ForOp rootLoop,
                                                        llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                        unsigned tileSizesNum, unsigned bandSize,
                                                        mlir::OpBuilder &builder);
static void cloneOpsToPointLoop(mlir::scf::ForOp pointLoop, llvm::ArrayRef<mlir::Operation *> ops, bool insertAtStart,
                                mlir::OpBuilder &builder, mlir::IRMapping &mapping);
static mlir::LogicalResult cloneNonPerfectChainIntoPointLoops(llvm::ArrayRef<mlir::scf::ForOp> band,
                                                              llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                              unsigned tileSizesNum, mlir::OpBuilder &builder,
                                                              mlir::IRMapping &mapping);
static mlir::Value remapOrSelf(mlir::Value v, mlir::IRMapping &mapping);

// Helper function to move loop body from src to dest
[[maybe_unused]] static void moveLoopBody(mlir::scf::ForOp src, mlir::scf::ForOp dest) {
  auto &destOps = dest.getBody()->getOperations();
  auto &srcOps = src.getBody()->getOperations();

  // Find the insertion point: before the terminator in dest body
  mlir::Block::iterator insertLoc;
  auto *terminator = dest.getBody()->getTerminator();
  if (terminator) {
    insertLoc = terminator->getIterator();
  } else {
    insertLoc = dest.getBody()->end();
  }

  // Move all operations except the terminator (scf.yield)
  if (srcOps.size() > 1) {
    destOps.splice(insertLoc, srcOps, srcOps.begin(), std::prev(srcOps.end()));
  }
}

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
  if (auto blockArg = upperBound.dyn_cast<BlockArgument>()) {
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
      if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
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

// Helper function to get constant step value from scf.for
static std::optional<int64_t> getConstantStep(mlir::scf::ForOp forOp) { return getConstantIndexValue(forOp.getStep()); }

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

// Helper function to recursively collect all nested scf.for loops within a loop's body
static void collectNestedLoopsInBody(mlir::scf::ForOp forOp, SmallVector<mlir::scf::ForOp, 6> &band) {
  Block *body = forOp.getBody();
  if (!body) {
    return;
  }

  // Walk through all operations in the body to find nested scf.for loops
  for (Operation &op : body->without_terminator()) {
    if (auto nestedFor = dyn_cast<mlir::scf::ForOp>(&op)) {
      band.push_back(nestedFor);
      // Recursively collect loops nested inside this loop
      collectNestedLoopsInBody(nestedFor, band);
    }
  }
}

// Helper function to collect bands from a block
// Each outermost scf.for loop starts a new band, containing itself and all nested scf.for loops
static void collectBandsFromBlock(Block *block, std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands) {
  for (Operation &op : *block) {
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(&op)) {
      // Check if this is an outermost loop (parent is not scf.for)
      // Traverse parent chain until we find a scf.for or func::FuncOp
      bool isOutermost = true;
      for (Operation *parent = forOp->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
        if (isa<mlir::scf::ForOp>(parent)) {
          isOutermost = false;
          break;
        }
        if (isa<mlir::func::FuncOp>(parent)) {
          break;  // Reached function boundary
        }
      }

      // Only collect bands starting from outermost loops
      if (isOutermost) {
        SmallVector<mlir::scf::ForOp, 6> band;
        band.push_back(forOp);
        // Recursively collect all nested loops in this loop's body
        collectNestedLoopsInBody(forOp, band);
        if (!band.empty()) {
          bands.push_back(band);
        }
      }
    }
  }
}

// Calculate all band tile sizes using auto-tiling
static LogicalResult calculateAllBandTileSizes(func::FuncOp funcOp, bool useAutoTiling,
                                               std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands,
                                               std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
                                               std::vector<SmallVector<int, 6>> &allBandConstraintMaxs,
                                               size_t &levelToTile) {
  bands.clear();
  allBandTileSizes.clear();
  allBandConstraintMaxs.clear();

  // Collect bands (including non-perfectly nested ones) using the same logic as collectBands
  Block *body = &funcOp.getBody().front();
  collectBandsFromBlock(body, bands);

  // If no bands found, return success (no tiling needed)
  if (bands.empty()) {
    return success();
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
  auto initGraph = parseIr(funcOp, bands);
  initGraph->setHardware(target);
  initGraph->setFeature(feature);
  initGraph->setArch(arch);
  initGraph->setIsDynamicShape(isDynamicShape);
  initGraph->setTilingMode("auto");

  auto modelGraph = buildModelGraph(initGraph);
  auto solver = getHeuristicTilingSolver(modelGraph);
  levelToTile = modelGraph->levelToTile;

  // Calculate tile sizes for each band (with constraint max for dynamic shapes)
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    SmallVector<mlir::scf::ForOp, 6> curBand = bands[bandIdx];
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

// Helper function to extract induction variables from a band
[[maybe_unused]] static void extractForInductionVarsStatic(ArrayRef<mlir::scf::ForOp> band,
                                                           SmallVectorImpl<mlir::Value> *origLoopIVs) {
  origLoopIVs->clear();
  origLoopIVs->reserve(band.size());
  std::transform(band.begin(), band.end(), std::back_inserter(*origLoopIVs),
                 [](mlir::scf::ForOp forOp) { return forOp.getInductionVar(); });
}

// Helper to check if a value is defined inside the band
[[maybe_unused]] static bool isDefinedInBand(Value v, mlir::scf::ForOp outermostLoop) {
  if (!v) return false;
  Operation *defOp = v.getDefiningOp();
  if (!defOp) return false;
  return outermostLoop->isAncestor(defOp);
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

// Helper: Calculate offset for point loop (offset = sum(ivs) * tilesize)
[[maybe_unused]] static Value calculateOffsetForPointLoop(mlir::Location loc,
                                                          ArrayRef<std::pair<Value, Value>> levelInfo, Value tilesize,
                                                          OpBuilder &builder) {
  SmallVector<Value> ivs;
  for (const auto &[iv, _] : levelInfo) {
    ivs.push_back(iv);
  }

  Value sumIvs;
  if (ivs.empty()) {
    sumIvs = builder.create<arith::ConstantIndexOp>(loc, 0);
  } else if (ivs.size() == 1) {
    sumIvs = ivs[0];
  } else {
    auto sumExpr = builder.getAffineDimExpr(0);
    for (size_t k = 1; k < ivs.size(); ++k) {
      sumExpr = sumExpr + builder.getAffineDimExpr(k);
    }
    auto sumMap = AffineMap::get(ivs.size(), 0, sumExpr, builder.getContext());
    sumIvs = builder.create<mlir::affine::AffineApplyOp>(loc, sumMap, ivs);
  }

  auto mulMap =
    AffineMap::get(1, 1, builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(0), builder.getContext());
  return builder.create<mlir::affine::AffineApplyOp>(loc, mulMap, ValueRange{sumIvs, tilesize});
}

// Helper: Create bounds for first level tile loop
static LoopBounds createFirstLevelTileLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value tilesize,
                                                 unsigned tilesizeInt, OpBuilder &builder,
                                                 std::map<int64_t, Value> &constantCache) {
  LoopBounds bounds;

  // Special handling for reduction loops: skip tiling, only execute once
  if (origLoop->hasAttr(kReductionLoopAttr)) {
    bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    bounds.ub = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    // For reduce loops with iter_args, drop inits to create a no-result loop.
    bounds.inits = (origLoop.getNumResults() == 0) ? origLoop.getInitArgs() : ValueRange{};
    return bounds;
  }

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
                                                  mlir::scf::ForOp prevLoop, mlir::scf::ForOp upperLoop,
                                                  OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  LoopBounds bounds;

  // Special handling for reduction loops: skip tiling
  // Set bounds to execute the full range: lb=0, ub=origUb, step=1
  if (origLoop->hasAttr(kReductionLoopAttr)) {
    bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    // Recreate constants at current insertion point to avoid dominance issues
    // when origUb is defined inside the original band.
    bounds.ub = recreateConstantOrSelf(origLoop.getUpperBound(), builder);
    // step = 1 for reduce loops (iterate through all elements)
    bounds.step = recreateConstantOrSelf(prevLoop.getStep(), builder);
    // Use original init args for reduce loops with iter_args (first level has no iter_args).
    if (origLoop.getNumResults() == 0) {
      bounds.inits = prevLoop.getRegionIterArgs();
    } else {
      bounds.inits = origLoop.getInitArgs();
    }
    return bounds;
  }

  Value origUb = origLoop.getUpperBound();

  // lb = 0
  bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);

  // ub = ceildiv(origUb, curTilesize)
  auto origUbConst = getConstantIndexValue(origUb);
  auto curConst = getConstantIndexValue(curTilesize);

  if (origUbConst && curConst && curConst.value() != 0) {
    // Both are compile-time constants: compute ceildiv statically
    int64_t ubVal = (origUbConst.value() + curConst.value() - 1) / curConst.value();
    bounds.ub = getOrCreateConstantStatic(loc, ubVal, builder, constantCache);
  } else {
    // curTilesize is dynamic: compute ceildiv dynamically
    auto ceildivMap =
      AffineMap::get(1, 1, builder.getAffineDimExpr(0).ceilDiv(builder.getAffineSymbolExpr(0)), builder.getContext());
    bounds.ub = builder.create<mlir::affine::AffineApplyOp>(
      loc, ceildivMap, ValueRange{recreateConstantOrSelf(origUb, builder), curTilesize});
  }

  // step = upper ub (kernel number)
  if (upperLoop) {
    bounds.step = upperLoop.getUpperBound();
  } else {
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
  }

  bounds.inits = prevLoop.getResults();
  return bounds;
}

// Helper: Create bounds for point loop
static LoopBounds createPointLoopBounds(mlir::Location loc, mlir::scf::ForOp origLoop, Value lastTilesize,
                                        ArrayRef<std::pair<Value, Value>> levelInfo, mlir::scf::ForOp prevLoop,
                                        OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  LoopBounds bounds;

  // Special handling for reduction loops: point loop only executes once
  if (origLoop->hasAttr(kReductionLoopAttr)) {
    bounds.lb = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    bounds.ub = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    bounds.step = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    // Use region iter args from outer loop for correct SSA flow
    bounds.inits = prevLoop.getRegionIterArgs();
    return bounds;
  }

  // Calculate sum of upper layer IVs
  SmallVector<Value> ivs;
  for (const auto &[iv, _] : levelInfo) {
    ivs.push_back(iv);
  }

  Value sumIvs;
  if (ivs.empty()) {
    sumIvs = builder.create<arith::ConstantIndexOp>(loc, 0);
  } else if (ivs.size() == 1) {
    sumIvs = ivs[0];
  } else {
    auto sumExpr = builder.getAffineDimExpr(0);
    for (size_t k = 1; k < ivs.size(); ++k) {
      sumExpr = sumExpr + builder.getAffineDimExpr(k);
    }
    auto sumMap = AffineMap::get(ivs.size(), 0, sumExpr, builder.getContext());
    sumIvs = builder.create<mlir::affine::AffineApplyOp>(loc, sumMap, ivs);
  }

  // lb = sum(ivs) * lastTilesize
  auto mulMap =
    AffineMap::get(1, 1, builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(0), builder.getContext());
  bounds.lb = builder.create<mlir::affine::AffineApplyOp>(loc, mulMap, ValueRange{sumIvs, lastTilesize});

  // ub = min((sum(ivs) + 1) * lastTilesize, origUb)
  Value sumIvsPlus1 = builder.create<arith::AddIOp>(loc, sumIvs, builder.create<arith::ConstantIndexOp>(loc, 1));
  Value ubCandidate = builder.create<mlir::affine::AffineApplyOp>(loc, mulMap, ValueRange{sumIvsPlus1, lastTilesize});

  Value origUb = origLoop.getUpperBound();
  Value clampedOrigUb = recreateConstantOrSelf(origUb, builder);

  // ub = min(ubCandidate, origUb)
  auto minMap = AffineMap::getMultiDimIdentityMap(2, builder.getContext());
  bounds.ub = builder.create<mlir::affine::AffineMinOp>(loc, minMap, ValueRange{ubCandidate, clampedOrigUb});

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

// Helper: Insert argInner calculation in point loop body
[[maybe_unused]] static void insertArgInnerCalculationInPointLoop(mlir::scf::ForOp newLoop, mlir::scf::ForOp origLoop,
                                                                  ArrayRef<std::pair<Value, Value>> levelInfo,
                                                                  Value lastTilesize, mlir::Location loc, int lastTile,
                                                                  ArrayRef<Value> tileSizeValues) {
  auto *newBody = newLoop.getBody();

  // Calculate offset
  SmallVector<Value> ivs;
  for (const auto &[iv, _] : levelInfo) {
    ivs.push_back(iv);
  }

  OpBuilder bodyBuilder(newLoop.getContext());
  bodyBuilder.setInsertionPointToStart(newBody);

  Value sumIvs;
  if (ivs.empty()) {
    sumIvs = bodyBuilder.create<arith::ConstantIndexOp>(loc, 0);
  } else if (ivs.size() == 1) {
    sumIvs = ivs[0];
  } else {
    auto sumExpr = bodyBuilder.getAffineDimExpr(0);
    for (size_t k = 1; k < ivs.size(); ++k) {
      sumExpr = sumExpr + bodyBuilder.getAffineDimExpr(k);
    }
    auto sumMap = AffineMap::get(ivs.size(), 0, sumExpr, bodyBuilder.getContext());
    sumIvs = bodyBuilder.create<mlir::affine::AffineApplyOp>(loc, sumMap, ivs);
  }

  auto mulMap = AffineMap::get(1, 1, bodyBuilder.getAffineDimExpr(0) * bodyBuilder.getAffineSymbolExpr(0),
                               bodyBuilder.getContext());
  Value offset = bodyBuilder.create<mlir::affine::AffineApplyOp>(loc, mulMap, ValueRange{sumIvs, lastTilesize});

  // argInner = point_iv + offset
  Value argInner = bodyBuilder.create<arith::AddIOp>(loc, newLoop.getInductionVar(), offset);

  // Replace origLoop IV with argInner
  origLoop.getInductionVar().replaceAllUsesWith(argInner);

  // Update iter args
  for (auto it : llvm::zip(origLoop.getRegionIterArgs(), newLoop.getRegionIterArgs())) {
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  }
}

// Helper: Process a single tile loop (extracted to reduce cyclomatic complexity)
static void processSingleTileLoop(int i, int j, int bandSize, int tileNum, MutableArrayRef<mlir::scf::ForOp> newLoops,
                                  ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                  ArrayRef<unsigned> tileSizesInt,
                                  SmallVectorImpl<SmallVector<std::pair<Value, Value>, 4>> &tileLevelInfo,
                                  Value kernelsizeConstant, mlir::Location loc,
                                  std::map<int64_t, Value> &constantCache) {
  int curTile = i * bandSize + j;
  int lastTile = curTile - bandSize;

  if (curTile >= static_cast<int>(newLoops.size())) {
    return;
  }
  mlir::scf::ForOp loop = newLoops[curTile];
  if (!loop) {
    return;
  }

  mlir::OpBuilder loopBuilder(loop);
  mlir::scf::ForOp origLoop = band[j];
  LoopBounds bounds;

  // Determine loop bounds based on level
  if (i == 0) {
    // First level tile loop
    unsigned tilesizeInt =
      (curTile < static_cast<int>(tileSizesInt.size())) ? tileSizesInt[curTile] : static_cast<unsigned>(-1);

    bounds =
      createFirstLevelTileLoopBounds(loc, origLoop, tileSizeValues[curTile], tilesizeInt, loopBuilder, constantCache);
  } else if (i == tileNum) {
    // Point loop
    if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) return;
    mlir::scf::ForOp prevLoop = newLoops[lastTile];

    bounds = createPointLoopBounds(loc, origLoop, tileSizeValues[lastTile], tileLevelInfo[j], prevLoop, loopBuilder,
                                   constantCache);
  } else {
    // Middle level tile loop
    if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) return;
    mlir::scf::ForOp prevLoop = newLoops[lastTile];

    int upperTile = (i - 1) * bandSize + j;
    mlir::scf::ForOp upperLoop =
      (upperTile >= 0 && upperTile < static_cast<int>(newLoops.size())) ? newLoops[upperTile] : mlir::scf::ForOp();

    bounds = createMiddleLevelTileLoopBounds(loc, origLoop, tileSizeValues[curTile], prevLoop, upperLoop, loopBuilder,
                                             constantCache);
  }

  // Replace old loop with new one
  mlir::scf::ForOp newLoop = replaceLoopWithNewBounds(loop, bounds, loc, loopBuilder);

  newLoops[curTile] = newLoop;

  // For point loop: preserve reduction attribute from original loop
  if (i == tileNum && origLoop->hasAttr(kReductionLoopAttr)) {
    newLoop->setAttr(kReductionLoopAttr, loopBuilder.getI64IntegerAttr(kVectorSize));
  }

  // Record tile level info for future offset calculations
  if (i < tileNum && curTile < static_cast<int>(tileSizeValues.size())) {
    Value tilesizeToRecord = tileSizeValues[curTile];

    // For dynamic first level: use kernelsize instead of ceildiv result
    if (i == 0 && !getConstantIndexValue(tilesizeToRecord)) {
      tilesizeToRecord = kernelsizeConstant;
    }

    tileLevelInfo[j].push_back({newLoop.getInductionVar(), tilesizeToRecord});
  }
}

// Helper function to construct tiled index (bounds and steps) using tileSizeValues from memref
static void constructTiledIndexStatic(MutableArrayRef<mlir::scf::ForOp> newLoops, ArrayRef<mlir::scf::ForOp> band,
                                      ArrayRef<Value> tileSizeValues, ArrayRef<unsigned> tileSizesInt,
                                      OpBuilder &builder, std::map<int64_t, Value> &constantCache,
                                      mlir::IRMapping &mapping) {
  int bandSize = static_cast<int>(band.size());
  if (bandSize == 0 || tileSizeValues.size() == 0) {
    return;
  }

  mlir::Location loc = band[0]->getLoc();
  int tileNum = static_cast<int>(tileSizeValues.size()) / bandSize;

  // Track tile level info for each dimension: {IV, tilesize}
  SmallVector<SmallVector<std::pair<Value, Value>, 4>, 4> tileLevelInfo(bandSize);

  // Pre-create KERNELSIZE constant for dynamic shape case
  OpBuilder::InsertionGuard guard(builder);
  if (!newLoops.empty() && newLoops[newLoops.size() - 1]) {
    builder.setInsertionPoint(newLoops[newLoops.size() - 1]);
  }
  Value kernelsizeConstant = builder.create<arith::ConstantIndexOp>(loc, kBlockDimSize);

  // Process each tile level and dimension
  for (int i = 0; i <= tileNum; ++i) {
    for (int j = 0; j < bandSize; ++j) {
      processSingleTileLoop(i, j, bandSize, tileNum, newLoops, band, tileSizeValues, tileSizesInt, tileLevelInfo,
                            kernelsizeConstant, loc, constantCache);
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

// Static helper function to create tail block for static bounds
static LogicalResult createTailBlockStaticImpl(mlir::scf::ForOp forOp, int64_t differenceUbAndLb, OpBuilder &builder,
                                               std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = forOp.getLoc();

  mlir::Value origLb = forOp.getLowerBound();
  mlir::Value origUb = forOp.getUpperBound();
  auto origStepOpt = getConstantStep(forOp);

  if (!origStepOpt.has_value()) {
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }

  int64_t origStep = origStepOpt.value();
  int64_t tailSize = differenceUbAndLb % origStep;

  if (tailSize == 0) {
    if (failed(createTailBlockForBodyStatic(forOp, builder, constantCache))) {
      return failure();
    }
    return success();
  }

  builder.setInsertionPoint(forOp);
  mlir::Value tailSizeVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
  mlir::Value newUb = builder.create<mlir::arith::SubIOp>(loc, origUb, tailSizeVal);

  int64_t newDifferenceUbAndLb = differenceUbAndLb - tailSize;

  mlir::scf::ForOp currentLoop = forOp;
  if (differenceUbAndLb < origStep && newDifferenceUbAndLb) {
    builder.setInsertionPoint(forOp);
    auto newLoop = builder.create<mlir::scf::ForOp>(loc, origLb, newUb, forOp.getStep(), forOp.getInitArgs());

    newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(), forOp.getBody()->getOperations(),
                                              forOp.getBody()->begin(), std::prev(forOp.getBody()->end()));

    forOp.replaceAllUsesWith(newLoop.getResults());
    forOp.erase();
    currentLoop = newLoop;

    mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
    builder.setInsertionPointAfter(currentLoop);
    auto tailLoop = builder.create<mlir::scf::ForOp>(loc, newUb, origUb, tailStepVal, currentLoop.getInitArgs());

    mlir::IRMapping mapping;
    mapping.map(currentLoop.getInductionVar(), tailLoop.getInductionVar());
    builder.setInsertionPointToStart(tailLoop.getBody());
    for (auto &op : currentLoop.getBody()->without_terminator()) {
      builder.clone(op, mapping);
    }

    if (failed(createTailBlockForBodyStatic(tailLoop, builder, constantCache))) {
      return failure();
    }
    return success();
  } else if (differenceUbAndLb >= origStep) {
    builder.setInsertionPoint(forOp);
    auto newLoop = builder.create<mlir::scf::ForOp>(loc, origLb, newUb, forOp.getStep(), forOp.getInitArgs());

    newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(), forOp.getBody()->getOperations(),
                                              forOp.getBody()->begin(), std::prev(forOp.getBody()->end()));

    forOp.replaceAllUsesWith(newLoop.getResults());
    forOp.erase();
    currentLoop = newLoop;
  }

  if (failed(createTailBlockForBodyStatic(currentLoop, builder, constantCache))) {
    return failure();
  }

  bool isEqualToBlock = (newDifferenceUbAndLb == 0) || (newDifferenceUbAndLb == origStep && tailSize == origStep);
  if (differenceUbAndLb < origStep || !newDifferenceUbAndLb || isEqualToBlock) {
    return success();
  }

  builder.setInsertionPointAfter(currentLoop);
  mlir::Value tailStepVal = getOrCreateConstantStatic(loc, tailSize, builder, constantCache);
  auto tailLoop = builder.create<mlir::scf::ForOp>(loc, newUb, origUb, tailStepVal, currentLoop.getInitArgs());

  mlir::IRMapping mapping;
  mapping.map(currentLoop.getInductionVar(), tailLoop.getInductionVar());
  builder.setInsertionPointToStart(tailLoop.getBody());
  for (auto &op : currentLoop.getBody()->without_terminator()) {
    builder.clone(op, mapping);
  }

  if (failed(createTailBlockForBodyStatic(tailLoop, builder, constantCache))) {
    return failure();
  }
  return success();
}

// Static helper function to create tail block for dynamic bounds
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value dynamicBound, OpBuilder &builder,
                                                std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = forOp.getLoc();

  mlir::Value origLb = forOp.getLowerBound();
  mlir::Value origUb = forOp.getUpperBound();
  auto origStepOpt = getConstantStep(forOp);

  if (!origStepOpt.has_value()) {
    builder.setInsertionPoint(forOp);
    return createTailBlockForBodyStatic(forOp, builder, constantCache);
  }

  int64_t origStep = origStepOpt.value();

  if (origStep == 1) {
    if (failed(createTailBlockForBodyStatic(forOp, builder, constantCache))) {
      return failure();
    }
    return success();
  }

  builder.setInsertionPoint(forOp);
  mlir::Value recalculatedDynamicBound = builder.create<mlir::arith::SubIOp>(loc, origUb, origLb);

  mlir::Value stepVal = getOrCreateConstantStatic(loc, origStep, builder, constantCache);
  mlir::Value remainder = builder.create<mlir::arith::RemSIOp>(loc, recalculatedDynamicBound, stepVal);

  mlir::Value newUb = builder.create<mlir::arith::SubIOp>(loc, origUb, remainder);

  auto newLoop = builder.create<mlir::scf::ForOp>(loc, origLb, newUb, forOp.getStep(), forOp.getInitArgs());

  newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(), forOp.getBody()->getOperations(),
                                            forOp.getBody()->begin(), std::prev(forOp.getBody()->end()));

  forOp.replaceAllUsesWith(newLoop.getResults());
  forOp.erase();
  mlir::scf::ForOp currentLoop = newLoop;

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
  auto tailLoop = builder.create<mlir::scf::ForOp>(loc, newUb, origUb, tailStepVal, currentLoop.getInitArgs());

  mlir::IRMapping mapping;
  mapping.map(currentLoop.getInductionVar(), tailLoop.getInductionVar());
  builder.setInsertionPointToStart(tailLoop.getBody());
  for (auto &op : currentLoop.getBody()->without_terminator()) {
    builder.clone(op, mapping);
  }

  if (failed(createTailBlockForBodyStatic(tailLoop, builder, constantCache))) {
    return failure();
  }
  return success();
}

// Helper function to apply tiling to a single band using tileSizeValues from memref
static LogicalResult applyTilingToBand(ArrayRef<mlir::scf::ForOp> band, ArrayRef<Value> tileSizeValues,
                                       ArrayRef<unsigned> tileSizesInt, OpBuilder &builder,
                                       std::map<int64_t, Value> &constantCache, func::FuncOp funcOp) {
  if (band.empty() || tileSizeValues.empty()) {
    return success();
  }

  if (failed(verifyBandIsNestedChain(band))) {
    return failure();
  }

  unsigned forNum = band.size();
  unsigned tileSizesNum = tileSizeValues.size();
  if (tileSizesNum == 0 || forNum == 0) {
    return success();
  }

  if (tileSizesNum >= forNum && tileSizesNum % forNum == 0) {
    mlir::scf::ForOp rootScfForOp = band[0];
    unsigned width = tileSizesNum + forNum;
    SmallVector<mlir::scf::ForOp, 6> tiledLoops(width);

    constructTiledLoopStatic(rootScfForOp, width, tiledLoops, builder, constantCache);

    // CRITICAL: Replace all dummy loops first, BEFORE cloning any operations!
    // This ensures that when we clone ops and establish IV mapping,
    // we map to the FINAL loop IVs, not the dummy ones.
    mlir::IRMapping dummyMapping;  // Unused, but required by constructTiledIndexStatic
    constructTiledIndexStatic(tiledLoops, band, tileSizeValues, tileSizesInt, builder, constantCache, dummyMapping);

    // NOW clone operations with correct IV mapping to final loops
    mlir::IRMapping mapping;
    if (failed(cloneNonPerfectChainIntoPointLoops(band, tiledLoops, tileSizesNum, builder, mapping))) {
      return failure();
    }

    if (failed(cloneComputeIntoInnermostPointLoop(band, tiledLoops, tileSizesNum, rootScfForOp, builder, mapping))) {
      return failure();
    }

    updatePointLoopYieldsFromOriginalLoops(band, tiledLoops, tileSizesNum, mapping, builder);
    forwardIterArgsThroughWrapperLoops(tiledLoops, builder);

    if (isReductionLoopWithIterArgs(rootScfForOp)) {
      if (failed(sinkReduceLoopResultsToMiddleLevel(rootScfForOp, tiledLoops, tileSizesNum, forNum, builder))) {
        return failure();
      }
    } else {
      // Replace original loop's results with outermost tiled loop's results before erasing.
      if (rootScfForOp.getNumResults() > 0 && tiledLoops[0]) {
        rootScfForOp.replaceAllUsesWith(tiledLoops[0].getResults());
      }
    }
    rootScfForOp.erase();
  } else {
    return failure();
  }

  return success();
}

// Helper function to collect bands from a function
static LogicalResult collectBands(func::FuncOp funcOp, std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands) {
  bands.clear();

  Block *body = &funcOp.getBody().front();
  collectBandsFromBlock(body, bands);

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

// Mark all innermost scf.for loops with vector attribute
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder, int64_t vectorSize) {
  funcOp->walk([&](mlir::scf::ForOp forOp) {
    if (isInnermostScfLoop(forOp)) {
      // set vector attribute to innermost loops
      if (!forOp->hasAttr(kReductionLoopAttr)) {
        forOp->setAttr(kVectorAttr, builder.getI64IntegerAttr(vectorSize));
      } else if (forOp->getParentOp() && isa<mlir::scf::ForOp>(forOp->getParentOp())) {
        // if reduction loop, remove reduction attribute and set vector attribute to parent loop
        if (isReductionLoopWithIterArgs(forOp)) {
          forOp->getParentOp()->setAttr(kReductionLoopAttr, builder.getI64IntegerAttr(vectorSize));
        } else {
          forOp->getParentOp()->setAttr(kVectorAttr, builder.getI64IntegerAttr(vectorSize));
        }
        forOp->removeAttr(kReductionLoopAttr);
        forOp->setAttr(kDeleteLoopAttr, builder.getUnitAttr());
      }
    }
  });
}

// Set hacc.block_dim attribute to funcOp based on outermost tiled loop's upper bound
static void setBlockDimAttribute(func::FuncOp funcOp, OpBuilder &builder) {
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

// Mark all outermost loops in each band with map_for_to_forall attribute
static void markOutermostLoopsForParallelMapping(func::FuncOp funcOp, OpBuilder &builder) {
  // Collect all outermost loops (those without parent scf.for)
  SmallVector<mlir::scf::ForOp, 8> outermostLoops;

  funcOp->walk([&](mlir::scf::ForOp forOp) {
    // Check if this loop is at the top level (no parent scf.for)
    // Traverse parent chain to find if there's a parent scf.for
    for (Operation *parent = forOp->getParentOp(); parent != nullptr; parent = parent->getParentOp()) {
      if (isa<func::FuncOp>(parent)) {
        break;  // Reached function boundary, this is outermost
      }
      if (isa<mlir::scf::ForOp>(parent)) {
        return;  // Has parent loop, not outermost
      }
    }
    // This is an outermost loop
    outermostLoops.push_back(forOp);
  });

  // Mark all outermost loops with map_for_to_forall attribute
  for (auto loop : outermostLoops) {
    loop->setAttr(kMapForToForallAttr, builder.getUnitAttr());
  }
}

// Inline reduce PointLoops that execute only once (lb=0, ub=1, step=1)
// These loops have kReductionLoopAttr and their IV is not actually used
static void inlineReducePointLoops(func::FuncOp funcOp, OpBuilder &builder) {
  // Collect all reduce point loops that need to be inlined
  SmallVector<mlir::scf::ForOp, 8> loopsToInline;

  funcOp->walk([&](mlir::scf::ForOp forOp) {
    // Check if this loop has delete attribute
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
    [[maybe_unused]] Block *parentBlock = loop->getBlock();

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
                                         SmallVector<Type> &argTypes, SmallVector<Type> &resTypes) {
  static constexpr int64_t kTilingStructMemrefSize = 64;

  argTypes.assign(origTy.getInputs().begin(), origTy.getInputs().end());

  auto i64Ty = builder.getI64Type();
  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
  auto memrefTy = MemRefType::get({kTilingStructMemrefSize}, i64Ty);

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

        Value step;
        if (constraintMax > 0) {
          // Use constraint upper bound: tile_size = min(constraintMax, dim)
          Value constraintMaxVal = b.create<arith::ConstantIndexOp>(loc, constraintMax);
          // min(constraintMax, dim)
          Value cmp = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, constraintMaxVal, dim);
          step = b.create<arith::SelectOp>(loc, cmp, constraintMaxVal, dim);
        } else {
          // Fallback to original behavior: ceildiv(dim, kBlockDimSize)
          Value c40 = b.create<arith::ConstantIndexOp>(loc, kBlockDimSize);
          step = b.create<arith::CeilDivSIOp>(loc, dim, c40);
        }

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
  auto *ctx = builder.getContext();
  auto loc = originalKernel.getLoc();
  auto origTy = originalKernel.getFunctionType();

  // Step 1: Calculate tile sizes (with constraint max for dynamic shapes)
  std::vector<SmallVector<mlir::scf::ForOp, 6>> bands;
  std::vector<SmallVector<unsigned, 6>> allBandTileSizes;
  std::vector<SmallVector<int, 6>> allBandConstraintMaxs;
  size_t levelToTile = 0;

  if (failed(calculateAllBandTileSizes(originalKernel, true, bands, allBandTileSizes, allBandConstraintMaxs,
                                       levelToTile))) {
    return failure();
  }

  // Step 2: Build function signature
  ctx->getOrLoadDialect<LLVM::LLVMDialect>();
  SmallVector<Type> argTypes, resTypes;
  buildTilingFunctionSignature(origTy, ctx, builder, argTypes, resTypes);

  // Step 3: Create and initialize tiling function
  auto f = createAndInitTilingFunc(originalKernel, argTypes, resTypes, builder);

  // Step 4: For simple case (no bands or static shape), just return
  if (bands.empty() || isStaticShape) {
    OpBuilder b(&f.getBody().front(), f.getBody().front().end());
    b.create<func::ReturnOp>(loc);
    tilingFunc = f;
    return success();
  }

  // Step 5: Build dynamic axis mappings (reuse existing function)
  std::vector<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &curBand = bands[bandIdx];
    allBandDynamicMappings.push_back(buildDynamicAxisMappingForBand(curBand, bandTileSizes, originalKernel));
  }

  // Step 6: Store tile sizes to memref (with constraint max for dynamic shapes)
  if (failed(storeTileSizesToMemref(f, originalKernel, bands, allBandTileSizes, allBandConstraintMaxs,
                                    allBandDynamicMappings,
                                    builder))) {
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
        bandDynamicMapping.push_back(
          {static_cast<unsigned>(argIndex), static_cast<unsigned>(dimIndex), upperBound});
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

// Helper: Compute dynamic tile size value (ceildiv(dim, 40))
static LogicalResult computeDynamicTileSizeValue(const DynamicAxisMapping &mapping, int constraintMax,
                                                 func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                 Value &result) {
  if (mapping.inputMemrefIndex == UINT_MAX || mapping.dimIndex == UINT_MAX) {
    return failure();
  }

  if (mapping.inputMemrefIndex >= originalKernel.getNumArguments()) {
    return failure();
  }

  Value memrefArg = originalKernel.getArgument(mapping.inputMemrefIndex);
  if (!isa<MemRefType>(memrefArg.getType())) {
    return failure();
  }

  Value dimIndexVal = builder.create<arith::ConstantIndexOp>(loc, mapping.dimIndex);
  Value dim = builder.create<memref::DimOp>(loc, memrefArg, dimIndexVal);

  if (constraintMax > 0) {
    // Use constraint upper bound: tile_size = min(constraintMax, dim)
    Value constraintMaxVal = builder.create<arith::ConstantIndexOp>(loc, constraintMax);
    // min(constraintMax, dim)
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, constraintMaxVal, dim);
    result = builder.create<arith::SelectOp>(loc, cmp, constraintMaxVal, dim);
  } else {
    // Fallback to original behavior: ceildiv(dim, kBlockDimSize)
    Value c40 = builder.create<arith::ConstantIndexOp>(loc, kBlockDimSize);
    result = builder.create<arith::CeilDivSIOp>(loc, dim, c40);
  }

  return success();
}

// Helper: Prepare tile sizes for static shape
static LogicalResult prepareTileSizesForStaticShape(func::FuncOp originalKernel, mlir::Location loc, OpBuilder &builder,
                                                    std::vector<SmallVector<mlir::scf::ForOp, 6>> &bandsToUse,
                                                    std::vector<SmallVector<Value, 6>> &allTileSizeValues,
                                                    std::vector<SmallVector<unsigned, 6>> &allBandTileSizesOut) {
  std::vector<SmallVector<unsigned, 6>> allBandTileSizes;
  std::vector<SmallVector<int, 6>> allBandConstraintMaxs;  // Constraint upper bounds for dynamic axes
  size_t levelToTile = 0;

  if (failed(calculateAllBandTileSizes(originalKernel, true, bandsToUse, allBandTileSizes, allBandConstraintMaxs,
                                       levelToTile))) {
    return failure();
  }

  // Build dynamic axis mappings for all bands
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

  // Output the original unsigned tile sizes
  allBandTileSizesOut = allBandTileSizes;

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

LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape) {
  auto loc = originalKernel.getLoc();

  // Step 1: Collect bands
  std::vector<SmallVector<mlir::scf::ForOp, 6>> bands;
  if (failed(collectBands(originalKernel, bands))) {
    return failure();
  }

  if (bands.empty()) {
    if (failed(wrapFunctionBodyWithFor(originalKernel, builder))) return failure();
    markInnermostLoopsWithVectorAttr(originalKernel, builder, kVectorSize);
    collectBands(originalKernel, bands);
  }

  // Step 2: Setup builder
  OpBuilder::InsertionGuard guard(builder);
  Block *body = &originalKernel.getBody().front();
  builder.setInsertionPointToStart(body);

  // Step 3: Prepare tile sizes
  // Always calculate tile sizes first to get unsigned values (with constraint max for dynamic axes)
  std::vector<SmallVector<unsigned, 6>> allBandTileSizesInt;
  std::vector<SmallVector<int, 6>> allBandConstraintMaxs;  // Constraint upper bounds for dynamic axes
  std::vector<SmallVector<mlir::scf::ForOp, 6>> bandsToUse = bands;
  size_t levelToTile = 0;
  if (failed(calculateAllBandTileSizes(originalKernel, true, bandsToUse, allBandTileSizesInt, allBandConstraintMaxs,
                                       levelToTile))) {
    return failure();
  }

  // Then prepare Value representations based on shape type
  std::vector<SmallVector<Value, 6>> allTileSizeValues;
  allTileSizeValues.reserve(bands.size());

  if (isStaticShape) {
    // Static shape: create Value constants from unsigned values
    std::vector<SmallVector<unsigned, 6>> tempTileSizes;  // Will be overwritten
    if (failed(
          prepareTileSizesForStaticShape(originalKernel, loc, builder, bandsToUse, allTileSizeValues, tempTileSizes))) {
      return failure();
    }
  } else {
    // Dynamic shape: load tile sizes from memref (but we already have unsigned values)
    std::vector<SmallVector<unsigned, 6>> tempTileSizes;  // Will be overwritten
    if (failed(prepareTileSizesFromMemref(originalKernel, bands, loc, builder, allTileSizeValues, tempTileSizes))) {
      return failure();
    }
  }

  // Step 4: Apply tiling to each band
  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &band = bandsToUse[bandIdx];
    const auto &tileSizeValues = allTileSizeValues[bandIdx];
    const auto &tileSizesInt = allBandTileSizesInt[bandIdx];

    std::map<int64_t, Value> constantCache;
    builder.setInsertionPointAfter(tileSizeValues.back().getDefiningOp());

    if (failed(applyTilingToBand(band, tileSizeValues, tileSizesInt, builder, constantCache, originalKernel))) {
      originalKernel.emitError("Failed to apply tiling to band " + std::to_string(bandIdx));
      return failure();
    }
  }

  // Step 5: Post-processing - mark attributes
  markInnermostLoopsWithVectorAttr(originalKernel, builder, kVectorSize);
  setBlockDimAttribute(originalKernel, builder);
  markOutermostLoopsForParallelMapping(originalKernel, builder);

  // Step 6: Inline reduce point loops (they execute only once, IV is unused)
  inlineReducePointLoops(originalKernel, builder);

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
  // For reduce loops: map to middle level loop IV instead of point loop IV
  // Also map region iter args to support loops with iter_args
  // Note: Create a mutable copy for initIVMapping
  SmallVector<mlir::scf::ForOp> tiledLoopsMutable(tiledLoops.begin(), tiledLoops.end());
  initIVMapping(band, tiledLoopsMutable, tileSizesNum, mapping);
  for (unsigned i = 0; i < forNum; ++i) {
    mlir::scf::ForOp origLoop = band[i];

    // For reduce loops, use the middle-level loop IV (level tileNum-1) instead of the point loop IV.
    // For normal loops, use point loop IV (tileSizesNum + i).
    unsigned tileNum = tileSizesNum / forNum;
    unsigned middleIdx = (tileNum > 1) ? ((tileNum - 1) * forNum + i) : (tileSizesNum + i);
    unsigned targetIdx = origLoop->hasAttr(kReductionLoopAttr) ? middleIdx : (tileSizesNum + i);

    mlir::scf::ForOp tiledLoop = tiledLoops[targetIdx];
    if (!origLoop || !tiledLoop) {
      continue;
    }

    // Map IV
    Value origIV = origLoop.getInductionVar();
    Value tiledIV = tiledLoop.getInductionVar();
    if (!mapping.contains(origIV)) {
      mapping.map(origIV, tiledIV);
    }

    // Map region iter args (for loops with iter_args)
    for (auto [origArg, tiledArg] : llvm::zip(origLoop.getRegionIterArgs(), tiledLoop.getRegionIterArgs())) {
      if (!mapping.contains(origArg)) {
        mapping.map(origArg, tiledArg);
      }
    }
  }

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
static void initIVMapping(llvm::ArrayRef<mlir::scf::ForOp> band, llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                          unsigned tileSizesNum, mlir::IRMapping &mapping) {
  unsigned forNum = band.size();
  unsigned tileNum = (forNum > 0) ? (tileSizesNum / forNum) : 0;

  for (unsigned i = 0; i < band.size(); ++i) {
    mlir::scf::ForOp loop = band[i];

    // For reduce loops, use the middle-level loop IV (level tileNum-1) instead of the point loop IV.
    // For normal loops, use point loop IV (tileSizesNum + i).
    unsigned middleIdx = (tileNum > 1) ? ((tileNum - 1) * forNum + i) : (tileSizesNum + i);
    unsigned targetIdx = loop->hasAttr(kReductionLoopAttr) ? middleIdx : (tileSizesNum + i);

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

    // Map loop results to corresponding point loop results.
    unsigned pointIdx = tileSizesNum + i;
    if (pointIdx < tiledLoops.size()) {
      mlir::scf::ForOp pointLoop = tiledLoops[pointIdx];
      if (pointLoop) {
        for (auto [origRes, tiledRes] : llvm::zip(loop.getResults(), pointLoop.getResults())) {
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
    std::transform(yieldOp.getOperands().begin(), yieldOp.getOperands().end(),
                   std::back_inserter(newOperands),
                   [&mapping](Value operand) { return remapOrSelf(operand, mapping); });

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

static LogicalResult collectReduceResultUserOps(mlir::scf::ForOp rootLoop, mlir::scf::ForOp outerLoop,
                                                llvm::SmallSet<Operation *, 16> &opSet) {
  Block *sourceBlock = outerLoop->getBlock();
  SmallVector<Value, 8> worklist(rootLoop.getResults().begin(), rootLoop.getResults().end());

  for (size_t i = 0; i < worklist.size(); ++i) {
    Value v = worklist[i];
    for (Operation *user : v.getUsers()) {
      if (user == rootLoop.getOperation()) {
        continue;
      }
      if (user->getBlock() != sourceBlock || user->isBeforeInBlock(outerLoop) || isa<func::ReturnOp>(user)) {
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

static LogicalResult sinkReduceLoopResultsToMiddleLevel(mlir::scf::ForOp rootLoop,
                                                        llvm::MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                                        unsigned tileSizesNum, unsigned bandSize,
                                                        mlir::OpBuilder &builder) {
  if (!isReductionLoopWithIterArgs(rootLoop)) {
    return success();
  }
  if (tiledLoops.empty()) {
    return failure();
  }

  mlir::scf::ForOp outerLoop = tiledLoops[0];
  if (!outerLoop) {
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
  if (failed(collectReduceResultUserOps(rootLoop, outerLoop, opSet))) {
    return failure();
  }

  if (opSet.empty()) {
    return success();
  }

  SmallVector<Operation *, 16> opsInOrder;
  collectOpsInOrderAfter(outerLoop.getOperation(), opSet, opsInOrder);
  moveOpsAfterLoop(middleLoop, opsInOrder, builder);
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

    // put into the parent loop of tile level 0 (index=i)
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

// Helper function to remap or return self if not mapped
[[maybe_unused]] static mlir::Value remapOrSelf(mlir::Value v, mlir::IRMapping &mapping) {
  if (!v) {
    return v;
  }
  if (mapping.contains(v)) {
    return mapping.lookup(v);
  }
  return v;
}

}  // namespace autotiling
}  // namespace mlir
