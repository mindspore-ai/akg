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
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/RegionUtils.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"

using llvm::SmallVector;
using llvm::SmallVectorImpl;
using llvm::DenseMap;
using mlir::autotiling::buildModelGraph;
using mlir::autotiling::buildNpuModelGraph;
using mlir::autotiling::getHeuristicTilingSolver;
using mlir::autotiling::getTileSizeWithSolver;
using mlir::autotiling::parseIr;
using mlir::autotiling::TilingTaskDesc;
using mlir::autotiling::TilingSolverPtr;
using hacc::KernelArgTypeAttr;
using hacc::KernelArgType;
using hacc::HACCFuncTypeAttr;
using hacc::HACCFuncType;

namespace mlir {
namespace autotiling {

// Forward declarations
static LogicalResult createTailBlockForBodyStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                                  std::map<int64_t, Value> &constantCache);
static LogicalResult createTailBlockStatic(mlir::scf::ForOp forOp, OpBuilder &builder,
                                           std::map<int64_t, Value> &constantCache);
static LogicalResult createTailBlockStaticImpl(mlir::scf::ForOp forOp, int64_t differenceUbAndLb,
                                               OpBuilder &builder,
                                               std::map<int64_t, Value> &constantCache);
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder);
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value dynamicBound,
                                                OpBuilder &builder,
                                                std::map<int64_t, Value> &constantCache);
static std::tuple<std::optional<int64_t>, mlir::Value, mlir::Value> getDifferenceUbAndLb(mlir::scf::ForOp forOp);
static std::optional<int64_t> getConstantIndexValue(mlir::Value value);
static std::optional<int64_t> getConstantStep(mlir::scf::ForOp forOp);
static void moveLoopBody(mlir::scf::ForOp src, mlir::scf::ForOp dest);

// Helper function to move loop body from src to dest
static void moveLoopBody(mlir::scf::ForOp src, mlir::scf::ForOp dest) {
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

// Helper function to get constant index value from Value
static std::optional<int64_t> getConstantIndexValue(mlir::Value value) {
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
static std::optional<int64_t> getConstantStep(mlir::scf::ForOp forOp) {
  return getConstantIndexValue(forOp.getStep());
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

// Structure to store dynamic axis mapping: {inputMemrefIndex, dimIndex}
struct DynamicAxisMapping {
  unsigned inputMemrefIndex;  // Index of the memref argument in originalKernel
  unsigned dimIndex;          // Dimension index within that memref
};

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
static void setTilingKeyAndDataArgAttrs(func::FuncOp func, unsigned keyIdx, unsigned tilingDataIdx,
                                         MLIRContext *ctx) {
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

// Helper function to collect a perfectly nested band starting from a root loop
// A perfectly nested band means each loop's body contains only the next loop (and terminator)
static void collectPerfectlyNestedBand(mlir::scf::ForOp rootLoop, SmallVector<mlir::scf::ForOp, 6> &band) {
  mlir::scf::ForOp currentLoop = rootLoop;
  while (currentLoop) {
    band.push_back(currentLoop);
    Block *body = currentLoop.getBody();
    if (!body || body->empty()) {
      break;
    }

    // Check if body contains only a single scf.for loop (perfectly nested)
    // Skip the terminator (scf.yield)
    Operation *firstOp = nullptr;
    for (Operation &op : body->without_terminator()) {
      if (firstOp == nullptr) {
        firstOp = &op;
      } else {
        // More than one operation means not perfectly nested
        currentLoop = nullptr;
        break;
      }
    }

    if (firstOp == nullptr) {
      // Empty body (except terminator)
      break;
    }

    // Check if the first operation is a scf.for loop
    if (auto nestedFor = dyn_cast<mlir::scf::ForOp>(firstOp)) {
      currentLoop = nestedFor;
    } else {
      // Not a loop, so this is the end of the perfectly nested band
      break;
    }
  }
}

// Helper function to collect bands from a block
// Only collects perfectly nested bands (each loop's body contains only the next loop)
static void collectBandsFromBlock(Block *block, std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands) {
  for (Operation &op : *block) {
    if (auto forOp = dyn_cast<mlir::scf::ForOp>(&op)) {
      // Check if this is an outermost loop (parent is not scf.for)
      Operation *parentOp = forOp->getParentOp();
      bool isOutermost = true;
      while (parentOp) {
        if (isa<mlir::scf::ForOp>(parentOp)) {
          isOutermost = false;
          break;
        }
        if (isa<mlir::func::FuncOp>(parentOp)) {
          break;
        }
        parentOp = parentOp->getParentOp();
      }

      // Only collect bands starting from outermost loops
      if (isOutermost) {
        SmallVector<mlir::scf::ForOp, 6> band;
        collectPerfectlyNestedBand(forOp, band);
        if (!band.empty()) {
          bands.push_back(band);
        }
      }
    }
  }
}

// Calculate all band tile sizes using auto-tiling
static LogicalResult calculateAllBandTileSizes(
    func::FuncOp funcOp,
    bool useAutoTiling,
    std::vector<SmallVector<mlir::scf::ForOp, 6>> &bands,
    std::vector<SmallVector<unsigned, 6>> &allBandTileSizes,
    size_t &levelToTile) {
  bands.clear();
  allBandTileSizes.clear();

  // Collect bands (only perfectly nested ones) using the same logic as collectBands
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

  // Check if dynamic shape (similar to isDynamicShape() member function)
  bool isDynamicShape = akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0;

  // BandCheck (for non-dynamic shapes)
  // TODO(yuziyu): Implement band checking for scf.for loops
  // For now, we skip it as it's a placeholder

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
  initGraph->setIsDynamicShape(isDynamicShape);
  initGraph->setTilingMode("auto");

  auto modelGraph = buildModelGraph(initGraph);
  auto solver = getHeuristicTilingSolver(modelGraph);
  levelToTile = modelGraph->levelToTile;

  // Calculate tile sizes for each band
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    SmallVector<mlir::scf::ForOp, 6> curBand = bands[bandIdx];
    SmallVector<unsigned, 6> bandTileSizes;

    for (size_t level = 0; level < levelToTile; ++level) {
      getTileSizeWithSolver(solver, curBand, &bandTileSizes,
                            TilingTaskDesc(bandIdx, level));
    }

    allBandTileSizes.push_back(bandTileSizes);
  }

  // Print all band tile sizes
  llvm::errs() << "=== All Band Tile Sizes ===\n";
  for (size_t i = 0; i < allBandTileSizes.size(); ++i) {
    llvm::errs() << "Band " << i << " tile sizes: [";
    for (size_t j = 0; j < allBandTileSizes[i].size(); ++j) {
      if (j > 0) llvm::errs() << ", ";
      if (allBandTileSizes[i][j] == static_cast<unsigned>(-1)) {
        llvm::errs() << "dynamic";
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
static void extractForInductionVarsStatic(ArrayRef<mlir::scf::ForOp> band,
                                          SmallVectorImpl<mlir::Value> *origLoopIVs) {
  origLoopIVs->clear();
  origLoopIVs->reserve(band.size());
  std::transform(band.begin(), band.end(), std::back_inserter(*origLoopIVs),
                 [](mlir::scf::ForOp forOp) { return forOp.getInductionVar(); });
}

// Helper function to construct tiled loop structure
static void constructTiledLoopStatic(mlir::scf::ForOp rootScfForOp, unsigned width,
                                     MutableArrayRef<mlir::scf::ForOp> tiledLoops,
                                     OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  mlir::Location loc = rootScfForOp.getLoc();
  mlir::Operation *topLoop = rootScfForOp.getOperation();
  mlir::scf::ForOp innermostPointLoop;

  // Create width number of nested loops
  for (unsigned i = 0; i < width; ++i) {
    builder.setInsertionPoint(topLoop);
    mlir::Value c0 = getOrCreateConstantStatic(loc, 0, builder, constantCache);
    mlir::Value c1 = getOrCreateConstantStatic(loc, 1, builder, constantCache);
    auto inits = rootScfForOp.getInitArgs();
    auto pointLoop = builder.create<mlir::scf::ForOp>(loc, c0, c0, c1, inits);
    // Insert topLoop before the terminator in pointLoop's body
    auto *terminator = pointLoop.getBody()->getTerminator();
    mlir::Block::iterator insertLoc = terminator ? terminator->getIterator() : pointLoop.getBody()->end();
    pointLoop.getBody()->getOperations().splice(insertLoc,
                                                      topLoop->getBlock()->getOperations(),
                                                      topLoop);
    tiledLoops[width - 1 - i] = pointLoop;
    topLoop = pointLoop.getOperation();
    if (i == 0) {
      innermostPointLoop = pointLoop;
    }
  }

  // Move the body of the innermost original loop to the innermost tiled loop
  // Find the innermost loop in the original band
  mlir::scf::ForOp innermostOrigLoop = rootScfForOp;
  while (true) {
    Block *body = innermostOrigLoop.getBody();
    if (body->getOperations().size() == 2) {
      Operation *firstOp = &body->front();
      if (auto nestedFor = dyn_cast<mlir::scf::ForOp>(firstOp)) {
        innermostOrigLoop = nestedFor;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  moveLoopBody(innermostOrigLoop, innermostPointLoop);
}

// Helper function to construct tiled index (bounds and steps) using tileSizeValues from memref
static void constructTiledIndexStatic(MutableArrayRef<mlir::scf::ForOp> newLoops,
                                      ArrayRef<mlir::scf::ForOp> band,
                                      ArrayRef<Value> tileSizeValues, OpBuilder &builder,
                                      std::map<int64_t, Value> &constantCache) {
  int bandSize = static_cast<int>(band.size());
  if (bandSize == 0 || tileSizeValues.size() == 0) {
    return;
  }

  mlir::Location loc = band[0]->getLoc();
  int tileNum = static_cast<int>(tileSizeValues.size()) / bandSize;

  for (int i = 0; i <= tileNum; ++i) {
    for (int j = 0; j < bandSize; ++j) {
      int curTile = i * bandSize + j;
      int lastTile = curTile - bandSize;

      if (curTile >= static_cast<int>(newLoops.size())) {
        continue;
      }
      mlir::scf::ForOp loop = newLoops[curTile];
      if (!loop) {
        continue;
      }

      mlir::OpBuilder loopBuilder(loop);
      mlir::Value lb, ub, step;
      mlir::ValueRange inits;

      if (i == 0) {
        mlir::scf::ForOp origLoop = band[j];
        lb = origLoop.getLowerBound();
        ub = origLoop.getUpperBound();
        if (curTile < static_cast<int>(tileSizeValues.size())) {
          step = tileSizeValues[curTile];
        } else {
          step = getOrCreateConstantStatic(loc, 1, loopBuilder, constantCache);
        }
        inits = origLoop.getInitArgs();
      } else if (i == tileNum) {
        if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) {
          continue;
        }
        mlir::scf::ForOp prevLoop = newLoops[lastTile];
        if (!prevLoop) {
          continue;
        }
        lb = prevLoop.getInductionVar();
        mlir::Value prevStep = prevLoop.getStep();
        ub = loopBuilder.create<mlir::arith::AddIOp>(loc, lb, prevStep);
        step = getOrCreateConstantStatic(loc, 1, loopBuilder, constantCache);
        inits = prevLoop.getResults();
      } else {
        if (lastTile < 0 || lastTile >= static_cast<int>(newLoops.size())) {
          continue;
        }
        mlir::scf::ForOp prevLoop = newLoops[lastTile];
        if (!prevLoop) {
          continue;
        }
        lb = prevLoop.getInductionVar();
        if (curTile < static_cast<int>(tileSizeValues.size())) {
          step = tileSizeValues[curTile];
          ub = loopBuilder.create<mlir::arith::AddIOp>(loc, lb, tileSizeValues[lastTile]);
        } else {
          step = getOrCreateConstantStatic(loc, 1, loopBuilder, constantCache);
          ub = loopBuilder.create<mlir::arith::AddIOp>(loc, lb, tileSizeValues[lastTile]);
        }
        inits = prevLoop.getResults();
      }

      loopBuilder.setInsertionPoint(loop);
      auto newLoop = loopBuilder.create<mlir::scf::ForOp>(loc, lb, ub, step, inits);

      auto *oldBody = loop.getBody();
      auto *newBody = newLoop.getBody();

      mlir::ValueRange yieldValues;
      mlir::scf::YieldOp oldYield = nullptr;
      if (oldBody) {
        auto *terminator = oldBody->getTerminator();
        if (terminator && isa<mlir::scf::YieldOp>(terminator)) {
          oldYield = cast<mlir::scf::YieldOp>(terminator);
          yieldValues = oldYield.getOperands();
        }
      }

      auto &newBodyOps = newBody->getOperations();
      if (!newBodyOps.empty()) {
        auto *defaultTerminator = newBody->getTerminator();
        if (defaultTerminator && isa<mlir::scf::YieldOp>(defaultTerminator)) {
          defaultTerminator->erase();
        }
      }

      auto &oldBodyOps = oldBody->getOperations();
      if (oldBodyOps.size() > 1) {
        auto terminatorIt = oldBody->end();
        --terminatorIt;
        mlir::Block::iterator insertLoc;
        auto *newTerminator = newBody->getTerminator();
        if (newTerminator) {
          insertLoc = newTerminator->getIterator();
        } else {
          insertLoc = newBodyOps.end();
        }
        newBodyOps.splice(insertLoc, oldBodyOps, oldBodyOps.begin(), terminatorIt);
        if (oldYield) {
          oldYield->erase();
        }
      } else if (oldBodyOps.size() == 1 && oldYield) {
        oldYield->erase();
      }

      auto *existingTerminator = newBody->getTerminator();
      if (existingTerminator && isa<mlir::scf::YieldOp>(existingTerminator)) {
        if (existingTerminator->getBlock() == newBody) {
          existingTerminator->erase();
        }
      }

      loopBuilder.setInsertionPointToEnd(newBody);
      if (newLoop.getNumRegionIterArgs() > 0) {
        if (oldYield && yieldValues.size() == newLoop.getNumRegionIterArgs()) {
          loopBuilder.create<mlir::scf::YieldOp>(loc, yieldValues);
        } else {
          loopBuilder.create<mlir::scf::YieldOp>(loc, newLoop.getRegionIterArgs());
        }
      } else {
        loopBuilder.create<mlir::scf::YieldOp>(loc);
      }

      loop.replaceAllUsesWith(newLoop.getResults());
      loop.erase();
      newLoops[curTile] = newLoop;
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
static LogicalResult createTailBlockStaticImpl(mlir::scf::ForOp forOp, int64_t differenceUbAndLb,
                                                OpBuilder &builder,
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

    newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(),
                                              forOp.getBody()->getOperations(),
                                              forOp.getBody()->begin(),
                                              std::prev(forOp.getBody()->end()));

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

    newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(),
                                             forOp.getBody()->getOperations(),
                                             forOp.getBody()->begin(),
                                             std::prev(forOp.getBody()->end()));

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
static LogicalResult createTailBlockDynamicImpl(mlir::scf::ForOp forOp, mlir::Value dynamicBound,
                                                OpBuilder &builder,
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

  newLoop.getBody()->getOperations().splice(newLoop.getBody()->begin(),
                                          forOp.getBody()->getOperations(),
                                          forOp.getBody()->begin(),
                                          std::prev(forOp.getBody()->end()));

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
                                       OpBuilder &builder, std::map<int64_t, Value> &constantCache) {
  if (band.empty() || tileSizeValues.empty()) {
    return success();
  }

  unsigned forNum = band.size();
  unsigned tileSizesNum = tileSizeValues.size();

  if (tileSizesNum == 0 || forNum == 0) {
    return success();
  }

  if (tileSizesNum > forNum && tileSizesNum % forNum == 0) {
    SmallVector<mlir::Value, 8> origLoopIVs;
    extractForInductionVarsStatic(band, &origLoopIVs);

    mlir::scf::ForOp rootScfForOp = band[0];
    unsigned width = tileSizesNum + forNum;
    SmallVector<mlir::scf::ForOp, 6> tiledLoops(width);

    constructTiledLoopStatic(rootScfForOp, width, tiledLoops, builder, constantCache);

    constructTiledIndexStatic(tiledLoops, band, tileSizeValues, builder, constantCache);

    for (unsigned i = 0; i < forNum; i++) {
      if (i < origLoopIVs.size() && (i + tileSizesNum) < tiledLoops.size()) {
        origLoopIVs[i].replaceAllUsesWith(tiledLoops[i + tileSizesNum].getInductionVar());
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
static void markInnermostLoopsWithVectorAttr(func::FuncOp funcOp, OpBuilder &builder) {
  funcOp->walk([&](mlir::scf::ForOp forOp) {
    if (isInnermostScfLoop(forOp)) {
      forOp->setAttr(kVectorAttr, builder.getI64IntegerAttr(4096));
    }
  });
}

// Create default tiling function
static LogicalResult createTilingFuncDefault(func::FuncOp originalKernel, OpBuilder &builder,
                                             func::FuncOp &tilingFunc, bool isStaticShape = false) {
  static constexpr int64_t kTilingStructMemrefSize = 64;

  auto *ctx = builder.getContext();
  auto loc = originalKernel.getLoc();
  auto origTy = originalKernel.getFunctionType();

  std::vector<SmallVector<mlir::scf::ForOp, 6>> bands;
  std::vector<SmallVector<unsigned, 6>> allBandTileSizes;
  size_t levelToTile = 0;

  if (failed(calculateAllBandTileSizes(originalKernel, true, bands, allBandTileSizes, levelToTile))) {
    return failure();
  }

  if (bands.empty() || isStaticShape) {
    ctx->getOrLoadDialect<LLVM::LLVMDialect>();

    SmallVector<Type> argTypes(origTy.getInputs().begin(), origTy.getInputs().end());

    auto i64Ty = builder.getI64Type();
    auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
    auto memrefTy = MemRefType::get({kTilingStructMemrefSize}, i64Ty);

    argTypes.push_back(llvmPtrTy);
    argTypes.push_back(memrefTy);

    SmallVector<Type> resTypes;

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

    OpBuilder b(&f.getBody().front(), f.getBody().front().end());

    Value tilingKey = f.getArgument(keyIdx);
    int64_t strategyIndex = getSelectedTilingStrategyIndex();
    Value strategyValue = b.create<arith::ConstantIntOp>(loc, strategyIndex, 64);
    b.create<LLVM::StoreOp>(loc, strategyValue, tilingKey);

    b.create<func::ReturnOp>(loc);

    tilingFunc = f;
    return success();
  }

  std::vector<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    SmallVector<mlir::scf::ForOp, 6> curBand = bands[bandIdx];
    std::vector<DynamicAxisMapping> bandDynamicMapping;

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      unsigned tileSize = bandTileSizes[tileIdx];
      if (tileSize == static_cast<unsigned>(-1)) {
        size_t axisIdx = tileIdx % curBand.size();
        mlir::scf::ForOp forOp = curBand[axisIdx];
        Value upperBound = forOp.getUpperBound();

        auto [argIndex, dimIndex] = traceDynamicUpperBound(upperBound, originalKernel);
        if (argIndex >= 0 && dimIndex >= 0) {
          bandDynamicMapping.push_back({static_cast<unsigned>(argIndex),
                                       static_cast<unsigned>(dimIndex)});
        } else {
          for (unsigned i = 0; i < originalKernel.getNumArguments(); ++i) {
            Value arg = originalKernel.getArgument(i);
            if (isa<MemRefType>(arg.getType())) {
              bandDynamicMapping.push_back({i, 0});
              break;
            }
          }
        }
      } else {
        bandDynamicMapping.push_back({UINT_MAX, UINT_MAX});
      }
    }

    allBandDynamicMappings.push_back(bandDynamicMapping);
  }

  ctx->getOrLoadDialect<LLVM::LLVMDialect>();

  SmallVector<Type> argTypes(origTy.getInputs().begin(), origTy.getInputs().end());

  auto i64Ty = builder.getI64Type();
  auto llvmPtrTy = LLVM::LLVMPointerType::get(ctx);
  auto memrefTy = MemRefType::get({kTilingStructMemrefSize}, i64Ty);

  argTypes.push_back(llvmPtrTy);
  argTypes.push_back(memrefTy);

  SmallVector<Type> resTypes;

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

  OpBuilder b(&f.getBody().front(), f.getBody().front().end());
  Value dataMem = f.getArgument(tilingDataIdx);

  Value tilingKey = f.getArgument(keyIdx);
  int64_t strategyIndex = getSelectedTilingStrategyIndex();
  Value strategyValue = b.create<arith::ConstantIntOp>(loc, strategyIndex, 64);
  b.create<LLVM::StoreOp>(loc, strategyValue, tilingKey);

  size_t memrefOffset = 0;
  for (size_t bandIdx = 0; bandIdx < bands.size(); ++bandIdx) {
    const auto &bandTileSizes = allBandTileSizes[bandIdx];
    const auto &bandDynamicMapping = allBandDynamicMappings[bandIdx];

    for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
      Value idx = b.create<arith::ConstantIndexOp>(loc, memrefOffset);

      unsigned tileSize = bandTileSizes[tileIdx];

      if (tileSize == static_cast<unsigned>(-1)) {
        const auto &mapping = bandDynamicMapping[tileIdx];

        if (mapping.inputMemrefIndex == UINT_MAX || mapping.dimIndex == UINT_MAX) {
          f.emitError("dynamic axis mapping not found for tile index " + std::to_string(tileIdx));
          return failure();
        }

        if (mapping.inputMemrefIndex >= f.getNumArguments() - 2) {
          f.emitError("invalid memref argument index " + std::to_string(mapping.inputMemrefIndex));
          return failure();
        }

        Value memrefArg = f.getArgument(mapping.inputMemrefIndex);
        if (!isa<MemRefType>(memrefArg.getType())) {
          f.emitError("argument at index " + std::to_string(mapping.inputMemrefIndex) + " is not a memref");
          return failure();
        }

        Value dimIndexVal = b.create<arith::ConstantIndexOp>(loc, mapping.dimIndex);
        Value dim = b.create<memref::DimOp>(loc, memrefArg, dimIndexVal);

        Value c39 = b.create<arith::ConstantIndexOp>(loc, 39);
        Value c40 = b.create<arith::ConstantIndexOp>(loc, 40);
        Value dimPlus39 = b.create<arith::AddIOp>(loc, dim, c39);
        Value step = b.create<arith::DivSIOp>(loc, dimPlus39, c40);

        Value stepI64 = b.create<arith::IndexCastOp>(loc, i64Ty, step);
        b.create<memref::StoreOp>(loc, stepI64, dataMem, ValueRange{idx});
      } else {
        Value val = b.create<arith::ConstantIntOp>(loc, static_cast<int64_t>(tileSize), 64);
        b.create<memref::StoreOp>(loc, val, dataMem, ValueRange{idx});
      }

      memrefOffset++;
    }
  }

  b.create<func::ReturnOp>(loc);

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

  out[0] = tilingFunc;
  return success();
}

int64_t getSelectedTilingStrategyIndex() {
  return 0;
}

LogicalResult applyTilingFromTilingFunc(func::FuncOp originalKernel, OpBuilder &builder, bool isStaticShape) {
  auto loc = originalKernel.getLoc();

  std::vector<SmallVector<mlir::scf::ForOp, 6>> bands;
  if (failed(collectBands(originalKernel, bands))) {
    return failure();
  }

  if (bands.empty()) {
    return success();
  }

  OpBuilder::InsertionGuard guard(builder);
  Block *body = &originalKernel.getBody().front();
  builder.setInsertionPointToStart(body);

  std::vector<SmallVector<Value, 6>> allTileSizeValues;
  allTileSizeValues.reserve(bands.size());

  std::vector<SmallVector<mlir::scf::ForOp, 6>> bandsToUse = bands;

  if (isStaticShape) {
    std::vector<SmallVector<unsigned, 6>> allBandTileSizes;
    size_t levelToTile = 0;
    std::vector<SmallVector<mlir::scf::ForOp, 6>> bandsForCalc;
    if (failed(calculateAllBandTileSizes(originalKernel, true, bandsForCalc, allBandTileSizes, levelToTile))) {
      return failure();
    }
    bandsToUse = bandsForCalc;

    std::vector<std::vector<DynamicAxisMapping>> allBandDynamicMappings;
    for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
      const auto &bandTileSizes = allBandTileSizes[bandIdx];
      SmallVector<mlir::scf::ForOp, 6> curBand = bandsToUse[bandIdx];
      std::vector<DynamicAxisMapping> bandDynamicMapping;

      for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
        unsigned tileSize = bandTileSizes[tileIdx];
        if (tileSize == static_cast<unsigned>(-1)) {
          size_t axisIdx = tileIdx % curBand.size();
          mlir::scf::ForOp forOp = curBand[axisIdx];
          Value upperBound = forOp.getUpperBound();

          auto [argIndex, dimIndex] = traceDynamicUpperBound(upperBound, originalKernel);
          if (argIndex >= 0 && dimIndex >= 0) {
            bandDynamicMapping.push_back({static_cast<unsigned>(argIndex),
                                         static_cast<unsigned>(dimIndex)});
          } else {
            for (unsigned i = 0; i < originalKernel.getNumArguments(); ++i) {
              Value arg = originalKernel.getArgument(i);
              if (isa<MemRefType>(arg.getType())) {
                bandDynamicMapping.push_back({i, 0});
                break;
              }
            }
          }
        } else {
          bandDynamicMapping.push_back({UINT_MAX, UINT_MAX});
        }
      }

      allBandDynamicMappings.push_back(bandDynamicMapping);
    }

    for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
      const auto &bandTileSizes = allBandTileSizes[bandIdx];
      const auto &bandDynamicMapping = allBandDynamicMappings[bandIdx];
      SmallVector<Value, 6> tileSizeValues;

      for (size_t tileIdx = 0; tileIdx < bandTileSizes.size(); ++tileIdx) {
        unsigned tileSize = bandTileSizes[tileIdx];
        Value tileSizeValue;

        if (tileSize == static_cast<unsigned>(-1)) {
          const auto &mapping = bandDynamicMapping[tileIdx];
          if (mapping.inputMemrefIndex == UINT_MAX || mapping.dimIndex == UINT_MAX) {
            originalKernel.emitError("dynamic axis mapping not found for tile index " + std::to_string(tileIdx));
            return failure();
          }

          Value memrefArg = originalKernel.getArgument(mapping.inputMemrefIndex);
          if (!isa<MemRefType>(memrefArg.getType())) {
            originalKernel.emitError("argument at index " +
                                     std::to_string(mapping.inputMemrefIndex) + " is not a memref");
            return failure();
          }

          Value dimIndexVal = builder.create<arith::ConstantIndexOp>(loc, mapping.dimIndex);
          Value dim = builder.create<memref::DimOp>(loc, memrefArg, dimIndexVal);

          Value c39 = builder.create<arith::ConstantIndexOp>(loc, 39);
          Value c40 = builder.create<arith::ConstantIndexOp>(loc, 40);
          Value dimPlus39 = builder.create<arith::AddIOp>(loc, dim, c39);
          tileSizeValue = builder.create<arith::DivSIOp>(loc, dimPlus39, c40);
        } else {
          tileSizeValue = builder.create<arith::ConstantIndexOp>(loc, static_cast<int64_t>(tileSize));
        }

        tileSizeValues.push_back(tileSizeValue);
      }

      allTileSizeValues.push_back(tileSizeValues);
    }
  } else {
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
      for (unsigned i = 0; i < bandTileSizesCount; ++i) {
        Value idx = builder.create<arith::ConstantIndexOp>(loc, memrefOffset + i);
        Value loaded = builder.create<memref::LoadOp>(loc, tileSizesMemref, ValueRange{idx});
        Value tileSizeIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), loaded);
        tileSizeValues.push_back(tileSizeIndex);
      }
      allTileSizeValues.push_back(tileSizeValues);
      memrefOffset += bandTileSizesCount;
    }
  }

  for (size_t bandIdx = 0; bandIdx < bandsToUse.size(); ++bandIdx) {
    const auto &band = bandsToUse[bandIdx];
    const auto &tileSizeValues = allTileSizeValues[bandIdx];

    std::map<int64_t, Value> constantCache;
    builder.setInsertionPointAfter(tileSizeValues.back().getDefiningOp());
    if (failed(applyTilingToBand(band, tileSizeValues, builder, constantCache))) {
      originalKernel.emitError("Failed to apply tiling to band " + std::to_string(bandIdx));
      return failure();
    }
  }

  markInnermostLoopsWithVectorAttr(originalKernel, builder);

  return success();
}

}  // namespace autotiling
}  // namespace mlir

