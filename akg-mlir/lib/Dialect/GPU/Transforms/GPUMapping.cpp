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

#include "akg/Dialect/GPU/Transforms/GPUMapping.h"

#include <deque>
#include <map>
#include <set>
#include <vector>
#include <nlohmann/json.hpp>
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Utils/GlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Utils/IOHelper.hpp"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/ParallelLoopMapper.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "akg/Utils/Constants.h"

namespace mlir {
#define GEN_PASS_DEF_AKGGPUMAPPING
#define GEN_PASS_DECL_AKGGPUMAPPING
#include "akg/Dialect/GPU/Passes.h.inc"

// using namespace akgglobal;  // NOLINT(build/namespaces)
using scf::ForOp;
using scf::ParallelOp;
using json = nlohmann::json;
using ShapeAlignTool = akgglobal::ShapeAlignTool;
using GpuScheduleTool = akgglobal::GpuScheduleTool;
using AxisInfo = akgglobal::AxisInfo;
using GpuInfo = mlir::akg::utils::GpuInfo;

namespace gpu {
namespace akg {
constexpr auto kInferredConfig = "inferredConfig";
constexpr auto kKernelNameAttrKey = "sym_name";
constexpr auto kDynamicShapeSize = -1;
}  // namespace akg
// using namespace akg;  // NOLINT(build/namespaces)
// using namespace mlir::akg::utils;  // NOLINT(build/namespaces)
using MappingLevel = mlir::akg::utils::MappingLevel;
using GpuCommonUtils = mlir::akg::utils::GpuCommonUtils;
using StrategyHelper = mlir::akg::utils::StrategyHelper;

namespace {
struct ParallelOpCmp {
  bool operator()(mlir::scf::ParallelOp lhs, mlir::scf::ParallelOp rhs) const {
    for (auto [lhsVar, rhsVar] : llvm::zip(lhs.getInductionVars(), rhs.getInductionVars())) {
      return lhsVar.getAsOpaquePointer() < rhsVar.getAsOpaquePointer();
    }
    return false;
  }
};

bool isConstant(mlir::Value value) {
  auto constantOp = dyn_cast_or_null<mlir::arith::ConstantOp>(value.getDefiningOp());
  return constantOp != nullptr;
}

int getIntConst(mlir::Value value) {
  auto constValueAttr = value.getDefiningOp()->getAttr("value");
  return dyn_cast<IntegerAttr>(constValueAttr).getInt();
}

int getMaxIntConst(mlir::Value value) {
  if (isConstant(value)) {
    return getIntConst(value);
  }
  int maxIntConst = 1;
  if (auto select = dyn_cast<mlir::arith::SelectOp>(value.getDefiningOp())) {
    auto trueValue = select.getTrueValue();
    if (isConstant(trueValue)) {
      maxIntConst = std::max<int>(maxIntConst, getIntConst(trueValue));
    }
    auto falseValue = select.getFalseValue();
    if (isConstant(falseValue)) {
      maxIntConst = std::max<int>(maxIntConst, getIntConst(falseValue));
    }
  }
  return maxIntConst;
}
}  // namespace

namespace {
// todo(baiji): -----Start COPY from ParallelLoopMapper.cpp -----
static constexpr int kNumHardwareIds = 3;

/// Computed the hardware id to use for a given mapping level. Will
/// assign x,y and z hardware ids for the first 3 dimensions and use
/// sequential after.
/// Make this use x for the inner-most loop that is
/// distributed to map to x, the next innermost to y and the next innermost to
/// z.
static Processor getHardwareIdForMapping(MappingLevel level, int dimension) {
  if (dimension >= kNumHardwareIds || level == mlir::akg::utils::MappingLevel::Sequential) {
    return Processor::Sequential;
  }
  switch (level) {
    case mlir::akg::utils::MappingLevel::MapGrid:
      switch (dimension) {
        case 0:
          return Processor::BlockX;
        case 1:
          return Processor::BlockY;
        case 2:
          return Processor::BlockZ;
        default:
          return Processor::Sequential;
      }
      break;
    case mlir::akg::utils::MappingLevel::MapBlock:
      switch (dimension) {
        case 0:
          return Processor::ThreadX;
        case 1:
          return Processor::ThreadY;
        case 2:
          return Processor::ThreadZ;
        default:
          return Processor::Sequential;
      }
    default: {
    }
  }
  return Processor::Sequential;
}
// todo(baiji): ----- End COPY from ParallelLoopMapper.cpp -----
}  // namespace
namespace {
struct MappingTask {
  MappingTask(ParallelOp op, mlir::Value loopVar, int problemSize, int reductionDim = -1)
      : op(op), loopVar(loopVar), problemSize(problemSize), reductionDim(reductionDim) {}
  ParallelOp op;
  mlir::Value loopVar;
  int problemSize;  // = (upperBound - lowerBound) / step
  MappingLevel level{MappingLevel::Unknown};
  int mapDim{0};  // 0 = x, 1 = y, 2 = z, 3+ = Sequential
  int reductionDim{-1};
  bool isDynamicAxis{false};
  [[nodiscard]] bool isReductionAxis() const { return reductionDim >= 0; }
  [[nodiscard]] bool isDynamicOuterAxis() const { return problemSize == 1 && isDynamicAxis; }
  [[nodiscard]] bool needToMap() const { return problemSize > 1 || isDynamicAxis; }
  void dump() {
    llvm::dbgs() << "Task : Length = " << problemSize << " MapLevel = " << static_cast<int>(level) << "\n";
    loopVar.dump();
  }
};

struct MappingTaskComparator {
  bool operator()(const MappingTask &a, const MappingTask &b) const { return a.op < b.op; }
};

struct MappingState {
  int currBlock{1};
  int currGrid{1};
  bool tryBlock{true};
  std::set<MappingTask, MappingTaskComparator> unsolvedTasks;
  std::map<MappingLevel, int> mapLevelCount;
  int totalAvailableBlocks{0};
  std::vector<int64_t> maxGrids;
};

struct AKGGPUMappingLoops : public impl::AKGGPUMappingBase<AKGGPUMappingLoops> {
  AKGGPUMappingLoops() {}
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mindspore::MindSporeDialect>();
    registry.insert<GPUDialect>();
  }

  void createMappingTask(ParallelOp parallelOp);
  void solveMapping();
  void markSolved(MappingTask task, const MappingLevel &level, MappingState &state);
  void markUnsolved(MappingTask task, MappingState &state);
  void solveBlockMappingTask(MappingState &state);
  void solveGridMappingTask(MappingState &state);
  void loadGlobalMapping();
  void mapParallelOp(ParallelOp parallelOp, const std::vector<MappingTask> &result);
  bool saveMappingResultToJson();
  std::string getInferredConfigJson();
  void collectDynamicTensorsAndIndices(func::FuncOp funcOp, std::map<size_t, Operation *> &tensors);
  void updateJsonWithTensorMapping(func::FuncOp funcOp, const std::map<size_t, Operation *> &tensors,
                                   json &jsonResults);
  std::pair<std::string, int> genAxisMappingId(Operation *axis);
  std::string getAkgKernelName();

  std::string device_target{mlir::akg::utils::kV100Device};
  std::deque<MappingTask> waitingList;
  std::map<ParallelOp, std::vector<MappingTask>, ParallelOpCmp> mapResults;
  std::vector<AxisInfo> axes;
  bool hasSequentialReduction{false};
  int proposedGrid{1};
  int proposedBlock{1};

 private:
  [[nodiscard]] bool isDynamicShape() const;
};

struct SCFForToParallelPattern : public RewritePattern {
  explicit SCFForToParallelPattern(MLIRContext *context) : RewritePattern(ForOp::getOperationName(), 1, context) {}

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (auto forOp = dyn_cast<ForOp>(op)) {
      auto parallelOp = rewriter.create<ParallelOp>(forOp.getLoc(), mlir::ValueRange(forOp.getLowerBound()),
                                                    mlir::ValueRange(forOp.getUpperBound()),
                                                    mlir::ValueRange(forOp.getStep()), nullptr);
      parallelOp.getRegion().takeBody(forOp.getRegion());
      Operation *newOp = parallelOp.getOperation();

      for (const auto &attr : op->getAttrs()) {
        newOp->setAttr(attr.getName(), attr.getValue());
      }
      rewriter.replaceOp(op, parallelOp.getResults());

      Operation *terminator = parallelOp.getRegion().getBlocks().front().getTerminator();
      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
        rewriter.setInsertionPoint(yieldOp);
        rewriter.replaceOpWithNewOp<scf::ReduceOp>(yieldOp);
      }
      return success();
    }
    return failure();
  }
};

static bool isNonZeroConstantOp(Operation *op) {
  if (!isa<arith::ConstantOp>(op)) {
    return false;
  }
  mlir::Attribute constantValue = op->getAttr("value");
  auto intAttr = dyn_cast<mlir::IntegerAttr>(constantValue);
  return intAttr && intAttr.getInt() != 0;
}

bool hasNonZeroConstant(Operation *op) {
  unsigned int flag = 0;
  for (auto operand : op->getOperands()) {
    auto prevOp = operand.getDefiningOp();
    if (prevOp != nullptr) {
      if (isa<arith::AddIOp>(op) && isNonZeroConstantOp(prevOp)) {
        return true;
      }
      flag |= (!hasNonZeroConstant(prevOp) ? 0 : 1);
    }
  }
  return static_cast<bool>(flag);
}

bool isPostFusionSingleStmt(Operation *op) {
  if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
    // in affine stmt, post fusion should be `- xxx + 240 == 0`
    if (cmpi.getPredicate() != arith::CmpIPredicate::eq) {
      return false;
    }
    auto right = op->getOperand(1).getDefiningOp();
    if (isNonZeroConstantOp(right)) {
      return true;
    }
    auto left = op->getOperand(0).getDefiningOp();
    return hasNonZeroConstant(left);
  }
  return false;
}

bool isPostFusionMultiStmt(Operation *op) {
  if (dyn_cast<arith::AndIOp>(op)) {
    for (auto operand : op->getOperands()) {
      if (isPostFusionMultiStmt(operand.getDefiningOp())) {
        return true;
      }
    }
    return false;
  }
  return isPostFusionSingleStmt(op);
}

void checkIfOpStatus(scf::IfOp ifOp, bool &shouldKeepIfOp, bool &postFusionMode) {
  // check whether the scf.if has mindspore.keepargs {BoundaryIf} inside.
  bool hasBoundaryIf = false;
  (void)ifOp.walk([&hasBoundaryIf](mindspore::KeepArgsOp op) {
    if (op.getOperation()->hasAttr("BoundaryIf")) {
      hasBoundaryIf = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (hasBoundaryIf) {
    shouldKeepIfOp = true;
    return;
  }
  postFusionMode = isPostFusionMultiStmt(ifOp.getOperand().getDefiningOp());
  shouldKeepIfOp = CommonUtils::isIfConditionRelatedToContent(ifOp);
}

static bool IsAncestorOrEqual(Operation *a, Operation *b) {
  auto blockA = a->getBlock();
  Operation *curOp = b;
  while (curOp != nullptr) {
    auto blockB = curOp->getBlock();
    if (blockA == blockB) {
      return true;
    }
    curOp = curOp->getParentOp();
  }
  return false;
}

static bool canMoveOpOutOfTarget(Operation *op, Operation *targetOp) {
  for (auto operand : op->getOperands()) {
    SmallVector<Operation *, kSmallVectorSizeEight> axesA;
    CommonUtils::collectRelatedAxes(operand, axesA);
    if (llvm::any_of(axesA, [targetOp](Operation *op) { return op == targetOp; })) {
      return false;
    }
  }

  // three cases about alloc/dealloc:
  // (1) scf.if content has all of alloc & use & dealloc, we can move them step by step;
  // (2) scf.if content has only use, alloc/dealloc is outside, we can move scf.if out;
  // (3) the move of scf.if ops break the relationship of alloc-use, stop the move.

  for (auto operand : op->getOperands()) {
    if (auto parentOp = operand.getDefiningOp()) {
      if (isa<memref::AllocOp>(parentOp)) {
        if (!IsAncestorOrEqual(parentOp, targetOp)) {
          return false;
        }
      }
    }
  }

  return true;
}

static Operation *getOutermostParallelOp(Operation *op) {
  Operation *curOp = op;
  Operation *targetOp = nullptr;
  while (curOp != nullptr) {
    if (isa<scf::ParallelOp>(curOp)) {
      targetOp = curOp;
    }
    curOp = curOp->getParentOp();
  }
  return targetOp;
}

static void handleOutermostIfOp(Region &region, scf::IfOp ifOp, Operation *funcOp, bool postFusionMode) {
  OpBuilder opBuilder(region);

  // this scf.if is the outer most scf.if. we should move out of
  // the scf.if.then block to outer most threadIdx.x

  // get the outermost thread parallelOp
  Operation *outermostSequentialOp = nullptr;
  Operation *curOp = ifOp.getOperation();
  Operation *outermostParallelOp = getOutermostParallelOp(curOp);
  while (curOp != nullptr) {
    if (auto parallelOp = dyn_cast<scf::ParallelOp>(curOp)) {
      // we can not move ops out of scf.parallel
      if (parallelOp.getOperation() == outermostParallelOp) {
        break;
      }
      if (gpu::GpuAttrUtils::getProcessorFromParallelOp(curOp) != gpu::Processor::Sequential) {
        break;
      }
      bool canMove = true;
      for (auto &op : llvm::make_early_inc_range(ifOp.getThenRegion().front())) {
        if (isa<scf::YieldOp>(op)) {
          continue;
        }
        if (!canMoveOpOutOfTarget(&op, curOp)) {
          canMove = false;
          break;
        }
      }
      if (canMove) {
        outermostSequentialOp = curOp;
      } else {
        break;
      }
    }
    curOp = curOp->getParentOp();
  }

  // does not exist sequential-for
  if (outermostSequentialOp == nullptr) {
    if (funcOp->hasAttr(mlir::akg::utils::kEnableParallelReduce) &&
        !funcOp->getAttrOfType<BoolAttr>(mlir::akg::utils::kEnableParallelReduce).getValue()) {
      return;
    }
    outermostSequentialOp = ifOp.getOperation();
  }

  if (!postFusionMode) {
    opBuilder.setInsertionPoint(outermostSequentialOp);
  } else {
    opBuilder.setInsertionPointAfter(outermostSequentialOp);
  }
  for (auto &op : llvm::make_early_inc_range(ifOp.getThenRegion().front())) {
    if (!isa<scf::YieldOp>(op)) {
      mlir::Operation *clonedOp = opBuilder.clone(op);
      op.replaceAllUsesWith(clonedOp);
    }
  }
  SmallVector<Operation *, kSmallVectorSizeEight> previousOps;
  CommonUtils::getAllPreviousRelatedOps(ifOp, previousOps);

  ifOp.erase();
  for (auto op : previousOps) {
    op->erase();
  }
}

static void FixForLogicToGpuParallel(Region &region) {
  SmallVector<Operation *, kSmallVectorSizeEight> ifOpsToHoist;
  OpBuilder opBuilder(region);
  auto funcOp = region.getParentOp();

  SmallVector<Operation *, kSmallVectorSizeEight> ifOps;
  (void)region.walk([&ifOps](scf::IfOp ifOp) { ifOps.push_back(ifOp.getOperation()); });
  for (auto opInit : ifOps) {
    auto ifOp = dyn_cast<scf::IfOp>(opInit);
    opBuilder.setInsertionPoint(ifOp.getOperation());
    bool shouldKeepIfOp = true;
    bool postFusionMode = true;
    checkIfOpStatus(ifOp, shouldKeepIfOp, postFusionMode);
    if (!shouldKeepIfOp) {
      Operation *parentOp = ifOp.getOperation()->getParentOp();
      while (parentOp != nullptr) {
        if (isa<scf::IfOp>(parentOp)) {
          break;
        }
        if (isa<scf::ParallelOp>(parentOp)) {
          break;
        }
        // todo(yanzhi): what' else?
        parentOp = parentOp->getParentOp();
      }
      // nested scf.if cases
      if (isa<scf::IfOp>(parentOp)) {
        opBuilder.setInsertionPointAfter(ifOp.getOperation());
        for (auto &op : llvm::make_early_inc_range(ifOp.getThenRegion().front())) {
          if (!isa<scf::YieldOp>(op)) {
            mlir::Operation *clonedOp = opBuilder.clone(op);
            op.replaceAllUsesWith(clonedOp);
          }
        }
        SmallVector<Operation *, kSmallVectorSizeEight> previousOps;
        CommonUtils::getAllPreviousRelatedOps(ifOp, previousOps);

        ifOp.erase();
        for (auto op : previousOps) {
          op->erase();
        }
      } else {
        // this scf.if is the outer most scf.if. we should move out of
        // the scf.if.then block to outer most threadIdx.x
        handleOutermostIfOp(region, ifOp, funcOp, postFusionMode);
      }
    }
  }
}

bool AKGGPUMappingLoops::isDynamicShape() const {
  return akgglobal::ShapeAlignTool::getInstance().getFuncArgSizes() > 0;
}

// Generate a string key for each mapping level and dim.
// e.g. MappingLevel::MapGrid + Dim0 will be translated to "blockIdx.x"
//      and MappingLevel::MapBlock + Dim1 will be translated to "threadIdx.y"
std::pair<std::string, int> AKGGPUMappingLoops::genAxisMappingId(Operation *op) {
  std::map<MappingLevel, std::string> levelMap = {
    {MappingLevel::MapGrid, "blockIdx"}, {MappingLevel::MapBlock, "threadIdx"}, {MappingLevel::Sequential, "Seq"}};
  std::map<int, std::string> dimMap = {{0, ".x"}, {1, ".y"}, {2, ".z"}};
  if (auto axis = dyn_cast<ParallelOp>(op)) {
    auto it = mapResults.find(axis);
    if (it == mapResults.end()) {
      llvm::errs() << "No mapping for axis, error.\n";
      return std::make_pair("", mlir::gpu::akg::kDynamicShapeSize);
    }
    for (auto res : it->second) {
      if (levelMap.find(res.level) == levelMap.end()) {
        continue;
      }
      std::string mapId = levelMap[res.level];
      // skip seq cases, will handle them later
      if (dimMap.find(res.mapDim) != dimMap.end() && res.level != MappingLevel::Sequential) {
        mapId += dimMap[res.mapDim];
      }
      // dynamic axis (with upper bound unknown) will keep problemSize to 1
      if (!res.isDynamicAxis || res.problemSize > 1) {
        return std::make_pair(mapId, res.problemSize);
      }
      return std::make_pair(mapId, mlir::gpu::akg::kDynamicShapeSize);
    }
  }
  return std::make_pair("", mlir::gpu::akg::kDynamicShapeSize);
}

static std::string updateSeqConfigId(const json &jsonResults, const std::string &symbolPart, const int64_t &constPart) {
  int maxSeq = 0;
  std::pair<std::string, int64_t> newPair = {symbolPart, constPart};
  for (const auto &item : jsonResults.items()) {
    if (item.key().find("Seq.") != std::string::npos) {
      int seqNum = std::stoi(item.key().substr(4));
      if (seqNum > maxSeq) {
        maxSeq = seqNum;
      }

      std::pair<std::string, int64_t> value = {item.value()[0].get<std::string>(), item.value()[1].get<int64_t>()};
      if (value == newPair) {
        return item.key();
      }
    }
  }

  return "Seq." + std::to_string(maxSeq + 1);
}

void AKGGPUMappingLoops::collectDynamicTensorsAndIndices(func::FuncOp funcOp, std::map<size_t, Operation *> &tensors) {
  auto getArgIndex = [&funcOp](Value memref) -> int {
    size_t i = 0;
    for (auto arg : funcOp.getBody().front().getArguments()) {
      if (arg == memref) {
        return static_cast<int>(i);
      }
      mlir::Value alloc = Value();
      GpuCommonUtils::findAllocOpForFuncArg(alloc, funcOp, arg);
      GpuCommonUtils::findExpandShapeOpForFuncArg(alloc, funcOp, arg);
      if (alloc && alloc == memref) {
        return static_cast<int>(i);
      }
      ++i;
    }
    return -1;
  };

  funcOp.walk([&getArgIndex, &tensors](Operation *op) {
    int tensorId = -1;
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      tensorId = getArgIndex(load.getMemref());
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      tensorId = getArgIndex(store.getMemref());
    } else if (auto vload = dyn_cast<vector::LoadOp>(op)) {
      tensorId = getArgIndex(vload.getBase());
    } else if (auto vstore = dyn_cast<vector::StoreOp>(op)) {
      tensorId = getArgIndex(vstore.getBase());
    }
    if (tensorId < 0) {
      return;
    }
    auto tid = static_cast<size_t>(tensorId);
    tensors[tid] = op;
  });
}

void AKGGPUMappingLoops::updateJsonWithTensorMapping(func::FuncOp funcOp, const std::map<size_t, Operation *> &tensors,
                                                     json &jsonResults) {
  ShapeAlignTool &tool = ShapeAlignTool::getInstance();

  for (auto it : tensors) {
    auto tid = it.first;
    Operation *tensorOp = it.second;

    mlir::ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(tensorOp)) {
      indices = load.getIndices();
    } else if (auto store = dyn_cast<memref::StoreOp>(tensorOp)) {
      indices = store.getIndices();
    } else if (auto vload = dyn_cast<vector::LoadOp>(tensorOp)) {
      indices = vload.getIndices();
    } else if (auto vstore = dyn_cast<vector::StoreOp>(tensorOp)) {
      indices = vstore.getIndices();
    } else {
      continue;
    }
    for (size_t dimId = 0; dimId < indices.size(); ++dimId) {
      SmallVector<Operation *, kSmallVectorSizeEight> relatedAxes;
      CommonUtils::collectRelatedAxes(indices[dimId], relatedAxes);
      std::string dynConfigId;
      std::string symbolPart = tool.getCurrShapeInfo(tid)[dimId];
      int64_t constPart = 1;
      for (auto axis : relatedAxes) {
        auto [configId, configSize] = genAxisMappingId(axis);
        if (configId.empty()) {
          continue;
        }
        if (configSize != akg::kDynamicShapeSize) {
          constPart *= configSize;
          if (jsonResults.find(configId) != jsonResults.end()) {
            jsonResults[configId] = configSize;
          }
        } else {
          if (dynConfigId.empty()) {
            dynConfigId = configId;
          }
          assert(dynConfigId == configId && "We only allow one symbol when solving dynamic mapping.");
        }
      }

      if (!symbolPart.empty() && !dynConfigId.empty()) {
        if (dynConfigId == "Seq") {
          dynConfigId = updateSeqConfigId(jsonResults, symbolPart, constPart);
        }
        jsonResults[dynConfigId] = std::make_pair(symbolPart, constPart);
      }
    }
  }
}

// Infer the mapping config for each dimension of each input tensor.
// The result will be organize in a `[Tensor[Dim[MapConfigs,],],]` form, e.g.:
// {"inferredConfig":[[["Block.y.32","Grid.x"],["Block.x.32","Grid.y.24"]],[["Block.x.32","Grid.y.24"]]]}
// in which there are two tensors and the first tensor has two dimensions; in the first dimension we have
// a dynamic-mapped config Grid.x that satisfied the expr `Grid.x * Block.y (which is 32) == Tensor0.Dim0.Shape`.
std::string AKGGPUMappingLoops::getInferredConfigJson() {
  json jsonResults;
  jsonResults["blockIdx.x"] = 1;
  jsonResults["blockIdx.y"] = 1;
  jsonResults["blockIdx.z"] = 1;
  jsonResults["threadIdx.x"] = 1;
  jsonResults["threadIdx.y"] = 1;
  jsonResults["threadIdx.z"] = 1;
  func::FuncOp funcOp = getOperation();

  if (!isDynamicShape()) {
    funcOp.walk([this, &jsonResults](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        auto [configId, configSize] = genAxisMappingId(op);
        if (configId.empty() || configSize == akg::kDynamicShapeSize) {
          return;
        }
        jsonResults[configId] = configSize;
      }
    });
    return jsonResults.dump();
  }

  std::map<size_t, Operation *> tensors;
  collectDynamicTensorsAndIndices(funcOp, tensors);
  updateJsonWithTensorMapping(funcOp, tensors, jsonResults);
  return jsonResults.dump();
}

std::string AKGGPUMappingLoops::getAkgKernelName() {
  std::string defaultName = "akg_kernel";
  for (auto attr : getOperation()->getAttrs()) {
    auto keyStr = dyn_cast<StringAttr>(attr.getName()).getValue().str();
    if (keyStr != mlir::gpu::akg::kKernelNameAttrKey) {
      continue;
    }
    return dyn_cast<StringAttr>(attr.getValue()).getValue().str();
  }
  return defaultName;
}

// Dump the mapping result to json file.
bool AKGGPUMappingLoops::saveMappingResultToJson() {
  std::string res = getInferredConfigJson();
  if (res.empty()) {
    llvm::report_fatal_error(llvm::StringRef("Infer config failed."));
  }
  auto kernelName = getAkgKernelName();
  (void)IOHelper::CheckOrCreateDirectory("./akg_kernel_meta/");
  std::string output_filename = "./akg_kernel_meta/" + kernelName + ".json";
  if (llvm::writeToOutput(output_filename, [&res](llvm::raw_ostream &OS) -> llvm::Error {
        OS << res;
        return llvm::Error::success();
      })) {
    llvm::report_fatal_error(llvm::StringRef("Write json file to " + output_filename + " failed."));
    return false;
  }
  return true;
}

static void SetRedutionMarkToParallelOp(Operation *funcOp) {
  OpBuilder builder(funcOp);
  funcOp->walk([&builder, &funcOp](Operation *redOp) {
    if (redOp->hasAttr(kReductionAxesStr)) {
      ArrayAttr attrs = dyn_cast<ArrayAttr>(redOp->getAttr(kReductionAxesStr));
      SmallVector<Operation *, kSmallVectorSizeEight> parallelOps;
      auto curOp = redOp;
      while (curOp) {
        if (isa<scf::ParallelOp>(curOp)) {
          parallelOps.push_back(curOp);
        }
        curOp = curOp->getParentOp();
      }
      std::reverse(parallelOps.begin(), parallelOps.end());
      for (auto attr : attrs) {
        auto idx = dyn_cast<IntegerAttr>(attr).getInt();
        parallelOps[idx]->setAttr(kReductionLoopAttr, builder.getUnitAttr());
      }
      if (!redOp->hasAttr(mlir::akg::utils::kEnableParallelReduce)) {
        (void)redOp->emitWarning("This reduction op does not have a \"gpu_parallel_reduce\" mark, set to false.");
        funcOp->setAttr(mlir::akg::utils::kEnableParallelReduce, builder.getBoolAttr(false));
      } else {
        funcOp->setAttr(mlir::akg::utils::kEnableParallelReduce,
                        redOp->getAttr(mlir::akg::utils::kEnableParallelReduce));
      }
    }
  });
}

// Load the mapping results that solved from AutoTiling.
// The results are passed down through GpuScheduleTool defined in akgglobal.
void AKGGPUMappingLoops::loadGlobalMapping() {
  int64_t totalMapSize = 1;
  int64_t totalProblemSize = 1;
  auto &gpuTool = GpuScheduleTool::getInstance();
  for (auto task : waitingList) {
    if (task.op->hasAttr(akgglobal::kLoopTag)) {
      auto name = cast<StringAttr>(task.op->getAttr(akgglobal::kLoopTag)).getValue().str();
      auto mapRes = GpuScheduleTool::getInstance().getMappingResult(name);
      totalProblemSize *= task.problemSize;
      if (mapRes.first == "GpuGrid") {
        task.level = MappingLevel::MapGrid;
        task.mapDim = mapRes.second;
        totalMapSize *= task.problemSize;
      } else if (mapRes.first == "GpuBlock") {
        task.level = MappingLevel::MapBlock;
        task.mapDim = mapRes.second;
        totalMapSize *= task.problemSize;
      } else {
        task.level = MappingLevel::Sequential;
      }
      if (gpuTool.isRuntimeVar(task.problemSize)) {
        auto var = gpuTool.getRuntimeArgument(task.problemSize);
        var.mapping = mapRes.first;
        var.mapDim = std::to_string(task.mapDim);
        gpuTool.updateRuntimeArgument(var);
      }
    }
    mapResults[task.op].push_back(task);
  }
  const int64_t factor = 16;
  if (totalMapSize * factor < totalProblemSize &&
      CommonUtils::getOperatorType(getOperation()) != OperatorTemplate::Reduction) {
    llvm::dbgs() << "WARNING " << getAkgKernelName() << " totalMapSize " << totalMapSize << " totalProblemSize "
                 << totalProblemSize << ", may have performance issue.\n";
  }
}

void AKGGPUMappingLoops::runOnOperation() {
  // 1. collect all of for loops and rewrite them to scf.parallel with temp attr: try_to_parallel
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();
  mlir::RewritePatternSet patterns(context);
  (void)patterns.insert<SCFForToParallelPattern>(context);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    signalPassFailure();
  }

  SetRedutionMarkToParallelOp(funcOp);
  for (Region &region : getOperation()->getRegions()) {
    // 2. find the root parallelOp to build mapping task top-down
    region.walk([this](ParallelOp parallelOp) {
      if (!parallelOp->getParentOfType<ParallelOp>()) {
        createMappingTask(parallelOp);
      }
    });
    if (GpuScheduleTool::getInstance().hasGlobalConfig()) {
      loadGlobalMapping();
    } else {
      solveMapping();
    }
    for (auto it : mapResults) {
      mapParallelOp(it.first, it.second);
    }
    // 5. fix the scf.for logic to parallel logic. Since the original scf.for is SIMD logic; while
    // gpu backend's scf.parallel should use SIMT logic. this func try to solve this issue by scanning
    // all of ops and rewrite some of them as SIMT code.
    FixForLogicToGpuParallel(region);

    // 6. todo(baji): use a option to control `If is dynamic:`
    (void)saveMappingResultToJson();
  }
}

static int getNestedNum(Operation *op) {
  auto num = 0;
  auto curOp = op->getParentOp();
  while (curOp != nullptr) {
    if (isa<scf::ParallelOp>(curOp)) {
      num++;
    }
    curOp = curOp->getParentOp();
  }
  return num;
}

void AKGGPUMappingLoops::createMappingTask(ParallelOp parallelOp) {
  for (auto [loopVar, lowerBoundVar, upperBoundVar, stepVar] : llvm::zip(
         parallelOp.getInductionVars(), parallelOp.getLowerBound(), parallelOp.getUpperBound(), parallelOp.getStep())) {
    size_t dim = getNestedNum(parallelOp.getOperation());
    bool isReduceAxis = parallelOp.getOperation()->hasAttr(kReductionLoopAttr);
    int reductionDim = isReduceAxis ? static_cast<int>(dim) : -1;
    auto lbConst = getMaxIntConst(lowerBoundVar);
    auto ubConst = getMaxIntConst(upperBoundVar);
    auto stepConst = getMaxIntConst(stepVar);
    if (stepConst == 0) {
      llvm::errs() << "Step cannot be zero.";
      continue;
    }
    auto problemSize = (ubConst - lbConst) / stepConst;
    auto task = MappingTask(parallelOp, loopVar, problemSize, reductionDim);
    task.isDynamicAxis = (!isConstant(lowerBoundVar) || !isConstant(upperBoundVar) || !isConstant(stepVar));
    waitingList.push_back(task);
  }
  for (Operation &op : *parallelOp.getBody()) {
    if (ParallelOp nested = dyn_cast<ParallelOp>(op)) {
      createMappingTask(nested);
    }
  }
}

// / Get the mapping level: outer-most loops map to Grid; inner-most loops map to Block;
// / Currently, we use a heuristic algorithm to solve the mapping level:
// / - we first get a `proposalGrid` and `proposalBlock` size;
// / - then we start to bind parallel loop to Block first and swap to Grid for next loop if success;
// / - if binding current loop makes the currBlock or currGrid exceed proposalSize,
// /   then we swap the mapping level and mark the loop as unsolved and it will be solved next round.
void AKGGPUMappingLoops::solveMapping() {
  int problemSize = 1;
  (void)std::for_each(waitingList.begin(), waitingList.end(),
                      [&problemSize](auto task) { problemSize *= task.problemSize; });

  std::tie(proposedGrid, proposedBlock) = StrategyHelper::getProposalParallelSize(problemSize, device_target);

  llvm::dbgs() << " problemSize = " << problemSize << ", proposedGrid = " << proposedGrid
               << " proposedBlock = " << proposedBlock << "\n";

  MappingState state;
  state.totalAvailableBlocks = GpuInfo::getInstance(device_target).getTotalAvailableBlocks();
  state.maxGrids = GpuInfo::getInstance(device_target).getMaxGrids();

  while (!waitingList.empty()) {
    if (state.tryBlock) {
      solveBlockMappingTask(state);
    } else {
      solveGridMappingTask(state);
    }
  }
}

void AKGGPUMappingLoops::markSolved(MappingTask task, const MappingLevel &level, MappingState &state) {
  auto actual_level = singleProcess ? MappingLevel::Sequential : level;
  task.level = actual_level;
  if (actual_level != MappingLevel::Sequential) {
    task.mapDim = state.mapLevelCount[actual_level]++;
    state.tryBlock = !state.tryBlock;
    if (actual_level == MappingLevel::MapGrid) {
      state.currGrid *= task.problemSize;
      llvm::dbgs() << "Successfully map " << task.problemSize << " task to grid, currGrid = " << state.currGrid
                   << ", flip to block\n";
    } else if (actual_level == MappingLevel::MapBlock) {
      state.currBlock *= task.problemSize;
      llvm::dbgs() << "Successfully map " << task.problemSize << " task to block, currBlock = " << state.currBlock
                   << ", flip to grid\n";
    }
  } else {
    task.mapDim = -1;
    llvm::dbgs() << "Successfully map " << task.problemSize << " task to sequential\n";
  }
  mapResults[task.op].push_back(task);
}

void AKGGPUMappingLoops::markUnsolved(MappingTask task, MappingState &state) {
  if (state.tryBlock) {
    llvm::dbgs() << "Try map block fail, push task back.\n";
    waitingList.push_back(task);
  } else {
    llvm::dbgs() << "Try map grid fail, push task front.\n";
    waitingList.push_front(task);
  }
  (void)state.unsolvedTasks.insert(task);
  state.tryBlock = !state.tryBlock;
}

void AKGGPUMappingLoops::solveBlockMappingTask(MappingState &state) {
  auto task = waitingList.back();
  waitingList.pop_back();
  bool disableThreadMapping = task.isReductionAxis();
  if (!task.needToMap() || disableThreadMapping) {
    markSolved(task, MappingLevel::Sequential, state);
    return;
  }
  bool badPerformance = state.currBlock * task.problemSize > proposedBlock;
  bool invalid = state.currBlock * task.problemSize > state.totalAvailableBlocks;
  bool transferred = state.unsolvedTasks.find(task) != state.unsolvedTasks.end();
  if (task.isDynamicOuterAxis() || invalid || (badPerformance && !transferred)) {
    llvm::dbgs() << "currBlock " << state.currBlock << " * " << task.problemSize << " >= proposedBlock("
                 << proposedBlock << ")\n";
    markUnsolved(task, state);
  } else {
    markSolved(task, MappingLevel::MapBlock, state);
  }
}

void AKGGPUMappingLoops::solveGridMappingTask(MappingState &state) {
  auto task = waitingList.front();
  waitingList.pop_front();
  auto currDim = state.mapLevelCount[MappingLevel::MapGrid];
  if (!task.needToMap() || currDim >= static_cast<int>(state.maxGrids.size()) ||
      task.problemSize > state.maxGrids[currDim]) {
    markSolved(task, MappingLevel::Sequential, state);
    return;
  }
  if (state.currGrid * task.problemSize <= proposedGrid ||
      state.unsolvedTasks.find(task) != state.unsolvedTasks.end()) {
    markSolved(task, MappingLevel::MapGrid, state);
  } else {
    markUnsolved(task, state);
  }
}

/// Add mapping information to the given parallelOp.
/// For each mappingLevel, map loop to dimension `x`, `y` and `z` in order.
/// Note that if the mapping dimension exceed 3, it will not be mapped and remain sequential.
/// E.g.
/// for grid.x
///  for grid.y
///   for grid.z
///    for sequential
///     for sequential
///      for block.z
///       for block.y
///        for block.x
///          body
void AKGGPUMappingLoops::mapParallelOp(ParallelOp parallelOp, const std::vector<MappingTask> &result) {
  FunctionOpInterface funcOp = getOperation();
  MLIRContext *ctx = parallelOp.getContext();
  // NOTE(baiji): manually load GPUDialect to avoid segfault during `b.getAttr<ParallelLoopDimMappingAttr>`
  ctx->loadDialect<GPUDialect>();

  Builder b(ctx);
  SmallVector<ParallelLoopDimMappingAttr, kSmallVectorSizeFour> attrs;
  attrs.reserve(parallelOp.getNumLoops());
  if (parallelOp.getNumLoops() != result.size()) {
    llvm::errs() << "parallelOp.getNumLoops() != mapResults.size(): " << parallelOp.getNumLoops() << " vs "
                 << mapResults.size();
    return;
  }
  for (int i = 0, e = parallelOp.getNumLoops(); i < e; ++i) {
    auto mapLevel = result[i].level;
    auto mapDim = result[i].mapDim;
    auto problemSize = result[i].problemSize;
    auto id = getHardwareIdForMapping(mapLevel, mapDim);
    auto attr = b.getAttr<ParallelLoopDimMappingAttr>(id, b.getDimIdentityMap(), b.getDimIdentityMap());
    attrs.push_back(attr);
    auto processorStr = stringifyProcessor(id);
    if (processorStr != "sequential") {
      if (result[i].isDynamicAxis && problemSize == 1) {
        funcOp->setAttr(processorStr, b.getI32IntegerAttr(-1));
      } else {
        funcOp->setAttr(processorStr, b.getI32IntegerAttr(problemSize));
      }
    }
  }
  (void)setMappingAttr(parallelOp, attrs);
}
}  // namespace
}  // namespace gpu
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createAKGGPUMapping() {
  return std::make_unique<gpu::AKGGPUMappingLoops>();
}
