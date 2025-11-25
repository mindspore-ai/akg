/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/GPU/Transforms/AKGGPUMapping.h"

#include <deque>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <vector>
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Utils/AKGGlobalVars.hpp"
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

namespace mlir {
#define GEN_PASS_DEF_AKGGPUMAPPING
#define GEN_PASS_DECL_AKGGPUMAPPING
#include "akg/Dialect/GPU/Passes.h.inc"
}  // namespace mlir

using namespace akgglobal;

namespace mlir {

using scf::ForOp;
using scf::ParallelOp;
using json = nlohmann::json;

namespace gpu {
namespace akg {
constexpr auto kInferredConfig = "inferredConfig";
constexpr auto kKernelNameAttrKey = "sym_name";
constexpr auto kDynamicShapeSize = -1;
}  // namespace akg
using namespace akg;
using namespace mlir::akg::utils;
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
/// todo: Make this use x for the inner-most loop that is
/// distributed to map to x, the next innermost to y and the next innermost to
/// z.
static Processor getHardwareIdForMapping(MappingLevel level, int dimension) {
  if (dimension >= kNumHardwareIds || level == Sequential) {
    return Processor::Sequential;
  }
  switch (level) {
    case MapGrid:
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
    case MapBlock:
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
  bool isReductionAxis() const { return reductionDim >= 0; }
  bool isDynamicOuterAxis() const { return problemSize == 1 && isDynamicAxis; }
  bool needToMap() const { return problemSize > 1 || isDynamicAxis; }
  void dump() {
    llvm::outs() << "Task : Length = " << problemSize << " MapLevel = " << level << "\n";
    loopVar.dump();
  }
};

struct MappingTaskComparator {
  bool operator()(const MappingTask &a, const MappingTask &b) const { return a.op < b.op; }
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
  void loadGlobalMapping();
  void mapParallelOp(ParallelOp parallelOp, const std::vector<MappingTask> &result);
  bool saveMappingResultToJson();
  std::string getInferredConfigJson();
  std::pair<std::string, int> genAxisMappingId(Operation *axis);
  std::string getAkgKernelName();

  std::string device_target{kV100Device};
  std::deque<MappingTask> waitingList;
  std::map<ParallelOp, std::vector<MappingTask>, ParallelOpCmp> mapResults;
  std::vector<AxisInfo> axes;
  bool hasSequentialReduction{false};
  int proposedGrid{1};
  int proposedBlock{1};

 private:
  bool isDynamicShape() const;
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
      if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)){
          rewriter.setInsertionPoint(yieldOp);
          rewriter.replaceOpWithNewOp<scf::ReduceOp>(yieldOp);
      }
      return success();
    }
    return failure();
  }
};

bool hasNonZeroConstant(Operation *op) {
  unsigned int flag = 0;
  for (auto operand : op->getOperands()) {
    auto prevOp = operand.getDefiningOp();
    if (prevOp) {
      if (isa<arith::AddIOp>(op)) {
        if (isa<arith::ConstantOp>(prevOp)) {
          mlir::Attribute constantValue = prevOp->getAttr("value");
          if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constantValue)) {
            if (intAttr.getInt() != 0) {
              return true;
            }
          }
        }
      }
      flag |= (hasNonZeroConstant(prevOp) == false ? 0 : 1);
    }
  }
  return (bool)flag;
}

bool isPostFusionSingleStmt(Operation *op) {
  if (auto cmpi = dyn_cast<arith::CmpIOp>(op)) {
    // in affine stmt, post fusion should be `- xxx + 240 == 0`
    if (cmpi.getPredicate() != arith::CmpIPredicate::eq) {
      return false;
    }
    auto right = op->getOperand(1).getDefiningOp();
    if (isa<arith::ConstantOp>(right)) {
      mlir::Attribute constantValue = right->getAttr("value");
      if (auto intAttr = dyn_cast<mlir::IntegerAttr>(constantValue)) {
        if (intAttr.getInt() != 0) {
          return true;
        }
      }
    }
    auto left = op->getOperand(0).getDefiningOp();
    return hasNonZeroConstant(left);
  }
  return false;
}

bool isPostFusionMultiStmt(Operation *op) {
  if (auto andi = dyn_cast<arith::AndIOp>(op)) {
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
  (void)ifOp.walk([&](mindspore::KeepArgsOp op) {
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
  if (CommonUtils::isIfConditionRelatedToContent(ifOp)) {
    shouldKeepIfOp = true;
  } else {
    shouldKeepIfOp = false;
  }
}

static bool IsAncestorOrEqual(Operation *a, Operation *b) {
  auto blockA = a->getBlock();
  Operation *curOp = b;
  while (curOp) {
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
    SmallVector<Operation *, 8> axesA;
    CommonUtils::collectRelatedAxes(operand, axesA);
    for (auto a : axesA) {
      if (targetOp == a) {
        return false;
      }
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
  while (curOp) {
    if (isa<scf::ParallelOp>(curOp)) {
      targetOp = curOp;
    }
    curOp = curOp->getParentOp();
  }
  return targetOp;
}

static void FixForLogicToGpuParallel(Region &region) {
  SmallVector<Operation *, 8> ifOpsToHoist;
  OpBuilder opBuilder(region);
  auto funcOp = region.getParentOp();

  SmallVector<Operation *, 8> ifOps;
  (void)region.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp.getOperation()); });
  for (auto opInit : ifOps) {
    auto ifOp = dyn_cast<scf::IfOp>(opInit);
    opBuilder.setInsertionPoint(ifOp.getOperation());
    bool shouldKeepIfOp = true;
    bool postFusionMode = true;
    checkIfOpStatus(ifOp, shouldKeepIfOp, postFusionMode);
    if (!shouldKeepIfOp) {
      Operation *parentOp = ifOp.getOperation()->getParentOp();
      while (parentOp) {
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

        SmallVector<Operation *, 8> previousOps;
        CommonUtils::getAllPreviousRelatedOps(ifOp, previousOps);

        ifOp.erase();
        for (auto op : previousOps) {
          op->erase();
        }
      } else {
        // this scf.if is the outer most scf.if. we should move out of
        // the scf.if.then block to outer most threadIdx.x

        // get the outermost thread parallelOp
        Operation *outermostSequentialOp = nullptr;
        Operation *curOp = ifOp.getOperation();
        Operation *outermostParallelOp = getOutermostParallelOp(curOp);
        while (curOp) {
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
              if (!isa<scf::YieldOp>(op)) {
                if (!canMoveOpOutOfTarget(&op, curOp)) {
                  canMove = false;
                  break;
                }
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
        if (!outermostSequentialOp) {
          if (funcOp->hasAttr(kEnableParallelReduce) &&
              funcOp->getAttrOfType<BoolAttr>(mlir::akg::utils::kEnableParallelReduce).getValue() == false) {
            continue;
          } else {
            outermostSequentialOp = ifOp.getOperation();
          }
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
        SmallVector<Operation *, 8> previousOps;
        CommonUtils::getAllPreviousRelatedOps(ifOp, previousOps);

        ifOp.erase();
        for (auto op : previousOps) {
          op->erase();
        }
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
      return std::make_pair("", kDynamicShapeSize);
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
      return std::make_pair(mapId, kDynamicShapeSize);
    }
  }
  return std::make_pair("", kDynamicShapeSize);
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
    funcOp.walk([&](Operation *op) {
      if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
        auto [configId, configSize] = genAxisMappingId(op);
        if (configId.empty() || configSize == kDynamicShapeSize) {
          return;
        }
        jsonResults[configId] = configSize;
      }
    });
    return jsonResults.dump();
  }

  auto getArgIndex = [&](Value memref) -> int {
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
  ShapeAlignTool &tool = ShapeAlignTool::getInstance();
  std::map<size_t, Operation *> tensors;
  funcOp.walk([&](Operation *op) {
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

  // We can only use outputs' dim to calculate mapping result because inputs' dim may be incorrect due to implicit
  // broadcast. We sort tensor by the tid so that we can ensure the output's mapping result will replace the inputs'.
  for (auto it : tensors) {
    auto tid = it.first;
    mlir::ValueRange indices;
    if (auto load = dyn_cast<memref::LoadOp>(it.second)) {
      indices = load.getIndices();
    } else if (auto store = dyn_cast<memref::StoreOp>(it.second)) {
      indices = store.getIndices();
    } else if (auto vload = dyn_cast<vector::LoadOp>(it.second)) {
      indices = vload.getIndices();
    } else if (auto vstore = dyn_cast<vector::StoreOp>(it.second)) {
      indices = vstore.getIndices();
    }
    for (size_t dimId = 0; dimId < indices.size(); ++dimId) {
      SmallVector<Operation *, 8> relatedAxes;
      CommonUtils::collectRelatedAxes(indices[dimId], relatedAxes);
      std::string dynConfigId;
      std::string symbolPart = tool.getCurrShapeInfo(tid)[dimId];
      int64_t constPart = 1;
      for (size_t axisId = 0; axisId < relatedAxes.size(); ++axisId) {
        auto axis = relatedAxes[axisId];
        auto [configId, configSize] = genAxisMappingId(axis);
        if (configId.empty()) {
          continue;
        }
        if (configSize != kDynamicShapeSize) {
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
  return jsonResults.dump();
}

std::string AKGGPUMappingLoops::getAkgKernelName() {
  std::string defaultName = "akg_kernel";
  for (auto attr : getOperation()->getAttrs()) {
    auto keyStr = dyn_cast<StringAttr>(attr.getName()).getValue().str();
    if (keyStr != kKernelNameAttrKey) {
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
  (void)DirUtils::CheckOrCreateDirectory("./akg_kernel_meta/");
  std::string output_filename = "./akg_kernel_meta/" + kernelName + ".json";
  if (llvm::writeToOutput(output_filename, [&](llvm::raw_ostream &OS) -> llvm::Error {
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
  funcOp->walk([&](Operation *redOp) {
    if (redOp->hasAttr(kReductionAxesStr)) {
      ArrayAttr attrs = dyn_cast<ArrayAttr>(redOp->getAttr(kReductionAxesStr));
      SmallVector<Operation *, 8> parallelOps;
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
        parallelOps[idx]->setAttr("reduceLoop", builder.getUnitAttr());
      }
      if (!redOp->hasAttr(kEnableParallelReduce)) {
        (void)redOp->emitWarning("This reduction op does not have a \"gpu_parallel_reduce\" mark, set to false.");
        funcOp->setAttr(kEnableParallelReduce, builder.getBoolAttr(false));
      } else {
        funcOp->setAttr(kEnableParallelReduce, redOp->getAttr(kEnableParallelReduce));
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
    if (task.op->hasAttr(kLoopTag)) {
      auto name = cast<StringAttr>(task.op->getAttr(kLoopTag)).getValue().str();
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
      CommonUtils::getOperatorType(getOperation()) != OperatorTemplate::Reduce) {
    llvm::outs() << "WARNING " << getAkgKernelName() << " totalMapSize " << totalMapSize << " totalProblemSize "
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
    region.walk([&](ParallelOp parallelOp) {
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
  while (curOp) {
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
    bool isReduceAxis = (parallelOp.getOperation()->hasAttr("reduceLoop")) ? true : false;
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
  int currBlock = 1;
  int currGrid = 1;
  bool tryBlock = true;
  std::set<MappingTask, MappingTaskComparator> unsolvedTasks;
  std::map<MappingLevel, int> mapLevelCount;
  auto MarkSolved = [&](MappingTask task, const MappingLevel &level) {
    auto actual_level = singleProcess ? MappingLevel::Sequential : level;
    task.level = actual_level;
    if (actual_level != MappingLevel::Sequential) {
      task.mapDim = mapLevelCount[actual_level]++;
      tryBlock = !tryBlock;
      if (actual_level == MappingLevel::MapGrid) {
        currGrid *= task.problemSize;
        llvm::outs() << "Successfully map " << task.problemSize << " task to grid, currGrid = " << currGrid
                     << ", flip to block\n";
      } else if (actual_level == MappingLevel::MapBlock) {
        currBlock *= task.problemSize;
        llvm::outs() << "Successfully map " << task.problemSize << " task to block, currBlock = " << currBlock
                     << ", flip to grid\n";
      }
    } else {
      task.mapDim = -1;
      llvm::outs() << "Successfully map " << task.problemSize << " task to sequential\n";
    }
    mapResults[task.op].push_back(task);
  };

  std::tie(proposedGrid, proposedBlock) = StrategyHelper::getProposalParallelSize(problemSize, device_target);

  llvm::outs() << " problemSize = " << problemSize << ", proposedGrid = " << proposedGrid
               << " proposedBlock = " << proposedBlock << "\n";

  auto MarkUnsolved = [this, &tryBlock, &unsolvedTasks](MappingTask task) {
    if (tryBlock) {
      llvm::outs() << "Try map block fail, push task back.\n";
      waitingList.push_back(task);
    } else {
      llvm::outs() << "Try map grid fail, push task front.\n";
      waitingList.push_front(task);
    }
    (void)unsolvedTasks.insert(task);
    tryBlock = !tryBlock;
  };
  auto totalAvailableBlocks = GpuInfo::getInstance(device_target).getTotalAvailableBlocks();
  auto maxGrids = GpuInfo::getInstance(device_target).getMaxGrids();
  while (!waitingList.empty()) {
    if (tryBlock) {
      auto task = waitingList.back();
      waitingList.pop_back();
      bool disableThreadMapping = task.isReductionAxis();
      if (!task.needToMap() || disableThreadMapping) {
        MarkSolved(task, MappingLevel::Sequential);
        continue;
      }
      bool badPerformance = currBlock * task.problemSize > proposedBlock;
      bool invalid = currBlock * task.problemSize > totalAvailableBlocks;
      bool transferred = unsolvedTasks.find(task) != unsolvedTasks.end();
      if (task.isDynamicOuterAxis() || invalid || (badPerformance && !transferred)) {
        llvm::outs() << "currBlock " << currBlock << " * " << task.problemSize << " >= proposedBlock(" << proposedBlock
                     << ")\n";
        MarkUnsolved(task);
      } else {
        MarkSolved(task, MappingLevel::MapBlock);
      }
    } else {
      auto task = waitingList.front();
      waitingList.pop_front();
      auto currDim = mapLevelCount[MappingLevel::MapGrid];
      if (!task.needToMap() || currDim >= static_cast<int>(maxGrids.size()) || task.problemSize > maxGrids[currDim]) {
        MarkSolved(task, MappingLevel::Sequential);
        continue;
      }
      if (currGrid * task.problemSize <= proposedGrid || unsolvedTasks.find(task) != unsolvedTasks.end()) {
        llvm::outs() << "Successfully map " << task.problemSize << " task to grid, currGrid = " << currGrid
                     << ", flip to block\n";
        MarkSolved(task, MappingLevel::MapGrid);
      } else {
        MarkUnsolved(task);
      }
    }
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
  SmallVector<ParallelLoopDimMappingAttr, 4> attrs;
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
