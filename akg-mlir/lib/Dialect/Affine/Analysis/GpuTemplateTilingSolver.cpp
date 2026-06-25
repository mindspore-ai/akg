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

#include "akg/Dialect/Affine/Analysis/GpuTemplateTilingSolver.h"

#include <vector>
#include "akg/Utils/GlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
namespace akg {
namespace autotiling {
using mlir::ReduceDirection;
using mlir::autotiling::AxisPtr;
using mlir::autotiling::ConfigPos;
using mlir::autotiling::ConfigPtr;
using mlir::autotiling::GpuBlock;
using mlir::autotiling::GpuGrid;
using mlir::autotiling::kGpuBlockCfg;
using mlir::autotiling::kGpuGridCfg;
using mlir::autotiling::kGpuSeqCfg;

// Template reduction-X or reduction-All strategy
std::tuple<int, int> GpuTemplateTilingSolver::getProperRedConfigsX(int reductionSize, bool useAtmoicReturn) {
  int blockNum = 0;
  if (useAtmoicReturn) {
    blockNum = (reductionSize - 1) / kMinReductionLengthForAtomic + 1;
    reductionSize = (reductionSize - 1) / blockNum + 1;
  }
  int threadNum = kNum32 > reductionSize ? reductionSize : kNum32;
  while (threadNum * kProperAccSeqNumX < reductionSize && threadNum < kMaxThreadsNum) {
    threadNum *= kNum2;
  }
  return std::make_tuple(blockNum, threadNum);
}

// Template parallel strategy
int GpuTemplateTilingSolver::getProperParallelAxesThread(const int &len, const bool &isReduceY,
                                                         const int &blockDimsLeft,
                                                         const bool &enableParallelReduction) {
  if (!isReduceY && enableParallelReduction) {
    return 0;
  }
  if (blockDimsLeft == 0) {
    return 0;
  }
  if (isReduceY) {
    return kDefaultNonReductionThreadNumY;
  }

  if (len < kMinReductionLengthForParallel) {
    auto base = 1;
    auto thread = kNum512;
    while (base < len) {
      base *= kNum2;
      thread /= kNum2;
    }
    return thread;
  }
  return kDefaultNonReductionThreadNumX;
}

// Template reduction-Y strategy
std::tuple<int, int> GpuTemplateTilingSolver::getProperRedConfigsY(int reductionSize, bool useAtmoicReturn) {
  int blockNum = 0;
  int SeqNum = 1;
  if (useAtmoicReturn) {
    int accSeq = 1;
    if (reductionSize < kNum256) {
      accSeq = kNum16;
    } else if (reductionSize < kNum1024) {
      accSeq = kNum32;
    } else {
      accSeq = kNum64;
    }
    blockNum = (reductionSize - 1) / accSeq + 1;
    reductionSize = (reductionSize - 1) / blockNum + 1;
  }
  SeqNum = reductionSize;
  return std::make_tuple(blockNum, SeqNum);
}

void GpuTemplateTilingSolver::SolveRedAxesWithoutThreadReduction(std::vector<AxisPtr> &axes,
                                                                 const std::vector<int> &processOrder,
                                                                 const std::vector<int> &redFlags,
                                                                 const std::vector<int> &dynFlags) {
  auto &tool = akgglobal::PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  int num = axes.size();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (redFlags[i] != 0) {
      auto a = axes[i];
      auto fullTile = (dynFlags[i] != 0) ? tool.getOnePrimeWithIdxUpdate() : a->range.second;
      if (dynFlags[i] != 0) {
        auto arg = gpuTool.addRuntimeArgument(fullTile);
        arg.mark = "reduce-small-seq";
        gpuTool.updateRuntimeArgument(arg);
      }
      a->doExtraTile();
      auto tilel0 = a->tryGetConfig(1);
      tilel0->value = fullTile;
      auto tilel1 = a->tryGetConfig(0);
      tilel1->value = fullTile;
      std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};
      a->setMappings(axisMap);
    }
  }
}

void GpuTemplateTilingSolver::SolveAxesLeft(std::vector<AxisPtr> &axes) {
  int num = axes.size();
  for (int i = 0; i < num; i++) {
    if (axes[i]->mappings.empty()) {
      auto a = axes[i];
      a->doExtraTile();
      auto tilel0 = a->tryGetConfig(1);
      tilel0->value = a->range.second;
      auto tilel1 = a->tryGetConfig(0);
      tilel1->value = a->range.second;
      // mark left axes as sequential
      std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};
      a->setMappings(axisMap);
    }
  }
}

static void initPrimes(std::vector<int> &primes, akgglobal::PrimeNumTool &tool) {
  primes.push_back(tool.getOnePrimeWithIdxUpdate());
  primes.push_back(tool.getOnePrimeWithIdxUpdate());
  primes.push_back(tool.getOnePrimeWithIdxUpdate());
}

struct AxisTileContext {
  AxisPtr axis;
  ConfigPtr threadTile;
  std::vector<std::string> axisMap;
  int len;
  std::vector<int> primes;
  bool isDynamic;
  int i;
  int num;
  bool handleRedAxis;
};

static void processDynamicThreadUse(AxisTileContext &ctx, akgglobal::GpuScheduleTool &gpuTool) {
  auto blockcfg = std::make_shared<GpuBlock>("DynBlock");
  auto dynTile = ctx.primes[0];
  ctx.threadTile->value = dynTile;
  blockcfg->value = dynTile;
  ctx.axis->configs[blockcfg->type].push_back(blockcfg);
  ctx.axisMap[kNum2] = kGpuBlockCfg;
  auto argBlock = gpuTool.addRuntimeArgument(dynTile);
  argBlock.mark = (ctx.i == ctx.num - 1) ? "thread-last" : "thread";
  argBlock.mark = (ctx.handleRedAxis ? "reduce-" : "parallel-") + argBlock.mark;
  gpuTool.updateRuntimeArgument(argBlock);
}

static void processFullThreadUse(AxisPtr &a, ConfigPtr &threadTile, int &len, int &threadNum,
                                 std::vector<std::string> &axisMap) {
  threadTile->value = len;
  auto blockcfg = std::make_shared<GpuBlock>("Manual");
  blockcfg->value = len;
  blockcfg->index = ConfigPos::kInner;
  a->configs[blockcfg->type].push_back(blockcfg);
  axisMap[kNum2] = kGpuBlockCfg;
  threadNum = threadNum / len;
  len = 1;
}

static void processPartThreadUse(AxisPtr &a, ConfigPtr &threadTile, int &len, int &threadNum,
                                 std::vector<std::string> &axisMap) {
  threadTile->value = threadNum;
  auto blockcfg = std::make_shared<GpuBlock>("Manual");
  blockcfg->value = threadNum;
  blockcfg->index = ConfigPos::kInner;
  a->configs[blockcfg->type].push_back(blockcfg);
  axisMap[kNum2] = kGpuBlockCfg;
  len = (len - 1) / threadNum + 1;
  threadNum = 0;
}

static void processDynamicBlockUse(AxisPtr &a, const std::vector<int> &primes, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("DynGrid");
  auto dynTile = primes[1];  // a placeholder prime for blockidx
  gridcfg->value = dynTile;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
}

static void processFullBlockUse(AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  gridcfg->value = len;
  gridcfg->index = ConfigPos::kOuter;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
  if (len != 0) {
    blockNum = blockNum / len;
  } else {
    blockNum = 0;
  }
  len = 1;
}

static void processPartBlockUse(AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  gridcfg->value = blockNum;
  gridcfg->index = ConfigPos::kOuter;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
  len = (len - 1) / blockNum + 1;
  blockNum = 0;
}

static void processLeftDynamic(ConfigPtr &seqTile, const ConfigPtr &threadTile, const bool &handleRedAxis,
                               const std::vector<int> &primes, akgglobal::GpuScheduleTool &gpuTool) {
  auto prime = primes[2];
  seqTile->value = prime * threadTile->value;
  auto arg0 = gpuTool.addRuntimeArgument(prime);
  arg0.mark = handleRedAxis ? "reduce-x-seq" : "parallel-seq";
  gpuTool.updateRuntimeArgument(arg0);
  auto arg1 = gpuTool.addRuntimeArgument(seqTile->value);
  arg1.mark = handleRedAxis ? "product" : "1";
  gpuTool.updateRuntimeArgument(arg1);
}

static bool shouldMapToThread(int blockDimsLeft, int threadNum, int len, bool isDynamic) {
  return blockDimsLeft > 0 && ((threadNum > 1 && len > 1) || (threadNum > 0 && isDynamic));
}

static bool shouldMapToBlock(int gridDimsLeft, int blockNum, int len, bool isDynamic) {
  return gridDimsLeft > 0 && ((blockNum > 1 && len > 1) || (blockNum > 0 && isDynamic));
}

static void mapAxisToThread(AxisTileContext &ctx, int &threadNum, akgglobal::GpuScheduleTool &gpuTool) {
  if (ctx.isDynamic) {
    processDynamicThreadUse(ctx, gpuTool);
  } else if (ctx.len <= threadNum) {
    processFullThreadUse(ctx.axis, ctx.threadTile, ctx.len, threadNum, ctx.axisMap);
  } else {
    processPartThreadUse(ctx.axis, ctx.threadTile, ctx.len, threadNum, ctx.axisMap);
  }
}

static void mapAxisToBlock(AxisTileContext &ctx, int &blockNum) {
  if (ctx.isDynamic) {
    processDynamicBlockUse(ctx.axis, ctx.primes, ctx.axisMap);
  } else if (ctx.len <= blockNum) {
    processFullBlockUse(ctx.axis, ctx.len, blockNum, ctx.axisMap);
  } else {
    processPartBlockUse(ctx.axis, ctx.len, blockNum, ctx.axisMap);
  }
}

static void setAxisSequential(const AxisTileContext &ctx, ConfigPtr seqTile, akgglobal::GpuScheduleTool &gpuTool) {
  if (!ctx.isDynamic) {
    seqTile->value = ctx.len * ctx.threadTile->value;
  } else {
    processLeftDynamic(seqTile, ctx.threadTile, ctx.handleRedAxis, ctx.primes, gpuTool);
  }
}

void GpuTemplateTilingSolver::SolveAxesWithBlockSeqThreadPattern(std::vector<AxisPtr> &axes,
                                                                 const std::vector<int> &processOrder,
                                                                 const std::vector<int> &flags,
                                                                 const std::vector<int> &dynFlags, int blockNum,
                                                                 int threadNum, int &gridDimsLeft, int &blockDimsLeft,
                                                                 const bool &handleRedAxis) {
  // since we do reorder thread/seq later for coalescing access, inner-tile maps to thread and outer-tile maps to
  // sequential
  int num = axes.size();
  auto &tool = akgglobal::PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (flags[i] == 0) {
      continue;
    }

    // 0. Init
    auto a = axes[i];
    a->doExtraTile();
    std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};
    int len = a->range.second;
    auto threadTile = a->tryGetConfig(1);
    threadTile->value = 1;
    std::vector<int> primes;
    bool isDynamic = dynFlags[i] != 0;
    if (isDynamic) {
      initPrimes(primes, tool);
    }

    AxisTileContext ctx{a, threadTile, axisMap, len, primes, isDynamic, i, num, handleRedAxis};

    if (shouldMapToThread(blockDimsLeft, threadNum, ctx.len, ctx.isDynamic)) {
      mapAxisToThread(ctx, threadNum, gpuTool);
      blockDimsLeft--;
    }

    if (shouldMapToBlock(gridDimsLeft, blockNum, ctx.len, ctx.isDynamic)) {
      mapAxisToBlock(ctx, blockNum);
      gridDimsLeft--;
    }

    auto seqTile = a->tryGetConfig(0);
    setAxisSequential(ctx, seqTile, gpuTool);
    a->setMappings(ctx.axisMap);
  }
}

static void processYDynamicBlock(AxisPtr &a, akgglobal::PrimeNumTool &tool, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  auto dynTile = tool.getOnePrimeWithIdxUpdate();
  gridcfg->value = dynTile;  // a placeholder prime for blockidx
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
}

static void processYFullBlock(AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  gridcfg->value = len;
  gridcfg->index = ConfigPos::kOuter;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
  if (len != 0) {
    blockNum = blockNum / len;
  } else {
    blockNum = 0;
  }
  len = 1;
}

static void processFakeMapThread(AxisPtr &a, std::vector<std::string> &axisMap) {
  auto seqOuterTile = a->tryGetConfig(1);
  seqOuterTile->value = 1;
  auto blockcfg = std::make_shared<GpuBlock>("Manual");
  blockcfg->value = 1;
  blockcfg->index = ConfigPos::kInner;
  a->configs[blockcfg->type].push_back(blockcfg);
  axisMap[kNum2] = kGpuBlockCfg;
  a->setMappings(axisMap);
}

void GpuTemplateTilingSolver::SolveRedAxesWithReductionY(std::vector<AxisPtr> &axes,
                                                         const std::vector<int> &processOrder,
                                                         const std::vector<int> &redFlags,
                                                         const std::vector<int> &dynFlags, int blockNum,
                                                         int &gridDimsLeft) {
  int num = axes.size();
  auto &tool = akgglobal::PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (redFlags[i] == 0) {
      continue;
    }

    // 0. Init
    auto a = axes[i];
    a->doExtraTile();
    std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};  // default
    int len = a->range.second;

    // 1. tile & map to blockIdx
    if (gridDimsLeft > 0 && blockNum >= 1) {
      if (dynFlags[i] != 0) {
        processYDynamicBlock(a, tool, axisMap);
      } else if (len <= blockNum) {
        processYFullBlock(a, len, blockNum, axisMap);
      } else {
        processYFullBlock(a, len, blockNum, axisMap);
      }
      gridDimsLeft--;
    }

    // 2. left length to sequential
    auto seqInnerTile = a->tryGetConfig(0);
    if (dynFlags[i] != 0) {
      seqInnerTile->value = tool.getOnePrimeWithIdxUpdate();
      auto arg0 = gpuTool.addRuntimeArgument(seqInnerTile->value);
      arg0.mark = "reduce-y-seq";
      gpuTool.updateRuntimeArgument(arg0);
    } else {
      seqInnerTile->value = len;
    }

    // 3. map a fake-thread to bypass RewriteReduceInMultiLevelMemory limit
    processFakeMapThread(a, axisMap);
  }
}

void GpuTemplateTilingSolver::collectReduceAxesFlags(const std::vector<AxisPtr> &axes, std::vector<int> &redFlags,
                                                     std::vector<int> &dynamicFlags, bool &hasLastUnknownRedAxis) {
  for (size_t i = 0; i < axes.size(); i++) {
    redFlags[i] = static_cast<int>(axes[i]->axisType.find(mlir::autotiling::Axis::AxisLabel::kReduction) !=
                                   axes[i]->axisType.end());
    dynamicFlags[i] =
      static_cast<int>(axes[i]->axisType.find(mlir::autotiling::Axis::AxisLabel::kDynamic) != axes[i]->axisType.end());
    if ((redFlags[i] != 0) && (i == axes.size() - 1) && (dynamicFlags[i] != 0)) {
      hasLastUnknownRedAxis = true;
    }
  }
}

void GpuTemplateTilingSolver::collectReduceAxesOrders(const std::vector<AxisPtr> &axes,
                                                      const std::vector<int> &dynamicFlags,
                                                      std::vector<int> &processOrder, std::vector<int> &templateOrder) {
  for (size_t i = 0; i < axes.size(); i++) {
    if (dynamicFlags[i] == 0) {
      templateOrder.push_back(i);
    } else {
      processOrder.push_back(i);
    }
  }
}

void GpuTemplateTilingSolver::collectReduceAxesSize(const std::vector<AxisPtr> &axes, const std::vector<int> &redFlags,
                                                    const std::vector<int> &dynamicFlags, int &reductionSize,
                                                    int &parallelSize, bool &hasDynamicRedAxes) {
  for (size_t i = 0; i < axes.size(); i++) {
    if (redFlags[i] != 0) {
      reductionSize *= axes[i]->range.second;
      hasDynamicRedAxes |= dynamicFlags[i];
    } else {
      parallelSize *= axes[i]->range.second;
    }
  }
}

void GpuTemplateTilingSolver::SolveScheduleForReductionOps(std::vector<AxisPtr> &axes, bool &enableParallelReduction,
                                                           bool &enableAtomicReduction, bool &useReorder) {
  int num = axes.size();
  int gridDimsLeft = 3;
  int blockDimsLeft = 3;
  int reductionSize = 1;
  int parallelSize = 1;
  std::vector<int> redFlags;
  std::vector<int> dynamicFlags;
  redFlags.resize(num);
  dynamicFlags.resize(num);
  bool hasDynamicRedAxes = false;
  bool hasLastUnknownRedAxis = false;
  // use `processOrder` to sort axes process order by template to dynamic, so that we can give more resource to template
  // axes.
  std::vector<int> processOrder;
  std::vector<int> templateOrder;
  // collect static/dynamic shapes information from axes

  collectReduceAxesFlags(axes, redFlags, dynamicFlags, hasLastUnknownRedAxis);
  collectReduceAxesOrders(axes, dynamicFlags, processOrder, templateOrder);
  collectReduceAxesSize(axes, redFlags, dynamicFlags, reductionSize, parallelSize, hasDynamicRedAxes);

  processOrder.insert(processOrder.end(), templateOrder.begin(), templateOrder.end());
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  gpuTool.reduceSizeStatic = reductionSize;
  gpuTool.parallelSizeStatic = parallelSize;
  bool isReduceY =
    akgglobal::GpuScheduleTool::getInstance().getReduceDirection() == static_cast<size_t>(ReduceDirection::Y);

  // reuse to map non-reduce-axes
  std::vector<int> parallelFlags(redFlags.size());
  for (auto i = 0; i < static_cast<int>(redFlags.size()); i++) {
    parallelFlags[i] = 1 - redFlags[i];
  }
  if (!hasDynamicRedAxes && reductionSize <= kMinReductionLengthForParallel) {
    enableParallelReduction = false;
  }
  auto properThreadNum = 0;
  properThreadNum = getProperParallelAxesThread(parallelSize, isReduceY, blockDimsLeft, enableParallelReduction);
  SolveAxesWithBlockSeqThreadPattern(axes, processOrder, parallelFlags, dynamicFlags, INT_MAX, properThreadNum,
                                     gridDimsLeft, blockDimsLeft, false);

  // Dynamic&Static mixed GPU Reduction algorithms dispatch.
  if ((!hasDynamicRedAxes && reductionSize <= kMinReductionLengthForParallel) ||
      ((!enableAtomicReduction && isReduceY) || !enableParallelReduction)) {
    gpuTool.dynAlgorithm = "reduce-small";
    useReorder = false;
    enableParallelReduction = false;
    enableAtomicReduction = false;
    SolveRedAxesWithoutThreadReduction(axes, processOrder, redFlags, dynamicFlags);
  } else if (isReduceY) {
    gpuTool.dynAlgorithm = "reduce-y";
    useReorder = true;
    enableParallelReduction = false;
    int redBlockNum;
    int redSeqNum;
    std::tie(redBlockNum, redSeqNum) = getProperRedConfigsY(reductionSize, enableAtomicReduction);
    SolveRedAxesWithReductionY(axes, processOrder, redFlags, dynamicFlags, redBlockNum, gridDimsLeft);
  } else {
    // reduce-X/All with thread parallel reduction
    gpuTool.dynAlgorithm = "reduce-x";
    useReorder = true;
    enableParallelReduction = true;
    int redBlockNum;
    int redThreadNum;
    if ((!hasDynamicRedAxes && reductionSize <= kMinReductionLengthForAtomic) || !enableAtomicReduction) {
      enableAtomicReduction = false;
      std::tie(redBlockNum, redThreadNum) = getProperRedConfigsX(reductionSize, false);
      if (hasLastUnknownRedAxis) {
        redThreadNum = std::max(1, redThreadNum - kNum32);  // reserve 32 thread for last axis
      }
    } else {
      enableAtomicReduction = true;
      std::tie(redBlockNum, redThreadNum) = getProperRedConfigsX(reductionSize, true);
      if (hasLastUnknownRedAxis) {
        redThreadNum = std::max(1, redThreadNum - kNum32);  // reserve 32 thread for last axis
      }
    }
    gpuTool.enableAtomic = enableAtomicReduction;
    SolveAxesWithBlockSeqThreadPattern(axes, processOrder, redFlags, dynamicFlags, redBlockNum, redThreadNum,
                                       gridDimsLeft, blockDimsLeft, true);
    blockDimsLeft = 0;
  }
}
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir
