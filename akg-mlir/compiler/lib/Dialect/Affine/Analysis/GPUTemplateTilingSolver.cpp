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

#include "akg/Dialect/Affine/Analysis/GpuTemplateTilingSolver.h"
#include "akg/Utils/AKGGlobalVars.hpp"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <vector>

using namespace akgglobal;

namespace mlir {
namespace akg {
namespace autotiling {
using namespace mlir::akg::utils;

// Template reduction-X or reduction-All strategy
std::tuple<int, int> GpuTemplateSolver::getProperRedConfigsX(int reductionSize, bool useAtmoicReturn) {
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
int GpuTemplateSolver::getProperParallelAxesThread(const int &len, const bool &isReduceY, const int &blockDimsLeft,
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
std::tuple<int, int> GpuTemplateSolver::getProperRedConfigsY(int reductionSize, bool useAtmoicReturn) {
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

void GpuTemplateSolver::SolveRedAxesWithoutThreadReduction(std::vector<AxisPtr> &axes,
                                                           const std::vector<int> &processOrder,
                                                           const std::vector<int> &redFlags,
                                                           const std::vector<int> &dynFlags) {
  auto &tool = PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  int num = axes.size();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (redFlags[i]) {
      auto a = axes[i];
      auto fullTile = dynFlags[i] ? tool.getOnePrimeWithIdxUpdate() : a->range.second;
      if (dynFlags[i]) {
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

void GpuTemplateSolver::SolveAxesLeft(std::vector<AxisPtr> &axes) {
  int num = axes.size();
  for (int i = 0; i < num; i++) {
    if (axes[i]->mappings.size() == 0) {
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

static void processDynamicThreadUse(autotiling::AxisPtr &a, autotiling::ConfigPtr &threadTile, const int &i,
                                    const int &num, const bool &handleRedAxis, const std::vector<int> &primes,
                                    std::vector<std::string> &axisMap, GpuScheduleTool &gpuTool) {
  auto blockcfg = std::make_shared<GpuBlock>("DynBlock");
  auto dynTile = primes[0];
  threadTile->value = dynTile;
  blockcfg->value = dynTile;
  a->configs[blockcfg->type].push_back(blockcfg);
  axisMap[kNum2] = kGpuBlockCfg;
  auto argBlock = gpuTool.addRuntimeArgument(dynTile);
  argBlock.mark = (i == num - 1) ? "thread-last" : "thread";
  argBlock.mark = (handleRedAxis ? "reduce-" : "parallel-") + argBlock.mark;
  gpuTool.updateRuntimeArgument(argBlock);
}

static void processFullThreadUse(autotiling::AxisPtr &a, autotiling::ConfigPtr &threadTile, int &len, int &threadNum,
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

static void processPartThreadUse(autotiling::AxisPtr &a, autotiling::ConfigPtr &threadTile, int &len, int &threadNum,
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

static void processDynamicBlockUse(autotiling::AxisPtr &a, const std::vector<int> &primes,
                                   std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("DynGrid");
  auto dynTile = primes[1];  // a placeholder prime for blockidx
  gridcfg->value = dynTile;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
}

static void processFullBlockUse(autotiling::AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
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

static void processPartBlockUse(autotiling::AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  gridcfg->value = blockNum;
  gridcfg->index = ConfigPos::kOuter;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
  len = (len - 1) / blockNum + 1;
  blockNum = 0;
}

static void processLeftDynamic(autotiling::ConfigPtr &seqTile, autotiling::ConfigPtr &threadTile,
                               const bool &handleRedAxis, const std::vector<int> &primes, GpuScheduleTool &gpuTool) {
  auto prime = primes[2];
  seqTile->value = prime * threadTile->value;
  auto arg0 = gpuTool.addRuntimeArgument(prime);
  arg0.mark = handleRedAxis ? "reduce-x-seq" : "parallel-seq";
  gpuTool.updateRuntimeArgument(arg0);
  auto arg1 = gpuTool.addRuntimeArgument(seqTile->value);
  arg1.mark = handleRedAxis ? "product" : "1";
  gpuTool.updateRuntimeArgument(arg1);
}

void GpuTemplateSolver::SolveAxesWithBlockSeqThreadPattern(std::vector<AxisPtr> &axes,
                                                           const std::vector<int> &processOrder,
                                                           const std::vector<int> &flags,
                                                           const std::vector<int> &dynFlags, int blockNum,
                                                           int threadNum, int &gridDimsLeft, int &blockDimsLeft,
                                                           const bool &handleRedAxis) {
  // since we do reorder thread/seq later for coalescing access, inner-tile maps to thread and outer-tile maps to
  // sequential
  int num = axes.size();
  auto &tool = PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (!flags[i]) {
      continue;
    }

    // 0. Init
    auto a = axes[i];
    a->doExtraTile();
    std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};  // default
    int len = a->range.second;
    auto threadTile = a->tryGetConfig(1);
    threadTile->value = 1;
    std::vector<int> primes;
    if (dynFlags[i]) {
      initPrimes(primes, tool);
    }

    // 1. tile & map to threadIdx
    if (blockDimsLeft > 0 && ((threadNum > 1 && len > 1) || (threadNum > 0 && dynFlags[i]))) {
      if (dynFlags[i]) {
        processDynamicThreadUse(a, threadTile, i, num, handleRedAxis, primes, axisMap, gpuTool);
      } else if (len <= threadNum) {
        processFullThreadUse(a, threadTile, len, threadNum, axisMap);
      } else {
        processPartThreadUse(a, threadTile, len, threadNum, axisMap);
      }
      blockDimsLeft--;
    }

    // 2. tile & map to blockIdx
    if (gridDimsLeft > 0 && ((blockNum > 1 && len > 1) || (blockNum > 0 && dynFlags[i]))) {
      if (dynFlags[i]) {
        processDynamicBlockUse(a, primes, axisMap);
      } else if (len <= blockNum) {
        processFullBlockUse(a, len, blockNum, axisMap);
      } else {
        processPartBlockUse(a, len, blockNum, axisMap);
      }
      gridDimsLeft--;
    }

    // 3. left length to sequential
    auto seqTile = a->tryGetConfig(0);
    if (!dynFlags[i]) {
      seqTile->value = len * threadTile->value;
    } else {
      processLeftDynamic(seqTile, threadTile, handleRedAxis, primes, gpuTool);
    }
    a->setMappings(axisMap);
  }
}

static void processYDynamicBlock(autotiling::AxisPtr &a, PrimeNumTool &tool, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  auto dynTile = tool.getOnePrimeWithIdxUpdate();
  gridcfg->value = dynTile;  // a placeholder prime for blockidx
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
}

static void processYFullBlock(autotiling::AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
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

static void processYPartBlock(autotiling::AxisPtr &a, int &len, int &blockNum, std::vector<std::string> &axisMap) {
  auto gridcfg = std::make_shared<GpuGrid>("Manual");
  gridcfg->value = blockNum;
  gridcfg->index = ConfigPos::kOuter;
  a->configs[gridcfg->type].push_back(gridcfg);
  axisMap[0] = kGpuGridCfg;
  len = (len - 1) / blockNum + 1;
  blockNum = 0;
}

static void processFakeMapThread(autotiling::AxisPtr &a, std::vector<std::string> &axisMap) {
  auto seqOuterTile = a->tryGetConfig(1);
  seqOuterTile->value = 1;
  auto blockcfg = std::make_shared<GpuBlock>("Manual");
  blockcfg->value = 1;
  blockcfg->index = ConfigPos::kInner;
  a->configs[blockcfg->type].push_back(blockcfg);
  axisMap[kNum2] = kGpuBlockCfg;
  a->setMappings(axisMap);
}

void GpuTemplateSolver::SolveRedAxesWithReductionY(std::vector<AxisPtr> &axes, const std::vector<int> &processOrder,
                                                   const std::vector<int> &redFlags, const std::vector<int> &dynFlags,
                                                   int blockNum, int &gridDimsLeft) {
  int num = axes.size();
  auto &tool = PrimeNumTool::getInstance();
  auto &gpuTool = akgglobal::GpuScheduleTool::getInstance();
  for (int idx = num - 1; idx >= 0; idx--) {
    auto i = processOrder[idx];
    if (!redFlags[i]) {
      continue;
    }

    // 0. Init
    auto a = axes[i];
    a->doExtraTile();
    std::vector<std::string> axisMap = {kGpuSeqCfg, kGpuSeqCfg, kGpuSeqCfg};  // default
    int len = a->range.second;

    // 1. tile & map to blockIdx
    if (gridDimsLeft > 0 && blockNum >= 1) {
      if (dynFlags[i]) {
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
    if (dynFlags[i]) {
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

void GpuTemplateSolver::collectReduceAxesFlags(const std::vector<AxisPtr> &axes, std::vector<int> &redFlags,
                                               std::vector<int> &dynamicFlags, bool &hasLastUnknownRedAxis) {
  for (size_t i = 0; i < axes.size(); i++) {
    redFlags[i] = axes[i]->axisType.find(Axis::AxisLabel::kReduction) != axes[i]->axisType.end();
    dynamicFlags[i] = axes[i]->axisType.find(Axis::AxisLabel::kDynamic) != axes[i]->axisType.end();
    if (redFlags[i] && (i == axes.size() - 1) && dynamicFlags[i]) {
      hasLastUnknownRedAxis = true;
    }
  }
}

void GpuTemplateSolver::collectReduceAxesOrders(const std::vector<AxisPtr> &axes, const std::vector<int> &dynamicFlags,
                                                std::vector<int> &processOrder, std::vector<int> &templateOrder) {
  for (size_t i = 0; i < axes.size(); i++) {
    if (!dynamicFlags[i]) {
      templateOrder.push_back(i);
    } else {
      processOrder.push_back(i);
    }
  }
}

void GpuTemplateSolver::collectReduceAxesSize(const std::vector<AxisPtr> &axes, const std::vector<int> &redFlags,
                                              const std::vector<int> &dynamicFlags, int &reductionSize,
                                              int &parallelSize, bool &hasDynamicRedAxes) {
  for (size_t i = 0; i < axes.size(); i++) {
    if (redFlags[i]) {
      reductionSize *= axes[i]->range.second;
      hasDynamicRedAxes |= dynamicFlags[i];
    } else {
      parallelSize *= axes[i]->range.second;
    }
  }
}

void GpuTemplateSolver::SolveScheduleForReductionOps(std::vector<AxisPtr> &axes, bool &enableParallelReduction,
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
  bool isReduceY = GpuScheduleTool::getInstance().getReduceDirection() == ReduceDirection::Y;

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
    int redBlockNum, redSeqNum;
    std::tie(redBlockNum, redSeqNum) = getProperRedConfigsY(reductionSize, enableAtomicReduction);
    SolveRedAxesWithReductionY(axes, processOrder, redFlags, dynamicFlags, redBlockNum, gridDimsLeft);
  } else {
    // reduce-X/All with thread parallel reduction
    gpuTool.dynAlgorithm = "reduce-x";
    useReorder = true;
    enableParallelReduction = true;
    int redBlockNum, redThreadNum;
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
