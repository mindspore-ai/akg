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

#ifndef COMPILER_INCLUDE_AKG_ANALYSIS_TILINGSTRATEGY_H_
#define COMPILER_INCLUDE_AKG_ANALYSIS_TILINGSTRATEGY_H_
#include <deque>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "akg/Analysis/Axis.h"
#include "akg/Analysis/Config.h"
#include "akg/Analysis/Model.h"
#include "akg/Dialect/Affine/Analysis/GpuTemplateTilingSolver.h"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace autotiling {
using llvm::SmallVector;

constexpr auto BEST_UNROLL_NUM = 256;
constexpr auto MIN_UNROLL_NUM = 8;
constexpr auto MIN_EXEC_NUM_PER_THREAD = 4096;
constexpr auto BEST_PARALLEL_NUM = 192;
constexpr auto PARALLEL_DECREASE_VALUE = 1;

enum Sketch {
  kAllStatic = 0,
  kOneDynamicInner,
  kTwoDynamicInner,
  kMoreDynamicInner,
  kLargeStaticInner,
  kSmallStaticInner
};

class TilingStrategy {
 public:
  TilingStrategy() {}
  virtual ~TilingStrategy() = default;

  explicit TilingStrategy(const std::unordered_set<std::string> &work_for_ops) : workForOps(work_for_ops) {}
  explicit TilingStrategy(mlir::autotiling::Axis::AxisLabel axisLabel) : workForAxisLabel(axisLabel) {}
  virtual void AddConstraint(ModelGraphPtr initGraph) {}
  virtual void AddNpuConstraint(NpuModelGraphPtr initGraph) {}
  virtual void AddGpuConstraint(GpuModelGraphPtr initGraph) {}
  virtual void AddCpuConstraint(CpuModelGraphPtr initGraph) {}
  std::unordered_set<std::string> workForOps;
  mlir::autotiling::Axis::AxisLabel workForAxisLabel{mlir::autotiling::Axis::AxisLabel::kDefault};
  bool IsRelevant(const AxisPtr &a, const InitGraphPtr graph);
  bool IsRelevant(const AxisPtr &a, const InitGraphPtr graph, std::unordered_set<std::string> workForOps);
};
using TilingStrategyPtr = std::shared_ptr<TilingStrategy>;

class RepositoryStrategy : public TilingStrategy {
 public:
  RepositoryStrategy() : TilingStrategy() {}
  virtual ~RepositoryStrategy() = default;

  void AddConstraint(ModelGraphPtr initGraph) override;

 private:
  int extraTileLevel = 2;
};

class DynamicShapeStrategy : public TilingStrategy {
 public:
  DynamicShapeStrategy() : TilingStrategy() {}
  virtual ~DynamicShapeStrategy() = default;

  void AddGpuConstraint(GpuModelGraphPtr initGraph) override;
  void AddCpuConstraint(CpuModelGraphPtr initGraph) override;

 private:
  Sketch SketchAnalysis(std::vector<int64_t> sketch);
  void DoConstTile(GpuModelGraphPtr initGraph, Sketch sketch);
  void DoVariableTile(GpuModelGraphPtr initGraph, Sketch sketch);
  int64_t largeShapeLimit = 1024;
  int microTileSize = 32;
};

class ReduceStrategy : public TilingStrategy {
 public:
  ReduceStrategy() : TilingStrategy({"Reduce"}) {}
  virtual ~ReduceStrategy() = default;

  void AddGpuConstraint(GpuModelGraphPtr gpuGraph) override;
};

class TransposeStrategy : public TilingStrategy {
 public:
  TransposeStrategy() : TilingStrategy() {}
  virtual ~TransposeStrategy() = default;

  void AddGpuConstraint(GpuModelGraphPtr gpuGraph) override;
  int maxExpectSeq = 4;
  int maxExpectSeqPerAxis = 2;
  size_t minRankForTranspose = 2;
};

class BroadcastStrategy : public TilingStrategy {
 public:
  BroadcastStrategy() : TilingStrategy({mlir::autotiling::Axis::AxisLabel::kVectorization}) {}
  virtual ~BroadcastStrategy() = default;

  void AddGpuConstraint(GpuModelGraphPtr gpuGraph) override;
  void AddCpuConstraint(CpuModelGraphPtr cpuGraph) override;

 private:
  int64_t getMaxByte(const CpuModelGraphPtr cpuGraph, NodePtr &minRankNode);
  int getMaxBroadcastAxes(const CpuModelGraphPtr cpuGraph, std::vector<AxisPtr> maxloopNest);
  bool searchForSmallShape(const GpuModelGraphPtr gpuGraph, const AxisPtr a);
  bool searchForLargeShape(const GpuModelGraphPtr gpuGraph, const AxisPtr a);
  NodePtr findMinRankNode(const GpuModelGraphPtr gpuGraph);
  int computeExpectedSeq(const GpuModelGraphPtr gpuGraph, const AxisPtr innerMostReadAxis);
  void searchSeqAxisFromInnerToOuter(const GpuModelGraphPtr gpuGraph, const NodePtr maxRankNode, int expectSeq);
  int proposedGrid = 1;
  int proposedBlock = 1;
  int minExpectSeq = 2;
  int maxExpectSeq = 4;
  int blockWasteCoef = 8;
  int gridWasteCoef = 2;
};

class UnrollStrategy : public TilingStrategy {
 public:
  UnrollStrategy() : TilingStrategy() {}
  virtual ~UnrollStrategy() = default;

  void AddCpuConstraint(CpuModelGraphPtr cpuGraph) override;
};

class ParallelStrategy : public TilingStrategy {
 public:
  ParallelStrategy() : TilingStrategy() {}
  virtual ~ParallelStrategy() = default;

  void AddGpuConstraint(GpuModelGraphPtr gpuGraph) override;
  void AddCpuConstraint(CpuModelGraphPtr cpuGraph) override;
  void AddNpuConstraint(NpuModelGraphPtr npuGraph) override;

 private:
  void InitProposalResource(const GpuModelGraphPtr gpuGraph);
  bool tryMapBlock(const GpuModelGraphPtr gpuGraph, const AxisPtr axis);
  bool tryMapGrid(const GpuModelGraphPtr gpuGraph, const AxisPtr axis);

  // Helper functions for AddNpuConstraint
  void collectAxesInfo(const SmallVector<AxisPtr> &axes, int pos);
  std::pair<int64_t, int64_t> allocateCoresForAxes(int64_t totalCores);
  void applyParallelTiling(const SmallVector<AxisPtr> &axes, int64_t coresForParallel, int64_t coresForReduce,
                           int64_t coreNum, int pos);
  // GPU
  bool currHasMinMax{false};
  int proposedGrid = 1;
  int proposedBlock = 1;
  int gridWasteCoef = 1;
  int blockWasteCoef = 8;
  int blockLimitCoef = 2;

  // NPU
  SmallVector<bool> isParallelAxis;
  int64_t totalParallelSize{0};
  int64_t totalReduceSize{0};
};

class VectorizationStrategy : public TilingStrategy {
 public:
  VectorizationStrategy() : TilingStrategy() {}
  virtual ~VectorizationStrategy() = default;

  void AddNpuConstraint(NpuModelGraphPtr npuGraph) override;

 private:
  SmallVector<int64_t> getDimSizes(const SmallVector<AxisPtr> &axes);
  int64_t computeVectorizationTilingKey(int64_t ubAvailableNum, const SmallVector<int64_t> &dims);
  void applyVectorizationTiling(const SmallVector<AxisPtr> &axes, int64_t ubAvailableNum, int64_t tilingKey, int pos);

  // UB size constants
  static constexpr int64_t kUBAlignSizeInBytes = 32;  // 32-byte alignment
  static constexpr int64_t kNumBitsInByte = 8;
};

class NpuDefaultTileStrategy : public TilingStrategy {
 public:
  NpuDefaultTileStrategy() : TilingStrategy() {}
  ~NpuDefaultTileStrategy() override = default;

  void AddNpuConstraint(NpuModelGraphPtr npuGraph) override;

 private:
  llvm::SmallVector<AxisPtr> collectAxes(const NpuModelGraphPtr npuGraph);
  std::unordered_map<size_t, unsigned> buildBandRankMap(const llvm::SmallVector<AxisPtr> &axes);
  llvm::SmallVector<unsigned, 4> parseTileSizesConfig(const NpuModelGraphPtr npuGraph);
  void applyTilingToAxes(const NpuModelGraphPtr npuGraph, const llvm::SmallVector<AxisPtr> &axes,
                         const std::unordered_map<size_t, unsigned> &bandRankMap,
                         const llvm::SmallVector<unsigned, 4> &tileSizes, bool isReduceOp);
};

class TilingStrategyManager {
 public:
  TilingStrategyManager() {}
  ~TilingStrategyManager() {}

  void addStrategy(const TilingStrategyPtr strategy) { this->strategies_.push_back(strategy); }

  void SetStrategies(const std::vector<TilingStrategyPtr> &strategies) {
    this->strategies_.assign(strategies.begin(), strategies.end());
  }

  void processOn(const ModelGraphPtr modelGraph);

  void processOn(const GpuModelGraphPtr gpuGraph);

  void processOn(const CpuModelGraphPtr cpuGraph);

  void processOn(const NpuModelGraphPtr npuGraph);

 private:
  std::vector<TilingStrategyPtr> strategies_;
};
using TilingStrategyManagerPtr = std::shared_ptr<TilingStrategyManager>;

}  // namespace autotiling
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_ANALYSIS_TILINGSTRATEGY_H_
