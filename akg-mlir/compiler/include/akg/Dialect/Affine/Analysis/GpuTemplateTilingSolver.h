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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_GPUTEMPLATETILINGSOLVER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_GPUTEMPLATETILINGSOLVER_H_
#include <deque>
#include <tuple>
#include <vector>
#include "akg/Dialect/Affine/Analysis/Axis.h"
#include "akg/Dialect/Affine/Analysis/Config.h"
#include "akg/Dialect/Affine/Analysis/Model.h"
#include "akg/Utils/AnalysisForGpu.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace akg {
namespace autotiling {

static constexpr int kMinReductionLengthForParallel = 64;
static constexpr int kMinReductionLengthForAtomic = 1024;
static constexpr int kDefaultNonReductionThreadNumX = 32;
static constexpr int kDefaultNonReductionThreadNumY = 32;
static constexpr int kProperAccSeqNumX = 4;
static constexpr int kProperAccSeqNumY = 16;
constexpr int kNum2 = 2;
constexpr int kNum16 = 16;
constexpr int kNum32 = 32;
constexpr int kNum64 = 64;
constexpr int kNum256 = 256;
constexpr int kNum512 = 512;
constexpr int kNum1024 = 1024;
constexpr int kMaxThreadsNum = 1024;

// GpuTemplateSolver is used to solve gpu backend fused-reduction ops tiling params, supports
// both static/dynamic shape cases. Inside this class, we provide diverse algorithms to ensure
// the performance in runtime. when some of shapes are unknown in compile time, we pass corresponding
// marks for runtime decisions. And those unknown axes are set with prime numbers to pass by mlir passes
// limitations.
class GpuTemplateSolver {
 public:
  GpuTemplateSolver() = default;

  // Compute static shape reduction-X proper tiling values
  static std::tuple<int, int> getProperRedConfigsX(int reductionSize, bool useAtmoicReturn);
  // Compute static shape reduction-Y proper tiling values
  static std::tuple<int, int> getProperRedConfigsY(int reductionSize, bool useAtmoicReturn);
  // Compute static shape parallel loops proper tiling value
  static int getProperParallelAxesThread(const int &len, const bool &isReduceY, const int &blockDimsLeft,
                                         const bool &enableParallelReduction);
  // Solve axes in cases of no thread-level parallel
  static void SolveRedAxesWithoutThreadReduction(std::vector<AxisPtr> &axes, const std::vector<int> &processOrder,
                                                 const std::vector<int> &redFlags, const std::vector<int> &dynFlags);
  // Solve axes in cases of reduction-Y algorithm
  static void SolveRedAxesWithReductionY(std::vector<AxisPtr> &axes, const std::vector<int> &processOrder,
                                         const std::vector<int> &redFlags, const std::vector<int> &dynFlags,
                                         int blockNum, int &gridDimsLeft);
  // Solve axes in cases of block-sequential-thread mapping algorithm
  static void SolveAxesWithBlockSeqThreadPattern(std::vector<AxisPtr> &axes, const std::vector<int> &processOrder,
                                                 const std::vector<int> &flags, const std::vector<int> &dynFlags,
                                                 int blockNum, int threadNum, int &gridDimsLeft, int &blockDimsLeft,
                                                 const bool &handleRedAxis);
  // Solve axes left to sequential
  static void SolveAxesLeft(std::vector<AxisPtr> &axes);
  // collect axes infos
  static void collectReduceAxesFlags(const std::vector<AxisPtr> &axes, std::vector<int> &redFlags,
                                     std::vector<int> &dynamicFlags, bool &hasLastUnknownRedAxis);

  static void collectReduceAxesOrders(const std::vector<AxisPtr> &axes, const std::vector<int> &dynamicFlags,
                                      std::vector<int> &processOrder, std::vector<int> &templateOrder);

  static void collectReduceAxesSize(const std::vector<AxisPtr> &axes, const std::vector<int> &redFlags,
                                    const std::vector<int> &dynamicFlags, int &reductionSize, int &parallelSize,
                                    bool &hasDynamicRedAxes);

  // Main enter of gpu template solver
  static void SolveScheduleForReductionOps(std::vector<AxisPtr> &axes, bool &enableParallelReduction,
                                           bool &enableAtomicReduction, bool &applyReorderPass);
};
}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_GPUTEMPLATETILINGSOLVER_H_
