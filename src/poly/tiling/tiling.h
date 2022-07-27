/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_H_
#define POLY_TILING_H_

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "poly/scop_info.h"
#include "poly/poly_util.h"
#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/node.h"
#include "poly/tiling/hermes/tensor.h"
#include "poly/tiling/tiling_analyzer.h"
#include "poly/tiling/tiling_algorithm.h"
#include "poly/tiling/tiling_strategy_manager.h"
#include "poly/tiling/tiling_solver.h"

namespace akg {
namespace ir {
namespace poly {

class TilingGenerator {
 public:
  explicit TilingGenerator(TilingAnalyzer &analyzer) : analyzer_(analyzer) {}
  ~TilingGenerator() = default;
  using DynamicMemInfo = TileCandidate::DynamicMemInfo;

  struct TileInfo {
    TileInfo(TileAxis *a, TileLevel l, int b) : axis(a), level(l), band(b) {}
    TileAxis *axis;
    TileLevel level;
    int band;
    int64_t min_tile = 0;
    int64_t deviation = 0;
  };

  struct ParamReplacement {
    std::vector<int64_t> mul_tile;
    std::vector<int64_t> mod_tile;
    std::vector<int64_t> l0_tile;
  };

  TileSizes Generate();

  TileSizes GenerateQuickly();

  std::pair<TileSizes, std::deque<ParamInfo>> GenerateDynamic();

  TileSizes HermesTiling(TileSizes dims);

  void ExtractAxisInfoFromScheduler(const isl::schedule &sch);

  Array<Expr> memory_constraints_;

 private:
  TileSizes ConvertToDims();

  bool IsConflictPrime(const int64_t prime, const ParamReplacement &prev);

  ParamReplacement CreateVarTileReplaceMap();

  int64_t CalL1VarTiling(int64_t l0_tiling, TileAxis *axis);

  static DimensionInfo ConvertDefaultInfo(TileAxis *axis);

  void ConvertVarTilesToDims();

  void ConvertShiftBoundToDims();

  void ConvertPragmaToDims(Map<Var, Expr> var_to_prime_record);

  bool IsSymbolicTiling();

  void CollectAxis();

  static void GetTransformedOutputShapeDatatype(const std::vector<std::shared_ptr<Tensor>> &output_tensors,
                                                std::vector<Tensor> &transformed_output_shape);

  static std::map<std::string, std::map<Tensor, int>> GetAxisToTensorToShapeIdMap(const std::shared_ptr<Node> &node);

  static std::vector<Axis> GetAxisDimFromGlobal(std::vector<Axis> &axis_of_node);

  TilingAnalyzer &analyzer_;
  TileCandidate *cand_{nullptr};
  int64_t mem_limit_[MEM_SCOPE_BULK]{0};
  std::deque<ParamInfo> param_info_;
  ParamReplacement param_replacement_;
  std::vector<int64_t> prev_tiling_;
  TileSizes dims_;
  bool tile_success_{true};

  const int kBit32 = 32;
};

TileSizes NullTiling();

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_H_
