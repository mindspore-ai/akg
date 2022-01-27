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
#ifndef POLY_TILING_STRATEGY_MANAGER_H_
#define POLY_TILING_STRATEGY_MANAGER_H_

#include <iostream>
#include <algorithm>
#include <deque>

#include "poly/tiling/tiling_analyzer.h"
#include "tiling_analyzer.h"

namespace akg {
namespace ir {
namespace poly {
class TilingStrategy {
 public:
  explicit TilingStrategy(const TilingAnalyzer *a) : analyzer_(a), target_(a->scop_info_.user_config_.GetTarget()) {}
  virtual ~TilingStrategy() { analyzer_ = nullptr; }
  virtual void AddNpuConstraint(){};
  virtual void AddGpuConstraint(){};
  virtual void AddCpuConstraint(){};

  std::string interested_attr_key;

 protected:
  const TilingAnalyzer *analyzer_;
  std::string target_;
  std::unordered_map<TileAxis *, std::vector<AttrInfo>> GetInterestedInfo(const std::string &attr_key,
                                                                          bool match_whole_word = true) const {
    std::unordered_map<TileAxis *, std::vector<AttrInfo>> result;
    std::vector<TileAxis *> axes =
      match_whole_word ? analyzer_->GetAxesOfAttr(attr_key) : analyzer_->GetAxesContainsAttr(attr_key);
    for (auto a : axes) {
      std::vector<AttrInfo> info;
      for (const auto &attr : a->attrs) {
        if ((match_whole_word && attr.attr_key != attr_key) ||
            (!match_whole_word && attr.attr_key.find(attr_key) == std::string::npos)) {
          continue;
        }
        info.emplace_back(attr);
      }
      result[a] = info;
    }
    return result;
  }

  // gpu configs
  int64_t warp_sizes_ = 32;
  int64_t total_available_thread_ = 1024;
  int64_t num_sm_ = 80;
  int64_t max_x_dim_block_ = 2147483647;
  int64_t max_y_z_dim_block_ = 65535;
  int64_t max_x_y_dim_thread_ = 1024;
  int64_t max_z_dim_thread_ = 64;
  size_t max_dim_ = 3;
  int64_t max_elem_per_thread_ = 1024;
  size_t tranpose_tiling_constraints_ = 32;
  int64_t reduce_length_limit_ = 32;

  // cpu config
  int64_t thread_num_ = 8;
  int64_t vector_size_ = 4;

  const static int binary_factor_{2};
  const static int decimal_factor_{10};
  const static int double_warp_size_{64};
  const static int quadruple_warp_size_{128};
};

class TilingStrategyManager {
 public:
  TilingStrategyManager() {}
  ~TilingStrategyManager() {}

  void SetStrategies(std::vector<TilingStrategy *> strategies) {
    this->strategies_.assign(strategies.begin(), strategies.end());
  }

  void ExecuteNpu() {
    for (auto strategy : this->strategies_) {
      strategy->AddNpuConstraint();
    }
  }

  void ExecuteGpu() {
    for (auto strategy : this->strategies_) {
      strategy->AddGpuConstraint();
    }
  }
  void ExecuteCpu() {
    for (auto strategy : this->strategies_) {
      strategy->AddCpuConstraint();
    }
  }

 private:
  std::vector<TilingStrategy *> strategies_;
};

class GpuDmaAnalysisStrategy : public TilingStrategy {
 public:
  explicit GpuDmaAnalysisStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class CustomTilingStrategy : public TilingStrategy {
 public:
  explicit CustomTilingStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = "CUSTOM"; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
  void ApplyCustomConstraints(TileAxis *axis, std::string con, TileLevel lv);
};

class ConflictTreeRangeStrategy : public TilingStrategy {
 public:
  explicit ConflictTreeRangeStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class ModStrategy : public TilingStrategy {
 public:
  explicit ModStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_MOD; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

// These strategies aim to deal with special insn in Npu core.
class CastStrategy : public TilingStrategy {
 public:
  explicit CastStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_CAST; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
  void AddCpuConstraint() override;
  void MarkDataSize() {
    auto interested_info = GetInterestedInfo(interested_attr_key);
    for (auto it : interested_info) {
      TileAxis *axis = it.first;
      for (const auto &attr : it.second) {
        std::vector<std::string> src_dst = akg::common::Split(attr.attr_value, "->");
        CHECK_EQ(src_dst.size(), 2U);

        std::vector<std::string> src_list = akg::common::Split(src_dst[0], ",");
        CHECK_GE(src_list.size(), 1U);
        for (const auto &src : src_list) {
          std::vector<std::string> src_info = akg::common::Split(src, ":");
          CHECK_EQ(src_info.size(), 2U);
          CHECK_NE(src_info[1], "");
          axis->data_size[src_info[0]].emplace_back(
            static_cast<int>(std::strtol(src_info[1].c_str(), nullptr, decimal_factor_)));
        }

        std::vector<std::string> dst_info = akg::common::Split(src_dst[1], ":");
        CHECK_EQ(dst_info.size(), 2U);
        CHECK_NE(dst_info[1], "");
        axis->data_size[dst_info[0]].emplace_back(
          static_cast<int>(std::strtol(dst_info[1].c_str(), nullptr, decimal_factor_)));
      }
    }
  }
};

class ReduceStrategy : public TilingStrategy {
 public:
  explicit ReduceStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
  void UpdateReduceThreads(bool square_thread, int64_t min_blocks, bool use_local);
  void UpdateThreadRange(bool square_thread);
  void UpdateAxes(int possible_blocks, int default_elem_per_thread);
  void AnalyzeReduceConfig(ReduceDirection direction, int band_index);
  void DealWith4DFusedReduce(const std::vector<akg::ir::poly::TileAxis *> &reduce_axes);

  void DisableReduceMapping();

  // Used by setting scop_info.enable_akg_reduce_lib.
  void AkgReduceLibStrategyOnGpu(int band_index);

  const bool UseRegisterMem();
  bool IsHalfReduce();

  // For this special case, we have tiling constraint on axis to calculate correct isl_footprint_box.
  const void DealWith4DFusedReduce();

  // For post reduce case, we should identify and disable atomic add for reduce axes.
  void DealWithPostReduceTensors();

  std::vector<TileAxis *> reduce_axes_;
  std::vector<TileAxis *> injective_axes_;
  bool all_reduce_{false};
  bool has_transpose_{false};
  int64_t reduce_length_;
  int64_t nonreduce_length_;
  int64_t min_ty_;
  int64_t min_ty_init_{8};
  int64_t max_coef_;
  int64_t reduce_threads_;
  int64_t injective_threads_;
  std::pair<int64_t, int64_t> tx_range_;
  std::pair<int64_t, int64_t> ty_range_;
  int64_t total_injective_size_;
  int64_t total_reduce_size_;
  int64_t possible_injective_blocks_;
  int64_t possible_reduce_blocks_;
};

class VectorizedStrategy : public TilingStrategy {
 public:
  explicit VectorizedStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class DmaAlignStrategy : public TilingStrategy {
 public:
  explicit DmaAlignStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_ALIGN; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class TensorOfTensorStrategy : public TilingStrategy {
 public:
  explicit TensorOfTensorStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class PassDownAttrStrategy : public TilingStrategy {
 public:
  explicit PassDownAttrStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class DynamicShapeLimitStrategy : public TilingStrategy {
 public:
  explicit DynamicShapeLimitStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {
    interested_attr_key = "DYN_SHAPE_LIMIT";
  }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class ShiftAxisStrategy : public TilingStrategy {
 public:
  explicit ShiftAxisStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_SHIFT; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;

  void TileEntirely() {
    auto interested_info = GetInterestedInfo(interested_attr_key);
    for (auto it : interested_info) {
      TileAxis *axis = it.first;
      int64_t const_extent = axis->GetConstExtent();
      if (const_extent == -1) {
        continue;
      }
      shifted_axes_.insert(axis);
      for (const auto &attr : it.second) {
        CHECK_NE(attr.attr_value, "");
        auto share_time = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, decimal_factor_));
        axis->TileRestrainToSingleValue(const_extent * (share_time + 1), CACHE1);
        break;
      }
    }
  }

  std::unordered_set<TileAxis *> shifted_axes_;
};  // namespace poly

class ModShiftAxisStrategy : public TilingStrategy {
 public:
  explicit ModShiftAxisStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_MODSHIFT; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class DynamicBoundStrategy : public TilingStrategy {
 public:
  explicit DynamicBoundStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_DYNAMIC_BOUND; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
};

class ConvStrategy : public TilingStrategy {
 public:
  explicit ConvStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_CONV; }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;

  std::unordered_map<std::string, Expr> conv_info_{};
  air::arith::Analyzer arith_ana_;

  void RestrainH(TileAxis *axis);
  void RestrainW(TileAxis *axis);

  // gpu tensor core strategy steps
  std::unique_ptr<MmaConv> InitGemmShape(const Mma &mma);
  std::pair<int64_t, int64_t> CalculateNumOfWarps(const Mma &mma);
  void CalculateMacroMma(const MmaConv &shape, const Mma &mma);
  void SetFinalConfig(const MmaConv &macro_mma, const Mma &mma);

  // Return a combination of total factor that can be divisible by shape_m and shape_n.
  const std::pair<int64_t, int64_t> GetDivisibleFactorForMN(
    int64_t shape_m, int64_t shape_n, int64_t total_factor, const Mma &mma);

  int w0_for_m_{1};
  int w1_for_n_{1};
  TileAxis *m_axis_{nullptr};
  TileAxis *h_axis_{nullptr};
  TileAxis *w_axis_{nullptr};
  TileAxis *n_axis_{nullptr};
  TileAxis *k_axis_{nullptr};
  int sm_bytes_{1};
  int reg_bytes_{1};
  int64_t min_blocks_{400};
  int64_t real_blocks_{0};
  int64_t default_num_warps_{1};
  MmaConv macro_mma_{128, 1, 1, 128, 32};
  int num_warps_low_{2};
  int num_warps_mid_{4};
  int num_warps_high_{8};
  int binary_factor_{2};
  int tensor_core_per_warp_{4};
  int use_local_group_high_{8};
};

class GemmStrategy : public TilingStrategy {
 public:
  explicit GemmStrategy(const TilingAnalyzer *a) : TilingStrategy(a) { interested_attr_key = AT_GEMM; }
  ~GemmStrategy() final {
    m_axis_ = nullptr;
    n_axis_ = nullptr;
    k_axis_ = nullptr;
  }
  void AddNpuConstraint() override;
  void AddGpuConstraint() override;

  // gpu tensor core strategy steps
  std::unique_ptr<Mma> InitGemmShape(const Mma &mma);
  std::pair<int64_t, int64_t> CalculateNumOfWarps(const Mma &mma);
  void CalculateMacroMma(const Mma &shape, const Mma &mma);
  void SetFinalConfig(const Mma &macro_mma, const Mma &mma);

  // common utils
  int EstimateSharedSize(const Mma &alloc, int dtype);
  int EstimateRegisterSize(const Mma &alloc, int dtype);
  // Return a combination of total factor that can be divisible by shape_m and shape_n.
  const std::pair<int64_t, int64_t> GetDivisibleFactorForMN(
    int64_t shape_m, int64_t shape_n, int64_t total_factor, const Mma &mma);

  int w0_for_m_{1};
  int w1_for_n_{1};
  TileAxis *m_axis_{nullptr};
  TileAxis *n_axis_{nullptr};
  TileAxis *k_axis_{nullptr};
  int sm_bytes_{1};
  int reg_bytes_{1};
  int64_t min_blocks_{2048};
  int64_t default_num_warps_{1};
  int64_t tile_stride_{32};
  Mma macro_mma_{128, 128, 32};
  int sm_bytes_div_factor_{3};
  int sm_bytes_mul_factor_{4};
  int use_local_group_high_{8};
  int default_num_warps_high_{4};
  int default_num_warps_low_{2};
  int int_bit_count_{32};
};

class GpuStrategy : public TilingStrategy {
 public:
  explicit GpuStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}

  void AddNpuConstraint() override;
  void AddGpuConstraint() override;
  std::vector<TileAxis::MappingConstraint> thread_binding_spaces_;  // [thread.x, thread.y, thread.z]
  std::vector<TileAxis::MappingConstraint> block_binding_spaces_;   // [block.x, block.y, block.z]

 private:
  const void ShowOptions();

  void AdjustThreadMappingLimit();

  const void TransposeSpeedup();

  void PadSpeedup();

  void InjectiveSpeedup();

  void VectorizationSpeedup();
  void CheckVectorizationForElemwiseOp();
  bool IsVectorized();

  void BroadcastSpeedup();
  std::unordered_set<int> broadcast_idx_;

  void AnalyzeBroadcastIdx();
  void GpuScalarBroadcastStrategy();
  void GpuVectorBroadcastStrategy();

  // Step 0. Init mapping limit according to operation type.
  void InitMappingLimit();

  // Step 1. Collect axes and sort them from inner to outer
  void BuildAxesQueue();

  void ApplyCustomConstraint();

  /*
   * Step 2. Tile inner axes first and map them to threads, and then tile outer axis and map the rest of them to blocks.
   * e.g.
   *   input: add op with shape [2, 32, 256, 32, 32]
   *   tile size: [1, 1, 1, 32, 32]
   *   band after tile:  [2, 32, 256, 1, 1] -> child [1, 1, 1, 32, 32]
   *   mapping: [2(b0), 32(b1), 4(b2), 1, 1] -> child [1, 1, 1, 32(t1), 32(t0)]
   */
  void InnerThreadOuterBlock(bool write_cfg);

  const int64_t GetThreadSize(const int64_t rest_threads, size_t inner_dim, const int64_t shape, const int64_t item);
  int64_t TileAfterThreadMapping(TileAxis *axis, size_t inner_dim, int64_t thread_size, const int64_t item);
  void ThreadConfiguration(ReduceDirection direct, bool use_lib);
  void SkipMapping(TileAxis *axis, int64_t shape, std::stringstream &ss, size_t inner_dim, size_t thread_dim);
  void GreedyMapBlocks(size_t ori_size, size_t block_dim);
  void MapPendingAxes(size_t ori_size, std::stringstream &ss, size_t thread_dim, bool write_cfg);
  int64_t ApplyCustomTile(TileAxis *axis, size_t inner_dim, int64_t thread_size, int64_t tile, int64_t shape);
  void WriteConfigBackInjective();
  std::pair<int64_t, int64_t> GetProposalParallelSize(int problem_size);
  int64_t AlignThreadToShape();
  void MapBroadcastElem(TileAxis *axis, std::vector<int> original_shape);
  void ApplyConstraintsToBindingSpace();

  int GetLocalAllocBufCount();
  bool NeedModifyOrderOfAxis();
  void InitMapping();
  Template template_{Template::DEFAULT};
  bool is_reduce_op_[TEMPLATE_BULK] = {false, false, true, true, true, false};

  std::deque<std::pair<TileAxis *, int64_t>> pending_axes_;
  std::vector<int64_t> block_limit_;
  std::vector<int64_t> thread_limit_;
  std::vector<int64_t> block_cfg_;
  std::vector<int64_t> thread_cfg_;
  std::unordered_map<TileAxis *, int64_t> thread_cfg_map_;
  std::unordered_map<TileAxis *, int64_t> block_cfg_map_;
  int block_count_{0};  // number of mapped blocks
  int64_t elem_per_thread_[3]{static_cast<int64_t>(SpItemPerThread::AUTO)};
  int64_t min_elem_for_io_bound_ = 2;
  size_t depth_{0};
  bool need_reverse_{false};
  bool reverse_binding_{false};
  int64_t fused_size_{1};
  std::unordered_map<int, std::string> mapping_idx_pos_ = {{0, "x"}, {1, "y"}, {2, "z"}};
  std::unordered_map<int, std::string> reduce_y_idx_pos_ = {{0, "y"}, {1, "x"}};
  int vectorized_bytes_{1};
  int band_index_{0};
  OuterBandNode *current_outer_bn_{nullptr};
  int64_t activated_threads_;
  std::pair<int64_t, int64_t> thread_coef_;
  std::pair<int64_t, int64_t> thread_coef_init_{8, 16};
  std::pair<int64_t, int64_t> active_blocks_per_sm_{5, 6};
  std::vector<size_t> indexing_;
  std::vector<TileAxis *> injective_axes_;
  int possible_threads_;
  int coalesced_size_;
  int total_injective_size_;
  int64_t total_vectorized_bytes_ = 16; // The default total number of bytes for vectorization is 16.
};

class CpuStrategy : public TilingStrategy {
 public:
  explicit CpuStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddCpuConstraint() override;

 private:
  void BuildAxesQueue();
  void RecordTileValue();
  void SetMatMulTileValue(int index);
  bool SetReduceYTileValue(int index);
  void SetMultiLevelTileValue();
  void SetUnrollTileValue(TileAxis *axis, const int64_t axis_size, int64_t &tile_left);
  void SetParallelTileValue(TileAxis *axis, const int64_t axis_size, const int64_t data_size,
                            bool is_unroll_axis = false, int64_t tile_left = 1);

  std::vector<std::vector<std::pair<TileAxis *, int64_t>>> pending_axes_;
  int min_exec_num_per_thread_{MIN_EXEC_NUM_PER_THREAD};
  int best_parallel_num_{BEST_PARALLEL_NUM};
  int parallel_decrease_value_{PARALLEL_DECREASE_VALUE};
  int best_unroll_num_{BEST_UNROLL_NUM};
  int min_unroll_num_{MIN_UNROLL_NUM};
  int best_factor_for_matmul_{MATMUL_BEST_FACTOR};
  int axis_m_{MATMUL_AXIS_M};
  int axis_n_{MATMUL_AXIS_N};
  int axis_k_{MATMUL_AXIS_K};
  std::unordered_map<int, std::string> axes_name_ = {{0, "gemm_m"}, {1, "gemm_n"}, {2, "gemm_k"}};
};

class CsrStrategy : public TilingStrategy {
 public:
  explicit CsrStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddGpuConstraint() override;

  int warp_factor_reduction_{2};
  int warp_factor_elemwise_{5};
};

class CountStrategy : public TilingStrategy {
 public:
  explicit CountStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  void AddGpuConstraint() override;
};

class MulticoreStrategy {
 public:
  MulticoreStrategy(TileCandidate &cand, TileLogger &logger) : cand_(cand), logger_(logger) {}
  ~MulticoreStrategy() {}
  int64_t AdjustTilingAccordingToMulticoreConstraint(TileAxis *multicore_axis, int64_t tiling_factor);

 private:
  TileCandidate &cand_;
  TileLogger &logger_;
  std::pair<int, int> GetProposalRangeForFullMulticore(TileAxis *axis);
  int GetProposalCoreNum();
};

class TilingPriorityScorer {
 public:
  explicit TilingPriorityScorer(TilingAnalyzer &analyzer) : analyzer_(analyzer), logger_(analyzer.GetTileLogger()) {}
  ~TilingPriorityScorer() {}

  /*
   * Compute a total score of priority for each tile axis considering all related features and corresponding weights.
   * Tile axis with higher score will have higher tiling priority (i.e. have more memory space).
   * Note that score of each feature is standardlised into range [1, tile_axis_size].
   */
  void SetPriorityByScoring();

  void SetParallelismWeight(const int parallelism) { weight_.parallelism = parallelism; }
  void SetVectorizationWeight(const int vectorization) { weight_.vectorization = vectorization; }
  void SetDataReuseWeight(const int tile_dependency) { weight_.tile_dependency = tile_dependency; }

 private:
  TilingAnalyzer &analyzer_;
  TileLogger &logger_;

  /*
   * Weight parameters for each feature in priority score model.
   * Initial weights are set empirically and changing they can support micro-tuning.
   */
  struct Weight {
    int parallelism{1};  // get lowest weight because coincident may not always trustable
    int tile_dependency{2};
    int vectorization{3};
    int Sum() const { return parallelism + vectorization + tile_dependency; }
  } weight_;

  /*
   * Parallelism is computed by checking coincident value in schedule tree for corresponding axis.
   * If an axis can be parallelised, the parallelism score is 0; otherwise it is 1.
   */
  std::vector<double> ComputeParallelism(std::vector<TileAxis *> tile_axes);

  /*
   * Tile dependency describes the relationship between tile axes: if more tile axes are dependended on one tile axis,
   * this tile axis will have higher tile dependency score and gets higher priority during tiling.
   * For example, reduce axis is usually depended by other axes and thus it should be put into local buffer first.
   */
  std::vector<double> ComputeTileDependency(std::vector<TileAxis *> tile_axes);

  /*
   * Vectorization is computed by accumulating the dimension index of corresponding axis on each buffer.
   * If an axis is related with more innermost dimensions of different buffers, the vectorization score is higher.
   */
  std::vector<double> ComputeVectorization(std::vector<TileAxis *> tile_axes);

  /*
   * Normalize data to range [1, range_max].
   * `range_max` is usually set to the size of tile axes that need to determine priority.
   */
  std::vector<double> MinMaxScaler(std::vector<double> data, int range_max = 1) {
    auto min = *min_element(data.begin(), data.end());
    auto max = *max_element(data.begin(), data.end());
    std::stringstream ss;
    ss << "Min: " << min << ", Max: " << max;
    logger_.AppendLog(DO_TILING, ss);
    std::vector<double> scaled_data(data.size(), 1);
    if (max - min == 0) {
      return scaled_data;
    }
    for (size_t i = 0; i < data.size(); ++i) {
      auto old_d = data[i];
      ss << "Orginal data: " << old_d;
      auto new_d = (old_d - min) / (max - min);
      new_d = range_max > 1 ? (new_d * (range_max - 1) + 1) : new_d;
      ss << " -> Scaled data: " << new_d;
      scaled_data[i] = new_d;
      logger_.AppendLog(DO_TILING, ss);
    }
    return scaled_data;
  }

  std::vector<TileAxis *> GetBandTileAxes(int band_idx) const {
    std::vector<TileAxis *> tile_axes;
    auto Collect = [&tile_axes, band_idx](TileAxis *axis) {
      if (axis->index == band_idx) {
        tile_axes.emplace_back(axis);
      }
    };
    analyzer_.ForEachAxisTopDown(Collect);
    return tile_axes;
  }
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_STRATEGY_MANAGER_H_
