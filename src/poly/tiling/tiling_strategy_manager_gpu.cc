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
#include "./tiling_strategy_manager.h"

#include <algorithm>
#include <numeric>
#include <sstream>
#include <build_module.h>
#include "../../src/include/build_module.h"
#include "./tiling_analyzer.h"
#include "poly/schedule_pass_gpu/register_memory_manager.h"

namespace akg {
namespace ir {
namespace poly {
template <typename T>
T SafeDivisor(T x) {
  CHECK(x != 0);
  return std::max<T>(x, 1);
}

bool TryCombination(int64_t shape_m, int64_t shape_n, const Mma &mma, int64_t factor1, int64_t factor2) {
  return (factor1 != 0 && factor2 != 0 && shape_m % factor1 == 0 && shape_n % factor2 == 0 &&
          shape_m / factor1 >= mma.m && shape_n / factor2 >= mma.n);
}

void GpuDmaAnalysisStrategy::AddGpuConstraint() {
  analyzer_->ForEachAxisTopDown(
    [](TileAxis *axis) { axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), TileLevel::CACHE1); });
}

void CastStrategy::AddGpuConstraint() { MarkDataSize(); }
void CastStrategy::AddCpuConstraint() { MarkDataSize(); }

void GemmStrategy::AddGpuConstraint() {
  if (!analyzer_->scop_info_.user_config_.GetEnableTensorCore() ||
      analyzer_->scop_info_.analysis_result_.GetIsGpuDmaAnalysed() ||
      analyzer_->scop_info_.user_config_.GetEnableConvTensorCore()) {
    return;
  }

  Mma mma = analyzer_->scop_info_.analysis_result_.GetMmaMode();

  // Step 1. Collect Batch, M, N, K axis info.
  std::unique_ptr<Mma> shape = InitGemmShape(mma);
  if (shape == nullptr) {
    return;
  }

  if (!analyzer_->scop_info_.user_config_.EnableStitchFusion()) {
    analyzer_->scop_info_.user_config_.SetEnableOneDimThread(true);
  }
  Mma middle_band = {shape->m / SafeDivisor(mma.m), shape->n / SafeDivisor(mma.n), shape->k / SafeDivisor(mma.k)};
  std::stringstream ss;
  ss << "[Gemm] M = " << shape->m << " N = " << shape->n << " K = " << shape->k << ", middle band = [" << middle_band.m
     << ", " << middle_band.n << ", " << middle_band.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  GpuInfo &gpu_info = GpuInfo::GetInstance(analyzer_->scop_info_.user_config_.GetDeviceType());
  sm_bytes_ = static_cast<int>(gpu_info.GetMemoryLimitInScope(static_cast<int>(MEM_SCOPE_SHARED)));
  sm_bytes_ = (sm_bytes_ / sm_bytes_div_factor_) * sm_bytes_mul_factor_;
  reg_bytes_ = static_cast<int>(MAX_REGISTER_PER_THREAD_BLOCK * REGISTER_ALLOC_RATIO);

  auto b_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_GEMM, kDsabi});
  auto bo_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_GEMM, kDsabo});
  (void)std::copy(bo_axes.begin(), bo_axes.end(), std::back_inserter(b_axes));
  for (auto b_axis : b_axes) {
    CHECK(b_axis->range_extent.as<IntImm>()) << "Dynamic shape is not supported in tensor core for now.";
    b_axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE1);
    b_axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE0);
    b_axis->thread_constraints.map_min_ = MIN_TILE;
    b_axis->thread_constraints.map_extent_ = MIN_TILE;
    CHECK(b_axis->range_extent.as<IntImm>());
    min_blocks_ /= b_axis->range_extent.as<IntImm>()->value;
    ss << "Map batch axis " << b_axis->range_extent.as<IntImm>()->value << " to block.";
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
  min_blocks_ = std::max<int>(1, min_blocks_);

  // Step 2. Calculate macro M, N, K tile size.
  CalculateMacroMma(*shape, mma);

  // Step 3. Calculate possible number of warps.
  auto warp_sizes = CalculateNumOfWarps(mma);
  std::tie(w0_for_m_, w1_for_n_) = warp_sizes;
  middle_band.m /= w0_for_m_;
  middle_band.n /= w1_for_n_;
  std::string warp_cfg = std::to_string(w0_for_m_) + " " + std::to_string(w1_for_n_);
  analyzer_->scop_info_.user_config_.RecordReplaceConfig(WARP_COMPUTE, warp_cfg, MappingType::REPLACE_THREADS);

  // Step 4. Set mapping and tiling config.
  SetFinalConfig(macro_mma_, mma);
}

std::pair<int64_t, int64_t> GemmStrategy::CalculateNumOfWarps(const Mma &mma) {
  int w0 = 1;
  int w1 = 1;
  int use_local_group = (macro_mma_.m / SafeDivisor(mma.m)) * (macro_mma_.n / SafeDivisor(mma.n));
  CHECK_GE(use_local_group, 1);
  if (use_local_group > use_local_group_high_) {
    default_num_warps_ = default_num_warps_high_;
  } else if (use_local_group > 1) {
    default_num_warps_ = default_num_warps_low_;
  }
  std::tie(w0, w1) = GetDivisibleFactorForMN(macro_mma_.m, macro_mma_.n, default_num_warps_, mma);
  std::stringstream ss;
  ss << "[Gemm] Try warp " << default_num_warps_ << " -> " << w0 << " * " << w1;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  return std::make_pair(w0, w1);
}

std::unique_ptr<Mma> GemmStrategy::InitGemmShape(const Mma &mma) {
  auto m_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_GEMM, kDsami});
  auto n_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_GEMM, kDsani});
  auto k_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_GEMM, kDsaki});
  if (m_axes.size() != 1U || n_axes.size() != 1U || k_axes.size() != 1U) {
    return nullptr;
  }

  m_axis_ = m_axes[0];
  n_axis_ = n_axes[0];
  k_axis_ = k_axes[0];
  if (m_axis_->range_extent.as<IntImm>() == nullptr || n_axis_->range_extent.as<IntImm>() == nullptr ||
      k_axis_->range_extent.as<IntImm>() == nullptr) {
    return nullptr;
  }
  auto shape_m = m_axis_->range_extent.as<IntImm>()->value;
  auto shape_n = n_axis_->range_extent.as<IntImm>()->value;
  auto shape_k = k_axis_->range_extent.as<IntImm>()->value;
  CHECK_EQ(shape_m % mma.m, 0) << "Shape m " << shape_m << " should be multiples of mma.m " << mma.m
                               << " to enable tensor core.";
  CHECK_EQ(shape_n % mma.n, 0) << "Shape n " << shape_n << " should be multiples of mma.n " << mma.n
                               << " to enable tensor core.";
  CHECK_EQ(shape_k % mma.k, 0) << "Shape k " << shape_k << " should be multiples of mma.k " << mma.k
                               << " to enable tensor core.";

  return std::make_unique<Mma>(Mma{shape_m, shape_n, shape_k});
}

int GemmStrategy::EstimateSharedSize(const Mma &alloc, int dtype) {
  std::string a_major = ROW_MAJOR;
  std::string b_major = ROW_MAJOR;
  auto major_map = analyzer_->scop_info_.analysis_result_.GetMatrixMatmulMajor();
  auto matmul_map = analyzer_->scop_info_.analysis_result_.GetMatrixMatmulMap();
  for (auto i : matmul_map) {
    if (i.second == MATRIX_A) {
      CHECK(major_map.find(i.first) != major_map.end());
      a_major = major_map[i.first];
    } else if (i.second == MATRIX_B) {
      CHECK(major_map.find(i.first) != major_map.end());
      b_major = major_map[i.first];
    }
  }

  // bank conflit avoid strategy
  auto matrix_a_size = a_major == ROW_MAJOR ? (alloc.m * (alloc.k + 16)) : ((alloc.m + 16) * alloc.k);
  auto matrix_b_size = b_major == COL_MAJOR ? (alloc.n * (alloc.k + 16)) : ((alloc.n + 16) * alloc.k);
  auto matrix_c_size = alloc.m * alloc.n;
  auto alloc_shared = (matrix_a_size + matrix_b_size) * dtype;  // single op does not alloc shared for matrix_c
  std::stringstream ss;
  ss << "[Shared] A(" << a_major << "), B(" << b_major << "); This config results matrix_a_size = " << matrix_a_size
     << " matrix_b_size = " << matrix_b_size << " matrix_c_size = " << matrix_c_size
     << " --> alloc shared = " << alloc_shared;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  return static_cast<int>(alloc_shared);
}

int GemmStrategy::EstimateRegisterSize(const Mma &alloc, int dtype) {
  auto alloc_reg_unit = std::max<int>(1, dtype / BYTES_PER_REGISTER);
  auto matrix_a_size = alloc.m * alloc.k * binary_factor_;
  auto matrix_b_size = alloc.n * alloc.k * binary_factor_;
  auto matrix_c_size = alloc.m * alloc.n;
  auto alloc_reg = (matrix_a_size + matrix_b_size + matrix_c_size) * alloc_reg_unit;
  std::stringstream ss;
  ss << "[Reg] This config results matrix_a_size = " << matrix_a_size << " matrix_b_size = " << matrix_b_size
     << " matrix_c_size = " << matrix_c_size << " --> alloc reg = " << alloc_reg;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  return static_cast<int>(alloc_reg);
}

void GemmStrategy::CalculateMacroMma(const Mma &shape, const Mma &mma) {
  std::stringstream ss;
  Mma default_macro_mma = macro_mma_;
  Mma macro_mma = {std::min<int>(macro_mma_.m, shape.m), std::min<int>(macro_mma_.n, shape.n),
                   std::min<int>(macro_mma_.k, shape.k)};
  ss << "[Init macro mma]: [" << macro_mma.m << ", " << macro_mma.n << ", " << macro_mma.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  while (shape.m % SafeDivisor(macro_mma_.m) != 0 && macro_mma_.m - tile_stride_ >= mma.m) {
    macro_mma_.m -= tile_stride_;
  }
  while (shape.n % SafeDivisor(macro_mma_.n) != 0 && macro_mma_.n - tile_stride_ >= mma.n) {
    macro_mma_.n -= tile_stride_;
  }
  if (shape.m % SafeDivisor(macro_mma_.m) != 0) {
    macro_mma_.m /= binary_factor_;
  }
  if (shape.n % SafeDivisor(macro_mma_.n) != 0) {
    macro_mma_.n /= binary_factor_;
  }
  while (shape.k % SafeDivisor(macro_mma_.k) != 0 && macro_mma_.k / binary_factor_ >= mma.k) {
    macro_mma_.k /= binary_factor_;
  }
  while ((shape.m / SafeDivisor(macro_mma_.m)) * (shape.n / SafeDivisor(macro_mma_.n)) < min_blocks_ &&
         macro_mma_.m == default_macro_mma.m && macro_mma_.n == default_macro_mma.n) {
    (shape.m < shape.n) ? (macro_mma_.m /= binary_factor_) : (macro_mma_.n /= binary_factor_);
  }
  if ((shape.m / SafeDivisor(macro_mma_.m)) * (shape.n / SafeDivisor(macro_mma_.n)) < min_blocks_ &&
      shape.k % SafeDivisor(macro_mma_.k * binary_factor_) == 0 &&
      shape.k / SafeDivisor(macro_mma_.k * binary_factor_) > 1) {
    macro_mma_.k *= binary_factor_;
  }
  if (shape.k == macro_mma_.k) {
    g_attrs.Set(kEnableTransferBuffer, air::make_const(Int(int_bit_count_), false));
  }
  ss << "[Final macro mma]: [" << macro_mma.m << ", " << macro_mma.n << ", " << macro_mma.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GemmStrategy::SetFinalConfig(const Mma &macro_mma, const Mma &mma) {
  std::stringstream ss;
  m_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.m)), CACHE1);
  m_axis_->thread_constraints.map_min_ = w0_for_m_ * w1_for_n_;
  m_axis_->thread_constraints.map_extent_ = w0_for_m_ * w1_for_n_;
  m_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.m)), CACHE0);

  n_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.n)), CACHE1);
  n_axis_->thread_constraints.map_min_ = warp_sizes_;
  n_axis_->thread_constraints.map_extent_ = warp_sizes_;
  n_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.n)), CACHE0);

  k_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.k)), CACHE1);
  k_axis_->thread_constraints.map_min_ = MIN_TILE;
  k_axis_->thread_constraints.map_extent_ = MIN_TILE;
  k_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.k)), CACHE0);
  ss << "[Final config] : L1(M, N, K) = " << macro_mma.m << ", " << macro_mma.n << ", " << macro_mma.k;
  ss << "; L0(M, N, K) = " << mma.m << ", " << mma.n << ", " << mma.k;
  ss << "; Thread(W0, W1, TX) = " << w0_for_m_ << ", " << w1_for_n_ << ", " << warp_sizes_;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

const std::pair<int64_t, int64_t> GetTensorCoreDivisibleFactorForMN(int64_t shape_m, int64_t shape_n,
                                                                    int64_t total_factor, int64_t binary_factor,
                                                                    const Mma &mma) {
  auto SwapWarp = [&shape_m, &shape_n, &mma](int64_t w0, int64_t w1) -> std::pair<int64_t, int64_t> {
    int64_t max_w0 = shape_m / SafeDivisor(mma.m);
    int64_t max_w1 = shape_n / SafeDivisor(mma.n);
    if (static_cast<size_t>(max_w0 - max_w1 > 0) ^ static_cast<size_t>(w0 - w1 > 0)) {
      return std::make_pair(w1, w0);
    }
    return std::make_pair(w0, w1);
  };
  int64_t w0 = static_cast<int64_t>(std::sqrt(total_factor));
  int64_t w1 = total_factor / SafeDivisor(w0);
  CHECK_EQ(w0 * w1, total_factor);
  std::tie(w0, w1) = SwapWarp(w0, w1);

  if (TryCombination(shape_m, shape_n, mma, w0, w1)) {
    return std::make_pair(w0, w1);
  } else {
    while (total_factor > 1) {
      CHECK(binary_factor != 0);
      total_factor /= binary_factor;
      w0 = std::sqrt(total_factor);
      w1 = total_factor / SafeDivisor(w0);
      CHECK_EQ(w0 * w1, total_factor);
      std::tie(w0, w1) = SwapWarp(w0, w1);
      if (TryCombination(shape_m, shape_n, mma, w0, w1)) {
        return std::make_pair(w0, w1);
      }
    }
  }
  return std::make_pair(1, 1);
}

const std::pair<int64_t, int64_t> GemmStrategy::GetDivisibleFactorForMN(int64_t shape_m, int64_t shape_n,
                                                                        int64_t total_factor, const Mma &mma) {
  return GetTensorCoreDivisibleFactorForMN(shape_m, shape_n, total_factor, binary_factor_, mma);
}

void ReduceStrategy::AnalyzeReduceConfig(ReduceDirection direction, int band_index) {
  if (!analyzer_->scop_info_.user_config_.EnableStitchFusion()) {
    if (reduce_length_ <= reduce_length_limit_) {
      analyzer_->scop_info_.user_config_.SetEnableOneDimThread(true);
      analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "ReduceLength <= 32, enable onedim thread.");
    }
    if ((direction == ReduceDirection::Y && reduce_length_ <= reduce_length_limit_) ||
        ((direction == ReduceDirection::X || direction == ReduceDirection::ALL) &&
         reduce_length_ < reduce_length_limit_)) {
      analyzer_->scop_info_.analysis_result_.SetUseGpuReduceLib(false);
      analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "Small Reduction (Y<=32, X<32), disable akg reduce lib.");
    }
    if (direction == ReduceDirection::X && reduce_length_ < binary_factor_ * reduce_length_limit_ &&
        nonreduce_length_ > max_y_z_dim_block_ * max_x_y_dim_thread_) {
      analyzer_->scop_info_.analysis_result_.SetUseGpuReduceLib(false);
      analyzer_->GetTileLogger().AppendLine(GPU_MAPPING,
                                            "Small Reduction (X<64) and large nonreduction axis (exceeding block and "
                                            "thread limit) , disable akg reduce lib.");
    }
  }
  if (!analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    DisableReduceMapping();
  } else {
    AkgReduceLibStrategyOnGpu(band_index);
  }
}

void ReduceStrategy::AddGpuConstraint() {
  for (int band_index = 0; band_index < static_cast<int>(analyzer_->RootAxis()->children.size()); ++band_index) {
    reduce_axes_ = analyzer_->GetAxesOfAttr(AT_REDUCE_AXIS, band_index);
    if (reduce_axes_.empty()) {
      continue;
    }
    if (analyzer_->scop_info_.user_config_.GetThreadConfig() != nullptr &&
        analyzer_->scop_info_.user_config_.GetBlockConfig() != nullptr &&
        analyzer_->scop_info_.user_config_.GetThreadConfig()->bound > 0 &&
        analyzer_->scop_info_.user_config_.GetBlockConfig()->bound > 0) {
      continue;
    }
    size_t depth = 0;
    auto HasTranspose = [this](const AttrInfo &info) {
      std::string key = info.attr_key;
      return key.find(AT_TRANSPOSE) != std::string::npos;
    };
    reduce_length_ = 1;
    nonreduce_length_ = 1;
    analyzer_->ForEachAxisTopDown([this, &depth, &band_index, &HasTranspose](TileAxis *axis) {
      if (axis->index != band_index && axis != analyzer_->RootAxis()) {
        return;
      }
      if (!has_transpose_) {
        has_transpose_ = std::any_of(axis->attrs.begin(), axis->attrs.end(), HasTranspose);
      }

      if (axis == analyzer_->RootAxis() || axis->is_inner) {
        return;
      }
      ++depth;
      if (axis->mc_sup) {
        (void)injective_axes_.emplace_back(axis);
        if (auto ext = axis->range_extent.as<IntImm>()) {
          nonreduce_length_ *= ext->value;
        }
        return;
      }
      if (auto ext = axis->range_extent.as<IntImm>()) {
        reduce_length_ *= ext->value;
      } else if (analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(axis->range_extent)) {
        int rest_thread = total_available_thread_ / analyzer_->scop_info_.analysis_result_.GetCsrFeatLen();
        reduce_length_ = std::min(analyzer_->scop_info_.user_config_.GetCsrThreadNum(), rest_thread);
      }
      if (std::count(reduce_axes_.begin(), reduce_axes_.end(), axis)) {
        return;
      }
      (void)reduce_axes_.emplace_back(axis);
    });
    all_reduce_ = reduce_axes_.size() == depth;
    auto current_outer_bn = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(band_index);
    auto direction = current_outer_bn->reduce_direction;
    AnalyzeReduceConfig(direction, band_index);
  }
}

void ReduceStrategy::DisableReduceMapping() {
  bool is_tuning = analyzer_->scop_info_.user_config_.GetIsTuning();
  for (auto axis : reduce_axes_) {
    axis->block_constraints.map_extent_ = MIN_TILE;
    axis->thread_constraints.map_extent_ = MIN_TILE;
    if (!is_tuning) {
      axis->TileRestrainEntire(TileLevel::CACHE1);
    }
  }
}

void ReduceStrategy::UpdateThreadRange(bool square_thread) {
  auto AlignToPowerOfTwo = [](int64_t original_factor) -> int64_t {
    while ((original_factor) & (original_factor - 1)) {
      --original_factor;
    }
    return original_factor;
  };
  if (square_thread) {
    tx_range_.first = AlignToPowerOfTwo(std::min(warp_sizes_, total_injective_size_));
    ty_range_.first = AlignToPowerOfTwo(std::min<int64_t>(min_ty_, total_reduce_size_));
    tx_range_.second = AlignToPowerOfTwo(
      std::min(tx_range_.second,
               static_cast<int64_t>(ceil(static_cast<float>(tx_range_.second) / SafeDivisor(ty_range_.first)))));
    tx_range_.second = AlignToPowerOfTwo(std::min(tx_range_.second, total_injective_size_));
  } else {
    if (analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
      tx_range_.first = AlignToPowerOfTwo(std::min(warp_sizes_, total_reduce_size_));
    } else {
      tx_range_.first = 1;
    }
    ty_range_.first = AlignToPowerOfTwo(std::min<int64_t>(min_ty_, total_injective_size_));
    tx_range_.second = AlignToPowerOfTwo(
      std::min(tx_range_.second,
               static_cast<int64_t>(ceil(static_cast<float>(tx_range_.second) / SafeDivisor(ty_range_.first)))));
    tx_range_.second = AlignToPowerOfTwo(std::min(tx_range_.second, total_reduce_size_));
  }
  ty_range_.second = std::min(
    ty_range_.second, static_cast<int64_t>(ceil(static_cast<float>(ty_range_.second) / SafeDivisor(tx_range_.first))));
}

void ReduceStrategy::ComputeProperReduceThreads(bool use_local) {
  for (auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_MOD) {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      auto mod_value = StrToDecimalInt64(attr.attr_value);
      axis->TileRestrainMod(CastInt64ToExpr(mod_value), TileLevel::CACHE1);
    }
    if (use_local) {
      auto ori_reduce_threads = reduce_threads_;
      CHECK(axis->c1_constraints.tile_mod_.as<IntImm>());
      auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
      while (tile_mod > reduce_threads_ && tile_mod % SafeDivisor(reduce_threads_) != 0) {
        --reduce_threads_;
      }
      if (ori_reduce_threads / SafeDivisor(reduce_threads_) > binary_factor_) {
        reduce_threads_ = ori_reduce_threads;
      }
    }
  }
}

void ReduceStrategy::UpdateReduceThreads(bool square_thread, int64_t min_blocks, bool use_local) {
  if (square_thread) {
    int coef = 1;
    while (coef < max_coef_) {
      if (total_reduce_size_ % SafeDivisor(ty_range_.first * coef) == 0 ||
          (coef < max_coef_ / binary_factor_ &&
           total_reduce_size_ % SafeDivisor(ty_range_.first * coef * binary_factor_) != 0)) {
        break;
      }
      coef *= binary_factor_;
    }
    reduce_threads_ = ty_range_.first * coef;
    injective_threads_ = tx_range_.second / SafeDivisor(coef);
  } else {
    int coef = 1;
    while (coef < max_coef_) {
      if (total_reduce_size_ % SafeDivisor(tx_range_.second / SafeDivisor(coef)) == 0 ||
          (coef < max_coef_ / binary_factor_ &&
           total_reduce_size_ % SafeDivisor((tx_range_.second / SafeDivisor(coef)) / binary_factor_) != 0)) {
        break;
      }
      coef *= binary_factor_;
    }
    if (analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
      reduce_threads_ = tx_range_.second / SafeDivisor(coef);
    }
    injective_threads_ = ty_range_.first * coef;
    if (total_reduce_size_ < warp_sizes_) {
      // we increase thread y for small reduction cases
      while ((coef < max_coef_) && (total_injective_size_ % SafeDivisor(injective_threads_) == 0) &&
             (total_injective_size_ / SafeDivisor(injective_threads_) > min_blocks)) {
        coef *= binary_factor_;
        injective_threads_ = ty_range_.first * coef;
      }
    }
  }

  ComputeProperReduceThreads(use_local);
}

void ReduceStrategy::UpdateAxes(int possible_blocks, int default_elem_per_thread) {
  if (analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    auto original_ept = default_elem_per_thread;
    // try to increase thread loop (no more than twice as original)
    while (possible_blocks > default_elem_per_thread && default_elem_per_thread != 0 &&
           possible_blocks % SafeDivisor(default_elem_per_thread) != 0) {
      ++default_elem_per_thread;
    }
    if (original_ept * binary_factor_ < default_elem_per_thread) {
      default_elem_per_thread = original_ept;
    }
    // try to decrease thread loop (no less than half of original)
    while (possible_blocks > default_elem_per_thread && default_elem_per_thread != 0 &&
           possible_blocks % SafeDivisor(default_elem_per_thread) != 0) {
      --default_elem_per_thread;
    }
    if (default_elem_per_thread * binary_factor_ < original_ept) {
      default_elem_per_thread = original_ept;
    }
  } else {
    default_elem_per_thread = 1;
  }

  std::stringstream ss;
  ss << "total_injective_size " << total_injective_size_ << " total_reduce_size " << total_reduce_size_;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "injective_threads " << injective_threads_ << " reduce_threads " << reduce_threads_;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "possible_blocks " << possible_blocks << " possible_injective_blocks " << possible_injective_blocks_
     << " possible_reduce_blocks " << possible_reduce_blocks_ << " default_elem_per_thread " << default_elem_per_thread;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "tx:[" << tx_range_.first << ", " << tx_range_.second << "]; ty:[" << ty_range_.first << ", "
     << ty_range_.second << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  auto inject_len = injective_axes_.size();
  for (size_t i = 0; i < inject_len; ++i) {
    auto axis_in = injective_axes_[i];
    if (i == inject_len - 1) {
      axis_in->thread_constraints.map_min_ = injective_threads_;
      axis_in->thread_constraints.map_extent_ = injective_threads_;
      axis_in->thread_constraints.item_process_ = MIN_TILE;
    } else {
      axis_in->thread_constraints.map_min_ = MIN_TILE;
      axis_in->thread_constraints.map_extent_ = MIN_TILE;
      axis_in->thread_constraints.item_process_ = MIN_TILE;
    }
  }
  for (auto axis : reduce_axes_) {
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_) {
      continue;
    }
    axis->thread_constraints.map_extent_ = reduce_threads_;
    axis->thread_constraints.item_process_ = default_elem_per_thread;
  }
}

void ReduceStrategy::CollectReduceAxesInfo() {
  for (auto axis : reduce_axes_) {
    total_reduce_size_ *= axis->extent_val;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_reduce_blocks_ *= axis->extent_val;
    } else {
      possible_reduce_blocks_ *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      reduce_threads_ *= axis->thread_constraints.map_min_;
    }
  }
}

void ReduceStrategy::CollectInjectiveAxesInfo() {
  for (auto axis : injective_axes_) {
    total_injective_size_ *= axis->extent_val;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_injective_blocks_ *= axis->extent_val;
    } else {
      possible_injective_blocks_ *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      injective_threads_ *= axis->thread_constraints.map_min_;
    }
  }
}

void ReduceStrategy::AkgReduceLibStrategyOnGpu(int band_index) {
  // disable atomic-add for bitwise-reduction
  auto current_outer_bn = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(band_index);
  bool disable_atomic = !analyzer_->scop_info_.user_config_.GetEnableAtomicAdd() ||
                        current_outer_bn->template_type == Template::BITWISE_REDUCTION;
  if (disable_atomic) {
    for (auto axis : reduce_axes_) {
      axis->block_constraints.map_extent_ = MIN_TILE;
    }
  }

  // disable atomic-add for post reduce tensors
  DealWithPostReduceTensors();

  if (has_transpose_) {
    for (auto axis : reduce_axes_) {
      axis->TileRestrainEntire(TileLevel::CACHE1);
      axis->block_constraints.map_extent_ = MIN_TILE;
    }
  }

  if (!all_reduce_) {
    DealWith4DFusedReduce();
  }

  total_reduce_size_ = 1;
  possible_reduce_blocks_ = 1;
  reduce_threads_ = 1;
  CollectReduceAxesInfo();

  total_injective_size_ = 1;
  possible_injective_blocks_ = 1;
  injective_threads_ = 1;
  CollectInjectiveAxesInfo();

  bool is_special_4d = reduce_threads_ != 1 || injective_threads_ != 1;
  if (is_special_4d) {
    return;
  }
  auto direction = current_outer_bn->reduce_direction;
  bool square_thread = direction == ReduceDirection::Y && analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib();
  bool use_local = UseRegisterMem();
  int64_t min_blocks = square_thread ? warp_sizes_ : 512;
  int64_t min_elem_per_thread = use_local ? binary_factor_ : 8;
  min_ty_ = min_ty_init_;
  if (((total_injective_size_ * total_reduce_size_) / min_blocks) / max_x_y_dim_thread_ < min_elem_per_thread) {
    min_blocks = warp_sizes_;
    min_ty_ = square_thread ? min_ty_ : 1;
  }

  tx_range_ = {1, max_x_y_dim_thread_};
  ty_range_ = {1, max_x_y_dim_thread_};
  UpdateThreadRange(square_thread);

  max_coef_ =
    std::max(ty_range_.second / SafeDivisor(ty_range_.first), tx_range_.second / SafeDivisor(tx_range_.first));
  UpdateReduceThreads(square_thread, min_blocks, use_local);

  int possible_blocks = static_cast<int>(
    ceil(static_cast<float>(((possible_injective_blocks_ * possible_reduce_blocks_) / SafeDivisor(injective_threads_)) /
                            SafeDivisor(reduce_threads_))));
  int proposal = use_local ? 8 : warp_sizes_;
  auto default_elem_per_thread =
    possible_reduce_blocks_ > 1
      ? std::max(
          std::min<int>(proposal, ((possible_blocks / SafeDivisor(min_blocks) + 1) / binary_factor_) * binary_factor_),
          1)
      : IsHalfReduce() ? double_warp_size_ : static_cast<int>(SpItemPerThread::FULL);
  UpdateAxes(possible_blocks, default_elem_per_thread);
}

const bool ReduceStrategy::UseRegisterMem() {
  for (auto &it : analyzer_->buf_info_) {
    auto buf = it.second.get();
    CHECK(buf != nullptr);
    if (buf->scope == TilingMemScope::MEM_SCOPE_LOCAL) {
      return true;
    }
  }
  return false;
}

bool ReduceStrategy::IsHalfReduce() {
  for (const auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_REDUCE_AXIS) {
        continue;
      }
      auto red_tensor_name = attr.attr_value;
      auto it = axis->data_size.find(red_tensor_name);
      if (it != axis->data_size.end() && *std::min_element(it->second.begin(), it->second.end()) == binary_factor_) {
        return true;
      }
    }
  }
  return false;
}

const void ReduceStrategy::DealWith4DFusedReduce() {
  auto mod_axes = analyzer_->GetAxesOfAttr(AT_MOD);
  for (auto axis : mod_axes) {
    if (axis->HasAttr(AT_REDUCE_AXIS) || axis->mc_sup == 0) {
      continue;
    }
    int last_mod_value = -1;
    size_t num_mod_axis = 0;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_MOD) {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      last_mod_value = StrToDecimalInt(attr.attr_value);
      ++num_mod_axis;
    }
    if (num_mod_axis < 1) {
      continue;
    }
    axis->TileRestrainLower(CastIntToExpr(last_mod_value), TileLevel::CACHE1);
    if (last_mod_value > max_x_y_dim_thread_) {
      LOG(WARNING) << "Cannot bind axis to " << last_mod_value << " threads, maximal thread number is "
                   << max_x_y_dim_thread_
                   << ". If fusing more than two axes together, footprint box calculated by isl may not be correct.";
      continue;
    }
    axis->thread_constraints.map_min_ = last_mod_value;
  }
}

void ReduceStrategy::DealWithPostReduceTensors() {
  std::unordered_set<std::string> post_reduce_tensors;
  auto root = analyzer_->RootAxis();
  for (const auto &attr : root->attrs) {
    if (attr.attr_key != AT_POST_FUSION_REDUCE_TENSOR) {
      continue;
    }
    auto tensor_name = attr.attr_value;
    (void)post_reduce_tensors.insert(tensor_name);
  }

  for (const auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_REDUCE_AXIS) {
        continue;
      }
      auto red_tensor_name = attr.attr_value;
      if (post_reduce_tensors.find(red_tensor_name) == post_reduce_tensors.end()) {
        continue;
      }
      axis->block_constraints.map_min_ = MIN_TILE;
      axis->block_constraints.map_extent_ = MIN_TILE;
      axis->thread_constraints.item_process_ = static_cast<int64_t>(SpItemPerThread::FULL);
    }
  }
}

int GpuStrategy::GetLocalAllocBufCount() {
  int count = 0;
  for (auto &it : analyzer_->buf_info_) {
    auto buf = it.second.get();
    CHECK(buf);
    if (buf->scope == TilingMemScope::MEM_SCOPE_LOCAL) {
      count++;
    }
  }
  return count;
}

void GpuStrategy::ApplyConstraintsToBindingSpace() {
  auto ParseBindingConstraint = [](const std::string &constraint, size_t max_size) {
    std::vector<std::string> sp = akg::common::Split(constraint, ",");
    std::vector<int64_t> ret;
    for (auto &val : sp) {
      if (ret.size() == max_size) {
        break;
      }
      CHECK(!val.empty());
      (void)ret.emplace_back(StrToDecimalInt64(val));
    }
    return ret;
  };

  std::unordered_set<std::string> thread_keys = {AT_THREAD_MIN, AT_THREAD_MAX, AT_THREAD_MOD};
  std::unordered_set<std::string> block_keys = {AT_BLOCK_MIN, AT_BLOCK_MAX, AT_BLOCK_MOD};
  for (const auto attr : analyzer_->RootAxis()->attrs) {
    std::vector<int64_t> constraint;
    std::vector<TileAxis::MappingConstraint> target;
    if (thread_keys.find(attr.attr_key) != thread_keys.end()) {
      constraint = ParseBindingConstraint(attr.attr_value, thread_binding_spaces_.size());
      target = thread_binding_spaces_;
    } else if (block_keys.find(attr.attr_key) != block_keys.end()) {
      constraint = ParseBindingConstraint(attr.attr_value, block_binding_spaces_.size());
      target = block_binding_spaces_;
    }
    if (constraint.empty()) {
      continue;
    }

    for (size_t i = 0; i < constraint.size(); ++i) {
      if (attr.attr_key.find("MIN") != std::string::npos) {
        target[i].map_min_ = std::max<int64_t>(target[i].map_min_, constraint[i]);
      } else if (attr.attr_key.find("MAX") != std::string::npos && constraint[i] > 0) {
        target[i].map_extent_ = std::min<int64_t>(target[i].map_extent_, constraint[i]);
      } else if (attr.attr_key.find("MOD") != std::string::npos) {
        target[i].map_mod_ = std::max<int64_t>(1, constraint[i]);
      }
    }

    if (thread_keys.find(attr.attr_key) != thread_keys.end()) {
      thread_binding_spaces_ = target;
    } else if (block_keys.find(attr.attr_key) != block_keys.end()) {
      block_binding_spaces_ = target;
    }
  }
}

void GpuStrategy::ApplyCustomConstraint() {
  // init binding space through template-determined limit
  thread_binding_spaces_.clear();
  block_binding_spaces_.clear();
  for (size_t i = 0; i < thread_limit_.size(); ++i) {
    TileAxis::MappingConstraint elem;
    elem.map_extent_ = thread_limit_[i];
    (void)thread_binding_spaces_.emplace_back(elem);
  }
  for (size_t i = 0; i < std::min(depth_, block_limit_.size()); ++i) {
    TileAxis::MappingConstraint elem;
    elem.map_extent_ = block_limit_[i];
    (void)block_binding_spaces_.emplace_back(elem);
  }

  ApplyConstraintsToBindingSpace();

  size_t cur_depth = 0;
  analyzer_->ForEachAxisTopDown([this, &cur_depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    auto cons = axis->GetConstConstraint(CACHE1);
    auto range_extent = axis->GetConstExtent();
    auto tile_min = cons.tile_min_.as<IntImm>()->value;
    int64_t tile_extent = cons.tile_extent_.as<IntImm>()->value;
    auto idx = reverse_binding_ ? cur_depth : (depth_ - 1) - cur_depth;

    auto thread_extent = tile_extent;
    if (idx < thread_binding_spaces_.size()) {
      thread_extent = std::min(thread_extent, thread_binding_spaces_[idx].map_extent_);
      thread_binding_spaces_[idx].map_extent_ = thread_extent;
    }

    auto block_extent = range_extent / SafeDivisor(tile_min);
    if (idx < block_binding_spaces_.size()) {
      block_extent = std::min<int64_t>(block_extent, block_binding_spaces_[idx].map_extent_);
      block_binding_spaces_[idx].map_extent_ = block_extent;
    }

    auto block_min = block_extent / SafeDivisor(thread_extent);
    if (idx < block_binding_spaces_.size()) {
      block_min = std::max<int64_t>(block_min, block_binding_spaces_[idx].map_min_);
      block_binding_spaces_[idx].map_min_ = block_min;
    }

    axis->thread_constraints.map_extent_ = thread_extent;
    axis->block_constraints.map_extent_ = block_extent;
    axis->block_constraints.map_min_ = block_min;
    if (idx < thread_binding_spaces_.size()) {
      axis->thread_constraints.map_mod_ = thread_binding_spaces_[idx].map_mod_;
    }
    if (idx < block_binding_spaces_.size()) {
      axis->block_constraints.map_mod_ = block_binding_spaces_[idx].map_mod_;
    }
    ++cur_depth;
  });
}

const void GpuStrategy::ShowOptions() {
  std::stringstream ss;
  ss << "Options:\n";
  std::string indent = "  ";
  ss << indent << "[EnableAkgReduceLib]: " << analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib() << "\n";
  ss << indent << "[EnableAtomicAdd]: " << analyzer_->scop_info_.user_config_.GetEnableAtomicAdd() << "\n";
  ss << indent << "[EnableStitchFusion]: " << analyzer_->scop_info_.user_config_.EnableStitchFusion() << "\n";
  ss << indent << "[EnableMatmul]: " << analyzer_->scop_info_.user_config_.GetEnableMatmul() << "\n";
  ss << indent << "[EnableTensorCore]: " << analyzer_->scop_info_.user_config_.GetEnableTensorCore() << "\n";
  ss << indent << "[EnableConvTensorCore]: " << analyzer_->scop_info_.user_config_.GetEnableConvTensorCore() << "\n";
  ss << indent << "[EnableOneDimThread]: " << analyzer_->scop_info_.user_config_.GetEnableOneDimThread() << "\n";
  ss << indent << "[EnableVectorization]: " << analyzer_->scop_info_.user_config_.GetEnableVectorization() << "\n";
  ss << indent << "[HasNormalTot]: " << analyzer_->scop_info_.analysis_result_.GetTensorOfTensor() << "\n";
  ss << indent << "[HasAtomicTot]: " << !analyzer_->scop_info_.analysis_result_.GetTensorOfTensorStmt().empty() << "\n";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

bool GpuStrategy::NeedModifyOrderOfAxis() {
  int last_axis = current_outer_bn_->last_axis;
  if (last_axis < 0 || last_axis >= static_cast<int>(pending_axes_.size())) {
    return false;
  }

  int real_pos = (static_cast<int>(pending_axes_.size()) - 1) - last_axis;
  if (real_pos == 0) {
    return false;
  }

  TileAxis *axis;
  int64_t shape;
  std::tie(axis, shape) = pending_axes_[static_cast<size_t>(real_pos)];
  pending_axes_.erase(pending_axes_.cbegin() + real_pos, pending_axes_.cbegin() + real_pos + 1);
  pending_axes_.push_front(std::make_pair(axis, shape));
  return true;
}

void GpuStrategy::CountGlobalBufferSize() {
  global_buf_size_ = 0;
  std::unordered_set<std::string> global_buf_names;
  for (auto &it : analyzer_->buf_info_) {
    auto buf = it.second.get();
    CHECK(buf != nullptr);
    if (buf->scope == TilingMemScope::MEM_SCOPE_GM) {
      global_buf_names.insert(buf->name);
    }
  }

  for (auto attr : analyzer_->RootAxis()->attrs) {
    auto buf = attr.attr_key;
    if (global_buf_names.count(buf)) {
      global_buf_size_ += StrToDecimalInt(attr.attr_value);
    }
  }
}

void GpuStrategy::SetInitTiledConfig() {
  InitMappingLimit();
  if (!analyzer_->scop_info_.user_config_.GetIsTuning()) {
    if (template_ == Template::BROADCAST_OP || template_ == Template::CUSTOM_CONFIG) {
      BroadcastSpeedup();
    } else if (template_ == Template::PAD_OP && !current_outer_bn_->enable_vectorization) {
      PadSpeedup();
    } else if (template_ == Template::TRANSPOSE_OP) {
      TransposeSpeedup();
    } else if (template_ == Template::REDUCTION || template_ == Template::BITWISE_REDUCTION) {
      use_shared_mem_ = true;
    }
  }
  BuildAxesQueue();
  if (analyzer_->scop_info_.user_config_.GetIsTuning()) {
    ApplyCustomConstraint();
    for (size_t i = 0; i < max_dim_; ++i) {
      TileAxis::MappingConstraint pad;
      if (i >= thread_binding_spaces_.size()) {
        (void)thread_binding_spaces_.emplace_back(pad);
      }
      if (i >= block_binding_spaces_.size()) {
        (void)block_binding_spaces_.emplace_back(pad);
      }
    }
  }
}

bool GpuStrategy::CheckNeedInjectiveSpeedUp() {
  std::stringstream ss;
  ss << "CheckNeedInjectiveSpeedUp: ";
  bool need_injective_speed_up = true;
  if ((template_ == Template::PURE_ELEM || template_ == Template::BROADCAST_OP || template_ == Template::EXTERN_CALL ||
       (template_ == Template::REDUCTION && !analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib())) &&
      analyzer_->scop_info_.analysis_result_.GetTensorOfTensor()) {
    need_injective_speed_up = !NeedModifyOrderOfAxis();
  } else {
    need_injective_speed_up = global_buf_size_ < max_buf_size_to_speedup_inj_;
  }
  return need_injective_speed_up;
}

void GpuStrategy::AddGpuConstraint() {
  CountGlobalBufferSize();
  bool is_first = true;
  std::stringstream ss;
  for (auto sorted_idx : analyzer_->GetSortedBands()) {
    band_index_ = sorted_idx;
    current_outer_bn_ = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(band_index_);
    SetInitTiledConfig();
    ss << "------Band " << band_index_ << ": " << analyzer_->scop_info_.analysis_result_.ShowOpTemplate(template_);
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

    if (analyzer_->scop_info_.user_config_.GetIsTuning()) return;
    // tensor of tensor
    bool need_injective_speed_up = CheckNeedInjectiveSpeedUp();

    block_count_ = 0;
    InnerThreadOuterBlock(is_first);

    // For the outer band is multiple filters, set the thread/block configuration according to the operator with the
    // highest priority, and other operators cannot change it.
    if (!is_first) {
      current_outer_bn_->enable_vectorization = false;
      continue;
    }

    if ((template_ == Template::PURE_ELEM || template_ == Template::PARTIAL_ELEM ||
         template_ == Template::EXTERN_CALL) &&
        need_injective_speed_up) {
      InjectiveSpeedup();
    }

    is_first = false;

    if ((template_ == Template::MATMUL || template_ == Template::CONV) &&
        analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
      continue;
    }

    analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
      if (axis == analyzer_->RootAxis()) {
        return;
      }
      if (!current_outer_bn_->enable_vectorization) {
        axis->TileRestrainToSingleValue(axis->c1_constraints.tile_min_, TileLevel::CACHE0);
      }
    });
  }
  ShowOptions();
}

void GpuStrategy::ThreadConfiguration(ReduceDirection direct, bool use_lib) {
  thread_limit_.clear();
  if (template_ == Template::CUSTOM_CONFIG) {
    auto thread_config = analyzer_->scop_info_.user_config_.GetThreadConfig();
    for (size_t i = 0; i < thread_config->bound; ++i) {
      auto idx = reverse_binding_ ? ((thread_config->bound - 1) - i) : i;
      if (idx >= depth_) {
        continue;
      }
      (void)thread_limit_.emplace_back(thread_config->GetAt(idx).second);
    }
  } else if (template_ == Template::REDUCTION || template_ == Template::BITWISE_REDUCTION) {
    if (direct == ReduceDirection::ALL && !use_lib) {
      thread_limit_ = {1};
    } else if (analyzer_->scop_info_.analysis_result_.GetCsr() && !use_lib) {
      thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_, max_z_dim_thread_};
    } else {
      thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_};
    }
  } else if (template_ == Template::TRANSPOSE_OP) {
    auto max_dim_thread = static_cast<int64_t>(std::floor(std::sqrt(total_available_thread_)));
    thread_limit_ = {max_dim_thread, max_dim_thread};
  } else if (template_ == Template::MATMUL) {
    // This is a naive tiling strategy used in gpu when thread and block configs are already set.
    // This strategy will tile up to three inner-most axes to 32 (for thread binding).
    if (analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
      thread_limit_ = {warp_sizes_, 16};
    } else {
      thread_limit_ = {warp_sizes_, 8};
    }
  } else {
    thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_, max_z_dim_thread_};
  }
}

void GpuStrategy::UpdateBlockConfig(ReduceDirection direct, bool use_lib) {
  block_limit_.clear();
  if (template_ == Template::CUSTOM_CONFIG) {
    auto block_config = analyzer_->scop_info_.user_config_.GetBlockConfig();
    for (size_t i = 0; i < block_config->bound - 1; ++i) {
      if (i >= depth_) {
        break;
      }
      (void)block_limit_.emplace_back(block_config->GetAt(i).second);
    }
  } else if (template_ == Template::REDUCTION && direct == ReduceDirection::ALL && !use_lib) {
    block_limit_ = {1};
  } else if (template_ == Template::CONV) {
    block_limit_ = {max_x_dim_block_, max_y_z_dim_block_, max_y_z_dim_block_, max_y_z_dim_block_};
  } else {
    block_limit_ = {max_x_dim_block_, max_y_z_dim_block_, max_y_z_dim_block_};
  }
}

void GpuStrategy::InitMappingLimit() {
  max_x_y_dim_thread_ = analyzer_->scop_info_.user_config_.GetMaxElemPerThread();

  // Determine op template
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](const TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->index != band_index_) {
      return;
    }
    ++depth;
  });
  depth_ = depth;
  if (analyzer_->scop_info_.user_config_.GetThreadConfig() != nullptr &&
      analyzer_->scop_info_.user_config_.GetBlockConfig() != nullptr &&
      analyzer_->scop_info_.user_config_.GetThreadConfig()->bound > 0 &&
      analyzer_->scop_info_.user_config_.GetBlockConfig()->bound > 0) {
    template_ = Template::CUSTOM_CONFIG;
  } else {
    template_ = current_outer_bn_->template_type;
  }
  ReduceDirection direct = current_outer_bn_->reduce_direction;
  bool use_lib = analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib();
  reverse_binding_ = use_lib && direct == ReduceDirection::Y;

  ThreadConfiguration(direct, use_lib);

  if (template_ != Template::CUSTOM_CONFIG && !analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
    AdjustThreadMappingLimit();
  }

  // Block configuration
  UpdateBlockConfig(direct, use_lib);
}

void GpuStrategy::BuildAxesQueue() {
  this->pending_axes_.clear();
  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis() || axis->index != band_index_) {
      return;
    }
    int extent = static_cast<int>(axis->extent_val);

    // When vectorization is enabled, make sure non-vectorization axes' c0 tile equals to 1
    if (current_outer_bn_->enable_vectorization) {
      axis->TileRestrainToSingleValue(axis->c0_constraints.tile_min_, TileLevel::CACHE0);
    }

    // For Conv, kh and kw are invalid for pending_axes
    if (!axis->is_inner && extent > 0) {
      this->pending_axes_.push_front(std::make_pair(axis, extent));
    }

    // init map extent to shape if they are not modified by other constraints
    axis->block_constraints.map_extent_ =
      axis->block_constraints.map_extent_ == 0 ? axis->extent_val : axis->block_constraints.map_extent_;
    axis->thread_constraints.map_extent_ =
      axis->thread_constraints.map_extent_ == 0 ? axis->extent_val : axis->thread_constraints.map_extent_;
    if (!axis->mc_sup &&
        (!analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib() || template_ == Template::PARTIAL_ELEM)) {
      axis->block_constraints.map_extent_ = 1;
      axis->thread_constraints.map_extent_ = 1;
      std::stringstream ss;
      ss << "Axis " << axis->index << "_" << axis->dim_axis;
      if (template_ == Template::PARTIAL_ELEM) {
        axis->c1_constraints.tile_extent_ = 1;
        ss << " Coincidence = 0 and template = PARTIAL_ELEM, disable block/thread mapping.";
      } else {
        ss << " Coincidence = 0 and Akg-reduce-lib not enabled, disable block/thread mapping.";
      }
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    }
  });
}

void GpuStrategy::SkipMapping(TileAxis *axis, int64_t shape, std::stringstream &ss, size_t inner_dim,
                              size_t thread_dim) {
  axis->thread_constraints.map_extent_ = 1;
  (void)axis->thread_constraints.map_cand_.emplace_back(1);
  auto tile = inner_dim < thread_dim ? elem_per_thread_[inner_dim] : 1;
  tile = tile == static_cast<int64_t>(SpItemPerThread::AUTO)
           ? std::min(axis->thread_constraints.item_process_, max_elem_per_thread_)
           : tile == static_cast<int64_t>(SpItemPerThread::FULL) ? std::min(shape, max_elem_per_thread_) : 1;
  CHECK(axis->c1_constraints.tile_min_.as<IntImm>() && axis->c1_constraints.tile_extent_.as<IntImm>());
  auto tile_min = axis->c1_constraints.tile_min_.as<IntImm>()->value;
  auto tile_extent = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
  if (tile_min == tile_extent && tile_extent != MIN_TILE) {
    ss << "tile extent is already determined = " << tile_extent;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    tile = tile_min;
  } else {
    if (axis->block_constraints.map_extent_ > 1) {
      // block.min <= shape / tile <= block.max
      tile = std::max(
        tile,
        std::max<int64_t>(
          static_cast<int64_t>(ceil(static_cast<float>(shape) / SafeDivisor(axis->block_constraints.map_extent_))), 1));
    } else {
      tile = std::min(tile, shape);
    }
  }
  tile = std::max<int64_t>(tile, tile_min);
  tile = std::min<int64_t>(tile, tile_extent);
  axis->TileRestrainLower(tile, TileLevel::CACHE1);
  ss << ", tile = " << tile;

  if (axis->mc_sup || analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    if (axis->block_constraints.map_extent_ > 1) {
      CHECK(tile);
      pending_axes_.push_back(std::make_pair(
        axis, std::max<int64_t>(static_cast<int64_t>(ceil(static_cast<float>(shape) / SafeDivisor(tile))), 1)));
      ss << ", map to block.";
    }
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GpuStrategy::GreedyMapBlocks(size_t ori_size, size_t block_dim) {
  // If all axes for block mapping are element-wise, we can map them in any order
  // so we need a greedy algorithm to map the most blocks;
  // otherwise, we can simply map from outer to inner in sequence.
  std::stringstream ss;
  if (template_ == Template::PURE_ELEM || template_ == Template::EXTERN_CALL) {
    std::map<int64_t, std::vector<size_t>, std::greater<int64_t>> sorted_by_gcd;
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      auto block_limit = i == 0 ? max_x_dim_block_ : max_y_z_dim_block_;
      auto use = (block_limit > 0 && pending_axes_[i].second > 0)
                   ? TilingAnalyzer::FindDivisibleTilingFactor(block_limit, pending_axes_[i].second)
                   : 1;
      if (sorted_by_gcd.find(use) == sorted_by_gcd.end()) {
        sorted_by_gcd[use] = {i};
      } else {
        (void)sorted_by_gcd[use].emplace_back(i);
      }
    }

    for (const auto &it : sorted_by_gcd) {
      auto index_list = it.second;
      for (const auto &i : index_list) {
        if (pending_axes_.size() - i > block_dim) {
          auto axis = pending_axes_[i].first;
          ss << "axis " << axis->index << "_" << axis->dim_axis

             << " exceeded block dim and should be mapped to block for higher performance, consider flatten";
          analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
          continue;
        }
        (void)indexing_.emplace_back(i);
      }
    }
  } else {
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      if (pending_axes_[i].second <= 1 && indexing_.size() == block_limit_.size()) {
        continue;
      }
      (void)indexing_.emplace_back(i);
    }
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GpuStrategy::CheckAlignedUse(int64_t &use, int64_t shape, TileAxis *axis, std::stringstream &ss) {
  if (axis->forbid_iso && use != 0 && shape % SafeDivisor(use) != 0) {
    auto aligned_use = TilingAnalyzer::FindDivisibleTilingFactor(use, shape);
    CHECK(aligned_use);
    if (aligned_use % SafeDivisor(axis->thread_constraints.map_mod_) != 0) {
      return;
    }
    // balance thread size and code complexity
    bool efficient = (use / SafeDivisor(aligned_use) < 4 || activated_threads_ >= warp_sizes_);
    ss << ", forbid iso and adjust use: original = " << use << ", adjust to " << aligned_use << " is efficient ? "
       << efficient;
    if (efficient) {
      use = aligned_use;
    }
  }
}

void GpuStrategy::MapPendingAxes(size_t ori_size, std::stringstream &ss, size_t thread_dim, bool write_cfg) {
  size_t inner_dim = 0;
  for (size_t i = 0; i < ori_size; ++i) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    int64_t rest_threads =
      std::min(total_available_thread_ / SafeDivisor(activated_threads_), thread_limit_[thread_cfg_.size()]);
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape
       << ", rest_threads = " << rest_threads;
    ss << "\n--------> Tile: " << axis->c1_constraints.tile_min_ << "," << axis->c1_constraints.tile_extent_;
    ss << "\n--------> Tile0: " << axis->c0_constraints.tile_min_ << "," << axis->c0_constraints.tile_extent_;
    ss << "\n--------> Thread: " << axis->thread_constraints.map_min_ << "," << axis->thread_constraints.map_extent_;
    ss << "\n--------> Block: " << axis->block_constraints.map_min_ << "," << axis->block_constraints.map_extent_;

    if (template_ != Template::CUSTOM_CONFIG) {
      rest_threads = std::min(rest_threads, axis->thread_constraints.map_extent_);
    }

    if ((thread_cfg_.size() >= thread_dim && write_cfg) || inner_dim >= max_dim_) {
      ss << ", no thread/dim rests";
      SkipMapping(axis, shape, ss, inner_dim, thread_dim);
      continue;
    }

    // For Conv, hi and wi are invalid for thread mapping
    if (axis->HasAttr(AttrInfo{AT_CONV, kDsahi}) || axis->HasAttr(AttrInfo{AT_CONV, kDsawi})) {
      SkipMapping(axis, shape, ss, inner_dim, thread_dim);
      continue;
    }

    if (rest_threads <= 1) {
      if (axis->mc_sup ||
          (template_ == Template::REDUCTION && analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib())) {
        if (write_cfg) {
          (void)thread_cfg_.emplace_back(1);
        }
        (void)axis->thread_constraints.map_cand_.emplace_back(1);
        thread_cfg_map_[axis] = static_cast<int64_t>(i);
      }
      SkipMapping(axis, shape, ss, inner_dim, thread_dim);
      continue;
    }

    auto item = elem_per_thread_[inner_dim] == static_cast<int64_t>(SpItemPerThread::AUTO)
                  ? axis->thread_constraints.item_process_
                  : elem_per_thread_[inner_dim];
    item = std::min(item, max_elem_per_thread_);
    int64_t use;
    if (analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(axis->range_extent)) {
      use = analyzer_->scop_info_.user_config_.GetCsrThreadNum();
    } else {
      use = GetThreadSize(rest_threads, inner_dim, shape, item);
      ss << ", proposed use = " << use;
      if (use >= axis->thread_constraints.map_mod_ && axis->thread_constraints.map_mod_ > 0) {
        use = AlignToDivisibleSize(use, axis->thread_constraints.map_mod_);
      }
      ss << ", thread mod = " << axis->thread_constraints.map_mod_;
      if (!write_cfg && thread_cfg_.size() > i) {
        use = thread_cfg_[i];
      }
    }

    if (axis->forbid_iso || !analyzer_->scop_info_.user_config_.GetEnableOneDimThread() || !thread_cfg_.empty()) {
      // do not align thread usage for threadX with one dim thread enabled
      ss << ", check aligned use...";
      CheckAlignedUse(use, shape, axis, ss);
    }

    activated_threads_ *= use;
    ss << ", final use = " << use << ", activated threads = " << activated_threads_;
    if (write_cfg) {
      (void)thread_cfg_.emplace_back(use);
    }
    (void)axis->thread_constraints.map_cand_.emplace_back(use);

    thread_cfg_map_[axis] = static_cast<int64_t>(i);
    axis->thread_constraints.map_extent_ = use;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    auto tile = TileAfterThreadMapping(axis, inner_dim, use, item);
    if (current_outer_bn_->enable_vectorization && axis->HasAttr(AT_VECTORIZED)) {
      CHECK(axis->c0_constraints.tile_min_.as<IntImm>());
      auto c0_tile = axis->c0_constraints.tile_min_.as<IntImm>()->value;
      if (tile * c0_tile <= shape) {
        tile *= c0_tile;
        ss << ", vectorized axis, multiply with " << c0_tile;
      }
    }
    CHECK(axis->c1_constraints.tile_mod_.as<IntImm>());
    auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
    CHECK(tile);
    ss << ", original tile = " << tile << ", tile mod = " << tile_mod;
    if (tile >= tile_mod && tile_mod > 0) {
      tile = AlignToDivisibleSize(tile, tile_mod);
    }
    ss << ", final c1 tile = " << tile;
    pending_axes_.push_back(std::make_pair(
      axis, std::max<int64_t>(static_cast<int64_t>(ceil(static_cast<float>(shape) / SafeDivisor(tile))), 1)));
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    ++inner_dim;
  }
}

void GpuStrategy::InnerThreadOuterBlock(bool write_cfg) {
  if (pending_axes_.empty()) {
    return;
  }
  std::stringstream ss;
  activated_threads_ = 1;
  int64_t activated_blocks = 1;

  auto thread_dim = std::min(thread_limit_.size(), max_dim_);
  auto block_dim = std::min(block_limit_.size(), max_dim_);

  if (analyzer_->scop_info_.user_config_.GetEnableConvTensorCore()) {
    block_dim = block_limit_.size();
  }

  // tile from inner to outer and map to thread
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "-----Map to thread-----");
  ss << "[Thread Limit]: ";
  for (auto l : thread_limit_) {
    ss << l << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  size_t ori_size = pending_axes_.size();
  MapPendingAxes(ori_size, ss, thread_dim, write_cfg);

  indexing_.clear();
  if (write_cfg) {
    for (size_t i = 0; i < block_dim; ++i) {
      (void)block_cfg_.emplace_back(1);
    }
  }
  GreedyMapBlocks(ori_size, block_dim);

  // map outer band to block according to predefined indice
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "-----Map to block-----");
  ss << "[Block Limit]: ";
  for (auto l : block_limit_) {
    ss << l << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  for (const auto &i : indexing_) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    auto idx = ((indexing_.size() - 1) - ((pending_axes_.size() - 1) - i));
    auto rest_blocks = idx < block_limit_.size() ? std::min(block_limit_[idx], axis->block_constraints.map_extent_) : 1;
    rest_blocks = std::min(rest_blocks, shape);
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape << ", block_idx = " << idx
       << ", rest blocks = " << rest_blocks;
    ss << "\n--------> Tile1: " << axis->c1_constraints.tile_min_ << "," << axis->c1_constraints.tile_extent_;
    ss << "\n--------> Thread: " << axis->thread_constraints.map_min_ << "," << axis->thread_constraints.map_extent_;
    ss << "\n--------> Block: " << axis->block_constraints.map_min_ << "," << axis->block_constraints.map_extent_;
    if (block_count_ >= static_cast<int>(block_dim)) {
      ss << "-> No mapping.";
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
      continue;
    }
    auto use = 1;
    if (rest_blocks > 0 && shape > 1) {
      auto aligned_blocks = TilingAnalyzer::FindDivisibleTilingFactor(rest_blocks, shape);
      ss << "aligned_blocks = " << aligned_blocks << ", rest_blocks = " << rest_blocks;
      if (aligned_blocks <= 1 || aligned_blocks * min_elem_for_io_bound_ * double_ < rest_blocks) {
        use = rest_blocks;
      } else {
        use = aligned_blocks;
      }
    }
    if (!write_cfg) {
      use = block_cfg_[(pending_axes_.size() - 1) - i];
    }
    activated_blocks *= use;
    ss << ", use = " << use << ", activated blocks = " << activated_blocks;
    if (write_cfg) {
      block_cfg_[(pending_axes_.size() - 1) - i] = use;
    }
    (void)axis->block_constraints.map_cand_.emplace_back(use);

    block_cfg_map_[axis] = (pending_axes_.size() - 1) - i;

    axis->block_constraints.map_extent_ = use;
    if (analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib() || axis->mc_sup) {
      ++block_count_;
    }

    auto extent = axis->extent_val;
    auto thread_size = axis->thread_constraints.map_extent_;
    auto upper = std::max<int64_t>(static_cast<int64_t>(ceil(static_cast<float>(extent) / SafeDivisor(use))), 1);
    CHECK(axis->c1_constraints.tile_min_.as<IntImm>());
    bool partial_block_and_thread = (axis->c1_constraints.tile_min_.as<IntImm>()->value > upper) && (use > thread_size);
    if (!use_shared_mem_ && axis->forbid_iso && axis->extent_val % upper != 0) {
      axis->TileRestrainToSingleValue(thread_size, TileLevel::CACHE1);
    } else if (partial_block_and_thread) {
      axis->TileRestrainToSingleValue(upper, TileLevel::CACHE1);
    } else {
      axis->TileRestrainUpper(upper, TileLevel::CACHE1);
    }
    ss << ", tile range = [" << axis->c1_constraints.tile_min_ << ", " << axis->c1_constraints.tile_extent_ << "]";
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

const int64_t GpuStrategy::GetThreadSize(const int64_t rest_threads, size_t inner_dim, const int64_t shape,
                                         const int64_t item) {
  // Current experience is that let mapped threads divisible by warp_size to increase performance.
  CHECK(item);
  int64_t thread_extent = item == SpItemPerThread::FULL
                            ? rest_threads
                            : static_cast<int64_t>(ceil(static_cast<float>(shape) / SafeDivisor(item)));
  thread_extent = std::min(thread_extent, shape);
  if (thread_extent > rest_threads || template_ == Template::CUSTOM_CONFIG) {
    return rest_threads;
  }
  auto proposal = inner_dim == 0 ? ((((thread_extent - 1) + warp_sizes_) / warp_sizes_) * warp_sizes_) : thread_extent;
  return std::min(rest_threads, proposal);
}

int64_t GpuStrategy::ApplyCustomTile(TileAxis *axis, size_t inner_dim, int64_t thread_size, int64_t tile,
                                     int64_t shape) {
  std::stringstream ss;
  if (template_ == Template::CUSTOM_CONFIG) {
    if (!analyzer_->scop_info_.user_config_.GetEnableAtomicAdd() &&
        (axis->HasAttr(AT_REDUCE_AXIS) || axis->mc_sup == 0)) {
      tile = shape;
      ss << "tile = shape to disable atomic add, ";
    } else if (tile < thread_size) {
      tile = thread_size;
      ss << "tile = thread size, ";
    } else {
      int64_t block_dim = reverse_binding_ ? static_cast<int64_t>((block_limit_.size() - 1) - inner_dim)
                                           : static_cast<int64_t>(inner_dim);
      int64_t least_blocks;
      if (block_dim >= 0 && block_dim < static_cast<int64_t>(block_limit_.size())) {
        least_blocks = block_limit_[block_dim];
      } else {
        least_blocks = std::accumulate(block_limit_.begin(), block_limit_.end(), 1, std::multiplies<int>());
      }
      auto max_tile = shape / SafeDivisor(least_blocks);
      if (thread_size != 0 && shape % SafeDivisor(thread_size) == 0) {
        // ensure no if condition in thread for-loop
        tile = TilingAnalyzer::FindDivisibleTilingFactor(max_tile, shape);
      } else {
        // ensure thread for-loop bound has no min/max
        tile = TilingAnalyzer::FindDivisibleTilingFactor(max_tile, thread_size);
      }
      ss << "reduce tile size to enable at least " << least_blocks << " blocks, ";
    }
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  return tile;
}

int64_t GpuStrategy::TileAfterThreadMapping(TileAxis *axis, size_t inner_dim, int64_t thread_size, const int64_t item) {
  std::stringstream ss;
  auto shape = axis->extent_val;
  CHECK(axis->c1_constraints.tile_min_.as<IntImm>() && axis->c1_constraints.tile_mod_.as<IntImm>() &&
        axis->c1_constraints.tile_extent_.as<IntImm>());
  auto tile_min = axis->c1_constraints.tile_min_.as<IntImm>()->value;
  auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
  auto tile_extent = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
  if (tile_min == tile_extent && tile_extent != MIN_TILE) {
    ss << "tile extent is already determined = " << tile_extent;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    return tile_extent;
  }

  auto tile = item == static_cast<int64_t>(SpItemPerThread::FULL)
                ? std::min(tile_extent, thread_size * max_elem_per_thread_)
                : std::min(tile_extent, thread_size * item);
  tile = std::max<int64_t>(tile, tile_min);
  if (analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
    if (tile < tile_mod) {
      // tile axis with mod value
      // e.g. tile cc0 with 128 in the following code
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (tile_mod % SafeDivisor(tile) != 0 && tile > thread_size) {
        --tile;
      }
    } else {
      // tile axis with div value
      // e.g. tile cc0 with 512 in the following code (which equals tile floordiv(cc0, 256) with 2)
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (shape % SafeDivisor(tile) != 0 && tile > thread_size) {
        --tile;
      }
    }
  }

  bool partial_block = (shape / SafeDivisor(tile) <= 1 && shape > tile);
  if (partial_block) {
    tile = shape;
  }

  tile = ApplyCustomTile(axis, inner_dim, thread_size, tile, shape);

  ss << "axis " << axis->index << "_" << axis->dim_axis << " elem_per_thread = " << item << ", tile = " << tile;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  axis->TileRestrainLower(CastInt64ToExpr(tile), TileLevel::CACHE1);
  return tile;
}

void GpuStrategy::AdjustThreadMappingLimit() {
  std::stringstream ss;
  std::vector<int64_t> map_mins;
  ss << "Original thread limit = ";
  for (auto tl : thread_limit_) {
    ss << tl << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  analyzer_->ForEachAxisTopDown([this, &map_mins](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    if (!axis->mc_sup && !analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib()) {
      return;
    }
    (void)map_mins.emplace_back(axis->thread_constraints.map_min_);
  });
  std::reverse(map_mins.begin(), map_mins.end());
  auto map_size = thread_limit_.size();
  for (size_t i = 0; i < map_mins.size(); ++i) {
    if (i > map_size) {
      continue;
    }
    for (size_t j = 0; j < map_size; ++j) {
      if (j == i) {
        continue;
      }
      int64_t res = static_cast<int64_t>(floor(static_cast<float>(thread_limit_[j]) / SafeDivisor(map_mins[i])));
      thread_limit_[j] = res;
    }
  }
  ss << "Adjust thread limit by axes' mapping mins = ";
  for (auto tl : thread_limit_) {
    ss << tl << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GpuStrategy::WriteConfigBackInjective() {
  std::stringstream ss;
  for (size_t i = 0; i < injective_axes_.size(); ++i) {
    ss << "replace block " << block_cfg_[i] << " with " << injective_axes_[i]->block_constraints.map_extent_
       << " replace thread " << thread_cfg_[(injective_axes_.size() - 1) - i] << " with "
       << injective_axes_[i]->thread_constraints.map_extent_;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    block_cfg_[i] = injective_axes_[i]->block_constraints.map_extent_;
    injective_axes_[i]->block_constraints.map_cand_.clear();
    injective_axes_[i]->block_constraints.map_cand_.push_back(block_cfg_[i]);

    if (g_csr.empty()) {
      thread_cfg_[(injective_axes_.size() - 1) - i] = injective_axes_[i]->thread_constraints.map_extent_;
      injective_axes_[i]->thread_constraints.map_cand_.clear();
      injective_axes_[i]->thread_constraints.map_cand_.push_back(thread_cfg_[(injective_axes_.size() - 1) - i]);
    }
  }
}

std::pair<int64_t, int64_t> GpuStrategy::GetProposalParallelSize(int problem_size) {
  int64_t block_size = 1;
  int64_t thread_size = 1;
  thread_coef_ = thread_coef_init_;
  if (problem_size <= warp_sizes_) {
    thread_size = warp_sizes_;
  } else if (problem_size <= warp_sizes_ * num_sm_) {
    thread_size = warp_sizes_;
    block_size = num_sm_;
  } else if (problem_size <= warp_sizes_ * thread_coef_.first * num_sm_ * active_blocks_per_sm_.first) {
    thread_size = TilingAnalyzer::FindDivisibleTilingFactor(warp_sizes_ * thread_coef_.first, problem_size);
    block_size = num_sm_ * active_blocks_per_sm_.first;
  } else if (problem_size <= warp_sizes_ * thread_coef_.second * num_sm_ * active_blocks_per_sm_.second) {
    thread_size = TilingAnalyzer::FindDivisibleTilingFactor(warp_sizes_ * thread_coef_.second, problem_size);
    block_size = num_sm_ * active_blocks_per_sm_.second;
  } else {
    thread_size = total_available_thread_;
    block_size = num_sm_ * active_blocks_per_sm_.second;
  }

  return std::make_pair(block_size, thread_size);
}

int64_t GpuStrategy::AlignThreadToShape() {
  std::stringstream ss;
  // sum of threadIdx.x, threadIdx.y and threadIdx.z
  auto total_threads = std::accumulate(thread_cfg_.begin(), thread_cfg_.end(), 1, std::multiplies<int>());
  for (size_t i = 0; i < injective_axes_.size(); ++i) {
    auto axis = injective_axes_[i];
    auto shape = axis->extent_val;
    CHECK(axis->c1_constraints.tile_extent_.as<IntImm>());
    auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
    auto thread_size = axis->thread_constraints.map_extent_;
    if (shape % SafeDivisor(thread_size) == 0) {
      continue;
    }

    int64_t lower = analyzer_->FindDivisibleTilingFactor(thread_size, shape);
    bool is_invalid = (lower % SafeDivisor(axis->thread_constraints.map_mod_ != 0));
    if (is_invalid) {
      ss << "thread size is invalid: " << lower << " % " << axis->thread_constraints.map_mod_ << " != 0";
      continue;
    }

    // The modified thread_size cannot be reduced to half of the original thread_size
    // The modified total thread_size cannot be reduced to half of 1024
    bool is_efficient = lower * binary_factor_ > thread_size ||
                        ((total_threads / SafeDivisor(thread_size)) * lower) * binary_factor_ >= max_x_y_dim_thread_;
    if (is_efficient) {
      ss << "align thread from " << thread_size << " to " << lower << " according to shape " << shape;
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
      axis->thread_constraints.map_extent_ = lower;
      total_threads = (total_threads / SafeDivisor(thread_size)) * lower;

      auto thread_loop = std::max<int>(1, tile_size / SafeDivisor(thread_size));
      tile_size = std::min(shape, thread_loop * lower);
      CHECK(axis->c1_constraints.tile_mod_.as<IntImm>());
      auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
      if (tile_size % SafeDivisor(tile_mod) != 0) {
        ss << "tile size is invalid: " << tile_size << " % " << tile_mod << " != 0";
        continue;
      }
      axis->TileRestrainToSingleValue(tile_size, TileLevel::CACHE1);
      CHECK(tile_size);
      axis->block_constraints.map_extent_ =
        static_cast<int64_t>(ceil(static_cast<float>(shape) / SafeDivisor(tile_size)));
    }
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  WriteConfigBackInjective();
  return total_threads;
}

void GpuStrategy::HandleShrinkThreadToBlock(int64_t &shrinked_threads, bool thread_to_block, std::stringstream &ss) {
  if (thread_to_block) {
    for (auto axis : injective_axes_) {
      if (shrinked_threads <= 0) {
        break;
      }
      auto thread_size = axis->thread_constraints.map_extent_;
      auto block_size = axis->block_constraints.map_extent_;
      CHECK(axis->c1_constraints.tile_extent_.as<IntImm>());
      auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
      auto coef = TilingAnalyzer::FindDivisibleTilingFactor(shrinked_threads, thread_size);
      CHECK(coef != 0);
      shrinked_threads /= SafeDivisor(coef);
      axis->thread_constraints.map_extent_ = thread_size / SafeDivisor(coef);
      axis->block_constraints.map_extent_ = block_size * coef;
      axis->TileRestrainToSingleValue(tile_size / SafeDivisor(coef), TileLevel::CACHE1);
      ss << "axis " << axis->dim_axis << " before shrink " << thread_size << " shrink size " << coef;
    }
  }
}

void GpuStrategy::InjectiveSpeedup() {
  // not need speedup if thread_cfg_ or block_cfg_ is empty
  if (thread_cfg_.size() == 0 || block_cfg_.size() == 0) {
    return;
  }
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "InjectiveSpeedup");
  std::stringstream ss;
  injective_axes_.clear();
  auto problem_size = 1;
  auto curr_elem_size = 1;
  analyzer_->ForEachAxisTopDown([this, &problem_size, &curr_elem_size](TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr || axis->is_inner ||
        axis->index != band_index_) {
      return;
    }
    (void)injective_axes_.emplace_back(axis);
    problem_size *= static_cast<int>(axis->extent_val);
    curr_elem_size *= axis->thread_constraints.item_process_;
  });

  // Step 1. Reduce code complexity by aligning thread size to shape
  int64_t total_threads = AlignThreadToShape();

  // Step 2. Adjust the ratio of thread for-loop, thread size and block size.
  if (!injective_axes_.size()) {
    return;
  }
  auto coaleasced_size = injective_axes_.back()->thread_constraints.map_extent_;
  auto total_blocks = std::accumulate(block_cfg_.begin(), block_cfg_.end(), 1, std::multiplies<int>());
  auto parallel_size = GetProposalParallelSize(problem_size);
  auto proposal_blocks = parallel_size.first;
  auto proposal_threads = parallel_size.second;
  auto proposal_elem_per_thread =
    coaleasced_size < warp_sizes_ ? 1 : total_blocks < proposal_blocks * 8 ? min_elem_for_io_bound_ : 8;
  proposal_elem_per_thread = proposal_elem_per_thread / SafeDivisor(curr_elem_size);
  CHECK(proposal_threads != 0 && total_blocks != 0);
  int64_t shrinked_threads =
    std::min<int64_t>(total_threads / SafeDivisor(proposal_threads), proposal_blocks / SafeDivisor(total_blocks));

  int64_t shrinked_blocks = (total_blocks - 1 + proposal_blocks) / SafeDivisor(proposal_blocks);

  auto thread_to_block = shrinked_threads > 0 && total_blocks < proposal_blocks;
  auto block_to_elem = proposal_elem_per_thread > 0 && shrinked_blocks > 0;
  auto thread_to_elem = proposal_elem_per_thread > 0 && !block_to_elem && shrinked_threads > 0 &&
                        total_blocks * shrinked_threads > proposal_blocks * proposal_elem_per_thread;
  ss << "problem_size = " << problem_size << " coaleasced_size = " << coaleasced_size
     << " total_blocks = " << total_blocks << " total_threads = " << total_threads
     << " proposal_blocks = " << proposal_blocks << " proposal_threads = " << proposal_threads
     << " proposal_elem_per_thread = " << proposal_elem_per_thread << " shrinked_threads = " << shrinked_threads
     << " shrinked_blocks = " << shrinked_blocks;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  ss << "Parallel size = [" << proposal_blocks << ", " << proposal_threads << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  HandleShrinkThreadToBlock(shrinked_threads, thread_to_block, ss);

  if (block_to_elem || thread_to_elem) {
    for (auto axis : injective_axes_) {
      auto shrink_limit = block_to_elem ? shrinked_blocks : shrinked_threads;
      if (shrink_limit <= 0) {
        break;
      }
      CHECK(axis->c1_constraints.tile_extent_.as<IntImm>());
      auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
      auto before_shrink = block_to_elem ? axis->block_constraints.map_extent_ : axis->thread_constraints.map_extent_;
      auto coef = std::min<int64_t>(proposal_elem_per_thread, shrink_limit);
      CHECK(coef);
      shrink_limit = std::min<int64_t>(shrink_limit, before_shrink);
      int64_t aligned_coef = TilingAnalyzer::FindDivisibleTilingFactor(shrink_limit, before_shrink);
      ss << "\nTo elem: before shrink = " << before_shrink << " shrink limit " << shrink_limit
         << " aligned_coef = " << aligned_coef;
      ss << " origianl coef = " << coef;
      if (aligned_coef * binary_factor_ > coef || axis->forbid_iso) {
        coef = aligned_coef;
      }
      ss << " final coef = " << coef << "\n";
      if (block_to_elem) {
        auto before_shrink_limit = std::max<int64_t>(before_shrink / SafeDivisor(coef), 1);
        auto actual_block = TilingAnalyzer::FindDivisibleTilingFactor(before_shrink_limit, before_shrink);
        auto actual_coef = before_shrink / SafeDivisor(actual_block);
        if (actual_coef > shrink_limit) {
          ss << "actual shrink = " << actual_coef << "exceed shrink limit: " << shrink_limit << ", continue.";
          continue;
        }
        ss << "block_to_elem: \n"
           << shrinked_blocks << " /= " << coef << "\n block = " << before_shrink << " / " << coef << " = "
           << axis->block_constraints.map_extent_ << ", curr actual_coef = " << actual_coef << "\n";
        coef = actual_coef;
        shrinked_blocks /= SafeDivisor(coef);
        axis->block_constraints.map_extent_ = actual_block;
      } else {
        shrinked_threads /= coef;
        axis->thread_constraints.map_extent_ = before_shrink / SafeDivisor(coef);
      }
      ss << "axis " << axis->dim_axis << " before shrink " << before_shrink << " shrink size " << coef;
      axis->TileRestrainToSingleValue(tile_size * coef, TileLevel::CACHE1);
    }
  }
  WriteConfigBackInjective();
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

const void GpuStrategy::TransposeSpeedup() {
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "TransposeSpeedup");
  if (!analyzer_->scop_info_.user_config_.EnableStitchFusion()) {
    analyzer_->scop_info_.user_config_.SetEnableOneDimThread(true);
  }
  current_outer_bn_->use_shared_memory = true;
  auto inner_axes = analyzer_->GetAxesOfAttr(AT_TRANSPOSE_INNERMOST_AXIS);
  if (inner_axes.size() == 1) {
    inner_axes[0]->TileRestrainLower(tranpose_tiling_constraints_, TileLevel::CACHE1);
    inner_axes[0]->thread_constraints.item_process_ = min_elem_for_io_bound_;
  } else {
    std::vector<TileAxis *> axes;
    auto problem_size = 1;
    auto curr_size = 1;

    analyzer_->ForEachAxisTopDown([this, &axes, &problem_size, &inner_axes, &curr_size](TileAxis *axis) {
      if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr || axis->index != band_index_) {
        return;
      }
      (void)axes.emplace_back(axis);
      problem_size *= static_cast<int>(axis->extent_val);
      if (find(inner_axes.begin(), inner_axes.end(), axis) != inner_axes.end()) {
        curr_size *= tranpose_tiling_constraints_;
      }
    });
    auto parallel_size = GetProposalParallelSize(problem_size);
    auto proposal_blocks = parallel_size.first;
    auto total_elem_for = problem_size / SafeDivisor(curr_size * proposal_blocks);
    std::stringstream ss;

    ss << "Propose block size = " << proposal_blocks << ", curr size = " << curr_size
       << " problem size = " << problem_size << ", min_elem_for_io_bound_=" << min_elem_for_io_bound_
       << ", total_elem_for = " << total_elem_for;

    auto cur_elem_for = 1;
    for (int i = static_cast<int>(axes.size()) - 1; i >= 0; --i) {
      if (total_elem_for / SafeDivisor(cur_elem_for) <= 1) {
        ss << "stop allocate elem for at axis " << i;
        break;
      }
      auto axis = axes[i];
      if (find(inner_axes.begin(), inner_axes.end(), axis) != inner_axes.end()) {
        axis->TileRestrainLower(tranpose_tiling_constraints_, TileLevel::CACHE1);
        axis->thread_constraints.item_process_ = min_elem_for_io_bound_;
        cur_elem_for *= min_elem_for_io_bound_;
      } else {
        axis->TileRestrainUpper(MIN_TILE, TileLevel::CACHE1);
      }
    }
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void GpuStrategy::PadSpeedup() {
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "PadSpeedup");
  std::stringstream ss;
  int64_t problem_size = 1;
  std::vector<TileAxis *> axes;
  analyzer_->ForEachAxisTopDown([this, &problem_size, &axes](TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr || axis->index != band_index_) {
      return;
    }
    problem_size *= axis->extent_val;
    (void)axes.emplace_back(axis);
  });
  auto coef = std::max<int64_t>(
    1, static_cast<int64_t>((problem_size / warp_sizes_) / SafeDivisor(num_sm_ * active_blocks_per_sm_.first)));
  ss << "Total reduce coef = " << coef;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  for (size_t i = axes.size() - 1; i > 0; --i) {
    auto axis = axes[i];
    axis->thread_constraints.item_process_ =
      std::max<int64_t>(min_elem_for_io_bound_, TilingAnalyzer::FindDivisibleTilingFactor(coef, axis->extent_val));
    CHECK(axis->thread_constraints.item_process_ != 0);
    coef = std::max<int64_t>(1, coef / SafeDivisor(axis->thread_constraints.item_process_));
    ss << "axis " << axis->index << "_" << axis->dim_axis
       << " set for-loop size = " << axis->thread_constraints.item_process_ << ", update coef = " << coef;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void GpuStrategy::BroadcastSpeedup() {
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "BroadcastSpeedup");
  std::stringstream ss;
  if (!analyzer_->scop_info_.user_config_.EnableStitchFusion()) {
    analyzer_->scop_info_.user_config_.SetEnableOneDimThread(true);
  }
  size_t depth = 0;
  auto problem_size = 1;
  analyzer_->ForEachAxisTopDown([this, &depth, &problem_size](TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr || axis->index != band_index_) {
      return;
    }
    ++depth;
    fused_size_ = axis->extent_val;
    problem_size *= axis->extent_val;
  });
  auto IncreaseForLoop = [this, &depth, &problem_size, &ss]() {
    TileAxis *first_axis = nullptr;
    TileAxis *last_axis = nullptr;
    auto parallel_size = GetProposalParallelSize(problem_size);
    auto vectorizaed_size = analyzer_->scop_info_.user_config_.GetVectorLength();
    analyzer_->ForEachAxisTopDown([this, &depth, &first_axis, &last_axis](TileAxis *axis) {
      if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr || axis->index != band_index_) {
        return;
      }
      axis->forbid_iso = true;
      if (!axis->HasAttr(AT_BROADCAST_INNERMOST_AXIS) && depth != 1 &&
          axis->extent_val > min_elem_for_io_bound_ * max_elem_per_thread_) {
        return;
      }
      if (first_axis == nullptr) {
        first_axis = axis;
      }
      last_axis = axis;
    });
    TileAxis *axis = nullptr;
    if (analyzer_->scop_info_.user_config_.GetEnableOneDimThread()) {
      axis = first_axis;
    } else {
      axis = last_axis;
    }

    ss << "\nProblem size = " << problem_size << " parallel size = " << parallel_size.first * parallel_size.second
       << " vec size " << vectorizaed_size;

    if (axis != nullptr && (!axis->HasAttr(AT_VECTORIZED) || !current_outer_bn_->enable_vectorization)) {
      auto min_aligned = analyzer_->FindDivisibleTilingFactor(min_elem_for_io_bound_, axis->extent_val);
      auto coef = current_outer_bn_->enable_vectorization ? 1 : double_;
      auto max_aligned = analyzer_->FindDivisibleTilingFactor(coef * min_elem_for_io_bound_, axis->extent_val);
      if (max_aligned > 1 &&
          problem_size >= parallel_size.first * parallel_size.second * vectorizaed_size * max_aligned) {
        axis->thread_constraints.item_process_ = max_aligned;
      } else if (min_aligned > 1 &&
                 problem_size >= parallel_size.first * parallel_size.second * vectorizaed_size * min_aligned) {
        axis->thread_constraints.item_process_ = min_aligned;
      } else if (problem_size >=
                 parallel_size.first * parallel_size.second * vectorizaed_size * min_elem_for_io_bound_) {
        axis->thread_constraints.item_process_ = min_elem_for_io_bound_;
      }
      ss << "\nBroadcast item process = " << axis->thread_constraints.item_process_;
    }
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  };
  // Only deal with broadcast + elemwise cases that all axes are fused into one.
  auto mod_axes = analyzer_->GetAxesContainsAttr(AT_MOD);
  if (depth != 1 || mod_axes.size() > 1U) {
    analyzer_->GetTileLogger().AppendLine(GPU_MAPPING,
                                          "Cannot deal with this broadcast, make all axes tile divisible to speedup.");
    IncreaseForLoop();
    return;
  }

  AnalyzeBroadcastIdx();

  if (mod_axes.empty() || broadcast_idx_.empty()) {
    IncreaseForLoop();
    GpuScalarBroadcastStrategy();
  } else {
    GpuVectorBroadcastStrategy();
  }
}

void GpuStrategy::AnalyzeBroadcastIdx() {
  for (const auto &attr : analyzer_->RootAxis()->attrs) {
    if (attr.attr_key.find(AT_BROADCAST) == std::string::npos) {
      continue;
    }
    auto op_types = common::Split(attr.attr_key, "_");
    for (const auto type : op_types) {
      if (type.find(AT_BROADCAST) == std::string::npos) {
        continue;
      }
      auto info = common::Split(type, "|");
      if (info.size() == 2U) {
        CHECK(!info[1].empty());
        (void)broadcast_idx_.insert(StrToDecimalInt(info[1]));
      }
    }
  }
  std::stringstream ss;
  ss << "Broadcast index = [";
  for (auto idx : broadcast_idx_) {
    ss << idx << ",";
  }
  ss << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GpuStrategy::GpuScalarBroadcastStrategy() {
  if (template_ != Template::CUSTOM_CONFIG) {
    template_ = Template::PURE_ELEM;  // change template to enable injective speed up
  }
  auto broadcast_axes = analyzer_->GetAxesContainsAttr(AT_BROADCAST);
  if (broadcast_axes.empty()) {
    return;
  }
  current_outer_bn_->use_shared_memory = false;
}

void GpuStrategy::MapBroadcastElem(TileAxis *axis, std::vector<int> original_shape) {
  // Mapping strategy specialized for broadcast + elementwise case
  auto broadcast_innermost = broadcast_idx_.find(original_shape.size() - 1) != broadcast_idx_.end();
  for (size_t i = 0; i < original_shape.size(); ++i) {
    if (original_shape[i] * possible_threads_ <= max_x_y_dim_thread_) {
      possible_threads_ *= original_shape[i];
    }
    auto rev_idx = (original_shape.size() - 1) - i;
    if (broadcast_idx_.find(rev_idx) == broadcast_idx_.end()) {
      total_injective_size_ *= original_shape[i];
      coalesced_size_ = coalesced_size_ == 0 ? original_shape[i] : coalesced_size_;
      if (broadcast_innermost) {
        auto prev_extent = axis->thread_constraints.map_extent_ > 0 ? axis->thread_constraints.map_extent_ : 1;
        auto thread_limit = max_x_y_dim_thread_ / SafeDivisor(prev_extent);
        auto coef = TilingAnalyzer::FindDivisibleTilingFactor(thread_limit, original_shape[i]);
        axis->thread_constraints.map_extent_ = prev_extent * coef;
        possible_threads_ = static_cast<int>(axis->thread_constraints.map_extent_);
      }
    } else if (broadcast_innermost) {
      auto prev_extent = axis->thread_constraints.map_extent_ > 0 ? axis->thread_constraints.map_extent_ : 1;
      axis->thread_constraints.map_extent_ =
        prev_extent * original_shape[i] <= max_x_y_dim_thread_ ? prev_extent * original_shape[i] : prev_extent;
      possible_threads_ = static_cast<int>(axis->thread_constraints.map_extent_);
    }
    coalesced_size_ = coalesced_size_ == 0 ? 1 : coalesced_size_;
  }
}

void GpuStrategy::GpuVectorBroadcastStrategy() {
  // Disable share and local promotion since isl cannot perfectly handle fusion cases.
  current_outer_bn_->use_shared_memory = false;
  current_outer_bn_->use_register_memory = false;
  auto interested_info = GetInterestedInfo(AT_MOD);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    std::stringstream ss;

    // Reconstruct original shape from fused axis
    std::vector<int> mod_values;
    for (const auto &attr : it.second) {
      CHECK(!attr.attr_value.empty());
      (void)mod_values.emplace_back(StrToDecimalInt(attr.attr_value));
    }
    std::sort(mod_values.begin(), mod_values.end());

    ss << "original shape before fused (in reversed order) :[";
    std::vector<int> original_shape;
    int prev_mod = 1;
    for (const auto m : mod_values) {
      CHECK_NE(prev_mod, 0);
      (void)original_shape.emplace_back(m / SafeDivisor(prev_mod));
      ss << original_shape.back() << ", ";
      prev_mod = m;
    }
    CHECK_NE(prev_mod, 0);
    (void)original_shape.emplace_back(fused_size_ / SafeDivisor(prev_mod));
    ss << original_shape.back() << "]";
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

    possible_threads_ = 1;
    coalesced_size_ = 0;
    total_injective_size_ = 1;
    MapBroadcastElem(axis, original_shape);

    int min_block = coalesced_size_ < warp_sizes_ ? 1024 : 512;
    if (coalesced_size_ >= warp_sizes_) {
      int elem_per_thread = 8;
      axis->thread_constraints.item_process_ = std::min(
        elem_per_thread,
        std::max<int>((((fused_size_ / SafeDivisor(possible_threads_)) / SafeDivisor(min_block) + 1) / binary_factor_) *
                        binary_factor_,
                      1));
      ss << "thread for-loop speedup = " << axis->thread_constraints.item_process_;
    } else if (total_injective_size_ > min_block) {
      while (possible_threads_ % warp_sizes_ != 0 && possible_threads_ < max_x_y_dim_thread_) {
        ++possible_threads_;
      }
      int elem_per_block = std::max<int>(16 / SafeDivisor(max_x_y_dim_thread_ / SafeDivisor(possible_threads_)), 1);
      auto proposal_blocks = std::max(
        min_block, std::max<int>((fused_size_ / SafeDivisor(possible_threads_)) / SafeDivisor(elem_per_block), 1));
      axis->block_constraints.map_extent_ = proposal_blocks;
      axis->thread_constraints.map_extent_ = possible_threads_;
      ss << "block for-loop speedup = " << elem_per_block;
    } else {
      ss << "default mapping.";
    }
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    ss << "possible_threads: " << possible_threads_ << ", coalesced_size: " << coalesced_size_
       << ", total_injective_size: " << total_injective_size_;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void CustomTilingStrategy::AddGpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key, false);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (auto attr : it.second) {
      ParseConstraintStr(attr.attr_key, attr.attr_value);
      ParseLevel();
      for (const auto &con : constraints_) {
        ApplyEachCustomConstraint(axis, con);
      }
    }
  }
}

void ConvStrategy::AddGpuConstraint() {
  if (!analyzer_->scop_info_.user_config_.GetEnableTensorCore() ||
      analyzer_->scop_info_.analysis_result_.GetIsGpuDmaAnalysed() ||
      !analyzer_->scop_info_.user_config_.GetEnableConvTensorCore()) {
    return;
  }

  Mma mma = analyzer_->scop_info_.analysis_result_.GetMmaMode();

  // Step 1. Collect M, H, W, N, K axis info.
  std::unique_ptr<MmaConv> shape = InitGemmShape(mma);
  if (shape == nullptr) {
    return;
  }

  if (!analyzer_->scop_info_.user_config_.EnableStitchFusion()) {
    analyzer_->scop_info_.user_config_.SetEnableOneDimThread(true);
  }
  MmaConv middle_band = {shape->m / SafeDivisor(mma.m), shape->h, shape->w, shape->n / SafeDivisor(mma.n),
                         shape->k / SafeDivisor(mma.k)};
  std::stringstream ss;
  ss << "[Conv] M = " << shape->m << " H = " << shape->h << " W = " << shape->w << " N = " << shape->n
     << " K = " << shape->k << ", middle band = [" << middle_band.m << ", " << middle_band.h << ", " << middle_band.w
     << middle_band.n << ", " << middle_band.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  // Step 2. Calculate macro M, H, W, N, K tile size.
  CalculateMacroMma(*shape, mma);

  // Step 3. Calculate possible number of warps.
  auto warp_sizes = CalculateNumOfWarps(mma);
  std::tie(w0_for_m_, w1_for_n_) = warp_sizes;
  middle_band.m /= w0_for_m_;
  middle_band.n /= w1_for_n_;
  std::string warp_cfg = std::to_string(w0_for_m_) + " " + std::to_string(w1_for_n_);
  analyzer_->scop_info_.user_config_.RecordReplaceConfig(WARP_COMPUTE, warp_cfg, MappingType::REPLACE_THREADS);
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  // Step 4. Set mapping and tiling config.
  SetFinalConfig(macro_mma_, mma);
}

std::pair<int64_t, int64_t> ConvStrategy::CalculateNumOfWarps(const Mma &mma) {
  int w0 = 1;
  int w1 = 1;
  // H and W do not participate in the calculation of the warp level
  int use_local_group = static_cast<int>((macro_mma_.m / SafeDivisor(mma.m)) * (macro_mma_.n / SafeDivisor(mma.n)));
  CHECK_GE(use_local_group, 1);
  if (use_local_group >= use_local_group_high_) {
    default_num_warps_ = num_warps_mid_;
  } else if (use_local_group > 1) {
    default_num_warps_ = num_warps_low_;
  }

  if ((macro_mma_.n / SafeDivisor(mma.n)) % binary_factor_ != 0) {
    default_num_warps_ = num_warps_low_;
  }

  if (macro_mma_.k == double_warp_size_ && macro_mma_.n >= quadruple_warp_size_) {
    default_num_warps_ = num_warps_high_;
  }

  std::tie(w0, w1) = GetDivisibleFactorForMN(macro_mma_.m, macro_mma_.n, default_num_warps_, mma);
  std::stringstream ss;
  ss << "[Conv] Try warp " << default_num_warps_ << " -> " << w0 << " * " << w1;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  return std::make_pair(w0, w1);
}

std::unique_ptr<MmaConv> ConvStrategy::InitGemmShape(const Mma &mma) {
  auto m_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_CONV, kDsami});
  auto h_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_CONV, kDsahi});
  auto w_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_CONV, kDsawi});
  auto n_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_CONV, kDsaoc});
  auto k_axes = analyzer_->GetAxesOfAttr(AttrInfo{AT_CONV, kDsaic});
  if (m_axes.size() != 1U || h_axes.size() != 1U || w_axes.size() != 1U || n_axes.size() != 1U || k_axes.size() != 1U) {
    return nullptr;
  }

  m_axis_ = m_axes[0];
  h_axis_ = h_axes[0];
  w_axis_ = w_axes[0];
  n_axis_ = n_axes[0];
  k_axis_ = k_axes[0];
  if (m_axis_->range_extent.as<IntImm>() == nullptr || h_axis_->range_extent.as<IntImm>() == nullptr ||
      w_axis_->range_extent.as<IntImm>() == nullptr || n_axis_->range_extent.as<IntImm>() == nullptr ||
      k_axis_->range_extent.as<IntImm>() == nullptr) {
    return nullptr;
  }
  auto shape_m = m_axis_->range_extent.as<IntImm>()->value;
  auto shape_h = h_axis_->range_extent.as<IntImm>()->value;
  auto shape_w = w_axis_->range_extent.as<IntImm>()->value;
  auto shape_n = n_axis_->range_extent.as<IntImm>()->value;
  auto shape_k = k_axis_->range_extent.as<IntImm>()->value;
  CHECK_EQ(shape_m % mma.m, 0) << "Shape m " << shape_m << " should be multiples of mma.m " << mma.m
                               << " to enable tensor core.";
  CHECK_EQ(shape_n % mma.n, 0) << "Shape n " << shape_n << " should be multiples of mma.n " << mma.n
                               << " to enable tensor core.";
  CHECK_EQ(shape_k % mma.k, 0) << "Shape k " << shape_k << " should be multiples of mma.k " << mma.k
                               << " to enable tensor core.";

  return std::make_unique<MmaConv>(MmaConv{shape_m, shape_h, shape_w, shape_n, shape_k});
}

void ConvStrategy::CalculateMacroMma(const MmaConv &shape, const Mma &mma) {
  std::stringstream ss;
  MmaConv macro_mma = {std::min<int>(macro_mma_.m, shape.m), std::min<int>(macro_mma_.h, shape.h),
                       std::min<int>(macro_mma_.w, shape.w), std::min<int>(macro_mma_.n, shape.n),
                       std::min<int>(macro_mma_.k, shape.k)};
  ss << "[Init macro mma]: [" << macro_mma.m << ", " << macro_mma.h << ", " << macro_mma.w << ", " << macro_mma.n
     << ", " << macro_mma.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  while (shape.m % SafeDivisor(macro_mma_.m) != 0 && macro_mma_.m / binary_factor_ >= mma.m) {
    macro_mma_.m /= binary_factor_;
  }

  // If n is bigger than 128 and is not multiple of 128, the case is not supported now
  if (shape.n % SafeDivisor(macro_mma_.n) != 0) {
    macro_mma_.n = shape.n;
  }

  while (shape.k % SafeDivisor(macro_mma_.k) != 0 && macro_mma_.k / binary_factor_ >= mma.k) {
    macro_mma_.k /= binary_factor_;
  }

  // Data volume in the M direction and data volume in the N direction should be close
  // split h and w direction, increase the data volume
  int temp_h = static_cast<int>(shape.h);
  int temp_w = static_cast<int>(shape.w);
  while (macro_mma_.m * macro_mma_.w * macro_mma_.h < quadruple_warp_size_) {
    if (temp_w % binary_factor_ == 0) {
      macro_mma_.w *= binary_factor_;
      temp_w /= binary_factor_;
    } else if (temp_h % binary_factor_ == 0) {
      macro_mma_.h *= binary_factor_;
      temp_h /= binary_factor_;
    } else {
      break;
    }
  }

  while ((shape.m / SafeDivisor(macro_mma_.m)) * (shape.h / SafeDivisor(macro_mma_.h)) *
             (shape.w / SafeDivisor(macro_mma_.w)) * (shape.n / SafeDivisor(macro_mma_.n)) <
           (min_blocks_ - warp_sizes_) &&
         (macro_mma_.m / SafeDivisor(macro_mma_.n)) * macro_mma_.h * macro_mma_.w > tensor_core_per_warp_) {
    // decrease h and increase the use of block
    if (macro_mma_.h % binary_factor_ == 0) {
      macro_mma_.h /= binary_factor_;
      continue;
    }

    // decrease w and increase the use of block
    if (macro_mma_.w % binary_factor_ == 0) {
      macro_mma_.w /= binary_factor_;
      continue;
    }

    if (macro_mma_.m / binary_factor_ >= mma.m) {
      macro_mma_.m /= binary_factor_;
    }
  }

  real_blocks_ = (shape.m / SafeDivisor(macro_mma_.m)) * (shape.h / SafeDivisor(macro_mma_.h)) *
                 (shape.w / SafeDivisor(macro_mma_.w)) * (shape.n / SafeDivisor(macro_mma_.n));
  if ((real_blocks_ > (min_blocks_ - warp_sizes_)) && (shape.k % SafeDivisor(macro_mma_.k * binary_factor_) == 0)) {
    macro_mma_.k *= binary_factor_;
  }
  ss << "[Final macro mma]: [" << macro_mma.m << ", " << macro_mma.h << ", " << macro_mma.w << ", " << macro_mma.n
     << ", " << macro_mma.k << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void ConvStrategy::SetFinalConfig(const MmaConv &macro_mma, const Mma &mma) {
  std::stringstream ss;
  m_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.m)), CACHE1);
  m_axis_->thread_constraints.map_min_ = w0_for_m_ * w1_for_n_;
  m_axis_->thread_constraints.map_extent_ = w0_for_m_ * w1_for_n_;
  m_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.m)), CACHE0);

  h_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.h)), CACHE1);
  h_axis_->thread_constraints.map_min_ = MIN_TILE;
  h_axis_->thread_constraints.map_extent_ = MIN_TILE;
  h_axis_->TileRestrainToSingleValue(CastIntToExpr(1), CACHE0);

  w_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.w)), CACHE1);
  w_axis_->thread_constraints.map_min_ = MIN_TILE;
  w_axis_->thread_constraints.map_extent_ = MIN_TILE;
  w_axis_->TileRestrainToSingleValue(CastIntToExpr(1), CACHE0);

  n_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.n)), CACHE1);
  n_axis_->thread_constraints.map_min_ = warp_sizes_;
  n_axis_->thread_constraints.map_extent_ = warp_sizes_;
  n_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.n)), CACHE0);

  k_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(macro_mma.k)), CACHE1);
  k_axis_->thread_constraints.map_min_ = MIN_TILE;
  k_axis_->thread_constraints.map_extent_ = MIN_TILE;
  k_axis_->TileRestrainToSingleValue(CastIntToExpr(static_cast<int>(mma.k)), CACHE0);
  ss << "[Final config] : L1(M, H, W, N, K) = " << macro_mma.m << ", " << macro_mma.h << ", " << macro_mma.w << ", "
     << macro_mma.n << ", " << macro_mma.k;
  ss << "; L0(M, H, W, N, K) = " << mma.m << ", " << 1 << ", " << 1 << ", " << mma.n << ", " << mma.k;
  ss << "; Thread(W0, W1, TX) = " << w0_for_m_ << ", " << w1_for_n_ << ", " << warp_sizes_;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

const std::pair<int64_t, int64_t> ConvStrategy::GetDivisibleFactorForMN(int64_t shape_m, int64_t shape_n,
                                                                        int64_t total_factor, const Mma &mma) {
  return GetTensorCoreDivisibleFactorForMN(shape_m, shape_n, total_factor, binary_factor_, mma);
}

void ShiftAxisStrategy::AddGpuConstraint() {
  TileEntirely();
  for (auto axis : shifted_axes_) {
    axis->block_constraints.map_extent_ = 1;
    axis->thread_constraints.map_extent_ = 1;
  }
}

void CsrStrategy::AddGpuConstraint() {
  if (!analyzer_->scop_info_.analysis_result_.GetCsr()) {
    return;
  }
  std::vector<TileAxis *> axes;
  int csr_axes_count = 0;
  analyzer_->ForEachAxisTopDown([&axes, &csr_axes_count, this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    if (analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(axis->range_extent)) {
      ++csr_axes_count;
    }
    axes.push_back(axis);
  });
  auto available_threads = total_available_thread_;
  int csr_thread_num = -1;
  std::unordered_map<int, int64_t> index_mapping;
  auto feat_len = analyzer_->scop_info_.analysis_result_.GetCsrFeatLen();
  if (feat_len > 1) {
    // CSR schedule with feature dimension (csr.values > 1d), axis has already been
    // fused to outer axis (static boundary), and inner axis (dynamic boundary).
    // Feature dimension will be mapped first
    CHECK(axes.size() == GPU_CSR_FUSION_AXES_SIZE);
    auto outer_axis = axes[OUTERMOST_AXIS];
    auto inner_axis = axes[OUTERMOST_AXIS + 1];
    bool use_reduce_lib = analyzer_->scop_info_.analysis_result_.GetUseGpuReduceLib();
    csr_thread_num = use_reduce_lib ? analyzer_->scop_info_.user_config_.GetCsrThreadNum() : GPU_CSR_NO_TILE;
    // For outer axis
    CHECK(!analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(outer_axis->range_extent));
    available_threads /= feat_len;
    int max_nodes_per_block = available_threads / csr_thread_num;
    while (max_nodes_per_block < 1 && (csr_thread_num >> 1) > reduce_length_limit_) {
      csr_thread_num >>= 1;
      max_nodes_per_block = available_threads / csr_thread_num;
    }
    auto nodes_per_block = std::min(max_nodes_per_block, GPU_CSR_BEST_NUM_NODES_PER_BLOCK);
    nodes_per_block = std::max(1, nodes_per_block);
    outer_axis->block_constraints.map_extent_ =
      std::ceil(static_cast<double>(outer_axis->extent_val) / feat_len / nodes_per_block);
    outer_axis->thread_constraints.map_extent_ = feat_len * nodes_per_block;
    // For inner axis
    CHECK(analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(inner_axis->range_extent));
    inner_axis->block_constraints.map_extent_ = GPU_CSR_NO_TILE;
    inner_axis->thread_constraints.map_extent_ = csr_thread_num;
    inner_axis->c1_constraints.tile_min_ = GPU_CSR_NO_TILE;
    inner_axis->c1_constraints.tile_extent_ = csr_thread_num;
    analyzer_->scop_info_.user_config_.SetCsrThreadNum(csr_thread_num);
    return;
  }
  std::sort(axes.begin(), axes.end(), [](TileAxis *a, TileAxis *b) {
    if (a->dim_axis == b->dim_axis) {
      return a->index < b->index;
    }
    return a->dim_axis > b->dim_axis;
  });
  for (size_t i = 0; i < axes.size(); ++i) {
    // CSR schedule without dimension (csr.values = 1d)
    auto axis = axes[i];
    if (analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(axis->range_extent)) {
      if (csr_thread_num != -1) {
        axis->block_constraints.map_extent_ = GPU_CSR_NO_TILE;
        axis->thread_constraints.map_extent_ = csr_thread_num;
        axis->c1_constraints.tile_extent_ = csr_thread_num;
      } else {
        int csr_avg_row = analyzer_->scop_info_.analysis_result_.GetCsrAvgRow();
        if (csr_avg_row <= 0) {
          csr_thread_num = analyzer_->scop_info_.user_config_.GetCsrThreadNum();
        } else {
          float warp_base_raw = std::log2(static_cast<float>(csr_avg_row) / WARP_SIZE);
          int warp_base;
          if (analyzer_->scop_info_.analysis_result_.GetOpTemplate() == Template::REDUCTION) {
            warp_base = std::clamp(static_cast<int>(std::floor(warp_base_raw)), 0, warp_factor_reduction_);
          } else {
            warp_base = std::clamp(static_cast<int>(std::ceil(warp_base_raw)), 0, warp_factor_elemwise_);
          }
          csr_thread_num = static_cast<int>(std::exp2(warp_base)) * WARP_SIZE;
        }
        csr_thread_num = std::min(static_cast<int64_t>(csr_thread_num), available_threads);
        axis->block_constraints.map_extent_ = GPU_CSR_NO_TILE;
        axis->thread_constraints.map_extent_ = csr_thread_num;
        axis->c1_constraints.tile_extent_ = csr_thread_num;
        analyzer_->scop_info_.user_config_.SetCsrThreadNum(csr_thread_num);
        available_threads /= SafeDivisor(csr_thread_num);
      }
    } else if (axis->dim_axis == 0) {
      axis->thread_constraints.map_extent_ = GPU_CSR_NO_TILE;
    } else {
      if (index_mapping.count(axis->dim_axis)) {
        auto prev_mapping = index_mapping[axis->dim_axis];
        if (axis->extent_val > prev_mapping) {
          auto additional_mapping = static_cast<int64_t>(
            std::ceil(static_cast<float>(axis->extent_val) / prev_mapping));
          additional_mapping = std::clamp<int64_t>(additional_mapping, 1, available_threads);
          available_threads /= SafeDivisor(additional_mapping);
          index_mapping[axis->dim_axis] = prev_mapping * additional_mapping;
        }
      } else {
        auto thread_num = std::min(axis->extent_val, available_threads);
        available_threads /= SafeDivisor(thread_num);
        index_mapping[axis->dim_axis] = thread_num;
      }
    }
  }
}

void CountStrategy::AddGpuConstraint() {
  std::unordered_set<TileAxis *> count_axes;
  for (int band_index = 0; band_index < static_cast<int>(analyzer_->RootAxis()->children.size()); ++band_index) {
    auto count_axes_vec = analyzer_->GetAxesOfAttr(AT_COUNT_AXIS, band_index);
    count_axes.insert(count_axes_vec.begin(), count_axes_vec.end());
  }
  analyzer_->ForEachAxisTopDown([this, count_axes](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    if (count_axes.count(axis)) {
      axis->thread_constraints.map_extent_ = 1;
      axis->block_constraints.map_extent_ = axis->extent_val;
    } else {
      axis->block_constraints.map_extent_ = 1;
    }
  });
}

void VectorizedStrategy::AddGpuConstraint() {
  auto vectorized_size = analyzer_->scop_info_.user_config_.GetVectorLength();
  if (!analyzer_->scop_info_.user_config_.GetEnableVectorization() || vectorized_size == 0) {
    return;
  }
  auto gpu_strategy = GpuStrategy(analyzer_);
  gpu_strategy.CountGlobalBufferSize();
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    auto curr_band = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(axis->index);
    if (!curr_band->enable_vectorization) {
      continue;
    }

    if (gpu_strategy.global_buf_size_ < gpu_strategy.min_buf_size_to_enable_vectorization_ &&
        analyzer_->RootAxis()->HasAttr(AT_HEAVY_ELTWISE)) {
      curr_band->enable_vectorization = false;
      continue;
    }

    auto parallel_size = gpu_strategy.GetProposalParallelSize(axis->extent_val);
    auto curr_template = curr_band->template_type;
    if (axis->extent_val % SafeDivisor(vectorized_size) != 0 ||
        (axis->extent_val < parallel_size.second * vectorized_size && curr_template != Template::PAD_OP)) {
      continue;
    }
    std::stringstream ss;
    ss << "Enable Vectorization for " << axis->index << "_" << axis->dim_axis;
    axis->thread_constraints.map_mod_ = vectorized_size;
    axis->c1_constraints.tile_mod_ = vectorized_size;
    axis->TileRestrainToSingleValue(CastIntToExpr(vectorized_size), TileLevel::CACHE0);
    if (axis->extent_val < parallel_size.first * parallel_size.second) {
      auto min_threads = std::min<int64_t>(total_available_thread_, axis->extent_val);
      axis->thread_constraints.map_extent_ = min_threads / SafeDivisor(vectorized_size);
    } else if (axis->extent_val > parallel_size.first * parallel_size.second * vectorized_size) {
      axis->thread_constraints.map_extent_ = total_available_thread_ / SafeDivisor(gpu_strategy.double_);
      axis->thread_constraints.item_process_ = 1;
    }
    ss << ", set thread extent to " << axis->thread_constraints.map_extent_;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void TensorOfTensorStrategy::AddGpuConstraint() {
  if (!analyzer_->scop_info_.analysis_result_.GetTensorOfTensorStmt().empty()) {
    // In this case, we have tensor of tensor with atomic operation.
    // Therefore, we disable shared-mem promotion to improve performance.
    for (int i = 0; i < analyzer_->scop_info_.analysis_result_.GetOuterBandNumber(); ++i) {
      auto band = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(i);
      band->use_shared_memory = false;
    }
  }
}

// No constraint found in cuda

void ModStrategy::AddGpuConstraint() {}

void ConflictTreeRangeStrategy::AddGpuConstraint() {}

void DmaAlignStrategy::AddGpuConstraint() {}

void PassDownAttrStrategy::AddGpuConstraint() {}

void DynamicShapeLimitStrategy::AddGpuConstraint() {}

void DynamicBoundStrategy::AddGpuConstraint() {}

void ModShiftAxisStrategy::AddGpuConstraint() {}

// end of null constraint
}  // namespace poly
}  // namespace ir
}  // namespace akg
