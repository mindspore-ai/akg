/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "tiling_strategy_manager.h"

#include <numeric>

#include "tiling_analyzer.h"
namespace akg {
namespace ir {
namespace poly {

void GpuDmaAnalysisStrategy::AddGpuConstraint() {
  analyzer_->ForEachAxisTopDown(
    [](TileAxis *axis) { axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), TileLevel::CACHE1); });
}

void CastStrategy::AddGpuConstraint() { MarkDataSize(); }

void GemmStrategy::AddGpuConstraint() {
  if (!analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
    return;
  }
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    axis->TileRestrainToSingleValue(CastIntToExpr(64), TileLevel::CACHE1);
    axis->TileRestrainToSingleValue(CastIntToExpr(16), TileLevel::CACHE0);
    for (const auto &attr : it.second) {
      if (attr.attr_value == "mi") {
        axis->thread_constraints.map_min_ = warp_sizes_;
        axis->thread_constraints.map_extent_ = warp_sizes_;
      } else if (attr.attr_value == "ni") {
        axis->thread_constraints.map_min_ = 4;
        axis->thread_constraints.map_extent_ = 4;
      }
    }
  }
}

void ReduceStrategy::AddGpuConstraint() {
  reduce_axes_ = analyzer_->GetAxesOfAttr(AT_REDUCE_AXIS);
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (!has_transpose_) {
      for (const auto &attr : axis->attrs) {
        if (attr.attr_key.find(AT_TRANSPOSE) != std::string::npos) {
          has_transpose_ = true;
          break;
        }
      }
    }

    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;
    if (axis->mc_sup) {
      injective_axes_.emplace_back(axis);
      return;
    }
    if (std::count(reduce_axes_.begin(), reduce_axes_.end(), axis)) {
      return;
    }
    reduce_axes_.emplace_back(axis);
  });
  all_reduce_ = reduce_axes_.size() == depth;
  if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    AkgReduceLibStrategyOnGpu();
  } else {
    SimpleStrategyOnGpu();
  }
}

void ReduceStrategy::SimpleStrategyOnGpu() {
  if (all_reduce_ || has_transpose_) {
    auto extent = all_reduce_ ? MIN_TILE : warp_sizes_;
    bool is_tuning = analyzer_->scop_info_.user_config_.GetIsTuning();
    for (auto axis : reduce_axes_) {
      axis->block_constraints.map_extent_ = MIN_TILE;
      axis->thread_constraints.map_extent_ = MIN_TILE;
      if (!is_tuning) {
        axis->TileRestrainToSingleValue(CastIntToExpr(extent), TileLevel::CACHE1);
      }
    }
  }
}

void ReduceStrategy::AkgReduceLibStrategyOnGpu() {
  // disable atomic-add for bitwise-reduction
  bool disable_atomic = !analyzer_->scop_info_.user_config_.GetEnableAtomicAdd();
  if (!disable_atomic) {
    for (auto it : analyzer_->scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
      if (analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_AND ||
          analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_OR) {
        disable_atomic = true;
        break;
      }
    }
  }
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

  bool square_thread = analyzer_->scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION;
  int64_t total_reduce_size = 1;
  int64_t total_injective_size = 1;
  int64_t injective_threads = 1;
  int64_t reduce_threads = 1;
  int64_t possible_reduce_blocks = 1;
  int64_t possible_injective_blocks = 1;

  if (!all_reduce_) {
    DealWith4DFusedReduce();
  }
  bool use_local = UseRegisterMem();

  for (auto axis : reduce_axes_) {
    CHECK(axis->range_extent.as<IntImm>());
    total_reduce_size *= axis->range_extent.as<IntImm>()->value;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_reduce_blocks *= axis->range_extent.as<IntImm>()->value;
    } else {
      possible_reduce_blocks *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      reduce_threads *= axis->thread_constraints.map_min_;
    }
  }
  for (auto axis : injective_axes_) {
    CHECK(axis->range_extent.as<IntImm>());
    total_injective_size *= axis->range_extent.as<IntImm>()->value;
    if (axis->block_constraints.map_extent_ == 0) {
      possible_injective_blocks *= axis->range_extent.as<IntImm>()->value;
    } else {
      possible_injective_blocks *= axis->block_constraints.map_extent_;
    }
    if (axis->thread_constraints.map_min_ == axis->thread_constraints.map_extent_ &&
        axis->thread_constraints.map_extent_ != 0) {
      injective_threads *= axis->thread_constraints.map_min_;
    }
  }
  bool is_special_4d = reduce_threads != 1 || injective_threads != 1;
  if (is_special_4d) {
    return;
  }

  int64_t min_blocks = square_thread ? 32 : 512;
  int64_t min_elem_per_thread = use_local ? 2 : 8;
  int64_t min_ty = 8;
  if (total_injective_size * total_reduce_size / min_blocks / max_x_y_dim_thread_ < min_elem_per_thread) {
    min_blocks = 32;
    min_ty = square_thread ? min_ty : 1;
  }

  std::pair<int64_t, int64_t> tx_range{1, max_x_y_dim_thread_};
  std::pair<int64_t, int64_t> ty_range{1, max_x_y_dim_thread_};
  auto AlignToPowerOfTwo = [](int64_t original_factor) -> int64_t {
    while ((original_factor) & (original_factor - 1)) {
      --original_factor;
    }
    return original_factor;
  };
  if (square_thread) {
    tx_range.first = AlignToPowerOfTwo(std::min(warp_sizes_, total_injective_size));
    ty_range.first = AlignToPowerOfTwo(std::min<int64_t>(min_ty, total_reduce_size));
    tx_range.second =
      AlignToPowerOfTwo(std::min<int64_t>(tx_range.second, ceil(static_cast<float>(tx_range.second) / ty_range.first)));
    tx_range.second = AlignToPowerOfTwo(std::min(tx_range.second, total_injective_size));
  } else {
    tx_range.first = AlignToPowerOfTwo(std::min(warp_sizes_, total_reduce_size));
    ty_range.first = AlignToPowerOfTwo(std::min<int64_t>(min_ty, total_injective_size));
    tx_range.second =
      AlignToPowerOfTwo(std::min<int64_t>(tx_range.second, ceil(static_cast<float>(tx_range.second) / ty_range.first)));
    tx_range.second = AlignToPowerOfTwo(std::min(tx_range.second, total_reduce_size));
  }
  ty_range.second =
    std::min(ty_range.second, static_cast<int64_t>(ceil(static_cast<float>(ty_range.second) / tx_range.first)));

  auto max_coef = std::max(ty_range.second / ty_range.first, tx_range.second / tx_range.first);
  if (square_thread) {
    int coef = 1;
    while (coef <= max_coef) {
      if (total_reduce_size % (ty_range.first * coef) == 0 ||
          (coef < max_coef / 2 && total_reduce_size % (ty_range.first * coef * 2) != 0)) {
        break;
      }
      coef *= 2;
    }
    reduce_threads = ty_range.first * coef;
    injective_threads = tx_range.second / coef;
  } else {
    int coef = 1;
    while (coef <= max_coef) {
      if (total_reduce_size % (tx_range.second / coef) == 0 ||
          (coef < max_coef / 2 && total_reduce_size % (tx_range.second / coef / 2) != 0)) {
        break;
      }
      coef *= 2;
    }
    reduce_threads = tx_range.second / coef;
    injective_threads = ty_range.first * coef;
  }
  for (auto axis : reduce_axes_) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_MOD) {
        continue;
      }
      CHECK_NE(attr.attr_value, "");
      auto mod_value = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10));
      axis->TileRestrainMod(CastInt64ToExpr(mod_value), TileLevel::CACHE1);
    }
    if (use_local) {
      auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
      while (tile_mod > reduce_threads && tile_mod % reduce_threads != 0) {
        --reduce_threads;
      }
    }
  }

  int possible_blocks =
    ceil(static_cast<float>(possible_injective_blocks * possible_reduce_blocks) / injective_threads / reduce_threads);
  int proposal = use_local ? 8 : 32;
  auto default_elem_per_thread = possible_reduce_blocks > 1
                                   ? std::max(std::min<int>(proposal, (possible_blocks / min_blocks + 1) / 2 * 2), 1)
                                 : IsHalfReduce() ? 64
                                                  : SpItemPerThread::FULL;

  auto original_ept = default_elem_per_thread;
  // try to increase thread loop (no more than twice as original)
  while (possible_blocks > default_elem_per_thread && possible_blocks % default_elem_per_thread != 0) {
    ++default_elem_per_thread;
  }
  if (original_ept * 2 < default_elem_per_thread) {
    default_elem_per_thread = original_ept;
  }
  // try to decrease thread loop (no less than half of original)
  while (possible_blocks > default_elem_per_thread && possible_blocks % default_elem_per_thread != 0) {
    --default_elem_per_thread;
  }
  if (default_elem_per_thread * 2 < original_ept) {
    default_elem_per_thread = original_ept;
  }
  std::stringstream ss;
  ss << "total_injective_size " << total_injective_size << " total_reduce_size " << total_reduce_size;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "injective_threads " << injective_threads << " reduce_threads " << reduce_threads;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "possible_blocks " << possible_blocks << " possible_injective_blocks " << possible_injective_blocks
     << " possible_reduce_blocks " << possible_reduce_blocks << " default_elem_per_thread " << default_elem_per_thread;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  ss << "tx:[" << tx_range.first << ", " << tx_range.second << "]; ty:[" << ty_range.first << ", " << ty_range.second
     << "]";
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  for (auto axis : injective_axes_) {
    axis->thread_constraints.map_min_ = injective_threads;
    axis->thread_constraints.map_extent_ = injective_threads;
    axis->thread_constraints.item_process_ = MIN_TILE;
  }
  for (auto axis : reduce_axes_) {
    axis->thread_constraints.map_extent_ = reduce_threads;
    axis->thread_constraints.item_process_ = default_elem_per_thread;
  }
}

bool ReduceStrategy::UseRegisterMem() {
  for (auto &it : analyzer_->buf_info_) {
    auto buf = it.second.get();
    CHECK(buf);
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
      if (it != axis->data_size.end() && *std::min_element(it->second.begin(), it->second.end()) == 2) {
        return true;
      }
    }
  }
  return false;
}

void ReduceStrategy::DealWith4DFusedReduce() {
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
      last_mod_value = static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10));
      ++num_mod_axis;
    }
    if (num_mod_axis < 1) {
      continue;
    }
    axis->TileRestrainToSingleValue(CastIntToExpr(last_mod_value), TileLevel::CACHE1);
    if (last_mod_value > max_x_y_dim_thread_) {
      LOG(WARNING) << "Cannot bind axis to " << last_mod_value << " threads, maximal thread number is "
                   << max_x_y_dim_thread_
                   << ". If fusing more than two axes together, footprint box calculated by isl may not be correct.";
      continue;
    }
    axis->thread_constraints.map_extent_ = last_mod_value;
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
    post_reduce_tensors.insert(tensor_name);
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
      axis->thread_constraints.item_process_ = SpItemPerThread::FULL;
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

void GpuStrategy::ApplyCustomConstraint() {
  auto ParseBindingConstraint = [](const std::string constraint, size_t max_size) {
    std::vector<std::string> sp = akg::common::Split(constraint, ",");
    std::vector<int64_t> ret;
    for (auto val : sp) {
      if (ret.size() == max_size) {
        break;
      }
      CHECK(!val.empty());
      ret.emplace_back(static_cast<int>(std::strtol(val.c_str(), nullptr, 10)));
    }
    return ret;
  };

  // init binding space through template-determined limit
  thread_binding_spaces_.clear();
  block_binding_spaces_.clear();
  for (size_t i = 0; i < thread_limit_.size(); ++i) {
    TileAxis::MappingConstraint elem;
    elem.map_extent_ = thread_limit_[i];
    thread_binding_spaces_.emplace_back(elem);
  }
  for (size_t i = 0; i < std::min(depth_, block_limit_.size()); ++i) {
    TileAxis::MappingConstraint elem;
    elem.map_extent_ = block_limit_[i];
    block_binding_spaces_.emplace_back(elem);
  }

  // add constraints to binding space according to custom tiling
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

  // apply custom constraint to corresponding axis and modify binding space according to tile range of axis
  size_t cur_depth = 0;
  analyzer_->ForEachAxisTopDown([this, &cur_depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    auto cons = axis->GetConstConstraint(CACHE1);
    auto range_extent = axis->GetConstExtent();
    int tile_min = cons.tile_min_.as<IntImm>()->value;
    int tile_extent = cons.tile_extent_.as<IntImm>()->value;
    auto idx = reverse_binding_ ? cur_depth : depth_ - 1 - cur_depth;

    auto thread_extent = tile_extent;
    if (idx < thread_binding_spaces_.size()) {
      thread_extent = std::min<int64_t>(thread_extent, thread_binding_spaces_[idx].map_extent_);
      thread_binding_spaces_[idx].map_extent_ = thread_extent;
    }

    auto block_extent = range_extent / tile_min;
    if (idx < block_binding_spaces_.size()) {
      block_extent = std::min<int64_t>(block_extent, block_binding_spaces_[idx].map_extent_);
      block_binding_spaces_[idx].map_extent_ = block_extent;
    }

    auto block_min = block_extent / std::max<int64_t>(1, thread_extent);
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

void GpuStrategy::AddGpuConstraint() {
  InitMappingLimit();
  if (!analyzer_->scop_info_.user_config_.GetIsTuning() &&
      (template_ == Template::BROADCAST_OP || template_ == Template::CUSTOM_CONFIG)) {
    BroadcastSpeedup();
  }
  BuildAxesQueue();
  if (analyzer_->scop_info_.user_config_.GetIsTuning()) {
    ApplyCustomConstraint();
    for (size_t i = 0; i < max_dim_; ++i) {
      TileAxis::MappingConstraint pad;
      if (i >= thread_binding_spaces_.size()) {
        thread_binding_spaces_.emplace_back(pad);
      }
      if (i >= block_binding_spaces_.size()) {
        block_binding_spaces_.emplace_back(pad);
      }
    }
    return;
  }
  InnerThreadOuterBlock();
  if (template_ == Template::PURE_ELEM) {
    InjectiveSpeedup();
  }
  SetMappingConfig();
  if (template_ != Template::MATMUL || !analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
    analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
      if (axis == analyzer_->RootAxis()) {
        return;
      }
      axis->TileRestrainToSingleValue(axis->c1_constraints.tile_min_, TileLevel::CACHE0);
    });
  }
  // TODO: This is a very naive strategy to avoid cuda launch out of resources
  //       and we should fix this in register memory promotion pass.
  if (template_ != Template::REDUCTION && template_ != Template::ALL_REDUCE) {
    auto local_buf_count = GetLocalAllocBufCount();
    auto thread_size = std::accumulate(thread_cfg_.begin(), thread_cfg_.end(), 1, std::multiplies<int>());
    if (local_buf_count >= 4 || local_buf_count * 4 * thread_size >= 65536) {
      analyzer_->scop_info_.user_config_.SetUseRegisterMemory(false);
    }
  }
}

void GpuStrategy::InitMappingLimit() {
  max_x_y_dim_thread_ = analyzer_->scop_info_.user_config_.GetMaxElemPerThread();
  DetermineTemplate();
  std::stringstream ss;
  reverse_binding_ = analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib() &&
                     analyzer_->scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION;

  if (template_ == Template::CUSTOM_CONFIG) {
    auto thread_config = analyzer_->scop_info_.user_config_.GetThreadConfig();
    for (size_t i = 0; i < thread_config->bound; ++i) {
      auto idx = reverse_binding_ ? thread_config->bound - 1 - i : i;
      if (idx >= depth_) {
        continue;
      }
      thread_limit_.emplace_back(thread_config->GetAt(idx).second);
    }
  } else if (template_ == Template::REDUCTION || template_ == Template::BITWISE_REDUCTION) {
    thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_};
  } else if (template_ == Template::ALL_REDUCE) {
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
      thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_};
    } else {
      thread_limit_ = {1};
    }
  } else if (template_ == Template::TRANSPOSE_OP) {
    analyzer_->scop_info_.user_config_.SetUseSharedMemory(false);
    analyzer_->scop_info_.user_config_.SetUseRegisterMemory(false);
    analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
      axis->thread_constraints.item_process_ = std::max(axis->thread_constraints.item_process_, min_elem_for_io_bound_);
    });
    thread_limit_ = {max_x_y_dim_thread_, max_x_y_dim_thread_};
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

  if (template_ != Template::CUSTOM_CONFIG && !analyzer_->scop_info_.user_config_.GetEnableTensorCore()) {
    AdjustThreadMappingLimit();
  }

  if (template_ == Template::CUSTOM_CONFIG) {
    auto block_config = analyzer_->scop_info_.user_config_.GetBlockConfig();
    for (int i = 0; i < static_cast<int>(block_config->bound) - 1; ++i) {
      if (i >= static_cast<int>(depth_)) {
        break;
      }
      block_limit_.emplace_back(block_config->GetAt(i).second);
    }
  } else if (template_ <= Template::REDUCTION) {
    block_limit_ = {max_x_dim_block_, max_y_z_dim_block_, max_y_z_dim_block_};
  } else if (template_ == Template::ALL_REDUCE && !analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    block_limit_ = {1};
  } else {
    block_limit_ = {max_x_dim_block_, max_y_z_dim_block_, max_y_z_dim_block_};
  }

  std::vector<std::string> elem_cfg = common::Split(analyzer_->scop_info_.user_config_.GetElemPerThread(), " ");
  for (size_t i = 0; i < max_dim_; ++i) {
    if (i < elem_cfg.size() && !elem_cfg[i].empty()) {
      elem_per_thread_[i] = static_cast<int64_t>(std::strtol(elem_cfg[i].c_str(), nullptr, 10));
    }
  }
}

void GpuStrategy::BuildAxesQueue() {
  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    const auto r = axis->range_extent.as<IntImm>();
    if (r && r->value > 0) {
      this->pending_axes_.push_front(std::make_pair(axis, r->value));
    }

    // init map extent to shape if they are not modified by other constraints
    axis->block_constraints.map_extent_ =
      axis->block_constraints.map_extent_ == 0 ? r->value : axis->block_constraints.map_extent_;
    axis->thread_constraints.map_extent_ =
      axis->thread_constraints.map_extent_ == 0 ? r->value : axis->thread_constraints.map_extent_;
  });
}

void GpuStrategy::InnerThreadOuterBlock() {
  if (pending_axes_.empty()) {
    return;
  }
  std::stringstream ss;
  int64_t activated_blocks = 1;
  int64_t activated_threads = 1;

  auto thread_dim = std::min(thread_limit_.size(), max_dim_);
  auto block_dim = std::min(block_limit_.size(), max_dim_);

  // tile from inner to outer and map to thread
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "-----Map to thread-----");
  ss << "[Thread Limit]: ";
  for (auto l : thread_limit_) {
    ss << l << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  size_t ori_size = pending_axes_.size();
  size_t inner_dim = 0;
  for (size_t i = 0; i < ori_size; ++i) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    int64_t rest_threads = std::min(max_x_y_dim_thread_ / activated_threads, thread_limit_[thread_cfg_.size()]);
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape
       << ", rest_threads = " << rest_threads;
    auto SkipMapping = [this, &axis, &shape, &ss, &inner_dim, &thread_dim]() {
      axis->thread_constraints.map_extent_ = 1;
      auto tile = inner_dim < thread_dim ? elem_per_thread_[inner_dim] : 1;
      tile = tile == SpItemPerThread::AUTO   ? std::min(axis->thread_constraints.item_process_, max_elem_per_thread_)
             : tile == SpItemPerThread::FULL ? std::min(shape, max_elem_per_thread_)
                                             : 1;
      auto tile_min = axis->c1_constraints.tile_min_.as<IntImm>()->value;
      auto tile_extent = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
      if (tile_min == tile_extent && tile_extent != MIN_TILE) {
        ss << "tile extent is already determined = " << tile_extent;
        analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
        tile = tile_min;
      } else {
        if (axis->block_constraints.map_extent_ > 1) {
          tile =
            std::max(tile, std::max<int64_t>(ceil(static_cast<float>(shape) / axis->block_constraints.map_extent_), 1));
        } else {
          tile = std::min(tile, shape);
        }
      }
      axis->TileRestrainLower(tile, TileLevel::CACHE1);
      ss << ", tile = " << tile;
      if (axis->block_constraints.map_extent_ > 1) {
        pending_axes_.push_back(std::make_pair(axis, std::max<int64_t>(ceil(static_cast<float>(shape) / tile), 1)));
        ss << ", map to block.";
      }
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    };

    if (template_ != Template::CUSTOM_CONFIG) {
      rest_threads = std::min(rest_threads, axis->thread_constraints.map_extent_);
    }

    if (thread_cfg_.size() >= thread_dim || inner_dim >= max_dim_) {
      ss << ", no thread/dim rests";
      SkipMapping();
      continue;
    }
    if (rest_threads <= 1) {
      if (axis->mc_sup ||
          (template_ == Template::REDUCTION && analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib())) {
        thread_cfg_.emplace_back(1);
      }
      SkipMapping();
      continue;
    }

    auto item = elem_per_thread_[inner_dim] == SpItemPerThread::AUTO ? axis->thread_constraints.item_process_
                                                                     : elem_per_thread_[inner_dim];
    item = std::min(item, max_elem_per_thread_);
    auto use = GetThreadSize(rest_threads, inner_dim, shape, item);
    if (axis->forbid_iso && shape % use != 0) {
      ss << ", forbid iso and adjust use: original = " << use;
      use = analyzer_->FindDivisibleTilingFactor(use, shape);
    }
    activated_threads *= use;
    ss << ", use = " << use << ", activated threads = " << activated_threads;
    thread_cfg_.emplace_back(use);
    axis->thread_constraints.map_extent_ = use;
    auto tile = TileAfterThreadMapping(axis, inner_dim, use, item);
    pending_axes_.push_back(std::make_pair(axis, std::max<int64_t>(ceil(static_cast<float>(shape) / tile), 1)));
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    ++inner_dim;
  }

  std::vector<size_t> indexing;
  for (size_t i = 0; i < block_dim; ++i) {
    block_cfg_.emplace_back(1);
  }
  // If all axes for block mapping are element-wise, we can map them in any order
  // so we need a greedy algorithm to map the most blocks;
  // otherwise, we can simply map from outer to inner in sequence.
  if (template_ == Template::PURE_ELEM) {
    std::map<int64_t, std::vector<size_t>, std::greater<int64_t>> sorted_by_gcd;
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      auto block_limit = i == 0 ? max_x_dim_block_ : max_y_z_dim_block_;
      auto use = (block_limit > 0 && pending_axes_[i].second > 0)
                   ? TilingAnalyzer::FindDivisibleTilingFactor(block_limit, pending_axes_[i].second)
                   : 1;
      if (sorted_by_gcd.find(use) == sorted_by_gcd.end()) {
        sorted_by_gcd[use] = {i};
      } else {
        sorted_by_gcd[use].emplace_back(i);
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
        indexing.emplace_back(i);
      }
    }
  } else {
    for (size_t i = pending_axes_.size() - 1; i >= ori_size; --i) {
      if (pending_axes_[i].second <= 1 && indexing.size() == block_limit_.size()) {
        continue;
      }
      indexing.emplace_back(i);
    }
  }

  // map outer band to block according to predefined indice
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "-----Map to block-----");
  ss << "[Block Limit]: ";
  for (auto l : block_limit_) {
    ss << l << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  for (const auto &i : indexing) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    auto idx = indexing.size() - 1 - (pending_axes_.size() - 1 - i);
    idx = reverse_binding_ ? std::min(indexing.size(), block_limit_.size()) - 1 - idx : idx;
    auto rest_blocks = idx < block_limit_.size() ? std::min(block_limit_[idx], axis->block_constraints.map_extent_) : 1;
    ss << "axis " << axis->index << "_" << axis->dim_axis << " shape = " << shape << ", block_idx = " << idx
       << ", rest blocks = " << rest_blocks;
    if (block_count_ >= static_cast<int>(block_dim)) {
      ss << "-> No mapping.";
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
      continue;
    }
    auto use = (rest_blocks > 0 && shape > 1) ? (TilingAnalyzer::FindDivisibleTilingFactor(rest_blocks, shape) > 1
                                                   ? TilingAnalyzer::FindDivisibleTilingFactor(rest_blocks, shape)
                                                   : rest_blocks)
                                              : 1;
    activated_blocks *= use;
    ss << ", use = " << use << ", activated blocks = " << activated_blocks;
    block_cfg_[pending_axes_.size() - 1 - i] = use;
    axis->block_constraints.map_extent_ = use;
    if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib() || axis->mc_sup) {
      ++block_count_;
    }
    CHECK(axis->range_extent.as<IntImm>());
    auto extent = axis->range_extent.as<IntImm>()->value;
    axis->TileRestrainUpper(std::max<int64_t>(ceil(static_cast<float>(extent) / use), 1), TileLevel::CACHE1);
    ss << ", tile range = [" << axis->c1_constraints.tile_min_ << ", " << axis->c1_constraints.tile_extent_ << "]";
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void GpuStrategy::SetMappingConfig() {
  std::stringstream ss;
  ss << "Use template " << template_map_[template_];
  if (template_ == Template::REDUCTION) {
    ss << "(" << analyzer_->scop_info_.analysis_result_.GetReduceDirection() << ")";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  // we need bind one axis at least
  if (thread_cfg_.empty()) {
    thread_cfg_.emplace_back(1);
  }
  if (block_cfg_.empty()) {
    block_cfg_.emplace_back(1);
  }
  if (block_count_ == 0) {
    block_count_ = 1;
  }

  std::string block_str = "";
  std::string thread_str = "";
  if (reverse_binding_) {
    for (int i = 0; i < static_cast<int>(block_cfg_.size()); ++i) {
      if (i >= block_count_) {
        continue;
      }
      block_str += (std::to_string(block_cfg_[i]) + " ");
    }
    // pad binding to at least two dim to bind reduce axis at thread y
    for (size_t i = thread_cfg_.size(); i < 2; ++i) {
      thread_cfg_.emplace_back(1);
    }

    for (int i = thread_cfg_.size() - 1; i >= 0; --i) {
      thread_str += (std::to_string(thread_cfg_[i]) + " ");
    }
  } else {
    // pad binding to at least two dim to bind reduce axis at block y
    for (size_t i = block_cfg_.size(); i < 2; ++i) {
      block_cfg_.emplace_back(1);
    }
    for (int i = block_cfg_.size() - 1; i >= 0; --i) {
      if (i >= block_count_) {
        continue;
      }
      block_str += (std::to_string(block_cfg_[i]) + " ");
    }
    for (const auto &size : thread_cfg_) {
      thread_str += (std::to_string(size) + " ");
    }
  }

  ss << "Block config = " << block_str;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  ss << "Thread config = " << thread_str;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  ss << "Tile = ";
  analyzer_->ForEachAxisTopDown([this, &ss](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ss << axis->c1_constraints.tile_extent_ << ",";
  });
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  if (template_ == Template::CUSTOM_CONFIG) {
    return;
  }
  analyzer_->scop_info_.user_config_.SetBlockConfig(block_str);
  analyzer_->scop_info_.user_config_.SetThreadConfig(thread_str);
}

int64_t GpuStrategy::GetThreadSize(const int64_t rest_threads, size_t inner_dim, const int64_t shape,
                                   const int64_t item) {
  // TODO: how to set best thread size according to current rest_thread and shape
  //       is not sure and profiling test is needed.

  // Current experience is that let mapped threads divisible by warp_size to increase performance.
  int64_t thread_extent = item == SpItemPerThread::FULL ? rest_threads : ceil(static_cast<float>(shape) / item);
  thread_extent = std::min(thread_extent, shape);
  if (thread_extent > rest_threads || template_ == Template::CUSTOM_CONFIG) {
    return rest_threads;
  }
  auto proposal = inner_dim == 0 ? ((thread_extent - 1 + warp_sizes_) / warp_sizes_ * warp_sizes_) : thread_extent;
  return std::min(rest_threads, proposal);
}

int64_t GpuStrategy::TileAfterThreadMapping(TileAxis *axis, size_t inner_dim, int64_t thread_size, const int64_t item) {
  std::stringstream ss;
  auto shape = axis->range_extent.as<IntImm>()->value;
  auto tile_min = axis->c1_constraints.tile_min_.as<IntImm>()->value;
  auto tile_mod = axis->c1_constraints.tile_mod_.as<IntImm>()->value;
  auto tile_extent = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
  if (tile_min == tile_extent && tile_extent != MIN_TILE) {
    ss << "tile extent is already determined = " << tile_extent;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    return tile_extent;
  }

  auto tile = item == SpItemPerThread::FULL ? std::min(tile_extent, thread_size * max_elem_per_thread_)
                                            : std::min(tile_extent, thread_size * item);

  if (analyzer_->scop_info_.user_config_.GetEnableAkgReduceLib()) {
    if (tile < tile_mod) {
      // tile axis with mod value
      // e.g. tile cc0 with 128 in the following code
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (tile_mod % tile != 0 && tile > thread_size) {
        --tile;
      }
    } else {
      // tile axis with div value
      // e.g. tile cc0 with 512 in the following code (which equals tile floordiv(cc0, 256) with 2)
      // for cc0 in 1024:
      //    A[0, floormod(cc0, 256)] = B[floordiv(cc0, 256), floormod(cc0, 256)]
      while (shape % tile != 0 && tile > thread_size) {
        --tile;
      }
    }
  }

  bool partial_block = (shape / tile <= 1 && shape > tile);
  if (partial_block) {
    tile = shape;
  }

  if (template_ == Template::CUSTOM_CONFIG) {
    if (!analyzer_->scop_info_.user_config_.GetEnableAtomicAdd() &&
        (axis->HasAttr(AT_REDUCE_AXIS) || axis->mc_sup == 0)) {
      tile = shape;
      ss << "tile = shape to disable atomic add, ";
    } else if (tile < thread_size) {
      tile = thread_size;
      ss << "tile = thread size, ";
    } else {
      auto block_dim = reverse_binding_ ? block_limit_.size() - 1 - inner_dim : inner_dim;
      int64_t least_blocks;
      if (block_dim >= 0 && block_dim < block_limit_.size()) {
        least_blocks = block_limit_[block_dim];
      } else {
        least_blocks = std::accumulate(block_limit_.begin(), block_limit_.end(), 1, std::multiplies<int>());
      }
      auto max_tile = shape / least_blocks;
      if (shape % thread_size == 0) {
        tile = analyzer_->FindDivisibleTilingFactor(max_tile, shape);  // ensure no if condition in thread for-loop
      } else {
        tile =
          analyzer_->FindDivisibleTilingFactor(max_tile, thread_size);  // ensure thread for-loop bound has no min/max
      }
      ss << "reduce tile size to enable at least " << least_blocks << " blocks, ";
    }
  }

  ss << "axis " << axis->index << "_" << axis->dim_axis << " elem_per_thread = " << item << ", tile = " << tile;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  axis->TileRestrainLower(CastInt64ToExpr(tile), TileLevel::CACHE1);
  return tile;
}

void GpuStrategy::DetermineTemplate() {
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
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
    return;
  }

  for (auto it : analyzer_->scop_info_.analysis_result_.GetReduceTensorInfoMap()) {
    if (analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_AND ||
        analyzer_->scop_info_.analysis_result_.GetReduceOpType(it.first) == AKG_REDUCE_OR) {
      template_ = Template::BITWISE_REDUCTION;
      return;
    }
  }

  if (!analyzer_->GetAxesOfAttr(AT_GEMM).empty()) {
    template_ = Template::MATMUL;
    return;
  }

  auto reduce_axes_ = analyzer_->GetAxesOfAttr(AT_REDUCE_AXIS);

  if (reduce_axes_.empty()) {
    bool has_transpose = false;
    analyzer_->ForEachAxisTopDown([this, &has_transpose](TileAxis *axis) {
      if (has_transpose) {
        return;
      }
      has_transpose =
        axis->HasAttr(AT_TRANSPOSE, true) || (axis->HasAttr(AT_BROADCAST, true) && axis->HasAttr(AT_TRANSFORM, true));
    });
    bool is_pure_elem =
      (analyzer_->GetAxesContainsAttr(AT_BROADCAST).empty() && analyzer_->GetAxesContainsAttr(AT_TRANSFORM).empty());
    template_ = has_transpose ? Template::TRANSPOSE_OP : is_pure_elem ? Template::PURE_ELEM : Template::BROADCAST_OP;
    return;
  }

  template_ = reduce_axes_.size() == depth ? Template::ALL_REDUCE : Template::REDUCTION;
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
    map_mins.emplace_back(axis->thread_constraints.map_min_);
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
      int64_t res = floor(static_cast<float>(thread_limit_[j]) / map_mins[i]);
      thread_limit_[j] = res;
    }
  }
  ss << "Adjust thread limit by axes' mapping mins = ";
  for (auto tl : thread_limit_) {
    ss << tl << ", ";
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
}

void GpuStrategy::InjectiveSpeedup() {
  // not need speedup if thread_cfg_ or block_cfg_ is empty
  if (thread_cfg_.size() == 0 || block_cfg_.size() == 0) {
    return;
  }
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "InjectiveSpeedup");
  std::stringstream ss;
  std::vector<TileAxis *> injective_axes;
  analyzer_->ForEachAxisTopDown([this, &injective_axes](TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr) {
      return;
    }
    injective_axes.emplace_back(axis);
  });

  auto WriteConfigBack = [this, &injective_axes, &ss]() {
    for (size_t i = 0; i < injective_axes.size(); ++i) {
      ss << "replace block " << block_cfg_[i] << " with " << injective_axes[i]->block_constraints.map_extent_
         << " replace thread " << thread_cfg_[injective_axes.size() - 1 - i] << " with "
         << injective_axes[i]->thread_constraints.map_extent_;
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
      block_cfg_[i] = injective_axes[i]->block_constraints.map_extent_;
      thread_cfg_[injective_axes.size() - 1 - i] = injective_axes[i]->thread_constraints.map_extent_;
    }
  };

  // Step 1. Reduce code complexity by aligning thread size to shape
  auto total_threads = std::accumulate(thread_cfg_.begin(), thread_cfg_.end(), 1, std::multiplies<int>());
  for (size_t i = 0; i < injective_axes.size(); ++i) {
    auto axis = injective_axes[i];
    auto shape = axis->range_extent.as<IntImm>()->value;
    auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
    auto thread_size = axis->thread_constraints.map_extent_;
    if (shape % thread_size == 0) {
      continue;
    }

    auto lower = thread_size;
    while (shape % lower != 0) {
      --lower;
    }
    bool is_efficient = lower * 2 > thread_size || total_threads / thread_size * lower * 2 >= max_x_y_dim_thread_;
    if (is_efficient) {
      ss << "align thread from " << thread_size << " to " << lower << " according to shape " << shape;
      analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
      axis->thread_constraints.map_extent_ = lower;
      total_threads = total_threads / thread_size * lower;

      auto thread_loop = std::max<int>(1, tile_size / thread_size);
      tile_size = std::min(shape, thread_loop * lower);
      axis->TileRestrainToSingleValue(tile_size, TileLevel::CACHE1);

      axis->block_constraints.map_extent_ = shape / tile_size;
    }
  }
  WriteConfigBack();

  // Step 2. Adjust the ratio of thread for-loop, thread size and block size.
  auto coaleasced_size = injective_axes.back()->thread_constraints.map_extent_;
  auto proposal_blocks = coaleasced_size >= warp_sizes_ ? 256 : 512;
  auto proposal_threads = (coaleasced_size >= warp_sizes_ && injective_axes.size() > 1U) ? 128
                          : coaleasced_size < max_x_y_dim_thread_                        ? 512
                                                                                         : max_x_y_dim_thread_;
  auto total_blocks = std::accumulate(block_cfg_.begin(), block_cfg_.end(), 1, std::multiplies<int>());
  auto proposal_elem_per_thread = coaleasced_size < warp_sizes_        ? 1
                                  : total_blocks < proposal_blocks * 8 ? min_elem_for_io_bound_
                                                                       : 8;
  auto shrinked_threads = total_threads / proposal_threads;
  auto shrinked_blocks = total_blocks / proposal_blocks;

  auto thread_to_block = shrinked_threads > 0 && total_blocks < proposal_blocks;
  auto block_to_elem = proposal_elem_per_thread > 0 && shrinked_blocks > 0;
  auto thread_to_elem = proposal_elem_per_thread > 0 && !block_to_elem && shrinked_threads > 0 &&
                        total_blocks * shrinked_threads > proposal_blocks * proposal_elem_per_thread;
  ss << "coaleasced_size = " << coaleasced_size << " total_blocks = " << total_blocks
     << " total_threads = " << total_threads << " proposal_blocks = " << proposal_blocks
     << " proposal_threads = " << proposal_threads << " proposal_elem_per_thread = " << proposal_elem_per_thread
     << " shrinked_threads = " << shrinked_threads << " shrinked_blocks = " << shrinked_blocks;
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  if (thread_to_block) {
    for (auto axis : injective_axes) {
      if (shrinked_threads <= 0) {
        break;
      }
      auto thread_size = axis->thread_constraints.map_extent_;
      auto block_size = axis->block_constraints.map_extent_;
      auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
      auto coef = analyzer_->FindDivisibleTilingFactor(shrinked_threads, thread_size);
      shrinked_threads /= coef;
      axis->thread_constraints.map_extent_ = thread_size / coef;
      axis->block_constraints.map_extent_ = block_size * coef;
      axis->TileRestrainToSingleValue(tile_size / coef, TileLevel::CACHE1);
      ss << "axis " << axis->dim_axis << " before shrink " << thread_size << " shrink size " << coef;
    }
  }

  if (block_to_elem || thread_to_elem) {
    for (auto axis : injective_axes) {
      auto shrink_limit = block_to_elem ? shrinked_blocks : shrinked_threads;
      if (shrink_limit <= 0) {
        break;
      }
      auto tile_size = axis->c1_constraints.tile_extent_.as<IntImm>()->value;
      auto before_shrink = block_to_elem ? axis->block_constraints.map_extent_ : axis->thread_constraints.map_extent_;
      auto coef =
        std::min<int64_t>(proposal_elem_per_thread, analyzer_->FindDivisibleTilingFactor(shrink_limit, before_shrink));
      auto aligned_coef = coef;
      while (shrink_limit % aligned_coef != 0) {
        --aligned_coef;
      }
      if (aligned_coef > coef / 2) {
        coef = aligned_coef;
      }
      if (block_to_elem) {
        shrinked_blocks /= coef;
        axis->block_constraints.map_extent_ = before_shrink / coef;
      } else {
        shrinked_threads /= coef;
        axis->thread_constraints.map_extent_ = before_shrink / coef;
      }
      ss << "axis " << axis->dim_axis << " before shrink " << before_shrink << " shrink size " << coef;
      axis->TileRestrainToSingleValue(tile_size * coef, TileLevel::CACHE1);
    }
  }
  analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

  WriteConfigBack();
}

void GpuStrategy::BroadcastSpeedup() {
  analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "BroadcastSpeedup");
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr) {
      return;
    }
    ++depth;
    fused_size_ = axis->range_extent.as<IntImm>()->value;
  });
  // Only deal with broadcast + elemwise cases that all axes are fused into one.
  auto mod_axes = analyzer_->GetAxesContainsAttr(AT_MOD);
  if (depth != 1 || mod_axes.size() > 1U) {
    analyzer_->GetTileLogger().AppendLine(GPU_MAPPING, "Cannot deal with this broadcast, make all axes tile divisible to speedup.");
    analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
      if (axis == analyzer_->RootAxis() || axis->range_extent.as<IntImm>() == nullptr) {
        return;
      }
      axis->forbid_iso = true;
    });
    return;
  }

  AnalyzeBroadcastIdx();

  if (mod_axes.empty() || broadcast_idx_.empty()) {
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
        broadcast_idx_.insert(static_cast<int>(std::strtol(info[1].c_str(), nullptr, 10)));
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
  analyzer_->scop_info_.user_config_.SetUseSharedMemory(false);
}

void GpuStrategy::GpuVectorBroadcastStrategy() {
  // Disable share and local promotion since isl cannot perfectly handle fusion cases.
  analyzer_->scop_info_.user_config_.SetUseSharedMemory(false);
  analyzer_->scop_info_.user_config_.SetUseRegisterMemory(false);
  auto interested_info = GetInterestedInfo(AT_MOD);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    std::stringstream ss;

    // Reconstruct original shape from fused axis
    std::vector<int> mod_values;
    for (const auto &attr : it.second) {
      CHECK(!attr.attr_value.empty());
      mod_values.emplace_back(static_cast<int>(std::strtol(attr.attr_value.c_str(), nullptr, 10)));
    }
    std::sort(mod_values.begin(), mod_values.end());

    ss << "original shape before fused (in reversed order) :[";
    std::vector<int> original_shape;
    int prev_mod = 1;
    for (const auto m : mod_values) {
      CHECK_NE(prev_mod, 0);
      original_shape.emplace_back(m / prev_mod);
      ss << original_shape.back() << ", ";
      prev_mod = m;
    }
    CHECK_NE(prev_mod, 0);
    original_shape.emplace_back(fused_size_ / prev_mod);
    ss << original_shape.back() << "]";
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);

    // Mapping strategy specialized for broadcast + elementwise case
    int possible_threads = 1;
    int coalesced_size = 0;
    int total_injective_size = 1;
    auto broadcast_innermost = broadcast_idx_.find(original_shape.size() - 1) != broadcast_idx_.end();
    for (size_t i = 0; i < original_shape.size(); ++i) {
      if (original_shape[i] * possible_threads <= max_x_y_dim_thread_) {
        possible_threads *= original_shape[i];
      }
      auto rev_idx = original_shape.size() - 1 - i;
      if (broadcast_idx_.find(rev_idx) == broadcast_idx_.end()) {
        total_injective_size *= original_shape[i];
        coalesced_size = coalesced_size == 0 ? original_shape[i] : coalesced_size;
        if (broadcast_innermost) {
          auto prev_extent = axis->thread_constraints.map_extent_ > 0 ? axis->thread_constraints.map_extent_ : 1;
          auto thread_limit = max_x_y_dim_thread_ / prev_extent;
          auto coef = analyzer_->FindDivisibleTilingFactor(thread_limit, original_shape[i]);
          axis->thread_constraints.map_extent_ = prev_extent * coef;
          possible_threads = axis->thread_constraints.map_extent_;
        }
      } else if (broadcast_innermost) {
        auto prev_extent = axis->thread_constraints.map_extent_ > 0 ? axis->thread_constraints.map_extent_ : 1;
        axis->thread_constraints.map_extent_ =
          prev_extent * original_shape[i] <= max_x_y_dim_thread_ ? prev_extent * original_shape[i] : prev_extent;
        possible_threads = axis->thread_constraints.map_extent_;
      }
      coalesced_size = coalesced_size == 0 ? 1 : coalesced_size;
    }

    int elem_per_thread = 8;
    int min_block = coalesced_size < warp_sizes_ ? 1024 : 512;
    if (coalesced_size >= warp_sizes_) {
      axis->thread_constraints.item_process_ =
        std::min(elem_per_thread, std::max<int>((fused_size_ / possible_threads / min_block + 1) / 2 * 2, 1));
      ss << "thread for-loop speedup = " << axis->thread_constraints.item_process_;
    } else if (total_injective_size > min_block) {
      while (possible_threads % warp_sizes_ != 0 && possible_threads < max_x_y_dim_thread_) {
        ++possible_threads;
      }
      int elem_per_block = std::max<int>(16 / (max_x_y_dim_thread_ / possible_threads), 1);
      auto proposal_blocks = std::max(min_block, std::max<int>(fused_size_ / possible_threads / elem_per_block, 1));
      axis->block_constraints.map_extent_ = proposal_blocks;
      axis->thread_constraints.map_extent_ = possible_threads;
      ss << "block for-loop speedup = " << elem_per_block;
    } else {
      ss << "default mapping.";
    }
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
    ss << "possible_threads: " << possible_threads << ", coalesced_size: " << coalesced_size
       << ", total_injective_size: " << total_injective_size;
    analyzer_->GetTileLogger().AppendLog(GPU_MAPPING, ss);
  }
}

void CustomTilingStrategy::AddGpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key, false);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (auto attr : it.second) {
      std::vector<std::string> modes = akg::common::Split(attr.attr_key, ":");
      CHECK_EQ(modes.size(), 2U);
      std::string constraint_str = attr.attr_value;
      if (constraint_str.find("->") != std::string::npos) {
        std::vector<std::string> res = akg::common::Split(constraint_str, "->");
        constraint_str = res[1];
      }
      std::vector<std::string> constraints = akg::common::Split(constraint_str, "_");
      CHECK_GE(constraints.size(), 1U);
      std::vector<std::string> level = akg::common::Split(constraints[0], ":");
      CHECK(level.size() == 2U && level[0] == "LEVEL");
      CHECK(level[1] == "C1" || level[1] == "C0");
      TileLevel lv = level[1] == "C1" ? CACHE1 : CACHE0;
      constraints.erase(constraints.begin());
      for (const auto &con : constraints) {
        std::vector<std::string> items = akg::common::Split(con, ":");
        CHECK_EQ(items.size(), 2U);
        CHECK_NE(items[0], "");
        CHECK_NE(items[1], "");
        if (items[0] == "MIN") {
          if (items[1] == "MIN") {
            if (lv == CACHE1) {
              axis->c1_constraints.tile_extent_ = axis->c1_constraints.tile_min_;
            } else if (lv == CACHE0) {
              axis->c0_constraints.tile_extent_ = axis->c0_constraints.tile_min_;
            }
          } else {
            if (lv == CACHE1) {
              axis->c1_constraints.tile_min_ = CastToExpr(items[1]);
            } else if (lv == CACHE0) {
              axis->c0_constraints.tile_min_ = CastToExpr(items[1]);
            }
          }
        } else if (items[0] == "FACTOR") {
          axis->TileRestrainToSingleValue(CastToExpr(items[1]), lv);
        } else if (items[0] == "FORBIDISO") {
          axis->forbid_iso = true;
        } else if (items[0] == "MAX") {
          if (items[1] == "FULL") {
            axis->TileRestrainEntire(lv);
          } else {
            if (lv == CACHE1) {
              axis->c1_constraints.tile_extent_ = CastToExpr(items[1]);
            } else if (lv == CACHE0) {
              axis->c0_constraints.tile_extent_ = CastToExpr(items[1]);
            }
          }
        } else if (items[0] == AT_MOD) {
          axis->TileRestrainMod(CastToExpr(items[1]), lv);
        }
      }
    }
  }
}

// No constraint found in cuda

void ModStrategy::AddGpuConstraint() {}

void ConflictTreeRangeStrategy::AddGpuConstraint() {}

void VectorizedStrategy::AddGpuConstraint() {}

void DmaAlignStrategy::AddGpuConstraint() {}

void TensorOfTensorStrategy::AddGpuConstraint() {}

void PassDownAttrStrategy::AddGpuConstraint() {}

void DynamicShapeLimitStrategy::AddGpuConstraint() {}

void DynamicBoundStrategy::AddGpuConstraint() {}

void ShiftAxisStrategy::AddGpuConstraint() {}

void ModShiftAxisStrategy::AddGpuConstraint() {}

void ConvStrategy::AddGpuConstraint() {}

// end of null constraint

}  // namespace poly
}  // namespace ir
}  // namespace akg
