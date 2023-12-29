/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

void CustomTilingStrategy::AddNpuConstraint() {
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

void ConflictTreeRangeStrategy::AddNpuConstraint() {
  auto ApplyConflictStrategy = [](TileAxis *axis) {
    int64_t const_extent = axis->GetConstExtent();
    if (const_extent == -1) {
      return;
    }
    // When axis has conflict ranges, it is likely a padded axis;
    // When padded axis has MOD attr, it is likely a transformed axis;
    // It is not safe to apply min tile(1) to padded-and-transformed axis
    // as poly may generate wrong index.
    if (!axis->HasAttr(AT_MOD)) {
      axis->InsertC1CandFactor(CastIntToExpr(MIN_TILE));
    }
    if (axis->HasAttr(AT_MODSHIFT)) {
      const_extent = (const_extent - axis->range_min);
      axis->RemoveAttr(AT_MODSHIFT);
    }
    if (axis->HasAttr(AT_SHIFT)) {
      axis->RemoveAttr(AT_SHIFT);
    }
    axis->range_min = MIN_TILE;
    axis->InsertC1CandFactor(CastInt64ToExpr(const_extent));
    axis->c1_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
    axis->c1_constraints.tile_extent_ = CastInt64ToExpr(const_extent);
    axis->c0_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
    axis->c0_constraints.tile_extent_ = CastInt64ToExpr(const_extent);
  };
  auto CheckRange = [&ApplyConflictStrategy](TileAxis *axis) {
    std::unordered_set<int64_t> offset;
    std::unordered_set<int64_t> extent;
    int64_t min_off = -1;
    for (const auto &r : axis->tree_ranges) {
      const auto int_range = r.second.as<IntImm>();
      if (int_range == nullptr) {
        return;
      }
      if (r.first != 0) {
        offset.insert(r.first);
        if (min_off == -1) {
          min_off = r.first;
        } else if (r.first < min_off) {
          min_off = r.first;
        }
      }
      if (int_range->value != 0) {
        extent.insert(int_range->value - r.first);
      }
    }
    for (auto o : offset) {
      if (o % min_off != 0) {
        ApplyConflictStrategy(axis);
        return;
      }
    }
    if (extent.size() >= 2U) {
      ApplyConflictStrategy(axis);
    }
  };
  analyzer_->ForEachAxisTopDown(CheckRange);
}

void ModStrategy::AddNpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (const auto &attr : it.second) {
      CHECK_NE(attr.attr_value, "");
      auto mod_value = StrToDecimalInt(attr.attr_value);
      axis->TileRestrainMod(mod_value, CACHE1);
    }
  }
}

void CastStrategy::AddNpuConstraint() { MarkDataSize(); }

void ReduceStrategy::AddNpuConstraint() {
  if (analyzer_->scop_info_.user_config_.GetIsDynamic() || analyzer_->is_retry_) {
    return;
  }
  for (auto axis : analyzer_->GetAxesOfAttr(AT_REDUCE_DST_LAST)) {
    auto min_size_to_align = GetMaxAlignBytes(axis->data_size);
    if (axis->range_extent.as<IntImm>() == nullptr || axis->range_extent.as<IntImm>()->value < min_size_to_align) {
      continue;
    }
    axis->TileRestrainLower(CastInt64ToExpr(min_size_to_align), TileLevel::CACHE1);
    axis->forbid_iso = true;
  }
}

void VectorizedStrategy::AddNpuConstraint() {
  if (analyzer_->op_type_ != TileOpType::VECTOR_OP) {
    return;
  }
  for (auto axis : analyzer_->GetAxesOfAttr(AT_VECTORIZED)) {
    if (axis->HasAttr(AT_DYNAMIC_BOUND) || axis->range_extent.as<IntImm>() == nullptr) {
      continue;
    }
    int64_t min_byte = -1;
    for (const auto &it : axis->data_size) {
      if (it.second.empty()) {
        continue;
      }
      int min_elem = *min_element(it.second.begin(), it.second.end());
      if (min_byte == -1 || min_byte > min_elem) {
        min_byte = min_elem;
      }
    }
    min_byte = min_byte <= 0 ? 1 : min_byte;
    auto proposal = CanonicalSimplify(CastIntToExpr(VECTORIZE_BYTE / min_byte));
    axis->TileRestrainMod(proposal, TileLevel::CACHE1);
    if (!analyzer_->is_retry_) {
      axis->TileRestrainLower(proposal, TileLevel::CACHE1);
    }
  }
}

void DmaAlignStrategy::AddNpuConstraint() {
  if (analyzer_->scop_info_.user_config_.GetIsDynamic() || analyzer_->is_retry_) {
    return;
  }
  for (auto axis : analyzer_->GetAxesContainsAttr(AT_ALIGN)) {
    for (const auto &attr : axis->attrs) {
      if ((attr.attr_key.find(AT_ALIGN) == std::string::npos) || (attr.attr_key.find(AT_DMA) == std::string::npos)) {
        continue;
      }
      auto align_size = GetMaxAlignBytes(axis->data_size);
      if (axis->GetConstExtent() > align_size) {
        axis->TileRestrainLower(CastInt64ToExpr(align_size), CACHE1);
      }
    }
  }

  for (auto axis : analyzer_->GetAxesOfAttr(AT_BROADCAST_INNERMOST_AXIS)) {
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_BROADCAST_INNERMOST_AXIS) {
        continue;
      }
      auto align_size = GetMaxAlignBytes(axis->data_size);
      auto it = axis->data_size.find(attr.attr_value);
      if (it != axis->data_size.end()) {
        int min_byte = -1;
        int min_elem = *min_element(it->second.begin(), it->second.end());
        if (min_byte == -1 || min_byte > min_elem) {
          min_byte = min_elem;
        }
        align_size = GetAlignBytes(min_byte);
      }
      if (axis->GetConstExtent() > align_size) {
        axis->TileRestrainLower(CastInt64ToExpr(align_size), CACHE1);
      }
    }
  }
}

void TensorOfTensorStrategy::AddNpuConstraint() {
  for (auto axis : analyzer_->GetAxesOfAttr(AT_TOT)) {
    if (!axis->HasAttr("ALIGN:DMA")) continue;
    axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE1);
  }
}

void PassDownAttrStrategy::AddNpuConstraint() {
  for (auto axis : analyzer_->GetAxesOfAttr(AttrInfo{"ATTR", "pass_down"})) {
    axis->TileRestrainEntire(CACHE1);
  }
}

void DynamicShapeLimitStrategy::AddNpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (const auto &attr : it.second) {
      CHECK_NE(attr.attr_value, "");
      axis->dyn_shape_limit = StrToDecimalInt(attr.attr_value);
    }
  }
}

void DynamicBoundStrategy::AddNpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (const auto &attr : it.second) {
      CHECK_NE(attr.attr_value, "");
      auto bound = StrToDecimalInt(attr.attr_value);
      axis->TileRestrainMod(bound, CACHE1);
      axis->forbid_iso = true;
    }
  }
}

void ShiftAxisStrategy::AddNpuConstraint() { TileEntirely(); }

void ModShiftAxisStrategy::AddNpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    int64_t const_extent = axis->GetConstExtent();
    if (const_extent == -1) {
      continue;
    }
    for (const auto &attr : it.second) {
      axis->forbid_iso = true;
      auto imm_min = axis->GetConstConstraint(CACHE1).tile_min_.as<IntImm>()->value;
      if (imm_min > const_extent) {
        CHECK_NE(attr.attr_value, "");
        auto share_time = StrToDecimalInt(attr.attr_value);
        axis->TileRestrainToSingleValue(const_extent * (share_time + 1), CACHE1);
      } else {
        auto ForbidOthersIso = [](TileAxis *a) { a->forbid_iso = true; };
        analyzer_->ForEachAxisTopDown(ForbidOthersIso);
      }
      break;
    }
  }
}

void ConvStrategy::AddNpuConstraint() {
  conv_info_ = analyzer_->scop_info_.mmu_info_.GetConvInfoForTiling();
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (const auto &attr : it.second) {
      axis->axis_type_ = attr.attr_value;
      if (attr.attr_value == kDsaN || attr.attr_value == kDsaC1InOut) {
        axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE1);
        axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE0);
      } else if (attr.attr_value == kDsaH) {
        RestrainH(axis);
      } else if (attr.attr_value == kDsaW) {
        if (analyzer_->scop_info_.mmu_info_.IsConvBackpropFilter()) {
          axis->TileRestrainEntire(CACHE1);
        } else {
          RestrainW(axis);
        }
      } else if (attr.attr_value.find(kDsaC0) != std::string::npos || attr.attr_value == kDsakh ||
                 attr.attr_value == kDsakw) {
        axis->TileRestrainEntire(CACHE1);
      } else if (attr.attr_value == kDsaC1In && analyzer_->scop_info_.user_config_.GetIsDynamic()) {
        // dynamic case
        axis->TileRestrainEntire(CACHE1);
      }
    }
  }
}

void ConvStrategy::RestrainH(TileAxis *axis) {
  CHECK(conv_info_.find(ATTR_CONV_FEATURE_H) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_PAD_TOP) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_STRIDE_H) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_DILATION_H) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_KERNEL_H) != conv_info_.end());
  Expr h = conv_info_[ATTR_CONV_FEATURE_H];
  Expr p_top = conv_info_[ATTR_CONV_PAD_TOP];
  Expr s_h = conv_info_[ATTR_CONV_STRIDE_H];
  Expr d_h = conv_info_[ATTR_CONV_DILATION_H];
  Expr k_h = conv_info_[ATTR_CONV_KERNEL_H];
  CHECK(h.defined() && p_top.defined() && s_h.defined() && d_h.defined() && k_h.defined()) << "Conv attr not defined.";
  Expr k_h_d = (k_h - 1) * d_h + 1;
  int tile_out_h = MIN_TILE + 1;
  while (arith_ana_.CanProve(
    ((air::ir::FloorDiv::make((axis->range_extent + tile_out_h - 1), CastIntToExpr(tile_out_h)) - 1) * tile_out_h - 1) *
          s_h +
        k_h_d >
      h + p_top &&
    tile_out_h <= axis->range_extent)) {
    tile_out_h += 1;
  }
  axis->TileRestrainLower(CastIntToExpr(tile_out_h), TileLevel::CACHE1);
}

void ConvStrategy::RestrainW(TileAxis *axis) {
  CHECK(conv_info_.find(ATTR_CONV_FEATURE_W) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_PAD_LEFT) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_STRIDE_W) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_DILATION_W) != conv_info_.end());
  CHECK(conv_info_.find(ATTR_CONV_KERNEL_W) != conv_info_.end());
  Expr w = conv_info_[ATTR_CONV_FEATURE_W];
  Expr p_left = conv_info_[ATTR_CONV_PAD_LEFT];
  Expr s_w = conv_info_[ATTR_CONV_STRIDE_W];
  Expr d_w = conv_info_[ATTR_CONV_DILATION_W];
  Expr k_w = conv_info_[ATTR_CONV_KERNEL_W];
  CHECK(w.defined() && p_left.defined() && s_w.defined() && d_w.defined() && k_w.defined()) << "Conv attr not defined.";
  Expr k_w_d = (k_w - 1) * d_w + 1;
  int tile_out_w = 1;
  while (arith_ana_.CanProve(
    ((air::ir::FloorDiv::make((axis->range_extent + tile_out_w - 1), CastIntToExpr(tile_out_w)) - 1) * tile_out_w - 1) *
          s_w +
        k_w_d >
      w + p_left &&
    tile_out_w <= axis->range_extent)) {
    tile_out_w += 1;
  }
  axis->TileRestrainLower(CastIntToExpr(tile_out_w), TileLevel::CACHE1);
}

void GemmStrategy::AddNpuConstraint() {
  auto interested_info = GetInterestedInfo(interested_attr_key);
  for (auto it : interested_info) {
    TileAxis *axis = it.first;
    for (const auto &attr : it.second) {
      axis->axis_type_ = attr.attr_value;
      if (attr.attr_value == kDsami || attr.attr_value == kDsani || attr.attr_value == kDsaki) {
        axis->TileRestrainMod(CastIntToExpr(MMA_UNIT), CACHE1);
        axis->TileRestrainMod(CastIntToExpr(MMA_UNIT), CACHE0);
        axis->TileRestrainToSingleValue(CastIntToExpr(MMA_UNIT), CACHE1);
        axis->TileRestrainToSingleValue(CastIntToExpr(MMA_UNIT), CACHE0);
      } else if (attr.attr_value == kDsabo || attr.attr_value == kDsabi) {
        axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE1);
        axis->TileRestrainToSingleValue(CastIntToExpr(MIN_TILE), CACHE0);
      } else if (attr.attr_value == kDsamo || attr.attr_value == kDsano|| attr.attr_value == kDsako) {
        axis->forbid_iso = true;
      }
    }
  }
}

// Adjust max core for element-wise and inner-most reduction operations to balance core number and granularity.
int MulticoreStrategy::GetProposalCoreNum() {
  int max_core = GetCoreNumConf();
  int problem_size = 1;

  for (auto axis : this->cand_.GetTileAxis()) {
    if (axis->range_extent.as<IntImm>() == nullptr) {
      return 0;
    }

    if ((axis->HasAttr(AT_TRANSFORM)) || (axis->HasAttr(AT_TRANSPOSE)) ||
        (axis->HasAttr(AT_REDUCE_AXIS) && !axis->HasAttr(AT_REDUCE_SRC_LAST))) {
      return max_core;
    }

    problem_size *= axis->range_extent.as<IntImm>()->value;
  }

  if (problem_size < max_core * MIN_CORE_GRANULARITY * MAX_REPEAT) {
    max_core = static_cast<int>(problem_size / DESIRE_CORE_GRANULARITY);
    if (max_core > 2 && max_core % 2 != 0) {
      max_core--;
    }
  }
  return max_core;
}

std::pair<int, int> MulticoreStrategy::GetProposalRangeForFullMulticore(TileAxis *multicore_axis) {
  int max_core = GetProposalCoreNum();
  int used_core = 1;
  std::pair<int, int> proposal_range = std::make_pair(cand_.GetMinFactorForMinDataGranularity(multicore_axis), -1);
  auto this_level_core = std::max(static_cast<int>(max_core / used_core), 1);
  std::stringstream ss;
  if (multicore_axis->range_extent.as<IntImm>() == nullptr || this_level_core <= 1) {
    return proposal_range;
  }
  auto shape = multicore_axis->range_extent.as<IntImm>()->value;
  bool is_last_level = false;
  for (auto other_axis : this->cand_.GetTileAxis()) {
    if (other_axis == multicore_axis) break;
    if (other_axis->index != multicore_axis->index || other_axis->HasAttr(AT_REDUCE_AXIS)) continue;
    int64_t c1_val = TileVarId::UNDEFINE;
    std::tie(c1_val, std::ignore) = cand_.GetConstTileVal(other_axis);
    if (c1_val == TileVarId::VAR) return proposal_range;
    if (c1_val == TileVarId::UNDEFINE) {
      CHECK(other_axis->c1_constraints.tile_min_.as<IntImm>())
        << "Static shape " << shape << " should have const tile min, while got "
        << other_axis->c1_constraints.tile_min_;
      c1_val = other_axis->c1_constraints.tile_min_.as<IntImm>()->value;
    }
    auto block_extent = std::max(static_cast<int>(other_axis->range_extent.as<IntImm>()->value / c1_val), 1);
    ss << "range " << multicore_axis->range_extent << " l1 tile " << c1_val << " -> block extent " << block_extent
       << " this level " << this_level_core;
    logger_.AppendLog(DO_TILING, ss);
    ss.str("");
    if (block_extent > this_level_core) {
      int factor = (block_extent + this_level_core - 1) / this_level_core;
      this_level_core = (block_extent + factor - 1) / factor;
      is_last_level = true;
    } else if (block_extent * 2 > this_level_core) {
      this_level_core = block_extent;
      is_last_level = true;
    } else {
      this_level_core = block_extent;
    }
    if (is_last_level) break;
    used_core *= this_level_core;
    this_level_core = std::max(static_cast<int>(max_core / used_core), 1);
    ss << "use core " << used_core << " this level " << this_level_core;
    logger_.AppendLog(DO_TILING, ss);
    ss.str("");
  }
  proposal_range.second = std::max(static_cast<int>(shape / this_level_core), 1);
  ss << " proposal range (" << proposal_range.first << ", " << proposal_range.second << ")";
  logger_.AppendLog(DO_TILING, ss);
  return proposal_range;
}

int64_t MulticoreStrategy::AdjustTilingAccordingToMulticoreConstraint(TileAxis *multicore_axis, int64_t tiling_factor) {
  CHECK_GT(tiling_factor, 0) << "tiling factor cant be zero or negative";
  auto proposal_range = GetProposalRangeForFullMulticore(multicore_axis);
  auto min_factor_for_enough_data = proposal_range.first;
  auto max_factor_for_full_cores = proposal_range.second;
  auto origin_factor = tiling_factor;
  std::stringstream ss;

  if ((!multicore_axis->mc_sup) || (multicore_axis->HasAttr(AT_REDUCE_AXIS) || (max_factor_for_full_cores <= 0))) {
    logger_.AppendLine(DO_TILING, "This axis is not suitable for multicore, return.");
    return origin_factor;
  }
  if (tiling_factor < cand_.GetMinFactorToEnableMulticore(multicore_axis)) {
    logger_.AppendLine(DO_TILING, "Inner-most tile size is smaller than 32 bytes, multicore is disable, return.");
    return origin_factor;
  }
  if ((tiling_factor <= min_factor_for_enough_data) ||
      (min_factor_for_enough_data >= GetCoreNumConf() * max_factor_for_full_cores)) {
    logger_.AppendLine(DO_TILING, "Cannot increase degree of parallelism by adjusting current tiling factor, return.");
    return origin_factor;
  }

  auto CheckConstConstraint = [this, &ss](Expr constraint) {
    if (constraint.as<IntImm>() == nullptr) {
      ss << "Static shape should have const constraint, while got " << constraint;
      logger_.LogFatalAndSaveLog(ss.str());
    }
  };
  CheckConstConstraint(multicore_axis->range_extent);
  CheckConstConstraint(multicore_axis->c1_constraints.tile_min_);
  CheckConstConstraint(multicore_axis->c1_constraints.tile_mod_);

  auto pending_blocks = cand_.GetMaximalPendingBlocks(multicore_axis);
  if (tiling_factor < max_factor_for_full_cores) {
    auto end = static_cast<int>(sqrt(max_factor_for_full_cores));
    while (max_factor_for_full_cores % tiling_factor != 0 && tiling_factor > end) {
      --tiling_factor;
    }
  } else if (max_factor_for_full_cores >= min_factor_for_enough_data) {
    tiling_factor = max_factor_for_full_cores;
  } else if (max_factor_for_full_cores < min_factor_for_enough_data) {
    // In this case, simply adjusting tiling factor to max_factor_for_full_core may lead to insufficient data
    // in each core while adjusting tiling factor to min_factor_for_enough_date may lead to fewer parallel cores.
    // Since pending blocks can compensate data in each core, we make decision upon on its value.
    tiling_factor = pending_blocks >= static_cast<int>(min_factor_for_enough_data / max_factor_for_full_cores)
                      ? max_factor_for_full_cores
                      : min_factor_for_enough_data;
  }

  auto shape = multicore_axis->range_extent.as<IntImm>()->value;
  bool efficient = (shape % tiling_factor == 0) >= (shape % origin_factor == 0);
  auto multicore_shrink_limit = 2;
  auto reduced_mem = std::max(origin_factor - tiling_factor, min_factor_for_enough_data - tiling_factor);
  if ((static_cast<int>(origin_factor / tiling_factor) > multicore_shrink_limit) && reduced_mem > pending_blocks) {
    ss << "If axis adjust to " << tiling_factor << ", " << reduced_mem << " memory is reduced;"
       << " while maximal pending blocks is only " << pending_blocks << ", adjust may not be efficient.";
    logger_.AppendLog(DO_TILING, ss);
    efficient = false;
  }
  bool valid = tiling_factor >= multicore_axis->c1_constraints.tile_min_.as<IntImm>()->value;
  if (tiling_factor >= multicore_axis->c1_constraints.tile_mod_.as<IntImm>()->value) {
    valid = valid && tiling_factor % multicore_axis->c1_constraints.tile_mod_.as<IntImm>()->value == 0;
  } else {
    auto weak_constraint = multicore_axis->c1_constraints.tile_mod_.as<IntImm>()->value % tiling_factor == 0;
    valid = valid && multicore_axis->HasAttr(AT_VECTORIZED) && weak_constraint;
  }
  ss << "--> Adjust tiling factor " << origin_factor << " to " << tiling_factor << " if valid(" << valid
     << ") and efficient(" << efficient << ") according to proposal range (" << min_factor_for_enough_data << ", "
     << max_factor_for_full_cores << ")";
  logger_.AppendLog(DO_TILING, ss);
  return (valid && efficient) ? tiling_factor : origin_factor;
}

void TilingPriorityScorer::SetPriorityByScoring() {
  std::stringstream ss;
  for (int band_idx = 0; band_idx < static_cast<int>(analyzer_.RootAxis()->children.size()); ++band_idx) {
    std::map<double, std::vector<TileAxis *>> priority_map;
    std::vector<TileAxis *> tile_axes = GetBandTileAxes(band_idx);

    auto norm_range = static_cast<int>(tile_axes.size());
    auto dd_scores = MinMaxScaler(ComputeTileDependency(tile_axes), norm_range);
    auto pl_scores = MinMaxScaler(ComputeParallelism(tile_axes), norm_range);
    auto vec_scores = MinMaxScaler(ComputeVectorization(tile_axes), norm_range);

    bool has_custom_priority = false;
    int default_priority = -1;
    for (int i = 0; i < static_cast<int>(tile_axes.size()); ++i) {
      auto axis = tile_axes[i];

      if (axis->priority != default_priority) {
        has_custom_priority = true;
        break;
      }

      ss << "Axis " << axis->index << " , " << axis->dim_axis << ": ";
      auto total_score = (weight_.tile_dependency * dd_scores[i] + weight_.parallelism * pl_scores[i] +
                          weight_.vectorization * vec_scores[i]) /
                         weight_.Sum();
      ss << "score = (tile dependency) " << weight_.tile_dependency << "*" << dd_scores[i] << " + (parallelism) "
         << weight_.parallelism << " * " << pl_scores[i] << " + (vectorization) " << weight_.vectorization << " * "
         << vec_scores[i] << " / " << weight_.Sum() << " = " << total_score;
      logger_.AppendLog(DO_TILING, ss);

      if (priority_map.find(total_score) == priority_map.end()) {
        priority_map[total_score] = {axis};
      } else {
        priority_map[total_score].emplace_back(axis);
      }
    }

    if (has_custom_priority) {
      continue;
    }

    int priority = static_cast<int>(tile_axes.size()) - 1;
    for (auto it : priority_map) {
      for (auto a : it.second) {
        a->priority = priority;
        priority -= 1;
      }
    }
  }
}

std::vector<double> TilingPriorityScorer::ComputeTileDependency(std::vector<TileAxis *> tile_axes) {
  std::vector<double> scores;
  scores.reserve(tile_axes.size());
  for (auto axis : tile_axes) {
    scores.emplace_back((axis->dim_axis + 1) * axis->HasAttr(AT_REDUCE_AXIS));
  }
  return scores;
}

std::vector<double> TilingPriorityScorer::ComputeParallelism(std::vector<TileAxis *> tile_axes) {
  std::vector<double> scores;
  scores.reserve(tile_axes.size());
  for (auto axis : tile_axes) {
    scores.emplace_back(!axis->mc_sup);
  }
  return scores;
}

std::vector<double> TilingPriorityScorer::ComputeVectorization(std::vector<TileAxis *> tile_axes) {
  std::vector<double> scores;
  scores.reserve(tile_axes.size());
  for (auto axis : tile_axes) {
    int vec_level = 0;
    for (auto it : analyzer_.buf_info_) {
      auto buf = it.second.get();
      int coef;
      if (buf->scope == TilingMemScope::MEM_SCOPE_GM) {
        // continuous dma copy is considered as the most important factor
        coef = 2;
      } else if (buf->scope == TilingMemScope::MEM_SCOPE_BUFFER) {
        // vectorization instruction is also important
        coef = 1;
      } else {
        // does not consider impact of C1 and C0 dma copy
        coef = 0;
      }
      int dim_depth = 1;
      for (auto &a : *(buf->tile_axis)) {
        if (a == axis) {
          vec_level += coef * dim_depth;
          break;
        }
        dim_depth += 1;
      }
    }
    scores.emplace_back(vec_level);
  }
  return scores;
}

// constraint not used in npu

void GpuStrategy::AddNpuConstraint() {}

void GpuDmaAnalysisStrategy::AddNpuConstraint() {}

}  // namespace poly
}  // namespace ir
}  // namespace akg
