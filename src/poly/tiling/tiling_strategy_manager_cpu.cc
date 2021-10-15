/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <numeric>

#include "../../src/include/build_module.h"
#include "./tiling_analyzer.h"
#include "poly/schedule_pass_gpu/register_memory_manager.h"

namespace akg {
namespace ir {
namespace poly {

void CpuStrategy::AddCpuConstraint() {
  InitMappingLimit();

  BuildAxesQueue();

  InjectiveSpeedup();
  SetMappingConfig();
  analyzer_->RootAxis()->MarkWithAttr(AttrInfo{AT_TEMPLATE, template_map_[Template::CPU]});
}

void CpuStrategy::InitMappingLimit() { DetermineTemplate(); }

void CpuStrategy::BuildAxesQueue() {
  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    const auto r = axis->range_extent.as<IntImm>();
    if (r && r->value > 0 && !axis->is_inner) {
      this->pending_axes_.emplace_back(std::make_pair(axis, r->value));
    }
  });

  int len = pending_axes_.size();
  bool is_reduce_op = (template_ == Template::ALL_REDUCTION || template_ == Template::REDUCTION ||
                      template_ == Template::BITWISE_REDUCTION);
  if (is_reduce_op) {
    if (analyzer_->scop_info_.analysis_result_.GetReduceDirection() == Y_DIRECTION) {
      analyzer_->scop_info_.analysis_result_.SetLastAxisInScheduleTree(0);
    } else {
      analyzer_->scop_info_.analysis_result_.SetLastAxisInScheduleTree(len - 1);
    }
  }
  if (template_ == Template::PURE_ELEM) {
    SetCoalescedAccess();
  }
}

void CpuStrategy::SetMappingConfig() {
  std::stringstream ss;
  ss << "Use template " << template_map_[template_];

  analyzer_->GetTileLogger().AppendLog(CPU_MAPPING, ss);

  ss << "Tile = ";
  analyzer_->ForEachAxisTopDown([this, &ss](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }

    ss << axis->c1_constraints.tile_extent_ << "," << axis->c0_constraints.tile_extent_ << ",";
  });
  analyzer_->GetTileLogger().AppendLog(CPU_MAPPING, ss);
}

void CpuStrategy::DetermineTemplate() {
  size_t depth = 0;
  analyzer_->ForEachAxisTopDown([this, &depth](TileAxis *axis) {
    if (axis == analyzer_->RootAxis()) {
      return;
    }
    ++depth;
  });
  depth_ = depth;
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

  if (!analyzer_->GetAxesOfAttr(AttrInfo{AT_OP_TYPE, AT_PAD}).empty()) {
    template_ = Template::PAD_OP;
    return;
  }

  if (!analyzer_->GetAxesOfAttr(AT_CONV).empty()) {
    template_ = Template::CONV;
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

  template_ = reduce_axes_.size() == depth ? Template::ALL_REDUCTION : Template::REDUCTION;
  return;
}

void CpuStrategy::SetUnrollTileValue(TileAxis *axis, const int64_t axis_size, int64_t &tile_left) {
  int64_t tile_val = best_unroll_num_;
  int64_t tile_size = axis_size;
  while (tile_size % tile_val != 0 && tile_val > this->min_unroll_num_) {
    tile_val /= 2;
  }
  tile_val = std::min(tile_size, tile_val);
  tile_left = tile_size / tile_val;
  axis->TileRestrainToSingleValue(Expr(tile_val), TileLevel::CACHE1);
  axis->TileRestrainToSingleValue(Expr(tile_val), TileLevel::CACHE0);
}

void CpuStrategy::SetParallelTileValue(TileAxis *axis, const int64_t axis_size, 
    const int64_t data_size, bool is_unroll_axis, int64_t tile_left) {
  int tile_size = axis_size;
  int parallel_num = best_parallel_num_;
  int c0_tile_value = 1;

  if (is_unroll_axis) {
    CHECK(axis->c0_constraints.tile_extent_.as<IntImm>());
    c0_tile_value = axis->c0_constraints.tile_extent_.as<IntImm>()->value;
    tile_size = tile_left;
  }
  int evaluate_num = data_size / min_exec_num_per_thread_;
  if (evaluate_num >= best_parallel_num_) {
    parallel_num = best_parallel_num_;
  } else if (evaluate_num > 1) {
    while (parallel_num > 0 && tile_size % parallel_num != 0) {
      if (parallel_num < evaluate_num) {
        break;
      }
      parallel_num -= parallel_decrease_value_;
    }
  } else {
    parallel_num = 1;
  }
  if (parallel_num <= 0) {
    parallel_num = evaluate_num;
  }
  int tile_value = std::max(tile_size * c0_tile_value / parallel_num, c0_tile_value);
  axis->TileRestrainToSingleValue(Expr(tile_value), TileLevel::CACHE1);
  axis->TileRestrainToSingleValue(Expr(c0_tile_value), TileLevel::CACHE0);
}

void CpuStrategy::InjectiveSpeedup() {
  size_t ori_size = pending_axes_.size();
  int64_t data_size = 1;
  for (int i = static_cast<int>(ori_size - 1); i >= 0; i--) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[i];
    data_size *= shape;
    int64_t tile_outer_left = 1;
    int vectorize_axis = analyzer_->scop_info_.analysis_result_.GetLastAxisInScheduleTree();

    if (vectorize_axis == i) {
      SetUnrollTileValue(axis, shape, tile_outer_left);
    }

    /* Set parallel tile size on the outermost axis */
    if (i == 0) {
      bool is_unroll_axis = vectorize_axis == 0  ? true : false;
      SetParallelTileValue(axis, shape, data_size, is_unroll_axis, tile_outer_left);
    }
  }
}

void CpuStrategy::SetCoalescedAccess() {
  isl::schedule_node root = analyzer_->sch_.get_root();
  isl::schedule_node node = GetOuterBand(root);
  if (!node.isa<isl::schedule_node_band>()) {
    return;
  }

  auto band_node = node.as<isl::schedule_node_band>();
  auto n_parallel_axis = CountConsecutiveCoincident(band_node);
  node = band_node.split(n_parallel_axis);

  std::unordered_set<std::string> skip_tensors;
  // Get read and write tensor information.
  auto reads_access = analyzer_->scop_info_.analysis_result_.GetReads().domain_factor_domain();
  int last_axis = GetLastAxis(node, reads_access, skip_tensors);
  if (last_axis != -1) {
    analyzer_->scop_info_.analysis_result_.SetLastAxisInScheduleTree(last_axis);
    return;
  }

  auto write_access = analyzer_->scop_info_.analysis_result_.GetWrites().domain_factor_domain();
  last_axis = GetLastAxis(node, write_access, skip_tensors);
  if (last_axis != -1) {
    analyzer_->scop_info_.analysis_result_.SetLastAxisInScheduleTree(last_axis);
    return;
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
