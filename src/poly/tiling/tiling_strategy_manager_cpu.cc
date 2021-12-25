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
#include "tiling_analyzer.h"
#include "tiling_strategy_manager.h"

namespace akg {
namespace ir {
namespace poly {

void CpuStrategy::AddCpuConstraint() {
  BuildAxesQueue();
  SetMultiLevelTileValue();
  RecordTileValue();
}

void CpuStrategy::BuildAxesQueue() {
  int band_size = analyzer_->scop_info_.analysis_result_.GetOuterBandNumber();
  pending_axes_.resize(band_size);
  analyzer_->ForEachAxisTopDown([this](TileAxis *axis) {
    if (axis == this->analyzer_->RootAxis()) {
      return;
    }
    const auto r = axis->range_extent.as<IntImm>();
    if (r && r->value > 0 && !axis->is_inner) {
      if (this->analyzer_->scop_info_.analysis_result_.GetOuterBandNode(axis->index)->template_type == Template::MATMUL) {
        axis->MarkWithAttr(AttrInfo{"axis_token", this->axes_name_[axis->dim_axis]});
      }
      this->pending_axes_[axis->index].emplace_back(std::make_pair(axis, r->value));
    }
  });
}

void CpuStrategy::RecordTileValue() {
  std::stringstream ss;
  for (auto i = 0; i < static_cast<int>(pending_axes_.size()); ++i) {
    auto current_outer_bn = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(i);
    ss << "Band No." << i << " use template "
       << analyzer_->scop_info_.analysis_result_.ShowOpTemplate(current_outer_bn->template_type);
    if (current_outer_bn->template_type == Template::REDUCTION ||
        current_outer_bn->template_type == Template::BITWISE_REDUCTION) {
      ss << "(" << analyzer_->scop_info_.analysis_result_.ShowReduceDirection(current_outer_bn->reduce_direction)
         << ")";
    }
    analyzer_->GetTileLogger().AppendLog(CPU_TILING, ss);
    ss << "Tile = {";
    for (auto &axes : pending_axes_[i]) {
      ss << "(" << axes.first->c1_constraints.tile_extent_ << ", " << axes.first->c0_constraints.tile_extent_ << "), ";
    }
    ss << "}.";
    analyzer_->GetTileLogger().AppendLog(CPU_TILING, ss);
  }
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

void CpuStrategy::SetParallelTileValue(TileAxis *axis, const int64_t axis_size, const int64_t data_size,
                                       bool is_unroll_axis, int64_t tile_left) {
  int64_t tile_size = axis_size;
  int64_t parallel_num = best_parallel_num_;
  int64_t c0_tile_value = 1;

  if (is_unroll_axis) {
    CHECK(axis->c0_constraints.tile_extent_.as<IntImm>());
    c0_tile_value = axis->c0_constraints.tile_extent_.as<IntImm>()->value;
    tile_size = tile_left;
  }
  int64_t evaluate_num = data_size / min_exec_num_per_thread_;
  if (evaluate_num >= best_parallel_num_) {
    parallel_num = std::min(axis_size, static_cast<int64_t>(best_parallel_num_));
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
  int64_t tile_value = axis_size / parallel_num;
  if (tile_value < min_unroll_num_) {
    tile_value = std::min(axis_size, static_cast<int64_t>(min_unroll_num_));
    c0_tile_value = tile_value;
  }
  tile_value = std::max(tile_value, c0_tile_value);
  axis->TileRestrainToSingleValue(Expr(tile_value), TileLevel::CACHE1);
  axis->TileRestrainToSingleValue(Expr(c0_tile_value), TileLevel::CACHE0);
}

void CpuStrategy::SetMatMulTileValue(int index) {
  for (int i = 0; i < static_cast<int>(pending_axes_[index].size()); ++i) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[index][i];
    int64_t value = shape;
    if ((i != axis_m_) && (shape % best_factor_for_matmul_ == 0)) {
      value = best_factor_for_matmul_;
    }
    axis->TileRestrainToSingleValue(Expr(value), TileLevel::CACHE1);
    axis->TileRestrainToSingleValue(Expr(value), TileLevel::CACHE0);
  }
}

void CpuStrategy::SetMultiLevelTileValue() {
  for (auto idx = 0; idx < static_cast<int>(pending_axes_.size()); ++idx) {
    auto op_type = analyzer_->scop_info_.analysis_result_.GetOuterBandNode()->template_type;
    if (op_type == Template::MATMUL) {
      SetMatMulTileValue(idx);
      continue;
    }
    size_t ori_size = pending_axes_[idx].size();
    int64_t data_size = 1;
    for (int i = static_cast<int>(ori_size - 1); i >= 0; i--) {
      TileAxis *axis;
      int64_t shape;
      std::tie(axis, shape) = pending_axes_[idx][i];
      data_size *= shape;
      int64_t tile_outer_left = 1;

      int vectorize_axis = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(idx)->last_axis;
      if (vectorize_axis == i) {
        SetUnrollTileValue(axis, shape, tile_outer_left);
      }

      /* Set parallel tile size on the outermost axis */
      if (i == 0) {
        bool is_unroll_axis = vectorize_axis == 0 ? true : false;
        SetParallelTileValue(axis, shape, data_size, is_unroll_axis, tile_outer_left);
      }
    }
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
