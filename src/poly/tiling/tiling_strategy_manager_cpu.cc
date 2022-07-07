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

constexpr size_t REDUCE_Y_LEAST_AXES_NUM = 2;
constexpr int REDUCE_Y_TILE_SIZE = 2048;
constexpr int REDUCE_Y_LEAST_BLOCK_SIZE = 8192;
constexpr int REDUCE_Y_LEAST_X_SIZE = 8;

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
      if (this->analyzer_->scop_info_.analysis_result_.GetOuterBandNode(axis->index)->template_type ==
          Template::MATMUL) {
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
  auto direction = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(current_band_)->reduce_direction;
  if (direction == ReduceDirection::Y || direction == ReduceDirection::ALL) {
    if (axis_size % parallel_num != 0) {
      tile_value = axis_size;
    }
  }
  if (tile_value < min_unroll_num_ && is_unroll_axis) {
    tile_value = std::min(axis_size, static_cast<int64_t>(min_unroll_num_));
    c0_tile_value = tile_value;
  }
  tile_value = std::max(tile_value, c0_tile_value);
  axis->TileRestrainToSingleValue(Expr(tile_value), TileLevel::CACHE1);
  axis->TileRestrainToSingleValue(Expr(c0_tile_value), TileLevel::CACHE0);
}

void CpuStrategy::SetConv2dTileValue(int index) {
  // format of conv2d tile should be: batch, oc_out, oh, ow, oc_in, ic_out.
  // all of them can be 1, so we use axes_names to check the exist of each axis.

  auto axes_names = analyzer_->scop_info_.analysis_result_.GetCpuConvolutionAxes();
  int64_t p = 0;

  // batch
  if (axes_names.find(CONV_BATCH) != std::string::npos) {
    TileAxis *batch_axis = nullptr;
    int64_t _;
    std::tie(batch_axis, _) = pending_axes_[index][p];
    batch_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE1);
    batch_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE0);    
    p += 1;
  }

  // oc_out
  if (axes_names.find(CONV_OC_OUT) != std::string::npos) {
    TileAxis *oc_out_axis = nullptr;
    int64_t _;
    std::tie(oc_out_axis, _) = pending_axes_[index][p];
    oc_out_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE1);
    oc_out_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE0);
    p += 1;
  }

  // oh
  if (axes_names.find(CONV_OH) != std::string::npos) {
    TileAxis *oh_axis = nullptr;
    int64_t _;
    std::tie(oh_axis, _) = pending_axes_[index][p];
    oh_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE1);
    oh_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE0);
    p += 1;
  }

  // ow
  if (axes_names.find(CONV_OW) != std::string::npos) {
    TileAxis *ow_axis = nullptr;
    int64_t ow_shape;
    std::tie(ow_axis, ow_shape) = pending_axes_[index][p];

    /* ow_inner should follow some strategy:
    1. ow_shape % ow_tile == 0
    2. ow_tile is smaller than simd length */
    int64_t ow_tile = 1;
    for (auto t = std::min((int64_t)31, ow_shape); t >= 1; t--) {
      CHECK(t != 0) << "Divisor t is 0, please check it.";
      if (ow_shape % t == 0) {
        ow_tile = t;
        break;
      }
    }
    ow_axis->TileRestrainToSingleValue(Expr(ow_tile), TileLevel::CACHE1);
    ow_axis->TileRestrainToSingleValue(Expr(ow_tile), TileLevel::CACHE0);
    p += 1;
  }

  // oc_in
  if (axes_names.find(CONV_OC_IN) != std::string::npos) {
    TileAxis *oc_in_axis = nullptr;
    int64_t oc_in_shape;
    std::tie(oc_in_axis, oc_in_shape) = pending_axes_[index][p];
    oc_in_axis->TileRestrainToSingleValue(Expr(oc_in_shape), TileLevel::CACHE1);
    oc_in_axis->TileRestrainToSingleValue(Expr(oc_in_shape), TileLevel::CACHE0);
    p += 1;
  }

  // ic_out
  if (axes_names.find(CONV_IC_OUT) != std::string::npos) {
    TileAxis *ic_out_axis = nullptr;
    int64_t ic_out_shape;
    std::tie(ic_out_axis, ic_out_shape) = pending_axes_[index][p];
    ic_out_axis->TileRestrainToSingleValue(Expr(ic_out_shape), TileLevel::CACHE1);
    ic_out_axis->TileRestrainToSingleValue(Expr((int64_t)1), TileLevel::CACHE0);
    p += 1;
  }

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

bool CpuStrategy::SetReduceYTileValue(int index) {
  auto axes_num = pending_axes_[index].size();
  CHECK(axes_num >= REDUCE_Y_LEAST_AXES_NUM) << "axes_num is less than 2";
  bool is_tiled = false;
  TileAxis *axis1, *axis0;
  int64_t shape1, shape0;
  std::tie(axis0, shape0) = pending_axes_[index][0];
  std::tie(axis1, shape1) = pending_axes_[index][1];
  int64_t value1 = shape1;
  if (shape1 >= REDUCE_Y_LEAST_BLOCK_SIZE && shape0 <= REDUCE_Y_LEAST_X_SIZE) {
    int64_t value0 = shape0;
    axis0->TileRestrainToSingleValue(Expr(value0), TileLevel::CACHE1);
    axis0->TileRestrainToSingleValue(Expr(value0), TileLevel::CACHE0);
    value1 = REDUCE_Y_TILE_SIZE;
    is_tiled = true;
  }
  axis1->TileRestrainToSingleValue(Expr(value1), TileLevel::CACHE1);
  axis1->TileRestrainToSingleValue(Expr(value1), TileLevel::CACHE0);
  return is_tiled;
}

void CpuStrategy::SetCsrTileValue() {
  TileAxis *axis;
  int64_t row_length;
  if (pending_axes_.size() <= 0) {
    LOG(WARNING) << "[Tiling CSR OP]:No axis to tile.";
    return;
  }
  if (pending_axes_[OUTERMOST_AXIS].size() <= 0) {
    LOG(WARNING) << "[Tiling CSR OP]:No outermost axis to tile.";
    return;
  }
  std::tie(axis, row_length) = pending_axes_[OUTERMOST_AXIS][OUTERMOST_AXIS];
  int64_t csr_tensor_size = row_length * analyzer_->scop_info_.analysis_result_.GetCsrAvgRow();
  if (csr_tensor_size > CPU_CSR_PARALLEL_CUTOFF) {
    axis->TileRestrainToSingleValue(Expr(CPU_CSR_TILING_FACTOR), TileLevel::CACHE1);
    axis->TileRestrainToSingleValue(Expr(CPU_CSR_TILING_FACTOR), TileLevel::CACHE0);
  }
  if (analyzer_->scop_info_.analysis_result_.GetOpTemplate() != Template::REDUCTION) {
    analyzer_->ForEachAxisTopDown([this](TileAxis *a) {
      if (a == this->analyzer_->RootAxis()) {
        return;
      }
      if (this->analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(a->range_extent)) {
        a->TileRestrainToSingleValue(Expr(CPU_CSR_TILING_FACTOR), TileLevel::CACHE1);
        a->TileRestrainToSingleValue(Expr(CPU_CSR_TILING_FACTOR), TileLevel::CACHE0);
      }
    });
  }
}

void CpuStrategy::SetMultiLevelTileValue() {
  if (analyzer_->scop_info_.analysis_result_.GetCsr()) {
    SetCsrTileValue();
    return;
  }
  for (auto idx = 0; idx < static_cast<int>(pending_axes_.size()); ++idx) {
    current_band_ = idx;
    auto op_type = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(idx)->template_type;
    if (op_type == Template::CONV) {
      SetConv2dTileValue(idx);
      continue;
    }
    if (op_type == Template::MATMUL) {
      SetMatMulTileValue(idx);
      continue;
    }
    auto reduce_direction = analyzer_->scop_info_.analysis_result_.GetReduceDirection();
    if (reduce_direction == ReduceDirection::Y) {
      bool is_tiled = SetReduceYTileValue(idx);
      if (is_tiled) {
        continue;
      }
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
