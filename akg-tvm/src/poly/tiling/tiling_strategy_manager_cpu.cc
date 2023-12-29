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
#include "poly/tiling/tiling_utils.h"

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

void CpuStrategy::GenConv2dTileByAxis(int64_t &p, int64_t tile1, int64_t tile0) {
  TileAxis *axis = nullptr;
  int64_t shape;
  std::tie(axis, shape) = pending_axes_[current_band_][p];
  CHECK(axis != nullptr);
  axis->TileRestrainToSingleValue(Expr((int64_t)tile1), TileLevel::CACHE1);
  axis->TileRestrainToSingleValue(Expr((int64_t)tile0), TileLevel::CACHE0);
  p += 1;
}

void CpuStrategy::SetConv2dTileValue() {
  // format of conv2d tile should be: batch, oc_out, oh, ow, oc_in, ic_out.
  // all of them can be 1, so we use axes_names to check the exist of each axis.
  auto axes_names = analyzer_->scop_info_.analysis_result_.GetCpuConvolutionAxes();
  int64_t p = 0;
  if (axes_names.find(CONV_BATCH) != std::string::npos) {  // batch
    GenConv2dTileByAxis(p, 1, 1);
  }
  if (axes_names.find(CONV_OC_OUT) != std::string::npos) {  // oc_out
    GenConv2dTileByAxis(p, 1, 1);
  }
  if (axes_names.find(CONV_OH) != std::string::npos) {  // oh
    GenConv2dTileByAxis(p, 1, 1);
  }
  if (axes_names.find(CONV_OW) != std::string::npos) {  // ow
    int64_t ow_shape = pending_axes_[current_band_][p].second;
    /* ow_inner should follow some strategy:
    1. ow_shape % ow_tile == 0
    2. ow_tile is smaller than simd length */
    int64_t ow_tile = 1;
    for (auto t = std::min((int64_t)15, ow_shape); t >= 1; t--) {
      CHECK(t != 0) << "Divisor t is 0, please check it.";
      if (ow_shape % t == 0) {
        ow_tile = t;
        break;
      }
    }
    GenConv2dTileByAxis(p, ow_tile, ow_tile);
  }
  if (axes_names.find(CONV_OC_IN) != std::string::npos) {  // oc_in
    int64_t oc_in_shape = pending_axes_[current_band_][p].second;
    GenConv2dTileByAxis(p, oc_in_shape, oc_in_shape);
  }
  if (axes_names.find(CONV_IC_OUT) != std::string::npos) {  // ic_out: reduction axis
    int64_t ic_out_shape = pending_axes_[current_band_][p].second;
    GenConv2dTileByAxis(p, ic_out_shape, 1);
  }
  if (axes_names.find(CONV_IC_OUT) == std::string::npos &&
      axes_names.find(CONV_KH) != std::string::npos) {  // kh: reduction axis
    GenConv2dTileByAxis(p, 1, 1);
  }
  if (axes_names.find(CONV_IC_OUT) == std::string::npos && axes_names.find(CONV_KH) == std::string::npos &&
      axes_names.find(CONV_KW) != std::string::npos) {  // kw: reduction axis
    GenConv2dTileByAxis(p, 1, 1);
  }
  if (axes_names.find(CONV_IC_OUT) == std::string::npos && axes_names.find(CONV_KH) == std::string::npos &&
      axes_names.find(CONV_KW) == std::string::npos && axes_names.find(CONV_IC_IN) != std::string::npos) {
    // ic_in: reduction axis
    int64_t ic_in_shape = pending_axes_[current_band_][p].second;
    GenConv2dTileByAxis(p, ic_in_shape, ic_in_shape);
  }
}

void CpuStrategy::SetMatMulTileValue() {
  auto pack_size = analyzer_->scop_info_.analysis_result_.GetPackBlockSize();
  for (int i = 0; i < static_cast<int>(pending_axes_[current_band_].size()); ++i) {
    TileAxis *axis;
    int64_t pack = 1;
    int64_t shape;
    const int64_t k_align_value = 8;
    std::tie(axis, shape) = pending_axes_[current_band_][i];
    int64_t value = shape;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key != AT_GEMM) {
        continue;
      }

      if (attr.attr_value == kDsabi) {
        value = 1;
      } else if (attr.attr_value == kDsaki) {
        value = std::max(shape / k_align_value * k_align_value, 1L);
      }
      axis->TileRestrainToSingleValue(Expr(value), TileLevel::CACHE1);

      if (attr.attr_value == kDsami) {
        pack = pack_size.pack_a_size;
      } else if (attr.attr_value == kDsani) {
        pack = pack_size.pack_b_size;
      }
      CHECK(pack != 0);
      value = std::max(value / pack * pack, 1L);
      axis->TileRestrainToSingleValue(Expr(value), TileLevel::CACHE0);

      auto current_outer_node = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(current_band_);
      current_outer_node->mnk_pos[attr.attr_value] = axis->dim_axis;
    }
  }
}

bool CpuStrategy::SetReduceYTileValue() {
  auto axes_num = pending_axes_[current_band_].size();
  CHECK(axes_num >= REDUCE_Y_LEAST_AXES_NUM) << "axes_num is less than 2";
  bool is_tiled = false;
  TileAxis *axis1, *axis0;
  int64_t shape1, shape0;
  std::tie(axis0, shape0) = pending_axes_[current_band_][0];
  std::tie(axis1, shape1) = pending_axes_[current_band_][1];
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

void CpuStrategy::SetElementWiseTileValue() {
  size_t ori_size = pending_axes_[current_band_].size();
  int64_t data_size = 1;
  for (int i = static_cast<int>(ori_size - 1); i >= 0; i--) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[current_band_][i];
    data_size *= shape;
    int64_t tile_outer_left = 1;

    int vectorize_axis = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(current_band_)->last_axis;
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

void CpuStrategy::SetTransposeTileValue() {
  std::unordered_set<std::string> write_tensor_name;
  auto current_outer_bn = analyzer_->scop_info_.analysis_result_.GetOuterBandNode(current_band_);
  isl::map_list access_list = current_outer_bn->writes.get_map_list();
  for (auto access : access_list) {
    auto access_id = access.domain_factor_domain().get_tuple_id(isl_dim_out);
    write_tensor_name.insert(access_id.get_name());
  }

  int transpose_write_axis_pos = -1;
  std::unordered_set<int> transpose_axis_pos;
  size_t ori_size = pending_axes_[current_band_].size();
  for (int i = static_cast<int>(ori_size - 1); i >= 0; i--) {
    TileAxis *axis;
    int64_t shape;
    std::tie(axis, shape) = pending_axes_[current_band_][i];

    bool is_inner_axis = false;
    bool is_transpose_axis = false;
    std::string tensor_name;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_value == AT_TRANSPOSE_AXIS) {
        is_transpose_axis = true;
      }

      if (attr.attr_key == AT_TRANSPOSE_INNERMOST_AXIS) {
        is_inner_axis = true;
        tensor_name = attr.attr_value;
      }
    }

    if (is_transpose_axis && is_inner_axis) {
      transpose_axis_pos.emplace(i);
      if (write_tensor_name.count(tensor_name) != 0) {
        transpose_write_axis_pos = i;
      }
    }
  }

  const size_t transpose_size = 2;
  const int64_t transpose_row = 8;
  const int64_t transpose_col = 4;
  if (transpose_axis_pos.size() == transpose_size) {
    current_outer_bn->enable_transpose = true;
    for (int i = static_cast<int>(ori_size - 1); i >= 0; i--) {
      TileAxis *axis;
      int64_t shape;
      std::tie(axis, shape) = pending_axes_[current_band_][i];
      int64_t tile_val = 1;
      if (analyzer_->IsPrime(shape)) {
        tile_val = shape < thread_num_ ? shape : thread_num_;
      } else {
        tile_val = analyzer_->GetLargestDivisor(shape - 1, shape);
      }
      axis->TileRestrainToSingleValue(Expr(tile_val), TileLevel::CACHE1);

      tile_val = 1;
      if (transpose_axis_pos.count(i) != 0) {
        if (transpose_write_axis_pos == i) {
          tile_val = shape < transpose_row ? shape : transpose_row;
        } else {
          tile_val = shape < transpose_col ? shape : transpose_col;
        }
      }
      axis->TileRestrainToSingleValue(Expr(tile_val), TileLevel::CACHE0);
    }
  } else {
    current_outer_bn->enable_transpose = false;
    SetElementWiseTileValue();
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
      SetConv2dTileValue();
      continue;
    }
    if (op_type == Template::MATMUL) {
      SetMatMulTileValue();
      continue;
    }
    auto reduce_direction = analyzer_->scop_info_.analysis_result_.GetReduceDirection();
    if (op_type == Template::REDUCTION && reduce_direction == ReduceDirection::Y) {
      bool is_tiled = SetReduceYTileValue();
      if (is_tiled) {
        continue;
      }
    }

    auto enable_transpose = analyzer_->scop_info_.user_config_.GetEnableTranspose();
    if (op_type == Template::TRANSPOSE_OP && enable_transpose) {
      SetTransposeTileValue();
      continue;
    }

    SetElementWiseTileValue();
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
