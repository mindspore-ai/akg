/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <cmath>

#include "poly/tiling/tiling_analyzer.h"
#include "poly/tiling/tiling_utils.h"
#include "poly/tiling/hermes/cube_tiling.h"
#include "poly/tiling/hermes/utils.h"

namespace akg {
namespace ir {
namespace poly {
// The tilings of MatMul must comply with the following rules :
// First rule:  Do not exceed the memory.
// Second rule: The size of the axis can be divisible by tiling.
// Third rule:  Use all the cores.
// Fourth rule: Maximize the tiling.
int64_t GetMatmulAxisNTiling(const Axis &axis, const std::vector<std::shared_ptr<Node>> &nodes, size_t mem_VC_size) {
  int64_t N_axis_tiling = 0;
  size_t available_mem_VC_size = GetAvailableMemVCSize(nodes, mem_VC_size);

  // Try to maximize the tiling of N and M. And carrying to the power of two to compute.
  auto first_tiling = static_cast<int64_t>(std::round(std::sqrt(available_mem_VC_size)));
  int64_t power_of_two = 1;
  while (power_of_two < first_tiling) {
    power_of_two *= kByTwoL;
  }
  first_tiling = power_of_two;

  // Search for a divisible tiling.
  if (axis.range_ % first_tiling == 0) {
    N_axis_tiling = first_tiling;
  } else {
    N_axis_tiling = TilingAnalyzer::GetLargestDivisor(first_tiling, axis.range_);
  }

  return N_axis_tiling;
}

size_t GetMulticoreFromAxis(size_t remaining_core, int64_t axis_size) {
  size_t core_used = 1;
  auto axis_range = static_cast<size_t>(axis_size);
  if (axis_range > 0 && remaining_core % axis_range == 0) {
    core_used = axis_range;
  } else {
    for (size_t i = remaining_core; i > 0; i--) {
      if (remaining_core % i == 0 && axis_range % i == 0) {
        core_used = i;
        break;
      }
    }
  }
  return core_used;
}

int64_t GetMatmulAxisMTiling(const Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  int64_t M_axis_tiling = 0;
  size_t available_mem_VC_size = GetAvailableMemVCSize(model_graph.nodes_, hardware.mem_VC_size_);

  // Try to get the information on the N-axis, if it exists.
  int64_t N_axis_range = 1;
  int64_t N_axis_tiling = 1;
  for (auto const &global_axis : ModelGraph::global_axis_vec_) {
    if (global_axis.type_.count(Axis::AxisLabel::kMatMulAxisN) != 0) {
      N_axis_range = global_axis.range_;
      N_axis_tiling = global_axis.c0_tiling_;
      break;
    }
  }

  size_t core_used = 1;
  // Calculate how many cores are used on the Batch-axis.
  for (auto const &global_axis : ModelGraph::global_axis_vec_) {
    if (global_axis.type_.count(Axis::AxisLabel::kMatMulAxisBatch) != 0) {
      core_used *= GetMulticoreFromAxis(hardware.num_core_ / core_used, global_axis.range_);
    }
  }
  // Calculate how many cores are used on the N-axis.
  if (core_used < kMinNumOfUsedCores) {
    core_used *= GetMulticoreFromAxis(hardware.num_core_ / core_used, N_axis_range / N_axis_tiling);
  }

  // Ensure that the VC memory is not exceeded.
  int64_t available_M_axis_tiling = static_cast<int64_t>(available_mem_VC_size) / N_axis_tiling;

  // Search for a divisible tiling.
  auto core_used_ratio = static_cast<int64_t>(SafeDivisor(hardware.num_core_ / core_used));
  if (axis.range_ % core_used_ratio == 0) {
    M_axis_tiling = TilingAnalyzer::GetLargestDivisor(available_M_axis_tiling, axis.range_ / core_used_ratio);
  } else {
    M_axis_tiling = TilingAnalyzer::GetLargestDivisor(available_M_axis_tiling, axis.range_);
  }

  return M_axis_tiling;
}

int64_t GetMatmulAxisKTiling(Axis &axis, const ModelGraph &model_graph, Hardware hardware) {
  // Try to get the information on the N-axis and the M-axis, if they exist.
  int64_t K_axis_tiling = 0;
  int64_t N_axis_tiling = 1;
  int64_t N_axis_range = 1;
  int64_t M_axis_tiling = 1;
  int64_t M_axis_range = 1;
  for (auto const &global_axis : ModelGraph::global_axis_vec_) {
    if (global_axis.type_.count(Axis::AxisLabel::kMatMulAxisN) != 0) {
      N_axis_tiling = global_axis.c0_tiling_;
      N_axis_range = global_axis.range_;
    }
    if (global_axis.type_.count(Axis::AxisLabel::kMatMulAxisM) != 0) {
      M_axis_tiling = global_axis.c0_tiling_;
      M_axis_range = global_axis.range_;
    }
  }

  // Ensure that the C0 memory is not exceeded.
  auto available_mem_C0_size = static_cast<int64_t>(hardware.mem_C0_size_ / kC0DataSizePerBatch);
  K_axis_tiling = (N_axis_tiling >= M_axis_tiling) ? (available_mem_C0_size / N_axis_tiling)
                                                   : (available_mem_C0_size / M_axis_tiling);

  // Search for a divisible tiling.
  K_axis_tiling = TilingAnalyzer::GetLargestDivisor(K_axis_tiling, axis.range_);

  if (model_graph.is_activated_double_buffer_) {
    int64_t largest_tensor_size =
      (N_axis_range >= M_axis_range) ? N_axis_range * axis.range_ : M_axis_range * axis.range_;
    if (largest_tensor_size < kLargeTensorSize) {
      if (K_axis_tiling % kByTwoL == 0) {  // if enough memory (tiling==axis) -> not divided by 2
        K_axis_tiling /= kByTwoL;
      }
    }
  }

  if (N_axis_range * M_axis_range >= kLargeTensorSize) {
    auto mem_C1_data_size_ratio = static_cast<int64_t>(hardware.mem_C1_size_ / kC1DataSizePerBatch);
    int64_t available_mem_C1_size = mem_C1_data_size_ratio / (N_axis_tiling + M_axis_tiling);
    axis.c1_tiling_ = TilingAnalyzer::GetLargestDivisor(available_mem_C1_size, axis.range_);
  }

  return K_axis_tiling;
}

bool HasElemwiseMultiPredInputOps(const std::vector<std::shared_ptr<Node>> &nodes) {
  for (auto const &node : nodes) {
    if (node->op_.op_type_ != Op::OpType::MatMul && node->op_.op_type_ != Op::OpType::BatchMatMul &&
        node->op_.op_type_ != Op::OpType::Input && node->pred_.size() > 1) {
      int input_nb = 0;
      for (auto const &pred : node->pred_) {
        if (pred->op_.op_type_ == Op::OpType::Input) {
          input_nb++;
        }
      }
      if (input_nb > 1) {
        return true;
      }
    }
  }
  return false;
}

bool IsNotReused(const std::vector<std::shared_ptr<Node>> &nodes) {
  bool is_not_reused = HasElemwiseMultiPredInputOps(nodes);
  for (auto const &node : nodes) {
    if (node->op_.op_type_ != Op::OpType::MatMul && node->op_.op_type_ != Op::OpType::BatchMatMul) {
      for (auto const &pred : node->pred_) {
        if (pred->axis_of_node_.size() == node->axis_of_node_.size()) {
          if (pred->output_tensors_[0]->shape_ != node->output_tensors_[0]->shape_) {
            return true;
          }
        } else {
          // condition on the input CONFLICTS with the cast judgment below,
          // to be removed if the broadcast judgment is used to replace the cast judgment
          if (node->op_.op_type_ != Op::OpType::Input && pred->op_.op_type_ != Op::OpType::Input) {
            return true;
          }
        }
      }
    }

    if (node->op_.op_type_ == Op::OpType::TransData || node->succ_.size() > 1 ||
        (node->op_.op_type_ == Op::OpType::Cast && !node->succ_.empty())) {
      return true;
    }
  }
  return is_not_reused;
}

size_t GetAvailableMemVCSize(const std::vector<std::shared_ptr<Node>> &nodes, size_t mem_VC_size) {
  bool is_elementwise_op = false;
  bool is_not_reused = false;
  bool is_select_op = false;

  is_not_reused = IsNotReused(nodes);

  for (auto const &node : nodes) {
    if (node->op_.op_type_ != Op::OpType::MatMul && node->op_.op_type_ != Op::OpType::BatchMatMul &&
        node->pred_.size() > 1) {
      is_elementwise_op = true;
    }
    if (node->op_.op_type_ == Op::OpType::Select) {
      is_select_op = true;
    }
  }

  size_t available_mem_VC_size = mem_VC_size / kVCDataSizePerBatch;
  if (is_elementwise_op) {
    available_mem_VC_size /= kByTwoUL;
  }
  if (is_not_reused) {
    available_mem_VC_size /= kByTwoUL;
  }
  if (is_select_op) {
    available_mem_VC_size /= kByTwoUL;
  }

  return available_mem_VC_size;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
