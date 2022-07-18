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
#include <string>

#include "poly/tiling/tiling_utils.h"
#include "poly/tiling/hermes/classify_axis.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/op.h"

namespace akg {
namespace ir {
namespace poly {
void ClassifyAxis(const Node &node) {
  size_t axis_total = node.axis_of_node_.size();
  auto op = node.op_.op_type_;
  if (axis_total == 1) {
    if (!(op == Op::OpType::ReduceX || op == Op::OpType::ReduceY || op == Op::OpType::ReduceDST)) {
      SetAxisTypeAsMultiCoreAndVectorization(node.axis_of_node_[0].dim_axis_);
    }
  } else if (axis_total > 1) {
    if (op == Op::OpType::MatMul || op == Op::OpType::BatchMatMul) {
      ClassifyMatmulAxis();
      return;
    }
    for (size_t i = 0; i < axis_total; i++) {
      DefineAxisType(node.axis_of_node_[i].dim_axis_);

      // Situation where BMM/MM becomes Mul.
      // Need to define the first dimension after the Batch dimensions as MultiCore.
      if (node.name_.find(kBatchMatMul) != std::string::npos) {
        size_t last_idx = node.output_tensors_[0]->shape_.size() - 1;
        size_t penultimate_idx = last_idx - 1;
        SetAxisTypeAsMultiCore(node.axis_of_node_[penultimate_idx].dim_axis_);
      }
    }
  }
}

void SetAxisTypeAsMultiCore(size_t dim_axis) {
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (axis.dim_axis_ == dim_axis) {
      axis.type_.insert(Axis::AxisLabel::kMultiCore);
    }
  }
}

void SetAxisTypeAsVectorization(size_t dim_axis) {
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (axis.dim_axis_ == dim_axis) {
      axis.type_.insert(Axis::AxisLabel::kVectorization);
    }
  }
}

void SetAxisTypeAsMultiCoreAndVectorization(size_t dim_axis) {
  for (auto &axis : ModelGraph::global_axis_vec_) {
    if (axis.dim_axis_ == dim_axis) {
      axis.type_.insert(Axis::AxisLabel::kMultiCore);
      axis.type_.insert(Axis::AxisLabel::kVectorization);
    }
  }
}

void ClassifyMatmulAxis() {
  for (auto &global_axis : ModelGraph::global_axis_vec_) {
    if (!global_axis.is_inner_) {
      auto gemm_axis = global_axis.gemm_axis_;
      if (gemm_axis == kDsami || gemm_axis == kDsani || gemm_axis == kDsaki) {
        global_axis.type_.insert(Axis::AxisLabel::kMatMulAxis16);
      } else if (gemm_axis == kDsano) {
        global_axis.type_.insert(Axis::AxisLabel::kMatMulAxisN);
      } else if (gemm_axis == kDsamo) {
        global_axis.type_.insert(Axis::AxisLabel::kMatMulAxisM);
      } else if (gemm_axis == kDsako) {
        global_axis.type_.insert(Axis::AxisLabel::kMatMulAxisK);
      } else {
        global_axis.type_.insert(Axis::AxisLabel::kMatMulAxisBatch);
      }
    }
  }
}

void DefineAxisType(size_t dim_axis) {
  if (!ModelGraph::global_axis_vec_.empty() && ModelGraph::global_axis_vec_[0].dim_axis_ == dim_axis) {
    SetAxisTypeAsMultiCore(dim_axis);
  } else {
    SetAxisTypeAsVectorization(dim_axis);
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
