/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/op.h"

namespace akg {
namespace ir {
namespace poly {
std::vector<Axis> ModelGraph::global_axis_vec_;
std::set<std::tuple<std::string, int, int64_t>> ModelGraph::name_dim_range_set_;

ModelGraph::ModelGraph(InitGraph &init_graph) {
  // Add additional nodes generated by Reduce Op to InitGraph.
  CompleteNodesGeneratedByReduce(init_graph);
  std::vector<std::shared_ptr<Node>> critical_nodes = GetCriticalNodes(init_graph);
  this->name_ = init_graph.name_;
  this->nodes_ = init_graph.nodes_;
  this->inputs_ = init_graph.inputs_;
  this->outputs_ = init_graph.outputs_;
  this->critical_nodes_ = critical_nodes;
  this->dominant_category_ = init_graph.OperatorCategory();
}

ModelGraph::ModelGraph(const InitGraph &init_graph, const std::vector<std::shared_ptr<Node>> &critical_nodes)
    : InitGraph{init_graph.name_, init_graph.nodes_, init_graph.inputs_, init_graph.outputs_},
      critical_nodes_{critical_nodes} {}

void ModelGraph::CompleteNodesGeneratedByReduce(InitGraph &init_graph) {
  bool is_reduce = false;
  size_t reduce_idx = 0;
  for (size_t index_node = init_graph.nodes_.size(); index_node > 0; --index_node) {
    if (init_graph.nodes_[index_node - 1]->op_.IsReduce()) {
      reduce_idx = index_node - 1;
      is_reduce = true;
      break;
    }
  }
  if (!is_reduce) {
    return;
  }

  auto reduce_node = std::make_shared<Node>(init_graph.nodes_[reduce_idx]);

  ReduceDirection reduce_type = GetReduceDirection(reduce_node);

  int64_t ax0 = 0;
  int64_t ax1 = 0;
  for (auto const &axis : ModelGraph::global_axis_vec_) {
    if (axis.dim_axis_ == 0) {
      ax0 = axis.range_;
    } else if (axis.dim_axis_ == 1) {
      ax1 = axis.range_;
    }
  }

  int64_t dst_shape_size = 0;
  int64_t src_shape_size = 0;

  if (reduce_type == ReduceDirection::ALL) {
    reduce_node->op_.op_type_ = Op::OpType::AllReduce;
    init_graph.nodes_[reduce_idx]->op_.op_type_ = Op::OpType::AllReduce;
    reduce_node->output_tensors_[0]->shape_[0] = kExtraMemoryCoeffRequiredByReduceDst;
    int64_t shape_val = kExtraMemoryCoeffRequiredByReduceDst;
    reduce_node->transformed_output_shape_[0].shape_.push_back(shape_val);

    dst_shape_size = kExtraMemoryCoeffRequiredByReduceDst;
    src_shape_size = kExtraMemoryCoeffRequiredByAllReduce;
  } else {
    if (reduce_type == ReduceDirection::Y) {
      reduce_node->op_.op_type_ = Op::OpType::ReduceY;
      init_graph.nodes_[reduce_idx]->op_.op_type_ = Op::OpType::ReduceY;
    } else {
      reduce_node->op_.op_type_ = Op::OpType::ReduceX;
      init_graph.nodes_[reduce_idx]->op_.op_type_ = Op::OpType::ReduceX;
    }
    dst_shape_size = ax0 * kExtraMemoryCoeffRequiredByReduceDst;
    src_shape_size = ax0 * ax1 / kExtraMemoryCoeffRequiredByReduceSrc;
  }

  std::shared_ptr<Node> dst_node =
    SetReduceSrcDstNodes(reduce_node, kDstTmpSuffix, Op::OpType::ReduceDST, dst_shape_size);
  std::shared_ptr<Node> src_node =
    SetReduceSrcDstNodes(reduce_node, kSrcTmpSuffix, Op::OpType::ReduceSRC, src_shape_size);

  dst_node->axis_of_node_ = reduce_node->axis_of_node_;
  dst_node->axis_to_tensor_to_shape_id_map_ = reduce_node->axis_to_tensor_to_shape_id_map_;

  src_node->axis_of_node_ = reduce_node->pred_[0]->axis_of_node_;
  if (reduce_type == ReduceDirection::ALL) {
    src_node->axis_to_tensor_to_shape_id_map_ = reduce_node->axis_to_tensor_to_shape_id_map_;
  } else {
    src_node->axis_to_tensor_to_shape_id_map_ = reduce_node->pred_[0]->axis_to_tensor_to_shape_id_map_;
  }

  init_graph.nodes_.push_back(dst_node);
  init_graph.nodes_.push_back(src_node);
}

std::shared_ptr<Node> ModelGraph::SetReduceSrcDstNodes(const std::shared_ptr<Node> &reduce_node,
                                                       const std::string &suffix, Op::OpType op_type,
                                                       int64_t shape_size) {
  std::shared_ptr<Node> node = std::make_shared<Node>();

  node->name_ = reduce_node->name_ + suffix;
  node->op_.op_type_ = op_type;

  std::shared_ptr<Tensor> output_tensor = std::make_shared<Tensor>();
  output_tensor->shape_.push_back(shape_size);
  output_tensor->datatype_ = reduce_node->output_tensors_[0]->datatype_;
  output_tensor->format_ = reduce_node->output_tensors_[0]->format_;
  node->output_tensors_.push_back(output_tensor);

  std::vector<int64_t> tensor_shape;
  tensor_shape.push_back(shape_size);
  node->transformed_output_shape_.emplace_back(
    Tensor(tensor_shape, reduce_node->output_tensors_[0]->datatype_, reduce_node->output_tensors_[0]->format_));

  node->input_tensors_ = reduce_node->input_tensors_;
  node->succ_ = reduce_node->succ_;
  node->pred_ = reduce_node->pred_;

  return node;
}

ReduceDirection ModelGraph::GetReduceDirection(const std::shared_ptr<Node> &reduce_node) {
  if (reduce_node->output_tensors_.size() == 1 && reduce_node->output_tensors_[0]->shape_.size() == 1 &&
      reduce_node->output_tensors_[0]->shape_[0] == 1) {
    return ReduceDirection::ALL;
  }
  ReduceDirection reduce_type = ReduceDirection::UNKNOWN;
  for (auto const &axis : ModelGraph::global_axis_vec_) {
    if (!axis.is_inner_ && axis.is_reduce_axis_) {
      if (axis.is_reduce_src_last_) {
        reduce_type = ReduceDirection::X;
      } else {
        reduce_type = ReduceDirection::Y;
      }
    }
  }
  if (reduce_type == ReduceDirection::UNKNOWN) {
    LOG(DEBUG) << "unknown reduce type";
  }
  return reduce_type;
}

std::tuple<int64_t, int> ModelGraph::GetMinShapeAndDataCoef(const Axis &axis) const {
  int64_t min_shape = INT64_MAX;
  int data_coef = INT32_MAX;
  for (auto const &node : this->nodes_) {
    for (auto const &node_axis : node->axis_of_node_) {
      if (node_axis.dim_axis_ == axis.dim_axis_ && node_axis.range_ < min_shape) {
        min_shape = node_axis.range_;
        data_coef = node->output_tensors_[0]->GetDataTypeCoef();
        break;
      }
    }
  }
  return std::make_tuple(min_shape, data_coef);
}

bool ModelGraph::IsInVector(const std::string &name, const std::vector<std::shared_ptr<Node>> &node_vec) {
  return std::any_of(node_vec.begin(), node_vec.end(),
                     [&name](const std::shared_ptr<Node> &node) { return name == node->name_; });
}

std::vector<std::shared_ptr<Node>> ModelGraph::GetCriticalNodes(const InitGraph &init_graph) {
  std::vector<std::shared_ptr<Node>> critical_nodes;
  if (!init_graph.nodes_.empty()) {
    critical_nodes.push_back(init_graph.nodes_[0]);
  }
  for (size_t i = 1; i < init_graph.nodes_.size(); i++) {
    if (init_graph.nodes_[i]->name_.find(HInputOp::input) == 0 ||
        critical_nodes.back()->name_.find(HInputOp::input) == 0) {
      critical_nodes.push_back(init_graph.nodes_[i]);
    } else if (init_graph.nodes_[i]->op_.op_type_ == Op::OpType::AllReduce ||
               init_graph.nodes_[i]->op_.op_type_ == Op::OpType::ReduceX ||
               init_graph.nodes_[i]->op_.op_type_ == Op::OpType::ReduceY ||
               init_graph.nodes_[i]->op_.op_type_ == Op::OpType::ReduceDST ||
               init_graph.nodes_[i]->op_.op_type_ == Op::OpType::ReduceSRC) {
      critical_nodes.push_back(init_graph.nodes_[i]);
    } else if (IsInVector(init_graph.nodes_[i]->name_, init_graph.outputs_)) {
      critical_nodes.push_back(init_graph.nodes_[i]);
    } else if (init_graph.nodes_[i]->succ_.size() > 1) {
      critical_nodes.push_back(init_graph.nodes_[i]);
    } else {
      int64_t curr_node_prod_out_shape = kMinShapeSize;
      int64_t last_critc_node_prod_out_shape = kMinShapeSize;
      for (auto const &out_t : init_graph.nodes_[i]->output_tensors_) {
        curr_node_prod_out_shape *= out_t->GetShapeProduct() * out_t->GetDataTypeCoef();
        if (curr_node_prod_out_shape < 0) {
          curr_node_prod_out_shape = INT64_MAX;
          break;
        }
      }
      for (auto const &out_t : critical_nodes.back()->output_tensors_) {
        last_critc_node_prod_out_shape *= out_t->GetShapeProduct() * out_t->GetDataTypeCoef();
        if (last_critc_node_prod_out_shape < 0) {
          last_critc_node_prod_out_shape = INT64_MAX;
          break;
        }
      }
      if (curr_node_prod_out_shape >= last_critc_node_prod_out_shape) {
        critical_nodes.pop_back();
        critical_nodes.push_back(init_graph.nodes_[i]);
      }
    }
  }
  return critical_nodes;
}

void ModelGraph::InsertToNameDimRangeSet(const std::string &axis_name, const int sch_dim, const int64_t &range) {
  for (auto &axis : name_dim_range_set_) {
    if (std::get<0>(axis) == axis_name && std::get<1>(axis) == sch_dim && std::get<2>(axis) == range) {
      return;
    }
  }
  name_dim_range_set_.insert(std::make_tuple(axis_name, sch_dim, range));
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
