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
#include <dmlc/logging.h>

#include "poly/tiling/hermes/stmt_info.h"
#include "poly/tiling/hermes/node.h"
#include "poly/tiling/hermes/model_graph.h"

namespace akg {
namespace ir {
namespace poly {
Node::Node() : op_{Op()} {}

Node::Node(const std::shared_ptr<Node> &orig_node)
    : name_{orig_node->name_},
      op_{orig_node->op_},
      succ_{orig_node->succ_},
      pred_{orig_node->pred_},
      axis_of_node_{orig_node->axis_of_node_},
      transformed_output_shape_{orig_node->transformed_output_shape_},
      axis_to_tensor_to_shape_id_map_{orig_node->axis_to_tensor_to_shape_id_map_},
      attrs_{orig_node->attrs_} {
  output_tensors_.reserve(orig_node->output_tensors_.size());
  for (auto const &tensor : orig_node->output_tensors_) {
    output_tensors_.push_back(std::make_shared<Tensor>(tensor));
  }
  input_tensors_.reserve(orig_node->input_tensors_.size());
  for (auto const &tensor : orig_node->input_tensors_) {
    input_tensors_.push_back(std::make_shared<Tensor>(tensor));
  }
}

Node::Node(const std::string &name, const Op &op, const std::vector<std::shared_ptr<Tensor>> &output_tensors,
           const std::vector<std::shared_ptr<Tensor>> &input_tensors, const std::vector<std::shared_ptr<Node>> &succ,
           const std::vector<std::shared_ptr<Node>> &pred, const std::vector<Axis> &axis,
           const std::vector<Tensor> &transformed_output_shape,
           const std::map<std::string, std::map<Tensor, int64_t>> &axis_to_tensor_to_shape_id_map,
           const std::list<attributes> &attrs)
    : name_{name},
      op_{op},
      output_tensors_{output_tensors},
      input_tensors_{input_tensors},
      succ_{succ},
      pred_{pred},
      axis_of_node_{axis},
      transformed_output_shape_{transformed_output_shape},
      axis_to_tensor_to_shape_id_map_{axis_to_tensor_to_shape_id_map},
      attrs_{attrs} {}

void Node::SetNodesAxisDim(std::vector<std::shared_ptr<Node>> &nodes) {
  for (auto &node : nodes) {
    std::string statement;
    if (node->orig_name_.empty()) {
      statement = StmtInfo::node_to_stmt_map_.find(node->name_)->second;
    } else {
      statement = StmtInfo::node_to_stmt_map_.find(node->orig_name_)->second;
    }
    std::vector<StmtInfo::StmtAxes> stmt_axes = StmtInfo::stmt_name_dim_range_map_.find(statement)->second;
    for (auto &axis : node->axis_of_node_) {
      bool is_axis_found = false;
      for (auto const &name_dim_range : stmt_axes) {
        if (axis.name_ == name_dim_range.name) {
          axis.dim_axis_ = name_dim_range.dim;
          is_axis_found = true;
          break;
        }
      }
      if (!is_axis_found) {
        axis.dim_axis_ = GetAxisDimFromNameRange(axis);
      }
    }
  }
}

int Node::GetAxisDimFromNameRange(const Axis &axis) {
  for (auto const &name_dim_range : ModelGraph::name_dim_range_set_) {
    if (axis.name_ == std::get<0>(name_dim_range) && axis.range_ == std::get<2>(name_dim_range)) {
      return std::get<1>(name_dim_range);
    }
  }
  return 0;
}

bool Node::HasName() const { return !(this->name_.empty()); }

bool Node::HasAxis(const Axis &axis) {
  for (size_t i = 0; i < this->axis_of_node_.size(); i++) {
    if (axis.dim_axis_ == this->axis_of_node_[i].dim_axis_) {
      return true;
    }
  }
  return false;
}

std::string AttrToString(const Node::attributes &attr) {
  switch (attr) {
    case Node::attributes::Transpose_A:
      return "transpose_a";
    case Node::attributes::Transpose_B:
      return "transpose_b";
    default:
      LOG(FATAL) << "[Node::attributes::AttrToString] This attribute has no string equivalent yet";
  }
  return "";
}

bool AttrIsTrue(const std::string &to_find, const std::string &str, size_t pos) {
  pos = str.find(to_find, pos);
  pos = str.find("'value': True", pos);
  return (pos != std::string::npos);
}

std::list<Node::attributes> FindAttributes(const std::vector<std::string> &attrs) {
  std::list<Node::attributes> att_l;
  std::string name = "'name':";
  size_t name_len = name.length();
  for (std::string const &str : attrs) {
    size_t pos = str.find(name);
    if (pos == std::string::npos) {
      continue;
    }
    pos += name_len;
    if (AttrIsTrue(AttrToString(Node::attributes::Transpose_A), str, pos)) {
      att_l.push_front(Node::attributes::Transpose_A);
    }
    if (AttrIsTrue(AttrToString(Node::attributes::Transpose_B), str, pos)) {
      att_l.push_front(Node::attributes::Transpose_B);
    }
  }

  return att_l;
}

bool Node::HasAttr(const std::string &attr) const {
  for (auto attribute : this->attrs_) {
    if (attr == "transpose_a" && attribute == Node::attributes::Transpose_A) {
      return true;
    }
    if (attr == "transpose_b" && attribute == Node::attributes::Transpose_B) {
      return true;
    }
  }
  return false;
}

std::string Node::ToString() const { return this->name_; }
}  // namespace poly
}  // namespace ir
}  // namespace akg
