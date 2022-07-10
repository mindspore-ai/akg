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

#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
Node::Node(const std::string &name, Op op, const std::vector<std::shared_ptr<Tensor>> &output_tensors,
           const std::vector<std::shared_ptr<Tensor>> &input_tensors, const std::vector<std::shared_ptr<Node>> &succ,
           const std::vector<std::shared_ptr<Node>> &pred, const std::vector<Axis> &axis,
           const std::vector<Tensor> &transformed_output_shape,
           const std::map<std::string, std::map<Tensor, int>> &axis_to_tensor_to_shape_id_map,
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

bool Node::HasName() const { return !(this->name_.empty()); }

bool Node::HasAxis(const Axis &axis) {
  int size = this->axis_of_node_.size();
  for (int i = 0; i < size; i++) {
    if (axis.dim_axis_ == this->axis_of_node_[i].dim_axis_) {
      return true;
    }
  }
  return false;
}

std::string AttrToString(Node::attributes attr) {
  switch (attr) {
    case Node::attributes::Transpose_A:
      return "transpose_a";
    case Node::attributes::Transpose_B:
      return "transpose_b";
  }
  LOG(FATAL) << "[Node::attributes::AttrToString] This attribute has no string equivalent yet";
  return "";
}

bool AttrIsTrue(const std::string &to_find, const std::string &str, size_t pos) {
  pos = str.find(to_find, pos);
  pos = str.find("'value': True", pos);
  return (pos != std::string::npos);
}

std::list<Node::attributes> FindAttributes(std::vector<std::string> attrs) {
  std::list<Node::attributes> att_l;
  std::string name = "'name':";
  size_t name_len = name.length();
  for (std::string str : attrs) {
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
