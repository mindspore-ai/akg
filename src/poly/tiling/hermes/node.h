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
#ifndef POLY_TILING_HERMES_NODE_H_
#define POLY_TILING_HERMES_NODE_H_

#include <list>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/op.h"
#include "poly/tiling/hermes/tensor.h"

namespace akg {
namespace ir {
namespace poly {
class Node {
 public:
  enum class attributes {
    Transpose_A,
    Transpose_B,
  };

  Node();
  explicit Node(const std::shared_ptr<Node> &);
  Node(const std::string &, const Op &, const std::vector<std::shared_ptr<Tensor>> &,
       const std::vector<std::shared_ptr<Tensor>> &, const std::vector<std::shared_ptr<Node>> &,
       const std::vector<std::shared_ptr<Node>> &, const std::vector<Axis> &, const std::vector<Tensor> &,
       const std::map<std::string, std::map<Tensor, int64_t>> &, const std::list<attributes> &);

  static void SetNodesAxisDim(std::vector<std::shared_ptr<Node>> &nodes);
  bool HasName() const;
  bool HasAxis(const Axis &);
  bool HasAttr(const std::string &attr) const;
  std::string ToString() const;

  std::string name_;
  std::string orig_name_;
  Op op_;
  std::vector<std::shared_ptr<Tensor>> output_tensors_;
  std::vector<std::shared_ptr<Tensor>> input_tensors_;
  std::vector<std::shared_ptr<Node>> succ_;
  std::vector<std::shared_ptr<Node>> pred_;
  std::vector<Axis> axis_of_node_;
  std::vector<Tensor> transformed_output_shape_;
  std::map<std::string, std::map<Tensor, int64_t>> axis_to_tensor_to_shape_id_map_;
  std::list<attributes> attrs_;
};

std::list<Node::attributes> FindAttributes(std::vector<std::string>);
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_NODE_H_
