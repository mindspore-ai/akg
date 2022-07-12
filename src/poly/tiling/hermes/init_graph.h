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
#ifndef POLY_TILING_HERMES_INIT_GRAPH_H_
#define POLY_TILING_HERMES_INIT_GRAPH_H_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "poly/tiling/hermes/node.h"

namespace akg {
namespace ir {
namespace poly {
class InitGraph {
 public:
  InitGraph(const std::string &, std::vector<std::shared_ptr<Node>> &, const std::vector<std::shared_ptr<Node>> &,
            const std::vector<std::shared_ptr<Node>> &);
  explicit InitGraph(const std::vector<std::shared_ptr<akg::ir::poly::Node>> &);
  InitGraph() = default;

  Op::OpCategory OperatorCategory();
  void RemoveNameless();
  void AddNodesName(std::vector<std::string>);
  std::string ToString();

  std::string name_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::vector<std::shared_ptr<Node>> inputs_;
  std::vector<std::shared_ptr<Node>> outputs_;

 private:
  void SetInputNodes();
  void SetOutputNodes();
  static void SetConstantNodes(const std::vector<std::shared_ptr<Node>> &nodes,
                               const std::vector<std::shared_ptr<Node>> &inputs);
  static std::set<int> UselessInput(const std::vector<std::shared_ptr<Node>> &inputs,
                                    const std::vector<std::shared_ptr<Node>> &nodes, std::set<int> to_remove);
  static void FixGraph(std::vector<std::shared_ptr<Node>> nodes, size_t zombie_id);
  static int IdOfNodeName(const std::string &name, const std::vector<std::shared_ptr<Node>> &nodes);
  static std::set<std::string> GetIntermediateOutputsNames(const std::vector<std::string> &names);
  static std::vector<std::shared_ptr<Node>> GetInputs(std::set<std::string> intermed_output_names,
                                                      const std::vector<std::shared_ptr<Node>> &nodes);
  static std::string FindName(std::vector<std::string> names, std::shared_ptr<Node> node);
  static bool HasConstantInput(const std::shared_ptr<Node> &node);
  static void FilterNames(std::vector<std::string> names, const std::string &out);

  template <typename T>
  std::vector<std::shared_ptr<T>> RefVecFromIdxVec(std::vector<int> indexes, std::vector<std::shared_ptr<T>> refs);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_INIT_GRAPH_H_
