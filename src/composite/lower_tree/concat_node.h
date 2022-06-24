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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_CONCAT_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_CONCAT_NODE_H_

#include "composite/lower_tree/pass_through_node.h"

namespace akg {
namespace lower {
class ConcatLowerNode : public PassThroughLowerNode{
 public:
  explicit ConcatLowerNode(const std::string &target, 
                          const std::string &kernel_name,
                          const std::vector<std::string> &input_tensor_names, 
                          const std::vector<std::string> &output_tensor_names,
                          const Array<NodeRef> &input_tensor_shapes, 
                          const Array<NodeRef> &output_tensor_shapes,
                          const std::vector<std::vector<int>> concat_shapes)
            : PassThroughLowerNode(target, 
                                  kernel_name, 
                                  input_tensor_names, 
                                  output_tensor_names, 
                                  input_tensor_shapes, 
                                  output_tensor_shapes){
    entrance_stage_ = StageType::Flattern;
    // entrance_stage_ = StageType::Poly;
    name_ = __FUNCTION__;
    concat_shapes_ = concat_shapes;                           
  }
    ~ConcatLowerNode() override = default;
 protected:
  virtual std::vector<Stmt> ModifyChildrenStmts(const std::vector<Stmt> &stmts) override;
  virtual LowerData MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified) override;
 private:
  std::vector<std::vector<int>> concat_shapes_;
};
} // namespace lower
} // akg
#endif