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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_PASS_THROUGH_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_PASS_THROUGH_NODE_H_

#include "composite/lower_tree/multichild_node.h"

namespace akg {
namespace lower {
class PassThroughLowerNode : public MultiChildLowerNode {
 public:
  explicit PassThroughLowerNode(const std::string &target, 
                          const std::string &kernel_name,
                          const std::vector<std::string> &input_tensor_names, 
                          const std::vector<std::string> &output_tensor_names,
                          const Array<NodeRef> &input_tensor_shapes, 
                          const Array<NodeRef> &output_tensor_shapes)
                          : MultiChildLowerNode(target, Array<NodeRef>(), Array<NodeRef>()){
    entrance_stage_ = StageType::Poly; 
    name_ = __FUNCTION__;
    kernel_name_ = kernel_name;
    input_tensor_names_ = input_tensor_names;
    output_tensor_names_ = output_tensor_names;
    input_tensor_shapes_ = input_tensor_shapes;
    output_tensor_shapes_ = output_tensor_shapes;
}

  ~PassThroughLowerNode() override = default;

 protected:
  std::string kernel_name_;
  std::vector<std::string> input_tensor_names_;
  std::vector<std::string> output_tensor_names_;
  Array<NodeRef> input_tensor_shapes_;
  Array<NodeRef> output_tensor_shapes_;
  std::vector<std::vector<std::string>> target_ops_arg_names_;
  air::DataType dtype_;
  Operation output_op_;
  Buffer output_buffer_;
  Tensor output_tensor_;
  Array<Operation> input_ops_;
  Array<Buffer> input_buffers_;
  Array<Tensor> input_tensors_;

  virtual LowerData MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified = {}) override;
  virtual Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &stmts) override;
  virtual void Lower(StageType to) override;
  virtual std::vector<Stmt> ModifyChildrenStmts(const std::vector<Stmt> &stmts) = 0;
};
} // namespace lower
} // akg
#endif