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

#include "composite/lower_tree/concat_node.h"
#include "../third_party/incubator-tvm/include/tvm/expr_operator.h"
#include <vector>


namespace akg {
namespace ir{
namespace{
constexpr int  INT_BITS = 32;
class ReplaceConcatTensorAddrs : public IRMutator {
 public:
  ReplaceConcatTensorAddrs(const Buffer &output_buffer, 
                        const std::string &output_tensor_name, 
                        const std::vector<int> &args_for_transformation):
                            output_buffer_(output_buffer), 
                            output_tensor_name_(output_tensor_name), 
                            args_for_transformation_(args_for_transformation) {}
  ~ReplaceConcatTensorAddrs() override = default;

  Stmt Mutate_(const Store *op, const Stmt &s){
    // auto operation = op.as<StoreNode>();
    auto stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<Store>();
    if(op->buffer_var->name_hint == output_tensor_name_){
       // std::cout << "#zyy Store op stmt = " << stmt << std::endl;
      constexpr int TWO = 2;
      auto new_index = Div::make(op->index, make_const(Int(INT_BITS), args_for_transformation_[0])); 
      new_index = Mul::make(new_index, make_const(Int(INT_BITS), args_for_transformation_[1]));
      new_index = Add::make(new_index, make_const(Int(INT_BITS), args_for_transformation_[TWO]));
      new_index = Add::make(op->index, new_index);
      return Store::make(output_buffer_->data, op->value, new_index, op->predicate);
    }else{
      return stmt;
    }
  }
 private:
  Buffer output_buffer_;
  std::string output_tensor_name_;
  // [a, x, b] ~ [a, y, b] ~ [a, z, b] --> [a, x+y+z, b]
  // i --> i + i / (y*b) * (x+z) * b + x * b
  // args_for_transformation = {y*b, (x+z)*b, x*b}
  std::vector<int> args_for_transformation_;
}; 
}  // namespace

Stmt ModifyConcatTensorAddrs(const Stmt &stmt, 
                            const Buffer &output_buffer, 
                            const std::string &output_tensor_name, 
                            const std::vector<int> &args_for_transformation){
    auto inst = ir::ReplaceConcatTensorAddrs(output_buffer, output_tensor_name, args_for_transformation);
    auto s1 = inst.Mutate(stmt);
    return s1;
}
}  // namespace ir

namespace lower {
std::vector<Stmt> ConcatLowerNode::ModifyChildrenStmts(const std::vector<Stmt> &stmts){
    std::vector<Stmt> modified_stmts(stmts.size());
    std::vector<int> acc_of_2nd_dim(concat_shapes_.size() + 1, 0);
    for(size_t i=0; i<concat_shapes_.size(); ++i){
        acc_of_2nd_dim[i+1] = acc_of_2nd_dim[i] + concat_shapes_[i][1];
    }
    // [a, x, b] ~ [a, y, b] ~ [a, z, b] --> [a, x+y+z, b]
    // i --> i + i / (y*b) * (x+z) * b + x * b
    // args_for_transformation[i] = {y*b, (x+z)*b, x*b}
    constexpr int TWO = 2;
    int b = concat_shapes_[0][TWO];
    std::vector<std::vector<int>> args_for_transformation(concat_shapes_.size());
    for(size_t i=0; i<concat_shapes_.size(); ++i){
        args_for_transformation[i].push_back(concat_shapes_[i][1] * b);
        args_for_transformation[i].push_back((acc_of_2nd_dim.back() - concat_shapes_[i][1]) * b);
        args_for_transformation[i].push_back(acc_of_2nd_dim[i] * b);
    }
    for(size_t i=0; i<stmts.size(); ++i){
        modified_stmts[i] = ir::ModifyConcatTensorAddrs(stmts[i], 
                                                        output_buffer_, 
                                                        target_ops_arg_names_[i].back(), 
                                                        args_for_transformation[i]);
    }
    return modified_stmts;
}

LowerData ConcatLowerNode::MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified) {
    CHECK(!datas.empty());
    
    for(const auto &data: datas){
        std::vector<std::string> arg_names;
        for(const auto &arg: data->args){
            arg_names.push_back(arg.as<TensorNode>()->op->name);
        }
        
        target_ops_arg_names_.push_back(arg_names);
    }

    dtype_ = datas[0]->args[0].as<TensorNode>()->dtype;

    output_op_ = PlaceholderOpNode::make("pass_through_" + output_tensor_names_[0], 
                                        Downcast<Array<Expr>>(output_tensor_shapes_[0]), 
                                        dtype_);
    output_tensor_ = output_op_.output(0);
    output_buffer_ = decl_buffer(Downcast<Array<Expr>>(output_tensor_shapes_[0]), 
                                dtype_, 
                                output_tensor_names_[0] + "_pass_through");

    auto merge_data = LowerDataNode::make();
    for (size_t idx = 0; idx < datas.size(); ++idx) {
        auto &data = datas[idx];
        for (auto iter : data->attrs) {
            merge_data->attrs.Set(iter.first, iter.second);
        }
        // TODO: figure out the meaning of shape_var
        for (auto shape_var : data->shape_vars) {
        merge_data->shape_vars.push_back(shape_var);
        }

        // skip the output
        for(size_t i = 0; i < data->args.size() - 1; ++i){
            merge_data->args.push_back(data->args[i]);
        }
        for(size_t i = 0; i < data->arg_list_0.size() - 1; ++i){
            merge_data->arg_list_0.push_back(data->arg_list_0[i]);
        }

        for (auto iter : data->binds) {
            if(iter.first->op->name != target_ops_arg_names_[idx].back()){
                merge_data->binds.Set(iter.first, iter.second);
            }
        }
        for (auto iter : data->binds_0) {
            if(iter.first->op->name != target_ops_arg_names_[idx].back()){
                merge_data->binds_0.Set(iter.first, iter.second);
            }
        }
    }

    merge_data->args.push_back(output_tensor_);
    merge_data->arg_list_0.push_back(output_buffer_);
    merge_data->binds.Set(output_tensor_, output_buffer_);
    merge_data->binds_0.Set(output_tensor_, output_buffer_);

    merge_data->config = datas[0]->config;

    merge_data->polyhedral = datas[0]->polyhedral;
    merge_data->target = datas[0]->target;

    merge_data->name = kernel_name_;

    return merge_data;
}

BaseLowerNodePtr CreateConcatLowerNode(const std::string &target, bool, 
                                        const Map<std::string, NodeRef> &construct_infos) {
  std::vector<std::string> input_tensor_names;
  auto input_names_array = Downcast<Array<NodeRef>>(construct_infos[kKernelInputs]);
  for(size_t i=0; i<input_names_array.size(); ++i){
      input_tensor_names.push_back(input_names_array[i].as<StringImm>()->value);
  }

  std::vector<std::string> output_tensor_names;
  auto output_names_array = Downcast<Array<NodeRef>>(construct_infos[kKernelOutputs]);
  for(size_t i=0; i<output_names_array.size(); ++i){
      output_tensor_names.push_back(output_names_array[i].as<StringImm>()->value);
  }

  std::vector<std::vector<int>> concat_shapes;
  auto concat_shapes_array = Downcast<Array<NodeRef>>(construct_infos["concat_shapes"]);
  for(size_t i=0; i<concat_shapes_array.size(); i++){
    std::vector<int> concat_shape;
    auto shape_array = Downcast<Array<NodeRef>>(concat_shapes_array[i]);
    for(size_t j=0; j<shape_array.size(); ++j){
        concat_shape.push_back(shape_array[j].as<IntImm>()->value);
    }
    concat_shapes.push_back(concat_shape);
  }
    
  return std::make_shared<ConcatLowerNode>(target, 
                                            construct_infos["kernel_name"].as<StringImm>()->value,
                                            input_tensor_names,
                                            output_tensor_names,
                                            Downcast<Array<NodeRef>>(construct_infos["input_tensor_shapes"]),
                                            Downcast<Array<NodeRef>>(construct_infos["output_tensor_shapes"]),
                                            concat_shapes);
}

REG_NODE_CREATOR(kCuda, "PassThroughConcat", CreateConcatLowerNode);
} // namespace lower
} // namespace akg
