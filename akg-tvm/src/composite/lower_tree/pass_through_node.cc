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

#include "composite/lower_tree/pass_through_node.h"
#include <vector>

namespace akg {
namespace lower {
void PassThroughLowerNode::Lower(StageType to){
    // 1. Run child.
    std::vector<LowerData> datas;
    std::vector<Stmt> stmts;
    for (size_t i = 0; i < children_.size(); ++i){
        auto &child = children_[i];
        child->Run(this);
        auto data = child->Data();
        auto stmt = Downcast<Stmt>(child->Node());

        datas.push_back(data);
        stmts.push_back(stmt);
    }

    // 2. Merge datas and block irs.
    Merge(datas, stmts);

    // 3. Run to.
    Postprocess(to);
}

LowerData PassThroughLowerNode::MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified) {
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
                                        Downcast<Array<Expr>>(output_tensor_shapes_[0]), dtype_);
    output_tensor_ = output_op_.output(0);
    output_buffer_ = decl_buffer(Downcast<Array<Expr>>(output_tensor_shapes_[0]), 
                                dtype_, output_tensor_names_[0] + "_pass_through");

    for(size_t i=0; i<input_tensor_shapes_.size(); ++i){
        input_ops_.push_back(PlaceholderOpNode::make("pass_through_" + input_tensor_names_[i], 
                                                    Downcast<Array<Expr>>(input_tensor_shapes_[i]), 
                                                    dtype_));
        input_tensors_.push_back(input_ops_[i].output(0));
        input_buffers_.push_back(decl_buffer(Downcast<Array<Expr>>(input_tensor_shapes_[i]), 
                                            dtype_, input_tensor_names_[i] + "_pass_through"));
    }
    
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
    }

    for(const auto &in_tensor : input_tensors_){
        merge_data->args.push_back(in_tensor);
    }
    merge_data->args.push_back(output_tensor_);

    for(const auto &in_buffer : input_buffers_){
        merge_data->arg_list_0.push_back(in_buffer);
    }
    merge_data->arg_list_0.push_back(output_buffer_);

    for(size_t i=0; i<input_tensors_.size(); ++i){
        merge_data->binds.Set(input_tensors_[i], input_buffers_[i]);
        merge_data->binds_0.Set(input_tensors_[i], input_buffers_[i]);
    }
    merge_data->binds.Set(output_tensor_, output_buffer_);
    merge_data->binds_0.Set(output_tensor_, output_buffer_);

    merge_data->config = datas[0]->config;

    merge_data->polyhedral = datas[0]->polyhedral;
    merge_data->target = datas[0]->target;

    merge_data->name = kernel_name_;

    return merge_data;
}

Stmt PassThroughLowerNode::MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) {
    auto dump_mng = DumpManager(data->name, data->config->dump_pass_ir);
    DUMP_ORIGIN_IR(dump_mng, block_irs);

    std::vector<Stmt> modified_stmts(block_irs.size());
    TRANSFORM_AND_TRY_DUMP(dump_mng, modified_stmts, ModifyChildrenStmts, block_irs);
    Stmt merged_ir = Block::make(modified_stmts);
    return merged_ir;
}
}  // namespace : lower
} // namespace : akg