/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "composite/lower_tree/multichild_node.h"
#include "composite/lower_tree/json_leaf.h"
#include "composite/lower_tree/stitch_fusion.h"
#include "composite/lower_tree/sync_process.h"
#include "composite/extract_build_info.h"

namespace akg {
namespace lower {
namespace {
constexpr auto kEnableMultiChild = "enable_multi_child";
void ModifyInfoPeeling(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos, BuildInfo *info) {
  if (forward_infos.find(kEnableMultiChild) == forward_infos.end()) {
    return;
  }

  if (attrs.find(kPeeling) != attrs.end()) {
    if (forward_infos.find(kPeeledTensors) != forward_infos.end()) {
      auto peeled_tensors = NodeRefToPeeling(Downcast<Map<std::string, Array<NodeRef>>>(forward_infos[kPeeledTensors]));
      info->opt.peel_info.SetPeelTensors(peeled_tensors);
    }
    auto peeling = attrs[kPeeling].as<StringImm>();
    CHECK(peeling != nullptr);
    info->opt.peel_info.peeling = peeling->value;
  }
}

void ModifyBackwardNames(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &attrs, BuildInfo &info,
                         Map<std::string, NodeRef> *backward_infos) {
  if (forward_infos.find(kEnableMultiChild) == forward_infos.end()) {
    return;
  }

  Array<Expr> input_names;
  for (auto input : info.input_names) {
    input_names.push_back(input);
  }
  backward_infos->Set(kInputNames, input_names);
  Array<Expr> output_names;
  for (auto output : info.output_names) {
    output_names.push_back(output);
  }
  backward_infos->Set(kOutputNames, output_names);
}

void ModifyBackwardPeeling(Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> &attrs, BuildInfo &info,
                           Map<std::string, NodeRef> *backward_infos) {
  if (forward_infos.find(kEnableMultiChild) == forward_infos.end()) {
    return;
  }

  if (attrs.find(kPeeling) != attrs.end()) {
    auto peeled_tensors = info.opt.peel_info.GetPeelTensors();
    backward_infos->Set(kPeeledTensors, PeelingToNodeRef(peeled_tensors));
  }
}

std::pair<bool, int64_t> GetTensorPeel(size_t i, const std::vector<std::pair<int, int64_t>> &dims) {
  for (auto dim : dims) {
    if (i == static_cast<size_t>(dim.first)) {
      return {true, dim.second};
    }
  }
  return {false, 1};
}
}  // namespace

Stmt PeelInfoMutator::Run(Stmt &s) {
  for (const auto &it : extern_buffer_) {
    s = ir::TensorStringSubstitute(s, it.second->name, it.first->op, it.first->value_index);
  }
  s = this->Mutate(s);
  s = ExtraModify(s);
  return s;
}

Expr PeelInfoMutator::Mutate_(const Call *op, const Expr &e) {
  if (op->func.defined() && peel_info_.IsPeeledTensor(op->func->func_name())) {
    return Call::make(op->type, op->name, FixArgs(op->args, op->name), op->call_type, op->func, op->value_index);
  }
  return IRMutator::Mutate_(op, e);
}
Stmt PeelInfoMutator::Mutate_(const Provide *op, const Stmt &s) {
  if (peel_info_.IsPeeledTensor(op->func->func_name())) {
    return Provide::make(op->func, op->value_index, this->Mutate(op->value), FixArgs(op->args, op->func->func_name()));
  }
  return IRMutator::Mutate_(op, s);
}

Stmt AddPeelInfoForLoop::ExtraModify(Stmt &s) {
  for (auto it = loop_var_.rbegin(); it != loop_var_.rend(); ++it) {
    s = For::make(it->second, 0, peel_info_.peels[it->first], ForType::Serial, DeviceAPI::None, s);
  }
  return s;
}

Array<Expr> AddPeelInfoForLoop::FixArgs(const Array<Expr> &args, const std::string &name) {
  auto dim = peel_info_.Getdim(name);
  Array<Expr> new_args;
  for (int i = 0; i < static_cast<int>(args.size()); ++i) {
    auto peel_tensor = GetTensorPeel(i, dim);
    if (peel_tensor.first) {
      new_args.push_back(loop_var_[i]);
    }
    new_args.push_back(args[i]);
  }
  return new_args;
}

Stmt AddInnerForAndBlockInfo::ExtraModify(Stmt &s) {
  // Add inner For.
  s = For::make(loop_var_, 0, inner_size_, ForType::Serial, DeviceAPI::None, s);

  // Add block info.
  Expr block_ext = make_const(Int(32), block_dim_);
  IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), block_ext), block_var_,
                                       air::IterVarType::kThreadIndex, BLOCK_IDX_X);
  s = AttrStmt::make(block_iv, air::ir::attr::thread_extent, block_ext, s);

  return s;
}

Array<Expr> AddInnerForAndBlockInfo::FixArgs(const Array<Expr> &args, const std::string &name) {
  auto dim = peel_info_.Getdim(name);
  Array<Expr> new_args;
  for (int i = 0; i < static_cast<int>(args.size()); ++i) {
    auto peel_tensor = GetTensorPeel(i, dim);
    if (peel_tensor.first) {
      new_args.push_back(offset_);
    }
    new_args.push_back(args[i]);
  }
  return new_args;
}

std::unordered_map<std::string, Peeling> GetOriginPeelInfo(const std::string &stitch_origin_json,
                                                           const Map<std::string, NodeRef> &attrs, bool fold_dim) {
  picojson::value v = String2Json(stitch_origin_json);
  BuildInfo info;
  info.opt.fold_dim = fold_dim;
  if (attrs.find(kPeeling) != attrs.end()) {
    auto peeling = attrs[kPeeling].as<StringImm>();
    CHECK(peeling != nullptr);
    info.opt.peel_info.peeling = peeling->value;
  }
  ExtractBuildInfo(v, info);
  return info.opt.peel_info.GetPeelTensors();
}

Map<std::string, Array<NodeRef>> PeelingToNodeRef(const std::unordered_map<std::string, Peeling> &peeled_tensors) {
  // Map<std::string, Array<Array<Expr>>>
  Map<std::string, Array<NodeRef>> peel_noderef;
  for (auto peel_tensor : peeled_tensors) {
    Array<NodeRef> peel_node;
    for (auto peel : peel_tensor.second) {
      Array<Expr> item;
      item.push_back(peel.first);
      item.push_back(peel.second);
      peel_node.push_back(item);
    }
    peel_noderef.Set(peel_tensor.first, peel_node);
  }

  return peel_noderef;
}

std::unordered_map<std::string, Peeling> NodeRefToPeeling(const Map<std::string, Array<NodeRef>> &peeled_noderef) {
  // Map<std::string, Array<Array<Expr>>>
  std::unordered_map<std::string, Peeling> peeled_tensors;
  for (auto peel_node : peeled_noderef) {
    Peeling peel_tensor;
    for (auto peel : peel_node.second) {
      auto peel_array = Downcast<Array<NodeRef>>(peel);
      std::pair<int, int64_t> item;
      item.first = static_cast<int>(ir::GetInt32Const(Downcast<Expr>(peel_array[0])));
      item.second = static_cast<int64_t>(ir::GetInt32Const(Downcast<Expr>(peel_array[1])));
      peel_tensor.push_back(item);
    }
    peeled_tensors.insert({peel_node.first, peel_tensor});
  }

  return peeled_tensors;
}

void MultiChildLowerNode::Merge(const std::vector<LowerData> &datas, std::vector<Stmt> &block_irs) {
  data_ = MergeDatas(datas);
  node_ref_ = MergeStmts(data_, block_irs);
  // Pass inputs' name and outputs' name as sub lower names.
  PassNamesOut();
  PostUpdateDataAndNodeRef(data_, node_ref_);
}

LowerData MultiChildLowerNode::MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified) {
  // Not merge sch, simple_mode, tuning, split_index.

  bool all_merge = specified.empty();
  auto merge_data = LowerDataNode::make();
  for (size_t idx = 0; idx < datas.size(); ++idx) {
    auto &data = datas[idx];
    if (!all_merge && specified.count(idx) == 0) continue;
    for (auto arg : data->args) {
      merge_data->args.push_back(arg);
    }
    for (auto arg_list : data->arg_list_0) {
      merge_data->arg_list_0.push_back(arg_list);
    }
    for (auto iter : data->attrs) {
      merge_data->attrs.Set(iter.first, iter.second);
    }
    for (auto iter : data->binds) {
      merge_data->binds.Set(iter.first, iter.second);
    }
    for (auto iter : data->binds_0) {
      merge_data->binds_0.Set(iter.first, iter.second);
    }
    for (auto shape_var : data->shape_vars) {
      merge_data->shape_vars.push_back(shape_var);
    }
  }

  merge_data->config = datas[0]->config;
  merge_data->name = datas[0]->name;
  if (merge_data->attrs.find(kOriginKernelName) != merge_data->attrs.end()) {
    CHECK(merge_data->attrs[kOriginKernelName]->IsInstance<StringImm>());
    std::string kernel_name = merge_data->attrs[kOriginKernelName].as<StringImm>()->value;
    if (forward_infos_.find(kKernelNamePosfix) != forward_infos_.end()) {
      kernel_name += "_" + forward_infos_[kKernelNamePosfix].as<StringImm>()->value;
    }
    merge_data->name = kernel_name;
  }

  merge_data->polyhedral = datas[0]->polyhedral;
  merge_data->target = datas[0]->target;
  return merge_data;
}

void MultiChildLowerNode::Postprocess(StageType to) {
  if (!StageTypeGT(data_->target, to, entrance_stage_)) {
    return;
  }
  StageLower stage_lower(data_, node_ref_, StageManager::Instance().NextStageType(data_->target, entrance_stage_));
  stage_lower.RunTo(to);
  node_ref_ = stage_lower.Node();
  data_ = stage_lower.Data();
  current_stage_ = to;
}

void MultiChildLowerNode::CollectOutputMap(const LowerData &data, const Map<std::string, NodeRef> &backward_info,
                                           std::unordered_map<std::string, NodeRef> &outputs2args) {
  std::vector<std::string> input_names;
  CHECK(backward_info.find(kInputNames) != backward_info.end());
  auto input_name_exprs = Downcast<Array<Expr>>(backward_info[kInputNames]);
  for (const auto &name : input_name_exprs) {
    auto it = name.as<StringImm>();
    CHECK(it != nullptr);
    input_names.push_back(it->value);
  }
  std::vector<std::string> output_names;
  CHECK(backward_info.find(kOutputNames) != backward_info.end());
  auto output_name_exprs = Downcast<Array<Expr>>(backward_info[kOutputNames]);
  for (const auto &name : output_name_exprs) {
    auto it = name.as<StringImm>();
    CHECK(it != nullptr);
    output_names.push_back(it->value);
  }

  size_t count = 0;
  for (const auto &x : data->arg_list_0) {
    auto buffer = x.as<BufferNode>();
    CHECK(buffer) << "Arg must be a BufferNode";
    if (std::find(input_names.begin(), input_names.end(), buffer->name) == std::end(input_names)) {
      CHECK(count < output_names.size());
      outputs2args[output_names[count]] = x;
      count++;
    }
  }
}

Array<NodeRef> MultiChildLowerNode::ReorderArgs(const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                                                const Array<NodeRef> &all_args,
                                                std::unordered_map<std::string, NodeRef> &outputs2args,
                                                const Array<NodeRef> &workspace) {
  // reorder args_list, now args_list satisfies: op1_input op2_input ... op1_output op2_output ... op1_workspace ...
  // suppose all args info from original json satisfies this order
  Array<NodeRef> input_args, ordered_args;
  std::map<std::string, std::vector<NodeRef>> vmap;
  std::vector<std::string> inputs_name = GetNames(inputs);
  std::vector<std::string> outputs_name = GetNames(outputs);
  for (auto arg : all_args) {
    auto buffer = arg.as<BufferNode>();
    CHECK(buffer) << "arg must be a BufferNode";
    if (std::find(inputs_name.begin(), inputs_name.end(), buffer->name) != std::end(inputs_name)) {
      if (vmap.find(buffer->name) == vmap.end()) {
        input_args.push_back(arg);
        vmap[buffer->name] = {};
      }
      vmap[buffer->name].push_back(arg);
    }
  }
  // input_args is not ordered as args list, should make it first.
  for (const auto &input : inputs_name) {
    for (const auto &arg : input_args) {
      if (arg.as<BufferNode>()->name == input) {
        ordered_args.push_back(arg);
        break;
      }
    }
  }
  // output args keep order as origin output
  for (const auto &output : outputs_name) {
    if (outputs2args.find(output) != outputs2args.end()) {
      ordered_args.push_back(outputs2args[output]);
    }
  }
  // workspace follows after inputs and outputs
  for (const auto &it : workspace) {
    ordered_args.push_back(it);
  }

  return ordered_args;
}

void MultiChildLowerNode::GetRealOutputs() {
  auto outputs_name = GetNames(outputs_);
  for (const auto &output : outputs_name) {
    if (outputs2args_.find(output) != outputs2args_.end()) {
      real_outputs_[output] = outputs2args_[output];
    }
  }
}

std::pair<Array<Expr>, std::vector<Map<std::string, NodeRef>>> MultiChildLowerNode::CatchChild() {
  Array<Expr> block_jsons;
  std::vector<Map<std::string, NodeRef>> block_attrs;
  for (auto child : children_) {
    Map<std::string, NodeRef> forward_infos;
    forward_infos.Set(kCatch, Expr("JsonLowerLeaf"));
    Excute(child, forward_infos, true, false);
    block_jsons.push_back(Downcast<Expr>(child->BackwardInfos()[kBlockJsons]));
    block_attrs.push_back(Downcast<Map<std::string, NodeRef>>(child->BackwardInfos()[kBlockAttrs]));
  }

  return {block_jsons, block_attrs};
}

void MultiChildLowerNode::AddPeelInfoForData(LowerData &data, PeelInfo &peel_info) {
  Array<NodeRef> out_args;
  Map<Tensor, Buffer> out_binds;
  for (const auto &kv : data->binds_0) {
    if (!peel_info.IsPeeledTensor(kv.second)) {
      out_binds.Set(kv.first, kv.second);
    }
  }

  Map<Buffer, Buffer> buffer_replace;
  for (const auto &x : data->arg_list_0) {
    if (x->IsInstance<BufferNode>() && peel_info.IsPeeledTensor(x)) {
      auto dim = peel_info.Getdim(x);
      auto old_shape = x.as<BufferNode>()->shape;
      Array<Expr> new_shape;
      for (int i = 0; i < static_cast<int>(old_shape.size()); ++i) {
        auto peel_tensor = GetTensorPeel(i, dim);
        if (peel_tensor.first) {
          new_shape.push_back(static_cast<int>(peel_tensor.second));
        }
        new_shape.push_back(old_shape[i]);
      }
      auto config = GetConfig();
      Tensor tt = air::placeholder(new_shape, x.as<BufferNode>()->dtype, x.as<BufferNode>()->name);
      auto buf = DeclBuffer(tt, config->data_alignment, config->offset_factor);
      out_args.push_back(buf);
      out_binds.Set(tt, buf);
      buffer_replace.Set(GetRef<Buffer>(x.as<BufferNode>()), buf);
    } else {
      out_args.push_back(x);
    }
  }
  data->arg_list_0 = out_args;
  data->binds_0 = out_binds;
  ReplaceBufferForALLArgsAndOutputs2args(buffer_replace);
}

void MultiChildLowerNode::ReplaceBufferForALLArgsAndOutputs2args(Map<Buffer, Buffer> &buffer_replace) {
  Array<NodeRef> new_args;
  for (auto &arg : all_args_) {
    CHECK(arg->IsInstance<BufferNode>());
    Buffer buffer_node = GetRef<Buffer>(arg.as<BufferNode>());
    if (buffer_replace.count(buffer_node)) {
      new_args.push_back(buffer_replace[buffer_node]);
    } else {
      new_args.push_back(arg);
    }
  }
  all_args_ = new_args;
  for (auto &kv : outputs2args_) {
    Buffer buffer_node = GetRef<Buffer>(kv.second.as<BufferNode>());
    if (buffer_replace.count(buffer_node)) {
      outputs2args_[kv.first] = buffer_replace[buffer_node];
    }
  }
}

PeelInfo MultiChildLowerNode::GetPeelInfoFromAttrs(const Map<std::string, NodeRef> &attrs) {
  PeelInfo peel_info;
  if (attrs.find(kPeeling) != attrs.end()) {
    auto peeling = attrs[kPeeling].as<StringImm>();
    CHECK(peeling);
    auto parsed_peeling = Str2Peeling(peeling->value);
    CHECK(!parsed_peeling.empty());
    peel_info.SetPeels(parsed_peeling);
    CHECK(backward_infos_.find(kPeeledTensors) != backward_infos_.end());
    auto peeled_tensors = NodeRefToPeeling(Downcast<Map<std::string, Array<NodeRef>>>(backward_infos_[kPeeledTensors]));
    peel_info.SetPeelTensors(peeled_tensors);
  }
  return peel_info;
}

void MultiChildLowerNode::PassNamesOut() {
  backward_infos_.Set(kInputNames, inputs_);
  backward_infos_.Set(kOutputNames, outputs_);
}

Map<std::string, NodeRef> MultiChildLowerNode::GetCommonForwardInfo() {
  Map<std::string, NodeRef> forward_infos;
  forward_infos.Set(kEnableMultiChild, Expr(true));
  return forward_infos;
}

REG_INFO_FUNC_BEFORE(kCce, "MultiChildLowerNode", ModifyInfoPeeling);
REG_BACKWARD_FUNC(kCuda, "MultiChildLowerNode", ModifyBackwardNames);
REG_BACKWARD_FUNC(kCce, "MultiChildLowerNode", ModifyBackwardNames);
REG_BACKWARD_FUNC(kCce, "MultiChildLowerNode", ModifyBackwardPeeling);
}  // namespace lower
}  // namespace akg
