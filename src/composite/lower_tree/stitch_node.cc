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

#include "composite/lower_tree/stitch_node.h"
#include "picojson.h"
#include "composite/lower_tree/json_leaf.h"
#include "composite/lower_tree/sync_process.h"
#include "composite/parser.h"
#include "composite/extract_build_info.h"

namespace akg {
namespace lower {
namespace {
constexpr auto kIdx = "part_idx";
constexpr auto kEnableStitch = "enable_stitch_fusion";
constexpr auto kSharedMemTensors = "shared_memory_tensors";
constexpr auto kStitchOriginJson = "stitch_origin_json";

Stmt String2LowerStmtSimple(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool poly,
                            bool buffer_stitch, bool fold_dim, std::vector<size_t> &split_index) {
  CHECK(json_str);
  picojson::value v = String2Json(json_str->value);
  BuildInfo info;
  info.opt.stitch = buffer_stitch;
  info.opt.fold_dim = fold_dim;
  info.opt.enable_dump = false;
  ExtractBuildInfo(v, info);

  LowerData data = LowerDataNode::make(GetScheduleWithBuildInfo(info), info.args, info.in_binds, attrs, kCuda,
                                       info.kernel_name + "_check", GetConfig(), poly);
  auto node_ref = StageLower(data).RunTo(StageType::BeforeLowerFunc).Node();
  for (auto si : data->split_index) {
    split_index.push_back(int64_t(si));
  }
  return Downcast<Stmt>(node_ref);
}

Map<std::string, NodeRef> SetSharedMemoryTensors(const Map<std::string, NodeRef> &attrs, const BuildInfo &info,
                                                 const Map<std::string, Array<NodeRef>> &alloc_map) {
  Map<std::string, NodeRef> new_attrs = attrs;
  std::string shared_name;
  for (auto &output : info.output_names) {
    if (alloc_map.count(output)) {
      auto iter = std::find_if(
        info.opt.tensor_map.begin(), info.opt.tensor_map.end(),
        [&output](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == output; });
      CHECK(iter != info.opt.tensor_map.end()) << "output Tensor " << output << " not built.";
      LOG(INFO) << "output: " << output << " " << iter->second;
      shared_name += iter->second->op->func_name() + " ";
    }
  }
  new_attrs.Set(kSharedMemTensors, Expr(shared_name));
  return new_attrs;
}

void ModifyStitchInfo(Map<std::string, NodeRef> &attrs, Map<std::string, NodeRef> &forward_infos, BuildInfo *info) {
  if (forward_infos.find(kEnableStitch) == forward_infos.end()) {
    return;
  }

  CHECK(forward_infos.find(kIdx) != forward_infos.end());
  CHECK(forward_infos.find(kFoldDim) != forward_infos.end());
  info->opt.stitch = true;
  info->opt.stitch_ir_idx = static_cast<size_t>(ir::GetInt32Const(Downcast<Expr>(forward_infos[kIdx])));
  info->opt.fold_dim = GetBoolValueFromMap(forward_infos, kFoldDim);
}

void ModifyStitchAttrs(Map<std::string, NodeRef> &forward_infos, BuildInfo &info, Map<std::string, NodeRef> *attrs) {
  if (forward_infos.find(kEnableStitch) == forward_infos.end()) {
    return;
  }

  CHECK(forward_infos.find(kAllocMap) != forward_infos.end());
  auto alloc_map = Downcast<Map<std::string, Array<NodeRef>>>(forward_infos[kAllocMap]);
  *attrs = SetSharedMemoryTensors(*attrs, info, alloc_map);
}

void AttachStitchDecorator(const std::string &target, BaseLowerNode *child, size_t index,
               Map<std::string, NodeRef> &forward_infos, Map<std::string, NodeRef> *backward_infos) {
  auto func = [&target, index, &forward_infos, backward_infos](BaseLowerNode *node, LowerRunner *next, StageType s) {
    JsonLowerLeaf *leaf = static_cast<JsonLowerLeaf*>(node);
    auto &attrs = leaf->Attrs();
    ModifyStitchInfo(attrs, forward_infos, &leaf->info_);
    if (forward_infos.find(kExtraAttrs) != forward_infos.end()) {
      auto extra_attrs = Downcast<Map<std::string, NodeRef>>(forward_infos[kExtraAttrs]);
      for (auto attr : extra_attrs) {
        attrs.Set(attr.first, attr.second);
      }
    }
    next->Lower(s);
    leaf->Data()->name += std::string("_Stitch_")  + std::to_string(index);
    if (target == "cuda") {
      LowerDataNode *data = leaf->Data().operator->();
      ModifyStitchAttrs(forward_infos, leaf->info_, &data->attrs);
    }
  };
  child->VisitLeaf([&func](JsonLowerLeaf *node) { node->Decorate(func); });
}
}  // namespace

bool CheckFoldDim(const NodeRef &block_json) {
  std::vector<int> fold_index;
  for (auto &stitch_json : Downcast<Array<Expr>>(block_json)) {
    CHECK(stitch_json.as<StringImm>());
    picojson::value v = String2Json(stitch_json.as<StringImm>()->value);
    BuildInfo info;
    ExtractBuildInfo(v, info);
    if (info.opt.fold_dims_.empty()) {
      return false;
    }
    if (fold_index.empty()) {
      fold_index = info.opt.fold_dims_.begin()->second;
    }
    for (auto &kv : info.opt.fold_dims_) {
      if (kv.second != fold_index) {
        return false;
      }
    }
  }
  return true;
}

void CudaStitchLowerNode::Lower(StageType to) {
  CHECK(children_.size() > 1);
  // 0. check.
  // Catch all children's jsons.
  Array<Expr> block_jsons;
  std::vector<Map<std::string, NodeRef>> block_attrs;
  std::tie(block_jsons, block_attrs) = CatchChild();

  bool fold_dim;
  if (block_attrs[0].defined() && block_attrs[0].find(kFoldDim) != block_attrs[0].end()) {
    fold_dim = GetBoolValueFromMap(block_attrs[0], kFoldDim);
  } else {
    fold_dim = CheckFoldDim(block_jsons);
  }

  std::vector<GridBlockDims> dim_array;
  std::vector<size_t> split_index;
  std::vector<Stmt> stitch_irs;
  std::vector<LowerData> datas;

  ir_type_array_.clear();
  // 1. Run child.
  for (size_t i = 0; i < children_.size(); ++i) {
    auto &child = children_[i];
    auto forward_infos =
      GetStitchForwardInfo(block_jsons[i], block_attrs[i], i, fold_dim, dim_array, ir_type_array_, split_index);
    Map<std::string, NodeRef> backward_infos;
    AttachMultiChildDecorator(child.get(), forward_infos, &backward_infos);
    AttachStitchDecorator(target_, child.get(), i, forward_infos, &backward_infos);
    // Excute child.
    child->Run(this);
    UpdateMergeInfos(backward_infos);
    auto data = child->Data();
    CollectOutputMap(data, backward_infos, outputs2args_);
    for (const auto &x : data->arg_list_0) {
      all_args_.push_back(x);
    }
    datas.push_back(data);
    auto stitch_ir = Downcast<Stmt>(child->Node());
    stitch_irs.push_back(InsertSync(stitch_ir));
  }

  // 2. Merge datas and block irs.
  Merge(datas, stitch_irs);

  // 3. Run to.
  Postprocess(to);
}

Map<std::string, NodeRef> CudaStitchLowerNode::GetStitchForwardInfo(
  Expr child_json, const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim,
  std::vector<GridBlockDims> &dim_array, std::vector<StitchOpType> &ir_type_array, std::vector<size_t> &split_index) {
  auto forward_infos = GetCommonForwardInfo();
  forward_infos.Set(kAllocMap, alloc_map_);
  forward_infos.Set(kIdx, Expr(i));
  forward_infos.Set(kFoldDim, Expr(fold_dim));

  // New attrs.
  std::vector<OpDesc> op_v = ParseOpDesc(child_json.as<StringImm>()->value);
  auto kernel_name = ParseKernelName(child_json.as<StringImm>()->value);
  const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool, bool, bool,
                           std::vector<size_t> &)>
    f = std::bind(&String2LowerStmtSimple, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3,
                  std::placeholders::_4, std::placeholders::_5, std::placeholders::_6);
  BufferStitchAttr stitch_attr_info(f);
  stitch_attr_info.GetBufferStitchAttr(child_json, op_v, child_attrs, true, fold_dim);
  dim_array.push_back(stitch_attr_info.dims);  // save current dims into array.
  auto stitch_type = stitch_attr_info.stitch_type_;
  IrAttrInfo ir_attr_info = GetIRAttr(stitch_type, stitch_attr_info, ir_type_array, dim_array, child_attrs);
  DumpIRAttr(kernel_name, ir_attr_info, i);
  ir_type_array.push_back(stitch_type);  // Note this should be done AFTER GetIrAttr.
  auto new_attrs = BindBlockAndThread(ir_attr_info.dims, true, ir_attr_info.attrs);
  if (i == 0) split_index = stitch_attr_info.split_index;
  new_attrs = SetAutoFuseAttr(split_index, new_attrs);

  forward_infos.Set(kEnableStitch, Expr(true));
  forward_infos.Set(kExtraAttrs, new_attrs);

  return forward_infos;
}

Stmt CudaStitchLowerNode::MergeStmts(const LowerData &data, std::vector<Stmt> &stitch_irs) {
  auto dump_mng = DumpManager(data->name + "_merge", data->config->dump_pass_ir);
  DUMP_ORIGIN_IR(dump_mng, stitch_irs);

  StitchAttrInfo stitch_attr;
  stitch_attr.type_array = ir_type_array_;

  StitchBufAlloc buf_manager(stitch_irs, alloc_map_, reuse_map_, clean_op_map_, outputs2args_);
  buf_manager.BufferAllocReuse();

  GetRealOutputs();

  Stmt stitched_ir;
  TRANSFORM_AND_TRY_DUMP(dump_mng, stitched_ir, StitchFusionGpu, stitch_irs, data->name, stitch_attr,
                         buf_manager.stitch_buffer_map, buf_manager.buf_within_op_map, buf_manager.allocate_revoke,
                         real_outputs_, workspace_args_);

  auto ElimDupInputs = [](Stmt stmt, const Array<NodeRef> &inputs) { return ElimDuplicateInputs(inputs).Run(stmt); };
  TRANSFORM_AND_TRY_DUMP(dump_mng, stitched_ir, ElimDupInputs, stitched_ir, inputs_);
  TRANSFORM_AND_TRY_DUMP(dump_mng, stitched_ir, ir::ProcessSyncInnerThread, stitched_ir);
  TRANSFORM_AND_TRY_DUMP(dump_mng, stitched_ir, ElimDupInputs, stitched_ir, inputs_);

  return stitched_ir;
}

void CudaStitchLowerNode::PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) {
  data->arg_list_0 = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_, workspace_args_);
}

void AscendStitchLowerNode::Lower(StageType to) {
  CHECK(children_.size() > 1);

  // 0. check.
  // Catch all children's jsons.
  bool fold_dim;
  std::vector<Map<std::string, NodeRef>> block_attrs;
  {
    Array<Expr> block_jsons;
    std::tie(block_jsons, block_attrs) = CatchChild();

    Map<std::string, NodeRef> &attrs = block_attrs[0];

    if (attrs.defined() && attrs.find(kFoldDim) != attrs.end()) {
      fold_dim = GetBoolValueFromMap(attrs, kFoldDim);
    } else {
      fold_dim = CheckFoldDim(block_jsons);
    }
  }

  std::vector<Stmt> stitch_irs;
  std::vector<LowerData> datas;
  // 1. Run child.
  for (size_t i = 0; i < children_.size(); ++i) {
    auto &child = children_[i];
    auto forward_infos = GetStitchForwardInfo(block_attrs[i], i, fold_dim);
    Map<std::string, NodeRef> backward_infos;
    AttachMultiChildDecorator(child.get(), forward_infos, &backward_infos);
    AttachStitchDecorator(target_, child.get(), i, forward_infos, &backward_infos);
    // Excute child.
    child->Run(this);
    UpdateMergeInfos(backward_infos);
    auto data = child->Data();
    CollectOutputMap(data, backward_infos, outputs2args_);
    for (const auto &x : data->arg_list_0) {
      all_args_.push_back(x);
    }
    datas.push_back(data);

    auto stitch_ir = Downcast<Stmt>(child->Node());
    stitch_irs.push_back(std::move(stitch_ir));
    auto split = Evaluate::make(Expr("===========split=========="));
    stitch_irs.push_back(std::move(split));
  }

  // 2. Merge datas and block irs.
  Merge(datas, stitch_irs);

  // 3. Run to.
  Postprocess(to);
}

Map<std::string, NodeRef> AscendStitchLowerNode::GetStitchForwardInfo(const Map<std::string, NodeRef> &child_attrs,
                                                                      size_t i, bool fold_dim) {
  auto forward_infos = GetCommonForwardInfo();
  forward_infos.Set(kIdx, Expr(i));
  forward_infos.Set(kFoldDim, Expr(fold_dim));

  auto peeled_tensors = GetOriginPeelInfo(stitch_origin_json_, child_attrs, fold_dim);
  forward_infos.Set(kPeeledTensors, PeelingToNodeRef(peeled_tensors));

  // Set compile attr for current split json
  Map<std::string, NodeRef> new_attrs;
  new_attrs.Set("enable_multicore", make_const(Int(32), 0));
  auto tiling_idx = "sub_attr_" + std::to_string(i + 1);
  for (const auto &it : child_attrs) {
    if (it.first != tiling_idx) {
      new_attrs.Set(it.first, it.second);
    } else {
      if (it.second.as<StrMapNode>() == nullptr) {
        continue;
      }
      auto tiling = Downcast<Map<std::string, NodeRef>>(it.second);
      for (const auto &t : tiling) {
        new_attrs.Set(t.first, t.second);
      }
    }
  }

  forward_infos.Set(kEnableStitch, Expr(true));
  forward_infos.Set(kExtraAttrs, new_attrs);

  return forward_infos;
}

Stmt AscendStitchLowerNode::AddPeelInfoForLoopAndData(Stmt &s, LowerData &data, Map<std::string, NodeRef> &attrs) {
  PeelInfo peel_info = GetPeelInfoFromAttrs(attrs);
  peel_info.CollectRealPeelTensors(data->arg_list_0, outputs2args_);
  AddPeelInfoForData(data, peel_info);
  s = AddPeelInfoForLoop(peel_info, data->binds_0).Run(s);
  DumpStmt2File("stitch_info/" + data_->name + "_after_add_loop.cc", s);
  return s;
}

std::unordered_map<std::string, NodeRef> AscendStitchLowerNode::GetStitchBuffer(
  const Map<std::string, Array<NodeRef>> &alloc_map) {
  std::unordered_map<std::string, NodeRef> stitch_buffer;
  for (auto &kv : outputs2args_) {
    if (alloc_map.count(kv.first)) {
      stitch_buffer.insert(kv);
    }
  }
  return stitch_buffer;
}

Map<Tensor, Buffer> AscendStitchLowerNode::FixBinds(const Map<Tensor, Buffer> &origin_binds,
                                                    const Array<NodeRef> &ordered_args) {
  Map<Tensor, Buffer> new_binds;
  for (auto &arg : ordered_args) {
    for (auto &kv : origin_binds) {
      if (kv.second == arg) {
        new_binds.Set(kv.first, kv.second);
      }
    }
  }
  return new_binds;
}

Stmt AscendStitchLowerNode::MergeStmts(const LowerData &data, std::vector<Stmt> &stitch_irs) {
  auto dump_mng = DumpManager(data->name + "_merge", data->config->dump_pass_ir);
  DUMP_ORIGIN_IR(dump_mng, stitch_irs);

  auto stitch_buffer = GetStitchBuffer(alloc_map_);
  GetRealOutputs();

  Stmt merged_ir;
  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, StitchFusionAscend, stitch_irs, data->name, stitch_buffer, real_outputs_,
                         workspace_args_, workspace_binds_);
  return merged_ir;
}

void AscendStitchLowerNode::PostUpdateDataAndNodeRef(LowerData &data, NodeRef &node_ref) {
  // Fix LowerData for stitch.
  Array<NodeRef> ordered_args = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_, workspace_args_);
  data->arg_list_0 = ordered_args;
  data->binds_0 = FixBinds(data->binds_0, ordered_args);
  // Add workspace tensors to binds
  for (const auto &it : workspace_binds_) {
    data->binds_0.Set(it.first, it.second);
  }

  CHECK(node_ref->IsInstance<Stmt::ContainerType>());
  auto stmt = Downcast<Stmt>(node_ref);
  node_ref = AddPeelInfoForLoopAndData(stmt, data, data->attrs);

  data->attrs.Set(kEnableMulticore, Expr(1));
}

BaseLowerNodePtr CreateCudaStitchLowerNode(const std::string &target, bool,
                                           const Map<std::string, NodeRef> &construct_infos) {
  CHECK(construct_infos.find(kAllocMap) != construct_infos.end());
  CHECK(construct_infos.find(kReuseMap) != construct_infos.end());
  CHECK(construct_infos.find(kCleanOpMap) != construct_infos.end());
  CHECK(construct_infos.find(kKernelInputs) != construct_infos.end());
  CHECK(construct_infos.find(kKernelOutputs) != construct_infos.end());
  return std::make_shared<CudaStitchLowerNode>(
    target, Downcast<Array<NodeRef>>(construct_infos[kKernelInputs]),
    Downcast<Array<NodeRef>>(construct_infos[kKernelOutputs]),
    Downcast<Map<std::string, Array<NodeRef>>>(construct_infos[kAllocMap]),
    Downcast<Map<std::string, Array<NodeRef>>>(construct_infos[kReuseMap]),
    Downcast<Map<std::string, Array<NodeRef>>>(construct_infos[kCleanOpMap]));
}

BaseLowerNodePtr CreateAscendStitchLowerNode(const std::string &target, bool,
                                             const Map<std::string, NodeRef> &construct_infos) {
  CHECK(construct_infos.find(kStitchOriginJson) != construct_infos.end());
  CHECK(construct_infos.find(kAllocMap) != construct_infos.end());
  CHECK(construct_infos.find(kKernelInputs) != construct_infos.end());
  CHECK(construct_infos.find(kKernelOutputs) != construct_infos.end());
  return std::make_shared<AscendStitchLowerNode>(
    target, Downcast<Array<NodeRef>>(construct_infos[kKernelInputs]),
    Downcast<Array<NodeRef>>(construct_infos[kKernelOutputs]),
    construct_infos[kStitchOriginJson].as<StringImm>()->value,
    Downcast<Map<std::string, Array<NodeRef>>>(construct_infos[kAllocMap]));
}

REG_NODE_CREATOR(kCuda, kStitch, CreateCudaStitchLowerNode);
REG_NODE_CREATOR(kCce, kStitch, CreateAscendStitchLowerNode);
}  // namespace lower

TVM_REGISTER_GLOBAL("check_fold_dim").set_body_typed(lower::CheckFoldDim);
}  // namespace akg
