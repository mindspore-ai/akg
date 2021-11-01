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

#include "composite/lower_tree/parallel_node.h"

#include <vector>
#include <string>
#include <unordered_map>
#include "composite/utils/dimension_peeling.h"
#include "composite/lower_tree/block_fusion.h"
#include "composite/lower_tree/sync_process.h"

namespace akg {
namespace lower {
constexpr auto kBlockPlan = "block_plan";
void CudaParallelLowerNode::ExcuteImpl(StageType to) {
  CHECK(children_.size() > 1);
  std::vector<LowerData> datas;
  std::vector<Stmt> block_irs;
  // 1. Run child.
  for (size_t i = 0; i < children_.size(); ++i) {
    auto &child = children_[i];
    auto forward_infos = GetCommonForwardInfo();
    forward_infos = AddNamePosfix(kParallel, forward_infos_, i, true, forward_infos);

    Excute(child, forward_infos);
    auto data = child->Data();
    CollectOutputMap(data, backward_infos_, outputs2args_);
    for (const auto &x : data->arg_list_0) {
      all_args_.push_back(x);
    }
    datas.push_back(data);
    block_irs.push_back(Downcast<Stmt>(child->Node()));
  }

  // 2. Merge datas and block irs.
  Merge(datas, block_irs);

  // 3. Run with merge infos.
  Postprocess(to);
}

void CudaParallelLowerNode::PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) {
  data->arg_list_0 = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
}

Stmt CudaParallelLowerNode::MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) {
  auto dump_mng = DumpManager(data->name + "_merge", data->config->dump_pass_ir);
  DUMP_ORIGIN_IR(dump_mng, block_irs);

  Stmt merged_ir;
  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::BlockFusion, block_irs, target_);

  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::ProcessSyncInnerThread, merged_ir);
  auto ElimDupInputs = [](Stmt stmt, const Array<NodeRef> &inputs) { return ElimDuplicateInputs(inputs).Run(stmt); };
  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ElimDupInputs, merged_ir, inputs_);
  return merged_ir;
}

void AscendParallelLowerNode::ExcuteImpl(StageType to) {
  CHECK(children_.size() > 1);
  std::vector<LowerData> datas;
  std::vector<Stmt> block_irs;
  // 1. Run child.
  for (size_t i = 0; i < children_.size(); ++i) {
    auto &child = children_[i];

    // Catch child's attrs.
    Map<std::string, NodeRef> catch_forward_info;
    catch_forward_info.Set(kCatch, Expr("JsonLowerLeaf"));
    Excute(child, catch_forward_info, true, false);
    auto child_attrs = Downcast<Map<std::string, NodeRef>>(child->BackwardInfos()[kBlockAttrs]);

    LowerData block_data;
    NodeRef block_ir;
    Map<std::string, NodeRef> backward_infos;
    auto forward_infos = GetCommonForwardInfo();
    forward_infos = AddNamePosfix(kParallel, forward_infos_, i, true, forward_infos);
    if (child_attrs.find(kBlockPlan) != child_attrs.end()) {
      auto old_entrance_stage = entrance_stage_;
      entrance_stage_ = StageType::BeforeFlattern;

      Excute(child, forward_infos);
      auto data = child->Data();

      std::unordered_map<std::string, NodeRef> tmp_outputs2args;
      CollectOutputMap(data, backward_infos_, tmp_outputs2args);

      auto block_plan = child_attrs[kBlockPlan].as<IntImm>();
      CHECK(block_plan);
      int block = block_plan->value;
      PeelInfo peel_info = GetPeelInfoFromAttrs(child_attrs);

      StageLower stage_lower(data, child->Node(),
                             StageManager::Instance().NextStageType(data_->target, entrance_stage_));
      stage_lower.ApplyMutator(
        [this, &peel_info, &tmp_outputs2args, &block](NodeRef &node_ref, LowerData &data) -> NodeRef {
          auto stmt = Downcast<Stmt>(node_ref);
          stmt = AddPeelInfoAndBlockAttr(stmt, data, peel_info, tmp_outputs2args, block);
          return NEXT_PASS(CanonicalSimplify, stmt);
        });
      stage_lower.RunTo(old_entrance_stage);

      block_ir = stage_lower.Node();
      block_data = stage_lower.Data();
      entrance_stage_ = old_entrance_stage;
    } else {
      Excute(child, forward_infos);
      block_ir = child->Node();
      block_data = child->Data();
    }

    CollectOutputMap(block_data, backward_infos_, outputs2args_);
    for (const auto &x : block_data->arg_list_0) {
      all_args_.push_back(x);
    }
    datas.push_back(block_data);
    block_irs.push_back(Downcast<Stmt>(block_ir));
  }

  // 2. Merge datas and block irs.
  Merge(datas, block_irs);

  // 3. Run with merge infos.
  Postprocess(to);
}

Stmt AscendParallelLowerNode::AddPeelInfoAndBlockAttr(Stmt &s, LowerData &data, PeelInfo &peel_info,
                                                      std::unordered_map<std::string, NodeRef> &outputs2args,
                                                      int block) {
  peel_info.CollectRealPeelTensors(data->arg_list_0, outputs2args);
  AddPeelInfoForData(data, peel_info);
  s = AddInnerForAndBlockInfo(peel_info, block, data->binds_0).Run(s);
  return s;
}

Stmt AscendParallelLowerNode::MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) {
  auto dump_mng = DumpManager(data->name + "_merge", data->config->dump_pass_ir);
  DUMP_ORIGIN_IR(dump_mng, block_irs);
  Stmt merged_ir;
  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::BlockFusion, block_irs, target_);
  auto ElimDupInputs = [](Stmt &stmt, const Array<NodeRef> &inputs) { return ElimDuplicateInputs(inputs).Run(stmt); };
  TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ElimDupInputs, merged_ir, inputs_);
  return merged_ir;
}

void AscendParallelLowerNode::PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) {
  data->arg_list_0 = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
}

BaseLowerNodePtr CreateCudaParallelLowerNode(const std::string &target, bool,
                                             const Map<std::string, NodeRef> &construct_infos) {
  CHECK(construct_infos.find(kKernelInputs) != construct_infos.end());
  CHECK(construct_infos.find(kKernelOutputs) != construct_infos.end());
  return std::make_shared<CudaParallelLowerNode>(target, Downcast<Array<NodeRef>>(construct_infos[kKernelInputs]),
                                                 Downcast<Array<NodeRef>>(construct_infos[kKernelOutputs]));
}

BaseLowerNodePtr CreateAscendParallelLowerNode(const std::string &target, bool,
                                               const Map<std::string, NodeRef> &construct_infos) {
  CHECK(construct_infos.find(kKernelInputs) != construct_infos.end());
  CHECK(construct_infos.find(kKernelOutputs) != construct_infos.end());
  return std::make_shared<AscendParallelLowerNode>(target, Downcast<Array<NodeRef>>(construct_infos[kKernelInputs]),
                                                   Downcast<Array<NodeRef>>(construct_infos[kKernelOutputs]));
}

REG_NODE_CREATOR(kCuda, kParallel, CreateCudaParallelLowerNode);
REG_NODE_CREATOR(kCce, kParallel, CreateAscendParallelLowerNode);
}  // namespace lower
}  // namespace akg
