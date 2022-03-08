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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_MULTICHILD_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_MULTICHILD_NODE_H_
#include "composite/utils/dimension_peeling.h"
#include "composite/lower_tree/base_node.h"
#include "composite/lower_tree/json_leaf.h"

namespace akg {
namespace lower {
constexpr auto kInputNames = "input_names";
constexpr auto kOutputNames = "output_names";
constexpr auto kPeeledTensors = "peeled_tensors";
constexpr auto kPeeling = "peeling";
std::unordered_map<std::string, Peeling> GetOriginPeelInfo(const std::string &stitch_origin_json,
                                                           const Map<std::string, NodeRef> &attrs, bool fold_dim);
Map<std::string, Array<NodeRef>> PeelingToNodeRef(const std::unordered_map<std::string, Peeling> &peeled_tensors);
std::unordered_map<std::string, Peeling> NodeRefToPeeling(const Map<std::string, Array<NodeRef>> &peeled_noderef);

class PeelInfoMutator : public IRMutator {
 public:
  PeelInfoMutator(const PeelInfo &peel_info, const Map<Tensor, Buffer> &extern_buffer)
      : peel_info_(peel_info), extern_buffer_(extern_buffer) {}
  ~PeelInfoMutator() = default;

  Stmt Run(Stmt &s);

 private:
  virtual Array<Expr> FixArgs(const Array<Expr> &args, const std::string &name) = 0;
  virtual Stmt ExtraModify(Stmt &s) { return s; }

  Expr Mutate_(const Call *op, const Expr &e) final;
  Stmt Mutate_(const Provide *op, const Stmt &s) final;

 protected:
  PeelInfo peel_info_;
  Map<Tensor, Buffer> extern_buffer_;
};

class AddPeelInfoForLoop : public PeelInfoMutator {
 public:
  AddPeelInfoForLoop(const PeelInfo &peel_info, const Map<Tensor, Buffer> &extern_buffer)
      : PeelInfoMutator(peel_info, extern_buffer) {
    for (auto &kv : peel_info_.peels) {
      loop_var_[kv.first] = Var("peel_" + std::to_string(kv.first));
    }
  }

 private:
  Stmt ExtraModify(Stmt &s) override;
  Array<Expr> FixArgs(const Array<Expr> &args, const std::string &name) override;

  std::map<int, Var> loop_var_;
};

class AddInnerForAndBlockInfo : public PeelInfoMutator {
 public:
  AddInnerForAndBlockInfo(const PeelInfo &peel_info, int block_dim, const Map<Tensor, Buffer> &extern_buffer)
      : PeelInfoMutator(peel_info, extern_buffer), block_dim_(block_dim) {
    block_var_ = Variable::make(Int(32), BLOCK_IDX_X);
    inner_size_ = peel_info_.peels.begin()->second / block_dim_;
    offset_ = Add::make(Mul::make(block_var_, Expr(inner_size_)), loop_var_);
  }

 private:
  Stmt ExtraModify(Stmt &s) override;
  Array<Expr> FixArgs(const Array<Expr> &args, const std::string &name) override;

 private:
  int block_dim_{1};
  Var loop_var_{"inner_peel"};
  int inner_size_{1};
  Var block_var_;
  Expr offset_{Expr(0)};
};

class MultiChildLowerNode : public BaseLowerNode {
 public:
  explicit MultiChildLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                               const Array<NodeRef> &kernel_outputs)
      : BaseLowerNode(target) {
    name_ = __FUNCTION__;
    inputs_ = kernel_inputs;
    outputs_ = kernel_outputs;
  }
  ~MultiChildLowerNode() override = default;

 protected:
  virtual void Merge(const std::vector<LowerData> &datas, std::vector<Stmt> &block_irs);
  virtual LowerData MergeDatas(const std::vector<LowerData> &datas, const std::set<size_t> &specified = {});
  virtual void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &node_ref) {}
  virtual Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) = 0;

  Map<std::string, NodeRef> GetCommonForwardInfo();

  void Postprocess(StageType to);
  void CollectOutputMap(const LowerData &data, const Map<std::string, NodeRef> &backward_info,
                        std::unordered_map<std::string, NodeRef> &outputs2args);
  Array<NodeRef> ReorderArgs(const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                             const Array<NodeRef> &all_args, std::unordered_map<std::string, NodeRef> &outputs2args,
                             const Array<NodeRef> &workspace = {});
  void GetRealOutputs();
  std::pair<Array<Expr>, std::vector<Map<std::string, NodeRef>>> CatchChild();
  void AddPeelInfoForData(LowerData &data, PeelInfo &peel_info);
  void ReplaceBufferForALLArgsAndOutputs2args(Map<Buffer, Buffer> &buffer_replace);
  PeelInfo GetPeelInfoFromAttrs(const Map<std::string, NodeRef> &attrs);

  void AttachMultiChildDecorator(BaseLowerNode *child, Map<std::string, NodeRef> &forward_infos,
                                 Map<std::string, NodeRef> *backward_infos);

  void UpdateMergeInfos(const Map<std::string, NodeRef> &infos) {
    for (auto iter : infos) {
      merge_infos_.Set(iter.first, iter.second);
    }
  }
  void ChildPostProcess(const LowerData &data, const Map<std::string, NodeRef> &backward_infos) {
    UpdateMergeInfos(backward_infos);
    CollectOutputMap(data, backward_infos, outputs2args_);
    for (const auto &x : data->arg_list_0) {
      all_args_.push_back(x);
    }
  }

  Array<NodeRef> inputs_;
  Array<NodeRef> outputs_;
  Array<NodeRef> all_args_;
  std::unordered_map<std::string, NodeRef> outputs2args_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  Array<NodeRef> workspace_args_;
  Map<std::string, NodeRef> merge_infos_;  // Child node -> parent node
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_MULTICHILD_NODE_H_
