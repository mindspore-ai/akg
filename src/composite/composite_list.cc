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

#include <sstream>
#include <pass/utils.h>
#include "composite/composite.h"
#include "build_module.h"
#include "composite/util.h"
#include "composite/parser.h"
#include "composite/block_fusion.h"
#include "composite/stitch_fusion.h"
#include "composite/dump.h"
#include "composite/sync_process.h"
#include "dimension_peeling.h"
#include "codegen/pass_mgr.h"

namespace akg {
Stmt String2LowerStmtSimple(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool poly,
                            bool buffer_stitch, bool fold_dim, std::vector<size_t> &split_index) {
  CHECK(json_str);
  picojson::value v = String2Json(json_str->value);
  BuildInfo info;
  info.opt.stitch = buffer_stitch;
  info.opt.fold_dim = fold_dim;
  info.opt.enable_dump = false;
  ExtractBuildInfo(v, info);
  std::string sch_name = GetSchedule(info.tensors);
  const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
  CHECK(sch_create != nullptr);
  Schedule sch = (*sch_create)(info.tensors, sch_name, poly);
  auto config = GetConfig();
  Array<NodeRef> args, shape_vars, arg_list_0;
  Map<Tensor, Buffer> binds, binds_0;
  auto stmt = LowerStmt(sch, info.args, shape_vars, info.kernel_name + "_check", info.in_binds, attrs, false, poly,
                        false, "cuda", config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
  return Downcast<Stmt>(stmt);
}

std::vector<std::string> GetNames(const Array<NodeRef> &io) {
  std::vector<std::string> names;
  for (const auto &arg : io) {
    CHECK(arg.as<StringImm>());
    auto arg_name = arg.as<StringImm>()->value;
    names.emplace_back(arg_name);
  }
  return names;
}

Array<NodeRef> ReorderArgs(const Array<NodeRef> &inputs, const Array<NodeRef> &outputs, const Array<NodeRef> &all_args,
                           std::unordered_map<std::string, NodeRef> &outputs2args) {
  // reorder args_list, now args_list satisfies: op1_input op2_input ... op1_output op2_output ...
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
  CHECK(inputs_name.size() == input_args.size());
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
  return ordered_args;
}

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

class ElimDuplicateInputs : public IRMutator {
 public:
  explicit ElimDuplicateInputs(const Array<NodeRef> &inputs) : names_(GetNames(inputs)) {}
  Stmt Run(Stmt &stmt) {
    is_mutate_ = false;
    static_cast<void>(Mutate(stmt));
    is_mutate_ = true;
    return Mutate(stmt);
  }

 private:
  Expr Mutate_(const Load *op, const Expr &e) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
      auto it = vars_.find(name);
      if (it != vars_.end()) {
        if (is_mutate_) return Load::make(op->type, it->second, this->Mutate(op->index), op->predicate);
      } else {
        vars_[name] = var;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    Var var = op->buffer_var;
    auto name = var->name_hint;
    if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
      auto it = vars_.find(name);
      if (it != vars_.end()) {
        if (is_mutate_) return Store::make(it->second, this->Mutate(op->value), this->Mutate(op->index), op->predicate);
      } else {
        vars_[name] = var;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool is_mutate_{false};
  std::unordered_map<std::string, Var> vars_;
  std::vector<std::string> names_;
};

#define DUMP_ORIGIN_IR(dump_manager, arg0) dump_manager.DumpStmt("Origin", arg0)
#define TRANSFORM_AND_TRY_DUMP(dump_manager, out0, call, arg0, ...) \
  do {                                                              \
    out0 = call(arg0, ##__VA_ARGS__);                               \
    dump_manager.DumpStmt(#call, out0);                             \
  } while (0)

class CompositeJsonList {
 public:
  CompositeJsonList(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                    const Array<NodeRef> &alloc_map_list, const Array<NodeRef> &reuse_map_list,
                    const Array<NodeRef> &clean_op_map_list, const Array<NodeRef> &attrs_list, bool poly,
                    const std::string &target)
      : json_str_node_(json_str_node),
        inputs_(inputs),
        outputs_(outputs),
        alloc_map_list_(alloc_map_list),
        reuse_map_list_(reuse_map_list),
        clean_op_map_list_(clean_op_map_list),
        attrs_list_(attrs_list),
        poly_(poly),
        target_(target) {}

  Module Build() {
    CHECK(!json_str_node_.empty());
    std::vector<Stmt> block_irs;
    // Build each segment alone.
    for (; block_json_idx_ < json_str_node_.size(); ++block_json_idx_) {
      auto &block_json = json_str_node_[block_json_idx_];
      auto attrs = Downcast<Map<std::string, NodeRef>>(attrs_list_[block_json_idx_]);
      auto json_type = GetJsonType(block_json);
      switch (json_type) {
        case NORMAL_JSON: {
          ++each_ir_idx_;
          auto single_ir = String2LowerStmt(block_json.as<StringImm>(), attrs);
          block_irs.push_back(single_ir);
          break;
        }
        case STITCHING_JSON: {
          fold_dim_ = CheckFoldDim(block_json);
          auto stitched_ir = StitchFusion(block_json, attrs);
          stitched_ir = ElimDuplicateInputs(inputs_).Run(stitched_ir);
          block_irs.push_back(stitched_ir);
          break;
        }
        default:
          CHECK(0) << "UNSUPPORTED JSON{" << json_type << "}: " << block_json;
          break;
      }
    }

    // Postprocess for segments: 1. Merge segments; 2. Process sync stmt; 3. Eliminate duplicate inputs.
    auto res_ir = MergeStmts(block_irs);
    auto build_rst = PostprocessToBuildRst(res_ir);
    return BuildToModule(build_rst, target_);
  }

 protected:
  virtual Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs) = 0;
  virtual Stmt StitchFusion(const NodeRef &block_json, Map<std::string, NodeRef> &attrs) = 0;
  virtual Stmt MergeStmts(std::vector<Stmt> &block_irs) = 0;
  virtual NodeRef PostprocessToBuildRst(Stmt &stmt) = 0;

  enum JsonType { NORMAL_JSON, STITCHING_JSON, UNKNOWN };
  JsonType GetJsonType(const NodeRef &json) {
    JsonType type = UNKNOWN;
    if (json.as<StringImm>()) {
      type = NORMAL_JSON;
    } else {
      type = STITCHING_JSON;
    }
    return type;
  }

  void GetRealOutputs() {
    auto outputs_name = GetNames(outputs_);
    for (const auto &output : outputs_name) {
      if (outputs2args_.find(output) != outputs2args_.end()) {
        real_outputs_[output] = outputs2args_[output];
      }
    }
  }
  Array<NodeRef> json_str_node_;
  Array<NodeRef> inputs_;
  Array<NodeRef> outputs_;
  Array<NodeRef> alloc_map_list_;
  Array<NodeRef> reuse_map_list_;
  Array<NodeRef> clean_op_map_list_;
  Array<NodeRef> attrs_list_;
  bool poly_{true};
  bool fold_dim_{true};
  std::string target_;
  Array<NodeRef> all_args_;
  std::unordered_map<std::string, NodeRef> outputs2args_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  std::string merge_name_;
  size_t each_ir_idx_{0};
  size_t block_json_idx_{0};
  std::vector<size_t> split_index_;
};

#ifdef USE_AKG_COMPILE_STUB
class CompositeJsonListGpu : public CompositeJsonList {
 public:
  CompositeJsonListGpu(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                       const Array<NodeRef> &alloc_map_list, const Array<NodeRef> &reuse_map_list,
                       const Array<NodeRef> &clean_op_map_list, const Array<NodeRef> &attrs_list, bool poly,
                       const std::string &target)
      : CompositeJsonList(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list, attrs_list,
                          poly, target) {}

  Stmt StitchFusion(const NodeRef &block_json, Map<std::string, NodeRef> &attrs) override {
    auto alloc_map = Downcast<Map<std::string, Array<NodeRef>>>(alloc_map_list_[block_json_idx_]);
    auto reuse_map = Downcast<Map<std::string, Array<NodeRef>>>(reuse_map_list_[block_json_idx_]);
    auto clean_op_map = Downcast<Map<std::string, Array<NodeRef>>>(clean_op_map_list_[block_json_idx_]);
    StitchAttrInfo stitch_attr;
    std::vector<Stmt> stitch_irs = LowerStitchIRs(block_json, stitch_attr, attrs, alloc_map);
    StitchBufAlloc buf_manager(stitch_irs, alloc_map, reuse_map, clean_op_map, outputs2args_);
    buf_manager.BufferAllocReuse();
    GetRealOutputs();
    auto stitched_ir = StitchFusionGpu(stitch_irs, merge_name_, stitch_attr, buf_manager.stitch_buffer_map,
                                       buf_manager.buf_within_op_map, buf_manager.allocate_revoke, real_outputs_);
    return stitched_ir;
  }

  Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs) override {
    Map<std::string, Array<NodeRef>> alloc_map;
    return String2LowerStmt(json_str, attrs, 0, 0, false, true, alloc_map);
  }

  Map<std::string, NodeRef> SetSharedMemoryTensors(const Map<std::string, NodeRef> &attrs, const BuildInfo &info,
                                                   const Map<std::string, Array<NodeRef>> &alloc_map) {
    Map<std::string, NodeRef> new_attrs = attrs;
    std::string shared_name;
    for (auto &input : info.input_names) {
      if (alloc_map.count(input)) {
        shared_name += input + " ";
      }
    }
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
    new_attrs.Set("shared_memory_tensors", Expr(shared_name));
    return new_attrs;
  }
  Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, int grid_dims,
                        int block_dims, bool buffer_stitch, bool fold_dim,
                        const Map<std::string, Array<NodeRef>> &alloc_map) {
    CHECK(json_str);
    picojson::value v = String2Json(json_str->value);
    BuildInfo info;
    info.opt.stitch_ir_idx_ = each_ir_idx_;
    info.opt.stitch = buffer_stitch;
    info.opt.fold_dim = fold_dim;
    ExtractBuildInfo(v, info);
    // ensure merge_name_ is the same as original json name
    if (merge_name_.empty()) merge_name_ = info.kernel_name;
    std::string sch_name = GetSchedule(info.tensors);
    const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
    CHECK(sch_create != nullptr);
    Schedule sch = (*sch_create)(info.tensors, sch_name, poly_, grid_dims, block_dims, buffer_stitch);
    auto config = GetConfig();
    // use each_ir_idx_ to distinct different subgraph
    std::string distinct_name = info.kernel_name + "_" + std::to_string(each_ir_idx_);
    Array<NodeRef> args, shape_vars, arg_list_0;
    Map<Tensor, Buffer> binds, binds_0;
    std::vector<size_t> split_index;
    auto new_attrs = SetSharedMemoryTensors(attrs, info, alloc_map);
    auto stmt = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, new_attrs, false, poly_, false,
                          "cuda", config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
    size_t count = 0;
    for (const auto &x : arg_list_0) {
      auto buffer = x.as<BufferNode>();
      CHECK(buffer) << "arg must be a BufferNode";
      if (std::find(info.input_names.begin(), info.input_names.end(), buffer->name) == std::end(info.input_names)) {
        CHECK(count < info.output_names.size());
        outputs2args_[info.output_names[count]] = x;
        count++;
      }
      all_args_.push_back(x);
    }
    return Downcast<Stmt>(stmt);
  }

  std::vector<Stmt> LowerStitchIRs(const NodeRef &block_json, StitchAttrInfo &stitch_attr,
                                   const Map<std::string, NodeRef> &attrs,
                                   const Map<std::string, Array<NodeRef>> &alloc_map) {
    std::vector<Stmt> stitch_irs;
    std::vector<Expr> loop_extent_array;
    std::vector<GridBlockDims> dim_array;
    std::vector<StitchOpType> ir_type_array;
    for (auto &stitch_json : Downcast<Array<Expr>>(block_json)) {
      ++each_ir_idx_;
      std::vector<OpDesc> op_v = ParseOpDesc(stitch_json.as<StringImm>()->value);
      auto kernel_name = ParseKernelName(stitch_json.as<StringImm>()->value);
      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;
      using std::placeholders::_4;
      using std::placeholders::_5;
      using std::placeholders::_6;
      const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool, bool, bool,
                               std::vector<size_t> &)>
        f = std::bind(&String2LowerStmtSimple, _1, _2, _3, _4, _5, _6);
      BufferStitchAttr stitch_attr_info(f);
      stitch_attr_info.GetBufferStitchAttr(stitch_json, op_v, attrs, poly_, fold_dim_);
      auto dims = stitch_attr_info.dims;
      auto stitch_type = stitch_attr_info.stitch_type_;
      dim_array.push_back(dims);  // save current dims into array.
      IrAttrInfo ir_attr_info = GetIRAttr(stitch_type, stitch_attr_info, ir_type_array, dim_array, attrs);
      DumpIRAttr(kernel_name, ir_attr_info, each_ir_idx_);
      ir_type_array.push_back(stitch_type);  // Note this should be done AFTER GetIrAttr.
      auto new_attrs = BindBlockAndThread(ir_attr_info.dims, poly_, ir_attr_info.attrs);
      if (each_ir_idx_ == 1) split_index_ = stitch_attr_info.split_index;
      new_attrs = SetAutoFuseAttr(split_index_, new_attrs);
      new_attrs.Set("enable_stitch_fusion", Expr(true));

      auto single_ir = String2LowerStmt(stitch_json.as<StringImm>(), new_attrs, ir_attr_info.grid_dims,
                                        ir_attr_info.block_dims, true, fold_dim_, alloc_map);
      stitch_irs.emplace_back(InsertSync(single_ir));
    }
    stitch_attr.type_array = ir_type_array;
    return stitch_irs;
  }

 private:
  Stmt MergeStmts(std::vector<Stmt> &block_irs) final {
    auto config = GetConfig();
    auto dump_mng = DumpManager(merge_name_ + "_merge", config->dump_pass_ir);
    DUMP_ORIGIN_IR(dump_mng, block_irs);

    Stmt merged_ir;
    if (block_irs.size() == 1) {
      merged_ir = block_irs[0];
    } else {
      auto attrs = Downcast<Map<std::string, NodeRef>>(attrs_list_[0]);
      if (attrs.find("pipeline_groups") != attrs.end()) {
        auto pipeline_groups = Downcast<Array<Array<NodeRef>>>(attrs["pipeline_groups"]);
        TRANSFORM_AND_TRY_DUMP(dump_mng, block_irs, ir::PipelineFusion, block_irs, pipeline_groups, target_);
      }
      TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::BlockFusion, block_irs, target_);
    }

    TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::ProcessSyncInnerThread, merged_ir);
    auto ElimDupInputs = [](Stmt stmt, const Array<NodeRef> &inputs) { return ElimDuplicateInputs(inputs).Run(stmt); };
    TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ElimDupInputs, merged_ir, inputs_);
    return merged_ir;
  }

  NodeRef PostprocessToBuildRst(Stmt &stmt) final {
    auto config = GetConfig();
    Array<NodeRef> ordered_args = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
    auto rst = LowerFunc(stmt, merge_name_, config, ordered_args);
    return BuildRstNode::make(rst, merge_name_);
  }
};

#else
class PeelInfoMutator : public IRMutator {
 public:
  PeelInfoMutator(const PeelInfo &peel_info, const Map<Tensor, Buffer> &extern_buffer)
      : peel_info_(peel_info), extern_buffer_(extern_buffer) {}
  ~PeelInfoMutator() = default;

  Stmt Run(Stmt &s) {
    for (const auto &it : extern_buffer_) {
      s = ir::TensorSubstitute2(s, it.second->name, it.first->op, it.first->value_index);
    }
    s = this->Mutate(s);
    s = ExtraModify(s);
    return s;
  }

 private:
  virtual Array<Expr> FixArgs(const Array<Expr> &args, const std::string &name) = 0;
  virtual Stmt ExtraModify(Stmt &s) { return s; }

  bool IsPeeledTensor(const std::string &name) {
    for (auto &tensor : peel_info_.real_peeled_tensors) {
      if (tensor.as<BufferNode>()->name == name) {
        return true;
      }
    }
    return false;
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->func.defined() && IsPeeledTensor(op->func->func_name())) {
      return Call::make(op->type, op->name, FixArgs(op->args, op->name), op->call_type, op->func, op->value_index);
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (IsPeeledTensor(op->func->func_name())) {
      return Provide::make(op->func, op->value_index, this->Mutate(op->value),
                           FixArgs(op->args, op->func->func_name()));
    }
    return IRMutator::Mutate_(op, s);
  }

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
  Stmt ExtraModify(Stmt &s) override {
    for (auto it = loop_var_.rbegin(); it != loop_var_.rend(); ++it) {
      s = For::make(it->second, 0, peel_info_.peels[it->first], ForType::Serial, DeviceAPI::None, s);
    }
    return s;
  }

  Array<Expr> FixArgs(const Array<Expr> &args, const std::string &) override {
    Array<Expr> new_args;
    for (size_t i = 0; i < args.size(); ++i) {
      if (peel_info_.peels.find(i) != peel_info_.peels.end()) {
        new_args.push_back(loop_var_[i]);
      }
      new_args.push_back(args[i]);
    }
    return new_args;
  }

  std::map<int, Var> loop_var_;
};

std::pair<bool, int64_t> GetTensorPeel(size_t i, const std::vector<std::pair<int, int64_t>> &dims) {
  for (auto dim : dims) {
    if (i == static_cast<size_t>(dim.first)) {
      return {true, dim.second};
    }
  }
  return {false, 1};
}

class AddInnerForAndBlockInfo : public PeelInfoMutator {
 public:
  AddInnerForAndBlockInfo(const PeelInfo &peel_info, int block_dim, const Map<Tensor, Buffer> &extern_buffer,
                          const std::unordered_map<std::string, NodeRef> &outputs2args)
      : PeelInfoMutator(peel_info, extern_buffer), block_dim_(block_dim) {
    block_var_ = Variable::make(Int(32), BLOCK_IDX_X);
    inner_size_ = peel_info_.peels.begin()->second / block_dim_;
    offset_ = Add::make(Mul::make(block_var_, Expr(inner_size_)), loop_var_);

    for (auto iter : outputs2args) {
      name_map_[iter.second.as<BufferNode>()->name] = iter.first;
    }
  }

 private:
  std::string GetOriginName(const std::string &name) {
    if (name_map_.find(name) != name_map_.end()) {
      return name_map_[name];
    }
    return name;
  }
  Stmt ExtraModify(Stmt &s) override {
    // Add inner For.
    s = For::make(loop_var_, 0, inner_size_, ForType::Serial, DeviceAPI::None, s);

    // Add block info.
    Expr block_ext = make_const(Int(32), block_dim_);
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), block_ext), block_var_,
                                         air::IterVarType::kThreadIndex, BLOCK_IDX_X);
    s = AttrStmt::make(block_iv, air::ir::attr::thread_extent, block_ext, s);

    return s;
  }

  Array<Expr> FixArgs(const Array<Expr> &args, const std::string &name) override {
    Array<Expr> new_args;
    std::string origin_name = GetOriginName(name);
    auto peel_iter = peel_info_.peeled_tensors.find(origin_name);
    CHECK(peel_iter != peel_info_.peeled_tensors.end());
    for (size_t i = 0; i < args.size(); ++i) {
      auto peel_tensor = GetTensorPeel(i, peel_iter->second);
      if (peel_tensor.first) {
        new_args.push_back(offset_);
      }
      new_args.push_back(args[i]);
    }
    return new_args;
  }

 private:
  int block_dim_{1};
  std::unordered_map<std::string, std::string> name_map_;
  Var loop_var_{"inner_peel"};
  int inner_size_{1};
  Var block_var_;
  Expr offset_{Expr(0)};
};

class CompositeJsonListAscend : public CompositeJsonList {
 public:
  CompositeJsonListAscend(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs,
                          const Array<NodeRef> &outputs, const Array<NodeRef> &alloc_map_list,
                          const Array<NodeRef> &reuse_map_list, const Array<NodeRef> &clean_op_map_list,
                          const Array<NodeRef> &attrs_list, bool poly, const std::string &target)
      : CompositeJsonList(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list, attrs_list,
                          poly, target) {}
  Stmt StitchFusion(const NodeRef &block_json, Map<std::string, NodeRef> &attrs) override {
    peeled_tensors_.clear();
    auto alloc_map = Downcast<Map<std::string, Array<NodeRef>>>(alloc_map_list_[block_json_idx_]);
    auto reuse_map = Downcast<Map<std::string, Array<NodeRef>>>(reuse_map_list_[block_json_idx_]);
    auto clean_op_map = Downcast<Map<std::string, Array<NodeRef>>>(clean_op_map_list_[block_json_idx_]);
    std::vector<Stmt> stitch_irs = LowerStitchIRs(block_json, attrs);
    auto stitch_buffer = GetStitchBuffer(alloc_map);
    GetRealOutputs();
    auto stitched_ir = StitchFusionAscend(stitch_irs, merge_name_, stitch_buffer, real_outputs_);
    MergeLowerData(stitch_lower_datas_);
    FixLowerDataForStitch();
    stitched_ir = AddPeelInfoForLoopAndData(stitched_ir, final_data_, attrs);
    g_attrs.Set(kEnableMulticore, Expr(1));
    stitched_ir =
      Downcast<Stmt>(LowerAscend(stitched_ir, final_data_, LowerStage::FLATTEN, LowerStage::BEFORE_REWRITE));
    lower_datas_.emplace_back(final_data_);
    return stitched_ir;
  }

  void FixLowerDataForStitch() {
    Array<NodeRef> ordered_args = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
    final_data_.arg_list_0_ = ordered_args;
    final_data_.binds_0_ = FixBinds(final_data_.binds_0_, ordered_args);
    final_data_.name = merge_name_;
  }

  static Map<Tensor, Buffer> FixBinds(const Map<Tensor, Buffer> &origin_binds, const Array<NodeRef> &ordered_args) {
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

  static void GetPeeledTensors(LowerData &data, PeelInfo &peel_info,
                               std::unordered_map<std::string, NodeRef> &outputs2args) {
    for (auto &t : peel_info.peeled_tensors) {
      auto real_tensor = t.first;
      if (outputs2args.count(t.first)) {
        real_tensor = outputs2args[t.first].as<BufferNode>()->name;
      }
      for (auto &arg : data.arg_list_0_) {
        auto buffer = arg.as<BufferNode>();
        CHECK(buffer) << "arg must be a BufferNode";
        if (buffer->name == real_tensor) {
          peel_info.real_peeled_tensors.push_back(arg);
        }
      }
    }
  }

  void AddPeelInfoForData(LowerData &data, PeelInfo &peel_info,
                          std::unordered_map<std::string, NodeRef> &outputs2args) {
    std::unordered_map<std::string, std::string> name_map;
    for (auto iter : outputs2args) {
      name_map[iter.second.as<BufferNode>()->name] = iter.first;
    }
    auto GetOriginName = [&name_map](const std::string &name) {
      if (name_map.find(name) != name_map.end()) {
        return name_map[name];
      }
      return name;
    };

    Array<NodeRef> out_args;
    Map<Tensor, Buffer> out_binds;
    for (const auto &kv : data.binds_0_) {
      bool found = false;
      for (auto &t : peel_info.real_peeled_tensors) {
        if (kv.second == t) found = true;
      }
      if (!found) out_binds.Set(kv.first, kv.second);
    }

    Map<Buffer, Buffer> buffer_replace;
    for (const auto &x : data.arg_list_0_) {
      if (x->IsInstance<BufferNode>()) {
        bool changed = false;
        for (auto &t : peel_info.real_peeled_tensors) {
          if (x == t) {
            auto old_shape = x.as<BufferNode>()->shape;
            Array<Expr> new_shape;
            auto peel_iter = peel_info.peeled_tensors.find(GetOriginName(x.as<BufferNode>()->name));
            CHECK(peel_iter != peel_info.peeled_tensors.end());
            for (size_t i = 0; i < old_shape.size(); ++i) {
              auto peel_tensor = GetTensorPeel(i, peel_iter->second);
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
            changed = true;
            break;
          }
        }
        if (changed) continue;
        out_args.push_back(x);
      } else {
        out_args.push_back(x);
      }
    }
    data.arg_list_0_ = out_args;
    data.binds_0_ = out_binds;
    ReplaceBufferForALLArgsAndOutputs2args(buffer_replace);
  }

  void ReplaceBufferForALLArgsAndOutputs2args(Map<Buffer, Buffer> &buffer_replace) {
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
  PeelInfo GetPeelInfoFromAttrs(const Map<std::string, NodeRef> &attrs) {
    PeelInfo peel_info;
    if (attrs.find("peeling") != attrs.end()) {
      auto peeling = attrs["peeling"].as<StringImm>();
      CHECK(peeling);
      auto parsed_peeling = Str2Peeling(peeling->value);
      CHECK(!parsed_peeling.empty());
      for (auto &kv : parsed_peeling) {
        peel_info.peels.insert(kv);
      }
      peel_info.peeled_tensors = peeled_tensors_;
    }
    return peel_info;
  }
  Stmt AddPeelInfoForLoopAndData(Stmt &s, LowerData &data, Map<std::string, NodeRef> &attrs) {
    PeelInfo peel_info = GetPeelInfoFromAttrs(attrs);
    GetPeeledTensors(data, peel_info, outputs2args_);
    AddPeelInfoForData(data, peel_info, outputs2args_);
    s = AddPeelInfoForLoop(peel_info, data.binds_0_).Run(s);
    DumpStmt2File("stitch_info/" + merge_name_ + "_after_add_loop.cc", s);
    return s;
  }

  std::unordered_map<std::string, NodeRef> GetStitchBuffer(const Map<std::string, Array<NodeRef>> &alloc_map) {
    std::unordered_map<std::string, NodeRef> stitch_buffer;
    for (auto &kv : outputs2args_) {
      if (alloc_map.count(kv.first)) {
        stitch_buffer.insert(kv);
      }
    }
    return stitch_buffer;
  }

  std::vector<Stmt> LowerStitchIRs(const NodeRef &block_json, const Map<std::string, NodeRef> &attrs) {
    stitch_lower_datas_.clear();
    auto split = Evaluate::make(Expr("===========split=========="));
    std::vector<Stmt> stitch_irs;
    for (auto &stitch_json : Downcast<Array<Expr>>(block_json)) {
      ++each_ir_idx_;

      // Set compile attr for current split json
      Map<std::string, NodeRef> new_attrs;
      new_attrs.Set("enable_multicore", make_const(Int(32), 0));
      auto tiling_idx = "sub_attr_" + std::to_string(each_ir_idx_);
      for (const auto &it : attrs) {
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

      auto single_ir = String2LowerBeforeFlattern(stitch_json.as<StringImm>(), new_attrs, true);
      stitch_irs.emplace_back(single_ir);
      stitch_irs.emplace_back(split);
    }
    return stitch_irs;
  }

  Stmt String2LowerBeforeFlattern(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool stitch) {
    CHECK(json_str);
    picojson::value v = String2Json(json_str->value);
    BuildInfo info;
    info.opt.stitch_ir_idx_ = each_ir_idx_;
    info.opt.stitch = stitch;
    info.opt.fold_dim = fold_dim_;
    if (attrs.find("peeling") != attrs.end()) {
      auto peeling = attrs["peeling"].as<StringImm>();
      CHECK(peeling != nullptr);
      info.opt.peel_info.peeling = peeling->value;
    }
    ExtractBuildInfo(v, info);
    peeled_tensors_.insert(info.opt.peel_info.peeled_tensors.begin(), info.opt.peel_info.peeled_tensors.end());
    // ensure merge_name_ is the same as original json name
    if (merge_name_.empty()) merge_name_ = info.kernel_name;
    Array<Operation> ops;
    std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
    Schedule sch = create_schedule(ops);
    auto config = GetConfig();
    // use each_ir_idx_ to distinct different subgraph
    std::string distinct_name = info.kernel_name + "_" + std::to_string(each_ir_idx_);
    Array<NodeRef> args, shape_vars, arg_list_0;
    Map<Tensor, Buffer> binds, binds_0;
    std::vector<size_t> split_index;
    auto node_ref = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, attrs, false, poly_, false,
                              "cce", config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
    auto stmt = Downcast<Stmt>(node_ref);
    LowerData data(info.args, arg_list_0, binds, binds_0, shape_vars, distinct_name, false, true, false, "cce", config);
    node_ref = LowerAscend(stmt, data, LowerStage::BEGIN, LowerStage::BEFORE_FLATTEN);

    stitch_lower_datas_.emplace_back(data);

    size_t count = 0;
    for (const auto &x : arg_list_0) {
      auto buffer = x.as<BufferNode>();
      CHECK(buffer) << "arg must be a BufferNode";
      if (std::find(info.input_names.begin(), info.input_names.end(), buffer->name) == std::end(info.input_names)) {
        CHECK(count < info.output_names.size());
        outputs2args_[info.output_names[count]] = x;
        count++;
      }
      all_args_.push_back(x);
    }
    return Downcast<Stmt>(node_ref);
  }

  Stmt AddPeelInfoAndBlockAttr(Stmt &s, LowerData &data, PeelInfo &peel_info,
                               std::unordered_map<std::string, NodeRef> &outputs2args, int block) {
    GetPeeledTensors(data, peel_info, outputs2args);
    AddPeelInfoForData(data, peel_info, outputs2args);
    s = AddInnerForAndBlockInfo(peel_info, block, data.binds_0_, outputs2args).Run(s);
    return s;
  }

  Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs) override {
    CHECK(json_str);
    picojson::value v = String2Json(json_str->value);
    BuildInfo info;
    info.opt.stitch_ir_idx_ = each_ir_idx_;
    info.opt.stitch = false;
    info.opt.fold_dim = true;

    if (attrs.find("peeling") != attrs.end()) {
      auto peeling = attrs["peeling"].as<StringImm>();
      CHECK(peeling != nullptr);
      info.opt.peel_info.peeling = peeling->value;
    }
    peeled_tensors_.clear();
    ExtractBuildInfo(v, info);
    peeled_tensors_.insert(info.opt.peel_info.peeled_tensors.begin(), info.opt.peel_info.peeled_tensors.end());

    // ensure merge_name_ is the same as original json name
    if (merge_name_.empty()) merge_name_ = info.kernel_name;
    Array<Operation> ops;
    std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
    Schedule sch = create_schedule(ops);
    auto config = GetConfig();
    // use each_ir_idx_ to distinct different subgraph
    std::string distinct_name = info.kernel_name + "_" + std::to_string(each_ir_idx_);
    Array<NodeRef> args, shape_vars, arg_list_0;
    Map<Tensor, Buffer> binds, binds_0;
    std::vector<size_t> split_index;
    auto node_ref = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, attrs, false, poly_, false,
                              "cce", config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
    auto stmt = Downcast<Stmt>(node_ref);

    LowerData data(info.args, arg_list_0, binds, binds_0, shape_vars, distinct_name, false, true, false, "cce", config);

    if (attrs.find("block_plan") != attrs.end()) {
      node_ref = LowerAscend(stmt, data, LowerStage::BEGIN, LowerStage::BEFORE_FLATTEN);
      stmt = Downcast<Stmt>(node_ref);

      std::unordered_map<std::string, NodeRef> tmp_outputs2args;
      size_t count = 0;
      for (const auto &x : data.arg_list_0_) {
        auto buffer = x.as<BufferNode>();
        CHECK(buffer) << "arg must be a BufferNode";
        if (std::find(info.input_names.begin(), info.input_names.end(), buffer->name) == std::end(info.input_names)) {
          CHECK(count < info.output_names.size());
          tmp_outputs2args[info.output_names[count]] = x;
          count++;
        }
      }

      auto block_plan = attrs["block_plan"].as<IntImm>();
      CHECK(block_plan);
      int block = block_plan->value;
      PeelInfo peel_info = GetPeelInfoFromAttrs(attrs);
      stmt = AddPeelInfoAndBlockAttr(stmt, data, peel_info, tmp_outputs2args, block);
      stmt = NEXT_PASS(CanonicalSimplify, stmt);
      node_ref = LowerAscend(stmt, data, LowerStage::FLATTEN, LowerStage::BEFORE_REWRITE);
    } else {
      node_ref = LowerAscend(stmt, data, LowerStage::BEGIN, LowerStage::BEFORE_REWRITE);
    }
    lower_datas_.emplace_back(data);

    size_t count = 0;
    for (const auto &x : data.arg_list_0_) {
      auto buffer = x.as<BufferNode>();
      CHECK(buffer) << "arg must be a BufferNode";
      if (std::find(info.input_names.begin(), info.input_names.end(), buffer->name) == std::end(info.input_names)) {
        CHECK(count < info.output_names.size());
        outputs2args_[info.output_names[count]] = x;
        count++;
      }
      all_args_.push_back(x);
    }
    return Downcast<Stmt>(node_ref);
  }

 private:
  Stmt MergeStmts(std::vector<Stmt> &block_irs) final {
    auto dump_mng = DumpManager(merge_name_ + "_merge", getenv(GetDumpIRFlag().c_str()) != nullptr);
    DUMP_ORIGIN_IR(dump_mng, block_irs);

    auto RewriteBlocks = [this](std::vector<Stmt> &block_irs) {
      for (size_t i = 0; i < block_irs.size(); ++i) {
        lower_datas_[i].name = std::string("part_").append(std::to_string(i));
        block_irs[i] =
          Downcast<Stmt>(LowerAscend(block_irs[i], lower_datas_[i], LowerStage::REWRITE, LowerStage::BEFORE_LOWERFUNC));
      }
      return block_irs;
    };
    Stmt merged_ir;
    if (block_irs.size() == 1) {
      TRANSFORM_AND_TRY_DUMP(dump_mng, block_irs, RewriteBlocks, block_irs);
      merged_ir = block_irs[0];
    } else {
      auto attrs = Downcast<Map<std::string, NodeRef>>(attrs_list_[0]);
      if (attrs.find("pipeline_groups") != attrs.end()) {
        auto pipeline_groups = Downcast<Array<Array<NodeRef>>>(attrs["pipeline_groups"]);
        TRANSFORM_AND_TRY_DUMP(dump_mng, block_irs, ir::PipelineFusion, block_irs, pipeline_groups, target_);
        RearrangeLowerData(pipeline_groups);
      }
      TRANSFORM_AND_TRY_DUMP(dump_mng, block_irs, RewriteBlocks, block_irs);
      TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ir::BlockFusion, block_irs, target_);
    }

    auto ElimDupInputs = [](Stmt &stmt, const Array<NodeRef> &inputs) { return ElimDuplicateInputs(inputs).Run(stmt); };
    TRANSFORM_AND_TRY_DUMP(dump_mng, merged_ir, ElimDupInputs, merged_ir, inputs_);
    return merged_ir;
  }

  void RearrangeLowerData(const Array<Array<NodeRef>> &pipeline_groups) {
    std::set<size_t> visited;
    std::vector<std::set<size_t>> groups;
    groups.resize(pipeline_groups.size());
    for (size_t i = 0; i < pipeline_groups.size(); ++i) {
      for (auto group_id : pipeline_groups[i]) {
        auto segment_id = group_id.as<IntImm>()->value;
        groups[i].insert(segment_id);
        visited.insert(segment_id);
      }
    }

    std::vector<LowerData> new_data;
    for (size_t i = 0; i < lower_datas_.size(); ++i) {
      if (visited.count(i) == 0) {
        new_data.push_back(lower_datas_[i]);
      }
    }

    for (const auto &g : groups) {
      MergeLowerData(lower_datas_, g);
      new_data.push_back(final_data_);
    }

    lower_datas_ = new_data;
  }

  void MergeLowerData(const std::vector<LowerData> &lower_datas, const std::set<size_t> &specified = {}) {
    bool all_merge = specified.empty();
    final_data_ = LowerData();
    for (size_t idx = 0; idx < lower_datas.size(); ++idx) {
      auto &lower_data = lower_datas[idx];

      if (!all_merge && specified.count(idx) == 0) continue;
      for (auto arg : lower_data.args_) {
        final_data_.args_.push_back(arg);
      }
      for (auto arg_list : lower_data.arg_list_0_) {
        final_data_.arg_list_0_.push_back(arg_list);
      }
      for (auto iter : lower_data.binds_) {
        final_data_.binds_.Set(iter.first, iter.second);
      }
      for (auto iter : lower_data.binds_0_) {
        final_data_.binds_0_.Set(iter.first, iter.second);
      }
      for (auto shape_var : lower_data.shape_vars_) {
        final_data_.shape_vars_.push_back(shape_var);
      }

      final_data_.config_ = lower_data.config_;
      final_data_.name = lower_data.name;
    }
  }

  NodeRef PostprocessToBuildRst(Stmt &stmt) final {
    MergeLowerData(lower_datas_);
    auto config = GetConfig();
    Array<NodeRef> ordered_args = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
    final_data_.arg_list_0_ = ordered_args;
    final_data_.name = merge_name_;
    auto rst = LowerAscend(stmt, final_data_, LowerStage::END, LowerStage::END);
    return BuildRstNode::make(rst, merge_name_);
  }

  std::unordered_map<std::string, std::vector<std::pair<int, int64_t>>> peeled_tensors_;
  std::vector<LowerData> stitch_lower_datas_;
  std::vector<LowerData> lower_datas_;
  LowerData final_data_;
};
#endif

Module CompositeWithJsonList(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs,
                             const Array<NodeRef> &outputs, const Array<NodeRef> &alloc_map_list,
                             const Array<NodeRef> &reuse_map_list, const Array<NodeRef> &clean_op_map_list,
                             const Array<NodeRef> &attrs_list, bool poly, const std::string &target) {
#ifdef USE_AKG_COMPILE_STUB
  if (target == "cuda") {
    return CompositeJsonListGpu(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list,
                                attrs_list, poly, target)
      .Build();
#else
  if (target == "cce") {
    return CompositeJsonListAscend(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list,
                                   attrs_list, poly, target)
      .Build();
#endif
  } else {
    CHECK(0) << "UNSUPPORTED TARGET: " << target;
    return Module();
  }
}

Stmt GetPeeledBody(const Stmt &stmt, const std::string &peeling) {
  CHECK(stmt.defined());
  DimensionPeeler peeler;
  peeler.Analyze(stmt);
  auto parsed_peeling = Str2Peeling(peeling);
  return peeler.GetPeelBody(parsed_peeling);
}

Map<std::string, NodeRef> CompositePeelAnalyze(const std::string &json_str, const Map<std::string, NodeRef> &attrs) {
  CHECK(!json_str.empty());
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  info.opt.tuning = true;
  if (attrs.defined() && attrs.find("fold_dim") != attrs.end()) {
    CHECK(attrs["fold_dim"].as<ExprNode>());
    auto fold_dim = ir::GetInt32Const(Downcast<Expr>(attrs["fold_dim"]));
    info.opt.fold_dim = static_cast<bool>(fold_dim);
  }
  ExtractBuildInfo(v, info);

  DimensionPeeler peeler;
  CHECK(info.opt.peel_info.stmt.defined());
  peeler.Analyze(info.opt.peel_info.stmt);
  auto peeling_space = peeler.GetPeelSpace();
  Array<Expr> parsed_peeling_space;
  for (const auto &it : peeling_space) {
    parsed_peeling_space.push_back(Peeling2Str(it));
  }

  Array<Expr> input_names_arr;
  for (const auto &name : info.input_names) {
    input_names_arr.push_back(Expr(name));
  }
  Array<Expr> output_names_arr;
  for (const auto &name : info.output_names) {
    output_names_arr.push_back(Expr(name));
  }
  Map<std::string, NodeRef> build_info;
  build_info.Set("op", Expr(info.kernel_name));
  build_info.Set("process", Expr(info.opt.target));
  build_info.Set("input_names", input_names_arr);
  build_info.Set("output_names", output_names_arr);

  Map<std::string, NodeRef> ret;
  ret.Set("stmt", info.opt.peel_info.stmt);
  ret.Set("build_info", build_info);
  ret.Set("peeling_space", parsed_peeling_space);
  return ret;
}

TVM_REGISTER_GLOBAL("composite_with_json_list").set_body_typed(CompositeWithJsonList);
TVM_REGISTER_GLOBAL("get_peeled_body").set_body_typed(GetPeeledBody);
TVM_REGISTER_GLOBAL("composite_peel_analyze").set_body_typed(CompositePeelAnalyze);
TVM_REGISTER_GLOBAL("check_fold_dim").set_body_typed(CheckFoldDim);
}  // namespace akg
