/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "build_module.h"
#include "composite/util.h"
#include "composite/parser.h"
#include "composite/optimize/optimize.h"
#include "composite/block_fusion.h"
#include "composite/stitch_fusion.h"
#include "composite/sync_process.h"

namespace akg {
class Emitter : public IRVisitor {
 public:
  Emitter(FuncTensorMap &tensor_map, BuildOpt &opt) : tensor_map_(tensor_map), opt_(opt) {}

 private:
  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "attrs") {
      op_attrs_ = Downcast<Map<std::string, NodeRef>>(op->node);
      Visit(op->body);
      op_attrs_ = {};
    } else {
      IRVisitor::Visit_(op);
    }
  }
  void Visit_(const Provide *op) override {
    CHECK(op->value.as<Call>());
    auto call = op->value.as<Call>();
    auto op_name = call->name;
    auto inputs = call->args;
    Array<NodeRef> real_input;
    for (const auto &input : inputs) {
      if (auto c = input.as<Call>()) {
        if (tensor_map_.count(c->func) == 0) {
          Tensor t = placeholder(c->args, c->type, c->name);
          tensor_map_[c->func] = t;
        }
        real_input.push_back(tensor_map_[c->func]);
      } else {
        real_input.push_back(input);
      }
    }
    const auto *topi_f = air::runtime::Registry::Get(op_name);
    if (topi_f == nullptr) {
      topi_f = air::runtime::Registry::Get(opt_.target + '_' + op_name);
    }
    CHECK(topi_f) << "Akg topi has no op: " << op_name;
    if (op_name == "Reshape") {  // reshape's attr may have shape [-1], it will cause error.
      op_attrs_.Set("shape", op->args);
    }
    Tensor t = (*topi_f)(real_input, op_attrs_);
    if (op_name == "Assign") {
      EmitAssign(t, inputs[0]);
    }

    tensor_map_[op->func] = t;
  }

  void EmitAssign(Tensor &t, const Expr &input) {
    // copy out to bind_input, bind_input is used to bind input[0]
    // d = Assign(a, b), bind_input = d, input0 = bind_input
    auto bind_input = compute(
      t->shape, [&](const Array<Var> &indices) { return t(indices); },
      "assign_tensor_" + std::to_string(assign_count_));
    tensor_map_[bind_input->op] = bind_input;
    opt_.sch_only.emplace_back(bind_input);
    opt_.inplaces[bind_input->op] = input;
    assign_count_++;
  }

 private:
  FuncTensorMap &tensor_map_;
  BuildOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
  int assign_count_{0};
};

void CollectBinds(FuncTensorMap &tensor_map, BuildInfo &info) {
  for (const auto &kv : info.opt.inplaces) {
    CHECK(tensor_map.count(kv.first)) << kv.first->func_name() << " not in tensor map";
    CHECK(tensor_map.count(kv.second.as<Call>()->func))
      << kv.second.as<Call>()->func->func_name() << " not in tensor map";
    auto first = tensor_map[kv.first];
    auto second = tensor_map[kv.second.as<Call>()->func];
    auto buf = decl_buffer(second->shape, second->dtype, second->op->name);
    info.in_binds.Set(first, buf);
    info.in_binds.Set(second, buf);
  }
}

void ProcessSames(FuncTensorMap &tensor_map, BuildOpt &opt) {
  // b = func(a)
  // c = InplaceAssign(x, y, b)     c = b
  // d = InplaceAssign(i, j, c)     d = c
  bool changed = true;
  while (!opt.sames.empty() && changed) {
    changed = false;
    for (auto it = opt.sames.begin(); it != opt.sames.end();) {
      if (tensor_map.count(it->second)) {
        tensor_map[it->first] = tensor_map[it->second];
        it = opt.sames.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
  }
}

void CollectInputs(FuncTensorMap &tensor_map, BuildInfo &info) {
  for (const auto &input : info.input_names) {
    auto iter =
      std::find_if(tensor_map.begin(), tensor_map.end(),
                   [&input](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == input; });
    CHECK(iter != tensor_map.end()) << "input Tensor " << input << " not built.";
    LOG(INFO) << "input: " << input << " " << iter->second;
    info.args.push_back(iter->second);
  }
}

void CollectOutputsAndComputes(FuncTensorMap &tensor_map, BuildInfo &info) {
  int count = 0;
  for (const auto &output : info.output_names) {
    auto iter = std::find_if(
      tensor_map.begin(), tensor_map.end(),
      [&output](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == output; });
    CHECK(iter != tensor_map.end()) << "output Tensor " << output << " not built.";
    LOG(INFO) << "output: " << output << " " << iter->second;
    info.tensors.push_back(iter->second);
    if (!info.opt.fakeout.count(iter->first)) {
      info.args.push_back(iter->second);
    } else {
      auto name = "fake_" + std::to_string(count);
      count++;
      Tensor t = placeholder(iter->second->shape, iter->second->dtype, name);
      info.args.push_back(t);
    }
  }
  for (const auto &inplace_itr : info.opt.inplaces) {
    auto iter =
      std::find_if(tensor_map.begin(), tensor_map.end(), [&inplace_itr](std::pair<const FunctionRef, Tensor> &kv) {
        return kv.first->func_name() == inplace_itr.first->func_name();
      });
    if (std::find_if(info.tensors.begin(), info.tensors.end(),
                     [&iter](const Tensor &t) { return t == iter->second; }) == info.tensors.end()) {
      info.tensors.push_back(iter->second);
    }
  }
}

void CollectSchOnlyComputes(BuildInfo &info) {
  for (const auto &tensor : info.opt.sch_only) {
    info.tensors.push_back(tensor);
  }
}

void CollectIsolatedInplaceTensor(BuildOpt &opt, FuncTensorMap &tensor_map) {
  // tensors which have never be used before is isolated and not be created,
  // so we should create them after emit.
  for (const auto &kv : opt.inplaces) {
    auto c = kv.second.as<Call>();
    if (tensor_map.find(c->func) == tensor_map.end()) {
      tensor_map[c->func] = placeholder(c->args, c->type, c->name);
    }
  }
}

void CollectBuildInfo(FuncTensorMap &tensor_map, BuildInfo &info) {
  DumpBuildInfo(info);
  CollectIsolatedInplaceTensor(info.opt, tensor_map);
  CollectBinds(tensor_map, info);
  ProcessSames(tensor_map, info.opt);
  CollectInputs(tensor_map, info);
  CollectOutputsAndComputes(tensor_map, info);
  CollectSchOnlyComputes(info);
  DumpBuildInfo(info);
}

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info) {
  CHECK(input_json.is<picojson::object>());
  info.opt.fold_dim = !info.opt.stitch;
  // 1. make stmt by input_json
  auto stmt = Parse(input_json, info);
  // 2. optimize stmt
  stmt = Optimize(stmt, info);
  // 3. emit stmt by topi
  FuncTensorMap tensor_map;
  Emitter(tensor_map, info.opt).Visit(stmt);
  // 4. collect build info: args, compute, binds
  CollectBuildInfo(tensor_map, info);
}

Stmt String2LowerStmtSimple(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, bool poly,
                            bool buffer_stitch, std::vector<size_t> &split_index) {
  CHECK(json_str);
  picojson::value v = String2Json(json_str->value);
  BuildInfo info;
  info.opt.stitch = buffer_stitch;
  info.opt.enable_dump = false;
  ExtractBuildInfo(v, info);
  std::string sch_name = GetSchedule(info.tensors);
  const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
  CHECK(sch_create != nullptr);
  Schedule sch = (*sch_create)(info.tensors, sch_name, poly);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  const char *akg_dump_pass_ir = getenv(GetDumpIRFlag().c_str());
  config->dump_pass_ir = akg_dump_pass_ir != nullptr;
  Array<NodeRef> args, shape_vars, arg_list_0;
  Map<Tensor, Buffer> binds, binds_0;
  auto stmt = LowerStmt(sch, info.args, shape_vars, info.kernel_name + "_check", info.in_binds, attrs, false, poly,
                        false, "cuda", config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
  return Downcast<Stmt>(stmt);
}

NodeRef CompositeWithJsonToFunc(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  const char *akg_dump_pass_ir = getenv(GetDumpIRFlag().c_str());
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  config->dump_pass_ir = akg_dump_pass_ir != nullptr;
  attrs.Set("pragma_reschedule", make_const(Int(32), 1));
  if (attrs.find("kernel_name") != attrs.end()) {
    CHECK(attrs["kernel_name"]->IsInstance<StringImm>());
    info.kernel_name = attrs["kernel_name"].as<StringImm>()->value;
  }
  Array<NodeRef> shape_vars;
  auto build_rst =
    akg::BuildToFunc(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, true, "cce", config);
  CHECK(build_rst.defined());
  return std::move(build_rst);
}

Module CompositeWithJsonGpu(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  const auto *build_func = air::runtime::Registry::Get("akg_build_gpu_module");
  CHECK(build_func != nullptr);
  std::string sch = GetSchedule(info.tensors);
  return (*build_func)(info.tensors, info.args, sch, info.kernel_name, attrs, poly, info.in_binds);
}

Module CompositeWithJson(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  if (GetProcess(json_str) == "cuda") {
    return CompositeWithJsonGpu(json_str, attrs, poly);
  }
  auto build_rst = CompositeWithJsonToFunc(json_str, attrs);
  return BuildToModule(build_rst);
}

NodeRef CompositeLower(const std::string &json_str, const Map<std::string, NodeRef> &attrs) {
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  bool tuning = attrs.find("tuning") != attrs.end();
  std::string target = "cce";
  if (GetProcess(json_str) == "cuda") {
    target = "cuda";
  }
  Array<NodeRef> shape_vars;
  return akg::Lower(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false, true, tuning, target,
                    config);
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

class ElimDuplicateInputs : public IRMutator {
 public:
  explicit ElimDuplicateInputs(const Array<NodeRef> &inputs) { names_ = GetNames(inputs); }
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

class CompositeJsonList {
 public:
  CompositeJsonList(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                    const Array<NodeRef> &alloc_map_list, const Array<NodeRef> &reuse_map_list,
                    const Array<NodeRef> &clean_op_map_list, const Array<NodeRef> &attrs_list, bool poly,
                    std::string target)
      : json_str_node_(json_str_node),
        inputs_(inputs),
        outputs_(outputs),
        alloc_map_list_(alloc_map_list),
        reuse_map_list_(reuse_map_list),
        clean_op_map_list_(clean_op_map_list),
        attrs_list_(attrs_list),
        poly_(poly),
        target_(target) {}

  virtual Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs) = 0;
  virtual Stmt StitchFusion(const NodeRef &block_json, Map<std::string, NodeRef> &attrs) = 0;

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
          auto stitched_ir = StitchFusion(block_json, attrs);
          stitched_ir = ElimDuplicateInputs(inputs_).Run(stitched_ir);
          block_irs.push_back(stitched_ir);
          break;
        }
        case UNKNOWN:
        default:
          CHECK(0) << "UNSUPPORTED JSON{" << json_type << "}: " << block_json;
          break;
      }
    }

    // Postprocess for segments: 1. Merge segments; 2. Process sync stmt; 3. Eliminate duplicate inputs.
    akg::BuildConfig final_config = akg::BuildConfig::Current();
    CHECK(final_config.defined());
    final_config->dump_pass_ir = getenv(GetDumpIRFlag().c_str()) != nullptr;
    auto res_ir = MergeStmts(block_irs, final_config);

    Array<NodeRef> ordered_args = ReorderArgs(inputs_, outputs_, all_args_, outputs2args_);
    auto rst = LowerFunc(res_ir, merge_name_, final_config, ordered_args);
    auto build_rst = BuildRstNode::make(rst, merge_name_);
    return BuildToModule(build_rst, target_);
  }

 protected:
  Stmt MergeStmts(std::vector<Stmt> block_irs, const BuildConfig &config) {
    auto dump_mng = DumpManager(merge_name_ + "_merge", config->dump_pass_ir);
    dump_mng.DumpStmt("Origin", block_irs);

    Stmt merged_ir;
    if (block_irs.size() == 1) {
      merged_ir = block_irs[0];
    } else {
      auto attrs = Downcast<Map<std::string, NodeRef>>(attrs_list_[0]);
      if (attrs.find("pipeline_groups") != attrs.end()) {
        auto pipeline_groups = Downcast<Array<Array<NodeRef>>>(attrs["pipeline_groups"]);
        block_irs = ir::PipelineFusion(block_irs, pipeline_groups);
        dump_mng.DumpStmt("PipelineFusion", block_irs);
      }

      merged_ir = ir::BlockFusion(block_irs);
      dump_mng.DumpStmt("BlockFusion", merged_ir);
    }

    merged_ir = ir::ProcessSyncInnerThread(merged_ir);
    dump_mng.DumpStmt("ProcessSyncInnerThread", merged_ir);
    merged_ir = ElimDuplicateInputs(inputs_).Run(merged_ir);
    dump_mng.DumpStmt("ElimDuplicateInputs", merged_ir);

    return merged_ir;
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
  std::string target_;
  Array<NodeRef> all_args_;
  std::unordered_map<std::string, NodeRef> outputs2args_;
  std::unordered_map<std::string, NodeRef> real_outputs_;
  std::string merge_name_;
  size_t each_ir_idx_{0};
  size_t block_json_idx_{0};
  std::vector<size_t> split_index_;
};

class CompositeJsonListGpu : public CompositeJsonList {
 public:
  CompositeJsonListGpu(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs, const Array<NodeRef> &outputs,
                       const Array<NodeRef> &alloc_map_list, const Array<NodeRef> &reuse_map_list,
                       const Array<NodeRef> &clean_op_map_list, const Array<NodeRef> &attrs_list, bool poly,
                       std::string target)
      : CompositeJsonList(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list, attrs_list,
                          poly, target) {}

  Stmt StitchFusion(const NodeRef &block_json, Map<std::string, NodeRef> &attrs) override {
    auto alloc_map = Downcast<Map<std::string, Array<NodeRef>>>(alloc_map_list_[block_json_idx_]);
    auto reuse_map = Downcast<Map<std::string, Array<NodeRef>>>(reuse_map_list_[block_json_idx_]);
    auto clean_op_map = Downcast<Map<std::string, Array<NodeRef>>>(clean_op_map_list_[block_json_idx_]);
    StitchAttrInfo stitch_attr;
    std::vector<Stmt> stitch_irs = LowerStitchIRs(block_json, stitch_attr, attrs);
    StitchBufAlloc buf_manager;
    buf_manager.BufferAllocReuse(stitch_irs, alloc_map, reuse_map, clean_op_map, outputs2args_);
    GetRealOutputs();
    auto stitched_ir = StitchFusionGpu(stitch_irs, merge_name_, stitch_attr, buf_manager.stitch_buffer_map,
                                       buf_manager.buf_within_op_map, buf_manager.allocate_revoke, real_outputs_);
    return stitched_ir;
  }

  Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs) override {
    return String2LowerStmt(json_str, attrs, 0, 0, false);
  }

  Stmt String2LowerStmt(const StringImm *json_str, const Map<std::string, NodeRef> &attrs, int grid_dims,
                        int block_dims, bool buffer_stitch) {
    CHECK(json_str);
    picojson::value v = String2Json(json_str->value);
    BuildInfo info;
    info.opt.stitch_ir_idx_ = each_ir_idx_;
    info.opt.stitch = buffer_stitch;
    ExtractBuildInfo(v, info);
    // ensure merge_name_ is the same as original json name
    if (merge_name_.empty()) merge_name_ = info.kernel_name;
    std::string sch_name = GetSchedule(info.tensors);
    const auto *sch_create = air::runtime::Registry::Get("select_cuda_scheduler");
    CHECK(sch_create != nullptr);
    Schedule sch = (*sch_create)(info.tensors, sch_name, poly_, grid_dims, block_dims, buffer_stitch);
    akg::BuildConfig config = akg::BuildConfig::Current();
    CHECK(config.defined());
    config->dump_pass_ir = getenv(GetDumpIRFlag().c_str()) != nullptr;
    // use each_ir_idx_ to distinct different subgraph
    std::string distinct_name = info.kernel_name + "_" + std::to_string(each_ir_idx_);
    Array<NodeRef> args, shape_vars, arg_list_0;
    Map<Tensor, Buffer> binds, binds_0;
    std::vector<size_t> split_index;
    auto stmt = LowerStmt(sch, info.args, shape_vars, distinct_name, info.in_binds, attrs, false, poly_, false, "cuda",
                          config, &args, &arg_list_0, &binds, &binds_0, &split_index, true);
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
                                   const Map<std::string, NodeRef> &attrs) {
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
      const std::function<Stmt(const StringImm *, const Map<std::string, NodeRef> &, bool, bool, std::vector<size_t> &)>
        f = std::bind(&String2LowerStmtSimple, _1, _2, _3, _4, _5);
      BufferStitchAttr stitch_attr_info(f);
      stitch_attr_info.GetBufferStitchAttr(stitch_json, op_v, attrs, poly_);
      auto dims = stitch_attr_info.dims;
      auto stitch_type = stitch_attr_info.stitch_type_;
      dim_array.push_back(dims);                         // save current dims into array.
      IrAttrInfo ir_attr_info = GetIRAttr(stitch_type, stitch_attr_info, ir_type_array, dim_array, attrs);
      DumpIRAttr(kernel_name, ir_attr_info, each_ir_idx_);
      ir_type_array.push_back(stitch_type);  // Note this should be done AFTER GetIrAttr.
      auto new_attrs = BindBlockAndThread(ir_attr_info.dims, poly_, ir_attr_info.attrs);
      if (each_ir_idx_ == 1) split_index_ = stitch_attr_info.split_index;
      new_attrs = SetAutoFuseAttr(split_index_, new_attrs);
      new_attrs.Set("enable_stitch_fusion", Expr(true));

      auto single_ir =
        String2LowerStmt(stitch_json.as<StringImm>(), new_attrs, ir_attr_info.grid_dims, ir_attr_info.block_dims, true);
      stitch_irs.emplace_back(InsertSync(single_ir));
    }
    stitch_attr.type_array = ir_type_array;
    return stitch_irs;
  }
};

Module CompositeWithJsonList(const Array<NodeRef> &json_str_node, const Array<NodeRef> &inputs,
                             const Array<NodeRef> &outputs, const Array<NodeRef> &alloc_map_list,
                             const Array<NodeRef> &reuse_map_list, const Array<NodeRef> &clean_op_map_list,
                             const Array<NodeRef> &attrs_list, bool poly, const std::string &target) {
  if (target == "cuda") {
    return CompositeJsonListGpu(json_str_node, inputs, outputs, alloc_map_list, reuse_map_list, clean_op_map_list,
                                attrs_list, poly, target)
      .Build();
  } else {
    CHECK(0) << "UNSUPPORTED TARGET: " << target;
    return Module();
  }
}

TVM_REGISTER_GLOBAL("composite_with_json_to_func").set_body_typed(CompositeWithJsonToFunc);
TVM_REGISTER_GLOBAL("composite_with_json").set_body_typed(CompositeWithJson);
TVM_REGISTER_GLOBAL("composite_with_json_list").set_body_typed(CompositeWithJsonList);
TVM_REGISTER_GLOBAL("composite_lower").set_body_typed(CompositeLower);
}  // namespace akg
