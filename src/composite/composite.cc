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

#include "composite/composite.h"

#include "build_module.h"
#include "composite/util.h"
#include "composite/parser.h"
#include "composite/optimize/optimize.h"
#include "composite/dump.h"

namespace akg {
class Emitter : public IRVisitor {
 public:
  explicit Emitter(BuildOpt &opt) : opt_(opt) {}

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
        if (opt_.tensor_map.count(c->func) == 0) {
          Tensor t = placeholder(c->args, c->type, c->name);
          opt_.tensor_map[c->func] = t;
        }
        real_input.push_back(opt_.tensor_map[c->func]);
      } else {
        real_input.push_back(input);
      }
    }
    if (op_name == "MatMul") {
      op_name = "BatchMatMul";
    }
    const auto *topi_f = air::runtime::Registry::Get(op_name);
    if (topi_f == nullptr && !opt_.target.empty()) {
      std::string target = opt_.target;
      target[0] = std::toupper(target[0]);
      topi_f = air::runtime::Registry::Get(target + op_name);
    }
    CHECK(topi_f) << "Akg topi has no op: " << op_name;
    if (op_name == "Reshape") {  // reshape's attr may have shape [-1], it will cause error.
      op_attrs_.Set("shape", op->args);
    }
    Tensor t = (*topi_f)(real_input, op_attrs_);
    if (op_name == "Assign") {
      EmitAssign(t, inputs[0]);
    }

    opt_.tensor_map[op->func] = t;
  }

  void EmitAssign(Tensor &t, const Expr &input) {
    // copy out to bind_input, bind_input is used to bind input[0]
    // d = Assign(a, b), bind_input = d, input0 = bind_input
    auto bind_input = compute(
      t->shape, [&](const Array<Var> &indices) { return t(indices); },
      "assign_tensor_" + std::to_string(assign_count_));
    opt_.tensor_map[bind_input->op] = bind_input;
    opt_.sch_only.emplace_back(bind_input);
    opt_.inplaces[bind_input->op] = input;
    assign_count_++;
  }

 private:
  BuildOpt &opt_;
  Map<std::string, NodeRef> op_attrs_;
  int assign_count_{0};
};

void CollectBinds(BuildInfo &info) {
  for (const auto &kv : info.opt.inplaces) {
    CHECK(info.opt.tensor_map.count(kv.first)) << kv.first->func_name() << " not in tensor map";
    CHECK(info.opt.tensor_map.count(kv.second.as<Call>()->func))
      << kv.second.as<Call>()->func->func_name() << " not in tensor map";
    auto first = info.opt.tensor_map[kv.first];
    auto second = info.opt.tensor_map[kv.second.as<Call>()->func];
    auto buf = decl_buffer(second->shape, second->dtype, second->op->name);
    info.in_binds.Set(first, buf);
    info.in_binds.Set(second, buf);
  }
}

void ProcessSames(BuildOpt &opt) {
  // b = func(a)
  // c = InplaceAssign(x, y, b)     c = b
  // d = InplaceAssign(i, j, c)     d = c
  bool changed = true;
  while (!opt.sames.empty() && changed) {
    changed = false;
    for (auto it = opt.sames.begin(); it != opt.sames.end();) {
      if (opt.tensor_map.count(it->second)) {
        opt.tensor_map[it->first] = opt.tensor_map[it->second];
        it = opt.sames.erase(it);
        changed = true;
      } else {
        ++it;
      }
    }
  }
}

void CollectInputs(BuildInfo &info) {
  for (const auto &input : info.input_names) {
    auto iter =
      std::find_if(info.opt.tensor_map.begin(), info.opt.tensor_map.end(),
                   [&input](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == input; });
    CHECK(iter != info.opt.tensor_map.end()) << "input Tensor " << input << " not built.";
    LOG(INFO) << "input: " << input << " " << iter->second;
    info.args.push_back(iter->second);
  }
}

void CollectOutputsAndComputes(BuildInfo &info) {
  int count = 0;
  for (const auto &output : info.output_names) {
    auto iter = std::find_if(
      info.opt.tensor_map.begin(), info.opt.tensor_map.end(),
      [&output](const std::pair<const FunctionRef, Tensor> &kv) { return kv.first->func_name() == output; });
    CHECK(iter != info.opt.tensor_map.end()) << "output Tensor " << output << " not built.";
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
    auto iter = std::find_if(info.opt.tensor_map.begin(), info.opt.tensor_map.end(),
                             [&inplace_itr](std::pair<const FunctionRef, Tensor> &kv) {
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

void CollectIsolatedInplaceTensor(BuildOpt &opt) {
  // tensors which have never be used before is isolated and not be created,
  // so we should create them after emit.
  for (const auto &kv : opt.inplaces) {
    auto c = kv.second.as<Call>();
    if (opt.tensor_map.find(c->func) == opt.tensor_map.end()) {
      opt.tensor_map[c->func] = placeholder(c->args, c->type, c->name);
    }
  }
}

void CollectBuildInfo(BuildInfo &info) {
  DumpBuildInfo(info);
  CollectIsolatedInplaceTensor(info.opt);
  CollectBinds(info);
  ProcessSames(info.opt);
  CollectInputs(info);
  CollectOutputsAndComputes(info);
  CollectSchOnlyComputes(info);
  DumpBuildInfo(info);
}

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info) {
  CHECK(input_json.is<picojson::object>());
  // 1. make stmt by input_json
  auto stmt = Parse(input_json, info);
  // 2. optimize stmt
  stmt = Optimize(stmt, info);
  if (info.opt.tuning) {
    return;
  }
  // 3. emit stmt by topi
  Emitter(info.opt).Visit(stmt);
  // 4. collect build info: args, compute, binds
  CollectBuildInfo(info);
}

Schedule GetScheduleWithBuildInfo(const BuildInfo &info) {
  Array<Operation> ops;
  std::for_each(info.tensors.begin(), info.tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  return create_schedule(ops);
}

Module CompositeWithJson(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Schedule sch = GetScheduleWithBuildInfo(info);
  auto config = GetConfig();
  if (attrs.find("kernel_name") != attrs.end()) {
    CHECK(attrs["kernel_name"]->IsInstance<StringImm>());
    info.kernel_name = attrs["kernel_name"].as<StringImm>()->value;
  }
  auto target = GetProcess(v);
  auto build_rst =
    akg::BuildToFunc(sch, info.args, Array<NodeRef>{}, info.kernel_name, info.in_binds, attrs, poly, target, config);
  CHECK(build_rst.defined());
  return BuildToModule(build_rst, target);
}

NodeRef CompositeLower(const std::string &json_str, const Map<std::string, NodeRef> &attrs) {
  picojson::value v = String2Json(json_str);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  Schedule sch = GetScheduleWithBuildInfo(info);
  auto config = GetConfig();
  bool tuning = attrs.find("tuning") != attrs.end();
  auto target = GetProcess(v);
  Array<NodeRef> shape_vars;

  if (attrs.find("ret_mode") != attrs.end()) {
    // This is used during auto tuning.
    if (attrs["ret_mode"]->IsInstance<IntImm>() && attrs["ret_mode"].as<IntImm>()->value == 1) {
      auto feature = std::vector<float>();
      // Set last arg to true to get pure stmt.
      auto stmt = Downcast<Stmt>(akg::Lower(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false,
                                            true, false, target, config, true));
      // Return args as well to get binds through get_binds api in python.
      return Array<NodeRef>({stmt, info.args});
    }
  }

  return akg::Lower(sch, info.args, shape_vars, info.kernel_name, info.in_binds, attrs, false, true, tuning, target,
                    config);
}

TVM_REGISTER_GLOBAL("composite_with_json").set_body_typed(CompositeWithJson);
TVM_REGISTER_GLOBAL("composite_lower").set_body_typed(CompositeLower);
}  // namespace akg
