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
#include "codegen/lower.h"
#include <algorithm>
#include "codegen/stage_lower.h"
#include "schedule_pass.h"

namespace akg {
void ConfigDumpIr(const std::string &name, const BuildConfig &config, bool lower_list) {
  CHECK(!name.empty()) << "name is empty.";
  CHECK(find_if(name.begin(), name.end(), [](char c) { return !std::isalnum(c) && c != '_'; }) == name.end())
    << "kernel name contains invalid chars: " << name;

  PassMgr::ClearPassId();
  g_attrs.Set(kKernelName, StringImm::make(name));
  g_attrs.Set(kDumpPassIr, air::make_const(Int(32), config->dump_pass_ir));
  if (config->dump_pass_ir) {
    std::string dump_ir_dir;
    if (g_attrs.GetStr(kDumpIrDir, &dump_ir_dir)) {
      PassMgr::SetDir(dump_ir_dir);
    } else {
      PassMgr::SetDir(name);
    }
    CreateDir(PassMgr::GetDir());
    std::string dump_poly_dir;
    if (!g_attrs.GetStr(kDumpPolyDir, &dump_poly_dir) || lower_list) {
      dump_poly_dir = PassMgr::GetDir() + "/poly";
      g_attrs.Set(kDumpPolyDir, StringImm::make(dump_poly_dir));
      CreateDir(dump_poly_dir);
    }
  }
}

LowerData LowerDataNode::make() {
  auto n = make_node<LowerDataNode>();
  return LowerData(n);
}

LowerData LowerDataNode::make(Schedule sch, const Array<NodeRef> &args, const Map<Tensor, Buffer> &binds,
                              const Map<std::string, NodeRef> &attrs, const std::string &target,
                              const std::string &name, const BuildConfig &config, bool polyhedral, bool tuning,
                              bool simple_mode, const Array<NodeRef> &shape_vars, const Array<Integer> &split_index,
                              const Array<NodeRef> &arg_list_0, const Map<Tensor, Buffer> &binds_0) {
  auto node = make_node<LowerDataNode>();

  node->args = args;
  node->arg_list_0 = arg_list_0;
  node->attrs = attrs;
  node->binds = binds;
  node->binds_0 = binds_0;
  node->config = config;
  node->name = name;
  node->polyhedral = polyhedral;
  node->sch = sch;
  node->shape_vars = shape_vars;
  node->simple_mode = simple_mode;
  node->split_index = split_index;
  node->target = target;
  node->tuning = tuning;

  return LowerData(node);
}
TVM_REGISTER_NODE_TYPE(LowerDataNode);
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable).set_dispatch<LowerDataNode>([](const ObjectRef &node, IRPrinter *p) {
  auto *data = static_cast<const LowerDataNode *>(node.get());
  std::stringstream ss;
  ss << "{" << std::endl;
  ss << "  name: " << data->name << std::endl;

  ss << "  args: " << std::endl;
  for (auto arg : data->args) {
    ss << "    " << arg << std::endl;
  }

  ss << "  arg_list_0: " << std::endl;
  for (auto arg_list : data->arg_list_0) {
    ss << "    " << arg_list << std::endl;
  }

  ss << "  binds: " << std::endl;
  for (auto iter : data->binds) {
    ss << "    " << iter.first << ": " << iter.second << std::endl;
  }

  ss << "  binds_0: " << std::endl;
  for (auto iter : data->binds_0) {
    ss << "    " << iter.first << ": " << iter.second << std::endl;
  }
  ss << "  attrs: " << std::endl;
  for (auto iter : data->attrs) {
    ss << "    " << iter.first << ": " << iter.second << std::endl;
  }

  ss << "  split index: [";
  for (auto i : data->split_index) {
    ss << i << ", ";
  }
  ss << "]" << std::endl;

  ss << "  polyhedral: " << data->polyhedral << std::endl;
  ss << "  simple mode: " << data->simple_mode << std::endl;
  ss << "  target: " << data->target << std::endl;
  ss << "  tuning: " << data->tuning << std::endl;
  ss << "}";
  p->stream << ss.str();
});

LowerImpl &LowerImpl::Instance() {
  static LowerImpl lower_impl;
  return lower_impl;
}

void LowerImpl::Register(const std::string &target, std::function<NodeRef(const LowerData &, bool)> impl) {
  if (impls_.find(target) != impls_.end()) {
    LOG(ERROR) << "Impl for " << target << " is all ready exist!";
    return;
  }

  impls_.insert({target, impl});
}

std::string GetErrorHint(const std::string &target) {
  static std::unordered_map<std::string, std::string> error_hint = {
    {"cce",
     "Can not enable target cce, because akg Ascend back-end's binary file is not linked to"
     " libakg.so during the compiling process, please check the following cases:\n"
     "case 1: If compile akg with -DUSE_KC_AIR=1, check if libakg_ext.a exists in"
     " akg_source_dir(CMAKE_CURRENT_SOURCE_DIR) or akg_build_dir(CMAKE_CURRENT_BINARY_DIR)."
     " If not, you need:\n"
     "        1. Compile libakg_ext.a by yourself, put it to akg_source_dir or akg_build_dir\n"
     "        2. Re-compile the source codes\n"
     "case 2: If compile akg without -DUSE_KC_AIR=1(compiling akg from mindspore belongs to"
     " this case), then you can perform the following steps:\n"
     "        1. Check if git-lfs is installed, if not, install git-lfs, refer"
     " https://github.com/git-lfs/git-lfs/wiki/installation\n"
     "        2. After installing git lfs, executing the following commands:\n"
     "           cd akg_source_dir (e.g. cd /home/user_name/akg)\n"
     "           git lfs pull\n"
     "        3. Re-compile the source codes"}};

  if (error_hint.find(target) == error_hint.end()) {
    return std::string("Unsupport lower for ") + target;
  }

  return error_hint[target];
}

NodeRef LowerImpl::Run(const LowerData &data, bool get_stmt) {
  Target target = Target::Create(data->target);
  CHECK(impls_.find(target->target_name) != impls_.end()) << GetErrorHint(target->target_name);
  return impls_[target->target_name](data, get_stmt);
}

Stmt LowerInitWithSchedule(LowerData &data) {
  CHECK(data->sch.defined()) << "sch is not defined.";
  ConfigDumpIr(data->name + "_0", data->config, true);

  GetBinds(data->args, data->binds, data->config, &data->arg_list_0, &data->binds_0);
  // Phase 0
  Target target_platform = Target::Create(data->target);
  if (data->polyhedral) {
    if (g_attrs.GetBool(kEnableAutoInline, true)) {
      akg::schedule::AutoInline(data->sch, target_platform, g_attrs.GetBool(kEnableCSE, false));
    }
    if ((target_platform->device_type == kDLGPU || target_platform->device_type == kDLCPU) &&
      g_attrs.GetBool(kEnableAutoFuse, true)) {
      std::vector<size_t> split_index;
      akg::schedule::AutoFuse(data->sch, g_attrs.GetStr(kAutoFuseSplit, ""), split_index,
                              g_attrs.GetBool("enable_stitch_fusion", 0));
      data->split_index = Array<Integer>(split_index.begin(), split_index.end());
    }
  }

  data->sch = data->sch.normalize();
  auto bounds = air::schedule::InferBound(data->sch);
  Stmt stmt = make_pass("schedule.ScheduleOps", data->sch, bounds, false);
  if (g_attrs.count(kTensorAttrs) > 0) {
    const auto tensor_attrs = Downcast<Map<Tensor, Map<std::string, NodeRef>>>(g_attrs[kTensorAttrs]);
    stmt = NEXT_PASS(AddTensorAttrs, stmt, tensor_attrs);
  }
  stmt = NEXT_PASS(TensorAccessRewrite, stmt);
  return stmt;
}

}  // namespace akg
