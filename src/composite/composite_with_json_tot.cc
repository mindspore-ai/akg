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
#include "codegen/stage_lower.h"
#include "pass/recover_tot.h"
#include "pass/replace_tot.h"

namespace akg {
namespace lower {
void TotReplace(NodeRef &node_ref, LowerData &data, Map<std::string, Map<std::string, NodeRef>> *tot_attr,
                const Array<Tensor> &noinline_indeed) {
  Stmt stmt = Downcast<Stmt>(node_ref);
  Array<NodeRef> res = ir::ReplaceTot(stmt);
  node_ref = res[0];
  *tot_attr = Downcast<Map<std::string, Map<std::string, NodeRef>>>(res[1]);
  // remove noinline_indeed from args
  Array<NodeRef> filter_args;
  for (const auto &old_arg : data->arg_list_0) {
    auto iter = std::find_if(noinline_indeed.begin(), noinline_indeed.end(), [old_arg](const Tensor &noinline) {
      if (old_arg.as<BufferNode>() != nullptr) {
        return Downcast<Buffer>(old_arg)->name == noinline->op->name;
      } else if (old_arg.as<TensorNode>() != nullptr) {
        return Downcast<Tensor>(old_arg)->op->name == noinline->op->name;
      }
      return false;
    });
    if (iter == noinline_indeed.end()) {
      filter_args.push_back(old_arg);
    }
  }
  data->arg_list_0 = filter_args;
  PassMgr::SetArgs(data->arg_list_0);
  // remove no_inline from binds_0
  Map<Tensor, Buffer> filter_binds_0;
  for (const auto &old_kv : data->binds_0) {
    auto iter = std::find_if(noinline_indeed.begin(), noinline_indeed.end(),
                             [old_kv](const Tensor &noinline) { return noinline->op->name == old_kv.first->op->name; });
    if (iter == noinline_indeed.end()) {
      filter_binds_0.Set(old_kv.first, old_kv.second);
    }
  }
  data->binds_0 = filter_binds_0;
}

void TotRecover(NodeRef &node_ref, LowerData &data, Map<std::string, Map<std::string, NodeRef>> *tot_attr) {
  Stmt stmt = Downcast<Stmt>(node_ref);
  stmt = ir::RecoverTot(stmt, *tot_attr, data->binds_0);
  node_ref = stmt;
}

Module CompositeWithJsonTotGpu(const std::string &json_str, const Map<std::string, NodeRef> &attrs, bool poly) {
  picojson::value v = String2Json(json_str);
  auto target = GetProcess(v);
  BuildInfo info;
  ExtractBuildInfo(v, info);
  if (attrs.find("kernel_name") != attrs.end()) {
    CHECK(attrs["kernel_name"]->IsInstance<StringImm>());
    info.kernel_name = attrs["kernel_name"].as<StringImm>()->value;
  }
  Array<Tensor> noinline_indeed = info.opt.noinline_indeed;
  Map<std::string, Map<std::string, NodeRef>> tot_attr;
  auto TotReplaceBind = std::bind(TotReplace, std::placeholders::_1, std::placeholders::_2, &tot_attr, noinline_indeed);
  auto TotRecoverBind = std::bind(TotRecover, std::placeholders::_1, std::placeholders::_2, &tot_attr);
  LowerData data = LowerDataNode::make(GetScheduleWithBuildInfo(info), info.args, info.in_binds, attrs, target,
                                       info.kernel_name, GetConfig(), poly);
  auto build_rst = StageLower(data)
                     .RunTo(StageType::Tuning)
                     .ApplyMutator(TotReplaceBind)
                     .RunTo(StageType::Poly)
                     .ApplyMutator(TotRecoverBind)
                     .RunTo()
                     .Node();
  CHECK(build_rst.defined());
  auto buildrst = BuildRstNode::make(build_rst, info.kernel_name);
  return BuildToModule(buildrst, target);
}
}  // namespace lower

TVM_REGISTER_GLOBAL("composite_with_json_tot").set_body_typed(lower::CompositeWithJsonTotGpu);
}  // namespace akg
