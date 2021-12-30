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

#include "composite/extract_build_info.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include "composite/optimize/optimize.h"
#include "composite/parser.h"
#include "composite/emitter.h"

namespace akg {
namespace {
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

// Make noninline_indeed tensors not auto-inlined by inserting to args
void InsertNoInlineTensors2Outputs(BuildInfo &info) {
  for (const auto &candidate : info.opt.noinline_candidate) {
    auto iter = std::find_if(info.args.begin(), info.args.end(), [&candidate](const NodeRef &arg) {
      return candidate->op->name == Downcast<Tensor>(arg)->op->name;
    });
    if (iter == info.args.end()) {
      info.args.push_back(candidate);
      info.opt.noinline_indeed.push_back(candidate);
    }
    auto it = std::find_if(info.tensors.begin(), info.tensors.end(),
                           [&candidate](const Tensor &tensor) { return candidate->op->name == tensor->op->name; });
    if (it == info.tensors.end()) {
      info.tensors.push_back(candidate);
    }
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
  InsertNoInlineTensors2Outputs(info);
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
}  // namespace

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
}  // namespace akg
