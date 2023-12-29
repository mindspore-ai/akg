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

#include "composite/extract_build_info.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include "composite/optimize/optimize.h"
#include "composite/parser.h"
#include "composite/emitter.h"
#include "composite/utils/dump_to_json.h"

namespace akg {
namespace {
constexpr auto kOptimizeForTBE = "optimize_for_tbe";

void CollectBinds(BuildInfo &info) {
  for (const auto &kv : info.opt.inplaces) {
    CHECK(info.opt.tensor_map.count(kv.first)) << kv.first->func_name() << " not in tensor map";
    Tensor first = info.opt.tensor_map[kv.first];
    Tensor second;
    if (auto c = kv.second.as<Call>()) {
      CHECK(info.opt.tensor_map.count(c->func)) << c->func->func_name() << " not in tensor map";
      second = info.opt.tensor_map[c->func];
    } else {
      second = Downcast<Tensor>(kv.second);
    }
    auto buf = decl_buffer(second->shape, second->dtype, second->op->name);
    info.in_binds.Set(first, buf);
    info.in_binds.Set(second, buf);
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
    if (auto c = kv.second.as<Call>()) {
      if (opt.tensor_map.find(c->func) == opt.tensor_map.end()) {
        opt.tensor_map[c->func] = placeholder(c->args, c->type, c->name);
      }
    }
  }
}

void CollectBuildInfo(BuildInfo &info) {
  DumpBuildInfo(info);
  CollectIsolatedInplaceTensor(info.opt);
  CollectBinds(info);
  CollectInputs(info);
  CollectOutputsAndComputes(info);
  CollectSchOnlyComputes(info);
  DumpBuildInfo(info);
}

bool EnableOptimizeForTBE(const Map<std::string, NodeRef> &attrs) {
  return (attrs.defined() && attrs.find(kOptimizeForTBE) != attrs.end() && attrs[kOptimizeForTBE].as<UIntImm>() &&
          attrs[kOptimizeForTBE].as<UIntImm>()->value != 0);
}
}  // namespace

void ExtractBuildInfo(const picojson::value &input_json, BuildInfo &info) {
  CHECK(input_json.is<picojson::object>());
  // 1. make stmt by input_json
  auto stmt = Parse(input_json, info);
  // optimize stmt for tbe
  bool optimize_for_tbe = EnableOptimizeForTBE(info.attrs);
  if (optimize_for_tbe) {
    // some passes will modify BuildInfo, and the modification can not be reused in AKG, so we copied BuildInfo
    BuildInfo build_info = info;
    auto s = OptimizeForTBE(stmt, build_info);
    DumpCompositeGraph(s, build_info);
  }
  // 2. optimize stmt
  stmt = Optimize(stmt, info);
  if (!optimize_for_tbe) {
    DumpCompositeGraph(stmt, info);
  }
  if (info.opt.tuning) {
    return;
  }
  // 3. emit stmt by topi
  Emitter(info.opt).Visit(stmt);
  // 4. collect build info: args, compute, binds
  CollectBuildInfo(info);
}
}  // namespace akg
