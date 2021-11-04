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

#include <string>

#include "picojson.h"
#include "composite/lower_tree/base_node.h"
#include "composite/utils/util.h"
#include "composite/utils/dimension_peeling.h"
#include "composite/extract_build_info.h"

namespace akg {
namespace lower {
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
    info.opt.fold_dim = GetBoolValueFromMap(attrs, "fold_dim");
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
}  // namespace lower

TVM_REGISTER_GLOBAL("get_peeled_body").set_body_typed(lower::GetPeeledBody);
TVM_REGISTER_GLOBAL("composite_peel_analyze").set_body_typed(lower::CompositePeelAnalyze);
}  // namespace akg
