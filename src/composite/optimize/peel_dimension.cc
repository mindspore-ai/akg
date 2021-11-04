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

#include "composite/optimize/peel_dimension.h"
#include "composite/utils/dimension_peeling.h"
#include "composite/utils/dump.h"
#include "composite/utils/dump_to_json.h"

namespace akg {
void DumpPeeledJson(const Stmt &peeled_stmt, const BuildInfo &info) {
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
  auto json = DumpToJson(peeled_stmt, build_info);
  DumpStr2File("stitch_info/" + info.kernel_name + "_peel_" + std::to_string(info.opt.stitch_ir_idx) + ".json", json);
}

Stmt PeelDimension::Run(const Stmt &stmt) {
  if (info_.opt.tuning || info_.opt.peel_info.peeling.empty()) {
    info_.opt.peel_info.stmt = stmt;
    return stmt;
  }
  DimensionPeeler peeler;
  peeler.Analyze(stmt);
  auto parsed_peeling = Str2Peeling(info_.opt.peel_info.peeling);
  Stmt peeled_stmt;
  if (info_.opt.peel_info.GetPeelTensors().empty()) {
    info_.opt.peel_info.SetPeelTensors(peeler.GetPeelTensors(parsed_peeling));
    peeled_stmt = peeler.GetPeelBody(parsed_peeling);
  } else {
    peeled_stmt = peeler.GetPeelBody(info_.opt.peel_info.GetPeelTensors());
  }
  DumpPeeledJson(peeled_stmt, info_);
  return peeled_stmt;
}
}  // namespace akg
