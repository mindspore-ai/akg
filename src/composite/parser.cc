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
#include "composite/parser.h"
#include <string>

namespace akg {
std::tuple<std::string, std::string, picojson::array, picojson::array, picojson::array> ParseInputJson(
  const picojson::value &input_json) {
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_desc;
  std::string kernel_name;
  std::string target;
  const picojson::value::object &input_obj = input_json.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op") {
      CHECK(item.second.is<std::string>());
      kernel_name = item.second.get<std::string>();
    } else if (item.first == "process") {
      CHECK(item.second.is<std::string>());
      target = item.second.get<std::string>();
    } else if (item.first == "input_desc") {
      if (item.second.is<picojson::null>()) {
        continue;
      }
      CHECK(item.second.is<picojson::array>());
      input_desc = item.second.get<picojson::array>();
    } else if (item.first == "output_desc") {
      CHECK(item.second.is<picojson::array>());
      output_desc = item.second.get<picojson::array>();
    } else if (item.first == "op_desc") {
      CHECK(item.second.is<picojson::array>());
      op_desc = item.second.get<picojson::array>();
    }
  }
  return std::make_tuple(kernel_name, target, input_desc, output_desc, op_desc);
}

void ParseInputTensors(const picojson::array &input_descs, std::vector<std::string> &input_tensors) {
  for (auto input_desc = input_descs.begin(); input_desc != input_descs.end(); ++input_desc) {
    CHECK(input_desc->is<picojson::array>());
    const picojson::array &input_desc_array = input_desc->get<picojson::array>();
    CHECK(input_desc_array.begin()->is<picojson::object>());
    const picojson::object &input_desc_obj = input_desc_array.begin()->get<picojson::object>();
    for (const auto &item : input_desc_obj) {
      if (item.first != "tensor_name") continue;
      CHECK(item.second.is<std::string>());
      std::string tensor_name = item.second.get<std::string>();
      input_tensors.emplace_back(tensor_name);
    }
  }
}

void ParseOutputTensors(const picojson::array &output_descs, std::vector<std::string> &output_tensors) {
  for (auto output_desc = output_descs.begin(); output_desc != output_descs.end(); ++output_desc) {
    CHECK(output_desc->is<picojson::object>());
    const picojson::object &output_desc_obj = output_desc->get<picojson::object>();
    for (const auto &item : output_desc_obj) {
      if (item.first != "tensor_name") continue;
      CHECK(item.second.is<std::string>());
      std::string tensor_name = item.second.get<std::string>();
      output_tensors.emplace_back(tensor_name);
    }
  }
}

std::string ParseKernelName(const std::string &json_str) {
  std::string kernel_name;
  picojson::value v = String2Json(json_str);
  picojson::array op_desc;
  const picojson::value::object &input_obj = v.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op") {
      CHECK(item.second.is<std::string>());
      kernel_name = item.second.get<std::string>();
    }
  }
  return kernel_name;
}

std::vector<OpDesc> ParseOpDesc(const std::string &json_str) {
  picojson::value v = String2Json(json_str);
  picojson::array op_desc;
  const picojson::value::object &input_obj = v.get<picojson::object>();
  for (const auto &item : input_obj) {
    if (item.first == "op_desc") {
      CHECK(item.second.is<picojson::array>());
      op_desc = item.second.get<picojson::array>();
    }
  }
  std::vector<std::string> input_tensors;
  std::vector<std::string> output_tensors;
  auto parser = OpDescsParser(op_desc, input_tensors, output_tensors);
  parser.Parse();
  return parser.op_descs_;
}

Stmt MakeStmt(const std::vector<OpDesc> &op_descs) {
  std::vector<Stmt> stmts;
  for (const auto &op_desc : op_descs) {
    Array<Expr> input;
    for (const auto &item : op_desc.input_descs) {
      if (item.as<TensorNode>()) {
        auto t = Downcast<Tensor>(item);
        input.push_back(Call::make(t->dtype, t->op->name, t->shape, Call::CallType::Halide, t->op));
      } else {
        input.push_back(Downcast<Expr>(item));
      }
    }

    Tensor output = Downcast<Tensor>(op_desc.output_descs[0]);
    auto op_name = op_desc.op_name;
    auto stmt =
      Provide::make(output->op, 0, Call::make(Int(32), op_name, input, Call::CallType::PureIntrinsic), output->shape);
    if (!op_desc.attrs.empty()) {
      stmt = AttrStmt::make(op_desc.attrs, "attrs", Expr(1), stmt);
    }
    stmts.emplace_back(stmt);
  }
  return Block::make(stmts);
}

Stmt Parse(const picojson::value &input_json, BuildInfo &info) {
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_descs;
  std::string kernelname;
  std::string target;
  // 1. parse input json
  std::tie(kernelname, target, input_desc, output_desc, op_descs) = ParseInputJson(input_json);
  info.kernel_name = kernelname;
  ParseInputTensors(input_desc, info.input_names);
  ParseOutputTensors(output_desc, info.output_names);
  // 2. parse op descs
  auto parser = OpDescsParser(op_descs, info.input_names, info.output_names);
  parser.Parse();
  info.opt.input_funcs = parser.input_funcs_;
  info.opt.output_funcs = parser.output_funcs_;
  info.opt.target = target;
  return MakeStmt(parser.op_descs_);
}
}  // namespace akg
