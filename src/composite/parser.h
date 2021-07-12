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
#ifndef COMPOSITE_PARSER_H_
#define COMPOSITE_PARSER_H_
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "composite/util.h"

namespace akg {
std::tuple<std::string, std::string, picojson::array, picojson::array, picojson::array> ParseInputJson(
  const picojson::value &input_json);
void ParseInputTensors(const picojson::array &input_descs, std::vector<std::string> &input_tensors);
void ParseOutputTensors(const picojson::array &output_descs, std::vector<std::string> &output_tensors);
std::vector<OpDesc> ParseOpDesc(const std::string &json_str);
std::string ParseKernelName(const std::string &json_str);
Stmt MakeStmt(const std::vector<OpDesc> &op_descs);
Stmt Parse(const picojson::value &input_json, BuildInfo &info);

class OpDescsParser {
 public:
  OpDescsParser(picojson::array op_descs_json, const std::vector<std::string> &input_tensors,
                const std::vector<std::string> &output_tensors)
      : op_descs_json_(std::move(op_descs_json)), input_tensors_(input_tensors), output_tensors_(output_tensors) {}
  ~OpDescsParser() = default;

  void Parse() {
    for (const auto &item : op_descs_json_) {
      CHECK(item.is<picojson::object>());
      const picojson::object &op_desc = item.get<picojson::object>();
      ParseOpDesc(op_desc);
    }
  }

  void Dump() {
    LOG(INFO) << "========OP_DESCS========";
    for (const auto &item : op_descs_) {
      LOG(INFO) << "op_name: " << item.op_name;
      for (const auto &attr : item.attrs) {
        LOG(INFO) << "attrs: " << attr.first << ":" << attr.second;
      }
      for (const auto &input : item.input_descs) {
        LOG(INFO) << "input: " << input;
      }
      for (const auto &output : item.output_descs) {
        LOG(INFO) << "output: " << output;
      }
      for (const auto &input_info : item.input_tensor_info) {
        LOG(INFO) << "input_info: ";
        LOG(INFO) << input_info.name_;
        LOG(INFO) << input_info.shape_;
      }
      for (const auto &output_info : item.output_tensor_info) {
        LOG(INFO) << "output_info: ";
        LOG(INFO) << output_info.name_;
        LOG(INFO) << output_info.shape_;
      }
    }
  }

 public:
  std::vector<OpDesc> op_descs_;
  FuncRefList input_funcs_;
  FuncRefList output_funcs_;

 private:
  const picojson::array op_descs_json_;
  const std::vector<std::string> input_tensors_;
  const std::vector<std::string> output_tensors_;
  std::unordered_map<std::string, Tensor> tensor_map_;

 private:
  static void ParseTensorValue(const picojson::value &tensor_value, const std::string &tensor_name,
                               const Array<Expr> &shape, const Type &type, Array<NodeRef> &input_output) {
    CHECK_EQ(shape.size(), 1) << "We should not make a expr for a not const tensor.";
    CHECK(Equal(shape[0], Expr(1))) << "We should not make a expr for a not const tensor.";
    CHECK(!tensor_value.is<picojson::null>()) << "We should has default value of tensor(expr): " << tensor_name;
    if (tensor_value.is<double>()) {
      input_output.push_back(make_const(type, tensor_value.get<double>()));
    } else if (tensor_value.is<int64_t>()) {
      input_output.push_back(make_const(type, tensor_value.get<int64_t>()));
    } else {
      CHECK(0) << "Unknown value type of tensor: " << tensor_name;
    }
  }

  void MakeTensors(const std::vector<TensorInfo> &tensor_info, Array<NodeRef> &tensors) {
    for (const auto &info : tensor_info) {
      if (info.has_value_) {
        // In case when current tensor already has value information
        ParseTensorValue(info.value_, info.name_, info.shape_, info.dtype_, tensors);
        continue;
      }
      if (tensor_map_.count(info.name_) == 0) {
        Tensor t = placeholder(info.shape_, info.dtype_, info.name_);
        tensor_map_[info.name_] = t;
        if (std::find(input_tensors_.begin(), input_tensors_.end(), info.name_) != input_tensors_.end()) {
          input_funcs_.emplace_back(t->op);
        }
        if (std::find(output_tensors_.begin(), output_tensors_.end(), info.name_) != output_tensors_.end()) {
          output_funcs_.emplace_back(t->op);
        }
      }
      tensors.push_back(tensor_map_[info.name_]);
    }
  }

  void ParseTensorInfo(const picojson::object &tensor_desc, std::vector<TensorInfo> &tensor_info) {
    TensorInfo info;
    for (const auto &item : tensor_desc) {
      if (item.first == "tensor_name") {
        CHECK(item.second.is<std::string>());
        info.name_ = item.second.get<std::string>();
      } else if (item.first == "format") {
        CHECK(item.second.is<std::string>());
        info.format_ = item.second.get<std::string>();
      } else if (item.first == "shape") {
        CHECK(item.second.is<picojson::array>());
        const picojson::array &dims = item.second.get<picojson::array>();
        for (const auto &dim : dims) {
          CHECK(dim.is<int64_t>());
          info.shape_.push_back(Expr(static_cast<int>(dim.get<int64_t>())));
        }
      } else if (item.first == "data_type") {
        CHECK(item.second.is<std::string>());
        std::string dtype_str = item.second.get<std::string>();
        if (type_mapping.find(dtype_str) == type_mapping.end()) {
          LOG(FATAL) << "Not support dtype str " << dtype_str;
        }
        info.dtype_ = type_mapping[dtype_str];
      } else if (item.first == "value" && !item.second.is<picojson::null>()) {
        info.has_value_ = true;
        info.value_ = item.second;
      }
    }

    tensor_info.emplace_back(info);
  }

  void ParseTensorFormat(const std::vector<TensorInfo> &tensor_info, Map<std::string, NodeRef> &attrs) {
    for (const auto &info : tensor_info) {
      if (!info.format_.empty()) {
        auto key = CreateDataFormatKey(info.name_);
        auto format = StringImm::make(info.format_);
        if (attrs.find(key) != attrs.end()) {
          LOG(WARNING) << key << " already exists in attrs";
        }
        attrs.Set(key, format);
      }
    }
  }

  void ParseInputTensors(const picojson::array &tensor_descs, OpDesc &op_desc_info) {
    std::vector<TensorInfo> tensor_info;
    for (const auto &tensor_desc_l0 : tensor_descs) {
      CHECK(tensor_desc_l0.is<picojson::array>());
      const picojson::array &tensor_desc_l1 = tensor_desc_l0.get<picojson::array>();
      for (const auto &tensor_desc : tensor_desc_l1) {
        CHECK(tensor_desc.is<picojson::object>());
        const picojson::object &tensor_desc_info = tensor_desc.get<picojson::object>();
        ParseTensorInfo(tensor_desc_info, tensor_info);
      }
    }
    // Gather data format information of input tensors
    ParseTensorFormat(tensor_info, op_desc_info.attrs);
    op_desc_info.input_tensor_info = tensor_info;
    MakeTensors(tensor_info, op_desc_info.input_descs);
  }

  void ParseOutputTensors(const picojson::array &tensor_descs, OpDesc &op_desc_info) {
    std::vector<TensorInfo> tensor_info;
    for (const auto &tensor_desc : tensor_descs) {
      CHECK(tensor_desc.is<picojson::object>());
      const picojson::object &tensor_desc_info = tensor_desc.get<picojson::object>();
      ParseTensorInfo(tensor_desc_info, tensor_info);
    }
    // Gather data format information of output tensors
    ParseTensorFormat(tensor_info, op_desc_info.attrs);
    op_desc_info.output_tensor_info = tensor_info;
    MakeTensors(tensor_info, op_desc_info.output_descs);
  }

  void ParseOpDesc(const picojson::object &op_desc) {
    OpDesc op_desc_info;
    auto it = op_desc.find("name");
    if (it != op_desc.end()) {
      op_desc_info.op_name = it->second.get<std::string>();
    }
    it = op_desc.find("attr");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &attrs = it->second.get<picojson::array>();
      ParseAttrs(attrs, &op_desc_info.attrs);
    }
    it = op_desc.find("input_desc");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &input_descs = it->second.get<picojson::array>();
      ParseInputTensors(input_descs, op_desc_info);
    }
    it = op_desc.find("output_desc");
    if (it != op_desc.end() && it->second.is<picojson::array>()) {
      const picojson::array &output_descs = it->second.get<picojson::array>();
      ParseOutputTensors(output_descs, op_desc_info);
    }
    // In some scenarios, for example, FRACTAL_NZ is transfered to DefaultFormat,
    // for the TransData operator to execute, output_shape(original_shape) information is necessary.
    if (op_desc_info.op_name == "TransData") {
      CHECK_EQ(op_desc_info.output_tensor_info.size(), 1);
      auto output_shape = op_desc_info.output_tensor_info[0].shape_;
      op_desc_info.attrs.Set("output_shape", output_shape);
    }
    op_descs_.emplace_back(op_desc_info);
  }

  static void ParseAttrs(const picojson::array &arr, Map<std::string, NodeRef> *op_attrs) {
    CHECK(op_attrs) << "input op_attrs is invalid.";
    for (const auto &item : arr) {
      CHECK(item.is<picojson::object>());
      const picojson::object &obj = item.get<picojson::object>();
      std::string name;
      NodeRef value;
      bool name_found = false;
      bool value_found = false;
      for (const auto &kv : obj) {
        // parse attr name
        if (kv.first == "name") {
          name = kv.second.get<std::string>();
          name_found = true;
          continue;
        }
        if (kv.first != "value") {
          continue;
        }
        // parse attr value
        value_found = true;
        if (kv.second.is<picojson::array>()) {
          Array<NodeRef> arr_v;
          const picojson::array &arr_s = kv.second.get<picojson::array>();
          for (const auto &v : arr_s) {
            if (v.is<int64_t>()) {
              arr_v.push_back(Integer(static_cast<int>(v.get<int64_t>())));
            } else if (v.is<std::string>()) {
              arr_v.push_back(StringImm::make(v.get<std::string>()));
            } else {
              LOG(FATAL) << "Not parsed type in array attr.";
            }
          }
          value = arr_v;
        } else if (kv.second.is<bool>()) {
          value = make_const(Int(1), kv.second.get<bool>());
        } else if (kv.second.is<int64_t>()) {
          value = Integer(static_cast<int>(kv.second.get<int64_t>()));
        } else if (kv.second.is<std::string>()) {
          value = StringImm::make(kv.second.get<std::string>());
        } else {
          LOG(FATAL) << "Not parsed type in op_attrs.";
        }
      }
      CHECK(name_found);
      CHECK(value_found);
      op_attrs->Set(name, value);
    }
  }
};
}  // namespace akg
#endif  // COMPOSITE_PARSER_H_
