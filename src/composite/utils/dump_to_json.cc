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
#include "composite/utils/dump_to_json.h"
#include <fstream>
#include <sys/stat.h>
#include <climits>
#include "picojson.h"
#include "composite/utils/util.h"

namespace akg {
namespace {
constexpr auto kDumpCompositeGraph = "dump_composite_graph";
constexpr auto kGlobalAttrs = "global_attrs";

class DumpToJsonVisitor : public IRVisitor {
 public:
  explicit DumpToJsonVisitor(const Map<std::string, NodeRef> &info) : build_info_(info) {}
  ~DumpToJsonVisitor() override = default;

  std::string Dump(const Stmt &stmt) {
    picojson::object desc;
    desc["composite"] = picojson::value(true);
    desc["op"] = picojson::value(GetStringValueFromBuildInfo("op"));
    desc["process"] = picojson::value(GetStringValueFromBuildInfo("process"));

    // Collect inputs and outputs name from build info
    InitInputsOutputs();

    // Collect inplace tensor names from build info
    CollectInplaceInfo();

    IRVisitor::Visit(stmt);
    desc["op_desc"] = picojson::value(op_desc_);
    FillInputsOutputs();
    desc["input_desc"] = picojson::value(input_desc_);
    desc["output_desc"] = picojson::value(output_desc_);
    if (!global_attrs_.empty()) {
      desc[kGlobalAttrs] = picojson::value(global_attrs_);
    }

    picojson::value v(desc);
    return v.serialize();
  }

  void Visit_(const AttrStmt *op) final {
    if (op->node.as<StrMapNode>() != nullptr) {
      if (op->attr_key == kGlobalAttrs) {
        // Collect global attr
        auto attrs = ParseAttr(Downcast<Map<std::string, NodeRef>>(op->node));
        if (!attrs.empty()) {
          for (const auto &attr : attrs) {
            global_attrs_.push_back(attr);
          }
        }
      } else {
        // Collect op attr
        op_attr_ = Downcast<Map<std::string, NodeRef>>(op->node);
      }
    }
    Visit(op->body);
  }

  void Visit_(const Provide *op) final {
    auto call = op->value.as<Call>();
    CHECK(call != nullptr);
    picojson::object op_desc;

    // Parse op name
    op_desc["name"] = picojson::value(call->name);

    // Parse op attr
    auto attr = ParseAttr(op_attr_);
    if (!attr.empty()) {
      op_desc["attr"] = picojson::value(attr);
    }

    // Parse op input desc
    auto args = call->args;
    picojson::array input_desc;
    for (size_t i = 0; i < args.size(); ++i) {
      picojson::object input;
      if (auto c = args[i].as<Call>()) {
        auto input_name = c->name;
        input = ParseTensor(input_name, c->type, c->args);
        // Gather json inputs
        if (io_.find(input_name) != io_.end()) {
          io_[input_name] = input;
        }
      } else {
        std::string input_name = "value_input_" + std::to_string(i);
        Array<Expr> shape{IntImm::make(Int(32), 1)};
        input = ParseTensor(input_name, args[i].type(), shape, args[i]);
      }
      picojson::array input_list{picojson::value(input)};
      input_desc.push_back(picojson::value(input_list));
    }
    op_desc["input_desc"] = picojson::value(input_desc);

    // Parse op output desc
    picojson::array output_desc;
    picojson::object output;
    CHECK(op->func.defined());
    auto output_name = op->func->func_name();
    output = ParseTensor(output_name, call->type, op->args);
    // Gather json outputs
    if (io_.find(output_name) != io_.end()) {
      io_[output_name] = output;
    }
    output_desc.push_back(picojson::value(output));
    op_desc["output_desc"] = picojson::value(output_desc);

    // Save current op
    op_desc_.push_back(picojson::value(op_desc));

    // Clear op attr
    op_attr_ = {};
  }

 private:
  void InitInputsOutputs() {
    if (build_info_.find("input_names") != build_info_.end()) {
      auto input_names = Downcast<Array<Expr>>(build_info_["input_names"]);
      for (const auto &name : input_names) {
        auto it = name.as<StringImm>();
        CHECK(it != nullptr);
        input_names_.push_back(it->value);
        io_[it->value] = {};
      }
    }

    if (build_info_.find("output_names") != build_info_.end()) {
      auto output_names = Downcast<Array<Expr>>(build_info_["output_names"]);
      for (const auto &name : output_names) {
        auto it = name.as<StringImm>();
        CHECK(it != nullptr);
        output_names_.push_back(it->value);
        io_[it->value] = {};
      }
    }
  }

  void FillInputsOutputs() {
    for (const auto &name : input_names_) {
      if (io_.find(name) == io_.end()) {
        LOG(WARNING) << "Input name " << name << " is not found in stmt.";
      } else if (!io_[name].empty()) {
        picojson::array input{picojson::value(io_[name])};
        input_desc_.push_back(picojson::value(input));
      }
    }

    for (const auto &name : output_names_) {
      if (io_.find(name) == io_.end()) {
        LOG(WARNING) << "Output name " << name << " is not found in stmt.";
      } else {
        output_desc_.push_back(picojson::value(io_[name]));
      }
    }
  }

  void CollectInplaceInfo() {
    if (build_info_.find("inplace") != build_info_.end()) {
      auto inplace = Downcast<Map<std::string, Expr>>(build_info_["inplace"]);
      for (const auto &it : inplace) {
        auto to = it.second.as<StringImm>();
        CHECK(to != nullptr);
        inplace_[it.first] = to->value;
      }
    }
  }

  picojson::value ParseConst(const Expr &e) {
    picojson::value v;
    if (e.as<StringImm>() != nullptr) {
      v = picojson::value(e.as<StringImm>()->value);
    } else if (e.as<IntImm>() != nullptr) {
      v = picojson::value(e.as<IntImm>()->value);
    } else if (e.as<FloatImm>() != nullptr) {
      v = picojson::value(e.as<FloatImm>()->value);
    } else {
      LOG(FATAL) << "Not parsed type " << e;
    }
    return v;
  }

  picojson::array ParseArray(const Array<Expr> &array) {
    picojson::array ret;
    for (const auto &it : array) {
      ret.push_back(ParseConst(it));
    }
    return ret;
  }

  picojson::array ParseAttr(const Map<std::string, NodeRef> &attr_map) {
    picojson::array attr;
    if (attr_map.empty()) {
      return attr;
    }
    for (const auto &it : attr_map) {
      picojson::object cur_attr;
      auto name = it.first;
      auto value = it.second;
      cur_attr["name"] = picojson::value(name);
      if (value.as<air::ArrayNode>() != nullptr) {
        auto v = Downcast<Array<Expr>>(value);
        cur_attr["value"] = picojson::value(ParseArray(v));
      } else if (value.as<ExprNode>() != nullptr) {
        auto v = Downcast<Expr>(value);
        cur_attr["value"] = ParseConst(v);
      } else {
        LOG(FATAL) << "Not parsed type " << value << " in op attr " << attr_map;
      }
      attr.push_back(picojson::value(cur_attr));
    }
    return attr;
  }

  picojson::object ParseTensor(const std::string &name, const air::DataType &type, const Array<Expr> &shape,
                               Expr value = Expr()) {
    picojson::object tensor;
    tensor["tensor_name"] = picojson::value(name);
    tensor["format"] = picojson::value("DefaultFormat");
    tensor["data_type"] = picojson::value(type2string(type));
    tensor["shape"] = picojson::value(ParseArray(shape));
    if (value.defined()) {
      tensor["value"] = ParseConst(value);
    }
    // Save inplace tensor name if exists.
    if (inplace_.find(name) != inplace_.end()) {
      tensor["inplace_to"] = picojson::value(inplace_[name]);
    }
    return tensor;
  }

  std::string GetFormatFromAttr(const std::string &name) {
    std::string format("DefaultFormat");
    auto format_key = CreateDataFormatKey(name);
    if (op_attr_.find(format_key) != op_attr_.end()) {
      CHECK(op_attr_[format_key].as<StringImm>());
      format = op_attr_[format_key].as<StringImm>()->value;
    }
    return format;
  }

  std::string GetStringValueFromBuildInfo(const std::string &key) {
    CHECK(build_info_.find(key) != build_info_.end()) << "Key " << key << " not found in build info " << build_info_;
    CHECK(build_info_[key].as<StringImm>() != nullptr);
    return build_info_[key].as<StringImm>()->value;
  }

  const Map<std::string, NodeRef> &build_info_;           // passed in from outside
  std::vector<std::string> input_names_;                  // json inputs name, get from build_info_
  std::vector<std::string> output_names_;                 // json outputs name, get from build_info_
  Map<std::string, NodeRef> op_attr_;                     // attr of current op in json["op_desc"]
  picojson::array op_desc_;                               // json["op_desc"]
  std::unordered_map<std::string, picojson::object> io_;  // inputs, outputs map
  picojson::array input_desc_;                            // json["input_desc"]
  picojson::array output_desc_;                           // json["output_desc"]
  picojson::array global_attrs_;                          // json["global_attrs"]
  std::unordered_map<std::string, std::string> inplace_;  // inplace_[a] = b, means a inplace to b
};

std::string GetRealPath(const std::string &path) {
  std::string res;
  if (path.c_str() == nullptr) {
    return res;
  }
  char real_path[PATH_MAX] = {0};
  if (strlen(path.c_str()) >= PATH_MAX || realpath(path.c_str(), real_path) == nullptr) {
    LOG(FATAL) << "Can not get real path from: " << path;
  }
  res = std::string(real_path);
  return res;
}

std::string GetCompositeGraphDumpPath(const std::string &path, const std::string &kernel_name) {
  auto dump_path = GetRealPath(path);
  if (dump_path.empty()) {
    return dump_path;
  }

  // If dump_path is a directory, we will concatenate the file name.
  struct stat s_info;
  if (stat(dump_path.c_str(), &s_info) == 0) {
    if (s_info.st_mode & S_IFDIR) {
      if (dump_path.back() != '/') {
        dump_path += '/';
      }
      dump_path = dump_path + kernel_name + ".info";
    }
  }
  return dump_path;
}
}  // namespace

std::string DumpToJson(const Stmt &stmt, const Map<std::string, NodeRef> &info) {
  CHECK(stmt.defined());
  DumpToJsonVisitor visitor(info);
  return visitor.Dump(stmt);
}

void DumpCompositeGraph(const Stmt &stmt, const BuildInfo &info) {
  if (!stmt.defined() || !info.attrs.defined() || info.attrs.find(kDumpCompositeGraph) == info.attrs.end()) {
    return;
  }
  auto path = info.attrs[kDumpCompositeGraph].as<StringImm>();
  CHECK(path);
  auto dump_path = GetCompositeGraphDumpPath(path->value, info.kernel_name);
  if (dump_path.empty()) {
    return;
  }

  // 1. Stmt --> Json str
  auto json_str = DumpToJson(stmt, SetBuildInfo(info));

  // 2. Save Json str to file
  std::ofstream of(dump_path);
  CHECK(of.is_open()) << "Failed to open \"" << dump_path
                      << "\" to dump composite graph. Please check the directory specified in \"" << kDumpCompositeGraph
                      << "\" exists and has write permissions.";
  of << json_str << std::endl;
  of.close();
}

TVM_REGISTER_GLOBAL("dump_to_json").set_body_typed(DumpToJson);
}  // namespace akg
