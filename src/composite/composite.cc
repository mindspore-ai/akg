/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

#include "build_module.h"
#include "common/array_api.h"
#include "composite/util.h"
#include "codegen/util.h"
#include "dmlc/logging.h"
#include "dmlc/common.h"
#include "picojson.h"
#include "topi/broadcast.h"
#include "topi/elemwise.h"

namespace akg {
static void create_op_inputs(const picojson::array &arr, Array<NodeRef> *current_op_inputs,
                             std::unordered_map<std::string, Tensor> *tensor_index_map) {
  CHECK(current_op_inputs) << "input current_op_inputs is invalid.";
  CHECK(tensor_index_map) << "input tensor_index_map is invalid.";
  for (auto i = arr.begin(); i != arr.end(); ++i) {
    CHECK(i->is<picojson::array>());
    const picojson::array &arr_t = i->get<picojson::array>();
    for (auto j = arr_t.begin(); j != arr_t.end(); ++j) {
      CHECK(j->is<picojson::object>());
      const picojson::object &obj = j->get<picojson::object>();
      std::string tensor_name;
      Array<Expr> shape;
      Type type;
      picojson::value tensor_value;
      bool has_tensor_value = false;
      for (auto k = obj.begin(); k != obj.end(); ++k) {
        if (k->first == "tensor_name") {
          CHECK(k->second.is<std::string>());
          tensor_name = k->second.get<std::string>();
          continue;
        }
        if (k->first == "shape") {
          CHECK(k->second.is<picojson::array>());
          const picojson::array &arr_s = k->second.get<picojson::array>();
          for (auto l = arr_s.begin(); l != arr_s.end(); ++l) {
            CHECK(l->is<int64_t>());
            shape.push_back(Expr(static_cast<int>(l->get<int64_t>())));
          }
          continue;
        }
        if (k->first == "data_type") {
          CHECK(k->second.is<std::string>());
          std::string dtype_str = k->second.get<std::string>();
          if (type_mapping.find(dtype_str) == type_mapping.end()) {
            LOG(FATAL) << "Not support dtype str " << dtype_str;
          }
          type = type_mapping[dtype_str];
          continue;
        }
        if (k->first == "value" && !k->second.is<picojson::null>()) {
          tensor_value = k->second;
          has_tensor_value = true;
        }
      }

      if (!has_tensor_value) {
        if (tensor_index_map->count(tensor_name) == 0) {
          Tensor t = placeholder(shape, type, tensor_name);
          (*tensor_index_map)[tensor_name] = t;
        }
        current_op_inputs->push_back((*tensor_index_map)[tensor_name]);
      } else {
        CHECK_EQ(shape.size(), 1) << "We should not make a expr for a not const tensor.";
        CHECK(Equal(shape[0], Expr(1))) << "We should not make a expr for a not const tensor.";
        CHECK(!tensor_value.is<picojson::null>()) << "We should has default value of tensor(expr): " << tensor_name;
        if (tensor_value.is<double>()) {
          current_op_inputs->push_back(make_const(type, tensor_value.get<double>()));
        } else if (tensor_value.is<int64_t>()) {
          current_op_inputs->push_back(make_const(type, tensor_value.get<int64_t>()));
        } else {
          CHECK(0) << "Unknown value type of tensor: " << tensor_name;
        }
      }
    }
  }
}

static void create_op_inputs(const picojson::array &arr, Array<NodeRef> *current_op_inputs,
                             std::unordered_map<std::string, Tensor> *tensor_index_map,
                             std::map<std::string, Array<NodeRef>> *output_with_input) {
  CHECK(current_op_inputs) << "current_op_inputs is invalid.";
  CHECK(tensor_index_map) << "tensor_index_map is invalid.";
  CHECK(output_with_input) << "output_with_input is invalid.";
  for (auto i = arr.begin(); i != arr.end(); ++i) {
    CHECK(i->is<picojson::array>());
    const picojson::array &arr_t = i->get<picojson::array>();
    for (auto j = arr_t.begin(); j != arr_t.end(); ++j) {
      CHECK(j->is<picojson::object>());
      const picojson::object &obj = j->get<picojson::object>();
      std::string tensor_name;
      Array<Expr> shape;
      Type type;
      picojson::value tensor_value;
      bool has_tensor_value = false;
      for (auto k = obj.begin(); k != obj.end(); ++k) {
        if (k->first == "tensor_name") {
          CHECK(k->second.is<std::string>());
          tensor_name = k->second.get<std::string>();
          continue;
        }
        if (k->first == "shape") {
          CHECK(k->second.is<picojson::array>());
          const picojson::array &arr_s = k->second.get<picojson::array>();
          for (auto l = arr_s.begin(); l != arr_s.end(); ++l) {
            CHECK(l->is<int64_t>());
            shape.push_back(Expr(static_cast<int>(l->get<int64_t>())));
          }
          continue;
        }
        if (k->first == "data_type") {
          CHECK(k->second.is<std::string>());
          std::string dtype_str = k->second.get<std::string>();
          if (type_mapping.find(dtype_str) == type_mapping.end()) {
            LOG(FATAL) << "Not support dtype str " << dtype_str;
          }
          type = type_mapping[dtype_str];
          continue;
        }
        if (k->first == "value" && !k->second.is<picojson::null>()) {
          tensor_value = k->second;
          has_tensor_value = true;
        }
      }

      if (output_with_input->count(tensor_name) != 0) {
        for (auto item : (*output_with_input)[tensor_name]) {
          current_op_inputs->push_back(item);
        }
        continue;
      }

      if (!has_tensor_value) {
        if (tensor_index_map->count(tensor_name) == 0) {
          Tensor t = placeholder(shape, type, tensor_name);
          (*tensor_index_map)[tensor_name] = t;
        }
        current_op_inputs->push_back((*tensor_index_map)[tensor_name]);
      } else {
        CHECK_EQ(shape.size(), 1) << "We should not make a expr for a not const tensor.";
        CHECK(Equal(shape[0], Expr(1))) << "We should not make a expr for a not const tensor.";
        CHECK(!tensor_value.is<picojson::null>()) << "We should has default value of tensor(expr): " << tensor_name;
        if (tensor_value.is<double>()) {
          current_op_inputs->push_back(make_const(type, tensor_value.get<double>()));
        } else if (tensor_value.is<int64_t>()) {
          current_op_inputs->push_back(make_const(type, tensor_value.get<int64_t>()));
        } else {
          CHECK(0) << "Unknown value type of tensor: " << tensor_name;
        }
      }
    }
  }
}

// will parse more info for check output tensor
static void parse_output_label(const picojson::array &arr, std::vector<std::string> *output_tensor_name) {
  CHECK(output_tensor_name) << "input output_tensor_name is invalid.";
  for (auto i = arr.begin(); i != arr.end(); ++i) {
    CHECK(i->is<picojson::object>());
    const picojson::object &obj = i->get<picojson::object>();
    for (auto j = obj.begin(); j != obj.end(); ++j) {
      if (j->first != "tensor_name") {
        continue;
      }
      CHECK(j->second.is<std::string>());
      output_tensor_name->push_back(j->second.get<std::string>());
    }
  }
}

static void parse_attrs(const picojson::array &arr, Array<NodeRef> *attrs_arr) {
  CHECK(attrs_arr) << "input attrs_arr is invalid.";
  for (auto i = arr.begin(); i != arr.end(); ++i) {
    CHECK(i->is<picojson::object>());
    const picojson::object &obj = i->get<picojson::object>();
    for (auto j = obj.begin(); j != obj.end(); ++j) {
      if (j->first != "value") {
        continue;
      }

      if (j->second.is<picojson::array>()) {
        Array<NodeRef> arr_v;
        const picojson::array &arr_s = j->second.get<picojson::array>();
        for (auto l = arr_s.begin(); l != arr_s.end(); ++l) {
          if (l->is<int64_t>()) {
            arr_v.push_back(Integer(static_cast<int>(l->get<int64_t>())));
          } else if (l->is<std::string>()) {
            arr_v.push_back(StringImm::make(l->get<std::string>()));
          } else {
            LOG(FATAL) << "Not parsed type in array attr.";
          }
        }
        attrs_arr->push_back(arr_v);
      } else if (j->second.is<bool>()) {
        attrs_arr->push_back(make_const(Int(1), j->second.get<bool>()));
      } else if (j->second.is<int64_t>()) {
        attrs_arr->push_back(Integer(static_cast<int>(j->second.get<int64_t>())));
      } else if (j->second.is<std::string>()) {
        attrs_arr->push_back(StringImm::make(j->second.get<std::string>()));
      } else {
        LOG(FATAL) << "Not parsed type in attrs.";
      }
    }
  }
}

void extract_op_info(const picojson::array &arr, std::unordered_map<std::string, Tensor> *tensor_index_map,
                     Map<Tensor, Buffer> *in_binds, std::unordered_set<std::string> *fake_output) {
  CHECK(tensor_index_map) << "input tensor_index_map is invalid.";
  CHECK(in_binds) << "input in_binds is invalid.";
  CHECK(fake_output) << "input fake_output is invalid.";
  std::string fusionOpName;
  Array<Tensor> fusion_tensor_arr;
  Array<NodeRef> current_op_inputs;
  Array<NodeRef> final_op_inputs;
  Array<NodeRef> attrs_arr;
  std::vector<std::string> output_tensor_labels;
  std::map<std::string, Array<NodeRef>> output_tensor_labels_with_input;

  for (auto i = arr.begin(); i != arr.end(); ++i) {
    CHECK(i->is<picojson::object>());
    const picojson::object &v_t = i->get<picojson::object>();
    std::string op_name;

    for (auto j = v_t.begin(); j != v_t.end(); ++j) {
      if (j->first == "fusion") {
        fusionOpName = j->second.get<std::string>();
        break;
      }
    }

    for (auto j = v_t.begin(); j != v_t.end(); ++j) {
      if (j->first == "name") {
        CHECK(j->second.is<std::string>());
        op_name = j->second.get<std::string>();
        break;
      }
    }

    for (auto j = v_t.begin(); j != v_t.end(); ++j) {
      if (j->first == "input_desc") {
        CHECK(j->second.is<picojson::array>());
        const picojson::array &local_arr = j->second.get<picojson::array>();
        if (!fusionOpName.empty() && fusionOpName.find("_end") == std::string::npos) {
          if (op_name == "ZerosLike") {
            // ZerosLike directly transform to zero
            Type type;
            CHECK_EQ(local_arr.size(), 1);
            const picojson::array &arr_t = local_arr[0].get<picojson::array>();
            for (auto jx = arr_t.begin(); jx != arr_t.end(); ++jx) {
              CHECK(jx->is<picojson::object>());
              const picojson::object &obj = jx->get<picojson::object>();
              for (auto k = obj.begin(); k != obj.end(); ++k) {
                if (k->first == "data_type") {
                  CHECK(k->second.is<std::string>());
                  std::string dtype_str = k->second.get<std::string>();
                  if (type_mapping.find(dtype_str) == type_mapping.end()) {
                    LOG(FATAL) << "Not support dtype str " << dtype_str;
                  }
                  type = type_mapping[dtype_str];
                  break;
                }
              }
            }
            current_op_inputs.push_back(make_zero(type));
          } else {
            create_op_inputs(local_arr, &current_op_inputs, tensor_index_map);
          }
        } else {
          create_op_inputs(local_arr, &final_op_inputs, tensor_index_map, &output_tensor_labels_with_input);
        }
        break;
      }
    }

    for (auto j = v_t.begin(); j != v_t.end(); ++j) {
      if (j->first == "output_desc") {
        CHECK(j->second.is<picojson::array>());
        const picojson::array &local_arr = j->second.get<picojson::array>();
        parse_output_label(local_arr, &output_tensor_labels);
        if (!fusionOpName.empty() && fusionOpName.find("_end") == std::string::npos) {
          for (auto &output : output_tensor_labels) {
            output_tensor_labels_with_input[output] = current_op_inputs;
          }
        }
        break;
      }
    }

    for (auto j = v_t.begin(); j != v_t.end(); ++j) {
      if (j->first == "attr") {
        if (j->second.is<picojson::array>()) {
          const picojson::array &local_arr = j->second.get<picojson::array>();
          parse_attrs(local_arr, &attrs_arr);
        }
        break;
      }
    }

    if (!fusionOpName.empty()) {
      if (fusionOpName.find("_end") == std::string::npos) {
        current_op_inputs = {};
        output_tensor_labels.clear();
        continue;
      }
      auto strList = dmlc::Split(fusionOpName, '_');
      CHECK(!strList.empty());
      op_name = strList[0];
      fusionOpName = "";
    }
    const auto *topi_f = ktvm::runtime::Registry::Get(op_name);
    CHECK(topi_f) << "Akg topi has no op: " << op_name;
    if (op_name == "InplaceAssign") {
      CHECK(output_tensor_labels.size() == 1 && final_op_inputs.size() == 3);
      Map<Tensor, Buffer> binds = (*topi_f)(final_op_inputs, attrs_arr);
      Tensor out = Downcast<Tensor>(final_op_inputs[2]);
      (*tensor_index_map)[output_tensor_labels.front()] = out;
      for (auto &it : binds) {
        in_binds->Set(it.first, it.second);
      }
      if (attrs_arr.size() == 1) {
        auto fake_val = attrs_arr[0].as<IntImm>();
        if (fake_val && fake_val->value > 0) {
          fake_output->insert(output_tensor_labels[0]);
        }
      }
    } else if (output_tensor_labels.size() == 1) {
      Tensor t;
      t = (*topi_f)(final_op_inputs, attrs_arr);
      (*tensor_index_map)[output_tensor_labels.front()] = t;
    } else {
      Array<Tensor> a;
      a = (*topi_f)(final_op_inputs, attrs_arr);
      CHECK_EQ(output_tensor_labels.size(), a.size());
      for (size_t x = 0; x < output_tensor_labels.size(); ++x) {
        (*tensor_index_map)[output_tensor_labels[x]] = a[x];
      }
    }

    final_op_inputs = {};
    attrs_arr = {};
    output_tensor_labels.clear();
    output_tensor_labels_with_input.clear();
  }
}

void extract_op_info(const picojson::value &v, Array<Tensor> *ops, Array<NodeRef> *args, std::string *kernel_name,
                     Map<Tensor, Buffer> *in_binds) {
  CHECK(ops) << "input ops is invalid.";
  CHECK(args) << "input args is invalid.";
  CHECK(kernel_name) << "input kernel_name is invalid.";
  CHECK(in_binds) << "input in_binds is invalid.";
  picojson::array input_desc;
  picojson::array output_desc;
  picojson::array op_desc;
  CHECK(v.is<picojson::object>());

  const picojson::value::object &obj = v.get<picojson::object>();
  for (auto i = obj.begin(); i != obj.end(); ++i) {
    if (i->first == "op") {
      CHECK(i->second.is<std::string>());
      *kernel_name = i->second.get<std::string>();
    } else if (i->first == "input_desc") {
      CHECK(i->second.is<picojson::array>());
      input_desc = i->second.get<picojson::array>();
    } else if (i->first == "output_desc") {
      CHECK(i->second.is<picojson::array>());
      output_desc = i->second.get<picojson::array>();
    } else if (i->first == "op_desc") {
      CHECK(i->second.is<picojson::array>());
      op_desc = i->second.get<picojson::array>();
    }
  }

  std::unordered_map<std::string, Tensor> tensor_index_map;
  std::unordered_set<std::string> fake_output;
  extract_op_info(op_desc, &tensor_index_map, in_binds, &fake_output);

  for (auto i = input_desc.begin(); i != input_desc.end(); ++i) {
    CHECK(i->is<picojson::array>());
    const picojson::array &arr_t = i->get<picojson::array>();
    CHECK(arr_t.begin()->is<picojson::object>());
    const picojson::object &local_obj = arr_t.begin()->get<picojson::object>();
    for (auto j = local_obj.begin(); j != local_obj.end(); ++j) {
      if (j->first != "tensor_name") continue;
      CHECK(j->second.is<std::string>());
      const std::string &tensor_name = j->second.get<std::string>();
      auto iter = tensor_index_map.find(tensor_name);
      if (iter != tensor_index_map.end()) {
        args->push_back(iter->second);
      } else {
        LOG(FATAL) << "Tensor " << tensor_name << " not built.";
      }
    }
  }

  for (auto i = output_desc.begin(); i != output_desc.end(); ++i) {
    CHECK(i->is<picojson::object>());
    const picojson::object &local_obj = i->get<picojson::object>();
    for (auto j = local_obj.begin(); j != local_obj.end(); ++j) {
      if (j->first != "tensor_name") continue;
      CHECK(j->second.is<std::string>());
      const std::string &tensor_name = j->second.get<std::string>();
      auto iter = tensor_index_map.find(tensor_name);
      if (iter != tensor_index_map.end()) {
        ops->push_back(iter->second);
        if (!fake_output.count(tensor_name)) {
          args->push_back(iter->second);
        }
      } else {
        LOG(FATAL) << "Tensor " << tensor_name << " not built.";
      }
    }
  }
}

NodeRef composite_with_json_to_func(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    LOG(ERROR) << "json parse error, error message: " << err;
  }
  const char *akg_dump_pass_ir = getenv("MS_AKG_DUMP_IR");
  Array<Tensor> tensors;
  Array<NodeRef> args;
  Array<NodeRef> shape_vars;
  Map<Tensor, Buffer> in_binds;
  std::string kernel_name;
  extract_op_info(v, &tensors, &args, &kernel_name, &in_binds);
  Array<Operation> ops;
  std::for_each(tensors.begin(), tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  config->dump_pass_ir = akg_dump_pass_ir != nullptr;
  attrs.Set("pragma_reschedule", make_const(Int(32), 1));
  auto build_rst = akg::BuildToFunc(sch, args, shape_vars, kernel_name, in_binds, attrs, true, false, config);
  CHECK(build_rst.defined());
  return build_rst;
}

Module composite_with_json(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  auto build_rst = composite_with_json_to_func(json_str, attrs);
  return BuildToModule(build_rst);
}

NodeRef composite_lower(const std::string &json_str, Map<std::string, NodeRef> attrs) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  if (!err.empty()) {
    LOG(ERROR) << "json parse error, error message: " << err;
  }
  Array<Tensor> tensors;
  Array<NodeRef> args;
  Array<NodeRef> shape_vars;
  Map<Tensor, Buffer> in_binds;
  std::string kernel_name;
  extract_op_info(v, &tensors, &args, &kernel_name, &in_binds);
  Array<Operation> ops;
  std::for_each(tensors.begin(), tensors.end(), [&ops](const Tensor &t) { ops.push_back(t->op); });
  Schedule sch = create_schedule(ops);
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  bool tuning = attrs.find("tuning") != attrs.end();
  return akg::Lower(sch, args, shape_vars, kernel_name, in_binds, attrs, false, true, tuning, false, config);
}

TVM_REGISTER_GLOBAL("composite_with_json_to_func").set_body_typed(composite_with_json_to_func);
TVM_REGISTER_GLOBAL("composite_with_json").set_body_typed(composite_with_json);

TVM_REGISTER_GLOBAL("composite_lower").set_body_typed(composite_lower);
}  // namespace akg
