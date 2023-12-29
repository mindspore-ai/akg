/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "util.h"
#include <fstream>
#include <map>
#include "pass/utils.h"

namespace akg {
std::vector<int64_t> ExtractIntVector(Array<Expr> &vec) {
  std::vector<int64_t> res;
  for (Expr s : vec) {
    int64_t val = -1;
    if (s.as<IntImm>()) {
      val = s.as<IntImm>()->value;
    } else if (s.as<UIntImm>()) {
      val = s.as<UIntImm>()->value;
    } else {
      CHECK(0);
    }
    res.push_back(val);
  }
  return res;
}

std::string type2string(const air::Type &type) {
  std::string type_str;
  for (const auto &it : type_mapping) {
    if (it.second == type) {
      type_str = it.first;
      break;
    }
  }
  return type_str;
}

bool IsBlockIdx(const std::string &name) { return name.find(BLOCKIDX) != std::string::npos; }
bool IsBlockIdxX(const std::string &name) { return name == BLOCK_IDX_X; }
bool IsBlockIdxY(const std::string &name) { return name == BLOCK_IDX_Y; }
bool IsBlockIdxZ(const std::string &name) { return name == BLOCK_IDX_Z; }
bool IsThreadIdx(const std::string &name) { return name.find(THREADIDX) != std::string::npos; }
bool IsThreadIdxX(const std::string &name) { return name == THREAD_IDX_X; }
bool IsThreadIdxY(const std::string &name) { return name == THREAD_IDX_Y; }
bool IsThreadIdxZ(const std::string &name) { return name == THREAD_IDX_Z; }

std::string GetRealTarget(const std::string &target) {
  static std::map<std::string, std::string> target_map = {{"aicore", "cce"}, {"cpu", "llvm"}};

  if (target_map.find(target) != target_map.end()) {
    return target_map[target];
  }
  return target;
}

std::string GetOption(const picojson::value &target_json) {
  std::string option = "";
  if (target_json.contains("arch")) {
    std::string default_sys = "none";
    std::string default_abi = "gnu";
    std::string default_vendor = "unknown";
    auto vendor_value = target_json.contains("vendor") ? target_json.get("vendor").get<std::string>() : default_vendor;
    auto sys_value = target_json.contains("system") ? target_json.get("system").get<std::string>() : default_sys;
    auto abi_value = target_json.contains("abi") ? target_json.get("abi").get<std::string>() : default_abi;
    option += " -target=" + target_json.get("arch").get<std::string>();
    option += "-" + vendor_value;
    option += "-" + sys_value;
    option += "-" + abi_value;
  }
  if (target_json.contains("cpu")) {
    option += " -mcpu=" + target_json.get("cpu").get<std::string>();
  }
  if (target_json.contains("feature")) {
    option += " -mattr=+" + target_json.get("feature").get<std::string>();
  }
  return option;
}

std::string GetProcess(const picojson::value &input_json) {
  const picojson::value::object &input_obj = input_json.get<picojson::object>();
  std::string target;
  std::string option = "";
  auto iter = input_obj.find("process");
  if (iter != input_obj.end()) {
    CHECK(iter->second.is<std::string>());
    target = iter->second.get<std::string>();
  }
  if (input_json.contains("target_info") && target != "aicore") {
    option = GetOption(input_json.get("target_info"));
  }
  return GetRealTarget(target) + option;
}

picojson::value String2Json(const std::string &json_str) {
  picojson::value v;
  std::string err = picojson::parse(v, json_str);
  CHECK(err.empty()) << "json parse error, error message: " << err;
  return v;
}
bool IsReduce(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin", "Argmax", "Argmin"};
  return elems.find(op_name) != elems.end();
}
bool IsTransform(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"Reshape", "ExpandDims", "Squeeze", "Flatten"};
  return elems.find(op_name) != elems.end();
}
bool IsInplaceAssign(const std::string &op_name) { return op_name == "InplaceAssign"; }
bool IsAssign(const std::string &op_name) { return op_name == "Assign"; }
bool IsOtherOp(const std::string &op_name) {
  // if topi support more, add to this list
  std::unordered_set<std::string> elems = {"MatMul",
                                           "BatchMatMul",
                                           "Conv",
                                           "Transpose",
                                           "Tile",
                                           "Assign",
                                           "InplaceAssign",
                                           "EquivFormat",
                                           "TransData",
                                           "AddMinValue",
                                           "BroadcastTo",
                                           "PadAkg",
                                           "UnPadAkg",
                                           "Conv2D",
                                           "CumSum",
                                           "CumProd",
                                           "StridedSlice",
                                           "Custom",
                                           "GatherNd",
                                           "TensorScatterAdd",
                                           "UnsortedSegmentSum",
                                           "Gather",
                                           "OneHot",
                                           "tuple_getitem",
                                           "CSRMV",
                                           "CSRReduceSum",
                                           "CSRMul",
                                           "CSRDiv",
                                           "CSRGather",
                                           "CSR2COO",
                                           "COO2CSR",
                                           "ElemAny",
                                           "CSRMM",
                                           "Pool2D",
                                           "LayoutTransform"};
  return elems.find(op_name) != elems.end();
}
bool IsElemwise(const std::string &op_name) {
  return !IsReduce(op_name) && !IsTransform(op_name) && !IsOtherOp(op_name);
}
bool EqualShape(const Array<Expr> &shape1, const Array<Expr> &shape2) {
  if (shape1.size() != shape2.size()) return false;
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (!Equal(shape1[i], shape2[i])) {
      return false;
    }
  }
  return true;
}

bool ShapeIsOne(const Array<Expr> &shape) { return shape.size() == 1 && Equal(shape[0], 1); }
bool ShapeSizeIsOne(const Array<Expr> &shape) {
  Expr size = 1;
  for (auto &i : shape) {
    size *= i;
  }
  return Equal(Simplify(size), 1);
}

bool ShapeCanBroadcast(const Array<Expr> &shape1, const Array<Expr> &shape2) {
  if (shape1.size() > shape2.size()) {
    return false;
  }
  auto start_comp_pos = shape2.size() - shape1.size();
  for (size_t i = 0; i < shape1.size(); ++i) {
    if (!Equal(shape1[i], shape2[start_comp_pos + i]) && !Equal(shape1[i], 1)) {
      return false;
    }
  }
  return true;
}

std::string GetOpName(const Provide *p) {
  auto call = p->value.as<Call>();
  CHECK(call);
  auto op_name = call->name;
  return op_name;
}
std::string CreateDataFormatKey(const std::string &tensor_name) {
  std::string key = tensor_name + "_format";
  return key;
}

bool GetBoolValueFromMap(const Map<std::string, NodeRef> &node_ref_map, const std::string &key) {
  CHECK(node_ref_map.defined());
  CHECK(node_ref_map.find(key) != node_ref_map.end());
  CHECK(node_ref_map[key].as<ExprNode>());
  auto value = ir::GetInt32Const(Downcast<Expr>(node_ref_map[key]));
  return static_cast<bool>(value);
}

Map<std::string, NodeRef> SetAutoFuseAttr(const std::vector<size_t> &split_index,
                                          const Map<std::string, NodeRef> &attrs) {
  Map<std::string, NodeRef> new_attrs;
  if (attrs.defined()) new_attrs = attrs;
  std::stringstream ss;
  for (const auto &split : split_index) {
    ss << split << " ";
  }
  new_attrs.Set("auto_fuse_split", Expr(ss.str()));
  return new_attrs;
}

Map<std::string, NodeRef> BindBlockAndThread(GridBlockDims &dims, bool poly, const Map<std::string, NodeRef> &attrs) {
  Map<std::string, NodeRef> new_attrs;
  if (attrs.defined()) new_attrs = attrs;
  if (poly) {
    std::stringstream ss;
    ss << dims.griddim_x << " " << dims.griddim_y << " " << dims.griddim_z;
    new_attrs.Set("bind_block", Expr(ss.str()));
    ss.str("");
    ss << dims.blockdim_x << " " << dims.blockdim_y << " " << dims.blockdim_z;
    new_attrs.Set("bind_thread", Expr(ss.str()));
    LOG(INFO) << new_attrs;
    return new_attrs;
  }
  return attrs;
}

Stmt InsertSync(Stmt &s) {
  return Block::make(
    s, Evaluate::make(Call::make(Int(32), "tvm_storage_sync", {StringImm::make("shared")}, Call::Intrinsic)));
}

namespace BroadcastReshapeUtil {
struct IndexGroup {
  std::vector<size_t> indexs;
  bool is_broadcast;
};

struct ReshapeRelation {
  IndexGroup index_group;
  Array<Expr> shape;
};

std::vector<bool> GetIsBroadVec(const Array<Expr> &input_shape, const Array<Expr> &output_shape) {
  CHECK_GE(output_shape.size(), input_shape.size());
  std::vector<bool> is_broadcast_vec(output_shape.size(), false);
  auto start_comp_index = output_shape.size() - input_shape.size();
  for (size_t i = 0; i < start_comp_index; ++i) {
    is_broadcast_vec[i] = true;
  }
  for (size_t i = 0; i < input_shape.size(); ++i) {
    if (!Equal(input_shape[i], output_shape[start_comp_index + i])) {
      is_broadcast_vec[i + start_comp_index] = true;
    }
  }
  return is_broadcast_vec;
}

std::vector<IndexGroup> GetIndexGroup(const Array<Expr> &shape, const std::vector<bool> &is_broadcast_vec) {
  auto start_index = is_broadcast_vec.size() - shape.size();
  std::vector<IndexGroup> index_groups;
  std::vector<size_t> indexs;
  indexs.push_back(0);
  for (size_t i = start_index + 1; i < is_broadcast_vec.size(); ++i) {
    if (is_broadcast_vec[i] != is_broadcast_vec[i - 1]) {
      index_groups.push_back(IndexGroup({indexs, is_broadcast_vec[i - 1]}));
      indexs.clear();
    }
    indexs.push_back(i);
  }
  index_groups.push_back(IndexGroup({indexs, is_broadcast_vec[is_broadcast_vec.size() - 1]}));
  return index_groups;
}

bool HasFoundReshape(const Array<Expr> &shape_change, const Expr &ori_size, size_t *j) {
  bool has_found_reshape = false;
  auto change_size = Expr(1);
  for (; *j < shape_change.size(); ++(*j)) {
    change_size = Simplify(change_size * shape_change[*j]);
    // For the ‘1’ in the shape_change, it may correspond to the current group or the next group.
    // It is difficult to determine the specific group of ‘1’, and some strategies may need to be added later.
    // The ‘1’ is currently correspond to the the next group.
    if (is_zero(Simplify(change_size - ori_size))) {
      has_found_reshape = true;
      ++(*j);
      break;
    }
  }
  return has_found_reshape;
}

bool CheckInputBroadcast(const std::string &type, const std::vector<IndexGroup> &index_groups, size_t i,
                         const Array<Expr> &shape_ori) {
  auto index_group = index_groups[i];
  auto indexs = index_groups[i].indexs;
  auto index_group_start = indexs[0];
  auto index_group_end = index_group_start + indexs.size();
  return type == "input" && index_groups[i].is_broadcast && i < index_groups.size() - 1 &&
         !is_const_int(shape_ori[index_group_end], 1);
}

std::vector<ReshapeRelation> GetReshapeRelation(const Array<Expr> &shape_ori, const std::vector<bool> &is_broadcast_vec,
                                                const Array<Expr> &shape_change, const std::string &type) {
  // If the ReshapeRelations cannot be obtained, for example,
  // when the boundary between the broadcast axis and the elewise axis is broken by reshape,
  // an empty vector is returned to indicate failure.
  auto index_groups = GetIndexGroup(shape_ori, is_broadcast_vec);
  std::vector<ReshapeRelation> reshape_relations;
  size_t j = 0;
  for (size_t i = 0; i < index_groups.size(); ++i) {
    auto index_group = index_groups[i];
    auto indexs = index_groups[i].indexs;
    auto index_group_start = indexs[0];
    auto index_group_end = index_group_start + indexs.size();
    auto ori_size = Expr(1);
    for (size_t ii = index_group_start; ii < index_group_end; ++ii) {
      ori_size = ori_size * shape_ori[ii];
    }
    ori_size = Simplify(ori_size);
    auto j_start = j;
    // The ending 1 is eliminated by reshape, for example, (x) = reshape((x, 1, 1))：
    if (j_start == shape_change.size() && i == index_groups.size() - 1 && is_const_int(ori_size, 1) &&
        !index_group.is_broadcast) {
      return reshape_relations;
    }

    if (!HasFoundReshape(shape_change, ori_size, &j)) {
      return {};
    }
    if (i == index_groups.size() - 1 && j < shape_change.size()) {
      // The shape after reshape has extra 1 at the end，for example, (x， 1， 1) = reshape((x))：
      while (j < shape_change.size()) {
        if (!is_const_int(Simplify(shape_change[j]), 1)) {
          return {};
        }
        ++j;
      }
    }
    Array<Expr> shape_reshape(shape_change.begin() + j_start, shape_change.begin() + j);

    CHECK(type == "input" || type == "output");
    // Let some the ‘1’ correspond to the current group.
    if (CheckInputBroadcast(type, index_groups, i, shape_ori)) {
      while (shape_reshape.size() < indexs.size() && j < shape_change.size() &&
             is_const_int(Simplify(shape_change[j]), 1)) {
        shape_reshape.push_back(shape_change[j]);
        ++j;
      }
    }

    reshape_relations.push_back(ReshapeRelation({index_group, shape_reshape}));
  }
  return reshape_relations;
}

std::vector<ReshapeRelation> GetInputReshapeRelation(const Array<Expr> &input_shape_ori,
                                                     const Array<Expr> &input_shape_change,
                                                     const Array<Expr> &output_shape_ori) {
  auto is_broadcast_vec = GetIsBroadVec(input_shape_ori, output_shape_ori);
  auto relations = GetReshapeRelation(input_shape_ori, is_broadcast_vec, input_shape_change, "input");
  return relations;
}

std::vector<ReshapeRelation> GetOutputReshapeRelation(const Array<Expr> &output_shape_ori,
                                                      const Array<Expr> &output_shape_change,
                                                      const Array<Expr> &input_shape_ori) {
  auto is_broadcast_vec = GetIsBroadVec(input_shape_ori, output_shape_ori);
  auto relations = GetReshapeRelation(output_shape_ori, is_broadcast_vec, output_shape_change, "output");
  return relations;
}

Array<Expr> BroadShapeChange(const Array<Expr> &shape_ori, const Array<Expr> &shape_reshape) {
  // There may be multiple legal shape_change,
  // and some strategies may need to be added later to determine shape_change.
  if (shape_reshape.size() == shape_ori.size()) {
    return shape_ori;
  } else {
    Array<Expr> shape_change(shape_reshape.size() - 1, Expr(1));
    auto shape_size = Expr(1);
    for (const auto &dim : shape_ori) {
      shape_size = shape_size * dim;
    }
    shape_change.push_back(Simplify(shape_size));
    return shape_change;
  }
}

Array<Expr> GetOutputShapeChange(const Array<Expr> &output_shape_ori, const Array<Expr> &input_shape_ori,
                                 const Array<Expr> &input_shape_change) {
  CHECK(ShapeCanBroadcast(input_shape_ori, output_shape_ori));
  auto reshape_relations = GetInputReshapeRelation(input_shape_ori, input_shape_change, output_shape_ori);
  if (reshape_relations.empty()) {
    return Array<Expr>();
  }
  Array<Expr> output_shape_change;
  auto excess_len = output_shape_ori.size() - input_shape_ori.size();
  for (size_t i = 0; i < excess_len; ++i) {
    output_shape_change.push_back(output_shape_ori[i]);
  }
  for (auto reshape_relaiton : reshape_relations) {
    auto index_group = reshape_relaiton.index_group;
    auto indexs = index_group.indexs;
    bool is_broadcast = index_group.is_broadcast;
    auto shape_reshape = reshape_relaiton.shape;
    if (!is_broadcast) {
      for (auto dim : shape_reshape) {
        output_shape_change.push_back(dim);
      }
    } else {
      auto index_start = indexs[0] + excess_len;
      auto index_end = index_start + indexs.size();
      auto shape_reshape_broad = BroadShapeChange(
        Array<Expr>(output_shape_ori.begin() + index_start, output_shape_ori.begin() + index_end), shape_reshape);
      for (const auto &dim : shape_reshape_broad) {
        output_shape_change.push_back(dim);
      }
    }
  }
  CHECK(ShapeCanBroadcast(input_shape_change, output_shape_change));
  return output_shape_change;
}

Array<Expr> GetInputShapeChange(const Array<Expr> &input_shape_ori, const Array<Expr> &output_shape_ori,
                                const Array<Expr> &output_shape_change) {
  // elewise:
  if (EqualShape(input_shape_ori, output_shape_ori)) {
    return output_shape_change;
  }
  // broacast:
  CHECK(ShapeCanBroadcast(input_shape_ori, output_shape_ori));
  auto reshape_relations = GetOutputReshapeRelation(output_shape_ori, output_shape_change, input_shape_ori);
  if (reshape_relations.empty()) {
    return Array<Expr>();
  }
  Array<Expr> input_shape_change;
  auto excess_len = output_shape_ori.size() - input_shape_ori.size();
  size_t i = excess_len > 0 ? 1 : 0;

  // Just to keep the shape of input unchanged,
  // delete this code without affecting the correctness of the function.
  auto start_group_relation = reshape_relations[0];
  auto start_index_group = start_group_relation.index_group;
  auto start_indexs = start_index_group.indexs;
  if (excess_len > 0 && start_indexs.size() > excess_len) {
    auto start_reshape_shape = start_group_relation.shape;
    CHECK(start_index_group.is_broadcast);
    auto input_start_broad_size = start_indexs.size() - excess_len;
    if (input_start_broad_size <= start_reshape_shape.size()) {
      input_shape_change = Array<Expr>(input_start_broad_size, Expr(1));
    }
  }

  for (; i < reshape_relations.size(); ++i) {
    auto reshape_relaiton = reshape_relations[i];
    auto index_group = reshape_relaiton.index_group;
    auto indexs = index_group.indexs;
    bool is_broadcast = index_group.is_broadcast;
    auto shape_reshape = reshape_relaiton.shape;
    if (!is_broadcast) {
      for (auto dim : shape_reshape) {
        input_shape_change.push_back(dim);
      }
    } else {
      for (size_t ii = 0; ii < shape_reshape.size(); ++ii) {
        input_shape_change.push_back(Expr(1));
      }
    }
  }
  CHECK(ShapeCanBroadcast(input_shape_change, output_shape_change));
  return input_shape_change;
}

FuncShape GetInputsChangeShape(const FunctionRef &output, Graph &g, const Array<Expr> &output_shape) {
  auto inputs = g.pre_graph[output];
  FuncShape input_map_shape_change;
  auto output_shape_ori = g.func_shape[output];
  auto output_changed = !EqualShape(output_shape_ori, output_shape);
  for (const auto &input : inputs) {
    auto input_shape_ori = g.func_shape[input];
    Array<Expr> input_shape_change;
    if (output_changed) {
      input_shape_change = GetInputShapeChange(input_shape_ori, output_shape_ori, output_shape);
      if (input_shape_change.empty()) {
        return FuncShape();
      }
    } else {
      input_shape_change = input_shape_ori;
    }
    input_map_shape_change[input] = input_shape_change;
  }
  return input_map_shape_change;
}
}  // namespace BroadcastReshapeUtil

akg::BuildConfig GetConfig() {
  akg::BuildConfig config = akg::BuildConfig::Current();
  CHECK(config.defined());
  config->dump_pass_ir = getenv(GetDumpIRFlag().c_str()) != nullptr;
  return config;
}

std::vector<std::string> GetNames(const Array<NodeRef> &io) {
  std::vector<std::string> names;
  for (const auto &arg : io) {
    CHECK(arg.as<StringImm>());
    auto arg_name = arg.as<StringImm>()->value;
    names.emplace_back(arg_name);
  }
  return names;
}

Stmt ElimDuplicateInputs::Run(Stmt &stmt) {
  is_mutate_ = false;
  static_cast<void>(Mutate(stmt));
  is_mutate_ = true;
  return Mutate(stmt);
}

Expr ElimDuplicateInputs::Mutate_(const Load *op, const Expr &e) {
  Var var = op->buffer_var;
  auto name = var->name_hint;
  if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      if (is_mutate_) return Load::make(op->type, it->second, this->Mutate(op->index), op->predicate);
    } else {
      vars_[name] = var;
    }
  }
  return IRMutator::Mutate_(op, e);
}

Stmt ElimDuplicateInputs::Mutate_(const Store *op, const Stmt &s) {
  Var var = op->buffer_var;
  auto name = var->name_hint;
  if (std::find(names_.begin(), names_.end(), name) != names_.end()) {
    auto it = vars_.find(name);
    if (it != vars_.end()) {
      if (is_mutate_) return Store::make(it->second, this->Mutate(op->value), this->Mutate(op->index), op->predicate);
    } else {
      vars_[name] = var;
    }
  }
  return IRMutator::Mutate_(op, s);
}

Map<std::string, NodeRef> SetBuildInfo(const BuildInfo &info) {
  Array<Expr> input_names_arr;
  for (const auto &name : info.input_names) {
    input_names_arr.push_back(Expr(name));
  }
  Array<Expr> output_names_arr;
  for (const auto &name : info.output_names) {
    output_names_arr.push_back(Expr(name));
  }
  // Record the inplace tensor names, which will saved in the .info file later.
  Map<std::string, Expr> inplace;
  for (const auto &it : info.opt.inplaces) {
    auto call = it.second.as<Call>();
    if (it.first.defined() && call != nullptr) {
      inplace.Set(it.first->func_name(), Expr(call->name));
    }
  }
  Map<std::string, NodeRef> build_info;
  build_info.Set("op", Expr(info.kernel_name));
  build_info.Set("process", Expr(info.opt.target));
  build_info.Set("input_names", input_names_arr);
  build_info.Set("output_names", output_names_arr);
  build_info.Set("inplace", inplace);
  return build_info;
}
}  // namespace akg
