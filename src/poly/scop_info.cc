/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <regex>
#include "pass/utils.h"
#include "scop.h"
#include "scop_builder.h"
#include "poly/dma_inject.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {
constexpr int kInvalidIntAttr = -1;
Expr kInvalidExprAttr;

bool Scop::IsConvBackpropInput() const {
  int n = ExtractIntFromAttrs(ATTR_CONV_BACKPROP_INPUT);
  return (IsConv() && (n != kInvalidIntAttr));
}

bool Scop::IsConvBackpropFilter() const {
  int n = ExtractIntFromAttrs(ATTR_CONV_BACKPROP_FILTER);
  return (IsConv() && (n != kInvalidIntAttr));
}

Expr Scop::ExtractExprFromAttrs(const std::string &name) const {
  for (auto i : data_.stmt_op_Info) {
    if (!i.second.isCube) {
      continue;
    }

    const Node *stmt_node = data_.statements.at(i.first);
    CHECK(stmt_node != nullptr);
    if (stmt_node->IsInstance<Provide>()) {
      auto provide = static_cast<const Provide *>(stmt_node);
      if (const auto cop = provide->func.as<ComputeOpNode>()) {
        if (cop->attrs.count(name) != 0) {
          return air::Downcast<Expr>(cop->attrs.at(name));
        }
      }
    }
  }
  return kInvalidExprAttr;
}

int Scop::ExtractIntFromAttrs(const std::string &name) const {
  Expr expr_attr = ExtractExprFromAttrs(name);
  if (expr_attr.defined()) {
    if (const auto int_op = expr_attr.as<IntImm>())
      return int_op->value;
    else
      LOG(FATAL) << "attr " << name << " is not an integer";
  }
  return kInvalidIntAttr;
}

std::unordered_set<std::string> Scop::ExtractWithStmtId() const {
  std::unordered_set<std::string> res;
  for (auto i : data_.stmt_op_Info) {
    if (!i.second.isWith) {
      continue;
    }
    res.insert(i.first.get_name());
  }
  return res;
}

std::string Scop::ExtractStringFromAttrs(const std::string &name) const {
  for (auto i : data_.stmt_op_Info) {
    if (!i.second.isCube) {
      continue;
    }

    const Node *stmt_node = data_.statements.at(i.first);
    if (stmt_node->IsInstance<Provide>()) {
      auto provide = static_cast<const Provide *>(stmt_node);
      if (const auto cop = provide->func.as<ComputeOpNode>()) {
        if (cop->attrs.count(name) != 0) {
          if (const auto str_op = cop->attrs.at(name).as<StringImm>()) {
            return str_op->value;
          } else {
            LOG(FATAL) << "attr " << name << " is not a string";
          }
        }
      }
    }
  }
  return "";
}

std::string Scop::ExtractStringFromAttrsAndInfo(const std::string &name) const {
  for (auto i : data_.stmt_op_Info) {
    if (!i.second.isCube) {
      continue;
    }

    const Node *stmt_node = data_.statements.at(i.first);
    if (stmt_node->IsInstance<Provide>()) {
      auto provide = static_cast<const Provide *>(stmt_node);
      if (const auto cop = provide->func.as<ComputeOpNode>()) {
        if (cop->attrs.count(name) != 0) {
          if (const auto str_op = cop->attrs.at(name).as<StringImm>()) {
            return str_op->value;
          } else {
            LOG(FATAL) << "attr " << name << " is not a string";
          }
        }
      }
    }
  }

  if (attr_info_.count(name) >= 1) {
    if (const auto str_op = attr_info_.at(name).as<StringImm>()) {
      return str_op->value;
    } else {
      LOG(FATAL) << "attr " << name << " is not a string";
    }
  }

  return "";
}

class DimInfoMatcher : public IRVisitor {
 public:
  DimInfoMatcher() : dim_("") {}
  ~DimInfoMatcher() override = default;

  std::string dim() { return dim_; }

  void Visit_(const AttrStmt *op) final {
    if (const auto Cop = op->node.as<ComputeOpNode>()) {
      for (auto iter : Cop->attrs) {
        if (dim_.empty() && iter.first == "dim") {
          if (auto dim = iter.second.as<StringImm>()) {
            dim_ = dim->value;
            break;
          }
        }
      }
    }
  }

 private:
  std::string dim_;
};

std::string TensorMarkTag(MemType mem_type, MemFlow mem_flow) {
  /******************************
   *  This interface is used to convert tensor MemType to isl schedule tree mark_tag,
   *  used to record the extension position for each tensor in isl schedule tree.
   *  Now REALIZE_L1/REALIZE_L0/REALIZE_UB mark_tag is equal to its MemType.
   *  For mem_type is DDR, mark_tag is empty string "".
   * */
  switch (mem_type) {
    case MemType::L1_:
      if (mem_flow.size() == 3 && mem_flow[0] == MemType::DDR && mem_flow[1] == MemType::L1_ &&
          mem_flow[2] == MemType::UBL1_)
        return REALIZE_L1UBL1;
      return REALIZE_L1;
    case MemType::UB_:
      // ordinary conv condition no fusion
      if (mem_flow.size() == 3 && mem_flow[0] == MemType::DDR && mem_flow[1] == mem_type &&
          mem_flow[2] == MemType::L0C_)
        return REALIZE_L0;
      return REALIZE_UB;
    case MemType::L0A_:
      return REALIZE_L0;
    case MemType::L0B_:
      return REALIZE_L0;
    case MemType::L0C_:
      return REALIZE_L0;
    case MemType::UBL0_:
      return REALIZE_UBL0;
    case MemType::UBL1_:
      if (mem_flow.size() == 2 && mem_flow[0] == MemType::DDR && mem_flow[1] == MemType::UBL1_) return REALIZE_L1;
      return REALIZE_UBL1;
    case MemType::DDR:
      return "";
    default:
      LOG(FATAL) << "undefined mem_type " << mem_type;
      return "";
  }
}

bool Scop::IsElewiseVMStmt(const isl::id &id) const {
  auto stmt = data_.statements.at(id);
  if (stmt != nullptr && stmt->IsInstance<Provide>()) {
    auto provide = static_cast<const Provide *>(stmt);
    if (auto call = provide->value.as<Call>()) {
      if (call->call_type != Call::CallType::Halide && (call->name == "vmadd" || call->name == "vmla")) return true;
    }
  }
  return false;
}

bool Scop::MayWriteAfterRead(const std::string &name) const {
  std::map<int, isl::id> def;
  std::map<int, isl::id> use;
  for (auto a : data_.writes.get_map_list()) {
    isl::id id = a.domain().unwrap().domain().get_tuple_id();
    std::string idstr = id.get_name();
    if (a.get_tuple_id(isl_dim_out).get_name() != name) continue;
    CHECK_GE(idstr.size(), 2);
    idstr = idstr.substr(2, idstr.size());
    int ref = static_cast<int>(WrappedStrtol(idstr));
    def[ref] = id;
  }
  for (auto a : data_.reads.get_map_list()) {
    isl::id id = a.domain().unwrap().domain().get_tuple_id();
    std::string idstr = id.get_name();
    if (a.get_tuple_id(isl_dim_out).get_name() != name) continue;
    CHECK_GE(idstr.size(), 2);
    idstr = idstr.substr(2, idstr.size());
    int ref = static_cast<int>(WrappedStrtol(idstr));
    use[ref] = id;
  }

  if (def.empty() || use.empty()) return false;
  if (def.begin()->first >= use.begin()->first) return true;
  // if A = f(A) exists, we think has WAR
  for (auto i : def) {
    if (use.count(i.first)) {
      // vmadd/vmla insn is in the form A = f(A), but there is no WAR dependence
      if (!IsElewiseVMStmt(i.second)) return true;
    }
  }
  return false;
}

bool Scop::IsA(const std::string &name) const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      if (info.second.A_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool Scop::IsB(const std::string &name) const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      if (info.second.B_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool Scop::IsC(const std::string &name) const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      if (info.second.C_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool Scop::IsCUB(const std::string &name) const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      if (info.second.C_ + "_local_UB" == name) {
        return true;
      }
    }
  }
  return false;
}

std::string Scop::GetAName() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      return info.second.A_;
    }
  }
  return "";
}

std::string Scop::GetBName() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      return info.second.B_;
    }
  }
  return "";
}

std::string Scop::GetCName() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) {
      return info.second.C_;
    }
  }
  return "";
}

bool Scop::IsIm2col() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isIm2col) return true;
  }
  return false;
}

bool Scop::IsLoad3dL1Ub() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isLoad3d) return true;
  }
  return false;
}

bool Scop::IsLoad3dL1UBStmt(const std::string &stmt_name) const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isLoad3d && info.first.name() == stmt_name) {
      return true;
    }
  }
  return false;
}

bool Scop::HasCube() const {
  for (auto &info : data_.stmt_op_Info) {
    if (info.second.isCube) return true;
  }
  return false;
}

bool Scop::IsGemmDataTransposeBlock() const {
  std::string trans_data_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE_BLOCK);
  return IsGemm() && !is_spec_gemm_ && (trans_data_block == "Y");
}

bool Scop::IsGemmWeightTransposeBlock() const {
  std::string trans_weight_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK);
  return IsGemm() && !is_spec_gemm_ && (trans_weight_block == "Y");
}

bool Scop::IsGemmDataTransposeInnerBlock() const {
  std::string trans_data_inner_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE_BLOCK_INNER);
  return IsGemm() && !is_spec_gemm_ && (trans_data_inner_block == "Y");
}
bool Scop::IsGemmWeightTransposeInnerBlock() const {
  std::string trans_weight_inner_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK_INNER);
  return IsGemm() && !is_spec_gemm_ && (trans_weight_inner_block == "Y");
}
bool Scop::IsGemmDataTranspose() const {
  std::string trans_data = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE);
  return IsGemm() && !is_spec_gemm_ &&
         ((trans_data == "Y") || IsGemmDataTransposeBlock() || IsGemmDataTransposeInnerBlock());
}

bool Scop::IsGemmWeightTranspose() const {
  std::string trans_weight = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE);
  return IsGemm() && !is_spec_gemm_ &&
         ((trans_weight == "Y") || IsGemmWeightTransposeBlock() || IsGemmWeightTransposeInnerBlock());
}

bool Scop::IsGemm() const { return HasCube() && !IsConv(); }

bool Scop::IsConv() const {
  std::string n = ExtractStringFromAttrs(ATTR_CONV_FEATURE_NAME);
  return (!n.empty());
}

const isl::union_set Scop::Domain() const { return schedule_.get_domain(); }

Tensor Scop::FindTensorInOrig(const isl::id &var) {
  for (auto i : binds_orig_) {
    if (i.first->op->name == var.get_name()) {
      return i.first;
    }
  }
  return Tensor();
}

Tensor Scop::FindTensorInOrig(const std::string &str) {
  for (auto i : binds_orig_) {
    if (i.first->op->name == str) {
      return i.first;
    }
  }
  return Tensor();
}

void Scop::UpdateComputeAttrInfo() {
  if (IsConv()) {
    FindComputeAttr(ConvATTRList);
  } else if (IsLoad3dL1Ub()) {
    FindComputeAttr(FastPoolingATTRList);
  }
}
void Scop::FindComputeAttr(const std::vector<std::string> &op_keys) {
  for (auto i : data_.stmt_op_Info) {
    if (i.second.isCube || i.second.isLoad3d) {
      const Node *stmt_node = data_.statements.at(i.first);
      if (stmt_node->IsInstance<Provide>()) {
        auto provide = static_cast<const Provide *>(stmt_node);
        const auto cop = provide->func.as<ComputeOpNode>();
        if (cop != nullptr) {
          for (auto j : op_keys) {
            std::string err = "Error: You need to set attr feature " + j + " at akg.tvm.compute()!";
            CHECK(cop->attrs.count(j) != 0) << err;
          }
          attr_info_ = cop->attrs;
        }
      }
      break;
    }
  }
}

// find the dtype of global buffer by the tensor name
Type Scop::GetDtypeOf(const std::string &tensor_name) const {
  for (auto i : binds_) {
    if (i.first->op->name == tensor_name) {
      return i.second->dtype;
    }
  }
  LOG(INFO) << " no such tensor in binds: " << tensor_name;
  return Int(32);
}

Type Scop::GetDtypeOf(const isl::ast_expr &e) const {
  if (auto op = e.as<isl::ast_expr_op>()) {
    isl::id var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
    return GetDtypeOf(var);
  }
  return Int(32);
}

bool Scop::IsInBinds(const std::string &name) const {
  for (auto i : binds_orig_) {
    if (name == i.first->op->name) {
      return true;
    }
  }
  return false;
}

std::string Scop::ConvOutName() {
  for (auto stmt : data_.stmt_op_Info) {
    if (stmt.second.isCube) {
      return stmt.second.C_;
    }
  }
  return "";
}

air::DataType Scop::MadCastType() {
  for (auto stmt : data_.stmt_op_Info) {
    if (stmt.second.isCube) {
      return stmt.second.MadType_;
    }
  }
  return Float(16);
}

std::string Scop::GetcDim() {
  auto matcher = DimInfoMatcher();
  matcher.Visit(body_);
  return matcher.dim();
}

bool Scop::IsFilterCanByPass() {
  bool can_bypass = true;
  auto filter_name = ExtractStringFromAttrs(ATTR_CONV_FILTER_NAME);
  if (tensor_mem_flows_.count(filter_name)) {
    auto filter_memflow = tensor_mem_flows_[filter_name];
    auto it = find(filter_memflow.begin(), filter_memflow.end(), UBL1_);
    if (it != filter_memflow.end()) can_bypass = false;
  }
  return can_bypass;
}

void Scop::ParseIntAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, int *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  if (auto i = e.as<IntImm>()) {
    *attr_to_set = static_cast<int>(i->value);
  } else if (auto ui = e.as<UIntImm>()) {
    *attr_to_set = static_cast<int>(ui->value);
  } else {
    LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as integer";
  }
}

void Scop::ParseBoolAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name, bool *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) != 0) {
    const int invalid_value = -1;
    int attr = invalid_value;
    ParseIntAttr(attrs, attr_name, &attr);
    if (attr != invalid_value) {
      CHECK(attr == 0 || attr == 1) << "Bool attribute " << attr_name << " must be 0 or 1, but found "
                                    << attrs.at(attr_name);
      *attr_to_set = static_cast<bool>(attr);
    }
  }
}

void Scop::ParseStringAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                           std::string *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  if (auto val = e.as<StringImm>()) {
    *attr_to_set = val->value;
  } else {
    LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as string";
  }
}

void Scop::ParseCustomTilingAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                 std::vector<NodeRef> *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  Array<NodeRef> array = air::runtime::Downcast<Array<NodeRef>>(e);
  for (auto d : array) {
    if (d.as<air::CustomTilingNode>()) {
      attr_to_set->emplace_back(d);
    } else {
      LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as CustomTilingNode";
    }
  }
}

void Scop::ParseDynamicShapeAttr(const Map<std::string, NodeRef> &attrs, const std::string &attr_name,
                                 std::vector<NodeRef> *attr_to_set) {
  CHECK(attr_to_set != nullptr);
  if (attrs.count(attr_name) == 0) return;
  const NodeRef &e = attrs.at(attr_name);
  Array<NodeRef> array = air::runtime::Downcast<Array<NodeRef>>(e);
  for (auto d : array) {
    if (d.as<air::DynamicShapeNode>()) {
      attr_to_set->emplace_back(d);
    } else {
      LOG(FATAL) << "Failed to parse attribute: " << attr_name << " = " << e << " as DynamicShapeNode";
    }
  }
}

void Scop::SetAttrs(const Map<std::string, NodeRef> &attrs) {
  if (attrs.empty()) return;
  ParseStringAttr(attrs, "dim", &b_dim_);
  ParseIntAttr(attrs, "kernel_h", &matB_dim_h_);
  ParseIntAttr(attrs, "kernel_w", &matB_dim_w_);
  ParseIntAttr(attrs, "conv_backprop_filter", &conv_back_prop_filter_);
  ParseIntAttr(attrs, "bypassL1", &bypassL1_);
  ParseIntAttr(attrs, "dump_tuning_level", &dump_tuning_level_);
  ParseBoolAttr(attrs, "pragma_rmselfdep", &remove_self_dependence_);
  ParseBoolAttr(attrs, "pragma_force_rmselfdep", &force_remove_self_dependence_);
  ParseBoolAttr(attrs, "pragma_reschedule", &compute_reschedule_);
  ParseBoolAttr(attrs, "pragma_remove_invariant_dependence", &remove_invariant_dependence_);
  ParseBoolAttr(attrs, "pragma_disable_schedule_shift", &disable_schedule_shift_);
  ParseBoolAttr(attrs, "pragma_enable_schedule_max_constant", &enable_schedule_max_constant_);
  ParseBoolAttr(attrs, "pragma_disable_loop_reversal", &disable_loop_reversal_);
  ParseBoolAttr(attrs, "pragma_disable_loop_fusion", &disable_loop_fusion_);
  ParseBoolAttr(attrs, "pragma_reorder_schedule", &reorder_schedule_);
  ParseBoolAttr(attrs, "pragma_modshift", &mod_schedule_shift_);
  ParseBoolAttr(attrs, "pragma_conv_special_dma", &conv_special_dma_);
  ParseBoolAttr(attrs, "pragma_checkcoincident", &tile_check_coincident_);
  ParseBoolAttr(attrs, "pragma_opt_for_davinci", &optimize_for_davinci_);
  ParseBoolAttr(attrs, "pragma_sink_last_axis", &sink_last_axis_);
  ParseBoolAttr(attrs, "pragma_keep_outer_band_order", &keep_outer_band_order_);
  ParseBoolAttr(attrs, "pragma_disable_group", &disable_group_);
  ParseBoolAttr(attrs, "pragma_tile_inner_band", &tile_inner_band_);
  ParseBoolAttr(attrs, "pragma_set_all_coincident", &pragma_set_all_coincident_);
  ParseBoolAttr(attrs, "enable_feature_library", &enable_feature_library_);
  ParseBoolAttr(attrs, "enable_hoist_cond_write", &enable_hoist_cond_write_);
  ParseBoolAttr(attrs, "enable_mark_multi_core", &enable_mark_multi_core_);
  ParseStringAttr(attrs, "kernel_name", &kernel_name_);
  ParseIntAttr(attrs, "dump_pass_ir", &dump_pass_ir_);
  ParseStringAttr(attrs, "dump_poly_dir", &dump_poly_dir_);
  ParseIntAttr(attrs, "isolated_idx", &isolated_idx_);
  ParseCustomTilingAttr(attrs, "custom_tiling", &custom_tiling_);
  ParseDynamicShapeAttr(attrs, "dynamic_shape", &dynamic_shape_);
  ParseIntAttr(attrs, "dynamic_shape_bound", &dynamic_shape_bound_);
  ParseIntAttr(attrs, "pragma_tilesize_is_var", &tile_size_is_var_);
  ParseIntAttr(attrs, "pragma_outerband_need_split", &outer_band_need_split_);
  ParseIntAttr(attrs, "pragma_is_conv", &pragma_is_conv_);
  ParseBoolAttr(attrs, "dynamic_shape_conv_full_parametric", &dynamic_shape_conv_full_parametric_);
  ParseBoolAttr(attrs, "pragma_analyze_reuse_buffer", &pragma_analyze_reuse_buffer_);
  ParseBoolAttr(attrs, "pragma_speedup_tiling", &pragma_speedup_tiling_);
  ParseBoolAttr(attrs, "pragma_allow_tail_tiling", &pragma_allow_tail_tiling_);
  ParseBoolAttr(attrs, "pragma_analyze_multicore", &pragma_analyze_multicore_);

  if (force_remove_self_dependence_) {
    LOG(WARNING) << "pragma_force_rmselfdep should be used with care. "
                 << "It removes all self dependence and cannot ensure that reduce axis do not use multicore.";
  }

  for (auto iter : attrs) {
    if (iter.first == ATTR_CONV_GMM_FEATURE || iter.first == ATTR_CONV_GMM_WEIGHT ||
        iter.first == ATTR_GEMM_DATA_TRANSPOSE || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE ||
        iter.first == ATTR_GEMM_DATA_TRANSPOSE_BLOCK || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK ||
        iter.first == ATTR_GEMM_DATA_TRANSPOSE_BLOCK_INNER || iter.first == ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK_INNER) {
      attr_info_.Set(iter.first, iter.second);
    }
  }
  if (is_spec_gemm_) attr_info_ = attrs;
}

void Scop::RecordReduceStmt(const isl::id &stmt_id, const std::vector<std::string> &reduce_axis_list) {
  data_.reduce_stmts[stmt_id] = reduce_axis_list;
}

int Scop::GetAttrValue(const std::string &key) {
  CHECK(attr_info_.find(key) != attr_info_.end());
  if (attr_info_[key].as<IntImm>() != nullptr) return attr_info_[key].as<IntImm>()->value;
  if (attr_info_[key].as<FloatImm>() != nullptr) {
    float res = attr_info_[key].as<FloatImm>()->value;
    LOG(WARNING) << "attr: " << key << " : should be an integer, but found float. Force convert to int.";
    return static_cast<int>(res);
  }
  return -1;
}

Tensor Scop::FindTensorWithLargestShape(const std::string &name) {
  size_t largest_size = 0;
  Tensor largest_tensor;
  for (auto i : buffer_def_infos_) {
    if (!i.tensor.defined()) continue;
    if (i.dst_tensor_id.get_name() == name) {
      size_t tensor_size = 1;
      for (auto dim : i.tensor->shape) {
        if (dim.as<IntImm>()) {
          tensor_size *= dim.as<IntImm>()->value;
        }
      }
      if (tensor_size > largest_size) {
        largest_size = tensor_size;
        largest_tensor = i.tensor;
      }
    }
  }
  for (auto i : binds_) {
    if (!i.first.defined()) continue;
    if (i.first->op->name == name) {
      size_t tensor_size = 1;
      for (auto dim : i.first->shape) {
        if (dim.as<IntImm>()) {
          tensor_size *= dim.as<IntImm>()->value;
        }
      }
      if (tensor_size > largest_size) {
        largest_size = tensor_size;
        largest_tensor = i.first;
      }
    }
  }
  if (largest_size > 0) return largest_tensor;
  CHECK(false) << name << " is not declared in binds and promoted arrays";
  return Tensor();
}

Tensor Scop::FindTensorWithLargestShape(const isl::id &var) { return FindTensorWithLargestShape(var.get_name()); }

Tensor Scop::FindTensor(const std::string &str) {
  for (auto i : buffer_def_infos_) {
    if (str == i.dst_tensor_id.get_name() && i.is_bind_tensor && i.tensor.defined()) {
      return i.tensor;
    }
  }
  for (auto i : binds_) {
    if (i.first->op->name == str) {
      return i.first;
    }
  }
  CHECK(false) << str << " is not declared in binds and promoted arrays";
  return Tensor();
}

Tensor Scop::FindTensor(const isl::id &var) {
  for (const auto &i : buffer_def_infos_) {
    if (i.dst_tensor_id.get_name() == var.get_name() && i.is_bind_tensor && i.tensor.defined()) {
      return i.tensor;
    }
  }
  for (const auto &i : binds_) {
    if (i.first->op->name == var.get_name()) {
      return i.first;
    }
  }
  CHECK(false) << var.to_str() << " is not declared in binds and promoted arrays";
  return Tensor();
}

isl::id Scop::GetOriginTensorId(const std::string &name) const {
  std::string tensor_name = name;
  size_t pos = name.find("_local_");
  if (std::string::npos != pos) {
    tensor_name = name.substr(0, pos);
  }
  return isl::id(ctx_, tensor_name);
}

isl::id Scop::GetOriginTensorId(const isl::id &id) const { return GetOriginTensorId(id.get_name()); }

bool Scop::InitRangeStrideVec() {
  if (!data_.range_stride.empty()) return false;

  if (data_.range_info.empty()) {
    LOG(WARNING) << "range_info is not specified, please check";
    return false;
  }

  data_.range_stride.push_back(1);
  for (uint64_t i = data_.range_info.size(); i >= 1; --i) {
    data_.range_stride.insert(data_.range_stride.begin(),
                              data_.range_info[i - 1].size() * (unsigned int)data_.range_stride[0]);
  }
  return true;
}

std::vector<int> Scop::GetIsolateVec(int range_idx) {
  static_cast<void>(InitRangeStrideVec());
  std::vector<int> idx;
  for (unsigned int i = 0; i < data_.range_stride.size() - 1; i++) {
    CHECK_NE(data_.range_stride[i], 0);
    CHECK_NE(data_.range_stride[i + 1], 0);
    idx.push_back(range_idx % data_.range_stride[i] / data_.range_stride[i + 1]);
  }
  return idx;
}

std::vector<Range> Scop::GetRange(int range_idx) {
  std::vector<int> idx = GetIsolateVec(range_idx);
  std::vector<Range> res;
  CHECK(idx.size() == data_.range_info.size());
  for (unsigned int i = 0; i < idx.size(); i++) {
    res.push_back(data_.range_info[i][(unsigned int)idx[i]]);
  }
  return res;
}

std::unordered_map<std::string, Expr> Scop::GetConvInfoForTiling() {
  std::unordered_map<std::string, Expr> conv_info;
  conv_info[ATTR_CONV_FEATURE_H] = this->ExtractExprFromAttrs(ATTR_CONV_FEATURE_H);
  conv_info[ATTR_CONV_FEATURE_W] = this->ExtractExprFromAttrs(ATTR_CONV_FEATURE_W);
  conv_info[ATTR_CONV_KERNEL_H] = this->ExtractExprFromAttrs(ATTR_CONV_KERNEL_H);
  conv_info[ATTR_CONV_KERNEL_W] = this->ExtractExprFromAttrs(ATTR_CONV_KERNEL_W);
  conv_info[ATTR_CONV_PAD_TOP] = this->ExtractExprFromAttrs(ATTR_CONV_PAD_TOP);
  conv_info[ATTR_CONV_PAD_LEFT] = this->ExtractExprFromAttrs(ATTR_CONV_PAD_LEFT);
  conv_info[ATTR_CONV_STRIDE_H] = this->ExtractExprFromAttrs(ATTR_CONV_STRIDE_H);
  conv_info[ATTR_CONV_STRIDE_W] = this->ExtractExprFromAttrs(ATTR_CONV_STRIDE_W);
  conv_info[ATTR_CONV_DILATION_H] = this->ExtractExprFromAttrs(ATTR_CONV_DILATION_H);
  conv_info[ATTR_CONV_DILATION_W] = this->ExtractExprFromAttrs(ATTR_CONV_DILATION_W);
  return conv_info;
}

void Scop::GetConvMNKInfo(std::vector<DimensionInfo> &dimInfos_conv) {
  std::vector<DimensionInfo> L1_factors;
  std::vector<DimensionInfo> L0_factors;

  std::unordered_set<std::string> conv_pragmas = {
    ATTR_CONV_TILE_W, ATTR_CONV_TILE_H,  ATTR_CONV_TILE_CO, ATTR_CONV_TILE_M,  ATTR_CONV_TILE_N,
    ATTR_CONV_TILE_K, ATTR_CONV_M_INNER, ATTR_CONV_N_INNER, ATTR_CONV_K_INNER, ATTR_CONV_TILE_CIN,
    ATTR_CONV_TILE_B, ATTR_CONV_TILE_KH, ATTR_CONV_TILE_KW};

  for (auto dim : dimInfos_conv) {
    if (conv_pragmas.find(dim.axis) != conv_pragmas.end()) {
      L0_factors.emplace_back(dim);
    } else {
      L1_factors.emplace_back(dim);
    }
  }
  dimInfos_conv = L1_factors;
  conv_mnk_dims_ = L0_factors;
  if (is_dynamic_) {
    for (const auto &dim : conv_mnk_dims_) {
      fractal_int_info_[dim.axis] = IntImm::make(Int(32), dim.l1_tiling_size);
      attr_info_.Set(dim.axis, IntImm::make(Int(32), dim.l1_tiling_size));
    }
  } else {
    const int c0_size = 16;
    const int int_imm_num_bits = 32;
    for (const auto &dim : conv_mnk_dims_) {
      int l0tile = static_cast<int>(dim.l0_tiling_size);
      if (dim.axis == ATTR_CONV_TILE_M || dim.axis == ATTR_CONV_TILE_N || dim.axis == ATTR_CONV_TILE_K) {
        // multiply outer tile size with inner size
        l0tile *= c0_size;
      }
      fractal_int_info_[dim.axis] = l0tile;
      attr_info_.Set(dim.axis, IntImm::make(Int(int_imm_num_bits), l0tile));
    }
  }
}

// Init set_dim info
void Scop::InitDimensionInfo(const isl::schedule &sch_init) {
  // get compute dim
  std::string dim = GetcDim();
  // get build dim
  if (dim.empty()) {
    dim = GetbDim();
  }

  // apply default tiling
  if (dim.empty()) {
    auto tiling_res = GenerateTiling(this, sch_init, custom_tiling_, dynamic_shape_);
    dim_infos_ = tiling_res.first;
    tiling_constraints_ = tiling_res.second;
    if (IsConv()) GetConvMNKInfo(dim_infos_);
    return;
  }

  const std::string pattern = " ";
  std::vector<std::string> str = Split(dim, pattern);
  const int dim_info_entry_size = 4;
  CHECK(!str.empty() && !(str.size() % dim_info_entry_size)) << "Error: You need to set dim !";
  int sequence = 0;
  for (size_t i = 0; i < str.size(); i += dim_info_entry_size) {
    Scop::DimensionInfo dim_info;
    char *endptr = nullptr;
    const int radix = 10;
    dim_info.index = strtol(str[i].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0') LOG(FATAL) << "failed to convert string " << str[i] << " to number";
    const int max_dim_index = 16;
    CHECK(dim_info.index < max_dim_index) << "set_dim index must be less than " << max_dim_index << "!";
    dim_info.axis = str[i + 1];
    const int default_tiling_size = 65535;
    endptr = nullptr;
    int64_t str_2_number = strtol(str[i + 2].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0' || str_2_number <= 0) {
      dim_info.l1_tiling_size = default_tiling_size;
    } else {
      dim_info.l1_tiling_size = str_2_number;
    }
    endptr = nullptr;
    int64_t str_3_number = strtol(str[i + 3].c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0' || str_3_number <= 0) {
      dim_info.l0_tiling_size = default_tiling_size;
    } else {
      dim_info.l0_tiling_size = str_3_number;
    }
    dim_info.dim_seq = sequence;
    sequence++;
    dim_infos_.push_back(dim_info);
  }
}

void Scop::MergeTilingInfo(Tiles &tiling_infos) {
  int64_t tiles_num = 0;
  for (unsigned i = 0; i < dim_infos_.size(); ++i) {
    if (tiles_num <= dim_infos_[i].index) {
      tiles_num = dim_infos_[i].index + 1;
    }
  }
  tiling_infos.resize((size_t)tiles_num);

  for (unsigned i = 0; i < dim_infos_.size(); ++i) {
    tiling_infos[(unsigned int)dim_infos_[i].index].dim_infos.push_back(dim_infos_[i]);
  }
}

void Scop::GetParams() {
  auto FloorDivToDiv = [](Expr expr) -> Expr {
    if (const auto add = expr.as<air::ir::Add>()) {
      // case 1: floordiv(a, b) + 1 ==> (a + b) / b
      if (const auto imm = add->b.as<IntImm>()) {
        if (imm->value == 1) {
          if (const auto fd = add->a.as<air::ir::FloorDiv>()) {
            if (const auto denominator = fd->b.as<IntImm>()) {
              if (denominator->value == 2) {
                return CanonicalSimplify(air::ir::Div::make((fd->a + fd->b), fd->b));
              }
            }
            return air::ir::Div::make(CanonicalSimplify(fd->a), fd->b) + 1;
          }
        }
      }
    }
    return expr;
  };
  for (auto x : binds_orig_) {
    for (const auto &expr : x.second->shape) {
      if (!is_const(expr)) {
        RegisterParam(FloorDivToDiv(expr));
      }
    }
  }

  for (auto it : outer_let_stmts_) {
    if (auto let_op = it.as<LetStmt>()) {
      if (let_op->var.type().is_int() || let_op->var.type().is_uint()) {
        CHECK(params_.count(let_op->var->name_hint) == 0) << "duplicate name in params: " << let_op->var;
        params_.emplace(let_op->var->name_hint, let_op->var);
        params_rev_map_.emplace(let_op->var->name_hint, let_op->var);
      }
    }
  }
}

std::pair<std::string, std::string> ExprToString(const Expr &expr) {
  std::ostringstream os;
  if (auto var = expr.as<Variable>()) {
    os << var->name_hint;
  } else {
    os << expr;
  }
  std::string expr_str = os.str();

  std::string name = expr_str;
  // replace special chars with '_'
  std::replace_if(
    name.begin(), name.end(), [](const char c) -> bool { return !std::isalnum(c); }, '_');
  // remove leading '_'
  auto it = std::find_if(name.begin(), name.end(), [](const char c) { return c != '_'; });
  name.erase(name.begin(), it);
  // remove redundant '_'
  std::regex rx("_+");
  name = std::regex_replace(name, rx, "_");
  return std::pair<std::string, std::string>(expr_str, name);
}

void Scop::RegisterParam(const Expr &expr) {
  if (is_const(expr)) return;
  if (auto op = expr.as<air::ir::Mul>()) {
    if (is_const(op->a)) {
      RegisterParam(op->b);
      return;
    }
    if (is_const(op->b)) {
      RegisterParam(op->a);
      return;
    }
  } else if (auto op = expr.as<air::ir::Add>()) {
    RegisterParam(op->a);
    RegisterParam(op->b);
    return;
  } else if (auto op = expr.as<air::ir::Sub>()) {
    RegisterParam(op->a);
    RegisterParam(op->b);
    return;
  } else if (auto op = expr.as<air::ir::FloorDiv>()) {
    RegisterParam(op->a);
    RegisterParam(op->b);
    return;
  }

  // register the expression itself
  auto pair = ExprToString(expr);
  auto expr_str = pair.first;
  auto name = pair.second;
  if (params_.count(expr_str) > 0) return;
  if (params_rev_map_.count(name) > 0) {
    int suffix = 1;
    while (params_rev_map_.count(name + std::to_string(suffix)) > 0) ++suffix;
    name = name + std::to_string(suffix);
  }
  params_.emplace(expr_str, Variable::make(expr.type(), name));
  params_rev_map_.emplace(name, expr);
}

Scop::AtomicType Scop::GetAtomicWrite(const isl::id &id) const {
  for (const auto &i : data_.statements) {
    const Node *stmt_node = i.second;
    if (stmt_node->IsInstance<Provide>()) {
      auto provide = static_cast<const Provide *>(stmt_node);
      if (const auto cop = provide->func.as<ComputeOpNode>()) {
        if (cop->attrs.count(ATTR_ATOMIC_ADD) != 0) {
          if (auto str_op = cop->attrs.at(ATTR_ATOMIC_ADD).as<StringImm>()) {
            auto str = str_op->value;
            if (str == id.get_name()) return Scop::AtomicType::Add;
          }
        }
      }
    }
  }
  return Scop::AtomicType::Equ;
}

void Scop::CreateConvModel(bool is_dynamic) {
  if (!attr_info_.empty()) {
    if (attr_info_.count(ATTR_CONV_BACKPROP_INPUT) > 0) {
      try {
        model_ = new ConvolutionBackpropInputModel(attr_info_, is_dynamic);
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionBackpropInputModel";
      }
    } else if (attr_info_.count(ATTR_CONV_BACKPROP_FILTER) > 0) {
      try {
        model_ = new ConvolutionBackpropFilterModel(attr_info_, is_dynamic);
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionBackpropFilterModel";
      }
    } else {
      try {
        model_ = new ConvolutionForwardModel(attr_info_, is_dynamic);
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionForwardModel";
      }
    }
    if (model_) {
      static_cast<void>(model_->infer_L1_tile());
    }
  }
}

void Scop::UpdateFractalIntInfoConvForward(int isolate_idx) {
  auto C0_SIZE = IntImm::make(Int(32), 16);
  fractal_int_info_[ATTR_CONV_TILE_N] = floordiv(model_->get_co_isolate_info(isolate_idx).inner, C0_SIZE);

  Expr m = model_->get_h_win_isolate_info(isolate_idx).inner * model_->get_w_win_isolate_info(isolate_idx).inner;
  fractal_int_info_[ATTR_CONV_GMM_M] = m;
  fractal_int_info_[ATTR_CONV_TILE_M] = floordiv(m + C0_SIZE - 1, C0_SIZE);
  fractal_int_info_[ATTR_CONV_M_INNER] = C0_SIZE;
  fractal_int_info_[ATTR_CONV_M_CUT_SIZE] = model_->get_w_win_isolate_info(isolate_idx).inner;
  if (!is_dynamic_) {
    if (IsConvBackpropInput()) {
      CHECK(model_->conv_.filter.kh.as<IntImm>());
      CHECK(model_->conv_.filter.kw.as<IntImm>());
      matB_dim_h_ = model_->conv_.filter.kh.as<IntImm>()->value;
      matB_dim_w_ = model_->conv_.filter.kw.as<IntImm>()->value;
    }
  } else {
    auto tile_h = ExtractExprFromAttrs(ATTR_CONV_TILE_H);
    tile_h = tile_h.get() ? tile_h : IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_TILE_H));
    if (!Equal(tile_h, -1)) fractal_int_info_[ATTR_CONV_TILE_H] = tile_h;
    auto tile_w = ExtractExprFromAttrs(ATTR_CONV_TILE_W);
    tile_w = tile_w.get() ? tile_w : IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_TILE_W));
    if (!Equal(tile_w, -1)) fractal_int_info_[ATTR_CONV_TILE_W] = tile_w;

    fractal_int_info_[ATTR_CONV_KERNEL_H] = IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_KERNEL_H));
    fractal_int_info_[ATTR_CONV_STRIDE_H] = IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_STRIDE_H));
    fractal_int_info_[ATTR_CONV_KERNEL_W] = IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_KERNEL_W));
    fractal_int_info_[ATTR_CONV_STRIDE_W] = IntImm::make(Int(32), ExtractIntFromAttrs(ATTR_CONV_STRIDE_W));
  }
}

void Scop::UpdateFractalIntInfoConvBackpropFilter(int isolate_idx) {
  // gemm_idx order as follow:
  // for (Ci Cut) {
  //   for (KH Cut) {
  //     for (KW Cut) {
  //       for (Co Cut) {
  //         for (Batch Cut) {
  //           for (H Cut) {
  //             for (W Cut) {
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  const int block_size = 16;

  fractal_int_info_[ATTR_SPEC_GEMM_BATCH] = model_->get_b_isolate_info(isolate_idx).inner;
  fractal_int_info_[ATTR_SPEC_GEMM_M] = model_->get_co_isolate_info(isolate_idx).inner;
  CHECK_EQ(fractal_int_info_[ATTR_SPEC_GEMM_M].as<IntImm>()->value % block_size, 0);
  fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN] = fractal_int_info_[ATTR_SPEC_GEMM_M];
  CHECK(fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN].as<IntImm>());
  CHECK(model_->tile_.cut_m.as<IntImm>());
  if (fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN].as<IntImm>()->value < model_->tile_.cut_m.as<IntImm>()->value) {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_M] = fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN];
  } else {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_M] = model_->tile_.cut_m;
  }
  fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN] =
    fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN].as<IntImm>()->value / block_size;
  fractal_int_info_[ATTR_SPEC_GEMM_M_INNER] = block_size;
  fractal_int_info_[ATTR_CONV_TILE_M] = fractal_int_info_[ATTR_SPEC_GEMM_M_ALIGN];
  fractal_int_info_[ATTR_CONV_M_INNER] = block_size;

  CHECK(model_->get_h_win_isolate_info(isolate_idx).inner.as<IntImm>());
  CHECK(model_->get_w_win_isolate_info(isolate_idx).inner.as<IntImm>());
  int h_tile = model_->get_h_win_isolate_info(isolate_idx).inner.as<IntImm>()->value;
  int w_tile = model_->get_w_win_isolate_info(isolate_idx).inner.as<IntImm>()->value;
  fractal_int_info_[ATTR_SPEC_GEMM_K] = h_tile * w_tile;
  fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN] = (h_tile * w_tile + block_size - 1) / block_size * block_size;
  CHECK(fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN].as<IntImm>());
  CHECK(model_->tile_.cut_k.as<IntImm>());
  if (fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN].as<IntImm>()->value < model_->tile_.cut_k.as<IntImm>()->value) {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_K] = fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN];
  } else {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_K] = model_->tile_.cut_k;
  }
  fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN] =
    fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN].as<IntImm>()->value / block_size;
  fractal_int_info_[ATTR_SPEC_GEMM_K_INNER] = block_size;
  fractal_int_info_[ATTR_CONV_TILE_K] = fractal_int_info_[ATTR_SPEC_GEMM_K_ALIGN];
  fractal_int_info_[ATTR_CONV_K_INNER] = block_size;

  CHECK(model_->get_ci_isolate_info(isolate_idx).inner.as<IntImm>());
  CHECK(model_->get_kh_isolate_info(isolate_idx).inner.as<IntImm>());
  CHECK(model_->get_kw_isolate_info(isolate_idx).inner.as<IntImm>());
  int ci_tile = model_->get_ci_isolate_info(isolate_idx).inner.as<IntImm>()->value;
  int kh_tile = model_->get_kh_isolate_info(isolate_idx).inner.as<IntImm>()->value;
  int kw_tile = model_->get_kw_isolate_info(isolate_idx).inner.as<IntImm>()->value;
  fractal_int_info_[ATTR_SPEC_GEMM_N] = ci_tile * kh_tile * kw_tile;
  CHECK_EQ(fractal_int_info_[ATTR_SPEC_GEMM_N].as<IntImm>()->value % block_size, 0);
  fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN] = fractal_int_info_[ATTR_SPEC_GEMM_N];
  CHECK(fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN].as<IntImm>());
  CHECK(model_->tile_.cut_n.as<IntImm>());
  if (fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN].as<IntImm>()->value < model_->tile_.cut_n.as<IntImm>()->value) {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_N] = fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN];
  } else {
    fractal_int_info_[ATTR_SPEC_GEMM_TILE_N] = model_->tile_.cut_n;
  }
  fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN] =
    fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN].as<IntImm>()->value / block_size;
  fractal_int_info_[ATTR_SPEC_GEMM_N_INNER] = block_size;
  fractal_int_info_[ATTR_CONV_TILE_N] = fractal_int_info_[ATTR_SPEC_GEMM_N_ALIGN];
  fractal_int_info_[ATTR_CONV_N_INNER] = block_size;

  out_reduce_init_ = 0;
  int l1_reduce_base = model_->b_base * model_->h_base * model_->w_base;
  if ((l1_reduce_base > 1 && isolate_idx % l1_reduce_base == 0) || (l1_reduce_base == 1)) {
    out_reduce_init_ = 1;
  }
}

void Scop::UpdateFractalIntInfo(int gemm_idx) {
  if (IsConvBackpropFilter()) {
    if (!is_dynamic_) UpdateFractalIntInfoConvBackpropFilter(gemm_idx);
  } else {
    UpdateFractalIntInfoConvForward(gemm_idx);
  }
}

Expr Scop::ReplacePragmaPrimeByVar(Expr pragma) {
  if (is_dynamic_) {
    if (const auto prime = pragma.as<IntImm>()) {
      for (auto dim : this->conv_mnk_dims_) {
        if (dim.pragma.defined() && ((dim.l1_tiling_size == prime->value))) {
          return RemoveCast(dim.l1_var);
        } else if (dim.l1_tiling_size / 16 == prime->value) {
          return floordiv(dim.l1_var + 15, 16);
        }
      }
    }
  }
  return pragma;
}

std::vector<std::vector<int>> Scop::AddTileInfo(const std::vector<std::vector<int>> &partition_info) {
  std::vector<std::vector<int>> info;
  PartitionSingle *single = PartitionSingle::getInstance();
  if (single == nullptr) {
    return partition_info;
  } else if (PartitionSingle::getTimes() < 2) {
    // first time gemm or m isolate main gemm
    return partition_info;
  }

  for (auto it : partition_info) {
    info.push_back(it);
  }
  return info;
}

static bool CompareFootprintOfMaps(const isl::map &local_access, const isl::map &global_access) {
  isl::multi_val local_write_footprint = local_access.range_simple_fixed_box_hull().size();
  isl::multi_val global_write_footprint = global_access.range_simple_fixed_box_hull().size();
  if (local_write_footprint.size() != global_write_footprint.size()) return false;
  unsigned int dim = local_write_footprint.size();
  for (unsigned i = 0; i < dim; ++i) {
    if (local_write_footprint.get_val(i) < global_write_footprint.get_val(i)) return false;
  }
  return true;
}

bool Scop::IsWriteWholeBufferFootPrint(const isl::id &poly_ref_id) const {
  for (const auto &buffer : active_buffer_footprints_) {
    auto group = buffer.second.cluster;
    for (const auto &reference : group->tensor_foot_prints) {
      if (reference->id == poly_ref_id) {
        CHECK(reference->type == ReferenceType::Write);
        return CompareFootprintOfMaps(reference->scoped_access, group->RichWriteRelations());
      }
    }
  }
  LOG(WARNING) << "buffer for " << poly_ref_id << " is not found";
  return false;
}

/*
 * Checks if a promoted tensor is written conditionally, and there is no other unconditional statement
 * in the same buffer that writes the whole promoted tensor.
 */
bool Scop::IsConditionalWriteTensor(const std::string &name,
                                    const std::vector<std::pair<isl::id, isl::id>> &write_stmts) const {
  bool has_conditional_write = false;
  bool has_unconditional_full_write = false;
  for (const auto &pair : write_stmts) {
    auto stmt_id = pair.first;
    auto poly_ref_id = pair.second;
    CHECK_GT(data_.statements.count(stmt_id), 0);
    const Node *stmt = data_.statements.at(stmt_id);
    if (stmt->IsInstance<IfThenElse>()) {
      has_conditional_write = true;
    } else if (IsWriteWholeBufferFootPrint(poly_ref_id)) {
      has_unconditional_full_write = true;
    }
  }
  return has_conditional_write && !has_unconditional_full_write;
}

void Scop::FindConditionalWritePromotions() {
  std::unordered_map<std::string, std::vector<std::pair<isl::id, isl::id>>> tensor_write_stmts_map;
  data_.writes.foreach_map([&tensor_write_stmts_map](const isl::map &map) -> void {
    std::string tensor_name = map.get_tuple_id(isl_dim_out).name();
    isl::id stmt_id = map.domain().unwrap().get_tuple_id(isl_dim_in);
    isl::id poly_ref_id = map.domain().unwrap().get_tuple_id(isl_dim_out);
    tensor_write_stmts_map[tensor_name].push_back(std::make_pair(stmt_id, poly_ref_id));
  });

  for (auto bind : binds_orig_) {
    auto name = bind.first->op->name;
    if (tensor_write_stmts_map.count(name) == 0) continue;
    if (IsConditionalWriteTensor(name, tensor_write_stmts_map[name])) {
      LOG(INFO) << "found conditionally written promoted tensor: " << name
                << ", buffer will be sinked to the computation.";
      conditional_write_buffer_footprints_.insert(name);
    }
  }
}

StmtIdHashMap Scop::StmtWriteMap() {
  StmtIdHashMap stmt_write_map;
  isl::union_map write_stmt = data_.writes.domain_factor_domain();
  for (auto stmt : write_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto write_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_write_map[stmtId].push_back(write_tensor);
  }
  return stmt_write_map;
}

StmtIdHashMap Scop::StmtReadMap() {
  StmtIdHashMap stmt_read_map;
  isl::union_map read_stmt = data_.reads.domain_factor_domain();
  for (auto stmt : read_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto read_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_read_map[stmtId].push_back(read_tensor);
  }
  return stmt_read_map;
}

StmtIdHashMap Scop::StmtCopyinMap() {
  StmtIdHashMap stmt_copyin_map;
  isl::union_map copyin_stmt = data_.copyin.domain_factor_domain();
  for (auto stmt : copyin_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto read_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_copyin_map[stmtId].push_back(read_tensor);
  }
  return stmt_copyin_map;
}

bool Scop::IsCopyinTensor(const std::string &tensor_name) {
  CHECK_NE(tensor_name, "");
  StmtIdHashMap copyin_map = StmtCopyinMap();
  for (const auto &item : copyin_map) {
    for (const auto &tensor : item.second) {
      if (tensor.get_name() == tensor_name) return true;
    }
  }
  return false;
}

bool Scop::IsConvHeadTail(const std::string &conv_output, const isl::id &stmtId, const StmtOpInfo &op_info,
                          const StmtIdHashMap &op_write_map) {
  if (!IsConv()) return false;

  if (op_info.isCube || op_info.isCubeAssign) return false;

  if (op_info.ops.size() != 1) return false;

  if (op_write_map.find(stmtId) == op_write_map.end()) return false;

  if (op_write_map.at(stmtId).size() != 1) return false;

  if (op_info.ops[0] == PolyOpType::broadcast || op_info.ops[0] == PolyOpType::assignment) {
    isl::id writeId = op_write_map.at(stmtId)[0];
    if (writeId.get_name() == conv_output) return true;
  }

  return false;
}

void Scop::CreateDataFlowInfo() {
  StmtIdHashMap op_write_map = StmtWriteMap();
  StmtIdHashMap op_read_map = StmtReadMap();
  std::string conv_output;
  if (IsConv()) {
    conv_output = ConvOutName();
  }
  uint64_t stmtNum = data_.stmt_op_Info.size();
  stmt_type_.resize(stmtNum);
  DMADataFlow dma_dataflow;
  for (auto stmt : data_.stmt_op_Info) {
    std::string name = stmt.first.get_name();
    size_t pos = name.find("_");
    CHECK(pos != name.size() - 1);
    std::string subNum = name.substr(pos + 1, name.size() - pos - 1);
    char *endptr = nullptr;
    const int radix = 10;
    size_t num = strtol(subNum.c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0') LOG(FATAL) << "failed to convert string " << subNum << " to number";

    if (IsConv() && IsConvHeadTail(conv_output, stmt.first, stmt.second, op_write_map)) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::VECTOR);
      continue;
    }

    if (stmt.second.isCube && IsConv()) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::CUBE_CONV);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::CUBE_CONV, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isCube && !IsConv()) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::CUBE_GEMM);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::CUBE_GEMM, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isIm2col || stmt.second.isLoad3d) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::IM2COL_UB);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::IM2COL_UB, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (!stmt.second.isCube && !stmt.second.isCubeAssign) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::VECTOR);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::VECTOR, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isCubeAssign) {
      stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::VECTOR);
    }
  }
  dma_dataflow.FusionAnalysis();
  dma_dataflow.OpDataflowInfo(tensor_name_flows_, tensor_mem_flows_);
}

void Scop::AddStateTensorsDataFlow() {
  // build init list
  // init list   TensorID   input0      DDR --> L1 --> L1 --> L0A
  //             TensorID   input1      DDR --> L0B
  //             TensorID   input2      DDR --> UB
  //             TensorID   output0     DDR <-- UB <-- L0C
  //             TensorID   max_1       UB  --> DDR
  // build whole list
  // add below node
  //   TensorID  input0_local_L1               L1 --> L1 --> L0A
  //   TensorID  input0_fractal_L1             L1 --> L0A
  //   TensorID  input0_fractal_L1_local_L0A   L0A
  //   TensorID  input1_local_L1_local_L0B     L0B
  //   TensorID  output0_local_UB               UB <-- L0C
  //   TensorID  output0_local_UB_local_L0C     L0C
  //   TensorID  input2_local_UB               UB
  //   TensorID   max_1_local_UB               UB
  CHECK_EQ(tensor_mem_flows_.size(), tensor_name_flows_.size());
  CHECK_GT(tensor_mem_flows_.size(), 0);
  for (const auto &tensor : tensor_mem_flows_) {
    std::string name = tensor.first;
    if (tensor_name_flows_.find(name) == tensor_name_flows_.end()) continue;
    auto it = std::find(tensor_mem_flows_[name].begin(), tensor_mem_flows_[name].end(), UBL1_);
    auto it2 = std::find(tensor_mem_flows_[name].begin(), tensor_mem_flows_[name].end(), L1_);
    if (it != tensor_mem_flows_[name].end() && it2 != tensor_mem_flows_[name].end()) {
      std::vector<std::string> name_flow1, name_flow2;
      MemFlow mem_flow1, mem_flow2;
      if (IsConv() || IsGemm()) {
        name_flow1.push_back(tensor_name_flows_[name][0]);
        mem_flow1.push_back(tensor_mem_flows_[name][0]);
        name_flow1.push_back(tensor_name_flows_[name][2]);
        mem_flow1.push_back(tensor_mem_flows_[name][2]);
        name_flow1.push_back(tensor_name_flows_[name][1]);
        mem_flow1.push_back(tensor_mem_flows_[name][1]);

        name_flow2.push_back(tensor_name_flows_[name][0]);
        mem_flow2.push_back(tensor_mem_flows_[name][0]);
        name_flow2.push_back(tensor_name_flows_[name][2]);
        mem_flow2.push_back(tensor_mem_flows_[name][2]);
        name_flow2.push_back(tensor_name_flows_[name][3]);
        mem_flow2.push_back(tensor_mem_flows_[name][3]);
      }
      if (IsConv() && IsA(name)) {
        name_flow2.push_back(tensor_name_flows_[name][4]);
        mem_flow2.push_back(tensor_mem_flows_[name][4]);
      }

      AddTensorDataFlow(mem_flow1, name_flow1);
      AddTensorDataFlow(mem_flow2, name_flow2);

      continue;
    }
    AddTensorDataFlow(tensor.second, tensor_name_flows_[name]);
  }

  size_t length = buffer_def_infos_.size();
  for (size_t tensor_idx = 0; tensor_idx < length; tensor_idx++) {
    if (buffer_def_infos_[tensor_idx].data_stream.size() == 1) continue;

    isl::id ancestor_id = buffer_def_infos_[tensor_idx].tensor_id;
    for (size_t idx = 1; idx < buffer_def_infos_[tensor_idx].data_stream.size(); ++idx) {
      if (idx + 1 == buffer_def_infos_[tensor_idx].data_stream.size()) continue;
      std::vector<std::pair<isl::id, MemType>> sub_data_stream = buffer_def_infos_[tensor_idx].PartialDataStream(idx);
      AddOneBufferDefInfo(ancestor_id, sub_data_stream);
    }
  }
}

void Scop::AddOneBufferDefInfo(const isl::id &ancestor_id,
                               const std::vector<std::pair<isl::id, MemType>> &data_stream) {
  if (data_stream.empty()) return;

  auto target = data_stream[0];
  isl::id tensor_id = target.first;
  MemType mem_type = target.second;
  isl::id dst_tensorId = isl::id(ctx_, TENSORLISTTAILNAME);
  MemType dst_mem_type = MemType::DDR;
  if (0 < data_stream.size() - 1) {
    dst_tensorId = data_stream[1].first;
    dst_mem_type = data_stream[1].second;
  }

  MemFlow mem_flow;
  for (const auto &item : data_stream) {
    mem_flow.push_back(item.second);
  }
  std::string mark_tag = TensorMarkTag(dst_mem_type, mem_flow);
  if (mark_tag.empty()) return;

  std::vector<size_t> sizes;
  BufferDefInfo promoted_info = BufferDefInfo{tensor_id,
                                              dst_tensorId,
                                              ancestor_id,
                                              mem_type,
                                              mark_tag,
                                              false,
                                              false,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              sizes,
                                              nullptr,
                                              isl::union_map::empty(CreateParamsSpace(ctx_))};
  MakeBufferFootprintCluster(promoted_info);
  buffer_def_infos_.push_back(promoted_info);
}

void Scop::AddTensorDataFlow(const std::vector<MemType> &memflow, const std::vector<std::string> &nameflow) {
  CHECK(memflow.size() == nameflow.size());
  uint64_t i = 0;
  /*********************************************
   *
   * init mem_type:        DDR
   * init tensor_id:       input0
   * init dst_tensorId:    input0_local_L1
   * init ancestor_id:     input0
   *
   * init mark_tag:        base on dst_tensorId mem_type, realize_L1
   * init data_stream:     input0 --> input0_local_L1 --> input0_fractal_L1 --> input0_fractal_L1_local_L0A
   **********************************************/
  std::string tensor_name = nameflow[i];
  MemType mem_type = memflow[i];

  isl::id tensor_id = isl::id(ctx_, tensor_name);
  isl::id ancestor_id = tensor_id;
  isl::id dst_tensorId = isl::id(ctx_, tensor_name);
  if (i < nameflow.size() - 1) {
    std::string dst_tensor_name = nameflow[i + 1];
    dst_tensorId = isl::id(ctx_, dst_tensor_name);
  }
  std::vector<std::pair<isl::id, MemType>> data_stream;

  for (size_t j = i; j < nameflow.size(); j++) {
    std::string tmp_name = nameflow[j];
    isl::id tmp_id = isl::id(ctx_, tmp_name);
    MemType tmp_mem_type = memflow[j];
    data_stream.emplace_back(std::make_pair(tmp_id, tmp_mem_type));
  }
  MemType dst_mem_type = MemType::DDR;
  if (data_stream.size() > 1) {
    dst_mem_type = data_stream[1].second;
  }
  std::string mark_tag = TensorMarkTag(dst_mem_type, memflow);
  if (IsIm2col() && mark_tag == REALIZE_L1) {
    mark_tag = REALIZE_UB;
  }

  bool isCopyin = IsCopyinTensor(tensor_id.get_name());
  if (!isCopyin && dst_mem_type == MemType::UBL1_) {
    mark_tag = REALIZE_L1UBL1;
  }

  std::vector<size_t> sizes;
  bool is_bind_tensor = true;
  BufferDefInfo promoted_info = BufferDefInfo{tensor_id,
                                              dst_tensorId,
                                              ancestor_id,
                                              mem_type,
                                              mark_tag,
                                              false,
                                              is_bind_tensor,
                                              data_stream,
                                              Tensor(),
                                              Handle(),
                                              sizes,
                                              nullptr,
                                              isl::union_map::empty(CreateParamsSpace(ctx_))};
  MakeBufferFootprintCluster(promoted_info);
  buffer_def_infos_.push_back(promoted_info);
}

void GatherVarNames(const Expr &expr, CondVarsMap &cond_vars, const isl::id &id) {
  std::unordered_set<Var, NodeHash, NodeEqual> vars_in_cond;
  GatherVars(expr, &vars_in_cond);
  for (const auto &var : vars_in_cond) {
    cond_vars[id].insert(var->name_hint);
  }
}

CondVarsMap Scop::ExtractCondVarsMap() const {
  CondVarsMap cond_vars;
  for (const auto &pair : data_.statements) {
    const isl::id &id = pair.first;
    const Node *stmt = pair.second;
    CHECK(stmt);
    if (stmt->IsInstance<IfThenElse>()) {
      const auto op = static_cast<const IfThenElse *>(stmt);
      GatherVarNames(op->condition, cond_vars, id);
    } else if (stmt->IsInstance<Provide>()) {
      const auto op = static_cast<const Provide *>(stmt);
      PostOrderVisit(op->value, [&id, &cond_vars](const NodeRef &node) -> void {
        if (auto op = node.as<Select>()) {
          GatherVarNames(op->condition, cond_vars, id);
        }
      });
    }
  }
  return cond_vars;
}

void Scop::AddPartitionInfoToData(const std::vector<std::vector<int>> &partition_info) {
  for (unsigned int i = 0; i < partition_info.size(); i++) {
    std::vector<Range> tmp;
    data_.range_info.push_back(tmp);
    for (unsigned int j = 1; j < partition_info[i].size(); j++) {
      data_.range_info[i].push_back(Range(Expr(partition_info[i][j - 1]), Expr(partition_info[i][j])));
    }
    if (partition_info[i].size() == 1) {
      data_.range_info[i].push_back(Range(Expr(0), Expr(0)));
    }
  }
}

void Scop::ComputeByPassL1() {
  if (bypassL1_ == 0) {
    int value = ExtractIntFromAttrs(ATTR_CONV_BYPASS_L1);
    if (value >= 0) {
      bypassL1_ = value;
    }
  }
  bypassL1_ = (IsFilterCanByPass()) ? bypassL1_ : 0;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
