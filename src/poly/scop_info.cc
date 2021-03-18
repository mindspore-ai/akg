/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include "scop_info.h"
#include <regex>
#include "poly/dma_inject.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {
constexpr int kInvalidIntAttr = -1;
Expr kInvalidExprAttr;

CubeInfo::~CubeInfo() {
  if (model_ != nullptr) {
    delete model_;
    model_ = nullptr;
  }
}
bool CubeInfo::IsConvBackpropInput() const {
  int n = ExtractIntFromAttrs(ATTR_CONV_BACKPROP_INPUT);
  return (IsConv() && (n != kInvalidIntAttr));
}

bool CubeInfo::IsConvBackpropFilter() const {
  int n = ExtractIntFromAttrs(ATTR_CONV_BACKPROP_FILTER);
  return (IsConv() && (n != kInvalidIntAttr));
}

Expr CubeInfo::ExtractExprFromAttrs(const std::string &name) const {
  for (auto i : analysis_result_.GetStmtOpInfoMap()) {
    if (!i.second.isMMU) {
      continue;
    }

    const Node *stmt_node = analysis_result_.GetStatementMap().at(i.first);
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

int CubeInfo::ExtractIntFromAttrs(const std::string &name) const {
  Expr expr_attr = ExtractExprFromAttrs(name);
  if (expr_attr.defined()) {
    if (const auto int_op = expr_attr.as<IntImm>())
      return int_op->value;
    else
      LOG(FATAL) << "attr " << name << " is not an integer";
  }
  return kInvalidIntAttr;
}

std::unordered_set<std::string> AnalysisResult::ExtractWithStmtId() const {
  std::unordered_set<std::string> res;
  for (auto i : GetStmtOpInfoMap()) {
    if (!i.second.isWith) {
      continue;
    }
    res.insert(i.first.get_name());
  }
  return res;
}

int UserConfig::GetDataType(const std::string &name) const {
  for (auto i : GetBind()) {
    if (i.first->op->name == name) {
      int size = i.first->dtype.bytes();
      return size;
    }
  }
  return 1;
}

std::string CubeInfo::ExtractStringFromAttrs(const std::string &name) const {
  for (auto i : analysis_result_.GetStmtOpInfoMap()) {
    if (!i.second.isMMU) {
      continue;
    }

    const Node *stmt_node = analysis_result_.GetStatementMap().at(i.first);
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

std::string CubeInfo::ExtractStringFromAttrsAndInfo(const std::string &name) const {
  for (auto i : analysis_result_.GetStmtOpInfoMap()) {
    if (!i.second.isMMU) {
      continue;
    }

    const Node *stmt_node = analysis_result_.GetStatementMap().at(i.first);
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

  if (GetConvAttrInfo().count(name) >= 1) {
    if (const auto str_op = GetConvAttrInfo().at(name).as<StringImm>()) {
      return str_op->value;
    } else {
      LOG(FATAL) << "attr " << name << " is not a string";
    }
  }

  return "";
}

bool ScopInfo::IsElewiseVMStmt(const isl::id &id) const {
  auto stmt = analysis_result_.GetStatementMap().at(id);
  if (stmt != nullptr && stmt->IsInstance<Provide>()) {
    auto provide = static_cast<const Provide *>(stmt);
    if (auto call = provide->value.as<Call>()) {
      if (call->call_type != Call::CallType::Halide && (call->name == "vmadd" || call->name == "vmla")) return true;
    }
  }
  return false;
}

bool ScopInfo::MayWriteAfterRead(const std::string &name) const {
  std::map<int, isl::id> def;
  std::map<int, isl::id> use;
  for (auto a : analysis_result_.GetWrites().get_map_list()) {
    isl::id id = a.domain().unwrap().domain().get_tuple_id();
    std::string idstr = id.get_name();
    if (a.get_tuple_id(isl_dim_out).get_name() != name) continue;
    CHECK_GE(idstr.size(), 2);
    idstr = idstr.substr(2, idstr.size());
    int ref = static_cast<int>(WrappedStrtol(idstr));
    def[ref] = id;
  }
  for (auto a : analysis_result_.GetReads().get_map_list()) {
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
      if (!IsElewiseVMStmt(i.second)) return true;
    }
  }
  return false;
}

bool CubeInfo::IsA(const std::string &name) const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      if (info.second.A_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool CubeInfo::IsB(const std::string &name) const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      if (info.second.B_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool CubeInfo::IsC(const std::string &name) const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      if (info.second.C_ == name) {
        return true;
      }
    }
  }
  return false;
}

bool CubeInfo::IsCUB(const std::string &name) const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      if (info.second.C_ + LOCAL_BUF == name) {
        return true;
      }
    }
  }
  return false;
}

std::string CubeInfo::GetAName() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      return info.second.A_;
    }
  }
  return "";
}

std::string CubeInfo::GetBName() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      return info.second.B_;
    }
  }
  return "";
}

std::string CubeInfo::GetCName() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) {
      return info.second.C_;
    }
  }
  return "";
}

bool CubeInfo::IsIm2col() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isIm2col) return true;
  }
  return false;
}

bool CubeInfo::IsLoadIm2colC1BUF() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.is_load_im2col) return true;
  }
  return false;
}

bool CubeInfo::IsLoadIm2colC1BUFStmt(const std::string &stmt_name) const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.is_load_im2col && info.first.name() == stmt_name) {
      return true;
    }
  }
  return false;
}

bool CubeInfo::HasCube() const {
  for (auto &info : analysis_result_.GetStmtOpInfoMap()) {
    if (info.second.isMMU) return true;
  }
  return false;
}

bool CubeInfo::IsGemmDataTransposeBlock() const {
  std::string trans_data_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE_BLOCK);
  return IsGemm() && !IsSpecGemm() && (trans_data_block == "Y");
}

bool CubeInfo::IsGemmWeightTransposeBlock() const {
  std::string trans_weight_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK);
  return IsGemm() && !IsSpecGemm() && (trans_weight_block == "Y");
}

bool CubeInfo::IsGemmDataTransposeInnerBlock() const {
  std::string trans_data_inner_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE_BLOCK_INNER);
  return IsGemm() && !IsSpecGemm() && (trans_data_inner_block == "Y");
}
bool CubeInfo::IsGemmWeightTransposeInnerBlock() const {
  std::string trans_weight_inner_block = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK_INNER);
  return IsGemm() && !IsSpecGemm() && (trans_weight_inner_block == "Y");
}
bool CubeInfo::IsGemmDataTranspose() const {
  std::string trans_data = ExtractStringFromAttrsAndInfo(ATTR_GEMM_DATA_TRANSPOSE);
  return IsGemm() && !IsSpecGemm() &&
         ((trans_data == "Y") || IsGemmDataTransposeBlock() || IsGemmDataTransposeInnerBlock());
}

bool CubeInfo::IsGemmWeightTranspose() const {
  std::string trans_weight = ExtractStringFromAttrsAndInfo(ATTR_GEMM_WEIGHT_TRANSPOSE);
  return IsGemm() && !IsSpecGemm() &&
         ((trans_weight == "Y") || IsGemmWeightTransposeBlock() || IsGemmWeightTransposeInnerBlock());
}

bool CubeInfo::IsGemm() const { return HasCube() && !IsConv(); }

bool CubeInfo::IsConv() const {
  std::string n = ExtractStringFromAttrs(ATTR_CONV_FEATURE_NAME);
  return (!n.empty());
}

void CubeInfo::UpdateComputeAttrInfo() {
  if (IsConv()) {
    FindComputeAttr(ConvATTRList);
  } else if (IsLoadIm2colC1BUF()) {
    FindComputeAttr(FastPoolingATTRList);
  }
}

void CubeInfo::FindComputeAttr(const std::vector<std::string> &op_keys) {
  for (auto i : analysis_result_.GetStmtOpInfoMap()) {
    if (i.second.isMMU || i.second.is_load_im2col) {
      const Node *stmt_node = analysis_result_.GetStatementMap().at(i.first);
      if (stmt_node->IsInstance<Provide>()) {
        auto provide = static_cast<const Provide *>(stmt_node);
        const auto cop = provide->func.as<ComputeOpNode>();
        if (cop != nullptr) {
          for (auto j : op_keys) {
            std::string err = "Error: You need to set attr feature " + j + " at akg.tvm.compute()!";
            CHECK(cop->attrs.count(j) != 0) << err;
          }
          SetConvAttrInfo(cop->attrs);
        }
      }
      break;
    }
  }
}

std::string CubeInfo::ConvOutName() {
  for (auto stmt : analysis_result_.GetStmtOpInfoMap()) {
    if (stmt.second.isMMU) {
      return stmt.second.C_;
    }
  }
  return "";
}

bool CubeInfo::IsFilterCanByPass() {
  bool can_bypass = true;
  auto filter_name = ExtractStringFromAttrs(ATTR_CONV_FILTER_NAME);
  auto tensor_mem_flows = analysis_result_.GetTensorMemFlows();
  if (tensor_mem_flows.count(filter_name)) {
    auto filter_memflow = tensor_mem_flows[filter_name];
    auto it = std::find(filter_memflow.begin(), filter_memflow.end(), BUF_C1_);
    if (it != filter_memflow.end()) can_bypass = false;
  }
  return can_bypass;
}

Tensor ScopInfo::FindTensorInOrig(const isl::id &var) {
  auto binds_orig = user_config_.GetOriginBind();
  for (auto i : binds_orig) {
    if (i.first->op->name == var.get_name()) {
      return i.first;
    }
  }
  return Tensor();
}

Tensor ScopInfo::FindTensorInOrig(const std::string &str) {
  auto binds_orig = user_config_.GetOriginBind();
  for (auto i : binds_orig) {
    if (i.first->op->name == str) {
      return i.first;
    }
  }
  return Tensor();
}

// find the dtype of global buffer by the tensor name
Type ScopInfo::GetDtypeOf(const std::string &tensor_name) const {
  auto binds = user_config_.GetBind();
  for (auto i : binds) {
    if (i.first->op->name == tensor_name) {
      return i.second->dtype;
    }
  }
  LOG(INFO) << " no such tensor in binds: " << tensor_name;
  return Int(32);
}

Type ScopInfo::GetDtypeOf(const isl::ast_expr &e) const {
  if (auto op = e.as<isl::ast_expr_op>()) {
    isl::id var = op.get_arg(0).as<isl::ast_expr_id>().get_id();
    return GetDtypeOf(var);
  }
  return Int(32);
}

bool ScopInfo::IsInBinds(const std::string &name) const {
  auto binds_orig = user_config_.GetOriginBind();
  for (auto i : binds_orig) {
    if (name == i.first->op->name) {
      return true;
    }
  }
  return false;
}

air::DataType CubeInfo::MadCastType() {
  for (auto stmt : analysis_result_.GetStmtOpInfoMap()) {
    if (stmt.second.isMMU) {
      return stmt.second.MadType_;
    }
  }
  return Float(16);
}

int CubeInfo::GetAttrValue(const std::string &key) {
  Map<std::string, NodeRef> attr_info = GetConvAttrInfo();
  CHECK(attr_info.find(key) != attr_info.end());
  if (attr_info[key].as<IntImm>() != nullptr) return attr_info[key].as<IntImm>()->value;
  if (attr_info[key].as<FloatImm>() != nullptr) {
    float res = attr_info[key].as<FloatImm>()->value;
    LOG(WARNING) << "attr: " << key << " : should be an integer, but found float. Force convert to int.";
    return static_cast<int>(res);
  }
  return -1;
}

Tensor ScopInfo::FindTensorWithLargestShape(const std::string &name) {
  size_t largest_size = 0;
  Tensor largest_tensor;
  for (auto i : analysis_result_.buffer_def_infos_) {
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
  auto binds = user_config_.GetBind();
  for (auto i : binds) {
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

Tensor ScopInfo::FindTensorWithLargestShape(const isl::id &var) { return FindTensorWithLargestShape(var.get_name()); }

Tensor ScopInfo::FindTensor(const std::string &str) {
  for (auto i : analysis_result_.buffer_def_infos_) {
    if (str == i.dst_tensor_id.get_name() && i.is_bind_tensor && i.tensor.defined()) {
      return i.tensor;
    }
  }
  auto binds = user_config_.GetBind();
  for (auto i : binds) {
    if (i.first->op->name == str) {
      return i.first;
    }
  }
  CHECK(false) << str << " is not declared in binds and promoted arrays";
  return Tensor();
}

Tensor ScopInfo::FindTensor(const isl::id &var) {
  for (const auto &i : analysis_result_.buffer_def_infos_) {
    if (i.dst_tensor_id.get_name() == var.get_name() && i.is_bind_tensor && i.tensor.defined()) {
      return i.tensor;
    }
  }
  auto binds = user_config_.GetBind();
  for (const auto &i : binds) {
    if (i.first->op->name == var.get_name()) {
      return i.first;
    }
  }

  CHECK(false) << var.to_str() << " is not declared in binds and promoted arrays";
  return Tensor();
}

isl::id ScopInfo::GetOriginTensorId(const std::string &name) const {
  std::string tensor_name = name;
  size_t pos = name.find("_local_");
  if (std::string::npos != pos) {
    tensor_name = name.substr(0, pos);
  }
  return isl::id(GetCtx(), tensor_name);
}

isl::id ScopInfo::GetOriginTensorId(const isl::id &id) const { return GetOriginTensorId(id.get_name()); }

bool CubeInfo::InitRangeStrideVec() {
  if (!GetRangeStride().empty()) return false;

  if (GetRangeInfo().empty()) {
    LOG(WARNING) << "range_info is not specified, please check";
    return false;
  }

  RecordRangeStrideBack(1);
  for (uint64_t i = GetRangeInfo().size(); i >= 1; --i) {
    RecordRangeStrideFront(GetRangeInfo()[i - 1].size() * (unsigned int)GetRangeStride()[0]);
  }
  return true;
}

std::vector<int> CubeInfo::GetIsolateVec(int range_idx) {
  static_cast<void>(InitRangeStrideVec());
  std::vector<int> idx;
  for (unsigned int i = 0; i < GetRangeStride().size() - 1; i++) {
    CHECK_NE(GetRangeStride()[i], 0);
    CHECK_NE(GetRangeStride()[i + 1], 0);
    idx.push_back(range_idx % GetRangeStride()[i] / GetRangeStride()[i + 1]);
  }
  return idx;
}

std::vector<Range> CubeInfo::GetRange(int range_idx) {
  std::vector<int> idx = GetIsolateVec(range_idx);
  std::vector<Range> res;
  CHECK(idx.size() == GetRangeInfo().size());
  for (unsigned int i = 0; i < idx.size(); i++) {
    res.push_back(GetRangeInfo()[i][(unsigned int)idx[i]]);
  }
  return res;
}

std::unordered_map<std::string, Expr> CubeInfo::GetConvInfoForTiling() {
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

void CubeInfo::SetConvMNKInfo() {
  TileSizes &dimInfos_conv = analysis_result_.GetTileSizes();
  TileSizes L1_factors;
  TileSizes L0_factors;

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
  analysis_result_.SetTileSizes(L1_factors);
  SetConvMNKDims(L0_factors);
  auto conv_mnk_dims = GetConvMNKDims();
  if (user_config_.GetIsDynamic()) {
    for (const auto &dim : conv_mnk_dims) {
      fractal_int_info_[dim.axis] = IntImm::make(Int(32), dim.c1_tiling_size);
      attr_info_.Set(dim.axis, IntImm::make(Int(32), dim.c1_tiling_size));
    }
  } else {
    const int c0_size = 16;
    const int int_imm_num_bits = 32;
    for (const auto &dim : conv_mnk_dims) {
      int l0tile = static_cast<int>(dim.c0_tiling_size);
      if (dim.axis == ATTR_CONV_TILE_M || dim.axis == ATTR_CONV_TILE_N || dim.axis == ATTR_CONV_TILE_K) {
        // multiply outer tile size with inner size
        l0tile *= c0_size;
      }
      fractal_int_info_[dim.axis] = l0tile;
      attr_info_.Set(dim.axis, IntImm::make(Int(int_imm_num_bits), l0tile));
    }
  }
}

void UserConfig::CollectParams() {
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
  auto binds_orig = GetOriginBind();
  for (auto x : binds_orig) {
    for (const auto &expr : x.second->shape) {
      if (!is_const(expr)) {
        RegisterParam(FloorDivToDiv(expr));
      }
    }
  }
  auto outer_let_stmts = GetOuterLetStmts();
  for (auto it : outer_let_stmts) {
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

void UserConfig::RegisterParam(const Expr &expr) {
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
  } else if (auto add = expr.as<air::ir::Add>()) {
    RegisterParam(add->a);
    RegisterParam(add->b);
    return;
  } else if (auto sub = expr.as<air::ir::Sub>()) {
    RegisterParam(sub->a);
    RegisterParam(sub->b);
    return;
  } else if (auto floodiv = expr.as<air::ir::FloorDiv>()) {
    RegisterParam(floodiv->a);
    RegisterParam(floodiv->b);
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

MappingCfg *UserConfig::GetThreadConfig() {
  bool enable_replace_cfg =
    (this->enable_one_dim_thread_ || this->vector_load_type_ || this->enable_tensor_core_use_poly_);
  if (!enable_replace_cfg) {
    return &thread_cfg_;
  }
  if (!this->GetReplaceConfig().count(COMPUTE)) {
    std::string new_cfg = "";
    for (size_t i = 0; i < this->thread_cfg_.bound; ++i) {
      int dim_size = this->thread_cfg_.GetAt(i).second;
      new_cfg += (std::to_string(dim_size) + " ");
    }
    this->SetThreadConfig(new_cfg);
  }
  return this->GetReplaceConfig()[COMPUTE];
}

void UserConfig::SetThreadConfig(const std::string &thread_cfg) {
  this->thread_cfg_.type = THREADS;
  if (this->enable_one_dim_thread_ || this->vector_load_type_ || this->enable_tensor_core_use_poly_) {
    std::vector<std::string> res = common::Split(thread_cfg, " ");
    int size = 1;
    for (size_t i = 0; i < res.size(); ++i) {
      CHECK(!res[i].empty());
      size *= std::stoi(res[i]);
    }
    this->thread_cfg_.BindFromStr(std::to_string(size));
    this->RecordReplaceConfig(COMPUTE, thread_cfg, MappingType::REPLACE_THREADS);
    return;
  }
  this->thread_cfg_.BindFromStr(thread_cfg);
}

void CubeInfo::CreateConvModel() {
  if (model_) return;
  if (!attr_info_.empty()) {
    if (attr_info_.count(ATTR_CONV_BACKPROP_INPUT) > 0) {
      try {
        model_ = new ConvolutionBackpropInputModel(attr_info_, user_config_.GetIsDynamic());
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionBackpropInputModel";
      }
    } else if (attr_info_.count(ATTR_CONV_BACKPROP_FILTER) > 0) {
      try {
        model_ = new ConvolutionBackpropFilterModel(attr_info_, user_config_.GetIsDynamic());
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionBackpropFilterModel";
      }
    } else {
      try {
        model_ = new ConvolutionForwardModel(attr_info_, user_config_.GetIsDynamic());
      } catch (const std::bad_alloc &) {
        LOG(FATAL) << "bad_alloc exception occurred when constructing ConvolutionForwardModel";
      }
    }
    if (model_) {
      static_cast<void>(model_->infer_CA1_tile());
    }
  }
}

void CubeInfo::UpdateFractalIntFirstInfo(bool is_conv_backprop_filter,
                                         const std::vector<size_t> &im2col_fp_cluster_size,
                                         const std::vector<size_t> &fractal_fp_cluster_size) {
  if (is_conv_backprop_filter) {
    UpdateFractalIntFirstInfoConvBackpropFilter(im2col_fp_cluster_size, fractal_fp_cluster_size);
  } else {
    UpdateFractalIntFirstInfoConvForward(im2col_fp_cluster_size, fractal_fp_cluster_size);
  }
}

void CubeInfo::UpdateFractalIntLastInfo(std::vector<size_t> filter_fp_cluster_size) {
  if (IsConvBackpropInput()) {
    CHECK_EQ(filter_fp_cluster_size.size(), 4);
    // conv_backprop_input filter: [ko, no, ni, ki]
    int64_t kh = ExtractIntFromAttrs(ATTR_CONV_KERNEL_H);
    int64_t kw = ExtractIntFromAttrs(ATTR_CONV_KERNEL_W);
    fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)filter_fp_cluster_size[0] / (kh * kw);
    fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)filter_fp_cluster_size[0] / (kh * kw);

    fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)filter_fp_cluster_size[2];
  } else if (IsConvBackpropFilter()) {
    CHECK_EQ(filter_fp_cluster_size.size(), 5);
    // conv_backprop_filter filter: [batch, no, mo, ni, mi]
    fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)filter_fp_cluster_size[3];
    fractal_int_info_[ATTR_CONV_GMM_M] = (int64_t)filter_fp_cluster_size[1] * filter_fp_cluster_size[3];
  } else {
    CHECK_EQ(filter_fp_cluster_size.size(), 4);
    // conv_forward filter: [ko, no, ni, ki]
    fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)filter_fp_cluster_size[1];
    fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)filter_fp_cluster_size[2];
  }
}

void CubeInfo::UpdateSpecGemmFractalInfo(const BufferDefInfo &tensor_info) {
  if (IsConv() && IsB(tensor_info.tensor_id.get_name())) {
    CHECK(tensor_info.footprints_cluster != nullptr);
    UpdateFractalIntLastInfo(tensor_info.footprints_cluster->GetFixedBoxSizes());
    fractal_str_info_[ATTR_CONV_GMM_WEIGHT] = tensor_info.dst_tensor_id.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  } else if (IsConv() && IsA(tensor_info.tensor_id.get_name())) {
    fractal_str_info_[ATTR_CONV_GMM_FEATURE] = tensor_info.data_stream[2].first.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  } else if (IsConv() && IsC(tensor_info.tensor_id.get_name())) {
    fractal_str_info_[ATTR_CONV_GMM_RES] = tensor_info.dst_tensor_id.get_name();
    CHECK_NE(tensor_info.dst_tensor_id.get_name(), "");
  }
}

void CubeInfo::UpdateFractalIntFirstInfoConvBackpropFilter(std::vector<size_t> im2col_fp_cluster_size,
                                                           std::vector<size_t> fractal_fp_cluster_size) {
  CHECK_EQ(fractal_fp_cluster_size.size(), 5);
  fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)fractal_fp_cluster_size[0];
  fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)fractal_fp_cluster_size[1];
  fractal_int_info_[ATTR_CONV_TILE_N] = (int64_t)fractal_fp_cluster_size[2];
  fractal_int_info_[ATTR_CONV_N_INNER] = (int64_t)fractal_fp_cluster_size[3];
  fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)fractal_fp_cluster_size[4];

  fractal_int_info_[ATTR_CONV_TILE_CO] = (int64_t)fractal_fp_cluster_size[2];

  CHECK_EQ(im2col_fp_cluster_size.size(), 6);
  fractal_int_info_[ATTR_CONV_GMM_K] = (int64_t)im2col_fp_cluster_size[1];
}

void CubeInfo::UpdateFractalIntFirstInfoConvForward(std::vector<size_t> im2col_fp_cluster_size,
                                                    std::vector<size_t> fractal_fp_cluster_size) {
  CHECK_EQ(fractal_fp_cluster_size.size(), 5);
  fractal_int_info_[ATTR_CONV_BATCH] = (int64_t)fractal_fp_cluster_size[0];
  fractal_int_info_[ATTR_CONV_TILE_M] = (int64_t)fractal_fp_cluster_size[1];
  fractal_int_info_[ATTR_CONV_TILE_K] = (int64_t)fractal_fp_cluster_size[2];
  fractal_int_info_[ATTR_CONV_M_INNER] = (int64_t)fractal_fp_cluster_size[3];
  fractal_int_info_[ATTR_CONV_K_INNER] = (int64_t)fractal_fp_cluster_size[4];

  CHECK_EQ(im2col_fp_cluster_size.size(), 6);
  fractal_int_info_[ATTR_CONV_GMM_M] = (int64_t)im2col_fp_cluster_size[1];
}

void CubeInfo::UpdateFractalIntInfoConvForward(int isolate_idx) {
  auto C0_SIZE = IntImm::make(Int(32), 16);
  fractal_int_info_[ATTR_CONV_TILE_N] = floordiv(model_->get_co_isolate_info(isolate_idx).inner, C0_SIZE);

  Expr m = model_->get_h_win_isolate_info(isolate_idx).inner * model_->get_w_win_isolate_info(isolate_idx).inner;
  fractal_int_info_[ATTR_CONV_GMM_M] = m;
  fractal_int_info_[ATTR_CONV_TILE_M] = floordiv(m + C0_SIZE - 1, C0_SIZE);
  fractal_int_info_[ATTR_CONV_M_INNER] = C0_SIZE;
  fractal_int_info_[ATTR_CONV_M_CUT_SIZE] = model_->get_w_win_isolate_info(isolate_idx).inner;
  if (!user_config_.GetIsDynamic()) {
    if (IsConvBackpropInput()) {
      CHECK(model_->conv_.filter.kh.as<IntImm>());
      CHECK(model_->conv_.filter.kw.as<IntImm>());
      user_config_.SetMatBDimH(model_->conv_.filter.kh.as<IntImm>()->value);
      user_config_.SetMatBDimW(model_->conv_.filter.kw.as<IntImm>()->value);
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

void CubeInfo::UpdateFractalIntInfoConvBackpropFilter(int isolate_idx) {
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

void CubeInfo::UpdateFractalIntInfo(int gemm_idx) {
  if (IsConvBackpropFilter()) {
    if (!user_config_.GetIsDynamic()) {
      UpdateFractalIntInfoConvBackpropFilter(gemm_idx);
    }
  } else {
    UpdateFractalIntInfoConvForward(gemm_idx);
  }
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

bool ScopInfo::IsWriteWholeBufferFootPrint(const isl::id &poly_ref_id) const {
  for (const auto &buffer : analysis_result_.active_buffer_footprints_) {
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
bool ScopInfo::IsConditionalWriteTensor(const std::string &name,
                                        const std::vector<std::pair<isl::id, isl::id>> &write_stmts) const {
  bool has_conditional_write = false;
  bool has_unconditional_full_write = false;
  for (const auto &pair : write_stmts) {
    auto stmt_id = pair.first;
    auto poly_ref_id = pair.second;
    CHECK_GT(analysis_result_.GetStatementMap().count(stmt_id), 0);
    const Node *stmt = analysis_result_.GetStatementMap().at(stmt_id);
    if (stmt->IsInstance<IfThenElse>()) {
      has_conditional_write = true;
    } else if (IsWriteWholeBufferFootPrint(poly_ref_id)) {
      has_unconditional_full_write = true;
    }
  }
  return has_conditional_write && !has_unconditional_full_write;
}

void ScopInfo::CollectConditionalWritePromotions() {
  std::unordered_map<std::string, std::vector<std::pair<isl::id, isl::id>>> tensor_write_stmts_map;
  analysis_result_.GetWrites().foreach_map([&tensor_write_stmts_map](const isl::map &map) -> void {
    std::string tensor_name = map.get_tuple_id(isl_dim_out).name();
    isl::id stmt_id = map.domain().unwrap().get_tuple_id(isl_dim_in);
    isl::id poly_ref_id = map.domain().unwrap().get_tuple_id(isl_dim_out);
    tensor_write_stmts_map[tensor_name].push_back(std::make_pair(stmt_id, poly_ref_id));
  });

  auto binds_orig = user_config_.GetOriginBind();
  for (auto bind : binds_orig) {
    auto name = bind.first->op->name;
    if (tensor_write_stmts_map.count(name) == 0) continue;
    if (IsConditionalWriteTensor(name, tensor_write_stmts_map[name])) {
      LOG(INFO) << "found conditionally written promoted tensor: " << name
                << ", buffer will be sinked to the computation.";
      analysis_result_.InsertConditionalWriteBufferFootprints(name);
    }
  }
}

StmtIdHashMap ScopInfo::StmtWriteMap() {
  StmtIdHashMap stmt_write_map;
  isl::union_map write_stmt = analysis_result_.GetWrites().domain_factor_domain();
  for (auto stmt : write_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto write_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_write_map[stmtId].push_back(write_tensor);
  }
  return stmt_write_map;
}

StmtIdHashMap ScopInfo::StmtReadMap() {
  StmtIdHashMap stmt_read_map;
  isl::union_map read_stmt = analysis_result_.GetReads().domain_factor_domain();
  for (auto stmt : read_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto read_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_read_map[stmtId].push_back(read_tensor);
  }
  return stmt_read_map;
}

StmtIdHashMap ScopInfo::StmtCopyinMap() {
  StmtIdHashMap stmt_copyin_map;
  isl::union_map copyin_stmt = analysis_result_.GetCopyin().domain_factor_domain();
  for (auto stmt : copyin_stmt.get_map_list()) {
    auto stmtId = stmt.domain().get_tuple_id();
    auto read_tensor = stmt.get_tuple_id(isl_dim_out);
    stmt_copyin_map[stmtId].push_back(read_tensor);
  }
  return stmt_copyin_map;
}

bool ScopInfo::IsCopyinTensor(const std::string &tensor_name) {
  CHECK_NE(tensor_name, "");
  StmtIdHashMap copyin_map = StmtCopyinMap();
  for (const auto &item : copyin_map) {
    for (const auto &tensor : item.second) {
      if (tensor.get_name() == tensor_name) return true;
    }
  }
  return false;
}

bool CubeInfo::IsConvHeadTail(const std::string &conv_output, const isl::id &stmtId, const StmtOpInfo &op_info,
                              const StmtIdHashMap &op_write_map) {
  if (!IsConv()) return false;

  if (op_info.isMMU || op_info.isMMUAssign) return false;

  if (op_info.ops.size() != 1) return false;

  if (op_write_map.find(stmtId) == op_write_map.end()) return false;

  if (op_write_map.at(stmtId).size() != 1) return false;

  if (op_info.ops[0] == PolyOpType::broadcast || op_info.ops[0] == PolyOpType::assignment) {
    isl::id writeId = op_write_map.at(stmtId)[0];
    if (writeId.get_name() == conv_output) return true;
  }

  return false;
}

void ScopInfo::CreateDataFlowInfo() {
  StmtIdHashMap op_write_map = StmtWriteMap();
  StmtIdHashMap op_read_map = StmtReadMap();
  std::string conv_output;
  if (mmu_info_.IsConv()) {
    conv_output = mmu_info_.ConvOutName();
  }
  uint64_t stmtNum = analysis_result_.GetStmtOpInfoMap().size();
  analysis_result_.stmt_type_.resize(stmtNum);
  DMADataFlow dma_dataflow;
  for (auto stmt : analysis_result_.GetStmtOpInfoMap()) {
    std::string name = stmt.first.get_name();
    size_t pos = name.find("_");
    CHECK(pos != name.size() - 1);
    std::string subNum = name.substr(pos + 1, name.size() - pos - 1);
    char *endptr = nullptr;
    const int radix = 10;
    size_t num = strtol(subNum.c_str(), &endptr, radix);
    if (endptr == nullptr || *endptr != '\0') LOG(FATAL) << "failed to convert string " << subNum << " to number";

    if (mmu_info_.IsConv() && mmu_info_.IsConvHeadTail(conv_output, stmt.first, stmt.second, op_write_map)) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::INST);
      continue;
    }

    if (stmt.second.isMMU && mmu_info_.IsConv()) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::MMU_CONV);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::MMU_CONV, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isMMU && !mmu_info_.IsConv()) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::MMU_GEMM);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::MMU_GEMM, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isIm2col || stmt.second.is_load_im2col) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::IM2COL_BUF);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::IM2COL_BUF, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (!stmt.second.isMMU && !stmt.second.isMMUAssign) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::INST);
      dma_dataflow.CreateStmtDataFlow(STMT_OP_TYPE::INST, stmt.first, stmt.second, op_read_map, op_write_map);
    }

    if (stmt.second.isMMUAssign) {
      analysis_result_.stmt_type_[num] = std::make_pair(stmt.first.get_name(), STMT_OP_TYPE::INST);
    }
  }
  dma_dataflow.FusionAnalysis();
  std::map<std::string, std::vector<std::string>> tensor_name_flows;
  std::map<std::string, MemFlow> tensor_mem_flows;
  dma_dataflow.OpDataflowInfo(tensor_name_flows, tensor_mem_flows);
  analysis_result_.SetTensorNameFlows(tensor_name_flows);
  analysis_result_.SetTensorMemFlows(tensor_mem_flows);
}

void ScopInfo::AddPartitionInfoToData(const std::vector<std::vector<int>> &partition_info) {
  for (unsigned int i = 0; i < partition_info.size(); i++) {
    std::vector<Range> tmp;
    for (unsigned int j = 1; j < partition_info[i].size(); j++) {
      mmu_info_.RecordRangeAt(i, Range(Expr(partition_info[i][j - 1]), Expr(partition_info[i][j])));
    }
    if (partition_info[i].size() == 1) {
      mmu_info_.RecordRangeAt(i, Range(Expr(0), Expr(0)));
    }
  }
}

void CubeInfo::ComputeByPassL1() {
  if (user_config_.GetByPathC1() == 0) {
    int value = ExtractIntFromAttrs(ATTR_CONV_BYPASS_L1);
    if (value >= 0) {
      user_config_.SetByPassL1(value);
    }
  }
  if (!IsFilterCanByPass()) {
    user_config_.SetByPassL1(0);
  }
}

void GatherVars(const Expr &expr, std::unordered_set<Var, air::NodeHash, air::NodeEqual> *vset) {
  PostOrderVisit(expr, [&vset](const NodeRef &node) {
    if (node.as<Variable>()) {
      vset->insert(Downcast<Var>(node));
    }
  });
}

void GatherVarNames(const Expr &expr, CondVarsMap &cond_vars, const isl::id &id) {
  std::unordered_set<Var, NodeHash, NodeEqual> vars_in_cond;
  GatherVars(expr, &vars_in_cond);
  for (const auto &var : vars_in_cond) {
    cond_vars[id].insert(var->name_hint);
  }
}

CondVarsMap AnalysisResult::GetCondVarsMap() {
  CondVarsMap cond_vars;
  for (const auto &pair : statements_) {
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

const BufferDefInfo &AnalysisResult::GetBufferDefInfo(const isl::id &tensor_id) const {
  for (const auto &idx : BufferDefInfos()) {
    if (idx.dst_tensor_id.get_name() == tensor_id.get_name()) {
      return idx;
    }
  }
  LOG(FATAL) << "Hoist footprint of tensor " << tensor_id << " has no buffer definition";
  return default_buffer_def_info_;
}

int AnalysisResult::CountBufferDefInfo(const isl::id &tensor_id) const {
  int num = 0;
  for (const auto &tensorIter : BufferDefInfos()) {
    if (tensorIter.dst_tensor_id.get_name() == tensor_id.get_name()) {
      num++;
    }
  }
  return num;
}

bool AnalysisResult::HasBufferDefInfo(const isl::id &tensor_id) const {
  for (const auto &idx : BufferDefInfos()) {
    if (idx.dst_tensor_id.get_name() == tensor_id.get_name()) {
      return true;
    }
  }
  return false;
}

bool AnalysisResult::IsPureReduceSum(const Add *add, const std::string &prov_func_name) {
  Expr dst, src;
  bool collect_lhs = true;
  auto FoundDst = [&add, &dst, &src, &collect_lhs, &prov_func_name](const NodeRef &node) {
    if (dst.defined() || src.defined()) {
      return;
    }
    const auto call = node.as<Call>();
    if (call == nullptr || call->func->func_name() != prov_func_name) {
      return;
    }
    if (collect_lhs) {
      dst = add->a;
      src = add->b;
    } else {
      dst = add->b;
      src = add->a;
    }
  };
  air::ir::PostOrderVisit(add->a, FoundDst);
  collect_lhs = false;
  air::ir::PostOrderVisit(add->b, FoundDst);

  if (!dst.defined() || !src.defined()) {
    return false;
  }

  std::vector<Expr> dst_indice;
  std::vector<Expr> src_indice;
  std::vector<std::vector<Expr>> src_list;
  bool collect_dst = true;
  auto IsNum = [](std::string name) -> bool {
    for (auto c : name)
      if (c > '9' || c < '0') return false;
    return true;
  };
  auto CollectIndex = [&dst_indice, &src_indice, &src_list, &collect_dst, &IsNum](const NodeRef &node) {
    const auto call = node.as<Call>();
    if (call == nullptr || POLY_SUPPORTED_OPS.count(call->name)) {
      return;
    }
    std::vector<Expr> target_vec;

    for (const auto arg : call->args) {
      bool is_num = arg.as<IntImm>() != nullptr;
      if (arg.as<Variable>()) {
        is_num = IsNum(arg.as<Variable>()->name_hint);
      }
      if (is_num) {
        continue;
      }
      target_vec.emplace_back(arg);
    }
    if (collect_dst) {
      dst_indice = target_vec;
    } else {
      src_indice.insert(src_indice.end(), target_vec.begin(), target_vec.end());
      src_list.emplace_back(target_vec);
    }
  };
  air::ir::PostOrderVisit(dst, CollectIndex);
  collect_dst = false;
  air::ir::PostOrderVisit(src, CollectIndex);

  auto ExprDiff = [](std::vector<Expr> a, std::vector<Expr> b) -> std::vector<Expr> {
    std::vector<Expr> res;
    for (auto e1 : a) {
      bool is_dup = false;
      auto c = b.empty() ? res : b;
      for (auto e2 : c) {
        if (!Equal(e2, e1)) {
          continue;
        }
        is_dup = true;
        break;
      }
      if (!is_dup) {
        res.emplace_back(e1);
      }
    }
    return res;
  };
  auto unique_dst_indice = ExprDiff(dst_indice, {});
  auto unique_src_indice = ExprDiff(src_indice, {});
  auto reduce_indice = ExprDiff(unique_src_indice, unique_dst_indice);
  auto unique_red_indice = ExprDiff(reduce_indice, {});

  // 1. check unique_dst.index + unique_reduce.index = unique_src.index
  if (unique_dst_indice.size() + unique_red_indice.size() != unique_src_indice.size()) {
    return false;
  }

  // 2. check each src size is equal (only allow broadcast)
  //    e.g.1 out[i, j] = src1[i, j, k] * src2[i]  -> is pure elem
  //    e.g.2 out[i, j] = src1[i, k] * src2[j, k]  -> not pure elem
  for (auto i = 0; i < static_cast<int>(src_list.size()) - 1; ++i) {
    auto only_cur = ExprDiff(src_list[i], src_list[i + 1]);
    auto only_next = ExprDiff(src_list[i + 1], src_list[i]);
    int diff_set_size = only_cur.size() + only_next.size();
    auto diff_size = std::abs(static_cast<int>(src_list[i].size()) - static_cast<int>(src_list[i + 1].size()));
    if (diff_set_size != diff_size) {
      return false;
    }
  }
  return true;
}

bool AnalysisResult::IsReduceInitStmt(const isl::id id) const {
  for (const auto &init_id : GetReduceInitIds()) {
    if (init_id.get_name() == id.get_name()) {
      return true;
    }
  }
  return false;
}

std::string AnalysisResult::GetReduceOpType(isl::id reduce_stmt) {
  auto red_map = GetReduceTensorInfoMap();
  auto it = red_map.find(reduce_stmt);
  if (it == red_map.end()) {
    return std::string();
  }
  auto provide = static_cast<const Provide *>(it->second.stmt_node);
  if (provide == nullptr) {
    return std::string();
  }
  if (provide->value.as<Max>()) {
    return AKG_REDUCE_MAX;
  }
  if (provide->value.as<Min>()) {
    return AKG_REDUCE_MIN;
  }
  if (provide->value.as<And>()) {
    return AKG_REDUCE_AND;
  }
  if (provide->value.as<Or>()) {
    return AKG_REDUCE_OR;
  }
  if (const auto add = provide->value.as<Add>()) {
    return IsPureReduceSum(add, provide->func->func_name()) ? AKG_REDUCE_SUM : AKG_REDUCE_UNSUPPORTED;
  }
  return std::string();
}

void AnalysisResult::MarkReduceOutTensor(const isl::schedule_node_band &band) {
  auto target_stmt = GetReduceWriteStmt(band);
  auto tensor = target_stmt.range();
  tensor.foreach_set([this](const isl::set &s) -> void { RecordReduceOutTensors(s.get_tuple_name()); });
}

isl::union_map AnalysisResult::GetReduceWriteStmt(const isl::schedule_node_band &band) {
  auto band_domain = band.get_domain();
  auto write_domain = GetWrites().domain_factor_domain();
  return write_domain.intersect_domain(band_domain);
}

static std::string MemTypeToString(const MemType &memType) {
  switch (memType) {
    case MemType::BUF_:
      return "UB";
    case MemType::C1_:
      return "L1";
    case MemType::BUF_C0_:
      return "UBL0";
    case MemType::BUF_C1_:
      return "UBL1";
    case MemType::C0A_:
      return "L0A";
    case MemType::C0B_:
      return "L0B";
    case MemType::C0C_:
      return "L0C";
    case MemType::DDR:
      return "GM";
    case MemType::SHARED_:
      return "SHARED";
    case MemType::LOCAL_:
      return "LOCAL";
    default:
      return "";
  }
}

std::string ScopInfo::GetIslReadName(const isl::id &cluster_id) {
  auto tensor_info = analysis_result_.GetBufferDefInfo(cluster_id);
  MemType memType = tensor_info.SrcMemType();
  return MemTypeToString(memType) + "read";
}

std::string ScopInfo::GetIslWriteName(const isl::id &cluster_id) {
  if (analysis_result_.HasBufferDefInfo(cluster_id)) {
    auto tensor_info = analysis_result_.GetBufferDefInfo(cluster_id);
    MemType memType = tensor_info.DstMemType();
    return MemTypeToString(memType) + "write";
  }
  return MemTypeToString(MemType::DDR) + "write";
}

std::string TensorMarkTag(MemType mem_type, MemFlow mem_flow) {
  /******************************
   *  This interface is used to convert tensor MemType to isl schedule tree mark_tag,
   *  used to record the extension position for each tensor in isl schedule tree.
   *  Now REALIZE_C1/REALIZE_C0/REALIZE_BUF mark_tag is equal to its MemType.
   *  For mem_type is DDR, mark_tag is empty string "".
   * */
  switch (mem_type) {
    case MemType::C1_:
      if (mem_flow.size() == 3 && mem_flow[0] == MemType::DDR && mem_flow[1] == MemType::C1_ &&
          mem_flow[2] == MemType::BUF_C1_)
        return REALIZE_C1BUFC1;
      return REALIZE_C1;
    case MemType::BUF_:
      // ordinary conv condition no fusion
      if (mem_flow.size() == 3 && mem_flow[0] == MemType::DDR && mem_flow[1] == mem_type &&
          mem_flow[2] == MemType::C0C_)
        return REALIZE_C0;
      return REALIZE_BUF;
    case MemType::C0A_:
      return REALIZE_C0;
    case MemType::C0B_:
      return REALIZE_C0;
    case MemType::C0C_:
      return REALIZE_C0;
    case MemType::BUF_C0_:
      return REALIZE_BUFC0;
    case MemType::BUF_C1_:
      if (mem_flow.size() == 2 && mem_flow[0] == MemType::DDR && mem_flow[1] == MemType::BUF_C1_) return REALIZE_C1;
      return REALIZE_BUFC1;
    case MemType::DDR:
      return "";
    default:
      LOG(FATAL) << "undefined mem_type " << mem_type;
      return "";
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
