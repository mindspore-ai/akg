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

#include <src/pass/ir_util.h>
#include <pass/expr_alg_simplify.h>

#include <utility>
#include "scop.h"

namespace akg {
namespace ir {
namespace poly {
class RestoreConstToMinMutator : public IRMutator {
 public:
  RestoreConstToMinMutator() = default;
  ~RestoreConstToMinMutator() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!op->extent.as<IntImm>()) {
      Expr extent = ExprSimplifier().RetroConstToMin(op->extent);
      if (!extent.same_as(op->extent)) {
        LOG(INFO) << "origin extent: " << op->extent << ", new extent: " << extent;
        Stmt body = Mutate(op->body);
        return For::make(op->loop_var, op->min, extent, op->for_type, op->device_api, body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
};

class RestoreCombinedParamsMutator : public IRMutator {
 public:
  explicit RestoreCombinedParamsMutator(const std::unordered_map<std::string, Expr> &params) : params(params) {}
  ~RestoreCombinedParamsMutator() override = default;

 private:
  Expr Mutate_(const Variable *op, const Expr &e) override {
    auto it = params.find(op->name_hint);
    if (it != params.end()) return it->second;
    return e;
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    auto stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    CHECK(op);
    if (op->attr_key == "pragma_gemm_l0") {
      Map<std::string, Range> new_range_map;
      auto range_map = Downcast<Map<std::string, Range>>(op->node);
      for (auto kv : range_map) {
        Range r = kv.second;
        new_range_map.Set(kv.first, Range(r->min, this->Mutate(r->extent)));
      }
      return AttrStmt::make(new_range_map, op->attr_key, op->value, op->body);
    }
    if (op->attr_key == "pragma_attrs") {
      Map<std::string, NodeRef> new_attrs;
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      for (auto kv : attrs) {
        Expr v = Downcast<Expr>(kv.second);
        new_attrs.Set(kv.first, this->Mutate(v));
      }
      return AttrStmt::make(new_attrs, op->attr_key, op->value, op->body);
    }
    return stmt;
  }
  const std::unordered_map<std::string, Expr> &params;
};

class ParameterizingTiling : public IRMutator {
 public:
  explicit ParameterizingTiling(std::map<int64_t, Expr> ptmap) : ptm(std::move(ptmap)) {}
  ~ParameterizingTiling() override = default;
  std::map<int64_t, Expr> ptm;
  Expr Mutate_(const IntImm *op, const Expr &e) final {
    Expr ret = IntImm::make(op->type, op->value);
    auto it = ptm.find(op->value);
    if (it != ptm.end()) {
      if (op->type != it->second.type())
        ret = Cast::make(op->type, it->second);
      else
        ret = it->second;
    }
    return ret;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    auto stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    CHECK(op);
    if (op->attr_key == "pragma_gemm_l0") {
      Map<std::string, Range> new_range_map;
      auto range_map = Downcast<Map<std::string, Range>>(op->node);
      for (auto kv : range_map) {
        Range r = kv.second;
        new_range_map.Set(kv.first, Range(r->min, this->Mutate(r->extent)));
      }
      return AttrStmt::make(new_range_map, op->attr_key, op->value, op->body);
    }
    if (op->attr_key == "pragma_attrs") {
      Map<std::string, NodeRef> new_attrs;
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      for (auto kv : attrs) {
        Expr v = Downcast<Expr>(kv.second);
        new_attrs.Set(kv.first, this->Mutate(v));
      }
      return AttrStmt::make(new_attrs, op->attr_key, op->value, op->body);
    }
    return stmt;
  }
};

void Scop::Full2PartialDynamic(std::unordered_map<std::string, Expr> &params_map,
                               const Map<std::string, NodeRef> &attr_info) {
  int64_t kh = default_kernel_h;
  int64_t kw = default_kernel_w;
  int64_t pt = 1;
  int64_t pb = 1;
  int64_t pl = 1;
  int64_t pr = 1;
  int64_t sh = 1;
  int64_t sw = 1;
  int64_t tile_co = 0;
  int64_t tile_ho = 0;
  int64_t tile_wo = 0;
  int64_t tile_mo = 0;
  int64_t tile_no = 0;
  int64_t tile_ko = 0;
  auto it = attr_info.find("pragma_conv_real_kh");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) kh = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_kw");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) kw = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_pt");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) pt = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_pb");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) pb = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_pl");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) pl = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_pr");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) pr = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_sh");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) sh = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_real_sw");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) sw = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_co");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_co = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_ho");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_ho = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_wo");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_wo = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_mo");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_mo = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_no");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_no = (*it).second.as<IntImm>()->value;
  it = attr_info.find("pragma_conv_tile_ko");
  if ((it != attr_info.end()) && (*it).second.as<IntImm>()) tile_ko = (*it).second.as<IntImm>()->value;
  params_map.insert(std::make_pair("KH", Expr((int32_t)kh)));
  params_map.insert(std::make_pair("KW", Expr((int32_t)kw)));
  params_map.insert(std::make_pair("PT", Expr((int32_t)pt)));
  params_map.insert(std::make_pair("PB", Expr((int32_t)pb)));
  params_map.insert(std::make_pair("PL", Expr((int32_t)pl)));
  params_map.insert(std::make_pair("PR", Expr((int32_t)pr)));
  params_map.insert(std::make_pair("SH", Expr((int32_t)sh)));
  params_map.insert(std::make_pair("SW", Expr((int32_t)sw)));
  if (tile_co > 0) params_map.insert(std::make_pair("T1_0_C1", Expr((int32_t)tile_co)));
  if (tile_ho > 0) params_map.insert(std::make_pair("T1_0_H", Expr((int32_t)tile_ho)));
  if (tile_wo > 0) params_map.insert(std::make_pair("T1_0_W", Expr((int32_t)tile_wo)));
  if (tile_mo > 0) params_map.insert(std::make_pair("T0_0_MO", Expr((int32_t)tile_mo)));
  if (tile_no > 0) params_map.insert(std::make_pair("T0_0_NO", Expr((int32_t)tile_no)));
  if (tile_ko > 0) params_map.insert(std::make_pair("T0_0_KO", Expr((int32_t)tile_ko)));
}

Stmt Scop::RestoreCombinedParams(Stmt stmt) {
  stmt = RestoreCombinedParamsMutator(params_rev_map_).Mutate(stmt);
  if (IsConv() && !is_spec_gemm_) {
    stmt = RestoreConstToMinMutator().Mutate(stmt);
  }
  stmt = ReplacePrimesWithParameters(stmt);
  if (tile_size_is_var_) {
    if (is_spec_gemm_) {
      std::unordered_map<std::string, Expr> params_map;
      params_map.insert(std::make_pair(
        "MO", floordiv(min(Var("T1_0_H"), floordiv(Var("H") + Var("PT") + Var("PB") - Var("KH"), Var("SH")) -
                                            Var("cc2") * Var("T1_0_H") + 1) *
                           min(Var("T1_0_W"), floordiv(Var("W") + Var("PL") + Var("PR") - Var("KW"), Var("SW")) -
                                                Var("cc3") * Var("T1_0_W") + 1) +
                         15,
                       Expr(16))));
      params_map.insert(std::make_pair("KO", Expr(128 * 11 * 31)));
      params_map.insert(std::make_pair("NO", min(Var("T1_0_C1"), (Var("CO1") - Var("cc1") * Var("T1_0_C1")))));
      stmt = RestoreCombinedParamsMutator(params_map).Mutate(stmt);
    } else if (!dynamic_shape_conv_full_parametric_) {
      std::unordered_map<std::string, Expr> params_map;
      Full2PartialDynamic(params_map, attr_info_);
      stmt = RestoreCombinedParamsMutator(params_map).Mutate(stmt);
    }
  }
  stmt = AddTilingStrategyApplet(stmt);
  stmt = air::ir::MergeNest(outer_let_stmts_, stmt);
  return stmt;
}

Stmt Scop::AddTilingStrategyApplet(Stmt stmt) {
  for (auto info = tiling_constraints_.rbegin(); info != tiling_constraints_.rend(); ++info) {
    if (info->type_key == "AttrStmt") {
      auto attr_key = info->key.as<StringImm>();
      CHECK(attr_key);
      stmt = AttrStmt::make(make_zero(Int(32)), attr_key->value, info->value, stmt);
    } else if (info->type_key == "LetStmt") {
      stmt = LetStmt::make(air::Downcast<Var>(info->key), info->value, stmt);
    } else {
      LOG(FATAL) << "Unsupported type_key for now: " << info->type_key;
    }
  }
  return stmt;
}

void Scop::InsertPairsSpecGemmTileVar(std::map<int64_t, Expr> &param_map) {
  const int t0_mo = PRIME_1;
  const int t0_ko = PRIME_2;
  const int t0_no = PRIME_3;
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_mo), Var("T0_0_MO")));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_ko), Var("T0_0_KO")));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_no), Var("T0_0_NO")));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_mo * 16), Var("T0_0_MO") * 16));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_ko * 16), Var("T0_0_KO") * 16));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_no * 16), Var("T0_0_NO") * 16));
}

void Scop::InsertPairsConvTileVar(Stmt &stmt, std::map<int64_t, Expr> &param_map) {
  // tile size
  const int t1_c1 = GetAttrValue(ATTR_CONV_TILE_CO) / 16;
  const int h_cut = GetAttrValue(ATTR_CONV_TILE_H);
  const int w_cut = GetAttrValue(ATTR_CONV_TILE_W);

  const int t0_m = GetAttrValue(ATTR_CONV_TILE_M);
  const int t0_k = GetAttrValue(ATTR_CONV_TILE_K);
  const int t0_n = GetAttrValue(ATTR_CONV_TILE_N);

  const int ci1 = 128;

  // kernel
  const int k_h = GetAttrValue(ATTR_CONV_KERNEL_H);
  const int k_w = GetAttrValue(ATTR_CONV_KERNEL_W);

  // pad
  const int p_t = GetAttrValue(ATTR_CONV_PAD_TOP);
  const int p_b = GetAttrValue(ATTR_CONV_PAD_BOTTOM);
  const int p_l = GetAttrValue(ATTR_CONV_PAD_LEFT);
  const int p_r = GetAttrValue(ATTR_CONV_PAD_RIGHT);

  // stride
  const int s_h = GetAttrValue(ATTR_CONV_STRIDE_H);
  const int s_w = GetAttrValue(ATTR_CONV_STRIDE_W);

  const int t1_h = (h_cut - k_h) / s_h + 1;
  const int t1_w = (w_cut - k_w) / s_w + 1;

  auto CI1 = Var("CI1");
  auto T1_0_C1 = Var("T1_0_C1");
  auto T1_0_H = Var("T1_0_H");
  auto T1_0_W = Var("T1_0_W");
  auto T0_0_MO = Var("T0_0_MO");
  auto T0_0_NO = Var("T0_0_NO");
  auto T0_0_KO = Var("T0_0_KO");
  auto KH = Var("KH");
  auto KW = Var("KW");
  auto SH = Var("SH");
  auto SW = Var("SW");
  auto PT = Var("PT");
  auto PB = Var("PB");
  auto PL = Var("PL");
  auto PR = Var("PR");

  if (dynamic_shape_conv_full_parametric_) {
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L1]", T1_0_C1 <= 4, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L1]", T1_0_H <= 18, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L1]", T1_0_W <= 1, stmt);

    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L0A]", T0_0_MO <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L0B]", T0_0_NO <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_L0C]", T0_0_KO <= 1, stmt);

    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", KH <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", KW <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", SH <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", SW <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", PT <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", PB <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", PL <= 1, stmt);
    stmt = AttrStmt::make(make_zero(Int(32)), "[MemoryLimit_UB]", PR <= 1, stmt);
  }

  // c1
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_c1), T1_0_C1));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_c1 * 16), T1_0_C1 * 16));

  // h
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h), T1_0_H));
  param_map.insert(std::make_pair(static_cast<int64_t>(-t1_h), T1_0_H * (-1)));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h - 1), T1_0_H - 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h + 1), T1_0_H + 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h + 2), T1_0_H + 2));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h * 16), T1_0_H * 16));

  // w
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w), T1_0_W));
  param_map.insert(std::make_pair(static_cast<int64_t>(-t1_w), T1_0_W * (-1)));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w - 1), T1_0_W - 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w + 1), T1_0_W + 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w + 2), T1_0_W + 2));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w * 16), T1_0_W * 16));

  // h & w
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h * t1_w), T1_0_H * T1_0_W));
  param_map.insert(std::make_pair(static_cast<int64_t>((t1_h * t1_w + 15) / 16), floordiv(T1_0_H * T1_0_W + 15, 16)));

  // kc1
  param_map.insert(std::make_pair(static_cast<int64_t>(ci1), CI1));

  param_map.insert(std::make_pair(static_cast<int64_t>(t0_m), T0_0_MO * 16));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_n), T0_0_NO * 16));
  param_map.insert(std::make_pair(static_cast<int64_t>(t0_k), T0_0_KO * 16));

  /*==================== for all parameters ===================*/
  param_map.insert(std::make_pair(static_cast<int64_t>(s_h), SH));
  param_map.insert(std::make_pair(static_cast<int64_t>(s_w), SW));

  param_map.insert(std::make_pair(static_cast<int64_t>(p_t), PT));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_t - 1), PT - 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_b), PB));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_l), PL));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_l - 1), PL - 1));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_r), PR));

  param_map.insert(std::make_pair(static_cast<int64_t>(k_h), KH));
  param_map.insert(std::make_pair(static_cast<int64_t>(k_w), KW));
  param_map.insert(std::make_pair(static_cast<int64_t>(k_h * k_w), KH * KW));
  param_map.insert(std::make_pair(static_cast<int64_t>(ci1 * k_h * k_w), CI1 * KH * KW));

  param_map.insert(std::make_pair(static_cast<int64_t>(t1_h * s_h), T1_0_H * SH));
  param_map.insert(std::make_pair(static_cast<int64_t>(-t1_h * s_h), T1_0_H * SH * (-1)));
  param_map.insert(std::make_pair(static_cast<int64_t>(t1_w * s_w), T1_0_W * SW));
  param_map.insert(std::make_pair(static_cast<int64_t>(-t1_w * s_w), T1_0_W * SW * (-1)));

  param_map.insert(std::make_pair(static_cast<int64_t>(p_t + p_b - k_h), PT + PB - KH));
  param_map.insert(std::make_pair(static_cast<int64_t>(p_l + p_r - k_w), PL + PR - KW));

  param_map.insert(std::make_pair(static_cast<int64_t>((t1_h - 1) * s_h + k_h), (T1_0_H - 1) * SH + KH));
  param_map.insert(std::make_pair(static_cast<int64_t>((t1_h - 1) * s_h + k_h - 1), (T1_0_H - 1) * SH + KH - 1));
  param_map.insert(std::make_pair(static_cast<int64_t>((t1_w - 1) * s_w + k_w), (T1_0_W - 1) * SW + KW));
  param_map.insert(std::make_pair(static_cast<int64_t>((t1_w - 1) * s_w + k_w - 1), (T1_0_W - 1) * SW + KW - 1));
}

void Scop::InsertRange(std::map<int64_t, Expr> &param_map, const std::pair<int64_t, Expr> &item) {
  param_map.insert(std::make_pair(-item.first, CanonicalSimplify(-1 * item.second)));
  param_map.insert(std::make_pair(item.first, CanonicalSimplify(item.second)));
  param_map.insert(std::make_pair(item.first - 1, CanonicalSimplify(item.second - 1)));
}

void Scop::InsertPairsSpecGemmOrConv(Stmt &stmt, std::map<int64_t, Expr> &param_map) {
  for (const auto &dims : dim_infos_) {
    if (dims.l1_var.defined()) {
      InsertRange(param_map, std::make_pair(dims.l1_tiling_size, dims.l1_var));
      param_map.insert(std::make_pair(dims.l1_tiling_size * 16, dims.l1_var * 16));
      if (dims.l0_var.defined()) {
        param_map.insert(std::make_pair(dims.l0_tiling_size, dims.l0_var));
      }
    }
  }
  int64_t m_size = 1;
  Expr m_size_expr = 1;
  int64_t t0_mo = 1;
  Expr t0_mo_expr = 1;
  for (const auto &dims : conv_mnk_dims_) {
    if (dims.axis == ATTR_CONV_TILE_M || dims.axis == ATTR_CONV_TILE_N || dims.axis == ATTR_CONV_TILE_K) {
      if (dims.l0_var.defined()) {
        if (!dims.l0_var.as<IntImm>()) {
          if (dims.axis == ATTR_CONV_TILE_M) {
            t0_mo = dims.l0_tiling_size;
            t0_mo_expr = dims.l0_var;
          }
          InsertRange(param_map, std::make_pair(dims.l0_tiling_size, dims.l0_var));
          param_map.insert(std::make_pair(dims.l0_tiling_size * 16, dims.l0_var * 16));
        }
      }
      if (dims.l1_var.defined()) {
        if (!dims.l1_var.as<IntImm>()) {
          InsertRange(param_map, std::make_pair(dims.l1_tiling_size, dims.l1_var));
        }
      }
    } else if (dims.axis == ATTR_CONV_TILE_H || dims.axis == ATTR_CONV_TILE_W) {
      CHECK(dims.l1_var.defined()) << dims.axis << "'s var not defined";
      m_size *= dims.l1_tiling_size;
      m_size_expr *= dims.l1_var;
      param_map.insert(std::make_pair(dims.l1_tiling_size, dims.l1_var));
    }
  }
  if (m_size != 1) {
    int64_t m_larger_size = ((m_size + 15) / 16) * 16;
    m_size_expr = CanonicalSimplify(m_size_expr);
    Expr m_larger_expr = CanonicalSimplify(floordiv((m_size_expr + 15), 16) * 16);
    int64_t tile_m_size = (m_size + 15) / 16;
    Expr tile_m = floordiv((m_size_expr + 15), 16);
    int64_t tile_m_size_1 = tile_m_size - 1;
    Expr tile_m_1 = CanonicalSimplify(tile_m - 1);  // (th*tw + 15) / 16 - 1
    int64_t tile_m_size_1_t_mo_1 = tile_m_size_1 / t0_mo + 1;
    Expr tile_m_1_t_mo_1 = CanonicalSimplify(floordiv(tile_m_1, t0_mo_expr) + 1);
    LOG(INFO) << tile_m_size_1_t_mo_1;
    LOG(INFO) << tile_m_1_t_mo_1;
    param_map.insert(std::make_pair(tile_m_size, tile_m));
    param_map.insert(std::make_pair(tile_m_size_1, tile_m_1));
    param_map.insert(std::make_pair(tile_m_size_1_t_mo_1, tile_m_1_t_mo_1));
    param_map.insert(std::make_pair(m_size, m_size_expr));
    param_map.insert(std::make_pair(m_larger_size, m_larger_expr));
  }
}

void Scop::InsertPairs(Stmt &stmt, std::map<int64_t, Expr> &param_map) {
  for (const auto &dims : dim_infos_) {
    if (dims.l1_var.defined()) {
      if (dims.pragma.defined()) {
        // pragma defined is used for special axes like shift axis
        int64_t tile = dims.l0_tiling_size;
        const auto shift_imm = dims.pragma.as<IntImm>();
        CHECK(shift_imm);
        int shift = shift_imm->value - 1;
        int64_t shifted_bound = dims.l1_tiling_size;
        int64_t bound = shifted_bound + shift - 1;
        Expr bound_param = dims.l1_var;
        Expr shifted_bound_param = dims.l1_var + 1 - shift;
        for (auto i = 1; i <= shift; ++i) {
          InsertRange(param_map, std::make_pair(shifted_bound * i, shifted_bound_param * i));
          InsertRange(param_map, std::make_pair(bound * i, bound_param * i));
          InsertRange(param_map, std::make_pair(bound / tile * i, floordiv(dims.l1_var, dims.l0_var) * i));
          param_map.insert(std::make_pair((bound - 1) * i, dims.l1_var * i));
          param_map.insert(
            std::make_pair((shifted_bound - tile) * i, (dims.l1_var - Expr(shift - 1) - dims.l0_var) * i));
        }
      } else {
        InsertRange(param_map, std::make_pair(dims.l1_tiling_size, dims.l1_var));
        InsertRange(param_map, std::make_pair(dims.l1_tiling_size * 2, dims.l1_var * 2));
        InsertRange(param_map, std::make_pair(dims.l1_tiling_size * (-2), dims.l1_var * (-2)));
        param_map.insert(std::make_pair(dims.l1_tiling_size * 2 + 1, dims.l1_var * 2 + 1));
        if (dims.l0_var.defined()) {
          // normal cube axis's L0 tile
          int64_t floor = dims.l1_tiling_size / dims.l0_tiling_size;
          int64_t mod = dims.l1_tiling_size - floor * dims.l0_tiling_size;
          InsertRange(param_map, std::make_pair(dims.l0_tiling_size, dims.l0_var));
          param_map.insert(std::make_pair(floor, floordiv(dims.l1_var, dims.l0_var)));
          param_map.insert(std::make_pair(mod, floormod(dims.l1_var, dims.l0_var)));
        }
      }
    }
  }
}

Stmt Scop::ReplacePrimesWithParameters(Stmt stmt) {
  std::map<int64_t, Expr> param_map;
  if (is_spec_gemm_ || IsConv()) {
    if (tile_size_is_var_) {
      if (is_spec_gemm_) {
        InsertPairsSpecGemmTileVar(param_map);
      } else {
        InsertPairsConvTileVar(stmt, param_map);
      }
    } else {
      InsertPairsSpecGemmOrConv(stmt, param_map);
    }
  } else {
    InsertPairs(stmt, param_map);
  }
  param_tiling_map_ = param_map;
  stmt = ParameterizingTiling(param_map).Mutate(stmt);
  return stmt;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
