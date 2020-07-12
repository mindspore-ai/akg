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

#include "pass/post_fusion_utils.h"

namespace akg {
namespace ir {
void FindMNKValue::Find(const Stmt &stmt) {
  // collect m,n,k index info
  update_map_value_ = true;
  this->Visit(stmt);
  update_map_value_ = false;

  // get m, n, k value
  this->Visit(stmt);
}

void FindMNKValue::Visit_(const Call *op) {
  if (!update_map_value_) {
    return;
  }

  size_t pos = op->name.find("_local_L0C");
  size_t len = op->args.size();
  if (len < 4) {
    return;
  }

  if (pos != std::string::npos) {
    maps_["no"] = op->args[len - 4];
    maps_["mo"] = op->args[len - 3];
    maps_["mi"] = op->args[len - 2];
    maps_["ni"] = op->args[len - 1];

    return;
  }

  pos = op->name.find("_local_L0B");
  if (pos != std::string::npos && maps_.find("ni") != maps_.end()) {
    // none transpose B ko, no, ni, ki, trans B no, ko, ki, ni
    if (IsSame(maps_["ni"], op->args[len - 2])) {
      // none transpose
      maps_["ko"] = op->args[len - 4];
      maps_["ki"] = op->args[len - 1];
    } else {
      // transpose
      maps_["ko"] = op->args[len - 3];
      maps_["ki"] = op->args[len - 2];
    }
  }

  IRVisitor::Visit_(op);
}

bool FindMNKValue::IsSame(const Expr &left, const Expr &right) {
  auto lV = left.as<Variable>();
  auto lR = right.as<Variable>();
  if (lV != nullptr && lR != nullptr) {
    return lV->name_hint == lR->name_hint;
  }

  if (lV == nullptr && lR == nullptr) {
    return !Compare(left, right);
  }

  return false;
}

void FindMNKValue::Visit_(const For *op) {
  CHECK(op->loop_var.as<Variable>() != nullptr);
  std::string name = op->loop_var.as<Variable>()->name_hint;

  switch (mnk_) {
    case GemmMNK::M: {
      if (maps_.find("mo") != maps_.end() && maps_["mo"].as<Variable>() != nullptr &&
          maps_["mo"].as<Variable>()->name_hint == name) {
        o_axis_ = op->extent;
        ov_axis_ = op->loop_var;
      }

      if (maps_.find("mi") != maps_.end() && maps_["mi"].as<Variable>() != nullptr &&
          maps_["mi"].as<Variable>()->name_hint == name) {
        i_axis_ = op->extent;
        iv_axis_ = op->loop_var;
      }

      break;
    }
    case GemmMNK::N: {
      if (maps_.find("no") != maps_.end() && maps_["no"].as<Variable>() != nullptr &&
          maps_["no"].as<Variable>()->name_hint == name) {
        o_axis_ = op->extent;
        ov_axis_ = op->loop_var;
      }

      if (maps_.find("ni") != maps_.end() && maps_["ni"].as<Variable>() != nullptr &&
          maps_["ni"].as<Variable>()->name_hint == name) {
        i_axis_ = op->extent;
        iv_axis_ = op->loop_var;
      }

      break;
    }
    case GemmMNK::K: {
      if (maps_.find("ko") != maps_.end() && maps_["ko"].as<Variable>() != nullptr &&
          maps_["ko"].as<Variable>()->name_hint == name) {
        o_axis_ = op->extent;
        ov_axis_ = op->loop_var;
      }

      if (maps_.find("ki") != maps_.end() && maps_["ki"].as<Variable>() != nullptr &&
          maps_["ki"].as<Variable>()->name_hint == name) {
        i_axis_ = op->extent;
        iv_axis_ = op->loop_var;
      }

      break;
    }
    default: {
      CHECK(false);
    }
  }

  IRVisitor::Visit_(op);
}

void FindMadAttrVar::Visit_(const AttrStmt *op) {
  if (op->attr_key == "pragma_gemm_l0") {
    // attr [{"m_size": range(min=0, ext=98), "": range(min=0, ext=1), "no_0": range(min=0, ext=2),
    // "m_lager_size": range(min=0, ext=112), "ko_4": range(min=0, ext=16), "mo_": range(min=0, ext=1)}]
    // pragma_gemm_l0 = 0
    auto opRange = Downcast<Map<std::string, Range>>(op->node);
    if (use_all_) {
      ranges_ = opRange;
    } else {
      for (auto i : opRange) {
        ranges_.Set(i.first, i.second);
      }
    }
  } else if (op->attr_key == "pragma_spec_gemm_attr") {
    // attr [{"": ee3, "no_0": ee0, "ko_4": ee4, "mo_": ee1}] pragma_spec_gemm_attr = 0
    auto axis_map = Downcast<Map<std::string, VarExpr>>(op->node);
    old_axis_map_ = axis_map;
  }

  IRVisitor::Visit_(op);
}

void FindMadAttrVar::Visit_(const For *op) {
  // if for loop_var name is not contain "ee", is_var_substitute_, true
  if (op->loop_var->name_hint.find("ee") == std::string::npos) {
    is_var_substitute_ = true;
  } else {
    IRVisitor::Visit_(op);
  }
}

Range FindMadAttrVar::FindNameRange(const std::string &name) {
  for (auto i : ranges_) {
    std::string key = i.first;
    if (key == name) {
      return i.second;
    }
  }

  return Range(Expr(-1), Expr(-1));
}

Range FindMadAttrVar::FindRange(const std::string &preName) {
  for (auto i : ranges_) {
    std::string key = i.first;
    size_t pos = key.find('_');
    if ((pos != std::string::npos) && (key.substr(0, pos) == preName)) {
      return i.second;
    }
  }

  return Range(Expr(-1), Expr(-1));
}

std::string FindMadAttrVar::FindAxisName(const std::string &preName) const {
  for (auto i : ranges_) {
    std::string key = i.first;
    size_t pos = key.find('_');
    if ((pos != std::string::npos) && (key.substr(0, pos) == preName)) {
      return i.first;
    }
  }

  return "";
}

std::string FindMadAttrVar::FindOldAxisName(const std::string &newName) {
  if (old_axis_map_.count(newName) >= 1) {
    return old_axis_map_[newName].operator->()->name_hint;
  }

  return "";
}

Expr SubstituteArgs::Mutate_(const Call *op, const Expr &e) {
  auto pos = op->func->func_name().find(bias_);
  if (pos != std::string::npos) {
    if (op->func->func_name() == bias_) {
      CHECK((bias_args_.size() == 5) && (bias_offset_.size() == 5)) << "args'size must be 5";

      Array<Expr> args;
      args.push_back(bias_args_[NN] - bias_offset_[NN]);
      args.push_back(bias_args_[C1] - bias_offset_[C1]);
      args.push_back(bias_args_[HH] - bias_offset_[HH]);
      args.push_back(bias_args_[WW] - bias_offset_[WW]);
      args.push_back(bias_args_[C0] - bias_offset_[C0]);

      return Call::make(op->type, op->name, args, Call::CallType::Halide, op->func, op->value_index);
    } else {
      return Call::make(op->type, op->name, bias_args_, Call::CallType::Halide, op->func, op->value_index);
    }
  } else if (is_reduce_) {
    bool find = false;
    for (auto it : reduce_tensor_set_) {
      if (it->func->func_name() == op->func->func_name()) {
        find = true;
        break;
      }
    }

    if (find) {
      return Call::make(op->type, op->name, reduce_args_, Call::CallType::Halide, op->func, op->value_index);
    } else {
      return Call::make(op->type, op->name, args_, Call::CallType::Halide, op->func, op->value_index);
    }
  } else {
    return Call::make(op->type, op->name, args_, Call::CallType::Halide, op->func, op->value_index);
  }
}

Stmt SubstituteArgs::Mutate_(const Provide *op, const Stmt &s) {
  auto pos = op->func->func_name().find(bias_);
  auto value = this->Mutate(op->value);

  if (pos != std::string::npos) {
    return Provide::make(op->func, op->value_index, value, bias_args_);
  } else if (is_reduce_) {
    return Provide::make(op->func, op->value_index, value, reduce_args_);
  } else {
    return Provide::make(op->func, op->value_index, value, args_);
  }
}

Stmt RealizeNewShape::Mutate_(const AttrStmt *op, const Stmt &s) {
  if (op->attr_key == "alloc_C") {
    l0write_region_.clear();

    Array<Expr> tmp;
    c_ub_l0idx_ = tmp;
    static_cast<void>(this->Mutate(op->body));

    mutate_ = true;
    auto stmt = IRMutator::Mutate_(op, s);
    mutate_ = false;

    return stmt;
  }

  if (op->attr_key == "pragma_cube_l0write") {
    is_l0write_ = true;
    auto stmt = this->Mutate(op->body);
    is_l0write_ = false;

    return stmt;
  }

  return IRMutator::Mutate_(op, s);
}

Stmt RealizeNewShape::Mutate_(const For *op, const Stmt &s) {
  if (is_l0write_) {
    CHECK(op->loop_var.as<Variable>() != nullptr);
    l0write_region_[op->loop_var.as<Variable>()->name_hint] = Range::make_by_min_extent(op->min, op->extent);
  }

  return IRMutator::Mutate_(op, s);
}

Stmt RealizeNewShape::Mutate_(const Provide *op, const Stmt &s) {
  if (is_l0write_) {
    c_ub_l0idx_ = op->args;
  }

  return IRMutator::Mutate_(op, s);
}

Stmt RealizeNewShape::Mutate_(const Realize *op, const Stmt &s) {
  auto name = op->func->func_name();
  if (mutate_ && name.find("_local_UB") != std::string::npos) {
    size_t pos = name.find('_');
    std::string bias = name.substr(0, pos);
    Region bounds;

    for (size_t i = 0; i < c_ub_l0idx_.size(); ++i) {
      if (Equal(c_ub_l0idx_[i], 0)) {
        if (i > HH) {
          bounds.push_back(Range::make_by_min_extent(Expr(0), 16));
        } else {
          bounds.push_back(Range::make_by_min_extent(Expr(0), 1));
        }

        continue;
      }

      if (((bias_ == bias) && (i != C1 && i != C0))) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), 1));
        continue;
      }

      if (i > HH) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), 16));
      } else {
        CHECK(c_ub_l0idx_[i].as<Variable>() != nullptr);
        bounds.push_back(l0write_region_[c_ub_l0idx_[i].as<Variable>()->name_hint]);
      }
    }

    auto body = this->Mutate(op->body);

    return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
  }

  return IRMutator::Mutate_(op, s);
}

void FractalInfoExtractor::Visit_(const For *op) {
  if (!pragma_gemm_l0_) {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;
    lv_map_[name] = var;
  }

  CHECK(op->loop_var.as<Variable>() != nullptr);
  std::string name = op->loop_var.as<Variable>()->name_hint;

  r_map_.emplace(std::pair<std::string, Range>(name, Range::make_by_min_extent(op->min, op->extent)));
  IRVisitor::Visit_(op);
  r_map_.erase(name);
}

void FractalInfoExtractor::Visit_(const AttrStmt *op) {
  if (op->attr_key == "pragma_gemm_l0") {
    // pragma_gemm_l0 attr info  based on n outer outer, m outer outer, k outer outer
    auto axisRange = Downcast<Map<std::string, Range>>(op->node);
    unsigned int bitmap = 0;
    Expr base;

    for (auto kv : axisRange) {
      std::string name = kv.first;
      auto bitNum = static_cast<unsigned int>(name[0] - 'k');
      auto bit = static_cast<unsigned int>(1 << bitNum);

      if (name.compare(0, 3, "mo_") == 0) {
        CHECK_EQ((bitmap & bit), 0);
        ComputeMadAxis(axis_map_["m"], name, kv.second);
        base = axis_map_["m"].base;
        bitmap |= bit;
      } else if (name.compare(0, 3, "no_") == 0) {
        CHECK_EQ((bitmap & bit), 0);
        ComputeMadAxis(axis_map_["n"], name, kv.second);
        bitmap |= bit;
      } else if (name.compare(0, 3, "ko_") == 0) {
        CHECK_EQ((bitmap & bit), 0);
        ComputeMadAxis(axis_map_["k"], name, kv.second);
        bitmap |= bit;
      }
    }

    Expr baseN = Expr(0);
    auto iter = axis_map_.find("n");
    if (iter != axis_map_.end()) {
      baseN = iter->second.base;
    }

    Stmt s = Stmt(GetObjPtr(op));
    ComputeMFormula(s, base, baseN);
    CHECK(bitmap == ((1 << ('m' - 'k')) | (1 << ('n' - 'k')) | (1 << ('k' - 'k')))) << "bitmap: " << bitmap;

    // extract (m/n/k, ii/oi) axis
    UpdateMNKAxis(GemmMNK::M, s);
    UpdateMNKAxis(GemmMNK::N, s);
    UpdateMNKAxis(GemmMNK::K, s);
  }

  IRVisitor::Visit_(op);
}

void FractalInfoExtractor::UpdateMNKAxis(GemmMNK mnk, const Stmt &s) {
  FindMNKValue f = FindMNKValue(mnk);
  f.Find(s);
  std::string key;

  switch (mnk) {
    case GemmMNK::M:
      key = "m";
      break;
    case GemmMNK::N:
      key = "n";
      break;
    case GemmMNK::K:
      key = "k";
      break;
    default:
      break;
  }

  CHECK_NE(key, "");
  axis_map_[key].oi = Range(0, f.GetOAxis());
  axis_map_[key].ii = Range(0, f.GetIAxis());
}

void FractalInfoExtractor::ComputeMadAxis(MadAxis &curAxis, const std::string &name, const Range &range) {
  if (is_zero(range->min)) {
    curAxis.base = Expr(0);
  } else if (is_zero(Simplify_cce(curAxis.oo->min - range->min)) &&
             (is_zero(Simplify_cce(curAxis.oo->extent - range->extent)))) {
    return;
  } else {
    CHECK(is_zero(Simplify_cce(curAxis.oo->min + curAxis.oo->extent - range->min)));
    curAxis.base = Simplify_cce(curAxis.base + curAxis.oo->extent * curAxis.oi->extent * curAxis.ii->extent);
  }

  if (is_one(range->extent)) {
    curAxis.var = VarExpr(ObjectPtr<Object>());
  } else if (lv_map_.count(name)) {
    curAxis.var = lv_map_[name];
  }

  curAxis.oo = range;
  curAxis.oi = Range(0, 1);
  curAxis.ii = Range(0, 1);
}

void FractalInfoExtractor::ComputeMFormula(const Stmt &smt, const Expr &base, const Expr &baseN) {
  auto fAttr = FindMadAttrVar(false);
  fAttr.Visit(smt);

  auto mFind = FindMNKValue(GemmMNK::M);
  mFind.Find(smt);
  Expr mo = mFind.GetOVAxis();
  Expr mi = mFind.GetIVAxis();
  Expr moExt = mFind.GetOAxis();
  Expr miExt = mFind.GetIAxis();

  auto nFind = FindMNKValue(GemmMNK::N);
  nFind.Find(smt);
  Expr noExt = nFind.GetOAxis();
  Expr niExt = nFind.GetIAxis();

  if (Equal(noExt, -1)) {
    noExt = Expr(1);
  }

  if (Equal(moExt, -1)) {
    moExt = Expr(1);
  }

  std::string axisName = fAttr.FindAxisName("mo");
  Expr base1 = Expr(0);
  if (!(fAttr.is_var_substitute_) && !fAttr.FindOldAxisName(axisName).empty()) {
    axisName = fAttr.FindOldAxisName(axisName);
  }

  if (lv_map_.find(axisName) != lv_map_.end()) {
    Range r = fAttr.FindRange("mo");
    if (r->extent.as<IntImm>() && r->extent.as<IntImm>()->value <= 1) {
      base1 = Expr(0);
    } else {
      base1 = lv_map_[axisName] * moExt;
    }
  } else if (fAttr.FindAxisName("mo") == "mo_") {
    base1 = Expr(0);
  }

  Expr base2 = Simplify_cce((base1 + mo) * miExt);
  if (miExt.as<IntImm>()->value < 16) {
    base2 = Expr(0);
  }

  Expr result = Simplify_cce(base + base2 + mi);

  Range wRange = fAttr.FindNameRange("w_size");
  Expr tileWo = wRange->extent;
  if (tileWo.as<IntImm>() && tileWo.as<IntImm>()->value < 1) {
    tileWo = Expr(1);
  }

  if (!is_dynamic_) CHECK_NE(GetIntConst(tileWo), 0);
  Expr hRes = ((result) / tileWo);
  Expr wRes = ((result) % tileWo);

  axisName = fAttr.FindAxisName("no");
  if (!(fAttr.is_var_substitute_) && !fAttr.FindOldAxisName(axisName).empty()) {
    axisName = fAttr.FindOldAxisName(axisName);
  }

  Expr noExpr = Expr(0);
  if (lv_map_.find(axisName) != lv_map_.end()) {
    Range r = fAttr.FindRange("no");
    if (!is_zero(Simplify_cce(r->extent - 1))) {
      noExpr = lv_map_[axisName];
    }
  }

  Expr nRes = Simplify_cce(noExpr * noExt);
  if (!gemmFormula_.empty()) {
    int last = static_cast<int>(gemmFormula_.size());
    // remove K isolate formula computation
    if (IsConvGemmKIsolate() && Equal(gemmFormula_[last - 1], base)) {
      return;
    }

    nRes = Simplify_cce((baseN + noExpr * noExt * niExt) / niExt);
  }

  gemmFormula_.push_back(nRes);
  gemmFormula_.push_back(hRes);
  gemmFormula_.push_back(wRes);
  gemmFormula_.push_back(base);
}

bool FractalInfoExtractor::IsConvGemmKIsolate() { return (!Equal(axis_map_["k"].base, 0)); }

bool hasAnyUpperStr(const std::string &s) {
  for (auto c : s) {
    if (c >= 'A' && c <= 'Z') return true;
  }

  return false;
}

void ExtractIterfromExpr::Visit_(const Variable *op) {
  if (!hasAnyUpperStr(op->name_hint)) idx_vec_.insert(op);

  IRVisitor::Visit_(op);
}

void ExtractIterfromExpr::Visit_(const Block *op) { this->Visit(op->rest); }

void ExtractIterfromExpr::Visit_(const Add *op) {
  auto va = op->a.as<Variable>();
  if (va != nullptr && mi_var_ == nullptr && !hasAnyUpperStr(va->name_hint)) {
    mi_var_ = op->a.as<Variable>();
  }

  auto vb = op->b.as<Variable>();
  if (vb != nullptr && mi_var_ == nullptr && !hasAnyUpperStr(vb->name_hint)) {
    mi_var_ = op->b.as<Variable>();
  }

  IRVisitor::Visit_(op);
}

Stmt GemmAxisMap::Mutate_(const Block *op, const Stmt &s) {
  if (op->rest.defined()) {
    Stmt first = this->Mutate(op->first);

    return Block::make(first, op->rest);
  }

  return this->Mutate(op->first);
}

Stmt GemmAxisMap::Mutate_(const Provide *op, const Stmt &s) {
  Array<Expr> right_args;
  CHECK(s.as<Provide>() != nullptr);
  auto right = s.as<Provide>()->value.as<Call>();
  if ((right != nullptr) && (right->call_type == Call::Halide)) {
    right_args = right->args;
  }

  if (is_conv_backprop_filter_) {
    UpdateAxisMap(right_args[1], "mo");
    UpdateAxisMap(right_args[2], "mi");
    UpdateAxisMap(right_args[3], "ni");
  } else {
    UpdateAxisMap(right_args[2], "mo");
    UpdateAxisMap(right_args[3], "mi");
    UpdateAxisMap(right_args[4], "ni");
  }

  return IRMutator::Mutate_(op, s);
}

void GemmAxisMap::UpdateAxisMap(const Expr &e, const std::string &v) {
  const auto tmp = e.as<Variable>();
  if (tmp != nullptr) {
    axis_map_info_[v] = e;
  }
}

void FindOutC1HW::Visit_(const Provide *op) {
  auto c = op->value.as<Call>();
  if (c && (IsInBinds(op->func->func_name(), binds_orig_) || IsInBinds(c->func->func_name(), binds_orig_))) {
    Array<Expr> left_args = op->args;
    Array<Expr> right_args = c->args;

    CHECK(right_args.size() == left_args.size()) << "Wrong args: left " << left_args << " right " << right_args;

    Expr H_expr = Simplify_cce(left_args[HH] - right_args[HH]);
    OutHExpr_ = H_expr;
    if (!is_const(H_expr)) {
      check_h_ = true;
      this->Visit(H_expr);
    }

    Expr W_expr = Simplify_cce(left_args[WW] - right_args[WW]);
    OutWExpr_ = W_expr;
    if (!is_const(W_expr)) {
      check_w_ = true;
      this->Visit(W_expr);
    }

    Expr C1_expr = Simplify_cce(left_args[C1] - right_args[C1]);
    if (!is_const(C1_expr)) {
      check_c1_ = true;
      this->Visit(C1_expr);
    }
  }

  IRVisitor::Visit_(op);
}

void FindOutC1HW::Visit_(const Variable *op) {
  if (loopvars_.count(op) == 0) {
    if (check_h_) {
      OutH_ = op;
      check_h_ = false;
    }

    if (check_c1_) {
      OutC1_ = op;
      check_c1_ = false;
    }

    if (check_w_) {
      OutW_ = op;
      check_w_ = false;
    }
  }

  IRVisitor::Visit_(op);
}

void FindOutC1HW::Visit_(const For *op) {
  loopvars_.insert(op->loop_var.get());
  IRVisitor::Visit_(op);
}

void RegionExtract::Visit_(const For *op) {
  const auto var = op->loop_var.as<Variable>();

  region_map_.emplace(std::pair<const Variable *, Range>(var, Range::make_by_min_extent(op->min, op->extent)));
  IRVisitor::Visit_(op);
  region_map_.erase(var);
}

void RegionExtract::Visit_(const Provide *op) {
  if (op->func.defined() && op->func->func_name() == name_) {
    Region new_bounds;
    // We need to consider multiple realizes of a same tensor.
    // Insert range bound if it is the first region, or the new region has more dims.
    // If the dimension already exists, merge (min, extent) to cover both ranges.
    for (size_t i = 0; i < op->args.size(); ++i) {
      auto bound = InferSimpleExprRange(op->args[i], &region_map_);

      if (bounds_.size() > i) {
        auto merged_min = Simplify_cce(min(bound->min, bounds_[i]->min));
        auto merged_max = Simplify_cce(max(bound->min + bound->extent, bounds_[i]->min + bounds_[i]->extent));
        auto merged_extent = Simplify_cce(merged_max - merged_min);

        new_bounds.push_back(Range::make_by_min_extent(merged_min, merged_extent));
      } else {
        new_bounds.push_back(bound);
      }
    }

    bounds_ = new_bounds;
  }
}

Expr TensorReplace::Mutate_(const Call *op, const Expr &e) {
  if (func_map_.count(op->name) > 0) {
    Expr new_call = Call::make(op->type, op->name, rhs_args_, op->call_type, func_map_[op->name], op->value_index);

    return new_call;
  }

  return IRMutator::Mutate_(op, e);
}

Stmt TensorReplace::Mutate_(const For *op, const Stmt &s) {
  VarExpr var = op->loop_var;
  std::string name = var->name_hint;

  inner_loopvar_map_.emplace(std::pair<std::string, const For *>(name, op));
  Stmt body = this->Mutate(op->body);
  inner_loopvar_map_.erase(name);

  if (drop_loop_.count(name) > 0) {
    drop_loop_.erase(name);
    return body;
  } else {
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }
}

Stmt TensorReplace::Mutate_(const Realize *op, const Stmt &s) {
  Region bounds;
  Stmt body = this->Mutate(op->body);

  for (auto arg : rhs_args_) {
    if (const auto var = arg.as<Variable>()) {
      if (std::any_of(outer_loopvar_map_.begin(), outer_loopvar_map_.end(),
                      [&](const std::pair<std::string, const For *> &kv) { return (kv.first == var->name_hint); })) {
        bounds.push_back(Range::make_by_min_extent(Expr(0), outer_loopvar_map_[var->name_hint]->extent));
      } else {
        CHECK(0) << "cannot find";
      }
    } else {
      CHECK(is_zero(arg)) << arg;
      bounds.push_back(Range::make_by_min_extent(Expr(0), Expr(1)));
    }
  }

  return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
}

Stmt TensorReplace::Mutate_(const Provide *op, const Stmt &s) {
  Expr newValue;
  std::string name = op->func->func_name();
  if (func_map_.count(name) == 0) {
    func_map_.emplace(std::pair<std::string, FunctionRef>(name, op->func));
  }
  newValue = this->Mutate(op->value);

  Array<Expr> newArgs;
  if (op->func->func_name().find("_local_UB") != std::string::npos) {
    newArgs = rhs_args_;
  } else {
    newArgs = lhs_args_;
  }

  auto c = newValue.as<Call>();
  CHECK(c != nullptr);
  auto value =
    Add::make(Call::make(c->type, op->func->func_name(), newArgs, c->call_type, op->func, op->value_index), newValue);
  Stmt ans = Provide::make(op->func, op->value_index, value, newArgs);

  if (auto call = op->value.as<Call>()) {
    Array<Expr> oldArgs = call->args;
    for (int i = static_cast<int>(oldArgs.size()) - 1; i >= 0; i--) {
      if (oldArgs[i].as<Variable>()) {
        std::string name_old = oldArgs[i].as<Variable>()->name_hint;

        drop_loop_.insert(name_old);
      }
    }
  }

  for (int i = static_cast<int>(rhs_args_.size()) - 1; i >= 0; i--) {
    if (rhs_args_[i].as<Variable>()) {
      const For *f = nullptr;
      std::string name_rhs = rhs_args_[i].as<Variable>()->name_hint;
      if (inner_loopvar_map_.count(name_rhs) > 0) {
        f = inner_loopvar_map_[name_rhs];
      } else {
        CHECK_GT(outer_loopvar_map_.count(name_rhs), 0);
        f = outer_loopvar_map_[name_rhs];
      }

      ans = For::make(f->loop_var, f->min, f->extent, f->for_type, f->device_api, ans);
    }
  }

  return ans;
}

void GetOuterAxisRHS::Visit_(const Provide *op) {
  if (op->func->func_name() == tensor_name_) {
    is_provide_ = true;
    this->Visit(op->value);
    is_provide_ = false;
  }
}

void GetOuterAxisRHS::Visit_(const Call *op) {
  if (is_provide_) {
    CHECK(op->args.size() > static_cast<unsigned int>(idx_));

    is_idx_ = true;
    this->Visit(op->args[idx_]);
    is_idx_ = false;
  }
}

void GetOuterAxisRHS::Visit_(const Variable *op) {
  if (is_provide_ && is_idx_) {
    for (const auto &kv : outer_loopvar_map_) {
      if (kv.first == op->name_hint) {
        var_ = kv.second;
      }
    }
  }
}

void GetOuterAxisLHS::Visit_(const Provide *op) {
  if (op->func->func_name() == tensor_name_) {
    CHECK(op->args.size() > static_cast<unsigned int>(idx_));

    is_idx_ = true;
    this->Visit(op->args[idx_]);
    is_idx_ = false;
  }
}

void GetOuterAxisLHS::Visit_(const Variable *op) {
  if (is_idx_) {
    for (const auto &kv : outer_loopvar_map_) {
      if (kv.first == op->name_hint) {
        var_ = kv.second;
        break;
      }
    }
  }
}

Stmt RemoveNullRealize::Mutate_(const Realize *op, const Stmt &s) {
  Stmt stmt = IRMutator::Mutate_(op, s);

  if (funcs_.count(op->func) > 0) {
    return stmt;
  } else {
    CHECK(stmt.as<Realize>() != nullptr);
    return stmt.as<Realize>()->body;
  }
}

Stmt RemoveNullRealize::Mutate_(const Provide *op, const Stmt &s) {
  funcs_.insert(op->func);

  return IRMutator::Mutate_(op, s);
}

Expr RemoveNullRealize::Mutate_(const Call *op, const Expr &e) {
  funcs_.insert(op->func);

  return IRMutator::Mutate_(op, e);
}

Stmt RemoveNullRealizeScope::Mutate_(const AttrStmt *op, const Stmt &s) {
  if (op->attr_key == air::ir::attr::realize_scope && !op->body.as<Realize>()) {
    return AttrStmt::make(make_zero(Int(32)), "old_realize", 0, this->Mutate(op->body));
  }

  if (op->attr_key == "alloc_C") {
    if (allocC_) {
      return this->Mutate(op->body);
    } else {
      allocC_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      allocC_ = false;

      return stmt;
    }
  }

  if (op->attr_key == "pragma_cube_l0write") {
    if (conv_.reduce_at_l1 && isolate_idx_ % conv_.l1_reduce_base == 0 && gemm_idx_ == gemm_num_) {
      return AttrStmt::make(op->node, "pragma_cube_l0write", Expr(++l0write_idx_), this->Mutate(op->body));
    } else if (!conv_.reduce_at_l1 && gemm_idx_ % conv_.l0_reduce_base == 0) {
      return AttrStmt::make(op->node, "pragma_cube_l0write", Expr(++l0write_idx_), this->Mutate(op->body));
    } else {
      return AttrStmt::make(op->node, "pragma_cube_l0write", Expr(-1), this->Mutate(op->body));
    }
  }

  if (op->attr_key == "pragma_gemm_l0") {
    ++gemm_idx_;
  }

  if (op->attr_key == "isolated_idx") {
    if (isolate_idx_ > 0) {
      CHECK_EQ(gemm_idx_, gemm_num_) << isolate_idx_ << " : " << gemm_idx_ << " : " << gemm_num_;
    }

    gemm_num_ = conv_.infer_L0_tile(isolate_idx_++);
    gemm_idx_ = 0;
  }

  return IRMutator::Mutate_(op, s);
}

Stmt MarkAxis::Mutate_(const For *op, const Stmt &s) {
  VarExpr var = op->loop_var;
  std::string name = var->name_hint;

  outerlv_map_.insert({name, op->loop_var});
  Stmt ans = this->Mutate(op->body);
  outerlv_map_.erase(name);

  if (name == kh_var_->name_hint) {
    ans = AttrStmt::make(make_zero(Int(32)), "KH_axis", kh_var_, ans);
    kh_var_ = VarExpr("");
  } else if (name == kw_var_->name_hint) {
    ans = AttrStmt::make(make_zero(Int(32)), "KW_axis", kw_var_, ans);
    kw_var_ = VarExpr("");
  }

  return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, ans);
}

Stmt MarkAxis::Mutate_(const AttrStmt *op, const Stmt &s) {
  if (op->attr_key == "isolated_idx") {
    GetOuterAxisLHS axisKH(outerlv_map_, output_name_, 1);
    axisKH.Visit(s);
    kh_var_ = axisKH.var_;

    GetOuterAxisLHS axisKW(outerlv_map_, output_name_, 2);
    axisKW.Visit(s);
    kw_var_ = axisKW.var_;
  }

  return IRMutator::Mutate_(op, s);
}

Stmt ElseCaseSplit::Mutate_(const For *op, const Stmt &s) {
  if (auto op_if = op->body.as<IfThenElse>()) {
    if (!op_if->else_case.defined()) {
      return IRMutator::Mutate_(op, s);
    }

    int extent = static_cast<int>(op->extent.as<IntImm>()->value);
    if (extent == 2) {
      auto expr_eq = op_if->condition.as<EQ>();
      if (!expr_eq || !expr_eq->a.as<Variable>() || !is_zero(expr_eq->b)) {
        return IRMutator::Mutate_(op, s);
      }

      if (expr_eq->a.as<Variable>()->name_hint == op->loop_var->name_hint) {
        std::unordered_map<const Variable *, Expr> then_map;
        then_map.emplace(op->loop_var.get(), Expr(0));
        Stmt then_case = Substitute(op_if->then_case, then_map);

        std::unordered_map<const Variable *, Expr> else_map;
        else_map.emplace(op->loop_var.get(), Expr(1));
        Stmt else_case = Substitute(op_if->else_case, else_map);

        return Block::make(then_case, else_case);
      }
    } else if (extent > 2) {
      auto expr_le = op_if->condition.as<LE>();
      if (!expr_le || !expr_le->a.as<Variable>() || !is_zero(expr_le->b + 2 - extent)) {
        return IRMutator::Mutate_(op, s);
      }

      if (expr_le->a.as<Variable>()->name_hint == op->loop_var->name_hint) {
        Stmt then_case =
          For::make(op->loop_var, op->min, op->extent - 1, op->for_type, op->device_api, op_if->then_case);

        std::unordered_map<const Variable *, Expr> else_map;
        else_map.emplace(op->loop_var.get(), Expr(extent - 1));
        Stmt else_case = Substitute(op_if->else_case, else_map);

        return Block::make(then_case, else_case);
      }
    }
  }

  return IRMutator::Mutate_(op, s);
}

void GatherOpAfterReduce::Visit_(const AttrStmt *op) {
  if (op->attr_key == "pragma_op_after_reduce") {
    visit_provide_ = true;
    already_in_ = false;
    relate_ = false;

    this->Visit(op->body);

    if (relate_ && !already_in_) {
      op_after_reduce_.emplace_back(op->body);
    }

    visit_provide_ = false;
  }

  IRVisitor::Visit_(op);
}

void GatherOpAfterReduce::Visit_(const Provide *op) {
  if (visit_provide_) {
    if (provides_.count(op->func->func_name())) {
      already_in_ = true;

      return;
    }

    if (op->func->func_name() == name_) {
      provides_.insert(op->func->func_name());
      relate_ = true;

      return;
    }

    this->Visit(op->value);

    if (relate_) {
      provides_.insert(op->func->func_name());
      if (op->func->func_name().find("local") != std::string::npos) {
        miss_realize_.insert(op);
      }

      return;
    }
  }

  IRVisitor::Visit_(op);
}

void GatherOpAfterReduce::Visit_(const Call *op) {
  if (visit_provide_) {
    if (op->func->func_name() == name_) {
      provides_.insert(op->func->func_name());
      relate_ = true;

      return;
    }

    if (provides_.count(op->func->func_name())) {
      relate_ = true;

      return;
    }
  }

  IRVisitor::Visit_(op);
}

void GatherC1Offset::Visit_(const AttrStmt *op) {
  if (op->attr_key == "pragma_fuse_vector") {
    in_fuse_vector_ = true;
    this->Visit(op->body);
    in_fuse_vector_ = false;

    return;
  }

  IRVisitor::Visit_(op);
}

void GatherC1Offset::Visit_(const Provide *op) {
  if (in_fuse_vector_ && IsInBinds(op->func->func_name(), binds_)) {
    CHECK_GE(op->args.size(), 4);

    found_ = true;
    gm_c1_ = op->args[C1];
    this->Visit(op->value);
    gm_c1_ = 0;
    found_ = false;
  }

  IRVisitor::Visit_(op);
}

void GatherC1Offset::Visit_(const Call *op) {
  if (found_ && !IsInBinds(op->func->func_name(), binds_)) {
    CHECK_GE(op->args.size(), 4);
    auto ub_c1 = op->args[C1];
    c1_offset_.emplace_back(Simplify_cce(gm_c1_ - ub_c1));
  }

  IRVisitor::Visit_(op);
}
}  // namespace ir
}  // namespace akg
