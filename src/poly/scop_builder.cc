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

#include "poly/scop_builder.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <tvm/schedule.h>

#include "pass/utils.h"
#include "construct_poly_accesses.h"
#include "poly/dsa_utils.h"
#include "poly/schedule_tree_util.h"

namespace akg {
namespace ir {
namespace poly {

// Note: We can only handle empty param_space for now.
isl::space CreateParamsSpace(const isl::ctx &ctx) { return isl::space(ctx, 0); }

isl::space CreateParamsSpace(const isl::ctx &ctx, const std::unordered_map<std::string, air::Var> &params) {
  auto space = isl::space(ctx, 0);

  // set parameter names
  for (auto it = params.begin(); it != params.end(); ++it) {
    space = space.add_param(isl::id(ctx, it->second->name_hint));
  }
  return space;
}

isl::aff Int2Aff(const isl::space &s, int64_t v) { return isl::aff(isl::local_space(s), isl::val(s.ctx(), v)); }

template <typename T>
inline std::vector<isl::aff> ConcatAffs(const isl::space &space, T *op, bool allow_min, bool allow_max) {
  std::vector<isl::aff> result;

  for (const auto &aff : Expr2AffBounds(space, op->a, allow_min, allow_max)) {
    result.push_back(aff);
  }
  for (const auto &aff : Expr2AffBounds(space, op->b, allow_min, allow_max)) {
    result.push_back(aff);
  }

  return result;
}

template <typename T>
inline std::vector<isl::aff> UniteAffs(const isl::space &space, T *op, isl::aff (isl::aff::*unite)(isl::aff) const) {
  std::vector<isl::aff> bounds_l = Expr2AffBounds(space, op->a, false, false);
  std::vector<isl::aff> bounds_r = Expr2AffBounds(space, op->b, false, false);
  CHECK_LE(bounds_l.size(), 1u);
  CHECK_LE(bounds_r.size(), 1u);

  if (bounds_l.size() > 0 && bounds_r.size() > 0) {
    return {(bounds_l[0].*unite)(bounds_r[0])};
  }

  return {};
}

template <typename T>
bool ExprType(const Expr &e) {
  return (e.as<T>() != nullptr);
}

std::vector<isl::aff> Variable2AffBounds(const isl::space &space, const Variable *v, bool ignore_error) {
  isl::id id(space.ctx(), v->name_hint);
  if (space.has_param(id)) {
    return {isl::aff::param_on_domain(space, id)};
  }
  CHECK(ignore_error) << "Can not find var: " << v->name_hint << " in isl::space: " << space << '\n';
  return {};
}

std::vector<isl::aff> FloorDiv2AffBounds(const isl::space &space, const FloorDiv *f_div) {
  if (f_div->type.is_int() || f_div->type.is_uint()) {
    auto left = Expr2AffBounds(space, f_div->a, false, false);
    auto right = Expr2AffBounds(space, f_div->b, false, false);
    if (left.size() == 0 || right.size() == 0) {
      return {};
    }
    return {(left[0].div)(right[0]).floor()};
  }
  return UniteAffs(space, f_div, &isl::aff::div);
}

std::vector<isl::aff> Div2AffBounds(const isl::space &space, const Div *div) {
  if (div->type.is_int() || div->type.is_uint()) {
    auto left = Expr2AffBounds(space, div->a, false, false);
    auto right = Expr2AffBounds(space, div->b, false, false);
    if (left.size() == 0 || right.size() == 0) {
      return {};
    }
    return {(left[0].div)(right[0]).floor()};
  }
  return UniteAffs(space, div, &isl::aff::div);
}

std::vector<isl::aff> FloorMod2AffBounds(const isl::space &space, const FloorMod *f_mod, bool ignore_error) {
  auto left = Expr2AffBounds(space, f_mod->a, false, false);
  Expr right = f_mod->b;
  if (const int64_t *val = as_const_int(right)) {
    isl::val v = isl::val(space.ctx(), *val);
    return {left[0].mod(v)};
  }
  CHECK(ignore_error) << "Mod's denominator is not a const_int\n";
  return {};
}

std::vector<isl::aff> Mod2AffBounds(const isl::space &space, const Mod *mod, bool ignore_error) {
  auto left = Expr2AffBounds(space, mod->a, false, false);
  Expr right = mod->b;
  if (const int64_t *val = as_const_int(right)) {
    isl::val v = isl::val(space.ctx(), *val);
    return {left[0].mod(v)};
  }
  CHECK(ignore_error) << "Mod's denominator is not a const_int \n";
  return {};
}

std::vector<isl::aff> Select2AffBounds(const isl::space &space, const Select *sel) {
  /**************************************
   * Support Select expression aff bounds computation
   * select((15 < int32(ceil((float32(w)*5.000000f)))), 15, int32(ceil((float32(w)*5.000000f))))
   **************************************/
  auto true_aff_bounds = Expr2AffBounds(space, sel->true_value, false, false);
  auto false_aff_bounds = Expr2AffBounds(space, sel->false_value, false, false);
  if (true_aff_bounds.size() == 0 || false_aff_bounds.size() == 0) {
    return {};
  }
  /********************************************************
   * temp method just add true_value aff and false_value aff
   *******************************************************/
  return {(true_aff_bounds[0].add)(false_aff_bounds[0])};
}

std::vector<isl::aff> Expr2AffBounds(const isl::space &space, const Expr &e, bool allow_min, bool allow_max,
                                     bool ignore_error) {
  CHECK(!(allow_min && allow_max));

  if (ExprType<Variable>(e)) {
    return Variable2AffBounds(space, e.as<Variable>(), ignore_error);
  } else if (const int64_t *i = as_const_int(e)) {
    return {Int2Aff(space, *i)};
  } else if (ExprType<FloatImm>(e)) {
    return {Int2Aff(space, int64_t(e.as<FloatImm>()->value))};
  } else if (ExprType<Cast>(e)) {
    return Expr2AffBounds(space, e.as<Cast>()->value, false, false);
  } else if (const auto call = e.as<Call>()) {
    if ((call->name == "floor" || call->name == "ceil") && call->args.size() == 1) {
      return Expr2AffBounds(space, call->args[0], false, false);
    }
    LOG(INFO) << "not parse call type: " << call->name << " with expr :" << e;
  } else if (ExprType<Min>(e)) {
    if (!allow_min) return {};
    return ConcatAffs(space, e.as<Min>(), allow_min, allow_max);
  } else if (ExprType<Max>(e)) {
    if (!allow_max) return {};
    return ConcatAffs(space, e.as<Max>(), allow_min, allow_max);
  } else if (ExprType<Add>(e)) {
    return UniteAffs(space, e.as<Add>(), &isl::aff::add);
  } else if (ExprType<Sub>(e)) {
    return UniteAffs(space, e.as<Sub>(), &isl::aff::sub);
  } else if (ExprType<Mul>(e)) {
    return UniteAffs(space, e.as<Mul>(), &isl::aff::mul);
  } else if (ExprType<FloorDiv>(e)) {
    return FloorDiv2AffBounds(space, e.as<FloorDiv>());
  } else if (ExprType<Div>(e)) {
    return Div2AffBounds(space, e.as<Div>());
  } else if (ExprType<FloorMod>(e)) {
    return FloorMod2AffBounds(space, e.as<FloorMod>(), ignore_error);
  } else if (ExprType<Mod>(e)) {
    return Mod2AffBounds(space, e.as<Mod>(), ignore_error);
  } else if (ExprType<Select>(e)) {
    return Select2AffBounds(space, e.as<Select>());
  }

  CHECK(ignore_error) << "Expr2AffBounds " << e << "\n";
  return {};
}

std::vector<isl::aff> Expr2AffChecked(const isl::space &space, const Expr &e, bool allow_min, bool allow_max) {
  bool ignore_error = false;
  return Expr2AffBounds(space, e, allow_min, allow_max, ignore_error);
}

isl::aff Expr2Aff(const isl::space &space, const Expr &e) {
  auto list = Expr2AffChecked(space, e, false, false);
  return list.empty() ? isl::aff() : list[0];
}

isl::multi_id CollectTensorCoordinate(const isl::space &pspace, const isl::id &id, size_t dim) {
  isl::id_list args(pspace.ctx(), 0);
  for (size_t i = 0; i < dim; ++i) {
    auto name = std::string("arg") + std::to_string(i);
    args = args.add(isl::id(pspace.ctx(), name));
  }
  return isl::multi_id(pspace.add_named_tuple_id_ui(id, static_cast<unsigned int>(dim)), args);
}

isl::map AddSuffix4Accesses(AccessMap &accesses, const isl::map &in_map, const Node *op, const isl::ctx &ctx) {
  auto tensor_map = in_map;

  // Based on different condition, add suffix to the domain space
  std::string suffix;
  if (accesses.count(op) > 0) {  // reuse existing tag if the op is accessed previously
    suffix = accesses[op].to_str();
  } else {  // create a new tag with unique name
    suffix = "__poly_ref_" + std::to_string(accesses.size());
  }

  isl::id suffix_id(ctx, suffix);
  if (accesses.count(op) == 0) {  // only insert, not replace
    accesses.emplace(op, suffix_id);
  }

  auto domain_space = tensor_map.get_space().domain();
  auto tag_space = domain_space.params().add_named_tuple_id_ui(suffix_id, 0);
  domain_space = domain_space.product(tag_space).unwrap();
  tensor_map = tensor_map.preimage_domain(isl::multi_aff::domain_map(domain_space));

  return tensor_map;
}

bool ParseWithStmt(const Expr &s, const AnalysisResult &result) {
  class ParseWith final : public IRVisitor {
   public:
    void Visit_(const Call *op) final {
      if (!find_tensor && (0 != writes.size())) {
        if (op->call_type == Call::Halide) {
          if (writes.find(op->name) != writes.end()) {
            find_tensor = true;
          }
        }
      }
      IRVisitor::Visit_(op);
    }
    bool find_tensor{false};
    std::unordered_set<std::string> writes;

    bool GetResult() const { return find_tensor; }

    ParseWith(const Expr &stmt, const AnalysisResult &result) {
      result.GetWrites().foreach_map([&, this](const isl::map m) -> void {
        writes.insert(m.get_tuple_id(isl_dim_out).get_name());
        return;
      });
      IRVisitor::Visit(stmt);
    }
    ~ParseWith() override = default;
  } paserWith(s, result);

  return paserWith.GetResult();
}

void ParseStmtOpCall(const isl::id &id, const Call *call, AnalysisResult &result, const FunctionRef &func) {
  CHECK(call);
  if (call->call_type == Call::PureIntrinsic) {
    if (0 == strcmp(call->name.c_str(), "with")) {
      result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::with);
      if (!result.GetStmtOpInfoMap().at(id).isWith) {
        for (unsigned i = 0; i < call->args.size(); ++i) {
          if (ParseWithStmt(call->args[i], result)) {
            result.GetStmtOpInfoMap().at(id).isWith = true;
            break;
          }
        }
      }
    } else if (0 == strcmp(call->name.c_str(), "load_im2col_c1_buf")) {
      result.GetStmtOpInfoMap().at(id).is_load_im2col = true;
      ParseStmtOps(id, call->args[0], result, func);
    } else if (0 == strcmp(call->name.c_str(), "mad")) {
      result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::mad);
      result.GetStmtOpInfoMap().at(id).isMMU = true;
      // assign + mad
      std::string name = id.get_name();
      size_t index = static_cast<size_t>(WrappedStrtol(name.substr(name.length() - 1)));
      std::string tmp = name.substr(0, name.length() - 1);
      std::stringstream ss;
      ss << tmp << index - 1;
      if (result.GetStmtOpInfoMap().count(isl::id(id.ctx(), ss.str())) > 0 &&
          result.GetStmtOpInfoMap().at(isl::id(id.ctx(), ss.str())).ops[0] == PolyOpType::broadcast)
        result.GetStmtOpInfoMap().at(isl::id(id.ctx(), ss.str())).isMMUAssign = true;
      // end
      result.GetStmtOpInfoMap().at(id).C_ = func->func_name();
      CHECK(call->args.size() == 2) << "invalid args of mad! ";
      auto mul_arg = call->args[0].as<Mul>() ? call->args[0].as<Mul>() : call->args[1].as<Mul>();

      if (call->args[1].as<Cast>()) {
        CHECK(call->args[1].as<Cast>()->value.as<Mul>());
        mul_arg = call->args[1].as<Cast>()->value.as<Mul>();
      }
      CHECK(mul_arg);
      auto a = mul_arg->a.as<Call>();
      auto b = mul_arg->b.as<Call>();
      // in gemm case, C = mad(C,  A * B)
      if (a && b) {
        result.GetStmtOpInfoMap().at(id).A_ = a->name;
        result.GetStmtOpInfoMap().at(id).B_ = b->name;
      }
      // in conv case, reassign A&B by attr
      if (func.as<ComputeOpNode>() != nullptr) {
        result.GetStmtOpInfoMap().at(id).MadType_ =
          call->args[1].as<Cast>() ? call->args[1].as<Cast>()->type : Float(16);
        for (auto i : func.as<ComputeOpNode>()->attrs) {
          if ("feature" == i.first) {
            result.GetStmtOpInfoMap().at(id).A_ = i.second.as<StringImm>()->value;
          }
          if ("filter" == i.first) {
            result.GetStmtOpInfoMap().at(id).B_ = i.second.as<StringImm>()->value;
          }
        }
      }
    } else if (POLY_SUPPORTED_OPS.count(call->name)) {
      auto it = POLY_SUPPORTED_OPS.find(call->name);
      result.GetStmtOpInfoMap().at(id).ops.push_back(it->second);
    } else {
      LOG(FATAL) << "Unknown pure intrinsic: " << call->name.c_str() << std::endl;
    }
  }
}

void ParseStmtOps(const isl::id &id, const Expr &val, AnalysisResult &result, const FunctionRef &func) {
  result.GetStmtOpInfoMap().at(id).isMMU = false;
  result.GetStmtOpInfoMap().at(id).isMMUAssign = false;
  if (auto add = val.as<Add>()) {
    if (isImm(add->a) || isImm(add->b)) {
      if (!isImm(add->a)) {  // if add->a is not a scalar, then put it into recursion
        ParseStmtOps(id, add->a, result, func);
      } else if (!isImm(add->b)) {  // if add->b is not a scalar, then put it into recursion
        ParseStmtOps(id, add->b, result, func);
      } else {  // if add->a and add->b are both scalar, then report error
        LOG(FATAL) << "Error: Scalar + Scalar, Please Check.";
      }
      result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_single_VS_add);
    } else {
      ParseStmtOps(id, add->a, result, func);
      ParseStmtOps(id, add->b, result, func);
      result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_add);
    }
  } else if (auto sub = val.as<Sub>()) {
    ParseStmtOps(id, sub->a, result, func);
    ParseStmtOps(id, sub->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_sub);
  } else if (auto mul = val.as<Mul>()) {
    if (isImm(mul->a) || isImm(mul->b)) {
      // if mul->a is not a scalar, then put it into recursion
      if (!isImm(mul->a)) {
        ParseStmtOps(id, mul->a, result, func);
      } else if (!isImm(mul->b)) {  // if mul->b is not a scalar, then put it into recursion
        ParseStmtOps(id, mul->b, result, func);
      } else {  // if mul->a and mul->b are both scalar, then report error
        LOG(FATAL) << "Error: Scalar + Scalar, Please Check.";
      }
      if (isZero(mul->b) || isZero(mul->a)) {
        result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::broadcast);
      } else {
        result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_single_VS_mul);
      }
    } else {
      ParseStmtOps(id, mul->a, result, func);
      ParseStmtOps(id, mul->b, result, func);
      result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_mul);
    }
  } else if (auto f_div = val.as<FloorDiv>()) {
    ParseStmtOps(id, f_div->a, result, func);
    ParseStmtOps(id, f_div->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_div);
  } else if (auto f_mod = val.as<FloorMod>()) {
    ParseStmtOps(id, f_mod->a, result, func);
    ParseStmtOps(id, f_mod->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_mod);
  } else if (auto div = val.as<Div>()) {
    ParseStmtOps(id, div->a, result, func);
    ParseStmtOps(id, div->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_div);
  } else if (auto mod = val.as<Mod>()) {
    ParseStmtOps(id, mod->a, result, func);
    ParseStmtOps(id, mod->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_mod);
  } else if (auto and_op = val.as<And>()) {
    ParseStmtOps(id, and_op->a, result, func);
    ParseStmtOps(id, and_op->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_and);
  } else if (auto or_op = val.as<Or>()) {
    ParseStmtOps(id, or_op->a, result, func);
    ParseStmtOps(id, or_op->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_or);
  } else if (auto min = val.as<Min>()) {
    ParseStmtOps(id, min->a, result, func);
    ParseStmtOps(id, min->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_min);
  } else if (auto max = val.as<Max>()) {
    ParseStmtOps(id, max->a, result, func);
    ParseStmtOps(id, max->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_max);
  } else if (auto ge = val.as<GE>()) {
    ParseStmtOps(id, ge->a, result, func);
    ParseStmtOps(id, ge->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto gt = val.as<GT>()) {
    ParseStmtOps(id, gt->a, result, func);
    ParseStmtOps(id, gt->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto le = val.as<LE>()) {
    ParseStmtOps(id, le->a, result, func);
    ParseStmtOps(id, le->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto lt = val.as<LT>()) {
    ParseStmtOps(id, lt->a, result, func);
    ParseStmtOps(id, lt->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto eq = val.as<EQ>()) {
    ParseStmtOps(id, eq->a, result, func);
    ParseStmtOps(id, eq->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if (auto ne = val.as<NE>()) {
    ParseStmtOps(id, ne->a, result, func);
    ParseStmtOps(id, ne->b, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_cmp);
  } else if ((isImm(val) || val.type().is_int()) && val.as<Call>() == nullptr) {
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::broadcast);
  } else if (auto sel = val.as<Select>()) {
    ParseStmtOps(id, sel->true_value, result, func);
    ParseStmtOps(id, sel->false_value, result, func);
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::pandora_select);
  } else if ((val.as<Cast>())) {
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::vec_single_cast);
  } else if (auto call = val.as<Call>()) {
    ParseStmtOpCall(id, call, result, func);
  } else {
    LOG(WARNING) << "====>> WARNING: operator unknown type! " << val << " type:" << val.type();
    result.GetStmtOpInfoMap().at(id).ops.push_back(PolyOpType::elewise_binary_unknown);
  }
}

void ParseStmtOps(const isl::id &id, const Evaluate *stmt, AnalysisResult &result, const isl::union_map &new_reads,
                  const isl::union_map &new_writes) {
  if (!stmt) return;
  StmtOpInfo stmt_op_Info;
  for (auto a : new_reads.get_map_list()) {
    auto tensor_id = a.get_tuple_id(isl_dim_out);
    stmt_op_Info.readtensors.push_back(tensor_id);
  }

  if (stmt->value.as<Call>() && stmt->value.as<Call>()->name == CALL_IM2COL_UB) {
    stmt_op_Info.ops.push_back(PolyOpType::im2col);
    stmt_op_Info.isIm2col = true;
  }

  if (result.GetStmtOpInfoMap().count(id)) {
    // cce_img2col_ has no read_tensors and no write_tensors
    // maybe no need follows
    auto oldReadTensors = result.GetStmtOpInfoMap()[id].readtensors;
    for (auto i : oldReadTensors) {
      stmt_op_Info.readtensors.push_back(i);
    }
  }
  result.RecordStmtOpInfo(id, stmt_op_Info);
}

void ParseStmtOps(const isl::id &id, const Provide *stmt, AnalysisResult &result, const isl::union_map &new_reads,
                  const isl::union_map &new_writes) {
  if (!stmt) return;
  StmtOpInfo stmt_op_Info;
  for (auto a : new_reads.get_map_list()) {
    auto tensor_id = a.get_tuple_id(isl_dim_out);
    stmt_op_Info.readtensors.push_back(tensor_id);
  }
  if ((stmt->value.as<Call>() && stmt->value.as<Call>()->call_type == Call::Halide)) {
    stmt_op_Info.ops.push_back(PolyOpType::assignment);
  }
  if (result.GetStmtOpInfoMap().count(id)) {
    // one id has multiple statements with readtensors
    auto oldReadTensors = result.GetStmtOpInfoMap()[id].readtensors;
    for (auto i : oldReadTensors) {
      stmt_op_Info.readtensors.push_back(i);
    }
  }
  result.RecordStmtOpInfo(id, stmt_op_Info);

  ParseStmtOps(id, stmt->value, result, stmt->func);
}

VarNames VisitVarNames(const air::Expr &arg, VarNames var_names, bool add_num) {
  if (const auto var = arg.as<air::ir::Variable>()) {
    var_names.emplace_back(var->name_hint);
  } else if (const auto sub = arg.as<air::ir::Sub>()) {
    var_names = VisitVarNames(sub->a, var_names, add_num);
    var_names = VisitVarNames(sub->b, var_names, add_num);
  } else if (const auto add = arg.as<air::ir::Add>()) {
    var_names = VisitVarNames(add->a, var_names, add_num);
    var_names = VisitVarNames(add->b, var_names, add_num);
  } else if (const auto mul = arg.as<air::ir::Mul>()) {
    var_names = VisitVarNames(mul->a, var_names, add_num);
    var_names = VisitVarNames(mul->b, var_names, add_num);
  } else if (const auto div = arg.as<air::ir::Div>()) {
    var_names = VisitVarNames(div->a, var_names, add_num);
    var_names = VisitVarNames(div->b, var_names, add_num);
  } else if (const auto mod = arg.as<air::ir::Mod>()) {
    var_names = VisitVarNames(mod->a, var_names, add_num);
    var_names = VisitVarNames(mod->b, var_names, add_num);
  } else if (const auto int_imm = arg.as<air::ir::IntImm>()) {
    if (add_num) {
      var_names.emplace_back(std::to_string(int_imm->value));
    }
  } else if (const auto f_mod = arg.as<air::ir::FloorMod>()) {
    var_names = VisitVarNames(f_mod->a, var_names, add_num);
    var_names = VisitVarNames(f_mod->b, var_names, add_num);
  } else if (const auto f_div = arg.as<air::ir::FloorDiv>()) {
    var_names = VisitVarNames(f_div->a, var_names, add_num);
    var_names = VisitVarNames(f_div->b, var_names, add_num);
  }
  return var_names;
}

bool IsNum(const std::string &name) {
  for (auto c : name) {
    if (c > '9' || c < '0') {
      return false;
    }
  }
  return true;
};

/* "macro_stmt" is introduced to handle IfThenElse IR Node type. In particular, an if statement with its body should
 * be handled as one "macro" statement. "macro_stmt" records the statement label of the entire "macro" statements. In
 * other words, each "kStatementLabel" of a Provide or Store type should be the same with its enclosing IfThenElse
 * statement, equal to "macro_stmt".
 *
 * "macro_stmt" should be set to -1 to represent scop building should follow the regular way, i.e., each statement
 * should be analyzed individually, and it may be updated to a nonnegative number once an IfThenElse Node type is
 * encountered.
 *
 * The first step is to update the reads/writes of scop_info by extracting access information in the conditional of an
 * IfThenElse type node, followed by the building of the then_case (and also else case) schedule tree, which only
 * updates reads/writes sets without updating schedule tree.
 *
 * Currently, the loop variables of those enclosed by an if statement appear as parameters of the schedule tree. This
 * may be updated in the future for better manipulation of schedule trees.
 */

isl::set CutSet(std::vector<Expr> cond_vec, const isl::set &set, bool is_else = false, bool is_or = false) {
  if (cond_vec.empty()) return set;

  isl::space space = set.get_space();
  std::vector<isl::set> set_vec;

  for (size_t index = 0; index < cond_vec.size(); index++) {
    auto i = cond_vec[index];
    if (const LT *lt = i.as<LT>()) {
      auto left = Expr2AffBounds(space, lt->a, false, true);
      auto right = Expr2AffBounds(space, lt->b, true, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].ge_set(right[0]));
      else
        set_vec.push_back(left[0].lt_set(right[0]));
    } else if (const LE *le = i.as<LE>()) {
      auto left = Expr2AffBounds(space, le->a, false, true);
      auto right = Expr2AffBounds(space, le->b, true, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].gt_set(right[0]));
      else
        set_vec.push_back(left[0].le_set(right[0]));
    } else if (const GT *gt = i.as<GT>()) {
      auto left = Expr2AffBounds(space, gt->a, true, false);
      auto right = Expr2AffBounds(space, gt->b, false, true);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].le_set(right[0]));
      else
        set_vec.push_back(left[0].gt_set(right[0]));
    } else if (const GE *ge = i.as<GE>()) {
      auto left = Expr2AffBounds(space, ge->a, true, false);
      auto right = Expr2AffBounds(space, ge->b, false, true);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].lt_set(right[0]));
      else
        set_vec.push_back(left[0].ge_set(right[0]));
    } else if (const EQ *eq = i.as<EQ>()) {
      auto left = Expr2AffBounds(space, eq->a, false, false);
      auto right = Expr2AffBounds(space, eq->b, false, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].ne_set(right[0]));
      else
        set_vec.push_back(left[0].eq_set(right[0]));
    } else if (const NE *ne = i.as<NE>()) {
      auto left = Expr2AffBounds(space, ne->a, false, false);
      auto right = Expr2AffBounds(space, ne->b, false, false);
      if (left.empty() || right.empty()) return set;
      if (is_else)
        set_vec.push_back(left[0].eq_set(right[0]));
      else
        set_vec.push_back(left[0].ne_set(right[0]));
    } else if (const And *and_op = i.as<And>()) {
      cond_vec.push_back(and_op->a);
      cond_vec.push_back(and_op->b);
    } else {
      CHECK(false) << " find unknown conditions: " << i;
    }
  }
  for (size_t i = 1; i < set_vec.size(); ++i) {
    if (is_or)
      set_vec[0] = set_vec[0].unite(set_vec[i]);
    else
      set_vec[0] = set_vec[0].intersect(set_vec[i]);
  }
  return set.intersect(set_vec[0]);
}

void OpTypeCollector::Collect() { this->Visit(stmt_); }

void OpTypeCollector::AnalyzeOpTemplate() {
  std::string concated_op_type;
  for (auto it : provides_ana_) {
    std::vector<ProvideEntry> pes = it.second;
    for (auto pe : pes) {
      concated_op_type += pe.basic_op_type + ",";
      if (scop_info_.user_config_.GetTarget() == TARGET_CUDA &&
          pe.basic_op_type.find(AT_TRANSPOSE) != std::string::npos &&
          pe.basic_op_type.find(AT_ELEMWISE) != std::string::npos) {
        AnalyzeGemmAxes(pe);
      }
    }
  }
  if (scop_info_.analysis_result_.GetOpTemplate() != Template::DEFAULT) {
    return;
  }

  if (concated_op_type.find(AT_CALL) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::EXTERN_CALL);
  } else if (concated_op_type.find(AT_REDUCE) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::REDUCTION);
  } else if (concated_op_type.find(AT_TRANSPOSE) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::TRANSPOSE_OP);
  } else if (concated_op_type.find(AT_PAD) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::PAD_OP);
  } else if (concated_op_type.find(AT_BROADCAST) != std::string::npos ||
             concated_op_type.find(AT_TRANSFORM) != std::string::npos) {
    scop_info_.analysis_result_.SetOpTemplate(Template::BROADCAST_OP);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::PURE_ELEM);
  }
}

void OpTypeCollector::WriteToScopInfo() {
  for (auto it : provides_ana_) {
    auto cur_loop = it.first;
    auto provs = it.second;
    for (auto prov : provs) {
      scop_info_.analysis_result_.RecordProvideAnalysis(cur_loop, prov);
    }
  }
}

void OpTypeCollector::Dump() {
  LOG(INFO) << "OP TEMPLATE = "
            << scop_info_.analysis_result_.ShowOpTemplate(scop_info_.analysis_result_.GetOpTemplate());
  for (auto it : provides_ana_) {
    auto loop = it.first;
    auto provs = it.second;
    LOG(INFO) << "Under loop " << loop->loop_var->name_hint;
    for (auto prov : provs) {
      LOG(INFO) << "[DST] " << prov.dst.name << " [OPTYPE] " << prov.basic_op_type;
    }
  }
}

void OpTypeCollector::Visit_(const AttrStmt *op) {
  cur_attr_ = op;
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const Realize *op) {
  local_buf_.insert(op->func->func_name());
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const Provide *op) {
  AnalyzeProvide(op);
  IRVisitor::Visit_(op);
}

void OpTypeCollector::Visit_(const For *op) {
  loop_count_ += 1;
  cur_loop_ = op;
  cur_band_.emplace_back(cur_loop_);
  IRVisitor::Visit_(op);
  cur_loop_ = op;
  loop_count_ -= 1;
  // end of an outer band
  if (loop_count_ == 0) {
    band_count_ += 1;
    cur_band_.clear();
  }
}

void OpTypeCollector::Visit_(const IfThenElse *op) {
  cur_if_ = op;
  IRVisitor::Visit_(op);
  cur_if_ = op;
}

void OpTypeCollector::AnalyzeProvide(const Provide *op) {
  if (cur_loop_ == nullptr) return;
  ProvideEntry prov;
  std::string basic_op_type = "";
  std::vector<TensorEntry> src_tensor;
  TensorEntry dst_tensor;
  std::vector<const Call *> src_call;
  auto GetSrc = [&, this](const NodeRef &op) {
    if (const auto call = op.as<Call>()) {
      if (call->call_type == Call::Extern && scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
        basic_op_type = AT_CALL;
      } else {
        src_call.emplace_back(call);
      }
    } else if (op.as<Select>() && scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      basic_op_type = AT_CALL;
    }
  };
  air::ir::PostOrderVisit(op->value, GetSrc);

  for (auto call : src_call) {
    TensorEntry tensor;
    tensor.name = call->name;
    // get variable names
    for (auto arg : call->args) {
      VarNames vname;
      vname = VisitVarNames(arg, vname);
      tensor.var_names.emplace_back(vname);
    }
    tensor = MatchLoopByName(tensor);
    tensor.args = call->args;
    tensor.band_index = band_count_;
    tensor.type_byte = call->type.bytes();
    src_tensor.emplace_back(tensor);
  }

  auto src_length = static_cast<int>(src_tensor.size());
  for (auto st : src_tensor) {
    if (st.name == "mad" || st.name == LOAD_IM2COL) {
      basic_op_type = "SP_CALL";
    }
  }
  if (scop_info_.user_config_.GetTarget() == TARGET_CCE && src_length > 2 && basic_op_type != "SP_CALL") {
    LOG(WARNING) << "Detect provide has " << src_tensor.size() << " source tensors.";
    LOG(WARNING) << "After ToThreeAddress pass, number of ops in src should be less than 2.";
  }
  dst_tensor.name = op->func->func_name();
  for (auto arg : op->args) {
    VarNames vname;
    vname = VisitVarNames(arg, vname);
    dst_tensor.var_names.emplace_back(vname);
  }
  dst_tensor = MatchLoopByName(dst_tensor);
  dst_tensor.args = op->args;
  dst_tensor.band_index = band_count_;
  dst_tensor.type_byte = scop_info_.user_config_.GetDataBytes(dst_tensor.name);
  prov.basic_op_type = basic_op_type.empty() ? GetBasicOpType(dst_tensor, src_tensor) : basic_op_type;
  prov.band_index = band_count_;
  prov.src = src_tensor;
  prov.dst = dst_tensor;
  prov.op = op;
  prov.cond = cur_if_;
  provides_ana_[cur_loop_].emplace_back(prov);
}

void OpTypeCollector::AnalyzeGemmAxes(const ProvideEntry &pe) {
  VarNames mx_c, mx_a, mx_b;
  int index_a = -1;
  int index_b = -1;
  auto EmplaceVarsInTensor = [](TensorEntry tensor, VarNames &var_list) -> void {
    for (const auto &vars_i : tensor.var_names) {
      for (const auto &name : vars_i) {
        if (IsNum(name)) {
          continue;
        }
        var_list.emplace_back(name);
      }
    }
  };

  // Visit source tensors to fill mx_a and mx_b. Also, we need to check whether this provide stmt
  // is in `C = C + A * B` form and directly return if the form is broken.
  if (pe.src.size() != 3U) {
    return;
  }
  EmplaceVarsInTensor(pe.dst, mx_c);
  bool found_c = false;
  for (size_t i = 0; i < pe.src.size(); ++i) {
    auto src = pe.src[i];
    if (src.name == pe.dst.name) {
      VarNames src_c;
      EmplaceVarsInTensor(src, src_c);
      if (src_c.size() != mx_c.size()) {
        return;
      }
      for (size_t i = 0; i < src_c.size(); ++i) {
        if (src_c[i] != mx_c[i]) {
          return;
        }
      }
      found_c = true;
    } else if (index_a == -1) {
      EmplaceVarsInTensor(src, mx_a);
      index_a = i;
    } else if (index_b == -1) {
      EmplaceVarsInTensor(src, mx_b);
      index_b = i;
    } else {
      return;
    }
  }
  if (!found_c || mx_a.empty()) {
    return;
  }

  // construct relationship between loop indices and loop type(b/m/n/k) and mark axis with corresponding attribute
  std::string attr_key = "";
  if (scop_info_.user_config_.GetEnableConvTensorCore()) {
    scop_info_.analysis_result_.SetOpTemplate(Template::CONV);
  } else {
    scop_info_.analysis_result_.SetOpTemplate(Template::MATMUL);
  }
}

// Match variable to loop by name since the name in current band is unique.
// If name is not unique, it means the axis is separated into different chunks
// and they will need same alignment rule.
TensorEntry OpTypeCollector::MatchLoopByName(TensorEntry tensor) {
  std::unordered_map<size_t, std::vector<const For *>> loop_pos;
  for (size_t p = 0; p < tensor.var_names.size(); ++p) {
    for (auto name : tensor.var_names[p]) {
      for (auto i = static_cast<int>(cur_band_.size()) - 1; i >= 0; i--) {
        const For *loop = cur_band_[i];
        if (loop != nullptr && loop->loop_var.get()->name_hint == name) {
          loop_pos[p].emplace_back(loop);
          break;
        }
      }
    }
  }
  tensor.loops = loop_pos;
  return tensor;
}

std::string OpTypeCollector::GetBasicOpType(const TensorEntry dst, const std::vector<TensorEntry> &srcs) {
  auto CountUniqueLoopName = [](std::vector<VarNames> var_names) -> size_t {
    std::unordered_set<std::string> uni_name;
    for (auto names : var_names) {
      for (auto n : names) {
        if (IsNum(n)) {
          continue;
        }
        uni_name.insert(n);
      }
    }
    return uni_name.size();
  };

  auto GetSingleOpType = [&CountUniqueLoopName, this](const TensorEntry d, const TensorEntry s) -> std::string {
    auto dst_vars = d.var_names;
    auto src_vars = s.var_names;
    auto dst_vars_size = CountUniqueLoopName(dst_vars);
    auto src_vars_size = CountUniqueLoopName(src_vars);
    std::string type = "";
    if (scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      if (this->local_buf_.find(s.name) == this->local_buf_.end()) type += "DMA2_";
      if (this->local_buf_.find(d.name) == this->local_buf_.end()) type += "DMA3_";
    }
    if (src_vars_size == 0 && scop_info_.user_config_.GetTarget() != TARGET_CUDA) {
      return type + "SP_CALL";
    }
    if (dst_vars_size < src_vars_size) {
      if (d.loops.size() < s.loops.size() && d.name != s.name) {
        // A[i,0] = B[i,j]
        return type + AT_REDUCE;
      } else {
        return type + "UNKNOWN";
      }
    } else if (dst_vars_size > src_vars_size) {
      // A[i,j] = B[i,0]
      return type + AT_BROADCAST;
    } else {
      // Index size is the same.
      while (!dst_vars.empty() && !src_vars.empty()) {
        // detect transpose first
        VarNames dst_name = dst_vars.back();
        VarNames src_name = src_vars.back();
        dst_vars.pop_back();
        src_vars.pop_back();
        VarNames dst_pure_name;
        VarNames src_pure_name;
        for (auto n : dst_name) {
          if (!IsNum(n)) {
            dst_pure_name.emplace_back(n);
          }
        }
        for (auto n : src_name) {
          if (!IsNum(n)) {
            src_pure_name.emplace_back(n);
          }
        }
        if (dst_pure_name.size() == src_pure_name.size()) {
          for (size_t j = 0; j < dst_pure_name.size(); ++j) {
            if (dst_pure_name[j] != src_pure_name[j]) {
              return type + AT_TRANSPOSE;
            }
          }
        }
      }
      if (d.loops.size() == s.loops.size()) {
        // A[i,j] = B[i,j]
        return type + AT_ELEMWISE;
      } else {
        // AutoFused case in cuda
        if (scop_info_.user_config_.GetTarget() == TARGET_CUDA && d.args.size() == s.args.size()) {
          for (size_t i = 0; i < d.args.size(); ++i) {
            if (Equal(d.args[i], s.args[i])) {
              continue;
            }
            if (arith_ana_.CanProve(s.args[i] == 0)) {
              // A[floordiv(i, 128), floormod(i, 128)] = B[0, floormod(i, 128)]
              type += AT_BROADCAST;
            } else if (arith_ana_.CanProve(d.args[i] == 0)) {
              // A[0, floormod(i, 128)] = B[floordiv(i, 128), floormod(i, 128)]
              type += AT_REDUCE;
            } else {
              type += AT_TRANSFORM;
            }
            type += ("|" + std::to_string(i) + "_");
          }
        } else {
          // A[0,i] = B[i,i]
          type += AT_TRANSFORM;
        }
        return type;
      }
    }
    return type;
  };

  std::string basic_op_type = "";
  bool is_unpad = (dst.name.find("unpad") != std::string::npos) || (dst.name.find("Unpad") != std::string::npos);
  bool is_pad = (dst.name.find("pad") != std::string::npos || dst.name.find("Pad") != std::string::npos);
  if (!is_unpad && is_pad) {
    basic_op_type += AT_PAD;
    basic_op_type += "_";
  }
  if (srcs.empty()) {
    // Dst = const
    if (this->local_buf_.find(dst.name) == this->local_buf_.end()) basic_op_type += "DMA3_";
    basic_op_type += "INIT";
  } else {
    for (auto s : srcs) {
      basic_op_type += GetSingleOpType(dst, s);
      basic_op_type += "_";
    }
  }
  return basic_op_type;
}

class GetBatchNum final : public IRVisitor {
 public:
  GetBatchNum(const NodeRef node) { IRVisitor::Visit(node); }
  ~GetBatchNum() override = default;

  void Visit_(const Call *op) final {
    if (visited_axis.size() == 0) {
      batch_axis_num = op->args.size();
      for (size_t i = 0; i < op->args.size(); i++) {
        visited_axis.push_back(op->args[i]);
      }
    } else {
      unsigned int same_axis_num = 0;
      for (size_t i = 0; (i < op->args.size()) && (i < visited_axis.size()); i++) {
        if (Equal(op->args[i], visited_axis[i])) {
          same_axis_num++;
        } else {
          break;
        }
      }
      if (batch_axis_num > same_axis_num) batch_axis_num = same_axis_num;
    }
    IRVisitor::Visit_(op);
  }

 public:
  std::vector<Expr> visited_axis;
  unsigned int batch_axis_num{0};
};

class ExtractAxisUsed : public IRVisitor {
 public:
  ExtractAxisUsed(std::vector<const Variable *> &vec, std::unordered_set<std::string> &red) : vec_(vec), red_(red) {}
  ~ExtractAxisUsed() override = default;
  void Visit_(const Variable *op) final {
    if (red_.count(op->name_hint) == 1) {
      vec_.push_back(op);
    }
    IRVisitor::Visit_(op);
  }

  void Run(const NodeRef &ref) { IRVisitor::Visit(ref); }

 private:
  std::vector<const Variable *> &vec_;
  std::unordered_set<std::string> &red_;
};

class ExtractAxisNotUsed : public IRVisitor {
 public:
  ExtractAxisNotUsed(std::vector<const Variable *> &vec, std::unordered_set<std::string> &red) : vec_(vec), red_(red) {}
  ~ExtractAxisNotUsed() override = default;
  void Visit_(const Variable *op) final {
    if (red_.count(op->name_hint) == 0) {
      vec_.push_back(op);
    }
    IRVisitor::Visit_(op);
  }

  void Run(const NodeRef &ref) { IRVisitor::Visit(ref); }

 private:
  std::vector<const Variable *> &vec_;
  std::unordered_set<std::string> &red_;
};

class ReduceInfoCollector {
 public:
  ReduceInfoCollector(ScopInfo &scop_info) : scop_info_(scop_info) {}
  ~ReduceInfoCollector() = default;
  void Run() {
    auto op_type = scop_info_.analysis_result_.ShowOpTemplate(scop_info_.analysis_result_.GetOpTemplate());
    if (op_type == "REDUCTION" || op_type == "ALL_REDUCE" || op_type == "BITWISE_REDUCTION") {
      RecordReduceInfo();
    } else if (op_type == "MATMUL" || op_type == "CONV") {
      RecordMatmulInfo();
    }
  }

  void RecordReduceInfo() {
    for (auto &rm : scop_info_.analysis_result_.GetReduceMap()) {
      auto op = rm.first;
      auto reduce_iter_var = rm.second;
      isl::id red_id;
      for (auto &c : scop_info_.analysis_result_.GetStatementMap()) {
        if (op == c.second) {
          red_id = c.first;
          break;
        }
      }

      OperatorDomainSpace op_domain;
      auto dm = scop_info_.analysis_result_.GetOperatorDomainMap();
      if (dm.count(red_id)) {
        op_domain = dm[red_id];
      }

      std::unordered_set<std::string> reduce_attrs;
      for (auto &r : reduce_iter_var) {
        reduce_attrs.insert(r->var->name_hint);
      }

      auto args = op->args;
      std::vector<const Variable *> vars_not_reduce;
      for (auto &a : args) {
        ExtractAxisNotUsed(vars_not_reduce, reduce_attrs).Run(a);
      }

      bool is_all_reduce = vars_not_reduce.size() == 0;
      scop_info_.user_config_.SetTileCheckCoincident(!is_all_reduce);
      isl::ctx ctx = op_domain.tuple.ctx();

      isl::aff_list aff_list = isl::aff_list(ctx, 0);
      for (auto id : op_domain.tuple.get_id_list()) {
        if (reduce_attrs.count(id.get_name()) == 1) {
          continue;
        }
        isl::aff aff = isl::aff::param_on_domain(op_domain.param_space, id);
        aff = aff.unbind_params_insert_domain(op_domain.tuple);
        aff_list = aff_list.add(aff);
      }
      isl::space op_domain_space = op_domain.tuple.get_space();
      isl::space space = op_domain_space.params().add_named_tuple_id_ui(red_id, aff_list.size());
      space = op_domain_space.product(space).unwrap();
      isl::union_map upa = isl::union_map(isl::map(isl::multi_aff(space, aff_list)));

      ReduceTensorInfo reduce_tensor_info;
      reduce_tensor_info.stmt_node = op;
      reduce_tensor_info.stmt_map = upa;
      scop_info_.analysis_result_.RecordReduceTensorInfoMap(red_id, reduce_tensor_info);
      auto type = scop_info_.analysis_result_.GetReduceOpType(red_id);
      if (AkgSupportedReduceOp.count(type) != 0) {
        reduce_tensor_info.write_tensor_name = op->func->func_name();
        SetReduceInitValue(reduce_tensor_info);
        SetReduceWriteDataType(reduce_tensor_info);
        scop_info_.analysis_result_.UpdateReduceTensorInfoMap(red_id, reduce_tensor_info);

        std::string reduce_direction;
        PostOrderVisit(op->value, [&reduce_direction, &reduce_attrs, op](const NodeRef &node) -> void {
          if (reduce_direction == Y_DIRECTION) {
            return;
          }
          auto call = node.as<Call>();
          if (call == nullptr || call->call_type != Call::CallType::Halide ||
              call->func->func_name() == op->func->func_name() || call->args.empty()) {
            return;
          }
          int call_size = static_cast<int>(call->args.size());
          int reduce_position = -1;
          int non_variable_count = 0;
          bool is_continuous = true;
          for (int i = call_size - 1; i >= 0; --i) {
            auto last_axis = call->args[i];
            auto mod = last_axis.as<FloorMod>();
            auto var = mod != nullptr ? mod->a.as<Variable>() : last_axis.as<Variable>();
            if (var != nullptr) {
              reduce_position = reduce_attrs.count(var->name_hint) ? i : reduce_position;
              is_continuous = false;
            } else if (var == nullptr && is_continuous) {
              ++non_variable_count;
            }
          }
          if (reduce_position == -1) {
            return;
          }
          if (reduce_position == call_size - non_variable_count - 1) {
            reduce_direction = X_DIRECTION;
          } else {
            reduce_direction = Y_DIRECTION;
          }
        });
        if (reduce_direction.empty()) {
          LOG(WARNING) << "Cannot identify reduce direction for stmt " << red_id;
        }
        scop_info_.analysis_result_.RecordReduceDirection(reduce_direction);
      } else {
        scop_info_.user_config_.SetEnableAkgReduceLib(false);
      }

      scop_info_.user_config_.SetEnableMatmul(false);
      scop_info_.user_config_.SetEnableTensorCore(false);
      scop_info_.user_config_.SetEnableTensorCoreUsePoly(false);
    }
  }

  void RecordMatmulInfo() {
    for (auto &rm : scop_info_.analysis_result_.GetReduceMap()) {
      auto op = rm.first;
      auto reduce_iter_var = rm.second;
      isl::id red_id;
      for (auto &c : scop_info_.analysis_result_.GetStatementMap()) {
        if (op == c.second) {
          red_id = c.first;
          break;
        }
      }

      OperatorDomainSpace op_domain;
      auto dm = scop_info_.analysis_result_.GetOperatorDomainMap();
      if (dm.count(red_id)) {
        op_domain = dm[red_id];
      }

      std::unordered_set<std::string> reduce_attrs;
      for (auto &r : reduce_iter_var) {
        reduce_attrs.insert(r->var->name_hint);
      }

      auto args = op->args;
      std::vector<const Variable *> vars_not_reduce;
      std::vector<const Variable *> vars_reduce;
      for (auto &a : args) {
        ExtractAxisNotUsed(vars_not_reduce, reduce_attrs).Run(a);
      }

      ExtractAxisUsed(vars_reduce, reduce_attrs).Run(op->value);

      GetBatchNum get_batch_num(op->value);
      scop_info_.analysis_result_.RecordNotReduceAxisForMatmul(vars_not_reduce);
      scop_info_.analysis_result_.RecordReduceAxisForMatmul(vars_reduce);
      scop_info_.analysis_result_.RecordBatchAxisNumForMatmul(get_batch_num.batch_axis_num);

      isl::ctx ctx = op_domain.tuple.ctx();

      isl::aff_list aff_list = isl::aff_list(ctx, 0);
      for (auto id : op_domain.tuple.get_id_list()) {
        if (reduce_attrs.count(id.get_name()) == 1) {
          continue;
        }
        isl::aff aff = isl::aff::param_on_domain(op_domain.param_space, id);
        aff = aff.unbind_params_insert_domain(op_domain.tuple);
        aff_list = aff_list.add(aff);
      }
      isl::space op_domain_space = op_domain.tuple.get_space();
      isl::space space = op_domain_space.params().add_named_tuple_id_ui(red_id, aff_list.size());
      space = op_domain_space.product(space).unwrap();
      isl::union_map upa = isl::union_map(isl::map(isl::multi_aff(space, aff_list)));

      if (CheckMatmul(op)) {
        scop_info_.user_config_.SetEnableMatmul(true);
        scop_info_.user_config_.SetEnableTensorCore(true);
        scop_info_.user_config_.SetEnableTensorCoreUsePoly(true);
        scop_info_.user_config_.SetEnableAkgReduceLib(false);
        // Default vectorization access mode (128 bits).
        if (scop_info_.user_config_.GetVectorLoadType() == 0) {
          scop_info_.user_config_.SetVectorLoadType(PROMOTE_VECTORIZATION_BIT);
        }
        RecordMatrixInfoForFuse(op);
      } else {
        scop_info_.user_config_.SetEnableAkgReduceLib(false);
        scop_info_.user_config_.SetEnableMatmul(false);
        scop_info_.user_config_.SetEnableTensorCore(false);
        scop_info_.user_config_.SetEnableTensorCoreUsePoly(false);
      }
    }
  }

  void RecordMatrixInfoForFuse(const Provide *op) {
    auto matmul_map = scop_info_.analysis_result_.GetMatrixMatmulMap();
    if (!matmul_map.empty()) {
      std::string accumulator = "";
      auto mp = GetMatmulTensorsName(scop_info_);
      if (mp.find(MATRIX_C) != mp.end()) {
        accumulator = mp[MATRIX_C];
      }
      CHECK(accumulator != "") << "MatMul info not enough!";
      Array<Expr> elem_tensors = GetBinaryOpExprChildren(op->value);
      if (!elem_tensors.empty()) {
        auto left = elem_tensors[0].as<Call>();
        auto right = elem_tensors[1].as<Call>();
        if ((left || right) &&
            (matmul_map.find(left->name) != matmul_map.end() || matmul_map.find(right->name) != matmul_map.end())) {
          if (op->func->func_name() != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(op->func->func_name(), MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(op->func->func_name(), ROW_MAJOR);
          }
          if (left && left->name != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(left->name, MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(left->name, ROW_MAJOR);
          }
          if (right && right->name != accumulator) {
            scop_info_.analysis_result_.RecordMatrixMatmulMap(right->name, MATRIX_ELSE);
            scop_info_.analysis_result_.RecordMatrixMatmulMajor(right->name, ROW_MAJOR);
          }
        }
      }
    }
  }
  void SetReduceWriteDataType(ReduceTensorInfo &reduce_tensor_info) {
    auto init_value = reduce_tensor_info.init_value;
    if (!init_value.defined()) {
      return;
    }
    reduce_tensor_info.write_dtype = init_value.type();
  }

  void SetReduceInitValue(ReduceTensorInfo &reduce_tensor_info) {
    Expr init_value;
    if (!reduce_tensor_info.stmt_node->IsInstance<Provide>()) {
      return;
    }
    auto provide = static_cast<const Provide *>(reduce_tensor_info.stmt_node);
    if (provide == nullptr) {
      return;
    }
    auto red_tensor_name = provide->func->func_name();
    for (auto it : scop_info_.analysis_result_.GetStatementMap()) {
      if (it.second->IsInstance<Provide>()) {
        auto prev_provide = static_cast<const Provide *>(it.second);
        if (prev_provide == nullptr || prev_provide == provide || prev_provide->func->func_name() != red_tensor_name) {
          continue;
        }
        init_value = prev_provide->value;
        scop_info_.analysis_result_.RecordReduceInitIds(it.first);
        break;
      }
    }
    if (!init_value.defined()) {
      return;
    }
    reduce_tensor_info.init_value = init_value;
  }

  bool GetRowColInfo(const Provide *op) {
    auto axis = scop_info_.analysis_result_.GetNotReduceAxisForMatmul();
    auto reduce_axis = scop_info_.analysis_result_.GetReduceAxisForMatmul();
    auto batch_num_axis = scop_info_.analysis_result_.GetBatchAxisNumForMatmul();
    if (axis.size() < 2 || reduce_axis.size() < 1 || axis.size() <= batch_num_axis) return false;

    const Variable *axis_var[2];
    const Variable *reduce_axis_var;
    axis_var[0] = axis[batch_num_axis];
    axis_var[1] = axis.back();
    reduce_axis_var = reduce_axis.back();

    class CollectInfoOfBody : public IRVisitor {
     public:
      CollectInfoOfBody() {}
      using IRVisitor::Visit_;

      void Visit_(const Call *op) final {
        IRVisitor::Visit_(op);
        args_.insert(std::make_pair(op->name, op->args));
      }

      std::unordered_map<std::string, Array<Expr>> GetArgs() { return args_; }

     private:
      std::unordered_map<std::string, Array<Expr>> args_;
    } collect_info_of_body;

    auto right = op->value;
    auto add_op = right.as<Add>();
    CHECK(add_op);
    auto tensor_c = add_op->a.as<Call>();
    if (tensor_c == nullptr) return false;

    Type tensor_c_type;
    if (!IsExistTensor(tensor_c->name, tensor_c_type)) return false;

    collect_info_of_body.Visit(add_op->b);

    for (auto iter : collect_info_of_body.GetArgs()) {
      auto name = iter.first;
      auto args = iter.second;
      if (args.size() < 2) continue;

      const Variable *var0 = args[batch_num_axis].as<Variable>();
      const Variable *var1 = args[args.size() - 1].as<Variable>();
      if (var0 == nullptr || var1 == nullptr) continue;

      std::string major;
      if ((var0 == reduce_axis_var) && (var1 == axis_var[0])) {
        major = COL_MAJOR;
      } else if ((var0 == reduce_axis_var) && (var1 == axis_var[1])) {
        major = ROW_MAJOR;
      } else if ((var0 == axis_var[0]) && (var1 == reduce_axis_var)) {
        major = ROW_MAJOR;
      } else if ((var0 == axis_var[1]) && (var1 == reduce_axis_var)) {
        major = COL_MAJOR;
      } else {
        return false;
      }
      scop_info_.analysis_result_.RecordMatrixMatmulMajor(name, major);
    }
    scop_info_.analysis_result_.RecordMatrixMatmulMajor(op->func->func_name(), ROW_MAJOR);
    return true;
  }

  bool IsExistTensor(const std::string &tensor_name, Type &tensor_type) {
    auto all_tensors = scop_info_.user_config_.GetRealizeTensors();
    for (auto it : all_tensors) {
      if (it->op->name == tensor_name) {
        tensor_type = it->dtype;
        return true;
      }
    }
    auto orig_binds = scop_info_.user_config_.GetOriginBind();
    for (auto it : orig_binds) {
      if (it.first->op->name == tensor_name) {
        tensor_type = it.first->dtype;
        return true;
      }
    }
    return false;
  }

  std::string GetTensorName(Expr tensor_data, bool &enable_tensor_core) {
    std::string tensor_name = "";
    if (tensor_data.as<Call>()) {
      auto tensor_data_p = tensor_data.as<Call>();
      Type tensor_type;
      if (!IsExistTensor(tensor_data_p->name, tensor_type)) {
        return tensor_name;
      }
      if ((tensor_type != Float(16)) && (tensor_type != Int(8))) {
        enable_tensor_core = false;
      }
      tensor_name = tensor_data_p->name;
    } else if (tensor_data.as<Cast>() &&
               ((tensor_data.as<Cast>()->type == Float(16)) || (tensor_data.as<Cast>()->type == Int(8)))) {
      auto tensor_data_p = tensor_data.as<Cast>();
      auto value = tensor_data_p->value;
      tensor_name = value.as<Call>()->name;
      scop_info_.analysis_result_.RecordCastTensors(tensor_name);
    }
    return tensor_name;
  }

  bool CheckMatmul(const Provide *op) {
    if (!scop_info_.user_config_.GetEnableMatmul()) {
      return false;
    }

    // C + A * B
    auto add_op = op->value.as<Add>();
    if (add_op == nullptr) {
      return false;
    }

    auto tensor_c = add_op->a.as<Call>();
    if (tensor_c == nullptr) {
      return false;
    }
    Type tensor_c_type;
    if (!IsExistTensor(tensor_c->name, tensor_c_type)) {
      return false;
    }
    if (tensor_c_type != Float(16) && tensor_c_type != Float(32) && tensor_c_type != Int(32)) {
      return false;
    }

    auto mul_op = akg::common::SplitCast(add_op->b, tensor_c_type).as<Mul>();
    if (mul_op == nullptr) {
      return false;
    }

    auto tensor_a = akg::common::SplitCast(mul_op->a, tensor_c_type);
    auto tensor_b = akg::common::SplitCast(mul_op->b, tensor_c_type);
    bool enable_tensor_core = true;
    std::string tensor_a_name = GetTensorName(tensor_a, enable_tensor_core);
    std::string tensor_b_name = GetTensorName(tensor_b, enable_tensor_core);
    if (!enable_tensor_core) {
      return false;
    }

    if (tensor_a_name.empty() || tensor_b_name.empty()) {
      return false;
    }

    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_a_name, MATRIX_A);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_b_name, MATRIX_B);
    scop_info_.analysis_result_.RecordMatrixMatmulMap(tensor_c->name, MATRIX_C);

    bool ret = GetRowColInfo(op);
    if (!ret) {
      return false;
    }

    SetMmaModeForTensor(tensor_a_name, tensor_b_name);

    if (tensor_c_type == Float(16)) {
      std::string shared_tensors = tensor_a_name + " " + tensor_b_name + " " + tensor_c->name;
      scop_info_.user_config_.SetSharedTensors(shared_tensors);
    }

    return true;
  }

  void SetMmaModeForTensor(const std::string &tensor_a_name, const std::string &tensor_b_name) {
    std::string custom_dim = scop_info_.user_config_.GetBDim();
    if (!custom_dim.empty() && !scop_info_.user_config_.GetEnableConvTensorCore()) {
      const int each_axis_size = 4;
      const int m_axis_pos = 1;
      const int n_axis_pos = 2;
      const int k_axis_pos = 3;

      Mma mma;
      std::vector<std::string> dim_str = Split(custom_dim, " ");
      int batch_number = static_cast<int>(scop_info_.analysis_result_.GetBatchAxisNumForMatmul()) > 0 ? 1 : 0;
      int real_m_axis_pos = (m_axis_pos + batch_number) * each_axis_size - 1;
      int real_n_axis_pos = (n_axis_pos + batch_number) * each_axis_size - 1;
      int real_k_axis_pos = (k_axis_pos + batch_number) * each_axis_size - 1;
      mma.m = static_cast<int>(WrappedStrtol(dim_str[real_m_axis_pos]));
      mma.n = static_cast<int>(WrappedStrtol(dim_str[real_n_axis_pos]));
      mma.k = static_cast<int>(WrappedStrtol(dim_str[real_k_axis_pos]));

      scop_info_.analysis_result_.SetMmaMode(mma);
      return;
    }

    Mma mma;
    auto matrix_a_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_a_name];
    auto matrix_b_major = scop_info_.analysis_result_.GetMatrixMatmulMajor()[tensor_b_name];
    if (matrix_a_major == COL_MAJOR && matrix_b_major == ROW_MAJOR) {
      mma = {32, 32, 4};
    } else {
      mma = {16, 16, 8};
    }
    scop_info_.analysis_result_.SetMmaMode(mma);
  }

 private:
  ScopInfo &scop_info_;
};  // namespace poly

isl::schedule MakeScheduleTree(const isl::space &param_space, isl::set param_set, const Stmt &stmt,
                               ScopInfo &scop_info) {
  scop_info.analysis_result_.RecordReads(isl::union_map::empty(param_space));
  scop_info.analysis_result_.RecordWrites(isl::union_map::empty(param_space));
  scop_info.analysis_result_.RecordCopyin(isl::union_map::empty(param_space));
  scop_info.analysis_result_.RecordFakeCopyin(isl::union_map::empty(param_space));
  scop_info.analysis_result_.RecordBindCopyin(isl::union_map::empty(param_space));
  scop_info.analysis_result_.RecordTransferStmt(isl::union_set::empty(param_space));
  scop_info.analysis_result_.RecordInnerBandDependency(isl::union_map::empty(param_space));
  isl::set set = isl::set::universe(param_space);

  set = set.intersect_params(param_set);
  isl::id_list outer(param_space.ctx(), 0);
  ssize_t macro_stmt = -1;
  auto schedule = MakeScheduleTreeHelper(stmt, scop_info, set, outer, macro_stmt);
  OpTypeCollector op_type_collector(scop_info, stmt);
  op_type_collector.Collect();
  op_type_collector.AnalyzeOpTemplate();
  op_type_collector.WriteToScopInfo();
  op_type_collector.Dump();
  if (scop_info.user_config_.GetTarget() == TARGET_CUDA) {
    ReduceInfoCollector reduce_info_coll(scop_info);
    reduce_info_coll.Run();
  }
  return schedule;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
