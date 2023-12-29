/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "build_module.h"
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
    } else if (call->name.rfind("tot_op_", 0) == 0) {
      auto it = POLY_SUPPORTED_OPS.find("tot_op");
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

VarNames VisitVarNames(const air::Expr &arg, VarNames var_names, bool add_num, bool visit_tot) {
  if (const auto var = arg.as<air::ir::Variable>()) {
    var_names.emplace_back(var->name_hint);
  } else if (const auto sub = arg.as<air::ir::Sub>()) {
    var_names = VisitVarNames(sub->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(sub->b, var_names, add_num, visit_tot);
  } else if (const auto add = arg.as<air::ir::Add>()) {
    var_names = VisitVarNames(add->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(add->b, var_names, add_num, visit_tot);
  } else if (const auto mul = arg.as<air::ir::Mul>()) {
    var_names = VisitVarNames(mul->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(mul->b, var_names, add_num, visit_tot);
  } else if (const auto div = arg.as<air::ir::Div>()) {
    var_names = VisitVarNames(div->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(div->b, var_names, add_num, visit_tot);
  } else if (const auto mod = arg.as<air::ir::Mod>()) {
    var_names = VisitVarNames(mod->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(mod->b, var_names, add_num, visit_tot);
  } else if (const auto int_imm = arg.as<air::ir::IntImm>()) {
    if (add_num) {
      var_names.emplace_back(std::to_string(int_imm->value));
    }
  } else if (const auto f_mod = arg.as<air::ir::FloorMod>()) {
    var_names = VisitVarNames(f_mod->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(f_mod->b, var_names, add_num, visit_tot);
  } else if (const auto f_div = arg.as<air::ir::FloorDiv>()) {
    var_names = VisitVarNames(f_div->a, var_names, add_num, visit_tot);
    var_names = VisitVarNames(f_div->b, var_names, add_num, visit_tot);
  } else if (const auto call = arg.as<air::ir::Call>()) {
    if (visit_tot) {
      for (auto call_arg: call->args) {
        var_names = VisitVarNames(call_arg, var_names, add_num, visit_tot);
      }
    }
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
  scop_info.analysis_result_.for_type_.clear();
  auto schedule = MakeScheduleTreeHelper(stmt, scop_info, set, outer, macro_stmt);
  return schedule;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
