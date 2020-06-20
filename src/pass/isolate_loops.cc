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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include "pass/utils.h"
#include "pass/expr_alg_simplify.h"

namespace akg {
namespace ir {
using std::make_pair;
class TiledForInfo {
 public:
  Var loop_var;
  Expr shape;
  Expr tile;
  Expr num_full_tiles;
  Expr len_partial_tile;
  Expr loop_min;
  bool allow_tail_cond{true};  // Is "if (ccX_tail > 0) {...}" legal?
};

bool IsInnerLoop(const Expr &extent, const TiledForInfo &for_info) {
  // Example inner loop: min(T1_0_0, (I0 - (cc0*T1_0_0)))
  Expr pattern = Min::make(for_info.tile, (for_info.shape - (for_info.loop_var * for_info.tile)));
  return ExprPatternMatch(extent, pattern);
}

bool IsShiftedInnerLoop(const Expr &min, const Expr &extent, const TiledForInfo &for_info) {
  // Example shifted inner loop:
  // for (cc3, 0, (min((T1_0_1 - 1), ((I1 + 10699) - (cc1*T1_0_1))) + 1))
  if (min.as<IntImm>() && min.as<IntImm>()->value == 0) {
    Expr pattern =
      Min::make((for_info.tile - 1), ((for_info.shape + Var("int")) - (for_info.loop_var * for_info.tile))) + 1;
    if (ExprPatternMatch(extent, pattern)) {
      return true;
    }
  }

  // Example shifted inner loop:
  // for (cc3, ((107*cc1) - 10700), ((min(I1, ((cc1*T1_0_1) - 10593)) + 10700) - (cc1*T1_0_1)))
  Expr min_pattern = (Var("int") * for_info.loop_var) - Var("int");
  Expr extent_pattern = (Min::make(for_info.shape, ((for_info.loop_var * for_info.tile) - Var("int")) + Var("int")) -
                         (for_info.loop_var * for_info.tile));
  return ExprPatternMatch(min, min_pattern) && ExprPatternMatch(extent, extent_pattern);
}

bool FindInnerLoop(const Stmt &stmt, const TiledForInfo &for_info) {
  bool found = false;
  PostOrderVisit(stmt, [&for_info, &found](const NodeRef &node) {
    if (auto op = node.as<For>()) {
      if (IsInnerLoop(op->extent, for_info) || IsShiftedInnerLoop(op->min, op->extent, for_info)) {
        found = true;
      }
    }
  });
  return found;
}

class GenBodyStmt : public IRMutator {
 public:
  explicit GenBodyStmt(const TiledForInfo &for_info) : for_info_(for_info) {}
  ~GenBodyStmt() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (IsInnerLoop(op->extent, for_info_)) {
      Stmt body = Mutate(op->body);
      return For::make(op->loop_var, op->min, for_info_.tile, op->for_type, op->device_api, body);
    }
    if (IsShiftedInnerLoop(op->min, op->extent, for_info_)) {
      Stmt body = Mutate(op->body);
      return For::make(op->loop_var, 0, for_info_.tile, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  const TiledForInfo &for_info_;
};

class GenTailStmt : public AttrIRMutator {
 public:
  explicit GenTailStmt(const TiledForInfo &for_info) : for_info_(for_info) {}
  ~GenTailStmt() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (IsInnerLoop(op->extent, for_info_)) {
      Stmt body = Mutate(op->body);
      return For::make(op->loop_var, op->min, for_info_.len_partial_tile, op->for_type, op->device_api, body);
    }
    if (IsShiftedInnerLoop(op->min, op->extent, for_info_)) {
      Stmt body = Mutate(op->body);
      return For::make(op->loop_var, 0, for_info_.len_partial_tile, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (op == for_info_.loop_var.get()) {
      if (is_const_int(for_info_.loop_min, 0)) {
        return for_info_.num_full_tiles;
      } else {
        return for_info_.num_full_tiles + for_info_.loop_min;
      }
    }
    return e;
  }

  const TiledForInfo &for_info_;
};

bool ParseForInfo(const Expr &loop_min, const Expr &extent, TiledForInfo *for_info) {
  if (!loop_min.as<IntImm>()) return false;

  std::vector<Expr> matches;
  // match extent: (((I0 - 1)/T1_0_0) + 1)
  Expr pattern1 = Div::make(Var("any") - 1, Var("varOrInt")) + 1;
  if (ExprPatternMatch(extent, pattern1, &matches)) {
    CHECK_EQ(matches.size(), 2);
    for_info->shape = matches[0];
    for_info->tile = matches[1];
    for_info->allow_tail_cond = true;
    return true;
  }

  // match extent: ((I0/T1_0_0) + 1)
  Expr pattern2 = Div::make(Var("any"), Var("varOrInt")) + 1;
  if (ExprPatternMatch(extent, pattern2, &matches)) {
    CHECK_EQ(matches.size(), 2);
    for_info->shape = matches[0];
    for_info->tile = matches[1];
    for_info->allow_tail_cond = false;
    return true;
  }

  // match extent: (min(92, ((I1 - 1)/T1_0_1)) + 1)
  Expr pattern3 = Min::make(Var("int"), Div::make(Var("any") - 1, Var("varOrInt"))) + 1;
  if (ExprPatternMatch(extent, pattern3, &matches)) {
    CHECK_EQ(matches.size(), 3);
    for_info->shape = matches[1];
    for_info->tile = matches[2];
    for_info->allow_tail_cond = true;
    return true;
  }

  return false;
}

class WrapIfAroundFor : public IRMutator {
 public:
  explicit WrapIfAroundFor(const Expr &tail_cond) : tail_cond(tail_cond) {}
  ~WrapIfAroundFor() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final { return IfThenElse::make(tail_cond, s, Stmt()); }

  const Expr &tail_cond;
};

class IsolateLoopsMutator : public IRMutator {
 private:
  Map<Expr, Expr> mod_constraints_;
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key.find("IsolateConstraint") != std::string::npos) {
      if (const auto eq = op->value.as<EQ>()) {
        if (eq->b.as<IntImm>() != nullptr && eq->b.as<IntImm>()->value == 0) {
          if (const auto mod = eq->a.as<Mod>()) {
            mod_constraints_.Set(mod->a, mod->b);
          } else if (const auto fmod = eq->a.as<FloorMod>()) {
            mod_constraints_.Set(fmod->a, fmod->b);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    // example for loop: for (cc0, 0, (((I0 - 1)/T1_0_0) + 1))
    TiledForInfo for_info;
    for_info.loop_var = op->loop_var;
    for_info.loop_min = op->min;
    if (!ParseForInfo(op->min, op->extent, &for_info)) {
      return IRMutator::Mutate_(op, s);
    }
    if (!FindInnerLoop(op->body, for_info)) {
      return IRMutator::Mutate_(op, s);
    }
    Expr num_full_tiles = Div::make(for_info.shape, for_info.tile);
    Var var_num_full_tiles = Variable::make(num_full_tiles.type(), op->loop_var->name_hint + "_body");
    for_info.num_full_tiles = var_num_full_tiles;

    Stmt stmt = Mutate(op->body);
    Stmt body_stmt = GenBodyStmt(for_info).Mutate(stmt);
    Stmt full_tiles =
      For::make(op->loop_var, op->min, for_info.num_full_tiles, op->for_type, op->device_api, body_stmt);

    Map<Expr, Expr> for_info_dict;
    for_info_dict.Set(Expr("loop_var"), for_info.loop_var);
    for_info_dict.Set(Expr("tile_size"), for_info.tile);
    for_info_dict.Set(Expr("shape"), for_info.shape);
    for_info_dict.Set(Expr("full_tile_num"), for_info.num_full_tiles);

    Stmt body;
    bool need_tail = mod_constraints_.find(for_info.shape) == mod_constraints_.end();
    if (need_tail) {
      Expr len_partial_tile = for_info.shape - var_num_full_tiles * for_info.tile;
      Var var_len_partial_tile = Variable::make(len_partial_tile.type(), op->loop_var->name_hint + "_tail");
      for_info.len_partial_tile = var_len_partial_tile;
      Stmt partial_tiles = GenTailStmt(for_info).Mutate(stmt);
      Expr tail_cond = (for_info.len_partial_tile > 0);
      if (for_info.allow_tail_cond) {
        partial_tiles = IfThenElse::make(tail_cond, partial_tiles, Stmt());
      } else {
        partial_tiles = WrapIfAroundFor(tail_cond).Mutate(partial_tiles);
      }
      for_info_dict.Set(Expr("partial_tile_size"), for_info.len_partial_tile);
      Stmt block = Block::make(full_tiles, partial_tiles);
      Stmt let_tail_assert =
        AssertStmt::make(var_len_partial_tile < for_info.tile, Expr("tail_size_constraint"), block);
      body = LetStmt::make(var_len_partial_tile, len_partial_tile, let_tail_assert);
    } else {
      body = full_tiles;
    }

    Stmt let_body = LetStmt::make(var_num_full_tiles, num_full_tiles, body);
    return let_body;
  }
};

class SubstituteMinMaxMutator : public AttrIRMutator {
 public:
  SubstituteMinMaxMutator(const std::vector<Expr> &orig_exprs, bool is_first)
      : orig_exprs(orig_exprs), is_first(is_first) {}
  ~SubstituteMinMaxMutator() override = default;

 private:
  template <class T>
  Expr GenericMinMaxSubstitute(const T *op, const Expr &e) {
    for (const auto &orig_expr : orig_exprs) {
      if (Equal(orig_expr, e)) {
        return is_first ? Mutate(op->a) : Mutate(op->b);
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Min *op, const Expr &e) final { return GenericMinMaxSubstitute(op, e); }

  Expr Mutate_(const Max *op, const Expr &e) final { return GenericMinMaxSubstitute(op, e); }

  const std::vector<Expr> &orig_exprs;
  const bool is_first;
};

class IsolateMinMaxLoopsMutator : public IRMutator {
 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    loop_vars.insert(op->loop_var.get());

    std::vector<std::pair<Expr, Expr>> constraints;
    PostOrderVisit(op->body, [&constraints](const NodeRef &node) {
      if (auto op = node.as<For>()) {
        auto FindMinMax = [&constraints](const NodeRef &expr) {
          if (auto min_op = expr.as<Min>()) {
            constraints.emplace_back(make_pair(min_op->a <= min_op->b, ktvm::Downcast<Expr>(expr)));
          } else if (auto max_op = expr.as<Max>()) {
            constraints.emplace_back(make_pair(max_op->a >= max_op->b, ktvm::Downcast<Expr>(expr)));
          }
        };
        PostOrderVisit(op->min, FindMinMax);
        PostOrderVisit(op->extent, FindMinMax);
      }
    });

    std::vector<std::pair<Expr, std::vector<Expr>>> reduced_constraints;
    ExprSimplifier simplifier;
    for (const auto &it : constraints) {
      auto constraint = it.first;
      auto orig_expr = it.second;
      if (!IsVarInExpr(op->loop_var, constraint)) continue;
      Expr new_constraint = simplifier.ReduceInequality(constraint, op->loop_var);
      bool found = false;
      for (auto &red : reduced_constraints) {
        if (!Equal(red.first, new_constraint)) {
          found = true;
          red.second.push_back(orig_expr);
          break;
        }
      }
      if (!found) {
        std::vector<Expr> orig_exprs = {orig_expr};
        reduced_constraints.emplace_back(std::make_pair(new_constraint, orig_exprs));
      }
    }

    Stmt body = Mutate(op->body);
    for (const auto &it : reduced_constraints) {
      Expr condition = it.first;
      Stmt then_case = SubstituteMinMaxMutator(it.second, true).Mutate(body);
      Stmt else_case = SubstituteMinMaxMutator(it.second, false).Mutate(body);
      body = IfThenElse::make(condition, then_case, else_case);
    }

    loop_vars.erase(op->loop_var.get());
    return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
  }

  std::unordered_set<const Variable *> loop_vars;
};

class ReIsolateIndex : public AttrIRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "isolated_idx") {
      return AttrStmt::make(make_zero(Int(32)), op->attr_key, index_++, op->body);
    }

    return AttrIRMutator::Mutate_(op, s);
  }

  int index_{0};
};

class RedefineDuplicateRealize : public AttrIRMutator {
 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (defined_realize.count(op->func) == 0) {
      return IRMutator::Mutate_(op, s);
    }

    Array<Expr> shape;
    for (auto it : op->bounds) {
      shape.push_back(it->extent);
    }
    auto new_tensor = PlaceholderOpNode::make(op->func->func_name(), shape, op->type).output(0);
    defined_realize.insert(new_tensor->op);
    auto stmt = TensorSubstitute(Mutate(op->body), new_tensor->op, op->func, op->value_index);
    return Realize::make(new_tensor->op, new_tensor->value_index, new_tensor->dtype, op->bounds, const_true(1), stmt);
  }

  std::unordered_set<FunctionRef, NodeHash, NodeEqual> defined_realize;
};

Stmt IsolateLoops(const Stmt &stmt, bool enable_isolate_min_max) {
  Stmt ret = IsolateLoopsMutator().Mutate(stmt);
  if (enable_isolate_min_max) {
    ret = IsolateMinMaxLoopsMutator().Mutate(ret);
  }
  ret = ReIsolateIndex().Mutate(ret);
  ret = RedefineDuplicateRealize().Mutate(ret);

  return ret;
}
}  // namespace ir
}  // namespace akg
