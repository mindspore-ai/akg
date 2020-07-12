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
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/operation.h>
#include <pass/storage_access.h>
#include <poly/poly_util.h>
#include <pass/utils.h>
#include <build_module.h>

namespace akg {
namespace ir {
class ExprVerify : public IRVisitor {
 public:
#define DEFINE_ILLEGAL_BINOP_VISIT_(OP) \
  void Visit_(const OP *op) final { verify_ = false; }

  DEFINE_ILLEGAL_BINOP_VISIT_(Sub)
  DEFINE_ILLEGAL_BINOP_VISIT_(Div)
  DEFINE_ILLEGAL_BINOP_VISIT_(Mod)
  DEFINE_ILLEGAL_BINOP_VISIT_(Min)
  DEFINE_ILLEGAL_BINOP_VISIT_(Max)
  DEFINE_ILLEGAL_BINOP_VISIT_(EQ)
  DEFINE_ILLEGAL_BINOP_VISIT_(NE)
  DEFINE_ILLEGAL_BINOP_VISIT_(LT)
  DEFINE_ILLEGAL_BINOP_VISIT_(LE)
  DEFINE_ILLEGAL_BINOP_VISIT_(GT)
  DEFINE_ILLEGAL_BINOP_VISIT_(GE)
  DEFINE_ILLEGAL_BINOP_VISIT_(And)
  DEFINE_ILLEGAL_BINOP_VISIT_(Or)

  void Visit_(const Add *op) final {
    const auto var_expr_a = op->a.as<Variable>();
    if (var_expr_a && var_expr_a->name_hint.rfind("blockIdx", 0) == std::string::npos) {
      const_coeff_ -= 1;
      coeff_map_.insert(std::make_pair(var_expr_a, 1));
    } else if (is_const(op->a)) {
      const_coeff_ += static_cast<int32_t>(op->a.as<IntImm>()->value);
    } else {
      this->Visit(op->a);
    }

    const auto var_expr_b = op->b.as<Variable>();
    if (var_expr_b && var_expr_b->name_hint.rfind("blockIdx", 0) == std::string::npos) {
      const_coeff_ -= 1;
      coeff_map_.insert(std::make_pair(var_expr_b, 1));
    } else if (is_const(op->b)) {
      const_coeff_ += static_cast<int32_t>(op->b.as<IntImm>()->value);
    } else {
      this->Visit(op->b);
    }
  }

  void Visit_(const Mul *op) final {
    if (is_const(op->a)) {
      const auto var_expr = op->b.as<Variable>();
      if (var_expr && var_expr->name_hint.rfind("blockIdx", 0) == std::string::npos) {
        auto coeff = static_cast<int32_t>(op->a.as<IntImm>()->value);
        const_coeff_ -= coeff;
        coeff_map_.insert(std::make_pair(var_expr, coeff));
      } else {
        verify_ = false;
      }
    } else if (is_const(op->b)) {
      const auto var_expr = op->a.as<Variable>();
      if (var_expr && var_expr->name_hint.rfind("blockIdx", 0) == std::string::npos) {
        auto coeff = static_cast<int32_t>(op->b.as<IntImm>()->value);
        const_coeff_ -= coeff;
        coeff_map_.insert(std::make_pair(var_expr, coeff));
      } else {
        verify_ = false;
      }
    } else {
      verify_ = false;
    }
  }

  void Visit_(const Variable *op) final {
    if (op->name_hint.rfind("blockIdx", 0) == std::string::npos) {
      const_coeff_ -= 1;
      coeff_map_.insert(std::make_pair(op, 1));
    } else {
      verify_ = false;
    }
  }

  bool verify_{true};
  std::unordered_map<const Variable *, int32_t> coeff_map_;
  int32_t const_coeff_{0};
};

class IRSubstituteWithAttr : public IRMutator {
 public:
  explicit IRSubstituteWithAttr(const std::unordered_map<const Variable *, Expr> &vemap) : vemap_(vemap) {}
  ~IRSubstituteWithAttr() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_im2col") {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      std::unordered_map<std::string, NodeRef> new_attrs;

      for (auto attr : attrs) {
        auto expr = Downcast<Expr>(attr.second);
        expr = Substitute(expr, vemap_);
        new_attrs.emplace(std::make_pair(attr.first, expr));
      }

      return AttrStmt::make(Map<std::string, NodeRef>(new_attrs.begin(), new_attrs.end()), op->attr_key, op->value,
                            op->body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (vemap_.count(op)) {
      return vemap_.at(op);
    } else {
      return e;
    }
  }

 private:
  const std::unordered_map<const Variable *, Expr> &vemap_;
};

/*
 * for (i, 0, n) {
 *   for (j, 0, m) {
 *     rem = (ai + bj + c) % d
 *     quo = (ai + bj + c) / d
 *     func(rem, quo)
 *   }
 * }
 *
 * Transform to:
 *
 * rem = (c - a - b) % d
 * quo = (c - a - b) / d
 * for (i, 0, n) {
 *   rem = rem + a % d
 *   array = (rem >= d) ? 1 : 0
 *   rem = rem - array * d
 *   quo = que + a / d + array
 *   rem' = rem
 *   quo' = quo
 *   for (j, 0, m) {
 *     rem' = rem' + b % d
 *     array = (rem' >= d) ? 1 : 0
 *     rem' = rem' - array * d
 *     quo' = quo' + b / d + array
 *     func(rem', quo')
 *   }
 * }
 *
 */

class QuoEliminater : public IRMutator {
 public:
  QuoEliminater() {}
  ~QuoEliminater() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;

    loop_vars_.push_back(var);
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_vars_.pop_back();

    if (VarExprMap_.count(var.get())) {
      CHECK(stmt.as<For>());
      Stmt body = stmt.as<For>()->body;

      for (auto e : VarExprMap_[var.get()]) {
        VarExpr array = VarExpr("array");

        VarExpr old_rem = substitute_map_[e].rem_var;
        VarExpr old_quo = substitute_map_[e].quo_var;

        auto it = ExprCoeffMap_.find(e);
        CHECK(it != ExprCoeffMap_.end());
        auto it2 = it->second.find(var.get());
        CHECK(it2 != it->second.end());
        int32_t rem_coeff = it2->second.rem;
        int32_t quo_coeff = it2->second.quo;

        Expr divisor;
        if (e.as<Mod>()) {
          divisor = e.as<Mod>()->b;
        } else {
          CHECK(e.as<Div>());
          divisor = e.as<Div>()->b;
        }

        // old_rem = new_rem
        // old_quo = new_quo
        // for (varexpr, min, extent) {
        //   old_rem = old_rem + rem_coeff
        //   array = (old_rem >= divisor) ? 1 : 0
        //   old_rem = old_rem - array * divisor
        //   old_quo = old_quo + quo_coeff + array
        // }

        // Load/Store
        Expr load_rem = Load::make(Int(32, 1), old_rem, 0, const_true(1));
        Expr load_quo = Load::make(Int(32, 1), old_quo, 0, const_true(1));
        Expr load_array = Load::make(Int(32, 1), array, 0, const_true(1));
        Stmt s2 = Store::make(old_rem, load_rem + rem_coeff, 0, const_true(1));
        Expr select = Select::make(load_rem >= divisor, 1, 0);
        Stmt s3 = Store::make(array, select, 0, const_true(1));
        Stmt s4 = Store::make(old_rem, load_rem - load_array * divisor, 0, const_true(1));
        Stmt s5 = Store::make(old_quo, load_quo + quo_coeff + load_array, 0, const_true(1));
        body = Block::make(s5, body);
        body = Block::make(s4, body);
        body = Block::make(s3, body);
        body = Block::make(s2, body);

        // Allocate Register
        body = Allocate::make(array, Int(32), {make_const(Int(32), 1)}, const_true(), body);
        body = AttrStmt::make(array, "storage_scope", Expr("local.REG"), body);
      }

      body = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);

      for (auto e : VarExprMap_[var.get()]) {
        VarExpr old_rem = substitute_map_[e].rem_var;
        VarExpr old_quo = substitute_map_[e].quo_var;

        auto it = ExprCoeffMap_.find(e);
        CHECK(it != ExprCoeffMap_.end());
        auto it2 = it->second.find(op->loop_var.get());
        CHECK(it2 != it->second.end());

        VarExpr new_rem = VarExpr("rem");
        VarExpr new_quo = VarExpr("quo");

        // Load/Store  require to: LICM（loop-invariant code motion )
        Expr load_rem = Load::make(Int(32, 1), new_rem, 0, const_true(1));
        Expr load_quo = Load::make(Int(32, 1), new_quo, 0, const_true(1));
        Stmt s0 = Store::make(old_rem, load_rem, 0, const_true(1));
        Stmt s1 = Store::make(old_quo, load_quo, 0, const_true(1));
        body = Block::make(s1, body);
        body = Block::make(s0, body);

        // Allocate Register
        body = Allocate::make(old_quo, Int(32), {make_const(Int(32), 1)}, const_true(), body);
        body = AttrStmt::make(old_quo, "storage_scope", Expr("local.REG"), body);
        body = Allocate::make(old_rem, Int(32), {make_const(Int(32), 1)}, const_true(), body);
        body = AttrStmt::make(old_rem, "storage_scope", Expr("local.REG"), body);

        // update substitute_map_
        substitute_map_[e].rem_var = new_rem;
        substitute_map_[e].quo_var = new_quo;
      }

      return variable_init(body);
    }

    return variable_init(stmt);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "img2col_cbuf_to_ca" || op->name == "img2col_cbuf_to_cb") {
      load3d_ = true;
      Expr expr = IRMutator::Mutate_(op, e);
      load3d_ = false;
      return expr;
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    if (load3d_) {
      level_++;
      max_level_++;

      Expr expr = IRMutator::Mutate_(op, e);

      if (level_ == max_level_) {
        if (!is_const(op->b)) {
          goto ILLEGAL_HANDLE;
        }
        auto divisor = static_cast<int32_t>(op->b.as<IntImm>()->value);

        ExprVerify checker;
        // Before: (ai + bj + c) * d % e
        // After Simplify_cce: (ai + bj + c) * d % e
        // After CanonicalSimplify: (a*d*i + b*d*j + c*d) % e
        checker.Visit(CanonicalSimplify(op->a));
        if (!checker.verify_) {
          goto ILLEGAL_HANDLE;
        }

        Expr find_expr = find_division_expr(expr);
        if (!find_expr.defined()) {
          VarCoeffMap coeff;
          for (auto kv : checker.coeff_map_) {
            coeff.insert(std::make_pair(kv.first, Division(kv.second, divisor)));
            VarExprMap_[kv.first].emplace_back(expr);
          }

          if (checker.const_coeff_ != 0) {
            coeff.insert(std::make_pair(nullptr, Division(checker.const_coeff_, divisor)));
          }

          ExprCoeffMap_.insert(std::make_pair(expr, std::move(coeff)));

          VarExpr new_rem = VarExpr("rem");
          VarExpr new_quo = VarExpr("quo");
          SubstituteBody body;
          body.rem_var = new_rem;
          body.quo_var = new_quo;
          substitute_map_.insert(std::make_pair(expr, body));
          find_expr = expr;
        }

        CHECK(substitute_map_.find(find_expr) != substitute_map_.end());
        expr = Load::make(Int(32, 1), substitute_map_[find_expr].rem_var, 0, const_true(1));
      }

    ILLEGAL_HANDLE:
      level_--;
      if (level_ == 0) {
        max_level_ = 0;
      }

      return expr;
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Div *op, const Expr &e) final {
    if (load3d_) {
      level_++;
      max_level_++;

      Expr expr = IRMutator::Mutate_(op, e);

      if (level_ == max_level_) {
        if (!is_const(op->b)) {
          goto ILLEGAL_HANDLE;
        }
        auto divisor = static_cast<int32_t>(op->b.as<IntImm>()->value);

        ExprVerify checker;
        // Before: (ai + bj + c) * d / e
        // After Simplify_cce: (ai + bj + c) * d / e
        // After CanonicalSimplify: (a*d*i + b*d*j + c*d) / e
        checker.Visit(CanonicalSimplify(op->a));
        if (!checker.verify_) {
          goto ILLEGAL_HANDLE;
        }

        Expr find_expr = find_division_expr(expr);
        if (!find_expr.defined()) {
          VarCoeffMap coeff;
          for (auto kv : checker.coeff_map_) {
            coeff.insert(std::make_pair(kv.first, Division(kv.second, divisor)));
            VarExprMap_[kv.first].emplace_back(expr);
          }

          if (checker.const_coeff_ != 0) {
            coeff.insert(std::make_pair(nullptr, Division(checker.const_coeff_, divisor)));
          }

          ExprCoeffMap_.insert(std::make_pair(expr, std::move(coeff)));

          VarExpr new_rem = VarExpr("rem");
          VarExpr new_quo = VarExpr("quo");
          SubstituteBody body;
          body.rem_var = new_rem;
          body.quo_var = new_quo;
          substitute_map_.insert(std::make_pair(expr, body));
          find_expr = expr;
        }

        CHECK(substitute_map_.find(find_expr) != substitute_map_.end());
        expr = Load::make(Int(32, 1), substitute_map_[find_expr].quo_var, 0, const_true(1));
      }

    ILLEGAL_HANDLE:
      level_--;
      if (level_ == 0) {
        max_level_ = 0;
      }

      return expr;
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  Stmt variable_init(Stmt body) {
    if (loop_vars_.empty()) {
      for (auto kv : substitute_map_) {
        auto it = ExprCoeffMap_.find(kv.first);
        CHECK(it != ExprCoeffMap_.end());
        auto it2 = it->second.find(nullptr);
        if (it2 != it->second.end()) {
          Stmt s0 = Store::make(kv.second.rem_var, it2->second.rem, 0, const_true(1));
          Stmt s1 = Store::make(kv.second.quo_var, it2->second.quo, 0, const_true(1));
          body = Block::make(s1, body);
          body = Block::make(s0, body);
        } else {
          Stmt s0 = Store::make(kv.second.rem_var, 0, 0, const_true(1));
          Stmt s1 = Store::make(kv.second.quo_var, 0, 0, const_true(1));
          body = Block::make(s1, body);
          body = Block::make(s0, body);
        }

        // Allocate Register
        body = Allocate::make(kv.second.quo_var, Int(32), {make_const(Int(32), 1)}, const_true(), body);
        body = AttrStmt::make(kv.second.quo_var, "storage_scope", Expr("local.REG"), body);
        body = Allocate::make(kv.second.rem_var, Int(32), {make_const(Int(32), 1)}, const_true(), body);
        body = AttrStmt::make(kv.second.rem_var, "storage_scope", Expr("local.REG"), body);
      }

      // map clear
      ExprCoeffMap_.clear();
      substitute_map_.clear();
    }

    return body;
  }

  Expr find_division_expr(const Expr expr) {
    const Div *div_op = expr.as<Div>();
    const Mod *mod_op = expr.as<Mod>();

    CHECK(div_op || mod_op) << "Only Div/Mod op allowed!";
    Expr dividend = (div_op ? div_op->a : mod_op->a);
    Expr divisor = (div_op ? div_op->b : mod_op->b);
    for (auto kv : ExprCoeffMap_) {
      if (const Div *div = kv.first.as<Div>()) {
        if (Equal(div->a, dividend) && is_const_int(div->b, divisor.as<IntImm>()->value)) {
          return kv.first;
        }
      } else if (const Mod *mod = kv.first.as<Mod>()) {
        if (Equal(mod->a, dividend) && is_const_int(mod->b, divisor.as<IntImm>()->value)) {
          return kv.first;
        }
      }
    }

    return Expr();
  }

  /** This lets you use an Mod/Div Expr as a key in a unordered_map of the form
   * unordered_map<Expr, Foo, ExprHash, ModDivExprEqual> */
  struct ModDivExprEqual {
    bool operator()(const Expr &a, const Expr &b) const {
      if (const Mod *mod_a = a.as<Mod>()) {
        if (const Mod *mod_b = b.as<Mod>()) {
          return ((mod_a->a.get() == mod_b->a.get()) && (mod_a->b.get() == mod_b->b.get()));
        } else if (const Div *div_b = b.as<Div>()) {
          return ((mod_a->a.get() == div_b->a.get()) && (mod_a->b.get() == div_b->b.get()));
        } else {
          return false;
        }
      } else if (const Div *div_a = a.as<Div>()) {
        if (const Mod *mod_b = b.as<Mod>()) {
          return ((div_a->a.get() == mod_b->a.get()) && (div_a->b.get() == mod_b->b.get()));
        } else if (const Div *div_b = b.as<Div>()) {
          return ((div_a->a.get() == div_b->a.get()) && (div_a->b.get() == div_b->b.get()));
        } else {
          return false;
        }
      } else {
        return false;
      }
    }
  };

  struct Division {
    Division(int32_t a, int32_t b) : dividend(a), divisor(b) {
      CHECK_NE(b, 0);
      quo = a / b;
      rem = a % b;

      // make sure rem >= 0
      while (rem < 0) {
        quo -= 1;
        rem += b;
      }
    }

    int32_t dividend{0};
    int32_t divisor{1};
    int32_t quo{0};  // quotient
    int32_t rem{0};  // remainder
  };

  using VarCoeffMap = std::unordered_map<const Variable *, struct Division>;
  /* expr -> varexpr -> coefficient */
  std::unordered_map<Expr, VarCoeffMap, ExprHash, ModDivExprEqual> ExprCoeffMap_;
  /* varexpr -> [expr1, expr2, ...] */
  std::unordered_map<const Variable *, std::vector<Expr>> VarExprMap_;

  struct SubstituteBody {
    VarExpr rem_var;
    VarExpr quo_var;
  };

  /* expr -> expr & varexpr */
  std::unordered_map<Expr, SubstituteBody, ExprHash, ModDivExprEqual> substitute_map_;

  std::vector<VarExpr> loop_vars_;
  bool load3d_{false};
  uint32_t level_{0};
  uint32_t max_level_{0};
};

/*
 * // attr [0] pragma_emit_insn = "dma_copy"
 * for (ee10, 0, 18) {
 *   for (ee11, 0, 2) {
 *     for (ee12, 0, 16) {
 *       for (ee13, 0, 16) {
 *         input1_local__l1_local__l0_b(ee10, ee11, ee12, ee13) =input1_local__l1((((0 - (ee10 % 9)) + (9*ee11)) + 8),
 * (ee10/9), ee13, ee12)
 *       }
 *     }
 *   }
 * }
 * -->
 * for (ee10_o, 0, 2) {
 *   // attr [0] pragma_emit_insn = "dma_copy"
 *   for (ee10_i, 0, 9) {
 *     for (ee11, 0, 2) {
 *       for (ee12, 0, 16) {
 *         for (ee13, 0, 16) {
 *           input1_local__l1_local__l0_b(ee10_o * 9 + ee10_i, ee11, ee12, ee13) =input1_local__l1((((0 - ((ee10_o * 9 +
 * ee10_i) % 9)) + (9*ee11)) + 8), ((ee10_o * 9 + ee10_i)/9), ee13, ee12)
 *         }
 *       }
 *     }
 *   }
 * }
 * -->
 * for (ee10_o, 0, 2) {
 *   // attr [0] pragma_emit_insn = "dma_copy"
 *   for (ee10_i, 0, 9) {
 *     for (ee11, 0, 2) {
 *       for (ee12, 0, 16) {
 *         for (ee13, 0, 16) {
 *           input1_local__l1_local__l0_b(ee10_o * 9 + ee10_i, ee11, ee12, ee13) =input1_local__l1((((0 - (ee10_i % 9))
 * + (9*ee11)) + 8), ee10_o, ee13, ee12)
 *         }
 *       }
 *     }
 *   }
 * }
 */
class LoopSplit : public IRMutator {
 public:
  LoopSplit() {}
  ~LoopSplit() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_attrs") {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.count(ATTR_CONV_KERNEL_H)) {
        kernel_h_ = Downcast<Expr>(attrs[ATTR_CONV_KERNEL_H]);
      }

      if (attrs.count(ATTR_CONV_KERNEL_W)) {
        kernel_w_ = Downcast<Expr>(attrs[ATTR_CONV_KERNEL_W]);
      }

      if (attrs.count(ATTR_CONV_FILTER_NAME)) {
        kernel_name_ = Downcast<Expr>(attrs[ATTR_CONV_FILTER_NAME]).as<StringImm>()->value;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    if (load2d_) {
      CHECK(is_zero(op->min));
      CHECK(is_const(op->extent));
    }

    loop_vars_.insert(std::make_pair(var.get(), op->extent));
    Stmt stmt = IRMutator::Mutate_(op, s);
    loop_vars_.erase(var.get());

    if (split_var_ != nullptr && split_var_->name_hint == name) {
      CHECK(axis_outer_.defined() && axis_inner_.defined());
      CHECK(stmt.as<For>());
      Stmt body = stmt.as<For>()->body;

      body = For::make(axis_inner_, 0, split_factor_, op->for_type, op->device_api, body);

      CHECK(op->extent.as<IntImm>());
      auto extent = static_cast<int32_t>(op->extent.as<IntImm>()->value);
      body = For::make(axis_outer_, 0, extent / split_factor_, op->for_type, op->device_api, body);

      split_var_ = nullptr;
      std::unordered_map<const Variable *, Expr> vmap;
      vmap.emplace(var.get(), axis_outer_ * split_factor_ + axis_inner_);
      body = IRSubstituteWithAttr(vmap).Mutate(body);

      return body;
    }

    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func->func_name() == kernel_name_ + "_local_L1_local_L0B") {
      Expr body = this->Mutate(op->value);

      Array<Expr> provide_new_args;
      for (auto arg : op->args) {
        const auto var = arg.as<Variable>();
        if (var == nullptr) {
          provide_new_args.push_back(arg);
          continue;
        }

        if (split_var_ != nullptr && split_var_->name_hint == var->name_hint) {
          CHECK(axis_outer_.defined() && axis_inner_.defined());
          provide_new_args.push_back(axis_outer_ * split_factor_ + axis_inner_);
        } else {
          provide_new_args.push_back(arg);
        }
      }

      return Provide::make(op->func, op->value_index, body, provide_new_args);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == kernel_name_ + "_local_L1") {
      load2d_ = true;
      split_var_ = nullptr;
      split_factor_ = -1;
      axis_outer_ = VarExpr(ObjectPtr<Object>());
      axis_inner_ = VarExpr(ObjectPtr<Object>());
      Expr expr = IRMutator::Mutate_(op, e);
      load2d_ = false;

      return expr;
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    if (load2d_) {
      const auto var = op->a.as<Variable>();
      CHECK(var) << "illegal expression " << e << " for Load2d";
      const auto pb = op->b.as<IntImm>();
      if (!global_attrs.GetBoolAttr(kIsDynamic, false))
        CHECK(pb && air::arith::Analyzer().CanProve(pb->value == kernel_h_ * kernel_w_));

      for (auto kv : loop_vars_) {
        if (kv.first == var) {
          auto extent = static_cast<int32_t>(kv.second.as<IntImm>()->value);
          auto divisor = static_cast<int32_t>(pb->value);
          CHECK_EQ(extent % divisor, 0);

          if (split_var_ == nullptr) {
            CHECK_EQ(split_factor_, -1);

            split_var_ = var;
            split_factor_ = divisor;

            axis_outer_ = VarExpr(var->name_hint + "_outer");
            axis_inner_ = VarExpr(var->name_hint + "_inner");
          } else {
            CHECK(split_var_ == var) << "Don't support multi loop var split: load2d(var % 3, var' % 3)";
            CHECK(split_factor_ == divisor) << "Don't support multi level split: load2d(var / 3, var % 9)";
            CHECK(axis_outer_.defined() && axis_inner_.defined());
          }

          Expr new_mod = axis_inner_;

          return new_mod;
        }
      }
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Div *op, const Expr &e) final {
    if (load2d_) {
      const auto var = op->a.as<Variable>();
      CHECK(var) << "illegal expression " << e << " for Load2d";
      const auto pb = op->b.as<IntImm>();
      if (!global_attrs.GetBoolAttr(kIsDynamic, false))
        CHECK(pb && air::arith::Analyzer().CanProve(pb->value == kernel_h_ * kernel_w_));

      for (auto kv : loop_vars_) {
        if (kv.first == var) {
          auto extent = static_cast<int32_t>(kv.second.as<IntImm>()->value);
          auto divisor = static_cast<int32_t>(pb->value);
          CHECK_EQ(extent % divisor, 0);

          if (split_var_ == nullptr) {
            CHECK_EQ(split_factor_, -1);

            split_var_ = var;
            split_factor_ = divisor;

            axis_outer_ = VarExpr(var->name_hint + "_outer");
            axis_inner_ = VarExpr(var->name_hint + "_inner");
          } else {
            CHECK(split_var_ == var) << "Don't support multi loop var split: load2d(var % 3, var' % 3)";
            CHECK(split_factor_ == divisor) << "Don't support multi level split: load2d(var / 3, var % 9)";
            CHECK(axis_outer_.defined() && axis_inner_.defined());
          }

          Expr new_div = axis_outer_;

          return new_div;
        }
      }
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  /* Variable -> extent */
  std::unordered_map<const Variable *, Expr> loop_vars_;
  bool load2d_{false};
  const Variable *split_var_{nullptr};
  int32_t split_factor_{-1};
  VarExpr axis_outer_;
  VarExpr axis_inner_;
  Expr kernel_h_{0};
  Expr kernel_w_{0};
  std::string kernel_name_;
};

Stmt ModDivEliminate(Stmt stmt) {
  stmt = QuoEliminater().Mutate(stmt);
  stmt = LoopSplit().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
