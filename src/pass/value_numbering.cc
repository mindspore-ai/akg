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
#include <tvm/operation.h>
#include <tvm/expr.h>
#include <dmlc/optional.h>

#include "ir_pass.h"
#include "pass/ir_util.h"
#include "pass/storage_access.h"
#include "pass/utils.h"
#include "ir_generictree.h"

/* located at 3rdparty/tvm/src/pass/ir_deep_compare.cc */
extern int air::ir::Compare(const Expr &lhs, const Expr &rhs);

namespace akg {
namespace ir {
/* Expect a straightline of code in forms of Block {.., Block {.., Block {.. ,..}}}
   and no 'environment' change (i.e. variable binding must be the same throughout the context) */
class LocalValueNumbering : public IRMutator {
 public:
  using VNExpr = Expr;
  using ValueNumberLabel = UIntImm;
  using ValueNumber = uint64_t;

  static VNExpr makeVNLabel(const air::DataType ty, ValueNumber vn) {
    NodePtr<UIntImm> node = make_node<UIntImm>();
    node->type = ty;
    node->value = vn;
    return Expr(node);
  }

  /* Check if VNExpr is a storage, considered as a TYPE-SAFE MACRO please */
  static const ValueNumberLabel *IsSolelyValueNumber(const VNExpr &x) { return x.as<ValueNumberLabel>(); }

  /* Get the left value of a provide statement */
  Expr leftHandSide(const Provide *op) {
    return Call::make(op->value.type(), op->func->func_name(), op->args, Call::Halide, op->func);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    /* Emphasized here to make sure the evaluation order is what we want;
       in case in the future IRMutator::Mutate_ make the order
       of visiting undefined. Totally a copy and paste */

    Stmt first = this->Mutate(op->first);  // this stmt must be the first to evaluate.
    Stmt rest = this->Mutate(op->rest);
    if (first.same_as(op->first) && rest.same_as(op->rest)) {
      return s;
    } else {
      return Block::make(first, rest);
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    LOG(DEBUG) << ">> VN provide:" << s;
    LOG(DEBUG) << "   func: " << op->func;
    LOG(DEBUG) << "   func->get(): " << op->func.get();
    LOG(DEBUG) << "   value_index " << op->value_index;
    LOG(DEBUG) << "   value: " << op->value;
    LOG(DEBUG) << "   args: " << op->args;
    LOG(DEBUG) << "   name: " << op->func->func_name();
    auto ret_stmt = s;
    Expr curr_rhs_expr_ = op->value;

    LOG(DEBUG) << "toValueNumbered Called";
    Expr curr_rhs_expr_in_vnform = toValueNumberedForm(this).Mutate(curr_rhs_expr_);
    LOG(DEBUG) << "toValueNumbered Exited";

    Expr curr_lhs_expr = leftHandSide(op);

    if (auto vn = LocalValueNumbering::IsSolelyValueNumber(curr_rhs_expr_in_vnform)) {
      mapping_[curr_lhs_expr] = std::make_tuple(vn->value, curr_lhs_expr, "");
    } else {
      if (available_.find(curr_rhs_expr_in_vnform) != available_.end()) {
        dmlc::optional<Expr> maybe_rhs = InverseMappingVN(available_[curr_rhs_expr_]);
        if (maybe_rhs.has_value()) {
          // Only when
          /* Discover an available expression with vn : available_[curr_rhs_expr_]!

            stmt1 = a op b    -- available expression
            stmt2 = a op b    -- current provide

            current provide is mutated to:

            stmt1 = a op b
            stmt2 = stmt1
          */

          /* Get the lhs value that was assigned to the curr_rhs_expr_in_vnform */
          // why InverseMappingVN is ensured? maybe there is no element in mapping_ has that value?
          Expr new_rhs = *maybe_rhs;

          ret_stmt = Provide::make(op->func, op->value_index, new_rhs, op->args);

          /* Does the current provide has an existing vn? */
          auto iter = mapping_.find(curr_lhs_expr);
          if (iter != mapping_.end()) { /* already has a vn */
            /* Invalidate all current available expressions that uses the old vn */
            ValueNumber old_vn = std::get<0>(iter->second);
            RemoveAvailableExpr_(old_vn);
          }

          // Make it sound --
          //  a(i - 3) = x1 + x2
          //  a(3) = x1 - x2
          //  b(i) = x1 + x2 // this line shouldn't be changed to a(i - 3) when i - 3 = 3
          auto it = mapping_.begin();
          auto next_it = it;
          for (; it != mapping_.end(); it = next_it) {
            ++next_it;
            CHECK(it->first.as<Call>());
            auto it_tensor = it->first.as<Call>();
            if (it_tensor->call_type == Call::Halide && curr_lhs_expr.as<Call>() &&
                it_tensor->name == curr_lhs_expr.as<Call>()->name) {
              // the tensor has same name; might access the same point
              ValueNumber it_tensor_vn = std::get<0>(it->second);
              RemoveAvailableExpr_(it_tensor_vn);
            }
          }
          /* Give the current provide the correct vn */
          mapping_[curr_lhs_expr] = std::make_tuple(available_[curr_rhs_expr_in_vnform], curr_lhs_expr, "");
        } else {
          // there is actually value number assigned to the value, just reuse
          mapping_[curr_lhs_expr] = std::make_tuple(available_[curr_rhs_expr_in_vnform], curr_lhs_expr, "");
        }
      } else {
        available_[curr_rhs_expr_] = GetValueNumber_(curr_lhs_expr, op->func->func_name());
      }
    }
    return ret_stmt;
  }

 private:
  struct syntaxCompare {
    /* Make as default ..
       but consider (+ v1 v2) and (+ v2 v1) should be considered equivalent, written in this way, we
       can easily extend to this scenario... the expression is in tree form originally, flattening to tuple
       is not intuitively a good idea., which will introduce a lot of boilerplate about transformation,
       and introduce "expression problem" again. (The whole point of visitor pattern)
     */
    bool operator()(const Expr &a, const Expr &b) const { return Compare(a, b) < 0; }
  };

  /* Directly using "Expr" as the key, where base case of AST Tree are only
     Immediate Integer stands for value numbering we use std::map and overloading
     the Compare operator where two keys are compared to be equal when two expression are equivalent
     in this way we can extend the pattern matching power and don't need to break the type.
     Thus available_ would be of type Expr -> ValueNumber */
  std::map<air::Expr, ValueNumber, syntaxCompare> available_;
  int global_vn_counter_{0};

  /* Further simplify mapping_ can be even simplified. */
  std::map<Expr, std::tuple<ValueNumber, Expr, std::string>, syntaxCompare> mapping_;

  /* To be made into Functional Object Only, by not allowed to be
     assigned (only allowed as right value) */
  class IsValueNumberInside : public IRVisitor {
   public:
    IsValueNumberInside(const ValueNumber toFind, const NodeRef &nodeRef) : checking(toFind), node(nodeRef) {}
    ~IsValueNumberInside() override = default;

    void Visit_(const ValueNumberLabel *op) override {
      if (op->value == checking) {
        findone = true;
      }
    }

    bool isTrue() {
      this->Visit(node);
      return findone;
    }

   private:
    const ValueNumber checking;
    NodeRef node;
    bool findone{false};
  };

  /* Used when try to invalidate a expression in the dictionary */
  void RemoveAvailableExpr_(ValueNumber vn) {
    auto it = available_.begin();
    auto next_it = it;
    for (; it != available_.end(); it = next_it) {
      ++next_it;
      // either when value number appear in the left hand side or Right hand side of the mapping
      // remove it
      if (IsValueNumberInside(vn, it->first).isTrue()) {
        available_.erase(it);
      }
    }
    return;
  }

  /* Responsible for the global value numbering counting
     automatically assign new vn if not assigned before */
  ValueNumber GetValueNumber_(const Expr &var, const std::string &name = "") {
    /* ASSERTION that var has to be in the indexing of array form
       To be made take args into account in the hash. */
    auto iter = mapping_.find(var);
    if (iter == mapping_.end()) {
      /* not found, assign a new one */
      mapping_[var] = std::make_tuple(global_vn_counter_, var, name);
      global_vn_counter_++;
      return std::get<0>(mapping_[var]);
    } else {
      return std::get<0>(iter->second);
    }
  }

  /* Reverse mapping of Bijection mapping "mapping_", when there is no mapping, returns none */
  dmlc::optional<Expr> InverseMappingVN(ValueNumber vn) {
    for (auto elem : mapping_) {
      if (std::get<0>(elem.second) == vn) return dmlc::optional<Expr>(std::get<1>(elem.second));
    }
    LOG(FATAL) << "Not found Inverse Mapping";
    return {};
  }

/* Get Symbol Printing for free, from IRPrinter (another advantage of using Expr as key) */
#ifdef PRINTMAPPING
  LOG(DEBUG) << "<<< Mapping (" << mapping_.size() << ") >>>";
  for (auto elem : mapping_) {
    LOG(DEBUG) << "v" << std::get<0>(elem.second) << ":" << std::get<2>(elem.second);
  }
  LOG(DEBUG) << "<<< Available (" << available_.size() << ") >>>";
  for (auto elem : available_) {
    LOG(DEBUG) << "v" << (elem.second) << ":" << (elem.first);
  }
#endif

  /* Every leaf will be changed into a value number, an astnode with valuenumber as base case
     actually expect a very simple a <+/-/...> b form, where a, b are at most queries on array */
  class toValueNumberedForm : public IRMutator {
   public:
#define vnlabel(x, y) LocalValueNumbering::makeVNLabel(x, y)
    explicit toValueNumberedForm(LocalValueNumbering *const thecaller) : thecaller(thecaller) {}
    ~toValueNumberedForm() override = default;

    VNExpr Mutate_(const Variable *op, const Expr &e) override {
      return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
    }

    VNExpr Mutate_(const Call *op, const Expr &e) override {
      /* when CallType = Halide, we encountered a leaf */
      if (op->call_type == Call::CallType::Halide) {
        return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
      }
      return this->IRMutator::Mutate_(op, e);
    }

    VNExpr Mutate_(const IntImm *op, const Expr &e) override {
      return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
    }
    VNExpr Mutate_(const UIntImm *op, const Expr &e) override {
      return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
    }
    VNExpr Mutate_(const FloatImm *op, const Expr &e) override {
      return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
    }
    VNExpr Mutate_(const StringImm *op, const Expr &e) override {
      return vnlabel(e.type(), thecaller->GetValueNumber_(e, ""));
    }
#undef vnlabel

   private:
    LocalValueNumbering *const thecaller;
  };
};  // namespace ir

namespace {
/* A fake basicblock identifier:
   Only identify the longest straightline code till the end
   Not allowing any sort of variable binding during the code since we haven't yet
   recorded the context of each (variable). To be made to adapt ToSequence here. */
class IsBasicBlock : public IRVisitor {
 public:
  void Visit_(const For *op) final { isBB = false; }
  void Visit_(const IfThenElse *op) final { isBB = false; }
  void Visit_(const LetStmt *op) final { isBB = false; }
  void Visit_(const Let *op) final { isBB = false; }
  void Visit_(const Block *op) final { this->IRVisitor::Visit_(op); }
  bool isBB{true};
};
}  // namespace
class SingleStageVN : public IRMutator {
 public:
  SingleStageVN() {}
  ~SingleStageVN() override = default;

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    /* block is recursively defined as List of Type A is either Nil or a tuple of (A, List)
       sequence of a = 1; b = 2; c = 3 will become "Block{ a= 1, {Block {b = 2, c = 3}}}" */
    IsBasicBlock bb;
    auto ret_stmt = s;
    bb.Visit(s);
    {
      if (bb.isBB) {
        /* this block is a straightline code */
        LOG(DEBUG) << "Find one basic block" << op->GetTypeKey();
        LocalValueNumbering vn;
        ret_stmt = vn.Mutate(s);
      } else {
        ret_stmt = this->IRMutator::Mutate_(op, s);
      }
    }
    return ret_stmt;
  }
};

Stmt ValueNumbering(const Stmt stmt) {
  /* New BB builder - Disable for now */
  return SingleStageVN().Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
