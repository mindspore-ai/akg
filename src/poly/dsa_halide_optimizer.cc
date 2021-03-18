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

#include <tvm/target_info.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include <iostream>
#include <stack>

#include "common/common_util.h"
#include "pass/utils.h"
#include "tvm.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {
constexpr auto PASS_DOWN = "pass_down";
enum OPERATOR_TYPE { T_OUTOF_LIB = 0, T_PASS_DOWN, T_TENSOR_OF_TENSOR, T_TENSOR_OF_TENSOR_ACCUM };
using OP_TYPE = OPERATOR_TYPE;
using TensorVarTab = std::unordered_map<const Variable *, std::vector<const IfThenElse *>>;

template <typename T>
bool isType(const Expr &e) {
  if (e.as<T>() != nullptr) {
    return true;
  }
  return false;
}

class OpDetector : public IRVisitor {
 public:
  OpDetector() {}
  ~OpDetector() override = default;

  void detect(const Stmt &s) { this->Visit(s); }

  void Visit_(const AttrStmt *op) final {
    CHECK(op);
    if (op->attr_key == PASS_DOWN) {
      type_ = OP_TYPE::T_PASS_DOWN;
      return;
    }
    return IRVisitor::Visit_(op);
  }

  void Visit_(const Realize *op) final {
    in_realize_ = true;
    if (mem_buffer_tab_.count(op->func->func_name()) == 0) {
      mem_buffer_tab_[op->func->func_name()] = 0;
    } else {
      mem_buffer_tab_[op->func->func_name()]++;
    }
    IRVisitor::Visit_(op);
    in_realize_ = false;
  }

  void Visit_(const For *op) final {
    for_stk.push(op);
    if (in_realize_) {
      if (tab_.count(op->loop_var.get()) == 0) {
        std::vector<const IfThenElse *> value;
        tab_[op->loop_var.get()] = value;
      }
    }
    IRVisitor::Visit_(op);
    for_stk.pop();
  }

  std::vector<const Variable *> GetExprSpecVar(const Expr &expr) {
    auto tabAdd = [](const Expr &expr, const TensorVarTab &tab) -> const Variable * {
      if (isType<Variable>(expr) && tab.count(expr.as<Variable>()) > 0) {
        return expr.as<Variable>();
      }
      return nullptr;
    };
    std::vector<const Variable *> res;
    if (isType<Add>(expr)) {
      const auto add_expr = expr.as<Add>();
      res.push_back(tabAdd(add_expr->a, tab_));
      res.push_back(tabAdd(add_expr->b, tab_));
    } else if (isType<Variable>(expr)) {
      res.push_back(tabAdd(expr, tab_));
    }
    return res;
  }

  void Visit_(const IfThenElse *op) final {
    CHECK(op);
    auto empty_case = op->else_case;
    auto checkTOfTensor = [](const Expr &e, const std::unordered_map<std::string, int> &table) {
      if (isType<Call>(e) && table.count(e.as<Call>()->name) > 0) {
        return true;
      }
      return false;
    };
    if (empty_case == Stmt() && isType<EQ>(op->condition)) {
      const auto equation = op->condition.as<EQ>();
      if (checkTOfTensor(equation->a, mem_buffer_tab_) || checkTOfTensor(equation->b, mem_buffer_tab_)) {
        type_ = OP_TYPE::T_TENSOR_OF_TENSOR;
        std::vector<const Variable *> vars;
        vars = checkTOfTensor(equation->b, mem_buffer_tab_) ? GetExprSpecVar(equation->a) : GetExprSpecVar(equation->b);
        for (const auto var : vars) {
          if (tab_.count(var) > 0) {
            tab_[var].push_back(op);
          }
        }
        return;
      }
    } else if (empty_case == Stmt() && isType<And>(op->condition)) {
      auto and_op = op->condition.as<And>();
      if (isType<EQ>(and_op->a) && isType<EQ>(and_op->b)) {
        auto eq_first = and_op->a.as<EQ>();
        auto eq_second = and_op->b.as<EQ>();
        if ((checkTOfTensor(eq_first->a, mem_buffer_tab_) || checkTOfTensor(eq_first->b, mem_buffer_tab_)) &&
            (checkTOfTensor(eq_second->a, mem_buffer_tab_) || checkTOfTensor(eq_second->b, mem_buffer_tab_))) {
          type_ = T_TENSOR_OF_TENSOR_ACCUM;
          elim_if_ = op;
          if (checkTOfTensor(eq_first->a, mem_buffer_tab_)) {
            tensor_map_[eq_first->b.as<Variable>()] = eq_first->a;
          } else {
            tensor_map_[eq_first->a.as<Variable>()] = eq_first->b;
          }
          if (checkTOfTensor(eq_second->a, mem_buffer_tab_)) {
            tensor_map_[eq_second->b.as<Variable>()] = eq_second->a;
          } else {
            tensor_map_[eq_second->a.as<Variable>()] = eq_second->b;
          }
          int count = tensor_map_.size();
          auto tmp_for_stk = for_stk;
          while (count > 0) {
            auto current_for = tmp_for_stk.top();
            if (tensor_map_.count(current_for->loop_var.get()) > 0 && elim_for_set_.count(current_for) == 0) {
              elim_for_set_.insert(current_for);
              count--;
            }
            tmp_for_stk.pop();
          }
          return;
        }
      }
    }
    return IRVisitor::Visit_(op);
  }

  OP_TYPE type_{OP_TYPE::T_OUTOF_LIB};
  TensorVarTab tab_;
  std::unordered_set<const For *> elim_for_set_;
  std::unordered_map<const Variable *, Expr> tensor_map_;
  const IfThenElse *elim_if_;

 private:
  bool in_realize_{false};
  std::stack<const For *> for_stk;
  std::unordered_map<std::string, int> mem_buffer_tab_;
};

/***********************************************************************
 * This pass eliminate the IR transformed by PASS RewriteTensorIdx
   for (aa, 0, 4) {
    for (cc1, 0, 192) {
      for (cc0, 0, 192) {
        if((cc0 == input_3(aa)) && (cc1 == input_3(aa))){
          res(aa, cc0) = res(aa, cc1) + whatever
        }
      }
    }
   }
    |
    v
   for (aa, 0, 4) {
     res(aa, input_3(aa)) = res(aa, input_3(aa)) + whatever
   }
 *
 ***************************************************************/

class ElimTensorIdx : public IRMutator {
 public:
  ElimTensorIdx(std::unordered_set<const For *> &elim_for_set, std::unordered_map<const Variable *, Expr> &tensor_map,
                const IfThenElse *elim_if)
      : elim_for_set_(elim_for_set), tensor_map_(tensor_map), elim_if_(elim_if) {}
  ~ElimTensorIdx() override = default;

  Stmt run(const Stmt &s) { return this->Mutate(s); }
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (elim_for_set_.find(op) != elim_for_set_.end()) {
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (op == elim_if_) {
      return Substitute(op->then_case, tensor_map_);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_set<const For *> elim_for_set_;
  std::unordered_map<const Variable *, Expr> tensor_map_;
  const IfThenElse *elim_if_;
};

/***********************************************************************
 * This pass used to remove pass_down attr and move down the marked axis
    for (cc9, 0, 56) {
        // attr [0] pass_down = 16
        for (cc11, 0, 16) {
          if (!(cc9 == 0)) {
            max_pool_hybrid_0_local_UB(0, 0, 0, cc9, cc11) =max(max_pool_hybrid_0_local_UB(0, 0, 0, cc9, cc11), i
 nput_1_local_UB(0, 0, ((2*0) + 0), ((2*cc9) - 1), cc11))
          }
        }
    }
    |
    v
   for (cc9, 0, 56) {
       if (!(cc9 == 0)) {
         for (cc11, 0, 16) {
           max_pool_hybrid_0_local_UB(0, 0, 0, cc9, cc11) =max(max_pool_hybrid_0_local_UB(0, 0, 0, cc9, cc11), i
 nput_1_local_UB(0, 0, ((2*0) + 0), ((2*cc9) - 1), cc11))
         }
       }
   }
 *
 ***************************************************************/

class PassDownForAxis : public IRMutator {
 public:
  PassDownForAxis() {}
  ~PassDownForAxis() override = default;

  Stmt run(const Stmt s) { return this->Mutate(s); }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    CHECK(op);
    if (op->attr_key == PASS_DOWN) {
      need_pass_down_ = true;
      Stmt stmt = this->Mutate(op->body);
      need_pass_down_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    CHECK(op);
    if (need_pass_down_ && pass_down_for_ == nullptr) {
      pass_down_for_ = op;
      Stmt stmt = this->Mutate(op->body);
      pass_down_for_ = nullptr;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    CHECK(op);
    if (need_pass_down_ && pass_down_for_ != nullptr) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      return For::make(pass_down_for_->loop_var, pass_down_for_->min, pass_down_for_->extent, pass_down_for_->for_type,
                       pass_down_for_->device_api, stmt);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool need_pass_down_{false};
  const For *pass_down_for_{nullptr};
};

/***********************************************************************
 * After the condition is removed, the sinked tensor write back to promotion
 // attr [placeholder(gather_output_local_UB, 0x2f725f0)] realize_scope = DOT_LOCAL_BUF
  realize gather_output_local_UB<float16>([0, 1], [0, 59], [0, 212]) {
    for (cc4, 0, 212) {
      // attr [reg0_local_REG] storage_scope = "local.REG"
      allocate reg0_local_REG[int32 * 1]
      reg0_local_REG[0] = input_2_local_UB(cc4)
      for (cc7, 0, 59) {
        gather_output_local_UB(0, cc7, cc4) = input_1_local_UB(0, cc7, reg0_local_REG[0])
      }
      for (cc7, 0, 59) {
        gather_output(cc0, ((59*cc1) + cc7), cc4) = gather_output_local_UB(0, cc7, cc4)
      }
    }
  }

  |
  v

 * // attr [placeholder(gather_output_local_UB, 0x2f725f0)] realize_scope = DOT_LOCAL_BUF
  realize gather_output_local_UB<float16>([0, 1], [0, 59], [0, 212]) {
    for (cc4, 0, 212) {
      // attr [reg0_local_REG] storage_scope = "local.REG"
      allocate reg0_local_REG[int32 * 1]
      reg0_local_REG[0] = input_2_local_UB(cc4)
      for (cc7, 0, 59) {
        gather_output_local_UB(0, cc7, cc4) = input_1_local_UB(0, cc7, reg0_local_REG[0])
      }
    }
    for (cc7, 0, 59) {
       for (cc4, 0, 212) {
        gather_output(cc0, ((59*cc1) + cc7), cc4) = gather_output_local_UB(0, cc7, cc4)
       }
    }
  }
***************************************************************/
class GatherWritePromotion : public IRMutator {
 public:
  explicit GatherWritePromotion(const std::set<const Variable *> &condition_removed_gather_vars)
      : gather_vars(condition_removed_gather_vars) {}
  ~GatherWritePromotion() override = default;

  Stmt run(const Stmt &s) {
    std::unordered_map<std::string, std::string> gather_write_tensors;
    auto getGatherWriteTensor_ = [&, this](const NodeRef &node) { this->getGatherWriteSink(node); };
    PostOrderVisit(s, getGatherWriteTensor_);
    return Mutate(s);
  }

  Stmt getWritePromotion(const std::unordered_map<const Variable *, const For *> &var2for, const Stmt &write) {
    //
    std::stack<const Variable *> src_vars;
    auto provide = write.as<Provide>();
    CHECK(provide);
    auto src = provide->value;
    PostOrderVisit(src, [&src_vars](const NodeRef &node) {
      if (auto var = node.as<Variable>()) {
        src_vars.push(var);
      }
    });

    Stmt write_promotion = write;
    while (!src_vars.empty()) {
      auto var = src_vars.top();
      CHECK_GT(var2for.count(var), 0);
      auto for_ = var2for.at(var);
      write_promotion =
        For::make(for_->loop_var, for_->min, for_->extent, for_->for_type, for_->device_api, write_promotion);
      src_vars.pop();
    }
    return write_promotion;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    auto tensor = op->func.get();
    if (gather_write_sink_.count(tensor) > 0) {
      auto gather_write_tensor = tensor;
      auto gather_write_tensor_gm_ = gather_write_sink_[tensor];
      bool record = true;
      std::unordered_map<const Variable *, const For *> var2for;
      Stmt gather_write_provide_;
      PackedFunc recordFor = PackedFunc([&](TVMArgs args, TVMRetValue *ret) {
        Stmt st = args[0];
        if (auto for_ = st.as<For>()) {
          if (record) {
            var2for[for_->loop_var.get()] = for_;
          }
        }
      });

      PackedFunc eliminateGatherWrite = PackedFunc([&](TVMArgs args, TVMRetValue *ret) {
        Stmt st = args[0];
        if (auto provide = st.as<Provide>()) {
          auto dst = provide->func;
          auto src = provide->value.as<Call>();
          if (dst.get() == gather_write_tensor_gm_ && src && src->func.get() == gather_write_tensor) {
            record = false;
            gather_write_provide_ = st;
            *ret = Evaluate::make(0);
          }
        }
      });
      auto body = air::ir::IRTransform(op->body, recordFor, eliminateGatherWrite, {Expr("Provide"), Expr("For")});

      CHECK(gather_write_provide_.defined());
      Stmt write_promotion = getWritePromotion(var2for, gather_write_provide_);
      body = Block::make(body, write_promotion);
      return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  void getGatherWriteSink(const NodeRef &node) {
    const Variable *gatherVar = nullptr;
    auto op = node.as<Block>();
    if (op && op->first.as<Store>()) {
      auto store = node.as<Block>()->first.as<Store>();
      if (gather_vars.count(store->buffer_var.get()) > 0) {
        gatherVar = store->buffer_var.get();
      }
    }
    if (gatherVar == nullptr) {
      return;
    }

    const Node *gather_write_sink_tensor = nullptr;
    const Node *gather_write_sink_tensor_gm = nullptr;
    PostOrderVisit(op->rest,
                   [&gatherVar, &gather_write_sink_tensor, &gather_write_sink_tensor_gm, this](const NodeRef &node) {
                     if (gather_write_sink_tensor == nullptr) {
                       if (auto provide = node.as<Provide>()) {
                         bool valueHasGatherVar = false;
                         PostOrderVisit(provide->value, [&gatherVar, &valueHasGatherVar](const NodeRef &node_) {
                           auto load = node_.as<Load>();
                           if (load && load->buffer_var.get() == gatherVar) {
                             valueHasGatherVar = true;
                           }
                         });
                         if (valueHasGatherVar && provide->func->IsInstance<OperationNode>()) {
                           gather_write_sink_tensor = provide->func.get();
                         }
                       }
                     } else {
                       if (auto provide = node.as<Provide>()) {
                         if (auto value = provide->value.as<Call>()) {
                           if (value->func.get() == gather_write_sink_tensor &&
                               this->dataScope(provide->func->func_name()) == "global") {
                             gather_write_sink_tensor_gm = provide->func.get();
                           }
                         }
                       }
                     }
                   });

    if (gather_write_sink_tensor_gm != nullptr) {
      gather_write_sink_[gather_write_sink_tensor] = gather_write_sink_tensor_gm;
    }
  }

  std::string dataScope(const std::string &name) {
    std::string local = "local.";
    std::map<std::string, std::string> mem_dict{{BUF, local + BUF}, {C1, local + C1},   {C0A, local + C0A},
                                                {C0B, local + C0B}, {C0C, local + C0C}, {REG, local + REG}};
    std::vector<std::string> split_list = akg::common::Split(name, ".");
    if (split_list.size() == 1) {
      split_list = akg::common::Split(name, "_local_");
    }

    std::string key = split_list[split_list.size() - 1];
    for (auto &iter : mem_dict) {
      std::string::size_type pos = split_list[split_list.size() - 1].find(iter.first);
      if (pos != std::string::npos) {
        key = iter.first;
        break;
      }
    }
    if (split_list.size() == 1) {
      return "global";
    }

    return mem_dict[key];
  }

 private:
  std::set<const Variable *> gather_vars;
  std::unordered_map<const Node *, const Node *> gather_write_sink_;
};

class GatherTransform : public IRMutator {
 public:
  explicit GatherTransform(const TensorVarTab &tab) : tab_(tab) {}
  ~GatherTransform() { for_ = nullptr; }

  Stmt run(const Stmt s) {
    bool all_zero = true;
    TensorVarTab filter_tab;
    for (auto item : tab_) {
      if (item.second.size() > 0) {
        all_zero = false;
        filter_tab[item.first] = item.second;
      }
    }

    if (!all_zero) {
      tab_ = filter_tab;
      const Stmt &res = this->Mutate(s);
      return res;
    }

    return s;
  }

  Stmt MakeRegAssign(const Var &var, const Stmt &body) {
    if (!isType<EQ>(condition)) {
      return body;
    }
    // get assign value
    Expr left = condition.as<EQ>()->a;
    Expr right = condition.as<EQ>()->b;
    auto assign_eq = [&](const Expr &arith, const Expr &call) {
      Expr simple = Simplify(arith - var);
      return Simplify(call - simple);
    };

    Expr assignValue = isType<Call>(right) ? assign_eq(left, right) : assign_eq(right, left);

    Stmt new_store = Store::make(repl_var, assignValue, make_const(Int(32), 0), Expr(1));
    Stmt temp1 = Block::make(new_store, body);
    Stmt new_allo = Allocate::make(repl_var, Int(32), {make_const(Int(32), 1)}, const_true(), temp1);
    Stmt new_attr = AttrStmt::make(repl_var, air::ir::attr::storage_scope, StringImm::make("local.REG"), new_allo);
    return new_attr;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (tab_.count(op->loop_var.get()) > 0) {
      for_ = op;
      // initialize buffer table
      mem_buffer_tab_.clear();
      // make register variable
      std::string reg_name = "reg" + std::to_string(reg_cnt) + "_local_REG";
      ++reg_cnt;
      repl_var = Variable::make(Int(32), reg_name);

      Stmt res = Mutate(op->body);
      if (need_transform_ && isType<EQ>(condition)) {
        need_transform_ = false;
        for_ = nullptr;
        return MakeRegAssign(op->loop_var, res);
      } else {
        for_ = nullptr;
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, op->body);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (need_transform_ && tab_.count(op)) {
      return Load::make(Int(32), repl_var, Expr(0), Expr(1));
    }
    return IRMutator::Mutate_(op, e);
  }

  bool tableFind(const Variable *key1, const IfThenElse *key2) {
    if (tab_.count(key1) > 0) {
      for (const auto item : tab_[key1]) {
        if (item == key2) {
          return true;
        }
      }
    }
    return false;
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto var_in_condition = for_ != nullptr ? for_->loop_var.get() : nullptr;
    if (var_in_condition != nullptr && tableFind(var_in_condition, op)) {
      need_transform_ = true;
      condition = op->condition;
      Stmt res = IRMutator::Mutate(op->then_case);
      if (condition.as<EQ>() != nullptr && condition.as<EQ>()->a.as<Add>()) {
        // tiling tensor of tensor case
        Expr cond = GE::make(Load::make(Int(32), repl_var, Expr(0), Expr(1)), Expr(0));
        Expr cond_less = LT::make(Load::make(Int(32), repl_var, Expr(0), Expr(1)), for_->extent);
        cond = And::make(cond, cond_less);
        return IfThenElse::make(cond, res, op->else_case);
      }
      condition_removed_repl_vars.insert(repl_var.get());
      return res;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Expr condition;
  int reg_cnt{0};
  TensorVarTab tab_;
  VarExpr repl_var;
  bool need_transform_{false};
  const For *for_{nullptr};
  std::unordered_map<std::string, int> mem_buffer_tab_;
  std::set<const Variable *> condition_removed_repl_vars;
};

class InductionVarElinate : public IRMutator {
 public:
  Stmt Run(const Stmt &s) { return Mutate(s); }

 private:
  bool inductionExprCheck(const Expr &e) {
    Var var("var", e.type());
    Expr pattern1 = ((var / (var + 1)) + 1);
    Expr pattern2 = (1 + Div::make(var - 1, var));
    Expr pattern3 = (Div::make(var - 1, var) + 1);
    if (ExprPatternMatch(e, pattern1) || ExprPatternMatch(e, pattern2) || ExprPatternMatch(e, pattern3)) {
      return true;
    }
    return false;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (inductionExprCheck(op->extent)) {
      if (elinate_vars.count(op->loop_var.get()) == 0) {
        elinate_vars[op->loop_var.get()] = Expr(0);
      }
      return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *var, const Expr &e) final {
    if (elinate_vars.count(var) > 0) {
      return elinate_vars[var];
    }
    return IRMutator::Mutate_(var, e);
  }

  std::unordered_map<const Variable *, Expr> elinate_vars;
};

class DynamicPaddingFix : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    bool need_fix = false;
    PostOrderVisit(s, [&](const NodeRef &descendant) {
      if (auto call = descendant.as<Call>()) {
        if (call->name == "load_im2col_c1_buf") {
          need_fix = true;
        }
      }
    });
    return need_fix ? Mutate(s) : s;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    auto ph = op->func.as<PlaceholderOpNode>();
    CHECK(ph);
    if (ph->name.find(LOCAL_C1) != std::string::npos) {
      fm_l1_ = ph->name;
      CHECK(op->body.as<For>());
      fm_h_ = op->body.as<For>();
      if (fm_h_->body.as<For>()) {
        fm_w_ = fm_h_->body.as<For>();
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (op->func.as<PlaceholderOpNode>() &&
        op->func.as<PlaceholderOpNode>()->name.find(LOCAL_C1) != std::string::npos) {
      Stmt body = Mutate(op->body);
      Array<Range> bounds;
      for (auto item : op->bounds) {
        isImm(item->extent) ? bounds.push_back(item) : bounds.push_back(Range(item->min, fm_w_var_));
      }
      return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op == fm_w_) {
      Stmt body = Mutate(op->body);
      Expr ext = op->extent;
      std::vector<Var> vec;
      GatherVars(ext, &vec);
      for (auto item : vec) {
        if (item.get()->name_hint.find("I") != std::string::npos) {
          fm_w_var_ = item;
          ext = item;
        }
      }
      return For::make(op->loop_var, op->min, ext, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func.as<PlaceholderOpNode>() && op->func.as<PlaceholderOpNode>()->name == fm_l1_) {
      Array<Expr> new_args;
      for (auto item : op->args) {
        new_args.push_back(MutateExpr(item));
      }
      return Provide::make(op->func, op->value_index, op->value, new_args);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr MutateExpr(const Expr &e) {
    const auto v = e.as<Variable>();
    if (v == nullptr) return e;
    auto addOffset = [](const Variable *var, const For *for_var) {
      if (for_var != nullptr && var == for_var->loop_var.get() && !is_zero(for_var->min)) {
        return Sub::make(Expr(0), for_var->min);
      }
      return Expr(0);
    };
    return Simplify(e + addOffset(v, fm_h_) + addOffset(v, fm_w_));
  }

 private:
  const For *fm_h_{nullptr};
  const For *fm_w_{nullptr};
  Var fm_w_var_;
  std::string fm_l1_{""};
};

Stmt DsaHalideOptimizer(const Stmt &s, bool dynamic_shape = false) {
  Stmt stmt = s;
  if (dynamic_shape) {
    stmt = InductionVarElinate().Run(s);
    stmt = Simplify_cce(stmt);
    stmt = DynamicPaddingFix().Run(stmt);
  }
  OpDetector detector;
  detector.detect(stmt);
  switch (detector.type_) {
    case OP_TYPE::T_PASS_DOWN:
      return PassDownForAxis().run(stmt);
    case OPERATOR_TYPE::T_TENSOR_OF_TENSOR:
      return GatherTransform(detector.tab_).run(stmt);
    case OPERATOR_TYPE::T_TENSOR_OF_TENSOR_ACCUM:
      return ElimTensorIdx(detector.elim_for_set_, detector.tensor_map_, detector.elim_if_).run(stmt);
    default:
      return stmt;
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
