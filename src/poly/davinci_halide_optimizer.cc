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

#include <tvm/target_info.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include <iostream>
#include <stack>

#include "pass/utils.h"
#include "tvm.h"
#include "emit_insn/insn_info.h"
#include "emit_insn/cce_params.h"

namespace akg {
namespace ir {
namespace poly {
constexpr auto PASS_DOWN = "pass_down";
enum OPERATOR_TYPE { T_OUTOF_LIB = 0, T_PASS_DOWN, T_TENSOR_OF_TENSOR };
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
    inRealize_ = true;
    IRVisitor::Visit_(op);
    inRealize_ = false;
  }

  void Visit_(const ProducerConsumer *op) final {
    if (memBufferTab_.count(op->func->func_name()) == 0) {
      memBufferTab_[op->func->func_name()] = 0;
    } else {
      memBufferTab_[op->func->func_name()]++;
    }
    return IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (inRealize_) {
      if (tab_.count(op->loop_var.get()) == 0) {
        std::vector<const IfThenElse *> value;
        tab_[op->loop_var.get()] = value;
      }
    }
    return IRVisitor::Visit_(op);
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
      const auto addExpr = expr.as<Add>();
      res.push_back(tabAdd(addExpr->a, tab_));
      res.push_back(tabAdd(addExpr->b, tab_));
    } else if (isType<Variable>(expr)) {
      res.push_back(tabAdd(expr, tab_));
    }
    return res;
  }

  void Visit_(const IfThenElse *op) final {
    CHECK(op);
    auto emptyCase = op->else_case;
    if (emptyCase == Stmt() && isType<EQ>(op->condition)) {
      const auto equation = op->condition.as<EQ>();
      auto checkTOfTensor = [](const Expr &e, const std::unordered_map<std::string, int> &table) {
        if (isType<Call>(e) && table.count(e.as<Call>()->name) > 0) {
          return true;
        }
        return false;
      };
      if (checkTOfTensor(equation->a, memBufferTab_) || checkTOfTensor(equation->b, memBufferTab_)) {
        type_ = OP_TYPE::T_TENSOR_OF_TENSOR;
        std::vector<const Variable *> vars;
        vars = checkTOfTensor(equation->b, memBufferTab_) ? GetExprSpecVar(equation->a) : GetExprSpecVar(equation->b);
        for (const auto var : vars) {
          if (tab_.count(var) > 0) {
            tab_[var].push_back(op);
          }
        }
        return;
      }
    }
    return IRVisitor::Visit_(op);
  }

  OP_TYPE type_{OP_TYPE::T_OUTOF_LIB};
  TensorVarTab tab_;

 private:
  bool inRealize_{false};
  std::unordered_map<std::string, int> memBufferTab_;
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
      needPassDown_ = true;
      Stmt stmt = this->Mutate(op->body);
      needPassDown_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    CHECK(op);
    if (needPassDown_ && passDownFor_ == nullptr) {
      passDownFor_ = op;
      Stmt stmt = this->Mutate(op->body);
      passDownFor_ = nullptr;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    CHECK(op);
    if (needPassDown_ && passDownFor_ != nullptr) {
      Stmt stmt = IRMutator::Mutate_(op, s);
      return For::make(passDownFor_->loop_var, passDownFor_->min, passDownFor_->extent, passDownFor_->for_type,
                       passDownFor_->device_api, stmt);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool needPassDown_{false};
  const For *passDownFor_{nullptr};
};

/***********************************************************************
 * After the condition is removed, the sinked tensor write back to promotion
 // attr [placeholder(gather_output_local_UB, 0x2f725f0)] realize_scope = "local.UB"
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

 * // attr [placeholder(gather_output_local_UB, 0x2f725f0)] realize_scope = "local.UB"
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
  explicit GatherWritePromotion(const std::set<const Variable *> &conditionRemovedGatherVars)
      : gatherVars(conditionRemovedGatherVars) {}
  ~GatherWritePromotion() override = default;

  Stmt run(const Stmt &s) {
    std::unordered_map<std::string, std::string> gatherWriteTensors;
    auto getGatherWriteTensor_ = [&, this](const NodeRef &node) { this->getGatherWriteSink(node); };
    PostOrderVisit(s, getGatherWriteTensor_);
    return Mutate(s);
  }

  Stmt getWritePromotion(const std::unordered_map<const Variable *, const For *> &var2For, const Stmt &write) {
    //
    std::stack<const Variable *> srcVars;
    auto provide = write.as<Provide>();
    CHECK(provide);
    auto src = provide->value;
    PostOrderVisit(src, [&srcVars](const NodeRef &node) {
      if (auto var = node.as<Variable>()) {
        srcVars.push(var);
      }
    });

    Stmt writePromotion = write;
    while (!srcVars.empty()) {
      auto var = srcVars.top();
      CHECK_GT(var2For.count(var), 0);
      auto for_ = var2For.at(var);
      writePromotion =
        For::make(for_->loop_var, for_->min, for_->extent, for_->for_type, for_->device_api, writePromotion);
      srcVars.pop();
    }
    return writePromotion;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    auto tensor = op->func.get();
    if (gatherWriteSink_.count(tensor) > 0) {
      auto gatherWriteTensor = tensor;
      auto gatherWriteTensorGm_ = gatherWriteSink_[tensor];
      bool record = true;
      std::unordered_map<const Variable *, const For *> var2For;
      Stmt gatherWriteProvide_;
      PackedFunc recordFor = PackedFunc([&](TVMArgs args, TVMRetValue *ret) {
        Stmt st = args[0];
        if (auto for_ = st.as<For>()) {
          if (record) {
            var2For[for_->loop_var.get()] = for_;
          }
        }
      });

      PackedFunc eliminateGatherWrite = PackedFunc([&](TVMArgs args, TVMRetValue *ret) {
        Stmt st = args[0];
        if (auto provide = st.as<Provide>()) {
          auto dst = provide->func;
          auto src = provide->value.as<Call>();
          if (dst.get() == gatherWriteTensorGm_ && src && src->func.get() == gatherWriteTensor) {
            record = false;
            gatherWriteProvide_ = st;
            *ret = Evaluate::make(0);
          }
        }
      });
      auto body = air::ir::IRTransform(op->body, recordFor, eliminateGatherWrite, {Expr("Provide"), Expr("For")});

      CHECK(gatherWriteProvide_.defined());
      Stmt writePromotion = getWritePromotion(var2For, gatherWriteProvide_);
      body = Block::make(body, writePromotion);
      return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  void getGatherWriteSink(const NodeRef &node) {
    const Variable *gatherVar = nullptr;
    auto op = node.as<Block>();
    if (op && op->first.as<Store>()) {
      auto store = node.as<Block>()->first.as<Store>();
      if (gatherVars.count(store->buffer_var.get()) > 0) {
        gatherVar = store->buffer_var.get();
      }
    }
    if (gatherVar == nullptr) {
      return;
    }

    const Node *gatherWriteSinkTensor = nullptr;
    const Node *gatherWriteSinkTensorGm = nullptr;
    PostOrderVisit(op->rest, [&gatherVar, &gatherWriteSinkTensor, &gatherWriteSinkTensorGm](const NodeRef &node) {
      if (gatherWriteSinkTensor == nullptr) {
        if (auto provide = node.as<Provide>()) {
          bool valueHasGatherVar = false;
          PostOrderVisit(provide->value, [&gatherVar, &valueHasGatherVar](const NodeRef &node_) {
            auto load = node_.as<Load>();
            if (load && load->buffer_var.get() == gatherVar) {
              valueHasGatherVar = true;
            }
          });
          if (valueHasGatherVar && provide->func->IsInstance<OperationNode>()) {
            gatherWriteSinkTensor = provide->func.get();
          }
        }
      } else {
        if (auto provide = node.as<Provide>()) {
          if (auto value = provide->value.as<Call>()) {
            if (value->func.get() == gatherWriteSinkTensor &&
                GetBufScope(provide->func->func_name()) == DMA_COPY_GLOBAL) {
              gatherWriteSinkTensorGm = provide->func.get();
            }
          }
        }
      }
    });

    if (gatherWriteSinkTensorGm != nullptr) {
      gatherWriteSink_[gatherWriteSinkTensor] = gatherWriteSinkTensorGm;
    }
  }

 private:
  std::set<const Variable *> gatherVars;
  std::unordered_map<const Node *, const Node *> gatherWriteSink_;
};

class GatherTransform : public IRMutator {
 public:
  explicit GatherTransform(const TensorVarTab &tab) : tab_(tab) {}
  ~GatherTransform() { for_ = nullptr; }

  Stmt run(const Stmt s) {
    bool allZero = true;
    TensorVarTab filterTab;
    for (auto item : tab_) {
      if (item.second.size() > 0) {
        allZero = false;
        filterTab[item.first] = item.second;
      }
    }

    if (!allZero) {
      tab_ = filterTab;
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
    auto assignEq = [&](const Expr &arith, const Expr &call) {
      Expr simple = Simplify(arith - var);
      return Simplify(call - simple);
    };

    Expr assignValue = isType<Call>(right) ? assignEq(left, right) : assignEq(right, left);

    Stmt newStore = Store::make(replVar, assignValue, make_const(Int(32), 0), Expr(1));
    Stmt temp1 = Block::make(newStore, body);
    Stmt newAllo = Allocate::make(replVar, Int(32), {make_const(Int(32), 1)}, const_true(), temp1);
    Stmt newAttr = AttrStmt::make(replVar, air::ir::attr::storage_scope, StringImm::make("local.REG"), newAllo);
    return newAttr;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (tab_.count(op->loop_var.get()) > 0) {
      for_ = op;
      // initialize buffer table
      memBufferTab_.clear();
      // make register variable
      std::string regName = "reg" + std::to_string(regCnt) + "_local_REG";
      ++regCnt;
      replVar = Variable::make(Int(32), regName);

      Stmt res = Mutate(op->body);
      if (needTransform_ && isType<EQ>(condition)) {
        needTransform_ = false;
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
    if (needTransform_ && tab_.count(op)) {
      return Load::make(Int(32), replVar, Expr(0), Expr(1));
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
    auto varInCondition = for_ != nullptr ? for_->loop_var.get() : nullptr;
    if (varInCondition != nullptr && tableFind(varInCondition, op)) {
      needTransform_ = true;
      condition = op->condition;
      Stmt res = IRMutator::Mutate(op->then_case);
      if (condition.as<EQ>() != nullptr && condition.as<EQ>()->a.as<Add>()) {
        // tiling tensor of tensor case
        Expr cond = GE::make(Load::make(Int(32), replVar, Expr(0), Expr(1)), Expr(0));
        Expr cond_less = LT::make(Load::make(Int(32), replVar, Expr(0), Expr(1)), for_->extent);
        cond = And::make(cond, cond_less);
        return IfThenElse::make(cond, res, op->else_case);
      }
      conditionRemovedReplVars.insert(replVar.get());
      return res;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Expr condition;
  int regCnt{0};
  TensorVarTab tab_;
  VarExpr replVar;
  bool needTransform_{false};
  const For *for_{nullptr};
  std::unordered_map<std::string, int> memBufferTab_;
  std::set<const Variable *> conditionRemovedReplVars;
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
      if (elinateVars.count(op->loop_var.get()) == 0) {
        elinateVars[op->loop_var.get()] = Expr(0);
      }
      return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *var, const Expr &e) final {
    if (elinateVars.count(var) > 0) {
      return elinateVars[var];
    }
    return IRMutator::Mutate_(var, e);
  }

  std::unordered_map<const Variable *, Expr> elinateVars;
};

class DynamicPaddingFix : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    bool need_fix = false;
    PostOrderVisit(s, [&](const NodeRef &descendant) {
      if (auto call = descendant.as<Call>()) {
        if (call->name == "load3d_l1_ub") {
          need_fix = true;
        }
      }
    });
    return need_fix ? Mutate(s) : s;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    auto ph = op->func.as<PlaceholderOpNode>();
    CHECK(ph);
    if (ph->name.find("_local_L1") != std::string::npos) {
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
        op->func.as<PlaceholderOpNode>()->name.find("_local_L1") != std::string::npos) {
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

Stmt DavinciHalideOptimizer(const Stmt &s, bool dynamicShape = false) {
  Stmt stmt = s;
  if (dynamicShape) {
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
    default:
      return stmt;
  }
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
