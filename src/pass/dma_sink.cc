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
#include <tvm/operation.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <pass/ir_util.h>
#include <cmath>
#include <unordered_set>
#include <unordered_map>
#include <map>

/*
 * realize input_data_local_UB([0, 1024], [0, 2048]) {
 * produce input_data_local_UB {
 * for(cc0, 0, 8192)
 *     for(cc1, 0, 1024){
 *          input_data_local_UB(cc0, cc1) = input_data(cc0, cc1)
 *     }
 *
 * for(cc2, 0, 2)
 *  for(cc0, 0, 8192)
 *    for(cc1, 0, 1024)
 *      if(ids_input_local_UB(cc0) == Expr(cc2)){
 *            out_local_UB[cc2, cc1] = out_local_UB[cc2, cc1] +  input_data_local_UB[cc0, cc1]
 *       }
 *
 * To
 *
 * for(cc2, 0, 2)
 *   for(cc0, 0, 8192)
 *      if(ids_input_local_UB(cc0) == Expr(cc2)){
 *            realize input_data_local_UB([0, 1], [0, 1024]) {
 *            produce input_data_local_UB {
 *            // attr ["input_data_local_UB"]
 *            for(cc1, 0, 1024) {
 *                input_data_local_UB[0, cc1] = input_data[cc0, cc1]
 *            }
 *            for(cc1, 0, 1024) {
 *                out_local_UB[cc2, cc1] = out_local_UB[cc2, cc1] +  input_data_local_UB[0, cc1]
 *            }
 *       }
 */

namespace akg {
namespace ir {
class DMASinker : public IRMutator {
 public:
  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    secondPass_ = true;
    stmt = this->Mutate(stmt);
    return stmt;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt ret = IRMutator::Mutate_(op, s);
    if (secondPass_ && findOp_) {
      if (pCall_ && op->func.get() == pCall_->func.get()) {
        if (ret.as<Realize>()) {
          return ret.as<Realize>()->body;
        }
      }
    }
    return ret;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (!secondPass_) {
      inProduce_ = true;
      Stmt ret = IRMutator::Mutate_(op, s);
      inProduce_ = false;
      return ret;
    } else {
      auto ret = IRMutator::Mutate_(op, s);
      if (findOp_ && pCall_ && op->func.get() == pCall_->func.get()) {
        return Evaluate::make(0);
      }
      return ret;
    }
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!secondPass_) {
      forExtent_[op->loop_var.get()] = &(op->extent);
      Stmt ret = IRMutator::Mutate_(op, s);
      forExtent_.erase(op->loop_var.get());
      return ret;
    } else {
      forVars_[op->loop_var.get()] = op;
      Stmt body = this->Mutate(op->body);
      if (op->loop_var.get() == loopVar_) {
        loopVar_ = nullptr;
        forVars_.erase(op->loop_var.get());
        return body;
      } else {
        forVars_.erase(op->loop_var.get());
        Stmt newFor = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
        return newFor;
      }
    }
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    int countNum = 0;
    auto _GetCondLoadNums = [&countNum](const NodeRef &op) {
      if (op->IsInstance<Call>() && op.as<Call>()->call_type == Call::CallType::Halide) {
        countNum = countNum + 1;
      }
    };
    PostOrderVisit(op->condition, _GetCondLoadNums);
    if (countNum > 0 && op->condition->IsInstance<EQ>()) {
      isLoadCond_ = true;
      pIfCond = &(op->condition);
    }
    auto ret = IRMutator::Mutate_(op, s);
    isLoadCond_ = false;
    return ret;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (inProduce_ && !secondPass_ && op->value.as<Call>()) {
      //  Mutate Condition: The size of the input data from gm is more than the UB size
      const int MaxInt32 = 2147483647;
      int num = 1;
      for (auto item : op->args) {
        const auto temp = item.as<Variable>();
        if (temp != nullptr && forExtent_.count(temp) > 0 && forExtent_[temp]->as<IntImm>()) {
          int tExtent = static_cast<int>(forExtent_[temp]->as<IntImm>()->value);
          CHECK_NE(num, 0);
          if (tExtent < MaxInt32 / num) {
            num = num * tExtent;
          } else {
            return s;
          }
        } else {
          return s;
        }
      }
      if (num >= MaxNumBits_) {
        bufferProducer_[op->func.get()] = op;
      }
    }

    inProvide_ = true;
    auto ret = IRMutator::Mutate_(op, s);
    inProvide_ = false;

    if (secondPass_ && rebuildStmt_) {
      CHECK(gmToUB_);
      auto callOp = gmToUB_->value.as<Call>();
      if (callOp && pCall_) {
        CHECK_GT(forVars_.count(loopVar_), 0);
        auto externFor = forVars_[loopVar_];
        CHECK(externFor);
        // new gm to ub
        Array<Expr> newArgs;
        newArgs.push_back(Expr(0));
        newArgs.push_back(externFor->loop_var);
        Array<Expr> argsCall;
        CHECK(!callOp->args.empty());
        CHECK(!pCall_->args.empty());
        if (callOp->args[0].as<Add>()) {
          argsCall.push_back(callOp->args[0].as<Add>()->a + pCall_->args[0]);
        } else {
          argsCall.push_back(pCall_->args[0]);
        }
        argsCall.push_back(externFor->loop_var);
        Expr new_call =
          Call::make(callOp->type, callOp->name, argsCall, callOp->call_type, callOp->func, callOp->value_index);
        Stmt bufferLoad = Provide::make(gmToUB_->func, 0, new_call, newArgs);
        Stmt vecLoad = For::make(externFor->loop_var, externFor->min, externFor->extent, externFor->for_type,
                                 externFor->device_api, bufferLoad);
        Stmt bufferProducer_new = ProducerConsumer::make(pCall_->func, true, vecLoad);

        // new For
        Stmt temp = For::make(externFor->loop_var, externFor->min, externFor->extent, externFor->for_type,
                              externFor->device_api, ret);

        // new Realize
        Stmt block = Block::make(bufferProducer_new, temp);
        Region new_bounds;
        new_bounds.push_back(Range::make_by_min_extent(IntImm::make(Int(32), 0), IntImm::make(Int(32), 1)));
        new_bounds.push_back(Range::make_by_min_extent(IntImm::make(Int(32), 0), externFor->extent));
        auto ret_new = Realize::make(pCall_->func, 0, pCall_->type, new_bounds, const_true(1), block);
        rebuildStmt_ = false;
        return ret_new;
      }
      return ret;
    } else {
      rebuildStmt_ = false;
      return ret;
    }
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (!secondPass_) {
      // dma sink situations:
      // 1. in if condition and in provide
      // 2. one src is copy from gm, that is in bufferProducer_
      // 3. the first arg of the target src is in the if condition, eg., A[cc0, cc1], cc0 in the if condition

      if (isLoadCond_ && inProvide_ && bufferProducer_.count(op->func.get()) > 0 && op->args.size() == 2) {
        Expr firstVar = op->args[0];
        bool findVar = false;
        auto _VarInCondExpr = [=, &findVar](const NodeRef &op) {
          if (op->IsInstance<Variable>()) {
            if (firstVar == op) {
              findVar = true;
            }
          }
        };
        PostOrderVisit(*pIfCond, _VarInCondExpr);
        if (findVar) {
          gmToUB_ = bufferProducer_[op->func.get()];
          pCall_ = op;
          loopVar_ = op->args[1].as<Variable>();
          findOp_ = true;
        }
      }
      return e;
    } else if (secondPass_ && findOp_ && op->call_type == Call::CallType::Halide && pCall_ &&
               op->func.get() == pCall_->func.get()) {
      Array<Expr> new_args;
      new_args.push_back(Expr(0));
      CHECK_GE(op->args.size(), 2);
      new_args.push_back(op->args[1]);
      auto new_call = Call::make(op->type, op->name, new_args, op->call_type, op->func, op->value_index);
      rebuildStmt_ = true;
      return new_call;
    } else {
      return e;
    }
  }

 private:
  std::unordered_map<const Node *, const Provide *> bufferProducer_;
  std::unordered_map<const Variable *, const For *> forVars_;
  std::unordered_map<const Variable *, const Expr *> forExtent_;

  const Expr *pIfCond{nullptr};
  const Provide *gmToUB_{nullptr};
  const Call *pCall_{nullptr};
  const Variable *loopVar_{nullptr};
  const int MaxNumBits_ = 534288;

  bool secondPass_{false};
  bool findOp_{false};
  bool rebuildStmt_{false};
  bool isLoadCond_{false};
  bool inProduce_{false};
  bool inProvide_{false};
};

Stmt DMASink(Stmt stmt) {
  stmt = DMASinker().Run(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
