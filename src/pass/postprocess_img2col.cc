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
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include "ir_pass.h"
#include "poly/poly_util.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
class Im2colCheck : public IRVisitor {
 public:
  bool IsLoad3d() const { return is_load3d_; }

 private:
  void Visit_(const Evaluate *op) override {
    if (auto call = op->value.as<Call>()) {
      constexpr int IM2COLARGNUM = 23;
      if (call->name == CALL_IM2COL_UB && call->args.size() == IM2COLARGNUM) {
        is_load3d_ = true;
      }
    }
    IRVisitor::Visit_(op);
  }

  bool is_load3d_{false};
};

class ReorderLoad3d : public IRMutator {
 public:
  ReorderLoad3d() = default;
  ~ReorderLoad3d() override = default;
  int LastVarNum() const { return last_var_num_; }

 private:
  Stmt Mutate_(const Block *op, const Stmt &s) final {
    auto first = op->first.as<For>();
    auto rest = op->rest.as<For>();
    if (first && rest) {
      reorder_ = true;
      ClearStack();
      Stmt new_first = Mutate(op->first);
      new_first = StackApply(new_first);
      ClearStack();
      Stmt new_rest = Mutate(op->rest);
      new_rest = StackApply(new_rest);
      reorder_ = false;
      return Block::make(new_first, new_rest);
    }
    return IRMutator::Mutate_(op, s);
  }

  void ClearStack() { for_stack_.clear(); }

  Stmt StackApply(const Stmt &s) {
    Stmt res = s;
    for (auto item : for_stack_) {
      res = For::make(item->loop_var, item->min, item->extent, item->for_type, item->device_api, res);
    }
    return res;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (reorder_ && !is_zero(Simplify_cce(op->extent - Expr(16)))) {
      for_stack_.push_back(op);
      return Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    std::string var_name = op->name_hint;
    std::string prefix = "cc";
    std::size_t pos = var_name.find(prefix);
    if (pos != std::string::npos) {
      CHECK_GE(var_name.size(), pos + prefix.size());
      std::string index = var_name.substr(pos + prefix.size(), var_name.size() - prefix.size());
      int index_value = static_cast<int>(std::strtol(index.c_str(), nullptr, 10));
      if (index_value > last_var_num_) {
        last_var_num_ = index_value;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  bool reorder_{false};
  std::vector<const For *> for_stack_;
  int last_var_num_{0};
};

class PostFusionLoad3d : public IRMutator {
 public:
  explicit PostFusionLoad3d(int var_idx) : var_idx_(var_idx) {}
  ~PostFusionLoad3d() override = default;

 private:
  Stmt Mutate_(const Block *op, const Stmt &s) final {
    auto first = op->first.as<For>();
    auto rest = op->rest.as<For>();
    if (first && rest) {
      mo_name_ = first->loop_var->name_hint;
      is_inside_mi_ = true;
      Stmt first_body = Mutate(first->body);
      CHECK(rest->body.as<For>());
      auto new_rest_body = Mutate_(rest->body.as<For>(), rest->body);
      new_rest_body = SubstituteLoopVar(new_rest_body, rest->loop_var.get(), Expr(first->loop_var));
      is_inside_mi_ = false;
      Stmt body = Block::make(first_body, new_rest_body);
      return For::make(first->loop_var, first->min, first->extent, first->for_type, first->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Region bounds;
    Stmt body = this->Mutate(op->body);
    size_t pos_UB = op->func->func_name().find("_local_UB");
    bool is_UB = pos_UB != std::string::npos;
    for (size_t i = 0; i < op->bounds.size(); ++i) {
      if (is_UB && i <= 1) {
        bounds.push_back(Range(0, 1));
      } else {
        bounds.push_back(op->bounds[i]);
      }
    }
    return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "buffer_bind_scope") {
      Stmt body = Mutate(op->body);
      const Call *tuple = op->value.as<Call>();
      CHECK(tuple && tuple->is_intrinsic(air::ir::intrinsic::tvm_tuple));
      Array<Expr> new_args;
      for (auto item : tuple->args) {
        if (item.as<Variable>() && item.as<Variable>()->name_hint != mo_name_) {
          new_args.push_back(Expr(0));
        } else {
          new_args.push_back(item);
        }
      }
      Expr value = Call::make(tuple->type, tuple->name, new_args, tuple->call_type, tuple->func, tuple->value_index);
      return AttrStmt::make(op->node, op->attr_key, value, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  std::string IncreaseIdx(const std::string &var_name) {
    std::string prefix = "cc";
    std::size_t pos = var_name.find(prefix);
    std::string res = var_name;
    if (pos != std::string::npos) {
      var_idx_++;
      std::ostringstream os;
      os << prefix << var_idx_;
      res = os.str();
    }
    return res;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_inside_mi_) {
      auto body = this->Mutate(op->body);
      Var mi(IncreaseIdx(op->loop_var->name_hint));
      auto new_body = SubstituteLoopVar(body, op->loop_var.get(), Expr(mi));
      return For::make(mi, op->min, op->extent, op->for_type, op->device_api, new_body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string mo_name_;
  bool is_inside_mi_{false};
  int var_idx_{0};
};

class FixRealizeLoad3d : public IRMutator {
 public:
  FixRealizeLoad3d() = default;
  ~FixRealizeLoad3d() override = default;

  Stmt Shape(const Stmt &s) { return Mutate(s); }

 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    judge_func_ = [](const std::string &s, const std::string &t) { return s.find(t) != std::string::npos; };

    Region bounds;
    Stmt body = this->Mutate(op->body);
    if (judge_func_(op->func->func_name(), "_local_UB")) {
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        if (idx_marker_.count(i) == 0) {
          bounds.push_back(Range(0, 1));
        } else {
          bounds.push_back(op->bounds[i]);
        }
      }
      idx_marker_.clear();
    }

    if (judge_func_(op->func->func_name(), "_local_L1")) {
      for (size_t i = 0; i < op->bounds.size(); ++i) {
        if (realize_L1_range_.count(i) > 0) {
          bounds.push_back(Range(0, realize_L1_range_[i]));
        } else {
          bounds.push_back(op->bounds[i]);
        }
      }
      realize_L1_range_.clear();
    }

    return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, body);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    auto first = op->first.as<For>();
    auto rest = op->rest.as<For>();
    if (first && rest) {
      Stmt first_body = Mutate(op->first);
      ub_for_collecter_.clear();
      Stmt rest_body = Mutate(op->rest);
      Stmt result = Block::make(first_body, rest_body);
      return result;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    is_L1_ = judge_func_(op->func->func_name(), "_local_L1");
    realize_L1_info_.clear();
    auto res = IRMutator::Mutate_(op, s);
    is_L1_ = false;
    return res;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_L1_ && realize_L1_info_.count(op->loop_var.get()) == 0) {
      realize_L1_info_[op->loop_var.get()] = op->extent;
    }
    if (ub_for_collecter_.count(op->loop_var.get()) == 0) {
      ub_for_collecter_[op->loop_var.get()] = true;
    }
    if (Compare(op->min, Expr(0)) > 0) {
      if (non_zero_L1_info_.count(op->loop_var.get()) == 0) {
        non_zero_L1_info_[op->loop_var.get()] = op->min;
      }
      Stmt body = Mutate(op->body);
      return For::make(op->loop_var, Expr(0), op->extent, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *cal, const Expr &e) final {
    if (is_L1_) {
      for (size_t idx = 0; idx < cal->args.size(); ++idx) {
        const auto var = cal->args[idx].as<Variable>();
        if (var && realize_L1_info_.count(var) > 0) {
          if (realize_L1_range_.count(idx) > 0) {
            // update value
            if (Compare(realize_L1_range_[idx], realize_L1_info_[var]) < 0) {
              realize_L1_range_[idx] = realize_L1_info_[var];
            }
          } else {
            // initial value
            realize_L1_range_[idx] = realize_L1_info_[var];
          }
        }
      }
    }

    if (cal->name.find("_local_UB") != std::string::npos) {
      for (size_t idx = 0; idx < cal->args.size(); ++idx) {
        auto item = cal->args[idx];
        if (item.as<Variable>() && ub_for_collecter_.count(item.as<Variable>()) > 0) {
          idx_marker_[idx] = true;
        }
      }
    }

    if (!non_zero_L1_info_.empty()) {
      Array<Expr> args;
      for (const auto &i : cal->args) {
        if (Compare(i, Expr(0)) == 0) {
          args.push_back(i);
        } else if (i.as<Variable>() && non_zero_L1_info_.count(i.as<Variable>()) == 0) {
          args.push_back(i);
        } else {
          Expr arg = i;
          const Variable *key = nullptr;
          for (const auto &item : non_zero_L1_info_) {
            std::map<const Variable *, Expr> varMap;
            varMap[item.first] = item.second;
            auto tmp = Simplify_cce(substitute(varMap, i));
            if (Compare(tmp, Expr(0)) == 0) {
              key = item.first;
              break;
            }
          }
          if (key != nullptr) {
            arg = Simplify_cce(arg + non_zero_L1_info_[key]);
          }
          args.push_back(Simplify_cce(arg));
        }
      }
      Expr res = Call::make(cal->type, cal->name, args, cal->call_type, cal->func, cal->value_index);
      non_zero_L1_info_.clear();
      return res;
    }

    return IRMutator::Mutate_(cal, e);
  }

 private:
  std::unordered_map<const Variable *, Expr> realize_L1_info_;
  std::unordered_map<const Variable *, Expr> non_zero_L1_info_;
  std::unordered_map<size_t, Expr> realize_L1_range_;
  std::unordered_map<size_t, bool> idx_marker_;
  std::unordered_map<const Variable *, bool> ub_for_collecter_;
  std::function<bool(const std::string &, const std::string &)> judge_func_;
  bool is_L1_{false};
};

Stmt PostProcessImg2col(Stmt stmt) {
  Im2colCheck checker;
  checker.Visit(stmt);
  if (!checker.IsLoad3d()) {
    return stmt;
  }
  ReorderLoad3d reorder;
  stmt = reorder.Mutate(stmt);
  PostFusionLoad3d actor(reorder.LastVarNum());
  stmt = actor.Mutate(stmt);
  FixRealizeLoad3d fixed;
  stmt = fixed.Shape(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
