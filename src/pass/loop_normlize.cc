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
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_visitor.h>
#include <queue>
#include <algorithm>
#include "pass/utils.h"
#include "build_module.h"

namespace akg {
namespace ir {
class SimplifyMod : public IRMutator {
 public:
  SimplifyMod() {}
  ~SimplifyMod() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    deq_outer_loops_.push_front(op);
    Stmt stmt = IRMutator::Mutate_(op, s);
    deq_outer_loops_.pop_front();
    return stmt;
  }

  Expr Mutate_(const Div *op, const Expr &e) final {
    int count_{0};
    div_ = nullptr;
    if (!in_provide_) return e;
    auto CheckVar = [&count_](const NodeRef &op) {
      const auto v = op.as<Variable>();
      if (v != nullptr) {
        count_++;
      }
    };
    PostOrderVisit(e, CheckVar);
    if (count_ > 0) div_ = op;
    return e;
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    int count_{0};
    mod_ = nullptr;
    if (!in_provide_) return e;
    auto CheckVar = [&count_](const NodeRef &op) {
      const auto v = op.as<Variable>();
      if (v != nullptr) {
        count_++;
      }
    };
    PostOrderVisit(e, CheckVar);
    if (count_ > 0) mod_ = op;
    return e;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Array<Expr> args;
    // save current outer loops
    Map<Var, Range> range;
    for (int i = static_cast<int>(deq_outer_loops_.size()) - 1; i >= 0; i--) {
      const For *f = deq_outer_loops_[i];
      range.Set(Var(f->loop_var), Range::make_by_min_extent(f->min, f->extent));
    }
    size_t sz = op->args.size();
    bool need_let{false};
    in_provide_ = true;
    // only search for one pattern right now
    div_ = nullptr;
    mod_ = nullptr;
    for (size_t i = 1; i < sz; i++) {
      static_cast<void>(this->Mutate(op->args[i - 1]));
      static_cast<void>(this->Mutate(op->args[i]));
      if (div_ != nullptr && mod_ != nullptr && Equal(div_->a, mod_->a) && Equal(div_->b, mod_->b)) {
        // new a let, insert before this provide
        need_let = true;
        break;
      }
    }
    in_provide_ = false;

    if (need_let) {
      Var var("tmp");
      Stmt new_s(s);
      new_s = substitute(div_->a, var, new_s);
      new_s = CanonicalSimplify(new_s, range);
      Stmt stmt = LetStmt::make(var, div_->a, new_s);
      return stmt;
    }
    for (auto i : op->args) {
      Expr pre = CanonicalSimplify(i, range);
      Expr expr = CanonicalSimplify(pre, range);
      while (!expr.same_as(pre)) {
        pre = expr;
        expr = CanonicalSimplify(pre, range);
      }
      args.push_back(expr);
    }
    Expr val = this->Mutate(op->value);
    return Provide::make(op->func, op->value_index, val, args);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args;
    // save current outer loops
    Map<Var, Range> range;
    for (int i = static_cast<int>(deq_outer_loops_.size()) - 1; i >= 0; i--) {
      const For *f = deq_outer_loops_[i];
      range.Set(Var(f->loop_var), Range::make_by_min_extent(f->min, f->extent));
    }
    for (auto i : op->args) {
      Expr pre = CanonicalSimplify(i, range);
      Expr expr = CanonicalSimplify(pre, range);
      while (!expr.same_as(pre)) {
        pre = expr;
        expr = CanonicalSimplify(pre, range);
      }
      args.push_back(expr);
    }
    return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
  }

 private:
  // outer loops for codegen
  std::deque<const For *> deq_outer_loops_;
  // status for provide / Call
  const Div *div_{nullptr};
  const Mod *mod_{nullptr};
  // status of provide
  bool in_provide_{false};
};

class FindAttrs : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_attrs") {
      attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    } else if (op->attr_key == "pragma_im2col") {
      find_ = true;
    }
    IRVisitor::Visit_(op);
  }

  Map<std::string, NodeRef> attrs;
  bool find_{false};
};

/*
  realize data_trans_hybrid_local_UB([0, 1], [0, 4], [0, 4], [0, 224], [0, 16]) {
    for (cc4, 0, 4) {
      for (cc5, 0, 7) {
        for (cc6, 0, 224) {
          for (cc7, 0, 16) {
            if (((((cc5 + 3) % 2) == 0) && ((cc6 % 2) == 0))) {
              data_trans_hybrid_local_UB(0, cc4, (cc5 + 3), cc6, cc7) =input0_local_UB(0, cc4, ((cc5 + 3)/2), (cc6/2),
  cc7)
            }
          }
        }
      }
    }
    for (cc4, 0, 4) {
      for (cc5, 0, 7) {
        for (cc6, 0, 224) {
          for (cc7, 0, 16) {
            if (!((((cc5 + 3) % 2) == 0) && ((cc6 % 2) == 0))) {
              data_trans_hybrid_local_UB(0, cc4, (cc5 + 3), cc6, cc7) =0.000000h
            }
          }
        }
      }
    }
  }
  ---->
  realize data_trans_hybrid_local_UB([0, 1], [0, 4], [0, 10], [0, 224], [0, 16]) {
    for (cc4, 0, 4) {
      for (cc5, 0, 7) {
        for (cc6, 0, 224) {
          for (cc7, 0, 16) {
            if (((((cc5 + 3) % 2) == 0) && ((cc6 % 2) == 0))) {
              data_trans_hybrid_local_UB(0, cc4, (cc5 + 3), cc6, cc7) =input0_local_UB(0, cc4, ((cc5 + 3)/2), (cc6/2),
  cc7)
            }
          }
        }
      }
    }
    for (cc4, 0, 4) {
      for (cc5, 0, 7) {
        for (cc6, 0, 224) {
          for (cc7, 0, 16) {
            if (!((((cc5 + 3) % 2) == 0) && ((cc6 % 2) == 0))) {
              data_trans_hybrid_local_UB(0, cc4, (cc5 + 3), cc6, cc7) =0.000000h
            }
          }
        }
      }
    }
  }

 */
class FixRealizeMultiDef : public IRMutator {
 public:
  FixRealizeMultiDef() {}
  ~FixRealizeMultiDef() override = default;

  Stmt Run(Stmt stmt) {
    FindAttrs collector;
    collector.Visit(stmt);

    if (collector.find_) {
      attrs_ = collector.attrs;
      stmt = this->Mutate(stmt);
    }

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    const auto var = op->loop_var.as<Variable>();

    regionMap_.emplace(std::pair<const Variable *, Range>{var, Range::make_by_min_extent(op->min, op->extent)});
    Stmt stmt = IRMutator::Mutate_(op, s);
    regionMap_.erase(var);

    return stmt;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    std::string name = op->func->func_name();

    realizeMap_.erase(name);
    defMap_.erase(name);
    boundsMap_.erase(name);
    realizeMap_.emplace(std::pair<std::string, bool>{name, true});
    defMap_.emplace(std::pair<std::string, int>{name, 0});
    Stmt stmt = IRMutator::Mutate_(op, s);
    realizeMap_.erase(name);

    if (defMap_[name] > 1) {
      Region bounds;

      CHECK_GT(boundsMap_.count(name), 0);
      std::transform(
        boundsMap_[name].begin(), boundsMap_[name].end(), std::back_inserter(bounds.CopyOnWrite()->data),
        [](const Range &range) { return (Range::make_by_min_extent(Expr(0), range->min + range->extent)); });

      return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, op->body);
    } else {
      return stmt;
    }
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    std::string name = op->func->func_name();
    if (realizeMap_.count(name) > 0) {
      CHECK_EQ(realizeMap_[name], true);
      CHECK_GT(defMap_.count(name), 0);

      int defCount_ = defMap_[name];

      Region new_bounds;
      for (size_t i = 0; i < op->args.size(); i++) {
        auto bound = InferSimpleExprRange(op->args[i], &regionMap_);
        if (defCount_ > 0) {
          CHECK_GT(boundsMap_.count(name), 0);

          auto new_min = Simplify(min(bound->min, boundsMap_[name][i]->min));
          auto new_max =
            Simplify(max(bound->min + bound->extent, boundsMap_[name][i]->min + boundsMap_[name][i]->extent));
          auto new_extent = Simplify(new_max - new_min);

          new_bounds.push_back(Range::make_by_min_extent(new_min, new_extent));
        } else {
          CHECK_EQ(defCount_, 0);
          new_bounds.push_back(bound);
        }
      }

      defMap_[name] = defCount_ + 1;

      boundsMap_[name] = new_bounds;
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<const Variable *, Range> regionMap_;
  Map<std::string, NodeRef> attrs_;
  std::unordered_map<std::string, bool> realizeMap_;
  std::unordered_map<std::string, int> defMap_;
  std::unordered_map<std::string, Region> boundsMap_;
};

/* fix realize shape when realize greater than loop extent,
/ attr [placeholder(A_local_L1, 0xfcad80)] realize_scope = "local.L1"
realize A_local_L1([0, 1], [0, 2], [0, 6], [0, 12], [0, 16]) {
  produce A_local_L1 {
    // attr [0] pragma_emit_insn = "dma_copy"
    for (cc2, 0, 2) {
      for (cc3, 0, 5) {
        for (cc4, 0, 10) {
          for (cc5, 0, 16) {
            A_local_L1(0, cc2, cc3, cc4, cc5) =A(0, cc2, cc3, cc4, cc5)
          }
        }
      }
    }
  }
-->
realize A_local_L1([0, 1], [0, 2], [0, 5], [0, 10], [0, 16])
  produce A_local_L1 {
  .......
  }
*/
class FixRealize : public IRMutator {
 public:
  FixRealize() {}
  ~FixRealize() override = default;

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    realize_.insert(op->func.get());
    Stmt stmt = this->Mutate(op->body);
    Array<Range> newr;

    if (realize_args_.count(op->func) && loop_ext_.count(op->func.get())) {
      Array<Expr> pargs_ = realize_args_[op->func];
      // do something, make a new realize with new range
      CHECK_EQ(pargs_.size(), op->bounds.size()) << " not match ";
      for (size_t i = 0; i < pargs_.size(); i++) {
        Expr var = pargs_[i];
        if (var.as<IntImm>()) {
          newr.push_back(op->bounds[i]);
        } else {
          // find related loop extent
          Expr extent = op->bounds[i]->extent;
          for (auto j : loop_ext_[op->func.get()]) {
            if (j->loop_var.get() == var.get() && (is_dynamic_ || j->extent.as<IntImm>())) {
              extent = j->extent;
            }
          }
          Expr cmp = Simplify(op->bounds[i]->extent - extent);
          if (!is_zero(cmp)) {
            newr.push_back(Range::make_by_min_extent(op->bounds[i]->min, extent));
          } else {
            newr.push_back(op->bounds[i]);
          }
        }
      }
    } else {
      newr = op->bounds;
    }

    return Realize::make(op->func, op->value_index, op->type, newr, op->condition, stmt);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_fractal" || op->attr_key == "pragma_filter" || op->attr_key == "pragma_im2col" ||
        op->attr_key == "pragma_ub_gm") {
      curnt_loops_.clear();
      found_ = false;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt stmt;
    if (op->for_type == ForType::Vectorized && !in_vectorized_) {
      curnt_loops_.clear();
      found_ = false;
      curnt_loops_.push_back(op);
      in_vectorized_ = true;
      stmt = IRMutator::Mutate_(op, s);
      in_vectorized_ = false;
    } else {
      if (!found_) curnt_loops_.push_back(op);
      stmt = IRMutator::Mutate_(op, s);
    }
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (realize_.count(op->func.get()) && !found_) {
      found_ = true;
      Array<Expr> empty;
      // we don't touch multi-def for the same provide
      if (realize_args_.count(op->func)) {
        realize_args_[op->func] = empty;
        loop_ext_.erase(op->func.get());
      } else {
        realize_args_[op->func] = op->args;
        loop_ext_[op->func.get()] = curnt_loops_;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::vector<const For *> curnt_loops_;
  std::unordered_set<const Node *> realize_;
  std::unordered_map<const Node *, std::vector<const For *>> loop_ext_;
  std::unordered_map<FunctionRef, Array<Expr>, air::NodeHash, air::NodeEqual> realize_args_;
  bool is_dynamic_ = global_attrs.GetBoolAttr(kIsDynamic, false);
  bool found_{false};
  bool in_vectorized_{false};
};

/* for loops with min != 1, we normalize to 0, only fix src op
    // attr [0] pragma_emit_insn = "dma_copy"
    for (cc2, 0, 2) {
      for (cc3, 1, 6) {
        for (cc4, 1, 11) {
          for (cc5, 0, 16) {
            A_local_L1(0, cc2, cc3, cc4, cc5) =A(0, cc2, (cc3 - 1), (cc4 - 1), cc5)
          }
        }
      }
    }
-->
  // attr [0] pragma_emit_insn = "dma_copy"
    for (cc2, 0, 2) {
      for (cc3, 0, 5) {
        for (cc4, 0, 10) {
          for (cc5, 0, 16) {
            A_local_L1(0, cc2, cc3, cc4, cc5) =A(0, cc2, cc3, cc4, cc5)
          }
        }
      }
    }
*/

class IsConv : public IRMutator {
 public:
  IsConv() {}
  ~IsConv() override = default;

  bool Check(const Stmt stmt) {
    static_cast<void>(this->Mutate(stmt));
    return is_conv_;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() && op->value.as<StringImm>()->value == "mad") {
      is_conv_ = true;
    } else if (op->attr_key == "pragma_load3d") {
      is_conv_ = true;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool is_conv_{false};
};

class Normlize : public IRMutator {
 public:
  Normlize() {}
  ~Normlize() override = default;

  Stmt Run(Stmt stmt) {
    is_conv_ = IsConv().Check(stmt);
    stmt = this->Mutate(stmt);
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (op->for_type == ForType::Vectorized && !in_vectorized_) {
      Stmt stmt;
      in_vectorized_ = true;
      smap.clear();
      if (!is_zero(op->min)) {
        smap[op->loop_var.get()] = op->min;
        in_fix_ = true;
        static_cast<void>(this->Mutate(op->extent));
        in_fix_ = false;
        Stmt body = this->Mutate(op->body);
        stmt = For::make(op->loop_var, make_const(Int(32), 0), op->extent, op->for_type, op->device_api, body);
      } else {
        stmt = IRMutator::Mutate_(op, s);
      }
      in_vectorized_ = false;
      return stmt;
    } else if (in_vectorized_ && !is_zero(op->min)) {
      smap[op->loop_var.get()] = op->min;
      in_fix_ = true;
      Expr extent = this->Mutate(op->extent);
      in_fix_ = false;
      Stmt body = this->Mutate(op->body);
      return For::make(op->loop_var, make_const(Int(32), 0), extent, op->for_type, op->device_api, body);
    } else if (!(op->extent.as<IntImm>())) {
      // normlize loop var in extent
      in_fix_ = true;
      Expr extent = this->Mutate(op->extent);
      in_fix_ = false;
      Stmt body = this->Mutate(op->body);
      return For::make(op->loop_var, op->min, extent, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (in_vectorized_) {
      in_fix_ = true;
      // fix the if condition
      auto cond = this->Mutate(op->condition);
      in_fix_ = false;
      auto then_case = this->Mutate(op->then_case);
      auto else_case = op->else_case;
      if (op->else_case.defined()) {
        else_case = this->Mutate(op->else_case);
      }
      return IfThenElse::make(cond, then_case, else_case);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    Array<Expr> args;

    if (in_vectorized_) {
      in_fix_ = true;
      // fix the src
      Expr body = this->Mutate(op->value);
      in_fix_ = false;

      // just dont fix conv local_l1's dst
      size_t pos = op->func->func_name().find("local_L1");
      if (!(is_conv_ && pos != std::string::npos)) {
        in_fix_ = true;
        // fix the dst when it's not a conv/gemm ops
        for (auto i : op->args) {
          Expr ret = this->Mutate(i);
          args.push_back(ret);
        }
        in_fix_ = false;
      } else {
        args = op->args;
      }
      return Provide::make(op->func, op->value_index, body, args);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    // replace src index here
    if (in_vectorized_ && in_fix_ && smap.count(op)) {
      return Simplify(e + smap[op]);
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_map<const Variable *, Expr> smap;
  bool in_vectorized_{false};
  bool in_fix_{false};
  bool is_conv_{false};
};

// for cfg loops not in emit stage
class CfgNormlize : public IRMutator {
 public:
  CfgNormlize() {}
  ~CfgNormlize() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!is_zero(op->min)) {
      // fix loop range
      smap[op->loop_var.get()] = op->min;
      Stmt body = this->Mutate(op->body);
      smap.erase(op->loop_var.get());
      return For::make(op->loop_var, make_const(Int(32), 0), op->extent, op->for_type, op->device_api, body);
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    // replace src index here
    if (smap.count(op)) {
      return e + smap[op];
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_map<const Variable *, Expr> smap;
};

class FixL0CRealize : public IRMutator {
 public:
  /* This class use to fix ***_local_UB_local_L0C realize size same with ***_local_UB realize size
   *
   * // attr [0] alloc_C = 1
     // attr [placeholder(result_local_UB, 0x2df5e70)] realize_scope = "local.UB"
     realize result_local_UB([0, 1], [0, 1], [0, 1], [0, 16], [0, 16]) {
      // attr [placeholder(result_local_UB_local_L0C, 0x22b6fd0)] realize_scope = "local.L0C"
      realize result_local_UB_local_L0C([0, 1], [0, 3], [0, 3], [0, 16], [0, 16]) {
   *
       to
   *
   * // attr [0] alloc_C = 1
     // attr [placeholder(result_local_UB, 0x2df5e70)] realize_scope = "local.UB"
     realize result_local_UB([0, 1], [0, 1], [0, 1], [0, 16], [0, 16]) {
      // attr [placeholder(result_local_UB_local_L0C, 0x22b6fd0)] realize_scope = "local.L0C"
      realize result_local_UB_local_L0C([0, 1], [0, 1], [0, 1], [0, 16], [0, 16]) {
   *
   * */
  FixL0CRealize() {}
  ~FixL0CRealize() override = default;

  Stmt Run(Stmt stmt) {
    if (IsConv().Check(stmt)) {
      stmt = this->Mutate(stmt);
    }
    return stmt;
  }

  bool isEndsWith(const std::string &tensor, const std::string &suffix) {
    if (tensor.size() < suffix.size()) {
      return false;
    }
    std::string compare = tensor.substr(tensor.size() - suffix.size());
    return (compare == suffix);
  }

  bool needFixRealize(const std::string &ubOutRealize, const std::string &l0CRealize) {
    std::string ubEnd = "local_UB";
    std::string l0CEnd = "local_UB_local_L0C";
    if (!isEndsWith(ubOutRealize, ubEnd)) {
      return false;
    }

    if (!isEndsWith(l0CRealize, l0CEnd)) {
      return false;
    }

    std::string prefixOut = ubOutRealize.substr(0, ubOutRealize.size() - ubEnd.size());
    std::string prefixL0C = l0CRealize.substr(0, l0CRealize.size() - l0CEnd.size());
    return (prefixOut == prefixL0C);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (!needFix_) {
      return s;
    }

    if (isEndsWith(op->func->func_name(), "local_UB")) {
      realizeQue_.push_front(op);
      Stmt res = IRMutator::Mutate_(op, s);
      realizeQue_.pop_front();
      return res;
    }

    if (isEndsWith(op->func->func_name(), "local_UB_local_L0C") && !realizeQue_.empty()) {
      const Realize *ubOutRealize = realizeQue_.front();
      Array<Range> bounds = op->bounds;

      if (needFixRealize(ubOutRealize->func->func_name(), op->func->func_name())) {
        bounds = ubOutRealize->bounds;
      }

      return Realize::make(op->func, op->value_index, op->type, bounds, op->condition, op->body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "alloc_C") {
      needFix_ = true;
      Stmt res = IRMutator::Mutate_(op, s);
      needFix_ = false;
      return res;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::deque<const Realize *> realizeQue_;
  bool needFix_{false};
};

Stmt Normlize_(Stmt stmt) {
  stmt = Normlize().Run(stmt);
  stmt = CfgNormlize().Mutate(stmt);
  return stmt;
}

Stmt LoopNormlize(Stmt stmt) {
  stmt = VectorizeFor().Mutate(stmt);
  stmt = Normlize_(stmt);
  stmt = FixRealize().Mutate(stmt);
  stmt = FixL0CRealize().Run(stmt);
  stmt = FixRealizeMultiDef().Run(stmt);
  stmt = RecoverFor().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
