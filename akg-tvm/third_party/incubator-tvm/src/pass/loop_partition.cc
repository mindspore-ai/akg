/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file loop_partition.cc
 */

/*
 * 2019.12.30 - Add new functions for loop_partition operation,
 *              such as Visit_ and Mutate_ and others, some code refactors.
 * 2021.11.3  - For the multi-core loop, keep the For whose extent is 1 to facilitate 
 *              the processing of subsequent the multi-core loop merging.
 */

#include <ir_pass.h>
#include <tvm.h>
#include <arithmetic/const_fold.h>
#include <arithmetic/int_set.h>
#include <pass/ir_util.h>
#include <runtime/thread_storage_scope.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include <unordered_map>
#include <unordered_set>

#include "pass/utils.h"
#include "pass/zero_elimination.h"

namespace air {
namespace ir {
#define DEBUG_LOOP_PARTITION 0

using akg::ir::CanProve;
using akg::ir::GetObjPtr;
using akg::ir::Simplify_cce;
using akg::ir::SuperSimplify;
using arith::DeduceBound;
using arith::Intersect;
using arith::IntSet;
using akg::ir::GatherVars;
using PartitionKey = std::pair<const Node*, Expr>;

typedef struct {
  bool operator()(const PartitionKey& lhs, const PartitionKey& rhs) const noexcept {
    return ((lhs.first == rhs.first) && (lhs.second.get() == rhs.second.get()));
  }
} PartitionKeyEqual;

typedef struct {
  std::size_t operator()(const PartitionKey& k) const noexcept {
    std::size_t h1 = std::hash<const Node*>{}(k.first);
    std::size_t h2 = ObjectHash()(GetObjPtr(k.second.get()));
    return h1 ^ h2;
  }
} PartitionKeyHash;

// Each mapping (cond, Expr) -> interval represents the fact that
// that op is proven to have value represented by the Expr in that interval
using Partition = std::unordered_map<PartitionKey, IntSet, PartitionKeyHash, PartitionKeyEqual>;

enum PartitionType { kRange, kEqual, kMaxMin };

bool ExprUseVars(const Expr& expr, const std::unordered_set<const Variable*>& vars) {
  bool success = false;
  PostOrderVisit(expr, [&vars, &success](const NodeRef& node) {
    if (const auto v = node.as<Variable>()) {
      if (vars.count(v)) {
        success = true;
        return;
      }
    }
  });
  return success;
}

// Select potential candidate IRs that can be partitioned.
// Rule:
//   - the range should not be const
//   - there exist an expression in the scope that use the var
class CandidateSelector final : public IRVisitor {
 public:
  using VarIsUsed = bool;
  explicit CandidateSelector(bool split_const_loop) : split_const_loop_(split_const_loop) {}
  ~CandidateSelector() override = default;

  void Visit_(const For* op) override {
    // partition const loop when sets split_const_loop_
    if (!is_const(op->min) || !is_const(op->extent) || split_const_loop_) {
      const Variable* var = op->loop_var.get();
      record_.emplace(var, false);
      IRVisitor::Visit_(op);
      if (record_.at(var) && !no_split_) {
        candidates.insert(op);
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "candidate: " << var->name_hint << std::endl;
        }
      }
      record_.erase(var);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const AttrStmt* op) override {
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
      if ((scope.rank == 0) && (!is_const(op->value) || split_const_loop_)) {
        record_.emplace(var.get(), false);
        IRVisitor::Visit_(op);
        if (record_.at(var.get()) && !no_split_) {
          candidates.insert(op);
        }
        record_.erase(var.get());
        return;
      }
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Block* op) override {
    bool temp = no_split_;
    this->Visit(op->first);
    // erase the no split state of first when visit rest.
    std::swap(temp, no_split_);
    this->Visit(op->rest);
    // restore the no split flag.
    no_split_ = no_split_ || temp;
  }

  void Visit_(const Call* op) override {
    if (op->is_intrinsic(Call::likely) || op->is_intrinsic("tvm_if_then_else")) {
      in_partitionable_op_ = true;
      IRVisitor::Visit(op->args[0]);
      in_partitionable_op_ = false;
      for (size_t i = 1; i < op->args.size(); i++) {
        IRVisitor::Visit(op->args[i]);
      }
    } else if (op->is_intrinsic(intrinsic::tvm_thread_allreduce)) {
      // no split if the body contains allreduce.
      no_split_ = true;
      return;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Select* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit(op->condition);
    in_partitionable_op_ = false;
    IRVisitor::Visit(op->true_value);
    IRVisitor::Visit(op->false_value);
  }

  void Visit_(const Mod* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const FloorMod* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const Div* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const FloorDiv* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const Max* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const Min* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit_(op);
    in_partitionable_op_ = false;
  }

  void Visit_(const Variable* op) override {
    if (in_partitionable_op_ && record_.count(op)) {
      record_.at(op) = true;
    }
  }

  void Visit_(const IfThenElse* op) override {
    in_partitionable_op_ = true;
    IRVisitor::Visit(op->condition);
    in_partitionable_op_ = false;
    IRVisitor::Visit(op->then_case);
    IRVisitor::Visit(op->else_case);
  }

  std::unordered_set<const Node*> candidates;

 private:
  bool in_partitionable_op_{false};
  bool no_split_{false};
  bool split_const_loop_;
  std::unordered_map<const Variable*, VarIsUsed> record_;
};

// Populate partitions data structure, i.e., for a specific variable,
// find an interval in which each condition has fixed true or false value
class PartitionFinder : public IRVisitor {
 public:
  explicit PartitionFinder(const VarExpr current_var, const Expr min, const Expr max,
                           bool remove_div_mod,
                           const std::unordered_map<const Variable*, IntSet>& hint_map,
                           const std::unordered_map<const Variable*, IntSet>& relax_map)
      : current_var_(current_var),
        min_(min),
        max_(max),
        remove_div_mod_(remove_div_mod),
        hint_map_(hint_map),
        relax_map_(relax_map) {
    for (const auto& kv : hint_map) {
      out_vars_.insert(kv.first);
    }
    for (const auto& kv : relax_map) {
      out_vars_.insert(kv.first);
    }
  }
  ~PartitionFinder() override = default;

  Partition partitions_;  // holds partition map for GE, LE, GT, LT, EQ, Max and Min ops
  PartitionType partition_type_{PartitionType::kRange};
  bool is_max_min_cond_{false};
  Node* max_min_op_{nullptr};
  Expr true_max_min_;
  Expr false_max_min_;
  std::unordered_map<const Node*, Expr> div_modulo_;  // holds partition map for Div and Mod ops

 private:
  void Visit_(const For* op) override {
    const Variable* var = op->loop_var.get();
    hint_map_.emplace(std::pair<const Variable*, IntSet>(
        var, IntSet::interval(op->min, op->min + op->extent - 1)));
    relax_map_.insert({var, IntSet::interval(op->min, op->min + op->extent - 1)});
    IRVisitor::Visit_(op);
    relax_map_.erase(var);
    hint_map_.erase(var);
  }

  void Visit_(const AttrStmt* op) override {
    // handle thread_axis
    if (op->attr_key == air::ir::attr::thread_extent) {
      const auto thread_axis = op->node.as<IterVarNode>();
      CHECK(thread_axis);
      const Variable* var = thread_axis->var.get();
      IntSet dom = IntSet::range(Range(make_zero(op->value.type()), op->value));
      hint_map_.emplace(std::pair<const Variable*, IntSet>(var, dom));
      relax_map_.emplace(std::pair<const Variable*, IntSet>(var, dom));
      IRVisitor::Visit_(op);
      relax_map_.erase(var);
      hint_map_.erase(var);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Call* op) override {
    if (op->is_intrinsic(Call::likely) || op->is_intrinsic("tvm_if_then_else")) {
      Expr cond = op->args[0];
      ExtractPartition(cond);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Max* op) override {
    Expr cond = GE::make(op->a, op->b);
    if (partitions_.empty()) {
      is_max_min_cond_ = true;
      max_min_op_ = static_cast<Node*>(const_cast<Max*>(op));
      true_max_min_ = op->a;
      false_max_min_ = op->b;
      ExtractPartition(cond);
      is_max_min_cond_ = false;
    }
    if (DEBUG_LOOP_PARTITION) {
      LOG(INFO) << "Looking at Max Expr: Max(" << op->a << ", " << op->b << ")" << std::endl;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Min* op) override {
    Expr cond = LE::make(op->a, op->b);
    if (partitions_.empty()) {
      is_max_min_cond_ = true;
      max_min_op_ = static_cast<Node*>(const_cast<Min*>(op));
      true_max_min_ = op->a;
      false_max_min_ = op->b;
      ExtractPartition(cond);
      is_max_min_cond_ = false;
    }
    if (DEBUG_LOOP_PARTITION) {
      LOG(INFO) << "Looking at Min Expr: Min(" << op->a << ", " << op->b << ")" << std::endl;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Mod* op) override {
    Expr e = GetRef<Expr>(op);  // make the modulo expression
    // if the Mod op uses current_var_ and is a partition-able Mod op, use it for partition.
    if (remove_div_mod_ &&
        ExprUseVars(e, std::unordered_set<const Variable*>({current_var_.get()})) &&
        CheckForValidDivMod<Mod>(e)) {
      div_modulo_[op] = op->b;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const FloorMod* op) override {
    Expr e = GetRef<Expr>(op);  // make the modulo expression
    // if the Mod op uses current_var_ and is a partition-able Mod op, use it for partition.
    if (remove_div_mod_ &&
        ExprUseVars(e, std::unordered_set<const Variable*>({current_var_.get()})) &&
        CheckForValidDivMod<FloorMod>(e)) {
      div_modulo_[op] = op->b;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Div* op) override {
    Expr e = GetRef<Expr>(op);
    // if the Div op uses current_var_ and is a partition-able Div op, use it for partition.
    if (remove_div_mod_ &&
        ExprUseVars(e, std::unordered_set<const Variable*>({current_var_.get()})) &&
        CheckForValidDivMod<Div>(e)) {
      div_modulo_[op] = op->b;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const FloorDiv* op) override {
    Expr e = GetRef<Expr>(op);
    // if the Div op uses current_var_ and is a partition-able Div op, use it for partition.
    if (remove_div_mod_ &&
        ExprUseVars(e, std::unordered_set<const Variable*>({current_var_.get()})) &&
        CheckForValidDivMod<FloorDiv>(e)) {
      div_modulo_[op] = op->b;
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const IfThenElse* op) override {
    Expr cond = op->condition;
    ExtractPartition(cond);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Select* op) override {
    Expr cond = op->condition;
    ExtractPartition(cond);
    IRVisitor::Visit_(op);
  }

  Expr InverseCond(const Expr& cond) {
    Expr inverse_cond;
    if (const LT* op = cond.as<LT>()) {
      // a < b -> a >= b
      inverse_cond = GE::make(op->a, op->b);
    } else if (const GT* op = cond.as<GT>()) {
      // a > b -> a <= b
      inverse_cond = LE::make(op->a, op->b);
    } else if (const LE* op = cond.as<LE>()) {
      // a <= b -> a > b
      inverse_cond = GT::make(op->a, op->b);
    } else if (const GE* op = cond.as<GE>()) {
      // a >= b -> a < b
      inverse_cond = LT::make(op->a, op->b);
    } else if (const EQ* op = cond.as<EQ>()) {
      // a == b -> a != b
      inverse_cond = NE::make(op->a, op->b);
    } else if (const NE* op = cond.as<NE>()) {
      // a != b -> a == b
      inverse_cond = EQ::make(op->a, op->b);
    }
    return inverse_cond;
  }

  /*
   * Numerator for Div or Mod for the partition must be a linear equation in current_var_ only and
   * Denominator must be a IntImm. E.g. of the form (k1 * i + k2) // k3 or (k1 * i + k2) % k3 where
   * k1, k2 and k3 are IntImm. require to: We can allow Mod or Div conditions in multiple IVs as the
   * simplifier is able to simplify them Add check that we are not dealing with Any tensor data
   * inside the condition e.g. (i + B[j]) % 2
   * */
  template <typename T>
  bool CheckForValidDivMod(const Expr& e) {
    if (!e.as<T>()) {
      return false;
    }
    CHECK(e.as<T>());
    Array<Expr> linearEquationCoeff =
        air::arith::DetectLinearEquation(e.as<T>()->a, {Var(current_var_)});
    // DetectLinearEquation returns an empty container if `cond` has some function call to
    // `current_var_` e.g. (a + Max(a, 1))
    if (linearEquationCoeff.empty() || !e.as<T>()->b.template as<IntImm>()) {
      return false;
    } else if (std::any_of(linearEquationCoeff.begin(), linearEquationCoeff.end(),
                           [](const Expr& i) { return (!i.as<IntImm>()); })) {
      return false;
    }
    return true;
  }

  /* Extract partition for Div/Mod first checks for the validity of the the condition.
   *
   * Currently, loop partition only supports conditions of the following form
   * (k1 * i + k2) % k3 `op` k4   OR (k1 * i + k2) // k3   OR in reverse order,
   *  k4 `op` (k1 * i + k2) % k3  OR  k4 `op` (k1 * i + k2) // k3
   *
   * where  the term (k1 * i + k2) can be any valid linear equation in variable i which covers case
   * of k1=1 and/or k2=0 and where k1, k2, k3, and k4 are strictly IntImm. `op` can be any CmpOpNode
   * either EQ, NE, GE, GT, LE or LT.
   *
   * If we have valid modulo or div ops then we record the Node and
   * divisor number k3 for the later use during loop partition.
   *
   * By default, any modulo or div conditions inside an ExternOp or HybridOp are not considered for
   * partitioning. The reason is that partitioning based on those conditions changes the iteration
   * order of the iteration variable, and in some cases if there is any dependency on that variable,
   * the partitioning will be illegal to perform. Therefore, we will be conservative and avoid
   * applying such partitioning when there is a potential for that dependency to exist, i.e., when
   * we have ExternOp or HybridOp
   */
  template <typename T1>
  void ExtractValidDivModInCond(const Expr& cond) {
    Expr div_mod_cond;
    if (const T1* op = cond.as<T1>()) {
      if (op->b.template as<IntImm>()) {
        div_mod_cond = op->a;
      } else if (op->a.template as<IntImm>()) {
        div_mod_cond = op->b;
      }
      if (div_mod_cond.defined()) {
        if (CheckForValidDivMod<Div>(div_mod_cond)) {
          CHECK(div_mod_cond.as<Div>());
          div_modulo_[cond.get()] = div_mod_cond.as<Div>()->b;
        } else if (CheckForValidDivMod<Mod>(div_mod_cond)) {
          CHECK(div_mod_cond.as<Mod>());
          div_modulo_[cond.get()] = div_mod_cond.as<Mod>()->b;
        } else if (CheckForValidDivMod<FloorDiv>(div_mod_cond)) {
          CHECK(div_mod_cond.as<FloorDiv>());
          div_modulo_[cond.get()] = div_mod_cond.as<FloorDiv>()->b;
        } else if (CheckForValidDivMod<FloorMod>(div_mod_cond)) {
          CHECK(div_mod_cond.as<FloorMod>());
          div_modulo_[cond.get()] = div_mod_cond.as<FloorMod>()->b;
        }
      }
    }
  }

  void ExtractPartitionDivMod(const Expr& cond) {
    if (cond.as<EQ>()) {
      ExtractValidDivModInCond<EQ>(cond);
    } else if (cond.as<NE>()) {
      ExtractValidDivModInCond<NE>(cond);
    } else if (cond.as<GT>()) {
      ExtractValidDivModInCond<GT>(cond);
    } else if (cond.as<GE>()) {
      ExtractValidDivModInCond<GE>(cond);
    } else if (cond.as<LT>()) {
      ExtractValidDivModInCond<LT>(cond);
    } else if (cond.as<LE>()) {
      ExtractValidDivModInCond<LE>(cond);
    }
    // no partition-able div-modulo condition
  }

  void ExtractPartitionRangeAndEqualEqual(const Expr& cond) {
    // For cond, find out the interval, if exists, in which we can prove that cond is
    // true. Also find the interval, if exists, in which we can prove that cond is
    // false.
    arith::Analyzer analyzer_;
    IntSet interval = DeduceBound(current_var_, cond, hint_map_, relax_map_);
    if (!interval.is_nothing() &&
        !ExprUseVars(interval.min(), std::unordered_set<const Variable*>({current_var_.get()})) &&
        !ExprUseVars(interval.max(), std::unordered_set<const Variable*>({current_var_.get()})) &&
        !arith::Intersect(&analyzer_, arith::IntervalSet(interval.min(), interval.max()),
                          arith::IntervalSet(min_, max_))
             ->IsEmpty()) {
      // cond is true within interval
      if (is_max_min_cond_) {
        partition_type_ = kMaxMin;
        partitions_[{max_min_op_, true_max_min_}] = interval;
      } else if (cond.as<EQ>()) {
        partition_type_ = kEqual;
        partitions_[{cond.get(), const_true()}] = interval;
      } else {
        partition_type_ = kRange;
        partitions_[{cond.get(), const_true()}] = interval;
      }
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "condition: " << cond << " is true for interval: [" << interval.min() << ", "
                  << interval.max() << "]" << std::endl;
      }
    }
    Expr inverse_cond = InverseCond(cond);
    if (inverse_cond.defined()) {
      IntSet interval_ = DeduceBound(current_var_, inverse_cond, hint_map_, relax_map_);
      if (!interval_.is_nothing() &&
          !ExprUseVars(interval_.min(),
                       std::unordered_set<const Variable*>({current_var_.get()})) &&
          !ExprUseVars(interval_.max(),
                       std::unordered_set<const Variable*>({current_var_.get()})) &&
          !arith::Intersect(&analyzer_, arith::IntervalSet(interval_.min(), interval_.max()),
                            arith::IntervalSet(min_, max_))
               ->IsEmpty()) {
        // cond is true within interval
        if (is_max_min_cond_) {
          partition_type_ = kMaxMin;
          partitions_[{max_min_op_, false_max_min_}] = interval_;
        } else if (inverse_cond.as<EQ>()) {
          partition_type_ = kEqual;
          partitions_[{cond.get(), const_false()}] = interval_;
        } else {
          partition_type_ = kRange;
          partitions_[{cond.get(), const_false()}] = interval_;
        }
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "condition: " << cond << " is false for interval: [" << interval_.min()
                    << ", " << interval_.max() << "]" << std::endl;
        }
      }
    }
  }

  void ExtractPartition(const Expr& cond) {
    if (ExprUseVars(cond, std::unordered_set<const Variable*>({current_var_.get()}))) {
      if (const And* op = cond.as<And>()) {
        ExtractPartition(op->a);
        ExtractPartition(op->b);
      } else if (const Or* op = cond.as<Or>()) {
        ExtractPartition(op->a);
        ExtractPartition(op->b);
      } else if (const Not* op = cond.as<Not>()) {
        ExtractPartition(op->a);
      } else {
        ExtractPartitionDivMod(cond);
        if (div_modulo_.empty() && partitions_.empty()) {
          ExtractPartitionRangeAndEqualEqual(cond);
        }
      }
    }
  }

  VarExpr current_var_;
  Expr min_, max_;
  bool remove_div_mod_;
  std::unordered_set<const Variable*> out_vars_;
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
};

// Replace the set of conditions given by ps with cond_value (true or false)
class ConditionEliminator : public IRMutator {
 public:
  explicit ConditionEliminator(const std::unordered_set<const Node*>& ps,
                               const Expr cond_value = const_true())
      : ps_(ps), cond_value_(cond_value) {}
  ~ConditionEliminator() override = default;

  using IRMutator::Mutate;
  Expr Mutate(Expr e) final {
    CHECK(ps_.size() == 1) << "Replacing more than one condition at a time";
    if (Equal(Expr(GetObjPtr(*ps_.begin())), e)) {
      return Mutate(cond_value_);
    }
    return IRMutator::Mutate(e);
  }

 private:
  std::unordered_set<const Node*> ps_;
  Expr cond_value_;
};

// Insert the partition branch at the innermost thread scope
class ThreadPartitionInserter : public IRMutator {
 public:
  explicit ThreadPartitionInserter(const std::unordered_set<const Node*>& ps, const Expr cond)
      : ps_(ps), cond_(cond), innermost_thread_scope_(false) {}
  ~ThreadPartitionInserter() override = default;

  Stmt Mutate_(const AttrStmt* op, const Stmt& s) final {
    if (op->attr_key == air::ir::attr::thread_extent) {
      innermost_thread_scope_ = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      // add branch code inside the innermost thread scope
      if (innermost_thread_scope_) {
        Stmt simplified_body = ConditionEliminator(ps_, const_true()).Mutate(op->body);
        Stmt body = IfThenElse::make(cond_, simplified_body, op->body);
        Expr value = this->Mutate(op->value);
        stmt = AttrStmt::make(op->node, op->attr_key, value, body);
      }
      innermost_thread_scope_ = false;
      return stmt;
    } else {
      return IRMutator::Mutate_(op, s);
    }
  }

 private:
  const std::unordered_set<const Node*>& ps_;
  Expr cond_;
  bool innermost_thread_scope_;
};

Stmt AppendStmts(const Stmt& a, const Stmt& b) {
  if (!a.defined()) {
    return b;
  } else if (!b.defined()) {
    return a;
  } else {
    return Block::make(a, b);
  }
}

// Try to partition range of iteration variables in order to remove (some)
// likely conditions
class LoopPartitioner : public IRMutator {
 public:
  explicit LoopPartitioner(bool split_const_loop, bool remove_div_mod)
      : remove_div_mod_(remove_div_mod), selector(CandidateSelector(split_const_loop)) {}
  ~LoopPartitioner() override = default;

  Stmt VisitAndMutate(Stmt& stmt) {
    stmt = Simplify_cce(stmt);
    stmt = RemoveNoOp(stmt);

    selector.Visit(stmt);
    return Mutate(stmt);
  }

  // One by one unfolds body Block into constituents Stmt and wraps them into AttrStmt or For Stmt
  // Returns a Block of AttrStmt or For stmt by concatinating each such wrapped stmt.
  // For the For loops, this has en effect of NOT having a Block of Stmt as its body. Body will
  // always be a single Stmt
  Stmt UnFoldBlocks(const Stmt& s, const Stmt& body) {
    if (const auto block = body.as<Block>()) {
      Stmt first = UnFoldBlocks(s, block->first);
      Stmt rest;
      if (block->rest.defined()) {
        rest = UnFoldBlocks(s, block->rest);
      }
      return AppendStmts(first, rest);
    } else if (const auto op = s.as<AttrStmt>()) {
      return AttrStmt::make(op->node, op->attr_key, op->value, body);
    } else if (const auto op = s.as<For>()) {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
    return s;
  }

  Stmt Mutate_(const For* op, const Stmt& stmt) override {
    hint_map_.insert(std::pair<const Variable*, IntSet>(
        op->loop_var.get(), IntSet::interval(op->min, op->min + op->extent - 1)));
    if (selector.candidates.count(op)) {
      Stmt s =
          TryPartition(op, stmt, op->loop_var, op->min, op->min + op->extent - 1, op->body, false);
      if (s.defined()) {
        hint_map_.erase(op->loop_var.get());
        return s;
      }
    }
    // normal path when loop partition fails
    // normal loop variable can be put into hint map.
    Stmt res = IRMutator::Mutate_(op, stmt);
    hint_map_.erase(op->loop_var.get());
    return res;
  }

  Stmt Mutate_(const AttrStmt* op, const Stmt& stmt) override {
    if (op->attr_key == "pragma_multi_core_depth" && op->body.as<For>()) {
      multi_core_loop_.insert(op->body.as<For>());
    }

    // check if the stmt is generated by a HybridOp or ExternOp
    if (op->attr_key == air::ir::attr::extern_scope) {
      is_extern_op_ = true;
      Stmt res = IRMutator::Mutate_(op, stmt);
      is_extern_op_ = false;
      if (!res.same_as(stmt)) {
        // if there is any loop partition or loops have been separated in individual nesting scope
        CHECK(res.as<AttrStmt>()) << "Mutate result from a AttrStmt op should be an AttrStmt";
        return UnFoldBlocks(res, res.as<AttrStmt>()->body);
      }
      return res;
    } else if (op->attr_key != air::ir::attr::thread_extent) {
      Stmt res = IRMutator::Mutate_(op, stmt);
      if (!res.same_as(stmt)) {  // if there is any loop partition
        CHECK(res.as<AttrStmt>()) << "Mutate result from a AttrStmt op should be an AttrStmt";
        return UnFoldBlocks(res, res.as<AttrStmt>()->body);
      }
      return res;
    } else {
      const auto iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Var var = iv->var;
      if (selector.candidates.count(op)) {
        Stmt s = TryPartition(op, stmt, var, 0, op->value - 1, op->body, true);
        if (s.defined()) return s;
      }

      // normal path when loop partition fails.
      runtime::ThreadScope scope = runtime::ThreadScope::make(iv->thread_tag);
      Stmt res;
      if (scope.rank == 1) {
        // threadIdx should be put into relax map, in case of divergence.
        relax_map_.insert(std::pair<const Variable*, IntSet>(
            var.get(), IntSet::interval(make_zero(var.type()), op->value - 1)));
        res = IRMutator::Mutate_(op, stmt);
        relax_map_.erase(var.get());
      } else {
        hint_map_.insert(std::pair<const Variable*, IntSet>(
            var.get(), IntSet::interval(make_zero(var.type()), op->value - 1)));
        res = IRMutator::Mutate_(op, stmt);
        hint_map_.erase(var.get());
      }
      return res;
    }
  }

 private:
  /*!
   * \brief Main function to try partition on a For node.
   *
   * \param node pointer to the node
   * \param stmt HalideIR AST of the stmt
   * \param var loop variable of the For loop under consideration
   * \param min Min of the For loop IV
   * \param max Max of the For loop IV
   * \param body For loop body stmt
   * \param partition_thread_scope whether to partition thread scope or not. Used for the GPUs
   *
   * \return Partitioned Stmt if succeeds else empty stmt
   * */
  Stmt TryPartition(const Node* node, const Stmt& stmt, VarExpr var, Expr min, Expr max, Stmt body,
                    bool partition_thread_scope);

  /*!
   * \brief Try partition based on conditions involving `Modulo` or `Div` op. Invoked from
   * TryPartition() only. This is different compared to TryPartitionRange in the sense that,
   * partitioned loops do not access the data sequentially.
   *
   * \param node pointer to the node
   * \param stmt HalideIR AST of the stmt
   * \param var loop variable of the For loop under consideration
   * \param min Min of the For loop IV
   * \param max Max of the For loop IV
   * \param body For loop body stmt
   * \param vrange TVM Map from Variable to Range. Used during simplify and related functions.
   * \param partition_map map containing conditions and intervals over which they are true or false,
   *        obtained from PartitionFinder
   * \param partition_thread_scope whether to partition thread scope or not. Used for the GPUs
   * \return Partitioned Stmt if succeeds else empty stmt
   * */
  Stmt TryPartitionDivMod(const Node* node, const Stmt& stmt, VarExpr var, Expr min, Expr max,
                          Stmt body, const Map<Var, Range>& vrange,
                          const std::unordered_map<const Node*, Expr>& partition_map,
                          bool partition_thread_scope);

  /*!
   * \brief Try partition based on conditions that do not contain `Div` or `Modulo` op. Invoked from
   * TryPartition() only. Partitions the loops such that data access is sequential.
   *
   * \param node pointer to the node
   * \param stmt HalideIR AST of the stmt
   * \param var loop variable of the For loop under consideration
   * \param min Min of the For loop IV
   * \param max Max of the For loop IV
   * \param body For loop body stmt
   * \param vrange TVM Map from Variable to Range. Used during simplify and related functions.
   * \param PartitionFinder
   * \param is_equal_equal true if we are partitioning for EQ or NE conditions, false otherwise
   * \param partition_thread_scope whether to partition thread scope or not. Used for the GPUs
   * \return Partitioned Stmt if succeeds else empty stmt
   * */
  Stmt TryPartitionRange(const Node* node, const Stmt& stmt, VarExpr var, Expr min, Expr max,
                         Stmt body, Map<Var, Range>& vrange, const PartitionFinder& finder,
                         bool partition_thread_scope);

  /*!
   * \brief Returns an interval over For loop's Range where some conditions are certainly either
   * True or False (given by cond_value). Returns this interval as well as those conditions that are
   * certainly either True or False over that interval.
   *
   * \param partitions' Map obtained from PartitionFinder that has RangeInfo for conditions for True
   * and/or False part. \param vrange TVM Map from Variable to Ranges. Used during simplify and
   * related functions. \param cond_value True or False. Used to get the True or False interval of a
   * condition.
   * */
  std::pair<IntSet, std::unordered_set<const Node*>> GetIntervalAndCondset(
      const Partition& partitions, const arith::IntervalSet& for_interval, Expr cond_value);

  /*!
   * \brief Makes the Normalized For loop with given extent and body.
   *        Used to create new partitioned loops that will replace the original loop.
   *
   * \param op Reference to the original loop node
   * \param extent Extent of the new loop for loop to be created
   * \param body For loop body
   * \param vrange TVM Map from Variable to Range, used for simplify and related functions
   * */
  inline Stmt MakeFor(const Node* op, Expr extent, Stmt body, const Map<Var, Range>& vrange);

  /* Candidate IRs that may be partitioned potentially */
  bool remove_div_mod_;  // if true it will remove division and modulo in the indexing
  bool is_extern_op_{false};
  std::unordered_map<const Variable*, IntSet> hint_map_;
  std::unordered_map<const Variable*, IntSet> relax_map_;
  arith::Analyzer analyzer_;
  CandidateSelector selector;
  std::unordered_set<const For*> multi_core_loop_;
};

// Returns an interval (in the first component) in which all the conditions
// given in the second component provably have value given by cond_value
std::pair<IntSet, std::unordered_set<const Node*>> LoopPartitioner::GetIntervalAndCondset(
    const Partition& partitions, const arith::IntervalSet& for_interval, const Expr cond_value) {
  using arith::IntervalSet;
  Array<IntSet> sets;
  std::unordered_set<const Node*> cond_set;

  for (const auto& kv : partitions) {
    if (Equal(kv.first.second, cond_value)) {
      arith::IntervalSet interval = Downcast<arith::IntervalSet>(kv.second);
      arith::IntervalSet intersection = arith::Intersect(&analyzer_, interval, for_interval);
      if (!intersection->IsEmpty()) {
        sets.push_back(kv.second);
        cond_set.insert(kv.first.first);
      }
    }
  }
  IntSet intset = sets.empty() ? IntSet::nothing() : Intersect(sets);  // equivalent to interval;
  return std::make_pair(intset, cond_set);
}

/* Splits the loop based on Modulo or Div standalone ops or conditions. These ops or conditions must
 * be partition-able (see CheckValidDivMod). Div or Mod ops can appear either in indexing of a
 * Tensor e.g A[i%2, j//3] or in a condition or in both the places. Div and mod can both appear at
 * the same time. e.g. A[i % 2 + i //3 , j]
 *
 * e.g. Assume we have following three ops on loop var `i`
 *
 * [1] (i % c1 Op k1), [2] (i % c2 Op k2) [3] (i // c3 Op k3) where Op by any CmpOpNode.
 *
 * Loop partition will generate `k` number of loops where `k = LCM(c1, c2, c2)` (Lowest Common
 * Multiplier)
 *
 * For each loop, `i` is replaced by k*i+ c4.
 * For each generated loop is different in terms of value of c4.
 * c4 takes one unique value for each loop from range [0, c4-1].
 *
 * Each loop can also be different in terms of `extent` if original loop doesn't have `Extent`
 * perfectly divisible by `k`. In such case, remainder of the extent is equally distributed amongst
 * the new loops beginning from new loop with IV `k*i`.
 *
 * e.g. (i % 2 == 0), (i % 3 == 3), (i // 6 == 4) and original loop has extent of `105`.
 *
 * k = LCM(2, 3, 6) = 6, 6 new loops will be generated with `i` replaced by
 * [1] `6i`, with extent = (105/6) + 1 = 17 + 1 = 18
 * [2] `6i+1` with extent = (105/6) + 1 = 17 + 1 = 18
 * [3] `6i+2` with extent = (105/6) + 1 = 17 + 1 = 18
 * [4] `6i+3` with extent = (105/6) = 17
 * [5] `6i+4` with extent = (105/6) = 17
 * [6] `6i+5` with extent = (105/6) = 17
 *
 * CAUTION: This loop partition alters the order of iterations of the loop.
 * require to: in case the LCM is very large, this will effectively just unroll the loop. we need to
 * check for such cases, also, if the LCM is very large, there is a possibility of Overflow. We
 * shouldn't create zero extent loops here otherwise if the LCM is large, it behaves like infinite
 * loop. Add check for Zero Extent loops or calculate in advance non-zero extent loops and only
 * iterate through them.
 * */
Stmt LoopPartitioner::TryPartitionDivMod(const air::Node* node, const Stmt& stmt, const VarExpr var,
                                         const Expr min, const Expr max, const Stmt body,
                                         const Map<Var, Range>& vrange,
                                         const std::unordered_map<const Node*, Expr>& partition_map,
                                         bool partition_thread_scope) {
  Stmt s;
  int running_lcm = 1;
  for (auto& kv : partition_map) {
    CHECK(kv.second.as<IntImm>()) << "Div or Mod divisor is not an IntImm";
    running_lcm = static_cast<int>(lcm(running_lcm, kv.second.as<IntImm>()->value));
  }

  Expr range = max - min + 1;
  Expr division = truncdiv(range, running_lcm);
  Expr remainder = truncmod(range, running_lcm);

  for (int i = 0; i < running_lcm; i++) {
    Expr new_extent = division + (i < remainder);  // compute new extent + distribute remainder
    Expr new_var = var * running_lcm + i;
    Stmt new_body = Substitute(body, {{Var{var}, new_var}});
    Stmt new_stmt = MakeFor(node, new_extent, new_body, vrange);
    new_stmt = Simplify_cce(new_stmt, vrange);
    new_stmt = VisitAndMutate(new_stmt);
    s = AppendStmts(s, new_stmt);
  }
  s = ConvertSSA(s);
  return s;
}

/*
 * Tries to recursively partition the range of the variable (given by var) of
 * the for loop (given by node and stmt) into a
 * number of disjoint ranges such that in some ranges one or more predicates
 * in the loopnest are provably true or false in each range. For example, given the
 * following loop to partition:
 * for (i = 0; i < 4; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(i*10 + j < 36))
 *            A[10*i+j] = B[10*i+j]
 *
 * We first partition range of i, i.e., [0,3] into subranges [0,2] and [3,3] because the
 * likely condition is always true for the first subrange but not always true for the
 * second subrange. Therefore, we'll have
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely(1))
 *           A[10*i+j] = B[10*i+j]
 * for (i = 0; i < 1; i++)
 *    for (j = 0; j < 10; j++)
 *        if (likely((i+3)*10 + j < 36))
 *            A[10*(i+3)+j] = B[10*(i+3)+j]
 * Which is simplified as:
 * for (i = 0; i < 3; i++)
 *    for (j = 0; j < 10; j++)
 *        A[10*i+j] = B[10*i+j]
 * for (j = 0; j < 10; j++) // loopnest 1
 *    if (likely(j < 6))
 *            A[30+j] = B[30+j]
 * Now, we recursively partition j in loopnest 1 into subranges [0,5] and [6,9] where the
 * condition is true for the first subrange and now always true for the second subrange.
 * for (j = 0; j < 6; j++)
 *    if (likely(1))
 *         A[30+j] = B[30+j]
 * for (j = 0; j < 4; j++) // loop 2
 *    if (likely(j < 0))
 *        A[36+j] = B[36+j]
 * Finally we recursively partition loop 2 above into subrange [0,3] where the
 * condition is false and empty interval where the condition is not false,
 * therefore we generate
 * for (j = 0; j < 4; j++)
 *    if (likely(0))
 *        A[36+j] = B[36+j]
 * which will eventually be simplified to empty code. And because only one loop was generated
 * from loop 2 we stop recursing.
 */

Stmt LoopPartitioner::TryPartitionRange(const air::Node* node, const Stmt& stmt, const VarExpr var,
                                        Expr min, Expr max, const Stmt body,
                                        Map<Var, Range>& vrange, const PartitionFinder& finder,
                                        bool partition_thread_scope) {
  using namespace arith;
  min = SuperSimplify(min, vrange);
  max = SuperSimplify(max, vrange);
  arith::IntervalSet for_interval(min, max);
  bool cond_value;
  IntSet middle_interval;
  std::unordered_set<const Node*> cond_set;
  CHECK(finder.partitions_.size() == 1 || finder.partitions_.size() == 2)
      << "Only solve one condition at a time";
  // find an interval in which all conditions on var are true
  if (finder.partition_type_ == kMaxMin) {
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions_, for_interval, finder.true_max_min_);
  } else {
    std::tie(middle_interval, cond_set) =
        GetIntervalAndCondset(finder.partitions_, for_interval, const_true());
  }
  if (middle_interval.is_nothing()) {
    // if such interval doesn't exist, find an interval in which all
    // conditions on var are false
    if (finder.partition_type_ == kMaxMin) {
      std::tie(middle_interval, cond_set) =
          GetIntervalAndCondset(finder.partitions_, for_interval, finder.false_max_min_);
    } else {
      std::tie(middle_interval, cond_set) =
          GetIntervalAndCondset(finder.partitions_, for_interval, const_false());
    }
    if (middle_interval.is_nothing())
      // we couldn't find an interval in which the conditions are provably true or false
      // Therefore, we can't partition the loop based on those conditions
      return Stmt();
    cond_value = false;
  } else {
    cond_value = true;
  }

  arith::IntervalSet middle_interval_i =
      Downcast<IntervalSet>(middle_interval);  // if the interval is [0, pos_inf] return
  if (is_zero(middle_interval.min()) && !middle_interval_i->HasUpperBound()) {
    return Stmt();
  }

  arith::IntervalSet intersection = arith::Intersect(&analyzer_, middle_interval_i, for_interval);
  if (Equal(Simplify_cce(intersection.max()), max) &&
      Equal(Simplify_cce(intersection.min()), min)) {
    if (DEBUG_LOOP_PARTITION) {
      LOG(INFO) << "Middle interval same as the For interval " << std::endl;
    }
    return Stmt();
  }

  if (DEBUG_LOOP_PARTITION) {
    LOG(INFO) << "Middle interval " << middle_interval_i << std::endl;
    LOG(INFO) << "Partition type: " << static_cast<int>(finder.partition_type_) << std::endl;
  }
  // middle_interval is the subrange of the loop variable range for which a
  // set of conditions are true (or false resp.)
  // The part of the loop variable range that is before (after resp.) that
  // subrange is prefixed with pre- (post- resp.)

  // Calculating pre-subrange and generating code for it.
  // pre-subrange = [min, body_begin)
  Expr body_begin, extent;
  Stmt pre_stmt;
  bool pre_stmt_recurse = true;
  if (middle_interval_i->HasLowerBound()) {
    body_begin = SuperSimplify(middle_interval.min(), vrange);
    Expr cond = SuperSimplify(body_begin - min >= 0, vrange);
    if (!CanProve(cond, vrange)) {
      LOG(WARNING) << "Cannot prove: " << cond << ", when generating the pre doubt loop";
      body_begin = Max::make(body_begin, min);
      // stop recursing on this interval if we can't prove it has non-negative length
      pre_stmt_recurse = false;
    }
    if (!partition_thread_scope) {
      Stmt pre_body = body;
      if (finder.partition_type_ == kEqual) {
        // only for the kEqual, we can be 100 % sure that it will take opposite value than mid_stmt
        pre_body = ConditionEliminator(cond_set, (cond_value ? const_false() : const_true()))
                       .Mutate(pre_body);
      }
      pre_body = Substitute(pre_body, {{Var{var}, var + min}});
      extent = SuperSimplify(body_begin - min, vrange);
      pre_stmt = MakeFor(node, extent, pre_body, vrange);
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "pre_stmt before simplify: " << std::endl << pre_stmt << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
      }
      if (const For* op = pre_stmt.as<For>()) {
        vrange.Set(Var{op->loop_var}, Range::make_by_min_extent(0, extent));
      }
      pre_stmt = Simplify_cce(pre_stmt, vrange);
    }
  } else {
    body_begin = min;
  }

  // Calculating post-subrange and generating code for it.
  // post-subrange = [post_doubt_begin, max + 1)
  Expr post_doubt_begin;
  Stmt post_stmt;
  bool post_stmt_recurse = true;
  if (middle_interval_i->HasUpperBound()) {
    post_doubt_begin = SuperSimplify(middle_interval.max() + 1, vrange);
    // require the extent to be non-negative
    Expr cond = SuperSimplify((max - post_doubt_begin + 1 >= 0), vrange);
    if (!CanProve(cond, vrange)) {
      LOG(WARNING) << "Cannot prove: " << cond << ", when generating the post doubt loop";
      post_doubt_begin = Min::make(post_doubt_begin, max + 1);
      // stop recursing on this interval if we can't prove it has non-negative length
      post_stmt_recurse = false;
    }
    if (!partition_thread_scope) {
      Stmt post_body = body;
      if (finder.partition_type_ == kEqual) {
        // only for the kEqual, we can be 100 % sure that it will take opposite value than mid_stmt
        post_body = ConditionEliminator(cond_set, (cond_value ? const_false() : const_true()))
                        .Mutate(post_body);
      }
      post_body = Substitute(post_body, {{Var{var}, var + post_doubt_begin}});
      extent = SuperSimplify(max - post_doubt_begin + 1, vrange);
      post_stmt = MakeFor(node, extent, post_body, vrange);
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "post stmt before simplify: " << std::endl << post_stmt << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
      }
      if (const For* op = post_stmt.as<For>()) {
        vrange.Set(Var{op->loop_var}, Range::make_by_min_extent(0, extent));
      }
      post_stmt = Simplify_cce(post_stmt, vrange);
    }
  } else {
    post_doubt_begin = max + 1;
  }

  Stmt s;

  // Generating code for middle subrange
  if (!partition_thread_scope) {
    Stmt mid_stmt, simplified_body;
    if (!CanProve(body_begin >= post_doubt_begin, vrange)) {
      // [body_begin, post_doubt_begin)
      // It is proven that over the mid_stmt given condition has certain value, replace it
      if (finder.partition_type_ == kMaxMin) {
        simplified_body = ConditionEliminator(
                              cond_set, (cond_value ? finder.true_max_min_ : finder.false_max_min_))
                              .Mutate(body);
      } else {
        simplified_body =
            ConditionEliminator(cond_set, (cond_value ? const_true() : const_false())).Mutate(body);
      }
      Stmt new_body = Substitute(simplified_body, {{Var{var}, var + body_begin}});
      extent = SuperSimplify(post_doubt_begin - body_begin, vrange);
      mid_stmt = MakeFor(node, extent, new_body, vrange);
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "mid stmt before simplify :" << std::endl << mid_stmt << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
      }
      if (const For* op = mid_stmt.as<For>()) {
        vrange.Set(Var{op->loop_var}, Range::make_by_min_extent(0, extent));
      }
      mid_stmt = Simplify_cce(mid_stmt);
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "mid_stmt before mutate: " << std::endl << mid_stmt << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "{ [MID] " << std::endl;
      }
      mid_stmt = VisitAndMutate(mid_stmt);
      if (DEBUG_LOOP_PARTITION) {
        LOG(INFO) << "}" << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
        LOG(INFO) << "mid_stmt after mutate: " << std::endl << mid_stmt << std::endl;
        LOG(INFO) << "==========================================================" << std::endl;
      }

      // Recurse for each non-empty subrange
      if (pre_stmt.defined() && pre_stmt_recurse) {
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "pre_stmt before mutate: " << std::endl << pre_stmt << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "{ [PRE]" << std::endl;
        }
        pre_stmt = VisitAndMutate(pre_stmt);
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "}" << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "pre_stmt after mutate: " << std::endl << pre_stmt << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
        }
      }
      if (post_stmt.defined() && post_stmt_recurse) {
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "post_stmt before mutate: " << std::endl << post_stmt << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "{ [POST]" << std::endl;
        }
        post_stmt = VisitAndMutate(post_stmt);
        if (DEBUG_LOOP_PARTITION) {
          LOG(INFO) << "}" << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
          LOG(INFO) << "post stmt after mutate: " << std::endl << post_stmt << std::endl;
          LOG(INFO) << "==========================================================" << std::endl;
        }
      }
    }
    s = AppendStmts(pre_stmt, mid_stmt);
    s = AppendStmts(s, post_stmt);
  } else {
    Expr cond = const_true();
    if (!CanProve(body_begin == min, vrange)) cond = cond && (var >= body_begin);
    if (!CanProve(post_doubt_begin == (max + 1), vrange)) cond = cond && (var < post_doubt_begin);
    s = ThreadPartitionInserter(cond_set, cond).Mutate(stmt);
  }
  s = ConvertSSA(s);
  return s;
}

Stmt LoopPartitioner::TryPartition(const Node* node, const Stmt& stmt, VarExpr var, Expr min,
                                   Expr max, Stmt body, bool partition_thread_scope) {
  if (DEBUG_LOOP_PARTITION) {
    LOG(INFO) << "========================================================================="
              << std::endl;
    LOG(INFO) << "Input to partition: " << std::endl << stmt << std::endl;
    LOG(INFO) << "==========================================================" << std::endl;
    LOG(INFO) << "Partitioning for variable: " << var << std::endl;
    LOG(INFO) << "For interval: [" << min << ", " << max << "]" << std::endl;
    LOG(INFO) << "==========================================================" << std::endl;
  }

  PartitionFinder finder(var, min, max, remove_div_mod_, hint_map_, relax_map_);
  finder.Visit(body);

  Map<Var, Range> vrange;  // vrange is the tvm Map used for the Simplify() and CanProve()
  for (const auto& kv : hint_map_) {
    vrange.Set(Var(GetObjPtr(kv.first)),
               Range::make_by_min_extent(kv.second.min(),
                                         ir::Simplify_cce(kv.second.max() - kv.second.min() + 1)));
  }

  if (!finder.partitions_.empty()) {
    return TryPartitionRange(node, stmt, std::move(var), std::move(min), std::move(max),
                             std::move(body), vrange, finder, partition_thread_scope);
  } else if (!finder.div_modulo_.empty()) {
    if (is_extern_op_) {
      LOG(WARNING) << "ExternOp or HybridOp has a partition-able condition based on Modulo Op. "
                      "But skipping the loop partition based on it, to avoid changing the loop "
                      "iteration order.";
      return Stmt();
    } else {
      return TryPartitionDivMod(node, stmt, std::move(var), std::move(min), std::move(max),
                                std::move(body), vrange, finder.div_modulo_,
                                partition_thread_scope);
    }
  }

  if (DEBUG_LOOP_PARTITION) {
    LOG(INFO) << "All maps are empty, nothing to partition" << std::endl;
  }
  return Stmt();
}

inline Stmt LoopPartitioner::MakeFor(const Node* node, const Expr extent, const Stmt body,
                                     const Map<Var, Range>& vrange) {
  const For* for_node = static_cast<const For*>(node);
  CHECK(for_node);
  if (CanProve(extent == make_const(Int(32), 1), vrange)) {
    // If the loop extent is 1, do not create the loop anymore
    auto stmt = Substitute(body, {{Var{for_node->loop_var}, make_const(Int(32), 0)}});
    // For the multi-core loop, keep the For whose extent is 1 to facilitate the processing of subsequent
    // the multi-core loop merging.
    if (multi_core_loop_.count(for_node)) {
      stmt = For::make(for_node->loop_var, 0, 1, for_node->for_type, for_node->device_api, stmt);
    }
    return stmt;
  } else {
    return For::make(for_node->loop_var, 0, extent, for_node->for_type, for_node->device_api, body);
  }
}

class RemoveLikelyTags : public IRMutator {
 public:
  using IRMutator::Mutate;

  Expr Mutate_(const Call* op, const Expr& e) override {
    if (op->is_intrinsic(Call::likely)) {
      CHECK_EQ(op->args.size(), 1);
      return IRMutator::Mutate(op->args[0]);
    } else {
      return IRMutator::Mutate_(op, e);
    }
  }
};

class IsConv_CCE : public IRVisitor {
 public:
  IsConv_CCE() {}
  ~IsConv_CCE() override = default;

  bool Check(const Stmt stmt) {
    this->Visit(stmt);
    return is_conv_;
  }

  void Visit_(const AttrStmt* op) final {
    if (air::ir::attr::IsPragmaKey(op->attr_key) && op->attr_key == "pragma_emit_insn" &&
        op->value.as<StringImm>() && op->value.as<StringImm>()->value == "mad") {
      is_conv_ = true;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Provide* op) final {
    if (const auto* compute_op = op->func.as<ComputeOpNode>()) {
      if (compute_op->attrs.count("pragma_conv_padding_top") ||
          compute_op->attrs.count("pragma_conv_padding_bottom") ||
          compute_op->attrs.count("pragma_conv_padding_left") ||
          compute_op->attrs.count("pragma_conv_padding_right") ||
          compute_op->attrs.count("pragma_conv_dilation_h") ||
          compute_op->attrs.count("pragma_conv_dilation_w")) {
        is_conv_ = true;
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  bool is_conv_{false};
};

class FindSumMulLoop : public IRVisitor {
 public:

  void Visit_(const Provide* op) override {
    if (in_loop) {
      if(duplicate) {
        not_valid = true;
        duplicate = false;
        return;
      }
      if(op->args.size() < 5) {
        not_valid = true;
        duplicate = false;
        return;
      }

      if(!init_found) {
        if(CanProve(op->value == 0)) {
          init_found = true;
          tensor_name = op->func->func_name();
          init_stmt = op;
        }
      } else {
        //Case 1: b = b + ((head * mean) * rsqrt)
        if (auto add = op->value.as<Add>()) {
          auto a_name = add->a.as<Call>();
          if(op->func->func_name() != tensor_name || (a_name && a_name->name != tensor_name)) {
            not_valid = true;
            return;
          }

          if (auto mul = add->b.as<Mul>()) {
            if (auto mul_b = mul->b.as<Mul>()) {
              if (auto call = mul_b->b.as<Call>()) {
                if (call->call_type == Call::CallType::PureIntrinsic &&
                    call->name == "exp") {
                    // Confirm with args that it is a reduction.
                    duplicate = true;
                    sum_mul_stmt = op;
                    stmt1 = Provide::make(op->func, op->value_index, add->a + (mul->a * mul_b->a),
                                          op->args);
                    
                    auto new_expr_b = (add->a * mul_b->b);
                    stmt2 = Provide::make(op->func, op->value_index, new_expr_b, op->args);

                    GatherVars(new_expr_b, &used_vars);
                    return;
                  }
                }
            } else  {//Check for second case
              //Case 2: b = b + ((((head * gamma) * x - m) * exp) * (a/var)
              if (auto mul_a = mul->a.as<Mul>()) {
                if (auto mul_a_a = mul_a->a.as<Mul>()) {
                  if (auto call = mul_a_a->b.as<Call>()) {
                    if (call->call_type == Call::CallType::PureIntrinsic && 
                        call->name == "exp") {
                      // Confirm with args that it is a reduction.
                      duplicate = true;
                      sum_mul_stmt = op;
                      stmt1 = Provide::make(op->func, op->value_index, add->a + (mul_a_a->a),
                                            op->args);
                      
                      auto new_expr_b = (((add->a * mul_a_a->b) * mul_a->b) * mul->b);
                      stmt2 = Provide::make(op->func, op->value_index, new_expr_b, op->args);

                      GatherVars(new_expr_b, &used_vars);
                      return;
                    }
                  }
                }
              }
            }
          }
          not_valid = true;
        } else {
          //Provide is not initialization and it's not the pattern we're looking for.
          not_valid = true;
        }
      }

    }
  }

  void Visit_(const For* op) override {
    if(finished) {
      return;
    }
    
    if(!in_loop) {
      in_loop = true;
      first_loop = op;
    }

    IRVisitor::Visit_(op);

    if(op == first_loop) {
      if (duplicate && !not_valid) {
        //The pattern was found in this loop and needs to be splitted
        finished = true;
      } else {
        init();
      }
    }
  }

/* 
 * Initialize all attributes in class to perform a new Visit and Mutate
 * */
  void init() {
    tensor_name = "";
    duplicate = false;
    finished = false;
    init_found = false;
    not_valid = false;
    in_loop = false;
  }

  std::string tensor_name;
  Stmt stmt1;
  Stmt stmt2;
  const For* first_loop;
  const Provide* init_stmt;
  const Provide* sum_mul_stmt;
  std::unordered_set<Var, air::NodeHash, air::NodeEqual> used_vars;
  bool duplicate = false;
 
 private:
  bool finished = false;
  bool init_found = false;
  bool not_valid = false;
  bool in_loop = false;
};

class SplitSumMulLoop : public IRMutator {
 public:
  Stmt VisitAndMutate(Stmt stmt) {
    Stmt ret;
    do {
      ret = stmt;
      finder.Visit(stmt);
      if(finder.duplicate) {
        stmt = Mutate(stmt);
        finder.init();
      }
    } while(!ret.same_as(stmt));
    return stmt;
  }

  Stmt Mutate_(const Provide* op, const Stmt& s) override {
     if (in_split_loop) {
      if(replace_round == 1) {
        if(op == finder.sum_mul_stmt) {

          return finder.stmt1;
        }
      } else if(replace_round == 2) {
        if(op == finder.sum_mul_stmt) {
          return finder.stmt2;
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block* op, const Stmt& s) override {
    if(auto prov = op->first.as<Provide>()) {
      //Check if this block contains the initialization of the tensor
      //If in the second round, remove it.
      if(replace_round == 2 && prov == finder.init_stmt) {
        return Mutate(op->rest);
      }
    }

    auto ret = IRMutator::Mutate_(op, s);
    return ret;
  }

  Stmt Mutate_(const For* op, const Stmt& s) override {
    Stmt new_stmt = s;
    //This is the loop that will be splitted.
    if(op == finder.first_loop && !in_split_loop) {
      in_split_loop = true;
      replace_round = 1;
      Stmt first_loop = IRMutator::Mutate_(op, s);
      replace_round = 2;
      Stmt second_loop = IRMutator::Mutate_(op, s);
      new_stmt = Block::make(first_loop, second_loop);
      replace_round = 0;
      in_split_loop = false;
      return new_stmt;
    }
    
    if(in_split_loop && replace_round == 2) {
      if(finder.used_vars.find(op->loop_var) == finder.used_vars.end()) {
        return Mutate(op->body);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  FindSumMulLoop finder;
  int replace_round = 0;
  bool in_split_loop = false;
};

Stmt LoopPartition(Stmt stmt, bool split_const_loop) {
  stmt = LoopPartitioner(split_const_loop, false).VisitAndMutate(stmt);
  stmt = RemoveLikelyTags().Mutate(stmt);
  return stmt;
}

Stmt LoopPartitionCCE(Stmt stmt, bool split_const_loop, bool remove_div_mod, bool partition_conv) {
  if (!partition_conv && IsConv_CCE().Check(stmt)) return stmt;
  if (DEBUG_LOOP_PARTITION) {
    LOG(INFO) << "In the Loop Partition:" << std::endl;
    LOG(INFO) << "==========================================================" << std::endl;
  }

  stmt = SplitSumMulLoop().VisitAndMutate(stmt);
  for (int i = 0; i < 5; i++) {
    stmt = LoopPartitioner(split_const_loop, remove_div_mod).VisitAndMutate(stmt);
    stmt = RemoveLikelyTags().Mutate(stmt);
    stmt = Simplify_cce(stmt);
    stmt = RemoveNoOp(stmt);
  }
  return stmt;
}
}  // namespace ir
}  // namespace air
