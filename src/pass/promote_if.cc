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

/**
 * This pass promotes the IF stmt out of FOR loop reasonably.
 *
 * For example, cases like this:
 *
 * for (i, 0, 16) {
 *     for (c, 0, 16) {
 *         if ((i * 2) == var) {
 *             out(i, c) = in(i, c)
 *         }
 *     }
 * }
 *
 * will be transformed into:
 *
 * for (i, 0, 16) {
 *     if ((i * 2) == var) {
 *         for (c, 0, 16) {
 *             out(i, c) = in(i, c)
 *         }
 *     }
 * }
 *
 */

#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_pass.h>
#include <tvm/operation.h>

#include <ir_pass.h>

namespace akg {
namespace ir {
class StmtSinker : public IRMutator {
 public:
  StmtSinker()
      : judge_func_(), pack_func_(), satisfactions_(), violations_(), support_else_case_(true), is_root_(true) {}
  ~StmtSinker() override = default;

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt rst;

    bool local_root = is_root_;
    if (local_root) {
      satisfactions_.emplace_back();
    }

    bool has_violation = false;
    if (!op->else_case.defined()) {
      is_root_ = false;
      if (judge_func_(op->condition)) {
        satisfactions_.back().push_back(op->condition);
      } else {
        violations_.push_back(op->condition);
        has_violation = true;
      }
      if (auto obj = op->then_case.as<IfThenElse>()) {
        rst = Mutate_(obj, op->then_case);
      } else {
        rst = pack_func_(AddIfStmt(op->then_case, violations_));
      }
    } else {
      if (!judge_func_(op->condition) || !support_else_case_) {
        rst = pack_func_(AddIfStmt(s, violations_));
      } else {
        Stmt then_stmt;
        if (auto obj = op->then_case.as<IfThenElse>()) {
          is_root_ = true;
          then_stmt = Mutate_(obj, op->then_case);
        } else {
          then_stmt = pack_func_(AddIfStmt(op->then_case, violations_));
        }

        Stmt else_stmt;
        if (auto obj = op->else_case.as<IfThenElse>()) {
          is_root_ = true;
          else_stmt = Mutate_(obj, op->else_case);
        } else {
          else_stmt = pack_func_(AddIfStmt(op->else_case, violations_));
        }

        rst = IfThenElse::make(op->condition, then_stmt, else_stmt);
      }
    }

    if (has_violation) {
      violations_.pop_back();
    }
    if (local_root) {
      rst = AddIfStmt(rst, satisfactions_.back());
      satisfactions_.pop_back();
    }
    return rst;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (auto obj = op->body.as<IfThenElse>()) {
      Var var(op->loop_var);
      judge_func_ = [&var](const Expr &e) { return !air::ir::ExprUseVar(e, var); };
      pack_func_ = [=](const Stmt &stmt) {
        return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, stmt);
      };
      return Mutate_(obj, op->body);
    }
    return s;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    auto obj_ph = op->node.as<PlaceholderOpNode>();
    if (op->attr_key == air::ir::attr::realize_scope && obj_ph) {
      if (auto obj = op->body.as<IfThenElse>()) {
        judge_func_ = [&, this](const Expr &e) { return !HasCallName(e, obj_ph->name); };
        pack_func_ = [=](const Stmt &stmt) { return AttrStmt::make(op->node, op->attr_key, op->value, stmt); };
        return Mutate_(obj, op->body);
      }
    }
    return s;
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (auto obj = op->body.as<IfThenElse>()) {
      judge_func_ = [&, this](const Expr &e) { return !HasCallName(e, op->func->func_name()); };
      pack_func_ = [=](const Stmt &stmt) {
        return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition, stmt);
      };
      support_else_case_ = false;
      return Mutate_(obj, op->body);
    }
    return s;
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    if (auto obj = op->body.as<IfThenElse>()) {
      judge_func_ = [](const Expr &) { return true; };
      pack_func_ = [=](const Stmt &stmt) { return ProducerConsumer::make(op->func, op->is_producer, stmt); };
      return Mutate_(obj, op->body);
    }
    return s;
  }

 private:
  bool HasCallName(const Expr &e, const std::string &str) {
    class Collector : public IRVisitor {
     public:
      explicit Collector(const std::string &s) : s_(s), has_(false) {}
      ~Collector() override = default;

      void Visit_(const Call *op) final {
        if (!has_ && op->name == s_) {
          has_ = true;
        }
      }
      const std::string &s_;
      bool has_{false};
    } collector(str);

    collector.Visit(e);
    return collector.has_;
  }

  Stmt AddIfStmt(const Stmt &s, const std::vector<Expr> &conds) {
    auto rst = s;
    for (auto it = conds.rbegin(); it != conds.rend(); ++it) {
      rst = IfThenElse::make(*it, rst);
    }
    return rst;
  }

  std::function<bool(const Expr &)> judge_func_;
  std::function<Stmt(const Stmt &)> pack_func_;
  std::list<std::vector<Expr>> satisfactions_;
  std::vector<Expr> violations_;
  bool support_else_case_, is_root_;
};

class IFPromoter : public IRMutator {
 public:
  IFPromoter() : block_map_() {}
  ~IFPromoter() override = default;

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    auto ptr = stmt.as<IfThenElse>();

    CHECK(ptr);
    auto then_case = ptr->then_case;
    static_cast<void>(
      ExtractCommonIf(then_case, ptr->else_case, [&stmt, ptr](Stmt &first, const Stmt &second, bool &changed) {
        if (ptr->then_case.same_as(first) && ptr->else_case.same_as(second)) {
          first = stmt;
        } else {
          first = IfThenElse::make(ptr->condition, first, second);
        }
        changed = true;
      }));
    return then_case;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    bool is_root = false;
    auto it = block_map_.find(s);
    if (it == block_map_.end()) {
      block_map_.emplace(s, s);
      is_root = true;
    }
    CHECK(op->first.defined());
    if (op->first->IsInstance<Block>()) {
      block_map_.emplace(op->first, s);
    }
    CHECK(op->rest.defined());
    if (op->rest->IsInstance<Block>()) {
      block_map_.emplace(op->rest, s);
    }

    auto stmt = IRMutator::Mutate_(op, s);

    if (is_root) {
      return ProcessBlock(stmt);
    }

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    return StmtSinker().Mutate(stmt);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    return StmtSinker().Mutate(stmt);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    return StmtSinker().Mutate(stmt);
  }

  Stmt Mutate_(const ProducerConsumer *op, const Stmt &s) final {
    auto stmt = IRMutator::Mutate_(op, s);
    return StmtSinker().Mutate(stmt);
  }

 private:
  void SearchStmts(const Stmt &s, std::vector<Stmt> &v) {
    if (auto b = s.as<Block>()) {
      SearchStmts(b->first, v);
      SearchStmts(b->rest, v);
    } else {
      v.push_back(s);
    }
  }

  bool ExtractCommonIf(Stmt &first, const Stmt &second,
                       const std::function<void(Stmt &s0, const Stmt &s1, bool &)> &process_leaves) {
    std::vector<Expr> cond0, cond1;
    Stmt leaf0, leaf1;

    auto extract_cond = [](std::vector<Expr> &c, const Stmt &s, Stmt &leaf) {
      leaf = s;
      auto ptr = leaf.as<IfThenElse>();
      while (ptr != nullptr && !ptr->else_case.defined()) {
        c.push_back(ptr->condition);
        leaf = ptr->then_case;
        ptr = leaf.as<IfThenElse>();
      }
    };
    extract_cond(cond0, first, leaf0);
    extract_cond(cond1, second, leaf1);

    std::vector<Expr> common_cond;
    for (auto it0 = cond0.begin(); it0 != cond0.end();) {
      bool find_common = false;
      for (auto it1 = cond1.begin(); it1 != cond1.end();) {
        if (Equal(*it0, *it1)) {
          common_cond.push_back(*it0);
          it0 = cond0.erase(it0);
          it1 = cond1.erase(it1);
          find_common = true;
          break;
        } else {
          ++it1;
        }
      }
      if (!find_common) {
        ++it0;
      }
    }

    for (auto it = cond0.rbegin(); it != cond0.rend(); ++it) {
      leaf0 = IfThenElse::make(*it, leaf0);
    }
    for (auto it = cond1.rbegin(); it != cond1.rend(); ++it) {
      leaf1 = IfThenElse::make(*it, leaf1);
    }

    bool rst = !common_cond.empty();
    process_leaves(leaf0, leaf1, rst);
    if (rst) {
      for (auto it = common_cond.rbegin(); it != common_cond.rend(); ++it) {
        leaf0 = IfThenElse::make(*it, leaf0);
      }
      first = leaf0;
    }
    return rst;
  }

  void TryToMergeIf(Stmt &first, const Stmt &second, bool &changed) {
    auto Merge = [this](Stmt &first, const Stmt &second) {
      auto if0 = first.as<IfThenElse>();
      auto if1 = second.as<IfThenElse>();
      if (if0 == nullptr || if1 == nullptr) {
        return false;
      }
      if (Equal(if0->condition, if1->condition)) {
        auto then_stmt = ProcessBlock(Block::make(if0->then_case, if1->then_case));
        std::vector<Stmt> else_stmts;
        if (if0->else_case.defined()) {
          else_stmts.push_back(if0->else_case);
        }
        if (if1->else_case.defined()) {
          else_stmts.push_back(if1->else_case);
        }
        auto else_stmt = ProcessBlock(Block::make(else_stmts));
        first = IfThenElse::make(if0->condition, then_stmt, else_stmt);
        return true;
      } else if (Equal(if0->condition, Simplify_cce(Not::make(if1->condition)))) {
        std::vector<Stmt> candidates;
        candidates.push_back(if0->then_case);
        if (if1->else_case.defined()) {
          candidates.push_back(if1->else_case);
        }
        auto then_stmt = ProcessBlock(Block::make(candidates));
        candidates.clear();

        if (if0->else_case.defined()) {
          candidates.push_back(if0->else_case);
        }
        candidates.push_back(if1->then_case);
        auto else_stmt = ProcessBlock(Block::make(candidates));

        first = IfThenElse::make(if0->condition, then_stmt, else_stmt);
        return true;
      }
      return false;
    };

    std::vector<Stmt> old_stmts, new_stmts;
    SearchStmts(first, old_stmts);
    SearchStmts(second, old_stmts);

    // clean Block, merge multi if;
    auto last_stmt = Evaluate::make(0);
    for (auto &e : old_stmts) {
      if (!Merge(last_stmt, e)) {
        new_stmts.push_back(last_stmt);
        last_stmt = e;
      } else {
        changed = true;
      }
    }
    if (changed) {
      new_stmts.erase(new_stmts.begin());
      new_stmts.push_back(last_stmt);
      first = Block::make(new_stmts);
    }
  }

  Stmt ProcessBlock(const Stmt &stmt) {
    std::vector<Stmt> old_stmts, new_stmts;
    SearchStmts(stmt, old_stmts);

    // clean Block, merge multi if;
    auto last_stmt = Evaluate::make(0);
    for (auto &e : old_stmts) {
      using std::placeholders::_1;
      using std::placeholders::_2;
      using std::placeholders::_3;
      if (!ExtractCommonIf(last_stmt, e, std::bind(&IFPromoter::TryToMergeIf, this, _1, _2, _3))) {
        new_stmts.push_back(last_stmt);
        last_stmt = e;
      }
    }
    new_stmts.erase(new_stmts.begin());
    new_stmts.push_back(last_stmt);
    return Block::make(new_stmts);
  }

  std::map<Stmt, Stmt> block_map_;
};

Stmt PromoteIfStmt(Stmt stmt, bool is_dynamic) {
  // We do RemoveNoOp because Poly and Simplification may generate trivial nodes.
  if (is_dynamic)
    stmt = RemoveNoOp(stmt);
  else
    stmt = RemoveNoOp(air::ir::CanonicalSimplify(stmt));
  return IFPromoter().Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
