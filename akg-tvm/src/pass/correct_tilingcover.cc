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

#include "tvm.h"

namespace akg {
namespace ir {
class CoverCorrect : public IRMutator {
 public:
  CoverCorrect() : loop_level(0) {}
  ~CoverCorrect() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    const auto savevar = op->node.as<Variable>();
    if (savevar != nullptr && savevar->name_hint == "coverHead_save") {
      const auto alloc = op->body.as<Allocate>();
      if (alloc != nullptr) {
        Stmt allocs = Allocate::make(alloc->buffer_var, alloc->type, alloc->extents, alloc->condition,
                                     Evaluate::make(0), alloc->new_expr, alloc->free_function);
        allocstmt = AttrStmt::make(op->node, op->attr_key, op->value, allocs);

        return IRMutator::Mutate(alloc->body);
      }
    }

    if (op->attr_key == "Cover_save") {
      if (record.size() == 0) {
        return Evaluate::make(0);
      }
    }
    Stmt stmt = IRMutator::Mutate_(op, s);
    op = stmt.as<AttrStmt>();
    CHECK(op);
    if (op->attr_key == "Cover_save") {
      const auto bl = op->body.as<Block>();
      CHECK(bl);
      const auto ift = bl->first.as<IfThenElse>();
      Expr condition;
      for (auto fit = record.begin(); fit != record.end(); ++fit) {
        Expr judge = EQ::make((*fit).loop_var, (*fit).min);
        if (condition.defined()) {
          condition = And::make(condition, judge);
        } else {
          condition = judge;
        }
      }

      CHECK(ift);
      Stmt ifstmt = IfThenElse::make(condition, ift->then_case, ift->else_case);
      stmt = Block::make(ifstmt, bl->rest);
    } else if (op->attr_key == "Corver_merge") {
      const auto ift = op->body.as<IfThenElse>();

      Expr condition;
      for (auto elem = record.begin(); elem != record.end(); ++elem) {
        Expr judeg = EQ::make((*elem).loop_var, (*elem).max);
        if (condition.defined()) {
          condition = And::make(condition, judeg);
        } else {
          condition = judeg;
        }
      }

      CHECK(ift);
      stmt = IfThenElse::make(condition, ift->then_case, ift->else_case);
    }

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    int current_level = loop_level;
    loop_level++;

    looprecord t;
    t.loop_var = op->loop_var;
    // min may be complicate expr including div/mod/max ops
    if (op->min.as<IntImm>()) {
      t.min = static_cast<int>(op->min.as<IntImm>()->value);
    } else {
      t.min = 0;
    }
    // extent may be complicate expr including div/mod/max ops
    if (op->extent->IsInstance<IntImm>()) {
      t.max = static_cast<int>(op->extent.as<IntImm>()->value) + t.min - 1;
    } else {
      t.max = t.min + 1;
    }
    record.push_back(t);
    Stmt stmt = IRMutator::Mutate_(op, s);
    record.pop_back();
    if (allocstmt.defined() && current_level == 0) {
      const auto attr = allocstmt.as<AttrStmt>();
      CHECK(attr);
      const auto alloc = attr->body.as<Allocate>();
      CHECK(alloc);
      Stmt allocs = Allocate::make(alloc->buffer_var, alloc->type, alloc->extents, alloc->condition, stmt,
                                   alloc->new_expr, alloc->free_function);
      stmt = AttrStmt::make(attr->node, attr->attr_key, attr->value, allocs);

      allocstmt = Stmt();
    }

    return stmt;
  }

 private:
  struct looprecord {
    Expr loop_var;
    int min{0};
    int max{0};
  };
  std::vector<looprecord> record;
  int loop_level;
  Stmt allocstmt;
};

Stmt TileCoverCorrect(Stmt stmt) {
  stmt = CoverCorrect().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
