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
#include "pass/analyze_align.h"
#include <cmath>
#include <map>
#include <vector>
#include <utility>
#include "pass/ir_util.h"
#include "ir_pass.h"
#include "emit_insn/cce_params.h"
#include "emit_insn/insn_info.h"
#include "emit_insn/insn_pattern.h"

namespace akg {
namespace ir {

int GetCommonDivisor(std::vector<int> numbers) {
  CHECK(numbers.size() >= 1);
  int divisor = numbers[0];
  for (size_t i = 1; i < numbers.size(); i++) {
    divisor = air::ir::gcd(divisor, numbers[i]);
  }
  return divisor;
}

namespace {

using Var2Scope = std::map<const Variable *, std::string>;

bool IsInStorageScope(const Var2Scope &table, const Variable *var) { return table.find(var) != table.end(); }

class FindSameNameBuf : public IRVisitor {
 public:
  FindSameNameBuf() = default;
  ~FindSameNameBuf() override = default;

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == air::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      CHECK(buf != nullptr);
      auto str = op->value.as<StringImm>();
      CHECK(str != nullptr);
      storage_scope_[buf] = str->value;
    }
    IRVisitor::Visit(op->body);
  }

  Var2Scope storage_scope_;
};

class InsertIsolate : public IRMutator {
 public:
  explicit InsertIsolate(const Var2Scope &table) : storage_scope_(table), first_with_bb_(0), insert_isolate_(false) {}
  ~InsertIsolate() override = default;

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    Stmt stmt = op->first;
    bool has_block = HasBlock(stmt);
    if (has_block) {
      insert_isolate_ = false;
      stmt = this->Mutate(op->first);
      if (HasOutput(stmt)) {
        first_with_bb_ = 0;
      }
      if (!insert_isolate_) {
        ++first_with_bb_;
      }
    } else {
      ++first_with_bb_;
    }

    CHECK(op->rest.defined());
    bool single_bb = first_with_bb_ == 1;
    Stmt rest = this->Mutate(op->rest);
    bool rest_hasout = HasOutput(rest);
    stmt = Block::make(stmt, rest);
    if (!has_block && single_bb && rest_hasout) {
      stmt = AttrStmt::make(make_zero(Int(32)), "isolate_range", 2, stmt);
      insert_isolate_ = true;
    }

    if (!has_block && first_with_bb_ > 0) {
      --first_with_bb_;
    }
    return stmt;
  }

 private:
  bool HasOutput(const Stmt &s) const {
    bool found_out = false;

    auto CheckOutput = [&found_out, this](const NodeRef &op) {
      const auto st = op.as<Store>();
      // A = A_ub
      if (st != nullptr && !IsInStorageScope(this->storage_scope_, st->buffer_var.get())) {
        found_out = true;
      }
    };
    PostOrderVisit(s, CheckOutput);
    return found_out;
  }

  bool HasBlock(const Stmt &s) const {
    bool found_block = false;

    auto CheckBlock = [&found_block](const NodeRef &op) {
      if (op.as<Block>() != nullptr) {
        found_block = true;
      }
    };
    PostOrderVisit(s, CheckBlock);
    return found_block;
  }

  const Var2Scope &storage_scope_;
  int first_with_bb_;
  bool insert_isolate_;
};

class CacheVisiter : public IRVisitor {
 public:
  CacheVisiter() = default;
  ~CacheVisiter() override = default;

  void Visit_(const Allocate *op) final {
    var_type_map[op->buffer_var.get()] = op->type;
    IRVisitor::Visit_(op);
  }

  std::unordered_map<const Variable *, Type> var_type_map;
};

// process each isolate_range once a time
class ProcessParts : public IRMutator {
 public:
  explicit ProcessParts(const Var2Scope &table) : level_(0), storage_scope_(table) {}
  ~ProcessParts() override = default;

  std::unordered_map<const Variable *, Type> var_type_map;

  Stmt Run(Stmt stmt) {
    CacheVisiter buffer_visitor;
    buffer_visitor.Visit(stmt);
    var_type_map = buffer_visitor.var_type_map;

    stmt = this->Mutate(stmt);
    if (level_ == 0) {
      stmt = AlignGen().Run(stmt, var_type_map);
    }
    return stmt;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (!HasIsolate(s)) {
      Stmt stmt = s;
      stmt = AlignGen().Run(stmt, var_type_map);
      level_++;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "isolate_range") {
      level_++;
      int cur_level = level_;
      Stmt stmt = IRMutator::Mutate_(op, s);
      // no isolate_range in this attr
      if (cur_level == level_) {
        stmt = AlignGen().Run(stmt, var_type_map);
      }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool HasIsolate(const Stmt &s) const {
    bool found_isolate = false;
    auto CheckIsolate = [&found_isolate](const NodeRef &op) {
      const auto attr = op.as<AttrStmt>();
      if (attr && attr->attr_key == "isolate_range") {
        found_isolate = true;
      }
    };
    PostOrderVisit(s, CheckIsolate);
    return found_isolate;
  }

  int level_;
  const Var2Scope &storage_scope_;
};
}  // namespace

Stmt AnalyzeMinAlignStatic(Stmt stmt) {
  stmt = air::ir::ConvertSSA(stmt);

  CacheVisiter buffer_visitor;
  buffer_visitor.Visit(stmt);

  FindSameNameBuf find_visitor;
  find_visitor.Visit(stmt);

  stmt = InsertIsolate(find_visitor.storage_scope_).Mutate(stmt);
  stmt = ProcessParts(find_visitor.storage_scope_).Run(stmt);
  stmt = RewriteByAlignStatic(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
