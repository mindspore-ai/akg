/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <tvm/ir_mutator.h>
#include <queue>
#include "pass/utils.h"

namespace akg {
namespace ir {
class BufferInfoFinder : public IRMutator {
 public:
  BufferInfoFinder() = default;
  ~BufferInfoFinder() override = default;

  std::unordered_map<const Variable *, std::unordered_set<const Variable *>> buff_indices_;

 private:
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    CollectVarsInfo(op->index, op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Load *op, const Expr &s) final {
    CollectVarsInfo(op->index, op->buffer_var);
    return IRMutator::Mutate_(op, s);
  }

  void AddVarToBuffIndices(const Var &buff, const std::vector<Var> &vars) {
    auto buff_var = buff.get();
    if (buff_indices_.count(buff_var) == 0) {
      std::unordered_set<const Variable *> tmp_vars;
      for (const auto &var : vars) {
        tmp_vars.insert(var.get());
      }
      buff_indices_[buff_var] = tmp_vars;
      return;
    }

    for (const auto &new_var : vars) {
      buff_indices_[buff_var].insert(new_var.get());
    }
  }

  void CollectVarsInfo(const Expr &e, const Var &buff) {
    std::vector<Var> vars;
    GatherVars(e, &vars);
    AddVarToBuffIndices(buff, vars);
  }
};

class BufferCheckVisitor : public IRVisitor {
 public:
  BufferCheckVisitor() {}
  ~BufferCheckVisitor() override = default;

  bool Run(const Variable *buffer, const Stmt &stmt) {
    if (!buffer) {
      return false;
    }
    buffer_ = buffer;
    found_ = false;
    IRVisitor::Visit(stmt);
    return found_;
  }

 private:
  void Visit_(const Store *op) final {
    if (op->buffer_var.get() == buffer_) {
      found_ = true;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) final {
    if (op->buffer_var.get() == buffer_) {
      found_ = true;
      return;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (!found_) {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const Block *op) final {
    if (!found_) {
      IRVisitor::Visit_(op);
    }
  }

  const Variable *buffer_{nullptr};
  bool found_{false};
};

class SinkAllocateMutator : public IRMutator {
 public:
  explicit SinkAllocateMutator(const BufferInfoFinder *finder) : finder_(finder) {}
  ~SinkAllocateMutator() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::storage_scope && op->value.as<StringImm>() &&
        op->value.as<StringImm>()->value.find("L1") == std::string::npos &&
        op->value.as<StringImm>()->value.find("L0") == std::string::npos) {
      if (const auto allocate = op->body.as<Allocate>()) {
        auto var = allocate->buffer_var.get();
        CHECK(var);
        buff_to_attr_[var] = op;
        auto new_stmt = IRMutator::Mutate_(op, s);
        auto new_attr = new_stmt.as<AttrStmt>();
        CHECK(new_attr);
        auto new_allocate = new_attr->body.as<Allocate>();
        CHECK(new_allocate);
        return new_allocate->body;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    std::unordered_map<const Variable *, const AttrStmt *> located_allocate;
    for (auto it = buff_to_attr_.begin(); it != buff_to_attr_.end();) {
      auto find_buffer_in_first = bcv_.Run(it->first, op->first);
      auto find_buffer_in_rest = bcv_.Run(it->first, op->rest);
      if (find_buffer_in_first && find_buffer_in_rest) {
        located_allocate[it->first] = it->second;
        it = buff_to_attr_.erase(it);
      } else {
        it++;
      }
    }

    if (located_allocate.empty()) {
      return IRMutator::Mutate_(op, s);
    }

    auto new_first = IRMutator::Mutate(op->first);
    auto new_rest = IRMutator::Mutate(op->rest);

    auto new_stmt = Block::make(new_first, new_rest);
    new_stmt = BuildNewStmt(located_allocate, new_stmt);

    return new_stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    std::unordered_map<const Variable *, const AttrStmt *> located_allocate;
    for (auto it = buff_to_attr_.begin(); it != buff_to_attr_.end();) {
      const auto &indices_it = finder_->buff_indices_.find(it->first);

      if (indices_it == finder_->buff_indices_.end()) {
        it++;
        continue;
      }

      if (indices_it->second.count(op->loop_var.get())) {
        located_allocate[it->first] = it->second;
        it = buff_to_attr_.erase(it);
      } else {
        it++;
      }
    }

    auto new_stmt = IRMutator::Mutate_(op, s);
    new_stmt = BuildNewStmt(located_allocate, new_stmt);
    return new_stmt;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    std::unordered_map<const Variable *, const AttrStmt *> located_allocate;
    for (auto it = buff_to_attr_.begin(); it != buff_to_attr_.end();) {
      if (bcv_.Run(it->first, s)) {
        located_allocate[it->first] = it->second;
        it = buff_to_attr_.erase(it);
      } else {
        it++;
      }
    }

    auto new_stmt = IRMutator::Mutate_(op, s);
    return BuildNewStmt(located_allocate, new_stmt);
  }

  Stmt BuildNewStmt(const std::unordered_map<const Variable *, const AttrStmt *> &buff_to_attr, Stmt &stmt) {
    auto new_stmt = stmt;
    for (const auto &it : buff_to_attr) {
      auto ori_attr = it.second;
      auto ori_allocate = ori_attr->body.as<Allocate>();
      CHECK(ori_allocate);
      auto new_allocate =
        Allocate::make(ori_allocate->buffer_var, ori_allocate->type, ori_allocate->extents, ori_allocate->condition,
                       new_stmt, ori_allocate->new_expr, ori_allocate->free_function);
      new_stmt = AttrStmt::make(ori_attr->node, ori_attr->attr_key, ori_attr->value, new_allocate);
    }

    return new_stmt;
  }

  std::unordered_map<const Variable *, const AttrStmt *> buff_to_attr_;
  const BufferInfoFinder *finder_{nullptr};
  BufferCheckVisitor bcv_;
};

/*
 * Sink the allocate to be close to the innermost loop where it is strongly related
 * Example:
 * // attr [input_1_local_UB] storage_scope = "local.UB"
 * allocate input_1_local_UB[xxx]
 * for (cc1, 0, I0) {
 *   for (cc2, 0, I1) {
 *     input_1_local_UB(0, cc2) = xxxx;
 *   }
 * }
 * --->
 * for (cc1, 0, I0) {
 *   // attr [input_1_local_UB] storage_scope = "local.UB"
 *   allocate input_1_local_UB[xxx]
 *   for (cc2, 0, I1) {
 *     input_1_local_UB(0, cc2) = xxxx;
 *   }
 * }
 */
Stmt SinkAllocate(const Stmt &stmt) {
  auto buff_info_finder = BufferInfoFinder();
  auto new_stmt = buff_info_finder.Mutate(stmt);
  new_stmt = SinkAllocateMutator(&buff_info_finder).Mutate(new_stmt);
  return new_stmt;
}
}  // namespace ir
}  // namespace akg
