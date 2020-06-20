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
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <pass/utils.h>

namespace akg {
namespace ir {
class StorageOffsetFinder : public IRVisitor {
 public:
  void Visit_(const Allocate *op) override {
    if (op->new_expr.defined()) {
      offset_[op->buffer_var.get()] = op->new_expr;
    }
    IRVisitor::Visit_(op);
  }
  std::unordered_map<const Variable *, Expr> offset_;
};

class StorageOffsetApply : public IRMutator {
 public:
  explicit StorageOffsetApply(std::unordered_map<const Variable *, Expr> &offset) : offset_(offset) {}
  ~StorageOffsetApply() override = default;

  Stmt Mutate_(const Allocate *op, const Stmt &s) override {
    auto alloc = op;
    auto it = offset_.find(op->buffer_var.get());
    if (it != offset_.end()) {
      Expr offset = it->second;
      offset_.erase(it);
      Stmt stmt;
      if (!offset_.empty()) {
        stmt = IRMutator::Mutate_(op, s);
        alloc = stmt.as<Allocate>();
      }
      CHECK(alloc);
      Expr address = Simplify(alloc->new_expr + offset);
      return Allocate::make(alloc->buffer_var, alloc->type, alloc->extents, alloc->condition, alloc->body, address,
                            "nop");
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<const Variable *, Expr> &offset_;
};

Stmt LowerStorageAccessInfoCCE(Stmt stmt) {
  StorageOffsetFinder finder;
  finder.Visit(stmt);
  stmt = ktvm::ir::LowerStorageAccessInfo(stmt);
  return StorageOffsetApply(finder.offset_).Mutate(stmt);
}
}  // namespace ir
}  // namespace akg
