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
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/packed_func_ext.h>
#include <tvm/target_info.h>
#include <pass/storage_access.h>
#include "pass/common.h"

namespace akg {
namespace ir {
// check if we have UB L1 L0
class HasLocalScope : public IRVisitor {
 public:
  HasLocalScope() {}
  ~HasLocalScope() override = default;

  bool HasLocal(const Stmt stmt) {
    this->Visit(stmt);
    return storage_scope_.size() > 0;
  }

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      auto scop = op->value.as<StringImm>()->value;
      ktvm::MemoryInfo info = ktvm::GetMemoryInfo(scop);
      if (info.defined()) {
        storage_scope_[buf] = StorageScope::make(scop);
      }
    }
    IRVisitor::Visit_(op);
  }

 private:
  // The storage scope of each buffer
  std::unordered_map<const Variable *, StorageScope> storage_scope_;
};

// auto inject pipe for target
using ktvm::ir::attr::coproc_scope;
using ktvm::runtime::PackedFunc;

class LoadMatcher : public IRVisitor {
 public:
  LoadMatcher() {}
  ~LoadMatcher() override = default;

  void Visit_(const Load *op) override {
    is_match_ = true;
    IRVisitor::Visit_(op);
  }

  bool isFind(const Expr &e) {
    this->Visit(e);
    return is_match_;
  }
  bool isFind(const Stmt &s) {
    this->Visit(s);
    return is_match_;
  }

 private:
  bool is_match_{false};
};

class InjectPip : public IRMutator {
 public:
  // Get the info by InjectAccessPtrMSG pass
  struct AccessPtrMSG {
    const Variable *dest;
    std::set<const Variable *> src;
    int RepeatTime;
    int RepeatStride;
    int BlockNumber;
    int BlockStride;
    int BlockSize;
    Expr ExtentUnit;
    Expr offset;
  };

  Stmt Inject(Stmt stmt) {
    if (!HasLocalScope().HasLocal(stmt)) return stmt;
    return this->Mutate(stmt);
  }

  // don't inject for insn which already has scope
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    if (op->attr_key == coproc_scope) {
      coproc_found = true;
      Stmt stmt = IRMutator::Mutate_(op, s);
      coproc_found = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    // if found only one insn in body, insert scope
    int pip = GetPipID(op->body);
    if (pip > 0 && !coproc_found && !IsLoopDep(op)) {
      Stmt stmt = AttrStmt::make(IntImm::make(Int(32), 0), coproc_scope, IntImm::make(Int(32), pip), s);

      // For Load
      if (pip != PIPE_S && LoadMatcher().isFind(op->body)) {
        stmt = AttrStmt::make(IntImm::make(Int(32), 0), coproc_scope, IntImm::make(Int(32), PIPE_S), stmt);
      }
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    const Call *insn = op->value.as<Call>();
    if (insn != nullptr && !coproc_found) {
      int pip = GetIntrinPipe(insn->name);
      if (pip > 0) {
        Stmt stmt = AttrStmt::make(IntImm::make(Int(32), 0), coproc_scope, IntImm::make(Int(32), pip), s);

        // For args
        if (pip != PIPE_S && LoadMatcher().isFind(op->value)) {
          stmt = AttrStmt::make(IntImm::make(Int(32), 0), coproc_scope, IntImm::make(Int(32), PIPE_S), stmt);
        }
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    Stmt stmt = AttrStmt::make(IntImm::make(Int(32), 0), coproc_scope, IntImm::make(Int(32), PIPE_S), s);
    return stmt;
  }

  bool IsLoopDep(const For *op) {
    // There are two cases for now when the coproc scope should be moved inside the for loop
    // E.g.
    //      // attr [iter_var(cce, , cce)] coproc_scope = 4
    //      for (...)
    // ==>
    //      for (...)
    //      // attr [iter_var(cce, , cce)] coproc_scope = 4
    AccessPtrMSG access_info = GetAccessInfo(op->body);
    // First check the info is correct
    if (access_info.RepeatTime == -1) {
      return false;
    }

    // Case 1: tvm_access_ptr inside for loop has same dst and src address
    // i.e. this statement is depend on itself (next loop's read must be done after previous loop's write)
    if (access_info.src.count(access_info.dest) != 0) {
      return true;
    }

    // Case 2: tvm_access_ptr inside for loop has write address overlap
    Expr extent = CalcRealExtent(access_info);
    std::unordered_map<const Variable *, Expr> next_loop_map;
    next_loop_map[op->loop_var.get()] = op->loop_var + 1;
    Map<Var, Range> range;
    range.Set(Var(op->loop_var), Range::make_by_min_extent(op->min, op->extent));
    ktvm::arith::Analyzer analyzer_;
    Expr t = Simplify_cce(ktvm::ir::Substitute(access_info.offset, next_loop_map) - access_info.offset < extent, range);
    return analyzer_.CanProve(t);
  }

  AccessPtrMSG GetAccessInfo(const Stmt &body) const {
    AccessPtrMSG access_info;
    bool is_init = false;
    auto GetInfo = [&access_info, &is_init](const NodeRef &op) {
      const auto tvm_access_ptr = op.as<Call>();
      if (tvm_access_ptr != nullptr && tvm_access_ptr->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr)) {
        CHECK_EQ(tvm_access_ptr->args.size(), 10U);
        int rw = tvm_access_ptr->args[4].as<IntImm>()->value;
        const auto op_address = tvm_access_ptr->args[1].as<Variable>();
        Expr offset = tvm_access_ptr->args[2];
        if ((uint32_t)rw & 1) access_info.src.insert(op_address);
        if ((uint32_t)rw & 2) {
          access_info.dest = op_address;
          access_info.offset = offset;
          auto *rt = tvm_access_ptr->args[5].as<IntImm>();
          auto *rs = tvm_access_ptr->args[6].as<IntImm>();
          auto *bn = tvm_access_ptr->args[7].as<IntImm>();
          auto *bst = tvm_access_ptr->args[8].as<IntImm>();
          auto *bsz = tvm_access_ptr->args[9].as<IntImm>();
          Expr ext_unit = make_const(tvm_access_ptr->args[2].type(), tvm_access_ptr->args[0].type().bits());
          bool AllInfoNotNull = (rt && rs && bn && bst && bsz);
          if (AllInfoNotNull) {
            access_info.RepeatTime = static_cast<int>(rt->value);
            access_info.RepeatStride = static_cast<int>(rs->value);
            access_info.BlockNumber = static_cast<int>(bn->value);
            access_info.BlockStride = static_cast<int>(bst->value);
            access_info.BlockSize = static_cast<int>(bsz->value);
            access_info.ExtentUnit = ext_unit;
            is_init = true;
          }
        }
      }
    };
    ktvm::ir::PostOrderVisit(body, GetInfo);
    if (!is_init) {
      access_info.RepeatTime = -1;  // generate error code to skip moving coproc
    }
    return access_info;
  }

  Expr CalcRealExtent(const AccessPtrMSG &msg) {
    if (msg.RepeatTime == -1 || msg.BlockStride == 0 || msg.RepeatStride == 0 || msg.BlockSize == 0) return Expr(-1);
    int one_repeat;
    if (msg.BlockStride % msg.BlockSize == 0) {
      one_repeat = msg.BlockSize * (msg.BlockStride / msg.BlockSize) * msg.BlockNumber;
    } else {  // adapt to vcg op, which simply returns stride as Arch::Vector::BLOCKS_PER_REPEAT
      one_repeat = msg.BlockSize * msg.BlockNumber * msg.BlockStride;
    }
    int real_ext = one_repeat * (msg.RepeatStride / one_repeat) * msg.RepeatTime;
    const auto eu = msg.ExtentUnit.as<IntImm>();
    CHECK(eu != nullptr && eu->value > 0);
    real_ext = real_ext / eu->value;  // change from Byte to unit byte
    return Expr(real_ext);
  }

 private:
  bool coproc_found{false};

  // the output of vcadd is 32B overlapped between loop, there must sync.
  // now we just force it inside loop to let bar.v inserted.
  std::unordered_set<std::string> force_inside_insn{"vcadd"};

  // find single insn in loop body
  int GetPipID(const Stmt &body) {
    int insn_num = 0;
    int pip = -1;
    auto GetPip = [this, &insn_num, &pip](const NodeRef &op) {
      const auto eval = op.as<Evaluate>();
      if (eval != nullptr) {
        const Call *insn = eval->value.as<Call>();
        if (insn != nullptr) {
          int pip_ = GetIntrinPipe(insn->name);
          if (pip_ > 0) {
            insn_num += 1;
            if (this->force_inside_insn.find(insn->name) != this->force_inside_insn.end()) {
              insn_num += 1;
            }
            pip = pip_;
          }
        }
      }
      const auto store = op.as<Store>();
      if (store != nullptr) {
        insn_num += 1;
        pip = PIPE_S;
      }
      const auto sync = op.as<AttrStmt>();
      if (sync != nullptr && sync->attr_key == coproc_scope) {
        // we have coproc already
        insn_num += 1;
      }
    };
    ktvm::ir::PostOrderVisit(body, GetPip);
    return insn_num == 1 ? pip : -1;
  }
};

// SingleCoprocForm: convert coproc Node to be unique
// currently, coproc node tree must exist Evaluate or Store Node.
// We recreate any Evaluate or Store Node to force coproc node tree recreate
class SingleCoprocForm : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    if (op->attr_key == coproc_scope) {
      CHECK(update_ == false);
      const Node *node = op->body.get();
      if (exist_nodes_.count(node) > 0) {
        update_ = true;
      } else {
        exist_nodes_.insert(node);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    if (update_) {
      update_ = false;
      return Evaluate::make(op->value);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (nullptr == op) {
      return s;
    }
    if (update_) {
      update_ = false;
      return Store::make(op->buffer_var, op->value, op->index, op->predicate);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  // exist coproc node have be visited
  std::unordered_set<const Node *> exist_nodes_;
  // flag to indicate if current coproc tree need recreate
  bool update_{false};
};

Stmt ConvertSingleCoprocForm(const Stmt stmt) { return SingleCoprocForm().Mutate(stmt); }

Stmt InjectPipe(Stmt stmt) {
  stmt = InjectPip().Inject(stmt);
  stmt = ConvertSingleCoprocForm(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
