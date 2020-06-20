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
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include <ir_pass.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/packed_func_ext.h>
#include <tvm/target_info.h>
#include <fstream>
#include "pass/common.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
using ktvm::ir::intrinsic::tvm_access_ptr;

// 1, find control flow loops
// 2, find location of set matrix,
// 3, find the place insn should insert
// 4, move code
// hoist intrinsic out of invariant loops
class PreHoist : public IRVisitor {
 public:
  PreHoist() {}
  ~PreHoist() override = default;

  void Visit_(const For *op) final {
    // push and pop loops
    bool can_promote = true;
    Expr last_val;
    auto Scan = [&can_promote, &last_val](const NodeRef &node) {
      const Call *c = node.as<Call>();
      if (c && c->name == "set_fmatrix") {
        if (last_val.defined() && !Equal(last_val, c->args[0])) {
          can_promote = false;
        }
        last_val = c->args[0];
      }
    };

    // for
    //     set 1
    //     set 2
    // can't promote set 1 or set 2 before for
    ktvm::ir::PostOrderVisit(op->body, Scan);
    if (can_promote) {
      deq_outer_loops_.push_front(op);
    }

    IRVisitor::Visit_(op);
    if (can_promote) {
      deq_outer_loops_.pop_front();
      // after hoist set-1 before for(i), we need clear last_set_matrix_
      // after visit of for(i) Node to enable hoist of set-2.
      // for(i)
      //   set-1
      // for(j)
      //   for (k)
      //     set-2
      if (last_insert_point_ == op) {
        last_set_matrix_ = nullptr;
      }
    }
  }

  void Visit_(const Evaluate *op) final {
    const Call *insn = op->value.as<Call>();
    if (insn != nullptr && insn->name == "set_fmatrix") {
      std::vector<const For *> outer_loops;
      for (size_t i = 0; i < deq_outer_loops_.size(); ++i) {
        outer_loops.push_back(deq_outer_loops_[i]);
      }

      Stmt stmt = GetRef<Stmt>(op);
      const Node *insert_point = nullptr;
      int cur_use_level = deq_outer_loops_.size();
      // if different value, can not hoist more than last_set_matrix_
      // make sure not cross last use if value is different, and cur depth <= last
      if (last_set_matrix_ == nullptr || Equal(last_set_matrix_->value, op->value) || cur_use_level < last_use_level_) {
        for (size_t i = 0; i < outer_loops.size(); ++i) {
          // no such var appear in set_matrix,
          // loop already have a value, diff with current, stop
          bool can_insert = true;
          if (before_.count(outer_loops[i]) > 0) {
            can_insert = Equal(before_[outer_loops[i]], stmt);
          }

          if (!ExprUseVar(op->value, Var(outer_loops[i]->loop_var)) && can_insert) {
            insert_point = outer_loops[i];
          } else {
            break;
          }
        }
      }

      // can hoist, replace with NOP; and record the place to insert this insn
      // if has been in the hoist_ map, stop insert
      if (insert_point != nullptr) {
        if (before_.count(insert_point) == 0) {
          before_[insert_point] = stmt;
        }
        remove_.insert(op);
        last_insert_point_ = insert_point;
      }
      last_set_matrix_ = op;
      last_use_level_ = outer_loops.size();
    }
    IRVisitor::Visit_(op);
  }

  // less learned here, if we change ir, the addr of For may changed
  // so we don't touch the original ir, just analyze in this class
  std::unordered_map<const Node *, Stmt> before_;
  std::unordered_set<const Node *> remove_;

 private:
  // outer loops for codegen
  std::deque<const For *> deq_outer_loops_;
  const Evaluate *last_set_matrix_{nullptr};
  int last_use_level_{0};
  const Node *last_insert_point_ = nullptr;
};

class Hoist : public IRMutator {
 public:
  Hoist(std::unordered_map<const Node *, Stmt> &before, std::unordered_set<const Node *> &remove)
      : before_(before), remove_(remove) {}
  ~Hoist() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (before_.count(op)) {
      stmt = Block::make(before_[op], stmt);
    }
    return stmt;
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    if (remove_.count(op)) {
      return Evaluate::make(0);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::unordered_map<const Node *, Stmt> &before_;
  std::unordered_set<const Node *> &remove_;
};

class ElimRptDef : public IRMutator {
 public:
  ElimRptDef() {}
  ~ElimRptDef() override = default;

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    const Call *insn = op->value.as<Call>();
    if (insn && (insn->name == "set_fmatrix")) {
      if (base_set_matrix_ == nullptr) {
        // first meet set_fmatrix
        base_set_matrix_ = op;
      } else if (Equal(base_set_matrix_->value, op->value)) {
        // compare the value, if the same with base_set_matrix_, remove it
        // if not, update base_set_matrix_
        return Evaluate::make(0);
      } else {
        base_set_matrix_ = op;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  const Evaluate *base_set_matrix_{nullptr};
};

class StorageHoistVerify {
 public:
  using StmtEntry = LivenessAnalyzer::StmtEntry;
  struct StorageEntry {
    std::string scope;
    int64_t size;
    int gen;
    int kill;
  };

  void Prepare(const Stmt stmt) {
    LivenessAnalyzer analyzer;
    analyzer.Analyze(stmt);
    seq_ = std::move(analyzer.liveness_);
    for (auto &v : analyzer.alloc_) {
      auto &info = v.second;
      if (info.touched.empty()) continue;
      auto size = static_cast<int64_t>(info.alloc->constant_allocation_size() * info.alloc->type.bytes() *
                                       info.alloc->type.lanes());

      std::string scope = info.scope.to_string();
      storage_.emplace(v.first, StorageEntry{scope, size, info.touched.front(), info.touched.back()});
      if (mem_live_.find(scope) == mem_live_.end()) {
        mem_live_.emplace(scope, std::vector<int64_t>(seq_.size(), 0));
      }
    }

    std::unordered_map<std::string, int64_t> mem_size;
    for (size_t i = 0; i < seq_.size(); ++i) {
      auto &s = seq_[i];
      for (const Variable *buf : s.gen) {
        StorageEntry &entry = storage_[buf];
        mem_size[entry.scope] += entry.size;
      }

      for (auto it : mem_size) {
        mem_live_[it.first][i] = it.second;
      }

      for (const Variable *buf : s.kill) {
        StorageEntry &entry = storage_[buf];
        mem_size[entry.scope] -= entry.size;
      }
    }
  }

  bool Verify(const Variable *buf, const For *loop) {
    StorageEntry &entry = storage_[buf];
    ktvm::MemoryInfo info = ktvm::GetMemoryInfo(entry.scope);
    if (!info.defined()) {
      return false;
    }

    int64_t mem_limit = info->max_num_bits / 8;
    int loop_start, loop_end;
    std::tie(loop_start, loop_end) = GetLoopIndex(loop);
    std::vector<int64_t> &mem_live = mem_live_[entry.scope];

    for (int i = loop_start; i < entry.gen; ++i) {
      if (mem_live[i] + entry.size > mem_limit) {
        return false;
      }
    }

    for (int i = entry.kill + 1; i <= loop_end; ++i) {
      if (mem_live[i] + entry.size > mem_limit) {
        return false;
      }
    }

    return true;
  }

  void Hoist(const Variable *buf, const For *loop) {
    StorageEntry &entry = storage_[buf];
    int loop_start, loop_end;
    std::tie(loop_start, loop_end) = GetLoopIndex(loop);
    std::vector<int64_t> &mem_live = mem_live_[entry.scope];

    for (int i = loop_start; i < entry.gen; ++i) {
      mem_live[i] += entry.size;
    }

    for (int i = entry.kill + 1; i <= loop_end; ++i) {
      mem_live[i] += entry.size;
    }

    entry.gen = loop_start;
    entry.kill = loop_end;
  }

 private:
  std::pair<int, int> GetLoopIndex(const For *loop) {
    int start = -1;
    int end = -1;
    for (int i = 0; i < static_cast<int>(seq_.size()); ++i) {
      if (seq_[i].stmt != loop) continue;

      if (start == -1) {
        start = i;
      } else {
        end = i;
        break;
      }
    }

    CHECK(start != -1 && end != -1);

    return std::make_pair(start, end);
  }

  std::vector<StmtEntry> seq_;
  std::unordered_map<const Variable *, StorageEntry> storage_;
  std::unordered_map<std::string, std::vector<int64_t>> mem_live_;
};

class InvarHoistVerify : public IRVisitor {
 public:
  InvarHoistVerify(const For *loop, const std::vector<NodeRef> &hoisted) : loop_(loop) {
    for (const NodeRef &ref : hoisted) {
      hoisted_.insert(ref.get());
    }
  }
  ~InvarHoistVerify() override = default;

  bool Verify(const NodeRef &node) {
    auto FindTouch = [this](const NodeRef &op) {
      if (const auto var = op.as<Variable>()) {
        this->touched_.insert(var);
      } else if (const auto load = op.as<Load>()) {
        this->touched_.insert(load->buffer_var.get());
      } else if (const auto store = op.as<Store>()) {
        this->touched_.insert(store->buffer_var.get());
      }
    };

    node_ = node;
    ktvm::ir::PostOrderVisit(node, FindTouch);
    this->Visit(loop_->body);

    return !defined_;
  }

  void Visit(const NodeRef &node) override {
    if ((node != node_) && !hoisted_.count(node.get()) && !defined_) {
      IRVisitor::Visit(node);
    }
  }

  void Visit_(const Call *op) final {
    if (op->is_intrinsic(tvm_access_ptr)) {
      CHECK(op->args.size() > 4 && op->args[4].defined()) << " invalid tvm_access_ptr! ";
      const auto rw = op->args[4].as<IntImm>();
      const auto buffer = op->args[1].as<Variable>();
      if (rw != nullptr && buffer != nullptr && (static_cast<uint64_t>(rw->value) & 2) &&
          (touched_.count(buffer) > 0)) {
        defined_ = true;
      }
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    if (touched_.count(op->buffer_var.get())) {
      defined_ = true;
    }

    IRVisitor::Visit_(op);
  }

 private:
  const For *loop_{nullptr};
  bool defined_{false};
  NodeRef node_;
  std::unordered_set<const Variable *> touched_;
  std::unordered_set<const Node *> hoisted_;
};

class InvarHoistPlan : public IRVisitor {
 public:
  struct HoistEntry {
    std::vector<NodeRef> nodes;
    std::vector<const Variable *> allocs;
  };

  InvarHoistPlan() : cur_scope_(nullptr) {}
  ~InvarHoistPlan() override = default;

  void Plan(const Stmt stmt) {
    storage_verifier_.Prepare(stmt);
    this->Visit(stmt);
  }

  void Visit(const NodeRef &node) override {
    bool is_hoist_node =
      node->IsInstance<ProducerConsumer>() || node->IsInstance<Evaluate>() || node->IsInstance<For>();
    if (replay_ || !is_hoist_node || !Hoist(node)) {
      IRVisitor::Visit(node);
    }
  }

  void Visit_(const For *op) override {
    if (replay_) {
      IRVisitor::Visit_(op);
      return;
    }

    LoopScope scope(op);
    std::swap(scope, cur_scope_);
    cur_scope_.touched.insert(op->loop_var.get());
    IRVisitor::Visit_(op);
    std::swap(cur_scope_, scope);
    if (!scope.hoist.nodes.empty()) {
      Sumbit(scope);
    }

    // update touch info for parent LoopScope, in replay node:
    // 1. do not add alloc buffer
    // 2. do not hoist
    replay_ = true;
    IRVisitor::Visit_(op);
    replay_ = false;
  }

  void Visit_(const AttrStmt *op) override {
    if (!replay_ && op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto var = op->node.as<Variable>();
      cur_scope_.allocs[var] = op;
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Variable *op) override {
    TouchVar(op);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) override {
    TouchVar(op->buffer_var.get());
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) override {
    TouchVar(op->buffer_var.get(), true);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) override {
    if (op->name == "set_fmatrix") {
      cur_scope_.fmatrix_touch = true;
    } else if (op->name.compare(0, 8, "img2col_") == 0) {
      cur_scope_.img2col_touch = true;
    } else if (op->is_intrinsic(tvm_access_ptr)) {
      const auto buf = op->args[1].as<Variable>();
      TouchVar(buf, true);
    }
    IRVisitor::Visit_(op);
  }

  std::unordered_map<const For *, HoistEntry> hoist_info_;

 private:
  struct LoopScope {
    explicit LoopScope(const For *n) : node(n) {}
    // loop node
    const For *node;
    // alloc var with storage_scope attr under this loop
    std::unordered_map<const Variable *, const AttrStmt *> allocs;
    // var touched of this loop. including loop_var and touched alloc var
    std::unordered_set<const Variable *> touched;
    // hoist entry
    HoistEntry hoist;
    // fmatrix have touched
    bool fmatrix_touch{false};
    bool img2col_touch{false};
  };

  bool Hoist(const NodeRef &node) {
    std::unordered_set<const Variable *> alloc_touch;
    return Hoist(node, alloc_touch);
  }

  bool Hoist(const NodeRef &node, std::unordered_set<const Variable *> &alloc_touch) {
    if (cur_scope_.node == nullptr || is_no_op(Downcast<Stmt>(node))) {
      return false;
    }

    bool invariant = true;
    auto Scan = [this, &alloc_touch, &invariant](const NodeRef &op) {
      if (invariant) {
        const Variable *var = nullptr;
        if (const auto load = op.as<Load>()) {
          var = load->buffer_var.get();
        } else if (const auto store = op.as<Store>()) {
          var = store->buffer_var.get();
        } else {
          var = op.as<Variable>();
        }

        if (var != nullptr) {
          if (this->cur_scope_.touched.count(var)) {
            invariant = false;
          } else if (this->cur_scope_.allocs.find(var) != this->cur_scope_.allocs.end()) {
            alloc_touch.insert(var);
          }
        } else if (const Call *call = op.as<Call>()) {
          if (this->cur_scope_.fmatrix_touch && call->name.compare(0, 8, "img2col_") == 0) {
            invariant = false;
          } else if (call->name == "set_fmatrix") {
            invariant = false;
          } else if (call->name == "set_vector_mask") {
            invariant = false;
          } else if (call->name == "vadd" || call->name == "vsub" || call->name == "vmul") {
            CHECK(call->args[0].as<Call>());
            CHECK(call->args[1].as<Call>());
            CHECK(call->args[2].as<Call>());
            Expr call_des = call->args[0].as<Call>()->args[1];
            Expr call_src0 = call->args[1].as<Call>()->args[1];
            Expr call_src1 = call->args[2].as<Call>()->args[1];
            if (call_des.same_as(call_src0) || call_des.same_as(call_src1)) invariant = false;
          } else if (call->name == "vadds" || call->name == "vmuls") {
            CHECK(call->args[0].as<Call>());
            CHECK(call->args[1].as<Call>());
            Expr call_des = call->args[0].as<Call>()->args[1];
            Expr call_src0 = call->args[1].as<Call>()->args[1];
            if (call_des.same_as(call_src0)) invariant = false;
          } else if (call->name == "set_rpn_cor_ir" || call->name == "scatter_vnchwconv_b16" ||
                     call->name == "set_atomic_add_open" || call->name == "set_atomic_add_close" ||
                     call->name == "vsel" || call->name.compare(0, 5, "vcmp_") == 0) {
            invariant = false;
          }
        }
      }
    };

    ktvm::ir::PostOrderVisit(node, Scan);
    if (invariant) {
      bool can_hoist = InvarHoistVerify(cur_scope_.node, cur_scope_.hoist.nodes).Verify(node);
      if (!can_hoist) {
        return false;
      }

      if (!alloc_touch.empty()) {
        for (const Variable *a : alloc_touch) {
          if (!storage_verifier_.Verify(a, cur_scope_.node)) {
            return false;
          }
        }

        for (const Variable *a : alloc_touch) {
          storage_verifier_.Hoist(a, cur_scope_.node);
          cur_scope_.hoist.allocs.emplace_back(a);
        }
      }

      cur_scope_.hoist.nodes.emplace_back(node);
    }

    return invariant;
  }

  void Sumbit(LoopScope &scope) {
    std::vector<NodeRef> stay_nodes;
    std::unordered_set<const Variable *> stay_allocs(scope.hoist.allocs.begin(), scope.hoist.allocs.end());

    for (NodeRef &ref : scope.hoist.nodes) {
      std::unordered_set<const Variable *> touched;
      auto Scan = [&stay_allocs, &touched](const NodeRef &op) {
        const Variable *var = nullptr;

        if (const auto load = op.as<Load>()) {
          var = load->buffer_var.get();
        } else if (const auto store = op.as<Store>()) {
          var = store->buffer_var.get();
        } else {
          var = op.as<Variable>();
        }

        if (var && stay_allocs.count(var)) {
          touched.insert(var);
        }
      };

      ktvm::ir::PostOrderVisit(ref, Scan);
      if (Hoist(ref, touched)) {
        for (const Variable *var : touched) {
          stay_allocs.erase(var);
        }
      } else {
        for (const Variable *var : touched) {
          cur_scope_.touched.insert(var);
        }
        stay_nodes.emplace_back(ref);
      }
    }

    // this check is to guarantee: if hoisted allocs is not empty, hoisted nodes must be non-empty too
    CHECK(!stay_nodes.empty() || stay_allocs.empty());
    if (!stay_nodes.empty()) {
      HoistEntry &entry = hoist_info_[scope.node];
      entry.nodes.swap(stay_nodes);
      entry.allocs = std::move(std::vector<const Variable *>(stay_allocs.begin(), stay_allocs.end()));
    }
  }

  void TouchVar(const Variable *var, bool def = false) {
    // the touch buffer may alloc in current LoopScope or upper LoopScope.
    // If in current LoopScope, we just touch it because there must a def after alloc,
    // If in upper LoopScope, we only touch it if it is def in current LoopScope.
    if (def || (cur_scope_.allocs.find(var) != cur_scope_.allocs.end())) {
      cur_scope_.touched.insert(var);
    }
  }

  LoopScope cur_scope_;
  // replay of for body, only for collect touch info for parent scope
  bool replay_{false};
  // verify if alloc scope can hoist
  StorageHoistVerify storage_verifier_;
};

// gather vmask state of each simd and for node.
// we will insert vmask stmt if the hoisted node has vector instruction
// to keep its "vmask state" not changed.
// the leaving redundant vmask stmts will be eliminated by following pass.
class VMaskGather : public IRVisitor {
 public:
  VMaskGather() {
    Expr ff = UIntImm::make(UInt(64), 0xffffffffffffffffL);
    cur_vmask_ = Call::make(Int(32), "set_vector_mask", {ff, ff}, Call::Extern);
  }
  ~VMaskGather() override = default;

  void Visit_(const For *op) final {
    for_vmask_[op] = cur_vmask_;
    IRVisitor::Visit_(op);
  }

  void Visit_(const Evaluate *op) final {
    if (const Call *call = op->value.as<Call>()) {
      if (call->name == "set_vector_mask") {
        cur_vmask_ = op->value;
      } else {
        int pipe = GetIntrinPipe(call->name);
        if (pipe == 2) {
          simd_vmask_[call] = cur_vmask_;
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse *op) final {
    Expr else_vmask = cur_vmask_;
    Visit(op->then_case);
    Expr then_vmask = cur_vmask_;

    if (op->else_case.defined()) {
      Visit(op->else_case);
      else_vmask = cur_vmask_;
    }

    bool equal = IsEqual(then_vmask, else_vmask);
    if (!equal) {
      LOG(WARNING) << "inconsistent vmask state from different branch paths";
    }
  }

  std::unordered_map<const Call *, Expr> simd_vmask_;
  std::unordered_map<const For *, Expr> for_vmask_;

 private:
  bool IsEqual(const Expr &ea, const Expr &eb) {
    CHECK(ea.as<Call>());
    CHECK(eb.as<Call>());

    const Array<Expr> &a = ea.as<Call>()->args;
    const Array<Expr> &b = eb.as<Call>()->args;
    CHECK(a.defined());
    CHECK(b.defined());

    if (a.size() != b.size()) {
      return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
      if (!Equal(a[i], b[i])) {
        return false;
      }
    }

    return true;
  }

  Expr cur_vmask_;
};

class InvarHoist : public IRMutator {
 public:
  using HoistEntry = InvarHoistPlan::HoistEntry;

  explicit InvarHoist(std::unordered_map<const For *, HoistEntry> &hoist_info) : hoist_info_(hoist_info) {
    for (auto it : hoist_info_) {
      for (NodeRef &ref : it.second.nodes) {
        nodes_[ref.get()] = Stmt();
      }

      for (const Variable *a : it.second.allocs) {
        allocs_[a] = std::make_pair(Stmt(), Stmt());
      }
    }
  }
  ~InvarHoist() override = default;

  Stmt Hoist(const Stmt stmt) {
    vmask_.Visit(stmt);
    Stmt res = Mutate(stmt);

    return AddPadInit(res);
  }

  Stmt AddPadInit(const Stmt &stmt) {
    Stmt res = stmt;
    if (addPadInit_ && initCall_ != nullptr) {
      Expr pad16_value = Cast::make(UInt(64), 0x0000);
      Stmt padding = Evaluate::make(Call::make(initCall_->type, "set_padding", {pad16_value}, Call::Extern));
      std::vector<Stmt> alls{res, padding};
      res = Block::make(alls);
    }
    return res;
  }

  Stmt Mutate(Stmt stmt) final {
    const Node *n = stmt.get();
    stmt = IRMutator::Mutate(stmt);
    auto it = nodes_.find(n);
    if (it != nodes_.end()) {
      it->second = stmt;
      return Evaluate::make(0);
    }
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    auto it = hoist_info_.find(op);
    if (it != hoist_info_.end()) {
      HoistEntry &entry = it->second;
      bool vmask_set = false;
      auto it2 = entry.nodes.rbegin();
      Stmt hoist_stmt = nodes_[it2->get()];
      Expr vmask = GetVmask(hoist_stmt);
      if (vmask.defined()) {
        vmask_set = true;
        hoist_stmt = Block::make(Evaluate::make(vmask), hoist_stmt);
      }

      while ((++it2) != entry.nodes.rend()) {
        auto &node_stmt = nodes_[it2->get()];
        hoist_stmt = Block::make(node_stmt, hoist_stmt);
        vmask = GetVmask(node_stmt);
        if (vmask.defined()) {
          vmask_set = true;
          hoist_stmt = Block::make(Evaluate::make(vmask), hoist_stmt);
        }
      }

      if (vmask_set) {
        vmask = vmask_.for_vmask_[op];
        CHECK(vmask.defined());
        stmt = Block::make(Evaluate::make(vmask), stmt);
      }
      stmt = Block::make(hoist_stmt, stmt);

      if (!entry.allocs.empty()) {
        std::vector<Stmt> alloc;
        for (auto it3 = entry.allocs.rbegin(); it3 != entry.allocs.rend(); ++it3) {
          const Variable *n = *it3;
          alloc.emplace_back(allocs_[n].first);
          alloc.emplace_back(allocs_[n].second);
        }

        stmt = ktvm::ir::MergeNest(alloc, stmt);
      }
    }

    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      auto it = allocs_.find(buf);
      if (it != allocs_.end()) {
        it->second.first = AttrStmt::make(op->node, op->attr_key, op->value, Evaluate::make(0));
        return Mutate(op->body);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    auto it = allocs_.find(op->buffer_var.get());
    if (it != allocs_.end()) {
      it->second.second = Allocate::make(op->buffer_var, op->type, op->extents, op->condition, Evaluate::make(0),
                                         op->new_expr, op->free_function);
      return Mutate(op->body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "img2col_cbuf_to_ub") {
      addPadInit_ = true;
      initCall_ = op;
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  Expr GetVmask(const Stmt stmt) {
    // because set_vector_mask is not hoistable, the stmt has unique vmask,
    // so we just get the first vmask.
    // the call node is not mutated during hoist, we are safe to get vmask
    // from vmask_.simd_vmask_ by Call node.
    Expr vmask;
    auto &simd_vmask = vmask_.simd_vmask_;
    auto vmask_scan = [&simd_vmask, &vmask](const NodeRef &op) {
      if (!vmask.defined()) {
        const Call *call = op.as<Call>();
        if (call && simd_vmask.find(call) != simd_vmask.end()) {
          vmask = simd_vmask[call];
        }
      }
    };

    ktvm::ir::PostOrderVisit(stmt, vmask_scan);

    return vmask;
  }

  // hoist info, from InvarHoistPlan
  std::unordered_map<const For *, HoistEntry> &hoist_info_;
  // nodes need to hoist
  std::unordered_map<const Node *, Stmt> nodes_;
  // alloc node need to hoist
  std::unordered_map<const Variable *, std::pair<Stmt, Stmt>> allocs_;
  // gather vmask state
  VMaskGather vmask_;
  bool addPadInit_{false};
  const Call *initCall_{nullptr};
};

Stmt InvariantHoist(Stmt stmt) {
  InvarHoistPlan plan;
  plan.Plan(stmt);

  if (plan.hoist_info_.empty()) {
    return stmt;
  }

  return InvarHoist(plan.hoist_info_).Hoist(stmt);
}

bool EqualAccess(const Call *a, const Call *b) {
  if (!a->is_intrinsic(tvm_access_ptr) || !b->is_intrinsic(tvm_access_ptr)) {
    return false;
  }

  for (size_t i = 0; i < a->args.size(); ++i) {
    const auto abuf = a->args[i].as<Variable>();
    const auto bbuf = b->args[i].as<Variable>();
    // 1. both abuf and bbuf is not Variable: compare by Equal
    if ((abuf == nullptr && bbuf == nullptr) && !Equal(a->args[i], b->args[i])) {
      return false;
    }

    // 2. one is abuf/bbuf is Variable, but the other is not, not equal
    if ((abuf == nullptr && bbuf != nullptr) || (abuf != nullptr && bbuf == nullptr)) {
      return false;
    }

    // 3. both abuf/bbuf is Variable, compare by name_hint
    if ((abuf != nullptr && bbuf != nullptr) && abuf->name_hint != bbuf->name_hint) {
      return false;
    }
  }

  return true;
}

bool EqualInsn(const Call *a, const Call *b) {
  if (a == nullptr || b == nullptr) return false;
  if (a->name != b->name) return false;
  if (a->args.size() != b->args.size()) return false;

  for (size_t i = 0; i < a->args.size(); ++i) {
    // check if Equal first
    if (!Equal(a->args[i], b->args[i])) return false;

    // consider variable name instead of address
    const Call *ac = a->args[i].as<Call>();
    const Call *bc = b->args[i].as<Call>();
    if ((ac == bc) && (ac == nullptr) && (!Equal(a->args[i], b->args[i]))) {
      return false;
    } else if (ac != nullptr && bc != nullptr && !EqualAccess(ac, bc)) {
      return false;
    }
  }

  return true;
}

/* after remove redundant dma copy,
   we need to extend the range of left allocate, unify variables

 a1 {
   def 1
   a2 {
    BB1: def 2
         use 1
         use 2
   }
 }

 a1 {
  a2 {
    BB2: use 1
         use 2
  }
 }

after merge ------>

 a1 {
   def 1
   a2 {
    BB1: def 2
         use 1
         use 2
    BB2: use 1
         use 2
   }
 }
*/

class InjectTopAlloc : public IRMutator {
 public:
  InjectTopAlloc(const Stmt inject, const Variable *var_) : inject_(inject), innermost_alloc_(var_) {}
  ~InjectTopAlloc() override = default;

  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    CHECK(success_) << " dma rm error, lost a block! ";

    return stmt;
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    const Variable *var = op->buffer_var.get();
    if (var != nullptr && var == innermost_alloc_) {
      // make a new block
      Stmt body = Block::make(op->body, inject_);
      success_ = true;

      return Allocate::make(op->buffer_var, op->type, op->extents, op->condition, body, op->new_expr,
                            op->free_function);
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  Stmt inject_;
  bool success_{false};
  const Variable *innermost_alloc_{nullptr};
};

class DmaSubstitute : public IRMutator {
 public:
  DmaSubstitute() {}
  ~DmaSubstitute() override = default;

  Stmt Run(Stmt stmt, const std::unordered_map<const Variable *, Expr> &vmap) {
    vmap_ = vmap;
    stmt = Substitute(stmt, vmap_);
    stmt = this->Mutate(stmt);

    return stmt;
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    Expr var = op->buffer_var;
    if (vmap_.count(op->buffer_var.get())) {
      var = vmap_[op->buffer_var.get()];
    }

    return Load::make(op->type, Downcast<Var>(var), op->index, op->predicate);
  }

 private:
  std::unordered_map<const Variable *, Expr> vmap_;
};

// elim repeat dma insns
// 1) two insns are the same
// 2) both of them are in the same loop nest level
// 3) no other def between them
class ElimRptDMA : public IRMutator {
 public:
  ElimRptDMA() {}
  ~ElimRptDMA() override = default;

  Stmt Run(Stmt stmt, bool &stop) {
    stmt = this->Mutate(stmt);
    if (to_var_ != nullptr) {
      if (top_alloc_.defined()) {
        stmt = InjectTopAlloc(top_alloc_, to_var_).Run(stmt);
      }
    } else {
      stop = true;
    }

    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    deq_outer_loops_.push_front(op);
    Stmt stmt = IRMutator::Mutate_(op, s);
    deq_outer_loops_.pop_front();

    return stmt;
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (to_var_ != nullptr) return s;

    bool single_insn = false;
    if (SingleInsnInFirst(op->first)) {
      // for single insns
      single_insn = true;
    }

    // save for next block
    BlockInfo elem = {op, single_insn};
    Blocks_.push_front(elem);

    bool found_first = false;
    auto before_first = insns_;
    Stmt first = this->Mutate(op->first);
    if (to_var_ != nullptr) found_first = true;

    bool found_rest = false;
    auto before_rest = insns_;
    Stmt rest = op->rest;
    if (op->rest.defined() && to_var_ == nullptr) {
      rest = this->Mutate(op->rest);
      if (to_var_ != nullptr) found_rest = true;
    }
    Stmt ret = Block::make(first, rest);
    // pop
    Blocks_.pop_front();

    std::set<const Variable *> outer_vars;
    for (auto i : cur_outer_alloc_) {
      if (const auto al2 = i->body.as<Allocate>()) {
        const Variable *var = al2->buffer_var.get();
        outer_vars.insert(var);
      }
    }

    // not find last_var, just return
    if (!found_first && !found_rest) {
      return ret;
    }

    // for block has fmatrix in first part, it should not be top block
    if (InThisScop(to_var_, before_first) && Blocks_.size() > 0 && Blocks_.front().single_insn_first) {
      return ret;
    }

    if (!SameLevel() || top_alloc_.defined() || !has_top_alloc_) {
      return ret;
    }

    // if def is in to_var_'s scop, no need to make block
    if (outer_vars.count(to_var_)) {
      has_top_alloc_ = false;
      return ret;
    }

    // top block
    if (InThisScop(to_var_, before_first) && found_rest) {
      has_top_alloc_ = false;
      top_alloc_ = ret;
      ret = Evaluate::make(0);
    } else if (InThisScop(to_var_, before_first) && found_first) {
      has_top_alloc_ = false;
      top_alloc_ = first;
      ret = Block::make(Evaluate::make(0), rest);
    } else if (InThisScop(to_var_, before_rest) && found_rest) {
      has_top_alloc_ = false;
      top_alloc_ = rest;
      ret = Block::make(first, Evaluate::make(0));
    }

    return ret;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    // only handle one multi_core block each time
    if (op->attr_key == "pragma_multi_core") {
      if (!in_multi_core_) {
        in_multi_core_ = true;
      } else {
        // stop go further
        return s;
      }
    }

    // should not be the top flag
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      if (op->body.as<Allocate>()) {
        cur_alloc_level_++;
        cur_outer_alloc_.insert(op);
        Stmt stmt = IRMutator::Mutate_(op, s);
        cur_outer_alloc_.erase(op);
        cur_alloc_level_--;

        if (const auto attr = stmt.as<AttrStmt>()) {
          if (const auto al2 = attr->body.as<Allocate>()) {
            const Variable *var = al2->buffer_var.get();
            // remove alloca only if src and dst has different variable address
            if (opt_insns_.count(var) && insns_[var->name_hint].var_ != var) {
              // elim AttrStmt and allocate in new stmt
              stmt = al2->body;
              // substitute var
              std::unordered_map<const Variable *, Expr> vmap;
              vmap.emplace(std::pair<const Variable *, Expr>{opt_insns_[var].src_, opt_insns_[var].dst_});
              stmt = DmaSubstitute().Run(stmt, vmap);
              has_top_alloc_ = true;
            }
          }
        }

        return stmt;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &s) final {
    const Call *insn = op->value.as<Call>();
    if (insn && (insn->name == "copy_gm_to_ubuf" || insn->name == "copy_gm_to_cbuf")) {
      const Call *arg_op = insn->args[0].as<Call>();
      if (arg_op != nullptr && arg_op->is_intrinsic(tvm_access_ptr)) {
        const auto buffer = arg_op->args[1].as<Variable>();
        // if we've had def,  and the same as previous def in the same loop level, return
        // evaluate(0); if not, we can not optimize this def, remove from map
        CHECK(buffer);

        if (insns_.count(buffer->name_hint)) {
          std::deque<const For *> result;
          std::set_difference(std::begin(deq_outer_loops_), std::end(deq_outer_loops_),
                              std::begin(insns_[buffer->name_hint].outer_loops_),
                              std::end(insns_[buffer->name_hint].outer_loops_), std::inserter(result, result.end()));

          if (result.size() == 0 && deq_outer_loops_.size() == insns_[buffer->name_hint].outer_loops_.size() &&
              EqualInsn(insn, insns_[buffer->name_hint].insn_)) {
            SubPair pair;
            pair.src_ = buffer;
            const Call *def = insns_[buffer->name_hint].insn_->args[0].as<Call>();
            CHECK(def != nullptr && def->is_intrinsic(tvm_access_ptr)) << " invalid def! ";
            pair.dst_ = def->args[1];
            to_var_ = insns_[buffer->name_hint].var_;
            opt_insns_.emplace(std::pair<const Variable *, SubPair>{buffer, pair});

            return Evaluate::make(0);
          } else {
            insns_.erase(buffer->name_hint);

            return IRMutator::Mutate_(op, s);
          }
        }

        Insn instance;
        instance.insn_ = insn;
        instance.outer_loops_ = deq_outer_loops_;
        instance.var_ = buffer;
        instance.outer_alloc_ = cur_outer_alloc_;
        insns_.emplace(std::pair<std::string, Insn>{buffer->name_hint, instance});
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  struct Insn {
    const Call *insn_;
    std::deque<const For *> outer_loops_;
    const Variable *var_;
    std::set<const AttrStmt *> outer_alloc_;
  };

  struct SubPair {
    const Variable *src_;
    Expr dst_;
  };

  struct BlockInfo {
    const Block *bb;
    bool single_insn_first;
  };

  // current outer loops
  std::deque<const For *> deq_outer_loops_;

  // flag for multi_core
  bool in_multi_core_{false};
  // status for block
  std::deque<BlockInfo> Blocks_;
  // insns_
  std::unordered_map<std::string, Insn> insns_;
  // tell the place of inject
  const Variable *to_var_{nullptr};
  size_t cur_alloc_level_{0};
  // optimized map
  std::unordered_map<const Variable *, SubPair> opt_insns_;
  // current outer alloc
  std::set<const AttrStmt *> cur_outer_alloc_;
  Stmt top_alloc_;
  // flag for top alloc
  bool has_top_alloc_{false};

  // detect target/to_var is in scop
  bool InThisScop(const Variable *to, std::unordered_map<std::string, Insn> insns) const {
    if (to == nullptr) return false;
    if (insns.count(to->name_hint)) {
      return insns[to->name_hint].var_ == to;
    }
    return false;
  }

  bool SameLevel() {
    // search for top AttrStmt and allocate stmt for optimized insn,
    // so we can insert top_alloc into inner src alloc
    // make it empty when optimized
    auto src_set = insns_[to_var_->name_hint].outer_alloc_;
    std::set<const AttrStmt *> result;
    std::set_intersection(src_set.begin(), src_set.end(), cur_outer_alloc_.begin(), cur_outer_alloc_.end(),
                          std::inserter(result, result.end()));
    // calculate the shared outer alloc
    return cur_alloc_level_ == result.size();
  }

  bool SingleInsnInFirst(const Stmt &stmt) const {
    int num = 0;
    auto Scan = [&num](const NodeRef &op) {
      const auto e = op.as<Evaluate>();
      if (e != nullptr) {
        num++;
      }
    };
    ktvm::ir::PostOrderVisit(stmt, Scan);

    return num <= 1;
  }
};

// filter Stmt in which can do ElimDMA
class FilterBody : public IRMutator {
 public:
  FilterBody() {}
  ~FilterBody() override = default;

  Stmt Run(Stmt stmt) {
    stmt = this->Mutate(stmt);
    if (count_ == 0) {
      stmt = ElimDMAReal(stmt);
    }

    return stmt;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_multi_core") {
      Stmt stmt = IRMutator::Mutate_(op, s);
      stmt = ElimDMAReal(stmt);

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  // suffix for dump
  int count_{0};

  Stmt ElimDMAReal(Stmt stmt) {
    bool stop = false;
    while (!stop) {
      count_++;
      stmt = RemoveNoOp(stmt);
      stmt = Simplify(stmt);
      stmt = ElimRptDMA().Run(stmt, stop);
    }

    return stmt;
  }
};

class GatherInsn : public IRVisitor {
 public:
  GatherInsn() {}
  ~GatherInsn() override = default;

  void Visit_(const Call *op) final {
    if (op->name == "copy_ubuf_to_gm" && atomic_add_.size() == 0) {
      const Call *arg_op = op->args[0].as<Call>();
      if (arg_op != nullptr && arg_op->is_intrinsic(tvm_access_ptr)) {
        const auto buffer = arg_op->args[1].as<Variable>();
        if (dma_.count(buffer) == 0) {
          dma_[buffer] = op;
        } else if (EqualInsn(op, dma_[buffer])) {
          // keep last one
          dma_[buffer] = op;
        } else {
          // if different gm is defined, don't touch
          dma_[buffer] = nullptr;
        }
      }
    }

    if (op->name == "copy_gm_to_ubuf" && atomic_add_.size() == 0) {
      const Call *arg_op = op->args[1].as<Call>();
      if (arg_op != nullptr && arg_op->is_intrinsic(tvm_access_ptr)) {
        const auto buffer = arg_op->args[1].as<Variable>();
        if (dma_.count(buffer) > 0) {
          // copy gm back, don't touch
          dma_[buffer] = nullptr;
        }
      }
    }

    if (op->name == "set_atomic_add_open") {
      atomic_add_.push_front(1);
    }

    if (op->name == "set_atomic_add_close") {
      atomic_add_.pop_front();
    }

    IRVisitor::Visit_(op);
  }

  std::unordered_map<const Variable *, const Call *> dma_;

 private:
  std::deque<int> atomic_add_;
};

class ElimUB2GMDMA : public IRMutator {
 public:
  explicit ElimUB2GMDMA(const std::unordered_map<const Variable *, const Call *> &dma) : dma_(dma) {}
  ~ElimUB2GMDMA() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "copy_ubuf_to_gm") {
      const Call *arg_op = op->args[0].as<Call>();
      if (arg_op != nullptr && arg_op->is_intrinsic(tvm_access_ptr)) {
        const auto buffer = arg_op->args[1].as<Variable>();
        // remove all except last one
        if (dma_.count(buffer) > 0 && dma_[buffer] != nullptr && dma_[buffer] != op) {
          return Expr(0);
        }
      }
    }

    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_map<const Variable *, const Call *> dma_;
};

Stmt ElimUB2GM(const Stmt stmt) {
  GatherInsn gather;
  gather.Visit(stmt);

  return ElimUB2GMDMA(gather.dma_).Mutate(stmt);
}

Stmt ElimDMA(Stmt stmt) {
  stmt = ElimUB2GM(stmt);
  return FilterBody().Run(stmt);
}

Stmt HoistInsn(Stmt stmt) {
  stmt = Simplify(stmt);
  PreHoist pre_hoist;
  pre_hoist.Visit(stmt);
  stmt = Hoist(pre_hoist.before_, pre_hoist.remove_).Mutate(stmt);
  stmt = ElimRptDef().Mutate(stmt);

  return stmt;
}
}  // namespace ir
}  // namespace akg
