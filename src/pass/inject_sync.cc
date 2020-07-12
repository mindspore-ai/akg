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
#include <limits.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <pass/ir_util.h>
#include "pass/common.h"

namespace akg {
namespace ir {
using air::ir::attr::coproc_scope;

class InnateSyncChecker {
 public:
  bool Check(const AttrStmt *from, const AttrStmt *to) {
    CHECK(from != nullptr && to != nullptr);
    CHECK(from->attr_key == coproc_scope) << "From is not a coproc_scope";
    CHECK(to->attr_key == coproc_scope) << "To is not a coproc_scope";

    const Call *insn = nullptr;
    auto collect_insn = [&insn](const NodeRef &op) {
      const auto eval = op.as<Evaluate>();
      if ((eval != nullptr) && (insn == nullptr)) {
        const auto call = eval->value.as<Call>();
        if (call != nullptr) insn = call;
      }
    };

    PostOrderVisit(from->body, collect_insn);
    const Call *from_insn = insn;
    insn = nullptr;

    PostOrderVisit(to->body, collect_insn);
    const Call *to_insn = insn;

    CHECK(from->value.as<IntImm>() != nullptr && to->value.as<IntImm>() != nullptr);
    int from_pipe = static_cast<int>(from->value.as<IntImm>()->value);
    int to_pipe = static_cast<int>(to->value.as<IntImm>()->value);

    return CheckInnateSync(from_insn, to_insn, from_pipe, to_pipe);
  }

 private:
  bool CheckInnateSync(const Call *from, const Call *to, int from_pipe, int to_pipe) {
    // from/to is null is expect scalar pipe
    if ((from_pipe == PIPE_S) && (from != nullptr) && (from->name == "reg_mov")) {
      CHECK_GE(from->args.size(), 1);
      const Call *m = from->args[0].as<Call>();
      if ((m != nullptr) && (m->name == "reg")) {
        return true;
      }
    }

    if (from_pipe == to_pipe) {
      // scalar barrier is hardware guarantees
      if (from_pipe == PIPE_S) {
        return true;
      }

      // spr barrier
      if ((from != nullptr && to != nullptr) &&
          (innateBarAfterSPR.count(from->name) > 0 || innateBarBeforeSPR.count(to->name) > 0)) {
        return true;
      }

      // When the from and to type are mad,
      // and from's M and N is one of the following conditions:
      // 1) M > 64 && N > 16
      // 2) 64 >= M > 32 && N > 32
      // Then hardware ensures that the synchronization is correct.
      if (from_pipe == PIPE_M && from != nullptr) {
        CHECK(from->name == "mad") << "Ensure your input is Mad AttrStmt.";
        CHECK_GE(from->args.size(), 6);

        const auto m = from->args[3].as<IntImm>();
        const auto n = from->args[5].as<IntImm>();
        if (m != nullptr && n != nullptr && ((m->value > 64 && n->value > 16) || (m->value > 32 && n->value > 32))) {
          return true;
        }
      }
    }

    return false;
  }

  std::unordered_set<std::string> innateBarBeforeSPR = {
    "set_vector_mask",     "set_rpn_offset", "set_fcol2img", "set_deqscale",
    "set_vector_mask_dup", "set_l1_3d_size", "set_fmatrix",
  };
  std::unordered_set<std::string> innateBarAfterSPR = {
    "set_vector_mask", "set_vector_mask_dup", "set_deqscale",     "get_vms4_sr", "get_status",
    "get_ctrl",        "set_fmatrix",         "set_l0_set_value", "set_padding",
  };
};

class SyncDetector : public IRVisitor {
 public:
  enum { PIPE_ALL_NUM = 7 };

  SyncDetector()
      : sync_push_name_("cce.coproc_dep_push"),
        sync_pop_name_("cce.coproc_dep_pop"),
        barrier_name_("cce.coproc_sync") {}

  ~SyncDetector() override = default;

  void Plan(const Stmt stmt) {
    df_ = BuildDfAnalyzer(stmt, false);
    std::unique_ptr<Compound> root_cmpd(new Compound(Compound::Type::COMP_ROOT, nullptr, nullptr));
    state_.cmpd = std::move(root_cmpd);
    this->Visit(stmt);
    FixAllPushPop();
    insert_after_[stmt.get()].emplace_back(MakeBarrier(static_cast<int>(PIPE_ALL_NUM)));
  }

  void Visit_(const AttrStmt *op) final {
    if (nullptr == op) {
      return;
    }

    if (op->attr_key == coproc_scope) {
      const auto ctx_id = op->value.as<IntImm>();
      CHECK(ctx_id != nullptr);
      std::shared_ptr<OpEntry> entry = std::make_shared<OpEntry>();
      entry->node = op->body.get();
      entry->index = state_.index_next++;
      std::shared_ptr<ScopeProc> proc = std::make_shared<ScopeProc>();

      proc->scope = ctx_id->value % 8;  // compact backward
      proc->op = entry.get();
      proc->attr_stmt = op;
      proc->id = proc_id_next_++;
      proc->cmpd = state_.cmpd.get();

      entry->proc.push_back(proc);
      entry->entry[proc->scope] = {proc.get()};
      entry->exit[proc->scope] = {proc.get()};
      state_.op.push_back(entry);

      Submit();
    }

    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) final {
    if (nullptr == op) {
      return;
    }

    SyncState body_state(new Compound(Compound::Type::COMP_FOR, op, state_.cmpd.get()));
    std::swap(state_, body_state);
    this->Visit(op->body);
    if (!state_.op.empty()) {
      HoistPushPopFix();
      FixAllPushPop();
      std::unordered_map<ScopeProc *, std::vector<ProcLink>> back_link;
      // and hoist event to back_link
      for (UnFixedEvent &e : state_.unfixed_event) {
        for (auto exit_proc : e.from_exit) {
          for (auto entry_proc : e.to_entry) {
            back_link[exit_proc].emplace_back(ProcLink{entry_proc, ProcLink::LINK_EVENT, e.id});
          }
        }
      }

      // and scope order to back_link
      for (auto exit_scope : state_.exit) {
        auto entry_scope = state_.entry.find(exit_scope.first);
        if (entry_scope != state_.entry.end()) {
          for (auto exit_proc : exit_scope.second) {
            for (auto entry_proc : entry_scope->second) {
              back_link[exit_proc].emplace_back(ProcLink{entry_proc, ProcLink::LINK_SCOPE_ORDER, 0});
            }
          }
        }
      }

      BackwardVisit(op, back_link);
      std::swap(state_, body_state);
      std::shared_ptr<OpEntry> entry = std::make_shared<OpEntry>();
      entry->node = op;
      entry->index = state_.index_next++;
      state_.op.push_back(entry);
      MergeState(body_state);
      Submit();
    } else {
      std::swap(state_, body_state);
    }
  }

  void Visit_(const IfThenElse *op) final {
    if (nullptr == op) {
      return;
    }

    SyncState then_state(new Compound(Compound::Type::COMP_IF_THEN, op, state_.cmpd.get()));
    SyncState else_state(new Compound(Compound::Type::COMP_IF_ELSE, op, state_.cmpd.get()));
    std::swap(state_, then_state);
    this->Visit(op->then_case);
    FixAllPushPop();
    std::swap(state_, then_state);
    if (op->else_case.defined()) {
      std::swap(state_, else_state);
      this->Visit(op->else_case);
      FixAllPushPop();
      std::swap(state_, else_state);
    }

    if (then_state.op.empty() && else_state.op.empty()) {
      return;
    }

    std::shared_ptr<OpEntry> entry = std::make_shared<OpEntry>();
    entry->node = op;
    entry->index = state_.index_next++;
    state_.op.push_back(entry);
    if (!then_state.op.empty()) {
      MergeState(then_state);
    }

    if (!else_state.op.empty()) {
      MergeState(else_state);
    }

    std::vector<std::shared_ptr<ScopeProc>> vproc_entry;
    std::vector<std::shared_ptr<ScopeProc>> vproc_exit;
    int vport_entry_id = entry->proc.front()->id;
    int vport_exit_id = entry->proc.back()->id;
    for (auto &it : entry->entry) {
      std::shared_ptr<ScopeProc> v_entry(
        new ScopeProc{vport_entry_id, it.first, entry.get(), nullptr, state_.cmpd.get()});
      std::shared_ptr<ScopeProc> v_exit(
        new ScopeProc{vport_exit_id, it.first, entry.get(), nullptr, state_.cmpd.get()});
      vproc_entry.emplace_back(v_entry);
      vproc_exit.emplace_back(v_exit);
      v_exit->link_from.emplace_back(ProcLink{v_entry.get(), ProcLink::LINK_SCOPE_ORDER, 0});
      SetReached(v_entry.get(), v_exit.get(), false);
    }

    entry->proc.insert(entry->proc.begin(), vproc_entry.begin(), vproc_entry.end());
    entry->proc.insert(entry->proc.end(), vproc_exit.begin(), vproc_exit.end());
    for (auto &v : vproc_entry) {
      for (ScopeProc *p : entry->entry[v->scope]) {
        p->link_from.emplace_back(ProcLink{v.get(), ProcLink::LINK_SCOPE_ORDER, 0});
        SetReached(v.get(), p, false);
      }
      entry->entry[v->scope] = {v.get()};
    }

    for (auto &v : vproc_exit) {
      for (ScopeProc *p : entry->exit[v->scope]) {
        v->link_from.emplace_back(ProcLink{p, ProcLink::LINK_SCOPE_ORDER, 0});
        SetReached(p, v.get(), false);
      }
      entry->exit[v->scope] = {v.get()};
    }

    Submit();
  }

  // insert before is stored in reverse order
  // the first element is closest to the node.
  std::unordered_map<const Node *, std::vector<Stmt>> insert_before_;
  std::unordered_map<const Node *, std::vector<Stmt>> insert_after_;

 private:
  struct ScopeProc;
  struct OpEntry;

  struct Compound {
    enum Type {
      COMP_IF_THEN,
      COMP_IF_ELSE,
      COMP_FOR,
      COMP_ROOT,
    };

    Compound(Type t, const Node *n, Compound *p) : type(t), node(n), parent(p) {
      level = parent != nullptr ? parent->level + 1 : 0;
    }

    Type type;
    int level;
    const Node *node{nullptr};
    Compound *parent;
    std::vector<std::unique_ptr<Compound>> children;
    std::unordered_map<int, std::vector<ScopeProc *>> entry;
    std::unordered_map<int, std::vector<ScopeProc *>> exit;
  };

  struct ProcLink {
    enum Type {
      LINK_EVENT,
      LINK_BARRIER,
      LINK_SCOPE_ORDER,
      LINK_INNATE,
    };

    // target op link to
    ScopeProc *proc;
    // link type
    Type type;
    // event id. valid if type is event.
    int event_id;
  };

  struct ScopeProc {
    // id of scope
    int id;
    // pipeline scope.
    int scope;
    // owner op.
    OpEntry *op;
    // attr node of this proc, used for DFAnalyzer query.
    const AttrStmt *attr_stmt;
    // compound of this proc
    Compound *cmpd;
    // direct linked proc.
    std::vector<ProcLink> link_from;
    // reachable procs cached, to speedup reachable searching.
    std::unordered_set<const ScopeProc *> reached;
    std::unordered_set<const ScopeProc *> ordered;
  };

  struct OpEntry {
    // node of this op, used to sync adding.
    const Node *node;
    // op index in block.
    int index;
    // procs of this op, in sequence order
    std::vector<std::shared_ptr<ScopeProc>> proc;
    // entry procs, map from scope id to Procs with direct tags.
    std::unordered_map<int, std::vector<ScopeProc *>> entry;
    // exit procs, map from scope id to Procs with direct tags.
    std::unordered_map<int, std::vector<ScopeProc *>> exit;
  };

  struct UnFixedEvent {
    // from proc
    ScopeProc *from;
    // to proc
    ScopeProc *to;
    // event id
    int id;
    // exit scope proc of from
    std::vector<ScopeProc *> from_exit;
    // entry scope proc of to
    std::vector<ScopeProc *> to_entry;
  };

  struct EventPool {
    enum { EVENT_NR = 4 };
    struct {
      // in using
      bool busy{false};
      // from op index, push insert is after
      int from_index{0};
      // to op index, pop is insert is before
      int to_index{0};
    } slot[EVENT_NR];
  };

  struct SyncState {
    explicit SyncState(Compound *c = nullptr) : cmpd(c) {}

    // next op index to alloc.
    int index_next{0};
    // ops of this block, in sequence order. with increasing op index.
    std::vector<std::shared_ptr<OpEntry>> op;
    // entry procs, map from scope id to Procs with direct tags.
    std::unordered_map<int, std::vector<ScopeProc *>> entry;
    // exit procs, map from scope id to Procs with direct tags.
    std::unordered_map<int, std::vector<ScopeProc *>> exit;
    // backward event sync link. need to fixed outside.
    std::vector<UnFixedEvent> unfixed_event;
    // need pop event
    std::list<UnFixedEvent> unpop_event;
    // need push event
    std::list<UnFixedEvent> unpush_event;
    // map from scope pair to EventPool.
    std::unordered_map<int, EventPool> event_pool;
    // compound of this state
    std::unique_ptr<Compound> cmpd;
  };

  int ScopePair(int from, int to) const {
    auto from_t = static_cast<unsigned int>(from);
    auto to_t = static_cast<unsigned int>(to);

    return static_cast<int>((from_t << 16) | to_t);
  }

  // Insert push fix stmt, and amend the links.
  void InsertPushFix(OpEntry *op, const UnFixedEvent &e, bool update_index = true) {
    if (nullptr != op) {
      insert_after_[op->node].push_back(MakePush(e.from->scope, e.to->scope, e.id));
      if (update_index) {
        int scope_pair = ScopePair(e.from->scope, e.to->scope);
        state_.event_pool[scope_pair].slot[e.id].from_index = op->index;
      }

      std::unordered_map<int, std::vector<ScopeProc *>>::iterator it;
      while ((it = op->exit.find(e.from->scope)) == op->exit.end()) {
        CHECK_GT(op->index, 0);
        op = state_.op[op->index - 1].get();
      }

      for (ScopeProc *exit : it->second) {
        e.to->link_from.emplace_back(ProcLink{exit, ProcLink::LINK_EVENT, e.id});
        SetReached(exit, e.to);
      }
    }
  }

  // Insert pop fix stmt, and amend the links.
  void InsertPopFix(OpEntry *op, const UnFixedEvent &e, bool update_index = true) {
    if (nullptr != op) {
      insert_before_[op->node].push_back(MakePop(e.from->scope, e.to->scope, e.id));
      if (update_index) {
        int scope_pair = ScopePair(e.from->scope, e.to->scope);
        state_.event_pool[scope_pair].slot[e.id].to_index = op->index;
      }

      std::unordered_map<int, std::vector<ScopeProc *>>::iterator it;
      while ((it = op->entry.find(e.to->scope)) == op->entry.end()) {
        CHECK((size_t)(uint32_t)(op->index + 1) < state_.op.size());
        op = state_.op[op->index + 1].get();
      }

      for (ScopeProc *entry : it->second) {
        entry->link_from.emplace_back(ProcLink{e.from, ProcLink::LINK_EVENT, e.id});
        SetReached(e.from, entry);
      }
    }
  }

  // seek last pop op with the same event
  OpEntry *SeekLastPopOp(const UnFixedEvent &e) {
    // visit backward on op list and find op with the same event with e.
    for (int op_idx = static_cast<int>(state_.op.size()) - 2; op_idx >= 0; op_idx--) {
      OpEntry *op = state_.op[op_idx].get();
      if (nullptr == op) {
        continue;
      }

      for (auto itr = op->proc.rbegin(); itr != op->proc.rend(); ++itr) {
        ScopeProc *proc = itr->get();
        if (nullptr == proc) {
          continue;
        }

        if (proc->scope != e.to->scope) {
          continue;
        }

        for (ProcLink &link : proc->link_from) {
          if ((link.proc->scope == e.from->scope) && (link.event_id == e.id)) {
            return proc->op;
          }
        }
      }
    }

    return nullptr;
  }

  // seek push fix op backward. the condition op is:
  // 1. dependency is exist
  // 2. event id is conflict.
  OpEntry *SeekPushFix(const UnFixedEvent &e, OpEntry *last_pop_op) {
    // because conflict event id is checked from 'from' proc, so the nearest
    // dependence may be ahead of conflict 'to' proc. we should find last conflict
    // op first.
    for (int op_idx = static_cast<int>(state_.op.size()) - 2; op_idx >= 0; op_idx--) {
      OpEntry *op = state_.op[op_idx].get();
      if ((op == nullptr) || (op == last_pop_op)) {
        // process conflict id in post, because reachable may ahead it
        return nullptr;
      }

      for (auto itr = op->proc.rbegin(); itr != op->proc.rend(); ++itr) {
        ScopeProc *proc = itr->get();
        if (proc->scope != e.from->scope) {
          continue;
        }

        // proc before e.to of the same op will add sync if depended, e.to must be reachable
        for (auto it2 = e.from->op->proc.rbegin(); it2 != e.from->op->proc.rend(); ++it2) {
          ScopeProc *p2 = it2->get();
          if ((p2->scope == e.to->scope) && DepBetween(proc, p2)) {
            return op;
          }
        }
      }
    }

    return nullptr;
  }

  // check if cur_op can do pop fix of e.
  // these condition will be treat as need pop:
  // 1. dependency is exist
  // 2. event id is conflict.
  bool NeedPopFix(const OpEntry *cur_op, const UnFixedEvent &e) {
    if (nullptr == cur_op) {
      return false;
    }

    for (auto proc : cur_op->proc) {
      if (proc == nullptr) {
        return false;
      }

      if (proc->scope == e.to->scope) {
        for (int i = e.from->op->index; i >= 0; i--) {
          // proc after e.from of the same op will add sync if depended, e.from must be reachable
          for (auto dep : state_.op[i]->proc) {
            if ((dep->scope == e.from->scope) && DepBetween(dep.get(), proc.get())) {
              return true;
            }
          }
        }
      } else if (proc->scope == e.from->scope) {
        auto proc_link_form = proc->link_from;
        if (std::any_of(proc_link_form.begin(), proc_link_form.end(), [&](const ProcLink &link) {
              return ((link.proc->scope == e.from->scope) && (link.type == ProcLink::LINK_EVENT) &&
                      (link.event_id == e.id));
            })) {
          return true;
        }
      }
    }

    return false;
  }

  // eliminate push and push pair.
  // if unpop_event and unpush have the same event id and pipe, they can be eliminated in pair.
  bool ElimPushPopFix(const OpEntry *cur_op, OpEntry *fix_op, const UnFixedEvent &unpush) {
    for (auto it = state_.unpop_event.begin(); it != state_.unpop_event.end(); ++it) {
      UnFixedEvent &unpop = *it;
      if ((unpop.from->op != cur_op) && (unpush.from->scope == unpop.from->scope) &&
          (unpush.to->scope == unpop.to->scope) && (unpush.id == unpop.id)) {
        if ((fix_op == nullptr) || (unpop.from->op->index >= fix_op->index)) {
          int scope_pair = ScopePair(unpop.from->scope, unpop.to->scope);
          state_.event_pool[scope_pair].slot[unpop.id].to_index = cur_op->index;
          unpush.to->link_from.emplace_back(ProcLink{unpop.from, ProcLink::LINK_EVENT, unpop.id});
          SetReached(unpop.from, unpush.to);
        } else {
          InsertPopFix(fix_op, unpop, false);
          InsertPushFix(fix_op, unpush, false);
        }

        state_.unpop_event.erase(it);

        return true;
      }
    }

    return false;
  }

  // prev hook function of submit. called before forward link is added.
  // in prev, unpush_event and unpop_event with dependency alternation is processed.
  void SubmitPrev(OpEntry *cur_op, std::unordered_map<UnFixedEvent *, OpEntry *> &last_pop) {
    CHECK(cur_op != nullptr);
    if (cur_op->node->IsInstance<For>()) {
      for (auto &it : state_.unpush_event) {
        UnFixedEvent &e = it;
        if (e.from->op == cur_op) {
          last_pop[&e] = SeekLastPopOp(e);
        }
      }

      for (auto it = state_.unpush_event.begin(); it != state_.unpush_event.end();) {
        UnFixedEvent &e = *it;
        if (e.from->op == cur_op) {
          CHECK(last_pop.find(&e) != last_pop.end());
          OpEntry *fix_op = SeekPushFix(e, last_pop[&e]);
          if (ElimPushPopFix(cur_op, fix_op, e)) {
            it = state_.unpush_event.erase(it);
            continue;
          }

          if (fix_op != nullptr) {
            InsertPushFix(fix_op, e);
            it = state_.unpush_event.erase(it);
            continue;
          }
        }
        ++it;
      }
    }
    for (auto it = state_.unpop_event.begin(); it != state_.unpop_event.end();) {
      if (it->from->op != cur_op && NeedPopFix(cur_op, *it)) {
        InsertPopFix(cur_op, *it);
        it = state_.unpop_event.erase(it);
      } else {
        ++it;
      }
    }
  }

  // post hook function of submit. called after forward link is added.
  // in post, unpush_event and unpop_event without dependency alternation is processed.
  void SubmitPost(const OpEntry *cur_op, std::unordered_map<UnFixedEvent *, OpEntry *> &last_pop) {
    // unpush_event reachable
    CHECK(cur_op != nullptr);
    if (cur_op->node->IsInstance<For>()) {
      for (int op_idx = static_cast<int>(state_.op.size()) - 2; op_idx >= 0; op_idx--) {
        for (auto it1 = state_.op[op_idx]->proc.rbegin(); it1 != state_.op[op_idx]->proc.rend(); ++it1) {
          ScopeProc *proc = it1->get();
          for (auto it2 = state_.unpush_event.begin(); it2 != state_.unpush_event.end();) {
            UnFixedEvent &e = *it2;
            if ((e.from->op == cur_op) && (((proc->scope == e.from->scope) && Reachable(proc, e.to)) ||
                                           (last_pop[&e] == state_.op[op_idx].get()))) {
              InsertPushFix(proc->op, e);
              it2 = state_.unpush_event.erase(it2);
            } else {
              ++it2;
            }
          }
        }
      }
    }

    // unpop_event reachable
    for (std::shared_ptr<ScopeProc> sptr : cur_op->proc) {
      ScopeProc *proc = sptr.get();
      for (auto it = state_.unpop_event.begin(); it != state_.unpop_event.end();) {
        UnFixedEvent &e = *it;
        bool fixed = false;
        if ((e.from->op != cur_op) && (e.to->scope == proc->scope)) {
          for (auto it1 = e.from->op->proc.rbegin(); it1 != e.from->op->proc.rend(); ++it1) {
            if (Reachable(it1->get(), proc)) {
              InsertPopFix(proc->op, e);
              it = state_.unpop_event.erase(it);
              fixed = true;
              break;
            }

            if (it1->get() == e.from) {
              break;
            }
          }
        }

        if (!fixed) ++it;
      }
    }
  }

  // Submit of current op. this submit is called when current op is finished.
  // In submit, we will add link with previous op if dependence exists.
  void Submit() {
    CHECK(state_.op.size() > 0) << "empty state_.op";
    std::shared_ptr<OpEntry> cur_op = state_.op.back();
    // and scope link
    for (auto &scope_entry : cur_op->entry) {
      auto scope_exit = state_.exit.find(scope_entry.first);
      if (scope_exit == state_.exit.end()) {
        continue;
      }

      for (ScopeProc *exit_proc : scope_exit->second) {
        for (ScopeProc *entry_proc : scope_entry.second) {
          entry_proc->link_from.emplace_back(ProcLink{exit_proc, ProcLink::LINK_SCOPE_ORDER, 0});
          SetReached(exit_proc, entry_proc, false);
        }
      }
    }

    // inject sync
    std::unordered_map<UnFixedEvent *, OpEntry *> last_pop;
    SubmitPrev(cur_op.get(), last_pop);
    for (int op_idx = static_cast<int>(state_.op.size()) - 2; op_idx >= 0; op_idx--) {
      for (std::shared_ptr<ScopeProc> cur_proc : cur_op->proc) {
        for (auto itr = state_.op[op_idx]->proc.rbegin(); itr != state_.op[op_idx]->proc.rend(); ++itr) {
          ScopeProc *from = itr->get();
          ScopeProc *to = cur_proc.get();

          if (!Reachable(from, to) && DepBetween(from, to)) {
            InjectSync(from, to);
          }
        }
      }
    }
    SubmitPost(cur_op.get(), last_pop);

    // update state.entry and state.exit
    for (auto &entry : cur_op->entry) {
      if (state_.entry.find(entry.first) == state_.entry.end()) {
        state_.entry[entry.first] = entry.second;
      }
    }

    for (auto &exit : cur_op->exit) {
      state_.exit[exit.first] = exit.second;
    }
  }

  bool PickBranchEntry(ScopeProc *from, ScopeProc *to, ScopeProc *dest, bool sync,
                       std::unordered_map<ScopeProc *, std::pair<Compound *, bool>> &path) {
    Compound *from_cmpd = from->cmpd;
    Compound *to_cmpd = to->cmpd;
    while (from_cmpd->level < to_cmpd->level) {
      if (to_cmpd->type != Compound::Type::COMP_IF_THEN && to_cmpd->type != Compound::Type::COMP_IF_ELSE) {
        to_cmpd = to_cmpd->parent;
        continue;
      }

      Compound *dest_cmpd = dest->cmpd;
      while (dest_cmpd->level > to_cmpd->level) {
        dest_cmpd = dest_cmpd->parent;
      }

      if (dest_cmpd == to_cmpd) return false;

      auto it = path.find(from);
      if (it != path.end()) {
        if ((it->second.second == false) && sync) it->second.second = true;
      } else {
        path.emplace(from, std::make_pair(to_cmpd, sync));
      }

      return true;
    }

    return false;
  }

  bool IsBranchBypassReach(ScopeProc *entry, ScopeProc *dest, Compound *branch, bool &sync) {
    Compound *bypass_branch = nullptr;
    for (auto &c : branch->parent->children) {
      if ((c.get() != branch) && (c.get()->node == branch->node)) {
        bypass_branch = c.get();
        break;
      }
    }

    if (bypass_branch == nullptr) return false;

    bool found = false;
    for (auto &it : bypass_branch->entry) {
      for (ScopeProc *e : it.second) {
        bool reach_1 = entry->reached.count(e) > 0;
        bool order_1 = entry->ordered.count(e) > 0;
        if (!reach_1 && !order_1) continue;

        bool reach_2 = e->reached.count(dest) > 0;
        bool order_2 = e->ordered.count(dest) > 0;
        if (!reach_2 && !order_2) continue;

        if (reach_1 || reach_2) {
          sync = true;
          return true;
        }

        sync = false;
        found = true;
      }
    }

    return found;
  }

  void SetReached(ScopeProc *from, ScopeProc *to, bool sync = true) {
    if (from->reached.count(to)) return;

    std::vector<std::pair<ScopeProc *, bool>> stack;
    std::unordered_map<ScopeProc *, std::pair<Compound *, bool>> branch_entry;

    stack.emplace_back(from, sync);
    while (!stack.empty()) {
      std::tie(from, sync) = stack.back();
      stack.pop_back();
      from->reached.insert(to->reached.begin(), to->reached.end());

      if (sync) {
        from->reached.insert(to);
        if (!to->ordered.empty()) {
          from->reached.insert(to->ordered.begin(), to->ordered.end());
        }
      } else if (from->scope == to->scope) {
        from->ordered.insert(to);
        if (!to->ordered.empty()) {
          from->ordered.insert(to->ordered.begin(), to->ordered.end());
        }
      }

      for (ProcLink &link : from->link_from) {
        bool link_sync = sync || link.type != ProcLink::LINK_SCOPE_ORDER;
        if ((link.proc->reached.count(to) == 0) && !PickBranchEntry(link.proc, from, to, link_sync, branch_entry)) {
          stack.emplace_back(link.proc, link_sync);
        }
      }
    }

    for (auto &it : branch_entry) {
      if (IsBranchBypassReach(it.first, to, it.second.first, sync)) {
        SetReached(it.first, to, sync && it.second.second);
      }
    }
  }

  bool Reachable(ScopeProc *from, const ScopeProc *to) const { return from->reached.count(to) > 0; }

  // inject sync forward.
  void InjectSync(ScopeProc *from, ScopeProc *to) {
    if ((nullptr == from) || (nullptr == to)) {
      return;
    }

    if (innate_.Check(from->attr_stmt, to->attr_stmt)) {
      if (from->scope == to->scope) {
        for (ProcLink &link : to->link_from) {
          if ((link.proc == from) && (link.type == ProcLink::LINK_SCOPE_ORDER)) {
            link.type = ProcLink::LINK_INNATE;
            SetReached(from, to);

            return;
          }
        }
      }

      to->link_from.emplace_back(ProcLink{from, ProcLink::LINK_INNATE, -1});
      SetReached(from, to);

      return;
    }

    if (from->scope == to->scope) {
      insert_before_[to->op->node].emplace_back(MakeBarrier(to->scope));
      for (ScopeProc *to_entry : to->op->entry[to->scope]) {
        for (ProcLink &link : to_entry->link_from) {
          if (link.type == ProcLink::LINK_SCOPE_ORDER) {
            link.type = ProcLink::LINK_BARRIER;
            SetReached(link.proc, to_entry);
          }
        }
      }
    } else {
      OpEntry *to_op = nullptr;
      int event_id = InjectEvent(from, to, &to_op);
      for (ScopeProc *from_exit : from->op->exit[from->scope]) {
        for (ScopeProc *to_entry : to_op->entry[to->scope]) {
          to_entry->link_from.emplace_back(ProcLink{from_exit, ProcLink::LINK_EVENT, event_id});
          SetReached(from_exit, to_entry);
        }
      }
    }
  }

  // backward reachable check. if reachable, return true.
  bool BackwardReachable(ScopeProc *from, const ScopeProc *to,
                         const std::unordered_map<ScopeProc *, std::vector<ProcLink>> &back_link) {
    // To avoid duplicate sync pair between same op pair, we record existed
    // sync to exist_sync for reuse.
    // To avoid circle dependence between OPs, we record backward link in back_link
    // for local reachable search. and this reachable search splits in 3 parts:
    // from -> back_link_from, back_link_from -> back_link_to, back_link_to -> to.
    // At least one sync in these parts.
    CHECK((from != nullptr) && (to != nullptr));
    CHECK(from->op->index >= to->op->index);

    for (auto proc_link : back_link) {
      CHECK(proc_link.first != nullptr);
      if (from->id > proc_link.first->id) {
        continue;
      }

      bool from_sync = Reachable(from, proc_link.first);
      bool from_order = from->scope == proc_link.first->scope;
      if (!from_sync && !from_order) continue;

      for (ProcLink &link : proc_link.second) {
        if (link.proc->id > to->id) continue;

        bool to_sync = Reachable(link.proc, to);
        bool to_order = link.proc->scope == to->scope;
        if (((from_sync || from_order) && (to_sync || to_order)) &&
            (from_sync || to_sync || (link.type != ProcLink::LINK_SCOPE_ORDER))) {
          return true;
        }
      }
    }

    return false;
  }

  // Inject barrier for backward of loop. it's a helper function of InjectSyncBackward.
  void InjectBarrierBackward(const ScopeProc *from, const ScopeProc *to,
                             std::unordered_map<ScopeProc *, std::vector<ProcLink>> &back_link) {
    CHECK((from != nullptr) && (from->op != nullptr));

    if (from->scope != PIPE_S) {  // scalar barrier is hardware guarantees
      insert_after_[from->op->node].emplace_back(MakeBarrier(from->scope));
    }

    bool need_back_link = true;
    CHECK(from->op->exit.find(from->scope) != from->op->exit.end());
    for (size_t i = from->op->index + 1; i < state_.op.size(); ++i) {
      if (state_.op[i]->entry.find(from->scope) != state_.op[i]->entry.end()) {
        for (ScopeProc *proc : state_.op[i]->entry[from->scope]) {
          for (ProcLink &link : proc->link_from) {
            if (link.type == ProcLink::LINK_SCOPE_ORDER) {
              link.type = ProcLink::LINK_BARRIER;
              SetReached(link.proc, proc);
              need_back_link = false;
            }
          }
        }

        if (!need_back_link) break;
      }
    }

    if (need_back_link) {
      // from node is in state_.exit. so at least order link exist.
      for (ScopeProc *proc : state_.exit[from->scope]) {
        CHECK(back_link.find(proc) != back_link.end());
        for (ProcLink &link : back_link[proc]) {
          if (link.type == ProcLink::LINK_SCOPE_ORDER) {
            link.type = ProcLink::LINK_BARRIER;
          }
        }
      }
    }
  }

  // Inject sync for backward of loop.
  // we will not add backward link to proc.link, because it will lead to a cycle of dependences.
  void InjectSyncBackward(ScopeProc *from, ScopeProc *to,
                          std::unordered_map<ScopeProc *, std::vector<ProcLink>> &back_link) {
    if (innate_.Check(from->attr_stmt, to->attr_stmt)) {
      if (from->scope == to->scope) {
        auto it = back_link.find(from);
        if (it != back_link.end()) {
          for (ProcLink &link : it->second) {
            if ((link.proc == to) && (link.type == ProcLink::LINK_SCOPE_ORDER)) {
              link.type = ProcLink::LINK_INNATE;
              return;
            }
          }
        }
      }

      back_link[from].emplace_back(ProcLink{to, ProcLink::LINK_INNATE, -1});

      return;
    }

    if (from->scope == to->scope) {
      InjectBarrierBackward(from, to, back_link);
      return;
    }

    OpEntry *to_op = nullptr;
    bool reused = false;
    int event_id = InjectEvent(from, to, &to_op, &reused);
    if (event_id != -1) {
      if (!reused) {  // add to unfixed_event only if not resued.
        state_.unfixed_event.emplace_back(
          UnFixedEvent{from, to, event_id, from->op->exit[from->scope], to->op->entry[to->scope]});
      } else if (to_op->index > from->op->index) {
        for (ScopeProc *from_exit : from->op->exit[from->scope]) {
          for (ScopeProc *to_entry : to_op->entry[to->scope]) {
            to_entry->link_from.emplace_back(ProcLink{from_exit, ProcLink::LINK_EVENT, event_id});
            SetReached(from_exit, to_entry);
          }
        }

        return;
      } else {
        auto state_event = state_.unfixed_event;
        auto it = std::find_if(state_event.begin(), state_event.end(), [&](const UnFixedEvent &e) {
          return ((e.from->scope == from->scope) && (e.to->op == to_op) && (event_id == e.id));
        });

        if (it != state_event.end()) {
          (*it).from = from;
        }
      }
    }

    for (ScopeProc *from_exit : from->op->exit[from->scope]) {
      for (ScopeProc *to_entry : to_op->entry[to->scope]) {
        back_link[from_exit].emplace_back(ProcLink{to_entry, ProcLink::LINK_EVENT, event_id});
      }
    }
  }

  // backward visit of op of loop, add sync if needed.
  void BackwardVisit(const For *op, std::unordered_map<ScopeProc *, std::vector<ProcLink>> &back_link) {
    if (nullptr == op) {
      return;
    }

    for (std::shared_ptr<OpEntry> op_entry : state_.op) {
      for (std::shared_ptr<ScopeProc> cur_proc : op_entry->proc) {
        for (auto itr = state_.op.rbegin(); itr != state_.op.rend(); ++itr) {
          for (auto proc = (*itr)->proc.rbegin(); proc != (*itr)->proc.rend(); ++proc) {
            ScopeProc *from = proc->get();
            ScopeProc *to = cur_proc.get();
            if (from->id < to->id) break;

            if (!BackwardReachable(from, to, back_link) && DepBetween(from, to, op)) {
              InjectSyncBackward(from, to, back_link);
            }
          }  // end for after_proc

          if (*itr == op_entry) {
            break;
          }
        }  // end for itr
      }    // end for cur_proc
    }
  }

  // Erase push stmt used for event id event.
  // We try to reuse with last event id and move its push stmt forward.
  Stmt ErasePushStmt(const OpEntry *op, int from_scope, int to_scope, int event_id) {
    Stmt stmt;
    if (nullptr == op) {
      return stmt;
    }

    auto insert = insert_after_.find(op->node);
    if (insert != insert_after_.end()) {
      for (auto it = insert->second.begin(); it != insert->second.end(); ++it) {
        const auto evaluate = it->as<Evaluate>();
        const Call *call = (evaluate == nullptr) ? nullptr : evaluate->value.as<Call>();
        if ((call == nullptr) || (call->name != sync_push_name_)) {
          continue;
        }

        const auto intImm0 = call->args[0].as<IntImm>();
        const auto intImm1 = call->args[1].as<IntImm>();
        if (intImm0 != nullptr && intImm1 != nullptr) {
          int64_t event = 0;
          int64_t from = intImm0->value;
          int64_t to = intImm1->value;
          const auto intImm2 = call->args[2].as<IntImm>();
          if (intImm2 != nullptr) {
            event = intImm2->value;
          }

          if ((from == from_scope) && (to == to_scope) && (event == event_id)) {
            stmt = *it;

            if (insert->second.size() == 1) {
              insert_after_.erase(op->node);
            } else {
              insert->second.erase(it);
            }

            break;
          }
        }
      }
    }

    return stmt;
  }

  // Reuse event id forward. find last link used the same id.
  int ReuseForward(const OpEntry *op, const ScopeProc *from, const ScopeProc *to, OpEntry **last_to, Stmt &push_stmt) {
    // find forward link. two kinds of forward link may be found:
    // 1. event push/pop between op. push is at the end of op. need forward push stmt.
    // 2. pop fix. push is in op. need to repush after fix op.
    CHECK((op != nullptr) && (from != nullptr) && (to != nullptr) && (last_to != nullptr));
    for (auto &it : op->proc) {
      ScopeProc *proc = it.get();
      CHECK(proc != nullptr);
      if ((proc->op != op) && (proc->scope == to->scope)) {
        for (ProcLink &link : proc->link_from) {
          CHECK(link.proc != nullptr);
          // keep last event push/pop order unchanged
          if ((link.proc->id < from->id) && (link.proc->scope == from->scope) &&
              ((link.proc->id < proc->id) == (link.proc->id < from->id))) {
            *last_to = proc->op;
            push_stmt = ErasePushStmt(op, from->scope, to->scope, link.event_id);
            // the push_stmt may be null if last proc and 'to' in the same large op
            if (push_stmt.defined()) {
              return link.event_id;
            }
          }
        }
      }
    }

    return -1;
  }

  // Reuse event id backward. process unpop or unfix event
  int ReuseBackward(const OpEntry *op, const ScopeProc *from, ScopeProc *to, OpEntry **last_to, Stmt &push_stmt) {
    CHECK((from != nullptr) && (to != nullptr));
    // phase 1: find unpop link from this op and fix it ahead.
    for (auto it = state_.unpop_event.begin(); it != state_.unpop_event.end(); ++it) {
      UnFixedEvent &e = *it;
      CHECK((e.from != nullptr) && (e.to != nullptr));
      if ((e.from->id < from->id) && (e.from->op == op) && (e.from->scope == from->scope) &&
          (e.to->scope == to->scope)) {
        // if from proc and unpop proc in same op, push should before to op, so:
        //  op(unpop, from), vpop, push, pop, op(to)
        // otherwise push should after from op, that's:
        //  op(unpop), vpop, op(from), push, pop, op(to)
        *last_to = from->op == op ? to->op : from->op;
        int event_id = e.id;
        InsertPopFix(*last_to, e, false);
        state_.unpop_event.erase(it);

        return event_id;
      }
    }

    // phase 2: find backward link.
    if (op != from->op) {
      for (auto &it : state_.unfixed_event) {
        UnFixedEvent &e = it;
        CHECK((e.from != nullptr) && (e.to != nullptr));

        if ((e.from->op == op) && (e.from->scope == from->scope) && (e.to->scope == to->scope)) {
          *last_to = e.to->op;
          push_stmt = ErasePushStmt(op, from->scope, to->scope, e.id);
          // push_stmt may be null if last is hoist to this state
          if (push_stmt.defined()) {
            return e.id;
          }
        }
      }
    }

    return -1;
  }

  // Inject event between op.
  // alloc event id and inject push/pop stmt. if event id is exhausted, try to reuse with last id
  int InjectEvent(const ScopeProc *from, ScopeProc *to, OpEntry **to_op, bool *reused = nullptr) {
    CHECK((from != nullptr) && (from->op != nullptr) && (to != nullptr) && (to->op != nullptr));
    int event_id = AllocEvent(from->op->index, from->scope, to->op->index, to->scope);
    if (event_id != -1) {
      insert_after_[from->op->node].emplace_back(MakePush(from->scope, to->scope, event_id));
      insert_before_[to->op->node].emplace_back(MakePop(from->scope, to->scope, event_id));
      *to_op = to->op;

      return event_id;
    }

    if (reused != nullptr) {
      *reused = true;
    }

    // share last push for event id exhaustion
    OpEntry *last_from = nullptr;
    OpEntry *last_to = nullptr;
    Stmt push_stmt;
    for (int op_idx = from->op->index; op_idx >= 0; --op_idx) {
      OpEntry *op = state_.op[op_idx].get();
      event_id = ReuseForward(op, from, to, &last_to, push_stmt);
      if (event_id != -1) {
        last_from = op;
        break;
      }

      event_id = ReuseBackward(op, from, to, &last_to, push_stmt);
      if (event_id != -1) {
        last_from = op;
        break;
      }
    }

    // use barrier(all) as final choice if last push not found (fix event hoist etc.)
    if (last_from == nullptr) {
      insert_after_[from->op->node].emplace_back(MakeBarrier(static_cast<int>(PIPE_ALL_NUM)));
      *to_op = to->op;

      return -1;
    }

    CHECK((last_from != nullptr) && (last_to != nullptr) && (event_id != -1));
    CHECK(
      !((to->op->index > last_from->index) && (to->op->index < last_to->index) && (from->op->index >= last_to->index)));
    EventPool &pool = state_.event_pool[ScopePair(from->scope, to->scope)];
    if (push_stmt.defined()) {
      insert_after_[from->op->node].emplace_back(push_stmt);
      pool.slot[event_id].from_index = from->op->index;
      // leave old link no change because it is harmless for link search.
      *to_op = last_to;
    } else {  // pop fix
      auto &insert_push = (last_to == to->op) ? insert_before_ : insert_after_;
      insert_push[last_to->node].emplace_back(MakePush(from->scope, to->scope, event_id));
      insert_before_[to->op->node].emplace_back(MakePop(from->scope, to->scope, event_id));
      pool.slot[event_id].from_index = last_to->index;
      pool.slot[event_id].to_index = to->op->index;
      *to_op = to->op;
    }

    return event_id;
  }

  // fix state.unpush_event to vproc
  void FixEntryPop() {
    std::vector<std::shared_ptr<ScopeProc>> vpush;
    OpEntry *cur_op = state_.op.front().get();
    int id = 0;

    if (cur_op->proc.empty() == false) {
      id = cur_op->proc.front()->id;
    }

    for (UnFixedEvent &e : state_.unpush_event) {
      std::shared_ptr<ScopeProc> vproc = std::make_shared<ScopeProc>();

      vproc->scope = e.from->scope;
      vproc->op = cur_op;
      vproc->attr_stmt = nullptr;
      vproc->id = id;
      vproc->cmpd = state_.cmpd.get();
      e.to->link_from.emplace_back(ProcLink{vproc.get(), ProcLink::LINK_EVENT, e.id});
      SetReached(vproc.get(), e.to);
      vpush.push_back(vproc);
      insert_before_[cur_op->node].push_back(MakePush(vproc->scope, e.to->scope, e.id));
      auto scope_proc = state_.entry.find(vproc->scope);
      if (scope_proc != state_.entry.end()) {
        for (ScopeProc *proc : scope_proc->second) {
          proc->link_from.emplace_back(ProcLink{vproc.get(), ProcLink::LINK_SCOPE_ORDER, 0});
          SetReached(vproc.get(), proc, false);
        }
      }

      state_.entry[vproc->scope] = {vproc.get()};
    }

    cur_op->proc.insert(cur_op->proc.begin(), vpush.begin(), vpush.end());
  }

  // fix state.unpop_event with vproc
  void FixExitPush() {
    OpEntry *cur_op = state_.op.back().get();
    int id = 0;
    if (cur_op->proc.empty() == false) {
      id = cur_op->proc.back()->id;
    }

    for (UnFixedEvent &e : state_.unpop_event) {
      std::shared_ptr<ScopeProc> vproc = std::make_shared<ScopeProc>();
      vproc->scope = e.to->scope;
      vproc->op = cur_op;
      vproc->attr_stmt = nullptr;
      vproc->id = id;
      vproc->cmpd = state_.cmpd.get();
      cur_op->proc.push_back(vproc);
      vproc->link_from.emplace_back(ProcLink{e.from, ProcLink::LINK_EVENT, e.id});
      SetReached(e.from, vproc.get());
      insert_after_[cur_op->node].push_back(MakePop(e.from->scope, vproc->scope, e.id));
      auto scope_proc = state_.exit.find(vproc->scope);
      if (scope_proc != state_.exit.end()) {
        for (ScopeProc *proc : scope_proc->second) {
          vproc->link_from.emplace_back(ProcLink{proc, ProcLink::LINK_SCOPE_ORDER, 0});
          SetReached(proc, vproc.get(), false);
        }
      }

      state_.exit[vproc->scope] = {vproc.get()};
    }
  }

  // fix all unpush and unpop event
  void FixAllPushPop() {
    if (!state_.unpush_event.empty()) {
      FixEntryPop();
      state_.unpush_event.clear();
    }

    if (!state_.unpop_event.empty()) {
      FixExitPush();
      state_.unpop_event.clear();
    }
  }

  /**
   * hoist paired unpush and unpop event upward. code is changed as follow:
   * + push(3, 2)
   *   loop (outer,) {
   * -   push(3, 2)
   *   loop (inner,) {
     ...
     }
   * -   pop(3, 2)
   *   }
   * + pop(3, 2)
   */
  void HoistPushPopFix() {
    CHECK(state_.unfixed_event.empty());
    for (auto it1 = state_.unpush_event.begin(); it1 != state_.unpush_event.end();) {
      bool matched = false;
      for (auto it2 = state_.unpop_event.begin(); it2 != state_.unpop_event.end(); ++it2) {
        if ((it1->from->scope == it2->from->scope) && (it1->to->scope == it2->to->scope) && (it1->id == it2->id)) {
          it1->from = it2->from;
          it1->from_exit = std::move(it2->from_exit);
          state_.unfixed_event.emplace_back(*it1);
          state_.unpop_event.erase(it2);
          matched = true;

          break;
        }
      }

      if (matched) {
        it1 = state_.unpush_event.erase(it1);
      } else {
        ++it1;
      }
    }
  }

  // Merge current syncState upward, and reduce to op of parent syncState
  void MergeState(SyncState &state) {
    CHECK(state_.op.size() > 0) << "empty state_.op";
    std::shared_ptr<OpEntry> cur_op = state_.op.back();
    for (std::shared_ptr<OpEntry> op : state.op) {
      for (std::shared_ptr<ScopeProc> proc : op->proc) {
        proc->op = cur_op.get();
        cur_op->proc.push_back(proc);
        if (proc->attr_stmt == nullptr) {  // vpush or vpop
          continue;
        }

        for (ProcLink &link : proc->link_from) {
          // in some reuse case when barrier all is used in forward link. the
          // event_id is -1. it should not be updated to event pool
          if ((link.type == ProcLink::LINK_EVENT) && (link.event_id >= 0)) {
            EventPool &mgr = state_.event_pool[ScopePair(link.proc->scope, proc->scope)];
            mgr.slot[link.event_id].busy = true;
            mgr.slot[link.event_id].from_index = cur_op->index;
            mgr.slot[link.event_id].to_index = cur_op->index;
          }
        }
      }
    }

    // resolve state.unfixed_event to unpop_event and unpush_event
    for (UnFixedEvent &unfix : state.unfixed_event) {
      state_.unpush_event.emplace_back(unfix);
      state_.unpop_event.emplace_back(unfix);
      EventPool &mgr = state_.event_pool[ScopePair(unfix.from->scope, unfix.to->scope)];
      mgr.slot[unfix.id].busy = true;
      mgr.slot[unfix.id].from_index = -1;
      mgr.slot[unfix.id].to_index = INT_MAX;
    }

    for (auto entry : state.entry) {
      auto base_entry = cur_op->entry.find(entry.first);
      if (base_entry != cur_op->entry.end()) {
        base_entry->second.insert(base_entry->second.end(), entry.second.begin(), entry.second.end());
      } else {
        cur_op->entry[entry.first] = entry.second;
      }
    }

    for (auto exit : state.exit) {
      auto base_exit = cur_op->exit.find(exit.first);
      if (base_exit != cur_op->exit.end()) {
        base_exit->second.insert(base_exit->second.end(), exit.second.begin(), exit.second.end());
      } else {
        cur_op->exit[exit.first] = exit.second;
      }
    }

    state.cmpd->entry = std::move(state.entry);
    state.cmpd->exit = std::move(state.exit);
    state_.cmpd->children.emplace_back(std::move(state.cmpd));
  }

  // Check if two Proc are dependent.
  // if loopback is not null, it is treated as backward.
  bool DepBetween(ScopeProc *from, ScopeProc *to, const For *loopback = nullptr) {
    CHECK((from != nullptr) && (to != nullptr));
    // vproc's attr_stmt is nullptr
    if ((from->attr_stmt == nullptr) || (to->attr_stmt == nullptr)) {
      return false;
    }

    return (loopback == nullptr) ? df_->DepForward(from->attr_stmt, to->attr_stmt)
                                 : df_->DepBackward(from->attr_stmt, to->attr_stmt, loopback);
  }

  // Alloc event id between different pipe.
  // recycle freed id in forward, and lazy reused in backward.
  int AllocEvent(int from_index, int from_scope, int to_index, int to_scope) {
    EventPool &pool = state_.event_pool[ScopePair(from_scope, to_scope)];

    if (from_index < to_index) {
      // in forward allocation, we just alloc new id, or extent its liveness if not overlap.
      // The liveness keep extent to prevent overlap with backward allocation.
      for (int i = 0; i < EventPool::EVENT_NR; ++i) {
        auto &slot = pool.slot[i];
        if ((!slot.busy) || ((slot.from_index <= slot.to_index) && (slot.to_index <= from_index))) {
          if (!slot.busy) slot.from_index = from_index;

          slot.to_index = to_index;
          slot.busy = true;

          return i;
        }
      }
    } else {
      // in backward allocation, we first alloc free id which is never used in forward.
      for (int i = 0; i < EventPool::EVENT_NR; ++i) {
        auto &slot = pool.slot[i];
        if (!slot.busy) {
          slot.from_index = from_index;
          slot.to_index = to_index;
          slot.busy = true;

          return i;
        }
      }
      // lazy recycle forward event in backward sync phase
      // If event id is reused multi-time forward, to_index is ahead of the first slot.from_index,
      // because link will exist if not.
      if (from_index > to_index) {
        for (int i = 0; i < EventPool::EVENT_NR; ++i) {
          auto &slot = pool.slot[i];
          if ((slot.from_index < slot.to_index) && (slot.from_index >= to_index) && (slot.to_index <= from_index)) {
            slot.from_index = from_index;
            slot.to_index = to_index;

            return i;
          }
        }
      }
    }

    return -1;
  }

  // Make a coproc push stmt, used for synchronization between different pipe
  Stmt MakePush(int from, int to, int event) {
    return Evaluate::make(Call::make(Int(32), sync_push_name_,
                                     {make_const(Int(32), from), make_const(Int(32), to), make_const(Int(32), event)},
                                     Call::Intrinsic));
  }

  // Make a coproc pop stmt, used for synchronization between different pipe
  Stmt MakePop(int from, int to, int event) {
    return Evaluate::make(Call::make(Int(32), sync_pop_name_,
                                     {make_const(Int(32), from), make_const(Int(32), to), make_const(Int(32), event)},
                                     Call::Intrinsic));
  }

  // Make a barrier stmt, used for synchronization between same pipe
  Stmt MakeBarrier(int scope) {
    return Evaluate::make(Call::make(Int(32), barrier_name_, {make_const(Int(32), scope)}, Call::Intrinsic));
  }

  // next proc id for id assignment
  int proc_id_next_{0};
  // Variables
  SyncState state_;
  // data flow analysis
  std::shared_ptr<DFAnalyzer> df_;
  // names of synchronization intrinsic
  std::string sync_push_name_;
  std::string sync_pop_name_;
  std::string barrier_name_;
  // innate sync guaranteed by hw.
  InnateSyncChecker innate_;
};

// SyncInjector: inject sync stmt around pipeline instruction.
// based on plan of CceInstDepDetector.
class SyncInjector : public IRMutator {
 public:
  Stmt Insert(const Stmt stmt) {
    SyncDetector sync_detector;
    sync_detector.Plan(stmt);
    insert_before_ = std::move(sync_detector.insert_before_);
    insert_after_ = std::move(sync_detector.insert_after_);

    return Mutate(stmt);
  }

  Stmt Mutate(Stmt stmt) final {
    const Node *node = stmt.get();
    stmt = IRMutator::Mutate(stmt);
    auto it = insert_before_.find(node);
    if (it != insert_before_.end()) {
      Stmt before = air::ir::MergeSeq(std::vector<Stmt>(it->second.rbegin(), it->second.rend()));
      stmt = Block::make(before, stmt);
    }

    it = insert_after_.find(node);
    if (it != insert_after_.end()) {
      Stmt after = air::ir::MergeSeq(it->second);
      stmt = Block::make(stmt, after);
    }

    return stmt;
  }

 private:
  std::unordered_map<const Node *, std::vector<Stmt>> insert_before_;
  std::unordered_map<const Node *, std::vector<Stmt>> insert_after_;
};

Stmt InjectSync(Stmt stmt) {
  stmt = ConvertSingleCoprocForm(stmt);
  return SyncInjector().Insert(stmt);
}
}  // namespace ir
}  // namespace akg
