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
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/target_info.h>
#include <tvm/arithmetic.h>
#include <arithmetic/compute_expr.h>
#include <runtime/thread_storage_scope.h>

#include <fstream>
#include <regex>

#include "ir_pass.h"
#include "build_module.h"
#include "pass/ir_util.h"
#include "emit_insn/insn_info.h"
#include "pass/storage_rewrite_cce.h"
#include "pass/common.h"
#include "pass/utils.h"
#include "pass/expr_alg_simplify.h"

namespace akg {
namespace ir {
using ktvm::arith::IntSet;
using ktvm::runtime::StorageRank;
using ktvm::runtime::StorageScope;

constexpr auto READ_MASK = 1;
constexpr auto WRITE_MASK = 2;

inline bool prove_equal(const Expr lhs, const Expr rhs) { return is_zero(Simplify(lhs - rhs)); }

void LivenessAnalyzer::Analyze(const Stmt stmt) {
  Visit(stmt);
  CHECK(alloc_keys_.size() == alloc_.size());
  for (auto &a : alloc_keys_) {
    auto &touched = alloc_.at(a).touched;
    if (!touched.empty()) {
      liveness_[touched.front()].gen.emplace_back(a);
      liveness_[touched.back()].kill.emplace_back(a);
    }
  }
}

void LivenessAnalyzer::Visit_(const AttrStmt *op) {
  if (op->attr_key == ktvm::ir::attr::storage_scope) {
    const auto buf = op->node.as<Variable>();
    auto pragma = op->value.as<StringImm>();
    CHECK(pragma != nullptr);
    if (alloc_.find(buf) == alloc_.end()) {
      alloc_keys_.emplace_back(buf);
    }
    alloc_[buf].scope = StorageScope::make(pragma->value);
  } else if (op->attr_key == "pragma_insn_partition") {
    PushScope(op);
    in_insn_partition_ = true;
    IRVisitor::Visit_(op);
    in_insn_partition_ = false;
    PopScope();
    return;
  }
  IRVisitor::Visit_(op);
}

void LivenessAnalyzer::Visit_(const Allocate *op) {
  const Variable *buf = op->buffer_var.get();
  auto it = alloc_.find(buf);
  CHECK(it != alloc_.end() && it->second.alloc == nullptr);
  it->second.alloc = op;
  it->second.level = static_cast<int>(scope_touch_.size());
  IRVisitor::Visit_(op);
}

void LivenessAnalyzer::Visit_(const For *op) {
  PushScope(op);
  IRVisitor::Visit_(op);
  PopScope();
}

void LivenessAnalyzer::Visit_(const IfThenElse *op) {
  PushScope(op);
  IRVisitor::Visit_(op);
  PopScope();
}

void LivenessAnalyzer::Visit_(const Store *op) {
  PushScope(op);
  IRVisitor::Visit_(op);
  TouchBuffer(op->buffer_var.get());
  PopScope();
}

void LivenessAnalyzer::Visit_(const Evaluate *op) {
  PushScope(op);
  IRVisitor::Visit_(op);
  PopScope();
}

void LivenessAnalyzer::Visit_(const Load *op) {
  IRVisitor::Visit_(op);
  TouchBuffer(op->buffer_var.get());
}

void LivenessAnalyzer::Visit_(const Variable *buf) { TouchBuffer(buf); }

void LivenessAnalyzer::TouchBuffer(const Variable *buf) {
  auto it = alloc_.find(buf);
  if (it != alloc_.end()) {
    AllocEntry &e = it->second;
    ScopeTouch &t = scope_touch_[e.level];
    if (!t.touched.count(buf)) {
      t.touched.insert(buf);
      e.touched.emplace_back(t.entry);
    }
  }
}

void LivenessAnalyzer::PushScope(const Node *stmt) {
  if (in_insn_partition_) return;
  int idx = static_cast<int>(liveness_.size());
  liveness_.emplace_back(StmtEntry{stmt});
  scope_touch_.emplace_back(ScopeTouch{idx});
}

void LivenessAnalyzer::PopScope() {
  if (in_insn_partition_) return;
  CHECK(!scope_touch_.empty());
  ScopeTouch &t = scope_touch_.back();
  int size = static_cast<int>(liveness_.size());
  if (t.entry < size - 1) {
    liveness_.emplace_back(StmtEntry{liveness_[t.entry].stmt});
    for (const Variable *buf : t.touched) {
      alloc_[buf].touched.emplace_back(size);
    }
  }
  scope_touch_.pop_back();
}

// check if dst_address can reuse src_address
bool InplaceOpVerifierCCE::CanReuse(const MemInfo &src_address, const MemInfo &dst_address, bool spec_ins) const {
  CHECK(prove_equal(src_address.repeatTime, dst_address.repeatTime));
  CHECK(prove_equal(src_address.blockNumber, dst_address.blockNumber));

  // rule 1, the block size and block stride should be 100% the same
  if (!prove_equal(dst_address.blockSize, src_address.blockSize) ||
      !prove_equal(dst_address.blockStride, src_address.blockStride)) {
    return false;
  }
  // rule 2, it should not use inplace when transpose exist
  auto src_vars = GetVarsInExpr(src_address.offset);
  auto dst_vars = GetVarsInExpr(dst_address.offset);
  Array<Expr> src_strides = ktvm::arith::DetectLinearEquation(src_address.offset, src_vars);
  Array<Expr> dst_strides = ktvm::arith::DetectLinearEquation(dst_address.offset, dst_vars);
  // repeat can be viewd as a var when repeatTime is greater than 1
  if (!prove_equal(dst_address.repeatTime, 1) && !src_vars.empty() && !dst_vars.empty()) {
    auto repeat_var = Variable::make(src_vars[0].type(), "repeat_var");
    Insert(src_vars, 0, repeat_var);
    Insert(dst_vars, 0, repeat_var);
    Insert(src_strides, 0, truncdiv(src_address.repeatStride, src_address.type.bytes()));
    Insert(dst_strides, 0, truncdiv(dst_address.repeatStride, dst_address.type.bytes()));
  }
  Array<Expr> tmp_shape(src_vars.size(), Expr(1));
  SortVarShapeAndStride(src_vars, tmp_shape, src_strides);
  SortVarShapeAndStride(dst_vars, tmp_shape, dst_strides);
  if (!IsSame(src_vars, dst_vars)) {
    return false;
  }

  if (prove_equal(dst_address.repeatTime, 1)) {
    return true;
  }
  // the dst memory should be smaller than src memory
  if (!prove_equal(src_address.repeatStride, dst_address.repeatStride) &&
      prove_equal(dst_address.repeatStride, max(dst_address.repeatStride, src_address.repeatStride))) {
    return false;
  }
  // if the dst and src are with same dtype,
  // but strides are different, and not contain 0,
  // then they can not be inplace according to ISA6.3 Section 8.1.2
  if (!spec_ins && dst_address.type.bits() == src_address.type.bits() &&
      !prove_equal(src_address.repeatStride, dst_address.repeatStride) && !prove_equal(dst_address.repeatStride, 0) &&
      !prove_equal(src_address.repeatStride, 0)) {
    return false;
  }
  // do not have overlap between src and dst for more than one repeat time
  if (prove_equal(src_address.repeatStride,
                  max(src_address.repeatStride,
                      src_address.blockSize + src_address.blockStride * (src_address.blockNumber - 1)))) {
    return true;
  }
  // all the Sizes and Strides are equal
  if (prove_equal(dst_address.repeatStride, src_address.repeatStride)) {
    return true;
  }
  // rule 2, the vector must be {"vadd", "vsub", "vmul", "vmax", "vmin", "vor", "vand"},
  // and only can reuse dst and src2
  if (spec_ins && prove_equal(dst_address.repeatStride, 0) && prove_equal(src_address.repeatStride, 0)) {
    return true;
  }
  return false;
}

bool InplaceOpVerifierCCE::Check(const Node *stmt, const Variable *dst, const Variable *src) {
  dst_ = dst;
  src_ = src;
  result_ = true;
  IRVisitor::Visit(GetRef<Stmt>(static_cast<const StmtNode *>(stmt)));
  return result_;
}

void InplaceOpVerifierCCE::Visit(const NodeRef &e) {
  if (result_) {
    IRVisitor::Visit(e);
  }
}

void InplaceOpVerifierCCE::Visit_(const Variable *op) {
  if (op == dst_ || op == src_) {
    result_ = false;
  }
}

void InplaceOpVerifierCCE::Visit_(const Call *op) {
  if (op->call_type == Call::Extern) {
    if (op->name == "copy_ubuf_to_ubuf") {
      result_ = false;
      return;
    }
    MemInfo src_address, dst_address;
    this->Visit(op->args[0]);  // dst
    if (mem_info_.base != dst_) {
      result_ = false;
      return;
    }
    dst_address = mem_info_;

    bool checkSrc = false;
    int srcCount = 0;
    for (srcCount = 1; srcCount < 3; ++srcCount) {
      const Call *insn = op->args[srcCount].as<Call>();
      if (insn != nullptr) {
        this->Visit(op->args[srcCount]);
        if (mem_info_.base == src_) {
          src_address = mem_info_;
          checkSrc = true;
          break;
        }
      } else {
        break;
      }
    }
    if (!checkSrc) {
      // can not find src in vector,such as in for_body
      result_ = false;
      return;
    }

    // when overlap, the instructions that can reuse dst address and src address in limited situation,
    // only src2 and dst can reuse,such as a=b+a ,so srcCount==2
    auto it = find(reuse_intrin_name_.begin(), reuse_intrin_name_.end(), op->name);
    if (!CanReuse(src_address, dst_address, it != reuse_intrin_name_.end() && srcCount == 2)) {
      result_ = false;
      return;
    }
  } else if (op->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr)) {
    CHECK_GE(op->args.size(), 10U);
    mem_info_.offset = op->args[2];
    mem_info_.extent = op->args[3];
    mem_info_.type = op->args[0].type();
    mem_info_.repeatTime = op->args[5];
    mem_info_.repeatStride = op->args[6];
    mem_info_.blockNumber = op->args[7];
    mem_info_.blockStride = op->args[8];
    mem_info_.blockSize = op->args[9];
    mem_info_.base = op->args[1].as<Variable>();
    CHECK(mem_info_.base != nullptr);
  } else {
    IRVisitor::Visit_(op);
  }
}

void InplaceOpVerifierCCE::Visit_(const Store *op) {
  const Variable *buf = op->buffer_var.get();
  if (buf == dst_ || buf == src_) {
    result_ = false;
    return;
  }
  IRVisitor::Visit_(op);
}
void InplaceOpVerifierCCE::Visit_(const Load *op) {
  const Variable *buf = op->buffer_var.get();
  if (buf == dst_ || buf == src_) {
    result_ = false;
    return;
  }
  IRVisitor::Visit_(op);
}

void PipelineAnalyzer::Visit_(const Load *op) {
  if (cur_proc_ != nullptr) {
    const Variable *buf = op->buffer_var.get();
    AccessBuffer(buf, false);
  }
  IRVisitor::Visit_(op);
}

void PipelineAnalyzer::Visit_(const Store *op) {
  if (cur_proc_ != nullptr) {
    const Variable *buf = op->buffer_var.get();
    AccessBuffer(buf, true);
  }
  IRVisitor::Visit_(op);
}

void PipelineAnalyzer::Visit_(const Call *op) {
  if (cur_proc_ != nullptr) {
    if (op->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr)) {
      CHECK_GE(op->args.size(), 5U);
      const auto buf = op->args[1].as<Variable>();
      const auto imm = op->args[4].as<IntImm>();
      if ((buf != nullptr) && (imm != nullptr)) {
        int rw = static_cast<int>(imm->value);
        AccessBuffer(buf, rw != READ_MASK);
      }
    } else if (op->name == "set_vector_mask") {
      cur_proc_->barrier = PIPE_V;
      return;
    }
  }
  IRVisitor::Visit_(op);
}

void PipelineAnalyzer::Visit_(const AttrStmt *op) {
  if (op->attr_key == ktvm::ir::attr::coproc_scope) {
    if (!playback_) {
      std::shared_ptr<Proc> proc = std::make_shared<Proc>(next_proc_index_++);
      proc_.emplace(op, proc);
      cur_proc_ = proc.get();
      IRVisitor::Visit_(op);
      cur_proc_ = nullptr;
    } else {  // nest coproc may need it
      IRVisitor::Visit_(op);
    }
    if (op->value.as<IntImm>() != nullptr) {
      int pipe = op->value.as<IntImm>()->value % static_cast<int>(MAX_PIPE);
      AppendSpan(pipe, proc_[op].get());
      return;
    }
  }
  if (op->attr_key == ktvm::ir::attr::storage_scope && !playback_) {
    const auto buf = op->node.as<Variable>();
    buffer_.emplace(buf, Buffer{nullptr, nullptr});
  }
  IRVisitor::Visit_(op);
}

void PipelineAnalyzer::Visit_(const For *op) {
  IRVisitor::Visit_(op);
  if (cur_proc_ == nullptr) {
    bool old = playback_;
    playback_ = true;
    Visit(op->body);
    playback_ = old;
  }
}

void PipelineAnalyzer::Visit_(const IfThenElse *op) {
  if (cur_proc_ == nullptr) {
    Visit(op->then_case);
  } else {
    IRVisitor::Visit_(op);
  }
}

bool PipelineAnalyzer::PipeConflict(const Variable *buf1, const Variable *buf2) {
  auto it1 = buffer_.find(buf1);
  auto it2 = buffer_.find(buf2);
  if (it1 == buffer_.end() || it2 == buffer_.end()) {
    return false;
  }
  Buffer &b1 = it1->second;
  Buffer &b2 = it2->second;
  // inplace op
  if (b1.entry == nullptr || b2.entry == nullptr || b1.exit->index == b2.entry->index ||
      b2.exit->index == b1.entry->index) {
    return false;
  }
  std::vector<std::pair<int, int>> dom1, dom2;
  GetDomain(b1, dom1);
  GetDomain(b2, dom2);
  for (std::pair<int, int> &d1 : dom1) {
    for (std::pair<int, int> &d2 : dom2) {
      if (!(d1.first > d2.second || d2.first > d1.second)) return true;
    }
  }
  return false;
}

void PipelineAnalyzer::AccessBuffer(const Variable *buf, bool w) {
  CHECK(buf != nullptr);
  Buffer &buffer = buffer_[buf];
  if (buffer.entry == nullptr) {
    buffer.entry = cur_proc_;
    w = true;  // force first touch to be w
  }
  buffer.exit = cur_proc_;
  if (w) {
    cur_proc_->wbuf.push_back(buf);
  } else {
    cur_proc_->rbuf.push_back(buf);
  }
}

bool PipelineAnalyzer::DepBetween(const Proc *p1, const Proc *p2) {
  CHECK((p1 != nullptr) && (p2 != nullptr));
  for (const Variable *w : p1->wbuf) {
    for (const Variable *ww : p2->wbuf) {
      if (w == ww) return true;
    }
    for (const Variable *r : p2->rbuf) {
      if (w == r) return true;
    }
  }
  for (const Variable *r : p1->rbuf) {
    for (const Variable *w : p2->wbuf) {
      if (r == w) return true;
    }
  }
  return false;
}

void PipelineAnalyzer::AppendSpan(int pipe, Proc *proc) {
  if (proc == nullptr) {
    return;
  }
  if (proc->barrier != -1) {
    Barrier(pipe, proc);
    return;
  }
  int start = pipe_[pipe].empty() ? 0 : pipe_[pipe].back()->start;
  std::vector<Span *> end_span;
  for (int i = 0; i < MAX_PIPE; ++i) {
    for (auto it = pipe_[i].rbegin(); it != pipe_[i].rend(); ++it) {
      Span *s = it->get();
      if (s->end < start) break;
      if (DepBetween(proc, s->proc)) {
        if (s->end != infinite_) {
          if (s->end >= start) start = s->end + 1;
        } else {
          if (s->start >= start) start = s->start + 1;
          for (; it != pipe_[i].rend(); ++it) {
            Span *ss = it->get();
            if (ss->end != infinite_) break;
            end_span.push_back(ss);
          }
        }
        break;
      }
    }
  }
  for (Span *s : end_span) {
    s->end = start - 1;
  }
  std::shared_ptr<Span> span = std::make_shared<Span>(proc, start, infinite_);
  pipe_[pipe].emplace_back(span);
  proc->span.push_back(span.get());
}

void PipelineAnalyzer::Barrier(int pipe, Proc *proc) {
  if (pipe_[pipe].empty() || nullptr == proc) return;
  int end_time = pipe_[pipe].back()->start;
  for (auto it = pipe_[pipe].rbegin(); it != pipe_[pipe].rend(); ++it) {
    Span *s = it->get();
    if (s->end != infinite_) break;
    s->end = end_time;
  }
  int bar_start = end_time + 1;
  std::shared_ptr<Span> span = std::make_shared<Span>(proc, bar_start, bar_start);
  pipe_[pipe].emplace_back(span);
  proc->span.push_back(span.get());
}

void PipelineAnalyzer::GetDomain(const Buffer &buf, std::vector<std::pair<int, int>> &dom) {
  if (buf.entry == buf.exit) {
    for (const Span *span : buf.entry->span) {
      dom.emplace_back(std::make_pair(span->start, span->end));
    }
    return;
  }
  const std::vector<const Span *> &entry_span = buf.entry->span;
  const std::vector<const Span *> &exit_span = buf.exit->span;
  size_t entry_idx = 0;
  size_t exit_idx = 0;
  while (entry_idx < entry_span.size() && exit_idx < exit_span.size()) {
    const Span *entry = entry_span[entry_idx];
    const Span *exit = exit_span[exit_idx];
    // move exit to back
    for (entry_idx++; entry_idx < entry_span.size(); ++entry_idx) {
      const Span *next_entry = entry_span[entry_idx];
      if (next_entry->start > exit->start) {
        for (exit_idx++; exit_idx < exit_span.size() && exit_span[exit_idx]->start < next_entry->start; ++exit_idx) {
          exit = exit_span[exit_idx];
        }
        break;
      }
    }
    if (entry_idx == entry_span.size()) {
      exit = exit_span.back();
    }
    dom.emplace_back(std::make_pair(entry->start, exit->end));
  }
}

void StorageSizeDetector::Visit_(const AttrStmt *op) {
  const std::regex memlimit_regex("\\[MemoryLimit_([A-Za-z0-9]+)\\]");
  if (std::regex_match(op->attr_key, memlimit_regex)) {
    constraint_.push_back(Simplify(op->value));
  }
  IRVisitor::Visit_(op);
}

void StorageSizeDetector::init(const Stmt &s) {
  PostOrderVisit(s, [&](const NodeRef &node) {
    if (auto op = node.as<For>()) {
      loop_vars_[op->loop_var.get()] = op;
    }
  });
}

void ExpandVarsInExtentAndCond(Expr &extent, Array<Expr> &cond,
                               const std::unordered_map<const Variable *, Expr> &var_map) {
  bool found_undefined_vars = false;
  std::unordered_set<Var, NodeHash, NodeEqual> vars_in_extent, vars_in_cond;
  GatherVars(extent, &vars_in_extent);
  for (auto expr : cond) {
    GatherVars(expr, &vars_in_cond);
  }
  while (!vars_in_extent.empty()) {
    Var var = *vars_in_extent.begin();
    vars_in_extent.erase(var);
    if (vars_in_cond.count(var) > 0) continue;
    auto it = var_map.find(var.get());
    if (it == var_map.end()) {
      found_undefined_vars = true;
      continue;
    }

    std::unordered_map<const Variable *, Expr> substitute_map;
    substitute_map[var.get()] = it->second;
    extent = Substitute(extent, substitute_map);
    GatherVars(it->second, &vars_in_extent);
  }

  if (found_undefined_vars) {
    // try expand the conds
    for (auto i = 0u; i < cond.size(); ++i) {
      cond.Set(i, Substitute(cond[i], var_map));
    }
  }
}

/* Cache entry pack format:
 * Array<Expr>
 * [0]: extent
 * [1: cond.size() + 1]: cond
 * [cond.size() + 1]: result(bound)
 *
 * Assume vars_set does not affect the Inferbound result.
 */

Expr StorageSizeDetector::CachedInferBound(const Expr &extent, const Array<Expr> &var_cond, const Array<Expr> &cond,
                                           const std::unordered_set<Var, NodeHash, NodeEqual> &vars_set) {
  for (const auto &cache_entry : cached_infer_bound_) {
    if (cache_entry.size() == cond.size() + var_cond.size() + 2 && Equal(extent, cache_entry[0])) {
      bool equal = true;
      for (auto i = 0u; i < var_cond.size(); ++i) {
        if (!Equal(var_cond[i], cache_entry[i + 1])) {
          equal = false;
        }
      }
      for (auto i = 0u; i < cond.size(); ++i) {
        if (!Equal(cond[i], cache_entry[i + var_cond.size() + 1])) {
          equal = false;
        }
      }
      if (equal) {
        return cache_entry[var_cond.size() + cond.size() + 1];
      }
    }
  }

  Bound bound = InferBoundOfExprWithCond(extent, var_cond, cond, vars_set);
  Expr bound_max = GetConstIntUpBound(bound.max);

  // add to cache
  Array<Expr> cache_entry;
  cache_entry.push_back(extent);
  for (const auto &var_c : var_cond) {
    cache_entry.push_back(var_c);
  }
  for (const auto &c : cond) {
    cache_entry.push_back(c);
  }

  cache_entry.push_back(bound_max);
  cached_infer_bound_.push_back(cache_entry);
  return bound_max;
}

void StorageSizeDetector::Visit_(const Allocate *op) {
  CHECK_GE(op->constant_allocation_size(), 0) << "allocation size < 0";
  uint64_t alloc_size = static_cast<uint64_t>(op->constant_allocation_size());
  if (alloc_size == 0) {
    has_dyn_shape_ = true;
    CHECK_GT(op->extents.size(), 0);
    Expr extent = op->extents[0];
    for (size_t i = 1; i < op->extents.size(); ++i) {
      extent = extent * op->extents[i];
    }
    extent = Simplify(extent);
    Array<Expr> cond;
    Array<Expr> var_cond;
    for (auto constraint : constraint_) {
      if (constraint.as<And>()) {
        cond.push_back(constraint.as<And>()->a);
        cond.push_back(constraint.as<And>()->b);
      } else {
        cond.push_back(constraint);
      }
    }
    for (auto assert : assertions_) {
      cond.push_back(assert);
    }
    std::unordered_set<Var, NodeHash, NodeEqual> vars_set;
    std::unordered_set<Var, NodeHash, NodeEqual> vars_in_extent;
    GatherVars(extent, &vars_in_extent);
    for (auto var : vars_in_extent) {
      auto it = loop_vars_.find(var.get());
      if (it != loop_vars_.end()) {
        auto for_op = it->second;
        var_cond.push_back(for_op->loop_var >= for_op->min);
        var_cond.push_back(for_op->loop_var < Simplify(for_op->min + for_op->extent));
        vars_set.insert(for_op->loop_var);
      }
    }
    ExpandVarsInExtentAndCond(extent, cond, let_vars_);

    Array<Expr> related_cond;
    std::unordered_set<Var, NodeHash, NodeEqual> extent_vars, constraint_vars;
    GatherVars(extent, &extent_vars);
    for (auto constraint : cond) {
      GatherVars(constraint, &constraint_vars);
    }

    for (auto constraint_var : constraint_vars) {
      if (vars_set.count(constraint_var) == 0) {
        var_cond.push_back(constraint_var > make_const(constraint_var.type(), 0));
        vars_set.insert(constraint_var);
      }
    }

    bool dump_infer_bound = false;
    Expr bound_max = CachedInferBound(extent, var_cond, cond, vars_set);
    ktvm::MemoryInfo info = ktvm::GetMemoryInfo("local.UB");
    uint64_t max_num_bytes = static_cast<uint64_t>(info->max_num_bits) / 8;

    if (is_const(bound_max) && GetIntConst(bound_max) < INT_MAX && GetIntConst(bound_max) > 0) {
      alloc_size = GetIntConst(bound_max);
      if (alloc_size >= max_num_bytes / 2) {
        dump_infer_bound = true;
        LOG(INFO) << " BEGIN ============================";
        LOG(INFO) << "[WARN] Bound max of buffer " << op->buffer_var << "may be too large, please check";
      }
    } else {
      alloc_size = max_num_bytes;
      dump_infer_bound = true;
      LOG(INFO) << "BEGIN ================================";
      LOG(INFO) << "[WARN] InferBound for buffer size failed, use default size(" << alloc_size
                << "): buffer = " << op->buffer_var;
    }
    if (dump_infer_bound) {
      LOG(INFO) << "InferBound: expr: " << extent;
      LOG(INFO) << "InferBound: cond: " << cond;
      LOG(INFO) << "InferBound: bound.max: " << bound_max;
      LOG(INFO) << "END==================================";
    }
  }
  size_[op->buffer_var.get()] = alloc_size * op->type.bits() * op->type.lanes();
  IRVisitor::Visit_(op);
}

void StorageSizeDetector::Visit_(const LetStmt *op) {
  let_vars_[op->var.get()] = op->value;
  IRVisitor::Visit_(op);
}

void StorageSizeDetector::Visit_(const AssertStmt *op) {
  assertions_.push_back(op->condition);
  IRVisitor::Visit_(op);
}

std::unordered_set<const Variable *> GatherVarsInStmts(const std::vector<Stmt> &v) {
  std::unordered_set<const Variable *> vars;
  for (auto s : v) {
    PostOrderVisit(s, [&](const NodeRef &node) {
      if (auto var = node.as<Variable>()) {
        vars.insert(var);
      }
    });
  }
  return vars;
}

class PeelLetStmtsOfVarsMutator : public IRMutator {
 public:
  PeelLetStmtsOfVarsMutator(const std::unordered_set<const Variable *> &vars, std::vector<Stmt> &let_stmts)
      : vars(vars), let_stmts(let_stmts) {}
  ~PeelLetStmtsOfVarsMutator() override = default;

  Stmt Run(const Stmt &s) {
    PostOrderVisit(s, [&](const NodeRef &node) {
      if (auto let = node.as<LetStmt>()) {
        let_var_map[let->var.get()] = let->value;
      }
    });

    std::vector<Var> undefined_vars;
    for (auto var : vars) {
      auto it = let_var_map.find(var);
      if (it != let_var_map.end()) {
        GatherVars(it->second, &undefined_vars);
      }
    }
    while (!undefined_vars.empty()) {
      Var var = undefined_vars.back();
      undefined_vars.pop_back();
      if (vars.count(var.get())) continue;
      auto it = let_var_map.find(var.get());
      if (it != let_var_map.end()) {
        vars.insert(it->first);
        GatherVars(it->second, &undefined_vars);
      }
    }

    return Mutate(s);
  }

 private:
  Stmt Mutate_(const LetStmt *op, const Stmt &s) final {
    if (vars.count(op->var.get()) > 0) {
      let_stmts.emplace_back(LetStmt::make(op->var, op->value, Evaluate::make(0)));
      return IRMutator::Mutate(op->body);
    } else {
      return IRMutator::Mutate_(op, s);
    }
  };

  std::unordered_set<const Variable *> vars;
  std::unordered_map<const Variable *, Expr> let_var_map;
  std::vector<Stmt> &let_stmts;
};

class RewriteAllocateSizeToMax : public IRMutator {
 public:
  explicit RewriteAllocateSizeToMax(const std::unordered_map<const Variable *, uint64_t> &sm) : size_map(sm) {}
  ~RewriteAllocateSizeToMax() override = default;
  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Stmt ret = IRMutator::Mutate_(op, s);
    op = ret.as<Allocate>();
    CHECK(op);
    auto it = size_map.find(op->buffer_var.get());
    if (it == size_map.end()) return ret;
    Array<Expr> new_extents = {IntImm::make(Int(32), it->second)};
    return Allocate::make(op->buffer_var, op->type, new_extents, op->condition, op->body, op->new_expr,
                          op->free_function);
  }

 private:
  std::unordered_map<const Variable *, uint64_t> size_map;
};

Stmt StoragePlanRewriterCCE::Rewrite(Stmt stmt, bool is_dynamic) {
  is_dynamic_ = is_dynamic;
  Prepare(stmt);
  bool is_dynamic_rewrite = false;
  std::vector<Stmt> nest;
  for (auto &scope : scope_allocs_) {
    // if static allocation failed, try dynamic allocation
    if (!DoRewrite(scope.first, scope.second.allocs)) {
      is_dynamic_rewrite = true;
      DoDynamicRewrite(scope.first, scope.second.allocs);
      MakeAlloc(scope.first, scope.second, nest, true);
    } else {
      MakeAlloc(scope.first, scope.second, nest, false);
    }
  }
  // remove original Allocate stmts
  stmt = Mutate(stmt);

  if (is_dynamic) {
    auto vars = GatherVarsInStmts(nest);
    std::vector<Stmt> outer_let_stmts;
    stmt = PeelLetStmtsOfVarsMutator(vars, outer_let_stmts).Run(stmt);
    stmt = ktvm::ir::MergeNest(outer_let_stmts, ktvm::ir::MergeNest(nest, stmt));
    if (!is_dynamic_rewrite) {
      stmt = RewriteAllocateSizeToMax(alloc_size_).Mutate(stmt);
    }
    return stmt;
  } else {
    return ktvm::ir::MergeNest(nest, stmt);
  }
}

// Make alloc with offset
void StoragePlanRewriterCCE::MakeAlloc(const std::string &scope_name, MemScope &scope, std::vector<Stmt> &nest,
                                       bool is_dynamic_scope) {
  for (auto &e : scope.allocs) {
    for (const Allocate *a : e->allocs) {
      Expr new_offset;
      if (is_dynamic_scope) {
        auto it = dynamic_alloc_offset_.find(a);
        CHECK(it != dynamic_alloc_offset_.end()) << "dynamic allocation offset not found";
        new_offset = it->second;
      } else if (ignore_ub_ && scope_name == "local.UB") {
        new_offset = a->new_expr;
      } else {
        const int BIT_NUM_PER_BYTE = 8;
        new_offset = make_const(Int(32), (e->offset + BIT_NUM_PER_BYTE - 1) / BIT_NUM_PER_BYTE);
      }
      nest.emplace_back(
        AttrStmt::make(a->buffer_var, ktvm::ir::attr::storage_scope, StringImm::make(scope_name), Evaluate::make(0)));
      nest.emplace_back(
        Allocate::make(a->buffer_var, a->type, a->extents, a->condition, Evaluate::make(0), new_offset));
    }
  }
}

Stmt StoragePlanRewriterCCE::Mutate_(const AttrStmt *op, const Stmt &s) {
  if (op->attr_key == ktvm::ir::attr::storage_scope) {
    return Mutate(op->body);
  }
  return IRMutator::Mutate_(op, s);
}

Stmt StoragePlanRewriterCCE::Mutate_(const Allocate *op, const Stmt &s) { return Mutate(op->body); }

void StoragePlanRewriterCCE::Prepare(const Stmt stmt) {
  pipe_analyzer_.Visit(stmt);
  LivenessAnalyzer liveness;
  liveness.Analyze(stmt);
  std::unordered_set<const Variable *> inplace_flag;
  std::unordered_map<const Node *, StmtEntry *> kill_entry;
  for (StmtEntry &s : liveness.liveness_) {
    kill_entry[s.stmt] = &s;
  }
  for (StmtEntry &s : liveness.liveness_) {
    for (const Variable *var : s.gen) {
      const AllocEntry &ae = liveness.alloc_.at(var);
      StorageEntry *entry = DetectInplace(s, kill_entry[s.stmt]->kill, ae, var, inplace_flag);
      if (entry == nullptr) {
        entry = GenBuffer(ae);
      }
      entry->allocs.emplace_back(ae.alloc);
      alloc_map_[var] = entry;
    }
    for (const Variable *var : s.kill) {
      const AllocEntry &ae = liveness.alloc_.at(var);
      if (!inplace_flag.count(var)) {
        KillBuffer(var, ae);
      }
    }
  }
}

StoragePlanRewriterCCE::StorageEntry *StoragePlanRewriterCCE::DetectInplace(
  const StmtEntry &s, const std::vector<const Variable *> &kill, const AllocEntry &ae, const Variable *var,
  std::unordered_set<const Variable *> &inplace_flag) {
  StoragePlanRewriterCCE::StorageEntry *dst_entry = nullptr;
  // only one inplace var for s.stmt
  bool inplace_found = false;
  for (const Variable *src : kill) {
    if (!inplace_flag.count(src) && alloc_map_.count(src)) {
      InplaceOpVerifierCCE visitor;
      StoragePlanRewriterCCE::StorageEntry *src_entry = alloc_map_.at(src);
      uint64_t const_nbits = alloc_size_[ae.alloc->buffer_var.get()];
      if (src_entry->scope == ae.scope && !inplace_found && src_entry->size >= const_nbits &&
          visitor.Check(s.stmt, var, src)) {
        dst_entry = src_entry;
        inplace_flag.insert(src);
        inplace_found = true;
      }
    }
  }
  return dst_entry;
}

StoragePlanRewriterCCE::StorageEntry *StoragePlanRewriterCCE::GenBuffer(const AllocEntry &ae) {
  MemScope &mem_scope = scope_allocs_[ae.scope.to_string()];
  std::unique_ptr<StorageEntry> entry(new StorageEntry());
  entry->size = alloc_size_[ae.alloc->buffer_var.get()];
  entry->alloc_time = mem_scope.time++;
  entry->scope = ae.scope;
  StorageEntry *e = entry.get();
  mem_scope.allocs.emplace_back(std::move(entry));
  return e;
}

void StoragePlanRewriterCCE::KillBuffer(const Variable *buf, const AllocEntry &ae) {
  CHECK(buf != nullptr);
  MemScope &mem_scope = scope_allocs_[ae.scope.to_string()];
  StorageEntry *entry = alloc_map_[buf];
  entry->free_time = mem_scope.time++;
}

// check if e1 an e2 is pipeline conflict
bool StoragePlanRewriterCCE::PipeConflict(const StoragePlanRewriterCCE::StorageEntry *e1,
                                          const StoragePlanRewriterCCE::StorageEntry *e2) {
  if (nullptr != e1 && nullptr != e2) {
    for (const Allocate *a1 : e1->allocs) {
      for (const Allocate *a2 : e2->allocs) {
        if (pipe_analyzer_.PipeConflict(a1->buffer_var.get(), a2->buffer_var.get())) return true;
      }
    }
  }
  return false;
}

// alloc buffer in speculative ways.
// alloc memory in 3 phase:
// 1. no pipe conflict with all existing allocation
// 2. no pipe conflict with last allocation, or no reuse with buffer just freed
// 3. any memory reusable
bool StoragePlanRewriterCCE::SpecAlloc(std::list<std::shared_ptr<MemoryBound>> &outline, std::vector<AllocRecord> &his,
                                       StoragePlanRewriterCCE::StorageEntry *e, uint64_t need_nbits, int spec_level,
                                       int child_idx) {
  CHECK(e != nullptr);
  auto level1_verify = [e, this](MemoryBound *last) -> bool {
    CHECK(last != nullptr);
    return !this->PipeConflict(last->entry, e) || e->alloc_time > last->time + 1;
  };
  auto level2_verify = [e, &his, this](uint64_t offset, uint64_t extent) -> bool {
    for (AllocRecord &r : his) {
      if ((r.insert->offset + r.insert->extent > offset) && (r.insert->offset < offset + extent) &&
          this->PipeConflict(r.insert->entry, e))
        return false;
    }
    return true;
  };
  for (auto start = outline.begin(); start != outline.end(); ++start) {
    uint64_t size = 0;
    for (auto end = start; end != outline.end(); ++end) {
      std::shared_ptr<MemoryBound> last = *end;
      if (e->alloc_time < last->time || (spec_level == 1 && !level1_verify(last.get()))) break;
      size += last->extent;
      if (size < need_nbits || (spec_level == 2 && !level2_verify((*start)->offset, need_nbits))) continue;
      // if nodes after start is size enough, alloc exclude the start to
      // avoid it to be fragment memory
      if (spec_level > 0 && (size - (*start)->extent >= need_nbits)) {
        size = size - (*start)->extent;
        ++start;
      }
      e->offset = (*start)->offset;
      ++end;
      std::shared_ptr<MemoryBound> bound;
      if (size > need_nbits) {
        bound = std::make_shared<MemoryBound>(last->time, last->offset + last->extent - (size - need_nbits),
                                              size - need_nbits, last->entry);
        end = outline.insert(end, bound);
      }
      bound = std::make_shared<MemoryBound>(e->free_time, e->offset, need_nbits, e);
      end = outline.insert(end, bound);

      his.emplace_back(AllocRecord{spec_level, child_idx, size > need_nbits, bound});
      // Violation: Check whether the container is empty before accessing "his"
      if (!his.empty()) {
        AllocRecord &r = his.back();
        r.replaced.splice(r.replaced.begin(), outline, start, end);
        return true;
      }
    }
  }
  return false;
}

bool StoragePlanRewriterCCE::MultiSpecAlloc(int &spec_level, const int spec_start_idx, const int MAX_SPEC_LEVEL,
                                            uint64_t &total_alloc_bits,
                                            std::list<std::shared_ptr<MemoryBound>> &outline,
                                            std::vector<AllocRecord> &history,
                                            StoragePlanRewriterCCE::StorageEntry *entry, const uint64_t need_nbits,
                                            int &child_idx) {
  if (entry == nullptr) {
    return false;
  }
  bool success = false;
  for (int i = spec_level; i >= 0; i--) {
    success = SpecAlloc(outline, history, entry, need_nbits, i, child_idx);
    if (success) {
      if (child_idx == spec_start_idx) {
        spec_level = MAX_SPEC_LEVEL;
      }
      if (entry->offset + need_nbits > total_alloc_bits) {
        total_alloc_bits = entry->offset + need_nbits;
      }
      child_idx++;
      break;
    }
  }
  return success;
}

// Dynamic allocation for merged data
void StoragePlanRewriterCCE::DoDynamicRewrite(const std::string scope,
                                              std::vector<std::unique_ptr<StorageEntry>> &allocs) {
  // By default, align to 4 bytes.
  size_t align_bytes = 4;
  ktvm::MemoryInfo info = ktvm::GetMemoryInfo(scope);
  if (info.defined()) {
    align_bytes = info->max_simd_bits / 8;
  }

  struct MemorySlot {
    Expr size;
    int use_until{0};
  };
  std::vector<MemorySlot> memory_slots;

  ExprSimplifier simplifier;
  for (auto &alloc : allocs) {
    Expr size;
    for (auto &buf : alloc->allocs) {
      CHECK(buf->extents.size() == 1) << "buffer must be flattened";
      Expr buf_bytes = buf->extents[0];
      buf_bytes = buf_bytes * make_const(buf_bytes.type(), buf->type.bytes());
      if (!simplifier.IsDivisible(buf_bytes, make_const(Int(32), align_bytes))) {
        buf_bytes = FloorDiv::make(buf_bytes + make_const(buf_bytes.type(), align_bytes - 1),
                                   make_const(buf_bytes.type(), align_bytes)) *
                    make_const(buf_bytes.type(), align_bytes);
      }
      if (size.defined()) {
        size = size + buf_bytes;
      } else {
        size = buf_bytes;
      }
    }
    CHECK(size.defined()) << "allocs is empty";
    size = simplifier.Simplify(size);

    Expr offset = 0;
    bool allocated = false;
    for (auto &slot : memory_slots) {
      if (slot.use_until <= alloc->alloc_time && Equal(slot.size, size)) {
        CHECK(alloc->free_time > alloc->alloc_time) << "alloc time must be before free time";
        slot.use_until = alloc->free_time;
        allocated = true;
        break;
      }
      offset = offset + slot.size;
    }
    offset = simplifier.Simplify(offset);

    if (!allocated) {
      MemorySlot new_slot;
      new_slot.size = size;
      new_slot.use_until = alloc->free_time;
      memory_slots.push_back(new_slot);
    }

    // store buffer offset
    for (auto buf : alloc->allocs) {
      // because buf is "const Allocate *", we cannot change it directly, we store the offset in another map
      CHECK(dynamic_alloc_offset_.count(buf) == 0) << "duplicate allocate";
      dynamic_alloc_offset_.emplace(buf, offset);
      LOG(INFO) << "dynamic alloc: buf " << buf->buffer_var << " size " << size << " allocated at offset " << offset;

      // get next offset
      offset = simplifier.Simplify(offset + buf->extents[0] * buf->type.bytes());
    }
  }
}

// New allocation for merged data
bool StoragePlanRewriterCCE::DoRewrite(const std::string scope, std::vector<std::unique_ptr<StorageEntry>> &allocs) {
  StorageEntry *e = allocs.front().get();
  std::vector<StorageEntry *> children;
  for (size_t i = 1; i < allocs.size(); ++i) {
    children.emplace_back(allocs[i].get());
  }
  ktvm::MemoryInfo info = ktvm::GetMemoryInfo(scope);
  // By default, align to 32 bits.
  size_t align = 32;
  uint64_t max_num_bits = 1024L * 1024 * 1024 * 8;
  if (info.defined()) {
    align = info->max_simd_bits;
    max_num_bits = info->max_num_bits;
  }
  std::list<std::shared_ptr<MemoryBound>> outline;
  std::vector<AllocRecord> history;
  const int MAX_SPEC_LEVEL = 2;
  int spec_level = MAX_SPEC_LEVEL;
  int spec_start_idx = 0;
  int child_idx = -1;
  uint64_t total_alloc_bits = 0;
  int children_num = static_cast<int>(children.size());
  outline.push_back(std::make_shared<MemoryBound>(-1, 0, max_num_bits, nullptr));
  StoragePlanRewriterCCE::StorageEntry *entry = e;
  while (child_idx < children_num) {
    uint64_t need_nbits = entry->size;
    if (need_nbits % align != 0) {
      need_nbits += align - (need_nbits % align);
    }
    bool success = false;
    success = MultiSpecAlloc(spec_level, spec_start_idx, MAX_SPEC_LEVEL, total_alloc_bits, outline, history, entry,
                             need_nbits, child_idx);
    if (!success) {  // speculate rollback
      if (child_idx > spec_start_idx) {
        spec_start_idx = child_idx;
      }
      // Check whether the container is empty before accessing "history"
      while (!history.empty()) {
        AllocRecord &r = history.back();
        auto it = std::find(outline.begin(), outline.end(), r.insert);
        CHECK(it != outline.end());
        if (r.tailed) {
          it = outline.erase(it);
        }
        it = outline.erase(it);
        outline.splice(it, r.replaced);
        child_idx = r.child_idx;
        spec_level = r.spec_level;
        history.pop_back();
        if (spec_level > 0) break;
      }
      if (spec_level <= 0 || child_idx < 0) {
        if (!is_dynamic_) {
          throw MemoryAllocationException(scope, need_nbits, total_alloc_bits);
        } else {
          LOG(WARNING) << "Dynamic shape static allocation exceed bound of memory tag " << scope << ": need "
                       << need_nbits << " bits, will use dynamic allocation instead";
          return false;
        }
      }
      spec_level--;
    }
    if (child_idx == children_num) break;
    entry = children[child_idx];
  }
  return true;
}

Stmt StorageRewriteCCE(Stmt stmt, const std::string &maxsat_filename, bool use_BC_opt, bool no_limits,
                       int maxsat_timeout) {
  Stmt toRet;
  StorageSizeDetector size_detector;
  size_detector.init(stmt);
  size_detector.Visit(stmt);
  toRet = StoragePlanRewriterCCE(false, size_detector.size_).Rewrite(stmt, size_detector.has_dyn_shape_);
  return toRet;
}
}  // namespace ir
}  // namespace akg
