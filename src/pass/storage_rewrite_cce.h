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
#ifndef PASS_STORAGE_REWRITE_CCE_H_
#define PASS_STORAGE_REWRITE_CCE_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <tvm/target_info.h>
#include <tvm/arithmetic.h>
#include <pass/ir_util.h>
#include <arithmetic/compute_expr.h>
#include <utility>
#include <list>
#include <memory>
#include <vector>
#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>

#include "pass/common.h"

namespace akg {
namespace ir {
using ktvm::arith::IntSet;

class InplaceOpVerifierCCE : public IRVisitor {
 public:
  bool Check(const Node *stmt, const Variable *dst, const Variable *src);
  void Visit(const NodeRef &e) final;
  void Visit_(const Variable *op) final;
  void Visit_(const Call *op) final;
  void Visit_(const Store *op) final;
  void Visit_(const Load *op) final;

 private:
  bool CanReuse(const MemInfo &src_address, const MemInfo &dst_address, bool spec_ins) const;
  bool result_{true};
  const Variable *dst_{nullptr};
  const Variable *src_{nullptr};
  std::vector<std::string> reuse_intrin_name_ = {"vadd", "vsub", "vmul", "vmax", "vmin", "vor", "vand"};
  MemInfo mem_info_;
};

/**
 * pipeline analyzer.
 * simulate executing pipeline and find out possible buffer dependence
 */
class PipelineAnalyzer : public IRVisitor {
 public:
  enum { MAX_PIPE = 8 };

  void Visit_(const Load *op) final;
  void Visit_(const Store *op) final;
  void Visit_(const Call *op) final;
  void Visit_(const AttrStmt *op) final;
  void Visit_(const For *op) final;
  void Visit_(const IfThenElse *op) final;
  /**
   * Check if buf1 and buf2 is pipeline confilict
   */
  bool PipeConflict(const Variable *buf1, const Variable *buf2);

 private:
  struct Span;
  // a proc is coproc stmt, executed continuous in a pipe.
  struct Proc {
    explicit Proc(int i) : index(i), barrier(-1) {}
    // proc index
    int index;
    // barrier pipe
    int barrier;
    // read buffer list
    std::vector<const Variable *> rbuf;
    // write buffer list
    std::vector<const Variable *> wbuf;
    // all execution spans of this proc
    std::vector<const Span *> span;
  };

  // a span is execution interval of a pipeline
  struct Span {
    Span(Proc *p, int s, int e) : proc(p), start(s), end(e) {}
    // proc of this span
    Proc *proc;
    // start time
    int start;
    // end time
    int end;
  };

  // in general, each local buffer is start with 1 write entry and n read exit
  struct Buffer {
    const Proc *entry;
    const Proc *exit;
  };

  /**
   * add new buffer access
   */
  void AccessBuffer(const Variable *buf, bool w);

  /**
   * check if two proc is dependent. dependent is checked as if:
   * 1. Write after Read of same buffer
   * 2. Read after Write of same buffer
   * 3. Write after Write of same buffer
   */
  bool DepBetween(const Proc *p1, const Proc *p2);

  /**
   * append new span to execution spans.
   */
  void AppendSpan(int pipe, Proc *proc);

  /**
   * append pipe barrier to execution spans.
   */
  void Barrier(int pipe, Proc *proc);

  /**
   * get execution domain of buffer.
   */
  void GetDomain(const Buffer &buf, std::vector<std::pair<int, int>> &dom);

  // map of buffer var to buffer entry
  std::unordered_map<const Variable *, Buffer> buffer_;
  // map of coproc node to proc entry
  std::unordered_map<const Node *, std::shared_ptr<Proc>> proc_;
  // execution spans of all pipe
  std::vector<std::shared_ptr<Span>> pipe_[MAX_PIPE];
  // current proc
  Proc *cur_proc_ = nullptr;
  // playback mode
  bool playback_ = false;
  // next proc index, used for proc index assignment.
  int next_proc_index_ = 0;
  // infinite time
  const int infinite_ = 0x3fffffff;
};

class StorageSizeDetector : public IRVisitor {
 public:
  void init(const Stmt &s);
  void Visit_(const AttrStmt *op) final;
  void Visit_(const Allocate *op) final;
  void Visit_(const LetStmt *op) final;
  void Visit_(const AssertStmt *op) final;

  std::unordered_map<const Variable *, uint64_t> size_;
  bool has_dyn_shape_{false};

 private:
  Expr CachedInferBound(const Expr &extent, const Array<Expr> &var_cond, const Array<Expr> &cond,
                        const std::unordered_set<Var, NodeHash, NodeEqual> &vars_set);

  std::vector<Expr> constraint_;
  std::unordered_map<const Variable *, const For *> loop_vars_;
  std::unordered_map<const Variable *, Expr> let_vars_;
  std::vector<Expr> assertions_;
  std::vector<Array<Expr>> cached_infer_bound_;
};

// Planner to plan and rewrite memory allocation.
class StoragePlanRewriterCCE : public IRMutator {
 public:
  using StmtEntry = LivenessAnalyzer::StmtEntry;
  using AllocEntry = LivenessAnalyzer::AllocEntry;

  explicit StoragePlanRewriterCCE(bool ignore_ub, std::unordered_map<const Variable *, uint64_t> &alloc_size)
      : ignore_ub_(ignore_ub), alloc_size_(alloc_size) {}
  ~StoragePlanRewriterCCE() override = default;

  Stmt Rewrite(Stmt stmt, bool is_dynamic = false);

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final;
  Stmt Mutate_(const Allocate *op, const Stmt &s) final;

 private:
  struct StorageEntry {
    // scope of buffer
    StorageScope scope;
    // allocs info
    std::vector<const Allocate *> allocs;
    // The constant size of the buffer in bits
    uint64_t size{0};
    // offset of the buffer in bits
    uint64_t offset{0};
    //  The alloc time of this entry
    int alloc_time{0};
    // The free time of this entry
    int free_time{0};
  };

  // memory bound of current allocation
  struct MemoryBound {
    MemoryBound(int t, uint64_t o, uint64_t e, const StorageEntry *s) : time(t), offset(o), extent(e), entry(s) {}
    // last time for free
    int time;
    // offset of taged memory
    uint64_t offset;
    // extent of this bound
    uint64_t extent;
    // storage entry of last allocation
    const StorageEntry *entry;
  };

  // record of buffer allocation. used for speculative rollback
  struct AllocRecord {
    // speculative level of this allocation
    int spec_level;
    // child index
    int child_idx;
    // if this allocation split lastt memory bound
    bool tailed;
    // inserted memory bound node
    std::shared_ptr<MemoryBound> insert;
    // replaced memory bound node
    std::list<std::shared_ptr<MemoryBound>> replaced;
  };

  struct MemScope {
    int time{0};
    std::vector<std::unique_ptr<StorageEntry>> allocs;
  };

  void MakeAlloc(const std::string &scope_name, MemScope &scope, std::vector<Stmt> &nest, bool is_dynamic_scope);

  void Prepare(Stmt stmt);
  StorageEntry *DetectInplace(const StmtEntry &s, const std::vector<const Variable *> &kill, const AllocEntry &ae,
                              const Variable *var, std::unordered_set<const Variable *> &inplace_flag);

  StorageEntry *GenBuffer(const AllocEntry &ae);
  void KillBuffer(const Variable *buf, const AllocEntry &ae);

  // check if e1 an e2 is pipeline conflict
  bool PipeConflict(const StorageEntry *e1, const StorageEntry *e2);

  // alloc buffer in speculative ways.
  // alloc memory in 3 phase:
  // 1. no pipe conflict with all existing allocation
  // 2. no pipe conflict with last allocation, or no reuse with buffer just freed
  // 3. any memory reusable
  bool SpecAlloc(std::list<std::shared_ptr<MemoryBound>> &outline, std::vector<AllocRecord> &his, StorageEntry *e,
                 uint64_t need_nbits, int spec_level, int child_idx);

  bool MultiSpecAlloc(int &spec_level, const int spec_start_idx, const int MAX_SPEC_LEVEL, uint64_t &total_alloc_bits,
                      std::list<std::shared_ptr<MemoryBound>> &outline, std::vector<AllocRecord> &history,
                      StorageEntry *entry, const uint64_t need_nbits, int &child_idx);

  bool DoRewrite(std::string scope, std::vector<std::unique_ptr<StorageEntry>> &allocs);
  void DoDynamicRewrite(std::string scope, std::vector<std::unique_ptr<StorageEntry>> &allocs);

  std::unordered_map<std::string, MemScope> scope_allocs_;
  // The allocation assign map
  std::unordered_map<const Variable *, StorageEntry *> alloc_map_;
  // pipe analyzer to detect false dependence
  PipelineAnalyzer pipe_analyzer_;
  // ignore ub memory updae
  bool ignore_ub_;
  // alloc size of each buffer
  std::unordered_map<const Variable *, uint64_t> &alloc_size_;
  // exists dynamic shape buffer
  bool is_dynamic_{false};
  // store allocations with dynamic shapes
  std::unordered_map<const Allocate *, Expr> dynamic_alloc_offset_;
};
}  // namespace ir
}  // namespace akg
#endif  // PASS_STORAGE_REWRITE_CCE_H_
