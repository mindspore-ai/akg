/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include <functional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

#include "composite/block_fusion.h"
#include "composite/sync_process.h"

namespace akg {
namespace ir {
namespace {
constexpr auto kBlockIdx = "blockIdx.";
constexpr auto kThreadIdx = "threadIdx.";
constexpr auto kBlockIdxX = "blockIdx.x";
constexpr auto kThreadIdxX = "threadIdx.x";
constexpr auto kThreadExtent = "thread_extent";
constexpr auto kPipelineTotalSMem = "pipeline_total_shared_memory";
constexpr auto kTotalSMem = "total_shared_memory";
constexpr int kBlockIdxLen = 9;
constexpr int kThreadIdxLen = 10;
}  // namespace

struct FuncInfo {
  Stmt stmt;
  Stmt origin_stmt;
  Var block;
  std::string origin_block_name;
  Expr block_ext = make_const(Int(32), 1);
  Var thread;
  std::string origin_thread_name;
  Expr thread_ext = make_const(Int(32), 1);
};

bool IsVarDefault(const Var &var) { return var->name_hint == "v"; }

int RegularizeOffset(int offset) {
  int base = 16;
  return (offset + base - 1) / base * base;
}

class SharedMemoryManager : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == "storage_scope" && op->value.as<StringImm>()->value == "shared") {
      const Variable *v = op->node.as<Variable>();
      shared_memory_set_.insert(v);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const Variable *buffer = op->buffer_var.get();
    if (shared_memory_set_.count(buffer) != 0) {
      // Add attribute to shared memory offset.
      Expr offset_expr = IntImm::make(Int(32), total_sm_size_);
      total_sm_size_ += op->type.bytes() * op->constant_allocation_size();
      total_sm_size_ = RegularizeOffset(total_sm_size_);
      return AttrStmt::make(op->buffer_var, "shared_memory_offset", offset_expr, stmt);
    }

    return stmt;
  }

  int GetTotalSMSize() { return total_sm_size_; }

 private:
  std::set<const Variable *> shared_memory_set_;
  int total_sm_size_{0};
};

class ArrangedSharedMemoryInfo : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == kPipelineTotalSMem) {
      have_arranged_ = true;
      cur_total_shared_memory_ = op->value.as<IntImm>()->value;
    }
    IRVisitor::Visit_(op);
  }

  std::pair<bool, int> GetInfo(const Stmt &stmt) {
    Visit(stmt);
    return std::make_pair(have_arranged_, cur_total_shared_memory_);
  }

 private:
  bool have_arranged_{false};
  int cur_total_shared_memory_{0};
};

class DimCompressor : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    is_collect_ = true;
    Stmt st = Mutate(s);
    is_collect_ = false;
    return Mutate(st);
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == kThreadExtent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Expr extent;
      std::string name = iv->var->name_hint;
      bool is_left = false;
      if (name.compare(0, kBlockIdxLen, kBlockIdx) == 0) {
        if (is_collect_) {
          block_idx_.emplace_back(iv->var, op->value);
        } else {
          is_left = LeftIdx(iv->var);
          extent = CompressIdx(block_idx_);
        }
      } else {
        CHECK_EQ(name.compare(0, kThreadIdxLen, kThreadIdx), 0);
        if (is_collect_) {
          thread_idx_.emplace_back(iv->var, op->value);
        } else {
          is_left = LeftIdx(iv->var);
          extent = CompressIdx(thread_idx_);
        }
      }
      if (!is_collect_ && is_left) {
        Stmt body = IRMutator::Mutate(op->body);
        if (!extent.defined()) {
          return body;
        } else {
          return AttrStmt::make(op->node, op->attr_key, extent, body);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Variable *op, const Expr &e) {
    if (!is_collect_) {
      auto it = replace_.find(op);
      return it == replace_.end() ? e : it->second;
    } else {
      return e;
    }
  }

  Var LeftBlock() {
    if (block_idx_.empty()) {
      return Var();
    }
    return block_idx_.back().first;
  }

  Var LeftThread() {
    if (thread_idx_.empty()) {
      return Var();
    }
    return thread_idx_.back().first;
  }

 private:
  bool LeftIdx(const Var &var) {
    bool is_left = false;
    if (!block_idx_.empty()) {
      Var bidx = block_idx_.back().first;
      is_left = is_left || bidx.get() == var.get();
    }

    if (!thread_idx_.empty()) {
      Var tidx = thread_idx_.back().first;
      is_left = is_left || tidx.get() == var.get();
    }
    return is_left;
  }

  Expr CompressIdx(const std::vector<std::pair<Var, Expr>> &idx) {
    CHECK(!idx.empty()) << "idx size must be greater than 0!";
    // expected idx order: z, x, y
    Var x = idx.back().first;
    Expr dx = idx.back().second;
    size_t idx_len = idx.size();
    if (idx_len == 1) {
      return idx[0].second;
    } else if (idx_len == 2) {
      replace_.emplace(idx[0].first.get(), x / dx);
      replace_.emplace(idx[1].first.get(), truncmod(x, dx));
      return Simplify(idx[0].second * dx);
    } else {
      CHECK_EQ(idx_len, 3);
      Expr dxy = Simplify(idx[1].second * idx[2].second);
      replace_.emplace(idx[0].first.get(), x / dxy);
      replace_.emplace(idx[1].first.get(), truncmod(x, dxy) / dx);
      replace_.emplace(idx[2].first.get(), truncmod(x, dx));
      return Simplify(dxy * idx[0].second);
    }
  }

  std::vector<std::pair<Var, Expr>> block_idx_;
  std::vector<std::pair<Var, Expr>> thread_idx_;
  std::unordered_map<const Variable *, Expr> replace_;
  bool is_collect_;
};

class DimInfoVisitor : public IRVisitor {
 public:
  DimInfoVisitor(FuncInfo &info, const Var &block_var, const Var &thread_var) : info_(info) {
    if (!IsVarDefault(block_var)) {
      block_name_ = block_var->name_hint;
    }
    if (!IsVarDefault(thread_var)) {
      thread_name_ = thread_var->name_hint;
    }
  }

  void Visit_(const AttrStmt *op) {
    if (op->attr_key == kThreadExtent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      std::string name = iv->var->name_hint;
      if (name.compare(block_name_) == 0) {
        info_.block_ext = op->value;
      } else if (name.compare(thread_name_) == 0) {
        info_.thread_ext = op->value;
      }
    }
    IRVisitor::Visit_(op);
  }
  FuncInfo &info_;

 private:
  std::string block_name_{kBlockIdxX};
  std::string thread_name_{kThreadIdxX};
};
class RemoveDimAttr : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == kThreadExtent) {
      return IRMutator::Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }
};

class BlockIndexRewrite final : public IRMutator {
 public:
  explicit BlockIndexRewrite(int offset) : offset_(offset) {}
  ~BlockIndexRewrite() override = default;
  Expr Mutate_(const Variable *op, const Expr &e) {
    if (op->name_hint == kBlockIdxX && offset_ != 0) {
      return Sub::make(e, Expr(offset_));
    }
    return e;
  }
  int offset_;
};

class LowerStmtsFusion {
 public:
  explicit LowerStmtsFusion(const std::vector<Stmt> &funcs) {
    funcs_.resize(funcs.size());
    for (size_t i = 0; i < funcs.size(); ++i) {
      funcs_[i].stmt = funcs[i];
    }

    func_transforms_ = {
      // 0. Manager shared memory information.
      std::bind(&LowerStmtsFusion::ArrangeSharedMemory, this, std::placeholders::_1),
      // 1. Make all parts with same dim, compress dim to one direction.
      std::bind(&LowerStmtsFusion::CompressDim, this, std::placeholders::_1),
      // 2. Unify dim var and get extent
      std::bind(&LowerStmtsFusion::UnifyDimInfo, this, std::placeholders::_1),
      // 3. Remove dim info
      std::bind(&LowerStmtsFusion::RemoveDimInfo, this, std::placeholders::_1),
      // 4. Update offset of blockIdx.x and caculate maximum.
      std::bind(&LowerStmtsFusion::ProcessBlockAndThread, this, std::placeholders::_1),
      // 5. Merge ir with IfThenElse
      std::bind(&LowerStmtsFusion::MergeIr, this, std::placeholders::_1),
    };
    stmt_transforms_ = {
      // Add new dim attr
      std::bind(&LowerStmtsFusion::AddNewDimAttrs, this, std::placeholders::_1),
    };
  }
  ~LowerStmtsFusion() = default;

  Stmt Process() {
    for (auto ft : func_transforms_) {
      ft(funcs_);
    }

    for (auto st : stmt_transforms_) {
      st(res_stmt_);
    }

    return res_stmt_;
  }

 protected:
  virtual void ArrangeSharedMemory(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      SharedMemoryManager sm_mng;
      func.stmt = sm_mng.Mutate(func.stmt);
      // Collect the maximum shared memory among of irs.
      total_shared_memory_ = std::max(total_shared_memory_, sm_mng.GetTotalSMSize());
    }
  }

  void CompressDim(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      DimCompressor dim_comp;
      func.stmt = dim_comp.Run(func.stmt);

      Var left_block = dim_comp.LeftBlock();
      Var left_thread = dim_comp.LeftThread();

      // Collect extent info to funcs_;
      DimInfoVisitor dv(func, left_block, left_thread);
      dv.Visit(func.stmt);

      // Replace all variable to left one.
      std::unordered_map<const Variable *, Expr> vmap;
      if (!IsVarDefault(left_block)) {
        auto block_var = Variable::make(left_block->type, kBlockIdxX);
        vmap[left_block.get()] = block_var;
        func.block = block_var;
      }
      if (!IsVarDefault(left_thread)) {
        auto thread_var = Variable::make(left_thread->type, kThreadIdxX);
        vmap[left_thread.get()] = thread_var;
        func.thread = thread_var;
      }
      func.stmt = Substitute(func.stmt, vmap);
    }
  }

  void UnifyDimInfo(std::vector<FuncInfo> &funcs_) {
    for (const auto &f : funcs_) {
      if (!IsVarDefault(f.block)) {
        block_var_ = f.block;
      }
      if (!IsVarDefault(f.thread)) {
        thread_var_ = f.thread;
      }
    }

    for (size_t i = 0; i < funcs_.size(); ++i) {
      FuncInfo &info = funcs_[i];
      std::unordered_map<const Variable *, Expr> vmap;
      vmap[info.block.get()] = block_var_;
      vmap[info.thread.get()] = thread_var_;
      info.stmt = Substitute(info.stmt, vmap);
    }
  }

  void RemoveDimInfo(std::vector<FuncInfo> &funcs_) {
    for (auto &func : funcs_) {
      func.stmt = RemoveDimAttr().Mutate(func.stmt);
    }
  }

  virtual void ProcessBlockAndThread(std::vector<FuncInfo> &funcs_) = 0;
  virtual void MergeIr(std::vector<FuncInfo> &funcs) = 0;
  virtual void AddNewDimAttrs(Stmt &stmt) = 0;

  std::vector<std::function<void(std::vector<FuncInfo> &)>> func_transforms_;
  std::vector<std::function<void(Stmt &)>> stmt_transforms_;

  std::vector<FuncInfo> funcs_;
  Var block_var_;
  Var thread_var_;

  std::vector<size_t> max_block_info_;
  size_t max_block_num_;
  size_t max_thread_num_;
  Stmt res_stmt_;
  int total_shared_memory_{0};
};

class LowerPipelineFusion : public LowerStmtsFusion {
 public:
  explicit LowerPipelineFusion(const std::vector<Stmt> &funcs) : LowerStmtsFusion(funcs) {
    for (size_t i = 0; i < funcs.size(); ++i) {
      funcs_[i].origin_stmt = funcs[i];
    }
  }
  ~LowerPipelineFusion() = default;

  std::vector<Stmt> Run() {
    std::vector<Stmt> res_stmts;
    res_stmts.emplace_back(Process());
    res_stmts.insert(res_stmts.end(), keep_origin_stms_.begin(), keep_origin_stms_.end());
    return res_stmts;
  }

 private:
  void ProcessBlockAndThread(std::vector<FuncInfo> &funcs_) override {
    // Caculate maximum of block and thread.
    max_thread_num_ = 0;
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs_) {
      max_block_num_ = std::max(max_block_num_, static_cast<size_t>(func.block_ext.as<IntImm>()->value));
      max_thread_num_ = std::max(max_thread_num_, static_cast<size_t>(func.thread_ext.as<IntImm>()->value));
    }

    if (IsVarDefault(block_var_)) {
      block_var_ = Variable::make(Int(32), kBlockIdxX);
    }
    if (IsVarDefault(thread_var_)) {
      thread_var_ = Variable::make(Int(32), kThreadIdxX);
    }
  }

  void MergeIr(std::vector<FuncInfo> &funcs_) override {
    /*
     * a. add IfThenElse if segment's block num is less than maximum block num
     * b. add IfThenElse if segment's thread num is less than maximum thread num
     *
     * For example:
     * // block_num_a, thread_num_a
     * kernel A {
     *   A_STMT
     * }
     * // block_num_b, thread_num_b
     * kernel B {
     *   B_STMT
     * }
     *
     * =====================================================================
     * ##### if block_num_a < block_num_b, thread_num_a < thread_num_b #####
     * // block_num = block_num_b, thread_num = thread_num_b
     * kernel AB {
     *   if (blockidx < block_num_a) {
     *     if (threadidx < thread_num_a) {
     *       A_STMT
     *     }
     *   }
     *   B_STMT
     * }
     * =====================================================================
     * ##### if block_num_b < block_num_a, thread_num_a < thread_num_b #####
     * // block_num = block_num_a, thread_num = thread_num_b
     * kernel AB {
     *   if (threadidx < thread_num_a) {
     *     A_STMT
     *   }
     *   if (blockidx < block_num_b) {
     *     B_STMT
     *   }
     * }
     * =====================================================================
     */
    std::vector<Stmt> stmts;
    for (const auto &func : funcs_) {
      int block_num = func.block_ext.as<IntImm>()->value;
      int thread_num = func.thread_ext.as<IntImm>()->value;

      Stmt cur_stmt = func.stmt;
      if (static_cast<int>(max_thread_num_) > thread_num) {
        // Not suitable stmt will keep origin.
        auto res = EvaluateVisitor().Run(func.stmt);
        if (res.second) {
          keep_origin_stms_.emplace_back(func.origin_stmt);
          continue;
        }
        cur_stmt = IfThenElse::make(thread_var_ < thread_num, cur_stmt);
      }
      if (static_cast<int>(max_block_num_) > block_num) {
        cur_stmt = IfThenElse::make(block_var_ < block_num, cur_stmt);
      }
      stmts.emplace_back(std::move(cur_stmt));
    }

    res_stmt_ = Block::make(stmts);
  }

  void AddNewDimAttrs(Stmt &stmt) override {
    Expr fusion_bx_ext = make_const(Int(32), max_block_num_);   // update it by fusion block extent
    Expr fusion_tx_ext = make_const(Int(32), max_thread_num_);  // update it by fusion thread extent

    IterVar thread_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_tx_ext), thread_var_,
                                          air::IterVarType::kThreadIndex, kThreadIdxX);
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_bx_ext), block_var_,
                                         air::IterVarType::kThreadIndex, kBlockIdxX);
    if (total_shared_memory_ > 0) {
      stmt = AttrStmt::make(make_zero(Int(32)), kPipelineTotalSMem, IntImm::make(Int(32), total_shared_memory_), stmt);
    }
    stmt = AttrStmt::make(thread_iv, kThreadExtent, fusion_tx_ext, stmt);
    stmt = AttrStmt::make(block_iv, kThreadExtent, fusion_bx_ext, stmt);
  }

  std::vector<Stmt> keep_origin_stms_;
};

class LowerBlockFusion : public LowerStmtsFusion {
 public:
  explicit LowerBlockFusion(const std::vector<Stmt> &funcs) : LowerStmtsFusion(funcs) {}
  ~LowerBlockFusion() = default;

 private:
  void ArrangeSharedMemory(std::vector<FuncInfo> &funcs_) override {
    for (auto &func : funcs_) {
      int cur_total_sm_size = 0;
      auto sm_info = ArrangedSharedMemoryInfo().GetInfo(func.stmt);
      if (sm_info.first) {
        cur_total_sm_size = sm_info.second;
      } else {
        SharedMemoryManager sm_mng;
        func.stmt = sm_mng.Mutate(func.stmt);
        cur_total_sm_size = sm_mng.GetTotalSMSize();
      }
      // Collect the maximum shared memory among of irs.
      total_shared_memory_ = std::max(total_shared_memory_, cur_total_sm_size);
    }
  }
  void ProcessBlockAndThread(std::vector<FuncInfo> &funcs_) override {
    // Update offset of blockIdx.x and caculate maximum.
    max_block_info_.clear();
    max_thread_num_ = 0;
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs_) {
      block_info.emplace_back(func.block_ext.as<IntImm>()->value);
      max_thread_num_ = std::max(max_thread_num_, static_cast<size_t>(func.thread_ext.as<IntImm>()->value));
    }

    for (auto it : block_info) {
      max_block_num_ += it;
      max_block_info_.emplace_back(max_block_num_);
    }

    size_t cur_block_num = 0;
    for (size_t i = 0; i < max_block_info_.size(); ++i) {
      int offset = static_cast<int>(cur_block_num);
      funcs_[i].stmt = BlockIndexRewrite(offset).Mutate(funcs_[i].stmt);
      cur_block_num = max_block_info_[i];
    }

    if (IsVarDefault(block_var_)) {
      block_var_ = Variable::make(Int(32), kBlockIdxX);
    }
    if (IsVarDefault(thread_var_)) {
      thread_var_ = Variable::make(Int(32), kThreadIdxX);
    }
  }

  void MergeIr(std::vector<FuncInfo> &funcs_) override {
    //   a.update thread_overflow by comparing thread extent with final extent
    //   b.update thread condition
    //   c.update block condition
    int fthread_num = funcs_.back().thread_ext.as<IntImm>()->value;
    bool thread_overflow = static_cast<int>(max_thread_num_) > fthread_num;
    Stmt res_stmt =
      thread_overflow ? IfThenElse::make(thread_var_ < fthread_num, funcs_.back().stmt) : funcs_.back().stmt;
    for (size_t i = funcs_.size() - 1; i > 0; --i) {
      auto &func = funcs_[i - 1];
      fthread_num = func.thread_ext.as<IntImm>()->value;
      thread_overflow = static_cast<int>(max_thread_num_) > fthread_num;
      Stmt stmt = thread_overflow ? IfThenElse::make(thread_var_ < fthread_num, func.stmt) : func.stmt;
      res_stmt = IfThenElse::make(block_var_ < static_cast<int>(max_block_info_[i - 1]), stmt, res_stmt);
    }
    res_stmt_ = res_stmt;
  }

  void AddNewDimAttrs(Stmt &stmt) override {
    Expr fusion_bx_ext = make_const(Int(32), max_block_num_);   // update it by fusion block extent
    Expr fusion_tx_ext = make_const(Int(32), max_thread_num_);  // update it by fusion thread extent

    IterVar thread_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_tx_ext), thread_var_,
                                          air::IterVarType::kThreadIndex, kThreadIdxX);
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_bx_ext), block_var_,
                                         air::IterVarType::kThreadIndex, kBlockIdxX);
    if (total_shared_memory_ > 0) {
      stmt = AttrStmt::make(make_zero(Int(32)), kTotalSMem, IntImm::make(Int(32), total_shared_memory_), stmt);
    }
    stmt = AttrStmt::make(thread_iv, kThreadExtent, fusion_tx_ext, stmt);
    stmt = AttrStmt::make(block_iv, kThreadExtent, fusion_bx_ext, stmt);
  }
};

bool CheckValidToPipeLine(const Stmt &stmt) {
  auto res = EvaluateVisitor().Run(stmt);
  return res.second;
}

std::vector<Stmt> PipelineFusion(const std::vector<Stmt> &stmts, const Array<Array<NodeRef>> &pipeline_groups) {
  std::vector<std::vector<Stmt>> stmt_groups;
  stmt_groups.resize(pipeline_groups.size());
  std::set<int> visited;
  for (size_t i = 0; i < pipeline_groups.size(); ++i) {
    for (auto group_id : pipeline_groups[i]) {
      auto segment_id = group_id.as<IntImm>()->value;
      stmt_groups[i].emplace_back(stmts[segment_id]);
      visited.insert(segment_id);
    }
  }

  std::vector<Stmt> result_stmts;
  // Excluded stmts keep unchanged.
  for (size_t i = 0; i < stmts.size(); ++i) {
    if (visited.count(i) == 0) {
      result_stmts.push_back(stmts[i]);
    }
  }

  for (const auto &sg : stmt_groups) {
    auto pipeline_stmts = LowerPipelineFusion(sg).Run();
    result_stmts.insert(result_stmts.end(), pipeline_stmts.begin(), pipeline_stmts.end());
  }
  return result_stmts;
}

Stmt BlockFusion(const std::vector<Stmt> &stmts) { return LowerBlockFusion(stmts).Process(); }
}  // namespace ir
}  // namespace akg
