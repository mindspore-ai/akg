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
#include <memory>
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

class DimCollector : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == kThreadExtent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Expr extent;
      std::string name = iv->var->name_hint;
      if (name.compare(0, kBlockIdxLen, kBlockIdx) == 0) {
        block_idxs_.emplace_back(iv->var, op->value);
      } else {
        CHECK_EQ(name.compare(0, kThreadIdxLen, kThreadIdx), 0);
        thread_idxs_.emplace_back(iv->var, op->value);
      }
    }
    return IRVisitor::Visit_(op);
  }

  Var LeftBlock() {
    if (block_idxs_.empty()) {
      return Var();
    }
    return block_idxs_.back().first;
  }

  Var LeftThread() {
    if (thread_idxs_.empty()) {
      return Var();
    }
    return thread_idxs_.back().first;
  }

  friend class DimCompressor;

 private:
  std::vector<std::pair<Var, Expr>> block_idxs_;
  std::vector<std::pair<Var, Expr>> thread_idxs_;
};

class DimCompressor : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    dim_collector_.Visit(s);
    return Mutate(s);
  }
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) {
    if (op->attr_key == kThreadExtent) {
      const IterVarNode *iv = op->node.as<IterVarNode>();
      CHECK(iv);
      Expr extent;
      std::string name = iv->var->name_hint;
      bool is_left = false;
      if (name.compare(0, kBlockIdxLen, kBlockIdx) == 0) {
        is_left = LeftIdx(iv->var);
        extent = CompressIdx(dim_collector_.block_idxs_);
      } else {
        CHECK_EQ(name.compare(0, kThreadIdxLen, kThreadIdx), 0);
        is_left = LeftIdx(iv->var);
        extent = CompressIdx(dim_collector_.thread_idxs_);
      }
      if (is_left) {
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
    auto it = replace_.find(op);
    return it == replace_.end() ? e : it->second;
  }

  Var LeftBlock() { return dim_collector_.LeftBlock(); }

  Var LeftThread() { return dim_collector_.LeftThread(); }

 private:
  bool LeftIdx(const Var &var) {
    bool is_left = false;
    if (!dim_collector_.block_idxs_.empty()) {
      Var bidx = dim_collector_.block_idxs_.back().first;
      is_left = is_left || bidx.get() == var.get();
    }

    if (!dim_collector_.thread_idxs_.empty()) {
      Var tidx = dim_collector_.thread_idxs_.back().first;
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

  DimCollector dim_collector_;
  std::unordered_map<const Variable *, Expr> replace_;
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

void RemoveDimInfo(std::vector<FuncInfo> &funcs) {
  for (auto &func : funcs) {
    func.stmt = RemoveDimAttr().Mutate(func.stmt);
  }
}

void ProcessDim(std::vector<FuncInfo> &funcs, Var &block_var, Var &thread_var) {
  // 1. Make all parts with same dim, compress dim to one direction.
  for (auto &func : funcs) {
    DimCompressor dim_comp;
    func.stmt = dim_comp.Run(func.stmt);

    Var left_block = dim_comp.LeftBlock();
    Var left_thread = dim_comp.LeftThread();

    // Collect extent info to funcs;
    DimInfoVisitor dv(func, left_block, left_thread);
    dv.Visit(func.stmt);

    // Replace all variable to left one.
    std::unordered_map<const Variable *, Expr> vmap;
    if (!IsVarDefault(left_block)) {
      auto block_var_tmp = Variable::make(left_block->type, kBlockIdxX);
      vmap[left_block.get()] = block_var_tmp;
      func.block = block_var_tmp;
    }
    if (!IsVarDefault(left_thread)) {
      auto thread_var_tmp = Variable::make(left_thread->type, kThreadIdxX);
      vmap[left_thread.get()] = thread_var_tmp;
      func.thread = thread_var_tmp;
    }
    func.stmt = Substitute(func.stmt, vmap);
  }

  // 2. Unify dim var and get extent
  for (const auto &f : funcs) {
    if (!IsVarDefault(f.block)) {
      block_var = f.block;
    }
    if (!IsVarDefault(f.thread)) {
      thread_var = f.thread;
    }
  }

  for (size_t i = 0; i < funcs.size(); ++i) {
    FuncInfo &info = funcs[i];
    std::unordered_map<const Variable *, Expr> vmap;
    vmap[info.block.get()] = block_var;
    vmap[info.thread.get()] = thread_var;
    info.stmt = Substitute(info.stmt, vmap);
  }
}

void ProcessDim(std::vector<FuncInfo> &funcs, Var &block_var) {
  // Collect extent info to funcs;
  for (auto &func : funcs) {
    DimCollector dim_colector;
    dim_colector.Visit(func.stmt);

    Var left_block = dim_colector.LeftBlock();
    Var left_thread = dim_colector.LeftThread();

    // Collect extent info to funcs;
    DimInfoVisitor dv(func, left_block, left_thread);
    dv.Visit(func.stmt);

    if (!IsVarDefault(left_block)) {
      func.block = left_block;
      block_var = left_block;
    }
  }

  for (auto &func : funcs) {
    std::unordered_map<const Variable *, Expr> vmap;
    vmap[func.block.get()] = block_var;
    func.stmt = Substitute(func.stmt, vmap);
  }
}

class LowerStmtsFusion {
 public:
  LowerStmtsFusion() = default;
  ~LowerStmtsFusion() = default;

  virtual void Init(const std::vector<Stmt> &funcs) {
    funcs_.resize(funcs.size());
    for (size_t i = 0; i < funcs.size(); ++i) {
      funcs_[i].stmt = funcs[i];
    }
    VariableReset();
    initialized_ = true;
  }

  Stmt Process() {
    if (!initialized_) {
      LOG(FATAL) << "Have not been initialized!";
    }
    for (auto ft : func_transforms_) {
      ft(funcs_);
    }

    for (auto st : stmt_transforms_) {
      st(res_stmt_);
    }

    initialized_ = false;
    return res_stmt_;
  }

 protected:
  virtual void VariableReset() = 0;

  std::vector<std::function<void(std::vector<FuncInfo> &)>> func_transforms_;
  std::vector<std::function<void(Stmt &)>> stmt_transforms_;

  std::vector<FuncInfo> funcs_;
  Stmt res_stmt_;
  bool initialized_{false};
};

class LowerPipelineFusion : public LowerStmtsFusion {
 public:
  LowerPipelineFusion() = default;
  ~LowerPipelineFusion() = default;

  void Init(const std::vector<Stmt> &funcs) override {
    funcs_.resize(funcs.size());
    for (size_t i = 0; i < funcs.size(); ++i) {
      funcs_[i].origin_stmt = funcs[i];
      funcs_[i].stmt = funcs[i];
    }

    VariableReset();
    initialized_ = true;
  }

  std::vector<Stmt> Run() {
    std::vector<Stmt> res_stmts;
    res_stmts.emplace_back(Process());
    res_stmts.insert(res_stmts.end(), keep_origin_stms_.begin(), keep_origin_stms_.end());
    return res_stmts;
  }

 protected:
  void VariableReset() override { keep_origin_stms_.clear(); }

  std::vector<Stmt> keep_origin_stms_;
};

class LowerPipelineFusionGpu : public LowerPipelineFusion {
 public:
  LowerPipelineFusionGpu() {
    func_transforms_ = {
      // 0. Manager shared memory information.
      std::bind(&LowerPipelineFusionGpu::ArrangeSharedMemory, this, std::placeholders::_1),
      // 1. Compress dim to one direction and unify dim var and get extend.
      std::bind(&LowerPipelineFusionGpu::ProcessDim, this, std::placeholders::_1),
      // 3. Remove dim info
      std::bind(RemoveDimInfo, std::placeholders::_1),
      // 4. Update offset of blockIdx.x and caculate maximum.
      std::bind(&LowerPipelineFusionGpu::ProcessBlockAndThread, this, std::placeholders::_1),
      // 5. Merge ir with IfThenElse
      std::bind(&LowerPipelineFusionGpu::MergeIr, this, std::placeholders::_1),
    };
    stmt_transforms_ = {
      // Add new dim attr
      std::bind(&LowerPipelineFusionGpu::AddNewDimAttrs, this, std::placeholders::_1),
    };
  }
  ~LowerPipelineFusionGpu() = default;

 protected:
  void VariableReset() override {
    block_var_ = Var();
    thread_var_ = Var();
    max_block_num_ = 0;
    max_thread_num_ = 0;
    total_shared_memory_ = 0;
    LowerPipelineFusion::VariableReset();
  }

 private:
  void ProcessDim(std::vector<FuncInfo> &funcs) { akg::ir::ProcessDim(funcs, block_var_, thread_var_); }

  void ArrangeSharedMemory(std::vector<FuncInfo> &funcs) {
    int total_shared_memory = 0;
    for (auto &func : funcs) {
      SharedMemoryManager sm_mng;
      func.stmt = sm_mng.Mutate(func.stmt);
      // Collect the maximum shared memory among of irs.
      total_shared_memory = std::max(total_shared_memory, sm_mng.GetTotalSMSize());
    }
    total_shared_memory_ = total_shared_memory;
  }

  void ProcessBlockAndThread(std::vector<FuncInfo> &funcs) {
    // Caculate maximum of block and thread.
    max_thread_num_ = 0;
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs) {
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

  void MergeIr(std::vector<FuncInfo> &funcs) {
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
    for (const auto &func : funcs) {
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

  void AddNewDimAttrs(Stmt &stmt) {
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

  Var block_var_;
  Var thread_var_;
  size_t max_block_num_;
  size_t max_thread_num_;

  int total_shared_memory_{0};
};

class LowerPipelineFusionAscend : public LowerPipelineFusion {
 public:
  LowerPipelineFusionAscend() {
    func_transforms_ = {
      // 1. Compress dim to one direction and unify dim var and get extend.
      std::bind(&LowerPipelineFusionAscend::ProcessDim, this, std::placeholders::_1),
      // 3. Remove dim info
      std::bind(RemoveDimInfo, std::placeholders::_1),
      // 4. Update offset of blockIdx.x and caculate maximum.
      std::bind(&LowerPipelineFusionAscend::ProcessBlock, this, std::placeholders::_1),
      // 5. Merge ir with IfThenElse
      std::bind(&LowerPipelineFusionAscend::MergeIr, this, std::placeholders::_1),
    };
    stmt_transforms_ = {
      // Add new dim attr
      std::bind(&LowerPipelineFusionAscend::AddNewDimAttrs, this, std::placeholders::_1),
    };
  }
  ~LowerPipelineFusionAscend() = default;

 protected:
  void VariableReset() override {
    block_var_ = Var();
    max_block_num_ = 0;
    LowerPipelineFusion::VariableReset();
  }

 private:
  void ProcessDim(std::vector<FuncInfo> &funcs) { akg::ir::ProcessDim(funcs, block_var_); }

  void ProcessBlock(std::vector<FuncInfo> &funcs) {
    // Caculate maximum of block.
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs) {
      max_block_num_ = std::max(max_block_num_, static_cast<size_t>(func.block_ext.as<IntImm>()->value));
    }

    if (IsVarDefault(block_var_)) {
      block_var_ = Variable::make(Int(32), kBlockIdxX);
    }
  }

  void MergeIr(std::vector<FuncInfo> &funcs) {
    /*
     * Almost same as LowerPipelineFusionGpu::MergeIr, but only consider block dim.
     */
    std::vector<Stmt> stmts;
    for (const auto &func : funcs) {
      int block_num = func.block_ext.as<IntImm>()->value;

      Stmt cur_stmt = func.stmt;
      if (static_cast<int>(max_block_num_) > block_num) {
        cur_stmt = IfThenElse::make(block_var_ < block_num, cur_stmt);
      }
      stmts.emplace_back(std::move(cur_stmt));
    }

    res_stmt_ = Block::make(stmts);
  }

  void AddNewDimAttrs(Stmt &stmt) {
    Expr fusion_bx_ext = make_const(Int(32), max_block_num_);  // update it by fusion block extent
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_bx_ext), block_var_,
                                         air::IterVarType::kThreadIndex, kBlockIdxX);
    stmt = AttrStmt::make(block_iv, kThreadExtent, fusion_bx_ext, stmt);
  }

  Var block_var_;
  size_t max_block_num_;
};

class LowerBlockFusionGpu : public LowerStmtsFusion {
 public:
  LowerBlockFusionGpu() {
    func_transforms_ = {
      // 0. Manager shared memory information.
      std::bind(&LowerBlockFusionGpu::ArrangeSharedMemory, this, std::placeholders::_1),
      // 1. Compress dim to one direction and unify dim var and get extend.
      std::bind(&LowerBlockFusionGpu::ProcessDim, this, std::placeholders::_1),
      // 3. Remove dim info
      std::bind(RemoveDimInfo, std::placeholders::_1),
      // 4. Update offset of blockIdx.x and caculate maximum.
      std::bind(&LowerBlockFusionGpu::ProcessBlockAndThread, this, std::placeholders::_1),
      // 5. Merge ir with IfThenElse
      std::bind(&LowerBlockFusionGpu::MergeIr, this, std::placeholders::_1),
    };
    stmt_transforms_ = {
      // Add new dim attr
      std::bind(&LowerBlockFusionGpu::AddNewDimAttrs, this, std::placeholders::_1),
    };
  }
  ~LowerBlockFusionGpu() = default;

 private:
  void ProcessDim(std::vector<FuncInfo> &funcs) { akg::ir::ProcessDim(funcs, block_var_, thread_var_); }

  void VariableReset() override {
    block_var_ = Var();
    thread_var_ = Var();
    max_block_info_.clear();
    max_block_num_ = 0;
    max_thread_num_ = 0;
    total_shared_memory_ = 0;
  }

  void ArrangeSharedMemory(std::vector<FuncInfo> &funcs) {
    for (auto &func : funcs) {
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

  void ProcessBlockAndThread(std::vector<FuncInfo> &funcs) {
    // Update offset of blockIdx.x and caculate maximum.
    max_block_info_.clear();
    max_thread_num_ = 0;
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs) {
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
      funcs[i].stmt = BlockIndexRewrite(offset).Mutate(funcs[i].stmt);
      cur_block_num = max_block_info_[i];
    }

    if (IsVarDefault(block_var_)) {
      block_var_ = Variable::make(Int(32), kBlockIdxX);
    }
    if (IsVarDefault(thread_var_)) {
      thread_var_ = Variable::make(Int(32), kThreadIdxX);
    }
  }

  void MergeIr(std::vector<FuncInfo> &funcs) {
    //   a.update thread_overflow by comparing thread extent with final extent
    //   b.update thread condition
    //   c.update block condition
    int fthread_num = funcs.back().thread_ext.as<IntImm>()->value;
    bool thread_overflow = static_cast<int>(max_thread_num_) > fthread_num;
    Stmt res_stmt =
      thread_overflow ? IfThenElse::make(thread_var_ < fthread_num, funcs.back().stmt) : funcs.back().stmt;
    for (size_t i = funcs.size() - 1; i > 0; --i) {
      auto &func = funcs[i - 1];
      fthread_num = func.thread_ext.as<IntImm>()->value;
      thread_overflow = static_cast<int>(max_thread_num_) > fthread_num;
      Stmt stmt = thread_overflow ? IfThenElse::make(thread_var_ < fthread_num, func.stmt) : func.stmt;
      res_stmt = IfThenElse::make(block_var_ < static_cast<int>(max_block_info_[i - 1]), stmt, res_stmt);
    }
    res_stmt_ = res_stmt;
  }

  void AddNewDimAttrs(Stmt &stmt) {
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

  Var block_var_;
  Var thread_var_;
  std::vector<size_t> max_block_info_;
  size_t max_block_num_;
  size_t max_thread_num_;

  int total_shared_memory_{0};
};

class LowerBlockFusionAscend : public LowerStmtsFusion {
 public:
  LowerBlockFusionAscend() {
    func_transforms_ = {
      // 1. Compress dim to one direction and unify dim var and get extend.
      std::bind(&LowerBlockFusionAscend::ProcessDim, this, std::placeholders::_1),
      // 3. Remove dim info
      std::bind(RemoveDimInfo, std::placeholders::_1),
      // 4. Update offset of blockIdx.x and caculate maximum.
      std::bind(&LowerBlockFusionAscend::ProcessBlock, this, std::placeholders::_1),
      // 5. Merge ir with IfThenElse
      std::bind(&LowerBlockFusionAscend::MergeIr, this, std::placeholders::_1),
    };
    stmt_transforms_ = {
      // Add new dim attr
      std::bind(&LowerBlockFusionAscend::AddNewDimAttrs, this, std::placeholders::_1),
    };
  }
  ~LowerBlockFusionAscend() = default;

 private:
  void VariableReset() override {
    block_var_ = Var();
    max_block_info_.clear();
    max_block_num_ = 0;
  }

  void ProcessDim(std::vector<FuncInfo> &funcs) { akg::ir::ProcessDim(funcs, block_var_); }

  void ProcessBlock(std::vector<FuncInfo> &funcs) {
    // Update offset of blockIdx.x and caculate maximum.
    max_block_info_.clear();
    max_block_num_ = 0;
    std::vector<size_t> block_info;
    for (auto &func : funcs) {
      block_info.emplace_back(func.block_ext.as<IntImm>()->value);
    }

    for (auto it : block_info) {
      max_block_num_ += it;
      max_block_info_.emplace_back(max_block_num_);
    }

    size_t cur_block_num = 0;
    for (size_t i = 0; i < max_block_info_.size(); ++i) {
      int offset = static_cast<int>(cur_block_num);
      funcs[i].stmt = BlockIndexRewrite(offset).Mutate(funcs[i].stmt);
      cur_block_num = max_block_info_[i];
    }

    if (IsVarDefault(block_var_)) {
      block_var_ = Variable::make(Int(32), kBlockIdxX);
    }
  }

  void MergeIr(std::vector<FuncInfo> &funcs) {
    // update block condition
    Stmt res_stmt = funcs.back().stmt;
    for (size_t i = funcs.size() - 1; i > 0; --i) {
      auto &func = funcs[i - 1];
      res_stmt = IfThenElse::make(block_var_ < static_cast<int>(max_block_info_[i - 1]), func.stmt, res_stmt);
    }
    res_stmt_ = res_stmt;
  }

  void AddNewDimAttrs(Stmt &stmt) {
    Expr fusion_bx_ext = make_const(Int(32), max_block_num_);  // update it by fusion block extent
    IterVar block_iv = IterVarNode::make(Range(make_const(Int(32), 0), fusion_bx_ext), block_var_,
                                         air::IterVarType::kThreadIndex, kBlockIdxX);
    stmt = AttrStmt::make(block_iv, kThreadExtent, fusion_bx_ext, stmt);
  }

  Var block_var_;
  std::vector<size_t> max_block_info_;
  size_t max_block_num_;
};

using PipelineFusionPtr = std::shared_ptr<LowerPipelineFusion>;
using BlockFusionPtr = std::shared_ptr<LowerStmtsFusion>;

bool CheckValidToPipeLine(const Stmt &stmt) {
  auto res = EvaluateVisitor().Run(stmt);
  return res.second;
}

inline PipelineFusionPtr GetPipelineFusionByPlatform(const std::string &target) {
  if (target == "cce") {
    return std::make_shared<LowerPipelineFusionAscend>();
  } else if (target == "cuda") {
    return std::make_shared<LowerPipelineFusionGpu>();
  }

  LOG(FATAL) << "Unsupport target: " << target;
  return nullptr;
}

inline BlockFusionPtr GetBlockFusionByPlatform(const std::string &target) {
  if (target == "cce") {
    return std::make_shared<LowerBlockFusionAscend>();
  } else if (target == "cuda") {
    return std::make_shared<LowerBlockFusionGpu>();
  }

  LOG(FATAL) << "Unsupport target: " << target;
  return nullptr;
}

std::vector<Stmt> PipelineFusion(const std::vector<Stmt> &stmts, const Array<Array<NodeRef>> &pipeline_groups,
                                 const std::string &target) {
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

  auto pipeline_fusion = GetPipelineFusionByPlatform(target);

  for (const auto &sg : stmt_groups) {
    pipeline_fusion->Init(sg);
    auto pipeline_stmts = pipeline_fusion->Run();
    result_stmts.insert(result_stmts.end(), pipeline_stmts.begin(), pipeline_stmts.end());
  }
  return result_stmts;
}

Stmt BlockFusion(const std::vector<Stmt> &stmts, const std::string &target) {
  auto block_fusion = GetBlockFusionByPlatform(target);
  block_fusion->Init(stmts);
  return block_fusion->Process();
}
}  // namespace ir
}  // namespace akg
