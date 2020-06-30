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
#include <tvm/runtime/registry.h>
#include <tvm/arithmetic.h>

#include <climits>

#include "ir_pass.h"
#include "contrib/cce_parm/cceconf.h"
#include "pass/utils.h"
#include "pass/ir_util.h"
#include "pass/expr_alg_simplify.h"
#include "build_module.h"

namespace akg {
namespace ir {
constexpr auto GM_ACCESS_MIN_SIZE = 32;

class MultiCoreAccessFinder : public IRVisitor {
 public:
  explicit MultiCoreAccessFinder(ktvm::arith::ConstIntBoundAnalyzer &bound) : bound_(bound) {}
  ~MultiCoreAccessFinder() override = default;
  struct TouchEntry {
    const Variable *buf;
    Expr offset;
    int64_t extent;
    bool atomic;
    bool tail_align;
  };
  std::vector<TouchEntry> load_;
  std::vector<TouchEntry> store_;

 private:
  void Visit_(const For *op) final {
    loop_stack_.push_back(op);
    IRVisitor::Visit_(op);
    loop_stack_.pop_back();
    const auto min = op->min.as<IntImm>();
    const auto ext = op->extent.as<IntImm>();
    if (min && ext) {
      bound_.Update(Var(op->loop_var), ktvm::arith::ConstIntBound(min->value, min->value + ext->value - 1));
    }
  }

  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      local_buf_.insert(buf);
    } else if (op->attr_key == "pragma_emit_insn") {
      insn_border_ = loop_stack_.empty() ? nullptr : loop_stack_.back();
      const auto val = op->value.as<StringImm>();
      if (val && val->value == "dma_atomic_add") {
        atomic_ = true;
        IRVisitor::Visit_(op);
        atomic_ = false;
        return;
      }
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Load *op) final {
    const Variable *buf = op->buffer_var.get();
    if (local_buf_.count(buf) == 0) {
      int unit = op->type.bytes();
      load_.emplace_back(TouchEntry{buf, op->index * unit, unit, atomic_, true});
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    const Variable *buf = op->buffer_var.get();
    if (local_buf_.count(buf) == 0) {
      bool tail_align = TailAlignCheck(op);
      int unit = op->value.type().bytes();
      int64_t extent = tail_align ? unit : GM_ACCESS_MIN_SIZE;
      store_.emplace_back(TouchEntry{buf, op->index * unit, extent, atomic_, tail_align});
    }
    IRVisitor::Visit_(op);
  }

  bool TailAlignCheck(const Store *op) {
    int64_t align = 0;
    auto GetAlign = [&align](const NodeRef &node) {
      auto load = node.as<Load>();
      if (load && load->predicate.as<IntImm>()) {
        align = load->predicate.as<IntImm>()->value;
      }
    };
    PostOrderVisit(op->value, GetAlign);
    int64_t size = align;
    if (size <= 0) {
      size = 1;
      for (auto it = loop_stack_.crbegin(); it != loop_stack_.crend(); ++it) {
        const For *loop = *it;
        if (loop == insn_border_) {
          break;
        }
        const auto imm = loop->extent.as<IntImm>();
        CHECK(imm != nullptr);
        size *= imm->value;
      }
    }
    int64_t unit = op->value.type().bytes();
    return (size * unit) >= GM_ACCESS_MIN_SIZE;
  }

  std::unordered_set<const Variable *> local_buf_;
  std::vector<const For *> loop_stack_;
  const For *insn_border_{nullptr};
  ktvm::arith::ConstIntBoundAnalyzer &bound_;
  bool atomic_{false};
};

class MultiCorePlan : public IRVisitor {
 public:
  explicit MultiCorePlan(int proposal) : proposal_(proposal) {}
  ~MultiCorePlan() override = default;

  void Plan(const Stmt &stmt) {
    Visit(stmt);
    if (split_level_ >= 0 && static_cast<int>(block_axis_.size()) != split_level_) {
      block_axis_.resize(split_level_);
    }
    if (block_axis_.empty()) {
      LOG(INFO) << "No block axis found, so this operator can only use a single core.";
      return;
    }
    VerifyDataDep(stmt);
    if (block_axis_.empty()) {
      LOG(INFO) << "Detected dependency on outer axis, so this operator can only use a single core.";
      return;
    }
    GenerateBlockCoef();
    LOG(INFO) << "Set " << proposal_ << " core, actually use " << block_num_ << " core";
  }

  std::vector<std::pair<const For *, int>> block_coef_;
  int block_num_{0};

 private:
  void Visit_(const For *op) final {
    if (static_cast<int>(block_axis_.size()) != cur_level_) {
      if (split_level_ == -1 || split_level_ > cur_level_) {
        split_level_ = cur_level_;
      }
      return;
    }
    block_axis_.push_back(op);
    cur_level_++;
    IRVisitor::Visit_(op);
    cur_level_--;
  }

  void Visit_(const AttrStmt *op) final {
    // because Realize scope maybe over extension just now, we cannot determine core-local
    // code segment by storage_scope. instead, we use emit_insn scope which is not robust in some cases.
    if (op->attr_key == "pragma_emit_insn") {
      if (split_level_ == -1 || split_level_ > cur_level_) {
        split_level_ = cur_level_;
      }
      return;
    }
    if (op->attr_key == "pragma_multi_core_depth") {
      if (GetIntConst(op->value) <= global_attrs.GetIntAttr(kMultiCoreLoopMaxDepth, INT_MAX)) {
        const For *loop = op->body.as<For>();
        if (loop) {
          dep_free_axis_.insert(loop);
        }
      }
    }
    IRVisitor::Visit_(op);
  }

  using TouchEntry = MultiCoreAccessFinder::TouchEntry;

  void VerifyDataDep(const Stmt &stmt) {
    ktvm::arith::Analyzer analyzer;
    MultiCoreAccessFinder finder(analyzer.const_int_bound);
    finder.Visit(stmt);
    std::vector<TouchEntry> load = std::move(finder.load_);
    std::vector<TouchEntry> store = std::move(finder.store_);

    auto DepBetween = [&analyzer](std::vector<TouchEntry> &entry, std::vector<TouchEntry> &next,
                                  bool dep_free) -> bool {
      for (TouchEntry &e : entry) {
        for (TouchEntry &n : next) {
          if (e.buf != n.buf) continue;
          if (e.atomic && n.atomic) continue;
          if (dep_free && e.tail_align && n.tail_align) continue;
          ktvm::arith::ConstIntBound be = analyzer.const_int_bound(e.offset);
          ktvm::arith::ConstIntBound bn = analyzer.const_int_bound(n.offset);
          if ((bn->min_value >= be->min_value && bn->min_value < be->max_value + e.extent) ||
              (be->min_value >= bn->min_value && be->min_value < bn->max_value + n.extent)) {
            return true;
          }
        }
      }
      return false;
    };
    for (size_t i = 0; i < block_axis_.size(); i++) {
      const For *op = block_axis_[i];
      std::vector<TouchEntry> load_n;
      std::vector<TouchEntry> store_n;
      Map<Var, Expr> vmap;
      Map<Var, Expr> vmap_n;
      vmap.Set(Var(op->loop_var), make_const(Int(32), 0));
      vmap_n.Set(Var(op->loop_var), make_const(Int(32), 1));
      for (TouchEntry &e : load) {
        load_n.emplace_back(TouchEntry{e.buf, Substitute(e.offset, vmap_n), e.extent, e.atomic, e.tail_align});
        e.offset = Substitute(e.offset, vmap);
      }
      for (TouchEntry &e : store) {
        store_n.emplace_back(TouchEntry{e.buf, Substitute(e.offset, vmap_n), e.extent, e.atomic, e.tail_align});
        e.offset = Substitute(e.offset, vmap);
      }
      bool dep_free = dep_free_axis_.count(op) > 0;
      bool dep1 = DepBetween(store, load_n, dep_free);
      bool dep2 = DepBetween(store, store_n, dep_free);
      bool dep3 = DepBetween(load, store_n, dep_free);
      if (dep1 || dep2 || dep3) {
        block_axis_.resize(i);
        break;
      }
    }
  }

  void GenerateBlockCoef() {
    // determine block num of each level (i.e. loop depth)
    int last_coef = 1;
    for (const auto &node : block_axis_) {
      if (!node->extent.as<IntImm>()) {
        LOG(INFO) << "for now, we do not support multi-core for axis with dynamic loop bound";
        break;
      }
      int extent = static_cast<int>(node->extent.as<IntImm>()->value);
      CHECK_GE(extent, 1);
      CHECK_GE(last_coef, 1);
      int coef = proposal_ / last_coef;
      bool is_last_level = false;
      if (extent >= coef) {
        CHECK_NE(coef, 0);
        int factor = (extent + coef - 1) / coef;
        CHECK_NE(factor, 0);
        coef = (extent + factor - 1) / factor;
        is_last_level = true;
      } else if (extent * 2 > coef) {
        coef = extent;
        is_last_level = true;
      } else {  // block_num / extent >= 2
        coef = extent;
        is_last_level = (node == block_axis_.back());
      }
      block_coef_.emplace_back(std::make_pair(node, coef));
      if (is_last_level) {
        block_num_ = last_coef * coef;
        break;
      } else {
        last_coef *= coef;
        continue;
      }
    }
  }

  std::vector<const For *> block_axis_;
  int cur_level_{0};
  int split_level_{-1};
  int proposal_;
  // poly set some axis is dependence free. this info can make dep analyze more precise
  std::unordered_set<const For *> dep_free_axis_;
};

class MultiCoreInsert : public IRMutator {
 public:
  MultiCoreInsert(int block_num, std::vector<std::pair<const For *, int>> &block_coef)
      : block_num_(block_num), block_coef_(block_coef) {}
  ~MultiCoreInsert() override = default;

  Stmt Insert(Stmt stmt) {
    IterVar block_idx = ktvm::thread_axis(Range(), "blockIdx.x");
    // determine loop var replacement
    Expr this_level_iv = block_idx;
    for (int i = static_cast<int>(block_coef_.size()) - 1; i >= 0; i--) {
      const For *op = block_coef_[i].first;
      int coef = block_coef_[i].second;
      CHECK_GT(coef, 0);
      int extent = static_cast<int>(op->extent.as<IntImm>()->value);
      Expr level_idx = i > 0 ? truncmod(this_level_iv, coef) : this_level_iv;
      if (extent == coef) {
        replace_[op->loop_var.get()] = level_idx;
      } else if (extent % coef != 0) {
        replace_[op->loop_var.get()] = level_idx * (extent / coef + 1) + op->loop_var;
      } else {
        CHECK_EQ(extent % coef, 0);
        replace_[op->loop_var.get()] = level_idx * (extent / coef) + op->loop_var;
      }
      this_level_iv = Simplify_cce(truncdiv(this_level_iv, coef));
    }
    stmt = Mutate(stmt);
    return AttrStmt::make(block_idx, "thread_extent", block_num_, stmt);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &stmt) final {
    if (op->attr_key == "pragma_multi_core_depth") {
      return Mutate(op->body);
    } else if (op->attr_key == "UnalignedDMA" || op->attr_key == "gm_to_cbuf" || op->attr_key == "pragma_im2col") {
      auto attrs = Downcast<Map<std::string, Expr>>(op->node);
      std::unordered_map<std::string, Expr> new_attrs;
      for (auto kv : attrs) {
        new_attrs.emplace(std::pair<std::string, Expr>{kv.first, Mutate(kv.second)});
      }
      return AttrStmt::make(Map<std::string, Expr>(new_attrs.begin(), new_attrs.end()), op->attr_key, Mutate(op->value),
                            Mutate(op->body));
    }
    return IRMutator::Mutate_(op, stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &stmt) final {
    if (cur_layer_ < static_cast<int>(block_coef_.size())) {
      CHECK(block_coef_[cur_layer_].first == op);
      int coef = block_coef_[cur_layer_].second;
      CHECK_GT(coef, 0);
      cur_layer_++;
      CHECK(op->extent.as<IntImm>());
      int64_t extent = op->extent.as<IntImm>()->value;
      Stmt stmt_new = Mutate(op->body);
      if (extent % coef != 0) {
        Expr new_loop_var = replace_[op->loop_var.get()];
        stmt_new = IfThenElse::make(new_loop_var < op->extent, stmt_new);
        stmt_new = For::make(op->loop_var, op->min, make_const(Int(32), extent / coef + 1), op->for_type,
                             op->device_api, stmt_new);
      } else if (extent != coef) {
        stmt_new =
          For::make(op->loop_var, op->min, make_const(Int(32), extent / coef), op->for_type, op->device_api, stmt_new);
      }
      return stmt_new;
    }
    return IRMutator::Mutate_(op, stmt);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    auto it = replace_.find(op);
    return it == replace_.end() ? e : it->second;
  }

 private:
  int cur_layer_{0};
  int block_num_;
  std::vector<std::pair<const For *, int>> &block_coef_;
  std::unordered_map<const Variable *, Expr> replace_;
};

/*
 * Merge outermost loops to a single loop.
 * The user should specify attr: merge_outer_loop_for_multicore = 1
 * The user should make sure that the outermost loops have the same loop min and extent,
 * and there is no dependence among outermost loops.
 */
class MergeOuterLoop : public IRMutator {
 private:
  Stmt Mutate_(const Block *op, const Stmt &stmt) override {
    const AttrStmt *outer_attr = nullptr;
    Stmt first_stmt = op->first;
    // merge outer loops recursively, e.g. { for (x) first; for (y) rest; } rest_nomerge
    if (first_stmt.as<Block>()) {
      first_stmt = Mutate(first_stmt);
    }
    if (auto first_op = first_stmt.as<AttrStmt>()) {
      outer_attr = first_op;
      first_stmt = first_op->body;
    }
    auto first = first_stmt.as<For>();
    if (!first) return stmt;

    Stmt rest_stmt = op->rest;
    // merge outer loops recursively, e.g. first_nomerge; { for (x) first; for (y) rest; }
    if (rest_stmt.as<Block>()) {
      rest_stmt = Mutate(rest_stmt);
    }
    if (auto rest_op = rest_stmt.as<AttrStmt>()) {
      if (!outer_attr) outer_attr = rest_op;
      rest_stmt = rest_op->body;
    }
    auto rest = rest_stmt.as<For>();
    if (!rest) return stmt;

    if (!Equal(first->min, rest->min)) return stmt;
    if (!Equal(first->extent, rest->extent)) return stmt;

    Map<Var, Expr> vmap;
    vmap.Set(rest->loop_var, first->loop_var);
    auto block = Block::make(first->body, Substitute(rest->body, vmap));
    // merge nested "for" loops recursively, e.g. for (x) for (y) first; for (x) for (y) rest;
    block = Mutate(block);
    auto for_block = For::make(first->loop_var, first->min, first->extent, first->for_type, first->device_api, block);

    if (outer_attr) {
      return AttrStmt::make(outer_attr->node, outer_attr->attr_key, outer_attr->value, for_block);
    } else {
      return for_block;
    }
  }
};

/*
 * extend outermost loops to a single loop, and enable multicore
 * The user should specify attr: merge_outer_loop_for_multicore = 2
 * The user should make sure that
 * 1) the outermost loops have the same loop min var
 * 2) and there is no dependence among outermost loops.
 */
class MergeAndExtendOuterLoop : public IRMutator {
 private:
  Stmt Mutate_(const Block *op, const Stmt &stmt) override {
    const AttrStmt *outer_attr = nullptr;
    Stmt first_stmt = op->first;

    if (first_stmt.as<Block>()) {  // merge outer loops recursively
      first_stmt = MergeAndExtendOuterLoop().Mutate(first_stmt);
    }

    if (auto first_op = first_stmt.as<AttrStmt>()) {
      outer_attr = first_op;
      first_stmt = first_op->body;
    }

    auto first = first_stmt.as<For>();
    if (!first) return stmt;

    Stmt rest_stmt = op->rest;
    if (rest_stmt.as<Block>()) {  // merge outer loops recursively
      rest_stmt = MergeAndExtendOuterLoop().Mutate(rest_stmt);
    }

    if (auto rest_op = rest_stmt.as<AttrStmt>()) {
      if (!outer_attr) outer_attr = rest_op;
      rest_stmt = rest_op->body;
    }

    auto rest = rest_stmt.as<For>();
    if (!rest) return stmt;

    if (!Equal(first->min, rest->min)) return stmt;

    Stmt for_block;
    // for cases that c1 axis is bigger than 32, just merge outerloop and do not extent them.
    auto rest_upper = rest->extent.as<IntImm>();
    auto rest_lower = rest->min.as<IntImm>();
    if (!rest_upper || !rest_lower) return stmt;

    if (rest_upper->value - rest_lower->value >= 32) {
      if (!Equal(first->extent, rest->extent)) return stmt;
      Map<Var, Expr> vmap;
      vmap.Set(rest->loop_var, first->loop_var);
      auto block = Block::make(first->body, Substitute(rest->body, vmap));
      for_block = For::make(first->loop_var, first->min, first->extent, first->for_type, first->device_api, block);
    } else {
      Map<Var, Expr> vmap;
      vmap.Set(rest->loop_var, first->loop_var - first->extent + rest->min);
      Expr condition = first->loop_var < first->min + first->extent;
      auto block = IfThenElse::make(condition, first->body, Substitute(rest->body, vmap));
      for_block =
        For::make(first->loop_var, first->min, first->extent + rest->extent, first->for_type, first->device_api, block);
    }

    if (outer_attr) {
      return AttrStmt::make(outer_attr->node, outer_attr->attr_key, outer_attr->value, for_block);
    } else {
      return for_block;
    }
  }
};

class MultiCorePartitioner : public IRMutator {
 public:
  MultiCorePartitioner() = default;
  ~MultiCorePartitioner() override = default;

  Stmt Partition(Stmt stmt) {
    const auto op = stmt.as<AttrStmt>();
    if (!op || op->attr_key != "thread_extent") return stmt;
    CHECK(op->node.as<IterVarNode>());
    CHECK(op->value.as<IntImm>());
    block_idx_ = op->node.as<IterVarNode>()->var;
    block_num_ = static_cast<int>(op->value.as<IntImm>()->value);
    stmt = Mutate(stmt);
    return ktvm::ir::ConvertSSA(stmt);
  }

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    std::vector<std::pair<int64_t, Stmt>> segments;
    Expr f_block;
    if (!op->extent.as<IntImm>() || !ExtractFblock(op, f_block) || !ExtractSegments(op, segments)) {
      return s;
    }
    std::vector<std::pair<Expr, Stmt>> branch_stmts;
    int64_t stride_len = op->extent.as<IntImm>()->value;
    int64_t loop_offset = 0;
    int64_t fblock_offset = 0;
    int64_t stride = 0;
    auto GenBranchStmt = [&op, &stride](Stmt stmt, int64_t extent) -> Stmt {
      std::unordered_map<const Variable *, Expr> vmap;
      if (extent == 1) {
        vmap.emplace(op->loop_var.get(), make_const(Int(32), stride));
        return Substitute(stmt, vmap);
      }
      if (stride > 0) {
        vmap.emplace(op->loop_var.get(), op->loop_var + make_const(Int(32), stride));
        stmt = Substitute(stmt, vmap);
      }
      return For::make(op->loop_var, make_const(Int(32), 0), make_const(Int(32), extent), ForType::Serial,
                       DeviceAPI::None, stmt);
    };
    size_t seg_idx = 0;
    while (seg_idx < segments.size()) {
      auto &seg = segments[seg_idx++];
      Stmt stmt;
      if (stride == 0) {
        if (seg.first >= stride_len) {
          CHECK_NE(stride_len, 0);
          if ((seg.first > stride_len) && (seg.first % stride_len > 0)) seg_idx--;
          stmt = GenBranchStmt(seg.second, stride_len);
          loop_offset += seg.first - seg.first % stride_len;
          fblock_offset += seg.first / stride_len;
          seg.first -= seg.first - seg.first % stride_len;
        } else {
          stmt = GenBranchStmt(seg.second, seg.first);
          loop_offset += seg.first;
          fblock_offset += 1;
          stride = seg.first;
          seg.first = 0;
        }
        branch_stmts.emplace_back(std::make_pair(f_block < make_const(Int(32), fblock_offset), stmt));
      } else {
        if (seg.first >= stride_len - stride) {
          if (seg.first > stride_len - stride) seg_idx--;
          stmt = GenBranchStmt(seg.second, stride_len - stride);
          loop_offset += stride_len - stride;
          seg.first -= stride_len - stride;
          stride = 0;
        } else {
          stmt = GenBranchStmt(seg.second, seg.first);
          loop_offset += seg.first;
          stride += seg.first;
          seg.first = 0;
        }
        auto &cur_stmt = branch_stmts.back().second;
        cur_stmt = Block::make(cur_stmt, stmt);
      }
    }
    CHECK(!branch_stmts.empty());
    Stmt block_branch = branch_stmts.back().second;
    for (size_t i = branch_stmts.size() - 1; i > 0; --i) {
      auto &branch = branch_stmts[i - 1];
      block_branch = IfThenElse::make(branch.first, branch.second, block_branch);
    }
    return block_branch;
  }

  bool ExtractFblock(const For *op, Expr &f_block) {
    const auto body = op->body.as<IfThenElse>();
    if (!body) return false;
    const auto lt = body->condition.as<LT>();
    if (!lt) return false;

    bool exist = false;
    PostOrderVisit(lt->a, [&exist, this](const NodeRef &node) {
      if (node.as<Variable>() == this->block_idx_.get()) {
        exist = true;
      }
    });
    if (!exist) return false;

    CHECK(!Equal(op->extent, 0));
    f_block = Simplify((lt->a - op->loop_var) / op->extent);
    return true;
  }

  bool ExtractSegments(const For *op, std::vector<std::pair<int64_t, Stmt>> &segments) {
    Stmt seg_stmt = op->body;
    const auto seg = seg_stmt.as<IfThenElse>();
    if (!seg) return false;

    int64_t loop_extent = 0;
    if (!seg->else_case.defined()) {
      const auto lt = seg->condition.as<LT>();
      if (!lt) return false;
      const auto imm = lt->b.as<IntImm>();
      if (!imm) return false;
      loop_extent = imm->value;
      seg_stmt = seg->then_case;
    } else {
      int fblock_num = block_num_;
      PostOrderVisit(seg->condition, [&fblock_num](const NodeRef &node) {
        const auto mod = node.as<Mod>();
        if (mod) {
          const auto imm = mod->b.as<IntImm>();
          if (imm) {
            fblock_num = imm->value;
          }
        }
      });
      CHECK(op->extent.as<IntImm>());
      loop_extent = fblock_num * op->extent.as<IntImm>()->value;
    }
    int64_t offset = 0;
    while (seg_stmt.as<IfThenElse>()) {
      const auto if_then_else = seg_stmt.as<IfThenElse>();
      const auto lt = if_then_else->condition.as<LT>();
      if (!lt) return false;
      const auto imm = lt->b.as<IntImm>();
      if (!imm) return false;
      segments.emplace_back(std::make_pair(imm->value - offset, if_then_else->then_case));
      offset = imm->value;
      seg_stmt = if_then_else->else_case;
    }
    segments.emplace_back(std::make_pair(loop_extent - offset, seg_stmt));
    return true;
  }

  int block_num_{0};
  Var block_idx_;
};

class LoopCompounder : public IRMutator {
 public:
  using TouchEntry = MultiCoreAccessFinder::TouchEntry;
  using SegStmt = std::pair<const For *, std::vector<Stmt>>;

  explicit LoopCompounder(int64_t proposal) : proposal_(proposal), block_size_(1) {}
  ~LoopCompounder() override = default;

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    const auto *extent = op->extent.as<IntImm>();
    if (extent != nullptr) {
      block_size_ *= extent->value;
      if (block_size_ >= proposal_) {
        return s;
      }
    }
    loop_stack_.emplace_back(op);
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      return s;  // local scope
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    std::vector<Stmt> stmts;
    UnrollSeq(op, stmts);
    auto seg_stmts = SplitSeq(stmts);
    if (seg_stmts.size() <= 1 || SegmentDepend(seg_stmts)) {
      return s;
    }
    std::vector<Expr> seg_offset;
    Expr offset = make_const(Int(32), 0);
    for (auto &seg : seg_stmts) {
      seg_offset.emplace_back(offset);
      offset = offset + (seg.first ? seg.first->extent : make_const(Int(32), 1));
    }
    Var loop_var("comp_idx", Int(32));
    Stmt res_stmt;
    for (size_t i = seg_stmts.size(); i > 0; --i) {
      auto &seg = seg_stmts[i - 1];
      Stmt seg_s = ktvm::ir::MergeSeq(seg.second);
      if (seg.first) {
        std::unordered_map<const Variable *, Expr> vmap;
        vmap.emplace(seg.first->loop_var.get(), loop_var - seg_offset[i - 1]);
        seg_s = Substitute(seg_s, vmap);
      }
      res_stmt = res_stmt.defined() ? IfThenElse::make(loop_var < seg_offset[i], seg_s, res_stmt) : seg_s;
    }
    res_stmt = For::make(loop_var, make_const(Int(32), 0), offset, ForType::Serial, DeviceAPI::None, res_stmt);
    return AttrStmt::make(StringImm::make("comp_block"), "pragma_multi_core_depth", make_const(Int(32), 0), res_stmt);
  }

  void UnrollSeq(const Block *op, std::vector<Stmt> &stmts) {
    if (op->first.as<Block>()) {
      UnrollSeq(op->first.as<Block>(), stmts);
    } else {
      stmts.push_back(op->first);
    }
    if (op->rest.as<Block>()) {
      UnrollSeq(op->rest.as<Block>(), stmts);
    } else {
      stmts.push_back(op->rest);
    }
  }

  std::vector<SegStmt> SplitSeq(std::vector<Stmt> &stmts) {
    std::vector<SegStmt> seg_stmt;
    std::vector<Stmt> current;
    for (Stmt &stmt : stmts) {
      const auto attr = stmt.as<AttrStmt>();
      if (attr && attr->attr_key == "pragma_multi_core_depth" && attr->body.as<For>()) {
        const For *loop = attr->body.as<For>();
        if (!current.empty()) {
          seg_stmt.emplace_back(nullptr, std::vector<Stmt>());
          std::swap(seg_stmt.back().second, current);
        }
        seg_stmt.emplace_back(loop, std::vector<Stmt>());
        seg_stmt.back().second.emplace_back(loop->body);
      } else {
        current.emplace_back(stmt);
      }
    }
    if (!current.empty()) {
      seg_stmt.emplace_back(nullptr, std::vector<Stmt>());
      std::swap(seg_stmt.back().second, current);
    }
    return seg_stmt;
  }

  bool SegmentDepend(std::vector<SegStmt> &seg_stmts) {
    ktvm::arith::Analyzer analyzer;
    struct SegAccess {
      std::vector<TouchEntry> loads;
      std::vector<TouchEntry> stores;
    };
    std::vector<SegAccess> access(seg_stmts.size());
    for (size_t i = 0; i < seg_stmts.size(); ++i) {
      auto &seg = seg_stmts[i];
      if (seg.first) {
        const auto min = seg.first->min.as<IntImm>();
        const auto ext = seg.first->extent.as<IntImm>();
        if (min && ext) {
          analyzer.const_int_bound.Update(Var(seg.first->loop_var),
                                          ktvm::arith::ConstIntBound(min->value, min->value + ext->value - 1));
        }
      }
      for (Stmt &stmt : seg.second) {
        MultiCoreAccessFinder finder(analyzer.const_int_bound);
        finder.Visit(stmt);
        access[i].loads.insert(access[i].loads.end(), finder.load_.begin(), finder.load_.end());
        access[i].stores.insert(access[i].stores.end(), finder.store_.begin(), finder.store_.end());
      }
    }
    std::unordered_map<const Variable *, Expr> vmap;
    for (const For *loop : loop_stack_) {
      vmap[loop->loop_var.get()] = loop->min;
    }
    for (auto &acc : access) {
      for (TouchEntry &e : acc.loads) {
        e.offset = Substitute(e.offset, vmap);
      }
      for (TouchEntry &e : acc.stores) {
        e.offset = Substitute(e.offset, vmap);
      }
    }
    auto DepBetween = [&analyzer](std::vector<TouchEntry> &first, std::vector<TouchEntry> &second) -> bool {
      for (TouchEntry &t1 : first) {
        for (TouchEntry &t2 : second) {
          if ((t1.buf != t2.buf) || (t1.atomic && t2.atomic)) continue;
          ktvm::arith::ConstIntBound b1 = analyzer.const_int_bound(t1.offset);
          ktvm::arith::ConstIntBound b2 = analyzer.const_int_bound(t2.offset);
          if ((b2->min_value >= b1->min_value && b2->min_value < b1->max_value + t1.extent) ||
              (b1->min_value >= b2->min_value && b1->min_value < b2->max_value + t2.extent)) {
            return true;
          }
        }
      }
      return false;
    };
    for (size_t i = 0; i < seg_stmts.size(); ++i) {
      for (size_t j = i + 1; j < seg_stmts.size(); ++j) {
        if (seg_stmts[i].first && seg_stmts[j].first) {
          continue;
        }
        if (DepBetween(access[i].loads, access[j].stores) || DepBetween(access[i].stores, access[j].loads) ||
            DepBetween(access[i].stores, access[j].stores)) {
          return true;
        }
      }
    }
    return false;
  }

  int64_t proposal_;
  int64_t block_size_;
  std::vector<const For *> loop_stack_;
};

class LoopUnCompunder : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_multi_core_depth") {
      auto node = op->node.as<StringImm>();
      if (node && node->value == "comp_block") {
        in_mc_scope_ = true;
        Stmt stmt = IRMutator::Mutate_(op, s);
        in_mc_scope_ = false;
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (in_mc_scope_) {
      in_mc_scope_ = false;
      if (op->loop_var->name_hint == "comp_idx") {
        std::vector<std::pair<int64_t, Stmt>> seg_stmts;
        ExtractSegments(op->body, 0, seg_stmts);
        CHECK(op->extent.as<IntImm>());
        std::unordered_map<const Variable *, Expr> vmap;
        std::vector<Stmt> seq;
        for (size_t i = 0; i < seg_stmts.size(); ++i) {
          int64_t extent = i + 1 < seg_stmts.size() ? seg_stmts[i + 1].first : op->extent.as<IntImm>()->value;
          CHECK(extent > seg_stmts[i].first);
          if ((extent - seg_stmts[i].first) == 1) {
            vmap[op->loop_var.get()] = make_const(Int(32), seg_stmts[i].first);
            seq.emplace_back(Substitute(seg_stmts[i].second, vmap));
          } else {
            Var loop_var("i", Int(32));
            vmap[op->loop_var.get()] = loop_var + make_const(Int(32), seg_stmts[i].first);
            Stmt res_stmt = Substitute(seg_stmts[i].second, vmap);
            res_stmt = For::make(loop_var, make_const(Int(32), 0), make_const(Int(32), extent - seg_stmts[i].first),
                                 ForType::Serial, DeviceAPI::None, res_stmt);
            seq.emplace_back(res_stmt);
          }
        }
        return ktvm::ir::MergeSeq(seq);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    return s;  // stop here
  }

  void ExtractSegments(const Stmt &stmt, int64_t offset, std::vector<std::pair<int64_t, Stmt>> &seg_stmt) {
    const auto op = stmt.as<IfThenElse>();
    if (op && op->condition.as<LT>()) {
      auto else_offset = op->condition.as<LT>()->b.as<IntImm>();
      CHECK(else_offset);
      seg_stmt.emplace_back(offset, op->then_case);
      ExtractSegments(op->else_case, else_offset->value, seg_stmt);
    } else {
      seg_stmt.emplace_back(offset, stmt);
    }
  }

  bool in_mc_scope_{false};
};

class DynamicShapeMulticoreLoopsFinder : public IRVisitor {
 public:
  DynamicShapeMulticoreLoopsFinder() = default;
  ~DynamicShapeMulticoreLoopsFinder() override = default;

  void Find(Stmt stmt) {
    Visit(stmt);
    if (multicore_loops_.empty()) return;
    bool can_mc = false;
    Expr mc_cond;
    std::tie(can_mc, mc_cond) = VerifyDataDep();
    if (!can_mc) {
      LOG(INFO) << "cannot use multi-core due to data dependency";
      multicore_loops_.clear();
    } else {
      if (mc_cond.defined()) {
        LOG(INFO) << "enabled multi-core with cond: " << mc_cond;
      } else {
        LOG(INFO) << "enabled multi-core";
      }
      multicore_cond_ = mc_cond;
    }
  }

  std::vector<const For *> multicore_loops_;
  Expr multicore_cond_;

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == ktvm::ir::attr::storage_scope) {
      const auto buf = op->node.as<Variable>();
      local_buf_.insert(buf);
    } else if (op->attr_key == "pragma_multi_core_depth") {
      if (!find_mc_loop_done_ && GetIntConst(op->value) <= global_attrs.GetIntAttr(kMultiCoreLoopMaxDepth, INT_MAX)) {
        mc_depth_on_ = true;
      }
    } else if (op->attr_key == "pragma_emit_insn") {
      insn_border_ = loop_stack_.empty() ? nullptr : loop_stack_.back();
    } else if (op->attr_key == "[MemoryLimit_UB]") {
      Expr constraint = Simplify(op->value);
      constraints_.push_back(constraint);
      auto FindVar = [this](const NodeRef &node) {
        if (node.as<Variable>()) {
          this->constraint_var_.insert(Downcast<Var>(node));
        }
      };
      PostOrderVisit(constraint, FindVar);
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const IfThenElse *op) final {
    if (!find_mc_loop_done_) {
      // only the then case can use multi-core
      Visit(op->then_case);
    } else {
      if_conditions_.push_back(op->condition);
      Visit(op->then_case);
      if_conditions_.pop_back();

      if (op->else_case.defined()) {
        if_conditions_.push_back(Not::make(op->condition));
        Visit(op->else_case);
        if_conditions_.pop_back();
      }
    }
  }

  void Visit_(const Block *op) final {
    if (!find_mc_loop_done_) {
      // only the first block can use multi-core
      Visit(op->first);
    } else {
      IRVisitor::Visit_(op);
    }
  }

  void Visit_(const For *op) final {
    if (mc_depth_on_) {
      multicore_loops_.push_back(op);
      mc_depth_on_ = false;
    }
    loop_stack_.push_back(op);
    IRVisitor::Visit_(op);
    loop_stack_.pop_back();
  }

  void Visit_(const Load *op) final {
    const Variable *buf = op->buffer_var.get();
    if (local_buf_.count(buf) == 0) {
      load_.emplace_back(TouchEntry{buf, true});
    }
    IRVisitor::Visit_(op);
  }

  void Visit_(const Store *op) final {
    const Variable *buf = op->buffer_var.get();
    if (local_buf_.count(buf) == 0) {
      int64_t align = 0;
      auto GetAlign = [&align](const NodeRef &node) {
        auto load = node.as<Load>();
        if (load && load->predicate.as<IntImm>()) {
          align = load->predicate.as<IntImm>()->value;
        }
      };
      PostOrderVisit(op->value, GetAlign);
      bool tail_align = false;
      Expr cond_tail_align;
      if (align > 0) {
        tail_align = align * op->value.type().bytes() >= GM_ACCESS_MIN_SIZE;
        if (!tail_align) {
          LOG(INFO) << "buffer " << buf->name_hint << " alignment is to small: allignment " << align << " ("
                    << align * op->value.type().bytes() << " bytes)";
        }
      } else if (!loop_stack_.empty()) {
        const For *continous_loop = nullptr;
        for (auto loop : loop_stack_) {
          Expr coef = GetLinearCoefOfVar(op->index, loop->loop_var);
          if (coef.defined() && Equal(coef, 1)) {
            continous_loop = loop;
          }
        }
        if (!continous_loop) {
          LOG(WARNING) << "cannot find continous loop from index: " << op->index;
          continous_loop = loop_stack_.back();
        }

        if (continous_loop != insn_border_ && IsVarInExpr(continous_loop->loop_var, op->index)) {
          Expr extent = continous_loop->extent;
          CHECK_GT(op->value.type().bytes(), 0);
          auto const_align = ProveTailAlign(extent, GM_ACCESS_MIN_SIZE / op->value.type().bytes());
          if (const_align.first) {  // can prove aligned or unaligned
            tail_align = const_align.second;
            if (!tail_align) {
              LOG(INFO) << "buffer " << buf->name_hint << " is proved to be unaligned: extent " << extent << " in loop "
                        << continous_loop->loop_var;
            }
          } else {  // cannot prove, need to add a runtime condition
            std::unordered_set<const Variable *> touch_var;
            PostOrderVisit(extent, [&touch_var](const NodeRef &node) {
              auto var = node.as<Variable>();
              if (var && !touch_var.count(var)) touch_var.insert(var);
            });
            if (std::find_if(loop_stack_.begin(), loop_stack_.end(), [&touch_var](const For *l) {
                  return touch_var.count(l->loop_var.get());
                }) == loop_stack_.end()) {
              CHECK_NE(op->value.type().bytes(), 0);
              cond_tail_align = (extent >= (GM_ACCESS_MIN_SIZE / op->value.type().bytes()));
            } else {
              LOG(INFO) << "buffer " << buf->name_hint << " alignment is " << align
                        << ", but continuous access condition cannot be determined";
            }
          }
        } else {
          LOG(INFO) << "buffer " << buf->name_hint << " alignment is " << align
                    << ", but there is no loop that accesses continuous buffer";
        }
      }

      if (cond_tail_align.defined() && !if_conditions_.empty()) {
        Expr store_cond;
        for (const auto &cond : if_conditions_) {
          if (IsExprContainLoopVars(cond)) continue;

          if (!store_cond.defined())
            store_cond = cond;
          else
            store_cond = store_cond && cond;
        }
        if (store_cond.defined()) {
          cond_tail_align = Simplify(Not::make(store_cond) || cond_tail_align);
        }
      }
      store_.emplace_back(TouchEntry{buf, tail_align, cond_tail_align});
    }
    IRVisitor::Visit_(op);
  }

  void Visit(const NodeRef &node) final {
    if (!find_mc_loop_done_ && node.as<StmtNode>()) {
      auto McFriendly = [&node]() -> bool {
        return (node.as<For>() || node.as<LetStmt>() || node.as<AttrStmt>() || node.as<AssertStmt>() ||
                node.as<Block>() || node.as<IfThenElse>());
      };
      if (!McFriendly()) {
        find_mc_loop_done_ = true;
        mc_depth_on_ = false;
      }
    }
    IRVisitor::Visit(node);
  }

  std::pair<bool, Expr> VerifyDataDep() {
    Expr cond;
    for (auto &s : store_) {
      for (auto &l : load_) {
        if (s.buf == l.buf) {
          LOG(INFO) << "found load-store dependency on buffer " << s.buf->name_hint;
          return std::make_pair(false, Expr());
        }
      }
      if (s.tail_align) continue;
      if (!s.cond_tail_align.defined()) {
        LOG(INFO) << "tail is unaligned, and condition is not found: " << s.buf->name_hint;
        return std::make_pair(false, Expr());
      }
      cond = cond.defined() ? cond && s.cond_tail_align : s.cond_tail_align;
    }
    return std::make_pair(true, cond);
  }

  std::pair<bool, bool> ProveTailAlign(const Expr &extent, int64_t min_align) {
    if (is_const(extent)) {
      return std::make_pair(true, GetIntConst(extent) >= min_align);
    }
    Array<Expr> cond;
    Array<Expr> var_cond;
    auto var_set = constraint_var_;
    if (!constraints_.empty()) {
      for (const auto &e : constraints_) {
        cond.push_back(e);
      }
    }
    for (const For *loop : loop_stack_) {
      var_cond.push_back(loop->loop_var >= loop->min);
      var_cond.push_back(loop->loop_var < loop->extent);
      var_set.insert(Downcast<Var>(loop->loop_var));
    }
    Bound bound = InferBoundOfExprWithCond(extent, var_cond, cond, var_set);
    Expr bound_min = GetConstIntLowBound(bound.min);
    Expr bound_max = GetConstIntUpBound(bound.max);
    ExprSimplifier simplifier;
    if (simplifier.IsDivisible(extent, min_align) || (is_const(bound_min) && GetIntConst(bound_min) >= min_align)) {
      return std::make_pair(true, true);
    } else {
      return std::make_pair(is_const(bound_max) && GetIntConst(bound_max) < min_align, false);
    }
  }

  bool IsExprContainLoopVars(const Expr &cond) {
    std::unordered_set<Var, NodeHash, NodeEqual> vars;
    GatherVars(cond, &vars);
    for (auto loop : loop_stack_) {
      if (vars.count(loop->loop_var) > 0) {
        return true;
      }
    }
    return false;
  }

  struct TouchEntry {
    const Variable *buf;
    bool tail_align;
    Expr cond_tail_align;
  };
  std::vector<TouchEntry> load_;
  std::vector<TouchEntry> store_;

  std::unordered_set<const Variable *> local_buf_;
  std::vector<const For *> loop_stack_;

  const For *insn_border_{nullptr};
  bool find_mc_loop_done_{false};
  bool mc_depth_on_{false};

  std::vector<Expr> constraints_;
  std::unordered_set<Var, NodeHash, NodeEqual> constraint_var_;

  std::vector<Expr> if_conditions_;
};

class DynamicShapeMulticoreInsert : public IRMutator {
 public:
  DynamicShapeMulticoreInsert(const std::vector<const For *> &multicore_loops, Expr multicore_cond,
                              const IterVar &block_idx, const Expr &block_num)
      : multicore_loops_(multicore_loops),
        multicore_cond_(std::move(multicore_cond)),
        block_idx_(block_idx),
        total_iters_(1) {
    for (auto loop : multicore_loops_) {
      total_iters_ = total_iters_ * loop->extent;
    }
    total_iters_ = Simplify(total_iters_);
    // there is bug in Simplify with floordiv!
    CHECK(!Equal(block_num, 0));
    iters_per_core_ = floordiv(total_iters_ + Expr(block_num - 1), Expr(block_num));
  }
  ~DynamicShapeMulticoreInsert() override = default;

  Stmt Insert(const Stmt &stmt) {
    Stmt mc_stmt = Mutate(stmt);
    if (multicore_cond_.defined()) {
      mc_stmt = IfThenElse::make(multicore_cond_, mc_stmt, stmt);
      mc_stmt = ktvm::ir::ConvertSSA(mc_stmt);
    }
    return mc_stmt;
  }

 private:
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (curr_index_ >= multicore_loops_.size()) return s;  // inner body

    bool is_outermost = (curr_index_ == 0);
    if (multicore_loops_.size() == 1) {  // only one loop that can use multicore
      Expr outermost_loop_min = iters_per_core_ * block_idx_;
      Expr outermost_loop_extent = Min::make(iters_per_core_, total_iters_ - outermost_loop_min);
      return For::make(op->loop_var, Simplify(outermost_loop_min), Simplify(outermost_loop_extent), op->for_type,
                       op->device_api, op->body);
    } else if (is_outermost) {  // outermost multicore loop
      Expr num_inner_iters = 1;
      for (auto i = curr_index_ + 1; i < multicore_loops_.size(); ++i) {
        num_inner_iters = num_inner_iters * multicore_loops_[i]->extent;
      }
      Expr outermost_loop_min = FloorDiv::make(iters_per_core_ * block_idx_, num_inner_iters);
      Expr outermost_loop_max = FloorDiv::make(iters_per_core_ * block_idx_ + iters_per_core_ - 1, num_inner_iters);
      Expr outermost_loop_extent = (outermost_loop_max + 1) - outermost_loop_min;

      ++curr_index_;
      Stmt stmt = For::make(op->loop_var, Simplify(outermost_loop_min), Simplify(outermost_loop_extent), op->for_type,
                            op->device_api, Mutate(op->body));
      --curr_index_;
      return stmt;
    }

    // remaining case: non-outermost multicore loop
    Expr outer_iters = 0;
    for (auto i = 0u; i < curr_index_; ++i) {
      if (i == 0) {
        outer_iters =
          Simplify(multicore_loops_[i]->loop_var - multicore_loops_[i]->min) * multicore_loops_[i + 1]->extent;
      } else {
        outer_iters = (outer_iters + Simplify(multicore_loops_[i]->loop_var - multicore_loops_[i]->min)) *
                      multicore_loops_[i + 1]->extent;
      }
    }

    bool is_innermost = (curr_index_ + 1 == multicore_loops_.size());
    if (is_innermost) {
      Expr curr_iter = Simplify(outer_iters + (op->loop_var - op->min));
      Expr condition = (curr_iter >= iters_per_core_ * block_idx_) &&
                       (curr_iter < iters_per_core_ * (block_idx_ + 1)) && (curr_iter < total_iters_);
      Stmt body = IfThenElse::make(condition, op->body, Stmt());
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    } else {
      ++curr_index_;
      Stmt stmt = IRMutator::Mutate_(op, s);
      --curr_index_;
      return stmt;
    }
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    Stmt first = Mutate(op->first);
    Stmt rest = IfThenElse::make(block_idx_ == 0, op->rest, Stmt());
    return Block::make(first, rest);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt then_case = Mutate(op->then_case);
    if (op->else_case.defined()) {
      Stmt else_case = IfThenElse::make(block_idx_ == 0, op->else_case, Stmt());
      return IfThenElse::make(op->condition, then_case, else_case);
    } else {
      return IfThenElse::make(op->condition, then_case, op->else_case);
    }
  }

  const std::vector<const For *> &multicore_loops_;
  Expr multicore_cond_;
  const IterVar &block_idx_;
  size_t curr_index_{0};
  Expr total_iters_;
  Expr iters_per_core_;
};

class InjectDynamicShapeMulticoreMutator : public IRMutator {
 public:
  bool multicore_enabled_{false};

  explicit InjectDynamicShapeMulticoreMutator(const Expr &proposal_block) : proposal_block_(proposal_block) {}
  ~InjectDynamicShapeMulticoreMutator() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_multi_core_depth") {
      auto finder = DynamicShapeMulticoreLoopsFinder();
      finder.Find(s);
      auto multicore_loops = finder.multicore_loops_;

      if (multicore_loops.empty()) return s;

      if (!multicore_enabled_) {
        block_idx_ = ktvm::thread_axis(Range(), "blockIdx.x");
        multicore_enabled_ = true;
      }
      LOG(INFO) << proposal_block_;

      auto multicore_stmt =
        DynamicShapeMulticoreInsert(multicore_loops, finder.multicore_cond_, block_idx_, proposal_block_).Insert(s);
      return AttrStmt::make(block_idx_, "thread_extent", INT_MAX, multicore_stmt);
    }
    return IRMutator::Mutate_(op, s);
  }

  const Expr &proposal_block_;
  IterVar block_idx_;
};

Stmt InjectDynamicShapeMulticore(const Stmt &stmt, const Var &proposal_block, int &thread_ext) {
  auto expr_pb = Expr(proposal_block);
  auto mutator = InjectDynamicShapeMulticoreMutator(expr_pb);
  auto multicore_stmt = mutator.Mutate(stmt);
  if (mutator.multicore_enabled_) {
    thread_ext = -1;
    return multicore_stmt;
  } else {
    return stmt;
  }
}

Stmt InjectDynamicShapeMulticore(const Stmt &stmt, const int proposal_block) {
  auto expr_pb = Expr(proposal_block);
  auto mutator = InjectDynamicShapeMulticoreMutator(expr_pb);
  auto multicore_stmt = mutator.Mutate(stmt);
  return (mutator.multicore_enabled_ ? multicore_stmt : stmt);
}

Stmt PeelOuterLetAttr(Stmt stmt, std::vector<Stmt> &outer_stmts) {
  while (stmt.as<LetStmt>() || stmt.as<AttrStmt>()) {
    if (auto let = stmt.as<LetStmt>()) {
      outer_stmts.emplace_back(LetStmt::make(let->var, let->value, Evaluate::make(0)));
      stmt = let->body;
    } else if (auto attr = stmt.as<AttrStmt>()) {
      if (attr->attr_key == "pragma_multi_core_depth") {
        break;
      }
      outer_stmts.emplace_back(AttrStmt::make(attr->node, attr->attr_key, attr->value, Evaluate::make(0)));
      stmt = attr->body;
    }
  }
  return stmt;
}
Array<NodeRef> InjectMultiCoreVar(Stmt stmt, const Var &block_dim, int merge_outer_loop) {
  std::vector<Stmt> outer_stmts;
  stmt = PeelOuterLetAttr(stmt, outer_stmts);

  if (merge_outer_loop == 1) {
    stmt = MergeOuterLoop().Mutate(stmt);
    LOG(INFO) << "enable merging outer loop.";
  } else if (merge_outer_loop == 2) {
    stmt = MergeAndExtendOuterLoop().Mutate(stmt);
    LOG(INFO) << "extend and merge outer loop.";
  }
  int len = 0;
  stmt = InjectDynamicShapeMulticore(stmt, block_dim, len);
  std::vector<Stmt> thread_bind_attrs;
  stmt = PeelOuterLetAttr(stmt, thread_bind_attrs);
  stmt = ktvm::ir::MergeNest(thread_bind_attrs, ktvm::ir::MergeNest(outer_stmts, stmt));
  Array<NodeRef> retArray;
  retArray.push_back(stmt);
  retArray.push_back(IntImm::make(Int(32), len));
  return retArray;
}

class ScalarMerge : public IRMutator {
 public:
  Stmt Run(const Stmt &s, const Stmt &core_body) {
    core_body_ = core_body;
    return Mutate(s);
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_multi_core_depth" && Compare(op->value, make_const(op->value.type(), 1)) == 0) {
      return core_body_;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt core_body_;
};

class MultiCoreScalarMerge : public IRMutator {
 public:
  Stmt Run(const Stmt &s, const Stmt &scalar) {
    scalar_ = scalar;
    return Mutate(s);
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "thread_extent") {
      Stmt body = Mutate(op->body);
      Stmt res = ScalarMerge().Run(scalar_, body);
      return AttrStmt::make(op->node, op->attr_key, op->value, res);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt scalar_;
};

class ScalarPeel : public IRMutator {
 public:
  Stmt Run(const Stmt &s) {
    Stmt res = Mutate(s);
    if (!before_scalar_store_) {
      multi_core_body_ = s;
      return Stmt();
    }
    return res;
  }

  Stmt multi_core_body_;
  bool find_multi_core_{false};
  bool before_scalar_store_{false};

 private:
  Stmt Mutate_(const Store *op, const Stmt &s) final {
    if (!find_multi_core_)  before_scalar_store_ = true;
    return IRMutator::Mutate_(op, s);
  }

  bool MultiCoreAttr(const AttrStmt *op) {
    if (op->attr_key == "pragma_multi_core_depth" &&
        Compare(op->value, make_const(op->value.type(), 1)) == 0) {
      return true;
    }
    return false;
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (MultiCoreAttr(op)) {
      find_multi_core_ = true;
      Stmt body = Mutate(op->body);
      multi_core_body_ = AttrStmt::make(op->node, op->attr_key, op->value, body);
      return AttrStmt::make(op->node, op->attr_key, op->value, Evaluate::make(0));
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (op->first.defined() && op->rest.defined() &&
        op->first.as<AttrStmt>() != nullptr && MultiCoreAttr(op->first.as<AttrStmt>())) {
      auto first = Mutate(op->first);
      auto rest = Mutate(op->rest);
      multi_core_body_ = Block::make(multi_core_body_, rest);
      return Block::make(first, Evaluate::make(0));
    }
    return IRMutator::Mutate_(op, s);
  }
};

Stmt InjectMultiCore(Stmt stmt, int max_block_dim, int merge_outer_loop, bool is_dynamic, bool scalar_rearrange) {
  std::vector<Stmt> outer_stmts;
  if (is_dynamic) {
    stmt = PeelOuterLetAttr(stmt, outer_stmts);
  }
  Stmt scalar_part;
  if (scalar_rearrange) {
    ScalarPeel peel;
    scalar_part = peel.Run(stmt);
    stmt = peel.multi_core_body_;
  }

  if (merge_outer_loop == 1) {
    stmt = MergeOuterLoop().Mutate(stmt);
    LOG(INFO) << "enable merging outer loop.";
  } else if (merge_outer_loop == 2) {
    stmt = MergeAndExtendOuterLoop().Mutate(stmt);
    LOG(INFO) << "extend and merge outer loop.";
  }

  int proposal_block = max_block_dim;
  if (max_block_dim < 1) {
    cceconf::CceConf *conf = cceconf::CceConf::getInstance();
    CHECK(conf);
    proposal_block = conf->getCoreValue("Core_num");
  }
  if (!is_dynamic) {
    stmt = LoopCompounder(proposal_block).Mutate(stmt);
    MultiCorePlan plan(proposal_block);
    plan.Plan(stmt);
    if (plan.block_num_ > 1) {
      stmt = MultiCoreInsert(plan.block_num_, plan.block_coef_).Insert(stmt);
    }
    stmt = LoopUnCompunder().Mutate(stmt);
    if (scalar_rearrange && scalar_part.defined()) {
      stmt = MultiCoreScalarMerge().Run(stmt, scalar_part);
    }
    return stmt;
  } else {
    stmt = InjectDynamicShapeMulticore(stmt, proposal_block);
    std::vector<Stmt> thread_bind_attrs;
    stmt = PeelOuterLetAttr(stmt, thread_bind_attrs);
    return ktvm::ir::MergeNest(thread_bind_attrs, ktvm::ir::MergeNest(outer_stmts, stmt));
  }
}

Stmt MultiCorePartition(const Stmt &stmt) { return MultiCorePartitioner().Partition(stmt); }
}  // namespace ir
}  // namespace akg
