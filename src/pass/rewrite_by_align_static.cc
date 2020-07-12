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
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <ir_pass.h>
#include <arithmetic/compute_expr.h>
#include <emit_insn/insn_info.h>

#include <cmath>
#include <limits>

#include "pass/analyze_align.h"

namespace akg {
namespace ir {
namespace {
/// Check if this buffer is UB
/// \param name - Buffer name
/// \return bool - If this buffer is UB
bool IsUbBuffer(const std::string &name) {
  return name.find(std::string("local_UB")) != std::string::npos ||
         name.find(std::string("local.UB")) != std::string::npos;
}

// axis partition
class AxisPartitioner : public IRMutator {
 public:
  AxisPartitioner() : var2scale_(), var2ext_(), counter_(0), in_insn_(false), in_store_(false) {}
  ~AxisPartitioner() override = default;

  Stmt Run(const Stmt s) { return Simplify(this->Mutate(s)); }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_ub_gm" || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                           exclude_list.count(op->value.as<StringImm>()->value) == 0)) {
      in_insn_ = true;
      counter_ = 0;
      auto ret = IRMutator::Mutate_(op, s);
      in_insn_ = false;
      return ret;
    } else if (op->attr_key == "pragma_emit_insn") {
      return s;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    bool insert_var = in_insn_ && is_const(op->extent);
    int64_t ext = GetIntConst(op->extent);
    if (insert_var) {
      var2ext_.emplace(op->loop_var.get(), GetIntConst(op->extent));
    }
    auto stmt = IRMutator::Mutate_(op, s);
    const For *opn = nullptr;
    if (in_insn_) {
      opn = stmt.as<For>();
      CHECK(opn);
      auto it = var2scale_.find(opn->loop_var.get());
      if (it != var2scale_.end()) {
        auto inner_var = Var(std::string("fv") + std::to_string(counter_++));
        auto new_expr = inner_var + it->second * opn->loop_var;
        auto body = Substitute(opn->body, {{opn->loop_var, new_expr}});
        body =

          For::make(inner_var, Expr(0), Expr(static_cast<int32_t>(it->second)), opn->for_type, opn->device_api, body);
        return For::make(opn->loop_var, opn->min, Expr(static_cast<int32_t>(ext / it->second)), opn->for_type,
                         opn->device_api, body);
      }
    }

    if (insert_var) {
      var2ext_.erase(opn->loop_var.get());
    }

    return stmt;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    in_store_ = in_insn_;
    auto store = IRMutator::Mutate_(op, s);
    if (in_insn_) {
      auto opn = store.as<Store>();
      CHECK(opn);

      auto align = GetIntConst(opn->predicate);
      if (align > 1) {
        Check(opn->index, align);
      }
      in_store_ = false;
    }
    return store;
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    auto load = IRMutator::Mutate_(op, e);
    if (in_store_) {
      auto opn = load.as<Load>();
      CHECK(opn);

      auto align = GetIntConst(opn->predicate);
      if (align > 1) {
        Check(opn->index, align);
      }
    }
    return load;
  }

 private:
  void Check(const Expr idx, int64_t align) {
    auto tmp_idx = Simplify(idx);

    auto all_vars = GetVarsInExpr(tmp_idx);

    for (size_t i = 0; i != all_vars.size(); ++i) {
      auto var = all_vars[i];

      auto strides = air::arith::DetectLinearEquation(tmp_idx, {var});
      if (strides.empty()) {
        continue;
      }

      auto it_ext = var2ext_.find(var.get());
      if (it_ext == var2ext_.end()) {
        continue;
      }

      int64_t coef = GetIntConst(strides[0]);
      int64_t ext = GetIntConst(it_ext->second);
      if (std::abs(coef) < align && std::abs(coef * ext) > align) {
        CHECK_NE(coef, 0);
        CHECK_NE(align, 0);

        if (align % coef != 0 || (coef * ext) % align != 0) {
          continue;
        }
        int64_t inner_ext = align / coef;

        auto it_scale = var2scale_.find(var.get());
        if (it_scale == var2scale_.end()) {
          var2scale_.emplace(var.get(), inner_ext);
        } else {
          CHECK_EQ(inner_ext, it_scale->second);
        }
      }
    }
  }

  std::map<const Variable *, int> var2scale_;
  std::map<const Variable *, int> var2ext_;
  int32_t counter_;
  bool in_insn_;
  bool in_store_;
};

// based on alignment, rewrite allocate
class RewriteAllocateAndIndex : public IRMutator {
 public:
  RewriteAllocateAndIndex() : scope_align_(), var2ext_(), fors_(), in_insn_(false) {}
  ~RewriteAllocateAndIndex() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == air::ir::attr::storage_scope) {
      auto scope_s = op->value.as<StringImm>()->value;
      if (scope_s == "local.UB") {
        const auto buf = op->node.as<Variable>();
        CHECK_EQ(scope_align_.count(buf), 0);
        scope_align_.emplace(buf, make_const(Int(32), -2));  // -2 means free align
      }
    }
    if (op->attr_key == "pragma_ub_gm" || (op->attr_key == "pragma_emit_insn" && op->value->IsInstance<StringImm>() &&
                                           (exclude_list.count(op->value.as<StringImm>()->value) == 0 ||
                                            op->value.as<StringImm>()->value == "scatter"))) {
      in_insn_ = true;
      auto ret = IRMutator::Mutate_(op, s);
      in_insn_ = false;
      return ret;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Allocate *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    const auto opn = stmt.as<Allocate>();
    // only rewrite ub
    CHECK(opn);
    auto it = scope_align_.find(opn->buffer_var.get());
    if (it != scope_align_.end()) {
      Expr blk_sz = make_const(Int(32), GetUbBlkSize(opn->type));

      Expr align = GetIntConst(it->second) > 0 ? it->second : blk_sz;

      Expr sz = air::arith::ComputeReduce<Mul>(opn->extents, make_const(Int(32), 1));
      Expr fixed_align = Simplify(((align + blk_sz - 1) / blk_sz * blk_sz));
      Expr fixed_sz = Simplify(((sz + align - 1) / align) * fixed_align);
      // only fix extent for extending size
      if (air::arith::Analyzer().CanProve((fixed_sz - sz) > 0)) {
        return Allocate::make(opn->buffer_var, opn->type, {fixed_sz}, opn->condition, opn->body, opn->new_expr,
                              opn->free_function);
      }
    }
    return stmt;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    fors_.push_back(op);

    bool insert_var = in_insn_ && is_const(op->extent);
    if (insert_var) {
      int64_t ext = GetIntConst(op->extent);
      var2ext_.emplace(op->loop_var.get(), ext);
    }
    auto stmt = IRMutator::Mutate_(op, s);
    if (insert_var) {
      var2ext_.erase(op->loop_var.get());
    }
    fors_.pop_back();
    return stmt;
  }

  Stmt Mutate_(const Store *op, const Stmt &s) final {
    auto it = scope_align_.find(op->buffer_var.get());
    if (it != scope_align_.end()) {
      it->second = op->predicate;
    }

    if (in_insn_) {
      Expr value = this->Mutate(op->value);

      int64_t align = GetIntConst(op->predicate);
      if (align > 0 && IsUbBuffer(op->buffer_var->name_hint)) {
        int64_t blk_sz = GetUbBlkSize(op->value.type());

        auto index = FixIndex(op->index, align, blk_sz);
        return Store::make(op->buffer_var, value, index, op->predicate);
      }
      return Store::make(op->buffer_var, value, op->index, op->predicate);
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == "col2img" && op->args.size()) {
      const Call *access_ptr_call = op->args[0].as<Call>();
      if (access_ptr_call) {
        if (access_ptr_call->args.size() >= 2) {
          const auto buffer_var = access_ptr_call->args[1].as<Variable>();
          if (buffer_var) {
            auto it = scope_align_.find(buffer_var);
            if (it != scope_align_.end()) {
              it->second = -2;  // -2 means free align
            }
          }
        }
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Load *op, const Expr &e) final {
    int64_t align = GetIntConst(op->predicate);
    if (in_insn_ && align > 0 && IsUbBuffer(op->buffer_var->name_hint)) {
      int64_t blk_sz = GetUbBlkSize(op->type);

      auto index = FixIndex(op->index, align, blk_sz);
      Expr ret = Load::make(op->type, op->buffer_var, index, op->predicate);
      return ret;
    }
    return e;
  }

 private:
  Expr FixIndex(const Expr &idx, int64_t align, int64_t blk_sz) {
    if (align == 1) {
      return Simplify(idx * Expr(static_cast<int32_t>(blk_sz)));
    }

    IndexOptimizer opt(true);
    auto tmp_idx_bk = Simplify(opt.Mutate(Simplify(idx)));
    auto tmp_idx = tmp_idx_bk;

    auto all_vars = GetVarsInExpr(tmp_idx);

    auto rst = Expr(0);

    CHECK_NE(blk_sz, 0);
    int64_t times = ((align + blk_sz - 1) / blk_sz * blk_sz);
    for (auto v : all_vars) {
      auto strides = air::arith::DetectLinearEquation(tmp_idx, {v});
      CHECK_EQ(strides.size(), 2);

      CHECK(is_const(strides[0]));
      CHECK_NE(align, 0);
      int64_t coef = GetIntConst(strides[0]);
      if (std::abs(coef) < align) {
        auto it = var2ext_.find(v.get());
        if (it != var2ext_.end() && std::abs(coef * it->second) <= align) {
          rst += v * strides[0];
        } else {
          return SimpleFix(tmp_idx_bk, opt.var2expr, align, times);
        }
      } else if (coef % align == 0) {
        auto new_coef = coef * times / align;
        rst += v * Expr(static_cast<int32_t>(new_coef));
      } else {
        return SimpleFix(tmp_idx_bk, opt.var2expr, align, times);
      }
      tmp_idx = strides[1];
    }

    CHECK(is_const(tmp_idx));
    if (GetIntConst(tmp_idx) % align != 0) {
      return SimpleFix(tmp_idx_bk, opt.var2expr, align, times);
    }

    auto offset = static_cast<int32_t>(GetIntConst(tmp_idx) * times / align);
    return Simplify(Substitute(rst + offset, opt.var2expr));
  }

  Expr SimpleFix(const Expr &idx, const Map<Var, Expr> &var2expr, int64_t align, int64_t new_align) {
    auto align32 = static_cast<int32_t>(align);
    CHECK_NE(new_align, 0);
    auto blksz32 = static_cast<int32_t>(new_align);
    air::arith::Analyzer analyzer;
    for (auto e : fors_) {
      analyzer.Bind(e->loop_var, Range::make_by_min_extent(e->min, e->extent));
    }
    for (auto e : var2expr) {
      analyzer.Bind(e.first, Range::make_by_min_extent(0, std::numeric_limits<int32_t>::max()));
    }

    auto tmp = analyzer.Simplify(Mul::make(Div::make(idx, align32), blksz32) + Mod::make(idx, align32));
    return Simplify(Substitute(tmp, var2expr));
  }

  // The storage scope of each buffer
  std::unordered_map<const Variable *, Expr> scope_align_;

  std::map<const Variable *, int> var2ext_;
  std::vector<const For *> fors_;
  bool in_insn_;
};
}  // namespace

Stmt RewriteByAlignStatic(Stmt stmt) {
  stmt = AxisPartitioner().Run(stmt);
  stmt = RewriteAllocateAndIndex().Mutate(stmt);
  return MergeLoops(stmt);
}
}  // namespace ir
}  // namespace akg
