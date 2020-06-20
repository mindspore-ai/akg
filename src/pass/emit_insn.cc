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
#include <tvm/packed_func_ext.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include "ir_pass.h"
#include "pass/ir_util.h"
#include "poly/poly_util.h"
#include "emit_insn/insn_emitter.h"

namespace akg {
namespace ir {
using ktvm::runtime::PackedFunc;

class FindPragmaAttrs : public IRVisitor {
 public:
  FindPragmaAttrs() = default;
  ~FindPragmaAttrs() override = default;

  NodeRef paramters_;

 private:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_attrs") {
      paramters_ = op->node;
    }

    IRVisitor::Visit_(op);
  }
};

class EmitInsns : public IRMutator {
 public:
  explicit EmitInsns(bool bisect_opt, bool cover_protect_opt, int comment_level)
      : enable_bisect_opt_(bisect_opt), enable_cover_protect_opt_(cover_protect_opt), comment_level_(comment_level) {}
  ~EmitInsns() override = default;

  Stmt Emit(const Stmt &stmt) {
    FindPragmaAttrs finder;
    finder.Visit(stmt);
    paramters_ = std::move(finder.paramters_);
    return this->Mutate(stmt);
  }

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "alloc_C") {
      collect_for_ = true;
    } else if (ktvm::ir::attr::IsPragmaKey(op->attr_key)) {
      if (op->attr_key == "pragma_fractal" || op->attr_key == "pragma_filter") {
        return Evaluate::make(0);
      } else if (insn_handle_functors_.count(op->attr_key) != 0) {
        // strip the loops
        StmtInfo for_info = GetForInfo(s);
        loops_.clear();
        for (size_t i = 0; i < for_info.vars_.size(); ++i) {
          loops_.push_back(for_info.ops_[i]);
        }
        Stmt r = (this->*insn_handle_functors_[op->attr_key])(op, s);
        CHECK(r.defined()) << "intrinsic rule must always return valid Expr";
        if (!r.same_as(s)) {
          return this->Mutate(r);
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    Stmt st = IRMutator::Mutate_(op, s);
    const auto ift = st.as<IfThenElse>();
    if (!ift) return st;
    Expr e = Simplify_cce(ift->condition);
    for (const auto &iv : loops_) {
      auto for_op = iv.as<For>();
      std::unordered_map<const Variable *, Expr> lvmp;
      std::unordered_map<const Variable *, Expr> uvmp;
      CHECK(for_op);
      lvmp[for_op->loop_var.get()] = for_op->min;
      uvmp[for_op->loop_var.get()] = for_op->extent - make_const(Int(32), 1);
      Expr lower_bound = Substitute(e, lvmp);
      lower_bound = Simplify_cce(lower_bound);
      Expr upper_bound = Substitute(e, uvmp);
      upper_bound = Simplify_cce(upper_bound);
      if (Equal(lower_bound, upper_bound)) {
        e = upper_bound;
      }
    }
    e = Simplify_cce(e);
    st = IfThenElse::make(e, ift->then_case, ift->else_case);

    return st;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (collect_for_ && collect_for_loops_.count(op->loop_var.get()) == 0) {
      collect_for_loops_.insert(op->loop_var.get());
    }
    return IRMutator::Mutate_(op, s);
  }

  Buffer CreateImg2colBuffer() {
    auto src = MakeBuf<Store>(img2col_store_, img2col_store_->value.type(), img2col_for_info_);
    std::unordered_set<const Variable *> vars_in_expr = GetVariablesInExpr(src->elem_offset);
    std::unordered_map<std::string, const Variable *> for_vars_map;
    for (auto item : collect_for_loops_) {
      if (for_vars_map.count(item->name_hint) == 0) {
        for_vars_map[item->name_hint] = item;
      }
    }

    std::unordered_map<const Variable *, Expr> value_map;
    Expr ele_offset = src->elem_offset;
    for (auto var : vars_in_expr) {
      if (for_vars_map.count(var->name_hint) < 1) {
        if (value_map.count(var) < 1) {
          value_map[var] = Expr(0);
        }
      }
    }
    if (!value_map.empty()) {
      ele_offset = Substitute(ele_offset, value_map);
    }
    collect_for_loops_.clear();

    return BufferNode::make(src->data, src->dtype, src->shape, src->strides, ele_offset, src->name, src->scope,
                            src->data_alignment, src->offset_factor, src->buffer_type);
  }

  Stmt EmitVecIntrin(const AttrStmt *op, const Stmt &s) {
    StmtInfo for_info = GetForInfo(s);
    CHECK(op->value.as<StringImm>());
    std::string str = op->value.as<StringImm>()->value;
    static_cast<void>(this->Mutate(op->body));
    loops_.clear();

    auto store = GetStores(s)[0].as<Store>();
    if (paramters_.defined() && Downcast<Map<std::string, NodeRef>>(paramters_).count("feature")) {
      std::string feature = Downcast<Map<std::string, NodeRef>>(paramters_)["feature"].as<StringImm>()->value;
      if (store && store->buffer_var->name_hint == feature + "_local_L1") {
        img2col_store_ = store;
        img2col_for_info_ = for_info;
      }
    }

    Stmt r = InsnEmit(str, op->body, enable_bisect_opt_, enable_cover_protect_opt_, comment_level_);
    return r;
  }

  Stmt EmitImg2col(const AttrStmt *op, const Stmt &s) {
    CHECK(op);
    collect_for_ = false;
    static_cast<void>(this->Mutate(op->body));
    loops_.clear();
    Buffer src = CreateImg2colBuffer();
    CHECK(op->node.as<StrMapNode>());
    Stmt r = Im2ColEmitter(op->body, op->node.as<StrMapNode>()->data, src, false);
    return r;
  }

  Stmt EmitImg2colL1UB(const AttrStmt *op, const Stmt &s) {
    CHECK(op);
    collect_for_ = false;
    static_cast<void>(this->Mutate(op->body));
    loops_.clear();
    Buffer src = CreateImg2colBuffer();
    CHECK(op->node.as<StrMapNode>());
    Stmt r = Im2ColEmitterL1UB(op->body, op->node.as<StrMapNode>()->data, src, false);
    return r;
  }

  using EmitFunctor = Stmt (EmitInsns::*)(const AttrStmt *, const Stmt &);
  bool collect_for_{false};
  std::unordered_set<const Variable *> collect_for_loops_;
  std::map<std::string, EmitFunctor> insn_handle_functors_ = {
    {"pragma_emit_insn", &EmitInsns::EmitVecIntrin},
    {"pragma_im2col", &EmitInsns::EmitImg2col},
    {"pragma_load3d", &EmitInsns::EmitImg2colL1UB},
  };

  std::vector<Stmt> loops_;
  const Store *img2col_store_{nullptr};
  StmtInfo img2col_for_info_;
  NodeRef paramters_;
  bool enable_bisect_opt_{true};
  bool enable_cover_protect_opt_{true};
  int comment_level_{0};
};

class PreEmit : public IRMutator {
 public:
  PreEmit() = default;
  ~PreEmit() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "gm_to_cbuf") {
      attrOp_ = op;
      return op->body;
    } else if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
               op->value.as<StringImm>()->value == "dma_copy") {
      Stmt stmt = IRMutator::Mutate_(op, s);

      if (attrOp_) {
        stmt = AttrStmt::make(attrOp_->node, attrOp_->attr_key, attrOp_->value, stmt);
        attrOp_ = nullptr;
      }

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  const AttrStmt *attrOp_{nullptr};
};

class UnalignedMad : public IRMutator {
 public:
  UnalignedMad() = default;
  ~UnalignedMad() override = default;

 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "gm_to_cbuf") {
      Map<std::string, Expr> attrs = Downcast<Map<std::string, Expr>>(op->node);

      new_dma_ = true;
      static_cast<void>(IRMutator::Mutate_(op, s));
      new_dma_ = false;

      auto old_dst = dma_op_->args[0].as<Call>();
      CHECK(old_dst);
      CHECK_EQ(old_dst->args.size(), 5);
      Expr new_dst =
        Call::make(old_dst->type, old_dst->name,
                   {old_dst->args[0], old_dst->args[1], attrs["dstOffset"], old_dst->args[3], old_dst->args[4]},
                   old_dst->call_type, old_dst->func, old_dst->value_index);

      auto old_src = dma_op_->args[1].as<Call>();
      CHECK(old_src);
      CHECK_EQ(old_src->args.size(), 5);
      Expr new_src =
        Call::make(old_src->type, old_src->name,
                   {old_src->args[0], old_src->args[1], attrs["srcOffset"], old_src->args[3], old_src->args[4]},
                   old_src->call_type, old_src->func, old_src->value_index);

      CHECK_EQ(dma_op_->args.size(), 8);
      Expr new_call = Call::make(dma_op_->type, dma_op_->name,
                                 {new_dst, new_src, dma_op_->args[2], attrs["nBurst"], attrs["lenBurst"],
                                  attrs["srcStride"], attrs["dstStride"], dma_op_->args[7]},
                                 dma_op_->call_type, dma_op_->func, dma_op_->value_index);

      dma_op_ = nullptr;

      return Evaluate::make(new_call);
    } else if (op->attr_key == "pragma_gemm_l0") {
      Map<std::string, Range> opRange = Downcast<Map<std::string, Range>>(op->node);
      k_size_ = -1;
      k_tail_ = -1;
      k_tail_size_ = -1;

      for (auto kv : opRange) {
        if (kv.first == "k_tail") {
          k_tail_ = static_cast<int>(kv.second->extent.as<IntImm>()->value);
        } else if (kv.first == "k_tail_size") {
          k_tail_size_ = static_cast<int>(kv.second->extent.as<IntImm>()->value);
        } else if (kv.first == "k_size") {
          k_size_ = static_cast<int>(kv.second->extent.as<IntImm>()->value);
        }
      }
    } else if (op->attr_key == "pragma_mad_out_axis_k") {
      var_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      var_ = Expr();

      return stmt;
    } else if (op->attr_key == "UnalignedDMA") {
      auto attrs = Downcast<Map<std::string, Expr>>(op->node);
      if (attrs.count("srcStrideFrom") > 0) {
        srcStrideFrom_ = attrs["srcStrideFrom"];
      } else {
        srcStrideFrom_ = Expr();
      }

      if (attrs.count("srcStrideTo") > 0) {
        srcStrideTo_ = attrs["srcStrideTo"];
      } else {
        srcStrideTo_ = Expr();
      }

      if (attrs.count("offset") > 0) {
        offset_ = attrs["offset"];
      } else {
        offset_ = Expr();
      }
    } else if (op->attr_key == "pragma_attrs") {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.count(ATTR_CONV_FILTER_NAME) > 0) {
        CHECK(attrs[ATTR_CONV_FILTER_NAME].as<StringImm>());
        filter_ = attrs[ATTR_CONV_FILTER_NAME].as<StringImm>()->value;
      }

      if (attrs.count(ATTR_CONV_BACKPROP_FILTER) > 0) {
        CHECK(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
        conv_backprop_filter_ = static_cast<int>(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>()->value);
      } else {
        conv_backprop_filter_ = 0;
      }
    } else if (op->attr_key == "pragma_emit_insn" && op->value.as<StringImm>() &&
               op->value.as<StringImm>()->value == "mad") {
      var_ = Expr();
    }

    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if ((conv_backprop_filter_ && op->name == "mad") && (k_size_ != -1)) {
      CHECK_EQ(op->args.size(), 7);
      std::vector<Expr> args(op->args.size());
      // mad(addrC, addrA, addrB, M, K, N, Control)
      for (size_t i = 0; i < args.size(); i++) {
        args[i] = op->args[i];
      }

      if (k_tail_ != -1) {
        if (var_.defined()) {
          args[4] = Select::make(EQ::make(var_, k_tail_), Expr(k_tail_size_), Expr(k_size_));
        } else {
          CHECK(is_zero(k_tail_));
          args[4] = Expr(k_tail_size_);
        }
      } else {
        args[4] = Expr(k_size_);
      }

      return Call::make(op->type, op->name, args, op->call_type, op->func, op->value_index);
    } else if (conv_backprop_filter_ && op->name == "copy_gm_to_cbuf") {
      if (new_dma_) {
        dma_op_ = op;
        return e;
      }

      if (offset_.defined() && srcStrideFrom_.defined() && srcStrideTo_.defined() &&
          !is_zero(srcStrideFrom_ - srcStrideTo_)) {
        CHECK_EQ(op->args.size(), 8);
        const Call *dst = op->args[0].as<Call>();
        CHECK(dst);
        CHECK(dst->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr));
        CHECK_EQ(dst->args.size(), 5U);

        const Call *src = op->args[1].as<Call>();
        CHECK(src);
        CHECK(src->is_intrinsic(ktvm::ir::intrinsic::tvm_access_ptr));
        CHECK_EQ(src->args.size(), 5U);

        CHECK(src->args[1].as<Variable>());
        CHECK(dst->args[1].as<Variable>());
        std::string src_name = src->args[1].as<Variable>()->name_hint;
        std::string dst_name = dst->args[1].as<Variable>()->name_hint;
        if (dst_name == filter_ + "_local_L1" && src_name == filter_) {
          Expr new_src = Call::make(src->type, src->name,
                                    {src->args[0], src->args[1], src->args[2] - offset_, src->args[3], src->args[4]},
                                    src->call_type);

          if (is_zero(op->args[5] - srcStrideFrom_)) {
            return Call::make(
              op->type, op->name,
              {op->args[0], new_src, op->args[2], op->args[3], op->args[4], srcStrideTo_, op->args[6], op->args[7]},
              op->call_type);
          } else {
            // when nBurst == 1, srcStride = dstStride = 0
            CHECK(is_zero(op->args[5])) << op->args[5] << " : " << srcStrideFrom_;
            CHECK(is_zero(op->args[6])) << op->args[6];

            return Call::make(
              op->type, op->name,
              {op->args[0], new_src, op->args[2], op->args[3], op->args[4], op->args[5], op->args[6], op->args[7]},
              op->call_type);
          }
        }
      }
    }

    return IRMutator::Mutate_(op, e);
  }

  std::string filter_;
  Expr srcStrideFrom_;
  Expr srcStrideTo_;
  Expr offset_;
  int k_size_{-1};
  int k_tail_{-1};
  int k_tail_size_{0};
  Expr var_;
  int conv_backprop_filter_{0};
  bool new_dma_{false};
  const Call *dma_op_{nullptr};
};

class RegCondition : public IRMutator {
 public:
  RegCondition() = default;
  ~RegCondition() override = default;

 private:
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    auto then_case = this->Mutate(op->then_case);
    auto else_case = op->else_case;
    if (op->else_case.defined()) {
      else_case = this->Mutate(op->else_case);
    }

    int count_num = 0;
    auto load_count = [&count_num](const NodeRef &op) {
      if (op->IsInstance<Load>()) {
        count_num = count_num + 1;
      }
    };
    PostOrderVisit(op->condition, load_count);

    if (count_num > 0) {
      std::string reg_name = "reg" + std::to_string(reg_cnt_);
      ++reg_cnt_;
      VarExpr new_var = Variable::make(UInt(1), reg_name);
      Stmt new_store = Store::make(new_var, op->condition, make_const(Int(32), 0), const_true(1));
      Expr new_load = Load::make(UInt(1), new_var, Expr(0), const_true(1));
      Stmt new_if = IfThenElse::make(new_load, then_case, else_case);
      Stmt temp = Block::make(new_store, new_if);
      Stmt new_alloc = Allocate::make(new_var, UInt(1), {make_const(Int(32), 1)}, const_true(), temp);
      Stmt new_attr = AttrStmt::make(new_var, ktvm::ir::attr::storage_scope, StringImm::make("local.REG"), new_alloc);
      return new_attr;
    }
    return IRMutator::Mutate_(op, s);
  }

  int reg_cnt_{0};
};

Stmt EmitInsn(Stmt stmt, bool enable_bisect, bool enable_cover_protect, const Map<Tensor, Buffer> &extern_buffer,
              bool is_dynamic) {
  char *debug_var = getenv("DEBUG_MODE");
  bool debug_mode = debug_var && strcmp("1", debug_var) == 0;
  if (!is_dynamic) {
    stmt = Simplify_cce(stmt);
  }
  if (debug_mode) {
    stmt = EmitInsnDebug(stmt);
  }
  stmt = PreEmit().Mutate(stmt);
  if (!is_dynamic) {
    char *comment_var = getenv("COMMENT_LEVEL");
    int comment_level = 0;
    if (comment_var) {
      comment_level = static_cast<int>(strtol(comment_var, nullptr, 10));
    }
    stmt = EmitInsns(enable_bisect, enable_cover_protect, comment_level).Emit(stmt);
  } else {
    stmt = EmitInsnWithDynamicShapes(stmt, extern_buffer);
  }
  stmt = UnalignedMad().Mutate(stmt);
  stmt = RegCondition().Mutate(stmt);
  return stmt;
}
}  // namespace ir
}  // namespace akg
