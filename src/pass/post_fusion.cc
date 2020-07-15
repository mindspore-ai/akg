/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "pass/post_fusion_utils.h"

namespace akg {
namespace ir {
class MakeFuseStmt : public IRMutator {
 public:
  MakeFuseStmt(const Map<Tensor, Buffer> &extern_buffer, const std::string &bias, bool is_conv_backprop_filter,
               const Expr &new_c1_l0out, const Expr &new_h_l0out, const Expr &new_w_l0out)
      : binds_(extern_buffer),
        bias_(bias),
        is_conv_backprop_filter_(is_conv_backprop_filter),
        new_c1_l0out_(new_c1_l0out),
        new_h_l0out_(new_h_l0out),
        new_w_l0out_(new_w_l0out) {}
  ~MakeFuseStmt() override = default;

  Stmt Mutate_(const Block *op, const Stmt &s) final {
    if (first_blk_) {
      first_blk_ = false;
      auto first = this->Mutate(op->first);
      is_l0write_ = false;
      auto rest = this->Mutate(op->rest);
      return Block::make(first, rest);
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (is_l0write_) {
      l0write_for_.emplace_back(op);
    } else {
      auto f = op;
      while (f->body.as<For>()) {
        f = f->body.as<For>();
      }
      if (auto p = f->body.as<Provide>()) {
        auto new_provide = this->Mutate_(p, f->body);
        Stmt stmt = new_provide;
        for (auto it = l0write_for_.rbegin(); it < l0write_for_.rend(); ++it) {
          const For *fs = *it;
          if (auto bias = p->value.as<Call>()) {
            if (bias_ == bias->name) {
              if (!(fs->loop_var.get() == c_ub_l0idx_[C1].get() || fs->loop_var.get() == c_ub_l0idx_[C0].get())) {
                continue;
              }
            }
          }
          if (is_reduce_ && !is_reduce_body_) {
            if (!std::any_of(reduce_args_.begin(), reduce_args_.end(),
                             [=](const Expr &e) { return e.same_as(fs->loop_var); })) {
              continue;
            }
          }
          stmt = For::make(fs->loop_var, fs->min, fs->extent, fs->for_type, fs->device_api, stmt);
        }
        if (is_reduce_init_) {
          is_reduce_init_ = false;
          stmt = AttrStmt::make(make_zero(Int(32)), "pragma_reduce_init", Expr(1), stmt);
        }
        if (is_reduce_body_) {
          is_reduce_body_ = false;
          stmt = AttrStmt::make(make_zero(Int(32)), "pragma_reduce_body", Expr(1), stmt);
        }
        if (is_reduce_out_) {
          is_reduce_out_ = false;
          stmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_atomic_add"), stmt);
        }
        if (is_op_after_reduce_) {
          is_op_after_reduce_ = false;
          stmt = AttrStmt::make(make_zero(Int(32)), "pragma_op_after_reduce", Expr(1), stmt);
        }
        return stmt;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (std::any_of(reduce_tensor_set_.begin(), reduce_tensor_set_.end(),
                    [=](const Provide *p) { return p->func.same_as(op->func); })) {
      find_reduce_tensor_ = true;
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const EQ *op, const Expr &e) final {
    if (find_last_reduce_if_ && is_reduce_) {
      if (isImm(op->b) && !Equal(op->b, 0)) {
        is_last_reduce_if_ = true;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  // rm reduce last if
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    find_last_reduce_if_ = true;
    auto stmt = this->Mutate(op->then_case);
    static_cast<void>(this->Mutate(op->condition));
    find_last_reduce_if_ = false;
    if (is_last_reduce_if_) {
      is_last_reduce_if_ = false;
      return stmt;
    }
    return IfThenElse::make(op->condition, stmt, op->else_case);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!is_l0write_) {
      // for now, when have reduce tensor, will fix provide to reduce axis.
      find_reduce_tensor_ = false;
      static_cast<void>(this->Mutate(op->value));
      if (find_reduce_tensor_) {
        is_reduce_ = true;
        reduce_tensor_set_.insert(op);
        if (IsInBinds(op->func->func_name(), binds_)) {
          is_reduce_out_ = true;
        }
      } else {
        is_reduce_ = false;
      }
      find_reduce_tensor_ = false;

      if (op->func->func_name().find("red_local") != std::string::npos) {
        if (isImm(op->value)) {
          is_reduce_init_ = true;
        } else {
          is_reduce_body_ = true;
        }
        is_reduce_ = true;
        reduce_args_ = op->args;
        reduce_tensor_set_.insert(op);
      } else {
        if (is_reduce_) {
          is_op_after_reduce_ = true;
        }
      }
      Array<Expr> offset;
      auto bias = op->value.as<Call>();
      if (IsInBinds(op->func->func_name(), binds_) || (bias && bias_ == bias->name)) {
        /// compute offset
        Array<Expr> left_args = op->args;
        Array<Expr> right_args;
        if (auto right = op->value.as<Call>()) {
          if (right->call_type == Call::Halide) {
            right_args = right->args;
          }
        }
        CHECK(right_args.size() == left_args.size()) << "Wrong args: left " << left_args << " right " << right_args;
        for (unsigned int i = 0; i < left_args.size(); ++i) {
          offset.push_back(Simplify_cce(left_args[i] - right_args[i]));
        }
      }

      CHECK_GE(c_ub_l0idx_.size(), 4);
      Array<Expr> args;
      args.push_back(c_ub_l0idx_[NN]);
      args.push_back(c_ub_l0idx_[C1]);
      args.push_back(c_ub_l0idx_[HH]);
      args.push_back(c_ub_l0idx_[WW]);
      if (!is_conv_backprop_filter_) {
        CHECK_EQ(c_ub_l0idx_.size(), 5);
        args.push_back(c_ub_l0idx_[C0]);
      }
      if (is_reduce_) {
        Array<Expr> args2;
        args2.push_back((is_reduce_ && Equal(reduce_args_[NN], Expr(0))) ? Expr(0) : c_ub_l0idx_[0]);
        args2.push_back((is_reduce_ && Equal(reduce_args_[C1], Expr(0))) ? Expr(0) : c_ub_l0idx_[1]);
        args2.push_back((is_reduce_ && Equal(reduce_args_[HH], Expr(0))) ? Expr(0) : c_ub_l0idx_[2]);
        args2.push_back((is_reduce_ && Equal(reduce_args_[WW], Expr(0))) ? Expr(0) : c_ub_l0idx_[3]);
        if (!is_conv_backprop_filter_) {
          CHECK_EQ(c_ub_l0idx_.size(), 5);
          args2.push_back((is_reduce_ && Equal(reduce_args_[4], Expr(0))) ? Expr(0) : c_ub_l0idx_[4]);
        }
        reduce_args_ = args2;
      }

      Array<Expr> bias_args;
      bias_args.push_back(Expr(0));
      bias_args.push_back(c_ub_l0idx_[C1]);
      bias_args.push_back(Expr(0));
      bias_args.push_back(Expr(0));
      if (!is_conv_backprop_filter_) {
        bias_args.push_back(c_ub_l0idx_[C0]);
      }

      Stmt stmt =
        SubstituteArgs(args, bias_args, reduce_args_, bias_, offset, is_reduce_, reduce_tensor_set_).Mutate(s);
      auto f = FindCUBCall(l0write_p_->func->func_name());
      f.Visit(stmt);
      if (f.c_ub_ != nullptr) {
        stmt = TensorSubstitute(stmt, f.c_ub_->func, l0write_p_->func, l0write_p_->value_index);
      }
      if (IsInBinds(op->func->func_name(), binds_)) {
        Array<Expr> args_;
        args_.push_back((is_reduce_ && Equal(reduce_args_[NN], Expr(0))) ? Expr(0)
                                                                         : offset[0] + stmt.as<Provide>()->args[NN]);
        args_.push_back(offset[C1] + stmt.as<Provide>()->args[C1] + new_c1_l0out_);
        args_.push_back((is_reduce_ && Equal(reduce_args_[HH], Expr(0))) ? Expr(0) : offset[HH] + new_h_l0out_);
        args_.push_back((is_reduce_ && Equal(reduce_args_[WW], Expr(0))) ? Expr(0) : offset[WW] + new_w_l0out_);
        CHECK(stmt.as<Provide>());
        if (!is_conv_backprop_filter_) {
          args_.push_back((is_reduce_ && Equal(reduce_args_[C0], Expr(0))) ? offset[C0]
                                                                           : offset[C0] + stmt.as<Provide>()->args[4]);
        }
        stmt =
          Provide::make(stmt.as<Provide>()->func, stmt.as<Provide>()->value_index, stmt.as<Provide>()->value, args_);
      }
      CHECK(stmt.as<Provide>());
      if (auto bias2 = stmt.as<Provide>()->value.as<Call>()) {
        if (bias_ == bias2->name) {
          Array<Expr> args_;
          args_.push_back(bias2->args[NN]);
          args_.push_back(bias2->args[C1] + new_c1_l0out_);
          args_.push_back(bias2->args[HH]);
          args_.push_back(bias2->args[WW]);
          args_.push_back(bias2->args[C0]);
          auto new_value =
            Call::make(bias2->type, bias2->name, args_, Call::CallType::Halide, bias2->func, bias2->value_index);
          stmt = Provide::make(stmt.as<Provide>()->func, stmt.as<Provide>()->value_index, new_value,
                               stmt.as<Provide>()->args);
        }
      }
      if (is_reduce_out_) {
        auto call = op->value.as<Call>();
        auto provide = stmt.as<Provide>();
        if ((call != nullptr) && (provide != nullptr)) {
          auto value = Add::make(Call::make(call->type, provide->func->func_name(), provide->args, call->call_type,
                                            provide->func, provide->value_index),
                                 provide->value);
          stmt = Provide::make(provide->func, provide->value_index, value, provide->args);
        }
      }
      if (is_reduce_body_) {
        stmt = AttrStmt::make(make_zero(Int(32)), "pragma_reduce_provide", Expr(1), stmt);
      }
      return stmt;
    } else {
      l0write_p_ = op;
      c_ub_l0idx_ = op->args;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool first_blk_{true};
  bool is_l0write_{true};
  bool is_reduce_{false};
  bool is_reduce_init_{false};
  bool is_reduce_body_{false};
  bool is_reduce_out_{false};
  bool is_op_after_reduce_{false};
  bool find_last_reduce_if_{false};
  bool is_last_reduce_if_{false};
  bool find_reduce_tensor_{false};
  Array<Expr> reduce_args_;
  std::unordered_set<const Provide *> reduce_tensor_set_;
  const Provide *l0write_p_{nullptr};
  std::vector<const For *> l0write_for_;
  Array<Expr> c_ub_l0idx_;
  Map<Tensor, Buffer> binds_;
  std::string bias_{""};
  bool is_conv_backprop_filter_{false};
  const Expr new_c1_l0out_;
  const Expr new_h_l0out_;
  const Expr new_w_l0out_;
};

class PostFusionAct : public IRMutator {
 public:
  PostFusionAct(const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic)
      : binds_(extern_buffer), is_dynamic_(is_dynamic) {}

  PostFusionAct(const Map<Tensor, Buffer> &extern_buffer, ConvolutionBackpropFilterModel &conv)
      : binds_(extern_buffer), conv_(conv) {
    is_dynamic_ = conv_.is_dynamic_;
    isolate_idx_max_ = conv_.infer_L1_tile();
  }

  ~PostFusionAct() override = default;

  Stmt Run(const Stmt &s) {
    Convolution collector;
    collector.Visit(s);
    attrs_ = collector.attrs_;
    if (attrs_.count(ATTR_CONV_BACKPROP_FILTER)) {
      is_conv_backprop_filter_ = GET_INTIMM_ATTR(attrs_, ATTR_CONV_BACKPROP_FILTER) != 0;
    }

    CHECK(attrs_[ATTR_CONV_FEATURE_NAME].as<StringImm>());
    feature_ = GET_STRINGIMM_ATTR_DEFAULT(attrs_, ATTR_CONV_FEATURE_NAME, "");
    CHECK(attrs_[ATTR_CONV_FILTER_NAME].as<StringImm>());
    filter_ = GET_STRINGIMM_ATTR_DEFAULT(attrs_, ATTR_CONV_FILTER_NAME, "");
    CHECK(attrs_[ATTR_CONV_RES_NAME].as<StringImm>());
    output_ = GET_STRINGIMM_ATTR_DEFAULT(attrs_, ATTR_CONV_RES_NAME, "");
    CHECK(attrs_[ATTR_CONV_BIAS_NAME].as<StringImm>());
    bias_ = GET_STRINGIMM_ATTR_DEFAULT(attrs_, ATTR_CONV_BIAS_NAME, "");

    padLeft_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_LEFT]);
    padTop_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_TOP]);
    strideH_ = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_H]);
    kernelH_ = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_H]);
    strideW_ = Downcast<Expr>(attrs_[ATTR_CONV_STRIDE_W]);
    kernelW_ = Downcast<Expr>(attrs_[ATTR_CONV_KERNEL_W]);

    // get conv L1 to UB, Ho, Wo formula:  ho = m / x, wo = m % x
    FractalInfoExtractor extractor = FractalInfoExtractor(is_dynamic_);
    extractor.Visit(s);
    exprs_ = extractor.gemmFormula_;

    Stmt stmt = this->Mutate(s);
    mutate_ = true;
    l1Write_idx_ = 0;
    stmt = this->Mutate(stmt);

    if (is_conv_backprop_filter_) {
      return stmt;
    }
    return RealizeNewShape(bias_).Mutate(stmt);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    cur_iters_.insert(op->loop_var.get());
    loopvar_map_.emplace(std::pair<std::string, const For *>(name, op));
    Stmt stmt = IRMutator::Mutate_(op, s);
    loopvar_map_.erase(name);
    cur_iters_.erase(op->loop_var.get());
    return stmt;
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    std::string name = op->func->func_name();
    if (is_conv_backprop_filter_) {
      if (name == feature_ + "_local_L1") {
        const Variable *ci{nullptr};
        // [batch, cin_c1, h, w, cin_c0]
        CHECK_EQ(op->args.size(), 5);
        GET_OUTER_AXIS(ci, 1);

        const Call *rhl = op->value.as<Call>();
        CHECK(rhl);
        std::string rhl_name = rhl->func->func_name();
        CHECK(rhl_name == feature_);
        OutAxisExtract extract_ci(ci, cur_iters_);
        extract_ci.Visit(rhl->args[1]);
        axis_ci_ = getVarExpr(extract_ci.axis_oo_);
      } else if (name == filter_ + "_local_L1") {
        const Variable *co{nullptr};
        // [batch, cout_c1, out_h, out_w, cout_c0]
        CHECK_EQ(op->args.size(), 5);
        GET_OUTER_AXIS(co, 1);

        const Call *rhl = op->value.as<Call>();
        CHECK(rhl);
        std::string rhl_name = rhl->func->func_name();
        CHECK(rhl_name == filter_);
        OutAxisExtract extract_co(co, cur_iters_);
        extract_co.Visit(rhl->args[1]);
        axis_co_ = getVarExpr(extract_co.axis_oo_);
      } else if (name == feature_ + "_fractal_L1_local_L0B") {
        const Variable *no_{nullptr};

        // [batch, ko, no, ni, ki]
        CHECK_EQ(op->args.size(), 5);
        GET_OUTER_AXIS(no_, 2);

        const Call *rhl = op->value.as<Call>();
        CHECK(rhl);
        std::string rhl_name = rhl->func->func_name();
        CHECK(rhl_name == feature_ + "_fractal_L1");

        OutAxisExtract extract_noo(no_, cur_iters_);
        extract_noo.Visit(rhl->args[2]);
        axis_no_ = getVarExpr(extract_noo.axis_oo_);
      } else if (name == filter_ + "_local_L1_local_L0A") {
        const Variable *mo_{nullptr};

        // [batch, mo, ko, mi, ki]
        CHECK_EQ(op->args.size(), 5);
        GET_OUTER_AXIS(mo_, 1);

        const Call *rhl = op->value.as<Call>();
        CHECK(rhl);
        std::string rhl_name = rhl->func->func_name();
        CHECK(rhl_name == filter_ + "_local_L1");
        OutAxisExtract extract_moo(mo_, cur_iters_);
        extract_moo.Visit(rhl->args[1]);
        axis_mo_ = getVarExpr(extract_moo.axis_oo_);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_attrs") {
      if (mutate_) {
        ++outermost_part_;
      }
    }
    if (op->attr_key == "pragma_fuse_vector") {
      if (!mutate_) {
        auto f = FindOutC1HW(binds_);
        f.Visit(s);
        OutHExpr_.emplace_back(f.OutHExpr_);
        OutWExpr_.emplace_back(f.OutWExpr_);
        if (f.OutH_ != nullptr) {
          OutH_.insert(f.OutH_);
          cutH_ = true;
        }
        if (f.OutW_ != nullptr) {
          cutW_ = true;
        }
        if (f.OutC1_ != nullptr) {
          OutC1_.insert(f.OutC1_);
        }
        if (is_conv_backprop_filter_) {
          while (static_cast<int>(fuse_vector_.size()) <= l1Write_idx_) {
            RegionExtract extract(output_ + "_local_UB");
            extract.Visit(s);

            Array<Expr> shape;
            for (auto range : extract.bounds_) {
              CHECK(is_zero(range->min));
              shape.push_back(range->extent);
            }

            InnerRealize innerRealize;
            innerRealize.Visit(s);
            auto t = placeholder(shape, innerRealize.realize_op_->type, innerRealize.realize_op_->func->func_name());

            Stmt stmt = TensorSubstitute(s, innerRealize.realize_op_->func, t->op, t->value_index);
            fuse_vector_.emplace_back(stmt);
          }
        } else {
          fuse_vector_.emplace_back(s);
        }
        return Evaluate::make(0);
      }
    }
    if (op->attr_key == "pragma_cube_l0write") {
      const auto &stmt_l0write = s;
      if (l1Write_idx_ < op->value.as<IntImm>()->value) {
        l1Write_idx_ = static_cast<int>(op->value.as<IntImm>()->value);
      }
      if (mutate_ && (!is_conv_backprop_filter_ || op->value.as<IntImm>()->value >= 0)) {
        if (!old_stmt_l1writes_.empty() || !fuse_vector_.empty()) {
          if (is_conv_backprop_filter_) {
            Stmt fusing_stmt_filter =
              !old_stmt_l1writes_.empty() ? old_stmt_l1writes_[outermost_part_ - 1] : fuse_vector_[l1Write_idx_];

            InnerAxisCollect innerAxis;
            innerAxis.Visit(s);
            for (auto kv : innerAxis.loopvar_map_) {
              loopvar_map_.emplace(std::pair<std::string, const For *>(kv.first, kv.second));
            }
            Stmt fused_stmt;
            if (!is_dynamic_)
              fused_stmt = makeFusedStmtBackprop(stmt_l0write, fusing_stmt_filter);
            else
              fused_stmt = fusing_stmt_filter;
            fused_stmt = Block::make(stmt_l0write, fused_stmt);

            for (const auto &kv : innerAxis.loopvar_map_) {
              loopvar_map_.erase(kv.first);
            }

            return fused_stmt;
          } else {
            Stmt fusing_stmt =
              !old_stmt_l1writes_.empty() ? old_stmt_l1writes_[outermost_part_ - 1] : fuse_vector_[outermost_part_ - 1];
            fusing_stmt = RealizeNewFunc().Mutate(fusing_stmt);
            // each group has four exprs
            CHECK(count_ * 4 < exprs_.size()) << "count:" << count_ << " , exprs' size:" << exprs_.size();
            Stmt fused_stmt = Block::make(stmt_l0write, fusing_stmt);
            auto new_c1 = exprs_[count_ * 4];
            auto new_h = exprs_[count_ * 4 + 1];
            auto new_w = exprs_[count_ * 4 + 2];
            fused_stmt = MakeFuseStmt(binds_, bias_, is_conv_backprop_filter_, new_c1, new_h, new_w).Mutate(fused_stmt);
            fused_stmt = updateNewL1Write(fused_stmt, new_h, new_w);
            ++count_;
            return fused_stmt;
          }
        }
      }
    }
    if (op->attr_key == "pragma_cube_l1write") {
      if (!mutate_) {
        auto f = FindOutC1HW(binds_);
        f.Visit(s);
        OutHExpr_.emplace_back(f.OutHExpr_);
        OutWExpr_.emplace_back(f.OutWExpr_);
        if (f.OutH_) {
          OutH_.insert(f.OutH_);
          cutH_ = true;
        }
        if (f.OutW_) {
          cutW_ = true;
        }
        if (f.OutC1_) {
          OutC1_.insert(f.OutC1_);
        }
        old_stmt_l1writes_.emplace_back(op->body);
        return Evaluate::make(0);
      }
    }
    if (op->attr_key == "pragma_fix_ifcondition") {
      if (mutate_) {
        mutate_if_ = true;
        auto stmt = this->Mutate(op->body);
        mutate_if_ = false;
        stmt = CanonicalSimplify(stmt);
        return stmt;
      }
    }
    if (op->attr_key == "isolated_idx") {
      if (mutate_) {
        if (is_conv_backprop_filter_) {
          gemm_idx_max_ = conv_.infer_L0_tile(++isolate_idx_);
        }
        gemm_idx_ = -1;
      }
    }

    if (op->attr_key == "pragma_gemm_l0") {
      if (mutate_) {
        ++gemm_idx_;
      }
    }

    if (op->attr_key == "KH_axis") {
      kh_axis_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      kh_axis_ = Expr(0);

      return stmt;
    }

    if (op->attr_key == "KW_axis") {
      kw_axis_ = op->value;
      Stmt stmt = IRMutator::Mutate_(op, s);
      kw_axis_ = Expr(0);

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (mutate_) {
      if (mutate_if_) {
        auto p = op->then_case.as<Provide>();
        CHECK(p);
        axis_h_ = p->args[HH].as<Variable>();
        axis_w_ = p->args[WW].as<Variable>();
        CHECK(axis_h_);
        CHECK(axis_w_);
        first_ = true;
        auto new_cond = this->Mutate(op->condition);
        return IfThenElse::make(new_cond, op->then_case, op->else_case);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Expr Mutate_(const Mod *op, const Expr &e) final {
    if (mutate_) {
      if (mutate_if_) {
        if (op->a.as<Variable>() == axis_h_ && first_) {
          first_ = false;
          // inputH base = TileH - kernelH + strideH
          // outputH base = (TileH - kernelH) // strideH + 1
          // when strideH = 1, inputH base == outputH base,
          // it's a trick to use outputH base as inputH base because of conv_backward's stride must
          // be 1. if stride > 1, it will go wrong at here.
          CHECK(Equal(strideH_, 1)) << "only support stride == 1 for now.";
          // axis_inner + axis_outer*(TileH - kernelH + strideH) - padleft
          auto offset = cutH_ ? padTop_ : Expr(0);
          return Mod::make(Simplify_cce(op->a + OutHExpr_[outermost_part_ - 1] - offset), op->b);
        }
        if (op->a.as<Variable>() == axis_w_ && !first_) {
          CHECK(Equal(strideW_, 1)) << "only support stride == 1 for now.";
          auto offset = cutW_ ? padLeft_ : Expr(0);
          return Mod::make(Simplify_cce(op->a + OutWExpr_[outermost_part_ - 1] - offset), op->b);
        }
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const FloorMod *op, const Expr &e) final {
    if (mutate_) {
      if (mutate_if_) {
        if (op->a.as<Variable>() == axis_h_ && first_) {
          first_ = false;
          CHECK(Equal(strideH_, 1)) << "only support stride == 1 for now.";
          auto offset = cutH_ ? padTop_ : Expr(0);
          return FloorMod::make(Simplify_cce(op->a + OutHExpr_[outermost_part_ - 1] - offset), op->b);
        }
        if (op->a.as<Variable>() == axis_w_ && !first_) {
          CHECK(Equal(strideW_, 1)) << "only support stride == 1 for now.";
          auto offset = cutW_ ? padLeft_ : Expr(0);
          return FloorMod::make(Simplify_cce(op->a + OutWExpr_[outermost_part_ - 1] - offset), op->b);
        }
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  VarExpr getVarExpr(const Variable *var) {
    if (var == nullptr) {
      return VarExpr();
    }
    auto it = std::find_if(
      loopvar_map_.begin(), loopvar_map_.end(),
      [=](const std::pair<std::string, const For *> &kv) { return kv.second->loop_var->name_hint == var->name_hint; });
    if (it != loopvar_map_.end()) {
      return (*it).second->loop_var;
    }
    return VarExpr();
  }

  void getMNBase(Expr &m_base, Expr &n_base_l1, Expr &n_base_l0) {
    // range_idx order as follow:
    // for (Ci Cut) {
    //   for (KH Cut) {
    //     for (KW Cut) {
    //       for (Co Cut) {
    //         for (Batch Cut) {
    //           for (H Cut) {
    //             for (W Cut) {

    CHECK_LT(isolate_idx_, isolate_idx_max_);
    CHECK_LT(gemm_idx_, gemm_idx_max_);

    int kh = static_cast<int>(attrs_[ATTR_CONV_KERNEL_H].as<IntImm>()->value);
    int kw = static_cast<int>(attrs_[ATTR_CONV_KERNEL_W].as<IntImm>()->value);
    CHECK(conv_.tile_.cut_b.as<IntImm>());
    int tileB = conv_.tile_.cut_b.as<IntImm>()->value;
    CHECK_EQ(tileB, 1);

    int idx;
    IsolateInfo info;

    /* calculate m axis base at L1 level */
    Expr mBase = Expr(0);
    idx = conv_.get_co_idx(isolate_idx_);
    for (int i = 0; i < idx; ++i) {
      info = conv_.co_info[i];
      mBase += info.outer * info.inner;
    }
    info = conv_.get_co_isolate_info(isolate_idx_);
    CHECK(info.outer.as<IntImm>());
    if (info.outer.as<IntImm>()->value > 1) {
      mBase += axis_co_ * Expr(info.inner);
    }

    /* calculate n axis base at L1 level */
    n_base_l1 = Expr(0);
    idx = conv_.get_ci_idx(isolate_idx_);
    for (int i = 0; i < idx; ++i) {
      info = conv_.ci_info[i];
      n_base_l1 += info.outer * info.inner * kh * kw;
    }
    info = conv_.get_ci_isolate_info(isolate_idx_);
    CHECK(info.outer.as<IntImm>());
    if (info.outer.as<IntImm>()->value > 1) {
      n_base_l1 += axis_ci_ * Expr(info.inner * kh * kw);
    }
    n_base_l1 = Simplify_cce(floordiv(n_base_l1, block_size_));

    /* calculate m axis base at L0 level */
    idx = conv_.get_m_idx(gemm_idx_);
    for (int i = 0; i < idx; ++i) {
      info = conv_.m_info[i];
      mBase += info.outer * info.inner;
    }
    info = conv_.get_m_isolate_info(gemm_idx_);
    CHECK(info.outer.as<IntImm>());
    if (info.outer.as<IntImm>()->value == 1) {
      m_base = mBase;
    } else {
      m_base = mBase + axis_mo_ * Expr(info.inner);
    }
    m_base = Simplify_cce(floordiv(m_base, block_size_));

    /* calculate n axis base at L0 level */
    n_base_l0 = Expr(0);
    idx = conv_.get_n_idx(gemm_idx_);
    for (int i = 0; i < idx; ++i) {
      info = conv_.n_info[i];
      n_base_l0 += info.outer * info.inner;
    }
    info = conv_.get_n_isolate_info(gemm_idx_);
    CHECK(info.outer.as<IntImm>());
    if (info.outer.as<IntImm>()->value > 1) {
      n_base_l0 += axis_no_ * Expr(info.inner);
    }
    n_base_l0 = Simplify_cce(floordiv(n_base_l0, block_size_));
  }

  Stmt makeFusedStmtBackprop(const Stmt &l0write, const Stmt &l1Write) {
    ProvideExtract extractL0;
    extractL0.Visit(l0write);
    CHECK(extractL0.op_.size() == 1) << " vs. " << extractL0.op_.size();
    const Provide *l0WriteProvide = extractL0.op_[0];

    ProvideExtract extractL1;
    extractL1.Visit(l1Write);
    CHECK(!extractL1.op_.empty()) << " vs. " << extractL1.op_.size();

    // [No, Mo, Mi, Ni]
    CHECK_EQ(l0WriteProvide->args.size(), 4);

    Array<Expr> lhs_args, rhs_args;
    Expr m_base, n_base_l1, n_base_l0;
    getMNBase(m_base, n_base_l1, n_base_l0);
    CHECK(!Equal(kernelH_, 0));
    CHECK(!Equal(kernelW_, 0));
    CHECK(conv_.get_kh_isolate_info(isolate_idx_).inner.as<IntImm>());
    CHECK(conv_.get_kw_isolate_info(isolate_idx_).inner.as<IntImm>());
    int kh_cut = conv_.get_kh_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
    int kw_cut = conv_.get_kw_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
    CHECK_NE(kh_cut, 0);
    CHECK_NE(kw_cut, 0);

    // [No, Mo, Mi, Ni]
    for (size_t i = 0; i < 4; i++) {
      if (l0WriteProvide->args[i].as<Variable>()) {
        if (i == 0) {
          lhs_args.push_back(floordiv(floordiv(n_base_l1, kernelW_), kernelH_) +
                             floordiv(floordiv(n_base_l0 + l0WriteProvide->args[i], kw_cut), kh_cut));

          int idx;
          IsolateInfo info;

          idx = conv_.get_kh_idx(isolate_idx_);
          info = conv_.kh_info[idx];
          Expr idx_kh = Expr(conv_.calc_till_idx(&conv_.kh_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_kh += kh_axis_ * Expr(info.inner);
          }
          idx_kh += floormod(floordiv(n_base_l0 + l0WriteProvide->args[i], kw_cut), kh_cut);
          lhs_args.push_back(idx_kh);

          idx = conv_.get_kw_idx(isolate_idx_);
          info = conv_.kw_info[idx];
          Expr idx_kw = Expr(conv_.calc_till_idx(&conv_.kw_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_kw += kw_axis_ * Expr(info.inner);
          }
          idx_kw += floormod(n_base_l0 + l0WriteProvide->args[i], kw_cut);
          lhs_args.push_back(idx_kw);
        } else if (i == 1) {
          lhs_args.push_back(m_base + l0WriteProvide->args[i]);
        } else {
          lhs_args.push_back(l0WriteProvide->args[i]);
        }

        rhs_args.push_back(l0WriteProvide->args[i]);
      } else {
        if (i == 0) {
          lhs_args.push_back(floordiv(floordiv(n_base_l1, kernelW_), kernelH_) +
                             floordiv(floordiv(n_base_l0, kw_cut), kh_cut));

          int idx;
          IsolateInfo info;

          idx = conv_.get_kh_idx(isolate_idx_);
          info = conv_.kh_info[idx];
          Expr idx_kh = Expr(conv_.calc_till_idx(&conv_.kh_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_kh += kh_axis_ * Expr(info.inner);
          }
          idx_kh += floormod(floordiv(n_base_l0, kw_cut), kh_cut);
          lhs_args.push_back(idx_kh);

          idx = conv_.get_kw_idx(isolate_idx_);
          info = conv_.kw_info[idx];
          Expr idx_kw = Expr(conv_.calc_till_idx(&conv_.kw_info, idx));
          CHECK(info.outer.as<IntImm>());
          if (info.outer.as<IntImm>()->value > 1) {
            idx_kw += kw_axis_ * Expr(info.inner);
          }
          idx_kw += floormod(n_base_l0, kw_cut);
          lhs_args.push_back(idx_kw);
        } else if (i == 1) {
          lhs_args.push_back(m_base);
        } else {
          lhs_args.push_back(Expr(0));
        }

        rhs_args.push_back(Expr(0));
      }
    }
    auto stmt = TensorReplace(l0WriteProvide->func, lhs_args, rhs_args, loopvar_map_).Mutate(l1Write);
    stmt = AttrStmt::make(make_zero(Int(32)), "pragma_emit_insn", Expr("dma_atomic_add"), stmt);
    return stmt;
  }

  Stmt updateNewL1Write(Stmt &s, const Expr &outer, const Expr &inner) {
    ExtractIterfromExpr extractor;
    extractor.Visit(outer + inner);
    std::set<const Variable *> tmp = extractor.GetIdxVar();
    std::set<const Variable *> idx_vec;
    // remove outer variables like mooo in (moo, mo, mi)
    std::set_difference(tmp.begin(), tmp.end(), cur_iters_.begin(), cur_iters_.end(),
                        std::inserter(idx_vec, idx_vec.end()));

    // remove mi variable
    const Variable *mi_var = extractor.GetMivar();
    const Variable *moVar = nullptr;
    if (mi_var == nullptr) {
      // parse outer: ee9/3, inner: ee9%3
      if (auto div = outer.as<Div>()) {
        CHECK(div->a.as<Variable>());
        mi_var = div->a.as<Variable>();
      }
      if (auto var = outer.as<Variable>()) {
        mi_var = var;
      }
    }

    if (mi_var != nullptr) {
      auto iter = idx_vec.find(mi_var);
      if (iter != idx_vec.end()) {
        idx_vec.erase(iter);
      }
      CHECK_LE(idx_vec.size(), 1);
      if (!idx_vec.empty()) {
        moVar = *idx_vec.begin();
      }
    }

    auto axisMap = GemmAxisMap(is_conv_backprop_filter_);
    static_cast<void>(axisMap.Mutate(s));
    for (const auto &i : axisMap.axis_map_info_) {
      if (i.first == "mo" && moVar != nullptr) {
        s = SubstituteLoopVar(s, moVar, i.second);
      } else if (i.first == "mi" && mi_var != nullptr) {
        s = SubstituteLoopVar(s, mi_var, i.second);
      }
    }
    return s;
  }

 private:
  Map<Tensor, Buffer> binds_;
  bool is_dynamic_{false};
  ConvolutionBackpropFilterModel conv_;

  int isolate_idx_max_{0};
  int gemm_idx_max_{0};
  int isolate_idx_{-1};
  int gemm_idx_{-1};
  int l1Write_idx_{0};
  const int block_size_{16};
  Expr padLeft_{0};
  Expr padTop_{0};
  Expr strideH_{0};
  Expr kernelH_{0};
  Expr strideW_{0};
  Expr kernelW_{0};

  size_t count_{0};
  size_t outermost_part_{0};

  bool cutH_{false};
  bool cutW_{false};
  bool mutate_{false};
  bool mutate_if_{false};
  bool first_{true};
  bool is_conv_backprop_filter_{false};

  std::string feature_{""};
  std::string filter_{""};
  std::string output_{""};
  std::string bias_{""};

  VarExpr axis_ci_;
  VarExpr axis_co_;
  VarExpr axis_mo_;
  VarExpr axis_no_;

  const Variable *axis_h_{nullptr};
  const Variable *axis_w_{nullptr};
  Expr kh_axis_{0};
  Expr kw_axis_{0};
  std::vector<Stmt> old_stmt_l1writes_;
  std::vector<Stmt> fuse_vector_;
  std::set<const Variable *> cur_iters_;
  std::vector<Expr> exprs_;
  std::unordered_set<const Variable *> OutH_;
  std::unordered_set<const Variable *> OutC1_;
  std::vector<Expr> OutHExpr_;
  std::vector<Expr> OutWExpr_;
  Map<std::string, NodeRef> attrs_;
  std::unordered_map<std::string, const For *> loopvar_map_;
};

class PartialDmaAdapt : public IRMutator {
 public:
  explicit PartialDmaAdapt(const Map<Tensor, Buffer> &binds) : binds_(binds) {}
  ~PartialDmaAdapt() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_gemm_l0") {
      auto r = Downcast<Map<std::string, Range>>(op->node);
      static_cast<void>(IRMutator::Mutate_(op, s));

      std::string m_partial_str = "m_size";
      std::string m_ceil_str = "m_lager_size";
      if (r.count(m_partial_str)) {
        CHECK(Equal(r[m_partial_str]->min, 0));
        if (auto imm = r[m_partial_str]->extent.as<IntImm>()) {
          m_partial_ = IntImm::make(Int(32), imm->value);
        } else {
          m_partial_ = r[m_partial_str]->extent;
        }
      } else {
        m_partial_ = 0;
      }

      if (r.count(m_ceil_str)) {
        CHECK(Equal(r[m_ceil_str]->min, 0));
        if (auto imm = r[m_ceil_str]->extent.as<IntImm>()) {
          m_ceil_ = IntImm::make(Int(32), imm->value);
        } else {
          m_ceil_ = r[m_ceil_str]->extent;
        }
      } else {
        m_ceil_ = 0;
      }

      if (Equal(m_partial_, m_ceil_)) {
        return s;
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;
    loopVarMap_.emplace(std::pair<std::string, const For *>(name, op));
    Stmt stmt = IRMutator::Mutate_(op, s);
    loopVarMap_.erase(name);

    return stmt;
  }

  Expr Mutate_(const Div *op, const Expr &e) final {
    if (find_m_len_) {
      auto add = op->a.as<Add>();
      if (add && is_const(add->b)) {
        m_len_ = add->a;
      } else {
        m_len_ = op->a;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    std::string name = op->func->func_name();
    if (std::any_of(binds_.begin(), binds_.end(),
                    [=](const std::pair<Tensor, Buffer> &bind) { return (name == bind.first->op->name); })) {
      // [OutN, OutC1, OutH, OutW, OutC0]
      CHECK_EQ(op->args.size(), 5);
      find_m_len_ = true;
      static_cast<void>(this->Mutate(op->args[2]));
      auto stmt = s;
      if (!Equal(m_len_, Expr(0))) {
        stmt = IfThenElse::make(LT::make(m_len_, m_partial_), stmt);
        stmt = AttrStmt::make(make_zero(Int(32)), "pragma_partial_dma_condition", Expr(1), stmt);
      }
      find_m_len_ = false;
      m_len_ = Expr(0);
      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  bool find_m_len_{false};
  Expr m_partial_{0};
  Expr m_ceil_{0};
  Expr m_len_{0};
  Map<Tensor, Buffer> binds_;
  std::unordered_map<std::string, const For *> loopVarMap_;
};

class AlignedMAdapt : public IRMutator {
 public:
  AlignedMAdapt(ConvolutionBackpropFilterModel &conv, const std::string &name) : conv_(conv), filter_name_(name) {
    isolate_idx_max_ = conv_.infer_L1_tile();
  }
  ~AlignedMAdapt() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    VarExpr var = op->loop_var;
    std::string name = var->name_hint;

    Expr extent = op->extent;
    if (!is_const(extent)) {
      lv_map_.insert(std::pair<std::string, Expr>(name, k_l1_));
      outerlv_map_.insert({name, op->loop_var});
      Stmt body = this->Mutate(op->body);
      outerlv_map_.erase(name);
      lv_map_.erase(name);

      CHECK_LT(isolate_idx_, isolate_idx_max_);
      CHECK(conv_.get_h_win_isolate_info(isolate_idx_).inner.as<IntImm>());
      CHECK(conv_.get_w_win_isolate_info(isolate_idx_).inner.as<IntImm>());
      CHECK(conv_.conv_.block_size.as<IntImm>());
      int h_cut = conv_.get_h_win_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
      int w_cut = conv_.get_w_win_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
      int block_size = conv_.conv_.block_size.as<IntImm>()->value;
      CHECK_NE(block_size, 0);
      int mo = (h_cut * w_cut + block_size - 1) / block_size;

      if (name == ko_name_) {
        ko_name_ = "";
        if (is_const(op->min)) {
          return For::make(op->loop_var, op->min, Expr(mo), op->for_type, op->device_api, body);
        } else {
          return For::make(op->loop_var, Expr(0), Expr(mo), op->for_type, op->device_api, body);
        }
      }
      CHECK(conv_.get_co_isolate_info(isolate_idx_).inner.as<IntImm>());
      int co_cut = conv_.get_co_isolate_info(isolate_idx_).inner.as<IntImm>()->value;

      Expr h_win_base = Expr(conv_.calc_till_idx(&conv_.h_win_info, conv_.get_h_idx(isolate_idx_)));
      CHECK(conv_.get_h_win_isolate_info(isolate_idx_).outer.as<IntImm>());
      if (conv_.get_h_win_isolate_info(isolate_idx_).outer.as<IntImm>()->value > 1) {
        h_win_base += h_var_ * h_cut;
      }

      Expr co_base = Expr(conv_.calc_till_idx(&conv_.co_info, conv_.get_co_idx(isolate_idx_)));
      CHECK(conv_.get_co_isolate_info(isolate_idx_).outer.as<IntImm>());
      if (conv_.get_co_isolate_info(isolate_idx_).outer.as<IntImm>()->value > 1) {
        co_base += co_var_ * conv_.get_co_isolate_info(isolate_idx_).inner;
      }

      int h_win = 0;
      for (const auto &h_info : conv_.h_win_info) {
        CHECK(h_info.inner.as<IntImm>());
        CHECK(h_info.outer.as<IntImm>());
        h_win += h_info.inner.as<IntImm>()->value * h_info.outer.as<IntImm>()->value;
      }
      int w_win = 0;
      for (const auto &w_info : conv_.w_win_info) {
        CHECK(w_info.inner.as<IntImm>());
        CHECK(w_info.outer.as<IntImm>());
        w_win += w_info.inner.as<IntImm>()->value * w_info.outer.as<IntImm>()->value;
      }
      int co = 0;
      for (const auto &co_info : conv_.co_info) {
        CHECK(co_info.inner.as<IntImm>());
        CHECK(co_info.outer.as<IntImm>());
        co += co_info.inner.as<IntImm>()->value * co_info.outer.as<IntImm>()->value;
      }

      Expr dstOffset = Expr(0);
      Expr srcOffset = Expr(0);
      CHECK(conv_.b_info[0].outer.as<IntImm>());
      if (conv_.b_info[0].outer.as<IntImm>()->value > 1) {
        srcOffset += batch_var_ * w_win * h_win * co;
      }
      srcOffset += co_base * w_win * h_win;
      srcOffset += h_win_base * w_win * block_size;

      CHECK_EQ(conv_.tile_.cut_b.as<IntImm>()->value, 1);
      CHECK(conv_.w_base == 1 && conv_.w_win_info[0].outer.as<IntImm>()->value == 1) << "only support cut H now!";

      Expr nBurst = Expr(co_cut / block_size);
      Expr lenBurst = Expr(h_cut * w_cut);
      Expr srcStride = Expr((h_win - h_cut) * w_win);
      Expr dstStride = Expr(mo * block_size - h_cut * w_cut);

      std::unordered_map<std::string, Expr> attrs;
      attrs["dstOffset"] = dstOffset;
      attrs["srcOffset"] = srcOffset;
      attrs["nBurst"] = nBurst;
      attrs["lenBurst"] = lenBurst;
      attrs["srcStride"] = srcStride;
      attrs["dstStride"] = dstStride;

      body = For::make(op->loop_var, Expr(0), Expr(16), op->for_type, op->device_api, body);

      if (gm_to_cbuf_) {
        return body;
      }
      gm_to_cbuf_ = true;
      return AttrStmt::make(Map<std::string, Expr>(attrs.begin(), attrs.end()), "gm_to_cbuf", Expr(0), body);
    }

    outerlv_map_.insert({name, op->loop_var});
    Stmt body = this->Mutate(op->body);
    outerlv_map_.erase(name);

    if (name == ko_name_) {
      CHECK(conv_.get_h_win_isolate_info(isolate_idx_).inner.as<IntImm>());
      CHECK(conv_.get_w_win_isolate_info(isolate_idx_).inner.as<IntImm>());
      CHECK(conv_.conv_.block_size.as<IntImm>());
      int h_cut = conv_.get_h_win_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
      int w_cut = conv_.get_w_win_isolate_info(isolate_idx_).inner.as<IntImm>()->value;
      int block_size = conv_.conv_.block_size.as<IntImm>()->value;
      int mo = (h_cut * w_cut + block_size - 1) / block_size;

      ko_name_ = "";

      return For::make(op->loop_var, op->min, Expr(mo), op->for_type, op->device_api, body);
    } else {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_attrs") {
      gm_to_cbuf_ = false;
      FindKL1 finder;
      finder.Visit(s);
      k_l1_ = finder.k_L1_;

      GetOuterAxisRHS axisBatch(outerlv_map_, filter_name_ + "_local_L1", NN);
      axisBatch.Visit(s);
      batch_var_ = axisBatch.var_;

      GetOuterAxisRHS axisCo(outerlv_map_, filter_name_ + "_local_L1", C1);
      axisCo.Visit(s);
      co_var_ = axisCo.var_;

      GetOuterAxisRHS axisH(outerlv_map_, filter_name_ + "_local_L1", HH);
      axisH.Visit(s);
      h_var_ = axisH.var_;

      GetOuterAxisRHS axisW(outerlv_map_, filter_name_ + "_local_L1", WW);
      axisW.Visit(s);
      w_var_ = axisW.var_;
    } else if (op->attr_key == "isolated_idx") {
      ++isolate_idx_;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func->func_name() == filter_name_ + "_local_L1") {
      if (auto var = op->args[KO].as<Variable>()) {
        ko_name_ = var->name_hint;
      } else {
        CHECK(is_zero(op->args[KO]));
        ko_name_ = "";
      }

      if (auto var = op->args[KI].as<Variable>()) {
        arg_name_ = var->name_hint;
      } else {
        CHECK(is_zero(op->args[KI]));
        arg_name_ = "";
      }

      if (lv_map_.count(arg_name_) > 0) {
        Expr condition = LT::make(block_size_ * op->args[KO] + op->args[KI], lv_map_[arg_name_]);
        return IfThenElse::make(condition, s);
      }
    }

    return IRMutator::Mutate_(op, s);
  }

 private:
  ConvolutionBackpropFilterModel conv_;
  Expr k_l1_;
  VarExpr batch_var_;
  VarExpr co_var_;
  VarExpr h_var_;
  VarExpr w_var_;
  bool gm_to_cbuf_{false};
  int isolate_idx_{-1};
  int isolate_idx_max_{-1};
  const int block_size_{16};
  std::string ko_name_{""};
  std::string arg_name_{""};
  std::string filter_name_{""};
  std::unordered_map<std::string, Expr> lv_map_;
  std::unordered_map<std::string, VarExpr> outerlv_map_;
};

/* after poly:
if ((cc2 == 0)) {
  for (cc10, 0, 32) {
    for (cc11, 0, 16) {
      output0_red_local_UB(0, cc10, 0, 0, cc11) = 0f
    }
  }
}
for (cc10, 0, 32) {
  for (cc11, 0, 16) {
    for (cc12, 0, 6) {
      for (cc15, 0, 28) {
        output0_red_local_UB(0, cc10, 0, 0, cc11) = (output0_red_local_UB(0, cc10, 0, 0, cc11) +
output0_local_UB(0, cc10, cc12, cc15, cc11))
      }
    }
  }
}
FIX===================>
for (ee10, 0, 2) {
  for (ee11, 0, 7) {
    if ((ee11 == 0)) {
      for (ee13, 0, 16) {
        output0_red_local_UB(0, ee10, 0, 0, ee13) = 0f
      }
    }
    for (ee12, 0, 16) {
      for (ee13, 0, 16) {
        if ((((ee11*16) + ee12) < 112)) {
          output0_red_local_UB(0, ee10, 0, 0, ee13) = (output0_red_local_UB(0, ee10, 0, 0, ee13) +
output0_local_UB(0, ee10, ee11, ee12, ee13))
      }
    }
  }
}
*/
class FixReduceCond : public IRMutator {
 public:
  FixReduceCond() = default;
  ~FixReduceCond() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_fuse_vector") {
      in_fuse_vector_ = true;
      partial_cond_ = nullptr;
      auto f = FindPartialDmaCond();
      f.Visit(s);
      if (f.cond_) {
        partial_cond_ = f.cond_;
      }
      auto stmt = IRMutator::Mutate_(op, s);
      in_fuse_vector_ = false;
      return stmt;
    }
    if (!in_fuse_vector_) {
      return IRMutator::Mutate_(op, s);
    }

    if (op->attr_key == "pragma_reduce_init") {
      is_init_ = true;
      static_cast<void>(this->Mutate(op->body));
      is_init_ = false;
      return Evaluate::make(0);
    }
    if (op->attr_key == "pragma_reduce_body") {
      fors_.clear();
      reduce_body_ = nullptr;
      is_body_ = true;
      static_cast<void>(this->Mutate(op->body));
      is_body_ = false;

      // find reduce not axis
      for (auto it = fors_.begin(); it != fors_.end();) {
        if (!std::any_of(reduce_body_->args.begin(), reduce_body_->args.end(),
                         [=](const Expr &e) { return e.same_as((*it)->loop_var); })) {
          it = fors_.erase(it);
        } else {
          ++it;
        }
      }

      insert_init_ = true;
      auto stmt = this->Mutate(op->body);
      insert_init_ = false;
      reduce_body_ = nullptr;
      return stmt;
    }
    if (op->attr_key == "pragma_reduce_provide") {
      if (partial_cond_) {
        auto stmt = IfThenElse::make(partial_cond_->condition, this->Mutate(op->body), partial_cond_->else_case);
        return AttrStmt::make(make_zero(Int(32)), "pragma_reduce_partial_dma_condition", Expr(1), stmt);
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!in_fuse_vector_ || !outermost_for_) {
      return IRMutator::Mutate_(op, s);
    }
    if (is_body_) {
      fors_.emplace_back(op);
    }
    if (insert_init_) {
      // find axis after first reduce axis
      auto it =
        std::find_if(fors_.begin(), fors_.end(), [=](const For *f) { return f->loop_var.same_as(op->loop_var); });
      if (it != fors_.end()) {
        fors_.erase(it);
      } else {
        auto stmt =
          Provide::make(reduce_body_->func, reduce_body_->value_index, reduce_init_value_, reduce_body_->args);
        std::reverse(fors_.begin(), fors_.end());
        for (auto f : fors_) {
          stmt = For::make(f->loop_var, f->min, f->extent, f->for_type, f->device_api, stmt);
        }
        fors_.clear();
        stmt = IfThenElse::make(EQ::make(op->loop_var, Expr(0)), stmt);
        stmt = AttrStmt::make(make_zero(Int(32)), "pragma_reduce_init", Expr(1), stmt);
        outermost_for_ = false;
        auto body = this->Mutate(op->body);
        outermost_for_ = true;
        auto b = Block::make(stmt, body);
        auto st = For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, b);
        return st;
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!in_fuse_vector_) {
      return IRMutator::Mutate_(op, s);
    }
    if (is_init_) {
      reduce_init_value_ = op->value;
    }
    if (is_body_) {
      reduce_body_ = op;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool in_fuse_vector_{false};
  bool outermost_for_{true};  // is reduce body's outermost for
  bool is_init_{false};
  bool is_body_{false};
  bool insert_init_{false};
  std::vector<const For *> fors_;
  Expr reduce_init_value_;
  const Provide *reduce_body_{nullptr};
  const IfThenElse *partial_cond_{nullptr};
};

class FixOpAfterReduceAxis : public IRMutator {
 public:
  explicit FixOpAfterReduceAxis(std::vector<const For *> &loopvar) : loopvar_(loopvar) {
    if (loopvar.size() == 1) {
      c0_ = loopvar_[0]->loop_var;
      c0_extent_ = loopvar_[0]->extent;
    } else {
      has_c1_ = true;
      c0_ = loopvar_[0]->loop_var;
      c0_extent_ = loopvar_[0]->extent;
      c1_ = loopvar_[1]->loop_var;
      c1_extent_ = loopvar_[1]->extent;
    }
  }
  ~FixOpAfterReduceAxis() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    Array<Expr> args;
    args.push_back(op->args[NN]);
    args.push_back(has_c1_ ? c1_ : Expr(0));  // non reduce axis
    args.push_back(op->args[HH]);
    args.push_back(op->args[WW]);
    args.push_back(c0_);  // non reduce axis
    return Call::make(op->type, op->name, args, Call::CallType::Halide, op->func, op->value_index);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    auto value = this->Mutate(op->value);
    Array<Expr> args;
    args.push_back(op->args[NN]);
    args.push_back(has_c1_ ? c1_ : Expr(0));  // non reduce axis
    args.push_back(op->args[HH]);
    args.push_back(op->args[WW]);
    args.push_back(c0_);  // non reduce axis
    auto stmt = Provide::make(op->func, op->value_index, value, args);
    if (has_c1_) {
      stmt = For::make(c0_, Expr(0), c0_extent_, loopvar_[0]->for_type, loopvar_[0]->device_api, stmt);
      stmt = For::make(c1_, Expr(0), c1_extent_, loopvar_[1]->for_type, loopvar_[1]->device_api, stmt);
    } else {
      stmt = For::make(c0_, Expr(0), c0_extent_, loopvar_[0]->for_type, loopvar_[0]->device_api, stmt);
    }
    return stmt;
  }
  Stmt Mutate_(const For *op, const Stmt &s) final { return this->Mutate(op->body); }

 private:
  bool has_c1_{false};
  Var c1_;
  Var c0_;
  Expr c1_extent_{1};
  Expr c0_extent_{1};
  std::vector<const For *> loopvar_;
};

class FixOuterAxis : public IRMutator {
 public:
  FixOuterAxis(const Map<Tensor, Buffer> &binds, const Provide *conv_out) : binds_(binds), conv_out_(conv_out) {
    Array<Expr> left_args = conv_out_->args;
    Array<Expr> right_args;
    if (auto right = conv_out_->value.as<Call>()) {
      if (right->call_type == Call::Halide) {
        right_args = right->args;
        for (unsigned int i = 0; i < left_args.size(); ++i) {
          offset_.push_back(Simplify_cce(left_args[i] - right_args[i]));
        }
      }
    }
  }
  ~FixOuterAxis() override = default;

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (mutate_ && IsInBinds(op->func->func_name(), binds_)) {
      Array<Expr> args;
      args.push_back(op->args[NN]);
      args.push_back(orig_args_[C1] + offset_[C1]);  // non reduce axis
      args.push_back(op->args[HH]);
      args.push_back(op->args[WW]);
      args.push_back(op->args[C0]);  // non reduce axis
      return Call::make(op->type, op->name, args, Call::CallType::Halide, op->func, op->value_index);
    } else {
      orig_args_ = op->args;
    }
    return IRMutator::Mutate_(op, e);
  }
  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (IsInBinds(op->func->func_name(), binds_)) {
      mutate_ = false;
      static_cast<void>(this->Mutate(op->value));
      mutate_ = true;
      auto value = this->Mutate(op->value);
      Array<Expr> args;
      args.push_back(op->args[NN]);
      args.push_back(orig_args_[C1] + offset_[C1]);  // non reduce axis
      args.push_back(op->args[HH]);
      args.push_back(op->args[WW]);
      args.push_back(op->args[C0]);  // non reduce axis
      return Provide::make(op->func, op->value_index, value, args);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool mutate_{false};
  Map<Tensor, Buffer> binds_;
  const Provide *conv_out_{nullptr};
  Array<Expr> orig_args_;
  Array<Expr> offset_;
};

/* when cut reduce axis, the op after reduce will only exist in last part:
 * for ()
 *   c_ub = c_l0c
 *   #miss#
 * for ()
 *   c_ub = c_l0c
 *   red_ub = red_ub + c_ub
 *   red_ub = red_ub * 0.1
 *   red = red + red_ub
 * to support atomic add, need reduce in each gm.
 * FIX================>
 * for ()
 *   c_ub = c_l0c
 *   red_ub = red_ub + c_ub
 *   red_ub = red_ub * 0.1      |<--   NOTICE: 1.should substitute var & tensor out of block.
 *   red = red + red_ub         |  |           2.should add realize for reduce tensor.
 * for ()                          |
 *   c_ub = c_l0c                  |
 *   red_ub = red_ub + c_ub        |
 *   red_ub = red_ub * 0.1      |---
 *   red = red + red_ub         |
 */
class FixOpAfterReduce : public IRMutator {
 public:
  explicit FixOpAfterReduce(const Map<Tensor, Buffer> &binds) : binds_(binds) {}
  ~FixOpAfterReduce() override = default;

  Stmt run(const Stmt &s) {
    mutate_ = false;
    static_cast<void>(this->Mutate(s));
    if (op_after_reduces.empty()) {
      return s;
    }
    Stmt fix_stmt = Evaluate::make(0);
    for (const auto &a : op_after_reduces) {
      fix_stmt = Block::make(fix_stmt, a);
    }
    fix_stmt_ = fix_stmt;
    count_ = 0;
    mutate_ = true;
    return this->Mutate(s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_fuse_vector") {
      ++count_;
      in_fuse_vector_ = true;
      if (!mutate_) {
        op_after_reduces.clear();
        missing_realize.clear();
        op_after_reduce_flag_.push_back(false);
        conv_out_.push_back(nullptr);
      } else {
        fix_ = true;
      }
      auto stmt = IRMutator::Mutate_(op, s);
      in_fuse_vector_ = false;
      return stmt;
    }
    if (op->attr_key == "pragma_op_after_reduce") {
      if (!mutate_) {
        op_after_reduces.push_back(s);
        op_after_reduce_flag_[count_ - 1] = true;
        get_miss_ = true;
        static_cast<void>(this->Mutate(op->body));
        get_miss_ = false;
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (!mutate_) {
      realizes.insert(op);
    }
    if (fix_ && !op_after_reduce_flag_[count_ - 1]) {
      fix_stmt_ = TensorSubstitute2(fix_stmt_, op->func->func_name(), op->func, op->value_index);
      auto attr = op->body.as<AttrStmt>();
      if (!(attr && attr->attr_key == air::ir::attr::realize_scope)) {
        fix_ = false;
        auto stmt = fix_stmt_;
        for (auto a : missing_realize) {
          for (auto b : realizes) {
            if (b->func == a->func) {
              stmt = Realize::make(b->func, b->value_index, b->type, b->bounds, b->condition, stmt);
              stmt = AttrStmt::make(b->func, air::ir::attr::realize_scope, Expr("local.UB"), stmt);
            }
          }
        }
        stmt = RealizeNewFunc().Mutate(stmt);
        CHECK_LE(loopvar_[count_ - 1].size(), 2);
        if (!loopvar_[count_ - 1].empty()) {
          stmt = FixOpAfterReduceAxis(loopvar_[count_ - 1]).Mutate(stmt);
        }
        if (conv_out_[count_ - 1]) {
          stmt = FixOuterAxis(binds_, conv_out_[count_ - 1]).Mutate(stmt);
        }
        return Realize::make(op->func, op->value_index, op->type, op->bounds, op->condition,
                             Block::make(op->body, stmt));
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (!in_fuse_vector_) {
      return IRMutator::Mutate_(op, s);
    }
    if (get_miss_) {
      if (op->func->func_name().find("local_UB") != std::string::npos) {
        missing_realize.insert(op);
      }
    }
    if (!mutate_) {
      if (count_ && IsInBinds(op->func->func_name(), binds_)) {
        conv_out_[count_ - 1] = op;
        if (auto call = op->value.as<Call>()) {
          if (auto var = call->args[C1].as<Variable>()) {
            c1_var_ = var;
          }
          if (auto var = call->args[C0].as<Variable>()) {
            c0_var_ = var;
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }
  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (!in_fuse_vector_) {
      return IRMutator::Mutate_(op, s);
    }
    if (!mutate_ && count_) {
      c1_var_ = nullptr;
      c0_var_ = nullptr;
      static_cast<void>(this->Mutate(op->body));
      if (((c1_var_ != nullptr) && (op->loop_var.get() == c1_var_)) || (c0_var_ && op->loop_var.get() == c0_var_)) {
        loopvar_.resize(count_);
        if (std::find(loopvar_[count_ - 1].begin(), loopvar_[count_ - 1].end(), op) == loopvar_[count_ - 1].end()) {
          loopvar_[count_ - 1].emplace_back(op);
        }
      }
      c1_var_ = nullptr;
      c0_var_ = nullptr;
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  bool mutate_{false};
  bool fix_{false};
  bool get_miss_{false};
  bool in_fuse_vector_{false};
  size_t count_{0};
  Stmt fix_stmt_;
  const Variable *c1_var_{nullptr};
  const Variable *c0_var_{nullptr};
  Map<Tensor, Buffer> binds_;
  std::unordered_set<const Provide *> missing_realize;
  std::unordered_set<const Realize *> realizes;
  std::vector<bool> op_after_reduce_flag_;
  std::vector<Stmt> op_after_reduces;
  std::vector<const Provide *> conv_out_;
  std::vector<std::vector<const For *>> loopvar_;
};

Stmt ReduceFusion(Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer) {
  ReduceFusionCheck checker;
  checker.Visit(stmt);
  if (!checker.is_reduce_fusion_) {
    return stmt;
  }
  stmt = FixReduceCond().Mutate(stmt);
  stmt = FixOpAfterReduce(extern_buffer).run(stmt);
  return stmt;
}

class UBToGmDmaOpt : public IRMutator {
 public:
  explicit UBToGmDmaOpt(bool allDynamicConv) : allDynamicConv_(allDynamicConv) {}
  ~UBToGmDmaOpt() override = default;

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (l1Write_) {
      VarExpr var = op->loop_var;
      std::string name = var->name_hint;

      loopvar_map_.emplace(std::pair<std::string, Expr>{name, op->extent});
      Stmt body = this->Mutate(op->body);
      loopvar_map_.erase(name);

      if (name == miAxis_) {
        miAxis_ = "";
        miExtent_ = op->extent;
        return For::make(op->loop_var, op->min, mLen_, op->for_type, op->device_api, body);
      } else if (name == moAxis_) {
        moAxis_ = "";
        if (allDynamicConv_) {
          auto min_expr = op->extent.as<Min>();
          CHECK(min_expr);
          auto sub_expr = min_expr->b.as<Sub>();
          CHECK(sub_expr);
          merge_ = true;
          // min(TMO, floordiv(len + 15, 16) - axis*TMO) -> min(TMO*miExtent, len - axis*TMO*miExtent)
          auto merged =
            Min::make(min_expr->a * miExtent_, Sub::make(this->Mutate(sub_expr->a), sub_expr->b * miExtent_));
          merge_ = false;
          auto f = body.as<For>();
          CHECK(f);
          return For::make(f->loop_var, f->min, merged, f->for_type, f->device_api, f->body);
        } else {
          return body;
        }
      }

      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (l1Write_ && op->attr_key == "pragma_partial_dma_condition") {
      partialCond_ = true;
      Stmt stmt = this->Mutate(op->body);
      partialCond_ = false;

      return stmt;
    } else if (op->attr_key == "pragma_fuse_vector" || op->attr_key == "pragma_cube_l1write") {
      l1Write_ = true;
      Stmt stmt = this->Mutate(op->body);
      l1Write_ = false;

      return stmt;
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (!partialCond_) {
      return IRMutator::Mutate_(op, s);
    }

    CHECK(!op->else_case.defined());
    auto lt = op->condition.as<LT>();
    CHECK(lt);

    Expr cond = op->condition;
    if (auto binary = lt->a.as<Add>()) {
      // ([ee1 * 256 +] [ee11 * 16 +] ee12) % w_cut
      auto var = binary->b.as<Variable>();
      CHECK(var);
      CHECK_GT(loopvar_map_.count(var->name_hint), 0);
      CHECK(is_const_int(loopvar_map_[var->name_hint], 16));
      miAxis_ = var->name_hint;
      mLen_ = Simplify_cce(mLen_ * loopvar_map_[miAxis_]);

      condition_ = true;
      cond = Simplify_cce(this->Mutate(cond));
      condition_ = false;

      if (findMooAxis_) {
        findMooAxis_ = false;
      } else {
        mLen_ = lt->b;
      }
    } else if (auto single = lt->a.as<Variable>()) {
      // ee12 % w_cut
      CHECK_GT(loopvar_map_.count(single->name_hint), 0);
      CHECK(is_const_int(loopvar_map_[single->name_hint], 16));
      miAxis_ = single->name_hint;
      mLen_ = lt->b;
    }

    Stmt then_case = Simplify_cce(this->Mutate(op->then_case));

    if (allDynamicConv_) {
      return then_case;
    } else {
      return IfThenElse::make(cond, then_case);
    }
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    std::string varName = op->name_hint;

    if (condition_) {
      if (varName == miAxis_) {
        return e;
      } else if (loopvar_map_.count(varName) > 0) {
        mLen_ = Simplify_cce(mLen_ * loopvar_map_[varName]);
        moAxis_ = varName;
        return Expr(0);
      } else {
        findMooAxis_ = true;
      }
    }

    if (partialCond_ && op->name_hint == moAxis_ && loopvar_map_.count(varName) > 0) {
      return Expr(0);
    }

    return e;
  }
  Expr Mutate_(const FloorDiv *op, const Expr &e) final {
    if (allDynamicConv_ && merge_) {
      // floordiv(len + 15, 16) -> len
      if (auto add = op->a.as<Add>()) {
        return add->a;
      }
    }
    return IRMutator::Mutate_(op, e);
  }

 private:
  std::unordered_map<std::string, Expr> loopvar_map_;
  std::string moAxis_;
  std::string miAxis_;
  Expr mLen_{1};
  Expr miExtent_{16};
  bool allDynamicConv_{false};
  bool partialCond_{false};
  bool condition_{false};
  bool l1Write_{false};
  bool findMooAxis_{false};
  bool merge_{false};
};

Stmt DmaFlatten(Stmt stmt, bool all_dynamic_conv) {
  Convolution collector;
  collector.Visit(stmt);
  if (!collector.isConv_) {
    return stmt;
  }

  return UBToGmDmaOpt(all_dynamic_conv).Mutate(stmt);
}

Stmt PostFusion(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic) {
  Convolution collector;
  collector.Visit(stmt);
  if (!collector.isConv_) {
    return stmt;
  }

  bool isConvBackpropFilter = false;
  if (IS_ATTR_EXIST(collector.attrs_, ATTR_CONV_BACKPROP_FILTER)) {
    CHECK(collector.attrs_[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
    isConvBackpropFilter = GET_INTIMM_ATTR(collector.attrs_, ATTR_CONV_BACKPROP_FILTER) != 0;
  }

  if (isConvBackpropFilter) {
    std::string output_name = GET_STRINGIMM_ATTR_DEFAULT(collector.attrs_, ATTR_CONV_RES_NAME, "");
    stmt = MarkAxis(output_name).Mutate(stmt);
    stmt = ElseCaseSplit().Mutate(stmt);

    ConvolutionBackpropFilterModel conv(collector.attrs_, is_dynamic);
    stmt = RemoveNullRealize().Mutate(stmt);
    stmt = RemoveNullRealizeScope(conv).Mutate(stmt);
    stmt = PostFusionAct(extern_buffer, conv).Run(stmt);
    CHECK(IS_ATTR_EXIST(collector.attrs_, ATTR_CONV_FILTER_NAME));
    CHECK(collector.attrs_[ATTR_CONV_FILTER_NAME].as<StringImm>());
    std::string filter_name = GET_STRINGIMM_ATTR_DEFAULT(collector.attrs_, ATTR_CONV_FILTER_NAME, "");
    if (!is_dynamic) stmt = AlignedMAdapt(conv, filter_name).Mutate(stmt);
  } else {
    stmt = PostFusionAct(extern_buffer, is_dynamic).Run(stmt);
    stmt = PartialDmaAdapt(extern_buffer).Mutate(stmt);
    stmt = ReduceFusion(stmt, extern_buffer);
  }

  stmt = RemoveNoOp(stmt);

  return stmt;
}
}  // namespace ir
}  // namespace akg
