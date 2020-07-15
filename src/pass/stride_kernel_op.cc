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
#include <tvm/ir_visitor.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include "ir_pass.h"
#include "poly/poly_util.h"
#include "pass/convolution_model.h"

namespace akg {
namespace ir {
enum AxisName { N = 0, C1, H, W, C0, INVALID = 99 };

class AttrsCollector : public IRVisitor {
 public:
  void Visit_(const AttrStmt *op) final {
    if (op->attr_key == "pragma_attrs") {
      attrs = Downcast<Map<std::string, NodeRef>>(op->node);
    }
    IRVisitor::Visit_(op);
  }

  Map<std::string, NodeRef> attrs;
};

class FindMul : public IRVisitor {
 public:
  explicit FindMul(std::string var) : var_(std::move(var)) {}
  ~FindMul() override = default;
  void Visit_(const Mul *op) final {
    if (auto a = op->a.as<Variable>()) {
      if (a->name_hint == var_) {
        mul_ = op;
      }
    }
    if (auto b = op->b.as<Variable>()) {
      if (b->name_hint == var_) {
        mul_ = op;
      }
    }
    IRVisitor::Visit_(op);
  }

  const Mul *mul_{nullptr};

 private:
  std::string var_{""};
};

class SubstituteHW : public IRMutator {
 public:
  explicit SubstituteHW(const Map<std::string, NodeRef> &attrs, bool stride_bigger_than_kernel)
      : stride_w_(Downcast<Expr>(attrs[ATTR_CONV_STRIDE_W])),
        stride_h_(Downcast<Expr>(attrs[ATTR_CONV_STRIDE_H])),
        kernel_w_(Downcast<Expr>(attrs[ATTR_CONV_KERNEL_W])),
        kernel_h_(Downcast<Expr>(attrs[ATTR_CONV_KERNEL_H])),
        stride_bigger_than_kernel_(stride_bigger_than_kernel) {
    if (attrs.count(ATTR_CONV_BACKPROP_FILTER)) {
      CHECK(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
      conv_backprop_filter_ = attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>()->value;
    }

    if (conv_backprop_filter_) {
      tile_kw_ = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KW, kernel_w_);
      tile_kh_ = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KH, kernel_h_);
    } else {
      tile_kw_ = kernel_w_;
      tile_kh_ = kernel_h_;
    }

    CHECK(attrs[ATTR_CONV_FEATURE_NAME].as<StringImm>());
    feature_ = attrs[ATTR_CONV_FEATURE_NAME].as<StringImm>()->value;
  }
  ~SubstituteHW() override = default;

  Stmt Mutate_(const Provide *op, const Stmt &s) final {
    if (op->func->func_name() == feature_ + "_local_L1") {
      Expr var_h = op->args[H];
      Expr var_w = op->args[W];
      if (auto c = op->value.as<Call>()) {
        Array<Expr> args;
        args.push_back(c->args[N]);
        args.push_back(c->args[C1]);
        std::string sv;
        auto vh = var_h.as<Variable>();
        if (!Equal(var_h, 0) && vh) {
          sv = vh->name_hint;
        }
        auto fh = FindMul(sv);
        auto arg_h = air::ir::CanonicalSimplify(c->args[H]);
        fh.Visit(arg_h);

        if (((!conv_backprop_filter_ && stride_bigger_than_kernel_) ||
             (conv_backprop_filter_ && is_const(stride_h_) && is_const(tile_kh_) &&
              air::arith::Analyzer().CanProve(stride_h_ > tile_kh_))) &&
            !Equal(var_h, 0) && fh.mul_) {
          Expr expr = GetRef<Expr>(fh.mul_);
          args.push_back(air::ir::substitute(expr, var_h, arg_h));
          h_iters_.insert(var_h.as<Variable>());
          mutate_refs_.insert(op->func);
        } else {
          args.push_back(c->args[H]);
        }

        std::string sw;
        auto vw = var_w.as<Variable>();
        if (!Equal(var_w, 0) && vw) {
          sw = vw->name_hint;
        }
        auto fw = FindMul(sw);
        auto arg_w = air::ir::CanonicalSimplify(c->args[W]);
        fw.Visit(arg_w);

        if (((!conv_backprop_filter_ && stride_bigger_than_kernel_) ||
             (conv_backprop_filter_ && is_const(stride_w_) && is_const(tile_kw_) &&
              air::arith::Analyzer().CanProve(stride_w_ > tile_kw_))) &&
            !Equal(var_w, 0) && fw.mul_) {
          Expr expr = GetRef<Expr>(fw.mul_);
          args.push_back(air::ir::substitute(expr, var_w, arg_w));
          w_iters_.insert(var_w.as<Variable>());
          mutate_refs_.insert(op->func);
        } else {
          args.push_back(c->args[W]);
        }

        args.push_back(c->args[C0]);
        auto new_value = Call::make(c->type, c->name, args, c->call_type, c->func, c->value_index);
        return Provide::make(op->func, op->value_index, new_value, op->args);
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Expr stride_w_{0};
  Expr stride_h_{0};
  Expr kernel_w_{0};
  Expr kernel_h_{0};
  Expr tile_kw_{0};
  Expr tile_kh_{0};
  int conv_backprop_filter_{0};
  std::string feature_;
  bool stride_bigger_than_kernel_{false};

 public:
  std::set<const Variable *> w_iters_;
  std::set<const Variable *> h_iters_;
  std::set<FunctionRef> mutate_refs_;
};

class StrideKernelOpAct : public IRMutator {
 public:
  explicit StrideKernelOpAct(Map<Tensor, Buffer> extern_buffer, const Map<std::string, NodeRef> &attrs,
                             std::set<const Variable *> w_iters, std::set<const Variable *> h_iters,
                             std::set<FunctionRef> mutate_refs, bool is_dynamic, bool stride_bigger_than_kernel)
      : stride_w_(Downcast<Expr>(attrs[ATTR_CONV_STRIDE_W])),
        stride_h_(Downcast<Expr>(attrs[ATTR_CONV_STRIDE_H])),
        kernel_w_(Downcast<Expr>(attrs[ATTR_CONV_KERNEL_W])),
        kernel_h_(Downcast<Expr>(attrs[ATTR_CONV_KERNEL_H])),
        extern_buffer_(std::move(extern_buffer)),
        attrs_(attrs),
        w_iters_(std::move(w_iters)),
        h_iters_(std::move(h_iters)),
        mutate_refs_(std::move(mutate_refs)),
        is_dynamic_(is_dynamic),
        stride_bigger_than_kernel_(stride_bigger_than_kernel) {
    if (attrs.count(ATTR_CONV_BACKPROP_FILTER)) {
      CHECK(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
      conv_backprop_filter_ = attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>()->value;
    }

    if (conv_backprop_filter_) {
      tile_kw_ = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KW, kernel_w_);
      tile_kh_ = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KH, kernel_h_);
    } else {
      tile_kw_ = kernel_w_;
      tile_kh_ = kernel_h_;
    }

    feature_ = GET_STRINGIMM_ATTR_DEFAULT(attrs_, ATTR_CONV_FEATURE_NAME, "");
    fm_h_ = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_H, 0);
    fm_w_ = GET_EXPR_ATTR(attrs_, ATTR_CONV_FEATURE_W, 0);

    pad_w_left_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_LEFT]);
    pad_w_right_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_RIGHT]);
    pad_h_top_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_TOP]);
    pad_h_bottom_ = Downcast<Expr>(attrs_[ATTR_CONV_PAD_BOTTOM]);

    for (auto i : extern_buffer_) {
      if (i.second->name == feature_) {
        tensor_feature_ = i.first;
      }
    }
  }
  ~StrideKernelOpAct() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "pragma_attrs") {
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      if (attrs.count(ATTR_CONV_BACKPROP_FILTER) && attrs.count(ATTR_CONV_KERNEL_H) &&
          attrs.count(ATTR_CONV_KERNEL_W) && attrs.count(ATTR_CONV_FEATURE_C)) {
        int kh = GET_INTIMM_ATTR_DEFAULT(attrs, ATTR_CONV_KERNEL_H, 0);
        int kw = GET_INTIMM_ATTR_DEFAULT(attrs, ATTR_CONV_KERNEL_W, 0);
        int ci = GET_INTIMM_ATTR_DEFAULT(attrs, ATTR_CONV_FEATURE_C, 0);
        if (kh == 7 && kw == 7 && ci == 16) {
          conv_backprop_filter_ = 0;
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    if (h_iters_.count(op->loop_var.get())) {
      auto loop_min = op->min.as<IntImm>();
      CHECK(loop_min);
      Stmt stmt;
      if (!is_dynamic_) {
        auto loop_extend = op->extent.as<IntImm>();
        CHECK(loop_extend);
        CHECK(stride_h_.as<IntImm>());
        CHECK(kernel_h_.as<IntImm>());
        CHECK(pad_h_top_.as<IntImm>());
        CHECK(pad_h_bottom_.as<IntImm>());
        h_base_ = static_cast<int>(loop_min->value);
        int extend = static_cast<int>(loop_extend->value);
        h_min_ = CanonicalSimplify(h_base_ + (extend - 1) * stride_h_ + kernel_h_);
        CHECK(h_min_.as<IntImm>());
        CHECK(w_pad_.as<IntImm>());
        int stride_h = static_cast<int>(stride_h_.as<IntImm>()->value);
        int kernel_h = static_cast<int>(kernel_h_.as<IntImm>()->value);
        h_max_ = h_base_ + extend * stride_h + kernel_h;
        CHECK(tensor_feature_->shape[H].as<IntImm>());
        h_pad_ = static_cast<int>(tensor_feature_->shape[H].as<IntImm>()->value + pad_h_top_.as<IntImm>()->value +
                                  pad_h_bottom_.as<IntImm>()->value);
        stmt = IRMutator::Mutate_(op, s);
        if (conv_backprop_filter_) {
          extent_h_ = Expr(fm_h_);
        } else {
          if (h_base_ != 0 || h_max_ < h_pad_ ||
              ((w_base_ != 0 || w_max_ >= w_pad_.as<IntImm>()->value) &&
               w_min_ * h_pad_ >= h_min_.as<IntImm>()->value * w_pad_.as<IntImm>()->value)) {
            extent_h_ = h_min_;
          } else {
            extent_h_ = Expr(h_pad_);
          }
        }
      } else {
        auto loop_extend = op->extent;
        h_base_ = static_cast<int>(loop_min->value);
        h_min_ = h_base_ + (loop_extend - 1) * stride_h_ + kernel_h_;
        stmt = IRMutator::Mutate_(op, s);
        if (conv_backprop_filter_) {
          extent_h_ = Expr(fm_h_);
        } else {
          extent_h_ = Expr(h_min_);
        }
      }
      if (auto f = stmt.as<For>()) {
        stmt = For::make(f->loop_var, f->min * stride_h_, extent_h_, f->for_type, f->device_api, f->body);
      }
      return stmt;
    } else if (w_iters_.count(op->loop_var.get())) {
      auto loop_min = op->min.as<IntImm>();
      CHECK(loop_min);
      CHECK(stride_w_.as<IntImm>());
      CHECK(kernel_w_.as<IntImm>());
      Stmt stmt;
      if (!is_dynamic_) {
        auto loop_extend = op->extent.as<IntImm>();
        CHECK(loop_extend);
        w_base_ = static_cast<int>(loop_min->value);
        int extend = static_cast<int>(loop_extend->value);
        int stride_w = static_cast<int>(stride_w_.as<IntImm>()->value);
        int kernel_w = static_cast<int>(kernel_w_.as<IntImm>()->value);
        w_min_ = w_base_ + (extend - 1) * stride_w + kernel_w;
        w_max_ = w_base_ + extend * stride_w + kernel_w;
        CHECK(h_min_.as<IntImm>());
        CHECK(w_pad_.as<IntImm>());
        CHECK(tensor_feature_->shape[W].as<IntImm>());
        w_pad_ = static_cast<int>(tensor_feature_->shape[W].as<IntImm>()->value) + pad_w_left_ + pad_w_right_;
        stmt = IRMutator::Mutate_(op, s);
        if (conv_backprop_filter_) {
          extent_w_ = Expr(fm_w_);
        } else {
          if (w_base_ != 0 || w_max_ < w_pad_.as<IntImm>()->value ||
              ((h_base_ != 0 || h_max_ < h_pad_) &&
               w_min_ * h_pad_ < h_min_.as<IntImm>()->value * w_pad_.as<IntImm>()->value)) {
            extent_w_ = Expr(w_min_);
          } else {
            extent_w_ = w_pad_;
          }
        }
      } else {
        w_base_ = static_cast<int>(loop_min->value);
        w_pad_ = tensor_feature_->shape[W] + pad_w_left_ + pad_w_right_;
        stmt = IRMutator::Mutate_(op, s);
        if (conv_backprop_filter_) {
          extent_w_ = Expr(fm_w_);
        } else {
          extent_w_ = w_pad_;
        }
      }
      if (auto f = stmt.as<For>()) {
        stmt = For::make(f->loop_var, f->min * stride_w_, extent_w_, f->for_type, f->device_api, f->body);
      }
      return stmt;
    } else {
      Stmt stmt = IRMutator::Mutate_(op, s);
      return stmt;
    }
  }

  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    Stmt stmt = IRMutator::Mutate_(op, s);
    if (mutate_refs_.count(op->func)) {
      auto r = stmt.as<Realize>();
      Region bounds;
      bounds.push_back(op->bounds[N]);
      bounds.push_back(op->bounds[C1]);

      if (conv_backprop_filter_) {
        bounds.push_back(Range::make_by_min_extent(0, fm_h_));
        bounds.push_back(Range::make_by_min_extent(0, fm_w_));
      } else {
        if (((!conv_backprop_filter_ && stride_bigger_than_kernel_) ||
             (conv_backprop_filter_ && is_const(stride_h_) && is_const(tile_kh_) &&
              air::arith::Analyzer().CanProve(stride_h_ > tile_kh_))) &&
            (!Equal(extent_h_, 0))) {
          bounds.push_back(Range::make_by_min_extent(op->bounds[H]->min, extent_h_));
          extent_h_ = 0;
        } else {
          bounds.push_back(op->bounds[H]);
        }
        if (((!conv_backprop_filter_ && stride_bigger_than_kernel_) ||
             (conv_backprop_filter_ && is_const(stride_w_) && is_const(tile_kw_) &&
              air::arith::Analyzer().CanProve(stride_w_ > tile_kw_))) &&
            (!Equal(extent_w_, 0))) {
          bounds.push_back(Range::make_by_min_extent(op->bounds[W]->min, extent_w_));
          extent_w_ = 0;
        } else {
          bounds.push_back(op->bounds[W]);
        }
      }

      bounds.push_back(op->bounds[C0]);
      if (r) {
        stmt = Realize::make(r->func, r->value_index, r->type, bounds, r->condition, r->body);
      }
    }
    return stmt;
  }

 private:
  Expr fm_w_{0};
  Expr fm_h_{0};
  Expr stride_w_{0};
  Expr stride_h_{0};
  Expr kernel_w_{0};
  Expr kernel_h_{0};
  Expr tile_kw_{0};
  Expr tile_kh_{0};
  Expr pad_w_left_{0};
  Expr pad_w_right_{0};
  Expr pad_h_top_{0};
  Expr pad_h_bottom_{0};
  int h_base_{0};
  Expr h_min_{0};
  int h_max_{0};
  int h_pad_{0};
  int w_base_{0};
  int w_min_{0};
  int w_max_{0};
  Expr w_pad_{0};
  Expr extent_h_{0};
  Expr extent_w_{0};
  std::string feature_{""};
  Tensor tensor_feature_;
  Map<Tensor, Buffer> extern_buffer_;
  Map<std::string, NodeRef> attrs_;
  std::set<const Variable *> w_iters_;
  std::set<const Variable *> h_iters_;
  std::set<FunctionRef> mutate_refs_;
  int conv_backprop_filter_{0};
  bool is_dynamic_{false};
  bool stride_bigger_than_kernel_{false};
};

Stmt StrideKernelOp(Stmt stmt, const Map<Tensor, Buffer> &extern_buffer, bool is_dynamic) {
  AttrsCollector collector;
  collector.Visit(stmt);
  Map<std::string, NodeRef> attrs = collector.attrs;
  if (attrs.count(ATTR_CONV_STRIDE_W) && attrs.count(ATTR_CONV_KERNEL_W) && attrs.count(ATTR_CONV_STRIDE_H) &&
      attrs.count(ATTR_CONV_KERNEL_H)) {
    Expr stride_w = Downcast<Expr>(attrs[ATTR_CONV_STRIDE_W]);
    Expr stride_h = Downcast<Expr>(attrs[ATTR_CONV_STRIDE_H]);
    Expr kernel_w = Downcast<Expr>(attrs[ATTR_CONV_KERNEL_W]);
    Expr kernel_h = Downcast<Expr>(attrs[ATTR_CONV_KERNEL_H]);

    int conv_backprop_filter = 0;
    if (attrs.count(ATTR_CONV_BACKPROP_FILTER)) {
      CHECK(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>());
      conv_backprop_filter = static_cast<int>(attrs[ATTR_CONV_BACKPROP_FILTER].as<IntImm>()->value);
    }

    Expr tile_kw = 0;
    Expr tile_kh = 0;
    if (conv_backprop_filter) {
      tile_kw = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KW, kernel_w);
      tile_kh = GET_EXPR_ATTR(attrs, ATTR_CONV_TILE_KH, kernel_h);
    } else {
      tile_kw = kernel_w;
      tile_kh = kernel_h;
    }
    if (is_const(stride_w) && is_const(tile_kw) && is_const(stride_h) && is_const(tile_kh) &&
        air::arith::Analyzer().CanProve(stride_w <= tile_kw) &&
        air::arith::Analyzer().CanProve(stride_h <= tile_kh)) {
      return stmt;
    }
    if (!conv_backprop_filter) {
      auto orig_stmt = stmt;
      SubstituteHW subHW(attrs, true);
      stmt = subHW.Mutate(stmt);
      stmt =
        StrideKernelOpAct(extern_buffer, attrs, subHW.w_iters_, subHW.h_iters_, subHW.mutate_refs_, is_dynamic, true)
          .Mutate(stmt);
      stmt = IfThenElse::make(GT::make(stride_h, kernel_h), stmt, orig_stmt);
    } else {
      SubstituteHW subHW(attrs, false);
      stmt = subHW.Mutate(stmt);
      stmt =
        StrideKernelOpAct(extern_buffer, attrs, subHW.w_iters_, subHW.h_iters_, subHW.mutate_refs_, is_dynamic, false)
          .Mutate(stmt);
    }
  }
  return stmt;
}
}  // namespace ir
}  // namespace akg
