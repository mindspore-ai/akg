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
#include <ir_pass.h>
#include <tvm.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>
#include <map>
#include <stack>
#include "pass/coarsen-img2col.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
using air::arith::DetectLinearEquation;
using air::arith::EvalSet;
using air::arith::IntSet;

Stmt LowerImg2ColMutator::Mutate_(const Evaluate *op, const Stmt &s) {
  const Call *call_op = op->value.as<Call>();
  if (call_op && ((call_op->name == kCCEImg2ColIntrinName) || (call_op->name == kCCEImg2ColUBIntrinName))) {
    if (call_op->name == kCCEImg2ColUBIntrinName) {
      kImg2ColName = "img2col_cbuf_to_ub";
      kImg2ColL1UB_ = true;
    }

    return MutateImg2Col(call_op);
  } else {
    return s;
  }
}

Expr LowerImg2ColMutator::GetOutValue(const Expr &fmW, const Expr &padL, const Expr &padR, const Expr &kernelW,
                                      const Expr &strideW) {
  Expr out = Simplify_cce(FloorDiv::make(fmW + padL + padR - kernelW, strideW) + Expr(1));
  return out;
}

bool LowerImg2ColMutator::NeedTiling() const { return kImg2ColAllTimes > 1; }

bool LowerImg2ColMutator::NeedUpdateInputBuffer(const Call *op) {
  if ((kImg2ColName == "img2col_cbuf_to_ub") && (op != nullptr) &&
      (op->args.size() == static_cast<size_t>(CCEImg2ColArgIdx::kArgNums))) {
    Expr fmW = op->args[CCEImg2ColArgIdx::kFMWidth];
    Expr fmH = op->args[CCEImg2ColArgIdx::kFMHeight];

    Expr kernelH = op->args[CCEImg2ColArgIdx::kKernelH];
    Expr kernelW = op->args[CCEImg2ColArgIdx::kKernelW];

    Expr strideW = op->args[CCEImg2ColArgIdx::kStrideW];
    Expr strideH = op->args[CCEImg2ColArgIdx::kStrideH];

    Expr padBottom = op->args[CCEImg2ColArgIdx::kPadBottom];
    Expr padTop = op->args[CCEImg2ColArgIdx::kPadTop];
    Expr padLeft = op->args[CCEImg2ColArgIdx::kPadLeft];
    Expr padRight = op->args[CCEImg2ColArgIdx::kPadRight];

    Expr outW = GetOutValue(fmW, padLeft, padRight, kernelW, strideW);
    Expr outH = GetOutValue(fmH, padTop, padBottom, kernelH, strideH);

    Expr res = Simplify_cce(Mul::make(outH, outW));
    res = Simplify_cce(Mod::make(res, Expr(16)));

    if (!is_zero(res) && is_zero(padTop) && is_zero(padBottom) && is_zero(padLeft) && is_zero(padRight)) {
      return true;
    }
  }

  return false;
}

Stmt LowerImg2ColMutator::MutateImg2Col(const Call *op) {
  CHECK(op->call_type == Call::CallType::Extern);

  Array<Expr> img2col_args;
  img2col_args.assign(op->args.begin(), op->args.begin() + CCEImg2ColArgIdx::kLastImg2ColArg);
  bool ext_fm_h = false;
  if (NeedUpdateInputBuffer(op)) {
    ext_fm_h = true;
  }

  auto set_fmatrix_call =
    Call::make(Int(32), kSetFmatrixName, MakeSetFmatrixArgs(op->args, ext_fm_h), Call::CallType::Extern);
  auto img2col_call = Call::make(Int(32), kImg2ColName, UpdateImg2ColArgs(op->args), Call::CallType::Extern);

  Stmt fmatrix_stmt = Evaluate::make(set_fmatrix_call);
  Stmt img2col_stmt = Evaluate::make(img2col_call);
  std::vector<Stmt> calls{fmatrix_stmt, img2col_stmt};
  auto new_block = Block::make(calls);
  if (op->name == kCCEImg2ColUBIntrinName) {
    kImg2ColTimes++;
  }

  return new_block;
}

Array<Expr> LowerImg2ColMutator::UpdateImg2ColArgs(const Array<Expr> &args) {
  Array<Expr> res;
  if (!kImg2ColL1UB_ || !NeedTiling()) {
    res.assign(args.begin(), args.begin() + CCEImg2ColArgIdx::kLastImg2ColArg);
    return res;
  }

  for (size_t i = 0; i < args.size(); ++i) {
    if (i == CCEImg2ColArgIdx::kInitFMPosH) {
      if (kImg2ColTimes == 0) {
        res.push_back(-args[CCEImg2ColArgIdx::kPadTop]);
      } else {
        res.push_back(Expr(0));
      }
    } else if (i == CCEImg2ColArgIdx::kInitFMPosW) {
      res.push_back(-args[CCEImg2ColArgIdx::kPadLeft]);
    } else {
      res.push_back(args[i]);
    }
  }

  return Array<Expr>(res.begin(), res.begin() + CCEImg2ColArgIdx::kLastImg2ColArg);
}

Array<Expr> LowerImg2ColMutator::MakeSetFmatrixArgs(const Array<Expr> &all_args, bool extent_fm_h) {
  Expr pad_t = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kPadTop]);
  Expr pad_b = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kPadBottom]);
  Expr pad_l = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kPadLeft]);
  Expr pad_r = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kPadRight]);

  Expr fm_h = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kFMHeight]);
  Expr fm_w = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kFMWidth]);

  if (extent_fm_h) {
    fm_h = Cast::make(Int(64), all_args[CCEImg2ColArgIdx::kFMHeight] * Expr(32));
  }

  // update tiling strategy
  if (NeedTiling() && kImg2ColL1UB_) {
    Expr h_cut(2);
    if (kImg2ColTimes == static_cast<int>(CCEImg2colTiling::kImg2colHead)) {
      pad_b = Cast::make(Int(64), 0);
      Expr fm_h_expr =
        Simplify_cce(all_args[CCEImg2ColArgIdx::kKernelH] + all_args[CCEImg2ColArgIdx::kStrideH] * (h_cut - Expr(1)) -
                     all_args[CCEImg2ColArgIdx::kPadTop]);
      fm_h = Cast::make(Int(64), fm_h_expr);
    } else if (kImg2ColTimes == static_cast<int>(CCEImg2colTiling::kImg2colBody)) {
      pad_t = Cast::make(Int(64), 0);
      pad_b = Cast::make(Int(64), 0);
      Expr fm_h_expr =
        Simplify_cce(all_args[CCEImg2ColArgIdx::kKernelH] + all_args[CCEImg2ColArgIdx::kStrideH] * (h_cut - Expr(1)));
      fm_h = Cast::make(Int(64), fm_h_expr);
    } else if (kImg2ColTimes == static_cast<int>(CCEImg2colTiling::kImg2colTail)) {
      pad_t = Cast::make(Int(64), 0);
      Expr out_h = GetOutValue(all_args[CCEImg2ColArgIdx::kFMHeight], all_args[CCEImg2ColArgIdx::kPadTop],
                               all_args[CCEImg2ColArgIdx::kPadBottom], all_args[CCEImg2ColArgIdx::kKernelH],
                               all_args[CCEImg2ColArgIdx::kStrideH]);
      Expr fm_h_expr = Simplify_cce(all_args[CCEImg2ColArgIdx::kPadTop] + all_args[CCEImg2ColArgIdx::kFMHeight] -
                                    all_args[CCEImg2ColArgIdx::kStrideH] * (out_h - h_cut));
      fm_h = Cast::make(Int(64), fm_h_expr);
    }
  }

  // The order of padding bits expected by set_fmatrix
  // is different than the order they're passed to cce_img2col_!
  Expr c56 = Mul::make(pad_b, make_const(Int(64), 1L << 56));
  Expr c48 = Mul::make(pad_t, make_const(Int(64), 1L << 48));
  Expr c40 = Mul::make(pad_r, make_const(Int(64), 1L << 40));
  Expr c32 = Mul::make(pad_l, make_const(Int(64), 1L << 32));
  Expr c16 = Mul::make(fm_h, make_const(Int(64), 1L << 16));

  Expr e1 = Add::make(c56, c48);
  Expr e2 = Add::make(e1, c40);
  Expr e3 = Add::make(e2, c32);
  Expr e4 = Add::make(e3, c16);
  Expr e5 = Add::make(e4, fm_w);

  Array<Expr> fmatrix_arg;
  fmatrix_arg.push_back(e5);

  return fmatrix_arg;
}

Stmt CoarsenImg2ColMutator::Mutate_(const For *op, const Stmt &s) {
  CHECK(is_zero(op->min));

  LoopNestInfo cur_loop = {Var(op->loop_var), op->extent, op->body.as<Evaluate>()};
  loopNest_.push_back(cur_loop);
  Stmt body = this->Mutate(op->body);
  loopNest_.pop_back();

  // If coarsening was successful, remove the innermost loop and
  // replace it with its body
  if (img2col_coarsened_) {
    CHECK(body.as<Evaluate>());
    img2col_coarsened_ = false;

    return body;
  } else {
    if (body.same_as(op->body)) {
      return s;
    } else {
      return For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, body);
    }
  }
}

Expr CoarsenImg2ColMutator::Mutate_(const Call *op, const Expr &e) {
  if ((op->name == kCCEImg2ColIntrinName) || (op->name == kCCEImg2ColUBIntrinName)) {
    return MutateImg2Col(op, e);
  }

  if (op->name == kDMACopyName) {
    CHECK(as_const_int(op->args[kBurstLengthArgIdx]));
    burst_length_ = static_cast<int>(*as_const_int(op->args[kBurstLengthArgIdx]));
  }

  return e;
}  // namespace ir

Expr CoarsenImg2ColMutator::MutateImg2Col(const Call *op, const Expr &e) {
#define CHECK_CONDITION(cond) \
  do {                        \
    if (!(cond)) {            \
      return default_call;    \
    }                         \
  } while (0)

  // Prepare the default return value, to be used if coarsening fails
  Array<Expr> default_args(op->args);
  // If coarsening fails, the C1 parameter is redundant (also handled by the base address offset)
  default_args.Set(CCEImg2ColArgIdx::kInitFMPosC1, make_const(Int(32), 0));
  Expr default_call =
    Call::make(Int(32), op->name == kCCEImg2ColIntrinName ? kCCEImg2ColIntrinName : kCCEImg2ColUBIntrinName,
               default_args, Call::CallType::Extern);

  CHECK(op->call_type == Call::CallType::Extern);
  CHECK(op->args.size() == kCCEImg2ColArgNum);

  CHECK_CONDITION(!opt_turn_off_coarsening_);

  // Verify the innermost loop only contains the cce_img2col call, and nothing else
  CHECK_CONDITION(loopNest_.back().isEvaluate);

  Array<Expr> new_fmatrix_args;
  new_fmatrix_args.assign(op->args.begin() + CCEImg2ColArgIdx::kLastImg2ColArg, op->args.end());

  Map<Var, Range> vrange;
  std::unordered_map<const Variable *, IntSet> vrange_intset;
  for (const auto &cur_loop : loopNest_) {
    vrange.Set(Var(cur_loop.loopVar), Range::make_by_min_extent(0, cur_loop.loopExtent));
    vrange_intset[cur_loop.loopVar.get()] = IntSet::range(Range::make_by_min_extent(0, cur_loop.loopExtent));
  }

  Expr fm_width_expr = Simplify_cce(op->args[CCEImg2ColArgIdx::kFMWidth]);
  CHECK_CONDITION(is_const(fm_width_expr));

  int64_t fm_width = 0;
  if (as_const_int(fm_width_expr)) {
    CHECK(*as_const_int(fm_width_expr));
    fm_width = *as_const_int(fm_width_expr);
  } else if (as_const_uint(fm_width_expr)) {
    CHECK(*as_const_uint(fm_width_expr));
    fm_width = static_cast<int64_t>(*as_const_uint(fm_width_expr));
  } else {
    LOG(FATAL) << "fm_width_expr is neither int nor uint";
  }

  // some basic constraints
  CHECK_CONDITION(is_const_int(op->args[CCEImg2ColArgIdx::kDilationH], 1) &&
                  is_const_int(op->args[CCEImg2ColArgIdx::kDilationW], 1) &&
                  is_const_int(op->args[CCEImg2ColArgIdx::kRepeatMode], 0) &&
                  is_const_int(op->args[CCEImg2ColArgIdx::kC0Mode], 0));

  CHECK_CONDITION(is_const_int(op->args[CCEImg2ColArgIdx::kRepeats], 1));

  // kernel shape is constant
  CHECK_CONDITION(is_const(op->args[CCEImg2ColArgIdx::kKernelH]) && is_const(op->args[CCEImg2ColArgIdx::kKernelW]));
  CHECK(as_const_int(op->args[CCEImg2ColArgIdx::kKernelH]));
  CHECK(as_const_int(op->args[CCEImg2ColArgIdx::kKernelW]));
  int64_t kernel_h = *as_const_int(op->args[CCEImg2ColArgIdx::kKernelH]);
  int64_t kernel_w = *as_const_int(op->args[CCEImg2ColArgIdx::kKernelW]);

  Expr fp_prog_w = op->args[CCEImg2ColArgIdx::kFPosW];
  Expr fp_prog_h = op->args[CCEImg2ColArgIdx::kFPosH];
  auto innermost_var = loopNest_.back().loopVar;

  auto fine_output_ptr = op->args[CCEImg2ColArgIdx::kOutputPtr].as<Call>();
  auto fine_input_ptr = op->args[CCEImg2ColArgIdx::kInputPtr].as<Call>();
  CHECK_CONDITION(fine_input_ptr && fine_output_ptr);
  CHECK_CONDITION(fine_input_ptr->call_type == Call::Intrinsic && fine_output_ptr->call_type == Call::Intrinsic);
  CHECK_CONDITION(fine_input_ptr->name == air::ir::intrinsic::tvm_access_ptr &&
                  fine_output_ptr->name == air::ir::intrinsic::tvm_access_ptr);
  int innermost_extent = GetInteger(loopNest_.back().loopExtent);
  CHECK_GT(innermost_extent, 0);

  Array<Expr> output_coeffs = air::arith::DetectLinearEquation(fine_output_ptr->args[2], {innermost_var});
  // the output offset is in the form of X + 256 * var
  // where X is invariant w.r.t to var
  air::arith::Analyzer analyzer_;
  CHECK_CONDITION(output_coeffs.size() != 0 && analyzer_.CanProve(output_coeffs[0] == (kFractalSize * kFractalSize)));
  Array<Expr> coarse_output_args{fine_output_ptr->args[0], fine_output_ptr->args[1], output_coeffs[1],
                                 fine_output_ptr->args[3], fine_output_ptr->args[4]};
  auto coarse_output_ptr =
    Call::make(fine_output_ptr->type, fine_output_ptr->name, coarse_output_args, fine_output_ptr->call_type);

  Expr fp_coarse_w;
  Expr fp_coarse_h;
  // We will try to find fm_height from the base address' C1 offset
  int32_t fm_height = 0;
  // Case 1: kernel is 1x1
  if ((kernel_h == 1) && (kernel_w == 1)) {
    CHECK_CONDITION(analyzer_.CanProve(Simplify_cce(fp_prog_h, vrange) == 0));
    CHECK_CONDITION(analyzer_.CanProve(Simplify_cce(fp_prog_w, vrange) == 0));

    fp_coarse_w = Simplify_cce(fp_prog_w, vrange);
    fp_coarse_h = Simplify_cce(fp_prog_h, vrange);

    CHECK_CONDITION(GetRowsFromBaseAddress(fine_input_ptr->args[2], innermost_var, innermost_var, kernel_h, kernel_w,
                                           fm_width, &fm_height));
  } else {
    Expr fp_prog_w_ = op->args[CCEImg2ColArgIdx::kFPosW];
    Expr fp_prog_h_ = op->args[CCEImg2ColArgIdx::kFPosH];
    auto innermost_var_ = loopNest_.back().loopVar;

    Expr e1, e2;
    // Try to pattern-match the fetch position expressions, to find the expression
    // for the current Combo-dimension index, e.g., (k1_outer*8 + k_c)
    const auto op_w = fp_prog_w_.as<FloorMod>();
    if (op_w && as_const_int(op_w->b) && (*as_const_int(op_w->b) == kernel_w)) {
      e1 = op_w->a;
    }

    if (const auto op_h = fp_prog_h_.as<FloorMod>()) {
      if (as_const_int(op_h->b) && (*as_const_int(op_h->b) == kernel_w)) {
        const auto op_div = op_h->a.as<FloorDiv>();
        if (op_div && as_const_int(op_div->b) && (*as_const_int(op_div->b) == kernel_h)) {
          e2 = op_div->a;
        }
      }
      // Sometimes the FloorMod has been removed by simplification
    } else if (const auto op_div = fp_prog_h_.as<FloorDiv>()) {
      if (as_const_int(op_div->b) && (*as_const_int(op_div->b) == kernel_h)) {
        IntSet nset = EvalSet(fp_prog_h_, vrange_intset);
        // Don't need the FloorMod if it will have no effect
        if (analyzer_.CanProve(Simplify_cce(nset.min(), vrange) >= make_const(Int(32), 0)) &&
            analyzer_.CanProve(Simplify_cce(nset.max(), vrange) < make_const(Int(32), kernel_w))) {
          e2 = op_div->a;
        }
      }
    }

    CHECK_CONDITION(e1.defined() || e2.defined() || (innermost_extent == 1));

    if (innermost_extent == 1) {
      fp_coarse_w = Simplify_cce(substitute(innermost_var_, make_const(Int(32), 0), fp_prog_w_));
      fp_coarse_h = Simplify_cce(substitute(innermost_var_, make_const(Int(32), 0), fp_prog_h_));
    } else {
      bool Acond = false;
      bool Bcond = false;
      bool Ccond = false;
      bool Dcond = false;

      if (e1.defined()) {
        Expr e1_div = FloorDiv::make(e1, make_const(Int(32), kernel_h));
        Expr sub_e1 = FloorMod::make(e1_div, make_const(Int(32), kernel_w));
        Acond = analyzer_.CanProve(Simplify(sub_e1, vrange) == Simplify(fp_prog_h_, vrange));
        Acond = Acond || analyzer_.CanProve(CanonicalSimplify(sub_e1, vrange) == CanonicalSimplify(fp_prog_h_, vrange));
      }

      if (e2.defined()) {
        Expr sub_e2 = FloorMod::make(e2, make_const(Int(32), kernel_w));
        Bcond = analyzer_.CanProve(Simplify(sub_e2, vrange) == Simplify(fp_prog_w_, vrange));
        if (!Bcond && e1.defined()) {
          Bcond = CheckEqualMod(e2, e1, static_cast<int>(kernel_w));
        }
        Bcond = Bcond || analyzer_.CanProve(CanonicalSimplify(sub_e2, vrange) == CanonicalSimplify(fp_prog_w_, vrange));
      }
      CHECK_CONDITION(Acond || Bcond);

      if (e1.defined()) {
        Ccond =
          GetRowsFromBaseAddress(fine_input_ptr->args[2], innermost_var_, e1, kernel_h, kernel_w, fm_width, &fm_height);
      }
      if (e2.defined()) {
        Dcond =
          GetRowsFromBaseAddress(fine_input_ptr->args[2], innermost_var_, e2, kernel_h, kernel_w, fm_width, &fm_height);
      }
      CHECK_CONDITION(Ccond || Dcond);

      if (e1.defined()) {
        Expr e1_zerod = Simplify(substitute(innermost_var_, make_const(Int(32), 0), e1));
        if (!ExprUseVar(e1_zerod, innermost_var_)) {
          fp_coarse_w = MakeFPW(e1_zerod, static_cast<int>(kernel_w));
          fp_coarse_h = MakeFPH(e1_zerod, static_cast<int>(kernel_w), static_cast<int>(kernel_h));
        }
      }
      if (e2.defined()) {
        Expr e2_zerod = Simplify(substitute(innermost_var_, make_const(Int(32), 0), e2));
        if (!ExprUseVar(e2_zerod, innermost_var_)) {
          fp_coarse_w = MakeFPW(e2_zerod, static_cast<int>(kernel_w));
          fp_coarse_h = MakeFPH(e2_zerod, static_cast<int>(kernel_w), static_cast<int>(kernel_h));
        }
      }
    }
  }

  CHECK_CONDITION(fp_coarse_h.defined());
  CHECK_CONDITION(fp_coarse_w.defined());

  Expr coarse_c1;
  Expr coarse_top_left_h;
  Expr coarse_base_address;
  // If we were able to derive fm_height from the base address, move the C1 and top-left corner offsets
  // out of the base address and to their respective img2col params
  if (fm_height > 0) {
    new_fmatrix_args.Set(CCEImg2ColArgIdx::kFMHeight - CCEImg2ColArgIdx::kLastImg2ColArg,
                         make_const(Int(32), fm_height));

    Expr row_size = make_const(Int(32), kFractalSize) * make_const(Int(32), fm_width);
    Expr c1_size = row_size * make_const(Int(32), fm_height);

    // Translate the base address' offset to the top-left corner H coordinate
    Expr c1_offset = op->args[CCEImg2ColArgIdx::kInitFMPosC1] * c1_size;
    Expr top_left_offset = fine_input_ptr->args[2] - c1_offset;
    Expr top_left_offset_rows = top_left_offset / row_size;
    Expr top_left_h = Simplify(top_left_offset_rows + op->args[CCEImg2ColArgIdx::kInitFMPosH]);
    coarse_top_left_h = Simplify(substitute(innermost_var, make_const(Int(32), 0), top_left_h));

    coarse_c1 = Simplify(substitute(innermost_var, make_const(Int(32), 0), op->args[CCEImg2ColArgIdx::kInitFMPosC1]));

    // fine_input_ptr->args[2] contains C1 offset and top-left corner offset. Since both have
    // been moved to coarse_c1 and top_left_h_zerod respectively, the base address is
    // now just 0.
    coarse_base_address = make_const(Int(32), 0);
  } else {
    // We couldn't derive fm_height from the base address. If the coarsened
    // im2col call won't traverse the C1 dimension, then we don't actually need an accurate fm_height,
    // if we use the base address offset to move the top-left corner downwards.
    CHECK_CONDITION(innermost_extent <= kernel_h * kernel_w);
    CHECK_CONDITION(!ExprUseVar(op->args[CCEImg2ColArgIdx::kInitFMPosH], innermost_var));
    coarse_top_left_h = op->args[CCEImg2ColArgIdx::kInitFMPosH];
    coarse_c1 = make_const(Int(32), 0);
    coarse_base_address = Simplify(substitute(innermost_var, make_const(Int(32), 0), fine_input_ptr->args[2]));
  }

  CHECK_CONDITION(coarse_c1.defined());
  CHECK_CONDITION(coarse_top_left_h.defined());
  CHECK_CONDITION(coarse_base_address.defined());

  Array<Expr> coarse_input_args{fine_input_ptr->args[0], fine_input_ptr->args[1], coarse_base_address,
                                fine_input_ptr->args[3], fine_input_ptr->args[4]};
  Expr coarse_input_ptr =
    Call::make(fine_input_ptr->type, fine_input_ptr->name, coarse_input_args, fine_input_ptr->call_type);
  Array<Expr> new_args = {coarse_output_ptr,
                          coarse_input_ptr,
                          fp_coarse_w,
                          fp_coarse_h,
                          op->args[CCEImg2ColArgIdx::kInitFMPosW],
                          coarse_top_left_h,
                          coarse_c1,
                          op->args[CCEImg2ColArgIdx::kStrideW],
                          op->args[CCEImg2ColArgIdx::kStrideH],
                          op->args[CCEImg2ColArgIdx::kKernelW],
                          op->args[CCEImg2ColArgIdx::kKernelH],
                          op->args[CCEImg2ColArgIdx::kDilationW],
                          op->args[CCEImg2ColArgIdx::kDilationH],
                          op->args[CCEImg2ColArgIdx::kDestJumpOffset],
                          op->args[CCEImg2ColArgIdx::kRepeatMode],
                          innermost_extent,  // new number of repeats
                          op->args[CCEImg2ColArgIdx::kC0Mode]};

  Array<Expr> cce_img2col_args(new_args);
  if (new_fmatrix_args.size() <= cce_img2col_args.size()) {
    std::copy(new_fmatrix_args.begin(), new_fmatrix_args.end(),
              std::back_inserter(cce_img2col_args.CopyOnWrite()->data));
  } else {
    LOG(FATAL) << "size of array new_fmatrix_args exceeds capacity of cce_img2col_args.";
  }

  img2col_coarsened_ = true;
  return Call::make(Int(32), op->name == kCCEImg2ColIntrinName ? kCCEImg2ColIntrinName : kCCEImg2ColUBIntrinName,
                    cce_img2col_args, Call::CallType::Extern);
#undef CHECK_CONDITION
}

int CoarsenImg2ColMutator::GetInteger(Expr extent) {
  // constant folding.
  int value = -1;

  extent = Simplify(extent);
  if (const auto exti = extent.as<IntImm>()) {
    value = static_cast<int>(exti->value);
  }

  if (const auto extu = extent.as<UIntImm>()) {
    value = static_cast<int>(extu->value);
  }

  return value;
}

// Return fm_height
bool CoarsenImg2ColMutator::GetRowsFromBaseAddress(const Expr &base_address, const Var &innermost_var,
                                                   const Expr &e_pattern, int64_t kernel_h, int64_t kernel_w,
                                                   int64_t fm_width, int32_t *fm_height) {
  Expr div_once = FloorDiv::make(e_pattern, make_const(Int(32), kernel_h));
  Expr div_twice = Simplify(FloorDiv::make(div_once, make_const(Int(32), kernel_w)));
  Var temp = innermost_var.copy_with_suffix("temp");
  auto input_idx = substitute(div_twice, temp, base_address);
  if (ExprUseVar(input_idx, innermost_var)) {
    return false;
  }

  Array<Expr> input_coeffs = air::arith::DetectLinearEquation(input_idx, {temp});
  if (input_coeffs.size() < 2) {
    return false;
  }

  if (!is_const(input_coeffs[0])) {
    return false;
  }

  CHECK(as_const_int(input_coeffs[0]));
  int64_t input_coeff = *as_const_int(input_coeffs[0]);
  if (input_coeff % (kFractalSize * fm_width) != 0) {
    return false;
  }

  *fm_height = static_cast<int32_t>(input_coeff / (kFractalSize * fm_width));

  return true;
}

Expr CoarsenImg2ColMutator::MakeFPW(const Expr &e, int32_t kernel_w) {
  return FloorMod::make(e, make_const(Int(32), kernel_w));
}

Expr CoarsenImg2ColMutator::MakeFPH(const Expr &e, int32_t kernel_w, int32_t kernel_h) {
  Expr tmp = FloorDiv::make(e, make_const(Int(32), kernel_h));
  return FloorMod::make(tmp, make_const(Int(32), kernel_w));
}

bool CoarsenImg2ColMutator::CheckEqualMod(const Expr &e1, const Expr &e2, int m) {
  Array<Var> loopVars;
  std::transform(loopNest_.begin(), loopNest_.end(), std::back_inserter(loopVars.CopyOnWrite()->data),
                 [](const LoopNestInfo &v) { return (v.loopVar); });
  Array<Expr> e1_coeffs = air::arith::DetectLinearEquation(e1, loopVars);
  Array<Expr> e2_coeffs = air::arith::DetectLinearEquation(e2, loopVars);
  if (e1_coeffs.size() != e2_coeffs.size()) {
    return false;
  }

  for (unsigned i = 0; i < e1_coeffs.size(); i++) {
    if ((!as_const_int(e1_coeffs[i])) || (!as_const_int(e2_coeffs[i]))) {
      return false;
    }

    int c1 = *as_const_int(e1_coeffs[i]);
    int c2 = *as_const_int(e2_coeffs[i]);
    if ((c1 % m) != (c2 % m)) {
      return false;
    }
  }

  return true;
}

/* Combines Mad init call and Mad accumulate call into one
 e.g. if (k == 0) {
          mad(dst, src, m, k, n, 1)
      } else {
          mad(dst, src, m, k, n, 0)
      }

      combines the above into
      mad(dst, src, m, k, n, (k == 0))

      If the LoopPartition based on EQ op is enabled then, then halideIR won't have (k == 0) condition in the first
 place and this Mad coarsening won't be used in that case.
*/
class CoarsenMad : public IRMutator {
  Stmt Mutate_(const IfThenElse *op, const Stmt &s) final {
    if (const EQ *op_cond = op->condition.as<EQ>()) {
      CHECK(op_cond);
      CHECK(op->then_case.as<Evaluate>());
      CHECK(op->else_case.as<Evaluate>());
      if (is_zero(op_cond->b) && op->then_case.as<Evaluate>()->value.as<Call>() && op->else_case.defined() &&
          op->else_case.as<Evaluate>()->value.as<Call>()) {
        const Call *then_mad = op->then_case.as<Evaluate>()->value.as<Call>();
        const Call *else_mad = op->else_case.as<Evaluate>()->value.as<Call>();
        CHECK(then_mad);
        CHECK(else_mad);
        if ((then_mad->name == else_mad->name) && (then_mad->name == "mad") &&
            (then_mad->args.size() == else_mad->args.size())) {
          bool is_same_ = true;
          size_t arg_len = then_mad->args.size();
          for (size_t i = 0; i < arg_len - 1; i++) {
            if (!then_mad->args[i].same_as(else_mad->args[i])) {
              is_same_ = false;
            }
          }

          if (is_one(then_mad->args[arg_len - 1]) && is_zero(else_mad->args[arg_len - 1]) && is_same_) {
            Array<Expr> coarsen_call_args = then_mad->args;
            coarsen_call_args.Set(arg_len - 1, op->condition);

            return Evaluate::make(Call::make(then_mad->type, then_mad->name, coarsen_call_args, then_mad->call_type,
                                             then_mad->func, then_mad->value_index));
          }
        }
      }
    }

    return IRMutator::Mutate_(op, s);
  }
};

class Im2colAddressCollector : public IRVisitor {
 public:
  explicit Im2colAddressCollector(const std::string &callName) : im2colName_(callName) {}
  ~Im2colAddressCollector() override = default;

  bool Collecttor(const Stmt &s) {
    Visit(s);
    return updateAddress_;
  }

  void Visit_(const Evaluate *op) override {
    const Call *call_op = op->value.as<Call>();
    if (call_op && call_op->name == im2colName_) {
      NeedUpdateBufferAddress(call_op);
      im2colTimes_++;
    }

    return;
  }

  void Visit_(const For *op) override {
    if (firstFor_ == nullptr) {
      firstFor_ = op;
      c1_ = op->extent;
    }
    stack_.push(op);

    return IRVisitor::Visit_(op);
  }

  void Visit_(const AttrStmt *op) override {
    if (op->attr_key == "thread_extent") {
      multiCore_ = true;
    }

    return IRVisitor::Visit_(op);
  }

  void NeedUpdateBufferAddress(const Call *op) {
    if (op != nullptr && op->args.size() == static_cast<size_t>(CCEImg2ColArgIdx::kArgNums)) {
      fmW_ = op->args[CCEImg2ColArgIdx::kFMWidth];
      fmH_ = op->args[CCEImg2ColArgIdx::kFMHeight];
      kernelH_ = op->args[CCEImg2ColArgIdx::kKernelH];
      kernelW_ = op->args[CCEImg2ColArgIdx::kKernelW];
      Expr strideW = op->args[CCEImg2ColArgIdx::kStrideW];
      Expr strideH = op->args[CCEImg2ColArgIdx::kStrideH];
      Expr padBottom = op->args[CCEImg2ColArgIdx::kPadBottom];
      Expr padTop = op->args[CCEImg2ColArgIdx::kPadTop];
      Expr padLeft = op->args[CCEImg2ColArgIdx::kPadLeft];
      Expr padRight = op->args[CCEImg2ColArgIdx::kPadRight];
      outW_ = Simplify_cce(FloorDiv::make(fmW_ + padLeft + padRight - kernelW_, strideW) + Expr(1));
      outH_ = Simplify_cce(FloorDiv::make(fmH_ + padTop + padBottom - kernelH_, strideH) + Expr(1));
      if (!is_zero(padBottom) || !is_zero(padTop) || !is_zero(padLeft) || !is_zero(padRight)) {
        updateAddress_ = true;
        Expr res = Simplify_cce(Mul::make(outH_, outW_));
        res = Simplify_cce(Mod::make(res, Expr(16)));
        if (is_zero(res) && !multiCore_) {
          onlyUpdateUBtoGM_ = true;
          while (!stack_.empty()) {
            const For *top = stack_.top();
            stack_.pop();
            if (!stack_.empty() && stack_.top() == firstFor_) {
              c1_ = top->extent;
            }
          }
        }
      }
    }
  }

  bool updateAddress_{false};
  bool onlyUpdateUBtoGM_{false};
  bool multiCore_{false};
  std::string im2colName_;
  int im2colTimes_{0};
  Expr fmW_;
  Expr fmH_;
  Expr outW_;
  Expr outH_;
  Expr kernelH_;
  Expr kernelW_;
  Expr c1_;
  const For *firstFor_{nullptr};
  std::stack<const For *> stack_;
};

class Im2colAddressMutator : public IRMutator {
 public:
  explicit Im2colAddressMutator(const std::string &name) : callName_(name) {}
  ~Im2colAddressMutator() override = default;

  Stmt update(Stmt &s, const Expr &fmH, const Expr &fmW, const Expr &outH, const Expr &outW, const Expr &kernelH,
              const Expr &kernelW, const Expr &c1, const int &callTimes, bool onlyUpdateUBtoGM) {
    callTimes_ = callTimes;
    fmH_ = fmH;
    fmW_ = fmW;
    outH_ = outH;
    outW_ = outW;
    kernelH_ = kernelH;
    kernelW_ = kernelW;
    c1_ = c1;
    onlyUpdateUBtoGM_ = onlyUpdateUBtoGM;
    s = Mutate(s);

    return s;
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    stack_.push(op);
    Stmt res = IRMutator::Mutate_(op, s);
    auto eval = op->body.as<Evaluate>();
    if ((eval != nullptr) && eval->value.as<Call>() && (eval->value.as<Call>()->name == callName_)) {
      auto forRes = res.as<For>();
      if (forRes != nullptr) {
        if (onlyUpdateUBtoGM_) {
          res = forRes->body;
        } else {
          Expr ext = Simplify_cce(Div::make(Mul::make(forRes->extent, 16), Mul::make(outW_, outW_)));
          res = For::make(forRes->loop_var, forRes->min, ext, forRes->for_type, forRes->device_api, forRes->body);
        }
      }
    }
    stack_.pop();

    return res;
  }

  Array<Expr> updateArgs(const Array<Expr> &args, const std::unordered_map<size_t, Expr> &posMap) {
    Array<Expr> newArgs;
    for (size_t i = 0; i < args.size(); ++i) {
      auto iter = posMap.find(i);
      if (iter != posMap.end()) {
        newArgs.push_back(iter->second);
      } else {
        newArgs.push_back(args[i]);
      }
    }

    return newArgs;
  }

  Expr updateCallArgs(const Call *op, const std::unordered_map<size_t, Expr> &posMap) {
    CHECK(op);
    Array<Expr> args = op->args;
    Array<Expr> newArgs = updateArgs(args, posMap);

    return Call::make(op->type, op->name, newArgs, op->call_type, op->func, op->value_index);
  }

  Expr Mutate_(const Variable *op, const Expr &e) final {
    if (op->name_hint == "blockIdx.x") {
      batchVar_ = op;
      batchExpr = e;
    }

    return IRMutator::Mutate_(op, e);
  }

  Expr Mutate_(const Call *op, const Expr &e) final {
    if (op->name == callName_) {
      LOG(INFO) << callName_;
      Array<Expr> args = op->args;
      Expr arg0 = args[0];
      Expr arg1 = args[1];
      auto dst = arg0.as<Call>();
      auto src = arg1.as<Call>();
      CHECK(dst);
      CHECK(src);

      const For *top = nullptr;
      if (!stack_.empty()) {
        top = stack_.top();
      }

      const int c0 = 16;
      if (dst != nullptr) {
        Expr outSize = Mul::make(outW_, outH_);
        CHECK(top);
        Expr dstAddress = Mul::make(top->loop_var, Mul::make(outSize, c0));
        std::unordered_map<size_t, Expr> argsMap;
        argsMap[2] = dstAddress;
        if (onlyUpdateUBtoGM_) {
          argsMap[2] = Expr(0);
        }
        arg0 = updateCallArgs(dst, argsMap);
      }

      if (src != nullptr) {
        Expr fmSize = Mul::make(fmW_, fmH_);
        CHECK(top);
        Expr srcAddress = Mul::make(top->loop_var, Mul::make(fmSize, c0));
        std::unordered_map<size_t, Expr> argsMap;
        argsMap[2] = srcAddress;
        if (onlyUpdateUBtoGM_) {
          argsMap[2] = Expr(0);
        }
        arg1 = updateCallArgs(src, argsMap);
      }

      std::unordered_map<size_t, Expr> callArgsMap;
      callArgsMap[0] = arg0;
      callArgsMap[1] = arg1;
      Expr res = updateCallArgs(op, callArgsMap);

      return res;
    } else if (op->name == "copy_ubuf_to_gm") {
      Array<Expr> args = op->args;
      Expr arg0 = args[0];
      Expr arg1 = args[1];
      auto dst = arg0.as<Call>();
      auto src = arg1.as<Call>();
      const For *c1 = nullptr;
      const For *reduce = nullptr;
      if (stack_.size() > 2) {
        reduce = stack_.top();
        stack_.pop();
        c1 = stack_.top();
        stack_.pop();
        batchVar_ = stack_.top()->loop_var.get();
        batchExpr = stack_.top()->loop_var;
        stack_.push(c1);
        stack_.push(reduce);
      } else if (stack_.size() > 1) {
        reduce = stack_.top();
        stack_.pop();
        c1 = stack_.top();
        stack_.push(reduce);
      }

      if (dst != nullptr) {
        const int fractalBlockSize = 256;
        CHECK(reduce);
        Expr reduceOffset = Mul::make(reduce->loop_var, Mul::make(c1_, fractalBlockSize));
        CHECK(c1);
        Expr c1Offset = Mul::make(c1->loop_var, fractalBlockSize);
        Expr dstAddress = Add::make(reduceOffset, c1Offset);
        if (batchVar_ != nullptr) {
          Expr channelSize = Mul::make(c1_, 16);
          Expr kernelSize = Mul::make(kernelH_, kernelW_);
          Expr batchBlockSize = Simplify_cce(Mul::make(Mul::make(outH_, outW_), Mul::make(channelSize, kernelSize)));
          Expr batchOffset = Mul::make(batchExpr, batchBlockSize);
          dstAddress = Add::make(dstAddress, batchOffset);
        }
        std::unordered_map<size_t, Expr> argsMap;
        argsMap[2] = dstAddress;
        arg0 = updateCallArgs(dst, argsMap);
      }

      if (src != nullptr) {
        std::unordered_map<size_t, Expr> argsMap;
        argsMap[2] = Expr(0);
        arg1 = updateCallArgs(src, argsMap);
      }

      std::unordered_map<size_t, Expr> callArgsMap;
      if (needUpdate()) {
        callArgsMap[0] = arg0;
      }

      callArgsMap[1] = arg1;
      Expr res = updateCallArgs(op, callArgsMap);

      return res;
    }

    return IRMutator::Mutate_(op, e);
  }

  bool needUpdate() const { return callTimes_ == 1; }

 private:
  int callTimes_{0};
  std::string callName_;
  bool onlyUpdateUBtoGM_{false};
  Expr fmW_{0};
  Expr fmH_{0};
  Expr outW_{0};
  Expr outH_{0};
  Expr kernelH_{0};
  Expr kernelW_{0};
  Expr c1_{0};
  std::stack<const For *> stack_;
  const Variable *batchVar_{nullptr};
  Expr batchExpr;
};

Stmt CoarsenImg2Col(Stmt stmt) {
  stmt = Simplify_cce(stmt);  // do `Mod` expressions simplifications
  Im2colAddressCollector check("cce_img2col_ub");
  if (check.Collecttor(stmt)) {
    stmt = Im2colAddressMutator("cce_img2col_ub")
             .update(stmt, check.fmH_, check.fmW_, check.outH_, check.outW_, check.kernelH_, check.kernelW_, check.c1_,
                     check.im2colTimes_, check.onlyUpdateUBtoGM_);
  }

  stmt = CoarsenMad().Mutate(stmt);
  Stmt coarsened = CoarsenImg2ColMutator().Mutate(stmt);
  LowerImg2ColMutator lower;
  lower.setTimes(check.im2colTimes_);
  coarsened = lower.Mutate(coarsened);

  return coarsened;
}
}  // namespace ir
}  // namespace akg
