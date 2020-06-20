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

#include <tvm/base.h>
#include <tvm/api_registry.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>

#include "cce_params.h"
#include "insn_builder.h"
#include "insn_pattern.h"

namespace akg {
Stmt EmitCor(const Expr &loop_idx, const Expr &thresh_hold, const Buffer &dst, const Buffer &BufferA,
             const Buffer &BufferB) {
  const int BOX_PER_INSN = 16;
  Stmt result;
  Stmt stmt;
  Expr call_dst;
  Expr call_src0;
  Expr call_src1;
  Array<Expr> args;
  auto dst_offset = loop_idx * BOX_PER_INSN;
  auto vec_rpn_param_dtype = UInt(64);
  const Type t_half = Float(16);

  // compare iou result with threshold
  auto masks = GetVecMask(16, 1, t_half);
  result = EmitSetVecMaskIntrin(result, t_half, masks);
  call_dst = GetAccessPtr(dst, "w", dst_offset);
  args = {call_dst, thresh_hold, Expr(1), Expr(1), Expr(1), Expr(0), Expr(0)};
  stmt = Evaluate::make(Call::make(t_half, INTRIN_NAME_VECTOR_DUP, args, Call::Extern));
  result = InsertBody(result, stmt);

  result = EmitSetVecMaskIntrin(result, t_half);
  call_dst = GetAccessPtr(BufferA, "w", Expr(0));
  call_src0 = GetAccessPtr(BufferB, "r", Expr(0));
  call_src1 = GetAccessPtr(dst, "r", dst_offset);
  args = {call_dst, call_src0, call_src1, Expr(2), Expr(1), Expr(1), Expr(0), Expr(8), Expr(8), Expr(0)};
  stmt = Evaluate::make(Call::make(t_half, "vcmpv_ge", args, Call::Extern));
  result = InsertBody(result, stmt);

  call_dst = GetAccessPtr(BufferA, "w", Expr(32));
  call_src0 = GetAccessPtr(BufferB, "r", Expr(256));
  call_src1 = GetAccessPtr(dst, "r", dst_offset);
  args = {call_dst, call_src0, call_src1, loop_idx * 2, Expr(1), Expr(1), Expr(0), Expr(8), Expr(8), Expr(0)};
  stmt = Evaluate::make(Call::make(t_half, "vcmpv_ge", args, Call::Extern));
  result = InsertBody(result, stmt);

  masks = GetVecMask(32, 1, t_half);
  result = EmitSetVecMaskIntrin(result, t_half, masks);
  args = {GetAccessPtr(dst, "w", dst_offset), make_const(t_half, 0), Expr(1), Expr(1), Expr(1), Expr(0), Expr(0)};
  stmt = Evaluate::make(Call::make(t_half, INTRIN_NAME_VECTOR_DUP, args, Call::Extern));
  result = InsertBody(result, stmt);

  result = EmitSetVecMaskIntrin(result, t_half);
  stmt = Evaluate::make(Call::make(vec_rpn_param_dtype, "set_rpn_cor_ir", {0}, Call::Extern));
  result = InsertBody(result, stmt);

  // idx << 56 || 1 << 32 || 1 << 16
  Expr selection_cond = LT::make(0, loop_idx);
  const uint64_t idx_config = ((uint64_t)1 << 56u);
  const uint64_t bits_config = ((uint64_t)1 << 32u) + ((uint64_t)1 << 16u);
  Expr Xt = Expr(loop_idx * make_const(vec_rpn_param_dtype, idx_config) + make_const(vec_rpn_param_dtype, bits_config));
  call_dst = GetAccessPtr(BufferA, "w", Expr(0));
  call_src0 = GetAccessPtr(dst, "r", Expr(0));
  args = {call_dst, call_src0, Xt};
  stmt = Evaluate::make(Call::make(t_half, "rpn_cor", args, Call::Extern));
  result = InsertBody(result, IfThenElse::make(selection_cond, stmt));

  // rpn_cor_diag
  call_dst = GetAccessPtr(dst, "w", dst_offset);
  call_src0 = GetAccessPtr(BufferA, "r", dst_offset);
  args = {call_dst, call_src0};
  stmt = Evaluate::make(Call::make(t_half, "rpn_cor_diag", args, Call::Extern));
  result = InsertBody(result, stmt);

  return result;
}

Stmt EmitIou(const Expr &loop_idx, bool nms_alloc, const int &box_num1, const Buffer &src_0, const Buffer &src_1,
             const Buffer &dst, const Buffer &BufferA, const Buffer &BufferB) {
  const int BOX_PER_INSN = 16;
  int repeat = box_num1 / BOX_PER_INSN;
  Stmt result;
  Stmt stmt;
  Expr call_dst;
  Expr call_src0;
  Expr call_src1;
  Array<Expr> args;
  Expr dst_offset = 0;
  auto vec_rpn_param_dtype = UInt(64);
  if (!nms_alloc) {
    dst_offset = box_num1 * BOX_PER_INSN * loop_idx;
  }

  // calculate the area of input box
  call_dst = GetAccessPtr(BufferB, "w", Expr(0));
  call_src0 = GetAccessPtr(src_0, "r", loop_idx * BOX_PER_INSN * 8);
  const uint64_t base_config = uint64_t(1) << 56u;
  args = {call_dst, call_src0, make_const(vec_rpn_param_dtype, base_config)};
  stmt = Evaluate::make(Call::make(Float(16), "vrpac", args, Call::Extern));
  result = InsertBody(result, stmt);

  call_dst = GetAccessPtr(BufferB, "w", BOX_PER_INSN);
  call_src0 = GetAccessPtr(src_1, "r", Expr(0));
  const uint64_t my_config = static_cast<uint64_t>(static_cast<int64_t>(repeat)) << 56u;
  args = {call_dst, call_src0, make_const(vec_rpn_param_dtype, my_config)};
  stmt = Evaluate::make(Call::make(Float(16), "vrpac", args, Call::Extern));
  result = InsertBody(result, stmt);

  call_dst = GetAccessPtr(BufferA, "w", Expr(0));
  call_src0 = GetAccessPtr(BufferB, "r", BOX_PER_INSN);
  call_src1 = GetAccessPtr(BufferB, "r", Expr(0));
  args = {call_dst, call_src0, call_src1, make_const(vec_rpn_param_dtype, my_config)};
  stmt = Evaluate::make(Call::make(Float(16), "vaadd", args, Call::Extern));
  result = InsertBody(result, stmt);

  call_dst = GetAccessPtr(BufferB, "w", Expr(0));
  call_src0 = GetAccessPtr(src_1, "r", Expr(0));
  call_src1 = GetAccessPtr(src_0, "r", loop_idx * BOX_PER_INSN * 8);
  args = {call_dst, call_src0, call_src1, make_const(vec_rpn_param_dtype, my_config)};
  stmt = Evaluate::make(Call::make(Float(16), "viou", args, Call::Extern));
  result = InsertBody(result, stmt);

  // Here The repeat overflow only consider when box is less than 4K, in which repeat is less than 255 * 2.
  // Need to optimize for further demend.
  const int half_max_repeat = 128;
  const int max_repeat = 255;
  const int vec_max_len = 128;
  int repeat_offset = repeat >= half_max_repeat ? max_repeat * vec_max_len : 0;
  int repeat_last = repeat >= half_max_repeat ? repeat * 2 - max_repeat : repeat * 2;

  if (repeat >= half_max_repeat) {
    call_dst = GetAccessPtr(BufferA, "w", Expr(0));
    call_src0 = GetAccessPtr(BufferA, "r", Expr(0));
    call_src1 = GetAccessPtr(BufferB, "r", Expr(0));
    args = {call_dst, call_src0, call_src1, Expr(255), Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
    stmt = Evaluate::make(Call::make(Float(16), "vsub", args, Call::Extern));
    result = InsertBody(result, stmt);
  }
  call_dst = GetAccessPtr(BufferA, "w", repeat_offset);
  call_src0 = GetAccessPtr(BufferA, "r", repeat_offset);
  call_src1 = GetAccessPtr(BufferB, "r", repeat_offset);
  args = {call_dst, call_src0, call_src1, repeat_last, Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
  stmt = Evaluate::make(Call::make(Float(16), "vsub", args, Call::Extern));
  result = InsertBody(result, stmt);

  if (repeat >= half_max_repeat) {
    call_dst = GetAccessPtr(dst, "w", dst_offset);
    call_src0 = GetAccessPtr(BufferA, "r", Expr(0));
    args = {call_dst, call_src0, max_repeat, Expr(1), Expr(1), Expr(8), Expr(8)};
    stmt = Evaluate::make(Call::make(Float(16), "vrec", args, Call::Extern));
    result = InsertBody(result, stmt);
  }
  call_dst = GetAccessPtr(dst, "w", dst_offset + repeat_offset);
  call_src0 = GetAccessPtr(BufferA, "r", repeat_offset);
  args = {call_dst, call_src0, repeat_last, Expr(1), Expr(1), Expr(8), Expr(8)};
  stmt = Evaluate::make(Call::make(Float(16), "vrec", args, Call::Extern));
  result = InsertBody(result, stmt);

  if (repeat == 1 || nms_alloc) {
    if (repeat >= half_max_repeat) {
      call_dst = GetAccessPtr(dst, "w", dst_offset);
      call_src0 = GetAccessPtr(BufferB, "r", Expr(0));
      call_src1 = GetAccessPtr(dst, "r", dst_offset);
      args = {call_dst, call_src0, call_src1, max_repeat, Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
      stmt = Evaluate::make(Call::make(Float(16), "vmul", args, Call::Extern));
      result = InsertBody(result, stmt);
    }

    call_dst = GetAccessPtr(dst, "w", dst_offset + repeat_offset);
    call_src0 = GetAccessPtr(BufferB, "r", repeat_offset);
    call_src1 = GetAccessPtr(dst, "r", dst_offset + repeat_offset);
    args = {call_dst, call_src0, call_src1, repeat_last, Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
    stmt = Evaluate::make(Call::make(Float(16), "vmul", args, Call::Extern));
    result = InsertBody(result, stmt);
  } else {
    if (repeat >= half_max_repeat) {
      call_dst = GetAccessPtr(BufferA, "w", Expr(0));
      call_src0 = GetAccessPtr(BufferB, "r", Expr(0));
      call_src1 = GetAccessPtr(dst, "r", dst_offset);
      args = {call_dst, call_src0, call_src1, max_repeat, Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
      stmt = Evaluate::make(Call::make(Float(16), "vmul", args, Call::Extern));
      result = InsertBody(result, stmt);
    }

    call_dst = GetAccessPtr(BufferA, "w", repeat_offset);
    call_src0 = GetAccessPtr(BufferB, "r", repeat_offset);
    call_src1 = GetAccessPtr(dst, "r", dst_offset + repeat_offset);
    args = {call_dst, call_src0, call_src1, repeat_last, Expr(1), Expr(1), Expr(1), Expr(8), Expr(8), Expr(8)};
    stmt = Evaluate::make(Call::make(Float(16), "vmul", args, Call::Extern));
    result = InsertBody(result, stmt);

    auto i_var = VarExpr("iVar");
    call_dst = GetAccessPtr(dst, "w", dst_offset + BOX_PER_INSN * i_var);
    call_src0 = GetAccessPtr(BufferA, "r", BOX_PER_INSN * BOX_PER_INSN * i_var);
    args = {call_dst, call_src0, Expr(0), Expr(16), Expr(1), Expr(0), repeat - 1};
    stmt = Evaluate::make(Call::make(Float(16), "copy_ubuf_to_ubuf", args, Call::Extern));
    stmt = For::make(i_var, Expr(0), Expr(repeat), ForType::Serial, DeviceAPI::None, stmt);
    result = InsertBody(result, stmt);
  }
  return result;
}

Stmt EmitProposalSort(const Stmt &store, const Buffer &src, const Buffer &dst, bool topksort) {
  const int RG_PRO_ELEM = 8;
  Stmt result;
  Expr call_dst;
  Expr call_src0;
  Expr call_src1;
  Array<Expr> args;
  CHECK(store.as<Store>());
  CHECK(store.as<Store>()->value.as<Call>());
  int topk = GetInt32Const(store.as<Store>()->value.as<Call>()->args[2]);
  int sort_len = GetInt32Const(src->shape[0]);
  int buf_len = sort_len;
  std::vector<int> base_sort_len_list = {16, 64, 256, 1024, 4096};
  int base_len = 16;
  auto vec_rpn_param_dtype = UInt(64);

  auto it = std::find_if(base_sort_len_list.begin(), base_sort_len_list.end(),
                         [sort_len](const int &len) { return len >= sort_len; });
  if (it != base_sort_len_list.end()) {
    base_len = *it;
  }
  CHECK(base_len >= sort_len);
  if (sort_len < topk * 2 && topksort) {
    buf_len = topk * 2;
  }
  VarExpr reg_addr = VarExpr("reg_addr", vec_rpn_param_dtype);
  Var reg_addr_buf = Var("reg_addr_buf", vec_rpn_param_dtype);
  Buffer reg_addr_buffer = BufferNode::make(reg_addr_buf, vec_rpn_param_dtype, {4}, Array<Expr>(), Expr(),
                                            "reg_addr_buf", SCOPE_UBUF, 0, 0, BufferType::kDefault);
  Var sort_buf = Var("sort_buf", Float(16));
  Buffer sort_buffer = BufferNode::make(sort_buf, Float(16), {buf_len, 8}, Array<Expr>(), Expr(), "sort_buf",
                                        SCOPE_UBUF, 0, 0, BufferType::kDefault);

  // Sort data to make it sorted in blocks of 16 elements
  const int sort_len_per_repeat = 16;
  auto Sort16 = [&result, vec_rpn_param_dtype](const Buffer buf_a, Buffer buf_b, int sort_len,
                                               int init_offset = 0) -> Buffer {
    const int max_repeat = 255;
    int repeat_times = sort_len / sort_len_per_repeat;
    int norm_repeat = repeat_times / max_repeat;
    int last_repeat = repeat_times % max_repeat;
    // sort multiply block in uint of 16
    for (int i = 0; i < norm_repeat; ++i) {
      auto call_dst = GetAccessPtr(buf_b, "w", i * max_repeat * sort_len_per_repeat * RG_PRO_ELEM + init_offset);
      auto call_src0 = GetAccessPtr(buf_a, "r", i * max_repeat * sort_len_per_repeat * RG_PRO_ELEM + init_offset);
      auto args = {call_dst, call_src0, make_const(vec_rpn_param_dtype, (uint64_t)255 << 56u)};
      Stmt stmt = Evaluate::make(Call::make(Float(16), "vbitsort", args, Call::Extern));
      result = InsertBody(result, stmt);
    }
    if (last_repeat) {
      auto call_dst =
        GetAccessPtr(buf_b, "w", norm_repeat * max_repeat * sort_len_per_repeat * RG_PRO_ELEM + init_offset);
      auto call_src0 =
        GetAccessPtr(buf_a, "r", norm_repeat * max_repeat * sort_len_per_repeat * RG_PRO_ELEM + init_offset);
      auto args = {call_dst, call_src0, make_const(vec_rpn_param_dtype, (uint64_t)(uint32_t)last_repeat << 56u)};
      Stmt stmt = Evaluate::make(Call::make(Float(16), "vbitsort", args, Call::Extern));
      result = InsertBody(result, stmt);
    }
    return buf_b;
  };

  // Sort data to make it sorted in blocks of base_len elements
  // it is a general sort module. But because of the hardware limitation,
  // we should deal with the special case when it is larger than 4096, see the if target_n > 4096
  std::function<Buffer(Buffer, Buffer, int, int, int)> SortN;
  SortN = [&result, &SortN, reg_addr, reg_addr_buffer, Sort16, vec_rpn_param_dtype](
            const Buffer buf_a, const Buffer buf_b, int sort_len, int base_len, int init_offset = 0) -> Buffer {
    const int target_n = base_len;
    CHECK_NE(target_n, 0);
    constexpr int factor = 4;
    int income_n;
    Buffer pre_dst;
    // make the list sorted in block
    if (target_n == sort_len_per_repeat) {
      return Sort16(buf_a, buf_b, sort_len, init_offset);
    } else {
      income_n = target_n / factor;
      pre_dst = SortN(buf_a, buf_b, sort_len, income_n, init_offset);
    }
    int repeat_times = sort_len / target_n;
    int left_len = sort_len % target_n;
    auto src_buf = pre_dst;
    auto dst_buf = pre_dst == buf_b ? buf_a : buf_b;
    if (repeat_times > 0) {
      for (int j = 0; j < factor; ++j) {
        int i_offset = j * income_n * RG_PRO_ELEM + init_offset;
        auto value =
          Call::make(vec_rpn_param_dtype, "address_value", {GetAccessPtr(src_buf, "r", i_offset)}, Call::Extern) *
          make_const(vec_rpn_param_dtype, 2);
        Stmt store_local = Store::make(reg_addr, value, make_const(Int(32), j), const_true(1));
        result = InsertBody(result, store_local);
        auto load = Load::make(reg_addr.type(), reg_addr, j, const_true(1));

        auto call_dst_t0 = GetAccessPtr(reg_addr_buffer, "w", j);
        auto args_t0 = {call_dst_t0, Call::make(reg_addr.type(), "reg", {load}, Call::Extern)};
        Stmt stmt = Evaluate::make(Call::make(vec_rpn_param_dtype, INTRIN_NAME_REG_MOV, args_t0, Call::Extern));
        result = InsertBody(result, stmt);
      }
      auto call_dst_t1 = GetAccessPtr(dst_buf, "w", init_offset);
      auto call_src0_t1 = GetAccessPtr(reg_addr_buffer, "r");
      Array<Expr> args_t1 = {call_dst_t1, call_src0_t1, repeat_times, income_n, income_n,
                             income_n,    income_n,     Expr(0),      Expr(15)};
      Stmt stmt = Evaluate::make(Call::make(Float(16), "vmrgsort4", args_t1, Call::Extern));
      result = InsertBody(result, stmt);
    }
    if (left_len > 0) {
      // sort the left length of list, left_len would less than target_n
      // use merge sort as insert sort, left_len is already sorted in
      // blocks (income_n size sorted)
      std::vector<int> list_len(factor, 0);
      int tmp = left_len;
      for (int i = 0; i < factor; ++i) {
        if (tmp > income_n) {
          list_len[i] = income_n;
        } else {
          list_len[i] = tmp;
          break;
        }
        tmp -= income_n;
      }
      // left length is less than 16, just need a move intrinsic
      if (list_len[1] == 0) {
        auto call_dst = GetAccessPtr(dst_buf, "w", (repeat_times * target_n) * RG_PRO_ELEM + init_offset);
        auto call_src0 = GetAccessPtr(src_buf, "r", (repeat_times * target_n) * RG_PRO_ELEM + init_offset);
        Array<Expr> args = {call_dst, call_src0, Expr(0), Expr(1), left_len * RG_PRO_ELEM / sort_len_per_repeat,
                            Expr(1),  Expr(1)};
        auto stmt = Evaluate::make(Call::make(Float(16), "copy_ubuf_to_ubuf", args, Call::Extern));
        result = InsertBody(result, stmt);
      } else {
        int mask_signal = 0;
        for (int j = 0; j < factor; ++j) {
          if (list_len[j] == 0) {
            break;
          }
          int i_offset = (repeat_times * factor + j) * income_n * RG_PRO_ELEM + init_offset;
          auto addr_value =
            Call::make(vec_rpn_param_dtype, "address_value", {GetAccessPtr(src_buf, "r", i_offset)}, Call::Extern);
          auto value = addr_value * make_const(vec_rpn_param_dtype, 2);
          Stmt store_local = Store::make(reg_addr, value, make_const(Int(32), j), const_true(1));
          result = InsertBody(result, store_local);
          auto load = Load::make(reg_addr.type(), reg_addr, j, const_true(1));

          auto call_dst = GetAccessPtr(reg_addr_buffer, "w", j);
          auto args = {call_dst, Call::make(reg_addr.type(), "reg", {load}, Call::Extern)};
          Stmt stmt = Evaluate::make(Call::make(vec_rpn_param_dtype, INTRIN_NAME_REG_MOV, args, Call::Extern));
          result = InsertBody(result, stmt);
          mask_signal = mask_signal * 2 + 1;
        }

        auto call_dst = GetAccessPtr(dst_buf, "w", (repeat_times * target_n) * RG_PRO_ELEM + init_offset);
        auto call_src0 = GetAccessPtr(reg_addr_buffer, "r");
        Array<Expr> args = {call_dst,    call_src0,   Expr(1), list_len[0], list_len[1],
                            list_len[2], list_len[3], Expr(0), mask_signal};
        Stmt stmt = Evaluate::make(Call::make(Float(16), "vmrgsort4", args, Call::Extern));
        result = InsertBody(result, stmt);
      }
    }
    return dst_buf;
  };

  auto res_buf = SortN(src, sort_buffer, sort_len, base_len, 0);
  auto len_burst = topk / 2 + topk % 2;
  if (topksort) {
    for (int j = 0; j < 2; ++j) {
      Buffer buf = j == 0 ? res_buf : dst;
      auto value = Call::make(vec_rpn_param_dtype, "address_value", {GetAccessPtr(buf, "r", Expr(0))}, Call::Extern) *
                   make_const(vec_rpn_param_dtype, 2);
      Stmt store_local = Store::make(reg_addr, value, make_const(Int(32), j), const_true(1));
      result = InsertBody(result, store_local);
      auto load = {Load::make(reg_addr.type(), reg_addr, j, const_true(1))};
      call_dst = GetAccessPtr(reg_addr_buffer, "w", j);
      args = {call_dst, Call::make(reg_addr.type(), "reg", load, Call::Extern)};
      Stmt stmt = Evaluate::make(Call::make(vec_rpn_param_dtype, INTRIN_NAME_REG_MOV, args, Call::Extern));
      result = InsertBody(result, stmt);
    }
    auto dst_buf = res_buf == src ? sort_buffer : src;
    call_dst = GetAccessPtr(dst_buf, "w");
    call_src0 = GetAccessPtr(reg_addr_buffer, "r");
    args = {call_dst, call_src0, Expr(1), topk, topk, Expr(0), Expr(0), Expr(0), Expr(3)};
    Stmt stmt = Evaluate::make(Call::make(Float(16), "vmrgsort4", args, Call::Extern));
    result = InsertBody(result, stmt);
    res_buf = dst_buf;
  }
  call_dst = GetAccessPtr(dst, "w");
  call_src0 = GetAccessPtr(res_buf, "r");
  args = {call_dst, call_src0, Expr(0), Expr(1), len_burst, Expr(1), Expr(1)};
  Stmt stmt = Evaluate::make(Call::make(Float(16), "copy_ubuf_to_ubuf", args, Call::Extern));
  result = InsertBody(result, stmt);

  result = Allocate::make(sort_buf, Float(16), {buf_len, 8}, const_true(), result);
  result = AttrStmt::make(sort_buf, STORAGE_SCOPE, Expr(SCOPE_UBUF), result);
  result = Allocate::make(reg_addr_buf, vec_rpn_param_dtype, {make_const(Int(32), 4)}, const_true(), result);
  result = AttrStmt::make(reg_addr_buf, STORAGE_SCOPE, Expr(SCOPE_UBUF), result);
  result = Allocate::make(reg_addr, vec_rpn_param_dtype, {make_const(Int(32), 4)}, const_true(), result);
  result = AttrStmt::make(reg_addr, STORAGE_SCOPE, Expr(SCOPE_REG), result);
  return result;
}
}  // namespace akg
