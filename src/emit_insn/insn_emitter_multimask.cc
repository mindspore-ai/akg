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

#include "insn_emitter_multimask.h"
#include <bitset>
#include "pass/ir_util.h"
#include "common/array_api.h"
#include "insn_builder.h"
#include "insn_info.h"
#include "insn_pattern.h"

namespace akg {
namespace ir {
namespace {
int LeastCommonMultiple(int a, int b) { return (a * b / air::ir::gcd(a, b)); }

int GetTensorSize(Array<Expr> shape) {
  int size = 1;
  for (auto dim : shape) {
    CHECK(dim.as<IntImm>());
    size = size * dim.as<IntImm>()->value;
  }
  return size;
}

Array<Expr> GetVecMaskWithOffset(int data_len, int begin, Type data_type) {
  int vec_max_len = GetVecMaxLen(data_type);
  if (data_len + begin > vec_max_len || data_len < 1) {
    LOG(FATAL) << "Get vector mask error.";
  }
  if (vec_max_len != 64 && vec_max_len != 128) {
    LOG(FATAL) << "Error: mask length is error.";
  }

  std::bitset<128> submask, mask;
  for (int i = 0; i < 64; i++) {
    submask.set(i);
  }
  for (int i = begin; i < begin + data_len; i++) {
    mask.set(i);
  }

  Array<Expr> ret;
  ret.push_back(Expr((uint64_t)((mask >> 64) & submask).to_ullong()));
  ret.push_back(Expr((uint64_t)(mask & submask).to_ullong()));
  return ret;
}
}  // namespace

Stmt MultiMaskEmitter(const Stmt &stmt) {
  StmtInfoList dst_info_list;
  StmtInfoList src_info_list;
  StmtStoreInfo scalar_info;
  StmtInfo for_info;
  StmtInfo if_info;
  GetCompactComputationInfo(stmt, dst_info_list, src_info_list, if_info, for_info, true, true);
  CHECK(!dst_info_list.empty());
  CHECK(!src_info_list.empty());
  Type i_type = dst_info_list[0]->dtype_;

  const int block_size = GetUbBlkSize(dst_info_list[0]->dtype_);
  const int simd_size = block_size * 8;
  int broadcast_len = GetTensorSize(dst_info_list[0]->shape_) / GetTensorSize(src_info_list[0]->shape_);
  int data_len = GetTensorSize(src_info_list[0]->shape_);

  Array<Buffer> dst_list, src_list;
  GetBufferIdFromStmt(stmt, dst_list, src_list);
  CHECK(!dst_list.empty());
  CHECK(!src_list.empty());
  Buffer dst_buffer_id = dst_list[0];
  Buffer src_buffer_id = src_list[0];

  int i_loop_elements = LeastCommonMultiple(broadcast_len, block_size);
  int mask_num = i_loop_elements / broadcast_len;

  Stmt body = Evaluate::make(0);

  for (int i = 0; i < std::min(mask_num, data_len); i++) {
    auto i_var = VarExpr("broadcast_idx" + std::to_string(i));
    auto dst_block_offset = (i * broadcast_len) % block_size;
    auto dst_block_cnt = (i * broadcast_len) - dst_block_offset;

    CHECK_NE(i_loop_elements, 0);
    auto loop_num = (data_len + mask_num - 1 - i) * broadcast_len / i_loop_elements;

    // GenHead
    Expr base_addr_offset = i_var * i_loop_elements + dst_block_cnt;
    auto base_src = src_buffer_id.vload({i_var * mask_num + i}, i_type);
    int head_size = std::min(simd_size - dst_block_offset, broadcast_len);
    auto vec_mask_head = GetVecMaskWithOffset(head_size, dst_block_offset, i_type);
    auto head_mask = EmitSetVecMaskIntrin(Stmt(), i_type, vec_mask_head);
    auto base_dst = GetAccessPtr(dst_buffer_id, "w", base_addr_offset);
    auto head_dump =
      Evaluate::make(Call::make(i_type, "vector_dup", {base_dst, base_src, 1, 1, 1, 1, 1}, Call::Extern));
    auto head = Block::make({head_mask, head_dump});
    auto head_stmt = For::make(i_var, Expr(0), Expr(loop_num), ForType::Serial, DeviceAPI::None, head);
    auto ret_stmt = head_stmt;

    // GenBody
    if (dst_block_offset + broadcast_len >= simd_size * 2) {
      auto i_var_body = VarExpr(i_var->name_hint + "_body");
      Expr base_addr_offset_body = i_var_body * i_loop_elements + dst_block_cnt;
      auto base_src_body = src_buffer_id.vload({i_var_body * mask_num + i}, i_type);
      int repeat_size = (dst_block_offset + broadcast_len) / simd_size - 1;
      auto vec_mask_body = GetVecMaskWithOffset(simd_size, 0, i_type);
      auto full_mask = EmitSetVecMaskIntrin(Stmt(), i_type, vec_mask_body);

      int body_addr_offset = simd_size;
      Expr body_dst = GetAccessPtr(dst_buffer_id, "w", base_addr_offset_body + body_addr_offset);
      auto body_dump =
        Evaluate::make(Call::make(i_type, "vector_dup", {body_dst, base_src_body, repeat_size, 1, 1, 8, 8}, Call::Extern));
      auto body_gen = Block::make({full_mask, body_dump});
      auto body_stmt = For::make(i_var_body, Expr(0), Expr(loop_num), ForType::Serial, DeviceAPI::None, body_gen);
      ret_stmt = Block::make(ret_stmt, body_stmt);
    }

    // GenTail
    if ((dst_block_offset + broadcast_len) % simd_size != 0 && dst_block_offset + broadcast_len > simd_size) {
      auto i_var_tail = VarExpr(i_var->name_hint + "_tail");
      Expr base_addr_offset_body = i_var_tail * i_loop_elements + dst_block_cnt;
      auto base_src_body = src_buffer_id.vload({i_var_tail * mask_num + i}, i_type);
      int tail_size = (dst_block_offset + broadcast_len) % simd_size;
      auto vec_mask_tail = GetVecMaskWithOffset(tail_size, 0, i_type);
      auto tail_mask = EmitSetVecMaskIntrin(Stmt(), i_type, vec_mask_tail);
      int tail_addr_offset = dst_block_offset + broadcast_len - tail_size;
      Expr tail_dst = GetAccessPtr(dst_buffer_id, "w", base_addr_offset_body + tail_addr_offset);
      auto tail_dump =
        Evaluate::make(Call::make(i_type, "vector_dup", {tail_dst, base_src_body, 1, 1, 1, 1, 1}, Call::Extern));
      auto tail = Block::make({tail_mask, tail_dump});
      auto tail_stmt = For::make(i_var_tail, Expr(0), Expr(loop_num), ForType::Serial, DeviceAPI::None, tail);
      ret_stmt = Block::make(ret_stmt, tail_stmt);
    }

    body = Block::make({body, ret_stmt});
  }
  return body;
}
}  // namespace ir
}  // namespace akg
