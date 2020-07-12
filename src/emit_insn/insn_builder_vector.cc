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

#include <tvm/runtime/packed_func.h>
#include <tvm/base.h>
#include <tvm/api_registry.h>
#include <tvm/ir_mutator.h>

#include <cmath>
#include <set>
#include <climits>
#include <numeric>

#include "common/array_api.h"
#include "cce_params.h"
#include "insn_builder.h"
#include "insn_pattern.h"
#include "pass/expr_alg_simplify.h"

namespace akg {
/// Add set_vector_mask intrin before and after a stmt
/// \param stmt
/// \param vec_arg_info
/// \param dtype
/// \return
Stmt InsertSetMaskIntrin(Stmt stmt, const VectorArgInfo &vec_arg_info, const Type &dtype) {
  if (vec_arg_info.defined()) {
    auto vec_mask = vec_arg_info->vec_mask_;
    auto first_mask = EmitSetVecMaskIntrin(Stmt(), dtype, vec_mask);
    auto second_mask = EmitSetVecMaskIntrin(Stmt(), dtype);
    stmt = Block::make({first_mask, stmt, second_mask});
  }

  return stmt;
}

/// Fill data into insn args
/// \param repeat
/// \param dst_offset
/// \param src0_offset
/// \param src1_offset
/// \return
Map<std::string, Expr> VectorInsnBuilder::GenInsnArgs(const Expr &repeat, const Expr &dst_offset,
                                                      const Expr &src0_offset, const Expr &src1_offset) {
  Map<std::string, Expr> insn_args;
  insn_args.Set("repeat", repeat);
  insn_args.Set("dstOffset", dst_offset);
  if (src1_offset.defined()) {
    insn_args.Set("src0Offset", src0_offset);
    insn_args.Set("src1Offset", src1_offset);
  } else {
    insn_args.Set("srcOffset", src0_offset);
  }

  return insn_args;
}

/// Emit intrin with repeat loop check
/// \param arg_info
/// \return
Stmt SingleVecInsnBuilder::EmitExpandedIntrin(const VectorArgInfo &arg_info) {
  CHECK(arg_info.defined());

  Stmt stmt;

  Expr repeat = arg_info->repeat_;
  int dst_stride_m0 = GetInt32Const(arg_info->dst_stride_m0_);
  int src_stride_m0 = GetInt32Const(arg_info->src_stride_m0_list_[0]);
  int dst_stride_m1 = GetInt32Const(arg_info->dst_stride_m1_);
  int src_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[0]);

  int dst_block_size = GetUbBlkSize(dst_info_->dtype_);
  int src_block_size = GetUbBlkSize(src_info_->dtype_);

  if (dst_stride_m0 >= MAX_STRIDE_M0_SINGLE || src_stride_m0 >= MAX_STRIDE_M0_SINGLE) {
    LOG(FATAL) << "StrideM0 must be less than 65536";
  }

  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;

  Var local_var = Var("broadcast_for_vec_local_UB", Handle());
  stmt = CreateBroadcast(arg_info, local_var, stmt);

  // Handle stride_m1 loop of single vector intrin, if stride_m1 > 255, it will be separated
  if (dst_stride_m1 >= MAX_STRIDE_M1 || src_stride_m1 >= MAX_STRIDE_M1) {
    auto var = Var("repeatStrideM1Idx");
    arg_info.GetNode()->dst_stride_m1_ = Expr(0);
    dst_offset = Simplify(dst_offset + var * Expr(dst_stride_m1 * dst_block_size));
    arg_info.GetNode()->src_stride_m1_list_ = {Expr(0)};
    src_offset = Simplify(src_offset + var * Expr(src_stride_m1 * src_block_size));
    stmt = InsertBody(stmt, EmitIntrinBody(arg_info, GenInsnArgs(Expr(1), dst_offset, src_offset)));
    stmt = For::make(var, Expr(0), repeat, ForType::Serial, DeviceAPI::None, stmt);
  } else {
    if (GetInt32Const(repeat) < MAX_REPEAT) {
      stmt = InsertBody(stmt, EmitIntrinBody(arg_info, GenInsnArgs(repeat, dst_offset, src_offset)));
    } else {
      stmt = InsertBody(stmt, EmitIntrinRepeatLoop(arg_info));
    }
  }

  if (!dst_info_->var_.empty() && src_info_->var_.empty() && intrin_name_ != INTRIN_NAME_VECTOR_DUP) {
    // need to broadcast src first
    stmt = Allocate::make(local_var, src_info_->dtype_, {Expr(src_block_size * FULL_BLOCK_NUM)}, const_true(), stmt);
    if (!src_info_->scope_.empty()) {
      stmt = AttrStmt::make(local_var, STORAGE_SCOPE, StringImm::make(src_info_->scope_), stmt);
    }
  }

  CHECK(stmt.defined()) << "Error: Stmt is undefined!";

  return stmt;
}

/// Emit single vector intrin
/// \param arg_info
/// \param args
/// \return
Stmt SingleVecInsnBuilder::EmitIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args) {
  Stmt body;

  CHECK(!arg_info->src_stride_m0_list_.empty());
  CHECK(!arg_info->src_stride_m1_list_.empty());

  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src_buffer_id = GenBufferId(src_info_);

  Expr repeat = args["repeat"];
  Expr dst_offset = Sub::make(args["dstOffset"], arg_info->block_offset_);
  Expr src_offset = args["srcOffset"];
  Expr src_stride_m1 = arg_info->src_stride_m1_list_[0];

  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset);
  auto src = GetAccessPtr(src_buffer_id, "r", src_offset);

  if (broadcast_buffer_.defined()) {
    src_stride_m1 = 0;
    src = GetAccessPtr(broadcast_buffer_, "r", Expr(0));
  }

  Array<Expr> stride_args = {arg_info->dst_stride_m0_, arg_info->src_stride_m0_list_[0], arg_info->dst_stride_m1_,
                             src_stride_m1};
  Array<Expr> insn_args = {dst, src, repeat};
  if (arg_info->scalar_.defined()) {
    auto scalar = arg_info->scalar_;
    if (tmp_buffer_.defined()) {
      dst = GetAccessPtr(tmp_buffer_, "w", dst_offset);
    }

    insn_args = {dst, scalar, repeat};

    if (intrin_name_ != INTRIN_NAME_VECTOR_DUP) {
      Insert(insn_args, 1, src);
    }
  }
  insn_args = MergeTwo(insn_args, stride_args);
  body = EmitCceIntrinTemplate(Stmt(), dst.type(), insn_args, intrin_name_);

  return body;
}

/// Create broadcast intrin if src is scalar
/// \param arg_info
/// \param local_var
/// \param stmt
/// \return
Stmt SingleVecInsnBuilder::CreateBroadcast(const VectorArgInfo &arg_info, const Var &local_var, Stmt stmt) {
  if (!dst_info_->var_.empty() && src_info_->var_.empty() && intrin_name_ != INTRIN_NAME_VECTOR_DUP) {
    // need to broadcast src first
    auto src_block_size = GetUbBlkSize(src_info_->dtype_);
    broadcast_buffer_ = BufferNode::make(local_var, src_info_->dtype_, {Expr(src_block_size * FULL_BLOCK_NUM)}, {},
                                         src_info_->elem_offset_, "broadcast_for_vec_local_UB", src_info_->scope_,
                                         src_info_->data_alignment_, 1, BufferType::kDefault);
    auto broad_dst = GetAccessPtr(broadcast_buffer_, "w", 0);
    Array<Expr> args = {
      broad_dst, GenBufferId(src_info_).vload({Expr(0)}, src_info_->dtype_), Expr(1), Expr(1), Expr(1), Expr(0),
      Expr(0)};
    stmt = EmitSetVecMaskIntrin(stmt, src_info_->dtype_, GetAllMask(src_info_->dtype_));
    stmt = InsertBody(stmt, EmitCceIntrinTemplate(Stmt(), src_info_->dtype_, args, INTRIN_NAME_VECTOR_DUP));
    stmt = EmitSetVecMaskIntrin(stmt, dst_info_->dtype_, arg_info->vec_mask_);
  }

  return stmt;
}

/// if repeat-size > cce_max_repeat, then split it into loop as "Davinci ISA User Guide t6.3 (8.2.2)" mentioned
/// max_cce_repeat = 255, considering params are about 2 cycles, set it to be  255 // 2 = 127
/// RepeatOffset is the offset for two cycles.
/// \param arg_info
/// \return
Stmt SingleVecInsnBuilder::EmitIntrinRepeatLoop(const VectorArgInfo &arg_info) {
  CHECK(arg_info.defined());
  Stmt stmt_body;
  Stmt stmt_tail;
  Expr repeat = arg_info->repeat_;
  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;
  int dst_stride_m1 = GetInt32Const(arg_info->dst_stride_m1_);
  CHECK(!arg_info->src_stride_m1_list_.empty());
  int src_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[0]);
  int dst_block_size = GetUbBlkSize(dst_info_->dtype_);
  int src_block_size = GetUbBlkSize(src_info_->dtype_);

  CHECK_NE(repeat_step_size_, 0);

  Expr src_repeat_offset = GetRepeatOffset(src_block_size, Expr(repeat_step_size_), src_stride_m1);
  Expr dst_repeat_offset = GetRepeatOffset(dst_block_size, Expr(repeat_step_size_), dst_stride_m1);

  // repeat body
  Expr n_loop = truncdiv(repeat, repeat_step_size_);
  CHECK(GetInt32Const(n_loop) > 0) << "Error: n_loop must be larger than 0";
  if (GetInt32Const(n_loop) == 1) {
    stmt_body = EmitIntrinBody(arg_info, GenInsnArgs(Expr(repeat_step_size_), dst_offset, src_offset));
  } else {
    auto var = VarExpr("repeatStepIdx");
    auto dst_offset_fixed = dst_offset + var * dst_repeat_offset;
    auto src_offset_fixed = src_offset + var * src_repeat_offset;
    stmt_body = EmitIntrinBody(arg_info, GenInsnArgs(Expr(repeat_step_size_), dst_offset_fixed, src_offset_fixed));
    stmt_body = For::make(var, Expr(0), n_loop, ForType::Serial, DeviceAPI::None, stmt_body);
  }

  auto stmt = stmt_body;

  // repeat tail
  if (GetInt32Const(repeat) % repeat_step_size_ > 0) {
    stmt_tail = EmitIntrinBody(
      arg_info, GenInsnArgs(Expr(GetInt32Const(repeat) % repeat_step_size_), dst_offset + n_loop * dst_repeat_offset,
                            src_offset + n_loop * src_repeat_offset));
    stmt = InsertBody(stmt, stmt_tail);
  }

  // around by attr pragma_insn_partition
  stmt = AttrStmt::make(make_zero(Int(32)), "pragma_insn_partition", Expr(0), stmt);

  return stmt;
}

/// Handle body and tail intrin of single vector
/// \return
Array<Stmt> SingleVecInsnBuilder::EmitIntrin() {
  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;
  Array<Stmt> insn_list;

  if (body_arg_info_.defined()) {
    int body_num = body_arg_info_->body_num_;
    CHECK(body_num > 0) << "Error: body num should be larger than 0";

    auto var = Var("vec_i");
    if (body_num != 1) {
      auto offset = body_arg_info_->body_offset_;
      dst_info_.GetNode()->insn_offset_ = dst_offset + var * offset;
      src_info_.GetNode()->insn_offset_ = src_offset + var * offset;
    }
    Stmt body = EmitExpandedIntrin(body_arg_info_);
    if (body_num != 1) {
      body = For::make(var, Expr(0), Expr(body_num), ForType::Serial, DeviceAPI::None, body);
    }
    body = InsertSetMaskIntrin(body, body_arg_info_, dtype_);
    insn_list.push_back(body);
  }

  if (tail_arg_info_.defined()) {
    dst_info_.GetNode()->insn_offset_ = dst_offset + tail_arg_info_->dst_head_;
    src_info_.GetNode()->insn_offset_ = src_offset + tail_arg_info_->src_head_list_[0];

    Stmt tail = EmitExpandedIntrin(tail_arg_info_);
    tail = InsertSetMaskIntrin(tail, tail_arg_info_, dtype_);
    insn_list.push_back(tail);
  }

  return insn_list;
}

/// Emit binary vector va intrin
/// \param arg_info
/// \param args
/// \return
Stmt MultiVecInsnBuilder::BinaryVecVAIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args) {
  CHECK(arg_info.defined());
  CHECK_GE(src_info_list_.size(), 2);

  auto src0_info = src_info_list_[0];
  auto src1_info = src_info_list_[1];
  CHECK(src0_info.defined());
  CHECK(src1_info.defined());

  const int va_num = 3;
  std::map<std::string, std::string> va_intrin_map = {{"vadd", "scatter_vadd"}};
  CHECK(va_intrin_map.count(intrin_name_) != 0)
    << "intrin " << intrin_name_ << " doesn't have corresponding va intrin!";

  Expr repeat = args["repeat"];
  Expr dst_offset = args["dstOffset"];
  Expr src0_offset = args["src0Offset"];
  Expr src1_offset = args["src1Offset"];

  Stmt stmt;
  Expr dst_stride = arg_info->dst_stride_m1_;
  Expr src0_stride = arg_info->src_stride_m1_list_[0];
  Expr src1_stride = arg_info->src_stride_m1_list_[1];
  const Type array_type = UInt(64);
  const Expr buffer_size = Expr(FULL_BLOCK_NUM);

  Buffer dst_buffer_id = GenBufferId(dst_info_);
  Buffer src0_buffer_id = GenBufferId(src0_info);
  Buffer src1_buffer_id = GenBufferId(src1_info);

  Array<Buffer> buffer_list;
  // create uint64 array
  for (size_t i = 0; i < va_num; ++i) {
    std::string array_name = "add_array" + std::to_string(i);
    Buffer addr_buffer = BufferNode::make(Var(array_name, Handle()), array_type, {buffer_size}, {}, Expr(), array_name,
                                          SCOPE_REG, 0, 0, BufferType::kDefault);
    buffer_list.push_back(addr_buffer);
  }

  for (int i = 0; i < FULL_BLOCK_NUM; ++i) {
    Expr dst = GetAccessPtr(dst_buffer_id, "r", dst_offset + arg_info->dst_vasrc_extent_list_[i]);
    Expr src0 = GetAccessPtr(src0_buffer_id, "r", src0_offset + arg_info->src0_vasrc_extent_list_[i]);
    Expr src1 = GetAccessPtr(src1_buffer_id, "r", src1_offset + arg_info->src1_vasrc_extent_list_[i]);
    Array<Expr> access_list = {dst, src0, src1};

    for (size_t j = 0; j < va_num; ++j) {
      Expr addr = Load::make(array_type, buffer_list[j]->data, Expr(j), const_true());
      stmt = InsertBody(stmt, Evaluate::make(Call::make(
                                UInt(64), "printer_cast",
                                {Call::make(UInt(64), "reg", {addr}, Call::Extern), access_list[j]}, Call::Extern)));
    }
  }

  // call set_va_reg_sb
  Array<Expr> va_args;
  for (size_t i = 0; i < va_num; ++i) {
    std::string va_name = "VA" + std::to_string(i);
    va_args = {StringImm::make(va_name), GetAccessPtr(buffer_list[i], "r", 0)};
    stmt = EmitCceIntrinTemplate(stmt, dst_info_->dtype_, va_args, "set_va_reg_sb");
  }

  // call scatter intrin
  auto call = Call::make(Int(32), "tvm_cce_string_print", {StringImm::make("f16")}, Call::PureIntrinsic);
  va_args = {
    call,       StringImm::make("VA2"), StringImm::make("VA0"), StringImm::make("VA1"), repeat, dst_stride, src0_stride,
    src1_stride};
  stmt = EmitCceIntrinTemplate(stmt, dst_info_->dtype_, va_args, va_intrin_map[intrin_name_]);

  // allocate storage scope
  for (size_t i = 0; i < va_num; ++i) {
    size_t real_idx = va_num - 1 - i;
    stmt = AttrStmt::make(
      buffer_list[real_idx]->data, STORAGE_SCOPE, Expr(SCOPE_REG),
      Allocate::make(buffer_list[real_idx]->data, buffer_list[real_idx]->dtype, {buffer_size}, const_true(), stmt));
  }

  return stmt;
}

/// Emit binary vector intrin
/// \param arg_info
/// \param args
/// \return
Stmt MultiVecInsnBuilder::BinaryVecIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args) {
  CHECK(arg_info.defined());
  CHECK_GE(src_info_list_.size(), 2);

  auto src0_info = src_info_list_[0];
  auto src1_info = src_info_list_[1];
  CHECK(src0_info.defined());
  CHECK(src1_info.defined());

  if (arg_info->is_vaarg_) {
    return BinaryVecVAIntrinBody(arg_info, args);
  }

  Expr repeat = args["repeat"];
  Expr dst_offset = args["dstOffset"];
  Expr src0_offset = args["src0Offset"];
  Expr src1_offset = args["src1Offset"];

  Stmt stmt;
  Expr dst_stride_m0 = arg_info->dst_stride_m0_;
  Expr src0_stride_m0 = arg_info->src_stride_m0_list_[0];
  Expr src1_stride_m0 = arg_info->src_stride_m0_list_[1];
  Expr dst_stride_m1 = arg_info->dst_stride_m1_;
  Expr src0_stride_m1 = arg_info->src_stride_m1_list_[0];
  Expr src1_stride_m1 = arg_info->src_stride_m1_list_[1];

  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src0_buffer_id = GenBufferId(src0_info);
  auto src1_buffer_id = GenBufferId(src1_info);

  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset);
  auto src0 = GetAccessPtr(src0_buffer_id, "r", src0_offset);
  auto src1 = GetAccessPtr(src1_buffer_id, "r", src1_offset);

  if (arg_info->last_axis_info_.src_op_.defined()) {
    auto src_index = arg_info->last_axis_info_.src_index_;
    auto src_op = arg_info->last_axis_info_.src_op_;
    auto new_intrin_name = arg_info->last_axis_info_.intrin_name_;
    Array<Expr> args_local;
    if (src_index == 0) {
      args_local = {dst, src1, src_op, repeat, dst_stride_m0, src1_stride_m0, dst_stride_m1, src1_stride_m1};
    } else {
      args_local = {dst, src0, src_op, repeat, dst_stride_m0, src0_stride_m0, dst_stride_m1, src0_stride_m1};
    }

    stmt = EmitCceIntrinTemplate(Stmt(), dst.type(), args_local, new_intrin_name);
  } else {
    Array<Expr> args_local = {
      dst,           src0,           src1,          repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
      dst_stride_m1, src0_stride_m1, src1_stride_m1};

    stmt = EmitCceIntrinTemplate(Stmt(), dst.type(), args_local, intrin_name_);
  }

  return stmt;
}

/// Emit multi vector intrin
/// \param arg_info
/// \param dst_offset
/// \param src_offset_list
/// \return
Stmt MultiVecInsnBuilder::MultiVecIntrinBody(const VectorArgInfo &arg_info, Expr dst_offset,
                                             Array<Expr> src_offset_list) {
  CHECK(arg_info.defined());
  Stmt stmt = Stmt();
  std::map<std::string, std::string> support_list = {
    {"vselect_LT", "vcmp_lt"}, {"vselect_EQ", "vcmp_eq"}, {"vselect_GT", "vcmp_gt"}};
  Array<Expr> src_list;

  auto dst_buffer_id = GenBufferId(dst_info_);
  auto var = VarExpr("repeatIndex");

  Expr dst_stride_m0 = arg_info->dst_stride_m0_;
  Expr dst_stride_m1 = arg_info->dst_stride_m1_;

  auto coef = Expr(block_size_ * 8);

  auto all_mask = GetAllMask(dst_info_->dtype_);
  bool mask_all = true;
  for (size_t i = 0; i < all_mask.size(); ++i) {
    if (!air::ir::Equal(arg_info->vec_mask_[i], all_mask[i])) {
      mask_all = false;
      break;
    }
  }
  if (!mask_all) {
    uint64_t low = GetUIntConst(arg_info->vec_mask_[0]);
    int low_log = 64;
    if (low < ULLONG_MAX) {
      low += 1;
      low_log = static_cast<int>(log2(low));
    }
    uint64_t high = GetUIntConst(arg_info->vec_mask_[1]);
    int high_log = 64;
    if (high < ULLONG_MAX) {
      high += 1;
      high_log = static_cast<int>(log2(high));
    }
    coef = make_const(Int(32), low_log + high_log);
  }

  if (GetInt32Const(arg_info->repeat_) > 1) {
    dst_offset = dst_offset + var * coef;
    for (size_t i = 0; i < src_info_list_.size(); ++i) {
      src_offset_list.Set(i, src_offset_list[i] + var * coef);
    }
  }

  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset);

  for (size_t i = 0; i < src_info_list_.size(); ++i) {
    if (src_info_list_[i]->buffer_.defined()) {
      src_list.push_back(GetAccessPtr(src_info_list_[i]->buffer_, "r", Expr(0)));
    } else {
      auto src = GenBufferId(src_info_list_[i]);
      src_list.push_back(GetAccessPtr(src, "r", src_offset_list[i]));
    }
  }

  CHECK(support_list.count(intrin_name_) == 1) << "Multi vector insn emitter only support vsel and vcmp intrin";
  std::string cmp_intrin_name = support_list[intrin_name_];
  CHECK(src_list.size() == 4) << cmp_intrin_name << " expect 4 operands, but found " << src_list.size() << " operands";
  Array<Expr> args = {src_list[0],
                      src_list[1],
                      Expr(1),
                      dst_stride_m0,
                      arg_info->src_stride_m0_list_[0],
                      arg_info->src_stride_m0_list_[1],
                      dst_stride_m1,
                      arg_info->src_stride_m1_list_[0],
                      arg_info->src_stride_m1_list_[1]};
  stmt = EmitCceIntrinTemplate(stmt, src_info_list_[0]->dtype_, args, cmp_intrin_name);
  args = {dst,
          src_list[2],
          src_list[3],
          Expr(1),
          dst_stride_m0,
          arg_info->src_stride_m0_list_[2],
          arg_info->src_stride_m0_list_[3],
          dst_stride_m1,
          arg_info->src_stride_m1_list_[2],
          arg_info->src_stride_m1_list_[3]};
  stmt = EmitCceIntrinTemplate(stmt, dst_info_->dtype_, args, "vsel");

  if (GetInt32Const(arg_info->repeat_) > 1) {
    stmt = For::make(var, Expr(0), arg_info->repeat_, ForType::Serial, DeviceAPI::None, stmt);
  }
  return stmt;
}

/// if repeat-size > cce_max_repeat, then split it into loop as "Davinci ISA User Guide t6.3 (8.2.2)" mentioned
/// max_cce_repeat = 255, considering params are about 2 cycles, set it to be  255 // 2 = 127
/// RepeatOffset is the offset for two cycles.
/// \param arg_info
/// \return
Stmt MultiVecInsnBuilder::BinaryVecIntrinRepeatLoop(const VectorArgInfo &arg_info) {
  CHECK(arg_info.defined());
  CHECK_GE(src_info_list_.size(), 2);

  auto src0_info = src_info_list_[0];
  auto src1_info = src_info_list_[1];
  CHECK(src0_info.defined());
  CHECK(src1_info.defined());
  CHECK_NE(repeat_step_size_, 0);

  Stmt stmt;
  Expr repeat = arg_info->repeat_;
  int dst_stride_m0 = GetInt32Const(arg_info->dst_stride_m0_);
  int dst_stride_m1 = GetInt32Const(arg_info->dst_stride_m1_);
  int src0_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[0]);
  int src1_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[1]);

  Expr dst_offset = dst_info_->insn_offset_;
  Expr src0_offset = src0_info->insn_offset_;
  Expr src1_offset = src1_info->insn_offset_;

  Expr src0_repeat_offset = GetRepeatOffset(block_size_, Expr(repeat_step_size_), src0_stride_m1);
  Expr src1_repeat_offset = GetRepeatOffset(block_size_, Expr(repeat_step_size_), src1_stride_m1);
  Expr dst_repeat_offset = GetRepeatOffset(block_size_, Expr(repeat_step_size_), dst_stride_m1);

  // repeat body
  Expr n_loop = truncdiv(repeat, repeat_step_size_);
  CHECK(GetInt32Const(n_loop) > 0) << "Error: n_loop must be larger than 0";
  if (GetInt32Const(n_loop) == 1) {
    stmt = BinaryVecIntrinBody(arg_info, GenInsnArgs(Expr(repeat_step_size_), dst_offset, src0_offset, src1_offset));
  } else {
    auto var = VarExpr("repeatStepIdx");
    auto dst_offset_fixed = dst_offset + var * dst_repeat_offset;
    auto src0_offset_fixed = src0_offset + var * src0_repeat_offset;
    auto src1_offset_fixed = src1_offset + var * src1_repeat_offset;
    stmt = BinaryVecIntrinBody(
      arg_info, GenInsnArgs(Expr(repeat_step_size_), dst_offset_fixed, src0_offset_fixed, src1_offset_fixed));
    // inner insn reduction mode, need insert pipe barrier.
    if (dst_stride_m0 == 0 || dst_stride_m1 == 0) {
      stmt = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(Int(32), 2), stmt);
    }
    stmt = For::make(var, Expr(0), n_loop, ForType::Serial, DeviceAPI::None, stmt);
  }

  // repeat tail
  if (GetInt32Const(repeat) % repeat_step_size_ > 0) {
    auto stmt_tail = BinaryVecIntrinBody(
      arg_info, GenInsnArgs(Expr(GetInt32Const(repeat) % repeat_step_size_), dst_offset + n_loop * dst_repeat_offset,
                            src0_offset + n_loop * src0_repeat_offset, src1_offset + n_loop * src1_repeat_offset));
    stmt = InsertBody(stmt, stmt_tail);
  }

  // around by attr pragma_insn_partition
  stmt = AttrStmt::make(make_zero(Int(32)), "pragma_insn_partition", Expr(0), stmt);

  return stmt;
}

/// if repeat-size > cce_max_repeat, then split it into loop as "Davinci ISA User Guide t6.3 (8.2.2)" mentioned
/// max_cce_repeat = 255, considering params are about 2 cycles, set it to be  255 // 2 = 127
/// RepeatOffset is the offset for two cycles.
/// \param arg_info
/// \param is_binary
/// \return
Stmt MultiVecInsnBuilder::MultiVecIntrinRepeatLoop(const VectorArgInfo &arg_info, bool is_binary) {
  CHECK(arg_info.defined());
  Stmt stmt;
  if (is_binary) {
    return BinaryVecIntrinRepeatLoop(arg_info);
  }

  Array<Expr> src_plain_offset_list;

  for (const auto &i : src_info_list_) {
    src_plain_offset_list.push_back(i->insn_offset_);
  }
  stmt = MultiVecIntrinBody(arg_info, dst_info_->insn_offset_, src_plain_offset_list);

  // around by attr pragma_insn_partition
  stmt = AttrStmt::make(make_zero(Int(32)), "pragma_insn_partition", Expr(0), stmt);

  return stmt;
}

/// Handle stride_m1 in binary/multi vector intrin, if stride_m1 > 255, it will be saperated
/// \param arg_info
/// \return
Stmt MultiVecInsnBuilder::EmitExpandedIntrin(const VectorArgInfo &arg_info) {
  CHECK(arg_info.defined());
  std::set<std::string> multi_vec_list = {"vselect_LT", "vselect_EQ", "vselect_GT"};
  bool is_binary = src_info_list_.size() == 2 || multi_vec_list.count(intrin_name_) == 0;
  bool use_vaintrin = false;

  Expr repeat = arg_info->repeat_;
  int dst_stride_m0 = GetInt32Const(arg_info->dst_stride_m0_);
  int dst_stride_m1 = GetInt32Const(arg_info->dst_stride_m1_);
  if (dst_stride_m0 >= MAX_STRIDE_M0) {
    LOG(INFO) << "StrideM0 larger than " << MAX_STRIDE_M0 << ", use VA intrin";
    use_vaintrin = true;
  }

  std::vector<int> src_stride_m0_list;
  std::vector<int> src_stride_m1_list;
  std::vector<int> stride_m1_overflow_list;
  for (size_t i = 0; i < arg_info->src_stride_m0_list_.size(); ++i) {
    int m0 = GetInt32Const(arg_info->src_stride_m0_list_[i]);
    int m1 = GetInt32Const(arg_info->src_stride_m1_list_[i]);
    src_stride_m0_list.push_back(m0);
    src_stride_m1_list.push_back(m1);
    if (m0 >= MAX_STRIDE_M0 && !use_vaintrin) {
      LOG(INFO) << "StrideM0 larger than " << MAX_STRIDE_M0 << ", use VA intrin";
      use_vaintrin = true;
    }
    if (m1 >= MAX_STRIDE_M1) {
      stride_m1_overflow_list.push_back(static_cast<int>(i));
    }
  }

  Expr dst_offset = dst_info_->insn_offset_;
  Array<Expr> src_offset_list;
  std::transform(src_info_list_.begin(), src_info_list_.end(), std::back_inserter(src_offset_list.CopyOnWrite()->data),
                 [](const StmtStoreInfo &v) { return (v->insn_offset_); });

  if (use_vaintrin) {
    arg_info.GetNode()->is_vaarg_ = true;
    for (int i = 0; i < 8; ++i) {
      arg_info.GetNode()->dst_vasrc_extent_list_.push_back(Expr(dst_stride_m0 * i * block_size_));
      arg_info.GetNode()->src0_vasrc_extent_list_.push_back(Expr(src_stride_m0_list[0] * i * block_size_));
      arg_info.GetNode()->src1_vasrc_extent_list_.push_back(Expr(src_stride_m0_list[1] * i * block_size_));
    }
  }

  Stmt stmt;
  if (dst_stride_m1 >= MAX_STRIDE_M1 || !stride_m1_overflow_list.empty()) {
    auto var = VarExpr("strideM1Idx");
    if (dst_stride_m1 >= MAX_STRIDE_M1) {
      arg_info.GetNode()->dst_stride_m1_ = Expr(0);
      dst_offset = Simplify(dst_offset + var * Expr(dst_stride_m0 * block_size_ * 8));
    }
    for (size_t i = 0; i < stride_m1_overflow_list.size(); ++i) {
      arg_info.GetNode()->src_stride_m1_list_.Set(i, Expr(0));
      src_offset_list.Set(i, Simplify(src_offset_list[i] + var * Expr(src_stride_m0_list[i] * block_size_ * 8)));
    }
    if (is_binary) {
      stmt = BinaryVecIntrinBody(arg_info, GenInsnArgs(Expr(1), dst_offset, src_offset_list[0], src_offset_list[1]));
    } else {
      stmt = MultiVecIntrinBody(arg_info, dst_offset, src_offset_list);
    }
    stmt = For::make(var, Expr(0), repeat, ForType::Serial, DeviceAPI::None, stmt);
  } else {
    if (GetInt32Const(repeat) < MAX_REPEAT) {
      if (is_binary) {
        stmt = BinaryVecIntrinBody(arg_info, GenInsnArgs(repeat, dst_offset, src_offset_list[0], src_offset_list[1]));
      } else {
        stmt = MultiVecIntrinBody(arg_info, dst_offset, src_offset_list);
      }
    } else {
      stmt = MultiVecIntrinRepeatLoop(arg_info, is_binary);
    }
  }

  CHECK(stmt.defined()) << "stmt is undefined!";

  return stmt;
}

/// Handle body and tail intrin in binary/multi vector intrin
/// \return
Array<Stmt> MultiVecInsnBuilder::EmitIntrin() {
  Expr dst_offset = dst_info_->insn_offset_.defined() ? dst_info_->insn_offset_ : Expr(0);
  Array<Expr> src_offset_list;
  std::transform(src_info_list_.begin(), src_info_list_.end(), std::back_inserter(src_offset_list.CopyOnWrite()->data),
                 [](const StmtStoreInfo &v) { return (v->insn_offset_.defined() ? v->insn_offset_ : Expr(0)); });

  bool is_data_dependent = arg_info_->arg_type_ == ARG_VECTOR_REDUCTION;

  Array<Stmt> insn_list;
  if (body_arg_info_.defined()) {
    int body_repeat = GetInt32Const(body_arg_info_->repeat_);
    int body_num = body_arg_info_->body_num_;
    CHECK(body_num > 0) << "Error: body_num must be larger than 0";

    auto var = VarExpr("vec_i");
    if (body_num != 1) {
      dst_info_.GetNode()->insn_offset_ = dst_offset + var * body_arg_info_->body_offset_;
      for (size_t i = 0; i < src_info_list_.size(); ++i) {
        src_info_list_[i].GetNode()->insn_offset_ = src_offset_list[i] + var * body_arg_info_->body_offset_;
      }
    }
    Stmt body = EmitExpandedIntrin(body_arg_info_);
    if (body_num != 1) {
      body = For::make(var, Expr(0), Expr(body_num), ForType::Serial, DeviceAPI::None, body);
    }
    if (is_data_dependent && body_repeat < MAX_REPEAT) {
      body = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(Int(32), 2), body);
    }
    body = InsertSetMaskIntrin(body, body_arg_info_, dtype_);
    if (body.defined()) {
      insn_list.push_back(body);
    }
  }
  if (tail_arg_info_.defined()) {
    int tail_repeat = GetInt32Const(tail_arg_info_->repeat_);
    dst_info_.GetNode()->insn_offset_ = dst_offset + tail_arg_info_->dst_head_;
    for (size_t i = 0; i < src_info_list_.size(); ++i) {
      src_info_list_[i].GetNode()->insn_offset_ = src_offset_list[i] + tail_arg_info_->src_head_list_[i];
    }

    Stmt tail = EmitExpandedIntrin(tail_arg_info_);
    if (is_data_dependent && tail_repeat < MAX_REPEAT) {
      tail = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(Int(32), 2), tail);
    }
    tail = InsertSetMaskIntrin(tail, tail_arg_info_, dtype_);
    if (tail.defined()) {
      insn_list.push_back(tail);
    }
  }

  return insn_list;
}

/// Emit reduce last axis intrin(vcadd/vcgmax/vcgmin)
/// \param arg_info
/// \param is_final_cmd
/// \param args
/// \return
Stmt ReduceLastAxisInsnBuilder::EmitIntrinBody(const VectorArgInfo &arg_info, bool is_final_cmd,
                                               const Map<std::string, Expr> &args) {
  CHECK(arg_info.defined());
  Expr repeat = args["repeat"];
  Expr dst_offset = args["dstOffset"];
  Expr src_offset = args["srcOffset"];

  Expr src = reduction_tail_intrin_ ? GetAccessPtr(local_dst_buffer_, "r", src_offset)
                                    : GetAccessPtr(GenBufferId(src_info_), "r", src_offset);
  Expr dst;
  if (is_final_cmd) {
    dst = GetAccessPtr(final_dst_buffer_, "w", dst_offset);
  } else {
    CHECK(local_dst_buffer_.defined());
    dst = GetAccessPtr(local_dst_buffer_, "w", dst_offset);
  }

  Array<Expr> insn_args = {
    dst, src, repeat, arg_info->dst_stride_m1_, arg_info->src_stride_m0_list_[0], arg_info->src_stride_m1_list_[0]};
  Stmt stmt = EmitCceIntrinTemplate(Stmt(), dst.type(), insn_args, intrin_name_);
  return stmt;
}

/// if repeat-size > cce_max_repeat, then split it into loop as "Davinci ISA User Guide t6.3 (8.2.2)" mentioned
/// max_cce_repeat = 255, considering params are about 2 cycles, set it to be  255 // 2 = 127
/// RepeatOffset is the offset for two cycles.
/// \param arg_info
/// \param is_final_cmd
/// \return
Stmt ReduceLastAxisInsnBuilder::EmitIntrinRepeatLoop(const VectorArgInfo &arg_info, bool is_final_cmd) {
  CHECK(arg_info.defined());
  Stmt stmt;
  Expr repeat = arg_info->repeat_;
  int src_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[0]);
  auto dst_offset = dst_info_->insn_offset_;
  auto src_offset = src_info_->insn_offset_;
  CHECK_NE(repeat_step_size_, 0);
  Expr src_repeat_offset = GetRepeatOffset(block_size_, Expr(repeat_step_size_), src_stride_m1);
  Expr n_loop = truncdiv(repeat, repeat_step_size_);
  // repeat body
  CHECK(GetInt32Const(n_loop) > 0) << "Error: n_loop must be larger than 0";
  if (GetInt32Const(n_loop) == 1) {
    stmt = EmitIntrinBody(arg_info, is_final_cmd, GenInsnArgs(repeat_step_size_, dst_offset, src_offset));
  } else {
    auto var = VarExpr("repeatStepIdx");
    auto dst_offset_fixed = dst_offset + var * repeat_step_size_;
    auto src_offset_fixed = src_offset + var * src_repeat_offset;
    stmt = EmitIntrinBody(arg_info, is_final_cmd, GenInsnArgs(repeat_step_size_, dst_offset_fixed, src_offset_fixed));
    stmt = For::make(var, Expr(0), n_loop, ForType::Serial, DeviceAPI::None, stmt);
  }

  // repeat tail
  if (GetInt32Const(repeat) % repeat_step_size_ > 0) {
    auto stmt_tail =
      EmitIntrinBody(arg_info, is_final_cmd,
                     GenInsnArgs(Expr(GetInt32Const(repeat) % repeat_step_size_),
                                 dst_offset + n_loop * repeat_step_size_, src_offset + n_loop * src_repeat_offset));
    stmt = InsertBody(stmt, stmt_tail);
  }

  // around by attr pragma_insn_partition
  stmt = AttrStmt::make(make_zero(Int(32)), "pragma_insn_partition", Expr(0), stmt);

  return stmt;
}

/// Emit reduce last axis intrin with repeat loop
/// \param arg_info
/// \param is_final_cmd
/// \return
Stmt ReduceLastAxisInsnBuilder::EmitExpandedIntrin(const akg::VectorArgInfo &arg_info, bool is_final_cmd) {
  CHECK(arg_info.defined());
  Stmt stmt;
  int repeat = GetInt32Const(arg_info->repeat_);
  auto src_offset = dst_insn_offset_as_src_.defined() ? dst_insn_offset_as_src_ : src_info_->insn_offset_;
  auto dst_offset = dst_info_->insn_offset_;
  if (is_final_cmd && arg_info->insn_offset_scale_factor_.defined()) {
    dst_offset = truncdiv(dst_offset, arg_info->insn_offset_scale_factor_);
    if (intrin_name_ != "vcadd") {
      dst_offset *= GetUbBlkSize(dst_info_->dtype_);
    }
  }

  if (repeat < MAX_REPEAT) {
    stmt = EmitIntrinBody(arg_info, is_final_cmd, GenInsnArgs(repeat, dst_offset, src_offset));
  } else {
    stmt = EmitIntrinRepeatLoop(arg_info, is_final_cmd);
  }

  return stmt;
}

/// Handle body, tail and mix result intrin in reduce last axis mode
/// \return
Array<Stmt> ReduceLastAxisInsnBuilder::EmitIntrin() {
  std::map<std::string, std::string> op_dict = {{"vadd", "add"}, {"vmin", "min"}, {"vmax", "max"}};
  CHECK_NE(op_dict.count(intrin_name_), 0) << "Op " << intrin_name_ << " not support reduction last axis yet!";
  CHECK(dtype_.is_float());

  Array<VectorArgInfo> reduction_tail_args = arg_info_->reduction_tail_args_;
  std::string cmd = op_dict[intrin_name_];
  if (cmd == "add") {
    if (dtype_.bits() != 32 && dtype_.bits() != 16) {
      LOG(FATAL) << "reduce_last_axis add only support float16 and float32 while dtype is " << dtype_;
    }
  } else {
    if (dtype_.bits() != 16) {
      LOG(FATAL) << "reduce_last_axis only support float16 while dtype is " << dtype_;
    }
  }

  intrin_name_ = "vc" + cmd;
  if (cmd == "max" || cmd == "min") {
    intrin_name_ = "vcg" + cmd;
  }

  auto init_dst_offset = dst_info_->insn_offset_;
  auto init_src_offset = src_info_->insn_offset_;
  int all_cmd_num = 0;
  int body_num = 0;
  if (body_arg_info_.defined()) {
    body_num = body_arg_info_->body_num_;
    all_cmd_num += body_num;
  }
  if (tail_arg_info_.defined()) {
    all_cmd_num += 1;
  }
  all_cmd_num += static_cast<int>(reduction_tail_args.size());

  Array<Stmt> statements;

  bool is_final_cmd = all_cmd_num == 1;
  if (body_arg_info_.defined()) {
    auto vec_mask = body_arg_info_->vec_mask_;

    Stmt body = EmitExpandedIntrin(body_arg_info_, is_final_cmd);
    body = InsertSetMaskIntrin(body, body_arg_info_, dtype_);
    if (is_final_cmd && vec_mask == GetAllMask(dtype_)) {
      body = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(Int(32), 2), body);
    }
    statements.push_back(body);
  }
  if (tail_arg_info_.defined()) {
    dst_info_.GetNode()->insn_offset_ = init_dst_offset + tail_arg_info_->dst_head_;
    src_info_.GetNode()->insn_offset_ = init_src_offset + tail_arg_info_->src_head_list_[0];
    is_final_cmd = all_cmd_num == body_num + 1;
    auto vec_mask = tail_arg_info_->vec_mask_;

    Stmt tail = EmitExpandedIntrin(tail_arg_info_, is_final_cmd);
    tail = InsertSetMaskIntrin(tail, tail_arg_info_, dtype_);
    if (is_final_cmd && vec_mask == GetAllMask(dtype_)) {
      tail = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(Int(32), 2), tail);
    }
    statements.push_back(tail);
  }
  // Handle reduction last axis tail
  reduction_tail_intrin_ = true;
  int index = tail_arg_info_.defined() ? 2 : 1;
  for (auto vec_arg : reduction_tail_args) {
    dst_info_.GetNode()->insn_offset_ = init_dst_offset + vec_arg->dst_head_;
    dst_insn_offset_as_src_ = init_dst_offset + vec_arg->src_head_list_[0];
    src_info_.GetNode()->insn_offset_ = init_src_offset + vec_arg->src_head_list_[0];
    is_final_cmd = all_cmd_num == body_num + index;
    Stmt mix = EmitExpandedIntrin(vec_arg, is_final_cmd);
    mix = InsertSetMaskIntrin(mix, vec_arg, dtype_);
    statements.push_back(mix);
    index += 1;
  }

  return statements;
}

/// Emit a vadd or vcg intrin to help implement reduce last axis
/// \param vadd
/// \param dst_info
/// \param arg_info
/// \param intrin_name
/// \param src_tmp_buffer
/// \param dst_insn_offset_as_src
/// \return
Stmt EmitExpandedReduceHelperIntrinTemplate(bool vadd, const StmtStoreInfo &dst_info, const VectorArgInfo &arg_info,
                                            const std::string &intrin_name, Buffer src_tmp_buffer,
                                            const Expr &dst_insn_offset_as_src = Expr()) {
  CHECK(arg_info.defined());
  CHECK(dst_info.defined());

  // get arguments
  const int repeat = GetInt32Const(arg_info->repeat_);
  Expr dst_stride_m0 = arg_info->dst_stride_m0_;
  Expr src0_stride_m0 = arg_info->src_stride_m0_list_[0];
  const int src0_stride_m1 = GetInt32Const(arg_info->src_stride_m1_list_[0]);
  const int block_size = GetUbBlkSize(dst_info->dtype_);

  Expr dst_offset = dst_info->insn_offset_;
  Expr src_offset = dst_insn_offset_as_src.defined() ? dst_insn_offset_as_src : dst_info->insn_offset_;
  auto dst_buffer = GenBufferId(dst_info);

  // generate (loop) body
  const auto body = [&src_tmp_buffer, &dst_buffer, &vadd, &dst_stride_m0, &src0_stride_m0, &src0_stride_m1, &arg_info,
                     &dst_info,
                     &intrin_name](const Expr &src_offset_fixed, const Expr dst_offset_fixed, Expr repeat) -> Stmt {
    // get address
    auto src = GetAccessPtr(src_tmp_buffer, "r", src_offset_fixed);
    auto dst = GetAccessPtr(dst_buffer, "w", dst_offset_fixed);
    Array<Expr> insn_args;
    if (vadd) {
      CHECK_GE(arg_info->src_stride_m0_list_.size(), 2);
      auto dst_as_src = GetAccessPtr(dst_buffer, "r", dst_offset_fixed);
      insn_args = {dst,
                   src,
                   dst_as_src,
                   repeat,
                   dst_stride_m0,
                   src0_stride_m0,
                   arg_info->src_stride_m0_list_[1],
                   arg_info->dst_stride_m1_,
                   src0_stride_m1,
                   arg_info->src_stride_m1_list_[1]};
    } else {
      dst = GetAccessPtr(src_tmp_buffer, "w", dst_offset_fixed);
      insn_args = {dst, src, repeat, dst_stride_m0, src0_stride_m0, src0_stride_m1};
    }
    return EmitCceIntrinTemplate(Stmt(), dst_info->dtype_, insn_args, intrin_name);
  };

  Stmt stmt_body;
  Stmt stmt_tail;
  const int step_size = MAX_REPEAT - 1;
  if (repeat < MAX_REPEAT) {
    return body(src_offset, dst_offset, Expr(repeat));
  } else {
    // if repeat-size > cce_max_repeat, then split it into loop as "Davinci ISA User Guide t6.3 (8.2.2)" mentioned
    Expr src_repeat_offset = GetRepeatOffset(block_size, step_size, src0_stride_m1);
    Expr n_loop = truncdiv(Expr(repeat), Expr(step_size));
    // repeat body
    if (GetInt32Const(n_loop) == 1) {
      stmt_body = body(src_offset, dst_offset, Expr(repeat));
    } else {
      auto idx = VarExpr("repeatStepIdx");
      auto src_offset_fixed = src_offset + idx * src_repeat_offset;
      auto dst_offset_fixed = dst_offset + idx * step_size;
      stmt_body = body(src_offset_fixed, dst_offset_fixed, Expr(step_size));
      stmt_body = For::make(idx, Expr(0), n_loop, ForType::Serial, DeviceAPI::None, stmt_body);
    }

    auto stmt = stmt_body;
    // repeat tail
    if (repeat % step_size > 0) {
      stmt_tail =
        body(src_offset + n_loop * src_repeat_offset, dst_offset + n_loop * step_size, Expr(repeat % step_size));
      stmt = InsertBody(stmt, stmt_tail);
    }
    // around by attr pragma_insn_partition
    stmt = AttrStmt::make(make_zero(Int(32)), "pragma_insn_partition", Expr(0), stmt);

    return stmt;
  }
}

/// Handle body and tail intrin in vcg intrin which helps implement reduce last axis
/// \param vadd
/// \param dst_info
/// \param arg_info
/// \param intrin_name
/// \param src_tmp_buffer
/// \return
Array<Stmt> EmitComposedReduceHelperIntrinTemplate(bool vadd, const StmtStoreInfo &dst_info, const ArgInfo &arg_info,
                                                   const std::string &intrin_name, const Buffer &src_tmp_buffer) {
  CHECK(arg_info.defined());
  CHECK(dst_info.defined());

  VectorArgInfo body_arg_info = arg_info->body_arg_info_;
  VectorArgInfo tail_arg_info = arg_info->tail_arg_info_;
  Type dtype = dst_info->dtype_;

  Array<Stmt> insn_list;

  if (body_arg_info.defined()) {
    // Reduce last axis only have 1 body
    CHECK_EQ(body_arg_info->body_num_, 1);
    Stmt body = EmitExpandedReduceHelperIntrinTemplate(vadd, dst_info, body_arg_info, intrin_name, src_tmp_buffer);
    body = InsertSetMaskIntrin(body, body_arg_info, dtype);
    insn_list.push_back(body);
  }

  if (tail_arg_info.defined()) {
    Expr dst_insn_offset_as_src = dst_info->insn_offset_ + tail_arg_info->src_head_list_[0];
    dst_info.GetNode()->insn_offset_ = dst_info->insn_offset_ + tail_arg_info->dst_head_;
    Stmt tail = EmitExpandedReduceHelperIntrinTemplate(vadd, dst_info, tail_arg_info, intrin_name, src_tmp_buffer,
                                                       dst_insn_offset_as_src);
    tail = InsertSetMaskIntrin(tail, tail_arg_info, dtype);
    insn_list.push_back(tail);
  }

  return insn_list;
}

/// Generate ArgInfo for reduction helper intrin
/// \param dst_info
/// \param for_extent
/// \param scalar
/// \param type
/// \return
ArgInfo GenReduceHelperArgInfo(StmtStoreInfo &dst_info, int for_extent, const Expr &scalar, const std::string &type) {
  std::vector<std::string> type_list = {"VecDup", "VcgCmd", "VecAdd"};
  auto iter = find(type_list.begin(), type_list.end(), type);
  CHECK(iter != type_list.end());
  size_t index = std::distance(type_list.begin(), iter);
  int vec_max_len = GetVecMaxLen(dst_info->dtype_);
  int block_size = GetUbBlkSize(dst_info->dtype_);
  // Here multiply block_size to fit vcgmax and vcgmin
  int last_dim_shape = for_extent * block_size;

  auto arg_info = ArgInfo(make_node<ArgInfoNode>());

  auto fill_args = [](VectorArgInfo &args, int body_num, const Expr &repeat, const Expr &dst_stride_m0,
                      const Expr &dst_stride_m1, const Expr &scalar, const Expr &dst_head,
                      const Array<Expr> &src_head_list, const Array<Expr> &src_stride_m0_list,
                      const Array<Expr> &src_stride_m1_list, const Array<Expr> &vec_mask) -> void {
    args.GetNode()->body_num_ = body_num;
    args.GetNode()->repeat_ = repeat;
    args.GetNode()->dst_stride_m0_ = dst_stride_m0;
    args.GetNode()->dst_stride_m1_ = dst_stride_m1;
    args.GetNode()->scalar_ = scalar;
    args.GetNode()->dst_head_ = dst_head;
    args.GetNode()->src_head_list_ = src_head_list;
    args.GetNode()->src_stride_m0_list_ = src_stride_m0_list;
    args.GetNode()->src_stride_m1_list_ = src_stride_m1_list;
    args.GetNode()->vec_mask_ = vec_mask;
  };

  auto body_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  auto tail_args = VectorArgInfo(make_node<VectorArgInfoNode>());
  bool has_body = false;
  bool has_tail = false;
  int body_len = 0;
  int tail_len = 0;
  Array<Var> elim_var;

  CHECK_NE(vec_max_len, 0);

  switch (index) {
    case 0: {
      // VecDup
      body_len = FloorTo(last_dim_shape, vec_max_len);
      if (body_len > 0) {
        fill_args(body_args, 1, body_len / vec_max_len, 1, 8, scalar, Expr(), {}, {1}, {8},
                  GetVecMask(vec_max_len, 1, dst_info->dtype_));
        has_body = true;
      }
      tail_len = last_dim_shape % vec_max_len;
      if (tail_len > 0) {
        Expr dst_head = MakeConstScalar(Int(32), body_len);
        fill_args(tail_args, 0, 1, 1, 0, scalar, dst_head, {dst_head}, {1}, {0},
                  GetVecMask(tail_len, 1, dst_info->dtype_));
        has_tail = true;
      }
      if (!dst_info->var_.empty() && GetInt32Const(GetItem(dst_info->strides_, -1)) > 0) {
        elim_var = dst_info->var_;
      }
      break;
    }
    case 1: {
      int cmd_body_len = 0;
      body_len = FloorTo(last_dim_shape, vec_max_len);
      const int repeat_stride = 8;
      if (body_len > 0) {
        int repeat = body_len / vec_max_len;
        cmd_body_len += repeat_stride * repeat;
        fill_args(body_args, 1, repeat, 1, Expr(), Expr(), Expr(), {}, {1}, {8},
                  GetVecMask(vec_max_len, 1, dst_info->dtype_));
        has_body = true;
      }
      tail_len = last_dim_shape % vec_max_len;
      if (tail_len > 0) {
        fill_args(tail_args, 0, 1, 1, Expr(), Expr(), Expr(cmd_body_len), {Expr(body_len)}, {1}, {0},
                  GetVecMask(tail_len, 1, dst_info->dtype_));
        has_tail = true;
      }
      if (!dst_info->var_.empty()) {
        elim_var = dst_info->var_;
      }
      break;
    }
    case 2: {
      body_len = FloorTo(for_extent, vec_max_len);
      if (body_len > 0) {
        fill_args(body_args, 1, body_len / vec_max_len, 1, 8, Expr(), Expr(), {}, {1, 1}, {8, 8},
                  GetVecMask(vec_max_len, 1, dst_info->dtype_));
        has_body = true;
      }
      tail_len = for_extent % vec_max_len;
      if (tail_len > 0) {
        Expr dst_head = MakeConstScalar(Int(32), body_len);
        fill_args(tail_args, 0, 1, 1, 0, Expr(), dst_head, {dst_head, dst_head}, {1, 1}, {0, 0},
                  GetVecMask(tail_len, 1, dst_info->dtype_));
        has_tail = true;
      }
      if (!dst_info->var_.empty()) {
        elim_var = dst_info->var_;
      }
      break;
    }
    default:
      LOG(FATAL) << "Unsupported instruction type";
  }

  dst_info.GetNode()->insn_offset_ = GetInsnOffset(dst_info, elim_var);
  if (has_body) {
    arg_info.GetNode()->body_arg_info_ = body_args;
  }
  if (has_tail) {
    arg_info.GetNode()->tail_arg_info_ = tail_args;
  }

  return arg_info;
}

/// Emit a series of intrins to implement last-axis reduction
/// \param dst_info
/// \param src_info
/// \param if_info
/// \param for_info
/// \param arg_info
/// \param intrin_name
/// \return
Stmt EmitCceBinaryVectorToReduceLastAxis(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info,
                                         const StmtInfo &if_info, const StmtInfo &for_info, const ArgInfo &arg_info,
                                         const std::string &intrin_name) {
  CHECK(arg_info.defined());
  CHECK(dst_info.defined());
  CHECK(src_info.defined());
  const std::set<std::string> support_list = {"vadd", "vmax", "vmin"};
  CHECK_GT(support_list.count(intrin_name), 0) << "Unsupported intrin_name " + intrin_name;

  StmtStoreInfo vec_dup_dst_info{dst_info.Copy()};
  StmtStoreInfo vcg_cmd_dst_info{dst_info.Copy()};
  StmtStoreInfo vadd_dst_info{dst_info.Copy()};

  int block_size = GetUbBlkSize(dst_info->dtype_);
  CHECK_NE(block_size, 0);

  auto MultiFold = [dst_info, block_size](int a, const Stmt &b) {
    auto for_op = b.as<For>();
    CHECK(for_op != nullptr);
    Var last_dim_var;
    int last_stride = 0;
    if (!dst_info->var_.empty()) {
      last_dim_var = GetItem(dst_info->var_, -1);
      last_stride = GetInt32Const(GetItem(dst_info->strides_, -1));
    }

    int for_ext = GetInt32Const(for_op->extent) - GetInt32Const(for_op->min);
    if (last_stride != 0) {
      for_ext *= last_stride;
    }

    if (last_dim_var.defined() && Equal(for_op->loop_var, last_dim_var) && for_ext % block_size != 0) {
      for_ext = (for_ext + block_size - 1) / block_size * block_size;
    }
    return a * for_ext;
  };
  int for_extent = std::accumulate(for_info.ops_.begin(), for_info.ops_.end(), 1, MultiFold);
  Expr scalar = MakeConstScalar(dst_info->dtype_, 0);
  if (intrin_name == "vmax") {
    scalar = dst_info->dtype_.min();
  } else if (intrin_name == "vmin") {
    scalar = dst_info->dtype_.max();
  }

  std::string vcg_cmd = "vcg" + intrin_name.substr(1, intrin_name.size() - 1);

  if (arg_info->pattern_ == PATTERN_2D_BLOCK) {
    // When in partial 2D pattern, for_extent is src_shape[-2], because for loop has been eliminated
    for_extent = GetInt32Const(GetItem(src_info->shape_, -2));
  }

  Var final_var = Var(dst_info->name_ + "_dst_tmp", Handle());
  Var local_var = Var(dst_info->name_ + "_src_tmp", Handle());
  auto insn_offset_scale_factor = Expr(block_size);
  if (arg_info->body_arg_info_.defined() && arg_info->body_arg_info_->insn_offset_scale_factor_.defined()) {
    insn_offset_scale_factor = arg_info->body_arg_info_->insn_offset_scale_factor_;
  } else if (arg_info->tail_arg_info_.defined() && arg_info->tail_arg_info_->insn_offset_scale_factor_.defined()) {
    insn_offset_scale_factor = arg_info->tail_arg_info_->insn_offset_scale_factor_;
  }
  Array<Expr> size = {make_const(Int(32), block_size * for_extent)};
  Array<Expr> safe_size = {insn_offset_scale_factor * for_extent};
  auto final_dst_buffer =
    BufferNode::make(final_var, dst_info->dtype_, size, {}, dst_info->elem_offset_, dst_info->name_, dst_info->scope_,
                     dst_info->data_alignment_, 1, BufferType::kDefault);
  auto local_dst_buffer =
    BufferNode::make(local_var, dst_info->dtype_, safe_size, {}, dst_info->elem_offset_, dst_info->name_,
                     dst_info->scope_, dst_info->data_alignment_, 1, BufferType::kDefault);

  auto vec_dup_arg_info = GenReduceHelperArgInfo(vec_dup_dst_info, for_extent, scalar, "VecDup");

  SingleVecInsnBuilder single_vec_builder = SingleVecInsnBuilder(vec_dup_dst_info, vec_dup_dst_info, vec_dup_arg_info,
                                                                 INTRIN_NAME_VECTOR_DUP, final_dst_buffer);
  auto insn_list = single_vec_builder.EmitIntrin();
  auto stmt = std::accumulate(insn_list.begin(), insn_list.end(), Stmt(),
                              [](const Stmt &s0, const Stmt &s1) { return InsertBody(s0, s1); });

  if (arg_info->pattern_ == PATTERN_2D_BLOCK) {
    // When in partial 2D pattern, use final dst buffer as tmp buffer
    local_dst_buffer = final_dst_buffer;
  }

  ReduceLastAxisInsnBuilder reduce_builder =
    ReduceLastAxisInsnBuilder(dst_info, src_info, arg_info, intrin_name, final_dst_buffer, local_dst_buffer);
  Array<Stmt> ret = reduce_builder.EmitIntrin();
  stmt = FoldInsnWithForInfo(ret, if_info, for_info, stmt);

  if (intrin_name != "vadd" && arg_info->pattern_ == PATTERN_1D) {
    auto vcg_cmd_arg_info = GenReduceHelperArgInfo(vcg_cmd_dst_info, for_extent, scalar, "VcgCmd");
    if (dst_info->dtype_ == Float(32) || dst_info->dtype_ == Int(32)) {
      auto var = VarExpr("scalar_idx");
      Array<Expr> indices(vcg_cmd_dst_info->strides_.size() - 1u, Expr(0));
      indices.push_back(var);
      auto store = final_dst_buffer.vstore(indices, final_dst_buffer.vload(indices, final_dst_buffer->dtype));
      stmt = InsertBody(
        stmt, Block::make(Stmt(), For::make(var, Expr(0), Expr(for_extent), ForType::Serial, DeviceAPI::None, store)));
    } else {
      auto vcg_list =
        EmitComposedReduceHelperIntrinTemplate(false, vcg_cmd_dst_info, vcg_cmd_arg_info, vcg_cmd, final_dst_buffer);
      stmt = std::accumulate(vcg_list.begin(), vcg_list.end(), stmt,
                             [](const Stmt &s0, const Stmt &s1) { return InsertBody(s0, s1); });
    }
  }

  auto vadd_arg_info = GenReduceHelperArgInfo(vadd_dst_info, for_extent, scalar, "VecAdd");
  auto reduce_vadd_list =
    EmitComposedReduceHelperIntrinTemplate(true, vadd_dst_info, vadd_arg_info, intrin_name, final_dst_buffer);
  stmt = std::accumulate(reduce_vadd_list.begin(), reduce_vadd_list.end(), stmt,
                         [](const Stmt &s0, const Stmt &s1) { return InsertBody(s0, s1); });

  stmt = Allocate::make(local_var, dst_info->dtype_, safe_size, const_true(), stmt);
  if (!dst_info->scope_.empty()) {
    stmt = AttrStmt::make(local_var, STORAGE_SCOPE, StringImm::make(dst_info->scope_), stmt);
  }
  stmt = Allocate::make(final_var, dst_info->dtype_, size, const_true(), stmt);
  if (!dst_info->scope_.empty()) {
    stmt = AttrStmt::make(final_var, STORAGE_SCOPE, StringImm::make(dst_info->scope_), stmt);
  }
  return stmt;
}

/// Emit a series of intrins to implement bisection reduction
/// \param wrapper
/// \param if_info
/// \param intrin_name
/// \return
Stmt EmitCceBinaryVectorToBisectionReduction(const BisectionInfoWrapper &wrapper, const StmtInfo &if_info,
                                             const std::string &intrin_name) {
  Stmt stmt;

  CHECK_GE(wrapper.bisec_info_list_.size(), 1);
  if (wrapper.bisec_info_list_.size() == 1) {
    CHECK_EQ(wrapper.bisec_info_list_[0].size(), 3);
  }

  for (size_t i = 0; i < wrapper.bisec_info_list_.size(); ++i) {
    auto dst_info = wrapper.bisec_info_list_[i][0];
    auto for_info = wrapper.for_info_list_[i];
    if (wrapper.bisec_info_list_[i].size() == 2) {
      CHECK(!wrapper.dma_arg_info_map_.empty());
      auto src_info = wrapper.bisec_info_list_[i][1];
      DmaInsnBuilder dma_builder = DmaInsnBuilder(dst_info, src_info, "copy_ubuf_to_ubuf", wrapper.dma_arg_info_map_);
      auto insn = dma_builder.EmitSingleIntrin();
      stmt = FoldInsnWithForInfo({insn}, if_info, for_info, stmt);
    } else {
      auto src_info_list = GetRange(wrapper.bisec_info_list_[i], 1, 2);
      auto arg_info = wrapper.arg_info_list_[i];
      CHECK(arg_info.defined());
      switch (arg_info->arg_type_) {
        case ARG_VECTOR_REDUCTION_LAST_AXIS: {
          stmt = InsertBody(stmt, EmitCceBinaryVectorToReduceLastAxis(dst_info, src_info_list[1], if_info, for_info,
                                                                      arg_info, intrin_name));
          break;
        }
        default: {
          MultiVecInsnBuilder builder = MultiVecInsnBuilder(dst_info, src_info_list, arg_info, intrin_name);
          auto insn = builder.EmitIntrin();
          stmt = FoldInsnWithForInfo(insn, if_info, for_info, stmt);
        }
      }
    }
  }

  // Bisect sum buffer shape is input shape while last dim is shape[-1] * stride[-1] then align to blocksize
  auto dst_info = wrapper.bisec_info_list_[0][0];
  auto GetLastStride = [&dst_info]() -> int {
    auto strides = dst_info->strides_;
    // Remove dummy last dim
    if (!dst_info->var_.empty() && GetItem(dst_info->var_, -1)->name_hint == DummyLastVar) {
      strides = RemoveItemAtIndex(strides, -1);
    }
    return strides.empty() ? 1 : GetInt32Const(GetItem(strides, -1));
  };
  int block_size = GetUbBlkSize(dst_info->dtype_);
  auto allocate_shape = wrapper.original_shape_;
  int last_stride = GetLastStride();
  int fix_last_dim = CeilTo(GetInt32Const(GetItem(allocate_shape, -1)) * last_stride, block_size);
  SetItem(allocate_shape, -1, Expr(fix_last_dim));
  stmt = Allocate::make(dst_info->data_, dst_info->dtype_, allocate_shape, const_true(), stmt);
  stmt = AttrStmt::make(dst_info->data_, STORAGE_SCOPE, Expr("local.UB"), stmt);

  return stmt;
}
}  // namespace akg
