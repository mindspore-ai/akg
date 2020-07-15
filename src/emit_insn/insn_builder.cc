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

#include "insn_builder.h"

#include <tvm/runtime/packed_func.h>
#include <tvm/base.h>
#include <tvm/api_registry.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_visitor.h>

#include <climits>
#include <cmath>
#include <algorithm>
#include <utility>

#include "common/array_api.h"
#include "cce_params.h"
#include "insn_pattern.h"

namespace akg {
using air::runtime::PackedFunc;
using air::runtime::TVMArgs;
using air::runtime::TVMRetValue;

/// extent is the memory size that will be read/wrote by the cce_instruction
/// reference "Davinci ISA User Guide 8.2.2"
/// \param block_size
/// \param repeat
/// \param stride_m1
/// \return
Expr GetRepeatOffset(const int &block_size, const Expr &repeat, const int &stride_m1) {
  if (stride_m1 > 0) {
    return repeat * stride_m1 * block_size;
  }

  return Expr(0);
}

/// Generate IfThenElse block with given if information
/// \param stmt
/// \param if_info
/// \param for_info
/// \param if_stmt
/// \return
Stmt GenIf(Stmt stmt, StmtInfo &if_info, StmtInfo &for_info, const Stmt &if_stmt) {
  auto if_op = if_stmt.as<IfThenElse>();

  if_info.vars_ = {};
  size_t idx = 0;
  if (GetIndexOfElement(if_info.ops_, if_stmt, idx)) {
    if_info.ops_ = RemoveItemAtIndex(if_info.ops_, idx);
  }
  for (auto op : if_info.ops_) {
    CHECK(op.as<IfThenElse>());
    auto vars = GetVarsInExpr(op.as<IfThenElse>()->condition);
    for (auto var : vars) {
      if_info.vars_.push_back(var);
    }
  }

  // When Load Expr in if_op.condition, extract the condition as a reg, and use the reg as the if condition
  Array<NodeRef> loads;
  PackedFunc count_load = PackedFunc([&loads](const TVMArgs args, TVMRetValue *ret) {
    const auto &s_ptr = args[0].operator ObjectRef();
    if (s_ptr->IsInstance<Load>()) {
      loads.push_back(Downcast<NodeRef>(s_ptr));
    }
  });
  CHECK(if_op != nullptr);
  PostOrderVisit(if_op->condition, count_load);

  if (!loads.empty()) {
    auto buffer_var = VarExpr("reg_buf");
    auto store = Store::make(buffer_var, if_op->condition, Expr(0), Expr(1));

    stmt = GenIfAndFor(stmt, if_info, for_info, false);
    if (stmt.defined()) {
      stmt = IfThenElse::make(Load::make(Bool(), buffer_var, Expr(0), Expr(1)), stmt);
    } else {
      stmt = IfThenElse::make(Load::make(Bool(), buffer_var, Expr(0), Expr(1)), Evaluate::make(0));
    }

    stmt = InsertBody(stmt, store, false);
    stmt = Allocate::make(buffer_var, Bool(), {Expr(1)}, make_const(Bool(), 1), stmt);
    stmt = AttrStmt::make(buffer_var, STORAGE_SCOPE, StringImm::make(SCOPE_REG), stmt);
  } else {
    stmt = GenIfAndFor(stmt, if_info, for_info, false);
    if (stmt.defined()) {
      stmt = IfThenElse::make(Simplify(if_op->condition), stmt);
    } else {
      stmt = IfThenElse::make(Simplify(if_op->condition), Evaluate::make(0));
    }
  }

  return stmt;
}

/// Generate For block with given for information
/// \param stmt
/// \param if_info
/// \param for_info
/// \param for_var_idx
/// \return
Stmt GenFor(Stmt stmt, StmtInfo &if_info, StmtInfo &for_info, size_t for_var_idx) {
  auto for_var = GetItem(for_info.vars_, for_var_idx);
  auto for_stmt = GetItem(for_info.ops_, for_var_idx);

  for_info.RemoveItem(for_var_idx);

  stmt = GenIfAndFor(stmt, if_info, for_info, false);
  auto for_op = for_stmt.as<For>();
  CHECK(for_op != nullptr);
  if (stmt.defined()) {
    if (Equal(for_op->extent, Expr(1))) {
      stmt = substitute(for_var, Expr(0), stmt);
      stmt = Simplify(stmt);
    } else {
      stmt = For::make(for_var, for_op->min, for_op->extent, for_op->for_type, for_op->device_api, stmt);
    }
  } else {
    stmt = For::make(for_var, for_op->min, for_op->extent, for_op->for_type, for_op->device_api, Evaluate::make(0));
  }

  return stmt;
}

/// Generate If block and For block recursively
/// \param stmt
/// \param if_info
/// \param for_info
/// \param need_reverse
/// \return
Stmt GenIfAndFor(Stmt stmt, StmtInfo &if_info, StmtInfo &for_info, bool need_reverse) {
  if (need_reverse) {
    for_info.vars_ = Reverse(for_info.vars_);
    for_info.ops_ = Reverse(for_info.ops_);
  }

  if (!if_info.ops_.empty()) {
    auto if_op = if_info.ops_[0];
    bool emit_if = true;
    CHECK(if_op.as<IfThenElse>());
    auto index_vars = GetVarsInExpr(if_op.as<IfThenElse>()->condition);
    for (auto var : index_vars) {
      if (IsInArray(for_info.vars_, var)) {
        emit_if = false;
        break;
      }
    }
    if (emit_if) {
      stmt = GenIf(stmt, if_info, for_info, if_op);
    }
  }

  if (!for_info.vars_.empty()) {
    stmt = GenFor(stmt, if_info, for_info, for_info.vars_.size() - 1);
  }
  return stmt;
}

/// Insert body before or after stmt
/// \param stmt
/// \param body
/// \param after
/// \return
Stmt InsertBody(Stmt stmt, const Stmt &body, bool after) {
  if (!body.defined()) {
    LOG(FATAL) << "body not defined!";
  }

  if (stmt.defined()) {
    if (!after) {
      stmt = Block::make(body, stmt);
    } else {
      stmt = Block::make(stmt, body);
    }
  } else {
    stmt = body;
  }

  return stmt;
}

/// Gen BufferId with computation info
/// \param com_info computation info
/// \return
Buffer GenBufferId(const StmtStoreInfo &com_info) {
  auto new_buf = BufferNode::make(com_info->data_, com_info->dtype_, com_info->shape_, com_info->strides_,
                                  CanonicalSimplify(com_info->elem_offset_), com_info->name_, com_info->scope_,
                                  com_info->data_alignment_, 1, BufferType::kDefault);

  return new_buf;
}

/// Set offset to buffer base ptr
/// \param buffer Symbolic data buffer
/// \param label buffer memory scope
/// \param offset buffer offset
/// \return get an access pointer to the head of buffer
Expr GetAccessPtr(const Buffer &buffer, const std::string &label, Expr offset) {
  offset = CanonicalSimplify(offset);
  uint64_t mask_long = 0;
  if (label.find('r') != std::string::npos) {
    mask_long = mask_long | 1u;
  }
  if (label.find('w') != std::string::npos) {
    mask_long = mask_long | 2u;
  }
  return buffer.access_ptr(static_cast<int>(mask_long), Handle(), 1, offset);
}

/// Get all mask list
/// \param dtype type of data
/// \return mask list
Array<Expr> GetAllMask(const Type dtype) {
  Array<Expr> all_mask;
  auto vec_mask_dtype = UInt(64);
  if (dtype.bits() == 32) {
    all_mask.push_back(make_const(vec_mask_dtype, 0));
    all_mask.push_back(make_const(vec_mask_dtype, ULLONG_MAX));
  } else {
    all_mask.push_back(make_const(vec_mask_dtype, ULLONG_MAX));
    all_mask.push_back(make_const(vec_mask_dtype, ULLONG_MAX));
  }
  return all_mask;
}

/// Emit set_vector_mask intrin with given vec_mask
/// \param stmt
/// \param dtype
/// \param vec_mask
/// \return
Stmt EmitSetVecMaskIntrin(Stmt stmt, const Type &dtype, Array<Expr> vec_mask) {
  if (vec_mask.empty()) {
    vec_mask = GetAllMask(dtype);
  }

  auto body = Evaluate::make(Call::make(dtype, INTRIN_NAME_SET_VEC_MASK, vec_mask, Call::Extern));
  stmt = InsertBody(stmt, body);

  return stmt;
}

/// Emit basic intrin with given args and intrin_name
/// \param stmt
/// \param type
/// \param args
/// \param intrin_name
/// \return
Stmt EmitCceIntrinTemplate(Stmt stmt, const Type &type, const Array<Expr> &args, const std::string &intrin_name) {
  auto evaluate = Evaluate::make(Call::make(type, intrin_name, args, Call::Extern));
  stmt = InsertBody(stmt, evaluate);

  return stmt;
}

/// Get Buffer of store and load
/// \param s stmt to be processed
/// \param dst_buffer_id_list stores Buffer list to be returned
/// \param src_buffer_id_list loads Buffer list to be returned
void GetBufferIdFromStmt(const Stmt stmt, Array<Buffer> &dst_buffer_id_list, Array<Buffer> &src_buffer_id_list) {
  Array<NodeRef> store, loads;
  GetStoreAndLoads(stmt, store, loads);
  StmtInfo if_info, for_info;
  GetIfForInfo(stmt, if_info, for_info);
  auto dst_info_list = GetComputationInfo(store, for_info);
  auto src_info_list = GetComputationInfo(loads, for_info);
  std::transform(dst_info_list.begin(), dst_info_list.end(), std::back_inserter(dst_buffer_id_list.CopyOnWrite()->data),
                 GenBufferId);
  std::transform(src_info_list.begin(), src_info_list.end(), std::back_inserter(src_buffer_id_list.CopyOnWrite()->data),
                 GenBufferId);
}

/// Set mask for argmax
/// \param len
/// \return
std::pair<uint64_t, uint64_t> ArgmaxInsnBuilder::SetMaskArgMax(int len) const {
#define WIDTH 32
  auto len_l = static_cast<int64_t>(std::min(len, WIDTH));
  auto len_h = static_cast<int64_t>(std::max(len - WIDTH, 0));
#undef WIDTH
  auto mask1 = 0ull;
  auto mask2 = 0ull;
  for (uint64_t i = 0; i < static_cast<uint64_t>(len_h); i++) {
    mask1 |= 1ull << (i * 2);
  }
  for (uint64_t i = 0; i < static_cast<uint64_t>(len_l); i++) {
    mask2 |= 1ull << (i * 2);
  }
  return std::make_pair(mask1, mask2);
}

/// Generate first part of argmax intrins
/// \param remain_len
/// \return
Stmt ArgmaxInsnBuilder::GenArgmaxLayer1(Expr &remain_len) {
  const Type idtype = UInt(16);
  auto vec_mask_dtype = UInt(64);

  Buffer cmp =
    BufferNode::make(Var("cmp", Handle()), idtype, {Expr(1)}, {}, Expr(), "cmp", SCOPE_REG, 0, 0, BufferType::kDefault);
  cnt_ =
    BufferNode::make(Var("cnt", Handle()), idtype, {Expr(1)}, {}, Expr(), "cnt", SCOPE_REG, 0, 0, BufferType::kDefault);
  cmp0_ = Load::make(idtype, cmp->data, 0, const_true());
  cnt0_ = Load::make(idtype, cnt_->data, 0, const_true());

  Buffer dst_buffer_id = GenBufferId(dst_info_);
  Expr dst_for_r = GetAccessPtr(dst_buffer_id, "r", dst_info_->index_);

  auto _GenIf1 = [this, idtype, dst_for_r]() {
    Stmt body = Store::make(cnt_->data, make_const(idtype, 0), 0, const_true());
    body = InsertBody(
      body, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                      {GetAccessPtr(t_buffer_, "w", k_res_offset_), dst_for_r}, Call::Extern)));

    body = InsertBody(body, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                                      {GetAccessPtr(t_buffer_, "w", k_cnt_offset_),
                                                       Call::make(cnt0_.type(), "reg", {cnt0_}, Call::Extern)},
                                                      Call::Extern)));
    body = AttrStmt::make(GetCceAxis(), "coproc_scope", 1, body);
    return IfThenElse::make(cmp0_ == init_value_, body);
  };

  Stmt body1 = Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                         {Call::make(idtype, "reg", {cmp0_}, Call::Extern), dst_for_r}, Call::Extern));
  body1 = InsertBody(body1, _GenIf1());
  // copy previous result
  body1 = InsertBody(
    body1, Evaluate::make(Call::make(
             dst_info_->dtype_, INTRIN_NAME_REG_MOV,
             {GetAccessPtr(tmp_buffer_, "w", Expr(0)), GetAccessPtr(t_buffer_, "r", k_res_offset_)}, Call::Extern)));
  body1 = InsertBody(body1, Evaluate::make(Call::make(dst_info_->dtype_, INTRIN_NAME_REG_MOV,
                                                      {GetAccessPtr(tmp_buffer_, "w", Expr(1)),
                                                       GetAccessPtr(t_buffer_, "r", k_res_offset_ + 1)},
                                                      Call::Extern)));
  remain_len += 1;

  // remain_len
  uint64_t mask1, mask2;
  std::tie(mask1, mask2) = SetMaskArgMax(GetInt32Const(remain_len));
  body1 = InsertBody(body1, Evaluate::make(Call::make(
                              dst_info_->dtype_, INTRIN_NAME_SET_VEC_MASK,
                              {make_const(vec_mask_dtype, mask1), make_const(vec_mask_dtype, mask2)}, Call::Extern)));
  body1 = InsertBody(body1, Evaluate::make(Call::make(dst_info_->dtype_, intrin_name_,
                                                      {GetAccessPtr(tmp_buffer_, "rw", remain_len * 2),
                                                       GetAccessPtr(tmp_buffer_, "r", Expr(0)), 1, 1, 1, 0},
                                                      Call::Extern)));

  body1 = InsertBody(body1, GenArgmaxLayer2(remain_len));
  body1 = AttrStmt::make(cnt_->data, STORAGE_SCOPE, Expr(SCOPE_REG),
                         Allocate::make(cnt_->data, cnt_->dtype, {Expr(1)}, const_true(), body1));
  body1 = AttrStmt::make(cmp->data, STORAGE_SCOPE, Expr(SCOPE_REG),
                         Allocate::make(cmp->data, cmp->dtype, {Expr(1)}, const_true(), body1));
  return body1;
}

/// Generate second part of argmax intrins
/// \param remain_len
/// \return
Stmt ArgmaxInsnBuilder::GenArgmaxLayer2(Expr &remain_len) {
  const Type idtype = UInt(16);
  const int k_intrin_unit = 128;

  Buffer reg2 = BufferNode::make(Var("reg2", Handle()), idtype, {Expr(1)}, {}, Expr(), "reg2", SCOPE_REG, 0, 0,
                                 BufferType::kDefault);
  Buffer ires = BufferNode::make(Var("ires", Handle()), idtype, {Expr(1)}, {}, Expr(), "ires", SCOPE_REG, 0, 0,
                                 BufferType::kDefault);
  reg20_ = Load::make(idtype, reg2->data, Expr(0), const_true());
  ires0_ = Load::make(idtype, ires->data, Expr(0), const_true());

  Stmt body2 = Evaluate::make(Call::make(
    idtype, INTRIN_NAME_REG_MOV,
    {Call::make(reg20_.type(), "reg", {reg20_}, Call::Extern), GetAccessPtr(tmp_buffer_, "r"), remain_len * 2 + 1},
    Call::Extern));
  body2 = InsertBody(body2, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                                      {Call::make(ires0_.type(), "reg", {ires0_}, Call::Extern),
                                                       GetAccessPtr(tmp_buffer_, "r"), reg20_ + make_const(Int(32), 1)},
                                                      Call::Extern)));
  // get count
  body2 = InsertBody(body2, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                                      {Call::make(idtype, "reg", {cnt0_}, Call::Extern),
                                                       GetAccessPtr(t_buffer_, "r"), k_cnt_offset_},
                                                      Call::Extern)));

  // check the first loop or else
  Stmt if_block =
    Store::make(ires->data, ires0_ + (reg20_ - make_const(idtype, 2)) * make_const(idtype, k_intrin_unit / 2) + cnt0_,
                Expr(0), const_true());
  if_block = AttrStmt::make(GetCceAxis(), "coproc_scope", Expr(1), if_block);
  body2 = InsertBody(body2, IfThenElse::make(reg20_ != Expr(0x0), if_block));

  body2 = InsertBody(body2, GenArgmaxLayer3(remain_len));
  body2 = AttrStmt::make(ires->data, STORAGE_SCOPE, Expr(SCOPE_REG),
                         Allocate::make(ires->data, ires->dtype, {Expr(1)}, const_true(), body2));
  body2 = AttrStmt::make(reg2->data, STORAGE_SCOPE, Expr(SCOPE_REG),
                         Allocate::make(reg2->data, reg2->dtype, {Expr(1)}, const_true(), body2));
  return body2;
}

/// Generate third part of argmax intrins
/// \param remain_len
/// \return
Stmt ArgmaxInsnBuilder::GenArgmaxLayer3(Expr &remain_len) {
  const Type idtype = UInt(16);
  int buf_len = GetInt32Const(GetItem(src_info_->shape_, -1));
  Buffer dst_buffer_id = GenBufferId(dst_info_);
  Expr dst_for_w = GetAccessPtr(dst_buffer_id, "w", dst_info_->index_);

  Stmt body3 = Store::make(cnt_->data, cnt0_ + make_const(idtype, buf_len), Expr(0), const_true());
  // store count
  body3 = InsertBody(body3, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                                      {GetAccessPtr(t_buffer_, "w", k_cnt_offset_),
                                                       Call::make(idtype, "reg", {cnt0_}, Call::Extern)},
                                                      Call::Extern)));
  // store the result of this loop
  body3 = InsertBody(
    body3, Evaluate::make(Call::make(
             dst_info_->dtype_, INTRIN_NAME_REG_MOV,
             {GetAccessPtr(t_buffer_, "w", k_res_offset_), GetAccessPtr(tmp_buffer_, "r", reg20_)}, Call::Extern)));

  body3 = InsertBody(body3, Evaluate::make(Call::make(idtype, INTRIN_NAME_REG_MOV,
                                                      {GetAccessPtr(t_buffer_, "w", k_res_offset_ + 1),
                                                       Call::make(idtype, "reg", {ires0_}, Call::Extern)},
                                                      Call::Extern)));
  // return result
  body3 = InsertBody(body3, Evaluate::make(Call::make(
                              idtype, INTRIN_NAME_REG_MOV,
                              {dst_for_w, Call::make(ires0_.type(), "reg", {ires0_}, Call::Extern)}, Call::Extern)));
  return AttrStmt::make(GetCceAxis(), "coproc_scope", Expr(1), body3);
}

/// Emit argmax/argmin intrinsic
/// Due to the index of max/min value is required, the scalar DMA intrinsic must be used to get the index
/// \return
Array<Stmt> ArgmaxInsnBuilder::EmitIntrin() {
  CHECK_EQ(dtype_, Float(16)) << "reduce_last_axis only supports float16 while dtype is " << dtype_;

  std::string cmd;
  if (intrin_name_ == "argmax") {
    cmd = "max";
  } else if (intrin_name_ == "argmin") {
    cmd = "min";
  } else {
    LOG(FATAL) << "op " << intrin_name_ << " is not supported yet.";
  }

  intrin_name_ = std::string("vc") + cmd;

  Expr remain_len = Expr(0);

  Stmt stmt;
  if (body_arg_info_.defined()) {
    remain_len = body_arg_info_->repeat_;
    CHECK(body_arg_info_->body_num_ == 1) << "bodyNum should be 1.";
    bool has_set_mask = false;
    dst_info_.GetNode()->insn_offset_ = Expr(2);
    if (!IsSame(body_arg_info_->vec_mask_, GetAllMask(dst_info_->dtype_))) {
      has_set_mask = true;
      stmt = EmitSetVecMaskIntrin(stmt, dst_info_->dtype_, body_arg_info_->vec_mask_);
    }
    stmt = InsertBody(stmt, EmitExpandedIntrin(body_arg_info_, false));
    if (has_set_mask) {
      stmt = EmitSetVecMaskIntrin(stmt, dst_info_->dtype_);
    }
  }
  if (tail_arg_info_.defined()) {
    dst_info_.GetNode()->insn_offset_ = Expr(remain_len * 2 + 2);
    src_info_.GetNode()->insn_offset_ = src_info_->insn_offset_ + tail_arg_info_->src_head_list_[0];
    bool has_set_mask = false;
    if (!IsSame(tail_arg_info_->vec_mask_, GetAllMask(dst_info_->dtype_))) {
      has_set_mask = true;
      stmt = EmitSetVecMaskIntrin(stmt, dst_info_->dtype_, tail_arg_info_->vec_mask_);
    }
    stmt = InsertBody(stmt, EmitExpandedIntrin(tail_arg_info_, false));
    remain_len += 1;
    if (has_set_mask) {
      stmt = EmitSetVecMaskIntrin(stmt, dst_info_->dtype_);
    }
  }

  stmt = InsertBody(stmt, GenArgmaxLayer1(remain_len));
  return {stmt};
}

/// Create tmp buffer and generate argmax intrins
/// \param if_info
/// \param for_info
/// \param arg_info
/// \param dst_info
/// \param src_info
/// \param intrin_name
/// \param init
/// \return
Stmt EmitCceArgmaxIntrinHub(StmtInfo if_info, StmtInfo for_info, const ArgInfo &arg_info, const StmtStoreInfo &dst_info,
                            const StmtStoreInfo &src_info, const std::string &intrin_name, const Expr &init) {
  int block_size = GetUbBlkSize(dst_info->dtype_);
  int vec_max_len = GetVecMaxLen(dst_info->dtype_);
  CHECK_NE(vec_max_len, 0);
  CHECK_NE(block_size, 0);
  int buf_size = (GetInt32Const(GetItem(src_info->shape_, -1)) + vec_max_len - 1) / vec_max_len * 2;
  int align_buf_size = (buf_size + block_size - 1) / block_size * block_size + block_size;

  Buffer tmp_buffer = BufferNode::make(Var("tmp_buf", Handle()), dst_info->dtype_, {Expr(align_buf_size)}, {}, Expr(),
                                       "tmp_buf", SCOPE_UBUF, 0, 0, BufferType::kDefault);
  Buffer t_buffer = BufferNode::make(Var("t_buf", Handle()), dst_info->dtype_, {Expr(16)}, {}, Expr(), "t_buf",
                                     SCOPE_UBUF, 0, 0, BufferType::kDefault);
  ArgmaxInsnBuilder builder = ArgmaxInsnBuilder(dst_info, src_info, arg_info, intrin_name, tmp_buffer, t_buffer, init);
  Stmt stmt = builder.EmitIntrin()[0];

  if (for_info.ops_.empty()) {
    VarExpr fargmax_i("fargmax_i");
    stmt = For::make(fargmax_i, Expr(0), Expr(1), ForType::Unrolled, DeviceAPI::None, stmt);
  } else {
    stmt = GenIfAndFor(stmt, if_info, for_info);
  }

  stmt = AttrStmt::make(t_buffer->data, STORAGE_SCOPE, Expr(SCOPE_UBUF),
                        Allocate::make(t_buffer->data, t_buffer->dtype, {Expr(16)}, const_true(), stmt));
  stmt =
    AttrStmt::make(tmp_buffer->data, STORAGE_SCOPE, Expr(SCOPE_UBUF),
                   Allocate::make(tmp_buffer->data, tmp_buffer->dtype, {Expr(align_buf_size)}, const_true(), stmt));
  return stmt;
}

/// Call special api to generate int162int32 cast intrin
/// \param src
/// \param dst_info
/// \return
Stmt EmitFargmaxCast(const Array<Buffer> &src, const StmtStoreInfo &dst_info) {
  Type idtype = Int(32);
  auto index = dst_info->index_;
  auto dst = GenBufferId(dst_info);

  return Evaluate::make(
    Call::make(idtype, "argmax_cast", {GetAccessPtr(dst, "w", index), GetAccessPtr(src[0], "r", index)}, Call::Extern));
}

/// Emit dropout intrins
/// \param dst_info_list
/// \param src_info_list
/// \param mask
/// \param arg_info
/// \param if_info
/// \param for_info
/// \return
Stmt EmitDropout(const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list, const StmtStoreInfo &mask,
                 const ArgInfo &arg_info, const StmtInfo &if_info, const StmtInfo &for_info) {
  auto dtype = src_info_list[0]->dtype_;
  SingleVecInsnBuilder single_vec_builder = SingleVecInsnBuilder(dst_info_list[0], src_info_list[0], arg_info, "vsel");
  auto insn_list = single_vec_builder.EmitIntrin();

  Stmt rst;
  Var zero_buf = Var("zero_for_vsel", dtype);
  Array<Expr> zero_shape = {make_const(Int(32), GetVecMaxLen(dtype))};
  Buffer zero = BufferNode::make(zero_buf, dtype, zero_shape, Array<Expr>(), Expr(), "zero_for_vsel", SCOPE_UBUF, 0, 0,
                                 BufferType::kDefault);

  Array<Expr> args = {GetAccessPtr(zero, "w"), make_const(dtype, 0), 1, 1, 1, 8, 8};
  rst = EmitCceIntrinTemplate(rst, dtype, args, "vector_dup");

  auto msk_buf = GenBufferId(mask);

  for (size_t i = 0; i != insn_list.size(); ++i) {
    auto new_stmt = DropoutCallBuilder(zero, msk_buf).Mutate(insn_list[i]);
    insn_list.Set(i, new_stmt);
  }

  rst = FoldInsnWithForInfo(insn_list, if_info, for_info, rst);
  rst = Allocate::make(zero_buf, dtype, zero_shape, const_true(), rst);
  rst = AttrStmt::make(zero_buf, STORAGE_SCOPE, Expr(SCOPE_UBUF), rst);
  return rst;
}

/// Generate bit move intrin
/// \param eq
/// \param mask_var
/// \param idx
/// \param step
/// \param move_right
/// \return
Stmt MakeIfMask(const Expr &eq, const Var &mask_var, const Expr &idx, const Expr &step, bool move_right = true) {
  std::string move_call = move_right ? "bit_move_right" : "bit_move_left";
  auto vec_mask_dtype = UInt(64);
  return IfThenElse::make(
    eq, Store::make(mask_var, make_const(vec_mask_dtype, 0), idx, Expr(1)),
    Store::make(
      mask_var,
      Call::make(vec_mask_dtype, move_call, {Load::make(vec_mask_dtype, mask_var, idx, Expr(1)), step}, Call::Extern),
      idx, Expr(1)));
}

/// Emit mutable mask intrins
/// \param insn
/// \param params
/// \return
Stmt EmitMutableMaskGen(const Stmt &insn, const MutableMaskParams &params) {
  bool is_fp32 = params.is_fp32_;
  bool lower = params.lower_;
  Var mask_var = params.mask_var_;
  Expr loop_var = params.loop_var_;
  int loop_extent = GetInt32Const(params.loop_extent_);
  Expr half_mask_len = is_fp32 ? Expr(0) : Expr(64);
  Expr full_mask_len = is_fp32 ? Expr(64) : Expr(128);
  Expr div_var = loop_var / full_mask_len * full_mask_len;
  auto vec_mask_dtype = UInt(64);

  auto stmt = InsertBody(
    insn, Store::make(mask_var, is_fp32 ? make_const(vec_mask_dtype, 0) : vec_mask_dtype.max(), Expr(0), Expr(1)));
  stmt = InsertBody(stmt, Store::make(mask_var, vec_mask_dtype.max(), Expr(1), Expr(1)));
  if (!lower) {
    stmt = InsertBody(
      stmt, Store::make(mask_var, is_fp32 ? make_const(vec_mask_dtype, 0) : vec_mask_dtype.max(), Expr(2), Expr(1)));
    stmt = InsertBody(stmt, Store::make(mask_var, vec_mask_dtype.max(), Expr(3), Expr(1)));
  }

  auto if_case = Evaluate::make(0);
  if (lower) {
    if (is_fp32) {
      if_case = MakeIfMask(EQ::make(loop_var - div_var, full_mask_len - 1), mask_var, 1, loop_var - div_var + 1, false);
    } else {
      if (loop_extent >= GetIntConst(half_mask_len)) {
        if_case = Block::make(Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(1), Expr(1)),
                              MakeIfMask(EQ::make(loop_var - div_var, full_mask_len - 1), mask_var, 0,
                                         loop_var - div_var + 1 - half_mask_len, false));
      }
      if_case = IfThenElse::make(
        LT::make(loop_var, div_var + half_mask_len),
        MakeIfMask(EQ::make(loop_var - div_var, half_mask_len - 1), mask_var, 1, loop_var - div_var + 1, false),
        if_case);
    }
  } else {
    div_var = Simplify((loop_var - full_mask_len) / full_mask_len * full_mask_len);
    if (is_fp32) {
      if (loop_extent >= GetIntConst(full_mask_len)) {
        if_case = IfThenElse::make(
          LT::make(loop_var, Simplify(full_mask_len + full_mask_len + div_var)),
          MakeIfMask(EQ::make(loop_var % full_mask_len, 0), mask_var, 3, full_mask_len - loop_var % full_mask_len),
          if_case);
      }
      if_case = IfThenElse::make(LT::make(loop_var, full_mask_len),
                                 Block::make({MakeIfMask(EQ::make(loop_var, 0), mask_var, 1, full_mask_len - loop_var),
                                              Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(3), Expr(1))}),
                                 if_case);
    } else {
      if (loop_extent >= GetIntConst(full_mask_len + half_mask_len)) {
        if_case = IfThenElse::make(LT::make(loop_var, Simplify(full_mask_len + full_mask_len + div_var)),
                                   MakeIfMask(EQ::make(loop_var % full_mask_len, half_mask_len), mask_var, 2,
                                              full_mask_len - loop_var % full_mask_len));
      }
      if (loop_extent >= GetIntConst(full_mask_len)) {
        if_case = IfThenElse::make(LT::make(loop_var, Simplify(full_mask_len + half_mask_len + div_var)),
                                   Block::make(Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(2), Expr(1)),
                                               MakeIfMask(EQ::make(loop_var % full_mask_len, 0), mask_var, 3,
                                                          half_mask_len - loop_var % full_mask_len)),
                                   if_case);
      }
      if (loop_extent >= GetIntConst(half_mask_len)) {
        if_case = IfThenElse::make(
          LT::make(loop_var, full_mask_len),
          Block::make({MakeIfMask(EQ::make(loop_var, half_mask_len), mask_var, 0, full_mask_len - loop_var),
                       Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(2), Expr(1)),
                       Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(3), Expr(1))}),
          if_case);
      }
      if_case = IfThenElse::make(LT::make(loop_var, half_mask_len),
                                 Block::make({Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(0), Expr(1)),
                                              MakeIfMask(EQ::make(loop_var, 0), mask_var, 1, half_mask_len - loop_var),
                                              Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(2), Expr(1)),
                                              Store::make(mask_var, make_const(vec_mask_dtype, 0), Expr(3), Expr(1))}),
                                 if_case);
    }
  }
  stmt = InsertBody(stmt, if_case);

  return stmt;
}

/// Emit mutable mask in different data type and case
/// \param insn
/// \param dst_info_list
/// \param src_info_list
/// \param params
/// \return
Stmt EmitMutableMaskVec(Stmt insn, const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                        const MutableMaskParams &params) {
  bool is_fp32 = params.is_fp32_;
  bool lower = params.lower_;
  Var mask_var = params.mask_var_;
  Expr loop_var = params.loop_var_;
  Expr loop_extent = params.loop_extent_;
  Stmt broadcast = params.broadcast_;
  Buffer const_buffer = params.const_buffer_;
  Expr full_mask_len = is_fp32 ? Expr(64) : Expr(128);
  Expr div_var = lower ? loop_var / full_mask_len * full_mask_len
                       : Simplify((loop_var - full_mask_len) / full_mask_len * full_mask_len);
  auto dst_info = GetItem(dst_info_list, 0);
  auto src_info0 = GetItem(src_info_list, 0);
  Type dtype = dst_info->dtype_;
  auto vec_mask_dtype = UInt(64);

  Expr src = GetAccessPtr(GenBufferId(src_info0), "r", 0);
  if (broadcast.defined()) {
    src = GetAccessPtr(const_buffer, "r", 0);
  }

  // head
  Expr start_addr = lower ? div_var : Expr(0);
  auto head_insn = EmitSetVecMaskIntrin(
    Stmt(), dtype,
    {Load::make(vec_mask_dtype, mask_var, Expr(0), Expr(1)), Load::make(vec_mask_dtype, mask_var, Expr(1), Expr(1))});
  head_insn = EmitCceIntrinTemplate(
    head_insn, dtype, {GetAccessPtr(GenBufferId(dst_info), "w", start_addr), src, make_const(dtype, 0), 1, 1, 1, 0, 0},
    "vadds");
  head_insn = EmitSetVecMaskIntrin(head_insn, dtype);
  Expr head_cond =
    is_fp32 ? NE::make(Load::make(vec_mask_dtype, mask_var, Expr(1), Expr(1)), make_const(vec_mask_dtype, 0))
            : Or::make(NE::make(Load::make(vec_mask_dtype, mask_var, Expr(0), Expr(1)), make_const(vec_mask_dtype, 0)),
                       NE::make(Load::make(vec_mask_dtype, mask_var, Expr(1), Expr(1)), make_const(vec_mask_dtype, 0)));
  head_insn = IfThenElse::make(head_cond, head_insn);
  insn = InsertBody(insn, head_insn);

  Stmt body_insn;
  Expr body_repeat = Expr(0);
  if (GetIntConst(loop_extent) > GetIntConst(full_mask_len * 2)) {
    // body
    Expr src_stride_m1 = Expr(8);
    if (broadcast.defined()) {
      src_stride_m1 = Expr(0);
    }

    body_insn = EmitSetVecMaskIntrin(Stmt(), dtype);
    Expr body_cond;
    start_addr += full_mask_len;
    if (lower) {
      body_repeat = Simplify((loop_extent - full_mask_len - div_var) / full_mask_len);
      body_cond = LT::make(loop_var, (loop_extent - 1) / full_mask_len * full_mask_len - full_mask_len);
    } else {
      body_repeat = (loop_var - full_mask_len) / full_mask_len;
      body_cond = GT::make((loop_var - full_mask_len) / full_mask_len, 0);
    }

    body_insn = EmitCceIntrinTemplate(body_insn, dtype,
                                      {GetAccessPtr(GenBufferId(dst_info), "w", start_addr), src, make_const(dtype, 0),
                                       body_repeat, 1, 1, 8, src_stride_m1},
                                      "vadds");
    body_insn = IfThenElse::make(body_cond, body_insn);
  }

  // tail
  Stmt tail_insn;
  Expr tail_cond;
  auto mask = GetAllMask(dtype);
  Expr dst_offset;
  if (lower) {
    int full_len = GetInt32Const(full_mask_len);
    if (GetIntConst(loop_extent) > full_len) {
      if ((GetIntConst(loop_extent) - full_len) % full_len != 0) {
        mask = GetVecMask((GetInt32Const(loop_extent) - full_len) % full_len, 1, dtype);
      }
      dst_offset = Simplify((loop_extent - 1) / full_mask_len * full_mask_len);
      tail_cond = (LT::make(loop_var, (loop_extent - 1) / full_mask_len * full_mask_len));
    }
  } else {
    mask = {Load::make(vec_mask_dtype, mask_var, Expr(2), Expr(1)),
            Load::make(vec_mask_dtype, mask_var, Expr(3), Expr(1))};
    dst_offset = full_mask_len * body_repeat + full_mask_len;
    tail_cond =
      is_fp32
        ? NE::make(Load::make(vec_mask_dtype, mask_var, Expr(3), Expr(1)), make_const(vec_mask_dtype, 0))
        : Or::make(NE::make(Load::make(vec_mask_dtype, mask_var, Expr(2), Expr(1)), make_const(vec_mask_dtype, 0)),
                   NE::make(Load::make(vec_mask_dtype, mask_var, Expr(3), Expr(1)), make_const(vec_mask_dtype, 0)));
  }
  tail_insn = EmitSetVecMaskIntrin(Stmt(), dtype, mask);
  tail_insn = EmitCceIntrinTemplate(
    tail_insn, dtype, {GetAccessPtr(GenBufferId(dst_info), "w", dst_offset), src, make_const(dtype, 0), 1, 1, 1, 0, 0},
    "vadds");
  tail_insn = IfThenElse::make(tail_cond, tail_insn);
  insn = InsertBody(insn, tail_insn);

  // Put body after tail to avoid set_mask elim issue
  if (body_insn.defined()) {
    insn = InsertBody(insn, body_insn);
  }

  return insn;
}

/// Copy for info and used it to fold compute intrins, if use one for info several times, it will break SSA rule
/// \param insn_list
/// \param if_info
/// \param for_info
/// \param result
/// \return
Stmt FoldInsnWithForInfo(const Array<Stmt> &insn_list, const StmtInfo &if_info, const StmtInfo &for_info, Stmt result) {
  for (auto insn : insn_list) {
    auto tmp_for = for_info.Copy();
    StmtInfo tmp_if = StmtInfo();
    tmp_if.ops_ = if_info.ops_;
    tmp_if.vars_ = {};
    for (const auto &var : if_info.vars_) {
      auto it = std::find_if(tmp_for.vars_.begin(), tmp_for.vars_.end(),
                             [var](const VarExpr &v) { return (v->name_hint) == (var->name_hint); });
      if (it != tmp_for.vars_.end()) {
        tmp_if.vars_.push_back(*it);
        for (size_t j = 0; j < tmp_if.ops_.size(); ++j) {
          auto new_op = substitute(var, *it, tmp_if.ops_[j]);
          tmp_if.ops_.Set(j, new_op);
        }
      }
    }

    auto new_insn = insn;
    for (size_t i = 0; i < for_info.vars_.size(); ++i) {
      new_insn = substitute(for_info.vars_[i], tmp_for.vars_[i], new_insn);
    }
    result = InsertBody(result, GenIfAndFor(new_insn, tmp_if, tmp_for));
  }

  return result;
}

TVM_REGISTER_API("cce_util.GetRepeatOffset").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = GetRepeatOffset(args[0], args[1], args[2]);
});

TVM_REGISTER_API("cce_util.GenBufferId").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = GenBufferId(args[0]);
});

TVM_REGISTER_API("cce_util.GetAccessPtr").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = GetAccessPtr(args[0], args[1], args[2]);
});

TVM_REGISTER_API("cce_util.GetAllMask").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = GetAllMask(args[0]);
});

TVM_REGISTER_API("cce_util.EmitSetVecMaskIntrin").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = EmitSetVecMaskIntrin(args[0], args[1], args[2]);
});

TVM_REGISTER_API("cce_util.EmitCceIntrinTemplate").set_body([](const TVMArgs &args, TVMRetValue *ret) {
  *ret = EmitCceIntrinTemplate(args[0], args[1], args[2], args[3]);
});
}  // namespace akg
