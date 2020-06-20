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

#include "insn_builder.h"

namespace akg {
/// Emit dma copy intrin
/// \param params
/// \param pad_mode
/// \param cr_mode
/// \param intrin_name
/// \param is_data_dependent
/// \return
Stmt DmaInsnBuilder::EmitCceCopyIntrin(const Map<std::string, Expr> &params, const std::string &pad_mode,
                                       const std::string &cr_mode, const std::string &intrin_name,
                                       bool is_data_dependent) {
  Stmt stmt;
  Expr dst = params["dst"];
  Expr src = params["src"];
  Expr sid = params["sid"];
  Expr n_burst = params["nBurst"];
  Expr len_burst = params["lenBurst"];
  Expr src_stride = params["srcStride"];
  Expr dst_stride = params["dstStride"];
  auto mode_param_dtype = Int(32);

  if (GetInt32Const(n_burst) < 0 || GetInt32Const(len_burst) < 0 || GetInt32Const(src_stride) < 0 ||
      GetInt32Const(dst_stride) < 0) {
    LOG(FATAL) << "Error: parameter cannot be zero!";
  }

  // the maximum value of intrinsic parameters mentioned in "Davinci ISA User Guide"
  if (GetInt32Const(sid) >= MAX_SID || GetInt32Const(n_burst) >= MAX_NBURST ||
      GetInt32Const(len_burst) >= MAX_LENBURST || GetInt32Const(src_stride) >= MAX_STRIDE ||
      GetInt32Const(dst_stride) >= MAX_STRIDE) {
    LOG(FATAL) << "Error: parameter overflow!";
  }

  Array<Expr> args = {dst, src, Expr(sid), Expr(n_burst), Expr(len_burst), Expr(src_stride), Expr(dst_stride)};
  if (intrin_name == "copy_gm_to_cbuf") {
    if (pad_mode.empty()) {
      LOG(FATAL) << "Error: pad_mode can't be empty";
    }
    auto call = Call::make(mode_param_dtype, "tvm_cce_string_print", {StringImm::make(pad_mode)}, Call::PureIntrinsic);
    args.push_back(call);
  } else if (intrin_name == "copy_matrix_cc_to_ubuf") {
    if (cr_mode.empty()) {
      LOG(FATAL) << "Error: cr_mode can't be empty";
    }
    auto call = Call::make(mode_param_dtype, "tvm_cce_string_print", {StringImm::make(cr_mode)}, Call::PureIntrinsic);
    args.push_back(call);
  }

  auto body = Evaluate::make(Call::make(dst.type(), intrin_name, args, Call::Extern));
  if (is_data_dependent && intrin_name == INTRIN_NAME_COPY_UB_TO_GM) {
    body = AttrStmt::make(GetCceAxis(), "coproc_scope", make_const(mode_param_dtype, 6), body);
  }

  stmt = InsertBody(stmt, body);

  return stmt;
}

/// Generate Comment on DMA Copy
/// \param n_burst
/// \param len_burst
/// \param src_stride
/// \param dst_stride
/// \param block_size
/// \param real_block_size
/// \return
Expr GenerateDmaCopyComment(const Expr &n_burst, const Expr &len_burst, const Expr &src_stride, const Expr &dst_stride,
                            const Expr &block_size, const Expr &real_block_size) {
  Expr burst_comment = Sub::make(Var("nBurst"), n_burst);
  burst_comment = Mul::make(burst_comment, Sub::make(Var("lenBurst"), len_burst));
  burst_comment = Mul::make(burst_comment, Sub::make(Var("srcStride"), src_stride));
  burst_comment = Mul::make(burst_comment, Sub::make(Var("dstStride"), dst_stride));
  burst_comment = Mul::make(burst_comment, Sub::make(Var("blockSize"), block_size));
  burst_comment = Mul::make(burst_comment, Sub::make(Var("realBlockSize"), real_block_size));
  return burst_comment;
}

/// Generate dma copy intrin, if data is not 32b aligned, there will be some helper intrin to avoid data overlay
/// \param args
/// \param src_offset_fixed
/// \param dst_offset_fixed
/// \return
Stmt DmaInsnBuilder::CopyIntrinBody(const Map<std::string, Expr> &args, const Expr &src_offset_fixed,
                                    const Expr &dst_offset_fixed) {
  Stmt body;
  Expr sid = Expr(0);
  if (args.count("sid") != 0) {
    sid = args["sid"];
  }

  int real_burst_size = GetInt32Const(arg_info_["realBurstSize"]);
  auto d_type = dst_info_->dtype_;

  Expr n_burst = args["nBurst"];
  Expr len_burst = args["lenBurst"];
  Expr dst_stride = args["dstStride"];
  Expr src_stride = args["srcStride"];

  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src_buffer_id = GenBufferId(src_info_);
  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset_fixed);
  auto src = GetAccessPtr(src_buffer_id, "r", src_offset_fixed);

  bool is_data_dependent = false;
  CHECK_NE(block_size_, 0);
  if (real_burst_size % block_size_ == 0) {
    is_data_dependent = false;
  } else {
    if ((intrin_name_ == "copy_ubuf_to_ubuf") && (src_info_->data_alignment_ != dst_info_->data_alignment_)) {
      LOG(INFO) << "Src and Dst data alignments of DMA MOV are different.";
    } else {
      is_data_dependent = true;
    }
  }
  Expr dst_index = dst_buffer_id->elem_offset + dst_offset_fixed;
  Expr src_index = src_buffer_id->elem_offset + src_offset_fixed;
  auto original_dma_insn = EmitCceCopyIntrin({{"dst", dst},
                                              {"src", src},
                                              {"sid", sid},
                                              {"nBurst", n_burst},
                                              {"lenBurst", len_burst},
                                              {"srcStride", src_stride},
                                              {"dstStride", dst_stride}},
                                             pad_mode_, cr_mode_, intrin_name_, is_data_dependent);
  Expr burst_comment = GenerateDmaCopyComment(n_burst, len_burst, src_stride, dst_stride, block_size_, real_burst_size);
  bool need_protect = false;
  if (intrin_name_ == INTRIN_NAME_COPY_UB_TO_GM && real_burst_size % block_size_ != 0 && GetIntConst(dst_stride) == 0 &&
      real_burst_size > block_size_ && GetIntConst(n_burst) == 1) {
    // tail alignment
    std::string tmp_buffer_name = "tail_tensor_local_UB";
    // main body
    auto main_stmt = EmitCceCopyIntrin({{"dst", dst},
                                        {"src", src},
                                        {"sid", sid},
                                        {"nBurst", n_burst},
                                        {"lenBurst", len_burst - 1},
                                        {"srcStride", src_stride},
                                        {"dstStride", dst_stride}},
                                       pad_mode_, cr_mode_, intrin_name_, is_data_dependent);
    auto buffer_var = Var(tmp_buffer_name, Handle());
    auto new_buffer = BufferNode::make(buffer_var, d_type, {block_size_}, Array<Expr>(), Expr(), tmp_buffer_name,
                                       src_info_->scope_, -1, 0, BufferType::kDefault);
    auto i_var = VarExpr("tail_index");
    auto tmp_offset =
      GetInt32Const((dst_stride + len_burst - 2) * n_burst * block_size_) + real_burst_size % block_size_;
    auto value = Load::make(d_type, src_buffer_id->data,
                            src_buffer_id->elem_offset + src_offset_fixed + tmp_offset + i_var, Expr(1));
    auto tail_copy = new_buffer.vstore({i_var}, value);
    auto ub_src = GetAccessPtr(new_buffer, "r", Expr(0));
    auto gm_dst = GetAccessPtr(dst_buffer_id, "w", dst_offset_fixed + tmp_offset);

    auto tail_block_copy = For::make(i_var, Expr(0), block_size_, ForType::Serial, DeviceAPI::None, tail_copy);
    // tail store with tail alignment
    auto tail_store = EmitCceCopyIntrin({{"dst", gm_dst},
                                         {"src", ub_src},
                                         {"sid", sid},
                                         {"nBurst", n_burst},
                                         {"lenBurst", 1},
                                         {"srcStride", src_stride},
                                         {"dstStride", dst_stride}},
                                        pad_mode_, cr_mode_, intrin_name_, is_data_dependent);
    body = Block::make(tail_block_copy, tail_store);
    body = Allocate::make(buffer_var, d_type, {block_size_}, UIntImm::make(UInt(1), 1), body);
    body = AttrStmt::make(buffer_var, STORAGE_SCOPE, Expr("local.UB"), body);
    body = Block::make(main_stmt, body);

    CommentManager::GetInstance().AddComment("Overlap_optimize", "head overlap");

    // Insert If condition to perform overlap optimization
    if (enable_cover_protect_) {
      body = IfThenElse::make(const_true(), body, original_dma_insn);
      body = AttrStmt::make(make_zero(Int(32)), "src_var_stride", src_index, body);
      body = AttrStmt::make(make_zero(Int(32)), "dst_var_stride", dst_index, body);
      body = AttrStmt::make(make_zero(Int(32)), "intrin_args", burst_comment, body);
      body = AttrStmt::make(make_zero(Int(32)), "overlap_optimize", Expr("head"), body);
    }
    return body;
  } else if (!is_atomic_add_ && intrin_name_ == INTRIN_NAME_COPY_UB_TO_GM && real_burst_size % block_size_ != 0 &&
             GetIntConst(dst_stride) == 0) {
    std::string tmp_buffer_name = "concat_cover";
    auto tmp_buffer_len = IntImm::make(Int(32), block_size_);
    int tmp_buffer_len2 = GetInt32Const((dst_stride + len_burst) * n_burst * block_size_);

    auto buffer_var = Var(tmp_buffer_name, Handle());
    auto new_buffer = BufferNode::make(buffer_var, d_type, {tmp_buffer_len}, Array<Expr>(), Expr(), tmp_buffer_name,
                                       src_info_->scope_, -1, 0, BufferType::kDefault);

    auto ub_dst = GetAccessPtr(new_buffer, "w", Expr(0));
    auto gm_src = GetAccessPtr(dst_buffer_id, "r", dst_offset_fixed + Expr(tmp_buffer_len2 - block_size_));
    body = EmitCceCopyIntrin({{"dst", ub_dst},
                              {"src", gm_src},
                              {"sid", sid},
                              {"nBurst", n_burst},
                              {"lenBurst", 1},
                              {"srcStride", src_stride},
                              {"dstStride", dst_stride}},
                             pad_mode_, cr_mode_, INTRIN_NAME_COPY_GM_TO_UB, is_data_dependent);
    if ((d_type.is_int() || d_type.is_uint()) && !(d_type.is_int() && d_type.bits() == 32)) {
      auto var = VarExpr("scalar_idx");
      auto indices = src_offset_fixed + (real_burst_size) + var;
      auto value = new_buffer.vload({Expr(real_burst_size) + var}, new_buffer->dtype);
      auto store = Store::make(src_buffer_id->data, value, indices, const_true());
      body = InsertBody(
        body, For::make(var, Expr(0), Expr(block_size_ - real_burst_size), ForType::Serial, DeviceAPI::None, store));
    } else {
      auto mask_list = GenMaskVec(d_type, real_burst_size % block_size_, block_size_);
      body = EmitSetVecMaskIntrin(body, d_type, mask_list);
      auto add_dst = GetAccessPtr(src_buffer_id, "w", src_offset_fixed + Expr(tmp_buffer_len2 - block_size_));
      auto add_src1 = GetAccessPtr(new_buffer, "r", Expr(0));

      if (d_type.is_float()) {
        Array<Expr> insn_args = {add_dst, add_src1, Expr(0), Expr(1), Expr(1), Expr(1), Expr(1), Expr(1)};
        body = Block::make(body, Evaluate::make(Call::make(d_type, "vadds", insn_args, Call::Extern)));
        body = EmitSetVecMaskIntrin(body, d_type);
      } else {
        auto zero_dst = GetAccessPtr(src_buffer_id, "w", src_offset_fixed + Expr(tmp_buffer_len2 - block_size_));
        auto add_src0 = GetAccessPtr(src_buffer_id, "r", src_offset_fixed + Expr(tmp_buffer_len2 - block_size_));

        Array<Expr> insn_args = {zero_dst, Expr(0), Expr(1), Expr(1), Expr(1), Expr(1), Expr(1)};
        body = Block::make(body, Evaluate::make(Call::make(d_type, INTRIN_NAME_VECTOR_DUP, insn_args, Call::Extern)));
        insn_args = {add_dst, add_src0, add_src1, Expr(1), Expr(1), Expr(1), Expr(1), Expr(1), Expr(1), Expr(1)};
        body = Block::make(body, Evaluate::make(Call::make(d_type, "vadd", insn_args, Call::Extern)));
        body = EmitSetVecMaskIntrin(body, d_type);
      }
    }

    body = Allocate::make(buffer_var, d_type, {tmp_buffer_len}, UIntImm::make(UInt(1), 1), body);
    if (!src_info_->scope_.empty()) {
      body = AttrStmt::make(buffer_var, STORAGE_SCOPE, StringImm::make(src_info_->scope_), body);
    }

    // Insert If condition to perform overlap optimization
    if (enable_cover_protect_) {
      need_protect = true;
      body = IfThenElse::make(const_true(), body);
    }
    CommentManager::GetInstance().AddComment("Overlap_optimize", "tail overlap");
  }

  body = InsertBody(body, original_dma_insn);
  // Insert If condition to perform overlap optimization
  if (enable_cover_protect_ && intrin_name_ == INTRIN_NAME_COPY_UB_TO_GM) {
    body = AttrStmt::make(make_zero(Int(32)), "src_var_stride", src_index, body);
    body = AttrStmt::make(make_zero(Int(32)), "dst_var_stride", dst_index, body);
    body = AttrStmt::make(make_zero(Int(32)), "intrin_args", burst_comment, body);
    if (need_protect) {
      body = AttrStmt::make(make_zero(Int(32)), "overlap_optimize", Expr("tail"), body);
    }
  }
  return body;
}

/// If n_burst-size > cce_max_n_burst, then split it into loop as "Davinci ISA User Guide t6.3 (9.10.2)" mentioned
/// \return
Stmt DmaInsnBuilder::CopyIntrinBurstLoop() {
  int n_burst_value = GetInt32Const(arg_info_["nBurst"]);
  int len_burst_value = GetInt32Const(arg_info_["lenBurst"]);
  int src_stride_value = GetInt32Const(arg_info_["srcStride"]);
  int dst_stride_value = GetInt32Const(arg_info_["dstStride"]);
  int n_burst_step_size = MAX_NBURST - 1;

  int src_burst_offset = (dst_stride_value + len_burst_value) * block_size_ * n_burst_step_size;
  int dst_burst_offset = (dst_stride_value + len_burst_value) * block_size_ * n_burst_step_size;
  int n_loop = n_burst_value / n_burst_step_size;

  auto var = VarExpr("burstIdx");
  Expr src_offset_fix = src_info_->insn_offset_ + var * Expr(src_burst_offset);
  Expr dst_offset_fix = dst_info_->insn_offset_ + var * Expr(dst_burst_offset);

  Stmt body = CopyIntrinBody({{"nBurst", n_burst_step_size},
                              {"lenBurst", len_burst_value},
                              {"srcStride", src_stride_value},
                              {"dstStride", dst_stride_value}},
                             src_offset_fix, dst_offset_fix);
  body = For::make(var, Expr(0), Expr(n_loop), ForType::Serial, DeviceAPI::None, body);

  int n_burst_tail = n_burst_value - n_loop * n_burst_step_size;
  if (n_burst_tail > 0) {
    src_offset_fix = src_info_->insn_offset_ + Expr(n_loop * src_burst_offset);
    dst_offset_fix = dst_info_->insn_offset_ + Expr(n_loop * dst_burst_offset);

    body = InsertBody(body, CopyIntrinBody({{"nBurst", Expr(n_burst_tail)},
                                            {"lenBurst", len_burst_value},
                                            {"srcStride", src_stride_value},
                                            {"dstStride", dst_stride_value}},
                                           src_offset_fix, dst_offset_fix));
  }

  return body;
}

/// If stride-size > cce_max_stride, then split it into loop as "Davinci ISA User Guide" mentioned
/// \return
Stmt DmaInsnBuilder::CopyIntrinStrideLoop() {
  int n_burst_value = GetInt32Const(arg_info_["nBurst"]);
  int len_burst_value = GetInt32Const(arg_info_["lenBurst"]);
  int src_stride_value = GetInt32Const(arg_info_["srcStride"]);
  int dst_stride_value = GetInt32Const(arg_info_["dstStride"]);
  int src_stride_offset = (src_stride_value + len_burst_value) * block_size_;
  int dst_stride_offset = (dst_stride_value + len_burst_value) * block_size_;

  auto var = VarExpr("strideIdx");
  Expr src_offset_fix = src_info_->insn_offset_ + var * Expr(src_stride_offset);
  Expr dst_offset_fix = dst_info_->insn_offset_ + var * Expr(dst_stride_offset);

  Stmt body = CopyIntrinBody({{"nBurst", 1}, {"lenBurst", len_burst_value}, {"srcStride", 0}, {"dstStride", 0}},
                             src_offset_fix, dst_offset_fix);
  body = For::make(var, Expr(0), Expr(n_burst_value), ForType::Serial, DeviceAPI::None, body);

  return body;
}

/// Handle stride loop and burst loop of dma copy intrin
/// \return
Stmt DmaInsnBuilder::EmitSingleIntrin() {
  if (is_load2_d_) {
    return EmitIntrinLoad2D();
  }

  Stmt stmt;
  int n_burst_value = GetInt32Const(arg_info_["nBurst"]);
  int src_stride_value = GetInt32Const(arg_info_["srcStride"]);
  int dst_stride_value = GetInt32Const(arg_info_["dstStride"]);
  if (src_stride_value < MAX_STRIDE && dst_stride_value < MAX_STRIDE) {
    if (n_burst_value < MAX_NBURST) {
      stmt = CopyIntrinBody(arg_info_, src_info_->insn_offset_, dst_info_->insn_offset_);
    } else {
      stmt = CopyIntrinBurstLoop();
    }
  } else {
    stmt = CopyIntrinStrideLoop();
  }

  CHECK(stmt.defined()) << "stmt is undefined!";

  return stmt;
}

/// Emit vtranspose intrin
/// \param h
/// \param w
/// \return
Stmt TransposeInsnBuilder::VtransIntrinBody(const Expr &h, const Expr &w) {
  auto dst_offset_fixed = (w * loop_height_ + h) * data_len_per_intrin_;
  auto src_offset_fixed = (h * loop_width_ + w) * data_len_per_intrin_;
  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src_buffer_id = GenBufferId(src_info_);

  CHECK_NE(block_size_, 0);
  if (GetIntConst(loop_height_) > 1) {
    if (GetIntConst(loop_width_) > 1) {
      dst_buffer_id = src_buffer_id;
      src_buffer_id = dst_tmp_buffer_;
    } else {
      dst_buffer_id = dst_tmp_buffer_;
      src_offset_fixed += src_info_->insn_offset_;
    }
  } else if (GetIntConst(loop_width_) > 1) {
    src_buffer_id = dst_tmp_buffer_;
    dst_offset_fixed += dst_info_->insn_offset_;
  } else {
    if (GetIntConst(shape_width_) % block_size_ != 0) {
      dst_buffer_id = dst_tmp_buffer_;
    } else {
      dst_offset_fixed += dst_info_->insn_offset_;
    }
    src_offset_fixed += src_info_->insn_offset_;
  }

  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset_fixed);
  auto src = GetAccessPtr(src_buffer_id, "r", src_offset_fixed);

  return EmitCceIntrinTemplate(Stmt(), dst_info_->dtype_, {dst, src}, "vtranspose");
}

/// Emit copy_ub_to_ub intrin when transpose if needed
/// \param h
/// \param w
/// \param is_pre
/// \param params
/// \return
Stmt TransposeInsnBuilder::UbCopyIntrinBody(const Expr &h, const Expr &w, bool is_pre, const Array<Expr> &params) {
  CHECK_EQ(params.size(), 4);
  auto dst_offset_fixed = dst_info_->insn_offset_;
  auto src_offset_fixed = src_info_->insn_offset_;
  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src_buffer_id = GenBufferId(src_info_);

  if (is_pre) {
    // Only src need to add insn_offset
    dst_offset_fixed = (h * loop_width_ + w) * data_len_per_intrin_;
    src_offset_fixed += Simplify(h * loop_width_ * block_size_ * block_size_ + w * block_size_);
    dst_buffer_id = dst_tmp_buffer_;
  } else {
    // Only dst need to add insn_offset
    dst_offset_fixed += Simplify(w * loop_height_ * block_size_ * block_size_ + h * block_size_);
    src_offset_fixed = (w * loop_height_ + h) * data_len_per_intrin_;
    if (GetIntConst(loop_width_) <= 1) {
      src_buffer_id = dst_tmp_buffer_;
    }
  }

  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset_fixed);
  auto src = GetAccessPtr(src_buffer_id, "r", src_offset_fixed);

  Array<Expr> insn_args = {dst, src, Expr(0), params[0], params[1], params[2], params[3]};

  return EmitCceIntrinTemplate(Stmt(), dst_info_->dtype_, insn_args, "copy_ubuf_to_ubuf");
}

/// Emit transpose intrin with copy_ub_to_up to reorder data
/// \return
Stmt TransposeInsnBuilder::EmitSingleIntrin() {
  Stmt stmt;
  auto var = Var(dst_info_->name_ + "_tmp", Handle());
  CHECK_NE(block_size_, 0);
  dst_tmp_buffer_ = BufferNode::make(var, dst_info_->dtype_, {loop_width_ * loop_height_ * block_size_ * block_size_},
                                     dst_info_->strides_, Expr(0), dst_info_->name_, dst_info_->scope_,
                                     dst_info_->data_alignment_, 1, BufferType::kDefault);

  if (GetIntConst(loop_height_) > 1) {
    if (GetIntConst(loop_width_) > 1) {
      auto w = VarExpr("loop_w_post");
      auto h = VarExpr("loop_h_post");
      stmt = UbCopyIntrinBody(h, w, true, pre_ub_copy_arg_info_);
      stmt = For::make(w, Expr(0), loop_width_, ForType::Serial, DeviceAPI::None, stmt);
      stmt = For::make(h, Expr(0), loop_height_, ForType::Serial, DeviceAPI::None, stmt);
    }

    auto w = VarExpr("loop_w");
    auto h = VarExpr("loop_h");
    Stmt trans_body;
    if (GetIntConst(loop_width_) > 1) {
      trans_body = VtransIntrinBody(h, w);
      trans_body = For::make(w, Expr(0), loop_width_, ForType::Serial, DeviceAPI::None, trans_body);
    } else {
      trans_body = VtransIntrinBody(h, Expr(0));
    }
    trans_body = For::make(h, Expr(0), loop_height_, ForType::Serial, DeviceAPI::None, trans_body);
    stmt = InsertBody(stmt, trans_body);

    w = VarExpr("loop_w_pre");
    h = VarExpr("loop_h_pre");
    Stmt ub_body;
    if (GetIntConst(loop_width_) > 1) {
      ub_body = UbCopyIntrinBody(h, w, false, post_ub_copy_arg_info_);
      ub_body = For::make(w, Expr(0), loop_width_, ForType::Serial, DeviceAPI::None, ub_body);
    } else {
      ub_body = UbCopyIntrinBody(h, Expr(0), false, post_ub_copy_arg_info_);
    }
    ub_body = For::make(h, Expr(0), loop_height_, ForType::Serial, DeviceAPI::None, ub_body);
    stmt = InsertBody(stmt, ub_body);
  } else if (GetIntConst(loop_width_) > 1) {
    auto w = VarExpr("loop_w_pre");
    auto copy_body = UbCopyIntrinBody(Expr(0), w, true, pre_ub_copy_arg_info_);
    copy_body = For::make(w, Expr(0), loop_width_, ForType::Serial, DeviceAPI::None, copy_body);
    w = VarExpr("loop_w");
    auto trans_body = VtransIntrinBody(Expr(0), w);
    trans_body = For::make(w, Expr(0), loop_width_, ForType::Serial, DeviceAPI::None, trans_body);
    stmt = Block::make(copy_body, trans_body);
  } else {
    stmt = VtransIntrinBody(Expr(0), Expr(0));
    if (GetIntConst(shape_width_) % block_size_ != 0) {
      auto copy = UbCopyIntrinBody(Expr(0), Expr(0), false, post_ub_copy_arg_info_);
      stmt = InsertBody(stmt, copy);
    }
  }

  if (GetIntConst(loop_height_) > 1 || GetIntConst(loop_width_) > 1 || GetIntConst(shape_width_) % block_size_ != 0) {
    stmt = Allocate::make(var, dst_info_->dtype_, {loop_width_ * loop_height_ * block_size_ * block_size_},
                          UIntImm::make(UInt(1), 1), stmt);
    if (!dst_info_->scope_.empty()) {
      stmt = AttrStmt::make(var, STORAGE_SCOPE, StringImm::make(dst_info_->scope_), stmt);
    }
  }

  CHECK(stmt.defined()) << "stmt is undefined!";

  return stmt;
}

/// Emit Load2D intrin
/// \return
Stmt DmaInsnBuilder::EmitIntrinLoad2D() {
  Stmt stmt;
  const int repeat = GetInt32Const(arg_info_["repeat"]);
  const int src_stride = GetInt32Const(arg_info_["srcStride"]);
  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;

  if (src_stride < MAX_STRIDE) {
    if (repeat < MAX_REPEAT) {
      stmt = Load2DIntrinBody(repeat, src_stride, src_offset, dst_offset);
    } else {
      stmt = Load2DIntrinRepeatLoop();
    }
  } else {
    stmt = Load2DIntrinStrideLoop();
  }

  CHECK(stmt.defined()) << "stmt is undefined!";

  return stmt;
}

/// Handle Load 2D intrin repeat loop
/// \return
Stmt DmaInsnBuilder::Load2DIntrinRepeatLoop() {
  int step_size_repeat = MAX_REPEAT - 1;
  int src_stride = GetInt32Const(arg_info_["srcStride"]);
  int repeat = GetInt32Const(arg_info_["repeat"]);
  int cube_size = GetScopeBlockSize(dst_info_, src_info_);
  int src_repeat_offset = src_stride * cube_size * step_size_repeat;
  int dst_repeat_offset = 1 * cube_size * step_size_repeat;
  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;

  auto var = VarExpr("repeatStepIdx");
  Expr src_offset_fixed = src_offset + var * Expr(src_repeat_offset);
  Expr dst_offset_fixed = dst_offset + var * Expr(dst_repeat_offset);
  auto body = Load2DIntrinBody(step_size_repeat, src_stride, src_offset_fixed, dst_offset_fixed);

  CHECK_NE(step_size_repeat, 0);
  const int n_repeat_epilogue = repeat % step_size_repeat;
  if (n_repeat_epilogue > 0) {
    src_offset_fixed = src_offset + Expr(repeat / step_size_repeat) * Expr(src_repeat_offset);
    dst_offset_fixed = dst_offset + Expr(repeat / step_size_repeat) * Expr(dst_repeat_offset);
    body = InsertBody(body, Load2DIntrinBody(n_repeat_epilogue, src_stride, src_offset_fixed, dst_offset_fixed));
  }
  return body;
}

/// Handle Load 2D intrin stride loop
/// \return
Stmt DmaInsnBuilder::Load2DIntrinStrideLoop() {
  Stmt stmt;
  int src_stride = GetInt32Const(arg_info_["srcStride"]);
  int repeat = GetInt32Const(arg_info_["repeat"]);
  int cube_size = GetScopeBlockSize(dst_info_, src_info_);
  int src_stride_offset = src_stride * cube_size;
  int dst_stride_offset = cube_size;

  VarExpr var = VarExpr("repeatIdx");
  Expr dst_offset = dst_info_->insn_offset_;
  Expr src_offset = src_info_->insn_offset_;
  Expr src_offset_fixed = Expr(src_offset) + src_stride_offset * var;
  Expr dst_offset_fixed = Expr(dst_offset) + dst_stride_offset * var;
  stmt = Load2DIntrinBody(1, 1, src_offset_fixed, dst_offset_fixed);
  stmt = For::make(var, Expr(0), Expr(repeat), ForType::Serial, DeviceAPI::None, stmt);
  return stmt;
}

/// Generate Load 2D intrin body
/// \param repeat
/// \param src_stride
/// \param src_offset_fixed
/// \param dst_offset_fixed
/// \return
Stmt DmaInsnBuilder::Load2DIntrinBody(int repeat, int src_stride, const Expr &src_offset_fixed,
                                      const Expr &dst_offset_fixed) {
  Expr transpose_call = arg_info_["transposeCall"];
  int base_idx = GetInt32Const(arg_info_["baseIdx"]);
  int sid = GetInt32Const(arg_info_["sid"]);

  auto dst_buffer_id = GenBufferId(dst_info_);
  auto src_buffer_id = GenBufferId(src_info_);
  auto dst = GetAccessPtr(dst_buffer_id, "w", dst_offset_fixed);
  auto src = GetAccessPtr(src_buffer_id, "r", src_offset_fixed);
  Array<Expr> load_args = {dst, src, base_idx, repeat, src_stride, sid};

  Expr is_trans_call = StringImm::make(is_one(transpose_call) ? "true" : "false");
  if (is_one(transpose_call) || (intrin_name_ != "load_gm_to_ca" && intrin_name_ != "load_gm_to_cb")) {
    Expr transpose = Call::make(Int(32), "tvm_cce_string_print", {is_trans_call}, Call::PureIntrinsic);
    load_args.push_back(transpose);
  }

  return EmitCceIntrinTemplate(Stmt(), dst_info_->dtype_, load_args, intrin_name_);
}
}  // namespace akg
