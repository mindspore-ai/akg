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

#ifndef EMIT_INSN_INSN_BUILDER_H_
#define EMIT_INSN_INSN_BUILDER_H_

#include <tvm/ir_pass.h>

#include <string>

#include "ir_pass.h"
#include "insn_info.h"
#include "cce_params.h"
namespace akg {
struct MutableMaskParams {
  Var mask_var_;
  Expr loop_var_;
  Expr loop_extent_;
  bool is_fp32_ = false;
  bool lower_ = false;
  Buffer const_buffer_;
  Stmt broadcast_;
};

class CommentManager {
 public:
  CommentManager(const CommentManager &) = delete;
  CommentManager &operator=(const CommentManager &) = delete;

  static CommentManager &GetInstance() {
    static CommentManager comment_manager;
    return comment_manager;
  }

  void CleanComments() { comments_.clear(); }

  std::string GenComment(int comment_level) {
    std::string comment;
    std::vector<std::string> comment_list;
    switch (comment_level) {
      case 1:
        comment_list = comment_list_level1_;
        break;
      case 2:
        comment_list = comment_list_level2_;
        break;
      case 3:
        comment_list = comment_list_level3_;
        break;
      default:
        LOG(WARNING) << "Unknown comment level " << comment_level;
        break;
    }
    for (const auto &key : comment_list) {
      if (!comments_[key].empty()) {
        comment += "#" + key + " " + comments_[key];
      }
    }
    return comment;
  }

  void AddComment(const std::string &key, const std::string &value) {
    if (!value.empty() &&
        find(comment_list_level3_.begin(), comment_list_level3_.end(), key) != comment_list_level3_.end()) {
      comments_[key] = value;
    }
  }

 private:
  CommentManager()
      : comment_list_level1_({"Bisect_optimize", "Overlap_optimize", "Atomic_add"}),
        comment_list_level2_({"Insn_name", "Insn_type", "Compute_type", "Pattern", "Vadds_replace_copy",
                              "Bisect_optimize", "Overlap_optimize", "Atomic_add"}),
        comment_list_level3_({"Insn_name", "Insn_type", "Compute_type", "Pattern", "Vadds_replace_copy",
                              "Bisect_optimize", "Overlap_optimize", "Atomic_add", "Mask_rate", "Alignment",
                              "Contain_tail"}) {}
  ~CommentManager() = default;

 private:
  std::vector<std::string> comment_list_level1_;
  std::vector<std::string> comment_list_level2_;
  std::vector<std::string> comment_list_level3_;
  std::map<std::string, std::string> comments_;
};

class InsnBuilder {
 public:
  InsnBuilder() = default;
  InsnBuilder(const StmtStoreInfo &dst, const StmtInfoList &srcs, const std::string &intrin_name)
      : dst_info_(dst), src_info_list_(srcs), intrin_name_(intrin_name) {
    CHECK(!intrin_name_.empty());
  }
  virtual ~InsnBuilder() = default;
  virtual Stmt EmitSingleIntrin() { return Stmt(); }
  virtual Array<Stmt> EmitIntrin() { return {}; }

 protected:
  StmtStoreInfo dst_info_;
  StmtInfoList src_info_list_;
  std::string intrin_name_;
};

class DmaInsnBuilder : public InsnBuilder {
 public:
  DmaInsnBuilder(const StmtStoreInfo &dst, const StmtStoreInfo &src, const std::string &intrin_name,
                 const Map<std::string, Expr> &args, bool is_load2_d = false, bool atomic = false,
                 bool enable_cover_protect = false)
      : InsnBuilder(dst, {src}, intrin_name),
        src_info_(src),
        arg_info_(args),
        is_load2_d_(is_load2_d),
        is_atomic_add_(atomic),
        enable_cover_protect_(enable_cover_protect) {
    CHECK(src_info_.defined());
    block_size_ = GetScopeBlockSize(dst_info_, src_info_);

    if (!is_load2_d_ && (arg_info_.count("nBurst") == 0 || arg_info_.count("lenBurst") == 0 ||
                         arg_info_.count("dstStride") == 0 || arg_info_.count("srcStride") == 0)) {
      LOG(FATAL) << "argInfo is incomplete!";
    }

    if (arg_info_.count("padMode") != 0 && arg_info_["padMode"].as<StringImm>()) {
      pad_mode_ = arg_info_["padMode"].as<StringImm>()->value;
    }
    if (arg_info_.count("crMode") != 0 && arg_info_["crMode"].as<StringImm>()) {
      cr_mode_ = arg_info_["crMode"].as<StringImm>()->value;
    }
  }

  ~DmaInsnBuilder() override = default;
  Stmt EmitSingleIntrin() override;
  Array<Stmt> EmitIntrin() override {
    LOG(FATAL) << "DMAInsnBuilder only return one stmt, please call EmitSingleIntrin";
    return {};
  }

 private:
  Stmt EmitIntrinLoad2D();
  Stmt EmitCceCopyIntrin(const Map<std::string, Expr> &params, const std::string &pad_mode, const std::string &cr_mode,
                         const std::string &intrin_name, bool is_data_dependent);
  Stmt CopyIntrinBody(const Map<std::string, Expr> &arg_info, const Expr &src_offset_fixed = Expr(0),
                      const Expr &dst_offset_fixed = Expr(0));
  Stmt CopyIntrinBurstLoop();
  Stmt CopyIntrinStrideLoop();
  Stmt Load2DIntrinRepeatLoop();
  Stmt Load2DIntrinStrideLoop();
  Stmt Load2DIntrinBody(int repeat, int src_stride, const Expr &src_offset_fixed, const Expr &dst_offset_fixed);

  StmtStoreInfo src_info_;
  Map<std::string, Expr> arg_info_;
  bool is_load2_d_;
  bool is_atomic_add_;
  bool enable_cover_protect_;
  int block_size_;
  std::string pad_mode_;
  std::string cr_mode_;
};

class TransposeInsnBuilder : public InsnBuilder {
 public:
  TransposeInsnBuilder(const StmtStoreInfo &dst, const StmtStoreInfo &src, const Array<Expr> &args,
                       const Array<Expr> &pre_ub_param, const Array<Expr> &post_ub_param)
      : InsnBuilder(dst, {src}, "transpose"),
        src_info_(src),
        arg_info_(args),
        pre_ub_copy_arg_info_(pre_ub_param),
        post_ub_copy_arg_info_(post_ub_param) {
    CHECK(src_info_.defined());
    CHECK_EQ(arg_info_.size(), 3);
    loop_width_ = arg_info_[0];
    loop_height_ = arg_info_[1];
    shape_width_ = arg_info_[2];
    data_len_per_intrin_ = 256;
    block_size_ = GetUbBlkSize(dst_info_->dtype_);
  }
  ~TransposeInsnBuilder() override = default;

  Stmt EmitSingleIntrin() final;

 private:
  Stmt VtransIntrinBody(const Expr &h, const Expr &w);
  Stmt UbCopyIntrinBody(const Expr &h, const Expr &w, bool is_pre, const Array<Expr> &params);

  StmtStoreInfo src_info_;
  Array<Expr> arg_info_;
  Array<Expr> pre_ub_copy_arg_info_;
  Array<Expr> post_ub_copy_arg_info_;
  Expr loop_width_;
  Expr loop_height_;
  Expr shape_width_;
  int data_len_per_intrin_;
  int block_size_;
  Buffer dst_tmp_buffer_;
};

class VectorInsnBuilder : public InsnBuilder {
 public:
  VectorInsnBuilder(const StmtStoreInfo &dst, const StmtInfoList &srcs, const ArgInfo &args,
                    const std::string &intrin_name)
      : InsnBuilder(dst, srcs, intrin_name), arg_info_(args) {
    CHECK(dst_info_.defined());
    CHECK(!src_info_list_.empty());
    CHECK(arg_info_.defined());
    body_arg_info_ = arg_info_->body_arg_info_;
    tail_arg_info_ = arg_info_->tail_arg_info_;
    dtype_ = dst_info_->dtype_;
    repeat_step_size_ = MAX_REPEAT - 1;
  }
  ~VectorInsnBuilder() override = default;
  Stmt EmitSingleIntrin() override {
    LOG(FATAL) << "VectorInsnBuilder only return stmt array, please call EmitIntrin";
    return Stmt();
  }

 protected:
  Map<std::string, Expr> GenInsnArgs(const Expr &repeat, const Expr &dst_offset, const Expr &src0_offset,
                                     const Expr &src1_offset = Expr());

  ArgInfo arg_info_;
  VectorArgInfo body_arg_info_;
  VectorArgInfo tail_arg_info_;
  Type dtype_;
  int repeat_step_size_;
};

class SingleVecInsnBuilder : public VectorInsnBuilder {
 public:
  SingleVecInsnBuilder(const StmtStoreInfo &dst, const StmtStoreInfo &src, const ArgInfo &args,
                       const std::string &intrin_name, const Buffer &tmp_buf = Buffer())
      : VectorInsnBuilder(dst, {src}, args, intrin_name), src_info_(src_info_list_[0]), tmp_buffer_(tmp_buf) {
    CHECK(src_info_.defined());
  }
  ~SingleVecInsnBuilder() override = default;

  Array<Stmt> EmitIntrin() final;

 private:
  Stmt EmitExpandedIntrin(const VectorArgInfo &arg_info);
  Stmt EmitIntrinRepeatLoop(const VectorArgInfo &arg_info);
  Stmt EmitIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args);
  Stmt CreateBroadcast(const VectorArgInfo &arg_info, const Var &local_var, Stmt stmt);

  StmtStoreInfo src_info_;
  Buffer tmp_buffer_;
  Buffer broadcast_buffer_;
};

class MultiVecInsnBuilder : public VectorInsnBuilder {
 public:
  MultiVecInsnBuilder(const StmtStoreInfo &dst, const StmtInfoList &srcs, const ArgInfo &args,
                      const std::string &intrin_name)
      : VectorInsnBuilder(dst, srcs, args, intrin_name) {
    block_size_ = GetUbBlkSize(dst_info_->dtype_);
  }
  ~MultiVecInsnBuilder() override = default;

  Array<Stmt> EmitIntrin() final;

 private:
  Stmt EmitExpandedIntrin(const VectorArgInfo &arg_info);
  Stmt BinaryVecIntrinRepeatLoop(const VectorArgInfo &arg_info);
  Stmt MultiVecIntrinRepeatLoop(const VectorArgInfo &arg_info, bool is_binary);
  Stmt BinaryVecIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args);
  Stmt BinaryVecVAIntrinBody(const VectorArgInfo &arg_info, const Map<std::string, Expr> &args);
  Stmt MultiVecIntrinBody(const VectorArgInfo &arg_info, Expr dst_offset, Array<Expr> src_offset_list);

  int block_size_;
};

class ReduceLastAxisInsnBuilder : public VectorInsnBuilder {
 public:
  ReduceLastAxisInsnBuilder(const StmtStoreInfo &dst, const StmtStoreInfo &src, const ArgInfo &args,
                            const std::string &intrin_name, const Buffer &final_dst = Buffer(),
                            const Buffer &local_dst = Buffer())
      : VectorInsnBuilder(dst, {src}, args, intrin_name),
        src_info_(src),
        final_dst_buffer_(final_dst),
        local_dst_buffer_(local_dst),
        reduction_tail_intrin_(false) {
    CHECK(src_info_.defined());
    block_size_ = GetUbBlkSize(dst_info_->dtype_);
  }
  ~ReduceLastAxisInsnBuilder() override = default;

  Array<Stmt> EmitIntrin() override;

 protected:
  Stmt EmitExpandedIntrin(const VectorArgInfo &arg_info, bool is_final_cmd);

  StmtStoreInfo src_info_;

 private:
  Stmt EmitIntrinBody(const VectorArgInfo &arg_info, bool is_final_cmd, const Map<std::string, Expr> &args);
  Stmt EmitIntrinRepeatLoop(const VectorArgInfo &arg_info, bool is_final_cmd);

  Buffer final_dst_buffer_;
  Buffer local_dst_buffer_;
  int block_size_;
  bool reduction_tail_intrin_;
  Expr dst_insn_offset_as_src_;
};

class ArgmaxInsnBuilder : public ReduceLastAxisInsnBuilder {
 public:
  ArgmaxInsnBuilder(const StmtStoreInfo &dst, const StmtStoreInfo &src, const ArgInfo &args,
                    const std::string &intrin_name, const Buffer &tmp_buf, const Buffer &t_buf, const Expr &init_val)
      : ReduceLastAxisInsnBuilder(dst, src, args, intrin_name, Buffer(), tmp_buf),
        tmp_buffer_(tmp_buf),
        t_buffer_(t_buf),
        init_value_(init_val),
        k_cnt_offset_(Expr(1)),
        k_res_offset_(Expr(2)) {}
  ~ArgmaxInsnBuilder() override = default;

  Array<Stmt> EmitIntrin() final;

 private:
  std::pair<uint64_t, uint64_t> SetMaskArgMax(int len) const;
  Stmt GenArgmaxLayer1(Expr &remain_len);
  Stmt GenArgmaxLayer2(Expr &remain_len);
  Stmt GenArgmaxLayer3(Expr &remain_len);

  Buffer tmp_buffer_;
  Buffer t_buffer_;
  Expr init_value_;
  Expr k_cnt_offset_;
  Expr k_res_offset_;
  Expr cmp0_;
  Expr cnt0_;
  Expr reg20_;
  Expr ires0_;
  Buffer cnt_;
};

Expr GetRepeatOffset(const int &block_size, const Expr &repeat, const int &stride_m1 = 0);

Stmt GenIfAndFor(Stmt stmt, StmtInfo &if_info, StmtInfo &for_info, bool need_reverse = true);

Stmt InsertBody(Stmt stmt, const Stmt &body, bool after = true);

Buffer GenBufferId(const StmtStoreInfo &com_info);

Expr GetAccessPtr(const Buffer &buffer, const std::string &label, Expr offset = Expr(0));

Array<Expr> GetAllMask(Type dtype);

Stmt EmitSetVecMaskIntrin(Stmt stmt, const Type &dtype, Array<Expr> vec_mask = Array<Expr>());

Stmt EmitCceIntrinTemplate(Stmt stmt, const Type &type, const Array<Expr> &args, const std::string &intrin_name);

void GetBufferIdFromStmt(Stmt stmt, Array<Buffer> &dst_buffer_id_list, Array<Buffer> &src_buffer_id_list);

Stmt FoldInsnWithForInfo(const Array<Stmt> &insn_list, const StmtInfo &if_info, const StmtInfo &for_info, Stmt result);

Stmt EmitFargmaxCast(const Array<Buffer> &src, const StmtStoreInfo &dst_info);

Stmt EmitIou(const Expr &loop_idx, bool nms_alloc, const int &box_num1, const Buffer &src_0, const Buffer &src_1,
             const Buffer &dst, const Buffer &BufferA, const Buffer &BufferB);

Stmt EmitCor(const Expr &loop_idx, const Expr &thresh_hold, const Buffer &dst, const Buffer &buf_a,
             const Buffer &buf_b);

Stmt EmitProposalSort(const Stmt &store, const Buffer &src, const Buffer &dst, bool topksort);

Stmt EmitDropout(const StmtInfoList &dst_info, const StmtInfoList &src_info_list, const StmtStoreInfo &mask,
                 const ArgInfo &arg_info, const StmtInfo &if_info, const StmtInfo &for_info);

Stmt EmitCceBinaryVectorToReduceLastAxis(const StmtStoreInfo &dst_info, const StmtStoreInfo &src_info,
                                         const StmtInfo &if_info, const StmtInfo &for_info, const ArgInfo &arg_info,
                                         const std::string &intrin_name);

Stmt EmitCceArgmaxIntrinHub(StmtInfo if_info, StmtInfo for_info, const ArgInfo &arg_info, const StmtStoreInfo &dst_info,
                            const StmtStoreInfo &src_info, const std::string &intrin_name, const Expr &init);

Stmt EmitMutableMaskGen(const Stmt &insn, const MutableMaskParams &params);

Stmt EmitMutableMaskVec(Stmt insn, const StmtInfoList &dst_info_list, const StmtInfoList &src_info_list,
                        const MutableMaskParams &params);

Stmt EmitCceBinaryVectorToBisectionReduction(const BisectionInfoWrapper &wrapper, const StmtInfo &if_info,
                                             const std::string &intrin_name);

class DropoutCallBuilder : public IRMutator {
 public:
  DropoutCallBuilder(const Buffer &b, const Buffer &msk) : acc(GetAccessPtr(b, "r")), msk(msk) {}
  ~DropoutCallBuilder() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &e) override {
    if (op->name == "vsel") {
      auto &old = op->args;
      Array<Expr> args_ = {old[0], old[1], acc, old[2], old[3], old[4], 1, old[5], old[6], 0};
      return Call::make(op->type, "vsel", args_, Call::Extern);
    }
    return IRMutator::Mutate_(op, e);
  }

  Stmt Mutate_(const Evaluate *op, const Stmt &stmt) override {
    auto s = IRMutator::Mutate_(op, stmt);
    auto ptr = s.as<Evaluate>();
    CHECK(ptr);
    auto call = ptr->value.as<Call>();
    CHECK(call);
    if (call->name == "vsel") {
      auto acc_call = call->args[0].as<Call>();
      if (acc_call != nullptr && acc_call->name == air::ir::intrinsic::tvm_access_ptr) {
        auto set_cmp = EmitCceIntrinTemplate(Stmt(), msk->dtype,
                                             {GetAccessPtr(msk, "r", truncdiv(acc_call->args[2], 8))}, "set_cmpmask");
        return Block::make(set_cmp, s);
      }
    }

    return s;
  }

  Expr acc;
  const Buffer &msk;
};

}  // namespace akg

#endif  // EMIT_INSN_INSN_BUILDER_H_
