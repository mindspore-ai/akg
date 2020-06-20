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

#ifndef PASS_APPEND_STRIDE_MSG_H_
#define PASS_APPEND_STRIDE_MSG_H_

#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>
#include <runtime/thread_storage_scope.h>
#include "tvm.h"
#include "arch.h"

namespace akg {
namespace ir {
/**
 * Visit Call stmt Expand their tvm_access_prt's stride message
 * ```
 * Before:
 *     tvm_access_prt(Type, bufferVar, offset, extent, rw)
 * After:
 *     tvm_access_prt(Type, bufferVar, offset, extent, rw,
 *                    RepeatTime, RepeatStride, BlockNumber, BlockStride, BlockSize)
 * ```
 */
class AppendStrideMessage final : public IRMutator {
 public:
  AppendStrideMessage() = default;
  ~AppendStrideMessage() override = default;

 private:
  // Save the args, when we want to append them to tvm_access_prt
  // 0) Repeat Time
  // 1) Repeat Stride
  // 2) Block Number/ Number Burst
  // 3) Block Stride/ Length Burst
  // 4) Block Size  / Burst Size
  std::vector<Expr> args_storage_;

  Expr Mutate_(const Call *op, const Expr &expr) final;
  // It maps the string to target instructions
  std::map<std::string, std::function<Expr(Expr)>> call_to_func_{
    {"copy_gm_to_cbuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_gm_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_ubuf_to_cbuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_ubuf_to_gm", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_matrix_ubuf_to_cc", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_vector_ubuf_to_cc", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_small_matrix_ubuf_to_cc", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_matrix_cc_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_vector_cc_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_small_matrix_cc_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_ubuf_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"copy_cbuf_to_ubuf", [this](Expr expr) { return MutateDmaMov(std::move(expr)); }},
    {"load_cbuf_to_ca", [this](Expr expr) { return MutateDmaLoad2D(std::move(expr)); }},
    {"load_cbuf_to_cb", [this](Expr expr) { return MutateDmaLoad2D(std::move(expr)); }},
    {"load_gm_to_ca", [this](Expr expr) { return MutateDmaLoad2D(std::move(expr)); }},
    {"load_gm_to_cb", [this](Expr expr) { return MutateDmaLoad2D(std::move(expr)); }},
    {"load_gm_to_cbuf", [this](Expr expr) { return MutateDmaLoad2D(std::move(expr)); }},
    {"img2col_cbuf_to_ca", [this](Expr expr) { return MutateDmaLoad3D(std::move(expr)); }},
    {"img2col_cbuf_to_cb", [this](Expr expr) { return MutateDmaLoad3D(std::move(expr)); }},
    {"img2col_cbuf_to_ub", [this](Expr expr) { return MutateDmaLoad3D(std::move(expr)); }},
    {"broadcast_ub_to_cc", [this](Expr expr) { return MutateDmaBroadCast(std::move(expr)); }},
    {"vector_dup", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vabs", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vexp", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vrelu", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vrec", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vln", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vrsqrt", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vsqrt", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vnot", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vadd", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vsub", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmul", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vdiv", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmax", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmin", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmadd", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmaddrelu", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmla", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vor", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vand", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vcmp_eq", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmp_ne", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmp_lt", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmp_le", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmp_gt", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmp_ge", [this](Expr expr) { return MutateCMPOpVector(std::move(expr)); }},
    {"vcmpv_eq", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vcmpv_ne", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vcmpv_lt", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vcmpv_le", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vcmpv_gt", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vcmpv_ge", [this](Expr expr) { return MutateVCMPVOpVector(std::move(expr)); }},
    {"vsel", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vaxpy", [this](Expr expr) { return MutateSingleVSOpVector(std::move(expr)); }},
    {"vadds", [this](Expr expr) { return MutateSingleVSOpVector(std::move(expr)); }},
    {"vmuls", [this](Expr expr) { return MutateSingleVSOpVector(std::move(expr)); }},
    {"vconv_f322f16", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162f32", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s8", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162u8", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_deq", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s32f", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s32c", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s32r", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_u82f16", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_s82f16", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s32a", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s32z", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s8a", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s8f", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s8c", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162s8z", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322s32a", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322s32f", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322s32c", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322s32z", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322s32r", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162u8a", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162u8f", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162u8c", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f162u8z", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_s322f32", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vconv_f322f16o", [this](Expr expr) { return MutateSingleOpVector(std::move(expr)); }},
    {"vmulconv_f162s8", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vmulconv_f162u8", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vadddeqrelu", [this](Expr expr) { return MutateMultiOpVector(std::move(expr)); }},
    {"vcadd", [this](Expr expr) { return MutateReduceVCNmlOpVector(std::move(expr)); }},
    {"vcmax", [this](Expr expr) { return MutateReduceVCArgOpVector(std::move(expr)); }},
    {"vcmin", [this](Expr expr) { return MutateReduceVCArgOpVector(std::move(expr)); }},
    {"vcgadd", [this](Expr expr) { return MutateReduceVCGOpVector(std::move(expr)); }},
    {"vcgmax", [this](Expr expr) { return MutateReduceVCGOpVector(std::move(expr)); }},
    {"vcgmin", [this](Expr expr) { return MutateReduceVCGOpVector(std::move(expr)); }}};

  std::map<std::string, std::unordered_set<int>> command_src_length_check_ = {
    {"copy_matrix_ubuf_to_cc", {2, 4}},
    {"copy_vector_ubuf_to_cc", {2, 4}},
    {"copy_small_matrix_ubuf_to_cc", {2, 4}},
    {"copy_matrix_cc_to_ubuf", {2, 4}},
    {"copy_small_matrix_cc_to_ubuf", {2, 4}},
    {"dma_broad_cast", {2, 4}},
    {"dma_load_2d", {1, 2, 4}},
    {"dma_load_3d", {1, 2, 4}},
  };
  std::map<std::string, std::unordered_set<int>> command_dst_length_check_ = {
    {"copy_matrix_ubuf_to_cc", {2, 4}},
    {"copy_vector_ubuf_to_cc", {2, 4}},
    {"copy_small_matrix_ubuf_to_cc", {2, 4}},
    {"copy_matrix_cc_to_ubuf", {2, 4}},
    {"copy_small_matrix_cc_to_ubuf", {2, 4}},
    {"dma_broad_cast", {2, 4}},
    {"dma_load_2d", {1, 2, 4}},
    {"dma_load_3d", {1, 2, 4}},
  };
  /// Declaration of specific instruction for args calculation
  // Get type length of dst and src buffers in one call
  std::pair<int, int> GetDstSrcTypeLength(const Call *call, bool allow_exceed);
  // Common method to check dst and src buffer's type length
  void CheckCommandLength(const std::string &cmd_name, int dstTypeLength, int srcTypeLength);
  // Manage the DmaMov's stride message.
  Expr MutateDmaMov(const Expr &expr);
  // Load2Dop , add 5 parameters indclude repeat time, repeat stride, block number, block stride,
  // block size
  Expr MutateDmaLoad2D(const Expr &expr);
  // Load3Dop , add 5 parameters indclude repeat time, repeat stride, block number, block stride,
  // block size
  Expr MutateDmaLoad3D(const Expr &expr);
  // BroadCastop , add 5 parameters indclude repeat time, repeat stride, block number, block stride,
  // block size
  Expr MutateDmaBroadCast(const Expr &expr);

  Expr AppendDmaStrideMsg(const Expr &expr, Arch::BurstLengthByte srcBurstLengthByte,
                          Arch::BurstLengthByte dstBurstLengthByte, Arch::GapUnitByte srcGapUnitByte,
                          Arch::GapUnitByte dstGapUnitByte);

  // SingleOp pattern, the RepeatCount/BlockStride/RepeatStride index of single operation vector are
  // 2/3/5
  Expr MutateSingleOpVector(const Expr &expr);

  // SingleVSOp pattern, the RepeatCount/BlockStride/RepeatStride index of singleVS operation vector
  // are 3/4/6
  Expr MutateSingleVSOpVector(const Expr &expr);

  // MultiOp pattern, the RepeatCount/BlockStride/RepeatStride index of multiple operation vector
  // are 3/4/7
  Expr MutateMultiOpVector(const Expr &expr);

  // CMPOp pattern, the RepeatCount/BlockStride/RepeatStride index of compare operation vector are
  // 2/4/6
  Expr MutateCMPOpVector(const Expr &expr);

  // all args are calculated in vector emit_insn, we just fetch and put them into first 2-3 Call
  // args. there are 5 args need to handle: repeat count, repeat stride, block number, block stride,
  // block size. the block number is 8 and block size is 32 that are definited before.
  Expr MutateVector(const Expr &expr, int repeatCIndex, int repeatSIndex, int blockSIndex);

  // VCMPOp pattern, the RepeatCount/BlockStride/RepeatStride index of compare operation vector are
  // 3/4/7. Specially, the dst block stride is 1B, repeat stride is 8B, if src type cost 32bit(4B)
  // the dst block stride is 2B, repeat stride is 16B, if src type cost 16bit(2B)
  Expr MutateVCMPVOpVector(const Expr &expr);

  // for vcadd
  Expr MutateReduceVCNmlOpVector(const Expr &expr);

  // for vcmax vcmin
  Expr MutateReduceVCArgOpVector(const Expr &expr);

  // for vcgXXX
  Expr MutateReduceVCGOpVector(const Expr &expr);

  // ReduceOp pattern, the RepeatCount/BlockStride/RepeatStride index of compare operation vector are
  // 2/3/5. Specially, the dst block may be 0, which need be calculate by coef.
  Expr MutateReduceOpVector(const Expr &expr, uint8_t coef);
};

/**
 * Visit Call stmt Clear their tvm_access_prt's stride message
 * ```
 * Before:
 *     tvm_access_prt(Type, bufferVar, offset, extent, rw,
 *                    RepeatTime, RepeatStride, BlockNumber, BlockStride, BlockSize)
 *
 * After:
 *     tvm_access_prt(Type, bufferVar, offset, extent, rw)
 * ```
 */
class RemoveStrideMessage final : public IRMutator {
 public:
  RemoveStrideMessage() = default;
  ~RemoveStrideMessage() override = default;

 private:
  Expr Mutate_(const Call *op, const Expr &expr) final;
};
}  // namespace ir
}  // namespace akg
#endif  // PASS_APPEND_STRIDE_MSG_H_
