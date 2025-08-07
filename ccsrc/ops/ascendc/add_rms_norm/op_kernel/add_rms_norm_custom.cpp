/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

/*!
 * \file add_rms_norm.cpp
 * \brief
 */

#include "kernel_operator.h"

using namespace AscendC;

#ifdef __CCE_KT_TEST__
#define __aicore__
#else
#define __aicore__ [aicore]
#endif

#if __CCE_AICORE__ != 220
#define bfloat16_t int16_t
#endif
constexpr int32_t BUFFER_NUM = 1;        // tensor num for each queue
constexpr int32_t NUM_PER_REP_FP32 = 64; // ONE_REPEAT_BYTE_SIZE / sizeof(float);
constexpr int32_t NUM_PER_BLK_FP32 = 8;
constexpr float MINUS_HALF = -0.5;
constexpr float ZERO = 0;
constexpr float ONE = 1;

template <typename T> __aicore__ inline T CeilDiv(T x, T y) { return y == 0 ? x : (x + y - 1) / y; }

template <typename Tp, Tp v> struct integral_constant { static constexpr Tp value = v; };
using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename> struct is_same : public false_type {};
template <typename Tp> struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline void ReduceSumFP32(const LocalTensor<float> &dst_local,
                                     const LocalTensor<float> &src_local,
                                     const LocalTensor<float> &work_local, int32_t count) {
  // count need smaller than 255 repeat
  if (g_coreType == AIV) {
    uint64_t mask = NUM_PER_REP_FP32;
    int32_t repeatTimes = count / NUM_PER_REP_FP32;
    int32_t tailCount = count % NUM_PER_REP_FP32;
    int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
    BinaryRepeatParams repeatParams;
    repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
    repeatParams.src0BlkStride = 1;
    repeatParams.src1RepStride = 0;
    repeatParams.src1BlkStride = 1;
    repeatParams.dstRepStride = 0;
    repeatParams.dstBlkStride = 1;
    Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
    pipe_barrier(PIPE_V);
    if (likely(repeatTimes > 0)) {
      Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
      pipe_barrier(PIPE_V);
    }
    if (unlikely(tailCount != 0)) {
      Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
      pipe_barrier(PIPE_V);
    }
    AscendCUtils::SetMask<float>(NUM_PER_REP_FP32);
    vcadd((__ubuf__ float *)dst_local.GetPhyAddr(), (__ubuf__ float *)work_local.GetPhyAddr(), 1, 0,
          1, 0, false);
    pipe_barrier(PIPE_V);
  }
}

__aicore__ inline void ReduceSumCustom(const LocalTensor<float> &dst_local,
                                       const LocalTensor<float> &src_local,
                                       const LocalTensor<float> &work_local, int32_t count) {
#if __CCE_AICORE__ == 220
  ReduceSumFP32(dst_local, src_local, work_local, count);
#else
  ReduceSum(dst_local, src_local, dst_local, count);
#endif
}

__aicore__ inline void ReduceSumFP32ToBlock(const LocalTensor<float> &dst_local,
                                            const LocalTensor<float> &src_local,
                                            const LocalTensor<float> &work_local, int32_t count) {
  // count need smaller than 255 repeat
  uint64_t mask = NUM_PER_REP_FP32;
  int32_t repeatTimes = count / NUM_PER_REP_FP32;
  int32_t tailCount = count % NUM_PER_REP_FP32;
  int32_t bodyCount = repeatTimes * NUM_PER_REP_FP32;
  BinaryRepeatParams repeatParams;
  repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
  repeatParams.src0BlkStride = 1;
  repeatParams.src1RepStride = 0;
  repeatParams.src1BlkStride = 1;
  repeatParams.dstRepStride = 0;
  repeatParams.dstBlkStride = 1;
  Duplicate(work_local, ZERO, NUM_PER_REP_FP32);
  pipe_barrier(PIPE_V);
  if (likely(repeatTimes > 0)) {
    Add(work_local, src_local, work_local, mask, repeatTimes, repeatParams);
    pipe_barrier(PIPE_V);
  }
  if (unlikely(tailCount != 0)) {
    Add(work_local, src_local[bodyCount], work_local, tailCount, 1, repeatParams);
    pipe_barrier(PIPE_V);
  }
  BlockReduceSum(dst_local, work_local, 1, mask, 1, 1, DEFAULT_REPEAT_STRIDE);
  pipe_barrier(PIPE_V);
}

__aicore__ inline void BlockReduceSumFP32(const LocalTensor<float> &dst_local,
                                          const LocalTensor<float> &src_local, int32_t count) {
  // count need multiple of 8
  int32_t repeatTimes = count / NUM_PER_REP_FP32;
  int32_t tailCount = count % NUM_PER_REP_FP32;
  int32_t dstAddr = repeatTimes * 8;
  int32_t srcAddr = repeatTimes * NUM_PER_REP_FP32;
  if (likely(repeatTimes > 0)) {
    BlockReduceSum(dst_local, src_local, repeatTimes, NUM_PER_REP_FP32, 1, 1,
                   DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
  }
  if (tailCount != 0) {
    BlockReduceSum(dst_local[dstAddr], src_local[srcAddr], 1, tailCount, 1, 1,
                   DEFAULT_REPEAT_STRIDE);
    pipe_barrier(PIPE_V);
  }
}

template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U &dstTensor, const R &srcTensor,
                                      const uint32_t count) {
#if __CCE_AICORE__ == 220
  DataCopyParams copyParams;
  copyParams.blockLen = count * sizeof(T);
  copyParams.blockCount = 1;
  if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
    DataCopyPadParams padParams;
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
  } else {
    DataCopyPad(dstTensor, srcTensor, copyParams);
  }
#else
  // only support count greater than 32byte
  int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
  if (count % numPerBlock == 0) {
    DataCopy(dstTensor, srcTensor, count);
  } else {
    if constexpr (is_same<U, AscendC::LocalTensor<T>>::value) {
      int32_t num = AlignUp(count, numPerBlock);
      DataCopy(dstTensor, srcTensor, num);
    } else {
      int32_t num = count / numPerBlock * numPerBlock;
      DataCopy(dstTensor, srcTensor, num);
      set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
      for (int32_t i = 0; i < numPerBlock; i++) {
        T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
        srcTensor.SetValue(i, tensorValue);
      }
      set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
      DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
    }
  }
#endif
}

template <typename T> class KernelAddRmsNorm {
public:
  __aicore__ inline KernelAddRmsNorm() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                              GM_ADDR x, uint32_t numRow, uint32_t numCol, uint32_t blockFactor,
                              uint32_t rowFactor, uint32_t ubFactor, float epsilon,
                              bool is_cast_gamma = false) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->numRow = numRow;
    this->numCol = numCol;
    this->blockFactor = blockFactor;
    this->rowFactor = rowFactor;
    this->ubFactor = ubFactor;
    this->epsilon = epsilon;
    this->avgFactor = (float)1.0 / numCol;
    this->is_cast_gamma = is_cast_gamma;

    if (GetBlockIdx() < GetBlockNum() - 1) {
      this->rowWork = blockFactor;
    } else if (GetBlockIdx() == GetBlockNum() - 1) {
      this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
    } else {
    }
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer((__gm__ T *)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    x2Gm.SetGlobalBuffer((__gm__ T *)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    if (is_cast_gamma) {
      gammaGmFp32.SetGlobalBuffer((__gm__ float *)gamma, numCol);
    } else {
      gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);
    }
    yGm.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    rstdGm.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * blockFactor, blockFactor);
    xGm.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(inQueueX, BUFFER_NUM, ubFactor * sizeof(T));
    if (is_cast_gamma) {
      pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(float));
    } else {
      pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    }
    pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      pipe.InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    pipe.InitBuffer(sqxBuf, ubFactor * sizeof(float));
    pipe.InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    CopyInGamma();
    uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
    if (is_cast_gamma) {
      LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
      // SubProcess(0, rowFactor, gammaLocal);
      for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
        SubProcessFp32(i_o, rowFactor, gammaLocal);
      }
      SubProcessFp32(i_o_max - 1, row_tail, gammaLocal);
      inQueueGamma.FreeTensor(gammaLocal);
    } else {
      LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
      // SubProcess(0, rowFactor, gammaLocal);
      for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
        SubProcess(i_o, rowFactor, gammaLocal);
      }
      SubProcess(i_o_max - 1, row_tail, gammaLocal);
      inQueueGamma.FreeTensor(gammaLocal);
    }
  }

  __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num,
                                    LocalTensor<T> &gammaLocal) {
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
      CopyIn(gm_bias);
      Compute(i_i, gammaLocal, rstdLocal);
      CopyOutY(gm_bias);
    }
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o, calc_row_num);
  }

  __aicore__ inline void SubProcessFp32(uint32_t i_o, uint32_t calc_row_num,
                                        LocalTensor<float> &gammaLocal) {
    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      uint32_t gm_bias = (i_o * rowFactor + i_i) * numCol;
      CopyIn(gm_bias);
      ComputeFp32(i_i, gammaLocal, rstdLocal);
      CopyOutY(gm_bias);
    }
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o, calc_row_num);
  }

private:
  __aicore__ inline void CopyIn(uint32_t gm_bias) {
    LocalTensor<T> x1Local_in = inQueueX.AllocTensor<T>();
    LocalTensor<T> x2Local = sqxBuf.Get<T>();
    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      x2Local = x2Local[ubFactor];
    }

    DataCopyCustom<T>(x1Local_in, x1Gm[gm_bias], numCol);
    DataCopyCustom<T>(x2Local, x2Gm[gm_bias], numCol);
    inQueueX.EnQue(x1Local_in);
    auto x1Local = inQueueX.DeQue<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      Add(xLocal, x1Local, x2Local, numCol);
      pipe_barrier(PIPE_V);
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
      pipe_barrier(PIPE_V);
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 = sqxBuf.Get<float>();
      Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, numCol);
      Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, numCol);
      pipe_barrier(PIPE_V);
      Add(x1_fp32, x1_fp32, x2_fp32, numCol);
      pipe_barrier(PIPE_V);
      Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, numCol);
      pipe_barrier(PIPE_V);

      // cast for precision issue
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
      pipe_barrier(PIPE_V);
    } else {
      Add(x1Local, x1Local, x2Local, numCol);
      pipe_barrier(PIPE_V);
      Adds(xLocal, x1Local, (float)0, numCol);
    }
    inQueueX.FreeTensor(x1Local);

    // CopyOut x1 + x2
    outQueueY.EnQue(xLocal);
    auto x_out = outQueueY.DeQue<T>();
    DataCopyCustom<T>(xGm[gm_bias], x_out, numCol);
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void CopyInGamma() {
    if (is_cast_gamma) {
      LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
      DataCopyCustom<float>(gammaLocal, gammaGmFp32, numCol);
      inQueueGamma.EnQue(gammaLocal);
    } else {
      LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
      DataCopyCustom<T>(gammaLocal, gammaGm, numCol);
      inQueueGamma.EnQue(gammaLocal);
    }
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<float> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Mul(sqx, xLocal, xLocal, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);
    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstd_value, numCol);
    inQueueX.FreeTensor(xLocal);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, numCol);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<bfloat16_t> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);
    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);

    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, numCol);
    pipe_barrier(PIPE_V);
    LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
    pipe_barrier(PIPE_V);
    Cast(x_fp32, yLocal, RoundMode::CAST_NONE, numCol);
    pipe_barrier(PIPE_V);
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, numCol); // gamma_fp32 reuse sqx
    pipe_barrier(PIPE_V);
    Mul(x_fp32, x_fp32, sqx, numCol);
    pipe_barrier(PIPE_V);
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
    pipe_barrier(PIPE_V);

    event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
    wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);

    outQueueY.EnQue<bfloat16_t>(yLocal);
  }

  __aicore__ inline void Compute(uint32_t inner_progress, LocalTensor<half> gammaLocal,
                                 LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);

    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, numCol);
    pipe_barrier(PIPE_V);
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

    event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
    wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);

    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, numCol);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<half>(yLocal);
  }

  __aicore__ inline void ComputeFp32(uint32_t inner_progress, LocalTensor<float> gammaLocal,
                                     LocalTensor<float> rstdLocal) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();

    Mul(sqx, x_fp32, x_fp32, numCol);
    pipe_barrier(PIPE_V);

    Muls(sqx, sqx, avgFactor, numCol);
    pipe_barrier(PIPE_V);

    ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
    pipe_barrier(PIPE_V);

    Adds(sqx, sqx, epsilon, 1);
    pipe_barrier(PIPE_V);

    Sqrt(sqx, sqx, 1);
    Duplicate(reduce_buf_local, ONE, 1);
    pipe_barrier(PIPE_V);
    Div(sqx, reduce_buf_local, sqx, 1);
    pipe_barrier(PIPE_V);

    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = sqx.GetValue(0);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    rstdLocal.SetValue(inner_progress, rstd_value);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, numCol);
    pipe_barrier(PIPE_V);
    Mul(x_fp32, x_fp32, gammaLocal, numCol);
    pipe_barrier(PIPE_V);
    if (is_same<T, half>::value) {
      LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();

      Cast(yLocal, x_fp32, RoundMode::CAST_NONE, numCol);

      event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
      set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
      wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);
      pipe_barrier(PIPE_V);

      outQueueY.EnQue<half>(yLocal);
    } else {
      LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();

      Cast(yLocal, x_fp32, RoundMode::CAST_RINT, numCol);
      pipe_barrier(PIPE_V);

      event_t event_v_mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
      set_flag(PIPE_V, PIPE_MTE2, event_v_mte);
      wait_flag(PIPE_V, PIPE_MTE2, event_v_mte);
      pipe_barrier(PIPE_V);

      outQueueY.EnQue<bfloat16_t>(yLocal);
    }
  }

  __aicore__ inline void CopyOutY(uint32_t progress) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    DataCopyCustom<T>(yGm[progress], yLocal, numCol);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOutRstd(uint32_t outer_progress, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
    // #if __CCE_AICORE__ == 220
    //     DataCopyCustom<float>(rstdGm[outer_progress * rowFactor], rstdLocal, num);
    // #endif
    outQueueRstd.FreeTensor(rstdLocal);
  }

private:
  TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;

  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> reduceFp32Buf;
  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<float> gammaGmFp32;
  GlobalTensor<T> yGm;
  GlobalTensor<float> rstdGm;
  GlobalTensor<T> xGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor; // number of calculations rows on each core
  uint32_t rowFactor;
  uint32_t ubFactor;
  float epsilon;
  float avgFactor;
  bool is_cast_gamma;

  uint32_t rowWork = 1;
};

template <typename T> class KernelAddRmsNormSplitD {
public:
  __aicore__ inline KernelAddRmsNormSplitD() {}
  __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd,
                              GM_ADDR x, GM_ADDR workspace, uint32_t numRow, uint32_t numCol,
                              uint32_t blockFactor, uint32_t rowFactor, uint32_t ubFactor,
                              float epsilon, bool is_cast_gamma = false) {
    ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
    this->numRow = numRow;
    this->numCol = numCol;
    this->blockFactor = blockFactor;
    this->rowFactor = rowFactor;
    this->ubFactor = ubFactor;
    this->epsilon = epsilon;
    this->avgFactor = (float)1.0 / numCol;
    this->is_cast_gamma = is_cast_gamma;

    if (GetBlockIdx() < GetBlockNum() - 1) {
      this->rowWork = blockFactor;
    } else if (GetBlockIdx() == GetBlockNum() - 1) {
      this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
    } else {
    }
    // get start index for current core, core parallel
    x1Gm.SetGlobalBuffer((__gm__ T *)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    x2Gm.SetGlobalBuffer((__gm__ T *)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    if (is_cast_gamma) {
      gammaGmFp32.SetGlobalBuffer((__gm__ float *)gamma, numCol);
    } else {
      gammaGm.SetGlobalBuffer((__gm__ T *)gamma, numCol);
    }
    yGm.SetGlobalBuffer((__gm__ T *)y + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
    rstdGm.SetGlobalBuffer((__gm__ float *)rstd + GetBlockIdx() * blockFactor, blockFactor);
    xGm.SetGlobalBuffer((__gm__ T *)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);

    // pipe alloc memory to queue, the unit is Bytes.
    // We need 2 buffers here for both x1 and x2.
    pipe.InitBuffer(inQueueX, BUFFER_NUM, 2 * ubFactor * sizeof(T));
    if (is_cast_gamma) {
      pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(float));
    } else {
      pipe.InitBuffer(inQueueGamma, BUFFER_NUM, ubFactor * sizeof(T));
    }
    pipe.InitBuffer(outQueueY, BUFFER_NUM, ubFactor * sizeof(T));
    pipe.InitBuffer(outQueueRstd, BUFFER_NUM, rowFactor * sizeof(float));

    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      pipe.InitBuffer(xFp32Buf, ubFactor * sizeof(float));
    }
    pipe.InitBuffer(sqxBuf, ubFactor * sizeof(float));
    pipe.InitBuffer(sumBuf, rowFactor * NUM_PER_BLK_FP32 * sizeof(float));
    pipe.InitBuffer(reduceFp32Buf, NUM_PER_REP_FP32 * sizeof(float));
  }

  __aicore__ inline void Process() {
    uint32_t i_o_max = CeilDiv(rowWork, rowFactor);
    uint32_t row_tail = rowWork - (i_o_max - 1) * rowFactor;
    uint32_t j_max = CeilDiv(numCol, ubFactor);
    uint32_t col_tail = numCol - (j_max - 1) * ubFactor;
    for (uint32_t i_o = 0; i_o < i_o_max - 1; i_o++) {
      SubProcess(i_o, rowFactor, j_max, col_tail);
    }
    SubProcess(i_o_max - 1, row_tail, j_max, col_tail);
  }

  __aicore__ inline void SubProcess(uint32_t i_o, uint32_t calc_row_num, uint32_t j_max,
                                    uint32_t col_tail) {
    LocalTensor<float> sumLocal = sumBuf.Get<float>();

    LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
    Duplicate(rstdLocal, (float)0.0, calc_row_num);
    pipe_barrier(PIPE_V);
    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeFormer(i_o, calc_row_num, j, rstdLocal, sumLocal, ubFactor);
    }
    // do tail
    ComputeFormer(i_o, calc_row_num, j_max - 1, rstdLocal, sumLocal, col_tail);
    ComputeRstd(rstdLocal, calc_row_num);

    for (uint32_t j = 0; j < j_max - 1; j++) {
      ComputeLatter(i_o, calc_row_num, j, rstdLocal, ubFactor);
    }
    ComputeLatter(i_o, calc_row_num, j_max - 1, rstdLocal, col_tail);
    outQueueRstd.EnQue<float>(rstdLocal);
    CopyOutRstd(i_o, calc_row_num);
  }

private:
  __aicore__ inline void CopyInAndAdd(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    LocalTensor<T> x1x2_in = inQueueX.AllocTensor<T>();
    LocalTensor<T> x1_in = x1x2_in[0];
    LocalTensor<T> x2_in = x1x2_in[ubFactor];
    DataCopyCustom<T>(x1_in, x1Gm[i_idx * numCol + j_idx * ubFactor], num);
    DataCopyCustom<T>(x2_in, x2Gm[i_idx * numCol + j_idx * ubFactor], num);
    inQueueX.EnQue(x1x2_in);
    LocalTensor<T> x1x2Local = inQueueX.DeQue<T>();

    auto x1Local = x1x2Local[0];
    auto x2Local = x1x2Local[ubFactor];

    LocalTensor<T> xLocal = outQueueY.AllocTensor<T>();

    if constexpr (is_same<T, half>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();

      Add(xLocal, x1Local, x2Local, num);
      pipe_barrier(PIPE_V);
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      // x1+x2 saved in x1_fp32
    } else if constexpr (is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
      LocalTensor<float> x2_fp32 = x1x2Local.template ReinterpretCast<float>();

      Cast(x1_fp32, x1Local, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      Cast(x2_fp32, x2Local, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);

      Add(x1_fp32, x1_fp32, x2_fp32, num);
      pipe_barrier(PIPE_V);
      Cast(xLocal, x1_fp32, RoundMode::CAST_RINT, num);
      pipe_barrier(PIPE_V);

      // cast for precision issue
      Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      // x1+x2 saved in x1_fp32
    } else {
      Add(x1Local, x1Local, x2Local, num);
      pipe_barrier(PIPE_V);
      Adds(xLocal, x1Local, (float)0.0, num);
      // x1+x2 saved in inQueueX
    }
    inQueueX.FreeTensor(x1x2Local);

    // copy out to workspace && x_out
    outQueueY.EnQue(xLocal);
    auto x_out = outQueueY.DeQue<T>();
    DataCopyCustom<T>(xGm[i_idx * numCol + j_idx * ubFactor], x_out, num);
    outQueueY.FreeTensor(x_out);
  }

  __aicore__ inline void ComputeFormer(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                       LocalTensor<float> &rstdLocal, LocalTensor<float> &sumLocal,
                                       uint32_t num) {
    for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
      CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
      ComputeSum(i_i, sumLocal, num);
    }
    BlockReduceSumFP32(sumLocal, sumLocal, calc_row_num * NUM_PER_BLK_FP32);
    Add(rstdLocal, rstdLocal, sumLocal, calc_row_num);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ComputeSum(uint32_t i_i_idx, LocalTensor<float> &sumLocal, uint32_t num) {
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    if constexpr (is_same<T, half>::value || is_same<T, bfloat16_t>::value) {
      LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
      pipe_barrier(PIPE_V);
      Mul(sqx, x_fp32, x_fp32, num);
    } else {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<float>();
      pipe_barrier(PIPE_V);
      Mul(sqx, xLocal, xLocal, num);
      inQueueX.FreeTensor(xLocal);
    }
    pipe_barrier(PIPE_V);
    Muls(sqx, sqx, avgFactor, num);
    pipe_barrier(PIPE_V);
    // 8 means 8 fp32 pre block
    ReduceSumFP32ToBlock(sumLocal[i_i_idx * 8], sqx, reduce_buf_local, num);
  }

  __aicore__ inline void ComputeRstd(LocalTensor<float> rstdLocal, uint32_t num) {
    LocalTensor<float> reduce_buf_local = reduceFp32Buf.Get<float>();
    Adds(rstdLocal, rstdLocal, epsilon, num);
    pipe_barrier(PIPE_V);
    Sqrt(rstdLocal, rstdLocal, num);
    Duplicate(reduce_buf_local, ONE, num);
    pipe_barrier(PIPE_V);
    Div(rstdLocal, reduce_buf_local, rstdLocal, num);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void ComputeLatter(uint32_t i_o_idx, uint32_t calc_row_num, uint32_t j_idx,
                                       LocalTensor<float> &rstdLocal, uint32_t num) {
    CopyInGamma(j_idx, num);
    if (is_cast_gamma) {
      LocalTensor<float> gammaLocal = inQueueGamma.DeQue<float>();
      for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
        CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
        ComputeYFp32(i_i, gammaLocal, rstdLocal, num);
        CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
      }
      inQueueGamma.FreeTensor(gammaLocal);
    } else {
      LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
      for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
        CopyInAndAdd(i_o_idx * rowFactor + i_i, j_idx, num);
        ComputeY(i_i, gammaLocal, rstdLocal, num);
        CopyOutY(i_o_idx * rowFactor + i_i, j_idx, num);
      }
      inQueueGamma.FreeTensor(gammaLocal);
    }
  }

  __aicore__ inline void CopyInGamma(uint32_t j_idx, uint32_t num) {
    if (is_cast_gamma) {
      LocalTensor<float> gammaLocal = inQueueGamma.AllocTensor<float>();
      DataCopyCustom<float>(gammaLocal, gammaGmFp32[j_idx * ubFactor], num);
      inQueueGamma.EnQue(gammaLocal);
    } else {
      LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
      DataCopyCustom<T>(gammaLocal, gammaGm[j_idx * ubFactor], num);
      inQueueGamma.EnQue(gammaLocal);
    }
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<half> &gammaLocal,
                                  LocalTensor<float> &rstdLocal, uint32_t num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, num);
    pipe_barrier(PIPE_V);
    LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
    Cast(yLocal, x_fp32, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<half>(yLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<float> &gammaLocal,
                                  LocalTensor<float> &rstdLocal, uint32_t num) {
    LocalTensor<float> xLocal = inQueueX.AllocTensor<float>(); // inQueueX.DeQue<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
    Muls(yLocal, xLocal, rstd_value, num);
    inQueueX.FreeTensor(xLocal);
    pipe_barrier(PIPE_V);
    Mul(yLocal, gammaLocal, yLocal, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<float>(yLocal);
  }

  __aicore__ inline void ComputeY(uint32_t i_i_idx, LocalTensor<bfloat16_t> &gammaLocal,
                                  LocalTensor<float> &rstdLocal, uint32_t num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, num);
    pipe_barrier(PIPE_V);
    LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
    pipe_barrier(PIPE_V);
    Cast(x_fp32, yLocal, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Cast(sqx, gammaLocal, RoundMode::CAST_NONE, num);
    pipe_barrier(PIPE_V);
    Mul(x_fp32, x_fp32, sqx, num);
    pipe_barrier(PIPE_V);
    Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
    pipe_barrier(PIPE_V);
    outQueueY.EnQue<bfloat16_t>(yLocal);
  }

  __aicore__ inline void ComputeYFp32(uint32_t i_i_idx, LocalTensor<float> &gammaLocal,
                                      LocalTensor<float> &rstdLocal, uint32_t num) {
    LocalTensor<float> x_fp32 = xFp32Buf.Get<float>();
    LocalTensor<float> sqx = sqxBuf.Get<float>();
    event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    set_flag(PIPE_V, PIPE_S, event_v_s);
    wait_flag(PIPE_V, PIPE_S, event_v_s);
    float rstd_value = rstdLocal.GetValue(i_i_idx);
    event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    set_flag(PIPE_S, PIPE_V, event_s_v);
    wait_flag(PIPE_S, PIPE_V, event_s_v);
    pipe_barrier(PIPE_V);
    Muls(x_fp32, x_fp32, rstd_value, num);
    pipe_barrier(PIPE_V);
    Mul(x_fp32, gammaLocal, x_fp32, num);
    pipe_barrier(PIPE_V);
    if (is_same<T, half>::value) {
      LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
      Cast(yLocal, x_fp32, RoundMode::CAST_NONE, num);
      pipe_barrier(PIPE_V);
      outQueueY.EnQue<half>(yLocal);
    } else {
      LocalTensor<bfloat16_t> yLocal = outQueueY.AllocTensor<bfloat16_t>();
      Cast(yLocal, x_fp32, RoundMode::CAST_RINT, num);
      pipe_barrier(PIPE_V);
      outQueueY.EnQue<bfloat16_t>(yLocal);
    }
  }

  __aicore__ inline void CopyOutY(uint32_t i_idx, uint32_t j_idx, uint32_t num) {
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    pipe_barrier(PIPE_ALL);
    DataCopyCustom<T>(yGm[i_idx * numCol + j_idx * ubFactor], yLocal, num);
    pipe_barrier(PIPE_ALL);
    outQueueY.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyOutRstd(uint32_t i_o_idx, uint32_t num) {
    LocalTensor<float> rstdLocal = outQueueRstd.DeQue<float>();
#if __CCE_AICORE__ == 220
    DataCopyCustom<float>(rstdGm[i_o_idx * rowFactor], rstdLocal, num);
#endif
    outQueueRstd.FreeTensor(rstdLocal);
  }

private:
  TPipe pipe;
  // create queues for input, in this case depth is equal to buffer num
  TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueGamma;
  // create queues for output, in this case depth is equal to buffer num
  TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY, outQueueRstd;
  TBuf<TPosition::VECCALC> xFp32Buf;
  TBuf<TPosition::VECCALC> sqxBuf;
  TBuf<TPosition::VECCALC> sumBuf;
  TBuf<TPosition::VECCALC> reduceFp32Buf;

  GlobalTensor<T> x1Gm;
  GlobalTensor<T> x2Gm;
  GlobalTensor<T> gammaGm;
  GlobalTensor<float> gammaGmFp32;
  GlobalTensor<T> yGm;
  GlobalTensor<float> rstdGm;
  GlobalTensor<T> xGm;

  uint32_t numRow;
  uint32_t numCol;
  uint32_t blockFactor; // number of calculations rows on each core
  uint32_t rowFactor;
  uint32_t ubFactor;
  float epsilon;
  float avgFactor;
  bool is_cast_gamma;
  uint32_t rowWork = 1;

  int tempbufNum;
};

inline __aicore__ int32_t AlignDiv32(int32_t n) { return ((n + 31) & ~31) / 32; }

extern "C" __global__ __aicore__ void add_rms_norm_custom(GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma,
                                                          GM_ADDR y, GM_ADDR rstd, GM_ADDR x,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
  if (TILING_KEY_IS(10)) {
    KernelAddRmsNorm<half> op;
    op.Init(x1, x2, gamma, y, rstd, x, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  } else if (TILING_KEY_IS(20)) {
    KernelAddRmsNorm<float> op;
    op.Init(x1, x2, gamma, y, rstd, x, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  } else if (TILING_KEY_IS(30)) {
    KernelAddRmsNorm<bfloat16_t> op;
    op.Init(x1, x2, gamma, y, rstd, x, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
    KernelAddRmsNormSplitD<half> op;
    op.Init(x1, x2, gamma, y, rstd, x, usrWorkspace, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  } else if (TILING_KEY_IS(21)) {
    KernelAddRmsNormSplitD<float> op;
    op.Init(x1, x2, gamma, y, rstd, x, usrWorkspace, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  } else if (TILING_KEY_IS(31)) {
    KernelAddRmsNormSplitD<bfloat16_t> op;
    op.Init(x1, x2, gamma, y, rstd, x, usrWorkspace, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon);
    op.Process();
  }

  if (TILING_KEY_IS(110)) {
    KernelAddRmsNorm<half> op;
    op.Init(x1, x2, gamma, y, rstd, x, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon, true);
    op.Process();
  } else if (TILING_KEY_IS(130)) {
    KernelAddRmsNorm<bfloat16_t> op;
    op.Init(x1, x2, gamma, y, rstd, x, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon, true);
    op.Process();
  } else if (TILING_KEY_IS(111)) {
    KernelAddRmsNormSplitD<half> op;
    op.Init(x1, x2, gamma, y, rstd, x, usrWorkspace, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon, true);
    op.Process();
  } else if (TILING_KEY_IS(131)) {
    KernelAddRmsNormSplitD<bfloat16_t> op;
    op.Init(x1, x2, gamma, y, rstd, x, usrWorkspace, tilingData.num_row, tilingData.num_col,
            tilingData.block_factor, tilingData.row_factor, tilingData.ub_factor,
            tilingData.epsilon, true);
    op.Process();
  }
}

void add_rms_norm_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *x1, uint8_t *x2,
                            uint8_t *gamma, uint8_t *y, uint8_t *rstd, uint8_t *x,
                            uint8_t *workspace, uint8_t *tiling) {
  add_rms_norm_custom<<<blockDim, l2ctrl, stream>>>(x1, x2, gamma, y, rstd, x, workspace, tiling);
}
