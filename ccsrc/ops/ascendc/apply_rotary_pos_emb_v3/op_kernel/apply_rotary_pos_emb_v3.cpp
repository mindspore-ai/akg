/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;
template <typename T, bool IS_SPLIT>
class KernelApplyRotaryPosEmbV3 {
 public:
  __aicore__ inline KernelApplyRotaryPosEmbV3() {}
  __aicore__ inline void Init(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin, GM_ADDR oquery, GM_ADDR okey,
                              GM_ADDR workspace, ApplyRotaryPosEmbV3TilingData *tiling, AscendC::TPipe *tPipe) {
    pipe = tPipe;
    tilingData = tiling;
    if constexpr (IS_SPLIT) {
      this->originQueryHiddenSize = tilingData->queryHeadDim * tilingData->qHeadNum;
      this->originKeyHiddenSize = tilingData->queryHeadDim * tilingData->kHeadNum;
    } else {
      this->originQueryHiddenSize = tilingData->qHiddenSize;
      this->originKeyHiddenSize = tilingData->kHiddenSize;
    }
    this->blockId = AscendC::GetBlockIdx();
    this->loop = this->blockId < tilingData->tokensTail ? tilingData->tokensPerCore + 1 : tilingData->tokensPerCore;
    this->startOffset = this->blockId < tilingData->tokensTail
                          ? (this->blockId * tilingData->tokensPerCore + this->blockId)
                          : (this->blockId * tilingData->tokensPerCore + tilingData->tokensTail);
    queryKeyCalHiddenSize = tilingData->qHiddenSize + tilingData->kHiddenSize;
    queryKeyInHiddenSize = this->originQueryHiddenSize + this->originKeyHiddenSize;

    queryInParams = {static_cast<uint16_t>(tilingData->qHeadNum),
                     static_cast<uint16_t>((tilingData->rotaryDim * sizeof(T)) / 32),
                     static_cast<uint16_t>(((tilingData->queryHeadDim - tilingData->rotaryDim) * sizeof(T)) / 32),
                     static_cast<uint16_t>(0)};
    keyInParams = {static_cast<uint16_t>(tilingData->kHeadNum),
                   static_cast<uint16_t>((tilingData->rotaryDim * sizeof(T)) / 32),
                   static_cast<uint16_t>(((tilingData->queryHeadDim - tilingData->rotaryDim) * sizeof(T)) / 32),
                   static_cast<uint16_t>(0)};
    queryOutParams = {static_cast<uint16_t>(tilingData->qHeadNum),
                      static_cast<uint16_t>((tilingData->rotaryDim * sizeof(T)) / 32), static_cast<uint16_t>(0),
                      static_cast<uint16_t>(((tilingData->queryHeadDim - tilingData->rotaryDim) * sizeof(T)) / 32)};
    keyOutParams = {static_cast<uint16_t>(tilingData->kHeadNum),
                    static_cast<uint16_t>((tilingData->rotaryDim * sizeof(T)) / 32), static_cast<uint16_t>(0),
                    static_cast<uint16_t>(((tilingData->queryHeadDim - tilingData->rotaryDim) * sizeof(T)) / 32)};
    uint16_t blockLen = tilingData->cosHeadDim * sizeof(T) / 32;
    uint16_t blockCount = tilingData->qHeadNum + tilingData->kHeadNum;
    qkPrepareDataCopyParams = {blockCount, blockLen, blockLen, blockLen};
    qGm.SetGlobalBuffer((__gm__ T *)query + startOffset * this->originQueryHiddenSize, this->originQueryHiddenSize);
    kGm.SetGlobalBuffer((__gm__ T *)key + startOffset * this->originKeyHiddenSize, this->originKeyHiddenSize);
    qOutGm.SetGlobalBuffer((__gm__ T *)oquery + startOffset * this->originQueryHiddenSize, this->originQueryHiddenSize);
    kOutGm.SetGlobalBuffer((__gm__ T *)okey + startOffset * this->originKeyHiddenSize, this->originKeyHiddenSize);
    cosGm.SetGlobalBuffer((__gm__ T *)cos + startOffset * tilingData->cosHeadDim, tilingData->cosHeadDim);
    sinGm.SetGlobalBuffer((__gm__ T *)sin + startOffset * tilingData->cosHeadDim, tilingData->cosHeadDim);

    pipe->InitBuffer(qInQueue, BUFFER_NUM, (queryKeyInHiddenSize) * sizeof(T));
    pipe->InitBuffer(cosInQueue, BUFFER_NUM, tilingData->cosHeadDim * sizeof(T));
    pipe->InitBuffer(sinInQueue, BUFFER_NUM, tilingData->cosHeadDim * sizeof(T));
    pipe->InitBuffer(qOutQueue, BUFFER_NUM, (queryKeyCalHiddenSize) * sizeof(T));
    pipe->InitBuffer(originBuf, queryKeyCalHiddenSize * sizeof(T));
    pipe->InitBuffer(rotaryBuf, queryKeyCalHiddenSize * sizeof(T));
    pipe->InitBuffer(cosBuf, queryKeyCalHiddenSize * sizeof(T));
    pipe->InitBuffer(sinBuf, queryKeyCalHiddenSize * sizeof(T));
    pipe->InitBuffer(scatterBuf, queryKeyCalHiddenSize * sizeof(T));
  }
  __aicore__ inline void Process() {
    PrePareScatterOffset();
    for (uint32_t i = 0; i < loop; i++) {
      CopyIn(i);
      PrepareCosSin();
      PrepareQK();
      Compute();
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t index) {
    AscendC::LocalTensor<T> qLocal = qInQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> cosLocal = cosInQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> sinLocal = sinInQueue.AllocTensor<T>();
    AscendC::DataCopy(qLocal, qGm[index * this->originQueryHiddenSize], this->originQueryHiddenSize);
    AscendC::DataCopy(qLocal[this->originQueryHiddenSize], kGm[index * this->originKeyHiddenSize],
                      this->originKeyHiddenSize);
    AscendC::DataCopy(cosLocal, cosGm[index * tilingData->cosHeadDim], tilingData->cosHeadDim);
    AscendC::DataCopy(sinLocal, sinGm[index * tilingData->cosHeadDim], tilingData->cosHeadDim);
    qInQueue.EnQue(qLocal);
    cosInQueue.EnQue(cosLocal);
    sinInQueue.EnQue(sinLocal);
  }
  __aicore__ inline void PrepareCosSin() {
    AscendC::LocalTensor<T> cos = cosInQueue.DeQue<T>();
    AscendC::LocalTensor<T> cosTmp = cosBuf.Get<T>();
    for (uint32_t i = 0; i < 2 * (tilingData->qHeadNum + tilingData->kHeadNum); ++i) {
      AscendC::DataCopy(cosTmp[i * tilingData->cosHeadDim], cos, tilingData->cosHeadDim);
    }
    AscendC::LocalTensor<T> sin = sinInQueue.DeQue<T>();
    AscendC::LocalTensor<T> sinTmp = sinBuf.Get<T>();
    AscendC::DataCopy(sinTmp[tilingData->cosHeadDim], sin, tilingData->cosHeadDim);
    AscendC::Muls(sinTmp[0], sin, static_cast<T>(-1), static_cast<int32_t>(tilingData->cosHeadDim));
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t i = 1; i < tilingData->qHeadNum + tilingData->kHeadNum; ++i) {
      AscendC::DataCopy(sinTmp[i * tilingData->rotaryDim], sinTmp, tilingData->rotaryDim);
    }
    AscendC::PipeBarrier<PIPE_V>();

    cosInQueue.FreeTensor(cos);
    sinInQueue.FreeTensor(sin);
  }

  __aicore__ inline void PrepareQK() {
    AscendC::LocalTensor<T> queryKey = qInQueue.DeQue<T>();
    AscendC::LocalTensor<T> origin = originBuf.Get<T>();
    AscendC::LocalTensor<T> rotary = rotaryBuf.Get<T>();

    uint64_t rsvdCnt = 0;
    uint32_t maskIndex{0};
    if constexpr (IS_SPLIT) {
      for (uint32_t i = 0; i < tilingData->qHeadNum + tilingData->kHeadNum; ++i) {
        maskIndex = i * tilingData->rotaryDim;
        AscendC::GatherMask<T>(origin[maskIndex], queryKey[i * tilingData->queryHeadDim], 1, true,
                               tilingData->rotaryDim, gatherMask, rsvdCnt);
        AscendC::GatherMask<T>(origin[maskIndex + tilingData->cosHeadDim], queryKey[i * tilingData->queryHeadDim], 2,
                               true, tilingData->rotaryDim, gatherMask, rsvdCnt);
      }
    } else {
      for (uint32_t i = 0; i < tilingData->qHeadNum + tilingData->kHeadNum; ++i) {
        maskIndex = i * tilingData->rotaryDim;
        AscendC::GatherMask<T>(origin[maskIndex], queryKey[maskIndex], 1, true, tilingData->rotaryDim, gatherMask,
                               rsvdCnt);
        AscendC::GatherMask<T>(origin[maskIndex + tilingData->cosHeadDim], queryKey[maskIndex], 2, true,
                               tilingData->rotaryDim, gatherMask, rsvdCnt);
      }
    }
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::DataCopy(rotary, origin[tilingData->cosHeadDim], qkPrepareDataCopyParams);
    AscendC::DataCopy(rotary[tilingData->cosHeadDim], origin, qkPrepareDataCopyParams);
    AscendC::PipeBarrier<PIPE_V>();

    qInQueue.FreeTensor(queryKey);
  }
  __aicore__ inline void PrePareScatterOffset() {
    AscendC::LocalTensor<int32_t> dstOffset = scatterBuf.Get<int32_t>();
    for (uint32_t i = 0; i < (tilingData->qHeadNum + tilingData->kHeadNum); ++i) {
      AscendC::ArithProgression<int32_t>(dstOffset[i * tilingData->rotaryDim],
                                         static_cast<int32_t>(i * tilingData->rotaryDim * sizeof(T)), 2 * sizeof(T),
                                         static_cast<int32_t>(tilingData->cosHeadDim));
      AscendC::ArithProgression<int32_t>(dstOffset[i * tilingData->rotaryDim + tilingData->cosHeadDim],
                                         static_cast<int32_t>((i * tilingData->rotaryDim + 1) * sizeof(T)),
                                         2 * sizeof(T), static_cast<int32_t>(tilingData->cosHeadDim));
    }
    AscendC::PipeBarrier<PIPE_V>();
  }
  __aicore__ inline void Compute() {
    AscendC::LocalTensor<T> qOutLocal = qOutQueue.AllocTensor<T>();
    AscendC::LocalTensor<T> origin = originBuf.Get<T>();
    AscendC::LocalTensor<T> rotary = rotaryBuf.Get<T>();
    AscendC::LocalTensor<T> cos = cosBuf.Get<T>();
    AscendC::LocalTensor<T> sin = sinBuf.Get<T>();
    AscendC::Mul(origin, origin, cos, queryKeyCalHiddenSize);
    AscendC::Mul(rotary, rotary, sin, queryKeyCalHiddenSize);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Add(origin, origin, rotary, queryKeyCalHiddenSize);
    AscendC::PipeBarrier<PIPE_V>();

    AscendC::LocalTensor<int32_t> dstOffset = scatterBuf.Get<int32_t>();
    AscendC::Scatter(qOutLocal, origin, dstOffset.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0),
                     (queryKeyCalHiddenSize));
    qOutQueue.EnQue(qOutLocal);
  }
  __aicore__ inline void CopyOut(int32_t index) {
    AscendC::LocalTensor<T> out = qOutQueue.DeQue<T>();
    if constexpr (IS_SPLIT) {
      AscendC::DataCopy(qOutGm[index * this->originQueryHiddenSize], out, queryOutParams);
      AscendC::DataCopy(kOutGm[index * this->originKeyHiddenSize], out[tilingData->qHiddenSize], keyOutParams);
    } else {
      AscendC::DataCopy(qOutGm[index * tilingData->qHiddenSize], out, tilingData->qHiddenSize);
      AscendC::DataCopy(kOutGm[index * tilingData->kHiddenSize], out[tilingData->qHiddenSize], tilingData->kHiddenSize);
    }
    qOutQueue.FreeTensor(out);
  }

 private:
  AscendC::TPipe *pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> qInQueue;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> cosInQueue;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> sinInQueue;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> qOutQueue;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> originBuf;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> rotaryBuf;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> cosBuf;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> sinBuf;
  AscendC::TBuf<AscendC::QuePosition::VECCALC> scatterBuf;
  AscendC::GlobalTensor<T> qGm;
  AscendC::GlobalTensor<T> kGm;
  AscendC::GlobalTensor<T> cosGm;
  AscendC::GlobalTensor<T> sinGm;
  AscendC::GlobalTensor<T> qOutGm;
  AscendC::GlobalTensor<T> kOutGm;
  AscendC::GatherMaskParams gatherMask{1, 1, 8, 0};
  AscendC::DataCopyParams qkPrepareDataCopyParams;
  AscendC::DataCopyParams queryOutParams, keyOutParams, queryInParams, keyInParams;
  uint32_t blockId{0};
  uint32_t startOffset{0};
  uint32_t loop{0};
  uint32_t originQueryHiddenSize{0};
  uint32_t originKeyHiddenSize{0};
  uint32_t queryKeyCalHiddenSize{0};
  uint32_t queryKeyInHiddenSize{0};
  ApplyRotaryPosEmbV3TilingData *tilingData = nullptr;
};

extern "C" __global__ __aicore__ void apply_rotary_pos_emb_v3(GM_ADDR query, GM_ADDR key, GM_ADDR cos, GM_ADDR sin,
                                                              GM_ADDR outq, GM_ADDR outk, GM_ADDR workspace,
                                                              GM_ADDR tiling) {
  GET_TILING_DATA(tilingData, tiling);
  GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
  AscendC::TPipe pipe;
  if (TILING_KEY_IS(20)) {
    KernelApplyRotaryPosEmbV3<float, false> op;
    op.Init(query, key, cos, sin, outq, outk, workspace, &tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(10)) {
    KernelApplyRotaryPosEmbV3<half, false> op;
    op.Init(query, key, cos, sin, outq, outk, workspace, &tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(21)) {
    KernelApplyRotaryPosEmbV3<float, true> op;
    op.Init(query, key, cos, sin, outq, outk, workspace, &tilingData, &pipe);
    op.Process();
  } else if (TILING_KEY_IS(11)) {
    KernelApplyRotaryPosEmbV3<half, true> op;
    op.Init(query, key, cos, sin, outq, outk, workspace, &tilingData, &pipe);
    op.Process();
  }
}

#ifndef ASCENDC_CPU_DEBUG
void apply_rotary_pos_emb_v3_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *query, uint8_t *key,
                                uint8_t *cos, uint8_t *sin, uint8_t *outq, uint8_t *outk, uint8_t *workspace,
                                uint8_t *tiling) {
  apply_rotary_pos_emb_v3<<<blockDim, l2ctrl, stream>>>(query, key, cos, sin, outq, outk, workspace, tiling);
}
#endif
