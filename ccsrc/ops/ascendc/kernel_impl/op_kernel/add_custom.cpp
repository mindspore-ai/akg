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
// tensor num for each queue
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y, typename TYPE_Z> class KernelAdd {
    using T = TYPE_X;
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum) 
    {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }
        xGm.SetGlobalBuffer((__gm__ TYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ TYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        zGm.SetGlobalBuffer((__gm__ TYPE_Z*)z + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileDataNum * sizeof(TYPE_Z));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.AllocTensor<TYPE_Y>();
      AscendC::DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
      AscendC::DataCopy(yLocal, yGm[progress * this->tileDataNum], this->processDataNum);
      inQueueX.EnQue(xLocal);
      inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
      AscendC::LocalTensor<TYPE_Y> yLocal = inQueueY.DeQue<TYPE_Y>();
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.AllocTensor<TYPE_Z>();
      if constexpr (std::is_same_v<T, int8_t>) {
        auto p1 = tmp1.Get<half>();
        auto p2 = tmp2.Get<half>();
        AscendC::Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(p2, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Add(p2, p1, p2, this->processDataNum);
        AscendC::Cast(p1.ReinterpretCast<int16_t>(), p2, AscendC::RoundMode::CAST_RINT, this->processDataNum);
        AscendC::ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum); 
        AscendC::ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum);
        AscendC::Cast(p2, p1.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
        AscendC::Cast(zLocal, p2, AscendC::RoundMode::CAST_NONE, this->processDataNum);
      }
      else {
        AscendC::Add(zLocal, xLocal, yLocal, this->processDataNum);
      }
      outQueueZ.EnQue<TYPE_Z>(zLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
      AscendC::LocalTensor<TYPE_Z> zLocal = outQueueZ.DeQue<TYPE_Z>();  
      AscendC::DataCopy(zGm[progress * this->tileDataNum], zLocal, this->processDataNum);
      outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_Y> yGm;
    AscendC::GlobalTensor<TYPE_Z> zGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);  
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void add_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* z,
    uint8_t* workspace, uint8_t* tiling)
{
    add_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, workspace, tiling);
}
#endif
