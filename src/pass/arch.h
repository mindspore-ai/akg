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

#ifndef PASS_ARCH_H_
#define PASS_ARCH_H_

#include <cstdint>

#include "lang/cgtypes.h"

namespace Arch {
enum BurstLengthByte {
  BURST_LENGTH_NONE = 0,
  BURST_LENGTH_16 = 16,
  BURST_LENGTH_32 = 32,
  BURST_LENGTH_64 = 64,
  BURST_LENGTH_128 = 128,
  BURST_LENGTH_256 = 256,
  BURST_LENGTH_512 = 512,
  BURST_LENGTH_1024 = 1024,
};

enum GapUnitByte {
  GAP_NONE = 0,
  OTHER = 32,
  L0C8 = 256,
  L0C16 = 512,
  L0C32 = 1024,
  L0C16V = L0C16,
  L0C32V = L0C32,
  L0CSC16 = 128,
  L0CSC32 = 256,
  SRC_L0CSC16 = 128,
  DST_L0CSC16 = 512,
};

class Vector {
 public:
  static const uint8_t MAX_REPEAT_TIMES = 255;
  static const uint8_t BLOCK_SIZE = 32;
  static const uint8_t BLOCKS_PER_REPEAT = 8;
  static const uint8_t BYTES_PRODUCED_PER_VCMP = 16;
  static const uint16_t MAX_VEC_VEC_BLOCK_STRIDE = 255;
  static const uint16_t MAX_VEC_SCALAR_BLOCK_STRIDE = 65535;
  static const uint16_t TRANSPOSE_BLOCK_SIZE = 512;  // in bytes
  static const uint16_t TRANSPOSE_ROW_SIZE = 16;
  static const uint16_t MASK_LEN_IN_BITS = 128;
  static const uint16_t BITWISE_DATATYPE_LEN = 2;
};

class Matrix {
 public:
  // Block size for 1-byte and 2-byte scalars
  static const uint16_t BLOCK_SIZE_16 = 512;
  // Block size for 4-byte scalars
  static const uint16_t BLOCK_SIZE_32 = 1024;
  // Matrix shape limitation
  static const uint16_t MATRIX_MAX_M = (1U << 12) - 1;
  static const uint16_t MATRIX_MAX_K = (1U << 12) - 1;
  static const uint16_t MATRIX_MAX_N = (1U << 12) - 1;
};

struct CopyIntrinsicSpec {
  constexpr CopyIntrinsicSpec(int64_t maxBurstCount_, int64_t maxBurstSize_, int64_t maxSrcBurstGapOrStride_,
                              int64_t maxDstBurstGapOrStride_)
      : maxBurstCount(maxBurstCount_),
        maxBurstSize(maxBurstSize_),
        maxSrcBurstGapOrStride(maxSrcBurstGapOrStride_),
        maxDstBurstGapOrStride(maxDstBurstGapOrStride_) {}
  ~CopyIntrinsicSpec() = default;
  int64_t maxBurstCount;
  int64_t maxBurstSize;
  int64_t maxSrcBurstGapOrStride;
  int64_t maxDstBurstGapOrStride;
};

class CopyOpLimit {
 public:
  static const int64_t COPY_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPY_MAX_BURSTSIZE = Vector::BLOCK_SIZE * ((1U << 16) - 1);
  static const int64_t COPY_MAX_SRCBURSTGAP = Vector::BLOCK_SIZE * ((1U << 16) - 1);
  static const int64_t COPY_MAX_DSTBURSTGAP = Vector::BLOCK_SIZE * ((1U << 16) - 1);

  static const int64_t COPYMATRIX_8BIT_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPYMATRIX_8BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_8BIT_MAX_SRCBURSTGAP = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_8BIT_MAX_DSTBURSTGAP = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);

  static const int64_t COPYMATRIX_16BIT_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPYMATRIX_16BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_16BIT_MAX_SRCBURSTGAP = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_16BIT_MAX_DSTBURSTGAP = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);

  static const int64_t COPYMATRIX_32BIT_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPYMATRIX_32BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_32 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_32BIT_MAX_SRCBURSTGAP = Matrix::BLOCK_SIZE_32 * ((1U << 16) - 1);
  static const int64_t COPYMATRIX_32BIT_MAX_DSTBURSTGAP = Matrix::BLOCK_SIZE_32 * ((1U << 16) - 1);

  static const int64_t COPYVECTOR_16BIT_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPYVECTOR_16BIT_MAX_BURSTSIZE = Vector::BLOCK_SIZE * ((1U << 16) - 1);
  static const int64_t COPYVECTOR_16BIT_MAX_SRCBURSTGAP = Vector::BLOCK_SIZE * ((1U << 16) - 1);
  static const int64_t COPYVECTOR_16BIT_MAX_DSTBURSTGAP = Vector::BLOCK_SIZE * ((1U << 16) - 1);

  static const int64_t COPYVECTOR_32BIT_MAX_BURSTCOUNT = (1U << 12) - 1;
  static const int64_t COPYVECTOR_32BIT_MAX_BURSTSIZE = Vector::BLOCK_SIZE * 2 * ((1U << 16) - 1);
  static const int64_t COPYVECTOR_32BIT_MAX_SRCBURSTGAP = Vector::BLOCK_SIZE * 2 * ((1U << 16) - 1);
  static const int64_t COPYVECTOR_32BIT_MAX_DSTBURSTGAP = Vector::BLOCK_SIZE * 2 * ((1U << 16) - 1);

  static const int64_t LOAD_8BIT_MAX_BURSTCOUNT = (1U << 8) - 1;
  static const int64_t LOAD_8BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_16;
  static const int64_t LOAD_8BIT_MAX_SRCBURSTSTRIDE = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t LOAD_8BIT_MAX_DSTBURSTSTRIDE = Matrix::BLOCK_SIZE_16;

  static const int64_t LOAD_16BIT_MAX_BURSTCOUNT = (1U << 8) - 1;
  static const int64_t LOAD_16BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_16;
  static const int64_t LOAD_16BIT_MAX_SRCBURSTSTRIDE = Matrix::BLOCK_SIZE_16 * ((1U << 16) - 1);
  static const int64_t LOAD_16BIT_MAX_DSTBURSTSTRIDE = Matrix::BLOCK_SIZE_16;

  static const int64_t LOAD_32BIT_MAX_BURSTCOUNT = (1U << 8) - 1;
  static const int64_t LOAD_32BIT_MAX_BURSTSIZE = Matrix::BLOCK_SIZE_32;
  static const int64_t LOAD_32BIT_MAX_SRCBURSTSTRIDE = Matrix::BLOCK_SIZE_32 * ((1U << 16) - 1);
  static const int64_t LOAD_32BIT_MAX_DSTBURSTSTRIDE = Matrix::BLOCK_SIZE_32;

  static const int64_t VTRANSPOSE_MAX_BURSTCOUNT = 1;
  static const int64_t VTRANSPOSE_MAX_BURSTSIZE = Vector::TRANSPOSE_BLOCK_SIZE;
  static const int64_t VTRANSPOSE_MAX_SRCBURSTSTRIDE = 0;
  static const int64_t VTRANSPOSE_MAX_DSTBURSTSTRIDE = 0;

  static constexpr CopyIntrinsicSpec copyInstrSpec = {COPY_MAX_BURSTCOUNT, COPY_MAX_BURSTSIZE, COPY_MAX_SRCBURSTGAP,
                                                      COPY_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec copyMatrix8BitInstrSpec = {
    COPYMATRIX_8BIT_MAX_BURSTCOUNT, COPYMATRIX_8BIT_MAX_BURSTSIZE, COPYMATRIX_8BIT_MAX_SRCBURSTGAP,
    COPYMATRIX_8BIT_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec copyMatrix16BitInstrSpec = {
    COPYMATRIX_16BIT_MAX_BURSTCOUNT, COPYMATRIX_16BIT_MAX_BURSTSIZE, COPYMATRIX_16BIT_MAX_SRCBURSTGAP,
    COPYMATRIX_16BIT_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec copyMatrix32BitInstrSpec = {
    COPYMATRIX_32BIT_MAX_BURSTCOUNT, COPYMATRIX_32BIT_MAX_BURSTSIZE, COPYMATRIX_32BIT_MAX_SRCBURSTGAP,
    COPYMATRIX_32BIT_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec copyVector16BitInstrSpec = {
    COPYVECTOR_16BIT_MAX_BURSTCOUNT, COPYVECTOR_16BIT_MAX_BURSTSIZE, COPYVECTOR_16BIT_MAX_SRCBURSTGAP,
    COPYVECTOR_16BIT_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec copyVector32BitInstrSpec = {
    COPYVECTOR_32BIT_MAX_BURSTCOUNT, COPYVECTOR_32BIT_MAX_BURSTSIZE, COPYVECTOR_32BIT_MAX_SRCBURSTGAP,
    COPYVECTOR_32BIT_MAX_DSTBURSTGAP};
  static constexpr CopyIntrinsicSpec load8BitInstrSpec = {LOAD_8BIT_MAX_BURSTCOUNT, LOAD_8BIT_MAX_BURSTSIZE,
                                                          LOAD_8BIT_MAX_SRCBURSTSTRIDE, LOAD_8BIT_MAX_DSTBURSTSTRIDE};
  static constexpr CopyIntrinsicSpec load16BitInstrSpec = {
    LOAD_16BIT_MAX_BURSTCOUNT, LOAD_16BIT_MAX_BURSTSIZE, LOAD_16BIT_MAX_SRCBURSTSTRIDE, LOAD_16BIT_MAX_DSTBURSTSTRIDE};
  static constexpr CopyIntrinsicSpec load32BitInstrSpec = {
    LOAD_32BIT_MAX_BURSTCOUNT, LOAD_32BIT_MAX_BURSTSIZE, LOAD_32BIT_MAX_SRCBURSTSTRIDE, LOAD_32BIT_MAX_DSTBURSTSTRIDE};
  static constexpr CopyIntrinsicSpec vtransposeInstrSpec = {
    VTRANSPOSE_MAX_BURSTCOUNT, VTRANSPOSE_MAX_BURSTSIZE, VTRANSPOSE_MAX_SRCBURSTSTRIDE, VTRANSPOSE_MAX_DSTBURSTSTRIDE};
  static constexpr CopyIntrinsicSpec broadcastUB2CCInstrSpec = {
    16, LOAD_16BIT_MAX_BURSTSIZE, LOAD_16BIT_MAX_SRCBURSTSTRIDE, LOAD_16BIT_MAX_DSTBURSTSTRIDE};
};

class BRCOpLimit {
 public:
  static const int64_t BRC_INTER_BROADCAST_NUM = 16;
  static const int64_t BRC_STANDARD_TRANSER_NUM = 16;
  static const int64_t BRC_16BIT_STANDARD_SRC_BURST_SIZE = BRC_STANDARD_TRANSER_NUM * 2;
  static const int64_t BRC_32BIT_STANDARD_SRC_BURST_SIZE = BRC_STANDARD_TRANSER_NUM * 4;
  static const int64_t BRC_16BIT_STANDARD_DST_BURST_SIZE = BRC_16BIT_STANDARD_SRC_BURST_SIZE * BRC_INTER_BROADCAST_NUM;
  static const int64_t BRC_32BIT_STANDARD_DST_BURST_SIZE = BRC_32BIT_STANDARD_SRC_BURST_SIZE * BRC_INTER_BROADCAST_NUM;
  static const int64_t BRC_MAX_BROADCAST_COUNT = (1U << 8) - 1;
  static const int64_t BRC_MAX_TRANSFER_COUNT = (1U << 8) - 1;
  static const int64_t BRC_16BIT_STANDARD_MAX_SRC_TRANSFER_GAP = BRC_16BIT_STANDARD_SRC_BURST_SIZE * ((1U << 8) - 1);
  static const int64_t BRC_32BIT_STANDARD_MAX_SRC_TRANSFER_GAP = BRC_32BIT_STANDARD_SRC_BURST_SIZE * ((1U << 8) - 1);
  static const int64_t BRC_16BIT_STANDARD_MAX_DST_TRANSFER_GAP = BRC_16BIT_STANDARD_DST_BURST_SIZE * ((1U << 8) - 1);
  static const int64_t BRC_32BIT_STANDARD_MAX_DST_TRANSFER_GAP = BRC_32BIT_STANDARD_DST_BURST_SIZE * ((1U << 8) - 1);
};

struct VecIntrinsicSpec {
  constexpr VecIntrinsicSpec(int64_t maxBlockSize_, int64_t maxBlockCnt_, int64_t maxRepeatCnt_,
                             int64_t maxBlockStride_, int64_t maxRepeatStride_)
      : maxBlockSize(maxBlockSize_),
        maxBlockCnt(maxBlockCnt_),
        maxRepeatCnt(maxRepeatCnt_),
        maxBlockStride(maxBlockStride_),
        maxRepeatStride(maxRepeatStride_) {}
  ~VecIntrinsicSpec() = default;
  int64_t maxBlockSize;
  int64_t maxBlockCnt;
  int64_t maxRepeatCnt;
  int64_t maxBlockStride;
  int64_t maxRepeatStride;
};

class VecOpLimit {
 public:
  static const int64_t MAX_BLOCK_SIZE = 32;  // byte
  static const int64_t MAX_BLOCK_CNT = 8;

  static const int64_t MAX_TWO_OPERANDS_REPEAT_CNT = (1U << 8) - 1;
  static const int64_t MAX_TWO_OPERANDS_BLOCK_STRIDE = ((1U << 15) - 1) * MAX_BLOCK_SIZE;  // byte
  static const int64_t MAX_TWO_OPERANDS_REPEAT_STRIDE = ((1U << 8) - 1) * MAX_BLOCK_SIZE;  // byte

  static const int64_t MAX_THREE_OPERANDS_REPEAT_CNT = (1U << 8) - 1;
  static const int64_t MAX_THREE_OPERANDS_BLOCK_STRIDE = ((1U << 8) - 1) * MAX_BLOCK_SIZE;   // byte
  static const int64_t MAX_THREE_OPERANDS_REPEAT_STRIDE = ((1U << 8) - 1) * MAX_BLOCK_SIZE;  // byte

  static constexpr VecIntrinsicSpec singleSrcVecInstrSpec = {MAX_BLOCK_SIZE, MAX_BLOCK_CNT, MAX_TWO_OPERANDS_REPEAT_CNT,
                                                             MAX_TWO_OPERANDS_BLOCK_STRIDE,
                                                             MAX_TWO_OPERANDS_REPEAT_STRIDE};
  static constexpr VecIntrinsicSpec twoSrcsVecInstrSpec = {MAX_BLOCK_SIZE, MAX_BLOCK_CNT, MAX_THREE_OPERANDS_REPEAT_CNT,
                                                           MAX_THREE_OPERANDS_BLOCK_STRIDE,
                                                           MAX_THREE_OPERANDS_REPEAT_STRIDE};
};
}  // namespace Arch

#endif  // PASS_ARCH_H_
