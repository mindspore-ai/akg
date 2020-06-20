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

#ifndef EMIT_INSN_CCE_PARAMS_H_
#define EMIT_INSN_CCE_PARAMS_H_

#include <tvm/base.h>
#include <tvm/ir_pass.h>
#include "ir_pass.h"
/// define the buffer var
#define SCOPE_CBUF "local.L1"
#define SCOPE_UBUF "local.UB"
#define SCOPE_CA "local.L0A"
#define SCOPE_CB "local.L0B"
#define SCOPE_CC "local.L0C"
#define SCOPE_REG "local.REG"
#define SCOPE_AICPU "local_aicpu"

#define DMA_COPY "dma_copy"
#define DMA_COPY_GLOBAL "global"

#define STORAGE_SCOPE "storage_scope"

/// define intrin name
#define INTRIN_NAME_VECTOR_DUP "vector_dup"
#define INTRIN_NAME_SET_VEC_MASK "set_vector_mask"
#define INTRIN_NAME_COPY_UB_TO_UB "copy_ubuf_to_ubuf"
#define INTRIN_NAME_COPY_UB_TO_GM "copy_ubuf_to_gm"
#define INTRIN_NAME_COPY_GM_TO_UB "copy_gm_to_ubuf"
#define INTRIN_NAME_REG_MOV "reg_mov"

#define WGT_WIDTH 16
#define INP_WIDTH 16
#define OUT_WIDTH 16
#define BLOCK_IN 16
#define BLOCK_OUT 16
#define BLOCK_REDUCE 16

#define INP_ELEM_BYTES (BLOCK_IN * BLOCK_REDUCE * INP_WIDTH / 8)
#define WGT_ELEM_BYTES (BLOCK_OUT * BLOCK_REDUCE * WGT_WIDTH / 8)
#define OUT_ELEM_BYTES (BLOCK_IN * BLOCK_OUT * OUT_WIDTH / 8)
#define GLB_ELEM_BYTES (16 * OUT_WIDTH / 8)

#define BITS_PER_BYTE 8

#define FREE_ALIGN (-2)

/// the maximum value of intrinsic parameters mentioned in "Davinci ISA User Guide"
constexpr int MAX_SID = 16;                  // 1 << 4
constexpr int MAX_NBURST = 4096;             // 1 << 12
constexpr int MAX_LENBURST = 65536;          // 1 << 16
constexpr int MAX_REPEAT = 256;              // 1 << 8
constexpr int MAX_STRIDE_M0 = 256;           // 1 << 8
constexpr int MAX_STRIDE_M0_SINGLE = 65536;  // 1 << 16
constexpr int MAX_STRIDE_M1 = 256;           // 1 << 8
constexpr int MAX_STRIDE = 65536;            // 1 << 16
constexpr int FULL_BLOCK_NUM = 8;

#endif  // EMIT_INSN_CCE_PARAMS_H_
