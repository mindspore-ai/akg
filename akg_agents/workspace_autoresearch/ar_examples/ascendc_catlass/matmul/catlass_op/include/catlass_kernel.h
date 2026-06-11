/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_OP_INCLUDE_CATLASS_KERNEL_H_
#define CATLASS_OP_INCLUDE_CATLASS_KERNEL_H_
#include <cstddef>
#include <cstdint>

namespace catlass_kernel {

void basic_matmul(uint32_t coreNum, void* stream, uint32_t M, uint32_t N, uint32_t K,
                  uint8_t* ptrA, uint8_t* ptrB, uint8_t* ptrC);

size_t get_splitk_matmul_workspace_size(uint32_t coreNum, uint32_t M, uint32_t N, uint32_t K);
void splitk_matmul(uint32_t coreNum, void* stream, uint32_t M, uint32_t N, uint32_t K,
                   uint8_t* ptrA, uint8_t* ptrB, uint8_t* ptrC, uint8_t* workspace);

size_t get_optimized_matmul_workspace_size(uint32_t M, uint32_t N, uint32_t K);
void optimized_matmul(uint32_t coreNum, void* stream, uint32_t M, uint32_t N, uint32_t K,
                      uint8_t* ptrA, uint8_t* ptrB, uint8_t* ptrC, uint8_t* workspace);

} // namespace catlass_kernel

#endif // CATLASS_OP_INCLUDE_CATLASS_KERNEL_H_
