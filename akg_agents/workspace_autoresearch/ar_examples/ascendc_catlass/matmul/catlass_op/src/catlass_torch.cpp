/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <tiling/platform/platform_ascendc.h>
#include "catlass_kernel.h"

#define RUN_NPU_FUNC(func, ...)                                                              \
    do {                                                                                     \
        if ((func) == nullptr) {                                                             \
            throw std::runtime_error(                                                        \
                std::string("Function pointer is null at ") + __FILE__ + ":" +               \
                std::to_string(__LINE__) + " in " + #func);                                  \
        }                                                                                    \
        at_npu::native::OpCommand::RunOpApiV2(#func, [=]() -> int {                          \
            func(__VA_ARGS__);                                                               \
            return 0;                                                                        \
        });                                                                                  \
    } while (false)

namespace catlass_torch {

at::Tensor basic_matmul(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    auto M = static_cast<uint32_t>(A.size(0));
    auto K_A = static_cast<uint32_t>(A.size(1));
    auto K_B = static_cast<uint32_t>(B.size(0));
    auto N = static_cast<uint32_t>(B.size(1));
    TORCH_CHECK(K_A == K_B, "Inner dimensions must match, got ", K_A, " vs ", K_B);

    auto aContig = A.contiguous();
    auto bContig = B.contiguous();

    auto C = at::empty({M, N}, A.options());

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto stream = c10_npu::getCurrentNPUStream().stream();

    RUN_NPU_FUNC(catlass_kernel::basic_matmul, aicCoreNum, stream, M, N, K_A,
                 static_cast<uint8_t*>(aContig.data_ptr()),
                 static_cast<uint8_t*>(bContig.data_ptr()),
                 static_cast<uint8_t*>(C.data_ptr()));

    return C;
}

at::Tensor splitk_matmul(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    auto M = static_cast<uint32_t>(A.size(0));
    auto K_A = static_cast<uint32_t>(A.size(1));
    auto K_B = static_cast<uint32_t>(B.size(0));
    auto N = static_cast<uint32_t>(B.size(1));
    TORCH_CHECK(K_A == K_B, "Inner dimensions must match, got ", K_A, " vs ", K_B);

    auto aContig = A.contiguous();
    // Kernel uses ColumnMajor for B, so transpose row-major (K,N) to (N,K) and interpret as ColumnMajor(K,N)
    auto bTransposed = B.t().contiguous();

    auto C = at::empty({M, N}, A.options());

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto stream = c10_npu::getCurrentNPUStream().stream();

    size_t workspaceSize = catlass_kernel::get_splitk_matmul_workspace_size(aicCoreNum, M, N, K_A);
    auto workspace = at::empty(workspaceSize, at::dtype(at::kByte).device(aContig.device()));

    RUN_NPU_FUNC(catlass_kernel::splitk_matmul, aicCoreNum, stream, M, N, K_A,
                 static_cast<uint8_t*>(aContig.data_ptr()),
                 static_cast<uint8_t*>(bTransposed.data_ptr()),
                 static_cast<uint8_t*>(C.data_ptr()),
                 static_cast<uint8_t*>(workspace.data_ptr()));

    return C;
}

at::Tensor optimized_matmul(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.device().type() == c10::DeviceType::PrivateUse1, "A must be on NPU");
    TORCH_CHECK(B.device().type() == c10::DeviceType::PrivateUse1, "B must be on NPU");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");

    auto M = static_cast<uint32_t>(A.size(0));
    auto K_A = static_cast<uint32_t>(A.size(1));
    auto K_B = static_cast<uint32_t>(B.size(0));
    auto N = static_cast<uint32_t>(B.size(1));
    TORCH_CHECK(K_A == K_B, "Inner dimensions must match, got ", K_A, " vs ", K_B);

    auto aContig = A.contiguous();
    // Transpose B: kernel expects ColumnMajor(K,N), PyTorch tensor is RowMajor(K,N).
    // Transpose to RowMajor(N,K) which is reinterpretable as ColumnMajor(K,N).
    auto bTransposed = B.t().contiguous();

    auto C = at::empty({M, N}, A.options());

    auto aicCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    auto stream = c10_npu::getCurrentNPUStream().stream();

    size_t workspaceSize = catlass_kernel::get_optimized_matmul_workspace_size(M, N, K_A);
    auto workspace = at::empty(workspaceSize, at::dtype(at::kByte).device(aContig.device()));

    RUN_NPU_FUNC(catlass_kernel::optimized_matmul, aicCoreNum, stream, M, N, K_A,
                 static_cast<uint8_t*>(aContig.data_ptr()),
                 static_cast<uint8_t*>(bTransposed.data_ptr()),
                 static_cast<uint8_t*>(C.data_ptr()),
                 static_cast<uint8_t*>(workspace.data_ptr()));

    return C;
}

} // namespace catlass_torch

TORCH_LIBRARY(catlass, m) {
    m.def("basic_matmul(Tensor A, Tensor B) -> Tensor");
    m.def("optimized_matmul(Tensor A, Tensor B) -> Tensor");
    m.def("splitk_matmul(Tensor A, Tensor B) -> Tensor");
}

TORCH_LIBRARY_IMPL(catlass, PrivateUse1, m) {
    m.impl("basic_matmul", catlass_torch::basic_matmul);
    m.impl("optimized_matmul", catlass_torch::optimized_matmul);
    m.impl("splitk_matmul", catlass_torch::splitk_matmul);
}
