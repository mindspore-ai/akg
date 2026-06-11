/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATLASS_OP_INCLUDE_CATLASS_H_
#define CATLASS_OP_INCLUDE_CATLASS_H_

#include <torch/extension.h>

namespace catlass_torch {

at::Tensor basic_matmul(const at::Tensor& A, const at::Tensor& B);
at::Tensor optimized_matmul(const at::Tensor& A, const at::Tensor& B);
at::Tensor splitk_matmul(const at::Tensor& A, const at::Tensor& B);

}

#endif // CATLASS_OP_INCLUDE_CATLASS_H_
