# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pypto
import torch

_PYPTO_RUN_MODE = int(os.getenv("AIKG_PYPTO_RUN_MODE", "0"))
_PYPTO_RUNTIME_DEBUG_MODE = int(os.getenv("AIKG_PYPTO_RUNTIME_DEBUG_MODE", "0"))


def create_relu_kernel(numel: int):
    @pypto.frontend.jit(
        runtime_options={"run_mode": _PYPTO_RUN_MODE},
        debug_options={"runtime_debug_mode": _PYPTO_RUNTIME_DEBUG_MODE},
    )
    def relu_kernel_npu(
        x: pypto.Tensor((numel,), pypto.DT_FP32),
    ) -> pypto.Tensor((numel,), pypto.DT_FP32):
        output = pypto.tensor([numel], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        output[:] = pypto.maximum(x, 0.0)
        return output

    return relu_kernel_npu


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_contiguous = x.contiguous()
        original_dtype = x_contiguous.dtype

        x_fp32 = x_contiguous.to(torch.float32)
        x_flat = x_fp32.reshape(-1).contiguous()
        out_flat = create_relu_kernel(x_flat.numel())(x_flat)
        out = out_flat.reshape_as(x_fp32)
        return out.to(original_dtype)
