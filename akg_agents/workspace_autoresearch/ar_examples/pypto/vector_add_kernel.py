import os

import pypto
import torch


_PYPTO_RUN_MODE = int(os.getenv("AIKG_PYPTO_RUN_MODE", "0"))
_PYPTO_RUNTIME_DEBUG_MODE = int(os.getenv("AIKG_PYPTO_RUNTIME_DEBUG_MODE", "0"))


def create_vector_add_kernel(numel: int):
    @pypto.frontend.jit(
        runtime_options={"run_mode": _PYPTO_RUN_MODE},
        debug_options={"runtime_debug_mode": _PYPTO_RUNTIME_DEBUG_MODE},
    )
    def vector_add_kernel_npu(
        x: pypto.Tensor((numel,), pypto.DT_FP32),
        y: pypto.Tensor((numel,), pypto.DT_FP32),
    ) -> pypto.Tensor((numel,), pypto.DT_FP32):
        output = pypto.tensor([numel], pypto.DT_FP32)
        pypto.set_vec_tile_shapes(8192)
        output[:] = pypto.add(x, y)
        return output

    return vector_add_kernel_npu


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_contiguous = x.contiguous()
        y_contiguous = y.contiguous()
        original_dtype = x_contiguous.dtype

        x_fp32 = x_contiguous.to(torch.float32)
        y_fp32 = y_contiguous.to(torch.float32)
        x_flat = x_fp32.reshape(-1).contiguous()
        y_flat = y_fp32.reshape(-1).contiguous()

        out_flat = create_vector_add_kernel(x_flat.numel())(x_flat, y_flat)
        out = out_flat.reshape_as(x_fp32)
        return out.to(original_dtype)
