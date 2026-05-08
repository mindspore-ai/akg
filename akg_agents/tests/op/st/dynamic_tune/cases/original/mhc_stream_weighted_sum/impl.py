from __future__ import annotations

try:
    from akg_agents.op.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

try:
    import torch_npu  # type: ignore
except ImportError:
    torch_npu = None  # type: ignore[assignment]

RATE = 4
HIDDEN_SIZE = 8192
DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_BLOCK_H = 4096


@triton.jit
def akg_agents_fused_hyper_connection_stream_weighted_sum_kernel(
    h_pre_ptr, original_streams_ptr, output_ptr,
    S, B, H,
    stride_h_s, stride_h_b, stride_h_r,
    stride_x_s, stride_x_b,
    stride_o_s, stride_o_b, stride_o_h,
    BLOCK_H: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    total_tasks = S * B
    for task_idx in range(pid, total_tasks, CORE_NUM):
        s = task_idx // B
        b = task_idx % B
        h_base = h_pre_ptr + s * stride_h_s + b * stride_h_b
        w0 = tl.load(h_base + 0 * stride_h_r).to(tl.float32)
        w1 = tl.load(h_base + 1 * stride_h_r).to(tl.float32)
        w2 = tl.load(h_base + 2 * stride_h_r).to(tl.float32)
        w3 = tl.load(h_base + 3 * stride_h_r).to(tl.float32)
        x_base = original_streams_ptr + s * stride_x_s + b * stride_x_b
        o_base = output_ptr + s * stride_o_s + b * stride_o_b
        for h_start in range(0, H, BLOCK_H):
            h_offsets = h_start + tl.arange(0, BLOCK_H)
            mask = h_offsets < H
            b0 = tl.load(x_base + 0 * H + h_offsets, mask=mask, other=0.0).to(tl.float32)
            b1 = tl.load(x_base + 1 * H + h_offsets, mask=mask, other=0.0).to(tl.float32)
            b2 = tl.load(x_base + 2 * H + h_offsets, mask=mask, other=0.0).to(tl.float32)
            b3 = tl.load(x_base + 3 * H + h_offsets, mask=mask, other=0.0).to(tl.float32)
            res = w0 * b0 + w1 * b1 + w2 * b2 + w3 * b3
            tl.store(o_base + h_offsets * stride_o_h, res.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, rate=RATE, hidden_size=HIDDEN_SIZE, input_dtype=DEFAULT_INPUT_DTYPE):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype
        try:
            self.VEC_CORE_NUM = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)  # type: ignore[union-attr]
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, h_pre, original_streams):
        S, B, _, R = h_pre.shape
        _, _, packed_hidden = original_streams.shape
        assert packed_hidden == self.rate * self.hidden_size
        assert R == self.rate
        h_pre = h_pre.contiguous()
        original_streams = original_streams.contiguous()
        chunk_seq = min(int(original_streams.shape[0]), int(S))
        core_num = max(1, int(self.VEC_CORE_NUM))
        block_h = max(1024, int(DEFAULT_BLOCK_H))
        output = torch.empty((S, B, self.hidden_size), dtype=self.input_dtype, device=h_pre.device)
        for start in range(0, S, chunk_seq):
            end = min(start + chunk_seq, S)
            cur_s = end - start
            h_chunk = h_pre[start:end].contiguous()
            x_chunk = original_streams[start:end].contiguous()
            out_chunk = output[start:end].contiguous()
            akg_agents_fused_hyper_connection_stream_weighted_sum_kernel[(core_num,)](
                h_chunk, x_chunk, out_chunk,
                cur_s, B, self.hidden_size,
                h_chunk.stride(0), h_chunk.stride(1), h_chunk.stride(3),
                x_chunk.stride(0), x_chunk.stride(1),
                out_chunk.stride(0), out_chunk.stride(1), out_chunk.stride(2),
                BLOCK_H=block_h, CORE_NUM=core_num,
            )
        return output
