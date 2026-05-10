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
HIDDEN_SIZE = 3584
PACKED_HIDDEN_SIZE = RATE * HIDDEN_SIZE
DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_BLOCK_N = 2048


@triton.jit
def akg_agents_fused_hyper_connection_output_cell_kernel(
    h_res_ptr, h_post_ptr, original_streams_ptr, sublayer_out_ptr, output_ptr,
    S, B, R: tl.constexpr, H: tl.constexpr,
    stride_h_res_s, stride_h_res_b, stride_h_res_r, stride_h_res_k,
    stride_h_post_s, stride_h_post_b, stride_h_post_r,
    stride_orig_s, stride_orig_b, stride_orig_r, stride_orig_h,
    stride_sub_s, stride_sub_b, stride_sub_h,
    stride_out_s, stride_out_b, stride_out_r, stride_out_h,
    BLOCK_N: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    total_sb = S * B
    for sb_idx in range(pid, total_sb, CORE_NUM):
        s_idx = sb_idx // B
        b_idx = sb_idx % B
        h_res_base = h_res_ptr + s_idx * stride_h_res_s + b_idx * stride_h_res_b
        h_post_base = h_post_ptr + s_idx * stride_h_post_s + b_idx * stride_h_post_b
        offs_r = tl.arange(0, R)
        h_res_0 = tl.load(h_res_base + offs_r * stride_h_res_r + 0 * stride_h_res_k).to(tl.float32)
        h_res_1 = tl.load(h_res_base + offs_r * stride_h_res_r + 1 * stride_h_res_k).to(tl.float32)
        h_res_2 = tl.load(h_res_base + offs_r * stride_h_res_r + 2 * stride_h_res_k).to(tl.float32)
        h_res_3 = tl.load(h_res_base + offs_r * stride_h_res_r + 3 * stride_h_res_k).to(tl.float32)
        h_post = tl.load(h_post_base + offs_r * stride_h_post_r).to(tl.float32)
        orig_base = original_streams_ptr + s_idx * stride_orig_s + b_idx * stride_orig_b
        sub_base = sublayer_out_ptr + s_idx * stride_sub_s + b_idx * stride_sub_b
        out_base = output_ptr + s_idx * stride_out_s + b_idx * stride_out_b
        for n_start in range(0, H, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            mask_n = offs_n < H
            sub_offsets = sub_base + offs_n * stride_sub_h
            sub_slice = tl.load(sub_offsets, mask=mask_n, other=0.0).to(tl.float32)
            x_0 = tl.load(orig_base + 0 * stride_orig_r + offs_n * stride_orig_h, mask=mask_n, other=0.0).to(tl.float32)
            x_1 = tl.load(orig_base + 1 * stride_orig_r + offs_n * stride_orig_h, mask=mask_n, other=0.0).to(tl.float32)
            x_2 = tl.load(orig_base + 2 * stride_orig_r + offs_n * stride_orig_h, mask=mask_n, other=0.0).to(tl.float32)
            x_3 = tl.load(orig_base + 3 * stride_orig_r + offs_n * stride_orig_h, mask=mask_n, other=0.0).to(tl.float32)
            residual = h_res_0[:, None] * x_0[None, :]
            residual += h_res_1[:, None] * x_1[None, :]
            residual += h_res_2[:, None] * x_2[None, :]
            residual += h_res_3[:, None] * x_3[None, :]
            post = h_post[:, None] * sub_slice[None, :]
            result = residual + post
            out_offsets = out_base + offs_r[:, None] * stride_out_r + offs_n[None, :] * stride_out_h
            tl.store(out_offsets, result.to(tl.bfloat16), mask=mask_n[None, :])


class ModelNew(nn.Module):
    def __init__(self, rate=RATE, hidden_size=HIDDEN_SIZE, input_dtype=DEFAULT_INPUT_DTYPE):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype
        try:
            import torch
            import triton
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.VEC_CORE_NUM = properties.get("num_vectorcore", 40)
        except Exception:
            self.VEC_CORE_NUM = 40

    def forward(self, h_res, h_post, original_streams, sublayer_out):
        S, B, R, _ = h_res.shape
        _, _, packed_hidden = original_streams.shape
        _, _, H = sublayer_out.shape
        assert packed_hidden == self.rate * self.hidden_size
        assert R == self.rate
        assert H == self.hidden_size
        h_res = h_res.contiguous()
        h_post = h_post.contiguous()
        original_streams = original_streams.contiguous()
        sublayer_out = sublayer_out.contiguous()
        chunk_seq = min(int(original_streams.shape[0]), int(S))
        core_num = max(1, int(self.VEC_CORE_NUM))
        block_n = max(256, int(DEFAULT_BLOCK_N))
        output = torch.empty_like(original_streams)
        for start in range(0, S, chunk_seq):
            end = min(start + chunk_seq, S)
            cur_s = end - start
            h_res_chunk = h_res[start:end].contiguous()
            h_post_chunk = h_post[start:end].contiguous()
            orig_chunk = original_streams[start:end].contiguous()
            sub_chunk = sublayer_out[start:end].contiguous()
            out_chunk = output[start:end].view(cur_s, B, self.rate, self.hidden_size)
            akg_agents_fused_hyper_connection_output_cell_kernel[(core_num,)](
                h_res_chunk, h_post_chunk, orig_chunk, sub_chunk, out_chunk,
                cur_s, B, R, H,
                h_res_chunk.stride(0), h_res_chunk.stride(1), h_res_chunk.stride(2), h_res_chunk.stride(3),
                h_post_chunk.stride(0), h_post_chunk.stride(1), h_post_chunk.stride(2),
                orig_chunk.stride(0), orig_chunk.stride(1), self.hidden_size, 1,
                sub_chunk.stride(0), sub_chunk.stride(1), sub_chunk.stride(2),
                out_chunk.stride(0), out_chunk.stride(1), out_chunk.stride(2), out_chunk.stride(3),
                BLOCK_N=block_n, CORE_NUM=core_num,
            )
        return output
