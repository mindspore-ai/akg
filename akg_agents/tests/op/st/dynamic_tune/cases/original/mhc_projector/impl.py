from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore
import triton  # type: ignore
import triton.language as tl  # type: ignore

RATE = 4
HIDDEN_SIZE = 3584

DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_INIT_GATING_FACTOR = 0.01
DEFAULT_EXPAND_POST = 2.0
DEFAULT_WEIGHT_SEED = 2026

DEFAULT_NORM_BLOCK_SIZE = 8192
DEFAULT_POST_BLOCK_SIZE = 32


@triton.jit
def normalization_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    batch_size,
    hidden_size,
    stride_s,
    stride_b,
    stride_h,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    total_positions = seq_len * batch_size
    for position_idx in range(pid, total_positions, CORE_NUM):
        s = position_idx // batch_size
        b = position_idx % batch_size
        row_start = s * stride_s + b * stride_b
        sum_squares = tl.zeros((), dtype=tl.float32)
        for offset in range(0, hidden_size, BLOCK_SIZE):
            offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hidden_size
            x = tl.load(input_ptr + row_start + offsets * stride_h, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)
            sum_squares += tl.sum(x_f32 * x_f32, axis=0)
        mean_square = sum_squares / hidden_size
        rsqrt_val = 1.0 / tl.sqrt(mean_square + eps)
        for offset in range(0, hidden_size, BLOCK_SIZE):
            offsets = offset + tl.arange(0, BLOCK_SIZE)
            mask = offsets < hidden_size
            x = tl.load(input_ptr + row_start + offsets * stride_h, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)
            tl.store(output_ptr + row_start + offsets * stride_h, x_f32 * rsqrt_val, mask=mask)


@triton.jit
def post_process_kernel(
    projected_ptr,
    h_pre_ptr,
    h_post_ptr,
    h_res_logits_ptr,
    seq_len,
    batch_size,
    rate: tl.constexpr,
    alpha_pre: tl.float32,
    alpha_post: tl.float32,
    alpha_res: tl.float32,
    expand_post: tl.float32,
    bias_pre_ptr,
    bias_post_ptr,
    bias_res_ptr,
    BLOCK_SIZE: tl.constexpr,
    CORE_NUM: tl.constexpr,
):
    pid = tl.program_id(0)
    total_positions = seq_len * batch_size
    dim = rate + rate + rate * rate
    for position_idx in range(pid, total_positions, CORE_NUM):
        s = position_idx // batch_size
        b = position_idx % batch_size
        base_offset = (s * batch_size + b) * dim
        h_pre_raw = tl.load(projected_ptr + base_offset + tl.arange(0, rate))
        h_post_raw = tl.load(projected_ptr + base_offset + rate + tl.arange(0, rate))
        h_res_raw = tl.load(projected_ptr + base_offset + 2 * rate + tl.arange(0, rate * rate))
        bias_pre = tl.load(bias_pre_ptr + tl.arange(0, rate))
        bias_post = tl.load(bias_post_ptr + tl.arange(0, rate))
        bias_res = tl.load(bias_res_ptr + tl.arange(0, rate * rate))
        h_pre = tl.sigmoid(alpha_pre * h_pre_raw + bias_pre)
        h_post = expand_post * tl.sigmoid(alpha_post * h_post_raw + bias_post)
        h_res_logits = alpha_res * h_res_raw + bias_res
        h_pre_out_offset = (s * batch_size + b) * rate
        tl.store(h_pre_ptr + h_pre_out_offset + tl.arange(0, rate), h_pre)
        tl.store(h_post_ptr + h_pre_out_offset + tl.arange(0, rate), h_post)
        h_res_out_offset = (s * batch_size + b) * rate * rate
        tl.store(h_res_logits_ptr + h_res_out_offset + tl.arange(0, rate * rate), h_res_logits)


def _resolve_vec_core_num():
    try:
        import torch_npu  # type: ignore
        value = torch_npu.npu.npu_config.get_device_limit(0).get("vector_core_num", 40)
        return int(value)
    except Exception:
        return 40


def _clamp_chunk_seq(chunk_seq, seq_len):
    chunk_seq = int(chunk_seq)
    if chunk_seq <= 0:
        raise ValueError(f"chunk_seq 必须 > 0，得到 {chunk_seq}")
    return min(seq_len, chunk_seq)


class ModelNew(nn.Module):
    def __init__(
        self,
        rate=RATE,
        hidden_size=HIDDEN_SIZE,
        init_gating_factor=DEFAULT_INIT_GATING_FACTOR,
        expand_post=DEFAULT_EXPAND_POST,
        input_dtype=DEFAULT_INPUT_DTYPE,
        seed=DEFAULT_WEIGHT_SEED,
    ):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.expand_post = expand_post
        self.input_dtype = input_dtype
        self.eps = 1e-6

        torch.manual_seed(seed)
        dim = rate + rate + rate * rate
        mapping_weight = torch.randn(rate * hidden_size, dim, dtype=torch.float32) * 1e-4
        self.register_buffer("mapping_weight", mapping_weight)
        alpha = torch.full((1, 1, 1, 1), init_gating_factor, dtype=torch.float32)
        self.register_buffer("alpha_pre", alpha.clone())
        self.register_buffer("alpha_post", alpha.clone())
        self.register_buffer("alpha_res", alpha.clone())
        pre_bias = torch.full((1, 1, 1, rate), -torch.log(torch.tensor(3.0)), dtype=torch.float32)
        post_bias = torch.zeros((1, 1, 1, rate), dtype=torch.float32)
        res_bias = ((torch.eye(rate, dtype=torch.float32) - 1.0) * 5.0).reshape(1, 1, 1, rate * rate)
        self.register_buffer("bias_pre", pre_bias)
        self.register_buffer("bias_post", post_bias)
        self.register_buffer("bias_res", res_bias)
        self.default_core_num = _resolve_vec_core_num()
        self.default_norm_block_size = DEFAULT_NORM_BLOCK_SIZE
        self.default_post_block_size = DEFAULT_POST_BLOCK_SIZE

    def forward(self, hidden_states):
        seq_len, batch_size, packed_hidden_size = hidden_states.shape
        assert packed_hidden_size == self.rate * self.hidden_size
        hidden_states = hidden_states.contiguous()
        chunk_seq = _clamp_chunk_seq(seq_len, seq_len)
        core_num = max(1, int(self.default_core_num))
        norm_block_size = max(1024, int(self.default_norm_block_size))
        post_block_size = max(16, int(self.default_post_block_size))
        h_pre = torch.empty((seq_len, batch_size, self.rate), dtype=torch.float32, device=hidden_states.device)
        h_post = torch.empty((seq_len, batch_size, self.rate), dtype=torch.float32, device=hidden_states.device)
        h_res_logits = torch.empty((seq_len, batch_size, self.rate * self.rate), dtype=torch.float32, device=hidden_states.device)
        alpha_pre = float(self.alpha_pre[0, 0, 0, 0].item())
        alpha_post = float(self.alpha_post[0, 0, 0, 0].item())
        alpha_res = float(self.alpha_res[0, 0, 0, 0].item())
        bias_pre = self.bias_pre.view(-1)
        bias_post = self.bias_post.view(-1)
        bias_res = self.bias_res.view(-1)
        for start in range(0, seq_len, chunk_seq):
            end = min(start + chunk_seq, seq_len)
            hidden_chunk = hidden_states[start:end].contiguous()
            cur_seq = end - start
            normalized = torch.empty_like(hidden_chunk, dtype=torch.float32)
            normalization_kernel[(core_num,)](
                hidden_chunk, normalized, cur_seq, batch_size, packed_hidden_size,
                normalized.stride(0), normalized.stride(1), normalized.stride(2),
                eps=self.eps, BLOCK_SIZE=norm_block_size, CORE_NUM=core_num,
            )
            projected = torch.matmul(normalized, self.mapping_weight)
            post_process_kernel[(core_num,)](
                projected, h_pre[start:end], h_post[start:end], h_res_logits[start:end],
                cur_seq, batch_size, self.rate, alpha_pre, alpha_post, alpha_res, float(self.expand_post),
                bias_pre, bias_post, bias_res, BLOCK_SIZE=post_block_size, CORE_NUM=core_num,
            )
        return h_pre, h_post, h_res_logits
