import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/model_executor/forward_batch_info.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   compute_position_triton(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)
# Triton Kernel:
#   compute_position_kernel - 计算批次中每个序列的位置索引
# ============================================================================


@triton.jit
def compute_position_kernel(
    positions,
    extend_start_loc,
    extend_prefix_lens,
    extend_seq_lens,
    has_prefix: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(0).to(tl.int64)

    prefix_len = tl.load(extend_prefix_lens + pid) if has_prefix else 0
    seq_len = tl.load(extend_seq_lens + pid)

    # NOTE: This can be slow for large bs
    cumsum_start = tl.cast(0, tl.int64)
    for i in range(pid):
        cumsum_start += tl.load(extend_seq_lens + i)

    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        tl.store(
            positions + cumsum_start + offset,
            prefix_len + offset,
            mask=offset < seq_len,
        )
    tl.store(extend_start_loc + pid, cumsum_start)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, extend_prefix_lens, extend_seq_lens):
        batch_size = extend_seq_lens.shape[0]
        has_prefix = extend_prefix_lens.shape[0] == batch_size
        extend_seq_lens_sum = extend_seq_lens.sum().item()

        positions = torch.empty(
            extend_seq_lens_sum, dtype=torch.int64, device=extend_seq_lens.device
        )
        extend_start_loc = torch.empty(
            batch_size, dtype=torch.int32, device=extend_seq_lens.device
        )

        # Launch kernel
        compute_position_kernel[(batch_size,)](
            positions,
            extend_start_loc,
            extend_prefix_lens,
            extend_seq_lens,
            has_prefix,
        )

        return positions, extend_start_loc


class ModelSGLang(nn.Module):
    def __init__(self):
        super(ModelSGLang, self).__init__()

    def forward(self, extend_prefix_lens, extend_seq_lens):
        from sglang.srt.model_executor.forward_batch_info import compute_position_triton
        extend_seq_lens_sum = extend_seq_lens.sum().item()
        return compute_position_triton(extend_prefix_lens, extend_seq_lens, extend_seq_lens_sum)


def get_inputs():
    # Example dimensions
    batch_size = 16
    max_seq_len = 128
    dtype_int64 = torch.int64

    extend_prefix_lens = torch.randint(0, 50, (batch_size,), dtype=dtype_int64)
    extend_seq_lens = torch.randint(20, max_seq_len, (batch_size,), dtype=dtype_int64)

    return [extend_prefix_lens, extend_seq_lens]


def get_init_inputs():
    return []
