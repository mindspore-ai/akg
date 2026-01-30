import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/speculative/eagle_info_v2.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   内部使用,用于EAGLE v2推测解码
# Triton Kernel:
#   fill_accepted_out_cache_loc_kernel - 填充接受的输出缓存位置
# ============================================================================


@triton.jit
def fill_accepted_out_cache_loc_kernel(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, accept_index, out_cache_loc):
        size = len(accept_index)
        # Find next power of 2 for size_upper (required by triton kernel)
        size_upper = 1
        while size_upper < size:
            size_upper *= 2

        # Count how many valid indices we have
        num_accepted = (accept_index >= 0).sum().item()
        accepted_out_cache_loc = torch.empty(
            (num_accepted,),
            dtype=out_cache_loc.dtype,
            device=out_cache_loc.device,
        )

        fill_accepted_out_cache_loc_kernel[(size,)](
            accept_index,
            out_cache_loc,
            accepted_out_cache_loc,
            size_upper,
        )

        return accepted_out_cache_loc


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, accept_index, out_cache_loc):
        size = len(accept_index)

        # Count how many valid indices we have
        num_accepted = (accept_index >= 0).sum().item()
        accepted_out_cache_loc = torch.empty(
            (num_accepted,),
            dtype=out_cache_loc.dtype,
            device=out_cache_loc.device,
        )

        # PyTorch implementation
        dst_idx = 0
        for pid in range(size):
            # Count valid indices before current position
            masks = (accept_index[:pid] != -1)
            dst = masks.sum().item()

            src = accept_index[pid].item()
            if src > -1:
                value = out_cache_loc[src]
                accepted_out_cache_loc[dst] = value
                dst_idx += 1

        return accepted_out_cache_loc


def get_inputs():
    # Example dimensions for Eagle V2 speculative decoding
    size = 64  # Total number of candidate positions (e.g., batch_size * (spec_steps + 1))
    max_cache_size = 128  # Size of out_cache_loc
    dtype = torch.int64

    # Create test inputs (not using device='cuda' as requested)
    # accept_index: -1 means rejected, >= 0 means accepted with the corresponding index
    accept_index = torch.randint(-1, max_cache_size, (size,), dtype=torch.int32)
    # Make some indices invalid (-1) to simulate rejection
    accept_index[accept_index < max_cache_size // 4] = -1

    # out_cache_loc: cache locations for draft tokens
    out_cache_loc = torch.randint(0, 100000, (max_cache_size,), dtype=dtype)

    return [accept_index, out_cache_loc]


def get_init_inputs():
    return []
