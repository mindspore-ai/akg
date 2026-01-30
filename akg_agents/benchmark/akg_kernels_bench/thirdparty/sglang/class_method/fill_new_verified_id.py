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
#   fill_new_verified_id_kernel - 填充新验证的token ID
# ============================================================================


@triton.jit
def fill_new_verified_id_kernel(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, verified_id, accept_lens, num_draft_tokens):
        bs = len(accept_lens)
        new_verified_id = torch.empty(
            (bs,),
            dtype=verified_id.dtype,
            device=verified_id.device,
        )

        fill_new_verified_id_kernel[(bs,)](
            verified_id,
            accept_lens,
            new_verified_id,
            num_draft_tokens,
        )

        return new_verified_id


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(self, verified_id, accept_lens, num_draft_tokens):
        bs = len(accept_lens)
        new_verified_id = torch.empty(
            (bs,),
            dtype=verified_id.dtype,
            device=verified_id.device,
        )

        # PyTorch implementation
        for pid in range(bs):
            accept_length = accept_lens[pid].item()
            verified_id_idx = num_draft_tokens * pid + accept_length - 1
            new_verified_id[pid] = verified_id[verified_id_idx]

        return new_verified_id


def get_inputs():
    # Example dimensions for Eagle V2 speculative decoding
    batch_size = 8
    num_draft_tokens = 20  # number of draft tokens per request (e.g., topk * speculative_num_steps)
    dtype = torch.int32

    # Create test inputs (not using device='cuda' as requested)
    # verified_id contains all the verified token IDs for all batches
    verified_id = torch.randint(0, 50000, (batch_size * num_draft_tokens,), dtype=dtype)
    # accept_lens contains the number of accepted tokens for each batch (1 to num_draft_tokens)
    accept_lens = torch.randint(1, num_draft_tokens + 1, (batch_size,), dtype=dtype)

    return [verified_id, accept_lens, num_draft_tokens]


def get_init_inputs():
    return []
