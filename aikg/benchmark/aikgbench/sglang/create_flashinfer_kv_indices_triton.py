import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/attention/utils.py
# 测试文件: test/srt/test_create_kvindices.py (test_create_kvindices)
# SGLang API调用:
#   create_flashinfer_kv_indices_triton[(batch,)](req_to_token, req_pool_indices,
#       page_kernel_lens, kv_indptr, kv_start_idx, kv_indices, req_to_token_stride)
# Triton Kernel:
#   create_flashinfer_kv_indices_triton - 创建FlashInfer的KV索引
# ============================================================================


@triton.jit
def create_flashinfer_kv_indices_triton_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx=None):
        batch = len(req_pool_indices)
        total_len = kv_indptr[-1].item()
        kv_indices = torch.empty(total_len, dtype=torch.int64, device=req_to_token.device)
        create_flashinfer_kv_indices_triton_kernel[(batch,)](
            req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx,
            kv_indices, req_to_token.size(1),
        )
        return kv_indices


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self):
        super(ModelSglang, self).__init__()

    def forward(self, req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx=None):
        """
        调用SGLang create_flashinfer_kv_indices_triton API
        """
        from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton

        batch = len(req_pool_indices)
        total_len = kv_indptr[-1].item()
        kv_indices = torch.empty(total_len, dtype=torch.int64, device=req_to_token.device)

        create_flashinfer_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_indptr,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices


def get_inputs():
    batch_size = 37
    max_batch = 4096
    max_context_len = 4096

    req_to_token = torch.arange(
        max_batch * max_context_len, dtype=torch.int32
    ).reshape((max_batch, max_context_len))

    req_pool_indices = torch.randint(0, max_batch, (batch_size,), dtype=torch.int32)

    page_kernel_lens = torch.randint(1, max_context_len, (batch_size,), dtype=torch.int32)

    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(page_kernel_lens, dim=0)

    kv_start_idx = None

    return [req_to_token, req_pool_indices, page_kernel_lens, kv_indptr, kv_start_idx]


def get_init_inputs():
    return []
