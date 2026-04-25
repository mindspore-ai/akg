import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/layers/attention/utils.py
# 测试文件: 无独立测试文件 (被 flashmla_backend.py 调用)
# SGLang API调用:
#   create_flashmla_kv_indices_triton[(batch,)](req_to_token, req_pool_indices,
#       page_kernel_lens, kv_start_idx, kv_indices, req_to_token_stride,
#       kv_indices_stride, PAGED_SIZE=page_size)
# Triton Kernel:
#   create_flashmla_kv_indices_triton - 创建FlashMLA的KV索引
# ============================================================================

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)


@triton.jit
def create_flashmla_kv_indices_triton_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    for i in range(num_pages_loop):
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )


class Model(nn.Module):
    """Triton kernel实现"""

    def __init__(self, page_size=64):
        super(Model, self).__init__()
        self.page_size = page_size

    def forward(self, req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx=None):
        batch = len(req_pool_indices)
        max_kv_len = page_kernel_lens.max().item()
        max_num_pages = (max_kv_len + self.page_size - 1) // self.page_size
        kv_indices = torch.zeros(batch, max_num_pages, dtype=torch.int64, device=req_to_token.device)
        create_flashmla_kv_indices_triton_kernel[(batch,)](
            req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx, kv_indices,
            req_to_token.size(1), kv_indices.size(1), PAGED_SIZE=self.page_size,
        )
        return kv_indices


class ModelSglang(nn.Module):
    """SGLang API调用"""

    def __init__(self, page_size=64):
        super(ModelSglang, self).__init__()
        self.page_size = page_size

    def forward(self, req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx=None):
        """
        调用SGLang create_flashmla_kv_indices_triton API
        """
        from sglang.srt.layers.attention.utils import create_flashmla_kv_indices_triton

        batch = len(req_pool_indices)
        max_kv_len = page_kernel_lens.max().item()
        max_num_pages = (max_kv_len + self.page_size - 1) // self.page_size
        kv_indices = torch.zeros(batch, max_num_pages, dtype=torch.int64, device=req_to_token.device)

        create_flashmla_kv_indices_triton[(batch,)](
            req_to_token,
            req_pool_indices,
            page_kernel_lens,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
            kv_indices.size(1),
            PAGED_SIZE=self.page_size,
        )
        return kv_indices


def get_inputs():
    batch_size = 4
    max_batch = 4096
    max_context_len = 512

    req_to_token = torch.arange(
        max_batch * max_context_len, dtype=torch.int32
    ).reshape((max_batch, max_context_len))

    req_pool_indices = torch.randint(0, max_batch, (batch_size,), dtype=torch.int32)

    page_kernel_lens = torch.randint(1, max_context_len, (batch_size,), dtype=torch.int32)

    kv_start_idx = None

    return [req_to_token, req_pool_indices, page_kernel_lens, kv_start_idx]


def get_init_inputs():
    page_size = 64
    return [page_size]
