import torch
import torch.nn as nn
import triton
import triton.language as tl

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件: python/sglang/srt/mem_cache/chunked_prefix_cache.py
# 测试文件: 无独立测试文件
# SGLang API调用:
#   内部使用,用于分块前缀缓存
# Triton Kernel:
#   create_chunked_prefix_cache_kv_indices_kernel - 为分块前缀缓存创建KV索引
# ============================================================================


@triton.jit
def create_chunked_prefix_cache_kv_indices_kernel(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        req_to_token,
        req_pool_indices,
        chunk_starts,
        chunk_seq_lens,
        chunk_cu_seq_lens,
    ):
        batch_size = req_pool_indices.shape[0]
        num_chunk_tokens = chunk_cu_seq_lens[-1].item()

        # Create output tensor
        chunk_kv_indices = torch.empty(
            num_chunk_tokens, dtype=torch.int32, device=req_to_token.device
        )

        # Launch kernel with batch_size programs
        grid = (batch_size,)
        create_chunked_prefix_cache_kv_indices_kernel[grid](
            req_to_token,
            req_pool_indices,
            chunk_starts,
            chunk_seq_lens,
            chunk_cu_seq_lens,
            chunk_kv_indices,
            req_to_token.shape[1],
        )

        return chunk_kv_indices


class ModelTorch(nn.Module):
    def __init__(self):
        super(ModelTorch, self).__init__()

    def forward(
        self,
        req_to_token,
        req_pool_indices,
        chunk_starts,
        chunk_seq_lens,
        chunk_cu_seq_lens,
    ):
        batch_size = req_pool_indices.shape[0]
        num_chunk_tokens = chunk_cu_seq_lens[-1].item()

        # Create output tensor
        chunk_kv_indices = torch.empty(
            num_chunk_tokens, dtype=torch.int32, device=req_to_token.device
        )

        # Process each batch item
        for i in range(batch_size):
            req_pool_index = req_pool_indices[i].item()
            chunk_start_pos = chunk_starts[i].item()
            chunk_seq_len = chunk_seq_lens[i].item()
            chunk_kv_indices_offset = chunk_cu_seq_lens[i].item()

            # Extract tokens from req_to_token for this chunk
            tokens = req_to_token[
                req_pool_index,
                chunk_start_pos : chunk_start_pos + chunk_seq_len
            ]

            # Store in output
            chunk_kv_indices[
                chunk_kv_indices_offset : chunk_kv_indices_offset + chunk_seq_len
            ] = tokens

        return chunk_kv_indices


def get_inputs():
    # Example dimensions
    batch_size = 8
    max_batch = 16
    max_context_len = 1024
    dtype = torch.int32

    # req_to_token: mapping from request to tokens
    req_to_token = torch.randint(
        0, 10000, (max_batch, max_context_len), dtype=dtype
    )

    # req_pool_indices: which request pool index each batch item uses
    req_pool_indices = torch.randint(0, max_batch, (batch_size,), dtype=dtype)

    # chunk_starts: starting position of each chunk
    chunk_starts = torch.randint(0, max_context_len // 2, (batch_size,), dtype=dtype)

    # chunk_seq_lens: length of each chunk (varying lengths)
    chunk_seq_lens = torch.randint(10, 200, (batch_size,), dtype=dtype)

    # chunk_cu_seq_lens: cumulative sum of chunk lengths (batch_size + 1)
    chunk_cu_seq_lens = torch.zeros(batch_size + 1, dtype=dtype)
    chunk_cu_seq_lens[1:] = torch.cumsum(chunk_seq_lens, dim=0)

    return [req_to_token, req_pool_indices, chunk_starts, chunk_seq_lens, chunk_cu_seq_lens]


def get_init_inputs():
    return []
