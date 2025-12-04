# ============================================================================
# vLLM参考信息
# ============================================================================
# 源文件: vllm/attention/ops/triton_reshape_and_cache_flash.py
# vLLM函数: triton_reshape_and_cache_flash
# 实现类型: Triton kernel
# 测试文件: tests/kernels/attention/test_cache.py
# 输入参考: NUM_TOKENS=[42], NUM_HEADS=[8], HEAD_SIZES=[64,80,256], BLOCK_SIZES=[8,16,32]
# ============================================================================

import torch
import torch.nn as nn

# ============================================================================
# 从vLLM复制的Triton kernel实现
# ============================================================================
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_kernel_flash(
    key_ptr,  # [num_tokens, num_heads, head_size]
    value_ptr,  # [num_tokens, num_heads, head_size]
    key_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    k_scale,  # float32
    v_scale,  # float32
    # strides
    key_stride: tl.int64,
    value_stride: tl.int64,
    block_stride: tl.int64,
    page_stride: tl.int64,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    # FP8 flags
    FP8_KV_CACHE: tl.constexpr,
    # tune parameters
    TILE_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(axis=0)
    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    tile_i = tl.program_id(axis=1)
    tile_offs = tl.arange(0, TILE_SIZE)
    tile_pos = tile_i * TILE_SIZE + tile_offs

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size

    src_key_idx = token_idx * key_stride
    src_value_idx = token_idx * value_stride

    tgt_idx = block_idx * block_stride + block_offset * page_stride

    key_load = tl.load(
        key_ptr + src_key_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        key_tile = key_load if key_load.dtype.is_fp8() else key_load / tl.load(k_scale)
    else:
        key_tile = key_load

    value_load = tl.load(
        value_ptr + src_value_idx + tile_pos, mask=tile_pos < (num_heads * head_size)
    )
    if FP8_KV_CACHE:
        if value_load.dtype.is_fp8():
            value_tile = value_load
        else:
            value_tile = value_load / tl.load(v_scale)
    else:
        value_tile = value_load

    tl.store(
        key_cache_ptr + tgt_idx + tile_pos,
        key_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    tl.store(
        value_cache_ptr + tgt_idx + tile_pos,
        value_tile,
        mask=tile_pos < (num_heads * head_size),
    )
    return


def triton_reshape_and_cache_flash_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
):
    """从vLLM复制的host侧调用代码"""
    num_heads = key.shape[1]
    head_size = key.shape[2]
    block_size = key_cache.shape[1]
    n = num_heads * head_size

    key_stride = key.stride()[0]
    value_stride = value.stride()[0]
    block_stride = key_cache.stride()[0]
    page_stride = key_cache.stride()[1]

    FP8_KV_CACHE = kv_cache_dtype.startswith("fp8")

    TILE_SIZE = min(2048, triton.next_power_of_2(n))
    num_stages = 10
    num_warps = 16
    if torch.cuda.get_device_capability(key.device)[0] < 9:
        TILE_SIZE = min(512, TILE_SIZE)

    grid = lambda meta: (
        slot_mapping.shape[0],
        triton.cdiv(n, meta["TILE_SIZE"]),
    )

    reshape_and_cache_kernel_flash[grid](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale=k_scale,
        v_scale=v_scale,
        key_stride=key_stride,
        value_stride=value_stride,
        block_stride=block_stride,
        page_stride=page_stride,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        FP8_KV_CACHE=FP8_KV_CACHE,
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )


# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """原生实现（直接调用复制的Triton kernel）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> tuple:
        triton_reshape_and_cache_flash_impl(
            key, value, key_cache, value_cache,
            slot_mapping, "auto", k_scale, v_scale
        )
        return key_cache, value_cache


class ModelVLLM(nn.Module):
    """vLLM优化实现（通过vLLM库调用）"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> tuple:
        from vllm.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )
        triton_reshape_and_cache_flash(
            key, value, key_cache, value_cache,
            slot_mapping, "auto", k_scale, v_scale
        )
        return key_cache, value_cache


def get_inputs():
    """生成测试输入"""
    num_tokens = 42
    num_heads = 8
    head_size = 64
    block_size = 16
    num_blocks = 1024
    
    key = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value = torch.randn(num_tokens, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    key_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    value_cache = torch.zeros(num_blocks, block_size, num_heads, head_size, dtype=torch.bfloat16, device="cuda")
    
    # 生成slot_mapping（每个token映射到cache中的一个slot）
    slot_mapping = torch.randint(0, num_blocks * block_size, (num_tokens,), dtype=torch.int64, device="cuda")
    
    k_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    v_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    
    return [key, value, key_cache, value_cache, slot_mapping, k_scale, v_scale]


def get_init_inputs():
    """获取初始化参数"""
    return []

