# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

import torch
import triton
import triton.language as tl


def get_npu_properties():
    """获取NPU属性"""
    return {"num_aicore": 20}


def torch_dtype_to_triton_dtype(dtype):
    """转换torch dtype到triton dtype"""
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


@triton.jit
def compute_matmul_block(
        mat_a, mat_b, mat_c,
        m_start: tl.constexpr, n_start: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
):
    """通用计算函数:K循环迭代,加载块,点积,存储"""
    mat_c_block = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        mat_a_offset = ((m_start + tl.arange(0, BLOCK_M)) * K)[:, None] + (
            k_start + tl.arange(0, BLOCK_K)
        )[None, :]
        mat_a_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
            (k_start + tl.arange(0, BLOCK_K)) < K
        )[None, :]
        mat_a_block = tl.load(mat_a + mat_a_offset, mask=mat_a_mask, other=0.0)
        tl.compile_hint(mat_a_block, "dot_pad_only_k")

        mat_b_offset = ((k_start + tl.arange(0, BLOCK_K)) * N)[:, None] + (
            n_start + tl.arange(0, BLOCK_N)
        )[None, :]
        mat_b_mask = ((k_start + tl.arange(0, BLOCK_K)) < K)[:, None] & (
            (n_start + tl.arange(0, BLOCK_N)) < N
        )[None, :]
        mat_b_block = tl.load(mat_b + mat_b_offset, mask=mat_b_mask, other=0.0)
        tl.compile_hint(mat_b_block, "dot_pad_only_k")

        mat_c_block = tl.dot(mat_a_block, mat_b_block, mat_c_block)

    mat_c_offset = ((m_start + tl.arange(0, BLOCK_M)) * N)[:, None] + (
        n_start + tl.arange(0, BLOCK_N)
    )[None, :]
    mat_c_mask = ((m_start + tl.arange(0, BLOCK_M)) < M)[:, None] & (
        (n_start + tl.arange(0, BLOCK_N)) < N
    )[None, :]
    tl.store(mat_c + mat_c_offset, mat_c_block.to(OUTPUT_DTYPE), mask=mat_c_mask)


@triton.autotune(
    configs=[
        triton.Config({'GROUP_SIZE': 1}),
        triton.Config({'GROUP_SIZE': 2}),
        triton.Config({'GROUP_SIZE': 3}),
        triton.Config({'GROUP_SIZE': 4}),
        triton.Config({'GROUP_SIZE': 5}),
        triton.Config({'GROUP_SIZE': 8}),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def matmul_kernel_swizzle2d(
        mat_a, mat_b, mat_c,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        num_cores: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        DIRECTION: tl.constexpr,
        OUTPUT_DTYPE: tl.constexpr,
):
    """
    使用Swizzle2D分组重排策略,组内块共享数据,提升缓存局部性

    重要:此kernel使用固定核心数启动,每个核心处理多个块
    - grid=(num_cores,) 即 (20,) 启动20个核心
    - 每个核心通过for循环处理 NUM_BLOCKS//num_cores 个块
    - 不是grid=(NUM_BLOCKS,)启动所有块!

    DIRECTION=0 (M≥N): 行优先分组,使用tl.swizzle2d
    DIRECTION=1 (M<N): 列优先分组,手动实现
    """
    pid = tl.program_id(axis=0)  # 当前核心ID: 0~19
    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N  # 总块数

    # 每个核心循环处理多个块: pid, pid+num_cores, pid+2*num_cores, ...
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        block_m = block_idx // NUM_BLOCKS_N
        block_n = block_idx % NUM_BLOCKS_N

        if DIRECTION == 0:
            task_m_idx, task_n_idx = tl.swizzle2d(
                block_m, block_n,
                NUM_BLOCKS_M, NUM_BLOCKS_N,
                GROUP_SIZE
            )
        else:
            size_gj = GROUP_SIZE * NUM_BLOCKS_M
            group_id = block_idx // size_gj
            off_n = group_id * GROUP_SIZE
            cur_size_g = tl.minimum(NUM_BLOCKS_N - off_n, GROUP_SIZE)
            local_ij = block_idx % size_gj
            task_m_idx = local_ij // cur_size_g
            task_n_idx = off_n + local_ij % cur_size_g

        m_start = task_m_idx * BLOCK_M
        n_start = task_n_idx * BLOCK_N

        compute_matmul_block(mat_a, mat_b, mat_c, m_start, n_start, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, OUTPUT_DTYPE)


def triton_matmul(
    mat_a,
    mat_b,
    dtype=torch.bfloat16,
):
    """
    Triton矩阵乘法 A[M,K] @ B[K,N] = C[M,N]

    关键:kernel启动使用固定核心数
    - 使用 matmul_kernel_swizzle2d[(num_cores,)](...) 启动
    - grid=(num_cores,) 即 (20,) 固定启动20个核心
    - 不要使用 grid=(NUM_BLOCKS_M*NUM_BLOCKS_N,) !

    分块大小:
    - float16/bfloat16: BLOCK_M=128, BLOCK_K=256, BLOCK_N=256
    - float32: BLOCK_M=BLOCK_K=BLOCK_N=128
    """
    m = mat_a.shape[0]
    k = mat_a.shape[1]
    n = mat_b.shape[1]

    # 根据矩阵形状选择分组方向
    DIRECTION = 1 if m < n else 0

    # 根据数据类型选择块大小
    if dtype in [torch.float16, torch.bfloat16]:
        BLOCK_M, BLOCK_K, BLOCK_N = 128, 256, 256
    elif dtype == torch.float32:
        BLOCK_M, BLOCK_K, BLOCK_N = 128, 128, 128
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    mat_c = torch.empty(m, n, dtype=dtype, device=mat_a.device)
    num_cores = get_npu_properties()["num_aicore"]  # 20
    output_dtype = torch_dtype_to_triton_dtype(dtype)

    # 关键:使用固定核心数启动,grid=(num_cores,)即(20,)
    matmul_kernel_swizzle2d[(num_cores,)](
        mat_a, mat_b, mat_c,
        m, n, k, num_cores,
        BLOCK_M, BLOCK_N, BLOCK_K,
        DIRECTION=DIRECTION,
        OUTPUT_DTYPE=output_dtype
    )

    return mat_c
