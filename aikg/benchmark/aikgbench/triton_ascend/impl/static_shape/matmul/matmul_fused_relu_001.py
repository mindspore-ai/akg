import torch
import triton
import triton.language as tl
import triton.runtime.driver as driver


# get device properties of npu
def get_npu_properties():
    device = torch.npu.current_device()
    return driver.active.utils.get_device_properties(device)


@triton.jit
def aikg_op_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_cores: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SWIZZLE_COUNT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    task_m_idx = 0
    task_n_idx = 0

    NUM_BLOCKS_M = triton.cdiv(M, BLOCK_SIZE_M)
    NUM_BLOCKS_N = triton.cdiv(N, BLOCK_SIZE_N)
    NUM_BLOCKS = NUM_BLOCKS_M * NUM_BLOCKS_N
    for block_idx in range (
            pid, NUM_BLOCKS, num_cores
        ):
        # 以nZ方向为例
        in_batch_idx = block_idx % (NUM_BLOCKS_M * NUM_BLOCKS_N)
        in_batch_idx = block_idx % NUM_BLOCKS
        tile_block_loop = (NUM_BLOCKS_M + SWIZZLE_COUNT - 1) // SWIZZLE_COUNT
        tile_block_idx = in_batch_idx // (SWIZZLE_COUNT * NUM_BLOCKS_N)
        in_tile_block_idx = in_batch_idx % (SWIZZLE_COUNT * NUM_BLOCKS_N)

        n_row = SWIZZLE_COUNT
        if tile_block_idx == tile_block_loop - 1:
            n_row = NUM_BLOCKS_M - SWIZZLE_COUNT * tile_block_idx
        task_m_idx = tile_block_idx * SWIZZLE_COUNT + in_tile_block_idx % n_row
        task_n_idx = in_tile_block_idx // n_row
        if tile_block_idx % 2 != 0:
            task_n_idx = NUM_BLOCKS_N - task_n_idx - 1
        
        m_start = task_m_idx * BLOCK_SIZE_M
        n_start = task_n_idx * BLOCK_SIZE_N
        
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            k_offset = k * BLOCK_SIZE_K
            
            a_offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
            a_offsets_k = k_offset + tl.arange(0, BLOCK_SIZE_K)[None, :]
            a_mask = (a_offsets_m < M) & (a_offsets_k < K)
            
            b_offsets_k = k_offset + tl.arange(0, BLOCK_SIZE_K)[:, None]
            b_offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
            b_mask = (b_offsets_k < K) & (b_offsets_n < N)
            
            a = tl.load(a_ptr + a_offsets_m * stride_am + a_offsets_k * stride_ak, 
                    mask=a_mask, other=0.0)
            b = tl.load(b_ptr + b_offsets_k * stride_bk + b_offsets_n * stride_bn, 
                    mask=b_mask, other=0.0)

            acc += tl.dot(a, b)
        
        # ReLU fusion
        acc = tl.maximum(acc, 0.0)

        c_offsets_m = m_start + tl.arange(0, BLOCK_SIZE_M)[:, None]
        c_offsets_n = n_start + tl.arange(0, BLOCK_SIZE_N)[None, :]
        c_mask = (c_offsets_m < M) & (c_offsets_n < N)
        
        tl.store(c_ptr + c_offsets_m * stride_cm + c_offsets_n * stride_cn, 
                acc, mask=c_mask)

def matmul_fused_relu_001_op_triton_torch(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Triton 矩阵乘法启动函数
    
    Args:
        A: Input tensor of shape (M, K).
        B: Input tensor of shape (K, N).
    
    Returns:
        Output tensor of shape (M, N).
    """
    M, K1 = A.shape
    K2, N = B.shape
    assert K1 == K2, f"矩阵维度不匹配: {K1} != {K2}"
    
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 256
    BLOCK_SIZE_K = 256
    SWIZZLE_COUNT = 4

    num_cores = get_npu_properties()["num_aicore"]
    aikg_op_kernel[num_cores](
        A, B, C,
        M, N, K1,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        num_cores,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        SWIZZLE_COUNT
    )
    return C