import torch
import triton
import triton.language as tl

@triton.jit
def tensor_matrix_multiplication_3d_kernel(
    output_ptr,
    input_ptr_A,
    input_ptr_B,
    N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bl: tl.constexpr,
    stride_cm: tl.constexpr,
    stride_cl: tl.constexpr,
    TILE_K: tl.constexpr,
    TILE_L: tl.constexpr,
    INNER_TILE_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr
):
    pid = tl.program_id(0)
    total_rows = N * M
    base_rows_per_core = total_rows // BLOCK_DIM
    remainder = total_rows % BLOCK_DIM
    
    if pid < remainder:
        start_row = pid * (base_rows_per_core + 1)
        num_rows = base_rows_per_core + 1
    else:
        start_row = remainder * (base_rows_per_core + 1) + (pid - remainder) * base_rows_per_core
        num_rows = base_rows_per_core
    
    num_blocks = num_rows // INNER_TILE_M
    
    for block_idx in range(num_blocks):
        start_m = start_row + block_idx * INNER_TILE_M
        
        for l_start in range(0, L, TILE_L):
            l_end = l_start + TILE_L
            accumulator = tl.zeros((INNER_TILE_M, TILE_L), dtype=tl.float32)
            
            for k_start in range(0, K, TILE_K):
                k_end = k_start + TILE_K
                
                # 加载A块
                row_offsets_A = start_m + tl.arange(0, INNER_TILE_M)
                col_offsets_A = tl.arange(0, TILE_K) + k_start
                a_ptrs = input_ptr_A + row_offsets_A[:, None] * stride_am + col_offsets_A[None, :] * stride_ak
                a_block = tl.load(a_ptrs)
                
                # 加载B块
                row_offsets_B = tl.arange(0, TILE_K) + k_start
                col_offsets_B = tl.arange(0, TILE_L) + l_start
                b_ptrs = input_ptr_B + row_offsets_B[:, None] * stride_bk + col_offsets_B[None, :] * stride_bl
                b_block = tl.load(b_ptrs)
                
                # 矩阵乘法并累加
                accumulator += tl.dot(a_block, b_block)
            
            # 存储结果
            c_value = accumulator.to(tl.float16)
            row_offsets_C = start_m + tl.arange(0, INNER_TILE_M)
            col_offsets_C = tl.arange(0, TILE_L) + l_start
            c_ptrs = output_ptr + row_offsets_C[:, None] * stride_cm + col_offsets_C[None, :] * stride_cl
            tl.store(c_ptrs, c_value)

def tensor_matrix_multiplication_3d_triton_torch(A, B):
    assert A.dim() == 3 and B.dim() == 2
    N, M, K = A.shape
    assert B.shape[0] == K
    L = B.shape[1]
    
    C = torch.empty((N, M, L), device=A.device, dtype=A.dtype)
    
    A_2d = A.view(-1, K)
    C_2d = C.view(-1, L)
    
    stride_am = A_2d.stride(0)
    stride_ak = A_2d.stride(1)
    stride_bk = B.stride(0)
    stride_bl = B.stride(1)
    stride_cm = C_2d.stride(0)
    stride_cl = C_2d.stride(1)
    
    BLOCK_DIM = 16
    TILE_K = 128
    TILE_L = 128
    INNER_TILE_M = 128
    
    grid = (BLOCK_DIM,)
    
    tensor_matrix_multiplication_3d_kernel[grid](
        C_2d,
        A_2d,
        B,
        N,
        M,
        K,
        L,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bl,
        stride_cm,
        stride_cl,
        TILE_K,
        TILE_L,
        INNER_TILE_M,
        BLOCK_DIM
    )
    
    return C