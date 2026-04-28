import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_BM': 32, 'SUB_BLOCK_SIZE_BM': 32, 'BLOCK_SIZE_N': 128}),
        triton.Config({'BLOCK_SIZE_BM': 16, 'SUB_BLOCK_SIZE_BM': 16, 'BLOCK_SIZE_N': 256}),
        triton.Config({'BLOCK_SIZE_BM': 8, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_BM': 512, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),
        triton.Config({'BLOCK_SIZE_BM': 416, 'SUB_BLOCK_SIZE_BM': 8, 'BLOCK_SIZE_N': 512}),
    ],
    key=['BM', 'N']
)
@triton.jit
def reduction_weighted_swiglu_bwd_kernel(
    x_ptr, weight_ptr, grad_ptr, 
    weighted_x_ptr, grad_weight_ptr, grad_x_ptr,
    BM, N,
    stride_x_bm, stride_x_n,
    stride_weight_bm, stride_weight_n,
    stride_grad_bm, stride_grad_n,
    stride_weighted_x_bm, stride_weighted_x_n,
    stride_grad_weight_bm, stride_grad_weight_n,
    stride_grad_x_bm,
    BLOCK_SIZE_BM: tl.constexpr,
    SUB_BLOCK_SIZE_BM: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    for bm_start in range(0, BLOCK_SIZE_BM, SUB_BLOCK_SIZE_BM):
        bm_offsets = pid * BLOCK_SIZE_BM + bm_start + tl.arange(0, SUB_BLOCK_SIZE_BM)
        bm_mask = bm_offsets < BM

        accumulator = tl.zeros((SUB_BLOCK_SIZE_BM,), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_SIZE_N):
            n_offsets = n_start + tl.arange(0, BLOCK_SIZE_N)
            n_mask = n_offsets < N
            
            bm_offsets_bc = bm_offsets[:, None] 
            n_offsets_bc = n_offsets[None, :] 
            
            x_offsets = bm_offsets_bc * stride_x_bm + n_offsets_bc * stride_x_n
            weight_offsets = bm_offsets_bc * stride_weight_bm + n_offsets_bc * stride_weight_n
            grad_offsets = bm_offsets_bc * stride_grad_bm + n_offsets_bc * stride_grad_n
            
            mask = bm_mask[:, None] & n_mask[None, :]
            
            x_vals = tl.load(x_ptr + x_offsets, mask=mask, other=0.0)
            weight_vals = tl.load(weight_ptr + x_offsets, mask=mask, other=0.0)
            grad_vals = tl.load(grad_ptr + grad_offsets, mask=mask, other=0.0)

            product_weight_x = x_vals * weight_vals
            product_grad_weight = weight_vals * grad_vals

            weight_x_offsets = bm_offsets_bc * stride_weighted_x_bm + n_offsets_bc * stride_weighted_x_n
            grad_weight_offsets = bm_offsets_bc * stride_grad_weight_bm + n_offsets_bc * stride_grad_weight_n

            tl.store(weighted_x_ptr + weight_x_offsets, product_weight_x, mask=mask)
            tl.store(grad_weight_ptr + grad_weight_offsets, product_grad_weight, mask=mask)
            
            product_grad_x = x_vals * grad_vals
            partial_sum = tl.sum(product_grad_x, axis=1)
            accumulator += partial_sum
        
        output_offsets = bm_offsets * stride_grad_x_bm
        tl.store(grad_x_ptr + output_offsets, accumulator, mask=bm_mask)

def reduction_weighted_swiglu_bwd_triton_torch(x, weight, grad):
    assert x.shape == grad.shape
    
    B, M, N = x.shape
    BM = B * M
    
    weighted_x = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    grad_weight = torch.empty((B, M, N), device=x.device, dtype=x.dtype)
    grad_x = torch.empty((B, M), device=x.device, dtype=x.dtype)
    
    x_reshaped = x.reshape(BM, N)
    weight_reshaped = weight.reshape(BM, N)
    grad_reshaped = grad.reshape(BM, N)

    weighted_x_reshaped = weighted_x.reshape(BM, N)
    grad_weight_reshaped = grad_weight.reshape(BM, N)
    grad_x_reshaped = grad_x.reshape(BM)
    
    stride_x_bm, stride_x_n = x_reshaped.stride()
    stride_weight_bm, stride_weight_n = weight_reshaped.stride()
    stride_grad_bm, stride_grad_n = grad_reshaped.stride()

    stride_weighted_x_bm, stride_weighted_x_n = weighted_x_reshaped.stride()
    stride_grad_weight_bm, stride_grad_weight_n = grad_weight_reshaped.stride()
    stride_grad_x_bm = grad_x_reshaped.stride()[0]
    
    grid = lambda meta: (triton.cdiv(BM, meta['BLOCK_SIZE_BM']),)
    
    reduction_weighted_swiglu_bwd_kernel[grid](
        x_reshaped, weight_reshaped, grad_reshaped, 
        weighted_x_reshaped, grad_weight_reshaped, grad_x_reshaped,
        BM, N,
        stride_x_bm, stride_x_n,
        stride_weight_bm, stride_weight_n,
        stride_grad_bm, stride_grad_n,
        stride_weighted_x_bm, stride_weighted_x_n,
        stride_grad_weight_bm, stride_grad_weight_n,
        stride_grad_x_bm,
    )
    
    return weighted_x, grad_weight, grad_x