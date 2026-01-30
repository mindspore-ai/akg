import torch
import triton
import triton.language as tl

@triton.jit
def amax_kernel(
    in_ptr0, 
    out_ptr0, 
    n_elements: tl.constexpr,
):
    offsets = tl.arange(0, n_elements)

    block_data = tl.load(in_ptr0 + offsets)

    max_val = tl.max(block_data)
    tl.store(out_ptr0, max_val)

def amax_triton_torch(input0):
    """
    1D, reduce_axis = 0
    """
    n_elements = input0.numel()
    output0 = torch.tensor([-float('inf')], dtype=input0.dtype, device=input0.device)

    amax_kernel[(1, )](
        input0, 
        output0, 
        n_elements
    )
    return output0