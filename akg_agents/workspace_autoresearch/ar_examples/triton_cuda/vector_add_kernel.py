import torch
import triton
import triton.language as tl


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, out_ptr, n_elements: tl.constexpr,
                       block_size: tl.constexpr):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        out = torch.empty_like(x)
        n_elements = out.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _vector_add_kernel[grid](x, y, out, n_elements,
                                 block_size=block_size)
        return out
