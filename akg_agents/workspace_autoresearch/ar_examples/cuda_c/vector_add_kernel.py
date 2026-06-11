import torch
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y);

torch::Tensor vector_add(torch::Tensor x, torch::Tensor y) {
    return vector_add_cuda(x, y);
}
"""

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void vector_add_kernel(const float* x, const float* y, float* out,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] + y[idx];
    }
}

torch::Tensor vector_add_cuda(torch::Tensor x, torch::Tensor y) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""

_module = load_inline(
    name="ar_example_cuda_vector_add",
    cpp_sources=[_CPP_SRC],
    cuda_sources=[_CUDA_SRC],
    functions=["vector_add"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        return _module.vector_add(x, y)
