import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# 自定义CUDA C内核
cuda_c_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float data = input[idx];
        float result = fmaxf(0.0f, data);
        output[idx] = result;
    }
}

torch::Tensor relu_kernel_call(torch::Tensor input) {
    int size = input.numel();
    auto output = torch::zeros_like(input);
    
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    relu_kernel<<<num_blocks, block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""
cpp_source = """torch::Tensor relu_kernel_call(torch::Tensor input);"""

kernel_module = load_inline(
    name="relu_cuda_c",
    cpp_sources=cpp_source,
    cuda_sources=cuda_c_source,
    functions=["relu_kernel_call"],
    verbose=True,
    extra_cflags=["-O3"],
    extra_ldflags=["-lcudart"],
)

def relu_cuda_c_torch(x):
    return kernel_module.relu_kernel_call(x)