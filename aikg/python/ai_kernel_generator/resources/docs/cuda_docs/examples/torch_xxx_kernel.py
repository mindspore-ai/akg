import torch
from torch.utils.cpp_extension import load_inline
# 自定义CUDA C内核
source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void op_name_kernel(const Type* input, Type* output, int size) {
    //内核代码
}
//内核调用
torch::Tensor op_name_kernel_call(torch::Tensor input){
    auto size = input.numel();
    auto output = torch::zeros_like(input);
    //计算num_blocks、block_size
    op_name_kernel<<<num_blocks、block_size>>>(input.data_ptr<float>(), output.data_ptr<float>(), size);
    return output;
}
"""
cpp_src = ("torch::Tensor op_name_kernel_call(torch::Tensor input);")
# JIT编译
kernel_module = load_inline(
    name = "op_name_cuda",
    cpp_sources = cpp_src,
    cuda_sources = source,
    functions = ["op_name_kernel_call"],
    verbose = True,
    extra_cflags = [""],
    extra_ldflags = [""],
)

def op_name_kernel_call(x):
    return kernel_module.op_name_kernel_call(x)