import torch
from torch.utils.cpp_extension import load_inline

# 内联C++扩展代码
cpp_source = """
#include <torch/extension.h>

torch::Tensor op_name_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) x = x.contiguous();
    //具体的代码实现！
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("op_name_kernel", &op_name_kernel, "CPU op_name operator");
}
"""

# 动态加载C++扩展
op_name_module = load_inline(
    name="custom_op_name",
    cpp_sources=cpp_source,
    extra_cflags=["-O3"],
    verbose=True
)

# Python接口函数


def op_name(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        x = x.cpu()
    return op_name_module.op_name_kernel(x)
