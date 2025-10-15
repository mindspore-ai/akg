import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# 内联C++扩展代码
cpp_source = """
#include <torch/extension.h>
torch::Tensor ReLU_kernel(torch::Tensor x) {
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    torch::ScalarType dtype = x.scalar_type();
    bool need_convert = (dtype != torch::kFloat32 && dtype != torch::kInt32 && dtype != torch::kInt64);
    torch::Tensor input = need_convert ? x.to(torch::kFloat32) : x;
    torch::Tensor output = torch::zeros_like(input);

    if (input.scalar_type() == torch::kFloat32) {
        auto x_ptr = input.data_ptr<float>();
        auto out_ptr = output.data_ptr<float>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0.0f, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt32) {
        auto x_ptr = input.data_ptr<int32_t>();
        auto out_ptr = output.data_ptr<int32_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(0, x_ptr[i]);
        }
    } else if (input.scalar_type() == torch::kInt64) {
        auto x_ptr = input.data_ptr<int64_t>();
        auto out_ptr = output.data_ptr<int64_t>();
        int64_t numel = input.numel();
        for (int64_t i = 0; i < numel; ++i) {
            out_ptr[i] = std::max(static_cast<int64_t>(0), x_ptr[i]);
        }
    }
    if (need_convert) {
        output = output.to(dtype);
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ReLU_kernel", &ReLU_kernel, "ReLU C++ kernel");
}
"""

# 动态加载C++扩展
ReLU_module = load_inline(
    name="ReLU",
    cpp_sources=cpp_source,
    extra_cflags=["-O3"],
    verbose=True
)
# 内核调用
def relu_cpp_torch(x: torch.Tensor) -> torch.Tensor:
    if x.device.type != "cpu":
        x = x.cpu()
    return ReLU_module.ReLU_kernel(x)
