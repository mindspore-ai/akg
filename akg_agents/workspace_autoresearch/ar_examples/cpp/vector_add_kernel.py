import torch
from torch.utils.cpp_extension import load_inline


_CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor vector_add(torch::Tensor x, torch::Tensor y) {
    return x + y;
}
"""

_module = load_inline(
    name="ar_example_cpp_vector_add",
    cpp_sources=[_CPP_SRC],
    functions=["vector_add"],
    extra_cflags=["-O3"],
    verbose=False,
)


class ModelNew(torch.nn.Module):
    def forward(self, x, y):
        return _module.vector_add(x, y)
