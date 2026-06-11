#pragma once

#include <torch/extension.h>

namespace ascend_kernel {

at::Tensor add_custom_torch(const at::Tensor& x1, const at::Tensor& x2);

}  // namespace ascend_kernel
