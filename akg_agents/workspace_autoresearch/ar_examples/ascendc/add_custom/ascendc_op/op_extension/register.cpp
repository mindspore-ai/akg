#include <torch/extension.h>
#include <torch/library.h>

#include "ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(npu, m) {
  m.def("add_custom(Tensor x1, Tensor x2) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
  m.impl("add_custom", TORCH_FN(ascend_kernel::add_custom_torch));
}

at::Tensor add_custom_meta(const at::Tensor& x1, const at::Tensor& x2) {
  return at::empty_like(x1);
}

TORCH_LIBRARY_IMPL(npu, Meta, m) {
  m.impl("add_custom", &add_custom_meta);
}

}  // namespace
