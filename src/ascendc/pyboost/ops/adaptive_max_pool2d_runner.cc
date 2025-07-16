/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ascendc_pyboost_runner.h"

namespace ms_custom_ops {
class AdaptiveMaxPool2dRunner : public AscendCOpRunner {
 public:
  using AscendCOpRunner::AscendCOpRunner;
};

std::vector<ms::Tensor> custom_adaptive_max_pool2d(const ms::Tensor &x, const std::vector<int64_t> &output_size) {
  auto out_dtype = x.data_type();
  auto out_shape = x.shape();
  size_t len = out_shape.size();
  out_shape[len - 2] = output_size[0];
  out_shape[len - 1] = output_size[1];
  auto out = ms::Tensor(out_dtype, out_shape);
  auto out_indices = ms::Tensor(kNumberTypeInt64, out_shape);

  auto runner = std::make_shared<AdaptiveMaxPool2dRunner>("AdaptiveMaxPool2d");
  auto ms_x = x.tensor();
  auto ms_out = out.tensor();
  auto ms_out_indices = out_indices.tensor();
  runner->SetLaunchFunc(LAUNCH_ASCENDC_FUNC(aclnnAdaptiveMaxPool2d, ms_x, output_size, ms_out, ms_out_indices));
  runner->Run({x}, {out, out_indices});
  return {out, out_indices};
}
}  // namespace ms_custom_ops

py::object pyboost_adaptive_max_pool2d(const ms::Tensor &x, const std::vector<int64_t> &output_size) {
  return ms_custom_ops::AdaptiveMaxPool2dRunner::Call<2>(ms_custom_ops::custom_adaptive_max_pool2d, x, output_size);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("adaptive_max_pool2d", &pyboost_adaptive_max_pool2d, "adaptive maxpool 2d", pybind11::arg("x"),
        pybind11::arg("output_size"));
}

