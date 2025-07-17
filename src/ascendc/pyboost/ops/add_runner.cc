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
class AddRunner : public AscendCOpRunner {
 public:
  using AscendCOpRunner::AscendCOpRunner;
};

ms::Tensor custom_add(const ms::Tensor &x, const ms::Tensor &y) {
  auto out = ms::Tensor(x.data_type(), x.shape());
  auto runner = std::make_shared<AddRunner>("AddCustom");
  auto ms_x = x.tensor();
  auto ms_y = y.tensor();
  auto ms_out = out.tensor();
  runner->SetLaunchFunc(LAUNCH_ASCENDC_FUNC(aclnnAddCustom, ms_x, ms_y, ms_out));
  runner->Run({x, y}, {out});
  return out;
}
}  // namespace ms_custom_ops

py::object pyboost_add(const ms::Tensor &x, const ms::Tensor &y) {
  return ms_custom_ops::AddRunner::Call<1>(ms_custom_ops::custom_add, x, y);
}

MS_CUSTOM_OPS_EXTENSION_MODULE(m) {
  m.def("add", &pyboost_add, "add", pybind11::arg("x"), pybind11::arg("y"));
}
