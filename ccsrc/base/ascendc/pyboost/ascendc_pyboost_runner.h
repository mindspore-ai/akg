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
#ifndef MS_CUSTOM_OPS_OP_DEF_ASCENDC_PYBOOST_ASCENDC_PYBOOST_RUNNER_H_
#define MS_CUSTOM_OPS_OP_DEF_ASCENDC_PYBOOST_ASCENDC_PYBOOST_RUNNER_H_

#include "module.h"
#include "ms_extension/all.h"
#include <functional>
#include <optional>
#include <set>

namespace ms_custom_ops {
using AscendCLaunchFunc =
    std::function<void(mindspore::device::DeviceContext *, size_t)>;

class AscendCOpRunner final : public PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;
  void SetLaunchFunc(AscendCLaunchFunc func) { launch_func_ = func; }

protected:
  void LaunchKernel() override {
    MS_EXCEPTION_IF_NULL(launch_func_);
    launch_func_(_device_context_, _stream_id_);
  }

  void _DispatchLaunchTask() override { LaunchKernel(); }
  AscendCLaunchFunc launch_func_{nullptr};
};

inline mindspore::tensor::TensorPtr Tensor2Ptr(const ms::Tensor &t) {
  return t.is_defined() ? t.tensor() : nullptr;
}

inline std::vector<mindspore::tensor::TensorPtr>
Tensor2Ptr(const std::vector<ms::Tensor> &tensors) {
  std::vector<mindspore::tensor::TensorPtr> result;
  result.reserve(tensors.size());
  for (const auto &t : tensors) {
    result.push_back(t.tensor());
  }
  return result;
}

inline std::optional<mindspore::tensor::TensorPtr>
Tensor2Ptr(const std::optional<ms::Tensor> &opt_tensor) {
  if (opt_tensor.has_value()) {
    return Tensor2Ptr(opt_tensor.value());
  }
  return std::nullopt;
}

template <typename T> inline constexpr T Tensor2Ptr(const T &t) { return t; }

#define LAUNCH_ASCENDC_FUNC(aclnn_api, ...)                                    \
  [](auto &&... args) {                                                        \
    auto args_t = std::make_tuple(                                             \
      ms_custom_ops::Tensor2Ptr(std::forward<decltype(args)>(args))...);       \
    return [args_t](auto __dev_ctx, auto __stream_id) {                        \
      std::apply(                                                              \
          [&](auto &&... args) {                                               \
            LAUNCH_ACLNN(aclnn_api, __dev_ctx, __stream_id, args...);          \
          },                                                                   \
          args_t);                                                             \
    };                                                                         \
  }(__VA_ARGS__)
} // namespace ms_custom_ops

#endif // MS_CUSTOM_OPS_OP_DEF_ASCENDC_PYBOOST_ASCENDC_PYBOOST_RUNNER_H_
