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

#include "ms_extension/all.h"
#include <functional>
#include <optional>
#include <set>
#include "module.h"


namespace ms_custom_ops {
using namespace mindspore;
using namespace ms::pynative;

using AscendCLaunchFunc =
    std::function<void(mindspore::device::DeviceContext *, size_t)>;
using AscendCWorkSpaceFunc = std::function<size_t()>;

class AscendCOpRunner : public PyboostRunner {
public:
  using PyboostRunner::PyboostRunner;
  void SetLaunchFunc(AscendCLaunchFunc func) { launch_func_ = func; }
  void SetWorkSpaceFunc(AscendCWorkSpaceFunc func) { workspace_func_ = func; }

protected:
  size_t CalcWorkspace() override {
    if (workspace_func_ != nullptr) {
      return workspace_func_();
    }
    return 0;
  }

  void LaunchKernel() override {
    if (launch_func_ != nullptr) {
      launch_func_(_device_context_, _stream_id_);
    }
  }
  void _DispatchLaunchTask() override { LaunchKernel(); }
  AscendCLaunchFunc launch_func_{nullptr};
  AscendCWorkSpaceFunc workspace_func_{nullptr};
};

#define LAUNCH_ASCENDC_FUNC(aclnn_api, ...)                                    \
  [__VA_ARGS__](auto __device_context, auto __stream_id) {                     \
    LAUNCH_ACLNN(aclnn_api, __device_context, __stream_id, __VA_ARGS__);       \
  }
} // namespace ms_custom_ops

#endif // MS_CUSTOM_OPS_OP_DEF_ASCENDC_PYBOOST_ASCENDC_PYBOOST_RUNNER_H_
