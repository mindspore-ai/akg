/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_TENSORDEVICE_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_TENSORDEVICE_H_
#include <memory>

namespace mlir {
namespace runtime {
class TensorDevice {
 public:
  TensorDevice(void *host_address, size_t nbytes, bool is_output)
      : host_address_(host_address), nbytes_(nbytes), is_output_(is_output) {}
  TensorDevice(void *host_address, void *device_address, size_t nbytes, bool is_output)
      : host_address_(host_address), device_address_(device_address), nbytes_(nbytes), is_output_(is_output) {
        if(device_address_!=nullptr)
          is_host_ = false;
      }
  ~TensorDevice() {
    host_address_ = nullptr;
    device_address_ = nullptr;
  }
  size_t GetDataSize() { return nbytes_; }
  void *GetHostAddress() { return host_address_; }
  void *GetDeviceAddress() { return device_address_; }
  void SetDeviceAddress(void *device_address) { device_address_ = device_address; }
  bool IsOutput() { return is_output_; }
  bool IsHostTensor() { return is_host_; }

 private:
  void *host_address_{nullptr};
  size_t nbytes_{0};
  bool is_output_{false};
  void *device_address_{nullptr};
  bool is_host_{true};
};
using TensorDevicePtr = std::shared_ptr<TensorDevice>;
}  // namespace runtime
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_TENSORDEVICE_H_
