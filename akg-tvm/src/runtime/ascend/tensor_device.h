/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef SRC_RUNTIME_ASCEND_TENSOR_DEVICE_H_
#define SRC_RUNTIME_ASCEND_TENSOR_DEVICE_H_
#include <memory>

namespace air {
namespace runtime {
class TensorDevice {
 public:
  TensorDevice(void *host_address, size_t nbytes, bool is_output)
      : host_address_(host_address), nbytes_(nbytes), is_output_(is_output) {}
  ~TensorDevice() {
    host_address_ = nullptr;
    device_address_ = nullptr;
  }
  size_t GetDataSize() { return nbytes_; }
  void *GetHostAddress() { return host_address_; }
  void *GetDeviceAddress() { return device_address_; }
  void SetDeviceAddress(void *device_address) { device_address_ = device_address; }
  bool IsOutput() { return is_output_; }

 private:
  void *host_address_{nullptr};
  size_t nbytes_{0};
  bool is_output_{false};
  void *device_address_{nullptr};
};
using TensorDevicePtr = std::shared_ptr<TensorDevice>;
}  // namespace runtime
}  // namespace air
#endif  // SRC_RUNTIME_ASCEND_TENSOR_DEVICE_H_
