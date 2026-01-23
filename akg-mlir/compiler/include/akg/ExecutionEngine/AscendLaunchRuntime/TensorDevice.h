/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <cstddef>
#include <memory>
#include <type_traits>
#include <cstring>
#include <stdexcept>

namespace mlir {
namespace runtime {

class BaseDevice {
 public:
  virtual ~BaseDevice() = default;
 protected:
  BaseDevice() = default;
};

class ScalarDevice : public BaseDevice {
 public:
  explicit ScalarDevice(void* data_ptr) : data_ptr_(data_ptr) {}

  ~ScalarDevice() override = default;

  void* GetScalarValuePtr() const {
    return data_ptr_;
  }

 private:
  void* data_ptr_;
};

class TensorDevice : public BaseDevice {
 public:
  TensorDevice(void *host_address, size_t nbytes, bool is_output)
      : host_address_(host_address),
        device_address_(nullptr),
        nbytes_(nbytes),
        is_output_(is_output) {}

  TensorDevice(void *host_address, void *device_address, size_t nbytes, bool is_output)
      : host_address_(host_address),
        device_address_(device_address),
        nbytes_(nbytes),
        is_output_(is_output) {
    if (device_address_ != nullptr) is_host_ = false;
  }

  ~TensorDevice() override = default;

  size_t GetDataSize() const { return nbytes_; }
  void *GetHostAddress() const { return host_address_; }
  void *GetDeviceAddress() const { return device_address_; }
  bool IsOutput() const { return is_output_; }
  bool IsHostTensor() const { return is_host_; }

  void SetDeviceAddress(void *device_address) {
    device_address_ = device_address;
  }

 private:
  void *host_address_;
  void *device_address_;
  size_t nbytes_;
  bool is_output_;
  bool is_host_{true};
};

inline std::shared_ptr<TensorDevice> AsTensorDevice(const std::shared_ptr<BaseDevice>& base) {
  return std::dynamic_pointer_cast<TensorDevice>(base);
}

inline std::shared_ptr<ScalarDevice> AsScalarDevice(const std::shared_ptr<BaseDevice>& base) {
  return std::dynamic_pointer_cast<ScalarDevice>(base);
}

inline void* GetScalarValuePtr(const std::shared_ptr<BaseDevice>& base) {
  if (auto scalar = AsScalarDevice(base)) {
    return scalar->GetScalarValuePtr();
  }
  throw std::runtime_error("GetScalarValuePtr() called on non-scalar device");
}

using BaseDevicePtr = std::shared_ptr<BaseDevice>;
using ScalarDevicePtr = std::shared_ptr<ScalarDevice>;
using TensorDevicePtr = std::shared_ptr<TensorDevice>;
}  // namespace runtime
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_TENSORDEVICE_H_
