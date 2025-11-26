
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

#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDRUN_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDRUN_H_

#include <pybind11/pybind11.h>
#include <cstdint>
#include <string>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendLaunchRuntime.h"

namespace py = pybind11;
struct AscendTensorObjStructPyTorch{
  py::object tensor_info;
  py::buffer shape_info;
  uint64_t nbytes;
  bool is_output;
  bool is_bf16;
  AscendTensorObjStructPyTorch() : nbytes(0), is_output(false), is_bf16(false) {}
  void set_value(py::object tensor, py::buffer shape, uint64_t bytes, bool output, bool bf16) {
    tensor_info = tensor;
    shape_info = shape;
    nbytes = bytes;
    is_output = output;
    is_bf16 = bf16;
  }

  void* data_ptr() {
    if (py::hasattr(tensor_info, "data_ptr")) {
      return reinterpret_cast<void*>(tensor_info.attr("data_ptr")().cast<intptr_t>());
    } else if (py::isinstance<py::buffer>(tensor_info)) {
      py::buffer buffer_info = py::cast<py::buffer>(tensor_info);
      return buffer_info.request().ptr;
    }
    throw std::runtime_error(R"(function data_ptr error: Unknown tensor type, 
       expected tensor should be pytorch tensor of numpy!)");
    return nullptr;
  }

  bool is_host(){
    if (py::hasattr(tensor_info, "data_ptr") && py::hasattr(tensor_info, "device") &&
        tensor_info.attr("device").attr("type").cast<std::string>() == "npu")
      return false;
    return true;
  }
};

using AscendTensorObjStructPyTorchPtr = std::shared_ptr<AscendTensorObjStructPyTorch>;
extern "C" void akg_ascend_run(std::string path, std::string kernel_name,
    int device_id, bool is_dynamic, const py::args &args);
#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDRUN_H_
