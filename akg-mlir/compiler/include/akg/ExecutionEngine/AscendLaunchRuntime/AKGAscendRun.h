
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
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendLaunchRuntime.h"

namespace py = pybind11;
struct AscendTensorObjStruct {
  py::buffer buffer_info;
  py::buffer shape_info;
  unsigned long long nbytes;
  bool is_output;
  bool is_dynamic;
  bool is_bf16;
  AscendTensorObjStruct() : nbytes(0), is_output(false), is_dynamic(false), is_bf16(false) {}
  void set_value(py::buffer buffer, py::buffer shape, unsigned long long bytes, bool output, bool dynamic, bool bf16) {
    buffer_info = buffer;
    shape_info = shape;
    nbytes = bytes;
    is_output = output;
    is_dynamic = dynamic;
    is_bf16 = bf16;
  }
};

using AscendTensorObjStructPtr = std::shared_ptr<AscendTensorObjStruct>;

extern "C" void akg_ascend_run(std::string path, std::string kernel_name, int device_id, bool is_dynamic, const py::args &args);
#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_AKGASCENDRUN_H_