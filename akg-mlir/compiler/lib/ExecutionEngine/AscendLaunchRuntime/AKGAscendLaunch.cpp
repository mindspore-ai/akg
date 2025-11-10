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

#include <algorithm>
#include <iostream>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendRun.h"

typedef int64_t (*tiling_function_t)(void*);
typedef int64_t (*get_tiling_struct_size_function_t)();

std::vector<std::vector<uint16_t>> bf16s_;

void F32ToBF16(float *input, uint16_t *output, uint32_t size) {
  while (size-- != 0) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(output, input, 2);
#else
    memcpy(output, (char *)input + 2, 2);
#endif
    input++;
    output++;
  }
}

void BF16ToF32(uint16_t *input, float *output, uint32_t size) {
  memset(output, 0, size * 4);
  while (size-- != 0) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    memcpy(output, input, 2);
#else
    memcpy((char *)output + 2, input, 2);
#endif
    input++;
    output++;
  }
}

py::buffer_info ConvertToBF16(py::buffer_info &buf) {
  size_t size = buf.size;
  std::vector<uint16_t> bf16(size);
  F32ToBF16(reinterpret_cast<float *>(buf.ptr), bf16.data(), size);
  std::for_each(buf.strides.begin(), buf.strides.end(), [](ssize_t &stride) { stride /= 2; });
  py::buffer_info new_buf(bf16.data(), 2, py::format_descriptor<uint16_t>::format(), buf.ndim, buf.shape,
                          buf.strides);
  bf16s_.push_back(std::move(bf16));
  return new_buf;
}

void ConvertToFP32(py::buffer_info &bf16_buf, py::buffer_info &fp32_buf) {
  size_t size = bf16_buf.size;
  float *fp32 = static_cast<float *>(fp32_buf.ptr);
  BF16ToF32(reinterpret_cast<uint16_t *>(bf16_buf.ptr), fp32, size);
  return;
}

void akg_ascend_run(std::string path, std::string kernel_name, int device_id, bool is_dynamic, const py::args &args) {
  // 1. we get input_tensor and output tensor
  auto input_tensors = std::vector<mlir::runtime::TensorDevicePtr>();
  auto input_shapes = std::vector<std::vector<int64_t>>();
  std::map<uint64_t, py::buffer_info> bf16_buf_map;

  for (uint16_t i = 0; i < args.size(); i++) {
    auto tensor_obj_ptr = args[i].cast<AscendTensorObjStructPyTorchPtr>();
    auto tensor = tensor_obj_ptr->tensor_info;
    auto is_bf16 = (bool)(tensor_obj_ptr->is_bf16);

    void* data_addr = tensor_obj_ptr->data_ptr();
    if (is_bf16 && py::isinstance<py::buffer>(tensor)) {
      py::buffer_info buffer_info = py::cast<py::buffer>(tensor).request();
      buffer_info = ConvertToBF16(buffer_info);
      data_addr = buffer_info.ptr;
      bf16_buf_map[i] = std::move(buffer_info);
    }
    auto bytes = (uint64_t)(tensor_obj_ptr->nbytes);
    auto is_output = (bool)(tensor_obj_ptr->is_output);
    auto is_host = (bool)(tensor_obj_ptr->is_host());
    if (is_host) {
      input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_addr, nullptr, bytes, is_output));
    } else {
      input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(nullptr, data_addr, bytes, is_output));
    }

    if (is_dynamic) {
      auto input_shape = std::vector<int64_t>();
      py::buffer_info shape_info = tensor_obj_ptr->shape_info.request();
      int64_t *shape = reinterpret_cast<int64_t *>(shape_info.ptr);
      for (size_t idx = 0; idx < shape_info.size; idx++)
        input_shape.push_back(reinterpret_cast<int64_t>(shape[idx]));
      input_shapes.push_back(input_shape);
    }
  }

  int64_t tiling_key;
  int64_t tiling_struct_size;
  std::vector<void *> runtimeargs;
  if (is_dynamic) {
    std::string so_path = path + "/lib" + kernel_name + ".so";
    void *handle = dlopen(so_path.data(), RTLD_LAZY);
    if (!handle) {
      std::cerr << "Failed to load library: " << dlerror() << std::endl;
      return;
    }

    std::string tiling_struct_size_name = kernel_name + "_get_tiling_struct_size_function";
    get_tiling_struct_size_function_t get_tiling_struct_size_function =
        (get_tiling_struct_size_function_t)dlsym(handle, tiling_struct_size_name.data());
    tiling_struct_size = get_tiling_struct_size_function();
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      std::cerr << "Failed to load symbol: " << dlsym_error << std::endl;
      dlclose(handle);
      return;
    }

    if (tiling_struct_size > 0) {
      int64_t offset = 0;
      std::string tiling_function_name = kernel_name + "_tiling_function";
      tiling_function_t tiling_function = (tiling_function_t)dlsym(handle, tiling_function_name.data());
      dlsym_error = dlerror();
      if (dlsym_error) {
        std::cerr << "Failed to load symbol: " << dlsym_error << std::endl;
        dlclose(handle);
        return;
      }
      int64_t* arg_tiling_host = static_cast<int64_t*>(aligned_alloc(8, tiling_struct_size * sizeof(int64_t)));
      for (size_t idx = 0; idx < input_tensors.size(); idx++) {
        auto tensor = input_tensors[idx];
        auto shape = input_shapes[idx];
        runtimeargs.push_back(tensor->GetDeviceAddress());
        runtimeargs.push_back(tensor->GetDeviceAddress());
        runtimeargs.push_back(reinterpret_cast<void*>(offset));

        int64_t size = 1;
        for (auto dim : shape) {
          runtimeargs.push_back(reinterpret_cast<void*>(dim));
          size *= dim;
        }
        for (auto& dim : shape) {
          int64_t stride = size / dim;
          runtimeargs.push_back(reinterpret_cast<void*>(stride));
          size = stride;
        }
      }
      runtimeargs.push_back(reinterpret_cast<void*>(&tiling_key));
      runtimeargs.push_back(reinterpret_cast<void*>(arg_tiling_host));
      runtimeargs.push_back(reinterpret_cast<void*>(arg_tiling_host));
      runtimeargs.push_back(reinterpret_cast<void*>(offset));
      runtimeargs.push_back(reinterpret_cast<void*>(tiling_struct_size));
      runtimeargs.push_back(reinterpret_cast<void*>(1));
      tiling_function((void*)(runtimeargs.data()));
      input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(arg_tiling_host,
            nullptr, tiling_struct_size * sizeof(int64_t), false));
    }
  }

  auto kernel_runtime = mlir::runtime::AscendKernelRuntime(device_id);
  kernel_runtime.RunOpImpl(path, kernel_name, is_dynamic, input_tensors, input_shapes, tiling_key, tiling_struct_size);

  for (auto iter = bf16_buf_map.begin(); iter != bf16_buf_map.end(); iter++) {
    auto tensor_obj_ptr = args[iter->first].cast<AscendTensorObjStructPyTorchPtr>();
    py::buffer_info res_buf = py::cast<py::buffer>(tensor_obj_ptr->tensor_info).request();
    ConvertToFP32(bf16_buf_map[iter->first], res_buf);
  }
  return;
}

// PYBIND interface
PYBIND11_MODULE(akgAscendLaunch, m) {
  py::class_<AscendTensorObjStructPyTorch,
    std::shared_ptr<AscendTensorObjStructPyTorch>>(m, "AscendTensorObjStructPyTorch")
    .def(py::init<>())
    .def_readwrite("tensor_info", &AscendTensorObjStructPyTorch::tensor_info)
    .def_readwrite("shape_info", &AscendTensorObjStructPyTorch::shape_info)
    .def_readwrite("nbytes", &AscendTensorObjStructPyTorch::nbytes)
    .def_readwrite("is_output", &AscendTensorObjStructPyTorch::is_output)
    .def_readwrite("is_bf16", &AscendTensorObjStructPyTorch::is_bf16)
    .def("set_value", &AscendTensorObjStructPyTorch::set_value)
    .def("data_ptr", &AscendTensorObjStructPyTorch::data_ptr);
  // ascend_run call
  m.def("akg_ascend_run", &akg_ascend_run);
}
