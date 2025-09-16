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
  std::map<long unsigned int, py::buffer_info> bf16_buf_map;

  for (long unsigned int i = 0; i < args.size(); i++) {
    auto tensor_obj_ptr = args[i].cast<AscendTensorObjStructPtr>();
    py::buffer_info buffer_info = tensor_obj_ptr->buffer_info.request();
    auto is_bf16 = (bool)(tensor_obj_ptr->is_bf16);
    if (is_bf16) {
      buffer_info = ConvertToBF16(buffer_info);
      bf16_buf_map[i] = std::move(buffer_info);
    }
    auto data_addr = buffer_info.ptr;
    auto bytes = (unsigned long long)(tensor_obj_ptr->nbytes);
    auto is_output = (bool)(tensor_obj_ptr->is_output);
    input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_addr, bytes, is_output));
    // TODO: dynamic shape judge;

    if (is_dynamic) {
      auto input_shape = std::vector<int64_t>();
      py::buffer_info shape_info = tensor_obj_ptr->shape_info.request();
      int64_t *shape = reinterpret_cast<int64_t *>(shape_info.ptr);
      for (size_t idx = 0; idx < shape_info.size; idx++)
        input_shape.push_back(reinterpret_cast<int64_t>(shape[idx]));
      input_shapes.push_back(input_shape);
    }
  }

  int64_t offset = 0;
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
    get_tiling_struct_size_function_t get_tiling_struct_size_function = (get_tiling_struct_size_function_t)dlsym(handle, tiling_struct_size_name.data());
    tiling_struct_size = get_tiling_struct_size_function();
    const char* dlsym_error = dlerror();
    if (dlsym_error) {
      std::cerr << "Failed to load symbol: " << dlsym_error << std::endl;
      dlclose(handle);
      return;
    }

    if (tiling_struct_size > 0) {
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
      input_tensors.push_back(std::make_shared<mlir::runtime::TensorDevice>(arg_tiling_host, tiling_struct_size * sizeof(int64_t), false));
    }
  }

  auto kernel_runtime = mlir::runtime::AscendKernelRuntime(device_id);
  kernel_runtime.RunOpImpl(path, kernel_name, is_dynamic, input_tensors, input_shapes, tiling_key, tiling_struct_size);

  for(auto iter = bf16_buf_map.begin(); iter != bf16_buf_map.end(); iter++) {
    auto tensor_obj_ptr = args[iter->first].cast<AscendTensorObjStructPtr>();
    py::buffer_info res_buf = tensor_obj_ptr->buffer_info.request();
    ConvertToFP32(bf16_buf_map[iter->first], res_buf);
  }
  return;
}

// PYBIND interface
PYBIND11_MODULE(akgAscendLaunch, m) {
  py::class_<AscendTensorObjStruct, std::shared_ptr<AscendTensorObjStruct>>(m, "AscendTensorObjStruct")
    .def(py::init<>())
    .def_readwrite("buffer_info", &AscendTensorObjStruct::buffer_info)
    .def_readwrite("shape_info", &AscendTensorObjStruct::shape_info)
    .def_readwrite("nbytes", &AscendTensorObjStruct::nbytes)
    .def_readwrite("is_output", &AscendTensorObjStruct::is_output)
    .def_readwrite("is_dynamic", &AscendTensorObjStruct::is_dynamic)
    .def_readwrite("is_bf16", &AscendTensorObjStruct::is_bf16)
    .def("set_value", &AscendTensorObjStruct::set_value);

  // ascend_run call
  m.def("akg_ascend_run", &akg_ascend_run);
}
