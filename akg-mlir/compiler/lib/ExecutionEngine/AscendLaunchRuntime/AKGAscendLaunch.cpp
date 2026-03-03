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

mlir::runtime::BaseDevicePtr CreateScalarDevice(const py::handle& arg) {
  void* data_ptr = nullptr;

  if (py::isinstance<py::int_>(arg)) {
    int64_t val = arg.cast<int64_t>();
    data_ptr = reinterpret_cast<void*>(val);
  } else if (py::isinstance<py::float_>(arg)) {
    double val = arg.cast<double>();
    static_assert(sizeof(double) == sizeof(void*), "double size mismatch");
    std::memcpy(&data_ptr, &val, sizeof(void*));
  } else if (py::isinstance<py::bool_>(arg)) {
    bool val = arg.cast<bool>();
    data_ptr = reinterpret_cast<void*>(static_cast<intptr_t>(val));
  }

  return std::make_shared<mlir::runtime::ScalarDevice>(data_ptr);
}

void ParseInputArgs(bool is_dynamic,
                    std::vector<mlir::runtime::BaseDevicePtr> &input,
                    std::vector<std::vector<int64_t>> &input_shapes,
                    std::map<uint64_t, py::buffer_info> &bf16_buf_map,
                    const py::args &args) {
  auto is_tensor_arg = [](const py::handle &h) {
    return py::isinstance<AscendTensorObjStructPyTorch>(h);
  };

  for (uint16_t i = 0; i < args.size(); i++) {
    if (!is_tensor_arg(args[i])) {
      input.push_back(CreateScalarDevice(args[i]));
      if (is_dynamic) input_shapes.emplace_back();
      continue;
    }
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
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_addr, nullptr, bytes, is_output));
    } else {
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(nullptr, data_addr, bytes, is_output));
    }

    if (is_dynamic) {
      auto input_shape = std::vector<int64_t>();
      py::buffer_info shape_info = tensor_obj_ptr->shape_info.request();
      int64_t *shape = reinterpret_cast<int64_t *>(shape_info.ptr);
      for (int64_t idx = 0; idx < shape_info.size; idx++) {
        int64_t dim_value = reinterpret_cast<int64_t>(shape[idx]);
        input_shape.push_back(dim_value);
      }
      input_shapes.push_back(input_shape);
    }
  }
}

void akg_ascend_run(std::string path, std::string kernel_name, int device_id, bool is_dynamic,
                    bool use_mem_pool, const py::args &args, py::kwargs kwargs) {
  void *external_stream = nullptr;
  if (kwargs.contains("stream") && !kwargs["stream"].is_none()) {
    external_stream = reinterpret_cast<void *>(kwargs["stream"].cast<intptr_t>());
  }
  akg_log_init();
  auto input = std::vector<mlir::runtime::BaseDevicePtr>();
  auto input_shapes = std::vector<std::vector<int64_t>>();
  std::map<uint64_t, py::buffer_info> bf16_buf_map;

  ParseInputArgs(is_dynamic, input, input_shapes, bf16_buf_map, args);

  int64_t tiling_key;
  int64_t tiling_struct_size;
  std::vector<void *> runtimeargs;
  if (is_dynamic) {
    use_mem_pool = true;
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
      int64_t* arg_tiling_host = static_cast<int64_t*>(
          aligned_alloc(8, tiling_struct_size * sizeof(int64_t)));

      for (size_t idx = 0; idx < input.size(); idx++) {
        auto base = input[idx];
        auto shape = input_shapes[idx];

        auto tensor = mlir::runtime::AsTensorDevice(base);
        if (!tensor) {
          runtimeargs.push_back(mlir::runtime::GetScalarValuePtr(base));
        } else {
          void* dev_addr  = tensor->GetDeviceAddress();
          void* host_addr = tensor->GetHostAddress();
          void* eff_addr = (dev_addr ? dev_addr : host_addr);
          runtimeargs.push_back(eff_addr);
          runtimeargs.push_back(eff_addr);
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
      }
      DLOG(INFO) << "Tiling args - tiling_key: " << tiling_key
           << ", offset: " << offset
           << ", tiling_struct_size: " << tiling_struct_size;
      runtimeargs.push_back(reinterpret_cast<void*>(&tiling_key));
      runtimeargs.push_back(reinterpret_cast<void*>(arg_tiling_host));
      runtimeargs.push_back(reinterpret_cast<void*>(arg_tiling_host));
      runtimeargs.push_back(reinterpret_cast<void*>(offset));
      runtimeargs.push_back(reinterpret_cast<void*>(tiling_struct_size));
      runtimeargs.push_back(reinterpret_cast<void*>(1));
      tiling_function((void*)(runtimeargs.data()));
      for (int64_t i = 0; i < tiling_struct_size; i++) {
        DLOG(INFO) << "arg_tiling_host[" << i << "]: " << arg_tiling_host[i];
      }
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(arg_tiling_host,
            nullptr, tiling_struct_size * sizeof(int64_t), false));
    }
  }

  for (auto iter = runtimeargs.begin(); iter != runtimeargs.end(); iter++) {
    DLOG(INFO) << "runtimeargs[" << iter - runtimeargs.begin() << "]: " << *iter;
  }

  auto kernel_runtime = mlir::runtime::AscendKernelRuntime(device_id, use_mem_pool, external_stream);
  kernel_runtime.RunOpImpl(path, kernel_name, is_dynamic, input, input_shapes, tiling_key, tiling_struct_size);

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
