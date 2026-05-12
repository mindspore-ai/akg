/**
 * Copyright 2024-2026 Huawei Technologies Co., Ltd
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
#include <cstdint>
#include <iterator>
#include <iostream>
#include <fstream>
#include <mutex>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AKGAscendRun.h"

typedef uint64_t (*TilingFunc)(void *);
typedef uint64_t (*GetTilingSizeFunc)();
typedef void (*TorchRunFunc)(char const *, std::function<int ()>);
constexpr auto kTilingMemSize = 1024;
constexpr auto kTilingSizeFuncName = "_get_tiling_struct_size_function";
constexpr auto kTilingFuncName = "_tiling_function";

class TilingMemory {
 public:
  TilingMemory() = default;
  std::unique_ptr<int64_t[]> tiling_host;
  static TilingMemory *GetInstance();

 private:
  static std::shared_ptr<TilingMemory> tiling_memory_singleton_;
};

std::shared_ptr<TilingMemory> TilingMemory::tiling_memory_singleton_ = nullptr;

TilingMemory *TilingMemory::GetInstance() {
  static std::once_flag tiling_mem_once;
  std::call_once(tiling_mem_once, []() {
    tiling_memory_singleton_.reset(new TilingMemory());
    tiling_memory_singleton_.get()->tiling_host.reset(new int64_t[kTilingMemSize]);
  });
  return tiling_memory_singleton_.get();
}

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
  py::buffer_info new_buf(bf16.data(), 2, py::format_descriptor<uint16_t>::format(), buf.ndim, buf.shape, buf.strides);
  bf16s_.push_back(std::move(bf16));
  return new_buf;
}

void ConvertToFP32(py::buffer_info &bf16_buf, py::buffer_info &fp32_buf) {
  size_t size = bf16_buf.size;
  float *fp32 = static_cast<float *>(fp32_buf.ptr);
  BF16ToF32(reinterpret_cast<uint16_t *>(bf16_buf.ptr), fp32, size);
  return;
}

mlir::runtime::BaseDevicePtr CreateScalarDevice(const py::handle &arg) {
  void *data_ptr = nullptr;

  if (py::isinstance<py::int_>(arg)) {
    int64_t val = arg.cast<int64_t>();
    data_ptr = reinterpret_cast<void *>(val);
  } else if (py::isinstance<py::float_>(arg)) {
    double val = arg.cast<double>();
    static_assert(sizeof(double) == sizeof(void *), "double size mismatch");
    std::memcpy(&data_ptr, &val, sizeof(void *));
  } else if (py::isinstance<py::bool_>(arg)) {
    bool val = arg.cast<bool>();
    data_ptr = reinterpret_cast<void *>(static_cast<intptr_t>(val));
  }

  return std::make_shared<mlir::runtime::ScalarDevice>(data_ptr);
}

void ParseInputArgs(bool is_dynamic, std::vector<mlir::runtime::BaseDevicePtr> &input,
                    std::vector<std::vector<int64_t>> &input_shapes, std::map<uint64_t, py::buffer_info> &bf16_buf_map,
                    const py::list &processed_args) {
  /* Tensor also implements buffer protocol; check tensor first for device ptr. */
  auto is_tensor_arg = [](const py::handle &h) { return py::hasattr(h, "data_ptr") && py::hasattr(h, "nbytes"); };
  auto is_numpy_arg = [](const py::handle &h) { return py::isinstance<py::buffer>(h) && !py::hasattr(h, "data_ptr"); };

  for (uint16_t i = 0; i < processed_args.size(); i++) {
    py::tuple tup = processed_args[i].cast<py::tuple>();
    py::object data = tup[0].cast<py::object>();
    bool is_output = tup[1].cast<bool>();
    bool is_bf16 = tup[2].cast<bool>();
    py::object shape_obj = tup[3];

    if (py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data) || py::isinstance<py::bool_>(data)) {
      input.push_back(CreateScalarDevice(data));
      if (is_dynamic) input_shapes.emplace_back();
      continue;
    }

    void *data_addr = nullptr;
    uint64_t bytes = 0;
    bool use_host = true;

    if (is_tensor_arg(data)) {
      data_addr = reinterpret_cast<void *>(data.attr("data_ptr")().cast<intptr_t>());
      bytes = data.attr("nbytes").cast<uint64_t>();
      use_host = false;
    } else if (is_numpy_arg(data)) {
      py::buffer_info buffer_info = py::cast<py::buffer>(data).request();
      bytes = static_cast<uint64_t>(buffer_info.size * buffer_info.itemsize);
      if (is_bf16 && buffer_info.itemsize == 4) {
        py::buffer_info bf16_buffer = ConvertToBF16(buffer_info);
        data_addr = bf16_buffer.ptr;
        bf16_buf_map[i] = std::move(bf16_buffer);
        bytes /= 2;
      } else {
        data_addr = buffer_info.ptr;
      }
      use_host = true;
    } else {
      throw std::runtime_error("processed_args element must be numpy, tensor, or scalar");
    }

    if (is_dynamic) {
      py::list shape_list = shape_obj.cast<py::list>();
      std::vector<int64_t> input_shape;
      input_shape.reserve(shape_list.size());
      std::transform(shape_list.begin(), shape_list.end(), std::back_inserter(input_shape),
                     [](const py::handle &dim_h) { return dim_h.cast<int64_t>(); });
      input_shapes.push_back(std::move(input_shape));
    }

    if (use_host) {
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(data_addr, nullptr, bytes, is_output));
    } else {
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(nullptr, data_addr, bytes, is_output));
    }
  }
}

void akg_ascend_run(std::string path, std::string kernel_name, int device_id, bool is_dynamic, bool use_mem_pool,
                    const py::args &args, py::kwargs kwargs) {
  void *external_stream = nullptr;
  if (kwargs.contains("stream") && !kwargs["stream"].is_none()) {
    intptr_t h = kwargs["stream"].cast<intptr_t>();
    external_stream = (h == 0) ? reinterpret_cast<void *>(static_cast<uintptr_t>(-1)) : reinterpret_cast<void *>(h);
  }
  py::list processed_args = kwargs["processed_args"].cast<py::list>();
  auto input = std::vector<mlir::runtime::BaseDevicePtr>();
  auto input_shapes = std::vector<std::vector<int64_t>>();
  std::map<uint64_t, py::buffer_info> bf16_buf_map;

  ParseInputArgs(is_dynamic, input, input_shapes, bf16_buf_map, processed_args);

  int64_t tiling_key, tiling_size = 0;
  std::vector<void *> runtimeargs;
  if (is_dynamic) {
    use_mem_pool = true;
    std::string so_path = path + "/lib" + kernel_name + ".so";
    void *handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle != nullptr) << "dlopen failed, file: " << so_path << ", Error:" << dlerror();

    std::string tiling_size_func_name = kernel_name + kTilingSizeFuncName;

    void *tiling_size_func = dlsym(handle, tiling_size_func_name.c_str());
    CHECK(tiling_size_func != nullptr) << "dlsym failed, symbol: " << tiling_size_func_name << " error:" << dlerror();
    tiling_size = reinterpret_cast<GetTilingSizeFunc>(tiling_size_func)();

    if (tiling_size > 0) {
      int64_t offset = 0;
      std::string tiling_func_name = kernel_name + kTilingFuncName;
      void *tiling_func = dlsym(handle, tiling_func_name.c_str());
      CHECK(tiling_func != nullptr) << "dlsym failed, symbol: " << tiling_func_name << " error:" << dlerror();
      auto tiling_func_ptr = reinterpret_cast<TilingFunc>(tiling_func);

      for (size_t idx = 0; idx < input.size(); idx++) {
        auto base = input[idx];
        auto shape = input_shapes[idx];

        auto tensor = mlir::runtime::AsTensorDevice(base);
        if (!tensor) {
          runtimeargs.push_back(mlir::runtime::GetScalarValuePtr(base));
        } else {
          void *dev_addr = tensor->GetDeviceAddress();
          void *host_addr = tensor->GetHostAddress();
          void *eff_addr = (dev_addr ? dev_addr : host_addr);
          runtimeargs.push_back(eff_addr);
          runtimeargs.push_back(eff_addr);
          runtimeargs.push_back(reinterpret_cast<void *>(offset));
          int64_t size = 1;
          for (auto dim : shape) {
            runtimeargs.push_back(reinterpret_cast<void *>(dim));
            size *= dim;
          }

          for (auto &dim : shape) {
            int64_t stride = size / dim;
            runtimeargs.push_back(reinterpret_cast<void *>(stride));
            size = stride;
          }
        }
      }
      DLOG(INFO) << "Tiling args - tiling_key: " << tiling_key << ", offset: " << offset
                 << ", tiling_size: " << tiling_size;
      runtimeargs.push_back(reinterpret_cast<void *>(&tiling_key));
      runtimeargs.push_back(reinterpret_cast<void *>(TilingMemory::GetInstance()->tiling_host.get()));
      runtimeargs.push_back(reinterpret_cast<void *>(TilingMemory::GetInstance()->tiling_host.get()));
      runtimeargs.push_back(reinterpret_cast<void *>(offset));
      runtimeargs.push_back(reinterpret_cast<void *>(tiling_size));
      runtimeargs.push_back(reinterpret_cast<void *>(1));
      tiling_func_ptr((void *)(runtimeargs.data()));
      for (uint64_t i = 0; i < tiling_size; i++) {
        DLOG(INFO) << "tiling data[" << i << "]: " << (TilingMemory::GetInstance()->tiling_host.get())[i];
      }
      input.push_back(std::make_shared<mlir::runtime::TensorDevice>(TilingMemory::GetInstance()->tiling_host.get(),
                                                                    nullptr, tiling_size * sizeof(int64_t), false));
    }
  }
  for (auto iter = runtimeargs.begin(); iter != runtimeargs.end(); iter++) {
    DLOG(INFO) << "runtimeargs[" << iter - runtimeargs.begin() << "]: " << *iter;
  }

  auto *kernel_runtime =
    mlir::runtime::AscendKernelRuntime::GetOrCreateRuntime(device_id, use_mem_pool, external_stream);
  kernel_runtime->RunOpImpl(path, kernel_name, is_dynamic, input, input_shapes, tiling_key, tiling_size);

  for (auto iter = bf16_buf_map.begin(); iter != bf16_buf_map.end(); iter++) {
    py::tuple tup = processed_args[iter->first].cast<py::tuple>();
    py::object data_src = tup[0].cast<py::object>();
    py::buffer_info res_buf = py::cast<py::buffer>(data_src).request();
    ConvertToFP32(bf16_buf_map[iter->first], res_buf);
  }
  return;
}

void *GetPointer(py::object arg) {
  if (py::isinstance<py::int_>(arg)) {
    int64_t val = arg.cast<int64_t>();
    return reinterpret_cast<void *>(val);
  } else if (py::isinstance<py::float_>(arg)) {
    double val = arg.cast<double>();
    return reinterpret_cast<void *>(static_cast<intptr_t>(val));
  } else if (py::isinstance<py::bool_>(arg)) {
    bool val = arg.cast<bool>();
    return reinterpret_cast<void *>(static_cast<intptr_t>(val));
  } else if (py::hasattr(arg, "data_ptr")) {
    return reinterpret_cast<void *>(arg.attr("data_ptr")().cast<intptr_t>());
  }
  return nullptr;
}

py::tuple GetHostFunctions(std::string kernel_name, std::string lib_path) {
  std::string tiling_size_func_name = kernel_name + kTilingSizeFuncName;
  std::string tiling_func_name = kernel_name + kTilingFuncName;

  void *handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  CHECK(handle != nullptr) << "dlopen failed, file: " << lib_path << ", Error:" << dlerror();

  void *kernel_func = dlsym(handle, kernel_name.c_str());
  CHECK(kernel_func != nullptr) << "dlsym failed, symbol: " << kernel_name << " error:" << dlerror();

  void *tiling_func = dlsym(handle, tiling_func_name.c_str());
  CHECK(tiling_func != nullptr) << "dlsym failed, symbol: " << tiling_func_name << " error:" << dlerror();

  void *tiling_size_func = dlsym(handle, tiling_size_func_name.c_str());
  CHECK(tiling_size_func != nullptr) << "dlsym failed, symbol: " << tiling_size_func_name << " error:" << dlerror();
  uint64_t tiling_size = reinterpret_cast<GetTilingSizeFunc>(tiling_size_func)();

  return py::make_tuple(reinterpret_cast<uint64_t>(kernel_func), reinterpret_cast<uint64_t>(tiling_func),
                        reinterpret_cast<uint64_t>(tiling_size_func), tiling_size);
}

std::vector<void *> InitKernelArgs(const py::args &args, uint64_t tiling_func = 0, uint64_t tiling_size = 0) {
  std::vector<void *> runtimeargs;
  for (size_t idx = 0; idx < args.size(); idx++) {
    runtimeargs.push_back(GetPointer(args[idx].cast<py::object>()));
    DLOG(INFO) << "runtimeargs[" << idx << "]: " << runtimeargs[idx];
  }

  if (tiling_size > 0 && tiling_size < kTilingMemSize) {
    int64_t tiling_key;
    int64_t offset = 0;
    runtimeargs.push_back(reinterpret_cast<void *>(&tiling_key));
    runtimeargs.push_back(reinterpret_cast<void *>(TilingMemory::GetInstance()->tiling_host.get()));
    runtimeargs.push_back(reinterpret_cast<void *>(TilingMemory::GetInstance()->tiling_host.get()));
    runtimeargs.push_back(reinterpret_cast<void *>(offset));
    runtimeargs.push_back(reinterpret_cast<void *>(tiling_size));
    runtimeargs.push_back(reinterpret_cast<void *>(1));
    auto tiling_func_ptr = reinterpret_cast<TilingFunc>(tiling_func);
    tiling_func_ptr((void *)(runtimeargs.data()));
    runtimeargs.resize(runtimeargs.size() - 6);
    runtimeargs.push_back(reinterpret_cast<void *>(tiling_key));
    for (uint64_t i = 0; i < tiling_size; i++) {
      DLOG(INFO) << "tiling data[" << i << "]: " << (TilingMemory::GetInstance()->tiling_host.get())[i];
      runtimeargs.push_back(reinterpret_cast<void *>((TilingMemory::GetInstance()->tiling_host.get())[i]));
    }
  }
  return runtimeargs;
}

void Launch(uint64_t kernel_func, uint64_t tiling_func, uint64_t tiling_size, uint64_t block_num, uint64_t stream,
            bool is_dynamic, const py::args &args) {
  auto runtimeargs = InitKernelArgs(args, tiling_func, tiling_size);
  mlir::runtime::KernelLaunch(kernel_func, block_num, (rtStream_t)stream, runtimeargs, is_dynamic);
}

void Launch(uint64_t kernel_func, uint64_t block_num, uint64_t stream, bool is_dynamic, const py::args &args) {
  auto runtimeargs = InitKernelArgs(args);
  mlir::runtime::KernelLaunch(kernel_func, block_num, (rtStream_t)stream, runtimeargs, is_dynamic);
}

void TorchLaunch(std::string kernel_name, std::string torch_path, uint64_t kernel_func, uint64_t tiling_func,
                 uint64_t tiling_size, uint64_t block_num, uint64_t stream, bool is_dynamic, const py::args &args) {
  auto runtimeargs = InitKernelArgs(args, tiling_func, tiling_size);
  static void *torch_run_func = nullptr;
  if (torch_run_func == nullptr) {
    if (torch_path.empty()) {
      mlir::runtime::KernelLaunch(kernel_func, block_num, (rtStream_t)stream, runtimeargs, is_dynamic);
      return;
    }
    std::string so_path = torch_path + "/_inductor/ascend_npu_ir/ascend_npu_ir/lib/libcpp_common.so";
    std::string func_name = "_Z14opcommand_callPKcSt8functionIFivEE";
    void *handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle != nullptr) << "dlopen failed, file: " << so_path << ", Error:" << dlerror();
    torch_run_func = dlsym(handle, func_name.c_str());
    CHECK(torch_run_func != nullptr) << "dlsym failed, symbol: opcommand_call, error:"
                                     << dlerror();
  }

  auto launch_call = [kernel_func, block_num, stream, runtimeargs, is_dynamic] {
    mlir::runtime::KernelLaunch(kernel_func, block_num, (rtStream_t)stream, runtimeargs, is_dynamic);
    return 0;
  };
  reinterpret_cast<TorchRunFunc>(torch_run_func)(kernel_name.c_str(), launch_call);
}

// PYBIND interface
PYBIND11_MODULE(akgAscendLaunch, m) {
  py::class_<AscendTensorObjStructPyTorch, std::shared_ptr<AscendTensorObjStructPyTorch>>(
    m, "AscendTensorObjStructPyTorch")
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
  m.def("get_host_functions", &GetHostFunctions);
  m.def("get_device_function", &mlir::runtime::GetKernelFunction);
  m.def("launch", py::overload_cast<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool, const py::args &>(&Launch),
        "Launch kernel with tiling support");
  m.def("launch", py::overload_cast<uint64_t, uint64_t, uint64_t, bool, const py::args &>(&Launch),
        "Launch kernel without tiling support");
  m.def("torch_launch", &TorchLaunch, "Launch kernel for torch_npu");
  akg_log_init();
}
