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

#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <iostream>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "akg/ExecutionEngine/AscendLaunchRuntime/AscendRun.h"
#include "akg/ExecutionEngine/AscendLaunchRuntime/logger.h"

namespace {
using TilingFunc = uint64_t (*)(void *);
using GetTilingSizeFunc = uint64_t (*)();
using TorchRunFunc = void (*)(const char *, std::function<int()>);
constexpr auto kTilingMemSize = 1024;
constexpr auto kTilingSizeFuncName = "_get_tiling_struct_size_function";
constexpr auto kTilingFuncName = "_tiling_function";
constexpr size_t kRuntimeArgCount = 6;
constexpr uint16_t kMsprofReportDataMagicNum = 0x5a5a;
constexpr uint16_t kMsprofReportNodeLevel = 10000;
constexpr uint32_t kMsprofReportNodeBasicInfoType = 0;
constexpr uint32_t kMsprofReportNodeLaunchType = 5;
constexpr uint32_t kMsprofGeTaskTypeAiCore = 0;
constexpr uint32_t kMsprofCompactInfoDataLength = 40;

struct AkgMsprofApi {
  uint16_t magicNumber;
  uint16_t level;
  uint32_t type;
  uint32_t threadId;
  uint32_t reserve;
  uint64_t beginTime;
  uint64_t endTime;
  uint64_t itemId;
};

#pragma pack(push, 1)
struct AkgMsprofNodeBasicInfo {
  uint64_t opName;
  uint32_t taskType;
  uint64_t opType;
  uint32_t blockDim;
  uint32_t opFlag;
};
#pragma pack(pop)

struct AkgMsprofCompactInfo {
  uint16_t magicNumber;
  uint16_t level;
  uint32_t type;
  uint32_t threadId;
  uint32_t dataLen;
  uint64_t timeStamp;
  union {
    uint8_t info[kMsprofCompactInfoDataLength];
    AkgMsprofNodeBasicInfo nodeBasicInfo;
  } data;
};

using MsprofSysCycleTimeFunc = uint64_t (*)();
using MsprofGetHashIdFunc = uint64_t (*)(const char *, size_t);
using MsprofReportApiFunc = int32_t (*)(uint32_t, const AkgMsprofApi *);
using MsprofReportCompactInfoFunc = int32_t (*)(uint32_t, const void *, uint32_t);

struct AkgMsprofReporter {
  MsprofSysCycleTimeFunc sys_cycle_time{nullptr};
  MsprofGetHashIdFunc get_hash_id{nullptr};
  MsprofReportApiFunc report_api{nullptr};
  MsprofReportCompactInfoFunc report_compact_info{nullptr};

  bool IsAvailable() const {
    return sys_cycle_time != nullptr && get_hash_id != nullptr && report_api != nullptr &&
           report_compact_info != nullptr;
  }
};

void *LoadMsprofSymbol(const char *symbol) {
  void *func = dlsym(RTLD_DEFAULT, symbol);
  if (func != nullptr) {
    return func;
  }
  const char *libs[] = {"libmsprofiler.so", "libprofapi.so", "libascendcl.so"};
  for (auto lib : libs) {
    void *handle = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      continue;
    }
    func = dlsym(handle, symbol);
    if (func != nullptr) {
      return func;
    }
  }
  return nullptr;
}

const AkgMsprofReporter &GetAkgMsprofReporter() {
  static AkgMsprofReporter reporter;
  static std::once_flag once;
  std::call_once(once, []() {
    reporter.sys_cycle_time = reinterpret_cast<MsprofSysCycleTimeFunc>(LoadMsprofSymbol("MsprofSysCycleTime"));
    reporter.get_hash_id = reinterpret_cast<MsprofGetHashIdFunc>(LoadMsprofSymbol("MsprofGetHashId"));
    reporter.report_api = reinterpret_cast<MsprofReportApiFunc>(LoadMsprofSymbol("MsprofReportApi"));
    reporter.report_compact_info =
      reinterpret_cast<MsprofReportCompactInfoFunc>(LoadMsprofSymbol("MsprofReportCompactInfo"));
  });
  return reporter;
}

void ReportAkgDynamicLaunch(const std::string &kernel_name, uint64_t block_num, uint64_t begin_time,
                            uint64_t end_time) {
  const auto &reporter = GetAkgMsprofReporter();
  if (!reporter.IsAvailable()) {
    return;
  }
  auto thread_id = static_cast<uint32_t>(syscall(SYS_gettid));
  auto op_name = reporter.get_hash_id(kernel_name.c_str(), kernel_name.size());

  AkgMsprofApi api{};
  api.magicNumber = kMsprofReportDataMagicNum;
  api.level = kMsprofReportNodeLevel;
  api.type = kMsprofReportNodeLaunchType;
  api.threadId = thread_id;
  api.beginTime = begin_time;
  api.endTime = end_time;
  api.itemId = op_name;
  reporter.report_api(0, &api);

  AkgMsprofCompactInfo node_basic_info{};
  std::memset(node_basic_info.data.info, 0, sizeof(node_basic_info.data.info));
  node_basic_info.magicNumber = kMsprofReportDataMagicNum;
  node_basic_info.level = kMsprofReportNodeLevel;
  node_basic_info.type = kMsprofReportNodeBasicInfoType;
  node_basic_info.threadId = thread_id;
  node_basic_info.timeStamp = end_time;
  node_basic_info.data.nodeBasicInfo.opName = op_name;
  node_basic_info.data.nodeBasicInfo.taskType = kMsprofGeTaskTypeAiCore;
  node_basic_info.data.nodeBasicInfo.opType = op_name;
  node_basic_info.data.nodeBasicInfo.blockDim = static_cast<uint32_t>(block_num);
  node_basic_info.data.nodeBasicInfo.opFlag = 0;
  reporter.report_compact_info(0, &node_basic_info, sizeof(AkgMsprofCompactInfo));
}

class AscendLaunchTilingMemory {
 public:
  AscendLaunchTilingMemory() = default;
  std::unique_ptr<int64_t[]> tiling_host;
  static AscendLaunchTilingMemory *GetInstance();

 private:
  static std::shared_ptr<AscendLaunchTilingMemory> tiling_memory_singleton_;
};

std::shared_ptr<AscendLaunchTilingMemory> AscendLaunchTilingMemory::tiling_memory_singleton_ = nullptr;

AscendLaunchTilingMemory *AscendLaunchTilingMemory::GetInstance() {
  static std::once_flag tiling_mem_once;
  std::call_once(tiling_mem_once, []() {
    tiling_memory_singleton_.reset(new AscendLaunchTilingMemory());
    tiling_memory_singleton_.get()->tiling_host.reset(new int64_t[kTilingMemSize]);
  });
  return tiling_memory_singleton_.get();
}

mlir::runtime::BaseDevicePtr CreateScalarDevice(const py::handle &arg) {
  void *data_ptr = nullptr;

  if (py::isinstance<py::int_>(arg)) {
    auto val = arg.cast<int64_t>();
    data_ptr = reinterpret_cast<void *>(val);  // NOLINT
  } else if (py::isinstance<py::float_>(arg)) {
    auto val = arg.cast<double>();
    static_assert(sizeof(double) == sizeof(void *), "double size mismatch");
    std::memcpy(&data_ptr, &val, sizeof(void *));
  } else if (py::isinstance<py::bool_>(arg)) {
    bool val = arg.cast<bool>();
    data_ptr = reinterpret_cast<void *>(static_cast<intptr_t>(val));  // NOLINT
  }

  return std::make_shared<mlir::runtime::ScalarDevice>(data_ptr);
}

void ParseInputArgs(bool is_dynamic, std::vector<mlir::runtime::BaseDevicePtr> &input,
                    std::vector<std::vector<int64_t>> &input_shapes, const py::list &processed_args) {
  /* Tensor also implements buffer protocol; check tensor first for device ptr. */
  auto is_tensor_arg = [](const py::handle &h) { return py::hasattr(h, "data_ptr") && py::hasattr(h, "nbytes"); };
  auto is_numpy_arg = [](const py::handle &h) { return py::isinstance<py::buffer>(h) && !py::hasattr(h, "data_ptr"); };

  for (const auto &processed_arg : processed_args) {
    auto tup = processed_arg.cast<py::tuple>();
    auto data = tup[0].cast<py::object>();
    bool is_output = tup[1].cast<bool>();
    py::object shape_obj = tup[2];

    if (py::isinstance<py::int_>(data) || py::isinstance<py::float_>(data) || py::isinstance<py::bool_>(data)) {
      input.push_back(CreateScalarDevice(data));
      if (is_dynamic) {
        input_shapes.emplace_back();
      }
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
      data_addr = buffer_info.ptr;
      use_host = true;
    } else {
      throw std::runtime_error("processed_args element must be numpy, tensor, or scalar");
    }

    if (is_dynamic) {
      auto shape_list = shape_obj.cast<py::list>();
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

void ProcessTilingInputArg(mlir::runtime::BaseDevicePtr base, const std::vector<int64_t> &shape, int64_t offset,
                           std::vector<void *> &runtimeargs) {
  auto tensor = mlir::runtime::AsTensorDevice(base);
  if (!tensor) {
    runtimeargs.push_back(mlir::runtime::GetScalarValuePtr(base));
    return;
  }
  void *dev_addr = tensor->GetDeviceAddress();
  void *host_addr = tensor->GetHostAddress();
  void *eff_addr = ((dev_addr != nullptr) ? dev_addr : host_addr);
  runtimeargs.push_back(eff_addr);
  runtimeargs.push_back(eff_addr);
  runtimeargs.push_back(reinterpret_cast<void *>(offset));  // NOLINT
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

static void processTilingWithSize(const std::string &kernel_name, void *handle,
                                  std::vector<mlir::runtime::BaseDevicePtr> &input,
                                  const std::vector<std::vector<int64_t>> &input_shapes, int64_t &tiling_key,
                                  uint64_t tiling_size, std::vector<void *> &runtimeargs) {
  int64_t offset = 0;
  std::string tiling_func_name = kernel_name + kTilingFuncName;
  void *tiling_func = dlsym(handle, tiling_func_name.data());
  CHECK(tiling_func != nullptr) << "dlsym failed, symbol: " << tiling_func_name << " error:" << dlerror();
  auto tiling_func_ptr = reinterpret_cast<TilingFunc>(tiling_func);  // NOLINT

  for (size_t idx = 0; idx < input.size(); idx++) {
    ProcessTilingInputArg(input[idx], input_shapes[idx], offset, runtimeargs);
  }
  DLOG(INFO) << "Tiling args - tiling_key: " << tiling_key << ", offset: " << offset
             << ", tiling_size: " << tiling_size;
  runtimeargs.push_back(static_cast<void *>(&tiling_key));
  runtimeargs.push_back(static_cast<void *>(AscendLaunchTilingMemory::GetInstance()->tiling_host.get()));
  runtimeargs.push_back(static_cast<void *>(AscendLaunchTilingMemory::GetInstance()->tiling_host.get()));
  runtimeargs.push_back(reinterpret_cast<void *>(offset));       // NOLINT
  runtimeargs.push_back(reinterpret_cast<void *>(tiling_size));  // NOLINT
  runtimeargs.push_back(reinterpret_cast<void *>(1));            // NOLINT
  tiling_func_ptr(reinterpret_cast<void *>(runtimeargs.data()));
  for (uint64_t i = 0; i < tiling_size; i++) {
    DLOG(INFO) << "tiling data[" << i << "]: " << (AscendLaunchTilingMemory::GetInstance()->tiling_host.get())[i];
  }
  input.push_back(std::make_shared<mlir::runtime::TensorDevice>(
    AscendLaunchTilingMemory::GetInstance()->tiling_host.get(), nullptr, tiling_size * sizeof(int64_t), false));
}

void ProcessDynamicTiling(const std::string &path, const std::string &kernel_name,
                          std::vector<mlir::runtime::BaseDevicePtr> &input,
                          const std::vector<std::vector<int64_t>> &input_shapes, int64_t &tiling_key,
                          uint64_t &tiling_size, std::vector<void *> &runtimeargs) {
  std::string so_path = path + "/lib" + kernel_name + ".so";
  void *handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  CHECK(handle != nullptr) << "dlopen failed, file: " << so_path << ", Error:" << dlerror();

  std::string tiling_size_func_name = kernel_name + kTilingSizeFuncName;
  void *tiling_size_func = dlsym(handle, tiling_size_func_name.data());
  CHECK(tiling_size_func != nullptr) << "dlsym failed, symbol: " << tiling_size_func_name << " error:" << dlerror();
  tiling_size = reinterpret_cast<GetTilingSizeFunc>(tiling_size_func)();  // NOLINT
  if (tiling_size > 0) {
    processTilingWithSize(kernel_name, handle, input, input_shapes, tiling_key, tiling_size, runtimeargs);
  }
}

void akg_ascend_run(std::string path, std::string kernel_name, int device_id, bool is_dynamic, bool use_mem_pool,
                    const py::args &args, py::kwargs kwargs) {
  void *external_stream = nullptr;
  if (kwargs.contains("stream") && !kwargs["stream"].is_none()) {
    auto h = kwargs["stream"].cast<intptr_t>();
    external_stream =
      (h == 0) ? reinterpret_cast<void *>(static_cast<uintptr_t>(-1)) : reinterpret_cast<void *>(h);  // NOLINT
  }
  auto processed_args = kwargs["processed_args"].cast<py::list>();
  auto input = std::vector<mlir::runtime::BaseDevicePtr>();
  auto input_shapes = std::vector<std::vector<int64_t>>();

  ParseInputArgs(is_dynamic, input, input_shapes, processed_args);

  int64_t tiling_key;
  uint64_t tiling_size = 0;
  std::vector<void *> runtimeargs;
  if (is_dynamic) {
    use_mem_pool = true;
    ProcessDynamicTiling(path, kernel_name, input, input_shapes, tiling_key, tiling_size, runtimeargs);
  }
  for (auto iter = runtimeargs.begin(); iter != runtimeargs.end(); iter++) {
    DLOG(INFO) << "runtimeargs[" << iter - runtimeargs.begin() << "]: " << *iter;
  }

  auto *kernel_runtime =
    mlir::runtime::AscendLaunchRuntime::GetOrCreateRuntime(device_id, use_mem_pool, external_stream);
  kernel_runtime->RunOpImpl(path, kernel_name, is_dynamic, input, input_shapes, tiling_key, tiling_size);
  return;
}

void *GetPointer(py::object arg) {
  if (py::isinstance<py::int_>(arg)) {
    auto val = arg.cast<int64_t>();
    return reinterpret_cast<void *>(val);  // NOLINT
  }
  if (py::isinstance<py::float_>(arg)) {
    double val = arg.cast<double>();
    return reinterpret_cast<void *>(static_cast<intptr_t>(val));  // NOLINT
  } else if (py::isinstance<py::bool_>(arg)) {
    bool val = arg.cast<bool>();
    return reinterpret_cast<void *>(static_cast<intptr_t>(val));  // NOLINT
  } else if (py::hasattr(arg, "data_ptr")) {
    return reinterpret_cast<void *>(arg.attr("data_ptr")().cast<intptr_t>());  // NOLINT
  }
  return nullptr;
}

py::tuple GetHostFunctions(std::string kernel_name, std::string lib_path) {
  std::string tiling_size_func_name = kernel_name + kTilingSizeFuncName;
  std::string tiling_func_name = kernel_name + kTilingFuncName;

  void *handle = dlopen(lib_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  CHECK(handle != nullptr) << "dlopen failed, file: " << lib_path << ", Error:" << dlerror();

  void *kernel_func = dlsym(handle, kernel_name.data());
  CHECK(kernel_func != nullptr) << "dlsym failed, symbol: " << kernel_name << " error:" << dlerror();

  void *tiling_func = dlsym(handle, tiling_func_name.data());
  CHECK(tiling_func != nullptr) << "dlsym failed, symbol: " << tiling_func_name << " error:" << dlerror();

  void *tiling_size_func = dlsym(handle, tiling_size_func_name.data());
  CHECK(tiling_size_func != nullptr) << "dlsym failed, symbol: " << tiling_size_func_name << " error:" << dlerror();
  uint64_t tiling_size = reinterpret_cast<GetTilingSizeFunc>(tiling_size_func)();  // NOLINT

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
    runtimeargs.push_back(static_cast<void *>(&tiling_key));
    runtimeargs.push_back(static_cast<void *>(AscendLaunchTilingMemory::GetInstance()->tiling_host.get()));
    runtimeargs.push_back(static_cast<void *>(AscendLaunchTilingMemory::GetInstance()->tiling_host.get()));
    runtimeargs.push_back(reinterpret_cast<void *>(offset));           // NOLINT
    runtimeargs.push_back(reinterpret_cast<void *>(tiling_size));      // NOLINT
    runtimeargs.push_back(reinterpret_cast<void *>(1));                // NOLINT
    auto tiling_func_ptr = reinterpret_cast<TilingFunc>(tiling_func);  // NOLINT
    tiling_func_ptr(reinterpret_cast<void *>(runtimeargs.data()));
    runtimeargs.resize(runtimeargs.size() - kRuntimeArgCount);
    runtimeargs.push_back(reinterpret_cast<void *>(tiling_key));  // NOLINT
    for (uint64_t i = 0; i < tiling_size; i++) {
      DLOG(INFO) << "tiling data[" << i << "]: " << (AscendLaunchTilingMemory::GetInstance()->tiling_host.get())[i];
      runtimeargs.push_back(
        reinterpret_cast<void *>((AscendLaunchTilingMemory::GetInstance()->tiling_host.get())[i]));  // NOLINT
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
    std::string so_path = torch_path + "/lib/libtorch_npu.so";
    std::string func_name = "_Z14opcommand_callPKcSt8functionIFivEE";
    void *handle = dlopen(so_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
    CHECK(handle != nullptr) << "dlopen failed, file: " << so_path << ", Error:" << dlerror();
    torch_run_func = dlsym(handle, func_name.data());
    CHECK(torch_run_func != nullptr) << "dlsym failed, symbol: opcommand_call, error:" << dlerror();
  }

  auto launch_call = [kernel_name, kernel_func, block_num, stream, runtimeargs, is_dynamic] {
    const auto &reporter = GetAkgMsprofReporter();
    uint64_t begin_time = (is_dynamic && reporter.IsAvailable()) ? reporter.sys_cycle_time() : 0;
    mlir::runtime::KernelLaunch(kernel_func, block_num, (rtStream_t)stream, runtimeargs, is_dynamic);
    if (is_dynamic && reporter.IsAvailable()) {
      ReportAkgDynamicLaunch(kernel_name, block_num, begin_time, reporter.sys_cycle_time());
    }
    return 0;
  };
  reinterpret_cast<TorchRunFunc>(torch_run_func)(kernel_name.c_str(), launch_call);
}
}  // namespace

// PYBIND interface
PYBIND11_MODULE(ascend_launch, m) {
  mlir::runtime::akg_log_init();
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging("akg");
  }
  // ascend_run call
  m.def("akg_ascend_run", &akg_ascend_run);
  m.def("get_host_functions", &GetHostFunctions);
  m.def("get_device_function", &mlir::runtime::GetKernelFunction);
  m.def("launch", py::overload_cast<uint64_t, uint64_t, uint64_t, uint64_t, uint64_t, bool, const py::args &>(&Launch),
        "Launch kernel with tiling support");
  m.def("launch", py::overload_cast<uint64_t, uint64_t, uint64_t, bool, const py::args &>(&Launch),
        "Launch kernel without tiling support");
  m.def("torch_launch", &TorchLaunch, "Launch kernel for torch_npu");
}
