/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "akg/ExecutionEngine/AscendLaunchRuntime/ProfileMgr.h"

#include <sys/syscall.h>
#include <unistd.h>
#include <string>
#include "akg/ExecutionEngine/AscendLaunchRuntime/logger.h"

namespace mlir {
namespace runtime {

bool ProfileMgr::StartupProfiling(const std::string &profiling_dir, const uint32_t device_id) {
  LOG(INFO) << "Start to profile";

  int32_t current_device = -1;
  int ret = aclrtGetDevice(&current_device);
  if (ret != ACL_SUCCESS || current_device != static_cast<int32_t>(device_id)) {
    ret = aclrtSetDevice(device_id);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Failed to set device " << device_id << ", ret = " << ret;
      return false;
    }
  }

  aclrtContext context = nullptr;
  ret = aclrtGetCurrentContext(&context);
  if (ret != ACL_SUCCESS || context == nullptr) {
    LOG(ERROR) << "No valid ACL context on current thread!";
    LOG(ERROR) << "This may indicate ACL runtime is not properly initialized";
    return false;
  }

  uint32_t device_count = 0;
  ret = aclrtGetDeviceCount(&device_count);
  if (ret != ACL_SUCCESS || device_count == 0) {
    LOG(ERROR) << "No devices available!";
    return false;
  }
  if (device_id >= device_count) {
    LOG(ERROR) << "Invalid device_id " << device_id << ", only " << device_count << " devices available";
    return false;
  }

  ret = aclprofInit(profiling_dir.c_str(), profiling_dir.length());
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclprofInit failed, ret = " << ret;
    return false;
  }

  uint32_t device_list[1] = {device_id};
  uint32_t device_num = 1;
  uint32_t mask = ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_AICORE_METRICS | ACL_PROF_RUNTIME_API;
  acl_config_ = aclprofCreateConfig(device_list, device_num, ACL_AICORE_PIPE_UTILIZATION, nullptr, mask);
  if (acl_config_ == nullptr) {
    LOG(ERROR) << "Failed to call aclprofCreateConfig function.";
    return false;
  }

  ret = aclprofStart(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclprofStart start failed, ret = " << ret;
    return false;
  }

  return true;
}

bool ProfileMgr::StopProfiling() {
  LOG(INFO) << "End to profile";
  int ret = aclprofStop(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclprofDestroyConfig failed, ret = " << ret;
    return false;
  }
  ret = aclprofDestroyConfig(acl_config_);
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclprofDestroyConfig failed, ret = " << ret;
    return false;
  }
  ret = aclprofFinalize();
  if (ret != ACL_ERROR_NONE) {
    LOG(ERROR) << "aclProfFinalize failed, ret = " << ret;
    return false;
  }
  return true;
}

ProfileMgr &ProfileMgr::GetInstance() {
  static ProfileMgr mgr;
  return mgr;
}

void ascend_start_profiling(const std::string &profiling_dir, int device_id) {
  ProfileMgr::GetInstance().StartupProfiling(profiling_dir, device_id);
  return;
}

void ascend_stop_profiling() {
  ProfileMgr::GetInstance().StopProfiling();
  return;
}

// PYBIND interface
// cppcheck-suppress syntaxError
PYBIND11_MODULE(profile_mgr, m) {
  mlir::runtime::akg_log_init();
  if (!google::IsGoogleLoggingInitialized()) {
    google::InitGoogleLogging("akg");
  }
  m.doc() = "pybind ascend profiling";  // optional module docstring
  m.def("ascend_start_profiling", &ascend_start_profiling);
  m.def("ascend_stop_profiling", &ascend_stop_profiling);
}

}  // namespace runtime
}  // namespace mlir
