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

#ifndef TVM_RUNTIME_CCE_PROFILE_MGR_H_
#define TVM_RUNTIME_CCE_PROFILE_MGR_H_

#include "akg/ExecutionEngine/AscendLaunchRuntime/CceAcl.h"
#include <pybind11/pybind11.h>

namespace mlir {
namespace runtime {

class ProfileMgr {
 public:
  ProfileMgr() = default;
  ~ProfileMgr() = default;

  static ProfileMgr &GetInstance();

  bool StartupProfiling(const uint32_t device_id = 0);

  bool StopProfiling();

 private:
  aclprofConfig *acl_config_{nullptr};
};

using ProfileMgrPtr = std::shared_ptr<ProfileMgr>;

extern "C" void ascend_start_profiling(int device_id);

extern "C" void ascend_stop_profiling();

}  // namespace runtime
}  // namespace mlir
#endif
