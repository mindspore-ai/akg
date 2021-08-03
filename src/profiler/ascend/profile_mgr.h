/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <string>
#include "toolchain/prof_callback.h"
#include "toolchain/prof_acl_api.h"


enum ProfCommandHandleType {
  kProfCommandhandleInit = 0,
  kProfCommandhandleStart,
  kProfCommandhandleStop,
  kProfCommandhandleFinalize,
  kProfCommandhandleModelSubscribe,
  kProfCommandhandleModelUnsubscribe
};

namespace air {
namespace runtime {

class ProfileMgr {
 public:
  ProfileMgr() = default;
  ~ProfileMgr() = default;

  void RegCtrlCallback(MsprofCtrlCallback func) {
    ctrl_cb_ = func;
  }

  void RegSetDeviceCallback(MsprofSetDeviceCallback func) {
    set_device_cb_ = func;
  }

  void RegReporterCallback(MsprofReporterCallback func) {
    reporter_cb_ = func;
  }

  Status CommandHandle(ProfCommandHandleType type, void *data, uint32_t len);

  static ProfileMgr &GetInstance();

  uint32_t GetCurrentDeviceId() const { return device_id_; }

  void SetDeviceId(uint32_t id) { device_id_ = id; }

  bool StartupProfiling(uint32_t device_id);

  bool StopProfiling();

  void ForceMsprofilerInit();

  void SetKernelLabel(std::string &label) { kernel_label_ = label; }

  std::string GetKernelLabel() const { return kernel_label_; }

 private:
  uint64_t GetProfilingModule() const;
  uint64_t GetJobId() const;
  Status GetProfConf(MsprofGeOptions *prf);
  bool ProfStartUp(MsprofGeOptions *prof_conf);
  Status PluginInit() const;
  void PluginUnInit() const;

  MsprofCtrlCallback ctrl_cb_ {nullptr};
  MsprofSetDeviceCallback set_device_cb_ {nullptr};
  MsprofReporterCallback reporter_cb_ {nullptr};
  uint32_t device_id_ {0};
  std::string kernel_label_;
};

}
}
#endif
