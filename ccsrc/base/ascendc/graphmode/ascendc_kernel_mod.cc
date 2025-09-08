/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "ascendc_kernel_mod.h"
#include <algorithm>
#include <vector>
#include <map>
#include <set>
#include <functional>
#include "mindspore/core/include/utils/ms_utils.h"
#include "mindspore/core/include/mindapi/ir/tensor.h"
#include "mindspore/ops/kernel/ascend/acl_ir/acl_helper.h"

namespace ms_custom_ops {
bool AclnnCustomKernelMod::is_dynamic_ = false;

bool AclnnCustomKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  MS_LOG(DEBUG) << "AclnnCustomKernelMod Init";
  return true;
}

int AclnnCustomKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ret = KernelMod::Resize(inputs, outputs);
  if (UseSimulationApi()) {
    return ret;
  }
  GetWorkSpaceInfo(inputs, outputs);
  return ret;
}

bool AclnnCustomKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  return true;
}

std::vector<size_t> AclnnCustomKernelMod::GetLaunchIgnoredInputAddressIdx() const {
  static const std::map<std::string, std::vector<size_t>> launch_ignored_input_addr_idx = {
    {kTransposeOpName, {kIndex1}}};
  if (launch_ignored_input_addr_idx.count(kernel_name_) > 0) {
    return launch_ignored_input_addr_idx.at(kernel_name_);
  }
  return {};
}

AclnnCustomKernelMod::~AclnnCustomKernelMod() {
  (void)std::for_each(hash_cache_.begin(), hash_cache_.end(), [&](CacheTuple &item) {
    auto cache = std::get<kIndex2>(item);
    cache(device::ascend::ProcessCacheType::kReleaseParamsAndExecutor, {});
  });
}
}  // namespace ms_custom_ops
