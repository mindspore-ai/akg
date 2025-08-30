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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_PAGED_CACHE_LOAD_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_PAGED_CACHE_LOAD_H__

#include <cstdint>

namespace ms_custom_ops {
enum PagedCacheLoadInputIndex : size_t {
  kPCLInputKeyCacheIndex = 0,
  kPCLInputValueCacheIndex,
  kPCLInputBlockTableIndex,
  kPCLInputSeqLensIndex,
  kPCLInputKeyIndex,
  kPCLInputValueIndex,
  kPCLInputSeqStartsIndex,
  kPCLInputParamKvCacheCfgIndex,
  kPCLInputParamIsSeqLensCumsumTypeIndex,
  kPCLInputParamHasSeqStartsIndex,
  kPCLInputsNum
};

enum PagedCacheLoadOutputIndex : size_t {
  kPCLOutputKeyOutIndex = 0,
  kPCLOutputValueOutIndex,
  kPCLOutputsNum
};

inline internal::InternalOpPtr CreatePagedCacheLoadOpWithFormat(const internal::InputsImmutableInfoList &inputs,
                                                                const internal::OutputsImmutableInfoList &outputs,
                                                                const internal::PagedCacheLoadParam &param) {
  if (param.kv_cache_cfg_type == 1) {
    auto inputs_clone = inputs;
    inputs_clone[kPCLInputKeyCacheIndex].SetFormat(internal::kFormatFRACTAL_NZ);
    inputs_clone[kPCLInputValueCacheIndex].SetFormat(internal::kFormatFRACTAL_NZ);
    return internal::CreatePagedCacheLoadOp(inputs_clone, outputs, param, internal::kInternalPagedCacheLoadOpName);
  }
  return internal::CreatePagedCacheLoadOp(inputs, outputs, param, internal::kInternalPagedCacheLoadOpName);
};
}  // namespace ms_custom_ops
#endif
