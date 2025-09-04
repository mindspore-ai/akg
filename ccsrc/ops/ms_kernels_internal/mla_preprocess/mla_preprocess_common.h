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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_MLA_PREPROCESS_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_MLA_PREPROCESS_H__

#include <cstdint>

namespace ms_custom_ops {
enum MlaPreprocessInputIndex : size_t {
  kMlaPreprocessInput1Index = 0,
  kMlaPreprocessGamma1Index = 1,
  kMlaPreprocessBeta1Index = 2,
  kMlaPreprocessQuantScale1Index = 3,
  kMlaPreprocessQuantOffset1Index = 4,
  kMlaPreprocessWdqkvIndex = 5,
  kMlaPreprocessBias1Index = 6,
  kMlaPreprocessGamma2Index = 7,
  kMlaPreprocessBeta2Index = 8,
  kMlaPreprocessQuantScale2Index = 9,
  kMlaPreprocessQuantOffset2Index = 10,
  kMlaPreprocessGamma3Index = 11,
  kMlaPreprocessSin1Index = 12,
  kMlaPreprocessCos1Index = 13,
  kMlaPreprocessSin2Index = 14,
  kMlaPreprocessCos2Index = 15,
  kMlaPreprocessKeyCacheIndex = 16,
  kMlaPreprocessSlotMappingIndex = 17,
  kMlaPreprocessWuqIndex = 18,
  kMlaPreprocessBias2Index = 19,
  kMlaPreprocessWukIndex = 20,
  kMlaPreprocessDeScale1Index = 21,
  kMlaPreprocessDeScale2Index = 22,
  kMlaPreprocessCtkvScaleIndex = 23,
  kMlaPreprocessQnopeScaleIndex = 24,
  kMlaPreprocessKropeCacheIndex = 25,
  kMlaPreprocessParamCacheModeIndex = 26,
  kMlaPreProcessInputsNum = 27
};

enum MlaPreprocessOutputIndex : size_t {
  kMlaPreprocessOutputQueryOutIndex = 0,
  kMlaPreprocessOutputKeyOutIndex = 1,
  kMlaPreprocessOutputQropeIndex = 2,
  kMlaPreprocessOutputKropeIndex = 3, 
  kMlaPreprocessOutputsNum = 4
};

constexpr int64_t kMlaPreCacheModeQK = 0;
constexpr int64_t kMlaPreCacheModeQKSplitQuant = 2;
constexpr int64_t kMlaPreCacheModeQKSplitNz = 3;

inline internal::InternalOpPtr CreateMlaPreprocessOpWithFormat(const internal::InputsImmutableInfoList &inputs,
                                                               const internal::OutputsImmutableInfoList &outputs,
                                                               const internal::MlaPreprocessParam &param) {
  auto inputs_clone = inputs;
  inputs_clone[kMlaPreprocessWdqkvIndex].SetFormat(internal::kFormatFRACTAL_NZ);
  inputs_clone[kMlaPreprocessWuqIndex].SetFormat(internal::kFormatFRACTAL_NZ);
  if (param.cache_mode == kMlaPreCacheModeQKSplitQuant || param.cache_mode == kMlaPreCacheModeQKSplitNz) {
    inputs_clone[kMlaPreprocessKeyCacheIndex].SetFormat(internal::kFormatFRACTAL_NZ);
    inputs_clone[kMlaPreprocessKropeCacheIndex].SetFormat(internal::kFormatFRACTAL_NZ);
  }
  return internal::CreateMlaPreprocessOp(inputs_clone, outputs, param, internal::kInternalMlaPreprocessOpName);
};



}  // namespace ms_custom_ops
#endif
