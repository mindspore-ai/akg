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

#ifndef __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_MLA_H__
#define __MS_CUSTOM_OPS_CCSRC_OPS_MS_KERNELS_INTERNAL_MLA_H__

#include <cstdint>

namespace ms_custom_ops {
enum MlaInputIndex : size_t {
  kMlaInputQnopeIndex = 0,
  kMlaInputQropeIndex,
  kMlaInputKvCacheIndex,
  kMlaInputKropeIndex,
  kMlaInputBlockTablesIndex,
  kMlaInputAttnMaskIndex,
  kMlaInputDeqScaleQkIndex,
  kMlaInputDeqScalePvIndex,
  kMlaInputQueryLensIndex,
  kMlaInputContextLensIndex,
  kMlaInputNumHeadIndex,
  kMlaInputScaleValueIndex,
  kMlaInputNumKVHeadIndex,
  kMlaInputMaskTypeIndex,
  kMlaInputInputFormatIndex,
  kMlaInputIsRingIndex,
  kMlaInputsNum
};

enum MlaMaskMode : int8_t {
  kMaskNone = 0,
  kMaskNorm,
  kMaskAlibi,
  kMaskSpec,
  kMaskFree,
};

enum MlaInputFormat : int8_t { kKVFormatND = 0, kKVFormatNZ };
}  // namespace ms_custom_ops

#endif