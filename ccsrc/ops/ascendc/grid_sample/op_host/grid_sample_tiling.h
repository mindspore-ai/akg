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
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampleTilingData)
  TILING_DATA_FIELD_DEF(float, h_in);
  TILING_DATA_FIELD_DEF(float, w_in);
  TILING_DATA_FIELD_DEF(int32_t, h_out);
  TILING_DATA_FIELD_DEF(int32_t, w_out);
  TILING_DATA_FIELD_DEF(int32_t, n_in);
  TILING_DATA_FIELD_DEF(int32_t, c_in);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSample, GridSampleTilingData)
}
#endif // ADD_CUSTOM_TILING_H
