/**
 * @file add_custom_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADD_RMS_NORM_CUSTOM_TILING_H
#define ADD_RMS_NORM_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AddRmsNormTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, num_row);
  TILING_DATA_FIELD_DEF(uint32_t, num_col);
  TILING_DATA_FIELD_DEF(uint32_t, block_factor);
  TILING_DATA_FIELD_DEF(uint32_t, row_factor);
  TILING_DATA_FIELD_DEF(uint32_t, ub_factor);
  TILING_DATA_FIELD_DEF(float, epsilon);
  TILING_DATA_FIELD_DEF(float, avg_factor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddRmsNormCustom, AddRmsNormTilingData)
}
#endif // ADD_RMS_NORM_CUSTOM_TILING_H
