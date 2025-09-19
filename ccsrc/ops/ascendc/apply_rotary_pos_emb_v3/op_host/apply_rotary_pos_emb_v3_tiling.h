/**
 * @file add_custom_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ApplyRotaryPosEmbV3TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, tilingId);
  TILING_DATA_FIELD_DEF(uint32_t, useCoreNum);
  TILING_DATA_FIELD_DEF(uint32_t, tokensPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tokensTail);
  TILING_DATA_FIELD_DEF(uint32_t, qHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, kHeadNum);
  TILING_DATA_FIELD_DEF(uint32_t, qHiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, kHiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, queryHeadDim);
  TILING_DATA_FIELD_DEF(uint32_t, cosHeadDim);
  TILING_DATA_FIELD_DEF(uint32_t, rotaryDim);
  TILING_DATA_FIELD_DEF(uint32_t, layout);
  TILING_DATA_FIELD_DEF(uint32_t, rotaryMode);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ApplyRotaryPosEmbV3, ApplyRotaryPosEmbV3TilingData)
}
#endif // ADD_CUSTOM_TILING_H
