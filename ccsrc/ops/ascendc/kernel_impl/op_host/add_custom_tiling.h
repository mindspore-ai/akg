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
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, smallCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigCoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalBigTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalSmallTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, smallTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, bigTailDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, tailBlockNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AddCustom, TilingData)
}
#endif // ADD_CUSTOM_TILING_H
