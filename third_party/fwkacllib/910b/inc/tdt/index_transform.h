/**
* @file index_transform.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2018-2019. All Rights Reserved.
*
* This program is used to get logical device id by phy device id.
*/

#ifndef INC_TDT_INDEX_TRANSFORM_H
#define INC_TDT_INDEX_TRANSFORM_H

#include "stdint.h"
/**
* @ingroup IndexTransform
* @brief get logical device id by phy device id.
*
* @par Function get logical device id by phy device id.
*
* @param  phyId [IN] physical device id
* @param  logicalId [OUT] logical device id
* @retval 0 Success
* @retval OtherValues Fail
*
* @par Dependency
* @li libruntime.so: Library to which the interface belongs.
*/

int32_t IndexTransform(const uint32_t phyId, uint32_t &logicId);
#endif
