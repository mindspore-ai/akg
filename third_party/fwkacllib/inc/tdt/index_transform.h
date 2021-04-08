/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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
