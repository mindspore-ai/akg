/**
* @file tdt_device.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is used to push data
*/

#ifndef HOST_INNER_INC_TDT_DEVICE_H_
#define HOST_INNER_INC_TDT_DEVICE_H_

#include <string.h>
#include <memory>
#include <vector>
#include "tdt/data_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace tdt {
/**
 * @ingroup  TdtDevicePushData
 * @brief Tdt device push data to queue for ops.
 *
 * @par Function
 * Tdt device push data to queue for ops.
 *
 * @param channelName [IN] type #String. queue channel name
 * @param items [IN] type #vector<DataItem> DataItem is defined in data_common.h.  input data
 * @retval 0 Success
 * @retval OtherValues Fail
 *
 * @par Dependency
 * @li libtdtdevice.so: Library to which the interface belongs.
 * @li tdt_device.h: Header file where the interface declaration is located.
 * @li data_common.h: Header file where 'DataItem' defined
 *
 */
int32_t TdtDevicePushData(const std::string &channelName, std::vector<DataItem> &items);
}  // namespace tdt
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // HOST_INNER_INC_TDT_DEVICE_H_
