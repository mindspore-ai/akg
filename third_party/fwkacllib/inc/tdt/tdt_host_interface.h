/**
* @file tdt_host_interface.h
*
* Copyright (C) Huawei Technologies Co., Ltd. 2019-2020. All Rights Reserved.
*
* This program is used to host server
*/

#ifndef HOST_INNER_INC_TDT_HOST_INTERFACE_H_
#define HOST_INNER_INC_TDT_HOST_INTERFACE_H_

#include <string.h>
#include <memory>
#include <vector>
#include "tdt/data_common.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace tdt {
/**
* @ingroup TdtHostInit
* @brief Initialize the interface, start and initialize various general thread, log and other services
*
* @par Function
* Initialize the interface, start and initialize various general thread, log and other services
*
* @param  deviceId [IN] type #unsigned int. Physical device ID
* @retval #0 Success
* @retval #Not 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtHostInit(uint32_t deviceId);

/**
* @ingroup TdtHostPushData
* @brief Blocking queue. When the queue is full, the Push interface will block.
*
* @par Function
* Blocking queue. When the queue is full, the Push interface will block.
*
* @param channelName [IN] type #String. queue channel name
* @param items [IN] type #vector<DataItem> DataItem is defined in data_common.h.  input data
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'DataItem' defined
*/
int32_t TdtHostPushData(const std::string &channelName, const std::vector<DataItem> &item, uint32_t deviceId = 0);

/**
* @ingroup TdtHostDestroy
* @brief Notify TDT component to close related resources
*
* @par Function
* Notify TDT component to close related resources
*
* @param  NA
* @retval 0 Success
* @retval OtherValues Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtHostDestroy();

/**
* @ingroup TdtHostPreparePopData
* @brief Prepare pop data from Tdt data storage queue
*
* @par Function
* Prepare pop data from Tdt data storage queue
*
* @param NA
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'DataItem' defined
*/
int32_t TdtHostPreparePopData();

/**
* @ingroup TdtHostPopData
* @brief POP data from Tdt data storage queue
*
* @par Function
* POP data from Tdt data storage queue
*
* @param channelName [IN] type #String. queue channel name
* @param items [IN] type #vector<DataItem> DataItem is defined in data_common.h.  input data
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'DataItem' defined
*/
int32_t TdtHostPopData(const std::string &channelName, std::vector<DataItem> &item);

/**
* @ingroup TdtHostStop
* @brief Activate the thread that reads data externally from Tdt and
* send end of sequence data so that the external thread can exit
*
* @par Function
* Activate the thread that reads data externally from Tdt and send
* end of sequence data so that the external thread can exit
*
* @param  channelName [IN] type #String. queue channel name
* @retval 0 Success
* @retval OtherValues Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtHostStop(const std::string &channelName);

/**
* @ingroup TdtInFeedInit
* @brief Initialize the interface, start and initialize various general thread, log and other services
*
* @par Function
* Initialize the interface, start and initialize various general thread, log and other services
*
* @param  deviceId [IN] type #unsigned int. logic device ID
* @retval #0 Success
* @retval #Not 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtInFeedInit(uint32_t deviceId);

/**
* @ingroup TdtOutFeedInit
* @brief Initialize the interface, start and initialize various general thread, log and other services
*
* @par Function
* Initialize the interface, start and initialize various general thread, log and other services
*
* @param  deviceId [IN] type #unsigned int. logic device ID
* @retval #0 Success
* @retval #Not 0 Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtOutFeedInit(uint32_t deviceId);

/**
* @ingroup TdtInFeedDestroy
* @brief Notify TDT component to close related resources
*
* @par Function
* Notify TDT component to close related resources
*
* @param  NA
* @retval 0 Success
* @retval OtherValues Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtInFeedDestroy(uint32_t deviceId);

/**
* @ingroup TdtOutFeedDestroy
* @brief Notify TDT component to close related resources
*
* @par Function
* Notify TDT component to close related resources
*
* @param  NA
* @retval 0 Success
* @retval OtherValues Fail
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tdt_host_interface.h: Header file where the interface declaration is located.
*/
int32_t TdtOutFeedDestroy();
}  // namespace tdt
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // HOST_INNER_INC_TDT_HOST_INTERFACE_H_
