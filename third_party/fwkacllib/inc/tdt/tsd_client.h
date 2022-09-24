/**
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2018-2021. All rights reserved.
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

#ifndef TDT_HOST_INNER_INC_TSD_CLIENT_H
#define TDT_HOST_INNER_INC_TSD_CLIENT_H

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include "tsd/status.h"
#include "toolchain/prof_callback.h"

#ifdef WIN_TSD
#define TDT_LIB_EXPORT __declspec(dllexport)
#else
#define TDT_LIB_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct InitFlowGwInfo {
    const char_t *groupName;
    uint64_t schedPolicy;
    uint64_t reschedInterval;
    char_t rsv[128];
};

typedef enum {
    TSD_CAPABILITY_PIDQOS = 0,
    TSD_CAPABILITY_BUT
} TsdCapabilityType;
/**
* @ingroup Open
* @brief Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param logicDeviceId [IN] type #unsigned int. Logic device ID
* @param rankSize [IN] type #unsigned int. The rankSize of the training.
* The default value is 1. When rankSize is greater than 1,
* HCCP will be pulled to perform set communication related operations.
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdOpen(const uint32_t logicDeviceId, const uint32_t rankSize);

/**
* @ingroup Open
* @brief Used for the Framework process to communicate with the TSDDaemon process in 1981,
* and notify TSD to complete the initialization of other processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param logicDeviceId [IN] type #unsigned int. Logic device ID
* @param rankSize [IN] type #unsigned int. The rankSize of the training.
* The default value is 1. When rankSize is greater than 1,
* HCCP will be pulled to perform set communication related operations.
* @param deviceMode [IN] type unsigned int. The device running mode of aicpuSd,
* it include chipMode and DieMode
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdOpenEx(const uint32_t logicDeviceId, const uint32_t rankSize, const uint32_t deviceMode);

/**
* @ingroup InitialQs
* @brief Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of QS processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param logicDeviceId [IN] type #unsigned int. Logic device ID
* @param groupName [IN] type #char pointer. qs group name send by host process
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdInitQs(const uint32_t logicDeviceId, const char_t * const groupName = nullptr);

/**
* @ingroup InitFlowGw
* @brief Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of FlowGw processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param logicDeviceId [IN] type #unsigned int. Logic device ID
* @param initInfo [IN] type #InitFlowGwInfo pointer. Initialization parameters
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdInitFlowGw(const uint32_t logicDeviceId, const InitFlowGwInfo * const initInfo);

/**
* @ingroup Close
* @brief notify TSDClient close resource
*
* @par Function
* notify TSDClient close resource
*
* @param NA
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency

* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdClose(const uint32_t logicDeviceId);

/**
* @ingroup UpdateProfilingMode
* @brief notify TSDClient update profiling mode
*
* @par Function
* notify TSDClient update profiling mode
*
* @param NA
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t UpdateProfilingMode(const uint32_t logicDeviceId, const uint32_t flag);

/**
* @ingroup TsdSetMsprofReporterCallback
* @brief 用于推理场景下设置aicpu的profilng的callback函数
*
* @par Function
* 设置offline模式下aicpu_sd进程的profiling的callback函数
*
* @param callback [IN] type #MsprofReporterCallback. 回调函数
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
* @li prof_callback.h: Headerfile where 'MsprofReporterCallback' defined
*/
TDT_LIB_EXPORT uint32_t TsdSetMsprofReporterCallback(const MsprofReporterCallback callback);

/**
* @ingroup TsdSetAttr
* @brief used to set tsd attr
*
* @par key
* key set for tsd attr,now only support RunMode
*
* @par value
* value set to run correspond mode, PROCESS_MODE or THREAD_MODE
* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdSetAttr(const char * const attrKey, const char * const attrValue);

/**
* @ingroup TsdCapabilityGet
* @brief use tsd to get some capability
*
* @par type
* capability type
*
* @par ptr
* the result
* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdCapabilityGet(const uint32_t logicDeviceId, const int32_t type, const uint64_t ptr);


/**
* @ingroup GetHdcConctStatus
* @brief used to get hdc connection status
*
* @par logicDeviceId
* logic device id
*
* @par hdcSessStat
* hdc session status, DRV_ERROR_SOCKET_CONNECT or DRV_ERROR_SOCKET_CLOSE
* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t GetHdcConctStatus(const uint32_t logicDeviceId, int32_t *hdcSessStat);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // TDT_HOST_INNER_INC_TSD_CLIENT_H
