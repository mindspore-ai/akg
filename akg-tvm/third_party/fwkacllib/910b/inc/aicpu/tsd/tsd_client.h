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
    TSD_CAPABILITY_LEVEL  = 1,
    TSD_CAPABILITY_BUT    = 0xFF
} TsdCapabilityType;

typedef enum {
    SUB_PROCESS_STATUS_NORMAL = 0,
    SUB_PROCESS_STATUS_EXITED = 1,
    SUB_PROCESS_STATUS_STOPED = 2,
    SUB_PROCESS_STATUS_MAX    = 0xFF
} SubProcessStatus;

typedef enum {
    TSD_SUB_PROC_HCCP           = 0,           // hccp process
    TSD_SUB_PROC_COMPUTE        = 1,           // aicpu_schedule process
    TSD_SUB_PROC_CUSTOM_COMPUTE = 2,           // aicpu_cust_schedule process
    TSD_SUB_PROC_QUEUE_SCHEDULE = 3,           // queue_schedule process
    TSD_SUB_PROC_UDF            = 4,           // udf process
    TSD_SUB_PROC_NPU            = 5,           // npu process
    TSD_SUB_PROC_PROXY          = 6,           // proxy process
    TSD_SUB_PROC_MAX            = 0xFF
} SubProcType;

struct ProcStatusInfo {
    pid_t pid;
    SubProcessStatus curStat;
};

struct ProcEnvParam {
    const char   *envName;
    uint64_t     nameLen;
    const char   *envValue;
    uint64_t     valueLen;
};
struct ProcExtParam {
    const char  *paramInfo;
    uint64_t    paramLen;
};
struct ProcOpenArgs {
    SubProcType  procType;
    ProcEnvParam *envParaList;
    uint64_t     envCnt;
    const char   *filePath;
    uint64_t     pathLen;
    ProcExtParam *extParamList;
    uint64_t     extParamCnt;
    pid_t        *subPid;
};
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
* @ingroup OpenAicpuSd
* @brief Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @par Function
* Used for the Framework process to communicate with the TSDDaemon process,
* and notify TSD to complete the initialization of other processes
*
* @param logicDeviceId [IN] type #unsigned int. Logic device ID
* @retval TDT_OK Success
* @retval OtherValues Failure
*
* @par Dependency
* @li libtsdclient.so: Library to which the interface belongs.
* @li tsd_client.h: Header file where the interface declaration is located.
* @li data_common.h: Header file where 'TDT_StatusT' defined
*/
TDT_LIB_EXPORT uint32_t TsdOpenAicpuSd(const uint32_t logicDeviceId);

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

/**
* @ingroup TsdBindHostPid
* @brief Tsd supply bindhostpid function
*
* @par logicDeviceId
* logic device id
*
* @par processType
* device processtype

* @par devicePid
* device pid

* @par hostPid
* host pid

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdBindHostPid(const uint32_t logicDeviceId, const SubProcType processType,
                                       const int32_t devicePid, const int32_t hostPid);

/**
* @ingroup TsdFileLoad
* @brief Tsd omfile send function
*
* @par logicDeviceId
* logic device id
*
* @par filePath
* the path of the file

* @par pathLen
* the path length

* @par fileName
* the name of the file

* @par fileNameLen
* the filename length

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdFileLoad(const uint32_t logicDeviceId, const char *filePath, const uint64_t pathLen,
                                    const char *fileName, const uint64_t fileNameLen);

/**
* @ingroup TsdFileUnLoad
* @brief Tsd remove omfile
*
* @par logicDeviceId
* logic device id
*
* @par filePath
* the path of the file

* @par pathLen
* the path length

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdFileUnLoad(const uint32_t logicDeviceId, const char *filePath, const uint64_t pathLen);

/**
* @ingroup TsdGetProcStatus
* @brief Tsd query subproc status
*
* @par logicDeviceId
* logic device id
*
* @par pidArry
* the pid list

* @par arrayLen
* the list length

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdGetProcStatus(const uint32_t logicDeviceId, ProcStatusInfo *pidInfo,
                                         const uint32_t arrayLen);

/**
* @ingroup TsdProcessOpen
* @brief Tsd pull nn or udf process
*
* @par logicDeviceId
* logic device id
*
* @par openArgs
* the args of nn or udf needed

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs *openArgs);

/**
* @ingroup TsdProcessClose
* @brief Tsd close
*
* @par logicDeviceId
* logic device id
*
* @par closePid
* the pid need to be close

* @retval TDT_OK Success
* @retval OtherValues Failure
*/
TDT_LIB_EXPORT uint32_t TsdProcessClose(const uint32_t logicDeviceId, const pid_t closePid);

#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // TDT_HOST_INNER_INC_TSD_CLIENT_H
