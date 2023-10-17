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

#ifndef INC_TDT_TSD_H
#define INC_TDT_TSD_H
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

/**
* @ingroup  Tsdaemon.
*
* Identifies that HCCP or Compute_process is waiting for
* Tsdaemon to issue a shutdown command.
*/
typedef enum {
    TSD_HCCP = 0,    /**< HCCP*/
    TSD_COMPUTE, /**< Compute_process*/
    TSD_CUSTOM_COMPUTE, /**< Custom Compute_process*/
    TSD_QS,
    TSD_WAITTYPE_MAX /**< Max*/
} TsdWaitType;

typedef enum {
    TSD_EVENT_START_RSP = 0,                  // 0 start response: client -> server tsd
    TSD_EVENT_SHUTDOWN_RSP = 1,               // 1 shutdown response: client -> server tsd
    TSD_EVENT_NOTIFY_AICPUINFO = 2,           // 2 inform tsdaemon start aicpu_cust_schedule
    TSD_EVENT_NOTIFY_AICPUINFO_RSP = 3,       // 3 start aicpu_cust_schedule response
    TSD_EVENT_ABNORMAL = 4,                   // 4 aicpu_sd destroy: client -> server tsd
    TSD_EVENT_SHUTDOWN = 5,                   // 5 shutdown:tsd server -> client
    TSD_EVENT_LOAD_SO = 6,                    // 6 aicpu_sd inform cust aicpusd to load so
    TSD_EVENT_START_OR_STOP_FAIL = 7,         // 7 send host start/stop err msg client-> server tsd
    TSD_EVENT_GET_CAPABILITY = 8,             // 8 send get capability msg to subprocess
    TSD_EVENT_START_AICPU_SD_MODULE = 30,     // 30 send start aicpu module msg
    TSD_EVENT_START_AICPU_SD_MODULE_RSP = 31, // 31 start aicpu module msg rsp
    TSD_EVENT_START_QS_MODULE = 32,           // 32 send start qs module msg
    TSD_EVENT_START_QS_MODULE_RSP = 33,       // 33 start qs module msg rsp
    TSD_EVENT_STOP_AICPU_SD_MODULE = 34,      // 34 send stop aicpu module msg
    TSD_EVENT_STOP_AICPU_SD_MODULE_RSP = 35,  // 35 stop aicpu module msg rsp
    TSD_EVENT_STOP_QS_MODULE = 36,            // 36 send stop qs module msg
    TSD_EVENT_STOP_QS_MODULE_RSP = 37,        // 37 stop qs module msg rsp
    TSD_EVENT_STOP_SUB_PROCESS_WAIT = 38,     // 38 subprocess wait and suspend
    TSD_EVENT_STOP_AICPU_PROCESS_WAIT = 39,     // 39 aicpu process wait cust aicpu attach success
    TSD_EVENT_TYPE_MAX
} TsdSubEventType;

struct TsdCapabilityMsgInfo {
    uint32_t subCapabityType;
    uint32_t resultInfo;
};

#define MAX_EVENT_PRI_MSG_LENGTH (96U)
struct TsdSubEventInfo {
    uint32_t deviceId;                    // device id
    uint32_t srcPid;                      // send process pid
    uint32_t dstPid;                      // receive process pid
    uint32_t hostPid;                     // host pid
    uint32_t vfId;                        // vf id
    uint32_t procType;                    // process type
    uint32_t eventType;                   // event type
    uint32_t startProcPid;                // startup process pid
    char priMsg[MAX_EVENT_PRI_MSG_LENGTH]; // so name
};

typedef int32_t (* SubProcEventCallBackFuncInfo)(const struct TsdSubEventInfo * const msg);

struct SubProcEventCallBackInfo {
    uint32_t eventType;
    SubProcEventCallBackFuncInfo callBackFunc;
};

/**
* @ingroup TsdWaitForShutdown
* @brief Wait for the TSD process to issue the shutdown command
*
* @par Function
* Wait for the TSD process to issue the shutdown command
*
* @param NA
* @param deviceId [IN] type #unsigned int. Physical device ID
* @param waitType [IN] type #TsdWaitType. HCCP or CP
* @param hostPid [IN] type #unsigned int. Host pid
* @param vfId [IN] type #unsigned int. Virtual force Id
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdppc.so: Library to which the interface belongs.
* @li tsd.h: Header file where the interface declaration is located.
*/
int32_t TsdWaitForShutdown(const uint32_t deviceId, const TsdWaitType waitType,
                           const uint32_t hostPid, const uint32_t vfId);

/**
* @ingroup TsdDestroy
* @brief tsd event client send abnormal msg to tsd event server
*
* @par Function
* tsd event client send abnormal msg to tsd event server
*
* @param NA
* @param deviceID [IN] type #unsigned int. Physical device ID
* @param waitType [IN] type #TsdWaitType. HCCP or CP
* @param hostPid [IN] type #unsigned int. Host pid
* @param vfId [IN] type #unsigned int. Virtual force id
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdppc.so: Library to which the interface belongs.
* @li tsd.h: Header file where the interface declaration is located.
*/
int32_t TsdDestroy(const uint32_t deviceId, const TsdWaitType waitType,
                   const uint32_t hostPid, const uint32_t vfId);

/**
* @ingroup CreateOrFindCustPid
* @brief inform tsdaemon start aicpu_cust_schedule
*
* @par Function
* inform tsdaemon start aicpu_cust_schedule
*
* @param NA
* @param deviceID [IN] type #unsigned int. Physical device ID
* @param loadLibNum [IN] type #unsigned int. Load so nums
* @param loadLibName [IN] type #char *. Load so names
* @param hostPid [IN] type #unsigned int. Host pid
* @param vfId [IN] vf id
* @param groupNameList [IN] group name list which cust aicpu need to attach
* @param groupNameNum [IN] group name number which cust aicpu need to attach
* @param custProcPid [OUT] cust aicpu pid
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdppc.so: Library to which the interface belongs.
* @li tsd.h: Header file where the interface declaration is located.
*/
int32_t CreateOrFindCustPid(const uint32_t deviceId, const uint32_t loadLibNum, const char * const loadLibName[],
                            const uint32_t hostPid, const uint32_t vfId, const char * const groupNameList,
                            const uint32_t groupNameNum, int32_t * const custProcPid, bool * const firstStart);

/**
* @ingroup TsdStartupResponse
* @brief Wait for the TSD process to issue the shutdown command
*
* @par Function
* Wait for the TSD process to issue the shutdown command
*
* @param NA
* @param deviceId [IN] type #unsigned int. Physical device ID
* @param waitType [IN] type #TsdWaitType. HCCP or CP
* @param hostPid [IN] type #unsigned int. Host pid
* @param vfId [IN] type #unsigned int. Virtual force Id
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdppc.so: Library to which the interface belongs.
* @li tsd.h: Header file where the interface declaration is located.
*/
int32_t StartupResponse(const uint32_t deviceId, const TsdWaitType waitType,
                        const uint32_t hostPid, const uint32_t vfId);


/**
* @ingroup TsdDoLoopProcessEvent
* @brief loop to keep process main thread running
*
* @par Function
* loop to keep process main thread running
*
* @param NA
* @param deviceId [IN] type #unsigned int. Physical device ID
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsdppc.so: Library to which the interface belongs.
* @li tsd.h: Header file where the interface declaration is located.
*/
int32_t WaitForShutDown(const uint32_t deviceId);

/**
* @ingroup tsd_event_client
* @brief sub process send error code while start/stop period
* @param [in] deviceId : device id
* @param [in] waitType : process type
* @param [in] hostPid :  host pid
* @param [in] vfId : vf id
* @param [in] errCode : errCode: errMsg code produced by host
* @param [in] errLen : errLen: errMsg code length
* @return TSD_OK: sucess, other: error code
*/
int32_t TsdReportStartOrStopErrCode(const uint32_t deviceId, const TsdWaitType waitType,
                                    const uint32_t hostPid, const uint32_t vfId,
                                    const char *errCode, const uint32_t errLen);


/**
* @ingroup tsd_event_client
* @brief sub process send capability msg to tsd
* @param [in] deviceId : device id
* @param [in] waitType : process type
* @param [in] hostPid :  host pid
* @param [in] vfId : vf id
* @param [in] msgInfo : capability
* @return TSD_OK: sucess, other: error code
*/
int32_t ReportMsgToTsd(const uint32_t deviceId, const TsdWaitType waitType,
                       const uint32_t hostPid, const uint32_t vfId,
                       const char * const msgInfo);

/**
* @ingroup tsd_event_client
* @brief reg event call back func to tsdclient
* @param [in] SubProcEventCallBackInfo : event id, callbackfunc
* @return TSD_OK: sucess, other: error code
*/
int32_t RegEventMsgCallBackFunc(const struct SubProcEventCallBackInfo *regInfo);

/**
* @ingroup tsd_event_client
* @brief unreg event call back func to tsdclient
* @param [in]  : event id
* @return void
*/
void UnRegEventMsgCallBackFunc(const uint32_t eventType);

/**
* @ingroup tsd_event_client
* @brief reg event call back func to tsdclient
* @param [in] SubProcEventCallBackInfo : event id, callbackfunc
* @return TSD_OK: sucess, other: error code
*/

/**
* @ingroup tsd_event_client
* @brief sub module send start success message to tsd
* @param [in] deviceId : device id
* @param [in] waitType : process type
* @param [in] hostPid :  host pid
* @param [in] vfId : vf id
* @param [in] eventType : eventType
* @return TSD_OK: sucess, other: error code
*/
int32_t SubModuleProcessResponse(const uint32_t deviceId, const TsdWaitType waitType,
                                 const uint32_t hostPid, const uint32_t vfId,
                                 const uint32_t eventType);

/**
* @ingroup tsd_event_client
* @brief send start up msg to tsd and wait process
* @param [in] deviceId : device id
* @param [in] waitType : process type
* @param [in] hostPid :  host pid
* @param [in] vfId : vf id
* @return TSD_OK: sucess, other: error code
*/
int32_t StartUpRspAndWaitProcess(const uint32_t deviceId, const TsdWaitType waitType,
                                 const uint32_t hostPid, const uint32_t vfId);

/**
* @ingroup StopWaitForCustAicpu
* @brief aicpu wait aicpu_cust_schedule attatch grp success
* @return TSD_OK: sucess, other: error code
*/
int32_t StopWaitForCustAicpu();
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // INC_TDT_TSD_H
