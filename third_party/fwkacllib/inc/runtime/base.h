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

#ifndef __CCE_RUNTIME_BASE_H__
#define __CCE_RUNTIME_BASE_H__

#include <stdint.h>
#include "toolchain/prof_callback.h"

#if defined(__cplusplus)
extern "C" {
#endif

// If you need export the function of this library in Win32 dll, use __declspec(dllexport)
#ifndef RTS_API
#ifdef RTS_DLL_EXPORT
#define RTS_API __declspec(dllexport)
#else
#define RTS_API
#endif
#endif

typedef int32_t rtError_t;
static const int32_t RT_ERROR_NONE = 0; // success

/**
 * @ingroup dvrt_base
 * @brief runtime exception numbers.
 */
typedef enum tagRtExceptionType {
    RT_EXCEPTION_NONE = 0,
    RT_EXCEPTION_TS_DOWN = 1,
    RT_EXCEPTION_TASK_TIMEOUT = 2,
    RT_EXCEPTION_TASK_FAILURE = 3,
    RT_EXCEPTION_DEV_RUNNING_DOWN = 4,
    RT_EXCEPTION_STREAM_ID_FREE_FAILED = 5
} rtExceptionType;

/**
 * @ingroup dvrt_base
 * @brief Switch type.
 */
typedef enum tagRtCondition {
    RT_EQUAL = 0,
    RT_NOT_EQUAL,
    RT_GREATER,
    RT_GREATER_OR_EQUAL,
    RT_LESS,
    RT_LESS_OR_EQUAL
} rtCondition_t;

/**
 * @ingroup dvrt_base
 * @brief Data Type of Extensible Switch Task.
 */
typedef enum tagRtSwitchDataType {
    RT_SWITCH_INT32 = 0,
    RT_SWITCH_INT64 = 1,
} rtSwitchDataType_t;

typedef enum tagRtStreamFlagType {
    RT_HEAD_STREAM = 0,  // first stream
    RT_INVALID_FLAG = 0xFFFFFFFF,
} rtStreamFlagType_t;

typedef enum tagRtLimitType {
    RT_LIMIT_TYPE_LOW_POWER_TIMEOUT = 0,  // timeout for power down , ms
} rtLimitType_t;

typedef struct rtExceptionInfo {
    uint32_t taskid;
    uint32_t streamid;
    uint32_t tid;
    uint32_t deviceid;
    uint32_t retcode;
} rtExceptionInfo;

typedef void (*rtErrorCallback)(rtExceptionType);

typedef void (*rtTaskFailCallback)(rtExceptionInfo *exceptionInfo);

typedef void (*rtDeviceStateCallback)(uint32_t devId, bool isOpen);

/**
 * @ingroup profiling_base
 * @brief dataType: rtProfCtrlType_t
 * @brief data: data swtich or reporter function
 * @brief dataLen: length of data
 */
typedef rtError_t (*rtProfCtrlHandle)(uint32_t dataType, void *data, uint32_t dataLen);

/**
 * @ingroup dvrt_base
 * @brief stream handle.
 */
typedef void *rtStream_t;

/**
 * @ingroup dvrt_base
 * @brief runtime event handle.
 */
typedef void *rtEvent_t;

/**
 * @ingroup dvrt_base
 * @brief label handle.
 */
typedef void *rtLabel_t;

/**
 * @ingroup dvrt_base
 * @brief model handle.
 */
typedef void *rtModel_t;

#define RT_PROF_MAX_DEV_NUM 64

/**
 * @ingroup profiling_base
 * @brief profiling command info
 */
typedef struct rtProfCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[RT_PROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
} rtProfCommandHandle_t;

/**
 * @ingroup profiling_base
 * @brief type of app register profiling switch or reporter callback
 */
typedef enum {
    RT_PROF_CTRL_INVALID = 0,
    RT_PROF_CTRL_SWITCH,
    RT_PROF_CTRL_REPORTER,
    RT_PROF_CTRL_BUTT
} rtProfCtrlType_t;

/**
 * @ingroup profiling_base
 * @brief runtime handle.
 */
RTS_API rtError_t rtSetProfDirEx(const char *profDir, const char *address, const char *jobCtx);

/**
 * @ingroup profiling_base
 * @brief init profiler object.
 */
RTS_API rtError_t rtProfilerInit(const char *profDir, const char *address, const char *jobCtx);

/**
 * @ingroup profiling_base
 * @brief config rts profiler.
 */
RTS_API rtError_t rtProfilerConfig(uint16_t type);

/**
 * @ingroup profiling_base
 * @brief start rts profiler.
 */
RTS_API rtError_t rtProfilerStart(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList);

/**
 * @ingroup profiling_base
 * @brief stop rts profiler.
 */
RTS_API rtError_t rtProfilerStop(uint64_t profConfig, int32_t numsDev, uint32_t *deviceList);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTraceEx(uint64_t id, uint64_t modelId, uint16_t tagId, rtStream_t stream);

/**
 * @ingroup profiling_base
 * @brief ts set profiling reporter callback.
 */
RTS_API rtError_t rtSetMsprofReporterCallback(MsprofReporterCallback callback);

/**
 * @ingroup profiling_base
 * @brief add the map of deviceId and GE model index, called by ge
 * @param [in] geModelIdx  The index of GE model
 * @param [in] deviceId    The id of device
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t rtSetDeviceIdByGeModelIdx(uint32_t geModelIdx, uint32_t deviceId);

/**
 * @ingroup profiling_base
 * @brief del the map of deviceId and GE model index, called by ge
 * @param [in] geModelIdx  The index of GE model
 * @param [in] deviceId    The id of device
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t rtUnsetDeviceIdByGeModelIdx(uint32_t geModelIdx, uint32_t deviceId);

/**
 * @ingroup profiling_base
 * @brief find deviceId by GE model index, called by profiling
 * @param [in]  geModelIdx  The index of GE model
 * @param [out] deviceId    The id of device
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 * @return ACL_ERROR_RT_INTERNAL_ERROR for can't find deviceId by geModelIdx
 */
RTS_API rtError_t rtGetDeviceIdByGeModelIdx(uint32_t geModelIdx, uint32_t *deviceId);

/**
 * @ingroup profiling_base
 * @brief set profling switch, called by profiling
 * @param [in]  data  rtProfCommandHandle
 * @param [out] len   length of data
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t rtProfSetProSwitch(void *data, uint32_t len);

/**
 * @ingroup profiling_base
 * @brief register callback of upper app, called by ge or acl
 * @param [in]  moduleId of APP
 * @param [in]  callback function when switch or reporter change
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t rtProfRegisterCtrlCallback(uint32_t moduleId, rtProfCtrlHandle callback);

/**
 * @ingroup dvrt_base
 * @brief Returns the last error from a runtime call.
 */
RTS_API rtError_t rtGetLastError();

/**
 * @ingroup dvrt_base
 * @brief Returns the last error from a runtime call.
 */
RTS_API rtError_t rtPeekAtLastError();

/**
 * @ingroup dvrt_base
 * @brief register callback for error code
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetExceptCallback(rtErrorCallback callback);

/**
 * @ingroup dvrt_base
 * @brief register callback for task fail
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetTaskFailCallback(rtTaskFailCallback callback);

/**
 * @ingroup dvrt_base
 * @brief register callback for deviceid
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback Device state callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtRegDeviceStateCallback(const char *regName, rtDeviceStateCallback callback);

/**
 * @ingroup dvrt_base
 * @brief register callback for fail task
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback fail task callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtRegTaskFailCallbackByModule(const char *moduleName, rtTaskFailCallback callback);

/**
 * @ingroup dvrt_base
 * @brief notify handle.
 */
typedef void *rtNotify_t;

/**
 * @ingroup dvrt_base
 * @brief create label instance
 * @param [out]    label   created label
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreate(rtLabel_t *label);

/**
 * @ingroup dvrt_base
 * @brief create label instance
 * @param [out] label  created label
 * @param [in] model  label set model
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateV2(rtLabel_t *label, rtModel_t model);

/**
 * @ingroup dvrt_base
 * @brief set label and stream instance
 * @param [in] label   set label
 * @param [in] stream  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief destroy label instance
 * @param [in] label   label to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelDestroy(rtLabel_t label);

/**
 * @ingroup dvrt_base
 * @brief label switch instance
 * @param [in] ptr  address to get value compared
 * @param [in] condition
 * @param [in] value  to compare
 * @param [in] true_label   goto label
 * @param [in] stream  to submit label_switch task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t trueLabel,
                                rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief goto label instance
 * @param [in] label   goto label
 * @param [in] stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief name label instance
 * @param [in] label  instance
 * @param [in] name  label name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNameLabel(rtLabel_t label, const char *name);

/**
 * @ingroup dvrt_base
 * @brief label switch by index
 * @param [in] ptr  index value ptr
 * @param [in] max  index max value
 * @param [in] labelInfoPtr  label content info ptr
 * @param [in] stream  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief stream goto label
 * @param [in] label  goto label
 * @param [in] stream  stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGotoEx(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [in] label  model label list
 * @param [in] labelNumber  label number
 * @param [in] dst  device ptr
 * @param [in] dstMax  dst size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] label  created label handle
 * @param [in] stream  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] label  created label handle
 * @param [in] model  label bind model
 * @param [in] stream  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateExV2(rtLabel_t *label, rtModel_t model, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief get current thread last stream id and task id
 * @param [out] stream id and task id
 * @param [in] null
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for input null ptr
 */
RTS_API rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId);
#if defined(__cplusplus)
}
#endif

#endif  // __CCE_RUNTIME_BASE_H__
