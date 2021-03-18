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

#ifdef __cplusplus
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

/**
 * @ingroup dvrt_base
 * @brief runtime error numbers.
 */
typedef enum tagRtError {
    RT_ERROR_NONE = 0x0,                      // success
    RT_ERROR_INVALID_VALUE = 0x1,             // invalid value
    RT_ERROR_MEMORY_ALLOCATION = 0x2,         // memory allocation fail
    RT_ERROR_INVALID_RESOURCE_HANDLE = 0x3,   // invalid handle
    RT_ERROR_INVALID_DEVICE_POINTER = 0x4,    // invalid device point
    RT_ERROR_INVALID_MEMCPY_DIRECTION = 0x5,  // invalid memory copy dirction
    RT_ERROR_INVALID_DEVICE = 0x6,            // invalid device
    RT_ERROR_NO_DEVICE = 0x7,                 // no valid device
    RT_ERROR_CMD_OCCUPY_FAILURE = 0x8,        // command occpuy failure
    RT_ERROR_SET_SIGNAL_FAILURE = 0x9,        // set signal failure
    RT_ERROR_UNSET_SIGNAL_FAILURE = 0xA,      // unset signal failure
    RT_ERROR_OPEN_FILE_FAILURE = 0xB,         // unset signal failure
    RT_ERROR_WRITE_FILE_FAILURE = 0xC,
    RT_ERROR_MEMORY_ADDRESS_UNALIGNED = 0xD,
    RT_ERROR_DRV_ERR = 0xE,
    RT_ERROR_LOST_HEARTBEAT = 0xF,
    RT_ERROR_REPORT_TIMEOUT = 0x10,
    RT_ERROR_NOT_READY = 0x11,
    RT_ERROR_DATA_OPERATION_FAIL = 0x12,
    RT_ERROR_INVALID_L2_INSTR_SIZE = 0x13,
    RT_ERROR_DEVICE_PROC_HANG_OUT = 0x14,
    RT_ERROR_DEVICE_POWER_UP_FAIL = 0x15,
    RT_ERROR_DEVICE_POWER_DOWN_FAIL = 0x16,
    RT_ERROR_FEATURE_NOT_SUPPROT = 0x17,
    RT_ERROR_KERNEL_DUPLICATE = 0x18,         // register same kernel repeatly
    RT_ERROR_MODEL_STREAM_EXE_FAILED = 0x91,  // the model stream failed
    RT_ERROR_MODEL_LOAD_FAILED = 0x94,        // the model stream failed
    RT_ERROR_END_OF_SEQUENCE = 0x95,          // end of sequence
    RT_ERROR_NO_STREAM_CB_REG = 0x96,         // no callback register info for stream
    RT_ERROR_DATA_DUMP_LOAD_FAILED = 0x97,    // data dump load info fail
    RT_ERROR_CALLBACK_THREAD_UNSUBSTRIBE = 0x98,    // callback thread unsubstribe
    RT_ERROR_RESERVED
} rtError_t;

/**
 * @ingroup dvrt_base
 * @brief runtime exception numbers.
 */
typedef enum tagRtExceptionType {
  RT_EXCEPTION_NONE = 0,
  RT_EXCEPTION_TS_DOWN = 1,
  RT_EXCEPTION_TASK_TIMEOUT = 2,
  RT_EXCEPTION_TASK_FAILURE = 3,
  RT_EXCEPTION_DEV_RUNNING_DOWN = 4
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

typedef void (*rtErrorCallback)(rtExceptionType);

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
 * @ingroup profiling_base
 * @brief runtime handle.
 */
RTS_API rtError_t rtSetProfDirEx(const char *profDir, const char *address, const char *jobCtx);

/**
 * @ingroup profiling_base
 * @brief init profiler object.
 */
RTS_API rtError_t rtProfilerInit(const char *profdir, const char *address, const char *job_ctx);

/**
 * @ingroup profiling_base
 * @brief start rts profiler.
 */
RTS_API rtError_t rtProfilerStart(void);

/**
 * @ingroup profiling_base
 * @brief stop rts profiler.
 */
RTS_API rtError_t rtProfilerStop(void);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stream);

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
 * @brief set polling receive mode for task report
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetPollingMode();

/**
 * @ingroup dvrt_base
 * @brief register callback for error code
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetExceptCallback(rtErrorCallback callback);

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
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelCreate(rtLabel_t *label);

/**
 * @ingroup dvrt_base
 * @brief set label and stream instance
 * @param [in] label   set label
 * @param [in] stream  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelSet(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief destroy label instance
 * @param [in] label   label to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
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
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t value, rtLabel_t trueLabel,
                                rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief goto label instance
 * @param [in] label   goto label
 * @param [in] stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelGoto(rtLabel_t label, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief name label instance
 * @param [in] label  instance
 * @param [in] name  label name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
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
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t max, void *labelInfoPtr, rtStream_t stream);

/**
 * @ingroup dvrt_base
 * @brief stream goto label
 * @param [in] label  goto label
 * @param [in] stream  stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
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
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelListCpy(rtLabel_t *label, uint32_t labelNumber, void *dst, uint32_t dstMax);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] label  created label handle
 * @param [in] stream  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input handle
 */
RTS_API rtError_t rtLabelCreateEx(rtLabel_t *label, rtStream_t stream);
#ifdef __cplusplus
}
#endif

#endif  // __CCE_RUNTIME_BASE_H__
