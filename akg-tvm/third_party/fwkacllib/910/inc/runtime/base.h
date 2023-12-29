/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: base.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_BASE_H
#define CCE_RUNTIME_BASE_H

#include <stdbool.h>
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

#ifndef char_t
typedef char char_t;
#endif

#ifndef float32_t
typedef float float32_t;
#endif

#ifndef float64_t
typedef double float64_t;
#endif

/**
 * @ingroup dvrt_base
 * @brief device mode.
 */
typedef enum tagRtDeviceMode {
    RT_DEVICE_MODE_SINGLE_DIE = 0,
    RT_DEVICE_MODE_MULTI_DIE,
    RT_DEVICE_MODE_RESERVED
} rtDeviceMode;

/**
 * @ingroup dvrt_base
 * @brief device status.
 */
typedef enum tagRtDeviceStatus {
    RT_DEVICE_STATUS_NORMAL = 0,
    RT_DEVICE_STATUS_ABNORMAL,
    RT_DEVICE_STATUS_END = 0xFFFF
} rtDeviceStatus;

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

typedef enum schemModeType {
    RT_SCHEM_MODE_NORMAL = 0,
    RT_SCHEM_MODE_BATCH,
    RT_SCHEM_MODE_SYNC,
    RT_SCHEM_MODE_END
} rtschemModeType_t;

typedef struct tagRtTaskCfgInfo {
    uint8_t qos;
    uint8_t partId;
    uint8_t schemMode; // rtschemModeType_t 0:normal;1:batch;2:sync
    uint8_t res[1]; // res
} rtTaskCfgInfo_t;

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
    RT_INVALID_FLAG = 0x7FFFFFFF,
} rtStreamFlagType_t;

typedef enum tagRtLimitType {
    RT_LIMIT_TYPE_LOW_POWER_TIMEOUT = 0,  // timeout for power down , ms
} rtLimitType_t;

typedef enum tagRtFloatOverflowMode {
    RT_OVERFLOW_MODE_SATURATION = 0,
    RT_OVERFLOW_MODE_INFNAN,
    RT_OVERFLOW_MODE_UNDEF,
} rtFloatOverflowMode_t;

typedef enum tagRtExceptionExpandType {
    RT_EXCEPTION_INVALID = 0,
    RT_EXCEPTION_FFTS_PLUS,
} rtExceptionExpandType_t;

typedef struct rtFftsPlusExDetailInfo {
    uint16_t contextId;
    uint16_t threadId;
} rtFftsPlusExDetailInfo_t;

typedef struct rtExceptionExpandInfo {
    rtExceptionExpandType_t type;
    union {
        rtFftsPlusExDetailInfo_t fftsPlusInfo;
    } u;
} rtExceptionExpandInfo_t;

typedef struct rtExceptionInfo {
    uint32_t taskid;
    uint32_t streamid;
    uint32_t tid;
    uint32_t deviceid;
    uint32_t retcode;
    rtExceptionExpandInfo_t expandInfo;
} rtExceptionInfo_t;

typedef void (*rtErrorCallback)(rtExceptionType);

typedef void (*rtTaskFailCallback)(rtExceptionInfo_t *exceptionInfo);

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
 * @brief Program handle.
 */
typedef void *rtBinHandle;

/**
 * @ingroup dvrt_base
 * @brief Kernel handle.
 */
typedef void *rtFuncHandle;

/**
 * @ingroup dvrt_base
 * @brief Args handle.
 */
typedef void *rtLaunchArgsHandle;

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

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
typedef struct rtCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char_t path[PATH_LEN_MAX + 1];
    char_t profData[PARAM_LEN_MAX + 1];
} rtCommandHandleParams_t;

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
    rtCommandHandleParams_t commandHandleParams;
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

typedef enum {
    RT_UTIL_TYPE_AICORE = 0,
    RT_UTIL_TYPE_AIVECTOR,
    RT_UTIL_TYPE_AICPU,
    RT_UTIL_TYPE_MAX
} rtTypeUtil_t;

/**
 * @ingroup profiling_base
 * @brief runtime handle.
 */
RTS_API rtError_t rtSetProfDirEx(const char_t *profDir, const char_t *address, const char_t *jobCtx);

/**
 * @ingroup profiling_base
 * @brief init profiler object.
 */
RTS_API rtError_t rtProfilerInit(const char_t *profDir, const char_t *address, const char_t *jobCtx);

/**
 * @ingroup profiling_base
 * @brief config rts profiler.
 */
RTS_API rtError_t rtProfilerConfig(uint16_t profConfig);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTrace(uint64_t id, bool notify, uint32_t flags, rtStream_t stm);

/**
 * @ingroup profiling_base
 * @brief ts send keypoint profiler log.
 */
RTS_API rtError_t rtProfilerTraceEx(uint64_t id, uint64_t modelId, uint16_t tagId, rtStream_t stm);

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
 * @ingroup profiling_base
 * @brief set profling switch, called by profiling
 * @param [in]  data  rtProfilingCommandHandle
 * @param [in]  len   length of data
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for error input
 */
RTS_API rtError_t rtProfilingCommandHandle(uint32_t type, void *data, uint32_t len);

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

typedef enum DevCallBackDir {
    DEV_CB_POS_FRONT = 1,
    DEV_CB_POS_BACK = 2,
    DEV_CB_POS_END
} rtDevCallBackDir_t;
/**
 * @ingroup dvrt_base
 * @brief register callback for deviceid by position
 * @param [in] regName unique register name, can't be null
 * @param [in] callback Device state callback function
 * @param [in] notifyPos callback notify Postion
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtRegDeviceStateCallbackEx(const char_t *regName, rtDeviceStateCallback callback,
    const rtDevCallBackDir_t notifyPos);

/**
 * @ingroup dvrt_base
 * @brief register callback for deviceid
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback Device state callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtRegDeviceStateCallback(const char_t *regName, rtDeviceStateCallback callback);

/**
 * @ingroup dvrt_base
 * @brief register callback for fail task
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback fail task callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtRegTaskFailCallbackByModule(const char_t *moduleName, rtTaskFailCallback callback);

/**
 * @ingroup dvrt_base
 * @brief notify handle.
 */
typedef void *rtNotify_t;

/**
 * @ingroup dvrt_base
 * @brief create label instance
 * @param [out]    lbl   created label
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreate(rtLabel_t *lbl);

/**
 * @ingroup dvrt_base
 * @brief create label instance
 * @param [out] lbl  created label
 * @param [in] mdl  label set model
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateV2(rtLabel_t *lbl, rtModel_t mdl);

/**
 * @ingroup dvrt_base
 * @brief set label and stream instance
 * @param [in] lbl   set label
 * @param [in] stm  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSet(rtLabel_t lbl, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief destroy label instance
 * @param [in] lbl   label to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelDestroy(rtLabel_t lbl);

/**
 * @ingroup dvrt_base
 * @brief label switch instance
 * @param [in] ptr  address to get value compared
 * @param [in] condition
 * @param [in] val  to compare
 * @param [in] true_label   goto label
 * @param [in] stm  to submit label_switch task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitch(void *ptr, rtCondition_t condition, uint32_t val, rtLabel_t trueLabel,
                                rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief goto label instance
 * @param [in] lbl   goto label
 * @param [in] stm  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGoto(rtLabel_t lbl, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief name label instance
 * @param [in] lbl  instance
 * @param [in] name  label name
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNameLabel(rtLabel_t lbl, const char_t *name);

/**
 * @ingroup dvrt_base
 * @brief label switch by index
 * @param [in] ptr  index value ptr
 * @param [in] maxValue  index max value
 * @param [in] labelInfoPtr  label content info ptr
 * @param [in] stm  set stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelSwitchByIndex(void *ptr, uint32_t maxValue, void *labelInfoPtr, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief stream goto label
 * @param [in] lbl  goto label
 * @param [in] stm  stream  to submit label_goto task
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelGotoEx(rtLabel_t lbl, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [in] lbl  model label list
 * @param [in] labelNumber  label number
 * @param [in] dst  device ptr
 * @param [in] dstMax  dst size
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelListCpy(rtLabel_t *lbl, uint32_t labelNumber, void *dst, uint32_t dstMax);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] lbl  created label handle
 * @param [in] stm  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateEx(rtLabel_t *lbl, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief labels to dev info
 * @param [out] lbl  created label handle
 * @param [in] mdl  label bind model
 * @param [in] stm  label bind stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtLabelCreateExV2(rtLabel_t *lbl, rtModel_t mdl, rtStream_t stm);

/**
 * @ingroup dvrt_base
 * @brief get current thread last stream id and task id
 * @param [out] stm id and task id
 * @param [in] null
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for input null ptr
 */
RTS_API rtError_t rtGetTaskIdAndStreamID(uint32_t *taskId, uint32_t *streamId);

/**
 * @ingroup dvrt_base
 * @brief get max model num
 * @param [out] max model num
 * @param [in] null
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtGetMaxModelNum(uint32_t *maxModelCount);

/**
 * @ingroup dvrt_base
 * @brief set stream mode
 * @param [in] stm  stream needed to be set mode
 * @param [in] stmMode mode
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtStreamSetMode(rtStream_t stm, const uint64_t stmMode);

/**
 * @ingroup dvrt_base
 * @brief get stream mode
 * @param [in] stm  stream needed to get its mode
 * @param [out] stmMode mode pointer
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtStreamGetMode(rtStream_t const stm, uint64_t * const stmMode);

#define RT_PROCESS_SIGN_LENGTH (49)

typedef enum tagRtDevDrvProcessType {
    RT_DEVDRV_PROCESS_CP1 = 0,   /* aicpu_scheduler */
    RT_DEVDRV_PROCESS_CP2,       /* custom_process */
    RT_DEVDRV_PROCESS_DEV_ONLY,  /* TDT */
    RT_DEVDRV_PROCESS_QS,        /* queue_scheduler */
    RT_DEVDRV_PROCESS_HCCP,      /* hccp server */
    RT_DEVDRV_PROCESS_USER,      /* user proc, can bind many on host or device */
    RT_DEVDRV_PROCESS_CPTYPE_MAX
} rtDevDrvProcessType_t;

typedef struct tagRtBindHostpidInfo {
    int32_t hostPid;
    uint32_t vfId;
    uint32_t chipId;
    int32_t mode;
    rtDevDrvProcessType_t cpType;
    uint32_t len;
    char sign[RT_PROCESS_SIGN_LENGTH];
} rtBindHostpidInfo;

/**
 * @ingroup dvrt_base
 * @brief Bind Device custom-process to aicpu-process.
 * @param [in] info The Information about the bound hostid.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtBindHostPid(rtBindHostpidInfo info);

/**
 * @ingroup dvrt_base
 * @brief Unbind Device custom-process to aicpu-process.
 * @param [in] info The Information about the bound hostid.
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtUnbindHostPid(rtBindHostpidInfo info);

/**
 * @ingroup dvrt_base
 * @brief Query the binding information of the devpid.
 * @param [in] pid: dev pid
 * @param [in] chipId chip id
 * @param [in] vfId vf id
 * @param [in] hostPid host pid
 * @param [in] cpType type of custom-process
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
RTS_API rtError_t rtQueryProcessHostPid(int32_t pid, uint32_t *chipId, uint32_t *vfId, uint32_t *hostPid,
    uint32_t *cpType);
#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_BASE_H
