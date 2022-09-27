/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: rt_model.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_RT_MODEL_H
#define CCE_RUNTIME_RT_MODEL_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum tagModelTaskType {
    RT_MODEL_TASK_KERNEL = 0,
    RT_MODEL_TASK_EVENT_RECORD,
    RT_MODEL_TASK_EVENT_WAIT,
    RT_MODEL_TASK_FUSION_START,
    RT_MODEL_TASK_FUSION_END,
    RT_MODEL_TASK_KERNEL_EX,
    RT_MODEL_TASK_HCCL,
    RT_MODEL_TASK_STREAM_SWITCH,
    RT_MODEL_TASK_STREAM_ACTIVE,
    RT_MODEL_TASK_LABEL_SET,
    RT_MODEL_TASK_LABEL_SWITCH,
    RT_MODEL_TASK_LABEL_GOTO,
    RT_MODEL_TASK_PROFILER_TRACE,
    RT_MODEL_TASK_MEMCPY_ASYNC,
    RT_MODEL_TASK_NOTIFY_RECORD,
    RT_MODEL_TASK_NOTIFY_WAIT,
    RT_MODEL_TASK_REDUCE_ASYNC,
    RT_MODEL_TASK_RDMA_SEND,
    RT_MODEL_TASK_EVENT_RESET,
    RT_MODEL_TASK_MODEL_END_GRAPH,
    RT_MODEL_TASK_STREAM_SWITCH_N,
    RT_MODEL_TASK_RDMA_DB_SEND,
    RT_MODEL_TASK_MEMCPY_ADDR_ASYNC,
    RT_MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX,
    RT_MODEL_TASK_STREAM_LABEL_GOTO,
    RT_MODEL_TASK_MODEL_EXIT,
    RT_MODEL_TASK_ALL_KERNEL,
    RT_MODEL_TASK_PROFILER_TRACE_EX,
    RT_MODEL_TASK_FFTS_TASK,
    RT_MODEL_TASK_FFTS_PLUS_TASK,
    RT_MODEL_TASK_DSA_TASK,
    RT_MODEL_TASK_CMO,
    RT_MODEL_TASK_BARRIER,
    RT_MODEL_TASK_NPU_GET_FLOAT_STATUS,
    RT_MODEL_TASK_NPU_CLEAR_FLOAT_STATUS,
    RT_MODEL_TASK_DVPP,
} rtModelTaskType_t;

typedef enum tagModelStreamType {
    RT_MODEL_HEAD_STREAM = 0,
    RT_MODEL_WAIT_ACTIVE_STREAM = 1
} rtModelStreamType_t;

typedef enum tagModelQueueFlag {
    RT_MODEL_INPUT_QUEUE = 0,
    RT_MODEL_OUTPUT_QUEUE = 1
} rtModelQueueFlag_t;

#define EXECUTOR_NONE (0x0U)
#define EXECUTOR_TS (0x01U)
#define EXECUTOR_AICPU (0x02U)

/*
 * @ingroup rt_model
 * @brief debug flag for kernel exception dump
 */
#define RT_DEBUG_FLAG_AICORE_OVERFLOW (0x1U << 0U)
#define RT_DEBUG_FLAG_ATOMIC_ADD_OVERFLOW (0x1U << 1U)

/**
 * @ingroup
 * @brief the type defination of aicpu model task command
 */
typedef enum tagTsAicpuModelCmd {
    TS_AICPU_MODEL_LOAD = 1,
    TS_AICPU_MODEL_EXECUTE,
    TS_AICPU_MODEL_DESTROY,
    TS_AICPU_MODEL_ABORT,
    TS_AICPU_MODEL_RESERVED,
} tsAicpuModelCmd;

typedef struct tagAicpuTaskInfo {
    uint32_t taskID;
    uint32_t streamID;
    uint32_t kernelType;
    uint64_t kernelName;
    uint64_t kernelSo;
    uint64_t paraBase;
    uint32_t taskFlag;
} rtAicpuTaskInfo_t;

typedef struct tagModelStreamInfo {
    uint32_t streamID;
    uint32_t streamFlag;
} rtModelStreamInfo_t;

typedef struct tagModelQueueInfo {
    uint32_t queueID;
    uint32_t flag;
} rtModelQueueInfo_t;

typedef struct tagAicpuModelInfo {
    uint32_t moduleID;
    uint32_t tsId;
    uint16_t streamInfoNum;
    uint16_t aicpuTaskNum;
    uint64_t streamInfoPtr;
    uint64_t aicpuTaskPtr;
    uint16_t queueSize;
    uint64_t queueInfoPtr;
} rtAicpuModelInfo_t;

typedef struct tagKernelTaskInfo {
    uint16_t blockDim;
    uint16_t argsCount;
    uint16_t argsSize;
    uint16_t reserved;
    const char_t *stubFunc;
    uint8_t *smDesc;
    const uint8_t *args;
    uint16_t *argsOffset;
} rtKernelTaskInfo_t;

typedef struct tagAllKernelTaskInfo {
    uint16_t blockDim;
    uint16_t argsCount;
    uint16_t argsSize;
    uint16_t reserved;
    uint64_t tilingKey;
    void *handle;
    uint8_t *smDesc;
    const uint8_t *args;
    uint16_t *argsOffset;
} rtAllKernelTaskInfo_t;

typedef struct tagKernelTaskInfoEx {
    uint32_t flags;
    uint32_t argsSize;
    const void *args;
    uint32_t reserved[6];
} rtKernelTaskInfoEx_t;

typedef struct tagEventTaskInfo {
    uint32_t eventID;
    uint32_t reserved[9];
} rtEventTaskInfo_t;

typedef struct tagStreamSwitchTaskInfo {
    int64_t value;
    uint64_t pValuePtr;
    uint32_t trueStreamID;
    uint32_t dataType;
    uint32_t reserved[4];
} rtStreamSwitchTaskInfo_t;

typedef struct tagStreamSwitchNTaskInfo {
    uint64_t pValuePtr;
    uint64_t pTrueStreamPtr;
    uint32_t size;
    uint32_t elementSize;
    uint32_t dataType;
    uint32_t reserved[3];
} rtStreamSwitchNTaskInfo_t;

typedef struct tagStreamActiveTaskInfo {
    uint32_t activeStreamID;
    uint32_t reserved[9];
} rtStreamActiveTaskInfo_t;

typedef struct tagSetTaskInfo {
    uint16_t labelId;
    uint32_t reserved[9];
} rtLabelSetTaskInfo_t;

typedef struct tagSwitchTaskInfo {
    uint32_t value;
    uint32_t reserved[9];
} rtLabelSwitchTaskInfo_t;

typedef struct tagLabelGotoTaskInfo {
    uint16_t labelId;
    uint32_t reserved[9];
} rtLabelGotoTaskInfo_t;

typedef struct tagProfilerTraceTaskInfo {
    uint64_t profilerTraceId;
    uint32_t notify : 8;
    uint32_t reserved_ : 24;
    uint32_t flags;
    uint32_t reserved[6];
} rtProfilerTrace_t;

typedef struct tagProfilerTraceExTaskInfo {
    uint64_t profilerTraceId;
    uint64_t modelId;
    uint16_t tagId;
    uint8_t reserved[22];
} rtProfilerTraceEx_t;

typedef struct tagrtMemcpyAsyncTaskInfo {
    const void *dst;
    uint64_t destMax;
    const void *src;
    uint64_t count;
    uint32_t kind;
    uint32_t reserved;
} rtMemcpyAsyncTaskInfo_t;

typedef struct tagrtNotifyTaskInfo {
    uint32_t notifyID;
    uint32_t reserved[9];
} rtNotifyTaskInfo_t;

typedef struct tagrtReduceAsyncTaskInfo {
    const void *dst;
    uint64_t destMax;
    const void *src;
    uint64_t count;
    uint32_t kind;
    uint32_t type;
} rtReduceAsyncTaskInfo_t;

typedef struct tagrtRdmaSendTaskInfo {
    uint32_t index;
    uint32_t wqe_index;
    uint32_t reserved[8];
} rtRdmaSendTaskInfo_t;

typedef struct tagrtRdmaDbSendTaskInfo {
    uint64_t dbInfo;
    uint32_t dbIndex;
    uint32_t reserved[7]; // offset 7
} rtRdmaDbSendTaskInfo_t;

typedef struct tagrtModelEndGraphTaskInfo {
    uint32_t modelId;
    uint32_t executorFlag;
    uint32_t reserved[8];
} rtModelEndGraphTaskInfo_t;

typedef struct tagrtModelExitInfo {
    uint32_t modelId;
    uint32_t streamId;
    uint32_t reserved[8];
} rtModelExitTaskInfo_t;


typedef struct tagrtStreamLabelSwitchByIndexTask_t {
    uint64_t indexPtr;
    uint64_t labelInfoPtr;
    uint32_t max;
    uint8_t reserved[20];
} rtStreamLabelSwitchByIndexTask_t;

typedef struct tagrtStreamLabelGotoTask_t {
    uint16_t labelId;
    uint16_t modelId;
    uint8_t reserved[36];
} rtStreamLabelGotoTask_t;

typedef struct tagrtNpuGetFloatStatusTask_t {
    uint64_t outputAddr;
    uint64_t outputSize;
    uint32_t checkMode;
    uint8_t reserved[20];
} rtNpuGetFloatStatusTask_t;

typedef struct tagrtNpuClearFloatStatusTask_t {
    uint32_t checkMode;
    uint8_t reserved[36];
} rtNpuClearFloatStatusTask_t;

typedef struct tagTaskInfo {
    uint32_t type;
    uint32_t streamID;
    union {
        rtKernelTaskInfoEx_t kernelTaskEx;
        rtKernelTaskInfo_t kernelTask;
        rtAllKernelTaskInfo_t allKernelTask;
        rtEventTaskInfo_t eventTask;
        rtStreamSwitchTaskInfo_t streamSwitchTask;
        rtStreamActiveTaskInfo_t streamActiveTask;
        rtLabelSetTaskInfo_t labelSetTask;
        rtLabelSwitchTaskInfo_t labelSwitchTask;
        rtLabelGotoTaskInfo_t labelGotoTask;
        rtProfilerTrace_t profilertraceTask;
        rtProfilerTraceEx_t profilertraceExTask;
        rtMemcpyAsyncTaskInfo_t memcpyAsyncTask;
        rtNotifyTaskInfo_t notifyTask;
        rtReduceAsyncTaskInfo_t reduceAsyncTask;
        rtRdmaSendTaskInfo_t rdmaSendTask;
        rtRdmaDbSendTaskInfo_t rdmaDbSendTask;
        rtModelEndGraphTaskInfo_t modelEndGraphTask;
        rtModelExitTaskInfo_t modelExitTask;
        rtStreamSwitchNTaskInfo_t streamSwitchNTask;
        rtStreamLabelSwitchByIndexTask_t streamLabelSwitchIndexTask;
        rtStreamLabelGotoTask_t streamLabelGotoTask;
        rtNpuGetFloatStatusTask_t npuGetFloatStatusTask;
        rtNpuClearFloatStatusTask_t npuClearFloatStatusTask;
        uint32_t reserved[10];
    } u;
} rtTaskInfo_t;

typedef struct tagNodeInfo_t {
    uint32_t nodeIdx;
    uint32_t reserved[1];
} rtNodeInfo;

typedef struct tagHwtsInfo_t {
    uint16_t taskId;
    uint16_t sqExeHead;
    uint16_t streamExeHead;
    uint16_t reserved[2];
} rtHwtsInfo;

typedef struct tagLabelDevInfo_t {
    uint16_t modelId;
    uint16_t streamId;
    uint16_t labelId;
    union {
        rtNodeInfo nodeInfo;
        rtHwtsInfo hwtsInfo;
        uint16_t reserved[5];
    }u;
}rtLabelDevInfo;

typedef rtError_t (*rtTaskGenCallback)(rtModel_t mdl, rtTaskInfo_t *taskInfo);

/**
 * @ingroup rt_model
 * @brief set callback for generate model
 * @param [in] callBack   callback function
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetTaskGenCallback(rtTaskGenCallback callback);

/**
 * @ingroup rt_model
 * @brief create model instance
 * @param [out]    mdl     created model
 * @param [in]     flag    reserved
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelCreate(rtModel_t *mdl, uint32_t flag);

/**
 * @ingroup rt_model
 * @brief set ge model id to aicpu
 * @param [in]     model   aicpu model
 * @param [in]     extid   ge model id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtModelSetExtId(rtModel_t mdl, uint32_t extId);

/**
 * @ingroup rt_model
 * @brief destroy model instance
 * @param [in] mdl   model to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelDestroy(rtModel_t mdl);

/**
 * @ingroup rt_model
 * @brief bind model and stream instance
 * @param [in] mdl   binded model
 * @param [in] stm  binded stream
 * @param [in] flag    reserved
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelBindStream(rtModel_t mdl, rtStream_t stm, uint32_t flag);

/**
 * @ingroup rt_model
 * @brief unbind model and stream instance
 * @param [in] mdl   unbinded model
 * @param [in] stm  unbinded stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelUnbindStream(rtModel_t mdl, rtStream_t stm);

/**
 * @ingroup rt_model
 * @brief tell runtime Model has been Loaded
 * @param [in] mdl   model to execute
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtModelLoadComplete(rtModel_t mdl);

/**
 * @ingroup rt_model
 * @brief execute model instance
 * @param [in] mdl   model to execute
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelExecute(rtModel_t mdl, rtStream_t stm, uint32_t flag);

/**
 * @ingroup rt_model
 * @brief get model the last persist task id
 * @param [in] mdl   model to execute
 * @param [out] taskId last task id of the model
 * @param [out] streamId last steam id of the model
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelGetTaskId(rtModel_t mdl, uint32_t *taskId, uint32_t *streamId);

/**
 * @ingroup rt_model
 * @brief add a end graph task to stream
 * @param [in] mdl   model to execute
 * @param [in] end graph stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEndGraph(rtModel_t mdl, rtStream_t stm);

/**
 * @ingroup rt_model
 * @brief add a end graph task with flag to stream
 * @param [in] mdl   model to execute
 * @param [in] end graph stream
 * @param [in] flags   AICPU datadump
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtEndGraphEx(rtModel_t mdl, rtStream_t stm, uint32_t flags);

/**
 * @ingroup rt_model
 * @brief add a end graph task to stream
 * @param [in] mdl   model to execute
 * @param [in] flags EXECUTOR_TS | EXECUTOR_AICPU
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelExecutorSet(rtModel_t mdl, uint8_t flags);

/**
 * @ingroup rt_model
 * @brief abort model
 * @param [in] mdl   model to abort
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelAbort(rtModel_t mdl);

/**
 * @ingroup rt_model
 * @brief end graph task to model default stream
 * @param [in] mdl   model to execute
 * @param [in] end graph stream
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelExit(rtModel_t mdl, rtStream_t stm);

/**
 * @ingroup rt_model
 * @brief bind queue
 * @param [in] mdl     model to bind
 * @param [in] queueId   queueId to bind
 * @param [in] flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelBindQueue(rtModel_t mdl, uint32_t queueId, rtModelQueueFlag_t flag);

/**
 * @ingroup rt_model
 * @brief get model id
 * @param [in] mdl
 * @param [out] modelId   model id
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelGetId(rtModel_t mdl, uint32_t *modelId);

/*
 * @ingroup rt_model
 * @brief enable debug for dump overflow exception
 * @param [in] addr: ddr address of kernel exception dumpped
 * @param [in] mdl: model handle
 * @param [in] flag: debug flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDebugRegister(rtModel_t mdl, uint32_t flag, const void *addr,
                                  uint32_t *streamId, uint32_t *taskId);

/*
 * @ingroup rt_model
 * @brief disable debug for dump overflow exception
 * @param [in] mdl: model handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDebugUnRegister(rtModel_t mdl);

/**
 * @ingroup rt_model
 * @brief set model group id
 * @param [in]    mdl     model
 * @param [in]     schGrpId    groupId  (0,4) 0:default invalid value   1-4 valid value Maximum support 4 groups
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtModelSetSchGroupId(rtModel_t mdl, const int16_t schGrpId);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_RT_MODEL_H
