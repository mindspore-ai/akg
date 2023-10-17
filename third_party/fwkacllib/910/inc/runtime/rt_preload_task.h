/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: parma rt_preload_task.h
 * Create: 2023-06-02
 */
#ifndef CCE_RUNTIME_RT_PRELOAD_TASK_H
#define CCE_RUNTIME_RT_PRELOAD_TASK_H

#include "runtime/base.h"

#if defined(__cplusplus)
extern "C" {
#endif

typedef enum tagRtTaskBuffType {
    HWTS_STATIC_TASK_DESC = 0,       /**< static task */
    HWTS_DYNAMIC_TASK_DESC = 1,      /**< dynamic task */
    PARAM_TASK_INFO_DESC = 2,        /**< parma task */
    MAX_TASK,
} rtTaskBuffType_t;

typedef enum tagRtTaskType {
    RT_TASK_TYPE_KERNEL_AICORE = 0,
    RT_TASK_TYPE_KERNEL_AICPU = 1,
    RT_TASK_TYPE_KERNEL_NANO_AICORE = 2,
    RT_TASK_TYPE_KERNEL_NANO_AICPU_HOSTFUNC = 3,
    RT_TASK_TYPE_MAX,
} rtTaskType_t;

typedef struct {
    uint64_t kernelBinOffset;
    uint64_t argsOffset;                           // need add rtTaskInput_t.argOffset
    uint32_t literalBuffLen;
    uint16_t blockDim;
    uint8_t kernelFlag;
} rtAicoreTaskParam_t;

typedef struct {
    uint16_t type;
    uint16_t preP;
    uint16_t posP;
    uint16_t dump;
    uint16_t conds;
    uint16_t uf;
    uint16_t sw;
    uint16_t prefetchNum;
    uint16_t softUser;
    uint16_t kernelCredit;
    uint32_t taskParamOffset;                      // need add rtTaskInput_t.argOffset
} rtHwtsStaticTaskDesc_t;

typedef struct {
    uint8_t vld;
    uint32_t codeSize;
    uint32_t dynTaskDescSize;
    uint32_t blockDim;
    uint32_t taskPcOffset;
} rtHwtsDynamicTaskDesc_t;

typedef struct {
    uint32_t opType;
    uint32_t dataSize;
    uint32_t dstOffset;
    uint32_t srcOffset;
} rtPrefetchBufInfo_t;

typedef struct {
    rtPrefetchBufInfo_t prefetchBufInfo[16];
    uint16_t prefetchBufSize;
    uint64_t paramBufInfo[16];
    uint16_t paramBufSize;
    uint32_t bufSize;
    void* bufInfo;
} rtParamBufDesc_t;

typedef struct {
    rtTaskBuffType_t type;
    union {
        rtHwtsStaticTaskDesc_t hwtsTaskDesc;
        rtHwtsDynamicTaskDesc_t hwtsDynamicTaskDesc;
        rtParamBufDesc_t paramBufDesc;
    }u;
} rtNanoDefaultTaskParam_t;

typedef struct {
    rtTaskType_t taskType;
    rtTaskBuffType_t bufType;
    uint16_t streamId;
    union {
        rtAicoreTaskParam_t aicoreTask;
        rtNanoDefaultTaskParam_t nanoAicoreTask;
        rtNanoDefaultTaskParam_t nanoHostFuncTask;
    }u;
} rtCompilerPartinfo_t;

typedef struct {
    void* dataBuffer;                       // current write addr
    uint32_t bufferLen;                     // the space of dataBuffer left
    rtCompilerPartinfo_t compilerInfo;      // task info for complie
    uint64_t argOffset;                     // args offset
} rtTaskInput_t;

/**
 * @ingroup rt_preload_task
 * @brief exeom task build
 * @param [in] type               task type
 * @param [out] bufferLen         the space of dataBuffer left
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetTaskBufferLen(const rtTaskBuffType_t type, uint32_t * const bufferLen);

/**
 * @ingroup rt_preload_task
 * @brief exeom task build
 * @param [in] taskInput        task info for complie
 * @param [out] taskLen         current tasklen
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtTaskBuild(const rtTaskInput_t * const taskInput, uint32_t* taskLen);

/**
 * @ingroup rt_preload_task
 * @brief get elf header offset
 * @param [in] elfData   kernel bin addr
 * @param [in] elfLen
 * @param [out] offset   elf header offset
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetElfOffset(void * const elfData, const uint32_t elfLen, uint32_t* offset);

/**
 * @ingroup rt_preload_task
 * @brief not use now, return RT_ERROR_NONE
 * @param [in] isHuge
 * @param [out] bufferLen
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetStreamBufferLen(const bool isHuge, uint32_t * const bufferLen);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_RT_PRELOAD_TASK_H
