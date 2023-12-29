/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: ffts interface
 */

#ifndef CCE_RUNTIME_RT_FFTS_H
#define CCE_RUNTIME_RT_FFTS_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

#define RT_FFTS_MAX_SUB_TASK_NUM    32U
#define RT_FFTS_MAX_TICKET_CACHE_NUM    64U
#define RT_FFTS_MAX_MANUAL_THREAD_NUM   16U
#define RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK    8U
#define RT_FFTS_MANUAL_SRC_DEPEND_TBL_LEN    32U

typedef enum tagFftsType {
    RT_FFTS_TYPE_AUTO_THREAD = 2,   // ffts auto thread mode, same as ffts define
    RT_FFTS_TYPE_MANUAL_THREAD = 3, // ffts manual thread mode, same as ffts define
} rtFftsType_t;

typedef enum tagFftsSubTaskType {
    RT_FFTS_SUB_TASK_TYPE_AIC = 0,
    RT_FFTS_SUB_TASK_TYPE_AIV = 1,
    RT_FFTS_SUB_TASK_TYPE_NOP = 2,
    RT_FFTS_SUB_TASK_TYPE_NOTIFY_WAIT = 3,
    RT_FFTS_SUB_TASK_TYPE_NOTIFY_RECORD = 4,
    RT_FFTS_SUB_TASK_TYPE_WRITE_VALUE = 5,
    RT_FFTS_SUB_TASK_TYPE_MIX_AIC = 6,
    RT_FFTS_SUB_TASK_TYPE_MIX_AIV = 7,
    RT_FFTS_SUB_TASK_TYPE_SDMA = 8,
    RT_FFTS_SUB_TASK_TYPE_RESERVED = 9,
} rtFftsSubTaskType_t;

typedef struct tagManualThreadDmuInfo {
    uint64_t dataAddr; // device mem
    uint16_t numOuter;
    uint16_t numInner;
    uint32_t strideOuter;
    uint32_t lenInner;
    uint32_t strideInner;
} rtManualThreadDmuInfo_t;

typedef struct tagManualThreadDependency {
    uint8_t dependency[RT_FFTS_MANUAL_SRC_DEPEND_TBL_LEN];
} rtManualThreadDependency_t;

typedef struct tagManualThreadAicAivInfo {
    uint64_t taskParamAddr; // device mem
    uint16_t taskParamOffset;
    // when satMode=1 and FP16 computation with none INF inputs overflows/underflows, results will be +/-INF of FP16
    // when satMode=0 and FP16 computation with none INF inputs overflows/underflows,
    // results will be saturated to +/-MAX of FP16
    uint8_t satMode;
    uint8_t scheduleMode;   // 0:normal mode, 1:batch mode, 2:sync mode 3:reserved
    uint8_t iCachePrefetchCnt; // units is 2K
    uint8_t prefetchEnableBitmap; // 8 bit bitmap  1 0 1 0
    uint8_t prefetchOnceBitmap; // 8 bit bitmap  1 0 1 0
    uint16_t prefetchOnceDmuNum; // prefetch_once_dmu_descriptor_index in ffts
    // num： thread0_prefetch_dmu_descriptor_index – prefetch_once_dmu_descriptor_index
    uint16_t threadPrefetchDmuIdx[RT_FFTS_MAX_MANUAL_THREAD_NUM]; // max valid is threadDim
    uint16_t threadBlkDim[RT_FFTS_MAX_MANUAL_THREAD_NUM];
    const char_t *threadTaskFuncStub[RT_FFTS_MAX_MANUAL_THREAD_NUM];

    rtManualThreadDmuInfo_t *prefetchList; // dmu desc 0-64k, length is the last threadPrefetchDmuIdx[threadDim-1]
    rtManualThreadDependency_t srcDepTbl[RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK];
} rtManualThreadAicAivInfo_t;

typedef struct tagAutoThreadPrefetch {
    uint64_t dataAddr; // device mem
    uint32_t dataAddrOffset;
    uint32_t nonTailDataLen;
    uint32_t tailDataLen;
} rtAutoThreadPrefetch_t;

typedef struct tagAutoThreadAicAivInfo {
    uint64_t taskParamAddr; // device mem
    uint16_t taskParamOffset;
    /*
     * when satMode=1 and FP16 computation with none INF inputs overflows/underflows, results will be +/-INF of FP16
     * when satMode=0 and FP16 computation with none INF inputs overflows/underflows, results will be saturated to
     *     +/-MAX of FP16
     */
    uint8_t satMode;
    uint8_t scheduleMode;   // 0:normal mode, 1:batch mode, 2:sync mode 3:reserved
    uint8_t iCachePrefetchCnt; // units is 2K
    uint8_t prefetchEnableBitmap;   // 8 bit bitmap
    uint8_t prefetchOnceBitmap;     // 8 bit bitmap

    uint16_t tailBlkDim;
    uint16_t nonTailBlkDim;

    const char_t *nonTailTaskFuncStub;
    const char_t *tailTaskFuncStub;

    // for prefetch, valid num is prefetchEnableBitmap bit count.
    // if prefetchEnableBitmap='00010011', need prefetch number is 3, srcPrefetch is only 0, 1, 2 is valid
    rtAutoThreadPrefetch_t srcPrefetch[RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK];
} rtAutoThreadAicAivInfo_t;

typedef struct tagAutoThreadCacheInfo {
    uint64_t dataAddr; // device mem
    uint32_t dataAddrOffset;
    uint32_t nonTailDataLen;
    uint32_t tailDataLen;
    uint16_t ticketCacheRefCnt;
} rtAutoThreadCacheInfo_t;

typedef struct tagManualThreadCacheInfo {
    rtManualThreadDmuInfo_t *dmuList;  // 0-64k
    uint16_t dmuNum;
    uint16_t sliceDmuIdx[RT_FFTS_MAX_MANUAL_THREAD_NUM];
    uint16_t ticketCacheRefCntTbl[RT_FFTS_MAX_MANUAL_THREAD_NUM];
} rtManualThreadCacheInfo_t;

typedef enum tagCacheOp {
    RT_CACHE_OP_NONE = 0,
    RT_CACHE_OP_FLUSH = 1,
    RT_CACHE_OP_INVALIDATE = 2,
    RT_CACHE_OP_WRITE_BACK = 3,
} rtCacheOp_t;

typedef struct tagTicketCache {
    rtCacheOp_t cacheOption;
    uint8_t ticketCacheWindow;
    union {
        rtAutoThreadCacheInfo_t autoThreadCache;
        rtManualThreadCacheInfo_t manualThreadCache;
    } custom;
} rtTicketCache_t;

typedef struct tagManualThreadNopInfo {
    // depend srcTickCacheVldBitmap in rtFftsSubTaskInfo_t
    rtManualThreadDependency_t srcDepTbl[RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK];
} rtManualThreadNopInfo_t;

typedef struct tagFftsSubTaskInfo {
    rtFftsSubTaskType_t subTaskType;
    uint16_t threadDim;
    uint8_t dstTickCacheVldBitmap;
    uint8_t srcTickCacheVldBitmap;
    uint8_t srcDataOutOfSubGraphBitmap;
    uint8_t dstTickCacheID[RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK];
    uint8_t srcTickCacheID[RT_FFTS_MAX_TICKET_CACHE_PER_SUBTASK];
    union {
        rtAutoThreadAicAivInfo_t autoThreadAicAiv;
        rtManualThreadAicAivInfo_t manualThreadAicAiv;
        rtManualThreadNopInfo_t manualThreadNop;
    } custom;
} rtFftsSubTaskInfo_t;

typedef struct tagFftsDescInfo {
    uint8_t tm; // thread subtask kickstart mode, 0:order, 1:disorder
    uint8_t di; // discard invalidate
    uint8_t dw; // discard write back
    uint8_t df; // discard flush
    uint8_t dataSplitUnit;  // split source or ticket cache by 2^dataSplitUnit MB
    uint8_t prefetchOstNum;
    uint8_t cacheMaintainOstNum;
    uint8_t aicPrefetchUpper;
    uint8_t aicPrefetchLower;
    uint8_t aivPrefetchUpper;
    uint8_t aivPrefetchLower;
} rtFftsDescInfo_t;

typedef struct tagFftsTaskInfo {
    rtFftsType_t fftsType;
    uint16_t subTaskNum;
    uint16_t tickCacheNum;
    rtFftsDescInfo_t fftsDesc;
    // sub task desc, real num is subTaskNum
    rtFftsSubTaskInfo_t subTask[RT_FFTS_MAX_SUB_TASK_NUM];

    // ticket cache, real number is tickCacheNum.
    rtTicketCache_t ticketCache[RT_FFTS_MAX_TICKET_CACHE_NUM];
} rtFftsTaskInfo_t;

RTS_API rtError_t rtFftsTaskLaunch(rtFftsTaskInfo_t *fftsTaskInfo, rtStream_t stm);
RTS_API rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);

RTS_API rtError_t rtFftsTaskLaunchWithFlag(rtFftsTaskInfo_t *fftsTaskInfo, rtStream_t stm, uint32_t flag);

#if defined(__cplusplus)
}
#endif
#endif // CCE_RUNTIME_RT_FFTS_H
