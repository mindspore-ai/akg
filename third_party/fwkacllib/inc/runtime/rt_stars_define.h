/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: the definition of stars
 */

#ifndef CCE_RUNTIME_RT_STARS_DEFINE_H
#define CCE_RUNTIME_RT_STARS_DEFINE_H

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

typedef struct tagStarsSqeHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    uint8_t reserved : 1;

    uint16_t blockDim;

    uint16_t rtStreamId;
    uint16_t taskId;
} rtStarsSqeHeader_t;

typedef struct tagStarsDsaSqe {
    // 0-7 bytes
    rtStarsSqeHeader_t sqeHeader;
    // 8-11 bytes
    uint32_t start : 1;
    uint32_t functionType : 3;
    uint32_t dataType : 3;
    uint32_t algoType : 3;
    uint32_t paramVldBitmap : 5;
    uint32_t paramAddrValBitmap : 7;
    uint32_t reserved0 : 10;
    // 12-15 bytes
    uint16_t sqeIndex;
    uint8_t kernelCredit;
    uint8_t reserved1;
    // 16-31 bytes
    uint32_t dsaCfgResultAddrLow;
    uint32_t dsaCfgResultAddrHigh;
    uint32_t dsaCfgStateAddrLow;
    uint32_t dsaCfgStateAddrHigh;
    // 32-47 bytes
    uint32_t dsaCfgParamAddrLow;
    uint32_t dsaCfgParamAddrHigh;
    uint32_t dsaCfgSeedLow;
    uint32_t dsaCfgSeedHigh;
    // 48-63 bytes
    uint32_t dsaCfgNumberLow;
    uint32_t dsaCfgNumberHigh;
    uint32_t reserved2[2];
} rtStarsDsaSqe_t;

// ffts+ type
typedef enum tagFftsPlusType {
    RT_FFTS_PLUS_TYPE_RES1 = 2,   // Reserved
    RT_FFTS_PLUS_TYPE_RES2 = 3,   // Reserved
    RT_FFTS_PLUS_TYPE = 4,        // FFTS+ mode
} rtFftsPlusType_t;

typedef struct tagStarsFftsPlusHeader {
    uint8_t type : 6;
    uint8_t l1Lock : 1;
    uint8_t l1Unlock : 1;

    uint8_t ie : 2;
    uint8_t preP : 2;
    uint8_t postP : 2;
    uint8_t wrCqe : 1;
    /* tell mcu if this subgraph is overflow-enabled and mcu will send this flag to aicpu when aicpu ctx is excuted */
    uint8_t overflowEn : 1;

    uint16_t blockDim;

    uint16_t rtStreamId;
    uint16_t taskId;
} rtStarsFftsPlusHeader_t;
// ffts+ sqe
typedef struct tagFftsPlusSqe {
    // 0-7 bytes
    rtStarsSqeHeader_t sqeHeader; // use rtStarsFftsPlusHeader_t instead
    // 8-11 bytes
    uint16_t fftsType : 3;
    uint16_t reserved1 : 9;
    uint16_t wrrRatio : 4;
    uint16_t reserved2;
    // 12-15 bytes
    uint16_t sqeIndex;
    uint8_t  kernelCredit;
    uint8_t  reserved4;
    // 16-23 bytes
    uint32_t stackPhyBaseL;
    uint32_t stackPhyBaseH;
    // 24-31 bytes
    uint16_t  totalContextNum;
    uint16_t  readyContextNum;
    uint16_t  preloadContextNum;
    uint16_t  reserved5;
    // 32-35 bytes
    uint16_t  reserved6;
    uint16_t  prefetchOstNum : 5;
    uint16_t  reserved9 : 3;
    uint16_t  cmaintOstNum : 5;
    uint16_t  reserved10 : 3;
    // 36-39 bytes
    uint16_t  aicPrefetchLower : 5;
    uint16_t  reserved11 : 3;
    uint16_t  aicPrefetchUpper : 5;
    uint16_t  reserved12 : 3;
    uint16_t  aivPrefetchLower : 5;
    uint16_t  reserved13 : 3;
    uint16_t  aivPrefetchUpper : 5;
    uint16_t  reserved14 : 3;
    // 40-47 bytes
    uint32_t contextAddressBaseL;
    uint32_t contextAddressBaseH : 17;
    uint32_t reserved15 : 15;
    // 48-63 bytes
    uint32_t reserved16[4];
} rtFftsPlusSqe_t;

typedef struct tagCmoTaskInfo {
    uint8_t  qos;
    uint8_t  partId;
    uint8_t  pmg;
    uint8_t  reserved;
    uint16_t cmoType;
    uint16_t opCode;
    uint16_t numInner;
    uint16_t numOuter;
    uint32_t logicId;
    uint32_t lengthInner;
    uint64_t sourceAddr;
    uint32_t striderOuter;
    uint32_t striderInner;
} rtCmoTaskInfo_t;

typedef struct tagBarrierCmoInfo {
    uint16_t cmoType; // 0 is barrier, 1 is invalid, Prefetch is 2, Write_back is 3, FE/GE only use invalid type.
    uint32_t logicId;
} rtBarrierCmoInfo_t;

#define RT_CMO_MAX_BARRIER_NUM 6U // 6U is max support
typedef struct tagBarrierTaskInfo {
    uint8_t logicIdNum;
    rtBarrierCmoInfo_t cmoInfo[RT_CMO_MAX_BARRIER_NUM];
} rtBarrierTaskInfo_t;

#pragma pack(pop)

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_STARS_DEFINE_H