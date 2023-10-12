/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: the definition of ffts plus
 */

#ifndef CCE_RUNTIME_RT_FFTS_PLUS_DEFINE_H
#define CCE_RUNTIME_RT_FFTS_PLUS_DEFINE_H

#include "base.h"

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
extern "C" {
#endif

#pragma pack(push)
#pragma pack (1)

// hardware context type
typedef enum tagFftsPlusHwType {
    RT_HW_CTX_TYPE_AIC = 0,
    RT_HW_CTX_TYPE_AIV = 1,
    RT_HW_CTX_TYPE_NOTIFY_WAIT = 3,
    RT_HW_CTX_TYPE_NOTIFY_RECORD = 4,
    RT_HW_CTX_TYPE_WRITE_VALUE = 5,
    RT_HW_CTX_TYPE_MIX_AIC = 6,
    RT_HW_CTX_TYPE_MIX_AIV = 7,
    RT_HW_CTX_TYPE_SDMA = 8,
    RT_HW_CTX_TYPE_FLUSH_DATA = 9,
    RT_HW_CTX_TYPE_INVALIDATE_DATA = 10,
    RT_HW_CTX_TYPE_WRITEBACK_DATA = 11,
    RT_HW_CTX_TYPE_AICPU = 12,
    RT_HW_CTX_TYPE_LOAD = 13,
    RT_HW_CTX_TYPE_MAX = 14,
} rtFftsPlusHwType_t;

// hardware context type
typedef enum tagFftsPlusSoftType {
    RT_SOFT_CTX_TYPE_COND_SWITCH = 1,
    RT_SOFT_CTX_TYPE_CASE_SWITCH = 2,
    RT_SOFT_CTX_TYPE_AT_START = 3,
    RT_SOFT_CTX_TYPE_AT_END = 4,
    RT_SOFT_CTX_TYPE_LABEL = 5,
    RT_SOFT_CTX_PERSISTENT_CACHE = 6,
    RT_SOFT_CTX_DSA = 7,
    RT_SOFT_CTX_TYPE_MAX = 8,
} rtFftsPlusSoftType_t;

typedef enum tagFftsPlusContextType {
    RT_CTX_TYPE_AICORE = 0x0000,
    RT_CTX_TYPE_AIV = 0x0001,
    RT_CTX_TYPE_NOTIFY_WAIT = 0x0003,
    RT_CTX_TYPE_NOTIFY_RECORD = 0x0004,
    RT_CTX_TYPE_WRITE_VALUE = 0x0005,
    RT_CTX_TYPE_MIX_AIC = 0x0006,
    RT_CTX_TYPE_MIX_AIV = 0x0007,
    RT_CTX_TYPE_SDMA = 0x0008,
    RT_CTX_TYPE_FLUSH_DATA = 0x0009,
    RT_CTX_TYPE_INVALIDATE_DATA = 0x000A,
    RT_CTX_TYPE_WRITEBACK_DATA = 0x000B,
    RT_CTX_TYPE_AICPU = 0x000C,
    RT_CTX_TYPE_COND_SWITCH = 0x010D,
    RT_CTX_TYPE_CASE_SWITCH = 0x020D,
    RT_CTX_TYPE_AT_START = 0x0300,
    RT_CTX_TYPE_AT_END = 0x0400,
    RT_CTX_TYPE_LABEL = 0x0500,
    RT_CTX_TYPE_PERSISTENT_CACHE = 0x0600,
    RT_CTX_TYPE_DSA = 0x0700,
}rtFftsPlusContextType_t;

// condition type
typedef enum tagFftsPlusCondType {
    RT_COND_TYPE_EQUAL = 0,
    RT_COND_TYPE_NOTEQUAL = 1,
    RT_COND_TYPE_GREATER = 2,
    RT_COND_TYPE_GREATER_OR_EQUAL = 3,
    RT_COND_TYPE_LESS = 4,
    RT_COND_TYPE_LESS_OR_EQUAL = 5,
    RT_COND_TYPE_MAX = 6,
} rtFftsPlusCondType_t;

// the definition of ffts plus context

#define RT_CTX_SUCCESSOR_NUM   26

// ffts plus common context
typedef struct tagFftsPlusComCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t rsv1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t rsv2;
    uint8_t rsv3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t rsv4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-71
    uint32_t rsv5[2];
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-127
    uint32_t res6[13];
} rtFftsPlusComCtx_t;

// aic/aiv context
typedef struct tagFftsPlusAicAivCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t resv : 6;
    uint8_t dumpSwitch : 1;
    uint8_t aten : 1;
    // 4-7
    uint8_t prefetchConfig;
    uint8_t resv1;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint16_t resv2;
    uint16_t policyPri;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t resv3 : 1;
    uint16_t schem : 2;
    uint16_t icachePrefetchCnt : 5;
    uint16_t resv4 : 7;
    uint16_t atm : 1;
    uint16_t prefetchEnableBitmap : 4;
    uint16_t res6 : 4;
    uint16_t prefetchOnceBitmap : 4;
    uint16_t res7 : 4;
    // 68-71
    uint16_t pmg : 2;
    uint16_t ns : 1;
    uint16_t partId : 8;
    uint16_t res8 : 1;
    uint16_t qos : 4;
    uint16_t res9;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t nonTailBlockdim;
    uint16_t tailBlockdim;
    // 80-83
    uint32_t taskParamPtrBaseL;
    // 84-87
    uint16_t taskParamPtrBaseH;
    uint16_t taskParamPtrOffset;
    // 88-95
    uint32_t res10;
    uint32_t res11;
    // 96-103
    uint32_t nonTailTaskStartPcL;
    uint16_t nonTailTaskStartPcH;
    uint16_t res12;
    // 104-111
    uint32_t tailTaskStartPcL;
    uint16_t tailTaskStartPcH;
    uint16_t res13;
    // 112-119
    uint32_t res14;
    uint32_t res15;
    // 120-127
    uint16_t srcSlot[4];    // src_slot0-3(context ID for source data which is out of subgraph)
} rtFftsPlusAicAivCtx_t;

// mix aic/aiv context
typedef struct tagFftsPlusMixAicAivCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t reserved1 : 6;
    uint8_t dumpSwitch : 1;
    uint8_t aten : 1;
    // 4-7
    uint8_t prefetchConfig;
    uint8_t reserved2;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint16_t reserved3;
    uint16_t policyPri;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t reserved4 : 1;
    uint16_t schem : 2;
    uint16_t aicIcachePrefetchCnt : 5;
    uint16_t aivIcachePrefetchCnt : 5;
    uint16_t reserved5 : 2;
    uint16_t atm : 1;
    uint16_t prefetchEnableBitmap : 4;
    uint16_t reserved6 : 4;
    uint16_t prefetchOnceBitmap : 4;
    uint16_t reserved7 : 4;
    // 68-71
    uint16_t pmg : 2;
    uint16_t ns : 1;
    uint16_t partId : 8;
    uint16_t reserved8 : 1;
    uint16_t qos : 4;
    uint8_t nonTailBlockRatioN;
    uint8_t tailBlockRatioN;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t nonTailBlockdim;
    uint16_t tailBlockdim;
    // 80-87
    uint32_t aicTaskParamPtrL;
    uint16_t aicTaskParamPtrH;
    uint16_t aicTaskParamPtrOffset;
    // 88-95
    uint32_t aivTaskParamPtrL;
    uint16_t aivTaskParamPtrH;
    uint16_t aivTaskParamPtrOffset;
    // 96-103
    uint32_t nonTailAicTaskStartPcL;
    uint16_t nonTailAicTaskStartPcH;
    uint16_t tailAicTaskStartPcH;
    // 104-111
    uint32_t tailAicTaskStartPcL;
    uint32_t nonTailAivTaskStartPcL;
    // 112-119
    uint16_t nonTailAivTaskStartPcH;
    uint16_t tailAivTaskStartPcH;
    uint32_t tailAivTaskStartPcL;
    // 120-127
    uint16_t srcSlot[4];    // src_slot0-3(context ID for source data which is out of subgraph)
} rtFftsPlusMixAicAivCtx_t;

// sdma context
typedef struct tagFftsPlusSdmaCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 6;
    uint8_t dumpSwitch : 1;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2;
    uint8_t res3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint8_t res5;
    uint8_t res6 : 7;
    uint8_t atm : 1;
    uint16_t res7;
    // 68-71
    uint16_t pmg : 2;
    uint16_t ns : 1;
    uint16_t partId : 8;
    uint16_t res8 : 1;
    uint16_t qos : 4;
    uint16_t res9;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint32_t sdmaSqeHeader;  // (FORMAT/MPAMNS/PARTID/DRO/SRO/QOS/DNS/SNS/DSSV/SSSV/IE/UPCODE)
    // 80-83
    uint16_t sourceStreamId;
    uint16_t sourceSubstreamId;
    // 84-87
    uint16_t destinationStreamId;
    uint16_t destinationSubstreamId;
    // 88-127
    uint32_t sourceAddressBaseL;
    uint32_t sourceAddressBaseH;
    uint32_t sourceAddressOffset;
    uint32_t destinationAddressBaseL;
    uint32_t destinationAddressBaseH;
    uint32_t destinationAddressOffset;
    uint32_t nonTailDataLength;
    uint32_t tailDataLength;
    uint32_t res10[2];
} rtFftsPlusSdmaCtx_t;

// ffts plus notify record/wait context
typedef struct tagFftsPlusNotifyCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t res1;
    uint8_t res2;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res3;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t res4 : 14;
    uint16_t satm : 1;
    uint16_t atm : 1;
    uint16_t res6;
    // 68-71
    uint32_t res7;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t notifyIdBase;
    uint8_t autoWindow;
    uint8_t res8;
    // 80-127
    uint32_t res9[4];
    uint16_t notifyId[16];
} rtFftsPlusNotifyCtx_t;

// write Value context
typedef struct tagFftsPlusWriteValueCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t resv1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t resv2;
    uint8_t resv3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t resv4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t resv5 : 15;
    uint16_t atm : 1;
    uint16_t resv6;
    // 68-71
    uint32_t resv7;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint8_t awSize : 3;
    uint8_t awSnoop : 1;
    uint8_t resv8 : 4;
    uint8_t awCache : 4;
    uint8_t awProt : 3;
    uint8_t awVa : 1;

    uint8_t arSize : 3;
    uint8_t arSnoop : 1;
    uint8_t resv9 : 4;
    uint8_t arCache : 4;
    uint8_t arProt : 3;
    uint8_t arVa : 1;
    // 80-83
    uint32_t writeAddressBaseL;
    // 84-87
    uint32_t writeAddressBaseH : 17;
    uint32_t res10 : 15;
    // 88-91
    uint32_t writeAddressOffset;
    // 92-95
    uint32_t res11;
    // 96-111
    uint32_t writeValue[4]; // write_value_00 -> write_value_03
    // 112-127
    uint32_t res12[4];
} rtFftsPlusWriteValueCtx_t;

// ai cpu context
typedef struct tagFftsPlusAiCpuCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 6;
    uint8_t dumpSwitch : 1;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2;
    uint8_t res3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res4;
    // 12-63
    uint16_t successorContextID[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t res5 : 15;
    uint16_t atm : 1;
    uint16_t res6;
    // 68-71
    uint16_t sqeIndex;
    uint8_t kernelType : 7;
    uint8_t bm : 1;
    uint8_t topicType : 4;
    uint8_t qos : 3;
    uint8_t res7 : 1;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t nonTailBlockdim;
    uint16_t tailBlockdim;
    // 80-115
    uint32_t usrData[9];   // usr_data0 -> usr_data8 usr_data2(task_param_base_l) usr_data3(task_param_base_h)
    // 116--119
    uint32_t res8;
    // 120-123
    uint32_t subtopicId : 12;
    uint32_t topicId : 6;
    uint32_t groupId : 6;
    uint32_t usrDataLength : 8;
    // 124-127
    uint32_t taskParamOffset;
} rtFftsPlusAiCpuCtx_t;

// data context
typedef struct tagFftsPlusDataCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2;
    uint8_t res3;
    uint8_t cntInit; // cons_cnt_init / prod_cnt_init
    uint8_t cnt;     // cons_cnt / prod_cnt
    // 8-11
    uint32_t res4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t res5 : 15;
    uint16_t atm : 1;
    uint16_t res6;
    // 68-71
    uint16_t pmg : 2;
    uint16_t ns : 1;
    uint16_t partId : 8;
    uint16_t res7 : 1;
    uint16_t qos : 4;
    uint16_t res8;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t origConsumerCounter;
    uint16_t runConsumerCounter;
    // 80-83
    uint32_t addressBaseL;
    // 84-87
    uint32_t addressBaseH;
    // 88-91
    uint32_t addressOffset;
    // 92-95
    uint32_t res9;
    // 96-99
    uint16_t nonTailNumOutter;
    uint16_t nonTailNumInner;
    // 100-103
    uint32_t nonTailLengthInner;
    // 104-107
    uint32_t nonTailStrideOutter;
    // 108-111
    uint32_t nonTailStrideInner;
    // 112-115
    uint16_t tailNumOutter;
    uint16_t tailNumInner;
    // 116-119
    uint32_t tailLengthInner;
    // 120-123
    uint32_t tailStrideOutter;
    // 124-127
    uint32_t tailStrideInner;
} rtFftsPlusDataCtx_t;

// at start context
typedef struct tagFftsPlusAtStartCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t rs1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t rs2;
    uint8_t rs3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t rs4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t rs5;
    uint16_t rs6;
    // 68-71
    uint16_t rs7;
    uint16_t rs8;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint16_t threadIdInit;
    uint16_t threadWindowSize;
    // 80-127
    uint32_t res9[12];
} rtFftsPlusAtStartCtx_t;

// at end context
#define RT_CTX_SUCC_AT_START_SLOT_NUM   12
#define RT_CTX_SUCC_OUT_LABEL_SLOT_NUM  12

typedef struct tagFftsPlusAtEndCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t atStartSlotNumber;
    uint8_t outLabelSlotNumber : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t res1;
    uint8_t res2;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res3;
    // 12-59
    uint16_t succAtStartSlot[RT_CTX_SUCC_AT_START_SLOT_NUM];
    uint16_t succOutLabelSlot[RT_CTX_SUCC_OUT_LABEL_SLOT_NUM];
    // 60-63
    uint16_t res4;
    uint16_t res5;
    // 64-67
    uint16_t res6;
    uint16_t res7;
    // 68-71
    uint16_t res8;
    uint16_t res9;
    // 72-75
    uint16_t threadId;
    uint16_t res10;
    // 76-79
    uint16_t res11;
    uint16_t res12;
    // 80-127
    uint32_t res13[12];
} rtFftsPlusAtEndCtx_t;

// label context
typedef struct tagFftsPlusLabelCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2;
    uint8_t res3;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res4;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-71
    uint32_t res5[2];
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;

    // 76-127
    uint32_t res6[13];
} rtFftsPlusLabelCtx_t;

// case switch context
typedef struct tagFftsPlusCaseSwitchCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t resv0 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t startLabelId;
    uint8_t labelListLen;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t resv1;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t resv2 : 15;
    uint16_t atm : 1;
    uint16_t resv3;
    // 68-71
    uint32_t resv4;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint8_t arSize : 3;
    uint8_t snoop : 1;
    uint8_t resv5 : 4;
    uint8_t arCache : 4;
    uint8_t arProt : 3;
    uint8_t va : 1;
    uint16_t resv6;
    // 80-83
    uint32_t loadAddress0BaseL;
    // 84-87
    uint32_t loadAddress0BaseH : 17;
    uint32_t resv7 : 14;
    uint32_t ld0En : 1;
    // 88-91
    uint32_t loadAddress0Offset;
    // 92-95
    uint32_t resv8;
    // 96-99
    uint32_t loadAddress1BaseL;
    // 100-103
    uint32_t loadAddress1BaseH : 17;
    uint32_t resv9 : 14;
    uint32_t ld1En : 1;
    // 104-107
    uint32_t loadAddress1Offset;
    // 108-127
    uint32_t resv10[5];
} rtFftsPlusCaseSwitchCtx_t;

// case default context
typedef struct tagFftsPlusCaseDefCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t rs0 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t startLabelId;
    uint8_t labelListLen;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t rs1;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint16_t rs2;
    uint16_t rs3;
    // 68-127
    uint32_t rs4[15];
} rtFftsPlusCaseDefCtx_t;

// condition switch context
#define RT_CTX_TRUE_SUCCESSOR_NUM 13
#define RT_CTX_FALSE_SUCCESSOR_NUM 13

typedef struct tagFftsPlusCondSwitchCtx {
    // 0-3 bytes
    uint16_t contextType;
    uint8_t trueSuccessorNum;
    uint8_t falseSuccessorNum : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t condition;
    uint8_t res1;
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res2;
    // 12-63
    uint16_t trueSuccessorList[RT_CTX_TRUE_SUCCESSOR_NUM];
    uint16_t falseSuccessorList[RT_CTX_FALSE_SUCCESSOR_NUM];
    // 64-67
    uint16_t res3 : 15;
    uint16_t atm : 1;
    uint16_t res4;
    // 68-71
    uint32_t res5;
    // 72-75
    uint16_t threadId;
    uint16_t threadDim;
    // 76-79
    uint8_t arSize : 3;
    uint8_t snoop : 1;
    uint8_t res6 : 4;
    uint8_t arCache : 4;
    uint8_t arProt : 3;
    uint8_t va : 1;
    uint16_t res7;
    // 80-83
    uint32_t loadAddress0BaseL;
    // 84-87
    uint32_t loadAddress0BaseH : 17;
    uint32_t res8 : 14;
    uint32_t ld0En : 1;
    // 88-91
    uint32_t loadAddress0Offset;
    // 92-95
    uint32_t res9;
    // 96-99
    uint32_t loadAddress1BaseL;
    // 100-103
    uint32_t loadAddress1BaseH : 17;
    uint32_t res10 : 14;
    uint32_t ld1En : 1;
    // 104-107
    uint32_t loadAddress1Offset;
    // 108-127
    uint32_t res11[3];
    uint32_t cmpValue1;
    uint32_t cmpValue2;
} rtFftsPlusCondSwitchCtx_t;

// ffts plus persistent cache context
typedef struct tagFftsPlusPersistentCacheCtx {
    // 0- 3bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 7;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2[2];
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint8_t res3[4];
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];
    // 64-67
    uint8_t persistentEnable : 1;
    uint8_t res4 : 7;
    uint8_t res5;
    uint16_t persistentSize;
    // 68-71
    uint32_t persistentId;
    // 72-127
    uint32_t res6[14];
} rtFftsPlusPersistentCacheCtx_t;


typedef struct tagFftsPlusDsaCtx {
    // 0-3bytes
    uint16_t contextType;
    uint8_t successorNum;
    uint8_t res1 : 6;
    uint8_t dumpSwitch : 1;
    uint8_t aten : 1;
    // 4-7
    uint8_t res2[2];
    uint8_t predCntInit;
    uint8_t predCnt;
    // 8-11
    uint32_t res3;
    // 12-63
    uint16_t successorList[RT_CTX_SUCCESSOR_NUM];

    // bottom half 64B
    // 0-3 bytes
    uint16_t numberOffset;
    uint16_t addressOffset;
    // 4-7 bytes
    uint32_t res5;
    // 8-11 bytes
    uint16_t threadId;
    uint16_t threadDim;
    // 12-15 bytes
    uint32_t start : 1;
    uint32_t functionType : 3;
    uint32_t dataType : 3;
    uint32_t algoType : 3;
    uint32_t paramVldBitmap : 5;
    uint32_t paramAddrValBitmap : 7;
    uint32_t res6 : 10;
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
    uint32_t res7[2];
} rtFftsPlusDsaCtx_t;

#pragma pack(pop)

#if defined(__cplusplus) && !defined(COMPILE_OMG_PACKAGE)
}
#endif
#endif // CCE_RUNTIME_RT_FFTS_PLUS_DEFINE_H
