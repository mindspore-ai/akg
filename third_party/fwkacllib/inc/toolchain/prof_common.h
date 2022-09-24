/**
 * @file prof_common.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */
#ifndef MSPROFILER_PROF_COMMON_H_
#define MSPROFILER_PROF_COMMON_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define MSPROF_DATA_HEAD_MAGIC_NUM  0x5a5a

enum MsprofDataTag {
    MSPROF_ACL_DATA_TAG = 0,            // acl data tag, range: 0~19
    MSPROF_GE_DATA_TAG_MODEL_LOAD = 20, // ge data tag, range: 20~39
    MSPROF_GE_DATA_TAG_FUSION = 21,
    MSPROF_GE_DATA_TAG_INFER = 22,
    MSPROF_GE_DATA_TAG_TASK = 23,
    MSPROF_GE_DATA_TAG_TENSOR = 24,
    MSPROF_GE_DATA_TAG_STEP = 25,
    MSPROF_GE_DATA_TAG_ID_MAP = 26,
    MSPROF_GE_DATA_TAG_HOST_SCH = 27,
    MSPROF_RUNTIME_DATA_TAG_API = 40,   // runtime data tag, range: 40~59
    MSPROF_RUNTIME_DATA_TAG_TRACK = 41,
    MSPROF_AICPU_DATA_TAG = 60,         // aicpu data tag, range: 60~79
    MSPROF_AICPU_MODEL_TAG = 61,
    MSPROF_HCCL_DATA_TAG = 80,          // hccl data tag, range: 80~99
    MSPROF_DP_DATA_TAG = 100,           // dp data tag, range: 100~119
    MSPROF_MSPROFTX_DATA_TAG = 120,     // hccl data tag, range: 120~139
    MSPROF_DATA_TAG_MAX = 65536,        // data tag value type is uint16_t
};

/**
 * @brief struct of mixed data
 */
#define MSPROF_MIX_DATA_RESERVE_BYTES 7
#define MSPROF_MIX_DATA_STRING_LEN 120
enum MsprofMixDataType {
    MSPROF_MIX_DATA_HASH_ID = 0,
    MSPROF_MIX_DATA_STRING,
};
struct MsprofMixData {
    uint8_t type;  // MsprofMixDataType
    uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
    union {
        uint64_t hashId;
        char dataStr[MSPROF_MIX_DATA_STRING_LEN];
    } data;
};

#define PATH_LEN_MAX 1023
#define PARAM_LEN_MAX 4095
struct MsprofCommandHandleParams {
    uint32_t pathLen;
    uint32_t storageLimit;  // MB
    uint32_t profDataLen;
    char path[PATH_LEN_MAX + 1];
    char profData[PARAM_LEN_MAX + 1];
};

/**
 * @brief profiling command info
 */
#define MSPROF_MAX_DEV_NUM 64
struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint64_t profSwitchHi;
    uint32_t devNums;
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
    uint32_t type;
    struct MsprofCommandHandleParams params;
};

/**
 * @brief struct of data reported by acl
 */
#define MSPROF_ACL_DATA_RESERVE_BYTES 32
#define MSPROF_ACL_API_NAME_LEN 64
enum MsprofAclApiType {
    MSPROF_ACL_API_TYPE_OP = 1,
    MSPROF_ACL_API_TYPE_MODEL,
    MSPROF_ACL_API_TYPE_RUNTIME,
    MSPROF_ACL_API_TYPE_OTHERS,
};
struct MsprofAclProfData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_ACL_DATA_TAG;
    uint32_t apiType;       // enum MsprofAclApiType
    uint64_t beginTime;
    uint64_t endTime;
    uint32_t processId;
    uint32_t threadId;
    char apiName[MSPROF_ACL_API_NAME_LEN];
    uint8_t  reserve[MSPROF_ACL_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by GE
 */
#define MSPROF_GE_MODELLOAD_DATA_RESERVE_BYTES 104
struct MsprofGeProfModelLoadData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_MODEL_LOAD;
    uint32_t modelId;
    MsprofMixData modelName;
    uint64_t startTime;
    uint64_t endTime;
    uint8_t  reserve[MSPROF_GE_MODELLOAD_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_FUSION_DATA_RESERVE_BYTES 8
#define MSPROF_GE_FUSION_OP_NUM 8
struct MsprofGeProfFusionData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_FUSION;
    uint32_t modelId;
    MsprofMixData fusionName;
    uint64_t inputMemSize;
    uint64_t outputMemSize;
    uint64_t weightMemSize;
    uint64_t workspaceMemSize;
    uint64_t totalMemSize;
    uint64_t fusionOpNum;
    uint64_t fusionOp[MSPROF_GE_FUSION_OP_NUM];
    uint8_t  reserve[MSPROF_GE_FUSION_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_INFER_DATA_RESERVE_BYTES 64
struct MsprofGeProfInferData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_INFER;
    uint32_t modelId;
    MsprofMixData modelName;
    uint32_t requestId;
    uint32_t threadId;
    uint64_t inputDataStartTime;
    uint64_t inputDataEndTime;
    uint64_t inferStartTime;
    uint64_t inferEndTime;
    uint64_t outputDataStartTime;
    uint64_t outputDataEndTime;
    uint8_t  reserve[MSPROF_GE_INFER_DATA_RESERVE_BYTES];
};

constexpr int32_t MSPROF_GE_TASK_DATA_RESERVE_BYTES = 12;
#define MSPROF_GE_OP_TYPE_LEN 56
enum MsprofGeTaskType {
    MSPROF_GE_TASK_TYPE_AI_CORE = 0,
    MSPROF_GE_TASK_TYPE_AI_CPU,
    MSPROF_GE_TASK_TYPE_AIV,
    MSPROF_GE_TASK_TYPE_WRITE_BACK,
    MSPROF_GE_TASK_TYPE_MIX_AIC,
    MSPROF_GE_TASK_TYPE_MIX_AIV,
    MSPROF_GE_TASK_TYPE_FFTS_PLUS,
    MSPROF_GE_TASK_TYPE_DSA,
    MSPROF_GE_TASK_TYPE_DVPP,
    MSPROF_GE_TASK_TYPE_INVALID
};

enum MsprofGeShapeType {
    MSPROF_GE_SHAPE_TYPE_STATIC = 0,
    MSPROF_GE_SHAPE_TYPE_DYNAMIC,
};
struct MsprofGeOpType {
    uint8_t type;  // MsprofMixDataType
    uint8_t rsv[MSPROF_MIX_DATA_RESERVE_BYTES];
    union {
        uint64_t hashId;
        char dataStr[MSPROF_GE_OP_TYPE_LEN];
    } data;
};
struct MsprofGeProfTaskData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_TASK;
    uint32_t taskType;      // MsprofGeTaskType
    MsprofMixData opName;
    MsprofGeOpType opType;
    uint64_t curIterNum;
    uint64_t timeStamp;
    uint32_t shapeType;     // MsprofGeShapeType
    uint32_t blockDims;
    uint32_t modelId;
    uint32_t streamId;
    uint32_t taskId;
    uint32_t threadId;
    uint32_t contextId;
    uint8_t  reserve[MSPROF_GE_TASK_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_TENSOR_DATA_RESERVE_BYTES 8
#define MSPROF_GE_TENSOR_DATA_SHAPE_LEN 8
#define MSPROF_GE_TENSOR_DATA_NUM 5
enum MsprofGeTensorType {
    MSPROF_GE_TENSOR_TYPE_INPUT = 0,
    MSPROF_GE_TENSOR_TYPE_OUTPUT,
};
struct MsprofGeTensorData {
    uint32_t tensorType;    // MsprofGeTensorType
    uint32_t format;
    uint32_t dataType;
    uint32_t shape[MSPROF_GE_TENSOR_DATA_SHAPE_LEN];
};

struct MsprofGeProfTensorData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_TENSOR;
    uint32_t modelId;
    uint64_t curIterNum;
    uint32_t streamId;
    uint32_t taskId;
    uint32_t tensorNum;
    MsprofGeTensorData tensorData[MSPROF_GE_TENSOR_DATA_NUM];
    uint8_t  reserve[MSPROF_GE_TENSOR_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_STEP_DATA_RESERVE_BYTES 27
enum MsprofGeStepTag {
    MSPROF_GE_STEP_TAG_BEGIN = 0,
    MSPROF_GE_STEP_TAG_END,
};
struct MsprofGeProfStepData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_STEP;
    uint32_t modelId;
    uint32_t streamId;
    uint32_t taskId;
    uint64_t timeStamp;
    uint64_t curIterNum;
    uint32_t threadId;
    uint8_t  tag;           // MsprofGeStepTag
    uint8_t  reserve[MSPROF_GE_STEP_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_ID_MAP_DATA_RESERVE_BYTES 6
struct MsprofGeProfIdMapData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_ID_MAP;
    uint32_t graphId;
    uint32_t modelId;
    uint32_t sessionId;
    uint64_t timeStamp;
    uint16_t mode;
    uint8_t  reserve[MSPROF_GE_ID_MAP_DATA_RESERVE_BYTES];
};

#define MSPROF_GE_HOST_SCH_DATA_RESERVE_BYTES 24
struct MsprofGeProfHostSchData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_GE_DATA_TAG_HOST_SCH;
    uint32_t threadId;      // record in start event
    uint64_t element;
    uint64_t event;
    uint64_t startTime;     // record in start event
    uint64_t endTime;       // record in end event
    uint8_t  reserve[MSPROF_GE_HOST_SCH_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by RunTime
 */
#define MSPROF_RUNTIME_API_DATA_RESERVE_BYTES 106
#define MSPROF_RUNTIME_TASK_ID_NUM 10
#define MSPROF_RUNTIME_API_NAME_LEN 64
struct MsprofRuntimeProfApiData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_RUNTIME_DATA_TAG_API;
    uint32_t threadId;
    uint64_t entryTime;
    uint64_t exitTime;
    uint64_t dataSize;
    uint8_t  apiName[MSPROF_RUNTIME_API_NAME_LEN];
    uint32_t retCode;
    uint32_t streamId;
    uint32_t taskNum;
    uint32_t taskId[MSPROF_RUNTIME_TASK_ID_NUM];
    uint16_t memcpyDirection;
    uint8_t  reserve[MSPROF_RUNTIME_API_DATA_RESERVE_BYTES];
};

#define MSPROF_RUNTIME_TRACK_DATA_RESERVE_BYTES 10
#define MSPROF_RUNTIME_TRACK_TASK_TYPE_LEN 32
struct MsprofRuntimeProfTrackData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_RUNTIME_DATA_TAG_TRACK;
    uint32_t threadId;
    uint64_t timeStamp;
    char taskType[MSPROF_RUNTIME_TRACK_TASK_TYPE_LEN];
    uint32_t taskId;
    uint16_t streamId;
    uint8_t  reserve[MSPROF_RUNTIME_TRACK_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by RunTime
 */
#define MSPROF_AICPU_DATA_RESERVE_BYTES 9
struct MsprofAicpuProfData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_AICPU_DATA_TAG;
    uint16_t streamId;
    uint16_t taskId;
    uint64_t runStartTime;
    uint64_t runStartTick;
    uint64_t computeStartTime;
    uint64_t memcpyStartTime;
    uint64_t memcpyEndTime;
    uint64_t runEndTime;
    uint64_t runEndTick;
    uint32_t threadId;
    uint32_t deviceId;
    uint64_t submitTick;
    uint64_t scheduleTick;
    uint64_t tickBeforeRun;
    uint64_t tickAfterRun;
    uint32_t kernelType;
    uint32_t dispatchTime;
    uint32_t totalTime;
    uint16_t fftsThreadId;
    uint8_t  version;
    uint8_t  reserve[MSPROF_AICPU_DATA_RESERVE_BYTES];
};

struct MsprofAicpuModelProfData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_AICPU_MODEL_TAG;
    uint32_t rsv;   // Ensure 8-byte alignment
    uint64_t timeStamp;
    uint64_t indexId;
    uint32_t modelId;
    uint16_t tagId;
    uint16_t rsv1;
    uint64_t eventId;
    uint8_t  reserve[24];
};

/**
 * @brief struct of data reported by DP
 */
#define MSPROF_DP_DATA_RESERVE_BYTES 16
#define MSPROF_DP_DATA_ACTION_LEN 16
#define MSPROF_DP_DATA_SOURCE_LEN 64
struct MsprofDpProfData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_DP_DATA_TAG;
    uint32_t rsv;   // Ensure 8-byte alignment
    uint64_t timeStamp;
    char action[MSPROF_DP_DATA_ACTION_LEN];
    char source[MSPROF_DP_DATA_SOURCE_LEN];
    uint64_t index;
    uint64_t size;
    uint8_t  reserve[MSPROF_DP_DATA_RESERVE_BYTES];
};

/**
 * @brief struct of data reported by HCCL
 */
#pragma pack(4)
struct MsprofHcclProfNotify {
    uint32_t taskID;
    uint64_t notifyID;
    uint32_t stage;
    uint32_t remoteRank;
    uint32_t transportType;
    uint32_t role; // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfReduce {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint32_t op;            // {0: sum, 1: mul, 2: max, 3: min}
    uint32_t dataType;      // data type {0: INT8, 1: INT16, 2: INT32, 3: FP16, 4:FP32, 5:INT64, 6:UINT64}
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: SDMA, 1: RDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfRDMA {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t notifyID;
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: RDMA, 1:SDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    uint32_t type;          // RDMA type {0: RDMASendNotify, 1:RDMASendPayload}
    double durationEstimated;
};

struct MsprofHcclProfMemcpy {
    uint32_t taskID;
    uint64_t src;
    uint64_t dst;
    uint64_t size;
    uint64_t notifyID;
    uint32_t linkType;      // link type {0: 'OnChip', 1: 'HCCS', 2: 'PCIe', 3: 'RoCE'}
    uint32_t remoteRank;
    uint32_t transportType; // transport type {0: RDMA, 1:SDMA, 2:LOCAL}
    uint32_t role;          // role {0: dst, 1:src}
    double durationEstimated;
};

struct MsprofHcclProfStageStep {
    uint32_t rank;
    uint32_t rankSize;
};

struct MsprofHcclProfFlag {
    uint64_t cclTag;
    uint64_t groupName;
    uint32_t localRank;
    uint32_t workFlowMode;
};

/**
 * @name MsprofHcclProfData
 * @brief struct of data reported by hccl
 */
struct MsprofHcclProfData {
    uint16_t magicNumber = MSPROF_DATA_HEAD_MAGIC_NUM;
    uint16_t dataTag = MSPROF_HCCL_DATA_TAG;
    uint32_t planeID;
    uint32_t deviceID;
    uint32_t streamID;
    double ts;
    char name[16];
    union {
        MsprofHcclProfNotify notify;
        MsprofHcclProfReduce reduce;
        MsprofHcclProfStageStep stageStep;
        MsprofHcclProfMemcpy forMemcpy;
        MsprofHcclProfRDMA RDMA;
        MsprofHcclProfFlag flag;
    } args;
};
#pragma pack()

/**
 * @name  MsprofStampInfo
 * @brief struct of data reported by msproftx
 */
struct MsprofStampInfo {
    uint16_t magicNumber;
    uint16_t dataTag;
    uint32_t processId;
    uint32_t threadId;
    uint32_t category;    // marker category
    uint32_t  eventType;
    int32_t payloadType;
    union PayloadValue {
        uint64_t ullValue;
        int64_t llValue;
        double dValue;
        uint32_t uiValue[2];
        int32_t iValue[2];
        float fValue[2];
    } payload;            // payload info for marker
    uint64_t startTime;
    uint64_t endTime;
    int32_t messageType;
    char message[128];
    uint8_t reserve0[4];
    uint8_t reserve1[72];
};

#ifdef __cplusplus
}
#endif

#endif  // MSPROFILER_PROF_COMMON_H_
