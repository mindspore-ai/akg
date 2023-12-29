/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef AICPUSD_AICPUSD_INFO_H
#define AICPUSD_AICPUSD_INFO_H

#include <cstdint>
#include <sched.h>
#include <sys/types.h>
#include "../common/type_def.h"

extern "C" {
struct __attribute__((visibility("default"))) AICPUActiveStream {
    uint32_t streamId;
};

static const uint32_t MAX_CUST_SO_NAME_LEN = 128U;
static const uint32_t OP_NAME_MAX_LEN = 50U;
static const uint32_t FUN_NAME_MAX_LEN = 30U;
static const uint32_t ERROR_KEY_INFO_MAX_LEN = 30U;
static const uint32_t MODULE_NAME_MAX_LEN = 10U;
static const uint32_t FILE_NAME_MAX_LEN = 30U;
static const uint16_t PRIORITY_MSG_CHECKCODE = 0xABCD;
static const int32_t  INVALID_ESCAPE_PRI_VALUE = -1;
// loadOpFromBuf task args
struct __attribute__((visibility("default"))) LoadOpFromBufArgs {
    uint64_t kernelSoBuf;        // the starting address of custom operator so buf
    uint32_t kernelSoBufLen;     // the length of custom operator so buf
    uint64_t kernelSoName;       // the starting address of custom operator so name
    uint32_t kernelSoNameLen;    // the length of custom operator so name
} __attribute__((packed));

// batchLoadOpFromBuf task args
struct __attribute__((visibility("default"))) BatchLoadOpFromBufArgs {
    uint32_t soNum;              // the number of so
    uint64_t opInfoArgs;
} __attribute__((packed));

/**
 * The mode of profiling
 */
enum __attribute__((visibility("default"))) ProfilingMode {
    PROFILING_CLOSE = 0,
    PROFILING_OPEN,
};

enum __attribute__((visibility("default"))) AICPUSubEvent {
    AICPU_SUB_EVENT_ACTIVE_STREAM = 0,
    AICPU_SUB_EVENT_EXECUTE_MODEL,
    AICPU_SUB_EVENT_REPEAT_MODEL,
    AICPU_SUB_EVENT_RECOVERY_STREAM,
    AICPU_SUB_EVENT_UPDATE_PROFILING_MODE,
    AICPU_SUB_EVENT_LOAD_SO,
    AICPU_SUB_EVENT_END_GRAPH,
    AICPU_SUB_EVENT_ACTIVE_MODEL,
    AICPU_SUB_EVENT_PREPARE_MEM,
    AICPU_SUB_EVENT_TABLE_UNLOCK,
    AICPU_SUB_EVENT_MAX_NUM,
};

enum __attribute__((visibility("default"))) AICPUCustSubEvent {
    // sub type begin with 10 for interface event
    AICPU_SUB_EVENT_BIND_SD_PID = 10,     // cust-sd bind sd pid, Implemented by cust-sd
    AICPU_SUB_EVENT_OPEN_CUSTOM_SO,  // open costom so file, Implemented by cust-sd
    AICPU_SUB_EVENT_CUST_UPDATE_PROFILING_MODE,  // update profiling mode, Implemented by cust-sd
    AICPU_SUB_EVENT_ABNORMAL_LOG,  // print aicpu cust schedule's error log
};

struct __attribute__((visibility("default"))) AICPUSubEventStreamInfo {
    uint32_t streamId;
};

struct __attribute__((visibility("default"))) AICPUProfilingModeInfo {
    uint32_t deviceId;
    pid_t hostpId;
    uint32_t flag;
};

struct __attribute__((visibility("default"))) AICPULoadSoInfo {
    uint32_t kernelSoIndex;
};

struct __attribute__((visibility("default"))) AICPUEndGraphInfo {
    uint32_t result;
};

struct __attribute__((visibility("default"))) AICPUSharderTaskInfo {
    uint32_t parallelId;

    bool operator==(const AICPUSharderTaskInfo &sharderInfo) const noexcept
    {
        return (parallelId == sharderInfo.parallelId);
    }
};

struct __attribute__((visibility("default"))) AICPUUnLockTableInfo {
    uint32_t tableId;
};

struct __attribute__((visibility("default"))) AICPUSubEventInfo {
    uint32_t modelId;
    union {
        AICPUSubEventStreamInfo streamInfo;
        AICPUProfilingModeInfo modeInfo;
        AICPULoadSoInfo loadSoInfo;
        AICPUEndGraphInfo endGraphInfo;
        AICPUSharderTaskInfo sharderTaskInfo;
        AICPUUnLockTableInfo unlockTableInfo;
    } para;
};

struct __attribute__((visibility("default"))) AICPUBindSdPidEventMsg {
    int32_t pid;
} __attribute__((packed));

struct __attribute__((visibility("default"))) AICPUOpenCustomSoEventMsg {
    char_t kernelSoName[MAX_CUST_SO_NAME_LEN];
} __attribute__((packed));

struct __attribute__((visibility("default"))) CpuSchedInitParam {
    uint32_t deviceId;
    pid_t hostPid;
    ProfilingMode profilingMode;
    char_t rsv[128];
} __attribute__((packed));

struct __attribute__((visibility("default"))) ModelQueueInfo {
    uint32_t queueId;
    uint32_t flag;
} __attribute__((packed));

struct __attribute__((visibility("default"))) ModelTaskInfo {
    uint32_t taskId;
    uint64_t kernelName;
    uint64_t paraBase;     // param地址
} __attribute__((packed));

struct __attribute__((visibility("default"))) ModelStreamInfo {
    uint32_t streamId;
    uint32_t streamFlag;
    uint16_t taskNum;
    ModelTaskInfo *tasks;
} __attribute__((packed));

struct __attribute__((visibility("default"))) AicpuPriInfo {
    uint16_t checkHead;
    int32_t pidPriority;
    int32_t eventPriority;
}__attribute__((packed));

struct __attribute__((visibility("default"))) ModelCommOpList {
    uint64_t commOpDescsListAddr;
    uint32_t opNum;
}__attribute__((packed));

// 配置类型，用于给AICPU控制不同的加载流程
enum __attribute__((visibility("default"))) ModelType {
    kModelWithEmbedding = 0,
    kModelWithSyncEvent = 1,
    kModelTypeNum
};

struct __attribute__((visibility("default"))) CommGroup {
    const char *groupName;
    uint32_t rankNum;
    uint32_t *rankIds;
}__attribute__((packed));

struct __attribute__((visibility("default"))) CommGroups {
    uint32_t groupNum;
    CommGroup *groups;
}__attribute__((packed));

struct __attribute__((visibility("default"))) ModelCfgInfo {
    uint16_t inBuffPoolSize;       // input buffer pool size
    uint16_t outBuffPoolSize;      // output buffer pool size
    uint64_t inBuffSize;           // input buffer size
    uint64_t outBuffSize;          // output buffer size
    int32_t tagId;                 // tag id for hccl
    int32_t rankId;                // rank id for hccl
    uint64_t rankTableLen;         // rank table length
    uint64_t rankTableAddr;        // rank table ptr
    uint64_t roleTableLen;         // cluster spec length
    uint64_t roleTableAddr;        // role table ptr
    uint64_t modelCommOpListAddr;  // ModelCommOpList ptr
    uint32_t modelType;
    uint64_t commGroupsAddr;       // communication groups ptr
    int32_t psId;                  // ps id
    bool supportCounterFilter;     // counter filter flag
    char rsv[103UL];
} __attribute__((packed));

struct __attribute__((visibility("default"))) ModelInfo {
    uint32_t modelId;
    uint16_t aicpuStreamNum;
    ModelStreamInfo *streams;
    uint16_t queueNum;
    ModelQueueInfo *queues;
    int32_t abnormalBreak;
    int32_t abnormalEnqueue;
    AicpuPriInfo aicpuPriInfo;
    uint64_t cfgInfoPtr = 0;
    int32_t abnormalEnable = 1;
    char rsv[98];
} __attribute__((packed));
}
#endif  // AICPUSD_AICPUSD_INFO_H