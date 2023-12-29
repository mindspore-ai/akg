/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: HCOM data type definition
 * Author: ligang
 * Create: 2019-05-24
 */

#ifndef HCCL_BASE_H
#define HCCL_BASE_H
#include <hccl/hccl_types.h>
#include <string>
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

typedef signed char s8;
typedef signed short s16;
typedef signed int s32;
typedef signed long long s64;
typedef unsigned char u8;
typedef unsigned short u16;
typedef unsigned int u32;
typedef unsigned long long u64;

/**
 * @brief Horovod Reduction opperation
 */
typedef enum {
    HOROVOD_REDUCE_AVERAGE = 0, /**< average */
    HOROVOD_REDUCE_SUM = 1,     /**< sum */
    HOROVOD_REDUCE_ADASUM = 2,  /**< adasum */
    HOROVOD_REDUCE_MIN = 3,     /**< min */
    HOROVOD_REDUCE_MAX = 4,     /**< max */
    HOROVOD_REDUCE_PROD = 5,    /**< proo */
    HOROVOD_REDUCE_RESERVED     /**< reserved */
} HorovodReduceOp;

const u32 HCCL_MAX_SEGMENT_NUM = 8;   // The max number of gradient segments.

/**
 * @brief the feature of the model
 */
struct model_feature {
    const char *model_name;  /**< The model name */
    u32 gradient_num;        /**< The number of gradients */
    float *gradient_size;    /**< The size of each gradient */
    float *gradient_time;    /**< The BP compution time of each gradient */
};

/**
 * @brief Memory Register Address Struct for Remote Access
 */
struct MemRegisterAddr {
    u64 addr;
    u64 length;
};
/*
 * @brief The max number of memory register addr for remote access.
 */
const u32 HCCL_MAX_MEM_REGISTER_NUM = 32;

enum GradSplitForceMode {
    FORCE_NONE,     /**< no force */
    FORCE_SIZE,     /**< force split gradient by size */
    FORCE_RESERVED  /**< reserved */
};

enum OriginalGraphShapeType {
    KNOWN_SHAPE,
    UNKNOWN_SHAPE,
    SHAPE_RESERVED  /**< reserved */
};

enum HcclEventType {
    HCCL_EVENT_SEND_COMPLETION = 0,
    HCCL_EVENT_RECV_REQUEST,
    HCCL_EVENT_RECV_COMPLETION,
    HCCL_EVENT_CONGESTION_RELIEF,
    HCCL_EVENT_RESERVED /**< reserved */
};

const u32 TAG_MAX_LEN = 127; // 最大的tag 长度
using TagAttr = struct TagAttrDef {
    char name[TAG_MAX_LEN + 1]; // tag标识
    // tag标识的接收数据，调用者是否会主动调用接收接口，0 = 否, 1 = 会(预留，暂不支持)。
    // 对于activeRecv = 0，当接收侧收到数据或者发送请求时，主动通知调用者。
    uint32_t activeRecv;
    uint32_t sendCredit; // 配置该tag允许inflight的send个数
    uint32_t eventId;
};

using HcclEventMsg = struct HcclEventMsgDef {
    HcclComm comm;
    u32 peerRank;
    u32 tag;
    // 0:HCCL_SEND_COMPLETION; 1:HCCL_RECV_COMPLETION; 2:HCCL_RECV_REQUEST; 3:HCCL_CONGESTION_RELIEF
    u32 hcclEventType;
    union {
        struct {
            u32 reserver;
        } sendCompletionItem;
        struct {
            u32 reserver;
        } recvRequestItem;
        struct {
            u32 reserver;
        } recvCompletionItem;
        struct CongestionReliefItem {
            u32 reserver;
        } congestionReliefItem;
    } desc;
};


/**
* @brief stream handle.
*/
typedef void *rtStream_t;

/**
* @brief model handle.
*/
typedef void *rtModel_t;

struct HcomRemoteOperation {
    void *opdesc;
    void *keyAddr;
    int *tableId;
    void *value;
    void *indices{nullptr};
    void *numUniqued{nullptr};
    void *psSeg{nullptr};
    void *psSegNum{nullptr};
};

struct HcomOperation {
    std::string hcclType;
    void *inputPtr{nullptr};
    void *outputPtr{nullptr};
    u64 count{0};
    HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
    HcclReduceOp opType{HCCL_REDUCE_RESERVED};
    u32 root{0};
    const char *group{nullptr};
};

struct HcomRemoteAccessAddrInfo {
    u32 remotetRankID;
    u64 remoteAddr;  // host embedding table addr
    u64 localAddr;  // device HBM addr
    u64 length;   // Memory Length in Bytes
};

struct HcomRemoteOperationParams {
    void *keyAddr{nullptr};
    int *tableId{nullptr};
    void *value{nullptr};
    std::string group;
    int count;
    int valueDim;
    HcclDataType keyType{HCCL_DATA_TYPE_RESERVED};
    HcclDataType valueType{HCCL_DATA_TYPE_RESERVED};
    int tag;
    rtStream_t stream;
    u64 flag;
    s32 insertOption;
    void *indices{};
    void *numUniqued{};
    void *psSeg{};
    void *psSegNum{};
    std::string hcclType;
    s32 flags{};
};

struct HcomAllToAllVParams {
    void *sendbuf{nullptr};     // device mem
    void *sendcounts{nullptr};  // device mem;  Type: uint_64
    void *sdispls{nullptr};     // device mem;  Type: uint_64
    HcclDataType sendtype{HCCL_DATA_TYPE_RESERVED};
    void *recvbuf{nullptr};  // device mem
    void *recvcounts{nullptr};  // device mem;  Type: uint_64
    void *rdispls{nullptr};  // device mem;  Type: uint_64
    HcclDataType recvtype{HCCL_DATA_TYPE_RESERVED};
    const char *group{nullptr};
};

struct HcomAllToAllVCParams {
    void *sendbuf{nullptr};     // device mem
    HcclDataType sendtype{HCCL_DATA_TYPE_RESERVED};
    void *recvbuf{nullptr};  // device mem
    HcclDataType recvtype{HCCL_DATA_TYPE_RESERVED};
    void *sendcountmatrix{nullptr};  // device mem;  Type: uint_64
    const char *group{nullptr};
};

struct HcomGatherAllToAllVParams {
    void *addrInfo;  // device mem;  contains host VA[uint_64]:  [addr, length, addr, length, addr, length, ...]
    void *addrInfoCountPerRank;  // device mem;  length: ranksize;  contains addrInfoCounts for every rank
    void *recvbuf;  // device mem
    void *recvcounts;  // device mem;  Type: uint_64
    void *rdispls;  // device mem;  Type: uint_64
    void *gatheredbuf;  // device mem
    s32 addrLength;
    HcclDataType recvtype;
    const char *group;
};

typedef enum workMode {
HCCL_MODE_NORMAL = 0, // 不支持任何Probe any，仅支持精确的probe
HCCL_MODE_ANY = 1,     // 仅支持ANY_SOURCE + ANY_TAG的probe
HCCL_MODE_PS = 2,     // 支持ps侧hccd功能
HCCL_MODE_AI_CPU = 3,     // 支持aicpu侧hccd功能
HCCL_MODE_SCHED_OS = 4
} WorkMode;

typedef struct tagCommAttr {
    WorkMode mode;  // 通信域内的probe工作模式
    uint32_t deviceId = 0;
} CommAttr;

typedef void* HcclMessage;
typedef void* HcclRequest;
typedef void* ServiceHandle;

typedef struct {
    int srcRank;    // 接收/探测到的msg/信封的发送端rank_id，MPI标准定义，调用者可以访问
    int tag;        // 接收/探测到的msg/信封的tag，MPI标准定义，调用者可以访问
    int error;      // 接收/探测的错误码0：no error，others：传输过程出错，MPI标准定义，调用者可以访问
    int cancelled;  // 指定实现，不建议调用者访问
    int count;      // 接收/探测到的payload大小，指定实现，不建议调用者访问
} HcclStatus;

using ReqStatus = struct tagReqStatus {
    int tableId;        // 查找的table_id
    int tag;            // 与算子IR中的tag相同
    int actualSize;     // 如果是接收接口的status, 还返回实际接收到的size
    int rsvd0;
};

using LookupReqStatus = struct tagLookupReqStatus {
    int tableId;        // 查找的table_id
    int tag;            // 与算子IR中的tag相同
    int actualCount;    // 如果是接收接口的status, 还返回实际接收到的keyCount
    int rsvd0;
};

using UpdateReqStatus = struct tagUpdateReqStatus {
    int tableId;            // 查找的table_id
    int tag;                // 与算子IR中的tag相同
    int actualKeyCount;     // 如果是接收接口的status, 还返回实际接收到的keyCount
    int actualValueCount;   // 如果是接收接口的status, 还返回实际接收到的keyCount
};

using CalcParams = struct tagCalcParams {
    int uniqueCost0;    // worker/client侧的去重开销, 单位:ms
    int gatherCost0;    // worker/client侧的恢复开销, 单位:ms
    int uniqueCost1;    // ps/server侧的去重开销, 单位:ms
    int gatherCost1;    // ps/server侧的恢复开销, 单位:ms
};

enum HcomSchedType {
    HCOM_SCHED_TYPE_OS,              // CPU执行，OS调度
    HCOM_SCHED_TYPE_NPU_TS_OFFLOAD,  // TS下沉调度，对应model执行
    HCOM_SCHED_TYPE_NPU_TS,          // TS非下沉调度，对应单算子执行
    HCOM_SCHED_TYPE_NUM
};

enum HcomOperationType {
    HCOM_OP_TYPE_SEND,
    HCOM_OP_TYPE_RECV,
    HCOM_OP_TYPE_BROADCAST,
    HCOM_OP_TYPE_GATHER,
    // 后续扩展集合通信
    HCOM_OP_TYPE_NUM
};

#define HCOM_GROUP_NAME_MAX_LEN 127

typedef struct {
    char group[HCOM_GROUP_NAME_MAX_LEN];
    uint32_t tag;           // 点到点通信的tag
    uint32_t peerRank;      // 点到点通信的对端rank
    void* addr;             // send/recv的发送或接收buffer
    uint64_t count;         // 数据数量
    HcclDataType dataType;  // 对应HcclDataType
    int32_t rsv0;
    int32_t rsv1;
    int32_t rsv2;
} HcomP2pOpInfo;

typedef struct {
    char group[HCOM_GROUP_NAME_MAX_LEN];
    void* inputAddr;
    void* outputAddr;
    uint64_t count;
    HcclDataType dataType;
    uint32_t root;
    HcclReduceOp reduceOp;
    int32_t rsv2;
    int32_t rsv3;
} HcomCollOpInfo;

typedef struct {
    HcomOperationType opType;       // op类型用于结合拓扑和rank_table一起决定需要创建哪些通信连接
    HcomSchedType schedType;        // 由于prepare接口需要NPU与CPU共用，创建的QP是不同的，引出需要调度器类型
    int32_t cxtId;                  // 集合通信的执行上下文标识，由调用者自定义，（暂时保留不使用）
                                    // 相同cxtId被认为是相同的执行上下文，比如stream/thread
                                    // 相同上下文的集合通信只能串行执行
    uint64_t flag;                  // bit0：接收数据量和地址是否动态, 1==动态, 此时info里的addr, count无效（预留不使用）
    union {
        HcomP2pOpInfo p2p;
        HcomCollOpInfo coll;
    } info;
} HcomOpDesc;

typedef struct {
    uint32_t status;                // 0: ok; 1: on-going; 2: error
    uint32_t rsv0;
    uint32_t rsv1;
    uint32_t rsv2;
} HcomStatus;

typedef void* HcomRequest;

#define HCCL_REQUEST_NULL   nullptr

#define HCCL_TAG_ANY (1 << 30)

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_BASE_H
