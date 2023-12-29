/**
 * @file prof_api.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef PROF_API_H
#define PROF_API_H

#include <stdint.h>
#include <stdbool.h>
#include "prof_callback.h"
#include "prof_common.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

const uint32_t MSPROF_REPORT_DATA_MAGIC_NUM = 0x5a5a;
const uint64_t MSPROF_EVENT_FLAG = 0xFFFFFFFFFFFFFFFFULL;
using VOID_PTR = void *;
using ProfCommandHandle = int32_t (*)(uint32_t type, VOID_PTR data, uint32_t len);
using MsprofReportHandle = int32_t (*)(uint32_t moduleId, uint32_t type, VOID_PTR data, uint32_t len);
using MsprofCtrlHandle = int32_t (*)(uint32_t type, VOID_PTR data, uint32_t len);
using MsprofSetDeviceHandle = int32_t (*)(VOID_PTR data, uint32_t len);

/* Msprof report level */
const uint16_t MSPROF_REPORT_PYTORCH_LEVEL = 30000;
const uint16_t MSPROF_REPORT_PTA_LEVEL = 25000;
const uint16_t MSPROF_REPORT_ACL_LEVEL = 20000;
const uint16_t MSPROF_REPORT_MODEL_LEVEL = 15000;
const uint16_t MSPROF_REPORT_NODE_LEVEL = 10000;
const uint16_t MSPROF_REPORT_HCCL_NODE_LEVEL = 5500;
const uint16_t MSPROF_REPORT_RUNTIME_LEVEL = 5000;

/* Msprof report type of acl(20000) level(acl), offset: 0x000000 */
const uint32_t MSPROF_REPORT_ACL_OP_BASE_TYPE            = 0x010000U;
const uint32_t MSPROF_REPORT_ACL_MODEL_BASE_TYPE         = 0x020000U;
const uint32_t MSPROF_REPORT_ACL_RUNTIME_BASE_TYPE       = 0x030000U;
const uint32_t MSPROF_REPORT_ACL_OTHERS_BASE_TYPE        = 0x040000U;


/* Msprof report type of acl(20000) level(host api), offset: 0x050000 */
const uint32_t MSPROF_REPORT_ACL_NN_BASE_TYPE            = 0x050000U;
const uint32_t MSPROF_REPORT_ACL_ASCENDC_TYPE            = 0x060000U;
const uint32_t MSPROF_REPORT_ACL_HOST_HCCL_BASE_TYPE     = 0x070000U;
const uint32_t MSPROF_REPORT_ACL_DVPP_BASE_TYPE          = 0x090000U;
const uint32_t MSPROF_REPORT_ACL_GRAPH_BASE_TYPE         = 0x0A0000U;

/* Msprof report type of model(15000) level, offset: 0x000000 */
const uint32_t MSPROF_REPORT_MODEL_GRAPH_ID_MAP_TYPE    = 0;         /* type info: graph_id_map */
const uint32_t MSPROF_REPORT_MODEL_EXECUTE_TYPE         = 1;         /* type info: execute */
const uint32_t MSPROF_REPORT_MODEL_LOAD_TYPE            = 2;         /* type info: load */
const uint32_t MSPROF_REPORT_MODEL_EXEOM_TYPE           = 3;         /* type info: exeom */
const uint32_t MSPROF_REPORT_MODEL_LOGIC_STREAM_TYPE    = 7;         /* type info: logic_stream_info */
const uint32_t MSPROF_REPORT_MODEL_UDF_BASE_TYPE        = 0x010000U;  /* type info: udf_info */
const uint32_t MSPROF_REPORT_MODEL_AICPU_BASE_TYPE      = 0x020000U;  /* type info: aicpu */

/* Msprof report type of node(10000) level, offset: 0x000000 */
const uint32_t MSPROF_REPORT_NODE_BASIC_INFO_TYPE       = 0;  /* type info: node_basic_info */
const uint32_t MSPROF_REPORT_NODE_TENSOR_INFO_TYPE      = 1;  /* type info: tensor_info */
const uint32_t MSPROF_REPORT_NODE_FUSION_OP_INFO_TYPE   = 2;  /* type info: funsion_op_info */
const uint32_t MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE  = 4;  /* type info: context_id_info */
const uint32_t MSPROF_REPORT_NODE_LAUNCH_TYPE           = 5;  /* type info: launch */
const uint32_t MSPROF_REPORT_NODE_TASK_MEMORY_TYPE      = 6;  /* type info: task_memory_info */
const uint32_t MSPROF_REPORT_NODE_HOST_OP_EXEC_TYPE     = 8;  /* type info: op exec */

/* Msprof report type of node(10000) level(ge api), offset: 0x010000 */
const uint32_t MSPROF_REPORT_NODE_GE_API_BASE_TYPE      = 0x010000U; /* type info: ge api */
const uint32_t MSPROF_REPORT_NODE_HCCL_BASE_TYPE        = 0x020000U; /* type info: hccl api */
const uint32_t MSPROF_REPORT_NODE_DVPP_API_BASE_TYPE    = 0x030000U; /* type info: dvpp api */

/* Msprof report type of hccl(5500) level(op api), offset: 0x010000 */
const uint32_t MSPROF_REPORT_HCCL_NODE_BASE_TYPE        = 0x010000U;
const uint32_t MSPROF_REPORT_HCCL_MASTER_TYPE           = 0x010001U;
const uint32_t MSPROF_REPORT_HCCL_SLAVE_TYPE            = 0x010002U;

enum ProfileCallbackType {
    PROFILE_CTRL_CALLBACK = 0,
    PROFILE_DEVICE_STATE_CALLBACK,
    PROFILE_REPORT_API_CALLBACK,
    PROFILE_REPORT_EVENT_CALLBACK,
    PROFILE_REPORT_COMPACT_CALLBACK,
    PROFILE_REPORT_ADDITIONAL_CALLBACK,
    PROFILE_REPORT_REG_TYPE_INFO_CALLBACK,
    PROFILE_REPORT_GET_HASH_ID_CALLBACK,
    PROFILE_HOST_FREQ_IS_ENABLE_CALLBACK
};

struct MsprofApi { // for MsprofReportApi
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t reserve;
    uint64_t beginTime;
    uint64_t endTime;
    uint64_t itemId;
};

struct MsprofEvent {  // for MsprofReportEvent
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t requestId; // 0xFFFF means single event
    uint64_t timeStamp;
    uint64_t eventFlag = MSPROF_EVENT_FLAG;
    uint64_t itemId;
};

struct MsprofRuntimeTrack {  // for MsprofReportCompactInfo buffer data
    uint16_t deviceId;
    uint16_t streamId;
    uint32_t taskId;
    uint64_t taskType;       // task message hash id
};

const uint16_t MSPROF_COMPACT_INFO_DATA_LENGTH = 40;
struct MsprofCompactInfo {  // for MsprofReportCompactInfo buffer data
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    union {
        uint8_t info[MSPROF_COMPACT_INFO_DATA_LENGTH];
        MsprofRuntimeTrack runtimeTrack;
        MsprofNodeBasicInfo nodeBasicInfo;
    } data;
};

const uint16_t MSPROF_ADDTIONAL_INFO_DATA_LENGTH = 232;
struct MsprofAdditionalInfo {  // for MsprofReportAdditionalInfo buffer data
    uint16_t magicNumber = MSPROF_REPORT_DATA_MAGIC_NUM;
    uint16_t level;
    uint32_t type;
    uint32_t threadId;
    uint32_t dataLen;
    uint64_t timeStamp;
    uint8_t  data[MSPROF_ADDTIONAL_INFO_DATA_LENGTH];
};

/*
 * @ingroup libprofapi
 * @name  profRegReporterCallback
 * @brief register report callback interface for atlas
 * @param [in] reporter: reporter callback handle
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegReporterCallback(MsprofReportHandle reporter);

/*
 * @ingroup libprofapi
 * @name  profRegCtrlCallback
 * @brief register control callback, interface for atlas
 * @param [in] handle: control callback handle
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegCtrlCallback(MsprofCtrlHandle handle);

/*
 * @ingroup libprofapi
 * @name  profRegDeviceStateCallback
 * @brief register device state notify callback, interface for atlas
 * @param [in] handle: handle of ProfNotifySetDevice
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profRegDeviceStateCallback(MsprofSetDeviceHandle handle);

/*
 * @ingroup libprofapi
 * @name  profGetDeviceIdByGeModelIdx
 * @brief get device id by model id, interface for atlas
 * @param [in] modelIdx: ge model id
 * @param [out] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profGetDeviceIdByGeModelIdx(const uint32_t modelIdx, uint32_t *deviceId);

/*
 * @ingroup libprofapi
 * @name  profSetProfCommand
 * @brief register set profiling command, interface for atlas
 * @param [in] command: 0 isn't aging, !0 is aging
 * @param [in] len: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profSetProfCommand(VOID_PTR command, uint32_t len);

/*
 * @ingroup libprofapi
 * @name  profSetStepInfo
 * @brief set step info for torch, interface for atlas
 * @param [in] indexId: id of iteration index
 * @param [in] tagId: id of tag
 * @param [in] stream: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t profSetStepInfo(const uint64_t indexId, const uint16_t tagId, void* const stream);

/*
 * @ingroup libprofapi
 * @name  MsprofRegisterProfileCallback
 * @brief register profile callback by callback type, interface for atlas
 * @param [in] callbackType: type of callback(reporter/ctrl/device state/command)
 * @param [in] callback: callback of profile
 * @param [in] len: callback length
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegisterProfileCallback(int32_t callbackType, VOID_PTR callback, uint32_t len);

/*
 * @ingroup libprofapi
 * @name  MsprofInit
 * @brief Profiling module init
 * @param [in] dataType: profiling type: ACL Env/ACL Json/GE Option
 * @param [in] data: profiling switch data
 * @param [in] dataLen: Length of data
 * @return 0:SUCCESS, >0:FAILED
 */
MSVP_PROF_API int32_t MsprofInit(uint32_t dataType, VOID_PTR data, uint32_t dataLen);

/*
 * @ingroup libprofapi
 * @name  MsprofRegisterCallback
 * @brief register profiling switch callback for module
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] api: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

/*
 * @ingroup libprofapi
 * @name  MsprofReportData
 * @brief report profiling data of module
 * @param [in] moduleId: module id
 * @param [in] type: report type(init/uninit/max length/hash)
 * @param [in] data: profiling data
 * @param [in] len: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportData(uint32_t moduleId, uint32_t type, VOID_PTR data, uint32_t len);

/*
 * @ingroup libprofapi
 * @name  MsprofReportApi
 * @brief report api timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] api: api of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi *api);

/*
 * @ingroup libprofapi
 * @name  MsprofReportEvent
 * @brief report event timestamp
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] event: event of timestamp data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportEvent(uint32_t agingFlag, const MsprofEvent *event);

/*
 * @ingroup libprofapi
 * @name  MsprofReportCompactInfo
 * @brief report profiling compact infomation
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of compact infomation
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

/*
 * @ingroup libprofapi
 * @name  MsprofReportAdditionalInfo
 * @brief report profiling additional infomation
 * @param [in] agingFlag: 0 isn't aging, !0 is aging
 * @param [in] data: profiling data of additional infomation
 * @param [in] length: length of profiling data
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

/*
 * @ingroup libprofapi
 * @name  MsprofRegTypeInfo
 * @brief reg mapping info of type id and type name
 * @param [in] level: level is the report struct's level
 * @param [in] typeId: type id is the report struct's type
 * @param [in] typeName: label of type id for presenting user
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char *typeName);

/*
 * @ingroup libprofapi
 * @name  MsprofGetHashId
 * @brief return hash id of hash info
 * @param [in] hashInfo: infomation to be hashed
 * @param [in] length: the length of infomation to be hashed
 * @return hash id
 */
MSVP_PROF_API uint64_t MsprofGetHashId(const char *hashInfo, size_t length);

/*
 * @ingroup libprofapi
 * @name  MsprofSetDeviceIdByGeModelIdx
 * @brief insert device id by model id
 * @param [in] geModelIdx: ge model id
 * @param [in] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofSetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);

/*
 * @ingroup libprofapi
 * @name  MsprofUnsetDeviceIdByGeModelIdx
 * @brief delete device id by model id
 * @param [in] geModelIdx: ge model id
 * @param [in] deviceId: device id
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofUnsetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);

/*
 * @ingroup libprofapi
 * @name  register report interface for atlas
 * @brief report api timestamp
 * @param [in] chipId: multi die's chip
 * @param [in] deviceId: device id
 * @param [in] isOpen: device is open
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofNotifySetDevice(uint32_t chipId, uint32_t deviceId, bool isOpen);

/*
 * @ingroup libprofapi
 * @name  MsprofFinalize
 * @brief profiling finalize
 * @return 0:SUCCESS, !0:FAILED
 */
MSVP_PROF_API int32_t MsprofFinalize();

/*
 * @ingroup libprofapi
 * @name  MsprofSysCycleTime
 * @brief get systime cycle time of CPU
 * @return system cycle time of CPU
 */
MSVP_PROF_API uint64_t MsprofSysCycleTime();
#ifdef __cplusplus
}
#endif

#endif
