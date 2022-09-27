/**
 * @file prof_callback.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef MSPROFILER_PROF_CALLBACK_H_
#define MSPROFILER_PROF_CALLBACK_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

/**
 * @name  MsprofErrorCode
 * @brief error code
 */
enum MsprofErrorCode {
    MSPROF_ERROR_NONE = 0,
    MSPROF_ERROR_MEM_NOT_ENOUGH,
    MSPROF_ERROR_GET_ENV,
    MSPROF_ERROR_CONFIG_INVALID,
    MSPROF_ERROR_ACL_JSON_OFF,
    MSPROF_ERROR,
};

#define MSPROF_ENGINE_MAX_TAG_LEN (63)

/**
 * @name  ReporterData
 * @brief struct of data to report
 */
struct ReporterData {
    char tag[MSPROF_ENGINE_MAX_TAG_LEN + 1];  // the sub-type of the module, data with different tag will be writen
    int deviceId;                             // the index of device
    size_t dataLen;                           // the length of send data
    unsigned char *data;                      // the data content
};

/**
 * @name  MsprofHashData
 * @brief struct of data to hash
 */
struct MsprofHashData {
    int deviceId;                             // the index of device
    size_t dataLen;                           // the length of data
    unsigned char *data;                      // the data content
    uint64_t hashId;                          // the id of hashed data
};

/**
 * @name  MsprofReporterModuleId
 * @brief module id of data to report
 */
enum MsprofReporterModuleId {
    MSPROF_MODULE_DATA_PREPROCESS = 0,    // DATA_PREPROCESS
    MSPROF_MODULE_HCCL,                   // HCCL
    MSPROF_MODULE_ACL,                    // AclModule
    MSPROF_MODULE_FRAMEWORK,              // Framework
    MSPROF_MODULE_RUNTIME,                // runtime
    MSPROF_MODULE_MSPROF                  // msprofTx
};

/**
 * @name  MsprofReporterCallbackType
 * @brief reporter callback request type
 */
enum MsprofReporterCallbackType {
    MSPROF_REPORTER_REPORT = 0,           // report data
    MSPROF_REPORTER_INIT,                 // init reporter
    MSPROF_REPORTER_UNINIT,               // uninit reporter
    MSPROF_REPORTER_DATA_MAX_LEN,         // data max length for calling report callback
    MSPROF_REPORTER_HASH                  // hash data to id
};

#define MSPROF_OPTIONS_DEF_LEN_MAX (2048)

/**
 * @name  MsprofGeOptions
 * @brief struct of MSPROF_CTRL_INIT_GE_OPTIONS
 */
struct MsprofGeOptions {
    char jobId[MSPROF_OPTIONS_DEF_LEN_MAX];
    char options[MSPROF_OPTIONS_DEF_LEN_MAX];
};

/**
 * @name  MsprofCtrlCallbackType
 * @brief ctrl callback request type
 */
enum MsprofCtrlCallbackType {
    MSPROF_CTRL_INIT_ACL_ENV = 0,           // start profiling with acl env
    MSPROF_CTRL_INIT_ACL_JSON,              // start pro with acl.json
    MSPROF_CTRL_INIT_GE_OPTIONS,            // start profiling with ge env and options
    MSPROF_CTRL_FINALIZE,                   // stop profiling
    MSPROF_CTRL_INIT_HELPER,                // start profiling in helper device
    MSPROF_CTRL_INIT_DYNA = 0xFF,           // start profiling for dynamic profiling
};

enum MsprofCommandHandleType {
    PROF_COMMANDHANDLE_TYPE_INIT = 0,
    PROF_COMMANDHANDLE_TYPE_START,
    PROF_COMMANDHANDLE_TYPE_STOP,
    PROF_COMMANDHANDLE_TYPE_FINALIZE,
    PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE,
    PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE
};

/**
 * @brief   profiling command type
 */
enum ProfCtrlType {
    PROF_CTRL_INVALID = 0,
    PROF_CTRL_SWITCH,
    PROF_CTRL_REPORTER,
    PROF_CTRL_STEPINFO,
    PROF_CTRL_BUTT
};

/**
 * @brief   Prof Chip ID
 */
enum Prof_Chip_ID {
    PROF_CHIP_ID0 = 0
};

typedef int32_t (*MsprofCtrlCallback)(uint32_t type, void *data, uint32_t len);
typedef int32_t (*MsprofReporterCallback)(uint32_t moduleId, uint32_t type, void *data, uint32_t len);

/**
 * @brief  the struct of profiling set setp info
 */
typedef struct ProfStepInfoCmd {
    uint64_t index_id;
    uint16_t tag_id;
    void *stream;
} ProfStepInfoCmd_t;

/**
 * @name  ProfCommandHandle
 * @brief callback to start/stop profiling
 * @param type      [IN] enum call back type
 * @param data      [IN] callback data
 * @param len       [IN] callback data size
 * @return enum MsprofErrorCode
 */
typedef int32_t (*ProfCommandHandle)(uint32_t type, void *data, uint32_t len);
/*
 * @name  profInit
 * @brief Profiling module init
 * @param [in] dataType: profiling type: ACL Env/ACL Json/GE Option
 * @param [in] data: profiling switch data
 * @param [in] dataLen: Length of data
 * @return 0:SUCCESS, >0:FAILED
 */
MSVP_PROF_API int32_t MsprofInit(uint32_t moduleId, void *data, uint32_t dataLen);
/**
 * @name  profRegisterCallback
 * @brief register callback to profiling
 * @param moduleId  [IN] module Id
 * @param handle    [IN] the pointer of callback
 */
MSVP_PROF_API int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);
/*
 * @name profReportData
 * @brief start reporter/stop reporter/report date
 * @param moduleId  [IN] enum profReporterModuleId
 * @param type      [IN] enum profReporterCallbackType
 * @param data      [IN] data (nullptr on INTI/UNINIT)
 * @param len       [IN] data size (0 on INIT/UNINIT)
 * @return enum MsprofErrorCod
 */
MSVP_PROF_API int32_t MsprofReportData(uint32_t moduleId, uint32_t type, void* data, uint32_t len);

MSVP_PROF_API int32_t MsprofSetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);
MSVP_PROF_API int32_t MsprofUnsetDeviceIdByGeModelIdx(const uint32_t geModelIdx, const uint32_t deviceId);
/*
 * @name profFinalize
 * @brief Finishing Profiling
 * @param NULL
 * @return 0:SUCCESS, >0:FAILED
 */
MSVP_PROF_API int32_t MsprofFinalize();
/**
 * @name  profNotifySetDevice
 * @brief notify set/reset device
 * @param devId     [IN] device id
 * @param isOpenDevice  [IN] true: set device, false: reset device
 */
MSVP_PROF_API int32_t MsprofNotifySetDevice(uint32_t chipId, uint32_t deviceId, bool isOpen);

#ifdef __cplusplus
}
#endif

#endif  // MSPROFILER_PROF_CALLBACK_H_
