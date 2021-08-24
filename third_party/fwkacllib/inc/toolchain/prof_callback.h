/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef MSPROFILER_PROF_CALLBACK_H_
#define MSPROFILER_PROF_CALLBACK_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


#include "stddef.h"
#include "stdint.h"

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

#define MSPROF_ENGINE_MAX_TAG_LEN (31)

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
    MSPROF_MODULE_RUNTIME                 // runtime
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

/**
 * @name  MsprofReporterCallback
 * @brief callback to start reporter/stop reporter/report date
 * @param moduleId  [IN] enum MsprofReporterModuleId
 * @param type      [IN] enum MsprofReporterCallbackType
 * @param data      [IN] callback data (nullptr on INTI/UNINIT)
 * @param len       [IN] callback data size (0 on INIT/UNINIT)
 * @return enum MsprofErrorCode
 */
typedef int32_t (*MsprofReporterCallback)(uint32_t moduleId, uint32_t type, void *data, uint32_t len);


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
    MSPROF_CTRL_INIT_ACL_JSON,              // start profiling with acl.json
    MSPROF_CTRL_INIT_GE_OPTIONS,            // start profiling with ge env and options
    MSPROF_CTRL_FINALIZE,                   // stop profiling
    MSPROF_CTRL_REPORT_FUN_P,               // for report callback
    MSPROF_CTRL_PROF_SWITCH_ON,             // for prof switch on
    MSPROF_CTRL_PROF_SWITCH_OFF             // for prof switch off
};

#define    PROF_COMMANDHANDLE_TYPE_INIT              (0)
#define    PROF_COMMANDHANDLE_TYPE_START             (1)
#define    PROF_COMMANDHANDLE_TYPE_STOP              (2)
#define    PROF_COMMANDHANDLE_TYPE_FINALIZE          (3)
#define    PROF_COMMANDHANDLE_TYPE_MODEL_SUBSCRIBE   (4)
#define    PROF_COMMANDHANDLE_TYPE_MODEL_UNSUBSCRIBE (5)

#define MSPROF_MAX_DEV_NUM (64)

struct MsprofCommandHandle {
    uint64_t profSwitch;
    uint32_t devNums; // length of device id list
    uint32_t devIdList[MSPROF_MAX_DEV_NUM];
    uint32_t modelId;
};

/**
 * @name  MsprofCtrlCallback
 * @brief callback to start/stop profiling
 * @param type      [IN] enum MsprofCtrlCallbackType
 * @param data      [IN] callback data
 * @param len       [IN] callback data size
 * @return enum MsprofErrorCode
 */
typedef int32_t (*MsprofCtrlCallback)(uint32_t type, void *data, uint32_t len);

/**
 * @name  MsprofSetDeviceCallback
 * @brief callback to notify set/reset device
 * @param devId     [IN] device id
 * @param isOpenDevice  [IN] true: set device, false: reset device
 */
typedef void (*MsprofSetDeviceCallback)(uint32_t devId, bool isOpenDevice);

/*
 * @name  MsprofInit
 * @brief Profiling module init
 * @param [in] dataType: profiling type: ACL Env/ACL Json/GE Option
 * @param [in] data: profiling switch data
 * @param [in] dataLen: Length of data
 * @return 0:SUCCESS, >0:FAILED
 */
int32_t MsprofInit(uint32_t dataType, void *data, uint32_t dataLen);

/*
 * @name AscendCL
 * @brief Finishing Profiling
 * @param NULL
 * @return 0:SUCCESS, >0:FAILED
 */
int32_t MsprofFinalize();
#ifdef __cplusplus
}
#endif

#endif  // MSPROFILER_PROF_CALLBACK_H_
