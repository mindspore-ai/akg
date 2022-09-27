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

#ifndef D_SYSLOG_H_
#define D_SYSLOG_H_

static const int TMP_LOG = 0;

#ifdef __cplusplus
#ifndef LOG_CPP
extern "C" {
#endif
#endif // __cplusplus

#ifndef LINUX
#define LINUX 0
#endif // LINUX

#ifndef WIN
#define WIN 1
#endif

#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE

#if (OS_TYPE == LINUX)
#define DLL_EXPORT __attribute__((visibility("default")))
#else
#define DLL_EXPORT _declspec(dllexport)
#endif

/**
 * @ingroup slog
 *
 * debug level id
 */
#define DLOG_DEBUG 0

/**
 * @ingroup slog
 *
 * info level id
 */
#define DLOG_INFO 1

/**
 * @ingroup slog
 *
 * warning level id
 */
#define DLOG_WARN 2

/**
 * @ingroup slog
 *
 * error level id
 */
#define DLOG_ERROR 3

/**
 * @ingroup slog
 *
 * don't print log
 */
#define DLOG_NULL 4

/**
 * @ingroup slog
 *
 * trace log print level id
 */
#define DLOG_TRACE 5

/**
 * @ingroup slog
 *
 * oplog log print level id
 */
#define DLOG_OPLOG 6

/**
 * @ingroup slog
 *
 * event log print level id
 */
#define DLOG_EVENT 0x10

/**
 * @ingroup slog
 *
 * max log length
 */
#define MSG_LENGTH 1024
#define DEBUG_LOG_MASK      (0x00010000)
#define SECURITY_LOG_MASK   (0x00100000)
#define RUN_LOG_MASK        (0x01000000)
#define OPERATION_LOG_MASK  (0x10000000)
#define RESERVERD_LENGTH 52

typedef struct {
    const char *cName;
    int cVal;
} DCODE;

typedef struct tagKV {
    char *kname;
    char *value;
} KeyValue;

typedef enum {
    APPLICATION = 0,
    SYSTEM
} ProcessType;

typedef struct {
    ProcessType type;
    unsigned int pid;
    unsigned int deviceId;
    char reserved[RESERVERD_LENGTH];
} LogAttr;

/**
 * @ingroup slog
 *
 * module id
 */
enum {
    SLOG,          /**< Slog */
    IDEDD,         /**< IDE daemon device */
    IDEDH,         /**< IDE daemon host */
    HCCL,          /**< HCCL */
    FMK,           /**< Adapter */
    HIAIENGINE,    /**< Matrix */
    DVPP,          /**< DVPP */
    RUNTIME,       /**< Runtime */
    CCE,           /**< CCE */
#if (OS_TYPE == LINUX)
    HDC,         /**< HDC */
#else
    HDCL,
#endif // OS_TYPE
    DRV,           /**< Driver */
    MDCFUSION,     /**< Mdc fusion */
    MDCLOCATION,   /**< Mdc location */
    MDCPERCEPTION, /**< Mdc perception */
    MDCFSM,
    MDCCOMMON,
    MDCMONITOR,
    MDCBSWP,    /**< MDC base software platform */
    MDCDEFAULT, /**< MDC undefine */
    MDCSC,      /**< MDC spatial cognition */
    MDCPNC,
    MLL,      /**< abandon */
    DEVMM,    /**< Dlog memory managent */
    KERNEL,   /**< Kernel */
    LIBMEDIA, /**< Libmedia */
    CCECPU,   /**< aicpu shedule */
    ASCENDDK, /**< AscendDK */
    ROS,      /**< ROS */
    HCCP,
    ROCE,
    TEFUSION,
    PROFILING, /**< Profiling */
    DP,        /**< Data Preprocess */
    APP,       /**< User Application */
    TS,        /**< TS module */
    TSDUMP,    /**< TSDUMP module */
    AICPU,     /**< AICPU module */
    LP,        /**< LP module */
    TDT,       /**< tsdaemon or aicpu shedule */
    FE,
    MD,
    MB,
    ME,
    IMU,
    IMP,
    GE, /**< Fmk */
    MDCFUSA,
    CAMERA,
    ASCENDCL,
    TEEOS,
    ISP,
    SIS,
    HSM,
    DSS,
    PROCMGR,     // Process Manager, Base Platform
    BBOX,
    AIVECTOR,
    TBE,
    FV,
    MDCMAP,
    TUNE,
    HSS, /**< helper */
    FFTS,
    OP,
    UDF,
    HICAID,
    INVLID_MOUDLE_ID
};

/**
 * @ingroup slog
 * @brief External log interface, which called by modules
 */
DLL_EXPORT void dlog_init(void);

/**
 * @ingroup slog
 * @brief dlog_getlevel: get module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), others: invalid
 * @param [out]enableEvent: 1: enable; 0: disable
 * @return: module level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 */
DLL_EXPORT int dlog_getlevel(int moduleId, int *enableEvent);

/**
 * @ingroup slog
 * @brief dlog_setlevel: set module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), -1: all modules, others: invalid
 * @param [in]level: log level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 * @param [in]enableEvent: 1: enable; 0: disable, others:invalid
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int dlog_setlevel(int moduleId, int level, int enableEvent);

/**
 * @ingroup slog
 * @brief CheckLogLevel: check module level enable or not
 * users no need to call it because all dlog interface(include inner interface) has already called
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]logLevel: eg: DLOG_EVENT/DLOG_ERROR/DLOG_WARN/DLOG_INFO/DLOG_DEBUG
 * @return: 1:enable, 0:disable
 */
DLL_EXPORT int CheckLogLevel(int moduleId, int logLevel);

/**
 * @ingroup slog
 * @brief DlogSetAttr: set log attr, default pid is 0, default device id is 0, default process type is APPLICATION
 * @param [in]logAttr: attr info, include pid(must be larger than 0), process type and device id(chip ID)
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogSetAttr(LogAttr logAttr);

/**
 * @ingroup slog
 * @brief dlog_error: print error log
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_error(moduleId, fmt, ...)                                          \
    do {                                                                          \
        DlogErrorInner(moduleId, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_warn: print warning log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_warn(moduleId, fmt, ...)                                               \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_WARN) == 1) {                                   \
            DlogWarnInner(moduleId, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_info: print info log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_info(moduleId, fmt, ...)                                               \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_INFO) == 1) {                                   \
            DlogInfoInner(moduleId, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_debug: print debug log
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_debug(moduleId, fmt, ...)                                              \
    do {                                                                              \
        if (CheckLogLevel(moduleId, DLOG_DEBUG) == 1) {                                  \
            DlogDebugInner(moduleId, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                               \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief dlog_event: print event log
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]fmt: log content
 */
#define dlog_event(moduleId, fmt, ...)                                          \
    do {                                                                          \
        DlogEventInner(moduleId, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief Dlog: print log, need caller to specify level
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]fmt: log content
 */
#define Dlog(moduleId, level, fmt, ...)                                                 \
    do {                                                                                  \
        if (CheckLogLevel(moduleId, level) == 1) {                                           \
            DlogInner(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);   \
        }                                                                                  \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogSub: print log, need caller to specify level and submodule
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]submodule: eg: engine
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]fmt: log content
 */
#define DlogSub(moduleId, submodule, level, fmt, ...)                                                   \
    do {                                                                                                  \
        if (CheckLogLevel(moduleId, level) == 1) {                                                           \
            DlogInner(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);    \
        }                                                                                                   \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogWithKV: print log, need caller to specify level and other paramters
 * call CheckLogLevel in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]pstKVArray: key-value array
 * @param [in]kvNum: key-value element num in array
 * @param [in]fmt: log content
 */
#define DlogWithKV(moduleId, level, pstKVArray, kvNum, fmt, ...)                                                \
    do {                                                                                                          \
        if (CheckLogLevel(moduleId, level) == 1) {                                                                   \
            DlogWithKVInner(moduleId, level, pstKVArray, kvNum, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);  \
        }                                                                                                           \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogFlush: flush log buffer to file
 */
DLL_EXPORT void DlogFlush(void);

/**
 * @ingroup slog
 * @brief Internal log interface, other modules are not allowed to call this interface
 */
void DlogErrorInner(int moduleId, const char *fmt, ...);
void DlogWarnInner(int moduleId, const char *fmt, ...);
void DlogInfoInner(int moduleId, const char *fmt, ...);
void DlogDebugInner(int moduleId, const char *fmt, ...);
void DlogEventInner(int moduleId, const char *fmt, ...);
void DlogInner(int moduleId, int level, const char *fmt, ...);
void DlogWithKVInner(int moduleId, int level, KeyValue *pstKVArray, int kvNum, const char *fmt, ...);

#ifdef __cplusplus
#ifndef LOG_CPP
}
#endif // LOG_CPP
#endif // __cplusplus

#ifdef LOG_CPP
#ifdef __cplusplus
extern "C" {
#endif
/**
 * @ingroup slog
 * @brief DlogGetlevelForC: get module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), others: invalid
 * @param [out]enableEvent: 1: enable; 0: disable
 * @return: module level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 */
DLL_EXPORT int DlogGetlevelForC(int moduleId, int *enableEvent);

/**
 * @ingroup slog
 * @brief DlogSetlevelForC: set module loglevel and enableEvent
 *
 * @param [in]moduleId: moudule id(see slog.h, eg: CCE), -1: all modules, others: invalid
 * @param [in]level: log level(0: debug, 1: info, 2: warning, 3: error, 4: null output)
 * @param [in]enableEvent: 1: enable; 0: disable, others:invalid
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogSetlevelForC(int moduleId, int level, int enableEvent);

/**
 * @ingroup slog
 * @brief CheckLogLevelForC: check module level enable or not
 * users no need to call it because all dlog interface(include inner interface) has already called
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]logLevel: eg: DLOG_EVENT/DLOG_ERROR/DLOG_WARN/DLOG_INFO/DLOG_DEBUG
 * @return: 1:enable, 0:disable
 */
DLL_EXPORT int CheckLogLevelForC(int moduleId, int logLevel);

/**
 * @ingroup slog
 * @brief DlogSetAttrForC: set log attr, default pid is 0, default device id is 0, default process type is APPLICATION
 * @param [in]logAttr: attr info, include pid(must be larger than 0), process type and device id(chip ID)
 * @return: 0: SUCCEED, others: FAILED
 */
DLL_EXPORT int DlogSetAttrForC(LogAttr logAttr);

/**
 * @ingroup slog
 * @brief DlogForC: print log, need caller to specify level
 * call CheckLogLevelForC in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]fmt: log content
 */
#define DlogForC(moduleId, level, fmt, ...)                                                 \
    do {                                                                                  \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                           \
            DlogInnerForC(moduleId, level, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__);   \
        }                                                                                  \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogSubForC: print log, need caller to specify level and submodule
 * call CheckLogLevelForC in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]submodule: eg: engine
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]fmt: log content
 */
#define DlogSubForC(moduleId, submodule, level, fmt, ...)                                                   \
    do {                                                                                                  \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                                           \
            DlogInnerForC(moduleId, level, "[%s:%d][%s]" fmt, __FILE__, __LINE__, submodule, ##__VA_ARGS__);    \
        }                                                                                                   \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogWithKVForC: print log, need caller to specify level and other paramters
 * call CheckLogLevelForC in advance to optimize performance, call interface with fmt input take time
 *
 * @param [in]moduleId: module id, eg: CCE
 * @param [in]level(0: debug, 1: info, 2: warning, 3: error, 5: trace, 6: oplog, 16: event)
 * @param [in]pstKVArray: key-value array
 * @param [in]kvNum: key-value element num in array
 * @param [in]fmt: log content
 */
#define DlogWithKVForC(moduleId, level, pstKVArray, kvNum, fmt, ...)                                                \
    do {                                                                                                          \
        if (CheckLogLevelForC(moduleId, level) == 1) {                                                                \
            DlogWithKVInnerForC(moduleId, level, pstKVArray, kvNum, "[%s:%d]" fmt, __FILE__, __LINE__, ##__VA_ARGS__); \
        }                                                                                                           \
    } while (TMP_LOG != 0)

/**
 * @ingroup slog
 * @brief DlogFlushForC: flush log buffer to file
 */
DLL_EXPORT void DlogFlushForC(void);

/**
 * @ingroup slog
 * @brief Internal log interface, other modules are not allowed to call this interface
 */
void DlogInnerForC(int moduleId, int level, const char *fmt, ...);
void DlogWithKVInnerForC(int moduleId, int level, KeyValue *pstKVArray, int kvNum, const char *fmt, ...);

#ifdef __cplusplus
}
#endif
#endif // LOG_CPP
#endif // D_SYSLOG_H_
