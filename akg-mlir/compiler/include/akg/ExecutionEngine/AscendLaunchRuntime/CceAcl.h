/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

/*!
 *  \file cce_acl.h
 *  \brief cce acl symbols
 */

/*!
 * 2024.1.24 - Add file cce_acl.h.
 */

/*!
 * 2024.5.13 - Add file CceAcl.h.
 */

#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEACL_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEACL_H_

#include <cstddef>
#include <cstdint>

#define RT_STREAM_DEFAULT (0x00U)
#define RT_STREAM_PERSISTENT (0x01U)
#define RT_STREAM_FORCE_COPY (0x02U)
#define RT_STREAM_HUGE (0x04U)
#define RT_STREAM_AICPU (0x08U)
#define RT_STREAM_FORBIDDEN_DEFAULT (0x10U)
#define RT_STREAM_HEAD (0x20U)
#define RT_STREAM_PRIMARY_DEFAULT (0x40U)
#define RT_STREAM_PRIMARY_FIRST_DEFAULT (0x80U)
#define RT_STREAM_OVERFLOW (0x100U)
#define RT_STREAM_FAST_LAUNCH (0x200U)
#define RT_STREAM_FAST_SYNC (0x400U)
#define RT_STREAM_CP_PROCESS_USE (0x800U)

// profiling
#define ACL_PROF_ACL_API 0x0001ULL
#define ACL_PROF_TASK_TIME 0x0002ULL
#define ACL_PROF_AICORE_METRICS 0x0004ULL
#define ACL_PROF_AICPU 0x0008ULL
#define ACL_PROF_L2CACHE 0x0010ULL
#define ACL_PROF_HCCL_TRACE 0x0020ULL
#define ACL_PROF_TRAINING_TRACE 0x0040ULL
#define ACL_PROF_MSPROFTX 0x0080ULL
#define ACL_PROF_RUNTIME_API 0x0100ULL
#define ACL_PROF_FWK_SCHEDULE_L0 0x0200ULL
#define ACL_PROF_TASK_TIME_L0 0x0800ULL
#define ACL_PROF_TASK_MEMORY 0x1000ULL
#define ACL_PROF_FWK_SCHEDULE_L1 0x01000000ULL
#define ACL_PROF_TASK_TIME_L2 0x2000ULL

#ifndef char_t
typedef char char_t;
#endif

static const int ACL_SUCCESS = 0;
static const int ACL_ERROR_NONE = 0;

static const int32_t ACL_RT_SUCCESS = 0;                              // success
static const int32_t ACL_ERROR_RT_PARAM_INVALID = 107000;             // param invalid
static const int32_t ACL_ERROR_RT_INVALID_DEVICEID = 107001;          // invalid device id
static const int32_t ACL_ERROR_RT_CONTEXT_NULL = 107002;              // current context null
static const int32_t ACL_ERROR_RT_STREAM_CONTEXT = 107003;            // stream not in current context
static const int32_t ACL_ERROR_RT_MODEL_CONTEXT = 107004;             // model not in current context
static const int32_t ACL_ERROR_RT_STREAM_MODEL = 107005;              // stream not in model
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_INVALID = 107006;   // event timestamp invalid
static const int32_t ACL_ERROR_RT_EVENT_TIMESTAMP_REVERSAL = 107007;  // event timestamp reversal
static const int32_t ACL_ERROR_RT_ADDR_UNALIGNED = 107008;            // memory address unaligned
static const int32_t ACL_ERROR_RT_FILE_OPEN = 107009;                 // open file failed
static const int32_t ACL_ERROR_RT_FILE_WRITE = 107010;                // write file failed
static const int32_t ACL_ERROR_RT_STREAM_SUBSCRIBE = 107011;          // error subscribe stream
static const int32_t ACL_ERROR_RT_THREAD_SUBSCRIBE = 107012;          // error subscribe thread
static const int32_t ACL_ERROR_RT_GROUP_NOT_SET = 107013;             // group not set
static const int32_t ACL_ERROR_RT_GROUP_NOT_CREATE = 107014;          // group not create
static const int32_t ACL_ERROR_RT_STREAM_NO_CB_REG = 107015;          // callback not register to stream
static const int32_t ACL_ERROR_RT_INVALID_MEMORY_TYPE = 107016;       // invalid memory type
static const int32_t ACL_ERROR_RT_INVALID_HANDLE = 107017;            // invalid handle
static const int32_t ACL_ERROR_RT_INVALID_MALLOC_TYPE = 107018;       // invalid malloc type
static const int32_t ACL_ERROR_RT_WAIT_TIMEOUT = 107019;              // wait timeout
static const int32_t ACL_ERROR_RT_TASK_TIMEOUT = 107020;              // task timeout

static const int32_t ACL_ERROR_RT_FEATURE_NOT_SUPPORT = 207000;  // feature not support
static const int32_t ACL_ERROR_RT_MEMORY_ALLOCATION = 207001;    // memory allocation error
static const int32_t ACL_ERROR_RT_MEMORY_FREE = 207002;          // memory free error
static const int32_t ACL_ERROR_RT_AICORE_OVER_FLOW = 207003;     // aicore over flow
static const int32_t ACL_ERROR_RT_NO_DEVICE = 207004;            // no device
static const int32_t ACL_ERROR_RT_RESOURCE_ALLOC_FAIL = 207005;  // resource alloc fail
static const int32_t ACL_ERROR_RT_NO_PERMISSION = 207006;        // no permission
static const int32_t ACL_ERROR_RT_NO_EVENT_RESOURCE = 207007;    // no event resource
static const int32_t ACL_ERROR_RT_NO_STREAM_RESOURCE = 207008;   // no stream resource
static const int32_t ACL_ERROR_RT_NO_NOTIFY_RESOURCE = 207009;   // no notify resource
static const int32_t ACL_ERROR_RT_NO_MODEL_RESOURCE = 207010;    // no model resource
static const int32_t ACL_ERROR_RT_NO_CDQ_RESOURCE = 207011;      // no cdq resource
static const int32_t ACL_ERROR_RT_OVER_LIMIT = 207012;           // over limit
static const int32_t ACL_ERROR_RT_QUEUE_EMPTY = 207013;          // queue is empty
static const int32_t ACL_ERROR_RT_QUEUE_FULL = 207014;           // queue is full
static const int32_t ACL_ERROR_RT_REPEATED_INIT = 207015;        // repeated init
static const int32_t ACL_ERROR_RT_AIVEC_OVER_FLOW = 207016;      // aivec over flow
static const int32_t ACL_ERROR_RT_OVER_FLOW = 207017;            // common over flow
static const int32_t ACL_ERROR_RT_DEVIDE_OOM = 207018;           // device oom
static const int32_t ACL_ERROR_RT_SEND_MSG = 207019;             // hdc send msg fail
static const int32_t ACL_ERROR_RT_COPY_USER_FAIL = 207020;       // copy data fail

static const int32_t ACL_ERROR_RT_INTERNAL_ERROR = 507000;                   // runtime internal error
static const int32_t ACL_ERROR_RT_TS_ERROR = 507001;                         // ts internal error
static const int32_t ACL_ERROR_RT_STREAM_TASK_FULL = 507002;                 // task full in stream
static const int32_t ACL_ERROR_RT_STREAM_TASK_EMPTY = 507003;                // task empty in stream
static const int32_t ACL_ERROR_RT_STREAM_NOT_COMPLETE = 507004;              // stream not complete
static const int32_t ACL_ERROR_RT_END_OF_SEQUENCE = 507005;                  // end of sequence
static const int32_t ACL_ERROR_RT_EVENT_NOT_COMPLETE = 507006;               // event not complete
static const int32_t ACL_ERROR_RT_CONTEXT_RELEASE_ERROR = 507007;            // context release error
static const int32_t ACL_ERROR_RT_SOC_VERSION = 507008;                      // soc version error
static const int32_t ACL_ERROR_RT_TASK_TYPE_NOT_SUPPORT = 507009;            // task type not support
static const int32_t ACL_ERROR_RT_LOST_HEARTBEAT = 507010;                   // ts lost heartbeat
static const int32_t ACL_ERROR_RT_MODEL_EXECUTE = 507011;                    // model execute failed
static const int32_t ACL_ERROR_RT_REPORT_TIMEOUT = 507012;                   // report timeout
static const int32_t ACL_ERROR_RT_SYS_DMA = 507013;                          // sys dma error
static const int32_t ACL_ERROR_RT_AICORE_TIMEOUT = 507014;                   // aicore timeout
static const int32_t ACL_ERROR_RT_AICORE_EXCEPTION = 507015;                 // aicore exception
static const int32_t ACL_ERROR_RT_AICORE_TRAP_EXCEPTION = 507016;            // aicore trap exception
static const int32_t ACL_ERROR_RT_AICPU_TIMEOUT = 507017;                    // aicpu timeout
static const int32_t ACL_ERROR_RT_AICPU_EXCEPTION = 507018;                  // aicpu exception
static const int32_t ACL_ERROR_RT_AICPU_DATADUMP_RSP_ERR = 507019;           // aicpu datadump response error
static const int32_t ACL_ERROR_RT_AICPU_MODEL_RSP_ERR = 507020;              // aicpu model operate response error
static const int32_t ACL_ERROR_RT_PROFILING_ERROR = 507021;                  // profiling error
static const int32_t ACL_ERROR_RT_IPC_ERROR = 507022;                        // ipc error
static const int32_t ACL_ERROR_RT_MODEL_ABORT_NORMAL = 507023;               // model abort normal
static const int32_t ACL_ERROR_RT_KERNEL_UNREGISTERING = 507024;             // kernel unregistering
static const int32_t ACL_ERROR_RT_RINGBUFFER_NOT_INIT = 507025;              // ringbuffer not init
static const int32_t ACL_ERROR_RT_RINGBUFFER_NO_DATA = 507026;               // ringbuffer no data
static const int32_t ACL_ERROR_RT_KERNEL_LOOKUP = 507027;                    // kernel lookup error
static const int32_t ACL_ERROR_RT_KERNEL_DUPLICATE = 507028;                 // kernel register duplicate
static const int32_t ACL_ERROR_RT_DEBUG_REGISTER_FAIL = 507029;              // debug register failed
static const int32_t ACL_ERROR_RT_DEBUG_UNREGISTER_FAIL = 507030;            // debug unregister failed
static const int32_t ACL_ERROR_RT_LABEL_CONTEXT = 507031;                    // label not in current context
static const int32_t ACL_ERROR_RT_PROGRAM_USE_OUT = 507032;                  // program register num use out
static const int32_t ACL_ERROR_RT_DEV_SETUP_ERROR = 507033;                  // device setup error
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TIMEOUT = 507034;              // vector core timeout
static const int32_t ACL_ERROR_RT_VECTOR_CORE_EXCEPTION = 507035;            // vector core exception
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TRAP_EXCEPTION = 507036;       // vector core trap exception
static const int32_t ACL_ERROR_RT_CDQ_BATCH_ABNORMAL = 507037;               // cdq alloc batch abnormal
static const int32_t ACL_ERROR_RT_DIE_MODE_CHANGE_ERROR = 507038;            // can not change die mode
static const int32_t ACL_ERROR_RT_DIE_SET_ERROR = 507039;                    // single die mode can not set die
static const int32_t ACL_ERROR_RT_INVALID_DIEID = 507040;                    // invalid die id
static const int32_t ACL_ERROR_RT_DIE_MODE_NOT_SET = 507041;                 // die mode not set
static const int32_t ACL_ERROR_RT_AICORE_TRAP_READ_OVERFLOW = 507042;        // aic trap read overflow
static const int32_t ACL_ERROR_RT_AICORE_TRAP_WRITE_OVERFLOW = 507043;       // aic trap write overflow
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TRAP_READ_OVERFLOW = 507044;   // aiv trap read overflow
static const int32_t ACL_ERROR_RT_VECTOR_CORE_TRAP_WRITE_OVERFLOW = 507045;  // aiv trap write overflow
static const int32_t ACL_ERROR_RT_STREAM_SYNC_TIMEOUT = 507046;              // stream sync time out
static const int32_t ACL_ERROR_RT_EVENT_SYNC_TIMEOUT = 507047;               // event sync time out
static const int32_t ACL_ERROR_RT_FFTS_PLUS_TIMEOUT = 507048;                // ffts+ timeout
static const int32_t ACL_ERROR_RT_FFTS_PLUS_EXCEPTION = 507049;              // ffts+ exception
static const int32_t ACL_ERROR_RT_FFTS_PLUS_TRAP_EXCEPTION = 507050;         // ffts+ trap exception

static const int32_t ACL_ERROR_RT_DRV_INTERNAL_ERROR = 507899;    // drv internal error
static const int32_t ACL_ERROR_RT_AICPU_INTERNAL_ERROR = 507900;  // aicpu internal error
static const int32_t ACL_ERROR_RT_SOCKET_CLOSE = 507901;          // hdc disconnect

typedef void *aclrtStream;
typedef void *aclrtContext;
typedef int aclError;
typedef int rtError_t;
typedef void *rtStream_t;

typedef enum aclrtMemcpyKind {
  ACL_MEMCPY_HOST_TO_HOST,
  ACL_MEMCPY_HOST_TO_DEVICE,
  ACL_MEMCPY_DEVICE_TO_HOST,
  ACL_MEMCPY_DEVICE_TO_DEVICE,
} aclrtMemcpyKind;

typedef enum aclrtMemMallocPolicy {
  ACL_MEM_MALLOC_HUGE_FIRST,
  ACL_MEM_MALLOC_HUGE_ONLY,
  ACL_MEM_MALLOC_NORMAL_ONLY,
  ACL_MEM_MALLOC_HUGE_FIRST_P2P,
  ACL_MEM_MALLOC_HUGE_ONLY_P2P,
  ACL_MEM_MALLOC_NORMAL_ONLY_P2P,
  ACL_MEM_TYPE_LOW_BAND_WIDTH = 0x0100,
  ACL_MEM_TYPE_HIGH_BAND_WIDTH = 0x1000,
} aclrtMemMallocPolicy;

typedef enum aclrtMemAttr {
  ACL_DDR_MEM,
  ACL_HBM_MEM,
  ACL_DDR_MEM_HUGE,
  ACL_DDR_MEM_NORMAL,
  ACL_HBM_MEM_HUGE,
  ACL_HBM_MEM_NORMAL,
  ACL_DDR_MEM_P2P_HUGE,
  ACL_DDR_MEM_P2P_NORMAL,
  ACL_HBM_MEM_P2P_HUGE,
  ACL_HBM_MEM_P2P_NORMAL,
} aclrtMemAttr;

// profiling
typedef struct aclprofConfig aclprofConfig;
typedef struct aclprofAicoreEvents aclprofAicoreEvents;
typedef enum {
  ACL_AICORE_ARITHMETIC_UTILIZATION = 0,
  ACL_AICORE_PIPE_UTILIZATION = 1,
  ACL_AICORE_MEMORY_BANDWIDTH = 2,
  ACL_AICORE_L0B_AND_WIDTH = 3,
  ACL_AICORE_RESOURCE_CONFLICT_RATIO = 4,
  ACL_AICORE_MEMORY_UB = 5,
  ACL_AICORE_L2_CACHE = 6,
  ACL_AICORE_PIPE_EXECUTE_UTILIZATION = 7,
  ACL_AICORE_NONE = 0xFF
} aclprofAicoreMetrics;

typedef struct tagRtDevBinary {
  uint32_t magic;    // magic number
  uint32_t version;  // version of binary
  const void *data;  // binary data
  uint64_t length;   // binary length
} rtDevBinary_t;

typedef struct tagRtSmData {
  uint64_t L2_mirror_addr;
  uint64_t L2_data_section_size;
  uint8_t L2_preload;
  uint8_t modified;
  uint8_t priority;
  int8_t pre_L2_page_offset_base;
  uint8_t L2_page_offset_base;
  uint8_t L2_load_to_ddr;
  uint8_t reserved[2];
} rtSmData_t;

typedef struct tagRtSmCtrl {
  rtSmData_t data[8];
  uint64_t size;
  uint8_t remap[64];
  uint8_t l2_in_main;
  uint8_t reserved[3];
} rtSmDesc_t;

aclError aclrtSetCurrentContext(aclrtContext context);
aclError aclrtGetDeviceCount(uint32_t *count);
aclError aclrtGetCurrentContext(aclrtContext *context);
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag);
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind,
                          aclrtStream stream);
aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t *free, size_t *total);
aclError aclrtSetDevice(int32_t deviceId);
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId);
aclError aclrtCreateStream(aclrtStream *stream);
aclError aclrtMallocHost(void **hostPtr, size_t size);
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy);
aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind);
aclError aclrtSynchronizeStream(aclrtStream stream);
aclError aclrtFree(void *devPtr);
aclError aclrtFreeHost(void *hostPtr);
aclError aclrtDestroyStream(aclrtStream stream);
aclError aclrtDestroyContext(aclrtContext context);
aclError aclrtResetDevice(int32_t deviceId);
aclError aclrtGetDevice(int32_t *deviceId);

// profiling
aclError aclprofInit(const char *profilerResultPath, size_t length);
aclError aclprofStart(const aclprofConfig *profilerConfig);
aclError aclprofStop(const aclprofConfig *profilerConfig);
aclError aclprofFinalize();
aclprofConfig *aclprofCreateConfig(uint32_t *deviceIdList, uint32_t deviceNums, aclprofAicoreMetrics aicoreMetrics,
                                   const aclprofAicoreEvents *aicoreEvents, uint64_t dataTypeConfig);
aclError aclprofDestroyConfig(const aclprofConfig *profilerConfig);

extern "C" {
rtError_t rtGetC2cCtrlAddr(uint64_t *addr, uint32_t *len);
rtError_t rtConfigureCall(uint32_t numBlocks, rtSmDesc_t *smDesc = nullptr, rtStream_t stm = nullptr);
rtError_t rtDevBinaryRegister(const rtDevBinary_t *bin, void **hdl);
rtError_t rtDevBinaryUnRegister(void *hdl);
rtError_t rtFunctionRegister(void *binHandle, const void *stubFunc, const char_t *stubName, const void *kernelInfoExt,
                             uint32_t funcMode);
rtError_t rtKernelLaunch(const void *stubFunc, uint32_t blockDim, void *arg, uint32_t argsSize, rtSmDesc_t *smDesc,
                         rtStream_t stm);
rtError_t rtLaunch(const void *stubFunc);
rtError_t rtSetupArgument(const void *args, uint32_t size, uint32_t offset);
}

#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGASCENDLAUNCHRUNTIME_CCEACL_H_
