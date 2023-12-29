/**
* Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*
* Description:aicpu common header
* Create: 2021-11-24
*/
#ifndef AICPU_CONTEXT_H
#define AICPU_CONTEXT_H

#include <cstdint>
#include <string>
#include <map>
#include <functional>
#include <unistd.h>

namespace aicpu {
typedef struct {
    uint32_t deviceId;    // device id
    uint32_t tsId;        // ts id
    pid_t hostPid;        // host pid
    uint32_t vfId;        // vf id
} aicpuContext_t;

typedef struct {
    uint64_t tickBeforeRun;
    uint64_t drvSubmitTick;
    uint64_t drvSchedTick;
    uint32_t kernelType;
} aicpuProfContext_t;

enum AicpuRunMode : uint32_t {
    PROCESS_PCIE_MODE = 0U,    // dc, with host mode
    PROCESS_SOCKET_MODE = 1U,  // MDC
    THREAD_MODE = 2U,          // ctrlcpu/minirc/lhisi
    INVALID_MODE = 3U,
};

enum AicpuDvppChlType : uint32_t {
    AICPU_DVPP_CHL_VPC  = 0U,
    AICPU_DVPP_CHL_VDEC = 1U,
    AICPU_DVPP_CHL_BUTT
};

typedef struct {
    uint32_t streamId;
    uint64_t taskId;
} streamAndTaskId_t;

typedef enum {
    AICPU_ERROR_NONE = 0,   // success
    AICPU_ERROR_FAILED = 1, // failed
} status_t;

enum CtxType : int32_t {
    CTX_DEFAULT = 0,
    CTX_PROF,
    CTX_DEBUG
};

const std::string CONTEXT_KEY_OP_NAME = "opname";
const std::string CONTEXT_KEY_PHASE_ONE_FLAG = "phaseOne";
const std::string CONTEXT_KEY_WAIT_TYPE = "waitType";
const std::string CONTEXT_KEY_WAIT_ID = "waitId";

/**
 * set aicpu context for current thread.
 * @param [in]ctx aicpu context
 * @return status whether this operation success
 */
status_t aicpuSetContext(aicpuContext_t *ctx);

/**
 * get aicpu context from current thread.
 * @param [out]ctx aicpu context
 * @return status whether this operation success
 */
status_t __attribute__((weak)) aicpuGetContext(aicpuContext_t *ctx);

/**
 * set aicpu prof context for current thread.
 * @param [ctx]ctx aicpu prof context
 * @return status whether this operation success
 */
status_t __attribute__((weak)) aicpuSetProfContext(const aicpuProfContext_t &ctx);

/**
 * get aicpu context prof from current thread.
 * @return ctx aicpu prof context
 */
__attribute__((weak)) const aicpuProfContext_t &aicpuGetProfContext();

/**
 * init context for task monitor, called in compute process start.
 * @param [in]aicpuCoreCnt aicpu core number
 * @return status whether this operation success
 */
status_t InitTaskMonitorContext(uint32_t aicpuCoreCnt);

/**
 * set aicpu thread index for task monitor, called in thread callback function.
 * @param [in]threadIndex aicpu thread index
 * @return status whether this operation success
 */
status_t SetAicpuThreadIndex(uint32_t threadIndex);

/**
 * get aicpu thread index.
 * @return uint32
 */
uint32_t GetAicpuThreadIndex();

/**
 * set op name for task monitor.
 * called in libtf_kernels.so(tf op) or libaicpu_processer.so(others) or cpu kernel framework.
 * @param [in]opname op name
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetOpname(const std::string &opname);

/**
 * get op name for task monitor
 * @param [in]threadIndex thread index
 * @param [out]opname op name
 * @return status whether this operation success
 */
status_t GetOpname(uint32_t threadIndex, std::string &opname);

/**
 * get task and stream id.
 * @param [in]taskId task id.
 * @param [in]streamId stream id.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) GetTaskAndStreamId(uint64_t &taskId, uint32_t &streamId);

/**
 * set task and stream id.
 * @param [in]taskId task id.
 * @param [in]streamId stream id.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetTaskAndStreamId(uint64_t taskId, uint32_t streamId);

/**
 * Set the dvpp channel id of stream
 * @param [in]streamId: stream id
 * @param [in]chlType: channel type
 * @param [in]channelId: dvpp channel id
 * @return return the valid dvpp channel id of stream
 */
int32_t __attribute__((weak)) InitStreamDvppChannel(uint32_t streamId,
    AicpuDvppChlType chlType, int32_t channelId);

/**
 * Undo set the dvpp channel id of stream
 * @param [in]streamId: stream id
 * @param [in]chlType: channel type
 * @return return the dvpp channel id of stream
 */
int32_t __attribute__((weak)) UnInitStreamDvppChannel(uint32_t streamId, AicpuDvppChlType chlType);

/**
 * Get the dvpp channel id of stream
 * @param [in]streamId: stream id
 * @param [in]chlType: channel type
 * @return return the dvpp channel id of stream
 */
int32_t __attribute__((weak)) GetStreamDvppChannelId(uint32_t streamId, AicpuDvppChlType chlType);

/**
 * Get the dvpp channel id of current stream
 * @param [in]chlType: channel type
 * @return return the dvpp channel id of stream
 */
int32_t __attribute__((weak)) GetCurTaskDvppChannelId(AicpuDvppChlType chlType);

/**
 * set the dvpp buff len of current stream
 * @param [in]chlType: channel type
 * @param [in]buffLen: buff len
 * @param [in]buff: buff addr
 * @return return NA
 */
void __attribute__((weak)) SetStreamDvppBuffBychlType(const AicpuDvppChlType chlType,
                                                      const uint64_t buffLen, uint8_t *buff);

/**
 * set the dvpp buff len of current stream
 * @param [in]chlType: channel type
 * @param [in]streamId: stream id
 * @param [in]buffLen: buff len
 * @param [in]buff: buff addr
 * @return return NA
 */
void __attribute__((weak)) SetStreamDvppBuffByStreamId(const AicpuDvppChlType chlType, const uint32_t streamId,
                                                       const uint64_t buffLen, uint8_t *buff);

/**
 * set the dvpp buff len of current stream
 * @param [in]chlType: channel type
 * @param [out]buff: buff addr
 * @param [out]buffLen: buff len
 * @return return NA
 */
void __attribute__((weak)) GetDvppBufAndLenBychlType(const AicpuDvppChlType chlType,
                                                     uint8_t **buff, uint64_t *buffLen);

/**
 * set the dvpp buff len of current stream
 * @param [in]streamId: stream id
 * @param [in]chlType: channel type
 * @param [out]buff: buff addr
 * @return return NA
 */
void __attribute__((weak)) GetDvppBufAndLenByStreamId(const uint32_t streamId,
                                                      const AicpuDvppChlType chlType, uint8_t **buff);


/**
 * set thread local context of key
 * @param [in]key context key
 * @param [in]value context value
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by SetThreadCtxInfo
 */
status_t __attribute__((weak)) SetThreadLocalCtx(const std::string &key, const std::string &value);

/**
 * get thread local context of key
 * @param [in]key context key
 * @param [out]value context value
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by GetThreadCtxInfo
 */
status_t GetThreadLocalCtx(const std::string &key, std::string &value);

/**
 * remove local context of key
 * @param [in]key context key
 * @return status whether this operation success
 * @note Deprecated from 20201216, Replaced by RemoveThreadCtxInfo
 */
status_t RemoveThreadLocalCtx(const std::string &key);

/**
 * get all thread context info of type
 * @param [in]type: ctx type
 * @param [in]threadIndex: thread index
 * @return const std::map<std::string, std::string> &: all thread context info
 */
const std::map<std::string, std::string> &GetAllThreadCtxInfo(aicpu::CtxType type, uint32_t threadIndex);

/**
 * set run mode.
 * @param [in]runMode: run mode.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) SetAicpuRunMode(uint32_t runMode);

/**
 * get run mode.
 * @param [out]runMode: run mode.
 * @return status whether this operation success
 */
status_t __attribute__((weak)) GetAicpuRunMode(uint32_t &runMode);

/**
 * Register callback function by eventId and subeventId
 * @param eventId event id
 * @param subeventId subevent id
 * @param func call back function
 */
status_t __attribute__((weak)) RegisterEventCallback(const uint32_t eventId,
    const uint32_t subeventId, std::function<void (void*)> func,
    const bool isNeedClear = true);

/**
 * Do callback function by eventId and subeventId
 * @param eventId event id
 * @param subeventId subevent id
 * @param param event param
 */
status_t __attribute__((weak)) DoEventCallback(const uint32_t eventId,
    const uint32_t subeventId, void * const param);

/**
 * Unregister callback function by eventId and subeventId
 * @param eventId event id
 * @param subeventId subevent id
 */
status_t __attribute__((weak)) UnRegisterCallback(const uint32_t eventId, const uint32_t subeventId);

/**
 * get unique vf id
 * @return unique vf id
 */
uint32_t __attribute__((weak)) GetUniqueVfId();

/**
 * Set unique vf id
 * @param uniqueVfId unique vf id
 */
void __attribute__((weak)) SetUniqueVfId(const uint32_t uniqueVfId);

/**
 * SetCustAicpuSdFlag
 * @param [in]  is Cust Aicpu Sd or  not
 * @return NA
 */
void __attribute__((weak)) SetCustAicpuSdFlag(const bool isCustAicpuSdFlag);
/**
 * IsCustAicpuSd
 * @param [in]  NA
 * @return bool  is cusAicpuSd or not
 */
bool __attribute__((weak)) IsCustAicpuSd();
} // namespace aicpu

extern "C" {
/**
 * set thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @param [in]value: value of context info
 * @return status whether this operation success
 */
__attribute__((visibility("default"))) aicpu::status_t SetThreadCtxInfo(aicpu::CtxType type, const std::string &key,
    const std::string &value);

/**
 * get thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @param [out]value: value of context info
 * @return status whether this operation success
 */
__attribute__((visibility("default"))) aicpu::status_t GetThreadCtxInfo(aicpu::CtxType type, const std::string &key,
    std::string &value);

/**
 * remove thread context info of type
 * @param [in]type: ctx type
 * @param [in]key: key of context info
 * @return status whether this operation success
 */
__attribute__((visibility("default"))) aicpu::status_t RemoveThreadCtxInfo(aicpu::CtxType type, const std::string &key);
}
#endif // AICPU_CONTEXT_H_
