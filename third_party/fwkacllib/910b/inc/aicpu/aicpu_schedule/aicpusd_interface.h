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

#ifndef AICPUSD_INTERFACE_H
#define AICPUSD_INTERFACE_H

#include <sched.h>

#include "aicpusd_info.h"
#include "prof_callback.h"
#include "aicpu/tsd/tsd.h"
#include "aicpu/aicpu_schedule/aicpu_sharder/aicpu_async_event.h"

extern "C" {
enum __attribute__((visibility("default"))) ErrorCode : int32_t {
    AICPU_SCHEDULE_SUCCESS,
    AICPU_SCHEDULE_FAIL,
    AICPU_SCHEDULE_ABORT,
};

/**
 * @brief it is used to load the task and stream info.
 * @param [in] ptr : the address of the task and stream info
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t AICPUModelLoad(void *ptr);

/**
 * @brief it is used to destory the model.
 * @param [in] modelId : The id of model will be destroy.
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t AICPUModelDestroy(uint32_t modelId);

/**
 * @brief it is used to execute the model.
 * @param [in] modelId : The id of model will be run.
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t AICPUModelExecute(uint32_t modelId);

/**
 * @brief it is used to init aicpu scheduler for acl.
 * @param [in] deviceId : The id of self cpu.
 * @param [in] hostPid :  pid of host appication
 * @param [in] profilingMode : it used to open or close profiling.
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t InitAICPUScheduler(uint32_t deviceId, pid_t hostPid,
                                                                  ProfilingMode profilingMode);

/**
 * @brief it is used to update profiling mode for acl.
 * @param [in] deviceId : The id of self cpu.
 * @param [in] hostPid : The id of host
 * @param [in] flag : flag[0] == 1 means PROFILING_OPEN, otherwise PROFILING_CLOSE.
 * @return AICPU_SCHEDULE_OK: sucess  other: error code in StatusCode
 */
__attribute__((visibility("default"))) int32_t UpdateProfilingMode(uint32_t deviceId, pid_t hostPid, uint32_t flag);

/**
 * @brief it is used to stop the aicpu scheduler for acl.
 * @param [in] deviceId : The id of self cpu.
 * @param [in] hostPid : pid of host appication
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t StopAICPUScheduler(uint32_t deviceId, pid_t hostPid);

/**
 * @ingroup AicpuScheduleInterface
 * @brief it use to execute the model from call interface.
 * @param [in] drvEventInfo : event info.
 * @param [out] eventAck : event ack.
 * @return 0: sucess, other: error code
 */
__attribute__((visibility("default"))) int32_t AICPUExecuteTask(struct event_info* drvEventInfo,
                                                                struct event_ack* drvEventAck);

/**
 * @ingroup AicpuScheduleInterface
 * @brief it use to preload so.
 * @param [in] soName : so name.
 * @return 0: sucess, other: error code
 */
__attribute__((visibility("default"))) int32_t AICPUPreOpenKernels(const char *soName);

/**
 * @brief it is used to load op mapping info for data dump.
 * @param [in] infoAddr : The pointer of info.
 * @param [in] len : The length of info
 * @return AICPU_SCHEDULE_OK: sucess  other: error code in StatusCode
 */
__attribute__((visibility("default"))) int32_t LoadOpMappingInfo(const void *infoAddr, uint32_t len);

/**
 * @brief it is used to set report callback function.
 * @param [in] reportCallback : report callback function.
 * @return AICPU_SCHEDULE_OK: sucess  other: error code in StatusCode
 */
__attribute__((visibility("default"))) int32_t AicpuSetMsprofReporterCallback(MsprofReporterCallback reportCallback);

/**
 * @brief it is used to init aicpu scheduler for helper.
 * @param [in] initParam : init param.
 * @return AICPU_SCHEDULE_SUCCESS: sucess  other: error code in ErrorCode
 */
__attribute__((visibility("default"))) int32_t InitCpuScheduler(const CpuSchedInitParam * const initParam);

/**
 * @brief it is used to load model with queue.
 * @param [in] ptr : the address of the model info
 * @return AICPU_SCHEDULE_OK: sucess  other: error code
 */
__attribute__((visibility("default"))) int32_t AicpuLoadModelWithQ(void *ptr);

/**
 * @brief it is used to stop aicpu module.
 * @param [in] eventInfo : the message send by tsd
 * @return AICPU_SCHEDULE_OK: sucess  other: error code
 */
__attribute__((visibility("default"))) int32_t StopAicpuSchedulerModule(const struct TsdSubEventInfo * const
                                                                        eventInfo);
/**
 * @brief it is used to start aicpu module.
 * @param [in] eventInfo : the message send by tsd
 * @return AICPU_SCHEDULE_OK: sucess  other: error code
 */
__attribute__((visibility("default"))) int32_t StartAicpuSchedulerModule(const struct TsdSubEventInfo * const
                                                                         eventInfo);

/**
 * @brief it is used to send retcode to ts.
 * @return AICPU_SCHEDULE_OK: sucess  other: error code
 */
__attribute__((weak)) __attribute__((visibility("default"))) void AicpuReportNotifyInfo(
    const aicpu::AsyncNotifyInfo &notifyInfo);

/**
 * @brief it is used to get task default timeout, uint is second.
 * @return timeout: unit is second
 */
__attribute__((weak)) __attribute__((visibility("default"))) uint32_t AicpuGetTaskDefaultTimeout();

/**
 * @brief Check if the scheduling module stops running
 * @return true or false
 */
__attribute__((weak)) __attribute__((visibility("default"))) bool AicpuIsStoped();

/**
 * @brief it is used to register last word.
 * @param [in] mark : module lable.
 * @param [in] callback : record last word callback.
 * @param [out] cancelDeadline : cancel reg closer.
 */
__attribute__((weak)) __attribute__((visibility("default"))) void RegLastwordCallback(const std::string mark,
    std::function<void ()> callback, std::function<void ()> &cancelReg);

/**
 * @brief it is used to load model with event.
 * @param [in] ptr : the address of the model info
 * @return AICPU_SCHEDULE_OK: sucess  other: error code
 */
__attribute__((visibility("default"))) int32_t AicpuLoadModel(void *ptr);
}
#endif  // INC_AICPUSD_AICPUSD_INTERFACE_H_
