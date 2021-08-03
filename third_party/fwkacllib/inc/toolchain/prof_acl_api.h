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

#ifndef MSPROFILER_API_PROF_ACL_API_H_
#define MSPROFILER_API_PROF_ACL_API_H_

// DataTypeConfig
#define PROF_ACL_API                0x00000001
#define PROF_TASK_TIME              0x00000002
#define PROF_AICORE_METRICS         0x00000004
#define PROF_AICPU_TRACE            0x00000008
#define PROF_MODEL_EXECUTE          0x00000010
#define PROF_RUNTIME_API            0x00000020
#define PROF_RUNTIME_TRACE          0x00000040
#define PROF_SCHEDULE_TIMELINE      0x00000080
#define PROF_SCHEDULE_TRACE         0x00000100
#define PROF_AIVECTORCORE_METRICS   0x00000200
#define PROF_SUBTASK_TIME           0x00000400

#define PROF_TRAINING_TRACE         0x00000800
#define PROF_HCCL_TRACE             0x00001000

#define PROF_TASK_TRACE             0x00001852

// system profilinig switch
#define PROF_CPU                    0x00010000
#define PROF_HARDWARE_MEMORY        0x00020000
#define PROF_IO                     0x00040000
#define PROF_INTER_CONNECTION       0x00080000
#define PROF_DVPP                   0x00100000
#define PROF_SYS_AICORE_SAMPLE      0x00200000
#define PROF_AIVECTORCORE_SAMPLE    0x00400000

#define PROF_MODEL_LOAD             0x8000000000000000

// DataTypeConfig MASK
#define PROF_ACL_API_MASK                0x00000001
#define PROF_TASK_TIME_MASK              0x00000002
#define PROF_AICORE_METRICS_MASK         0x00000004
#define PROF_AICPU_TRACE_MASK            0x00000008
#define PROF_MODEL_EXECUTE_MASK          0x00000010
#define PROF_RUNTIME_API_MASK            0x00000020
#define PROF_RUNTIME_TRACE_MASK          0x00000040
#define PROF_SCHEDULE_TIMELINE_MASK      0x00000080
#define PROF_SCHEDULE_TRACE_MASK         0x00000100
#define PROF_AIVECTORCORE_METRICS_MASK   0x00000200
#define PROF_SUBTASK_TIME_MASK           0x00000400

#define PROF_TRAINING_TRACE_MASK         0x00000800
#define PROF_HCCL_TRACE_MASK             0x00001000

// system profilinig mask
#define PROF_CPU_MASK                    0x00010000
#define PROF_HARDWARE_MEMORY_MASK        0x00020000
#define PROF_IO_MASK                     0x00040000
#define PROF_INTER_CONNECTION_MASK       0x00080000
#define PROF_DVPP_MASK                   0x00100000
#define PROF_SYS_AICORE_SAMPLE_MASK      0x00200000
#define PROF_AIVECTORCORE_SAMPLE_MASK    0x00400000

#define PROF_MODEL_LOAD_MASK             0x8000000000000000

#ifndef OS_TYPE
#define OS_TYPE 0
#endif // OS_TYPE

#if (OS_TYPE != LINUX)
#define MSVP_PROF_API __declspec(dllexport)
#else
#define MSVP_PROF_API __attribute__((visibility("default")))
#endif

#include <cstdint>
#include <stddef.h>

namespace Msprofiler {
namespace Api {
/**
 * @name  ProfGetOpExecutionTime
 * @brief get op execution time of specific part of data
 * @param data  [IN] data read from pipe
 * @param len   [IN] data length
 * @param index [IN] index of part(op)
 * @return op execution time (us)
 */
MSVP_PROF_API uint64_t ProfGetOpExecutionTime(const void *data, uint32_t len, uint32_t index);
}
}

#ifdef __cplusplus
extern "C" {
#endif

MSVP_PROF_API uint64_t ProfGetOpExecutionTime(const void *data, uint32_t len, uint32_t index);

typedef uint32_t Status;
typedef struct aclprofSubscribeConfig aclprofSubscribeConfig1;
///
/// @ingroup AscendCL
/// @brief subscribe profiling data of graph
/// @param [in] graphId: the graph id subscribed
/// @param [in] profSubscribeConfig: pointer to config of model subscribe
/// @return Status result of function
///
Status aclgrphProfGraphSubscribe(const uint32_t graphId,
    const aclprofSubscribeConfig1 *profSubscribeConfig);

///
/// @ingroup AscendCL
/// @brief unsubscribe profiling data of graph
/// @param [in] graphId: the graph id subscribed
/// @return Status result of function
///
Status aclgrphProfGraphUnSubscribe(const uint32_t graphId);

/**
 * @ingroup AscendCL
 * @brief get graph id from subscription data
 *
 * @param  opInfo [IN]     pointer to subscription data
 * @param  opInfoLen [IN]  memory size of subscription data
 *
 * @retval graph id of subscription data
 * @retval 0 for failed
 */
size_t aclprofGetGraphId(const void *opInfo, size_t opInfoLen, uint32_t index);
#ifdef __cplusplus
}
#endif

#endif  // MSPROFILER_API_PROF_ACL_API_H_
