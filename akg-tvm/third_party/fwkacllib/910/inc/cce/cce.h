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

#ifndef CCE_H__
#define CCE_H__

#include <stdint.h>
#include "cce_def.hpp"

namespace cce {

/**
 * @ingroup cce
 * @brief create cc handler
 * @param [in|out] handle   point of cc handler
 * @return ccStatus_t
 */
ccStatus_t ccCreate(ccHandle_t *handle);

/**
 * @ingroup cce
 * @brief destroy cc handler
 * @param [in] *handle   cc handler
 * @return ccStatus_t
 */
ccStatus_t ccDestroy(ccHandle_t *handle);

/**
 * @ingroup cce
 * @brief bind stream with specified cc handler
 * @param [in] handle   cc handler
 * @param [in] streamId   stream
 * @return ccStatus_t
 */
ccStatus_t ccSetStream(ccHandle_t handle, rtStream_t streamId);

/**
 * @ingroup cce
 * @brief get the stream from cc handler
 * @param [in] handle   cc handler
 * @param [in|out] streamId   point of stream
 * @return ccStatus_t
 */
ccStatus_t ccGetStream(ccHandle_t handle, rtStream_t *streamId);

/**
 * @ingroup cce
 * @brief get the stream from cc handler
 * @param [in] dataTypeTransMode   mode of data type transform
 * @param [in] inputData   input data point
 * @param [in] inputDataSize   input data size
 * @param [in|out] outputData   output data point
 * @param [in] outputDataSize   output data size
 * @return ccStatus_t
 */
ccStatus_t ccTransDataType(ccDataTypeTransMode_t dataTypeTransMode, const void *inputData, uint32_t inputDataSize,
                           void *outputData, const uint32_t outputDataSize);
/**
 * @ingroup cce
 * @brief cce sys init func
 */
void cceSysInit();

/**
 * @ingroup cce
 * @brief cce Log Start up func
 */
void cceLogStartup();

/**
 * @ingroup cce
 * @brief cce Log Shut down func
 */
void cceLogShutdown();

/**
 * @ingroup cce
 * @brief set the profiling on or off
 * @param [in] const unsigned char* target: The engine gets it from ENV. Don't need care about it.
 * @param const char* job_ctx: identifies profiling job
 * @param [in] uint32_t flag: value: 0, on ; 1, off.
 * @return ccStatus_t value: 0, success; 1, fail.
 */
ccStatus_t CceProfilingConfig(const char *target, const char *job_ctx, uint32_t flag);

};  // namespace cce

#endif  // CCE_H__
