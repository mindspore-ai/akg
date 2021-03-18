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

#ifndef __CCE_RUNTIME_STREAM_H__
#define __CCE_RUNTIME_STREAM_H__

#include "base.h"
#include "event.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup stream_flags
 * @brief stream op bit flags
 */
#define RT_STREAM_DEFAULT (0x00)
#define RT_STREAM_PERSISTENT (0x01)
#define RT_STREAM_FORCE_COPY (0x02)
#define RT_STREAM_HUGE (0x04)
#define RT_STREAM_AICPU (0x08)
#define RT_STREAM_FORBIDDEN_DEFAULT (0x10)
#define RT_STREAM_HEAD (0x20)

/**
 * @ingroup stream_type
 * @brief stream type
 */
#define RT_NORMAL_STREAM    (0x00)
#define RT_HUGE_STREAM      (0x01)

/**
 * priority level default value when create a stream
 */
#define RT_STREAM_PRIORITY_DEFAULT (0)

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return RT_ERROR_INVALID_VALUE for error input priority
 */
RTS_API rtError_t rtStreamCreate(rtStream_t *stream, int32_t priority);

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @param [in] flags  stream op flags
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return RT_ERROR_INVALID_VALUE for error input priority
 */
RTS_API rtError_t rtStreamCreateWithFlags(rtStream_t *stream, int32_t priority, uint32_t flags);

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stream   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
RTS_API rtError_t rtStreamDestroy(rtStream_t stream);

/**
 * @ingroup dvrt_stream
 * @brief wait an recorded event for stream
 * @param [in] stream   the wait stream
 * @param [in] event   the event to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream or event handle
 */
RTS_API rtError_t rtStreamWaitEvent(rtStream_t stream, rtEvent_t event);

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stream   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream or event handle
 */
RTS_API rtError_t rtStreamSynchronize(rtStream_t stream);

/**
 * @ingroup dvrt_stream
 * @brief queries an asynchronous stream for completion status
 * @param [in] stream   stream to query
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_NOT_READY for not complete
 */
RTS_API rtError_t rtStreamQuery(rtStream_t stream);

/**
 * @ingroup dvrt_stream
 * @brief get stream id from a stream handle
 * @param [in] stream   stream hadle
 * @param [in] streamId   stream id
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
RTS_API rtError_t rtGetStreamId(rtStream_t stream, int32_t *streamId);

/**
 * @ingroup dvrt_stream
 * @brief inquire max stream count and max task count per stream
 * @param [in] streamType   Stream Type
 * @param [in] MaxStrCount   Max stream count
 * @param [in] MaxTaskCount   max task count per stream
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
RTS_API rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t *MaxStrCount, uint32_t *MaxTaskCount);

/**
 * @ingroup dvrt_stream
 * @brief Name a stream
 * @param [in] stream_  stream to be named
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 */
RTS_API rtError_t rtNameStream(rtStream_t stream_, const char *name);

/**
 * @ingroup dvrt_stream
 * @brief switch to the corresponding stream according to the contents of the ptr
 * @param [in] ptr  Determine the address where the value of the true and false branches is located
 * @param [in] condition switch condition
 * @param [in] value  switch value
 * @param [in] true_stream  Stream that needs to be activated when the value is non-zero
 * @param [in] stream input stream to init task
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 * @return RT_ERROR_INVALID_DEVICE for invalid device handle
 * @return ERROR_RECYCLE for switching task init failed or submit failed
 */
RTS_API rtError_t rtStreamSwitch(void *ptr, rtCondition_t condition, int64_t value, rtStream_t true_stream,
                                 rtStream_t stream);

/**
 * @brief execute extensible stream switch task
 * @param [in] ptr   pointer of value
 * @param [in] condition   judge condition
 * @param [in] value_ptr   pointer of target value
 * @param [in] true_stream   stream to be activated when value is not zero
 * @param [in] stream   stream id
 * @param [in] dataType   data type of target value
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for not complete
 */
RTS_API rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *value_ptr, rtStream_t true_stream,
                                   rtStream_t stream, rtSwitchDataType_t dataType);

/**
 * @ingroup dvrt_stream
 * @brief Active a stream
 * @param [in] active_stream stream to be activated
 * @param [in] stream input stream to init task
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for invalid resource handle
 * @return RT_ERROR_INVALID_DEVICE for invalid device handle
 * @return ERROR_RECYCLE for switching task init failed or submit failed
 */
RTS_API rtError_t rtStreamActive(rtStream_t active_stream, rtStream_t stream);

/**
 * @brief execute extensible stream case switch task
 * @param [in] ptr   pointer of value
 * @param [in] size  pointer num of value
 * @param [in] valuePtr  pointer of target value, length = size * elementSize
 * @param [in] trueStreamPtr streams to be activated
 * @param [in] elementSize  size of to be activated true streams
 * @param [in] stream input stream to init task
 * @param [in] dataType   data type of target value
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for not complete
 */
RTS_API rtError_t rtStreamSwitchN(void *ptr, uint32_t size, void *valuePtr, rtStream_t *trueStreamPtr,
                                  uint32_t elementSize, rtStream_t stream, rtSwitchDataType_t dataType);
#ifdef __cplusplus
}
#endif

#endif  // __CCE_RUNTIME_STREAM_H__
