/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: stream.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_STREAM_H
#define CCE_RUNTIME_STREAM_H

#include "base.h"
#include "event.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup stream_flags
 * @brief stream op bit flags
 */
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

/**
 * @ingroup stream_type
 * @brief stream type
 */
#define RT_NORMAL_STREAM    (0x00U)
#define RT_HUGE_STREAM      (0x01U)

/**
 * priority level default value when create a stream
 */
#define RT_STREAM_PRIORITY_DEFAULT (0U)

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stm   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamCreate(rtStream_t *stm, int32_t priority);

/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stm   created stream
 * @param [in] priority   stream priority
 * @param [in] flags  stream op flags
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamCreateWithFlags(rtStream_t *stm, int32_t priority, uint32_t flags);

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stm   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamDestroy(rtStream_t stm);

/**
 * @ingroup dvrt_stream
 * @brief wait an recorded event for stream
 * @param [in] stm   the wait stream
 * @param [in] event   the event to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamWaitEvent(rtStream_t stm, rtEvent_t evt);

/**
 * @ingroup dvrt_stream
 * @brief wait an recorded event for stream, used for 1951 pg1
 * @param [in] stm   the wait stream
 * @param [in] event   the event to wait
 * @param [in] timeout   timeout value for 1951 pg1
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamWaitEventWithTimeout(rtStream_t stm, rtEvent_t evt, uint32_t timeout);

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stm   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamSynchronize(rtStream_t stm);

/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete and set timeout
 * @param [in] stm   stream to wait
 * @param [in] timeout   timeout value,the unit is milliseconds
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamSynchronizeWithTimeout(rtStream_t stm, int32_t timeout);

/**
 * @ingroup dvrt_stream
 * @brief queries an asynchronous stream for completion status
 * @param [in] stm   stream to query
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_STREAM_NOT_COMPLETE for not complete
 */
RTS_API rtError_t rtStreamQuery(rtStream_t stm);

/**
 * @ingroup dvrt_stream
 * @brief get stream id from a stream handle
 * @param [in] stm   stream hadle
 * @param [in] streamId   stream id
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetStreamId(rtStream_t stm, int32_t *streamId);

/**
 * @ingroup dvrt_stream
 * @brief inquire max stream count and max task count per stream
 * @param [in] streamType   Stream Type
 * @param [in] MaxStrCount   Max stream count
 * @param [in] MaxTaskCount   max task count per stream
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t *maxStrCount, uint32_t *maxTaskCount);

/**
 * @ingroup dvrt_stream
 * @brief Name a stream
 * @param [in] stm  stream to be named
 * @param [in] name   identification name
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtNameStream(rtStream_t stm, const char_t *name);

/**
 * @ingroup dvrt_stream
 * @brief switch to the corresponding stream according to the contents of the ptr
 * @param [in] ptr  Determine the address where the value of the true and false branches is located
 * @param [in] condition switch condition
 * @param [in] val  switch value
 * @param [in] trueStream  Stream that needs to be activated when the value is non-zero
 * @param [in] stm input stream to init task
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamSwitch(void *ptr, rtCondition_t condition, int64_t val, rtStream_t trueStream,
                                 rtStream_t stm);

/**
 * @brief execute extensible stream switch task
 * @param [in] ptr   pointer of value
 * @param [in] condition   judge condition
 * @param [in] value_ptr   pointer of target value
 * @param [in] true_stream   stream to be activated when value is not zero
 * @param [in] stm   stream id
 * @param [in] dataType   data type of target value
 * @return RT_ERROR_NONE for complete
 */
RTS_API rtError_t rtStreamSwitchEx(void *ptr, rtCondition_t condition, void *valuePtr, rtStream_t trueStream,
                                   rtStream_t stm, rtSwitchDataType_t dataType);

/**
 * @ingroup dvrt_stream
 * @brief Active a stream
 * @param [in] activeStream stream to be activated
 * @param [in] stm input stream to init task
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtStreamActive(rtStream_t activeStream, rtStream_t stm);

/**
 * @brief execute extensible stream case switch task
 * @param [in] ptr   pointer of value
 * @param [in] size  pointer num of value
 * @param [in] valuePtr  pointer of target value, length = size * elementSize
 * @param [in] trueStreamPtr streams to be activated
 * @param [in] elementSize  size of to be activated true streams
 * @param [in] stm input stream to init task
 * @param [in] dataType   data type of target value
 * @return RT_ERROR_NONE for complete
 */
RTS_API rtError_t rtStreamSwitchN(void *ptr, uint32_t size, void *valuePtr, rtStream_t *trueStreamPtr,
                                  uint32_t elementSize, rtStream_t stm, rtSwitchDataType_t dataType);

/*
 * @ingroup dvrt_stream
 * @brief enable debug for dump overflow exception with stream
 * @param [in] addr: ddr address of kernel exception dumpped
 * @param [in] stm: stream handle
 * @param [in] flag: debug flag
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDebugRegisterForStream(rtStream_t stm, uint32_t flag, const void *addr,
                                           uint32_t *streamId, uint32_t *taskId);

/*
 * @ingroup rt_model
 * @brief disable debug for dump overflow exception with stream
 * @param [in] stm: stream handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtDebugUnRegisterForStream(rtStream_t stm);

/*
 * @ingroup dvrt_stream
 * @brief enable or disable stream overflow
 * @param [in] stm: stream handle
 * @param [in] flag: 0:disable others:enable
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtSetStreamOverflowSwitch(rtStream_t stm, uint32_t flags);

/*
 * @ingroup dvrt_stream
 * @brief get whether overflow of the stream is enable or disable
 * @param [in] stm: stream handle
 * @param [out] flag: 0:disable others:enable
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 */
RTS_API rtError_t rtGetStreamOverflowSwitch(rtStream_t stm, uint32_t *flags);

#if defined(__cplusplus)
}
#endif

#endif  // CCE_RUNTIME_STREAM_H
