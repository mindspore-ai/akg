/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: the definition of stars
 */

#ifndef CCE_RUNTIME_RT_STARS_H
#define CCE_RUNTIME_RT_STARS_H

#include "base.h"
#include "rt_stars_define.h"
#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup rt_stars
 * @brief launch stars task.
 * used for send star sqe directly.
 * @param [in] taskSqe     stars task sqe
 * @param [in] sqeLen      stars task sqe length
 * @param [in] stm      associated stream
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtStarsTaskLaunch(const void *taskSqe, uint32_t sqeLen, rtStream_t stm);


/**
 * @ingroup rt_stars
 * @brief create cdq instance.
 * @param [in] batchNum     batch number
 * @param [in] batchSize    batch size
 * @param [in] queName      cdq name
 * @return RT_ERROR_NONE for ok, ACL_ERROR_RT_NO_CDQ_RESOURCE for no cdq resources
 */
RTS_API rtError_t rtCdqCreate(uint32_t batchNum, uint32_t batchSize, const char_t *queName);

/**
 * @ingroup rt_stars
 * @brief destroy cdq instance.
 * @param [in] queName      cdq name
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtCdqDestroy(const char_t *queName);

/**
 * @ingroup rt_stars
 * @brief get free batch in the queue.
 * @param [in] queName      cdq name
 * @param [in] timeout      batch size
 * @param [out] batchId     batch index
 * @return RT_ERROR_NONE for ok, ACL_ERROR_RT_WAIT_TIMEOUT for timeout
 */
RTS_API rtError_t rtCdqAllocBatch(const char_t *queName, int32_t timeout, uint32_t *batchId);

/**
 * @ingroup rt_stars
 * @brief launch a write_cdqm task on the stream.
 * When the task is executed, the data information will be inserted into the cdqe index position of the queue.
 * @param [in] queName      cdq name
 * @param [in] cdqeIndex    cdqe index
 * @param [in] data         cdqe infomation
 * @param [in] dataSize     data size
 * @param [in] stm       launch task on the stream
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtCdqEnQueue(const char_t *queName, uint32_t cdqeIndex, void *data, uint32_t dataSize,
    rtStream_t stm);

/**
 * @ingroup rt_stars
 * @brief launch a write_cdqm task on the stream.
 * When the task is executed, the data information will be inserted into the cdqe index position of the queue.
 * @param [in] queName      cdq name
 * @param [in] cdqeIndex    cdqe index
 * @param [in] data         cdqe infomation
 * @param [in] dataSize     data size
 * @param [in] stm       launch task on the stream
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtCdqEnQueuePtrMode(const char_t *queName, uint32_t cdqeIndex, const void *ptrAddr,
    rtStream_t stm);

/**
 * @ingroup rt_stars
 * @brief launch common cmo task on the stream.
 * @param [in] taskInfo     cmo task info
 * @param [in] stm          launch task on the stream
 * @param [in] flag         flag
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtCmoTaskLaunch(rtCmoTaskInfo_t *taskInfo, rtStream_t stm, uint32_t flag);

/**
 * @ingroup rt_stars
 * @brief launch barrier cmo task on the stream.
 * @param [in] taskInfo     barrier task info
 * @param [in] stm          launch task on the stream
 * @param [in] flag         flag
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtBarrierTaskLaunch(rtBarrierTaskInfo_t *taskInfo, rtStream_t stm, uint32_t flag);

/**
 * @ingroup rt_stars
 * @brief dvpp group handle.
 */
typedef void *rtDvppGrp_t;

typedef struct tagDvppGrpRptInfo {
    uint32_t deviceId;
    uint32_t streamId;
    uint32_t taskId;
    uint8_t sqeType;
    uint8_t cqeErrorCode;
    uint8_t reserve[2];
    uint32_t accErrorCode;
} rtDvppGrpRptInfo_t;

typedef void (*rtDvppGrpCallback)(rtDvppGrpRptInfo_t *rptInfo);

/**
 * @ingroup rt_stars
 * @brief create dvpp group.
 * @param [in] flags     group flag, reserved parameter
 * @param [out] grp      group handle
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtDvppGroupCreate(rtDvppGrp_t *grp, uint32_t flags);

/**
 * @ingroup rt_stars
 * @brief destroy dvpp group.
 * @param [in] grp      group handle
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtDvppGroupDestory(rtDvppGrp_t grp);

/**
 * @ingroup rt_stars
 * @brief create stream with grp handle
 * @param [in|out] stm   created stream
 * @param [in] priority   stream priority
 * @param [in] flags  stream op flags
 * @param [in] grp    grp handle
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtStreamCreateByGrp(rtStream_t *stm, int32_t priority, uint32_t flags, rtDvppGrp_t grp);

/**
 * @ingroup rt_stars
 * @brief wait report by grp
 * @param [in] grp              group handle
 * @param [in] callBackFunc     callback
 * @param [in] timeout          wait timeout config, ms, -1: wait forever
 * @return RT_ERROR_NONE for ok, others failed
 */
RTS_API rtError_t rtDvppWaitGroupReport(rtDvppGrp_t grp, rtDvppGrpCallback callBackFunc, int32_t timeout);

#if defined(__cplusplus)
}
#endif
#endif  // CCE_RUNTIME_RT_STARS_H
