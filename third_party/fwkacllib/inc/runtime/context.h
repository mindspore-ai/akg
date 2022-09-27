/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: context.h
 * Create: 2020-01-01
 */

#ifndef CCE_RUNTIME_CONTEXT_H
#define CCE_RUNTIME_CONTEXT_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

/**
 * @ingroup rt_context
 * @brief runtime context handle.
 */
typedef void *rtContext_t;

typedef enum tagDryRunFlag {
    RT_DRYRUN_FLAG_FALSE = 0,
    RT_DRYRUN_FLAG_TRUE = 1,
} rtDryRunFlag_t;

typedef enum tagCtxMode {
    RT_CTX_NORMAL_MODE = 0,
    RT_CTX_GEN_MODE = 1,
} rtCtxMode_t;

typedef struct tagRtGroupInfo {
    int32_t groupId;
    uint32_t flag;
    uint32_t aicoreNum;
    uint32_t aicpuNum;
    uint32_t aivectorNum;
    uint32_t sdmaNum;
    uint32_t activeStreamNum;
    void *extrPtr;
} rtGroupInfo_t;

/**
 * @ingroup rt_context
 * @brief create context and associates it with the calling thread
 * @param [out] createCtx   created context
 * @param [in] flags   context creation flag. set to 0.
 * @param [in] devId    device to create context on
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxCreate(rtContext_t *createCtx, uint32_t flags, int32_t devId);

/**
 * @ingroup rt_context
 * @brief create context and associates it with the calling thread
 * @param [out] createCtx   created context
 * @param [in] flags   context creation flag. set to 0.
 * @param [in] devId    device to create context on
 * @param [in] deviceMode    the device mode
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxCreateV2(rtContext_t *createCtx, uint32_t flags, int32_t devId, rtDeviceMode deviceMode);

/**
 * @ingroup rt_context
 * @brief create context and associates it with the calling thread
 * @param [out] createCtx   created context
 * @param [in] flags   context creation flag. set to 0.
 * @param [in] devId    device to create context on
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxCreateEx(rtContext_t *createCtx, uint32_t flags, int32_t devId);

/**
 * @ingroup rt_context
 * @brief destroy context instance
 * @param [in] destroyCtx   context to destroy
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxDestroy(rtContext_t destroyCtx);

/**
 * @ingroup rt_context
 * @brief destroy context instance
 * @param [in] destroyCtx   context to destroy
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxDestroyEx(rtContext_t destroyCtx);

/**
 * @ingroup rt_context
 * @brief binds context to the calling CPU thread.
 * @param [in] currentCtx   context to bind. if NULL, unbind current context.
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxSetCurrent(rtContext_t currentCtx);

/**
 * @ingroup rt_context
 * @brief block for a context's tasks to complete
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxSynchronize(void);

/**
 * @ingroup rt_context
 * @brief returns the context bound to the calling CPU thread.
 * @param [out] currentCtx   returned context
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxGetCurrent(rtContext_t *currentCtx);

/**
 * @ingroup rt_context
 * @brief returns the primary context of device.
 * @param [out] primaryCtx   returned context
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtGetPriCtxByDeviceId(int32_t devId, rtContext_t *primaryCtx);

/**
 * @ingroup rt_context
 * @brief returns the device ID for the current context
 * @param [out] devId   returned device id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxGetDevice(int32_t *devId);

/**
 * @ingroup
 * @brief set group id
 * @param [in] groupid
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtSetGroup(int32_t groupId);

/**
 * @ingroup
 * @brief get group info
 * @param [in] groupid count
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtGetGroupInfo(int32_t groupId, rtGroupInfo_t *groupInfo, uint32_t cnt);

/**
 * @ingroup
 * @brief get group count
 * @param [in] groupid count
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtGetGroupCount(uint32_t *cnt);

/**
 * @ingroup rt_context
 * @brief set context INF mode
 * @param [in] infMode
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetCtxINFMode(bool infMode);

#if defined(__cplusplus)
}
#endif


#endif  // CCE_RUNTIME_CONTEXT_H
