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

#ifndef __CCE_RUNTIME_CONTEXT_H__
#define __CCE_RUNTIME_CONTEXT_H__

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
 * @param [out] ctx   created context
 * @param [in] flags   context creation flag. set to 0.
 * @param [in] device    device to create context on
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxCreate(rtContext_t *ctx, uint32_t flags, int32_t device);

/**
 * @ingroup rt_context
 * @brief create context and associates it with the calling thread
 * @param [out] ctx   created context
 * @param [in] flags   context creation flag. set to 0.
 * @param [in] device    device to create context on
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxCreateEx(rtContext_t *ctx, uint32_t flags, int32_t device);

/**
 * @ingroup rt_context
 * @brief destroy context instance
 * @param [in] ctx   context to destroy
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxDestroy(rtContext_t ctx);

/**
 * @ingroup rt_context
 * @brief destroy context instance
 * @param [in] ctx   context to destroy
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxDestroyEx(rtContext_t ctx);

/**
 * @ingroup rt_context
 * @brief binds context to the calling CPU thread.
 * @param [in] ctx   context to bind. if NULL, unbind current context.
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxSetCurrent(rtContext_t ctx);

/**
 * @ingroup rt_context
 * @brief block for a context's tasks to complete
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxSynchronize(void);

/**
 * @ingroup rt_context
 * @brief returns the context bound to the calling CPU thread.
 * @param [out] ctx   returned context
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxGetCurrent(rtContext_t *ctx);

/**
 * @ingroup rt_context
 * @brief returns the primary context of device.
 * @param [out] ctx   returned context
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtGetPriCtxByDeviceId(int32_t device, rtContext_t *ctx);

/**
 * @ingroup rt_context
 * @brief returns the device ID for the current context
 * @param [out] device   returned device id
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtCtxGetDevice(int32_t *device);

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
RTS_API rtError_t rtGetGroupInfo(int32_t groupId, rtGroupInfo_t *groupInfo, uint32_t count);

/**
 * @ingroup
 * @brief get group count
 * @param [in] groupid count
 * @return RT_ERROR_NONE for ok, errno for failed
 */
RTS_API rtError_t rtGetGroupCount(uint32_t *count);

/**
 * @ingroup rt_context
 * @brief set context INF mode
 * @param [in] mode
 * @return RT_ERROR_NONE for ok
 */
RTS_API rtError_t rtSetCtxINFMode(bool mode);

#if defined(__cplusplus)
}
#endif


#endif  // __CCE_RUNTIME_CONTEXT_H__
