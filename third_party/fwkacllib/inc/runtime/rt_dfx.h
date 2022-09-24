/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
 * Description: dfx interface
 */

#ifndef CCE_RUNTIME_RT_DFX_H
#define CCE_RUNTIME_RT_DFX_H

#include "base.h"

#if defined(__cplusplus)
extern "C" {
#endif

// max task tag buffer is 1024(include '\0')
#define TASK_TAG_MAX_LEN    1024U

/**
 * @brief set task tag.
 * once set is only use by one task and thread local.
 * attention:
 *  1. it's used for dump current task in active stream now.
 *  2. it must be called be for task submit and will be invalid after task submit.
 * @param [in] taskTag  task tag, usually it's can be node name or task name.
 *                      must end with '\0' and max len is TASK_TAG_MAX_LEN.
 * @return RT_ERROR_NONE for ok
 * @return other failed
 */
RTS_API rtError_t rtSetTaskTag(const char_t *taskTag);

/**
 * @brief set aicpu device attribute.
 * it is used for aicpu device to be aware of enviroment config
 * @param [in] key  attrubute key.
 * @param [in] val  attrubute value.
 * @return RT_ERROR_NONE for ok
 * @return other failed
 */
RTS_API rtError_t rtSetAicpuAttr(const char_t *key, const char_t *val);

#if defined(__cplusplus)
}
#endif
#endif // CCE_RUNTIME_RT_DFX_H
