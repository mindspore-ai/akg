/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 * Description:
 * Author: huawei
 * Create: 2019-10-15
 */
#ifndef __AICPUFW_ERRCODE_LOG_H__
#define __AICPUFW_ERRCODE_LOG_H__

#include "aicpu_errcode_def.h"

typedef enum tagAicpuModuleId {
    // to be aicpu
    AICPU_CORE = 0,
    AICPU_OS_ENGINE = 2,
    // AICPU_MAX             = 4,
    // follow to be blas tagDevModuleId
    // blas module id
    // nn module id
} AicpuModuleId_t;

typedef enum AICPUFW_DEV_ERR_CODE {
    /* For aicpu FW ,is isolated to abave ALG err */
    TASK_STATE_SUCCESS = 0,
    AICPU_FW_TASK_STATE_TIMEOUT = 1,
    AICPU_FW_TASK_STATE_EXCEPTION = 2,
    AICPU_FW_HEART_BEAT_BADMOD_EXCEPTION = 3,
    AICPU_FW_HEART_BEAT_DOMEM_ABORT_EXCEPTION = 4,
    AICPU_FW_HEART_BEAT_DOSPPC_ABORT_EXCEPTION = 5,
    AICPU_FW_HEART_BEAT_DOUNDEF_ABORT_EXCEPTION = 6,
    AICPU_FW_TASK_INPUT_PARA_ERROR = 7,

    AICPU_FW_TASK_STATE_MAX = 13,
    /* aicpu core FPSR, Floating-point Status Register */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_IOC = 14, /**< 无效浮点操作 */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_DZC = 15, /**< 除0操作 */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_OFC = 16, /**< 上溢出 */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_UFC = 17, /**< 下溢出 */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_IXC = 18, /**< 运算出现不完全精确的浮点数 */
    AICPU_FW_TASK_STATE_FPSR_FLOAT_IDC = 21, /**< 浮点数输入错误 */

    AICPU_FW_DEV_STATUS_ERR_MAX = 24, /* do nothing */
} ENUM_AICPUFW_DEV_ERR_CODE;

// error code
#define AICPU_FW_ERR_CODE(eid) AICPU_EID2FW(AICPU_CORE, eid)
#define AICPU_OS_ERR_CODE(eid) AICPU_EID2OS(AICPU_OS_ENGINE, eid)

#endif
