/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
 * Description:
 * Author: huawei
 * Create: 2019-10-15
 */
#ifndef __AICPU_ERRCODE_DEF_H__
#define __AICPU_ERRCODE_DEF_H__

#include "npu_error_define.h"

/* bit 31-bit30 to be hiai aicpu local */
/* bit 29 -bit28 to be hiai aicpu code type */
/* bit 27 -bit25 to be hiai aicpu  error level */
/* bit 24 -bit17 to be hiai mod */
/* bit16 -bit7 to be hiai aicpu submodule id */
#define HIAI_AICPU_DEVMODULEID_MASK 0x0001FF80
#define SHIFT_SUBMODE_MASK 7
#define HIAI_AICPU_MID_MASK 0x3FF
/* bit6 -bit0 to be hiai aicpu module error id */
#define HIAI_AICPU_ERROR_ID_MASK 0x0000007F
#define SHIFT_ERROR_ID_MASK 0

#define HIAI_AICPU_DEVMOD_CFG(mid) \
    (HIAI_AICPU_DEVMODULEID_MASK & ((((unsigned int)mid) & HIAI_AICPU_MID_MASK) << SHIFT_SUBMODE_MASK))

#define HIAI_AICPU_ERRID_CFG(eid) (HIAI_AICPU_ERROR_ID_MASK & ((unsigned int)eid))

#define AICPU_EID(mid, eid)                                                                                        \
    (int)(HIAI_NPU_ERR_CODE_HEAD(HIAI_DEVICE, ERROR_CODE, NORMAL_LEVEL, HIAI_AICPU) | HIAI_AICPU_DEVMOD_CFG(mid) | \
          HIAI_AICPU_ERRID_CFG(eid))

#define AICPU_EID_EX(code_type, err_lvl, aicpu2ex, mid, eid)                                          \
    (HIAI_NPU_ERR_CODE_HEAD(HIAI_DEVICE, code_type, err_lvl, aicpu2ex) | HIAI_AICPU_DEVMOD_CFG(mid) | \
     HIAI_AICPU_ERRID_CFG(eid))

#define AICPU_EID2FW(mid, eid)                                                                                \
    (HIAI_NPU_ERR_CODE_HEAD(HIAI_DEVICE, ERROR_CODE, NORMAL_LEVEL, HIAI_AICPU) | HIAI_AICPU_DEVMOD_CFG(mid) | \
     HIAI_AICPU_ERRID_CFG(eid))
#define AICPU_EID2OS(mid, eid) AICPU_EID2FW(mid, eid)

#endif
