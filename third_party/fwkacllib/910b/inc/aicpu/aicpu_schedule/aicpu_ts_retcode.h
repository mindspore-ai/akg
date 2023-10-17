/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef AICPU_TS_RETCODE_H
#define AICPU_TS_RETCODE_H

namespace aicpu {
    // ret code for aicpu to ts
    enum class AICPU_TS_RETCODE {
        // ret code start with 0x07, previous has been used.
        TASK_STATE_AICPU_PHASE2_TIMEOUT = 0x07,
    };
} // namespace aicpu

#endif // AICPU_TS_RETCODE_H