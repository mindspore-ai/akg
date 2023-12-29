/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 */

#ifndef AICPU_ERROR_LOG_API_H
#define AICPU_ERROR_LOG_API_H

#include <queue>
#include <string>
#include "aicpu/common/type_def.h"

namespace aicpu {
    void InitAicpuErrorLog();

    void RestoreErrorLog(const char_t * const funcName, const char_t * const filePath, const int32_t lineNumber,
                         const uint64_t tid, const char_t * const format, ...);

    __attribute__((weak)) std::queue<std::string> *GetErrorLog();
}

#endif