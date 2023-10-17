/**
 * @file aoe_error_manager.h
 *
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 */

#ifndef AOE_ERROR_MANAGER_H
#define AOE_ERROR_MANAGER_H

#include <string>
#include <map>
#include "common/util/error_manager/error_manager.h"
#include "aoe_types.h"

namespace Aoe {
using ErrorContext = error_message::Context;

class AoeErrorManager {
public:
    static AoeErrorManager &GetInstance();
    AoeStatus AoeInitErrorManager(const std::string &path);
    AoeStatus AoeReportInterErrorMessage(const std::string &errorCode, const std::string &errorMsg) const;
    AoeStatus AoeReportErrorMessage(const std::string &errorCode,
        const std::map<std::string, std::string> &argsMap) const;
    std::string AoeGetErrorMessage() const;
    AoeStatus AoeOutputErrorMessage(int32_t handle) const;
    std::string AoeGetWarningMessage() const;
    AoeStatus AoeOutputMessage(int32_t handle) const;
    ErrorContext &AoeGetErrorContext() const;
    void AoeSetErrorContext(const ErrorContext &context) const;
    void AoeGenWorkStreamIdDefault() const;
private:
    AoeErrorManager() {}
    ~AoeErrorManager() {}
    AoeErrorManager(const AoeErrorManager &) = delete;
    AoeErrorManager(AoeErrorManager &&) = delete;
    AoeErrorManager &operator=(const AoeErrorManager &) = delete;
    AoeErrorManager &operator=(AoeErrorManager &&) = delete;
    bool isInit_{false};
};
}

#endif
