/**
 * Copyright (c) Hisilicon Technologies Co., Ltd. 2019-2021. All rights reserved.
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

#ifndef TSD_EZCOM_H
#define TSD_EZCOM_H

#include <string>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

namespace tsd {
enum AlarmLevel { CRITICAL = 1, MAJOR, MINOR, SUGGESTION };
enum AlarmReason {
    PROC_EXIT_ABNORMAL = 1,
    PROC_START_FAIL,
    PROC_EXIT_TSD_CLOSE,
    PROC_EXIT_HDC_DISCONNECT,
    PROC_EXIT_PID_ABNORMAL,
    PROC_EXIT_TASK_TIMEOUT
};
enum AlarmModule {
    QS = 1,
    AICPU_SD,
    HCCP,
    CUSTOM_AICPU_SD,
    ALARM_MODULE_MAX,
};
struct AlarmMessage {
    std::string pidAppName;
    AlarmModule module;
    AlarmLevel level;
    uint32_t procId;
    int32_t subPid;
    AlarmReason reason;
    uint32_t addtionalSize;
};

enum AlarmMessageType {
    PROCESS_NAME = 1,
    ALARM_MESSAGE = 2,
    TSD_STARTED = 3,
};

struct Tsd2dmReqMsg {
    std::string pidAppName;
    AlarmMessageType msgType;
    AlarmMessage alarmMessage;
    int32_t subPid;
    uint32_t rosNodePid;
    int32_t pgid;
};

using TsdCallbackFuncType = void (*)(struct Tsd2dmReqMsg *req);

/**
* @ingroup RegisterTsdEzcomCallBack
* @brief used for datamaster to regist ezcom message handler
*
* @param NA
* @param callBack [IN] ezcom message handler
* @retval 0 Success
* @retval OtherValues 0 Fail
*
* @par Dependency
* @li libtsd_eventclient.so: Library to which the interface belongs.
* @li tsdclient.h: Header file where the interface declaration is located.
*/
int32_t RegisterTsdEzcomCallBack(const TsdCallbackFuncType callBack);

/**
* @ingroup UnregisterTsdEzcomCallBack
* @brief used for datamaster to unregist ezcom message handler
*
* @param NA
* @par Dependency
* @li tsdclient.h: Header file where the interface declaration is located.
*/
void UnregisterTsdEzcomCallBack();
}  // namespace tsd
#ifdef __cplusplus
}
#endif  // __cplusplus
#endif  // TSD_EZCOM_H
