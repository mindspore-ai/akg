/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AICPU_ASYNC_EVENT_H
#define AICPU_ASYNC_EVENT_H

#include <functional>
#include "aicpu_context.h"

namespace aicpu {
using NotifyFunc = std::function<void(void *param, const uint32_t paramLen)>;
using EventProcessCallBack = std::function<void(void *param)>;

struct AsyncNotifyInfo {
    uint8_t waitType;
    uint32_t waitId;
    uint64_t taskId;
    uint32_t streamId;
    uint32_t retCode;
    aicpu::aicpuContext_t ctx;
};

class AsyncEventManager {
public:
    /**
     * Get the unique object of this class
     */
    static AsyncEventManager &GetInstance();

    /**
     * Register notify callback function
     * @param notify wait notify callback function
     */
    void Register(const NotifyFunc &notify);

    /**
     * Notify wait task
     * @param notifyParam notify param info
     * @param paramLen notifyParam len
     */
    void NotifyWait(void * const notifyParam, const uint32_t paramLen);

    /**
     * Register Event callback function, async op call
     * @param eventId EventId
     * @param subEventId queue id
     * @param cb Event callback function
     * @param times Callback execute times
     * @return whether register success
     */
    bool RegEventCb(const uint32_t eventId, const uint32_t subEventId,
                    const EventProcessCallBack &cb, const int32_t times = 1);

    /**
     * Unregister Event callback function, async op call
     * @param eventID EventId
     * @param subEventId queue id
     */
    void UnregEventCb(const uint32_t eventId, const uint32_t subEventId);

    /**
     * Process event
     * @param eventId EventId
     * @param subEventId queue id
     * @param param event param
     */
    void ProcessEvent(const uint32_t eventId, const uint32_t subEventId, void * const param = nullptr);

private:
    AsyncEventManager();
    ~AsyncEventManager();

    AsyncEventManager(const AsyncEventManager &) = delete;
    AsyncEventManager &operator = (const AsyncEventManager &) = delete;
    AsyncEventManager(AsyncEventManager &&) = delete;
    AsyncEventManager &operator = (AsyncEventManager &&) = delete;

    // wait notify funciton
    NotifyFunc notifyFunc_;
};
}  // namespace aicpu

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Notify wait task
 * @param notifyParam notify info
 * @param paramLen
 */
__attribute__((weak)) void AicpuNotifyWait(void *notifyParam, const uint32_t paramLen);

/**
 * Register Event callback function, async op call
 * @param eventId EventId
 * @param subEventId queue id
 * @param cb Event callback function
 * @return whether register success
 */
__attribute__((weak)) bool AicpuRegEventCb(const uint32_t eventId,
    const uint32_t subEventId, const aicpu::EventProcessCallBack &cb);

/**
 * Register Event callback function, async op call
 * @param eventId EventId
 * @param subEventId queue id
 * @param cb Event callback function
 * @param times Callback execute times
 * @return whether register success
 */
__attribute__((weak)) bool AicpuRegEventCbWithTimes(const uint32_t eventId, const uint32_t subEventId,
    const aicpu::EventProcessCallBack &cb, const int32_t times);

/**
 * Unregister Event callback function, async op call
 * @param eventId EventId
 * @param subEventId queue id
 */
__attribute__((weak)) void AicpuUnregEventCb(const uint32_t eventId, const uint32_t subEventId);
#ifdef __cplusplus
}
#endif
#endif  // AICPU_ASYNC_EVENT_H_
