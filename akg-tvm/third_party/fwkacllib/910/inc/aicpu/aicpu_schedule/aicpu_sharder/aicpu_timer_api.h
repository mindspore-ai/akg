/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef AICPU_TIMER_API_H
#define AICPU_TIMER_API_H

#include <functional>

namespace aicpu {
    using TimeoutCallback = std::function<void()>;
    using TimerHandle = uint64_t;

    /**
     * Start the timer
     * @param [out]timerHandle: The timer handle
     * @param [in]TimeoutCallback: The struct for timer config
     * @param [in]timeInS: Timeout in seconds
     * @return bool success of failed
     */
    bool __attribute__((weak)) StartTimer(TimerHandle &timerHandle, const TimeoutCallback &callBack,
                                          const uint32_t timeInS);

    /**
     * Stop the timer
     * @param [in]timerHandle: The struct for timer config
     * @return void
     */
    void __attribute__((weak)) StopTimer(const TimerHandle timerHandle);
} // namespace aicpu
#endif // AICPU_SHARDER_TIMER_API_H
