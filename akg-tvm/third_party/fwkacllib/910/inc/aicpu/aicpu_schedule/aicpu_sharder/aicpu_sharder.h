/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
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

#ifndef AICPU_SHARDER_H
#define AICPU_SHARDER_H

#include <mutex>
#include <atomic>
#include <functional>
#include <vector>

namespace aicpu {
using Closure = std::function<void()>;
using ClosureBool = std::function<bool()>;
using RunnerBool = std::function<bool(Closure, bool, uint32_t)>;
using SharderWork = std::function<void(int64_t, int64_t)>;
using EnqueueBool = std::function<bool(Closure, uint32_t)>;
using SubmitEvent = std::function<uint32_t(uint32_t, uint32_t)>;
using DequeueBool = std::function<bool(uint32_t)>;

class SharderNonBlock {
public:
    /**
     * Get the unique object of this class
     */
    static SharderNonBlock &GetInstance();

    /**
     * Register schedule callback function, doTask function and cpu core number
     * called by compute process
     * @param schedule Schedule callback function
     * @param doTask Callback function for itself schedule
     * @param cpuCoreNum aicpu core number
     * @param splitTaskEnqueue split task enqueue
     * @param submitSplitEvent submit split event
     */
    void Register(const RunnerBool &schedule, const ClosureBool &doTask, const uint32_t cpuCoreNum,
                  const EnqueueBool &splitTaskEnqueue, const SubmitEvent &submitSplitEvent,
                  const DequeueBool &dequeueDoClosure);

    /**
     * Shards the "total" unit of work refer "perUintSize"
     * @param total Total unit of work
     * @param perUnitSize Minimum shard unit
     * @param work should be a callable taking (int64, int64) arguments.
                   work(start, limit) computes the work units from [start, limit),
                   i.e., [start, limit) is a shard.
     */
    void ParallelFor(const int64_t total, const int64_t perUnitSize, const SharderWork &work);

    /**
     * Shards the unit of work refer for hash
     * @param total, Total unit of work
     * @param cpuNums Number of cpu cores
     * @param work should be a callable taking (int64, int64) arguments.
                   work(cur, cpuNums) computes the work units with input hash with (cpuNums-1) equals cur,
                   i.e. specially used by parallel unique op
     */
    void ParallelForHash(const int64_t total, const int64_t cpuNums, const SharderWork &work);

    /**
     * Schedule a task use schedule function registered by compute process,
     * note that the task will actually executed asynchronously
     * @param closure Closure function with nothrow
     */
    void Schedule(const Closure &aicpuClosure);

    /**
     * Get CPU number
     * @param None
     * @return CPU number
     */
    uint32_t GetCPUNum();

private:
    SharderNonBlock();
    ~SharderNonBlock() = default;

    SharderNonBlock(const SharderNonBlock &) = delete;
    SharderNonBlock &operator = (const SharderNonBlock &) = delete;
    SharderNonBlock(SharderNonBlock &&) = delete;
    SharderNonBlock &operator = (SharderNonBlock &&) = delete;

    /**
     * Closure function enqueue
     * @param closure Closure function can be called
     * @param isSplitKernel whether is split kernel
     * @return whether enqueue of closure success
     */
    bool Enqueue(const Closure &aicpuClosure, const uint32_t parallelId, const bool isSplitKernel = false) const;
    bool SplitTaskEnqueue(const Closure &aicpuClosure, const uint32_t parallelId) const;
    uint32_t SubmitSplitEvent(uint32_t submitNum, const uint32_t parallelId) const;
    bool DequeueDoClosure(const uint32_t parallelId) const;
    bool IsAicpuRunMdcMode();
    void DoTaskItself(const uint32_t parallelId, std::atomic<int64_t> &cpuNumCounter);

    /**
     * Calculate how many times, which ceiled, "x" is "base".
     * i.e., x is 1, base is 2, this function will return 1
     * @param x An integral
     * @param base An integral as base when cal multiple
     * @return ceiled multiple
     */
    inline int64_t CeilMultiple(const int64_t x, const int64_t base) const;

    /**
     * Shards the "total" unit of work refer "perUintSize"
     * @param total Total unit of work
     * @param shardNum parralle number
     * @param blockSize Minimum shard unit
     * @param work should be a callable taking (int64, int64) arguments.
                   work(start, limit) computes the work units from [start, limit),
                   i.e., [start, limit) is a shard.
     */
    void ExecuteParallelFor(const int64_t total, const int64_t shardNum,
                            const int64_t blockSize, const SharderWork &work, const uint32_t parallelId);

    void ExecuteParallelForHash(const int64_t total, const int64_t cpuNums, const SharderWork &work,
                                const uint32_t parallelId);

private:
    RunnerBool schedule_; // enqueue runner
    ClosureBool doTask_;  // a callback, do task from task queue
    uint32_t cpuCoreNum_; // aicpu core number
    std::atomic<uint32_t> parallelId_; // the id for parallel run kernel
    std::mutex parallelIdMutex_;
    EnqueueBool splitTaskEnqueue_;
    SubmitEvent submitSplitEvent_;
    DequeueBool dequeueDoClosure_;
};                        // SharderNonBlock
} // namespace aicpu

extern "C" {
/**
 * Shards the "total" unit of work refer "perUintSize"
 * @param total Total unit of work
 * @param perUnitSize Minimum shard unit
 * @param work should be a callable taking (int64, int64) arguments.
                 work(start, limit) computes the work units from [start, limit),
                i.e., [start, limit) is a shard.
 */
__attribute__((visibility("default"))) void ParallelFor(int64_t total, int64_t perUnitSize,
    const aicpu::SharderWork &work);

/**
 * Get CPU number
 * @param None
 * @return CPU number
 */
__attribute__((visibility("default"))) uint32_t GetCPUNum();
}

#endif // AICPU_SHARDER_H_
