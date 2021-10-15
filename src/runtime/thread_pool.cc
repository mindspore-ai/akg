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

#include <algorithm>
#include <exception>
#if AKG_USE_OPENMP
#include <omp.h>
#endif
#include <dmlc/logging.h>
#include "thread_pool.h"

namespace mindspore {
namespace common {

size_t MaxThreadNumber() {
  size_t process_core_num = std::thread::hardware_concurrency();
#if defined(_M_X64) || defined(__x86_64__)
    process_core_num /= 2;  // ignore hyper-threading
#endif

  if (process_core_num < 1) {
    process_core_num = 1;
  }
  size_t thread_num;
#if ENABLE_D || ENABLE_GPU
  thread_num = process_core_num / kDeviceNum;
#else
  if (const char* val = getenv("AKG_NUM_THREADS")) {
    thread_num = std::min((size_t)atoi(val), process_core_num);
  } else {
    thread_num = process_core_num;
  }
#endif
  if (thread_num < 1) {
    thread_num = 1;
  }
  return thread_num;
}

ThreadPool::ThreadPool() {
  max_thread_num_ = MaxThreadNumber();
}

void ThreadPool::SyncRunLoop() {
  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(task_mutex_);
      task_cond_var_.wait(lock, [this] { return !task_queue_.empty() || exit_run_; });
      if (exit_run_) {
        return;
      }
      task = task_queue_.front();
      task_queue_.pop();
    }
    try {
      task();
    } catch (std::exception &e) {
      LOG(ERROR) << "Have exception in run loop of thread";
    }
    {
      std::unique_lock<std::mutex> task_lock(task_mutex_);
      task_finished_count_ = task_finished_count_ + 1;
    }
    finished_cond_var_.notify_one();
  }
}

bool ThreadPool::SyncRun(const std::vector<Task> &tasks) {
  if (tasks.size() == 1) {
    auto ret = tasks[0]();
    return ret;
  }
  std::unique_lock<std::mutex> lock(pool_mtx_);
  exit_run_ = false;
  size_t task_num = tasks.size();
  size_t thread_num = sync_run_threads_.size();
  if (thread_num < max_thread_num_ && thread_num < task_num) {
    auto new_thread_num = max_thread_num_;
    if (task_num < max_thread_num_) {
      new_thread_num = task_num;
    }
    for (size_t i = thread_num; i < new_thread_num; ++i) {
      sync_run_threads_.emplace_back(std::thread(&ThreadPool::SyncRunLoop, this));
    }
  }

  for (auto &task : tasks) {
    std::lock_guard<std::mutex> task_lock(task_mutex_);
    task_queue_.push(task);
    task_cond_var_.notify_one();
  }
  {
    std::unique_lock<std::mutex> task_lock(task_mutex_);
    finished_cond_var_.wait(task_lock, [this, task_num] { return task_num == task_finished_count_; });
    task_finished_count_ = 0;
  }
  return SUCCESS;
}

ThreadPool &ThreadPool::GetInstance() {
  static ThreadPool instance{};
  return instance;
}

void ThreadPool::ClearThreadPool() {
  std::lock_guard<std::mutex> sync_run_lock(pool_mtx_);
  if (exit_run_) {
    return;
  }
  exit_run_ = true;
  task_cond_var_.notify_all();
  for (auto &it : sync_run_threads_) {
    if (it.joinable()) {
      it.join();
    }
  }
  sync_run_threads_.clear();
}

ThreadPool::~ThreadPool() {
  try {
    ClearThreadPool();
  } catch (...) {
  }
}
}  // namespace common
}  // namespace mindspore

#ifdef __cplusplus
extern "C" {
#endif


/*!
 * \brief The callback function to execute a parallel lambda
 *          with akg runtime.
 * \param task_id the task id of the function.
 * \param penv The parallel environment backs the execution.
 * \param cdata The supporting closure data.
 */
typedef int (*FAKGParallelLambda)(
    int task_id, int num_task, void* cdata);

int AKGBackendParallelLaunch(
    FAKGParallelLambda flambda,
    void* cdata,
    int num_task) {
#if !AKG_USE_OPENMP
  auto& thread_pool = mindspore::common::ThreadPool::GetInstance();
  std::vector<std::function<int()>> tasks;
  int max_task_num = static_cast<int>(thread_pool.GetSyncRunThreadNum());
  max_task_num = std::min(num_task, max_task_num);
  for (int i = 0; i < max_task_num; ++i) {
    auto block = [&, i]() {
        flambda(i, max_task_num, cdata);
        return 0;
    };
    tasks.emplace_back(block);
  }
  thread_pool.SyncRun(tasks);
#else
  int num_workers = std::min(static_cast<int>(mindspore::common::MaxThreadNumber()), num_task);
  omp_set_num_threads(num_workers);
  #pragma omp parallel num_threads(num_workers)
  {
    flambda(omp_get_thread_num(), num_workers, cdata);
  }
#endif
  return 0;
}

#ifdef __cplusplus
}
#endif
