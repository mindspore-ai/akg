/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <omp.h>
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>
#include "akg/ExecutionEngine/AKGParallellLaunchRuntime.h"

using namespace mlir::runtime;

extern "C" int AKGBackendParallelLaunch(FAKGParallelLambda flambda, void *cData, void *externData, int taskNums) {
  auto nThreads = omp_get_max_threads();
  int numWorkers = std::min(taskNums, nThreads);
#pragma omp parallel num_threads(numWorkers)
  { flambda(omp_get_thread_num(), numWorkers, cData, externData); }
  return 0;
}
