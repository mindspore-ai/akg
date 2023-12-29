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
#ifndef COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGPARALLELLLAUNCHRUNTIME_H_
#define COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGPARALLELLLAUNCHRUNTIME_H_
#include <stdint.h>
#include <cstddef>

#ifdef mlir_akgParallelLaunch_runtime_EXPORTS
// We are building this library
#define MLIR_AKGPARALLELLAUNCHRUNTIME_DEFINE_FUNCTIONS
#endif  // mlir_akgParallelLaunch_runtime_EXPORTS

namespace mlir {
namespace runtime {
typedef int (*FAKGParallelLambda)(int taskId, int taskNums, void *cData, void *externData);
extern "C" int AKGBackendParallelLaunch(FAKGParallelLambda flambda, void *cData, void *externData, int taskNums);
}  // namespace runtime
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_EXECUTIONENGINE_AKGPARALLELLLAUNCHRUNTIME_H_

