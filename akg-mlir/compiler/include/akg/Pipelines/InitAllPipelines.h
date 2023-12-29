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

#ifndef COMPILER_INCLUDE_AKG_PIPELINES_INITALLPIPELINES_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_INITALLPIPELINES_H_

#include "akg/Pipelines/CommonOpt.h"
#include "akg/Pipelines/CPUOpt.h"
#include "akg/Pipelines/GPUPipelines/GPUOpt.h"
#include "akg/Pipelines/GPUPipelines/GPUDynOpt.h"
#include "akg/Pipelines/GPUPipelines/GPUCodegen.h"

using namespace mlir;

namespace mlir {
inline void registerAllPiplines() {
  registerCpuOptPipeline();
  registerSpliterOptPipeline();
  registerGpuOptPipeline();
  registerGpuDynOptPipeline();
  registerGpuCodegenPipeline();
}
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_INITALLPIPELINES_H_
