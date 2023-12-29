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

#ifndef COMPILER_INCLUDE_AKG_PIPELINES_COMMONOPT_H_
#define COMPILER_INCLUDE_AKG_PIPELINES_COMMONOPT_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

namespace mlir {
struct SpliterOptPipelineOptions
    : public PassPipelineOptions<SpliterOptPipelineOptions> {
  Option<std::string> dumpDir{
      *this, "dump-dir",
      llvm::cl::desc("An optional attribute to speicify the directory that spliter dump sub-graph into."),
      llvm::cl::init("")};
};

void createSpliterOptPipeline(OpPassManager &pm,
                              const SpliterOptPipelineOptions &options);

inline void registerSpliterOptPipeline() {
  PassPipelineRegistration<SpliterOptPipelineOptions>(
      "spliter-opt", "Spliter Opt Pipeline", createSpliterOptPipeline);
}
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_PIPELINES_COMMONOPT_H_
