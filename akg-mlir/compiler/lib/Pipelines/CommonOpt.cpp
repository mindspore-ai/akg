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

#include "akg/Pipelines/CommonOpt.h"

#include <string>

#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
void createSpliterOptPipelineImpl(OpPassManager &pm, const SpliterOptPipelineOptions &options) {
  OpPassManager &nestedFunctionPM = pm.nest<func::FuncOp>();
  nestedFunctionPM.addPass(mlir::createInferSymbolicShapesPass());
}
}  // namespace

namespace mlir {
void createSpliterOptPipeline(OpPassManager &pm, const SpliterOptPipelineOptions &options) {
  createSpliterOptPipelineImpl(pm, options);
}
}  // namespace mlir

