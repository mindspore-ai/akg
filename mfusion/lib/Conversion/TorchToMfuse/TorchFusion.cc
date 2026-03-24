/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include <functional>
#include <utility>
#include <vector>

#include "mfusion/Conversion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"

namespace mlir {

namespace TorchD = torch::Torch;

namespace {

struct TorchFusionPass : public PassWrapper<TorchFusionPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "torch-fusion"; }
  StringRef getDescription() const final {
    return "Torch dialect fusion pipeline (RoPE etc.) before Convert Torch to Mfuse";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TorchD::TorchDialect>();
  }

  void runOnOperation() override {
    using PassCreator = std::function<std::unique_ptr<Pass>()>;
    std::vector<std::pair<const char *, PassCreator>> passes = {
        {"torch-fuse-rms-norm", []() { return createTorchFuseRmsNormPass(); }},
        {"torch-fuse-rope", []() { return createTorchFuseRoPEPass(); }},
    };

    Operation *op = getOperation();
    MLIRContext &ctx = getContext();

    PassManager pm(&ctx);
    for (const auto &[name, creator] : passes) {
      (void)name;
      pm.addPass(creator());
    }
    if (failed(pm.run(op))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<Pass> createTorchFusionPass() { return std::make_unique<TorchFusionPass>(); }

}  // namespace mlir
