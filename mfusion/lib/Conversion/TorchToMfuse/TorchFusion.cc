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

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "mfusion/Conversion/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"

namespace mlir {

namespace TorchD = torch::Torch;

namespace {

FailureOr<llvm::SmallVector<int64_t, 4>> getConstantReductionDims(Value dimValue, int64_t inputRank, Location loc,
                                                                  PatternRewriter &rewriter) {
  bool reduceAll = isa<TorchD::NoneType>(dimValue.getType());
  llvm::SmallVector<int64_t, 4> dims;
  if (!reduceAll) {
    llvm::SmallVector<Value, 4> dimValues;
    if (!TorchD::getListConstructElements(dimValue, dimValues)) {
      return rewriter.notifyMatchFailure(loc, "dim must come from list construct");
    }
    if (dimValues.empty()) {
      reduceAll = true;
    } else {
      dims.reserve(dimValues.size());
      for (Value dimVal : dimValues) {
        int64_t dim = 0;
        if (!matchPattern(dimVal, TorchD::m_TorchConstantInt(&dim))) {
          return rewriter.notifyMatchFailure(loc, "dim list must be constant ints");
        }
        dim = TorchD::toPositiveDim(dim, inputRank);
        if (!TorchD::isValidDim(dim, inputRank)) {
          return rewriter.notifyMatchFailure(loc, "dim out of range");
        }
        if (std::find(dims.begin(), dims.end(), dim) != dims.end()) {
          return rewriter.notifyMatchFailure(loc, "duplicate reduction dims are not supported");
        }
        dims.push_back(dim);
      }
    }
  }

  if (reduceAll) {
    dims.resize(inputRank);
    std::iota(dims.begin(), dims.end(), 0);
  }
  std::sort(dims.begin(), dims.end());
  return dims;
}

bool reductionDimsEqual(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

bool keepdimValuesEqual(TorchD::AtenMeanDimOp meanOp, TorchD::AtenVarCorrectionOp varOp) {
  bool meanKeepdim = false;
  bool varKeepdim = false;
  if (!matchPattern(meanOp.getKeepdim(), TorchD::m_TorchConstantBool(&meanKeepdim))) {
    return false;
  }
  if (!matchPattern(varOp.getKeepdim(), TorchD::m_TorchConstantBool(&varKeepdim))) {
    return false;
  }
  return meanKeepdim == varKeepdim;
}

/// Merge torch.aten.mean.dim + torch.aten.var.correction on the same input into var_mean.correction.
struct TorchMergeMeanVarPattern : public OpRewritePattern<TorchD::AtenVarCorrectionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TorchD::AtenVarCorrectionOp varOp, PatternRewriter &rewriter) const override {
    Value self = varOp.getSelf();
    auto selfType = dyn_cast<TorchD::BaseTensorType>(self.getType());
    if (!selfType || !selfType.hasSizes()) {
      return failure();
    }
    if (llvm::any_of(selfType.getSizes(),
                     [](int64_t size) { return size == TorchD::kUnknownSize; })) {
      return failure();
    }

    int64_t inputRank = static_cast<int64_t>(selfType.getSizes().size());
    auto varDimsOr = getConstantReductionDims(varOp.getDim(), inputRank, varOp.getLoc(), rewriter);
    if (failed(varDimsOr)) {
      return failure();
    }

    for (Operation *user : self.getUsers()) {
      auto meanOp = dyn_cast<TorchD::AtenMeanDimOp>(user);
      if (!meanOp || meanOp.getSelf() != self) {
        continue;
      }
      if (!keepdimValuesEqual(meanOp, varOp)) {
        continue;
      }

      auto meanDimsOr = getConstantReductionDims(meanOp.getDim(), inputRank, meanOp.getLoc(), rewriter);
      if (failed(meanDimsOr) || !reductionDimsEqual(*varDimsOr, *meanDimsOr)) {
        continue;
      }

      SmallVector<Type, 2> resultTypes = {varOp.getType(), meanOp.getType()};
      auto varMeanOp = rewriter.create<TorchD::AtenVarMeanCorrectionOp>(
        varOp.getLoc(), resultTypes, self, varOp.getDim(), varOp.getCorrection(), varOp.getKeepdim());
      rewriter.replaceOp(varOp, varMeanOp.getResult(0));
      rewriter.replaceOp(meanOp, varMeanOp.getResult(1));
      return success();
    }

    return failure();
  }
};

struct TorchMergeMeanVarPass : public PassWrapper<TorchMergeMeanVarPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "torch-merge-mean-var"; }
  StringRef getDescription() const final {
    return "Merge torch.aten.mean.dim + torch.aten.var.correction into torch.aten.var_mean.correction";
  }

  void getDependentDialects(DialectRegistry &registry) const override { registry.insert<TorchD::TorchDialect>(); }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TorchMergeMeanVarPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

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
        {"torch-merge-mean-var", []() { return createTorchMergeMeanVarPass(); }},
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

std::unique_ptr<Pass> createTorchMergeMeanVarPass() { return std::make_unique<TorchMergeMeanVarPass>(); }

std::unique_ptr<Pass> createTorchFusionPass() { return std::make_unique<TorchFusionPass>(); }

}  // namespace mlir
