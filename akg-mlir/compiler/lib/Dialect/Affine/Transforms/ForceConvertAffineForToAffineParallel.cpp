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

#include "akg/Dialect/Affine/Transforms/ForceConvertAffineForToAffineParallel.h"

#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_FORCECONVERTAFFINEFORTOAFFINEPARALLEL
#define GEN_PASS_DEF_FORCECONVERTAFFINEFORTOAFFINEPARALLEL
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace affine {
namespace {

constexpr auto kGpuReduceStr = "gpu-reduction";
constexpr auto kReductionAxesStr = "reduction_axes";

struct AffineForToParallelPattern : public RewritePattern {
  AffineForToParallelPattern(MLIRContext *context, std::string matchOpType)
      : RewritePattern(affine::AffineForOp::getOperationName(), 1, context) {
    this->matchOpType = matchOpType;
  }

  void initialize() { setHasBoundedRewriteRecursion(); }

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
      auto parallelOp = rewriter.create<affine::AffineParallelOp>(
        forOp.getLoc(), TypeRange(), ArrayRef<arith::AtomicRMWKind>(), llvm::ArrayRef(forOp.getLowerBoundMap()),
        forOp.getLowerBoundOperands(), llvm::ArrayRef(forOp.getUpperBoundMap()), forOp.getUpperBoundOperands(),
        llvm::ArrayRef(forOp.getStepAsInt()));
      parallelOp.getRegion().takeBody(forOp.getRegion());
      Operation *newOp = parallelOp.getOperation();

      // Copy all Attrs from affine::AffineForOp to affine::AffineParallelOp
      for (const auto &attr : op->getAttrs()) {
        newOp->setAttr(attr.getName(), attr.getValue());
      }
      rewriter.replaceOp(op, parallelOp.getResults());
      return success();
    }
    return failure();
  }

 private:
  std::string matchOpType;
};
struct ForceConvertAffineForToAffineParallel
    : public mlir::impl::ForceConvertAffineForToAffineParallelBase<ForceConvertAffineForToAffineParallel> {
  ForceConvertAffineForToAffineParallel() = default;
  explicit ForceConvertAffineForToAffineParallel(const std::string matchOpType) { this->matchOpType = matchOpType; }

  StringRef getArgument() const final { return "force-convert-affine-for-to-affine-parallel"; }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();

    mlir::RewritePatternSet patterns(context);
    patterns.insert<AffineForToParallelPattern>(context, this->matchOpType);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace affine
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createForceConvertAffineForToAffineParallelPass() {
  return std::make_unique<affine::ForceConvertAffineForToAffineParallel>();
}
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createForceConvertAffineForToAffineParallelPass(
  std::string matchOpType) {
  return std::make_unique<affine::ForceConvertAffineForToAffineParallel>(matchOpType);
}
