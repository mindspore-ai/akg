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

#include "akg/Dialect/Linalg/Transforms/LinalgSimplify.h"
#include <optional>
#include <utility>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGSIMPLIFY
#define GEN_PASS_DEF_LINALGSIMPLIFY
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

using ValCollector = llvm::SmallVector<Value, 2>;
// map between producer and it's consumers.
static llvm::DenseMap<Value, ValCollector> producerConsumerMap;
// collect all symbolic-equal dimensions.
static std::unordered_map<std::string, ValCollector> dimMap;

/// fuse one of the dynamic broadCasts to their corresponding elementwise Op.
// Convert from:
// %empty0 = tensor.empty(...)
// %producer0 = linalg.generic {}
//   ins(%arg0)
//   outs(%empty0)
// %collapsed0 = tensor.collapse_shape %6 [[0, 1]] {AttachDynTile}
//   : tensor<?x?xf16> into tensor<?xf16>
// %empty1 = tensor.empty(...)
// %producer1 = linalg.generic {}
//  ins(%arg1)
//  outs(%empty1)
// %collapsed1 = tensor.collapse_shape %10 [[0, 1]] {AttachDynTile}
//  : tensor<?x?xf16> into tensor<?xf16>
// %consumer = linalg.generic {} ins(%collapsed0, %collapsed_1 ...) outs(%11 ...)

// To
// %empty0 = tensor.empty(...)
// %producer0 = linalg.generic {}
//   ins(%arg0)
//   outs(%empty0)
// %collapsed0 = tensor.collapse_shape %6 [[0, 1]] {AttachDynTile}
//  : tensor<?x?xf16> into tensor<?xf16>
// %empty1 = tensor.empty(...)
// %producer1 = linalg.generic {}
//  ins(%arg1)
//  outs(%empty1)
// %collapsed1 = tensor.collapse_shape %10 [[0, 1]] {AttachDynTile}
//  : tensor<?x?xf16> into tensor<?xf16>
// %consumer = linalg.generic {} ins(%collapsed0, %collapsed_1 ...) outs(%11 ...)

static void elementwiseOpOperandSimplify(PatternRewriter &rewriter, OpOperand *fusedOperand) {
  auto producer = fusedOperand->get().getDefiningOp<GenericOp>();
  auto consumer = dyn_cast<GenericOp>(fusedOperand->getOwner());
  if (producer == nullptr || consumer == nullptr) {
    return;
  }
  // if producer and consumer already have the same getOutputs, return.
  if (producer.getOutputs()[0].getDefiningOp() == consumer.getOutputs()[0].getDefiningOp()) {
    return;
  }
  // if producer' outs has more than 1 user, return
  // In other words, tensor.emptyOp is only used by its corresponding genericOp
  // before LinalgSimplify Pass.
  if (!producer.getOutputs()[0].hasOneUse()) {
    return;
  }
  if (cast<ShapedType>(producer.getOutputs()[0].getType()).getElementType() !=
      cast<ShapedType>(consumer.getOutputs()[0].getType()).getElementType()) {
    return;
  }
  // if no symbolic expression, return.
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (!analysis.isSameSymbolicShape(producer.getOutputs()[0].getType(), consumer.getOutputs()[0].getType())) {
    return;
  }

  producerConsumerMap[producer.getOutputs()[0]].emplace_back(consumer.getOutputs()[0]);
  if (producerConsumerMap.find(consumer.getOutputs()[0]) != producerConsumerMap.end()) {
    auto iterator0 = producerConsumerMap[producer.getOutputs()[0]].end();
    auto iterator1 = producerConsumerMap[consumer.getOutputs()[0]].begin();
    auto iterator2 = producerConsumerMap[consumer.getOutputs()[0]].end();
    producerConsumerMap[producer.getOutputs()[0]].insert(iterator0, iterator1, iterator2);
  }
}

class ElementwiseOpOperandSimplify : public OpRewritePattern<GenericOp> {
 public:
  ElementwiseOpOperandSimplify(MLIRContext *context, ControlFusionFn fun, PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(std::move(fun)) {}
  LogicalResult matchAndRewrite(GenericOp genericOp, PatternRewriter &rewriter) const override {
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!controlFn(&opOperand)) {
        continue;
      }
      elementwiseOpOperandSimplify(rewriter, &opOperand);
    }
    return success();
  }

 private:
  ControlFusionFn controlFn;
};

void mlir::populateElementwiseOpsSimplify(RewritePatternSet &patterns,
                                          const ControlFusionFn &controlElementwiseOpsFusion) {
  auto *context = patterns.getContext();
  (void)patterns.add<ElementwiseOpOperandSimplify>(context, controlElementwiseOpsFusion);
}

namespace {
struct LinalgSimplify : public impl::LinalgSimplifyBase<LinalgSimplify> {
  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    producerConsumerMap.clear();
    dimMap.clear();

    RewritePatternSet simplifyPatterns(context);
    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;

    // optional:Add broadcastTo transform patterns for Fusion. use populateBroadcastToTransform Pass
    ControlFusionFn controlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer;
    };

    // Add elementwise op simplify patterns.
    populateElementwiseOpsSimplify(simplifyPatterns, controlFn);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(simplifyPatterns), grc);

    // Fold consumer's outs to producer's outs
    for (auto &producer : producerConsumerMap) {
      for (auto &consumer : producer.second) {
        consumer.replaceAllUsesWith(producer.first);
      }
    }
    // Add the patterns that clean up dead operands and results.
    RewritePatternSet csePatterns0(context);
    populateEraseUnusedOperandsAndResultsPatterns(csePatterns0);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(csePatterns0), grc);

    // Binding all equal (such as symbolic-equal) dimensions to the same SSA value.
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    // Analyze tensor.castOp and tensor.dimOp to get some equal relations.
    // 1 Analyze tensor.castOp
    // 1.1 Collect tensor.cast() first.
    SmallVector<Operation *> castCollector;
    op->walk<WalkOrder::PreOrder>([&](tensor::CastOp castOp) {
      castCollector.push_back(castOp.getOperation());
      return WalkResult::advance();
    });
    // Reverse traversal castOp, map all associated sourceDim/destDim to a unique num.
    // a   b           tensor.cast a to c
    //  \ /            tensor.cast b to c
    //   c   d         tensor.cast c to e
    //    \ /          tensor.cast d to e
    //     e
    //   symMap0
    // c -------> #num0
    // d -------> #num0
    // a -------> #num0
    // b -------> #num0
    uint64_t uniqueNum = 0;
    std::map<std::string, uint64_t> symMap0;
    for (int64_t i = castCollector.size() - 1; i >= 0; i--) {
      tensor::CastOp castOp = dyn_cast<tensor::CastOp>(castCollector[i]);
      Type inTy = castOp.getSource().getType();
      Type outTy = castOp.getType();
      llvm::SmallVector<std::string> inSymbol = *analysis.getSymbolicShape(inTy);
      llvm::SmallVector<std::string> outSymbol = *analysis.getSymbolicShape(outTy);
      for (uint64_t j = 0; j < inSymbol.size(); j++) {
        if (cast<ShapedType>(inTy).getShape()[j] != ShapedType::kDynamic ||
            cast<ShapedType>(outTy).getShape()[j] != ShapedType::kDynamic) {
          continue;
        }
        if (symMap0.find(outSymbol[j]) == symMap0.end()) {
          symMap0[outSymbol[j]] = uniqueNum++;
        }
        symMap0[inSymbol[j]] = symMap0[outSymbol[j]];
      }
    }
    // analysis tensor.dim()
    std::map<uint64_t, std::string> symMap1;
    op->walk([&](tensor::DimOp dimOp) {
      std::optional<int64_t> maybeConstantIndex = dimOp.getConstantIndex();
      if (!maybeConstantIndex) {
        return WalkResult::advance();
      }
      std::optional<std::string> dim = analysis.getSymbolicDim(dimOp.getSource().getType(), *maybeConstantIndex);
      if (!dim) {
        return WalkResult::advance();
      }
      if (symMap0.find(*dim) == symMap0.end()) {
        symMap0[*dim] = uniqueNum++;
      }
      if (symMap1.find(symMap0[*dim]) == symMap1.end()) {
        // if "tensor.dim(..) -> b" appears first, then create symMap1 as follow:
        //   symMap0        symMap1
        // a -------> #num0 -------> a
        // b -------> #num0 -------> a
        // c -------> #num0 -------> a
        // d -------> #num0 -------> a
        symMap1[symMap0[*dim]] = *dim;
      }
      // equal dimOps are added in sequence by id.
      dimMap[symMap1[symMap0[*dim]]].emplace_back(dimOp.getOperation()->getResults()[0]);
      return WalkResult::advance();
    });
    for (auto &dimOp : dimMap) {
      for (int i = dimOp.second.size() - 1; i >= 1; i--) {
        dimOp.second[i].replaceAllUsesWith(dimOp.second[0]);
      }
    }
    // Add the patterns that clean up dead operands and results.
    RewritePatternSet csePatterns1(context);
    populateEraseUnusedOperandsAndResultsPatterns(csePatterns1);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(csePatterns1), grc);
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> mlir::createLinalgSimplifyPass() { return std::make_unique<LinalgSimplify>(); }
