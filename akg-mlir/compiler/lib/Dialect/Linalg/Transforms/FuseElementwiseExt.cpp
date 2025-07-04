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

#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGELEMENTWISEFUSIONEXT
#define GEN_PASS_DEF_LINALGELEMENTWISEFUSIONEXT
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;

/// For a given list of indices in the range of the `indexingMap` that are
/// folded, return the indices of the corresponding domain. Return
/// `std::nullopt` on failure. Ensures that all the elements of the returned
/// reassociation are distinct.
static ReassociationIndices
getDomainReassociation(AffineMap indexingMap,
                       ReassociationIndicesRef rangeReassociation) {
  assert(indexingMap.isProjectedPermutation() &&
         "expected projected permutation");

  ReassociationIndices domainReassociation = llvm::to_vector<4>(
      llvm::map_range(rangeReassociation, [&](int64_t pos) -> int64_t {
        return cast<AffineDimExpr>(indexingMap.getResults()[pos]).getPosition();
      }));
  // The projected permutation semantics ensures that there is no repetition of
  // the domain indices.
  return domainReassociation;
}

/// For a given `dimSequence`, check if the sequence is conserved in the
/// `indexingMap`. `indexingMap` is expected to be a projected permutation.
/// Non-existence of the sequence returns true as well.
static bool isDimSequencePreservedV2(AffineMap indexingMap,
                                          ReassociationIndicesRef dimSequence) {
  assert(!dimSequence.empty() &&
         "expected non-empty list for dimension sequence");
  assert(indexingMap.isProjectedPermutation() &&
         "expected indexing map to be projected permutation");

  llvm::SmallDenseSet<unsigned, 4> sequenceElements;
  sequenceElements.insert(dimSequence.begin(), dimSequence.end());

  unsigned dimSequenceStart = dimSequence[0];
  for (const auto &expr : enumerate(indexingMap.getResults())) {
    unsigned dimInMapStart = cast<AffineDimExpr>(expr.value()).getPosition();
    // 1.  Check if this start of the sequence.
    if (dimInMapStart == dimSequenceStart) {
      if (expr.index() + dimSequence.size() > indexingMap.getNumResults())
        return false;
      // 1a. Check if sequence is preserved.
      for (const auto &dimInSequence : enumerate(dimSequence)) {
        unsigned dimInMap =
            cast<AffineDimExpr>(
                indexingMap.getResult(expr.index() + dimInSequence.index()))
                .getPosition();
        if (dimInMap != dimInSequence.value())
          return false;
      }
      // Found the sequence. Projected permutation
      // enforces that all AffineDimExprs in the result are unique, so no
      // further checks are needed.
      return true;
    }
    // 2. If position in the expr (which is of type AffineDimExpr) is part
    // of sequence, return false here. This implies the entire sequence does not
    // exist in the indexing map.
    if (sequenceElements.count(dimInMapStart))
      return false;
  }
  // 3. No element of sequence found. Return true.
  return true;
}

// Return the list of dimensions of the iteration domain that can be
// collapsed to allow for fusion with the a producer that is an expand_shape
// operation. If all dimensions created by expansion can be collapsed in the
// iteration space then the reshape is defunct.
//
// Example:
//
// ```mlir
// #map = affine_map<(d0, d1) -> (d0, d1)>
// %1 = tensor.expand_shape %0 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// %2 = tensor.empty [..] : tensor<?x4xf32>
// %3 = linalg.generic {
//     indexing_maps = [#map, #map],
//     iterator_types = ["parallel" ,"parallel"]}
//     ins(%1 : tensor<?x4xf32>) outs(%2 : tensor<?x4xf32>) {.. }
// ```
//
// can be fused by collapsing the dimensions of the iteration space.
//
// ```mlir
// #map = affine_map<(d0) -> (d0)>
// %2 = tensor.empty [..] : tensor<?xf32>
// %3 = linalg.generic {
//     indexing_maps = [#map, #map],
//     iterator_types = ["parallel"]}
//     ins(%1 : tensor<?xf32>) outs(%2 : tensor<?xf32>) {.. }
// %4 = tensor.expand_shape %3 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// ```
//
// In the following example,
//
// ```mlir
// #map0 = affine_map<(d0, d1) -> (d0, d1)>
// #map1 = affine_map<(d0, d1) -> (d1, d0)>
// %1 = tensor.expand_shape %0 [[0, 1]] : tensor<?xf32> into tensor<?x4xf32>
// %2 = tensor.empty [..] : tensor<4x?xf32>
// %2 = linalg.generic {
//     indexing_maps = [#map0, #map1],
//     iterator_types = ["parallel" ,"parallel"]}
//     ins(%1 : tensor<?x4xf32>) outs(%2 : tensor<4x?xf32>) {.. }
// ```
//
// the reshape cannot be fused with the generic op by collapsing the op
// dimensions since the indexing maps will have to contain mods and divs
// to preserve the accesses pattern. When no dimensions of the iteration
// space are collapsable and empty vector is returned.
static SmallVector<ReassociationIndices>
getCollapsableIterationSpaceDims(GenericOp genericOp, OpOperand *fusableOperand,
                                 ArrayRef<ReassociationIndices> reassociation) {
  // Some basic checks for this fusion to be valid.
  if (!genericOp.hasPureTensorSemantics() || genericOp.getNumDpsInits() != 1)
    return {};

  if (!llvm::all_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
        return map.isProjectedPermutation();
      })) {
    return {};
  }

  // Compute all the loops with the reduction iterator types.
  SmallVector<unsigned> reductionDims;
  genericOp.getReductionDims(reductionDims);

  llvm::SmallDenseSet<unsigned, 4> processedIterationDims;
  AffineMap indexingMap = genericOp.getMatchingIndexingMap(fusableOperand);
  auto iteratorTypes = genericOp.getIteratorTypesArray();
  SmallVector<ReassociationIndices> iterationSpaceReassociation;
  for (ReassociationIndicesRef foldedRangeDims : reassociation) {
    assert(!foldedRangeDims.empty() && "unexpected empty reassociation");

    // Ignore dims that are not folded.
    if (foldedRangeDims.size() == 1)
      continue;

    ReassociationIndices foldedIterationSpaceDims =
        getDomainReassociation(indexingMap, foldedRangeDims);

    // Check that the folded iteration dims do not contain already processed
    // dims.
    if (llvm::any_of(foldedIterationSpaceDims, [&](int64_t dim) {
          return processedIterationDims.count(dim);
        }))
      continue;

    // Check that all folded iterator types are all parallel or all reductions.
    utils::IteratorType startIteratorType =
        iteratorTypes[foldedIterationSpaceDims[0]];
    if (!isParallelIterator(startIteratorType) &&
        !isReductionIterator(startIteratorType))
      continue;
    if (llvm::any_of(foldedIterationSpaceDims, [&](int64_t dim) {
          return iteratorTypes[dim] != startIteratorType;
        }))
      continue;

    // If the folded dimensions correspond to a "reduction" iterator type,
    // the folded dimensions need to be "in-order". Strictly speaking this is
    // not necessary, for reductions that are associative and commutative,  but
    // using a more strict definition of reduction for now.
    if (isReductionIterator(startIteratorType)) {
      bool isContiguous = false;
      for (const auto &startDim : llvm::enumerate(reductionDims)) {
        // Move window in `reductionDims` to start of the folded iteration dims.
        if (startDim.value() != foldedIterationSpaceDims[0])
          continue;
        // If sizes doesnt match, trivial not contiguous. This condition should
        // not be hit.
        if (startDim.index() + foldedIterationSpaceDims.size() >
            reductionDims.size())
          break;
        // Check that the contiguity is maintained.
        isContiguous = true;
        for (const auto &foldedDim :
             llvm::enumerate(foldedIterationSpaceDims)) {
          if (reductionDims[foldedDim.index() + startDim.index()] !=
              foldedDim.value()) {
            isContiguous = false;
            break;
          }
        }
        break;
      }
      if (!isContiguous)
        continue;
    }

    // Check that the sequence is preserved in all indexing maps.
    if (llvm::any_of(genericOp.getIndexingMapsArray(),
                     [&](AffineMap indexingMap) {
                       return !isDimSequencePreservedV2(indexingMap,
                                                      foldedIterationSpaceDims);
                     }))
      continue;

    processedIterationDims.insert(foldedIterationSpaceDims.begin(),
                                  foldedIterationSpaceDims.end());
    iterationSpaceReassociation.emplace_back(
        std::move(foldedIterationSpaceDims));
  }

  return iterationSpaceReassociation;
}

class FoldReshapeWithGenericOpByCollapsing
    : public OpRewritePattern<tensor::CollapseShapeOp> {
public:
  FoldReshapeWithGenericOpByCollapsing(MLIRContext *context,
                                       ControlFusionFn foldReshapes,
                                       PatternBenefit benefit = 1)
      : OpRewritePattern<tensor::CollapseShapeOp>(context, benefit),
        controlFoldingReshapes(std::move(foldReshapes)) {}

  LogicalResult matchAndRewrite(tensor::CollapseShapeOp collapseOp,
                                PatternRewriter &rewriter) const override {
    // Fold only if all constraints of fusing with reshape by expansion are met.
    auto producerResult = collapseOp.getSrc().dyn_cast<OpResult>();
    if (!producerResult) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "source not produced by an operation");
    }

    auto genericOp = dyn_cast<GenericOp>(producerResult.getOwner());
    if (!genericOp) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "producer not a generic op");
    }

    auto fuseInitOperand =
        genericOp.getDpsInitOperand(producerResult.getResultNumber());

    SmallVector<ReassociationIndices> collapsableIterationDims =
        getCollapsableIterationSpaceDims(genericOp, fuseInitOperand,
                                         collapseOp.getReassociationIndices());
    if (collapsableIterationDims.empty()) {
      return rewriter.notifyMatchFailure(collapseOp,
                                         "index map cannot be collapsed");
    }
    if (!controlFoldingReshapes(fuseInitOperand)) {
      return rewriter.notifyMatchFailure(collapseOp, "control function failed");
    }
    /*
    std::optional<CollapseResult> replacements =
        collapseOpIterationDims(genericOp, collapsableIterationDims, rewriter);

    Value reshapeReplacement =
        (replacements->results)[collapseOp.getSrc().cast<OpResult>().getResultNumber()];
    if (auto expandOp =
            reshapeReplacement.getDefiningOp<tensor::ExpandShapeOp>()) {
      reshapeReplacement = expandOp.getSrc();
    }

    rewriter.replaceOp(collapseOp, reshapeReplacement);
    rewriter.replaceOp(genericOp, replacements->results);
    */
    return success();
  }

private:
  ControlFusionFn controlFoldingReshapes;
};

namespace {
static bool checkFusedOpDominateAllProducerUsers(Operation *fusedOp, Operation *producer, DominanceInfo &domInfo) {
  for (auto res : producer->getResults()) {
    for (auto user : res.getUsers()) {
      if (!domInfo.properlyDominates(fusedOp, user)) {
        return false;
      }
    }
  }
  return true;
}

static bool CheckIfMatchDominateInSimplePattern0(Operation *fusedOp, Operation *op, DominanceInfo &domInfo) {
  // if op has more than one Result, give up
  if (op->getNumResults() > 1) {
    return false;
  }
  // if op has more than one User, give up
  if (op->getNumResults() == 1 && !op->getResults()[0].hasOneUse()) {
    return false;
  }
  // if op is "ReturnOp" or dominated by fusedOp, order-preserving can be tried here.
  if (isa<func::ReturnOp>(op) || domInfo.properlyDominates(fusedOp, op)) {
    return true;
  }
  // check op's user
  Operation *userOp = *(op->getResults()[0].getUsers().begin());
  return CheckIfMatchDominateInSimplePattern0(fusedOp, userOp, domInfo);
}

// convert:
//  "A-------->B-------->C-------->ReturnOp"
//                            |
//                         fusedOp
// to:
//  "A-------->B-------->C-------->ReturnOp"
//       |
//    fusedOp
static bool TryingtToPreserveOrderInSimplePattern0(Operation *fusedOp, Operation *op, DominanceInfo &domInfo) {
  Operation *userOp = *(op->getResults()[0].getUsers().begin());
  if (isa<func::ReturnOp>(userOp) || domInfo.properlyDominates(fusedOp, userOp)) {
    op->moveBefore(userOp);
    return true;
  }
  if (TryingtToPreserveOrderInSimplePattern0(fusedOp, userOp, domInfo)) {
    op->moveBefore(userOp);
    return true;
  }
  return false;
}

class FuseElementwiseOpsExt : public OpRewritePattern<GenericOp> {
 public:
  FuseElementwiseOpsExt(MLIRContext *context, ControlFusionFn fun, DominanceInfo &domInfo, PatternBenefit benefit = 1)
      : OpRewritePattern<GenericOp>(context, benefit), controlFn(std::move(fun)), domInfo(domInfo) {}

  LogicalResult matchAndRewrite(GenericOp genericOp, PatternRewriter &rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      if (!areElementwiseOpsFusable(&opOperand)) {
        continue;
      }

      if (!controlFn(&opOperand)) {
        continue;
      }

      FailureOr<linalg::ElementwiseOpFusionResult> fusionResult = fuseElementwiseOps(rewriter, &opOperand);
      if (succeeded(fusionResult)) {
        Operation *fusedOp = fusionResult->fusedOp;
        // 1 replace old consumer: Risk-free replacement
        auto replacements = fusedOp->getResults().take_back(genericOp.getNumResults());
        rewriter.replaceOp(genericOp, replacements);
        // 2 replace old producer: Windy replacement
        Operation *producer = opOperand.get().getDefiningOp();
        // 2.1 Pattern0: producer and it's user have only one user.
        if (CheckIfMatchDominateInSimplePattern0(fusedOp, producer, domInfo)) {
          (void)TryingtToPreserveOrderInSimplePattern0(fusedOp, producer, domInfo);
        }
        // final check: whether fusedOp dominates all producer's users
        // todo: Order-preserving, make fusedOp dominates all of the producer's users.
        if (!checkFusedOpDominateAllProducerUsers(fusedOp, producer, domInfo)) {
          return success();
        }
        replacements = fusedOp->getResults().take_front(producer->getNumResults());
        rewriter.replaceOp(producer, replacements);
        return success();
      }
    }

    return failure();
  }

 private:
  ControlFusionFn controlFn;
  DominanceInfo &domInfo;
};

struct LinalgElementwiseFusionExtPass : public impl::LinalgElementwiseFusionExtBase<LinalgElementwiseFusionExtPass> {
  LinalgElementwiseFusionExtPass() : LinalgElementwiseFusionExtBase() {
    // alwayTrueControlFn as default ControlFn
    controlFn = [](OpOperand *fusedOperand) {
      Operation *producer = fusedOperand->get().getDefiningOp();
      return producer != nullptr;
    };
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();
    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;

    RewritePatternSet patterns(context);
    DominanceInfo &domInfo = getAnalysis<DominanceInfo>();
    ;
    (void)patterns.add<FuseElementwiseOpsExt>(context, controlFn, domInfo);
    // Add the patterns that clean up dead operands and results.
    populateEraseUnusedOperandsAndResultsPatterns(patterns);
    populateFoldReshapeOpsByExpansionPatterns(patterns, controlFn);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns), grc);

    RewritePatternSet patterns1(context);
    patterns1.add<FoldReshapeWithGenericOpByCollapsing>(patterns1.getContext(), controlFn);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns1), grc);
  }

 private:
  ControlFusionFn controlFn;
};

}  // namespace

std::unique_ptr<mlir::Pass> mlir::createLinalgElementwiseFusionExtPass() {
  return std::make_unique<LinalgElementwiseFusionExtPass>();
}
