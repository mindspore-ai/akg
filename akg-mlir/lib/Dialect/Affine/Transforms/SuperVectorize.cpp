/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
//===- SuperVectorize.cpp - Vectorize Pass Impl ---------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
// This file implements vectorization of loops, operations and data types to
// a target-independent, n-D super-vector abstraction.
//===----------------------------------------------------------------------===//

#include "akg/Dialect/Affine/Transforms/SuperVectorize.h"
#include <optional>
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "akg/Utils/Constants.h"

namespace mlir {
namespace affine {
#define GEN_PASS_DEF_AFFINEVECTORIZE
#include "mlir/Dialect/Affine/Passes.h.inc"

}  // namespace affine
}  // namespace mlir

using namespace mlir;    // NOLINT(build/namespaces)
using namespace affine;  // NOLINT(build/namespaces)
using namespace vector;  // NOLINT(build/namespaces)

#define DEBUG_TYPE "early-vect-ext"

using llvm::dbgs;

namespace {

static NestedPattern &vectorTransferPattern() {
  static auto pattern =
    affine::matcher::Op([](Operation &op) { return isa<vector::TransferReadOp, vector::TransferWriteOp>(op); });
  return pattern;
}

/// Returns a FilterFunctionType that can be used in NestedPattern to match a
/// loop whose underlying load/store accesses are either invariant or all
// varying along the `fastestVaryingMemRefDimension`.
static FilterFunctionType isVectorizableLoopPtrFactory(const DenseSet<Operation *> &parallelLoops,
                                                       int fastestVaryingMemRefDimension) {
  return [&parallelLoops, fastestVaryingMemRefDimension](Operation &forOp) {
    auto loop = cast<AffineForOp>(forOp);
    if (!parallelLoops.contains(loop)) {
      return false;
    }
    int memRefDim = -1;
    auto vectorizableBody = isVectorizableLoopBody(loop, &memRefDim, vectorTransferPattern());
    if (!vectorizableBody) {
      return false;
    }
    return memRefDim == -1 || fastestVaryingMemRefDimension == -1 || memRefDim == fastestVaryingMemRefDimension;
  };
}

/// Creates a vectorization pattern from the command line arguments.
/// Up to 3-D patterns are supported.
/// If the command line argument requests a pattern of higher order, returns an
/// empty pattern list which will conservatively result in no vectorization.
static std::optional<NestedPattern> makePattern(const DenseSet<Operation *> &parallelLoops, int vectorRank,
                                                ArrayRef<int64_t> fastestVaryingPattern) {
  using affine::matcher::For;
  constexpr int64_t kInvalidDim = -1;
  constexpr size_t kPatternSizeTwo = 2;
  constexpr size_t kPatternSizeThree = 3;
  constexpr int kRankSizeOne = 1;
  constexpr int kRankSizeTwo = 2;
  constexpr int kRankSizeThree = 3;
  int64_t d0 = fastestVaryingPattern.empty() ? kInvalidDim : fastestVaryingPattern[0];
  int64_t d1 = fastestVaryingPattern.size() < kPatternSizeTwo ? kInvalidDim : fastestVaryingPattern[1];
  int64_t d2 = fastestVaryingPattern.size() < kPatternSizeThree ? kInvalidDim : fastestVaryingPattern[2];
  switch (vectorRank) {
    case kRankSizeOne:
      return For(isVectorizableLoopPtrFactory(parallelLoops, d0));
    case kRankSizeTwo:
      return For(isVectorizableLoopPtrFactory(parallelLoops, d0), For(isVectorizableLoopPtrFactory(parallelLoops, d1)));
    case kRankSizeThree:
      return For(
        isVectorizableLoopPtrFactory(parallelLoops, d0),
        For(isVectorizableLoopPtrFactory(parallelLoops, d1), For(isVectorizableLoopPtrFactory(parallelLoops, d2))));
    default: {
      return std::nullopt;
    }
  }
}

/// Base state for the vectorize pass.
/// Command line arguments are preempted by non-empty pass arguments.
struct VectorizeAKG : public affine::impl::AffineVectorizeBase<VectorizeAKG> {
  using Base::Base;

  void runOnOperation() override;
};

}  // namespace

static void vectorizeLoopIfProfitable(Operation *loop, unsigned depthInPattern, unsigned patternDepth,
                                      VectorizationStrategy *strategy) {
  assert(patternDepth > depthInPattern && "patternDepth is greater than depthInPattern");
  if (patternDepth - depthInPattern > strategy->vectorSizes.size()) {
    // Don't vectorize this loop
    return;
  }
  strategy->loopToVectorDim[loop] = strategy->vectorSizes.size() - (patternDepth - depthInPattern);
}

/// Implements a simple strawman strategy for vectorization.
/// Given a matched pattern `matches` of depth `patternDepth`, this strategy
/// greedily assigns the fastest varying dimension ** of the vector ** to the
/// innermost loop in the pattern.
/// When coupled with a pattern that looks for the fastest varying dimension in
/// load/store MemRefs, this creates a generic vectorization strategy that works
/// for any loop in a hierarchy (outermost, innermost or intermediate).
/// In the future we should additionally increase the power of the
/// profitability analysis along 3 directions
///   1. account for loop extents (both static and parametric + annotations);
///   2. account for data layout permutations;
///   3. account for impact of vectorization on maximal loop fusion.
/// Then we can quantify the above to build a cost model and search over
/// strategies.
static LogicalResult analyzeProfitability(ArrayRef<NestedMatch> matches, unsigned depthInPattern, unsigned patternDepth,
                                          VectorizationStrategy *strategy) {
  for (auto m : matches) {
    if (failed(analyzeProfitability(m.getMatchedChildren(), depthInPattern + 1, patternDepth, strategy))) {
      return failure();
    }
    vectorizeLoopIfProfitable(m.getMatchedOperation(), depthInPattern, patternDepth, strategy);
  }
  return success();
}

///// end Hoist to a VectorizationStrategy.cpp when appropriate /////

/// Registers the vector replacement of a scalar operation and its result
/// values. Both operations must have the same number of results.
/// This utility is used to register the replacement for the vast majority of
/// the vectorized operations.
/// Example of scalar-to-vector replacement mapping:
///   * 'replaced': %0 = arith.addf %1, %2 : f32
///   * 'replacement': %0 = arith.addf %1, %2 : vector<128xf32>
void VectorizationState::registerOpVectorReplacement(Operation *replaced, Operation *replacement) {
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ commit vectorized op:\n");
  LLVM_DEBUG(dbgs() << *replaced << "\n");
  LLVM_DEBUG(dbgs() << "into\n");
  LLVM_DEBUG(dbgs() << *replacement << "\n");

  assert(replaced->getNumResults() == replacement->getNumResults() && "Unexpected replaced and replacement results");
  assert(opVectorReplacement.count(replaced) == 0 && "already registered");
  opVectorReplacement[replaced] = replacement;

  for (auto resultTuple : llvm::zip(replaced->getResults(), replacement->getResults())) {
    registerValueVectorReplacementImpl(std::get<0>(resultTuple), std::get<1>(resultTuple));
  }
}

/// Registers the vector replacement of a scalar value. The replacement
/// operation should have a single result, which replaces the scalar value.
/// This utility is used to register the vector replacement of block arguments
/// and operation results which are not directly vectorized (i.e., their
/// scalar version still exists after vectorization), like uniforms.
/// Example of scalar-to-vector replacement mapping:
///   * 'replaced': block argument or operation outside of the vectorized loop.
///   * 'replacement': %0 = vector.broadcast %1 : f32 to vector<128xf32>
void VectorizationState::registerValueVectorReplacement(Value replaced, Operation *replacement) {
  assert(replacement->getNumResults() == 1 && "Expected single-result replacement");
  if (Operation *defOp = replaced.getDefiningOp()) {
    registerOpVectorReplacement(defOp, replacement);
  } else {
    registerValueVectorReplacementImpl(replaced, replacement->getResult(0));
  }
}

/// Registers the vector replacement of a block argument (e.g., iter_args).
/// Example of scalar-to-vector replacement mapping:
///   * 'replaced': 'iter_arg' block argument.
///   * 'replacement': vectorized 'iter_arg' block argument.
void VectorizationState::registerBlockArgVectorReplacement(BlockArgument replaced, BlockArgument replacement) {
  registerValueVectorReplacementImpl(replaced, replacement);
}

void VectorizationState::registerValueVectorReplacementImpl(Value replaced, Value replacement) {
  assert(!valueVectorReplacement.contains(replaced) && "Vector replacement already registered");
  assert(isa<VectorType>(replacement.getType()) && "Expected vector type in vector replacement");
  valueVectorReplacement.map(replaced, replacement);
}

/// Registers the scalar replacement of a scalar value. 'replacement' must be
/// scalar. Both values must be block arguments. Operation results should be
/// replaced using the 'registerOp*' utilitites.
/// This utility is used to register the replacement of block arguments
/// that are within the loop to be vectorized and will continue being scalar
/// within the vector loop.
/// Example of scalar-to-vector replacement mapping:
///   * 'replaced': induction variable of a loop to be vectorized.
///   * 'replacement': new induction variable in the new vector loop.
void VectorizationState::registerValueScalarReplacement(BlockArgument replaced, BlockArgument replacement) {
  registerValueScalarReplacementImpl(replaced, replacement);
}

/// Registers the scalar replacement of a scalar result returned from a
/// reduction loop. 'replacement' must be scalar.
/// This utility is used to register the replacement for scalar results of
/// vectorized reduction loops with iter_args.
/// Example of reduction-loop scalar result replacement:
///   * 'replaced': %0 = affine.for %i = 0 to 512 iter_args(%x = ...) -> (f32)
///   * 'replacement': %1 = vector.reduction <add>, %0 : vector<4xf32> into f32
void VectorizationState::registerLoopResultScalarReplacement(Value replaced, Value replacement) {
  assert(isa<AffineForOp>(replaced.getDefiningOp()));
  assert(loopResultScalarReplacement.count(replaced) == 0 && "already registered");
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ will replace a result of the loop "
                       "with scalar: "
                    << replacement);
  loopResultScalarReplacement[replaced] = replacement;
}

void VectorizationState::registerValueScalarReplacementImpl(Value replaced, Value replacement) {
  assert(!valueScalarReplacement.contains(replaced) && "Scalar value replacement already registered");
  assert(!isa<VectorType>(replacement.getType()) && "Expected scalar type in scalar replacement");
  valueScalarReplacement.map(replaced, replacement);
}

/// Returns in 'replacedVals' the scalar replacement for values in 'inputVals'.
void VectorizationState::getScalarValueReplacementsFor(ValueRange inputVals, SmallVectorImpl<Value> &replacedVals) {
  std::transform(inputVals.begin(), inputVals.end(), std::back_inserter(replacedVals),
                 [this](Value inputVal) { return valueScalarReplacement.lookupOrDefault(inputVal); });
}

/// Erases a loop nest, including all its nested operations.
static void eraseLoopNest(AffineForOp forOp) {
  LLVM_DEBUG(dbgs() << "[early-vect]+++++ erasing:\n" << forOp << "\n");
  forOp.erase();
}

/// Erases the scalar loop nest after its successful vectorization.
void VectorizationState::finishVectorizationPattern(AffineForOp rootLoop) {
  LLVM_DEBUG(dbgs() << "\n[early-vect] Finalizing vectorization\n");
  eraseLoopNest(rootLoop);
}

namespace {
struct ComputeMemoryOpIndicesParams {
  VectorizationState &state;
  SmallVectorImpl<Value> &results;
};
}  // namespace

// Apply 'map' with 'mapOperands' returning resulting values in 'results'.
static void computeMemoryOpIndices(Operation *op, AffineMap map, ValueRange mapOperands,
                                   const ComputeMemoryOpIndicesParams &params) {
  auto &[state, results] = params;
  for (auto resultExpr : map.getResults()) {
    auto singleResMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(), resultExpr);
    auto afOp = state.builder.create<AffineApplyOp>(op->getLoc(), singleResMap, mapOperands);
    results.push_back(afOp);
  }
}

/// Returns the vector type resulting from applying the provided vectorization
/// strategy on the scalar type.
static VectorType getVectorType(Type scalarTy, const VectorizationStrategy *strategy) {
  assert(!isa<VectorType>(scalarTy) && "Expected scalar type");
  return VectorType::get(strategy->vectorSizes, scalarTy);
}

/// Tries to transform a scalar constant into a vector constant. Returns the
/// vector constant if the scalar type is valid vector element type. Returns
/// nullptr, otherwise.
static arith::ConstantOp vectorizeConstant(arith::ConstantOp constOp, VectorizationState &state) {
  Type scalarTy = constOp.getType();
  if (!VectorType::isValidElementType(scalarTy)) {
    return nullptr;
  }

  auto vecTy = getVectorType(scalarTy, state.strategy);
  auto vecAttr = DenseElementsAttr::get(vecTy, constOp.getValue());

  OpBuilder::InsertionGuard guard(state.builder);
  Operation *parentOp = state.builder.getInsertionBlock()->getParentOp();
  // Find the innermost vectorized ancestor loop to insert the vector constant.
  while ((parentOp != nullptr) && (state.vecLoopToVecDim.count(parentOp) == 0u)) {
    parentOp = parentOp->getParentOp();
  }
  assert(parentOp && state.vecLoopToVecDim.count(parentOp) && isa<AffineForOp>(parentOp) &&
         "Expected a vectorized for op");
  auto vecForOp = cast<AffineForOp>(parentOp);
  state.builder.setInsertionPointToStart(vecForOp.getBody());
  auto newConstOp = state.builder.create<arith::ConstantOp>(constOp.getLoc(), vecAttr);

  // Register vector replacement for future uses in the scope.
  state.registerOpVectorReplacement(constOp, newConstOp);
  return newConstOp;
}

/// Creates a constant vector filled with the neutral elements of the given
/// reduction. The scalar type of vector elements will be taken from
/// `oldOperand`.
static arith::ConstantOp createInitialVector(arith::AtomicRMWKind reductionKind, Value oldOperand,
                                             VectorizationState &state) {
  Type scalarTy = oldOperand.getType();
  if (!VectorType::isValidElementType(scalarTy)) {
    return nullptr;
  }

  Attribute valueAttr = getIdentityValueAttr(reductionKind, scalarTy, state.builder, oldOperand.getLoc());
  auto vecTy = getVectorType(scalarTy, state.strategy);
  auto vecAttr = DenseElementsAttr::get(vecTy, valueAttr);
  auto newConstOp = state.builder.create<arith::ConstantOp>(oldOperand.getLoc(), vecAttr);

  return newConstOp;
}

/// Creates a mask used to filter out garbage elements in the last iteration
/// of unaligned loops. If a mask is not required then `nullptr` is returned.
/// The mask will be a vector of booleans representing meaningful vector
/// elements in the current iteration. It is filled with ones for each iteration
/// except for the last one, where it has the form `11...100...0` with the
/// number of ones equal to the number of meaningful elements (i.e. the number
/// of iterations that would be left in the original loop).
static Value createMask(AffineForOp vecForOp, VectorizationState &state) {
  assert(state.strategy->vectorSizes.size() == 1 && "Creating a mask non-1-D vectors is not supported.");
  assert(vecForOp.getStep() == state.strategy->vectorSizes[0] &&
         "Creating a mask for loops with non-unit original step size is not "
         "supported.");

  // Check if we have already created the mask.
  if (Value mask = state.vecLoopToMask.lookup(vecForOp)) {
    return mask;
  }

  // If the loop has constant bounds and the original number of iterations is
  // divisible by the vector size then we don't need a mask.
  if (vecForOp.hasConstantBounds()) {
    int64_t originalTripCount = vecForOp.getConstantUpperBound() - vecForOp.getConstantLowerBound();
    if (originalTripCount % vecForOp.getStepAsInt() == 0) {
      return nullptr;
    }
  }

  OpBuilder::InsertionGuard guard(state.builder);
  state.builder.setInsertionPointToStart(vecForOp.getBody());

  // We generate the mask using the `vector.create_mask` operation which accepts
  // the number of meaningful elements (i.e. the length of the prefix of 1s).
  // To compute the number of meaningful elements we subtract the current value
  // of the iteration variable from the upper bound of the loop. Example:
  //     // 500 is the upper bound of the loop
  //     #map = affine_map<(d0) -> (500 - d0)>
  //     %elems_left = affine.apply #map(%iv)
  //     %mask = vector.create_mask %elems_left : vector<128xi1>

  Location loc = vecForOp.getLoc();

  // First we get the upper bound of the loop using `affine.apply` or
  // `affine.min`.
  AffineMap ubMap = vecForOp.getUpperBoundMap();
  Value ub;
  if (ubMap.getNumResults() == 1) {
    ub = state.builder.create<AffineApplyOp>(loc, vecForOp.getUpperBoundMap(), vecForOp.getUpperBoundOperands());
  } else {
    ub = state.builder.create<AffineMinOp>(loc, vecForOp.getUpperBoundMap(), vecForOp.getUpperBoundOperands());
  }
  // Then we compute the number of (original) iterations left in the loop.
  AffineExpr subExpr = state.builder.getAffineDimExpr(0) - state.builder.getAffineDimExpr(1);
  Value itersLeft =
    makeComposedAffineApply(state.builder, loc, AffineMap::get(2, 0, subExpr), {ub, vecForOp.getInductionVar()});
  // If the affine maps were successfully composed then `ub` is unneeded.
  if (ub.use_empty()) {
    ub.getDefiningOp()->erase();
  }
  // Finally we create the mask.
  Type maskTy = VectorType::get(state.strategy->vectorSizes, state.builder.getIntegerType(1));
  Value mask = state.builder.create<vector::CreateMaskOp>(loc, maskTy, itersLeft);

  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ creating a mask:\n" << itersLeft << "\n" << mask << "\n");

  state.vecLoopToMask[vecForOp] = mask;
  return mask;
}

/// Returns true if the provided value is vector uniform given the vectorization
/// strategy.
// For now, only values that are induction variables of loops not in
// `loopToVectorDim` or invariants to all the loops in the vectorization
// strategy are considered vector uniforms.
static bool isUniformDefinition(Value value, const VectorizationStrategy *strategy) {
  AffineForOp forOp = getForInductionVarOwner(value);
  if (forOp && strategy->loopToVectorDim.count(forOp) == 0) {
    return true;
  }

  for (auto loopToDim : strategy->loopToVectorDim) {
    auto loop = cast<AffineForOp>(loopToDim.first);
    if (!loop.isDefinedOutsideOfLoop(value)) {
      return false;
    }
  }
  return true;
}

/// Generates a broadcast op for the provided uniform value using the
/// vectorization strategy in 'state'.
static Operation *vectorizeUniform(Value uniformVal, VectorizationState &state) {
  OpBuilder::InsertionGuard guard(state.builder);
  Value uniformScalarRepl = state.valueScalarReplacement.lookupOrDefault(uniformVal);
  state.builder.setInsertionPointAfterValue(uniformScalarRepl);

  auto vectorTy = getVectorType(uniformVal.getType(), state.strategy);
  auto bcastOp = state.builder.create<BroadcastOp>(uniformVal.getLoc(), vectorTy, uniformScalarRepl);
  state.registerValueVectorReplacement(uniformVal, bcastOp);
  return bcastOp;
}

/// Tries to vectorize a given `operand` by applying the following logic:
/// 1. if the defining operation has been already vectorized, `operand` is
///    already in the proper vector form;
/// 2. if the `operand` is a constant, returns the vectorized form of the
///    constant;
/// 3. if the `operand` is uniform, returns a vector broadcast of the `op`;
/// 4. otherwise, the vectorization of `operand` is not supported.
/// Newly created vector operations are registered in `state` as replacement
/// for their scalar counterparts.
/// In particular this logic captures some of the use cases where definitions
/// that are not scoped under the current pattern are needed to vectorize.
/// One such example is top level function constants that need to be splatted.
/// Returns an operand that has been vectorized to match `state`'s strategy if
/// vectorization is possible with the above logic. Returns nullptr otherwise.
/// Handle more complex cases.
static Value vectorizeOperand(Value operand, VectorizationState &state) {
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ vectorize operand: " << operand);
  // If this value is already vectorized, we are done.
  if (Value vecRepl = state.valueVectorReplacement.lookupOrNull(operand)) {
    LLVM_DEBUG(dbgs() << " -> already vectorized: " << vecRepl);
    return vecRepl;
  }

  // An vector operand that is not in the replacement map should never reach
  // this point. Reaching this point could mean that the code was already
  // vectorized and we shouldn't try to vectorize already vectorized code.
  assert(!isa<VectorType>(operand.getType()) && "Vector op not found in replacement map");

  // Vectorize constant.
  if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
    auto vecConstant = vectorizeConstant(constOp, state);
    LLVM_DEBUG(dbgs() << "-> constant: " << vecConstant);
    return vecConstant.getResult();
  }

  // Vectorize uniform values.
  if (isUniformDefinition(operand, state.strategy)) {
    Operation *vecUniform = vectorizeUniform(operand, state);
    LLVM_DEBUG(dbgs() << "-> uniform: " << *vecUniform);
    return vecUniform->getResult(0);
  }

  // Check for unsupported block argument scenarios. A supported block argument
  // should have been vectorized already.
  if (operand.getDefiningOp() == nullptr) {
    LLVM_DEBUG(dbgs() << "-> unsupported block argument\n");
  } else {
    // Generic unsupported case.
    LLVM_DEBUG(dbgs() << "-> non-vectorizable\n");
  }

  return nullptr;
}

/// Vectorizes an affine load with the vectorization strategy in 'state' by
/// generating a 'vector.transfer_read' op with the proper permutation map
/// inferred from the indices of the load. The new 'vector.transfer_read' is
/// registered as replacement of the scalar load. Returns the newly created
/// 'vector.transfer_read' if vectorization was successful. Returns nullptr,
/// otherwise.
static Operation *vectorizeAffineLoad(AffineLoadOp loadOp, VectorizationState &state) {
  MemRefType memRefType = loadOp.getMemRefType();
  Type elementType = memRefType.getElementType();
  auto vectorType = VectorType::get(state.strategy->vectorSizes, elementType);

  // Replace map operands with operands from the vector loop nest.
  SmallVector<Value, kSmallVectorSizeEight> mapOperands;
  state.getScalarValueReplacementsFor(loadOp.getMapOperands(), mapOperands);

  // Compute indices for the transfer op. AffineApplyOp's may be generated.
  SmallVector<Value, kSmallVectorSizeEight> indices;
  indices.reserve(memRefType.getRank());
  if (loadOp.getAffineMap() != state.builder.getMultiDimIdentityMap(memRefType.getRank())) {
    computeMemoryOpIndices(loadOp, loadOp.getAffineMap(), mapOperands, {state, indices});
  } else {
    indices.append(mapOperands.begin(), mapOperands.end());
  }

  // Compute permutation map using the information of new vector loops.
  auto permutationMap = makePermutationMap(state.builder.getInsertionBlock(), indices, state.vecLoopToVecDim);
  if (!permutationMap) {
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ can't compute permutationMap\n");
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ permutationMap: ");
  LLVM_DEBUG(permutationMap.print(dbgs()));

  auto transfer = state.builder.create<vector::TransferReadOp>(loadOp.getLoc(), vectorType, loadOp.getMemRef(), indices,
                                                               permutationMap);

  // Register replacement for future uses in the scope.
  state.registerOpVectorReplacement(loadOp, transfer);
  return transfer;
}

/// Vectorizes an affine store with the vectorization strategy in 'state' by
/// generating a 'vector.transfer_write' op with the proper permutation map
/// inferred from the indices of the store. The new 'vector.transfer_store' is
/// registered as replacement of the scalar load. Returns the newly created
/// 'vector.transfer_write' if vectorization was successful. Returns nullptr,
/// otherwise.
static Operation *vectorizeAffineStore(AffineStoreOp storeOp, VectorizationState &state) {
  MemRefType memRefType = storeOp.getMemRefType();
  Value vectorValue = vectorizeOperand(storeOp.getValueToStore(), state);
  if (!vectorValue) {
    return nullptr;
  }

  // Replace map operands with operands from the vector loop nest.
  SmallVector<Value, kSmallVectorSizeEight> mapOperands;
  state.getScalarValueReplacementsFor(storeOp.getMapOperands(), mapOperands);

  // Compute indices for the transfer op. AffineApplyOp's may be generated.
  SmallVector<Value, kSmallVectorSizeEight> indices;
  indices.reserve(memRefType.getRank());
  if (storeOp.getAffineMap() != state.builder.getMultiDimIdentityMap(memRefType.getRank())) {
    computeMemoryOpIndices(storeOp, storeOp.getAffineMap(), mapOperands, {state, indices});
  } else {
    indices.append(mapOperands.begin(), mapOperands.end());
  }

  // Compute permutation map using the information of new vector loops.
  auto permutationMap = makePermutationMap(state.builder.getInsertionBlock(), indices, state.vecLoopToVecDim);
  if (!permutationMap) {
    return nullptr;
  }
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ permutationMap: ");
  LLVM_DEBUG(permutationMap.print(dbgs()));

  auto transfer = state.builder.create<vector::TransferWriteOp>(storeOp.getLoc(), vectorValue, storeOp.getMemRef(),
                                                                indices, permutationMap);
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ vectorized store: " << transfer);

  // Register replacement for future uses in the scope.
  state.registerOpVectorReplacement(storeOp, transfer);
  return transfer;
}

/// Returns true if `value` is a constant equal to the neutral element of the
/// given vectorizable reduction.
static bool isNeutralElementConst(arith::AtomicRMWKind reductionKind, Value value, VectorizationState &state) {
  Type scalarTy = value.getType();
  if (!VectorType::isValidElementType(scalarTy)) {
    return false;
  }
  Attribute valueAttr = getIdentityValueAttr(reductionKind, scalarTy, state.builder, value.getLoc());
  if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(value.getDefiningOp())) {
    return constOp.getValue() == valueAttr;
  }
  return false;
}

/// Vectorizes a loop with the vectorization strategy in 'state'. A new loop is
/// created and registered as replacement for the scalar loop. The builder's
/// insertion point is set to the new loop's body so that subsequent vectorized
/// operations are inserted into the new loop. If the loop is a vector
/// dimension, the step of the newly created loop will reflect the vectorization
/// factor used to vectorized that dimension.
static Operation *vectorizeAffineForOp(AffineForOp forOp, VectorizationState &state) {
  const VectorizationStrategy &strategy = *state.strategy;
  auto loopToVecDimIt = strategy.loopToVectorDim.find(forOp);
  bool isLoopVecDim = loopToVecDimIt != strategy.loopToVectorDim.end();
  // Vectorization of reduction loops is not supported for non-unit steps.
  if (isLoopVecDim && forOp.getNumIterOperands() > 0 && forOp.getStep() != 1) {
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ unsupported step size for reduction loop: " << forOp.getStep() << "\n");
    return nullptr;
  }

  // If we are vectorizing a vector dimension, compute a new step for the new
  // vectorized loop using the vectorization factor for the vector dimension.
  // Otherwise, propagate the step of the scalar loop.
  int64_t newStep;
  if (isLoopVecDim) {
    unsigned vectorDim = loopToVecDimIt->second;
    assert(vectorDim < strategy.vectorSizes.size() && "vector dim overflow");
    int64_t forOpVecFactor = strategy.vectorSizes[vectorDim];
    newStep = forOp.getStepAsInt() * forOpVecFactor;
  } else {
    newStep = forOp.getStepAsInt();
  }

  // Get information about reduction kinds.
  ArrayRef<LoopReduction> reductions;
  if (isLoopVecDim && forOp.getNumIterOperands() > 0) {
    auto it = strategy.reductionLoops.find(forOp);
    assert(it != strategy.reductionLoops.end() && "Reduction descriptors not found when vectorizing a reduction loop");
    reductions = it->second;
    assert(reductions.size() == forOp.getNumIterOperands() &&
           "The size of reductions array must match the number of iter_args");
  }

  // Vectorize 'iter_args'.
  SmallVector<Value, kSmallVectorSizeEight> vecIterOperands;
  if (!isLoopVecDim) {
    std::transform(forOp.getInits().begin(), forOp.getInits().end(), std::back_inserter(vecIterOperands),
                   [&state](Value operand) { return vectorizeOperand(operand, state); });
  } else {
    // For reduction loops we need to pass a vector of neutral elements as an
    // initial value of the accumulator. We will add the original initial value
    // later.
    std::transform(llvm::zip(reductions, forOp.getInits()).begin(), llvm::zip(reductions, forOp.getInits()).end(),
                   std::back_inserter(vecIterOperands), [&state](auto redAndOperand) {
                     return createInitialVector(std::get<0>(redAndOperand).kind, std::get<1>(redAndOperand), state);
                   });
  }

  auto vecForOp =
    state.builder.create<AffineForOp>(forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
                                      forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), newStep, vecIterOperands,
                                      [](OpBuilder &, Location, Value, ValueRange) {
                                        // Make sure we don't create a default terminator in the loop body as
                                        // the proper terminator will be added during vectorization.
                                      });

  // A vector replacement will also be added in the future when
  // vectorization of linear ops is supported.
  // Register loop-related replacements:
  //   1) The new vectorized loop is registered as vector replacement of the
  //      scalar loop.
  //   2) The new iv of the vectorized loop is registered as scalar replacement
  //      since a scalar copy of the iv will prevail in the vectorized loop.
  //   3) The new 'iter_args' region arguments are registered as vector
  //      replacements since they have been vectorized.
  //   4) If the loop performs a reduction along the vector dimension, a
  //      `vector.reduction` or similar op is inserted for each resulting value
  //      of the loop and its scalar value replaces the corresponding scalar
  //      result of the loop.
  state.registerOpVectorReplacement(forOp, vecForOp);
  state.registerValueScalarReplacement(forOp.getInductionVar(), vecForOp.getInductionVar());
  for (auto iterTuple : llvm ::zip(forOp.getRegionIterArgs(), vecForOp.getRegionIterArgs())) {
    state.registerBlockArgVectorReplacement(std::get<0>(iterTuple), std::get<1>(iterTuple));
  }

  if (isLoopVecDim) {
    for (unsigned i = 0; i < vecForOp.getNumIterOperands(); ++i) {
      // First, we reduce the vector returned from the loop into a scalar.
      Value reducedRes =
        getVectorReductionOp(reductions[i].kind, state.builder, vecForOp.getLoc(), vecForOp.getResult(i));
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ creating a vector reduction: " << reducedRes);
      // Then we combine it with the original (scalar) initial value unless it
      // is equal to the neutral element of the reduction.
      Value origInit = forOp.getOperand(forOp.getNumControlOperands() + i);
      Value finalRes = reducedRes;
      if (!isNeutralElementConst(reductions[i].kind, origInit, state)) {
        finalRes = arith::getReductionOp(reductions[i].kind, state.builder, reducedRes.getLoc(), reducedRes, origInit);
      }
      state.registerLoopResultScalarReplacement(forOp.getResult(i), finalRes);
    }
    state.vecLoopToVecDim[vecForOp] = loopToVecDimIt->second;
  }

  // Change insertion point so that upcoming vectorized instructions are
  // inserted into the vectorized loop's body.
  state.builder.setInsertionPointToStart(vecForOp.getBody());

  // If this is a reduction loop then we may need to create a mask to filter out
  // garbage in the last iteration.
  if (isLoopVecDim && forOp.getNumIterOperands() > 0) {
    createMask(vecForOp, state);
  }

  return vecForOp;
}

static Operation *vectorizeAffineIfOp(AffineIfOp ifOp, VectorizationState &state) {
  SmallVector<Value, kSmallVectorSizeEight> vecIterOperands;
  state.getScalarValueReplacementsFor(ifOp.getOperands(), vecIterOperands);
  auto vecIfOp = state.builder.create<AffineIfOp>(ifOp.getLoc(), ifOp.getIntegerSet(), vecIterOperands, false);
  vecIfOp.getThenRegion().walk<WalkOrder::PreOrder>([](AffineYieldOp op) { op.erase(); });
  state.registerOpVectorReplacement(ifOp, vecIfOp);

  state.builder.setInsertionPointToStart(&vecIfOp.getThenRegion().front());
  return vecIfOp;
}

/// Vectorizes arbitrary operation by plain widening. We apply generic type
/// widening of all its results and retrieve the vector counterparts for all its
/// operands.
static Operation *widenOp(Operation *op, VectorizationState &state) {
  SmallVector<Type, kSmallVectorSizeEight> vectorTypes;
  std::transform(op->getResults().begin(), op->getResults().end(), std::back_inserter(vectorTypes),
                 [&state](Value result) { return VectorType::get(state.strategy->vectorSizes, result.getType()); });

  SmallVector<Value, kSmallVectorSizeEight> vectorOperands;
  for (Value operand : op->getOperands()) {
    Value vecOperand = vectorizeOperand(operand, state);
    if (!vecOperand) {
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ an operand failed vectorize\n");
      return nullptr;
    }
    vectorOperands.push_back(vecOperand);
  }

  // Create a clone of the op with the proper operands and return types.
  // The following assumes there is always an op with a fixed
  // name that works both in scalar mode and vector mode.
  // Is it worth considering an Operation.clone operation which
  // changes the type so we can promote an Operation with less boilerplate?
  Operation *vecOp =
    state.builder.create(op->getLoc(), op->getName().getIdentifier(), vectorOperands, vectorTypes, op->getAttrs());
  state.registerOpVectorReplacement(op, vecOp);
  return vecOp;
}

/// Vectorizes a yield operation by widening its types. The builder's insertion
/// point is set after the vectorized parent op to continue vectorizing the
/// operations after the parent op. When vectorizing a reduction loop a mask may
/// be used to prevent adding garbage values to the accumulator.
static Operation *vectorizeAffineYieldOp(AffineYieldOp yieldOp, VectorizationState &state) {
  Operation *newYieldOp = widenOp(yieldOp, state);
  Operation *newParentOp = state.builder.getInsertionBlock()->getParentOp();

  // If there is a mask for this loop then we must prevent garbage values from
  // being added to the accumulator by inserting `select` operations, for
  // example:
  //   %val_masked = select %mask, %val, %neutralCst : vector<128xi1>,
  //   vector<128xf32>
  //   %res = arith.addf %acc, %val_masked : vector<128xf32>
  //   affine.yield %res : vector<128xf32>
  if (Value mask = state.vecLoopToMask.lookup(newParentOp)) {
    state.builder.setInsertionPoint(newYieldOp);
    for (unsigned i = 0; i < newYieldOp->getNumOperands(); ++i) {
      SmallVector<Operation *> combinerOps;
      Value reducedVal = matchReduction(cast<AffineForOp>(newParentOp).getRegionIterArgs(), i, combinerOps);
      assert(reducedVal && "expect non-null value for parallel reduction loop");
      assert(combinerOps.size() == 1 && "expect only one combiner op");
      // IterOperands are neutral element vectors.
      Value neutralVal = cast<AffineForOp>(newParentOp).getInits()[i];
      state.builder.setInsertionPoint(combinerOps.back());
      Value maskedReducedVal = state.builder.create<arith::SelectOp>(reducedVal.getLoc(), mask, reducedVal, neutralVal);
      LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ masking an input to a binary op that"
                           "produces value for a yield Op: "
                        << maskedReducedVal);
      combinerOps.back()->replaceUsesOfWith(reducedVal, maskedReducedVal);
    }
  }

  state.builder.setInsertionPointAfter(newParentOp);
  return newYieldOp;
}

/// Encodes Operation-specific behavior for vectorization. In general we
/// assume that all operands of an op must be vectorized but this is not
/// always true. In the future, it would be nice to have a trait that
/// describes how a particular operation vectorizes. For now we implement the
/// case distinction here. Returns a vectorized form of an operation or
/// nullptr if vectorization fails.
// Consider adding a trait to Op to describe how it gets vectorized.
// Maybe some Ops are not vectorizable or require some tricky logic, we cannot
// do one-off logic here; ideally it would be TableGen'd.
static Operation *vectorizeOneOperation(Operation *op, VectorizationState &state) {
  // Sanity checks.
  assert(!isa<vector::TransferReadOp>(op) && "vector.transfer_read cannot be further vectorized");
  assert(!isa<vector::TransferWriteOp>(op) && "vector.transfer_write cannot be further vectorized");
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    return vectorizeAffineLoad(loadOp, state);
  }
  if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    return vectorizeAffineStore(storeOp, state);
  }
  if (auto ifOp = dyn_cast<AffineIfOp>(op)) {
    return vectorizeAffineIfOp(ifOp, state);
  }
  if (auto forOp = dyn_cast<AffineForOp>(op)) {
    return vectorizeAffineForOp(forOp, state);
  }
  if (auto yieldOp = dyn_cast<AffineYieldOp>(op)) {
    return vectorizeAffineYieldOp(yieldOp, state);
  }
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    return vectorizeConstant(constant, state);
  }

  // Other ops with regions are not supported.
  if (op->getNumRegions() != 0) {
    return nullptr;
  }

  return widenOp(op, state);
}

/// Recursive implementation to convert all the nested loops in 'match' to a 2D
/// vector container that preserves the relative nesting level of each loop with
/// respect to the others in 'match'. 'currentLevel' is the nesting level that
/// will be assigned to the loop in the current 'match'.
static void getMatchedAffineLoopsRec(NestedMatch match, unsigned currentLevel,
                                     std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> &loops) {
  // Add a new empty level to the output if it doesn't exist already.
  assert(currentLevel <= loops.size() && "Unexpected currentLevel");
  if (currentLevel == loops.size()) {
    loops.emplace_back();
  }

  // Add current match and recursively visit its children.
  loops[currentLevel].push_back(cast<AffineForOp>(match.getMatchedOperation()));
  for (auto childMatch : match.getMatchedChildren()) {
    getMatchedAffineLoopsRec(childMatch, currentLevel + 1, loops);
  }
}

/// Converts all the nested loops in 'match' to a 2D vector container that
/// preserves the relative nesting level of each loop with respect to the others
/// in 'match'. This means that every loop in 'loops[i]' will have a parent loop
/// in 'loops[i-1]'. A loop in 'loops[i]' may or may not have a child loop in
/// 'loops[i+1]'.
static void getMatchedAffineLoops(NestedMatch match,
                                  std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> &loops) {
  getMatchedAffineLoopsRec(match, 0, loops);
}

/// Internal implementation to vectorize affine loops from a single loop nest
/// using an n-D vectorization strategy.
static LogicalResult vectorizeLoopNest(std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> &loops,
                                       const VectorizationStrategy &strategy) {
  assert(loops[0].size() == 1 && "Expected single root loop");
  AffineForOp rootLoop = loops[0][0];
  VectorizationState state(rootLoop.getContext());
  state.builder.setInsertionPointAfter(rootLoop);
  state.strategy = &strategy;

  // Since patterns are recursive, they can very well intersect.
  // Since we do not want a fully greedy strategy in general, we decouple
  // pattern matching, from profitability analysis, from application.
  // As a consequence we must check that each root pattern is still
  // vectorizable. If a pattern is not vectorizable anymore, we just skip it.
  // Implement a non-greedy profitability analysis that keeps only
  // non-intersecting patterns.
  if (!isVectorizableLoopBody(rootLoop, vectorTransferPattern())) {
    LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ loop is not vectorizable");
    return failure();
  }

  //////////////////////////////////////////////////////////////////////////////
  // Vectorize the scalar loop nest following a topological order. A new vector
  // loop nest with the vectorized operations is created along the process. If
  // vectorization succeeds, the scalar loop nest is erased. If vectorization
  // fails, the vector loop nest is erased and the scalar loop nest is not
  // modified.
  //////////////////////////////////////////////////////////////////////////////

  auto opVecResult = rootLoop.walk<WalkOrder::PreOrder>([&state](Operation *op) {
    LLVM_DEBUG(dbgs() << "[early-vect]+++++ Vectorizing: " << *op);
    Operation *vectorOp = vectorizeOneOperation(op, state);
    if (!vectorOp) {
      LLVM_DEBUG(dbgs() << "[early-vect]+++++ failed vectorizing the operation: " << *op << "\n");
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });
  if (opVecResult.wasInterrupted()) {
    LLVM_DEBUG(dbgs() << "[early-vect]+++++ failed vectorization for: " << rootLoop << "\n");
    // Erase vector loop nest if it was created.
    auto vecRootLoopIt = state.opVectorReplacement.find(rootLoop);
    if (vecRootLoopIt != state.opVectorReplacement.end()) {
      eraseLoopNest(cast<AffineForOp>(vecRootLoopIt->second));
    }

    return failure();
  }
  // Replace results of reduction loops with the scalar values computed using
  // `vector.reduce` or similar ops.
  for (auto resPair : state.loopResultScalarReplacement) {
    resPair.first.replaceAllUsesWith(resPair.second);
  }

  assert(state.opVectorReplacement.count(rootLoop) == 1 && "Expected vector replacement for loop nest");
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ success vectorizing pattern");
  LLVM_DEBUG(dbgs() << "\n[early-vect]+++++ vectorization result:\n" << *state.opVectorReplacement[rootLoop]);

  // Finish this vectorization pattern.
  state.finishVectorizationPattern(rootLoop);
  return success();
}

/// Extracts the matched loops and vectorizes them following a topological
/// order. A new vector loop nest will be created if vectorization succeeds. The
/// original loop nest won't be modified in any case.
static LogicalResult vectorizeRootMatch(NestedMatch m, const VectorizationStrategy &strategy) {
  std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> loopsToVectorize;
  getMatchedAffineLoops(m, loopsToVectorize);
  return vectorizeLoopNest(loopsToVectorize, strategy);
}

/// Traverses all the loop matches and classifies them into intersection
/// buckets. Two matches intersect if any of them encloses the other one. A
/// match intersects with a bucket if the match intersects with the root
/// (outermost) loop in that bucket.
static void computeIntersectionBuckets(
  ArrayRef<NestedMatch> matches, std::vector<SmallVector<NestedMatch, kSmallVectorSizeEight>> &intersectionBuckets) {
  assert(intersectionBuckets.empty() && "Expected empty output");
  // Keeps track of the root (outermost) loop of each bucket.
  SmallVector<AffineForOp, kSmallVectorSizeEight> bucketRoots;

  for (const NestedMatch &match : matches) {
    AffineForOp matchRoot = cast<AffineForOp>(match.getMatchedOperation());
    bool intersects = false;
    for (size_t i = 0, end = intersectionBuckets.size(); i < end; ++i) {
      AffineForOp bucketRoot = bucketRoots[i];
      // Add match to the bucket if the bucket root encloses the match root.
      if (bucketRoot->isAncestor(matchRoot)) {
        intersectionBuckets[i].push_back(match);
        intersects = true;
        break;
      }
      // Add match to the bucket if the match root encloses the bucket root. The
      // match root becomes the new bucket root.
      if (matchRoot->isAncestor(bucketRoot)) {
        bucketRoots[i] = matchRoot;
        intersectionBuckets[i].push_back(match);
        intersects = true;
        break;
      }
    }

    // Match doesn't intersect with any existing bucket. Create a new bucket for
    // it.
    if (!intersects) {
      bucketRoots.push_back(matchRoot);
      intersectionBuckets.emplace_back();
      intersectionBuckets.back().push_back(match);
    }
  }
}

namespace {
struct VectorizeLoopsParams {
  ArrayRef<int64_t> fastestVaryingPattern;
  const ReductionLoopMap &reductionLoops;
};
}  // namespace

/// Internal implementation to vectorize affine loops in 'loops' using the n-D
/// vectorization factors in 'vectorSizes'. By default, each vectorization
/// factor is applied inner-to-outer to the loops of each loop nest.
/// 'fastestVaryingPattern' can be optionally used to provide a different loop
/// vectorization order. `reductionLoops` can be provided to specify loops which
/// can be vectorized along the reduction dimension.
static void vectorizeLoops(Operation *parentOp, const DenseSet<Operation *> &loops, ArrayRef<int64_t> vectorSizes,
                           const VectorizeLoopsParams &params) {
  auto &[fastestVaryingPattern, reductionLoops] = params;
  assert((reductionLoops.empty() || vectorSizes.size() == 1) &&
         "Vectorizing reductions is supported only for 1-D vectors");

  // Compute 1-D, 2-D or 3-D loop pattern to be matched on the target loops.
  std::optional<NestedPattern> pattern = makePattern(loops, vectorSizes.size(), fastestVaryingPattern);
  if (!pattern) {
    LLVM_DEBUG(dbgs() << "\n[early-vect] pattern couldn't be computed\n");
    return;
  }

  LLVM_DEBUG(dbgs() << "\n******************************************");
  LLVM_DEBUG(dbgs() << "\n******************************************");
  LLVM_DEBUG(dbgs() << "\n[early-vect] new pattern on parent op\n");
  LLVM_DEBUG(dbgs() << *parentOp << "\n");

  unsigned patternDepth = pattern->getDepth();

  // Compute all the pattern matches and classify them into buckets of
  // intersecting matches.
  SmallVector<NestedMatch, kSmallVectorSizeThirtyTwo> allMatches;
  pattern->match(parentOp, &allMatches);
  std::vector<SmallVector<NestedMatch, kSmallVectorSizeEight>> intersectionBuckets;
  computeIntersectionBuckets(allMatches, intersectionBuckets);

  // Iterate over all buckets and vectorize the matches eagerly. We can only
  // vectorize one match from each bucket since all the matches within a bucket
  // intersect.
  for (auto &intersectingMatches : intersectionBuckets) {
    for (NestedMatch &match : intersectingMatches) {
      VectorizationStrategy strategy;
      // Depending on profitability, elect to reduce the vector size.
      strategy.vectorSizes.assign(vectorSizes.begin(), vectorSizes.end());
      strategy.reductionLoops = reductionLoops;
      if (failed(analyzeProfitability(match.getMatchedChildren(), 1, patternDepth, &strategy))) {
        continue;
      }
      vectorizeLoopIfProfitable(match.getMatchedOperation(), 0, patternDepth, &strategy);
      // Vectorize match. Skip the rest of intersecting matches in the bucket if
      // vectorization succeeded.
      // If pattern does not apply, report it; alter the cost/benefit.
      // Some diagnostics if failure to vectorize occurs.
      if (succeeded(vectorizeRootMatch(match, strategy))) {
        break;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "\n");
}

/// Applies vectorization to the current function by searching over a bunch of
/// predetermined patterns.
void VectorizeAKG::runOnOperation() {
  func::FuncOp f = getOperation();
  if (!fastestVaryingPattern.empty() && fastestVaryingPattern.size() != vectorSizes.size()) {
    f.emitRemark(
      "Fastest varying pattern specified with different size than "
      "the vector size.");
    return signalPassFailure();
  }

  if (vectorizeReductions && vectorSizes.size() != 1) {
    f.emitError("Vectorizing reductions is supported only for 1-D vectors.");
    return signalPassFailure();
  }

  if (llvm::any_of(vectorSizes, [](int64_t size) { return size <= 0; })) {
    f.emitError("Vectorization factor must be greater than zero.");
    return signalPassFailure();
  }

  DenseSet<Operation *> parallelLoops;
  ReductionLoopMap reductionLoops;

  // If 'vectorize-reduction=true' is provided, we also populate the
  // `reductionLoops` map.
  if (vectorizeReductions) {
    f.walk([&parallelLoops, &reductionLoops](AffineForOp loop) {
      SmallVector<LoopReduction, kSmallVectorSizeTwo> reductions;
      if (isLoopParallelAKG(loop, &reductions)) {
        parallelLoops.insert(loop);
        // If it's not a reduction loop, adding it to the map is not necessary.
        if (!reductions.empty()) {
          reductionLoops[loop] = reductions;
        }
      }
    });
  } else {
    f.walk([&parallelLoops](AffineForOp loop) {
      if (isLoopParallelAKG(loop)) {
        parallelLoops.insert(loop);
      }
    });
  }

  // Thread-safe RAII local context, BumpPtrAllocator freed on exit.
  NestedPatternContext mlContext;
  vectorizeLoops(f, parallelLoops, vectorSizes, {fastestVaryingPattern, reductionLoops});
}

namespace mlir {
namespace affine {

std::optional<NestedPattern> makePatternAKG(const DenseSet<Operation *> &parallelLoops, int vectorRank,
                                            ArrayRef<int64_t> fastestVaryingPattern) {
  return makePattern(parallelLoops, vectorRank, fastestVaryingPattern);
}

void computeIntersectionBucketsAKG(ArrayRef<NestedMatch> matches,
                                   std::vector<SmallVector<NestedMatch, kSmallVectorSizeEight>> &intersectionBuckets) {
  computeIntersectionBuckets(matches, intersectionBuckets);
}

LogicalResult analyzeProfitabilityAKG(ArrayRef<NestedMatch> matches, unsigned depthInPattern, unsigned patternDepth,
                                      VectorizationStrategy *strategy) {
  return analyzeProfitability(matches, depthInPattern, patternDepth, strategy);
}

void vectorizeLoopIfProfitableAKG(Operation *loop, unsigned depthInPattern, unsigned patternDepth,
                                  VectorizationStrategy *strategy) {
  vectorizeLoopIfProfitable(loop, depthInPattern, patternDepth, strategy);
}

void getMatchedAffineLoopsAKG(NestedMatch match, std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> &loops) {
  getMatchedAffineLoops(match, loops);
}

LogicalResult vectorizeLoopNestAKG(std::vector<SmallVector<AffineForOp, kSmallVectorSizeTwo>> &loops,
                                   const VectorizationStrategy &strategy) {
  return vectorizeLoopNest(loops, strategy);
}

}  // namespace affine

namespace vector {

Operation *vectorizeOneOperationAKG(Operation *op, VectorizationState &state) {
  return vectorizeOneOperation(op, state);
}

}  // namespace vector
}  // namespace mlir
