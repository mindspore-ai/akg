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
//===--------------- SuperVectorize.h - vectorize op ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SUPERVECTORIZE_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SUPERVECTORIZE_H_

#include "mlir/Dialect/Affine/Analysis/NestedMatcher.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace affine {
/// It is convenient to reference static functions in AKG.
std::optional<NestedPattern> makePatternAKG(const DenseSet<Operation *> &parallelLoops, int vectorRank,
                                            ArrayRef<int64_t> fastestVaryingPattern = {});

void computeIntersectionBucketsAKG(ArrayRef<NestedMatch> matches,
                                   std::vector<SmallVector<NestedMatch, 8>> &intersectionBuckets);

LogicalResult analyzeProfitabilityAKG(ArrayRef<NestedMatch> matches, unsigned depthInPattern, unsigned patternDepth,
                                      VectorizationStrategy *strategy);

void vectorizeLoopIfProfitableAKG(Operation *loop, unsigned depthInPattern, unsigned patternDepth,
                                  VectorizationStrategy *strategy);

void getMatchedAffineLoopsAKG(NestedMatch match, std::vector<SmallVector<AffineForOp, 2>> &loops);

LogicalResult vectorizeLoopNestAKG(std::vector<SmallVector<AffineForOp, 2>> &loops,
                                   const VectorizationStrategy &strategy);
}  // namespace affine

namespace vector {
struct VectorizationState {
  VectorizationState(MLIRContext *context) : builder(context) {}

  /// Registers the vector replacement of a scalar operation and its result
  /// values. Both operations must have the same number of results.
  ///
  /// This utility is used to register the replacement for the vast majority of
  /// the vectorized operations.
  ///
  /// Example:
  ///   * 'replaced': %0 = arith.addf %1, %2 : f32
  ///   * 'replacement': %0 = arith.addf %1, %2 : vector<128xf32>
  void registerOpVectorReplacement(Operation *replaced, Operation *replacement);

  /// Registers the vector replacement of a scalar value. The replacement
  /// operation should have a single result, which replaces the scalar value.
  ///
  /// This utility is used to register the vector replacement of block arguments
  /// and operation results which are not directly vectorized (i.e., their
  /// scalar version still exists after vectorization), like uniforms.
  ///
  /// Example:
  ///   * 'replaced': block argument or operation outside of the vectorized
  ///     loop.
  ///   * 'replacement': %0 = vector.broadcast %1 : f32 to vector<128xf32>
  void registerValueVectorReplacement(Value replaced, Operation *replacement);

  /// Registers the vector replacement of a block argument (e.g., iter_args).
  ///
  /// Example:
  ///   * 'replaced': 'iter_arg' block argument.
  ///   * 'replacement': vectorized 'iter_arg' block argument.
  void registerBlockArgVectorReplacement(BlockArgument replaced, BlockArgument replacement);

  /// Registers the scalar replacement of a scalar value. 'replacement' must be
  /// scalar. Both values must be block arguments. Operation results should be
  /// replaced using the 'registerOp*' utilitites.
  ///
  /// This utility is used to register the replacement of block arguments
  /// that are within the loop to be vectorized and will continue being scalar
  /// within the vector loop.
  ///
  /// Example:
  ///   * 'replaced': induction variable of a loop to be vectorized.
  ///   * 'replacement': new induction variable in the new vector loop.
  void registerValueScalarReplacement(BlockArgument replaced, BlockArgument replacement);

  /// Registers the scalar replacement of a scalar result returned from a
  /// reduction loop. 'replacement' must be scalar.
  ///
  /// This utility is used to register the replacement for scalar results of
  /// vectorized reduction loops with iter_args.
  ///
  /// Example 2:
  ///   * 'replaced': %0 = affine.for %i = 0 to 512 iter_args(%x = ...) -> (f32)
  ///   * 'replacement': %1 = vector.reduction <add>, %0 : vector<4xf32> into
  ///   f32
  void registerLoopResultScalarReplacement(Value replaced, Value replacement);

  /// Returns in 'replacedVals' the scalar replacement for values in
  /// 'inputVals'.
  void getScalarValueReplacementsFor(ValueRange inputVals, SmallVectorImpl<Value> &replacedVals);

  /// Erases the scalar loop nest after its successful vectorization.
  void finishVectorizationPattern(affine::AffineForOp rootLoop);

  // Used to build and insert all the new operations created. The insertion
  // point is preserved and updated along the vectorization process.
  OpBuilder builder;

  // Maps input scalar operations to their vector counterparts.
  DenseMap<Operation *, Operation *> opVectorReplacement;
  // Maps input scalar values to their vector counterparts.
  IRMapping valueVectorReplacement;
  // Maps input scalar values to their new scalar counterparts in the vector
  // loop nest.
  IRMapping valueScalarReplacement;
  // Maps results of reduction loops to their new scalar counterparts.
  DenseMap<Value, Value> loopResultScalarReplacement;

  // Maps the newly created vector loops to their vector dimension.
  DenseMap<Operation *, unsigned> vecLoopToVecDim;

  // Maps the new vectorized loops to the corresponding vector masks if it is
  // required.
  DenseMap<Operation *, Value> vecLoopToMask;

  // The strategy drives which loop to vectorize by which amount.
  const affine::VectorizationStrategy *strategy = nullptr;

 private:
  /// Internal implementation to map input scalar values to new vector or scalar
  /// values.
  void registerValueVectorReplacementImpl(Value replaced, Value replacement);
  void registerValueScalarReplacementImpl(Value replaced, Value replacement);
};

Operation *vectorizeOneOperationAKG(Operation *op, VectorizationState &state);

}  // namespace vector
}  // namespace mlir
#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_TRANSFORMS_SUPERVECTORIZE_H_
