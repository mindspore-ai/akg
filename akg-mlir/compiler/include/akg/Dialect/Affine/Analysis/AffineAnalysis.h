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
//===- AffineAnalysis.h - analyses for affine structures --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes for methods that perform analysis
// involving affine structures (AffineExprStorage, AffineMap, IntegerSet, etc.)
// and other IR structures that in turn use these.
//
//===----------------------------------------------------------------------===//

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AFFINEANALYSIS_H
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AFFINEANALYSIS_H

#include <optional>
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"

namespace mlir {
class Operation;

namespace affine {
class AffineApplyOp;
class AffineForOp;
class AffineValueMap;
class FlatAffineRelation;
class FlatAffineValueConstraints;

/// Returns true if `forOp' is a parallel loop. If `parallelReductions` is
/// provided, populates it with descriptors of the parallelizable reductions and
/// treats them as not preventing parallelization.
bool isLoopParallelAKG(AffineForOp forOp, SmallVectorImpl<LoopReduction> *parallelReductions = nullptr);

/// Returns true if `forOp' doesn't have memory dependences preventing
/// parallelization. Memrefs that are allocated inside `forOp` do not impact its
/// dependences and parallelism. This function does not check iter_args (for
/// values other than memref types) and should be used only as a building block
/// for complete parallelism-checking functions.
bool isLoopMemoryParallelAKG(AffineForOp forOp);

/// Encapsulates a memref load or store access information.
struct AKGMemRefAccess {
  Value memref;
  Operation *opInst;
  SmallVector<Value, 4> indices;

  /// Constructs a MemRefAccess from a load or store operation.
  // TODO: add accessors to standard op's load, store, DMA op's to return
  // MemRefAccess, i.e., loadOp->getAccess(), dmaOp->getRead/WriteAccess.
  explicit AKGMemRefAccess(Operation *opInst);

  // Returns the rank of the memref associated with this access.
  unsigned getRank() const;
  // Returns true if this access is of a store op.
  bool isStore() const;

  /// Creates an access relation for the access. An access relation maps
  /// elements of an iteration domain to the element(s) of an array domain
  /// accessed by that iteration of the associated statement through some array
  /// reference. For example, given the MLIR code:
  ///
  /// affine.for %i0 = 0 to 10 {
  ///   affine.for %i1 = 0 to 10 {
  ///     %a = affine.load %arr[%i0 + %i1, %i0 + 2 * %i1] : memref<100x100xf32>
  ///   }
  /// }
  ///
  /// The access relation, assuming that the memory locations for %arr are
  /// represented as %m0, %m1 would be:
  ///
  ///   (%i0, %i1) -> (%m0, %m1)
  ///   %m0 = %i0 + %i1
  ///   %m1 = %i0 + 2 * %i1
  ///   0  <= %i0 < 10
  ///   0  <= %i1 < 10
  ///
  /// Returns failure for yet unimplemented/unsupported cases (see docs of
  /// mlir::getIndexSet and mlir::getRelationFromMap for these cases).
  LogicalResult getAccessRelation(FlatAffineRelation &accessRel) const;

  /// Populates 'accessMap' with composition of AffineApplyOps reachable from
  /// 'indices'.
  void getAccessMap(AffineValueMap *accessMap) const;

  /// Equal if both affine accesses can be proved to be equivalent at compile
  /// time (considering the memrefs, their respective affine access maps  and
  /// operands). The equality of access functions + operands is checked by
  /// subtracting fully composed value maps, and then simplifying the difference
  /// using the expression flattener.
  /// TODO: this does not account for aliasing of memrefs.
  bool operator==(const AKGMemRefAccess &rhs) const;
  bool operator!=(const AKGMemRefAccess &rhs) const { return !(*this == rhs); }
};

DependenceResult checkMemrefAccessDependenceAKG(const AKGMemRefAccess &srcAccess, const AKGMemRefAccess &dstAccess,
                                                unsigned loopDepth,
                                                FlatAffineValueConstraints *dependenceConstraints = nullptr,
                                                SmallVector<DependenceComponent, 2> *dependenceComponents = nullptr,
                                                bool allowRAR = false);

}  // namespace affine
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_AFFINEANALYSIS_H
