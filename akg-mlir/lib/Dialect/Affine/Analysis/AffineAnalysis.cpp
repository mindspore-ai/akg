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
//===- AffineAnalysis.cpp - Affine structures analysis routines -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements miscellaneous analysis routines for affine structures
// (expressions, maps, sets), and other utilities relying on such analysis.
//
//===----------------------------------------------------------------------===//
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"

#include <algorithm>
#include <optional>
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#define DEBUG_TYPE "akg-affine-analysis"

namespace mlir {
namespace affine {

using presburger::BoundType;
using presburger::Identifier;
using presburger::IntegerPolyhedron;
using presburger::IntegerRelation;
using presburger::PresburgerSpace;
using presburger::VarKind;

Value getSourceMemRef(Value memrefVal, bool *hasSubView, memref::SubViewOp *firstSubView) {
  if (hasSubView) *hasSubView = false;
  if (firstSubView) *firstSubView = memref::SubViewOp();
  Value current = memrefVal;
  while (true) {
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      if (hasSubView) *hasSubView = true;
      if (firstSubView && !*firstSubView) *firstSubView = subview;
      current = subview.getSource();
    } else if (auto reshape = current.getDefiningOp<memref::ReshapeOp>()) {
      current = reshape.getSource();
    } else if (auto expand = current.getDefiningOp<memref::ExpandShapeOp>()) {
      current = expand.getSrc();
    } else if (auto collapse = current.getDefiningOp<memref::CollapseShapeOp>()) {
      current = collapse.getSrc();
    } else if (auto cast = current.getDefiningOp<memref::ReinterpretCastOp>()) {
      current = cast.getSource();
    } else if (auto memorySpaceCastOp = current.getDefiningOp<memref::MemorySpaceCastOp>()) {
      current = memorySpaceCastOp.getSource();
    } else {
      break;
    }
  }
  return current;
}

/// Returns true if `v` is allocated locally to `enclosingOp` -- i.e., it is
/// allocated by an operation nested within `enclosingOp`.
static bool isLocallyDefined(Value v, Operation *enclosingOp) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp) return false;

  if (hasSingleEffect<MemoryEffects::Allocate>(defOp, v) && enclosingOp->isProperAncestor(defOp)) return true;

  // Aliasing ops.
  auto viewOp = dyn_cast<ViewLikeOpInterface>(defOp);
  return viewOp && isLocallyDefined(viewOp.getViewSource(), enclosingOp);
}

/// Computes the iteration domain for 'op' and populates 'indexSet', which
/// encapsulates the constraints involving loops surrounding 'op' and
/// potentially involving any Function symbols. The dimensional variables in
/// 'indexSet' correspond to the loops surrounding 'op' from outermost to
/// innermost.
static LogicalResult getOpIndexSet(Operation *op, FlatAffineValueConstraints *indexSet) {
  SmallVector<Operation *, 4> ops;
  getEnclosingAffineOps(*op, &ops);
  return getIndexSet(ops, indexSet);
}

bool isLoopMemoryParallelAKG(AffineForOp forOp) {
  // Any memref-typed iteration arguments are treated as serializing.
  if (llvm::any_of(forOp.getResultTypes(), [](Type type) { return isa<BaseMemRefType>(type); })) {
    return false;
  }

  // Collect all load and store ops in loop nest rooted at 'forOp'.
  SmallVector<Operation *, 8> loadAndStoreOps;
  auto walkResult = forOp.walk([&](Operation *op) -> WalkResult {
    if (auto readOp = dyn_cast<AffineReadOpInterface>(op)) {
      // Memrefs that are allocated inside `forOp` need not be considered.
      if (!isLocallyDefined(readOp.getMemRef(), forOp)) loadAndStoreOps.push_back(op);
    } else if (auto writeOp = dyn_cast<AffineWriteOpInterface>(op)) {
      // Filter out stores the same way as above.
      if (!isLocallyDefined(writeOp.getMemRef(), forOp)) loadAndStoreOps.push_back(op);
    } else if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
      if (!isLocallyDefined(transferReadOp.getSource(), forOp)) {
        loadAndStoreOps.push_back(op);
      }
    } else if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op)) {
      if (!isLocallyDefined(transferWriteOp.getSource(), forOp)) {
        loadAndStoreOps.push_back(op);
      }
    } else if (!isa<AffineForOp, AffineYieldOp, AffineIfOp>(op) && !hasSingleEffect<MemoryEffects::Allocate>(op) &&
               !isMemoryEffectFree(op)) {
      // Alloc-like ops inside `forOp` are fine (they don't impact parallelism)
      // as long as they don't escape the loop (which has been checked above).
      return WalkResult::interrupt();
    }

    return WalkResult::advance();
  });

  // Stop early if the loop has unknown ops with side effects.
  if (walkResult.wasInterrupted()) return false;

  // Dep check depth would be number of enclosing loops + 1.
  unsigned depth = getNestingDepth(forOp) + 1;

  // Check dependences between all pairs of ops in 'loadAndStoreOps'.
  for (auto *srcOp : loadAndStoreOps) {
    affine::AKGMemRefAccess srcAccess(srcOp);
    for (auto *dstOp : loadAndStoreOps) {
      affine::AKGMemRefAccess dstAccess(dstOp);
      DependenceResult result = checkMemrefAccessDependenceAKG(srcAccess, dstAccess, depth);
      if (result.value != DependenceResult::NoDependence) return false;
    }
  }
  return true;
}

/// Returns true if `forOp' is a parallel loop. If `parallelReductions` is
/// provided, populates it with descriptors of the parallelizable reductions and
/// treats them as not preventing parallelization.
bool isLoopParallelAKG(AffineForOp forOp, SmallVectorImpl<LoopReduction> *parallelReductions) {
  unsigned numIterArgs = forOp.getNumIterOperands();

  // Loop is not parallel if it has SSA loop-carried dependences and reduction
  // detection is not requested.
  if (numIterArgs > 0 && !parallelReductions) return false;

  // Find supported reductions of requested.
  if (parallelReductions) {
    getSupportedReductions(forOp, *parallelReductions);
    // Return later to allow for identifying all parallel reductions even if the
    // loop is not parallel.
    if (parallelReductions->size() != numIterArgs) return false;
  }

  // Check memory dependences.
  return isLoopMemoryParallelAKG(forOp);
}

// Returns the number of outer loop common to 'src/dstDomain'.
// Loops common to 'src/dst' domains are added to 'commonLoops' if non-null.
static unsigned getNumCommonLoops(const FlatAffineValueConstraints &srcDomain,
                                  const FlatAffineValueConstraints &dstDomain,
                                  SmallVectorImpl<AffineForOp> *commonLoops = nullptr) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops = std::min(srcDomain.getNumDimVars(), dstDomain.getNumDimVars());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if ((!isAffineForInductionVar(srcDomain.getValue(i)) && !isAffineParallelInductionVar(srcDomain.getValue(i))) ||
        (!isAffineForInductionVar(dstDomain.getValue(i)) && !isAffineParallelInductionVar(dstDomain.getValue(i))) ||
        srcDomain.getValue(i) != dstDomain.getValue(i))
      break;
    if (commonLoops != nullptr) commonLoops->push_back(getForInductionVarOwner(srcDomain.getValue(i)));
    ++numCommonLoops;
  }
  if (commonLoops != nullptr) assert(commonLoops->size() == numCommonLoops);
  return numCommonLoops;
}

/// Returns the closest surrounding block common to `opA` and `opB`. `opA` and
/// `opB` should be in the same affine scope. Returns nullptr if such a block
/// does not exist (when the two ops are in different blocks of an op starting
/// an `AffineScope`).
static Block *getCommonBlockInAffineScope(Operation *opA, Operation *opB) {
  // Get the chain of ancestor blocks for the given `MemRefAccess` instance. The
  // chain extends up to and includnig an op that starts an affine scope.
  auto getChainOfAncestorBlocks = [&](Operation *op, SmallVectorImpl<Block *> &ancestorBlocks) {
    Block *currBlock = op->getBlock();
    // Loop terminates when the currBlock is nullptr or its parent operation
    // holds an affine scope.
    while (currBlock && !currBlock->getParentOp()->hasTrait<OpTrait::AffineScope>()) {
      ancestorBlocks.push_back(currBlock);
      currBlock = currBlock->getParentOp()->getBlock();
    }
    assert(currBlock && "parent op starting an affine scope is always expected");
    ancestorBlocks.push_back(currBlock);
  };

  // Find the closest common block.
  SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
  getChainOfAncestorBlocks(opA, srcAncestorBlocks);
  getChainOfAncestorBlocks(opB, dstAncestorBlocks);

  Block *commonBlock = nullptr;
  for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
       i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j]; i--, j--)
    commonBlock = srcAncestorBlocks[i];

  return commonBlock;
}

/// Returns true if the ancestor operation of 'srcAccess' appears before the
/// ancestor operation of 'dstAccess' in their common ancestral block. The
/// operations for `srcAccess` and `dstAccess` are expected to be in the same
/// affine scope and have a common surrounding block within it.
static bool srcAppearsBeforeDstInAncestralBlock(const AKGMemRefAccess &srcAccess, const AKGMemRefAccess &dstAccess) {
  // Get Block common to 'srcAccess.opInst' and 'dstAccess.opInst'.
  Block *commonBlock = getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst);
  assert(commonBlock && "ops expected to have a common surrounding block in affine scope");

  // Check the dominance relationship between the respective ancestors of the
  // src and dst in the Block of the innermost among the common loops.
  Operation *srcOp = commonBlock->findAncestorOpInBlock(*srcAccess.opInst);
  assert(srcOp && "src access op must lie in common block");
  Operation *dstOp = commonBlock->findAncestorOpInBlock(*dstAccess.opInst);
  assert(dstOp && "dest access op must lie in common block");

  // Determine whether dstOp comes after srcOp.
  return srcOp->isBeforeInBlock(dstOp);
}

// Adds ordering constraints to 'dependenceDomain' based on number of loops
// common to 'src/dstDomain' and requested 'loopDepth'.
// Note that 'loopDepth' cannot exceed the number of common loops plus one.
// EX: Given a loop nest of depth 2 with IVs 'i' and 'j':
// *) If 'loopDepth == 1' then one constraint is added: i' >= i + 1
// *) If 'loopDepth == 2' then two constraints are added: i == i' and j' > j + 1
// *) If 'loopDepth == 3' then two constraints are added: i == i' and j == j'
static void addOrderingConstraints(const FlatAffineValueConstraints &srcDomain,
                                   const FlatAffineValueConstraints &dstDomain, unsigned loopDepth,
                                   IntegerRelation *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numSrcDims = srcDomain.getNumDimVars();
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  unsigned numCommonLoopConstraints = std::min(numCommonLoops, loopDepth);
  for (unsigned i = 0; i < numCommonLoopConstraints; ++i) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    if (i == loopDepth - 1) {
      eq[numCols - 1] = -1;
      dependenceDomain->addInequality(eq);
    } else {
      dependenceDomain->addEquality(eq);
    }
  }
}

// Computes distance and direction vectors in 'dependences', by adding
// variables to 'dependenceDomain' which represent the difference of the IVs,
// eliminating all other variables, and reading off distance vectors from
// equality constraints (if possible), and direction vectors from inequalities.
static void computeDirectionVector(const FlatAffineValueConstraints &srcDomain,
                                   const FlatAffineValueConstraints &dstDomain, unsigned loopDepth,
                                   IntegerPolyhedron *dependenceDomain,
                                   SmallVector<DependenceComponent, 2> *dependenceComponents) {
  // Find the number of common loops shared by src and dst accesses.
  SmallVector<AffineForOp, 4> commonLoops;
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain, &commonLoops);
  if (numCommonLoops == 0) return;
  // Compute direction vectors for requested loop depth.
  unsigned numIdsToEliminate = dependenceDomain->getNumVars();
  // Add new variables to 'dependenceDomain' to represent the direction
  // constraints for each shared loop.
  dependenceDomain->insertVar(VarKind::SetDim, /*pos=*/0,
                              /*num=*/numCommonLoops);

  // Add equality constraints for each common loop, setting newly introduced
  // variable at column 'j' to the 'dst' IV minus the 'src IV.
  SmallVector<int64_t, 4> eq;
  eq.resize(dependenceDomain->getNumCols());
  unsigned numSrcDims = srcDomain.getNumDimVars();
  // Constraint variables format:
  // [num-common-loops][num-src-dim-ids][num-dst-dim-ids][num-symbols][constant]
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[j] = 1;
    eq[j + numCommonLoops] = 1;
    eq[j + numCommonLoops + numSrcDims] = -1;
    dependenceDomain->addEquality(eq);
  }

  // Eliminate all variables other than the direction variables just added.
  dependenceDomain->projectOut(numCommonLoops, numIdsToEliminate);

  // Scan each common loop variable column and set direction vectors based
  // on eliminated constraint system.
  dependenceComponents->resize(numCommonLoops);
  for (unsigned j = 0; j < numCommonLoops; ++j) {
    (*dependenceComponents)[j].op = commonLoops[j].getOperation();
    auto lbConst = dependenceDomain->getConstantBound64(BoundType::LB, j);
    (*dependenceComponents)[j].lb = lbConst.value_or(std::numeric_limits<int64_t>::min());
    auto ubConst = dependenceDomain->getConstantBound64(BoundType::UB, j);
    (*dependenceComponents)[j].ub = ubConst.value_or(std::numeric_limits<int64_t>::max());
  }
}

LogicalResult AKGMemRefAccess::getAccessRelation(IntegerRelation &rel) const {
  // Create set corresponding to domain of access.
  FlatAffineValueConstraints domain;
  if (failed(getOpIndexSet(opInst, &domain))) return failure();

  // Get access relation from access map.
  AffineValueMap accessValueMap;
  getAccessMap(&accessValueMap);
  if (failed(getRelationFromMap(accessValueMap, rel))) return failure();

  // Merge and align domain ids of `rel` with ids of `domain`. Since the domain
  // of the access map is a subset of the domain of access, the domain ids of
  // `rel` are guaranteed to be a subset of ids of `domain`.
  unsigned inserts = 0;
  for (unsigned i = 0, e = domain.getNumDimVars(); i < e; ++i) {
    const Identifier domainIdi = Identifier(domain.getValue(i));
    const Identifier *findBegin = rel.getIds(VarKind::SetDim).begin() + i;
    const Identifier *findEnd = rel.getIds(VarKind::SetDim).end();
    const Identifier *itr = std::find(findBegin, findEnd, domainIdi);
    if (itr != findEnd) {
      rel.swapVar(i, i + std::distance(findBegin, itr));
    } else {
      ++inserts;
      rel.insertVar(VarKind::SetDim, i);
      rel.setId(VarKind::SetDim, i, domainIdi);
    }
  }

  // Append domain constraints to `rel`.
  IntegerRelation domainRel = domain;
  if (rel.getSpace().isUsingIds() && !domainRel.getSpace().isUsingIds()) domainRel.resetIds();
  domainRel.appendVar(VarKind::Range, accessValueMap.getNumResults());
  domainRel.mergeAndAlignSymbols(rel);
  domainRel.mergeLocalVars(rel);
  rel.append(domainRel);

  rel.convertVarKind(VarKind::SetDim, 0, accessValueMap.getNumDims() + inserts, VarKind::Domain);

  return success();
}

// Populates 'accessMap' with composition of AffineApplyOps reachable from
// indices of MemRefAccess.
void AKGMemRefAccess::getAccessMap(AffineValueMap *accessMap) const {
  // Get affine map from AffineLoad/Store.
  AffineMap map;
  if (auto loadOp = dyn_cast<AffineReadOpInterface>(opInst)) {
    map = loadOp.getAffineMap();
  } else if (auto storeOp = dyn_cast<AffineWriteOpInterface>(opInst)) {
    map = storeOp.getAffineMap();
  } else if (auto readOp = dyn_cast<vector::TransferReadOp>(opInst)) {
    map = readOp.getPermutationMap();
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(opInst)) {
    map = writeOp.getPermutationMap();
  }

  // update map
  // (d0) -> (0) ==> (d0) -> (d0)
  // (d0, d1) -> (d1) ==> (d0, d1) -> (d0, d1)
  if (isa<vector::TransferReadOp, vector::TransferWriteOp>(opInst)) {
    auto resultDims = map.getNumResults();
    for (size_t i = 0; i < resultDims; ++i) {
      map = map.dropResult(i);
    }
    auto dims = map.getNumDims();
    for (size_t i = 0; i < dims; ++i) {
      auto insertDimExpr = getAffineDimExpr(i, opInst->getContext());
      map = map.insertResult(insertDimExpr, i);
    }
  }

  SmallVector<Value, 8> operands(indices.begin(), indices.end());
  fullyComposeAffineMapAndOperands(&map, &operands);
  map = simplifyAffineMap(map);
  canonicalizeMapAndOperands(&map, &operands);
  accessMap->reset(map, operands);
}

// Builds a flat affine constraint system to check if there exists a dependence
// between memref accesses 'srcAccess' and 'dstAccess'.
// Returns 'NoDependence' if the accesses can be definitively shown not to
// access the same element.
// Returns 'HasDependence' if the accesses do access the same element.
// Returns 'Failure' if an error or unsupported case was encountered.
// If a dependence exists, returns in 'dependenceComponents' a direction
// vector for the dependence, with a component for each loop IV in loops
// common to both accesses (see Dependence in AffineAnalysis.h for details).
//
// The memref access dependence check is comprised of the following steps:
// *) Build access relation for each access. An access relation maps elements
//    of an iteration domain to the element(s) of an array domain accessed by
//    that iteration of the associated statement through some array reference.
// *) Compute the dependence relation by composing access relation of
//    `srcAccess` with the inverse of access relation of `dstAccess`.
//    Doing this builds a relation between iteration domain of `srcAccess`
//    to the iteration domain of `dstAccess` which access the same memory
//    location.
// *) Add ordering constraints for `srcAccess` to be accessed before
//    `dstAccess`.
//
// This method builds a constraint system with the following column format:
//
//  [src-dim-variables, dst-dim-variables, symbols, constant]
//
// For example, given the following MLIR code with "source" and "destination"
// accesses to the same memref label, and symbols %M, %N, %K:
//
//   affine.for %i0 = 0 to 100 {
//     affine.for %i1 = 0 to 50 {
//       %a0 = affine.apply
//         (d0, d1) -> (d0 * 2 - d1 * 4 + s1, d1 * 3 - s0) (%i0, %i1)[%M, %N]
//       // Source memref access.
//       store %v0, %m[%a0#0, %a0#1] : memref<4x4xf32>
//     }
//   }
//
//   affine.for %i2 = 0 to 100 {
//     affine.for %i3 = 0 to 50 {
//       %a1 = affine.apply
//         (d0, d1) -> (d0 * 7 + d1 * 9 - s1, d1 * 11 + s0) (%i2, %i3)[%K, %M]
//       // Destination memref access.
//       %v1 = load %m[%a1#0, %a1#1] : memref<4x4xf32>
//     }
//   }
//
// The access relation for `srcAccess` would be the following:
//
//   [src_dim0, src_dim1, mem_dim0, mem_dim1,  %N,   %M,  const]
//       2        -4       -1         0         1     0     0     = 0
//       0         3        0        -1         0    -1     0     = 0
//       1         0        0         0         0     0     0    >= 0
//      -1         0        0         0         0     0     100  >= 0
//       0         1        0         0         0     0     0    >= 0
//       0        -1        0         0         0     0     50   >= 0
//
//  The access relation for `dstAccess` would be the following:
//
//   [dst_dim0, dst_dim1, mem_dim0, mem_dim1,  %M,   %K,  const]
//       7         9       -1         0        -1     0     0     = 0
//       0         11       0        -1         0    -1     0     = 0
//       1         0        0         0         0     0     0    >= 0
//      -1         0        0         0         0     0     100  >= 0
//       0         1        0         0         0     0     0    >= 0
//       0        -1        0         0         0     0     50   >= 0
//
//  The equalities in the above relations correspond to the access maps while
//  the inequalities corresspond to the iteration domain constraints.
//
// The dependence relation formed:
//
//   [src_dim0, src_dim1, dst_dim0, dst_dim1,  %M,   %N,   %K,  const]
//      2         -4        -7        -9        1     1     0     0    = 0
//      0          3         0        -11      -1     0     1     0    = 0
//       1         0         0         0        0     0     0     0    >= 0
//      -1         0         0         0        0     0     0     100  >= 0
//       0         1         0         0        0     0     0     0    >= 0
//       0        -1         0         0        0     0     0     50   >= 0
//       0         0         1         0        0     0     0     0    >= 0
//       0         0        -1         0        0     0     0     100  >= 0
//       0         0         0         1        0     0     0     0    >= 0
//       0         0         0        -1        0     0     0     50   >= 0
//
//
DependenceResult checkMemrefAccessDependenceAKG(const AKGMemRefAccess &srcAccess, const AKGMemRefAccess &dstAccess,
                                                unsigned loopDepth, FlatAffineValueConstraints *dependenceConstraints,
                                                SmallVector<DependenceComponent, 2> *dependenceComponents,
                                                bool allowRAR, bool checkSrcBeforeDst) {
  LLVM_DEBUG(llvm::dbgs() << "Checking for dependence at depth: " << Twine(loopDepth) << " between:\n";);
  LLVM_DEBUG(srcAccess.opInst->dump());
  LLVM_DEBUG(dstAccess.opInst->dump());

  // Return 'NoDependence' if these accesses do not access the same memref.
  if (srcAccess.memref != dstAccess.memref) return DependenceResult::NoDependence;

  // Return 'NoDependence' if one of these accesses is not an
  // AffineWriteOpInterface.
  if (!allowRAR && !isa<AffineWriteOpInterface, vector::TransferWriteOp>(srcAccess.opInst) &&
      !isa<AffineWriteOpInterface, vector::TransferWriteOp>(dstAccess.opInst))
    return DependenceResult::NoDependence;

  // We can't analyze further if the ops lie in different affine scopes or have
  // no common block in an affine scope.
  if (getAffineScope(srcAccess.opInst) != getAffineScope(dstAccess.opInst)) return DependenceResult::Failure;
  if (!getCommonBlockInAffineScope(srcAccess.opInst, dstAccess.opInst)) return DependenceResult::Failure;

  // Create access relation from each MemRefAccess.
  PresburgerSpace space = PresburgerSpace::getRelationSpace();
  IntegerRelation srcRel(space), dstRel(space);
  if (failed(srcAccess.getAccessRelation(srcRel))) return DependenceResult::Failure;
  if (failed(dstAccess.getAccessRelation(dstRel))) return DependenceResult::Failure;

  FlatAffineValueConstraints srcDomain = srcRel.getDomainSet();
  FlatAffineValueConstraints dstDomain = dstRel.getDomainSet();

  // Return 'NoDependence' if loopDepth > numCommonLoops and if the ancestor
  // operation of 'srcAccess' does not properly dominate the ancestor
  // operation of 'dstAccess' in the same common operation block.
  // Note: this check is skipped if 'allowRAR' is true, because RAR deps
  // can exist irrespective of lexicographic ordering b/w src and dst.
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  assert(loopDepth <= numCommonLoops + 1);
  if (checkSrcBeforeDst && !allowRAR && loopDepth > numCommonLoops &&
      !srcAppearsBeforeDstInAncestralBlock(srcAccess, dstAccess)) {
    return DependenceResult::NoDependence;
  }

  // Compute the dependence relation by composing `srcRel` with the inverse of
  // `dstRel`. Doing this builds a relation between iteration domain of
  // `srcAccess` to the iteration domain of `dstAccess` which access the same
  // memory locations.
  dstRel.inverse();
  // For 0-d spaces, there will be no IDs. Enable if that's the case.
  // This ensures Value identifiers survive through mergeAndCompose.
  if (!dstRel.getSpace().isUsingIds()) dstRel.resetIds();
  if (!srcRel.getSpace().isUsingIds()) srcRel.resetIds();
  dstRel.mergeAndCompose(srcRel);
  dstRel.convertVarKind(VarKind::Domain, 0, dstRel.getNumDomainVars(), VarKind::Range, 0);
  IntegerPolyhedron dependenceDomain(dstRel);

  // Add 'src' happens before 'dst' ordering constraints.
  if (checkSrcBeforeDst) addOrderingConstraints(srcDomain, dstDomain, loopDepth, &dependenceDomain);

  // Return 'NoDependence' if the solution space is empty: no dependence.
  if (dependenceDomain.isEmpty()) return DependenceResult::NoDependence;

  // Compute dependence direction vector and return true.
  if (dependenceComponents != nullptr)
    computeDirectionVector(srcDomain, dstDomain, loopDepth, &dependenceDomain, dependenceComponents);

  LLVM_DEBUG(llvm::dbgs() << "Dependence polyhedron:\n");
  LLVM_DEBUG(dependenceDomain.dump());

  FlatAffineValueConstraints result(dependenceDomain);
  if (dependenceConstraints) *dependenceConstraints = result;
  return DependenceResult::HasDependence;
}

// Constructs  MemRefAccess populating it with the memref, its indices and
// opinst from 'loadOrStoreOpInst'.
AKGMemRefAccess::AKGMemRefAccess(Operation *loadOrStoreOpInst) {
  opInst = loadOrStoreOpInst;
  if (auto readOp = dyn_cast<vector::TransferReadOp>(loadOrStoreOpInst)) {
    Value raw = readOp.getSource();
    memref = getSourceMemRef(raw);
    llvm::append_range(indices, readOp.getIndices());
  } else if (auto writeOp = dyn_cast<vector::TransferWriteOp>(loadOrStoreOpInst)) {
    Value raw = writeOp.getSource();
    memref = getSourceMemRef(raw);
    llvm::append_range(indices, writeOp.getIndices());
  } else if (auto loadOp = dyn_cast<AffineReadOpInterface>(loadOrStoreOpInst)) {
    Value raw = loadOp.getMemRef();
    memref = getSourceMemRef(raw);
    llvm::append_range(indices, loadOp.getMapOperands());
  } else {
    assert(isa<AffineWriteOpInterface>(loadOrStoreOpInst) && "Affine read/write op or Vector read/write op expected");
    auto storeOp = cast<AffineWriteOpInterface>(loadOrStoreOpInst);
    Value raw = storeOp.getMemRef();
    memref = getSourceMemRef(raw);
    llvm::append_range(indices, storeOp.getMapOperands());
  }
}

// Adds loop IV bounds to 'cst' for loop IVs not found in 'ivs'.
static LogicalResult addMissingLoopIVBounds(SmallPtrSet<Value, 8> &ivs, FlatAffineValueConstraints *cst) {
  for (unsigned i = 0, e = cst->getNumDimVars(); i < e; ++i) {
    auto value = cst->getValue(i);
    if (ivs.count(value) == 0) {
      assert(isAffineForInductionVar(value));
      auto loop = getForInductionVarOwner(value);
      if (failed(cst->addAffineForOpDomain(loop))) return failure();
    }
  }
  return success();
}

static LogicalResult mergeSliceStateIntoUnion(ComputationSliceState &tmpSliceState,
                                              FlatAffineValueConstraints &sliceUnionCst) {
  if (sliceUnionCst.getNumDimAndSymbolVars() == 0) {
    // Initialize 'sliceUnionCst' with the bounds computed in previous step.
    if (failed(tmpSliceState.getAsConstraints(&sliceUnionCst))) {
      LLVM_DEBUG(llvm::dbgs() << "Unable to compute slice bound constraints\n");
      return failure();
    }
    assert(sliceUnionCst.getNumDimAndSymbolVars() > 0);
    return success();
  }

  // Compute constraints for 'tmpSliceState' in 'tmpSliceCst'.
  FlatAffineValueConstraints tmpSliceCst;
  if (failed(tmpSliceState.getAsConstraints(&tmpSliceCst))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute slice bound constraints\n");
    return failure();
  }

  // Align coordinate spaces of 'sliceUnionCst' and 'tmpSliceCst' if needed.
  if (!sliceUnionCst.areVarsAlignedWithOther(tmpSliceCst)) {
    // Pre-constraint var alignment: record loop IVs used in each constraint
    // system.
    SmallPtrSet<Value, 8> sliceUnionIVs;
    for (unsigned k = 0, l = sliceUnionCst.getNumDimVars(); k < l; ++k) sliceUnionIVs.insert(sliceUnionCst.getValue(k));
    SmallPtrSet<Value, 8> tmpSliceIVs;
    for (unsigned k = 0, l = tmpSliceCst.getNumDimVars(); k < l; ++k) tmpSliceIVs.insert(tmpSliceCst.getValue(k));

    sliceUnionCst.mergeAndAlignVarsWithOther(/*offset=*/0, &tmpSliceCst);

    // Post-constraint var alignment: add loop IV bounds missing after
    // var alignment to constraint systems. This can occur if one constraint
    // system uses an loop IV that is not used by the other. The call
    // to unionBoundingBox below expects constraints for each Loop IV, even
    // if they are the unsliced full loop bounds added here.
    if (failed(addMissingLoopIVBounds(sliceUnionIVs, &sliceUnionCst))) return failure();
    if (failed(addMissingLoopIVBounds(tmpSliceIVs, &tmpSliceCst))) return failure();
  }
  // Compute union bounding box of 'sliceUnionCst' and 'tmpSliceCst'.
  if (sliceUnionCst.getNumLocalVars() > 0 || tmpSliceCst.getNumLocalVars() > 0 ||
      failed(sliceUnionCst.unionBoundingBox(tmpSliceCst))) {
    LLVM_DEBUG(llvm::dbgs() << "Unable to compute union bounding box of slice bounds\n");
    return failure();
  }
  return success();
}

static LogicalResult processOpPairForSliceUnion(const AKGMemRefAccess &srcAccess, Operation *i, Operation *j,
                                                unsigned loopDepth, unsigned numCommonLoops, bool isBackwardSlice,
                                                FlatAffineValueConstraints &sliceUnionCst,
                                                std::vector<std::pair<Operation *, Operation *>> &dependentOpPairs) {
  AKGMemRefAccess dstAccess(j);
  if (srcAccess.memref != dstAccess.memref) return success();

  // Upstream getComputationSliceState asserts loopDepth ≤ nesting depth of opsA in
  // forward mode and ≤ nesting depth of opsB in backward mode (the "depSource" side
  // it projects against). When the producer write lives shallower than the requested
  // fusion depth, clamp per pair rather than failing — the slice IVs end up at the
  // natural depth, and computeSliceUnionAKG lifts the rest via sibling matching
  // (slice-IV extension in backward, receiver extension in forward).
  unsigned naturalDepth = isBackwardSlice ? getNestingDepth(j) : getNestingDepth(i);
  unsigned effLoopDepth = std::min(loopDepth, naturalDepth);

  bool readReadAccesses = isa<AffineReadOpInterface>(srcAccess.opInst) && isa<AffineReadOpInterface>(dstAccess.opInst);
  FlatAffineValueConstraints dependenceConstraints;
  // Check dependence between 'srcAccess' and 'dstAccess'.
  DependenceResult result = checkMemrefAccessDependenceAKG(srcAccess, dstAccess, /*loopDepth=*/numCommonLoops + 1,
                                                           &dependenceConstraints, /*dependenceComponents=*/nullptr,
                                                           /*allowRAR=*/readReadAccesses, /*checkSrcBeforeDst=*/false);

  if (result.value == DependenceResult::Failure) {
    LLVM_DEBUG(llvm::dbgs() << "Dependence check failed\n");
    return failure();
  }
  if (result.value == DependenceResult::NoDependence) return success();

  dependentOpPairs.emplace_back(i, j);

  // Compute slice bounds for 'srcAccess' and 'dstAccess'.
  ComputationSliceState tmpSliceState;
  mlir::affine::getComputationSliceState(i, j, &dependenceConstraints, effLoopDepth, isBackwardSlice, &tmpSliceState);

  return mergeSliceStateIntoUnion(tmpSliceState, sliceUnionCst);
}

// Public per the declaration in akg/Dialect/Affine/Analysis/AffineAnalysis.h.
bool loopBoundsMatch(AffineForOp a, AffineForOp b) {
  if (a.getStep() != b.getStep()) return false;
  if (a.hasConstantLowerBound() && b.hasConstantLowerBound()) {
    if (a.getConstantLowerBound() != b.getConstantLowerBound()) return false;
  } else if (a.getLowerBoundMap() != b.getLowerBoundMap()) {
    return false;
  }
  if (a.hasConstantUpperBound() && b.hasConstantUpperBound()) {
    if (a.getConstantUpperBound() != b.getConstantUpperBound()) return false;
  } else if (a.getUpperBoundMap() != b.getUpperBoundMap()) {
    return false;
  }
  return true;
}

// True if the slice's lb/ub at level i are constant single-result maps that exactly cover the
// loop's natural constant iteration range — i.e. the level is unconstrained by data dependence
// and is eligible for structural single-point alignment.
static bool sliceLevelIsFullLoopRange(const ComputationSliceState &slice, unsigned i, AffineForOp loop) {
  AffineMap lbMap = slice.lbs[i];
  AffineMap ubMap = slice.ubs[i];
  if (!lbMap || !ubMap || lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1) return false;
  if (!loop.hasConstantLowerBound() || !loop.hasConstantUpperBound()) return false;
  auto lbConst = dyn_cast<AffineConstantExpr>(lbMap.getResult(0));
  auto ubConst = dyn_cast<AffineConstantExpr>(ubMap.getResult(0));
  if (!lbConst || !ubConst) return false;
  return lbConst.getValue() == loop.getConstantLowerBound() && ubConst.getValue() == loop.getConstantUpperBound();
}

// Locate v in the dim portion of ops; insert at the dim/symbol boundary if absent. Updates
// numDims when an insert occurs. Returns v's dim index.
static unsigned findOrInsertDimOperand(SmallVector<Value, 4> &ops, unsigned &numDims, Value v) {
  for (unsigned k = 0; k < numDims && k < ops.size(); ++k) {
    if (ops[k] == v) return k;
  }
  size_t insertPos = std::min<size_t>(numDims, ops.size());
  ops.insert(ops.begin() + insertPos, v);
  unsigned idx = numDims;
  numDims += 1;
  return idx;
}

// Undo dependence-driven slice tightening that comes from an inner affine.if gating
// the source op. getOpIndexSet folds every enclosing AffineIfOp's set into the source
// access domain, so when the producer store is wrapped in e.g. `affine.if (-d0 >= 0)(%arg8)`
// the slice IV for %arg8 gets pinned to a single point. That is correct for the specific
// dependence pair, but fuseLoops clones only the slice and erases the producer nest —
// so any sibling work in that loop body OUTSIDE the gating if (e.g. an unconditional load/
// store on a separate memref) is silently dropped at the non-pinned iterations.
//
// For each slice level whose IV is constrained by a gating AffineIfOp on src's parent
// chain, this checks the enclosing producer loop's body for ops that (a) are not on src's
// path and (b) are not nested under any of those gating ifs. If such "outside-the-gating"
// work exists, the slice level is reset to the loop's full constant domain so the cloned
// body preserves all original producer iterations.
static void collectGatingIfsAndIVs(Operation *src, SmallPtrSet<Operation *, 4> &gatingIfs, DenseSet<Value> &gatedIVs) {
  for (Operation *p = src->getParentOp(); p; p = p->getParentOp()) {
    auto ifOp = dyn_cast<AffineIfOp>(p);
    if (!ifOp) continue;
    gatingIfs.insert(p);
    for (Value operand : ifOp->getOperands()) {
      if (getForInductionVarOwner(operand)) gatedIVs.insert(operand);
    }
  }
}

static bool sliceLevelAlreadyAtFullRange(const ComputationSliceState &slice, unsigned k, int64_t loopLb,
                                         int64_t loopUb) {
  AffineMap lbMap = slice.lbs[k];
  AffineMap ubMap = slice.ubs[k];
  if (!lbMap || !ubMap || lbMap.getNumResults() != 1 || ubMap.getNumResults() != 1) return false;
  auto lbC = dyn_cast<AffineConstantExpr>(lbMap.getResult(0));
  auto ubC = dyn_cast<AffineConstantExpr>(ubMap.getResult(0));
  return lbC && ubC && lbC.getValue() == loopLb && ubC.getValue() == loopUb;
}

static bool buildPathToLoop(Operation *src, AffineForOp loop, SmallPtrSet<Operation *, 8> &onPath) {
  for (Operation *p = src; p; p = p->getParentOp()) {
    onPath.insert(p);
    if (p == loop.getOperation()) return true;
  }
  return false;
}

static bool loopBodyHasWorkOutsideGating(Operation *loopOp, const SmallPtrSet<Operation *, 8> &onPath,
                                         const SmallPtrSet<Operation *, 4> &gatingIfs) {
  bool hasOutsideWork = false;
  loopOp->walk([&](Operation *op) {
    if (op == loopOp) return WalkResult::advance();
    if (onPath.contains(op)) return WalkResult::advance();
    if (op->hasTrait<OpTrait::IsTerminator>()) return WalkResult::advance();
    for (Operation *gif : gatingIfs) {
      if (gif->isAncestor(op)) return WalkResult::advance();
    }
    hasOutsideWork = true;
    return WalkResult::interrupt();
  });
  return hasOutsideWork;
}

static void expandSliceBoundsForInnerIfGating(ComputationSliceState &slice, unsigned numSliceLoopIVs,
                                              ArrayRef<Operation *> opsA) {
  if (opsA.empty()) return;
  Operation *src = opsA.front();

  SmallPtrSet<Operation *, 4> gatingIfs;
  DenseSet<Value> gatedIVs;
  collectGatingIfsAndIVs(src, gatingIfs, gatedIVs);
  if (gatingIfs.empty() || gatedIVs.empty()) return;

  for (unsigned k = 0; k < numSliceLoopIVs; ++k) {
    if (!gatedIVs.contains(slice.ivs[k])) continue;

    AffineForOp loop = getForInductionVarOwner(slice.ivs[k]);
    if (!loop || !loop.hasConstantLowerBound() || !loop.hasConstantUpperBound()) continue;

    int64_t loopLb = loop.getConstantLowerBound();
    int64_t loopUb = loop.getConstantUpperBound();

    if (sliceLevelAlreadyAtFullRange(slice, k, loopLb, loopUb)) continue;

    SmallPtrSet<Operation *, 8> onPath;
    if (!buildPathToLoop(src, loop, onPath)) continue;

    if (!loopBodyHasWorkOutsideGating(loop.getOperation(), onPath, gatingIfs)) continue;

    MLIRContext *ctx = loop->getContext();
    slice.lbs[k] = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, getAffineConstantExpr(loopLb, ctx));
    slice.ubs[k] = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0, getAffineConstantExpr(loopUb, ctx));
    slice.lbOperands[k].clear();
    slice.ubOperands[k].clear();
  }
}

// Force outer slice levels into a single-point against the partner-side IV when both sides
// have matching structural bounds. Without this, dependence-based slicing leaves outer
// non-coupled levels at full range, causing fuseLoops to replicate the cloned body across
// that dim. Bound rewrites: lbs[i] -> d_i, ubs[i] -> d_i + step. surroundingLoops are the
// receiving-side nest; the ivs-side loop is reached via getForInductionVarOwner.
static void alignSliceWithPartnerLoops(ComputationSliceState &slice, unsigned loopDepth, unsigned numSliceLoopIVs,
                                       ArrayRef<AffineForOp> surroundingLoops) {
  for (unsigned i = 0; i < loopDepth && i < numSliceLoopIVs && i < surroundingLoops.size(); ++i) {
    AffineForOp recvLoop = surroundingLoops[i];
    AffineForOp ivsLoop = getForInductionVarOwner(slice.ivs[i]);
    if (!ivsLoop) continue;
    if (!loopBoundsMatch(ivsLoop, recvLoop)) continue;
    if (!sliceLevelIsFullLoopRange(slice, i, ivsLoop)) continue;

    AffineMap lbMap = slice.lbs[i];
    AffineMap ubMap = slice.ubs[i];
    Value partnerIV = recvLoop.getInductionVar();
    unsigned lbNumDims = lbMap.getNumDims();
    unsigned lbNumSyms = lbMap.getNumSymbols();
    unsigned ubNumDims = ubMap.getNumDims();
    unsigned ubNumSyms = ubMap.getNumSymbols();
    unsigned lbDimIdx = findOrInsertDimOperand(slice.lbOperands[i], lbNumDims, partnerIV);
    unsigned ubDimIdx = findOrInsertDimOperand(slice.ubOperands[i], ubNumDims, partnerIV);
    int64_t step = ivsLoop.getStepAsInt();
    MLIRContext *ctx = partnerIV.getContext();
    slice.lbs[i] = AffineMap::get(lbNumDims, lbNumSyms, getAffineDimExpr(lbDimIdx, ctx));
    slice.ubs[i] = AffineMap::get(ubNumDims, ubNumSyms, getAffineDimExpr(ubDimIdx, ctx) + step);
  }
}

static void collectForbiddenWriteSinks(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                                       DenseSet<Value> &forbiddenWriteSinks) {
  for (Operation *op : opsA) {
    if (auto w = dyn_cast<AffineWriteOpInterface>(op)) forbiddenWriteSinks.insert(getSourceMemRef(w.getMemRef()));
  }
  for (Operation *op : opsB) {
    if (auto r = dyn_cast<AffineReadOpInterface>(op)) forbiddenWriteSinks.insert(getSourceMemRef(r.getMemRef()));
  }
}

static bool candidateWritesForbidden(AffineForOp forOp, const DenseSet<Value> &forbiddenWriteSinks) {
  bool hit = false;
  forOp.getOperation()->walk([&](Operation *nested) {
    auto w = dyn_cast<AffineWriteOpInterface>(nested);
    if (!w) return WalkResult::advance();
    if (forbiddenWriteSinks.count(getSourceMemRef(w.getMemRef()))) {
      hit = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return hit;
}

static AffineForOp findMatchingSiblingLoop(Block::iterator searchStart, Block *searchBlock,
                                           ArrayRef<AffineForOp> partnerLoops, unsigned d,
                                           const DenseSet<Value> &forbiddenWriteSinks) {
  for (auto it = searchStart; it != searchBlock->end(); ++it) {
    if (it->hasTrait<OpTrait::IsTerminator>()) break;
    auto forOp = dyn_cast<AffineForOp>(*it);
    if (!forOp) continue;
    if (!loopBoundsMatch(forOp, partnerLoops[d])) continue;
    if (candidateWritesForbidden(forOp, forbiddenWriteSinks)) continue;
    return forOp;
  }
  return AffineForOp();
}

// Find a chain of opsA-side sibling for-loops whose bounds successively match
// partnerLoops[startDepth..loopDepth-1]. Walks the body of opsA's common parent
// after the last opsA op, descending into each matched sibling's body to look for
// the next level. Skips candidates whose body writes into a memref that opsA
// writes (mediator state) or opsB reads (consumer-visible state) — those would
// corrupt fused semantics.
//
// Returns the chain on success, empty vector on failure. Shared by both
// slice-IV extension (backward) and receiver extension (forward).
static SmallVector<AffineForOp, 4> findOpsASiblingChain(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                                                        ArrayRef<AffineForOp> partnerLoops, unsigned startDepth,
                                                        unsigned loopDepth) {
  SmallVector<AffineForOp, 4> extLoops;
  if (startDepth >= loopDepth) return extLoops;
  if (loopDepth > partnerLoops.size()) return extLoops;
  if (opsA.empty()) return extLoops;

  Operation *commonParent = opsA.front()->getParentOp();
  for (Operation *op : opsA.drop_front()) {
    if (op->getParentOp() != commonParent) return extLoops;
  }
  auto parentLoop = dyn_cast_or_null<AffineForOp>(commonParent);
  if (!parentLoop) return extLoops;

  Operation *afterOp = opsA.front();
  for (Operation *op : opsA.drop_front()) {
    if (afterOp->isBeforeInBlock(op)) afterOp = op;
  }

  DenseSet<Value> forbiddenWriteSinks;
  collectForbiddenWriteSinks(opsA, opsB, forbiddenWriteSinks);

  Block *searchBlock = parentLoop.getBody();
  Block::iterator searchStart = std::next(Block::iterator(afterOp));
  for (unsigned d = startDepth; d < loopDepth; ++d) {
    AffineForOp matched = findMatchingSiblingLoop(searchStart, searchBlock, partnerLoops, d, forbiddenWriteSinks);
    if (!matched) return extLoops;
    extLoops.push_back(matched);
    searchBlock = matched.getBody();
    searchStart = searchBlock->begin();
  }
  return extLoops;
}

// Backward-mode lift: when slice IVs (from opsA side) are shallower than the
// requested fusion depth, append IVs from matched opsA siblings with single-point
// bounds coupled to partnerLoops[d] (the receiver side, deep). Pre-spine ops
// replicate at the outer level — same execution count they had in src — instead
// of being smeared across the receiver's inner range.
static bool extendSliceWithSrcSiblingLoops(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                                           ComputationSliceState &slice, unsigned currentNumIVs, unsigned loopDepth,
                                           ArrayRef<AffineForOp> partnerLoops) {
  if (currentNumIVs >= loopDepth) return false;
  if (opsA.empty()) return false;

  // Continuity: existing slice IVs must end at opsA's common parent so the new
  // sibling IVs splice on without a gap. This also filters out the forward-slice
  // case where slice.ivs are opsB-side IVs (covered by the receiver-extension path).
  Operation *commonParent = opsA.front()->getParentOp();
  auto parentLoop = dyn_cast_or_null<AffineForOp>(commonParent);
  if (!parentLoop) return false;
  if (currentNumIVs > 0) {
    AffineForOp lastIVLoop = getForInductionVarOwner(slice.ivs[currentNumIVs - 1]);
    if (lastIVLoop != parentLoop) return false;
  }

  auto extLoops = findOpsASiblingChain(opsA, opsB, partnerLoops, currentNumIVs, loopDepth);
  if (extLoops.empty()) return false;

  // opsA is non-const Operation*, and Operation::getContext() is const-callable, so this
  // avoids the const-cast trap of going through ArrayRef<AffineForOp>::operator[] (which
  // returns const T& and would discard qualifiers on OpState::getContext()).
  MLIRContext *ctx = opsA.front()->getContext();
  AffineExpr d0 = getAffineDimExpr(0, ctx);
  for (unsigned d = currentNumIVs; d < loopDepth; ++d) {
    AffineForOp srcLoop = extLoops[d - currentNumIVs];
    AffineForOp partnerLoop = partnerLoops[d];
    int64_t step = srcLoop.getStepAsInt();

    slice.ivs.push_back(srcLoop.getInductionVar());
    slice.lbs.push_back(AffineMap::get(/*dimCount=*/1, /*symCount=*/0, d0));
    slice.ubs.push_back(AffineMap::get(/*dimCount=*/1, /*symCount=*/0, d0 + step));
    SmallVector<Value, 4> operands{partnerLoop.getInductionVar()};
    slice.lbOperands.push_back(operands);
    slice.ubOperands.push_back(operands);
  }
  return true;
}

// Forward-mode lift: when the receiver (loops surrounding opsA) is shallower than
// the requested fusion depth, extend it with matched opsA-side siblings. In forward
// mode the slice IVs already come from opsB at full depth, so what's missing is
// the receiver chain — without these matched siblings, target-block selection
// would index past the end of the receiver nest. partnerLoops are the deep opsB
// loops the siblings match against.
//
// Subsequent alignSliceWithPartnerLoops couples slice.ivs[d] (opsB IV) with the
// appended sibling's IV at single-point bounds — yielding the same merge-point
// shape as the backward-mode extension produces, just from the other direction.
static bool extendReceiverWithSiblingForLoops(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                                              SmallVectorImpl<AffineForOp> &receiverLoops,
                                              ArrayRef<AffineForOp> partnerLoops, unsigned currentDepth,
                                              unsigned loopDepth) {
  if (currentDepth >= loopDepth) return false;
  if (opsA.empty()) return false;

  // The receiver chain must end at opsA's common parent so the matched siblings
  // attach right where opsA's nest ends. Without this anchor we'd be guessing at
  // which subtree the appended for-loops belong to.
  Operation *commonParent = opsA.front()->getParentOp();
  auto parentLoop = dyn_cast_or_null<AffineForOp>(commonParent);
  if (!parentLoop) return false;
  if (currentDepth > 0 && receiverLoops[currentDepth - 1] != parentLoop) return false;

  auto extLoops = findOpsASiblingChain(opsA, opsB, partnerLoops, currentDepth, loopDepth);
  if (extLoops.empty()) return false;

  std::copy(extLoops.begin(), extLoops.end(), std::back_inserter(receiverLoops));
  return true;
}

// True if any slice level has more than one iteration (full-range or non-1 trip count).
// Used to decide whether the cloned body is purely sequential (anchor-based insertPoint)
// or carries leftover inner fors (fall back to block begin / end-1).
static bool sliceHasInnerFor(const ComputationSliceState &slice, unsigned numSliceLoopIVs) {
  for (unsigned i = 0; i < numSliceLoopIVs; ++i) {
    AffineMap lb = slice.lbs[i];
    AffineMap ub = slice.ubs[i];
    if (!lb || !ub || lb.getNumResults() != 1 || ub.getNumResults() != 1) return true;
    AffineExpr diff = ub.getResult(0) - lb.getResult(0);
    auto cst = dyn_cast<AffineConstantExpr>(diff);
    if (!cst || cst.getValue() != 1) return true;
  }
  return false;
}

// Add to candidates any op in targetBlock whose memref accesses conflict with the cloned
// for-op's reads/writes (RAW where dst reads cloned-writes; WAR/WAW where dst writes a
// memref cloned reads or writes). dependentOpPairs only carries the primary fusion mediator,
// so non-mediator memref conflicts also constrain the safe insertion zone.
static void addMemrefConflictCandidates(ArrayRef<Operation *> clonedOps, Block *targetBlock,
                                        ArrayRef<AffineForOp> surroundingLoops, DenseSet<Operation *> &candidates) {
  if (clonedOps.empty()) return;
  Block *parentBlock = surroundingLoops[0]->getBlock();
  Operation *clonedRoot = parentBlock->findAncestorOpInBlock(*clonedOps[0]);
  if (!clonedRoot) return;

  DenseSet<Value> clonedWrites, clonedReads;
  clonedRoot->walk([&](Operation *nested) {
    if (auto w = dyn_cast<AffineWriteOpInterface>(nested)) clonedWrites.insert(getSourceMemRef(w.getMemRef()));
    if (auto r = dyn_cast<AffineReadOpInterface>(nested)) clonedReads.insert(getSourceMemRef(r.getMemRef()));
  });
  auto opConflicts = [&](Operation &op) {
    bool hit = false;
    op.walk([&](Operation *nested) {
      if (auto r = dyn_cast<AffineReadOpInterface>(nested)) {
        if (clonedWrites.count(getSourceMemRef(r.getMemRef()))) {
          hit = true;
          return WalkResult::interrupt();
        }
      } else if (auto w = dyn_cast<AffineWriteOpInterface>(nested)) {
        Value mem = getSourceMemRef(w.getMemRef());
        if (clonedWrites.count(mem) || clonedReads.count(mem)) {
          hit = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return hit;
  };
  for (Operation &op : *targetBlock) {
    if (opConflicts(op)) candidates.insert(&op);
  }
}

// Expand candidates along SSA forward (uses) and backward (defs) edges, mapping users/defs
// back to their top-level ancestor in targetBlock. Keeps tightly coupled computations
// (e.g. co-used loads feeding a single arith op) contiguous around the insertion point.
static void expandCandidatesViaSSA(Block *targetBlock, DenseSet<Operation *> &candidates) {
  SmallVector<Operation *, 8> worklist(candidates.begin(), candidates.end());
  auto visit = [&](Operation *neighbor) {
    if (!neighbor) return;
    Operation *top = targetBlock->findAncestorOpInBlock(*neighbor);
    if (top && candidates.insert(top).second) worklist.push_back(top);
  };
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    op->walk([&](Operation *nested) {
      for (Value res : nested->getResults()) {
        for (Operation *user : res.getUsers()) visit(user);
      }
      for (Value operand : nested->getOperands()) visit(operand.getDefiningOp());
    });
  }
}

// Pick the anchor iterator inside targetBlock for inserting the cloned slice when the cloned
// body is purely sequential. Combines: dependentOpPairs (primary fusion mediator), memref-
// conflict candidates from the cloned for-op's full body, and SSA-edge cluster expansion.
// Returns fallbackPoint when no candidate is found.
static Block::iterator chooseInsertPoint(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB,
                                         const std::vector<std::pair<Operation *, Operation *>> &dependentOpPairs,
                                         Block *targetBlock, ArrayRef<AffineForOp> surroundingLoops,
                                         bool isBackwardSlice, Block::iterator fallbackPoint) {
  DenseSet<Operation *> candidates;
  auto addCandidate = [&](Operation *op) {
    if (!op) return;
    if (Operation *top = targetBlock->findAncestorOpInBlock(*op)) candidates.insert(top);
  };
  for (const auto &dep : dependentOpPairs) addCandidate(isBackwardSlice ? dep.second : dep.first);
  addMemrefConflictCandidates(isBackwardSlice ? opsA : opsB, targetBlock, surroundingLoops, candidates);
  expandCandidatesViaSSA(targetBlock, candidates);

  Operation *anchor = nullptr;
  for (Operation *cand : candidates) {
    if (!anchor) {
      anchor = cand;
      continue;
    }
    bool replace = isBackwardSlice ? cand->isBeforeInBlock(anchor) : anchor->isBeforeInBlock(cand);
    if (replace) anchor = cand;
  }
  return anchor ? (isBackwardSlice ? anchor->getIterator() : std::next(anchor->getIterator())) : fallbackPoint;
}

SliceComputationResult computeSliceUnionAKG(ArrayRef<Operation *> opsA, ArrayRef<Operation *> opsB, unsigned loopDepth,
                                            unsigned numCommonLoops, bool isBackwardSlice,
                                            ComputationSliceState *sliceUnion) {
  // Standard dependence-based approach.
  FlatAffineValueConstraints sliceUnionCst;
  assert(sliceUnionCst.getNumDimAndSymbolVars() == 0);
  std::vector<std::pair<Operation *, Operation *>> dependentOpPairs;
  for (auto *i : opsA) {
    AKGMemRefAccess srcAccess(i);
    const bool hasFailure = std::any_of(opsB.begin(), opsB.end(), [&](Operation *j) {
      return failed(processOpPairForSliceUnion(srcAccess, i, j, loopDepth, numCommonLoops, isBackwardSlice,
                                               sliceUnionCst, dependentOpPairs));
    });

    if (hasFailure) return SliceComputationResult::GenericFailure;
  }

  // Empty union.
  if (sliceUnionCst.getNumDimAndSymbolVars() == 0) return SliceComputationResult::GenericFailure;

  // surroundingLoops = receiver nest where the cloned slice will be inserted.
  //   backward: dep.second (opsB side); forward: dep.first (opsA side).
  // partnerLoops = the opposite (always opsB, i.e. consumer-side) nest used as the
  //   deep reference for sibling-based depth lifts. In backward this matches the
  //   receiver; in forward (shallow opsA) it provides the deep loops the receiver
  //   needs to grow toward.
  SmallVector<Operation *, 4> ops;
  ops.reserve(dependentOpPairs.size());
  std::transform(dependentOpPairs.begin(), dependentOpPairs.end(), std::back_inserter(ops),
                 [isBackwardSlice](const auto &dep) { return isBackwardSlice ? dep.second : dep.first; });

  SmallVector<Operation *, 4> partnerOps;
  partnerOps.reserve(dependentOpPairs.size());
  std::transform(dependentOpPairs.begin(), dependentOpPairs.end(), std::back_inserter(partnerOps),
                 [](const auto &dep) { return dep.second; });

  SmallVector<AffineForOp, 4> surroundingLoops;
  unsigned innermostCommonLoopDepth = getInnermostCommonLoopDepth(ops, &surroundingLoops);
  SmallVector<AffineForOp, 4> partnerLoops;
  unsigned partnerCommonLoopDepth = getInnermostCommonLoopDepth(partnerOps, &partnerLoops);
  if (loopDepth > innermostCommonLoopDepth) {
    // Receiver is shallower than the requested fusion depth. Backward mode handles
    // this via slice-IV extension further down (slice IVs grow from opsA's parent
    // body; the receiver was already deep enough by virtue of being opsB). Forward
    // mode is the inverse: slice IVs are already deep (opsB-sized), but the
    // receiver needs matched opsA-side siblings to host the deeper levels.
    bool extended = !isBackwardSlice && loopDepth <= partnerCommonLoopDepth &&
                    extendReceiverWithSiblingForLoops(opsA, opsB, surroundingLoops, partnerLoops,
                                                      innermostCommonLoopDepth, loopDepth);
    if (!extended) {
      LLVM_DEBUG(llvm::dbgs() << "Exceeds max loop depth\n");
      return SliceComputationResult::GenericFailure;
    }
  }

  // Store 'numSliceLoopIVs' before converting dst loop IVs to dims.
  unsigned numSliceLoopIVs = sliceUnionCst.getNumDimVars();

  // Convert any dst loop IVs which are symbol variables to dim variables.
  sliceUnionCst.convertLoopIVSymbolsToDims();
  sliceUnion->clearBounds();
  sliceUnion->lbs.resize(numSliceLoopIVs, AffineMap());
  sliceUnion->ubs.resize(numSliceLoopIVs, AffineMap());

  // Get slice bounds from slice union constraints 'sliceUnionCst'.
  sliceUnionCst.getSliceBounds(/*offset=*/0, numSliceLoopIVs, opsA[0]->getContext(), &sliceUnion->lbs,
                               &sliceUnion->ubs);

  // Add slice bound operands of union.
  SmallVector<Value, 4> sliceBoundOperands;
  sliceUnionCst.getValues(numSliceLoopIVs, sliceUnionCst.getNumDimAndSymbolVars(), &sliceBoundOperands);

  // Copy src loop IVs from 'sliceUnionCst' to 'sliceUnion'.
  sliceUnion->ivs.clear();
  sliceUnionCst.getValues(0, numSliceLoopIVs, &sliceUnion->ivs);

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceUnion->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceUnion->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

  // Producer levels gated by an inner affine.if get pinned to a single iteration in
  // the dependence-driven slice. If the gated loop has work outside that if (siblings of
  // the gating chain), preserve those iterations by expanding back to the loop's full
  // constant domain — fuseLoops would otherwise drop them when erasing the producer nest.
  expandSliceBoundsForInnerIfGating(*sliceUnion, numSliceLoopIVs, opsA);

  // Outer-level structural alignment runs before insertPoint selection so
  // sliceHasInnerFor sees the post-alignment lbs/ubs.
  alignSliceWithPartnerLoops(*sliceUnion, loopDepth, numSliceLoopIVs, surroundingLoops);

  // When the src access lives shallower than the requested fusion depth, try to extend
  // the slice with src sibling for-loops that align with dst's deeper levels. This is
  // what unblocks ProducerConsumer fusion where the producer write is at depth K but the
  // consumer body sits at depth K+N: the matched src sibling becomes the merge point and
  // pre-spine ops (the write and its siblings) replicate at the outer dst level — same
  // execution count they had in src — instead of being smeared across dst's inner range.
  //
  // Match against partnerLoops (always opsB-side, deep). In backward partnerLoops ==
  // surroundingLoops; in forward (shallow opsA) the function exits via the IV-continuity
  // check anyway — the depth lift in that direction is handled above by
  // extendReceiverWithSiblingForLoops.
  if (numSliceLoopIVs < loopDepth &&
      extendSliceWithSrcSiblingLoops(opsA, opsB, *sliceUnion, numSliceLoopIVs, loopDepth, partnerLoops)) {
    numSliceLoopIVs = loopDepth;
  }

  // Choose insertPoint based on whether the cloned dst body keeps inner fors.
  // - All slice levels are 1-iter: cloned body is sequential ops only, anchor
  //   the insert point on the dependent op so the new consumer/producer stays
  //   close to the existing one (helps memref scalar promotion).
  // - Otherwise the cloned body carries leftover inner fors; fall back to
  //   start/end of the surrounding body so the new fors don't get stranded
  //   parallel to siblings in the middle of flat producer/consumer chains.
  Block *targetBlock = surroundingLoops[loopDepth - 1].getBody();
  auto fallbackPoint = isBackwardSlice ? targetBlock->begin() : std::prev(targetBlock->end());
  if (sliceHasInnerFor(*sliceUnion, numSliceLoopIVs)) {
    sliceUnion->insertPoint = fallbackPoint;
  } else {
    sliceUnion->insertPoint =
      chooseInsertPoint(opsA, opsB, dependentOpPairs, targetBlock, surroundingLoops, isBackwardSlice, fallbackPoint);
  }

  // Check if the slice computed is valid. Return success only if it is verified
  // that the slice is valid, otherwise return appropriate failure status.
  std::optional<bool> isSliceValid = sliceUnion->isSliceValid();
  if (!isSliceValid) {
    LLVM_DEBUG(llvm::dbgs() << "Cannot determine if the slice is valid\n");
    return SliceComputationResult::GenericFailure;
  }
  if (!*isSliceValid) return SliceComputationResult::IncorrectSliceFailure;

  return SliceComputationResult::Success;
}

unsigned AKGMemRefAccess::getRank() const { return cast<MemRefType>(memref.getType()).getRank(); }

bool AKGMemRefAccess::isStore() const { return isa<AffineWriteOpInterface>(opInst); }

/// Equal if both affine accesses are provably equivalent (at compile
/// time) when considering the memref, the affine maps and their respective
/// operands. The equality of access functions + operands is checked by
/// subtracting fully composed value maps, and then simplifying the difference
/// using the expression flattener.
/// TODO(scheduler): this does not account for aliasing of memrefs.
bool AKGMemRefAccess::operator==(const AKGMemRefAccess &rhs) const {
  if (memref != rhs.memref) return false;

  AffineValueMap diff, thisMap, rhsMap;
  getAccessMap(&thisMap);
  rhs.getAccessMap(&rhsMap);
  AffineValueMap::difference(thisMap, rhsMap, &diff);
  return llvm::all_of(diff.getAffineMap().getResults(), [](AffineExpr e) { return e == 0; });
}

}  // namespace affine
}  // namespace mlir
