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
  // Check if 'loopDepth' exceeds nesting depth of src/dst ops.
  if ((!isBackwardSlice && loopDepth > getNestingDepth(i)) || (isBackwardSlice && loopDepth > getNestingDepth(j))) {
    LLVM_DEBUG(llvm::dbgs() << "Invalid loop depth\n");
    return failure();
  }

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
  mlir::affine::getComputationSliceState(i, j, &dependenceConstraints, loopDepth, isBackwardSlice, &tmpSliceState);

  return mergeSliceStateIntoUnion(tmpSliceState, sliceUnionCst);
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

  // Gather loops surrounding ops from loop nest where slice will be inserted.
  SmallVector<Operation *, 4> ops;
  ops.reserve(dependentOpPairs.size());

  std::transform(dependentOpPairs.begin(), dependentOpPairs.end(), std::back_inserter(ops),
                 [isBackwardSlice](const auto &dep) { return isBackwardSlice ? dep.second : dep.first; });

  SmallVector<AffineForOp, 4> surroundingLoops;
  unsigned innermostCommonLoopDepth = getInnermostCommonLoopDepth(ops, &surroundingLoops);
  if (loopDepth > innermostCommonLoopDepth) {
    LLVM_DEBUG(llvm::dbgs() << "Exceeds max loop depth\n");
    return SliceComputationResult::GenericFailure;
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

  // Set loop nest insertion point to block start at 'loopDepth'.
  sliceUnion->insertPoint = isBackwardSlice ? surroundingLoops[loopDepth - 1].getBody()->begin()
                                            : std::prev(surroundingLoops[loopDepth - 1].getBody()->end());

  // Give each bound its own copy of 'sliceBoundOperands' for subsequent
  // canonicalization.
  sliceUnion->lbOperands.resize(numSliceLoopIVs, sliceBoundOperands);
  sliceUnion->ubOperands.resize(numSliceLoopIVs, sliceBoundOperands);

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
/// TODO: this does not account for aliasing of memrefs.
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
