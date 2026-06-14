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
//===- ArithToHIVM.cpp - conversion from Arith to HIVM dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "akg/Conversion/ArithToHIVM/ArithToHIVM.h"

#include <algorithm>
#include <climits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <utility>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Utils/AnalysisForNpu.hpp"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOHIVM
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {

using akg::alignUpInt64;
using akg::ceilDivInt64;
using akg::computeBishengInlineBroadcastSourceStorageBytes;
using akg::computeBishengLastDimTransposeBufferBytes;
using akg::computeBishengNpuVectorStorageBytes;
using akg::computeBishengStrideAlignedStorageBytes;
using akg::computeBishengStrideAlignedStorageBytesWithTrailingUnit;
using akg::computeBishengStructuredNpuVectorStorageBytes;
using akg::getElementBitWidth;
using akg::kNpuUbAlignBytes;
using akg::multiplyAndCap;

static bool setBufferSizeMark(PatternRewriter &rewriter, Location loc, Value buffer, int64_t bytes) {
  if (bytes <= 0 || bytes == LLONG_MAX || !isa<MemRefType>(buffer.getType())) {
    return false;
  }
  auto markOp = rewriter.create<annotation::MarkOp>(loc, buffer);
  markOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(bytes));
  return true;
}

static bool propagateBufferSizeMark(ConversionPatternRewriter &rewriter, Location loc, Value src, Value dest) {
  for (Operation *user : src.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (auto attr = markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr)) {
        auto srcShapedType = dyn_cast<ShapedType>(src.getType());
        auto destShapedType = dyn_cast<ShapedType>(dest.getType());
        if (!srcShapedType || !destShapedType) {
          return false;
        }
        int64_t srcWidth = getElementBitWidth(srcShapedType.getElementType());
        int64_t destWidth = getElementBitWidth(destShapedType.getElementType());
        if (srcWidth <= 0 || destWidth <= 0) {
          return false;
        }
        int64_t oldSize = attr.getInt();
        int64_t newSize = oldSize;
        if (!srcShapedType.hasRank() || !destShapedType.hasRank() ||
            !llvm::equal(srcShapedType.getShape(), destShapedType.getShape()) || srcShapedType.getRank() <= 1 ||
            destWidth > srcWidth) {
          newSize = alignUpInt64(ceilDivInt64(multiplyAndCap(oldSize, destWidth), srcWidth), kNpuUbAlignBytes);
        }

        return setBufferSizeMark(rewriter, loc, dest, newSize);
      }
    }
  }
  return false;
}

/// Returns one folded upper bound per result rank. Rank-wide maxSizes are kept
/// as-is so static result dims can still use a larger allocated upper bound.
static FailureOr<SmallVector<int64_t>> foldMaxValsForNpuMark(npuvector::NPUVectorType npuTy, ValueRange maxSizes) {
  if (maxSizes.size() != static_cast<size_t>(npuTy.getRank())) {
    return failure();
  }

  SmallVector<int64_t> raw;
  for (Value v : maxSizes) {
    auto cop = v.getDefiningOp<arith::ConstantOp>();
    if (!cop) {
      return failure();
    }
    auto ia = dyn_cast<IntegerAttr>(cop.getValue());
    if (!ia) {
      return failure();
    }
    raw.push_back(ia.getInt());
  }
  return raw;
}

static FailureOr<SmallVector<int64_t>> inferNPUVectorMaxShape(Value value, int depth = 8) {
  auto npuTy = dyn_cast<npuvector::NPUVectorType>(value.getType());
  if (!npuTy) {
    return failure();
  }
  if (!npuTy.hasDynamicShape()) {
    return SmallVector<int64_t>(npuTy.getShape());
  }

  Operation *defOp = value.getDefiningOp();
  if ((defOp == nullptr) || depth <= 0) {
    return failure();
  }
  if (auto readOp = dyn_cast<npuvector::TransferReadOp>(defOp)) {
    return foldMaxValsForNpuMark(npuTy, readOp.getMaxSizes());
  }
  if (auto brcOp = dyn_cast<npuvector::BroadcastOp>(defOp)) {
    return foldMaxValsForNpuMark(npuTy, brcOp.getMaxSizes());
  }
  if (auto transposeOp = dyn_cast<npuvector::TransposeOp>(defOp)) {
    auto sourceMaxShape = inferNPUVectorMaxShape(transposeOp.getVector(), depth - 1);
    if (failed(sourceMaxShape)) {
      return failure();
    }
    ArrayRef<int64_t> perm = transposeOp.getPermutation();
    if (perm.size() != sourceMaxShape->size()) {
      return failure();
    }
    SmallVector<int64_t> transposed;
    transposed.reserve(perm.size());
    for (int64_t dim : perm) {
      if (dim < 0 || static_cast<size_t>(dim) >= sourceMaxShape->size()) {
        return failure();
      }
      transposed.push_back((*sourceMaxShape)[static_cast<size_t>(dim)]);
    }
    return transposed;
  }

  for (Value operand : defOp->getOperands()) {
    auto operandTy = dyn_cast<npuvector::NPUVectorType>(operand.getType());
    if (!operandTy || operandTy.getRank() != npuTy.getRank()) {
      continue;
    }
    auto inferred = inferNPUVectorMaxShape(operand, depth - 1);
    if (succeeded(inferred)) {
      return inferred;
    }
  }
  return failure();
}

static bool setTransposeResultBufferSizeMark(PatternRewriter &rewriter, Location loc, MemRefType resultType,
                                             Type elemType, ArrayRef<int64_t> resultMaxShape,
                                             ArrayRef<int64_t> sourceMaxShape, ArrayRef<int64_t> sourceTypeShape,
                                             ArrayRef<int64_t> perm, Value buffer) {
  if (resultMaxShape.empty()) {
    return false;
  }
  int64_t bytes = computeBishengStructuredNpuVectorStorageBytes(resultMaxShape, resultType.getShape(), elemType);
  int64_t transposeBytes =
    computeBishengLastDimTransposeBufferBytes(sourceMaxShape, sourceTypeShape, perm, elemType, true);
  return setBufferSizeMark(rewriter, loc, buffer, std::max(bytes, transposeBytes));
}

static bool setNPUVectorBufferSizeMark(PatternRewriter &rewriter, Location loc, npuvector::NPUVectorType npuTy,
                                       Type elemType, ArrayRef<int64_t> maxShape, Value buffer) {
  return setBufferSizeMark(rewriter, loc, buffer,
                           computeBishengNpuVectorStorageBytes(maxShape, npuTy.getShape(), elemType));
}

static FailureOr<SmallVector<int64_t>> inferNPUVectorMaxShapeFromOperands(Operation *op,
                                                                          npuvector::NPUVectorType npuTy) {
  if (!npuTy.hasDynamicShape()) {
    return SmallVector<int64_t>(npuTy.getShape());
  }

  SmallVector<int64_t> merged;
  for (Value operand : op->getOperands()) {
    auto operandTy = dyn_cast<npuvector::NPUVectorType>(operand.getType());
    if (!operandTy || operandTy.getRank() != npuTy.getRank()) {
      continue;
    }
    auto inferred = inferNPUVectorMaxShape(operand);
    if (failed(inferred)) {
      continue;
    }
    if (merged.empty()) {
      merged = *inferred;
      continue;
    }
    if (merged.size() != inferred->size()) {
      return failure();
    }
    for (size_t i = 0; i < merged.size(); ++i) {
      merged[i] = std::max(merged[i], (*inferred)[i]);
    }
  }
  if (merged.empty()) {
    return failure();
  }
  return merged;
}

static bool setNPUVectorResultBufferSizeMark(PatternRewriter &rewriter, Location loc, Operation *op, Value buffer,
                                             Type elemTypeOverride = Type()) {
  if ((op == nullptr) || op->getNumResults() != 1) {
    return false;
  }
  auto npuTy = dyn_cast<npuvector::NPUVectorType>(op->getResult(0).getType());
  if (!npuTy || !npuTy.hasDynamicShape()) {
    return false;
  }
  auto maxShape = inferNPUVectorMaxShapeFromOperands(op, npuTy);
  return succeeded(maxShape) &&
         setNPUVectorBufferSizeMark(rewriter, loc, npuTy, elemTypeOverride ? elemTypeOverride : npuTy.getElementType(),
                                    *maxShape, buffer);
}

static bool setNPUVectorValueBufferSizeMark(PatternRewriter &rewriter, Location loc, Value value, Value buffer,
                                            Type elemTypeOverride = Type()) {
  auto npuTy = dyn_cast<npuvector::NPUVectorType>(value.getType());
  if (!npuTy || !npuTy.hasDynamicShape()) {
    return false;
  }
  auto maxShape = inferNPUVectorMaxShape(value);
  return succeeded(maxShape) &&
         setNPUVectorBufferSizeMark(rewriter, loc, npuTy, elemTypeOverride ? elemTypeOverride : npuTy.getElementType(),
                                    *maxShape, buffer);
}

static bool setOrPropagateBufferSizeMark(ConversionPatternRewriter &rewriter, Location loc, Operation *op,
                                         Value fallbackSrc, Value buffer, Type elemTypeOverride = Type()) {
  if (setNPUVectorResultBufferSizeMark(rewriter, loc, op, buffer, elemTypeOverride)) {
    return true;
  }
  return propagateBufferSizeMark(rewriter, loc, fallbackSrc, buffer);
}

static void propagateSelectBufferSizeMark(ConversionPatternRewriter &rewriter, Location loc, Value trueVal,
                                          Value falseVal, Value dest) {
  if (isa<ShapedType>(trueVal.getType())) {
    if (propagateBufferSizeMark(rewriter, loc, trueVal, dest)) {
      return;
    }
  }
  if (isa<ShapedType>(falseVal.getType())) {
    propagateBufferSizeMark(rewriter, loc, falseVal, dest);
  }
}

static bool isScalarType(Type type) { return isa<IntegerType, FloatType, IndexType>(type); }

static bool requiresVectorRhsForHIVMLowering(Operation *user) {
  return isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MulSIExtendedOp, arith::MulUIExtendedOp>(user);
}

static bool isVectorOrNPUVectorLike(Type type) { return isa<VectorType, npuvector::NPUVectorType>(type); }

static bool hasVectorOrNPUVectorLikeResult(Operation *op) {
  return llvm::any_of(op->getResultTypes(), isVectorOrNPUVectorLike);
}

static bool isI1VectorOrNPUVectorLike(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getElementType().isInteger(1);
  }
  if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(type)) {
    return npuVectorType.getElementType().isInteger(1);
  }
  return false;
}

static SmallVector<unsigned> getOperandIndices(Operation *user, Value operand) {
  SmallVector<unsigned> indices;
  for (unsigned i = 0, e = user->getNumOperands(); i < e; ++i) {
    if (user->getOperand(i) == operand) {
      indices.push_back(i);
    }
  }
  return indices;
}

static bool isScalarBroadcast(Value value) {
  auto broadcast = value.getDefiningOp<npuvector::BroadcastOp>();
  return broadcast && isScalarType(broadcast.getSource().getType());
}

static bool isSupportedBinaryComputeScalarFoldUser(Operation *user, ArrayRef<unsigned> operandIndices) {
  if (user->getNumOperands() != 2 || !hasVectorOrNPUVectorLikeResult(user)) {
    return false;
  }

  if (requiresVectorRhsForHIVMLowering(user)) {
    return false;
  }

  bool foldsLhs = llvm::is_contained(operandIndices, 0);
  bool foldsRhs = llvm::is_contained(operandIndices, 1);
  if (!foldsLhs && !foldsRhs) {
    return false;
  }

  // Folding both operands would leave no vector input to drive the HIVM op.
  if (foldsLhs && foldsRhs) {
    return false;
  }

  if (foldsLhs && !user->hasTrait<OpTrait::IsCommutative>()) {
    return false;
  }

  Value otherOperand = user->getOperand(foldsLhs ? 1 : 0);
  return !isScalarType(otherOperand.getType()) && !isScalarBroadcast(otherOperand);
}

static bool isSupportedSelectLikeScalarFoldUser(Operation *user, ArrayRef<unsigned> operandIndices) {
  if (user->getNumOperands() != 3 || !hasVectorOrNPUVectorLikeResult(user)) {
    return false;
  }

  if (!isI1VectorOrNPUVectorLike(user->getOperand(0).getType())) {
    return false;
  }

  // Keep the condition vector-shaped. HIVM select value operands may be scalar.
  return llvm::all_of(operandIndices, [](unsigned idx) { return idx == 1 || idx == 2; });
}

static bool isSupportedBroadcastScalarFoldUser(Operation *user, Value broadcastResult) {
  SmallVector<unsigned> operandIndices = getOperandIndices(user, broadcastResult);
  if (operandIndices.empty()) {
    return false;
  }

  return isSupportedBinaryComputeScalarFoldUser(user, operandIndices) ||
         isSupportedSelectLikeScalarFoldUser(user, operandIndices);
}

static std::optional<std::pair<ArrayRef<int64_t>, Type>> getShapeAndElemType(Type type) {
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return std::make_pair(vectorType.getShape(), vectorType.getElementType());
  }
  if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(type)) {
    return std::make_pair(npuVectorType.getShape(), npuVectorType.getElementType());
  }
  return std::nullopt;
}

static std::optional<unsigned> getMemRefDynamicSizeOperandIndex(MemRefType baseType, int64_t targetDim) {
  unsigned dynIdx = 0;
  for (int64_t i = 0; i < baseType.getRank(); ++i) {
    if (!baseType.isDynamicDim(i)) {
      continue;
    }
    if (i == targetDim) {
      return dynIdx;
    }
    ++dynIdx;
  }
  return std::nullopt;
}

template <typename AllocLikeOp>
static std::optional<Value> getDynamicDimFromAllocLike(AllocLikeOp allocLikeOp, int64_t dim, MemRefType baseType) {
  auto dynIdx = getMemRefDynamicSizeOperandIndex(baseType, dim);
  if (!dynIdx) {
    return std::nullopt;
  }
  auto dynSizes = allocLikeOp.getDynamicSizes();
  if (*dynIdx >= dynSizes.size()) {
    return std::nullopt;
  }
  return dynSizes[*dynIdx];
}

static std::optional<Value> traceToScfForInitArgFromIterArg(Value curMemRef) {
  auto blockArg = dyn_cast<BlockArgument>(curMemRef);
  if (!blockArg) {
    return std::nullopt;
  }
  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp) {
    return std::nullopt;
  }

  auto iterArgs = forOp.getRegionIterArgs();
  auto initArgs = forOp.getInitArgs();
  for (unsigned i = 0, e = iterArgs.size(); i < e; ++i) {
    if (iterArgs[i] == curMemRef) {
      return initArgs[i];
    }
  }
  return std::nullopt;
}

static std::optional<Value> traceToScfForInitArgFromResult(Value curMemRef) {
  auto forOp = curMemRef.getDefiningOp<scf::ForOp>();
  if (!forOp) {
    return std::nullopt;
  }

  auto results = forOp.getResults();
  auto initArgs = forOp.getInitArgs();
  for (unsigned i = 0, e = results.size(); i < e; ++i) {
    if (results[i] == curMemRef) {
      return initArgs[i];
    }
  }
  return std::nullopt;
}

static SmallVector<int64_t> decomposePermToAdjacentSwaps(ArrayRef<int64_t> perm) {
  auto rank = static_cast<int64_t>(perm.size());
  SmallVector<int64_t> current(rank);
  for (int64_t i = 0; i < rank; ++i) {
    current[i] = i;
  }
  SmallVector<int64_t> swaps;
  for (int64_t i = 0; i < rank; ++i) {
    int64_t target = perm[i];
    if (current[i] == target) {
      continue;
    }
    int64_t j = i + 1;
    for (; j < rank && current[j] != target; ++j) {
    }
    if (j >= rank) {
      return {};
    }
    for (int64_t k = j; k > i; --k) {
      std::swap(current[k], current[k - 1]);
      swaps.push_back(k - 1);
    }
  }
  return swaps;
}

static SmallVector<int64_t> buildAdjacentSwapPerm(int64_t rank, int64_t a) {
  SmallVector<int64_t> perm(rank);
  for (int64_t i = 0; i < rank; ++i) {
    perm[i] = i;
  }
  perm[a] = a + 1;
  perm[a + 1] = a;
  return perm;
}

static bool traceSubviewDim(memref::SubViewOp subviewOp, MemRefType baseType, int64_t dim,
                            std::optional<Value> &resolvedOut) {
  int64_t sourceRank = cast<MemRefType>(subviewOp.getSource().getType()).getRank();
  int64_t resultRank = baseType.getRank();
  int64_t sourceDim = sourceRank - resultRank + dim;
  auto mixedSizes = subviewOp.getMixedSizes();
  if (sourceDim < 0 || sourceDim >= static_cast<int64_t>(mixedSizes.size())) {
    return false;
  }
  if (auto mixVal = mixedSizes[sourceDim].dyn_cast<Value>()) {
    resolvedOut = mixVal;
  }
  return false;
}

static bool traceCollapseDim(memref::CollapseShapeOp collapseOp, Value &curMemRef, MemRefType &baseType, int64_t &dim) {
  auto srcType = cast<MemRefType>(collapseOp.getSrc().getType());
  auto reassoc = collapseOp.getReassociationIndices();
  if (dim < 0 || dim >= static_cast<int64_t>(reassoc.size())) {
    return false;
  }

  int64_t dynSrcDim = -1;
  for (int64_t srcDim : reassoc[dim]) {
    if (!srcType.isDynamicDim(srcDim)) {
      continue;
    }
    if (dynSrcDim != -1) {
      return false;
    }
    dynSrcDim = srcDim;
  }
  if (dynSrcDim < 0) {
    return false;
  }

  curMemRef = collapseOp.getSrc();
  baseType = srcType;
  dim = dynSrcDim;
  return true;
}

static bool expandGroupIsTraceable(ArrayRef<int64_t> group, MemRefType baseType, int64_t dim) {
  for (int64_t resultDim : group) {
    if (resultDim == dim) {
      continue;
    }
    if (baseType.isDynamicDim(resultDim) || baseType.getDimSize(resultDim) != 1) {
      return false;
    }
  }
  return true;
}

static bool traceExpandDim(memref::ExpandShapeOp expandOp, Value &curMemRef, MemRefType &baseType, int64_t &dim) {
  auto srcType = cast<MemRefType>(expandOp.getSrc().getType());
  auto reassoc = expandOp.getReassociationIndices();
  for (int64_t srcDim = 0; srcDim < static_cast<int64_t>(reassoc.size()); ++srcDim) {
    const auto &group = reassoc[srcDim];
    if (!llvm::is_contained(group, dim)) {
      continue;
    }
    if (!srcType.isDynamicDim(srcDim) || !expandGroupIsTraceable(group, baseType, dim)) {
      return false;
    }
    curMemRef = expandOp.getSrc();
    baseType = srcType;
    dim = srcDim;
    return true;
  }
  return false;
}

template <typename AllocLikeOp>
static bool traceAllocDim(AllocLikeOp allocLikeOp, int64_t dim, MemRefType baseType,
                          std::optional<Value> &resolvedOut) {
  if (auto dimVal = getDynamicDimFromAllocLike(allocLikeOp, dim, baseType)) {
    resolvedOut = *dimVal;
  }
  return false;
}

static bool advanceMemRefDimTrace(Value &curMemRef, MemRefType &baseType, int64_t &dim,
                                  std::optional<Value> &resolvedOut) {
  resolvedOut.reset();
  if (auto bitcastOp = curMemRef.getDefiningOp<hivm::BitcastOp>()) {
    curMemRef = bitcastOp.getSrc();
    return true;
  }
  if (auto subviewOp = curMemRef.getDefiningOp<memref::SubViewOp>()) {
    return traceSubviewDim(subviewOp, baseType, dim, resolvedOut);
  }
  if (auto collapseOp = curMemRef.getDefiningOp<memref::CollapseShapeOp>()) {
    return traceCollapseDim(collapseOp, curMemRef, baseType, dim);
  }
  if (auto expandOp = curMemRef.getDefiningOp<memref::ExpandShapeOp>()) {
    return traceExpandDim(expandOp, curMemRef, baseType, dim);
  }
  if (auto allocOp = curMemRef.getDefiningOp<memref::AllocOp>()) {
    return traceAllocDim(allocOp, dim, baseType, resolvedOut);
  }
  if (auto allocaOp = curMemRef.getDefiningOp<memref::AllocaOp>()) {
    return traceAllocDim(allocaOp, dim, baseType, resolvedOut);
  }
  if (auto next = traceToScfForInitArgFromIterArg(curMemRef)) {
    curMemRef = *next;
    return true;
  }
  if (auto next = traceToScfForInitArgFromResult(curMemRef)) {
    curMemRef = *next;
    return true;
  }
  return false;
}

static FailureOr<Value> getMemRefDimValue(Value memref, int64_t dim) {
  auto baseType = dyn_cast<MemRefType>(memref.getType());
  if (!baseType) {
    return failure();
  }

  Value curMemRef = memref;
  for (unsigned step = 0; step < 32; ++step) {
    std::optional<Value> resolved;
    if (!advanceMemRefDimTrace(curMemRef, baseType, dim, resolved)) {
      if (resolved) {
        return *resolved;
      }
      return failure();
    }
  }
  return failure();
}

static FailureOr<Value> allocMemRef(ConversionPatternRewriter &rewriter, Location loc, MemRefType type,
                                    Value dimSource) {
  SmallVector<Value> allocOperands;
  for (int i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i)) {
      auto dimVal = getMemRefDimValue(dimSource, i);
      if (failed(dimVal)) {
        return failure();
      }
      allocOperands.push_back(*dimVal);
    }
  }
  auto allocOp = rewriter.create<memref::AllocOp>(loc, type, allocOperands);
  return allocOp.getResult();
}

static FailureOr<DenseI64ArrayAttr> inferElementwiseBroadcastAttr(Value lhs, Value rhs, MemRefType resultTy,
                                                                  PatternRewriter &rewriter) {
  SmallVector<int64_t> broadcastDims;
  auto collectFromOperand = [&](Value operand) -> LogicalResult {
    auto operandTy = dyn_cast<MemRefType>(operand.getType());
    if (!operandTy) {
      return success();
    }
    if (operandTy.getRank() != resultTy.getRank()) {
      return failure();
    }

    for (int64_t i = 0; i < resultTy.getRank(); ++i) {
      bool operandStaticOne = !operandTy.isDynamicDim(i) && operandTy.getDimSize(i) == 1;
      bool resultStaticOne = !resultTy.isDynamicDim(i) && resultTy.getDimSize(i) == 1;
      if (operandStaticOne && !resultStaticOne) {
        broadcastDims.push_back(i);
        continue;
      }

      if (!operandTy.isDynamicDim(i) && !resultTy.isDynamicDim(i) &&
          operandTy.getDimSize(i) != resultTy.getDimSize(i)) {
        return failure();
      }
    }
    return success();
  };

  if (failed(collectFromOperand(lhs)) || failed(collectFromOperand(rhs))) {
    return failure();
  }

  llvm::sort(broadcastDims);
  broadcastDims.erase(std::unique(broadcastDims.begin(), broadcastDims.end()), broadcastDims.end());
  return rewriter.getDenseI64ArrayAttr(broadcastDims);
}

static DenseI64ArrayAttr getElementwiseBroadcastAttr(Value lhs, Value rhs, Value resBuf, PatternRewriter &rewriter) {
  auto resTy = dyn_cast<MemRefType>(resBuf.getType());
  if (!resTy) {
    return {};
  }

  auto inferredAttr = inferElementwiseBroadcastAttr(lhs, rhs, resTy, rewriter);
  if (failed(inferredAttr)) {
    return {};
  }
  return *inferredAttr;
}

static FailureOr<SmallVector<int64_t>> inferResultMaxShapeForBufferMark(Operation *op, MemRefType resultTy) {
  if ((op == nullptr) || op->getNumResults() == 0) {
    return failure();
  }
  Type resultType = op->getResult(0).getType();
  if (auto npuTy = dyn_cast<npuvector::NPUVectorType>(resultType)) {
    return inferNPUVectorMaxShapeFromOperands(op, npuTy);
  }
  if (resultTy.hasStaticShape()) {
    return SmallVector<int64_t>(resultTy.getShape());
  }
  return failure();
}

static void markInlineBroadcastOperandBufferSizeAtLeast(PatternRewriter &rewriter, Location loc, Value operand,
                                                        MemRefType resultTy, ArrayRef<int64_t> resultMaxShape,
                                                        Operation *anchor);

static void markInlineBroadcastOperandsBufferSizeAtLeast(PatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
                                                         Value resBuf, Operation *anchor) {
  auto resultTy = dyn_cast<MemRefType>(resBuf.getType());
  if (!resultTy) {
    return;
  }

  auto resultMaxShape = inferResultMaxShapeForBufferMark(anchor, resultTy);
  if (failed(resultMaxShape)) {
    return;
  }

  markInlineBroadcastOperandBufferSizeAtLeast(rewriter, loc, lhs, resultTy, *resultMaxShape, anchor);
  markInlineBroadcastOperandBufferSizeAtLeast(rewriter, loc, rhs, resultTy, *resultMaxShape, anchor);
}

template <typename HIVMOp>
static void createHIVMBinaryOp(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf,
                               Operation *anchor = nullptr) {
  DenseI64ArrayAttr broadcastAttr = getElementwiseBroadcastAttr(lhs, rhs, resBuf, rewriter);

  if (broadcastAttr && !broadcastAttr.empty()) {
    markInlineBroadcastOperandsBufferSizeAtLeast(rewriter, loc, lhs, rhs, resBuf, anchor);
    rewriter.create<HIVMOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf},
                            rewriter.getDenseI64ArrayAttr({}), broadcastAttr);
    return;
  }

  rewriter.create<HIVMOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf});
}

template <typename HIVMOp>
struct HIVMElementwiseBinaryCreator {
  static void create(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf,
                     Operation *anchor = nullptr) {
    createHIVMBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf, anchor);
  }
};

template <>
struct HIVMElementwiseBinaryCreator<hivm::VShROp> {
  static void create(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf,
                     Operation *anchor = nullptr) {
    DenseI64ArrayAttr broadcastAttr = getElementwiseBroadcastAttr(lhs, rhs, resBuf, rewriter);
    if (broadcastAttr && !broadcastAttr.empty()) {
      markInlineBroadcastOperandsBufferSizeAtLeast(rewriter, loc, lhs, rhs, resBuf, anchor);
      rewriter.create<hivm::VShROp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf},
                                    rewriter.getBoolAttr(true), rewriter.getDenseI64ArrayAttr({}), broadcastAttr);
      return;
    }

    rewriter.create<hivm::VShROp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf},
                                  rewriter.getBoolAttr(true));
  }
};

template <typename HIVMOp>
static void createHIVMElementwiseBinaryOp(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
                                          Value resBuf, Operation *anchor = nullptr) {
  HIVMElementwiseBinaryCreator<HIVMOp>::create(rewriter, loc, lhs, rhs, resBuf, anchor);
}

// If this binary op's sole result is the i-th scf.yield operand inside an scf.for, reuse
// forOp.getInitArgs()[i] as the HIVM output buffer (in-place on the loop-carried memref).
// Returns null to fall back to allocMemRef when not applicable.
static Value tryGetInPlaceInitIfResultIsYieldOperand(Operation *arithOp) {
  Block *body = arithOp->getBlock();
  auto yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
  auto forOp = dyn_cast<scf::ForOp>(body->getParent()->getParentOp());
  if (!yieldOp || !forOp) {
    return {};
  }
  Value result = arithOp->getResult(0);
  for (unsigned i = 0, n = yieldOp.getNumOperands(); i < n; ++i) {
    if (yieldOp.getOperand(i) == result && i < forOp.getInitArgs().size()) {
      return forOp.getInitArgs()[i];
    }
  }
  return {};
}

template <typename ArithOp, typename HIVMOp>
struct BinaryArithToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  static constexpr bool isCommutative() {
    return std::is_same_v<ArithOp, arith::AddFOp> || std::is_same_v<ArithOp, arith::AddIOp> ||
           std::is_same_v<ArithOp, arith::MulFOp> || std::is_same_v<ArithOp, arith::MulIOp> ||
           std::is_same_v<ArithOp, arith::MaxSIOp> || std::is_same_v<ArithOp, arith::MaxUIOp> ||
           std::is_same_v<ArithOp, arith::MinSIOp> || std::is_same_v<ArithOp, arith::MinUIOp>;
  }

  LogicalResult matchAndRewrite(ArithOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto shapeAndElem = getShapeAndElemType(op.getResult().getType());
    if (!shapeAndElem) {
      return failure();
    }

    Value lhsMemRef = adaptor.getLhs();
    Value rhsMemRef = adaptor.getRhs();

    bool rhsIsMemRef = isa<MemRefType>(rhsMemRef.getType());

    if (!isa<MemRefType>(lhsMemRef.getType())) {
      // Swap operands for commutative ops if the left-hand side is a scalar and the right-hand side is a memref.
      if constexpr (isCommutative()) {
        bool lhsIsScalar = isScalarType(lhsMemRef.getType());
        if (lhsIsScalar && rhsIsMemRef) {
          std::swap(lhsMemRef, rhsMemRef);
          rhsIsMemRef = isa<MemRefType>(rhsMemRef.getType());
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    }

    bool rhsIsScalar = isScalarType(rhsMemRef.getType());
    if (!rhsIsMemRef && !rhsIsScalar) {
      return failure();
    }

    auto memRefType = MemRefType::get(shapeAndElem->first, shapeAndElem->second);
    Value resBuf = tryGetInPlaceInitIfResultIsYieldOperand(op.getOperation());
    if (!resBuf) {
      Value dimSource = lhsMemRef;
      auto resBufOr = allocMemRef(rewriter, loc, memRefType, dimSource);
      if (failed(resBufOr) && rhsIsMemRef) {
        dimSource = rhsMemRef;
        resBufOr = allocMemRef(rewriter, loc, memRefType, dimSource);
      }
      if (failed(resBufOr)) {
        return failure();
      }
      resBuf = *resBufOr;
      Value markSource = dimSource;
      if (rhsIsMemRef) {
        auto rhsType = cast<MemRefType>(rhsMemRef.getType());
        if (rhsType.getRank() == memRefType.getRank() && llvm::equal(rhsType.getShape(), memRefType.getShape())) {
          markSource = rhsMemRef;
        }
      }
      if (!setNPUVectorResultBufferSizeMark(rewriter, loc, op.getOperation(), resBuf) &&
          !propagateBufferSizeMark(rewriter, loc, markSource, resBuf) && markSource != dimSource) {
        propagateBufferSizeMark(rewriter, loc, dimSource, resBuf);
      }
    }

    createHIVMBinaryOp<HIVMOp>(rewriter, loc, lhsMemRef, rhsMemRef, resBuf, op.getOperation());

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

inline bool isOverFlowMode(Type inType, Type outType) {
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  return (isI16ToI8 || isI32ToI16 || isI32ToI8 || isI64ToI8 || isI64ToI16 || isI64ToI32);
}

static bool isVcSupportedFloatCastPair(Type inType, Type outType) {
  return (inType.isF16() || inType.isBF16() || inType.isF32() || inType.isF64()) &&
         (outType.isF16() || outType.isBF16() || outType.isF32() || outType.isF64());
}

template <typename CastOp>
struct UnaryArithToHIVMCast : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<CastOp>::OpAdaptor;

  static hivm::RoundMode selectRoundModeForTruncF(Type inType, Type outType) {
    if (isVcSupportedFloatCastPair(inType, outType)) {
      return hivm::RoundMode::RINT;
    }
    llvm_unreachable("unsupported datatype for arith::TruncFOp to hivm");
  }

  static hivm::RoundMode selectRoundModeForExtF(Type inType, Type outType) {
    if (isVcSupportedFloatCastPair(inType, outType)) {
      return hivm::RoundMode::RINT;
    }
    llvm_unreachable("unsupported datatype for arith::ExtFOp to hivm");
  }

  static hivm::RoundMode selectRoundMode(CastOp op) {
    auto inType = getElementTypeOrSelf(op.getOperand().getType());
    auto outType = getElementTypeOrSelf(op.getResult().getType());
    if (isa<arith::TruncFOp>(op)) {
      return selectRoundModeForTruncF(inType, outType);
    }
    if (isa<arith::ExtFOp>(op)) {
      return selectRoundModeForExtF(inType, outType);
    } else if (isa<arith::TruncIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::RINT;
    } else if (isa<arith::ExtSIOp>(op) || isa<arith::ExtUIOp>(op) || isa<arith::SIToFPOp>(op) ||
               isa<arith::UIToFPOp>(op)) {
      return hivm::RoundMode::RINT;
    } else if (isa<arith::FPToSIOp>(op) || isa<arith::FPToUIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::TRUNC;
    }
    llvm_unreachable("unsupported arith op to hivm");
  }

  LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    ArrayRef<int64_t> shape;
    Type elemType;
    if (auto vectorType = dyn_cast<VectorType>(resType)) {
      shape = vectorType.getShape();
      elemType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      shape = npuVectorType.getShape();
      elemType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    Value srcMemRef = adaptor.getOperands()[0];
    Type srcElemType = getElementTypeOrSelf(srcMemRef.getType());

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    if (memRefType.getNumDynamicDims() > 0) {
      for (int i = 0; i < memRefType.getRank(); ++i) {
        if (memRefType.isDynamicDim(i)) {
          auto dimVal = getMemRefDimValue(srcMemRef, i);
          if (failed(dimVal)) {
            return failure();
          }
          allocOperands.push_back(*dimVal);
        }
      }
    }
    hivm::RoundMode rounding = selectRoundMode(op);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
    auto setCastBufferSizeMark = [&](Value buffer, Type markElemType) {
      if (setNPUVectorResultBufferSizeMark(rewriter, loc, op.getOperation(), buffer, markElemType)) {
        return true;
      }
      return propagateBufferSizeMark(rewriter, loc, srcMemRef, buffer);
    };

    Value resBuf;
    if ((isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) && srcElemType.isInteger(8) && elemType.isF32()) {
      auto midMemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midBuf = rewriter.create<memref::AllocOp>(loc, midMemRefType, allocOperands);
      setCastBufferSizeMark(midBuf, rewriter.getF16Type());
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midBuf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      setCastBufferSizeMark(resBuf, elemType);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midBuf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else if (isa<arith::ExtUIOp>(op) && srcElemType.isInteger(1) && elemType.isInteger(32)) {
      auto i8MemRefType = MemRefType::get(shape, rewriter.getI8Type());
      Value i8Buf = rewriter.create<memref::AllocOp>(loc, i8MemRefType, allocOperands);
      setCastBufferSizeMark(i8Buf, rewriter.getI8Type());
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, i8Buf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      setCastBufferSizeMark(resBuf, elemType);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, i8Buf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else if (isa<arith::ExtUIOp>(op) && srcElemType.isInteger(1) && elemType.isInteger(64)) {
      auto f32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value f32Buf = rewriter.create<memref::AllocOp>(loc, f32MemRefType, allocOperands);
      setCastBufferSizeMark(f32Buf, rewriter.getF32Type());
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, f32Buf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      setCastBufferSizeMark(resBuf, elemType);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f32Buf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else {
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      setCastBufferSizeMark(resBuf, elemType);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, resBuf, roundingAttr, hivm::TypeFnAttr{});
    }

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename CastOp>
struct UnaryNPUVectorToHIVMCast : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<CastOp>::OpAdaptor;

  static hivm::RoundMode selectRoundModeForTruncF(Type inType, Type outType) {
    if (isVcSupportedFloatCastPair(inType, outType)) {
      return hivm::RoundMode::RINT;
    }
    llvm_unreachable("unsupported datatype for npuvector::TruncFOp to hivm");
  }

  static hivm::RoundMode selectRoundModeForExtF(Type inType, Type outType) {
    if (isVcSupportedFloatCastPair(inType, outType)) {
      return hivm::RoundMode::RINT;
    }
    llvm_unreachable("unsupported datatype for npuvector::ExtFOp to hivm");
  }

  static hivm::RoundMode selectRoundMode(CastOp op) {
    auto inType = getElementTypeOrSelf(op.getOperand().getType());
    auto outType = getElementTypeOrSelf(op.getResult().getType());

    if (isa<npuvector::TruncFOp>(op)) {
      return selectRoundModeForTruncF(inType, outType);
    }
    if (isa<npuvector::ExtFOp>(op)) {
      return selectRoundModeForExtF(inType, outType);
    } else if (isa<npuvector::TruncIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::RINT;
    } else if (isa<npuvector::ExtSIOp>(op) || isa<npuvector::ExtUIOp>(op) || isa<npuvector::SIToFPOp>(op) ||
               isa<npuvector::UIToFPOp>(op)) {
      return hivm::RoundMode::RINT;
    } else if (isa<npuvector::FPToSIOp>(op) || isa<npuvector::FPToUIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::TRUNC;
    }
    llvm_unreachable("unsupported npuvector op to hivm");
  }

  // Returns the chain of element types (intermediate types plus the final
  // destination type) to cast through. Some casts are not supported directly by
  // HIR and must go through intermediate types:
  //   i1 -> i32 : i1 -> i8 -> i32
  //   i1 -> i64 : i1 -> f32 -> i64
  //   i8 -> i64 : i8 -> f16 -> f32 -> i64
  //   i8 -> f32 : i8 -> f16 -> f32
  static SmallVector<Type> getCastElementChain(CastOp op, Type srcElemType, Type dstElemType, OpBuilder &b) {
    if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(1) && dstElemType.isInteger(32)) {
      return {b.getI8Type(), dstElemType};
    }
    if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(1) && dstElemType.isInteger(64)) {
      return {b.getF32Type(), dstElemType};
    }
    if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(8) && dstElemType.isInteger(64)) {
      return {b.getF16Type(), b.getF32Type(), dstElemType};
    }
    if ((isa<npuvector::SIToFPOp>(op) || isa<npuvector::UIToFPOp>(op)) && srcElemType.isInteger(8) &&
        dstElemType.isF32()) {
      return {b.getF16Type(), dstElemType};
    }
    return {dstElemType};
  }

  LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!npuVectorType) {
      return failure();
    }

    ArrayRef<int64_t> shape = npuVectorType.getShape();
    Type elemType = npuVectorType.getElementType();

    Value srcMemRef = adaptor.getOperands()[0];
    Type srcElemType = getElementTypeOrSelf(srcMemRef.getType());

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(srcMemRef, i);
        if (failed(dimVal)) {
          return failure();
        }
        allocOperands.push_back(*dimVal);
      }
    }
    hivm::RoundMode rounding = selectRoundMode(op);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);
    auto maxShape = inferNPUVectorMaxShapeFromOperands(op.getOperation(), npuVectorType);
    auto setCastBufferSizeMark = [&](Value buffer, Type markElemType) {
      if (succeeded(maxShape) &&
          setNPUVectorBufferSizeMark(rewriter, loc, npuVectorType, markElemType, *maxShape, buffer)) {
        return true;
      }
      return propagateBufferSizeMark(rewriter, loc, srcMemRef, buffer);
    };

    // Some casts are not supported directly by HIR and must go through
    // intermediate types; emit the cast chain step by step.
    Value cur = srcMemRef;
    Value resBuf;
    for (Type stepElem : getCastElementChain(op, srcElemType, elemType, rewriter)) {
      resBuf = rewriter.create<memref::AllocOp>(loc, MemRefType::get(shape, stepElem), allocOperands);
      setCastBufferSizeMark(resBuf, stepElem);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, cur, resBuf, roundingAttr, hivm::TypeFnAttr{});
      cur = resBuf;
    }

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithBitcastToHIVM : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern<arith::BitcastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<arith::BitcastOp>::OpAdaptor;

  LogicalResult matchAndRewrite(arith::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    ArrayRef<int64_t> shape;
    Type elemType;
    if (auto vectorType = dyn_cast<VectorType>(resType)) {
      shape = vectorType.getShape();
      elemType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      shape = npuVectorType.getShape();
      elemType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    Value srcMemRef = adaptor.getIn();

    auto memRefType = MemRefType::get(shape, elemType);

    Value res = rewriter.create<hivm::BitcastOp>(loc, memRefType, srcMemRef);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NPUVectorBitcastToHIVM : public OpConversionPattern<npuvector::BitcastOp> {
  using OpConversionPattern<npuvector::BitcastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<npuvector::BitcastOp>::OpAdaptor;

  LogicalResult matchAndRewrite(npuvector::BitcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!npuVectorType) {
      return failure();
    }

    ArrayRef<int64_t> shape = npuVectorType.getShape();
    Type elemType = npuVectorType.getElementType();

    Value srcMemRef = adaptor.getIn();
    auto memRefType = MemRefType::get(shape, elemType);

    Value res = rewriter.create<hivm::BitcastOp>(loc, memRefType, srcMemRef);
    rewriter.replaceOp(op, res);
    return success();
  }
};

static FailureOr<Value> castI8ToF16ForHIVMValue(ConversionPatternRewriter &rewriter, Location loc, Value input,
                                                std::optional<int64_t> alignedBufferBytes = std::nullopt) {
  Type elemType = getElementTypeOrSelf(input.getType());
  if (!elemType.isInteger(8)) {
    return input;
  }

  if (auto inputType = dyn_cast<MemRefType>(input.getType())) {
    auto f16Type = MemRefType::get(inputType.getShape(), rewriter.getF16Type());
    auto f16Buf = allocMemRef(rewriter, loc, f16Type, input);
    if (failed(f16Buf)) {
      return failure();
    }
    if (alignedBufferBytes && *alignedBufferBytes > 0) {
      // Same-shape i8 -> f16 temps still need their own stride-aligned byte size.
      setBufferSizeMark(rewriter, loc, *f16Buf, *alignedBufferBytes);
    } else {
      propagateBufferSizeMark(rewriter, loc, input, *f16Buf);
    }
    auto roundAttr = rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);
    rewriter.create<hivm::VCastOp>(loc, TypeRange{}, input, *f16Buf, roundAttr, hivm::TypeFnAttr{});
    return *f16Buf;
  }

  if (!isScalarType(input.getType())) {
    return failure();
  }
  return rewriter.create<arith::SIToFPOp>(loc, rewriter.getF16Type(), input).getResult();
}

static LogicalResult legalizeI8VCmpOperands(ConversionPatternRewriter &rewriter, Location loc, Value &lhs, Value &rhs,
                                            std::optional<int64_t> alignedBufferBytes = std::nullopt) {
  auto castedLhs = castI8ToF16ForHIVMValue(rewriter, loc, lhs, alignedBufferBytes);
  if (failed(castedLhs)) {
    return failure();
  }
  auto castedRhs = castI8ToF16ForHIVMValue(rewriter, loc, rhs, alignedBufferBytes);
  if (failed(castedRhs)) {
    return failure();
  }
  lhs = *castedLhs;
  rhs = *castedRhs;
  return success();
}

static std::optional<int64_t> computeNPUVectorF16TempBufferBytes(Operation *op, Type resType,
                                                                 ConversionPatternRewriter &rewriter) {
  auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType);
  if (!npuVectorType) {
    return std::nullopt;
  }

  auto maxShape = inferNPUVectorMaxShapeFromOperands(op, npuVectorType);
  if (failed(maxShape)) {
    return std::nullopt;
  }

  int64_t bytes = computeBishengNpuVectorStorageBytes(*maxShape, npuVectorType.getShape(), rewriter.getF16Type());
  if (bytes <= 0 || bytes == LLONG_MAX) {
    return std::nullopt;
  }
  return bytes;
}

template <typename CompareOp>
struct ArithCmpToHIVM : OpConversionPattern<CompareOp> {
  using OpConversionPattern<CompareOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<CompareOp>::OpAdaptor;

  static hivm::CompareMode selectPredicate(arith::CmpFOp op) {
    switch (op.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        return hivm::CompareMode::EQ;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        return hivm::CompareMode::NE;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        return hivm::CompareMode::LE;
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        return hivm::CompareMode::LT;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        return hivm::CompareMode::GE;
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        return hivm::CompareMode::GT;
      default:
        llvm_unreachable("unsupported arith cmp predicate to hivm");
    }
  }

  static hivm::CompareMode selectPredicate(arith::CmpIOp op) {
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return hivm::CompareMode::EQ;
      case arith::CmpIPredicate::ne:
        return hivm::CompareMode::NE;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return hivm::CompareMode::LT;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return hivm::CompareMode::GT;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return hivm::CompareMode::LE;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return hivm::CompareMode::GE;
    }
    llvm_unreachable("unsupported arith cmp predicate to hivm");
  }

  LogicalResult matchAndRewrite(CompareOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    ArrayRef<int64_t> shape;
    Type elemType;
    if (auto vectorType = dyn_cast<VectorType>(resType)) {
      shape = vectorType.getShape();
      elemType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      shape = npuVectorType.getShape();
      elemType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    std::optional<int64_t> resultBufferBytes;
    if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      auto maxShape = inferNPUVectorMaxShapeFromOperands(op.getOperation(), npuVectorType);
      if (succeeded(maxShape)) {
        resultBufferBytes = computeBishengNpuVectorStorageBytes(*maxShape, npuVectorType.getShape(), elemType);
      }
    }

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(lhs, i);
        if (failed(dimVal)) {
          return failure();
        }
        allocOperands.push_back(*dimVal);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    if (!resultBufferBytes || !setBufferSizeMark(rewriter, loc, resBuf, *resultBufferBytes)) {
      propagateSelectBufferSizeMark(rewriter, loc, lhs, rhs, resBuf);
    }

    hivm::CompareMode predicate = selectPredicate(op);
    auto predicateAttr = rewriter.getAttr<hivm::CompareModeAttr>(predicate);

    if (failed(legalizeI8VCmpOperands(rewriter, loc, lhs, rhs))) {
      return failure();
    }

    DenseI64ArrayAttr broadcastAttr = getElementwiseBroadcastAttr(lhs, rhs, resBuf, rewriter);
    if (broadcastAttr && !broadcastAttr.empty()) {
      markInlineBroadcastOperandsBufferSizeAtLeast(rewriter, loc, lhs, rhs, resBuf, op.getOperation());
      rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr,
                                    rewriter.getDenseI64ArrayAttr({}), broadcastAttr);
    } else {
      rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);
    }

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename CompareOp>
struct NPUVectorCmpToHIVM : OpConversionPattern<CompareOp> {
  using OpConversionPattern<CompareOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<CompareOp>::OpAdaptor;

  static hivm::CompareMode selectPredicate(npuvector::CmpFOp op) {
    switch (op.getPredicate()) {
      case arith::CmpFPredicate::OEQ:
      case arith::CmpFPredicate::UEQ:
        return hivm::CompareMode::EQ;
      case arith::CmpFPredicate::ONE:
      case arith::CmpFPredicate::UNE:
        return hivm::CompareMode::NE;
      case arith::CmpFPredicate::OLE:
      case arith::CmpFPredicate::ULE:
        return hivm::CompareMode::LE;
      case arith::CmpFPredicate::OLT:
      case arith::CmpFPredicate::ULT:
        return hivm::CompareMode::LT;
      case arith::CmpFPredicate::OGE:
      case arith::CmpFPredicate::UGE:
        return hivm::CompareMode::GE;
      case arith::CmpFPredicate::OGT:
      case arith::CmpFPredicate::UGT:
        return hivm::CompareMode::GT;
      default:
        llvm_unreachable("unsupported npuvector cmp predicate to hivm");
    }
  }

  static hivm::CompareMode selectPredicate(npuvector::CmpIOp op) {
    switch (op.getPredicate()) {
      case arith::CmpIPredicate::eq:
        return hivm::CompareMode::EQ;
      case arith::CmpIPredicate::ne:
        return hivm::CompareMode::NE;
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return hivm::CompareMode::LT;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return hivm::CompareMode::GT;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return hivm::CompareMode::LE;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return hivm::CompareMode::GE;
    }
    llvm_unreachable("unsupported npuvector cmp predicate to hivm");
  }

  LogicalResult matchAndRewrite(CompareOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
    if (!npuVectorType) {
      return failure();
    }

    ArrayRef<int64_t> shape = npuVectorType.getShape();
    Type elemType = npuVectorType.getElementType();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    FailureOr<SmallVector<int64_t>> maxShape = inferNPUVectorMaxShapeFromOperands(op.getOperation(), npuVectorType);
    int64_t resultBufferBytes = 0;
    std::optional<int64_t> i8ToF16BufferBytes;
    if (succeeded(maxShape)) {
      resultBufferBytes = computeBishengNpuVectorStorageBytes(*maxShape, npuVectorType.getShape(), elemType);
      int64_t f16Bytes =
        computeBishengNpuVectorStorageBytes(*maxShape, npuVectorType.getShape(), rewriter.getF16Type());
      if (f16Bytes > 0 && f16Bytes != LLONG_MAX) {
        i8ToF16BufferBytes = f16Bytes;
      }
    }

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(lhs, i);
        if (failed(dimVal)) {
          return failure();
        }
        allocOperands.push_back(*dimVal);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    if (failed(maxShape) || !setBufferSizeMark(rewriter, loc, resBuf, resultBufferBytes)) {
      propagateSelectBufferSizeMark(rewriter, loc, lhs, rhs, resBuf);
    }

    hivm::CompareMode predicate = selectPredicate(op);
    auto predicateAttr = rewriter.getAttr<hivm::CompareModeAttr>(predicate);

    if (failed(legalizeI8VCmpOperands(rewriter, loc, lhs, rhs, i8ToF16BufferBytes))) {
      return failure();
    }

    DenseI64ArrayAttr broadcastAttr = getElementwiseBroadcastAttr(lhs, rhs, resBuf, rewriter);
    if (broadcastAttr && !broadcastAttr.empty()) {
      markInlineBroadcastOperandsBufferSizeAtLeast(rewriter, loc, lhs, rhs, resBuf, op.getOperation());
      rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr,
                                    rewriter.getDenseI64ArrayAttr({}), broadcastAttr);
    } else {
      rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);
    }

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename ArithOp>
struct ArithMulExtToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  LogicalResult matchAndRewrite(ArithOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type lowType = op.getLow().getType();
    Type highType = op.getHigh().getType();

    ArrayRef<int64_t> lowShape;
    Type lowElemType;
    auto lowVectorType = dyn_cast<VectorType>(lowType);
    if (lowVectorType) {
      lowShape = lowVectorType.getShape();
      lowElemType = lowVectorType.getElementType();
    } else {
      auto lowNpuVectorType = dyn_cast<npuvector::NPUVectorType>(lowType);
      if (lowNpuVectorType) {
        lowShape = lowNpuVectorType.getShape();
        lowElemType = lowNpuVectorType.getElementType();
      } else {
        return failure();
      }
    }

    ArrayRef<int64_t> highShape;
    Type highElemType;
    auto highVectorType = dyn_cast<VectorType>(highType);
    if (highVectorType) {
      highShape = highVectorType.getShape();
      highElemType = highVectorType.getElementType();
    } else {
      auto highNpuVectorType = dyn_cast<npuvector::NPUVectorType>(highType);
      if (highNpuVectorType) {
        highShape = highNpuVectorType.getShape();
        highElemType = highNpuVectorType.getElementType();
      } else {
        return failure();
      }
    }

    auto lowMemRefType = MemRefType::get(lowShape, lowElemType);
    auto highMemRefType = MemRefType::get(highShape, highElemType);

    SmallVector<Value> lowAllocOperands;
    for (int i = 0; i < lowMemRefType.getRank(); ++i) {
      if (lowMemRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(lhs, i);
        if (failed(dimVal)) {
          return failure();
        }
        lowAllocOperands.push_back(*dimVal);
      }
    }
    Value lowBuf = rewriter.create<memref::AllocOp>(loc, lowMemRefType, lowAllocOperands);
    if (!setNPUVectorValueBufferSizeMark(rewriter, loc, op.getLow(), lowBuf)) {
      propagateBufferSizeMark(rewriter, loc, lhs, lowBuf);
    }

    SmallVector<Value> highAllocOperands;
    for (int i = 0; i < highMemRefType.getRank(); ++i) {
      if (highMemRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(lhs, i);
        if (failed(dimVal)) {
          return failure();
        }
        highAllocOperands.push_back(*dimVal);
      }
    }
    Value highBuf = rewriter.create<memref::AllocOp>(loc, highMemRefType, highAllocOperands);
    if (!setNPUVectorValueBufferSizeMark(rewriter, loc, op.getHigh(), highBuf)) {
      propagateBufferSizeMark(rewriter, loc, lhs, highBuf);
    }

    SmallVector<Value> dsts = {lowBuf, highBuf};

    rewriter.create<hivm::VMulExtOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange(dsts));

    rewriter.replaceOp(op, dsts);
    return success();
  }
};

template <typename ArithOp, typename HIVMOp>
struct ElementwiseOpToHIVMBinary : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  static constexpr bool isCommutative() {
    return std::is_same_v<ArithOp, arith::MinNumFOp> || std::is_same_v<ArithOp, arith::MinimumFOp> ||
           std::is_same_v<ArithOp, arith::MaxNumFOp> || std::is_same_v<ArithOp, arith::MaximumFOp>;
  }

  LogicalResult matchAndRewrite(ArithOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto shapeAndElem = getShapeAndElemType(op.getResult().getType());
    if (!shapeAndElem) {
      return failure();
    }

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    bool rhsIsMemRef = isa<MemRefType>(rhs.getType());

    if (!isa<MemRefType>(lhs.getType())) {
      if constexpr (isCommutative()) {
        bool lhsIsScalar = isScalarType(lhs.getType());
        if (lhsIsScalar && rhsIsMemRef) {
          std::swap(lhs, rhs);
          rhsIsMemRef = isa<MemRefType>(rhs.getType());
        } else {
          return failure();
        }
      } else {
        return failure();
      }
    }

    bool rhsIsScalar = isScalarType(rhs.getType());
    if (!rhsIsMemRef && !rhsIsScalar) {
      return failure();
    }

    auto memRefType = MemRefType::get(shapeAndElem->first, shapeAndElem->second);
    Value resBuf = tryGetInPlaceInitIfResultIsYieldOperand(op.getOperation());
    if (!resBuf) {
      Value dimSource = lhs;
      auto resBufOr = allocMemRef(rewriter, loc, memRefType, dimSource);
      if (failed(resBufOr) && rhsIsMemRef) {
        dimSource = rhs;
        resBufOr = allocMemRef(rewriter, loc, memRefType, dimSource);
      }
      if (failed(resBufOr)) {
        return failure();
      }
      resBuf = *resBufOr;
      if (!setNPUVectorResultBufferSizeMark(rewriter, loc, op.getOperation(), resBuf)) {
        propagateBufferSizeMark(rewriter, loc, dimSource, resBuf);
      }
    }

    createHIVMElementwiseBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf, op.getOperation());

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename SelectOp>
struct ArithSelectToHIVM : public OpConversionPattern<SelectOp> {
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SelectOp>::OpAdaptor;

  LogicalResult matchAndRewrite(SelectOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    ArrayRef<int64_t> shape;
    Type elemType;
    if (auto vectorType = dyn_cast<VectorType>(resType)) {
      shape = vectorType.getShape();
      elemType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      shape = npuVectorType.getShape();
      elemType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    Value cond = adaptor.getCondition();
    Value trueVal = adaptor.getTrueValue();
    Value falseVal = adaptor.getFalseValue();
    std::optional<int64_t> f16TempBufferBytes;
    if (elemType.isInteger(8)) {
      f16TempBufferBytes = computeNPUVectorF16TempBufferBytes(op.getOperation(), resType, rewriter);
    }

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(cond, i);
        if (failed(dimVal)) {
          return failure();
        }
        allocOperands.push_back(*dimVal);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    if (!setNPUVectorResultBufferSizeMark(rewriter, loc, op.getOperation(), resBuf)) {
      propagateSelectBufferSizeMark(rewriter, loc, trueVal, falseVal, resBuf);
    }

    if (elemType.isInteger(8)) {
      auto castedTrueVal = castI8ToF16ForHIVMValue(rewriter, loc, trueVal, f16TempBufferBytes);
      if (failed(castedTrueVal)) {
        return failure();
      }
      auto castedFalseVal = castI8ToF16ForHIVMValue(rewriter, loc, falseVal, f16TempBufferBytes);
      if (failed(castedFalseVal)) {
        return failure();
      }

      auto f16MemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value f16ResBuf = rewriter.create<memref::AllocOp>(loc, f16MemRefType, allocOperands);
      if (!f16TempBufferBytes || !setBufferSizeMark(rewriter, loc, f16ResBuf, *f16TempBufferBytes)) {
        propagateSelectBufferSizeMark(rewriter, loc, *castedTrueVal, *castedFalseVal, f16ResBuf);
      }

      rewriter.create<hivm::VSelOp>(loc, TypeRange{}, ValueRange{cond, *castedTrueVal, *castedFalseVal},
                                    ValueRange{f16ResBuf}, Value(), SmallVector<int64_t>{}, SmallVector<int64_t>{});
      auto roundAttr = rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::TRUNC);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f16ResBuf, resBuf, roundAttr, hivm::TypeFnAttr{});

      rewriter.replaceOp(op, resBuf);
      return success();
    }

    rewriter.create<hivm::VSelOp>(loc, TypeRange{}, ValueRange{cond, trueVal, falseVal}, ValueRange{resBuf}, Value(),
                                  SmallVector<int64_t>{}, SmallVector<int64_t>{});

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithConstantToHIVM : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Type type = op.getType();
    ArrayRef<int64_t> shape;
    Type elementType;

    if (auto vectorType = dyn_cast<VectorType>(type)) {
      shape = vectorType.getShape();
      elementType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(type)) {
      shape = npuVectorType.getShape();
      elementType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr) {
      return failure();
    }

    Location loc = op.getLoc();
    TypedAttr typedScalarAttr = denseAttr.getSplatValue<TypedAttr>();
    if (!typedScalarAttr) {
      return failure();
    }
    Value scalarConstant = rewriter.create<arith::ConstantOp>(loc, typedScalarAttr);

    auto memRefType = MemRefType::get(shape, elementType);
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, scalarConstant, resBuf,
                                  rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithNegfToHIVM : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern<arith::NegFOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<arith::NegFOp>::OpAdaptor;

  LogicalResult matchAndRewrite(arith::NegFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    ArrayRef<int64_t> shape;
    Type elemType;
    if (auto vectorType = dyn_cast<VectorType>(resType)) {
      shape = vectorType.getShape();
      elemType = vectorType.getElementType();
    } else if (auto npuVectorType = dyn_cast<npuvector::NPUVectorType>(resType)) {
      shape = npuVectorType.getShape();
      elemType = npuVectorType.getElementType();
    } else {
      return failure();
    }

    if (!isa<FloatType>(elemType)) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();

    auto memRefType = MemRefType::get(shape, elemType);

    Value zeroScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(elemType, 0.0));
    auto zeroBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(zeroBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *zeroBuf);
    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, zeroScalar, *zeroBuf, rewriter.getDenseI64ArrayAttr({}));

    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VSubOp>(loc, TypeRange{}, ValueRange{*zeroBuf, inputMemRef}, ValueRange{*resBuf});

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathExpToHIVM : public OpConversionPattern<math::ExpOp> {
  using OpConversionPattern<math::ExpOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::ExpOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::ExpOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;

    if (!isa<FloatType>(elemType)) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VExpOp>(loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathLogToHIVM : public OpConversionPattern<math::LogOp> {
  using OpConversionPattern<math::LogOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::LogOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::LogOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;

    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VLnOp>(loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathAbsFToHIVM : public OpConversionPattern<math::AbsFOp> {
  using OpConversionPattern<math::AbsFOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::AbsFOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::AbsFOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;

    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VAbsOp>(loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathSqrtToHIVM : public OpConversionPattern<math::SqrtOp> {
  using OpConversionPattern<math::SqrtOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::SqrtOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::SqrtOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;

    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VSqrtOp>(loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathRsqrtToHIVM : public OpConversionPattern<math::RsqrtOp> {
  using OpConversionPattern<math::RsqrtOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::RsqrtOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::RsqrtOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;

    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VRsqrtOp>(loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathTanhToHIVM : public OpConversionPattern<math::TanhOp> {
  using OpConversionPattern<math::TanhOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::TanhOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::TanhOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VTanhOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathSinToHIVM : public OpConversionPattern<math::SinOp> {
  using OpConversionPattern<math::SinOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::SinOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::SinOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }
    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VSinOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathCosToHIVM : public OpConversionPattern<math::CosOp> {
  using OpConversionPattern<math::CosOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::CosOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::CosOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }
    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VCosOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathErfToHIVM : public OpConversionPattern<math::ErfOp> {
  using OpConversionPattern<math::ErfOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::ErfOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::ErfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!elemType.isF16() && !elemType.isF32()) {
      return failure();
    }
    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VErfOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

/// Lowers math unary rounding ops (ceil/floor/round/roundeven/trunc) to same-type
/// f32 hivm.vcast with the corresponding round_mode.
template <typename MathOp, hivm::RoundMode RoundMode>
struct MathUnaryRoundToHIVM : public OpConversionPattern<MathOp> {
  using OpConversionPattern<MathOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<MathOp>::OpAdaptor;

  LogicalResult matchAndRewrite(MathOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();
    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!elemType.isF32()) {
      return failure();
    }
    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(RoundMode);
    rewriter.create<hivm::VCastOp>(loc, TypeRange{}, inputMemRef, *resBuf, roundingAttr, hivm::TypeFnAttr{});
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathAbsIToHIVM : public OpConversionPattern<math::AbsIOp> {
  using OpConversionPattern<math::AbsIOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::AbsIOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::AbsIOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto shapeAndElem = getShapeAndElemType(resType);
    if (!shapeAndElem) {
      return failure();
    }
    ArrayRef<int64_t> shape = shapeAndElem->first;
    Type elemType = shapeAndElem->second;
    if (!isa<IntegerType>(elemType)) {
      return failure();
    }

    Value inputMemRef = adaptor.getOperand();
    auto memRefType = MemRefType::get(shape, elemType);
    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    setOrPropagateBufferSizeMark(rewriter, loc, op.getOperation(), inputMemRef, *resBuf);
    rewriter.create<hivm::VAbsOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct VectorReductionToHIVM : public OpConversionPattern<vector::ReductionOp> {
  using OpConversionPattern<vector::ReductionOp>::OpConversionPattern;

  static LogicalResult getReduceOperation(vector::CombiningKind kind, hivm::ReduceOperation &reduceKind) {
    switch (kind) {
      case vector::CombiningKind::ADD:
        reduceKind = hivm::ReduceOperation::sum;
        return success();
      case vector::CombiningKind::MUL:
        reduceKind = hivm::ReduceOperation::prod;
        return success();
      case vector::CombiningKind::MINUI:
      case vector::CombiningKind::MINSI:
      case vector::CombiningKind::MINNUMF:
      case vector::CombiningKind::MINIMUMF:
        reduceKind = hivm::ReduceOperation::min;
        return success();
      case vector::CombiningKind::MAXUI:
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXNUMF:
      case vector::CombiningKind::MAXIMUMF:
        reduceKind = hivm::ReduceOperation::max;
        return success();
      case vector::CombiningKind::AND:
        reduceKind = hivm::ReduceOperation::andi;
        return success();
      case vector::CombiningKind::OR:
        reduceKind = hivm::ReduceOperation::ori;
        return success();
      case vector::CombiningKind::XOR:
        reduceKind = hivm::ReduceOperation::xori;
        return success();
      default:
        return failure();
    }
  }

  LogicalResult matchAndRewrite(vector::ReductionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value sourceMemRef = adaptor.getVector();
    if (!isa<MemRefType>(sourceMemRef.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref source");
    }

    auto srcMemRefType = cast<MemRefType>(sourceMemRef.getType());
    Type elemType = srcMemRefType.getElementType();
    int64_t rank = srcMemRefType.getRank();

    auto kind = op.getKind();
    hivm::ReduceOperation reduceKind;
    if (failed(getReduceOperation(kind, reduceKind))) {
      return failure();
    }

    SmallVector<int64_t> targetShape(rank, 1);
    auto resultMemRefType = MemRefType::get(targetShape, elemType);
    Value resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemRefType);

    SmallVector<int64_t> reduceDims;
    for (int64_t i = 0; i < rank; ++i) {
      reduceDims.push_back(i);
    }

    auto reduceOpAttr = hivm::ReduceOpAttr::get(op.getContext(), reduceKind);
    rewriter.create<hivm::VReduceOp>(loc, TypeRange{}, sourceMemRef, resultBuf, Value(), reduceOpAttr,
                                     rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    rewriter.replaceOp(op, resultBuf);

    return success();
  }
};

struct VectorBroadcastToHIVM : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    if (isa<VectorType>(source.getType())) {
      return rewriter.notifyMatchFailure(op, "vector broadcast not supported");
    }

    auto vecType = op.getVector().getType();
    auto memRefType = MemRefType::get(vecType.getShape(), vecType.getElementType());

    if (auto srcMemRefType = dyn_cast<MemRefType>(source.getType())) {
      if (srcMemRefType == memRefType) {
        rewriter.replaceOp(op, source);
        return success();
      }
    }

    Value resultBuf = rewriter.create<memref::AllocOp>(loc, memRefType);
    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, source, resultBuf, rewriter.getDenseI64ArrayAttr({}));

    rewriter.replaceOp(op, resultBuf);

    return success();
  }
};

struct ScfForToHIVM : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getResultTypes(),
                      [](Type t) { return isa<VectorType>(t) || isa<npuvector::NPUVectorType>(t); })) {
      return failure();
    }

    SmallVector<Value> newInitArgs(adaptor.getInitArgs().begin(), adaptor.getInitArgs().end());

    auto newForOp = rewriter.create<scf::ForOp>(op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
                                                adaptor.getStep(), newInitArgs);

    Block &oldBlock = op.getRegion().front();
    Block &newBlock = newForOp.getRegion().front();

    SmallVector<Value> newBlockArgs(newBlock.getArguments().begin(), newBlock.getArguments().end());

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockArgs);

    rewriter.replaceOp(op, newForOp.getResults());

    return success();
  }
};

struct ScfYieldToHIVM : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperands(), [](Value v) {
          return isa<VectorType>(v.getType()) || isa<npuvector::NPUVectorType>(v.getType());
        })) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ScfIfToHIVM : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::IfOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0) {
      return failure();
    }
    if (llvm::none_of(op.getResultTypes(),
                      [](Type t) { return isa<VectorType>(t) || isa<npuvector::NPUVectorType>(t); })) {
      return failure();
    }

    SmallVector<Type> newResultTypes;
    for (Type t : op.getResultTypes()) {
      auto shapeAndElem = getShapeAndElemType(t);
      if (shapeAndElem) {
        newResultTypes.push_back(MemRefType::get(shapeAndElem->first, shapeAndElem->second));
      } else {
        newResultTypes.push_back(t);
      }
    }

    // Do not use mergeBlocks(op.*Block(), newIfOp.*Block()): merge destroys the
    // source block and leaves the old scf.if with an empty then/else region until
    // replaceOp runs, which can segfault in ConversionPatternRewriter::applyRewrites
    // when the op is finally erased. Match AffineIfToSCFPattern: move regions with
    // inlineRegionBefore, then drop the placeholder block that scf.if builder adds.
    auto newIfOp =
      rewriter.create<scf::IfOp>(op.getLoc(), newResultTypes, adaptor.getCondition(), op.elseBlock() != nullptr);
    rewriter.inlineRegionBefore(op.getThenRegion(), &newIfOp.getThenRegion().back());
    rewriter.eraseBlock(&newIfOp.getThenRegion().back());
    if (op.elseBlock() != nullptr) {
      rewriter.inlineRegionBefore(op.getElseRegion(), &newIfOp.getElseRegion().back());
      rewriter.eraseBlock(&newIfOp.getElseRegion().back());
    }
    rewriter.replaceOp(op, newIfOp.getResults());
    return success();
  }
};

struct VectorTransferReadToHIVM : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    if (!isa<MemRefType>(source.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref source");
    }

    auto vecType = op.getVectorType();

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(op.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto memRefType = cast<MemRefType>(source.getType());
    int64_t memRefRank = memRefType.getRank();
    int64_t vecRank = vecType.getRank();

    // Handle leading dimensions for rank reduction.
    for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Set sizes to match vector shape.
    for (auto dim : vecType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Infer result type for the rank-reduced subview.
    auto resultType =
      memref::SubViewOp::inferRankReducedResultType(vecType.getShape(), memRefType, offsets, sizes, strides);

    Value finalSource =
      rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(resultType), source, offsets, sizes, strides);

    Type elemType = vecType.getElementType();
    auto targetMemRefType = MemRefType::get(vecType.getShape(), elemType);
    Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType);

    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, finalSource, tempBuf);

    rewriter.replaceOp(op, tempBuf);
    return success();
  }
};

struct VectorTransferWriteToHIVM : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value dataToWrite = adaptor.getVector();
    Value dest = adaptor.getSource();

    if (!isa<MemRefType>(dest.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref destination");
    }
    if (!isa<MemRefType>(dataToWrite.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref data source");
    }

    auto vecType = dyn_cast<VectorType>(op.getVector().getType());
    int64_t vecRank = vecType ? vecType.getRank() : 0;
    ArrayRef<int64_t> vecShape = vecType ? vecType.getShape() : ArrayRef<int64_t>{};

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(op.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto memRefType = cast<MemRefType>(dest.getType());
    int64_t memRefRank = memRefType.getRank();

    // Handle leading dimensions for rank reduction.
    for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Set sizes to match vector shape.
    for (auto dim : vecShape) {
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Infer result type for the rank-reduced subview.
    auto resultType = memref::SubViewOp::inferRankReducedResultType(vecShape, memRefType, offsets, sizes, strides);

    Value slicedDest =
      rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);

    Value finalData = dataToWrite;
    Value finalDest = slicedDest;

    auto destMemRefType = cast<MemRefType>(slicedDest.getType());
    auto dataMemRefType = cast<MemRefType>(dataToWrite.getType());

    // Align ranks between source data and destination slice.
    if (dataMemRefType.getRank() != destMemRefType.getRank()) {
      // Collapse source data to scalar if it has higher rank (e.g., memref<1> to memref<>).
      if (dataMemRefType.getRank() > destMemRefType.getRank()) {
        SmallVector<int64_t> scalarShape;
        auto targetType = MemRefType::get(scalarShape, dataMemRefType.getElementType());
        SmallVector<ReassociationIndices> reassociation;
        finalData = rewriter.create<memref::CollapseShapeOp>(loc, targetType, finalData, reassociation);
      }
    }

    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, finalData, finalDest);

    rewriter.eraseOp(op);

    return success();
  }
};

struct MemRefStoreToHIVM : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(adaptor.getValue().getType())) {
      return success();
    }

    Value valToStore = adaptor.getValue();
    Value memref = adaptor.getMemref();

    auto valType = dyn_cast<MemRefType>(valToStore.getType());
    auto memRefType = dyn_cast<MemRefType>(memref.getType());

    if (!valType || !memRefType) {
      return failure();
    }

    if (valType.getRank() == 1 && valType.getDimSize(0) == 1 && memRefType.getRank() == 0) {
      auto newType = MemRefType::get({1}, memRefType.getElementType());
      OpFoldResult offset = rewriter.getIndexAttr(0);
      SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
      SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
      Value casted = rewriter.create<memref::ReinterpretCastOp>(op.getLoc(), newType, memref, offset, sizes, strides);
      rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange{}, valToStore, casted);
      rewriter.eraseOp(op);
      return success();
    }

    if (valType) {
      rewriter.create<hivm::StoreOp>(op.getLoc(), TypeRange{}, valToStore, memref);
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

static LogicalResult rewritePartialMemRefReductionCollapse(npuvector::ReductionOp op,
                                                           ConversionPatternRewriter &rewriter, Location loc,
                                                           MemRefType srcMemRefType, Type elemType, Value resultBuf,
                                                           const llvm::DenseSet<int64_t> &reduceDimSet, int64_t rank) {
  SmallVector<int64_t> collapsedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!reduceDimSet.contains(i)) {
      collapsedShape.push_back(srcMemRefType.getDimSize(i));
    }
  }
  auto collapsedType = MemRefType::get(collapsedShape, elemType);

  SmallVector<ReassociationIndices> reassoc;
  for (int64_t i = 0; i < rank; ++i) {
    if (!reduceDimSet.contains(i)) {
      reassoc.push_back({i});
    } else if (!reassoc.empty()) {
      reassoc.back().push_back(i);
    } else {
      reassoc.push_back({i});
    }
  }
  if (reassoc.size() > static_cast<size_t>(collapsedShape.size())) {
    if (reassoc.size() == static_cast<size_t>(collapsedShape.size()) + 1 && reassoc.front().size() == 1 &&
        reduceDimSet.contains(reassoc.front().front())) {
      reassoc[1].insert(reassoc[1].begin(), reassoc[0].begin(), reassoc[0].end());
      reassoc.erase(reassoc.begin());
    }
  }
  Value collapsed = rewriter.create<memref::CollapseShapeOp>(loc, collapsedType, resultBuf, reassoc);
  rewriter.replaceOp(op, collapsed);
  return success();
}

struct NPUVectorReductionToHIVM : public OpConversionPattern<npuvector::ReductionOp> {
  using OpConversionPattern<npuvector::ReductionOp>::OpConversionPattern;

  static LogicalResult getReduceOperation(vector::CombiningKind kind, hivm::ReduceOperation &reduceKind) {
    switch (kind) {
      case vector::CombiningKind::ADD:
        reduceKind = hivm::ReduceOperation::sum;
        return success();
      case vector::CombiningKind::MUL:
        reduceKind = hivm::ReduceOperation::prod;
        return success();
      case vector::CombiningKind::MINUI:
      case vector::CombiningKind::MINSI:
      case vector::CombiningKind::MINNUMF:
      case vector::CombiningKind::MINIMUMF:
        reduceKind = hivm::ReduceOperation::min;
        return success();
      case vector::CombiningKind::MAXUI:
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXNUMF:
      case vector::CombiningKind::MAXIMUMF:
        reduceKind = hivm::ReduceOperation::max;
        return success();
      case vector::CombiningKind::AND:
        reduceKind = hivm::ReduceOperation::andi;
        return success();
      case vector::CombiningKind::OR:
        reduceKind = hivm::ReduceOperation::ori;
        return success();
      case vector::CombiningKind::XOR:
        reduceKind = hivm::ReduceOperation::xori;
        return success();
      default:
        return failure();
    }
  }

  LogicalResult matchAndRewrite(npuvector::ReductionOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value sourceMemRef = adaptor.getVector();
    if (!isa<MemRefType>(sourceMemRef.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref source");
    }

    auto srcMemRefType = cast<MemRefType>(sourceMemRef.getType());
    Type elemType = srcMemRefType.getElementType();
    int64_t rank = srcMemRefType.getRank();

    auto kind = op.getKind();
    hivm::ReduceOperation reduceKind;
    if (failed(getReduceOperation(kind, reduceKind))) {
      return failure();
    }

    SmallVector<int64_t> reduceDims;
    if (auto dimsAttr = op.getReductionDims(); dimsAttr && !dimsAttr->empty()) {
      reduceDims.assign(dimsAttr->begin(), dimsAttr->end());
    } else {
      for (int64_t i = 0; i < rank; ++i) {
        reduceDims.push_back(i);
      }
    }

    llvm::DenseSet<int64_t> reduceDimSet(reduceDims.begin(), reduceDims.end());
    SmallVector<int64_t> targetShape;
    for (int64_t i = 0; i < rank; ++i) {
      targetShape.push_back(reduceDimSet.contains(i) ? 1 : srcMemRefType.getDimSize(i));
    }
    auto resultMemRefType = MemRefType::get(targetShape, elemType);
    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (resultMemRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(sourceMemRef, i);
        if (failed(dimVal)) {
          return rewriter.notifyMatchFailure(op, "cannot resolve dynamic dim for reduction result alloc");
        }
        dynSizes.push_back(*dimVal);
      }
    }
    Value resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemRefType, dynSizes);

    auto reduceOpAttr = hivm::ReduceOpAttr::get(op.getContext(), reduceKind);
    rewriter.create<hivm::VReduceOp>(loc, TypeRange{}, sourceMemRef, resultBuf, Value(), reduceOpAttr,
                                     rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    Type resultType = op.getResult().getType();
    bool isPartial = static_cast<int64_t>(reduceDims.size()) < rank;

    if (isPartial) {
      return rewritePartialMemRefReductionCollapse(op, rewriter, loc, srcMemRefType, elemType, resultBuf, reduceDimSet,
                                                   rank);
    }

    if (isa<MemRefType>(resultType)) {
      rewriter.replaceOp(op, resultBuf);
      return success();
    }

    SmallVector<Value> indices;
    indices.reserve(rank);
    for (int64_t i = 0; i < rank; ++i) {
      indices.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }
    Value scalar = rewriter.create<memref::LoadOp>(loc, resultBuf, indices);
    rewriter.replaceOp(op, scalar);
    return success();
  }
};

static bool needsRank1StrideAlignedTransferReadBuffer(Value source, npuvector::NPUVectorType npuVecType) {
  if (npuVecType.getRank() != 1 || !npuVecType.hasDynamicShape()) {
    return false;
  }

  auto sourceTy = dyn_cast<MemRefType>(source.getType());
  if (!sourceTy || sourceTy.getRank() != 1) {
    return false;
  }

  SmallVector<int64_t> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(sourceTy, strides, offset)) || strides.size() != 1) {
    return false;
  }

  return strides.front() > 1;
}

static int64_t computeRank1StrideAlignedTransferReadBytes(ArrayRef<int64_t> maxShape,
                                                          npuvector::NPUVectorType npuVecType, Type elemType) {
  if (maxShape.size() != 1 || npuVecType.getRank() != 1) {
    return 0;
  }

  int64_t elementBits = getElementBitWidth(elemType);
  if (elementBits <= 0) {
    return 0;
  }

  SmallVector<char, 1> staticDims;
  staticDims.push_back(static_cast<char>(!ShapedType::isDynamic(npuVecType.getShape().front())));
  SmallVector<int32_t, 1> alignDims{0};
  return computeBishengStrideAlignedStorageBytesWithTrailingUnit(maxShape, staticDims, alignDims, elementBits);
}

static LogicalResult setTransferReadBufferSizeMarkIfNeeded(npuvector::TransferReadOp op, Value source, Value buf,
                                                           Type elemType, npuvector::NPUVectorType npuVecType,
                                                           ConversionPatternRewriter &rewriter) {
  if (!npuVecType.hasDynamicShape()) {
    return success();
  }

  auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
  if (failed(folded)) {
    return rewriter.notifyMatchFailure(op, "maxSizes for npuvector mark must be one constant index per result rank");
  }

  // BiShengIR later materializes rank-1 non-contiguous loads as padded 2D UB storage.
  if (needsRank1StrideAlignedTransferReadBuffer(source, npuVecType)) {
    int64_t strideAlignedBytes = computeRank1StrideAlignedTransferReadBytes(*folded, npuVecType, elemType);
    if (strideAlignedBytes > 0) {
      setBufferSizeMark(rewriter, op.getLoc(), buf, strideAlignedBytes);
      return success();
    }
  }

  setNPUVectorBufferSizeMark(rewriter, op.getLoc(), npuVecType, elemType, *folded, buf);
  return success();
}

static LogicalResult buildTransferReadSizesAndStrides(int64_t memRefRank, int64_t vecRank,
                                                      npuvector::NPUVectorType npuVecType, ValueRange dynamicSizes,
                                                      ConversionPatternRewriter &rewriter,
                                                      SmallVectorImpl<OpFoldResult> &sizes,
                                                      SmallVectorImpl<OpFoldResult> &strides) {
  for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
    sizes.push_back(rewriter.getIndexAttr(1));
    strides.push_back(rewriter.getIndexAttr(1));
  }

  const bool perAxis = static_cast<int64_t>(dynamicSizes.size()) == vecRank;
  size_t compressedIdx = 0;
  for (int64_t i = 0; i < vecRank; ++i) {
    if (npuVecType.isDynamicDim(i)) {
      if (perAxis) {
        sizes.push_back(dynamicSizes[static_cast<unsigned>(i)]);
      } else {
        if (compressedIdx >= dynamicSizes.size()) {
          return failure();
        }
        sizes.push_back(dynamicSizes[compressedIdx++]);
      }
    } else {
      sizes.push_back(rewriter.getIndexAttr(npuVecType.getDimSize(i)));
    }
    strides.push_back(rewriter.getIndexAttr(1));
  }

  if (!perAxis && compressedIdx != dynamicSizes.size()) {
    return failure();
  }

  return success();
}

static Value traceMemRefToRoot(Value v, int maxSteps = 32) {
  Value current = v;
  for (int i = 0; i < maxSteps; ++i) {
    Operation *def = current.getDefiningOp();
    if (def == nullptr) {
      break;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      current = subview.getSource();
    } else if (auto cast = dyn_cast<memref::CastOp>(def)) {
      current = cast.getSource();
    } else if (auto reinterp = dyn_cast<memref::ReinterpretCastOp>(def)) {
      current = reinterp.getSource();
    } else if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
      current = reshape.getSource();
    } else if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      current = expand.getSrc();
    } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      current = collapse.getSrc();
    } else if (auto bitcast = dyn_cast<hivm::BitcastOp>(def)) {
      current = bitcast.getSrc();
    } else {
      break;
    }
  }
  return current;
}

static bool isRootFromAlloc(Value root) {
  Operation *def = root.getDefiningOp();
  return (def != nullptr) && (isa<memref::AllocOp>(def) || isa<memref::AllocaOp>(def));
}

static bool setBufferSizeMarkAtLeast(PatternRewriter &rewriter, Location loc, Value buffer, int64_t bytes) {
  if (bytes <= 0 || bytes == LLONG_MAX || !isa<MemRefType>(buffer.getType())) {
    return false;
  }
  Value root = traceMemRefToRoot(buffer);
  if (!isRootFromAlloc(root)) {
    return false;
  }

  bool found = false;
  for (Operation *user : root.getUsers()) {
    auto markOp = dyn_cast<annotation::MarkOp>(user);
    if (!markOp) {
      continue;
    }
    auto attr = markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr);
    if (!attr) {
      continue;
    }
    found = true;
    if (attr.getInt() < bytes) {
      markOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(bytes));
    }
  }
  if (!found) {
    return setBufferSizeMark(rewriter, loc, root, bytes);
  }
  return true;
}

static bool hasSameMemRefShapeAndElementType(Value lhs, Value rhs) {
  auto lhsTy = dyn_cast<MemRefType>(lhs.getType());
  auto rhsTy = dyn_cast<MemRefType>(rhs.getType());
  return lhsTy && rhsTy && lhsTy.getElementType() == rhsTy.getElementType() && lhsTy.getShape() == rhsTy.getShape();
}

static bool isBeforeAnchorInSameBlock(Operation *op, Operation *anchor) {
  return (anchor != nullptr) && op->getBlock() == anchor->getBlock() && op->isBeforeInBlock(anchor);
}

static bool isDpsInitRoot(DestinationStyleOpInterface op, Value root) {
  return llvm::any_of(op.getDpsInits(), [&](Value init) { return traceMemRefToRoot(init) == root; });
}

static void markInplaceProducerChainBufferSizeAtLeast(PatternRewriter &rewriter, Location loc, Value buffer,
                                                      Operation *anchor, int64_t bytes,
                                                      llvm::SmallPtrSetImpl<Operation *> &visited) {
  if (!setBufferSizeMarkAtLeast(rewriter, loc, buffer, bytes)) {
    return;
  }

  Value root = traceMemRefToRoot(buffer);
  for (Operation *user : llvm::make_early_inc_range(root.getUsers())) {
    if (user == anchor || !isBeforeAnchorInSameBlock(user, anchor) || !isa<hivm::HIVMStructuredOp>(user)) {
      continue;
    }
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user);
    if (!dpsOp || !isDpsInitRoot(dpsOp, root) || !visited.insert(user).second) {
      continue;
    }

    for (Value input : dpsOp.getDpsInputs()) {
      if (traceMemRefToRoot(input) == root || !hasSameMemRefShapeAndElementType(input, buffer)) {
        continue;
      }
      markInplaceProducerChainBufferSizeAtLeast(rewriter, loc, input, user, bytes, visited);
    }
  }
}

static void markInplaceProducerChainBufferSizeAtLeast(PatternRewriter &rewriter, Location loc, Value buffer,
                                                      Operation *anchor, int64_t bytes) {
  llvm::SmallPtrSet<Operation *, 8> visited;
  markInplaceProducerChainBufferSizeAtLeast(rewriter, loc, buffer, anchor, bytes, visited);
}

static FailureOr<SmallVector<int64_t>> inferInlineBroadcastOperandMaxShape(Value operand, MemRefType resultTy,
                                                                           ArrayRef<int64_t> resultMaxShape) {
  auto operandTy = dyn_cast<MemRefType>(operand.getType());
  if (!operandTy || operandTy.getRank() != resultTy.getRank() ||
      static_cast<int64_t>(resultMaxShape.size()) != resultTy.getRank()) {
    return failure();
  }

  bool hasBroadcastDim = false;
  SmallVector<int64_t> operandMaxShape;
  operandMaxShape.reserve(resultMaxShape.size());
  for (int64_t i = 0; i < resultTy.getRank(); ++i) {
    if (operandTy.isDynamicDim(i)) {
      operandMaxShape.push_back(resultMaxShape[static_cast<size_t>(i)]);
      continue;
    }

    int64_t operandDim = operandTy.getDimSize(i);
    int64_t resultMaxDim = std::max<int64_t>(resultMaxShape[static_cast<size_t>(i)], 1);
    if (operandDim == 1 && resultMaxDim > 1) {
      hasBroadcastDim = true;
    }
    operandMaxShape.push_back(std::max<int64_t>(operandDim, 1));
  }
  if (!hasBroadcastDim) {
    return failure();
  }
  return operandMaxShape;
}

static FailureOr<SmallVector<int64_t>> traceExpandShapeMaxShape(memref::ExpandShapeOp expandOp,
                                                                ArrayRef<int64_t> resultMaxShape) {
  auto srcTy = dyn_cast<MemRefType>(expandOp.getSrc().getType());
  auto resultTy = dyn_cast<MemRefType>(expandOp.getResult().getType());
  if (!srcTy || !resultTy || static_cast<int64_t>(resultMaxShape.size()) != resultTy.getRank()) {
    return failure();
  }

  auto reassoc = expandOp.getReassociationIndices();
  if (static_cast<int64_t>(reassoc.size()) != srcTy.getRank()) {
    return failure();
  }

  SmallVector<int64_t> srcMaxShape;
  srcMaxShape.reserve(static_cast<size_t>(srcTy.getRank()));
  for (int64_t srcDim = 0; srcDim < srcTy.getRank(); ++srcDim) {
    if (!srcTy.isDynamicDim(srcDim)) {
      srcMaxShape.push_back(std::max<int64_t>(srcTy.getDimSize(srcDim), 1));
      continue;
    }

    int64_t product = 1;
    for (int64_t resultDim : reassoc[static_cast<size_t>(srcDim)]) {
      if (resultDim < 0 || resultDim >= static_cast<int64_t>(resultMaxShape.size())) {
        return failure();
      }
      product = multiplyAndCap(product, std::max<int64_t>(resultMaxShape[static_cast<size_t>(resultDim)], 1));
    }
    srcMaxShape.push_back(product);
  }
  return srcMaxShape;
}

static FailureOr<SmallVector<int64_t>> traceCollapseShapeMaxShape(memref::CollapseShapeOp collapseOp,
                                                                  ArrayRef<int64_t> resultMaxShape) {
  auto srcTy = dyn_cast<MemRefType>(collapseOp.getSrc().getType());
  auto resultTy = dyn_cast<MemRefType>(collapseOp.getResult().getType());
  if (!srcTy || !resultTy || static_cast<int64_t>(resultMaxShape.size()) != resultTy.getRank()) {
    return failure();
  }

  auto reassoc = collapseOp.getReassociationIndices();
  if (static_cast<int64_t>(reassoc.size()) != resultTy.getRank()) {
    return failure();
  }

  SmallVector<int64_t> srcMaxShape(static_cast<size_t>(srcTy.getRank()), 1);
  for (int64_t resultDim = 0; resultDim < resultTy.getRank(); ++resultDim) {
    int64_t staticProduct = 1;
    int64_t dynamicDim = -1;
    for (int64_t srcDim : reassoc[static_cast<size_t>(resultDim)]) {
      if (srcDim < 0 || srcDim >= srcTy.getRank()) {
        return failure();
      }
      if (srcTy.isDynamicDim(srcDim)) {
        if (dynamicDim >= 0) {
          return failure();
        }
        dynamicDim = srcDim;
        continue;
      }
      int64_t dim = std::max<int64_t>(srcTy.getDimSize(srcDim), 1);
      srcMaxShape[static_cast<size_t>(srcDim)] = dim;
      staticProduct = multiplyAndCap(staticProduct, dim);
    }
    if (dynamicDim >= 0) {
      int64_t resultMax = std::max<int64_t>(resultMaxShape[static_cast<size_t>(resultDim)], 1);
      if (staticProduct > 1 && resultMax % staticProduct != 0) {
        return failure();
      }
      srcMaxShape[static_cast<size_t>(dynamicDim)] =
        staticProduct > 1 ? std::max<int64_t>(resultMax / staticProduct, 1) : resultMax;
    }
  }
  return srcMaxShape;
}

static FailureOr<SmallVector<int64_t>> inferInlineBroadcastRootMaxShape(Value operand,
                                                                        ArrayRef<int64_t> operandMaxShape,
                                                                        Value &root) {
  Value current = operand;
  SmallVector<int64_t> currentMaxShape(operandMaxShape.begin(), operandMaxShape.end());
  for (int depth = 0; depth < 32; ++depth) {
    Operation *def = current.getDefiningOp();
    if (def == nullptr) {
      break;
    }

    if (auto expandOp = dyn_cast<memref::ExpandShapeOp>(def)) {
      auto next = traceExpandShapeMaxShape(expandOp, currentMaxShape);
      if (failed(next)) {
        return failure();
      }
      current = expandOp.getSrc();
      currentMaxShape = *next;
      continue;
    }
    if (auto collapseOp = dyn_cast<memref::CollapseShapeOp>(def)) {
      auto next = traceCollapseShapeMaxShape(collapseOp, currentMaxShape);
      if (failed(next)) {
        return failure();
      }
      current = collapseOp.getSrc();
      currentMaxShape = *next;
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      current = castOp.getSource();
      continue;
    }
    break;
  }

  if (!isRootFromAlloc(current)) {
    return failure();
  }
  root = current;
  auto rootTy = dyn_cast<MemRefType>(root.getType());
  if (!rootTy || rootTy.getRank() != static_cast<int64_t>(currentMaxShape.size())) {
    return failure();
  }
  return currentMaxShape;
}

static void markInlineBroadcastOperandBufferSizeAtLeast(PatternRewriter &rewriter, Location loc, Value operand,
                                                        MemRefType resultTy, ArrayRef<int64_t> resultMaxShape,
                                                        Operation *anchor) {
  auto operandMaxShape = inferInlineBroadcastOperandMaxShape(operand, resultTy, resultMaxShape);
  if (failed(operandMaxShape)) {
    return;
  }

  Value root;
  auto rootMaxShape = inferInlineBroadcastRootMaxShape(operand, *operandMaxShape, root);
  if (failed(rootMaxShape)) {
    return;
  }

  auto rootTy = dyn_cast<MemRefType>(root.getType());
  if (!rootTy) {
    return;
  }
  int64_t bytes =
    computeBishengInlineBroadcastSourceStorageBytes(*rootMaxShape, rootTy.getShape(), rootTy.getElementType());
  markInplaceProducerChainBufferSizeAtLeast(rewriter, loc, root, anchor, bytes);
}

static LogicalResult rewriteRank0MemRefToVectorTransferRead(npuvector::TransferReadOp op,
                                                            npuvector::TransferReadOp::Adaptor adaptor, Value source,
                                                            npuvector::NPUVectorType npuVecType,
                                                            ValueRange dynamicSizes,
                                                            ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();

  Type elemType = npuVecType.getElementType();
  auto targetMemRefType = MemRefType::get(npuVecType.getShape(), elemType);

  Value scalar = rewriter.create<memref::LoadOp>(loc, source, ValueRange{});

  SmallVector<Value> allocOperands;
  if (targetMemRefType.getNumDynamicDims() > 0) {
    if (static_cast<int64_t>(dynamicSizes.size()) == npuVecType.getRank()) {
      for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
        if (targetMemRefType.isDynamicDim(i)) {
          allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
        }
      }
    } else {
      allocOperands.assign(dynamicSizes.begin(), dynamicSizes.end());
    }
  }
  Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

  if (failed(setTransferReadBufferSizeMarkIfNeeded(op, source, tempBuf, elemType, npuVecType, rewriter))) {
    return failure();
  }

  rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, scalar, tempBuf, rewriter.getDenseI64ArrayAttr({}));
  rewriter.replaceOp(op, tempBuf);
  return success();
}

struct NPUVectorTransferReadToHIVM : public OpConversionPattern<npuvector::TransferReadOp> {
  using OpConversionPattern<npuvector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::TransferReadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    if (!isa<MemRefType>(source.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref source");
    }

    Type resultType = op.getResult().getType();
    auto npuVecType = dyn_cast<npuvector::NPUVectorType>(resultType);
    if (!npuVecType) {
      return rewriter.notifyMatchFailure(op, "expected npuvector type");
    }

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(adaptor.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto memRefType = cast<MemRefType>(source.getType());
    int64_t memRefRank = memRefType.getRank();
    int64_t vecRank = npuVecType.getRank();

    auto dynamicSizes = adaptor.getDynamicSizes();

    if (memRefRank == 0 && vecRank > 0) {
      return rewriteRank0MemRefToVectorTransferRead(op, adaptor, source, npuVecType, dynamicSizes, rewriter);
    }

    if (failed(
          buildTransferReadSizesAndStrides(memRefRank, vecRank, npuVecType, dynamicSizes, rewriter, sizes, strides))) {
      return failure();
    }

    Type fullSubViewTy = memref::SubViewOp::inferResultType(memRefType, offsets, sizes, strides);
    if (!fullSubViewTy) {
      return failure();
    }
    auto fullTy = cast<MemRefType>(fullSubViewTy);
    if (fullTy.getRank() < vecRank) {
      return failure();
    }
    ArrayRef<int64_t> fs = fullTy.getShape();
    SmallVector<int64_t> reducedShape(fs.begin() + (fs.size() - static_cast<size_t>(vecRank)), fs.end());

    Type subViewResultType =
      memref::SubViewOp::inferRankReducedResultType(reducedShape, memRefType, offsets, sizes, strides);

    Value finalSource =
      rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(subViewResultType), source, offsets, sizes, strides);

    Value root = traceMemRefToRoot(source);
    if (isRootFromAlloc(root)) {
      rewriter.replaceOp(op, finalSource);
      return success();
    }

    Type elemType = npuVecType.getElementType();
    auto targetMemRefType = MemRefType::get(npuVecType.getShape(), elemType);

    SmallVector<Value> allocOperands;
    if (targetMemRefType.getNumDynamicDims() > 0) {
      if (static_cast<int64_t>(dynamicSizes.size()) == npuVecType.getRank()) {
        for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
          if (targetMemRefType.isDynamicDim(i)) {
            allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
          }
        }
      } else {
        allocOperands.assign(dynamicSizes.begin(), dynamicSizes.end());
      }
    }
    Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

    if (failed(setTransferReadBufferSizeMarkIfNeeded(op, finalSource, tempBuf, elemType, npuVecType, rewriter))) {
      return failure();
    }

    rewriter.create<hivm::LoadOp>(loc, TypeRange{}, finalSource, tempBuf);

    rewriter.replaceOp(op, tempBuf);
    return success();
  }
};

static LogicalResult rewriteNPUVectorTransferWriteRank0(npuvector::TransferWriteOp op,
                                                        npuvector::TransferWriteOp::Adaptor adaptor, Location loc,
                                                        Value dataToWrite, Value dest, MemRefType dataMemRefType,
                                                        MemRefType destMemRefType,
                                                        ConversionPatternRewriter &rewriter) {
  int64_t dataRank = dataMemRefType.getRank();
  if (dataRank == 0) {
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, dataToWrite, dest);
    rewriter.eraseOp(op);
    return success();
  }
  if (dataRank == 1) {
    OpFoldResult offset = rewriter.getIndexAttr(0);
    SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1)};
    SmallVector<OpFoldResult> strides = {rewriter.getIndexAttr(1)};
    auto dataType = MemRefType::get({1}, dataMemRefType.getElementType());
    Value castedData = dataToWrite;
    if (dataMemRefType != dataType) {
      castedData = rewriter.create<memref::ReinterpretCastOp>(loc, dataType, dataToWrite, offset, sizes, strides);
    }
    auto destType = MemRefType::get({1}, destMemRefType.getElementType());
    Value castedDest = rewriter.create<memref::ReinterpretCastOp>(loc, destType, dest, offset, sizes, strides);
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, castedData, castedDest);
    rewriter.eraseOp(op);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "unsupported rank-0 destination");
}

static LogicalResult buildNPUVectorTransferWriteSizesStrides(Value dataToWrite, MemRefType dataMemRefType,
                                                             int64_t memRefRank, int64_t dataRank,
                                                             ConversionPatternRewriter &rewriter,
                                                             SmallVectorImpl<OpFoldResult> &sizes,
                                                             SmallVectorImpl<OpFoldResult> &strides) {
  for (int64_t i = 0; i < memRefRank - dataRank; ++i) {
    sizes.push_back(rewriter.getIndexAttr(1));
    strides.push_back(rewriter.getIndexAttr(1));
  }
  for (int64_t i = 0; i < dataRank; ++i) {
    if (dataMemRefType.isDynamicDim(i)) {
      auto dimVal = getMemRefDimValue(dataToWrite, i);
      if (failed(dimVal)) {
        return failure();
      }
      sizes.push_back(*dimVal);
    } else {
      sizes.push_back(rewriter.getIndexAttr(dataMemRefType.getDimSize(i)));
    }
    strides.push_back(rewriter.getIndexAttr(1));
  }
  return success();
}

static LogicalResult materializeSubviewIndicesBefore(SmallVectorImpl<OpFoldResult> &offsets,
                                                     SmallVectorImpl<OpFoldResult> &sizes,
                                                     SmallVectorImpl<OpFoldResult> &strides, Operation *insertPt,
                                                     ConversionPatternRewriter &rewriter) {
  IRMapping mapping;
  auto materializeValue = [&](auto &&self, Value value) -> FailureOr<Value> {
    if (Value mapped = mapping.lookupOrNull(value)) {
      return mapped;
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != insertPt->getBlock() || defOp->isBeforeInBlock(insertPt)) {
      return value;
    }

    if (defOp->getNumRegions() != 0 || !isMemoryEffectFree(defOp)) {
      return failure();
    }

    for (Value operand : defOp->getOperands()) {
      FailureOr<Value> remapped = self(self, operand);
      if (failed(remapped)) {
        return failure();
      }
      mapping.map(operand, *remapped);
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertPt);
    rewriter.clone(*defOp, mapping);
    return mapping.lookup(value);
  };

  auto materializeRange = [&](SmallVectorImpl<OpFoldResult> &foldResults) -> LogicalResult {
    for (OpFoldResult &foldResult : foldResults) {
      auto value = foldResult.dyn_cast<Value>();
      if (!value) {
        continue;
      }
      FailureOr<Value> remapped = materializeValue(materializeValue, value);
      if (failed(remapped)) {
        return failure();
      }
      foldResult = *remapped;
    }
    return success();
  };

  return failure(failed(materializeRange(offsets)) || failed(materializeRange(sizes)) ||
                 failed(materializeRange(strides)));
}

static void eraseBufferMarks(Value buffer, ConversionPatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(buffer.getUsers())) {
    if (isa<annotation::MarkOp>(user)) {
      rewriter.eraseOp(user);
    }
  }
}

static Value reshapeMemRefToType(Value src, MemRefType dstTy, Location loc, ConversionPatternRewriter &rewriter) {
  auto srcTy = cast<MemRefType>(src.getType());
  if (srcTy == dstTy) {
    return src;
  }
  if (!srcTy.hasStaticShape() || !dstTy.hasStaticShape() || srcTy.getNumElements() != dstTy.getNumElements()) {
    return {};
  }
  std::optional<SmallVector<ReassociationIndices>> reassoc = getReassociationIndicesForReshape(srcTy, dstTy);
  if (!reassoc) {
    return {};
  }
  if (dstTy.getRank() > srcTy.getRank()) {
    return rewriter.create<memref::ExpandShapeOp>(loc, dstTy, src, *reassoc);
  }
  if (dstTy.getRank() < srcTy.getRank()) {
    return rewriter.create<memref::CollapseShapeOp>(loc, dstTy, src, *reassoc);
  }
  return {};
}

// True when `t` describes a packed, row-major, zero-offset slab: its elements are exactly
// the leading getNumElements(t) elements of the underlying buffer, in linear order. Such a
// view can be aliased as a contiguous prefix regardless of rank or extent (multi-dim and
// dynamic-extent contiguous reads included); strided gathers and non-zero offsets cannot.
// Unit-size dims are ignored (their stride is immaterial). Once a dynamic extent is seen the
// outer strides become unverifiable, so any further non-unit dim is rejected conservatively.
static bool isContiguousZeroOffsetSlab(MemRefType t) {
  int64_t offset = 0;
  SmallVector<int64_t> strides;
  if (failed(getStridesAndOffset(t, strides, offset)) || offset != 0) {
    return false;
  }
  ArrayRef<int64_t> shape = t.getShape();
  int64_t expected = 1;
  bool expectedKnown = true;
  for (int64_t i = t.getRank() - 1; i >= 0; --i) {
    int64_t dim = shape[static_cast<size_t>(i)];
    if (dim == 1) {
      continue;
    }
    int64_t stride = strides[static_cast<size_t>(i)];
    if (!expectedKnown || ShapedType::isDynamic(stride) || stride != expected) {
      return false;
    }
    if (ShapedType::isDynamic(dim)) {
      expectedKnown = false;
      continue;
    }
    expected *= dim;
  }
  return true;
}

// Returns true if `dest` (or any of its view aliases) is written by an op other than
// `anchor`. View-like ops are followed so writes through a subview/cast are detected.
// DPS-init operands, sibling transfer_writes and memref.copy targets count as writers;
// func.call and plain reads are treated as readers (matching the full-coverage path).
static bool allocDestHasForeignWriter(Value dest, Operation *anchor) {
  SmallVector<Value> worklist{dest};
  SmallPtrSet<Value, 8> visited;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (!visited.insert(cur).second) {
      continue;
    }
    for (OpOperand &use : cur.getUses()) {
      Operation *owner = use.getOwner();
      if (owner == anchor || isa<annotation::MarkOp>(owner)) {
        continue;
      }
      if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp, memref::ReshapeOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp, hivm::BitcastOp>(owner)) {
        worklist.append(owner->getResults().begin(), owner->getResults().end());
        continue;
      }
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(owner)) {
        if (dps.isDpsInit(&use)) {
          return true;
        }
        continue;
      }
      if (auto tw = dyn_cast<npuvector::TransferWriteOp>(owner)) {
        if (use.get() == tw.getSource()) {
          return true;
        }
        continue;
      }
      if (isa<memref::CopyOp>(owner) && use.getOperandNumber() == 1) {
        return true;
      }
    }
  }
  return false;
}

// Build a value of type `destTy` aliasing the leading getNumElements(destTy) elements of
// `dataRoot` (an alloc). The source's first elements are exactly what the copy stored into
// `dest`, so reads of `dest` observe identical data through this alias; trailing elements
// (only present when the source is larger) map to dest positions the copy never defined, so
// reading them was undefined either way. Returns null only when the source is too small to
// cover `dest` (no contiguous alias possible) or shapes cannot be reconciled.
static Value buildForwardedSourceView(Value dataRoot, MemRefType destTy, Location loc,
                                      ConversionPatternRewriter &rewriter) {
  auto rootTy = cast<MemRefType>(dataRoot.getType());
  if (!rootTy.hasStaticShape() || !destTy.hasStaticShape() || rootTy.getElementType() != destTy.getElementType()) {
    return {};
  }
  int64_t need = destTy.getNumElements();
  int64_t have = rootTy.getNumElements();
  if (have < need) {
    return {};
  }

  Value src = dataRoot;
  // Source larger than dest: collapse to 1D and slice the contiguous prefix of `need` elems.
  // The explicit canonical (offset 0, unit stride) result type keeps a default-layout memref
  // so the following reshape stays valid.
  if (have > need) {
    Value flat = src;
    if (rootTy.getRank() != 1) {
      ReassociationIndices group(static_cast<size_t>(rootTy.getRank()));
      std::iota(group.begin(), group.end(), 0);
      flat = rewriter.create<memref::CollapseShapeOp>(loc, src, ArrayRef<ReassociationIndices>{group});
    }
    auto prefixTy = MemRefType::get({need}, destTy.getElementType());
    src = rewriter.create<memref::SubViewOp>(loc, prefixTy, flat, ArrayRef<OpFoldResult>{rewriter.getIndexAttr(0)},
                                             ArrayRef<OpFoldResult>{rewriter.getIndexAttr(need)},
                                             ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1)});
  }

  // src and dest now share element count; only shape/rank may differ -> reshape (linear order
  // preserved). reshapeMemRefToType returns src unchanged when the types already match.
  return reshapeMemRefToType(src, destTy, loc, rewriter);
}

// "Take the read value, drop the write": when a copy moves data between two allocs (which
// would otherwise lower to an unsupported UB->UB store), redirect every later read of `dest`
// to a dest-typed alias of the source alloc, then delete the copy and the now-dead dest.
// This always handles the read-alloc -> write-alloc pattern; no full-coverage requirement.
// The only guards are mechanical (both roots are allocs, base-aligned offsets, dest not read
// before the write / not deallocated) plus a single-writer check: a `dest` assembled by
// several writers cannot be aliased to one source, so that (genuinely unsupported) case is
// left to the caller to diagnose rather than silently miscompiled.
static LogicalResult forwardAllocViewCopyTransferWrite(npuvector::TransferWriteOp op, Location loc, Value dataToWrite,
                                                       Value dest, ArrayRef<OpFoldResult> offsets,
                                                       ConversionPatternRewriter &rewriter) {
  auto dataSubview = dataToWrite.getDefiningOp<memref::SubViewOp>();
  if (!dataSubview) {
    return failure();
  }

  Value dataRoot = traceMemRefToRoot(dataToWrite);
  auto destTy = cast<MemRefType>(dest.getType());
  Operation *anchor = op.getOperation();
  auto isZero = [](OpFoldResult ofr) { return isConstantIntValue(ofr, 0); };
  // The forwarded alias maps dest[i] -> dataRoot[i] (linear). That is correct for reads of
  // any extent/rank as long as the read is a contiguous prefix of the root (a zero-offset
  // packed slab); a strided gather or offset read would alias the wrong elements. The write
  // into dest is contiguous from base by construction (offsets 0, unit strides).
  if (dataRoot == dest || !isRootFromAlloc(dataRoot) || !isRootFromAlloc(dest) || !llvm::all_of(offsets, isZero) ||
      !isContiguousZeroOffsetSlab(cast<MemRefType>(dataToWrite.getType())) || allocDestHasForeignWriter(dest, anchor)) {
    return failure();
  }

  DominanceInfo domInfo;
  if (llvm::any_of(dest.getUses(), [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user != anchor && !isa<annotation::MarkOp>(user) &&
               (isa<memref::DeallocOp>(user) || !domInfo.properlyDominates(anchor, user));
      })) {
    return failure();
  }

  rewriter.setInsertionPoint(anchor);
  Value forwarded = buildForwardedSourceView(dataRoot, destTy, loc, rewriter);
  if (!forwarded) {
    return failure();
  }

  eraseBufferMarks(dest, rewriter);
  dest.replaceUsesWithIf(forwarded, [&](OpOperand &use) { return use.getOwner() != anchor; });
  rewriter.eraseOp(op);
  if (dataSubview->use_empty()) {
    rewriter.eraseOp(dataSubview);
  }
  if (dest.use_empty()) {
    rewriter.eraseOp(dest.getDefiningOp());
  }
  return success();
}

static bool dominatesInsertPoint(DominanceInfo &domInfo, Value value, Operation *insertPt) {
  return domInfo.dominates(value, insertPt);
}

static Value resolveActualBuffer(Value dataToWrite) {
  auto forOp = dataToWrite.getDefiningOp<scf::ForOp>();
  if (!forOp) {
    return dataToWrite;
  }

  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    if (forOp.getResult(i) == dataToWrite) {
      return forOp.getInitArgs()[i];
    }
  }
  return dataToWrite;
}

static LogicalResult fuseTransferWriteOnProducerSide(npuvector::TransferWriteOp op, Location loc, Value dest,
                                                     Value actualBuf, Operation *allocDef, Operation *insertPt,
                                                     MemRefType resultMemType, ArrayRef<OpFoldResult> offsets,
                                                     ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                                                     ConversionPatternRewriter &rewriter) {
  eraseBufferMarks(actualBuf, rewriter);

  SmallVector<OpFoldResult> subviewOffsets(offsets.begin(), offsets.end());
  SmallVector<OpFoldResult> subviewSizes(sizes.begin(), sizes.end());
  SmallVector<OpFoldResult> subviewStrides(strides.begin(), strides.end());
  if (failed(materializeSubviewIndicesBefore(subviewOffsets, subviewSizes, subviewStrides, insertPt, rewriter))) {
    return rewriter.notifyMatchFailure(op, "failed to materialize subview indices before alloc-root rewrite");
  }

  rewriter.setInsertionPoint(insertPt);
  Value slicedDest =
    rewriter.create<memref::SubViewOp>(loc, resultMemType, dest, subviewOffsets, subviewSizes, subviewStrides);

  rewriter.replaceAllUsesWith(actualBuf, slicedDest);
  rewriter.replaceOp(allocDef, slicedDest);
  rewriter.eraseOp(op);
  return success();
}

// Sink producer to transfer_write when dest does not dominate the producer side.
static LogicalResult fuseTransferWriteBySinkingProducer(npuvector::TransferWriteOp op, Location loc, Value dest,
                                                        Value actualBuf, Operation *allocDef, MemRefType resultMemType,
                                                        ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
                                                        ArrayRef<OpFoldResult> strides, DominanceInfo &domInfo,
                                                        ConversionPatternRewriter &rewriter) {
  Operation *anchor = op.getOperation();
  Block *allocBlock = allocDef->getBlock();

  SmallVector<Operation *> writers;
  for (Operation *user : actualBuf.getUsers()) {
    if (user == anchor) {
      continue;
    }
    auto dpsUser = dyn_cast<DestinationStyleOpInterface>(user);
    if (!dpsUser || user->getBlock() != allocBlock || user->getNumResults() != 0) {
      return rewriter.notifyMatchFailure(op, "actualBuf has a non-relocatable user");
    }
    bool writesActualBuf = false;
    for (OpOperand &use : user->getOpOperands()) {
      if (use.get() != actualBuf) {
        if (!domInfo.dominates(use.get(), anchor)) {
          return rewriter.notifyMatchFailure(op, "writer operand does not dominate transfer_write");
        }
        continue;
      }
      if (dpsUser.isDpsInit(&use)) {
        writesActualBuf = true;
      } else {
        return rewriter.notifyMatchFailure(op, "actualBuf is read by a producer-side user");
      }
    }
    if (!writesActualBuf) {
      return rewriter.notifyMatchFailure(op, "user does not write actualBuf");
    }
    writers.push_back(user);
  }
  if (writers.empty()) {
    return rewriter.notifyMatchFailure(op, "no producer writer to sink");
  }
  llvm::sort(writers, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  eraseBufferMarks(actualBuf, rewriter);

  rewriter.setInsertionPoint(anchor);
  Value slicedDest = rewriter.create<memref::SubViewOp>(loc, resultMemType, dest, offsets, sizes, strides);

  IRMapping mapping;
  mapping.map(actualBuf, slicedDest);
  for (Operation *writer : writers) {
    rewriter.clone(*writer, mapping);
  }

  rewriter.eraseOp(op);
  for (Operation *writer : llvm::reverse(writers)) {
    rewriter.eraseOp(writer);
  }
  rewriter.eraseOp(allocDef);
  return success();
}

static LogicalResult lowerNPUVectorTransferWriteAllocRootOptimized(npuvector::TransferWriteOp op, Location loc,
                                                                   Value dest, Value dataToWrite, Type resultType,
                                                                   ArrayRef<OpFoldResult> offsets,
                                                                   ArrayRef<OpFoldResult> sizes,
                                                                   ArrayRef<OpFoldResult> strides,
                                                                   ConversionPatternRewriter &rewriter) {
  auto resultMemType = cast<MemRefType>(resultType);

  Value actualBuf = resolveActualBuffer(dataToWrite);

  Operation *allocDef = actualBuf.getDefiningOp();
  if ((allocDef == nullptr) || !isa<memref::AllocOp, memref::AllocaOp>(allocDef)) {
    // dataToWrite is not a producer-temp alloc. If it aliases another alloc this would be a
    // UB->UB store, which the hardware does not support and forwarding could not eliminate
    // (e.g. partial copy into an alloc that has a foreign writer). Report instead of falling
    // back to an illegal store.
    if (isRootFromAlloc(traceMemRefToRoot(dataToWrite))) {
      return rewriter.notifyMatchFailure(
        op, "unsupported UB->UB transfer_write into alloc: copy can be neither fused nor forwarded");
    }
    rewriter.setInsertionPoint(op);
    Value slicedDest = rewriter.create<memref::SubViewOp>(loc, resultMemType, dest, offsets, sizes, strides);
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, dataToWrite, slicedDest);
    rewriter.eraseOp(op);
    return success();
  }

  Operation *insertPt = nullptr;
  const Block *block = allocDef->getBlock();
  for (Operation *user : actualBuf.getUsers()) {
    if (user == op.getOperation()) {
      continue;
    }
    if (user->getBlock() == block && ((insertPt == nullptr) || user->isBeforeInBlock(insertPt))) {
      insertPt = user;
    }
  }
  if (insertPt == nullptr) {
    insertPt = allocDef->getNextNode();
  }

  DominanceInfo domInfo;
  auto indexAvailable = [&](ArrayRef<OpFoldResult> foldResults) {
    return llvm::all_of(foldResults, [&](OpFoldResult foldResult) {
      auto value = foldResult.dyn_cast<Value>();
      if (!value) {
        return true;
      }
      if (dominatesInsertPoint(domInfo, value, insertPt)) {
        return true;
      }
      Operation *def = value.getDefiningOp();
      return def && def->getBlock() == insertPt->getBlock();
    });
  };
  const bool destDominates = dominatesInsertPoint(domInfo, dest, insertPt);
  const bool indicesAvailable = indexAvailable(offsets) && indexAvailable(sizes) && indexAvailable(strides);

  if (destDominates && indicesAvailable) {
    return fuseTransferWriteOnProducerSide(op, loc, dest, actualBuf, allocDef, insertPt, resultMemType, offsets, sizes,
                                           strides, rewriter);
  }

  return fuseTransferWriteBySinkingProducer(op, loc, dest, actualBuf, allocDef, resultMemType, offsets, sizes, strides,
                                            domInfo, rewriter);
}

struct NPUVectorTransferWriteToHIVM : public OpConversionPattern<npuvector::TransferWriteOp> {
  using OpConversionPattern<npuvector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::TransferWriteOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value dataToWrite = adaptor.getVector();
    Value dest = adaptor.getSource();

    // No-op when data and dest share the same alloc root (for results: map through init args).
    if (isa<MemRefType>(dataToWrite.getType()) && isa<MemRefType>(dest.getType())) {
      Value dataRoot = dataToWrite;
      if (auto forOp = dataToWrite.getDefiningOp<scf::ForOp>()) {
        for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
          if (forOp.getResult(i) == dataToWrite) {
            dataRoot = traceMemRefToRoot(forOp.getInitArgs()[i]);
            break;
          }
        }
      } else {
        dataRoot = traceMemRefToRoot(dataToWrite);
      }
      Value destRoot = traceMemRefToRoot(dest);
      if (dataRoot == destRoot && isRootFromAlloc(destRoot)) {
        rewriter.eraseOp(op);
        return success();
      }
    }

    if (!isa<MemRefType>(dest.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref destination");
    }
    if (!isa<MemRefType>(dataToWrite.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref data source");
    }

    auto dataMemRefType = cast<MemRefType>(dataToWrite.getType());
    auto destMemRefType = cast<MemRefType>(dest.getType());

    int64_t memRefRank = destMemRefType.getRank();
    int64_t dataRank = dataMemRefType.getRank();

    if (memRefRank == 0) {
      return rewriteNPUVectorTransferWriteRank0(op, adaptor, loc, dataToWrite, dest, dataMemRefType, destMemRefType,
                                                rewriter);
    }

    Value destRoot = traceMemRefToRoot(dest);
    const bool destIsAllocRoot = isRootFromAlloc(destRoot);

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(adaptor.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    if (failed(buildNPUVectorTransferWriteSizesStrides(dataToWrite, dataMemRefType, memRefRank, dataRank, rewriter,
                                                       sizes, strides))) {
      return failure();
    }

    auto resultType =
      memref::SubViewOp::inferRankReducedResultType(dataMemRefType.getShape(), destMemRefType, offsets, sizes, strides);

    if (destIsAllocRoot &&
        succeeded(forwardAllocViewCopyTransferWrite(op, loc, dataToWrite, dest, offsets, rewriter))) {
      return success();
    }

    if (destIsAllocRoot) {
      return lowerNPUVectorTransferWriteAllocRootOptimized(op, loc, dest, dataToWrite, resultType, offsets, sizes,
                                                           strides, rewriter);
    }

    rewriter.setInsertionPoint(op);
    Value slicedDest =
      rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, dataToWrite, slicedDest);
    rewriter.eraseOp(op);
    return success();
  }
};

/// VBrcOp: scalar src requires empty broadcast_dims; memref/tensor src requires
/// non-empty dims indexing **static size-1** axes. Rank must match between src and dst.
static FailureOr<DenseI64ArrayAttr> getVbrcBroadcastDimsForMemRefSource(MemRefType srcTy, PatternRewriter &rewriter) {
  SmallVector<int64_t> dims;
  for (int64_t i = 0; i < srcTy.getRank(); ++i) {
    if (!srcTy.isDynamicDim(i) && srcTy.getDimSize(i) == 1) {
      dims.push_back(i);
    }
  }
  if (dims.empty()) {
    return failure();
  }
  return rewriter.getDenseI64ArrayAttr(dims);
}

/// Insert a static size-1 axis at logical index `insertSingletonDimIndex` in [0, srcRank]
/// via memref.expand_shape (rank +1). Splits input dim k into (1, d_k), or last dim into
/// (d_last, 1) when inserting past the last logical axis.
static FailureOr<Value> insertSingletonAtPosition(PatternRewriter &rewriter, Location loc, Value src, MemRefType srcTy,
                                                  Type elemType, int64_t insertSingletonDimIndex) {
  const int64_t srcRank = srcTy.getRank();
  assert(insertSingletonDimIndex >= 0 && insertSingletonDimIndex <= srcRank);

  auto staticDimOrDynamic = [&](int64_t dimIdx) -> int64_t {
    return srcTy.isDynamicDim(dimIdx) ? ShapedType::kDynamic : srcTy.getDimSize(dimIdx);
  };

  if (insertSingletonDimIndex < srcRank) {
    SmallVector<int64_t> newShape(static_cast<size_t>(srcRank + 1));
    for (int64_t i = 0; i < insertSingletonDimIndex; ++i) {
      newShape[static_cast<size_t>(i)] = staticDimOrDynamic(i);
    }
    newShape[static_cast<size_t>(insertSingletonDimIndex)] = 1;
    newShape[static_cast<size_t>(insertSingletonDimIndex + 1)] = staticDimOrDynamic(insertSingletonDimIndex);
    for (int64_t i = insertSingletonDimIndex + 1; i < srcRank; ++i) {
      newShape[static_cast<size_t>(i + 1)] = staticDimOrDynamic(i);
    }

    auto newMemTy = MemRefType::get(newShape, elemType);
    SmallVector<ReassociationIndices, 8> reassoc;
    reassoc.reserve(static_cast<size_t>(srcRank));
    for (int64_t j = 0; j < insertSingletonDimIndex; ++j) {
      reassoc.push_back(ReassociationIndices{j});
    }
    reassoc.push_back(ReassociationIndices{insertSingletonDimIndex, insertSingletonDimIndex + 1});
    for (int64_t j = insertSingletonDimIndex + 1; j < srcRank; ++j) {
      reassoc.push_back(ReassociationIndices{j + 1});
    }

    return rewriter.create<memref::ExpandShapeOp>(loc, newMemTy, src, reassoc).getResult();
  }

  SmallVector<int64_t> newShape(static_cast<size_t>(srcRank + 1));
  for (int64_t i = 0; i < srcRank - 1; ++i) {
    newShape[static_cast<size_t>(i)] = staticDimOrDynamic(i);
  }
  newShape[static_cast<size_t>(srcRank - 1)] = staticDimOrDynamic(srcRank - 1);
  newShape[static_cast<size_t>(srcRank)] = 1;
  auto newMemTy = MemRefType::get(newShape, elemType);
  SmallVector<ReassociationIndices, 8> reassoc;
  for (int64_t i = 0; i < srcRank - 1; ++i) {
    reassoc.push_back(ReassociationIndices{i});
  }
  reassoc.push_back(ReassociationIndices{srcRank - 1, srcRank});
  return rewriter.create<memref::ExpandShapeOp>(loc, newMemTy, src, reassoc).getResult();
}

/// Expand rank-0 memref to rank `dstRank` with all static-1 dims (VBrc rank match).
static FailureOr<Value> expandZeroRankMemRefToOnes(PatternRewriter &rewriter, Location loc, Value src, Type elemType,
                                                   int64_t dstRank) {
  if (dstRank == 0) {
    return src;
  }
  SmallVector<int64_t> shape(static_cast<size_t>(dstRank), 1);
  auto resTy = MemRefType::get(shape, elemType);
  SmallVector<ReassociationIndices> empty;
  return rewriter.create<memref::ExpandShapeOp>(loc, resTy, src, empty).getResult();
}

/// `m[i]` maps source dim i to destination axis; |m| == srcRank, entries unique in [0, dstRank).
/// Inserts missing destination axes as static 1 (no transpose; vectorization orders m ascending).
static FailureOr<Value> expandMemRefRankWithBroadcastMapping(PatternRewriter &rewriter, Location loc, Value src,
                                                             MemRefType srcTy, Type elemType, ArrayRef<int64_t> m,
                                                             int64_t dstRank) {
  const int64_t srcRank = srcTy.getRank();
  if (static_cast<int64_t>(m.size()) != srcRank) {
    return failure();
  }

  llvm::SmallDenseSet<int64_t> seenDest;
  for (int64_t ax : m) {
    if (ax < 0 || ax >= dstRank) {
      return failure();
    }
    if (!seenDest.insert(ax).second) {
      return failure();
    }
  }

  if (srcRank == 0) {
    return expandZeroRankMemRefToOnes(rewriter, loc, src, elemType, dstRank);
  }

  Value cur = src;
  MemRefType curTy = srcTy;

  SmallVector<int64_t> axisDest;
  axisDest.reserve(static_cast<size_t>(srcRank));
  for (int64_t i = 0; i < srcRank; ++i) {
    axisDest.push_back(m[i]);
  }

  SmallVector<int64_t> missingDestAxes;
  missingDestAxes.reserve(static_cast<size_t>(dstRank - srcRank));
  for (int64_t j = 0; j < dstRank; ++j) {
    if (seenDest.count(j) == 0) {
      missingDestAxes.push_back(j);
    }
  }
  llvm::sort(missingDestAxes);

  for (int64_t missingDestAxis : missingDestAxes) {
    int64_t insertSingletonDimIndex = 0;
    const auto curRank = static_cast<int64_t>(axisDest.size());
    while (insertSingletonDimIndex < curRank &&
           axisDest[static_cast<size_t>(insertSingletonDimIndex)] < missingDestAxis) {
      ++insertSingletonDimIndex;
    }
    FailureOr<Value> ins = insertSingletonAtPosition(rewriter, loc, cur, curTy, elemType, insertSingletonDimIndex);
    if (failed(ins)) {
      return failure();
    }
    cur = *ins;
    curTy = cast<MemRefType>(cur.getType());
    axisDest.insert(axisDest.begin() + insertSingletonDimIndex, missingDestAxis);
  }

  if (curTy.getRank() != dstRank) {
    return failure();
  }
  return cur;
}

static bool isSupportedBroadcastVectorFoldUser(Operation *user, Value broadcastResult) {
  if (!user->hasTrait<OpTrait::Elementwise>() || user->getNumOperands() != 2) {
    return false;
  }

  if (user->getOperand(0) != broadcastResult && user->getOperand(1) != broadcastResult) {
    return false;
  }

  return !isa<arith::MulSIExtendedOp, arith::MulUIExtendedOp>(user);
}

static bool tryFoldScalarBroadcast(npuvector::BroadcastOp op, Value source, ConversionPatternRewriter &rewriter) {
  if (!isScalarType(source.getType())) {
    return false;
  }

  Value broadcastVal = op.getResult();
  const bool hasSinglePointVectorLhsUser = llvm::any_of(broadcastVal.getUsers(), [broadcastVal](Operation *user) {
    if (user->getNumOperands() != 2 || user->getOperand(1) != broadcastVal) {
      return false;
    }
    auto shapeAndElem = getShapeAndElemType(user->getOperand(0).getType());
    return shapeAndElem && llvm::all_of(shapeAndElem->first, [](int64_t dim) { return dim == 1; });
  });
  if (hasSinglePointVectorLhsUser) {
    return false;
  }

  const bool hasNonFoldUser = llvm::any_of(broadcastVal.getUsers(), [broadcastVal](Operation *user) {
    return !isSupportedBroadcastScalarFoldUser(user, broadcastVal) && !isa<annotation::MarkOp>(user);
  });
  if (hasNonFoldUser) {
    return false;
  }

  rewriter.replaceOp(op, source);

  auto constOp = source.getDefiningOp<arith::ConstantOp>();
  if (constOp && constOp->getResult(0).use_empty()) {
    rewriter.eraseOp(constOp);
  }
  return true;
}

static LogicalResult allocBroadcastBuffer(npuvector::BroadcastOp op, Location loc, MemRefType memRefType,
                                          npuvector::NPUVectorType npuVecType, Type elemType, ValueRange dynSizes,
                                          ConversionPatternRewriter &rewriter, Value &outBuf) {
  SmallVector<Value> allocOperands;
  if (memRefType.getNumDynamicDims() > 0 && !dynSizes.empty()) {
    if (static_cast<int64_t>(dynSizes.size()) == npuVecType.getRank()) {
      for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
        if (memRefType.isDynamicDim(i)) {
          allocOperands.push_back(dynSizes[static_cast<unsigned>(i)]);
        }
      }
    } else {
      allocOperands.assign(dynSizes.begin(), dynSizes.end());
    }
  }
  outBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
  if (npuVecType.hasDynamicShape()) {
    auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
    if (failed(folded)) {
      return rewriter.notifyMatchFailure(op, "maxSizes for npuvector mark must be one constant index per result rank");
    }
    setNPUVectorBufferSizeMark(rewriter, loc, npuVecType, elemType, *folded, outBuf);
  }
  return success();
}

static int64_t computeRankExtendedVbrcSourceBytes(npuvector::BroadcastOp op, npuvector::NPUVectorType npuVecType,
                                                  MemRefType expandedTy, Type elemType) {
  SmallVector<int64_t> dstMaxShape;
  if (npuVecType.hasDynamicShape()) {
    auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
    if (failed(folded)) {
      return 0;
    }
    dstMaxShape = *folded;
  } else {
    dstMaxShape.assign(npuVecType.getShape().begin(), npuVecType.getShape().end());
  }
  if (static_cast<int64_t>(dstMaxShape.size()) != expandedTy.getRank()) {
    return 0;
  }

  SmallVector<int64_t> expandedMaxShape;
  expandedMaxShape.reserve(static_cast<size_t>(expandedTy.getRank()));
  for (int64_t i = 0; i < expandedTy.getRank(); ++i) {
    expandedMaxShape.push_back(expandedTy.isDynamicDim(i) ? dstMaxShape[static_cast<size_t>(i)]
                                                          : expandedTy.getDimSize(i));
  }
  return computeBishengStrideAlignedStorageBytes(expandedMaxShape, expandedTy.getShape(), elemType);
}

static LogicalResult prepareMemrefVbrc(npuvector::BroadcastOp op, Value source, MemRefType dstMemTy,
                                       npuvector::NPUVectorType npuVecType, Type elemType, Location loc,
                                       ConversionPatternRewriter &rewriter, Value &outVbrcSrc,
                                       DenseI64ArrayAttr &outBroadcastDims,
                                       int64_t *outRankExtendedSourceBytes = nullptr) {
  if (outRankExtendedSourceBytes != nullptr) {
    *outRankExtendedSourceBytes = 0;
  }
  if (!isa<MemRefType>(source.getType())) {
    outVbrcSrc = source;
    outBroadcastDims = rewriter.getDenseI64ArrayAttr({});
    return success();
  }
  auto srcMemTy = cast<MemRefType>(source.getType());
  int64_t srcRank = srcMemTy.getRank();
  int64_t dstRank = dstMemTy.getRank();

  if (srcRank > dstRank) {
    return rewriter.notifyMatchFailure(op, "npuvector.broadcast: source memref rank exceeds destination");
  }

  outVbrcSrc = source;
  if (srcRank < dstRank) {
    SmallVector<int64_t> mVec;
    DenseI64ArrayAttr axesAttr = op.getDimensionAttr();
    if (!axesAttr.empty()) {
      ArrayRef<int64_t> m = axesAttr.asArrayRef();
      if (static_cast<int64_t>(m.size()) != srcRank) {
        return rewriter.notifyMatchFailure(op, "dimension length must equal source rank");
      }
      mVec.assign(m.begin(), m.end());
    } else {
      mVec.reserve(static_cast<size_t>(srcRank));
      for (int64_t i = 0; i < srcRank; ++i) {
        mVec.push_back(i);
      }
    }

    FailureOr<Value> expanded =
      expandMemRefRankWithBroadcastMapping(rewriter, loc, outVbrcSrc, srcMemTy, elemType, mVec, dstRank);
    if (failed(expanded)) {
      return rewriter.notifyMatchFailure(op,
                                         "npuvector.broadcast: rank extension (expand_shape) failed, "
                                         "check dimension (injective, in range, consistent rank)");
    }
    outVbrcSrc = *expanded;
    auto expandedTy = cast<MemRefType>(outVbrcSrc.getType());
    SmallVector<int64_t> brcDims;
    for (int64_t i = 0; i < dstRank; ++i) {
      if (!expandedTy.isDynamicDim(i) && expandedTy.getDimSize(i) == 1) {
        brcDims.push_back(i);
      }
    }
    if (brcDims.empty()) {
      return rewriter.notifyMatchFailure(op, "npuvector.broadcast: VBrc needs static size-1 axes after rank extension");
    }
    if (outRankExtendedSourceBytes != nullptr) {
      // BiShengIR rank-extends memref-source VBrc by adding static size-1 axes.
      // MarkStrideAlign then pads that hidden axis (e.g. ?xf32 -> ?x8xf32);
      // PlanMemory may inplace-reuse that source with its producer inputs.
      *outRankExtendedSourceBytes = computeRankExtendedVbrcSourceBytes(op, npuVecType, expandedTy, elemType);
    }
    outBroadcastDims = rewriter.getDenseI64ArrayAttr(brcDims);
    return success();
  }

  auto brcDims = getVbrcBroadcastDimsForMemRefSource(srcMemTy, rewriter);
  if (failed(brcDims)) {
    return rewriter.notifyMatchFailure(op,
                                       "npuvector.broadcast: vector vbrc needs static size-1 dims or rank extension");
  }
  outBroadcastDims = *brcDims;
  return success();
}

static bool tryFoldVectorBroadcast(npuvector::BroadcastOp op, Value source, MemRefType dstMemTy,
                                   npuvector::NPUVectorType npuVecType, Type elemType, Location loc,
                                   ConversionPatternRewriter &rewriter) {
  if (!isa<MemRefType>(source.getType())) {
    return false;
  }

  Value broadcastVal = op.getResult();
  const bool hasNonFoldUser = llvm::any_of(broadcastVal.getUsers(), [broadcastVal](Operation *user) {
    return !isSupportedBroadcastVectorFoldUser(user, broadcastVal) && !isa<annotation::MarkOp>(user);
  });
  if (hasNonFoldUser) {
    return false;
  }

  Value vbrcSrc;
  DenseI64ArrayAttr broadcastDimsAttr;
  if (failed(
        prepareMemrefVbrc(op, source, dstMemTy, npuVecType, elemType, loc, rewriter, vbrcSrc, broadcastDimsAttr))) {
    return false;
  }
  if (broadcastDimsAttr.empty()) {
    return false;
  }

  rewriter.replaceOp(op, vbrcSrc);
  return true;
}

struct NPUVectorBroadcastToHIVM : public OpConversionPattern<npuvector::BroadcastOp> {
  using OpConversionPattern<npuvector::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    Type resultType = op.getResult().getType();
    auto npuVecType = dyn_cast<npuvector::NPUVectorType>(resultType);
    if (!npuVecType) {
      return failure();
    }

    if (op.getSource().getType() == resultType) {
      rewriter.replaceOp(op, source);
      return success();
    }

    Type elemType = npuVecType.getElementType();
    if (elemType.isIndex()) {
      elemType = rewriter.getI64Type();
      source = rewriter.create<arith::IndexCastOp>(loc, elemType, source);
    }
    auto memRefType = MemRefType::get(npuVecType.getShape(), elemType);

    if (tryFoldScalarBroadcast(op, source, rewriter)) {
      return success();
    }
    if (tryFoldVectorBroadcast(op, source, memRefType, npuVecType, elemType, loc, rewriter)) {
      return success();
    }

    Value resultBuf;
    if (failed(allocBroadcastBuffer(op, loc, memRefType, npuVecType, elemType, adaptor.getDynamicSizes(), rewriter,
                                    resultBuf))) {
      return failure();
    }

    DenseI64ArrayAttr broadcastDimsAttr;
    Value vbrcSrc;
    int64_t rankExtendedSourceBytes = 0;
    if (failed(prepareMemrefVbrc(op, source, memRefType, npuVecType, elemType, loc, rewriter, vbrcSrc,
                                 broadcastDimsAttr, &rankExtendedSourceBytes))) {
      return failure();
    }

    markInplaceProducerChainBufferSizeAtLeast(rewriter, loc, source, op.getOperation(), rankExtendedSourceBytes);
    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, vbrcSrc, resultBuf, broadcastDimsAttr);

    rewriter.replaceOp(op, resultBuf);
    return success();
  }
};

struct NPUVectorTransposeToHIVM : public OpConversionPattern<npuvector::TransposeOp> {
  using OpConversionPattern<npuvector::TransposeOp>::OpConversionPattern;

  static FailureOr<SmallVector<int64_t>> permuteMaxShape(ArrayRef<int64_t> maxShape, ArrayRef<int64_t> perm) {
    if (maxShape.size() != perm.size()) {
      return failure();
    }
    SmallVector<int64_t> resultMaxShape;
    resultMaxShape.reserve(perm.size());
    for (int64_t dim : perm) {
      if (dim < 0 || static_cast<size_t>(dim) >= maxShape.size()) {
        return failure();
      }
      resultMaxShape.push_back(maxShape[static_cast<size_t>(dim)]);
    }
    return resultMaxShape;
  }

  static void markTransposeSourceBufferSizeAtLeast(ConversionPatternRewriter &rewriter, Location loc,
                                                   npuvector::TransposeOp op, Value src,
                                                   ArrayRef<int64_t> sourceMaxShape) {
    auto sourceNpuType = dyn_cast<npuvector::NPUVectorType>(op.getVector().getType());
    if (!sourceNpuType || sourceMaxShape.empty()) {
      return;
    }
    int64_t bytes = computeBishengStructuredNpuVectorStorageBytes(sourceMaxShape, sourceNpuType.getShape(),
                                                                  sourceNpuType.getElementType());
    int64_t transposeBytes = computeBishengLastDimTransposeBufferBytes(
      sourceMaxShape, sourceNpuType.getShape(), op.getPermutation(), sourceNpuType.getElementType(), false);
    bytes = std::max(bytes, transposeBytes);
    setBufferSizeMarkAtLeast(rewriter, loc, src, bytes);
  }

  static FailureOr<Value> lowerTranspose2Axis(ConversionPatternRewriter &rewriter, Location loc, Value src,
                                              MemRefType srcType, ArrayRef<int64_t> perm, Type elemType,
                                              ArrayRef<int64_t> sourceMaxShape, ArrayRef<int64_t> resultMaxShape) {
    int64_t rank = srcType.getRank();
    SmallVector<int64_t> resultShape(rank);
    for (int64_t i = 0; i < rank; ++i) {
      resultShape[i] = srcType.getDimSize(perm[i]);
    }
    auto resultMemRefType = MemRefType::get(resultShape, elemType);
    SmallVector<Value> allocOperands;
    for (int64_t i = 0; i < rank; ++i) {
      if (resultMemRefType.isDynamicDim(i)) {
        auto dimVal = getMemRefDimValue(src, perm[i]);
        if (failed(dimVal)) {
          return failure();
        }
        allocOperands.push_back(*dimVal);
      }
    }
    Value resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemRefType, allocOperands);
    if (!setTransposeResultBufferSizeMark(rewriter, loc, resultMemRefType, elemType, resultMaxShape, sourceMaxShape,
                                          srcType.getShape(), perm, resultBuf)) {
      propagateBufferSizeMark(rewriter, loc, src, resultBuf);
    }
    rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, src, resultBuf, rewriter.getDenseI64ArrayAttr(perm));
    return resultBuf;
  }

  static FailureOr<Value> lowerTransposeMultiAxis(ConversionPatternRewriter &rewriter, Location loc, Value src,
                                                  MemRefType srcType, ArrayRef<int64_t> swapSeq, Type elemType,
                                                  ArrayRef<int64_t> sourceMaxShape) {
    int64_t rank = srcType.getRank();
    Value currentBuf = src;
    MemRefType currentType = srcType;
    SmallVector<int64_t> currentMaxShape(sourceMaxShape.begin(), sourceMaxShape.end());
    bool hasMaxShape = currentMaxShape.size() == static_cast<size_t>(rank);

    for (int64_t a : swapSeq) {
      SmallVector<int64_t> newShape(rank);
      SmallVector<int64_t> newMaxShape;
      if (hasMaxShape) {
        newMaxShape.resize(rank);
      }
      for (int64_t i = 0; i < rank; ++i) {
        int64_t srcDim = (i == a) ? a + 1 : (i == a + 1) ? a : i;
        newShape[i] = currentType.getDimSize(srcDim);
        if (hasMaxShape) {
          newMaxShape[i] = currentMaxShape[static_cast<size_t>(srcDim)];
        }
      }
      auto newMemRefType = MemRefType::get(newShape, elemType);
      SmallVector<Value> allocOperands;
      for (int64_t i = 0; i < rank; ++i) {
        if (newMemRefType.isDynamicDim(i)) {
          int64_t srcDim = (i == a) ? a + 1 : (i == a + 1) ? a : i;
          auto dimVal = getMemRefDimValue(currentBuf, srcDim);
          if (failed(dimVal)) {
            return failure();
          }
          allocOperands.push_back(*dimVal);
        }
      }
      Value newBuf = rewriter.create<memref::AllocOp>(loc, newMemRefType, allocOperands);
      SmallVector<int64_t> swapPerm = buildAdjacentSwapPerm(rank, a);
      if (hasMaxShape) {
        int64_t sourceBytes =
          computeBishengStructuredNpuVectorStorageBytes(currentMaxShape, currentType.getShape(), elemType);
        int64_t transposeSourceBytes =
          computeBishengLastDimTransposeBufferBytes(currentMaxShape, currentType.getShape(), swapPerm, elemType, false);
        setBufferSizeMarkAtLeast(rewriter, loc, currentBuf, std::max(sourceBytes, transposeSourceBytes));
      }
      if (!hasMaxShape ||
          !setTransposeResultBufferSizeMark(rewriter, loc, newMemRefType, elemType, newMaxShape, currentMaxShape,
                                            currentType.getShape(), swapPerm, newBuf)) {
        propagateBufferSizeMark(rewriter, loc, currentBuf, newBuf);
      }
      rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, currentBuf, newBuf,
                                          rewriter.getDenseI64ArrayAttr(swapPerm));
      currentBuf = newBuf;
      currentType = newMemRefType;
      if (hasMaxShape) {
        currentMaxShape = std::move(newMaxShape);
      }
    }
    return currentBuf;
  }

  LogicalResult matchAndRewrite(npuvector::TransposeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = adaptor.getVector();

    if (!isa<MemRefType>(src.getType())) {
      return rewriter.notifyMatchFailure(op, "expected memref source");
    }

    auto resultType = op.getResult().getType();
    auto npuResultType = dyn_cast<npuvector::NPUVectorType>(resultType);
    if (!npuResultType) {
      return rewriter.notifyMatchFailure(op, "expected npuvector result type");
    }

    ArrayRef<int64_t> perm = op.getPermutation();
    auto srcType = cast<MemRefType>(src.getType());
    int64_t rank = srcType.getRank();

    int transposeAxisNum = 0;
    for (int64_t i = 0; i < rank; ++i) {
      if (perm[i] != i) {
        ++transposeAxisNum;
      }
    }

    Type elemType = npuResultType.getElementType();
    SmallVector<int64_t> resultMaxShape;
    auto sourceMaxShape = inferNPUVectorMaxShape(op.getVector());
    if (succeeded(sourceMaxShape)) {
      auto permutedMaxShape = permuteMaxShape(*sourceMaxShape, perm);
      if (succeeded(permutedMaxShape)) {
        resultMaxShape = std::move(*permutedMaxShape);
      }
    }

    if (transposeAxisNum == 0) {
      rewriter.replaceOp(op, src);
      return success();
    }

    if (transposeAxisNum == 2) {
      ArrayRef<int64_t> sourceMaxShapeRef;
      if (succeeded(sourceMaxShape)) {
        sourceMaxShapeRef = *sourceMaxShape;
        markTransposeSourceBufferSizeAtLeast(rewriter, loc, op, src, *sourceMaxShape);
      }
      auto resultBuf =
        lowerTranspose2Axis(rewriter, loc, src, srcType, perm, elemType, sourceMaxShapeRef, resultMaxShape);
      if (failed(resultBuf)) {
        return failure();
      }
      rewriter.replaceOp(op, *resultBuf);
      return success();
    }

    SmallVector<int64_t> swapSeq = decomposePermToAdjacentSwaps(perm);
    if (swapSeq.empty()) {
      return rewriter.notifyMatchFailure(op, "failed to decompose permutation");
    }

    ArrayRef<int64_t> sourceMaxShapeRef;
    if (succeeded(sourceMaxShape)) {
      sourceMaxShapeRef = *sourceMaxShape;
    }
    auto resultBuf = lowerTransposeMultiAxis(rewriter, loc, src, srcType, swapSeq, elemType, sourceMaxShapeRef);
    if (failed(resultBuf)) {
      return failure();
    }
    rewriter.replaceOp(op, *resultBuf);
    return success();
  }
};

struct NPUVectorIndexCastToHIVM : public OpConversionPattern<npuvector::IndexCastOp> {
  using OpConversionPattern<npuvector::IndexCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::IndexCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    rewriter.replaceOp(op, input);
    return success();
  }
};

}  // namespace

/// Second-phase rewrite: strip scf.for iter_args/results after partial conversion; replace
/// iter_args with inits; empty yield; remap for results to yielded memref.
/// Runs as greedy patterns after applyPartialConversion (not inside conversion pattern set).
struct ScfForStripRedundantCarriedValues : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter &rewriter) const override {
    unsigned n = forOp.getNumRegionIterArgs();
    if (n == 0) {
      return failure();
    }
    if (forOp.getInitArgs().size() != n) {
      return failure();
    }

    Block *oldBody = forOp.getBody();
    auto yieldOp = dyn_cast<scf::YieldOp>(oldBody->getTerminator());
    if (!yieldOp || yieldOp.getNumOperands() != n) {
      return failure();
    }

    SmallVector<Value> resultReplacements;
    resultReplacements.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
      Value yielded = yieldOp.getOperand(i);
      if (!isa<MemRefType>(yielded.getType())) {
        return failure();
      }
      resultReplacements.push_back(yielded);
    }

    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);
    auto newFor = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());

    Block *newBody = newFor.getBody();
    forOp.getInductionVar().replaceAllUsesWith(newFor.getInductionVar());
    for (auto [oldIterArg, initVal] : llvm::zip(forOp.getRegionIterArgs(), forOp.getInitArgs())) {
      oldIterArg.replaceAllUsesWith(initVal);
    }

    if (Operation *term = newBody->getTerminator()) {
      rewriter.eraseOp(term);
    }

    Operation *oldTerminator = oldBody->getTerminator();
    for (Operation &op : llvm::make_early_inc_range(*oldBody)) {
      if (&op == oldTerminator) {
        break;
      }
      op.moveBefore(newBody, newBody->end());
    }

    rewriter.setInsertionPointToEnd(newBody);
    rewriter.create<scf::YieldOp>(loc);

    for (unsigned i = 0; i < n; ++i) {
      forOp.getResult(i).replaceAllUsesWith(resultReplacements[i]);
    }
    rewriter.eraseOp(forOp);
    return success();
  }
};

void hivm::populateArithToHIVMConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<BinaryArithToHIVM<arith::AddFOp, hivm::VAddOp>, BinaryArithToHIVM<arith::AddIOp, hivm::VAddOp>,
               BinaryArithToHIVM<arith::MulFOp, hivm::VMulOp>, BinaryArithToHIVM<arith::MulIOp, hivm::VMulOp>,
               BinaryArithToHIVM<arith::SubFOp, hivm::VSubOp>, BinaryArithToHIVM<arith::SubIOp, hivm::VSubOp>,
               BinaryArithToHIVM<arith::DivFOp, hivm::VDivOp>, BinaryArithToHIVM<arith::DivSIOp, hivm::VDivOp>,
               BinaryArithToHIVM<arith::DivUIOp, hivm::VDivOp>, BinaryArithToHIVM<arith::MaxSIOp, hivm::VMaxOp>,
               BinaryArithToHIVM<arith::MaxUIOp, hivm::VMaxOp>, BinaryArithToHIVM<arith::MinSIOp, hivm::VMinOp>,
               BinaryArithToHIVM<arith::MinUIOp, hivm::VMinOp>>(patterns.getContext());

  patterns.add<VectorTransferReadToHIVM, VectorTransferWriteToHIVM>(patterns.getContext());

  patterns.add<UnaryArithToHIVMCast<arith::ExtFOp>, UnaryArithToHIVMCast<arith::FPToSIOp>,
               UnaryArithToHIVMCast<arith::FPToUIOp>, UnaryArithToHIVMCast<arith::SIToFPOp>,
               UnaryArithToHIVMCast<arith::UIToFPOp>, UnaryArithToHIVMCast<arith::ExtSIOp>,
               UnaryArithToHIVMCast<arith::ExtUIOp>, UnaryArithToHIVMCast<arith::TruncIOp>,
               UnaryArithToHIVMCast<arith::TruncFOp>, ArithCmpToHIVM<arith::CmpFOp>, ArithCmpToHIVM<arith::CmpIOp>,
               ArithMulExtToHIVM<arith::MulSIExtendedOp>, ArithMulExtToHIVM<arith::MulUIExtendedOp>>(
    patterns.getContext());
  patterns.add<
    ElementwiseOpToHIVMBinary<arith::AndIOp, hivm::VAndOp>, ElementwiseOpToHIVMBinary<arith::OrIOp, hivm::VOrOp>,
    ElementwiseOpToHIVMBinary<arith::XOrIOp, hivm::VXorOp>, ElementwiseOpToHIVMBinary<arith::RemFOp, hivm::VModOp>,
    ElementwiseOpToHIVMBinary<arith::RemSIOp, hivm::VModOp>, ElementwiseOpToHIVMBinary<arith::RemUIOp, hivm::VModOp>,
    ElementwiseOpToHIVMBinary<arith::MinNumFOp, hivm::VMinOp>,
    ElementwiseOpToHIVMBinary<arith::MinimumFOp, hivm::VMinOp>,
    ElementwiseOpToHIVMBinary<arith::MaxNumFOp, hivm::VMaxOp>,
    ElementwiseOpToHIVMBinary<arith::MaximumFOp, hivm::VMaxOp>, ElementwiseOpToHIVMBinary<arith::ShLIOp, hivm::VShLOp>,
    ElementwiseOpToHIVMBinary<arith::ShRSIOp, hivm::VShROp>, ElementwiseOpToHIVMBinary<arith::ShRUIOp, hivm::VShROp>>(
    patterns.getContext());
  patterns.add<ArithBitcastToHIVM>(patterns.getContext());
  patterns.add<NPUVectorBitcastToHIVM>(patterns.getContext());
  patterns.add<ArithSelectToHIVM<arith::SelectOp>>(patterns.getContext());
  patterns.add<ArithSelectToHIVM<npuvector::SelectOp>>(patterns.getContext());
  patterns.add<ArithConstantToHIVM>(patterns.getContext());
  patterns.add<ArithNegfToHIVM>(patterns.getContext());
  patterns.add<MathExpToHIVM>(patterns.getContext());
  patterns.add<MathLogToHIVM>(patterns.getContext());
  patterns.add<MathAbsFToHIVM>(patterns.getContext());
  patterns.add<MathSqrtToHIVM>(patterns.getContext());
  patterns.add<MathRsqrtToHIVM>(patterns.getContext());
  patterns.add<MathTanhToHIVM>(patterns.getContext());
  patterns.add<MathSinToHIVM>(patterns.getContext());
  patterns.add<MathCosToHIVM>(patterns.getContext());
  patterns.add<MathErfToHIVM>(patterns.getContext());
  patterns.add<MathUnaryRoundToHIVM<math::CeilOp, hivm::RoundMode::CEIL>>(patterns.getContext());
  patterns.add<MathUnaryRoundToHIVM<math::FloorOp, hivm::RoundMode::FLOOR>>(patterns.getContext());
  patterns.add<MathUnaryRoundToHIVM<math::RoundOp, hivm::RoundMode::ROUND>>(patterns.getContext());
  patterns.add<MathUnaryRoundToHIVM<math::RoundEvenOp, hivm::RoundMode::RINT>>(patterns.getContext());
  patterns.add<MathUnaryRoundToHIVM<math::TruncOp, hivm::RoundMode::TRUNC>>(patterns.getContext());
  patterns.add<MathAbsIToHIVM>(patterns.getContext());
  patterns.add<VectorReductionToHIVM>(patterns.getContext());
  patterns.add<NPUVectorReductionToHIVM>(patterns.getContext());
  patterns.add<NPUVectorTransferReadToHIVM, NPUVectorTransferWriteToHIVM, NPUVectorBroadcastToHIVM,
               NPUVectorTransposeToHIVM, NPUVectorIndexCastToHIVM>(patterns.getContext());
  patterns.add<UnaryNPUVectorToHIVMCast<npuvector::ExtFOp>, UnaryNPUVectorToHIVMCast<npuvector::TruncFOp>,
               UnaryNPUVectorToHIVMCast<npuvector::ExtSIOp>, UnaryNPUVectorToHIVMCast<npuvector::ExtUIOp>,
               UnaryNPUVectorToHIVMCast<npuvector::TruncIOp>, UnaryNPUVectorToHIVMCast<npuvector::SIToFPOp>,
               UnaryNPUVectorToHIVMCast<npuvector::UIToFPOp>, UnaryNPUVectorToHIVMCast<npuvector::FPToSIOp>,
               UnaryNPUVectorToHIVMCast<npuvector::FPToUIOp>>(patterns.getContext());
  patterns.add<NPUVectorCmpToHIVM<npuvector::CmpFOp>, NPUVectorCmpToHIVM<npuvector::CmpIOp>>(patterns.getContext());
  patterns.add<VectorBroadcastToHIVM>(patterns.getContext());
  patterns.add<ScfForToHIVM>(patterns.getContext());
  patterns.add<ScfIfToHIVM>(patterns.getContext());
  patterns.add<ScfYieldToHIVM>(patterns.getContext());
}

namespace {
static bool isVectorOrNPUVectorType(Type type) { return isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type); }

static bool isLegalArithOp(Operation *op) {
  return !std::any_of(op->getResultTypes().begin(), op->getResultTypes().end(), isVectorOrNPUVectorType);
}

static bool isLegalMathOp(Operation *op) {
  return !std::any_of(op->getResultTypes().begin(), op->getResultTypes().end(), isVectorOrNPUVectorType);
}

static bool isLegalSCFForOp(scf::ForOp op) {
  if (std::any_of(op.getResultTypes().begin(), op.getResultTypes().end(), isVectorOrNPUVectorType)) {
    return false;
  }
  for (auto arg : op.getRegion().getArguments()) {
    if (isVectorOrNPUVectorType(arg.getType())) {
      return false;
    }
  }
  return true;
}

static bool isLegalSCFYieldOp(scf::YieldOp op) {
  for (auto operand : op.getOperands()) {
    if (isVectorOrNPUVectorType(operand.getType())) {
      return false;
    }
  }
  return true;
}

struct ArithToHIVMConversionPass : public impl::ConvertArithToHIVMBase<ArithToHIVMConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, tensor::TensorDialect, memref::MemRefDialect, vector::VectorDialect,
                    arith::ArithDialect, math::MathDialect, scf::SCFDialect, annotation::AnnotationDialect,
                    hacc::HACCDialect>();
  }
  void runOnOperation() override;
};

void ArithToHIVMConversionPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!hacc::utils::isDevice(func)) {
    return;
  }

  ConversionTarget target(getContext());
  // HIVM and Tensor are legal
  target.addLegalDialect<hivm::HIVMDialect, tensor::TensorDialect, memref::MemRefDialect, scf::SCFDialect,
                         BuiltinDialect, annotation::AnnotationDialect>();
  target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalArithOp);
  target.addDynamicallyLegalDialect<math::MathDialect>(isLegalMathOp);
  target.addDynamicallyLegalOp<scf::ForOp>(isLegalSCFForOp);
  target.addDynamicallyLegalOp<scf::IfOp>(
    [](scf::IfOp op) { return llvm::none_of(op.getResultTypes(), isVectorOrNPUVectorType); });
  target.addDynamicallyLegalOp<scf::YieldOp>(isLegalSCFYieldOp);
  target.addIllegalOp<vector::ReductionOp, vector::TransferReadOp, vector::TransferWriteOp, vector::BroadcastOp>();
  target.addIllegalOp<npuvector::ReductionOp, npuvector::TransferReadOp, npuvector::TransferWriteOp,
                      npuvector::BroadcastOp, npuvector::TransposeOp, npuvector::ExtFOp, npuvector::TruncFOp,
                      npuvector::ExtSIOp, npuvector::ExtUIOp, npuvector::TruncIOp, npuvector::SIToFPOp,
                      npuvector::UIToFPOp, npuvector::FPToSIOp, npuvector::FPToUIOp, npuvector::BitcastOp,
                      npuvector::CmpIOp, npuvector::CmpFOp, npuvector::SelectOp>();

  RewritePatternSet patterns(&getContext());
  hivm::populateArithToHIVMConversionPatterns(patterns);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  RewritePatternSet stripPatterns(&getContext());
  stripPatterns.add<ScfForStripRedundantCarriedValues>(stripPatterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(stripPatterns)))) {
    signalPassFailure();
    return;
  }
}
}  // namespace

std::unique_ptr<Pass> createArithToHIVMConversionPass() { return std::make_unique<ArithToHIVMConversionPass>(); }

}  // namespace mlir
