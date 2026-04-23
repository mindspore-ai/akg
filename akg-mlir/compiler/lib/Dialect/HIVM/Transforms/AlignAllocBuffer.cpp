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
 * WITHOUT WARRANTIES OR ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * HIVM alloc alignment: baseline 32 bytes; rewrite = larger alloc + subview + UCC.
 *
 * vtranspose `alignBytes` (bytes)
 * | Case | Rule |
 * | A base | Each swapped dim: dynamic, or (extent*elemBytes)%32!=0 -> mark 32 |
 * | B f32, exactly 2 swap dims, not double-aligned | Drop A; src 32/64, dst 64/32 on the two dims |
 * | Else | A only. Double-aligned: some dim has ceil(dimBytes,32)*32 divisible by 64 |
 *
 * vcast (i32|i16 -> i8 only). elemS/elemD element bytes, k=elemS/elemD, Ndst=(32/elemS)*k
 * | Case | Rule |
 * | Src rank-1 short | ceil(extent*k, Ndst)*Ndst |
 * | Src rank-1 long or dynamic | Ndst*Ndst*k |
 * | Src rank>=2 inner | 32 |
 * | Src rank>=2 outer | Ndst*k |
 * | Dst rank-1 | (32/elemD)^2 (1024 for i8) |
 * | Dst rank>=2 | 32 |
 * Emit marks when unaligned or dynamic; rewrite: alignUnit = alignBytes/elemBytes per dim, alignUp.
 */
#include <algorithm>
#include <climits>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "akg/Dialect/HIVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
#define GEN_PASS_DEF_ALIGNALLOCBUFFER
#include "akg/Dialect/HIVM/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "align-alloc-buffer"

namespace mlir {
namespace {

constexpr StringRef kAkgAllocAlignDimsAttr = "akg.alloc_align_dims";
constexpr StringRef kAkgAllocAlignBytesAttr = "akg.alloc_align_bytes";
constexpr StringRef kAnnotationMarkMnemonic = "annotation.mark";
constexpr StringRef kBufferSizeInByteAttrName = "buffer_size_in_byte";
constexpr unsigned kBitsPerByte = 8;
constexpr unsigned kIntrBytesPerBlock = 32;

struct MarkedDim {
  Value buffer;
  int32_t dim;
  int32_t alignBytes;
};

static bool isHivmOpWithName(Operation *op, StringRef nameSuffix) {
  if (!op) return false;
  StringRef name = op->getName().getStringRef();
  if (!name.starts_with("hivm.")) return false;
  return name.ends_with(nameSuffix);
}

static std::optional<memref::AllocOp> traceToRootAlloc(Value value) {
  Value current = value;
  for (unsigned guard = 0; guard < 256; ++guard) {
    Operation *def = current.getDefiningOp();
    if (!def) return std::nullopt;
    if (auto alloc = dyn_cast<memref::AllocOp>(def)) return alloc;
    if (auto sub = dyn_cast<memref::SubViewOp>(def)) {
      current = sub.getSource();
      continue;
    }
    if (auto re = dyn_cast<memref::ReinterpretCastOp>(def)) {
      current = re.getSource();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      current = castOp.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      current = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      current = expand.getSrc();
      continue;
    }
    return std::nullopt;
  }
  return std::nullopt;
}

static void mergeOneMark(DenseMap<int32_t, int32_t> &dimToBytes, int32_t dim, int32_t byteCount) {
  if (byteCount <= 0) return;
  auto it = dimToBytes.find(dim);
  if (it == dimToBytes.end()) {
    dimToBytes[dim] = byteCount;
  } else {
    it->second = static_cast<int32_t>(std::lcm(static_cast<int64_t>(it->second), static_cast<int64_t>(byteCount)));
  }
}

static void setAllocMarkAttrs(memref::AllocOp alloc, const DenseMap<int32_t, int32_t> &dimToBytes) {
  if (dimToBytes.empty()) return;
  SmallVector<int32_t> orderedDims;
  orderedDims.reserve(dimToBytes.size());
  std::transform(dimToBytes.begin(), dimToBytes.end(), std::back_inserter(orderedDims),
                 [](const auto &entry) { return entry.first; });
  llvm::sort(orderedDims);
  SmallVector<int32_t> dimArray;
  SmallVector<int32_t> byteArray;
  for (int32_t dimIndex : orderedDims) {
    dimArray.push_back(dimIndex);
    byteArray.push_back(dimToBytes.lookup(dimIndex));
  }
  OpBuilder builder(alloc.getContext());
  alloc->setAttr(kAkgAllocAlignDimsAttr, builder.getDenseI32ArrayAttr(dimArray));
  alloc->setAttr(kAkgAllocAlignBytesAttr, builder.getDenseI32ArrayAttr(byteArray));
}

static void recordMarkToAllocs(DenseMap<Operation *, DenseMap<int32_t, int32_t>> &perRoot,
                               const std::vector<MarkedDim> &batch) {
  for (const MarkedDim &one : batch) {
    std::optional<memref::AllocOp> root = traceToRootAlloc(one.buffer);
    if (!root) {
      continue;
    }
    mergeOneMark(perRoot[root->getOperation()], one.dim, one.alignBytes);
  }
}

static void buildTransposeLoopDimsFromPerm(ArrayRef<int64_t> perm, SmallVectorImpl<int64_t> &out) {
  for (int64_t axis = 0; axis < static_cast<int64_t>(perm.size()); ++axis) {
    if (perm[axis] != axis) {
      out.push_back(axis);
    }
  }
}

static bool isLastDimTransposeLike(ArrayRef<int64_t> perm) {
  SmallVector<int64_t> loopDims;
  buildTransposeLoopDimsFromPerm(perm, loopDims);
  if (perm.empty()) return false;
  int64_t lastAxis = static_cast<int64_t>(perm.size()) - 1;
  return llvm::any_of(loopDims, [lastAxis](int64_t axis) { return axis == lastAxis; });
}

static uint64_t ceilFactorUint(uint64_t numerator, uint64_t denominator) {
  return (numerator + denominator - 1u) / denominator;
}

/// Transpose last-axis path: hardware line size 32 bytes (UB/L1 block length). Mark a dim when
/// `(extent * elemBytes) % 32 != 0` or the extent is dynamic.
static LogicalResult gatherTransposeMarksCollect32(Operation *transposeOp, ArrayRef<int64_t> transposeLoopDims,
                                                   std::vector<MarkedDim> &outMarks) {
  const uint32_t hwAlignBytes = static_cast<uint32_t>(kIntrBytesPerBlock);
  const unsigned operandBound = std::min(static_cast<unsigned>(2), transposeOp->getNumOperands());
  for (unsigned operandIdx = 0; operandIdx < operandBound; ++operandIdx) {
    Value operand = transposeOp->getOperand(operandIdx);
    auto memTy = dyn_cast<MemRefType>(operand.getType());
    if (!memTy) continue;
    unsigned elemBits = memTy.getElementTypeBitWidth();
    if (elemBits % kBitsPerByte != 0u) continue;
    unsigned elemBytes = elemBits / kBitsPerByte;
    ArrayRef<int64_t> shape = memTy.getShape();
    for (int64_t loopDim : transposeLoopDims) {
      if (loopDim < 0 || loopDim >= static_cast<int64_t>(shape.size())) continue;
      if (ShapedType::isDynamic(shape[loopDim])) {
        outMarks.push_back(MarkedDim{operand, static_cast<int32_t>(loopDim), static_cast<int32_t>(hwAlignBytes)});
        continue;
      }
      uint64_t dimBytes = static_cast<uint64_t>(shape[loopDim]) * static_cast<uint64_t>(elemBytes);
      if (dimBytes % static_cast<uint64_t>(hwAlignBytes) != 0u) {
        outMarks.push_back(MarkedDim{operand, static_cast<int32_t>(loopDim), static_cast<int32_t>(hwAlignBytes)});
      }
    }
  }
  return success();
}

/// B32 f32 transpose: `alignedSrcDimBytes = CEIL_FACTOR(dimBytes, hw)`; if
/// `alignedSrcDimBytes % (hw*2)==0` on any involved dim, the buffer is already double-aligned.
static bool transposeSrcAlreadyDoubleAlignB32(Value srcVal, ArrayRef<int64_t> transposeLoopDims,
                                              uint32_t hwAlignBytes) {
  auto memTy = dyn_cast<MemRefType>(srcVal.getType());
  if (!memTy || transposeLoopDims.empty()) return false;
  Type elemTy = memTy.getElementType();
  if (!elemTy.isF32() || memTy.getElementTypeBitWidth() != 32u) return false;
  unsigned elemBytes = memTy.getElementTypeBitWidth() / kBitsPerByte;
  ArrayRef<int64_t> srcShape = memTy.getShape();
  for (int64_t transDim : transposeLoopDims) {
    if (transDim < 0 || transDim >= static_cast<int64_t>(srcShape.size())) continue;
    if (ShapedType::isDynamic(srcShape[transDim])) return false;
    uint64_t dimBytes = static_cast<uint64_t>(srcShape[transDim]) * static_cast<uint64_t>(elemBytes);
    uint64_t hwU64 = static_cast<uint64_t>(hwAlignBytes);
    uint64_t roundedUp = ceilFactorUint(dimBytes, hwU64) * hwU64;
    uint64_t doubleBlk = static_cast<uint64_t>(hwAlignBytes) * 2u;
    if (roundedUp % doubleBlk == 0u) return true;
  }
  return false;
}

/// B32 f32 transpose when not already double-aligned and exactly two transpose loop dimensions.
static void gatherTransposeMarksF32DoubleAlign(Operation *transposeOp, ArrayRef<int64_t> transposeLoopDims,
                                               std::vector<MarkedDim> &outMarks) {
  Value src = transposeOp->getOperand(0);
  Value dst = transposeOp->getOperand(1);
  int64_t dim0 = transposeLoopDims[0];
  int64_t dim1 = transposeLoopDims[1];
  const uint32_t hw = static_cast<uint32_t>(kIntrBytesPerBlock);
  outMarks.push_back(MarkedDim{src, static_cast<int32_t>(dim0), static_cast<int32_t>(hw)});
  outMarks.push_back(MarkedDim{src, static_cast<int32_t>(dim1), static_cast<int32_t>(hw * 2u)});
  outMarks.push_back(MarkedDim{dst, static_cast<int32_t>(dim0), static_cast<int32_t>(hw * 2u)});
  outMarks.push_back(MarkedDim{dst, static_cast<int32_t>(dim1), static_cast<int32_t>(hw)});
}

static void handleTranspose(Operation *transposeOp, DenseMap<Operation *, DenseMap<int32_t, int32_t>> &perRoot) {
  if (!transposeOp->hasAttr("permutation")) return;
  auto permAttr = transposeOp->getAttrOfType<DenseI64ArrayAttr>("permutation");
  if (!permAttr || permAttr.empty()) return;
  ArrayRef<int64_t> perm = permAttr.asArrayRef();
  if (transposeOp->getAttrOfType<BoolAttr>("disable_align") &&
      transposeOp->getAttrOfType<BoolAttr>("disable_align").getValue()) {
    return;
  }
  if (!isLastDimTransposeLike(perm)) return;

  SmallVector<int64_t> transposeLoopDims;
  buildTransposeLoopDimsFromPerm(perm, transposeLoopDims);
  std::vector<MarkedDim> batch;
  if (failed(gatherTransposeMarksCollect32(transposeOp, transposeLoopDims, batch))) return;

  Value src = transposeOp->getOperand(0);
  auto srcMemTy = dyn_cast<MemRefType>(src.getType());
  const uint32_t hwBytes = static_cast<uint32_t>(kIntrBytesPerBlock);
  const bool isB32Transpose =
    srcMemTy && srcMemTy.getElementType().isF32() && srcMemTy.getElementTypeBitWidth() == 32u;
  if (isB32Transpose && transposeLoopDims.size() == 2u &&
      !transposeSrcAlreadyDoubleAlignB32(src, transposeLoopDims, hwBytes)) {
    batch.clear();
    gatherTransposeMarksF32DoubleAlign(transposeOp, transposeLoopDims, batch);
  }
  recordMarkToAllocs(perRoot, batch);
}

static LogicalResult collectCastAlignDims(int64_t rank, SmallVectorImpl<int64_t> &castAlignDimsOut) {
  if (rank == 1) {
    castAlignDimsOut.push_back(0);
    return success();
  }
  if (rank >= 2) {
    castAlignDimsOut.push_back(rank - 2);
    castAlignDimsOut.push_back(rank - 1);
    return success();
  }
  return failure();
}

static uint32_t vcastRank1HwAlignBytes(ArrayRef<int64_t> shape, int64_t numElemPerBlockForDst, int64_t bytesFactor) {
  if (!ShapedType::isDynamic(shape[0]) && shape[0] <= numElemPerBlockForDst) {
    uint64_t numerator = static_cast<uint64_t>(shape[0]) * static_cast<uint64_t>(bytesFactor);
    uint64_t denom = static_cast<uint64_t>(numElemPerBlockForDst);
    return static_cast<uint32_t>(ceilFactorUint(numerator, denom) * denom);
  }
  uint64_t blockProd = static_cast<uint64_t>(numElemPerBlockForDst) * static_cast<uint64_t>(numElemPerBlockForDst);
  return static_cast<uint32_t>(blockProd * static_cast<uint64_t>(bytesFactor));
}

static void maybePushVcastSrcMarkRank1(Value srcVal, ArrayRef<int64_t> shape, int64_t srcElemBytes,
                                       uint32_t hwAlignBytes, std::vector<MarkedDim> &outMarks) {
  if (ShapedType::isDynamic(shape[0]) || (shape[0] * srcElemBytes) % static_cast<int64_t>(hwAlignBytes) != 0) {
    outMarks.push_back(MarkedDim{srcVal, 0, static_cast<int32_t>(hwAlignBytes)});
  }
}

/// vcast rank >= 2 source: inner dim uses 32-byte hardware alignment; outer dim uses
/// `numElemPerBlockForDst * bytesFactor` bytes.
static void maybePushVcastSrcMarksHighRank(Value srcVal, ArrayRef<int64_t> shape, ArrayRef<int64_t> castAlignDims,
                                           int64_t srcElemBytes, int64_t numElemPerBlockForDst, int64_t bytesFactor,
                                           std::vector<MarkedDim> &outMarks) {
  const uint32_t innerHwBytes = static_cast<uint32_t>(kIntrBytesPerBlock);
  int64_t innerDim = castAlignDims[1];
  if (ShapedType::isDynamic(shape[innerDim]) ||
      (static_cast<uint64_t>(shape[innerDim]) * static_cast<uint64_t>(srcElemBytes)) %
          static_cast<uint64_t>(innerHwBytes) !=
        0u) {
    outMarks.push_back(MarkedDim{srcVal, static_cast<int32_t>(innerDim), static_cast<int32_t>(innerHwBytes)});
  }
  uint32_t outerHwBytes = static_cast<uint32_t>(numElemPerBlockForDst * bytesFactor);
  int64_t outerDim = castAlignDims[0];
  if (ShapedType::isDynamic(shape[outerDim]) ||
      (static_cast<uint64_t>(shape[outerDim]) * static_cast<uint64_t>(srcElemBytes)) %
          static_cast<uint64_t>(outerHwBytes) !=
        0u) {
    outMarks.push_back(MarkedDim{srcVal, static_cast<int32_t>(outerDim), static_cast<int32_t>(outerHwBytes)});
  }
}

/// vcast destination: rank-1 dst uses `(INTR/elem)^2` bytes; higher rank uses 32-byte hardware alignment on
/// selected dims via `(extent * dstElemBytes) % hwAlignBytes`.
static void pushVcastDstMarks(Value dstVal, ShapedType dstShaped, ArrayRef<int64_t> castAlignDims, int64_t dstElemBytes,
                              std::vector<MarkedDim> &outMarks) {
  uint32_t dstHwAlignBytes = static_cast<uint32_t>(kIntrBytesPerBlock);
  if (dstShaped.getRank() == 1) {
    uint64_t numElemPerBlockDst = static_cast<uint64_t>(kIntrBytesPerBlock) / static_cast<uint64_t>(dstElemBytes);
    dstHwAlignBytes = static_cast<uint32_t>(numElemPerBlockDst * numElemPerBlockDst);
  }
  ArrayRef<int64_t> dstShape = dstShaped.getShape();
  for (int64_t checkDim : castAlignDims) {
    if (ShapedType::isDynamic(dstShape[checkDim])) {
      outMarks.push_back(MarkedDim{dstVal, static_cast<int32_t>(checkDim), static_cast<int32_t>(dstHwAlignBytes)});
      continue;
    }
    uint64_t dimBytes = static_cast<uint64_t>(dstShape[checkDim]) * static_cast<uint64_t>(dstElemBytes);
    if (dimBytes % static_cast<uint64_t>(dstHwAlignBytes) != 0u) {
      outMarks.push_back(MarkedDim{dstVal, static_cast<int32_t>(checkDim), static_cast<int32_t>(dstHwAlignBytes)});
    }
  }
}

static LogicalResult gatherCastMarks(Operation *castOp, std::vector<MarkedDim> &outMarks) {
  if (castOp->getNumOperands() < 2) return failure();
  Value srcVal = castOp->getOperand(0);
  Value dstVal = castOp->getOperand(1);
  auto srcType = dyn_cast<ShapedType>(srcVal.getType());
  auto dstType = dyn_cast<ShapedType>(dstVal.getType());
  if (!srcType || !dstType) return failure();
  Type srcElem = getElementTypeOrSelf(srcType);
  Type dstElem = getElementTypeOrSelf(dstType);
  const bool i32ToI8 = srcElem.isInteger(32) && dstElem.isInteger(8);
  const bool i16ToI8 = srcElem.isInteger(16) && dstElem.isInteger(8);
  if (!i32ToI8 && !i16ToI8) return success();

  int64_t srcElemBytes = static_cast<int64_t>(srcType.getElementType().getIntOrFloatBitWidth()) / kBitsPerByte;
  int64_t dstElemBytes = static_cast<int64_t>(dstType.getElementType().getIntOrFloatBitWidth()) / kBitsPerByte;
  int64_t bytesFactor = srcElemBytes / dstElemBytes;

  SmallVector<int64_t> castAlignDims;
  int64_t rank = srcType.getRank();
  if (failed(collectCastAlignDims(rank, castAlignDims))) return failure();

  ArrayRef<int64_t> shape = srcType.getShape();
  int64_t numElemPerBlock = static_cast<int64_t>(kIntrBytesPerBlock) / srcElemBytes;
  int64_t numElemPerBlockForDst = numElemPerBlock * bytesFactor;

  if (rank == 1) {
    uint32_t hwAlignBytes = vcastRank1HwAlignBytes(shape, numElemPerBlockForDst, bytesFactor);
    maybePushVcastSrcMarkRank1(srcVal, shape, srcElemBytes, hwAlignBytes, outMarks);
  } else {
    maybePushVcastSrcMarksHighRank(srcVal, shape, castAlignDims, srcElemBytes, numElemPerBlockForDst, bytesFactor,
                                   outMarks);
  }

  pushVcastDstMarks(dstVal, dstType, castAlignDims, dstElemBytes, outMarks);
  return success();
}

static void handleVCast(Operation *castOp, DenseMap<Operation *, DenseMap<int32_t, int32_t>> &perRoot) {
  std::vector<MarkedDim> batch;
  if (failed(gatherCastMarks(castOp, batch))) return;
  recordMarkToAllocs(perRoot, batch);
}

static OpFoldResult alignUpOfr(OpBuilder &builder, Location loc, OpFoldResult length, uint64_t alignUnit) {
  assert(alignUnit != 0u);
  std::optional<int64_t> lenConst = getConstantIntValue(length);
  if (lenConst.has_value()) {
    uint64_t value = static_cast<uint64_t>(lenConst.value());
    uint64_t aligned = ((value + alignUnit - 1u) / alignUnit) * alignUnit;
    return builder.getIndexAttr(static_cast<int64_t>(aligned));
  }
  if (alignUnit == 1u) return length;
  Value lengthValue = length.get<Value>();
  Value padValue = builder.create<arith::ConstantIndexOp>(loc, alignUnit - 1u);
  Value sumValue = builder.create<arith::AddIOp>(loc, lengthValue, padValue);
  Value unitValue = builder.create<arith::ConstantIndexOp>(loc, alignUnit);
  Value remainder = builder.create<arith::RemSIOp>(loc, sumValue, unitValue);
  return builder.create<arith::SubIOp>(loc, sumValue, remainder).getResult();
}

static Value createAlignedSubview(PatternRewriter &rewriter, memref::AllocOp allocOp,
                                  ArrayRef<OpFoldResult> originalShape, ArrayRef<OpFoldResult> alignedShape) {
  SmallVector<Value> dynSizes;
  SmallVector<int64_t> staticSizes;
  dispatchIndexOpFoldResults(alignedShape, dynSizes, staticSizes);
  MemRefType unalignedTy = cast<MemRefType>(allocOp.getType());
  MemRefType alignedTy = MemRefType::Builder(unalignedTy).setShape(staticSizes);
  rewriter.setInsertionPoint(allocOp);
  auto alignedAlloc = rewriter.create<memref::AllocOp>(allocOp.getLoc(), alignedTy, dynSizes);
  SmallVector<OpFoldResult> offsets(alignedTy.getRank(), rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(alignedTy.getRank(), rewriter.getIndexAttr(1));
  MemRefType subviewTy = cast<MemRefType>(
    memref::SubViewOp::inferRankReducedResultType(unalignedTy.getShape(), alignedTy, offsets, originalShape, strides));
  return rewriter
    .create<memref::SubViewOp>(allocOp.getLoc(), subviewTy, alignedAlloc.getResult(), offsets, originalShape, strides)
    .getResult();
}

using UccList = SmallVector<UnrealizedConversionCastOp, 4>;

static void appendPendingCastsExcluding(UnrealizedConversionCastOp skipCast, const UccList &fromList,
                                        UccList &pendingCasts) {
  std::copy_if(fromList.begin(), fromList.end(), std::back_inserter(pendingCasts),
               [skipCast](UnrealizedConversionCastOp nextCast) { return nextCast != skipCast; });
}

static UnrealizedConversionCastOp propagateSubviewThroughCast(RewriterBase &rewriterInner,
                                                              UnrealizedConversionCastOp conversionOp,
                                                              memref::SubViewOp subviewOp) {
  OpBuilder::InsertionGuard innerGuard(rewriterInner);
  rewriterInner.setInsertionPoint(subviewOp);
  auto newSourceTy = cast<MemRefType>(conversionOp.getOperand(0).getType());
  MemRefType newResultTy = cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
    subviewOp.getType().getShape(), newSourceTy, subviewOp.getMixedOffsets(), subviewOp.getMixedSizes(),
    subviewOp.getMixedStrides()));
  Value newSubview = rewriterInner.create<memref::SubViewOp>(subviewOp.getLoc(), newResultTy,
                                                             conversionOp.getOperand(0), subviewOp.getMixedOffsets(),
                                                             subviewOp.getMixedSizes(), subviewOp.getMixedStrides());
  auto newConversion = rewriterInner.create<UnrealizedConversionCastOp>(
    subviewOp.getLoc(), TypeRange{subviewOp.getType()}, ValueRange{newSubview});
  rewriterInner.replaceOp(subviewOp, newConversion.getResult(0));
  return newConversion;
}

static FailureOr<UccList> propagateExpandThroughCast(RewriterBase &rewriterInner,
                                                     UnrealizedConversionCastOp conversionOp,
                                                     memref::ExpandShapeOp expandOp) {
  OpBuilder::InsertionGuard innerGuard(rewriterInner);
  rewriterInner.setInsertionPoint(expandOp);
  Value innerValue = conversionOp.getOperand(0);
  auto innerTy = cast<MemRefType>(innerValue.getType());
  ArrayRef<int64_t> outShape = expandOp.getResultType().getShape();
  FailureOr<MemRefType> maybeExpandedTy =
    memref::ExpandShapeOp::computeExpandedType(innerTy, outShape, expandOp.getReassociationIndices());
  if (failed(maybeExpandedTy)) return failure();
  Value newExpand = rewriterInner.create<memref::ExpandShapeOp>(expandOp.getLoc(), *maybeExpandedTy, innerValue,
                                                                expandOp.getReassociationIndices());
  auto newConversion = rewriterInner.create<UnrealizedConversionCastOp>(
    expandOp.getLoc(), TypeRange{expandOp.getType()}, ValueRange{newExpand});
  rewriterInner.replaceOp(expandOp, newConversion.getResult(0));
  return UccList{newConversion};
}

static FailureOr<UccList> propagateCollapseThroughCast(RewriterBase &rewriterInner,
                                                       UnrealizedConversionCastOp conversionOp,
                                                       memref::CollapseShapeOp collapseOp) {
  OpBuilder::InsertionGuard innerGuard(rewriterInner);
  rewriterInner.setInsertionPoint(collapseOp);
  auto srcTy = cast<MemRefType>(conversionOp.getOperand(0).getType());
  auto reassociation = collapseOp.getReassociationIndices();
  if (!memref::CollapseShapeOp::isGuaranteedCollapsible(srcTy, reassociation)) return failure();
  MemRefType collapsedTy = memref::CollapseShapeOp::computeCollapsedType(srcTy, reassociation);
  auto newCollapse = rewriterInner.create<memref::CollapseShapeOp>(collapseOp.getLoc(), collapsedTy,
                                                                   conversionOp.getOperand(0), reassociation);
  if (collapsedTy == collapseOp.getResultType()) {
    rewriterInner.replaceOp(collapseOp, newCollapse.getResult());
    return UccList{conversionOp};
  }
  auto newConversion = rewriterInner.create<UnrealizedConversionCastOp>(
    collapseOp.getLoc(), TypeRange{collapseOp.getType()}, ValueRange{newCollapse.getResult()});
  rewriterInner.replaceOp(collapseOp, newConversion.getResult(0));
  return UccList{newConversion};
}

static FailureOr<UccList> propagateMemrefCastThroughCast(RewriterBase &rewriterInner,
                                                         UnrealizedConversionCastOp conversionOp,
                                                         memref::CastOp castOp) {
  Value innerSource = conversionOp.getOperand(0);
  Type destTy = castOp.getDest().getType();
  if (!memref::CastOp::areCastCompatible({innerSource.getType()}, {destTy})) return failure();
  rewriterInner.setInsertionPoint(castOp);
  Value newCast = rewriterInner.create<memref::CastOp>(castOp.getLoc(), destTy, innerSource).getResult();
  rewriterInner.replaceAllUsesWith(castOp.getResult(), newCast);
  rewriterInner.eraseOp(castOp);
  return UccList{conversionOp};
}

static bool opResultsContainMemrefType(Operation *userOp) {
  return llvm::any_of(userOp->getResultTypes(), [](Type resultTy) { return isa<MemRefType>(resultTy); });
}

static bool opRegionsHaveMemrefBlockArgs(Operation *userOp) {
  return llvm::any_of(userOp->getRegions(), [](Region &region) {
    return llvm::any_of(region.getArguments(),
                        [](BlockArgument blockArg) { return isa<MemRefType>(blockArg.getType()); });
  });
}

static FailureOr<UccList> propagateLeafMemrefUseThroughCast(RewriterBase &rewriterInner,
                                                            UnrealizedConversionCastOp conversionOp,
                                                            Operation *userOp) {
  if (opResultsContainMemrefType(userOp)) {
    return failure();
  }
  if (opRegionsHaveMemrefBlockArgs(userOp)) {
    return failure();
  }
  rewriterInner.modifyOpInPlace(
    userOp, [&]() { userOp->replaceUsesOfWith(conversionOp.getResult(0), conversionOp.getOperand(0)); });
  return UccList{conversionOp};
}

static std::optional<UccList> tryFoldRoundTripConversionPair(RewriterBase &rewriterInner,
                                                             UnrealizedConversionCastOp sinkCast,
                                                             UnrealizedConversionCastOp userCast) {
  if (userCast.getOperand(0) != sinkCast.getResult(0)) return std::nullopt;
  if (userCast.getResult(0).getType() != sinkCast.getOperand(0).getType()) return std::nullopt;
  rewriterInner.replaceAllUsesWith(userCast.getResult(0), sinkCast.getOperand(0));
  rewriterInner.eraseOp(userCast);
  return UccList{};
}

static FailureOr<UccList> dispatchUnrealizedCastUser(RewriterBase &rewriterInner,
                                                     UnrealizedConversionCastOp currentCast, Operation *userOp) {
  return llvm::TypeSwitch<Operation *, FailureOr<UccList>>(userOp)
    .Case<memref::SubViewOp>([&](memref::SubViewOp subviewOp) -> FailureOr<UccList> {
      return UccList{propagateSubviewThroughCast(rewriterInner, currentCast, subviewOp)};
    })
    .Case<memref::ExpandShapeOp>([&](memref::ExpandShapeOp expandOp) -> FailureOr<UccList> {
      return propagateExpandThroughCast(rewriterInner, currentCast, expandOp);
    })
    .Case<memref::CollapseShapeOp>([&](memref::CollapseShapeOp collapseOp) -> FailureOr<UccList> {
      return propagateCollapseThroughCast(rewriterInner, currentCast, collapseOp);
    })
    .Case<memref::CastOp>([&](memref::CastOp castOp) -> FailureOr<UccList> {
      return propagateMemrefCastThroughCast(rewriterInner, currentCast, castOp);
    })
    .Case<UnrealizedConversionCastOp>([&](UnrealizedConversionCastOp innerCast) -> FailureOr<UccList> {
      std::optional<UccList> folded = tryFoldRoundTripConversionPair(rewriterInner, currentCast, innerCast);
      if (folded.has_value()) return std::move(folded.value());
      return failure();
    })
    .Case<memref::ReshapeOp>([](memref::ReshapeOp) -> FailureOr<UccList> { return failure(); })
    .Case<memref::ReinterpretCastOp>([](memref::ReinterpretCastOp) -> FailureOr<UccList> { return failure(); })
    .Default([&](Operation *genericOp) -> FailureOr<UccList> {
      return propagateLeafMemrefUseThroughCast(rewriterInner, currentCast, genericOp);
    });
}

/// Replace all uses of `fromValue` with a cast from `toValue`, then push that cast through memref view ops.
/// On failure the greedy driver rolls back the pattern.
static LogicalResult replaceAllocUsesPropagateCast(PatternRewriter &rewriter, Location loc, Value fromValue,
                                                   Value toValue) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfterValue(toValue);
  UccList pendingCasts;
  auto rootCast = rewriter.create<UnrealizedConversionCastOp>(loc, TypeRange{fromValue.getType()}, ValueRange{toValue});
  pendingCasts.push_back(rootCast);
  rewriter.replaceAllUsesWith(fromValue, rootCast.getResult(0));

  for (size_t workIdx = 0; workIdx < pendingCasts.size(); ++workIdx) {
    UnrealizedConversionCastOp currentCast = pendingCasts[workIdx];
    SmallVector<OpOperand *> useOperands =
      llvm::map_to_vector(currentCast->getUses(), [](OpOperand &operand) { return &operand; });
    for (OpOperand *useOperand : useOperands) {
      FailureOr<UccList> stepOut = dispatchUnrealizedCastUser(rewriter, currentCast, useOperand->getOwner());
      if (failed(stepOut)) return failure();
      appendPendingCastsExcluding(currentCast, *stepOut, pendingCasts);
    }
  }

  for (UnrealizedConversionCastOp deadCast : pendingCasts) {
    if (!deadCast->use_empty()) return failure();
    rewriter.eraseOp(deadCast);
  }
  return success();
}

/// Match `annotation.mark` by operation name (generic op registration when Annotation dialect is loaded).
static bool isAnnotationMarkOp(Operation *operation) {
  return operation && operation->getName().getStringRef() == kAnnotationMarkMnemonic;
}

static bool markHasBufferSizeKey(Operation *markOp) {
  if (!isAnnotationMarkOp(markOp)) return false;
  if (markOp->hasAttr(kBufferSizeInByteAttrName)) return true;
  Attribute keysRaw = markOp->getAttr("keys");
  auto keysAttr = dyn_cast<ArrayAttr>(keysRaw);
  if (!keysAttr) return false;
  return llvm::any_of(keysAttr.getValue(), [](Attribute keyEntry) {
    auto strKey = dyn_cast<StringAttr>(keyEntry);
    return strKey && strKey.getValue() == kBufferSizeInByteAttrName;
  });
}

/// Build `annotation.mark` via generic OperationState (type registered when Annotation dialect loads).
static void emitStaticBufferMark(OpBuilder &builder, Location loc, Value memrefVal, int64_t totalBytes) {
  MLIRContext *context = builder.getContext();
  OperationState state(loc, OperationName(kAnnotationMarkMnemonic, context));
  state.addOperands(memrefVal);
  state.addAttribute(kBufferSizeInByteAttrName, builder.getIndexAttr(totalBytes));
  builder.create(state);
}

static std::optional<int64_t> foldLinearAllocBytes(ArrayRef<OpFoldResult> sizes, uint64_t elemBytes) {
  uint64_t prod = elemBytes;
  for (OpFoldResult sizeOfr : sizes) {
    std::optional<int64_t> extent = getConstantIntValue(sizeOfr);
    if (!extent.has_value()) return std::nullopt;
    uint64_t dim = static_cast<uint64_t>(extent.value());
    uint64_t next = prod * dim;
    if (dim != 0u && next / dim != prod) return std::nullopt;
    prod = next;
  }
  if (prod > static_cast<uint64_t>(INT64_MAX)) return std::nullopt;
  return static_cast<int64_t>(prod);
}

/// Upper bound for tile sizes like `arith.minsi %v, %cN` → at most N (when N is constant).
static std::optional<int64_t> inferTileUpperBound(Value dimensionValue, unsigned recursionDepth = 0) {
  constexpr unsigned depthLimit = 24;
  if (recursionDepth >= depthLimit) return std::nullopt;
  Operation *definition = dimensionValue.getDefiningOp();
  if (!definition) return std::nullopt;
  if (auto constantIdx = dyn_cast<arith::ConstantIndexOp>(definition)) return constantIdx.value();
  if (auto minSi = dyn_cast<arith::MinSIOp>(definition)) {
    if (auto bound = minSi.getRhs().getDefiningOp<arith::ConstantIndexOp>()) return bound.value();
    if (auto bound = minSi.getLhs().getDefiningOp<arith::ConstantIndexOp>()) return bound.value();
    auto leftUb = inferTileUpperBound(minSi.getLhs(), recursionDepth + 1);
    auto rightUb = inferTileUpperBound(minSi.getRhs(), recursionDepth + 1);
    if (leftUb.has_value() && rightUb.has_value()) return std::min(leftUb.value(), rightUb.value());
    return std::nullopt;
  }
  return std::nullopt;
}

static uint64_t alignExtentToUnit(uint64_t extent, uint32_t alignUnit) {
  if (alignUnit <= 1u) return extent;
  uint64_t unit = static_cast<uint64_t>(alignUnit);
  return ((extent + unit - 1u) / unit) * unit;
}

/// When SSA extents do not fold to constants, derive static storage bytes from tile upper bounds
/// (e.g. minsi with %c4 / %c16) × per-dim align units — matches max-buffer semantics like 4×16 → 16×16.
static std::optional<int64_t> tryStaticBytesFromTileUpperBounds(ArrayRef<OpFoldResult> originalShape,
                                                                ArrayRef<uint32_t> alignUnitsPerDim,
                                                                unsigned elemBytes) {
  if (originalShape.size() != alignUnitsPerDim.size()) return std::nullopt;
  if (elemBytes == 0u) return std::nullopt;
  uint64_t runningElems = 1;
  for (size_t idx = 0; idx < originalShape.size(); ++idx) {
    std::optional<int64_t> upperBound;
    if (std::optional<int64_t> constantDim = getConstantIntValue(originalShape[idx]))
      upperBound = constantDim;
    else if (Value dimensionValue = originalShape[idx].dyn_cast<Value>())
      upperBound = inferTileUpperBound(dimensionValue);
    if (!upperBound.has_value() || upperBound.value() < 0) return std::nullopt;
    uint64_t alignedExtent = alignExtentToUnit(static_cast<uint64_t>(upperBound.value()), alignUnitsPerDim[idx]);
    uint64_t nextElems = runningElems * alignedExtent;
    if (alignedExtent != 0u && nextElems / alignedExtent != runningElems) return std::nullopt;
    runningElems = nextElems;
  }
  uint64_t bytes = runningElems * static_cast<uint64_t>(elemBytes);
  if (bytes > static_cast<uint64_t>(INT64_MAX)) return std::nullopt;
  return static_cast<int64_t>(bytes);
}

static std::optional<int64_t> resolveAlignedStorageBytesStatic(ArrayRef<OpFoldResult> alignedSizes,
                                                               ArrayRef<OpFoldResult> originalAllocShape,
                                                               ArrayRef<uint32_t> alignUnitsPerDim,
                                                               unsigned elemBytes) {
  if (std::optional<int64_t> linear = foldLinearAllocBytes(alignedSizes, elemBytes)) return linear;
  return tryStaticBytesFromTileUpperBounds(originalAllocShape, alignUnitsPerDim, elemBytes);
}

/// After expanding root alloc, refresh `buffer_size_in_byte` on the logical view (Subview result).
/// Only emits the same static form `{ buffer_size_in_byte = N : index }` as upstream; never `keys`/`values`.
static void refreshBufferSizeMarks(PatternRewriter &rewriter, Location loc, Value markedMemref,
                                   ArrayRef<OpFoldResult> alignedSizes, ArrayRef<OpFoldResult> originalAllocShape,
                                   ArrayRef<uint32_t> alignUnitsPerDim, unsigned elemBytes) {
  SmallVector<Operation *, 4> marks;
  auto markUsers = markedMemref.getUsers();
  std::copy_if(markUsers.begin(), markUsers.end(), std::back_inserter(marks),
               [](Operation *user) { return markHasBufferSizeKey(user); });
  if (marks.empty()) return;

  std::optional<int64_t> newBytes =
    resolveAlignedStorageBytesStatic(alignedSizes, originalAllocShape, alignUnitsPerDim, elemBytes);
  if (!newBytes.has_value()) return;

  for (Operation *markOp : marks) rewriter.eraseOp(markOp);

  rewriter.setInsertionPointAfterValue(markedMemref);
  emitStaticBufferMark(rewriter, loc, markedMemref, newBytes.value());
}

struct AlignMarkedAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp, PatternRewriter &rewriter) const override {
    auto dimsAttr = allocOp->getAttrOfType<DenseI32ArrayAttr>(kAkgAllocAlignDimsAttr);
    auto bytesAttr = allocOp->getAttrOfType<DenseI32ArrayAttr>(kAkgAllocAlignBytesAttr);
    if (!dimsAttr || !bytesAttr) return failure();

    SmallVector<OpFoldResult> shape = allocOp.getMixedSizes();
    auto alignDims = dimsAttr.asArrayRef();
    auto alignBytes = bytesAttr.asArrayRef();
    if (alignDims.size() != alignBytes.size()) return failure();

    llvm::SmallVector<uint32_t> alignUnits(shape.size(), 1u);
    unsigned elemBits = cast<MemRefType>(allocOp.getType()).getElementTypeBitWidth();
    if (elemBits % kBitsPerByte != 0u) return failure();
    unsigned elemBytes = elemBits / kBitsPerByte;
    for (size_t idx = 0; idx < alignDims.size(); ++idx) {
      uint64_t bytes = static_cast<uint64_t>(alignBytes[idx]);
      if (bytes == 0u) continue;
      unsigned dimIndex = static_cast<unsigned>(alignDims[idx]);
      if (dimIndex >= alignUnits.size()) return failure();
      alignUnits[dimIndex] = static_cast<uint32_t>(bytes / elemBytes);
      if (alignUnits[dimIndex] == 0u) return failure();
    }

    SmallVector<OpFoldResult> alignedShape(shape.size());
    for (size_t dimIdx = 0; dimIdx < shape.size(); ++dimIdx) {
      alignedShape[dimIdx] = alignUpOfr(rewriter, allocOp.getLoc(), shape[dimIdx], alignUnits[dimIdx]);
    }

    bool shapesMatch = true;
    for (size_t dimIdx = 0; dimIdx < shape.size(); ++dimIdx) {
      std::optional<int64_t> alignedConst = getConstantIntValue(alignedShape[dimIdx]);
      std::optional<int64_t> originalConst = getConstantIntValue(shape[dimIdx]);
      if (alignedConst.has_value() && originalConst.has_value()) {
        if (alignedConst.value() != originalConst.value()) {
          shapesMatch = false;
          break;
        }
        continue;
      }
      Value alignedValue = alignedShape[dimIdx].dyn_cast<Value>();
      Value originalValue = shape[dimIdx].dyn_cast<Value>();
      if (alignedValue != originalValue) {
        shapesMatch = false;
        break;
      }
    }

    if (shapesMatch) {
      rewriter.modifyOpInPlace(allocOp, [&]() {
        allocOp->removeAttr(kAkgAllocAlignDimsAttr);
        allocOp->removeAttr(kAkgAllocAlignBytesAttr);
      });
      return success();
    }

    Location rewriteLoc = allocOp.getLoc();
    Value alignedView = createAlignedSubview(rewriter, allocOp, shape, alignedShape);
    if (failed(replaceAllocUsesPropagateCast(rewriter, rewriteLoc, allocOp.getMemref(), alignedView))) return failure();
    rewriter.eraseOp(allocOp);
    refreshBufferSizeMarks(rewriter, rewriteLoc, alignedView, alignedShape, shape, alignUnits, elemBytes);
    return success();
  }
};

struct AlignAllocBufferPass : public impl::AlignAllocBufferBase<AlignAllocBufferPass> {
  void runOnOperation() override;
};

void AlignAllocBufferPass::runOnOperation() {
  func::FuncOp function = getOperation();

  DenseMap<Operation *, DenseMap<int32_t, int32_t>> marksPerRoot;
  function.walk([&](Operation *operation) {
    if (isHivmOpWithName(operation, "vtranspose")) {
      handleTranspose(operation, marksPerRoot);
    } else if (isHivmOpWithName(operation, "vcast")) {
      handleVCast(operation, marksPerRoot);
    }
  });

  for (auto &entry : marksPerRoot) {
    auto allocOp = dyn_cast<memref::AllocOp>(entry.first);
    if (!allocOp) continue;
    setAllocMarkAttrs(allocOp, entry.second);
  }

  RewritePatternSet patterns(&getContext());
  patterns.add<AlignMarkedAllocPattern>(patterns.getContext());
  if (failed(applyPatternsAndFoldGreedily(function, std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAlignAllocBufferPass() { return std::make_unique<AlignAllocBufferPass>(); }

}  // namespace mlir
