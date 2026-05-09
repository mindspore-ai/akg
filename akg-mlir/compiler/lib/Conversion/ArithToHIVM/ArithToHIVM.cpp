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
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOHIVM
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

namespace mlir {
namespace {

static void propagateBufferSizeMark(ConversionPatternRewriter &rewriter, Location loc, Value src, Value dest) {
  for (Operation *user : src.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (auto attr = markOp->getAttrOfType<IntegerAttr>(kBufferSizeInByteAttr)) {
        auto srcType = cast<ShapedType>(src.getType()).getElementType();
        auto destType = cast<ShapedType>(dest.getType()).getElementType();

        if (srcType.isIndex() || destType.isIndex()) {
          return;
        }

        unsigned srcWidth = srcType.getIntOrFloatBitWidth();
        unsigned destWidth = destType.getIntOrFloatBitWidth();

        if (srcWidth == 0) {
          return;
        }

        int64_t oldSize = attr.getInt();
        int64_t newSize = (oldSize * destWidth) / srcWidth;

        auto newMarkOp = rewriter.create<annotation::MarkOp>(loc, dest);
        newMarkOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(newSize));
        break;
      }
    }
  }
}

static bool isSupportedBroadcastScalarFoldUser(Operation *user, Value broadcastResult) {
  // For commutative binary ops, allow folding when the broadcast feeds either operand.
  if (isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp, arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp,
          arith::MinUIOp, arith::MinNumFOp, arith::MinimumFOp, arith::MaxNumFOp, arith::MaximumFOp>(user)) {
    return user->getNumOperands() > 1 &&
           (user->getOperand(0) == broadcastResult || user->getOperand(1) == broadcastResult);
  }

  // For non-commutative ops, only fold when the broadcast is the right-hand operand.
  if (isa<arith::SubFOp, arith::SubIOp, arith::RemSIOp, arith::RemUIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(
        user)) {
    return user->getNumOperands() > 1 && user->getOperand(1) == broadcastResult;
  }

  return false;
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
  int64_t rank = static_cast<int64_t>(perm.size());
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

static bool advanceMemRefDimTrace(Value &curMemRef, MemRefType &baseType, int64_t &dim,
                                  std::optional<Value> &resolvedOut) {
  resolvedOut.reset();
  if (auto bitcastOp = curMemRef.getDefiningOp<hivm::BitcastOp>()) {
    curMemRef = bitcastOp.getSrc();
    return true;
  }
  if (auto subviewOp = curMemRef.getDefiningOp<memref::SubViewOp>()) {
    int64_t sourceRank = cast<MemRefType>(subviewOp.getSource().getType()).getRank();
    int64_t resultRank = baseType.getRank();
    int64_t sourceDim = sourceRank - resultRank + dim;
    auto mixedSizes = subviewOp.getMixedSizes();
    if (sourceDim >= 0 && sourceDim < static_cast<int64_t>(mixedSizes.size())) {
      if (Value mixVal = mixedSizes[sourceDim].dyn_cast<Value>()) {
        resolvedOut = mixVal;
        return false;
      }
    }
    return false;
  }
  if (auto collapseOp = curMemRef.getDefiningOp<memref::CollapseShapeOp>()) {
    auto srcType = cast<MemRefType>(collapseOp.getSrc().getType());
    auto reassoc = collapseOp.getReassociationIndices();
    if (dim < 0 || dim >= static_cast<int64_t>(reassoc.size())) return false;
    const auto &group = reassoc[dim];
    int64_t dynSrcDim = -1;
    for (int64_t srcDim : group) {
      if (srcType.isDynamicDim(srcDim)) {
        if (dynSrcDim != -1) {
          dynSrcDim = -1;
          break;
        }
        dynSrcDim = srcDim;
      }
    }
    if (dynSrcDim < 0) return false;
    curMemRef = collapseOp.getSrc();
    baseType = srcType;
    dim = dynSrcDim;
    return true;
  }
  if (auto allocOp = curMemRef.getDefiningOp<memref::AllocOp>()) {
    if (auto dimVal = getDynamicDimFromAllocLike(allocOp, dim, baseType)) {
      resolvedOut = *dimVal;
      return false;
    }
    return false;
  }
  if (auto allocaOp = curMemRef.getDefiningOp<memref::AllocaOp>()) {
    if (auto dimVal = getDynamicDimFromAllocLike(allocaOp, dim, baseType)) {
      resolvedOut = *dimVal;
      return false;
    }
    return false;
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
  if (!baseType) return failure();

  Value curMemRef = memref;
  for (unsigned step = 0; step < 32; ++step) {
    std::optional<Value> resolved;
    if (!advanceMemRefDimTrace(curMemRef, baseType, dim, resolved)) {
      if (resolved) return *resolved;
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

template <typename HIVMOp>
static void createHIVMBinaryOp(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf) {
  rewriter.create<HIVMOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf});
}

template <typename HIVMOp>
struct HIVMElementwiseBinaryCreator {
  static void create(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf) {
    createHIVMBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf);
  }
};

template <>
struct HIVMElementwiseBinaryCreator<hivm::VShROp> {
  static void create(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs, Value resBuf) {
    rewriter.create<hivm::VShROp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf},
                                  rewriter.getBoolAttr(true));
  }
};

template <typename HIVMOp>
static void createHIVMElementwiseBinaryOp(ConversionPatternRewriter &rewriter, Location loc, Value lhs, Value rhs,
                                          Value resBuf) {
  HIVMElementwiseBinaryCreator<HIVMOp>::create(rewriter, loc, lhs, rhs, resBuf);
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
        bool lhsIsScalar = isa<IntegerType, FloatType, IndexType>(lhsMemRef.getType());
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

    bool rhsIsScalar = isa<IntegerType, FloatType, IndexType>(rhsMemRef.getType());
    if (!rhsIsMemRef && !rhsIsScalar) {
      return failure();
    }

    auto memRefType = MemRefType::get(shapeAndElem->first, shapeAndElem->second);
    Value resBuf = tryGetInPlaceInitIfResultIsYieldOperand(op.getOperation());
    if (!resBuf) {
      auto resBufOr = allocMemRef(rewriter, loc, memRefType, lhsMemRef);
      if (failed(resBufOr)) {
        return failure();
      }
      resBuf = *resBufOr;
      propagateBufferSizeMark(rewriter, loc, lhsMemRef, resBuf);
    }

    createHIVMBinaryOp<HIVMOp>(rewriter, loc, lhsMemRef, rhsMemRef, resBuf);

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
    } else if (isa<arith::ExtFOp>(op)) {
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

    Value resBuf;
    if ((isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) && srcElemType.isInteger(8) && elemType.isF32()) {
      auto midMemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midBuf = rewriter.create<memref::AllocOp>(loc, midMemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midBuf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midBuf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else if (isa<arith::ExtUIOp>(op) && srcElemType.isInteger(1) && elemType.isInteger(64)) {
      auto f32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value f32Buf = rewriter.create<memref::AllocOp>(loc, f32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, f32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, f32Buf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f32Buf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else {
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
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
    } else if (isa<npuvector::ExtFOp>(op)) {
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

    Value resBuf;
    // i1 -> i64 is not supported directly (HIR rejects bool_to_int64); use i1 -> f32 -> i64.
    // i8 -> f32 is not supported directly, so we convert i8 -> f16 -> f32.
    // i8 -> i64 is not supported directly, so we convert i8 -> f16 -> f32 -> i64.
    if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(1) && elemType.isInteger(64)) {
      auto f32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value f32Buf = rewriter.create<memref::AllocOp>(loc, f32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, f32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, f32Buf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f32Buf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(8) && elemType.isInteger(64)) {
      auto midF16MemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midF16Buf = rewriter.create<memref::AllocOp>(loc, midF16MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midF16Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midF16Buf, roundingAttr, hivm::TypeFnAttr{});
      auto midF32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value midF32Buf = rewriter.create<memref::AllocOp>(loc, midF32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midF32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midF16Buf, midF32Buf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midF32Buf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else if ((isa<npuvector::SIToFPOp>(op) || isa<npuvector::UIToFPOp>(op)) && srcElemType.isInteger(8) &&
               elemType.isF32()) {
      auto midMemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midBuf = rewriter.create<memref::AllocOp>(loc, midMemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midBuf, roundingAttr, hivm::TypeFnAttr{});
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midBuf, resBuf, roundingAttr, hivm::TypeFnAttr{});
    } else {
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, resBuf, roundingAttr, hivm::TypeFnAttr{});
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

    hivm::CompareMode predicate = selectPredicate(op);
    auto predicateAttr = rewriter.getAttr<hivm::CompareModeAttr>(predicate);

    rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);

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

    hivm::CompareMode predicate = selectPredicate(op);
    auto predicateAttr = rewriter.getAttr<hivm::CompareModeAttr>(predicate);

    rewriter.create<hivm::VCmpOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);

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
    propagateBufferSizeMark(rewriter, loc, lhs, lowBuf);

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
    propagateBufferSizeMark(rewriter, loc, lhs, highBuf);

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
        bool lhsIsScalar = isa<IntegerType, FloatType, IndexType>(lhs.getType());
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

    bool rhsIsScalar = isa<IntegerType, FloatType, IndexType>(rhs.getType());
    if (!rhsIsMemRef && !rhsIsScalar) {
      return failure();
    }

    auto memRefType = MemRefType::get(shapeAndElem->first, shapeAndElem->second);
    Value resBuf = tryGetInPlaceInitIfResultIsYieldOperand(op.getOperation());
    if (!resBuf) {
      auto resBufOr = allocMemRef(rewriter, loc, memRefType, lhs);
      if (failed(resBufOr)) {
        return failure();
      }
      resBuf = *resBufOr;
      propagateBufferSizeMark(rewriter, loc, lhs, resBuf);
    }

    createHIVMElementwiseBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf);

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
    propagateBufferSizeMark(rewriter, loc, cond, resBuf);

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
    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, zeroScalar, *zeroBuf, rewriter.getDenseI64ArrayAttr({}));

    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
    rewriter.create<hivm::VErfOp>(loc, TypeRange{}, inputMemRef, *resBuf);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathCeilToHIVM : public OpConversionPattern<math::CeilOp> {
  using OpConversionPattern<math::CeilOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::CeilOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::CeilOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::CEIL);
    rewriter.create<hivm::VCastOp>(loc, TypeRange{}, inputMemRef, *resBuf, roundingAttr, hivm::TypeFnAttr{});
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathFloorToHIVM : public OpConversionPattern<math::FloorOp> {
  using OpConversionPattern<math::FloorOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::FloorOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::FloorOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const final {
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::FLOOR);
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
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
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

  LogicalResult matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() == 0) return failure();
    if (llvm::none_of(op.getResultTypes(),
                      [](Type t) { return isa<VectorType>(t) || isa<npuvector::NPUVectorType>(t); })) {
      return failure();
    }

    SmallVector<Type> newResultTypes;
    for (Type t : op.getResultTypes()) {
      auto shapeAndElem = getShapeAndElemType(t);
      if (shapeAndElem)
        newResultTypes.push_back(MemRefType::get(shapeAndElem->first, shapeAndElem->second));
      else
        newResultTypes.push_back(t);
    }

    // Do not use mergeBlocks(op.*Block(), newIfOp.*Block()): merge destroys the
    // source block and leaves the old scf.if with an empty then/else region until
    // replaceOp runs, which can segfault in ConversionPatternRewriter::applyRewrites
    // when the op is finally erased. Match AffineIfToSCFPattern: move regions with
    // inlineRegionBefore, then drop the placeholder block that scf.if builder adds.
    auto newIfOp = rewriter.create<scf::IfOp>(op.getLoc(), newResultTypes, adaptor.getCondition(),
                                              op.elseBlock() != nullptr);
    rewriter.inlineRegionBefore(op.getThenRegion(), &newIfOp.getThenRegion().back());
    rewriter.eraseBlock(&newIfOp.getThenRegion().back());
    if (op.elseBlock()) {
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
                                                             const llvm::DenseSet<int64_t> &reduceDimSet,
                                                             int64_t rank) {
  SmallVector<int64_t> collapsedShape;
  for (int64_t i = 0; i < rank; ++i) {
    if (!reduceDimSet.contains(i)) collapsedShape.push_back(srcMemRefType.getDimSize(i));
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
      for (int64_t i = 0; i < rank; ++i) reduceDims.push_back(i);
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
        if (failed(dimVal))
          return rewriter.notifyMatchFailure(op, "cannot resolve dynamic dim for reduction result alloc");
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

static bool transferReadBroadcastCanFoldToScalar(npuvector::TransferReadOp op) {
  for (Operation *user : op.getResult().getUsers()) {
    if (isSupportedBroadcastScalarFoldUser(user, op.getResult())) {
      continue;
    }
    if (isa<annotation::MarkOp>(user)) {
      continue;
    }
    return false;
  }
  return true;
}

static int64_t computeNPUVectorMarkBufferBytes(npuvector::NPUVectorType ty, Type elemType,
                                               ArrayRef<int64_t> maxPerDynamicDim) {
  int64_t n = 1;
  size_t di = 0;
  for (int64_t d : ty.getShape()) {
    if (ShapedType::isDynamic(d)) {
      if (di >= maxPerDynamicDim.size()) return 0;
      n *= maxPerDynamicDim[di++];
    } else {
      n *= d;
    }
  }
  return n * (static_cast<int64_t>(elemType.getIntOrFloatBitWidth()) / 8);
}

/// Constant max SSA values → one i64 per `?` for buffer marking (accepts per-rank or legacy).
static FailureOr<SmallVector<int64_t>> foldMaxValsForNpuMark(npuvector::NPUVectorType npuTy, ValueRange maxSizes) {
  SmallVector<int64_t> raw;
  for (Value v : maxSizes) {
    auto cop = v.getDefiningOp<arith::ConstantOp>();
    if (!cop) return failure();
    auto ia = dyn_cast<IntegerAttr>(cop.getValue());
    if (!ia) return failure();
    raw.push_back(ia.getInt());
  }
  unsigned numDyn = npuTy.getNumDynamicDims();
  if (raw.size() == static_cast<size_t>(npuTy.getRank())) {
    SmallVector<int64_t> perDyn;
    for (int64_t i = 0; i < npuTy.getRank(); ++i) {
      if (npuTy.isDynamicDim(i)) perDyn.push_back(raw[static_cast<size_t>(i)]);
    }
    if (perDyn.size() != numDyn) return failure();
    return perDyn;
  }
  if (raw.size() == 1 && numDyn > 1) raw.assign(numDyn, raw[0]);
  if (raw.size() != numDyn) return failure();
  return raw;
}

static LogicalResult setTransferReadBufferSizeMarkIfNeeded(npuvector::TransferReadOp op, Value buf, Type elemType,
                                                           npuvector::NPUVectorType npuVecType,
                                                           ConversionPatternRewriter &rewriter) {
  if (!npuVecType.hasDynamicShape() || op.getMaxSizes().empty()) {
    return success();
  }

  auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
  if (failed(folded)) return failure();

  auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), buf);
  markOp->setAttr(kBufferSizeInByteAttr,
                  rewriter.getIndexAttr(computeNPUVectorMarkBufferBytes(npuVecType, elemType, *folded)));
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
        if (compressedIdx >= dynamicSizes.size()) return failure();
        sizes.push_back(dynamicSizes[compressedIdx++]);
      }
    } else {
      sizes.push_back(rewriter.getIndexAttr(npuVecType.getDimSize(i)));
    }
    strides.push_back(rewriter.getIndexAttr(1));
  }

  if (!perAxis && compressedIdx != dynamicSizes.size()) return failure();

  return success();
}

static Value traceMemRefToRoot(Value v, int maxSteps = 32) {
  Value current = v;
  for (int i = 0; i < maxSteps; ++i) {
    Operation *def = current.getDefiningOp();
    if (!def) break;

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
  return def && (isa<memref::AllocOp>(def) || isa<memref::AllocaOp>(def));
}

static LogicalResult rewriteRank0MemRefToVectorTransferRead(npuvector::TransferReadOp op,
                                                            npuvector::TransferReadOp::Adaptor adaptor, Value source,
                                                            npuvector::NPUVectorType npuVecType,
                                                            ValueRange dynamicSizes,
                                                            ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();

  if (!adaptor.getIndices().empty()) {
    return failure();
  }

  if (transferReadBroadcastCanFoldToScalar(op)) {
    Value scalar = rewriter.create<memref::LoadOp>(loc, source, ValueRange{});
    rewriter.replaceOp(op, scalar);
    return success();
  }

  Type elemType = npuVecType.getElementType();
  auto targetMemRefType = MemRefType::get(npuVecType.getShape(), elemType);

  Value scalar = rewriter.create<memref::LoadOp>(loc, source, ValueRange{});

  SmallVector<Value> allocOperands;
  if (targetMemRefType.getNumDynamicDims() > 0) {
    if (static_cast<int64_t>(dynamicSizes.size()) == npuVecType.getRank()) {
      for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
        if (targetMemRefType.isDynamicDim(i)) allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
      }
    } else {
      allocOperands.assign(dynamicSizes.begin(), dynamicSizes.end());
    }
  }
  Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

  if (failed(setTransferReadBufferSizeMarkIfNeeded(op, tempBuf, elemType, npuVecType, rewriter))) {
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
    if (!fullSubViewTy) return failure();
    auto fullTy = cast<MemRefType>(fullSubViewTy);
    if (fullTy.getRank() < vecRank) return failure();
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
          if (targetMemRefType.isDynamicDim(i)) allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
        }
      } else {
        allocOperands.assign(dynamicSizes.begin(), dynamicSizes.end());
      }
    }
    Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

    if (failed(setTransferReadBufferSizeMarkIfNeeded(op, tempBuf, elemType, npuVecType, rewriter))) {
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
  if (!adaptor.getIndices().empty()) {
    return rewriter.notifyMatchFailure(op, "rank-0 destination expects empty indices");
  }
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

/// Unified optimized lowering when transfer_write dest traces to memref.alloc/alloca.
/// Traces dataToWrite through scf.for results to find the actual buffer (alloc),
/// then inserts a subview of dest before that buffer's earliest use and RAUW-replaces it.
/// Falls back to hivm::StoreOp when dataToWrite does not originate from an alloc.
static LogicalResult lowerNPUVectorTransferWriteAllocRootOptimized(npuvector::TransferWriteOp op, Location loc,
                                                                   Value dest, Value dataToWrite, Type resultType,
                                                                   ArrayRef<OpFoldResult> offsets,
                                                                   ArrayRef<OpFoldResult> sizes,
                                                                   ArrayRef<OpFoldResult> strides,
                                                                   ConversionPatternRewriter &rewriter) {
  auto resultMemType = cast<MemRefType>(resultType);

  // Trace through scf.for result to the underlying init buffer.
  Value actualBuf = dataToWrite;
  if (auto forOp = dataToWrite.getDefiningOp<scf::ForOp>()) {
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      if (forOp.getResult(i) == dataToWrite) {
        actualBuf = forOp.getInitArgs()[i];
        break;
      }
    }
  }

  // Require actualBuf from alloc/alloca for RAUW; otherwise fallback to Store.
  Operation *allocDef = actualBuf.getDefiningOp();
  if (!allocDef || !isa<memref::AllocOp, memref::AllocaOp>(allocDef)) {
    rewriter.setInsertionPoint(op);
    Value slicedDest = rewriter.create<memref::SubViewOp>(loc, resultMemType, dest, offsets, sizes, strides);
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, dataToWrite, slicedDest);
    rewriter.eraseOp(op);
    return success();
  }

  // Strip annotation::MarkOps attached to actualBuf.
  for (Operation *user : llvm::make_early_inc_range(actualBuf.getUsers())) {
    if (isa<annotation::MarkOp>(user)) rewriter.eraseOp(user);
  }

  // Insert subview before earliest same-block user of actualBuf (SSA dominance).
  Block *block = op->getBlock();
  Operation *insertPt = op.getOperation();
  for (Operation *user : actualBuf.getUsers()) {
    if (user == op.getOperation()) continue;
    if (user->getBlock() == block && user->isBeforeInBlock(insertPt)) insertPt = user;
  }
  rewriter.setInsertionPoint(insertPt);
  Value slicedDest = rewriter.create<memref::SubViewOp>(loc, resultMemType, dest, offsets, sizes, strides);

  rewriter.replaceAllUsesWith(actualBuf, slicedDest);
  rewriter.eraseOp(op);

  if (allocDef->use_empty()) rewriter.eraseOp(allocDef);
  return success();
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
    if (!srcTy.isDynamicDim(i) && srcTy.getDimSize(i) == 1) dims.push_back(i);
  }
  if (dims.empty()) return failure();
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
    for (int64_t i = 0; i < insertSingletonDimIndex; ++i) newShape[static_cast<size_t>(i)] = staticDimOrDynamic(i);
    newShape[static_cast<size_t>(insertSingletonDimIndex)] = 1;
    newShape[static_cast<size_t>(insertSingletonDimIndex + 1)] = staticDimOrDynamic(insertSingletonDimIndex);
    for (int64_t i = insertSingletonDimIndex + 1; i < srcRank; ++i)
      newShape[static_cast<size_t>(i + 1)] = staticDimOrDynamic(i);

    auto newMemTy = MemRefType::get(newShape, elemType);
    SmallVector<ReassociationIndices, 8> reassoc;
    reassoc.reserve(static_cast<size_t>(srcRank));
    for (int64_t j = 0; j < insertSingletonDimIndex; ++j) reassoc.push_back(ReassociationIndices{j});
    reassoc.push_back(ReassociationIndices{insertSingletonDimIndex, insertSingletonDimIndex + 1});
    for (int64_t j = insertSingletonDimIndex + 1; j < srcRank; ++j) reassoc.push_back(ReassociationIndices{j + 1});

    return rewriter.create<memref::ExpandShapeOp>(loc, newMemTy, src, reassoc).getResult();
  }

  SmallVector<int64_t> newShape(static_cast<size_t>(srcRank + 1));
  for (int64_t i = 0; i < srcRank - 1; ++i) newShape[static_cast<size_t>(i)] = staticDimOrDynamic(i);
  newShape[static_cast<size_t>(srcRank - 1)] = staticDimOrDynamic(srcRank - 1);
  newShape[static_cast<size_t>(srcRank)] = 1;
  auto newMemTy = MemRefType::get(newShape, elemType);
  SmallVector<ReassociationIndices, 8> reassoc;
  for (int64_t i = 0; i < srcRank - 1; ++i) reassoc.push_back(ReassociationIndices{i});
  reassoc.push_back(ReassociationIndices{srcRank - 1, srcRank});
  return rewriter.create<memref::ExpandShapeOp>(loc, newMemTy, src, reassoc).getResult();
}

/// Expand rank-0 memref to rank `dstRank` with all static-1 dims (VBrc rank match).
static FailureOr<Value> expandZeroRankMemRefToOnes(PatternRewriter &rewriter, Location loc, Value src, Type elemType,
                                                   int64_t dstRank) {
  if (dstRank == 0) return src;
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
  if (static_cast<int64_t>(m.size()) != srcRank) return failure();

  llvm::SmallDenseSet<int64_t> seenDest;
  for (int64_t ax : m) {
    if (ax < 0 || ax >= dstRank) return failure();
    if (!seenDest.insert(ax).second) return failure();
  }

  if (srcRank == 0) return expandZeroRankMemRefToOnes(rewriter, loc, src, elemType, dstRank);

  Value cur = src;
  MemRefType curTy = srcTy;

  SmallVector<int64_t> axisDest;
  axisDest.reserve(static_cast<size_t>(srcRank));
  for (int64_t i = 0; i < srcRank; ++i) axisDest.push_back(m[i]);

  SmallVector<int64_t> missingDestAxes;
  missingDestAxes.reserve(static_cast<size_t>(dstRank - srcRank));
  for (int64_t j = 0; j < dstRank; ++j) {
    if (seenDest.count(j) == 0) missingDestAxes.push_back(j);
  }
  llvm::sort(missingDestAxes);

  for (int64_t missingDestAxis : missingDestAxes) {
    int64_t insertSingletonDimIndex = 0;
    const int64_t curRank = static_cast<int64_t>(axisDest.size());
    while (insertSingletonDimIndex < curRank &&
           axisDest[static_cast<size_t>(insertSingletonDimIndex)] < missingDestAxis)
      ++insertSingletonDimIndex;
    FailureOr<Value> ins = insertSingletonAtPosition(rewriter, loc, cur, curTy, elemType, insertSingletonDimIndex);
    if (failed(ins)) return failure();
    cur = *ins;
    curTy = cast<MemRefType>(cur.getType());
    axisDest.insert(axisDest.begin() + insertSingletonDimIndex, missingDestAxis);
  }

  if (curTy.getRank() != dstRank) return failure();
  return cur;
}

// Constant scalar with only foldable users: replace broadcast by the source value.
static bool tryFoldBroadcast(npuvector::BroadcastOp op, Value source, ConversionPatternRewriter &rewriter) {
  auto constOp = source.getDefiningOp<arith::ConstantOp>();
  TypedAttr scalarAttr;
  if (constOp) {
    if (auto ta = dyn_cast<TypedAttr>(constOp.getValue())) scalarAttr = isa<DenseElementsAttr>(ta) ? TypedAttr{} : ta;
  }
  Value broadcastVal = op.getResult();
  const bool hasNonFoldUser = llvm::any_of(broadcastVal.getUsers(), [broadcastVal](Operation *user) {
    return !isSupportedBroadcastScalarFoldUser(user, broadcastVal) && !isa<annotation::MarkOp>(user);
  });
  if (!scalarAttr || hasNonFoldUser) return false;
  rewriter.replaceOp(op, source);
  if (constOp && constOp->getResult(0).use_empty()) rewriter.eraseOp(constOp);
  return true;
}

static LogicalResult allocBroadcastBuffer(npuvector::BroadcastOp op, Location loc, MemRefType memRefType,
                                          npuvector::NPUVectorType npuVecType, Type elemType, ValueRange dynSizes,
                                          ConversionPatternRewriter &rewriter, Value &outBuf) {
  SmallVector<Value> allocOperands;
  if (memRefType.getNumDynamicDims() > 0 && !dynSizes.empty()) {
    if (static_cast<int64_t>(dynSizes.size()) == npuVecType.getRank()) {
      for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
        if (memRefType.isDynamicDim(i)) allocOperands.push_back(dynSizes[static_cast<unsigned>(i)]);
      }
    } else {
      allocOperands.assign(dynSizes.begin(), dynSizes.end());
    }
  }
  outBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
  if (npuVecType.hasDynamicShape() && !op.getMaxSizes().empty()) {
    auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
    if (failed(folded)) return failure();
    auto markOp = rewriter.create<annotation::MarkOp>(loc, outBuf);
    markOp->setAttr(kBufferSizeInByteAttr,
                    rewriter.getIndexAttr(computeNPUVectorMarkBufferBytes(npuVecType, elemType, *folded)));
  }
  return success();
}

static LogicalResult prepareMemrefVbrc(npuvector::BroadcastOp op, Value source, MemRefType dstMemTy,
                                       npuvector::NPUVectorType npuVecType, Type elemType, Location loc,
                                       ConversionPatternRewriter &rewriter, Value &outVbrcSrc,
                                       DenseI64ArrayAttr &outBroadcastDims) {
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
      if (static_cast<int64_t>(m.size()) != srcRank)
        return rewriter.notifyMatchFailure(op, "dimension length must equal source rank");
      mVec.assign(m.begin(), m.end());
    } else {
      mVec.reserve(static_cast<size_t>(srcRank));
      for (int64_t i = 0; i < srcRank; ++i) mVec.push_back(i);
    }

    FailureOr<Value> expanded =
      expandMemRefRankWithBroadcastMapping(rewriter, loc, outVbrcSrc, srcMemTy, elemType, mVec, dstRank);
    if (failed(expanded))
      return rewriter.notifyMatchFailure(op,
                                         "npuvector.broadcast: rank extension (expand_shape) failed, "
                                         "check dimension (injective, in range, consistent rank)");
    outVbrcSrc = *expanded;
    auto expandedTy = cast<MemRefType>(outVbrcSrc.getType());
    SmallVector<int64_t> brcDims;
    for (int64_t i = 0; i < dstRank; ++i) {
      if (!expandedTy.isDynamicDim(i) && expandedTy.getDimSize(i) == 1) brcDims.push_back(i);
    }
    if (brcDims.empty())
      return rewriter.notifyMatchFailure(op, "npuvector.broadcast: VBrc needs static size-1 axes after rank extension");
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

    if (tryFoldBroadcast(op, source, rewriter)) return success();

    Value resultBuf;
    if (failed(allocBroadcastBuffer(op, loc, memRefType, npuVecType, elemType, adaptor.getDynamicSizes(), rewriter,
                                    resultBuf)))
      return failure();

    DenseI64ArrayAttr broadcastDimsAttr;
    Value vbrcSrc;
    if (failed(
          prepareMemrefVbrc(op, source, memRefType, npuVecType, elemType, loc, rewriter, vbrcSrc, broadcastDimsAttr)))
      return failure();

    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, vbrcSrc, resultBuf, broadcastDimsAttr);

    rewriter.replaceOp(op, resultBuf);
    return success();
  }
};

struct NPUVectorTransposeToHIVM : public OpConversionPattern<npuvector::TransposeOp> {
  using OpConversionPattern<npuvector::TransposeOp>::OpConversionPattern;

  static FailureOr<Value> lowerTranspose2Axis(ConversionPatternRewriter &rewriter, Location loc, Value src,
                                              MemRefType srcType, ArrayRef<int64_t> perm, Type elemType) {
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
        if (failed(dimVal)) return failure();
        allocOperands.push_back(*dimVal);
      }
    }
    Value resultBuf = rewriter.create<memref::AllocOp>(loc, resultMemRefType, allocOperands);
    propagateBufferSizeMark(rewriter, loc, src, resultBuf);
    rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, src, resultBuf, rewriter.getDenseI64ArrayAttr(perm));
    return resultBuf;
  }

  static FailureOr<Value> lowerTransposeMultiAxis(ConversionPatternRewriter &rewriter, Location loc, Value src,
                                                  MemRefType srcType, ArrayRef<int64_t> swapSeq, Type elemType) {
    int64_t rank = srcType.getRank();
    Value currentBuf = src;
    MemRefType currentType = srcType;

    for (int64_t a : swapSeq) {
      SmallVector<int64_t> newShape(rank);
      for (int64_t i = 0; i < rank; ++i) {
        int64_t srcDim = (i == a) ? a + 1 : (i == a + 1) ? a : i;
        newShape[i] = currentType.getDimSize(srcDim);
      }
      auto newMemRefType = MemRefType::get(newShape, elemType);
      SmallVector<Value> allocOperands;
      for (int64_t i = 0; i < rank; ++i) {
        if (newMemRefType.isDynamicDim(i)) {
          int64_t srcDim = (i == a) ? a + 1 : (i == a + 1) ? a : i;
          auto dimVal = getMemRefDimValue(currentBuf, srcDim);
          if (failed(dimVal)) return failure();
          allocOperands.push_back(*dimVal);
        }
      }
      Value newBuf = rewriter.create<memref::AllocOp>(loc, newMemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, currentBuf, newBuf);
      SmallVector<int64_t> swapPerm = buildAdjacentSwapPerm(rank, a);
      rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, currentBuf, newBuf,
                                          rewriter.getDenseI64ArrayAttr(swapPerm));
      currentBuf = newBuf;
      currentType = newMemRefType;
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
      if (perm[i] != i) ++transposeAxisNum;
    }

    Type elemType = npuResultType.getElementType();

    if (transposeAxisNum == 0) {
      rewriter.replaceOp(op, src);
      return success();
    }

    if (transposeAxisNum == 2) {
      auto resultBuf = lowerTranspose2Axis(rewriter, loc, src, srcType, perm, elemType);
      if (failed(resultBuf)) return failure();
      rewriter.replaceOp(op, *resultBuf);
      return success();
    }

    SmallVector<int64_t> swapSeq = decomposePermToAdjacentSwaps(perm);
    if (swapSeq.empty()) {
      return rewriter.notifyMatchFailure(op, "failed to decompose permutation");
    }

    auto resultBuf = lowerTransposeMultiAxis(rewriter, loc, src, srcType, swapSeq, elemType);
    if (failed(resultBuf)) return failure();
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
/// iter_args with inits; empty yield; remap for results to yielded memref or init[i].
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
      if (isa<MemRefType>(yielded.getType())) {
        resultReplacements.push_back(yielded);
      } else {
        resultReplacements.push_back(forOp.getInitArgs()[i]);
      }
    }

    Location loc = forOp.getLoc();
    rewriter.setInsertionPoint(forOp);
    scf::ForOp newFor = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());

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
  patterns.add<MathCeilToHIVM>(patterns.getContext());
  patterns.add<MathFloorToHIVM>(patterns.getContext());
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
                    arith::ArithDialect, math::MathDialect, scf::SCFDialect, annotation::AnnotationDialect>();
  }
  void runOnOperation() override;
};

void ArithToHIVMConversionPass::runOnOperation() {
  ConversionTarget target(getContext());
  // HIVM and Tensor are legal
  target.addLegalDialect<hivm::HIVMDialect, tensor::TensorDialect, memref::MemRefDialect, scf::SCFDialect,
                         BuiltinDialect, annotation::AnnotationDialect>();
  target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalArithOp);
  target.addDynamicallyLegalDialect<math::MathDialect>(isLegalMathOp);
  target.addDynamicallyLegalOp<scf::ForOp>(isLegalSCFForOp);
  target.addDynamicallyLegalOp<scf::IfOp>([](scf::IfOp op) {
    return llvm::none_of(op.getResultTypes(), isVectorOrNPUVectorType);
  });
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
