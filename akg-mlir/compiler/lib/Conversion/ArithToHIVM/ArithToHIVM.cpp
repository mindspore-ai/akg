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
#include <optional>
#include <type_traits>

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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
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
  if (isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp,
          arith::MaxSIOp, arith::MaxUIOp, arith::MinSIOp, arith::MinUIOp,
          arith::MinNumFOp, arith::MinimumFOp, arith::MaxNumFOp,
          arith::MaximumFOp>(user)) {
    return user->getNumOperands() > 1 &&
           (user->getOperand(0) == broadcastResult ||
            user->getOperand(1) == broadcastResult);
  }

  // For non-commutative ops, only fold when the broadcast is the right-hand operand.
  if (isa<arith::SubFOp, arith::SubIOp, arith::RemSIOp, arith::RemUIOp,
          arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp>(user)) {
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

static std::optional<unsigned> getMemRefDynamicSizeOperandIndex(MemRefType baseType,
                                                               int64_t targetDim) {
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
static std::optional<Value> getDynamicDimFromAllocLike(AllocLikeOp allocLikeOp,
                                                       int64_t dim,
                                                       MemRefType baseType) {
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

static FailureOr<Value> getMemRefDimValue(Value memref, int64_t dim) {
  auto baseType = dyn_cast<MemRefType>(memref.getType());
  if (!baseType) {
    return failure();
  }

  Value curMemRef = memref;
  for (unsigned step = 0; step < 32; ++step) {
    if (auto bitcastOp = curMemRef.getDefiningOp<hivm::BitcastOp>()) {
      curMemRef = bitcastOp.getSrc();
      continue;
    }
    if (auto subviewOp = curMemRef.getDefiningOp<memref::SubViewOp>()) {
      int64_t sourceRank =
          cast<MemRefType>(subviewOp.getSource().getType()).getRank();
      int64_t resultRank = baseType.getRank();
      int64_t sourceDim = sourceRank - resultRank + dim;
      auto mixedSizes = subviewOp.getMixedSizes();
      if (sourceDim >= 0 && sourceDim < static_cast<int64_t>(mixedSizes.size())) {
        if (Value v = mixedSizes[sourceDim].dyn_cast<Value>()) {
          return v;
        }
      }
      break;
    }
    if (auto allocOp = curMemRef.getDefiningOp<memref::AllocOp>()) {
      if (auto dimVal = getDynamicDimFromAllocLike(allocOp, dim, baseType)) {
        return *dimVal;
      }
      break;
    }
    if (auto allocaOp = curMemRef.getDefiningOp<memref::AllocaOp>()) {
      if (auto dimVal = getDynamicDimFromAllocLike(allocaOp, dim, baseType)) {
        return *dimVal;
      }
      break;
    }
    if (auto next = traceToScfForInitArgFromIterArg(curMemRef)) {
      curMemRef = *next;
      continue;
    }
    if (auto next = traceToScfForInitArgFromResult(curMemRef)) {
      curMemRef = *next;
      continue;
    }
    break;
  }
  return failure();
}

static FailureOr<Value> allocMemRef(ConversionPatternRewriter &rewriter,
                                  Location loc, MemRefType type, Value dimSource) {
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
static void createHIVMBinaryOp(ConversionPatternRewriter &rewriter, Location loc,
                               Value lhs, Value rhs, Value resBuf) {
  rewriter.create<HIVMOp>(loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf});
}

template <typename HIVMOp>
struct HIVMElementwiseBinaryCreator {
  static void create(ConversionPatternRewriter &rewriter, Location loc,
                     Value lhs, Value rhs, Value resBuf) {
    createHIVMBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf);
  }
};

template <>
struct HIVMElementwiseBinaryCreator<hivm::VShROp> {
  static void create(ConversionPatternRewriter &rewriter, Location loc,
                     Value lhs, Value rhs, Value resBuf) {
    rewriter.create<hivm::VShROp>(loc, TypeRange{}, ValueRange{lhs, rhs},
                                  ValueRange{resBuf}, rewriter.getBoolAttr(true));
  }
};

template <typename HIVMOp>
static void createHIVMElementwiseBinaryOp(ConversionPatternRewriter &rewriter, Location loc,
                                         Value lhs, Value rhs, Value resBuf) {
  HIVMElementwiseBinaryCreator<HIVMOp>::create(rewriter, loc, lhs, rhs, resBuf);
}

template <typename ArithOp, typename HIVMOp>
struct BinaryArithToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  static constexpr bool isCommutative() {
    return std::is_same_v<ArithOp, arith::AddFOp> ||
           std::is_same_v<ArithOp, arith::AddIOp> ||
           std::is_same_v<ArithOp, arith::MulFOp> ||
           std::is_same_v<ArithOp, arith::MulIOp> ||
           std::is_same_v<ArithOp, arith::MaxSIOp> ||
           std::is_same_v<ArithOp, arith::MaxUIOp> ||
           std::is_same_v<ArithOp, arith::MinSIOp> ||
           std::is_same_v<ArithOp, arith::MinUIOp>;
  }

  LogicalResult matchAndRewrite(ArithOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    auto resBufOr = allocMemRef(rewriter, loc, memRefType, lhsMemRef);
    if (failed(resBufOr)) {
      return failure();
    }
    Value resBuf = *resBufOr;
    propagateBufferSizeMark(rewriter, loc, lhsMemRef, resBuf);

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
  return (isI16ToI8 || isI32ToI16 || isI32ToI8 ||
          isI64ToI8 || isI64ToI16 || isI64ToI32);
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
    } else if (isa<arith::ExtSIOp>(op) || isa<arith::ExtUIOp>(op) ||
               isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) {
      return hivm::RoundMode::RINT;
    } else if (isa<arith::FPToSIOp>(op) || isa<arith::FPToUIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::TRUNC;
    }
    llvm_unreachable("unsupported arith op to hivm");
  }

  LogicalResult matchAndRewrite(CastOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    if ((isa<arith::SIToFPOp>(op) || isa<arith::UIToFPOp>(op)) &&
        srcElemType.isInteger(8) && elemType.isF32()) {
      auto midMemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midBuf = rewriter.create<memref::AllocOp>(loc, midMemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midBuf, roundingAttr);
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midBuf, resBuf, roundingAttr);
    } else if (isa<arith::ExtUIOp>(op) && srcElemType.isInteger(1) && elemType.isInteger(64)) {
      auto f32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value f32Buf = rewriter.create<memref::AllocOp>(loc, f32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, f32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, f32Buf, roundingAttr);
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f32Buf, resBuf, roundingAttr);
    } else {
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, resBuf, roundingAttr);
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
    } else if (isa<npuvector::ExtSIOp>(op) || isa<npuvector::ExtUIOp>(op) ||
               isa<npuvector::SIToFPOp>(op) || isa<npuvector::UIToFPOp>(op)) {
      return hivm::RoundMode::RINT;
    } else if (isa<npuvector::FPToSIOp>(op) || isa<npuvector::FPToUIOp>(op)) {
      if (isOverFlowMode(inType, outType)) {
        return hivm::RoundMode::TRUNCWITHOVERFLOW;
      }
      return hivm::RoundMode::TRUNC;
    }
    llvm_unreachable("unsupported npuvector op to hivm");
  }

  LogicalResult matchAndRewrite(CastOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(1) &&
        elemType.isInteger(64)) {
      auto f32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value f32Buf = rewriter.create<memref::AllocOp>(loc, f32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, f32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, f32Buf, roundingAttr);
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, f32Buf, resBuf, roundingAttr);
    } else if (isa<npuvector::ExtUIOp>(op) && srcElemType.isInteger(8) &&
        elemType.isInteger(64)) {
      auto midF16MemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midF16Buf = rewriter.create<memref::AllocOp>(loc, midF16MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midF16Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midF16Buf, roundingAttr);
      auto midF32MemRefType = MemRefType::get(shape, rewriter.getF32Type());
      Value midF32Buf = rewriter.create<memref::AllocOp>(loc, midF32MemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midF32Buf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midF16Buf, midF32Buf, roundingAttr);
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midF32Buf, resBuf, roundingAttr);
    } else if ((isa<npuvector::SIToFPOp>(op) || isa<npuvector::UIToFPOp>(op)) &&
        srcElemType.isInteger(8) && elemType.isF32()) {
      auto midMemRefType = MemRefType::get(shape, rewriter.getF16Type());
      Value midBuf = rewriter.create<memref::AllocOp>(loc, midMemRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, midBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, midBuf, roundingAttr);
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, midBuf, resBuf, roundingAttr);
    } else {
      resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
      propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);
      rewriter.create<hivm::VCastOp>(loc, TypeRange{}, srcMemRef, resBuf, roundingAttr);
    }

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithBitcastToHIVM : public OpConversionPattern<arith::BitcastOp> {
  using OpConversionPattern<arith::BitcastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<arith::BitcastOp>::OpAdaptor;

  LogicalResult matchAndRewrite(arith::BitcastOp op,
                                OpAdaptor adaptor,
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

    Value res = rewriter.create<hivm::BitcastOp>(
        loc,
        memRefType,
        srcMemRef);

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct NPUVectorBitcastToHIVM : public OpConversionPattern<npuvector::BitcastOp> {
  using OpConversionPattern<npuvector::BitcastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<npuvector::BitcastOp>::OpAdaptor;

  LogicalResult matchAndRewrite(npuvector::BitcastOp op,
                                OpAdaptor adaptor,
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

  LogicalResult matchAndRewrite(CompareOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

    rewriter.create<hivm::VCmpOp>(
        loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);

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

  LogicalResult matchAndRewrite(CompareOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

    rewriter.create<hivm::VCmpOp>(
        loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf}, predicateAttr);

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename ArithOp>
struct ArithMulExtToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  LogicalResult matchAndRewrite(ArithOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

    rewriter.create<hivm::VMulExtOp>(
        loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange(dsts));

    rewriter.replaceOp(op, dsts);
    return success();
  }
};

template <typename ArithOp, typename HIVMOp>
struct ElementwiseOpToHIVMBinary : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  static constexpr bool isCommutative() {
    return std::is_same_v<ArithOp, arith::MinNumFOp> ||
           std::is_same_v<ArithOp, arith::MinimumFOp> ||
           std::is_same_v<ArithOp, arith::MaxNumFOp> ||
           std::is_same_v<ArithOp, arith::MaximumFOp>;
  }

  LogicalResult matchAndRewrite(ArithOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    auto resBufOr = allocMemRef(rewriter, loc, memRefType, lhs);
    if (failed(resBufOr)) {
      return failure();
    }
    Value resBuf = *resBufOr;
    propagateBufferSizeMark(rewriter, loc, lhs, resBuf);

    createHIVMElementwiseBinaryOp<HIVMOp>(rewriter, loc, lhs, rhs, resBuf);

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

template <typename SelectOp>
struct ArithSelectToHIVM : public OpConversionPattern<SelectOp> {
  using OpConversionPattern<SelectOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<SelectOp>::OpAdaptor;

  LogicalResult matchAndRewrite(SelectOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

    rewriter.create<hivm::VSelOp>(
        loc,
        TypeRange{},
        ValueRange{cond, trueVal, falseVal},
        ValueRange{resBuf},
        Value(),
        SmallVector<int64_t>{},
        SmallVector<int64_t>{});

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithConstantToHIVM : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op,
                                OpAdaptor adaptor,
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

    rewriter.create<hivm::VBrcOp>(
        loc, TypeRange{}, scalarConstant, resBuf,
        rewriter.getDenseI64ArrayAttr(ArrayRef<int64_t>{}));

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

struct ArithNegfToHIVM : public OpConversionPattern<arith::NegFOp> {
  using OpConversionPattern<arith::NegFOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<arith::NegFOp>::OpAdaptor;

  LogicalResult matchAndRewrite(arith::NegFOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

    Value zeroScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 0.0));
    auto zeroBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(zeroBuf)) {
      return failure();
    }
    rewriter.create<hivm::VBrcOp>(
        loc, TypeRange{}, zeroScalar, *zeroBuf,
        rewriter.getDenseI64ArrayAttr({}));

    auto resBuf = allocMemRef(rewriter, loc, memRefType, inputMemRef);
    if (failed(resBuf)) {
      return failure();
    }
    propagateBufferSizeMark(rewriter, loc, inputMemRef, *resBuf);
    rewriter.create<hivm::VSubOp>(
        loc, TypeRange{}, ValueRange{*zeroBuf, inputMemRef}, ValueRange{*resBuf});

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathExpToHIVM : public OpConversionPattern<math::ExpOp> {
  using OpConversionPattern<math::ExpOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::ExpOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    rewriter.create<hivm::VExpOp>(
        loc, TypeRange{}, inputMemRef, *resBuf);

    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathLogToHIVM : public OpConversionPattern<math::LogOp> {
  using OpConversionPattern<math::LogOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::LogOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::LogOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::AbsFOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::SqrtOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::RsqrtOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::TanhOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::SinOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::CosOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::ErfOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(math::CeilOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    rewriter.create<hivm::VCastOp>(loc, TypeRange{}, inputMemRef, *resBuf, roundingAttr);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathFloorToHIVM : public OpConversionPattern<math::FloorOp> {
  using OpConversionPattern<math::FloorOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::FloorOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::FloorOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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
    rewriter.create<hivm::VCastOp>(loc, TypeRange{}, inputMemRef, *resBuf, roundingAttr);
    rewriter.replaceOp(op, *resBuf);
    return success();
  }
};

struct MathAbsIToHIVM : public OpConversionPattern<math::AbsIOp> {
  using OpConversionPattern<math::AbsIOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<math::AbsIOp>::OpAdaptor;

  LogicalResult matchAndRewrite(math::AbsIOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
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

  LogicalResult matchAndRewrite(vector::ReductionOp op,
                                OpAdaptor adaptor,
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
    rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{}, sourceMemRef, resultBuf, Value(),
        reduceOpAttr, rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    rewriter.replaceOp(op, resultBuf);

    return success();
  }
};

struct VectorBroadcastToHIVM : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                OpAdaptor adaptor,
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
    rewriter.create<hivm::VBrcOp>(
        loc,
        TypeRange{},
        source,
        resultBuf,
        rewriter.getDenseI64ArrayAttr({}));

    rewriter.replaceOp(op, resultBuf);

    return success();
  }
};


struct ScfForToHIVM : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(scf::ForOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Only handle loops that have at least one vector-typed iteration argument.
    if (llvm::none_of(op.getResultTypes(), [](Type t) {
          return isa<VectorType>(t) || isa<npuvector::NPUVectorType>(t);
        })) {
      return failure();
    }

    auto newForOp = rewriter.create<scf::ForOp>(
        op.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
        adaptor.getStep(), adaptor.getInitArgs());

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

  LogicalResult matchAndRewrite(scf::YieldOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperands(), [](Value v) {
          return isa<VectorType>(v.getType()) ||
             isa<npuvector::NPUVectorType>(v.getType());
        })) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};


struct VectorTransferReadToHIVM : public OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern<vector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                OpAdaptor adaptor,
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
    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        vecType.getShape(), memRefType, offsets, sizes, strides);

    Value finalSource = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), source, offsets, sizes, strides);

    Type elemType = vecType.getElementType();
    auto targetMemRefType = MemRefType::get(vecType.getShape(), elemType);
    Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType);

    rewriter.create<hivm::LoadOp>(
        loc,
        TypeRange{},
        finalSource,
        tempBuf);

    rewriter.replaceOp(op, tempBuf);
    return success();
  }
};


struct VectorTransferWriteToHIVM : public OpConversionPattern<vector::TransferWriteOp> {
  using OpConversionPattern<vector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                OpAdaptor adaptor,
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
    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        vecShape, memRefType, offsets, sizes, strides);

    Value slicedDest = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);

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
          finalData = rewriter.create<memref::CollapseShapeOp>(
              loc, targetType, finalData, reassociation);
       }
    }

    rewriter.create<hivm::StoreOp>(
        loc,
        TypeRange{},
        finalData,
        finalDest);

    rewriter.eraseOp(op);

    return success();
  }
};

struct MemRefStoreToHIVM : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                OpAdaptor adaptor,
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
      Value casted = rewriter.create<memref::ReinterpretCastOp>(
          op.getLoc(), newType, memref, offset, sizes, strides);
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

  LogicalResult matchAndRewrite(npuvector::ReductionOp op,
                                OpAdaptor adaptor,
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
    rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{}, sourceMemRef, resultBuf, Value(),
        reduceOpAttr, rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    Type resultType = op.getResult().getType();
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

static int64_t computeNPUVectorMarkBufferBytes(npuvector::NPUVectorType ty,
                                               Type elemType,
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
static FailureOr<SmallVector<int64_t>> foldMaxValsForNpuMark(npuvector::NPUVectorType npuTy,
                                                             ValueRange maxSizes) {
  SmallVector<int64_t> raw;
  for (Value v : maxSizes) {
    auto cop = v.getDefiningOp<arith::ConstantOp>();
    if (!cop)
      return failure();
    auto ia = dyn_cast<IntegerAttr>(cop.getValue());
    if (!ia)
      return failure();
    raw.push_back(ia.getInt());
  }
  unsigned numDyn = npuTy.getNumDynamicDims();
  if (raw.size() == static_cast<size_t>(npuTy.getRank())) {
    SmallVector<int64_t> perDyn;
    for (int64_t i = 0; i < npuTy.getRank(); ++i) {
      if (npuTy.isDynamicDim(i))
        perDyn.push_back(raw[static_cast<size_t>(i)]);
    }
    if (perDyn.size() != numDyn)
      return failure();
    return perDyn;
  }
  if (raw.size() == 1 && numDyn > 1)
    raw.assign(numDyn, raw[0]);
  if (raw.size() != numDyn)
    return failure();
  return raw;
}

static LogicalResult setTransferReadBufferSizeMarkIfNeeded(npuvector::TransferReadOp op,
                                                           Value buf,
                                                           Type elemType,
                                                           npuvector::NPUVectorType npuVecType,
                                                           ConversionPatternRewriter &rewriter) {
  if (!npuVecType.hasDynamicShape() || op.getMaxSizes().empty()) {
    return success();
  }

  auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
  if (failed(folded))
    return failure();

  auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), buf);
  markOp->setAttr(kBufferSizeInByteAttr,
                  rewriter.getIndexAttr(computeNPUVectorMarkBufferBytes(
                      npuVecType, elemType, *folded)));
  return success();
}

static LogicalResult buildTransferReadSizesAndStrides(int64_t memRefRank,
                                                      int64_t vecRank,
                                                      npuvector::NPUVectorType npuVecType,
                                                      ValueRange dynamicSizes,
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
    if (perAxis) {
      sizes.push_back(dynamicSizes[static_cast<unsigned>(i)]);
    } else {
      if (npuVecType.isDynamicDim(i)) {
        if (compressedIdx >= dynamicSizes.size())
          return failure();
        sizes.push_back(dynamicSizes[compressedIdx++]);
      } else {
        sizes.push_back(rewriter.getIndexAttr(npuVecType.getDimSize(i)));
      }
    }
    strides.push_back(rewriter.getIndexAttr(1));
  }

  if (!perAxis && compressedIdx != dynamicSizes.size())
    return failure();

  return success();
}

static Value traceMemRefToRoot(Value v, int maxSteps = 32) {
  Value current = v;
  for (int i = 0; i < maxSteps; ++i) {
    Operation *def = current.getDefiningOp();
    if (!def) break;

    if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
      current = subview.getSource();
    } else if (auto cast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      current = cast.getSource();
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
                                                           npuvector::TransferReadOp::Adaptor adaptor,
                                                           Value source,
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
        if (targetMemRefType.isDynamicDim(i))
          allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
      }
    } else {
      allocOperands.assign(dynamicSizes.begin(), dynamicSizes.end());
    }
  }
  Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

  if (failed(setTransferReadBufferSizeMarkIfNeeded(op, tempBuf, elemType, npuVecType, rewriter))) {
    return failure();
  }

  rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, scalar, tempBuf,
                               rewriter.getDenseI64ArrayAttr({}));
  rewriter.replaceOp(op, tempBuf);
  return success();
}

struct NPUVectorTransferReadToHIVM : public OpConversionPattern<npuvector::TransferReadOp> {
  using OpConversionPattern<npuvector::TransferReadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::TransferReadOp op,
                                OpAdaptor adaptor,
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
      return rewriteRank0MemRefToVectorTransferRead(op, adaptor, source, npuVecType,
                                                   dynamicSizes, rewriter);
    }

    if (failed(buildTransferReadSizesAndStrides(memRefRank, vecRank, npuVecType, dynamicSizes,
                                               rewriter, sizes, strides))) {
      return failure();
    }

    Type fullSubViewTy = memref::SubViewOp::inferResultType(memRefType, offsets, sizes, strides);
    if (!fullSubViewTy)
      return failure();
    auto fullTy = cast<MemRefType>(fullSubViewTy);
    if (fullTy.getRank() < vecRank)
      return failure();
    ArrayRef<int64_t> fs = fullTy.getShape();
    SmallVector<int64_t> reducedShape(fs.begin() + (fs.size() - static_cast<size_t>(vecRank)),
                                      fs.end());

    Type subViewResultType = memref::SubViewOp::inferRankReducedResultType(
        reducedShape, memRefType, offsets, sizes, strides);

    Value finalSource = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(subViewResultType), source, offsets, sizes, strides);

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
          if (targetMemRefType.isDynamicDim(i))
            allocOperands.push_back(dynamicSizes[static_cast<unsigned>(i)]);
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

static LogicalResult rewriteNPUVectorTransferWriteRank0(
    npuvector::TransferWriteOp op,
    npuvector::TransferWriteOp::Adaptor adaptor,
    Location loc,
    Value dataToWrite,
    Value dest,
    MemRefType dataMemRefType,
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
      castedData = rewriter.create<memref::ReinterpretCastOp>(
          loc, dataType, dataToWrite, offset, sizes, strides);
    }
    auto destType = MemRefType::get({1}, destMemRefType.getElementType());
    Value castedDest = rewriter.create<memref::ReinterpretCastOp>(
        loc, destType, dest, offset, sizes, strides);
    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, castedData, castedDest);
    rewriter.eraseOp(op);
    return success();
  }
  return rewriter.notifyMatchFailure(op, "unsupported rank-0 destination");
}

static LogicalResult buildNPUVectorTransferWriteSizesStrides(
    Value dataToWrite,
    MemRefType dataMemRefType,
    int64_t memRefRank,
    int64_t dataRank,
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

static bool tryRewriteNPUVectorTransferWriteToAllocSlicedDest(
    npuvector::TransferWriteOp op,
    Location loc,
    Value dest,
    Value dataToWrite,
    Type resultType,
    ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides,
    ConversionPatternRewriter &rewriter) {
  Value root = traceMemRefToRoot(dest);
  if (!isRootFromAlloc(root)) {
    return false;
  }
  Operation *producer = nullptr;
  for (Operation *user : dataToWrite.getUsers()) {
    if (isa<annotation::MarkOp>(user)) {
      continue;
    }
    producer = user;
    break;
  }
  if (producer) {
    rewriter.setInsertionPoint(producer);
    Value slicedDest = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);
    for (unsigned i = 0, e = producer->getNumOperands(); i < e; ++i) {
      if (producer->getOperand(i) == dataToWrite) {
        rewriter.modifyOpInPlace(producer, [&]() {
          producer->setOperand(i, slicedDest);
        });
        break;
      }
    }
    for (Operation *user : llvm::make_early_inc_range(dataToWrite.getUsers())) {
      if (isa<annotation::MarkOp>(user)) {
        rewriter.eraseOp(user);
      }
    }
    if (auto allocOp = dataToWrite.getDefiningOp<memref::AllocOp>()) {
      rewriter.eraseOp(allocOp);
    }
  }
  rewriter.eraseOp(op);
  return true;
}

struct NPUVectorTransferWriteToHIVM : public OpConversionPattern<npuvector::TransferWriteOp> {
  using OpConversionPattern<npuvector::TransferWriteOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::TransferWriteOp op,
                                OpAdaptor adaptor,
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

    auto dataMemRefType = cast<MemRefType>(dataToWrite.getType());
    auto destMemRefType = cast<MemRefType>(dest.getType());

    int64_t memRefRank = destMemRefType.getRank();
    int64_t dataRank = dataMemRefType.getRank();

    if (memRefRank == 0) {
      return rewriteNPUVectorTransferWriteRank0(op, adaptor, loc, dataToWrite, dest,
                                                dataMemRefType, destMemRefType,
                                                rewriter);
    }

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(adaptor.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    if (failed(buildNPUVectorTransferWriteSizesStrides(
            dataToWrite, dataMemRefType, memRefRank, dataRank, rewriter, sizes,
            strides))) {
      return failure();
    }

    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        dataMemRefType.getShape(), destMemRefType, offsets, sizes, strides);

    if (tryRewriteNPUVectorTransferWriteToAllocSlicedDest(
            op, loc, dest, dataToWrite, resultType, offsets, sizes, strides,
            rewriter)) {
      return success();
    }

    Value slicedDest = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);

    rewriter.create<hivm::StoreOp>(
        loc,
        TypeRange{},
        dataToWrite,
        slicedDest);

    rewriter.eraseOp(op);

    return success();
  }
};

struct NPUVectorBroadcastToHIVM : public OpConversionPattern<npuvector::BroadcastOp> {
  using OpConversionPattern<npuvector::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::BroadcastOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value source = adaptor.getSource();

    Type resultType = op.getResult().getType();
    auto npuVecType = dyn_cast<npuvector::NPUVectorType>(resultType);
    if (!npuVecType) {
      return failure();
    }

    Type elemType = npuVecType.getElementType();
    if (elemType.isIndex()) {
      elemType = rewriter.getI64Type();
      source = rewriter.create<arith::IndexCastOp>(loc, elemType, source);
    }
    auto memRefType = MemRefType::get(npuVecType.getShape(), elemType);

    auto constOp = source.getDefiningOp<arith::ConstantOp>();
    TypedAttr scalarAttr;
    if (constOp) {
      if (auto ta = dyn_cast<TypedAttr>(constOp.getValue()))
        scalarAttr = isa<DenseElementsAttr>(ta) ? TypedAttr{} : ta;
    }

    Value broadcastVal = op.getResult();
    const bool hasNonFoldUser = llvm::any_of(
        broadcastVal.getUsers(), [broadcastVal](Operation *user) {
          return !isSupportedBroadcastScalarFoldUser(user, broadcastVal) &&
                 !isa<annotation::MarkOp>(user);
        });

    bool foldBroadcast = scalarAttr && !hasNonFoldUser;
    if (foldBroadcast) {
      rewriter.replaceOp(op, source);
      if (constOp && constOp->getResult(0).use_empty()) {
        rewriter.eraseOp(constOp);
      }
      return success();
    }

    SmallVector<Value> allocOperands;
    auto dynSizes = adaptor.getDynamicSizes();
    if (memRefType.getNumDynamicDims() > 0 && !dynSizes.empty()) {
      if (static_cast<int64_t>(dynSizes.size()) == npuVecType.getRank()) {
        [&]() {
          for (int64_t i = 0; i < npuVecType.getRank(); ++i) {
            if (memRefType.isDynamicDim(i))
              allocOperands.push_back(dynSizes[static_cast<unsigned>(i)]);
          }
        }();
      } else {
        allocOperands.assign(dynSizes.begin(), dynSizes.end());
      }
    }
    Value resultBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    if (npuVecType.hasDynamicShape() && !op.getMaxSizes().empty()) {
      auto folded = foldMaxValsForNpuMark(npuVecType, op.getMaxSizes());
      if (failed(folded))
        return failure();
      auto markOp = rewriter.create<annotation::MarkOp>(loc, resultBuf);
      markOp->setAttr(kBufferSizeInByteAttr,
                      rewriter.getIndexAttr(computeNPUVectorMarkBufferBytes(
                          npuVecType, elemType, *folded)));
    }

    rewriter.create<hivm::VBrcOp>(loc, TypeRange{}, source, resultBuf, rewriter.getDenseI64ArrayAttr({}));

    rewriter.replaceOp(op, resultBuf);
    return success();
  }
};

struct NPUVectorTransposeToHIVM : public OpConversionPattern<npuvector::TransposeOp> {
  using OpConversionPattern<npuvector::TransposeOp>::OpConversionPattern;

  static FailureOr<Value> lowerTranspose2Axis(ConversionPatternRewriter &rewriter,
                                               Location loc, Value src,
                                               MemRefType srcType,
                                               ArrayRef<int64_t> perm,
                                               Type elemType) {
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
    Value resultBuf =
        rewriter.create<memref::AllocOp>(loc, resultMemRefType, allocOperands);
    rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, src, resultBuf,
                                        rewriter.getDenseI64ArrayAttr(perm));
    return resultBuf;
  }

  static FailureOr<Value> lowerTransposeMultiAxis(
      ConversionPatternRewriter &rewriter, Location loc, Value src,
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
      Value newBuf =
          rewriter.create<memref::AllocOp>(loc, newMemRefType, allocOperands);
      SmallVector<int64_t> swapPerm = buildAdjacentSwapPerm(rank, a);
      rewriter.create<hivm::VTransposeOp>(loc, TypeRange{}, currentBuf, newBuf,
                                          rewriter.getDenseI64ArrayAttr(swapPerm));
      currentBuf = newBuf;
      currentType = newMemRefType;
    }
    return currentBuf;
  }

  LogicalResult matchAndRewrite(npuvector::TransposeOp op,
                                OpAdaptor adaptor,
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

    auto resultBuf = lowerTransposeMultiAxis(rewriter, loc, src, srcType, swapSeq,
                                             elemType);
    if (failed(resultBuf)) return failure();
    rewriter.replaceOp(op, *resultBuf);
    return success();
  }
};

struct NPUVectorIndexCastToHIVM : public OpConversionPattern<npuvector::IndexCastOp> {
  using OpConversionPattern<npuvector::IndexCastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(npuvector::IndexCastOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands()[0];
    rewriter.replaceOp(op, input);
    return success();
  }
};

}  // namespace

void hivm::populateArithToHIVMConversionPatterns(
    RewritePatternSet &patterns) {

  patterns.add<
      BinaryArithToHIVM<arith::AddFOp, hivm::VAddOp>,
      BinaryArithToHIVM<arith::AddIOp, hivm::VAddOp>,
      BinaryArithToHIVM<arith::MulFOp, hivm::VMulOp>,
      BinaryArithToHIVM<arith::MulIOp, hivm::VMulOp>,
      BinaryArithToHIVM<arith::SubFOp, hivm::VSubOp>,
      BinaryArithToHIVM<arith::SubIOp, hivm::VSubOp>,
      BinaryArithToHIVM<arith::DivFOp, hivm::VDivOp>,
      BinaryArithToHIVM<arith::DivSIOp, hivm::VDivOp>,
      BinaryArithToHIVM<arith::DivUIOp, hivm::VDivOp>,
      BinaryArithToHIVM<arith::MaxSIOp, hivm::VMaxOp>,
      BinaryArithToHIVM<arith::MaxUIOp, hivm::VMaxOp>,
      BinaryArithToHIVM<arith::MinSIOp, hivm::VMinOp>,
      BinaryArithToHIVM<arith::MinUIOp, hivm::VMinOp>>(
      patterns.getContext());

  patterns.add<
      VectorTransferReadToHIVM,
      VectorTransferWriteToHIVM>(patterns.getContext());

  patterns.add<
      UnaryArithToHIVMCast<arith::ExtFOp>,
      UnaryArithToHIVMCast<arith::FPToSIOp>,
      UnaryArithToHIVMCast<arith::FPToUIOp>,
      UnaryArithToHIVMCast<arith::SIToFPOp>,
      UnaryArithToHIVMCast<arith::UIToFPOp>,
      UnaryArithToHIVMCast<arith::ExtSIOp>,
      UnaryArithToHIVMCast<arith::ExtUIOp>,
      UnaryArithToHIVMCast<arith::TruncIOp>,
      UnaryArithToHIVMCast<arith::TruncFOp>,
      ArithCmpToHIVM<arith::CmpFOp>,
      ArithCmpToHIVM<arith::CmpIOp>,
      ArithMulExtToHIVM<arith::MulSIExtendedOp>,
      ArithMulExtToHIVM<arith::MulUIExtendedOp>>(
      patterns.getContext());
  patterns.add<
      ElementwiseOpToHIVMBinary<arith::AndIOp, hivm::VAndOp>,
      ElementwiseOpToHIVMBinary<arith::OrIOp, hivm::VOrOp>,
      ElementwiseOpToHIVMBinary<arith::XOrIOp, hivm::VXorOp>,
      ElementwiseOpToHIVMBinary<arith::RemFOp, hivm::VModOp>,
      ElementwiseOpToHIVMBinary<arith::RemSIOp, hivm::VModOp>,
      ElementwiseOpToHIVMBinary<arith::RemUIOp, hivm::VModOp>,
      ElementwiseOpToHIVMBinary<arith::MinNumFOp, hivm::VMinOp>,
      ElementwiseOpToHIVMBinary<arith::MinimumFOp, hivm::VMinOp>,
      ElementwiseOpToHIVMBinary<arith::MaxNumFOp, hivm::VMaxOp>,
      ElementwiseOpToHIVMBinary<arith::MaximumFOp, hivm::VMaxOp>,
      ElementwiseOpToHIVMBinary<arith::ShLIOp, hivm::VShLOp>,
      ElementwiseOpToHIVMBinary<arith::ShRSIOp, hivm::VShROp>,
      ElementwiseOpToHIVMBinary<arith::ShRUIOp, hivm::VShROp>>(
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
  patterns.add<
      NPUVectorTransferReadToHIVM,
      NPUVectorTransferWriteToHIVM,
      NPUVectorBroadcastToHIVM,
      NPUVectorTransposeToHIVM,
      NPUVectorIndexCastToHIVM>(patterns.getContext());
  patterns.add<
      UnaryNPUVectorToHIVMCast<npuvector::ExtFOp>,
      UnaryNPUVectorToHIVMCast<npuvector::TruncFOp>,
      UnaryNPUVectorToHIVMCast<npuvector::ExtSIOp>,
      UnaryNPUVectorToHIVMCast<npuvector::ExtUIOp>,
      UnaryNPUVectorToHIVMCast<npuvector::TruncIOp>,
      UnaryNPUVectorToHIVMCast<npuvector::SIToFPOp>,
      UnaryNPUVectorToHIVMCast<npuvector::UIToFPOp>,
      UnaryNPUVectorToHIVMCast<npuvector::FPToSIOp>,
      UnaryNPUVectorToHIVMCast<npuvector::FPToUIOp>>(patterns.getContext());
  patterns.add<
      NPUVectorCmpToHIVM<npuvector::CmpFOp>,
      NPUVectorCmpToHIVM<npuvector::CmpIOp>>(patterns.getContext());
  patterns.add<VectorBroadcastToHIVM>(patterns.getContext());
  patterns.add<ScfForToHIVM>(patterns.getContext());
  patterns.add<ScfYieldToHIVM>(patterns.getContext());
}

namespace {
static bool isVectorOrNPUVectorType(Type type) {
  return isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type);
}

static bool isLegalArithOp(Operation *op) {
  return !std::any_of(op->getResultTypes().begin(), op->getResultTypes().end(), isVectorOrNPUVectorType);
}

static bool isLegalMathOp(Operation *op) {
  return !std::any_of(op->getResultTypes().begin(), op->getResultTypes().end(),
                      isVectorOrNPUVectorType);
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

struct ArithToHIVMConversionPass
    : public impl::ConvertArithToHIVMBase<ArithToHIVMConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, tensor::TensorDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    arith::ArithDialect, math::MathDialect, scf::SCFDialect,
                    annotation::AnnotationDialect>();
  }
  void runOnOperation() override;
};

void ArithToHIVMConversionPass::runOnOperation() {
  ConversionTarget target(getContext());
  // HIVM and Tensor are legal
  target.addLegalDialect<hivm::HIVMDialect, tensor::TensorDialect,
                          memref::MemRefDialect, scf::SCFDialect, BuiltinDialect,
                          annotation::AnnotationDialect>();
  target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalArithOp);
  target.addDynamicallyLegalDialect<math::MathDialect>(isLegalMathOp);
  target.addDynamicallyLegalOp<scf::ForOp>(isLegalSCFForOp);
  target.addDynamicallyLegalOp<scf::YieldOp>(isLegalSCFYieldOp);
  target.addIllegalOp<vector::ReductionOp, vector::TransferReadOp,
                          vector::TransferWriteOp, vector::BroadcastOp>();
  target.addIllegalOp<npuvector::ReductionOp, npuvector::TransferReadOp,
                          npuvector::TransferWriteOp, npuvector::BroadcastOp,
                          npuvector::TransposeOp,
                          npuvector::ExtFOp, npuvector::TruncFOp,
                          npuvector::ExtSIOp, npuvector::ExtUIOp,
                          npuvector::TruncIOp, npuvector::SIToFPOp,
                          npuvector::UIToFPOp, npuvector::FPToSIOp,
                          npuvector::FPToUIOp, npuvector::BitcastOp,
                          npuvector::CmpIOp, npuvector::CmpFOp,
                          npuvector::SelectOp>();

  RewritePatternSet patterns(&getContext());
  hivm::populateArithToHIVMConversionPatterns(patterns);
  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

std::unique_ptr<Pass> createArithToHIVMConversionPass() {
  return std::make_unique<ArithToHIVMConversionPass>();
}

}  // namespace mlir
