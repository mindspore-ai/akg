/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
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

        if (srcType.isIndex() || destType.isIndex()) return;

        unsigned srcWidth = srcType.getIntOrFloatBitWidth();
        unsigned destWidth = destType.getIntOrFloatBitWidth();

        if (srcWidth == 0) return;

        int64_t oldSize = attr.getInt();
        int64_t newSize = (oldSize * destWidth) / srcWidth;

        auto newMarkOp = rewriter.create<annotation::MarkOp>(loc, dest);
        newMarkOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(newSize));
        break;
      }
    }
  }
}

template <typename ArithOp, typename HIVMOp>
struct BinaryArithToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  LogicalResult matchAndRewrite(ArithOp op,
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

    Value lhsMemRef = adaptor.getLhs();
    Value rhsMemRef = adaptor.getRhs();

    auto memRefType = MemRefType::get(shape, elemType);

    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        Value dim = rewriter.create<memref::DimOp>(loc, lhsMemRef, i);
        allocOperands.push_back(dim);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    propagateBufferSizeMark(rewriter, loc, lhsMemRef, resBuf);

    rewriter.create<HIVMOp>(
        loc,
        TypeRange{},
        ValueRange{lhsMemRef, rhsMemRef},
        ValueRange{resBuf});

    rewriter.replaceOp(op, resBuf);
    return success();
  }
};

inline bool isOverFlowMode(Type inType, Type outType) {
  const bool isF32ToI16 = inType.isF32() && outType.isInteger(16);
  const bool isF32ToI8 = inType.isF32() && outType.isInteger(8);
  const bool isF16ToI8 = inType.isF16() && outType.isInteger(8);
  const bool isI16ToI8 = inType.isInteger(16) && outType.isInteger(8);
  const bool isI32ToI8 = inType.isInteger(32) && outType.isInteger(8);
  const bool isI32ToI16 = inType.isInteger(32) && outType.isInteger(16);
  const bool isI64ToI8 = inType.isInteger(64) && outType.isInteger(8);
  const bool isI64ToI16 = inType.isInteger(64) && outType.isInteger(16);
  const bool isI64ToI32 = inType.isInteger(64) && outType.isInteger(32);
  return (isI16ToI8 || isI32ToI16 || isI32ToI8 || isF32ToI16 || isF32ToI8 ||
          isF16ToI8 || isI64ToI8 || isI64ToI16 || isI64ToI32);
}

template <typename CastOp>
struct UnaryArithToHIVMCast : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<CastOp>::OpAdaptor;

  static hivm::RoundMode selectRoundModeForTruncF(Type inType, Type outType) {
    if (inType.isF32() && (outType.isF16() || outType.isBF16() || outType.isF32()))
      return hivm::RoundMode::RINT;
    llvm_unreachable("unsupported datatype for arith::TruncFOp to hivm");
  }

  static hivm::RoundMode selectRoundModeForExtF(Type inType, Type outType) {
    if ((inType.isF16() || inType.isBF16()) && outType.isF32())
      return hivm::RoundMode::RINT;
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

    auto memRefType = MemRefType::get(shape, elemType);
    SmallVector<Value> allocOperands;
    for (int i = 0; i < memRefType.getRank(); ++i) {
      if (memRefType.isDynamicDim(i)) {
        Value dim = rewriter.create<memref::DimOp>(loc, srcMemRef, i);
        allocOperands.push_back(dim);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    propagateBufferSizeMark(rewriter, loc, srcMemRef, resBuf);

    hivm::RoundMode rounding = selectRoundMode(op);
    auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(rounding);

    rewriter.create<hivm::VCastOp>(
        loc,
        TypeRange{},
        srcMemRef,
        resBuf,
        roundingAttr);

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
        Value dim = rewriter.create<memref::DimOp>(loc, lhs, i);
        allocOperands.push_back(dim);
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
    if (auto lowVectorType = dyn_cast<VectorType>(lowType)) {
      lowShape = lowVectorType.getShape();
      lowElemType = lowVectorType.getElementType();
    } else if (auto lowNpuVectorType = dyn_cast<npuvector::NPUVectorType>(lowType)) {
      lowShape = lowNpuVectorType.getShape();
      lowElemType = lowNpuVectorType.getElementType();
    } else {
      return failure();
    }

    ArrayRef<int64_t> highShape;
    Type highElemType;
    if (auto highVectorType = dyn_cast<VectorType>(highType)) {
      highShape = highVectorType.getShape();
      highElemType = highVectorType.getElementType();
    } else if (auto highNpuVectorType = dyn_cast<npuvector::NPUVectorType>(highType)) {
      highShape = highNpuVectorType.getShape();
      highElemType = highNpuVectorType.getElementType();
    } else {
      return failure();
    }

    auto lowMemRefType = MemRefType::get(lowShape, lowElemType);
    auto highMemRefType = MemRefType::get(highShape, highElemType);

    SmallVector<Value> lowAllocOperands;
    for (int i = 0; i < lowMemRefType.getRank(); ++i) {
      if (lowMemRefType.isDynamicDim(i)) {
        Value dim = rewriter.create<memref::DimOp>(loc, lhs, i);
        lowAllocOperands.push_back(dim);
      }
    }
    Value lowBuf = rewriter.create<memref::AllocOp>(loc, lowMemRefType, lowAllocOperands);
    propagateBufferSizeMark(rewriter, loc, lhs, lowBuf);

    SmallVector<Value> highAllocOperands;
    for (int i = 0; i < highMemRefType.getRank(); ++i) {
      if (highMemRefType.isDynamicDim(i)) {
        Value dim = rewriter.create<memref::DimOp>(loc, lhs, i);
        highAllocOperands.push_back(dim);
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

  LogicalResult matchAndRewrite(ArithOp op,
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
        Value dim = rewriter.create<memref::DimOp>(loc, lhs, i);
        allocOperands.push_back(dim);
      }
    }
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType, allocOperands);
    propagateBufferSizeMark(rewriter, loc, lhs, resBuf);

    if constexpr (std::is_same_v<HIVMOp, hivm::VShROp>) {
      rewriter.create<HIVMOp>(
          loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf},
          rewriter.getBoolAttr(true));
    } else {
      rewriter.create<HIVMOp>(
          loc, TypeRange{}, ValueRange{lhs, rhs}, ValueRange{resBuf});
    }

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
        Value dim = rewriter.create<memref::DimOp>(loc, cond, i);
        allocOperands.push_back(dim);
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
    if (!denseAttr)
      return failure();

    Location loc = op.getLoc();
    TypedAttr typedScalarAttr = denseAttr.getSplatValue<TypedAttr>();
    if (!typedScalarAttr)
      return failure();
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
        reduceKind = hivm::ReduceOperation::min;
        return success();
      case vector::CombiningKind::MAXUI:
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXNUMF:
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
    for (int64_t i = 0; i < rank; ++i) reduceDims.push_back(i);

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


struct AffineForToHIVM : public OpConversionPattern<affine::AffineForOp> {
  using OpConversionPattern<affine::AffineForOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineForOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Only handle loops that have at least one vector-typed iteration argument.
    if (llvm::none_of(op.getResultTypes(), [](Type t) {
          return isa<VectorType>(t) || isa<npuvector::NPUVectorType>(t);
        })) {
      return failure();
    }

    auto lbOperands = adaptor.getLowerBoundOperands();
    auto ubOperands = adaptor.getUpperBoundOperands();
    auto allOperands = adaptor.getOperands();
    auto iterOperands = allOperands.drop_front(lbOperands.size() + ubOperands.size());

    SmallVector<Value> newIterOperands(iterOperands.begin(), iterOperands.end());

    auto newForOp = rewriter.create<affine::AffineForOp>(
        op.getLoc(), lbOperands, op.getLowerBoundMap(),
        ubOperands, op.getUpperBoundMap(), op.getStep().getSExtValue(), newIterOperands);

    Block &oldBlock = op.getRegion().front();
    Block &newBlock = newForOp.getRegion().front();

    SmallVector<Value> newBlockArgs(newBlock.getArguments().begin(), newBlock.getArguments().end());

    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockArgs);

    rewriter.replaceOp(op, newForOp.getResults());

    return success();
  }
};


struct AffineYieldToHIVM : public OpConversionPattern<affine::AffineYieldOp> {
  using OpConversionPattern<affine::AffineYieldOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineYieldOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (llvm::none_of(op.getOperands(), [](Value v) {
          return isa<VectorType>(v.getType()) || isa<npuvector::NPUVectorType>(v.getType());
        })) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<affine::AffineYieldOp>(op, adaptor.getOperands());
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


struct AffineStoreToHIVM : public OpConversionPattern<affine::AffineStoreOp> {
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(affine::AffineStoreOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto operands = adaptor.getOperands();
    Value valToStore = operands[0];
    Value memref = operands[1];

    auto memRefType = cast<MemRefType>(memref.getType());
    int64_t rank = memRefType.getRank();

    Value storeValueMemref = valToStore;
    if (!isa<MemRefType>(valToStore.getType())) {
      return failure();
    }

    ValueRange mapOperands = operands.drop_front(2);

    auto map = op.getAffineMap();
    SmallVector<OpFoldResult> offsets;
    offsets.reserve(map.getNumResults());
    for (unsigned i = 0; i < map.getNumResults(); ++i) {
      AffineExpr expr = map.getResult(i);
      if (auto cstExpr = dyn_cast<AffineConstantExpr>(expr)) {
        offsets.push_back(rewriter.getIndexAttr(cstExpr.getValue()));
        continue;
      }
      if (auto dimExpr = dyn_cast<AffineDimExpr>(expr)) {
        offsets.push_back(mapOperands[dimExpr.getPosition()]);
        continue;
      }
      if (auto symExpr = dyn_cast<AffineSymbolExpr>(expr)) {
        offsets.push_back(mapOperands[map.getNumDims() + symExpr.getPosition()]);
        continue;
      }
      auto apply = rewriter.create<affine::AffineApplyOp>(loc, map.getSubMap({i}), mapOperands);
      offsets.push_back(apply.getResult());
    }

    SmallVector<OpFoldResult> sizes(rank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    SmallVector<int64_t> resultShape(rank, 1);
    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        resultShape, memRefType, offsets, sizes, strides);
    Value subView = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), memref, offsets, sizes, strides);

    rewriter.create<hivm::StoreOp>(loc, TypeRange{}, storeValueMemref, subView);
    rewriter.eraseOp(op);
    return success();
  }
};

struct MemRefStoreToHIVM : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(memref::StoreOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
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
        reduceKind = hivm::ReduceOperation::min;
        return success();
      case vector::CombiningKind::MAXUI:
      case vector::CombiningKind::MAXSI:
      case vector::CombiningKind::MAXNUMF:
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
    for (int64_t i = 0; i < rank; ++i) reduceDims.push_back(i);

    auto reduceOpAttr = hivm::ReduceOpAttr::get(op.getContext(), reduceKind);
    rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{}, sourceMemRef, resultBuf, Value(),
        reduceOpAttr, rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    rewriter.replaceOp(op, resultBuf);

    return success();
  }
};

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
    if (!npuVecType) return failure();

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(adaptor.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto memRefType = cast<MemRefType>(source.getType());
    int64_t memRefRank = memRefType.getRank();
    int64_t vecRank = npuVecType.getRank();

    // Handle leading dimensions for rank reduction.
    for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    auto dynamicSizes = adaptor.getDynamicSizes();
    size_t dynamicSizeIdx = 0;

    // Set sizes to match vector shape.
    for (int64_t i = 0; i < vecRank; ++i) {
      if (npuVecType.isDynamicDim(i)) {
         if (dynamicSizeIdx < dynamicSizes.size()) {
             sizes.push_back(dynamicSizes[dynamicSizeIdx++]);
         } else {
             return failure();
         }
      } else {
         sizes.push_back(rewriter.getIndexAttr(npuVecType.getDimSize(i)));
      }
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Infer result type for the rank-reduced subview.
    auto subViewResultType = memref::SubViewOp::inferRankReducedResultType(
        npuVecType.getShape(), memRefType, offsets, sizes, strides);

    Value finalSource = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(subViewResultType), source, offsets, sizes, strides);

    Type elemType = npuVecType.getElementType();
    auto targetMemRefType = MemRefType::get(npuVecType.getShape(), elemType);

    // Prepare allocation operands for dynamic shapes.
    SmallVector<Value> allocOperands(dynamicSizes.size());
    std::copy(dynamicSizes.begin(), dynamicSizes.end(), allocOperands.begin());
    Value tempBuf = rewriter.create<memref::AllocOp>(loc, targetMemRefType, allocOperands);

    Value maxSize = adaptor.getMaxSize();
    if (maxSize && !npuVecType.hasStaticShape()) {
      if (auto constOp = maxSize.getDefiningOp<arith::ConstantOp>()) {
        auto markOp = rewriter.create<annotation::MarkOp>(loc, tempBuf);
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          auto byteWidth = elemType.getIntOrFloatBitWidth() / 8;
          markOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(intAttr.getInt() * byteWidth));
        }
      } else {
        return failure();
      }
    }

    rewriter.create<hivm::LoadOp>(
        loc,
        TypeRange{},
        finalSource,
        tempBuf);

    rewriter.replaceOp(op, tempBuf);
    return success();
  }
};

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

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(adaptor.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    int64_t memRefRank = destMemRefType.getRank();
    int64_t dataRank = dataMemRefType.getRank();

    // Handle leading dimensions for rank reduction.
    for (int64_t i = 0; i < memRefRank - dataRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Set sizes to match vector shape.
    for (int64_t i = 0; i < dataRank; ++i) {
      if (dataMemRefType.isDynamicDim(i)) {
          Value dim = rewriter.create<memref::DimOp>(loc, dataToWrite, i);
          sizes.push_back(dim);
      } else {
          sizes.push_back(rewriter.getIndexAttr(dataMemRefType.getDimSize(i)));
      }
      strides.push_back(rewriter.getIndexAttr(1));
    }

    // Infer result type for the rank-reduced subview.
    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        dataMemRefType.getShape(), destMemRefType, offsets, sizes, strides);

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
    auto memRefType = MemRefType::get(npuVecType.getShape(), elemType);

    Value resultBuf = rewriter.create<memref::AllocOp>(loc, memRefType, op.getDynamicSizes());

    // Similarly, use op.getMaxSize() instead of adaptor.getMaxSize()
    Value maxSize = op.getMaxSize();
    if (maxSize && !npuVecType.hasStaticShape()) {
      if (auto constOp = maxSize.getDefiningOp<arith::ConstantOp>()) {
        auto markOp = rewriter.create<annotation::MarkOp>(loc, resultBuf);
        if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
          auto byteWidth = elemType.getIntOrFloatBitWidth() / 8;
          markOp->setAttr(kBufferSizeInByteAttr, rewriter.getIndexAttr(intAttr.getInt() * byteWidth));
        }
      }
    }

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
  patterns.add<ArithSelectToHIVM<arith::SelectOp>>(patterns.getContext());
  patterns.add<ArithConstantToHIVM>(patterns.getContext());
  patterns.add<VectorReductionToHIVM>(patterns.getContext());
  patterns.add<NPUVectorReductionToHIVM>(patterns.getContext());
  patterns.add<
      NPUVectorTransferReadToHIVM,
      NPUVectorTransferWriteToHIVM,
      NPUVectorBroadcastToHIVM>(patterns.getContext());
  patterns.add<VectorBroadcastToHIVM>(patterns.getContext());
  patterns.add<AffineForToHIVM>(patterns.getContext());
  patterns.add<AffineYieldToHIVM>(patterns.getContext());
  patterns.add<AffineStoreToHIVM>(patterns.getContext());
  patterns.add<MemRefStoreToHIVM>(patterns.getContext());
  patterns.add<ScfForToHIVM>(patterns.getContext());
  patterns.add<ScfYieldToHIVM>(patterns.getContext());
}

namespace {
static bool isVectorOrNPUVectorType(Type type) {
  return isa<VectorType>(type) || isa<npuvector::NPUVectorType>(type);
}

static bool isLegalArithOp(Operation *op) {
  if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
    return !isVectorOrNPUVectorType(constantOp.getType());
  }
  return !std::any_of(op->getResultTypes().begin(), op->getResultTypes().end(), isVectorOrNPUVectorType);
}

static bool isLegalAffineForOp(affine::AffineForOp op) {
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

static bool isLegalAffineYieldOp(affine::AffineYieldOp op) {
  for (auto operand : op.getOperands()) {
    if (isVectorOrNPUVectorType(operand.getType())) {
      return false;
    }
  }
  return true;
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

static bool isLegalMemRefStoreOp(memref::StoreOp op) {
  if (auto defOp = op.getValue().getDefiningOp()) {
    if (isa<vector::ReductionOp, vector::TransferReadOp, vector::BroadcastOp,
            npuvector::ReductionOp, npuvector::TransferReadOp, npuvector::BroadcastOp,
            arith::MulSIExtendedOp, arith::MulUIExtendedOp>(defOp)) {
      return false;
    }
  }
  return !isa<MemRefType>(op.getValue().getType());
}

struct ArithToHIVMConversionPass
    : public impl::ConvertArithToHIVMBase<ArithToHIVMConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, tensor::TensorDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    arith::ArithDialect, affine::AffineDialect, scf::SCFDialect,
                    annotation::AnnotationDialect>();
  }
  void runOnOperation() override;
};

void ArithToHIVMConversionPass::runOnOperation() {
  ConversionTarget target(getContext());
  // HIVM and Tensor are legal
  target.addLegalDialect<hivm::HIVMDialect, tensor::TensorDialect,
                          memref::MemRefDialect, affine::AffineDialect, scf::SCFDialect, BuiltinDialect,
                          annotation::AnnotationDialect>();
  target.addDynamicallyLegalDialect<arith::ArithDialect>(isLegalArithOp);
  target.addDynamicallyLegalOp<affine::AffineForOp>(isLegalAffineForOp);
  target.addDynamicallyLegalOp<affine::AffineYieldOp>(isLegalAffineYieldOp);
  target.addDynamicallyLegalOp<scf::ForOp>(isLegalSCFForOp);
  target.addDynamicallyLegalOp<scf::YieldOp>(isLegalSCFYieldOp);
  target.addDynamicallyLegalOp<memref::StoreOp>(isLegalMemRefStoreOp);
  target.addIllegalOp<affine::AffineStoreOp>();
  target.addIllegalOp<vector::ReductionOp, vector::TransferReadOp,
                          vector::TransferWriteOp, vector::BroadcastOp>();
  target.addIllegalOp<npuvector::ReductionOp, npuvector::TransferReadOp,
                          npuvector::TransferWriteOp, npuvector::BroadcastOp>();

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
