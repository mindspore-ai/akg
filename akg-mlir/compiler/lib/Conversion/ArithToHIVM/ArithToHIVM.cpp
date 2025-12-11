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

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
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
template <typename ArithOp, typename HIVMOp>
struct BinaryArithToHIVM : public OpConversionPattern<ArithOp> {
  using OpConversionPattern<ArithOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<ArithOp>::OpAdaptor;

  LogicalResult matchAndRewrite(ArithOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Type resType = op.getResult().getType();

    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType) {
      return failure();
    }

    Value lhsMemRef = adaptor.getLhs();
    Value rhsMemRef = adaptor.getRhs();

    Type elemType = vectorType.getElementType();
    if (elemType.isInteger(1)) {
        elemType = rewriter.getIntegerType(8);
    }
    auto memRefType = MemRefType::get(vectorType.getShape(), elemType);

    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

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

    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType) {
      return failure();
    }

    Value srcMemRef = adaptor.getOperands()[0];

    Type elemType = vectorType.getElementType();
    if (elemType.isInteger(1)) {
        elemType = rewriter.getIntegerType(8);
    }
    auto memRefType = MemRefType::get(vectorType.getShape(), elemType);
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

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

    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType) {
      return failure();
    }

    Value srcMemRef = adaptor.getIn();

    auto memRefType = MemRefType::get(vectorType.getShape(), vectorType.getElementType());

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
    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type elemType = vectorType.getElementType();
    if (elemType.isInteger(1)) {
        elemType = rewriter.getIntegerType(8);
    }
    auto memRefType = MemRefType::get(vectorType.getShape(), elemType);
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

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

    auto lowVectorType = dyn_cast<VectorType>(lowType);
    auto highVectorType = dyn_cast<VectorType>(highType);

    if (!lowVectorType || !highVectorType)
      return failure();

    auto lowMemRefType = MemRefType::get(lowVectorType.getShape(), lowVectorType.getElementType());
    auto highMemRefType = MemRefType::get(highVectorType.getShape(), highVectorType.getElementType());

    Value lowBuf = rewriter.create<memref::AllocOp>(loc, lowMemRefType);
    Value highBuf = rewriter.create<memref::AllocOp>(loc, highMemRefType);

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
    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType)
      return failure();

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    Type elemType = vectorType.getElementType();
    if (elemType.isInteger(1)) {
        elemType = rewriter.getIntegerType(8);
    }
    auto memRefType = MemRefType::get(vectorType.getShape(), elemType);
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

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
    auto vectorType = dyn_cast<VectorType>(resType);
    if (!vectorType)
      return failure();

    Value cond = adaptor.getCondition();
    Value trueVal = adaptor.getTrueValue();
    Value falseVal = adaptor.getFalseValue();

    Type elemType = vectorType.getElementType();
    if (elemType.isInteger(1)) {
        elemType = rewriter.getIntegerType(8);
    }
    auto memRefType = MemRefType::get(vectorType.getShape(), elemType);
    Value resBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

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
    auto vectorType = dyn_cast<VectorType>(op.getType());
    if (!vectorType)
      return failure();

    auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue());
    if (!denseAttr)
      return failure();

    Location loc = op.getLoc();
    TypedAttr typedScalarAttr = denseAttr.getSplatValue<TypedAttr>();
    if (!typedScalarAttr)
      return failure();
    Value scalarConstant = rewriter.create<arith::ConstantOp>(loc, typedScalarAttr);

    auto memRefType = MemRefType::get(vectorType.getShape(), vectorType.getElementType());
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

  static Value getInitValue(vector::ReductionOp op, Type elemType,
                            vector::CombiningKind kind,
                            ConversionPatternRewriter &rewriter, Location loc) {
    if (op.getNumOperands() == 2) {
      return op.getOperand(1);
    }

    TypedAttr identityAttr;
    if (isa<FloatType>(elemType)) {
      double val = (kind == vector::CombiningKind::MUL) ? 1.0 : 0.0;
      identityAttr = rewriter.getFloatAttr(elemType, val);
    } else {
      int64_t val = 0;
      if (kind == vector::CombiningKind::MUL) val = 1;
      if (kind == vector::CombiningKind::AND) val = -1;
      identityAttr = rewriter.getIntegerAttr(elemType, val);
    }
    return rewriter.create<arith::ConstantOp>(loc, identityAttr);
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

    Value initVal = getInitValue(op, elemType, kind, rewriter, loc);

    SmallVector<Value> zeros;
    for (int i = 0; i < rank; ++i) zeros.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 0));
    rewriter.create<memref::StoreOp>(loc, initVal, resultBuf, zeros);

    SmallVector<int64_t> reduceDims;
    for (int64_t i = 0; i < rank; ++i) reduceDims.push_back(i);

    auto reduceOpAttr = hivm::ReduceOpAttr::get(op.getContext(), reduceKind);
    rewriter.create<hivm::VReduceOp>(
        loc, TypeRange{}, sourceMemRef, resultBuf, Value(),
        reduceOpAttr, rewriter.getDenseI64ArrayAttr(reduceDims), Value());

    Value scalarResult = rewriter.create<memref::LoadOp>(loc, resultBuf, zeros);
    rewriter.replaceOp(op, scalarResult);

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

    if (isa<VectorType>(source.getType()) || isa<MemRefType>(source.getType())) {
       return rewriter.notifyMatchFailure(op, "only scalar broadcast supported");
    }

    auto vecType = op.getVector().getType();
    auto memRefType = MemRefType::get(vecType.getShape(), vecType.getElementType());

    Value resultBuf = rewriter.create<memref::AllocOp>(loc, memRefType);

    rewriter.create<hivm::VBrcOp>(
        loc,
        TypeRange{},  // void return
        source,       // Scalar Input
        resultBuf,    // Output MemRef
        rewriter.getDenseI64ArrayAttr({}));

    rewriter.replaceOp(op, resultBuf);

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

    for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    for (auto dim : vecType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        vecType.getShape(), memRefType, offsets, sizes, strides);

    Value slicedSource = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), source, offsets, sizes, strides);

    Value finalSource = slicedSource;
    Type elemType = vecType.getElementType();

    if (elemType.isInteger(1)) {
       elemType = rewriter.getIntegerType(8);

       auto i8MemRefType = MemRefType::get(vecType.getShape(), elemType);

       finalSource = rewriter.create<hivm::BitcastOp>(loc, i8MemRefType, slicedSource);
    }

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

    auto vecType = op.getVectorType();

    SmallVector<OpFoldResult> offsets = getAsOpFoldResult(op.getIndices());
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides;

    auto memRefType = cast<MemRefType>(dest.getType());
    int64_t memRefRank = memRefType.getRank();
    int64_t vecRank = vecType.getRank();

    for (int64_t i = 0; i < memRefRank - vecRank; ++i) {
      sizes.push_back(rewriter.getIndexAttr(1));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    for (auto dim : vecType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(dim));
      strides.push_back(rewriter.getIndexAttr(1));
    }

    auto resultType = memref::SubViewOp::inferRankReducedResultType(
        vecType.getShape(), memRefType, offsets, sizes, strides);

    Value slicedDest = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(resultType), dest, offsets, sizes, strides);

    Value finalData = dataToWrite;
    Value finalDest = slicedDest;

    auto destMemRefType = cast<MemRefType>(slicedDest.getType());
    auto dataMemRefType = cast<MemRefType>(dataToWrite.getType());

    if (destMemRefType.getElementType().isInteger(1)) {
      auto i8Type = rewriter.getIntegerType(8);
      auto i8MemRefType = MemRefType::get(destMemRefType.getShape(), i8Type);

      finalDest = rewriter.create<hivm::BitcastOp>(loc, i8MemRefType, finalDest);

      if (dataMemRefType.getElementType().isInteger(1)) {
        Value dataI8Buf = rewriter.create<memref::AllocOp>(loc, i8MemRefType);
        auto roundingAttr = rewriter.getAttr<hivm::RoundModeAttr>(hivm::RoundMode::RINT);
        rewriter.create<hivm::VCastOp>(
            loc, TypeRange{}, finalData, dataI8Buf, roundingAttr);
        finalData = dataI8Buf;
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
  patterns.add<VectorBroadcastToHIVM>(patterns.getContext());
}

namespace {
struct ArithToHIVMConversionPass
    : public impl::ConvertArithToHIVMBase<ArithToHIVMConversionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, tensor::TensorDialect,
                    memref::MemRefDialect, vector::VectorDialect,
                    arith::ArithDialect>();
  }
  void runOnOperation() override;
};

void ArithToHIVMConversionPass::runOnOperation() {
  ConversionTarget target(getContext());
  // HIVM and Tensor are legal
  target.addLegalDialect<hivm::HIVMDialect, tensor::TensorDialect,
                          memref::MemRefDialect, BuiltinDialect>();
  target.addDynamicallyLegalDialect<arith::ArithDialect>([](Operation *op) {
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      return !isa<RankedTensorType>(constantOp.getType()) &&
             !isa<VectorType>(constantOp.getType());
    }
    for (Type type : op->getResultTypes()) {
      if (isa<VectorType>(type))
        return false;
    }
    return true;
  });
  target.addIllegalOp<vector::ReductionOp, vector::TransferReadOp, vector::TransferWriteOp, vector::BroadcastOp>();

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
