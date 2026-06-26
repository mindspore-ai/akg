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
//===- ArithToLinalg.cpp - conversion from Arith to Linalg dialect --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "akg/Conversion/ArithToLinalg/ArithToLinalg.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTARITHTOLINALG
#include "akg/Conversion/Passes.h.inc"
}  // namespace mlir

namespace {
using mlir::applyPartialConversion;
using mlir::ArrayRef;
using mlir::ConversionTarget;
using mlir::DenseIntOrFPElementsAttr;
using mlir::dyn_cast;
using mlir::failure;
using mlir::isa;
using mlir::LogicalResult;
using mlir::Operation;
using mlir::OpRewritePattern;
using mlir::PatternRewriter;
using mlir::RankedTensorType;
using mlir::RewritePatternSet;
using mlir::SmallVector;
using mlir::success;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;
static bool operateOnTensors(Operation *op) {
  return llvm::all_of(op->getOperandTypes(), [](Type type) { return isa<RankedTensorType>(type); });
}

template <typename UnaryOp, mlir::linalg::UnaryFn linalgFn>
struct ElementwiseOpToLinalgUnary : OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp op, PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op)) {
      return failure();
    }
    Value inner = op.getOperand();
    SmallVector<Value> dsts;
    if (failed(mlir::tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts))) {
      return failure();
    }
    auto unaryAttr = rewriter.getAttr<mlir::linalg::UnaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
    rewriter.replaceOpWithNewOp<mlir::linalg::ElemwiseUnaryOp>(op, ValueRange{inner}, ValueRange{dsts},
                                                               ArrayRef{fnAttr});
    return success();
  }
};

template <typename BinaryOp, mlir::linalg::BinaryFn linalgFn>
struct ElementwiseOpToLinalgBinary : OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op, PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op)) {
      return failure();
    }
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    SmallVector<Value> dsts;
    if (failed(mlir::tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts))) {
      return failure();
    }
    auto binaryAttr = rewriter.getAttr<mlir::linalg::BinaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
    rewriter.replaceOpWithNewOp<mlir::linalg::ElemwiseBinaryOp>(op, ValueRange{lhs, rhs}, ValueRange{dsts},
                                                                ArrayRef{fnAttr});
    return success();
  }
};
}  // namespace

/// @brief
// Three kinds of conversions are applied:
// 1. arith ops to linalg unary/binary ops
// 2. math ops to linalg unary/binary ops
// 3. linalg.map op to linalg unary/binary ops
void mlir::arith::populateArithToLinalgConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ElementwiseOpToLinalgBinary<mlir::arith::AddFOp, mlir::linalg::BinaryFn::add>,
               ElementwiseOpToLinalgBinary<mlir::arith::AddIOp, mlir::linalg::BinaryFn::add>,
               ElementwiseOpToLinalgBinary<mlir::arith::SubFOp, mlir::linalg::BinaryFn::sub>,
               ElementwiseOpToLinalgBinary<mlir::arith::SubIOp, mlir::linalg::BinaryFn::sub>,
               ElementwiseOpToLinalgBinary<mlir::arith::MulFOp, mlir::linalg::BinaryFn::mul>,
               ElementwiseOpToLinalgBinary<mlir::arith::MulIOp, mlir::linalg::BinaryFn::mul>,
               ElementwiseOpToLinalgBinary<mlir::arith::DivFOp, mlir::linalg::BinaryFn::div>,
               ElementwiseOpToLinalgBinary<mlir::arith::DivSIOp, mlir::linalg::BinaryFn::div>,
               ElementwiseOpToLinalgBinary<mlir::arith::DivUIOp, mlir::linalg::BinaryFn::div_unsigned>,
               ElementwiseOpToLinalgBinary<mlir::arith::MaxSIOp, mlir::linalg::BinaryFn::max_signed>,
               ElementwiseOpToLinalgBinary<mlir::arith::MaxUIOp, mlir::linalg::BinaryFn::max_unsigned>,
               ElementwiseOpToLinalgBinary<mlir::arith::MinSIOp, mlir::linalg::BinaryFn::min_signed>,
               ElementwiseOpToLinalgBinary<mlir::arith::MinUIOp, mlir::linalg::BinaryFn::min_unsigned>,
               ElementwiseOpToLinalgUnary<mlir::math::ExpOp, mlir::linalg::UnaryFn::exp>,
               ElementwiseOpToLinalgUnary<mlir::math::LogOp, mlir::linalg::UnaryFn::log>,
               ElementwiseOpToLinalgUnary<mlir::math::AbsFOp, mlir::linalg::UnaryFn::abs>,
               ElementwiseOpToLinalgUnary<mlir::math::CeilOp, mlir::linalg::UnaryFn::ceil>,
               ElementwiseOpToLinalgUnary<mlir::math::SqrtOp, mlir::linalg::UnaryFn::sqrt>,
               ElementwiseOpToLinalgUnary<mlir::math::FloorOp, mlir::linalg::UnaryFn::floor>>(patterns.getContext());
}

namespace {
struct ConvertArithToLinalgPass : public mlir::impl::ConvertArithToLinalgBase<ConvertArithToLinalgPass> {
  void runOnOperation() override;
};
}  // namespace

void ConvertArithToLinalgPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  // Elementwise arith Ops should be converted.
  target.addDynamicallyLegalDialect<mlir::arith::ArithDialect, mlir::math::MathDialect>([](Operation *op) {
    if (auto constantOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(constantOp.getValue());
      return !(denseAttr && denseAttr.isSplat());
    }
    return !operateOnTensors(op);
  });

  RewritePatternSet patterns(&getContext());
  mlir::arith::populateArithToLinalgConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

namespace mlir {
std::unique_ptr<Pass> createArithToLinalgConversionPass() { return std::make_unique<ConvertArithToLinalgPass>(); }
}  // namespace mlir
