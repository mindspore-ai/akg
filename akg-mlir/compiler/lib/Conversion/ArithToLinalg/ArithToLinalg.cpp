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
} // namespace mlir

using namespace mlir;

static bool operateOnTensors(Operation *op) {
  return llvm::all_of(op->getOperandTypes(),
                      [](Type type) { return isa<RankedTensorType>(type); });
}

template <typename UnaryOp, linalg::UnaryFn linalgFn>
struct ElementwiseOpToLinalgUnary : OpRewritePattern<UnaryOp> {
  using OpRewritePattern<UnaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(UnaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value inner = op.getOperand();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto unaryAttr = rewriter.getAttr<linalg::UnaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", unaryAttr);
    rewriter.replaceOpWithNewOp<linalg::ElemwiseUnaryOp>(
        op, ValueRange{inner}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

template <typename BinaryOp, linalg::BinaryFn linalgFn>
struct ElementwiseOpToLinalgBinary : OpRewritePattern<BinaryOp> {
  using OpRewritePattern<BinaryOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOp op,
                                PatternRewriter &rewriter) const final {
    if (!operateOnTensors(op))
      return failure();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    SmallVector<Value> dsts;
    if (failed(
            tensor::getOrCreateDestinations(rewriter, op.getLoc(), op, dsts)))
      return failure();
    auto binaryAttr = rewriter.getAttr<linalg::BinaryFnAttr>(linalgFn);
    auto fnAttr = rewriter.getNamedAttr("fun", binaryAttr);
    rewriter.replaceOpWithNewOp<linalg::ElemwiseBinaryOp>(
        op, ValueRange{lhs, rhs}, ValueRange{dsts}, ArrayRef{fnAttr});
    return success();
  }
};

/// @brief
// Three kinds of conversions are applied:
// 1. arith ops to linalg unary/binary ops
// 2. math ops to linalg unary/binary ops
// 3. linalg.map op to linalg unary/binary ops
void mlir::arith::populateArithToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      ElementwiseOpToLinalgBinary<arith::AddFOp, linalg::BinaryFn::add>,
      ElementwiseOpToLinalgBinary<arith::AddIOp, linalg::BinaryFn::add>,
      ElementwiseOpToLinalgBinary<arith::SubFOp, linalg::BinaryFn::sub>,
      ElementwiseOpToLinalgBinary<arith::SubIOp, linalg::BinaryFn::sub>,
      ElementwiseOpToLinalgBinary<arith::MulFOp, linalg::BinaryFn::mul>,
      ElementwiseOpToLinalgBinary<arith::MulIOp, linalg::BinaryFn::mul>,
      ElementwiseOpToLinalgBinary<arith::DivFOp, linalg::BinaryFn::div>,
      ElementwiseOpToLinalgBinary<arith::DivSIOp, linalg::BinaryFn::div>,
      ElementwiseOpToLinalgBinary<arith::DivUIOp,
                                  linalg::BinaryFn::div_unsigned>,
      ElementwiseOpToLinalgBinary<arith::MaxSIOp, linalg::BinaryFn::max_signed>,
      ElementwiseOpToLinalgBinary<arith::MaxUIOp,
                                  linalg::BinaryFn::max_unsigned>,
      ElementwiseOpToLinalgBinary<arith::MinSIOp, linalg::BinaryFn::min_signed>,
      ElementwiseOpToLinalgBinary<arith::MinUIOp,
                                  linalg::BinaryFn::min_unsigned>,
      ElementwiseOpToLinalgUnary<math::ExpOp, linalg::UnaryFn::exp>,
      ElementwiseOpToLinalgUnary<math::LogOp, linalg::UnaryFn::log>,
      ElementwiseOpToLinalgUnary<math::AbsFOp, linalg::UnaryFn::abs>,
      ElementwiseOpToLinalgUnary<math::CeilOp, linalg::UnaryFn::ceil>,
      ElementwiseOpToLinalgUnary<math::SqrtOp, linalg::UnaryFn::sqrt>,
      ElementwiseOpToLinalgUnary<math::FloorOp, linalg::UnaryFn::floor>>(
      patterns.getContext());
}

namespace {
struct ConvertArithToLinalgPass
    : public impl::ConvertArithToLinalgBase<ConvertArithToLinalgPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertArithToLinalgPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect>();
  // Elementwise arith Ops should be converted.
  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      [](Operation *op) {
        if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
          auto denseAttr =
              dyn_cast<DenseIntOrFPElementsAttr>(constantOp.getValue());
          if (denseAttr && denseAttr.isSplat())
            return false;
          return true;
        }
        return !operateOnTensors(op);
      });

  RewritePatternSet patterns(&getContext());
  mlir::arith::populateArithToLinalgConversionPatterns(patterns);
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<Pass> mlir::createArithToLinalgConversionPass() {
  return std::make_unique<ConvertArithToLinalgPass>();
}
