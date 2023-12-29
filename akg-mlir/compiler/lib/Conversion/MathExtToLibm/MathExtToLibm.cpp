// ===-- MathToLibm.cpp - conversion from Math to libm calls ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//  MathExtToLibm.cpp   based on MathToLibm.cpp
// ===----------------------------------------------------------------------===//
/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Conversion/MathExtToLibm/MathExtToLibm.h"

#include <optional>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Math/IR/MathExtOps.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/ConversionUtils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

using namespace mlir;
using namespace mlir::tosa;
using namespace mlir::mindspore;
using namespace mlir::LLVM;
using namespace mlir::mathExt;

namespace {
// Pattern to convert vector operations to scalar operations. This is needed as
// libm calls require scalars.
template <typename Op>
struct VecOpToScalarOp : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
// Pattern to promote an op of a smaller floating point type to F32.
template <typename Op>
struct PromoteOpToF32 : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;
};
// Pattern to convert scalar math operations to calls to libm functions.
// Additionally the libm function signatures are declared.
template <typename Op>
struct ScalarOpToLibmCall : public OpRewritePattern<Op> {
 public:
  using OpRewritePattern<Op>::OpRewritePattern;
  ScalarOpToLibmCall<Op>(MLIRContext *context, const StringRef floatFunc, const StringRef doubleFunc,
                         const PatternBenefit benefit)
      : OpRewritePattern<Op>(context, benefit), floatFunc(floatFunc), doubleFunc(doubleFunc){};

  LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const final;

 private:
  std::string floatFunc, doubleFunc;
};
}  // namespace

template <typename Op>
LogicalResult VecOpToScalarOp<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto opType = op.getType();
  auto loc = op.getLoc();
  auto vecType = opType.template dyn_cast<VectorType>();
  if (!vecType) {
    return failure();
  }
  if (!vecType.hasRank()) {
    return failure();
  }

  auto shape = vecType.getShape();
  int64_t numElements = vecType.getNumElements();
  Value result = rewriter.create<arith::ConstantOp>(
    loc, DenseElementsAttr::get(vecType, FloatAttr::get(vecType.getElementType(), 0.0)));
  SmallVector<int64_t> strides = computeStrides(shape);
  for (auto linearIndex = 0; linearIndex < numElements; ++linearIndex) {
    SmallVector<int64_t> positions = delinearize(strides, linearIndex);
    SmallVector<Value> operands;
    for (auto input : op->getOperands()) {
      operands.push_back(rewriter.create<vector::ExtractOp>(loc, input, positions));
    }
    Value scalarOp = rewriter.create<Op>(loc, vecType.getElementType(), operands);
    result = rewriter.create<vector::InsertOp>(loc, scalarOp, result, positions);
  }
  rewriter.replaceOp(op, {result});
  return success();
}

template <typename Op>
LogicalResult PromoteOpToF32<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto opType = op.getType();
  if (!opType.template isa<Float16Type, BFloat16Type>()) {
    return failure();
  }

  auto loc = op.getLoc();
  auto f32 = rewriter.getF32Type();
  auto extendedOperands = llvm::to_vector(llvm::map_range(op->getOperands(), [&](const Value operand) -> Value {
    return rewriter.create<arith::ExtFOp>(loc, f32, operand);
  }));
  auto newOp = rewriter.create<Op>(loc, f32, extendedOperands);
  (void)rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, opType, newOp);
  return success();
}

template <typename Op>
LogicalResult ScalarOpToLibmCall<Op>::matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  auto module = SymbolTable::getNearestSymbolTable(op);
  auto type = op.getType();
  if (!type.template isa<Float32Type, Float64Type>()) {
    return failure();
  }

  auto name = type.getIntOrFloatBitWidth() == 64 ? doubleFunc : floatFunc;
  auto opFunc = dyn_cast_or_null<SymbolOpInterface>(SymbolTable::lookupSymbolIn(module, name));
  // Forward declare function if it hasn't already been
  if (!opFunc) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&module->getRegion(0).front());
    auto opFunctionTy = FunctionType::get(rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
    opFunc = rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), name, opFunctionTy);
    opFunc.setPrivate();

    // By definition Math dialect operations imply LLVM's "readnone"
    // function attribute, so we can set it here to provide more
    // optimization opportunities (e.g. LICM) for backends targeting LLVM IR.
    // This will have to be changed, when strict FP behavior is supported
    // by Math dialect.
    opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(), UnitAttr::get(rewriter.getContext()));
  }
  assert(isa<FunctionOpInterface>(SymbolTable::lookupSymbolIn(module, name)));

  (void)rewriter.replaceOpWithNewOp<func::CallOp>(op, name, op.getType(), op->getOperands());

  return success();
}

void mlir::populateMathExtToLibmConversionPatterns(RewritePatternSet &patterns, PatternBenefit benefit) {
  (void)patterns.add<VecOpToScalarOp<mathExt::AcosOp>, VecOpToScalarOp<mathExt::AsinOp>>(patterns.getContext(),
                                                                                         benefit);
  (void)patterns.add<PromoteOpToF32<mathExt::AcosOp>, PromoteOpToF32<mathExt::AsinOp>>(patterns.getContext(), benefit);

  (void)patterns.add<ScalarOpToLibmCall<mathExt::AcosOp>>(patterns.getContext(), "acosf", "acos", benefit);
  (void)patterns.add<ScalarOpToLibmCall<mathExt::AsinOp>>(patterns.getContext(), "asinf", "asin", benefit);
}

namespace {
struct MathExtToLibmPass : public MathExtToLibmBase<MathExtToLibmPass> {
  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<linalg::LinalgDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<mathExt::MathExtDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<bufferization::BufferizationDialect>();
  }
};
}  // namespace

void MathExtToLibmPass::runOnOperation() {
  auto module = getOperation();

  RewritePatternSet patterns(&getContext());
  populateMathExtToLibmConversionPatterns(patterns, 1);

  ConversionTarget target(getContext());
  target.addLegalDialect<arith::ArithDialect, BuiltinDialect, func::FuncDialect, vector::VectorDialect,
                         math::MathDialect>();
  target.addIllegalDialect<mathExt::MathExtDialect>();
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createMathExtToLibmPass() {
  return std::make_unique<MathExtToLibmPass>();
}
