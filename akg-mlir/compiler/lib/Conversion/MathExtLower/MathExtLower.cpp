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

#include "akg/Conversion/MathExtLower/MathExtLower.h"

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Math/IR/MathExtOps.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Utils.h"
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
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
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

// Nan in IEE754 can be represent as 0x7FC00000(in binary)
// then we can bitcast the fp32 to uint32, then we compare the value and the NanVal;

template <typename SrcOp>
class ConvertMathExtIsnanOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, Adaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value opnd = op->getOperand(0);
    auto loc = op->getLoc();
    (void)adaptor;

    // int32_t ix
    // GET_FLOAT_WORD(ix,opnd)
    // ix &= 0x7fffffff
    // ix = 0x7f800000 - ix
    // ix = (int)(((uint32_t)(ix))>>31)
    // return Trunc(ix)
    const int32_t bitMask = 0x7FFFFFFF;
    const int32_t bound = 0x7f800000;
    auto bitMaskVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(bitMask));
    auto boundkVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(bound));
    Value ix = rewriter.create<arith::BitcastOp>(loc, rewriter.getI32Type(), opnd).getResult();
    ix = rewriter.create<arith::AndIOp>(loc, bitMaskVal, ix).getResult();
    ix = rewriter.create<arith::SubIOp>(loc, boundkVal, ix).getResult();
    auto shiftVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(31));
    ix = rewriter.create<arith::ShRUIOp>(loc, ix, shiftVal).getResult();
    ix = rewriter.create<arith::TruncIOp>(loc, rewriter.getI1Type(), ix).getResult();
    rewriter.replaceOp(op, ix);
    return success();
  }
};

// the PosInf in IEE754 can be represent as 0x7F800000(in binary);
// the NegInf can be represent as 0xFF800000(binary)
// so the fp32 can first bitcast to uint32;
// then we can judge the val and the PosInf/NegInf;
template <typename SrcOp>
class ConvertMathExtIsinfOp : public OpConversionPattern<SrcOp> {
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  using Adaptor = typename SrcOp::Adaptor;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, Adaptor adaptor, ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value opnd = op->getOperand(0);
    (void)adaptor;

    auto loc = op->getLoc();
    // create pi const Op;
    const uint32_t posInf = 0x7F800000;
    const uint32_t negInf = 0xFF800000;
    auto posInfVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr((int32_t)posInf));
    auto negInfVal = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr((int32_t)negInf));
    auto posVal = rewriter.create<arith::BitcastOp>(loc, rewriter.getF32Type(), posInfVal.getResult());
    auto negVal = rewriter.create<arith::BitcastOp>(loc, rewriter.getF32Type(), negInfVal.getResult());
    auto cmp0 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ, opnd, posVal);
    auto cmp1 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UEQ, opnd, negVal);
    auto orVal = rewriter.create<arith::OrIOp>(loc, cmp0.getResult(), cmp1.getResult());
    rewriter.replaceOp(op, orVal.getResult());
    return success();
  }
};

struct MathExtLowerPass : public MathExtLowerBase<MathExtLowerPass> {
 public:
  MathExtLowerPass() = default;

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

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target
      .addLegalDialect<vector::VectorDialect, gpu::GPUDialect, scf::SCFDialect, arith::ArithDialect,
                       linalg::LinalgDialect, linalgExt::LinalgExtDialect, cf::ControlFlowDialect, mlir::BuiltinDialect,
                       tensor::TensorDialect, func::FuncDialect, math::MathDialect, shape::ShapeDialect,
                       LLVM::LLVMDialect, AffineDialect, memref::MemRefDialect, bufferization::BufferizationDialect>();
    target.addIllegalDialect<mindspore::MindSporeDialect>();
    target.addIllegalDialect<mathExt::MathExtDialect>();
    target.addLegalOp<mathExt::AcosOp>();
    target.addLegalOp<mathExt::AsinOp>();
    // clang-format off
    (void)patterns.add<
      ConvertMathExtIsinfOp<mathExt::IsinfOp>,
      ConvertMathExtIsnanOp<mathExt::IsnanOp>
    >(patterns.getContext());
    // clang-format on

    // finalizing Conversion
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> mlir::createMathExtLowerPass() { return std::make_unique<MathExtLowerPass>(); }
