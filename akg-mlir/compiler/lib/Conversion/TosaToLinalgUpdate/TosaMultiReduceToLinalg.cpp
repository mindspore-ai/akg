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

#include "akg/Conversion/TosaToLinalgUpdate/TosaMultiReduceToLinalg.h"

#include <algorithm>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#ifndef GEN_PASS_DEF_TOSAMULTIREDUCETOLINALG
#define GEN_PASS_DEF_TOSAMULTIREDUCETOLINALG
#include "akg/Conversion/Passes.h.inc"
#endif
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace tosa {
namespace {

const size_t kVectorSizeEight = 8;

bool IsTosaReduceOp(Operation *redOp) {
  if (isa<tosa::ReduceAllOp>(redOp) || isa<tosa::ReduceAnyOp>(redOp) || isa<tosa::ReduceMaxOp>(redOp) ||
      isa<tosa::ReduceMinOp>(redOp) || isa<tosa::ReduceProdOp>(redOp) || isa<tosa::ReduceSumOp>(redOp)) {
    return true;
  }
  return false;
}

size_t valueUsageCount(Value v, func::FuncOp funcOp) {
  size_t cnt = 0;
  funcOp.walk([&](Operation *op) {
    auto operands = op->getOperands();
    cnt += (size_t)std::count_if(operands.begin(), operands.end(), [&v](const Value operand) { return operand == v; });
  });
  return cnt;
}

void findNextRedOp(size_t currIdx, size_t groupNum, const SmallVector<Operation *, kVectorSizeEight> &redOpList,
                   SmallVector<bool, kVectorSizeEight> &usedRedOps,
                   SmallVector<SmallVector<Operation *, kVectorSizeEight>, kVectorSizeEight> &redOpsGroups,
                   func::FuncOp funcOp) {
  if (currIdx == redOpList.size()) {
    return;
  }
  redOpsGroups[groupNum].push_back(redOpList[currIdx]);
  usedRedOps[currIdx] = true;
  for (size_t i = currIdx + 1; i < redOpList.size(); i++) {
    if (redOpList[i]->getOperands()[0] == redOpList[currIdx]->getResults()[0] &&
        redOpList[i]->getName() == redOpList[currIdx]->getName() &&
        valueUsageCount(redOpList[currIdx]->getResults()[0], funcOp) == 1) {
      findNextRedOp(i, groupNum, redOpList, usedRedOps, redOpsGroups, funcOp);
    }
  }
}

static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy, PatternRewriter &rewriter) {
  if (isa<tosa::ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, 0.0);
  }

  if (isa<tosa::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.getIntegerAttr(elementTy, 0);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, 1.0);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.getIntegerAttr(elementTy, 1);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy,
                                 APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), false));
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), true));
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));
  }

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getAllOnes(1));
  }

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getZero(1));
  }

  if (isa<tosa::ArgMaxOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), true));
  }

  if (isa<tosa::ArgMaxOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));
  }

  return {};
}

static Value createLinalgBodyCalculationForReduceOp(Operation *op, ValueRange args, Type elementTy,
                                                    PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<tosa::ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }

  if (isa<tosa::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::AddIOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MulFOp>(loc, args);
  }

  if (isa<tosa::ReduceProdOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::MulIOp>(loc, args);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MinNumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMinOp>(op) && isa<IntegerType>(elementTy)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MaxNumFOp>(loc, args[0], args[1]);
  }

  if (isa<tosa::ReduceMaxOp>(op) && isa<IntegerType>(elementTy)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<tosa::ReduceAllOp>(op) && elementTy.isInteger(1)) {
    return rewriter.create<arith::AndIOp>(loc, args);
  }

  if (isa<tosa::ReduceAnyOp>(op) && elementTy.isInteger(1)) {
    return rewriter.create<arith::OrIOp>(loc, args);
  }

  return {};
}

bool IntInVector(int i, const SmallVector<int, 8> &redAxes) {
  return std::any_of(redAxes.begin(), redAxes.end(), [i](int v) { return v == i; });
}

static LogicalResult reduceMatchAndRewriteHelper(Operation *op, PatternRewriter &rewriter) {
  func::FuncOp funcOp = dyn_cast<func::FuncOp>(op->getParentOp());
  SmallVector<Operation *, kVectorSizeEight> redOpList;
  SmallVector<bool, 8> usedRedOps;
  SmallVector<SmallVector<Operation *, 8>, 8> redOpsGroups;

  // collect all tosa.reduce
  funcOp.walk([&](Operation *redOp) {
    if (IsTosaReduceOp(redOp)) {
      redOpList.push_back(redOp);
      usedRedOps.push_back(false);
    }
  });

  for (size_t i = 0; i < redOpList.size(); i++) {
    if (!usedRedOps[i]) {
      redOpsGroups.push_back(SmallVector<Operation *, 8>());
      findNextRedOp(i, redOpsGroups.size() - 1, redOpList, usedRedOps, redOpsGroups, funcOp);
    }
  }

  // rewrite
  for (auto redOps : redOpsGroups) {
    SmallVector<int, 8> redAxes;
    for (auto redOp : redOps) {
      auto axis = dyn_cast<IntegerAttr>(redOp->getAttr("axis")).getValue().getSExtValue();
      redAxes.push_back(static_cast<int>(axis));
    }
    Operation *firstRedOp = redOps[0];
    Operation *lastRedOp = redOps[redOps.size() - 1];
    auto loc = lastRedOp->getLoc();

    auto inputTy = cast<ShapedType>(firstRedOp->getOperand(0).getType());
    auto resultTy = cast<ShapedType>(lastRedOp->getResult(0).getType());
    auto elementTy = resultTy.getElementType();
    Value input = firstRedOp->getOperand(0);

    llvm::SmallVector<int64_t> reduceShape;
    SmallVector<Value> dynDims;
    for (unsigned i = 0; i < inputTy.getRank(); i++) {
      if (!IntInVector(static_cast<int>(i), redAxes)) {
        reduceShape.push_back(inputTy.getDimSize(i));
        if (inputTy.isDynamicDim(i)) {
          dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
        }
      }
    }

    Type reduceTy = RankedTensorType::get(reduceShape, resultTy.getElementType());

    loc = firstRedOp->getLoc();
    rewriter.setInsertionPointAfter(lastRedOp);
    auto emptyTensor =
      rewriter.create<tensor::EmptyOp>(loc, reduceShape, resultTy.getElementType(), dynDims).getResult();

    auto fillValueAttr = createInitialValueForReduceOp(firstRedOp, elementTy, rewriter);
    if (!fillValueAttr) {
      return rewriter.notifyMatchFailure(op, "No initial value found for reduction operation");
    }
    auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
    auto filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{fillValue}, ValueRange{emptyTensor}).result();

    SmallVector<AffineExpr, 2> srcExprs;
    SmallVector<AffineExpr, 2> dstExprs;
    SmallVector<utils::IteratorType, 4> iteratorTypes;
    for (int64_t i = 0, rank = inputTy.getRank(); i != rank; ++i) {
      srcExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));

      iteratorTypes.push_back(IntInVector(static_cast<int>(i), redAxes) ? utils::IteratorType::reduction
                                                                        : utils::IteratorType::parallel);
      if (!IntInVector(static_cast<int>(i), redAxes)) {
        dstExprs.push_back(mlir::getAffineDimExpr(i, rewriter.getContext()));
      }
    }

    bool didEncounterError = false;
    auto maps = AffineMap::inferFromExprList({srcExprs, dstExprs}, rewriter.getContext());
    auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, reduceTy, input, filledTensor, maps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto result = createLinalgBodyCalculationForReduceOp(firstRedOp, blockArgs, elementTy, rewriter);
        if (result) {
          didEncounterError = true;
        }

        (void)nestedBuilder.create<linalg::YieldOp>(loc, result);
      });

    if (!didEncounterError) {
      return rewriter.notifyMatchFailure(op, "unable to create linalg.generic body for reduce op");
    }
    SmallVector<ReassociationExprs, 4> reassociationMap;
    uint64_t expandInputRank = cast<ShapedType>(linalgOp.getResults()[0].getType()).getRank();
    reassociationMap.resize(expandInputRank);

    size_t reassociationIdx = 0;
    if (expandInputRank != 0) {
      for (unsigned i = 0; i < inputTy.getRank(); i++) {
        size_t clipIdx = reassociationIdx < expandInputRank ? reassociationIdx : expandInputRank - 1;
        if (!IntInVector(static_cast<int>(i), redAxes)) {
          reassociationIdx++;
        }
        reassociationMap[clipIdx].push_back(rewriter.getAffineDimExpr(i));
      }
    }
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(lastRedOp, resultTy, linalgOp.getResults()[0], reassociationMap);
  }

  return success();
}
template <typename SrcOp>
class ReduceConverterPattern : public RewritePattern {
 public:
  explicit ReduceConverterPattern(MLIRContext *context) : RewritePattern(SrcOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const final {
    return reduceMatchAndRewriteHelper(op, rewriter);
  }
};

struct TosaMultiReduceToLinalg : public impl::TosaMultiReduceToLinalgBase<TosaMultiReduceToLinalg> {
  TosaMultiReduceToLinalg() = default;
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceAllOp>>(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceAnyOp>>(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceMaxOp>>(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceMinOp>>(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceProdOp>>(context);
    (void)patterns.insert<ReduceConverterPattern<tosa::ReduceSumOp>>(context);
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace tosa
}  // namespace mlir

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createTosaMultiReduceToLinalgPass() {
  return std::make_unique<tosa::TosaMultiReduceToLinalg>();
}
