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
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "akg/Dialect/MindSpore/Passes.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#ifndef GEN_PASS_DECL_MAKEDYNAMICBROADCASTABLE
#define GEN_PASS_DECL_MAKEDYNAMICBROADCASTABLE
#ifndef GEN_PASS_DEF_MAKEDYNAMICBROADCASTABLE
#define GEN_PASS_DEF_MAKEDYNAMICBROADCASTABLE
#include "akg/Dialect/MindSpore/Passes.h.inc"
#endif
#endif
}  // namespace mlir

using namespace mlir;
using namespace akgglobal;
using namespace mlir::tosa;
using namespace mlir::shape;
using namespace mlir::mindspore;
static bool ignoreImplicitBroadcast = false;

static bool needInsertBroadCastOrCast(const Type &oprand, const Type &target) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  int64_t opndRank = oprand.cast<ShapedType>().getRank();
  int64_t targetRank = target.cast<ShapedType>().getRank();
  for (int64_t i = opndRank - 1; i >= 0; i--) {
    int64_t oprandDim = oprand.cast<ShapedType>().getShape()[i];
    if (oprandDim != ShapedType::kDynamic) {
      continue;
    }
    if (analysis.isSameSymbolicDim(oprand, i, target, i + (targetRank - opndRank))) {
      continue;
    }
    return true;
  }
  return false;
}

static Type GetCastedShape(Type target, Type oprand, SmallVector<int64_t> &needCastDims) {
  int64_t opndRank = oprand.cast<ShapedType>().getRank();
  int64_t targetRank = target.cast<ShapedType>().getRank();
  auto opndStaticShape = oprand.cast<ShapedType>().getShape();
  SmallVector<int64_t> targetStaticShape(target.cast<ShapedType>().getShape());
  auto elementTy = oprand.cast<ShapedType>().getElementType();
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  llvm::SmallVector<std::string> castedSymbolShape = *(analysis.getSymbolicShape(target));
  if (targetRank - opndRank > 0) {
    (void)castedSymbolShape.erase(castedSymbolShape.begin(), castedSymbolShape.begin() + (targetRank - opndRank));
    (void)targetStaticShape.erase(targetStaticShape.begin(), targetStaticShape.begin() + (targetRank - opndRank));
  }

  // if dim  = ?, cast oprand's dim symbol to target's symbol.
  // if dim != ?, ignore it.
  for (int64_t i = opndRank - 1; i >= 0; i--) {
    if (opndStaticShape[i] != ShapedType::kDynamic) {
      castedSymbolShape[i] = std::to_string(opndStaticShape[i]);
      targetStaticShape[i] = opndStaticShape[i];
      needCastDims[i] = 0;
      continue;
    }
    needCastDims[i] = 1;
  }
  Type castedTy = RankedTensorType::get(targetStaticShape, elementTy);
  return analysis.updateSymbolicShape(castedTy, castedSymbolShape);
}

namespace {
template <typename OpTy>
struct EnableTosaBroadCastOp : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy binaryOp, PatternRewriter &rewriter) const override {
    mlir::Location loc = binaryOp.getLoc();
    Value input1 = binaryOp.getInput1();
    Value input2 = binaryOp.getInput2();
    Value output = binaryOp.getResult();
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    if (!analysis.hasSymbolicShape(input1.getType()) || !analysis.hasSymbolicShape(input2.getType()) ||
        !analysis.hasSymbolicShape(output.getType())) {
      return success();
    }
    int64_t rank = output.getType().cast<ShapedType>().getRank();
    bool lhsNeedInsertBroadCastOrCast = needInsertBroadCastOrCast(input1.getType(), output.getType());
    bool rhsNeedInsertBroadCastOrCast = needInsertBroadCastOrCast(input2.getType(), output.getType());
    // ignore implicit broadcast, cast inputs' shape to out's shape directly. Note that this will generate incorrect
    // code, which needs to be corrected in the following pass.
    if (ignoreImplicitBroadcast) {
      // todo: different rank
      if (lhsNeedInsertBroadCastOrCast) {
        SmallVector<int64_t> lhsNeedCastDims(rank, 0);
        Type newType = GetCastedShape(output.getType(), input1.getType(), lhsNeedCastDims);
        Value newInput = rewriter.create<tensor::CastOp>(loc, newType, input1);
        binaryOp.getOperation()->setOperand(0, newInput);
      }
      if (rhsNeedInsertBroadCastOrCast) {
        SmallVector<int64_t> rhsNeedCastDims(rank, 0);
        Type newType = GetCastedShape(output.getType(), input2.getType(), rhsNeedCastDims);
        Value newInput = rewriter.create<tensor::CastOp>(loc, newType, input2);
        binaryOp.getOperation()->setOperand(1, newInput);
      }
      return success();
    }

    if (lhsNeedInsertBroadCastOrCast && rhsNeedInsertBroadCastOrCast) {
      Value shapeof1 = rewriter.create<shape::ShapeOfOp>(loc, input1);
      Value shapeof2 = rewriter.create<shape::ShapeOfOp>(loc, input2);
      Value bs = rewriter.create<shape::BroadcastOp>(loc, shape::getExtentTensorType(rewriter.getContext(), rank),
                                                     shapeof1, shapeof2, /*error=*/nullptr);
      Value newInput1 = rewriter.create<mindspore::BroadcastToOp>(loc, output.getType(), input1, bs);
      Value newInput2 = rewriter.create<mindspore::BroadcastToOp>(loc, output.getType(), input2, bs);
      // update operands
      binaryOp.getOperation()->setOperand(0, newInput1);
      binaryOp.getOperation()->setOperand(1, newInput2);
    } else if (lhsNeedInsertBroadCastOrCast || rhsNeedInsertBroadCastOrCast) {
      Value input = lhsNeedInsertBroadCastOrCast ? input1 : input2;
      Value bs = rewriter.create<shape::ShapeOfOp>(loc, lhsNeedInsertBroadCastOrCast ? input2 : input1);
      Value newInput = rewriter.create<mindspore::BroadcastToOp>(loc, output.getType(), input, bs);
      binaryOp.getOperation()->setOperand(lhsNeedInsertBroadCastOrCast ? 0 : 1, newInput);
    }
    return success();
  }
};

}  // namespace

namespace {
/**
 * @brief Trace the dimension of ops that need to be fixed to the input arguments.
 *        e.g.
 *          func.func @elem_broadcast_last_5(%arg1: tensor<?xf32, {SymShapeAttr = ["s1"]}>) {
 *            %0 = "tosa.reshape"(%arg1) {new_shape = array<i64: 1, -1>} : (tensor<?xf32, {SymShapeAttr = ["s1"]}>) ->
 * tensor<1x?xf32, {SymShapeAttr = ["1", "s1"]}> %cast_0 = tensor.cast %0 : tensor<1x?xf32, {SymShapeAttr = ["1",
 * "s1"]}> to tensor<1x?xf32, {SymShapeAttr = ["1", "s23"]}>
 *            ...
 *          }
 *        `%cast_0` shows there is a shape-cast from "s1" to "s23" and that is the `dim1` of `%0`
 *        The ShapeTracer will trace `dim1` of `%0` all the way to func arguments and returns `dim0` of `%arg1`
 */
class ShapeTracer {
 public:
  explicit ShapeTracer(SmallVector<tensor::CastOp> needFixOps) : needFixOps(needFixOps) {}
  SmallVector<int64_t> Trace(Value arg) {
    auto rank = arg.getType().cast<ShapedType>().getRank();
    SmallVector<int64_t> needCastDims(rank, 0);
    for (auto castOp : needFixOps) {
      auto src = castOp.getSource();
      if (src == arg) {
        (void)GetCastedShape(castOp.getDest().getType(), src.getType(), needCastDims);
        break;
      } else if (src.getDefiningOp()) {
        // todo(baiji): support more shape reconstruct op like transpose
        if (GoThroughIntermediateNoSideEffectOps(arg, castOp, needCastDims)) {
          break;
        }
      }
    }
    return needCastDims;
  }

 private:
  void ReassociateShape(const Type before, const Type after, SmallVector<int64_t> beforeNeedCast,
                        SmallVector<int64_t> &afterNeedCast) const {
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    llvm::SmallVector<std::string> castedSymbolShapeBefore = *(analysis.getSymbolicShape(before));
    llvm::SmallVector<std::string> castedSymbolShapeAfter = *(analysis.getSymbolicShape(after));
    assert(castedSymbolShapeBefore.size() == beforeNeedCast.size());
    assert(castedSymbolShapeAfter.size() == afterNeedCast.size());
    for (auto [shp, cast] : llvm::zip(castedSymbolShapeBefore, beforeNeedCast)) {
      if (cast == (int64_t)0) {
        continue;
      }
      for (size_t i = 0; i < castedSymbolShapeAfter.size(); ++i) {
        if (shp == castedSymbolShapeAfter[i]) {
          afterNeedCast[i] = 1;
          break;
        }
      }
    }
  }

  bool GoThroughIntermediateNoSideEffectOps(Value arg, tensor::CastOp castOp,
                                            SmallVector<int64_t> &needCastDims) const {
    auto src = castOp.getSource();
    auto check = [&](Operation *op) -> bool {
      if (!op) {
        return false;
      }
      if (auto reshape = dyn_cast<tosa::ReshapeOp>(op)) {
        return reshape.getInput1() == arg;
      }
      if (auto reshape = dyn_cast<mindspore::ReshapeOp>(op)) {
        return reshape.getInput() == arg;
      }
      if (auto cast = dyn_cast<mindspore::CastOp>(op)) {
        return cast.getInput() == arg;
      }
      return false;
    };
    if (!check(src.getDefiningOp())) {
      return false;
    }
    auto rank = src.getType().cast<ShapedType>().getRank();
    SmallVector<int64_t> srcNeedCastDims(rank, 0);
    (void)GetCastedShape(castOp.getDest().getType(), src.getType(), srcNeedCastDims);
    ReassociateShape(src.getType(), arg.getType(), srcNeedCastDims, needCastDims);
    return true;
  }
  SmallVector<tensor::CastOp> needFixOps;
};

/// Pass that enables broadcast by making all input arrays have the same
/// number of dimensions. Insert RESHAPE operations to lower rank operand
struct MakeDynamicBroadcastable : public impl::MakeDynamicBroadcastableBase<MakeDynamicBroadcastable> {
 public:
  MakeDynamicBroadcastable() {}
  explicit MakeDynamicBroadcastable(bool ignoreImplicitBroadcast) {
    this->IgnoreImplicitBroadcast = ignoreImplicitBroadcast;
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<
                    mindspore::MindSporeDialect,
                    shape::ShapeDialect,
                    tosa::TosaDialect,
                    tensor::TensorDialect
                  >();
    // clang-format on
  }

  void insertDynamicIndexingAttr() {
    func::FuncOp funcOp;
    SmallVector<tensor::CastOp> needFixOps;
    getOperation()->walk([&](Operation *op) {
      if (auto f = dyn_cast<func::FuncOp>(op)) {
        funcOp = f;
      } else if (auto c = dyn_cast<tensor::CastOp>(op)) {
        needFixOps.push_back(c);
      }
    });
    if (!funcOp) {
      llvm::errs() << "No funcOp, cannot insertDynamicIndexingAttr.\n";
      return;
    }
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    ShapeTracer tracer(needFixOps);
    for (size_t argIdx = 0; argIdx < funcOp.getBody().front().getArguments().size(); ++argIdx) {
      auto arg = funcOp.getBody().front().getArgument(argIdx);
      auto needCastDims = tracer.Trace(arg);
      tool.recordNeedFixIndice(argIdx, needCastDims);
    }
  }

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    MLIRContext *ctx = func.getContext();
    ignoreImplicitBroadcast = IgnoreImplicitBroadcast;
    // Add the generated patterns to the list.
    // mindspore dialect
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::AddOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::SubOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::MulOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::DivOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::PowOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::EqualOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::GreaterOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::GreaterEqualOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::LogicalAndOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::LessOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::LessEqualOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<mindspore::NotEqualOp>>(ctx);

    // tosa dialect
    (void)patterns.add<EnableTosaBroadCastOp<tosa::AddOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::ArithmeticRightShiftOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::EqualOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::GreaterEqualOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::GreaterOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::LogicalAndOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::LogicalLeftShiftOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::LogicalXorOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::MaximumOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::MulOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::PowOp>>(ctx);
    (void)patterns.add<EnableTosaBroadCastOp<tosa::SubOp>>(ctx);

    // Use TopDownTraversal for compile time reasons
    GreedyRewriteConfig grc;
    grc.useTopDownTraversal = true;
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns), grc);
    if (ignoreImplicitBroadcast) {
      insertDynamicIndexingAttr();
      ignoreImplicitBroadcast = false;
    }
  }
};
}  // namespace

std::unique_ptr<Pass> mlir::createMakeDynamicBroadcastablePass() {
  return std::make_unique<MakeDynamicBroadcastable>();
}

std::unique_ptr<Pass> mlir::createMakeDynamicBroadcastablePass(bool ignoreImplicitBroadcast) {
  return std::make_unique<MakeDynamicBroadcastable>(ignoreImplicitBroadcast);
}
