/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include <numeric>
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
// #include "bishengir/Dialect/HACC/IR/HACC.h"
// #include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir {
#ifndef GEN_PASS_CLASSES
#define GEN_PASS_CLASSES
#include "akg/Conversion/Passes.h.inc"
#endif

static std::set<std::string> linalgUnarySet = {"mindspore.exp", "mindspore.log", "mindspore.abs", "mindspore.neg",
                                               "mindspore.sqrt"};
static std::set<std::string> linalgBinarySet = {"mindspore.add", "mindspore.mul",     "mindspore.sub",
                                                "mindspore.div", "mindspore.maximum", "mindspore.minimum",
                                                "mindspore.pow"};

bool isIntegerType(Operation *op) {
  Type type = op->getResultTypes()[0];
  auto elemTy = getElementTypeOrSelf(type);
  if (isa<IntegerType>(elemTy)) return true;
  return false;
}

static linalg::UnaryFn getLinalgUnaryKind(Operation *op) {
  linalg::UnaryFn kind;
  llvm::TypeSwitch<Operation *>(op)
    .Case([&](mindspore::ExpOp) { kind = linalg::UnaryFn::exp; })
    .Case([&](mindspore::AbsOp) { kind = linalg::UnaryFn::abs; })
    .Case([&](mindspore::LogOp) { kind = linalg::UnaryFn::log; })
    .Case([&](mindspore::NegateOp) { kind = linalg::UnaryFn::negf; })
    .Case([&](mindspore::SqrtOp) { kind = linalg::UnaryFn::sqrt; })
    .Default([](Operation *) {});
  return kind;
}

static linalg::BinaryFn getLinalgBinaryKind(Operation *op) {
  linalg::BinaryFn kind;
  llvm::TypeSwitch<Operation *>(op)
    .Case([&](mindspore::AddOp) { kind = linalg::BinaryFn::add; })
    .Case([&](mindspore::MulOp) { kind = linalg::BinaryFn::mul; })
    .Case([&](mindspore::SubOp) { kind = linalg::BinaryFn::sub; })
    .Case([&](mindspore::DivOp) { kind = linalg::BinaryFn::div; })
    .Case([&](mindspore::MaximumOp) { kind = linalg::BinaryFn::max_signed; })
    .Case([&](mindspore::MinimumOp) { kind = linalg::BinaryFn::min_signed; })
    .Case([&](mindspore::PowOp) { kind = linalg::BinaryFn::powf; })
    .Default([](Operation *) {});
  return kind;
}

static Attribute getOperationKindAttribute(Operation *op) {
  std::string opName = op->getName().getStringRef().str();
  Attribute attr;
  if (linalgUnarySet.count(opName) || (opName == "mindspore.abs" && !isIntegerType(op))) {
    linalg::UnaryFn kind = getLinalgUnaryKind(op);
    attr = linalg::UnaryFnAttr::get(op->getContext(), kind);
  } else if (linalgBinarySet.count(opName)) {
    linalg::BinaryFn kind = getLinalgBinaryKind(op);
    attr = linalg::BinaryFnAttr::get(op->getContext(), kind);
  }
  return attr;
}

static Operation *createElemwiseOp(Operation *op, Value emptyTensor, SmallVector<NamedAttribute> &attrs,
                                   PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  std::string opName = op->getName().getStringRef().str();
  Operation *namedOp = nullptr;
  auto src = op->getOperands();
  if (linalgUnarySet.count(opName)) {
    namedOp = rewriter.create<linalg::ElemwiseUnaryOp>(loc, src, emptyTensor, attrs);
  } else if (linalgBinarySet.count(opName)) {
    namedOp = rewriter.create<linalg::ElemwiseBinaryOp>(loc, src, emptyTensor, attrs);
  }
  return namedOp;
}

static LogicalResult elementwiseMatchAndRewriteHelper(Operation *op, PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  SmallVector<NamedAttribute> attrs;
  Attribute attr = getOperationKindAttribute(op);
  auto nameAttr = StringAttr::get(op->getContext(), "fun");
  attrs.push_back({nameAttr, attr});
  auto resultTy = cast<ShapedType>(op->getResult(0).getType());
  Value inputRes;
  for (auto input : op->getOperands()) {
    auto inputTy = dyn_cast<ShapedType>(input.getType());
    if (inputTy && inputTy == resultTy) {
      inputRes = input;
      break;
    }
  }
  SmallVector<Value> dynDims;
  for (int i = 0; i < resultTy.getRank(); i++) {
    if (resultTy.isDynamicDim(i)) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, inputRes, i));
    }
  }
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType(), dynDims);

  auto namedOp = createElemwiseOp(op, emptyTensor, attrs, rewriter);
  rewriter.replaceOp(op, namedOp->getResults());
  return success();
}

static LogicalResult invMatchAndRewriteHelper(Operation *op, PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  SmallVector<NamedAttribute> attrs;
  linalg::BinaryFn kind = linalg::BinaryFn::div;
  Attribute attr = linalg::BinaryFnAttr::get(op->getContext(), kind);
  auto nameAttr = StringAttr::get(op->getContext(), "fun");
  attrs.push_back({nameAttr, attr});
  auto resultTy = cast<ShapedType>(op->getResult(0).getType());
  Value inputRes;
  for (auto input : op->getOperands()) {
    auto inputTy = dyn_cast<ShapedType>(input.getType());
    if (inputTy && inputTy == resultTy) {
      inputRes = input;
      break;
    }
  }
  SmallVector<Value> dynDims;
  for (int i = 0; i < resultTy.getRank(); i++) {
    if (resultTy.isDynamicDim(i)) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, inputRes, i));
    }
  }
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType(), dynDims);

  auto one = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(resultTy.getElementType(), 1.0));
  auto namedOp =
    rewriter.create<linalg::ElemwiseBinaryOp>(loc, ValueRange{one, op->getOperands()[0]}, emptyTensor, attrs);
  rewriter.replaceOp(op, namedOp->getResults());
  return success();
}

static SmallVector<ReassociationExprs> getExpandMap(SmallVector<int64_t> axes, int64_t expandInputRank,
                                                    int64_t expandOutputRank, PatternRewriter &rewriter) {
  int64_t posAtInput = 0;
  SmallVector<ReassociationExprs> reassociation_map = {};
  ReassociationExprs expand_strategy;
  for (int64_t i = 0; i < expandOutputRank; i++) {
    expand_strategy.push_back(rewriter.getAffineDimExpr(i));
    if (!llvm::is_contained(axes, i)) {
      posAtInput += 1;
      if (posAtInput != expandInputRank) {  // not the last unreduced dimension
        reassociation_map.push_back(expand_strategy);
        expand_strategy = {};
      }
    }
  }
  reassociation_map.push_back(expand_strategy);
  return reassociation_map;
}

static Value createExpandShapeOp(Operation *op, PatternRewriter &rewriter, Value expandSrc, Value expandDst,
                                 uint64_t axis) {
  SmallVector<int64_t> dims = {static_cast<int64_t>(axis)};
  int64_t expandInputRank = cast<ShapedType>(expandSrc.getType()).getRank();
  int64_t expandOutputRank = cast<ShapedType>(expandDst.getType()).getRank();
  auto reassociation = getExpandMap(dims, expandInputRank, expandOutputRank, rewriter);
  Value expandShapeOp =
    rewriter.create<tensor::ExpandShapeOp>(op->getLoc(), expandDst.getType(), expandSrc, reassociation);
  return expandShapeOp;
}

// Returns the constant initial value for a given reduction operation. The
// attribute type varies depending on the element type required.
static TypedAttr createInitialValueForReduceOp(Operation *op, Type elementTy, PatternRewriter &rewriter) {
  if (isa<mindspore::ReduceSumOp>(op) && isa<FloatType>(elementTy)) return rewriter.getFloatAttr(elementTy, 0.0);

  if (isa<mindspore::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) return rewriter.getIntegerAttr(elementTy, 0);

  if (isa<mindspore::ReduceProdOp>(op) && isa<FloatType>(elementTy)) return rewriter.getFloatAttr(elementTy, 1.0);

  if (isa<mindspore::ReduceProdOp>(op) && isa<IntegerType>(elementTy)) return rewriter.getIntegerAttr(elementTy, 1);

  if (isa<mindspore::ReduceMinOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), false));

  if (isa<mindspore::ReduceMinOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<mindspore::ReduceMaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<mindspore::ReduceMaxOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  if (isa<mindspore::ReduceAllOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getAllOnes(1));

  if (isa<mindspore::ReduceAnyOp>(op) && elementTy.isInteger(1))
    return rewriter.getIntegerAttr(elementTy, APInt::getZero(1));

  if (isa<mindspore::ArgMaxOp>(op) && isa<FloatType>(elementTy))
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(cast<FloatType>(elementTy).getFloatSemantics(), true));

  if (isa<mindspore::ArgMaxOp>(op) && isa<IntegerType>(elementTy))
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));

  return {};
}

// Creates the body calculation for a reduction. The operations vary depending
// on the input type.
static Value createLinalgBodyCalculationForReduceOp(Operation *op, ValueRange args, Type elementTy,
                                                    PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  if (isa<mindspore::ReduceSumOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }

  if (isa<mindspore::ReduceSumOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::AddIOp>(loc, args);
  }

  if (isa<mindspore::ReduceProdOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MulFOp>(loc, args);
  }

  if (isa<mindspore::ReduceProdOp>(op) && isa<IntegerType>(elementTy)) {
    return rewriter.create<arith::MulIOp>(loc, args);
  }

  if (isa<mindspore::ReduceMinOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MinimumFOp>(loc, args[0], args[1]);
  }

  if (isa<mindspore::ReduceMinOp>(op) && isa<IntegerType>(elementTy)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<mindspore::ReduceMaxOp>(op) && isa<FloatType>(elementTy)) {
    return rewriter.create<arith::MaximumFOp>(loc, args[0], args[1]);
  }

  if (isa<mindspore::ReduceMaxOp>(op) && isa<IntegerType>(elementTy)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }

  if (isa<mindspore::ReduceAllOp>(op) && elementTy.isInteger(1)) return rewriter.create<arith::AndIOp>(loc, args);

  if (isa<mindspore::ReduceAnyOp>(op) && elementTy.isInteger(1)) return rewriter.create<arith::OrIOp>(loc, args);

  return {};
}

// Performs the match and rewrite for reduction operations. This includes
// declaring a correctly sized initial value, and the linalg.generic operation
// that reduces across the specified axis.
static LogicalResult reduceMatchAndRewriteHelper(Operation *op, uint64_t axis, PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto inputTy = cast<ShapedType>(op->getOperand(0).getType());
  auto resultTy = cast<ShapedType>(op->getResult(0).getType());
  auto elementTy = resultTy.getElementType();
  Value input = op->getOperand(0);

  SmallVector<int64_t> reduceShape;
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (axis != i) {
      reduceShape.push_back(inputTy.getDimSize(i));
      if (inputTy.isDynamicDim(i)) dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
    }
  }
  // First fill the output buffer with the init value.
  auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, reduceShape, elementTy, dynDims).getResult();
  auto fillValueAttr = createInitialValueForReduceOp(op, elementTy, rewriter);
  if (!fillValueAttr) return rewriter.notifyMatchFailure(op, "No initial value found for reduction operation");

  auto fillValue = rewriter.create<arith::ConstantOp>(loc, fillValueAttr);
  auto filledTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{fillValue}, ValueRange{emptyTensor}).result();

  bool didEncounterError = false;
  auto reduceOp = rewriter.create<linalg::ReduceOp>(
    loc, input, filledTensor, axis, [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
      auto result = createLinalgBodyCalculationForReduceOp(op, blockArgs, elementTy, rewriter);
      if (result) didEncounterError = true;

      nestedBuilder.create<linalg::YieldOp>(loc, result);
    });

  auto expandShapeOp = createExpandShapeOp(op, rewriter, reduceOp.getOperation()->getResult(0), op->getResult(0), axis);
  rewriter.replaceOp(op, expandShapeOp);

  if (!didEncounterError) return rewriter.notifyMatchFailure(op, "unable to create linalg.generic body for reduce op");
  return success();
}

static bool findIntermediateShape(ArrayRef<int64_t> lhsShape, ArrayRef<int64_t> rhsShape,
                                  SmallVector<int64_t> &intermediateShape, bool isDynamic) {
  if (isDynamic) {
    intermediateShape = {ShapedType::kDynamic};
    return true;
  }

  if (lhsShape.empty() || rhsShape.empty()) {
    intermediateShape = {};
    return true;
  }

  unsigned currLhsDim = 0, currRhsDim = 0;
  while (currLhsDim < lhsShape.size() && currRhsDim < rhsShape.size()) {
    int64_t rhsSize = rhsShape[currRhsDim];
    int64_t lhsSize = lhsShape[currLhsDim];
    while (lhsSize != rhsSize && currLhsDim < lhsShape.size() && currRhsDim < rhsShape.size()) {
      if (lhsSize < rhsSize) {
        currLhsDim++;
        if (currLhsDim < lhsShape.size()) {
          lhsSize *= lhsShape[currLhsDim];
        }
      } else {
        currRhsDim++;
        if (currRhsDim < rhsShape.size()) {
          rhsSize *= rhsShape[currRhsDim];
        }
      }
    }
    if (lhsSize == rhsSize) {
      intermediateShape.push_back(lhsSize);
    }
    currRhsDim++;
    currLhsDim++;
  }

  // If the iterators didn't reach the end and their leftover dimensions are not
  // equal to 1 an intermediate shape was not found.
  while (currLhsDim < lhsShape.size()) {
    if (lhsShape[currLhsDim++] != 1) {
      return false;
    }
  }

  while (currRhsDim < rhsShape.size()) {
    if (rhsShape[currRhsDim++] != 1) {
      return false;
    }
  }

  return true;
}

static bool createReassociationMapsForCollapse(PatternRewriter &rewriter, ArrayRef<int64_t> srcShape,
                                               ArrayRef<int64_t> dstShape,
                                               SmallVector<ReassociationExprs, 4> &reassociationMap, bool isDynamic) {
  // If the shape is dynamic, create a map for collapsing into one dimension.
  if (isDynamic) {
    SmallVector<AffineExpr, 2> exprs;
    for (int i = 0, s = srcShape.size(); i < s; ++i) exprs.push_back(rewriter.getAffineDimExpr(i));
    reassociationMap = {exprs};
    return true;
  }

  if (dstShape.empty()) {
    reassociationMap = {};
    return true;
  }

  reassociationMap.resize(dstShape.size());
  unsigned currSrcDim = 0, currDstDim = 0;
  while (currSrcDim < srcShape.size() && currDstDim < dstShape.size()) {
    int64_t dstSize = dstShape[currDstDim];
    int64_t srcSize = srcShape[currSrcDim];
    while (srcSize < dstSize && currSrcDim < srcShape.size()) {
      reassociationMap[currDstDim].push_back(rewriter.getAffineDimExpr(currSrcDim++));
      srcSize *= srcShape[currSrcDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currDstDim].push_back(rewriter.getAffineDimExpr(currSrcDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currDstDim == dstShape.size() - 1 || dstShape[currDstDim + 1] != 1) {
        while (currSrcDim < srcShape.size() && srcShape[currSrcDim] == 1) {
          reassociationMap[currDstDim].push_back(rewriter.getAffineDimExpr(currSrcDim++));
        }
      }
    }
    currDstDim++;
  }

  // If both iterators didn't reach the end, we have leftover dimensions which
  // implies that we have a mismatch in shape.
  return currSrcDim == srcShape.size() && currDstDim == dstShape.size();
}

Value createCollapse(PatternRewriter &rewriter, Location loc, ShapedType resultTy, Value operand) {
  ShapedType operandTy = cast<ShapedType>(operand.getType());
  if (resultTy == operandTy) return operand;

  bool isDynamic = !operandTy.hasStaticShape();

  if (isDynamic && resultTy.getRank() != 1) {
    (void)rewriter.notifyMatchFailure(loc, "Cannot collapse dynamic dims to more than one dimension");
    return {};
  }

  SmallVector<ReassociationExprs, 4> reassociationMap;
  if (!createReassociationMapsForCollapse(rewriter, operandTy.getShape(), resultTy.getShape(), reassociationMap,
                                          isDynamic)) {
    (void)rewriter.notifyMatchFailure(loc, "mindspore.reshape Attempting to collapse into an incompatible shape");
    return {};
  }

  SmallVector<int64_t> intermediateShape;
  if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(), intermediateShape, isDynamic)) {
    (void)rewriter.notifyMatchFailure(loc, "mindspore.reshape Cannot collapse into given shape");
    return {};
  }
  return rewriter.create<tensor::CollapseShapeOp>(loc, resultTy, operand, reassociationMap);
}

Value createExpand(PatternRewriter &rewriter, Location loc, ShapedType resultTy, Value operand) {
  ShapedType operandTy = cast<ShapedType>(operand.getType());
  if (resultTy == operandTy) return operand;

  bool isDynamic = !operandTy.hasStaticShape();

  if (isDynamic && operandTy.getRank() != 1) {
    (void)rewriter.notifyMatchFailure(loc, "Cannot expand dynamic dims from more than one dimension");
    return {};
  }

  SmallVector<ReassociationExprs, 4> reassociationMap;
  if (!createReassociationMapsForCollapse(rewriter, resultTy.getShape(), operandTy.getShape(), reassociationMap,
                                          isDynamic)) {
    (void)rewriter.notifyMatchFailure(loc, "mindspore.reshape Attempting to expand into an incompatible shape");
    return {};
  }

  SmallVector<int64_t> intermediateShape;
  if (!findIntermediateShape(operandTy.getShape(), resultTy.getShape(), intermediateShape, isDynamic) ||
      intermediateShape != operandTy.getShape()) {
    (void)rewriter.notifyMatchFailure(loc, "mindspore.reshape Cannot expand into given shape");
    return {};
  }
  return rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, operand, reassociationMap);
}

static LogicalResult reshapeMatchAndRewriteHelper(mindspore::ReshapeOp reshape, PatternRewriter &rewriter) {
  ShapedType operandTy = cast<ShapedType>(reshape.getInput().getType());
  ShapedType resultTy = cast<ShapedType>(reshape.getType());
  bool isDynamic = !operandTy.hasStaticShape();

  SmallVector<int64_t> intermediateShape;
  if (!findIntermediateShape(resultTy.getShape(), operandTy.getShape(), intermediateShape, isDynamic)) {
    return rewriter.notifyMatchFailure(reshape,
                                       "mindspore.reshape Cannot identify an intermediate shape between "
                                       "the given two shapes");
  }

  auto intermediateTy = RankedTensorType::get(intermediateShape, reshape.getType().getElementType());

  Value collapse = createCollapse(rewriter, reshape.getLoc(), intermediateTy, reshape.getInput());
  if (!collapse) return failure();

  Value expand = createExpand(rewriter, reshape.getLoc(), resultTy, collapse);
  if (!expand) return failure();

  rewriter.replaceOp(reshape, expand);
  return success();
}

static std::vector<int> ArrayAttrToVectorInt(ArrayAttr array) {
  std::vector<int> res;
  for (auto v : array.getValue()) {
    if (auto intAttr = dyn_cast<StringAttr>(v)) {
      int value = atoi(intAttr.getValue().str().c_str());
      res.push_back(value);
    }
  }
  return res;
}

static DenseI64ArrayAttr computeDiffShape(mindspore::BroadcastToOp brcOp) {
  Value input = brcOp.getInput();
  Value output = brcOp.getOutput();
  auto inputShape = cast<ShapedType>(input.getType()).getShape();
  auto outputShape = cast<ShapedType>(output.getType()).getShape();
  auto inputShapeSize = inputShape.size();
  auto outputShapeSize = outputShape.size();

  SmallVector<int64_t> dim;
  size_t inIdx = 0, outIdx = 0;
  while (inIdx < inputShapeSize && outIdx < outputShapeSize) {
    if (outputShape[outIdx] == inputShape[inIdx]) {
      outIdx++;
      inIdx++;
    } else {
      size_t tmpIdx = inIdx;
      while (tmpIdx < inputShapeSize && inputShape[tmpIdx] == 1) tmpIdx++;
      if (tmpIdx >= inputShapeSize) continue;
      if (inputShape[tmpIdx] == outputShape[outIdx]) {
        inIdx = tmpIdx + 1;
      }
      dim.push_back(outIdx);
      outIdx++;
    }
  }
  while (outIdx < outputShapeSize) dim.push_back(outIdx++);

  auto dimension = DenseI64ArrayAttr::get(brcOp.getContext(), ArrayRef<int64_t>(dim));
  return dimension;
}

static DenseI64ArrayAttr computeSameShape(mindspore::BroadcastToOp brcOp) {
  Value output = brcOp.getOutput();
  auto symbolAttr = dyn_cast<DictionaryAttr>(brcOp.getOperation()->getAttr("frontend_symbol"));
  auto intputShapeAttr = dyn_cast<ArrayAttr>(symbolAttr.get("input_0"));
  auto inputShape = ArrayAttrToVectorInt(intputShapeAttr);
  auto outputShape = cast<ShapedType>(output.getType()).getShape();

  SmallVector<int64_t> dim;
  for (size_t idx = 0; idx < inputShape.size(); idx++) {
    if (inputShape[idx] != outputShape[idx] || inputShape[idx] == 1) {
      dim.push_back(idx);
    }
  }
  auto dimension = DenseI64ArrayAttr::get(brcOp.getContext(), ArrayRef<int64_t>(dim));
  return dimension;
}

static DenseI64ArrayAttr computeDimension(mindspore::BroadcastToOp brcOp) {
  Value output = brcOp.getOutput();

  auto symbolAttr = dyn_cast<DictionaryAttr>(brcOp.getOperation()->getAttr("frontend_symbol"));
  auto intputShapeAttr = dyn_cast<ArrayAttr>(symbolAttr.get("input_0"));
  auto outputShape = cast<ShapedType>(output.getType()).getShape();

  DenseI64ArrayAttr dimension;
  if (intputShapeAttr.getValue().size() == outputShape.size())
    dimension = computeSameShape(brcOp);
  else
    dimension = computeDiffShape(brcOp);
  return dimension;
}

static Value getDynamicRankTensor(mindspore::BroadcastToOp brcOp) {
  SmallVector<Operation *, 8> msOps;
  auto func = brcOp.getOperation()->getParentOp();
  func->walk([&](Operation *op) {
    if (isa<mindspore::AddOp, mindspore::MulOp, mindspore::SubOp, mindspore::DivOp, mindspore::PowOp,
            mindspore::MaximumOp, mindspore::MinimumOp, mindspore::EqualOp, mindspore::GreaterOp,
            mindspore::GreaterEqualOp, mindspore::LogicalAndOp, mindspore::LogicalOrOp, mindspore::SelectOp,
            LLVM::ReturnOp>(op))
      msOps.push_back(op);
  });
  for (auto msOp : msOps) {
    if (isa<LLVM::ReturnOp>(msOp)) {
      auto oper0 = msOp->getOperands()[0];
      if (oper0.getDefiningOp() == brcOp) return oper0;
    }
    auto oper0 = msOp->getOperands()[0];
    auto oper1 = msOp->getOperands()[1];
    if (oper0.getDefiningOp() == brcOp) return oper1;
    if (oper1.getDefiningOp() == brcOp) return oper0;
  }
  return Value();
}

static LogicalResult broadcastMatchAndRewriteHelper(mindspore::BroadcastToOp brcOp, PatternRewriter &rewriter) {
  auto loc = brcOp.getLoc();
  Value input = brcOp.getInput();
  Value output = brcOp.getOutput();

  SmallVector<Value> dynDims;
  auto brcDst = getDynamicRankTensor(brcOp);
  auto brcDstTy = cast<ShapedType>(brcDst.getType());
  for (int i = 0; i < brcDstTy.getRank(); i++) {
    if (brcDstTy.isDynamicDim(i)) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, brcDst, i));
    }
  }

  auto resultTy = cast<ShapedType>(output.getType());
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType(), dynDims);
  auto dimension = computeDimension(brcOp);

  rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(brcOp, input, emptyTensor, dimension);
  return success();
}

static LogicalResult roundMatchAndRewriteHelper(mindspore::RoundOp roundOp, PatternRewriter &rewriter) {
  auto loc = roundOp.getLoc();
  Value input = roundOp.getOperand();
  Value output = roundOp.getResult();

  SmallVector<Value> dynDims;
  ShapedType shapedType = input.getType().cast<ShapedType>();
  ArrayRef<int64_t> typeShapes = shapedType.getShape();
  for (size_t i = 0; i < typeShapes.size(); i++) {
    if (typeShapes[i] == ShapedType::kDynamic) {
      auto dynDim = rewriter.create<tensor::DimOp>(loc, input, i);
      dynDims.push_back(dynDim);
    }
  }
  auto dstElemType = getElementTypeOrSelf(output);
  auto resultType = RankedTensorType::get(shapedType.getShape(), dstElemType);
  Value emptyTensor =
    rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType(), dynDims);

  rewriter.replaceOpWithNewOp<linalg::RoundOp>(roundOp, input, emptyTensor);
  return success();
}

static LogicalResult selectMatchAndRewriteHelper(mindspore::SelectOp selectOp, PatternRewriter &rewriter) {
  auto loc = selectOp.getLoc();
  Value condition = selectOp.getPred();
  Value trueValue = selectOp.getOnTrue();
  Value falseValue = selectOp.getOnFalse();
  Value output = selectOp.getResult();

  SmallVector<Value> dynDims;
  auto resultTy = output.getType().cast<ShapedType>();
  Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultTy.getShape(), resultTy.getElementType(), dynDims);

  rewriter.replaceOpWithNewOp<linalg::SelectOp>(selectOp, ValueRange{condition, trueValue, falseValue},
                                                ValueRange{emptyTensor});
  return success();
}

static LogicalResult isFiniteMatchAndRewriteHelper(Operation *op, PatternRewriter &rewriter) {
  // auto namedOp = rewriter.create<hfusion::IsFiniteOp>(loc, resultTy, op->getOperands()[0]);
  // rewriter.replaceOp(op, namedOp->getResults());
  return success();
}

template <typename SrcOp>
class MindSporeElemwiseConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op, PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(op, rewriter);
  }
};

template <typename SrcOp>
class MindSporeReduceConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op, PatternRewriter &rewriter) const final {
    return reduceMatchAndRewriteHelper(op, *(op.getAxis().data()), rewriter);
  }
};

class MindSporeInplaceAssignConverter : public OpRewritePattern<mindspore::InplaceAssignOp> {
 public:
  using OpRewritePattern<mindspore::InplaceAssignOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::InplaceAssignOp op, PatternRewriter &rewriter) const final {
    auto linalgCopyOp = rewriter
                          .create<linalg::CopyOp>(op.getLoc(), op.getType(),
                                                  op.getInput1(),  // ins
                                                  op.getInput0(),  // outs
                                                  ArrayRef<NamedAttribute>())
                          .getResult(0);

    rewriter.replaceOp(op, linalgCopyOp);
    return success();
  }
};

class MindSporeReshapeConverter : public OpRewritePattern<mindspore::ReshapeOp> {
 public:
  using OpRewritePattern<mindspore::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::ReshapeOp op, PatternRewriter &rewriter) const final {
    return reshapeMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeBroadcastConverter : public OpRewritePattern<mindspore::BroadcastToOp> {
 public:
  using OpRewritePattern<mindspore::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::BroadcastToOp op, PatternRewriter &rewriter) const final {
    return broadcastMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeRoundConverter : public OpRewritePattern<mindspore::RoundOp> {
 public:
  using OpRewritePattern<mindspore::RoundOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::RoundOp op, PatternRewriter &rewriter) const final {
    return roundMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeSelectConverter : public OpRewritePattern<mindspore::SelectOp> {
 public:
  using OpRewritePattern<mindspore::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::SelectOp op, PatternRewriter &rewriter) const final {
    return selectMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeIsFiniteConverter : public OpRewritePattern<mindspore::IsFiniteOp> {
 public:
  using OpRewritePattern<mindspore::IsFiniteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::IsFiniteOp op, PatternRewriter &rewriter) const final {
    return isFiniteMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeInvConverter : public OpRewritePattern<mindspore::InvOp> {
 public:
  using OpRewritePattern<mindspore::InvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::InvOp op, PatternRewriter &rewriter) const final {
    return invMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeConstConverter : public OpRewritePattern<mindspore::ConstOp> {
 public:
  using OpRewritePattern<mindspore::ConstOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::ConstOp op, PatternRewriter &rewriter) const final {
    (void)rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
    return success();
  }
};

void populateLowerMindSporeToLinalgNamedPattern(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
    MindSporeElemwiseConverter<mindspore::AddOp>,
    MindSporeElemwiseConverter<mindspore::MulOp>,
    MindSporeElemwiseConverter<mindspore::SubOp>,
    MindSporeElemwiseConverter<mindspore::DivOp>,
    MindSporeElemwiseConverter<mindspore::PowOp>,
    MindSporeElemwiseConverter<mindspore::MaximumOp>,
    MindSporeElemwiseConverter<mindspore::MinimumOp>,
    MindSporeElemwiseConverter<mindspore::ExpOp>,
    MindSporeElemwiseConverter<mindspore::AbsOp>,
    MindSporeElemwiseConverter<mindspore::LogOp>,
    MindSporeElemwiseConverter<mindspore::NegateOp>,
    MindSporeElemwiseConverter<mindspore::SqrtOp>,
    MindSporeElemwiseConverter<mindspore::RsqrtOp>,
    MindSporeReduceConverter<mindspore::ReduceMaxOp>,
    MindSporeReduceConverter<mindspore::ReduceMinOp>,
    MindSporeReduceConverter<mindspore::ReduceSumOp>,
    MindSporeReduceConverter<mindspore::ReduceProdOp>,
    MindSporeInplaceAssignConverter,
    MindSporeReshapeConverter,
    MindSporeBroadcastConverter,
    MindSporeRoundConverter,
    MindSporeSelectConverter,
    MindSporeIsFiniteConverter,
    MindSporeInvConverter,
    MindSporeConstConverter
  >(patterns.getContext());
  // clang-format on
  return;
}

struct ConvertMindSporeToLinalgNamedPass : public ConvertMindSporeToLinalgNamedBase<ConvertMindSporeToLinalgNamedPass> {
 public:
  ConvertMindSporeToLinalgNamedPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<math::MathDialect>();
    // registry.insert<hacc::HACCDialect>();
    // registry.insert<hfusion::HFusionDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect, tensor::TensorDialect,
                           math::MathDialect>();  //, hfusion::HFusionDialect>();

    // func->setAttr("hacc.function_kind",
    //   hacc::HACCFuncTypeAttr::get(func->getContext(), hacc::HACCFuncType::HOST));

    populateLowerMindSporeToLinalgNamedPattern(patterns);
    populateLowerMindSporeCompareToLinalgPattern(patterns);
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createMindSporeToLinalgNamedPass() {
  return std::make_unique<ConvertMindSporeToLinalgNamedPass>();
}
}  // namespace mlir
