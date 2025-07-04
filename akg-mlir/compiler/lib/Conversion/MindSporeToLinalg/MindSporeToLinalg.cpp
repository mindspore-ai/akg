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

#include <numeric>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Conversion/Passes.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
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

static Value createAsinhOp(Operation *op, const ValueRange args, ArrayRef<Type> resultTypes,
                           PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto mulOp = rewriter.create<mlir::arith::MulFOp>(loc, resultTypes, args[0], args[0]);
  auto constOp = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(resultTypes[0], 1.0));
  auto addOp = rewriter.create<mlir::arith::AddFOp>(loc, resultTypes, constOp.getResult(), mulOp.getResult());
  auto sqrtOp = rewriter.create<mlir::math::SqrtOp>(loc, resultTypes, addOp.getResult());
  auto add2Op = rewriter.create<mlir::arith::AddFOp>(loc, resultTypes, args[0], sqrtOp.getResult());
  auto logOp = rewriter.create<mlir::math::LogOp>(loc, resultTypes, add2Op.getResult());
  return logOp;
}

static Value createAcoshOp(Operation *op, const ValueRange args, ArrayRef<Type> resultTypes,
                           PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto mulOp = rewriter.create<mlir::arith::MulFOp>(loc, resultTypes, args[0], args[0]);
  auto constOp = rewriter.create<mlir::arith::ConstantOp>(loc, rewriter.getFloatAttr(resultTypes[0], 1.0));
  auto subOp = rewriter.create<mlir::arith::SubFOp>(loc, resultTypes, mulOp.getResult(), constOp.getResult());
  auto sqrtOp = rewriter.create<mlir::math::SqrtOp>(loc, resultTypes, subOp.getResult());
  auto addOp = rewriter.create<mlir::arith::AddFOp>(loc, resultTypes, args[0], sqrtOp.getResult());
  auto logOp = rewriter.create<mlir::math::LogOp>(loc, resultTypes, addOp.getResult());
  return logOp;
}

static Value createMathOps(Operation *op, ValueRange args, ArrayRef<Type> resultTypes, PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy = op->getOperand(0).getType().cast<ShapedType>().getElementType();
  // mindspore::SinOp
  if (isa<mindspore::SinOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<mlir::math::SinOp>(loc, resultTypes, args);
  }

  // mindspore::CosOp
  if (isa<mindspore::CosOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<mlir::math::CosOp>(loc, resultTypes, args);
  }

  if (isa<mindspore::Atan2Op>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<mlir::math::Atan2Op>(loc, resultTypes, args);
  }

  if (isa<mindspore::AsinhOp>(op) && elementTy.isa<FloatType>()) {
    return createAsinhOp(op, args, resultTypes, rewriter);
  }

  if (isa<mindspore::AcoshOp>(op) && elementTy.isa<FloatType>()) {
    return createAcoshOp(op, args, resultTypes, rewriter);
  }
  return nullptr;
}

static Value createLinalgBodyCalculationForElementwiseOp(Operation *op, ValueRange args, ArrayRef<Type> resultTypes,
                                                         PatternRewriter &rewriter) {
  Location loc = op->getLoc();
  auto elementTy = op->getOperand(0).getType().cast<ShapedType>().getElementType();
  // mindspore::AddNOp
  if (isa<mindspore::AddNOp>(op) && elementTy.isa<FloatType>()) {
    Value add = rewriter.create<mlir::arith::AddFOp>(loc, resultTypes, args[0], args[1]);
    for (uint64_t i = 2; i < args.size(); i++) {
      add = rewriter.create<mlir::arith::AddFOp>(loc, resultTypes, add, args[i]);
    }
    return add;
  }

  if (auto val = createMathOps(op, args, resultTypes, rewriter)) {
    return val;
  }

  // mindspore::DivOp
  if (isa<mindspore::DivOp>(op)) {
    if (elementTy.isa<IntegerType>()) {
      return rewriter.create<arith::DivSIOp>(loc, resultTypes, args);
    }
    if (elementTy.isa<FloatType>()) {
      return rewriter.create<arith::DivFOp>(loc, resultTypes, args);
    }
  }

  if (isa<mindspore::SqrtOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<mlir::math::SqrtOp>(loc, resultTypes, args);
  }

  // MindSpore::LessOp
  if (isa<mindspore::LessOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, args[0], args[1]);
  }

  if (isa<mindspore::LessOp>(op) && elementTy.isSignlessInteger()) {
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, args[0], args[1]);
  }

  // MindSpore::LessEqualOp
  if (isa<mindspore::LessEqualOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE, args[0], args[1]);
  }

  if (isa<mindspore::LessEqualOp>(op) && elementTy.isSignlessInteger()) {
    return rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle, args[0], args[1]);
  }

  (void)rewriter.notifyMatchFailure(op, "unhandled op for linalg body calculation for elementwise op");
  return nullptr;
}

static LogicalResult elementwiseMatchAndRewriteHelper(Operation *operation, PatternRewriter &rewriter) {
  auto loc = operation->getLoc();

  assert(operation->getNumResults() == 1 && "All MindSpore elementwise ops should only return a single result.");

  auto results = operation->getResults();
  auto resultTy = operation->getResult(0).getType().dyn_cast<ShapedType>();
  if (!resultTy) {
    return rewriter.notifyMatchFailure(operation, "All results must be a shaped type");
  }

  unsigned rank = resultTy.getRank();

  // Construct the indexing maps needed for linalg.generic ops.
  SmallVector<Type> bodyArgTypes;
  auto oprds = operation->getOperands();
  (void)std::transform(oprds.begin(), oprds.end(), std::back_inserter(bodyArgTypes),
                       [](const Value &in) { return getElementTypeOrSelf(in.getType()); });

  SmallVector<Type> opResultTypes;
  SmallVector<Value> emptyTensors;

  SmallVector<Value> dynDims;
  dynDims.resize(results.front().getType().cast<ShapedType>().getRank());

  for (auto arg : oprds) {
    auto operandTy = arg.getType().cast<ShapedType>();
    for (int i = 0; i < operandTy.getRank(); i++) {
      if (operandTy.isDynamicDim((unsigned long)i) && !dynDims[(unsigned int)i]) {
        dynDims[i] = rewriter.create<tensor::DimOp>(loc, arg, (unsigned long)i);
      }
    }
  }

  SmallVector<Value> filteredDims = condenseValues(dynDims);

  for (auto result : results) {
    if (RankedTensorType rankedType = result.getType().dyn_cast<RankedTensorType>()) {
      emptyTensors.push_back(rewriter.create<tensor::EmptyOp>(loc, rankedType, filteredDims));
    } else {
      auto curResultTy = result.getType().template cast<ShapedType>();
      emptyTensors.push_back(
        rewriter.create<tensor::EmptyOp>(loc, curResultTy.getShape(), curResultTy.getElementType(), filteredDims));
    }
    opResultTypes.push_back(result.getType());
  }

  auto bodyResultTypes =
    llvm::to_vector<4>(llvm::map_range(emptyTensors, [](Value v) { return getElementTypeOrSelf(v); }));

  SmallVector<Value, 2> operands;
  SmallVector<AffineMap, 2> indexingMaps;
  indexingMaps.reserve(operation->getNumOperands() + bodyResultTypes.size());

  // Input indexing maps may be broadcasted.
  for (Value operand : oprds) {
    ShapedType type = operand.getType().cast<ShapedType>();
    if (type.getShape() == resultTy.getShape()) {
      operands.push_back(operand);
      indexingMaps.push_back(rewriter.getMultiDimIdentityMap(rank));
      continue;
    }

    SmallVector<int64_t, 4> newShape;
    SmallVector<AffineExpr, 4> affineExprs;
    newShape.reserve(type.getRank());
    for (const auto &it : llvm::enumerate(type.getShape())) {
      if (it.value() == resultTy.getDimSize(it.index())) {
        newShape.push_back(it.value());
        affineExprs.push_back(mlir::getAffineDimExpr(it.index(), rewriter.getContext()));
      }
    }

    if (newShape.size() != rank) {
      operand = rewriter.create<mindspore::ReshapeOp>(loc, RankedTensorType::get(newShape, type.getElementType()),
                                                      operand, rewriter.getDenseI64ArrayAttr(newShape));
    }

    operands.push_back(operand);
    indexingMaps.push_back(AffineMap::get(
      /*dimCount=*/rank, /*symbolCount=*/0, affineExprs, rewriter.getContext()));
  }

  indexingMaps.append(operation->getNumResults(), rewriter.getMultiDimIdentityMap(rank));

  bool didEncounterError = false;
  auto linalgOp = rewriter.create<linalg::GenericOp>(
    loc, opResultTypes, operands, emptyTensors, indexingMaps, getNParallelLoopsAttrs(rank),
    [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
      Value opResult = createLinalgBodyCalculationForElementwiseOp(
        operation, blockArgs.take_front(operation->getNumOperands()), bodyResultTypes, rewriter);
      if (!opResult) {
        didEncounterError = true;
        return;
      }
      (void)nestedBuilder.create<linalg::YieldOp>(loc, opResult);
    });

  if (didEncounterError) {
    return rewriter.notifyMatchFailure(operation, "unable to create linalg.generic body for elementwise op");
  }

  rewriter.replaceOp(operation, linalgOp.getResult(0));
  return success();
}

template <typename SrcOp>
class MindSporePointwiseConverter : public OpRewritePattern<SrcOp> {
 public:
  using OpRewritePattern<SrcOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SrcOp op, PatternRewriter &rewriter) const final {
    return elementwiseMatchAndRewriteHelper(op, rewriter);
  }
};

class MindSporeInplaceAssignConverter : public OpConversionPattern<mindspore::InplaceAssignOp> {
 public:
  using OpConversionPattern<mindspore::InplaceAssignOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mindspore::InplaceAssignOp op, OpAdaptor,
                                ConversionPatternRewriter &rewriter) const override {
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

class MindSporeAssignOpConverter : public OpRewritePattern<mindspore::AssignOp> {
 public:
  using OpRewritePattern<mindspore::AssignOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mindspore::AssignOp op, PatternRewriter &rewriter) const final {
    auto linalgCopyOp =
      rewriter
        .create<linalg::CopyOp>(op.getLoc(), op.getType(), op.getInput1(), op.getInput0(), ArrayRef<NamedAttribute>())
        .getResult(0);
    rewriter.replaceOp(op, linalgCopyOp);
    return success();
  }
};

class MindSporeGatherConverter : public OpConversionPattern<mindspore::GatherOp> {
 public:
  using OpConversionPattern<mindspore::GatherOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mindspore::GatherOp op, typename mindspore::GatherOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    Value data = op.getOperands()[0];
    Value indices = op.getOperands()[1];
    Value output = op.getOutput();
    auto indicesRank = indices.getType().cast<ShapedType>().getRank();
    auto outputTy = op.getType().template cast<ShapedType>();
    auto outputRank = output.getType().template cast<ShapedType>().getRank();
    // axis
    int64_t axis = op.getAxisAttr().getInt();
    if (axis < 0) {
      axis += data.getType().cast<ShapedType>().getRank();
    }
    // batchDims
    uint64_t batchDims = 0;
    if (op.getBatchDims() != std::nullopt) {
      batchDims = *op.getBatchDims();
    }
    if (batchDims != 0) {
      return rewriter.notifyMatchFailure(op, "BatchDims != 0 may not be expressed on linalg.generic");
    }
    // create outs emptyTensor
    Value emptyTensor;
    if (auto rankedTensorType = op.getType().template cast<RankedTensorType>()) {
      emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(), outputTy.getElementType(),
                                                     rankedTensorType.getEncoding());
    } else {
      emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(), outputTy.getElementType());
    }
    // create index_mapping attr
    SmallVector<AffineExpr> affineExprs;
    for (int i = 0; i < indicesRank; i++) {
      affineExprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }
    auto indicesMap = AffineMap::get(outputRank, 0, affineExprs, rewriter.getContext());
    auto outputMap = rewriter.getMultiDimIdentityMap(outputRank);
    SmallVector<AffineMap> indexingMaps({indicesMap, outputMap});

    auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, ArrayRef<Type>({output.getType()}), ValueRange{indices}, ValueRange{emptyTensor}, indexingMaps,
      getNParallelLoopsAttrs(outputRank), [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        auto sliceVal = blockArgs[0];
        SmallVector<Value> dataIndices;
        // out's shape = input shape[: axis] + indices[batchDims:] + input shape[: axis + 1]
        for (int i = 0; i < axis; i++) {
          dataIndices.push_back(rewriter.create<linalg::IndexOp>(nestedLoc, i));
        }

        Value sliceIdx = rewriter.create<arith::IndexCastOp>(nestedLoc, rewriter.getIndexType(), sliceVal);
        dataIndices.push_back(sliceIdx);

        for (int i = axis + indicesRank; i < outputRank; i++) {
          dataIndices.push_back(rewriter.create<linalg::IndexOp>(nestedLoc, i));
        }

        Value extract = rewriter.create<tensor::ExtractOp>(nestedLoc, data, ValueRange{dataIndices});
        (void)rewriter.create<linalg::YieldOp>(nestedLoc, extract);
      });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

class MindSporeUnsortedSegmentSumOpConverter : public OpConversionPattern<mindspore::UnsortedSegmentSumOp> {
 public:
  using OpConversionPattern<mindspore::UnsortedSegmentSumOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(mindspore::UnsortedSegmentSumOp op, typename mindspore::UnsortedSegmentSumOp::Adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    auto outputTy = op.getType().cast<RankedTensorType>();
    auto outputElementTy = outputTy.getElementType();
    auto zeroAttrEType = rewriter.getZeroAttr(outputElementTy);
    auto zeroEType = rewriter.create<arith::ConstantOp>(loc, zeroAttrEType);
    auto zeroICst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneICst = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto indicesTy = op.getSegmentIds().getType().dyn_cast<TensorType>();
    auto dataTy = op.getX().getType().dyn_cast<TensorType>();
    // Collapse acc and data reductiondims on 1D for tiling purpose.
    // We collapse index dimensions to improve pipelining opportunities.
    Value collapsedData;
    // if there is no need to collapse indices  Dims nor reduction dims
    if (indicesTy.getRank() == 1 && dataTy.getRank() - indicesTy.getRank() <= 1) {
      collapsedData = op.getX();
    } else {
      //  Collapse indices dims
      SmallVector<ReassociationIndices> dataReassociationMap;
      dataReassociationMap.push_back({llvm::to_vector(llvm::seq<int64_t>(0, indicesTy.getRank()))});
      // Collapse reduction dims
      dataReassociationMap.push_back({llvm::to_vector(llvm::seq<int64_t>(indicesTy.getRank(), dataTy.getRank()))});
      collapsedData = rewriter.create<tensor::CollapseShapeOp>(loc, op.getX(), dataReassociationMap);
    }
    Value collapsedIndex;
    if (indicesTy.getRank() == 1) {
      collapsedIndex = op.getSegmentIds();
    } else {
      SmallVector<ReassociationIndices> indexReassociationMap;
      indexReassociationMap.push_back({llvm::to_vector(llvm::seq<int64_t>(0, indicesTy.getRank()))});
      collapsedIndex = rewriter.create<tensor::CollapseShapeOp>(loc, op.getSegmentIds(), indexReassociationMap);
    }
    auto collapsedDataTy = collapsedData.getType().cast<RankedTensorType>();
    auto uniqSize = SmallVector<int64_t>({1});
    ArrayRef<int64_t> sliceShape = outputTy.getRank() == 1 ? uniqSize : collapsedDataTy.getShape().take_back();
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputTy, ValueRange{});
    Value zeroTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zeroEType}, ValueRange{emptyTensor}).result();
    Value collapsedResult;
    if (dataTy.getRank() - indicesTy.getRank() <= 1) {
      collapsedResult = zeroTensor;
    } else {
      auto collapsedAccShape = llvm::to_vector(outputTy.getShape().take_front());
      collapsedAccShape.push_back(sliceShape[0]);
      auto collapsedAccTy = outputTy.clone(collapsedAccShape).cast<RankedTensorType>();
      SmallVector<ReassociationIndices> accReassociationMap{{0}};
      // Associate reductionDims
      accReassociationMap.push_back({llvm::to_vector(llvm::seq<int64_t>(1, outputTy.getRank()))});
      collapsedResult = rewriter.create<tensor::CollapseShapeOp>(loc, collapsedAccTy, zeroTensor, accReassociationMap);
    }
    SmallVector<Value> lbs({zeroICst});
    SmallVector<Value> steps({oneICst});
    SmallVector<Value> ubs({rewriter.createOrFold<tensor::DimOp>(loc, collapsedIndex, 0)});

    // Acc/Data slice is 1 for the index dimension followed by the rest of the acc size
    SmallVector<Value> sliceSizes{oneICst};
    SmallVector<Value> accOffsets(1, zeroICst);
    SmallVector<Value> accStrides(1, oneICst);
    if (outputTy.getRank() != 1) {
      auto sliceSize = rewriter.createOrFold<tensor::DimOp>(loc, collapsedData, collapsedDataTy.getRank() - 1);
      sliceSizes.push_back(sliceSize);
      accOffsets.push_back(zeroICst);
      accStrides.push_back(oneICst);
    }

    auto loops = buildUSSLoopNest(rewriter, op, loc, lbs, ubs, steps, collapsedData, collapsedResult, collapsedIndex,
                                  sliceSizes, accStrides, accOffsets);
    // expand addOp back to result Shape
    Value expandAdd;
    if (dataTy.getRank() - indicesTy.getRank() <= 1) {
      expandAdd = loops.results.front();
    } else {
      SmallVector<ReassociationIndices> accReassociationMap{{0}};
      // Associate reductionDims
      accReassociationMap.push_back({llvm::to_vector(llvm::seq<int64_t>(1, outputTy.getRank()))});
      expandAdd = rewriter.create<tensor::ExpandShapeOp>(loc, outputTy, loops.results.front(), accReassociationMap);
    }
    rewriter.replaceOp(op, {expandAdd});
    return success();
  }

  scf::LoopNest buildUSSLoopNest(ConversionPatternRewriter &rewriter, mindspore::UnsortedSegmentSumOp op, Location loc,
                                 ValueRange lbs, ValueRange ubs, ValueRange steps, Value collapsedData,
                                 Value collapsedResult, Value collapsedIndex, SmallVectorImpl<Value> &sliceSizes,
                                 SmallVectorImpl<Value> &accStrides, SmallVectorImpl<Value> &accOffsets) const {
    auto outputTy = op.getType().cast<RankedTensorType>();
    auto zeroICst = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto oneICst = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto collapsedDataTy = collapsedData.getType().cast<RankedTensorType>();
    auto uniqSize = SmallVector<int64_t>({1});
    ArrayRef<int64_t> sliceShape = outputTy.getRank() == 1 ? uniqSize : collapsedDataTy.getShape().take_back();
    auto sliceTy = outputTy.clone(sliceShape).cast<RankedTensorType>();
    SmallVector<Value> dataOffsets(collapsedDataTy.getRank(), zeroICst);
    SmallVector<Value> dataStrides(collapsedDataTy.getRank(), oneICst);
    auto toOpFoldResult = [](Value v) -> OpFoldResult {
      auto op = v.getDefiningOp<arith::ConstantIndexOp>();
      if (!op) {
        return v;
      }
      return op.getValue();
    };

    auto buildBody = [&](OpBuilder &builder, Location loc, ValueRange ivs, ValueRange args) -> scf::ValueVector {
      // data offsets = [{{ivs, ...}}, 0].
      std::copy(ivs.begin(), ivs.end(), dataOffsets.begin());

      auto dataSlice = builder.create<tensor::ExtractSliceOp>(
        loc, sliceTy, collapsedData, llvm::to_vector(llvm::map_range(dataOffsets, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(sliceSizes, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(dataStrides, toOpFoldResult)));

      auto index = builder.create<tensor::ExtractOp>(loc, collapsedIndex, ivs);
      auto castIndex = builder.createOrFold<arith::IndexCastOp>(loc, builder.getIndexType(), index);
      accOffsets[0] = castIndex;
      auto accSlice = builder.create<tensor::ExtractSliceOp>(
        loc, sliceTy, args[0], llvm::to_vector(llvm::map_range(accOffsets, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(sliceSizes, toOpFoldResult)),
        llvm::to_vector(llvm::map_range(accStrides, toOpFoldResult)));
      // create an elementwise generic adding extracted acc and data
      // we do not generate tosa::AddOp as they generate expensive allocs
      SmallVector<AffineMap> indexingMaps = {rewriter.getMultiDimIdentityMap(1), rewriter.getMultiDimIdentityMap(1)};

      auto iteratorTypes = getNParallelLoopsAttrs(1);
      auto addOp =
        rewriter
          .create<linalg::GenericOp>(
            loc, TypeRange{sliceTy}, ValueRange{dataSlice}, ValueRange{accSlice}, indexingMaps, iteratorTypes,
            [](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
              Value opResult = nestedBuilder.create<arith::AddFOp>(nestedLoc, blockArgs[0], blockArgs[1]);
              nestedBuilder.create<linalg::YieldOp>(nestedLoc, opResult);
            })
          .getResult(0);
      return {builder.create<tensor::InsertSliceOp>(loc, addOp, args[0],
                                                    llvm::to_vector(llvm::map_range(accOffsets, toOpFoldResult)),
                                                    llvm::to_vector(llvm::map_range(sliceSizes, toOpFoldResult)),
                                                    llvm::to_vector(llvm::map_range(accStrides, toOpFoldResult)))};
    };
    return scf::buildLoopNest(rewriter, loc, lbs, ubs, steps, ValueRange{collapsedResult}, buildBody);
  }
};

template <typename sourceOp, typename targetOp>
class MindsporeSpecificOpConverter : public OpConversionPattern<sourceOp> {
 public:
  using OpConversionPattern<sourceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(sourceOp op, typename sourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Location loc = op.getLoc();
    // mindspore::GatherOp
    if (mindspore::GatherOp gatherOp = dyn_cast<mindspore::GatherOp>(op.getOperation())) {
      uint64_t batchDims = 0;
      if (gatherOp.getBatchDims() != std::nullopt) {
        batchDims = *gatherOp.getBatchDims();
      }
      if (batchDims != 0) {
        return rewriter.notifyMatchFailure(op, "BatchDims != 0 is unsupported temporarily");
      }
    }
    // mindspore::UnsortedSegmentSumOp
    if (isa<mindspore::UnsortedSegmentSumOp>(op.getOperation())) {
      Attribute numSegmentsAttr = dyn_cast<mindspore::UnsortedSegmentSumOp>(op.getOperation()).getNumSegmentsAttr();
      if (numSegmentsAttr.isa<DenseI64ArrayAttr>()) {
        return rewriter.notifyMatchFailure(op, "num_segments as tensor is unsupported temporarily");
      }
    }

    auto outputTy = op.getType().template cast<ShapedType>();

    // create outs emptyTensor
    Value emptyTensor;
    if (auto rankedTensorType = op.getType().template cast<RankedTensorType>()) {
      emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(), outputTy.getElementType(),
                                                     rankedTensorType.getEncoding());
    } else {
      emptyTensor = rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(), outputTy.getElementType());
    }

    // append emptyTensor to output's operands.
    SmallVector<Value> operands(adaptor.getOperands());
    operands.push_back(emptyTensor);
    // create targetOp
    (void)rewriter.replaceOpWithNewOp<targetOp>(op, outputTy, operands, op->getAttrs());
    return success();
  }
};

static TypedAttr createIntInitValue(Operation *op, const Type elementTy, PatternRewriter &rewriter) {
  if (isa<mindspore::ReduceSumOp>(op)) {
    return rewriter.getIntegerAttr(elementTy, 0);
  }
  if (isa<mindspore::ReduceAnyOp>(op)) {
    return rewriter.getIntegerAttr(elementTy, 0);
  }
  if (isa<mindspore::ReduceProdOp>(op)) {
    return rewriter.getIntegerAttr(elementTy, 1);
  }
  if (isa<mindspore::ReduceMinOp>(op)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMaxValue(elementTy.getIntOrFloatBitWidth()));
  }
  if (isa<mindspore::ReduceMaxOp>(op)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getSignedMinValue(elementTy.getIntOrFloatBitWidth()));
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static TypedAttr createFloatInitValue(Operation *op, const Type elementTy, PatternRewriter &rewriter) {
  if (isa<mindspore::ReduceSumOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.getFloatAttr(elementTy, 0.0);
  }
  if (isa<mindspore::ReduceProdOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.getFloatAttr(elementTy, 1.0);
  }
  if (isa<mindspore::ReduceMinOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.getFloatAttr(elementTy,
                                 APFloat::getLargest(elementTy.cast<FloatType>().getFloatSemantics(), false));
  }
  if (isa<mindspore::ReduceMaxOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(elementTy.cast<FloatType>().getFloatSemantics(), true));
  }
  if (isa<mindspore::ReduceAnyOp>(op) && elementTy.isa<FloatType>()) {
    return rewriter.getFloatAttr(elementTy, APFloat::getLargest(elementTy.cast<FloatType>().getFloatSemantics(), true));
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static TypedAttr createInitValueForReduce(Operation *op, const Type elementTy, PatternRewriter &rewriter) {
  if (isa<mindspore::ReduceAllOp>(op) && elementTy.isInteger(1)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getAllOnes(1));
  }
  if (isa<mindspore::ReduceAnyOp>(op) && elementTy.isInteger(1)) {
    return rewriter.getIntegerAttr(elementTy, APInt::getZero(1));
  }
  if (elementTy.isa<FloatType>()) {
    return createFloatInitValue(op, elementTy, rewriter);
  }
  if (elementTy.isa<IntegerType>()) {
    return createIntInitValue(op, elementTy, rewriter);
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static Value createIntCombiner(Operation *op, ValueRange args, OpBuilder &rewriter) {
  Location loc = op->getLoc();
  if (isa<mindspore::ReduceProdOp>(op)) {
    return rewriter.create<arith::MulIOp>(loc, args);
  }
  if (isa<mindspore::ReduceMinOp>(op)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }
  if (isa<mindspore::ReduceSumOp>(op)) {
    return rewriter.create<arith::AddIOp>(loc, args);
  }
  if (isa<mindspore::ReduceAnyOp>(op)) {
    return rewriter.create<arith::OrIOp>(loc, args);
  }
  if (isa<mindspore::ReduceMaxOp>(op)) {
    auto predicate = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, args[0], args[1]);
    return rewriter.create<arith::SelectOp>(loc, predicate, args[0], args[1]);
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static Value createFloatCombiner(Operation *op, ValueRange args, OpBuilder &rewriter) {
  Location loc = op->getLoc();
  if (isa<mindspore::ReduceProdOp>(op)) {
    return rewriter.create<arith::MulFOp>(loc, args);
  }
  if (isa<mindspore::ReduceMinOp>(op)) {
    return rewriter.create<arith::MinNumFOp>(loc, args[0], args[1]);
  }
  if (isa<mindspore::ReduceSumOp>(op)) {
    return rewriter.create<arith::AddFOp>(loc, args);
  }
  if (isa<mindspore::ReduceMaxOp>(op)) {
    return rewriter.create<arith::MaxNumFOp>(loc, args[0], args[1]);
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static Value createCombinerForReduce(Operation *op, ValueRange args, const Type elementTy, OpBuilder &rewriter) {
  Location loc = op->getLoc();
  if (isa<mindspore::ReduceAllOp>(op) && elementTy.isInteger(1)) {
    return rewriter.create<arith::AndIOp>(loc, args);
  }

  if (isa<mindspore::ReduceAnyOp>(op) && elementTy.isInteger(1)) {
    return rewriter.create<arith::OrIOp>(loc, args);
  }

  if (elementTy.isa<FloatType>()) {
    return createFloatCombiner(op, args, rewriter);
  }
  if (elementTy.isa<IntegerType>()) {
    return createIntCombiner(op, args, rewriter);
  }
  llvm_unreachable("current reduce pattern is not supported");
}

static bool isRedundantReduce(SmallVector<int64_t> axes, const ShapedType inputTy) {
  bool is_redundant_reduce = true;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    if (llvm::is_contained(axes, i) && (inputTy.getShape()[i]) != 1) {
      is_redundant_reduce = false;
    }
  }
  return is_redundant_reduce;
}

static bool isAllReduce(SmallVector<int64_t> axes, const ShapedType inputTy) {
  bool is_all_reduce = true;
  for (int64_t i = 0; i < inputTy.getRank(); i++) {
    if (!llvm::is_contained(axes, i) && (inputTy.getShape()[i]) != 1) {
      is_all_reduce = false;
    }
  }
  return is_all_reduce;
}

static SmallVector<ReassociationExprs> getExpandMap(SmallVector<int64_t> axes, int64_t expandInputRank,
                                                    int64_t expandOutputRank, ConversionPatternRewriter &rewriter) {
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

static SmallVector<int64_t> inferReduceOutShape(SmallVector<int64_t> axes, const ShapedType inputTy) {
  SmallVector<int64_t> reduceOutShape;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (!llvm::is_contained(axes, int64_t(i))) {
      reduceOutShape.push_back(inputTy.getDimSize(i));
    }
  }
  return reduceOutShape;
}

SmallVector<Value> inferDynDims(SmallVector<int64_t> axes, const ShapedType inputTy, const Location loc, Value opnd,
                                ConversionPatternRewriter &rewriter) {
  SmallVector<Value> dynDims;
  for (unsigned i = 0; i < inputTy.getRank(); i++) {
    if (!llvm::is_contained(axes, int64_t(i)) && inputTy.isDynamicDim(i)) {
      dynDims.push_back(rewriter.create<tensor::DimOp>(loc, opnd, i));
    }
  }
  return dynDims;
}

class ConvertMindSporeTileOp : public OpRewritePattern<mindspore::TileOp> {
 public:
  using OpRewritePattern<mindspore::TileOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(mindspore::TileOp op, PatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    Value input = op.getInput();
    auto inputTy = input.getType().cast<ShapedType>();
    auto inputShape = inputTy.getShape();
    auto elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();
    uint64_t newRankSize = op.getNewRankSize();
    assert(newRankSize >= (uint64_t)inputTy.getRank());
    auto resultTy = op.getType().cast<ShapedType>();
    ArrayRef<int64_t> multiples = op.getMultiples();

    // todo(xinkai): reuse the tosa.tile op's lower pattern temporarily. In fact, the semantics of mindspore.tile
    // and tosa.tile are slightly different.
    if (newRankSize > (uint64_t)rank) {
      llvm_unreachable("Now in TileOp, The new rank size must equal to the input's rank.");
    }

    // Broadcast the newly added dimensions to their appropriate multiple.
    SmallVector<int64_t, 2> genericShape;
    for (int i = 0; i < rank; i++) {
      int64_t dim = multiples[i];
      genericShape.push_back(dim == -1 ? ShapedType::kDynamic : dim);
      genericShape.push_back(inputShape[i]);
    }

    SmallVector<Value> dynDims;
    for (int i = 0; i < inputTy.getRank(); i++) {
      if (inputTy.isDynamicDim(i) || multiples[i] == -1) {
        dynDims.push_back(rewriter.create<tensor::DimOp>(loc, input, i));
      }
    }

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), genericShape, elementTy, dynDims);

    // We needs to map the input shape to the non-broadcasted dimensions.
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);
    for (unsigned i = 0; i < rank; ++i) {
      const uint32_t indexThreshold = 2;
      dimExprs.push_back(rewriter.getAffineDimExpr(i * indexThreshold + 1));
    }
    auto readAffineMap = AffineMap::get(/*dimCount=*/rank * 2, /*symbolCount=*/0, dimExprs, rewriter.getContext());

    SmallVector<AffineMap, 2> affineMaps = {readAffineMap, rewriter.getMultiDimIdentityMap(genericShape.size())};

    auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, RankedTensorType::get(genericShape, elementTy), input, ValueRange{emptyTensor}, affineMaps,
      getNParallelLoopsAttrs(genericShape.size()), [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        (void)nestedBuilder.create<linalg::YieldOp>(op.getLoc(), *args.begin());
      });

    (void)rewriter.replaceOpWithNewOp<mindspore::ReshapeOp>(op, resultTy, genericOp.getResult(0),
                                                            rewriter.getDenseI64ArrayAttr(resultTy.getShape()));
    return success();
  }
};

class ConvertMindSporeBroadcastToOp : public OpRewritePattern<mindspore::BroadcastToOp> {
 public:
  using OpRewritePattern<mindspore::BroadcastToOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mindspore::BroadcastToOp op, PatternRewriter &rewriter) const final {
    // mindspore.tile
    MLIRContext *context = rewriter.getContext();
    auto loc = op.getLoc();
    Value input = op.getInput();
    auto inputTy = input.getType().cast<ShapedType>();
    auto inputElementTy = inputTy.getElementType();
    assert(inputTy.getRank() > 0);
    uint64_t newRankSize = op.getNewRankSize();
    assert(newRankSize >= (uint64_t)inputTy.getRank());

    // new shape as value
    if (op.getNewShapeValue()) {
      return broadcastToOpLowerHelper(op, rewriter);
    }

    // new shape as attr
    // broadcasts input tensor to a given shape. The dim of input shape must be smaller than or equal
    // to that of target shape. Suppose input shape is (x1,x2,...,xm), target shape is (*, y1,y2,...,ym),
    // where * means any additional dimension.
    llvm::ArrayRef<int64_t> newShape = *(op.getNewShape());
    llvm::ArrayRef<int64_t> inputShape = inputTy.getShape();
    // 1.If the value pairs at a specific dim are equal, then that value goes right into
    // that dim of output shape.
    if (newShape == inputShape) {
      rewriter.replaceOp(op, input);
      return success();
    }
    // 2.If additional dimensions exist, then insert reshapeOp
    uint64_t extraDimSize = newRankSize - (uint64_t)inputTy.getRank();
    if (extraDimSize > 0) {
      llvm::SmallVector<int64_t> newInputShape(extraDimSize, 1);
      (void)std::transform(inputTy.getShape().begin(), inputTy.getShape().end(), std::back_inserter(newInputShape),
                           [](const int64_t &dim) { return dim; });
      NamedAttribute reshapeAttr = NamedAttribute(StringAttr::get(context, "new_shape"),
                                                  DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(newInputShape)));
      RankedTensorType reshapeTy = RankedTensorType::get(newInputShape, inputElementTy);
      auto reshape = rewriter.create<mindspore::ReshapeOp>(loc, reshapeTy, input, reshapeAttr);
      input = reshape.getResult();
      inputShape = input.getType().cast<ShapedType>().getShape();
    }
    // 3.broadcasts input tensor to a given shape
    llvm::SmallVector<int64_t> multiples;
    for (uint64_t i = 0; i < newRankSize; i++) {
      // If the value of the target shape in the dimension is -1, the value of the output shape in the
      // dimension is the value of the corresponding input shape in the dimension.
      int64_t multiple = (newShape[i] == -1) ? 1 : newShape[i] / inputShape[i];
      (void)multiples.emplace_back(multiple);
    }
    NamedAttribute multiplesAttr = NamedAttribute(StringAttr::get(context, "multiples"),
                                                  DenseI64ArrayAttr::get(context, ArrayRef<int64_t>(multiples)));
    RankedTensorType ty = RankedTensorType::get(newShape, inputElementTy);
    auto tileOp = rewriter.create<mindspore::TileOp>(loc, ty, input, multiplesAttr);
    rewriter.replaceOp(op, tileOp.getResult());
    return success();
  }

 private:
  // This converter translate a BroadcastToOp with Value to a expand, broadcast, and
  // collapse. First, expands the dimension which needs to be broadcast to 2 dims.
  // This dim is then broadcasted to the appropriate multiple.
  // Finally. collapse the result into dynamic tile operation's output.
  LogicalResult broadcastToOpLowerHelper(mindspore::BroadcastToOp op, PatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputTy = input.getType().cast<ShapedType>();
    auto inputShape = inputTy.getShape();
    auto resultTy = op.getType().cast<ShapedType>();
    auto resultShape = resultTy.getShape();
    auto elementTy = inputTy.getElementType();
    int64_t rank = inputTy.getRank();
    Value newShape = op.getNewShapeValue();
    uint64_t newRankSize = op.getNewRankSize();
    assert(newRankSize >= (uint64_t)inputTy.getRank());

    // create emptyOp and genericOp
    // genericShape : emptyOp and genericOp's shape
    SmallVector<int64_t, 2> genericShape;
    SmallVector<Value> dynDims;
    // indexMaping Of input
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(rank);

    // create collapseShape Op
    // reassociationMap of tensor.collapse_shape Op
    SmallVector<ReassociationIndices, 4> reassociationMap;
    // collapseShape Op's output shape.
    SmallVector<int64_t, 2> collapseShape;
    // if collapseShape != genericShape, tensor.castOp must created.
    bool needCastOp = false;
    // lower:
    // %out = "tosa.dyn_tile"(%in0, %in1) : (tensor<32x12xs2xs3xf16>,
    // tensor<4xindex>) -> tensor<32x12xs2xs5xf16> TO %c2 = arith.constant 2 :
    // index %c3 = arith.constant 3 : index %dim_2 = tensor.dim %in0, %c2 :
    // tensor<32x12xs2xs3xf16> %broadcasted_3 = shape.get_extent %in1, %c3 :
    // tensor<4xindex>, index -> index %dim_3 = tensor.dim %in0, %c3 :
    // tensor<32x12xs2xs3xf16> %multiple_3 = arith.divsi %broadcasted_3, %dim_3
    // : index dynDims : %dim_2, %multiple_3, %dim_3 genericShape :
    // tensor<32x12x?x?x?xf16> %emptyOp = tensor.empty(%dim_2, %multiple_3,
    // %dim_3) : tensor<32x12x?x?x?xf16> #map_in = affine_map<(d0, d1, d2, d3,
    // d4) -> (d0, d1, d2, d4)> #map_out = affine_map<(d0, d1, d2, d3, d4) ->
    // (d0, d1, d2, d3, d4)> %genericOp = linalg.generic {indexing_maps =
    // [#map_in, #map_out], iterator_types = ["parallel", "parallel",
    //                             "parallel", "parallel", "parallel"]}
    //                              ins(%in0 : tensor<32x12xs2xs3xf16>)
    //                              outs(%emptyOp : tensor<32x12x?x?x?xf16>) {
    // ^bb0(%in: f16, %out: f16):
    //    linalg.yield %in : f16
    // } -> tensor<32x12x?x?x?xf16>
    // reassociationMap: [[0], [1], [2], [3, 4]]
    // %collapsed = tensor.collapse_shape %genericOp [[0], [1], [2], [3, 4]] :
    // tensor<32x12x?x?x?xf16>
    //    into tensor<32x12xs2xs5xf16>
    int64_t dimIndices = 0;
    for (int i = 0; i < rank; i++) {
      int64_t outDimSize = resultShape[i];
      int64_t inputDimSize = inputShape[i];
      // if outDimSize = 1, it means inputDimSize MUST be 1;
      if (outDimSize == 1) {
        assert(inputDimSize == 1);
        genericShape.push_back(inputDimSize);
        collapseShape.push_back(inputDimSize);
        ReassociationIndices indices({dimIndices});
        reassociationMap.push_back(indices);
        dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices));
        dimIndices++;
        continue;
      }

      if (outDimSize > 1) {
        if (inputDimSize == ShapedType::kDynamic) {
          genericShape.push_back(ShapedType::kDynamic);
          genericShape.push_back(inputShape[i]);
          collapseShape.push_back(ShapedType::kDynamic);
          Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
          Value multi = rewriter.create<arith::DivSIOp>(loc, rewriter.getIndexType(),
                                                        rewriter.create<arith::ConstantIndexOp>(loc, outDimSize), dim);
          dynDims.push_back(multi);
          dynDims.push_back(dim);
          ReassociationIndices indices({dimIndices, dimIndices + 1});
          reassociationMap.push_back(indices);
          dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices + 1));
          dimIndices++;
          dimIndices++;
          needCastOp = true;
          continue;
        }
        if (inputDimSize == outDimSize) {
          genericShape.push_back(outDimSize);
          collapseShape.push_back(outDimSize);
          ReassociationIndices indices({dimIndices});
          reassociationMap.push_back(indices);
          dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices));
          dimIndices++;
          continue;
        }
        assert(inputDimSize >= 1 && inputDimSize < outDimSize);
        genericShape.push_back(outDimSize / inputDimSize);
        genericShape.push_back(inputDimSize);
        collapseShape.push_back(outDimSize);
        ReassociationIndices indices({dimIndices, dimIndices + 1});
        reassociationMap.push_back(indices);
        dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices + 1));
        dimIndices++;
        dimIndices++;
        continue;
      }

      // default: output type is dynamic
      if (inputDimSize == 1) {
        genericShape.push_back(ShapedType::kDynamic);
        genericShape.push_back(inputDimSize);
        collapseShape.push_back(ShapedType::kDynamic);
        Value multi = rewriter.create<shape::GetExtentOp>(loc, newShape, i);
        dynDims.push_back(multi);
        ReassociationIndices indices({dimIndices, dimIndices + 1});
        reassociationMap.push_back(indices);
        dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices + 1));
        dimIndices++;
        dimIndices++;
        continue;
      }
      if (inputDimSize > 1) {
        genericShape.push_back(ShapedType::kDynamic);
        genericShape.push_back(inputShape[i]);
        collapseShape.push_back(ShapedType::kDynamic);
        Value multi = rewriter.create<shape::GetExtentOp>(loc, newShape, i);
        multi = rewriter.create<arith::DivSIOp>(loc, rewriter.getIndexType(), multi,
                                                rewriter.create<arith::ConstantIndexOp>(loc, inputDimSize));
        dynDims.push_back(multi);
        ReassociationIndices indices({dimIndices, dimIndices + 1});
        reassociationMap.push_back(indices);
        dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices + 1));
        dimIndices++;
        dimIndices++;
        continue;
      }
      // input type and output type are both dynamic shape
      SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
      if (analysis.isSameSymbolicShape(op.getType(), input.getType())) {
        genericShape.push_back(ShapedType::kDynamic);
        collapseShape.push_back(ShapedType::kDynamic);
        Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
        dynDims.push_back(dim);
        ReassociationIndices indices({dimIndices});
        reassociationMap.push_back(indices);
        dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices));
        dimIndices++;
        continue;
      }
      // expands each tiled dimension
      genericShape.push_back(ShapedType::kDynamic);
      genericShape.push_back(inputShape[i]);
      collapseShape.push_back(ShapedType::kDynamic);
      Value dim = rewriter.create<tensor::DimOp>(loc, input, i);
      Value multi = rewriter.create<shape::GetExtentOp>(loc, newShape, i);
      multi = rewriter.create<arith::DivSIOp>(loc, rewriter.getIndexType(), multi, dim);
      dynDims.push_back(multi);
      dynDims.push_back(dim);
      ReassociationIndices indices({dimIndices, dimIndices + 1});
      reassociationMap.push_back(indices);
      dimExprs.push_back(rewriter.getAffineDimExpr(dimIndices + 1));
      dimIndices++;
      dimIndices++;
      continue;
    }

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), genericShape, elementTy, dynDims);
    // We needs to map the input shape to the non-broadcasted dimensions.
    auto readAffineMap = AffineMap::get(/*dimCount=*/dimIndices, /*symbolCount=*/0, dimExprs, rewriter.getContext());
    assert((int64_t)genericShape.size() == dimIndices);
    SmallVector<AffineMap, 2> affineMaps = {readAffineMap, rewriter.getMultiDimIdentityMap(genericShape.size())};

    auto genericOp = rewriter.create<linalg::GenericOp>(
      loc, RankedTensorType::get(genericShape, elementTy), input, ValueRange{emptyTensor}, affineMaps,
      getNParallelLoopsAttrs(genericShape.size()), [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        (void)nestedBuilder.create<linalg::YieldOp>(op.getLoc(), *args.begin());
      });
    if (needCastOp) {
      auto collapseShapeOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, RankedTensorType::get(collapseShape, elementTy), genericOp.getResult(0), reassociationMap);
      collapseShapeOp.getOperation()->setAttr("AttachDynTile", UnitAttr::get(rewriter.getContext()));
      (void)rewriter.replaceOpWithNewOp<tensor::CastOp>(op, resultTy, collapseShapeOp.getResult());
      return success();
    }
    auto collapseShapeOp =
      rewriter.create<tensor::CollapseShapeOp>(loc, resultTy, genericOp.getResult(0), reassociationMap);
    collapseShapeOp.getOperation()->setAttr("AttachDynTile", UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOp(op.getOperation(), collapseShapeOp.getResult());
    return success();
  }
};

// reduce ops
template <typename sourceOp>
class ConvertMindSporeReduceOp : public OpConversionPattern<sourceOp> {
 public:
  using OpConversionPattern<sourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(sourceOp mindsporeOp, typename sourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    Operation *op = mindsporeOp;
    Value opnd = adaptor.getInput();
    auto loc = op->getLoc();
    auto resultTy = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    auto resultElementTy = resultTy.getElementType();
    auto encoding = resultTy.getEncoding();
    auto inputTy = opnd.getType().cast<RankedTensorType>();

    SmallVector<int64_t> axes(adaptor.getAxis().begin(), adaptor.getAxis().end());

    bool keep_dims = false;
    if (adaptor.getKeepdimsAttr()) {
      keep_dims = adaptor.getKeepdimsAttr().getValue();
    }

    bool is_redundant_reduce = isRedundantReduce(axes, inputTy);
    bool is_all_reduce = isAllReduce(axes, inputTy);

    if (is_redundant_reduce && keep_dims) {
      rewriter.replaceOp(op, opnd);
      return success();
    }

    std::string process;
    if (op->getParentOp()->getAttr("process")) {
      process = op->getParentOp()->getAttr("process").dyn_cast<StringAttr>().getValue().str();
    } else {
      process = "cpu";
      emitWarning(op->getParentOp()->getLoc()) << " Cannot find processing type (cpu or cuda). Default is cpu. \n";
    }
    if (is_all_reduce && (inputTy.getRank() != 1) && (process == "cpu")) {
      int64_t total_size =
        std::accumulate(inputTy.getShape().begin(), inputTy.getShape().end(), 1, std::multiplies<int64_t>());
      Type collapseTy = RankedTensorType::get(total_size, resultTy.getElementType(), encoding);
      SmallVector<ReassociationExprs> collapse_map = {};
      ReassociationExprs expand_strategy;
      for (int64_t i = 0; i < inputTy.getRank(); i++) {
        expand_strategy.push_back(rewriter.getAffineDimExpr(i));
      }
      collapse_map.push_back(expand_strategy);
      auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(loc, collapseTy, opnd, collapse_map);
      opnd = collapseOp.getResult();
      inputTy = opnd.getType().cast<RankedTensorType>();
      axes = {0};
    }

    SmallVector<int64_t> reduceOutShape = inferReduceOutShape(axes, inputTy);
    SmallVector<Value> dynDims = inferDynDims(axes, inputTy, loc, opnd, rewriter);
    Type reduceTy = RankedTensorType::get(reduceOutShape, resultTy.getElementType(), encoding);
    // ElemAny is a type of reduceAnyOp in special scenarios. It's input and result have different element types,
    // and all it's axis are "reduction" types.
    // Example:
    //        %3 = "mindspore.reduce_any"(%2) {axis = array<i64: 0, 1>, keepdims = true, ori_op = "ElemAny"} :
    //          (tensor<4096x1024xi1>) -> tensor<1xf32>
    // ElemAny pre-processing : Cast input to newType(I32Type) and create linalg.genericOp based on the newTpe.
    if (isa<mindspore::ReduceAnyOp>(op) && op->getAttr("ori_op").cast<StringAttr>().str() == "ElemAny") {
      inputTy = RankedTensorType::get(inputTy.getShape(), rewriter.getI32Type(), inputTy.getEncoding());
      opnd = rewriter.create<tosa::CastOp>(op->getLoc(), inputTy, opnd);
      reduceTy = RankedTensorType::get(reduceOutShape, inputTy.getElementType(), encoding);
      resultElementTy = inputTy.getElementType();
    }
    // First create the init tensor for linalg.reduce
    auto emptyTensor =
      rewriter.create<tensor::EmptyOp>(loc, reduceOutShape, resultElementTy, dynDims, encoding).getResult();
    auto initValueAttr = createInitValueForReduce(op, resultElementTy, rewriter);
    auto initValue = rewriter.create<arith::ConstantOp>(loc, initValueAttr);
    auto initTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{initValue}, ValueRange{emptyTensor}).result();
    // create the map linalg.reduce
    SmallVector<utils::IteratorType> iteratorTypes;
    SmallVector<AffineExpr> reduceInputExprs;
    SmallVector<AffineExpr> reduceOutputExprs;
    for (int64_t i = 0; i < inputTy.getRank(); i++) {
      reduceInputExprs.push_back(mlir::getAffineDimExpr((unsigned int)i, rewriter.getContext()));
      iteratorTypes.push_back(llvm::is_contained(axes, i) ? utils::IteratorType::reduction
                                                          : utils::IteratorType::parallel);
      if (!llvm::is_contained(axes, i)) {
        reduceOutputExprs.push_back(mlir::getAffineDimExpr((unsigned int)i, rewriter.getContext()));
      }
    }
    auto maps = AffineMap::inferFromExprList({reduceInputExprs, reduceOutputExprs}, rewriter.getContext());
    auto ReductionOp = rewriter.create<linalg::GenericOp>(
      loc, reduceTy, opnd, initTensor, maps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location, const ValueRange blockArgs) {
        auto res = createCombinerForReduce(op, blockArgs, resultElementTy, nestedBuilder);
        (void)nestedBuilder.create<linalg::YieldOp>(loc, res);
      });
    opnd = ReductionOp.getResult(0);
    // ElemAny post-processing : Cast ReductionOp's result back to ElemAny's type.
    if (isa<mindspore::ReduceAnyOp>(op) && op->getAttr("ori_op").cast<StringAttr>().str() == "ElemAny") {
      auto newTy = RankedTensorType::get(reduceOutShape, resultTy.getElementType(), encoding);
      opnd = rewriter.create<tosa::CastOp>(op->getLoc(), newTy, opnd);
    }
    int64_t expandInputRank = initTensor.getType().cast<ShapedType>().getRank();
    int64_t expandOutputRank = resultTy.getRank();
    if (keep_dims && (expandInputRank != 0)) {  // keepdims, dim >= 1 after reduce, expand to output
                                                // shape
      SmallVector<ReassociationExprs> reassociation_map =
        getExpandMap(axes, expandInputRank, expandOutputRank, rewriter);
      auto expandOp = rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, opnd, reassociation_map);
      rewriter.replaceOp(op, expandOp.getResult());
    } else if (expandInputRank == 0) {  // dim = 0 after reduce, expand to output shape. e.g. f32->1xf32, f32->1x1xf32
      SmallVector<ReassociationExprs> reassociation_map = {};
      auto expandOp = rewriter.create<tensor::ExpandShapeOp>(loc, resultTy, opnd, reassociation_map);
      rewriter.replaceOp(op, expandOp.getResult());
    } else {  // dim >= 1 after reduce, keepdims = false
      rewriter.replaceOp(op, opnd);
    }

    return success();
  }
};

template <typename SrcOp>
class MindsporeMatMulOpConverter : public OpConversionPattern<SrcOp> {
 public:
  using OpConversionPattern<SrcOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(SrcOp mindsporeOp, typename SrcOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const final {
    auto transposeA = false;
    if (adaptor.getTransposeAAttr()) {
      transposeA = adaptor.getTransposeAAttr().getValue();
    }
    auto transposeB = false;
    if (adaptor.getTransposeBAttr()) {
      transposeB = adaptor.getTransposeBAttr().getValue();
    }

    Operation *op = mindsporeOp;
    auto output = op->getResult(0);
    auto outputTy = output.getType().cast<ShapedType>();
    auto outputRank = outputTy.getShape().size();
    const size_t nonBatchAxisNum = 2;
    size_t batchNum = outputRank - nonBatchAxisNum;
    // If transposeA is true, inputA shape is [K,M] or [B,K,M], else is [M,K] or [B,M,K]
    auto locMInInput = int(batchNum) + int(transposeA);
    // If transposeB is true, inputB shape is [N,K] or [B,N,K], else is [K,N] or [B,K,N]
    auto locNInInput = int(batchNum) + int(!transposeB);

    auto firstOperandTy = op->getOperand(0).getType().cast<ShapedType>();
    auto secondOperandTy = op->getOperand(1).getType().cast<ShapedType>();
    Location loc = op->getLoc();

    SmallVector<Value> dynDims;
    dynDims.resize(outputRank);
    for (size_t locBatch = 0; locBatch < batchNum; locBatch++) {
      if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(static_cast<unsigned int>(locBatch))) {
        dynDims[locBatch] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), int(locBatch));
      }
    }

    if (!firstOperandTy.hasRank() || firstOperandTy.isDynamicDim(static_cast<unsigned int>(locMInInput))) {
      dynDims[batchNum] = rewriter.create<tensor::DimOp>(loc, op->getOperand(0), locMInInput);
    }

    if (!secondOperandTy.hasRank() || secondOperandTy.isDynamicDim(static_cast<unsigned int>(locNInInput))) {
      dynDims[batchNum + 1] = rewriter.create<tensor::DimOp>(loc, op->getOperand(1), locNInInput);
    }

    SmallVector<Value> filteredDims = condenseValues(dynDims);

    auto outputElementTy = outputTy.getElementType();
    auto zeroAttr = rewriter.getZeroAttr(outputElementTy);
    Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto emptyTensor =
      rewriter.create<tensor::EmptyOp>(loc, outputTy.getShape(), outputTy.getElementType(), filteredDims);
    Value zeroTensor = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyTensor}).result();

    if (transposeA) {
      return rewriter.notifyMatchFailure(op, "TransposeA is not supported yet in linalg matmul/batchmatmul ops");
    }

    constexpr auto kMatmulRank = 2;
    constexpr auto kBatchMatmulRank = 3;
    switch (int(transposeB)) {
      case 0:
        switch (outputRank) {
          case kMatmulRank:
            return ReplaceMatMulOp<linalg::MatmulOp>(mindsporeOp, adaptor, rewriter, zeroTensor);
          case kBatchMatmulRank:
            return ReplaceMatMulOp<linalg::BatchMatmulOp>(mindsporeOp, adaptor, rewriter, zeroTensor);
          default:
            return failure();
        }
      case 1:
        switch (outputRank) {
          case kMatmulRank:
            return ReplaceMatMulOp<linalg::MatmulTransposeBOp>(mindsporeOp, adaptor, rewriter, zeroTensor);
          case kBatchMatmulRank:
            return ReplaceMatMulOp<linalg::BatchMatmulTransposeBOp>(mindsporeOp, adaptor, rewriter, zeroTensor);
          default:
            return failure();
        }
      default:
        break;
    }
    return success();
  }

 private:
  template <typename LinalgMatMulOp>
  LogicalResult ReplaceMatMulOp(SrcOp mindsporeOp, typename SrcOp::Adaptor adaptor, ConversionPatternRewriter &rewriter,
                                Value zeroTensor) const {
    (void)rewriter.replaceOpWithNewOp<LinalgMatMulOp>(mindsporeOp, TypeRange{mindsporeOp.getType()},
                                                      ValueRange{adaptor.getInputA(), adaptor.getInputB()},
                                                      ValueRange{zeroTensor});
    return success();
  }
};

void mlir::populateLowerMindSporeToLinalgPattern(RewritePatternSet &patterns) {
  // clang-format off
  (void)patterns.add<
    MindSporePointwiseConverter<mindspore::SinOp>,
    MindSporePointwiseConverter<mindspore::CosOp>,
    MindSporePointwiseConverter<mindspore::DivOp>,
    MindSporePointwiseConverter<mindspore::AcosOp>,
    MindSporePointwiseConverter<mindspore::AcoshOp>,
    MindSporePointwiseConverter<mindspore::AsinOp>,
    MindSporePointwiseConverter<mindspore::AsinhOp>,
    MindSporePointwiseConverter<mindspore::IsinfOp>,
    MindSporePointwiseConverter<mindspore::IsnanOp>,
    MindSporePointwiseConverter<mindspore::AddNOp>,
    MindSporePointwiseConverter<mindspore::SqrtOp>,
    MindSporePointwiseConverter<mindspore::LessEqualOp>,
    MindSporePointwiseConverter<mindspore::LessOp>,
    MindSporePointwiseConverter<mindspore::Atan2Op>,
    ConvertMindSporeReduceOp<mindspore::ReduceSumOp>,
    ConvertMindSporeReduceOp<mindspore::ReduceAllOp>,
    ConvertMindSporeReduceOp<mindspore::ReduceAnyOp>,
    ConvertMindSporeReduceOp<mindspore::ReduceMaxOp>,
    ConvertMindSporeReduceOp<mindspore::ReduceMinOp>,
    ConvertMindSporeReduceOp<mindspore::ReduceProdOp>,
    ConvertMindSporeTileOp,
    ConvertMindSporeBroadcastToOp,
    MindSporeAssignOpConverter,
    MindSporeInplaceAssignConverter,
    // mindspore.gather
    // (1).mindspore.gather->linalg.generic->affine loop
    MindSporeGatherConverter,
    // (2).mindspore.gather->linalg.gather->affine loop
    // MindsporeSpecificOpConverter<mindspore::GatherOp, linalgExt::GatherOp>,
    MindsporeSpecificOpConverter<mindspore::UnsortedSegmentSumOp, linalgExt::UnsortedSegmentSumOp>,
    MindSporeUnsortedSegmentSumOpConverter,
    MindsporeMatMulOpConverter<mindspore::MatMulOp>,
    MindsporeMatMulOpConverter<mindspore::BatchMatMulOp>
  >(patterns.getContext());
  // clang-format on
  return;
}

struct ConvertMindSporeToLinalgPass : public ConvertMindSporeToLinalgBase<ConvertMindSporeToLinalgPass> {
 public:
  ConvertMindSporeToLinalgPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<linalgExt::LinalgExtDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<shape::ShapeDialect>();
    registry.insert<math::MathDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, tosa::TosaDialect, linalg::LinalgDialect, linalgExt::LinalgExtDialect,
                           tensor::TensorDialect, func::FuncDialect, math::MathDialect, shape::ShapeDialect,
                           LLVM::LLVMDialect, scf::SCFDialect>();
    target.addIllegalDialect<mindspore::MindSporeDialect>();
    target.addLegalOp<mindspore::ReshapeOp>();
    target.addLegalOp<mindspore::SliceOp>();
    target.addLegalOp<mindspore::Strided_SliceOp>();

    FunctionOpInterface func = getOperation();
    mlir::populateLowerMindSporeToLinalgPattern(patterns);
    mlir::populateMindSporeLowerPattern(patterns);

    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createMindSporeToLinalgPass() {
  return std::make_unique<ConvertMindSporeToLinalgPass>();
}
