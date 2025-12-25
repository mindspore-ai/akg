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

#include "akg/Dialect/Affine/Transforms/VectorTransferTensorize.h"

#include <algorithm>
#include <cstdint>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "vector-transfer-tensorize"

using namespace mlir;  // NOLINT(build/namespaces)

namespace affine = mlir::affine;
namespace arith = mlir::arith;
namespace math = mlir::math;
namespace vector = mlir::vector;
namespace buffer = mlir::bufferization;
namespace memref = mlir::memref;
namespace tensor = mlir::tensor;
namespace func = mlir::func;
namespace linalg = mlir::linalg;

static RankedTensorType memrefToStaticTensor(MemRefType memrefType) {
  if (!memrefType || !memrefType.hasStaticShape()) {
    return {};
  }
  return RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
}

static bool isNumericLike(Type type) {
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    return rankedTensorType.getElementType().isIntOrFloat();
  }
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return vectorType.getElementType().isIntOrFloat();
  }
  return type.isIntOrFloat();
}

struct PendingWriteInfo {
  vector::TransferWriteOp writeOperation;
  tensor::InsertSliceOp insertSliceOperation;
  Value logicalBase;
};

struct TensorizationState {
  explicit TensorizationState(MLIRContext *context) : builder(context) {}
  OpBuilder builder;
  IRMapping valueMap;
  llvm::DenseSet<Operation *> operationsToErase;
  llvm::DenseSet<const Operation *> invalidOperations;
  llvm::SmallVector<PendingWriteInfo> pendingWrites;

  void set(Value originalValue, Value tensorizedValue) { valueMap.map(originalValue, tensorizedValue); }
  Value get(Value value) const { return valueMap.lookupOrNull(value); }
  void markErase(Operation *operation) { operationsToErase.insert(operation); }
  void trackInvalid(Operation *operation) { invalidOperations.insert(operation); }
  bool isInvalid(Operation *operation) const { return invalidOperations.contains(operation); }
};

static mlir::FailureOr<mlir::Value> convertScalarOrRank0ToTensor(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::RankedTensorType targetTensorType, mlir::Value scalarValue) {
  mlir::Type elementType = targetTensorType.getElementType();
  if (auto scalarTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
          scalarValue.getType())) {
    if (scalarTensorType.getRank() == 0 &&
        scalarTensorType.getElementType() == elementType) {
      if (targetTensorType.getRank() == 1 &&
          targetTensorType.getDimSize(0) == 1) {
        llvm::SmallVector<mlir::ReassociationIndices>
            expandReassociation = {};
        auto expandOp = builder.create<tensor::ExpandShapeOp>(
            location, targetTensorType, scalarValue,
            expandReassociation);
        return expandOp.getResult();
      }
      return mlir::failure();
    }
    return mlir::failure();
  }
  return mlir::failure();
}

static mlir::FailureOr<tensor::EmptyOp> convertAlloc(
    memref::AllocOp allocOperation,
    TensorizationState &tensorizationState) {
  auto tensorType = memrefToStaticTensor(allocOperation.getType());
  if (!tensorType) {
    return mlir::failure();
  }
  auto emptyTensorOperation =
      tensorizationState.builder.create<tensor::EmptyOp>(
          allocOperation.getLoc(), tensorType.getShape(),
          tensorType.getElementType());
  tensorizationState.set(allocOperation.getResult(),
                         emptyTensorOperation.getResult());
  tensorizationState.markErase(allocOperation);
  return emptyTensorOperation;
}

static void convertToMemref(buffer::ToMemrefOp toMemrefOperation,
                            TensorizationState &tensorizationState) {
  tensorizationState.set(toMemrefOperation.getResult(),
                         toMemrefOperation.getTensor());
  tensorizationState.markErase(toMemrefOperation);
}

static mlir::LogicalResult convertToTensor(
    buffer::ToTensorOp toTensorOperation,
    TensorizationState &tensorizationState) {
  mlir::Value mappedValue =
      tensorizationState.get(toTensorOperation.getMemref());
  if (!mappedValue) {
    return mlir::failure();
  }
  toTensorOperation.getResult().replaceAllUsesWith(mappedValue);
  tensorizationState.markErase(toTensorOperation);
  return mlir::success();
}

static mlir::FailureOr<tensor::ExtractSliceOp> build1DExtract(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::Value sourceTensor, mlir::ValueRange indexValues,
    unsigned sliceLength) {
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(sourceTensor.getType());
  if (!sourceTensorType || !sourceTensorType.hasStaticShape()) {
    return mlir::failure();
  }
  unsigned rank = sourceTensorType.getRank();
  llvm::SmallVector<mlir::OpFoldResult> strideFoldResults(
      rank, builder.getIndexAttr(1));
  llvm::SmallVector<mlir::OpFoldResult> offsetFoldResults;
  offsetFoldResults.reserve(rank);
  if (indexValues.empty()) {
    for (unsigned index = 0; index < rank; ++index) {
      offsetFoldResults.push_back(builder.getIndexAttr(0));
    }
  } else if (indexValues.size() == rank) {
    for (mlir::Value indexValue : indexValues) {
      if (auto constantOperation =
              indexValue.getDefiningOp<arith::ConstantOp>()) {
        if (auto integerAttribute = mlir::dyn_cast<mlir::IntegerAttr>(
                constantOperation.getValue())) {
          offsetFoldResults.push_back(
              builder.getIndexAttr(integerAttribute.getInt()));
          continue;
        }
      }
      offsetFoldResults.push_back(indexValue);
    }
  } else {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::OpFoldResult> sizeFoldResults(
      rank, builder.getIndexAttr(1));
  int64_t fullLength = sourceTensorType.getShape().back();
  int64_t chosenLength =
      sliceLength ? static_cast<int64_t>(sliceLength) : fullLength;
  sizeFoldResults.back() = builder.getIndexAttr(chosenLength);
  auto resultTensorType = mlir::RankedTensorType::get(
      {chosenLength}, sourceTensorType.getElementType());
  auto extractSliceOperation = builder.create<tensor::ExtractSliceOp>(
      location, resultTensorType, sourceTensor, offsetFoldResults,
      sizeFoldResults, strideFoldResults);
  return extractSliceOperation;
}

static arith::ConstantOp makeDenseConstant(
    mlir::OpBuilder &builder, mlir::Location location,
    mlir::RankedTensorType targetTensorType,
    mlir::DenseElementsAttr denseAttribute) {
  if (denseAttribute.getNumElements() ==
      targetTensorType.getNumElements()) {
    if (denseAttribute.getType().cast<mlir::ShapedType>()
            .getElementType() == targetTensorType.getElementType()) {
      return builder.create<arith::ConstantOp>(
          location, denseAttribute.reshape(targetTensorType));
    }
  }
  llvm::SmallVector<mlir::Attribute> elementAttributes;
  elementAttributes.reserve(denseAttribute.getNumElements());
  mlir::Type elementType = targetTensorType.getElementType();
  if (elementType.isa<mlir::FloatType>()) {
    auto floatValues = denseAttribute.getValues<llvm::APFloat>();
    std::transform(floatValues.begin(), floatValues.end(),
                   std::back_inserter(elementAttributes),
                   [elementType](const llvm::APFloat &floatValue) {
                     return mlir::FloatAttr::get(elementType, floatValue);
                   });
  } else if (elementType.isa<mlir::IntegerType>()) {
    auto integerValues = denseAttribute.getValues<llvm::APInt>();
    std::transform(integerValues.begin(), integerValues.end(),
                   std::back_inserter(elementAttributes),
                   [elementType](const llvm::APInt &integerValue) {
                     return mlir::IntegerAttr::get(elementType, integerValue);
                   });
  } else {
    auto attributeValues = denseAttribute.getValues<mlir::Attribute>();
    elementAttributes.insert(elementAttributes.end(),
                           attributeValues.begin(), attributeValues.end());
  }
  return builder.create<arith::ConstantOp>(
      location,
      mlir::DenseElementsAttr::get(targetTensorType, elementAttributes));
}

static mlir::FailureOr<arith::ConstantOp> scalarOrVectorToTensorConstant(
    arith::ConstantOp constantOperation,
    mlir::RankedTensorType targetTensorType, mlir::OpBuilder &builder) {
  if (!targetTensorType) {
    return mlir::failure();
  }
  mlir::Attribute attribute = constantOperation.getValue();
  if (auto denseAttribute =
          mlir::dyn_cast<mlir::DenseElementsAttr>(attribute)) {
    auto shapedType =
        mlir::dyn_cast<mlir::ShapedType>(denseAttribute.getType());
    if (shapedType && shapedType.getNumElements() ==
                          targetTensorType.getNumElements()) {
      return makeDenseConstant(builder, constantOperation.getLoc(),
                               targetTensorType, denseAttribute);
    }
    return mlir::failure();
  }
  mlir::Attribute elementAttribute = attribute;
  mlir::Type elementType = targetTensorType.getElementType();
  if (auto integerAttribute =
          mlir::dyn_cast<mlir::IntegerAttr>(attribute)) {
    if (integerAttribute.getType() != elementType) {
      elementAttribute =
          mlir::IntegerAttr::get(elementType, integerAttribute.getInt());
    }
  } else if (auto floatAttribute =
                 mlir::dyn_cast<mlir::FloatAttr>(attribute)) {
    if (floatAttribute.getType() != elementType) {
      elementAttribute = mlir::FloatAttr::get(elementType,
                                              floatAttribute.getValue());
    }
  } else {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Attribute> repeatedAttributes(
      targetTensorType.getNumElements(), elementAttribute);
  return builder.create<arith::ConstantOp>(
      constantOperation.getLoc(),
      mlir::DenseElementsAttr::get(targetTensorType, repeatedAttributes));
}

static mlir::LogicalResult upgradeConstantToTensor(
    arith::ConstantOp constantOperation,
    mlir::RankedTensorType targetTensorType,
    TensorizationState &tensorizationState) {
  if (tensorizationState.get(constantOperation.getResult())) {
    return mlir::success();
  }
  auto tensorConstantOperation = scalarOrVectorToTensorConstant(
      constantOperation, targetTensorType, tensorizationState.builder);
  if (mlir::failed(tensorConstantOperation)) {
    return mlir::failure();
  }
  tensorizationState.set(constantOperation.getResult(),
                         tensorConstantOperation->getResult());
  return mlir::success();
}

static mlir::LogicalResult replaceVectorConstantWithTensor(
    arith::ConstantOp constantOperation, mlir::OpBuilder &builder) {
  auto vectorType =
      mlir::dyn_cast<mlir::VectorType>(constantOperation.getType());
  if (!vectorType) {
    return mlir::failure();
  }
  auto denseAttribute = mlir::dyn_cast<mlir::DenseElementsAttr>(
      constantOperation.getValue());
  if (!denseAttribute ||
      !mlir::isa<mlir::VectorType>(denseAttribute.getType())) {
    return mlir::failure();
  }
  auto targetTensorType = mlir::RankedTensorType::get(
      vectorType.getShape(), vectorType.getElementType());
  builder.setInsertionPoint(constantOperation);
  auto newConstant = makeDenseConstant(
      builder, constantOperation.getLoc(), targetTensorType,
      denseAttribute);
  constantOperation.getResult().replaceAllUsesWith(
      newConstant.getResult());
  constantOperation.erase();
  return mlir::success();
}

static mlir::FailureOr<mlir::Operation *> convertCollapseShape(
    memref::CollapseShapeOp collapseShapeOp,
    TensorizationState &tensorizationState) {
  mlir::Value sourceMemref = collapseShapeOp.getSrc();
  mlir::Value sourceTensor = tensorizationState.get(sourceMemref);
  if (!sourceTensor) {
    return mlir::failure();
  }
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(sourceTensor.getType());
  if (!sourceTensorType) {
    return mlir::failure();
  }
  auto memrefResultType = collapseShapeOp.getResultType();
  auto tensorResultType = mlir::RankedTensorType::get(
      memrefResultType.getShape(), memrefResultType.getElementType());
  auto reassociationIndices = collapseShapeOp.getReassociationIndices();
  tensorizationState.builder.setInsertionPoint(collapseShapeOp);
  auto tensorCollapseOp =
      tensorizationState.builder.create<tensor::CollapseShapeOp>(
          collapseShapeOp.getLoc(), tensorResultType, sourceTensor,
          reassociationIndices);
  tensorizationState.set(collapseShapeOp.getResult(),
                         tensorCollapseOp.getResult());
  tensorizationState.markErase(collapseShapeOp);
  return tensorCollapseOp.getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertExpandShape(
    memref::ExpandShapeOp expandShapeOp,
    TensorizationState &tensorizationState) {
  mlir::Value sourceMemref = expandShapeOp.getSrc();
  mlir::Value sourceTensor = tensorizationState.get(sourceMemref);
  if (!sourceTensor) {
    return mlir::failure();
  }
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(sourceTensor.getType());
  if (!sourceTensorType) {
    return mlir::failure();
  }
  auto memrefResultType = expandShapeOp.getResultType();
  auto tensorResultType = mlir::RankedTensorType::get(
      memrefResultType.getShape(), memrefResultType.getElementType());
  auto reassociationIndices = expandShapeOp.getReassociationIndices();
  tensorizationState.builder.setInsertionPoint(expandShapeOp);
  auto tensorExpandOp =
      tensorizationState.builder.create<tensor::ExpandShapeOp>(
          expandShapeOp.getLoc(), tensorResultType, sourceTensor,
          reassociationIndices);
  tensorizationState.set(expandShapeOp.getResult(),
                         tensorExpandOp.getResult());
  tensorizationState.markErase(expandShapeOp);
  return tensorExpandOp.getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertTransferRead(
    vector::TransferReadOp transferReadOperation,
    TensorizationState &tensorizationState) {
  mlir::Value sourceTensor =
      tensorizationState.get(transferReadOperation.getSource());
  if (!sourceTensor) {
    return mlir::failure();
  }
  auto sourceTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(sourceTensor.getType());
  if (!sourceTensorType) {
    return mlir::failure();
  }
  unsigned sliceLength =
      transferReadOperation.getVectorType().getNumElements();
  tensorizationState.builder.setInsertionPoint(transferReadOperation);
  if (sourceTensorType.getRank() == 0) {
    auto tensorRank1Type = mlir::RankedTensorType::get(
        {1}, sourceTensorType.getElementType());
    llvm::SmallVector<mlir::ReassociationIndices> expandReassociation =
        {};
    auto tensorExpandOp =
        tensorizationState.builder.create<tensor::ExpandShapeOp>(
            transferReadOperation.getLoc(), tensorRank1Type,
            sourceTensor, expandReassociation);
    sourceTensor = tensorExpandOp.getResult();
  }
  auto extractSliceOperation = build1DExtract(
      tensorizationState.builder, transferReadOperation.getLoc(),
      sourceTensor, transferReadOperation.getIndices(), sliceLength);
  if (mlir::failed(extractSliceOperation)) {
    return mlir::failure();
  }
  tensorizationState.set(transferReadOperation.getResult(),
                         extractSliceOperation->getResult());
  tensorizationState.markErase(transferReadOperation);
  if (mlir::Value paddingValue = transferReadOperation.getPadding()) {
    if (auto constantOperation =
            paddingValue.getDefiningOp<arith::ConstantOp>()) {
      tensorizationState.markErase(constantOperation);
    }
  }
  return extractSliceOperation->getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertCreateMask(
    vector::CreateMaskOp createMaskOp,
    TensorizationState &tensorizationState) {
  auto vectorType = createMaskOp.getVectorType();
  if (vectorType.getRank() != 1) {
    return mlir::failure();
  }
  int64_t maskSize = vectorType.getShape()[0];
  mlir::Value dynamicSize = createMaskOp.getOperand(0);
  tensorizationState.builder.setInsertionPoint(createMaskOp);
  mlir::Location loc = createMaskOp.getLoc();
  auto tensorType = mlir::RankedTensorType::get(
      {maskSize}, tensorizationState.builder.getI1Type());
  auto generateOp =
      tensorizationState.builder.create<tensor::GenerateOp>(
          loc, tensorType, mlir::ValueRange{});
  mlir::Region &region = generateOp.getRegion();
  mlir::Block *block =
      tensorizationState.builder.createBlock(&region);
  block->addArgument(tensorizationState.builder.getIndexType(), loc);
  mlir::OpBuilder::InsertionGuard guard(tensorizationState.builder);
  tensorizationState.builder.setInsertionPointToStart(block);
  mlir::Value index = block->getArgument(0);
  mlir::Value isInBounds =
      tensorizationState.builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::slt, index, dynamicSize);
  tensorizationState.builder.create<tensor::YieldOp>(loc, isInBounds);
  tensorizationState.set(createMaskOp.getResult(),
                         generateOp.getResult());
  tensorizationState.markErase(createMaskOp);
  return generateOp.getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertTransferWrite(
    vector::TransferWriteOp transferWriteOperation,
    TensorizationState &tensorizationState) {
  mlir::Value logicalBaseValue =
      tensorizationState.get(transferWriteOperation.getSource());
  if (!logicalBaseValue) {
    return mlir::failure();
  }
  auto destinationTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(logicalBaseValue.getType());
  if (!destinationTensorType) {
    return mlir::failure();
  }
  mlir::Value originalSourceValue =
      transferWriteOperation->getOperand(0);
  mlir::Type originalSourceType = originalSourceValue.getType();
  unsigned sliceLength = 0;
  mlir::Type elementType = nullptr;
  if (auto tensorType =
          mlir::dyn_cast<mlir::RankedTensorType>(originalSourceType)) {
    if (tensorType.getRank() != 1) {
      return mlir::failure();
    }
    sliceLength = tensorType.getDimSize(0);
    elementType = tensorType.getElementType();
  } else if (auto vectorType =
                 mlir::dyn_cast<mlir::VectorType>(originalSourceType)) {
    sliceLength = vectorType.getNumElements();
    elementType = vectorType.getElementType();
  } else {
    return mlir::failure();
  }
  auto sliceTensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(sliceLength)}, elementType);
  mlir::Value sourceSliceValue =
      tensorizationState.get(originalSourceValue);
  if (!sourceSliceValue) {
    if (auto constantOperation =
            originalSourceValue.getDefiningOp<arith::ConstantOp>()) {
      auto tensorConstantOperation = scalarOrVectorToTensorConstant(
          constantOperation, sliceTensorType,
          tensorizationState.builder);
      if (mlir::failed(tensorConstantOperation)) {
        return mlir::failure();
      }
      sourceSliceValue = tensorConstantOperation->getResult();
    }
  }
  if (!sourceSliceValue) {
    if (originalSourceType == sliceTensorType) {
      sourceSliceValue = originalSourceValue;
    } else if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
                   originalSourceType)) {
      (void)tensorType;
      return mlir::failure();
    }
  }
  if (!sourceSliceValue ||
      sourceSliceValue.getType() != sliceTensorType) {
    return mlir::failure();
  }
  unsigned rank = destinationTensorType.getRank();
  llvm::SmallVector<mlir::OpFoldResult> strideFoldResults(
      rank, tensorizationState.builder.getIndexAttr(1));
  llvm::SmallVector<mlir::OpFoldResult> sizeFoldResults(
      rank, tensorizationState.builder.getIndexAttr(1));
  sizeFoldResults.back() =
      tensorizationState.builder.getIndexAttr(sliceLength);
  llvm::SmallVector<mlir::OpFoldResult> offsetFoldResults(
      transferWriteOperation.getIndices().begin(),
      transferWriteOperation.getIndices().end());
  auto insertSliceOperation =
      tensorizationState.builder.create<tensor::InsertSliceOp>(
          transferWriteOperation.getLoc(), sourceSliceValue,
          logicalBaseValue, offsetFoldResults, sizeFoldResults,
          strideFoldResults);
  tensorizationState.pendingWrites.push_back(
      PendingWriteInfo{transferWriteOperation, insertSliceOperation,
                       logicalBaseValue});
  tensorizationState.markErase(transferWriteOperation);
  return insertSliceOperation.getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertAffineLoad(
    affine::AffineLoadOp affineLoadOperation,
    TensorizationState &tensorizationState) {
  mlir::Value memrefValue = affineLoadOperation.getMemref();
  mlir::Value sourceTensor = tensorizationState.get(memrefValue);
  if (!sourceTensor) {
    if (auto toMemrefOperation =
            memrefValue.getDefiningOp<buffer::ToMemrefOp>()) {
      sourceTensor = toMemrefOperation.getTensor();
      tensorizationState.set(memrefValue, sourceTensor);
    } else {
      return mlir::failure();
    }
  }
  unsigned sliceLength = 1;
  mlir::ValueRange indexOperands =
      affineLoadOperation.getMapOperands();
  auto extractSliceOperation = build1DExtract(
      tensorizationState.builder, affineLoadOperation.getLoc(),
      sourceTensor, indexOperands, sliceLength);
  if (mlir::failed(extractSliceOperation)) {
    return mlir::failure();
  }
  affineLoadOperation.getResult().replaceAllUsesWith(
      extractSliceOperation->getResult());
  tensorizationState.set(affineLoadOperation.getResult(),
                         extractSliceOperation->getResult());
  tensorizationState.markErase(affineLoadOperation);
  return extractSliceOperation->getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertAffineStore(
    affine::AffineStoreOp affineStoreOperation,
    TensorizationState &tensorizationState) {
  mlir::Value memrefValue = affineStoreOperation.getMemref();
  mlir::Value baseTensor = tensorizationState.get(memrefValue);
  if (!baseTensor) {
    if (auto toMemrefOperation =
            memrefValue.getDefiningOp<buffer::ToMemrefOp>()) {
      baseTensor = toMemrefOperation.getTensor();
    } else if (auto allocOperation =
                   memrefValue.getDefiningOp<memref::AllocOp>()) {
      auto emptyTensor =
          convertAlloc(allocOperation, tensorizationState);
      if (mlir::failed(emptyTensor)) {
        return mlir::failure();
      }
      baseTensor = emptyTensor->getResult();
    } else {
      return mlir::failure();
    }
  }
  auto destinationTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(baseTensor.getType());
  if (!destinationTensorType) {
    return mlir::failure();
  }
  unsigned rank = destinationTensorType.getRank();
  mlir::Value storeValue = affineStoreOperation.getValue();
  auto storeTensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(storeValue.getType());
  if (!storeTensorType) {
    return mlir::failure();
  }
  mlir::Type elementType = storeTensorType.getElementType();
  if (!elementType.isIntOrFloat()) {
    return mlir::failure();
  }
  if (rank == 0) {
    if (storeTensorType.getRank() != 0) {
      return mlir::failure();
    }
    tensorizationState.set(memrefValue, storeValue);
    tensorizationState.markErase(affineStoreOperation);
    if (mlir::Operation *definingOp = storeValue.getDefiningOp()) {
      return definingOp;
    }
    return mlir::failure();
  }
  auto sliceTensorType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(1)}, elementType);
  mlir::Value sliceTensor;
  if (storeTensorType.getRank() == 0) {
    llvm::SmallVector<mlir::ReassociationIndices> reassociationIndices;
    auto expandShapeOperation =
        tensorizationState.builder.create<tensor::ExpandShapeOp>(
            affineStoreOperation.getLoc(), sliceTensorType, storeValue,
            reassociationIndices);
    sliceTensor = expandShapeOperation.getResult();
  } else if (storeTensorType.getRank() == 1 &&
             storeTensorType.getDimSize(0) == 1) {
    if (storeTensorType == sliceTensorType) {
      sliceTensor = storeValue;
    } else {
      return mlir::failure();
    }
  } else {
    return mlir::failure();
  }
  mlir::ValueRange indexOperands =
      affineStoreOperation.getMapOperands();
  llvm::SmallVector<mlir::Value, 4> indexValues(indexOperands.begin(),
                                                 indexOperands.end());
  if (indexValues.empty()) {
    tensorizationState.builder.setInsertionPoint(affineStoreOperation);
    for (unsigned index = 0; index < rank; ++index) {
      indexValues.push_back(
          tensorizationState.builder.create<arith::ConstantIndexOp>(
              affineStoreOperation.getLoc(), 0));
    }
  } else if (indexValues.size() != rank) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::OpFoldResult> strideFoldResults(
      rank, tensorizationState.builder.getIndexAttr(1));
  llvm::SmallVector<mlir::OpFoldResult> sizeFoldResults(
      rank, tensorizationState.builder.getIndexAttr(1));
  if (rank > 0) {
    sizeFoldResults.back() =
        tensorizationState.builder.getIndexAttr(1);
  }
  llvm::SmallVector<mlir::OpFoldResult> offsetFoldResults;
  offsetFoldResults.reserve(rank);
  for (mlir::Value indexValue : indexValues) {
    if (auto constantOperation =
            indexValue.getDefiningOp<arith::ConstantOp>()) {
      if (auto integerAttribute = mlir::dyn_cast<mlir::IntegerAttr>(
              constantOperation.getValue())) {
        offsetFoldResults.push_back(
            tensorizationState.builder.getIndexAttr(
                integerAttribute.getInt()));
        continue;
      }
    }
    offsetFoldResults.push_back(indexValue);
  }
  auto insertSliceOperation =
      tensorizationState.builder.create<tensor::InsertSliceOp>(
          affineStoreOperation.getLoc(), sliceTensor, baseTensor,
          offsetFoldResults, sizeFoldResults, strideFoldResults);
  tensorizationState.markErase(affineStoreOperation);
  return insertSliceOperation.getOperation();
}

static mlir::FailureOr<mlir::Operation *> cloneElementWise(
    mlir::Operation *originalOperation,
    llvm::ArrayRef<mlir::Value> newOperands,
    TensorizationState &tensorizationState) {
  bool isSelectOperation = mlir::isa<arith::SelectOp>(originalOperation);
  unsigned startIndex = isSelectOperation ? 1 : 0;
  mlir::RankedTensorType referenceTensorType = nullptr;
  for (unsigned operandIndex = startIndex;
       operandIndex < newOperands.size(); ++operandIndex) {
    mlir::Value operandValue = newOperands[operandIndex];
    if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
            operandValue.getType())) {
      referenceTensorType = rankedTensorType;
      break;
    }
  }
  if (!referenceTensorType) {
    for (mlir::Value operandValue : newOperands) {
      if (auto rankedTensorType =
              mlir::dyn_cast<mlir::RankedTensorType>(
                  operandValue.getType())) {
        referenceTensorType = rankedTensorType;
        break;
      }
    }
  }
  if (!referenceTensorType) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Type> resultTypes;
  resultTypes.reserve(originalOperation->getNumResults());
  auto makeTensorType = [&](mlir::Type elementType)
      -> mlir::RankedTensorType {
    return mlir::RankedTensorType::get(referenceTensorType.getShape(),
                                       elementType);
  };
  bool isCompareOperation =
      mlir::isa<arith::CmpFOp, arith::CmpIOp>(originalOperation);
  mlir::Type i1Type =
      mlir::IntegerType::get(tensorizationState.builder.getContext(), 1);
  for (mlir::Type oldResultType : originalOperation->getResultTypes()) {
    if (isCompareOperation) {
      resultTypes.push_back(makeTensorType(i1Type));
    } else {
      mlir::Type resultElementType =
          referenceTensorType.getElementType();
      if (auto oldTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              oldResultType)) {
        resultElementType = oldTensorType.getElementType();
      } else if (auto oldVectorType =
                     mlir::dyn_cast<mlir::VectorType>(oldResultType)) {
        resultElementType = oldVectorType.getElementType();
      } else if (oldResultType.isIntOrFloat()) {
        resultElementType = oldResultType;
      }
      resultTypes.push_back(makeTensorType(resultElementType));
    }
  }
  mlir::OperationState newOperationState(originalOperation->getLoc(),
                                         originalOperation->getName());
  newOperationState.addOperands(newOperands);
  for (auto attribute : originalOperation->getAttrs()) {
    newOperationState.addAttribute(attribute.getName(),
                                   attribute.getValue());
  }
  newOperationState.addTypes(resultTypes);
  mlir::Operation *newOperation =
      tensorizationState.builder.create(newOperationState);
  if (!newOperation) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Value> finalResults;
  finalResults.reserve(newOperation->getNumResults());
  for (auto [resultIndex, actualResult] :
       llvm::enumerate(newOperation->getResults())) {
    mlir::Type expectedType = resultTypes[resultIndex];
    if (actualResult.getType() == expectedType) {
      finalResults.push_back(actualResult);
      continue;
    }
    if (auto tensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(expectedType)) {
      auto convertedValue = convertScalarOrRank0ToTensor(
          tensorizationState.builder, originalOperation->getLoc(),
          tensorType, actualResult);
      if (mlir::succeeded(convertedValue)) {
        finalResults.push_back(*convertedValue);
        continue;
      }
    }
    newOperation->erase();
    return mlir::failure();
  }
  for (auto [oldResult, newResult] :
       llvm::zip(originalOperation->getResults(), finalResults)) {
    oldResult.replaceAllUsesWith(newResult);
    tensorizationState.set(oldResult, newResult);
  }
  tensorizationState.markErase(originalOperation);
  return newOperation;
}

static mlir::FailureOr<mlir::Operation *> tensorizeOperation(
    mlir::Operation *operation, TensorizationState &tensorizationState);

static mlir::RankedTensorType inferReferenceTensorType(
    mlir::Operation *operation,
    const TensorizationState &tensorizationState) {
  bool isSelectOperation = mlir::isa<arith::SelectOp>(operation);
  unsigned startIndex = isSelectOperation ? 1 : 0;
  for (unsigned operandIndex = startIndex;
       operandIndex < operation->getNumOperands(); ++operandIndex) {
    mlir::Value operandValue = operation->getOperand(operandIndex);
    if (mlir::Value mappedValue =
            tensorizationState.get(operandValue)) {
      if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              mappedValue.getType())) {
        return rankedTensorType;
      }
    }
  }
  for (unsigned operandIndex = startIndex;
       operandIndex < operation->getNumOperands(); ++operandIndex) {
    mlir::Value operandValue = operation->getOperand(operandIndex);
    if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
            operandValue.getType())) {
      return rankedTensorType;
    }
  }
  for (mlir::Value operandValue : operation->getOperands()) {
    if (mlir::Value mappedValue =
            tensorizationState.get(operandValue)) {
      if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
              mappedValue.getType())) {
        return rankedTensorType;
      }
    }
  }
  for (mlir::Value operandValue : operation->getOperands()) {
    if (auto rankedTensorType = mlir::dyn_cast<mlir::RankedTensorType>(
            operandValue.getType())) {
      return rankedTensorType;
    }
  }
  for (mlir::Value operandValue : operation->getOperands()) {
    if (auto vectorType =
            mlir::dyn_cast<mlir::VectorType>(operandValue.getType())) {
      return mlir::RankedTensorType::get(vectorType.getShape(),
                                         vectorType.getElementType());
    }
  }
  for (mlir::Type resultType : operation->getResultTypes()) {
    if (auto rankedTensorType =
            mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
      return rankedTensorType;
    }
    if (auto vectorType = mlir::dyn_cast<mlir::VectorType>(resultType)) {
      return mlir::RankedTensorType::get(vectorType.getShape(),
                                         vectorType.getElementType());
    }
  }
  return nullptr;
}

static mlir::FailureOr<mlir::Operation *> convertElementWise(
    mlir::Operation *operation,
    TensorizationState &tensorizationState) {
  if (operation->hasTrait<mlir::OpTrait::IsTerminator>()) {
    return mlir::failure();
  }
  if (mlir::isa<vector::ReductionOp>(operation)) {
    return mlir::failure();
  }
  if (operation->getNumRegions() != 0) {
    return mlir::failure();
  }
  if (llvm::any_of(operation->getOperandTypes(),
                   [](mlir::Type type) {
                     return !isNumericLike(type);
                   })) {
    return mlir::failure();
  }
  if (llvm::any_of(operation->getResultTypes(),
                   [](mlir::Type type) {
                     return !isNumericLike(type);
                   })) {
    return mlir::failure();
  }
  bool allTensorOperands = llvm::all_of(
      operation->getOperandTypes(), [](mlir::Type type) {
        return mlir::isa<mlir::RankedTensorType>(type);
      });
  bool allTensorResults = llvm::all_of(
      operation->getResultTypes(), [](mlir::Type type) {
        return mlir::isa<mlir::RankedTensorType>(type);
      });
  if (allTensorOperands && allTensorResults) {
    bool hasPendingMappedVector = false;
    for (mlir::Value operandValue : operation->getOperands()) {
      if (mlir::Value mappedValue =
              tensorizationState.get(operandValue)) {
        if (mappedValue != operandValue &&
            mlir::isa<mlir::RankedTensorType>(mappedValue.getType())) {
          hasPendingMappedVector = true;
          break;
        }
      }
    }
    if (!hasPendingMappedVector) {
      return mlir::failure();
    }
  }
  auto referenceTensorType =
      inferReferenceTensorType(operation, tensorizationState);
  if (!referenceTensorType) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Value> newOperands;
  newOperands.reserve(operation->getNumOperands());
  const bool isSelectOperation = mlir::isa<arith::SelectOp>(operation);
  for (unsigned operandIndex = 0;
       operandIndex < operation->getNumOperands(); ++operandIndex) {
    mlir::Value operandValue = operation->getOperand(operandIndex);
    mlir::RankedTensorType expectedTensorType = referenceTensorType;
    if (isSelectOperation && operandIndex == 0) {
      mlir::Type i1Type = mlir::IntegerType::get(
          tensorizationState.builder.getContext(), 1);
      expectedTensorType = mlir::RankedTensorType::get(
          referenceTensorType.getShape(), i1Type);
    }
    if (operandValue.getType() == expectedTensorType) {
      newOperands.push_back(operandValue);
      continue;
    }
    if (mlir::Value mappedValue =
            tensorizationState.get(operandValue)) {
      if (mappedValue.getType() == expectedTensorType) {
        newOperands.push_back(mappedValue);
        continue;
      } else {
        if (mlir::Operation *definingOperation =
                operandValue.getDefiningOp()) {
          tensorizationState.builder.setInsertionPoint(
              definingOperation);
          (void)tensorizeOperation(definingOperation,
                                   tensorizationState);
          mlir::Value remappedValue =
              tensorizationState.get(operandValue);
          if (remappedValue &&
              remappedValue.getType() == expectedTensorType) {
            newOperands.push_back(remappedValue);
            continue;
          }
        }
      }
    }
    if (auto constantOperation =
            operandValue.getDefiningOp<arith::ConstantOp>()) {
      if (mlir::failed(upgradeConstantToTensor(
              constantOperation, expectedTensorType,
              tensorizationState))) {
        return mlir::failure();
      }
      newOperands.push_back(
          tensorizationState.get(constantOperation.getResult()));
      continue;
    }
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(
            expectedTensorType)) {
      auto convertedValue = convertScalarOrRank0ToTensor(
          tensorizationState.builder, operation->getLoc(), tensorType,
          operandValue);
      if (mlir::succeeded(convertedValue)) {
        newOperands.push_back(*convertedValue);
        continue;
      }
    }
    if (mlir::isa<mlir::VectorType>(operandValue.getType())) {
      if (mlir::Operation *definingOperation =
              operandValue.getDefiningOp()) {
        tensorizationState.builder.setInsertionPoint(definingOperation);
        (void)tensorizeOperation(definingOperation,
                                 tensorizationState);
        if (mlir::Value tensorizedValue =
                tensorizationState.get(operandValue);
            tensorizedValue &&
            tensorizedValue.getType() == expectedTensorType) {
          newOperands.push_back(tensorizedValue);
          continue;
        }
      }
    }
    return mlir::failure();
  }
  return cloneElementWise(operation, newOperands, tensorizationState);
}

static FailureOr<Operation *> convertAffineForWithIterArgs(
    affine::AffineForOp affineForOperation,
    TensorizationState &tensorizationState) {
  auto initialValues = affineForOperation.getInits();
  auto &oldBodyBlock = affineForOperation.getRegion().front();
  auto *oldTerminatorOperation = oldBodyBlock.getTerminator();
  auto oldYieldOperation =
      dyn_cast_or_null<affine::AffineYieldOp>(oldTerminatorOperation);
  if (!oldYieldOperation) {
    return failure();
  }
  if (initialValues.empty()) {
    return failure();
  }
  if (oldBodyBlock.getNumArguments() <= 1) {
    return failure();
  }
  Value oldIterationArgument = oldBodyBlock.getArgument(1);
  Type oldIterationType = oldIterationArgument.getType();
  bool iterationIsVector = isa<VectorType>(oldIterationType);
  bool iterationIsTensor = isa<RankedTensorType>(oldIterationType);
  bool iterationIsScalar = !iterationIsVector && !iterationIsTensor;
  Value oldYieldValue = oldYieldOperation.getOperand(0);
  Type oldYieldType = oldYieldValue.getType();
  bool yieldIsVector = isa<VectorType>(oldYieldType);
  bool yieldIsTensor = isa<RankedTensorType>(oldYieldType);
  bool yieldIsScalar = !yieldIsVector && !yieldIsTensor;
  if (iterationIsTensor && yieldIsTensor) {
    return failure();
  }
  bool canHandle = (iterationIsVector && yieldIsVector) ||
                   (iterationIsScalar && yieldIsScalar) ||
                   (iterationIsScalar && yieldIsTensor) ||
                   (iterationIsVector && yieldIsTensor);
  if (!canHandle) {
    return failure();
  }
  Location location = affineForOperation.getLoc();
  SmallVector<Value> newInitialValues;
  newInitialValues.reserve(initialValues.size());
  tensorizationState.builder.setInsertionPoint(affineForOperation);
  for (unsigned i = 0; i < initialValues.size(); ++i) {
    Value initialValue = initialValues[i];
    Type initialType = initialValue.getType();
    if (auto tensorType = dyn_cast<RankedTensorType>(initialType)) {
      newInitialValues.push_back(initialValue);
      continue;
    }
    if (!isa<VectorType>(initialType) &&
        !isa<RankedTensorType>(initialType)) {
      auto rank0TensorType = RankedTensorType::get({}, initialType);
      (void)rank0TensorType;
      auto emptyOp = tensorizationState.builder.create<tensor::EmptyOp>(
          location, ArrayRef<int64_t>{}, initialType);
      auto fillOp = tensorizationState.builder.create<linalg::FillOp>(
          location, ValueRange{initialValue},
          ValueRange{emptyOp.getResult()});
      newInitialValues.push_back(fillOp.getResult(0));
      continue;
    }
    return failure();
  }
  tensorizationState.builder.setInsertionPoint(affineForOperation);
  int64_t stepInteger = affineForOperation.getStep().getSExtValue();
  auto newAffineForOperation =
      tensorizationState.builder.create<affine::AffineForOp>(
          location, affineForOperation.getLowerBoundOperands(),
          affineForOperation.getLowerBoundMap(),
          affineForOperation.getUpperBoundOperands(),
          affineForOperation.getUpperBoundMap(), stepInteger,
          ValueRange(newInitialValues));
  auto &newBodyBlock = newAffineForOperation.getRegion().front();
  newBodyBlock.clear();
  IRMapping localMapping;
  localMapping.map(oldBodyBlock.getArgument(0), newBodyBlock.getArgument(0));
  for (unsigned index = 0; index < initialValues.size(); ++index) {
    Value oldIterationValue = oldBodyBlock.getArgument(1 + index);
    Value newIterationValue = newBodyBlock.getArgument(1 + index);
    localMapping.map(oldIterationValue, newIterationValue);
    tensorizationState.set(oldIterationValue, newIterationValue);
  }
  tensorizationState.builder.setInsertionPointToStart(&newBodyBlock);
  affine::AffineYieldOp newYieldOperation = nullptr;
  DenseMap<Operation *, Operation *> oldToClonedOp;
  for (Operation &bodyOperation : oldBodyBlock) {
    for (auto result : bodyOperation.getResults()) {
      if (Value tensorizedValue = tensorizationState.get(result)) {
        if (tensorizedValue.getType() != result.getType()) {
          localMapping.map(result, tensorizedValue);
        }
      }
    }
  }
  for (Operation &bodyOperation : llvm::make_early_inc_range(oldBodyBlock)) {
    Operation *clonedOperation = tensorizationState.builder.clone(bodyOperation, localMapping);
    if (auto clonedYieldOperation = dyn_cast<affine::AffineYieldOp>(clonedOperation)) {
      newYieldOperation = clonedYieldOperation;
    }
    oldToClonedOp[&bodyOperation] = clonedOperation;
    auto oldResults = bodyOperation.getResults();
    auto newResults = clonedOperation->getResults();
    assert(oldResults.size() == newResults.size());
    for (auto [oldResult, newResult] : llvm::zip(oldResults, newResults)) {
      if (!localMapping.contains(oldResult)) {
        localMapping.map(oldResult, newResult);
      }
      tensorizationState.set(oldResult, newResult);
    }
  }
  for (Operation &clonedOp : llvm::make_early_inc_range(newBodyBlock)) {
    if (isa<affine::AffineYieldOp>(clonedOp)) {
      continue;
    }
    bool usesIterArgs = false;
    for (Value operand : clonedOp.getOperands()) {
      if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
        if (blockArg.getOwner() == &newBodyBlock &&
            blockArg.getArgNumber() > 0) {
          usesIterArgs = true;
          break;
        }
      }
    }
    if (!usesIterArgs) {
      continue;
    }
    tensorizationState.builder.setInsertionPoint(&clonedOp);
    auto tensorizeResult =
        tensorizeOperation(&clonedOp, tensorizationState);
    if (succeeded(tensorizeResult)) {
      Operation *tensorizedOp = *tensorizeResult;
      assert(clonedOp.getNumResults() ==
             tensorizedOp->getNumResults());
      for (auto [clonedResult, tensorizedResult] :
           llvm::zip(clonedOp.getResults(),
                     tensorizedOp->getResults())) {
        clonedResult.replaceAllUsesWith(tensorizedResult);
      }
      Operation *correspondingOldOp = nullptr;
      for (auto &[oldOp, mappedClonedOp] : oldToClonedOp) {
        if (mappedClonedOp == &clonedOp) {
          correspondingOldOp = oldOp;
          break;
        }
      }
      if (correspondingOldOp) {
        auto oldResults = correspondingOldOp->getResults();
        auto tensorizedResults = tensorizedOp->getResults();
        for (auto [oldResult, tensorizedResult] :
             llvm::zip(oldResults, tensorizedResults)) {
          localMapping.map(oldResult, tensorizedResult);
        }
      }
    }
  }
  if (!newYieldOperation) {
    newAffineForOperation.erase();
    return failure();
  }
  tensorizationState.builder.setInsertionPoint(newYieldOperation);
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYieldOperation.getNumOperands());
  for (unsigned index = 0; index < oldYieldOperation.getNumOperands();
       ++index) {
    Value oldValue = oldYieldOperation.getOperand(index);
    Value mappedValue = localMapping.lookupOrNull(oldValue);
    if (!mappedValue) {
      mappedValue = tensorizationState.get(oldValue);
    }
    if (!mappedValue) {
      newAffineForOperation.erase();
      return failure();
    }
    Type expectedType = newBodyBlock.getArgument(1 + index).getType();
    if (mappedValue.getType() != expectedType) {
      newAffineForOperation.erase();
      return failure();
    }
    if (index < initialValues.size()) {
      Type originalIterArgType =
          oldBodyBlock.getArgument(1 + index).getType();
      bool wasScalar = !isa<VectorType>(originalIterArgType) &&
                       !isa<RankedTensorType>(originalIterArgType);
      if (wasScalar) {
        Type mappedType = mappedValue.getType();
        if (!isa<VectorType>(mappedType) &&
            !isa<RankedTensorType>(mappedType)) {
          auto emptyOp =
              tensorizationState.builder.create<tensor::EmptyOp>(
                  location, ArrayRef<int64_t>{}, mappedType);
          auto fillOp = tensorizationState.builder.create<linalg::FillOp>(
              location, ValueRange{mappedValue},
              ValueRange{emptyOp.getResult()});
          mappedValue = fillOp.getResult(0);
        }
      }
    }
    newYieldOperands.push_back(mappedValue);
  }
  newYieldOperation->setOperands(newYieldOperands);
  affineForOperation.replaceAllUsesWith(
      newAffineForOperation.getResults());
  tensorizationState.markErase(affineForOperation);
  return newAffineForOperation.getOperation();
}

static mlir::FailureOr<mlir::Operation *> convertVectorReduction(
    vector::ReductionOp vectorReductionOperation,
    TensorizationState &tensorizationState) {
  mlir::Location location = vectorReductionOperation.getLoc();
  mlir::Value inputValue = vectorReductionOperation->getOperand(0);
  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(inputValue.getType());
  if (!tensorType) {
    return mlir::failure();
  }
  if (tensorType.getRank() != 1) {
    return mlir::failure();
  }
  mlir::Type elementType = tensorType.getElementType();
  if (!elementType.isa<mlir::FloatType>()) {
    return mlir::failure();
  }
  if (vectorReductionOperation.getKind() !=
      vector::CombiningKind::ADD) {
    return mlir::failure();
  }
  if (mlir::isa<mlir::RankedTensorType>(
          vectorReductionOperation.getResult().getType())) {
    return mlir::failure();
  }
  mlir::OpBuilder &builder = tensorizationState.builder;
  mlir::OpBuilder::InsertionGuard insertionGuard(builder);
  builder.setInsertionPoint(vectorReductionOperation);
  auto initTensorType = mlir::RankedTensorType::get({}, elementType);
  mlir::Value initEmptyTensor =
      builder
          .create<tensor::EmptyOp>(location, llvm::ArrayRef<int64_t>{},
                                   elementType)
          .getResult();
  auto zeroAttribute = builder.getFloatAttr(
      elementType.cast<mlir::FloatType>(), 0.0);
  mlir::Value zeroScalar =
      builder
          .create<arith::ConstantOp>(location, elementType,
                                     zeroAttribute)
          .getResult();
  auto fillOperation = builder.create<linalg::FillOp>(
      location, mlir::ValueRange{zeroScalar},
      mlir::ValueRange{initEmptyTensor});
  mlir::Value initTensor = fillOperation.getResult(0);
  llvm::SmallVector<int64_t, 1> reduceDimensions = {0};
  auto linalgReduceOperation = builder.create<linalg::ReduceOp>(
      location, mlir::TypeRange{initTensorType},
      mlir::ValueRange{inputValue}, mlir::ValueRange{initTensor},
      reduceDimensions);
  {
    mlir::Region &region = linalgReduceOperation.getRegion();
    auto *block = new mlir::Block();
    region.push_back(block);
    block->addArgument(elementType, location);
    block->addArgument(elementType, location);
    mlir::OpBuilder::InsertionGuard blockGuard(builder);
    builder.setInsertionPointToStart(block);
    mlir::Value elementValue = block->getArgument(0);
    mlir::Value accumulatorValue = block->getArgument(1);
    mlir::Value sumValue = builder.create<arith::AddFOp>(
        location, elementValue, accumulatorValue);
    builder.create<linalg::YieldOp>(location, sumValue);
  }
  mlir::Value reducedTensor = linalgReduceOperation.getResult(0);
  vectorReductionOperation.replaceAllUsesWith(reducedTensor);
  tensorizationState.set(vectorReductionOperation.getResult(),
                         reducedTensor);
  tensorizationState.markErase(vectorReductionOperation);
  return linalgReduceOperation.getOperation();
}

static mlir::FailureOr<mlir::Operation *> tensorizeOperation(
    mlir::Operation *operation,
    TensorizationState &tensorizationState) {
  if (mlir::isa<affine::AffineYieldOp>(operation)) {
    return mlir::failure();
  }
  if (auto allocOperation = mlir::dyn_cast<memref::AllocOp>(operation)) {
    return convertAlloc(allocOperation, tensorizationState);
  }
  if (auto collapseShapeOp =
          mlir::dyn_cast<memref::CollapseShapeOp>(operation)) {
    return convertCollapseShape(collapseShapeOp, tensorizationState);
  }
  if (auto expandShapeOp =
          mlir::dyn_cast<memref::ExpandShapeOp>(operation)) {
    return convertExpandShape(expandShapeOp, tensorizationState);
  }
  if (auto toMemrefOperation =
          mlir::dyn_cast<buffer::ToMemrefOp>(operation)) {
    convertToMemref(toMemrefOperation, tensorizationState);
    return toMemrefOperation.getOperation();
  }
  if (auto toTensorOperation =
          mlir::dyn_cast<buffer::ToTensorOp>(operation)) {
    (void)convertToTensor(toTensorOperation, tensorizationState);
    return toTensorOperation.getOperation();
  }
  if (auto transferReadOperation =
          mlir::dyn_cast<vector::TransferReadOp>(operation)) {
    return convertTransferRead(transferReadOperation,
                               tensorizationState);
  }
  if (auto transferWriteOperation =
          mlir::dyn_cast<vector::TransferWriteOp>(operation)) {
    return convertTransferWrite(transferWriteOperation,
                                tensorizationState);
  }
  if (auto createMaskOp =
          mlir::dyn_cast<vector::CreateMaskOp>(operation)) {
    return convertCreateMask(createMaskOp, tensorizationState);
  }
  if (auto affineLoadOperation =
          mlir::dyn_cast<affine::AffineLoadOp>(operation)) {
    return convertAffineLoad(affineLoadOperation, tensorizationState);
  }
  if (auto affineStoreOperation =
          mlir::dyn_cast<affine::AffineStoreOp>(operation)) {
    return convertAffineStore(affineStoreOperation,
                              tensorizationState);
  }
  if (auto vectorReductionOperation =
          mlir::dyn_cast<vector::ReductionOp>(operation)) {
    return convertVectorReduction(vectorReductionOperation,
                                   tensorizationState);
  }
  if (auto affineForOperation =
          mlir::dyn_cast<affine::AffineForOp>(operation)) {
    if (affineForOperation.getNumIterOperands() > 0) {
      return convertAffineForWithIterArgs(affineForOperation,
                                          tensorizationState);
    }
  }
  return convertElementWise(operation, tensorizationState);
}

namespace {
#define GEN_PASS_DECL_VECTORTRANSFERTENSORIZE
#define GEN_PASS_DEF_VECTORTRANSFERTENSORIZE
#include "akg/Dialect/Affine/Passes.h.inc"

struct VectorTransferTensorizePass
    : public impl::VectorTransferTensorizeBase<
          VectorTransferTensorizePass> {
  void getDependentDialects(
      DialectRegistry &dialectRegistry) const override {
    dialectRegistry.insert<affine::AffineDialect, arith::ArithDialect,
                           math::MathDialect, vector::VectorDialect,
                           tensor::TensorDialect,
                           buffer::BufferizationDialect,
                           memref::MemRefDialect, func::FuncDialect,
                           linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    func::FuncOp functionOperation = getOperation();
    TensorizationState tensorizationState(functionOperation.getContext());
    {
      SmallVector<arith::ConstantOp> vectorConstants;
      functionOperation.walk([&](arith::ConstantOp constantOperation) {
        if (isa<VectorType>(constantOperation.getType())) {
          vectorConstants.push_back(constantOperation);
        }
      });
      for (auto constantOperation : vectorConstants) {
        (void)replaceVectorConstantWithTensor(
            constantOperation, tensorizationState.builder);
      }
    }
    bool changed = true;
    unsigned maxIterations = 10;
    unsigned iteration = 0;
    while (changed && iteration < maxIterations) {
      changed = false;
      iteration++;
      SmallVector<Operation *> workList;
      functionOperation.walk<WalkOrder::PostOrder>(
          [&](Operation *operation) {
            if (!operation) {
              return;
            }
            if (tensorizationState.operationsToErase.contains(
                    operation)) {
              return;
            }
            if (tensorizationState.isInvalid(operation)) {
              return;
            }
            if (operation == functionOperation.getOperation()) {
              return;
            }
            if (!operation->getBlock()) {
              return;
            }
            if (isa<func::FuncOp>(operation)) {
              return;
            }
            workList.push_back(operation);
          });
      for (Operation *operation : workList) {
        if (!operation) {
          continue;
        }
        if (tensorizationState.operationsToErase.contains(operation)) {
          continue;
        }
        if (tensorizationState.isInvalid(operation)) {
          continue;
        }
        if (!operation->getBlock()) {
          continue;
        }
        if (isa<func::FuncOp>(operation)) {
          continue;
        }
        tensorizationState.builder.setInsertionPoint(operation);
        auto tensorizeResult =
            tensorizeOperation(operation, tensorizationState);
        if (succeeded(tensorizeResult)) {
          changed = true;
        }
      }
    }
    auto eraseDeadMarkedOperations = [&]() {
      SmallVector<Operation *> operationsToDelete;
      operationsToDelete.reserve(
          tensorizationState.operationsToErase.size());
      for (Operation *operation :
           tensorizationState.operationsToErase) {
        if (!operation || !operation->getBlock()) {
          continue;
        }
        if (operation->use_empty()) {
          operationsToDelete.push_back(operation);
        }
      }
      if (operationsToDelete.empty()) {
        return false;
      }
      for (Operation *operation : operationsToDelete) {
        tensorizationState.operationsToErase.erase(operation);
        if (operation && operation->getBlock()) {
          operation->erase();
        }
      }
      return true;
    };
    while (eraseDeadMarkedOperations()) {
    }
  }
};
}  // namespace

namespace mlir::affine {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createVectorTransferTensorizePass() {
  return std::make_unique<VectorTransferTensorizePass>();
}
}  // namespace mlir::affine

