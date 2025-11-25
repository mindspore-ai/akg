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

//===- VectorTransferTensorize.cpp ------------------------------*- C++ -*-===//
//
// Converts:
//   1) vector.transfer_{read|write}
//   2) scalar / vector arith.constant
//   3) element-wise pure ops into tensor
//
//===----------------------------------------------------------------------===*/

#include "akg/Dialect/Affine/Transforms/VectorTransferTensorize.h"

#include <algorithm>
#include <cstdint>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "vector-transfer-tensorize"

using namespace mlir;  // NOLINT(build/namespaces)

namespace arith  = mlir::arith;
namespace math   = mlir::math;
namespace vector = mlir::vector;
namespace buffer = mlir::bufferization;
namespace memref = mlir::memref;
namespace tensor = mlir::tensor;
namespace func   = mlir::func;
namespace affine = mlir::affine;
namespace func   = mlir::func;

//------------------------------------------------------------------------------
// Utilities
//------------------------------------------------------------------------------

static RankedTensorType convertMemrefToTensorType(MemRefType memrefType) {
  if (!memrefType || !memrefType.hasStaticShape()) return {};
  return RankedTensorType::get(memrefType.getShape(), memrefType.getElementType());
}

static bool isNumericTypeLike(Type type) {
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>())
    return rankedTensorType.getElementType().isIntOrFloat();
  if (auto vectorType = type.dyn_cast<VectorType>())
    return vectorType.getElementType().isIntOrFloat();
  return type.isIntOrFloat();
}

//------------------------------------------------------------------------------
// Tensorization state
//------------------------------------------------------------------------------

struct TensorizationState {
  explicit TensorizationState(MLIRContext *context) : builder(context) {}

  OpBuilder builder;
  IRMapping valueMapping;                       // original value → tensor value
  llvm::DenseSet<Operation *> operationsToErase;

  void map(Value original, Value tensorized) { valueMapping.map(original, tensorized); }
  void replaceMapping(Value original, Value tensorized) {
    valueMapping.erase(original);
    map(original, tensorized);
  }
  Value lookup(Value value) const { return valueMapping.lookupOrNull(value); }
  void markForErasure(Operation *operation) { operationsToErase.insert(operation); }
};

//------------------------------------------------------------------------------
// memref.alloc → tensor.empty
//------------------------------------------------------------------------------

static FailureOr<tensor::EmptyOp> convertAllocOp(memref::AllocOp allocOp, TensorizationState &state) {
  auto tensorType = convertMemrefToTensorType(allocOp.getType());
  if (!tensorType) return failure();

  auto emptyTensor =
      state.builder.create<tensor::EmptyOp>(allocOp.getLoc(), tensorType.getShape(), tensorType.getElementType());

  state.map(allocOp.getResult(), emptyTensor.getResult());
  state.markForErasure(allocOp);
  return emptyTensor;
}

//------------------------------------------------------------------------------
// bufferization.to_{memref|tensor}
//------------------------------------------------------------------------------

static void convertToMemrefOp(buffer::ToMemrefOp toMemrefOp, TensorizationState &state) {
  state.map(toMemrefOp.getResult(), toMemrefOp.getTensor());
  state.markForErasure(toMemrefOp);
}

static LogicalResult convertToTensorOp(buffer::ToTensorOp toTensorOp, TensorizationState &state) {
  Value mappedMemref = state.lookup(toTensorOp.getMemref());
  if (!mappedMemref) return failure();
  toTensorOp.getResult().replaceAllUsesWith(mappedMemref);
  state.markForErasure(toTensorOp);
  return success();
}

//------------------------------------------------------------------------------
// tensor.extract_slice helpers
//------------------------------------------------------------------------------

static FailureOr<tensor::ExtractSliceOp> buildOneDimensionalExtractSlice(OpBuilder &builder, Location loc,
                                                                         Value sourceTensor, ValueRange indices,
                                                                         unsigned sliceLength) {
  auto tensorType = sourceTensor.getType().dyn_cast<RankedTensorType>();
  if (!tensorType || !tensorType.hasStaticShape()) return failure();

  unsigned rank = tensorType.getRank();
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets(indices.begin(), indices.end());
  SmallVector<OpFoldResult> sizes(rank, builder.getIndexAttr(1));

  int64_t fullLength = tensorType.getShape().back();
  int64_t length = sliceLength ? static_cast<int64_t>(sliceLength) : fullLength;
  sizes.back() = builder.getIndexAttr(length);

  auto resultTensorType = RankedTensorType::get({length}, tensorType.getElementType());
  return builder.create<tensor::ExtractSliceOp>(loc, resultTensorType, sourceTensor, offsets, sizes, strides);
}

//------------------------------------------------------------------------------
// const → tensor
//------------------------------------------------------------------------------

static FailureOr<arith::ConstantOp> createTensorConstantFromScalar(arith::ConstantOp scalarConstant,
                                                                   RankedTensorType targetTensorType,
                                                                   OpBuilder &builder) {
  Attribute valueAttr = scalarConstant.getValue();

  if (auto denseAttr = valueAttr.dyn_cast<DenseElementsAttr>()) {
    if (denseAttr.getType() == targetTensorType)
      return builder.create<arith::ConstantOp>(scalarConstant.getLoc(), denseAttr);
    if (denseAttr.getNumElements() == targetTensorType.getNumElements())
      return builder.create<arith::ConstantOp>(scalarConstant.getLoc(), denseAttr.reshape(targetTensorType));
    return failure();
  }

  Attribute elementAttr = valueAttr;
  Type elementType = targetTensorType.getElementType();
  if (auto intAttr = valueAttr.dyn_cast<IntegerAttr>())
    elementAttr = (intAttr.getType() == elementType) ? elementAttr : IntegerAttr::get(elementType, intAttr.getInt());
  else if (auto floatAttr = valueAttr.dyn_cast<FloatAttr>())
    elementAttr =
        (floatAttr.getType() == elementType) ? elementAttr : FloatAttr::get(elementType, floatAttr.getValue());
  else
    return failure();

  SmallVector<Attribute> repeatedValues(targetTensorType.getNumElements(), elementAttr);
  auto denseAttr = DenseElementsAttr::get(targetTensorType, repeatedValues);
  return builder.create<arith::ConstantOp>(scalarConstant.getLoc(), denseAttr);
}

static LogicalResult upgradeConstantToTensor(arith::ConstantOp constantOp, RankedTensorType targetTensorType,
                                             TensorizationState &state) {
  if (state.lookup(constantOp.getResult())) return success();
  auto tensorConstantOr = createTensorConstantFromScalar(constantOp, targetTensorType, state.builder);
  if (failed(tensorConstantOr)) return failure();
  state.map(constantOp.getResult(), (*tensorConstantOr).getResult());
  return success();
}

//------------------------------------------------------------------------------
// transfer_read / transfer_write
//------------------------------------------------------------------------------

static FailureOr<Operation *> convertTransferReadOp(vector::TransferReadOp readOp, TensorizationState &state) {
  Value sourceTensor = state.lookup(readOp.getSource());
  if (!sourceTensor) return failure();

  arith::ConstantOp paddingConstant =
      readOp.getPadding() ? readOp.getPadding().getDefiningOp<arith::ConstantOp>() : nullptr;

  unsigned sliceLength = readOp.getVectorType().getNumElements();
  auto extractSliceOr =
      buildOneDimensionalExtractSlice(state.builder, readOp.getLoc(), sourceTensor, readOp.getIndices(), sliceLength);
  if (failed(extractSliceOr)) return failure();

  state.map(readOp.getResult(), (*extractSliceOr).getResult());
  state.markForErasure(readOp);
  if (paddingConstant) state.markForErasure(paddingConstant.getOperation());

  return extractSliceOr->getOperation();
}

static FailureOr<Operation *> convertTransferWriteOp(vector::TransferWriteOp writeOp, TensorizationState &state) {
  Value destinationTensor = state.lookup(writeOp.getSource());
  if (!destinationTensor) return failure();

  unsigned sliceLength = writeOp.getVectorType().getNumElements();
  auto sliceTensorType = RankedTensorType::get({sliceLength}, writeOp.getVectorType().getElementType());

  Value sourceSlice = state.lookup(writeOp.getVector());
  if (!sourceSlice) {
    if (auto constantOp = writeOp.getVector().getDefiningOp<arith::ConstantOp>()) {
      auto tensorConstantOr = createTensorConstantFromScalar(constantOp, sliceTensorType, state.builder);
      if (failed(tensorConstantOr)) return failure();
      sourceSlice = (*tensorConstantOr).getResult();
    }
  }
  if (!sourceSlice || sourceSlice.getType() != sliceTensorType) return failure();

  auto destinationTensorType = destinationTensor.getType().cast<RankedTensorType>();
  unsigned rank = destinationTensorType.getRank();
  SmallVector<OpFoldResult> strides(rank, state.builder.getIndexAttr(1));
  SmallVector<OpFoldResult> sizes(rank, state.builder.getIndexAttr(1));
  sizes.back() = state.builder.getIndexAttr(sliceLength);
  SmallVector<OpFoldResult> offsets(writeOp.getIndices().begin(), writeOp.getIndices().end());

  auto insertSliceOp = state.builder.create<tensor::InsertSliceOp>(writeOp.getLoc(), sourceSlice, destinationTensor,
                                                                   offsets, sizes, strides);

  affine::AffineForOp currentLoop = writeOp->getParentOfType<affine::AffineForOp>();
  if (!currentLoop) {
    state.markForErasure(writeOp);
    return insertSliceOp.getOperation();
  }

  Value currentInsideValue = insertSliceOp.getResult();
  Value currentDestination = destinationTensor;
  Value originalInitTensor = destinationTensor;
  Operation *outermostChangedLoop = nullptr;

  while (currentLoop) {
    IRRewriter rewriter(currentLoop.getContext());

    auto newLoop = cast<affine::AffineForOp>(*currentLoop.replaceWithAdditionalYields(
        rewriter, /*initOperand=*/originalInitTensor, /*replaceInitOperandUsesInLoop=*/false,
        [&](OpBuilder &b, Location, ArrayRef<BlockArgument>) { return SmallVector<Value>{currentInsideValue}; }));

    outermostChangedLoop = newLoop.getOperation();

    Value yieldedInsideValue = newLoop.getBody()->getArguments().back();
    Value yieldedOutsideValue = newLoop.getResults().back();

    auto replaceInsideUses = [&](OpOperand &use) -> bool { return newLoop->isProperAncestor(use.getOwner()); };
    currentDestination.replaceUsesWithIf(yieldedInsideValue,
                                         mlir::function_ref<bool(OpOperand &)>(replaceInsideUses));

    unsigned initOperandPosition = newLoop->getNumOperands() - 1;
    auto replaceOutsideUses = [&](OpOperand &use) -> bool {
      Operation *user = use.getOwner();
      if (user == newLoop.getOperation() && use.getOperandNumber() == static_cast<int>(initOperandPosition))
        return false;
      if (newLoop->isProperAncestor(user)) return false;
      return true;
    };
    currentDestination.replaceUsesWithIf(yieldedOutsideValue,
                                         mlir::function_ref<bool(OpOperand &)>(replaceOutsideUses));

    currentInsideValue = yieldedOutsideValue;
    currentLoop = newLoop->getParentOfType<affine::AffineForOp>();
  }

  state.replaceMapping(destinationTensor, currentInsideValue);
  state.markForErasure(writeOp);
  state.markForErasure(insertSliceOp.getOperation());
  return outermostChangedLoop ? outermostChangedLoop : insertSliceOp.getOperation();
}

//------------------------------------------------------------------------------
// Element-wise tensorization
//------------------------------------------------------------------------------

static FailureOr<Operation *> cloneElementWiseOp(Operation *originalOp, ArrayRef<Value> newOperands,
                                                 TensorizationState &state) {
  OperationState newState(originalOp->getLoc(), originalOp->getName());
  newState.addOperands(newOperands);
  newState.addAttributes(originalOp->getAttrs());

  SmallVector<Type> resultTensorTypes;
  for (Type originalResultType : originalOp->getResultTypes()) {
    if (auto rankedTensorType = originalResultType.dyn_cast<RankedTensorType>())
      resultTensorTypes.push_back(rankedTensorType);
    else if (auto vectorType = originalResultType.dyn_cast<VectorType>())
      resultTensorTypes.push_back(
          RankedTensorType::get(vectorType.getShape(), vectorType.getElementType()));
    else
      return failure();
  }
  newState.addTypes(resultTensorTypes);

  Operation *newOp = state.builder.create(newState);
  for (auto [oldResult, newResult] : llvm::zip(originalOp->getResults(), newOp->getResults()))
    state.map(oldResult, newResult);

  state.markForErasure(originalOp);
  return newOp;
}

static FailureOr<Operation *> convertElementWiseOp(Operation *op, TensorizationState &state) {
  if (op->getNumRegions() != 0) return failure();

  if (llvm::any_of(op->getOperandTypes(), [](Type t) { return !isNumericTypeLike(t); })) {
    return failure();
  }

  if (llvm::any_of(op->getResultTypes(), [](Type t) { return !isNumericTypeLike(t); })) {
    return failure();
  }

  RankedTensorType referenceTensorType = nullptr;
  for (Value operand : op->getOperands()) {
    if (auto mapped = state.lookup(operand)) {
      if (auto rankedType = mapped.getType().dyn_cast<RankedTensorType>()) {
        referenceTensorType = rankedType;
        break;
      }
    }
    if (auto rankedType = operand.getType().dyn_cast<RankedTensorType>()) {
      referenceTensorType = rankedType;
      break;
    }
  }
  if (!referenceTensorType) return failure();

  SmallVector<Value> newOperands;
  for (Value operand : op->getOperands()) {
    if (auto mapped = state.lookup(operand)) {
      newOperands.push_back(mapped);
      continue;
    }
    if (operand.getType() == referenceTensorType) {
      newOperands.push_back(operand);
      continue;
    }
    if (auto constantOp = operand.getDefiningOp<arith::ConstantOp>()) {
      if (failed(upgradeConstantToTensor(constantOp, referenceTensorType, state))) return failure();
      newOperands.push_back(state.lookup(constantOp.getResult()));
      continue;
    }
    return failure();
  }
  return cloneElementWiseOp(op, newOperands, state);
}

//------------------------------------------------------------------------------
// Dispatch
//------------------------------------------------------------------------------

static FailureOr<Operation *> tensorizeOperation(Operation *op, TensorizationState &state) {
  if (auto allocOp = dyn_cast<memref::AllocOp>(op)) return convertAllocOp(allocOp, state);

  if (auto toMemrefOp = dyn_cast<buffer::ToMemrefOp>(op)) {
    convertToMemrefOp(toMemrefOp, state);
    return toMemrefOp.getOperation();
  }
  if (auto toTensorOp = dyn_cast<buffer::ToTensorOp>(op)) {
    (void)convertToTensorOp(toTensorOp, state);
    return toTensorOp.getOperation();
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) return convertTransferReadOp(readOp, state);
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) return convertTransferWriteOp(writeOp, state);

  return convertElementWiseOp(op, state);
}

namespace {
#define GEN_PASS_DECL_VECTORTRANSFERTENSORIZE
#define GEN_PASS_DEF_VECTORTRANSFERTENSORIZE
#include "akg/Dialect/Affine/Passes.h.inc"

struct VectorTransferTensorizePass
    : public impl::VectorTransferTensorizeBase<VectorTransferTensorizePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, math::MathDialect, vector::VectorDialect, tensor::TensorDialect,
                    buffer::BufferizationDialect, memref::MemRefDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    TensorizationState state(funcOp.getContext());

    SmallVector<Operation *> workList;
    funcOp.walk([&](Operation *operation) { workList.push_back(operation); });

    for (size_t idx = 0; idx < workList.size(); ++idx) {
      Operation *op = workList[idx];
      if (!op || op->getBlock() == nullptr) continue;
      if (state.operationsToErase.contains(op)) continue;
      if (isa<func::FuncOp>(op)) continue;
      state.builder.setInsertionPoint(op);
      (void)tensorizeOperation(op, state);
    }

    auto eraseDeadMarkedOps = [&]() {
      SmallVector<Operation *> toDelete;
      for (Operation *operation : state.operationsToErase)
        if (operation->use_empty()) toDelete.push_back(operation);
      for (Operation *operation : toDelete) {
        state.operationsToErase.erase(operation);
        operation->erase();
      }
      return !toDelete.empty();
    };
    while (eraseDeadMarkedOps()) {}

    bool memrefOpsRemain = false;
    funcOp.walk([&](Operation *operation) {
      if (operation->getDialect()->getNamespace() == memref::MemRefDialect::getDialectNamespace()) {
        memrefOpsRemain = true;
        operation->emitError("memref op survived tensor-only conversion");
      }
    });
    if (memrefOpsRemain) signalPassFailure();
  }
};
}  // namespace

namespace mlir::affine {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createVectorTransferTensorizePass() {
  return std::make_unique<VectorTransferTensorizePass>();
}
}  // namespace mlir::affine
