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

#include "akg/Dialect/Linalg/Transforms/TemplateOpFusion.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/Linalg/Utils/FusionUtils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/SuperVectorize.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DECL_LINALGTEMPLATEOPOPFUSION
#define GEN_PASS_DEF_LINALGTEMPLATEOPOPFUSION
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalgExt;

#define DEBUG_TYPE "template-op-fusion"

/// Append to `fusedOpIndexingMapAttrs` the indexing maps for the operands of
/// the `producer` to use in the fused operation given the indexing map of the
/// result of the producer in the consumer.
static AffineMap getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(OpOperand *producerOpOperand,
                                                                        AffineMap producerResultIndexMap,
                                                                        AffineMap fusedConsumerArgIndexMap) {
  // The indexing map in the consumer op (fusedConsumerArgIndexMap) is a map
  // from consumer loop -> consumer arg tensor index/producer result tensor
  // index. The fused loop is same as the consumer loop. For each producer arg
  // the indexing map to be computed is a map from consumer loop -> producer
  // arg tensor index.
  // producerResultIndexMap is a map from producer loop -> tensor index.
  // Compute the inverse to get map from tensor index -> producer loop.
  // The inverse is a map from producer result tensor index -> producer loop.
  AffineMap invProducerResultIndexMap = inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap && "expected producer result indexing map to be invertible");

  LinalgOp producer = cast<LinalgOp>(producerOpOperand->getOwner());
  // argMap is a map from producer loop -> producer arg tensor index.
  AffineMap argMap = producer.getMatchingIndexingMap(producerOpOperand);

  // Compose argMap with invProducerResultIndexMap to get a map from
  // producer result tensor index -> producer arg tensor index.
  AffineMap t1 = argMap.compose(invProducerResultIndexMap);

  // Compose t1 with fusedConsumerArgIndexMap gives an indexing map from
  // consumer loop/ fused loop -> producer arg tensor index.
  return t1.compose(fusedConsumerArgIndexMap);
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isIdentityWithBroadcast(AffineMap map) const {
  SmallVector<unsigned, 2> broadcastDims;
  bool isMinorIdentityWithBroadcasting = map.isMinorIdentityWithBroadcasting(&broadcastDims);
  if (!isMinorIdentityWithBroadcasting) {
    return false;
  }
  // constraint: broadcast dims must be in high dims
  auto size = broadcastDims.size();
  return llvm::none_of(broadcastDims, [&](unsigned id) { return id >= size; });
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isIdentityWithPermutation(AffineMap map) const {
  if (!map.isProjectedPermutation(false)) {
    return false;
  }

  llvm::SmallVector<unsigned int> permutateDims;
  if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutateDims)) {
    return false;
  }

  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isIdentityWithPermutation(
  AffineMap map, llvm::SmallVector<unsigned int> &permutationDims) const {
  if (!map.isProjectedPermutation(false)) {
    return false;
  }

  if (!map.isPermutationOfMinorIdentityWithBroadcasting(permutationDims)) {
    return false;
  }

  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isMapFusable(llvm::SmallVectorImpl<AffineMap> &maps,
                                                           Attribute fusionKindAttr) const {
  if (!fusionKindAttr.isa<StringAttr>()) {
    return false;
  }

  auto fusionKindStr = fusionKindAttr.cast<StringAttr>().getValue();
  if (fusion::FusionDialect::isIdentityFusion(fusionKindStr)) {
    return llvm::all_of(maps, [](AffineMap map) { return map.isIdentity(); });
  }

  if (fusion::FusionDialect::isIdentityWithBroadPermFusion(fusionKindStr)) {
    return llvm::all_of(maps, [&](AffineMap map) {
      if (isIdentityWithBroadcast(map)) {
        return true;
      }

      if (isIdentityWithPermutation(map)) {
        return true;
      }

      // not identity map or broadcast map or permutation map
      return false;
    });
  }

  if (fusion::FusionDialect::isInvertibleFusion(fusionKindStr)) {
    // no other requirements
    return true;
  }

  return false;
}

template <bool isFrontFusion, typename MatchOp>
unsigned int FuseTemplateOps<isFrontFusion, MatchOp>::getShapeSize(Type type) const {
  unsigned int shapeSize = 1;
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    if (shapedType.hasRank()) {
      shapeSize = (unsigned int)shapedType.getRank();
    }
  }

  return shapeSize;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isTemplateFuncFusable(func::FuncOp funcOp, Value funcOperand,
                                                                    unsigned int &shapeSize) const {
  // check if template func contains fusion load and fusion insert for
  // corresponding operand
  bool findFusionLoad = false;
  bool findFusionStore = false;
  bool findFusionInsert = false;
  bool findFusionMultiLoad = false;

  bool isLoadShapeNoRank = false;
  bool isStoreShapeNoRank = false;
  Value funcMemRefOperand;

  funcOp->walk([&](Operation *op) {
    if constexpr (isFrontFusion) {
      if (findFusionLoad && findFusionInsert && findFusionMultiLoad) {
        // skip if already found
        return WalkResult::skip();
      }
    } else {
      if (findFusionStore && findFusionInsert && findFusionMultiLoad) {
        // skip if already found
        return WalkResult::skip();
      }
    }

    if (isa<bufferization::ToMemrefOp>(op)) {
      bufferization::ToMemrefOp toMemrefOp = cast<bufferization::ToMemrefOp>(op);

      if (toMemrefOp.getTensor() == funcOperand) {
        funcMemRefOperand = toMemrefOp.getResult();
      }
    }

    if (isa<fusion::LoadOp>(op)) {
      fusion::LoadOp loadOp = cast<fusion::LoadOp>(op);

      if (loadOp.getMemRef() == funcMemRefOperand) {
        shapeSize = getShapeSize(loadOp.getResult().getType());
        findFusionLoad = true;
      }

      if (auto shapedType = dyn_cast<ShapedType>(loadOp.getResult().getType())) {
        isLoadShapeNoRank = !shapedType.hasRank();
      }

      return WalkResult::advance();
    }

    if (isa<fusion::StoreOp>(op)) {
      fusion::StoreOp storeOp = cast<fusion::StoreOp>(op);

      if (storeOp.getMemRef() == funcMemRefOperand) {
        shapeSize = getShapeSize(storeOp.getValueToStore().getType());
        findFusionStore = true;
      }

      if (auto shapedType = dyn_cast<ShapedType>(storeOp.getValueToStore().getType())) {
        isStoreShapeNoRank = !shapedType.hasRank();
      }

      return WalkResult::advance();
    }

    if (isa<fusion::MultiLoadOp>(op)) {
      fusion::MultiLoadOp multiLoadOp = cast<fusion::MultiLoadOp>(op);

      if (multiLoadOp.getMemRef() == funcMemRefOperand) {
        findFusionMultiLoad = true;
      }

      return WalkResult::advance();
    }

    if (isa<fusion::InsertOp>(op)) {
      fusion::InsertOp insertOp = cast<fusion::InsertOp>(op);

      if (insertOp.getMemref() == funcMemRefOperand) {
        findFusionInsert = true;
      }

      return WalkResult::advance();
    }

    return WalkResult::advance();
  });

  if constexpr (isFrontFusion) {
    return findFusionLoad && findFusionInsert && findFusionMultiLoad && !isLoadShapeNoRank;
  } else {
    return findFusionStore && findFusionInsert && findFusionMultiLoad && !isStoreShapeNoRank;
  }
}

template <>
unsigned int FuseTemplateOps<true>::getOperandIndexInTemplateOp(TemplateOp templateOp, OpOperand *operand) const {
  auto operands = templateOp.getOperands();
  auto it = llvm::find(operands, operand->get());
  assert(it != operands.end() && "find operand failed");

  unsigned operandIndx = it - operands.begin();
  return operandIndx;
}

template <>
unsigned int FuseTemplateOps<false>::getOperandIndexInTemplateOp(TemplateOp templateOp, OpOperand *operand) const {
  auto results = templateOp.getResults();
  auto it = llvm::find(results, operand->get());
  assert(it != results.end() && "find operand failed");

  unsigned operandIndx = it - results.begin() + (unsigned)templateOp.getNumDpsInputs();
  return operandIndx;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::canTemplateOpFuseOperand(TemplateOp templateOp, GenericOp genericOp,
                                                                       OpOperand *fusedOperand) const {
  // get template func op
  auto fnSym = templateOp.getOperation()->getAttr(TemplateFuncAttrName).cast<SymbolRefAttr>();
  auto funcOp = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(templateOp, fnSym));

  // verify if template op has fusion dialect for the operand
  unsigned operandIndx = getOperandIndexInTemplateOp(templateOp, fusedOperand);
  auto funcOperand = funcOp.getArgument(operandIndx);
  unsigned int shapeSize;
  if (!isTemplateFuncFusable(funcOp, funcOperand, shapeSize)) {
    LLVM_DEBUG(llvm::dbgs() << "canTemplateOpFuseOperand failure: isTemplateFuncFusable failed\n");
    return false;
  }

  // verify if the map can be fused
  auto invGenericFusedOperandIndexMap = getInvGenericOperandIndexMap(genericOp, fusedOperand);
  if (!invGenericFusedOperandIndexMap.has_value()) {
    LLVM_DEBUG(llvm::dbgs() << "canTemplateOpFuseOperand failure: "
                               "invGenericFusedOperandIndexMap failed\n");
    return false;
  }
  llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandMaps =
    getFusedOperandToAllMaps(genericOp, invGenericFusedOperandIndexMap.value());

  llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandVecMaps;
  for (size_t i = 0; i < genericFuseOperand2AllOperandMaps.size(); i++) {
    genericFuseOperand2AllOperandVecMaps.emplace_back(
      getMinorSubMapWithPadZero(genericFuseOperand2AllOperandMaps[i], shapeSize));
  }

  auto fusionKindAttr = funcOp->getAttr(fusion::FusionDialect::getFusionKindAttrName());
  if (!isMapFusable(genericFuseOperand2AllOperandVecMaps, fusionKindAttr)) {
    LLVM_DEBUG(llvm::dbgs() << "canTemplateOpFuseOperand failure: isMapFusable failed\n");
    return false;
  }

  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isReductionAndRemoveSize(TemplateOp templateOp, GenericOp genericOp,
                                                                       OpOperand *fusedOperand) const {
  // Ensure that the fusion does not remove size information required to
  // get the loop bounds. For non-reduction templateOps, this is trivially the
  // case due to the output operand. For reductions, we need to check that after
  // the fusion, each loop dimension has at least one input that defines it.
  if (templateOp.getNumReductionLoops() == 0) {
    return true;
  }

  BitVector coveredDims(templateOp.getNumLoops(), false);
  auto addToCoveredDims = [&](AffineMap map) {
    for (auto result : map.getResults()) {
      if (auto dimExpr = result.dyn_cast<AffineDimExpr>()) {
        coveredDims[dimExpr.getPosition()] = true;
      }
    }
  };

  llvm::SmallVector<Value> ioValues;
  if constexpr (isFrontFusion) {
    ioValues.append(templateOp.getOperands().begin(), templateOp.getOperands().end());
  } else {
    ioValues.append(templateOp.getInputs().begin(), templateOp.getInputs().end());
    ioValues.append(templateOp.getResults().begin(), templateOp.getResults().end());
  }

  for (auto pair : llvm::zip(ioValues, templateOp.getIndexingMapsArray())) {
    Value operand = std::get<0>(pair);
    if (operand == fusedOperand->get()) {
      continue;
    }
    AffineMap operandMap = std::get<1>(pair);

    addToCoveredDims(operandMap);
  }

  OpOperandVector genericCheckOperands;
  AffineMap templateOpFusedOperandIndexMap;
  AffineMap genericFusedOperandIndexMap;
  if constexpr (isFrontFusion) {
    genericCheckOperands = genericOp.getDpsInputOperands();

    templateOpFusedOperandIndexMap = templateOp.getMatchingIndexingMap(fusedOperand);
    genericFusedOperandIndexMap = genericOp.getIndexingMapMatchingResult(fusedOperand->get().cast<OpResult>());
  } else {
    auto concatRanges = llvm::concat<OpOperand *>(genericOp.getDpsInputOperands(), genericOp.getDpsInitOperands());
    genericCheckOperands.append(concatRanges.begin(), concatRanges.end());

    templateOpFusedOperandIndexMap = templateOp.getIndexingMapMatchingResult(fusedOperand->get().cast<OpResult>());
    genericFusedOperandIndexMap = genericOp.getMatchingIndexingMap(fusedOperand);
  }

  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    AffineMap newIndexingMap = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
      operand, genericFusedOperandIndexMap, templateOpFusedOperandIndexMap);
    addToCoveredDims(newIndexingMap);
  }
  if (!coveredDims.all()) {
    return false;
  }

  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::isFusionableOp(LinalgOp op) const {
  auto fusionFlagAttr = op->getAttr(fusion::FusionDialect::getFusionFlagAttrName());
  if (!fusionFlagAttr) {
    // regard it as fusionable op if not set fusion.flag
    return true;
  }

  assert(fusionFlagAttr.isa<StringAttr>() && "fusion.flag should be string");

  return fusion::FusionDialect::isFusionableOp(fusionFlagAttr.cast<StringAttr>().getValue());
}

template <>
bool FuseTemplateOps<true>::isWriteOnlyOutputs(LinalgOp op) const {
  for (const auto &outputOpOperand : llvm::enumerate(op.getDpsInitOperands())) {
    BlockArgument outputArg = op.getRegionOutputArgs()[outputOpOperand.index()];
    if (outputArg.hasOneUse()) {
      return false;
    }
  }
  return true;
}

template <>
bool FuseTemplateOps<false>::isWriteOnlyOutputs(LinalgOp op) const {
  if (!isa<TemplateOp>(op)) {
    return false;
  }
  auto fnSym = op.getOperation()->getAttr(TemplateFuncAttrName).cast<SymbolRefAttr>();
  auto funcOp = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(op, fnSym));
  auto outputBegin = op.getNumDpsInputs();
  auto outputEnd = outputBegin + op.getNumDpsInits();
  for (auto resultIndex : llvm::seq<long>(outputBegin, outputEnd)) {
    if (auto accessAttr = funcOp.getArgAttrOfType<StringAttr>(
          resultIndex, bufferization::BufferizationDialect::kBufferAccessAttrName)) {
      StringRef str = accessAttr.getValue();
      if (str != "write") {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::canGenericOpFuseable(TemplateOp &templateOp, GenericOp &genericOp,
                                                                   OpOperand *fusedOperand) const {
  auto fnSym = templateOp->getAttr(TemplateFuncAttrName).cast<SymbolRefAttr>();
  auto funcOp = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(templateOp, fnSym));
  unsigned operandIndx = getOperandIndexInTemplateOp(templateOp, fusedOperand);
  auto funcOperand = funcOp.getArgument(operandIndx);
  Value funcMemRefOperand;
  bool needVectorize = false;
  funcOp->walk([&](Operation *op) {
    if (isa<bufferization::ToMemrefOp>(op)) {
      bufferization::ToMemrefOp toMemrefOp = cast<bufferization::ToMemrefOp>(op);

      if (toMemrefOp.getTensor() == funcOperand) {
        funcMemRefOperand = toMemrefOp.getResult();
      }
    }

    if (isa<fusion::InsertOp>(op)) {
      fusion::InsertOp insertOp = cast<fusion::InsertOp>(op);
      if (insertOp.getMemref() == funcMemRefOperand) {
        needVectorize |= insertOp.getResult().getType().isa<VectorType>();
        return WalkResult::advance();
      }
    }
    return WalkResult::advance();
  });

  Block &genericBlock = genericOp->getRegion(0).front();
  for (auto &op : genericBlock.without_terminator()) {
    if (needVectorize && isa<arith::ConstantOp>(op)) {
      return false;
    }
    // check if the operands defined out of operation block are constant ops.
    if (!checkConstantOpOutOfBlock(op, nullptr, needVectorize)) {
      LLVM_DEBUG(llvm::dbgs() << "canGenericOpFuseable failure: genericOp use external var \n");
      return false;
    }
  }

  auto isGenericOpInputAllTensor = llvm::all_of(
    genericOp.getDpsInputOperands(), [](OpOperand *oper) { return oper->get().getType().isa<TensorType>(); });
  if (!isGenericOpInputAllTensor) {
    LLVM_DEBUG(llvm::dbgs() << "canGenericOpFuseable failure: "
                               "isGenericOpInputAllTensor failed \n");
    return false;
  }

  // Check the genericOp has all "parallel" iterator type.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops()) {
    LLVM_DEBUG(llvm::dbgs() << "canGenericOpFuseable failure: genericOp loops failed \n");
    return false;
  }
  return true;
}

template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::areTemplateOpsFusable(OpOperand *fusedOperand) const {
  auto producer = fusedOperand->get().getDefiningOp<ProducerOp>();
  auto consumer = dyn_cast<ConsumerOp>(fusedOperand->getOwner());

  GenericOp genericOp;
  TemplateOp templateOp;
  if constexpr (isFrontFusion) {
    genericOp = producer;
    templateOp = consumer;
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable : " << templateOp << " operand " << fusedOperand->get() << "\n");
  } else {
    templateOp = producer;
    genericOp = consumer;
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable : " << genericOp << " operand " << fusedOperand->get() << "\n");
  }

  // Check producer and consumer are generic ops.
  if (producer == nullptr || consumer == nullptr) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: op failed \n");
    return false;
  }

  if (!isFusionableOp(producer) || !isFusionableOp(consumer)) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: op fusionable failed \n");
    return false;
  }

  if (!isWriteOnlyOutputs(producer)) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: op result read failed \n");
    return false;
  }

  // Producer and consumer must have tensor semantics.
  if (!producer.hasTensorSemantics() || !consumer.hasTensorSemantics()) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: hasTensorSemantics failed \n");
    return false;
  }

  if (producer.hasIndexSemantics() || consumer.hasIndexSemantics()) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: hasIndexSemantics failed \n");
    return false;
  }

  // Fused operand must be tensor type
  if (!fusedOperand->get().getType().isa<TensorType>()) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: tensor type failed \n");
    return false;
  }

  // Only allow fusing the producer of an input operand for now.
  if (!consumer.isDpsInput(fusedOperand) || !producer->hasOneUse()) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: one use failed \n");
    return false;
  }

  if (genericOp.getNumDpsInits() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: genericOp init failed \n");
    return false;
  }

  if constexpr (isFrontFusion) {
    // Get the consumer index map. The number of results of the consumer index
    // map must match the number of loops of the producer.
    AffineMap consumerIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
    if (consumerIndexMap.getNumResults() != producer.getNumLoops()) {
      LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: consumer index failed \n");
      return false;
    }
  } else {
    // Get the consumer index map. The number of results of the consumer index
    // map must match the number of non reduction loops of the producer.
    AffineMap consumerIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
    if (consumerIndexMap.getNumResults() != producer.getNumLoops() - producer.getNumReductionLoops()) {
      LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: consumer index failed \n");
      return false;
    }
  }

  if constexpr (isFrontFusion) {
    if (producer.getNumDpsInputs() == 0) {
      LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: producer input failed \n");
      return false;
    }
  }

  if constexpr (!isFrontFusion) {
    if (fusedOperand->get().getType() != genericOp.getResult(0).getType()) {
      return false;
    }
  }

  // Check the genericOp
  if (!canGenericOpFuseable(templateOp, genericOp, fusedOperand)) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: canGenericOpFuseable failed \n");
    return false;
  }

  // CHECK if template op contains fusion.load and fusion insert for
  // corresponding arg
  if (!canTemplateOpFuseOperand(templateOp, genericOp, fusedOperand)) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: canTemplateOpFuseOperand failed \n");
    return false;
  }

  if (!isReductionAndRemoveSize(templateOp, genericOp, fusedOperand)) {
    LLVM_DEBUG(llvm::dbgs() << "areTemplateOpsFusable failure: isReductionAndRemoveSize failed \n");
    return false;
  }
  return true;
}

template <bool isFrontFusion, typename MatchOp>
Value FuseTemplateOps<isFrontFusion, MatchOp>::castTensorToMemref(OpBuilder builder, Value value) const {
  Type type = value.getType();
  if (auto rankedTensorType = type.dyn_cast<RankedTensorType>()) {
    Type memrefType = MemRefType::get(rankedTensorType.getShape(), rankedTensorType.getElementType());
    auto insertedOp = builder.create<bufferization::ToMemrefOp>(value.getLoc(), memrefType, value);
    return insertedOp.getResult();
  }

  if (auto unrankedTensorType = type.dyn_cast<UnrankedTensorType>()) {
    Type memrefType = UnrankedMemRefType::get(unrankedTensorType.getElementType(), {});
    auto insertedOp = builder.create<bufferization::ToMemrefOp>(value.getLoc(), memrefType, value);
    return insertedOp.getResult();
  }

  // not tensor and no need to cast
  return value;
}

template <bool isFrontFusion, typename MatchOp>
Type FuseTemplateOps<isFrontFusion, MatchOp>::castElementType(Type type, Type elementType) const {
  if (auto shapedtype = type.dyn_cast<ShapedType>()) {
    return shapedtype.clone(elementType);
  } else {
    return elementType;
  }
}

template <bool isFrontFusion, typename MatchOp>
Type FuseTemplateOps<isFrontFusion, MatchOp>::castToDynShape(Type v) const {
  if (auto shapeType = v.dyn_cast<ShapedType>()) {
    llvm::SmallVector<int64_t> shape(shapeType.getRank(), ShapedType::kDynamic);
    return shapeType.clone(shape);
  } else {
    return v;
  }
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::replaceFuncParamType(func::FuncOp func, int64_t index,
                                                                   Type newType) const {
  auto origFuncType = func.getFunctionType();
  auto inputsType = origFuncType.getInputs();
  auto resultsType = origFuncType.getResults();

  // change type of index-th argument
  func.getArgument(index).setType(newType);

  // modify function type of func accordingly
  SmallVector<Type, 2> newInputsType;
  for (size_t i = 0; i < inputsType.size(); i++) {
    if (i == (size_t)index) {
      newInputsType.push_back(func.getArgument(index).getType());
    } else {
      newInputsType.push_back(inputsType[i]);
    }
  }

  auto newFuncType = FunctionType::get(origFuncType.getContext(), newInputsType, resultsType);
  func.setFunctionType(newFuncType);
}

// check if the operands defined out of operation block are all constant ops.
// collect the constant operands defined out of operation block.
template <bool isFrontFusion, typename MatchOp>
bool FuseTemplateOps<isFrontFusion, MatchOp>::checkConstantOpOutOfBlock(Operation &op,
                                                                        SmallPtrSet<Operation *, 4> *constantOps,
                                                                        bool needVectorize) const {
  for (auto operand : op.getOperands()) {
    if (isa<BlockArgument>(operand)) {
      continue;
    }
    auto defOp = operand.getDefiningOp();
    if (!defOp) {
      LLVM_DEBUG(llvm::dbgs() << "Operand out of the Block.\n");
      return false;
    }
    if (op.getBlock() != defOp->getBlock()) {
      if (!isa<arith::ConstantOp>(defOp) || needVectorize) {
        LLVM_DEBUG(llvm::dbgs() << "Operation can not fuse into template.\n");
        return false;
      }
      if (constantOps != nullptr) {
        (void)constantOps->insert(defOp);
      }
    }
  }
  return true;
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::insertOperations(fusion::InsertOp insertOp, GenericOp genericOp,
                                                               OpOperand *genericFusedOperand,
                                                               const SmallVectorImpl<Value> &newFuseFuncLoads,
                                                               MapType mapType) const {
  auto recordMapper = [&](Block &genericOpBlock, IRMapping &mapper) {
    int64_t index = 0;
    for (auto pair : llvm::zip(genericOp.getDpsInputOperands(), genericOpBlock.getArguments())) {
      if (std::get<0>(pair) == genericFusedOperand) {
        if constexpr (isFrontFusion) {
          if (mapType != IDENTITY) {
            mapper.map(std::get<1>(pair), insertOp.getData());
          } else {
            mapper.map(std::get<1>(pair), insertOp.getResult());
          }
        } else {
          mapper.map(std::get<1>(pair), insertOp.getData());
        }
        continue;
      }
      mapper.map(std::get<1>(pair), newFuseFuncLoads[index]);
      index++;
    }
  };

  auto replaceInsertResult = [&](Block &genericOpBlock, IRMapping &mapper, SmallPtrSet<Operation *, 4> &newOps) {
    auto genericYieldOp = cast<linalg::YieldOp>(genericOpBlock.getTerminator());
    auto newArg = mapper.lookup(genericYieldOp.getOperand(0));
    if constexpr (isFrontFusion) {
      insertOp.getResult().replaceAllUsesExcept(newArg, newOps);
    } else {
      insertOp.getDataMutable().assign(newArg);
    }
  };

  bool needVectorize = insertOp.getResult().getType().isa<VectorType>();
  if (!needVectorize) {
    OpBuilder builder(insertOp);
    if constexpr (isFrontFusion) {
      builder.setInsertionPointAfter(insertOp);
    }

    IRMapping mapper;
    Block &genericOpBlock = genericOp->getRegion(0).front();
    recordMapper(genericOpBlock, mapper);

    SmallPtrSet<Operation *, 4> newOps;

    // collect the constant operands defined out of operation block.
    SmallPtrSet<Operation *, 4> constantOps;
    for (auto &op : genericOpBlock.without_terminator()) {
      (void)checkConstantOpOutOfBlock(op, &constantOps, needVectorize);
    }
    for (auto constantOp : constantOps) {
      auto newOp = builder.clone(*constantOp, mapper);
      newOps.insert(newOp);
    }

    for (auto &op : genericOpBlock.without_terminator()) {
      if (!isa<IndexOp>(op)) {
        auto newOp = builder.clone(op, mapper);
        newOps.insert(newOp);
      }
    }

    replaceInsertResult(genericOpBlock, mapper, newOps);
  } else {
    vector::VectorizationState state(insertOp.getContext());
    if constexpr (isFrontFusion) {
      state.builder.setInsertionPointAfter(insertOp);
    } else {
      state.builder.setInsertionPoint(insertOp);
    }

    VectorizationStrategy strategy;
    auto vectorShape = insertOp.getResult().getType().cast<VectorType>().getShape();
    strategy.vectorSizes.assign(vectorShape.begin(), vectorShape.end());
    state.strategy = &strategy;

    Block &genericOpBlock = genericOp->getRegion(0).front();
    recordMapper(genericOpBlock, state.valueVectorReplacement);

    SmallPtrSet<Operation *, 4> newOps;
    for (auto &op : genericOpBlock.without_terminator()) {
      if (!isa<IndexOp>(op)) {
        auto newOp = vector::vectorizeOneOperationAKG(&op, state);
        newOps.insert(newOp);
      }
    }

    replaceInsertResult(genericOpBlock, state.valueVectorReplacement, newOps);
  }
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::setMappedShape(SmallVectorImpl<int64_t> &mappedShape,
                                                             const SmallVectorImpl<int64_t> &shape,
                                                             AffineMap map) const {
  for (unsigned int i = 0; i < map.getNumResults(); i++) {
    if (auto constExpr = map.getResult(i).dyn_cast<AffineConstantExpr>()) {
      assert(constExpr.getValue() == 0 && "unsupported map");
      mappedShape.emplace_back(1);
    } else if (auto dimExpr = map.getResult(i).dyn_cast<AffineDimExpr>()) {
      mappedShape.emplace_back(shape[dimExpr.getPosition()]);
    } else {
      llvm_unreachable("unsupported map\n");
    }
  }
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::setMappedIndices(SmallVectorImpl<Value> &mappedIndices,
                                                               const SmallVectorImpl<Value> &indices, AffineMap map,
                                                               OpBuilder &builder, const Location &loc) const {
  for (size_t i = 0; i < map.getNumResults(); i++) {
    auto subMap = map.getSubMap(i);
    auto newIndex = builder.create<AffineApplyOp>(loc, subMap, indices);
    mappedIndices.push_back(newIndex);
  }
}

template <bool isFrontFusion, typename MatchOp>
Type FuseTemplateOps<isFrontFusion, MatchOp>::getMappedType(Type origType, AffineMap map) const {
  if (!origType.isa<ShapedType>()) {
    return origType;
  }

  auto origShapedType = origType.cast<ShapedType>();
  assert(origShapedType.hasRank());
  auto origShapeSize = origShapedType.getRank();
  auto vecMap = getMinorSubMapWithPadZero(map, origShapeSize);
  if (isIdentityWithBroadcast(vecMap) || isIdentityWithPermutation(vecMap)) {
    // get unbroadcast pr unpermutate shape type
    assert(origType.isa<VectorType>() && "unsupported type!");
    auto vectorResultType = origType.cast<VectorType>();
    llvm::SmallVector<int64_t, 3> unMappedShape(vectorResultType.getShape());
    llvm::SmallVector<int64_t, 3> mappedShape;
    auto newMap = origShapeSize > map.getNumResults() ? map : vecMap;
    setMappedShape(mappedShape, unMappedShape, newMap);
    auto newVectorResultType = vectorResultType.clone(mappedShape);
    return newVectorResultType;
  }

  return origType;
}

template <bool isFrontFusion, typename MatchOp>
llvm::Optional<AffineMap> FuseTemplateOps<isFrontFusion, MatchOp>::getInvGenericOperandIndexMap(
  GenericOp genericOp, OpOperand *operand) const {
  // generic loop -> generic operand tensor index
  AffineMap operandIndexMap;
  if constexpr (isFrontFusion) {
    assert(operand->get().isa<OpResult>());
    operandIndexMap = genericOp.getIndexingMapMatchingResult(operand->get().cast<OpResult>());
  } else {
    operandIndexMap = genericOp.getMatchingIndexingMap(operand);
  }

  if (!operandIndexMap.isPermutation()) {
    // map is not invertible
    return std::nullopt;
  }

  // generic operand tensor index -> generic loop
  AffineMap invOperandIndexMap = inversePermutation(operandIndexMap);

  return invOperandIndexMap;
}

template <bool isFrontFusion, typename MatchOp>
llvm::SmallVector<AffineMap> FuseTemplateOps<isFrontFusion, MatchOp>::getFusedOperandToAllMaps(
  GenericOp genericOp, AffineMap invGenericFusedOperandIndexMap) const {
  llvm::SmallVector<AffineMap> fusedOperand2AllMaps;

  for (const auto operand :
       llvm::concat<OpOperand *>(genericOp.getDpsInputOperands(), genericOp.getDpsInitOperands())) {
    // generic loop -> operand tensor index
    AffineMap producerSrcMap = genericOp.getMatchingIndexingMap(operand);

    // fused operand index-> operand tensor index (fused operand index ->
    // generic loop -> operand tensor index)
    AffineMap fusedOperandToOperandMap = producerSrcMap.compose(invGenericFusedOperandIndexMap);

    fusedOperand2AllMaps.emplace_back(fusedOperandToOperandMap);
  }

  return fusedOperand2AllMaps;
}

template <bool isFrontFusion, typename MatchOp>
AffineMap FuseTemplateOps<isFrontFusion, MatchOp>::getMinorSubMapWithPadZero(AffineMap map,
                                                                             unsigned int numResults) const {
  auto subMap = map.getMinorSubMap(numResults);
  if (numResults <= map.getNumResults()) {
    return subMap;
  }

  auto padSize = numResults - map.getNumResults();
  AffineMap padMap = subMap;
  OpBuilder builder(subMap.getContext());
  for (unsigned int i = 0; i < padSize; i++) {
    padMap = padMap.insertResult(builder.getAffineConstantExpr(0), i);
  }
  return padMap;
}

template <>
TemplateOp FuseTemplateOps<true>::createFusedTemplateOp(GenericOp producer, TemplateOp consumer,
                                                        OpOperand *fusedOperand,
                                                        FusedTemplateOpInfo *fusedTemplateOpInfo,
                                                        RewriterBase &rewriter) const {
  assert(consumer.isDpsInput(fusedOperand) && "expected producer of input operand");

  // 1. Compute the fused operands list and indexing maps.
  SmallVector<Value> fusedOperands;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedIndexMaps;
  fusedOperands.reserve(producer.getNumOperands() + consumer.getNumOperands());
  fusedResultTypes.reserve(producer.getNumDpsInits() + consumer.getNumDpsInits());
  fusedIndexMaps.reserve(producer.getNumOperands() + consumer.getNumOperands());
  // Splice consumer inputs/maps till fusedOperand.
  SmallVector<OpOperand *> consumerInputs = consumer.getDpsInputOperands();
  SmallVector<OpOperand *>::iterator it = llvm::find(consumerInputs, fusedOperand);
  assert(it != consumerInputs.end() && "expected to find the consumer operand");
  for (OpOperand *opOperand : llvm::make_range(consumerInputs.begin(), it)) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getMatchingIndexingMap(opOperand));
  }

  fusedTemplateOpInfo->fusedOperandIndx = fusedOperands.size();
  fusedTemplateOpInfo->insertArgumentsIndx = fusedOperands.size() + 1;

  // Splice producer's input operands/maps.
  auto producerResult = fusedOperand->get().cast<OpResult>();
  AffineMap producerResultIndexMap = producer.getIndexingMapMatchingResult(producerResult);
  for (OpOperand *opOperand : producer.getDpsInputOperands()) {
    fusedOperands.push_back(opOperand->get());
    // Compute indexing maps for the producer args in the fused operation.
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(
      opOperand, producerResultIndexMap, consumer.getMatchingIndexingMap(fusedOperand));
    fusedIndexMaps.push_back(map);
  }
  fusedTemplateOpInfo->genericFusedOperand = producer.getDpsInputOperand(0);

  // Splice remaining consumer's input operands/maps (drop fusedOperand).
  for (OpOperand *opOperand : llvm::make_range(std::next(it), consumerInputs.end())) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getMatchingIndexingMap(opOperand));
  }

  // Splice all of consumer's output operands (skip operands: added by the
  // builder).
  for (OpOperand *opOperand : consumer.getDpsInitOperands()) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(consumer.getMatchingIndexingMap(opOperand));
    fusedResultTypes.push_back(opOperand->get().getType());
  }

  // 2. Create fusedConsumer according to new operands and new index map
  auto attrs = consumer->getAttrDictionary();
  NamedAttrList attributes(attrs);
  attributes.set(consumer.getIndexingMapsAttrName(), rewriter.getAffineMapArrayAttr(fusedIndexMaps));
  attributes.set(consumer.getOperandSegmentSizesAttrName(),
                 rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(fusedOperands.size() - consumer.getNumDpsInits()),
                                                static_cast<int32_t>(consumer.getNumDpsInits())}));

  auto fusedConsumer =
    rewriter.create<TemplateOp>(consumer.getLoc(), fusedResultTypes, fusedOperands, attributes.getAttrs());

  // Construct an AffineMap from consumer loops to producer loops.
  // consumer loop -> tensor index
  AffineMap consumerResultIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
  // tensor index -> producer loop
  AffineMap invProducerResultIndexMap = inversePermutation(producerResultIndexMap);
  assert(invProducerResultIndexMap && "expected producer result indexig map to be invertible");
  // consumer loop -> producer loop
  AffineMap consumerToProducerLoopsMap = invProducerResultIndexMap.compose(consumerResultIndexMap);
  llvm::SmallDenseSet<int> preservedProducerResults;
  // generate TemplateOp region
  generateFusedElementwiseOpRegion<GenericOp, TemplateOp>(rewriter, fusedConsumer, consumerToProducerLoopsMap,
                                                          fusedOperand, preservedProducerResults);

  return fusedConsumer;
}

template <>
TemplateOp FuseTemplateOps<false>::createFusedTemplateOp(TemplateOp producer, GenericOp consumer,
                                                         OpOperand *fusedOperand,
                                                         FusedTemplateOpInfo *fusedTemplateOpInfo,
                                                         RewriterBase &rewriter) const {
  assert(consumer.isDpsInput(fusedOperand) && "expected producer of input operand");

  // 1. Compute the fused operands list and indexing maps.
  SmallVector<Value> fusedOperands;
  SmallVector<Type> fusedResultTypes;
  SmallVector<AffineMap> fusedIndexMaps;
  fusedOperands.reserve(producer->getNumOperands() + consumer->getNumOperands());
  fusedResultTypes.reserve(producer.getNumDpsInits() + consumer.getNumDpsInits());
  fusedIndexMaps.reserve(producer->getNumOperands() + consumer->getNumOperands());
  // Splice producer input operands/maps.
  for (OpOperand *opOperand : producer.getDpsInputOperands()) {
    fusedOperands.push_back(opOperand->get());
    fusedIndexMaps.push_back(producer.getMatchingIndexingMap(opOperand));
  }
  fusedTemplateOpInfo->insertArgumentsIndx = fusedOperands.size();

  // Splice consumer input operands/maps except fusedOperand.
  auto producerResult = fusedOperand->get().cast<OpResult>();
  AffineMap producerResultIndexMap = producer.getIndexingMapMatchingResult(producerResult);
  AffineMap consumerFusedOperandIndexMap = consumer.getMatchingIndexingMap(fusedOperand);
  for (OpOperand *opOperand : consumer.getDpsInputOperands()) {
    if (opOperand == fusedOperand) {
      continue;
    }

    fusedOperands.push_back(opOperand->get());
    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(opOperand, consumerFusedOperandIndexMap,
                                                                           producerResultIndexMap);
    fusedIndexMaps.push_back(map);
  }
  fusedTemplateOpInfo->genericFusedOperand = fusedOperand;
  fusedTemplateOpInfo->fusedOperandIndx = fusedOperands.size();

  // Splice consumer output operands
  for (OpOperand *opOperand : consumer.getDpsInitOperands()) {
    fusedOperands.push_back(opOperand->get());

    AffineMap map = getIndexingMapOfProducerOperandsInCoordinatesOfFusedOp(opOperand, consumerFusedOperandIndexMap,
                                                                           producerResultIndexMap);
    fusedIndexMaps.push_back(map);
    fusedResultTypes.push_back(opOperand->get().getType());
  }

  // 2. Create fusedConsumer according to new operands and new index map
  auto attrs = producer->getAttrDictionary();
  NamedAttrList attributes(attrs);
  attributes.set(producer.getIndexingMapsAttrName(), rewriter.getAffineMapArrayAttr(fusedIndexMaps));
  attributes.set(producer.getOperandSegmentSizesAttrName(),
                 rewriter.getDenseI32ArrayAttr({static_cast<int32_t>(fusedOperands.size() - producer.getNumDpsInits()),
                                                static_cast<int32_t>(producer.getNumDpsInits())}));

  auto fusedProducer =
    rewriter.create<TemplateOp>(consumer.getLoc(), fusedResultTypes, fusedOperands, attributes.getAttrs());

  // generate dummy region
  Block *fusedBlock = new Block();
  fusedProducer.getRegion().push_back(fusedBlock);

  for (auto operand : fusedProducer.getOperands()) {
    fusedBlock->addArgument(getElementTypeOrSelf(operand.getType()), consumer.getLoc());
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(fusedBlock);

  llvm::SmallVector<Value> yieldOperands;
  for (auto resultType : fusedResultTypes) {
    auto result = rewriter.create<arith::ConstantOp>(
      fusedProducer.getLoc(), rewriter.getZeroAttr(getElementTypeOrSelf(resultType)), getElementTypeOrSelf(resultType));
    yieldOperands.push_back(result);
  }

  rewriter.create<linalg::YieldOp>(fusedProducer.getLoc(), yieldOperands);

  return fusedProducer;
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::insertFuncArguments(func::FuncOp funcOp, TemplateOp fusedTemplated,
                                                                  GenericOp genericOp,
                                                                  FusedTemplateOpInfo *fusedTemplateOpInfo) const {
  LLVM_DEBUG(llvm::dbgs() << "func insert place : " << fusedTemplateOpInfo->insertArgumentsIndx << "\n");
  int64_t index = fusedTemplateOpInfo->insertArgumentsIndx;
  for (auto &operand : genericOp.getDpsInputOperands()) {
    if (operand != fusedTemplateOpInfo->genericFusedOperand) {
      funcOp.insertArgument(index, fusedTemplated->getOperand(index).getType(),
                            DictionaryAttr::get(funcOp.getContext()), fusedTemplated->getOperand(index).getLoc());
      index++;
    }
  }
}

template <bool isFrontFusion, typename MatchOp>
void FuseTemplateOps<isFrontFusion, MatchOp>::collectFusedFuncTensors(SmallVectorImpl<Value> &fusedFuncTensors,
                                                                      func::FuncOp funcOp, GenericOp genericOp,
                                                                      FusedTemplateOpInfo *fusedTemplateOpInfo) const {
  fusedFuncTensors.push_back(funcOp.getArgument(fusedTemplateOpInfo->fusedOperandIndx));
  for (int64_t i = 0; i < genericOp.getNumDpsInputs() - 1; i++) {
    fusedFuncTensors.push_back(funcOp.getArgument(i + fusedTemplateOpInfo->insertArgumentsIndx));
  }
  assert(!fusedFuncTensors.empty());
}

template <bool isFrontFusion, typename MatchOp>
WalkResult FuseTemplateOps<isFrontFusion, MatchOp>::recordFuseMemrefs(Operation *op,
                                                                      const SmallVectorImpl<Value> &fusedFuncTensors,
                                                                      SmallVectorImpl<Value> &fuseFuncMemrefs,
                                                                      bool needModifyType) const {
  bufferization::ToMemrefOp toMemrefOp = cast<bufferization::ToMemrefOp>(op);
  if (toMemrefOp.getTensor() != fusedFuncTensors[0]) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "recordFuseMemrefs tensor : " << fusedFuncTensors[0] << "\n");

  if (needModifyType) {
    auto tensorType = toMemrefOp.getTensor().getType().cast<TensorType>();
    auto newMemrefType = MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    toMemrefOp.getResult().setType(newMemrefType);
  }

  // colllect memrefs which map to fused input arguments
  fuseFuncMemrefs.push_back(toMemrefOp.getResult());
  LLVM_DEBUG(llvm::dbgs() << "recordFuseMemrefs memref : " << fuseFuncMemrefs[0] << "\n");

  OpBuilder builder(toMemrefOp);
  builder.setInsertionPointAfter(toMemrefOp);

  llvm::SmallVector<Value> insertedFuncMemrefs;
  for (auto it = fusedFuncTensors.end() - 1; it > fusedFuncTensors.begin(); it--) {
    // generate to ToMemrefOp for new tensor srcs
    insertedFuncMemrefs.push_back(castTensorToMemref(builder, *it));
  }

  fuseFuncMemrefs.append(insertedFuncMemrefs.rbegin(), insertedFuncMemrefs.rend());

  return WalkResult::advance();
}

template <bool isFrontFusion, typename MatchOp>
template <typename OP>
void FuseTemplateOps<isFrontFusion, MatchOp>::updatePadType(OpBuilder &builder, Operation *op) const {
  OP loadOp = cast<OP>(op);
  if (!loadOp.getPadding()) {
    return;
  }
  Type padType = loadOp.getPadding().getType();
  Type loadType = getElementTypeOrSelf(loadOp.getMemref());
  if (padType == loadType) {
    return;
  }
  bool isBitExtend = padType.getIntOrFloatBitWidth() < loadType.getIntOrFloatBitWidth();
  builder.setInsertionPoint(loadOp);
  auto castOp = isBitExtend ? builder.create<arith::ExtFOp>(loadOp->getLoc(), loadType, loadOp.getPadding())
                            : builder.create<arith::TruncFOp>(loadOp->getLoc(), loadType, loadOp.getPadding());
  loadOp.setPadding(castOp->getResult(0));
  return;
}

template <bool isFrontFusion, typename MatchOp>
template <typename OP>
WalkResult FuseTemplateOps<isFrontFusion, MatchOp>::processFusionLoad(Operation *op, GenericOp genericOp,
                                                                      OpOperand *fusedOperand,
                                                                      const SmallVectorImpl<Value> &fuseFuncMemrefs,
                                                                      bool needCastType, MapType &mapType) const {
  OP loadOp = cast<OP>(op);

  if (loadOp.getMemRef() != fuseFuncMemrefs[0]) {
    return WalkResult::advance();
  }

  if (std::is_same_v<OP, fusion::LoadOp>) {
    LLVM_DEBUG(llvm::dbgs() << "processFusionLoad memref : " << fuseFuncMemrefs[0] << "\n");
  } else if (std::is_same_v<OP, fusion::MultiLoadOp>) {
    LLVM_DEBUG(llvm::dbgs() << "processFusionLoad<MultiLoad> memref : " << fuseFuncMemrefs[0] << "\n");
  }

  // 1 get genericFuseOperand2AllOperandMaps
  auto invGenericFusedOperandIndexMap = getInvGenericOperandIndexMap(genericOp, fusedOperand);
  assert(invGenericFusedOperandIndexMap.has_value() &&
         "expected genericOp fused operand indexing map to be invertible");
  llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandMaps =
    getFusedOperandToAllMaps(genericOp, invGenericFusedOperandIndexMap.value());
  assert(genericFuseOperand2AllOperandMaps.size() > 0);

  // 2 modify loadOp
  // change load indices according to map
  AffineMap genericFusedOperandToFinalOperand;
  if constexpr (isFrontFusion) {
    genericFusedOperandToFinalOperand = genericFuseOperand2AllOperandMaps.front();
  } else {
    genericFusedOperandToFinalOperand = genericFuseOperand2AllOperandMaps.back();
  }

  OpBuilder builder(loadOp);
  llvm::SmallVector<Value, 6> origIndices(loadOp.getIndices());
  llvm::SmallVector<Value, 6> newIndices;

  setMappedIndices(newIndices, origIndices, genericFusedOperandToFinalOperand, builder, loadOp->getLoc());
  loadOp.getIndicesMutable().assign(newIndices);

  // change load result type according to map
  auto origResultType = loadOp.getResult().getType();
  auto newResultType = getMappedType(origResultType, genericFusedOperandToFinalOperand);
  if (needCastType) {
    // change the element type of load result
    newResultType = castElementType(newResultType, getElementTypeOrSelf(loadOp.getMemref()));
    origResultType = castElementType(origResultType, getElementTypeOrSelf(loadOp.getMemref()));
  }
  loadOp.getResult().setType(newResultType);

  // change load padding type
  updatePadType<OP>(builder, loadOp);

  // MultiLoadOp based on other load data to changes mapType independently
  if (!std::is_same_v<OP, fusion::LoadOp>) {
    return WalkResult::advance();
  }
  mapType = IDENTITY;

  // change fusion.load to fusion.load + fusion.broadcast if load result
  // should be broadcast
  builder.setInsertionPointAfter(loadOp);
  auto loadShapeSize = getShapeSize(origResultType);
  if (loadShapeSize > 1) {
    llvm::SmallPtrSet<Operation *, 3> newOps;
    auto producerResultToSrc0VecMap = getMinorSubMapWithPadZero(genericFusedOperandToFinalOperand, loadShapeSize);
    if (isIdentityWithBroadcast(producerResultToSrc0VecMap) && !producerResultToSrc0VecMap.isIdentity()) {
      mapType = BROADCAST;
      auto multiLoadOp = builder.create<fusion::MultiLoadOp>(
        loadOp.getLoc(), loadOp.getResult().getType(), loadOp.getMemRef(), loadOp.getIndices(), Value(), ArrayAttr());
      newOps.insert(multiLoadOp);
      auto insertedOp = builder.create<fusion::InsertOp>(multiLoadOp.getLoc(), loadOp.getResult().getType(),
                                                         loadOp.getMemRef(), loadOp.getResult());
      newOps.insert(insertedOp);
      auto broadcastOp =
        builder.create<fusion::BroadcastOp>(insertedOp.getLoc(), origResultType, insertedOp.getResult());
      newOps.insert(broadcastOp);
      loadOp.getResult().replaceAllUsesExcept(broadcastOp.getResult(), newOps);
    } else {
      llvm::SmallVector<unsigned int> permutationDims;
      if (isIdentityWithPermutation(producerResultToSrc0VecMap, permutationDims) &&
          !producerResultToSrc0VecMap.isIdentity()) {
        mapType = PERMUTATION;
        auto multiLoadOp = builder.create<fusion::MultiLoadOp>(
          loadOp.getLoc(), loadOp.getResult().getType(), loadOp.getMemRef(), loadOp.getIndices(), Value(), ArrayAttr());
        newOps.insert(multiLoadOp);
        auto insertedOp = builder.create<fusion::InsertOp>(multiLoadOp.getLoc(), loadOp.getResult().getType(),
                                                           loadOp.getMemRef(), loadOp.getResult());
        newOps.insert(insertedOp);
        auto transp =
          builder.getI64ArrayAttr(llvm::SmallVector<int64_t>(permutationDims.begin(), permutationDims.end()));
        auto permutateOp =
          builder.create<fusion::TransposeOp>(insertedOp.getLoc(), origResultType, insertedOp.getResult(), transp);
        newOps.insert(permutateOp);
        loadOp.getResult().replaceAllUsesExcept(permutateOp.getResult(), newOps);
      }
    }
  }

  return WalkResult::advance();
}

template <bool isFrontFusion, typename MatchOp>
WalkResult FuseTemplateOps<isFrontFusion, MatchOp>::processFusionMultiLoad(
  Operation *op, GenericOp genericOp, OpOperand *fusedOperand, const SmallVectorImpl<Value> &fuseFuncMemrefs,
  SmallVectorImpl<Value> &newFuseFuncLoads, bool needCastType, MapType mapType) const {
  fusion::MultiLoadOp multiLoadOp = cast<fusion::MultiLoadOp>(op);

  if (multiLoadOp.getMemref() != fuseFuncMemrefs[0]) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "processFusionMultiLoad memref : " << fuseFuncMemrefs[0] << "\n");

  llvm::SmallVector<Value, 6> origIndices(multiLoadOp.getIndices());
  auto origResultType = multiLoadOp.getResult().getType();
  auto loadShapeSize = getShapeSize(origResultType);

  // 1 process fusion.multi_load
  processFusionLoad<fusion::MultiLoadOp>(op, genericOp, fusedOperand, fuseFuncMemrefs, needCastType, mapType);

  // 2 get genericFuseOperand2AllOperandMaps
  auto invGenericFusedOperandIndexMap = getInvGenericOperandIndexMap(genericOp, fusedOperand);
  assert(invGenericFusedOperandIndexMap.has_value() &&
         "expected genericOp fused operand indexing map to be invertible");
  llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandMaps =
    getFusedOperandToAllMaps(genericOp, invGenericFusedOperandIndexMap.value());
  assert(genericFuseOperand2AllOperandMaps.size() > 0);

  // 3 generate new fusion.load for new srcs and collect newFuseFuncLoads
  OpBuilder builder(multiLoadOp);
  builder.setInsertionPointAfter(multiLoadOp);
  OpOperand *skipOperand;

  if constexpr (isFrontFusion) {
    skipOperand = genericOp.getDpsInputOperand(0);
  } else {
    skipOperand = fusedOperand;
  }

  int64_t fuseFuncMemrefIndx = genericOp.getNumDpsInputs() - 1;
  for (int64_t i = genericOp.getNumDpsInputs() - 1; i >= 0; i--) {
    if (genericOp.getDpsInputOperand(i) == skipOperand) {
      continue;
    }
    if (fuseFuncMemrefs[fuseFuncMemrefIndx].getType().isa<MemRefType>() ||
        fuseFuncMemrefs[fuseFuncMemrefIndx].getType().isa<UnrankedMemRefType>()) {
      auto producerResultToCurSrcMap = genericFuseOperand2AllOperandMaps[i];
      llvm::SmallVector<Value, 6> curSrcIndices;
      setMappedIndices(curSrcIndices, origIndices, producerResultToCurSrcMap, builder, multiLoadOp->getLoc());
      auto curLoadResultType = getMappedType(origResultType, producerResultToCurSrcMap);
      curLoadResultType = castElementType(curLoadResultType, getElementTypeOrSelf(fuseFuncMemrefs[fuseFuncMemrefIndx]));
      auto insertLoadOp =
        builder.create<fusion::LoadOp>(multiLoadOp.getLoc(), curLoadResultType, fuseFuncMemrefs[fuseFuncMemrefIndx],
                                       curSrcIndices, Value(), ArrayAttr());
      builder.create<fusion::MultiLoadOp>(multiLoadOp.getLoc(), curLoadResultType, fuseFuncMemrefs[fuseFuncMemrefIndx],
                                          curSrcIndices, Value(), ArrayAttr());
      auto insertedOp = builder.create<fusion::InsertOp>(multiLoadOp.getLoc(), curLoadResultType,
                                                         fuseFuncMemrefs[fuseFuncMemrefIndx], insertLoadOp.getResult());
      Value result = insertedOp.getResult();

      if (loadShapeSize > 1) {
        auto curSrcResultType =
          castElementType(origResultType, getElementTypeOrSelf(fuseFuncMemrefs[fuseFuncMemrefIndx]));
        auto producerResultToCurSrcVecMap = getMinorSubMapWithPadZero(producerResultToCurSrcMap, loadShapeSize);
        if (isIdentityWithBroadcast(producerResultToCurSrcVecMap) && !producerResultToCurSrcVecMap.isIdentity()) {
          auto broadcastOp =
            builder.create<fusion::BroadcastOp>(insertedOp.getLoc(), curSrcResultType, insertedOp.getResult());
          result = broadcastOp.getResult();
        } else {
          llvm::SmallVector<unsigned int> permutationDims;
          if (isIdentityWithPermutation(producerResultToCurSrcVecMap, permutationDims) &&
              !producerResultToCurSrcVecMap.isIdentity()) {
            auto transp =
              builder.getI64ArrayAttr(llvm::SmallVector<int64_t>(permutationDims.begin(), permutationDims.end()));
            auto permutateOp = builder.create<fusion::TransposeOp>(insertedOp.getLoc(), curSrcResultType,
                                                                   insertedOp.getResult(), transp);
            result = permutateOp.getResult();
          }
        }
      }
      newFuseFuncLoads.push_back(result);
    } else {
      // not shaped type and no need to generate memref.load
      newFuseFuncLoads.push_back(fuseFuncMemrefs[fuseFuncMemrefIndx]);
    }
    fuseFuncMemrefIndx--;
  }
  if (mapType != IDENTITY) {
    op->erase();
  }
  return WalkResult::advance();
}

template <bool isFrontFusion, typename MatchOp>
LogicalResult FuseTemplateOps<isFrontFusion, MatchOp>::fuseElementwiseToTemplateFunc(
  TemplateOp fusedTemplated, GenericOp genericOp, OpOperand *fusedOperand,
  FusedTemplateOpInfo *fusedTemplateOpInfo) const {
  // Find template func op
  auto fnSym = fusedTemplated->getAttr(TemplateFuncAttrName).cast<SymbolRefAttr>();
  auto funcOp = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupNearestSymbolFrom(fusedTemplated, fnSym));

  // insert func arguments for new srcs
  insertFuncArguments(funcOp, fusedTemplated, genericOp, fusedTemplateOpInfo);

  // replace fuseOperandIndx-th input type of func argument when fusion cast
  // operation
  auto fuseOperandIndx = fusedTemplateOpInfo->fusedOperandIndx;
  LLVM_DEBUG(llvm::dbgs() << "func fuseOperandIndx : " << fuseOperandIndx << "\n");
  bool needCastType = getElementTypeOrSelf(funcOp.getArgument(fuseOperandIndx).getType()) !=
                      getElementTypeOrSelf(fusedTemplated->getOperand(fuseOperandIndx).getType());
  bool needModifyType =
    funcOp.getArgument(fuseOperandIndx).getType() != fusedTemplated->getOperand(fuseOperandIndx).getType();
  if (needModifyType) {
    auto newType = castToDynShape(fusedTemplated->getOperand(fuseOperandIndx).getType());
    replaceFuncParamType(funcOp, fuseOperandIndx, newType);
  }

  // Collect fused input arguments which map to original producer inputs
  SmallVector<Value> fusedFuncTensors;
  collectFusedFuncTensors(fusedFuncTensors, funcOp, genericOp, fusedTemplateOpInfo);

  // Record memrefs which map to fused input arguments
  SmallVector<Value> fuseFuncMemrefs;
  // Record load values for newly generated memref.load
  SmallVector<Value> newFuseFuncLoads;

  MapType mapType = IDENTITY;
  // Search fusion op and insert other op
  auto funcWalkResult = funcOp.walk([&](Operation *op) {
    if (isa<bufferization::ToMemrefOp>(op)) {
      // Find the memref var of fused tensor operand and generate ToMemrefOp for
      // newly-fused tensor srcs
      return recordFuseMemrefs(op, fusedFuncTensors, fuseFuncMemrefs, needModifyType);
    }

    if (isa<fusion::LoadOp>(op)) {
      return processFusionLoad<fusion::LoadOp>(op, genericOp, fusedOperand, fuseFuncMemrefs, needCastType, mapType);
    }

    if (isa<fusion::MultiLoadOp>(op)) {
      return processFusionMultiLoad(op, genericOp, fusedOperand, fuseFuncMemrefs, newFuseFuncLoads, needCastType,
                                    mapType);
    }

    if (isa<fusion::InsertOp>(op)) {
      return processFusionInsert(op, genericOp, fusedTemplateOpInfo->genericFusedOperand, fuseFuncMemrefs,
                                 newFuseFuncLoads, needModifyType, mapType);
    }

    if (isa<fusion::StoreOp>(op)) {
      return processFusionStore(op, genericOp, fusedOperand, fuseFuncMemrefs, needCastType);
    }
    return WalkResult::advance();
  });
  if (funcWalkResult.wasInterrupted()) {
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "fused func : \n" << funcOp << "\n");
  return success();
}

template <bool isFrontFusion, typename MatchOp>
WalkResult FuseTemplateOps<isFrontFusion, MatchOp>::processFusionInsert(Operation *op, GenericOp genericOp,
                                                                        OpOperand *genericFusedOperand,
                                                                        const SmallVectorImpl<Value> &fuseFuncMemrefs,
                                                                        const SmallVectorImpl<Value> &newFuseFuncLoads,
                                                                        bool needCastType, MapType mapType) const {
  fusion::InsertOp insertOp = cast<fusion::InsertOp>(op);

  if (insertOp.getMemref() != fuseFuncMemrefs[0]) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "processFusionInsert memref : " << fuseFuncMemrefs[0] << "\n");
  assert(newFuseFuncLoads.size() == fuseFuncMemrefs.size() - 1);

  // cast data and result if fusion cast operation
  if (needCastType) {
    auto newType = castElementType(insertOp.getData().getType(), getElementTypeOrSelf(insertOp.getMemref()));
    insertOp.getData().setType(newType);
    insertOp.getResult().setType(newType);
  }

  // Insert after fusion insert op
  insertOperations(insertOp, genericOp, genericFusedOperand, newFuseFuncLoads, mapType);
  return WalkResult::advance();
}

template <bool isFrontFusion, typename MatchOp>
WalkResult FuseTemplateOps<isFrontFusion, MatchOp>::processFusionStore(Operation *op, GenericOp genericOp,
                                                                       OpOperand *fusedOperand,
                                                                       const SmallVectorImpl<Value> &fuseFuncMemrefs,
                                                                       bool needCastType) const {
  fusion::StoreOp storeOp = cast<fusion::StoreOp>(op);

  if (storeOp.getMemRef() != fuseFuncMemrefs[0]) {
    return WalkResult::advance();
  }

  LLVM_DEBUG(llvm::dbgs() << "processFusionStore memref : " << fuseFuncMemrefs[0] << "\n");
  // 1 get genericFuseOperand2AllOperandMaps
  auto invGenericFusedOperandIndexMap = getInvGenericOperandIndexMap(genericOp, fusedOperand);
  assert(invGenericFusedOperandIndexMap.has_value() &&
         "expected genericOp fused operand indexing map to be invertible");
  llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandMaps =
    getFusedOperandToAllMaps(genericOp, invGenericFusedOperandIndexMap.value());
  assert(genericFuseOperand2AllOperandMaps.size() > 0);

  // 2 modify storeOp
  // change store indices according to map
  AffineMap genericFusedOperandToFinalOperand;
  if constexpr (isFrontFusion) {
    genericFusedOperandToFinalOperand = genericFuseOperand2AllOperandMaps.front();
  } else {
    genericFusedOperandToFinalOperand = genericFuseOperand2AllOperandMaps.back();
  }

  OpBuilder builder(storeOp);
  llvm::SmallVector<Value, 6> origIndices(storeOp.getIndices());
  llvm::SmallVector<Value, 6> newIndices;
  setMappedIndices(newIndices, origIndices, genericFusedOperandToFinalOperand, builder, storeOp->getLoc());
  storeOp.getIndicesMutable().assign(newIndices);

  // change store data type according to map
  auto origDataType = storeOp.getValue().getType();
  auto newDataType = getMappedType(origDataType, genericFusedOperandToFinalOperand);
  if (needCastType) {
    // change the element type of store data
    newDataType = castElementType(newDataType, getElementTypeOrSelf(storeOp.getMemref()));
    origDataType = castElementType(origDataType, getElementTypeOrSelf(storeOp.getMemref()));
  }
  storeOp.getValue().setType(origDataType);

  // change fusion.store to fusion.permutate + fusion.store if store data
  // should be permutate
  auto storeShapeSize = getShapeSize(origDataType);
  if (storeShapeSize > 1) {
    auto genericFusedOperandToFinalOperandVecMap =
      getMinorSubMapWithPadZero(genericFusedOperandToFinalOperand, storeShapeSize);
    llvm::SmallVector<unsigned int> permutationDims;
    if (isIdentityWithPermutation(genericFusedOperandToFinalOperandVecMap, permutationDims) &&
        !genericFusedOperandToFinalOperandVecMap.isIdentity()) {
      auto transp = builder.getI64ArrayAttr(llvm::SmallVector<int64_t>(permutationDims.begin(), permutationDims.end()));
      auto permutateOp = builder.create<fusion::TransposeOp>(storeOp.getLoc(), newDataType, storeOp.getValue(), transp);
      storeOp.setOperand(0, permutateOp.getResult());
    }
  }

  return WalkResult::advance();
}

template <bool isFrontFusion, typename MatchOp>
LogicalResult FuseTemplateOps<isFrontFusion, MatchOp>::fuseTemplateOps(RewriterBase &rewriter,
                                                                       OpOperand *fusedOperand) const {
  if constexpr (isFrontFusion) {
    LLVM_DEBUG(llvm::dbgs() << "front fusion\n");
  } else {
    LLVM_DEBUG(llvm::dbgs() << "back fusion\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "fuseTemplateOps operand : " << fusedOperand->get() << "\n");
  auto producerResult = fusedOperand->get().cast<OpResult>();
  auto producer = cast<ProducerOp>(producerResult.getOwner());
  auto consumer = cast<ConsumerOp>(fusedOperand->getOwner());

  GenericOp genericOp;
  TemplateOp templateOp;
  if constexpr (isFrontFusion) {
    genericOp = producer;
    templateOp = consumer;
  } else {
    templateOp = producer;
    genericOp = consumer;
  }

  FusedTemplateOpInfo fusedTemplateOpInfo;
  auto fusedTemplated = createFusedTemplateOp(producer, consumer, fusedOperand, &fusedTemplateOpInfo, rewriter);
  LLVM_DEBUG(llvm::dbgs() << "fuseTemplateOps fusedTemplated : " << fusedTemplated << "\n");

  // Fuse producer into template function
  if (succeeded(fuseElementwiseToTemplateFunc(fusedTemplated, genericOp, fusedOperand, &fusedTemplateOpInfo))) {
    rewriter.replaceOp(consumer, fusedTemplated.getResults());
    return success();
  }
  return failure();
}

void mlir::populateTemplateOpsFusionPatterns(RewritePatternSet &patterns) {
  auto *context = patterns.getContext();
  (void)patterns.add<FuseTemplateOps<true>>(context);
  (void)patterns.add<FuseTemplateOps<false>>(context);
}

namespace {
struct LinalgTemplateOpFusionPass : public impl::LinalgTemplateOpOpFusionBase<LinalgTemplateOpFusionPass> {
  LinalgTemplateOpFusionPass() = default;
  LinalgTemplateOpFusionPass(const LinalgTemplateOpFusionPass &) = default;
  LinalgTemplateOpFusionPass &operator=(const LinalgTemplateOpFusionPass &) = delete;

  explicit LinalgTemplateOpFusionPass(const linalgExt::TemplateOpFusionOptions &options) {
    this->optReshapeByExpand = options.optReshapeByExpand;
    this->optReshapeByCollapse = options.optReshapeByCollapse;
  }

  void runOnOperation() override;
};
}  // namespace

void LinalgTemplateOpFusionPass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);

  ControlFusionFn defaultControlFn = [](OpOperand *fusedOperand) {
    Operation *producer = fusedOperand->get().getDefiningOp();
    return producer && producer->hasOneUse();
  };

  // Add template op fusion patterns.
  populateTemplateOpsFusionPatterns(patterns);
  if (optReshapeByExpand) {
    populateFoldReshapeOpsByExpansionPatterns(patterns, defaultControlFn);
  }
  if (optReshapeByCollapse) {
    populateFoldReshapeOpsByCollapsingPatterns(patterns, defaultControlFn);
  }

  // General canonicalization patterns.
  AffineApplyOp::getCanonicalizationPatterns(patterns, context);
  GenericOp::getCanonicalizationPatterns(patterns, context);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, context);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, context);
  context->getLoadedDialect<LinalgDialect>()->getCanonicalizationPatterns(patterns);
  // Add constant folding patterns.
  populateConstantFoldLinalgOperations(patterns, defaultControlFn);

  // Use TopDownTraversal for compile time reasons
  GreedyRewriteConfig grc;
  grc.useTopDownTraversal = true;
  (void)applyPatternsAndFoldGreedily(op->getRegions(), std::move(patterns), grc);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLinalgTemplateOpFusionPass(
  const linalgExt::TemplateOpFusionOptions &options) {
  return std::make_unique<LinalgTemplateOpFusionPass>(options);
}
