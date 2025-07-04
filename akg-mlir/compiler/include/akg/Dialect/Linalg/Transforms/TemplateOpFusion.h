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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_LINALG_TRANSFORMS_TEMPLATEOPFUSION_H_
#define COMPILER_INCLUDE_AKG_DIALECT_LINALG_TRANSFORMS_TEMPLATEOPFUSION_H_

#include <optional>
#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"
#include "akg/Dialect/Linalg/Passes.h"
#include "akg/Dialect/Linalg/Utils/FusionUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// using namespace mlir;
// using namespace mlir::linalg;

#if 0

namespace mlir {
namespace linalg {
struct FusedTemplateOpInfo {
  // index of fused operand in fused template operands
  int64_t fusedOperandIndx;
  // the index to insert generic operands
  int64_t insertArgumentsIndx;
  // the fused operand of generic
  // for back fusion, it is fused operand
  // for front fusion, it is the first input of generic operand
  OpOperand *genericFusedOperand;
};

/// Patterns to fuse fusable generic op to  template op.
template <bool isFrontFusion = true,
          typename MatchOp = typename std::conditional<isFrontFusion, TemplateOp, GenericOp>::type>
class FuseTemplateOps : public OpRewritePattern<MatchOp> {
 public:
  explicit FuseTemplateOps(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<MatchOp>(context, benefit) {}

  LogicalResult matchAndRewrite(MatchOp op, PatternRewriter &rewriter) const override {
    // Find the first operand that is defined by another generic op on tensors.
    for (OpOperand &opOperand : op->getOpOperands()) {
      if (!areTemplateOpsFusable(&opOperand)) {
        continue;
      }
      return fuseTemplateOps(rewriter, &opOperand);
    }
    return failure();
  }

 private:
  using ProducerOp = typename std::conditional<isFrontFusion, GenericOp, TemplateOp>::type;
  using ConsumerOp = typename std::conditional<isFrontFusion, TemplateOp, GenericOp>::type;
  enum MapType { IDENTITY = 0, BROADCAST, PERMUTATION };
  bool isIdentityWithBroadcast(AffineMap map) const;
  bool isIdentityWithPermutation(AffineMap map) const;
  bool isIdentityWithPermutation(AffineMap map, llvm::SmallVector<unsigned int> &permutationDims) const;
  bool isMapFusable(llvm::SmallVectorImpl<AffineMap> &map, Attribute fusionKindAttr) const;
  unsigned int getShapeSize(Type type) const;
  bool isTemplateFuncFusable(func::FuncOp funcOp, Value funcOperand, unsigned int &loadShapeSize) const;
  unsigned getOperandIndexInTemplateOp(TemplateOp templateOp, OpOperand *operand) const;
  bool canTemplateOpFuseOperand(TemplateOp templateOp, GenericOp genericOp, OpOperand *fusedOperand) const;
  bool canGenericOpFuseable(TemplateOp &templateOp, GenericOp &genericOp, OpOperand *fusedOperand) const;
  bool isReductionAndRemoveSize(TemplateOp templateOp, GenericOp genericOp, OpOperand *fusedOperand) const;
  bool isFusionableOp(LinalgOp op) const;
  bool isWriteOnlyOutputs(LinalgOp op) const;
  bool areTemplateOpsFusable(OpOperand *fusedOperand) const;
  Value castTensorToMemref(OpBuilder builder, Value value) const;
  Type castElementType(Type type, Type elementType) const;
  Type castToDynShape(Type v) const;
  void replaceFuncParamType(func::FuncOp func, int64_t index, Type newType) const;
  void insertOperations(fusion::InsertOp insertOp, GenericOp genericOp, OpOperand *fusedOperand,
                        const SmallVectorImpl<Value> &newFuseFuncLoads, MapType mapType) const;
  void setMappedShape(SmallVectorImpl<int64_t> &mappedShape, const SmallVectorImpl<int64_t> &shape,
                      AffineMap map) const;
  void setMappedIndices(SmallVectorImpl<Value> &mappedIndices, const SmallVectorImpl<Value> &indices, AffineMap map,
                        OpBuilder &builder, const Location &loc) const;
  Type getMappedType(Type origType, AffineMap map) const;
  std::optional<AffineMap> getInvGenericOperandIndexMap(GenericOp genericOp, OpOperand *operand) const;
  llvm::SmallVector<AffineMap> getFusedOperandToAllMaps(GenericOp producer,
                                                        AffineMap invGenericFusedOperandIndexMap) const;
  AffineMap getMinorSubMapWithPadZero(AffineMap map, unsigned numResults) const;
  TemplateOp createFusedTemplateOp(ProducerOp producer, ConsumerOp consumer, OpOperand *fusedOperand,
                                   FusedTemplateOpInfo *fusedTemplateOpInfo, RewriterBase &rewriter) const;
  void insertFuncArguments(func::FuncOp funcOp, TemplateOp fusedTemplated, GenericOp genericOp,
                           FusedTemplateOpInfo *fusedTemplateOpInfo) const;
  void collectFusedFuncTensors(SmallVectorImpl<Value> &fusedFuncTensors, func::FuncOp funcOp, GenericOp genericOp,
                               FusedTemplateOpInfo *fusedTemplateOpInfo) const;
  bool checkConstantOpOutOfBlock(Operation &op, SmallPtrSet<Operation *, 4> *constantOps, bool needVectorize) const;
  WalkResult recordFuseMemrefs(Operation *op, const SmallVectorImpl<Value> &fusedFuncTensors,
                               SmallVectorImpl<Value> &fuseFuncMemrefs, bool needModifyType) const;
  template <typename OP>
  void updatePadType(OpBuilder &builder, Operation *op) const;
  template <typename OP>
  WalkResult processFusionLoad(Operation *op, GenericOp producer, OpOperand *fusedOperand,
                               const SmallVectorImpl<Value> &fuseFuncMemrefs, bool needCastType,
                               MapType &mapType) const;
  WalkResult processFusionMultiLoad(Operation *op, GenericOp producer, OpOperand *fusedOperand,
                                    const SmallVectorImpl<Value> &fuseFuncMemrefs,
                                    SmallVectorImpl<Value> &newFuseFuncLoads, bool needCastType, MapType mapType) const;
  WalkResult processFusionStore(Operation *op, GenericOp genericOp, OpOperand *fusedOperand,
                                const SmallVectorImpl<Value> &fuseFuncMemrefs, bool needCastType) const;
  WalkResult processFusionInsert(Operation *op, GenericOp genericOp, OpOperand *fusedOperand,
                                 const SmallVectorImpl<Value> &fuseFuncMemrefs,
                                 const SmallVectorImpl<Value> &newFuseFuncLoads, bool needModifyType,
                                 MapType mapType) const;
  LogicalResult fuseElementwiseToTemplateFunc(TemplateOp templateOp, GenericOp genericOp, OpOperand *fusedOperand,
                                              FusedTemplateOpInfo *fusedTemplateOpInfo) const;
  LogicalResult fuseTemplateOps(RewriterBase &rewriter, OpOperand *fusedOperand) const;
  llvm::SmallVector<AffineMap> getGenericFuseOperand2AllOperandMaps(GenericOp genericOp, OpOperand *fusedOperand) const;
  AffineMap changeLoadOpOrStoreOpIndices(llvm::SmallVector<AffineMap> genericFuseOperand2AllOperandMaps) const;
};
}  // namespace linalg
}  // namespace mlir

#endif

#endif  // COMPILER_INCLUDE_AKG_DIALECT_LINALG_TRANSFORMS_TEMPLATEOPFUSION_H_
