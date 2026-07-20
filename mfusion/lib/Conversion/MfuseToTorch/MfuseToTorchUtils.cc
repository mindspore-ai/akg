/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#include "MfuseToTorchUtils.h"

#include <algorithm>

#include "mfusion/Dialect/Mfuse/Transforms/Outlining/FusionAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/IR/TorchTypes.h"

namespace mlir {
namespace mfuse {

namespace TorchD = mlir::torch::Torch;

bool isDvmKernelGenerator(llvm::StringRef kernelGenerator) { return kernelGenerator == "dvm"; }

bool isInsideDvmCopiedSubgraph(Operation *op) {
  auto func = op->getParentOfType<func::FuncOp>();
  if (!func) {
    return false;
  }
  if (func->hasAttr(mfusion_attrs::kOutlined)) {
    return false;
  }
  auto fusionType = func->getAttrOfType<StringAttr>(mfusion_attrs::kFusionType);
  return fusionType && fusionType.getValue() == "dvm";
}

FailureOr<Value> buildSwapLastTwoDimsPermute(Location loc, Value v, ConversionPatternRewriter &rewriter) {
  auto vtt = dyn_cast<TorchD::ValueTensorType>(v.getType());
  if (!vtt || !vtt.hasSizes()) {
    return failure();
  }
  auto sizes = vtt.getSizes();
  int64_t rank = static_cast<int64_t>(sizes.size());
  if (rank < 2) {
    return failure();
  }

  SmallVector<int64_t> newSizes(sizes.begin(), sizes.end());
  std::swap(newSizes[rank - 2], newSizes[rank - 1]);
  Type permResultType = vtt.getWithSizesAndDtype(newSizes, vtt.getOptionalDtype());

  SmallVector<Value> permDims;
  permDims.reserve(static_cast<size_t>(rank));
  for (int64_t i = 0; i < rank - 2; ++i) {
    permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(i)));
  }
  permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(rank - 1)));
  permDims.push_back(rewriter.create<TorchD::ConstantIntOp>(loc, rewriter.getI64IntegerAttr(rank - 2)));

  MLIRContext *ctx = rewriter.getContext();
  auto listType = TorchD::ListType::get(ctx, TorchD::IntType::get(ctx));
  Value permList = rewriter.create<TorchD::PrimListConstructOp>(loc, listType, permDims);
  return rewriter.create<TorchD::AtenPermuteOp>(loc, permResultType, v, permList).getResult();
}

bool isLegalAddmmInputShape(Type mmOutType, Type inputType) {
  auto mmTy = dyn_cast<RankedTensorType>(mmOutType);
  auto inTy = dyn_cast<RankedTensorType>(inputType);
  if (!mmTy || !inTy || !mmTy.hasStaticShape() || !inTy.hasStaticShape()) {
    return false;
  }
  if (mmTy.getRank() != 2) {
    return false;
  }
  if (mmTy.getElementType() != inTy.getElementType()) {
    return false;
  }
  ArrayRef<int64_t> mmShape = mmTy.getShape();
  ArrayRef<int64_t> inShape = inTy.getShape();
  const int64_t n = mmShape[1];
  if (inTy.getRank() == 2 && inShape[0] == mmShape[0] && inShape[1] == mmShape[1]) {
    return true;
  }
  if (inTy.getRank() == 1 && inShape[0] == n) {
    return true;
  }
  if (inTy.getRank() == 2 && inShape[0] == 1 && inShape[1] == n) {
    return true;
  }
  return false;
}

FailureOr<RankedTensorType> infer2DMatmulOutType(Type selfTy, Type otherTy, bool trans1, bool trans2) {
  auto self = dyn_cast<RankedTensorType>(selfTy);
  auto other = dyn_cast<RankedTensorType>(otherTy);
  if (!self || !other || self.getRank() != 2 || other.getRank() != 2) {
    return failure();
  }
  if (!self.hasStaticShape() || !other.hasStaticShape()) {
    return failure();
  }
  ArrayRef<int64_t> s = self.getShape();
  ArrayRef<int64_t> o = other.getShape();
  const int64_t m = trans1 ? s[1] : s[0];
  const int64_t k1 = trans1 ? s[0] : s[1];
  const int64_t k2 = trans2 ? o[1] : o[0];
  const int64_t n = trans2 ? o[0] : o[1];
  if (k1 != k2) {
    return failure();
  }
  if (self.getElementType() != other.getElementType()) {
    return failure();
  }
  return RankedTensorType::get({m, n}, self.getElementType());
}

FailureOr<Value> createAtenAddmm(Location loc, Type resultType, Value mat1, Value mat2, Value input, bool trans1,
                                 bool trans2, ConversionPatternRewriter &rewriter) {
  // Cross-op getRemappedValue may still yield RankedTensorType when the producer
  // has not been converted yet; refuse rather than emit illegal aten.addmm.
  if (!isa<TorchD::ValueTensorType>(mat1.getType()) || !isa<TorchD::ValueTensorType>(mat2.getType()) ||
      !isa<TorchD::ValueTensorType>(input.getType())) {
    return failure();
  }
  if (trans1) {
    auto permOr = buildSwapLastTwoDimsPermute(loc, mat1, rewriter);
    if (failed(permOr)) {
      return failure();
    }
    mat1 = *permOr;
  }
  if (trans2) {
    auto permOr = buildSwapLastTwoDimsPermute(loc, mat2, rewriter);
    if (failed(permOr)) {
      return failure();
    }
    mat2 = *permOr;
  }
  Value betaVal =
      rewriter.create<TorchD::ConstantFloatOp>(loc, FloatAttr::get(rewriter.getF64Type(), 1.0)).getResult();
  Value alphaVal =
      rewriter.create<TorchD::ConstantFloatOp>(loc, FloatAttr::get(rewriter.getF64Type(), 1.0)).getResult();
  return rewriter.create<TorchD::AtenAddmmOp>(loc, resultType, input, mat1, mat2, betaVal, alphaVal).getResult();
}

static bool matchAclnnMmAsAddProducer(Value candidate, Value other, AclnnMmAddFoldMatch &out) {
  auto mm = candidate.getDefiningOp<AclnnMmOp>();
  if (!mm || !candidate.hasOneUse()) {
    return false;
  }
  out.mmOp = mm;
  out.input = other;
  return true;
}

bool matchAclnnMmAddFoldFromAdd(AddOp addOp, AclnnMmAddFoldMatch &out) {
  Value x = addOp.getX();
  Value y = addOp.getY();
  if (matchAclnnMmAsAddProducer(x, y, out) || matchAclnnMmAsAddProducer(y, x, out)) {
    out.addOp = addOp;
    return true;
  }
  return false;
}

bool matchAclnnMmAddFoldFromMm(AclnnMmOp mmOp, AclnnMmAddFoldMatch &out) {
  Value mmOut = mmOp.getOut();
  if (!mmOut.hasOneUse()) {
    return false;
  }
  auto addOp = dyn_cast<AddOp>(*mmOut.getUsers().begin());
  if (!addOp) {
    return false;
  }
  out.mmOp = mmOp;
  out.addOp = addOp;
  out.input = (addOp.getX() == mmOut) ? addOp.getY() : addOp.getX();
  return true;
}

bool isAclnnMmAddFoldLegal(AclnnMmAddFoldMatch &match) {
  if (!match.mmOp || !match.addOp) {
    return false;
  }
  return isLegalAddmmInputShape(match.mmOp.getOut().getType(), match.input.getType());
}

LogicalResult rewriteAclnnMmAddToAtenAddmm(AclnnMmAddFoldMatch &match, const TypeConverter &converter, Value mat1,
                                           Value mat2, Value remappedInput, ConversionPatternRewriter &rewriter) {
  if (!match.mmOp || !match.addOp) {
    return failure();
  }
  auto resultType = converter.convertType(match.addOp.getResult().getType());
  if (!resultType) {
    return failure();
  }
  auto addmmOr = createAtenAddmm(match.mmOp.getLoc(), resultType, mat1, mat2, remappedInput, match.mmOp.getTransX1(),
                                 match.mmOp.getTransX2(), rewriter);
  if (failed(addmmOr)) {
    return failure();
  }
  rewriter.replaceOp(match.addOp, *addmmOr);
  if (match.mmOp->use_empty()) {
    rewriter.eraseOp(match.mmOp);
  }
  return success();
}

}  // namespace mfuse
}  // namespace mlir
