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

#include "mfusion/Dialect/Mfuse/Support/VarianceUtils.h"

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>

#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace mfuse {
namespace {

bool isPositiveDimDivisor(Value v) {
  auto constOp = v.getDefiningOp<ConstantOp>();
  if (!constOp) {
    return false;
  }
  auto dense = dyn_cast<DenseElementsAttr>(constOp.getValue());
  if (!dense || !dense.isSplat()) {
    return false;
  }
  auto tensorType = dyn_cast<RankedTensorType>(v.getType());
  if (!tensorType || !isScalarOrSingleElement(tensorType)) {
    return false;
  }
  Type elemType = dense.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elemType)) {
    (void)floatType;
    return dense.getSplatValue<APFloat>().convertToDouble() > 0.0;
  }
  if (auto intType = dyn_cast<IntegerType>(elemType)) {
    (void)intType;
    return dense.getSplatValue<APInt>().getSExtValue() > 0;
  }
  return false;
}

bool hasFullyStaticShape(RankedTensorType type) {
  if (!type || !type.hasRank()) {
    return false;
  }
  return llvm::none_of(type.getShape(), ShapedType::isDynamic);
}

}  // namespace

Value peelBroadcast(Value v) {
  while (auto bcast = v.getDefiningOp<BroadcastToOp>()) {
    v = bcast.getInput();
  }
  return v;
}

FailureOr<int64_t> getStaticReductionSize(ArrayRef<int64_t> dims, RankedTensorType inputType) {
  int64_t reductionSize = 1;
  for (int64_t dim : dims) {
    int64_t dimSize = inputType.getDimSize(dim);
    if (dimSize == ShapedType::kDynamic || dimSize <= 0) {
      return failure();
    }
    if (reductionSize > std::numeric_limits<int64_t>::max() / dimSize) {
      return failure();
    }
    reductionSize *= dimSize;
  }
  return reductionSize;
}

llvm::SmallVector<int64_t, 4> getSortedReductionDims(ArrayRef<Attribute> dimAttrs) {
  llvm::SmallVector<int64_t, 4> dims;
  dims.reserve(dimAttrs.size());
  for (Attribute dimAttr : dimAttrs) {
    if (auto intAttr = dyn_cast<IntegerAttr>(dimAttr)) {
      dims.push_back(intAttr.getValue().getSExtValue());
      continue;
    }
    if (auto arrayAttr = dyn_cast<ArrayAttr>(dimAttr)) {
      auto nested = getSortedReductionDims(arrayAttr.getValue());
      dims.append(nested.begin(), nested.end());
      continue;
    }
    return {};
  }
  std::sort(dims.begin(), dims.end());
  return dims;
}

llvm::SmallVector<int64_t, 4> getSortedReductionDims(ArrayAttr dimAttr) {
  return getSortedReductionDims(dimAttr.getValue());
}

llvm::SmallVector<int64_t, 4> getSortedReductionDims(AclnnVarOp op) {
  return getSortedReductionDims(op.getDim());
}

llvm::SmallVector<int64_t, 4> getSortedReductionDims(AclnnVarMeanOp op) {
  return getSortedReductionDims(op.getDim());
}

bool reductionDimsEqual(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return false;
    }
  }
  return true;
}

ReduceMeanOp findMatchingReduceMean(Value x, ArrayRef<int64_t> dims, bool keepdim) {
  Value keyX = getCanonicalFusionTensor(x);
  auto matchesReduceMean = [&](Value root) -> ReduceMeanOp {
    for (Operation *user : root.getUsers()) {
      auto reduceMean = dyn_cast<ReduceMeanOp>(user);
      if (!reduceMean || reduceMean.getKeepdim() != keepdim ||
          getCanonicalFusionTensor(reduceMean.getInput()) != keyX) {
        continue;
      }
      auto reduceDims = reduceMean.getDimensions();
      if (!reduceDims) {
        continue;
      }
      llvm::SmallVector<int64_t, 4> reduceDimValues;
      reduceDimValues.reserve(reduceDims.size());
      std::transform(reduceDims.begin(), reduceDims.end(), std::back_inserter(reduceDimValues),
                     [](Attribute dimAttr) { return cast<IntegerAttr>(dimAttr).getValue().getSExtValue(); });
      std::sort(reduceDimValues.begin(), reduceDimValues.end());
      if (reductionDimsEqual(reduceDimValues, dims)) {
        return reduceMean;
      }
    }
    return nullptr;
  };
  if (auto reduceMean = matchesReduceMean(x)) {
    return reduceMean;
  }
  if (keyX != x) {
    return matchesReduceMean(keyX);
  }
  return nullptr;
}

Value createVarianceFromExistingMean(PatternRewriter &rewriter, Location loc, Value x, Value mean,
                                     ArrayRef<int64_t> dims, int64_t correction, RankedTensorType varType,
                                     bool keepdim, Operation *dominanceAnchor) {
  auto inType = cast<RankedTensorType>(x.getType());
  auto dimsAttr = rewriter.getI64ArrayAttr(dims);
  auto keepdimAttr = rewriter.getBoolAttr(keepdim);

  auto meanBcType = RankedTensorType::get(inType.getShape(), inType.getElementType());
  Value meanBc = mean;
  if (mean.getType() != meanBcType) {
    meanBc = rewriter.create<BroadcastToOp>(loc, meanBcType, mean).getResult();
  }

  Value centered;
  Value keyX = getCanonicalFusionTensor(x);
  auto tryReuseCenterSub = [&](Value root) -> bool {
    for (Operation *user : root.getUsers()) {
      auto subOp = dyn_cast<SubOp>(user);
      if (!subOp || getCanonicalFusionTensor(subOp.getX()) != keyX) {
        continue;
      }
      if (dominanceAnchor && !subOp->isBeforeInBlock(dominanceAnchor)) {
        continue;
      }
      Value subMean = peelBroadcast(subOp.getY());
      Value peeledMean = peelBroadcast(meanBc);
      if (subMean == peeledMean || subMean == meanBc || subOp.getY() == meanBc) {
        centered = subOp.getResult();
        return true;
      }
    }
    return false;
  };
  if (!tryReuseCenterSub(x) && keyX != x) {
    (void)tryReuseCenterSub(keyX);
  }
  if (!centered) {
    centered = rewriter.create<SubOp>(loc, inType, x, meanBc).getResult();
  } else if (Operation *centerDef = centered.getDefiningOp()) {
    // Reused center_sub may be defined earlier in the block; emit the var chain after it.
    rewriter.setInsertionPointAfter(centerDef);
  }

  Value squared = rewriter.create<MulOp>(loc, inType, centered, centered).getResult();
  Value sumSq = rewriter.create<ReduceSumOp>(loc, varType, squared, dimsAttr, keepdimAttr).getResult();

  auto reductionSize = getStaticReductionSize(dims, inType);
  int64_t divisor = reductionSize.value_or(0) - correction;
  auto elemType = inType.getElementType();
  auto scalarType = RankedTensorType::get({}, elemType);
  auto scalarAttr = DenseElementsAttr::get(scalarType, rewriter.getFloatAttr(elemType, static_cast<double>(divisor)));
  Value divisorVal = rewriter.create<ConstantOp>(loc, scalarType, scalarAttr).getResult();
  return rewriter.create<DivOp>(loc, varType, sumSq, divisorVal).getResult();
}

FailureOr<Value> decomposeAclnnVar(AclnnVarOp op, PatternRewriter &rewriter) {
  Value x = op.getSelf();
  auto inType = dyn_cast<RankedTensorType>(x.getType());
  if (!inType || !inType.hasRank()) {
    return failure();
  }

  // Dynamic-shape graphs need runtime DVM CodeGen; keep aclnn.var for aclgraph capture.
  if (!hasFullyStaticShape(inType)) {
    return failure();
  }

  auto dims = getSortedReductionDims(op);
  if (dims.empty()) {
    return failure();
  }

  auto reductionSizeOr = getStaticReductionSize(dims, inType);
  if (failed(reductionSizeOr)) {
    return failure();
  }

  int64_t correction = op.getCorrection();
  int64_t divisor = *reductionSizeOr - correction;
  if (divisor <= 0) {
    return failure();
  }

  auto varType = dyn_cast<RankedTensorType>(op.getVarianceOut().getType());
  if (!varType) {
    return failure();
  }

  bool keepdim = op.getKeepdim();
  // Match legacy fold-layernorm-var-correction: only expand when a sibling reduce_mean exists.
  auto existingMean = findMatchingReduceMean(x, dims, keepdim);
  if (!existingMean) {
    return failure();
  }

  return createVarianceFromExistingMean(rewriter, op.getLoc(), x, existingMean.getResult(), dims, correction,
                                       varType, keepdim, op.getOperation());
}

FailureOr<std::pair<Value, Value>> decomposeAclnnVarMean(AclnnVarMeanOp op, PatternRewriter &rewriter) {
  Value x = op.getSelf();
  auto inType = dyn_cast<RankedTensorType>(x.getType());
  if (!inType || !inType.hasRank()) {
    return failure();
  }

  if (!hasFullyStaticShape(inType)) {
    return failure();
  }

  auto dims = getSortedReductionDims(op);
  if (dims.empty()) {
    return failure();
  }

  auto reductionSizeOr = getStaticReductionSize(dims, inType);
  if (failed(reductionSizeOr)) {
    return failure();
  }

  int64_t correction = op.getCorrection();
  if (*reductionSizeOr - correction <= 0) {
    return failure();
  }

  auto varType = dyn_cast<RankedTensorType>(op.getVarianceOut().getType());
  auto meanType = dyn_cast<RankedTensorType>(op.getMeanOut().getType());
  if (!varType || !meanType) {
    return failure();
  }

  bool keepdim = op.getKeepdim();
  auto dimsAttr = rewriter.getI64ArrayAttr(dims);
  auto keepdimAttr = rewriter.getBoolAttr(keepdim);
  auto meanOp = rewriter.create<ReduceMeanOp>(op.getLoc(), meanType, x, dimsAttr, keepdimAttr);
  Value var = createVarianceFromExistingMean(rewriter, op.getLoc(), x, meanOp.getResult(), dims, correction,
                                             varType, keepdim, op.getOperation());
  return std::pair<Value, Value>{var, Value(meanOp.getResult())};
}

bool matchDecomposedVarianceChain(Value sqrtInput, Value x, DecomposedVarianceChain &chain) {
  Value varVal = getCanonicalFusionTensor(peelBroadcast(sqrtInput));
  auto varDivOp = varVal.getDefiningOp<DivOp>();
  if (!varDivOp || !isPositiveDimDivisor(varDivOp.getOther())) {
    return false;
  }
  Value sumVal = peelBroadcast(varDivOp.getSelf());
  auto varReduce = sumVal.getDefiningOp<ReduceSumOp>();
  if (!varReduce || !varReduce.getKeepdim()) {
    return false;
  }
  auto dims = varReduce.getDimensions();
  if (!dims || dims.empty()) {
    return false;
  }

  Value sqVal = varReduce.getInput();
  auto sqMul = sqVal.getDefiningOp<MulOp>();
  if (!sqMul) {
    return false;
  }
  Value lhs = sqMul.getLhs();
  Value rhs = sqMul.getRhs();
  if (lhs != rhs) {
    return false;
  }
  Value centered = lhs.getDefiningOp<SubOp>() ? lhs : (rhs.getDefiningOp<SubOp>() ? rhs : Value{});
  auto centerSub = centered.getDefiningOp<SubOp>();
  if (!centerSub || getCanonicalFusionTensor(centerSub.getX()) != getCanonicalFusionTensor(x)) {
    return false;
  }

  chain.varDiv = varDivOp;
  chain.varReduceSum = varReduce;
  chain.varSquareMul = sqMul;
  chain.centerSub = centerSub;
  return true;
}

}  // namespace mfuse
}  // namespace mlir
