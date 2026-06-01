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

#ifndef MFUSION_DIALECT_MFUSE_SUPPORT_VARIANCE_UTILS_H
#define MFUSION_DIALECT_MFUSE_SUPPORT_VARIANCE_UTILS_H

#include <utility>

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace mfuse {

/// Peel broadcast_to chains to the underlying tensor value.
Value peelBroadcast(Value v);

FailureOr<int64_t> getStaticReductionSize(ArrayRef<int64_t> dims, RankedTensorType inputType);

llvm::SmallVector<int64_t, 4> getSortedReductionDims(AclnnVarOp op);

bool reductionDimsEqual(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs);

ReduceMeanOp findMatchingReduceMean(Value x, ArrayRef<int64_t> dims, bool keepdim);

/// Build var = sum((x-mean)^2) / (N - correction) with keepdim, reusing an existing mean.
Value createVarianceFromExistingMean(PatternRewriter &rewriter, Location loc, Value x, Value mean,
                                     ArrayRef<int64_t> dims, int64_t correction, RankedTensorType varType,
                                     bool keepdim);

llvm::SmallVector<int64_t, 4> getSortedReductionDims(ArrayAttr dimAttr);

llvm::SmallVector<int64_t, 4> getSortedReductionDims(ArrayRef<Attribute> dimAttrs);

llvm::SmallVector<int64_t, 4> getSortedReductionDims(AclnnVarMeanOp op);

/// Decompose mfuse.aclnn.var into meta ops when input is static and a sibling reduce_mean exists.
FailureOr<Value> decomposeAclnnVar(AclnnVarOp op, PatternRewriter &rewriter);

/// Decompose mfuse.aclnn.var_mean into reduce_mean + variance meta ops when input is static.
FailureOr<std::pair<Value, Value>> decomposeAclnnVarMean(AclnnVarMeanOp op, PatternRewriter &rewriter);

struct DecomposedVarianceChain {
  DivOp varDiv;
  ReduceSumOp varReduceSum;
  MulOp varSquareMul;
  SubOp centerSub;
};

/// Match sqrt(var) prologue where var = sum((x-mean)^2) / positive_scalar.
bool matchDecomposedVarianceChain(Value sqrtInput, Value x, DecomposedVarianceChain &chain);

}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_DIALECT_MFUSE_SUPPORT_VARIANCE_UTILS_H
