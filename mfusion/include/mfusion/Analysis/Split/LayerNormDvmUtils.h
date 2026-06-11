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

#ifndef MFUSION_ANALYSIS_SPLIT_LAYERNORMDVMUTILS_H
#define MFUSION_ANALYSIS_SPLIT_LAYERNORMDVMUTILS_H

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace mlir {
class RewritePatternSet;

namespace mfuse {
namespace layernorm_dvm {

bool hasLayerNormDvmAttr(Operation *op);
bool hasLayerNormDvmAffinityAttr(Operation *op);
void tagLayerNormDvmOp(Operation *op, llvm::StringRef groupId = {});
void tagLayerNormDvmAffinityOp(Operation *op, llvm::StringRef groupId = {});
void tagLayerNormDvmForwardOps(ArrayRef<Operation *> ops);
unsigned tagLayerNormDvmBackwardOpsExclusive(ArrayRef<Operation *> ops, llvm::StringRef groupId = {});

void registerLayerNormDvmRegionPatterns(RewritePatternSet &patterns);

struct LayerNormDvmMatch {
  bool matched = false;
  Value x;
  Value gamma;
  Value beta;
  AddOp betaAdd;
  DivOp normDiv;
  MulOp gammaMul;
  SubOp centerSub;
  ReduceSumOp reduceSum;
  ReduceMeanOp reduceMean;
  DivOp meanDiv;
  SqrtOp sqrtOp;
  AddOp rstdAdd;
  MulOp varSquareMul;
  ReduceSumOp varReduceSum;
  DivOp varDiv;
  llvm::SmallVector<Operation *, 16> ops;
};

LayerNormDvmMatch matchLayerNormDvmFromBetaAdd(AddOp addOp);

struct LayerNormDvmBwdMatch {
  bool matched = false;
  Value x;
  Value rstd;
  DivOp gradDiv;
  SubOp centerSub;
  ReduceSumOp reduceSum;
  ReduceMeanOp reduceMean;
  DivOp meanDiv;
  llvm::SmallVector<Operation *, 32> ops;
};

LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromGradDiv(DivOp gradDiv);
LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromCqxnSum(ReduceSumOp sumOp);
LayerNormDvmBwdMatch matchLayerNormDvmBackwardFromCuahirSum(ReduceSumOp sumOp);

}  // namespace layernorm_dvm
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_LAYERNORMDVMUTILS_H
