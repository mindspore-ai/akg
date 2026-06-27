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

}  // namespace mfuse
}  // namespace mlir
