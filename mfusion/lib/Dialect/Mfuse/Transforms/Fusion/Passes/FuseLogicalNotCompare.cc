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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseLogicalNotCompare.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DEF_FUSELOGICALNOTCOMPARE
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Ordered compare negation is only safe when both operands are known to be
// NaN-free. External floating-point values are not provably NaN-free here, so
// they must keep logical_not(compare(...)). Integer/index values are always
// safe, and a cast/full is safe only if its source is already provably safe.
static bool isProvablyNaNFreeValue(Value value) {
  auto shapedType = dyn_cast<ShapedType>(value.getType());
  if (!shapedType) {
    return false;
  }

  Type elementType = shapedType.getElementType();
  if (elementType.isIntOrIndex()) {
    return true;
  }
  if (!isa<FloatType>(elementType)) {
    return false;
  }

  if (auto constantOp = value.getDefiningOp<ConstantOp>()) {
    auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    if (!denseAttr) {
      return false;
    }
    return llvm::all_of(denseAttr.getValues<APFloat>(), [](const APFloat &floatValue) { return !floatValue.isNaN(); });
  }

  if (auto castOp = value.getDefiningOp<CastOp>()) {
    // Casting a provably NaN-free value cannot introduce NaN.
    return isProvablyNaNFreeValue(castOp.getInput());
  }

  if (auto fullOp = value.getDefiningOp<FullOp>()) {
    // Filling a tensor with a provably NaN-free scalar stays NaN-free.
    return isProvablyNaNFreeValue(fullOp.getFillValue());
  }

  return false;
}

static bool hasNaNFreeComparisonOperands(Value lhs, Value rhs) {
  return isProvablyNaNFreeValue(lhs) && isProvablyNaNFreeValue(rhs);
}

template <typename SrcOp, typename DstOp, bool RequiresNaNCheck = false>
class FoldLogicalNotComparePattern : public OpRewritePattern<LogicalNotOp> {
 public:
  using OpRewritePattern<LogicalNotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LogicalNotOp logicalNotOp, PatternRewriter &rewriter) const override {
    auto srcOp = logicalNotOp.getInput().template getDefiningOp<SrcOp>();
    if (!srcOp) {
      return failure();
    }

    if (RequiresNaNCheck && !hasNaNFreeComparisonOperands(srcOp.getSelf(), srcOp.getOther())) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<DstOp>(logicalNotOp, logicalNotOp.getType(), srcOp.getSelf(), srcOp.getOther());
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseLogicalNotCompare, FoldLogicalNotComparePattern<EqOp, NeOp>,
                         FoldLogicalNotComparePattern<NeOp, EqOp>, FoldLogicalNotComparePattern<GtOp, LeOp, true>,
                         FoldLogicalNotComparePattern<GeOp, LtOp, true>, FoldLogicalNotComparePattern<LtOp, GeOp, true>,
                         FoldLogicalNotComparePattern<LeOp, GtOp, true>)

}  // namespace mfuse
}  // namespace mlir
