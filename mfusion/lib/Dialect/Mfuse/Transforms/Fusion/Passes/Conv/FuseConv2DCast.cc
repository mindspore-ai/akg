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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/Conv/FuseConv2DCast.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Dialect/Mfuse/Support/ArithUtils.h"
#include "mfusion/Support/Logging.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_FUSECONV2DCAST
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

// Reference: conv2d_cast_fusion_pass.cc pattern 2: AclnnConv2D (fp16) -> Cast (fp32) -> AclnnConv2D (fp32), remove
// Cast. Constraints: no dynamic; AclnnConv2D inputs not from StridedRead.

/// Find any CastOp user of \p value that casts f16 to f32. Returns the first matching CastOp if found.
static CastOp getF16ToF32Cast(Value value) {
  auto inType = dyn_cast<RankedTensorType>(value.getType());
  if (!inType || !isa<Float16Type>(inType.getElementType())) {
    return nullptr;
  }
  for (Operation *user : value.getUsers()) {
    auto castOp = dyn_cast<CastOp>(user);
    if (!castOp) {
      continue;
    }
    auto castResultType = dyn_cast<RankedTensorType>(castOp.getResult().getType());
    if (!castResultType) {
      continue;
    }
    auto castDtype = dyn_cast_or_null<FloatType>(castResultType.getElementType());
    if (!castDtype || !castDtype.isF32()) {
      continue;
    }
    return castOp;
  }
  return CastOp();
}

/// Return true if the value is produced by an op whose name contains "strided_read".
static bool isDefinedByStridedRead(Value value) {
  Operation *def = value.getDefiningOp();
  if (!def) {
    return false;
  }
  return def->getName().getStringRef().contains("strided_read");
}

/// Pattern: AclnnConv2D (f16) -> Cast (f32) -> AclnnConv2D (f32 output).
/// Constraints:
/// - AclnnConv2D input f16, Cast output f32.
/// - AclnnConv2D and Cast are not dynamic (static shapes).
/// - No AclnnConv2D input is produced by StridedRead.
/// - Conv has exactly one user (the f16->f32 Cast); otherwise we do not fuse to avoid increasing conv count.
class FuseConv2DCastPattern : public OpRewritePattern<AclnnConv2DOp> {
 public:
  using OpRewritePattern<AclnnConv2DOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AclnnConv2DOp convOp, PatternRewriter &rewriter) const override {
    Value convResult = convOp.getResult();
    auto castOp = getF16ToF32Cast(convResult);
    if (!castOp) {
      return rewriter.notifyMatchFailure(convOp, "no f16->f32 Cast user");
    }
    if (!convResult.hasOneUse()) {
      return rewriter.notifyMatchFailure(convOp, "conv must have exactly one user (the Cast) to fuse");
    }

    if (hasDynamicShape(convOp.getResult().getType()) || hasDynamicShape(castOp.getResult().getType())) {
      return rewriter.notifyMatchFailure(convOp, "conv or cast has dynamic shape");
    }
    if (hasDynamicShape(convOp.getInput().getType()) || hasDynamicShape(convOp.getWeight().getType())) {
      return rewriter.notifyMatchFailure(convOp, "conv input has dynamic shape");
    }

    if (isDefinedByStridedRead(convOp.getInput()) || isDefinedByStridedRead(convOp.getWeight())) {
      return rewriter.notifyMatchFailure(convOp, "conv input from StridedRead");
    }

    MLOG(DEBUG) << "FuseConv2DCastPattern matched AclnnConv2DOp@" << convOp.getLoc() << " + Cast(f16->f32)@"
                << castOp.getLoc() << " -> single AclnnConv2D(f32)";

    Type outType = castOp.getResult().getType();
    Value newConv = rewriter.create<AclnnConv2DOp>(convOp.getLoc(), outType, convOp.getInput(), convOp.getWeight(),
                                                   convOp.getStride(), convOp.getPadding(), convOp.getDilation(),
                                                   convOp.getTransposedAttr(), convOp.getOutputPadding(),
                                                   convOp.getGroupsAttr());
    MLOG(DEBUG) << "FuseConv2DCast: created AclnnConv2DOp@" << newConv.getDefiningOp()->getLoc() << " (f32 output)";
    rewriter.replaceOp(castOp, newConv);
    MLOG(DEBUG) << "FuseConv2DCast: replaced CastOp with AclnnConv2DOp";
    return success();
  }
};

}  // namespace

DEFINE_MFUSE_FUSION_PASS(FuseConv2DCast, FuseConv2DCastPattern)

}  // namespace mfuse

}  // namespace mlir
