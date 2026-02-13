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

#include "mfusion/Dialect/Mfuse/Transforms/Fusion/Passes/FuseGelu.h"

#include <cmath>
#include <utility>

#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Utils/ArithUtils.h"
#include "mfusion/Dialect/Mfuse/Transforms/Fusion/FusionPassMacros.h"
#include "mfusion/Support/Logging.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_FUSEGELU
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

namespace mfuse {
namespace {

constexpr double kSqrt2OverPi = 0.79788456080286535588;  // sqrt(2/pi)
constexpr double kGeluCoeff = 0.044715;
constexpr double kGeluHalfCoeff = 0.5;    // 0.5 coefficient in GELU formula
constexpr double kGeluPowExponent = 3.0;  // Exponent for x^3 in GELU formula
constexpr double kTolerance = 1e-6;

/// If \p v is defined by mfuse.add or mfuse.aclnn.add (with alpha=1), set \p outX, \p outY and return true.
static bool getAddLikeOperands(Value v, Value &outX, Value &outY) {
  if (auto a = v.getDefiningOp<AddOp>()) {
    outX = a.getX();
    outY = a.getY();
    return true;
  }
  if (auto a = v.getDefiningOp<AclnnAddOp>()) {
    if (!isConstOne(a.getAlpha(), kTolerance)) return false;
    outX = a.getX();
    outY = a.getY();
    return true;
  }
  return false;
}

}  // namespace

/**
 * Fuse GELU approximation pattern into mfuse.gelu.
 * Pattern: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * After canonicalization, constants appear on the right side of operators:
 * x * (tanh((x + x^3 * 0.044715) * sqrt(2/pi)) + 1) * 0.5
 *
 * Canonicalized form structure:
 * - Mul(x, Mul(Add(tanh(...), 1), 0.5)) - x on left, Mul(Add(...), 0.5) on right
 * - Add(tanh(...), 1) - tanh on left, 1 on right
 * - Mul(Add(...), sqrt(2/pi)) - Add on left, sqrt on right
 * - Add(x, Mul(x^3, 0.044715)) - x on left, Mul on right
 * - Mul(x^3, 0.044715) - x^3 on left, 0.044715 on right
 */
class FuseGeluPattern : public OpRewritePattern<MulOp> {
 public:
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp rootMul, PatternRewriter &rewriter) const override {
    Value lhs = rootMul.getLhs();
    Value rhs = rootMul.getRhs();

    // Pattern 1: Mul(Mul(x, 0.5), Add(tanh(...), 1)) - original form
    // Pattern 2: Mul(x, Mul(Add(tanh(...), 1), 0.5)) - canonicalized form
    // Add can be mfuse.add or mfuse.aclnn.add (alpha=1) from Torch convert.
    MulOp halfMul = nullptr;
    Value addOneTanhOpX, addOneTanhOpY;
    Value x;

    auto setAddOneTanh = [&](Value ax, Value ay, Value tx, MulOp hm) {
      addOneTanhOpX = ax;
      addOneTanhOpY = ay;
      x = tx;
      halfMul = hm;
    };

    // Try Pattern 1: Mul(Mul(x, 0.5), Add(...))
    if (auto m = lhs.getDefiningOp<MulOp>()) {
      double scalar;
      Value tensorOp;
      if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - kGeluHalfCoeff) <= kTolerance) {
        Value addX, addY;
        if (getAddLikeOperands(rhs, addX, addY)) {
          Value tanhVal = addX;
          Value ones = addY;
          if (!isConstOne(ones, kTolerance)) {
            tanhVal = addY;
            ones = addX;
          }
          if (isConstOne(ones, kTolerance)) setAddOneTanh(addX, addY, tensorOp, m);
        }
      }
    }
    // Try Pattern 1 commutative: Mul(Add(...), Mul(x, 0.5))
    if (!halfMul) {
      if (auto m = rhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - kGeluHalfCoeff) <= kTolerance) {
          Value addX, addY;
          if (getAddLikeOperands(lhs, addX, addY)) {
            Value tanhVal = addX;
            Value ones = addY;
            if (!isConstOne(ones, kTolerance)) {
              tanhVal = addY;
              ones = addX;
            }
            if (isConstOne(ones, kTolerance)) setAddOneTanh(addX, addY, tensorOp, m);
          }
        }
      }
    }

    // Try Pattern 2: Mul(x, Mul(Add(...), 0.5))
    if (!halfMul) {
      if (auto m = rhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - kGeluHalfCoeff) <= kTolerance) {
          Value addX, addY;
          if (getAddLikeOperands(tensorOp, addX, addY) && isConstOne(addY, kTolerance)) {
            x = lhs;
            halfMul = m;
            addOneTanhOpX = addX;
            addOneTanhOpY = addY;
          }
        }
      }
    }
    // Try Pattern 2 commutative: Mul(Mul(Add(...), 0.5), x)
    if (!halfMul) {
      if (auto m = lhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - kGeluHalfCoeff) <= kTolerance) {
          Value addX, addY;
          if (getAddLikeOperands(tensorOp, addX, addY) && isConstOne(addY, kTolerance)) {
            x = rhs;
            halfMul = m;
            addOneTanhOpX = addX;
            addOneTanhOpY = addY;
          }
        }
      }
    }

    if (!halfMul) {
      return failure();
    }

    // Extract ones and tanhVal, handling commutativity for Add
    Value ones = addOneTanhOpX;
    Value tanhVal = addOneTanhOpY;
    if (!isConstOne(ones, kTolerance)) {
      ones = addOneTanhOpY;
      tanhVal = addOneTanhOpX;
    }

    auto tanhOp = tanhVal.getDefiningOp<AclnnTanhOp>();
    if (!tanhOp) {
      return failure();
    }
    Value inner = tanhOp.getInput();

    // After canonicalization: Mul(Add(...), sqrt(2/pi)) - Add on left, sqrt on right
    MulOp mulSqrt = nullptr;
    Value addInner;
    if (auto m = inner.getDefiningOp<MulOp>()) {
      double sqrtScalar;
      Value sqrtOperand;
      if (isScalarMul(m, sqrtScalar, sqrtOperand)) {
        if (std::abs(sqrtScalar - kSqrt2OverPi) <= kTolerance) {
          mulSqrt = m;
          addInner = sqrtOperand;  // Add is on left
        }
      }
    }
    if (!mulSqrt) {
      return failure();
    }

    Value xAdd, mulC;
    if (!getAddLikeOperands(addInner, xAdd, mulC)) {
      return failure();
    }
    // Add(x, Mul(x^3, 0.044715)) - handle commutativity for non-constant operands
    if (xAdd != x) {
      std::swap(xAdd, mulC);
      if (xAdd != x) {
        return failure();
      }
    }

    // After canonicalization: Mul(x^3, 0.044715) - x^3 on left, 0.044715 on right
    MulOp mulCoz = nullptr;
    Value pow3;
    if (auto m = mulC.getDefiningOp<MulOp>()) {
      double coeffScalar;
      Value powOperand;
      if (isScalarMul(m, coeffScalar, powOperand)) {
        if (std::abs(coeffScalar - kGeluCoeff) <= kTolerance) {
          mulCoz = m;
          pow3 = powOperand;  // Pow(x,3) is on left
        }
      }
    }
    if (!mulCoz) {
      return failure();
    }

    auto powOp = pow3.getDefiningOp<PowOp>();
    if (!powOp) {
      return failure();
    }
    Value xPow = powOp.getBase();
    Value exponent = powOp.getExponent();
    if (!isSingleElementFloat(exponent, kGeluPowExponent)) {
      return failure();
    }

    // Verify all x references point to the same value
    if (xPow != x) {
      return failure();
    }

    // Check if tensor type has static shape
    auto tensorType = dyn_cast<TensorType>(x.getType());
    if (!tensorType || !tensorType.hasStaticShape()) {
      return failure();
    }

    MLOG(DEBUG) << "FuseGeluPattern matched GELU approximation pattern";

    // Create AclnnGeluOp
    auto gelu = rewriter.create<AclnnGeluOp>(rootMul.getLoc(), x.getType(), x);
    MLOG(DEBUG) << "Created new AclnnGeluOp";
    rewriter.replaceOp(rootMul, gelu.getResult());
    MLOG(DEBUG) << "Replaced original GELU approximation pattern with new AclnnGeluOp";
    return success();
  }
};

DEFINE_MFUSE_FUSION_PASS(FuseGelu, FuseGeluPattern)
}  // namespace mfuse

}  // namespace mlir
