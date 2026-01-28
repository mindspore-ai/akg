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

#include "mfusion/Dialect/Muse/Transforms/Fusion/GeluFusion.h"

#include <cmath>

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/Transforms/Passes.h"
#include "mfusion/Dialect/Muse/Utils/ArithUtils.h"
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
#include "mfusion/Dialect/Muse/Transforms/Passes.h.inc"

namespace muse {
namespace {

constexpr double kSqrt2OverPi = 0.79788456080286535588;  // sqrt(2/pi)
constexpr double kGeluCoeff = 0.044715;
constexpr double kTolerance = 1e-6;

}  // namespace

/**
 * Fuse GELU approximation pattern into muse.gelu.
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
    // Both operands are non-constants, so handle commutativity
    MulOp halfMul = nullptr;
    AddOp addOneTanh = nullptr;
    Value x;

    // Try Pattern 1: Mul(Mul(x, 0.5), Add(...))
    if (auto m = lhs.getDefiningOp<MulOp>()) {
      double scalar;
      Value tensorOp;
      if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - 0.5) <= kTolerance) {
        if (auto a = rhs.getDefiningOp<AddOp>()) {
          // Check Add(tanh(...), 1) - after canonicalization, 1 is on right
          // But handle both orders for non-constant Add
          Value tanhVal = a.getX();
          Value ones = a.getY();
          if (!isConstOne(ones, kTolerance)) {
            tanhVal = a.getY();
            ones = a.getX();
          }
          if (isConstOne(ones, kTolerance)) {
            x = tensorOp;  // x is the non-scalar operand of Mul(x, 0.5)
            halfMul = m;
            addOneTanh = a;
          }
        }
      }
    }
    // Try Pattern 1 commutative: Mul(Add(...), Mul(x, 0.5))
    if (!halfMul || !addOneTanh) {
      if (auto m = rhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - 0.5) <= kTolerance) {
          if (auto a = lhs.getDefiningOp<AddOp>()) {
            Value tanhVal = a.getX();
            Value ones = a.getY();
            if (!isConstOne(ones, kTolerance)) {
              tanhVal = a.getY();
              ones = a.getX();
            }
            if (isConstOne(ones, kTolerance)) {
              x = tensorOp;
              halfMul = m;
              addOneTanh = a;
            }
          }
        }
      }
    }

    // Try Pattern 2: Mul(x, Mul(Add(...), 0.5))
    if (!halfMul || !addOneTanh) {
      if (auto m = rhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - 0.5) <= kTolerance) {
          if (auto a = tensorOp.getDefiningOp<AddOp>()) {
            // After canonicalization: Add(tanh(...), 1) - tanh on left, 1 on right (constant)
            Value tanhVal = a.getX();
            Value ones = a.getY();
            if (isConstOne(ones, kTolerance)) {
              x = lhs;
              halfMul = m;
              addOneTanh = a;
            }
          }
        }
      }
    }
    // Try Pattern 2 commutative: Mul(Mul(Add(...), 0.5), x)
    if (!halfMul || !addOneTanh) {
      if (auto m = lhs.getDefiningOp<MulOp>()) {
        double scalar;
        Value tensorOp;
        if (isScalarMul(m, scalar, tensorOp) && std::abs(scalar - 0.5) <= kTolerance) {
          if (auto a = tensorOp.getDefiningOp<AddOp>()) {
            Value tanhVal = a.getX();
            Value ones = a.getY();
            if (isConstOne(ones, kTolerance)) {
              x = rhs;
              halfMul = m;
              addOneTanh = a;
            }
          }
        }
      }
    }

    if (!halfMul || !addOneTanh) {
      return failure();
    }

    // Extract ones and tanhVal, handling commutativity for Add
    Value ones = addOneTanh.getX();
    Value tanhVal = addOneTanh.getY();
    if (!isConstOne(ones, kTolerance)) {
      ones = addOneTanh.getY();
      tanhVal = addOneTanh.getX();
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

    auto addOp = addInner.getDefiningOp<AddOp>();
    if (!addOp) {
      return failure();
    }

    // Add(x, Mul(x^3, 0.044715)) - both operands are non-constants, handle commutativity
    Value xAdd = addOp.getX();
    Value mulC = addOp.getY();
    if (xAdd != x) {
      // Try Add(Mul(x^3, 0.044715), x) - commutativity for non-constant operands
      xAdd = addOp.getY();
      mulC = addOp.getX();
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
    if (!isSingleElementFloat(exponent, 3.0)) {
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

    // Create AclnnGeluOp
    auto gelu = rewriter.create<AclnnGeluOp>(rootMul.getLoc(), x.getType(), x);
    rewriter.replaceOp(rootMul, gelu.getResult());

    // Erase unused operations
    SmallVector<Operation *, 16> toErase = {halfMul, addOneTanh, tanhOp, mulSqrt, addOp, mulCoz, powOp};
    if (auto onesOp = ones.getDefiningOp()) {
      toErase.push_back(onesOp);
    }
    for (Operation *op : toErase) {
      if (op && op->use_empty()) {
        rewriter.eraseOp(op);
      }
    }
    return success();
  }
};

struct FuseGeluPass : public impl::FuseGeluBase<FuseGeluPass> {
  void runOnOperation() override {
    // Run canonicalization pass first to normalize operand order
    // This ensures constants appear on the right side of operators
    PassManager pm(&getContext());
    OpPassManager &opm = pm.nest<ModuleOp>();
    opm.addPass(createCanonicalizerPass());
    if (failed(pm.run(getOperation()))) {
      signalPassFailure();
      return;
    }

    // Then run the GELU fusion pattern
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseGeluPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace muse

std::unique_ptr<Pass> createFuseGeluPass() {
  return std::make_unique<muse::FuseGeluPass>();
}

}  // namespace mlir
