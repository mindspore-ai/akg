/**
 * Copyright 2025-2026 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/BF16ToF32.h"

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_BF16TOF32
#define GEN_PASS_DECL_BF16TOF32
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

#define DEBUG_TYPE "bf16-to-f32"

namespace mlir {
namespace {

// Helper function to convert bf16 value to f32
static Value convertBF16ToF32(OpBuilder &builder, Location loc, Value value) {
  Type valueType = value.getType();
  if (isa<BFloat16Type>(valueType)) {
    auto f32Type = builder.getF32Type();
    if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      FloatAttr floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
      APFloat bf16Value = floatAttr.getValue();
      bool losesInfo = false;
      APFloat f32Value(bf16Value);
      f32Value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
      return builder.create<arith::ConstantOp>(loc, f32Type, FloatAttr::get(f32Type, f32Value));
    } else if (auto truncFOp = dyn_cast<arith::TruncFOp>(value.getDefiningOp())) {
      if (isa<Float32Type>(truncFOp.getIn().getType())) {
        return truncFOp.getIn();
      }
    }
    return builder.create<arith::ExtFOp>(loc, f32Type, value);
  }
  return value;
}

// Helper function to convert wider float value to bf16
static Value convertWideFPToBF16(OpBuilder &builder, Location loc, Value value) {
  Type valueType = value.getType();
  if (isa<Float32Type, Float64Type>(valueType)) {
    auto bf16Type = builder.getBF16Type();
    if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      FloatAttr floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
      APFloat f32Value = floatAttr.getValue();
      bool losesInfo = false;
      APFloat bf16Value(f32Value);
      bf16Value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
      FloatAttr bf16Attr = FloatAttr::get(bf16Type, bf16Value);
      return builder.create<arith::ConstantOp>(loc, bf16Type, bf16Attr);
    } else if (auto extFOp = dyn_cast<arith::ExtFOp>(value.getDefiningOp())) {
      if (extFOp.getIn().getType().isBF16()) {
        return extFOp.getIn();
      }
    }
    return builder.create<arith::TruncFOp>(loc, bf16Type, value);
  }
  return value;
}

// Pattern to convert affine.load from bf16 memref to f32
// Benefit = 3: Higher priority to process loads first (before arithmetic ops)
struct AffineLoadBF16ToF32Pattern : public OpRewritePattern<affine::AffineLoadOp> {
  explicit AffineLoadBF16ToF32Pattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<affine::AffineLoadOp>(context, benefit) {}

  LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp, PatternRewriter &rewriter) const override {
    Value memref = loadOp.getMemRef();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());

    if (!memrefType || !isa<BFloat16Type>(memrefType.getElementType())) {
      return failure();  // Not a bf16 memref, skip
    }

    Value loadedValue = loadOp.getResult();

    // Check if all uses are only affine.store or arith::ExtFOp
    // If so, we don't need to convert (store will handle it, or ExtFOp already exists)
    if (!llvm::any_of(loadedValue.getUses(), [](OpOperand &use) {
          return !isa<affine::AffineStoreOp, arith::ExtFOp, arith::BitcastOp>(use.getOwner());
        })) {
      return failure();  // All uses are store BitcastOp or ExtFOp, skip conversion
    }
    // The memref has bf16 element type, so the load will return a bf16 value
    // We need to convert the loaded bf16 value to f32
    // Insert the conversion right after the load operation
    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);

    // Create the conversion operation (f32Value uses loadedValue as input)
    Value f32Value = convertBF16ToF32(rewriter, loc, loadedValue);
    Operation *conversionOp = f32Value.getDefiningOp();

    // Replace all uses of the original load result with the converted f32 value
    // But exclude the conversion operation itself (f32Value's defining op uses loadedValue)
    rewriter.replaceUsesWithIf(loadedValue, f32Value, [conversionOp](OpOperand &use) {
      // Replace all uses except the one in the conversion operation itself
      return use.getOwner() != conversionOp;
    });

    return success();
  }
};

struct TruncFOpPattern : public OpRewritePattern<arith::TruncFOp> {
  explicit TruncFOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::TruncFOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::TruncFOp truncOp, PatternRewriter &rewriter) const override {
    Value value = truncOp.getIn();
    if (value.getType() == truncOp.getResult().getType()) {
      rewriter.replaceOp(truncOp, value);
      return success();
    }
    if (isa<arith::ConstantOp>(value.getDefiningOp())) {
      Value bf16Value = convertWideFPToBF16(rewriter, truncOp.getLoc(), value);
      rewriter.replaceOp(truncOp, bf16Value);
    }
    return failure();
  }
};

struct ExtFOpPattern : public OpRewritePattern<arith::ExtFOp> {
  explicit ExtFOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::ExtFOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::ExtFOp extFOp, PatternRewriter &rewriter) const override {
    Value value = extFOp.getIn();
    if (value.getType() == extFOp.getResult().getType()) {
      rewriter.replaceOp(extFOp, value);
      return success();
    }
    return failure();
  }
};

struct BitcastOpPattern : public OpRewritePattern<arith::BitcastOp> {
  explicit BitcastOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::BitcastOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::BitcastOp bitcastOp, PatternRewriter &rewriter) const override {
    Value value = bitcastOp.getIn();
    if (value.getType() == bitcastOp.getResult().getType()) {
      rewriter.replaceOp(bitcastOp, value);
      return success();
    }
    auto resType = bitcastOp.getResult().getType();
    if (isa<Float32Type>(value.getType()) && resType.getIntOrFloatBitWidth() == 16) {
      Value bf16Value = convertWideFPToBF16(rewriter, bitcastOp.getLoc(), value);
      rewriter.replaceOpWithNewOp<arith::BitcastOp>(bitcastOp, resType, bf16Value);
    }
    return failure();
  }
};

// Pattern to convert affine.store wider float value to bf16 memref
// Benefit = 2: Medium priority, process stores after loads but before arithmetic ops
struct AffineStoreWideFPToBF16Pattern : public OpRewritePattern<affine::AffineStoreOp> {
  explicit AffineStoreWideFPToBF16Pattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<affine::AffineStoreOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp, PatternRewriter &rewriter) const override {
    Value valueToStore = storeOp.getValueToStore();
    Value memref = storeOp.getMemRef();

    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return failure();
    }

    // Check if memref is bf16 type and value is wider float
    if (!isa<BFloat16Type>(memrefType.getElementType()) || !isa<Float32Type, Float64Type>(valueToStore.getType())) {
      return failure();  // Not storing f32 to bf16 memref, skip
    }

    // Convert wider float value to bf16 before storing
    Value bf16Value = convertWideFPToBF16(rewriter, storeOp.getLoc(), valueToStore);
    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(storeOp, bf16Value, memref,
                                                       storeOp.getAffineMapAttr().getValue(), storeOp.getIndices());
    return success();
  }
};

FailureOr<Operation *> convertOpResultTypes(Operation *op, ValueRange operands, const TypeConverter &converter,
                                            PatternRewriter &rewriter) {
  assert(op && "Invalid op");
  Location loc = op->getLoc();
  if (converter.isLegal(op->getResultTypes())) return rewriter.notifyMatchFailure(loc, "op already legal");

  OperationState newOp(loc, op->getName());

  SmallVector<Value, 4> newOperands;
  for (auto operand : operands) {
    if (isa<BFloat16Type>(operand.getType())) {
      Value f32Value = convertBF16ToF32(rewriter, loc, operand);
      newOperands.push_back(f32Value);
    } else {
      newOperands.push_back(operand);
    }
  }
  newOp.addOperands(newOperands);

  SmallVector<Type> newResultTypes;
  if (failed(converter.convertTypes(op->getResultTypes(), newResultTypes)))
    return rewriter.notifyMatchFailure(loc, "couldn't convert return types");

  newOp.addTypes(newResultTypes);
  newOp.addAttributes(op->getAttrs());
  return rewriter.create(newOp);
}

struct LegalizeBF16RewritePattern final : RewritePattern {
  explicit LegalizeBF16RewritePattern(MLIRContext *context, const TypeConverter &converter, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag{}, benefit, context), typeConverter(converter) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, arith::TruncFOp, arith::ConstantOp, arith::BitcastOp>(op)) {
      return failure();
    }
    FailureOr<Operation *> legalized = convertOpResultTypes(op, op->getOperands(), typeConverter, rewriter);
    if (failed(legalized)) return failure();

    rewriter.replaceOp(op, (*legalized));
    return success();
  }

 private:
  TypeConverter typeConverter;
};

class BF16ToF32Pass : public impl::BF16ToF32Base<BF16ToF32Pass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) -> std::optional<Type> { return type; });
    typeConverter.addConversion([](FloatType type) -> std::optional<Type> {
      if (type.isBF16()) return Float32Type::get(type.getContext());
      return std::nullopt;
    });

    // Add patterns for affine operations
    patterns.add<AffineLoadBF16ToF32Pattern>(context);
    patterns.add<AffineStoreWideFPToBF16Pattern>(context);
    patterns.add<ExtFOpPattern>(context);
    patterns.add<TruncFOpPattern>(context);
    patterns.add<BitcastOpPattern>(context);
    patterns.add<LegalizeBF16RewritePattern>(context, typeConverter);

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    // Apply patterns using greedy pattern rewrite driver
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> createBF16ToF32Pass() { return std::make_unique<BF16ToF32Pass>(); }
}  // namespace mlir
