/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
    return builder.create<arith::ExtFOp>(loc, f32Type, value);
  }
  return value;
}

// Helper function to convert f32 value to bf16
static Value convertF32ToBF16(OpBuilder &builder, Location loc, Value value) {
  Type valueType = value.getType();
  if (isa<Float32Type>(valueType)) {
    auto bf16Type = builder.getBF16Type();
    if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      FloatAttr floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
      APFloat f32Value = floatAttr.getValue();
      bool losesInfo = false;
      APFloat bf16Value(f32Value);
      bf16Value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
      FloatAttr bf16Attr = FloatAttr::get(bf16Type, bf16Value);
      return builder.create<arith::ConstantOp>(loc, bf16Type, bf16Attr);
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
    if (!llvm::any_of(loadedValue.getUses(),
                      [](OpOperand &use) { return !isa<affine::AffineStoreOp, arith::ExtFOp>(use.getOwner()); })) {
      return failure();  // All uses are store or ExtFOp, skip conversion
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

// Pattern to convert affine.const from bf16 to f32
struct ConstantOpBF16ToF32Pattern : public OpRewritePattern<arith::ConstantOp> {
  explicit ConstantOpBF16ToF32Pattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<arith::ConstantOp>(context, benefit) {}

  LogicalResult matchAndRewrite(arith::ConstantOp constantOp, PatternRewriter &rewriter) const override {
    auto floatType = dyn_cast<FloatType>(constantOp.getType());
    if (!floatType || !floatType.isBF16()) return failure();

    if (!isa<FloatAttr>(constantOp.getValue())) return failure();

    // Check if all uses are only affine.store or arith::ExtFOp
    // If so, we don't need to convert (store will handle it, or ExtFOp already exists)
    if (!llvm::any_of(constantOp.getResult().getUsers(),
                      [](OpOperand use) { return !isa<affine::AffineStoreOp, arith::ExtFOp>(use.getOwner()); })) {
      return failure();  // All uses are store or ExtFOp, skip conversion
    }

    FloatAttr floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
    APFloat bf16Value = floatAttr.getValue();

    bool losesInfo = false;
    APFloat fp32Value(bf16Value);
    fp32Value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);

    Type fp32Type = rewriter.getF32Type();
    FloatAttr fp32Attr = FloatAttr::get(fp32Type, fp32Value);

    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constantOp, fp32Type, fp32Attr);
    return success();
  }
};

struct TruncFOpPattern : public OpRewritePattern<arith::TruncFOp> {
  explicit TruncFOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::TruncFOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::TruncFOp truncOp, PatternRewriter &rewriter) const override {
    Value value = truncOp.getIn();
    if (value.getType().isBF16()) {
      rewriter.replaceOp(truncOp, value);
      return success();
    }
    return failure();
  }
};

struct ExtFOpPattern : public OpRewritePattern<arith::ExtFOp> {
  explicit ExtFOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::ExtFOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::ExtFOp extfOp, PatternRewriter &rewriter) const override {
    auto operand = extfOp.getOperand();
    if (isa<Float32Type>(operand.getType())) {
      rewriter.replaceOp(extfOp, operand);
      return success();
    }
    return failure();
  }
};

// Pattern to convert affine.store f32 value to bf16 memref
// Benefit = 2: Medium priority, process stores after loads but before arithmetic ops
struct AffineStoreF32ToBF16Pattern : public OpRewritePattern<affine::AffineStoreOp> {
  explicit AffineStoreF32ToBF16Pattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<affine::AffineStoreOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp, PatternRewriter &rewriter) const override {
    Value valueToStore = storeOp.getValueToStore();
    Value memref = storeOp.getMemRef();

    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return failure();
    }

    // Check if memref is bf16 type and value is f32
    bool isBF16Memref = isa<BFloat16Type>(memrefType.getElementType());
    bool isF32Value = isa<Float32Type>(valueToStore.getType());

    if (!isBF16Memref || !isF32Value) {
      return failure();  // Not storing f32 to bf16 memref, skip
    }

    // Convert f32 value to bf16 before storing
    Location loc = storeOp.getLoc();
    Value bf16Value = convertF32ToBF16(rewriter, loc, valueToStore);

    // Create new store with bf16 value
    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(storeOp, bf16Value, memref, storeOp.getIndices());

    return success();
  }
};

FailureOr<Operation *> convertOpResultTypes(Operation *op, ValueRange operands, const TypeConverter &converter,
                                            PatternRewriter &rewriter) {
  assert(op && "Invalid op");
  Location loc = op->getLoc();
  if (converter.isLegal(op->getResultTypes())) return rewriter.notifyMatchFailure(loc, "op already legal");

  OperationState newOp(loc, op->getName());
  newOp.addOperands(operands);

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
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, arith::TruncFOp, arith::ConstantOp>(op)) {
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
    patterns.add<AffineStoreF32ToBF16Pattern>(context);
    patterns.add<ExtFOpPattern>(context);
    patterns.add<TruncFOpPattern>(context);
    patterns.add<ConstantOpBF16ToF32Pattern>(context);

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
