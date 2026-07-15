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

#include "akg/Dialect/Affine/Transforms/LegalizeTypeForAscend.h"

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
#include "akg/Utils/Constants.h"

namespace mlir {
#define GEN_PASS_DEF_LEGALIZETYPEFORASCEND
#define GEN_PASS_DECL_LEGALIZETYPEFORASCEND
#include "akg/Dialect/Affine/Passes.h.inc"

}  // namespace mlir

#define DEBUG_TYPE "legalize-type-for-ascend"

namespace mlir {
namespace {

// Helper function to convert i64 value to i32
// TruncIOp requires signless integer types, so we always produce signless i32.
// For signed/unsigned i64, the value is first bitcast to signless before trunci.
// Signedness is handled by ExtUIOp/ExtSIOp when storing back to i64 memref.
static Value convertToI32(OpBuilder &builder, Location loc, Value value) {
  Type valueType = value.getType();
  if (auto intTy = dyn_cast<IntegerType>(valueType)) {
    if (intTy.getWidth() == kI64BitWidth) {
      auto i32Type = builder.getI32Type();
      if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
        IntegerAttr intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
        APInt i64Val = intAttr.getValue();
        APInt i32Val = i64Val.trunc(kI32BitWidth);
        return builder.create<arith::ConstantOp>(loc, i32Type, builder.getIntegerAttr(i32Type, i32Val));
      }
      // TruncIOp only accepts signless integers; bitcast signed/unsigned to signless first
      Value truncInput = value;
      if (intTy.getSignedness() != IntegerType::Signless) {
        auto signlessI64 = builder.getIntegerType(kI64BitWidth);
        truncInput = builder.create<arith::BitcastOp>(loc, signlessI64, value);
      }
      return builder.create<arith::TruncIOp>(loc, i32Type, truncInput);
    }
  }
  return value;
}

// Helper function to convert bf16 value to f32
static Value convertToF32(OpBuilder &builder, Location loc, Value value) {
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
    }
    if (auto truncFOp = dyn_cast<arith::TruncFOp>(value.getDefiningOp())) {
      if (isa<Float32Type>(truncFOp.getIn().getType())) {
        return truncFOp.getIn();
      }
    }
    return builder.create<arith::ExtFOp>(loc, f32Type, value);
  }
  if (isa<Float64Type>(valueType)) {
    auto f32Type = builder.getF32Type();
    if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      FloatAttr floatAttr = dyn_cast<FloatAttr>(constantOp.getValue());
      APFloat f64Value = floatAttr.getValue();
      bool losesInfo = false;
      APFloat f32Value(f64Value);
      f32Value.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
      return builder.create<arith::ConstantOp>(loc, f32Type, FloatAttr::get(f32Type, f32Value));
    } else if (auto extFOp = dyn_cast<arith::ExtFOp>(value.getDefiningOp())) {
      if (isa<Float32Type>(extFOp.getIn().getType())) {
        return extFOp.getIn();
      }
    }
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
    }
    if (auto extFOp = dyn_cast<arith::ExtFOp>(value.getDefiningOp())) {
      if (extFOp.getIn().getType().isBF16()) {
        return extFOp.getIn();
      }
    }
    return builder.create<arith::TruncFOp>(loc, bf16Type, value);
  }
  return value;
}

// Decompose i8 -> i64 sign extension into steps that VCast supports (see ArithToHIVM
// UnaryNPUVectorToHIVMCast: i8 -> f16 -> i32 -> i64).
static Value convertI8ExtSiToI64(OpBuilder &builder, Location loc, Value v) {
  auto f16Ty = builder.getF16Type();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  Value f16Val = builder.create<arith::SIToFPOp>(loc, f16Ty, v);
  Value i32Val = builder.create<arith::FPToSIOp>(loc, i32Ty, f16Val);
  return builder.create<arith::ExtSIOp>(loc, i64Ty, i32Val);
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
    Value f32Value = convertToF32(rewriter, loc, loadedValue);
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

// Pattern to convert affine.load from i64 memref to i32
struct AffineLoadI64ToI32Pattern : public OpRewritePattern<affine::AffineLoadOp> {
  explicit AffineLoadI64ToI32Pattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<affine::AffineLoadOp>(context, benefit) {}

  LogicalResult matchAndRewrite(affine::AffineLoadOp loadOp, PatternRewriter &rewriter) const override {
    Value memref = loadOp.getMemRef();
    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return failure();
    }
    auto elemTy = dyn_cast<IntegerType>(memrefType.getElementType());
    if (!elemTy || elemTy.getWidth() != kI64BitWidth) {
      return failure();
    }

    Value loadedValue = loadOp.getResult();
    if (!llvm::any_of(loadedValue.getUses(),
                      [](OpOperand &use) { return !isa<affine::AffineStoreOp, arith::TruncIOp>(use.getOwner()); })) {
      return failure();
    }

    Location loc = loadOp.getLoc();
    rewriter.setInsertionPointAfter(loadOp);
    Value i32Value = convertToI32(rewriter, loc, loadedValue);
    Operation *conversionOp = i32Value.getDefiningOp();
    rewriter.replaceUsesWithIf(loadedValue, i32Value,
                               [conversionOp](OpOperand &use) { return use.getOwner() != conversionOp; });
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
    if (isa<BFloat16Type>(truncOp.getResult().getType()) && isa<arith::ConstantOp>(value.getDefiningOp())) {
      Value bf16Value = convertWideFPToBF16(rewriter, truncOp.getLoc(), value);
      rewriter.replaceOp(truncOp, bf16Value);
      return success();
    }
    if (isa<Float32Type>(truncOp.getResult().getType()) && isa<arith::ConstantOp>(value.getDefiningOp())) {
      Value f32Value = convertToF32(rewriter, truncOp.getLoc(), value);
      rewriter.replaceOp(truncOp, f32Value);
      return success();
    }
    return failure();
  }
};

struct TruncIOpPattern : public OpRewritePattern<arith::TruncIOp> {
  explicit TruncIOpPattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<arith::TruncIOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::TruncIOp truncOp, PatternRewriter &rewriter) const override {
    Value value = truncOp.getIn();
    if (value.getType() == truncOp.getResult().getType()) {
      rewriter.replaceOp(truncOp, value);
      return success();
    }
    // Fold trunci of a constant
    if (auto constantOp = dyn_cast<arith::ConstantOp>(value.getDefiningOp())) {
      auto resIntTy = dyn_cast<IntegerType>(truncOp.getResult().getType());
      if (!resIntTy) {
        return failure();
      }
      IntegerAttr intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
      if (!intAttr) {
        return failure();
      }
      APInt truncated = intAttr.getValue().trunc(resIntTy.getWidth());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(truncOp, resIntTy, rewriter.getIntegerAttr(resIntTy, truncated));
      return success();
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

// i8 -> i64 extsi: VCast-legal decomposition via float (see ArithToHIVM NPUVector path).
struct ExtSIOpPattern : public OpRewritePattern<arith::ExtSIOp> {
  explicit ExtSIOpPattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<arith::ExtSIOp>(context, benefit) {}
  LogicalResult matchAndRewrite(arith::ExtSIOp op, PatternRewriter &rewriter) const override {
    Value in = op.getIn();
    Type inTy = in.getType();
    Type outTy = op.getResult().getType();
    if (inTy == outTy) {
      rewriter.replaceOp(op, in);
      return success();
    }
    if (!inTy.isInteger(kI8BitWidth) || !outTy.isInteger(kI64BitWidth)) {
      return failure();
    }
    Value i64Val = convertI8ExtSiToI64(rewriter, op.getLoc(), in);
    rewriter.replaceOp(op, i64Val);
    return success();
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
    if (isa<Float32Type>(value.getType()) && resType.getIntOrFloatBitWidth() == kI16BitWidth) {
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

// Pattern to convert affine.store i32 value to i64 memref
// Benefit = 2: Medium priority, process stores after loads but before arithmetic ops
struct AffineStoreI64ToI32Pattern : public OpRewritePattern<affine::AffineStoreOp> {
  explicit AffineStoreI64ToI32Pattern(MLIRContext *context, PatternBenefit benefit = 2)
      : OpRewritePattern<affine::AffineStoreOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineStoreOp storeOp, PatternRewriter &rewriter) const override {
    Value valueToStore = storeOp.getValueToStore();
    Value memref = storeOp.getMemRef();

    auto memrefType = dyn_cast<MemRefType>(memref.getType());
    if (!memrefType) {
      return failure();
    }

    auto elemTy = dyn_cast<IntegerType>(memrefType.getElementType());
    if (!elemTy || elemTy.getWidth() != kI64BitWidth) {
      return failure();  // Not an i64 memref, skip
    }

    auto valTy = dyn_cast<IntegerType>(valueToStore.getType());
    if (!valTy || valTy.getWidth() != kI32BitWidth) {
      return failure();  // Value to store is not i32, skip
    }

    // Extend i32 value back to i64 before storing to i64 memref
    // Use ExtUIOp for unsigned, ExtSIOp otherwise (signed/signless)
    bool isUnsigned = (valTy.getSignedness() == IntegerType::Unsigned);
    Value i64Value = isUnsigned ? rewriter.create<arith::ExtUIOp>(storeOp.getLoc(), elemTy, valueToStore).getResult()
                                : rewriter.create<arith::ExtSIOp>(storeOp.getLoc(), elemTy, valueToStore);
    rewriter.replaceOpWithNewOp<affine::AffineStoreOp>(storeOp, i64Value, memref, storeOp.getAffineMapAttr().getValue(),
                                                       storeOp.getIndices());
    return success();
  }
};

// Pattern to convert affine.for wider float value to bf16 memref
struct AffineForFPToBF16Pattern : public OpRewritePattern<affine::AffineForOp> {
  explicit AffineForFPToBF16Pattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<affine::AffineForOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineForOp forOp, PatternRewriter &rewriter) const override {
    auto inits = forOp.getInitsMutable();
    if (inits.empty()) {
      return failure();
    }
    auto args = forOp.getRegionIterArgs();
    auto results = forOp.getResults();
    bool isMatch = false;
    for (size_t i = 0; i < inits.size(); i++) {
      Type valueType = inits[i].get().getType();
      if (isa<BFloat16Type, Float64Type>(valueType)) {
        isMatch = true;
        auto value = convertToF32(rewriter, forOp.getLoc(), inits[i].get());
        args[i].setType(value.getType());
        results[i].setType(value.getType());
        inits[i].set(value);
      }
    }
    if (!isMatch) {
      return failure();
    }
    return success();
  }
};

// Pattern to convert affine.for i64 induction init values to i32
struct AffineForI64ToI32Pattern : public OpRewritePattern<affine::AffineForOp> {
  explicit AffineForI64ToI32Pattern(MLIRContext *context, PatternBenefit benefit = 3)
      : OpRewritePattern<affine::AffineForOp>(context, benefit) {}
  LogicalResult matchAndRewrite(affine::AffineForOp forOp, PatternRewriter &rewriter) const override {
    auto inits = forOp.getInitsMutable();
    if (inits.empty()) {
      return failure();
    }
    auto args = forOp.getRegionIterArgs();
    auto results = forOp.getResults();
    bool isMatch = false;
    for (size_t i = 0; i < inits.size(); i++) {
      Type valueType = inits[i].get().getType();
      auto intTy = dyn_cast<IntegerType>(valueType);
      if (intTy && intTy.getWidth() == kI64BitWidth) {
        isMatch = true;
        auto value = convertToI32(rewriter, forOp.getLoc(), inits[i].get());
        args[i].setType(value.getType());
        results[i].setType(value.getType());
        inits[i].set(value);
      }
    }
    if (!isMatch) {
      return failure();
    }
    return success();
  }
};

FailureOr<Operation *> legalizeOp(Operation *op, ValueRange operands, const TypeConverter &converter,
                                  PatternRewriter &rewriter, bool enableI64ToI32) {
  assert(op && "Invalid op");
  Location loc = op->getLoc();

  bool isMatch = false;
  SmallVector<Value, kSmallVectorSizeFour> newOperands;
  for (auto operand : operands) {
    Type operandType = operand.getType();
    if (isa<BFloat16Type, Float64Type>(operandType)) {
      isMatch = true;
      Value f32Value = convertToF32(rewriter, loc, operand);
      newOperands.push_back(f32Value);
    } else if (enableI64ToI32) {
      auto intTy = dyn_cast<IntegerType>(operandType);
      if (intTy && intTy.getWidth() == kI64BitWidth) {
        isMatch = true;
        Value i32Value = convertToI32(rewriter, loc, operand);
        newOperands.push_back(i32Value);
        continue;
      }
      newOperands.push_back(operand);
    } else {
      newOperands.push_back(operand);
    }
  }
  if (!isMatch && converter.isLegal(op->getResultTypes())) {
    return rewriter.notifyMatchFailure(loc, "op already legal");
  }
  OperationState newOp(loc, op->getName());
  newOp.addOperands(newOperands);

  SmallVector<Type> newResultTypes;
  if (failed(converter.convertTypes(op->getResultTypes(), newResultTypes))) {
    return rewriter.notifyMatchFailure(loc, "couldn't convert return types");
  }

  newOp.addTypes(newResultTypes);
  newOp.addAttributes(op->getAttrs());
  return rewriter.create(newOp);
}

struct AffineIfAlignResultTypesWithYieldPattern : public OpRewritePattern<affine::AffineIfOp> {
  explicit AffineIfAlignResultTypesWithYieldPattern(MLIRContext *context, PatternBenefit benefit = 0)
      : OpRewritePattern<affine::AffineIfOp>(context, benefit) {}

  LogicalResult matchAndRewrite(affine::AffineIfOp ifOp, PatternRewriter &rewriter) const override {
    (void)rewriter;
    if (ifOp.getNumResults() == 0) {
      return failure();
    }
    auto thenYield = dyn_cast<affine::AffineYieldOp>(ifOp.getThenRegion().front().getTerminator());
    if (!thenYield) {
      return failure();
    }
    SmallVector<Type> yieldTypes(thenYield.getOperandTypes());
    if (yieldTypes.size() != ifOp.getNumResults()) {
      return failure();
    }
    if (!ifOp.getElseRegion().empty()) {
      auto elseYield = dyn_cast<affine::AffineYieldOp>(ifOp.getElseRegion().front().getTerminator());
      if (!elseYield) {
        return failure();
      }
      SmallVector<Type> elseTypes(elseYield.getOperandTypes());
      if (elseTypes != yieldTypes) {
        return failure();
      }
    }

    bool mismatch = false;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      if (ifOp.getResult(i).getType() != yieldTypes[i]) {
        mismatch = true;
        break;
      }
    }
    if (!mismatch) {
      return failure();
    }

    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      ifOp.getResult(i).setType(yieldTypes[i]);
    }
    return success();
  }
};

struct LegalizeBF16RewritePattern final : RewritePattern {
  explicit LegalizeBF16RewritePattern(MLIRContext *context, const TypeConverter &converter, bool enableI64ToI32,
                                      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag{}, benefit, context),
        typeConverter(converter),
        enableI64ToI32(enableI64ToI32) {}
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (isa<affine::AffineLoadOp, affine::AffineStoreOp, arith::ExtFOp, arith::TruncFOp, arith::ConstantOp,
            arith::BitcastOp, arith::ExtSIOp, arith::TruncIOp, arith::ExtUIOp>(op)) {
      return failure();
    }
    if (op->getNumRegions() != 0) {
      return failure();
    }
    FailureOr<Operation *> legalized = legalizeOp(op, op->getOperands(), typeConverter, rewriter, enableI64ToI32);
    if (failed(legalized)) {
      return failure();
    }

    rewriter.replaceOp(op, (*legalized));
    return success();
  }

 private:
  TypeConverter typeConverter;
  bool enableI64ToI32;
};

class LegalizeTypeForAscend : public impl::LegalizeTypeForAscendBase<LegalizeTypeForAscend> {
 public:
  explicit LegalizeTypeForAscend(bool enableI64ToI32_ = false) { enableI64ToI32 = enableI64ToI32_; }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *context = &getContext();

    // Pre-check: reject unsupported CmpIOp before pattern matching
    bool hasUnsupportedOp = false;
    func.walk([&](arith::CmpIOp cmpOp) {
      auto inTy = cmpOp.getLhs().getType();
      auto pre = cmpOp.getPredicate();
      if (pre != arith::CmpIPredicate::eq && pre != arith::CmpIPredicate::ne) {
        if (auto intTy = dyn_cast<IntegerType>(inTy)) {
          if (intTy.getWidth() >= kI32BitWidth) {
            emitError(cmpOp.getLoc()) << "arith::CmpIOp not support i32 or i64 integer in ascend.";
            hasUnsupportedOp = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
    if (hasUnsupportedOp) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(context);

    TypeConverter typeConverter;
    typeConverter.addConversion([](Type type) -> std::optional<Type> { return type; });
    typeConverter.addConversion([](FloatType type) -> std::optional<Type> {
      if (isa<BFloat16Type, Float64Type>(type)) {
        return Float32Type::get(type.getContext());
      }
      return std::nullopt;
    });
    if (enableI64ToI32) {
      typeConverter.addConversion([](IntegerType type) -> std::optional<Type> {
        if (type.getWidth() == kI64BitWidth) {
          return IntegerType::get(type.getContext(), kI32BitWidth);
        }
        return std::nullopt;
      });
    }

    // Add patterns for affine operations
    patterns.add<AffineLoadBF16ToF32Pattern>(context);
    patterns.add<AffineStoreWideFPToBF16Pattern>(context);
    patterns.add<AffineForFPToBF16Pattern>(context);
    patterns.add<ExtFOpPattern>(context);
    patterns.add<ExtSIOpPattern>(context);
    patterns.add<TruncFOpPattern>(context);
    patterns.add<BitcastOpPattern>(context);
    patterns.add<LegalizeBF16RewritePattern>(context, typeConverter, enableI64ToI32);
    patterns.add<AffineIfAlignResultTypesWithYieldPattern>(context);
    if (enableI64ToI32) {
      patterns.add<AffineLoadI64ToI32Pattern>(context);
      patterns.add<AffineStoreI64ToI32Pattern>(context);
      patterns.add<AffineForI64ToI32Pattern>(context);
      patterns.add<TruncIOpPattern>(context);
    }

    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;

    // Apply patterns using greedy pattern rewrite driver
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTypeForAscendPass(bool enableI64ToI32) {
  return std::make_unique<LegalizeTypeForAscend>(enableI64ToI32);
}
}  // namespace mlir
