/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//===----------------------------------------------------------------------===//
// NPUVectorVectorize.cpp
//
// SCF loop vectorization pass using NPUVector Dialect with dynamic shape support
//
// Architecture:
//   1. Read attributes (getVectorizationMode)
//   2. Decide vector factor (decideVectorFactor)
//   3. Create vectorized loop (createVectorizedLoop)
//   4. Vectorize loop body
//      ├── vectorizeLoad → npuvector.transfer_read
//      ├── vectorizeStore → npuvector.transfer_write
//      ├── vectorizeArithOp → type promotion (generic)
//      ├── vectorizeConstant → arith.constant dense
//      └── vectorizeUniform → npuvector.broadcast
//   5. Post-processing
//      ├── Elementwise: create scf.yield + delete scalar loop
//      └── Reduction: create scf.yield + npuvector.reduction + delete scalar loop
//===----------------------------------------------------------------------===//

#include "akg/Dialect/SCF/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Dialect/NPUVector/IR/NPUVector.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "npuvector-vectorize"

namespace mlir {
namespace scf {
#define GEN_PASS_DECL_NPUVECTORVECTORIZEPASS
#define GEN_PASS_DEF_NPUVECTORVECTORIZEPASS
#include "akg/Dialect/SCF/Passes.h.inc"
}  // namespace scf
}  // namespace mlir

using namespace mlir;  // NOLINT(build/namespaces)

namespace {

enum class VectorizationMode {
  None,
  Elementwise,
  Reduction,
};

struct VectorizationContext {
  OpBuilder &builder;
  IRMapping valueMapping;
  int64_t vectorFactor;
  Value vectorSizeValue;
  Value maxStepValue;
  VectorizationMode mode;
  bool isDynamic;

  scf::ForOp scalarLoop;
  scf::ForOp vecLoop;

  std::optional<arith::AtomicRMWKind> reductionKind;
  Value origInit;

  VectorizationContext(OpBuilder &b, int64_t vf, Value vfVal, Value maxVal,
                      VectorizationMode m, scf::ForOp loop, bool dynamic)
      : builder(b), vectorFactor(vf), vectorSizeValue(vfVal), maxStepValue(maxVal),
        mode(m), isDynamic(dynamic), scalarLoop(loop), vecLoop(nullptr),
        reductionKind(std::nullopt), origInit(nullptr) {}
};

static Value vectorizeUniform(Value uniformVal, VectorizationContext &ctx);

static VectorizationMode getVectorizationMode(scf::ForOp loop, int64_t &maxStepFromAttr) {
  if (auto reductionAttr = loop->getAttrOfType<IntegerAttr>(kReductionLoopAttr)) {
    maxStepFromAttr = reductionAttr.getInt();
    return VectorizationMode::Reduction;
  }

  if (auto vectorAttr = loop->getAttrOfType<IntegerAttr>(kVectorAttr)) {
    maxStepFromAttr = vectorAttr.getInt();
    return VectorizationMode::Elementwise;
  }

  return VectorizationMode::None;
}

// Returns: {VF, vectorSizeValue, skip, isDynamic}
static std::tuple<int64_t, Value, bool, bool> decideVectorFactor(
    scf::ForOp loop,
    int64_t defaultVF) {

  auto stepConst = loop.getStep().getDefiningOp<arith::ConstantIndexOp>();
  if (!stepConst || stepConst.value() != 1) {
    return {0, Value(), true, false};
  }

  auto ubConst = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto lbConst = loop.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();

  if (ubConst && lbConst) {
    int64_t tripCount = ubConst.value() - lbConst.value();
    return {tripCount, Value(), false, false};
  } else {
    return {defaultVF, loop.getUpperBound(), false, true};
  }
}

static std::optional<arith::AtomicRMWKind> detectReductionKind(scf::ForOp loop) {
  if (loop.getInitArgs().empty()) {
    return std::nullopt;
  }

  auto yieldOp = dyn_cast<scf::YieldOp>(loop.getBody()->getTerminator());
  if (!yieldOp || yieldOp.getNumOperands() == 0) {
    return std::nullopt;
  }

  Value yieldValue = yieldOp.getOperand(0);
  Operation *defOp = yieldValue.getDefiningOp();

  if (!defOp) {
    return std::nullopt;
  }

  Value iterArg = loop.getRegionIterArgs()[0];

  if (auto addOp = dyn_cast<arith::AddFOp>(defOp)) {
    if (addOp.getLhs() == iterArg || addOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::addf;
    }
  }

  if (auto mulOp = dyn_cast<arith::MulFOp>(defOp)) {
    if (mulOp.getLhs() == iterArg || mulOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::mulf;
    }
  }

  if (auto maxOp = dyn_cast<arith::MaximumFOp>(defOp)) {
    if (maxOp.getLhs() == iterArg || maxOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::maximumf;
    }
  }

  if (auto minOp = dyn_cast<arith::MinimumFOp>(defOp)) {
    if (minOp.getLhs() == iterArg || minOp.getRhs() == iterArg) {
      return arith::AtomicRMWKind::minimumf;
    }
  }

  return std::nullopt;
}

static Value createNeutralValue(
    arith::AtomicRMWKind kind,
    Type elemType,
    VectorizationContext &ctx,
    Location loc) {

  Attribute neutralAttr = arith::getIdentityValueAttr(kind, elemType, ctx.builder, loc);

  if (ctx.isDynamic) {
    Value neutralScalar = ctx.builder.create<arith::ConstantOp>(
        loc, mlir::cast<TypedAttr>(neutralAttr));

    npuvector::NPUVectorType vecType =
        npuvector::NPUVectorType::get({ShapedType::kDynamic}, elemType);

    SmallVector<Value> dynamicSizes;
    if (ctx.vectorSizeValue) {
      dynamicSizes.push_back(ctx.vectorSizeValue);
    }

    Value neutralVec = ctx.builder.create<npuvector::BroadcastOp>(
        loc, vecType, neutralScalar, ValueRange(dynamicSizes), ctx.maxStepValue);

    return neutralVec;

  } else {
    npuvector::NPUVectorType vecType =
        npuvector::NPUVectorType::get({ctx.vectorFactor}, elemType);

    auto vecAttr = DenseElementsAttr::get(vecType, neutralAttr);
    Value neutralVec = ctx.builder.create<arith::ConstantOp>(loc, vecAttr);

    return neutralVec;
  }
}

static bool isNeutralElement(arith::AtomicRMWKind kind, Value value,
                               OpBuilder &builder) {
  auto constOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }

  Attribute neutralAttr = arith::getIdentityValueAttr(
      kind, value.getType(), builder, value.getLoc());

  bool isNeutral = (constOp.getValue() == neutralAttr);

  return isNeutral;
}

static vector::CombiningKind convertToCombiningKind(arith::AtomicRMWKind kind) {
  switch (kind) {
    case arith::AtomicRMWKind::addf:
    case arith::AtomicRMWKind::addi:
      return vector::CombiningKind::ADD;

    case arith::AtomicRMWKind::mulf:
    case arith::AtomicRMWKind::muli:
      return vector::CombiningKind::MUL;

    case arith::AtomicRMWKind::maximumf:
      return vector::CombiningKind::MAXIMUMF;

    case arith::AtomicRMWKind::minimumf:
      return vector::CombiningKind::MINIMUMF;

    case arith::AtomicRMWKind::maxs:
      return vector::CombiningKind::MAXSI;

    case arith::AtomicRMWKind::mins:
      return vector::CombiningKind::MINSI;

    case arith::AtomicRMWKind::maxu:
      return vector::CombiningKind::MAXUI;

    case arith::AtomicRMWKind::minu:
      return vector::CombiningKind::MINUI;

    default:
      llvm_unreachable("Unsupported reduction kind");
  }
}

static scf::ForOp createVectorizedLoop(VectorizationContext &ctx) {
  Location loc = ctx.scalarLoop.getLoc();

  ctx.builder.setInsertionPoint(ctx.scalarLoop);

  Value newStepValue;

  if (ctx.isDynamic) {
    newStepValue = ctx.vectorSizeValue;
  } else {
    newStepValue = ctx.builder.create<arith::ConstantIndexOp>(loc, ctx.vectorFactor);
  }

  if (ctx.mode == VectorizationMode::Elementwise) {
    auto vecLoop = ctx.builder.create<scf::ForOp>(
        loc,
        ctx.scalarLoop.getLowerBound(),
        ctx.scalarLoop.getUpperBound(),
        newStepValue,
        std::nullopt,
        [](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          // Empty body builder
        });

    return vecLoop;

  } else {
    auto kind = detectReductionKind(ctx.scalarLoop);
    if (!kind) {
      return nullptr;
    }

    Type elemType = ctx.scalarLoop.getRegionIterArgs()[0].getType();

    Value neutralVec = createNeutralValue(*kind, elemType, ctx, loc);
    if (!neutralVec) {
      return nullptr;
    }

    ctx.reductionKind = kind;
    ctx.origInit = ctx.scalarLoop.getInitArgs()[0];

    auto vecLoop = ctx.builder.create<scf::ForOp>(
        loc,
        ctx.scalarLoop.getLowerBound(),
        ctx.scalarLoop.getUpperBound(),
        newStepValue,
        ValueRange{neutralVec},
        [](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          // Empty body builder
        });

    return vecLoop;
  }
}

static Value vectorizeLoad(memref::LoadOp loadOp, VectorizationContext &ctx) {
  MemRefType memRefType = loadOp.getMemRefType();
  Type elemType = memRefType.getElementType();
  Location loc = loadOp.getLoc();

  bool allIndicesLoopInvariant = true;
  for (Value idx : loadOp.getIndices()) {
    if (idx == ctx.scalarLoop.getInductionVar()) {
      allIndicesLoopInvariant = false;
      break;
    }

    if (idx.getParentBlock() == ctx.scalarLoop.getBody()) {
      allIndicesLoopInvariant = false;
      break;
    }
  }

  if (allIndicesLoopInvariant) {
    SmallVector<Value> indices;
    for (Value idx : loadOp.getIndices()) {
      Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
      indices.push_back(newIdx);
    }

    Value scalarLoad = ctx.builder.create<memref::LoadOp>(
        loc, loadOp.getMemRef(), indices);

    return vectorizeUniform(scalarLoad, ctx);
  }

  mlir::npuvector::NPUVectorType vecType;
  if (ctx.isDynamic) {
    vecType = mlir::npuvector::NPUVectorType::get({ShapedType::kDynamic}, elemType);
  } else {
    vecType = mlir::npuvector::NPUVectorType::get({ctx.vectorFactor}, elemType);
  }

  SmallVector<Value> indices;
  for (Value idx : loadOp.getIndices()) {
    Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
    indices.push_back(newIdx);
  }

  Value padding = ctx.builder.create<arith::ConstantOp>(
      loc,
      ctx.builder.getZeroAttr(elemType));

  SmallVector<Value> dynamicSizes;
  if (ctx.isDynamic && ctx.vectorSizeValue) {
    dynamicSizes.push_back(ctx.vectorSizeValue);
  }

  auto transferRead = ctx.builder.create<npuvector::TransferReadOp>(
      loc,
      vecType,
      loadOp.getMemRef(),
      ValueRange(indices),
      padding,
      Value(),
      ValueRange(dynamicSizes),
      ctx.maxStepValue);

  return transferRead.getResult();
}

static void vectorizeStore(memref::StoreOp storeOp, VectorizationContext &ctx) {
  Location loc = storeOp.getLoc();

  Value storeValue = storeOp.getValue();
  Value vectorValue = ctx.valueMapping.lookupOrNull(storeValue);
  if (!vectorValue) {
    OpBuilder::InsertionGuard guard(ctx.builder);
    if (ctx.vecLoop) {
      ctx.builder.setInsertionPoint(ctx.vecLoop);
    }

    vectorValue = vectorizeUniform(storeValue, ctx);
    if (!vectorValue) {
      return;
    }
    ctx.valueMapping.map(storeValue, vectorValue);
  }

  SmallVector<Value> indices;
  for (Value idx : storeOp.getIndices()) {
    Value newIdx = ctx.valueMapping.lookupOrDefault(idx);
    indices.push_back(newIdx);
  }

  auto npuVecType = mlir::dyn_cast<npuvector::NPUVectorType>(vectorValue.getType());
  if (!npuVecType) {
    llvm_unreachable("vectorizeStore: vector value must be NPUVectorType");
    return;
  }

  ctx.builder.create<npuvector::TransferWriteOp>(
      loc,
      Type(),
      vectorValue,
      storeOp.getMemRef(),
      ValueRange(indices),
      Value());
}

static Value vectorizeTypeConversion(Operation *op, VectorizationContext &ctx) {
  Location loc = op->getLoc();

  SmallVector<Value, 4> vecOperands;
  for (Value operand : op->getOperands()) {
    Value vecOperand = ctx.valueMapping.lookupOrNull(operand);
    if (!vecOperand) {
      if (operand.getParentBlock() != ctx.scalarLoop.getBody()) {
        OpBuilder::InsertionGuard guard(ctx.builder);
        if (ctx.vecLoop) {
          ctx.builder.setInsertionPoint(ctx.vecLoop);
        }
        vecOperand = vectorizeUniform(operand, ctx);
        if (!vecOperand) {
          return nullptr;
        }
        ctx.valueMapping.map(operand, vecOperand);
      } else {
        return nullptr;
      }
    }
    vecOperands.push_back(vecOperand);
  }

  SmallVector<Type, 4> vecResultTypes;
  for (Value result : op->getResults()) {
    Type scalarType = result.getType();
    npuvector::NPUVectorType vecType;
    if (ctx.isDynamic) {
      vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, scalarType);
    } else {
      vecType = npuvector::NPUVectorType::get({ctx.vectorFactor}, scalarType);
    }
    vecResultTypes.push_back(vecType);
  }

  StringRef arithOpName = op->getName().getStringRef();
  std::string npuvectorOpName = "npuvector." + arithOpName.split('.').second.str();
  OperationName npuvectorName(npuvectorOpName, ctx.builder.getContext());

  OperationState state(loc, npuvectorName);
  state.addOperands(vecOperands);
  state.addTypes(vecResultTypes);
  state.addAttributes(op->getAttrs());

  Operation *npuvectorOp = ctx.builder.create(state);

  if (!npuvectorOp || npuvectorOp->getNumResults() == 0) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to create npuvector operation: "
                            << npuvectorOpName << "\n");
    return nullptr;
  }

  return npuvectorOp->getResult(0);
}

static Value vectorizeArithOp(Operation *op, VectorizationContext &ctx) {
  if (isa<arith::ExtFOp, arith::TruncFOp,
          arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::SIToFPOp, arith::UIToFPOp,
          arith::FPToSIOp, arith::FPToUIOp,
          arith::BitcastOp, arith::IndexCastOp, arith::IndexCastUIOp,
          arith::CmpIOp, arith::CmpFOp, arith::SelectOp>(op)) {
    return vectorizeTypeConversion(op, ctx);
  }

  SmallVector<Value, 4> vectorOperands;
  for (Value operand : op->getOperands()) {
    Value vecOperand = ctx.valueMapping.lookupOrNull(operand);

    if (!vecOperand || !mlir::isa<npuvector::NPUVectorType>(vecOperand.getType())) {
      Value scalarOperand = vecOperand ? vecOperand : operand;

      if (scalarOperand.getParentBlock() != ctx.scalarLoop.getBody()) {
        OpBuilder::InsertionGuard guard(ctx.builder);

        if (ctx.vecLoop) {
          ctx.builder.setInsertionPoint(ctx.vecLoop);
        }

        vecOperand = vectorizeUniform(scalarOperand, ctx);
        if (!vecOperand) {
          return nullptr;
        }
        ctx.valueMapping.map(operand, vecOperand);

      } else {
        return nullptr;
      }
    }

    if (!mlir::isa<npuvector::NPUVectorType>(vecOperand.getType())) {
      llvm_unreachable("vectorizeArithOp: all operands must be NPUVectorType");
    }

    vectorOperands.push_back(vecOperand);
  }

  SmallVector<Type, 4> vectorTypes;
  for (Value result : op->getResults()) {
    Type scalarType = result.getType();
    npuvector::NPUVectorType vecType;
    if (ctx.isDynamic) {
      vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, scalarType);
    } else {
      vecType = npuvector::NPUVectorType::get({ctx.vectorFactor}, scalarType);
    }
    vectorTypes.push_back(vecType);
  }

  Operation *vecOp = ctx.builder.create(
      op->getLoc(),
      op->getName().getIdentifier(),
      vectorOperands,
      vectorTypes,
      op->getAttrs());

  return vecOp->getResult(0);
}

static Value vectorizeConstant(arith::ConstantOp constOp, VectorizationContext &ctx) {
  Type scalarType = constOp.getType();
  Location loc = constOp.getLoc();

  npuvector::NPUVectorType vecType;

  if (ctx.isDynamic) {
    if (!ctx.vectorSizeValue) {
      return nullptr;
    }

    vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, scalarType);

    SmallVector<Value> dynamicSizes;
    dynamicSizes.push_back(ctx.vectorSizeValue);

    Value scalarValue = ctx.builder.create<arith::ConstantOp>(
        loc, constOp.getValue());

    auto broadcast = ctx.builder.create<npuvector::BroadcastOp>(
        loc, vecType,
        scalarValue,
        ValueRange(dynamicSizes),
        ctx.maxStepValue);

    return broadcast.getResult();

  } else {
    vecType = npuvector::NPUVectorType::get({ctx.vectorFactor}, scalarType);

    auto vecAttr = DenseElementsAttr::get(vecType, constOp.getValue());
    auto vecConst = ctx.builder.create<arith::ConstantOp>(loc, vecAttr);

    return vecConst.getResult();
  }
}

static Value vectorizeUniform(Value uniformVal, VectorizationContext &ctx) {
  Type scalarType = uniformVal.getType();
  Location loc = uniformVal.getLoc();

  npuvector::NPUVectorType vecType;
  if (ctx.isDynamic) {
    vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, scalarType);
  } else {
    vecType = npuvector::NPUVectorType::get({ctx.vectorFactor}, scalarType);
  }

  SmallVector<Value> dynamicSizes;
  if (ctx.isDynamic && ctx.vectorSizeValue) {
    dynamicSizes.push_back(ctx.vectorSizeValue);
  }

  auto broadcast = ctx.builder.create<npuvector::BroadcastOp>(
      loc,
      vecType,
      uniformVal,
      ValueRange(dynamicSizes),
      ctx.maxStepValue);

  return broadcast.getResult();
}

static void cloneScalarOp(Operation &op, VectorizationContext &ctx) {
  OpBuilder::InsertionGuard guard(ctx.builder);

  if (ctx.vecLoop) {
    ctx.builder.setInsertionPoint(ctx.vecLoop);
  } else {
    ctx.builder.setInsertionPoint(ctx.scalarLoop);
  }

  Operation *clonedOp = ctx.builder.clone(op);

  for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
    Value mappedValue = ctx.valueMapping.lookupOrDefault(operand);
    clonedOp->setOperand(idx, mappedValue);
  }

  if (op.getNumResults() == 1) {
    ctx.valueMapping.map(op.getResult(0), clonedOp->getResult(0));
  }
}

static Value vectorizeIf(scf::IfOp ifOp, VectorizationContext &ctx);

static LogicalResult handleConstantOp(arith::ConstantOp constOp, VectorizationContext &ctx) {
  Type constType = constOp.getType();
  if (constType == ctx.builder.getIndexType()) {
    Value scalarConst = ctx.builder.create<arith::ConstantOp>(
        constOp.getLoc(), constOp.getValue());
    ctx.valueMapping.map(constOp.getResult(), scalarConst);
    return success();
  }

  Value vecValue = vectorizeConstant(constOp, ctx);
  if (!vecValue) {
    return failure();
  }
  ctx.valueMapping.map(constOp.getResult(), vecValue);
  return success();
}

static bool shouldCloneScalarOp(Operation &op, VectorizationContext &ctx) {
  if (isa<affine::AffineApplyOp, affine::AffineMaxOp, affine::AffineMinOp>(&op)) {
    return true;
  }

  if (op.getNumResults() == 1 &&
      op.getResult(0).getType() == ctx.builder.getIndexType()) {
    return true;
  }

  if (llvm::any_of(op.getOperands(), [&](Value operand) {
        return operand.getType() == ctx.builder.getIndexType();
      })) {
    return true;
  }

  return false;
}

static LogicalResult handleArithOrMathOpInRegion(Operation &op, VectorizationContext &ctx) {
  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName != "arith" && dialectName != "math") {
    return success();
  }

  if (op.getNumRegions() != 0) {
    return failure();
  }
  if (op.getNumResults() == 0) {
    return success();
  }

  Value vecValue = vectorizeArithOp(&op, ctx);
  if (!vecValue) {
    return failure();
  }

  if (op.getNumResults() == 1) {
    ctx.valueMapping.map(op.getResult(0), vecValue);
  }

  return success();
}

static LogicalResult handleArithOrMathOpInLoopBody(Operation &op, VectorizationContext &ctx) {
  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName != "arith" && dialectName != "math") {
    return success();
  }

  if (op.getNumRegions() != 0) {
    return failure();
  }
  if (op.getNumResults() == 0) {
    return success();
  }

  Value vecValue = vectorizeArithOp(&op, ctx);
  if (!vecValue) {
    return failure();
  }

  if (op.getNumResults() == 1) {
    ctx.valueMapping.map(op.getResult(0), vecValue);
  } else {
    ctx.valueMapping.map(op.getResult(0), vecValue);
  }

  return success();
}

static LogicalResult handleIfOpInRegion(scf::IfOp ifOp, VectorizationContext &ctx) {
  Value vecIfResult = vectorizeIf(ifOp, ctx);
  if (!vecIfResult) {
    return failure();
  }

  if (ifOp.getNumResults() == 1) {
    ctx.valueMapping.map(ifOp.getResult(0), vecIfResult);
  } else if (ifOp.getNumResults() > 1) {
    // Multiple return values: currently handled in a simplified way by mapping only the first result.
    ctx.valueMapping.map(ifOp.getResult(0), vecIfResult);
  }

  return success();
}

static LogicalResult handleIfOpInLoopBody(scf::IfOp ifOp, VectorizationContext &ctx) {
  Value vecIfResult = vectorizeIf(ifOp, ctx);
  if (!vecIfResult && ifOp.getNumResults() > 0) {
    return failure();
  }

  // The result needs to be mapped only when scf.if has return values.
  if (ifOp.getNumResults() > 0) {
    ctx.valueMapping.map(ifOp.getResult(0), vecIfResult);
  }

  return success();
}

static LogicalResult vectorizeOneOpInRegion(Operation &op, VectorizationContext &ctx) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
    Value vecValue = vectorizeLoad(loadOp, ctx);
    if (!vecValue) {
      return failure();
    }
    ctx.valueMapping.map(loadOp.getResult(), vecValue);
    return success();
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
    vectorizeStore(storeOp, ctx);
    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
    return handleConstantOp(constOp, ctx);
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
    return handleIfOpInRegion(ifOp, ctx);
  }

  if (shouldCloneScalarOp(op, ctx)) {
    cloneScalarOp(op, ctx);
    return success();
  }

  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName == "arith" || dialectName == "math") {
    return handleArithOrMathOpInRegion(op, ctx);
  }

  cloneScalarOp(op, ctx);
  return success();
}

static LogicalResult vectorizeOneOpInLoopBody(Operation &op, VectorizationContext &ctx) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
    Value vecValue = vectorizeLoad(loadOp, ctx);
    if (!vecValue) {
      return failure();
    }
    ctx.valueMapping.map(loadOp.getResult(), vecValue);
    return success();
  }

  if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
    vectorizeStore(storeOp, ctx);
    return success();
  }

  if (auto constOp = dyn_cast<arith::ConstantOp>(&op)) {
    return handleConstantOp(constOp, ctx);
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
    return handleIfOpInLoopBody(ifOp, ctx);
  }

  if (shouldCloneScalarOp(op, ctx)) {
    cloneScalarOp(op, ctx);
    return success();
  }

  StringRef dialectName = op.getDialect()->getNamespace();
  if (dialectName != "arith" && dialectName != "math") {
    return success();
  }

  return handleArithOrMathOpInLoopBody(op, ctx);
}

static LogicalResult vectorizeRegion(Region &region, VectorizationContext &ctx) {
  Block *block = &region.front();

  for (Operation &op : block->without_terminator()) {
    if (failed(vectorizeOneOpInRegion(op, ctx))) {
      return failure();
    }
  }

  Operation *originalTerminator = block->getTerminator();
  if (auto originalYieldOp = dyn_cast<scf::YieldOp>(originalTerminator)) {
    SmallVector<Value> vecYieldOperands;

    for (Value operand : originalYieldOp.getOperands()) {
      Value vecOperand = ctx.valueMapping.lookupOrNull(operand);

      if (!vecOperand) {
        if (operand.getParentBlock() != ctx.scalarLoop.getBody() ||
            operand.getDefiningOp() == nullptr) {
          vecOperand = vectorizeUniform(operand, ctx);
          if (!vecOperand) {
            return failure();
          }
          ctx.valueMapping.map(operand, vecOperand);
        } else {
          return failure();
        }
      }
      vecYieldOperands.push_back(vecOperand);
    }

    Block *currentBlock = ctx.builder.getInsertionBlock();
    if (!currentBlock) {
      return failure();
    }

    Operation *autoTerminator = currentBlock->getTerminator();
    if (auto autoYieldOp = dyn_cast<scf::YieldOp>(autoTerminator)) {
      ctx.builder.setInsertionPoint(autoYieldOp);
      ctx.builder.create<scf::YieldOp>(originalYieldOp.getLoc(), vecYieldOperands);
      autoYieldOp.erase();
    } else {
      ctx.builder.setInsertionPointToEnd(currentBlock);
      ctx.builder.create<scf::YieldOp>(originalYieldOp.getLoc(), vecYieldOperands);
    }
  }

  return success();
}

static Value vectorizeIf(scf::IfOp ifOp, VectorizationContext &ctx) {
  Location loc = ifOp.getLoc();
  Value condition = ifOp.getCondition();
  Value vecCondition = ctx.valueMapping.lookupOrNull(condition);
  if (!vecCondition) {
    vecCondition = condition;
  }

  SmallVector<Type> vecResultTypes;
  if (ifOp.getNumResults() > 0) {
    for (Type resultType : ifOp.getResultTypes()) {
      if (mlir::isa<npuvector::NPUVectorType>(resultType)) {
        vecResultTypes.push_back(resultType);
      } else {
        npuvector::NPUVectorType vecType;
        if (ctx.isDynamic) {
          vecType = npuvector::NPUVectorType::get({ShapedType::kDynamic}, resultType);
        } else {
          vecType = npuvector::NPUVectorType::get({ctx.vectorFactor}, resultType);
        }
        vecResultTypes.push_back(vecType);
      }
    }
  }

  // Currently, only support scf.if without an else branch, and therefore do not support scf.if with return values.
  bool hasElse = !ifOp.getElseRegion().empty();
  if (hasElse) {
    return nullptr;
  }

  if (ifOp.getNumResults() > 0) {
    return nullptr;
  }

  auto vecIfOp = ctx.builder.create<scf::IfOp>(loc, vecResultTypes, vecCondition, false);

  {
    OpBuilder::InsertionGuard guard(ctx.builder);
    ctx.builder.setInsertionPointToStart(vecIfOp.thenBlock());

    if (failed(vectorizeRegion(ifOp.getThenRegion(), ctx))) {
      vecIfOp.erase();
      return nullptr;
    }
  }

  if (vecIfOp.getNumResults() == 1) {
    return vecIfOp.getResult(0);
  } else if (vecIfOp.getNumResults() > 1) {
    // Multiple return values: currently handled in a simplified way by mapping only the first result.
    return vecIfOp.getResult(0);
  }

  return Value();
}

static LogicalResult vectorizeLoopBody(VectorizationContext &ctx) {
  if (ctx.vecLoop) {
    ctx.builder.setInsertionPointToStart(ctx.vecLoop.getBody());
  }

  if (ctx.vecLoop) {
    ctx.valueMapping.map(ctx.scalarLoop.getInductionVar(),
                         ctx.vecLoop.getInductionVar());
  }

  if (ctx.mode == VectorizationMode::Reduction) {
    for (auto [scalarArg, vecArg] : llvm::zip(
            ctx.scalarLoop.getRegionIterArgs(),
            ctx.vecLoop.getRegionIterArgs())) {
      ctx.valueMapping.map(scalarArg, vecArg);
    }
  }

  for (Operation &op : ctx.scalarLoop.getBody()->without_terminator()) {
    if (failed(vectorizeOneOpInLoopBody(op, ctx))) {
      return failure();
    }
  }

  return success();
}

static LogicalResult vectorizeElementwiseInline(VectorizationContext &ctx) {
  ctx.builder.setInsertionPoint(ctx.scalarLoop);

  ctx.valueMapping.map(ctx.scalarLoop.getInductionVar(),
                       ctx.scalarLoop.getLowerBound());

  if (failed(vectorizeLoopBody(ctx))) {
    return failure();
  }

  ctx.scalarLoop.erase();

  return success();
}

static void finalizeElementwise(VectorizationContext &ctx) {
  ctx.builder.setInsertionPointToEnd(ctx.vecLoop.getBody());
  ctx.builder.create<scf::YieldOp>(ctx.vecLoop.getLoc());

  ctx.scalarLoop.erase();
}

static void finalizeReduction(VectorizationContext &ctx) {
  Location loc = ctx.vecLoop.getLoc();

  ctx.builder.setInsertionPointToEnd(ctx.vecLoop.getBody());

  auto scalarYield = cast<scf::YieldOp>(ctx.scalarLoop.getBody()->getTerminator());
  Value scalarYieldValue = scalarYield.getOperand(0);
  Value vecYieldValue = ctx.valueMapping.lookupOrNull(scalarYieldValue);

  if (!vecYieldValue) {
    return;
  }

  ctx.builder.create<scf::YieldOp>(loc, vecYieldValue);

  ctx.builder.setInsertionPointAfter(ctx.vecLoop);

  vector::CombiningKind combiningKind = convertToCombiningKind(*ctx.reductionKind);

  Value vectorAcc = ctx.vecLoop.getResult(0);

  auto npuVectorType = mlir::cast<npuvector::NPUVectorType>(vectorAcc.getType());
  Type elemType = npuVectorType.getElementType();

  auto reductionOp = ctx.builder.create<npuvector::ReductionOp>(
      loc,
      elemType,
      combiningKind,
      vectorAcc,
      Value(),
      arith::FastMathFlags::none);

  Value reduced = reductionOp.getDest();

  Value finalResult = reduced;

  if (!isNeutralElement(*ctx.reductionKind, ctx.origInit, ctx.builder)) {
    switch (*ctx.reductionKind) {
      case arith::AtomicRMWKind::addf:
        finalResult = ctx.builder.create<arith::AddFOp>(loc, reduced, ctx.origInit);
        break;
      case arith::AtomicRMWKind::mulf:
        finalResult = ctx.builder.create<arith::MulFOp>(loc, reduced, ctx.origInit);
        break;
      case arith::AtomicRMWKind::maximumf:
        finalResult = ctx.builder.create<arith::MaximumFOp>(loc, reduced, ctx.origInit);
        break;
      case arith::AtomicRMWKind::minimumf:
        finalResult = ctx.builder.create<arith::MinimumFOp>(loc, reduced, ctx.origInit);
        break;
      default:
        break;
    }
  }

  Value scalarResult = ctx.scalarLoop.getResult(0);

  if (!scalarResult.use_empty()) {
    scalarResult.replaceAllUsesWith(finalResult);
  }

  ctx.scalarLoop.erase();
}

static LogicalResult vectorizeLoop(
    scf::ForOp scalarLoop,
    int64_t vf,
    Value vfValue,
    Value maxStepValue,
    bool isDynamic,
    VectorizationMode mode,
    OpBuilder &builder) {

  VectorizationContext ctx(builder, vf, vfValue, maxStepValue, mode, scalarLoop, isDynamic);

  if (ctx.mode == VectorizationMode::Elementwise) {
    return vectorizeElementwiseInline(ctx);

  } else {
    ctx.vecLoop = createVectorizedLoop(ctx);
    if (!ctx.vecLoop) {
      return failure();
    }

    if (failed(vectorizeLoopBody(ctx))) {
      ctx.vecLoop.erase();
      return failure();
    }

    finalizeReduction(ctx);
    return success();
  }
}

class NPUVectorVectorizePass
    : public mlir::scf::impl::NPUVectorVectorizePassBase<NPUVectorVectorizePass> {
 public:
  NPUVectorVectorizePass() = default;
  NPUVectorVectorizePass(const NPUVectorVectorizePass &) = default;

  StringRef getArgument() const override { return "npuvector-vectorize"; }

  StringRef getDescription() const override {
    return "SCF loop vectorization using NPUVector with dynamic shape support";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<scf::SCFDialect,
                    npuvector::NPUVectorDialect,
                    memref::MemRefDialect,
                    func::FuncDialect,
                    arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    OpBuilder builder(&getContext());

    SmallVector<scf::ForOp> candidateLoops;
    funcOp.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr(kVectorAttr) || forOp->hasAttr(kReductionLoopAttr)) {
        candidateLoops.push_back(forOp);
      }
    });

    for (scf::ForOp loop : candidateLoops) {
      int64_t maxStepFromAttr = -1;
      VectorizationMode mode = getVectorizationMode(loop, maxStepFromAttr);

      if (mode == VectorizationMode::None) {
        continue;
      }

      auto [vf, vfValue, skip, isDynamic] = decideVectorFactor(loop, defaultVF);
      if (skip || vf <= 0) {
        continue;
      }

      if (isDynamic) {
        OpBuilder sizeBuilder(&getContext());
        sizeBuilder.setInsertionPoint(loop);
        vfValue = sizeBuilder.create<arith::SubIOp>(
            loop.getLoc(), loop.getUpperBound(), loop.getLowerBound());
      }

      Value maxStepValue;
      if (maxStepFromAttr > 0 && isDynamic) {
        OpBuilder attrBuilder(&getContext());
        attrBuilder.setInsertionPoint(loop);
        maxStepValue = attrBuilder.create<arith::ConstantIndexOp>(
            loop.getLoc(), maxStepFromAttr);
      }

      (void)vectorizeLoop(loop, vf, vfValue, maxStepValue, isDynamic, mode, builder);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::scf::createNPUVectorVectorizePass() {
  return std::make_unique<NPUVectorVectorizePass>();
}
