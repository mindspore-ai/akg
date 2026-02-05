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

#include "akg/Transforms/AddOutParameter.h"
#include <algorithm>
#include <iterator>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

// HACC dialect enums/attrs
#include "bishengir/Dialect/HACC/IR/HACC.h"

#define DEBUG_TYPE "add-out-parameter"

namespace mlir {
#define GEN_PASS_DECL_ADDOUTPARAMETER
#define GEN_PASS_DEF_ADDOUTPARAMETER
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

namespace mlir {

using hacc::InputIdxAttr;
using hacc::KernelArgType;
using hacc::KernelArgTypeAttr;
using hacc::OutputIdxAttr;

namespace {

static void setHaccIOArgAttrs(func::FuncOp f, unsigned nInputs, unsigned nOutputs, OpBuilder &builder) {
  auto *ctx = builder.getContext();

  for (unsigned i = 0; i < nInputs; ++i) {
    f.setArgAttr(i, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kInput));
    f.setArgAttr(i, InputIdxAttr::name, InputIdxAttr::get(ctx, i));
  }

  for (unsigned i = 0; i < nOutputs; ++i) {
    unsigned argIdx = nInputs + i;
    f.setArgAttr(argIdx, KernelArgTypeAttr::name, KernelArgTypeAttr::get(ctx, KernelArgType::kOutput));
    f.setArgAttr(argIdx, OutputIdxAttr::name, OutputIdxAttr::get(ctx, i));
  }
}

static bool isAliasOfOriginalInput(Value v, func::FuncOp func, unsigned origNumInputs) {
  if (auto ba = mlir::dyn_cast<BlockArgument>(v)) {
    Block *owner = ba.getOwner();
    if (&func.front() != owner) return false;
    unsigned idx = ba.getArgNumber();
    return idx < origNumInputs;
  }
  return false;
}

static LogicalResult collectOriginalReturnValues(func::FuncOp func, unsigned origNumResults,
                                                 SmallVectorImpl<func::ReturnOp> &returns,
                                                 SmallVectorImpl<Value> &origReturnValues) {
  func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.empty()) return success();

  func::ReturnOp repr = returns.front();
  if (repr.getNumOperands() != origNumResults) {
    repr.emitError() << "number of return values (" << repr.getNumOperands()
                     << ") does not match number of function results (" << origNumResults << ")";
    return failure();
  }
  origReturnValues.append(repr.getOperands().begin(), repr.getOperands().end());
  return success();
}

static SmallVector<ReassociationIndices> parseReassociation(ArrayAttr reassocAttr) {
  SmallVector<ReassociationIndices> result;
  if (!reassocAttr) return result;

  result.reserve(reassocAttr.size());
  for (Attribute attr : reassocAttr) {
    auto arr = cast<ArrayAttr>(attr);
    ReassociationIndices inds;
    inds.reserve(arr.size());
    for (Attribute idxAttr : arr) {
      auto intAttr = cast<IntegerAttr>(idxAttr);
      inds.push_back(intAttr.getInt());
    }
    result.push_back(std::move(inds));
  }
  return result;
}

template <typename SrcOpTy, typename BackViewOpTy>
static LogicalResult rewriteSrcOutShapeOpToOutViewImpl(SrcOpTy shapeOp, Value outArg) {
  Value src = shapeOp.getSrc();
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto outTy = dyn_cast<MemRefType>(outArg.getType());
  if (!srcTy || !outTy) return success();

  auto func = shapeOp->template getParentOfType<func::FuncOp>();
  if (!func || func.empty()) return success();

  Block &entry = func.front();

  auto reassoc = parseReassociation(shapeOp.getReassociation());

  OpBuilder topBuilder(&entry, entry.begin());
  auto backViewOp = topBuilder.template create<BackViewOpTy>(shapeOp.getLoc(), srcTy, outArg, reassoc);
  Value backView = backViewOp.getResult();

  SmallVector<memref::StoreOp, 4> srcStores;
  for (Operation *user : src.getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemRef() == src) srcStores.push_back(store);
    }
  }

  for (auto store : srcStores) {
    OpBuilder b(store->getContext());
    b.setInsertionPointAfter(store);
    b.create<memref::StoreOp>(store.getLoc(), store.getValue(), backView, store.getIndices());
  }

  return success();
}

static LogicalResult rewriteSrcOutExpandOpToOutView(memref::ExpandShapeOp expandOp, Value outArg) {
  return rewriteSrcOutShapeOpToOutViewImpl<memref::ExpandShapeOp, memref::CollapseShapeOp>(expandOp, outArg);
}

static LogicalResult rewriteSrcOutCollapseToOutView(memref::CollapseShapeOp collapseOp, Value outArg) {
  return rewriteSrcOutShapeOpToOutViewImpl<memref::CollapseShapeOp, memref::ExpandShapeOp>(collapseOp, outArg);
}

template <typename SrcOpTy, typename NewViewOpTy>
static LogicalResult rewriteTempSourceToOutViewImpl(SrcOpTy srcOp, Value outArg) {
  Value tmp = srcOp.getSrc();
  auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
  auto outTy = dyn_cast<MemRefType>(outArg.getType());
  if (!tmpTy || !outTy) return success();

  auto reassoc = parseReassociation(srcOp.getReassociation());

  Operation *tmpDef = tmp.getDefiningOp();
  Value newViewVal;

  if (!tmpDef) {
    BlockArgument ba = mlir::cast<BlockArgument>(tmp);
    Block *parentBlock = ba.getOwner();
    if (!parentBlock) return success();

    auto insertPt = parentBlock->begin();
    OpBuilder b(parentBlock, insertPt);
    auto newView = b.template create<NewViewOpTy>(srcOp.getLoc(), tmpTy, outArg, reassoc);
    newViewVal = newView.getResult();
  } else {
    Block *parentBlock = tmpDef->getBlock();
    auto insertPt = std::next(Block::iterator(tmpDef));
    OpBuilder b(parentBlock, insertPt);
    auto newView = b.template create<NewViewOpTy>(srcOp.getLoc(), tmpTy, outArg, reassoc);
    newViewVal = newView.getResult();
  }

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : tmp.getUses()) {
    if (use.getOwner() == srcOp) continue;
    uses.push_back(&use);
  }
  for (auto *u : uses) u->set(newViewVal);

  srcOp.getResult().replaceAllUsesWith(outArg);
  srcOp.erase();

  if (auto tmpAlloc = tmp.getDefiningOp<memref::AllocOp>()) {
    if (tmpAlloc->use_empty()) tmpAlloc->erase();
  }

  return success();
}

static LogicalResult rewriteTempCollapseSourceToOutView(memref::CollapseShapeOp collapseOp, Value outArg) {
  return rewriteTempSourceToOutViewImpl<memref::CollapseShapeOp, memref::ExpandShapeOp>(collapseOp, outArg);
}

static LogicalResult rewriteTempExpandSourceToOutView(memref::ExpandShapeOp expandOp, Value outArg) {
  return rewriteTempSourceToOutViewImpl<memref::ExpandShapeOp, memref::CollapseShapeOp>(expandOp, outArg);
}

static Value buildReshapeFromOut(OpBuilder &b, Location loc, MemRefType tmpTy, Value outArg,
                                 ArrayRef<int64_t> staticDims) {
  MLIRContext *ctx = b.getContext();
  unsigned rank = tmpTy.getRank();

  auto idxTy = IndexType::get(ctx);
  auto shapeMemrefTy = MemRefType::get({static_cast<int64_t>(rank)}, idxTy);
  Value shapeMemref = b.create<memref::AllocOp>(loc, shapeMemrefTy).getResult();

  for (unsigned i = 0; i < rank; ++i) {
    Value dimVal;
    if (ShapedType::isDynamic(staticDims[i])) {
      dimVal = b.create<memref::DimOp>(loc, outArg, i);
    } else {
      dimVal = b.create<arith::ConstantIndexOp>(loc, staticDims[i]);
    }

    Value idx = b.create<arith::ConstantIndexOp>(loc, i);
    b.create<memref::StoreOp>(loc, dimVal, shapeMemref, ValueRange{idx});
  }

  auto newView = b.create<memref::ReshapeOp>(loc, tmpTy, outArg, shapeMemref);
  return newView.getResult();
}

static void tryEraseOldShapeMemref(Value tmpShape) {
  auto shapeAlloc = tmpShape.getDefiningOp<memref::AllocOp>();
  if (!shapeAlloc) return;

  SmallVector<Operation *> shapeUsers;
  std::copy(tmpShape.getUsers().begin(), tmpShape.getUsers().end(), std::back_inserter(shapeUsers));

  bool onlyStoreUsers = true;
  SmallVector<memref::StoreOp> storeOpsToErase;

  for (Operation *user : shapeUsers) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemRef() == tmpShape)
        storeOpsToErase.push_back(store);
      else
        onlyStoreUsers = false;
    } else {
      onlyStoreUsers = false;
    }
  }

  if (!onlyStoreUsers) return;

  for (memref::StoreOp s : storeOpsToErase) s.erase();

  if (shapeAlloc->use_empty()) shapeAlloc->erase();
}

static LogicalResult rewriteTempReshapeSourceToOutView(memref::ReshapeOp reshapeOp, Value outArg) {
  Value tmp = reshapeOp.getSource();
  auto tmpTy = dyn_cast<MemRefType>(tmp.getType());
  auto outTy = dyn_cast<MemRefType>(outArg.getType());
  if (!tmpTy || !outTy) return success();

  SmallVector<int64_t, 4> staticDims(tmpTy.getShape().begin(), tmpTy.getShape().end());

  Operation *tmpDef = tmp.getDefiningOp();
  Value newViewVal;

  Location loc = reshapeOp.getLoc();
  Value tmpShape = reshapeOp.getShape();

  if (!tmpDef) {
    BlockArgument ba = mlir::cast<BlockArgument>(tmp);
    Block *parentBlock = ba.getOwner();
    if (!parentBlock) return success();

    auto insertPt = parentBlock->begin();
    OpBuilder b(parentBlock, insertPt);
    newViewVal = buildReshapeFromOut(b, loc, tmpTy, outArg, staticDims);
  } else {
    Block *parentBlock = tmpDef->getBlock();
    auto insertPt = std::next(Block::iterator(tmpDef));
    OpBuilder b(parentBlock, insertPt);
    newViewVal = buildReshapeFromOut(b, loc, tmpTy, outArg, staticDims);
  }

  SmallVector<OpOperand *> uses;
  for (OpOperand &use : tmp.getUses()) {
    if (use.getOwner() == reshapeOp) continue;
    uses.push_back(&use);
  }
  for (auto *u : uses) u->set(newViewVal);

  reshapeOp.getResult().replaceAllUsesWith(outArg);
  reshapeOp.erase();

  if (auto tmpAlloc = tmp.getDefiningOp<memref::AllocOp>()) {
    if (tmpAlloc->use_empty()) tmpAlloc->erase();
  }

  tryEraseOldShapeMemref(tmpShape);

  return success();
}

static LogicalResult rewriteSrcOutReshapeOpToOutView(memref::ReshapeOp reshapeOp, Value outArg) {
  Value src = reshapeOp.getSource();
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  auto outTy = dyn_cast<MemRefType>(outArg.getType());
  if (!srcTy || !outTy) return success();

  auto func = reshapeOp->getParentOfType<func::FuncOp>();
  if (!func || func.empty()) return success();

  Block &entry = func.front();

  SmallVector<int64_t, 4> staticDims(srcTy.getShape().begin(), srcTy.getShape().end());

  OpBuilder topBuilder(&entry, entry.begin());
  Value backView = buildReshapeFromOut(topBuilder, reshapeOp.getLoc(), srcTy, outArg, staticDims);

  SmallVector<memref::StoreOp, 4> srcStores;
  for (Operation *user : src.getUsers()) {
    if (auto store = dyn_cast<memref::StoreOp>(user)) {
      if (store.getMemRef() == src) srcStores.push_back(store);
    }
  }

  for (auto store : srcStores) {
    OpBuilder b(store->getContext());
    b.setInsertionPointAfter(store);
    b.create<memref::StoreOp>(store.getLoc(), store.getValue(), backView, store.getIndices());
  }

  reshapeOp.erase();

  tryEraseOldShapeMemref(reshapeOp.getShape());

  return success();
}

static LogicalResult handleAllocReturn(Value oldResVal, Value outArg) {
  if (auto allocOp = oldResVal.getDefiningOp<memref::AllocOp>()) {
    Value root = allocOp.getResult();
    if (root != outArg) root.replaceAllUsesWith(outArg);
    if (allocOp->use_empty()) allocOp->erase();
  }
  return success();
}

static LogicalResult handleCollapseReturn(Value oldResVal, Value outArg, SmallVector<Value> &origReturnValues,
                                          SmallVector<Value> &newReturnValues) {
  auto collapseOp = oldResVal.getDefiningOp<memref::CollapseShapeOp>();
  if (!collapseOp) return success();

  Value src = collapseOp.getSrc();

  if (llvm::is_contained(origReturnValues, src) || llvm::is_contained(newReturnValues, src)) {
    return rewriteSrcOutCollapseToOutView(collapseOp, outArg);
  }

  return rewriteTempCollapseSourceToOutView(collapseOp, outArg);
}

static LogicalResult handleExpandReturn(Value oldResVal, Value outArg, SmallVector<Value> &origReturnValues,
                                        SmallVector<Value> &newReturnValues) {
  auto expandOp = oldResVal.getDefiningOp<memref::ExpandShapeOp>();
  if (!expandOp) return success();

  Value src = expandOp.getSrc();

  if (llvm::is_contained(origReturnValues, src) || llvm::is_contained(newReturnValues, src)) {
    return rewriteSrcOutExpandOpToOutView(expandOp, outArg);
  }

  return rewriteTempExpandSourceToOutView(expandOp, outArg);
}

static LogicalResult handleReshapeReturn(Value oldResVal, Value outArg, SmallVector<Value> &origReturnValues,
                                         SmallVector<Value> &newReturnValues) {
  auto reshapeOp = oldResVal.getDefiningOp<memref::ReshapeOp>();
  if (!reshapeOp) return success();

  Value src = reshapeOp.getSource();

  if (llvm::is_contained(origReturnValues, src) || llvm::is_contained(newReturnValues, src)) {
    return rewriteSrcOutReshapeOpToOutView(reshapeOp, outArg);
  }

  return rewriteTempReshapeSourceToOutView(reshapeOp, outArg);
}

static bool isReshapeSpecialCase(memref::ReshapeOp reshapeOp, MemRefType srcTy, MemRefType resTy) {
  if (srcTy && resTy && srcTy.hasStaticShape() && resTy.hasStaticShape()) {
    ArrayRef<int64_t> sShape = srcTy.getShape();
    ArrayRef<int64_t> rShape = resTy.getShape();

    auto getNumElems = [](ArrayRef<int64_t> shp) -> int64_t {
      if (shp.empty()) return 1;
      int64_t prod = 1;
      for (int64_t d : shp) {
        if (ShapedType::isDynamic(d)) return ShapedType::kDynamic;
        prod *= d;
      }
      return prod;
    };

    int64_t sElems = getNumElems(sShape);
    int64_t rElems = getNumElems(rShape);
    return sElems == 1 && rElems == 1;
  }
  return false;
}

static LogicalResult processReturnValue(unsigned resultIdx, Value oldResVal, Value maybeOutArg, func::ReturnOp ret,
                                        SmallVector<Value> &origReturnValues, SmallVector<Value> &newReturnValues) {
  if (mlir::isa<BlockArgument>(oldResVal)) {
    return success();
  }

  if (!maybeOutArg) {
    ret.emitError() << "no out-argument is assigned for this return value, "
                       "but it is not a block argument: "
                    << oldResVal;
    return failure();
  }

  Value outArg = maybeOutArg;

  if (oldResVal.getType() != outArg.getType()) {
    ret.emitError() << "type mismatch between return value (" << oldResVal.getType() << ") and out argument ("
                    << outArg.getType() << ")";
    return failure();
  }

  if (isa<memref::AllocOp>(oldResVal.getDefiningOp())) {
    if (failed(handleAllocReturn(oldResVal, outArg))) return failure();
    newReturnValues[resultIdx] = outArg;
    return success();
  }

  if (isa<memref::ReshapeOp>(oldResVal.getDefiningOp())) {
    // special case to memref.expandshape or memref.collapseshape
    auto reshapeOp = cast<memref::ReshapeOp>(oldResVal.getDefiningOp());
    auto srcTy = mlir::dyn_cast<MemRefType>(reshapeOp.getSource().getType());
    auto resTy = mlir::dyn_cast<MemRefType>(reshapeOp.getResult().getType());
    OpBuilder b(reshapeOp);
    if (isReshapeSpecialCase(reshapeOp, srcTy, resTy) && srcTy.getRank() == 0 && resTy.getRank() == 1) {
      // 0D -> 1D
      SmallVector<ReassociationIndices, 1> reassoc;
      auto expand = b.create<memref::ExpandShapeOp>(reshapeOp.getLoc(), resTy, reshapeOp.getSource(), reassoc);
      reshapeOp.replaceAllUsesWith(expand.getResult());
      reshapeOp.erase();
      oldResVal = expand.getResult();
    } else if (isReshapeSpecialCase(reshapeOp, srcTy, resTy) && srcTy.getRank() == 1 && resTy.getRank() == 0) {
      // 1D -> 0D
      SmallVector<ReassociationIndices, 1> reassoc;
      auto collapse = b.create<memref::CollapseShapeOp>(reshapeOp.getLoc(), resTy, reshapeOp.getSource(), reassoc);
      reshapeOp.replaceAllUsesWith(collapse.getResult());
      reshapeOp.erase();
      oldResVal = collapse.getResult();
    } else {
      if (failed(handleReshapeReturn(oldResVal, outArg, origReturnValues, newReturnValues))) return failure();
      newReturnValues[resultIdx] = outArg;
      return success();
    }
  }

  if (isa<memref::CollapseShapeOp>(oldResVal.getDefiningOp())) {
    if (failed(handleCollapseReturn(oldResVal, outArg, origReturnValues, newReturnValues))) return failure();
    newReturnValues[resultIdx] = outArg;
    return success();
  }

  if (isa<memref::ExpandShapeOp>(oldResVal.getDefiningOp())) {
    if (failed(handleExpandReturn(oldResVal, outArg, origReturnValues, newReturnValues))) return failure();
    newReturnValues[resultIdx] = outArg;
    return success();
  }

  ret.emitError() << "unsupported pattern for return value when converting to "
                     "out-parameter: "
                  << oldResVal;
  return success();
}

static LogicalResult handleAllReturns(func::FuncOp func, unsigned origNumResults, SmallVector<Value> maybeOutArgs,
                                      SmallVector<Value> &origReturnValues) {
  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp ret) { returns.push_back(ret); });

  if (returns.empty()) return success();

  SmallVector<Value> newReturnValues(origReturnValues.begin(), origReturnValues.end());

  for (auto ret : returns) {
    auto retOperands = ret.getOperands();
    if (retOperands.size() != origNumResults) {
      ret.emitError() << "number of return values (" << retOperands.size()
                      << ") does not match number of function results (" << origNumResults << ")";
      return failure();
    }

    for (unsigned i = 0; i < origNumResults; ++i) {
      if (failed(processReturnValue(i, retOperands[i], maybeOutArgs[i], ret, origReturnValues, newReturnValues))) {
        return failure();
      }
    }
  }
  return success();
}

template <typename OpTy>
static void eraseDeadOpsOfType(func::FuncOp func) {
  SmallVector<OpTy, 4> deadOps;
  func.walk([&](OpTy op) {
    if (op->use_empty()) deadOps.push_back(op);
  });
  for (auto op : deadOps) op->erase();
}

static void eraseDeadOps(func::FuncOp func) {
  eraseDeadOpsOfType<memref::ExpandShapeOp>(func);
  eraseDeadOpsOfType<memref::CollapseShapeOp>(func);
  eraseDeadOpsOfType<memref::AllocOp>(func);
}

static void rebuildAllReturns(const SmallVectorImpl<func::ReturnOp> &returns, unsigned origNumResults,
                              ArrayRef<bool> needOut, ArrayRef<Value> resultToOutArg) {
  for (auto ret : returns) {
    SmallVector<Value, 4> newRetOperands;
    newRetOperands.reserve(origNumResults);
    auto oldOps = ret.getOperands();
    for (unsigned i = 0; i < origNumResults; ++i) {
      if (needOut[i]) {
        newRetOperands.push_back(resultToOutArg[i]);
      } else {
        newRetOperands.push_back(oldOps[i]);
      }
    }
    OpBuilder builder(ret);
    builder.create<func::ReturnOp>(ret.getLoc(), newRetOperands);
    ret.erase();
  }
}

static LogicalResult rewriteReturnsAndAllocToUseOutParams(func::FuncOp func, unsigned origNumInputs,
                                                          unsigned origNumResults, ArrayRef<bool> needOut) {
  if (origNumResults == 0) return success();
  if (func.empty()) return success();

  Block &entry = func.front();
  if (entry.getNumArguments() < origNumInputs) {
    func.emitError() << "entry block arg count is less than original inputs";
    return failure();
  }

  func::ReturnOp anyRet;
  func.walk([&](func::ReturnOp r) {
    if (!anyRet) anyRet = r;
  });
  if (!anyRet) return success();

  if (anyRet.getNumOperands() != origNumResults) {
    anyRet.emitError() << "unexpected number of return operands: got " << anyRet.getNumOperands() << ", expected "
                       << origNumResults;
    return failure();
  }

  unsigned expectedTotalArgs = origNumInputs;
  for (unsigned i = 0; i < origNumResults; ++i)
    if (needOut[i]) ++expectedTotalArgs;

  if (entry.getNumArguments() != expectedTotalArgs) {
    func.emitError() << "entry block arg count mismatch after selective out-param "
                        "transform: expected "
                     << expectedTotalArgs << " got " << entry.getNumArguments();
    return failure();
  }

  SmallVector<Value> resultToOutArg(origNumResults, Value());
  unsigned curOutArgIdx = origNumInputs;
  for (unsigned i = 0; i < origNumResults; ++i) {
    if (needOut[i]) {
      resultToOutArg[i] = entry.getArgument(curOutArgIdx++);
    }
  }

  SmallVector<func::ReturnOp> returns;
  SmallVector<Value> origReturnValues;
  if (failed(collectOriginalReturnValues(func, origNumResults, returns, origReturnValues))) return failure();

  if (failed(handleAllReturns(func, origNumResults, resultToOutArg, origReturnValues))) return failure();

  rebuildAllReturns(returns, origNumResults, needOut, resultToOutArg);
  eraseDeadOps(func);

  return success();
}

static LogicalResult transformFunc(func::FuncOp func, OpBuilder &builder) {
  auto funcTy = func.getFunctionType();
  auto *ctx = builder.getContext();

  SmallVector<Type> origInputs(funcTy.getInputs().begin(), funcTy.getInputs().end());
  SmallVector<Type> origResults(funcTy.getResults().begin(), funcTy.getResults().end());

  unsigned origNumInputs = origInputs.size();
  unsigned origNumResults = origResults.size();

  if (origNumResults == 0) {
    setHaccIOArgAttrs(func, origNumInputs, /*nOutputs=*/0, builder);
    return success();
  }

  if (func.empty()) {
    SmallVector<Type> newInputs;
    newInputs.reserve(origNumInputs + origNumResults);
    newInputs.append(origInputs.begin(), origInputs.end());
    newInputs.append(origResults.begin(), origResults.end());

    auto newFuncTy = FunctionType::get(ctx, newInputs, origResults);
    func.setFunctionType(newFuncTy);

    setHaccIOArgAttrs(func, origNumInputs, origNumResults, builder);
    return success();
  }

  func::ReturnOp reprRet;
  func.walk([&](func::ReturnOp r) {
    if (!reprRet) reprRet = r;
  });
  if (!reprRet) {
    setHaccIOArgAttrs(func, origNumInputs, /*nOutputs=*/0, builder);
    return success();
  }

  if (reprRet.getNumOperands() != origNumResults) {
    reprRet.emitError() << "number of return values (" << reprRet.getNumOperands()
                        << ") does not match number of function results (" << origNumResults << ")";
  }

  SmallVector<bool, 4> needOut(origNumResults, true);
  for (unsigned i = 0; i < origNumResults; ++i) {
    Value rv = reprRet.getOperand(i);
    if (isAliasOfOriginalInput(rv, func, origNumInputs)) {
      needOut[i] = false;
    }
  }

  SmallVector<Type> newInputs;
  newInputs.reserve(origNumInputs + origNumResults);
  newInputs.append(origInputs.begin(), origInputs.end());
  for (unsigned i = 0; i < origNumResults; ++i) {
    if (needOut[i]) {
      newInputs.push_back(origResults[i]);
    }
  }

  auto newFuncTy = FunctionType::get(ctx, newInputs, origResults);
  func.setFunctionType(newFuncTy);

  Block &entry = func.front();

  if (entry.getNumArguments() != origNumInputs) {
    func.emitError() << "unexpected number of entry block arguments before transform: got " << entry.getNumArguments()
                     << ", expected " << origNumInputs;
    return failure();
  }

  for (unsigned i = 0; i < origNumResults; ++i) {
    if (needOut[i]) {
      (void)entry.addArgument(origResults[i], func.getLoc());
    }
  }

  unsigned nOutputs = 0;
  for (unsigned i = 0; i < origNumResults; ++i)
    if (needOut[i]) ++nOutputs;

  setHaccIOArgAttrs(func, origNumInputs, nOutputs, builder);

  if (failed(rewriteReturnsAndAllocToUseOutParams(func, origNumInputs, origNumResults, needOut))) return failure();

  return success();
}

struct AddOutParameter : public mlir::impl::AddOutParameterBase<AddOutParameter> {
  AddOutParameter() = default;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (!module) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(module.getContext());

    SmallVector<func::FuncOp, 8> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });

    auto it = std::find_if(funcs.begin(), funcs.end(), [&](func::FuncOp f) {
      if (failed(transformFunc(f, builder))) {
        f.emitError("AddOutParameter pass failed for this function");
        return true;
      }
      return false;
    });

    if (it != funcs.end()) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createAddOutParameterPass() {
  return std::make_unique<mlir::AddOutParameter>();
}
