/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Affine/Transforms/FixDynamicIndexing.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"

#include <set>
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"

namespace mlir {
#define GEN_PASS_DEF_FIXDYNAMICINDEXING
#define GEN_PASS_DECL_FIXDYNAMICINDEXING
#include "akg/Dialect/Affine/Passes.h.inc"
}  // namespace mlir

using namespace mlir;
using namespace llvm;
using namespace akgglobal;

namespace {

static std::vector<int> ArrayAttrToVectorInt(ArrayAttr array) {
  std::vector<int> res;
  for (auto v : array.getValue()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(v)) {
      res.push_back(intAttr.getInt());
    }
  }
  return res;
}

static bool IsDynamicDim(const mlir::Value memref, size_t dim) {
  if (!memref) {
    return false;
  }
  mlir::Type shapedValue = memref.getType();
  auto shapedType = cast<ShapedType>(shapedValue);
  if (!shapedType || !isa<MemRefType>(shapedType)) {
    return false;
  }
  return dim < static_cast<size_t>(shapedType.getRank()) && shapedType.isDynamicDim(dim);
}

/**
 * @brief A record of data flow from an argument's dynamic dimension to an actual memref.load/store
 * @arg srcArgIndex : the index of src argument from a func
 * @arg srcDataDims : the dynamic dimensions of src argument, i.e. the dataflow source
 * @arg srcDataMemrefDim : the `memref.dim` to fetch the srcDataDims from argArgIndex
 * @arg destDataDim : the dynamic dimensions of dest memref stmt, i.e. the dataflow dest
 *
 * e.g. for an input mlir like this
 * func.func @elem_broadcast_last_5(%arg0: memref<4096x?xf32>)
 *   %expand_shape = memref.expand_shape %arg0 [[0], [1, 2]] : memref<4096x?xf32> into memref<4096x1x?xf32>
 *   %c1 = arith.constant 1 : index
 *   %dim = memref.dim %arg0, %c1 : memref<4096x?xf32>
 *   %1 = memref.load %expand_shape[%arg4, %arg5, %arg6] : memref<4096x1x?xf32>
 *
 * Will generate two DynamicDataFlow records because of the expand_shape:
 *  #1:
 *    - srcArgIndex = 0 (i.e. %arg0)
 *    - srcDataDims = {1} (i.e. index to `?` in memref<4096x?xf32>)
 *    - srcDataMemrefDim = %dim
 *    - destDataDim = 1 (i.e. %arg5)
 *  #2:
 *    - srcArgIndex = 0 (i.e. %arg0)
 *    - srcDataDims = {1} (i.e. index to `?` in memref<4096x?xf32>)
 *    - srcDataMemrefDim = %dim
 *    - destDataDim = 2 (i.e. %arg6)
 */
struct DynamicDataFlow {
 public:
  DynamicDataFlow(size_t a, size_t s, size_t d) : srcArgIndex(a), destDataDim(d) { (void)this->srcDataDims.insert(s); }

  DynamicDataFlow(size_t a, ReassociationIndices s, size_t d) : srcArgIndex(a), destDataDim(d) {
    for (auto ss : s) {
      (void)this->srcDataDims.insert(static_cast<size_t>(ss));
    }
  }

  size_t srcArgIndex;
  std::set<size_t> srcDataDims;
  mlir::Value srcDataMemrefDim;
  size_t destDataDim;
};
using DynamicDataFlowPtr = std::shared_ptr<DynamicDataFlow>;

class FixDynamicIndexingPass : public impl::FixDynamicIndexingBase<FixDynamicIndexingPass> {
 public:
  FixDynamicIndexingPass() {}

  void runOnOperation() override;

  void CollectNeedFixArgsMap();
  void AffineMemrefLowerPass();
  void GetMemUserAndFixIndex(size_t argIdx, Operation *op, SmallVector<Operation *> &memrefOps);
  void CollectNeedFixMemrefs();
  void PrepareMemrefDimOp();
  void ReplaceInputDimsWithOutput();
  void InsertIfOpAndFixIndex();

  std::map<Operation *, std::vector<DynamicDataFlowPtr>> inputsFixPlan;
  std::map<Operation *, std::vector<DynamicDataFlowPtr>> outputsFixPlan;
  SmallVector<Operation *, 4> redInputs;
  SmallVector<Operation *, 4> redOutputs;

 private:
  void CollectNeedFixArgsMapFromAttr();
  void CollectNeedFixArgsMapFromGlobalMap();
  bool createIfOpWithVectorType(vector::LoadOp loadOp, mlir::Value buffer, DynamicDataFlowPtr df);
  template <typename T>
  void createIfOpWithIndexType(T loadOp, mlir::Value buffer, DynamicDataFlowPtr df) const;
  void CollectReductionRelatedOps();
  void InsertIfOpAndFixIndexImpl(vector::LoadOp loadOp, DynamicDataFlowPtr df);
  void InsertIfOpAndFixIndexImpl(memref::LoadOp loadOp, DynamicDataFlowPtr df);
  template <typename T, typename M>
  affine::AffineIfOp createAffineIfOp(T loadOp, const DynamicDataFlowPtr &df, M resultTypes) const;
  template <typename T>
  mlir::Value getOriginalUpperBound(T storeOp, size_t dim, SmallVector<Operation *> allFors);
  void UpdateFixIndex(size_t argIdx, SmallVector<ReassociationIndices, 4> reassociation, bool reversed);
  std::map<size_t, std::vector<DynamicDataFlowPtr>> argumentDataFlows;
  Operation *constZeroOp{nullptr};
};
}  // namespace

// Lower the affine.load/store to memref.load/store for if-inserting
// todo(baiji): move this into a single pass?
void FixDynamicIndexingPass::AffineMemrefLowerPass() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp);
  SmallVector<Operation *> toErase;
  funcOp.walk([&](Operation *op) {
    if ((inputsFixPlan.find(op) == inputsFixPlan.end() && outputsFixPlan.find(op) == outputsFixPlan.end() &&
         std::find(redInputs.begin(), redInputs.end(), op) == redInputs.end()) ||
        (std::find(redOutputs.begin(), redOutputs.end(), op) != redOutputs.end())) {
      return;
    }
    auto updateFixPlan = [&](Operation *newOp) {
      auto itin = inputsFixPlan.find(op);
      if (itin != inputsFixPlan.end()) {
        inputsFixPlan[newOp] = itin->second;
        (void)inputsFixPlan.erase(itin);
      }
      auto itout = outputsFixPlan.find(op);
      if (itout != outputsFixPlan.end()) {
        outputsFixPlan[newOp] = itout->second;
        (void)outputsFixPlan.erase(itout);
      }
    };
    if (auto affineLoadOp = dyn_cast<affine::AffineLoadOp>(op)) {
      builder.setInsertionPoint(affineLoadOp);
      mlir::Value memref = affineLoadOp.getMemRef();
      auto indices = affineLoadOp.getIndices();
      mlir::Value memrefLoad = builder.create<memref::LoadOp>(affineLoadOp.getLoc(), memref, indices);
      affineLoadOp.replaceAllUsesWith(memrefLoad);
      toErase.push_back(op);
      updateFixPlan(memrefLoad.getDefiningOp());
    } else if (auto affineStoreOp = dyn_cast<affine::AffineStoreOp>(op)) {
      mlir::Value memref = affineStoreOp.getMemRef();
      auto indices = affineStoreOp.getIndices();
      mlir::Value valueToStore = affineStoreOp.getValueToStore();
      builder.setInsertionPoint(affineStoreOp);
      auto memrefStore = builder.create<memref::StoreOp>(affineStoreOp.getLoc(), valueToStore, memref, indices);
      toErase.push_back(op);
      updateFixPlan(memrefStore);
    }
  });
  for (auto op : toErase) {
    op->erase();
  }
}

// Trace the `No.argIdx` func arg's dimension indexing during reassociation
void FixDynamicIndexingPass::UpdateFixIndex(size_t argIdx, SmallVector<ReassociationIndices, 4> reassociation,
                                            bool reversed) {
  std::set<size_t> needFixIdx;
  auto it = argumentDataFlows.find(argIdx);
  if (it == argumentDataFlows.end()) {
    return;
  }
  for (auto dataFlow : it->second) {
    for (auto fixIdx : dataFlow->srcDataDims) {
      (void)needFixIdx.insert(fixIdx);
    }
  }
  size_t reIdx = 0;
  for (auto needFixAt : reassociation) {
    if (needFixIdx.count(reIdx) == 0) {
      ++reIdx;
      continue;
    }
    mlir::Value srcDataMemrefDim;
    for (auto dataFlow : it->second) {
      if (!srcDataMemrefDim && dataFlow->srcDataDims.count(reIdx) != 0) {
        srcDataMemrefDim = dataFlow->srcDataMemrefDim;
      }
    }
    if (!srcDataMemrefDim) {
      continue;
    }

    if (!reversed) {
      // That is a expand shape
      std::vector<DynamicDataFlowPtr> reassociatedDataFlows;
      for (auto vi : needFixAt) {
        auto newDf = std::make_shared<DynamicDataFlow>(argIdx, reIdx, static_cast<size_t>(vi));
        newDf->srcDataMemrefDim = srcDataMemrefDim;
        reassociatedDataFlows.push_back(newDf);
      }
      argumentDataFlows[argIdx] = reassociatedDataFlows;
    } else {
      // That is a collapse shape
      auto newDf = std::make_shared<DynamicDataFlow>(argIdx, needFixAt, reIdx);
      newDf->srcDataMemrefDim = srcDataMemrefDim;
      argumentDataFlows[argIdx] = {newDf};
    }
    ++reIdx;
  }
}

void FixDynamicIndexingPass::GetMemUserAndFixIndex(size_t argIdx, Operation *op, SmallVector<Operation *> &memrefOps) {
  if (argumentDataFlows.find(argIdx) == argumentDataFlows.end()) {
    return;
  }

  // When reaching the final memref user of arguments, we need to fix
  // the indexing of original arguments if it is reassociated.
  if (auto expandShape = dyn_cast<memref::ExpandShapeOp>(op)) {
    UpdateFixIndex(argIdx, expandShape.getReassociationIndices(), false);
  } else if (auto collapse = dyn_cast<memref::CollapseShapeOp>(op)) {
    UpdateFixIndex(argIdx, collapse.getReassociationIndices(), true);
  }

  // Record the memref users.
  if (isa<memref::LoadOp, memref::StoreOp, affine::AffineLoadOp, affine::AffineStoreOp, vector::LoadOp,
          vector::StoreOp>(op)) {
    memrefOps.push_back(op);
  }

  // inputs may be used in alloc (e.g. through memref.Dim) but the users of allocOp
  // should not be involved in input alignment.
  if (isa<memref::AllocOp>(op)) {
    return;
  }

  for (auto user : op->getUsers()) {
    GetMemUserAndFixIndex(argIdx, user, memrefOps);
  }
};

void FixDynamicIndexingPass::CollectNeedFixArgsMapFromAttr() {
  func::FuncOp funcOp = getOperation();
  auto array = dyn_cast<ArrayAttr>(funcOp->getAttr(kNeedFix));
  if (!array) {
    llvm::errs() << "Cannot fix dynamic indexing: attr " << kNeedFix
                 << " value should be like [#Tensor[#Dim,],[...]], please check.\n";
    return;
  }
  size_t tensorIdx = 0;
  for (auto tensorArray : array.getValue()) {
    auto dimArray = dyn_cast<ArrayAttr>(tensorArray);
    if (!dimArray) {
      llvm::errs() << "Cannot fix dynamic indexing: attr " << kNeedFix
                   << " value should be like [#Tensor[#Dim,],[...]], please check.\n";
      return;
    }
    auto needFixAt = ArrayAttrToVectorInt(dimArray);
    for (size_t dim = 0; dim < needFixAt.size(); ++dim) {
      if (needFixAt[dim] != 0) {
        argumentDataFlows[tensorIdx].push_back(std::make_shared<DynamicDataFlow>(tensorIdx, dim, dim));
      }
    }
    ++tensorIdx;
  }
  (void)funcOp->removeAttr(kNeedFix);
}

void FixDynamicIndexingPass::CollectNeedFixArgsMapFromGlobalMap() {
  ShapeAlignTool &tool = ShapeAlignTool::getInstance();
  func::FuncOp funcOp = getOperation();
  for (size_t tensorIdx = 0; tensorIdx < tool.getFuncArgSizes(); ++tensorIdx) {
    if (tool.isOutput(tensorIdx)) {
      mlir::Value tensorArg = funcOp.getBody().front().getArgument(tensorIdx);
      auto currShape = tool.getCurrShapeInfo(tensorIdx);
      for (size_t dim = 0; dim < currShape.size(); ++dim) {
        if (IsDynamicDim(tensorArg, dim)) {
          argumentDataFlows[tensorIdx].push_back(std::make_shared<DynamicDataFlow>(tensorIdx, dim, dim));
        }
      }
    } else {
      auto needFixAt = tool.getNeedFixIndice(tensorIdx);
      for (size_t dim = 0; dim < needFixAt.size(); ++dim) {
        if (needFixAt[dim] != 0) {
          argumentDataFlows[tensorIdx].push_back(std::make_shared<DynamicDataFlow>(tensorIdx, dim, dim));
        }
      }
    }
  }
}

void FixDynamicIndexingPass::CollectNeedFixArgsMap() {
  func::FuncOp funcOp = getOperation();
  if (funcOp->hasAttr(kNeedFix)) {
    CollectNeedFixArgsMapFromAttr();
  } else {
    CollectNeedFixArgsMapFromGlobalMap();
  }
}

void FixDynamicIndexingPass::CollectReductionRelatedOps() {
  redInputs.clear();
  redOutputs.clear();
  SmallVector<mlir::Value> redArgs;
  auto funcOp = getOperation();
  if (!funcOp->hasAttr("OperatorType") ||
      dyn_cast<StringAttr>(funcOp->getAttr("OperatorType")).getValue().str() == "Reduce") {
    funcOp->walk([&](Operation *op) {
      if (op->hasAttr(kReductionTypeStr)) {
        auto operand0 = op->getOperand(0);
        redInputs.push_back(operand0.getDefiningOp());
        auto operand1 = op->getOperand(1);
        auto mem = cast<affine::AffineLoadOp>(operand1.getDefiningOp()).getMemref();
        redArgs.push_back(mem);

        for (Operation *nextOp : op->getUsers()) {
          if (auto store = dyn_cast<affine::AffineStoreOp>(nextOp)) {
            redOutputs.push_back(nextOp);
            break;
          }
        }
      }
    });
  }
  auto it = outputsFixPlan.begin();
  while (it != outputsFixPlan.end()) {
    mlir::Value mem;
    if (auto load = dyn_cast<affine::AffineLoadOp>(it->first)) {
      mem = load.getMemref();
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(it->first)) {
      mem = store.getMemref();
    }
    if (mem) {
      bool flag = false;
      for (auto arg : redArgs) {
        if (arg == mem) {
          flag = true;
          break;
        }
      }
      if (flag) {
        it = outputsFixPlan.erase(it);  // erase returns the iterator following the last removed element
      } else {
        ++it;
      }
    } else {
      (void)it->first->emitWarning("The op is not affine::AffineLoadOp/affine::AffineStoreOp, cannot get memref.");
      ++it;
    }
  }
}

void FixDynamicIndexingPass::CollectNeedFixMemrefs() {
  func::FuncOp funcOp = getOperation();
  ShapeAlignTool &tool = ShapeAlignTool::getInstance();

  for (size_t argIdx = 0; argIdx < funcOp.getBody().front().getArguments().size(); ++argIdx) {
    auto arg = funcOp.getBody().front().getArgument(static_cast<int>(argIdx));
    SmallVector<Operation *> memrefOps;
    for (auto user : arg.getUsers()) {
      GetMemUserAndFixIndex(argIdx, user, memrefOps);
    }

    for (auto op : memrefOps) {
      auto it = argumentDataFlows.find(argIdx);
      if (it == argumentDataFlows.end()) {
        continue;
      }
      for (auto dataFlow : it->second) {
        if (tool.isOutput(argIdx)) {
          // We don't need to fix the index of outputs cause all shapes of elem-wise inputs
          // are aligned to output in previous pass.
          // The reason we record the output fix plan here is to fix the upper-bound of affine.for
          // from inputs' dim to outputs' dim. And that is cause by Linalg's constraint.
          outputsFixPlan[op].push_back(dataFlow);
        } else if (it != argumentDataFlows.end()) {
          inputsFixPlan[op].push_back(dataFlow);
        }
      }
    }
  }
}

void FixDynamicIndexingPass::PrepareMemrefDimOp() {
  func::FuncOp funcOp = getOperation();
  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&(funcOp.getBody().front()));
  constZeroOp = builder.create<arith::ConstantIndexOp>(funcOp->getLoc(), 0);
  for (size_t argIdx = 0; argIdx < funcOp.getBody().front().getArguments().size(); ++argIdx) {
    auto it = argumentDataFlows.find(argIdx);
    if (it == argumentDataFlows.end()) {
      continue;
    }
    auto dataFlows = it->second;
    mlir::Value tensorArg = funcOp.getBody().front().getArgument(static_cast<int>(argIdx));

    for (size_t di = 0; di < dataFlows.size(); ++di) {
      auto df = dataFlows[di];
      for (auto i : df->srcDataDims) {
        if (IsDynamicDim(tensorArg, i)) {
          df->srcDataMemrefDim = builder.create<memref::DimOp>(funcOp->getLoc(), tensorArg, static_cast<int64_t>(i));
        }
      }
    }
  }
}

/**
 * @brief Insert the if condition in the corresponding position based on the analysis result of the previous process.
 * e.g
 * inputs:
 * %3 = memref.load %expand_shape[%arg4, %arg5, %arg6] : memref<4096x1x?xf32>
 * outputs:
 * %2 = affine.if #set()[%dim_2] -> index {
 *        affine.yield %arg6 : index
 *      } else {
 *        affine.yield %c0 : index
 *      }
 * %3 = memref.load %expand_shape[%arg4, %arg5, %2] : memref<4096x1x?xf32>
 */
template <typename T, typename M>
affine::AffineIfOp FixDynamicIndexingPass::createAffineIfOp(T loadOp, const DynamicDataFlowPtr &df,
                                                            M resultTypes) const {
  auto loadIndices = loadOp.getIndices();
  if (loadIndices.size() <= df->destDataDim) {
    llvm::errs() << "The value of destDataDim exceeds the upper limit of the indices.\n";
    return nullptr;
  }

  auto context = loadOp.getContext();
  auto expr = mlir::getAffineConstantExpr(1, context);
  auto affineExpr = mlir::getAffineSymbolExpr(0, context) - expr;
  SmallVector<AffineExpr, 4> exprs = {affineExpr};
  SmallVector<bool, 4> eqFlags = {true};
  // Create an if condition for a symbol and zero dim
  IntegerSet ifCondSet = IntegerSet::get(0, 1, exprs, eqFlags);

  SmallVector<mlir::Value, 4> setOperands = {df->srcDataMemrefDim};
  affine::canonicalizeSetAndOperands(&ifCondSet, &setOperands);

  OpBuilder b(loadOp);
  return b.create<affine::AffineIfOp>(loadOp.getLoc(), resultTypes, ifCondSet, setOperands, true);
}

template <typename T>
void FixDynamicIndexingPass::createIfOpWithIndexType(T loadOp, mlir::Value buffer, DynamicDataFlowPtr df) const {
  if (!IsDynamicDim(buffer, df->destDataDim)) {
    return;
  }

  auto loadIndices = loadOp.getIndices();
  auto destIndices = loadIndices[df->destDataDim];
  SmallVector<mlir::Type, 4> resultTypes = {destIndices.getType()};
  affine::AffineIfOp ifOp = createAffineIfOp(loadOp, df, resultTypes);
  if (!ifOp) {
    return;
  }

  OpBuilder b(loadOp);
  // insert then block
  SmallVector<mlir::Value, 4> thenYield = {constZeroOp->getResult(0)};
  b.setInsertionPointToStart(ifOp.getThenBlock());
  b.create<affine::AffineYieldOp>(loadOp.getLoc(), thenYield);
  // insert else block
  SmallVector<mlir::Value, 4> elseYield = {destIndices};
  b.setInsertionPointToStart(ifOp.getElseBlock());
  b.create<affine::AffineYieldOp>(loadOp.getLoc(), elseYield);

  // Replace the original indice with the result returned by affine.if.
  SmallVector<mlir::Value, 4> newLoadIndices;
  newLoadIndices.reserve(loadIndices.size());
  for (auto index : loadIndices) {
    if (index == destIndices) {
      newLoadIndices.push_back(ifOp.getResult(0));
      continue;
    }
    newLoadIndices.push_back(index);
  }
  loadOp.getIndicesMutable().assign(newLoadIndices);
}

bool FixDynamicIndexingPass::createIfOpWithVectorType(vector::LoadOp loadOp, mlir::Value buffer,
                                                      DynamicDataFlowPtr df) {
  if (!IsDynamicDim(buffer, df->destDataDim)) {
    return false;
  }
  auto loadIndices = loadOp.getIndices();
  if (df->destDataDim != loadIndices.size() - 1) {
    createIfOpWithIndexType(loadOp, buffer, df);
    return false;
  }

  affine::AffineIfOp ifOp = createAffineIfOp(loadOp, df, loadOp.getVectorType());
  if (!ifOp) {
    return false;
  }

  OpBuilder b(loadOp);
  auto destIndices = loadIndices[df->destDataDim];
  // insert then block
  b.setInsertionPointToStart(ifOp.getThenBlock());
  SmallVector<mlir::Value, 4> newLoadIndices;
  newLoadIndices.reserve(loadIndices.size());
  for (auto index : loadIndices) {
    if (index == destIndices) {
      newLoadIndices.push_back(constZeroOp->getResult(0));
    } else {
      newLoadIndices.push_back(index);
    }
  }
  mlir::Value memrefLoad = b.create<memref::LoadOp>(loadOp.getLoc(), buffer, newLoadIndices);
  auto broadcastOp = b.create<vector::BroadcastOp>(loadOp.getLoc(), loadOp.getVectorType(), memrefLoad);
  SmallVector<mlir::Value, 4> thenYield = {broadcastOp};
  b.create<affine::AffineYieldOp>(broadcastOp.getLoc(), thenYield);

  // insert else block
  b.setInsertionPointToStart(ifOp.getElseBlock());
  auto elseOp = b.clone(*loadOp.getOperation());
  SmallVector<mlir::Value, 4> elseYield = {elseOp->getResult(0)};
  b.create<affine::AffineYieldOp>(elseOp->getLoc(), elseYield);

  loadOp.getOperation()->getResult(0).replaceAllUsesWith(ifOp.getResult(0));
  return true;
}

void FixDynamicIndexingPass::InsertIfOpAndFixIndex() {
  SmallSet<Operation *, 8> eraseOp;
  for (auto it : inputsFixPlan) {
    // todo(baiji): Temp buffer may use StoreOp, need to support later.
    if (!isa<memref::LoadOp, vector::LoadOp>(it.first)) {
      continue;
    }

    for (auto df : it.second) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(it.first)) {
        createIfOpWithIndexType(loadOp, loadOp.getMemref(), df);
      } else if (auto readOp = dyn_cast<vector::LoadOp>(it.first)) {
        bool needErase = createIfOpWithVectorType(readOp, readOp.getBase(), df);
        if (needErase) {
          eraseOp.insert(readOp);
        }
      }
    }
  }

  for (auto op : eraseOp) {
    op->erase();
  }
}

template <typename T>
mlir::Value FixDynamicIndexingPass::getOriginalUpperBound(T storeOp, size_t dim, SmallVector<Operation *> allFors) {
  auto indices = storeOp.getIndices();
  assert(dim < indices.size());
  auto loopVar = indices[dim];
  for (auto op : allFors) {
    auto forOp = dyn_cast<affine::AffineForOp>(op);
    if (!forOp) {
      continue;
    }
    auto inductionVar = forOp.getInductionVar();
    if (inductionVar != loopVar) {
      continue;
    }
    for (size_t i = 0; i < forOp.getUpperBound().getNumOperands(); ++i) {
      auto ub = forOp.getUpperBound().getOperand(i);
      if (ub.getDefiningOp() && isa<memref::DimOp>(ub.getDefiningOp())) {
        return ub;
      }
    }
  }
  return mlir::Value();
}

void FixDynamicIndexingPass::ReplaceInputDimsWithOutput() {
  SmallVector<Operation *> allFors;
  getOperation()->walk([&](Operation *op) {
    if (!isa<affine::AffineForOp>(op)) {
      return;
    }
    allFors.push_back(op);
  });
  for (auto it : outputsFixPlan) {
    if (!isa<memref::StoreOp, vector::StoreOp>(it.first)) {
      continue;
    }
    for (auto df : it.second) {
      mlir::Value newUb;
      if (df->srcDataMemrefDim && df->srcDataMemrefDim.getDefiningOp() &&
          isa<memref::DimOp>(df->srcDataMemrefDim.getDefiningOp())) {
        newUb = df->srcDataMemrefDim;
      }
      if (!newUb) {
        continue;
      }

      mlir::Value originalUb;
      if (auto storeOp = dyn_cast<memref::StoreOp>(it.first)) {
        originalUb = getOriginalUpperBound(storeOp, df->destDataDim, allFors);
      } else if (auto storeOp = dyn_cast<vector::StoreOp>(it.first)) {
        originalUb = getOriginalUpperBound(storeOp, df->destDataDim, allFors);
      }

      if (!originalUb || originalUb == newUb) {
        continue;
      }

      originalUb.replaceAllUsesWith(newUb);
    }
  }
}

void FixDynamicIndexingPass::runOnOperation() {
  std::vector<SmallVector<affine::AffineForOp, 6>> bands;
  getTileableBands(getOperation(), &bands);
  std::string target{kTargetCpu};

  if (getOperation()->getAttr("process")) {
    target = dyn_cast<StringAttr>(getOperation()->getAttr("process")).getValue().str();
  }
  if (bands.size() > 1 && target == kTargetCuda) {
    llvm::report_fatal_error(llvm::StringRef("Detect multiple bands (nested affine for)."));
  }

  // 1. collect info
  CollectNeedFixArgsMap();
  PrepareMemrefDimOp();
  CollectNeedFixMemrefs();
  CollectReductionRelatedOps();

  // 2. lower affine.load/store to memref.load/store so that we can do conditional indexing
  AffineMemrefLowerPass();

  // 3. fix loops' upper bounds from getting inputs' dim to outputs'
  ReplaceInputDimsWithOutput();

  // 4. start to fix by inserting if op
  InsertIfOpAndFixIndex();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createFixDynamicIndexingPass() {
  return std::make_unique<FixDynamicIndexingPass>();
}
