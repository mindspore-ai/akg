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

#include "akg/Dialect/MindSpore/Spliter/Spliter.h"

#include "akg/Dialect/MindSpore/Spliter/AscendModel.h"
#include "akg/Dialect/MindSpore/Spliter/CpuModel.h"
#include "akg/Dialect/MindSpore/Spliter/GpuModel.h"
#include "akg/Dialect/MindSpore/Spliter/LiteGraph.h"
#include "akg/Dialect/MindSpore/Spliter/MindSporeToJson.h"
#include "akg/Dialect/MindSpore/Spliter/OpRegister.h"

#define DEBUG_TYPE "akg-mindspore-spliter"
namespace mlir::spliter {
ValueSeq getAreaInputs(const OpArea &area) {
  llvm::SmallDenseSet<Value> inputSet;
  llvm::SmallDenseSet<Operation *> opSet(area.begin(), area.end());
  ValueSeq orderedInputs;
  for (Operation *op : area) {
    for (Value opnd : op->getOperands()) {
      if (opSet.find(opnd.getDefiningOp()) == opSet.end() && !inputSet.contains(opnd)) {
        (void)inputSet.insert(opnd);
        orderedInputs.push_back(opnd);
      }
    }
  }
  return orderedInputs;
}

ValueSeq getAreaOutputs(const OpArea &area, func::FuncOp func) {
  llvm::SmallDenseSet<Operation *> opSet(area.begin(), area.end());
  llvm::SmallDenseSet<Operation *> opSetOutside;
  ValueSeq outputs;
  for (Operation &op : func.getBody().front()) {
    if (opSet.find(&op) == opSet.end()) {
      (void)opSetOutside.insert(&op);
    }
  }
  for (Operation *op : area) {
    for (Value result : op->getResults()) {
      bool hasUserOutsideArea =
        llvm::any_of(result.getUses(), [&](const OpOperand &use) { return opSetOutside.count(use.getOwner()); });
      // operand that has user outside the area is real output
      if (hasUserOutsideArea) {
        outputs.push_back(result);
      }
    }
  }
  return outputs;
}

TypeSeq getTypesFromValues(const ValueSeq &valueSeq) {
  TypeSeq typeSeq;
  (void)std::transform(valueSeq.begin(), valueSeq.end(), std::back_inserter(typeSeq),
                       [](const Value &value) { return value.getType(); });
  return typeSeq;
}

Location getAreaLoc(const OpArea &area) {
  std::vector<Location> locs;
  (void)std::transform(area.begin(), area.end(), std::back_inserter(locs), [](Operation *op) { return op->getLoc(); });
  return FusedLoc::get(area[0]->getContext(), locs);
}

std::string getAreaFuncName(const std::string oriFuncName) {
  static size_t count = 0;
  return oriFuncName + "_" + std::to_string(count++);
}

void extractAttrs(func::FuncOp func, const DictionaryAttr &attrs) {
  std::set<std::string> excludedAttr{"function_type", "sym_name"};
  (void)std::for_each(attrs.begin(), attrs.end(), [&](const mlir::NamedAttribute &attr) {
    if (excludedAttr.find(attr.getName().str()) == excludedAttr.end()) {
      func->setAttr(attr.getName(), attr.getValue());
    }
  });
}

func::FuncOp CreateAreaFunc(const OpArea &area, const ValueSeq &areaInputs, const ValueSeq &areaOutputs,
                            const Location &areaLoc, func::FuncOp oriFunc, OpBuilder &builder) {
  auto inputTypes = getTypesFromValues(areaInputs);
  auto outputTypes = getTypesFromValues(areaOutputs);
  auto areaFuncType = builder.getFunctionType(inputTypes, outputTypes);
  builder.setInsertionPoint(area.front()->getParentOp());
  auto oriFuncName = oriFunc.getSymName().str();
  auto oriAttrs = oriFunc->getAttrDictionary();
  func::FuncOp areaFunc = builder.create<func::FuncOp>(areaLoc, getAreaFuncName(oriFuncName), areaFuncType);
  extractAttrs(areaFunc, oriAttrs);
  Block *areaFuncBlock = areaFunc.addEntryBlock();
  builder.setInsertionPoint(areaFuncBlock, areaFuncBlock->end());
  IRMapping value_map;
  for (auto z : llvm::zip(areaInputs, areaFunc.getArguments())) {
    value_map.map(std::get<0>(z), std::get<1>(z));
  }
  for (Operation *op : area) {
    (void)builder.clone(*op, value_map);
  }
  llvm::SmallVector<Value, 4> areaFuncRet;
  for (Value output : areaOutputs) {
    areaFuncRet.push_back(value_map.lookupOrDefault(output));
  }
  (void)builder.create<func::ReturnOp>(areaLoc, areaFuncRet);
  return areaFunc;
}

void sinkAreaUses(const OpArea &area) {
  llvm::SmallDenseSet<Operation *> areaOps(area.begin(), area.end());
  llvm::SmallDenseSet<Operation *> areaUsesOps;
  llvm::SmallVector<Operation *> areaUsesOpsWithOrder;
  for (Operation &opRef : llvm::make_range(area.front()->getIterator(), area.back()->getIterator())) {
    auto op = &opRef;
    // skip op already in area
    if (areaOps.contains(op)) {
      continue;
    }
    // find area use op mixed in area
    if (llvm::any_of(op->getOperands(), [&areaOps, &areaUsesOps](Value opnd) {
          auto inputOp = opnd.getDefiningOp();
          return areaOps.contains(inputOp) || areaUsesOps.contains(inputOp);
        })) {
      (void)areaUsesOps.insert(op);
      areaUsesOpsWithOrder.push_back(op);
    }
  }
  for (auto op : llvm::reverse(areaUsesOpsWithOrder)) {
    op->moveAfter(area.back());
  }
}

func::CallOp CreateAreaCall(const OpArea &area, func::FuncOp areaFunc, const ValueSeq &areaInputs,
                            const ValueSeq &areaOutputs, const Location &areaLoc, OpBuilder &builder) {
  sinkAreaUses(area);
  builder.setInsertionPoint(area.back());
  auto areaCallOp = builder.create<func::CallOp>(areaLoc, areaFunc, areaInputs);
  for (auto z : llvm::zip(areaOutputs, areaCallOp.getResults())) {
    Value output = std::get<0>(z);
    Value callResult = std::get<1>(z);
    for (OpOperand &use : llvm::make_early_inc_range(output.getUses())) {
      use.set(callResult);
    }
  }
  return areaCallOp;
}

void eraseArea(const OpArea &area) {
  for (auto op : llvm::reverse(area)) {
    op->erase();
  }
}

OpArea absorbConstNode(const OpArea &area) {
  OpArea areaWithConst;
  for (Operation *op : area) {
    for (Value opnd : op->getOperands()) {
      Operation *input = opnd.getDefiningOp();
      if (input != nullptr && (isa<tosa::ConstOp>(input) || isa<mindspore::ConstOp>(input))) {
        areaWithConst.push_back(input);
      }
    }
    areaWithConst.push_back(op);
  }
  return areaWithConst;
}

func::FuncOp splitArea(const func::FuncOp func, const OpArea &area, OpBuilder &builder) {
  auto areaWithConst = absorbConstNode(area);
  auto areaInputs = getAreaInputs(areaWithConst);
  auto areaOutputs = getAreaOutputs(areaWithConst, static_cast<func::FuncOp>(func));
  auto areaLoc = getAreaLoc(areaWithConst);
  auto areaFunc =
    CreateAreaFunc(areaWithConst, areaInputs, areaOutputs, areaLoc, static_cast<func::FuncOp>(func), builder);
  (void)CreateAreaCall(areaWithConst, areaFunc, areaInputs, areaOutputs, areaLoc, builder);
  eraseArea(areaWithConst);
  return areaFunc;
}

ModelPtr getModel(func::FuncOp funcOp) {
  auto processAttr = funcOp->getAttrDictionary().get("process");
  if (processAttr != nullptr && processAttr.isa<StringAttr>()) {
    auto process = processAttr.cast<StringAttr>().str();
    if (process == kCudaProcess) {
      return std::make_shared<GpuModel>();
    } else if (process == kAicoreProcess) {
      return std::make_shared<AscendModel>();
    }
  }
  return std::make_shared<CpuModel>();
}

llvm::SmallVector<func::FuncOp> split(func::FuncOp funcOp) {
  initAllLiteOps();
  OpBuilder builder(funcOp.getContext());
  HashMap<spliter::NodePtr, Operation *> opNodeMap;
  LiteGraphPtr liteGraph = mindsporeToLiteGraph(funcOp, &opNodeMap);
  LLVM_DEBUG(llvm::dbgs() << liteGraph->toString());
  HashMap<const Operation *, size_t> nodeIdxMap;
  for (size_t i = 0; i < liteGraph->getOps().size(); ++i) {
    nodeIdxMap[opNodeMap[liteGraph->getOps()[i]]] = i;
  }
  auto model = getModel(funcOp);
  assert(model.get() != nullptr);
  model->run(liteGraph);
  auto &areas = model->getAreas();
  std::vector<OpArea> splitPlan;
  for (auto &area : areas) {
    std::vector<Operation *> nodes;
    for (auto &op : area->getOps()) {
      (void)nodes.emplace_back(opNodeMap[op]);
    }
    std::sort(nodes.begin(), nodes.end(),
              [&nodeIdxMap](const Operation *a, const Operation *b) { return nodeIdxMap[a] < nodeIdxMap[b]; });
    (void)splitPlan.emplace_back(std::move(nodes));
  }
  llvm::SmallVector<func::FuncOp> splitedFuncs;
  if (splitPlan.size() == 1) {
    splitedFuncs.push_back(funcOp);
    return splitedFuncs;
  }
  for (auto area : splitPlan) {
    splitedFuncs.push_back(splitArea(funcOp, area, builder));
  }
  return splitedFuncs;
}
}  // namespace mlir::spliter
