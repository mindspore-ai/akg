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
#include "akg/Transforms/FoldDimension.h"

#include <queue>
#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "akg/Utils/AKGGlobalVars.hpp"
#include "akg/Utils/AnalysisCommon.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
#define GEN_PASS_DEF_FOLDDIMENSION
#define GEN_PASS_DECL_FOLDDIMENSION
#include "akg/Transforms/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "fold-dimension"

// This pass analyses mindspore/tosa ops and folds axes of tensors to reduce dimensions.
// Expected results:

// Elemwise:
// Add(tensor<10, 20, 30, 40> + tensor<10, 20, 30, 40>) -> tensor<10, 20, 30, 40>
// ---->
// Add(tensor<10*20*30*40> + tensor<10*20*30*40>) -> tensor<10*20*30*40>

// Broadcast(StaticShape):
// Add(tensor<1, 20, 30, 40> + tensor<10, 20, 30, 1>) -> tensor<10, 20, 30, 40>
// ---->
// Add(tensor<1, 20*30, 40> + tensor<10, 20*30, 1>) -> tensor<10, 20*30, 40>

// Broadcast(DynamicShape):
// Add(tensor<1, 20, 30, s0> + tensor<s1, 20, 30, s2>) -> tensor<s1, 20, 30, s3>
// ---->
// Add(tensor<1, 20*30, s0> + tensor<s1, 20*30, s2>) -> tensor<s1, 20*30, s3>

// Reduce：
// ReduceSum(tensor<10, 20, 30, 40>, axis=2) -> tensor<10, 20, 1, 40>
// ---->
// ReduceSum(tensor<10*20, 30, 40>, axis=1) -> tensor<10*20, 1, 40>

constexpr auto kBroadcastInputNum = 2;
constexpr auto kLeftBroadcast = 0;
constexpr auto kRightBroadcast = 1;
constexpr auto kNoBroadcast = 2;

template <typename T>
void printVector(const llvm::SmallVector<T> info) {
  LLVM_DEBUG(llvm::dbgs() << "[");
  for (auto num : info) {
    LLVM_DEBUG(llvm::dbgs() << num << ", ");
  }
  LLVM_DEBUG(llvm::dbgs() << "]\n");
}

bool foldDimensionAnalyser::backtrackUpdateTensors(const Value &value, const llvm::SmallVector<int64_t> info) {
  std::queue<Value> updateList;
  llvm::DenseMap<Value, bool> visited;
  updateList.push(value);
  while (!updateList.empty()) {
    auto curTensor = updateList.front();
    auto oriSize = dyn_cast<ShapedType>(value.getType()).getShape().size();
    if (oriSize != 1) {
      if (oriSize != info.size()) {
        LLVM_DEBUG(llvm::dbgs() << "Unsupported different ranks before and after folding.\n");
        return false;
      }
      tensorInfoMap[curTensor].second.first = info;
    }
    visited[curTensor] = true;
    updateList.pop();
    for (auto parent : tensorInfoMap[curTensor].first.first) {
      if (visited.find(parent) == visited.end()) {
        updateList.push(parent);
      }
    }
    for (auto child : tensorInfoMap[curTensor].first.second) {
      if (visited.find(child) == visited.end()) {
        updateList.push(child);
      }
    }
  }
  return true;
}

void foldDimensionAnalyser::addOrUpdateTensorInfo(Operation *op) {
  llvm::SmallVector<int64_t> resInfo = opFoldableInfo;
  llvm::SmallDenseSet<Value> visitedInputs;
  llvm::SmallDenseSet<Value> firstVisitInputs;
  for (auto input : op->getOperands()) {
    if (tensorInfoMap.find(input) != tensorInfoMap.end()) {
      auto ret = visitedInputs.insert(input);
      if (!ret.second) {
        continue;
      }
      // compute res info = op + input
      auto parentInfo = tensorInfoMap[input].second.first;
      (void)std::transform(parentInfo.begin(), parentInfo.end(), resInfo.begin(), resInfo.begin(),
                           std::plus<int64_t>());
    } else {
      (void)firstVisitInputs.insert(input);
    }
  }
  auto seqResInfo = sequentialize(resInfo);

  auto res = op->getResult(0);

  LLVM_DEBUG(llvm::dbgs() << "-------------" << op->getName() << " start --------------\n");
  LLVM_DEBUG(res.getType().dump());
  printVector<int64_t>(opFoldableInfo);
  printVector<int64_t>(seqResInfo);
  LLVM_DEBUG(llvm::dbgs() << "-------------" << op->getName() << " end --------------\n");
  // input not visited : Add empty parents, current children and status
  for (auto item : firstVisitInputs) {
    llvm::SmallDenseSet<Value> emptyParents;
    llvm::SmallDenseSet<Value> children = {res};
    tensorInfoMap[item] = std::make_pair(std::make_pair(emptyParents, children), std::make_pair(seqResInfo, ""));
  }
  // input visited but status changed: update children, its status and its ancestors's
  for (auto item : visitedInputs) {
    if (seqResInfo != tensorInfoMap[item].second.first) {
      if (!backtrackUpdateTensors(item, seqResInfo)) {
        foldable = false;
        return;
      }
    }
    auto prevChildren = tensorInfoMap[item].first.second;
    auto ret = prevChildren.insert(res);
    if (ret.second) {
      tensorInfoMap[item].first.second = prevChildren;
    }
  }
  // output: Add parents and status
  llvm::SmallDenseSet<Value> emptyChildren;
  llvm::SmallDenseSet<Value> parents = firstVisitInputs;
  parents.insert(visitedInputs.begin(), visitedInputs.end());
  tensorInfoMap[res] = std::make_pair(std::make_pair(parents, emptyChildren),
                                      std::make_pair(seqResInfo, op->getName().getStringRef().str()));
}

void foldDimensionAnalyser::analyseElementwiseOp(Operation *op) {
  if (isa<mindspore::ConstOp>(op) || isa<tosa::ConstOp>(op)) {
    return;
  }

  auto res = op->getResult(0);
  auto resTy = dyn_cast<ShapedType>(res.getType());
  auto resRank = resTy.getShape().size();

  // current foldable status: [0,0,...,0]
  opFoldableInfo.resize(resRank, 0);
}

void foldDimensionAnalyser::analyseSymbolicBroadcastOp(const ShapedType ty0, const ShapedType ty1) {
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  auto symbolShape0 = analysis.getSymbolicShape(ty0);
  auto symbolShape1 = analysis.getSymbolicShape(ty1);
  auto shape0 = ty0.getShape();
  auto shape1 = ty1.getShape();
  // (2,3,4) * (2,3,1) = (2,3,4)
  auto inputRank = shape0.size();
  opFoldableInfo.reserve(inputRank);
  auto prevStatus = 0;
  auto curStatus = 0;
  auto foldedIndex = 0;
  constexpr auto kUnknownStatus = 3;
  // when the broadcast status changes, the folding status should change
  for (size_t i = 0; i < inputRank; i++) {
    auto bothStaticShapes = (!ShapedType::isDynamic(shape0[i]) && !ShapedType::isDynamic(shape1[i]));
    if (bothStaticShapes) {
      if (shape0[i] < shape1[i]) {
        prevStatus = kLeftBroadcast;
      } else if (shape0[i] > shape1[i]) {
        prevStatus = kRightBroadcast;
      } else {
        prevStatus = kNoBroadcast;
      }
    } else {
      if ((*symbolShape0)[i] == (*symbolShape1)[i]) {
        prevStatus = kNoBroadcast;
      } else {
        prevStatus = kUnknownStatus;
      }
    }
    if ((i > 0) && (curStatus != prevStatus || prevStatus == kUnknownStatus)) {
      foldedIndex++;
    }
    (void)opFoldableInfo.emplace_back(foldedIndex);
    curStatus = prevStatus;
  }
}

void foldDimensionAnalyser::analyseTensorCastOp(Operation *op) {
  auto input = op->getOperand(0);
  auto ty0 = dyn_cast<ShapedType>(input.getType());
  auto res = op->getResult(0);
  auto ty1 = dyn_cast<ShapedType>(res.getType());
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (!(analysis.hasSymbolicShape(ty0) && analysis.hasSymbolicShape(ty1))) {
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                               "] Unsupported when only one of 2 inputs has symbolic shape while dynamic shape in '"
                            << op->getName() << "'.\n");
    return;
  }
  analyseSymbolicBroadcastOp(ty0, ty1);
}

void foldDimensionAnalyser::analyseElemwiseBroadcastOp(Operation *op) {
  assert(op->getNumOperands() == kBroadcastInputNum);

  auto ty0 = dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto shape0 = ty0.getShape();
  auto ty1 = dyn_cast<ShapedType>(op->getOperand(1).getType());
  auto shape1 = ty1.getShape();
  auto isInput0Const = llvm::all_of(shape0, [](int64_t dim) -> bool { return dim == 1; });
  auto isInput1Const = llvm::all_of(shape1, [](int64_t dim) -> bool { return dim == 1; });
  if (isInput0Const || isInput1Const) {
    auto inputRank = isInput0Const ? shape1.size() : shape0.size();
    // pure 1 broadcast: (1) * (2,3) = (2,3) or (2,3) * (1,1) = (2,3)
    opFoldableInfo.resize(inputRank, 0);
  } else if (shape0.size() != shape1.size()) {
    // todo(zuohe): not support different rank broadcast
    // (2,3,4) * (2,3) = (2,3,4)
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported broadcast between different ranks in '" << op->getName()
                            << "'.\n");
    return;
  } else {
    assert(shape0.size() == shape1.size());
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    if (analysis.hasSymbolicShape(ty0) || analysis.hasSymbolicShape(ty1)) {
      if (!(analysis.hasSymbolicShape(ty0) && analysis.hasSymbolicShape(ty1))) {
        foldable = false;
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE
                                   "] Unsupported when only one of 2 inputs has symbolic shape while dynamic shape in '"
                                << op->getName() << "'.\n");
      }
      analyseSymbolicBroadcastOp(ty0, ty1);
    } else {
      // (2,3,4) * (2,3,1) = (2,3,4)
      auto inputRank = shape0.size();
      opFoldableInfo.reserve(inputRank);
      auto prevStatus = 0;
      auto curStatus = 0;
      auto foldedIndex = 0;
      // when the broadcast status changes, the folding status should change
      for (size_t i = 0; i < inputRank; i++) {
        if (shape0[i] < shape1[i]) {
          prevStatus = kLeftBroadcast;
        } else if (shape0[i] > shape1[i]) {
          prevStatus = kRightBroadcast;
        } else {
          prevStatus = kNoBroadcast;
        }
        if ((i > 0) && (curStatus != prevStatus)) {
          foldedIndex++;
        }
        (void)opFoldableInfo.emplace_back(foldedIndex);
        curStatus = prevStatus;
      }
    }
  }
}

void foldDimensionAnalyser::analyseBroadcastToOp(Operation *op) {
  assert(op->getNumOperands() == 1);

  auto inputTy = dyn_cast<ShapedType>(op->getOperand(0).getType());

  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (analysis.hasSymbolicShape(inputTy)) {
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported dynamic shape in '" << op->getName() << "'.\n");
    return;
  }

  auto inputShape = inputTy.getShape();
  auto inputRank = inputShape.size();
  auto res = op->getResult(0);
  auto resTy = dyn_cast<ShapedType>(res.getType());
  auto resShape = resTy.getShape();
  if (inputRank != resShape.size()) {
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported different ranks between input and output in '"
                            << op->getName() << "'.\n");
    return;
  }

  opFoldableInfo.reserve(inputRank);
  auto prevStatus = 0;
  auto curStatus = 0;
  auto foldedIndex = 0;
  for (size_t i = 0; i < inputRank; i++) {
    curStatus = int(inputShape[i] == resShape[i]);
    if (i > 0 && prevStatus != curStatus) {
      foldedIndex++;
    }
    (void)opFoldableInfo.emplace_back(foldedIndex);
    prevStatus = curStatus;
  }
}

void foldDimensionAnalyser::analyseReduceOp(Operation *op) {
  auto inputTy = dyn_cast<ShapedType>(op->getOperand(0).getType());
  auto inputRank = inputTy.getRank();
  auto res = op->getResult(0);
  auto resRank = dyn_cast<ShapedType>(res.getType()).getRank();
  if (inputRank != resRank) {
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported different ranks between input and output in '"
                            << op->getName() << "'.\n");
    return;
  }

  llvm::SmallVector<int64_t> axes;
  auto axis_attr = op->getAttr("axis");
  if (auto axis_array_attr = dyn_cast<DenseI64ArrayAttr>(axis_attr)) {
    auto axis_array = axis_array_attr.asArrayRef();
    (void)axes.insert(axes.end(), axis_array.begin(), axis_array.end());
  } else if (auto axis_int_attr = dyn_cast<IntegerAttr>(axis_attr)) {
    (void)axes.emplace_back(axis_int_attr.getInt());
  }

  // current foldable status: axes are seperated by reduction axes
  auto foldedIndex = 0;
  size_t axesIndex = 0;
  auto prevStatus = 0;
  auto curStatus = 0;
  // 0 = non-reduce, 1 = reduce
  // when the status changes, the folding status should change
  opFoldableInfo.reserve(inputTy.getShape().size());
  for (auto i = 0; i < inputRank; i++) {
    if (axesIndex < axes.size() && i == axes[axesIndex]) {
      prevStatus = 1;
      axesIndex++;
    } else {
      prevStatus = 0;
    }
    if (i > 0 && curStatus != prevStatus) {
      foldedIndex++;
    }
    (void)opFoldableInfo.emplace_back(foldedIndex);
    curStatus = prevStatus;
  }
}

void foldDimensionAnalyser::analyseReshapeOp(Operation *op) {
  auto inputTy = dyn_cast<ShapedType>(op->getOperand(0).getType());

  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  if (analysis.hasSymbolicShape(inputTy)) {
    foldable = false;
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported dynamic shape in '" << op->getName() << "'.\n");
    return;
  }

  auto res = op->getResult(0);
  auto input = op->getOperand(0);
  if (tensorInfoMap.find(input) != tensorInfoMap.end()) {
    // input visited: only update children
    auto prevChildren = tensorInfoMap[input].first.second;
    auto ret = prevChildren.insert(res);
    if (ret.second) {
      tensorInfoMap[input].first.second = prevChildren;
    }
  } else {
    // input not visited: add children and status = [0,0,..,0] length equal to input dim
    llvm::SmallDenseSet<Value> emptyParents;
    llvm::SmallDenseSet<Value> children = {res};
    auto inputRank = dyn_cast<ShapedType>(input.getType()).getRank();
    llvm::SmallVector<int64_t> foldableInfo(inputRank, 0);
    tensorInfoMap[input] = std::make_pair(std::make_pair(emptyParents, children), std::make_pair(foldableInfo, ""));
  }
  // output: Add parents and status = [0,0,..,0] length equal to output dim
  llvm::SmallDenseSet<Value> parents{input};
  llvm::SmallDenseSet<Value> emptyChildren;
  auto resRank = dyn_cast<ShapedType>(res.getType()).getRank();
  llvm::SmallVector<int64_t> foldableInfo(resRank, 0);
  tensorInfoMap[res] = std::make_pair(std::make_pair(parents, emptyChildren), std::make_pair(foldableInfo, "Reshape"));
}

bool foldDimensionAnalyser::checkBroadcast(Operation *op) const {
  bool isBroadcast = false;
  assert(op->getNumResults() == 1 && "Elementwise ops should only return one result.");
  auto resultTy = dyn_cast<ShapedType>(op->getResult(0).getType());
  for (Value operand : op->getOperands()) {
    ShapedType type = cast<ShapedType>(operand.getType());
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    bool hasDifferentSymbolShape = (analysis.hasSymbolicShape(type) && analysis.hasSymbolicShape(resultTy) &&
                                    !analysis.isSameSymbolicShape(type, resultTy));
    if ((type.getShape() != resultTy.getShape()) ||
        (op->getNumResults() == kBroadcastInputNum && hasDifferentSymbolShape)) {
      isBroadcast = true;
    }
  }
  return isBroadcast;
}

void foldDimensionAnalyser::analyseFoldDimension(const func::FuncOp funcOp) {
  funcOp->walk([&](Operation *op) {
    this->opFoldableInfo.clear();

    if (!(isa<tosa::TosaOp>(op) || isa<mindspore::MindSporeOp>(op) || isa<tensor::CastOp>(op))) {
      return;
    }

    if (isa<tensor::CastOp>(op)) {
      // special op in dynamic shape scenarios to express implicit broadcast
      analyseTensorCastOp(op);
    } else if (TosaOperatorType::isTosaElementwiseOp(op) || MindOperatorType::isMindElementwiseOp(op)) {
      if (checkBroadcast(op)) {
        analyseElemwiseBroadcastOp(op);
      } else {
        analyseElementwiseOp(op);
      }
    } else if (TosaOperatorType::isTosaReduceOp(op) || MindOperatorType::isMindReduceOp(op)) {
      analyseReduceOp(op);
    } else if (isa<tosa::ReshapeOp>(op) || isa<mindspore::ReshapeOp>(op)) {
      analyseReshapeOp(op);
    } else {
      foldable = false;
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported operator '" << op->getName() << "'.\n");
    }

    if (!foldable || this->opFoldableInfo.size() == 0) {
      return;
    }

    addOrUpdateTensorInfo(op);
  });
}

llvm::SmallVector<int64_t> foldDimensionAnalyser::getNormalizedFlattenShape(const Value &value) {
  // get the normalized flattened shape before reshape, to be compared with the shape after reshape
  auto info = tensorInfoMap[value].second.first;
  LLVM_DEBUG(value.getType().dump());
  printVector<int64_t>(info);
  auto shape = dyn_cast<ShapedType>(value.getType()).getShape();
  auto rank = shape.size();
  llvm::SmallVector<int64_t> normalizedShape;
  int64_t currentDim;
  for (size_t i = 0; i < rank; i++) {
    if (i == 0) {
      currentDim = shape[0];
    } else {
      // consecutive axes can be folded
      if (info[i] == info[i - 1]) {
        currentDim *= shape[i];
      } else {
        if (currentDim != 1) {
          (void)normalizedShape.emplace_back(currentDim);
        }
        currentDim = shape[i];
      }
    }
    if (i == rank - 1 && currentDim != 1) {
      (void)normalizedShape.emplace_back(currentDim);
    }
  }
  return normalizedShape;
}

void foldDimensionAnalyser::recordFuncArgs(func::FuncOp funcOp) {
  // if func args get folded, their needFixIndex and DeviceShape need to be updated
  // record all func args, and their new needFixIndex and Shape while recordTensorToBeFolded
  akgglobal::ShapeAlignTool &tool = akgglobal::ShapeAlignTool::getInstance();
  size_t argIdx = 0;
  for (auto value : funcOp.getBody().front().getArguments()) {
    funcArgsMap[value] = tool.getNeedFixIndice(argIdx);
    argIdx++;
  }
}

void foldDimensionAnalyser::updatefuncArgsMap(const Value &input, const llvm::SmallVector<int64_t> foldableInfo) {
  auto it = funcArgsMap.find(input);
  if (it == funcArgsMap.end()) {
    return;
  }
  auto inputRank = dyn_cast<ShapedType>(input.getType()).getShape().size();
  auto needFixIndices = it->second;
  int64_t needFix;
  llvm::SmallVector<int64_t> newNeedFixIndices;
  for (size_t i = 0; i < inputRank; i++) {
    if (i == 0) {
      needFix = needFixIndices[0];
    } else {
      if (foldableInfo[i] == foldableInfo[i - 1]) {
        needFix = needFixIndices[i] == 1 ? 1 : needFix;
      } else {
        (void)newNeedFixIndices.emplace_back(needFix);
        needFix = needFixIndices[i];
      }
    }
    if (i == inputRank - 1) {
      (void)newNeedFixIndices.emplace_back(needFix);
    }
  }
  it->second = newNeedFixIndices;
}

void foldDimensionAnalyser::getFoldedTypeWithSymbol(SymbolicShapeAnalysis &analysis, const ShapedType inputTy,
                                                    const llvm::SmallVector<int64_t> foldableInfo,
                                                    llvm::SmallVector<int64_t> *flattenedShape,
                                                    llvm::SmallVector<std::string> *flattenedSymbolShape) const {
  llvm::SmallVector<std::string> symbolShape = *analysis.getSymbolicShape(inputTy);
  std::string currentSymbol;
  auto shape = inputTy.getShape();
  auto currentDim = (inputTy.isDynamicDim(0)) ? ShapedType::kDynamic : shape[0];
  auto inputRank = shape.size();
  for (size_t i = 0; i < inputRank; i++) {
    if (i == 0) {
      currentSymbol = symbolShape[0];
    } else {
      // consecutive axes can be folded
      if (foldableInfo[i] == foldableInfo[i - 1]) {
        if (!inputTy.isDynamicDim(static_cast<unsigned int>(i)) && currentDim != ShapedType::kDynamic) {
          currentDim *= shape[i];
        } else {
          currentDim = ShapedType::kDynamic;
        }
        currentSymbol += "*" + symbolShape[i];
        SymEngine::Expression expr(currentSymbol);
        currentSymbol = analysis.getSymbolicDimFromExpression(expr);
      } else {
        (void)flattenedShape->emplace_back(currentDim);
        (void)flattenedSymbolShape->emplace_back(currentSymbol);
        currentDim = shape[i];
        currentSymbol = symbolShape[i];
      }
    }
    if (i == inputRank - 1) {
      (void)flattenedShape->emplace_back(currentDim);
      (void)flattenedSymbolShape->emplace_back(currentSymbol);
    }
  }
}

void foldDimensionAnalyser::getFoldedType(const ShapedType inputTy, const llvm::SmallVector<int64_t> foldableInfo,
                                          llvm::SmallVector<int64_t> *flattenedShape,
                                          llvm::SmallVector<int64_t> *normalizedShapeAfter) const {
  auto shape = inputTy.getShape();
  int64_t currentDim;
  auto inputRank = shape.size();
  for (size_t i = 0; i < inputRank; i++) {
    if (i == 0) {
      currentDim = shape[0];
    } else {
      // consecutive axes can be folded
      if (foldableInfo[i] == foldableInfo[i - 1]) {
        currentDim *= shape[i];
      } else {
        (void)flattenedShape->emplace_back(currentDim);
        if (currentDim != 1) {
          (void)normalizedShapeAfter->emplace_back(currentDim);
        }
        currentDim = shape[i];
      }
    }
    if (i == inputRank - 1) {
      (void)flattenedShape->emplace_back(currentDim);
      if (currentDim != 1) {
        (void)normalizedShapeAfter->emplace_back(currentDim);
      }
    }
  }
}

void foldDimensionAnalyser::recordTensorCanFold() {
  LLVM_DEBUG(llvm::dbgs() << "============MAP=========== \n");
  for (auto pair : tensorInfoMap) {
    LLVM_DEBUG(llvm::dbgs() << "key:");
    LLVM_DEBUG(pair.first.getType().dump());
    printVector<int64_t>(pair.second.second.first);
    LLVM_DEBUG(llvm::dbgs() << "type = " << pair.second.second.second << "\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "======================= \n");
  // for each tensor in the map, record their new folded types
  for (auto item : tensorInfoMap) {
    auto input = item.first;
    // skip already recorded
    if (tensorToBeFolded.find(input) != tensorToBeFolded.end()) {
      continue;
    }

    auto foldableInfo = item.second.second.first;
    auto operatorType = item.second.second.second;
    auto inputTy = dyn_cast<ShapedType>(input.getType());
    auto inputRank = inputTy.getShape().size();
    if (!inputTy || inputRank == 1 || inputRank < foldableInfo.size()) {
      continue;
    }

    // original shape (10x20x1x30x1x40)
    // foldableInfo = [0,0,1,2,2,3]
    // flattenedShape (200x1x30x40)
    Type newTy;
    llvm::SmallVector<int64_t> flattenedShape;
    SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
    // calculate folded shape
    if (analysis.hasSymbolicShape(inputTy)) {
      llvm::SmallVector<std::string> flattenedSymbolShape;
      getFoldedTypeWithSymbol(analysis, inputTy, foldableInfo, &flattenedShape, &flattenedSymbolShape);
      // If the innermost axis is static and >= threshold, CANNOT be folded with a dynamic axis.
      // Because dynamic innermost axis cannot be vectorized.
      auto shape = inputTy.getShape();
      auto originalLastIdx = shape.size() - 1;
      auto foldedLastIdx = flattenedShape.size() - 1;
      constexpr auto kMinInnerShape = 256;
      if (!inputTy.isDynamicDim(static_cast<unsigned int>(originalLastIdx)) &&
          shape[originalLastIdx] >= kMinInnerShape && (flattenedShape[foldedLastIdx] == ShapedType::kDynamic)) {
        foldable = false;
        return;
      }
      // calculate folded needFixIndices and record for funcArgs
      updatefuncArgsMap(input, foldableInfo);
      // record the new shape type for each tensor can be folded
      newTy = RankedTensorType::get(flattenedShape, inputTy.getElementType());
      newTy = dyn_cast<RankedTensorType>(analysis.updateSymbolicShape(newTy, flattenedSymbolShape));
    } else {
      llvm::SmallVector<int64_t> normalizedShapeAfter;
      getFoldedType(inputTy, foldableInfo, &flattenedShape, &normalizedShapeAfter);
      // For Reshape op, only fold when the normalized shapes (remove 1) of input and output are the same
      if (operatorType == "Reshape") {
        Value parent = *(item.second.first.first.begin());  // reshape only has one parent
        auto normalizedShapeBefore = getNormalizedFlattenShape(parent);
        // CAN be folded: Reshape (1x20x30x1) to (20x1x30x1), both are (20x30) after normalized (remove 1)
        // CANNOT be folded: Reshape (20x30) to (600) will generate expand/collapse ops and block loop fusion
        if (normalizedShapeAfter != normalizedShapeBefore) {
          foldable = false;
          return;
        }
      }
      // record the new shape type for each tensor can be folded
      newTy = RankedTensorType::get(flattenedShape, inputTy.getElementType());
    }

    tensorToBeFolded[input] = newTy;
  }
}

namespace {
void populateFoldDimension(func::FuncOp funcOp, llvm::DenseMap<Value, Type> tensorToBeFolded, FuncArgsMap funcArgsMap) {
  LLVM_DEBUG(llvm::dbgs() << "--------------tensorToBeFolded----------------");
  for (auto item : tensorToBeFolded) {
    LLVM_DEBUG(item.first.dump());
    LLVM_DEBUG(item.second.dump());
  }
  LLVM_DEBUG(llvm::dbgs() << "----------------------------------------------");
  // update func type
  akgglobal::ShapeAlignTool &tool = akgglobal::ShapeAlignTool::getInstance();
  SymbolicShapeAnalysis &analysis = SymbolicShapeAnalysis::getInstance();
  llvm::DenseMap<Value, bool> visited;
  size_t argIdx = 0;
  llvm::SmallVector<Type> newInTys;
  std::vector<std::string> newShape;
  for (auto value : funcOp.getBody().front().getArguments()) {
    if (tensorToBeFolded.find(value) != tensorToBeFolded.end()) {
      auto newTy = tensorToBeFolded[value];
      value.setType(newTy);
      if (analysis.getSymbolicShape(newTy)) {
        newShape.clear();
        llvm::SmallVector<std::string> symbolShape = *analysis.getSymbolicShape(newTy);
        (void)std::transform(symbolShape.begin(), symbolShape.end(), std::back_inserter(newShape),
                             [](std::string s) { return s; });
        tool.updateCurrShapeInfo(argIdx, newShape);
      }
      tool.recordNeedFixIndice(argIdx, funcArgsMap[value]);
      visited[value] = true;
    }
    (void)newInTys.emplace_back(value.getType());
    argIdx++;
  }
  llvm::SmallVector<Type> newResTys;
  funcOp.walk([&](func::ReturnOp op) {
    for (auto value : op.getOperation()->getOperands()) {
      if (tensorToBeFolded.find(value) != tensorToBeFolded.end()) {
        auto newTy = tensorToBeFolded[value];
        value.setType(newTy);
        if (analysis.hasSymbolicShape(newTy)) {
          llvm::SmallVector<std::string> symbolShape = *analysis.getSymbolicShape(newTy);
          newShape.clear();
          (void)std::transform(symbolShape.begin(), symbolShape.end(), std::back_inserter(newShape),
                               [](std::string s) { return s; });
          tool.updateCurrShapeInfo(argIdx, newShape);
        }
        tool.recordNeedFixIndice(argIdx, funcArgsMap[value]);
        visited[value] = true;
      }
      (void)newResTys.emplace_back(value.getType());
      argIdx++;
    }
  });
  // update all other tensors
  for (auto item : tensorToBeFolded) {
    auto value = item.first;
    if (visited[value]) {
      continue;
    }
    auto newTy = item.second;
    value.setType(newTy);
  }
  auto newFuncTy = mlir::FunctionType::get(funcOp.getContext(), newInTys, newResTys);
  funcOp.setType(newFuncTy);
}

template <typename T>
void rewriteConstOpAttr(T constOp) {
  OpBuilder builder(constOp);
  builder.setInsertionPoint(constOp);
  auto outTy = dyn_cast<ShapedType>(constOp.getResult().getType());
  if (!outTy) {
    return;
  }
  ElementsAttr elemAttr = dyn_cast<ElementsAttr>(constOp.getValue());
  if (!elemAttr) {
    return;
  }
  auto elemAttrValues = elemAttr.getValues<Attribute>();
  llvm::SmallVector<Attribute> outValues;
  (void)std::transform(elemAttrValues.begin(), elemAttrValues.end(), std::back_inserter(outValues),
                       [](const Attribute attr) { return attr; });
  auto newConstOp = builder.create<T>(constOp.getLoc(), outTy, DenseElementsAttr::get(outTy, outValues));
  constOp.getOperation()->replaceAllUsesWith(newConstOp.getOperation());
  constOp.erase();
}

void rewriteTosaReduceAxis(tosa::TosaOp op) {
  auto inputShape = dyn_cast<ShapedType>(op->getOperand(0).getType()).getShape();
  auto outputShape = dyn_cast<ShapedType>(op->getResult(0).getType()).getShape();
  int64_t newAxis = -1;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (outputShape[i] == 1 && inputShape[i] != outputShape[i]) {
      newAxis = static_cast<int64_t>(i);
      break;
    }
  }
  assert(newAxis >= 0);
  auto axisAttr = IntegerAttr::get(IntegerType::get(op.getContext(), 64), newAxis);
  op->setAttr("axis", axisAttr);
}

void rewriteMindReduceAxis(mindspore::MindSporeOp op) {
  auto inputShape = dyn_cast<ShapedType>(op->getOperand(0).getType()).getShape();
  auto outputShape = dyn_cast<ShapedType>(op->getResult(0).getType()).getShape();
  assert(inputShape.size() == outputShape.size());
  llvm::SmallVector<int64_t> newAxes;
  for (size_t i = 0; i < inputShape.size(); i++) {
    if (outputShape[i] == 1 && inputShape[i] != outputShape[i]) {
      (void)newAxes.emplace_back(static_cast<int64_t>(i));
      break;
    }
  }
  assert(newAxes.size() > 0);
  auto axisAttr = DenseI64ArrayAttr::get(op.getContext(), newAxes);
  op->setAttr("axis", axisAttr);
}

void rewriteReshapeAttr(Operation *op) {
  auto outputShape = dyn_cast<ShapedType>(op->getResult(0).getType()).getShape();
  auto newShapeAttr = DenseI64ArrayAttr::get(op->getContext(), outputShape);
  op->setAttr("new_shape", newShapeAttr);
}

void rewriteOpAttr(func::FuncOp funcOp) {
  funcOp->walk([&](Operation *op) {
    if (isa<tosa::ConstOp>(op)) {
      rewriteConstOpAttr<tosa::ConstOp>(dyn_cast<tosa::ConstOp>(op));
    } else if (isa<mindspore::ConstOp>(op)) {
      rewriteConstOpAttr<mindspore::ConstOp>(dyn_cast<mindspore::ConstOp>(op));
    } else if (MindOperatorType::isMindReduceOp(op)) {
      rewriteMindReduceAxis(dyn_cast<mindspore::MindSporeOp>(op));
    } else if (TosaOperatorType::isTosaReduceOp(op)) {
      rewriteTosaReduceAxis(dyn_cast<tosa::TosaOp>(op));
    } else if (isa<mindspore::ReshapeOp>(op) || isa<tosa::ReshapeOp>(op)) {
      rewriteReshapeAttr(op);
    }
  });
}

struct FoldDimension : public impl::FoldDimensionBase<FoldDimension> {
  FoldDimension() {}
  void runOnOperation() override {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOp();
    if (moduleOp->hasAttr("mindspore.symbol_calc_expr"))
      return;
    auto opTypeAttr = funcOp->getAttrOfType<StringAttr>("OperatorType");
    if (opTypeAttr == nullptr) {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported without attr 'OperatorType'.\n");
      return;
    }
    auto opTypeValue = opTypeAttr.getValue();
    if (opTypeValue != "Elementwise" && opTypeValue != "Broadcast" && opTypeValue != "Reduce" &&
        opTypeValue != "Reshape") {
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] Unsupported OperatorType '" << opTypeValue << "'.\n");
      return;
    }
    // analyse and collect foldable information
    foldDimensionAnalyser analyser;
    analyser.analyseFoldDimension(funcOp);
    // Not to fold if has unsupported operator types, different rank broadcast and reduce ops
    if (!analyser.foldable) {
      return;
    }
    analyser.recordFuncArgs(funcOp);
    analyser.recordTensorCanFold();
    // Not to fold if has ReshapeOp really modifying shapes
    if (!analyser.foldable) {
      return;
    }

    // modify tensor shapes according to their foldable info
    populateFoldDimension(funcOp, analyser.tensorToBeFolded, analyser.funcArgsMap);

    // rewrite op attrs according to updated outputs
    rewriteOpAttr(funcOp);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::createFoldDimensionPass() {
  return std::make_unique<FoldDimension>();
}
