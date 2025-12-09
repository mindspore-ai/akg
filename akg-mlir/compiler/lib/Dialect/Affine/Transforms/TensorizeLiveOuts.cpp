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

#include "akg/Dialect/Affine/Transforms/TensorizeLiveOuts.h"

#include <cstdint>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tensorize-live-outs"

using namespace mlir;  // NOLINT(build/namespaces)

namespace affine = mlir::affine;
namespace tensor = mlir::tensor;
namespace func   = mlir::func;
namespace arith  = mlir::arith;

static mlir::LogicalResult propagateTensorLiveOutsToParentLoops(
  func::FuncOp functionOperation);
static mlir::LogicalResult fixInnerLoopInitOperandsFromOuterLoops(
  func::FuncOp functionOperation);

// Used to mark tensor live-out iter/result indices processed by this pass.
// The attribute is attached to AffineForOp itself: DenseI64ArrayAttr,
// content is a list of iter indices.
static constexpr const char *tensorizeLiveOutAttributeName =
    "tensorize.liveout_indices";

static inline bool isBeforeInSameBlock(Operation *firstOperation,
                                        Operation *secondOperation) {
  return firstOperation && secondOperation &&
         firstOperation->getBlock() == secondOperation->getBlock() &&
         firstOperation->isBeforeInBlock(secondOperation);
}

static inline bool isAncestorOperation(Operation *maybeAncestorOperation,
                                        Operation *operation) {
  return maybeAncestorOperation && operation &&
         maybeAncestorOperation->isAncestor(operation);
}

static inline bool isProperAncestorOperation(
  Operation *maybeAncestorOperation, Operation *operation) {
  return maybeAncestorOperation && operation &&
         maybeAncestorOperation->isProperAncestor(operation);
}

// Read tensorize iter indices stored on the given for operation.
static inline mlir::DenseI64ArrayAttr getTensorizeIterAttribute(
    affine::AffineForOp affineForOperation) {
  return affineForOperation->getAttrOfType<mlir::DenseI64ArrayAttr>(
      tensorizeLiveOutAttributeName);
}

// Check whether the given iter index is marked as tensor live-out
// by this pass.
static inline bool isTensorizeIterIndex(
    affine::AffineForOp affineForOperation, unsigned iterIndex) {
  auto attribute = getTensorizeIterAttribute(affineForOperation);
  if (!attribute) { return false; }
  auto arrayRef = attribute.asArrayRef();
  return std::any_of(arrayRef.begin(), arrayRef.end(),
                     [iterIndex](int64_t value) {
                       return value == static_cast<int64_t>(iterIndex);
                     });
}

// Check whether the given result index is tensorize live-out
// (result index corresponds to iter index).
static inline bool isTensorizeLiveOutResult(
    affine::AffineForOp affineForOperation, unsigned resultIndex) {
  return isTensorizeIterIndex(affineForOperation, resultIndex);
}

// Set or extend tensorize iter indices on newForOperation.
// Take indices from oldForOperation and append newIndices,
// then write back to newForOperation.
static inline void appendTensorizeIterIndices(
    affine::AffineForOp oldForOperation,
    affine::AffineForOp newForOperation,
    llvm::ArrayRef<int64_t> newIndices) {
  llvm::SmallVector<int64_t, 8> mergedIndices;
  if (auto oldAttribute = getTensorizeIterAttribute(oldForOperation)) {
    auto arrayRef = oldAttribute.asArrayRef();
    mergedIndices.insert(mergedIndices.end(), arrayRef.begin(), arrayRef.end());
  }
  mergedIndices.append(newIndices.begin(), newIndices.end());
  if (!mergedIndices.empty()) {
    auto context = newForOperation.getContext();
    auto arrayAttribute =
        mlir::DenseI64ArrayAttr::get(context, mergedIndices);
    newForOperation->setAttr(tensorizeLiveOutAttributeName, arrayAttribute);
  }
}

static Value findOriginalBase(Value destinationValue) {
  Value currentValue = destinationValue;
  while (auto blockArgument = dyn_cast<BlockArgument>(currentValue)) {
    Operation *parentOperation = blockArgument.getOwner()->getParentOp();
    if (auto parentLoopOperation = dyn_cast<affine::AffineForOp>(parentOperation)) {
      unsigned argumentIndex = blockArgument.getArgNumber();
      unsigned numberOfInductionVariables = 1;
      if (argumentIndex >= numberOfInductionVariables) {
        unsigned initIndex = argumentIndex - numberOfInductionVariables;
        if (initIndex < parentLoopOperation.getNumIterOperands()) {
          unsigned operandIndex = 2 + initIndex;
          if (operandIndex <
              parentLoopOperation->getNumOperands()) {
            currentValue = parentLoopOperation->getOperand(operandIndex);
            continue;
          }
        }
      }
    }
    break;
  }
  return currentValue;
}

static Value findOriginalBaseRecursive(Value destinationValue) {
  Value currentValue = destinationValue;
  llvm::DenseSet<Value> visitedValues;
  while (true) {
    if (visitedValues.contains(currentValue)) { break; }
    visitedValues.insert(currentValue);
    if (auto blockArgument = dyn_cast<BlockArgument>(currentValue)) {
      Operation *parentOperation = blockArgument.getOwner()->getParentOp();
      if (auto parentLoopOperation = dyn_cast<affine::AffineForOp>(parentOperation)) {
        unsigned argumentIndex = blockArgument.getArgNumber();
        unsigned numberOfInductionVariables = 1;
        if (argumentIndex >= numberOfInductionVariables) {
          unsigned initIndex =
              argumentIndex - numberOfInductionVariables;
          if (initIndex < parentLoopOperation.getNumIterOperands()) {
            unsigned totalNumberOfOperands =
                parentLoopOperation.getNumOperands();
            unsigned numberOfIterOperands =
                parentLoopOperation.getNumIterOperands();
            unsigned operandIndex =
                totalNumberOfOperands - numberOfIterOperands + initIndex;
            if (operandIndex < totalNumberOfOperands) {
              currentValue = parentLoopOperation->getOperand(operandIndex);
              continue;
            }
          }
        }
      }
      break;
    }
    if (auto definingOperation = currentValue.getDefiningOp()) {
      if (auto forOperation = dyn_cast<affine::AffineForOp>(definingOperation)) {
        unsigned numberOfResults = forOperation.getNumResults();
        for (unsigned resultIndex = 0; resultIndex < numberOfResults;
             ++resultIndex) {
          if (forOperation.getResult(resultIndex) == currentValue) {
            unsigned numberOfIterOperands =
                forOperation.getNumIterOperands();
            if (resultIndex < numberOfIterOperands) {
              unsigned totalNumberOfOperands =
                  forOperation->getNumOperands();
              unsigned numberOfIterOperandsLocal = numberOfIterOperands;
              unsigned operandIndex = totalNumberOfOperands -
                                      numberOfIterOperandsLocal +
                                      resultIndex;
              if (operandIndex < totalNumberOfOperands) {
                currentValue = forOperation->getOperand(operandIndex);
                continue;
              }
            }
            break;
          }
        }
      }
    }
    break;
  }
  return currentValue;
}

static void collectInsertSliceOpsInLoop(
    affine::AffineForOp affineForOperation,
    llvm::SmallVectorImpl<mlir::Operation *> &insertSliceOperations) {
  insertSliceOperations.clear();
  mlir::Block *bodyBlock = affineForOperation.getBody();
  for (mlir::Operation &operation : bodyBlock->getOperations()) {
    if (mlir::isa<affine::AffineForOp>(&operation)) { continue; }
    if (auto insertSliceOperation =
            mlir::dyn_cast<tensor::InsertSliceOp>(&operation)) {
      insertSliceOperations.push_back(&operation);
    }
  }
  llvm::sort(
      insertSliceOperations,
      [](mlir::Operation *firstOperation,
         mlir::Operation *secondOperation) {
        if (firstOperation->getBlock() == secondOperation->getBlock()) {
          return firstOperation->isBeforeInBlock(secondOperation);
        }
        return firstOperation->getBlock() < secondOperation->getBlock();
      });
}

static mlir::FailureOr<affine::AffineForOp>
extendLoopWithInsertSliceLiveOuts(
    affine::AffineForOp affineForOperation,
    llvm::ArrayRef<mlir::Operation *> insertSliceOperations,
    llvm::SmallVectorImpl<mlir::Value> &newlyAddedResults,
    llvm::DenseMap<mlir::Operation *, mlir::Value>
        &insertSliceToYieldedResultMap,
    llvm::SmallVectorImpl<mlir::Value> &originalBaseValues,
    llvm::ArrayRef<mlir::Value> customInitOperands = {}) {
  newlyAddedResults.clear();
  insertSliceToYieldedResultMap.clear();
  originalBaseValues.clear();

  if (insertSliceOperations.empty()) { return affineForOperation; }

  llvm::SmallVector<mlir::Value, 8> initOperands;
  initOperands.reserve(insertSliceOperations.size());
  originalBaseValues.reserve(insertSliceOperations.size());

  if (!customInitOperands.empty() &&
      customInitOperands.size() == insertSliceOperations.size()) {
    initOperands.assign(customInitOperands.begin(), customInitOperands.end());
    for (mlir::Operation *operation : insertSliceOperations) {
      auto insertSliceOperation =
        mlir::cast<tensor::InsertSliceOp>(operation);
      originalBaseValues.push_back(
        findOriginalBase(insertSliceOperation.getDest()));
    }
  } else {
    for (mlir::Operation *operation : insertSliceOperations) {
      auto insertSliceOperation =
          mlir::cast<tensor::InsertSliceOp>(operation);
      mlir::Value originalBaseValue =
          findOriginalBase(insertSliceOperation.getDest());
      initOperands.push_back(originalBaseValue);
      originalBaseValues.push_back(originalBaseValue);
    }
  }

  mlir::PatternRewriter rewriter(affineForOperation.getContext());
  rewriter.setInsertionPoint(affineForOperation);

  unsigned oldNumberOfResults = affineForOperation.getNumResults();
  unsigned oldNumberOfBodyArguments =
      affineForOperation.getBody()->getNumArguments();
  unsigned oldNumberOfIterOperands =
      affineForOperation.getNumIterOperands();
  unsigned numberOfInserted = insertSliceOperations.size();

  mlir::Region *loopRegion = &affineForOperation.getRegion();

  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *>>
      baseToInsertSliceOperations;
  for (unsigned index = 0; index < numberOfInserted; ++index) {
    mlir::Operation *operation = insertSliceOperations[index];
    auto insertSliceOperation =
        mlir::cast<tensor::InsertSliceOp>(operation);
    mlir::Value baseValue = initOperands[index];
    baseToInsertSliceOperations[baseValue].push_back(operation);
  }

  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::OpOperand *>>
    baseToUsesInLoopMap;
  for (unsigned index = 0; index < numberOfInserted; ++index) {
    mlir::Value baseValue = initOperands[index];
    llvm::SmallVector<mlir::OpOperand *> usesInLoop;
    for (mlir::OpOperand &use : baseValue.getUses()) {
      mlir::Operation *userOperation = use.getOwner();
      if (userOperation->getParentRegion() == loopRegion) {
        usesInLoop.push_back(&use);
      }
    }
    if (!usesInLoop.empty()) {
      baseToUsesInLoopMap[baseValue] = std::move(usesInLoop);
    }
  }

  auto newLoopOr = affineForOperation.replaceWithAdditionalYields(
      rewriter, initOperands, false,
      [&](mlir::OpBuilder &, mlir::Location,
          llvm::ArrayRef<mlir::BlockArgument>) {
        llvm::SmallVector<mlir::Value, 8> yields;
        yields.reserve(insertSliceOperations.size());
        std::transform(insertSliceOperations.begin(), insertSliceOperations.end(),
                       std::back_inserter(yields),
                       [](mlir::Operation *operation) {
                         return mlir::cast<tensor::InsertSliceOp>(operation).getResult();
                       });
        return yields;
      });

  if (mlir::failed(newLoopOr)) { return mlir::failure(); }

  affine::AffineForOp newLoopOperation =
      mlir::cast<affine::AffineForOp>(*newLoopOr);
  newlyAddedResults.reserve(numberOfInserted);

  mlir::Block *newBodyBlock = newLoopOperation.getBody();
  llvm::SmallVector<mlir::BlockArgument, 8> iterArguments;
  iterArguments.reserve(numberOfInserted);
  for (unsigned index = 0; index < numberOfInserted; ++index) {
    iterArguments.push_back(
        newBodyBlock->getArgument(oldNumberOfBodyArguments + index));
  }

  for (unsigned index = 0; index < numberOfInserted; ++index) {
    auto insertSliceOperation =
        mlir::cast<tensor::InsertSliceOp>(insertSliceOperations[index]);
    insertSliceOperation.getDestMutable().assign(iterArguments[index]);
  }

  for (unsigned index = 0; index < numberOfInserted; ++index) {
    mlir::Value originalBaseValue = initOperands[index];
    mlir::BlockArgument iterArgument = iterArguments[index];
    auto usesIterator = baseToUsesInLoopMap.find(originalBaseValue);
    if (usesIterator == baseToUsesInLoopMap.end()) { continue; }
    auto slicesIterator = baseToInsertSliceOperations.find(originalBaseValue);
    if (slicesIterator == baseToInsertSliceOperations.end()) { continue; }
    auto &sliceOperations = slicesIterator->second;
    llvm::sort(
        sliceOperations,
        [](mlir::Operation *firstOperation,
           mlir::Operation *secondOperation) {
          return firstOperation->isBeforeInBlock(secondOperation);
        });
    for (mlir::OpOperand *use : usesIterator->second) {
      mlir::Operation *userOperation = use->getOwner();
      if (llvm::is_contained(sliceOperations, userOperation)) {
        continue;
      }
      mlir::Value replacementValue = iterArgument;
      for (mlir::Operation *sliceOperation : sliceOperations) {
        if (sliceOperation->isBeforeInBlock(userOperation)) {
          replacementValue =
              mlir::cast<tensor::InsertSliceOp>(sliceOperation)
                  .getResult();
        } else {
          break;
        }
      }
      use->set(replacementValue);
    }
  }

  for (unsigned index = 0; index < numberOfInserted; ++index) {
    mlir::Value resultValue =
        newLoopOperation.getResult(oldNumberOfResults + index);
    newlyAddedResults.push_back(resultValue);
    insertSliceToYieldedResultMap[insertSliceOperations[index]] =
        resultValue;
  }

  // Record newly added iter indices: from oldNumberOfIterOperands,
  // count numberOfInserted.
  llvm::SmallVector<int64_t, 8> newIterIndices;
  for (unsigned index = 0; index < numberOfInserted; ++index) {
    newIterIndices.push_back(
        static_cast<int64_t>(oldNumberOfIterOperands + index));
  }
  appendTensorizeIterIndices(affineForOperation, newLoopOperation,
                             newIterIndices);

  return newLoopOperation;
}

static void replaceBaseUsesOutsideLoop(
    affine::AffineForOp affineForOperation,
    llvm::ArrayRef<mlir::Operation *> insertSliceOperations,
    const llvm::DenseMap<mlir::Operation *, mlir::Value>
        &insertSliceToYieldedResultMap,
    llvm::ArrayRef<mlir::Value> originalBaseValues) {
  mlir::Operation *loopOperation = affineForOperation.getOperation();
  for (unsigned index = 0; index < insertSliceOperations.size(); ++index) {
    mlir::Operation *insertSliceOperation = insertSliceOperations[index];
    mlir::Value originalBaseValue = originalBaseValues[index];
    auto iterator = insertSliceToYieldedResultMap.find(insertSliceOperation);
    if (iterator == insertSliceToYieldedResultMap.end()) { continue; }
    mlir::Value loopResultValue = iterator->second;

    auto shouldReplace = [&](mlir::OpOperand &use) -> bool {
      mlir::Operation *userOperation = use.getOwner();
      if (userOperation == loopOperation) { return false; }
      if (isAncestorOperation(userOperation, loopOperation)) {
        return false;
      }
      if (userOperation->getBlock() == loopOperation->getBlock() &&
          isBeforeInSameBlock(userOperation, loopOperation)) {
        return false;
      }
      if (isProperAncestorOperation(loopOperation, userOperation)) {
        return false;
      }
      if (mlir::Operation *ancestorLoopOperation =
              userOperation->getParentOfType<affine::AffineForOp>()) {
        if (ancestorLoopOperation != loopOperation &&
            loopOperation->isAncestor(ancestorLoopOperation)) {
          return false;
        }
        if (ancestorLoopOperation != loopOperation &&
            !loopOperation->isAncestor(ancestorLoopOperation)) {
          return true;
        }
      }
      return true;
    };

    originalBaseValue.replaceUsesWithIf(
        loopResultValue, mlir::function_ref<bool(mlir::OpOperand &)>(
                             [&](mlir::OpOperand &use) {
                               return shouldReplace(use);
                             }));
  }
}

static mlir::LogicalResult processAffineForLoop(
    affine::AffineForOp affineForOperation,
    const llvm::DenseMap<mlir::Value, mlir::Value> *baseToInitValueMap =
        nullptr,
    llvm::DenseMap<mlir::Value, mlir::Value> *outBaseToResultValueMap =
        nullptr) {
  llvm::SmallVector<mlir::Operation *, 8> insertSliceOperations;
  collectInsertSliceOpsInLoop(affineForOperation, insertSliceOperations);
  if (insertSliceOperations.empty()) { return mlir::success(); }

  llvm::SmallVector<mlir::Value, 8> customInitOperands;
  if (baseToInitValueMap) {
    customInitOperands.reserve(insertSliceOperations.size());
    for (mlir::Operation *operation : insertSliceOperations) {
      auto insertSliceOperation =
          mlir::cast<tensor::InsertSliceOp>(operation);
      mlir::Value destinationValue = insertSliceOperation.getDest();
      mlir::Value originalBaseValue =
          findOriginalBase(destinationValue);
      if (destinationValue != originalBaseValue) {
        customInitOperands.push_back(destinationValue);
      } else {
        auto iterator = baseToInitValueMap->find(originalBaseValue);
        customInitOperands.push_back(
            iterator != baseToInitValueMap->end() ? iterator->second
                                                  : originalBaseValue);
      }
    }
  }

  llvm::SmallVector<mlir::Value, 8> newResults;
  llvm::DenseMap<mlir::Operation *, mlir::Value>
      insertSliceToYieldedResultMap;
  llvm::SmallVector<mlir::Value, 8> originalBaseValues;

  auto newLoopOr = extendLoopWithInsertSliceLiveOuts(
      affineForOperation, insertSliceOperations, newResults,
      insertSliceToYieldedResultMap, originalBaseValues,
      baseToInitValueMap
          ? llvm::ArrayRef<mlir::Value>(customInitOperands)
          : llvm::ArrayRef<mlir::Value>());
  if (mlir::failed(newLoopOr)) { return mlir::failure(); }
  affine::AffineForOp newLoopOperation = *newLoopOr;

  if (outBaseToResultValueMap) {
    for (unsigned index = 0; index < insertSliceOperations.size();
         ++index) {
      (*outBaseToResultValueMap)[originalBaseValues[index]] =
          insertSliceToYieldedResultMap[insertSliceOperations[index]];
    }
  }

  replaceBaseUsesOutsideLoop(newLoopOperation, insertSliceOperations,
                             insertSliceToYieldedResultMap,
                             originalBaseValues);
  return mlir::success();
}

static mlir::LogicalResult runInsertSliceToIterArgs(
    func::FuncOp functionOperation) {
  llvm::SmallVector<affine::AffineForOp, 16> allAffineForLoops;
  functionOperation.walk([&](affine::AffineForOp affineForOperation) {
    allAffineForLoops.push_back(affineForOperation);
  });

  llvm::sort(
      allAffineForLoops,
      [](affine::AffineForOp firstLoop, affine::AffineForOp secondLoop) {
        unsigned depthFirst = 0;
        unsigned depthSecond = 0;
        mlir::Operation *operationFirst = firstLoop.getOperation();
        mlir::Operation *operationSecond = secondLoop.getOperation();
        while (operationFirst) {
          operationFirst =
              operationFirst->getParentOfType<affine::AffineForOp>();
          if (operationFirst) { ++depthFirst; }
        }
        while (operationSecond) {
          operationSecond =
              operationSecond->getParentOfType<affine::AffineForOp>();
          if (operationSecond) { ++depthSecond; }
        }
        return depthFirst > depthSecond;
      });

  llvm::DenseMap<mlir::Value, mlir::Value> baseToLatestYieldedValueMap;

  for (affine::AffineForOp affineForOperation : allAffineForLoops) {
    llvm::SmallVector<mlir::Operation *, 8> insertSliceOperations;
    collectInsertSliceOpsInLoop(affineForOperation, insertSliceOperations);
    if (insertSliceOperations.empty()) { continue; }

    llvm::DenseMap<mlir::Value, mlir::Value> baseToInitValueMap;
    for (mlir::Operation *operation : insertSliceOperations) {
      auto insertSliceOperation =
          mlir::cast<tensor::InsertSliceOp>(operation);
      mlir::Value baseValue =
          findOriginalBase(insertSliceOperation.getDest());
      auto latestIterator = baseToLatestYieldedValueMap.find(baseValue);
      if (latestIterator != baseToLatestYieldedValueMap.end()) {
        baseToInitValueMap[baseValue] = latestIterator->second;
      }
    }

    llvm::DenseMap<mlir::Value, mlir::Value> baseToResultValueMap;
    if (mlir::failed(processAffineForLoop(
            affineForOperation,
            baseToInitValueMap.empty() ? nullptr : &baseToInitValueMap,
            &baseToResultValueMap))) {
      return mlir::failure();
    }

    for (auto &keyValue : baseToResultValueMap) {
      baseToLatestYieldedValueMap[keyValue.first] = keyValue.second;
    }
  }

  if (mlir::failed(
          propagateTensorLiveOutsToParentLoops(functionOperation))) {
    return mlir::failure();
  }
  if (mlir::failed(
          fixInnerLoopInitOperandsFromOuterLoops(functionOperation))) {
    return mlir::failure();
  }
  return mlir::success();
}

static void collectSiblingLoops(
    mlir::Block *parentBlock,
    llvm::SmallVectorImpl<affine::AffineForOp> &siblingLoops) {
  siblingLoops.clear();
  for (mlir::Operation &operation : *parentBlock) {
    if (auto forOperation =
            mlir::dyn_cast<affine::AffineForOp>(&operation)) {
      if (forOperation.getNumResults() > 0) {
        siblingLoops.push_back(forOperation);
      }
    }
  }
  llvm::sort(
      siblingLoops,
      [](affine::AffineForOp firstLoop, affine::AffineForOp secondLoop) {
        return firstLoop->isBeforeInBlock(secondLoop.getOperation());
      });
}

static void collectSiblingLoopResults(
    llvm::ArrayRef<affine::AffineForOp> siblingLoops,
    llvm::SmallVectorImpl<mlir::Value> &allResults,
    llvm::DenseMap<mlir::Value, mlir::Value> &resultToInitOperandMap) {
  allResults.clear();
  resultToInitOperandMap.clear();
  for (affine::AffineForOp affineForOperation : siblingLoops) {
    unsigned numberOfResults = affineForOperation.getNumResults();
    unsigned numberOfIterOperands =
        affineForOperation.getNumIterOperands();
    for (unsigned resultIndex = 0; resultIndex < numberOfResults;
         ++resultIndex) {
      // Only handle results marked by this pass
      // (corresponding to iter index resultIndex).
      if (!isTensorizeLiveOutResult(affineForOperation, resultIndex)) {
        continue;
      }

      mlir::Value resultValue =
          affineForOperation.getResult(resultIndex);
      allResults.push_back(resultValue);
      if (resultIndex < numberOfIterOperands) {
        unsigned totalNumberOfOperands =
            affineForOperation->getNumOperands();
        unsigned operandIndex = totalNumberOfOperands -
                                numberOfIterOperands + resultIndex;
        if (operandIndex < totalNumberOfOperands) {
          mlir::Value initOperand =
              affineForOperation->getOperand(operandIndex);
          resultToInitOperandMap[resultValue] =
              findOriginalBase(initOperand);
        }
      }
    }
  }
}

static mlir::FailureOr<affine::AffineForOp> passResultsToParentLoop(
    affine::AffineForOp parentLoopOperation,
    llvm::ArrayRef<mlir::Value> childResults,
    const llvm::DenseMap<mlir::Value, mlir::Value>
        &resultToInitOperandMap) {
  if (childResults.empty()) { return parentLoopOperation; }

  llvm::SmallVector<mlir::Value, 16> initOperands;
  initOperands.reserve(childResults.size());
  for (mlir::Value childResult : childResults) {
    auto iterator = resultToInitOperandMap.find(childResult);
    if (iterator != resultToInitOperandMap.end()) {
      initOperands.push_back(iterator->second);
    } else {
      initOperands.push_back(findOriginalBaseRecursive(childResult));
    }
  }

  mlir::PatternRewriter rewriter(parentLoopOperation.getContext());
  rewriter.setInsertionPoint(parentLoopOperation);

  llvm::SmallVector<mlir::BlockArgument, 16> parentIterArguments;
  unsigned oldNumberOfResults = parentLoopOperation.getNumResults();
  unsigned oldNumberOfIterOperands =
      parentLoopOperation.getNumIterOperands();

  auto newParentOr = parentLoopOperation.replaceWithAdditionalYields(
      rewriter, initOperands, false,
      [&](mlir::OpBuilder &, mlir::Location,
          llvm::ArrayRef<mlir::BlockArgument> newBlockArguments) {
        parentIterArguments.assign(newBlockArguments.begin(),
                                   newBlockArguments.end());
        llvm::SmallVector<mlir::Value, 16> yields(childResults.begin(),
                                                  childResults.end());
        return yields;
      });

  if (mlir::failed(newParentOr)) { return mlir::failure(); }
  affine::AffineForOp newParentLoopOperation =
      mlir::cast<affine::AffineForOp>(*newParentOr);

  // Append iter indices added from childResults to parent's tensorize
  // marking.
  llvm::SmallVector<int64_t, 8> newIterIndices;
  for (unsigned index = 0; index < parentIterArguments.size(); ++index) {
    newIterIndices.push_back(
        static_cast<int64_t>(oldNumberOfIterOperands + index));
  }
  appendTensorizeIterIndices(parentLoopOperation, newParentLoopOperation,
                             newIterIndices);

  (void)oldNumberOfResults;
  return newParentLoopOperation;
}

static void replaceChildResultsWithParentResults(
    affine::AffineForOp parentLoopOperation,
    llvm::ArrayRef<mlir::Value> childResults) {
  mlir::Operation *parentOperation = parentLoopOperation.getOperation();
  unsigned oldNumberOfResults =
      parentLoopOperation.getNumResults() - childResults.size();
  for (unsigned index = 0; index < childResults.size(); ++index) {
    mlir::Value childResult = childResults[index];
    mlir::Value parentResult =
        parentLoopOperation.getResult(oldNumberOfResults + index);
    auto shouldReplace = [&](mlir::OpOperand &use) -> bool {
      mlir::Operation *userOperation = use.getOwner();
      if (userOperation == parentOperation) { return false; }
      if (isAncestorOperation(userOperation, parentOperation)) {
        return false;
      }
      if (userOperation->getBlock() == parentOperation->getBlock() &&
          isBeforeInSameBlock(userOperation, parentOperation)) {
        return false;
      }
      if (isProperAncestorOperation(parentOperation, userOperation)) {
        return false;
      }
      return true;
    };
    childResult.replaceUsesWithIf(
        parentResult, mlir::function_ref<bool(mlir::OpOperand &)>(
                          [&](mlir::OpOperand &use) {
                            return shouldReplace(use);
                          }));
  }
}

static mlir::LogicalResult processSiblingLoopsUnderParent(
    mlir::Block *parentBlock,
    llvm::DenseSet<mlir::Operation *> &processedParentLoops) {
  llvm::SmallVector<affine::AffineForOp, 8> siblingLoops;
  collectSiblingLoops(parentBlock, siblingLoops);
  if (siblingLoops.empty()) { return mlir::success(); }

  llvm::SmallVector<mlir::Value, 16> childResults;
  llvm::DenseMap<mlir::Value, mlir::Value> resultToInitOperandMap;
  collectSiblingLoopResults(siblingLoops, childResults,
                            resultToInitOperandMap);
  if (childResults.empty()) { return mlir::success(); }

  mlir::Operation *parentOperation = parentBlock->getParentOp();
  if (!parentOperation) { return mlir::success(); }
  auto parentLoopOperation =
      mlir::dyn_cast<affine::AffineForOp>(parentOperation);
  if (!parentLoopOperation) { return mlir::success(); }
  if (processedParentLoops.contains(
          parentLoopOperation.getOperation())) {
    return mlir::success();
  }

  auto newParentOr = passResultsToParentLoop(
      parentLoopOperation, childResults, resultToInitOperandMap);
  if (mlir::failed(newParentOr)) { return mlir::failure(); }
  affine::AffineForOp newParentLoopOperation = *newParentOr;

  processedParentLoops.insert(newParentLoopOperation.getOperation());
  replaceChildResultsWithParentResults(newParentLoopOperation,
                                       childResults);
  return mlir::success();
}

static mlir::LogicalResult propagateTensorLiveOutsToParentLoops(
    func::FuncOp functionOperation) {
  llvm::DenseSet<mlir::Operation *> processedParentLoops;
  unsigned round = 0;
  unsigned maxRounds = 10;
  while (round++ < maxRounds) {
    llvm::DenseMap<mlir::Block *,
                   llvm::SmallVector<affine::AffineForOp, 8>>
        loopsByParentBlock;
    functionOperation.walk([&](affine::AffineForOp affineForOperation) {
      if (affineForOperation.getNumResults() == 0) { return; }
      mlir::Block *parentBlock = affineForOperation->getBlock();
      mlir::Operation *parentOperation = parentBlock->getParentOp();
      if (!parentOperation) { return; }
      if (auto parentLoopOperation =
              mlir::dyn_cast<affine::AffineForOp>(parentOperation)) {
        if (processedParentLoops.contains(
                parentLoopOperation.getOperation())) {
          return;
        }
        loopsByParentBlock[parentBlock].push_back(affineForOperation);
      }
    });

    if (loopsByParentBlock.empty()) { break; }

    llvm::SmallVector<mlir::Block *, 16> parentBlocks;
    parentBlocks.reserve(loopsByParentBlock.size());
    std::transform(loopsByParentBlock.begin(), loopsByParentBlock.end(),
                   std::back_inserter(parentBlocks),
                   [](const auto &keyValue) { return keyValue.first; });

    llvm::sort(
        parentBlocks,
        [](mlir::Block *firstBlock, mlir::Block *secondBlock) {
          unsigned depthFirst = 0;
          unsigned depthSecond = 0;
          mlir::Operation *operationFirst = firstBlock->getParentOp();
          mlir::Operation *operationSecond = secondBlock->getParentOp();
          while (operationFirst) {
            operationFirst =
                operationFirst->getParentOfType<affine::AffineForOp>();
            if (operationFirst) { ++depthFirst; }
          }
          while (operationSecond) {
            operationSecond =
                operationSecond->getParentOfType<affine::AffineForOp>();
            if (operationSecond) { ++depthSecond; }
          }
          return depthFirst > depthSecond;
        });

    bool changed = false;
    for (mlir::Block *parentBlock : parentBlocks) {
      if (mlir::failed(processSiblingLoopsUnderParent(
              parentBlock, processedParentLoops))) {
        return mlir::failure();
      }
      changed = true;
    }
    if (!changed) { break; }
  }
  return mlir::success();
}

// Used to record outer loop iterArgs processed by this pass
// and their original iter indices.
struct TensorizeOuterIterInfo {
  mlir::BlockArgument iterArgument;
  unsigned iterIndex;
};

static mlir::LogicalResult processLoopForFixingInitOperands(
    affine::AffineForOp outerLoopOperation) {
  if (outerLoopOperation.getNumIterOperands() == 0) {
    return mlir::success();
  }

  mlir::Block *outerBodyBlock = outerLoopOperation.getBody();
  llvm::SmallVector<TensorizeOuterIterInfo, 8> outerIterInfos;
  llvm::SmallVector<mlir::BlockArgument, 8> outerIterArguments;
  unsigned numberOfInductionVariables = 1;
  for (unsigned iterIndex = 0;
       iterIndex < outerLoopOperation.getNumIterOperands();
       ++iterIndex) {
    // Only collect iterArgs whose iter indices are marked by this pass.
    if (!isTensorizeIterIndex(outerLoopOperation, iterIndex)) {
      continue;
    }
    mlir::BlockArgument iterArgument =
        outerBodyBlock->getArgument(numberOfInductionVariables +
                                    iterIndex);
    TensorizeOuterIterInfo iterInfo{iterArgument, iterIndex};
    outerIterInfos.push_back(iterInfo);
    outerIterArguments.push_back(iterArgument);
  }

  // If outer loop has no iter marked by this pass,
  // no need to fix its child loops.
  if (outerIterInfos.empty()) { return mlir::success(); }

  for (mlir::Operation &operation : *outerBodyBlock) {
    if (auto innerLoopOperation =
            mlir::dyn_cast<affine::AffineForOp>(&operation)) {
      unsigned numberOfInnerIterOperands =
          innerLoopOperation.getNumIterOperands();
      if (numberOfInnerIterOperands == 0) {
        if (mlir::failed(
                processLoopForFixingInitOperands(innerLoopOperation))) {
          return mlir::failure();
        }
        continue;
      }
      for (unsigned iterIndex = 0;
           iterIndex < numberOfInnerIterOperands; ++iterIndex) {
        unsigned operandIndex =
            (innerLoopOperation->getNumOperands() ==
             innerLoopOperation.getNumIterOperands())
                ? iterIndex
                : (2 + iterIndex);
        if (operandIndex >= innerLoopOperation->getNumOperands()) {
          continue;
        }

        mlir::Value initOperand =
            innerLoopOperation->getOperand(operandIndex);
        mlir::Value innerOriginalBase =
            findOriginalBaseRecursive(initOperand);

        bool found = false;
        for (unsigned infoIndex = 0; infoIndex < outerIterInfos.size();
             ++infoIndex) {
          mlir::BlockArgument outerIterArgument =
              outerIterInfos[infoIndex].iterArgument;
          unsigned outerIterIndex = outerIterInfos[infoIndex].iterIndex;

          unsigned outerOperandIndex =
              (outerLoopOperation->getNumOperands() ==
               outerLoopOperation.getNumIterOperands())
                  ? outerIterIndex
                  : (2 + outerIterIndex);
          if (outerOperandIndex >=
              outerLoopOperation->getNumOperands()) {
            continue;
          }

          mlir::Value outerInitOperand =
              outerLoopOperation->getOperand(outerOperandIndex);
          mlir::Value outerOriginalBase =
              findOriginalBaseRecursive(outerInitOperand);

          if (initOperand == outerIterArgument) {
            found = true;
            break;
          }
          if (outerIterIndex < outerLoopOperation.getNumResults()) {
            mlir::Value outerResult =
                outerLoopOperation.getResult(outerIterIndex);
            if (initOperand == outerResult) {
              innerLoopOperation->setOperand(operandIndex,
                                             outerIterArgument);
              found = true;
              break;
            }
          }
          if (innerOriginalBase == outerOriginalBase) {
            innerLoopOperation->setOperand(operandIndex,
                                           outerIterArgument);
            found = true;
            break;
          }
          if (auto definingOperation = initOperand.getDefiningOp()) {
            if (auto forOperation = mlir::dyn_cast<affine::AffineForOp>(
                    definingOperation)) {
              unsigned numberOfResults = forOperation.getNumResults();
              for (unsigned resultIndex = 0;
                   resultIndex < numberOfResults; ++resultIndex) {
                if (forOperation.getResult(resultIndex) ==
                    initOperand) {
                  unsigned numberOfIterOperandsLocal =
                      forOperation.getNumIterOperands();
                  if (resultIndex < numberOfIterOperandsLocal) {
                    unsigned forOperandIndex =
                        (forOperation->getNumOperands() ==
                         numberOfIterOperandsLocal)
                            ? resultIndex
                            : (2 + resultIndex);
                    if (forOperandIndex <
                        forOperation->getNumOperands()) {
                      mlir::Value forInitOperand =
                          forOperation->getOperand(forOperandIndex);
                      mlir::Value forOriginalBase =
                          findOriginalBaseRecursive(forInitOperand);
                      if (forOriginalBase == outerOriginalBase) {
                        innerLoopOperation->setOperand(
                            operandIndex, outerIterArgument);
                        found = true;
                      }
                    }
                  }
                  break;
                }
              }
              if (found) { break; }
            }
          }
        }
      }
      if (mlir::failed(
              processLoopForFixingInitOperands(innerLoopOperation))) {
        return mlir::failure();
      }
    }
  }
  return mlir::success();
}

static mlir::LogicalResult fixInnerLoopInitOperandsFromOuterLoops(
    func::FuncOp functionOperation) {
  llvm::SmallVector<affine::AffineForOp, 16> allAffineForLoops;
  functionOperation.walk([&](affine::AffineForOp affineForOperation) {
    allAffineForLoops.push_back(affineForOperation);
  });

  llvm::sort(
      allAffineForLoops,
      [](affine::AffineForOp firstLoop, affine::AffineForOp secondLoop) {
        unsigned depthFirst = 0;
        unsigned depthSecond = 0;
        mlir::Operation *operationFirst = firstLoop.getOperation();
        mlir::Operation *operationSecond = secondLoop.getOperation();
        while (operationFirst) {
          operationFirst =
              operationFirst->getParentOfType<affine::AffineForOp>();
          if (operationFirst) { ++depthFirst; }
        }
        while (operationSecond) {
          operationSecond =
              operationSecond->getParentOfType<affine::AffineForOp>();
          if (operationSecond) { ++depthSecond; }
        }
        return depthFirst < depthSecond;
      });

  if (std::any_of(allAffineForLoops.begin(), allAffineForLoops.end(),
                  [](affine::AffineForOp affineForOperation) {
                    return mlir::failed(
                        processLoopForFixingInitOperands(affineForOperation));
                  })) {
    return mlir::failure();
  }

  return mlir::success();
}

namespace {
#define GEN_PASS_DECL_TENSORIZELIVEOUTS
#define GEN_PASS_DEF_TENSORIZELIVEOUTS
#include "akg/Dialect/Affine/Passes.h.inc"

struct TensorizeLiveOutsPass
    : public impl::TensorizeLiveOutsBase<TensorizeLiveOutsPass> {
  void getDependentDialects(
      mlir::DialectRegistry &dialectRegistry) const override {
    dialectRegistry.insert<affine::AffineDialect, tensor::TensorDialect,
                           arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp functionOperation = getOperation();
    if (mlir::failed(runInsertSliceToIterArgs(functionOperation))) {
      signalPassFailure();
      return;
    }
  }
};
}  // namespace

namespace mlir::affine {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTensorizeLiveOutsPass() {
  return std::make_unique<TensorizeLiveOutsPass>();
}
}  // namespace mlir::affine

