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

#include "akg/Dialect/Affine/Analysis/BufferAnalysis.h"

#include <algorithm>
#include <limits>
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/Support/Debug.h"

namespace mlir {
namespace akg {

//===----------------------------------------------------------------------===//
// UnionFind Implementation
//===----------------------------------------------------------------------===//

void UnionFind::ensureCapacity(size_t n) {
  if (n + 1 > parent.size()) {
    size_t oldSize = parent.size();
    parent.resize(n + 1, -1);
    minIndex.resize(n + 1);
    for (size_t i = oldSize; i < n + 1; ++i) {
      minIndex[i] = static_cast<int>(i);
    }
  }
}

int UnionFind::find(int x) {
  ensureCapacity(x);
  if (parent[x] < 0) {
    return x;
  }
  return parent[x] = find(parent[x]);
}

bool UnionFind::join(int a, int b) {
  ensureCapacity(std::max(a, b));
  a = find(a);
  b = find(b);
  if (a != b) {
    // Union by rank: attach smaller tree under root of larger tree.
    if (parent[a] > parent[b]) {
      std::swap(a, b);
    }
    parent[a] += parent[b];
    parent[b] = a;
    minIndex[a] = std::min(minIndex[b], minIndex[a]);
  }
  return true;
}

//===----------------------------------------------------------------------===//
// ValOperationIndexer Implementation
//===----------------------------------------------------------------------===//

mlir::FailureOr<Value> ValOperationIndexer::getVal(uint32_t idx) const {
  auto it = idxToVal.find(idx);
  if (it != idxToVal.end()) {
    return it->second;
  }
  return failure();
}

mlir::FailureOr<Operation *> ValOperationIndexer::getOp(uint32_t idx) const {
  auto it = idxToOp.find(idx);
  if (it != idxToOp.end()) {
    return it->second;
  }
  return failure();
}

uint32_t ValOperationIndexer::getClosestOpIdx(uint32_t idx) const {
  auto it = idxToOp.lower_bound(idx);
  if (it == idxToOp.end()) {
    return kOpNotFoundLiveRange;
  }
  return it->first;
}

bool ValOperationIndexer::insert(Value val) {
  llvm::outs() << val << " " << opCount << "\n";
  if (valToIdx.count(val)) {
    return false;
  }
  valToIdx[val] = opCount;
  idxToVal[opCount] = val;
  opCount++;
  return true;
}

bool ValOperationIndexer::insert(Operation *op) {
  if (opToIdx.count(op)) {
    return false;
  }
  opToIdx[op] = opCount;
  idxToOp[opCount] = op;
  opCount++;
  return true;
}

//===----------------------------------------------------------------------===//
// BufferAnalysis Implementation
//===----------------------------------------------------------------------===//

Value BufferAnalysis::getMemRefFromOp(Operation *op) {
  if (auto loadOp = dyn_cast<mlir::affine::AffineLoadOp>(op)) {
    return loadOp.getMemRef();
  }
  if (auto storeOp = dyn_cast<mlir::affine::AffineStoreOp>(op)) {
    return storeOp.getMemRef();
  }
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    return loadOp.getMemRef();
  }
  if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    return storeOp.getMemRef();
  }
  return Value();
}

void BufferAnalysis::adjustInplaceReuseOp(Operation *op) {
  if (!isa<mlir::affine::AffineForOp>(op)) {
    return;
  }

  auto forOp = cast<mlir::affine::AffineForOp>(op);
  llvm::SmallVector<Value> readMemRefs, writeMemRefs;

  forOp.walk([&](mlir::affine::AffineLoadOp loadOp) { readMemRefs.push_back(loadOp.getMemRef()); });

  forOp.walk([&](mlir::affine::AffineStoreOp storeOp) { writeMemRefs.push_back(storeOp.getMemRef()); });

  for (auto readMem : readMemRefs) {
    for (auto writeMem : writeMemRefs) {
      if (readMem == writeMem) {
        llvm::outs() << "In-place operation detected for memref: " << readMem << "\n";
      }
    }
  }
}

void BufferAnalysis::adjustCopyInCopyOut(Operation *op) {
  if (!options.enableDmaOpt) {
    return;
  }

  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    auto srcMemref = copyOp.getSource();
    auto dstMemref = copyOp.getTarget();
    if (valToLiveRangeIdx.count(srcMemref)) {
      auto rangesIndex = valToLiveRangeIdx.at(srcMemref);
      liveRanges[rangesIndex].end = indexer.getCurrentCount();
      llvm::outs() << "Extended live range of source " << srcMemref << " to " << indexer.getCurrentCount() << "\n";
    }
    if (valToLiveRangeIdx.count(dstMemref)) {
      auto rangesIndex = valToLiveRangeIdx.at(dstMemref);
      liveRanges[rangesIndex].end = indexer.getCurrentCount();
      llvm::outs() << "Extended live range of target " << dstMemref << " to " << indexer.getCurrentCount() << "\n";
    }
  }
}

uint32_t BufferAnalysis::insertValue(const Value &value, uint32_t pos, uint32_t weight) {
  llvm::outs() << "--- Inserting value " << value << " " << pos << " " << weight << "\n";
  assert(!valToLiveRangeIdx.count(value));
  liveRanges.emplace_back(pos, pos, weight);
  return valToLiveRangeIdx[value] = liveRanges.size() - 1;
}

int64_t BufferAnalysis::getExtraBufferSizeByFactor(Operation *op) const {
  if (auto forOp = dyn_cast<mlir::affine::AffineForOp>(op)) {
    bool hasReduction = false;
    forOp.walk([&](mlir::affine::AffineStoreOp storeOp) {
      Value storedMemref = storeOp.getMemRef();
      for (auto user : storedMemref.getUsers()) {
        if (auto loadOp = dyn_cast<mlir::affine::AffineLoadOp>(user)) {
          if (loadOp->getBlock() == storeOp->getBlock()) {
            hasReduction = true;
            return;
          }
        }
      }
    });

    if (hasReduction) {
      return 0;
    }
  }
  return 0;
}

llvm::SmallVector<Value> BufferAnalysis::getOperands(Operation &op) const {
  if (auto forOp = dyn_cast<mlir::affine::AffineForOp>(op)) {
    return llvm::SmallVector<Value>(forOp.getInits().begin(), forOp.getInits().end());
  }
  if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
    return llvm::SmallVector<Value>(returnOp.getOperands().begin(), returnOp.getOperands().end());
  }
  return llvm::SmallVector<Value>(op.getOperands().begin(), op.getOperands().end());
}

uint32_t BufferAnalysis::getValMultiBuffer(const Value &value, uint32_t def) const {
  auto it = options.multiBufferCount.find(value);
  if (it != options.multiBufferCount.end()) {
    return it->second;
  }
  return def;
}

uint32_t BufferAnalysis::getValDataTypeWeight(const Value &value, uint32_t def) const {
  auto it = dataTypeWeightMap.find(value);
  if (it != dataTypeWeightMap.end()) {
    return it->second;
  }
  return def;
}

uint32_t BufferAnalysis::updateAliasIntoFurthest(const Value &val, Operation *endOp) {
  auto valIdx = indexer.getIndex(val);
  llvm::outs() << "found valIdx " << valIdx << "\n";

  auto aliasParent = aliasSet.minIndex[aliasSet.find(valIdx)];
  llvm::outs() << "found alias parent " << aliasParent << "\n";
  llvm::outs() << "Ok found endIdx " << *endOp << "\n";

  auto endIdx = indexer.getIndex(endOp);
  llvm::outs() << "Ok found endIdx " << endIdx << "\n";

  if (!aliasFurthest.count(aliasParent)) {
    aliasFurthest[aliasParent] = -1;
  }

  auto &furthestPtr = aliasFurthest[aliasParent];
  llvm::outs() << endIdx << " " << furthestPtr << " end -- " << *endOp << "\n";

  if (endIdx > furthestPtr) {
    llvm::outs() << "Updating furthest " << endIdx << " " << furthestPtr << "\n";
    opToEndValIdx[furthestPtr].erase(aliasParent);
    furthestPtr = endIdx;
    opToEndValIdx[endIdx].insert(aliasParent);
  }
  return aliasParent;
}

void BufferAnalysis::processOperationForLiveRange(Operation *op, const mlir::LivenessBlockInfo *blockInfo) {
  if (isControlFlowOp(op)) {
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &nestedOp : nestedBlock) {
          processOperationForLiveRange(&nestedOp, blockInfo);
        }
      }
    }
    return;
  }

  // if (skippableOperation(op)) {
  //   return;
  // }

  uint32_t currentOpIndex = indexer.getIndex(op);

  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    Operation *endOp = blockInfo->getEndOperation(res, op);
    auto aliasParent = updateAliasIntoFurthest(res, endOp);

    auto currentWeight = getValMultiBuffer(res) * getValDataTypeWeight(res);
    llvm::outs() << "inserting " << res << "\n";

    // Aliased values don't contribute additional weight.
    if (aliasParent != indexer.getIndex(res)) {
      currentWeight = 0;
    }

    // bufferization.to_memref and bufferization.to_tensor results have zero weight
    // as they represent external buffers that are copied into local buffers.
    bool isToMemrefOp = isa<bufferization::ToMemrefOp>(op);
    bool isToTensorOp = isa<bufferization::ToTensorOp>(op);

    if (isToMemrefOp || isToTensorOp) {
      // Check if any operand of the operation is a Block argument
      bool hasBlockArgOperand = false;
      for (Value operand : op->getOperands()) {
        if (llvm::is_contained(block.getArguments(), operand)) {
          hasBlockArgOperand = true;
          break;
        }
      }

      if (hasBlockArgOperand) {
        llvm::outs() << (isToMemrefOp ? "ToMemrefOp" : "ToTensorOp")
                     << " with block argument detected, setting weight to 0 for " << res << "\n";
        currentWeight = 0;
      }
    }

    insertValue(res, aliasParent, currentWeight);
  }

  // Update live range end points for values that die at this operation.
  llvm::outs() << "Printing dead val at " << currentOpIndex << " " << *op << "\n";
  for (auto deadVal : opToEndValIdx[currentOpIndex]) {
    llvm::outs() << "Here is " << deadVal << "\n";
    Value curVal = indexer.getVal(deadVal).value();
    auto indexPos = valToLiveRangeIdx[curVal];
    liveRanges[indexPos].end = currentOpIndex;
  }

  // Add extra buffer for operations that need temporary buffers.
  if (auto extraWeight = getExtraBufferSizeByFactor(op)) {
    extraWeight *= std::max((uint32_t)1, getValMultiBuffer(op->getResult(0), 0));
    llvm::outs() << "Appending " << *op << " with " << extraWeight << "\n";
    liveRanges.emplace_back(currentOpIndex, currentOpIndex, extraWeight);
  }
}

void BufferAnalysis::processOperationForPostProcess(Operation *op) {
  if (isControlFlowOp(op)) {
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &nestedOp : nestedBlock) {
          processOperationForPostProcess(&nestedOp);
        }
      }
    }
    return;
  }

  if (skippableOperation(op)) {
    return;
  }

  adjustInplaceReuseOp(op);
  if (options.enableDmaOpt) {
    adjustCopyInCopyOut(op);
  }
}

void BufferAnalysis::gatherLiveRanges(const mlir::LivenessBlockInfo *blockInfo) {
  llvm::outs() << "Gathering live range information...\n";

  // Process operations in the block.
  for (auto &op : block) {
    processOperationForLiveRange(&op, blockInfo);
  }

  // Post-process for in-place reuse and DMA optimizations.
  for (auto &op : block) {
    processOperationForPostProcess(&op);
  }
}

void BufferAnalysis::recordDataTypeWeight(const Value &value, uint32_t *smallestTypeBits) {
  Type type = value.getType();

  uint32_t currentTypeBits = 0;
  if (auto memrefType = dyn_cast<MemRefType>(type)) {
    auto elementType = memrefType.getElementType();
    assert(elementType.isIntOrFloat() && "Can only handle int or float element type!");
    currentTypeBits = static_cast<uint32_t>(elementType.getIntOrFloatBitWidth());
  } else if (auto tensorType = dyn_cast<TensorType>(type)) {
    auto elementType = tensorType.getElementType();
    assert(elementType.isIntOrFloat() && "Can only handle int or float element type!");
    currentTypeBits = static_cast<uint32_t>(elementType.getIntOrFloatBitWidth());
  } else if (auto intType = dyn_cast<IntegerType>(type)) {
    currentTypeBits = static_cast<uint32_t>(intType.getWidth());
  } else if (auto floatType = dyn_cast<FloatType>(type)) {
    currentTypeBits = static_cast<uint32_t>(floatType.getWidth());
  } else {
    return;
  }

  dataTypeWeightMap[value] = currentTypeBits;
  *smallestTypeBits = std::min(*smallestTypeBits, currentTypeBits);
}

void BufferAnalysis::processOperationForDataTypeWeight(Operation *op, uint32_t *smallestTypeBits) {
  if (isControlFlowOp(op)) {
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &nestedOp : nestedBlock) {
          processOperationForDataTypeWeight(&nestedOp, smallestTypeBits);
        }
      }
    }
    return;
  }

  for (const auto &[idx, res] : llvm::enumerate(op->getResults())) {
    recordDataTypeWeight(res, smallestTypeBits);
  }

  for (const auto &[idx, operand] : llvm::enumerate(op->getOperands())) {
    recordDataTypeWeight(operand, smallestTypeBits);
  }
}

void BufferAnalysis::gatherDataTypeWeights() {
  llvm::outs() << "Gathering data type information...\n";
  uint32_t smallestTypeBits = std::numeric_limits<uint32_t>::max();

  // Process operations in the block.
  // Note: Block arguments are now tensors, which are converted to memrefs via
  // bufferization.to_memref operations. The memref results will be processed here.
  for (auto &op : block) {
    processOperationForDataTypeWeight(&op, &smallestTypeBits);
  }

  llvm::outs() << "Smallest type bits is " << smallestTypeBits << ", normalizing weights...\n";

  // Normalize weights based on smallest type.
  for (auto &[value, bits] : dataTypeWeightMap) {
    auto normalizedTypeBits = bits / smallestTypeBits;
    if (bits % smallestTypeBits != 0) {
      llvm::outs() << "WARN: Current type bits " << bits
                   << " is not divisible by the smallest type bits! Rounding up...\n";
      normalizedTypeBits = (bits + smallestTypeBits - 1) / smallestTypeBits;
    }
    bits = normalizedTypeBits;
  }
}

void BufferAnalysis::printLiveRanges() const {
  llvm::outs() << "Considering " << valToLiveRangeIdx.size() << " and " << liveRanges.size() - valToLiveRangeIdx.size()
               << " extra Live Range:\n";

  for (size_t i = 0; i < liveRanges.size(); i++) {
    llvm::outs() << "Live Range #" << i << ": \n";
    if (i == 0 || liveRanges[i].start != liveRanges[i - 1].start) {
      auto currentVal = indexer.getVal(liveRanges[i].start);
      if (succeeded(currentVal)) {
        llvm::outs() << currentVal.value() << ": \n";
      } else {
        llvm::outs() << *indexer.getOp(liveRanges[i].start).value() << ": \n";
      }
    }

    llvm::outs() << liveRanges[i].start << " " << liveRanges[i].end << " " << liveRanges[i].weight << "\n";
    llvm::outs() << "Done Live Range\n";
  }
}

int64_t BufferAnalysis::lineSweepRanges() {
  // Min-heap sorted by end time.
  llvm::PriorityQueue<WeightedEndPair, llvm::SmallVector<WeightedEndPair>, std::greater<WeightedEndPair>> earlyDone;

  int64_t maxBuffer = 0;
  int64_t currentBuffer = 0;

  for (const auto &liveRange : liveRanges) {
    if (liveRange.start == liveRange.end) {
      llvm::outs() << "WARN: dead operation or temporary buffer exists at position " << liveRange.start << "\n";
    }

    // Remove buffers that have ended before the current start.
    while (!earlyDone.empty() && earlyDone.top().first < liveRange.start) {
      currentBuffer -= earlyDone.top().second;
      earlyDone.pop();
    }

    earlyDone.push({liveRange.end, liveRange.weight});
    currentBuffer += liveRange.weight;
    maxBuffer = std::max(maxBuffer, currentBuffer);
  }
  return maxBuffer;
}

void BufferAnalysis::printAliasInfo() {
  for (const auto &[idx, val] : indexer.idxToVal) {
    auto aliasParent = indexer.getVal(aliasSet.minIndex[aliasSet.find(idx)]);
    if (aliasParent != val) {
      llvm::outs() << "value: " << val << " alias parent is: " << aliasParent << "\n";
    }
  }
}

void BufferAnalysis::processOperationForIndexing(Operation *op) {
  if (isControlFlowOp(op)) {
    indexer.insert(op);
    for (Region &region : op->getRegions()) {
      for (Block &nestedBlock : region) {
        for (Operation &nestedOp : nestedBlock) {
          processOperationForIndexing(&nestedOp);
        }
      }
    }
    return;
  }

  llvm::outs() << "Processing op: " << *op << "\n";

  for (auto res : op->getResults()) {
    indexer.insert(res);

    if (!isUsingBuffer(res)) {
      continue;
    }

    // Handle aliasing operations (subview, reshape, etc.).
    if (isMemRefAliasingOp(op)) {
      auto src = getAliasSource(op);
      if (indexer.valToIdx.count(src)) {
        auto aliasSrcPar = indexer.getIndex(src);
        aliasSet.join(aliasSrcPar, indexer.getIndex(res));
      }
    }
  }

  llvm::outs() << "Inserting op " << *op << "\n";
  indexer.insert(op);
}

void BufferAnalysis::gatherIndexingAndAlias() {
  llvm::outs() << "Gathering alias information...\n";

  // Process operations.
  for (auto &op : block) {
    processOperationForIndexing(&op);
  }
  printAliasInfo();
}

int64_t BufferAnalysis::countMaxBuffer() {
  const mlir::LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
  gatherIndexingAndAlias();
  gatherDataTypeWeights();
  gatherLiveRanges(blockInfo);
  llvm::sort(liveRanges);
  if (options.printLiveRange) {
    printLiveRanges();
  }
  return lineSweepRanges();
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

int64_t countMaxBuffer(mlir::func::FuncOp func, const BufferAnalysisOptions &options) {
  if (func.getBody().getBlocks().size() != 1) {
    return -1;
  }

  BufferAnalysis analysis(*func.getBody().begin(), options, func);
  return analysis.countMaxBuffer();
}

}  // namespace akg
}  // namespace mlir
