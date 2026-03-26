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

#include "akg/Analysis/BufferAnalysis.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <set>
#include <type_traits>
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/PriorityQueue.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace akg {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Get static total size of a shape.
static std::optional<int64_t> getStaticTotalSize(ArrayRef<int64_t> shapes) {
  int64_t totalSize = 1;
  for (int64_t dim : shapes) {
    if (ShapedType::isDynamic(dim)) {
      return std::nullopt;
    }
    totalSize *= dim;
  }
  return totalSize;
}

/// Trace back to the original memref from aliasing operations.
static Value tracebackMemRef(Value memrefVal) {
  Value current = memrefVal;
  while (true) {
    if (auto subview = current.getDefiningOp<memref::SubViewOp>()) {
      current = subview.getSource();
    } else if (auto reshape = current.getDefiningOp<memref::ReshapeOp>()) {
      current = reshape.getSource();
    } else if (auto expand = current.getDefiningOp<memref::ExpandShapeOp>()) {
      current = expand.getSrc();
    } else if (auto collapse = current.getDefiningOp<memref::CollapseShapeOp>()) {
      current = collapse.getSrc();
    } else if (auto cast = current.getDefiningOp<memref::ReinterpretCastOp>()) {
      current = cast.getSource();
    } else {
      break;
    }
  }
  return current;
}

/// Check if an operation defines an alloc-like operation.
static bool isDefiningOpAllocLike(Value value) {
  auto defOp = value.getDefiningOp();
  if (!defOp) return false;
  return isa<memref::AllocOp, memref::AllocaOp>(defOp);
}

/// Check if two buffers can be inplace reused (same element type and bit width).
static bool canInplaceReuse(Value genBuffer, Value killBuffer, const llvm::DenseMap<Value, BufferInfo> &bufferInfos) {
  auto genIt = bufferInfos.find(genBuffer);
  auto killIt = bufferInfos.find(killBuffer);
  if (genIt == bufferInfos.end() || killIt == bufferInfos.end()) {
    return false;
  }

  // Check if kill buffer size >= gen buffer size (can accommodate)
  if (killIt->second.constBits < genIt->second.constBits) {
    return false;
  }

  // Check if element types have the same bit width
  auto genBitWidth = genIt->second.elementType.getIntOrFloatBitWidth();
  auto killBitWidth = killIt->second.elementType.getIntOrFloatBitWidth();

  return genBitWidth == killBitWidth;
}

static bool isMemRefValue(Value value) { return isa<MemRefType>(value.getType()); }

static bool isScalarTrackedValue(Value value) { return value.getType().isIntOrFloat(); }

static bool isDefinedInsideRegion(Value value, Region &region) {
  Operation *defOp = value.getDefiningOp();
  if (defOp == nullptr) {
    return false;
  }
  for (Region *parent = defOp->getParentRegion(); parent != nullptr; parent = parent->getParentRegion()) {
    if (parent == &region) {
      return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BufferAnalysis Implementation
//===----------------------------------------------------------------------===//

namespace {
/// Get alias pairs from an operation (for subview, reshape, etc.)
SmallVector<std::pair<Value, Value>> getOperationAliasInfo(Operation *op) {
  SmallVector<std::pair<Value, Value>> aliasPairs;
  if (isMemRefAliasingOp(op)) {
    Value src = getAliasSource(op);
    for (Value result : op->getResults()) {
      if (isa<MemRefType>(result.getType())) {
        aliasPairs.push_back({result, src});
      }
    }
  }
  return aliasPairs;
}

template <typename ForOpType>
SmallVector<Value> getForOpYieldedValues(ForOpType forOp) {
  if constexpr (std::is_same_v<ForOpType, mlir::affine::AffineForOp>) {
    return forOp.getYieldedValues();
  } else {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    return SmallVector<Value>(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
  }
}
}  // namespace

OpInfo *BufferAnalysis::UpdateLinearOperation(Operation *op) {
  auto opInfo = std::make_unique<OpInfo>(op, seqIndex++);
  auto curOpInfo = opInfo.get();
  linearOperation.push_back(std::move(opInfo));
  return curOpInfo;
}

void BufferAnalysis::UpdateOpBufferInfo(Operation *op, const ValueRange &results) {
  for (const Value &operand : results) {
    auto it = buffer2status.find(operand);
    if (it != buffer2status.end()) {
      continue;
    }

    Type opType = operand.getType();
    int64_t constBits = 0;
    Type elementType;

    if (auto memRefType = dyn_cast<MemRefType>(opType)) {
      // Handle MemRef type
      Value traceValue = tracebackMemRef(operand);
      auto tracedMemRefType = cast<MemRefType>(traceValue.getType());
      elementType = tracedMemRefType.getElementType();
      std::optional<int64_t> totalStaticSize = getStaticTotalSize(tracedMemRefType.getShape());
      if (!totalStaticSize.has_value()) {
        // Skip dynamic shapes for now
        continue;
      }
      constBits = totalStaticSize.value() * static_cast<int64_t>(elementType.getIntOrFloatBitWidth());
    } else if (opType.isIntOrFloat()) {
      // Handle scalar types (bf16, f32, i32, etc.)
      elementType = opType;
      constBits = static_cast<int64_t>(opType.getIntOrFloatBitWidth());
    } else {
      // Skip unsupported types
      continue;
    }

    bufferInfos[operand] = {op, constBits, elementType};
    buffer2status[operand] = BufferStatus::DEFINED;
  }
}

void BufferAnalysis::UpdateBufferAlias(Value buffer, Value aliasBuffer) {
  UpdateBufferAlias(buffer, aliasBuffer, false);
}

void BufferAnalysis::UpdateBufferAlias(Value buffer, Value aliasBuffer, bool hasCond) {
  if (!isMemRefValue(buffer) || !isMemRefValue(aliasBuffer)) {
    return;
  }
  SetVector<Value> buffers = GetAliasBuffers(buffer);
  SetVector<Value> aliasBuffers = GetAliasBuffers(aliasBuffer);
  buffers.insert(buffer);
  aliasBuffers.insert(aliasBuffer);

  // Update alias map info for each buffer
  UpdateBuffer2AliasVec(buffers, aliasBuffers, hasCond);
  UpdateBuffer2AliasVec(aliasBuffers, buffers, hasCond);

  // AllocOp is DEFINED, not AllocOp is UNDEFINED
  if (!isDefiningOpAllocLike(buffer)) {
    buffer2status[buffer] = BufferStatus::UNDEFINED;
  }
}

void BufferAnalysis::UpdateBuffer2AliasVec(const SetVector<Value> &buffers, const SetVector<Value> &aliasBuffers,
                                           bool hasCond) {
  for (auto buffer : buffers) {
    for (auto aliasValue : aliasBuffers) {
      auto it = std::find_if(buffer2AliasVec[buffer].begin(), buffer2AliasVec[buffer].end(),
                             [aliasValue](const std::pair<Value, bool> &p) { return p.first == aliasValue; });
      if (it != buffer2AliasVec[buffer].end()) {
        it->second = it->second || hasCond;
      } else {
        buffer2AliasVec[buffer].push_back(std::make_pair(aliasValue, hasCond));
      }
    }
  }
}

SetVector<Value> BufferAnalysis::GetAliasBuffers(Value aliasBuffer) {
  SetVector<Value> aliasBuffers;
  auto it = buffer2AliasVec.find(aliasBuffer);
  if (it != buffer2AliasVec.end()) {
    for (auto &pair : it->second) {
      aliasBuffers.insert(pair.first);
    }
  }
  return aliasBuffers;
}

void BufferAnalysis::MaterializeScalarResults(OpInfo *opInfo, const ValueRange &results) {
  SmallVector<Value> scalarResults;
  for (Value result : results) {
    if (isScalarTrackedValue(result)) {
      scalarResults.push_back(result);
    }
  }
  if (scalarResults.empty()) {
    return;
  }
  UpdateOpBufferInfo(opInfo->operation, ValueRange(scalarResults));
  UpdateOpGenInfo(opInfo, ValueRange(scalarResults));
}

void BufferAnalysis::KillTransferredScalarSources(OpInfo *opInfo, Region &region, const ValueRange &sources) {
  SetVector<Value> scalarSources;
  for (Value source : sources) {
    if (isScalarTrackedValue(source) && isDefinedInsideRegion(source, region)) {
      scalarSources.insert(source);
    }
  }

  for (Value source : scalarSources) {
    auto iter = buffer2status.find(source);
    if (iter == buffer2status.end() || iter->second != BufferStatus::GENED) {
      continue;
    }
    genKillMap[opInfo].kill.push_back(source);
    iter->second = BufferStatus::KILLED;
  }
}

void BufferAnalysis::UpdateOpGenInfo(OpInfo *opInfo, const ValueRange &results) {
  if (results.empty()) {
    return;
  }
  for (Value operand : results) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto buffer : aliasBuffers) {
      UpdateOperandGenInfo(opInfo, buffer);
    }
  }
}

void BufferAnalysis::UpdateOperandGenInfo(OpInfo *opInfo, Value operand) {
  auto iter_buffer = buffer2status.find(operand);
  if (iter_buffer == buffer2status.end()) {
    return;
  }
  if (iter_buffer->second == BufferStatus::DEFINED) {
    genKillMap[opInfo].gen.push_back(operand);
    buffer2status[iter_buffer->first] = BufferStatus::GENED;
  } else if (iter_buffer->second == BufferStatus::KILLED) {
    llvm_unreachable("The buffer memory has been released and cannot be used again!");
  }
}

void BufferAnalysis::OpKillHandle(OpInfo *opInfo, Liveness live, Block *block) {
  const auto *liveBlockInfo = live.getLiveness(block);
  assert(liveBlockInfo != nullptr && opInfo != nullptr);
  auto currentLiveValues = liveBlockInfo->currentlyLiveValues(opInfo->operation);
  if (currentLiveValues.empty()) {
    return;
  }
  SetVector<Value> liveValues(currentLiveValues.begin(), currentLiveValues.end());
  for (const Value &operand : liveValues) {
    UpdateOpKillInfo(opInfo, operand, live);
  }
}

void BufferAnalysis::UpdateOpKillInfo(OpInfo *opInfo, Value operand, Liveness live) {
  auto aliasBuffers = GetAliasBuffers(operand);
  aliasBuffers.insert(operand);
  for (Value aliasBuffer : aliasBuffers) {
    auto iterBuffer = buffer2status.find(aliasBuffer);
    if (iterBuffer == buffer2status.end()) {
      continue;  // Skip this alias, continue checking others
    }
    if (iterBuffer->second == BufferStatus::GENED &&
        isParentOpDominate(iterBuffer->first.getDefiningOp(), opInfo->operation) &&
        IsBufferDeadAfter(opInfo->operation, aliasBuffer, live)) {
      genKillMap[opInfo].kill.push_back(aliasBuffer);
      buffer2status[iterBuffer->first] = BufferStatus::KILLED;
    }
  }
}

bool BufferAnalysis::isParentOpDominate(Operation *op1, Operation *op2) const {
  if (op1 == nullptr || op2 == nullptr) return false;
  if (op2->getParentOp() == nullptr || op1->getParentOp() == nullptr) return false;
  return op2->getParentOp()->isAncestor(op1->getParentOp());
}

bool BufferAnalysis::IsBlockAfter(Block *afterBlock, Block *beforeBlock) const {
  if (afterBlock == beforeBlock) {
    return false;
  }
  assert(afterBlock != nullptr && beforeBlock != nullptr);
  mlir::Region *region = beforeBlock->getParent();
  assert(region != nullptr);
  for (auto it = region->begin(); it != region->end(); ++it) {
    if (&*it == beforeBlock) {
      for (++it; it != region->end(); ++it) {
        if (&*it == afterBlock) {
          return true;
        }
      }
      break;
    }
  }
  return false;
}

bool BufferAnalysis::IsDeadAfterBlock(Value value, Block *block) const {
  for (auto &useOperand : value.getUses()) {
    Operation *useOp = useOperand.getOwner();
    assert(useOp != nullptr);
    Block *useBlock = useOp->getBlock();
    if (useBlock != block && IsBlockAfter(useBlock, block)) {
      return false;
    }
  }
  return true;
}

bool BufferAnalysis::IsBufferDeadAfter(Operation *op, Value buffer, Liveness live) const {
  // Check if the buffer is dead after the given operation.
  // A buffer is dead after an operation if:
  // 1. The MLIR liveness analysis indicates it's dead after this op
  // 2. There are no uses of this buffer in blocks after the current block
  if (!live.isDeadAfter(buffer, op)) {
    return false;
  }
  if (!IsDeadAfterBlock(buffer, op->getBlock())) {
    return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Template Helper Implementation for ForOp
//===----------------------------------------------------------------------===//

template <typename ForOpType>
SmallVector<Value> BufferAnalysis::GetLiveBuffersInLoopImpl(ForOpType loopOp, Liveness live) {
  SmallVector<Value> allocBeforeLoopBuffers;
  const auto *liveBlockInfo = live.getLiveness(loopOp->getBlock());
  assert(liveBlockInfo != nullptr);
  auto currentLiveValues = liveBlockInfo->currentlyLiveValues(loopOp.getOperation());
  if (currentLiveValues.empty()) {
    return allocBeforeLoopBuffers;
  }
  SetVector<Value> currentLiveValuesOrder;
  for (auto buffer : currentLiveValues) {
    currentLiveValuesOrder.insert(buffer);
  }
  for (const Value &operand : currentLiveValuesOrder) {
    auto aliasBuffers = GetAliasBuffers(operand);
    aliasBuffers.insert(operand);
    for (auto Buffer : aliasBuffers) {
      auto iter = buffer2status.find(Buffer);
      if (iter != buffer2status.end()) allocBeforeLoopBuffers.push_back(Buffer);
    }
  }
  return allocBeforeLoopBuffers;
}

template <typename ForOpType>
void BufferAnalysis::UpdateForOpInitArgsAliasImpl(ForOpType forOp) {
  // Get init args - different API for affine vs scf
  SmallVector<Value> inits;
  if constexpr (std::is_same_v<ForOpType, mlir::affine::AffineForOp>) {
    inits = forOp.getInits();
  } else {
    inits = forOp.getInitArgs();
  }

  if (inits.empty()) {
    return;
  }
  assert(inits.size() == forOp.getRegionIterArgs().size());
  for (auto [i, arg] : llvm::enumerate(inits)) {
    Value iterArg = forOp.getRegionIterArgs()[i];
    if (isMemRefValue(iterArg) && isMemRefValue(arg)) {
      // init args alias region iter args
      UpdateBufferAlias(iterArg, arg);
    }
  }
}

template <typename ForOpType>
void BufferAnalysis::UpdateForOpBufferAliasImpl(ForOpType forOp) {
  if (forOp.getResults().empty()) {
    return;
  }

  SmallVector<Value> yieldedValues = getForOpYieldedValues(forOp);

  if (!forOp.getRegionIterArgs().empty()) {
    assert(yieldedValues.size() == forOp.getRegionIterArgs().size());
    SmallVector<Value> inits;
    if constexpr (std::is_same_v<ForOpType, mlir::affine::AffineForOp>) {
      inits = forOp.getInits();
    } else {
      inits = forOp.getInitArgs();
    }
    assert(inits.size() == forOp.getRegionIterArgs().size());
    for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
      if (isMemRefValue(yieldedValues[i]) && isMemRefValue(arg)) {
        // yielded values alias region iter args
        UpdateBufferAlias(yieldedValues[i], arg);
      }
    }
  }
  assert(forOp->getResults().size() == yieldedValues.size());
  for (auto [i, arg] : llvm::enumerate(yieldedValues)) {
    Value result = forOp->getResult(i);
    if (isMemRefValue(result) && isMemRefValue(arg)) {
      // forOp result values alias region iter yielded values
      UpdateBufferAlias(result, arg);
    }
  }
}

template <typename ForOpType>
void BufferAnalysis::RecursiveForOpImpl(ForOpType forOp, Liveness live) {
  auto forBeginSeq = UpdateLinearOperation(forOp.getOperation());
  UpdateOpGenInfo(forBeginSeq, GetLiveBuffersInLoopImpl(forOp, live));
  UpdateForOpInitArgsAliasImpl(forOp);
  RecursionIR(&forOp.getRegion(), live);
  UpdateForOpBufferAliasImpl(forOp);
  SmallVector<Value> yieldedValues = getForOpYieldedValues(forOp);
  auto forEndSeq = UpdateLinearOperation(forOp.getOperation());
  MaterializeScalarResults(forEndSeq, forOp->getResults());
  KillTransferredScalarSources(forEndSeq, forOp.getRegion(), ValueRange(yieldedValues));
  OpKillHandle(forEndSeq, live, forOp->getBlock());
}

//===----------------------------------------------------------------------===//
// Template Helper Implementation for IfOp
//===----------------------------------------------------------------------===//

template <typename IfOpType, typename YieldOpType>
void BufferAnalysis::UpdateIfOpBufferAliasImpl(IfOpType ifOp, YieldOpType yieldOp) {
  if (ifOp.getResults().empty()) {
    return;
  }
  assert(ifOp->getResults().size() == yieldOp->getOperands().size());
  for (auto [i, arg] : llvm::enumerate(yieldOp->getOperands())) {
    Value result = ifOp->getResult(i);
    if (isMemRefValue(result) && isMemRefValue(arg)) {
      // Multiple buffers involved, requiring one-to-one correspondence
      UpdateBufferAlias(result, arg, /*hasCond=*/true);
    }
  }
}

template <typename IfOpType, typename YieldOpType>
void BufferAnalysis::RecursiveIfOpImpl(IfOpType ifOp, Liveness live) {
  // Process the operation of if as follows:
  // %0 = if %cond -> (memref<16xf16>)
  //        yield %alloc0: memref<16xf16>
  //      else:
  //        yield %alloc1 : memref<16xf16>
  (void)UpdateLinearOperation(ifOp.getOperation());
  RecursionIR(&ifOp.getThenRegion(), live);
  auto curIfElse = UpdateLinearOperation(ifOp.getOperation());
  SmallVector<Value> thenYieldedValues;
  SmallVector<Value> elseYieldedValues;

  // Get then yield op
  if (!ifOp.getThenRegion().empty()) {
    auto &thenBlock = ifOp.getThenRegion().front();
    if (auto thenYield = dyn_cast<YieldOpType>(thenBlock.getTerminator())) {
      thenYieldedValues = SmallVector<Value>(thenYield->getOperands().begin(), thenYield->getOperands().end());
      UpdateIfOpBufferAliasImpl(ifOp, thenYield);
    }
  }

  auto curIfEnd = curIfElse;
  // Check if else region exists - different API for affine vs scf
  bool hasElse = false;
  if constexpr (std::is_same_v<IfOpType, mlir::affine::AffineIfOp>) {
    hasElse = ifOp.hasElse();
  } else {
    hasElse = !ifOp.getElseRegion().empty();
  }

  if (hasElse) {
    RecursionIR(&ifOp.getElseRegion(), live);
    curIfEnd = UpdateLinearOperation(ifOp.getOperation());
    // Get else yield op
    if (!ifOp.getElseRegion().empty()) {
      auto &elseBlock = ifOp.getElseRegion().front();
      if (auto elseYield = dyn_cast<YieldOpType>(elseBlock.getTerminator())) {
        elseYieldedValues = SmallVector<Value>(elseYield->getOperands().begin(), elseYield->getOperands().end());
        UpdateIfOpBufferAliasImpl(ifOp, elseYield);
      }
    }
  }
  MaterializeScalarResults(curIfEnd, ifOp->getResults());
  KillTransferredScalarSources(curIfEnd, ifOp.getThenRegion(), ValueRange(thenYieldedValues));
  KillTransferredScalarSources(curIfEnd, ifOp.getElseRegion(), ValueRange(elseYieldedValues));
  OpKillHandle(curIfEnd, live, ifOp->getBlock());
}

//===----------------------------------------------------------------------===//
// Common IR Recursion
//===----------------------------------------------------------------------===//

void BufferAnalysis::RecursionIR(Region *region, Liveness live) {
  auto result = region->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto affineForOp = dyn_cast<mlir::affine::AffineForOp>(op)) {
      // Recursive control flow - affine.for
      RecursiveForOpImpl(affineForOp, live);
      return WalkResult::skip();
    } else if (auto affineIfOp = dyn_cast<mlir::affine::AffineIfOp>(op)) {
      // Recursive control flow - affine.if
      RecursiveIfOpImpl<mlir::affine::AffineIfOp, mlir::affine::AffineYieldOp>(affineIfOp, live);
      return WalkResult::skip();
    } else if (auto scfForOp = dyn_cast<mlir::scf::ForOp>(op)) {
      // Recursive control flow - scf.for
      RecursiveForOpImpl(scfForOp, live);
      return WalkResult::skip();
    } else if (auto scfIfOp = dyn_cast<mlir::scf::IfOp>(op)) {
      // Recursive control flow - scf.if
      RecursiveIfOpImpl<mlir::scf::IfOp, mlir::scf::YieldOp>(scfIfOp, live);
      return WalkResult::skip();
    }

    // Process operation
    auto curOpInfo = UpdateLinearOperation(op);

    if (isa<bufferization::ToMemrefOp, bufferization::ToTensorOp, arith::ConstantOp>(op)) {
      return WalkResult::advance();
    }

    auto aliasPairs = getOperationAliasInfo(op);
    if (!aliasPairs.empty() && !isa<arith::SelectOp>(op)) {
      // Handle aliasing operations (subview, reshape, etc.), but not SelectOp
      for (auto aliasPair : aliasPairs) {
        UpdateBufferAlias(aliasPair.first, aliasPair.second);
      }
    } else if (isa<memref::AllocOp, memref::AllocaOp>(op)) {
      // Handle memref alloc
      UpdateOpBufferInfo(op, op->getResults());
    } else if (auto affineLoadOp = dyn_cast<mlir::affine::AffineLoadOp>(op)) {
      // AffineLoad produces a scalar result, register it as buffer
      UpdateOpBufferInfo(op, op->getResults());
      UpdateOpGenInfo(curOpInfo, op->getResults());
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto memrefLoadOp = dyn_cast<memref::LoadOp>(op)) {
      // memref.load produces a scalar result, register it as buffer (corresponding to affine.load)
      UpdateOpBufferInfo(op, op->getResults());
      UpdateOpGenInfo(curOpInfo, op->getResults());
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (auto affineStoreOp = dyn_cast<mlir::affine::AffineStoreOp>(op)) {
      UpdateStoreOpInfo(curOpInfo, affineStoreOp.getMemRef(), live);
    } else if (auto memrefStoreOp = dyn_cast<memref::StoreOp>(op)) {
      // memref.store (corresponding to affine.store)
      UpdateStoreOpInfo(curOpInfo, memrefStoreOp.getMemRef(), live);
    } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      UpdateOpBufferInfo(op, selectOp->getResults());
      UpdateOpGenInfo(curOpInfo, selectOp->getResults());
      OpKillHandle(curOpInfo, live, op->getBlock());
    } else if (op->getNumResults() > 0) {
      // Handle all other operations that produce results (arith ops, etc.)
      bool hasScalarOrMemRefResult = false;
      for (Value result : op->getResults()) {
        Type resultType = result.getType();
        if (resultType.isIntOrFloat() || isa<MemRefType>(resultType)) {
          hasScalarOrMemRefResult = true;
          break;
        }
      }
      if (hasScalarOrMemRefResult) {
        UpdateOpBufferInfo(op, op->getResults());
        UpdateOpGenInfo(curOpInfo, op->getResults());
        OpKillHandle(curOpInfo, live, op->getBlock());
      }
    }
    return WalkResult::advance();
  });
  if (result == WalkResult::interrupt()) {
    llvm_unreachable("BufferAnalysis Traverse IR Failed!");
  }
}

void BufferAnalysis::UpdateStoreOpInfo(OpInfo *opInfo, const Value storeValue, Liveness live) {
  // The src of memref store may also serve as a gen buffer
  SmallVector<Value, 1> storeValues;
  storeValues.push_back(storeValue);
  UpdateOpGenInfo(opInfo, storeValues);
  // Collect kill buffers corresponding to operation
  OpKillHandle(opInfo, live, opInfo->operation->getBlock());
}

int64_t BufferAnalysis::getExtraBufferSizeByFactor(Operation *op) const {
  // Reference: ExtraBuffer.cpp getExtraBufferSizeForReduceOp (line 207-239)
  // and BufferUtils.cpp getExtraBufferSizeByFactor
  //
  // ExtraBuffer.cpp logic:
  //   if (auto reduceOp = dyn_cast<linalg::ReduceOp>(op)) {
  //     std::optional<int64_t> bufSize =
  //         unit == BufferSizeUnit::ELEMENT
  //             ? utils::traceToAllocMaxSize(dpsOp.getDpsInputOperand(0)->get())
  //             : 1;  // FACTOR mode returns 1
  //     return bufSize;
  //   }
  //
  // BufferUtils.cpp applies this factor:
  //   return *res * reduceOp.getNumResults() * getValDataTypeWeight(input)

  // Check if this is a reduce operation by checking for reduction_axes attribute
  if (!op->hasAttr("reduction_axes")) {
    return 0;
  }

  if (op->getNumOperands() == 0) {
    return 0;
  }

  // Get the first input operand for data type weight calculation
  // Reference: ExtraBuffer.cpp line 226 - getDpsInputOperand(0)->get()
  Value inputOperand = op->getOperand(0);

  // Get extra buffer factor for reduce operation
  // Reference: ExtraBuffer.cpp line 224-228
  //   For linalg::ReduceOp with FACTOR unit, bufSize = 1
  constexpr int64_t kReduceFactor = 1;

  // For multi-reduce, multiply by the number of results
  // Reference: BufferUtils.cpp line 292-293
  //   return *res * static_cast<int64_t>(reduceOp.getNumResults()) * ...
  int64_t numResults = std::max(static_cast<int64_t>(1), static_cast<int64_t>(op->getNumResults()));

  // Get data type weight for the input operand
  // Reference: BufferUtils.cpp line 295
  //   getValDataTypeWeight(reduceOp.getDpsInputOperand(0)->get())
  uint32_t dataTypeWeight = getValDataTypeWeight(inputOperand, 1, dataTypeWeightMap);

  // Final calculation: factor * numResults * dataTypeWeight
  return kReduceFactor * numResults * static_cast<int64_t>(dataTypeWeight);
}

void BufferAnalysis::GenerateBufferLife() {
  int scopeTime = 0;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    auto it = genKillMap.find(linearOperation[i].get());
    if (it == genKillMap.end()) {
      scopeTime++;
      continue;
    }
    // Time given to buffer start
    for (const Value &genBuffer : it->second.gen) {
      std::unique_ptr<BufferLife> bufferLife = std::make_unique<BufferLife>(genBuffer);
      bufferLife->allocTime = scopeTime;
      buffer2Life[genBuffer] = std::move(bufferLife);
    }
    // Time given to buffer end
    for (const Value &killBuffer : it->second.kill) {
      auto iter = buffer2Life.find(killBuffer);
      if (iter != buffer2Life.end()) {
        iter->second->freeTime = scopeTime;
      }
    }
    scopeTime++;
  }

  // For buffers that are still alive at function end (e.g., returned buffers),
  // set their freeTime to the end of the function scope
  for (auto &[buffer, life] : buffer2Life) {
    if (life->freeTime == -1 && life->allocTime != -1) {
      life->freeTime = scopeTime > 0 ? scopeTime - 1 : 0;
    }
  }
}

uint32_t BufferAnalysis::getValMultiBuffer(const Value &value, uint32_t def) const {
  auto it = options.multiBufferCount.find(value);
  if (it != options.multiBufferCount.end()) {
    return static_cast<uint32_t>(it->second);
  }
  return def;
}

uint32_t BufferAnalysis::getValDataTypeWeight(const Value &value, uint32_t def,
                                              const DataTypeWeightMap &weightMap) const {
  auto it = weightMap.find(value);
  if (it != weightMap.end()) {
    return it->second;
  }
  return def;
}

void BufferAnalysis::printLiveRanges() const {
  llvm::outs() << "\n==================== Live Ranges ====================\n";
  llvm::outs() << "Considering " << liveRanges.size() << " live ranges:\n\n";

  for (size_t i = 0; i < liveRanges.size(); ++i) {
    const auto &liveRange = liveRanges[i];
    llvm::outs() << "Live Range #" << i << ":\n";
    llvm::outs() << "  Start: " << liveRange.start << ", End: " << liveRange.end << ", Weight: " << liveRange.weight
                 << "\n";

    // Print operation if available
    if (liveRange.op) {
      llvm::outs() << "  Operation: ";
      liveRange.op->print(llvm::outs(), OpPrintingFlags().skipRegions());
      llvm::outs() << "\n";
    }
  }
  llvm::outs() << "=====================================================\n\n";
}

void BufferAnalysis::printBufferAnalysisInfo() const {
  llvm::outs() << "\n================== Buffer Analysis ==================\n\n";

  // Print linear operations with gen/kill info
  llvm::outs() << "--- Linear Operations ---\n";
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    const auto &opInfo = linearOperation[i];
    llvm::outs() << "[" << i << "] ";
    if (opInfo->operation) {
      opInfo->operation->print(llvm::outs(), OpPrintingFlags().skipRegions());
    }
    llvm::outs() << "\n";

    // Print gen/kill info for this operation
    auto it = genKillMap.find(opInfo.get());
    if (it != genKillMap.end()) {
      const auto &genKill = it->second;

      if (!genKill.gen.empty()) {
        llvm::outs() << "      GEN: ";
        for (size_t j = 0; j < genKill.gen.size(); ++j) {
          if (j > 0) llvm::outs() << ", ";
          genKill.gen[j].print(llvm::outs());
        }
        llvm::outs() << "\n";
      }

      if (!genKill.kill.empty()) {
        llvm::outs() << "      KILL: ";
        for (size_t j = 0; j < genKill.kill.size(); ++j) {
          if (j > 0) llvm::outs() << ", ";
          genKill.kill[j].print(llvm::outs());
        }
        llvm::outs() << "\n";
      }
    }
  }
  llvm::outs() << "\n";

  // Print live ranges (start, end, weight)
  printLiveRanges();

  // Calculate and print max buffer
  int64_t maxBuffer = lineSweepRanges();
  llvm::outs() << "--- Max Buffer ---\n";
  llvm::outs() << "  MaxBuffer: " << maxBuffer << "\n";
  llvm::outs() << "\n";

  llvm::outs() << "=====================================================\n\n";
}

llvm::DenseSet<Value> BufferAnalysis::gatherInplaceReuseBuffers() const {
  // Collect inplace reuse pairs to adjust buffer weights
  // Inplace reuse: if a buffer is killed at the same time another is generated,
  // and they have the same element type, the new buffer can reuse the memory
  llvm::DenseSet<Value> inplaceReuseBuffers;
  for (auto &[opInfo, genKill] : genKillMap) {
    if (genKill.gen.empty() || genKill.kill.empty()) {
      continue;
    }
    // Check if any gen buffer can reuse a kill buffer
    for (const Value &genBuffer : genKill.gen) {
      if (std::any_of(genKill.kill.begin(), genKill.kill.end(),
                      [&](const Value &killBuffer) { return canInplaceReuse(genBuffer, killBuffer, bufferInfos); })) {
        inplaceReuseBuffers.insert(genBuffer);
      }
    }
  }
  return inplaceReuseBuffers;
}

void BufferAnalysis::gatherDataTypeWeights() {
  // Clear and initialize data type weights for normalization
  dataTypeWeightMap.clear();
  smallestTypeBits = std::numeric_limits<uint32_t>::max();

  for (auto &[buffer, info] : bufferInfos) {
    if (!info.elementType.isIntOrFloat()) continue;
    uint32_t typeBits = static_cast<uint32_t>(info.elementType.getIntOrFloatBitWidth());
    dataTypeWeightMap[buffer] = typeBits;
    smallestTypeBits = std::min(smallestTypeBits, typeBits);
  }

  // Normalize weights by smallest type bits
  if (smallestTypeBits == std::numeric_limits<uint32_t>::max()) {
    smallestTypeBits = 1;  // Avoid division by zero
  }
  for (auto &[buffer, bits] : dataTypeWeightMap) {
    bits = (bits + smallestTypeBits - 1) / smallestTypeBits;  // Round up
  }
}

void BufferAnalysis::createLiveRangesFromBufferLife(const llvm::DenseSet<Value> &inplaceReuseBuffers,
                                                    const DataTypeWeightMap &dataTypeWeightMap) {
  // Create live ranges from buffer life
  // Weight = multi-buffer count * data type weight (similar to BufferUtils.cpp)
  // Reference: BufferUtils.cpp gatherLiveRanges - weight is a ratio, not bits
  for (auto &[buffer, life] : buffer2Life) {
    if (life->allocTime == -1 || life->freeTime == -1) {
      continue;
    }
    auto it = bufferInfos.find(buffer);
    if (it == bufferInfos.end()) {
      continue;
    }

    // Skip inplace reuse buffers (they don't add to memory footprint)
    if (inplaceReuseBuffers.contains(buffer)) {
      continue;
    }

    // Calculate weight: only multi-buffer and data type weight (ratio, not bits)
    // Similar to BufferUtils.cpp: currentWeight = getValMultiBuffer(res) *
    //                                    getValMultiBuffer(destOp.getDpsInits()[idx], 1) *
    //                                    getValDataTypeWeight(res)
    uint32_t multiBuffer = getValMultiBuffer(buffer, 1);
    uint32_t dataTypeWeight = getValDataTypeWeight(buffer, 1, dataTypeWeightMap);
    // Weight is a ratio, not bits - represents the relative buffer usage
    int64_t weight = static_cast<int64_t>(multiBuffer) * static_cast<int64_t>(dataTypeWeight);

    // Get the operation that defines this buffer
    Operation *op = it->second.operation;
    liveRanges.emplace_back(static_cast<uint32_t>(life->allocTime), static_cast<uint32_t>(life->freeTime), weight, op);
  }
}

void BufferAnalysis::addExtraBufferLiveRanges(const DataTypeWeightMap &dataTypeWeightMap) {
  // Add extra buffer live ranges for reduce operations
  // Reference: BufferUtils.cpp gatherLiveRanges - adds extra weight for reduce ops
  //
  // The pattern in BufferUtils.cpp (line 421-428):
  //   if (auto extraWeight = getExtraBufferSizeByFactor(&op)) {
  //     extraWeight *= std::max(
  //         (uint32_t)1, getValMultiBuffer(op.getResult(0), 0) +
  //                          getValMultiBuffer(
  //                              cast<linalg::LinalgOp>(op).getDpsInits()[0], 0));
  //     LDBG("Appending " << op << " with " << extraWeight);
  //     liveRanges.emplace_back(currentOpIndex, currentOpIndex, extraWeight);
  //   }

  // Build a map from linearOperation index to scopeTime
  llvm::DenseMap<size_t, uint32_t> opIndexToScopeTime;
  uint32_t scopeTime = 0;
  for (size_t i = 0; i < linearOperation.size(); ++i) {
    opIndexToScopeTime[i] = scopeTime;
    scopeTime++;
  }

  for (size_t i = 0; i < linearOperation.size(); ++i) {
    Operation *op = linearOperation[i]->operation;
    if (auto extraWeight = getExtraBufferSizeByFactor(op)) {
      // The extra buffer exists only at the operation time
      // Calculate multi-buffer factor following BufferUtils.cpp pattern:
      //   std::max((uint32_t)1, getValMultiBuffer(result, 0) +
      //                        getValMultiBuffer(init, 0))
      //
      // For reduce operations, this accounts for double-buffering scenarios
      // where both the result buffer and the init buffer may be multi-buffered
      uint32_t resultMultiBuffer = 0;
      uint32_t initMultiBuffer = 0;

      if (op->getNumResults() > 0) {
        Value firstResult = op->getResult(0);
        // Use default 0 here (same as BufferUtils.cpp)
        resultMultiBuffer = getValMultiBuffer(firstResult, 0);
      }

      // For destination-style ops (like reduce), check init operand
      // In affine dialect, the last operand is typically the init/accumulator
      if (op->getNumOperands() > 0) {
        // The init operand is typically the last operand for reduce operations
        Value initOperand = op->getOperand(op->getNumOperands() - 1);
        initMultiBuffer = getValMultiBuffer(initOperand, 0);
      }

      // Calculate multi-buffer factor (same logic as BufferUtils.cpp line 422-425)
      int64_t multiBufferFactor = std::max(static_cast<uint32_t>(1), resultMultiBuffer + initMultiBuffer);
      extraWeight *= multiBufferFactor;

      // Get the scopeTime for this operation
      uint32_t opScopeTime = opIndexToScopeTime[i];

      // Add extra buffer live range at the operation time
      // The live range is point-like (start == end) for temporary buffers
      liveRanges.emplace_back(opScopeTime, opScopeTime, extraWeight, op);
    }
  }
}

void BufferAnalysis::gatherLiveRanges() {
  // Reference: BufferUtils.cpp's gatherLiveRanges
  // Collect live ranges with gen, kill, and weight information

  // Clear existing live ranges
  liveRanges.clear();

  // Step 1: Collect inplace reuse buffers
  auto inplaceReuseBuffers = gatherInplaceReuseBuffers();

  // Step 2: Create live ranges from buffer life (use cached dataTypeWeightMap)
  createLiveRangesFromBufferLife(inplaceReuseBuffers, dataTypeWeightMap);

  // Step 3: Add extra buffer live ranges for reduce operations
  addExtraBufferLiveRanges(dataTypeWeightMap);

  // Sort live ranges by start time
  llvm::sort(liveRanges);
}

int64_t BufferAnalysis::lineSweepRanges() const {
  // Reference: BufferUtils.cpp's lineSweepRanges
  // Line sweep algorithm to find max buffer
  llvm::PriorityQueue<WeightedEndPair, llvm::SmallVector<WeightedEndPair>, std::greater<WeightedEndPair>> earlyDone;

  int64_t maxBuffer = 0;
  int64_t currentBuffer = 0;

  for (const auto &liveRange : liveRanges) {
    if (liveRange.start == liveRange.end) {
      // WARN: dead operation or temporary buffer exists at this position
      // Similar to BufferUtils.cpp
    }
    // Remove buffers that have ended before the current start
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

std::pair<int64_t, uint32_t> BufferAnalysis::calculateMaxBuffer() {
  Region &funcRegion = func.getBody();
  Liveness live(func);
  // Recursively obtaining IR information
  RecursionIR(&funcRegion, live);
  // The lifetime of the buffer
  GenerateBufferLife();
  // Initialize data type weights (cached in member variables)
  gatherDataTypeWeights();
  // Gather live ranges for analysis and printing
  gatherLiveRanges();
  // Print information if requested
  if (options.printBufferInfo) {
    printBufferAnalysisInfo();
  }

  int64_t maxBuffer = lineSweepRanges();
  return {maxBuffer, smallestTypeBits};
}

//===----------------------------------------------------------------------===//
// Public API Implementation
//===----------------------------------------------------------------------===//

std::pair<int64_t, uint32_t> countMaxBuffer(mlir::func::FuncOp func, const BufferAnalysisOptions &options) {
  if (func.getBody().getBlocks().size() != 1) {
    return {-1, 0};
  }

  BufferAnalysis analysis(func, options);
  return analysis.calculateMaxBuffer();
}

}  // namespace akg
}  // namespace mlir
