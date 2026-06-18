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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/Support/raw_ostream.h"

#include "akg/Utils/AnalysisCommon.hpp"
#include "akg/Analysis/MemoryAnalysis.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Types (TU-local; public  peak API is declared in MemoryAnalysis.h).
//===----------------------------------------------------------------------===//

constexpr int64_t MULTI_BUFFER_NUM = 2;

enum class ReduceOpKind {
  Sum,
  Prod,
  Max,
  Min,
  MaxWithIndexLeft,
  MaxWithIndexRight,
  MinWithIndexLeft,
  MinWithIndexRight,
  Any,
  All,
  XorI,
  OrI,
  AndI,
  None
};

struct BufferInfo {
  int64_t Index;
  bool isScalar = false;
  int64_t totalBufferSize = 0;
  Type elementType;
  llvm::SmallVector<int64_t, 4> dimLoopIndices;
  bool isVirtual = false;
  int64_t OriginOpRecordIndex;
  Value originalValue;
  int64_t allocTime = -1;
  int64_t freeTime = -1;
  int64_t multiNum = 1;
  bool isValid = true;
  bool ignoreInplace = false;
};

// Conditional alias edge (`hasCond=true`): mutually exclusive buffer views (scalar select / scf.if).
struct ConditionalAliasEdge {
  int64_t a;
  int64_t b;
  bool hasCond;
};

// Branch-exclusive  buffers for slice-lowered `scf.if` (no results).
struct ScfIfBranchInfo {
  int64_t ifOpRecordIndex = -1;
  llvm::SmallVector<int64_t, 4> thenExclusive;
  llvm::SmallVector<int64_t, 4> elseExclusive;
};

struct OpRecord {
  int64_t Index;
  int64_t VirtualIndex = -1;
  Operation *sourceOp = nullptr;
  int64_t opTimeIndex = -1;
  bool isVirtualOp = false;
  llvm::SmallVector<int64_t> VirtualopIndexes;
  int64_t opType = -1;
  llvm::SmallVector<int64_t> inputBufferIndexes;
  int64_t outputBufferIndex;
  int64_t extraBufferSize = 0;
  Operation *forOPRegion = nullptr;
  llvm::SmallVector<int64_t> generatedBufferIndexes;
  llvm::SmallVector<int64_t> killedBufferIndexes;
};

// Union-find over **indices** into external `bufferInfoList_` (no owned `BufferInfo` copies).
class BufferInfoUnionFind {
 public:
  void clear() { parent_.clear(); }

  bool empty() const { return parent_.empty(); }
  size_t size() const { return parent_.size(); }

  void makeSet(int64_t idx) {
    growToBufferCount(idx + 1);
    parent_[idx] = idx;
  }
  // Extend union-find so indices `[0, newCount)` are valid (each new slot is its own root).
  void growToBufferCount(int64_t newCount) {
    while (static_cast<int64_t>(parent_.size()) < newCount) {
      parent_.push_back(static_cast<int64_t>(parent_.size()));
    }
  }

  int64_t find(int64_t idx) const {
    if (parent_[idx] != idx) {
      parent_[idx] = find(parent_[idx]);
    }
    return parent_[idx];
  }
  // The buffer of 'a' is inplace moved into 'b' (therefore 'a' is occurred earlier than 'b' in timeline)
  void unite(int64_t a, int64_t b) { parent_[b] = a; }

  bool sameSet(int64_t a, int64_t b) const { return find(a) == find(b); }

 private:
  mutable llvm::SmallVector<int64_t> parent_;
};

struct InplaceChainSummary {
  int64_t earliestAllocTime = -1;
  int64_t latestFreeTime = -1;
  int64_t maxBufferSizeBits = 0;
  int64_t maxMultiNum = 0;
  int64_t allocatedStartBits = 0;
  int64_t allocatedEndBits = 0;
  bool isTempBuffer = false;
  // Per multibuffer slot address range; each slot is planned independently.
  llvm::SmallVector<std::pair<int64_t, int64_t>, 4> slotAllocations;
};

struct ChainPlanningEntry {
  int64_t root;
  int64_t slotIndex;
};

struct PlacedChainInstance {
  InplaceChainSummary lifetime;
  int64_t allocatedStartBits = 0;
  int64_t allocatedEndBits = 0;
};

static bool chainLifetimesOverlap(const InplaceChainSummary &a, const InplaceChainSummary &b) {
  if (a.earliestAllocTime < 0 || a.latestFreeTime < 0 || b.earliestAllocTime < 0 || b.latestFreeTime < 0) {
    return true;
  }
  return !(a.latestFreeTime < b.earliestAllocTime || b.latestFreeTime < a.earliestAllocTime);
}

static int64_t findLowestChainAllocationStart(const InplaceChainSummary &cur, int64_t sizeBits,
                                              llvm::ArrayRef<PlacedChainInstance> placed) {
  llvm::SmallVector<std::pair<int64_t, int64_t>, 8> occupied;
  occupied.reserve(placed.size());
  for (const PlacedChainInstance &instance : placed) {
    if (!chainLifetimesOverlap(cur, instance.lifetime)) {
      continue;
    }
    occupied.emplace_back(instance.allocatedStartBits, instance.allocatedEndBits);
  }
  llvm::sort(occupied, [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });

  int64_t candidate = 0;
  for (const auto &[start, end] : occupied) {
    if (start - candidate >= sizeBits) {
      return candidate;
    }
    candidate = std::max(candidate, end);
  }
  return candidate;
}

static int64_t chainPeakAllocatedEndBits(const InplaceChainSummary &chain) {
  int64_t peak = chain.allocatedEndBits;
  for (const auto &[start, end] : chain.slotAllocations) {
    (void)start;
    peak = std::max(peak, end);
  }
  return peak;
}

static void accumulateBufferIntoChainSummary(InplaceChainSummary &chain, const BufferInfo &info) {
  if (info.allocTime >= 0) {
    if (chain.earliestAllocTime < 0) {
      chain.earliestAllocTime = info.allocTime;
    } else {
      chain.earliestAllocTime = std::min(chain.earliestAllocTime, info.allocTime);
    }
  }
  if (info.freeTime >= 0) {
    chain.latestFreeTime = std::max(chain.latestFreeTime, info.freeTime);
  }
  chain.maxBufferSizeBits = std::max(chain.maxBufferSizeBits, info.totalBufferSize);
  chain.maxMultiNum = std::max(chain.maxMultiNum, info.multiNum);
}

static void registerExtraBufferChainSummary(InplaceChainSummary &chain, const OpRecord &opRecord,
                                            ArrayRef<BufferInfo> bufferList) {
  chain.isTempBuffer = true;
  chain.earliestAllocTime = opRecord.opTimeIndex;
  chain.latestFreeTime = opRecord.opTimeIndex;
  chain.maxBufferSizeBits = opRecord.extraBufferSize;
  chain.maxMultiNum = 1;
  if (opRecord.outputBufferIndex < 0 || opRecord.outputBufferIndex >= static_cast<int64_t>(bufferList.size())) {
    return;
  }
  chain.maxMultiNum = std::max<int64_t>(1, bufferList[static_cast<size_t>(opRecord.outputBufferIndex)].multiNum);
}

static bool compareChainPlanningEntries(const ChainPlanningEntry &a, const ChainPlanningEntry &b,
                                        const llvm::DenseMap<int64_t, InplaceChainSummary> &summaryMap) {
  const InplaceChainSummary &sa = summaryMap.find(a.root)->second;
  const InplaceChainSummary &sb = summaryMap.find(b.root)->second;
  if (sa.maxMultiNum != sb.maxMultiNum) {
    return sa.maxMultiNum > sb.maxMultiNum;
  }
  if (sa.earliestAllocTime != sb.earliestAllocTime) {
    return sa.earliestAllocTime < sb.earliestAllocTime;
  }
  if (a.root != b.root) {
    return a.root < b.root;
  }
  return a.slotIndex < b.slotIndex;
}

static void placeChainPlanningEntry(const ChainPlanningEntry &planEntry,
                                    llvm::DenseMap<int64_t, InplaceChainSummary> &summaryMap,
                                    llvm::SmallVectorImpl<PlacedChainInstance> &placedInstances) {
  InplaceChainSummary &chainSum = summaryMap[planEntry.root];
  const int64_t sizeBits = chainSum.maxBufferSizeBits;
  const int64_t startBits = findLowestChainAllocationStart(chainSum, sizeBits, placedInstances);
  const int64_t endBits = startBits + sizeBits;

  PlacedChainInstance instance;
  instance.lifetime = chainSum;
  instance.allocatedStartBits = startBits;
  instance.allocatedEndBits = endBits;
  placedInstances.push_back(instance);

  chainSum.slotAllocations.push_back({startBits, endBits});
  if (chainSum.slotAllocations.size() == 1) {
    chainSum.allocatedStartBits = startBits;
    chainSum.allocatedEndBits = endBits;
    return;
  }
  chainSum.allocatedStartBits = std::min(chainSum.allocatedStartBits, startBits);
  chainSum.allocatedEndBits = std::max(chainSum.allocatedEndBits, endBits);
}

//===----------------------------------------------------------------------===//
//  peak estimation pipeline
//===----------------------------------------------------------------------===//

namespace {
struct EquivalentLoadKey {
  Operation *enclosingFor = nullptr;
  llvm::SmallVector<Value, 4> operands;
  DictionaryAttr attrs;

  bool operator==(const EquivalentLoadKey &other) const {
    return enclosingFor == other.enclosingFor && operands == other.operands && attrs == other.attrs;
  }
};

static EquivalentLoadKey makeEquivalentLoadKey(memref::LoadOp loadOp, Operation *enclosingFor) {
  EquivalentLoadKey key;
  key.enclosingFor = enclosingFor;
  key.operands.assign(loadOp->operand_begin(), loadOp->operand_end());
  key.attrs = loadOp->getAttrDictionary();
  return key;
}

static int64_t findEquivalentLoadBufferIndex(ArrayRef<std::pair<EquivalentLoadKey, int64_t>> entries,
                                             const EquivalentLoadKey &key) {
  for (const auto &[entry, bufIdx] : entries) {
    if (entry == key) {
      return bufIdx;
    }
  }
  return -1;
}

static void dumpEquivalentLoadKey(llvm::raw_ostream &os, const EquivalentLoadKey &key, OpPrintingFlags printFlags) {
  os << " operands=[";
  for (size_t i = 0; i < key.operands.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    key.operands[i].print(os);
  }
  os << "] attrs=" << key.attrs;
}

static void dumpInplaceChainSummaryLine(llvm::raw_ostream &os, const InplaceChainSummary &summary) {
  os << " chainMaxBits=" << summary.maxBufferSizeBits << " chainMultiNum=" << summary.maxMultiNum
     << " isTempBuffer=" << (summary.isTempBuffer ? 1 : 0) << " alloc=[" << summary.earliestAllocTime << ','
     << summary.latestFreeTime << "] addr=[" << summary.allocatedStartBits << ',' << summary.allocatedEndBits << ']';
  if (summary.slotAllocations.empty()) {
    return;
  }
  os << " slots=[";
  for (size_t i = 0; i < summary.slotAllocations.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    os << '[' << summary.slotAllocations[i].first << ',' << summary.slotAllocations[i].second << ']';
  }
  os << ']';
}

static void dumpTempBufferChainLine(llvm::raw_ostream &os, int64_t chainRoot, ArrayRef<OpRecord> perOpList,
                                    OpPrintingFlags printFlags) {
  const int64_t opIdx = -chainRoot - 1;
  if (opIdx < 0 || opIdx >= static_cast<int64_t>(perOpList.size())) {
    return;
  }
  const OpRecord &opRecord = perOpList[static_cast<size_t>(opIdx)];
  os << "  extra-buffer totalBits=" << opRecord.extraBufferSize << " op:";
  if (opRecord.sourceOp) {
    opRecord.sourceOp->print(os, printFlags);
  } else {
    os << "<unknown>";
  }
  os << '\n';
}

static void dumpChainBufferLine(llvm::raw_ostream &os, int64_t bufIdx, const BufferInfo &info,
                                ArrayRef<OpRecord> perOpList, OpPrintingFlags printFlags) {
  os << "              buffer[" << bufIdx << "] totalBits=" << info.totalBufferSize;
  if (info.multiNum > 1) {
    os << " multiNum=" << info.multiNum;
  }
  os << " op:";
  Operation *originOp = nullptr;
  if (info.OriginOpRecordIndex >= 0 && info.OriginOpRecordIndex < static_cast<int64_t>(perOpList.size())) {
    originOp = perOpList[static_cast<size_t>(info.OriginOpRecordIndex)].sourceOp;
  }
  if (originOp) {
    originOp->print(os, printFlags);
  } else if (info.originalValue) {
    os << '<';
    info.originalValue.print(os);
    os << '>';
  } else {
    os << "<unknown>";
  }
  os << '\n';
}

static void dumpEquivalentLoadDedupSection(llvm::raw_ostream &os,
                                           ArrayRef<std::pair<EquivalentLoadKey, int64_t>> entries,
                                           ArrayRef<BufferInfo> bufferList, OpPrintingFlags printFlags) {
  if (entries.empty()) {
    return;
  }
  os << "     equivalent-load-dedup entries=" << entries.size() << '\n';
  for (const auto &[key, bufIdx] : entries) {
    os << "       bufferIdx=" << bufIdx;
    if (bufIdx >= 0 && bufIdx < static_cast<int64_t>(bufferList.size())) {
      const BufferInfo &info = bufferList[static_cast<size_t>(bufIdx)];
      os << " lifetime=[" << info.allocTime << ',' << info.freeTime << ']';
    }
    os << ' ';
    dumpEquivalentLoadKey(os, key, printFlags);
    os << '\n';
  }
}
}  // namespace

class MemoryPeakEstimator {
 public:
  explicit MemoryPeakEstimator(PeakAnalysisInput input);

  void run(PeakAnalysisResult &out);

  void dumpInplaceChains(llvm::raw_ostream &os) const;

  const llvm::DenseMap<Operation *, int64_t> &getPerOpIndexMap() const { return perOpIndexMap_; }
  const llvm::SmallVector<OpRecord, 0> &getPerOpList() const { return perOpList_; }

  bool hasInlineBroadcastLoopDims(Operation *op) const;
  bool hasInlineTransposeLoopDims(Operation *op) const;
  int64_t getDimBound(const BufferInfo &info, size_t dimIdx) const;

 private:
  PeakAnalysisInput input_;

  llvm::SmallVector<scf::ForOp, 8> orderedForOps_;
  llvm::DenseMap<scf::ForOp, int64_t> forOpToIndex_;

  llvm::SmallVector<OpRecord, 0> perOpList_;
  llvm::SmallVector<BufferInfo> bufferInfoList_;

  llvm::DenseMap<Operation *, int64_t> perOpIndexMap_;
  llvm::DenseMap<Value, int64_t> bufferInfoIndexMap_;
  llvm::SmallVector<std::pair<EquivalentLoadKey, int64_t>, 16> equivalentLoadBufferMap_;

  SmallVector<int64_t, 4> TimelineOpIndexList;

  BufferInfoUnionFind bufferInfoUnionFind_;

  // Per inplace-chain stats keyed by union-find root buffer index.
  llvm::DenseMap<int64_t, InplaceChainSummary> inplaceChainSummary_;

  llvm::SmallVector<ConditionalAliasEdge, 8> conditionalAliasEdges_;
  llvm::SmallVector<ScfIfBranchInfo, 4> scfIfBranchInfos_;

  void initPerOp_();
  OpRecord &createBaseOpRecord_(Operation *op);
  void initPerOpForForOp_(scf::ForOp forOp);
  void initPerOpForIfOp_(scf::IfOp ifOp);
  void initPerOpForGenericOp_(Operation *op);
  void initReduceOps_(OpRecord &rec, BufferInfo &outputBufferInfo, Operation *op);
  void InferOutputBufferShape_(BufferInfo &outputBufferInfo, Operation *op, OpRecord &rec);
  void buildForOpWalkOrder_();
  int64_t totalBitsfromBuffer(const BufferInfo &info) const;
  void DimLoopIndicesToShape(ArrayRef<int64_t> dimLoopIndices, SmallVectorImpl<int64_t> &outBounds) const;
  void inferEnclosingDimLoopIndices_(Operation *op, SmallVectorImpl<int64_t> &outIndices) const;
  int64_t getOrCreateBlockInputBufferIndex_(Value input, const OpRecord &rec, Operation *op);
  void analyzeConditionalControlFlow_();

  void modelVirtualOps();
  void eliminateRedundantOps();

  void computeBufferLifetimes();
  void modelReduceExtraBuffer();
  void markMultiBuffer();

  void analyzeIntraOpInplace();
  void analyzeInterOpInplace();
  int64_t computePeakBits();

  // Innermost enclosing `scf.for` for `op`, skipping reduce-x loops outward.
  Operation *innermostEnclosingScfFor(Operation *op);
};

//=================== extra buffer size calculation ===================
namespace {
constexpr int kVectorBlockSizeBit = 256;
constexpr unsigned kIntrBytesPerBlock = 32;
constexpr unsigned kIntrBytesPerRepeat = 256;
constexpr unsigned kArgIndexBitWidth = 32;

inline int64_t ceilFactor(int64_t x, int64_t y) { return (x + y - 1) / y * y; }

int64_t getNumPerBlockTy(Type t) {
  const unsigned bw = getElementTypeOrSelf(t).getIntOrFloatBitWidth();
  return static_cast<int64_t>(kIntrBytesPerBlock / (bw / 8));
}

int64_t getNumPerRepeatTy(Type t) {
  const unsigned bw = getElementTypeOrSelf(t).getIntOrFloatBitWidth();
  return static_cast<int64_t>(kIntrBytesPerRepeat / (bw / 8));
}

bool isArgminOrArgmaxKind(ReduceOpKind op) {
  return op == ReduceOpKind::MaxWithIndexLeft || op == ReduceOpKind::MaxWithIndexRight ||
         op == ReduceOpKind::MinWithIndexLeft || op == ReduceOpKind::MinWithIndexRight;
}

// Extra scratch element count for **last-dim** reduce (HIVM vector rules). Returns 0 if none.
static int64_t refineReduceExtraBufferSizeDynamic(ShapedType srcType, int64_t srcAllocTotalSize, int64_t reductionDim,
                                                  ReduceOpKind arithOp) {
  Type eleType = srcType.getElementType();
  if (eleType.isInteger() && (reductionDim == srcType.getRank() - 1)) {
    return arithOp == ReduceOpKind::XorI ? 3 * srcAllocTotalSize : 2 * srcAllocTotalSize;
  }
  return srcAllocTotalSize;
}

static int64_t refineReduceExtraBufferSizeInteger(int64_t srcAllocTotalSize, int64_t rDim, int64_t aDim,
                                                  int numPerBlock, int numPerRepeat, ReduceOpKind arithOp) {
  if (rDim > numPerRepeat) {
    if (arithOp == ReduceOpKind::XorI) {
      return aDim * numPerRepeat * 2 + aDim * numPerBlock;
    }
    return aDim * numPerRepeat + aDim * numPerBlock;
  }
  return arithOp == ReduceOpKind::XorI ? 3 * srcAllocTotalSize : 2 * srcAllocTotalSize;
}

static int64_t refineReduceExtraBufferSizeFloat(ShapedType srcType, int64_t srcAllocTotalSize, int64_t reductionDim,
                                                int64_t rDim, int64_t aDim, int numPerBlock, int numPerRepeat,
                                                ReduceOpKind arithOp) {
  if ((arithOp == ReduceOpKind::Max || arithOp == ReduceOpKind::Min) && reductionDim == 0 && srcType.getRank() == 1) {
    return rDim <= numPerRepeat ? 0 : numPerRepeat;
  }
  if (rDim < numPerBlock) {
    return rDim % 2 == 0 ? srcAllocTotalSize / 2 : 0;
  }
  if (rDim >= numPerBlock && rDim <= numPerRepeat) {
    return 0;
  }
  if (rDim > numPerRepeat && rDim <= numPerRepeat * 2) {
    return aDim * numPerRepeat;
  }
  if (rDim > numPerRepeat * 2) {
    return srcAllocTotalSize / 2;
  }
  return 0;
}

static bool isIntegerLikeReduceOp(ReduceOpKind arithOp, Type eleType) {
  return eleType.isInteger() || arithOp == ReduceOpKind::Prod || arithOp == ReduceOpKind::OrI ||
         arithOp == ReduceOpKind::XorI;
}

int64_t refineReduceExtraBufferSize(ShapedType srcType, int64_t srcAllocTotalSize, int64_t reductionDim,
                                    ReduceOpKind arithOp) {
  if (!srcType.hasStaticShape()) {
    return refineReduceExtraBufferSizeDynamic(srcType, srcAllocTotalSize, reductionDim, arithOp);
  }

  Type eleType = srcType.getElementType();
  const int numPerBlock = static_cast<int>(getNumPerBlockTy(eleType));
  const int numPerRepeat = static_cast<int>(getNumPerRepeatTy(eleType));
  int64_t rDim = srcType.getShape()[reductionDim];
  int64_t aDim = reductionDim == 0 ? 1 : srcType.getShape()[0];

  if (isIntegerLikeReduceOp(arithOp, eleType)) {
    return refineReduceExtraBufferSizeInteger(srcAllocTotalSize, rDim, aDim, numPerBlock, numPerRepeat, arithOp);
  }
  if (eleType.isF32() || eleType.isF16()) {
    return refineReduceExtraBufferSizeFloat(srcType, srcAllocTotalSize, reductionDim, rDim, aDim, numPerBlock,
                                            numPerRepeat, arithOp);
  }
  return srcAllocTotalSize;
}

int64_t getExtraBufferSizeForReduceOpSingleDim(ShapedType srcType, int64_t srcAllocTotalSize, int64_t reductionDim,
                                               ReduceOpKind arithOp) {
  assert(srcAllocTotalSize >= 0);

  if (isArgminOrArgmaxKind(arithOp)) {
    int64_t rank = srcType.getRank();
    int64_t elementBitWidth = srcType.getElementTypeBitWidth();
    assert(kVectorBlockSizeBit % elementBitWidth == 0);
    int64_t numElemPerBlock = kVectorBlockSizeBit / elementBitWidth;
    if (reductionDim == rank - 1) {
      return numElemPerBlock;
    }
    if (srcType.hasStaticShape()) {
      int64_t reductionDimLength = srcType.getShape()[reductionDim];
      int64_t totalBitLength =
        ceilFactor(reductionDimLength * kArgIndexBitWidth, kVectorBlockSizeBit) + kVectorBlockSizeBit;
      return totalBitLength / elementBitWidth;
    }
    return ceilFactor(static_cast<int64_t>(std::llround(1.5 * static_cast<double>(srcAllocTotalSize))),
                        numElemPerBlock);
  }
  if (arithOp == ReduceOpKind::Sum || arithOp == ReduceOpKind::Max || arithOp == ReduceOpKind::Min ||
      arithOp == ReduceOpKind::Prod || arithOp == ReduceOpKind::OrI || arithOp == ReduceOpKind::AndI) {
    if (reductionDim != srcType.getRank() - 1) {
      return srcAllocTotalSize;
    }
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize, reductionDim, arithOp);
  }
  if (arithOp == ReduceOpKind::XorI) {
    if (reductionDim != srcType.getRank() - 1) {
      int64_t elementPerBlock = kVectorBlockSizeBit / srcType.getElementTypeBitWidth();
      return srcAllocTotalSize + elementPerBlock;
    }
    return refineReduceExtraBufferSize(srcType, srcAllocTotalSize, reductionDim, arithOp);
  }
  return srcAllocTotalSize;
}

int64_t getExtraBufferSizeForReduceOp(ArrayRef<int64_t> reductionDims, ShapedType srcType, int64_t srcAllocTotalSize,
                                      ReduceOpKind arithOp) {
  int64_t bufSize = 0;
  for (int64_t reductionDim : reductionDims) {
    bufSize =
      std::max(bufSize, getExtraBufferSizeForReduceOpSingleDim(srcType, srcAllocTotalSize, reductionDim, arithOp));
  }
  return bufSize;
}

ReduceOpKind reduceKindFromOp(Operation *op) {
  return TypeSwitch<Operation *, ReduceOpKind>(op)
    .Case<arith::AddFOp, arith::AddIOp>([](auto) { return ReduceOpKind::Sum; })
    .Case<arith::MulFOp, arith::MulIOp>([](auto) { return ReduceOpKind::Prod; })
    .Case<arith::MaximumFOp, arith::MaxNumFOp>([](auto) { return ReduceOpKind::Max; })
    .Case<arith::MinimumFOp, arith::MinNumFOp>([](auto) { return ReduceOpKind::Min; })
    .Case<arith::OrIOp>([](auto) { return ReduceOpKind::OrI; })
    .Case<arith::AndIOp>([](auto) { return ReduceOpKind::AndI; })
    .Case<arith::XOrIOp>([](auto) { return ReduceOpKind::XorI; })
    .Default([](Operation *) { return ReduceOpKind::Sum; });
}

namespace {
constexpr int64_t kOpTypeUnknown = -1;
constexpr int64_t kOpTypeMemRefLoad = 1;
constexpr int64_t kOpTypeMemRefStore = 2;
constexpr int64_t kOpTypeArithNegF = 3;
constexpr int64_t kOpTypeArithElemwise = 4;
constexpr int64_t kOpTypeArithReduce = 5;
constexpr int64_t kOpTypeArithTranspose = 6;
constexpr int64_t kOpTypeArithBroadcast = 7;
constexpr int64_t kOpTypeArithExt = 8;
constexpr int64_t kOpTypeScfIf = 9;
constexpr int64_t kOpTypeArithSelect = 10;
constexpr int64_t kOpTypeScfFor = 11;
}  // namespace

static int64_t OpTypeCode(Operation *op) {
  if (!op) {
    return kOpTypeUnknown;
  }
  return TypeSwitch<Operation *, int64_t>(op)
    .Case<memref::LoadOp>([](auto) { return kOpTypeMemRefLoad; })
    .Case<memref::StoreOp>([](auto) { return kOpTypeMemRefStore; })
    .Case<scf::ForOp>([](auto) { return kOpTypeScfFor; })
    .Case<scf::IfOp>([](auto) { return kOpTypeScfIf; })
    .Case<arith::SelectOp>([](auto) { return kOpTypeArithSelect; })
    .Case<arith::AddFOp, arith::AddIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::SubFOp, arith::SubIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::MulFOp, arith::MulIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::RemFOp, arith::RemSIOp, arith::RemUIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::CeilDivSIOp, arith::CeilDivUIOp, arith::FloorDivSIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::MaximumFOp, arith::MaxNumFOp, arith::MaxSIOp, arith::MaxUIOp>(
      [](auto) { return kOpTypeArithElemwise; })
    .Case<arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp>(
      [](auto) { return kOpTypeArithElemwise; })
    .Case<arith::OrIOp, arith::AndIOp, arith::XOrIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::ShLIOp, arith::ShRUIOp, arith::ShRSIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::CmpFOp, arith::CmpIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::NegFOp>([](auto) { return kOpTypeArithNegF; })
    .Case<arith::ConstantIndexOp, arith::ConstantIntOp, arith::ConstantFloatOp, arith::ConstantOp>(
      [](auto) { return kOpTypeArithElemwise; })
    .Case<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp>([](auto) { return kOpTypeArithExt; })
    .Case<arith::TruncFOp, arith::TruncIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::BitcastOp, arith::IndexCastOp, arith::IndexCastUIOp>([](auto) { return kOpTypeArithElemwise; })
    .Case<arith::UIToFPOp, arith::SIToFPOp, arith::FPToSIOp, arith::FPToUIOp>(
      [](auto) { return kOpTypeArithElemwise; })
    .Case<math::AbsFOp, math::AbsIOp, math::CeilOp, math::FloorOp, math::CosOp, math::SinOp, math::TanhOp, math::ExpOp,
          math::Exp2Op, math::LogOp, math::Log2Op, math::Log10Op, math::SqrtOp, math::RsqrtOp, math::PowFOp,
          math::Atan2Op, math::ErfOp>([](auto) { return kOpTypeArithElemwise; })
    .Default([](Operation *op) {
      if (op && (isa<arith::ArithDialect>(op->getDialect()) || isa<math::MathDialect>(op->getDialect()))) {
        return kOpTypeArithElemwise;
      }
      return kOpTypeUnknown;
    });
}

static bool isBufferEligibleForIntraOpInplace(const BufferInfo &info) { return info.isValid && !info.isScalar; }

// Returns true for compute ops other than load/store/for/if/transpose/reduce/broadcast.
static bool isArithOp(int64_t opType) {
  switch (opType) {
    case kOpTypeMemRefLoad:
    case kOpTypeMemRefStore:
    case kOpTypeScfFor:
    case kOpTypeScfIf:
    case kOpTypeArithReduce:
    case kOpTypeArithTranspose:
    case kOpTypeArithBroadcast:
      return false;
    default:
      return true;
  }
}

static bool isScalarArithSelectOp(arith::SelectOp selectOp, const llvm::DenseMap<Value, int64_t> &bufMap,
                                  llvm::ArrayRef<BufferInfo> bufferList) {
  auto isScalarOperand = [&](Value v) {
    auto it = bufMap.find(v);
    if (it == bufMap.end()) {
      return false;
    }
    const int64_t idx = it->second;
    if (idx < 0 || idx >= static_cast<int64_t>(bufferList.size())) {
      return false;
    }
    return bufferList[idx].isScalar;
  };
  return isScalarOperand(selectOp.getTrueValue()) && isScalarOperand(selectOp.getFalseValue());
}

// Maps HIVM VAdd/VSub/VMax/... ISA inplace ops at tiled arith IR level.
static bool isIsaInplaceElemwiseOp(Operation *op) {
  return isa<arith::AddFOp, arith::AddIOp, arith::SubFOp, arith::SubIOp, arith::MulFOp, arith::MulIOp,
             arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp, arith::MinNumFOp, arith::OrIOp, arith::AndIOp>(op);
}

// Mirrors `MemPlan::IsReuseHIVMOp` / `isReusableVSelOp` at tiled arith IR level.
static bool isReusableArithSelectInplace(arith::SelectOp selectOp, const BufferInfo &gen, const BufferInfo &kill,
                                         const llvm::DenseMap<Value, int64_t> &bufMap,
                                         llvm::ArrayRef<BufferInfo> bufferList) {
  if (isScalarArithSelectOp(selectOp, bufMap, bufferList)) {
    return false;
  }
  const unsigned genBits = getElementTypeOrSelf(gen.elementType).getIntOrFloatBitWidth();
  const unsigned killBits = getElementTypeOrSelf(kill.elementType).getIntOrFloatBitWidth();
  if (genBits != killBits || gen.dimLoopIndices != kill.dimLoopIndices) {
    return false;
  }
  auto isKillSrcBuffer = [&](Value v) {
    auto it = bufMap.find(v);
    return it != bufMap.end() && it->second == kill.Index;
  };
  if (!isKillSrcBuffer(selectOp.getTrueValue()) && !isKillSrcBuffer(selectOp.getFalseValue())) {
    return false;
  }
  // Mirrors isReusableVSelOp: do not inplace onto condition memref (i1 cond, i64 src).
  auto condType = getElementTypeOrSelf(selectOp.getCondition().getType());
  auto srcType = getElementTypeOrSelf(selectOp.getTrueValue().getType());
  if (srcType.isInteger(64) && condType.isInteger(1)) {
    auto condIt = bufMap.find(selectOp.getCondition());
    if (condIt != bufMap.end() && condIt->second == kill.Index) {
      return false;
    }
  }
  return true;
}

static bool isOpSupportingIntraOpInplace(const MemoryPeakEstimator &est, Operation *op, int64_t opType,
                                         const BufferInfo &gen, const BufferInfo &kill,
                                         const llvm::DenseMap<Value, int64_t> &bufMap,
                                         llvm::ArrayRef<BufferInfo> bufferList) {
  if (!op || op->hasAttr(kReductionAxesStr)) {
    return false;
  }

  if (isIsaInplaceElemwiseOp(op) && !est.hasInlineBroadcastLoopDims(op) && !est.hasInlineTransposeLoopDims(op)) {
    return true;
  }

  if (!isArithOp(opType)) {
    return false;
  }

  if (est.hasInlineTransposeLoopDims(op)) {
    return false;
  }

  if (est.hasInlineBroadcastLoopDims(op)) {
    return gen.dimLoopIndices == kill.dimLoopIndices;
  }

  if (gen.elementType == kill.elementType) {
    return true;
  }

  if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    return isReusableArithSelectInplace(selectOp, gen, kill, bufMap, bufferList);
  }

  const unsigned genBits = getElementTypeOrSelf(gen.elementType).getIntOrFloatBitWidth();
  const unsigned killBits = getElementTypeOrSelf(kill.elementType).getIntOrFloatBitWidth();

  if (killBits == genBits) {
    return true;
  }

  if (killBits % genBits != 0) {
    return false;
  }

  return !(gen.dimLoopIndices.size() > 1 && est.getDimBound(gen, 0) != 1 && est.getDimBound(kill, 0) != 1);
}

static bool areConditionallyAliased(int64_t a, int64_t b, llvm::ArrayRef<ConditionalAliasEdge> edges) {
  for (const ConditionalAliasEdge &edge : edges) {
    if (!edge.hasCond) {
      continue;
    }
    if ((edge.a == a && edge.b == b) || (edge.a == b && edge.b == a)) {
      return true;
    }
  }
  return false;
}

static bool canIntraOpInplaceReuse(const MemoryPeakEstimator &est, Operation *op, int64_t opType, const BufferInfo &gen,
                                   const BufferInfo &kill, const llvm::DenseMap<Value, int64_t> &bufMap,
                                   llvm::ArrayRef<BufferInfo> bufferList) {
  if (gen.ignoreInplace || kill.ignoreInplace) {
    return false;
  }
  if (!isBufferEligibleForIntraOpInplace(gen) || !isBufferEligibleForIntraOpInplace(kill)) {
    return false;
  }
  return isOpSupportingIntraOpInplace(est, op, opType, gen, kill, bufMap, bufferList);
}

//  buffers defined in `region` but not as results of ops in `siblingRegion`.
static void collectRegionExclusiveBuffers(Region &region, Region *siblingRegion,
                                            const llvm::DenseMap<Value, int64_t> &bufMap,
                                            llvm::ArrayRef<BufferInfo> bufferList,
                                            llvm::SmallVectorImpl<int64_t> &out) {
  llvm::SmallDenseSet<int64_t> siblingIndices;
  if (siblingRegion) {
    siblingRegion->walk([&](Operation *inner) {
      for (Value res : inner->getResults()) {
        auto it = bufMap.find(res);
        if (it != bufMap.end()) {
          siblingIndices.insert(it->second);
        }
      }
    });
  }

  llvm::SmallDenseSet<int64_t> seen;
  region.walk([&](Operation *inner) {
    if (isa<scf::YieldOp>(inner)) {
      return;
    }
    for (Value res : inner->getResults()) {
      auto it = bufMap.find(res);
      if (it == bufMap.end()) {
        continue;
      }
      int64_t idx = it->second;
      if (idx < 0 || idx >= static_cast<int64_t>(bufferList.size())) {
        continue;
      }
      if (!bufferList[idx].isValid || siblingIndices.contains(idx)) {
        continue;
      }
      if (seen.insert(idx).second) {
        out.push_back(idx);
      }
    }
  });
}

static void recordConditionalAlias(llvm::SmallVectorImpl<ConditionalAliasEdge> &edges, int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a == b) {
    return;
  }
  edges.push_back({a, b, /*hasCond=*/true});
}

static int64_t inplaceChainEarliestAllocTime(const BufferInfoUnionFind &uf, ArrayRef<BufferInfo> buffers,
                                             int64_t bufIdx) {
  const int64_t root = uf.find(bufIdx);
  int64_t earliest = -1;
  for (int64_t i = 0, n = static_cast<int64_t>(buffers.size()); i < n; ++i) {
    if (!buffers[i].isValid || uf.find(i) != root || buffers[i].allocTime < 0) {
      continue;
    }
    if (earliest < 0) {
      earliest = buffers[i].allocTime;
    } else {
      earliest = std::min(earliest, buffers[i].allocTime);
    }
  }
  return earliest;
}

static int64_t inplaceChainMaxMultiNum(const BufferInfoUnionFind &uf, ArrayRef<BufferInfo> buffers, int64_t bufIdx) {
  const int64_t root = uf.find(bufIdx);
  int64_t maxMulti = 1;
  for (int64_t i = 0, n = static_cast<int64_t>(buffers.size()); i < n; ++i) {
    if (!buffers[i].isValid || uf.find(i) != root) {
      continue;
    }
    maxMulti = std::max(maxMulti, buffers[i].multiNum);
  }
  return maxMulti;
}

// `unite(earlier, later)` per `BufferInfoUnionFind` convention.
static void uniteBuffersByTimeline(BufferInfoUnionFind &uf, ArrayRef<BufferInfo> buffers, int64_t idxA, int64_t idxB) {
  if (idxA == idxB || uf.sameSet(idxA, idxB)) {
    return;
  }
  const int64_t timeA = buffers[idxA].allocTime;
  const int64_t timeB = buffers[idxB].allocTime;
  if (timeA <= timeB) {
    uf.unite(idxA, idxB);
  } else {
    uf.unite(idxB, idxA);
  }
}

void parseReductionAxesAttr(Operation *op, SmallVectorImpl<int64_t> &out) {
  out.clear();
  auto arr = op->getAttrOfType<ArrayAttr>(kReductionAxesStr);
  if (!arr) {
    return;
  }
  for (Attribute a : arr) {
    if (auto ia = dyn_cast<IntegerAttr>(a)) {
      out.push_back(ia.getInt());
    }
  }
}

static void getEnclosingScfForOps(Operation *op, SmallVectorImpl<scf::ForOp> &loops) {
  SmallVector<scf::ForOp, 8> innermostFirst;
  Region *region = op->getParentRegion();
  while (region) {
    Operation *parent = region->getParentOp();
    if (!parent) {
      break;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      innermostFirst.push_back(forOp);
    }
    region = parent->getParentRegion();
  }
  loops.clear();
  loops.reserve(innermostFirst.size());
  for (auto it = innermostFirst.rbegin(); it != innermostFirst.rend(); ++it) {
    loops.push_back(*it);
  }
}

static bool isOpInsideRegion(Operation *op, Region *region) {
  for (Region *cur = op->getParentRegion(); cur; cur = cur->getParentRegion()) {
    if (cur == region) {
      return true;
    }
  }
  return false;
}

static bool isScfRegionBlockInput(Value val, Operation *op) {
  auto blockArg = dyn_cast<BlockArgument>(val);
  if (!blockArg) {
    return false;
  }
  Block *ownerBlock = blockArg.getOwner();
  Region *ownerRegion = ownerBlock->getParent();
  if (!ownerRegion || !isOpInsideRegion(op, ownerRegion)) {
    return false;
  }
  Operation *parentOp = ownerRegion->getParentOp();
  if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
    if (ownerBlock != &forOp.getRegion().front()) {
      return false;
    }
    return blockArg.getArgNumber() > 0;
  }
  if (isa<scf::IfOp>(parentOp)) {
    return blockArg.getArgNumber() > 0;
  }
  return false;
}

static std::optional<int64_t> memrefDimForIvOnAccess(Operation *MemrefOp, Value iv) {
  if (auto loadOp = dyn_cast<affine::AffineLoadOp>(MemrefOp)) {
    for (auto [d, idxVal] : llvm::enumerate(loadOp.getIndices())) {
      if (idxVal == iv) {
        return static_cast<int64_t>(d);
      }
    }
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(MemrefOp)) {
    for (auto [d, idxVal] : llvm::enumerate(loadOp.getIndices())) {
      if (idxVal == iv) {
        return static_cast<int64_t>(d);
      }
    }
  } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(MemrefOp)) {
    for (auto [d, idxVal] : llvm::enumerate(storeOp.getIndices())) {
      if (idxVal == iv) {
        return static_cast<int64_t>(d);
      }
    }
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(MemrefOp)) {
    for (auto [d, idxVal] : llvm::enumerate(storeOp.getIndices())) {
      if (idxVal == iv) {
        return static_cast<int64_t>(d);
      }
    }
  }
  return std::nullopt;
}

// Map memref dims to 1-based for-op ids using this MemrefOp's indices and enclosing loops.
static void inferDimLoopIndices(Operation *MemrefOp, int64_t memrefRank,
                                const llvm::DenseMap<scf::ForOp, int64_t> &forOpToIndex,
                                SmallVectorImpl<int64_t> &outIndices) {
  outIndices.assign(std::max<int64_t>(memrefRank, 0), 0);
  if (!MemrefOp || memrefRank <= 0) {
    return;
  }

  SmallVector<scf::ForOp, 8> loops;
  getEnclosingScfForOps(MemrefOp, loops);
  for (scf::ForOp forOp : loops) {
    auto it = forOpToIndex.find(forOp);
    if (it == forOpToIndex.end()) {
      continue;
    }
    std::optional<int64_t> dim = memrefDimForIvOnAccess(MemrefOp, forOp.getInductionVar());
    if (!dim) {
      continue;
    }
    outIndices[*dim] = it->second;
  }
}

static int64_t DimLoopIndexToBound(int64_t loopIndex, ArrayRef<scf::ForOp> orderedForOps,
                                   const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop) {
  if (loopIndex <= 0 || loopIndex > static_cast<int64_t>(orderedForOps.size())) {
    return 1;
  }
  scf::ForOp forOp = orderedForOps[static_cast<size_t>(loopIndex - 1)];
  auto it = tileUpperBoundPerLoop.find(forOp);
  if (it != tileUpperBoundPerLoop.end() && it->second > 0) {
    return it->second;
  }
  return 1;
}

static int64_t totalBitsFromShape(ArrayRef<int64_t> shape, Type elementType, bool alignTo256Block = true) {
  int64_t elems = std::accumulate(shape.begin(), shape.end(), int64_t(1), std::multiplies<int64_t>());
  int64_t bits = elems * static_cast<int64_t>(getElementTypeOrSelf(elementType).getIntOrFloatBitWidth());
  if (!alignTo256Block || bits <= 0) {
    return bits;
  }
  return ceilFactor(bits, kVectorBlockSizeBit);
}

static int64_t bufferTotalBitsFromDimLoopIndices(ArrayRef<int64_t> dimLoopIndices, Type elementType,
                                                 ArrayRef<scf::ForOp> orderedForOps,
                                                 const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop,
                                                 bool alignBufferSizeTo256Bits) {
  if (dimLoopIndices.empty()) {
    return static_cast<int64_t>(getElementTypeOrSelf(elementType).getIntOrFloatBitWidth());
  }
  SmallVector<int64_t, 4> bounds;
  bounds.reserve(dimLoopIndices.size());
  std::transform(dimLoopIndices.begin(), dimLoopIndices.end(), std::back_inserter(bounds), [&](int64_t loopIndex) {
    return DimLoopIndexToBound(loopIndex, orderedForOps, tileUpperBoundPerLoop);
  });
  return totalBitsFromShape(bounds, elementType, alignBufferSizeTo256Bits);
}

static void collectInputShapeInfo(ArrayRef<int64_t> inputBufferIndexes, ArrayRef<BufferInfo> bufferList,
                                  bool &hasScalarInput, bool &hasVectorInput,
                                  SmallVectorImpl<ArrayRef<int64_t>> &nonScalarDimLoopIndices) {
  hasScalarInput = false;
  hasVectorInput = false;
  nonScalarDimLoopIndices.clear();
  for (int64_t index : inputBufferIndexes) {
    if (index < 0 || index >= static_cast<int64_t>(bufferList.size())) {
      continue;
    }
    const BufferInfo &inputInfo = bufferList[static_cast<size_t>(index)];
    if (inputInfo.isScalar) {
      hasScalarInput = true;
      continue;
    }
    hasVectorInput = true;
    nonScalarDimLoopIndices.push_back(inputInfo.dimLoopIndices);
  }
}

static bool allDimLoopIndicesSame(ArrayRef<ArrayRef<int64_t>> nonScalarDimLoopIndices) {
  for (size_t i = 1; i < nonScalarDimLoopIndices.size(); ++i) {
    if (nonScalarDimLoopIndices[i] != nonScalarDimLoopIndices.front()) {
      return false;
    }
  }
  return true;
}

static void assignBroadcastDimLoopIndices(ArrayRef<ArrayRef<int64_t>> nonScalarDimLoopIndices,
                                          SmallVectorImpl<int64_t> &outDimLoopIndices) {
  llvm::SmallSet<int64_t, 8> seen;
  for (ArrayRef<int64_t> dimLoopIndices : nonScalarDimLoopIndices) {
    for (int64_t loopIndex : dimLoopIndices) {
      if (seen.insert(loopIndex).second) {
        outDimLoopIndices.push_back(loopIndex);
      }
    }
  }
}

static void assignOutputDimLoopIndices(ArrayRef<ArrayRef<int64_t>> nonScalarDimLoopIndices,
                                       SmallVectorImpl<int64_t> &outDimLoopIndices, bool &allSame) {
  allSame = nonScalarDimLoopIndices.size() <= 1 || allDimLoopIndicesSame(nonScalarDimLoopIndices);
  if (allSame) {
    outDimLoopIndices.assign(nonScalarDimLoopIndices.front().begin(), nonScalarDimLoopIndices.front().end());
    return;
  }
  assignBroadcastDimLoopIndices(nonScalarDimLoopIndices, outDimLoopIndices);
}

static void setInlineBroadcastExtraBufferSize(ArrayRef<int64_t> inputBufferIndexes, ArrayRef<BufferInfo> bufferList,
                                              ArrayRef<int64_t> outputDimLoopIndices, bool needExtraBuffer,
                                              OpRecord &rec, ArrayRef<scf::ForOp> orderedForOps,
                                              const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop,
                                              bool alignBufferSizeTo256Bits) {
  if (!needExtraBuffer || rec.extraBufferSize != 0) {
    return;
  }

  int64_t extraBufferSize = 0;
  for (int64_t index : inputBufferIndexes) {
    if (index < 0 || index >= static_cast<int64_t>(bufferList.size())) {
      continue;
    }
    const BufferInfo &inputInfo = bufferList[static_cast<size_t>(index)];
    if (ArrayRef(inputInfo.dimLoopIndices) != outputDimLoopIndices) {
      extraBufferSize += bufferTotalBitsFromDimLoopIndices(outputDimLoopIndices, inputInfo.elementType, orderedForOps,
                                                           tileUpperBoundPerLoop, alignBufferSizeTo256Bits);
    }
  }
  if (extraBufferSize > 0) {
    rec.extraBufferSize = extraBufferSize;
  }
}

static void InferShapeFromInput(ArrayRef<int64_t> inputBufferIndexes, ArrayRef<BufferInfo> bufferList,
                                BufferInfo &outputBufferInfo, OpRecord &rec, ArrayRef<scf::ForOp> orderedForOps,
                                const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop,
                                bool alignBufferSizeTo256Bits) {
  outputBufferInfo.isScalar = true;
  outputBufferInfo.dimLoopIndices.clear();

  bool hasScalarInput = false;
  bool hasVectorInput = false;
  SmallVector<ArrayRef<int64_t>, 4> nonScalarDimLoopIndices;
  collectInputShapeInfo(inputBufferIndexes, bufferList, hasScalarInput, hasVectorInput, nonScalarDimLoopIndices);
  if (nonScalarDimLoopIndices.empty()) {
    return;
  }

  outputBufferInfo.isScalar = false;
  bool allSame = true;
  assignOutputDimLoopIndices(nonScalarDimLoopIndices, outputBufferInfo.dimLoopIndices, allSame);

  const bool needExtraBuffer = (hasScalarInput && hasVectorInput) || (nonScalarDimLoopIndices.size() >= 2 && !allSame);
  setInlineBroadcastExtraBufferSize(inputBufferIndexes, bufferList, outputBufferInfo.dimLoopIndices, needExtraBuffer,
                                    rec, orderedForOps, tileUpperBoundPerLoop, alignBufferSizeTo256Bits);
}

struct ScfForBoundSignature {
  Value lb;
  Value ub;
  Value step;
  bool operator==(const ScfForBoundSignature &other) const {
    return lb == other.lb && ub == other.ub && step == other.step;
  }
};

// For loops with the same (lb, ub, step) iterate the same axis; copy a non-zero tile bound
// within each equivalence class when some entries in `tileUpperBoundPerLoop` are zero/missing.
static void propagateTileUpperBoundsInMap(PeakAnalysisInput &input) {
  func::FuncOp func = input.func;
  if (!func) {
    return;
  }

  llvm::SmallVector<std::pair<ScfForBoundSignature, SmallVector<scf::ForOp, 8>>, 8> boundGroups;
  func.walk([&](scf::ForOp forOp) {
    ScfForBoundSignature signature{forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()};
    for (auto &[existing, forOps] : boundGroups) {
      if (existing == signature) {
        forOps.push_back(forOp);
        return;
      }
    }
    boundGroups.push_back({signature, {forOp}});
  });

  for (const auto &[signature, forOps] : boundGroups) {
    (void)signature;
    int64_t validBound = 0;
    for (scf::ForOp forOp : forOps) {
      auto it = input.tileUpperBoundPerLoop.find(forOp);
      if (it != input.tileUpperBoundPerLoop.end() && it->second > 0) {
        validBound = std::max(validBound, it->second);
      }
    }
    if (validBound <= 0) {
      continue;
    }
    for (scf::ForOp forOp : forOps) {
      int64_t current = 0;
      if (auto it = input.tileUpperBoundPerLoop.find(forOp); it != input.tileUpperBoundPerLoop.end()) {
        current = it->second;
      }
      if (current <= 0) {
        input.tileUpperBoundPerLoop[forOp] = validBound;
      }
    }
  }
}

}  // namespace

constexpr int64_t MultibufferFactor = 2;

MemoryPeakEstimator::MemoryPeakEstimator(PeakAnalysisInput input) : input_(std::move(input)) {}

void MemoryPeakEstimator::buildForOpWalkOrder_() {
  if (!orderedForOps_.empty()) {
    return;
  }
  if (!input_.func) {
    return;
  }
  input_.func.walk([&](scf::ForOp forOp) {
    forOpToIndex_[forOp] = static_cast<int64_t>(orderedForOps_.size()) + 1;
    orderedForOps_.push_back(forOp);
  });
}

void MemoryPeakEstimator::DimLoopIndicesToShape(ArrayRef<int64_t> dimLoopIndices,
                                                SmallVectorImpl<int64_t> &outBounds) const {
  outBounds.clear();
  outBounds.reserve(dimLoopIndices.size());
  std::transform(dimLoopIndices.begin(), dimLoopIndices.end(), std::back_inserter(outBounds), [&](int64_t loopIndex) {
    return DimLoopIndexToBound(loopIndex, orderedForOps_, input_.tileUpperBoundPerLoop);
  });
}

int64_t MemoryPeakEstimator::getDimBound(const BufferInfo &info, size_t dimIdx) const {
  if (dimIdx >= info.dimLoopIndices.size()) {
    return 1;
  }
  SmallVector<int64_t, 4> bounds;
  DimLoopIndicesToShape(info.dimLoopIndices, bounds);
  return bounds[dimIdx];
}

int64_t MemoryPeakEstimator::totalBitsfromBuffer(const BufferInfo &info) const {
  if (info.isScalar || info.dimLoopIndices.empty()) {
    return static_cast<int64_t>(getElementTypeOrSelf(info.elementType).getIntOrFloatBitWidth());
  }
  SmallVector<int64_t, 4> bounds;
  DimLoopIndicesToShape(info.dimLoopIndices, bounds);
  return totalBitsFromShape(bounds, info.elementType, input_.alignBufferSizeTo256Bits);
}

static bool isInsideScfForBody(Operation *op) {
  Region *region = op->getParentRegion();
  while (region) {
    Operation *parent = region->getParentOp();
    if (!parent) {
      return false;
    }
    if (isa<scf::ForOp>(parent)) {
      return true;
    }
    region = parent->getParentRegion();
  }
  return false;
}

void MemoryPeakEstimator::inferEnclosingDimLoopIndices_(Operation *op, SmallVectorImpl<int64_t> &outIndices) const {
  SmallVector<scf::ForOp, 8> loops;
  getEnclosingScfForOps(op, loops);
  llvm::DenseSet<int64_t> seen;
  outIndices.clear();
  for (scf::ForOp forOp : loops) {
    auto reduceIt = input_.isReduceXorAllVectorizeLoop.find(forOp);
    if (reduceIt != input_.isReduceXorAllVectorizeLoop.end() && reduceIt->second) {
      continue;
    }
    auto it = forOpToIndex_.find(forOp);
    if (it == forOpToIndex_.end()) {
      continue;
    }
    if (seen.insert(it->second).second) {
      outIndices.push_back(it->second);
    }
  }
}

int64_t MemoryPeakEstimator::getOrCreateBlockInputBufferIndex_(Value input, const OpRecord &rec, Operation *op) {
  if (auto mapIt = bufferInfoIndexMap_.find(input); mapIt != bufferInfoIndexMap_.end()) {
    return mapIt->second;
  }
  if (!isScfRegionBlockInput(input, op)) {
    return -1;
  }

  const int64_t valIndex = static_cast<int64_t>(bufferInfoList_.size());
  bufferInfoList_.push_back(BufferInfo{});
  BufferInfo &inputBufferInfo = bufferInfoList_[static_cast<size_t>(valIndex)];
  inputBufferInfo.Index = valIndex;
  inputBufferInfo.OriginOpRecordIndex = rec.Index;
  inputBufferInfo.elementType = input.getType();
  inputBufferInfo.originalValue = input;
  inferEnclosingDimLoopIndices_(op, inputBufferInfo.dimLoopIndices);
  inputBufferInfo.isScalar = inputBufferInfo.dimLoopIndices.empty();
  inputBufferInfo.isValid = isInsideScfForBody(op);
  inputBufferInfo.totalBufferSize = totalBitsfromBuffer(inputBufferInfo);
  bufferInfoIndexMap_[input] = valIndex;
  return valIndex;
}

Operation *MemoryPeakEstimator::innermostEnclosingScfFor(Operation *op) {
  Region *region = op->getParentRegion();
  while (region) {
    Operation *parent = region->getParentOp();
    if (!parent) {
      return nullptr;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      auto it = input_.isReduceXorAllVectorizeLoop.find(forOp);
      if (it != input_.isReduceXorAllVectorizeLoop.end() && it->second) {
        region = parent->getParentRegion();
        continue;
      }
      return parent;
    }
    region = parent->getParentRegion();
  }
  return nullptr;
}

static int64_t bufferIndexFromYieldValue(Value yieldVal, const llvm::DenseMap<Value, int64_t> &bufferInfoIndexMap,
                                         const llvm::DenseMap<Operation *, int64_t> &perOpIndexMap,
                                         llvm::ArrayRef<OpRecord> perOpList) {
  if (Operation *definingOp = yieldVal.getDefiningOp()) {
    if (auto defOpIt = perOpIndexMap.find(definingOp); defOpIt != perOpIndexMap.end()) {
      return perOpList[defOpIt->second].outputBufferIndex;
    }
  }
  if (auto yieldIt = bufferInfoIndexMap.find(yieldVal); yieldIt != bufferInfoIndexMap.end()) {
    return yieldIt->second;
  }
  return -1;
}

static void mapYieldedResultsToBufferIndexes(ArrayRef<Value> results, ArrayRef<Value> yieldValues,
                                             llvm::DenseMap<Value, int64_t> &bufferInfoIndexMap,
                                             const llvm::DenseMap<Operation *, int64_t> &perOpIndexMap,
                                             llvm::ArrayRef<OpRecord> perOpList) {
  const unsigned n = std::min(results.size(), yieldValues.size());
  for (unsigned i = 0; i < n; ++i) {
    const int64_t bufIdx =
      bufferIndexFromYieldValue(yieldValues[i], bufferInfoIndexMap, perOpIndexMap, perOpList);
    if (bufIdx >= 0) {
      bufferInfoIndexMap[results[i]] = bufIdx;
    }
  }
}

OpRecord &MemoryPeakEstimator::createBaseOpRecord_(Operation *op) {
  int64_t opIndex = perOpList_.size();
  perOpList_.push_back(OpRecord{});
  OpRecord &rec = perOpList_[opIndex];
  perOpIndexMap_[op] = opIndex;
  rec.Index = opIndex;
  rec.sourceOp = op;
  rec.forOPRegion = innermostEnclosingScfFor(op);
  rec.opType = OpTypeCode(op);
  rec.outputBufferIndex = -1;
  return rec;
}

void MemoryPeakEstimator::initPerOpForForOp_(scf::ForOp forOp) {
  OpRecord &rec = createBaseOpRecord_(forOp);
  if (forOp.getNumResults() == 0) {
    return;
  }
  auto yieldOp = cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
  SmallVector<Value, 4> forResults(forOp.getResults().begin(), forOp.getResults().end());
  SmallVector<Value, 4> yieldValues(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
  mapYieldedResultsToBufferIndexes(forResults, yieldValues, bufferInfoIndexMap_, perOpIndexMap_, perOpList_);
  if (auto outIt = bufferInfoIndexMap_.find(forOp.getResult(0)); outIt != bufferInfoIndexMap_.end()) {
    rec.outputBufferIndex = outIt->second;
  }
}

void MemoryPeakEstimator::initPerOpForIfOp_(scf::IfOp ifOp) {
  OpRecord &rec = createBaseOpRecord_(ifOp);
  if (ifOp.getNumResults() == 0) {
    return;
  }
  scf::YieldOp yieldOp = nullptr;
  if (Block *elseBlock = ifOp.elseBlock()) {
    yieldOp = cast<scf::YieldOp>(elseBlock->getTerminator());
  } else {
    yieldOp = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  }
  SmallVector<Value, 4> ifResults(ifOp.getResults().begin(), ifOp.getResults().end());
  SmallVector<Value, 4> yieldValues(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
  mapYieldedResultsToBufferIndexes(ifResults, yieldValues, bufferInfoIndexMap_, perOpIndexMap_, perOpList_);
  if (auto outIt = bufferInfoIndexMap_.find(ifOp.getResult(0)); outIt != bufferInfoIndexMap_.end()) {
    rec.outputBufferIndex = outIt->second;
  }
}

void MemoryPeakEstimator::InferOutputBufferShape_(BufferInfo &outputBufferInfo, Operation *op, OpRecord &rec) {
  if (isa<memref::LoadOp>(op) || isa<memref::StoreOp>(op)) {
    Value memrefVal;
    if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
      memrefVal = loadOp.getMemref();
    } else {
      memrefVal = cast<memref::StoreOp>(op).getMemRef();
    }
    auto memTy = cast<MemRefType>(memrefVal.getType());
    if (memTy.getRank() == 0) {
      outputBufferInfo.isScalar = true;
      outputBufferInfo.dimLoopIndices.clear();
      return;
    }
    SmallVector<int64_t, 8> dimLoopIndices;
    inferDimLoopIndices(op, memTy.getRank(), forOpToIndex_, dimLoopIndices);
    outputBufferInfo.dimLoopIndices.assign(dimLoopIndices.begin(), dimLoopIndices.end());
    if (dimLoopIndices.empty()) {
      outputBufferInfo.isScalar = true;
    }
    return;
  }

  InferShapeFromInput(rec.inputBufferIndexes, bufferInfoList_, outputBufferInfo, rec, orderedForOps_,
                      input_.tileUpperBoundPerLoop, input_.alignBufferSizeTo256Bits);
}

void MemoryPeakEstimator::initReduceOps_(OpRecord &rec, BufferInfo &outputBufferInfo, Operation *op) {
  outputBufferInfo.isVirtual = true;

  SmallVector<int64_t, 4> loopAxes;
  auto arr = op->getAttrOfType<ArrayAttr>(kReductionAxesStr);
  for (Attribute a : arr) {
    if (auto ia = dyn_cast<IntegerAttr>(a)) {
      loopAxes.push_back(ia.getInt());
    }
  }

  int64_t virtualOpIndex = perOpList_.size();
  perOpList_.push_back(OpRecord{});
  OpRecord &virtualOpRec = perOpList_[virtualOpIndex];
  rec.VirtualopIndexes.push_back(virtualOpIndex);
  virtualOpRec.Index = virtualOpIndex;
  virtualOpRec.sourceOp = op;
  virtualOpRec.isVirtualOp = true;
  virtualOpRec.VirtualIndex = 0;
  virtualOpRec.forOPRegion = innermostEnclosingScfFor(op);
  virtualOpRec.opType = OpTypeCode(op);
  virtualOpRec.inputBufferIndexes = rec.inputBufferIndexes;
  virtualOpRec.outputBufferIndex = rec.outputBufferIndex;

  int64_t virtualReduceOpIndex = perOpList_.size();
  perOpList_.push_back(OpRecord{});
  OpRecord &virtualReduceOpRec = perOpList_[virtualReduceOpIndex];
  virtualReduceOpRec.Index = virtualReduceOpIndex;
  virtualReduceOpRec.sourceOp = op;
  virtualReduceOpRec.isVirtualOp = true;
  virtualReduceOpRec.VirtualIndex = 1;
  virtualReduceOpRec.forOPRegion = innermostEnclosingScfFor(op);
  virtualReduceOpRec.opType = kOpTypeArithReduce;
  virtualReduceOpRec.inputBufferIndexes = {rec.outputBufferIndex};

  int64_t reductionBufferInfoIndex = bufferInfoList_.size();
  bufferInfoList_.push_back(BufferInfo{});
  BufferInfo &reductionBufferInfo = bufferInfoList_[reductionBufferInfoIndex];
  reductionBufferInfo.Index = reductionBufferInfoIndex;
  reductionBufferInfo.elementType = op->getResults()[0].getType();
  reductionBufferInfo.isValid = outputBufferInfo.isValid;

  SmallVector<int64_t, 4> reductionShapeIndices;
  for (size_t i = 0; i < outputBufferInfo.dimLoopIndices.size(); i++) {
    if (llvm::find(loopAxes, static_cast<int64_t>(i)) == loopAxes.end()) {
      reductionShapeIndices.push_back(outputBufferInfo.dimLoopIndices[i]);
    }
  }
  reductionBufferInfo.dimLoopIndices.assign(reductionShapeIndices.begin(), reductionShapeIndices.end());
  reductionBufferInfo.originalValue = outputBufferInfo.originalValue;
  reductionBufferInfo.isScalar = reductionShapeIndices.empty();
  reductionBufferInfo.totalBufferSize = totalBitsfromBuffer(reductionBufferInfo);
  reductionBufferInfo.OriginOpRecordIndex = virtualReduceOpIndex;
  virtualReduceOpRec.outputBufferIndex = reductionBufferInfoIndex;
  outputBufferInfo.OriginOpRecordIndex = virtualOpIndex;
  rec.outputBufferIndex = reductionBufferInfoIndex;
}

void MemoryPeakEstimator::initPerOpForGenericOp_(Operation *op) {
  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    if (Operation *enclosingFor = innermostEnclosingScfFor(op)) {
      const EquivalentLoadKey key = makeEquivalentLoadKey(loadOp, enclosingFor);
      if (int64_t existingIdx = findEquivalentLoadBufferIndex(equivalentLoadBufferMap_, key); existingIdx >= 0) {
        bufferInfoIndexMap_[loadOp.getResult()] = existingIdx;
        return;
      }
    }
  }

  OpRecord &rec = createBaseOpRecord_(op);

  for (Value input : op->getOperands()) {
    if (int64_t bufIdx = getOrCreateBlockInputBufferIndex_(input, rec, op); bufIdx >= 0) {
      rec.inputBufferIndexes.push_back(bufIdx);
    }
  }

  if (op->getNumResults() == 0) {
    return;
  }

  Value output = op->getResults()[0];
  int64_t valIndex = bufferInfoList_.size();
  bufferInfoList_.push_back(BufferInfo{});
  BufferInfo &outputBufferInfo = bufferInfoList_[valIndex];

  bufferInfoIndexMap_[output] = valIndex;
  outputBufferInfo.OriginOpRecordIndex = rec.Index;
  outputBufferInfo.Index = valIndex;
  outputBufferInfo.elementType = output.getType();

  InferOutputBufferShape_(outputBufferInfo, op, rec);
  outputBufferInfo.elementType = output.getType();
  outputBufferInfo.originalValue = output;
  outputBufferInfo.totalBufferSize = totalBitsfromBuffer(outputBufferInfo);

  if (!isInsideScfForBody(op)) {
    outputBufferInfo.isValid = false;
  }

  rec.outputBufferIndex = valIndex;

  if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    if (Operation *enclosingFor = innermostEnclosingScfFor(op)) {
      equivalentLoadBufferMap_.push_back({makeEquivalentLoadKey(loadOp, enclosingFor), valIndex});
    }
  }

  if (!op->hasAttr(kReductionAxesStr)) {
    return;
  }

  initReduceOps_(rec, outputBufferInfo, op);
}

static void walkFuncBodyUntilReturn(func::FuncOp func, llvm::function_ref<void(Operation *)> callback) {
  for (Operation &rootOp : func.getBody().front()) {
    if (isa<func::ReturnOp>(&rootOp)) {
      break;
    }
    rootOp.walk<WalkOrder::PostOrder>(callback);
  }
}

void MemoryPeakEstimator::initPerOp_() {
  equivalentLoadBufferMap_.clear();
  buildForOpWalkOrder_();
  walkFuncBodyUntilReturn(input_.func, [&](Operation *op) {
    if (isa<scf::YieldOp>(op) || isa<func::ReturnOp>(op)) {
      return;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      initPerOpForForOp_(forOp);
      return;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      initPerOpForIfOp_(ifOp);
      return;
    }
    initPerOpForGenericOp_(op);
  });
}

void MemoryPeakEstimator::analyzeConditionalControlFlow_() {
  conditionalAliasEdges_.clear();
  scfIfBranchInfos_.clear();

  walkFuncBodyUntilReturn(input_.func, [&](Operation *op) {
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      if (!isScalarArithSelectOp(selectOp, bufferInfoIndexMap_, bufferInfoList_)) {
        return;
      }
      auto resultIt = bufferInfoIndexMap_.find(selectOp.getResult());
      if (resultIt == bufferInfoIndexMap_.end()) {
        return;
      }
      const int64_t resultIdx = resultIt->second;

      auto markSelectOperand = [&](Value operand) {
        auto operandIt = bufferInfoIndexMap_.find(operand);
        if (operandIt == bufferInfoIndexMap_.end()) {
          return;
        }
        const int64_t operandIdx = operandIt->second;
        bufferInfoList_[operandIdx].ignoreInplace = true;
        recordConditionalAlias(conditionalAliasEdges_, resultIdx, operandIdx);
      };
      markSelectOperand(selectOp.getTrueValue());
      markSelectOperand(selectOp.getFalseValue());
      return;
    }

    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp) {
      return;
    }

    auto opIt = perOpIndexMap_.find(op);
    if (opIt == perOpIndexMap_.end()) {
      return;
    }

    if (ifOp.getNumResults() > 0) {
      auto thenYield = cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      scf::YieldOp elseYield = nullptr;
      if (Block *elseBlock = ifOp.elseBlock()) {
        elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());
      }
      for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
        auto resultIt = bufferInfoIndexMap_.find(ifOp.getResult(i));
        if (resultIt == bufferInfoIndexMap_.end()) {
          continue;
        }
        const int64_t resultIdx = resultIt->second;
        auto recordIfYieldAlias = [&](Value yieldVal) {
          auto yieldIt = bufferInfoIndexMap_.find(yieldVal);
          if (yieldIt != bufferInfoIndexMap_.end()) {
            recordConditionalAlias(conditionalAliasEdges_, resultIdx, yieldIt->second);
          }
        };
        recordIfYieldAlias(thenYield.getOperand(i));
        if (elseYield) {
          recordIfYieldAlias(elseYield.getOperand(i));
        }
      }
    }

    ScfIfBranchInfo branchInfo;
    branchInfo.ifOpRecordIndex = opIt->second;
    collectRegionExclusiveBuffers(ifOp.getThenRegion(), ifOp.elseBlock() ? &ifOp.getElseRegion() : nullptr,
                                    bufferInfoIndexMap_, bufferInfoList_, branchInfo.thenExclusive);
    if (ifOp.elseBlock()) {
      collectRegionExclusiveBuffers(ifOp.getElseRegion(), &ifOp.getThenRegion(), bufferInfoIndexMap_,
                                      bufferInfoList_, branchInfo.elseExclusive);
    }
    if (!branchInfo.thenExclusive.empty() || !branchInfo.elseExclusive.empty()) {
      scfIfBranchInfos_.push_back(std::move(branchInfo));
    }
  });
}

void MemoryPeakEstimator::dumpInplaceChains(llvm::raw_ostream &os) const {
  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t, 4>> chainBuffers;
  for (int64_t i = 0, n = static_cast<int64_t>(bufferInfoList_.size()); i < n; ++i) {
    const BufferInfo &info = bufferInfoList_[static_cast<size_t>(i)];
    if (!info.isValid) {
      continue;
    }
    const int64_t chainRoot = bufferInfoUnionFind_.empty() ? i : bufferInfoUnionFind_.find(i);
    chainBuffers[chainRoot].push_back(i);
  }
  for (const auto &entry : inplaceChainSummary_) {
    if (entry.second.isTempBuffer) {
      chainBuffers.try_emplace(entry.first);
    }
  }

  llvm::SmallVector<int64_t, 8> sortedRoots;
  sortedRoots.reserve(chainBuffers.size());
  std::transform(chainBuffers.begin(), chainBuffers.end(), std::back_inserter(sortedRoots),
                 [](const auto &entry) { return entry.first; });
  llvm::sort(sortedRoots);

  OpPrintingFlags printFlags;
  printFlags.elideLargeElementsAttrs();

  unsigned chainId = 0;
  for (int64_t chainRoot : sortedRoots) {
    const llvm::SmallVectorImpl<int64_t> &buffers = chainBuffers[chainRoot];
    os << "     inplace-chain[" << chainId++ << "] rep=" << chainRoot;
    if (auto summaryIt = inplaceChainSummary_.find(chainRoot); summaryIt != inplaceChainSummary_.end()) {
      dumpInplaceChainSummaryLine(os, summaryIt->second);
    }
    os << "       buffers=" << buffers.size() << '\n';

    if (buffers.empty()) {
      dumpTempBufferChainLine(os, chainRoot, perOpList_, printFlags);
    }

    llvm::SmallVector<int64_t, 4> sortedBufs(buffers.begin(), buffers.end());
    llvm::sort(sortedBufs);
    for (int64_t bufIdx : sortedBufs) {
      dumpChainBufferLine(os, bufIdx, bufferInfoList_[static_cast<size_t>(bufIdx)], perOpList_, printFlags);
    }
  }

  dumpEquivalentLoadDedupSection(os, equivalentLoadBufferMap_, bufferInfoList_, printFlags);
}

void MemoryPeakEstimator::modelVirtualOps() { return; }

void MemoryPeakEstimator::eliminateRedundantOps() { return; }

void MemoryPeakEstimator::computeBufferLifetimes() {
  // record TimelineOpIndexList, only record op whose output bufferinfo
  // update generatedBufferIndexes、killedBufferIndexes、allocTime、freeTime、opTimeIndex
  int64_t opTimeIndex = 0;
  walkFuncBodyUntilReturn(input_.func, [&](Operation *op) {
    auto opIt = perOpIndexMap_.find(op);
    if (opIt == perOpIndexMap_.end()) {
      return;
    }
    OpRecord &opRecord = perOpList_[opIt->second];
    if (opRecord.outputBufferIndex != -1 && !bufferInfoList_[opRecord.outputBufferIndex].isValid) {
      return;
    }

    // handle if's bufferlifeTimeline
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      if (ifOp.getNumResults() == 0 && !isInsideScfForBody(op)) {
        return;
      }
    }

    if (!opRecord.isVirtualOp) {
      TimelineOpIndexList.push_back(opRecord.Index);

      // scf.for / scf.if results alias buffers allocated inside the region; do not
      // treat them as gen at the for/if op (allocTime stays at the real producer).
      const bool skipGenAtControlFlowHeader = isa<scf::ForOp>(op) || isa<scf::IfOp>(op);
      if (!skipGenAtControlFlowHeader && opRecord.outputBufferIndex >= 0) {
        opRecord.generatedBufferIndexes.push_back(opRecord.outputBufferIndex);
        bufferInfoList_[opRecord.outputBufferIndex].allocTime = opTimeIndex;
      }

      for (int64_t inputIndex : opRecord.inputBufferIndexes) {
        bufferInfoList_[inputIndex].freeTime = opTimeIndex;
      }
      opRecord.opTimeIndex = opTimeIndex++;
    } else {
      for (int64_t index : opRecord.VirtualopIndexes) {
        TimelineOpIndexList.push_back(index);
        perOpList_[index].generatedBufferIndexes.push_back(perOpList_[index].outputBufferIndex);

        bufferInfoList_[perOpList_[index].outputBufferIndex].allocTime = opTimeIndex;

        for (int64_t inputIndex : perOpList_[index].inputBufferIndexes) {
          bufferInfoList_[inputIndex].freeTime = opTimeIndex;
        }

        perOpList_[index].opTimeIndex = opTimeIndex++;
      }
    }
  });

  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    for (int64_t inputIndex : opRecord.inputBufferIndexes) {
      if (bufferInfoList_[inputIndex].freeTime == opRecord.opTimeIndex) {
        opRecord.killedBufferIndexes.push_back(inputIndex);
      }
    }
  }
}

void MemoryPeakEstimator::modelReduceExtraBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType == kOpTypeArithReduce) {
      Operation *originOp = opRecord.sourceOp;

      SmallVector<int64_t, 4> loopAxes;
      auto arr = originOp->getAttrOfType<ArrayAttr>(kReductionAxesStr);
      for (Attribute a : arr) {
        if (auto ia = dyn_cast<IntegerAttr>(a)) {
          loopAxes.push_back(ia.getInt());
        }
      }
      const auto &inputDimLoopIndices = bufferInfoList_[opRecord.inputBufferIndexes[0]].dimLoopIndices;
      SmallVector<int64_t, 4> inputShape;
      DimLoopIndicesToShape(inputDimLoopIndices, inputShape);
      int64_t srcAllocElems =
        std::accumulate(inputShape.begin(), inputShape.end(), int64_t(1), std::multiplies<int64_t>());
      ReduceOpKind rkind = reduceKindFromOp(originOp);
      RankedTensorType effTy =
        RankedTensorType::get(inputShape, bufferInfoList_[opRecord.inputBufferIndexes[0]].elementType);
      int64_t extraBufferElems = getExtraBufferSizeForReduceOp(loopAxes, effTy, srcAllocElems, rkind);
      int64_t elementBitWidth = static_cast<int64_t>(
        getElementTypeOrSelf(bufferInfoList_[opRecord.inputBufferIndexes[0]].elementType).getIntOrFloatBitWidth());
      opRecord.extraBufferSize = extraBufferElems * elementBitWidth;
    }
  }
}

void MemoryPeakEstimator::markMultiBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType == kOpTypeMemRefLoad) {
      bufferInfoList_[opRecord.outputBufferIndex].multiNum = MULTI_BUFFER_NUM;
    } else if (opRecord.opType == kOpTypeMemRefStore) {
      bufferInfoList_[opRecord.inputBufferIndexes[0]].multiNum = MULTI_BUFFER_NUM;
    }
  }
}

bool MemoryPeakEstimator::hasInlineBroadcastLoopDims(Operation *op) const {
  if (!op) {
    return false;
  }
  auto opIt = perOpIndexMap_.find(op);
  if (opIt == perOpIndexMap_.end()) {
    return false;
  }
  if (opIt->second < 0 || opIt->second >= static_cast<int64_t>(perOpList_.size())) {
    return false;
  }
  const llvm::SmallVectorImpl<int64_t> &inputBufferIndexes = perOpList_[opIt->second].inputBufferIndexes;
  if (inputBufferIndexes.size() <= 1) {
    return false;
  }
  const int64_t lhsIdx = inputBufferIndexes[0];
  const int64_t rhsIdx = inputBufferIndexes[1];
  if (lhsIdx < 0 || rhsIdx < 0 || lhsIdx >= static_cast<int64_t>(bufferInfoList_.size()) ||
      rhsIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
    return false;
  }
  return bufferInfoList_[lhsIdx].dimLoopIndices.size() != bufferInfoList_[rhsIdx].dimLoopIndices.size();
}

bool MemoryPeakEstimator::hasInlineTransposeLoopDims(Operation *op) const {
  if (!op) {
    return false;
  }
  auto opIt = perOpIndexMap_.find(op);
  if (opIt == perOpIndexMap_.end()) {
    return false;
  }
  if (opIt->second < 0 || opIt->second >= static_cast<int64_t>(perOpList_.size())) {
    return false;
  }
  const llvm::SmallVectorImpl<int64_t> &inputBufferIndexes = perOpList_[opIt->second].inputBufferIndexes;
  if (inputBufferIndexes.size() <= 1) {
    return false;
  }
  const int64_t lhsIdx = inputBufferIndexes[0];
  const int64_t rhsIdx = inputBufferIndexes[1];
  if (lhsIdx < 0 || rhsIdx < 0 || lhsIdx >= static_cast<int64_t>(bufferInfoList_.size()) ||
      rhsIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
    return false;
  }
  if (bufferInfoList_[lhsIdx].dimLoopIndices.size() != bufferInfoList_[rhsIdx].dimLoopIndices.size()) {
    return false;
  }
  return bufferInfoList_[lhsIdx].dimLoopIndices != bufferInfoList_[rhsIdx].dimLoopIndices;
}

void MemoryPeakEstimator::analyzeIntraOpInplace() {
  bufferInfoUnionFind_.clear();
  bufferInfoUnionFind_.growToBufferCount(static_cast<int64_t>(bufferInfoList_.size()));
  for (int64_t i = 0, n = static_cast<int64_t>(bufferInfoList_.size()); i < n; ++i) {
    if (!bufferInfoList_[i].isValid) {
      continue;
    }
    bufferInfoUnionFind_.makeSet(i);
  }

  // Mirrors `MemPlan::GenerateInplaceList`: per timeline op, match gen/kill buffers and union inplace pairs.
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    Operation *op = opRecord.sourceOp;

    const llvm::SmallVectorImpl<int64_t> &genIndexes = opRecord.generatedBufferIndexes;
    if (genIndexes.empty()) {
      continue;
    }

    llvm::SmallVector<int64_t, 4> sortedGenIndexes(genIndexes.begin(), genIndexes.end());
    llvm::sort(sortedGenIndexes, [&](int64_t lhs, int64_t rhs) {
      const int64_t lhsMulti = inplaceChainMaxMultiNum(bufferInfoUnionFind_, bufferInfoList_, lhs);
      const int64_t rhsMulti = inplaceChainMaxMultiNum(bufferInfoUnionFind_, bufferInfoList_, rhs);
      const bool lhsSingleMulti = lhsMulti == 1;
      const bool rhsSingleMulti = rhsMulti == 1;
      if (lhsSingleMulti != rhsSingleMulti) {
        return lhsSingleMulti;
      }
      const int64_t lhsAlloc = inplaceChainEarliestAllocTime(bufferInfoUnionFind_, bufferInfoList_, lhs);
      const int64_t rhsAlloc = inplaceChainEarliestAllocTime(bufferInfoUnionFind_, bufferInfoList_, rhs);
      if (lhsAlloc != rhsAlloc) {
        return lhsAlloc < rhsAlloc;
      }
      return lhs < rhs;
    });

    for (int64_t genIdx : sortedGenIndexes) {
      if (genIdx < 0 || genIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
        continue;
      }
      if (bufferInfoList_[genIdx].ignoreInplace) {
        continue;
      }
      for (int64_t killIdx : opRecord.killedBufferIndexes) {
        if (bufferInfoList_[killIdx].ignoreInplace) {
          continue;
        }
        if (areConditionallyAliased(genIdx, killIdx, conditionalAliasEdges_)) {
          continue;
        }
        if (!canIntraOpInplaceReuse(*this, op, opRecord.opType, bufferInfoList_[genIdx], bufferInfoList_[killIdx],
                                    bufferInfoIndexMap_, bufferInfoList_)) {
          continue;
        }
        uniteBuffersByTimeline(bufferInfoUnionFind_, bufferInfoList_, killIdx, genIdx);
        break;
      }
    }
  }
}

void MemoryPeakEstimator::analyzeInterOpInplace() {
  // count the earliest alloc、latest free、max footprint of each inplace chain on BufferInfoUnionFind
  inplaceChainSummary_.clear();
  if (bufferInfoUnionFind_.empty()) {
    return;
  }

  for (int64_t i = 0, n = static_cast<int64_t>(bufferInfoList_.size()); i < n; ++i) {
    const BufferInfo &info = bufferInfoList_[i];
    if (!info.isValid) {
      continue;
    }
    const int64_t root = bufferInfoUnionFind_.find(i);
    accumulateBufferIntoChainSummary(inplaceChainSummary_[root], info);
  }

  for (int64_t index : TimelineOpIndexList) {
    const OpRecord &opRecord = perOpList_[index];
    if (opRecord.extraBufferSize <= 0 || opRecord.opTimeIndex < 0) {
      continue;
    }
    registerExtraBufferChainSummary(inplaceChainSummary_[-(index + 1)], opRecord, bufferInfoList_);
  }

  llvm::SmallVector<ChainPlanningEntry, 8> planningEntries;
  planningEntries.reserve(inplaceChainSummary_.size());
  for (const auto &entry : inplaceChainSummary_) {
    for (int64_t slot = 0; slot < entry.second.maxMultiNum; ++slot) {
      planningEntries.push_back({entry.first, slot});
    }
  }

  llvm::sort(planningEntries, [&](const ChainPlanningEntry &a, const ChainPlanningEntry &b) {
    return compareChainPlanningEntries(a, b, inplaceChainSummary_);
  });

  for (auto &entry : inplaceChainSummary_) {
    entry.second.slotAllocations.clear();
    entry.second.allocatedStartBits = 0;
    entry.second.allocatedEndBits = 0;
  }

  llvm::SmallVector<PlacedChainInstance, 16> placedInstances;
  placedInstances.reserve(planningEntries.size());
  for (const ChainPlanningEntry &planEntry : planningEntries) {
    placeChainPlanningEntry(planEntry, inplaceChainSummary_, placedInstances);
  }
}

int64_t MemoryPeakEstimator::computePeakBits() {
  int64_t peakBitsOut = 0;
  for (const auto &entry : inplaceChainSummary_) {
    peakBitsOut = std::max(peakBitsOut, chainPeakAllocatedEndBits(entry.second));
  }
  return peakBitsOut;
}

void MemoryPeakEstimator::run(PeakAnalysisResult &out) {
  out = PeakAnalysisResult{};
  if (input_.func.getOperation() == nullptr) {
    return;
  }
  propagateTileUpperBoundsInMap(input_);
  initPerOp_();
  analyzeConditionalControlFlow_();

  modelVirtualOps();
  eliminateRedundantOps();

  computeBufferLifetimes();
  modelReduceExtraBuffer();
  if (input_.enableMultibuffer) {
    markMultiBuffer();
  }
  analyzeIntraOpInplace();
  analyzeInterOpInplace();

  out.PeakBits = computePeakBits();
  out.valid = true;
}

void setTileUpperBoundForLoop(PeakAnalysisInput &input, scf::ForOp forOp, int64_t tileUpperBound) {
  if (!forOp) {
    return;
  }
  if (tileUpperBound <= 0) {
    input.tileUpperBoundPerLoop.erase(forOp);
    return;
  }
  input.tileUpperBoundPerLoop[forOp] = tileUpperBound;
}

void fillTileUpperBoundsByWalkOrder(PeakAnalysisInput &input, llvm::ArrayRef<int64_t> bounds) {
  input.tileUpperBoundPerLoop.clear();
  if (!input.func) {
    return;
  }
  unsigned i = 0;
  input.func.walk([&](scf::ForOp forOp) {
    if (i < bounds.size() && bounds[i] > 0) {
      input.tileUpperBoundPerLoop[forOp] = bounds[i];
    }
    ++i;
  });
}

int64_t estimateAndPrintPeakWithUnitTileSize(func::FuncOp func) {
  PeakAnalysisInput input;
  input.func = func;
  input.enableMultibuffer = true;
  input.alignBufferSizeTo256Bits = false;
  if (func) {
    func.walk([&](scf::ForOp forOp) {
      input.tileUpperBoundPerLoop[forOp] = 1;
      input.isReduceXorAllVectorizeLoop[forOp] = false;
    });
  }

  std::string funcName;
  if (func) {
    funcName = func.getSymName().str();
  }
  PeakAnalysisResult result;
  MemoryPeakEstimator estimator(std::move(input));
  estimator.run(result);
  estimator.dumpInplaceChains(llvm::dbgs());

  if (!funcName.empty()) {
    llvm::dbgs() << "// peak unit-tile @" << funcName << " valid=" << result.valid
                 << " PeakBits=" << result.PeakBits << '\n';
  } else {
    llvm::dbgs() << "// peak unit-tile valid=" << result.valid << " PeakBits=" << result.PeakBits << '\n';
  }
  return result.PeakBits;
}

void estimatePeakForTiling(const PeakAnalysisInput &analysisInput, PeakAnalysisResult &out) {
  func::FuncOp func = analysisInput.func;

  MemoryPeakEstimator estimator(analysisInput);
  estimator.run(out);

  llvm::dbgs() << "// analysis @" << func.getSymName().str() << " valid=" << out.valid
               << " PeakBits=" << out.PeakBits << " tileBounds=[";
  if (func) {
    bool first = true;
    func.walk([&](scf::ForOp forOp) {
      if (!first) {
        llvm::dbgs() << ", ";
      }
      first = false;
      auto it = analysisInput.tileUpperBoundPerLoop.find(forOp);
      llvm::dbgs() << (it != analysisInput.tileUpperBoundPerLoop.end() ? it->second : 0);
    });
  }
  llvm::dbgs() << "]\n";

  // estimator.dumpInplaceChains(llvm::dbgs());
}

}  // namespace mlir
