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
#include "akg/Dialect/Affine/Analysis/AffineAnalysis.h"

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
  llvm::SmallVector<int64_t, 2> extraBufferSizes;
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

constexpr int64_t kExtraBufferKeyStride = 64;

static int64_t extraBufferChainKey(int64_t opRecordIndex, unsigned extraSlot) {
  return -((opRecordIndex + 1) * kExtraBufferKeyStride + static_cast<int64_t>(extraSlot) + 1);
}

static std::pair<int64_t, unsigned> decodeExtraBufferChainKey(int64_t chainKey) {
  const int64_t encoded = -chainKey - 1;
  const int64_t opRecordIndex = encoded / kExtraBufferKeyStride - 1;
  const unsigned extraSlot = static_cast<unsigned>(encoded % kExtraBufferKeyStride);
  return {opRecordIndex, extraSlot};
}

static void registerExtraBufferChainSummary(InplaceChainSummary &chain, const OpRecord &opRecord,
                                            int64_t extraBufferSizeBits, unsigned extraSlot,
                                            ArrayRef<BufferInfo> bufferList) {
  chain.isTempBuffer = true;
  chain.earliestAllocTime = opRecord.opTimeIndex;
  chain.latestFreeTime = opRecord.opTimeIndex;
  chain.maxBufferSizeBits = extraBufferSizeBits;
  chain.maxMultiNum = 1;

  if (isa<memref::StoreOp>(opRecord.sourceOp) && !opRecord.extraBufferSizes.empty()) {
    const int64_t maxExtraBits =
      *std::max_element(opRecord.extraBufferSizes.begin(), opRecord.extraBufferSizes.end());
    if (extraBufferSizeBits == maxExtraBits) {
      const auto firstMaxIt =
        std::find(opRecord.extraBufferSizes.begin(), opRecord.extraBufferSizes.end(), maxExtraBits);
      const unsigned firstMaxSlot =
        static_cast<unsigned>(std::distance(opRecord.extraBufferSizes.begin(), firstMaxIt));
      if (extraSlot == firstMaxSlot) {
        chain.maxMultiNum = MULTI_BUFFER_NUM;
        return;
      }
    }
  }

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

struct OperandEquivalenceEntry {
  int64_t bufferIndex = -1;
  Value value;
  // Canonical signature for load index operands; compared instead of `value` when non-empty.
  std::string indexSig;

  bool operator==(const OperandEquivalenceEntry &other) const {
    if (bufferIndex >= 0 || other.bufferIndex >= 0) {
      return bufferIndex >= 0 && other.bufferIndex >= 0 && bufferIndex == other.bufferIndex;
    }
    if (!indexSig.empty() || !other.indexSig.empty()) {
      return !indexSig.empty() && !other.indexSig.empty() && indexSig == other.indexSig;
    }
    return value == other.value;
  }
};

struct EquivalentOpKey {
  Operation *enclosingFor = nullptr;
  int64_t opType = kOpTypeUnknown;
  StringRef opName;
  llvm::SmallVector<OperandEquivalenceEntry, 4> inputOperands;
  DictionaryAttr normalizedAttrs;

  bool operator==(const EquivalentOpKey &other) const {
    if (enclosingFor != other.enclosingFor || opType != other.opType || normalizedAttrs != other.normalizedAttrs ||
        inputOperands != other.inputOperands) {
      return false;
    }
    if (opType != kOpTypeUnknown) {
      return true;
    }
    return opName == other.opName;
  }
};

struct BrcCstKey {
  Value cst;
  llvm::SmallVector<int64_t, 4> dimLoopIndices;

  bool operator==(const BrcCstKey &other) const {
    return cst == other.cst && dimLoopIndices == other.dimLoopIndices;
  }
};

static std::optional<int64_t> findBrcCstBufferIndex(ArrayRef<std::pair<BrcCstKey, int64_t>> entries,
                                                    const BrcCstKey &key) {
  for (const auto &[entry, bufIdx] : entries) {
    if (entry == key) {
      return bufIdx;
    }
  }
  return std::nullopt;
}

static bool isConstantSsaValue(Value val) {
  Operation *defOp = val.getDefiningOp();
  if (!defOp) {
    return false;
  }
  return isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp, arith::ConstantFloatOp>(defOp);
}

static Attribute normalizeAttrForEquivalence(
  Attribute attr, Operation *op, llvm::function_ref<std::optional<int64_t>(Value)> resolveValueBuffer) {
  if (auto arr = dyn_cast<ArrayAttr>(attr)) {
    llvm::SmallVector<Attribute, 4> elems;
    elems.reserve(arr.size());
    std::transform(arr.begin(), arr.end(), std::back_inserter(elems),
                   [&](Attribute elem) { return normalizeAttrForEquivalence(elem, op, resolveValueBuffer); });
    return ArrayAttr::get(arr.getContext(), elems);
  }
  if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
    llvm::SmallVector<NamedAttribute, 4> entries;
    entries.reserve(dict.size());
    std::transform(dict.begin(), dict.end(), std::back_inserter(entries),
                   [&](NamedAttribute na) {
                     return NamedAttribute(
                       na.getName(), normalizeAttrForEquivalence(na.getValue(), op, resolveValueBuffer));
                   });
    return DictionaryAttr::get(dict.getContext(), entries);
  }
  for (Value operand : op->getOperands()) {
    if (auto constOp = operand.getDefiningOp<arith::ConstantOp>()) {
      if (constOp.getValue() == attr) {
        if (std::optional<int64_t> bufIdx = resolveValueBuffer(operand)) {
          return IntegerAttr::get(IntegerType::get(op->getContext(), 64), *bufIdx);
        }
      }
    }
  }
  return attr;
}

static DictionaryAttr normalizeOpAttrDictionary(
  DictionaryAttr attrs, Operation *op, llvm::function_ref<std::optional<int64_t>(Value)> resolveValueBuffer) {
  llvm::SmallVector<NamedAttribute, 4> entries;
  entries.reserve(attrs.size());
  std::transform(attrs.begin(), attrs.end(), std::back_inserter(entries),
                 [&](NamedAttribute na) {
                   return NamedAttribute(
                     na.getName(), normalizeAttrForEquivalence(na.getValue(), op, resolveValueBuffer));
                 });
  return DictionaryAttr::get(attrs.getContext(), entries);
}

static std::optional<int64_t> findEquivalentOpBufferIndex(ArrayRef<std::pair<EquivalentOpKey, int64_t>> entries,
                                                          const EquivalentOpKey &key) {
  for (const auto &[entry, bufIdx] : entries) {
    if (entry == key) {
      return bufIdx;
    }
  }
  return std::nullopt;
}

static void dumpEquivalentOpKey(llvm::raw_ostream &os, const EquivalentOpKey &key, OpPrintingFlags printFlags) {
  (void)printFlags;
  os << " opType=" << key.opType;
  if (key.opType == kOpTypeUnknown) {
    os << " opName=" << key.opName;
  }
  os << " inputOperands=[";
  for (size_t i = 0; i < key.inputOperands.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    const OperandEquivalenceEntry &entry = key.inputOperands[i];
    if (entry.bufferIndex >= 0) {
      os << "buf:" << entry.bufferIndex;
    } else if (!entry.indexSig.empty()) {
      os << "sig:" << entry.indexSig;
    } else {
      os << "val:";
      entry.value.print(os);
    }
  }
  os << "] attrs=" << key.normalizedAttrs;
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
  const auto [opIdx, extraSlot] = decodeExtraBufferChainKey(chainRoot);
  if (opIdx < 0 || opIdx >= static_cast<int64_t>(perOpList.size())) {
    return;
  }
  const OpRecord &opRecord = perOpList[static_cast<size_t>(opIdx)];
  os << "                     extra-buffer[" << extraSlot << "] totalBits=";
  if (extraSlot < opRecord.extraBufferSizes.size()) {
    os << opRecord.extraBufferSizes[extraSlot];
  } else {
    os << '?';
  }
  os << " op:";
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

static void dumpInt64IndexList(llvm::raw_ostream &os, ArrayRef<int64_t> values) {
  os << '[';
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    os << values[i];
  }
  os << ']';
}

static void dumpBufferIndexSummary(llvm::raw_ostream &os, int64_t bufIdx, ArrayRef<BufferInfo> bufferList) {
  os << bufIdx;
  if (bufIdx < 0 || bufIdx >= static_cast<int64_t>(bufferList.size())) {
    return;
  }
  const BufferInfo &info = bufferList[static_cast<size_t>(bufIdx)];
  os << "{bits=" << info.totalBufferSize << " scalar=" << (info.isScalar ? 1 : 0)
     << " valid=" << (info.isValid ? 1 : 0) << " virtual=" << (info.isVirtual ? 1 : 0)
     << " multiNum=" << info.multiNum << " alloc=" << info.allocTime << " free=" << info.freeTime
     << " originOpRec=" << info.OriginOpRecordIndex << " dimLoops=";
  dumpInt64IndexList(os, info.dimLoopIndices);
  os << '}';
}

static void dumpBufferIndexListSummary(llvm::raw_ostream &os, ArrayRef<int64_t> bufIdxs,
                                       ArrayRef<BufferInfo> bufferList) {
  os << '[';
  for (size_t i = 0; i < bufIdxs.size(); ++i) {
    if (i > 0) {
      os << ',';
    }
    dumpBufferIndexSummary(os, bufIdxs[i], bufferList);
  }
  os << ']';
}

static void dumpOpRecordLine(llvm::raw_ostream &os, const OpRecord &rec, ArrayRef<BufferInfo> bufferList,
                             OpPrintingFlags printFlags) {
  os << "       opRecord[" << rec.Index << "] opType=" << rec.opType << " opTime=" << rec.opTimeIndex
     << " isVirtual=" << (rec.isVirtualOp ? 1 : 0) << " virtualIndex=" << rec.VirtualIndex
     << " virtualOpIndexes=";
  dumpInt64IndexList(os, rec.VirtualopIndexes);
  os << " extraBufferSizes=";
  dumpInt64IndexList(os, rec.extraBufferSizes);
  os << " op:";
  if (rec.sourceOp) {
    rec.sourceOp->print(os, printFlags);
  } else {
    os << "<unknown>";
  }
  os << '\n';
  os << "         inputBuffers=";
  dumpBufferIndexListSummary(os, rec.inputBufferIndexes, bufferList);
  os << " outputBuffer=";
  dumpBufferIndexSummary(os, rec.outputBufferIndex, bufferList);
  os << " generated=";
  dumpBufferIndexListSummary(os, rec.generatedBufferIndexes, bufferList);
  os << " killed=";
  dumpBufferIndexListSummary(os, rec.killedBufferIndexes, bufferList);
  os << '\n';
}

static void dumpOpRecordsSection(llvm::raw_ostream &os, ArrayRef<OpRecord> perOpList, ArrayRef<BufferInfo> bufferList,
                                 OpPrintingFlags printFlags) {
  llvm::SmallVector<const OpRecord *, 0> records;
  records.reserve(perOpList.size());
  for (const OpRecord &rec : perOpList) {
    if (rec.opType == kOpTypeScfFor) {
      continue;
    }
    if (rec.sourceOp && isa<scf::ForOp>(rec.sourceOp)) {
      continue;
    }
    records.push_back(&rec);
  }

  os << "     op-records(excl-for) count=" << records.size() << '\n';
  for (const OpRecord *rec : records) {
    dumpOpRecordLine(os, *rec, bufferList, printFlags);
  }
}

static void dumpEquivalentOpDedupSection(llvm::raw_ostream &os, ArrayRef<std::pair<EquivalentOpKey, int64_t>> entries,
                                       ArrayRef<BufferInfo> bufferList, OpPrintingFlags printFlags) {
  if (entries.empty()) {
    return;
  }
  os << "     equivalent-op-dedup entries=" << entries.size() << '\n';
  for (const auto &[key, bufIdx] : entries) {
    os << "       bufferIdx=" << bufIdx;
    if (bufIdx >= 0 && bufIdx < static_cast<int64_t>(bufferList.size())) {
      const BufferInfo &info = bufferList[static_cast<size_t>(bufIdx)];
      os << " lifetime=[" << info.allocTime << ',' << info.freeTime << ']';
    }
    os << ' ';
    dumpEquivalentOpKey(os, key, printFlags);
    os << '\n';
  }
}
}  // namespace

class MemoryPeakEstimator {
 public:
  explicit MemoryPeakEstimator(PeakAnalysisInput input);

  void run(PeakAnalysisResult &out);

  void dumpInplaceChains(llvm::raw_ostream &os) const;
  void dumpOpRecords(llvm::raw_ostream &os) const;

  const llvm::DenseMap<Operation *, int64_t> &getPerOpIndexMap() const { return perOpIndexMap_; }
  const llvm::SmallVector<OpRecord, 0> &getPerOpList() const { return perOpList_; }

  bool hasInlineBroadcastLoopDims(Operation *op) const;
  bool hasInlineTransposeLoopDims(Operation *op) const;
  bool isMaterializedCstBufferIndex(int64_t bufIdx) const;
  int64_t findMaterializedCstBufferIndex(Value cst, ArrayRef<int64_t> dimLoopIndices) const;

 private:
  PeakAnalysisInput input_;

  llvm::SmallVector<scf::ForOp, 8> orderedForOps_;
  llvm::DenseMap<scf::ForOp, int64_t> forOpToIndex_;

  llvm::SmallVector<OpRecord, 0> perOpList_;
  llvm::SmallVector<BufferInfo> bufferInfoList_;

  llvm::DenseMap<Operation *, int64_t> perOpIndexMap_;
  llvm::DenseMap<Value, int64_t> bufferInfoIndexMap_;
  llvm::SmallVector<std::pair<EquivalentOpKey, int64_t>, 16> equivalentOpBufferMap_;
  llvm::SmallVector<std::pair<BrcCstKey, int64_t>, 16> brcCstBufferMap_;

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
  void inferShapeFromInput_(ArrayRef<int64_t> inputBufferIndexes, BufferInfo &outputBufferInfo, OpRecord &rec,
                            Operation *op);
  void buildForOpWalkOrder_();
  int64_t totalBitsfromBuffer(const BufferInfo &info) const;
  void alignLastAxisTileBoundInMap_(ArrayRef<int64_t> dimLoopIndices, Type elementType);
  void assignBlockInputBufferAllocTime_(scf::ForOp forOp, int64_t opTimeIndex);
  int64_t totalBitsFromMemRefValue_(Value memref) const;
  void DimLoopIndicesToShape(ArrayRef<int64_t> dimLoopIndices, SmallVectorImpl<int64_t> &outBounds) const;
  void inferEnclosingDimLoopIndices_(Operation *op, SmallVectorImpl<int64_t> &outIndices) const;
  void inferDimLoopIndicesFromForOps_(ArrayRef<scf::ForOp> loops,
                                            SmallVectorImpl<int64_t> &outIndices) const;
  void inferIterArgDimLoopIndices_(scf::ForOp owningFor, SmallVectorImpl<int64_t> &outIndices) const;
  void inferBroadcastOutputDimLoopIndices_(Operation *op, SmallVectorImpl<int64_t> &outIndices) const;
  static bool isScfRegionIterArgBuffer_(const BufferInfo &info);
  int64_t resolveOperandBufferIndex_(Value input, Operation *op);
  int64_t getOrCreateBlockInputBufferIndex_(Value input, const OpRecord &rec, Operation *op);
  int64_t getOrCreateBrcCstBuffer_(Value cst, ArrayRef<int64_t> dimLoopIndices, OpRecord &rec, Operation *op);
  EquivalentOpKey buildEquivalentOpKeyFromRecord_(Operation *op, Operation *enclosingFor,
                                                  ArrayRef<int64_t> orderedInputBufferIndexes);
  void registerEquivalentOpBuffer_(Operation *op, Operation *enclosingFor, int64_t outputBufferIndex,
                                   ArrayRef<int64_t> orderedInputBufferIndexes);
  void analyzeConditionalControlFlow_();
  void assignForEntryTimeline_(scf::ForOp forOp, int64_t &opTimeIndex);
  void assignForExitTimeline_(Operation *op, int64_t &opTimeIndex);
  void assignInputBufferFreeTime_(const OpRecord &opRecord, int64_t opTimeIndex);
  void assignIterArgFreeTimeAtForExit_(scf::ForOp forOp, int64_t opTimeIndex, OpRecord &forOpRecord);
  void assignOpTimeline_(Operation *op, int64_t &opTimeIndex);
  void assignGenBuffersAllocTime_(OpRecord &opRecord, int64_t opTimeIndex);
  void assignVirtualOpTimeline_(const OpRecord &opRecord, int64_t &opTimeIndex);

  void modelVirtualOps();
  void eliminateRedundantOps();

  void computeBufferLifetimes();
  void modelReduceExtraBuffer();
  void modelSelectExtraBuffer();
  void modelNegExtraBuffer();
  void modelLoadExtraBuffer();
  void modelStoreExtraBuffer();
  void markMultiBuffer();
  void invalidateFullyInlinedBrcCstBuffers_();

  int64_t scaledBufferBitsForStoreBroadcastChain_(const BufferInfo &info) const;
  void replaceExtraBufferSizesMatching_(int64_t oldSize, int64_t newSize);
  void propagateStoreBroadcastInputChainResize_(int64_t startBufIdx);

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
}  // namespace
namespace {
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

static bool shouldBlockBrcCstInplaceReuse(Operation *op, unsigned operandIndex) {
  if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp, arith::MulSIExtendedOp, arith::MulUIExtendedOp,
          arith::DivFOp>(op)) {
    return false;
  }
  if (operandIndex == 1) {
    return true;
  }
  return isa<arith::AddFOp, arith::AddIOp, arith::MulFOp, arith::MulIOp>(op);
}

static bool isKillBrcCst(const MemoryPeakEstimator &est, Operation *op, const OpRecord &opRecord,
                                        int64_t killBufIdx, ArrayRef<BufferInfo> bufferList) {
  if (!est.isMaterializedCstBufferIndex(killBufIdx)) {
    return false;
  }
  if (killBufIdx < 0 || killBufIdx >= static_cast<int64_t>(bufferList.size())) {
    return false;
  }
  const BufferInfo &killInfo = bufferList[static_cast<size_t>(killBufIdx)];
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    if (!isConstantSsaValue(operand) || operand != killInfo.originalValue) {
      continue;
    }
    const int64_t matIdx = est.findMaterializedCstBufferIndex(operand, killInfo.dimLoopIndices);
    if (matIdx == killBufIdx && shouldBlockBrcCstInplaceReuse(op, i)) {
      return true;
    }
  }
  return false;
}

static bool isOpSupportingIntraOpInplace(const MemoryPeakEstimator &est, Operation *op, int64_t opType,
                                         const OpRecord &opRecord, const BufferInfo &gen, const BufferInfo &kill,
                                         const llvm::DenseMap<Value, int64_t> &bufMap,
                                         llvm::ArrayRef<BufferInfo> bufferList) {
  if (!op || op->hasAttr(kReductionAxesStr)) {
    return false;
  }

  if (isIsaInplaceElemwiseOp(op) && !est.hasInlineBroadcastLoopDims(op) && !est.hasInlineTransposeLoopDims(op)) {
    return true;
  }

  if (opRecord.extraBufferSizes.size() > 1 ||
      (opRecord.extraBufferSizes.size() == 1 && !est.hasInlineBroadcastLoopDims(op) && opType != kOpTypeArithNegF)) {
    return false;
  }

  if (!isArithOp(opType)) {
    return false;
  }

  if (est.hasInlineTransposeLoopDims(op)) {
    return false;
  }

  if (est.hasInlineBroadcastLoopDims(op)) {
    if (isKillBrcCst(est, op, opRecord, kill.Index, bufferList)) {
      return false;
    }
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
  return (killBits % genBits == 0);
  // For dynamic axes, check if it will be collapsed later. If so, check if shape.size is greater than 1.
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

static bool canIntraOpInplaceReuse(const MemoryPeakEstimator &est, Operation *op, int64_t opType,
                                   const OpRecord &opRecord, const BufferInfo &gen, const BufferInfo &kill,
                                   const llvm::DenseMap<Value, int64_t> &bufMap,
                                   llvm::ArrayRef<BufferInfo> bufferList) {
  if (gen.ignoreInplace || kill.ignoreInplace) {
    return false;
  }
  if (!isBufferEligibleForIntraOpInplace(gen) || !isBufferEligibleForIntraOpInplace(kill)) {
    return false;
  }
  return isOpSupportingIntraOpInplace(est, op, opType, opRecord, gen, kill, bufMap, bufferList);
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

// 32B = 256bit alignment on last axis element count: f16/bf16 -> 16, f32/i32 -> 8, i8 -> 32, etc.
static int64_t lastAxis32ByteAlignElems(Type elementType) {
  const unsigned bitWidth = getElementTypeOrSelf(elementType).getIntOrFloatBitWidth();
  if (bitWidth == 0 || (256 % bitWidth) != 0) {
    return 1;
  }
  return 256 / static_cast<int64_t>(bitWidth);
}

static int64_t TotalBitsFromDimLoopIndicesInBroadcast(ArrayRef<int64_t> dimLoopIndices, Type elementType,
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
  if (!bounds.empty()) {
    bounds.back() = ceilFactor(bounds.back(), lastAxis32ByteAlignElems(elementType));
  }
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
    // Materialized broadcast constants keep output dimLoopIndices but originate from scalar SSA.
    if (inputInfo.isScalar || isConstantSsaValue(inputInfo.originalValue)) {
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

static void setInlineBroadcastExtraBufferSize(ArrayRef<int64_t> inputBufferIndexes, ArrayRef<BufferInfo> bufferList,
                                              ArrayRef<int64_t> outputDimLoopIndices, Type outputElementType,
                                              bool needExtraBuffer, OpRecord &rec, Operation *op,
                                              const MemoryPeakEstimator &est, ArrayRef<scf::ForOp> orderedForOps,
                                              const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop,
                                              bool alignBufferSizeTo256Bits) {
  if (!needExtraBuffer || !op) {
    return;
  }

  const int64_t outputTotalBits =
    TotalBitsFromDimLoopIndicesInBroadcast(outputDimLoopIndices, outputElementType, orderedForOps,
                                           tileUpperBoundPerLoop, alignBufferSizeTo256Bits);

  for (int64_t index : inputBufferIndexes) {
    if (index < 0 || index >= static_cast<int64_t>(bufferList.size())) {
      continue;
    }
    const BufferInfo &inputInfo = bufferList[static_cast<size_t>(index)];
    if (isKillBrcCst(est, op, rec, index, bufferList)) {
      rec.extraBufferSizes.push_back(outputTotalBits);
      continue;
    }
    if (ArrayRef(inputInfo.dimLoopIndices) != outputDimLoopIndices) {
      rec.extraBufferSizes.push_back(TotalBitsFromDimLoopIndicesInBroadcast(outputDimLoopIndices, inputInfo.elementType,
                                                                       orderedForOps, tileUpperBoundPerLoop,
                                                                       alignBufferSizeTo256Bits));
    }
  }
}

static std::string scfForLoopSig(scf::ForOp forOp);

static std::string loadIndexValueSig(Value v) {
  if (!v) {
    return "null";
  }
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Block *block = blockArg.getOwner();
    if (block && block->isEntryBlock()) {
      if (auto forOp = dyn_cast<scf::ForOp>(block->getParentOp())) {
        // Induction var (arg 0) or iter_args (arg >= 1) of a scf.for body block.
        return "larg" + std::to_string(blockArg.getArgNumber()) + "(" + scfForLoopSig(forOp) + ")";
      }
    }
    return "ba" + std::to_string(blockArg.getArgNumber());
  }
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    std::string s;
    llvm::raw_string_ostream os(s);
    constOp.getValue().print(os);
    os << ":" << v.getType();
    return "cst<" + os.str() + ">";
  }
  if (Operation *defOp = v.getDefiningOp()) {
    std::string s = "op<" + defOp->getName().getStringRef().str() + ">#" +
                    std::to_string(v.cast<OpResult>().getResultNumber());
    return s;
  }
  return "val<" + std::to_string(reinterpret_cast<uintptr_t>(v.getAsOpaquePointer())) + ">";
}

static std::string scfForLoopSig(scf::ForOp forOp) {
  SmallVector<scf::ForOp, 8> chain;
  scf::ForOp cur = forOp;
  while (cur) {
    chain.push_back(cur);
    Operation *parent = cur->getParentRegion() ? cur->getParentRegion()->getParentOp() : nullptr;
    cur = parent ? dyn_cast<scf::ForOp>(parent) : scf::ForOp();
  }
  std::string s;
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    s += "(" + loadIndexValueSig((*it).getLowerBound()) + "," + loadIndexValueSig((*it).getUpperBound()) + "," +
         loadIndexValueSig((*it).getStep()) + ")";
  }
  return s;
}

// For loops with the same scfForLoopSig iterate the same axis; copy a non-zero tile bound
// within each equivalence class when some entries in `tileUpperBoundPerLoop` are zero/missing.
static void propagateTileUpperBoundsInMap(PeakAnalysisInput &input) {
  func::FuncOp func = input.func;
  if (!func) {
    return;
  }

  llvm::StringMap<llvm::SmallVector<scf::ForOp, 8>> boundGroups;
  func.walk([&](scf::ForOp forOp) {
    boundGroups[scfForLoopSig(forOp)].push_back(forOp);
  });

  for (auto &entry : boundGroups) {
    const llvm::SmallVector<scf::ForOp, 8> &forOps = entry.getValue();
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
  llvm::StringMap<int64_t> sigToIndex;
  input_.func.walk([&](scf::ForOp forOp) {
    const std::string sig = scfForLoopSig(forOp);
    int64_t idx;
    if (auto it = sigToIndex.find(sig); it != sigToIndex.end()) {
      idx = it->second;
    } else {
      idx = static_cast<int64_t>(orderedForOps_.size()) + 1;
      sigToIndex[sig] = idx;
      orderedForOps_.push_back(forOp);
    }
    forOpToIndex_[forOp] = idx;
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

int64_t MemoryPeakEstimator::totalBitsfromBuffer(const BufferInfo &info) const {
  if (info.isScalar || info.dimLoopIndices.empty()) {
    return static_cast<int64_t>(getElementTypeOrSelf(info.elementType).getIntOrFloatBitWidth());
  }
  SmallVector<int64_t, 4> bounds;
  DimLoopIndicesToShape(info.dimLoopIndices, bounds);
  return totalBitsFromShape(bounds, info.elementType, input_.alignBufferSizeTo256Bits);
}

void MemoryPeakEstimator::alignLastAxisTileBoundInMap_(ArrayRef<int64_t> dimLoopIndices, Type elementType) {
  if (dimLoopIndices.empty() || dimLoopIndices.size() < 2) {
    return;
  }
  const int64_t lastLoop = dimLoopIndices.back();
  if (lastLoop <= 0 || lastLoop > static_cast<int64_t>(orderedForOps_.size())) {
    return;
  }
  scf::ForOp forOp = orderedForOps_[static_cast<size_t>(lastLoop - 1)];
  const int64_t currentBound = DimLoopIndexToBound(lastLoop, orderedForOps_, input_.tileUpperBoundPerLoop);
  const int64_t alignElems = lastAxis32ByteAlignElems(elementType);
  const int64_t aligned = ceilFactor(currentBound, alignElems);
  if (aligned != currentBound) {
    input_.tileUpperBoundPerLoop[forOp] = aligned;
  }
}

void MemoryPeakEstimator::assignBlockInputBufferAllocTime_(scf::ForOp forOp, int64_t opTimeIndex) {
  Block &body = forOp.getRegion().front();
  for (BlockArgument arg : body.getArguments()) {
    if (arg.getArgNumber() == 0) {
      continue;  // induction var has no buffer
    }
    auto it = bufferInfoIndexMap_.find(arg);
    if (it == bufferInfoIndexMap_.end()) {
      continue;
    }
    BufferInfo &info = bufferInfoList_[static_cast<size_t>(it->second)];
    if (info.allocTime < 0) {
      info.allocTime = opTimeIndex;
    }
  }
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

void MemoryPeakEstimator::inferDimLoopIndicesFromForOps_(ArrayRef<scf::ForOp> loops,
                                                                SmallVectorImpl<int64_t> &outIndices) const {
  llvm::DenseSet<int64_t> seen;
  outIndices.clear();
  for (scf::ForOp forOp : loops) {
    auto reduceIt = input_.isReduceXorAllVectorizeLoop.find(forOp);
    if (reduceIt != input_.isReduceXorAllVectorizeLoop.end() && reduceIt->second) {
      continue;
    }
    auto tileIt = input_.tileUpperBoundPerLoop.find(forOp);
    if (tileIt == input_.tileUpperBoundPerLoop.end() || tileIt->second <= 0) {
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

void MemoryPeakEstimator::inferIterArgDimLoopIndices_(scf::ForOp owningFor,
                                                       SmallVectorImpl<int64_t> &outIndices) const {
  SmallVector<scf::ForOp, 8> loops;
  getEnclosingScfForOps(owningFor, loops);
  loops.push_back(owningFor);
  inferDimLoopIndicesFromForOps_(loops, outIndices);
}

void MemoryPeakEstimator::inferBroadcastOutputDimLoopIndices_(Operation *op,
                                                               SmallVectorImpl<int64_t> &outIndices) const {
  SmallVector<scf::ForOp, 8> loops;
  getEnclosingScfForOps(op, loops);
  inferDimLoopIndicesFromForOps_(loops, outIndices);
}

bool MemoryPeakEstimator::isScfRegionIterArgBuffer_(const BufferInfo &info) {
  auto blockArg = dyn_cast<BlockArgument>(info.originalValue);
  if (!blockArg || blockArg.getArgNumber() == 0) {
    return false;
  }
  Block *block = blockArg.getOwner();
  if (!block || !block->isEntryBlock()) {
    return false;
  }
  return isa<scf::ForOp, scf::IfOp>(block->getParentOp());
}

int64_t MemoryPeakEstimator::resolveOperandBufferIndex_(Value input, Operation *op) {
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
  inputBufferInfo.OriginOpRecordIndex = -1;
  inputBufferInfo.elementType = input.getType();
  inputBufferInfo.originalValue = input;
  if (auto blockArg = dyn_cast<BlockArgument>(input)) {
    if (auto owningFor = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp())) {
      if (blockArg.getArgNumber() > 0) {
        inferIterArgDimLoopIndices_(owningFor, inputBufferInfo.dimLoopIndices);
      } else {
        inferEnclosingDimLoopIndices_(op, inputBufferInfo.dimLoopIndices);
      }
    } else {
      inferEnclosingDimLoopIndices_(op, inputBufferInfo.dimLoopIndices);
    }
  } else {
    inferEnclosingDimLoopIndices_(op, inputBufferInfo.dimLoopIndices);
  }
  inputBufferInfo.isScalar = inputBufferInfo.dimLoopIndices.empty();
  inputBufferInfo.isValid = isInsideScfForBody(op);
  alignLastAxisTileBoundInMap_(inputBufferInfo.dimLoopIndices, inputBufferInfo.elementType);
  inputBufferInfo.totalBufferSize = totalBitsfromBuffer(inputBufferInfo);
  bufferInfoIndexMap_[input] = valIndex;
  return valIndex;
}

int64_t MemoryPeakEstimator::getOrCreateBlockInputBufferIndex_(Value input, const OpRecord &rec, Operation *op) {
  const int64_t valIndex = resolveOperandBufferIndex_(input, op);
  if (valIndex < 0) {
    return -1;
  }
  if (bufferInfoList_[static_cast<size_t>(valIndex)].OriginOpRecordIndex < 0) {
    bufferInfoList_[static_cast<size_t>(valIndex)].OriginOpRecordIndex = rec.Index;
  }
  return valIndex;
}

int64_t MemoryPeakEstimator::getOrCreateBrcCstBuffer_(Value cst, ArrayRef<int64_t> dimLoopIndices, OpRecord &rec,
                                                      Operation *op) {
  BrcCstKey key;
  key.cst = cst;
  key.dimLoopIndices.assign(dimLoopIndices.begin(), dimLoopIndices.end());
  if (std::optional<int64_t> existingIdx = findBrcCstBufferIndex(brcCstBufferMap_, key)) {
    return *existingIdx;
  }

  const int64_t valIndex = static_cast<int64_t>(bufferInfoList_.size());
  bufferInfoList_.push_back(BufferInfo{});
  BufferInfo &materializedInfo = bufferInfoList_[static_cast<size_t>(valIndex)];
  materializedInfo.Index = valIndex;
  materializedInfo.OriginOpRecordIndex = rec.Index;
  materializedInfo.originalValue = cst;
  materializedInfo.elementType = getElementTypeOrSelf(cst.getType());
  materializedInfo.dimLoopIndices.assign(dimLoopIndices.begin(), dimLoopIndices.end());
  materializedInfo.isScalar = dimLoopIndices.empty();
  materializedInfo.isValid = true;
  alignLastAxisTileBoundInMap_(materializedInfo.dimLoopIndices, materializedInfo.elementType);
  materializedInfo.totalBufferSize = totalBitsfromBuffer(materializedInfo);
  brcCstBufferMap_.push_back({key, valIndex});
  rec.generatedBufferIndexes.push_back(valIndex);
  return valIndex;
}

bool MemoryPeakEstimator::isMaterializedCstBufferIndex(int64_t bufIdx) const {
  for (const auto &[key, idx] : brcCstBufferMap_) {
    (void)key;
    if (idx == bufIdx) {
      return true;
    }
  }
  return false;
}

int64_t MemoryPeakEstimator::findMaterializedCstBufferIndex(Value cst, ArrayRef<int64_t> dimLoopIndices) const {
  BrcCstKey key;
  key.cst = cst;
  key.dimLoopIndices.assign(dimLoopIndices.begin(), dimLoopIndices.end());
  if (std::optional<int64_t> idx = findBrcCstBufferIndex(brcCstBufferMap_, key)) {
    return *idx;
  }
  return -1;
}

EquivalentOpKey MemoryPeakEstimator::buildEquivalentOpKeyFromRecord_(Operation *op, Operation *enclosingFor,
                                                                     ArrayRef<int64_t> orderedInputBufferIndexes) {
  EquivalentOpKey key;
  // key.enclosingFor = enclosingFor;
  key.opType = OpTypeCode(op);
  key.opName = op->getName().getStringRef();

  auto appendOperandEntry = [&](unsigned operandIndex) {
    OperandEquivalenceEntry entry;
    const int64_t bufIdx =
      operandIndex < orderedInputBufferIndexes.size() ? orderedInputBufferIndexes[operandIndex] : -1;
    if (bufIdx >= 0) {
      entry.bufferIndex = bufIdx;
    } else {
      entry.value = op->getOperand(operandIndex);
    }
    key.inputOperands.push_back(entry);
  };

  if (isa<memref::LoadOp>(op)) {
    auto loadOp = cast<memref::LoadOp>(op);
    key.inputOperands.reserve(1 + loadOp.getIndices().size());
    appendOperandEntry(0);
    for (Value idx : loadOp.getIndices()) {
      OperandEquivalenceEntry entry;
      entry.indexSig = loadIndexValueSig(idx);
      key.inputOperands.push_back(entry);
    }
  } else {
    key.inputOperands.reserve(op->getNumOperands());
    for (unsigned i = 0; i < op->getNumOperands(); ++i) {
      appendOperandEntry(i);
    }
  }

  auto resolveValueBuffer = [&](Value val) -> std::optional<int64_t> {
    const int64_t idx = resolveOperandBufferIndex_(val, op);
    if (idx < 0) {
      return std::nullopt;
    }
    return idx;
  };
  key.normalizedAttrs = normalizeOpAttrDictionary(op->getAttrDictionary(), op, resolveValueBuffer);
  return key;
}

void MemoryPeakEstimator::registerEquivalentOpBuffer_(Operation *op, Operation *enclosingFor,
                                                      int64_t outputBufferIndex,
                                                      ArrayRef<int64_t> orderedInputBufferIndexes) {
  if (!enclosingFor || op->hasAttr(kReductionAxesStr) || op->getNumResults() > 1) {
    return;
  }
  const EquivalentOpKey key = buildEquivalentOpKeyFromRecord_(op, enclosingFor, orderedInputBufferIndexes);
  if (findEquivalentOpBufferIndex(equivalentOpBufferMap_, key)) {
    return;
  }
  equivalentOpBufferMap_.push_back({key, outputBufferIndex});
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

  inferShapeFromInput_(rec.inputBufferIndexes, outputBufferInfo, rec, op);
}

void MemoryPeakEstimator::inferShapeFromInput_(ArrayRef<int64_t> inputBufferIndexes, BufferInfo &outputBufferInfo,
                                               OpRecord &rec, Operation *op) {
  outputBufferInfo.isScalar = true;
  outputBufferInfo.dimLoopIndices.clear();

  bool hasScalarInput = false;
  bool hasVectorInput = false;
  SmallVector<ArrayRef<int64_t>, 4> nonScalarDimLoopIndices;
  collectInputShapeInfo(inputBufferIndexes, bufferInfoList_, hasScalarInput, hasVectorInput, nonScalarDimLoopIndices);
  if (nonScalarDimLoopIndices.empty()) {
    return;
  }

  outputBufferInfo.isScalar = false;
  const bool allSame =
    nonScalarDimLoopIndices.size() <= 1 || allDimLoopIndicesSame(nonScalarDimLoopIndices);
  if (allSame) {
    outputBufferInfo.dimLoopIndices.assign(nonScalarDimLoopIndices.front().begin(),
                                          nonScalarDimLoopIndices.front().end());
  } else {
    inferBroadcastOutputDimLoopIndices_(op, outputBufferInfo.dimLoopIndices);
  }

  const bool needExtraBuffer = (hasScalarInput && hasVectorInput) || (nonScalarDimLoopIndices.size() >= 2 && !allSame);
  setInlineBroadcastExtraBufferSize(inputBufferIndexes, bufferInfoList_, outputBufferInfo.dimLoopIndices,
                                    outputBufferInfo.elementType, needExtraBuffer, rec, op, *this, orderedForOps_,
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
  rec.VirtualopIndexes.push_back(virtualReduceOpIndex);
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
  Operation *enclosingFor = innermostEnclosingScfFor(op);

  OpRecord &rec = createBaseOpRecord_(op);

  llvm::SmallVector<int64_t, 4> orderedInputBuffers;
  orderedInputBuffers.reserve(op->getNumOperands());
  rec.inputBufferIndexes.clear();
  for (Value input : op->getOperands()) {
    if (isConstantSsaValue(input)) {
      orderedInputBuffers.push_back(-1);
      continue;
    }
    const int64_t bufIdx = getOrCreateBlockInputBufferIndex_(input, rec, op);
    orderedInputBuffers.push_back(bufIdx);
    if (bufIdx >= 0) {
      rec.inputBufferIndexes.push_back(bufIdx);
    }
  }

  if (op->getNumResults() == 0) {
    registerEquivalentOpBuffer_(op, enclosingFor, -1, orderedInputBuffers);
    return;
  }

  Value output = op->getResults()[0];
  BufferInfo shapeProbe;
  shapeProbe.elementType = output.getType();
  InferOutputBufferShape_(shapeProbe, op, rec);

  rec.inputBufferIndexes.clear();
  for (size_t i = 0; i < op->getNumOperands(); ++i) {
    Value input = op->getOperand(i);
    int64_t bufIdx = orderedInputBuffers[i];
    if (isConstantSsaValue(input)) {
      bufIdx = getOrCreateBrcCstBuffer_(input, shapeProbe.dimLoopIndices, rec, op);
      orderedInputBuffers[i] = bufIdx;
    }
    if (bufIdx >= 0) {
      rec.inputBufferIndexes.push_back(bufIdx);
    }
  }

  int64_t valIndex = -1;
  if (enclosingFor && !op->hasAttr(kReductionAxesStr)) {
    const EquivalentOpKey key = buildEquivalentOpKeyFromRecord_(op, enclosingFor, orderedInputBuffers);
    if (std::optional<int64_t> existingIdx = findEquivalentOpBufferIndex(equivalentOpBufferMap_, key);
        existingIdx && *existingIdx >= 0) {
      valIndex = *existingIdx;
    }
  }

  if (valIndex < 0) {
    valIndex = bufferInfoList_.size();
    bufferInfoList_.push_back(BufferInfo{});
    BufferInfo &outputBufferInfo = bufferInfoList_[valIndex];

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

    registerEquivalentOpBuffer_(op, enclosingFor, valIndex, orderedInputBuffers);
  }

  bufferInfoIndexMap_[output] = valIndex;
  rec.outputBufferIndex = valIndex;

  if (!op->hasAttr(kReductionAxesStr)) {
    return;
  }

  initReduceOps_(rec, bufferInfoList_[valIndex], op);
}

static void walkFuncBodyUntilReturn(func::FuncOp func, WalkOrder order,
                                    llvm::function_ref<void(Operation *)> callback) {
  for (Operation &rootOp : func.getBody().front()) {
    if (isa<func::ReturnOp>(&rootOp)) {
      break;
    }
    if (order == WalkOrder::PreOrder) {
      rootOp.walk<WalkOrder::PreOrder>(callback);
    } else {
      rootOp.walk<WalkOrder::PostOrder>(callback);
    }
  }
}

void MemoryPeakEstimator::initPerOp_() {
  equivalentOpBufferMap_.clear();
  brcCstBufferMap_.clear();
  buildForOpWalkOrder_();
  walkFuncBodyUntilReturn(input_.func, WalkOrder::PostOrder, [&](Operation *op) {
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

  walkFuncBodyUntilReturn(input_.func, WalkOrder::PostOrder, [&](Operation *op) {
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

static void dumpForLoopIndexMap(
  llvm::raw_ostream &os, func::FuncOp func,
  const llvm::DenseMap<scf::ForOp, int64_t> &forOpToIndex,
  const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop) {
  os << "     for-loop-index count=" << forOpToIndex.size() << '\n';
  if (!func) {
    return;
  }
  unsigned walkIdx = 0;
  func.walk([&](scf::ForOp forOp) {
    auto it = forOpToIndex.find(forOp);
    const int64_t loopIndex = it != forOpToIndex.end() ? it->second : -1;
    auto tileIt = tileUpperBoundPerLoop.find(forOp);
    const int64_t tile = tileIt != tileUpperBoundPerLoop.end() ? tileIt->second : 0;
    os << "       forWalk[" << walkIdx << "] index=" << loopIndex << " tile=" << tile
       << " sig=" << scfForLoopSig(forOp) << " iv=";
    forOp.getInductionVar().print(os);
    os << '\n';
    ++walkIdx;
  });
}

static void dumpBufferDimLoopIndices(llvm::raw_ostream &os, ArrayRef<BufferInfo> bufferList) {
  os << "     buffer-dimLoops count=" << bufferList.size() << '\n';
  for (size_t i = 0; i < bufferList.size(); ++i) {
    const BufferInfo &info = bufferList[i];
    os << "       buffer[" << i << "] dimLoops=";
    dumpInt64IndexList(os, info.dimLoopIndices);
    os << " bits=" << info.totalBufferSize << " scalar=" << (info.isScalar ? 1 : 0)
       << " valid=" << (info.isValid ? 1 : 0) << '\n';
  }
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

  dumpOpRecordsSection(os, perOpList_, bufferInfoList_, printFlags);

  dumpForLoopIndexMap(os, input_.func, forOpToIndex_, input_.tileUpperBoundPerLoop);
  dumpBufferDimLoopIndices(os, bufferInfoList_);

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

  dumpEquivalentOpDedupSection(os, equivalentOpBufferMap_, bufferInfoList_, printFlags);
}

void MemoryPeakEstimator::dumpOpRecords(llvm::raw_ostream &os) const {
  OpPrintingFlags printFlags;
  printFlags.elideLargeElementsAttrs();
  dumpOpRecordsSection(os, perOpList_, bufferInfoList_, printFlags);
}

void MemoryPeakEstimator::modelVirtualOps() { return; }

void MemoryPeakEstimator::eliminateRedundantOps() { return; }

void MemoryPeakEstimator::assignGenBuffersAllocTime_(OpRecord &opRecord, int64_t opTimeIndex) {
  for (int64_t genIdx : opRecord.generatedBufferIndexes) {
    if (genIdx < 0 || genIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
      continue;
    }
    BufferInfo &genBuffer = bufferInfoList_[static_cast<size_t>(genIdx)];
    if (isScfRegionIterArgBuffer_(genBuffer)) {
      continue;
    }
    if (genBuffer.allocTime < 0) {
      genBuffer.allocTime = opTimeIndex;
    }
  }
  if (opRecord.outputBufferIndex >= 0) {
    BufferInfo &outputBuffer = bufferInfoList_[opRecord.outputBufferIndex];
    if (isScfRegionIterArgBuffer_(outputBuffer)) {
      return;
    }
    if (outputBuffer.allocTime < 0) {
      if (!llvm::is_contained(opRecord.generatedBufferIndexes, opRecord.outputBufferIndex)) {
        opRecord.generatedBufferIndexes.push_back(opRecord.outputBufferIndex);
      }
      outputBuffer.allocTime = opTimeIndex;
    }
  }
}

void MemoryPeakEstimator::assignInputBufferFreeTime_(const OpRecord &opRecord, int64_t opTimeIndex) {
  for (int64_t inputIndex : opRecord.inputBufferIndexes) {
    if (inputIndex < 0 || inputIndex >= static_cast<int64_t>(bufferInfoList_.size())) {
      continue;
    }
    if (isScfRegionIterArgBuffer_(bufferInfoList_[static_cast<size_t>(inputIndex)])) {
      continue;
    }
    if (isMaterializedCstBufferIndex(inputIndex) &&
        isKillBrcCst(*this, opRecord.sourceOp, opRecord, inputIndex, bufferInfoList_)) {
      continue;
    }
    bufferInfoList_[static_cast<size_t>(inputIndex)].freeTime = opTimeIndex;
  }
}

void MemoryPeakEstimator::assignIterArgFreeTimeAtForExit_(scf::ForOp forOp, int64_t opTimeIndex,
                                                          OpRecord &forOpRecord) {
  for (BlockArgument arg : forOp.getRegion().front().getArguments()) {
    if (arg.getArgNumber() == 0) {
      continue;
    }
    auto it = bufferInfoIndexMap_.find(arg);
    if (it == bufferInfoIndexMap_.end()) {
      continue;
    }
    const int64_t bufIdx = it->second;
    bufferInfoList_[static_cast<size_t>(bufIdx)].freeTime = opTimeIndex;
    forOpRecord.killedBufferIndexes.push_back(bufIdx);
  }
}

void MemoryPeakEstimator::assignVirtualOpTimeline_(const OpRecord &opRecord, int64_t &opTimeIndex) {
  for (int64_t index : opRecord.VirtualopIndexes) {
    TimelineOpIndexList.push_back(index);
    BufferInfo &virtualOutputBuffer = bufferInfoList_[perOpList_[index].outputBufferIndex];
    if (virtualOutputBuffer.allocTime < 0) {
      perOpList_[index].generatedBufferIndexes.push_back(perOpList_[index].outputBufferIndex);
      virtualOutputBuffer.allocTime = opTimeIndex;
    }

    assignInputBufferFreeTime_(perOpList_[index], opTimeIndex);

    perOpList_[index].opTimeIndex = opTimeIndex++;
  }
}

void MemoryPeakEstimator::assignForEntryTimeline_(scf::ForOp forOp, int64_t &opTimeIndex) {
  assignBlockInputBufferAllocTime_(forOp, opTimeIndex);
  opTimeIndex++;
}

void MemoryPeakEstimator::assignForExitTimeline_(Operation *op, int64_t &opTimeIndex) {
  auto forOp = dyn_cast<scf::ForOp>(op);
  if (!forOp) {
    return;
  }

  auto opIt = perOpIndexMap_.find(op);
  if (opIt == perOpIndexMap_.end()) {
    return;
  }
  OpRecord &opRecord = perOpList_[opIt->second];

  if (opRecord.outputBufferIndex != -1 && !bufferInfoList_[opRecord.outputBufferIndex].isValid) {
    return;
  }

  TimelineOpIndexList.push_back(opRecord.Index);

  assignIterArgFreeTimeAtForExit_(forOp, opTimeIndex, opRecord);
  opRecord.opTimeIndex = opTimeIndex++;
}

void MemoryPeakEstimator::assignOpTimeline_(Operation *op, int64_t &opTimeIndex) {
  if (isa<scf::ForOp>(op)) {
    return;
  }

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

  if (opRecord.VirtualopIndexes.size() > 0) {
    assignVirtualOpTimeline_(opRecord, opTimeIndex);
    return;
  }

  TimelineOpIndexList.push_back(opRecord.Index);

  const bool skipGenAtControlFlowHeader = isa<scf::IfOp>(op);
  if (!skipGenAtControlFlowHeader) {
    assignGenBuffersAllocTime_(opRecord, opTimeIndex);
  }

  assignInputBufferFreeTime_(opRecord, opTimeIndex);
  opRecord.opTimeIndex = opTimeIndex++;
}

void MemoryPeakEstimator::computeBufferLifetimes() {
  // record TimelineOpIndexList, only record op whose output bufferinfo
  // update generatedBufferIndexes、killedBufferIndexes、allocTime、freeTime、opTimeIndex
  int64_t opTimeIndex = 0;
  walkFuncBodyUntilReturn(input_.func, WalkOrder::PreOrder, [&](Operation *op) {
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      assignForEntryTimeline_(forOp, opTimeIndex);
    }
  });
  walkFuncBodyUntilReturn(input_.func, WalkOrder::PostOrder, [&](Operation *op) {
    if (isa<scf::ForOp>(op)) {
      assignForExitTimeline_(op, opTimeIndex);
      return;
    }
    assignOpTimeline_(op, opTimeIndex);
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

// Mirrors HIVM VSelOp::allocExtraBuffersIfPossible() for tiled arith.select.
static int64_t getExtraBufferSizeBitsForArithSelect(arith::SelectOp selectOp) {
  const bool src0ScalarType = selectOp.getTrueValue().getType().isIntOrFloat();
  const bool src1ScalarType = selectOp.getFalseValue().getType().isIntOrFloat();
  const Type condType = getElementTypeOrSelf(selectOp.getCondition().getType());
  const Type srcType = getElementTypeOrSelf(selectOp.getTrueValue().getType());
  const unsigned srcBitWidth = srcType.getIntOrFloatBitWidth();

  if (srcType.isInteger(64) && (condType.isInteger(1) || src0ScalarType || src1ScalarType)) {
    return 0;
  }

  // VSel uses INTR_BYTES_PER_REPEAT / resWidth with resWidth in bits (256-bit vector block).
  const int64_t numElemsPerStride = 256 / static_cast<int64_t>(srcBitWidth);

  if (!srcType.isInteger(64)) {
    int64_t buffSize = numElemsPerStride;
    if (src0ScalarType) {
      buffSize += numElemsPerStride;
    }
    if (src1ScalarType) {
      buffSize += numElemsPerStride;
    }
    return buffSize * static_cast<int64_t>(srcBitWidth);
  }

  // Mirrors VSelOp::getExtraBufferSize() for i64 vector condition + vector src.
  return numElemsPerStride * static_cast<int64_t>(srcBitWidth);
}

void MemoryPeakEstimator::modelSelectExtraBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType != kOpTypeArithSelect) {
      continue;
    }
    auto selectOp = dyn_cast<arith::SelectOp>(opRecord.sourceOp);
    if (!selectOp) {
      continue;
    }
    const int64_t selectExtraBits = getExtraBufferSizeBitsForArithSelect(selectOp);
    if (selectExtraBits > 0) {
      opRecord.extraBufferSizes.push_back(selectExtraBits);
    }
  }
}

int64_t MemoryPeakEstimator::totalBitsFromMemRefValue_(Value memref) const {
  auto memTy = dyn_cast<MemRefType>(memref.getType());
  if (!memTy) {
    return 0;
  }
  int64_t elems = 1;
  for (int64_t dim : memTy.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      if (auto it = bufferInfoIndexMap_.find(memref); it != bufferInfoIndexMap_.end()) {
        const BufferInfo &info = bufferInfoList_[static_cast<size_t>(it->second)];
        if (info.totalBufferSize > 0) {
          return info.totalBufferSize;
        }
      }
      return 0;
    }
    elems *= dim;
  }
  int64_t bits = elems * static_cast<int64_t>(getElementTypeOrSelf(memTy.getElementType()).getIntOrFloatBitWidth());
  if (input_.alignBufferSizeTo256Bits && bits > 0) {
    bits = ((bits + 255) / 256) * 256;
  }
  return bits;
}

static bool memRefLastAxisStrideNotOne(MemRefType memTy) {
  if (!memTy || memTy.getRank() == 0) {
    return false;
  }
  SmallVector<int64_t, 4> strides;
  int64_t offset = 0;
  if (failed(getStridesAndOffset(memTy, strides, offset)) || strides.empty()) {
    return false;
  }
  const int64_t lastStride = strides.back();
  return lastStride != ShapedType::kDynamic && lastStride != 1;
}

void MemoryPeakEstimator::modelLoadExtraBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType != kOpTypeMemRefLoad) {
      continue;
    }
    auto loadOp = dyn_cast<memref::LoadOp>(opRecord.sourceOp);
    if (!loadOp) {
      continue;
    }

    auto loadMemTy = dyn_cast<MemRefType>(loadOp.getMemRef().getType());
    if (!memRefLastAxisStrideNotOne(loadMemTy)) {
      continue;
    }

    const Value rootMemref = affine::getSourceMemRef(loadOp.getMemRef());
    const int64_t rootBufferBits = totalBitsFromMemRefValue_(rootMemref);
    if (rootBufferBits > 0) {
      opRecord.extraBufferSizes.push_back(rootBufferBits);
    }
  }
}

static llvm::SmallDenseSet<int64_t, 4> inputDimLoopIndexSet(ArrayRef<int64_t> inputDimLoopIndices) {
  llvm::SmallDenseSet<int64_t, 4> inputLoops;
  for (int64_t loopIdx : inputDimLoopIndices) {
    if (loopIdx > 0) {
      inputLoops.insert(loopIdx);
    }
  }
  return inputLoops;
}

static bool storeBroadcastsOnLastAxis(ArrayRef<int64_t> storeDimLoopIndices,
                                      ArrayRef<int64_t> inputDimLoopIndices) {
  if (storeDimLoopIndices.empty()) {
    return false;
  }
  const llvm::SmallDenseSet<int64_t, 4> inputLoops = inputDimLoopIndexSet(inputDimLoopIndices);
  const int64_t lastLoop = storeDimLoopIndices.back();
  return lastLoop > 0 && !inputLoops.contains(lastLoop);
}

static int64_t computeStoreBroadcastExtraBits(ArrayRef<int64_t> storeDimLoopIndices,
                                              ArrayRef<int64_t> inputDimLoopIndices, Type elementType,
                                              ArrayRef<scf::ForOp> orderedForOps,
                                              const llvm::DenseMap<scf::ForOp, int64_t> &tileUpperBoundPerLoop,
                                              bool alignBufferSizeTo256Bits) {
  SmallVector<int64_t, 4> bounds;
  bounds.reserve(storeDimLoopIndices.size());
  std::transform(storeDimLoopIndices.begin(), storeDimLoopIndices.end(), std::back_inserter(bounds),
                 [&](int64_t loopIdx) {
                   return DimLoopIndexToBound(loopIdx, orderedForOps, tileUpperBoundPerLoop);
                 });
  if (storeBroadcastsOnLastAxis(storeDimLoopIndices, inputDimLoopIndices) && !bounds.empty()) {
    int64_t &lastBound = bounds.back();
    const int64_t alignElems = lastAxis32ByteAlignElems(elementType);
    lastBound = ceilFactor(lastBound, alignElems);
  }
  return totalBitsFromShape(bounds, elementType, alignBufferSizeTo256Bits);
}

static bool storeTargetHasBroadcastDimLoopIndices(ArrayRef<int64_t> storeDimLoopIndices,
                                                  ArrayRef<int64_t> inputDimLoopIndices) {
  const llvm::SmallDenseSet<int64_t, 4> inputLoops = inputDimLoopIndexSet(inputDimLoopIndices);
  for (int64_t loopIdx : storeDimLoopIndices) {
    if (loopIdx > 0 && !inputLoops.contains(loopIdx)) {
      return true;
    }
  }
  return false;
}

int64_t MemoryPeakEstimator::scaledBufferBitsForStoreBroadcastChain_(const BufferInfo &info) const {
  if (info.isScalar || info.dimLoopIndices.empty()) {
    int64_t bits = static_cast<int64_t>(getElementTypeOrSelf(info.elementType).getIntOrFloatBitWidth());
    bits *= 8;
    if (input_.alignBufferSizeTo256Bits && bits > 0) {
      bits = ceilFactor(bits, kVectorBlockSizeBit);
    }
    return bits;
  }
  SmallVector<int64_t, 4> bounds;
  DimLoopIndicesToShape(info.dimLoopIndices, bounds);
  int64_t bits = totalBitsFromShape(bounds, info.elementType, false);
  bits *= 8;
  if (input_.alignBufferSizeTo256Bits && bits > 0) {
    bits = ceilFactor(bits, kVectorBlockSizeBit);
  }
  return bits;
}

void MemoryPeakEstimator::replaceExtraBufferSizesMatching_(int64_t oldSize, int64_t newSize) {
  if (oldSize <= 0 || oldSize == newSize) {
    return;
  }
  for (OpRecord &rec : perOpList_) {
    std::replace_if(rec.extraBufferSizes.begin(), rec.extraBufferSizes.end(),
                    [oldSize](int64_t extraBits) { return extraBits == oldSize; }, newSize);
  }
}

void MemoryPeakEstimator::propagateStoreBroadcastInputChainResize_(int64_t startBufIdx) {
  llvm::SmallDenseSet<int64_t> visited;
  SmallVector<int64_t, 8> worklist;
  worklist.push_back(startBufIdx);
  while (!worklist.empty()) {
    const int64_t bufIdx = worklist.pop_back_val();
    if (!visited.insert(bufIdx).second) {
      continue;
    }
    if (bufIdx < 0 || bufIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
      continue;
    }
    BufferInfo &info = bufferInfoList_[static_cast<size_t>(bufIdx)];
    const int64_t oldSize = info.totalBufferSize;
    const int64_t newSize = scaledBufferBitsForStoreBroadcastChain_(info);
    if (newSize != oldSize) {
      replaceExtraBufferSizesMatching_(oldSize, newSize);
      info.totalBufferSize = newSize;
    }
    if (info.OriginOpRecordIndex < 0 || info.OriginOpRecordIndex >= static_cast<int64_t>(perOpList_.size())) {
      continue;
    }
    const OpRecord &originRec = perOpList_[static_cast<size_t>(info.OriginOpRecordIndex)];
    std::copy(originRec.inputBufferIndexes.begin(), originRec.inputBufferIndexes.end(),
              std::back_inserter(worklist));
  }
}

void MemoryPeakEstimator::modelStoreExtraBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType != kOpTypeMemRefStore) {
      continue;
    }
    auto storeOp = dyn_cast<memref::StoreOp>(opRecord.sourceOp);
    if (!storeOp) {
      continue;
    }

    const auto memTy = cast<MemRefType>(storeOp.getMemRef().getType());
    if (memTy.getRank() == 0) {
      continue;
    }
    SmallVector<int64_t, 4> storeDimLoopIndices;
    inferDimLoopIndices(storeOp, memTy.getRank(), forOpToIndex_, storeDimLoopIndices);

    auto valueIt = bufferInfoIndexMap_.find(storeOp.getValue());
    if (valueIt == bufferInfoIndexMap_.end()) {
      continue;
    }
    const BufferInfo &inputInfo = bufferInfoList_[static_cast<size_t>(valueIt->second)];
    if (!storeTargetHasBroadcastDimLoopIndices(storeDimLoopIndices, inputInfo.dimLoopIndices)) {
      continue;
    }

    propagateStoreBroadcastInputChainResize_(valueIt->second);

    const int64_t extraBits =
      computeStoreBroadcastExtraBits(storeDimLoopIndices, inputInfo.dimLoopIndices, storeOp.getValue().getType(),
                                     orderedForOps_, input_.tileUpperBoundPerLoop, input_.alignBufferSizeTo256Bits);
    if (extraBits <= 0) {
      continue;
    }
    const bool broadcastOnLastAxis = storeBroadcastsOnLastAxis(storeDimLoopIndices, inputInfo.dimLoopIndices);
    opRecord.extraBufferSizes.push_back(extraBits);
    if (broadcastOnLastAxis) {
      opRecord.extraBufferSizes.push_back(extraBits);
    }
  }
}

void MemoryPeakEstimator::modelNegExtraBuffer() {
  for (int64_t index : TimelineOpIndexList) {
    OpRecord &opRecord = perOpList_[index];
    if (opRecord.opType != kOpTypeArithNegF) {
      continue;
    }
    const int64_t outputIdx = opRecord.outputBufferIndex;
    if (outputIdx < 0 || outputIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
      continue;
    }
    opRecord.extraBufferSizes.push_back(bufferInfoList_[static_cast<size_t>(outputIdx)].totalBufferSize);
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
      if (static_cast<int64_t>(loopAxes.size()) == static_cast<int64_t>(inputShape.size()) &&
          !inputShape.empty()) {
        extraBufferElems *= static_cast<int64_t>(inputShape.size());
      }
      int64_t elementBitWidth = static_cast<int64_t>(
        getElementTypeOrSelf(bufferInfoList_[opRecord.inputBufferIndexes[0]].elementType).getIntOrFloatBitWidth());
      opRecord.extraBufferSizes.push_back(extraBufferElems * elementBitWidth);
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
  const OpRecord &opRecord = perOpList_[opIt->second];
  const llvm::SmallVectorImpl<int64_t> &inputBufferIndexes = opRecord.inputBufferIndexes;
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
    return true;
  }

  // brc cst is materialized to output shape but still lowered as broadcast constant.
  return std::any_of(inputBufferIndexes.begin(), inputBufferIndexes.end(),
                     [&](int64_t bufIdx) { return isKillBrcCst(*this, op, opRecord, bufIdx, bufferInfoList_); });
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

void MemoryPeakEstimator::invalidateFullyInlinedBrcCstBuffers_() {
  for (const auto &[key, brcCstBufIdx] : brcCstBufferMap_) {
    (void)key;
    if (brcCstBufIdx < 0 || brcCstBufIdx >= static_cast<int64_t>(bufferInfoList_.size())) {
      continue;
    }

    bool hasUsage = false;
    bool allUsagesInlineBroadcast = true;
    for (const OpRecord &rec : perOpList_) {
      if (rec.isVirtualOp || !rec.sourceOp) {
        continue;
      }
      bool usedAsInput = std::any_of(rec.inputBufferIndexes.begin(), rec.inputBufferIndexes.end(),
                                     [brcCstBufIdx](int64_t inputIdx) { return inputIdx == brcCstBufIdx; });
      if (!usedAsInput) {
        continue;
      }
      hasUsage = true;
      if (!isKillBrcCst(*this, rec.sourceOp, rec, brcCstBufIdx, bufferInfoList_)) {
        allUsagesInlineBroadcast = false;
        break;
      }
    }

    if (hasUsage && allUsagesInlineBroadcast) {
      bufferInfoList_[static_cast<size_t>(brcCstBufIdx)].isValid = false;
    }
  }
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
      if (isMaterializedCstBufferIndex(genIdx)) {
        continue;
      }
      for (int64_t killIdx : opRecord.killedBufferIndexes) {
        if (bufferInfoList_[killIdx].ignoreInplace) {
          continue;
        }
        if (areConditionallyAliased(genIdx, killIdx, conditionalAliasEdges_)) {
          continue;
        }
        if (!canIntraOpInplaceReuse(*this, op, opRecord.opType, opRecord, bufferInfoList_[genIdx],
                                    bufferInfoList_[killIdx], bufferInfoIndexMap_, bufferInfoList_)) {
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
    if (opRecord.opTimeIndex < 0 || opRecord.extraBufferSizes.empty()) {
      continue;
    }
    for (unsigned extraSlot = 0; extraSlot < opRecord.extraBufferSizes.size(); ++extraSlot) {
      const int64_t extraBits = opRecord.extraBufferSizes[extraSlot];
      if (extraBits <= 0) {
        continue;
      }
      registerExtraBufferChainSummary(inplaceChainSummary_[extraBufferChainKey(index, extraSlot)], opRecord, extraBits,
                                        extraSlot, bufferInfoList_);
    }
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
  modelSelectExtraBuffer();
  modelNegExtraBuffer();
  modelLoadExtraBuffer();
  modelStoreExtraBuffer();
  if (input_.enableMultibuffer) {
    markMultiBuffer();
  }
  invalidateFullyInlinedBrcCstBuffers_();
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
