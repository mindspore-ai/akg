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

#ifndef AKG_ANALYSIS_BUFFERANALYSIS_H
#define AKG_ANALYSIS_BUFFERANALYSIS_H

#include <map>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace akg {

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

/// Check if the operation is a memref reshaping operation.
inline bool isReshapingOp(Operation *op) {
  return isa<memref::CollapseShapeOp, memref::ReshapeOp, memref::ExpandShapeOp>(op);
}

/// Check if the operation is a memref slicing operation.
inline bool isSlicingOp(Operation *op) { return isa<memref::SubViewOp, memref::ReinterpretCastOp>(op); }

/// Check if the operation is a memref aliasing operation.
inline bool isMemRefAliasingOp(Operation *op) { return isReshapingOp(op) || isSlicingOp(op); }

/// Get the source memref for an aliasing operation.
inline Value getAliasSource(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
    .Case([](memref::ExpandShapeOp expand) { return expand.getSrc(); })
    .Case([](memref::CollapseShapeOp collapse) { return collapse.getSrc(); })
    .Case([](memref::SubViewOp subview) { return subview.getSource(); })
    .Case([](memref::ReshapeOp reshape) { return reshape.getSource(); })
    .Case([](memref::ReinterpretCastOp cast) { return cast.getSource(); })
    .Default([](Operation *op) {
      llvm_unreachable("Unsupported aliasing op");
      return Value();
    });
}

//===----------------------------------------------------------------------===//
// Value Comparator
//===----------------------------------------------------------------------===//

/// Value comparator for std::map.
inline bool isLessValue(const Value &a, const Value &b) { return a.getImpl() < b.getImpl(); }

struct ValueComparator {
  bool operator()(const Value &a, const Value &b) const { return isLessValue(a, b); }
};

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

using DataTypeWeightMap = llvm::DenseMap<Value, uint32_t>;

//===----------------------------------------------------------------------===//
// WeightedLiveRange
//===----------------------------------------------------------------------===//

/// Start, End, Weighted live range of operations.
struct WeightedLiveRange {
  uint32_t start;
  uint32_t end;
  int64_t weight;
  Operation *op{nullptr};

  explicit WeightedLiveRange(uint32_t s = 0, uint32_t e = 0, int64_t w = 1, Operation *o = nullptr)
      : start(s), end(e), weight(w), op(o) {}

  bool operator<(const WeightedLiveRange &other) const {
    return std::tie(start, end, weight) < std::tie(other.start, other.end, other.weight);
  }
};

using LiveRanges = llvm::SmallVector<WeightedLiveRange>;
using WeightedEndPair = std::pair<uint32_t, int64_t>;

//===----------------------------------------------------------------------===//
// BufferAnalysisOptions
//===----------------------------------------------------------------------===//

struct BufferAnalysisOptions {
  using MultiBufferMap = std::map<Value, size_t, ValueComparator>;

  /// Mapping from `value` to the multi-buffer count.
  MultiBufferMap multiBufferCount;

  /// If enabled, print detailed buffer analysis information.
  bool printBufferInfo{false};
};

//===----------------------------------------------------------------------===//
// BufferAnalysis
//===----------------------------------------------------------------------===//

/// Buffer status enumeration
enum BufferStatus { UNDEFINED = 0, DEFINED, GENED, KILLED };

/// Buffer information structure
struct BufferInfo {
  Operation *operation{nullptr};
  int64_t constBits{0};
  Type elementType;
};

/// Record buffer life interval information
struct BufferLife {
  explicit BufferLife(Value buffer, int64_t start, int64_t end) : buffer(buffer), allocTime(start), freeTime(end) {}
  explicit BufferLife(Value buffer) : buffer(buffer) {}
  Value buffer;
  int64_t allocTime{-1};
  int64_t freeTime{-1};
};

/// Linear operation info
struct OpInfo {
  explicit OpInfo(Operation *operation, int index) : operation(operation), index(index) {}
  Operation *operation{nullptr};
  int index{0};
};

/// Gen-kill entry
struct GenKillEntry {
  SmallVector<Value> gen;
  SmallVector<Value> kill;
};

/// Main buffer analysis class for affine and scf dialects.
/// Computes the maximum buffer requirement using live range analysis.
/// Reference: MemLivenessAnalysis from PlanMemory.cpp
class BufferAnalysis {
 public:
  explicit BufferAnalysis(mlir::func::FuncOp func, const BufferAnalysisOptions &options)
      : func(func), options(options) {}

  /// Calculate the maximum buffer requirement in bits.
  /// Returns a pair of (maxBuffer, smallestTypeBits).
  std::pair<int64_t, uint32_t> calculateMaxBuffer();

  /// Print live ranges in a format similar to BufferUtils.cpp's printLiveRanges.
  /// Shows each buffer's index, defining operation, size, and life range.
  void printLiveRanges() const;

  /// Print detailed buffer analysis information.
  /// Includes: linear operations with gen/kill info, live ranges, buffer usage over time,
  /// max buffer summary, and active buffers at max buffer time.
  void printBufferAnalysisInfo() const;

 private:
  mlir::func::FuncOp func;
  BufferAnalysisOptions options;

  /// Linear operation sequence
  SmallVector<std::unique_ptr<OpInfo>> linearOperation;
  /// Map from buffer value to its buffer information
  llvm::DenseMap<Value, BufferInfo> bufferInfos;
  /// Map from buffer to its lifetime
  llvm::DenseMap<Value, std::unique_ptr<BufferLife>> buffer2Life;
  /// Map from operation to its gen and kill buffer
  llvm::DenseMap<OpInfo *, GenKillEntry> genKillMap;
  /// Gen-kill status corresponding to buffer
  llvm::DenseMap<Value, BufferStatus> buffer2status;
  /// Map on buffer alias
  llvm::DenseMap<Value, SmallVector<std::pair<Value, bool>>> buffer2AliasVec;
  int seqIndex{0};
  /// Live ranges collected during analysis
  /// Note: buffer2Life stores raw lifetime (allocTime, freeTime) for each buffer,
  /// while liveRanges stores processed live ranges with weights derived from buffer2Life
  LiveRanges liveRanges;
  /// Smallest type bits across all buffers (used for normalization)
  uint32_t smallestTypeBits{0};
  /// Data type weight map (normalized by smallest type bits)
  DataTypeWeightMap dataTypeWeightMap;

  /// Update linear operation info
  OpInfo *UpdateLinearOperation(Operation *op);

  /// Obtain all information about the buffer
  void UpdateOpBufferInfo(Operation *op, const ValueRange &results);

  /// Update the relationship of buffer aliases
  void UpdateBufferAlias(Value buffer, Value aliasBuffer);

  /// Update the relationship of buffer aliases with condition flag
  void UpdateBufferAlias(Value buffer, Value aliasBuffer, bool hasCond);

  /// Update alias buffer and its condition
  void UpdateBuffer2AliasVec(const llvm::SetVector<Value> &buffers, const llvm::SetVector<Value> &aliasBuffers,
                             bool hasCond);

  /// Get alias buffers
  llvm::SetVector<Value> GetAliasBuffers(Value aliasBuffer);

  /// Process gen buffer based on the result value of op
  void UpdateOpGenInfo(OpInfo *opInfo, const ValueRange &results);

  /// Update normal operand gen information on buffer
  void UpdateOperandGenInfo(OpInfo *opInfo, Value operand);

  /// Kill buffer handle
  void OpKillHandle(OpInfo *opInfo, mlir::Liveness live, Block *block);

  /// Process kill buffer based on the result live of op
  void UpdateOpKillInfo(OpInfo *opInfo, Value operand, mlir::Liveness live);

  /// Determine whether two operations are in the same block or op2 is the ancestor of op1
  bool isParentOpDominate(Operation *op1, Operation *op2) const;

  /// Whether afterBlock is after beforeBlock
  bool IsBlockAfter(Block *afterBlock, Block *beforeBlock) const;

  /// Whether the value is dead after a certain block
  bool IsDeadAfterBlock(Value value, Block *block) const;

  /// Check if a single buffer is dead after the given operation
  bool IsBufferDeadAfter(Operation *op, Value buffer, mlir::Liveness live) const;

  //===--------------------------------------------------------------------===//
  // Template Helper Functions for ForOp and IfOp
  //===--------------------------------------------------------------------===//

  /// Template helper for recursive ForOp processing (affine.for and scf.for)
  template <typename ForOpType>
  void RecursiveForOpImpl(ForOpType forOp, mlir::Liveness live);

  /// Template helper for recursive IfOp processing (affine.if and scf.if)
  template <typename IfOpType, typename YieldOpType>
  void RecursiveIfOpImpl(IfOpType ifOp, mlir::Liveness live);

  /// Template helper for getting live buffers in loop (affine.for and scf.for)
  template <typename ForOpType>
  SmallVector<Value> GetLiveBuffersInLoopImpl(ForOpType loopOp, mlir::Liveness live);

  /// Template helper for updating ForOp init args alias (affine.for and scf.for)
  template <typename ForOpType>
  void UpdateForOpInitArgsAliasImpl(ForOpType forOp);

  /// Template helper for updating ForOp buffer alias (affine.for and scf.for)
  template <typename ForOpType>
  void UpdateForOpBufferAliasImpl(ForOpType forOp);

  /// Template helper for updating IfOp buffer alias (affine.if and scf.if)
  template <typename IfOpType, typename YieldOpType>
  void UpdateIfOpBufferAliasImpl(IfOpType ifOp, YieldOpType yieldOp);

  //===--------------------------------------------------------------------===//
  // Common Methods
  //===--------------------------------------------------------------------===//

  /// Recursively traverse IR
  void RecursionIR(Region *region, mlir::Liveness live);

  /// Update store op information
  void UpdateStoreOpInfo(OpInfo *opInfo, const Value storeValue, mlir::Liveness live);

  /// Generate buffer's life time
  void GenerateBufferLife();

  /// Get multi-buffer count for a value
  uint32_t getValMultiBuffer(const Value &value, uint32_t def) const;

  /// Get data type weight for a value
  uint32_t getValDataTypeWeight(const Value &value, uint32_t def, const DataTypeWeightMap &weightMap) const;

  /// Get extra buffer size for reduce operations (similar to BufferUtils.cpp)
  /// Returns extra buffer size as ratio, or 0 if not applicable
  int64_t getExtraBufferSizeByFactor(Operation *op) const;

  /// Gather live ranges from buffer life (similar to BufferUtils.cpp's gatherLiveRanges)
  /// Collects live ranges with gen, kill, and weight information
  void gatherLiveRanges();

  /// Collect inplace reuse buffers (buffers that can reuse killed buffers)
  llvm::DenseSet<Value> gatherInplaceReuseBuffers() const;

  /// Gather and normalize data type weights
  /// Initializes smallestTypeBits and dataTypeWeightMap member variables
  void gatherDataTypeWeights();

  /// Create live ranges from buffer life with weights
  void createLiveRangesFromBufferLife(const llvm::DenseSet<Value> &inplaceReuseBuffers,
                                      const DataTypeWeightMap &dataTypeWeightMap);

  /// Add extra buffer live ranges for reduce operations
  void addExtraBufferLiveRanges(const DataTypeWeightMap &dataTypeWeightMap);

  /// Line sweep algorithm to find max buffer (similar to BufferUtils.cpp's lineSweepRanges)
  /// Returns the maximum buffer usage ratio
  int64_t lineSweepRanges() const;
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

/// Count the maximum number of buffers needed simultaneously for a function.
/// Returns a pair of (maxBuffer, smallestTypeBits).
/// Returns (-1, 0) if the function has more than one block.
std::pair<int64_t, uint32_t> countMaxBuffer(mlir::func::FuncOp func, const BufferAnalysisOptions &options = {});

}  // namespace akg
}  // namespace mlir

#endif  // AKG_ANALYSIS_BUFFERANALYSIS_H
