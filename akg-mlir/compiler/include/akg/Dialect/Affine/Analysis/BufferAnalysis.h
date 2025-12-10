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

#ifndef AKG_DIALECT_AFFINE_ANALYSIS_BUFFERANALYSIS_H
#define AKG_DIALECT_AFFINE_ANALYSIS_BUFFERANALYSIS_H

#include <map>
#include <numeric>
#include <vector>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

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
// UnionFind
//===----------------------------------------------------------------------===//

/// Union-Find data structure for alias analysis.
/// Tracks connected components and maintains the minimum index in each set.
class UnionFind {
 public:
  explicit UnionFind(size_t n = 0) : minIndex(n), parent(n, -1) { std::iota(minIndex.begin(), minIndex.end(), 0); }

  /// Find the representative of the set containing x.
  int find(int x);

  /// Join the sets containing a and b.
  bool join(int a, int b);

  /// Minimum index in each connected component.
  std::vector<int> minIndex;

 private:
  /// Ensure capacity for index n.
  void ensureCapacity(size_t n);

  /// Parent array for union-find. Negative values indicate root with size.
  std::vector<int> parent;
};

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

using IdxToValMap = std::map<uint32_t, Value>;
using IdxToOpMap = std::map<uint32_t, Operation *>;
using DataTypeWeightMap = llvm::DenseMap<Value, uint32_t>;
using ValToIdxMap = llvm::DenseMap<Value, uint32_t>;
using OpToIdxMap = llvm::DenseMap<Operation *, uint32_t>;

//===----------------------------------------------------------------------===//
// WeightedLiveRange
//===----------------------------------------------------------------------===//

/// Start, End, Weighted live range of operations.
struct WeightedLiveRange {
  uint32_t start;
  uint32_t end;
  int64_t weight;

  explicit WeightedLiveRange(uint32_t s = 0, uint32_t e = 0, int64_t w = 1) : start(s), end(e), weight(w) {}

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

  /// If enabled, the buffer used by DMA operations will not be reused by Vector
  /// operations.
  bool enableDmaOpt = false;

  /// If enabled, print live range information for debugging.
  bool printLiveRange = false;
};

//===----------------------------------------------------------------------===//
// ValOperationIndexer
//===----------------------------------------------------------------------===//

/// Class to index values and operations with sequential indices.
class ValOperationIndexer {
 public:
  ValToIdxMap valToIdx;
  OpToIdxMap opToIdx;
  IdxToValMap idxToVal;
  IdxToOpMap idxToOp;

  static constexpr uint32_t kOpNotFoundLiveRange = static_cast<uint32_t>(-1);

  mlir::FailureOr<Value> getVal(uint32_t idx) const;
  mlir::FailureOr<Operation *> getOp(uint32_t idx) const;
  uint32_t getClosestOpIdx(uint32_t idx) const;
  uint32_t getIndex(Value val) const { return valToIdx.at(val); }
  uint32_t getIndex(Operation *op) const { return opToIdx.at(op); }
  uint32_t getCurrentCount() const { return opCount; }
  bool insert(Value val);
  bool insert(Operation *op);

 private:
  uint32_t opCount = 0;
};

//===----------------------------------------------------------------------===//
// BufferAnalysis
//===----------------------------------------------------------------------===//

/// Main buffer analysis class for affine dialect.
/// Computes the maximum buffer requirement using live range analysis.
class BufferAnalysis {
 public:
  BufferAnalysis(Block &block, const BufferAnalysisOptions &options, mlir::func::FuncOp op)
      : block(block), options(options), liveness(op) {}

  /// Count the maximum number of buffers needed simultaneously.
  int64_t countMaxBuffer();

 private:
  Block &block;
  BufferAnalysisOptions options;
  mlir::Liveness liveness;

  DataTypeWeightMap dataTypeWeightMap;
  llvm::DenseMap<Value, uint32_t> valToLiveRangeIdx;
  LiveRanges liveRanges;
  llvm::DenseMap<int64_t, llvm::DenseSet<uint32_t>> opToEndValIdx;
  llvm::DenseMap<uint32_t, int64_t> aliasFurthest;

  /// Alias information using union-find.
  UnionFind aliasSet;
  ValOperationIndexer indexer;

  /// Check if a value is a buffer value (memref type).
  static bool isUsingBuffer(const Value &value) { return isa<MemRefType>(value.getType()); }

  /// Skip operations that are ignorable for buffer analysis.
  static bool skippableOperation(Operation *op) { return isa<memref::AllocOp, memref::AllocaOp>(op); }

  /// Check if an operation is an affine memory read operation.
  static bool isAffineReadOp(Operation *op) { return isa<mlir::affine::AffineLoadOp>(op); }

  /// Check if an operation is an affine memory write operation.
  static bool isAffineWriteOp(Operation *op) { return isa<mlir::affine::AffineStoreOp>(op); }

  /// Check if an operation is a control flow operation (for, if, etc.)
  static bool isControlFlowOp(Operation *op) { return isa<mlir::affine::AffineForOp, mlir::affine::AffineIfOp>(op); }

  /// Get the memref from a load/store operation.
  static Value getMemRefFromOp(Operation *op);

  void adjustInplaceReuseOp(Operation *op);
  void adjustCopyInCopyOut(Operation *op);
  uint32_t insertValue(const Value &value, uint32_t pos, uint32_t weight = 1);
  void recordDataTypeWeight(const Value &value, uint32_t *smallestTypeBits);
  int64_t getExtraBufferSizeByFactor(Operation *op) const;
  llvm::SmallVector<Value> getOperands(Operation &op) const;
  uint32_t getValMultiBuffer(const Value &value, uint32_t def = 1) const;
  uint32_t getValDataTypeWeight(const Value &value, uint32_t def = 1) const;
  void gatherLiveRanges(const mlir::LivenessBlockInfo *blockInfo);
  void processOperationForLiveRange(Operation *op, const mlir::LivenessBlockInfo *blockInfo);
  void processOperationForPostProcess(Operation *op);
  uint32_t updateAliasIntoFurthest(const Value &value, Operation *endOp);
  void gatherDataTypeWeights();
  void processOperationForDataTypeWeight(Operation *op, uint32_t *smallestTypeBits);
  void gatherIndexingAndAlias();
  void processOperationForIndexing(Operation *op);
  void printLiveRanges() const;
  void printAliasInfo();
  int64_t lineSweepRanges();
};

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

/// Count the maximum number of buffers needed simultaneously for a function.
/// Returns -1 if the function has more than one block.
int64_t countMaxBuffer(mlir::func::FuncOp func, const BufferAnalysisOptions &options = {});

}  // namespace akg
}  // namespace mlir

#endif  // AKG_DIALECT_AFFINE_ANALYSIS_BUFFERANALYSIS_H
