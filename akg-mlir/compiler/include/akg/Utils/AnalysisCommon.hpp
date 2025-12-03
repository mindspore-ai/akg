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

#ifndef AKG_UTILS_ANALYSISCOMMON_H
#define AKG_UTILS_ANALYSISCOMMON_H

#include <algorithm>
#include <cmath>
#include <string>
#include "akg/Dialect/Affine/Analysis/DependenceAnalysis.h"
#include "akg/Dialect/MindSpore/IR/MindSporeOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

using akg::MemRefDependenceGraph;

constexpr auto kGlobalCache = 1;
constexpr auto kTargetCpu = "cpu";
constexpr auto kTargetCuda = "cuda";
constexpr auto kTargetGpu = "gpu";
constexpr auto kTargetNpu = "aicore";
constexpr auto kOperatorTypeStr = "OperatorType";
constexpr auto kReduceStr = "Reduce";
constexpr auto kReductionAxesStr = "reduction_axes";
constexpr auto kReductionTypeStr = "reduction_type";
constexpr auto kReductionLoopAttr = "reduction_loop";
constexpr auto kVectorize128Bit = 128;
constexpr auto kVectorize256Bit = 256;
constexpr auto kVectorize512Bit = 512;
constexpr auto kSSEInstructionSet = "sse";
constexpr auto kAVXInstructionSet = "avx";
constexpr auto kAVX2InstructionSet = "avx2";
constexpr auto kAVX512InstructionSet = "avx512";
constexpr auto kNEONInstructionSet = "neon";

enum OperatorTemplate { Default = 0, Elementwise, Broadcast, Reshape, Transpose, Reduce, Matmul, Conv };
const std::unordered_map<int, std::string> operatorTemplateMap = {{0, "Default"}, {1, "Elementwise"}, {2, "Broadcast"},
                                                                  {3, "Reshape"}, {4, "Transpose"},   {5, "Reduce"},
                                                                  {6, "Matmul"},  {7, "Conv"}};

enum ReduceDirection { UNKNOWN = 0, X, Y, ALL };
const std::unordered_map<int, std::string> reduceDirectionMap = {{0, "unknown"}, {1, "x"}, {2, "y"}, {3, "all"}};

const std::unordered_map<std::string, int> cpuInstructionSetMap = {{kNEONInstructionSet, kVectorize128Bit},
                                                                   {kSSEInstructionSet, kVectorize128Bit},
                                                                   {kAVXInstructionSet, kVectorize256Bit},
                                                                   {kAVX2InstructionSet, kVectorize256Bit},
                                                                   {kAVX512InstructionSet, kVectorize512Bit}};

class TosaOperatorType {
 public:
  TosaOperatorType() = default;

  static bool isTosaElementwiseOp(Operation *op) {
    return (isa<tosa::AddOp, tosa::SubOp, tosa::MulOp, tosa::NegateOp, tosa::PowOp, tosa::ReciprocalOp,
                tosa::RsqrtOp, tosa::LogOp, tosa::ExpOp, tosa::AbsOp, tosa::TanhOp, tosa::BitwiseAndOp,
                tosa::BitwiseOrOp, tosa::BitwiseNotOp, tosa::BitwiseXorOp, tosa::LogicalAndOp, tosa::LogicalNotOp,
                tosa::LogicalOrOp, tosa::LogicalXorOp, tosa::CastOp, tosa::LogicalLeftShiftOp,
                tosa::LogicalRightShiftOp, tosa::ArithmeticRightShiftOp, tosa::ClzOp, tosa::SelectOp, tosa::GreaterOp,
                tosa::GreaterEqualOp, tosa::EqualOp, tosa::MaximumOp, tosa::MinimumOp, tosa::CeilOp, tosa::FloorOp,
                tosa::ClampOp, tosa::SigmoidOp, tosa::IdentityOp, tosa::ConstOp, tosa::ReciprocalOp>(op));
  }

  static bool isTosaReduceOp(Operation *op) {
    return (isa<tosa::ReduceSumOp, tosa::ReduceProdOp, tosa::ReduceMaxOp, tosa::ReduceMinOp, tosa::ReduceAllOp,
                tosa::ReduceAnyOp>(op));
  }
};

class MindOperatorType {
 public:
  MindOperatorType() = default;

  static bool isMindElementwiseOp(Operation *op) {
    return (
      isa<mindspore::AddOp, mindspore::AddNOp, mindspore::DivOp, mindspore::SqrtOp, mindspore::CosOp, mindspore::SinOp,
          mindspore::AsinOp, mindspore::AcosOp, mindspore::AcoshOp, mindspore::AtanOp, mindspore::IsnanOp,
          mindspore::IsinfOp, mindspore::InplaceAssignOp, mindspore::AssignOp, mindspore::LessOp,
          mindspore::LessEqualOp>(op));
  }

  static bool isMindReduceOp(Operation *op) {
    return (isa<mindspore::ReduceSumOp, mindspore::ReduceProdOp, mindspore::ReduceMaxOp, mindspore::ReduceMinOp,
                mindspore::ReduceAllOp, mindspore::ReduceAnyOp>(op));
  }
};

using ParallelOpSet = llvm::SmallDenseSet<::mlir::Value>;

class CommonUtils {
 public:
  CommonUtils() = default;
  // Determines whether a value is in the upper and lower bounds of the loop.
  static bool isInForUbAndLb(affine::AffineForOp forOp, int64_t constraintValue) {
    // TODO(akg-dev): getResults().size() > 0
    auto ubMap = forOp.getUpperBoundMap().getResult(0);
    auto lbMap = forOp.getLowerBoundMap().getResult(0);
    if (!llvm::isa<AffineConstantExpr>(ubMap) || !llvm::isa<AffineConstantExpr>(lbMap)) {
      return true;
    }
    auto ubValue = llvm::dyn_cast<AffineConstantExpr>(ubMap).getValue();
    auto lbValue = llvm::dyn_cast<AffineConstantExpr>(lbMap).getValue();
    if (constraintValue < lbValue || constraintValue > ubValue) {
      return false;
    }
    return true;
  }

  // Obtains the constant variable in the if condition.
  static void getConstraintValues(IntegerSet set, SmallVector<int64_t, 4> &constraintValues) {
    for (auto constraint : set.getConstraints()) {
      int64_t constraintValue = -1;
      // TODO(akg-dev): if condition type is other
      if (constraint.getKind() == AffineExprKind::Add) {
        AffineBinaryOpExpr binaryExpr = llvm::cast<AffineBinaryOpExpr>(constraint);
        if (llvm::isa<AffineConstantExpr>(binaryExpr.getRHS())) {
          constraintValue = llvm::dyn_cast<AffineConstantExpr>(binaryExpr.getRHS()).getValue();
        }
      } else if (constraint.getKind() == AffineExprKind::DimId) {
        constraintValue = 0;
      } else {
        constraintValues.clear();
        return;
      }

      constraintValues.push_back(constraintValue);
    }
  }

  // Checks whether the constants in the if condition are within the upper and lower bounds of the corresponding for
  // loop.
  static bool isInRange(Operation *op) {
    if (!isa<affine::AffineIfOp>(op)) {
      return true;
    }
    affine::AffineIfOp ifOp = dyn_cast<affine::AffineIfOp>(op);
    SmallVector<int64_t, 4> constraintValues;
    getConstraintValues(ifOp.getIntegerSet(), constraintValues);
    if (constraintValues.empty()) {
      return true;
    }

    auto ifOps = ifOp.getOperands();
    assert(constraintValues.size() == ifOps.size());
    int64_t i = 0;
    for (auto value : ifOps) {
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (isa<IndexType>(blockArg.getType())) {
          Block *block = blockArg.getOwner();
          Operation *parentOp = block->getParentOp();
          if (auto forOp = dyn_cast<affine::AffineForOp>(parentOp)) {
            if (!isInForUbAndLb(forOp, constraintValues[i])) {
              return false;
            }
          }
        }
      }
      ++i;
    }
    return true;
  }

  // Get the common block of two Ops
  static Block *getCommonBlock(Operation *opA, Operation *opB) {
    auto getAllAncestorBlocks = [&](Operation *op, SmallVector<Block *, 4> &ancestorBlocks) {
      Block *currBlock = op->getBlock();
      while (currBlock) {
        ancestorBlocks.push_back(currBlock);
        currBlock = currBlock->getParentOp()->getBlock();
        ancestorBlocks.push_back(currBlock);
      }
    };

    // Find the closest common block including those in AffineIf.
    SmallVector<Block *, 4> srcAncestorBlocks, dstAncestorBlocks;
    getAllAncestorBlocks(opA, srcAncestorBlocks);
    getAllAncestorBlocks(opB, dstAncestorBlocks);

    Block *commonBlock = nullptr;
    for (int i = srcAncestorBlocks.size() - 1, j = dstAncestorBlocks.size() - 1;
         i >= 0 && j >= 0 && srcAncestorBlocks[i] == dstAncestorBlocks[j]; i--, j--) {
      commonBlock = srcAncestorBlocks[i];
    }
    return commonBlock;
  }

  // Obtains the inner op of opA and opB.
  static Operation *getInnerOrOuterOp(Operation *opA = nullptr, Operation *opB = nullptr, bool isInner = true) {
    if (!opA && !opB) {
      return nullptr;
    }
    if (!opA) {
      return opB;
    }
    if (!opB) {
      return opA;
    }

    if (opA->getBlock() == opB->getBlock()) {
      if (opA->isBeforeInBlock(opB)) {
        return isInner ? opB : opA;
      } else {
        return isInner ? opA : opB;
      }
    } else if (opA->getBlock()->findAncestorOpInBlock(*opB)) {
      return isInner ? opB : opA;
    } else if (opB->getBlock()->findAncestorOpInBlock(*opA)) {
      return isInner ? opA : opB;
    } else {
      llvm::errs() << "Operations related to the current operator must be in the same block.";
      return nullptr;
    }
  }

  static OperatorTemplate getOperatorType(Operation *op) {
    OperatorTemplate opType = OperatorTemplate::Default;
    if (!op->hasAttr(kOperatorTypeStr)) {
      llvm::errs() << "Unable to recognize attribute " << kOperatorTypeStr << ".\n";
      return opType;
    }
    // TODO(akg-dev): multi band
    auto opTypeStr = dyn_cast<StringAttr>(op->getAttr(kOperatorTypeStr)).getValue().str();
    for (auto it = operatorTemplateMap.begin(); it != operatorTemplateMap.end(); ++it) {
      if (it->second == opTypeStr) {
        return OperatorTemplate(it->first);
      }
    }
    return opType;
  }

  static ReduceDirection getReduceDirection(Operation *op) {
    if (!op) {
      return ReduceDirection::UNKNOWN;
    }
    std::string directionStr;
    op->walk([&directionStr](Operation *curOp) {
      if (curOp->getAttr(kReductionTypeStr)) {
        directionStr = cast<StringAttr>(curOp->getAttr(kReductionTypeStr)).getValue().str();
      }
    });

    for (auto it = reduceDirectionMap.begin(); it != reduceDirectionMap.end(); ++it) {
      if (it->second == directionStr) {
        return ReduceDirection(it->first);
      }
    }
    return ReduceDirection::UNKNOWN;
  }

  static affine::AffineLoadOp getReduceInitLoadOp(Operation *reduceOp) {
    if (!reduceOp->hasAttr(kReductionAxesStr)) {
      return nullptr;
    }
    auto lhsOp = reduceOp->getOperands()[0].getDefiningOp();
    auto rhsOp = reduceOp->getOperands()[1].getDefiningOp();
    if (!isa<affine::AffineLoadOp>(lhsOp) && !isa<affine::AffineLoadOp>(rhsOp)) {
      return nullptr;
    } else if (!isa<affine::AffineLoadOp>(lhsOp) && isa<affine::AffineLoadOp>(rhsOp)) {
      return dyn_cast<affine::AffineLoadOp>(rhsOp);
    } else if (isa<affine::AffineLoadOp>(lhsOp) && !isa<affine::AffineLoadOp>(rhsOp)) {
      return dyn_cast<affine::AffineLoadOp>(lhsOp);
    }

    auto lhsLoadOp = dyn_cast<affine::AffineLoadOp>(lhsOp);
    auto rhsLoadOp = dyn_cast<affine::AffineLoadOp>(rhsOp);

    auto lhsIndices = lhsLoadOp.getIndices();
    auto rhsIndices = rhsLoadOp.getIndices();
    if (lhsIndices.size() > rhsIndices.size()) {
      return rhsLoadOp;
    } else if (lhsIndices.size() < rhsIndices.size()) {
      return lhsLoadOp;
    } else {
      // TODO(akg-dev): keep_dim is true
      return nullptr;
    }
  }

  // get reduce init store op
  static affine::AffineStoreOp getReduceInitOp(Operation *reduceOp, Block *block) {
    auto initLoadOp = getReduceInitLoadOp(reduceOp);
    if (!initLoadOp) {
      return nullptr;
    }

    affine::AffineStoreOp initStoreOp = nullptr;
    // build dependence graph
    mlir::akg::MemRefDependenceGraph dependenceGraph = MemRefDependenceGraph(block);
    if (!dependenceGraph.init()) {
      return initStoreOp;
    }

    int nodeId = dependenceGraph.getNodeId(initLoadOp);
    if (nodeId == -1) {
      return initStoreOp;
    }
    llvm::DenseSet<unsigned> dependentIds;
    dependenceGraph.getPredecessorNodes(nodeId, dependentIds);
    for (auto id : dependentIds) {
      Operation *dependenceOp = dependenceGraph.getNode(id)->op;
      if (!isInRange(dependenceOp->getParentOp())) {
        continue;
      }
      if (auto store = dyn_cast<affine::AffineStoreOp>(dependenceOp)) {
        initStoreOp = store;
      }
    }
    return initStoreOp;
  }

  static void collectRelatedAxes(Value value, SmallVector<Operation *, 8> &axes) {
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      if (isa<IndexType>(blockArg.getType())) {
        Block *block = blockArg.getOwner();
        Operation *parentOp = block->getParentOp();
        if (isa<affine::AffineForOp>(parentOp) || isa<scf::ParallelOp>(parentOp)) {
          axes.push_back(parentOp);
          for (auto operand : parentOp->getOperands()) {
            collectRelatedAxes(operand, axes);
          }
        }
      }
    }

    if (Operation *parentOp = value.getDefiningOp()) {
      if (auto scfIf = dyn_cast<scf::IfOp>(parentOp)) {
        for (Operation &blockOp : scfIf.thenBlock()->getOperations()) {
          for (auto operand : blockOp.getOperands()) {
            collectRelatedAxes(operand, axes);
          }
        }
        for (Operation &blockOp : scfIf.elseBlock()->getOperations()) {
          for (auto operand : blockOp.getOperands()) {
            collectRelatedAxes(operand, axes);
          }
        }
      } else if (auto affineIf = dyn_cast<affine::AffineIfOp>(parentOp)) {
        affineIf.walk([&](affine::AffineYieldOp yieldOp) { collectRelatedAxes(yieldOp.getOperands()[0], axes); });
      } else {
        for (auto operand : parentOp->getOperands()) {
          collectRelatedAxes(operand, axes);
        }
      }
    }
  }

  static Value getStoreValue(Operation *storeOp) {
    Value storeValue;
    if (dyn_cast<affine::AffineStoreOp>(storeOp)) {
      storeValue = dyn_cast<affine::AffineStoreOp>(storeOp).getValueToStore();
    } else if (dyn_cast<memref::StoreOp>(storeOp)) {
      storeValue = dyn_cast<memref::StoreOp>(storeOp).getValueToStore();
    } else {
      assert(false && "can only get value from AffineStore or memref::StoreOp.");
    }
    return storeValue;
  }

  static ValueRange getStoreLoadIndices(Operation *op) {
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      return load.getIndices();
    }
    if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      return load.getIndices();
    }
    if (auto store = dyn_cast<memref::StoreOp>(op)) {
      return store.getIndices();
    }
    if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      return store.getIndices();
    }
    return ValueRange();
  }

  static Value getStoreMemref(Operation *storeOp) {
    if (!storeOp) {
      return Value();
    }
    if (!isa<affine::AffineStoreOp>(storeOp) && !isa<memref::StoreOp>(storeOp)) {
      llvm::errs() << "can only get memref from AffineStore or memref::StoreOp.\n";
      return Value();
    }
    if (storeOp->getNumOperands() < 2) {
      llvm::errs() << "store op has insufficient operands when querying memref.\n";
      return Value();
    }
    Value memref = storeOp->getOperand(1);
    if (!memref || !isa<BaseMemRefType>(memref.getType())) {
      return Value();
    }
    return memref;
  }

  static std::map<int, SmallVector<memref::AllocOp>> findTempBuffer(Operation *funcOp) {
    std::map<int, SmallVector<memref::AllocOp>> tempBuffers;
    SmallVector<Value> globalMemref;
    funcOp->walk([&](CopyOpInterface copyOp) { globalMemref.push_back(copyOp.getSource()); });
    funcOp->walk([&](memref::AllocOp allocOp) {
      auto cacheLevel = getCacheLevel(allocOp.getMemref());
      if (cacheLevel != kGlobalCache) {
        tempBuffers[cacheLevel].push_back(allocOp);
        return;
      }
      for (auto gm : globalMemref) {
        if (allocOp.getMemref() == gm) {
          return;
        }
      }
      tempBuffers[cacheLevel].push_back(allocOp);
    });
    return tempBuffers;
  }

  static int getCacheLevel(TypedValue<MemRefType> memref) {
    int cacheLevel = kGlobalCache;
    auto bufferType = memref.getType();
    auto memspace = bufferType.getMemorySpace();
    if (dyn_cast_or_null<IntegerAttr>(memspace)) {
      cacheLevel = cast<IntegerAttr>(memspace).getInt();
    }
    return cacheLevel;
  }

  static int getCacheLevel(Operation *funcOp, Value cache) {
    int cacheLevel = kGlobalCache;
    Operation *allocOp = getAllocOpOfValue(funcOp, cache);
    if (allocOp == nullptr) {
      return cacheLevel;
    }
    return getCacheLevel(dyn_cast<memref::AllocOp>(allocOp).getMemref());
  }

  /// e.g.1 get the AllocOp of "memref-involed" value %10
  /// %10 = memref.load %alloc_1[%c0] : memref<1xf32, 5>
  ///   (search for any AllocOp that is used by %10 will get the final result)
  ///   |---> %alloc_1 = memref.alloc() : memref<1xf32, 5>

  /// e.g.2 get the AllocOp of "memref-not-involed" value %9
  /// %9 = arith.addf %7, %8 : f32
  ///   (1. backtrace all operands of value %9, i.e. %7 and %8)
  ///   |---> %7 = memref.load %alloc_2[%c0, %c0] : memref<1x1xf32, 5>
  ///     |---> %alloc_2 = memref.alloc() : memref<1x1xf32, 5>
  ///   |---> %8 = memref.load %alloc_0[%c0] : memref<1xf32, 5>
  ///     |---> %alloc_0 = memref.alloc() : memref<1xf32, 5>
  ///  (2. return the AllocOp with largest rank, i.e. alloc in this case)
  static Operation *getAllocOpOfValue(Operation *funcOp, const Value &value) {
    Operation *ret = nullptr;
    funcOp->walk([&](memref::AllocOp parentOp) {
      for (auto user : parentOp->getUsers()) {
        if (user == value.getDefiningOp()) {
          ret = parentOp.getOperation();
          break;
        }
      }
    });

    if (ret == nullptr && value.getDefiningOp() != nullptr) {
      // Get the alloc with largest rank
      for (auto op : value.getDefiningOp()->getOperands()) {
        auto opAlloc = getAllocOpOfValue(funcOp, op);
        if (opAlloc == nullptr) {
          continue;
        }
        auto newAlloc = dyn_cast<memref::AllocOp>(opAlloc);
        auto newRank = cast<ShapedType>(newAlloc.getType()).getRank();
        if (ret == nullptr) {
          ret = opAlloc;
          continue;
        }
        auto currAlloc = dyn_cast<memref::AllocOp>(ret);
        auto currRank = cast<ShapedType>(currAlloc.getType()).getRank();
        if (newRank > currRank) {
          ret = opAlloc;
        }
      }
    }
    return ret;
  }

  /// @brief Find the buffer that fills the data into current buffer by tracing the data movement.
  /// @param funcOp funcOp
  /// @param fastMem  Current buffer that placed in fast memory, e.g. shared_mem or local_mem
  /// @return The buffer that placed in slow memory which fills the data into current buffer, e.g. global_mem,
  ///         as well the index of it
  static std::pair<Value, SmallVector<Value>> getUpperLevelBuffer(Operation *funcOp, Value fastMem) {
    std::pair<Value, SmallVector<Value>> slowMemAndIndices;
    auto allocA = getAllocOpOfValue(funcOp, fastMem);
    if (allocA == nullptr) {
      return slowMemAndIndices;
    }
    for (auto user : allocA->getUsers()) {
      if (!isa<memref::StoreOp, affine::AffineStoreOp>(user)) {
        continue;
      }
      auto value = getStoreValue(user);
      if (value.getDefiningOp() && isa<memref::LoadOp, affine::AffineLoadOp>(value.getDefiningOp())) {
        slowMemAndIndices = std::make_pair(value, getStoreLoadIndices(user));
      } else {
        auto index = getStoreLoadIndices(user);
        for (auto idx : index) {
          if (idx.getDefiningOp() && !isa<arith::ConstantOp>(idx.getDefiningOp())) {
            slowMemAndIndices = std::make_pair(getStoreMemref(user), index);
            break;
          }
        }
      }
    }
    return slowMemAndIndices;
  }

  static SmallVector<Value> getGlobalIndices(Operation *funcOp, Value operandRoot) {
    SmallVector<Value> indexRoot;
    while (getCacheLevel(funcOp, operandRoot) != kGlobalCache) {
      auto [upperOp, index] = getUpperLevelBuffer(funcOp, operandRoot);
      if (!upperOp || !upperOp.getDefiningOp() || operandRoot == upperOp) {
        break;
      }
      operandRoot = upperOp;
      if (isa<memref::LoadOp, affine::AffineLoadOp>(operandRoot.getDefiningOp())) {
        indexRoot = getStoreLoadIndices(operandRoot.getDefiningOp());
      } else {
        indexRoot = index;
      }
    }
    if (indexRoot.empty()) {
      indexRoot = getStoreLoadIndices(operandRoot.getDefiningOp());
    }
    if (indexRoot.empty()) {
      indexRoot.push_back(operandRoot);
    }
    return indexRoot;
  }

  static SmallVector<Operation *> getReduceOps(Operation *funcOp) {
    SmallVector<Operation *> redOps;
    funcOp->walk([&](Operation *curOp) {
      if (curOp->hasAttr(kReductionAxesStr)) {
        redOps.push_back(curOp);
      }
    });
    return redOps;
  }

  static void collectReductionAxesEachDimImpl(Operation *funcOp, SmallVector<SmallVector<Operation *, 8>> &res,
                                              SmallVector<Operation *> &redOps) {
    for (auto redOp : redOps) {
      if (!redOp) {
        return;
      }
      auto redDest = redOp->getOperands()[1];
      auto redSrc = redOp->getOperands()[0];
      auto indexDest = getGlobalIndices(funcOp, redDest);
      auto indexSrc = getGlobalIndices(funcOp, redSrc);

      SmallVector<SmallVector<Operation *, 8>> axesDestEachDim;
      SmallVector<SmallVector<Operation *, 8>> axesSrcEachDim;
      for (auto idx : indexDest) {
        SmallVector<Operation *, 8> axesDesc;
        collectRelatedAxes(idx, axesDesc);
        (void)std::unique(axesDesc.begin(), axesDesc.end());
        axesDestEachDim.push_back(axesDesc);
      }
      for (auto idx : indexSrc) {
        SmallVector<Operation *, 8> axesSrc;
        collectRelatedAxes(idx, axesSrc);
        (void)std::unique(axesSrc.begin(), axesSrc.end());
        bool isDuplicated = false;
        for (auto arr : axesSrcEachDim) {
          if (arr[0] == axesSrc[0]) {
            isDuplicated = true;
            break;
          }
        }
        if (!isDuplicated) {
          axesSrcEachDim.push_back(axesSrc);
        }
      }
      SmallVector<Operation *, 8> axesDestFlatten;
      for (auto axesDesc : axesDestEachDim) {
        for (auto axisDest : axesDesc) {
          axesDestFlatten.push_back(axisDest);
        }
      }
      for (auto axesSrc : axesSrcEachDim) {
        SmallVector<Operation *, 8> subRes;
        for (Operation *axisSrc : axesSrc) {
          bool is_reduction_axis = true;
          for (Operation *axisDest : axesDestFlatten) {
            if (axisDest == axisSrc) {
              is_reduction_axis = false;
              break;
            }
          }
          if (is_reduction_axis) {
            subRes.push_back(axisSrc);
          }
        }
        if (subRes.size() == 0) {
          continue;
        }
        bool isDuplicated = false;
        for (auto arr : res) {
          if (arr[0] == subRes[0]) {
            isDuplicated = true;
            break;
          }
        }
        if (!isDuplicated) {
          res.push_back(subRes);
        }
      }
    }
  }

  static SmallVector<SmallVector<Operation *, 8>> collectReductionAxesEachDim(Operation *funcOp) {
    SmallVector<SmallVector<Operation *, 8>> res;
    if (!isa<func::FuncOp>(funcOp)) {
      llvm::errs() << "funcOp should be a func::FuncOp, but got: " << *funcOp << "\n";
      return res;
    }
    SmallVector<Operation *> redOps = getReduceOps(funcOp);
    collectReductionAxesEachDimImpl(funcOp, res, redOps);
    return res;
  }

  static void collectBroadcastAxes(Operation *funcOp, llvm::SmallSet<Operation *, 8> &broadcastAxes) {
    SmallVector<Operation *> ops;
    funcOp->walk([&](Operation *op) {
      if (isa<affine::AffineLoadOp, affine::AffineStoreOp>(op)) {
        ops.push_back(op);
      }
    });

    llvm::SmallSet<Operation *, 8> maxAxes;
    llvm::SmallSet<Operation *, 8> minAxes;
    for (auto op : ops) {
      mlir::ValueRange indices;
      if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        indices = load.getIndices();
      } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
        indices = store.getIndices();
      }

      llvm::SmallSet<Operation *, 8> relatedAxes;
      for (size_t i = 0; i < indices.size(); ++i) {
        SmallVector<Operation *, 8> relatedAxesVec;
        CommonUtils::collectRelatedAxes(indices[i], relatedAxesVec);
        for (auto axis : relatedAxesVec) {
          (void)relatedAxes.insert(axis);
        }
      }

      if (maxAxes.empty() || maxAxes.size() <= relatedAxes.size()) {
        maxAxes = relatedAxes;
      }
      if (minAxes.empty() || minAxes.size() >= relatedAxes.size()) {
        minAxes = relatedAxes;
      }
    }

    if (minAxes.empty() || maxAxes.empty() || maxAxes.size() == minAxes.size()) {
      return;
    }

    for (auto axis : maxAxes) {
      if (minAxes.count(axis) == 0) {
        (void)broadcastAxes.insert(axis);
      }
    }
    return;
  }

  static SmallVector<Operation *, 8> collectReductionAxes(Operation *funcOp) {
    SmallVector<Operation *, 8> res;
    auto allRes = collectReductionAxesEachDim(funcOp);
    for (auto eachDim : allRes) {
      for (auto eachAxis : eachDim) {
        res.push_back(eachAxis);
      }
    }
    return res;
  }

  static bool isReduceAxis(SmallVector<Operation *, 8> reduceAxes, const Operation *axis) {
    for (const auto reduceAxis : reduceAxes) {
      if (axis == reduceAxis) {
        return true;
      }
    }
    return false;
  }

  static bool isReduceAxis(Operation *funcOp, const Operation *axis) {
    return isReduceAxis(collectReductionAxes(funcOp), axis);
  }

  static Value cloneOpWithNetOperands(mlir::OpBuilder &builder, const Location loc, mlir::Operation *old_op,
                                      mlir::Value new_lhs, mlir::Value new_rhs) {
    return TypeSwitch<Operation *, Value>(old_op)
      .Case([&](arith::AddFOp) { return builder.create<mlir::arith::AddFOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MulFOp) { return builder.create<mlir::arith::MulFOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::AddIOp) { return builder.create<mlir::arith::AddIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::AndIOp) { return builder.create<mlir::arith::AndIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::OrIOp) { return builder.create<mlir::arith::OrIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MulIOp) { return builder.create<mlir::arith::MulIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MinNumFOp) { return builder.create<mlir::arith::MinNumFOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MaxNumFOp) { return builder.create<mlir::arith::MaxNumFOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MinSIOp) { return builder.create<mlir::arith::MinSIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MaxSIOp) { return builder.create<mlir::arith::MaxSIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MinUIOp) { return builder.create<mlir::arith::MinUIOp>(loc, new_lhs, new_rhs); })
      .Case([&](arith::MaxUIOp) { return builder.create<mlir::arith::MaxUIOp>(loc, new_lhs, new_rhs); });
  }

  static bool isParallelBlockArgument(const ::mlir::Value &arg) {
    if (!isa<BlockArgument>(arg)) {
      return false;
    }
    BlockArgument blockArg = cast<BlockArgument>(arg);
    Block *ownerBlock = blockArg.getOwner();
    Operation *parentOp = ownerBlock->getParentOp();
    if (isa<scf::ParallelOp>(*parentOp)) {
      return true;
    }
    return false;
  }

  static void getRegionOperandsCollection(scf::IfOp IfOp, ParallelOpSet &set0) {
    Region &region = IfOp.getThenRegion();
    region.walk([&](mlir::Operation *op) {
      for (auto operand : op->getOperands()) {
        if (isParallelBlockArgument(operand)) {
          (void)set0.insert(operand);
        }
      }
    });
  }

  static void getOperandsCollectionBackwardImpl(Operation *op, ParallelOpSet &set0) {
    if (!op) {
      return;
    }
    if (isa<scf::ForOp>(*op)) {
      return;
    }
    // now, add the Operands to the set;
    for (auto operand : op->getOperands()) {
      if (isParallelBlockArgument(operand)) {
        (void)set0.insert(operand);
      }
      getOperandsCollectionBackwardImpl(operand.getDefiningOp(), set0);
    }
  }

  static void getOpearandsCollectionBackward(scf::IfOp IfOp, ParallelOpSet &set0) {
    if (isParallelBlockArgument(IfOp.getCondition())) {
      set0.insert(IfOp.getCondition());
    }
    if (Operation *op = IfOp.getCondition().getDefiningOp()) {
      getOperandsCollectionBackwardImpl(op, set0);
    }
  }

  static ParallelOpSet getIntersectionImpl(const ParallelOpSet &s0, const ParallelOpSet &s1) {
    ParallelOpSet result;
    for (auto it = s0.begin(); it != s0.end(); ++it) {
      if (s1.count(*it) > 0) {
        result.insert(*it);
      }
    }
    return result;
  }

  static ParallelOpSet getIntersection(const ParallelOpSet &s0, const ParallelOpSet &s1) {
    if (s0.size() < s1.size()) {
      return getIntersectionImpl(s0, s1);
    } else {
      return getIntersectionImpl(s1, s0);
    }
  }

  // 1.first try to collect all Operands in the if region;
  // 2.second try to backward collect all the define op from IfOp condition, until the find the first forOp which
  // define, or just reach the end
  static bool isIfConditionRelatedToContent(scf::IfOp IfOp) {
    // 1.first try to collect all op operands in the uif region;
    if (IfOp.getThenRegion().empty() || !IfOp.getElseRegion().empty()) {
      IfOp.emitWarning() << "the IfOp has " << (IfOp.getThenRegion().empty() ? "not a " : "a ") << "then region, "
                         << "has " << (IfOp.getElseRegion().empty() ? "not a " : "a ")
                         << "else region.Please notice!\n";
      IfOp.emitWarning() << "The func be called only in IfOp having one Then-region and no Else-Region.\n";
      return true;
    }

    ParallelOpSet set0, set1;
    // 1.first in the Then-region, collect all the operands;
    getRegionOperandsCollection(IfOp, set0);
    // 2.collect all the defining Op and related Operands, until access the boundary or the first ForOp;
    getOpearandsCollectionBackward(IfOp, set1);
    set0 = getIntersection(set0, set1);
    return (set0.size() != (size_t)0) ? true : false;
  }

  static void getAllPreviousRelatedOps(mlir::Operation *op, SmallVector<Operation *, 8> &prevOps) {
    mlir::Block *containingBlock = op->getBlock();
    // TODO(yanzhi): avoid duplicate op in vector
    for (auto operand : op->getOperands()) {
      if (auto prevOp = operand.getDefiningOp()) {
        if (prevOp->getBlock() == containingBlock) {
          prevOps.push_back(prevOp);
          getAllPreviousRelatedOps(prevOp, prevOps);
        }
      }
    }
  }

  static bool isRelatedOpInValueSet(mlir::Operation *op, SmallVector<mlir::Value, 8> &usedValues) {
    for (auto uv : usedValues) {
      for (auto operand : op->getOperands()) {
        if (uv == operand) {
          if (operand.getDefiningOp() && isa<arith::ConstantOp>(operand.getDefiningOp())) {
            continue;
          }
          return true;
        }
      }
      for (auto res : op->getResults()) {
        if (uv == res) {
          return true;
        }
      }
    }
    return false;
  }

  static void getAllPreviousRelatedOpsV2(mlir::Operation *op, SmallVector<Operation *, 8> &prevOps,
                                         SmallVector<mlir::Value, 8> &usedValues) {
    for (auto operand : op->getOperands()) {
      usedValues.push_back(operand);
    }
    usedValues.push_back(op->getResult(0));
    Operation *curOp = op->getPrevNode();
    while (curOp) {
      if (!isa<scf::YieldOp>(curOp)) {
        if (isRelatedOpInValueSet(curOp, usedValues)) {
          for (auto operand : curOp->getOperands()) {
            usedValues.push_back(operand);
          }
          prevOps.push_back(curOp);
        }
      }
      curOp = curOp->getPrevNode();
    }
  }

  static void getAllNextRelatedOps(mlir::Operation *op, SmallVector<Operation *, 8> &nextOps,
                                   SmallVector<mlir::Value, 8> &usedValues) {
    for (auto operand : op->getOperands()) {
      usedValues.push_back(operand);
    }
    for (auto res : op->getResults()) {
      usedValues.push_back(res);
    }
    Operation *curOp = op->getNextNode();
    while (curOp) {
      if (!isa<scf::YieldOp>(curOp)) {
        if (isRelatedOpInValueSet(curOp, usedValues)) {
          for (auto operand : curOp->getOperands()) {
            usedValues.push_back(operand);
          }
          for (auto res : curOp->getResults()) {
            usedValues.push_back(res);
          }
          nextOps.push_back(curOp);
        }
      }
      curOp = curOp->getNextNode();
    }
  }
};
}  // namespace mlir
#endif  // AKG_UTILS_ANALYSISCOMMON_H
