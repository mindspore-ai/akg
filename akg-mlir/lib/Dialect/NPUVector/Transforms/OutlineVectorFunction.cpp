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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <numeric>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Utils/AnalysisForNpu.hpp"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

#include "akg/Dialect/NPUVector/Passes.h"
#include "akg/Utils/AnalysisCommon.hpp"

namespace mlir {
namespace npuvector {
#define GEN_PASS_DECL_OUTLINEVECTORFUNCTION
#define GEN_PASS_DEF_OUTLINEVECTORFUNCTION
#include "akg/Dialect/NPUVector/Passes.h.inc"

namespace {

// (sizes, maxSizes) pair extracted from a vector value's def chain.
// .first  = sizes
// .second = maxSizes
using VecSizesPair = std::pair<SmallVector<Value>, SmallVector<Value>>;

static FailureOr<int64_t> getVectorWidth(func::FuncOp funcOp) {
  assert(funcOp && "funcOp is null, cannot get vector width");
  auto archAttr = funcOp->getAttrOfType<StringAttr>("arch");
  if (!archAttr) {
    return funcOp.emitOpError("does not have 'arch' attribute, cannot get vector width");
  }
  uint32_t vectorWidth = akg::NpuInfo::getInstance(archAttr.getValue().str()).getRegVectorLength();
  if (vectorWidth == 0) {
    return funcOp.emitOpError("failed to get vector width from arch attribute: ") << archAttr.getValue();
  }
  return static_cast<int64_t>(vectorWidth);
}

// Compute the UB innermost-dim alignment (in ELEMENTS) for `elemType` on
// the device described by `parentFunc`. Since `getVectorWidth` returns the
// register vector width in bytes, per-dtype element alignment is:
//   alignment = vectorWidth / sizeof(elemType)
// e.g. with vectorWidth = 256B:
//   f32 -> 256/4 = 64
//   f16 -> 256/2 = 128
static FailureOr<int64_t> getUbAlignment(func::FuncOp parentFunc, Type elemType) {
  auto vectorWidthOr = getVectorWidth(parentFunc);
  if (failed(vectorWidthOr)) {
    return failure();
  }
  int64_t vectorWidth = *vectorWidthOr;

  if (!elemType || !elemType.isIntOrFloat()) {
    return vectorWidth;
  }

  unsigned bitWidth = elemType.getIntOrFloatBitWidth();
  unsigned byteWidth = std::max(1u, bitWidth / 8);
  int64_t alignment = vectorWidth / static_cast<int64_t>(byteWidth);
  return alignment > 0 ? alignment : int64_t{1};
}

static SmallVector<int64_t> alignUbShape(ArrayRef<int64_t> shape, int64_t kUbAlignment) {
  SmallVector<int64_t> aligned(shape.begin(), shape.end());
  if (aligned.empty() || kUbAlignment <= 1) {
    return aligned;
  }

  for (int64_t d : aligned) {
    if (d <= 0 || ShapedType::isDynamic(d)) {
      return aligned;
    }
  }

  int64_t last = aligned.back();
  aligned.back() = ((last + kUbAlignment - 1) / kUbAlignment) * kUbAlignment;
  return aligned;
}

// Build aligned maxSizes Values from existing (constant index) maxSizes.
// Used for npuvector.transfer_read whose source is a UB memref inside a
// vf kernel: maxSizes must be a multiple of `kUbAlignment` on the innermost
// dim (consistent with alignUbShape).
static SmallVector<Value> alignedMaxSizeValues(OpBuilder &b, Location loc, ValueRange origMaxSizes,
                                               int64_t kUbAlignment) {
  SmallVector<int64_t> shape;
  shape.reserve(origMaxSizes.size());
  for (Value v : origMaxSizes) {
    auto cst = v.getDefiningOp<arith::ConstantIndexOp>();
    if (!cst) {
      // Cannot statically align; keep original.
      return llvm::to_vector(origMaxSizes);
    }
    shape.push_back(cst.value());
  }
  SmallVector<int64_t> aligned = alignUbShape(shape, kUbAlignment);
  SmallVector<Value> result;
  result.reserve(aligned.size());
  std::transform(aligned.begin(), aligned.end(), std::back_inserter(result),
                 [&](int64_t d) { return b.create<arith::ConstantIndexOp>(loc, d); });
  return result;
}

static bool isVecSizesValid(const VecSizesPair &p) { return !p.first.empty() || !p.second.empty(); }

class OutlineVectorFunction : public mlir::npuvector::impl::OutlineVectorFunctionBase<OutlineVectorFunction> {
 public:
  OutlineVectorFunction() = default;
  OutlineVectorFunction(const OutlineVectorFunction &) = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, npuvector::NPUVectorDialect, memref::MemRefDialect, func::FuncDialect,
                    arith::ArithDialect>();
  }

  // -------------------------------------------------------------------------
  // Op / type classifiers
  // -------------------------------------------------------------------------

  static bool isNpuVectorType(Type t) { return t && llvm::isa<mlir::npuvector::NPUVectorType>(t); }

  static bool isNpuVectorOp(Operation *op) {
    if (op == nullptr) {
      return false;
    }
    auto isVec = [](Value v) { return isNpuVectorType(v.getType()); };
    return llvm::any_of(op->getOperands(), isVec) || llvm::any_of(op->getResults(), isVec);
  }

  static bool isCloneableConstant(Operation *op) {
    if (op == nullptr) {
      return false;
    }
    return isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp, arith::ConstantFloatOp>(op);
  }

  bool isRealVectorComputeOp(Operation *op) const {
    if (op == nullptr) {
      return false;
    }
    if (isCloneableConstant(op)) {
      return false;
    }
    return isNpuVectorOp(op);
  }

  static bool isExtractableOp(Operation *op) {
    if (op == nullptr) {
      return false;
    }
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return false;
    }
    if (op->getNumRegions() > 0) {
      return false;
    }
    if (isNpuVectorOp(op)) {
      return true;
    }
    if (isCloneableConstant(op)) {
      return true;
    }
    return false;
  }

  // True for memref view ops whose first operand is the source memref.
  static bool isMemrefViewOp(Operation *op) {
    if (op == nullptr) {
      return false;
    }
    return isa<memref::CollapseShapeOp, memref::ExpandShapeOp, memref::SubViewOp, memref::CastOp,
               memref::ReinterpretCastOp>(op);
  }

  // Allocate a `memref.alloc` at the start of the entry block of the function
  // containing `nearOp`.  Hoisting UB allocations out of inner regions (such
  // as scf.for bodies) ensures each buffer is allocated once per kernel
  // invocation rather than once per iteration, and guarantees the allocation
  // dominates every use even when the call site is nested.  Falls back to
  // inserting before `nearOp` when no enclosing func::FuncOp is found.
  static Value allocAtFuncStart(Operation *nearOp, Location loc, MemRefType t) {
    auto parentFunc = (nearOp != nullptr) ? nearOp->getParentOfType<func::FuncOp>() : func::FuncOp();
    if (!parentFunc || parentFunc.getBody().empty()) {
      OpBuilder b(nearOp);
      return b.create<memref::AllocOp>(loc, t);
    }
    Block &entry = parentFunc.getBody().front();
    OpBuilder b(&entry, entry.begin());
    return b.create<memref::AllocOp>(loc, t);
  }

  // Hoist memref view ops (collapse/expand/subview/cast/reinterpret_cast) out
  // of `group` when all their operands are defined outside the group.  Such
  // ops only prepare a memref operand (e.g. the destination of a
  // transfer_write) and must stay in the parent function.  Otherwise, when the
  // corresponding kernel argument is later resized during UB promotion, an
  // in-kernel view op (e.g. `memref.collapse_shape` collapsing to a rank-0
  // memref) would become invalid.  Keeping these ops in the parent also lets
  // the read/write through them be recognized by UB promotion (the kernel arg
  // becomes the viewed memref directly), mirroring how view ops that already
  // precede the first vector op are handled.
  static void hoistMemrefViewOps(SmallVector<Operation *> &group, Operation *insertBeforeOp) {
    llvm::SetVector<Operation *> groupSet(group.begin(), group.end());
    bool changed = true;
    while (changed) {
      changed = false;
      for (Operation *op : group) {
        if (groupSet.count(op) == 0u) {
          continue;
        }
        if (!isMemrefViewOp(op)) {
          continue;
        }
        bool allExternal = llvm::all_of(op->getOperands(), [&](Value v) {
          Operation *d = v.getDefiningOp();
          return !d || !groupSet.count(d);
        });
        if (!allExternal) {
          continue;
        }
        op->moveBefore(insertBeforeOp);
        groupSet.remove(op);
        changed = true;
      }
    }
    SmallVector<Operation *> filtered;
    filtered.reserve(group.size());
    for (Operation *op : group) {
      if (groupSet.count(op) != 0u) {
        filtered.push_back(op);
      }
    }
    group = std::move(filtered);
  }

  // Walk back through memref view ops (collapse/expand/subview/cast/...) to
  // find the root memref value.  Returns `v` itself if no view chain found.
  static Value traceToRootMemref(Value v) {
    Value cur = v;
    while (cur) {
      if (llvm::isa<BlockArgument>(cur)) {
        return cur;
      }
      Operation *def = cur.getDefiningOp();
      if ((def == nullptr) || !isMemrefViewOp(def) || def->getNumOperands() == 0) {
        return cur;
      }
      cur = def->getOperand(0);
    }
    return cur;
  }

  // -------------------------------------------------------------------------
  // Vector sizes extraction
  // -------------------------------------------------------------------------

  static SmallVector<Value> collectIndexOperands(Operation *op) {
    SmallVector<Value> idxOpnds;
    for (Value o : op->getOperands()) {
      if (o.getType().isIndex()) {
        idxOpnds.push_back(o);
      }
    }
    return idxOpnds;
  }

  static std::optional<VecSizesPair> extractFromTransferRead(mlir::npuvector::TransferReadOp rd, int rank) {
    VecSizesPair res;
    for (Value m : rd.getMaxSizes()) {
      res.second.push_back(m);
    }

    SmallVector<Value> idxOpnds = collectIndexOperands(rd);
    if (static_cast<int>(idxOpnds.size()) >= 2 * rank) {
      res.first.clear();
      for (int i = 0; i < rank; ++i) {
        res.first.push_back(idxOpnds[idxOpnds.size() - 2 * rank + i]);
      }
      if (res.second.empty()) {
        for (int i = 0; i < rank; ++i) {
          res.second.push_back(idxOpnds[idxOpnds.size() - rank + i]);
        }
      }
    }
    if (isVecSizesValid(res)) {
      return res;
    }
    return std::nullopt;
  }

  static std::optional<VecSizesPair> extractFromGenericNpuVector(Operation *defOp, int rank) {
    SmallVector<Value> idxOpnds = collectIndexOperands(defOp);
    if (static_cast<int>(idxOpnds.size()) < 2 * rank) {
      return std::nullopt;
    }

    VecSizesPair res;
    for (int i = 0; i < rank; ++i) {
      res.first.push_back(idxOpnds[idxOpnds.size() - 2 * rank + i]);
    }
    for (int i = 0; i < rank; ++i) {
      res.second.push_back(idxOpnds[idxOpnds.size() - rank + i]);
    }
    return res;
  }

  static std::optional<VecSizesPair> tryExtractVecSizes(Value v, int depth = 0) {
    if (depth > 8) {
      return std::nullopt;
    }
    auto vt = llvm::dyn_cast<mlir::npuvector::NPUVectorType>(v.getType());
    if (!vt) {
      return std::nullopt;
    }

    Operation *defOp = v.getDefiningOp();
    if (defOp == nullptr) {
      return std::nullopt;
    }

    int rank = static_cast<int>(vt.getShape().size());
    if (rank == 0) {
      return VecSizesPair{};
    }

    if (auto rd = llvm::dyn_cast<mlir::npuvector::TransferReadOp>(defOp)) {
      if (auto r = extractFromTransferRead(rd, rank)) {
        return r;
      }
    }
    if ((defOp->getDialect() != nullptr) && defOp->getDialect()->getNamespace() == "npuvector") {
      if (auto r = extractFromGenericNpuVector(defOp, rank)) {
        return r;
      }
    }
    for (Value operand : defOp->getOperands()) {
      if (isNpuVectorType(operand.getType())) {
        if (auto r = tryExtractVecSizes(operand, depth + 1)) {
          return r;
        }
      }
    }
    return std::nullopt;
  }

  // =========================================================================
  // Step 1: Extract vector computations into standalone vf functions
  // =========================================================================

  static SmallVector<SmallVector<Operation *>> splitBlockSegments(Block &block) {
    SmallVector<SmallVector<Operation *>> segments;
    SmallVector<Operation *> current;
    for (Operation &op : block) {
      if (op.hasTrait<OpTrait::IsTerminator>() || op.getNumRegions() > 0) {
        if (!current.empty()) {
          segments.push_back(std::move(current));
          current.clear();
        }
        continue;
      }
      current.push_back(&op);
    }
    if (!current.empty()) {
      segments.push_back(std::move(current));
    }
    return segments;
  }

  bool findVectorRange(const SmallVector<Operation *> &seg, int &outFirst, int &outLast) const {
    outFirst = -1;
    outLast = -1;
    for (int i = 0; i < static_cast<int>(seg.size()); ++i) {
      if (isRealVectorComputeOp(seg[i])) {
        if (outFirst < 0) {
          outFirst = i;
        }
        outLast = i;
      }
    }
    return outFirst >= 0;
  }

  // A group consisting of a single npuvector.transfer_read is not worth
  // outlining: it just loads data and would produce a degenerate vf whose
  // only job is a read.  Leave such reads in place.
  static bool isSingleTransferReadGroup(const SmallVector<Operation *> &group) {
    return group.size() == 1 && llvm::isa<mlir::npuvector::TransferReadOp>(group.front());
  }

  void extractKernelsInBlock(func::FuncOp funcOp, Block &block, int &kernelId) {
    auto segments = splitBlockSegments(block);
    for (auto &seg : segments) {
      int firstIdx = -1, lastIdx = -1;
      if (!findVectorRange(seg, firstIdx, lastIdx)) {
        continue;
      }
      SmallVector<Operation *> group(seg.begin() + firstIdx, seg.begin() + lastIdx + 1);
      if (isSingleTransferReadGroup(group)) {
        continue;
      }
      extractGroupToFunction(funcOp, group, kernelId++);
    }
  }

  void extractKernels(func::FuncOp funcOp, int &kernelId) {
    SmallVector<Block *> blocks;
    funcOp.walk([&](Block *b) { blocks.push_back(b); });
    for (Block *b : blocks) {
      extractKernelsInBlock(funcOp, *b, kernelId);
    }
  }

  static llvm::SetVector<Value> collectExternalInputs(const SmallVector<Operation *> &group,
                                                      const llvm::SetVector<Operation *> &groupSet) {
    llvm::SetVector<Value> externalInputs;
    for (Operation *op : group) {
      for (Value operand : op->getOperands()) {
        Operation *defOp = operand.getDefiningOp();
        if ((defOp == nullptr) || (groupSet.count(defOp) == 0u)) {
          externalInputs.insert(operand);
        }
      }
    }
    return externalInputs;
  }

  static void partitionInputs(const llvm::SetVector<Value> &externalInputs, llvm::SetVector<Value> &realArgs,
                              SmallVector<Value> &constInputs) {
    for (Value v : externalInputs) {
      Operation *defOp = v.getDefiningOp();
      if ((defOp != nullptr) && isCloneableConstant(defOp)) {
        constInputs.push_back(v);
        continue;
      }
      realArgs.insert(v);
    }
  }

  static llvm::SetVector<Value> collectOutputs(const SmallVector<Operation *> &group,
                                               const llvm::SetVector<Operation *> &groupSet) {
    llvm::SetVector<Value> outputs;
    for (Operation *op : group) {
      for (Value res : op->getResults()) {
        for (Operation *user : res.getUsers()) {
          if (groupSet.count(user) == 0u) {
            outputs.insert(res);
            break;
          }
        }
      }
    }
    return outputs;
  }

  func::FuncOp buildKernelFuncShell(func::FuncOp parentFunc, Location loc, int id, ArrayRef<Type> inputTypes,
                                    ArrayRef<Type> outputTypes) {
    auto module = parentFunc->getParentOfType<ModuleOp>();
    OpBuilder modBuilder(module.getBodyRegion());
    modBuilder.setInsertionPointToEnd(module.getBody());
    std::string fname = (parentFunc.getName() + "_outlined_vf_" + std::to_string(id)).str();
    auto fnType = modBuilder.getFunctionType(inputTypes, outputTypes);
    auto newFunc = modBuilder.create<func::FuncOp>(loc, fname, fnType);
    newFunc.setPrivate();
    newFunc->setAttr(kVectorFunctionAttr, modBuilder.getUnitAttr());
    newFunc->setAttr(kNoInlineAttr, modBuilder.getUnitAttr());
    newFunc.addEntryBlock();
    return newFunc;
  }

  void populateKernelBody(func::FuncOp newFunc, const llvm::SetVector<Value> &realArgs,
                          const SmallVector<Value> &constInputs, const SmallVector<Operation *> &group,
                          const llvm::SetVector<Value> &outputs, Location loc) {
    Block *fnEntry = &newFunc.getBody().front();
    OpBuilder fnBuilder(fnEntry, fnEntry->begin());
    IRMapping mapping;

    unsigned i = 0;
    for (Value v : realArgs) {
      mapping.map(v, fnEntry->getArgument(i++));
    }

    for (Value v : constInputs) {
      Operation *cloned = fnBuilder.clone(*v.getDefiningOp());
      mapping.map(v, cloned->getResult(0));
    }
    for (Operation *op : group) {
      fnBuilder.clone(*op, mapping);
    }

    SmallVector<Value> retVals = llvm::to_vector(llvm::map_range(outputs, [&](Value v) { return mapping.lookup(v); }));
    fnBuilder.create<func::ReturnOp>(loc, retVals);
  }

  void emitCallAndErase(Operation *firstOp, func::FuncOp newFunc, const llvm::SetVector<Value> &realArgs,
                        const llvm::SetVector<Value> &outputs, SmallVector<Operation *> &group, Location loc) {
    OpBuilder callBuilder(firstOp);
    SmallVector<Value> callArgs(realArgs.begin(), realArgs.end());
    auto callOp = callBuilder.create<func::CallOp>(loc, newFunc, callArgs);
    callOp->setAttr(kVectorFunctionAttr, callBuilder.getUnitAttr());
    callOp->setAttr(kNoInlineAttr, callBuilder.getUnitAttr());

    unsigned i = 0;
    for (Value v : outputs) {
      v.replaceAllUsesWith(callOp.getResult(i));
      ++i;
    }
    for (auto it = group.rbegin(); it != group.rend(); ++it) {
      (*it)->erase();
    }
  }

  void extractGroupToFunction(func::FuncOp parentFunc, SmallVector<Operation *> &group, int id) {
    if (group.empty()) {
      return;
    }
    Operation *firstOp = group.front();
    Location loc = firstOp->getLoc();

    // Keep pure memref view ops (e.g. the collapse_shape feeding a
    // transfer_write destination) in the parent function instead of cloning
    // them into the kernel; see hoistMemrefViewOps for the rationale.
    hoistMemrefViewOps(group, firstOp);
    if (group.empty()) {
      return;
    }

    llvm::SetVector<Operation *> groupSet;
    for (Operation *op : group) {
      groupSet.insert(op);
    }

    auto externalInputs = collectExternalInputs(group, groupSet);
    llvm::SetVector<Value> realArgs;
    SmallVector<Value> constInputs;
    partitionInputs(externalInputs, realArgs, constInputs);
    auto outputs = collectOutputs(group, groupSet);

    auto getType = [](Value v) { return v.getType(); };
    SmallVector<Type> inputTypes = llvm::to_vector(llvm::map_range(realArgs, getType));
    SmallVector<Type> outputTypes = llvm::to_vector(llvm::map_range(outputs, getType));

    auto newFunc = buildKernelFuncShell(parentFunc, loc, id, inputTypes, outputTypes);
    populateKernelBody(newFunc, realArgs, constInputs, group, outputs, loc);
    emitCallAndErase(firstOp, newFunc, realArgs, outputs, group, loc);
  }

  // =========================================================================
  // Step 1.5: Convert vf return values into output memref params
  // Subsequent passes cannot handle vf functions that return npuvector or
  // scalar float values.  We drop the results and route each returned value
  // through an extra UB memref output parameter:
  //   - npuvector result -> aligned memref<...> UB; written via transfer_write
  //     inside the kernel, read back via transfer_read at the call site.
  //   - scalar (f32/f16/bf16/...) result -> memref<1xT> UB; written via
  //     memref.store inside the kernel, loaded back via memref.load outside.
  // =========================================================================

  struct RetUbInfo {
    bool isVector = false;
    MemRefType ubType;                       // UB memref type (aligned for vectors).
    SmallVector<int64_t> logicalShape;       // vector logical shape (sizes).
    mlir::npuvector::NPUVectorType vecType;  // original npuvector type.
    Type elemType;                           // element / scalar type.
  };

  // Compute UB info for a returned value.  Returns failure if the type cannot
  // be boxed into a UB memref (caller should then leave the kernel untouched).
  FailureOr<RetUbInfo> computeReturnUbInfo(Value retVal, func::FuncOp parentFunc) {
    RetUbInfo info;
    Type t = retVal.getType();
    if (auto vt = llvm::dyn_cast<mlir::npuvector::NPUVectorType>(t)) {
      Type elemType = vt.getElementType();
      SmallVector<int64_t> logical;
      if (auto vs = tryExtractVecSizes(retVal)) {
        logical = shapeFromMaxSizes(vs->second);
        if (logical.empty()) {
          logical = shapeFromMaxSizes(vs->first);
        }
      }
      if (logical.empty()) {
        for (int64_t d : vt.getShape()) {
          if (ShapedType::isDynamic(d)) {
            return failure();
          }
          logical.push_back(d);
        }
      }
      if (logical.empty()) {
        return failure();
      }
      auto kUbAlignmentOr = getUbAlignment(parentFunc, elemType);
      if (failed(kUbAlignmentOr)) {
        return failure();
      }
      info.isVector = true;
      info.vecType = vt;
      info.elemType = elemType;
      info.logicalShape = logical;
      info.ubType = MemRefType::get(alignUbShape(logical, *kUbAlignmentOr), elemType);
      return info;
    }
    if (t.isIntOrFloat()) {
      info.isVector = false;
      info.elemType = t;
      info.ubType = MemRefType::get({1}, t);
      return info;
    }
    return failure();
  }

  // Write a returned value into its UB output param inside the kernel.
  static void emitWriteToOutParam(OpBuilder &b, Location loc, Value retVal, Value outArg, const RetUbInfo &info,
                                  Value c0) {
    if (info.isVector) {
      unsigned rank = info.ubType.getRank();
      SmallVector<Value> idx((rank != 0u) ? rank : 1u, c0);
      b.create<mlir::npuvector::TransferWriteOp>(loc, TypeRange{}, retVal, outArg, idx, /*mask=*/Value());
    } else {
      b.create<memref::StoreOp>(loc, retVal, outArg, ValueRange{c0});
    }
  }

  // Read a value back from its UB output param at the call site.
  static Value emitReadFromOutParam(OpBuilder &b, Location loc, Value ubBuf, const RetUbInfo &info) {
    Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
    if (!info.isVector) {
      return b.create<memref::LoadOp>(loc, ubBuf, ValueRange{c0});
    }

    Type elemType = info.elemType;
    Value padding = b.create<arith::ConstantOp>(loc, elemType, b.getZeroAttr(elemType));
    SmallVector<Value> sizes;
    SmallVector<Value> maxSizes;
    ArrayRef<int64_t> bufShape = info.ubType.getShape();
    sizes.reserve(info.logicalShape.size());
    maxSizes.reserve(bufShape.size());
    std::transform(info.logicalShape.begin(), info.logicalShape.end(), std::back_inserter(sizes),
                   [&](int64_t d) { return b.create<arith::ConstantIndexOp>(loc, d); });
    std::transform(bufShape.begin(), bufShape.end(), std::back_inserter(maxSizes),
                   [&](int64_t d) { return b.create<arith::ConstantIndexOp>(loc, d); });
    unsigned rank = info.ubType.getRank();
    SmallVector<Value> rIdx((rank != 0u) ? rank : 1u, c0);
    return b.create<mlir::npuvector::TransferReadOp>(loc, info.vecType, ubBuf, rIdx, padding,
                                                     /*mask=*/Value(), sizes, maxSizes);
  }

  LogicalResult convertReturnsToOutParams(func::FuncOp kernelFunc) {
    if (kernelFunc.getBody().empty() || kernelFunc.getNumResults() == 0) {
      return success();
    }

    Block &entry = kernelFunc.getBody().front();
    auto retOp = llvm::dyn_cast<func::ReturnOp>(entry.getTerminator());
    if (!retOp || retOp.getNumOperands() == 0) {
      return success();
    }

    auto module = kernelFunc->getParentOfType<ModuleOp>();
    auto callOps = collectCallers(module, kernelFunc);
    func::FuncOp parentFunc = pickParentFunc(callOps);

    // Compute UB info for every returned value first; bail out (leave the
    // kernel untouched) if any result type cannot be boxed into a UB memref.
    SmallVector<RetUbInfo> retInfos;
    retInfos.reserve(retOp.getNumOperands());
    for (Value rv : retOp.getOperands()) {
      auto infoOr = computeReturnUbInfo(rv, parentFunc);
      if (failed(infoOr)) {
        return success();
      }
      retInfos.push_back(*infoOr);
    }

    Location loc = kernelFunc.getLoc();

    // 1. Append UB output params and write returned values into them.
    SmallVector<Value> outArgs;
    outArgs.reserve(retInfos.size());
    std::transform(retInfos.begin(), retInfos.end(), std::back_inserter(outArgs),
                   [&](const RetUbInfo &info) { return entry.addArgument(info.ubType, loc); });

    OpBuilder wb(retOp);
    Value c0k = wb.create<arith::ConstantIndexOp>(loc, 0);
    for (size_t k = 0; k < retInfos.size(); ++k) {
      emitWriteToOutParam(wb, loc, retOp.getOperand(k), outArgs[k], retInfos[k], c0k);
    }

    // 2. Replace the return with an empty one and drop the results.
    wb.create<func::ReturnOp>(loc);
    retOp.erase();
    SmallVector<Type> inTypes;
    inTypes.reserve(entry.getNumArguments());
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      inTypes.push_back(entry.getArgument(i).getType());
    }
    kernelFunc.setType(FunctionType::get(kernelFunc.getContext(), inTypes, TypeRange{}));

    // 3. Rewrite every call site: alloc UB, append as operands, read back.
    for (func::CallOp call : callOps) {
      OpBuilder cb(call);
      Location cloc = call.getLoc();
      SmallVector<Value> ubBufs;
      ubBufs.reserve(retInfos.size());
      std::transform(retInfos.begin(), retInfos.end(), std::back_inserter(ubBufs),
                     [&](const RetUbInfo &info) { return allocAtFuncStart(call, cloc, info.ubType); });

      SmallVector<Value> newOperands(call.getOperands().begin(), call.getOperands().end());
      newOperands.append(ubBufs.begin(), ubBufs.end());
      auto newCall = cb.create<func::CallOp>(cloc, kernelFunc, newOperands);
      if (call->hasAttr(kVectorFunctionAttr)) {
        newCall->setAttr(kVectorFunctionAttr, cb.getUnitAttr());
      }
      if (call->hasAttr(kNoInlineAttr)) {
        newCall->setAttr(kNoInlineAttr, cb.getUnitAttr());
      }

      cb.setInsertionPointAfter(newCall);
      for (size_t k = 0; k < retInfos.size(); ++k) {
        Value readback = emitReadFromOutParam(cb, cloc, ubBufs[k], retInfos[k]);
        call.getResult(k).replaceAllUsesWith(readback);
      }
      call.erase();
    }
    return success();
  }

  // =========================================================================
  // Step 2: gm -> ub promotion
  // =========================================================================

  using ArgReadMap = DenseMap<unsigned, mlir::npuvector::TransferReadOp>;
  using ArgWriteMap = DenseMap<unsigned, mlir::npuvector::TransferWriteOp>;

  static SmallVector<func::CallOp> collectCallers(ModuleOp module, func::FuncOp kernelFunc) {
    SmallVector<func::CallOp> callOps;
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == kernelFunc.getName()) {
        callOps.push_back(call);
      }
    });
    return callOps;
  }

  // Pick a representative non-vf parent function from a kernel's call sites.
  // Used to query `arch` for alignment computations.
  static func::FuncOp pickParentFunc(ArrayRef<func::CallOp> callOps) {
    for (func::CallOp call : callOps) {
      auto p = call->getParentOfType<func::FuncOp>();
      if (p && !p->hasAttr(kVectorFunctionAttr)) {
        return p;
      }
    }
    return callOps.empty() ? func::FuncOp() : callOps.front()->getParentOfType<func::FuncOp>();
  }

  static void collectArgToRead(func::FuncOp kernelFunc, ArgReadMap &argToRead) {
    kernelFunc.walk([&](mlir::npuvector::TransferReadOp op) {
      Value src = op.getSource();
      if (auto bArg = llvm::dyn_cast<BlockArgument>(src)) {
        if (bArg.getOwner() == &kernelFunc.getBody().front()) {
          argToRead[bArg.getArgNumber()] = op;
        }
      }
    });
  }

  static void collectArgToWrite(func::FuncOp kernelFunc, ArgWriteMap &argToWrite) {
    kernelFunc.walk([&](mlir::npuvector::TransferWriteOp op) {
      Value dst = op.getSource();
      if (auto bArg = llvm::dyn_cast<BlockArgument>(dst)) {
        if (bArg.getOwner() == &kernelFunc.getBody().front()) {
          argToWrite[bArg.getArgNumber()] = op;
        }
      }
    });
  }

  // Convert a ValueRange of constant index Values into per-dim static shape.
  // Returns empty if any value isn't an arith.constant index.
  static SmallVector<int64_t> shapeFromMaxSizes(ValueRange maxSizes) {
    SmallVector<int64_t> shape;
    for (Value mv : maxSizes) {
      auto cst = mv.getDefiningOp<arith::ConstantIndexOp>();
      if (!cst) {
        return {};
      }
      shape.push_back(cst.value());
    }
    return shape;
  }

  // Per-dim UB shape for a kernel arg.  The shape is taken directly from
  // the maxSizes operand of the npuvector.transfer_read (or, for write-only
  // args, from the producing transfer_read in the def chain of the value
  // being written).  This preserves the original npuvector rank/shape.
  static SmallVector<int64_t> computeArgShape(unsigned argIdx, MemRefType origType, ArgReadMap &argToRead,
                                              ArgWriteMap &argToWrite) {
    SmallVector<int64_t> shape;

    auto rIt = argToRead.find(argIdx);
    if (rIt != argToRead.end()) {
      shape = shapeFromMaxSizes(rIt->second.getMaxSizes());
      if (!shape.empty()) {
        return shape;
      }
    }

    auto wIt = argToWrite.find(argIdx);
    if (wIt != argToWrite.end()) {
      auto vs = tryExtractVecSizes(wIt->second.getVector());
      if (vs) {
        shape = shapeFromMaxSizes(vs->second);
        if (shape.empty()) {
          shape = shapeFromMaxSizes(vs->first);
        }
        if (!shape.empty()) {
          return shape;
        }
      }
    }

    if (origType && origType.hasStaticShape()) {
      shape.insert(shape.end(), origType.getShape().begin(), origType.getShape().end());
    }
    return shape;
  }

  static Value remapKernelValueToCaller(Value v, func::CallOp callOp, func::FuncOp kernelFunc) {
    if (auto bArg = llvm::dyn_cast<BlockArgument>(v)) {
      if (bArg.getOwner() == &kernelFunc.getBody().front()) {
        return callOp.getOperand(bArg.getArgNumber());
      }
      return {};
    }
    if (Operation *defOp = v.getDefiningOp()) {
      if (isCloneableConstant(defOp)) {
        OpBuilder b(callOp);
        b.setInsertionPoint(callOp);
        Operation *cloned = b.clone(*defOp);
        return cloned->getResult(0);
      }
    }
    return {};
  }

  static SmallVector<Value> buildGmIndices(ValueRange origIndices, func::CallOp callOp, func::FuncOp kernelFunc,
                                           Value c0) {
    SmallVector<Value> gmIndices;
    for (Value idx : origIndices) {
      Value mapped = remapKernelValueToCaller(idx, callOp, kernelFunc);
      if (!mapped) {
        mapped = c0;
      }
      gmIndices.push_back(mapped);
    }
    return gmIndices;
  }

  // Remap a kernel-side size ValueRange (the dynamic/max sizes of a transfer)
  // to caller values.  Returns empty if any element cannot be mapped, so the
  // caller can fall back to a static-size copy.
  static SmallVector<Value> remapSizesToCaller(ValueRange sizes, func::CallOp callOp, func::FuncOp kernelFunc) {
    SmallVector<Value> result;
    result.reserve(sizes.size());
    for (Value s : sizes) {
      Value mapped = remapKernelValueToCaller(s, callOp, kernelFunc);
      if (!mapped) {
        return {};
      }
      result.push_back(mapped);
    }
    return result;
  }

  // Emit GM -> UB transfer before the call.  `vecType`/`dynSizes`/`maxSizes`
  // describe the actual tile being moved; using the original transfer's
  // dynamic extents (rather than the padded UB size) ensures only the valid
  // region is copied for partial/tail tiles.
  static void emitGmToUbCopy(OpBuilder &builder, Location loc, Value origArg, Value ubBuf, ValueRange gmIndices,
                             Value c0, Type elemType, Type vecType, ValueRange dynSizes, ValueRange maxSizes,
                             unsigned ubRank) {
    Value padding = builder.create<arith::ConstantOp>(loc, elemType, builder.getZeroAttr(elemType));
    SmallVector<Value> ubIndices((ubRank != 0u) ? ubRank : 1u, c0);
    Value vec = builder.create<mlir::npuvector::TransferReadOp>(loc, vecType, origArg, gmIndices, padding,
                                                                /*mask=*/Value(), dynSizes, maxSizes);
    builder.create<mlir::npuvector::TransferWriteOp>(loc, TypeRange{}, vec, ubBuf, ubIndices, /*mask=*/Value());
  }

  // Emit UB -> GM transfer after the call.  Mirrors `emitGmToUbCopy`: only the
  // valid tile (described by `dynSizes`) is written back, so partial/tail
  // tiles do not overwrite neighboring GM data.
  static void emitUbToGmCopy(OpBuilder &builder, func::CallOp callOp, Location loc, Value ubBuf, Value origArg,
                             ValueRange gmIndices, Type elemType, Type vecType, ValueRange dynSizes,
                             ValueRange maxSizes, unsigned ubRank) {
    builder.setInsertionPointAfter(callOp);
    Value c0p = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value padding = builder.create<arith::ConstantOp>(loc, elemType, builder.getZeroAttr(elemType));
    SmallVector<Value> ubIndices((ubRank != 0u) ? ubRank : 1u, c0p);
    Value vec = builder.create<mlir::npuvector::TransferReadOp>(loc, vecType, ubBuf, ubIndices, padding,
                                                                /*mask=*/Value(), dynSizes, maxSizes);
    builder.create<mlir::npuvector::TransferWriteOp>(loc, TypeRange{}, vec, origArg, gmIndices, /*mask=*/Value());
    builder.setInsertionPoint(callOp);
  }

  // Determine tile extents (copyVecType, dynSizes, maxSizes) for GM<->UB copies.
  // Prefer the original transfer's dynamic sizes so partial/tail tiles copy only
  // the valid region.
  static void resolveGmUbCopyTileParams(func::CallOp callOp, func::FuncOp kernelFunc, OpBuilder &builder, Location loc,
                                        unsigned argIdx, ArgReadMap &argToRead, ArgWriteMap &argToWrite,
                                        ArrayRef<int64_t> ubShape, Type elemType, Type &copyVecType,
                                        SmallVector<Value> &dynSizes, SmallVector<Value> &maxSizes) {
    auto rIt = argToRead.find(argIdx);
    auto wIt = argToWrite.find(argIdx);
    copyVecType = mlir::npuvector::NPUVectorType::get(ubShape, elemType);
    bool dynamicCopy = false;
    SmallVector<Value> origDyn;
    SmallVector<Value> origMax;
    Type origVecType;
    if (rIt != argToRead.end()) {
      origVecType = rIt->second.getResult().getType();
      origDyn.assign(rIt->second.getDynamicSizes().begin(), rIt->second.getDynamicSizes().end());
      origMax.assign(rIt->second.getMaxSizes().begin(), rIt->second.getMaxSizes().end());
    } else if (wIt != argToWrite.end()) {
      origVecType = wIt->second.getVector().getType();
      if (auto vs = tryExtractVecSizes(wIt->second.getVector())) {
        origDyn = vs->first;
        origMax = vs->second;
      }
    }
    if (auto vt = llvm::dyn_cast_or_null<mlir::npuvector::NPUVectorType>(origVecType)) {
      unsigned numDyn = 0;
      for (int64_t d : vt.getShape()) {
        if (ShapedType::isDynamic(d)) {
          ++numDyn;
        }
      }
      SmallVector<Value> rd = remapSizesToCaller(origDyn, callOp, kernelFunc);
      SmallVector<Value> rm = remapSizesToCaller(origMax, callOp, kernelFunc);
      if (numDyn > 0 && rd.size() == numDyn && rm.size() == numDyn) {
        copyVecType = vt;
        dynSizes = std::move(rd);
        maxSizes = std::move(rm);
        dynamicCopy = true;
      }
    }
    if (!dynamicCopy) {
      std::transform(ubShape.begin(), ubShape.end(), std::back_inserter(dynSizes),
                     [&](int64_t d) { return builder.create<arith::ConstantIndexOp>(loc, d); });
      maxSizes = dynSizes;
    }
  }

  // Promote a single (call, arg) pair. Returns:
  //   - failure() on hard error (e.g. arch unavailable),
  //   - success() with Type() if the arg is not promoted,
  //   - success() with the new UB MemRefType if promoted.
  FailureOr<Type> promoteOneCallArg(func::CallOp callOp, unsigned i, func::FuncOp kernelFunc, ArgReadMap &argToRead,
                                    ArgWriteMap &argToWrite, const llvm::SetVector<Value> &parentFuncArgs,
                                    SmallVector<Value> &newCallArgs, func::FuncOp parentFunc) {
    Value origArg = callOp.getOperand(i);
    auto memrefType = llvm::dyn_cast<MemRefType>(origArg.getType());
    if (!memrefType) {
      return Type();
    }

    // Accept either:
    //   - a direct parent-function argument, or
    //   - a memref view chain whose root is a parent-function argument.
    Value rootMemref = traceToRootMemref(origArg);
    if (parentFuncArgs.count(rootMemref) == 0u) {
      return Type();
    }

    SmallVector<int64_t> ubShape = computeArgShape(i, memrefType, argToRead, argToWrite);
    if (ubShape.empty()) {
      return Type();
    }

    Type elemType = memrefType.getElementType();
    auto kUbAlignmentOr = getUbAlignment(parentFunc, elemType);
    if (failed(kUbAlignmentOr)) {
      return failure();
    }
    int64_t kUbAlignment = *kUbAlignmentOr;
    SmallVector<int64_t> ubBufShape = alignUbShape(ubShape, kUbAlignment);
    auto ubType = MemRefType::get(ubBufShape, elemType);

    OpBuilder builder(callOp);
    Location loc = callOp.getLoc();
    builder.setInsertionPoint(callOp);
    Value ubBuf = allocAtFuncStart(callOp, loc, ubType);
    Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);

    SmallVector<Value> gmIndices;
    auto rIt = argToRead.find(i);
    auto wIt = argToWrite.find(i);
    if (rIt != argToRead.end()) {
      gmIndices = buildGmIndices(rIt->second.getIndices(), callOp, kernelFunc, c0);
    } else if (wIt != argToWrite.end()) {
      gmIndices = buildGmIndices(wIt->second.getIndices(), callOp, kernelFunc, c0);
    }

    // Pad / truncate gmIndices so it matches the rank of the *view* memref.
    int64_t viewRank = memrefType.getRank();
    if (static_cast<int64_t>(gmIndices.size()) < viewRank) {
      while (static_cast<int64_t>(gmIndices.size()) < viewRank) {
        gmIndices.push_back(c0);
      }
    } else if (static_cast<int64_t>(gmIndices.size()) > viewRank) {
      gmIndices.resize(viewRank);
    }
    if (gmIndices.empty()) {
      gmIndices.push_back(c0);
    }

    Type copyVecType;
    SmallVector<Value> dynSizes;
    SmallVector<Value> maxSizes;
    resolveGmUbCopyTileParams(callOp, kernelFunc, builder, loc, i, argToRead, argToWrite, ubShape, elemType,
                              copyVecType, dynSizes, maxSizes);
    auto ubRank = static_cast<unsigned>(ubBufShape.size());

    if (rIt != argToRead.end()) {
      emitGmToUbCopy(builder, loc, origArg, ubBuf, gmIndices, c0, elemType, copyVecType, dynSizes, maxSizes, ubRank);
    }
    newCallArgs[i] = ubBuf;
    if (wIt != argToWrite.end()) {
      emitUbToGmCopy(builder, callOp, loc, ubBuf, origArg, gmIndices, elemType, copyVecType, dynSizes, maxSizes,
                     ubRank);
    }
    return Type(ubType);
  }

  // Returns failure() on hard error.  On success, sets `updated` to whether
  // any operand was promoted.
  LogicalResult processOneCall(func::CallOp callOp, func::FuncOp kernelFunc, ArgReadMap &argToRead,
                               ArgWriteMap &argToWrite, SmallVector<Type> &newArgTypes, bool &updated) {
    updated = false;
    auto parentFunc = callOp->getParentOfType<func::FuncOp>();
    if (!parentFunc) {
      return success();
    }

    llvm::SetVector<Value> parentFuncArgs;
    for (Value v : parentFunc.getArguments()) {
      parentFuncArgs.insert(v);
    }

    SmallVector<Value> newCallArgs(callOp.getOperands().begin(), callOp.getOperands().end());
    for (unsigned i = 0; i < callOp.getNumOperands(); ++i) {
      auto ubTypeOr =
        promoteOneCallArg(callOp, i, kernelFunc, argToRead, argToWrite, parentFuncArgs, newCallArgs, parentFunc);
      if (failed(ubTypeOr)) {
        return failure();
      }
      Type ubType = *ubTypeOr;
      if (ubType) {
        newArgTypes[i] = ubType;
        updated = true;
      }
    }
    callOp->setOperands(newCallArgs);
    return success();
  }

  static void collectTransfersToFix(func::FuncOp kernelFunc, Block &entry, const SmallVector<Type> &newArgTypes,
                                    SmallVector<mlir::npuvector::TransferReadOp> &readsToFix,
                                    SmallVector<mlir::npuvector::TransferWriteOp> &writesToFix) {
    kernelFunc.walk([&](mlir::npuvector::TransferReadOp op) {
      auto bArg = llvm::dyn_cast<BlockArgument>(op.getSource());
      if (!bArg || bArg.getOwner() != &entry) {
        return;
      }
      if (!newArgTypes[bArg.getArgNumber()]) {
        return;
      }
      readsToFix.push_back(op);
    });
    kernelFunc.walk([&](mlir::npuvector::TransferWriteOp op) {
      auto bArg = llvm::dyn_cast<BlockArgument>(op.getSource());
      if (!bArg || bArg.getOwner() != &entry) {
        return;
      }
      if (!newArgTypes[bArg.getArgNumber()]) {
        return;
      }
      writesToFix.push_back(op);
    });
  }

  LogicalResult updateKernelSignatureAndIndices(func::FuncOp kernelFunc, const SmallVector<Type> &newArgTypes,
                                                func::FuncOp parentFunc) {
    SmallVector<Type> finalArgTypes;
    for (unsigned i = 0; i < kernelFunc.getNumArguments(); ++i) {
      finalArgTypes.push_back(newArgTypes[i] ? newArgTypes[i] : kernelFunc.getArgument(i).getType());
    }
    auto newFnType = FunctionType::get(kernelFunc.getContext(), finalArgTypes, kernelFunc.getResultTypes());
    kernelFunc.setType(newFnType);
    Block &entry = kernelFunc.getBody().front();
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      entry.getArgument(i).setType(finalArgTypes[i]);
    }

    OpBuilder kb(&entry, entry.begin());
    Value c0k = kb.create<arith::ConstantIndexOp>(kernelFunc.getLoc(), 0);

    SmallVector<mlir::npuvector::TransferReadOp> readsToFix;
    SmallVector<mlir::npuvector::TransferWriteOp> writesToFix;
    collectTransfersToFix(kernelFunc, entry, newArgTypes, readsToFix, writesToFix);

    for (auto rd : readsToFix) {
      auto mt = llvm::dyn_cast<MemRefType>(rd.getSource().getType());
      unsigned r = mt ? mt.getRank() : rd.getIndices().size();
      SmallVector<Value> newIdx(r, c0k);
      rd.getIndicesMutable().assign(newIdx);

      // Source is now a UB memref: align maxSizes to a multiple of the
      // per-dtype UB alignment on the innermost dim.
      OpBuilder rb(rd);
      Type elemType = mt ? mt.getElementType()
                         : llvm::cast<mlir::npuvector::NPUVectorType>(rd.getResult().getType()).getElementType();
      auto kUbAlignmentOr = getUbAlignment(parentFunc, elemType);
      if (failed(kUbAlignmentOr)) {
        return failure();
      }
      int64_t kUbAlignment = *kUbAlignmentOr;
      SmallVector<Value> newMax = alignedMaxSizeValues(rb, rd.getLoc(), rd.getMaxSizes(), kUbAlignment);
      rd.getMaxSizesMutable().assign(newMax);
    }
    for (auto wr : writesToFix) {
      auto mt = llvm::dyn_cast<MemRefType>(wr.getSource().getType());
      unsigned r = mt ? mt.getRank() : wr.getIndices().size();
      SmallVector<Value> newIdx(r, c0k);
      wr.getIndicesMutable().assign(newIdx);
    }
    return success();
  }

  LogicalResult promoteToUB(func::FuncOp kernelFunc) {
    auto module = kernelFunc->getParentOfType<ModuleOp>();
    auto callOps = collectCallers(module, kernelFunc);
    if (callOps.empty()) {
      return success();
    }

    func::FuncOp parentFunc = pickParentFunc(callOps);

    ArgReadMap argToRead;
    ArgWriteMap argToWrite;
    collectArgToRead(kernelFunc, argToRead);
    collectArgToWrite(kernelFunc, argToWrite);

    SmallVector<Type> newArgTypes(kernelFunc.getNumArguments(), Type());
    bool sigUpdated = false;
    for (func::CallOp callOp : callOps) {
      bool updated = false;
      if (failed(processOneCall(callOp, kernelFunc, argToRead, argToWrite, newArgTypes, updated))) {
        return failure();
      }
      sigUpdated = sigUpdated || updated;
    }
    if (!sigUpdated) {
      return success();
    }
    return updateKernelSignatureAndIndices(kernelFunc, newArgTypes, parentFunc);
  }

  // =========================================================================
  // Step 2.3: Promote NPUVector typed kernel args to UB memref
  // =========================================================================

  // Per-dim UB shape for an npuvector-typed kernel arg.  Read directly from
  // the producing transfer_read's maxSizes operand on the caller side.
  static SmallVector<int64_t> computeVectorArgShape(mlir::npuvector::NPUVectorType vt, unsigned argIdx,
                                                    ArrayRef<func::CallOp> callOps) {
    for (func::CallOp callOp : callOps) {
      Value vec = callOp.getOperand(argIdx);
      auto vs = tryExtractVecSizes(vec);
      if (!vs) {
        continue;
      }
      SmallVector<int64_t> shape = shapeFromMaxSizes(vs->second);
      if (shape.empty()) {
        shape = shapeFromMaxSizes(vs->first);
      }
      if (!shape.empty()) {
        return shape;
      }
    }
    // Fallback: if vector type has fully static shape, use it.
    SmallVector<int64_t> shape;
    for (int64_t d : vt.getShape()) {
      if (ShapedType::isDynamic(d)) {
        return {};
      }
      shape.push_back(d);
    }
    return shape;
  }

  LogicalResult collectVectorArgsToPromote(Block &entry, ArrayRef<func::CallOp> callOps, func::FuncOp parentFunc,
                                           SmallVector<unsigned> &vecArgIdx,
                                           SmallVector<mlir::npuvector::NPUVectorType> &vecArgOrigType,
                                           SmallVector<MemRefType> &vecArgBufType) {
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      auto vt = llvm::dyn_cast<mlir::npuvector::NPUVectorType>(entry.getArgument(i).getType());
      if (!vt) {
        continue;
      }
      SmallVector<int64_t> shape = computeVectorArgShape(vt, i, callOps);
      if (shape.empty()) {
        continue;
      }

      auto kUbAlignmentOr = getUbAlignment(parentFunc, vt.getElementType());
      if (failed(kUbAlignmentOr)) {
        return failure();
      }
      int64_t kUbAlignment = *kUbAlignmentOr;
      SmallVector<int64_t> bufShape = alignUbShape(shape, kUbAlignment);
      vecArgIdx.push_back(i);
      vecArgOrigType.push_back(vt);
      vecArgBufType.push_back(MemRefType::get(bufShape, vt.getElementType()));
    }
    return success();
  }

  void updateCallersForVectorPromotion(ArrayRef<func::CallOp> callOps, ArrayRef<unsigned> vecArgIdx,
                                       ArrayRef<MemRefType> vecArgBufType) {
    for (func::CallOp callOp : callOps) {
      OpBuilder b(callOp);
      Location loc = callOp.getLoc();
      SmallVector<Value> newOperands(callOp.getOperands().begin(), callOp.getOperands().end());
      for (size_t k = 0; k < vecArgIdx.size(); ++k) {
        unsigned ai = vecArgIdx[k];
        Value vec = callOp.getOperand(ai);
        b.setInsertionPoint(callOp);
        Value ubBuf = allocAtFuncStart(callOp, loc, vecArgBufType[k]);
        Value c0 = b.create<arith::ConstantIndexOp>(loc, 0);
        unsigned rank = vecArgBufType[k].getRank();
        SmallVector<Value> wIdx((rank != 0u) ? rank : 1u, c0);
        b.create<mlir::npuvector::TransferWriteOp>(loc, TypeRange{}, vec, ubBuf, wIdx, /*mask=*/Value());
        newOperands[ai] = ubBuf;
      }
      callOp->setOperands(newOperands);
    }
  }

  LogicalResult emitVectorReloadAtEntry(OpBuilder &kb, Location kloc, Value c0k, BlockArgument arg,
                                        mlir::npuvector::NPUVectorType vt, func::FuncOp parentFunc) {
    Type elemType = vt.getElementType();
    Value padding = kb.create<arith::ConstantOp>(kloc, elemType, kb.getZeroAttr(elemType));

    // Collect logical shape (sizes) and aligned shape (maxSizes).
    SmallVector<int64_t> logicalShape;
    auto shape = vt.getShape();
    logicalShape.reserve(shape.size());
    std::transform(shape.begin(), shape.end(), std::back_inserter(logicalShape),
                   [](int64_t d) { return ShapedType::isDynamic(d) ? 0 : d; });
    auto kUbAlignmentOr = getUbAlignment(parentFunc, elemType);
    if (failed(kUbAlignmentOr)) {
      return failure();
    }
    int64_t kUbAlignment = *kUbAlignmentOr;
    SmallVector<int64_t> alignedShape = alignUbShape(logicalShape, kUbAlignment);
    SmallVector<Value> sizes;
    SmallVector<Value> maxSizes;
    sizes.reserve(logicalShape.size());
    maxSizes.reserve(alignedShape.size());
    for (size_t i = 0; i < logicalShape.size(); ++i) {
      sizes.push_back(kb.create<arith::ConstantIndexOp>(kloc, logicalShape[i]));
      maxSizes.push_back(kb.create<arith::ConstantIndexOp>(kloc, alignedShape[i]));
    }
    auto mt = llvm::dyn_cast<MemRefType>(arg.getType());
    unsigned rank = mt ? static_cast<unsigned>(mt.getRank()) : 1u;
    SmallVector<Value> rIdx((rank != 0u) ? rank : 1u, c0k);
    Value loaded =
      kb.create<mlir::npuvector::TransferReadOp>(kloc, vt, arg, rIdx, padding, /*mask=*/Value(), sizes, maxSizes);
    arg.replaceAllUsesExcept(loaded, loaded.getDefiningOp());
    return success();
  }

  LogicalResult rewriteKernelForVectorPromotion(func::FuncOp kernelFunc, Block &entry, ArrayRef<unsigned> vecArgIdx,
                                                ArrayRef<mlir::npuvector::NPUVectorType> vecArgOrigType,
                                                ArrayRef<MemRefType> vecArgBufType, func::FuncOp parentFunc) {
    SmallVector<Type> finalArgTypes;
    finalArgTypes.reserve(entry.getNumArguments());
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      finalArgTypes.push_back(entry.getArgument(i).getType());
    }
    for (size_t k = 0; k < vecArgIdx.size(); ++k) {
      finalArgTypes[vecArgIdx[k]] = vecArgBufType[k];
    }

    auto newFnType = FunctionType::get(kernelFunc.getContext(), finalArgTypes, kernelFunc.getResultTypes());
    kernelFunc.setType(newFnType);
    for (size_t k = 0; k < vecArgIdx.size(); ++k) {
      entry.getArgument(vecArgIdx[k]).setType(vecArgBufType[k]);
    }

    OpBuilder kb(&entry, entry.begin());
    Location kloc = kernelFunc.getLoc();
    Value c0k = kb.create<arith::ConstantIndexOp>(kloc, 0);

    for (size_t k = 0; k < vecArgIdx.size(); ++k) {
      BlockArgument arg = entry.getArgument(vecArgIdx[k]);
      if (failed(emitVectorReloadAtEntry(kb, kloc, c0k, arg, vecArgOrigType[k], parentFunc))) {
        return failure();
      }
    }
    return success();
  }

  LogicalResult promoteVectorArgsToUB(func::FuncOp kernelFunc) {
    auto module = kernelFunc->getParentOfType<ModuleOp>();
    auto callOps = collectCallers(module, kernelFunc);
    if (callOps.empty()) {
      return success();
    }
    if (kernelFunc.getBody().empty()) {
      return success();
    }

    func::FuncOp parentFunc = pickParentFunc(callOps);

    Block &entry = kernelFunc.getBody().front();
    SmallVector<unsigned> vecArgIdx;
    SmallVector<mlir::npuvector::NPUVectorType> vecArgOrigType;
    SmallVector<MemRefType> vecArgBufType;
    if (failed(collectVectorArgsToPromote(entry, callOps, parentFunc, vecArgIdx, vecArgOrigType, vecArgBufType))) {
      return failure();
    }
    if (vecArgIdx.empty()) {
      return success();
    }

    updateCallersForVectorPromotion(callOps, vecArgIdx, vecArgBufType);
    return rewriteKernelForVectorPromotion(kernelFunc, entry, vecArgIdx, vecArgOrigType, vecArgBufType, parentFunc);
  }

  // =========================================================================
  // Step 2.4: Drop unused kernel args
  // =========================================================================

  static SmallVector<unsigned> findUnusedArgs(Block &entry, llvm::BitVector &toErase) {
    SmallVector<unsigned> unusedIdx;
    for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
      if (entry.getArgument(i).use_empty()) {
        toErase.set(i);
        unusedIdx.push_back(i);
      }
    }
    return unusedIdx;
  }

  void stripCallerOperands(ModuleOp module, func::FuncOp kernelFunc, const llvm::DenseSet<unsigned> &unusedSet) {
    auto callOps = collectCallers(module, kernelFunc);
    for (func::CallOp call : callOps) {
      SmallVector<Value> newOperands;
      newOperands.reserve(call.getNumOperands() - unusedSet.size());
      for (unsigned i = 0; i < call.getNumOperands(); ++i) {
        if (unusedSet.count(i) == 0u) {
          newOperands.push_back(call.getOperand(i));
        }
      }
      call->setOperands(newOperands);
    }
  }

  void removeUnusedArgs(func::FuncOp kernelFunc) {
    if (kernelFunc.getBody().empty()) {
      return;
    }
    Block &entry = kernelFunc.getBody().front();
    if (entry.getNumArguments() == 0) {
      return;
    }

    llvm::BitVector toErase(entry.getNumArguments());
    auto unusedIdx = findUnusedArgs(entry, toErase);
    if (unusedIdx.empty()) {
      return;
    }

    llvm::DenseSet<unsigned> unusedSet(unusedIdx.begin(), unusedIdx.end());
    auto module = kernelFunc->getParentOfType<ModuleOp>();
    stripCallerOperands(module, kernelFunc, unusedSet);

    entry.eraseArguments(toErase);
    SmallVector<Type> newArgTypes =
      llvm::to_vector(llvm::map_range(entry.getArguments(), [](BlockArgument a) { return a.getType(); }));
    auto newFnType = FunctionType::get(kernelFunc.getContext(), newArgTypes, kernelFunc.getResultTypes());
    kernelFunc.setType(newFnType);
  }

  // =========================================================================
  // Pass entry
  // =========================================================================

  void collectTargetFuncs(ModuleOp module, SmallVector<func::FuncOp> &targetFuncs) {
    module.walk([&](func::FuncOp f) {
      if (!f->hasAttr(kVectorFunctionAttr)) {
        targetFuncs.push_back(f);
      }
    });
  }

  void collectKernelFuncs(ModuleOp module, SmallVector<func::FuncOp> &kernelFuncs) {
    module.walk([&](func::FuncOp f) {
      if (f->hasAttr(kVectorFunctionAttr)) {
        kernelFuncs.push_back(f);
      }
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Step 1: outline vector compute groups into private vf functions.
    SmallVector<func::FuncOp> targetFuncs;
    collectTargetFuncs(module, targetFuncs);
    int kernelId = 0;
    for (auto f : targetFuncs) {
      extractKernels(f, kernelId);
    }

    // Steps 1.5, 2, 2.3, 2.4
    SmallVector<func::FuncOp> kernelFuncs;
    collectKernelFuncs(module, kernelFuncs);

    // Step 1.5: drop vf return values, route them through UB memref out-params.
    if (std::any_of(kernelFuncs.begin(), kernelFuncs.end(),
                    [this](func::FuncOp f) { return failed(convertReturnsToOutParams(f)); })) {
      signalPassFailure();
      return;
    }

    if (std::any_of(kernelFuncs.begin(), kernelFuncs.end(),
                    [this](func::FuncOp f) { return failed(promoteToUB(f)); })) {
      signalPassFailure();
      return;
    }

    if (std::any_of(kernelFuncs.begin(), kernelFuncs.end(),
                    [this](func::FuncOp f) { return failed(promoteVectorArgsToUB(f)); })) {
      signalPassFailure();
      return;
    }

    for (auto f : kernelFuncs) {
      removeUnusedArgs(f);
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createOutlineVectorFunctionPass() {
  return std::make_unique<OutlineVectorFunction>();
}

}  // namespace npuvector
}  // namespace mlir
