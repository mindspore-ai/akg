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

//===- VectorTransferTensorize.cpp ------------------------------*- C++ -*-===//
//
//  Converts:
//     1. vector.transfer_{read|write}
//     2. scalar / vector arith.constant
//     3. elementwise pure operators (generic, interface/type-based)
//  into tensor form, and marks created tensor ops with a "restrict" attr.
//
//===----------------------------------------------------------------------===*/

#include "akg/Dialect/Affine/Transforms/VectorTransferTensorize.h"

#include <algorithm>
#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "vector-transfer-tensorize"

using mlir::AffineMapAttr;
using mlir::Attribute;
using mlir::DenseElementsAttr;
using mlir::DenseSet;
using mlir::Dialect;
using mlir::DialectRegistry;
using mlir::FailureOr;
using mlir::IRMapping;
using mlir::IntegerAttr;
using mlir::Location;
using mlir::MemRefType;
using mlir::OpBuilder;
using mlir::OpFoldResult;
using mlir::Operation;
using mlir::OperationState;
using mlir::Pass;
using mlir::RankedTensorType;
using mlir::SmallVector;
using mlir::StridedLayoutAttr;
using mlir::Type;
using mlir::Value;
using mlir::ValueRange;

namespace arith = mlir::arith;
namespace math = mlir::math;
namespace vector = mlir::vector;
namespace bufferization = mlir::bufferization;
namespace memref = mlir::memref;
namespace tensor = mlir::tensor;
namespace func = mlir::func;

//==============================================================================
// Utilities
//==============================================================================

static RankedTensorType memrefToTensorType(MemRefType mty) {
  if (!mty || !mty.hasStaticShape()) {
    return {};
  }
  return RankedTensorType::get(mty.getShape(), mty.getElementType());
}

static bool isNumericLike(Type t) {
  if (auto rt = t.dyn_cast<RankedTensorType>()) {
    return rt.getElementType().isIntOrFloat();
  }
  if (auto vt = t.dyn_cast<mlir::VectorType>()) {
    return vt.getElementType().isIntOrFloat();
  }
  return t.isIntOrFloat();
}

//==============================================================================
// TensorizeState
//==============================================================================

struct TensorizeState {
  explicit TensorizeState(mlir::MLIRContext *ctx) : builder(ctx) {}

  OpBuilder builder;
  IRMapping valueMap;                 // old value -> tensor value
  DenseSet<Operation *> coveredOps;   // Processed ops
  SmallVector<Operation *> newOps;    // New ops (for debugging or rollback)

  void track(Operation *op) { newOps.push_back(op); }
  void addMap(Value oldV, Value newV) {
    if (!valueMap.contains(oldV)) {
      valueMap.map(oldV, newV);
    }
  }
  void cover(Operation *op) { coveredOps.insert(op); }
};

//==============================================================================
// Build 1-D subview along the innermost dimension
// sliceLen == 0 → use the full length of the last dimension
// sliceLen != 0 → use the specified length
//==============================================================================

static FailureOr<Value> buildSubview1D(OpBuilder &b,
                                       Location loc,
                                       Value src,
                                       ValueRange indices,
                                       unsigned sliceLen = 0) {
  auto srcTy = src.getType().dyn_cast<MemRefType>();
  if (!srcTy || !srcTy.hasStaticShape()) {
    return mlir::failure();
  }
  unsigned rank = srcTy.getRank();
  if (rank == 0) {
    return mlir::failure();
  }

  SmallVector<OpFoldResult> offs;
  SmallVector<OpFoldResult> szs;
  SmallVector<OpFoldResult> strides(rank, b.getIndexAttr(1));
  bool dynOff = false;
  for (unsigned i = 0; i < rank - 1; ++i) {
    if (srcTy.getDimSize(i) == 1) {
      offs.push_back(b.getIndexAttr(0));
    } else if (i < indices.size()) {
      offs.push_back(indices[i]);
      dynOff = true;
    } else {
      offs.push_back(b.getIndexAttr(0));
    }
    szs.push_back(b.getIndexAttr(1));
  }

  offs.push_back(b.getIndexAttr(0));
  int64_t fullLen = srcTy.getShape().back();
  int64_t len = sliceLen ? static_cast<int64_t>(sliceLen) : fullLen;
  szs.push_back(b.getIndexAttr(len));

  MemRefType resTy;
  if (dynOff) {
    auto layout = StridedLayoutAttr::get(b.getContext(), mlir::ShapedType::kDynamic, {1});
    resTy = MemRefType::get({len},
                            srcTy.getElementType(),
                            layout,
                            srcTy.getMemorySpace());
  } else {
    resTy = MemRefType::get({len},
                            srcTy.getElementType(),
                            AffineMapAttr(),
                            srcTy.getMemorySpace());
  }

  auto sub = b.create<memref::SubViewOp>(loc, resTy, src, offs, szs, strides).getResult();
  return FailureOr<Value>(sub);
}

//==============================================================================
// Create tensor constant for broadcasting scalar / vector
//==============================================================================

static FailureOr<arith::ConstantOp> createTensorConstant(arith::ConstantOp cst,
                                                         RankedTensorType tty,
                                                         OpBuilder &b) {
  Attribute val = cst.getValue();
  // Direct DenseElements
  if (auto dense = val.dyn_cast<DenseElementsAttr>()) {
    arith::ConstantOp newOp;
    if (dense.getType() == tty) {
      newOp = b.create<arith::ConstantOp>(cst.getLoc(), dense);
    } else if (dense.getNumElements() == tty.getNumElements()) {
      newOp = b.create<arith::ConstantOp>(cst.getLoc(), dense.reshape(tty));
    } else {
      return mlir::failure();
    }
    return newOp;
  }
  // Scalar → broadcast
  Attribute elem = val;
  Type ety = tty.getElementType();
  if (auto ia = val.dyn_cast<IntegerAttr>()) {
    elem = (ia.getType() == ety) ? elem : IntegerAttr::get(ety, ia.getInt());
  } else if (auto fa = val.dyn_cast<mlir::FloatAttr>()) {
    elem = (fa.getType() == ety) ? elem : mlir::FloatAttr::get(ety, fa.getValue());
  } else {
    return mlir::failure();
  }
  SmallVector<Attribute> vec(tty.getNumElements(), elem);
  auto dense = DenseElementsAttr::get(tty, vec);
  auto newOp = b.create<arith::ConstantOp>(cst.getLoc(), dense);
  return newOp;
}

//==============================================================================
// Upgrade scalar constant to tensor
//==============================================================================

static mlir::LogicalResult upgradeScalarConstant(arith::ConstantOp cst,
                                                 RankedTensorType tty,
                                                 TensorizeState &s) {
  if (s.valueMap.contains(cst.getResult())) {
    return mlir::success();
  }
  auto tcstOr = createTensorConstant(cst, tty, s.builder);
  if (mlir::failed(tcstOr)) {
    return mlir::failure();
  }
  auto tcst = *tcstOr;
  s.track(tcst.getOperation());
  s.addMap(cst.getResult(), tcst.getResult());
  // Perform RAUW for non-transfer-read uses
  cst.getResult().replaceUsesWithIf(
      tcst.getResult(),
      [](mlir::OpOperand &use) {
        return !llvm::isa<vector::TransferReadOp>(use.getOwner());
      });
  return mlir::success();
}

//==============================================================================
// Tensorization helpers
//==============================================================================

static FailureOr<Operation *> tensorizeOneOp(Operation *, TensorizeState &);

static FailureOr<Operation *> handleTransferRead(vector::TransferReadOp rd,
                                                 TensorizeState &s) {
  unsigned sliceLen = rd.getVectorType().getNumElements();
  auto svOr = buildSubview1D(s.builder, rd.getLoc(), rd.getSource(), rd.getIndices(), sliceLen);
  if (mlir::failed(svOr)) {
    return mlir::failure();
  }
  auto tty = memrefToTensorType(llvm::cast<MemRefType>((*svOr).getType()));
  if (!tty) {
    return mlir::failure();
  }

  auto toT = s.builder.create<bufferization::ToTensorOp>(
                rd.getLoc(),
                tty,
                *svOr,
                /*restrict=*/true,
                /*writable=*/true);

  s.track(toT);
  s.addMap(rd.getResult(), toT.getResult());
  s.cover(rd);
  return FailureOr<Operation *>(toT.getOperation());
}

static FailureOr<Operation *> handleTransferWrite(vector::TransferWriteOp wr,
                                                  TensorizeState &s) {
  unsigned sliceLen = wr.getVectorType().getNumElements();
  auto svOr = buildSubview1D(s.builder, wr.getLoc(), wr.getSource(), wr.getIndices(), sliceLen);
  if (mlir::failed(svOr)) {
    return mlir::failure();
  }
  auto tty = memrefToTensorType(llvm::cast<MemRefType>((*svOr).getType()));
  if (!tty) {
    return mlir::failure();
  }

  Value vec = wr.getVector();
  Value tVal = s.valueMap.lookupOrNull(vec);
  if (!tVal) {
    if (auto cst = vec.getDefiningOp<arith::ConstantOp>()) {
      auto tcst = createTensorConstant(cst, tty, s.builder);
      if (mlir::failed(tcst)) {
        return mlir::failure();
      }
      s.track(tcst->getOperation());
      tVal = tcst->getResult();
    }
  }
  if (!tVal || tVal.getType() != tty) {
    return mlir::failure();
  }

  auto toMem = s.builder.create<bufferization::ToMemrefOp>(wr.getLoc(), (*svOr).getType(), tVal);
  auto cp = s.builder.create<memref::CopyOp>(wr.getLoc(), toMem, *svOr);
  s.track(toMem);
  s.track(cp);
  s.cover(wr);
  return FailureOr<Operation *>(cp.getOperation());
}

static FailureOr<Operation *> cloneElemOp(Operation *op,
                                          llvm::ArrayRef<Value> operands,
                                          TensorizeState &s) {
  OperationState st(op->getLoc(), op->getName());
  st.addOperands(operands);
  st.addAttributes(op->getAttrs());

  SmallVector<Type> resTys;
  resTys.reserve(op->getNumResults());
  for (Type rt : op->getResultTypes()) {
    if (auto tt = rt.dyn_cast<RankedTensorType>()) {
      resTys.push_back(tt);
      continue;
    }
    if (auto vt = rt.dyn_cast<mlir::VectorType>()) {
      resTys.push_back(
          RankedTensorType::get(vt.getShape(), vt.getElementType()));
      continue;
    }
    return mlir::failure();
  }
  st.addTypes(resTys);

  Operation *newOp = s.builder.create(st);
  s.track(newOp);
  for (auto [o, n] : llvm::zip(op->getResults(), newOp->getResults())) {
    s.addMap(o, n);
  }
  s.cover(op);
  return FailureOr<Operation *>(newOp);
}

static FailureOr<Operation *> handleElemOp(Operation *op, TensorizeState &s) {
  if (op->getNumRegions() != 0) {
    return mlir::failure();
  }
  if (auto mem = llvm::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (!mem.hasNoEffect()) {
      return mlir::failure();
    }
  }

  if (std::any_of(op->result_type_begin(), op->result_type_end(),
                  [](Type t) { return !isNumericLike(t); })) {
    return mlir::failure();
  }

  if (std::any_of(op->operand_type_begin(), op->operand_type_end(),
                  [](Type t) { return !isNumericLike(t); })) {
    return mlir::failure();
  }

  RankedTensorType tty = nullptr;
  bool found = false;
  for (Value v : op->getOperands()) {
    if (auto mapped = s.valueMap.lookupOrNull(v)) {
      if (auto rt = mapped.getType().dyn_cast<RankedTensorType>()) {
        tty = rt;
        found = true;
        break;
      }
    }
    if (auto rt = v.getType().dyn_cast<RankedTensorType>()) {
      tty = rt;
      found = true;
      break;
    }
  }
  if (!found || !tty) {
    return mlir::failure();
  }

  SmallVector<Value> newOps;
  newOps.reserve(op->getNumOperands());
  for (Value v : op->getOperands()) {
    if (auto mapped = s.valueMap.lookupOrNull(v)) {
      if (mapped.getType() != tty) {
        return mlir::failure();
      }
      newOps.push_back(mapped);
      continue;
    }
    if (v.getType() == tty) {
      newOps.push_back(v);
      continue;
    }
    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (mlir::failed(upgradeScalarConstant(cst, tty, s))) {
        return mlir::failure();
      }
      newOps.push_back(s.valueMap.lookup(cst.getResult()));
      continue;
    }
    return mlir::failure();
  }

  return cloneElemOp(op, newOps, s);
}

static FailureOr<Operation *> tensorizeOneOp(Operation *op, TensorizeState &s) {
  if (auto rd = llvm::dyn_cast<vector::TransferReadOp>(op)) {
    return handleTransferRead(rd, s);
  }
  if (auto wr = llvm::dyn_cast<vector::TransferWriteOp>(op)) {
    return handleTransferWrite(wr, s);
  }
  return handleElemOp(op, s);
}

namespace {
#define GEN_PASS_DECL_VECTORTRANSFERTENSORIZE
#define GEN_PASS_DEF_VECTORTRANSFERTENSORIZE
#include "akg/Dialect/Affine/Passes.h.inc"

struct VectorTransferTensorizePass
    : public impl::VectorTransferTensorizeBase<VectorTransferTensorizePass> {
  void getDependentDialects(DialectRegistry &r) const override {
    r.insert<arith::ArithDialect,
             math::MathDialect,
             vector::VectorDialect,
             bufferization::BufferizationDialect,
             memref::MemRefDialect,
             tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp fn = getOperation();
    TensorizeState s(fn.getContext());

    // === Main tensorization loop ==========================================
    SmallVector<Operation *> wl;
    fn.walk([&](Operation *op) { wl.push_back(op); });

    for (Operation *op : wl) {
      if (llvm::isa<func::FuncOp>(op) || s.coveredOps.contains(op)) {
        continue;
      }
      s.builder.setInsertionPoint(op);
      (void)tensorizeOneOp(op, s);
    }

    // === Remove covered ops that are no longer used =====================
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto it = s.coveredOps.begin(); it != s.coveredOps.end();) {
        Operation *op = *it;
        if (op->use_empty()) {
          ++it;
          s.coveredOps.erase(op);
          op->erase();
          changed = true;
        } else {
          ++it;
        }
      }
    }
  }
};
}  // namespace

namespace mlir::affine {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createVectorTransferTensorizePass() {
  return std::make_unique<VectorTransferTensorizePass>();
}

}  // namespace mlir::affine

