/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "akg/Dialect/NPUVector/Transforms/RefineNPUVectorStaticShape.h"

#include <algorithm>
#include <memory>
#include <optional>

#include "akg/Dialect/NPUVector/IR/NPUVector.h"
#include "akg/Dialect/NPUVector/Passes.h"
#include "akg/Utils/Constants.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace npuvector {
#define GEN_PASS_DECL_REFINENPUVECTORSTATICSHAPE
#define GEN_PASS_DEF_REFINENPUVECTORSTATICSHAPE
#include "akg/Dialect/NPUVector/Passes.h.inc"

namespace {

static std::optional<int64_t> getConstantIndex(Value value) {
  if (auto constantIndexOp = value.getDefiningOp<arith::ConstantIndexOp>()) {
    return constantIndexOp.value();
  }
  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (!value.getType().isIndex()) {
      return std::nullopt;
    }
    if (auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue())) {
      return integerAttr.getValue().getSExtValue();
    }
  }
  return std::nullopt;
}

static npuvector::NPUVectorType buildRefinedTypeFromSizes(npuvector::NPUVectorType oldType, ValueRange dynamicSizes) {
  const int64_t rank = oldType.getRank();
  if (rank == 0 || oldType.hasStaticShape() || dynamicSizes.empty()) {
    return oldType;
  }

  SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());
  bool changed = false;

  auto refineDim = [&](unsigned dim, Value sizeValue) {
    if (!ShapedType::isDynamic(newShape[dim])) {
      return;
    }
    std::optional<int64_t> size = getConstantIndex(sizeValue);
    if (!size) {
      return;
    }
    newShape[dim] = *size;
    changed = true;
  };

  if (dynamicSizes.size() == static_cast<size_t>(rank)) {
    for (auto [dim, sizeValue] : llvm::enumerate(dynamicSizes)) {
      refineDim(dim, sizeValue);
    }
  } else if (dynamicSizes.size() == static_cast<size_t>(oldType.getNumDynamicDims())) {
    unsigned sizeIdx = 0;
    for (unsigned dim = 0; dim < static_cast<unsigned>(rank); ++dim) {
      if (ShapedType::isDynamic(oldType.getDimSize(dim))) {
        refineDim(dim, dynamicSizes[sizeIdx++]);
      }
    }
  }

  if (!changed) {
    return oldType;
  }
  return npuvector::NPUVectorType::get(newShape, oldType.getElementType());
}

static bool setResultType(Value result, npuvector::NPUVectorType newType) {
  if (newType == nullptr || result.getType() == newType) {
    return false;
  }
  result.setType(newType);
  return true;
}

static bool refineFromDynamicSizes(Value result, ValueRange dynamicSizes) {
  auto oldType = dyn_cast<npuvector::NPUVectorType>(result.getType());
  if (oldType == nullptr) {
    return false;
  }
  return setResultType(result, buildRefinedTypeFromSizes(oldType, dynamicSizes));
}

static bool refineTransferRead(npuvector::TransferReadOp op) {
  return refineFromDynamicSizes(op.getResult(), op.getDynamicSizes());
}

static bool refineBroadcast(npuvector::BroadcastOp op) {
  return refineFromDynamicSizes(op.getResult(), op.getDynamicSizes());
}

static bool refineTranspose(npuvector::TransposeOp op) {
  auto srcType = dyn_cast<npuvector::NPUVectorType>(op.getVector().getType());
  auto oldType = dyn_cast<npuvector::NPUVectorType>(op.getResult().getType());
  if (srcType == nullptr || oldType == nullptr || srcType.getRank() != oldType.getRank()) {
    return false;
  }

  ArrayRef<int64_t> permutation = op.getPermutation();
  if (permutation.size() != static_cast<size_t>(oldType.getRank())) {
    return false;
  }

  SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());
  bool changed = false;
  for (unsigned resultDim = 0; resultDim < permutation.size(); ++resultDim) {
    int64_t sourceDim = permutation[resultDim];
    if (sourceDim < 0 || sourceDim >= srcType.getRank()) {
      return false;
    }
    int64_t sourceSize = srcType.getDimSize(static_cast<unsigned>(sourceDim));
    if (ShapedType::isDynamic(newShape[resultDim]) && !ShapedType::isDynamic(sourceSize)) {
      newShape[resultDim] = sourceSize;
      changed = true;
    }
  }

  if (!changed) {
    return false;
  }
  return setResultType(op.getResult(), npuvector::NPUVectorType::get(newShape, oldType.getElementType()));
}

static bool refineReduction(npuvector::ReductionOp op) {
  auto srcType = dyn_cast<npuvector::NPUVectorType>(op.getVector().getType());
  auto oldType = dyn_cast<npuvector::NPUVectorType>(op.getDest().getType());
  if (srcType == nullptr || oldType == nullptr || oldType.hasStaticShape()) {
    return false;
  }

  auto dimsAttr = op.getReductionDims();
  if (!dimsAttr.has_value() || dimsAttr->empty()) {
    return false;
  }

  const int64_t srcRank = srcType.getRank();
  const int64_t resultRank = oldType.getRank();
  llvm::DenseSet<int64_t> reduceDimSet(dimsAttr->begin(), dimsAttr->end());
  if (reduceDimSet.size() != dimsAttr->size()) {
    return false;
  }
  if (std::any_of(dimsAttr->begin(), dimsAttr->end(), [srcRank](int64_t dim) { return dim < 0 || dim >= srcRank; })) {
    return false;
  }

  int64_t expectedResultRank = 0;
  for (int64_t dim = 0; dim < srcRank; ++dim) {
    if (reduceDimSet.count(dim) == 0) {
      ++expectedResultRank;
    }
  }
  if (expectedResultRank != resultRank) {
    return false;
  }

  SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());
  bool changed = false;
  unsigned resultDim = 0;
  for (int64_t srcDim = 0; srcDim < srcRank; ++srcDim) {
    if (reduceDimSet.count(srcDim) != 0) {
      continue;
    }
    int64_t srcSize = srcType.getDimSize(static_cast<unsigned>(srcDim));
    if (ShapedType::isDynamic(newShape[resultDim]) && !ShapedType::isDynamic(srcSize)) {
      newShape[resultDim] = srcSize;
      changed = true;
    }
    ++resultDim;
  }

  if (!changed) {
    return false;
  }
  return setResultType(op.getDest(), npuvector::NPUVectorType::get(newShape, oldType.getElementType()));
}

static bool isShapePreservingElementwiseOp(Operation *op) {
  if (isa<npuvector::TransferReadOp, npuvector::TransferWriteOp, npuvector::BroadcastOp, npuvector::TransposeOp,
          npuvector::ReductionOp>(op)) {
    return false;
  }
  Dialect *dialect = op->getDialect();
  if (dialect == nullptr) {
    return false;
  }
  StringRef dialectName = dialect->getNamespace();
  return dialectName == "arith" || dialectName == "math" || dialectName == "npuvector";
}

static bool refineShapePreservingElementwise(Operation *op) {
  if (!isShapePreservingElementwiseOp(op) || op->getNumResults() != kUnaryOpOperandCount) {
    return false;
  }

  auto oldType = dyn_cast<npuvector::NPUVectorType>(op->getResult(0).getType());
  if (oldType == nullptr || oldType.hasStaticShape()) {
    return false;
  }

  for (Value operand : op->getOperands()) {
    auto operandType = dyn_cast<npuvector::NPUVectorType>(operand.getType());
    if (operandType == nullptr || operandType.getRank() != oldType.getRank()) {
      continue;
    }

    SmallVector<int64_t> newShape(oldType.getShape().begin(), oldType.getShape().end());
    bool changed = false;
    for (unsigned dim = 0; dim < static_cast<unsigned>(oldType.getRank()); ++dim) {
      int64_t operandSize = operandType.getDimSize(dim);
      if (ShapedType::isDynamic(newShape[dim]) && !ShapedType::isDynamic(operandSize)) {
        newShape[dim] = operandSize;
        changed = true;
      }
    }
    if (!changed) {
      continue;
    }
    return setResultType(op->getResult(0), npuvector::NPUVectorType::get(newShape, oldType.getElementType()));
  }
  return false;
}

static bool refineOneOp(Operation *op) {
  if (auto readOp = dyn_cast<npuvector::TransferReadOp>(op)) {
    return refineTransferRead(readOp);
  }
  if (auto broadcastOp = dyn_cast<npuvector::BroadcastOp>(op)) {
    return refineBroadcast(broadcastOp);
  }
  if (auto transposeOp = dyn_cast<npuvector::TransposeOp>(op)) {
    return refineTranspose(transposeOp);
  }
  if (auto reductionOp = dyn_cast<npuvector::ReductionOp>(op)) {
    return refineReduction(reductionOp);
  }
  return refineShapePreservingElementwise(op);
}

class RefineNPUVectorStaticShape
    : public mlir::npuvector::impl::RefineNPUVectorStaticShapeBase<RefineNPUVectorStaticShape> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, math::MathDialect, npuvector::NPUVectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    constexpr unsigned kMaxIterations = 16;
    for (unsigned iteration = 0; iteration < kMaxIterations; ++iteration) {
      bool changed = false;
      funcOp.walk([&](Operation *op) { changed = changed || refineOneOp(op); });
      if (!changed) {
        return;
      }
    }
    funcOp.emitError("refine-npuvector-static-shape: did not converge");
    signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRefineNPUVectorStaticShapePass() {
  return std::make_unique<RefineNPUVectorStaticShape>();
}

}  // namespace npuvector
}  // namespace mlir
