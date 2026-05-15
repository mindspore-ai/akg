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

#include "mfusion/Dialect/Mfuse/IR/MfuseTraits.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/Support/SymbolAttrUtils.h"
#include "symengine/integer.h"

namespace mlir::mfuse::detail {

static mlir::LogicalResult verifyResultShapeConsistency(mlir::Operation *op) {
  for (mlir::Value result : op->getResults()) {
    auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (!rankedType) {
      continue;
    }
    auto symAttr = mlir::dyn_cast_or_null<mlir::mfuse::SymbolicShapeAttr>(
      SymbolAttrUtils::getSymbolicShapeAttrFromEncoding(rankedType));
    if (!symAttr) {
      continue;
    }

    auto symExprs = symAttr.getSymEngineExprs();
    auto shape = rankedType.getShape();

    if (static_cast<int64_t>(symExprs.size()) != rankedType.getRank()) {
      return op->emitOpError() << "symbolic shape rank (" << symExprs.size() << ") does not match tensor rank ("
                               << rankedType.getRank() << ")";
    }

    for (int64_t i = 0; i < rankedType.getRank(); ++i) {
      bool isDynamic = mlir::ShapedType::isDynamic(shape[i]);
      bool isInteger = SymEngine::is_a<SymEngine::Integer>(*symExprs[i]);

      if (!isDynamic) {
        if (!isInteger) {
          return op->emitOpError() << "dimension " << i << " is static (" << shape[i]
                                   << ") but symbolic expression is not a constant: " << symExprs[i]->__str__();
        }
        auto symVal = static_cast<int64_t>(SymEngine::down_cast<const SymEngine::Integer &>(*symExprs[i]).as_int());
        if (symVal != shape[i]) {
          return op->emitOpError() << "dimension " << i << " is static (" << shape[i]
                                   << ") but symbolic expression has different value: " << symVal;
        }
      } else {
        if (isInteger) {
          return op->emitOpError() << "dimension " << i
                                   << " is dynamic but symbolic expression is a constant: " << symExprs[i]->__str__();
        }
      }
    }
  }
  return mlir::success();
}

mlir::LogicalResult verifySymbolicShapeTrait(mlir::Operation *op) {
  if (mlir::failed(verifyResultShapeConsistency(op))) {
    return mlir::failure();
  }

  bool hasInputSymbol = false;
  for (mlir::Value operand : op->getOperands()) {
    if (SymbolAttrUtils::hasSymbolicShapeEncoding(operand.getType())) {
      hasInputSymbol = true;
      break;
    }
  }

  if (!hasInputSymbol) {
    return mlir::success();
  }

  for (mlir::Value operand : op->getOperands()) {
    auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(operand.getType());
    if (rankedType && !rankedType.hasStaticShape() && !SymbolAttrUtils::hasSymbolicShapeEncoding(rankedType)) {
      return op->emitOpError() << "failed symbolic shape verification: because at least one "
                                  "input has a symbolic shape, all non-static ranked tensor "
                                  "inputs must also have a symbolic shape.";
    }
  }

  for (mlir::Value result : op->getResults()) {
    auto rankedType = llvm::dyn_cast<mlir::RankedTensorType>(result.getType());
    if (rankedType && !rankedType.hasStaticShape() && !SymbolAttrUtils::hasSymbolicShapeEncoding(rankedType)) {
      return op->emitOpError() << "failed symbolic shape verification: because at least one "
                                  "input has a symbolic shape, all non-static ranked tensor "
                                  "results must also have a symbolic shape.";
    }
  }

  return mlir::success();
}

}  // namespace mlir::mfuse::detail
