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

#ifndef MFUSION_DIALECT_MFUSE_MFUSEOPS_H
#define MFUSION_DIALECT_MFUSE_MFUSEOPS_H

#include <string>
#include <type_traits>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::mfuse {

// Forward declarations for operations
class FusedOp;
class YieldOp;

bool shouldInferSymbolicShape(mlir::ValueRange operands, mlir::RankedTensorType resultType);

template <typename ConcreteOp>
struct SymbolicBuilderHelper {
  static void build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Type resultType,
                    mlir::ValueRange operands, llvm::ArrayRef<mlir::NamedAttribute> attributes) {
    // Populate state first so inferSymbolicShapes can inspect it (e.g. attrs).
    state.addOperands(operands);
    state.addAttributes(attributes);

    auto finalType = resultType;

    // In Torch->Mfuse conversion, Mfuse ops are first built without symshape; a
    // later pass attaches symbolic encodings uniformly. Only attempt inference
    // when inputs already carry symbolic shape.
    auto rankedResult = resultType.dyn_cast<mlir::RankedTensorType>();
    if (rankedResult && shouldInferSymbolicShape(operands, rankedResult)) {
      auto typeOrError = ConcreteOp::inferSymbolicShapes(builder, state, resultType);
      if (mlir::succeeded(typeOrError)) {
        finalType = *typeOrError;
      }
    }

    state.addTypes(finalType);
  }
};

}  // namespace mlir::mfuse

#endif  // MFUSION_DIALECT_MFUSE_MFUSEOPS_H
