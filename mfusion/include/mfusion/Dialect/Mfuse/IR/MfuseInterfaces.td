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

#ifndef MFUSE_DIALECT_MFUSE_INTERFACES_TD
#define MFUSE_DIALECT_MFUSE_INTERFACES_TD

include "mlir/IR/OpBase.td"

def Mfuse_InferSymbolicShapeInterface : OpInterface<"InferSymbolicShapeInterface"> {
  let cppNamespace = "::mlir::mfuse";
  let description = [{
    Operations implementing this interface can infer symbolic expressions for
    result dimensions based on the builder insertion point, builder state and
    the result type.
  }];

  let methods = [
    StaticInterfaceMethod<
      "Infer symbolic expressions for result dimensions",
      "mlir::FailureOr<mlir::Type>",
      "inferSymbolicShapes",
      (ins "mlir::OpBuilder &":$builder,
           "const mlir::OperationState &":$state,
           "mlir::Type":$resultType)
    >
  ];
}

#endif // MFUSE_DIALECT_MFUSE_INTERFACES_TD
