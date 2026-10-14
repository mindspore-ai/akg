// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-FileCopyrightText: (c) 2026 Huawei Technologies Co., Ltd
//
// SPDX-License-Identifier: Apache-2.0
//
// This file was copied from project tenstorrent/tt-mlir
// (https://github.com/tenstorrent/tt-mlir) and modified for AKG-MLIR.

#ifndef AKG_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD
#define AKG_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EmitPy interfaces definitions.
//===----------------------------------------------------------------------===//

def PyExpressionInterface : OpInterface<"PyExpressionInterface"> {
  let description = [{
    Interface to mark operations that can be part of the PyExpression.
  }];

  let cppNamespace = "::mlir::emitpy";
  let methods = [
    InterfaceMethod<[{
      Check whether operation has side effects that may affect the expression
      evaluation.

      By default operation is marked as having side effects.

      ```c++
      class ConcreteOp ... {
      public:
        bool hasSideEffects() {
          // This way we can override the default implementation.
          return false;
        }
      };
      ```
    }],
      "bool", "hasSideEffects", (ins), /*methodBody=*/[{}],
       /*defaultImplementation=*/[{
        // Check if the operation has memory effects
        // Operations without memory effects are considered pure
        return !::mlir::isMemoryEffectFree($_op);
    }]>,
  ];
}

#endif // AKG_DIALECT_EMITPY_IR_EMITPYINTERFACES_TD
