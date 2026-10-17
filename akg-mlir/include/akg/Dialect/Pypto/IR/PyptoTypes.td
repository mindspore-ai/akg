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

#ifndef AKG_DIALECT_PYPTO_IR_PYPTOTYPES_TD
#define AKG_DIALECT_PYPTO_IR_PYPTOTYPES_TD

include "akg/Dialect/Pypto/IR/PyptoBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Pypto_TileType
//===----------------------------------------------------------------------===//

def Pypto_TileType : TypeDef<Pypto_Dialect, "Tile"> {
  let mnemonic = "tile";
  let summary = "A tile of elements resident in core-local PyPTO memory";
  let description = [{
    The `!pypto.tile` type represents a rectangular block of a PyPTO tensor
    held in core-local tile memory. It carries the element type and the tile
    shape explicitly, so the tile-model invariants (shape known to the PyPTO
    runtime, element type mappable to a PyPTO dtype) are checked by the IR
    verifier instead of by stringly-typed conventions downstream.

    Syntax:

    ```
    tile-type ::= `!pypto.tile` `<` type `,` `[` shape-list `]` `>`
    shape-list ::= (int-literal (`,` int-literal)*)?
    ```

    #### Examples:

    ```mlir
    !pypto.tile<f32, [4, 8]>
    !pypto.tile<f16, [128]>
    !pypto.tile<i32, []>   // rank-0 tile
    ```
  }];

  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "Type":$elementType, "ArrayRef<int64_t>":$shape
    ), [{
      return $_get(elementType.getContext(), elementType, shape);
    }]>
  ];

  let hasCustomAssemblyFormat = 1;
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    /// Returns true if the given type can be used as the element type of a
    /// tile. Tiles consist of integer, index, or float primitives.
    static bool isValidElementType(Type t) {
      return ::llvm::isa<IntegerType, IndexType, FloatType>(t);
    }

    /// Returns the rank of this tile.
    unsigned getRank() const { return getShape().size(); }

    /// Returns the total number of elements, or -1 when any dimension is
    /// dynamic.
    int64_t getNumElements() const {
      int64_t num = 1;
      for (int64_t dim : getShape()) {
        if (ShapedType::isDynamic(dim)) {
          return -1;
        }
        num *= dim;
      }
      return num;
    }
  }];
}

#endif // AKG_DIALECT_PYPTO_IR_PYPTOTYPES_TD
