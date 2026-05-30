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

#ifndef NPUVECTOR_TYPES
#define NPUVECTOR_TYPES

include "akg/Dialect/NPUVector/IR/NPUVectorBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// NPUVectorType
//===----------------------------------------------------------------------===//

def NPUVector_Type : TypeDef<NPUVector_Dialect, "Vector",
    [ShapedTypeInterface, ValueSemantics]> {
  let cppClassName = "NPUVectorType";
  let mnemonic = "npuvector";
  let summary = "Multi-dimensional SIMD vector type for NPU with dynamic shape support";
  let description = [{
    The `npuvector.npuvector` type represents a SIMD style vector specifically
    designed for NPU (Neural Processing Unit) targets. Unlike the standard
    `vector` type, this type has enhanced support for dynamic dimensions,
    allowing dimensions to be specified as dynamic (using `?`) at compile time.

    Syntax:

    ```
    npu-vector-type ::= `npuvector.npuvector` `<` vector-dim-list vector-element-type `>`
    vector-dim-list := (static-dim-list `x`)?
    static-dim-list ::= static-dim (`x` static-dim)*
    static-dim ::= decimal-literal | `?`
    ```

    Each dimension can be either a static positive integer or `?` to indicate
    a dynamic dimension. Dynamic dimensions are resolved at runtime.

    #### Examples:

    ```mlir
    // A 2D vector with static dimensions
    npuvector.npuvector<3x42xf32>

    // A 2D vector with one dynamic dimension
    npuvector.npuvector<?x42xf32>

    // A 2D vector with all dynamic dimensions
    npuvector.npuvector<?x?xf32>

    // A 3D vector with mixed static and dynamic dimensions
    npuvector.npuvector<4x?x8xf32>

    // A 1D vector with dynamic dimension
    npuvector.npuvector<?xf32>
    ```
  }];

  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "Type":$elementType
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "ArrayRef<int64_t>":$shape, "Type":$elementType
    ), [{
      return $_get(elementType.getContext(), shape, elementType);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Returns true if the given type can be used as an element of a NPUVector
    /// type. NPUVectors can consist of integer, index, or float primitives.
    static bool isValidElementType(Type t) {
      return ::llvm::isa<IntegerType, IndexType, FloatType>(t);
    }

    /// Returns if this type is ranked (always true for NPUVector).
    bool hasRank() const { return true; }

    /// Returns true if this type has any dynamic dimensions.
    bool hasDynamicShape() const {
      return llvm::any_of(getShape(), [](int64_t dim) {
        return ShapedType::isDynamic(dim);
      });
    }

    /// Returns the number of dynamic dimensions.
    unsigned getNumDynamicDims() const {
      unsigned count = 0;
      for (int64_t dim : getShape()) {
        if (ShapedType::isDynamic(dim))
          ++count;
      }
      return count;
    }

    /// Clone this NPUVector type with the given shape and element type.
    /// If the provided shape is `std::nullopt`, the current shape is used.
    NPUVectorType cloneWith(std::optional<ArrayRef<int64_t>> shape,
                            Type elementType) const;

    /// Return a clone of this type with the given new shape and element type.
    NPUVectorType clone(ArrayRef<int64_t> shape, Type elementType) const {
      return cloneWith(shape, elementType);
    }

    /// Return a clone of this type with the given new shape.
    NPUVectorType clone(ArrayRef<int64_t> shape) const {
      return cloneWith(shape, getElementType());
    }
  }];

  let hasCustomAssemblyFormat = 1;

  let genVerifyDecl = 1;
  let skipDefaultBuilders = 1;
}

#endif // NPUVECTOR_TYPES

