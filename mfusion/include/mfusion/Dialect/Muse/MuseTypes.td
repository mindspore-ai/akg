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

#ifndef MUSE_DIALECT_MUSE_TYPES_TD
#define MUSE_DIALECT_MUSE_TYPES_TD

include "mfusion/Dialect/Muse/MuseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Supported argument/result types for operations:
//   Tensor          --> Muse_TensorType
//   Tensor List     --> MuseTensorList
//   Integer         --> Muse_I64Type (replaces I64Attr)
//   Integer Array   --> Muse_I64ArrayType (replaces I64ArrayAttr, for axis, strides, shape, etc.)
//   Boolean         --> Muse_BooleanType (replaces BoolAttr)
//   Boolean Array   --> Muse_BooleanArrayType (replaces BoolArrayAttr)
//   Float           --> Muse_F64Type (replaces F64Attr)
//   Float Array     --> Muse_F64ArrayType (replaces F64ArrayAttr)
//   String          --> Muse_StringType (replaces StrAttr)
//===----------------------------------------------------------------------===//

// This type represents a list/collection of elements and is distinct from tensor types.
// Use this type as an operand to pass list values to operations.
def Muse_ListType : Muse_TypeWithContainedType<"List", "list"> {
  let summary = "List type for MUSE dialect";
  let description = [{
    Represents a list of elements. This type is used as an operand
    to pass list values to operations.
  }];
}

// List type constraint (accepts any list type)
def MuseAnyList : Type<CPred<"::llvm::isa<::mlir::muse::ListType>($_self)">,
                      "MUSE list type",
                      "::mlir::muse::ListType">;

// Helper to create list type constraints with specific element types
class Muse_List<Type containedType> : Type<
    And<[
      CPred<"::llvm::isa<::mlir::muse::ListType>($_self)">,
      SubstLeaves<"$_self", "::llvm::cast<::mlir::muse::ListType>($_self).getContainedType()", containedType.predicate>
    ]>,
    "List of " # containedType.summary,
    "::mlir::muse::ListType">;

//===----------------------------------------------------------------------===//
// Device Attribute
//===----------------------------------------------------------------------===//

def Muse_DeviceAttr : AttrDef<Muse_Dialect, "Device"> {
  let mnemonic = "device";
  let summary = "Device attribute for MUSE dialect";
  let description = [{
    Represents a device with type (e.g., "cpu", "npu") and index (signed integer).
    The index is a signed integer indicating the device index.
  }];

  let parameters = (ins
    "mlir::StringAttr":$device_type,
    "mlir::IntegerAttr":$index
  );

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Tensor Types
//===----------------------------------------------------------------------===//

def Muse_TensorType : Muse_Type<"Tensor", "tensor", [ShapedTypeInterface]> {
  let summary = "Tensor type for MUSE dialect";
  let description = [{
    Represents a multi-dimensional array with shape, element type, and optional device information.
    Device information includes device type (e.g., "cpu", "npu") and device index.
    Format: muse.tensor<shape x elementType, device=...> (e.g., muse.tensor<2x3xf32, device=<npu, 0>>)
  }];

  let parameters = (ins
    OptionalArrayRefParameter<"int64_t">:$shape,
    "mlir::Type":$elementType,
    OptionalParameter<"mlir::muse::DeviceAttr">:$device
  );

  let extraClassDeclaration = [{
    bool hasRank() const;
    mlir::ShapedType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const;
  }];

  let hasCustomAssemblyFormat = 1;
}


// Tensor type constraint for MUSE dialect operations
def MuseAnyTensor : Type<CPred<"::llvm::isa<::mlir::muse::TensorType>($_self)">,
                        "MUSE tensor type",
                        "::mlir::muse::TensorType">;

//===----------------------------------------------------------------------===//
// Scalar Types
//===----------------------------------------------------------------------===//

// 64-bit floating point scalar type (replaces F64Attr)
// This type wraps a 64-bit floating point value and is distinct from tensor types.
// Use this type as an operand to pass scalar double values to operations.
def Muse_F64Type : Muse_Type<"F64", "f64"> {
  let summary = "64-bit floating point scalar type for MUSE dialect";
  let description = [{
    Represents a 64-bit floating point scalar value. This type is used as an operand
    to replace F64Attr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    floating point values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// 64-bit integer scalar type (replaces I64Attr)
// This type wraps a 64-bit integer value and is distinct from tensor types.
// Use this type as an operand to pass scalar integer values to operations.
def Muse_I64Type : Muse_Type<"I64", "i64"> {
  let summary = "64-bit integer scalar type for MUSE dialect";
  let description = [{
    Represents a 64-bit integer scalar value. This type is used as an operand
    to replace I64Attr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    integer values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// Boolean scalar type (replaces BoolAttr)
// This type wraps a boolean value and is distinct from tensor types.
// Use this type as an operand to pass scalar boolean values to operations.
def Muse_BooleanType : Muse_Type<"Boolean", "boolean"> {
  let summary = "Boolean scalar type for MUSE dialect";
  let description = [{
    Represents a boolean scalar value. This type is used as an operand
    to replace BoolAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass scalar
    boolean values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// String type (replaces StrAttr)
// This type wraps a string value and is distinct from tensor types.
// Use this type as an operand to pass string values to operations.
def Muse_StringType : Muse_Type<"String", "string"> {
  let summary = "String type for MUSE dialect";
  let description = [{
    Represents a string value. This type is used as an operand
    to replace StrAttr, allowing all parameters to be passed as operands.
    This type is distinct from tensor types and can be used to pass string
    values to operations.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

//===----------------------------------------------------------------------===//
// Array Types
//===----------------------------------------------------------------------===//

// Tensor list type for Muse dialect operations
def MuseTensorList : Muse_List<MuseAnyTensor>;
def Muse_I64ArrayType : Muse_List<Muse_I64Type>;
def Muse_F64ArrayType : Muse_List<Muse_F64Type>;
def Muse_BooleanArrayType : Muse_List<Muse_BooleanType>;

// Dtype type (stores mlir::Type directly)
// This type wraps a MLIR element type and is distinct from tensor types.
// Use this type as an operand to pass data type values to operations like Cast.
def Muse_DtypeType : Muse_Type<"Dtype", "dtype"> {
  let summary = "Dtype type for MUSE dialect";
  let description = [{
    Represents a data type value by storing a MLIR element type directly.
    This type is used as an operand to pass data type values to operations like Cast.
    The element type (e.g., f32, i64) is stored directly, which can be converted
    to DataType enum when needed.
  }];

  let parameters = (ins
    "mlir::Type":$elementType
  );

  let assemblyFormat = [{
    `<` $elementType `>`
  }];
}

// Scalar type constraint (accepts any scalar type)
// This is a type constraint, not a concrete type. It accepts f64, i64, or boolean.
// Use this constraint for operation arguments that can accept any scalar type.
// For example, it can be used for the alpha parameter in Add operation.
def Muse_ScalarType : AnyTypeOf<[
  Muse_F64Type,
  Muse_I64Type,
  Muse_BooleanType
], "scalar type (f64/i64/boolean)">;

// None type (represents null/none values)
// This type wraps a None value and is distinct from tensor types.
// Use this type as an operand to pass None/null values to operations.
def Muse_NoneType : Muse_Type<"None", "none"> {
  let summary = "None type for MUSE dialect";
  let description = [{
    Represents a None/null value. This type is used as an operand
    to pass None values to operations. This type is distinct from tensor types
    and can be used to represent optional or null values.
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins), [{
      return Base::get(context);
    }]>
  ];

  let assemblyFormat = "";
}

// MUSE-specific optional constraint that accepts the base type or Muse_NoneType.
class MuseOptional<Type type>
    : AnyTypeOf<[type, Muse_NoneType]> {
  Type baseType = type;
}

#endif // MUSE_DIALECT_MUSE_TYPES_TD
