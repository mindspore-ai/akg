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

#ifndef MFUSE_DIALECT_MFUSE_TYPES_TD
#define MFUSE_DIALECT_MFUSE_TYPES_TD

include "mfusion/Dialect/Mfuse/IR/MfuseDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Supported argument/result types for operations:
//   List            --> Mfuse_ListType
//   String          --> Mfuse_StringType (replaces StrAttr)
//   Dtype           --> Mfuse_DtypeType (wraps element type)
//   None            --> Mfuse_NoneType
//   Scalar values   --> Rank-0 tensor (AnyScalar constraint)
//===----------------------------------------------------------------------===//

// This type represents a list/collection of elements and is distinct from tensor types.
// Use this type as an operand to pass list values to operations.
def Mfuse_ListType : Mfuse_TypeWithContainedType<"List", "list"> {
  let summary = "List type for MFUSE dialect";
  let description = [{
    Represents a list of elements. This type is used as an operand
    to pass list values to operations.
  }];
}

// Helper to create list type constraints with specific element types
class Mfuse_List<Type containedType> : Type<
    And<[
      CPred<"::llvm::isa<::mlir::mfuse::ListType>($_self)">,
      SubstLeaves<"$_self", "::llvm::cast<::mlir::mfuse::ListType>($_self).getContainedType()", containedType.predicate>
    ]>,
    "List of " # containedType.summary,
    "::mlir::mfuse::ListType">;

// TensorList constraint - list containing RankedTensor elements
// This is commonly used for operations that take multiple tensors (e.g., grouped_matmul)
def MfuseTensorList : Mfuse_List<AnyRankedTensor>;

//===----------------------------------------------------------------------===//
// Device Attribute
//===----------------------------------------------------------------------===//

def Mfuse_DeviceAttr : AttrDef<Mfuse_Dialect, "Device"> {
  let mnemonic = "device";
  let summary = "Device attribute for MFUSE dialect";
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

// String type (replaces StrAttr)
// This type wraps a string value and is distinct from tensor types.
// Use this type as an operand to pass string values to operations.
def Mfuse_StringType : Mfuse_Type<"String", "string"> {
  let summary = "String type for MFUSE dialect";
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

// Dtype type (stores mlir::Type directly)
// This type wraps a MLIR element type and is distinct from tensor types.
// Use this type as an operand to pass data type values to operations like Cast.
def Mfuse_DtypeType : Mfuse_Type<"Dtype", "dtype"> {
  let summary = "Dtype type for MFUSE dialect";
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

// Use rank-0 tensor to represent scalar in Mfuse dialect.
def AnyScalar : 0DTensorOf<[AnyType]>;

// None type (represents null/none values)
// This type wraps a None value and is distinct from tensor types.
// Use this type as an operand to pass None/null values to operations.
def Mfuse_NoneType : Mfuse_Type<"None", "none"> {
  let summary = "None type for MFUSE dialect";
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

// MFUSE-specific optional constraint that accepts the base type or Mfuse_NoneType.
class MfuseOptional<Type type>
    : AnyTypeOf<[type, Mfuse_NoneType]> {
  Type baseType = type;
}

// OptionalTensorList - TensorList or None
def MfuseOptionalTensorList : MfuseOptional<MfuseTensorList>;

#endif // MFUSE_DIALECT_MFUSE_TYPES_TD
