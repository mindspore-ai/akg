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

#ifndef MUSE_DIALECT_MUSE_CONSTANT_OPS_TD
#define MUSE_DIALECT_MUSE_CONSTANT_OPS_TD

include "mfusion/Dialect/Muse/MuseTypes.td"

//===----------------------------------------------------------------------===//
// Operations to create values of custom types from attributes
// These operations create initial values that can be modified at runtime.
//===----------------------------------------------------------------------===//

// Create a None value
def Muse_CreateNoneOp : Muse_Op<"constant.none", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create a None value";
  let description = [{
    Creates a value of Muse_NoneType.
  }];

  let arguments = (ins);
  let results = (outs Muse_NoneType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(NoneType::get($_builder.getContext()));
    }]>
  ];
}

// Create an i64 value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateI64Op : Muse_Op<"constant.i64", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create an i64 value from an attribute";
  let description = [{
    Creates a value of Muse_I64Type from an integer attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins I64Attr:$value);
  let results = (outs Muse_I64Type:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "int64_t":$value), [{
      $_state.addAttribute("value", $_builder.getI64IntegerAttr(value));
      $_state.addTypes(I64Type::get($_builder.getContext()));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a boolean value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateBooleanOp : Muse_Op<"constant.boolean", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create a boolean value from an attribute";
  let description = [{
    Creates a value of Muse_BooleanType from a boolean attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs Muse_BooleanType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "bool":$value), [{
      $_state.addAttribute("value", $_builder.getBoolAttr(value));
      $_state.addTypes(BooleanType::get($_builder.getContext()));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a f64 value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateF64Op : Muse_Op<"constant.f64", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create a f64 value from an attribute";
  let description = [{
    Creates a value of Muse_F64Type from a double attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins F64Attr:$value);
  let results = (outs Muse_F64Type:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "double":$value), [{
      $_state.addAttribute("value", $_builder.getF64FloatAttr(value));
      $_state.addTypes(F64Type::get($_builder.getContext()));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create an i64 array value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateI64ArrayOp : Muse_Op<"constant.i64_array", [ConstantLike, Pure]> {
  let summary = "Create an i64 array value from an attribute";
  let description = [{
    Creates a value of Muse_I64ArrayType from an integer array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins I64ArrayAttr:$value);
  let results = (outs Muse_I64ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::llvm::ArrayRef<int64_t>":$value), [{
      $_state.addAttribute("value", $_builder.getI64ArrayAttr(value));
      $_state.addTypes(ListType::get($_builder.getContext(), I64Type::get($_builder.getContext())));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a f64 array value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateF64ArrayOp : Muse_Op<"constant.f64_array", [ConstantLike, Pure]> {
  let summary = "Create a f64 array value from an attribute";
  let description = [{
    Creates a value of Muse_F64ArrayType from a double array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins F64ArrayAttr:$value);
  let results = (outs Muse_F64ArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::llvm::ArrayRef<double>":$value), [{
      $_state.addAttribute("value", $_builder.getF64ArrayAttr(value));
      $_state.addTypes(ListType::get($_builder.getContext(), F64Type::get($_builder.getContext())));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a boolean array value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateBooleanArrayOp : Muse_Op<"constant.boolean_array", [ConstantLike, Pure]> {
  let summary = "Create a boolean array value from an attribute";
  let description = [{
    Creates a value of Muse_BooleanArrayType from a boolean array attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins BoolArrayAttr:$value);
  let results = (outs Muse_BooleanArrayType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::llvm::ArrayRef<bool>":$value), [{
      $_state.addAttribute("value", $_builder.getBoolArrayAttr(value));
      $_state.addTypes(ListType::get($_builder.getContext(), BooleanType::get($_builder.getContext())));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a string value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateStringOp : Muse_Op<"constant.string", [ConstantLike, Pure]> {
  let summary = "Create a string value from an attribute";
  let description = [{
    Creates a value of Muse_StringType from a string attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Muse_StringType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::llvm::StringRef":$value), [{
      $_state.addAttribute("value", $_builder.getStringAttr(value));
      $_state.addTypes(StringType::get($_builder.getContext()));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

// Create a dtype value from an attribute
// The created value can be modified at runtime through other operations.
def Muse_CreateDtypeOp : Muse_Op<"constant.dtype", [ConstantLike, Pure]> {
  let summary = "Create a dtype value from a type attribute";
  let description = [{
    Creates a value of Muse_DtypeType from a type attribute.
    The type attribute represents an element type (e.g., f32, i64) which is
    stored directly in the Muse_DtypeType and can be converted to DataType enum when needed.
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs Muse_DtypeType:$result);

  let assemblyFormat = "$value attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins "::mlir::Type":$value), [{
      $_state.addAttribute("value", mlir::TypeAttr::get(value));
      $_state.addTypes(DtypeType::get($_builder.getContext(), value));
    }]>
  ];
  let extraClassDeclaration = [{
    ::mlir::OpFoldResult fold(FoldAdaptor adaptor) {
      return getValueAttr();
    }
  }];
}

#endif // MUSE_DIALECT_MUSE_CONSTANT_OPS_TD
