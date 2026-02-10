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

#ifndef MFUSE_DIALECT_MFUSE_CONSTANT_OPS_TD
#define MFUSE_DIALECT_MFUSE_CONSTANT_OPS_TD

include "mfusion/Dialect/Mfuse/MfuseTypes.td"

//===----------------------------------------------------------------------===//
// Operations to create values of custom types from attributes
// These operations create initial values that can be modified at runtime.
//===----------------------------------------------------------------------===//

// Create a None value
def Mfuse_CreateNoneOp : Mfuse_Op<"constant.none", [
    ConstantLike,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Create a None value";
  let description = [{
    Creates a value of Mfuse_NoneType.
  }];

  let arguments = (ins);
  let results = (outs Mfuse_NoneType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let builders = [
    OpBuilder<(ins), [{
      $_state.addTypes(NoneType::get($_builder.getContext()));
    }]>
  ];
}

// Create a string value from an attribute
// The created value can be modified at runtime through other operations.
def Mfuse_CreateStringOp : Mfuse_Op<"constant.string", [ConstantLike, Pure]> {
  let summary = "Create a string value from an attribute";
  let description = [{
    Creates a value of Mfuse_StringType from a string attribute.
    The created value can be modified at runtime through other operations.
  }];

  let arguments = (ins StrAttr:$value);
  let results = (outs Mfuse_StringType:$result);

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
def Mfuse_CreateDtypeOp : Mfuse_Op<"constant.dtype", [ConstantLike, Pure]> {
  let summary = "Create a dtype value from a type attribute";
  let description = [{
    Creates a value of Mfuse_DtypeType from a type attribute.
    The type attribute represents an element type (e.g., f32, i64) which is
    stored directly in the Mfuse_DtypeType and can be converted to DataType enum when needed.
  }];

  let arguments = (ins TypeAttr:$value);
  let results = (outs Mfuse_DtypeType:$result);

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

#endif // MFUSE_DIALECT_MFUSE_CONSTANT_OPS_TD
