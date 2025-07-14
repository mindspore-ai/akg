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

#ifndef AKG_DIALECT_HACC_IR_HACCATTRS_TD
#define AKG_DIALECT_HACC_IR_HACCATTRS_TD

include "akg/Dialect/HACC/IR/HACCBase.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

//===----------------------------------------------------------------------===//
// HACC Enums and Attributes
//===----------------------------------------------------------------------===//

class HACC_I32Enum<string name, string description, list<I32EnumAttrCase> cases>
    : I32EnumAttr<name, description, cases> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::hacc";
}

class HACC_Attr<string attrName, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HACC_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// Function Type
//===----------------------------------------------------------------------===//

def HACC_HOST : I32EnumAttrCase<"HOST", 1>;
def HACC_DEVICE : I32EnumAttrCase<"DEVICE", 2>;
def HACC_FuncTypeEnum : HACC_I32Enum<"HACCFuncType", "HACC Function Category", [
  HACC_HOST,
  HACC_DEVICE
]> {}

def HACC_FuncTypeAttr : HACC_Attr<"HACCFuncType", "function_kind"> {
  let parameters = (ins EnumParameter<HACC_FuncTypeEnum>:$function_kind);
  let assemblyFormat = "`<` params `>`";
  let description = [{
    HACC function type.
  }];
}

#endif // AKG_DIALECT_HACC_IR_HACCATTRS_TD
