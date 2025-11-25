/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef AKG_DIALECT_MINDSPORE_IR_MINDSPORE_TYPEBASE
#define AKG_DIALECT_MINDSPORE_IR_MINDSPORE_TYPEBASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"
include "akg/Dialect/MindSpore/IR/MindSporeBase.td"

//===----------------------------------------------------------------------===//
// Type defs
//===----------------------------------------------------------------------===//
class MindSpore_Type<string name, string typeMnemonic,
                 string baseCppClass = "::mlir::Type">
    : TypeDef<MindSpore_Dialect, name, [], baseCppClass> {
  let mnemonic = typeMnemonic;
}

def MindSpore_StringType : MindSpore_Type<"String", "str"> {
  let summary = "MindSpore StringType";
}

def MindSpore_NoneType: MindSpore_Type<"None", "none"> {
  let summary = "MindSpore NoneType";
}

//===----------------------------------------------------------------------===//
// Type predicates
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// MindSpore Tensor types
//===----------------------------------------------------------------------===//
def MindSpore_UInt8 : UI<8>;
def MindSpore_Int8 : I<8>;
def MindSpore_Int16 : I<16>;
def MindSpore_Int32 : I<32>;
def MindSpore_Int48 : I<48>;
def MindSpore_Int64 : I<64>;

def MindSpore_SignedInt : AnyTypeOf<[MindSpore_Int8,
                                    MindSpore_Int16,
                                    MindSpore_Int32,
                                    MindSpore_Int48,
                                    MindSpore_Int64]>;

def MindSpore_Bool : I<1>;

def MindSpore_Int : AnyTypeOf<[MindSpore_Bool,
                            MindSpore_UInt8,
                            MindSpore_SignedInt]>;

def MindSpore_Int32Or64 : AnyTypeOf<[MindSpore_Int32,
                                MindSpore_Int64]>;  

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//
def MindSpore_Float : AnyTypeOf<[F32, F16, BF16]>;

def MindSpore_IntOrFloat: AnyTypeOf<[MindSpore_Int, MindSpore_Float]>;

//===----------------------------------------------------------------------===//
// Multi-category types.
//===----------------------------------------------------------------------===//
def MindSpore_AnyNumber : AnyTypeOf<[MindSpore_Int, MindSpore_Float],
                               "number">;

//===----------------------------------------------------------------------===//
// Tensor types
//===----------------------------------------------------------------------===//

def MindSpore_Int32Tensor : TensorOf<[MindSpore_Int32]>;
def MindSpore_Int32Or64Tensor : TensorOf<[MindSpore_Int32Or64]>;

// Either ranked or unranked tensor of MindSpore supported element types.
def MindSpore_Tensor : TensorOf<[MindSpore_AnyNumber]>;
// Must be ranked but no further constraints
def MindSpore_RankedTensor : RankedTensorOf<[MindSpore_AnyNumber]>;

// We include unranked tensors as a supported type for all possible MindSpore
// Tensors as unranked does not guarantee invalid. If unranked tensors exist
// they should be shape propagate used MindSpore's shape inference pass and verified
// to not include any remaining unranked tensors.
def MindSpore_UnrankedTensor : UnrankedTensorOf<[MindSpore_AnyNumber]>;

def MindSpore_Tensor1D  : AnyTypeOf<[MindSpore_UnrankedTensor, 1DTensorOf<[MindSpore_AnyNumber]>]>;
def MindSpore_Tensor2D  : AnyTypeOf<[MindSpore_UnrankedTensor, 2DTensorOf<[MindSpore_AnyNumber]>]>;
def MindSpore_Tensor3D  : AnyTypeOf<[MindSpore_UnrankedTensor, 3DTensorOf<[MindSpore_AnyNumber]>]>;
def MindSpore_Tensor4D  : AnyTypeOf<[MindSpore_UnrankedTensor, 4DTensorOf<[MindSpore_AnyNumber]>]>;
def MindSpore_Tensor5D  : AnyTypeOf<[MindSpore_UnrankedTensor, TensorRankOf<[MindSpore_AnyNumber], [5]>]>;

//Ranked tensors up to given rank.
def MindSpore_Tensor1Dto4D : AnyTypeOf<[
    MindSpore_UnrankedTensor, TensorRankOf<[MindSpore_AnyNumber], [1, 2, 3, 4]>]>;

def MindSpore_Tensor1Dto6D : AnyTypeOf<[MindSpore_UnrankedTensor, TensorRankOf<[MindSpore_AnyNumber], [1, 2, 3, 4, 5, 6]>]>;

def MindSpore_Index1D : 1DTensorOf<[Index]>;
def MindSpore_Index2D : 2DTensorOf<[Index]>;

//===----------------------------------------------------------------------===//
// MindSpore Types
//===----------------------------------------------------------------------===//
def AnyMindSporeType : AnyTypeOf<[
  MindSpore_Tensor,
], "Any type that is legal to pass to a Torch kernel">;

#endif // AKG_DIALECT_MINDSPORE_IR_MINDSPORE_TYPEBASE
