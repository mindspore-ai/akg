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

#ifndef MFUSION_DIALECT_DVM_DVMDIALECT_TD
#define MFUSION_DIALECT_DVM_DVMDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// DVM Dialect Definition
//===----------------------------------------------------------------------===//

def Dvm_Dialect : Dialect {
  let name = "dvm";
  let summary = "DVM dialect for Ascend NPU";
  let cppNamespace = "::mlir::dvm";

  let description = [{
    DVM dialect provides primitive operations that map 1:1 to DVM C++ API.
    For Scheme D (OpCode payload), the DVM IR is serialized at compile-time and
    executed at runtime without MLIR dependency. Therefore, DVM ops should be
    designed to map as directly as possible to `inferrt/src/ops/ascend/dvm/prebuild/dvm.h`
    (dvm::Kernel methods + enums), so that JSON/bytecode payload can be encoded
    and decoded with minimal logic.

    DVM dialect uses MLIR builtin types (tensor, f32, etc.) to minimize dependencies.

    Design principles:
    1. Only define primitive operations (element-wise, binary, reduce, matmul, etc.)
    2. No predefined fusion operations - fusion happens by emitting ops into the same `dvm::Kernel`
    3. Operations correspond 1:1 with dvm.h API (Kernel methods + enum op types)
    4. Keep op attributes close to dvm.h signatures (e.g. reduce dims/keepdims, matmul transpose flags)
  }];
}

//===----------------------------------------------------------------------===//
// Base DVM Operation
//===----------------------------------------------------------------------===//

class Dvm_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dvm_Dialect, mnemonic, traits>;

#endif // MFUSION_DIALECT_DVM_DVMDIALECT_TD
