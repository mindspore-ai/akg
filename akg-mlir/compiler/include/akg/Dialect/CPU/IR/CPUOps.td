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

#ifndef AKG_DIALECT_CPU_IR_CPU_OPS
#define AKG_DIALECT_CPU_IR_CPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "akg/Dialect/CPU/IR/CPUBase.td"

//===----------------------------------------------------------------------===//
// CPU Operator.
//===----------------------------------------------------------------------===//
// class CPU_Op for the operation in this dialect
class CPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<CPU_Dialect, mnemonic, traits>;


//===----------------------------------------------------------------------===//
// Operator: Parallel_Launch
//===----------------------------------------------------------------------===//
def ParallelLaunchOp : CPU_Op<"ParallelLaunch"> {
  let summary = "CPU parallel launch operator";
  let description = [{
    A parallel launch ops for CPU backend.
  }];
  let arguments = (ins Variadic<AnyType>:$args, 
    FlatSymbolRefAttr:$callee);
  let results = (outs Variadic<AnyType>);
}
#endif // AKG_DIALECT_CPU_IR_CPU_OPS