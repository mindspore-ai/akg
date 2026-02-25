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

#ifndef MFUSE_DIALECT_MFUSE_OPS_TD
#define MFUSE_DIALECT_MFUSE_OPS_TD

include "mfusion/Dialect/Mfuse/MfuseConstantOps.td"
include "mfusion/Dialect/Mfuse/MfuseMetaOps.td"
include "mfusion/Dialect/Mfuse/MfuseAclnnOps.td"

def Mfuse_DvmCallOp : Mfuse_Op<"dvm_call"> {
  let summary = "Invoke a DVM fused subgraph";
  let description = [{
    Invokes a DVM fused subgraph.

    Attributes:
    - `subgraph_mlir`: The DVM dialect MLIR of the fused subgraph, used for
      backend DVM kernel generation.
    - `subgraph`: The function name of the fused subgraph in the current MLIR
      Module.
    - `is_dynamic`: Whether the fused subgraph has dynamic shape.
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    StrAttr:$subgraph_mlir,
    StrAttr:$subgraph,
    BoolAttr:$is_dynamic
  );

  let results = (outs
    Variadic<AnyRankedTensor>:$outputs
  );

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GraphKernel Fusion Operations
//===----------------------------------------------------------------------===//

def Mfuse_FusedOp : Mfuse_Op<"fused", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "GraphKernel fusion container operation";
  let description = [{
    The `mfuse.fused` operation represents a fused subgraph created by
    GraphKernel clustering passes. It contains a single block region
    with the fused operations.

    The `fusion_type` attribute indicates which kernel generator to use:
    - "dvm": Use DVM kernel generator
    - "akg": Use AKG kernel generator

    Example:
    ```mlir
    %result = mfuse.fused(%arg0, %arg1) {fusion_type = "dvm"} :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
      ^bb0(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>):
        %0 = mfuse.add %a, %b, %alpha : ...
        mfuse.yield %0 : tensor<4x4xf32>
    }
    ```
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs,
    OptionalAttr<StrAttr>:$fusion_type,
    OptionalAttr<StrAttr>:$kernel_name
  );

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);

  let hasVerifier = 1;

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, $results) $body
  }];

  let extraClassDeclaration = [{
    /// Returns true if this fusion uses DVM kernel generator.
    bool isDVMFusion() {
      auto type = getFusionType();
      return type && type.value() == "dvm";
    }

    /// Returns true if this fusion uses AKG kernel generator.
    bool isAKGFusion() {
      auto type = getFusionType();
      return type && type.value() == "akg";
    }

    /// Get the body block.
    Block &getBodyBlock() { return getBody().front(); }
  }];
}

def Mfuse_YieldOp : Mfuse_Op<"yield", [
    Pure,
    Terminator,
    ParentOneOf<["FusedOp"]>
  ]> {
  let summary = "Yield values from a fused region";
  let description = [{
    The `mfuse.yield` operation terminates the region of a `mfuse.fused`
    operation and returns the values to be used as the results of the
    parent operation.

    Example:
    ```mlir
    mfuse.yield %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$values);

  let assemblyFormat = [{
    ($values^ `:` type($values))? attr-dict
  }];

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, std::nullopt);
    }]>
  ];
}

#endif // MFUSE_DIALECT_MFUSE_OPS_TD
