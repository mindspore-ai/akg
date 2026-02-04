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

#ifndef MUSE_DIALECT_MUSE_OPS_TD
#define MUSE_DIALECT_MUSE_OPS_TD

include "mfusion/Dialect/Muse/MuseConstantOps.td"
include "mfusion/Dialect/Muse/MuseMetaOps.td"
include "mfusion/Dialect/Muse/MuseAclnnOps.td"

def Muse_CustomCallOp : Muse_Op<"custom_call"> {
  let summary = "Invoke a custom operator";
  let description = [{
    Invokes a custom operator by name.
    The op_name operand contains the name of the custom operator to invoke.
  }];

  let arguments = (ins
    Muse_StringType:$op_name,
    Variadic<AnyType>:$inputs
  );

  let results = (outs
    Variadic<AnyType>:$outputs
  );

  let assemblyFormat = [{
    $op_name `,` $inputs attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// GraphKernel Fusion Operations
//===----------------------------------------------------------------------===//

def Muse_FusedOp : Muse_Op<"fused", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "GraphKernel fusion container operation";
  let description = [{
    The `muse.fused` operation represents a fused subgraph created by
    GraphKernel clustering passes. It contains a single block region
    with the fused operations.

    The `fusion_type` attribute indicates which kernel generator to use:
    - "dvm": Use DVM kernel generator
    - "akg": Use AKG kernel generator

    Example:
    ```mlir
    %result = muse.fused(%arg0, %arg1) {fusion_type = "dvm"} :
        (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
      ^bb0(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>):
        %0 = muse.add %a, %b, %alpha : ...
        muse.yield %0 : tensor<4x4xf32>
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
    $inputs `,` attr-dict `:` functional-type($inputs, $results) $body
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

def Muse_YieldOp : Muse_Op<"yield", [
    Pure,
    Terminator,
    ParentOneOf<["FusedOp"]>
  ]> {
  let summary = "Yield values from a fused region";
  let description = [{
    The `muse.yield` operation terminates the region of a `muse.fused`
    operation and returns the values to be used as the results of the
    parent operation.

    Example:
    ```mlir
    muse.yield %0, %1 : tensor<4x4xf32>, tensor<4x4xf32>
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

#endif // MUSE_DIALECT_MUSE_OPS_TD
