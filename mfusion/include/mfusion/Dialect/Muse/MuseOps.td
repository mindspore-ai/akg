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

#endif // MUSE_DIALECT_MUSE_OPS_TD
