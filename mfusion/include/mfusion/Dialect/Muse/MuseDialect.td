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

#ifndef MUSE_DIALECT_MUSE_DIALECT_TD
#define MUSE_DIALECT_MUSE_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def Muse_Dialect : Dialect {
  let name = "muse";
  let cppNamespace = "::mlir::muse";

  let description = [{
    The MUSE (Machine Learning Runtime) dialect provides a set of operations and
    types for representing machine learning computations. This dialect is designed
    to be a high-level MLIR dialect that can be lowered to various targets.
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class Muse_Type<string name, string typeMnemonic, list<Trait> traits = []> :
    TypeDef<Muse_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Muse_TypeWithContainedType<string name, string typeMnemonic> : Muse_Type<name, typeMnemonic> {
  let parameters = (ins "::mlir::Type":$containedType);

  let builders = [
    TypeBuilderWithInferredContext<(ins "::mlir::Type":$containedType), [{
      return Base::get(containedType.getContext(), containedType);
    }]>
  ];

  let assemblyFormat = [{
    `<` $containedType `>`
  }];
}

// Markers for aclnn configuration dag structure
def AclnnConfig;
def IORef;

// Base class for all MUSE dialect operations.
class Muse_Op<string mnemonic, list<Trait> traits = []> :
    Op<Muse_Dialect, mnemonic, traits> {
  // Optional string for aclnn operation name.
  // When set, specifies the aclnn function name to use for code generation.
  string genAclnnOp = ?;

  // Optional dag for aclnn configuration with IORef list.
  // Structure: (AclnnConfig (IORef out0, in0), (IORef out1, in1), ...)
  // If not set, defaults to empty dag.
  dag genAclnnCfg = ?;
}

#endif // MUSE_DIALECT_MUSE_DIALECT_TD
