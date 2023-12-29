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

#ifndef FUSION_BASE
#define FUSION_BASE

include "mlir/IR/OpBase.td"

def Fusion_Dialect : Dialect {
  let name = "fusion";
  let cppNamespace = "::mlir::fusion";
  let description = [{The `fusion` dialect is intended to mark fusion
                          information and help op fusion.}];

  let dependentDialects = ["memref::MemRefDialect", "vector::VectorDialect"];

  let extraClassDeclaration = [{
    static StringRef getFusionKindAttrName() { return "fusion.kind"; }
    static bool isIdentityFusion(StringRef kind) { return kind == "identity"; }
    static bool isIdentityWithBroadPermFusion(StringRef kind) {
      return kind == "identityWithBroadPerm";
    }
    static bool isInvertibleFusion(StringRef kind) {
      return kind == "invertible";
    }

    static StringRef getFusionFlagAttrName() { return "fusion.flag"; }
    static bool isFusionableOp(StringRef flag) {
      return flag != "unfusionable";
    }
  }];

  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // FUSION_BASE
