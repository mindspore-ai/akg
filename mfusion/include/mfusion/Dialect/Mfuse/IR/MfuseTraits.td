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

#ifndef MFUSE_DIALECT_MFUSE_TRAITS_TD
#define MFUSE_DIALECT_MFUSE_TRAITS_TD

include "mlir/IR/OpBase.td"

// Enforces symbolic-shape consistency: if any operand of an op carries a
// symbolic-shape encoding, every non-static ranked-tensor result must too.
// The C++ implementation lives in MfuseTraits.h / MfuseTraits.cc.
def VerifySymbolicShapeTrait : NativeOpTrait<"VerifySymbolicShapeTrait"> {
  let cppNamespace = "::mlir::mfuse";
}

#endif // MFUSE_DIALECT_MFUSE_TRAITS_TD
