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

#ifndef AKG_DIALECT_MATH_IR_MATH_EXT_BASE
#define AKG_DIALECT_MATH_IR_MATH_EXT_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Defines MathExt Dialect.
//===----------------------------------------------------------------------===//

def MathExt_Dialect : Dialect {
  let name = "mathExt";
  let summary = "Extension for MathExt dialect";
  let description = [{
    Extension for MathExt dialect
  }];
  let cppNamespace = "::mlir::mathExt";
}

#endif // AKG_DIALECT_MATH_IR_MATH_EXT_BASE
