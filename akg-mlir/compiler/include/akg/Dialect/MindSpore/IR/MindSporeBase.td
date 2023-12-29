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

#ifndef AKG_DIALECT_MINDSPORE_IR_MINDSPORE_BASE
#define AKG_DIALECT_MINDSPORE_IR_MINDSPORE_BASE

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Defines MindSpore Dialect.
//===----------------------------------------------------------------------===//

def MindSpore_Dialect : Dialect {
  let name = "mindspore";
  let summary = "MindSpore dialect";
  let description = [{
    MindSpore dialect
  }];
  let cppNamespace = "mlir::mindspore";

  let dependentDialects = ["tensor::TensorDialect"];

  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif // AKG_DIALECT_MINDSPORE_IR_MINDSPORE_BASE
