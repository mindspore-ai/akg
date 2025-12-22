/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_NPUVECTOR_IR_NPUVECTOR_H_
#define COMPILER_INCLUDE_AKG_DIALECT_NPUVECTOR_IR_NPUVECTOR_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// NPUVector Dialect
//===----------------------------------------------------------------------===//
#include "akg/Dialect/NPUVector/IR/NPUVectorOpsDialect.h.inc"

//===----------------------------------------------------------------------===//
// NPUVector Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "akg/Dialect/NPUVector/IR/NPUVectorOpsTypes.h.inc"

//===----------------------------------------------------------------------===//
// NPUVector Dialect Operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.h.inc"

#endif  // COMPILER_INCLUDE_AKG_DIALECT_NPUVECTOR_IR_NPUVECTOR_H_
