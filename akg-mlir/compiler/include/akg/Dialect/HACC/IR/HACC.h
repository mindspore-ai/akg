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

#ifndef AKG_DIALECT_HACC_IR_HACC_H
#define AKG_DIALECT_HACC_IR_HACC_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

//===----------------------------------------------------------------------===//
// HACC Dialect
//===----------------------------------------------------------------------===//

#include "akg/Dialect/HACC/IR/HACCEnums.h.inc"

#include "akg/Dialect/HACC/IR/HACCBaseDialect.h.inc"

// generated type declarations
#define GET_TYPEDEF_CLASSES
#include "akg/Dialect/HACC/IR/HACCTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "akg/Dialect/HACC/IR/HACCAttrs.h.inc"

#endif // AKG_DIALECT_HACC_IR_HACC_H
