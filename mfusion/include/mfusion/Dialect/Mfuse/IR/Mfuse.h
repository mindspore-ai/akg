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

#ifndef MFUSE_DIALECT_MFUSE_MFUSE_H
#define MFUSE_DIALECT_MFUSE_MFUSE_H

#include <vector>

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "symengine/basic.h"

#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"

#include "mfusion/Dialect/Mfuse/IR/MfuseAttributes.h"

#define GET_TYPEDEF_CLASSES
#include "mfusion/Dialect/Mfuse/IR/MfuseTypes.h.inc"

#include "mfusion/Dialect/Mfuse/IR/MfuseInterfaces.h.inc"

#include "mfusion/Dialect/Mfuse/IR/MfuseOps.h"

#define GET_OP_CLASSES
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h.inc"

#endif  // MFUSE_DIALECT_MFUSE_MFUSE_H
