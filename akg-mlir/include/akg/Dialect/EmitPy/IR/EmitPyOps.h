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

#ifndef AKG_DIALECT_EMITPY_IR_EMITPYOPS_H
#define AKG_DIALECT_EMITPY_IR_EMITPYOPS_H

#include "akg/Dialect/EmitPy/IR/EmitPy.h"
#include "akg/Dialect/EmitPy/IR/EmitPyAttrs.h"
#include "akg/Dialect/EmitPy/IR/EmitPyTypes.h"
#include "akg/Dialect/EmitPy/Interfaces/EmitPyInterfaces.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#define GET_OP_CLASSES
#include "akg/Dialect/EmitPy/IR/EmitPyOps.h.inc"

#endif  // AKG_DIALECT_EMITPY_IR_EMITPYOPS_H
