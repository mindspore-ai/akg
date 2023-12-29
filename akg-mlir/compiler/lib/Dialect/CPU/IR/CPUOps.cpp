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

#include "akg/Dialect/CPU/IR/CPUOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/CommonFolders.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

#include "akg/Dialect/CPU/IR/CPUOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::CPU;

void mlir::CPU::CPUDialect::initialize() {
  addOperations<
#ifndef GET_OP_LIST
#define GET_OP_LIST
#include "akg/Dialect/CPU/IR/CPUOps.cpp.inc"
#endif
    >();
}

#ifndef GET_OP_CLASSES
#define GET_OP_CLASSES
#include "akg/Dialect/CPU/IR/CPUOps.cpp.inc"
#endif
