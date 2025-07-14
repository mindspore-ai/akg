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
#include "akg/Dialect/HACC/IR/HACC.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::hacc;

void mlir::hacc::HACCDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "akg/Dialect/HACC/IR/HACCAttrs.cpp.inc"
    >();
}

#include "akg/Dialect/HACC/IR/HACCBaseDialect.cpp.inc"
#include "akg/Dialect/HACC/IR/HACCEnums.cpp.inc"

#ifndef GET_ATTRDEF_CLASSES
#define GET_ATTRDEF_CLASSES
#include "akg/Dialect/HACC/IR/HACCAttrs.cpp.inc"
#endif
