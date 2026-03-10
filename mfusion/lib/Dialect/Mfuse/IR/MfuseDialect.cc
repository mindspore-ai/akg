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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SMLoc.h"

#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"
#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.h"

#include "mfusion/Dialect/Mfuse/IR/MfuseDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mfusion/Dialect/Mfuse/IR/Mfuse.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mfusion/Dialect/Mfuse/IR/MfuseTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mfusion/Dialect/Mfuse/IR/MfuseAttributes.cpp.inc"

void mlir::mfuse::MfuseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mfusion/Dialect/Mfuse/IR/Mfuse.cpp.inc"  // NOLINT(build/include)
    >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mfusion/Dialect/Mfuse/IR/MfuseTypes.cpp.inc"  // NOLINT(build/include)
    >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mfusion/Dialect/Mfuse/IR/MfuseAttributes.cpp.inc"  // NOLINT(build/include)
    >();
}
