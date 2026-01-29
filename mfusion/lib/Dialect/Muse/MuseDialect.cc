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

#include "mfusion/Dialect/Muse/Muse.h"
#include "mfusion/Dialect/Muse/MuseDialect.h"

#include "mfusion/Dialect/Muse/MuseDialect.cpp.inc"

#define GET_OP_CLASSES
#include "mfusion/Dialect/Muse/Muse.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "mfusion/Dialect/Muse/MuseTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "mfusion/Dialect/Muse/MuseAttributes.cpp.inc"

void mlir::muse::MuseDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mfusion/Dialect/Muse/Muse.cpp.inc"  // NOLINT(build/include)
    >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "mfusion/Dialect/Muse/MuseTypes.cpp.inc"  // NOLINT(build/include)
    >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "mfusion/Dialect/Muse/MuseAttributes.cpp.inc"  // NOLINT(build/include)
    >();
}
