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

#include "akg/Dialect/EmitPy/IR/EmitPyAttrs.h"

#include "akg/Dialect/EmitPy/IR/EmitPy.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::emitpy;  // NOLINT(build/namespaces)

#define GET_ATTRDEF_CLASSES
#include "akg/Dialect/EmitPy/IR/EmitPyAttrs.cpp.inc"

void EmitPyDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "akg/Dialect/EmitPy/IR/EmitPyAttrs.cpp.inc"  // NOLINT(build/include)
      >();
}
