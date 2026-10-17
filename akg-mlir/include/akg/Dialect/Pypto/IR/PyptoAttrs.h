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

#ifndef AKG_DIALECT_PYPTO_IR_PYPTOATTRS_H
#define AKG_DIALECT_PYPTO_IR_PYPTOATTRS_H

#include "mlir/IR/BuiltinAttributes.h"

// The enum utility declarations must come first: the attribute definitions
// below reference the generated enum classes, and the generated AttrDefs
// include does not pull them in. The enum include has no include guard, so
// this is its only inclusion point.
#include "akg/Dialect/Pypto/IR/PyptoEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "akg/Dialect/Pypto/IR/PyptoAttrs.h.inc"

#endif  // AKG_DIALECT_PYPTO_IR_PYPTOATTRS_H
