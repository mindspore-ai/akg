/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Fusion/IR/Fusion.h"
#include "akg/Utils/SmallVectorSize.h"

using namespace mlir;          // NOLINT(build/namespaces)
using namespace mlir::fusion;  // NOLINT(build/namespaces)

#include "akg/Dialect/Fusion/IR/FusionOpsDialect.cpp.inc"

void FusionDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "akg/Dialect/Fusion/IR/FusionOps.cpp.inc"
    >();
}
