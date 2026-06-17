/**
 * Copyright 2026 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef AKG_DIALECT_NPUVECTOR_TRANSFORMS_ELIMSCFITERARGS_H_
#define AKG_DIALECT_NPUVECTOR_TRANSFORMS_ELIMSCFITERARGS_H_

#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace npuvector {

std::unique_ptr<OperationPass<ModuleOp>> createElimScfIterArgsPass();
}  // namespace npuvector
}  // namespace mlir

#endif  // AKG_DIALECT_NPUVECTOR_TRANSFORMS_ELIMSCFITERARGS_H_
