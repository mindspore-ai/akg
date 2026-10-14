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

#ifndef AKG_TARGET_PYTHON_PYTHONEMITTER_H
#define AKG_TARGET_PYTHON_PYTHONEMITTER_H

#include <optional>
#include <string>

#include "mlir/Support/LLVM.h"

namespace mlir {
class Operation;
namespace emitpy {

/// Translates the given operation to Python code. The operation or operations
/// in the region of 'op' all need to be in EmitPy dialect, except for those
/// that are explicitly labeled as (dynamically) legal ops when converting to
/// EmitPy dialect. When `fileId` is provided, only the ops of the
/// `emitpy.file` with the matching id are emitted. Otherwise, all ops are
/// emitted.
LogicalResult
translateToPython(Operation *op, raw_ostream &os,
                  std::optional<std::string> fileId = std::nullopt);

/// Registers the "mlir-to-pypto" translation with the translate registry.
void registerToPyptoTranslation();

}  // namespace emitpy
}  // namespace mlir

#endif  // AKG_TARGET_PYTHON_PYTHONEMITTER_H
