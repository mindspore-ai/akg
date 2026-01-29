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

#include "mfusion/Dialect/Muse/Muse.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace muse {
//===----------------------------------------------------------------------===//
// Muse_DeviceAttr
//===----------------------------------------------------------------------===//
// Custom parser and printer for Muse_DeviceAttr
// Format: <"type", index>
// Example: <"npu", 0>
mlir::Attribute DeviceAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  std::string typeStr;
  int64_t index;

  if (parser.parseLess()) {
    return {};
  }

  // Parse string type
  if (parser.parseString(&typeStr)) {
    return {};
  }

  if (parser.parseComma()) {
    return {};
  }

  // Parse integer index (without type suffix)
  if (parser.parseInteger(index)) {
    return {};
  }

  if (parser.parseGreater()) {
    return {};
  }

  auto *ctx = parser.getContext();
  auto typeAttr = mlir::StringAttr::get(ctx, typeStr);
  auto indexAttr = mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), index);
  return DeviceAttr::get(ctx, typeAttr, indexAttr);
}

void DeviceAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  printer << "\"" << getDeviceType().getValue() << "\"";
  printer << ", ";
  printer << getIndex().getValue().getSExtValue();
  printer << ">";
}
}  // namespace muse
}  // namespace mlir
