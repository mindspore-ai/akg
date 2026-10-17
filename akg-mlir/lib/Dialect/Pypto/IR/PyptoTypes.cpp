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

#include "akg/Dialect/Pypto/IR/Pypto.h"
#include "akg/Dialect/Pypto/IR/PyptoTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

using namespace mlir;       // NOLINT(build/namespaces)
using namespace mlir::pypto;  // NOLINT(build/namespaces)

#define GET_TYPEDEF_CLASSES
#include "akg/Dialect/Pypto/IR/PyptoOpsTypes.cpp.inc"

void PyptoDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "akg/Dialect/Pypto/IR/PyptoOpsTypes.cpp.inc"  // NOLINT(build/include)
      >();
}

namespace {
// Reads one entry of a tile shape list. A '?' becomes a dynamic dimension;
// every other token must be a non-negative integer literal. A diagnostic is
// emitted at the current location and false is returned when neither of the
// two accepted forms is present.
bool parseShapeEntry(::mlir::AsmParser &parser, int64_t &dim) {
  if (succeeded(parser.parseOptionalQuestion())) {
    dim = ShapedType::kDynamic;
    return true;
  }
  if (succeeded(parser.parseInteger(dim)) && dim >= 0) {
    return true;
  }
  parser.emitError(parser.getCurrentLocation())
      << "expected a tile shape entry: '?' or a non-negative integer";
  return false;
}
}  // namespace

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

::mlir::LogicalResult mlir::pypto::TileType::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError, Type elementType,
    llvm::ArrayRef<int64_t> shape) {
  if (!TileType::isValidElementType(elementType)) {
    return emitError() << "tile element type must be integer, index, or float "
                          "type, got: "
                       << elementType;
  }
  for (int64_t dim : shape) {
    if (!ShapedType::isDynamic(dim) && dim <= 0) {
      return emitError() << "tile dimension must be positive or dynamic, got: "
                         << dim;
    }
  }
  return success();
}

Type TileType::parse(::mlir::AsmParser &parser) {
  Type elementType;
  if (failed(parser.parseLess()) || failed(parser.parseType(elementType)) ||
      failed(parser.parseComma()) || failed(parser.parseLSquare())) {
    return Type();
  }

  // An empty shape list closes immediately; otherwise the entries are comma
  // separated and the list has to be terminated by a closing bracket.
  SmallVector<int64_t> shape;
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int64_t dim = 0;
      if (!parseShapeEntry(parser, dim)) {
        return Type();
      }
      shape.push_back(dim);
    } while (succeeded(parser.parseOptionalComma()));
    if (failed(parser.parseRSquare())) {
      return Type();
    }
  }

  if (failed(parser.parseGreater())) {
    return Type();
  }
  return TileType::get(elementType, shape);
}

void TileType::print(::mlir::AsmPrinter &printer) const {
  printer << "<";
  printer.printType(getElementType());
  printer << ", [";
  ::llvm::ArrayRef<int64_t> shape = getShape();
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) {
      printer << ", ";
    }
    int64_t dim = shape[i];
    if (ShapedType::isDynamic(dim)) {
      printer << "?";
    } else {
      printer << dim;
    }
  }
  printer << "]>";
}
