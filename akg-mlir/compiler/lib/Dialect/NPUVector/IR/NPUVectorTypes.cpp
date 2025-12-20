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

#include "akg/Dialect/NPUVector/IR/NPUVector.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;  // NOLINT(build/namespaces)
using namespace mlir::npuvector;  // NOLINT(build/namespaces)

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions - must be included before implementations
//===----------------------------------------------------------------------===//

// Note: GET_TYPEDEF_CLASSES is included in NPUVectorDialect.cpp to ensure
// type storage classes are complete before addTypes is called.
// parseType and printType are generated there due to useDefaultTypePrinterParser = 1.
// We only include the type method implementations here.

//===----------------------------------------------------------------------===//
// NPUVectorType
//===----------------------------------------------------------------------===//

LogicalResult NPUVectorType::verify(function_ref<InFlightDiagnostic()> emitError, ArrayRef<int64_t> shape,
                                    Type elementType) {
  // Verify element type
  if (!NPUVectorType::isValidElementType(elementType)) {
    return emitError() << "element type must be integer, index, or float type, "
                       << "got: " << elementType;
  }

  // Verify shape - all dimensions must be positive or dynamic
  for (int64_t dim : shape) {
    if (!ShapedType::isDynamic(dim) && dim < 0) {
      return emitError() << "vector dimension must be non-negative or dynamic, "
                         << "got: " << dim;
    }
  }

  return success();
}

NPUVectorType NPUVectorType::cloneWith(std::optional<ArrayRef<int64_t>> shape, Type elementType) const {
  ArrayRef<int64_t> newShape = shape ? *shape : getShape();
  return NPUVectorType::get(newShape, elementType);
}

Type NPUVectorType::parse(AsmParser &parser) {
  SmallVector<int64_t> shape;
  Type elementType;

  if (parser.parseLess()) return Type();

  // Parse shape dimensions - similar to tensor/memref dimension parsing
  // Format: dim1 x dim2 x ... x elementType
  // Use parseDimensionList for proper dimension parsing with dynamic support
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/true)) return Type();

  // Parse element type
  if (parser.parseType(elementType)) return Type();

  if (parser.parseGreater()) return Type();

  return NPUVectorType::get(shape, elementType);
}

void NPUVectorType::print(AsmPrinter &printer) const {
  // Print only the content inside <>, not the <> themselves
  // The outer <> will be added by printDialectSymbol
  auto shape = getShape();
  for (unsigned i = 0, e = shape.size(); i < e; ++i) {
    if (i > 0) printer << "x";
    if (ShapedType::isDynamic(shape[i])) {
      printer << "?";
    } else {
      printer << shape[i];
    }
  }
  printer << "x";
  printer.printType(getElementType());
}
