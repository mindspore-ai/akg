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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;  // NOLINT(build/namespaces)
using namespace mlir::npuvector;  // NOLINT(build/namespaces)

#include "akg/Dialect/NPUVector/IR/NPUVectorOpsDialect.cpp.inc"

// Include type definitions to ensure storage class is complete before addTypes
// This must be included before calling addTypes, but the actual method
// implementations are in NPUVectorTypes.cpp
#define GET_TYPEDEF_CLASSES
#include "akg/Dialect/NPUVector/IR/NPUVectorOpsTypes.cpp.inc"

void mlir::npuvector::NPUVectorDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
// NOLINTNEXTLINE(build/include)
#include "akg/Dialect/NPUVector/IR/NPUVectorOpsTypes.cpp.inc"
    >();
  addOperations<
#define GET_OP_LIST
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.cpp.inc"
    >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *NPUVectorDialect::materializeConstant(OpBuilder &builder, Attribute value, Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

Attribute NPUVectorDialect::parseAttribute(DialectAsmParser &parser, Type type) const {
  // NPUVector dialect doesn't define custom attributes, so return empty
  parser.emitError(parser.getCurrentLocation()) << "unknown attribute in dialect `" << getNamespace() << "`";
  return {};
}

void NPUVectorDialect::printAttribute(Attribute attr, DialectAsmPrinter &printer) const {
  // NPUVector dialect doesn't define custom attributes
  llvm_unreachable("NPUVector dialect has no attributes to print");
}

/// Print a type registered to this dialect.
void NPUVectorDialect::printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const {
  // Custom print for NPUVectorType to use !npuvector<...> format (without type name)
  if (auto npuVectorType = ::mlir::dyn_cast<NPUVectorType>(type)) {
    // Print only the type parameters, not the mnemonic
    // The dialect namespace (!npuvector.) will be added by the caller
    npuVectorType.print(printer);
    return;
  }

  // Note: generatedTypePrinter is defined in NPUVectorOpsTypes.cpp.inc
  // which is included above with GET_TYPEDEF_CLASSES, so it's available here
  // Fallback to generated printer for other types (if any)
  if (::mlir::succeeded(generatedTypePrinter(type, printer))) return;
}

/// Parse a type registered to this dialect.
::mlir::Type NPUVectorDialect::parseType(::mlir::DialectAsmParser &parser) const {
  // When parseType is called, the parser is already positioned at the type
  // parameters (e.g., ?xf32 for !npuvector<?xf32>, without the < and >)
  // Parse dimension list and element type directly
  SmallVector<int64_t> shape;
  Type elementType;

  // Parse dimension list: ?x?x or ?x? or just ?
  // parseDimensionList expects format: dim1 x dim2 x ... x elementType
  if (parser.parseDimensionList(shape, /*allowDynamic=*/true, /*withTrailingX=*/true)) return {};

  // Parse element type
  if (parser.parseType(elementType)) return {};

  return NPUVectorType::get(shape, elementType);
}
