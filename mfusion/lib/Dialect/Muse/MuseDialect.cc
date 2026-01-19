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

// Custom parser and printer for Muse_DeviceAttr
// Format: <"type", index>
// Example: <"npu", 0>
namespace mlir {
namespace muse {
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

// Custom parser and printer for Muse_TensorType
// Format: muse.tensor<2x3xf32, device=...>
// Example: muse.tensor<2x3xf32, device=<npu, 0>>
mlir::Type TensorType::parse(mlir::AsmParser &odsParser) {
  llvm::SmallVector<int64_t, 4> shape;
  mlir::Type elementType;
  mlir::muse::DeviceAttr device;

  if (odsParser.parseLess()) {
    return {};
  }

  // Parse dimension list
  if (odsParser.parseDimensionList(shape)) {
    return {};
  }

  // Parse element type
  if (odsParser.parseType(elementType)) {
    return {};
  }

  // Parse optional device
  if (odsParser.parseOptionalComma().succeeded()) {
    if (odsParser.parseKeyword("device") || odsParser.parseEqual()) {
      return {};
    }
    mlir::Attribute deviceAttr;
    if (odsParser.parseAttribute(deviceAttr)) {
      return {};
    }
    device = mlir::dyn_cast<mlir::muse::DeviceAttr>(deviceAttr);
    if (!device) {
      odsParser.emitError(odsParser.getNameLoc(), "expected DeviceAttr");
      return {};
    }
  }

  if (odsParser.parseGreater()) {
    return {};
  }

  return mlir::muse::TensorType::get(odsParser.getContext(), shape, elementType, device);
}

void TensorType::print(mlir::AsmPrinter &odsPrinter) const {
  auto elementType = getElementType();
  if (!elementType) {
    return;
  }

  odsPrinter << "<";
  for (int64_t dim : getShape()) {
    if (dim == mlir::ShapedType::kDynamic)
      odsPrinter << "?x";
    else
      odsPrinter << dim << "x";
  }
  odsPrinter.printType(elementType);

  // Print optional device
  auto device = getDevice();
  if (device) {
    odsPrinter << ", device=";
    odsPrinter.printAttribute(device);
  }

  odsPrinter << ">";
}

bool TensorType::hasRank() const { return true; }

mlir::ShapedType TensorType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type elementType) const {
  return mlir::cast<mlir::ShapedType>(
    get(elementType.getContext(), shape.value_or(getShape()), elementType, getDevice()));
}

//===----------------------------------------------------------------------===//
// BindSymbolicShapeOp
//===----------------------------------------------------------------------===//

// muse.bind_symbolic_shape %6, [%0, %1, %2], affine_map<()[s0, s1, s2] ->
// (s0, s1 * 2 + s2, 3)> : !muse.tensor<2x2x3xf32, device=<npu, 0>>
mlir::ParseResult BindSymbolicShapeOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand operand;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> shapeSymbols;
  mlir::AffineMapAttr shapeExpressions;
  mlir::Type operandType;

  if (parser.parseOperand(operand) || parser.parseComma() || parser.parseLSquare() ||
      parser.parseOperandList(shapeSymbols) || parser.parseRSquare() || parser.parseComma() ||
      parser.parseAttribute(shapeExpressions, "shape_expressions", result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(operandType)) {
    return llvm::failure();
  }

  if (parser.resolveOperand(operand, operandType, result.operands) ||
      parser.resolveOperands(shapeSymbols, parser.getBuilder().getType<I64Type>(), result.operands)) {
    return llvm::failure();
  }

  return llvm::success();
}

// Use a custom printer here to avoid the AffineMap from getting hoisted
// when printed. This makes it so the AffineMap is printed inline with the op.
void BindSymbolicShapeOp::print(mlir::OpAsmPrinter &p) {
  p << " " << getOperand() << ", [";
  llvm::interleaveComma(getShapeSymbols(), p);
  p << "], " << "affine_map<" << getShapeExpressions().getValue() << ">";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          /*elidedAttrs=*/{"shape_expressions"});
  p << " : " << getOperand().getType();
}

llvm::LogicalResult BindSymbolicShapeOp::verify() {
  auto affineMap = getShapeExpressions().getValue();
  if (affineMap.getNumSymbols() != getShapeSymbols().size()) {
    return emitOpError() << "number of shape symbols (" << getShapeSymbols().size()
                         << ") must match number of symbols in affine map (" << affineMap.getNumSymbols() << ")";
  }

  for (auto symbol : getShapeSymbols()) {
    mlir::Operation *definingOp = symbol.getDefiningOp();
    if (!mlir::isa<SymbolicIntOp>(definingOp)) {
      return emitOpError() << "shape symbol must be produced by a SymbolicIntOp";
    }
  }

  return llvm::success();
}

//===----------------------------------------------------------------------===//
// EvalSymbolicExprOp
//===----------------------------------------------------------------------===//

llvm::LogicalResult EvalSymbolicExprOp::verify() {
  auto affineMap = getExpr().getValue();
  if (affineMap.getNumSymbols() != getOperands().size()) {
    return emitOpError() << "number of operands (" << getOperands().size()
                         << ") must match number of symbols in affine map (" << affineMap.getNumSymbols() << ")";
  }
  if (affineMap.getNumResults() != 1) {
    return emitOpError() << "affine map must have exactly one result";
  }
  return llvm::success();
}
}  // namespace muse
}  // namespace mlir

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
