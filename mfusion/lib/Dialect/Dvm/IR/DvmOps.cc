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

#include "mfusion/Dialect/Dvm/IR/DvmDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include <optional>

#define GET_OP_CLASSES
#include "mfusion/Dialect/Dvm/IR/Dvm.cpp.inc"

using namespace mlir;
using namespace mlir::dvm;

static bool isSupportedDvmScalarConstantType(Type type) {
  return type.isF32() || type.isF16() || type.isBF16() || type.isInteger(32);
}

static bool isSupportedDvmScalarAttrType(Type type) { return isSupportedDvmScalarConstantType(type); }

static ParseResult parseBinaryOpType(OpAsmParser &parser, OperationState &result) {
  StringRef opTypeName;
  if (parser.parseKeyword(&opTypeName)) {
    return failure();
  }
  auto opType = symbolizeBinaryOpType(opTypeName);
  if (!opType) {
    return parser.emitError(parser.getNameLoc(), "unknown dvm binary op type: ") << opTypeName;
  }
  result.addAttribute("op_type", BinaryOpTypeAttr::get(parser.getContext(), *opType));
  return success();
}

static FailureOr<DTypeAttr> parseDTypeAttr(OpAsmParser &parser) {
  StringRef typeName;
  if (parser.parseKeyword(&typeName)) {
    return failure();
  }
  auto dtype = symbolizeDType(typeName);
  if (!dtype) {
    return parser.emitError(parser.getNameLoc(), "unknown dvm dtype: ") << typeName;
  }
  return DTypeAttr::get(parser.getContext(), *dtype);
}

struct ParsedScalarLiteral {
  std::optional<APInt> intValue;
  std::optional<APFloat> floatValue;
};

static ParseResult parseScalarLiteral(OpAsmParser &parser, ParsedScalarLiteral &literal) {
  APInt intValue;
  OptionalParseResult intParseResult = parser.parseOptionalInteger(intValue);
  if (intParseResult.has_value()) {
    if (failed(intParseResult.value())) {
      return failure();
    }
    literal.intValue = intValue;
    return success();
  }

  APFloat floatValue(APFloat::IEEEdouble());
  if (parser.parseFloat(APFloat::IEEEdouble(), floatValue)) {
    return failure();
  }
  literal.floatValue = floatValue;
  return success();
}

static FailureOr<TypedAttr> buildScalarAttrFromLiteral(OpAsmParser &parser, const ParsedScalarLiteral &literal,
                                                       Type type, StringRef opName) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    if (!literal.intValue) {
      return parser.emitError(parser.getNameLoc(), opName) << " integer scalar requires an integer literal";
    }
    APInt value = literal.intValue->sextOrTrunc(intType.getWidth());
    return cast<TypedAttr>(IntegerAttr::get(intType, value));
  }

  auto floatType = dyn_cast<FloatType>(type);
  if (!floatType) {
    return parser.emitError(parser.getNameLoc(), opName) << " unsupported scalar type: " << type;
  }
  if (!literal.floatValue) {
    return parser.emitError(parser.getNameLoc(), opName) << " floating scalar requires a float literal";
  }

  APFloat value = *literal.floatValue;
  bool losesInfo = false;
  value.convert(floatType.getFloatSemantics(), APFloat::rmNearestTiesToEven, &losesInfo);
  return cast<TypedAttr>(FloatAttr::get(floatType, value));
}

static Type getElementType(Type type) {
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return shapedType.getElementType();
  }
  return {};
}

static bool isDTypeCompatibleWithElementType(DType dtype, Type elementType) {
  switch (dtype) {
    case DType::Bool:
      return elementType.isInteger(1);
    case DType::Float16:
      return elementType.isF16();
    case DType::BFloat16:
      return elementType.isBF16();
    case DType::Float32:
      return elementType.isF32();
    case DType::Int32:
      return elementType.isInteger(32);
    case DType::Int64:
      return elementType.isInteger(64);
  }
  return false;
}

LogicalResult BinaryOp::verify() {
  auto lhsElementType = getElementType(getLhs().getType());
  auto rhsElementType = getElementType(getRhs().getType());
  if (lhsElementType && rhsElementType && lhsElementType != rhsElementType) {
    return emitError("dvm.binary operands must have the same element type, got ")
           << lhsElementType << " vs " << rhsElementType;
  }
  return success();
}

LogicalResult BinaryScalarOp::verify() {
  auto scalarType = getScalarAttr().getType();
  if (!isSupportedDvmScalarAttrType(scalarType)) {
    return emitError("dvm.binary_scalar unsupported scalar type: ")
           << scalarType << "; supported types are f32, f16, bf16, i32";
  }
  return success();
}

ParseResult BinaryScalarOp::parse(OpAsmParser &parser, OperationState &result) {
  if (parseBinaryOpType(parser, result)) {
    return failure();
  }

  OpAsmParser::UnresolvedOperand inputOperand;
  ParsedScalarLiteral scalarLiteral;
  bool scalarOnLhs = false;

  OptionalParseResult inputParseResult = parser.parseOptionalOperand(inputOperand);
  if (inputParseResult.has_value()) {
    if (failed(inputParseResult.value())) {
      return failure();
    }
    scalarOnLhs = false;
    if (parser.parseComma() || parseScalarLiteral(parser, scalarLiteral)) {
      return failure();
    }
  } else {
    scalarOnLhs = true;
    if (parseScalarLiteral(parser, scalarLiteral) || parser.parseComma() || parser.parseOperand(inputOperand)) {
      return failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  Type firstType;
  Type secondType;
  Type resultType;
  if (parser.parseColon() || parser.parseType(firstType) || parser.parseComma() || parser.parseType(secondType) ||
      parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }

  Type inputType = scalarOnLhs ? secondType : firstType;
  Type scalarType = scalarOnLhs ? firstType : secondType;
  if (!isa<RankedTensorType>(inputType)) {
    return parser.emitError(parser.getNameLoc(), "dvm.binary_scalar tensor operand must have ranked tensor type");
  }

  auto typedScalarAttr = buildScalarAttrFromLiteral(parser, scalarLiteral, scalarType, "dvm.binary_scalar");
  if (failed(typedScalarAttr)) {
    return failure();
  }

  result.addAttribute("scalar", *typedScalarAttr);
  result.addAttribute("scalar_on_lhs", BoolAttr::get(parser.getContext(), scalarOnLhs));
  if (parser.resolveOperand(inputOperand, inputType, result.operands)) {
    return failure();
  }
  result.addTypes(resultType);
  return success();
}

void BinaryScalarOp::print(OpAsmPrinter &printer) {
  printer << " " << stringifyBinaryOpType(getOpType()) << " ";
  auto scalarType = getScalarAttr().getType();
  auto inputType = getInput().getType();
  if (getScalarOnLhs()) {
    printer.printAttributeWithoutType(getScalarAttr());
    printer << ", " << getInput();
    printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"op_type", "scalar", "scalar_on_lhs"});
    printer << " : " << scalarType << ", " << inputType << " -> " << getResult().getType();
    return;
  }

  printer << getInput() << ", ";
  printer.printAttributeWithoutType(getScalarAttr());
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"op_type", "scalar", "scalar_on_lhs"});
  printer << " : " << inputType << ", " << scalarType << " -> " << getResult().getType();
}

LogicalResult BroadcastScalarOp::verify() {
  auto scalarType = getScalarAttr().getType();
  if (!isSupportedDvmScalarAttrType(scalarType)) {
    return emitError("dvm.broadcast_scalar unsupported scalar type: ")
           << scalarType << "; supported scalar types are f32, f16, bf16, i32";
  }

  auto dtype = getTypeAttr().getValue();
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (!isDTypeCompatibleWithElementType(dtype, resultType.getElementType())) {
    return emitError("dvm.broadcast_scalar result element type ")
           << resultType.getElementType() << " does not match dtype " << stringifyDType(dtype);
  }

  auto shape = getShape();
  if (static_cast<int64_t>(shape.size()) != resultType.getRank()) {
    return emitError("dvm.broadcast_scalar shape rank must match result rank, got ")
           << shape.size() << " vs " << resultType.getRank();
  }

  for (auto [index, dimAttr] : llvm::enumerate(shape)) {
    auto dim = cast<IntegerAttr>(dimAttr).getInt();
    if (dim != resultType.getDimSize(index)) {
      return emitError("dvm.broadcast_scalar shape must match result shape at dim ")
             << index << ", got " << dim << " vs " << resultType.getDimSize(index);
    }
  }

  return success();
}

ParseResult BroadcastScalarOp::parse(OpAsmParser &parser, OperationState &result) {
  ParsedScalarLiteral scalarLiteral;
  if (parseScalarLiteral(parser, scalarLiteral)) {
    return failure();
  }

  if (parser.parseKeyword("shape")) {
    return failure();
  }
  ArrayAttr shapeAttr;
  if (parser.parseAttribute(shapeAttr, "shape", result.attributes)) {
    return failure();
  }

  if (parser.parseKeyword("type")) {
    return failure();
  }
  auto dtypeAttr = parseDTypeAttr(parser);
  if (failed(dtypeAttr)) {
    return failure();
  }
  result.addAttribute("type", *dtypeAttr);

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  Type scalarType;
  Type resultType;
  if (parser.parseColon() || parser.parseType(scalarType) || parser.parseArrow() || parser.parseType(resultType)) {
    return failure();
  }

  auto typedScalarAttr = buildScalarAttrFromLiteral(parser, scalarLiteral, scalarType, "dvm.broadcast_scalar");
  if (failed(typedScalarAttr)) {
    return failure();
  }

  result.addAttribute("scalar", *typedScalarAttr);
  result.addTypes(resultType);
  return success();
}

void BroadcastScalarOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printAttributeWithoutType(getScalarAttr());
  printer << " shape " << getShapeAttr();
  printer << " type " << stringifyDType(getTypeAttr().getValue());
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"scalar", "shape", "type"});
  printer << " : " << getScalarAttr().getType() << " -> " << getResult().getType();
}

// Verify that dvm.constant matches the scalar types supported by dvm.h scalar
// APIs: float, int32_t, Float16 and BFloat16. double, int64_t and bool are
// not supported.
LogicalResult ConstantOp::verify() {
  auto resultType = getResult().getType();
  auto valueType = getValueAttr().getType();
  if (valueType != resultType) {
    return emitError("dvm.constant value attribute type must match result type, got ")
           << valueType << " vs " << resultType;
  }

  auto rankedType = llvm::dyn_cast<RankedTensorType>(resultType);
  if (!rankedType) {
    return emitError("dvm.constant result must be a ranked tensor type");
  }
  if (rankedType.getRank() != 0) {
    return emitError("dvm.constant only supports scalar tensor (rank=0), got rank ") << rankedType.getRank();
  }
  auto denseAttr = llvm::dyn_cast<DenseElementsAttr>(getValueAttr());
  if (!denseAttr || denseAttr.getNumElements() != 1) {
    return emitError("dvm.constant value must be a single-element DenseElementsAttr");
  }
  auto elementType = rankedType.getElementType();
  if (!isSupportedDvmScalarConstantType(elementType)) {
    return emitError("dvm.constant unsupported element type: ") << elementType;
  }
  return success();
}

// Parse dvm.constant: dense<value> : type
ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  TypedAttr valueAttr;
  if (parser.parseAttribute(valueAttr, "value", result.attributes)) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Result type is the type of the value attribute
  auto type = valueAttr.getType();
  result.addTypes(type);

  return success();
}

// Print dvm.constant: dense<value> : type
void ConstantOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printAttributeWithoutType(getValue());
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << " : " << getResult().getType();
}
