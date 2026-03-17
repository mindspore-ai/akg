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
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace mfuse {

/// Helper function to add scalar marker attribute to rank-0 tensors if missing
static Type addScalarMarkerToType(Type type, MLIRContext *ctx) {
  if (auto rankedType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    // Only add scalar marker for rank-0 tensors
    if (rankedType.getRank() == 0) {
      auto encoding = rankedType.getEncoding();
      bool hasScalarMarker = false;

      // Check if encoding already has scalar marker
      if (encoding) {
        if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(encoding)) {
          hasScalarMarker = dictAttr.contains(mlir::mfuse::kScalarMarkerAttr);
        }
      }

      if (hasScalarMarker) {
        return type;
      }

      // If no scalar marker, add it
      // Create scalar marker attribute
      auto scalarMarkerAttr = mlir::NamedAttribute(mlir::StringAttr::get(ctx, mlir::mfuse::kScalarMarkerAttr),
                                                   mlir::StringAttr::get(ctx, ""));

      // Create new encoding with scalar marker
      mlir::SmallVector<mlir::NamedAttribute> attrs;
      if (encoding) {
        if (auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(encoding)) {
          // Keep existing attributes
          attrs.append(dictAttr.begin(), dictAttr.end());
        }
      }
      // Add scalar marker attribute
      attrs.emplace_back(scalarMarkerAttr);

      // Create new encoding dictionary
      auto newEncoding = mlir::DictionaryAttr::get(ctx, attrs);

      // Create new type with updated encoding
      return mlir::RankedTensorType::get(rankedType.getShape(), rankedType.getElementType(), newEncoding);
    }
  }
  // Return original type if no changes needed
  return type;
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===

void ConstantOp::build(OpBuilder &builder, OperationState &result, Type resultType, Attribute value) {
  // If resultType is not provided, use value's type
  if (!resultType) {
    auto typedValue = mlir::dyn_cast<mlir::TypedAttr>(value);
    if (typedValue) {
      resultType = typedValue.getType();
    }
  }

  // Add scalar marker to rank-0 tensors if missing
  resultType = addScalarMarkerToType(resultType, builder.getContext());

  // Set the value attribute and result type
  result.addAttribute("value", value);
  result.addTypes(resultType);
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  TypedAttr valueAttr;
  if (parser.parseAttribute(valueAttr, "value", result.attributes)) {
    return failure();
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  auto type = valueAttr.getType();
  // Add scalar marker to rank-0 tensors if missing
  type = addScalarMarkerToType(type, parser.getContext());
  result.addTypes(type);

  return success();
}

void ConstantOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printAttributeWithoutType(getValue());
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  // Print the result type, which will include the encoding (like is_scalar)
  printer << " : " << getResult().getType();
}

bool ConstantOp::isBuildableWith(Attribute value, Type type) {
  auto typedAttr = mlir::dyn_cast<TypedAttr>(value);
  if (!typedAttr) {
    return false;
  }

  auto attrType = typedAttr.getType();
  if (attrType == type) {
    return true;
  }

  auto attrTensor = mlir::dyn_cast<RankedTensorType>(attrType);
  auto targetTensor = mlir::dyn_cast<RankedTensorType>(type);

  if (!attrTensor || !targetTensor) {
    return false;
  }

  return attrTensor.getShape() == targetTensor.getShape() &&
         attrTensor.getElementType() == targetTensor.getElementType();
}

ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value, Type type, Location loc) {
  // Add scalar marker to rank-0 tensors if missing
  type = addScalarMarkerToType(type, builder.getContext());

  if (!isBuildableWith(value, type)) {
    return nullptr;
  }
  auto typedValue = mlir::cast<TypedAttr>(value);
  return builder.create<ConstantOp>(loc, type, typedValue);
}

LogicalResult ConstantOp::verify() {
  auto valueType = getValue().getType();
  auto resultType = getResult().getType();

  if (valueType == resultType) {
    return success();
  }

  auto valueTensor = dyn_cast<RankedTensorType>(valueType);
  auto resultTensor = dyn_cast<RankedTensorType>(resultType);

  if (!valueTensor || !resultTensor) {
    return emitOpError() << "value type " << valueType << " and result type " << resultType
                         << " must both be ranked tensor types";
  }

  if (valueTensor.getShape() != resultTensor.getShape()) {
    return emitOpError() << "shape mismatch: value has shape " << valueTensor.getShape() << " but result has shape "
                         << resultTensor.getShape();
  }

  if (valueTensor.getElementType() != resultTensor.getElementType()) {
    return emitOpError() << "element type mismatch: value has element type " << valueTensor.getElementType()
                         << " but result has element type " << resultTensor.getElementType();
  }

  if (valueTensor.getRank() == 0) {
    // Check if result tensor has scalar marker
    auto resultEncoding = resultTensor.getEncoding();
    if (resultEncoding) {
      auto dictAttr = mlir::dyn_cast<mlir::DictionaryAttr>(resultEncoding);
      if (!dictAttr || !dictAttr.contains(mlir::mfuse::kScalarMarkerAttr)) {
        return emitOpError() << "rank-0 constant must have scalar marker encoding";
      }
    } else {
      return emitOpError() << "rank-0 constant must have scalar marker encoding";
    }
  }
  return success();
}

OpFoldResult ConstantOp::fold(ConstantOpGenericAdaptor<ArrayRef<Attribute>> operands) { return getValue(); }

}  // namespace mfuse
}  // namespace mlir
