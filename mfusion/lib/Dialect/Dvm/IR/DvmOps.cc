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
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "mfusion/Dialect/Dvm/IR/Dvm.cpp.inc"

using namespace mlir;
using namespace mlir::dvm;

// Verify that dvm.constant only supports scalar tensor (rank=0)
LogicalResult ConstantOp::verify() {
  auto resultType = getResult().getType();
  auto rankedType = llvm::dyn_cast<RankedTensorType>(resultType);
  if (!rankedType) {
    return failure();
  }
  if (rankedType.getRank() != 0) {
    return emitError("dvm.constant only supports scalar tensor (rank=0), got rank ") << rankedType.getRank();
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
