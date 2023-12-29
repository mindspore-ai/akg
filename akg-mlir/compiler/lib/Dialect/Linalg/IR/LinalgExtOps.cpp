/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Linalg/IR/LinalgExtOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/SMLoc.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "akg/Dialect/Linalg/IR/LinalgExtOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::linalgExt;
void mlir::linalgExt::LinalgExtDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "akg/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
    >();
}

static void getGenericEffectsImpl(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects,
                                  ValueRange results, const OpOperandVector &inputOperands,
                                  const OpOperandVector &outputOperands) {
  for (auto *operand : inputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(), SideEffects::DefaultResource::get());
  }
  for (auto *operand : outputOperands) {
    if (!operand->get().getType().isa<MemRefType>()) {
      continue;
    }
    effects.emplace_back(MemoryEffects::Read::get(), operand->get(), SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), operand->get(), SideEffects::DefaultResource::get());
  }
}

// ===----------------------------------------------------------------------=== //
// GatherOp
// ===----------------------------------------------------------------------=== //

void GatherOp::print(OpAsmPrinter &p) {
  if (getNumOperands() > 0) {
    p << " " << getOperands();
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0) {
    p << " : " << getOperandTypes();
  }
}

ParseResult GatherOp::parse(OpAsmParser &parser, OperationState &result) {
  Type dataType, indicesType, outputType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(dataType) || parser.parseComma() ||
      parser.parseType(indicesType) || parser.parseComma() || parser.parseType(outputType))
    return failure();
  if (!outputType.isa<MemRefType>()) {
    result.addTypes(outputType);
  }
  return parser.resolveOperands(operands, {dataType, indicesType, outputType}, parser.getNameLoc(), result.operands);
}

void GatherOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

// ===----------------------------------------------------------------------=== //
// UnsortedSegmentSumOp
// ===----------------------------------------------------------------------=== //

void UnsortedSegmentSumOp::print(OpAsmPrinter &p) {
  if (getNumOperands() > 0) {
    p << " " << getOperands();
  }
  p.printOptionalAttrDict((*this)->getAttrs());
  if (getNumOperands() > 0) {
    p << " : " << getOperandTypes();
  }
}

ParseResult UnsortedSegmentSumOp::parse(OpAsmParser &parser, OperationState &result) {
  Type dataType, indicesType, outputType;
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/3) ||
      parser.parseOptionalAttrDict(result.attributes) || parser.parseColonType(dataType) || parser.parseComma() ||
      parser.parseType(indicesType) || parser.parseComma() || parser.parseType(outputType))
    return failure();
  if (!outputType.isa<MemRefType>()) {
    result.addTypes(outputType);
  }
  return parser.resolveOperands(operands, {dataType, indicesType, outputType}, parser.getNameLoc(), result.operands);
}

void UnsortedSegmentSumOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputOperands(), getDpsInitOperands());
}

#define GET_OP_CLASSES
#include "akg/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
