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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionImplementation.h"
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
                                  ValueRange results, const ValueRange &inputOperands,
                                  const ValueRange &outputOperands) {
  for (auto operand : inputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType())) {
      continue;
    }
  }
  for (auto operand : outputOperands) {
    if (!llvm::isa<MemRefType>(operand.getType())) {
      continue;
    }
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
      parser.parseType(indicesType) || parser.parseComma() || parser.parseType(outputType)) {
    return failure();
  }

  if (!outputType.isa<MemRefType>()) {
    result.addTypes(outputType);
  }
  return parser.resolveOperands(operands, {dataType, indicesType, outputType}, parser.getNameLoc(), result.operands);
}

void GatherOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(), getDpsInits());
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
      parser.parseType(indicesType) || parser.parseComma() || parser.parseType(outputType)) {
    return failure();
  }

  if (!outputType.isa<MemRefType>()) {
    result.addTypes(outputType);
  }
  return parser.resolveOperands(operands, {dataType, indicesType, outputType}, parser.getNameLoc(), result.operands);
}

void UnsortedSegmentSumOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  getGenericEffectsImpl(effects, getOperation()->getResults(), getDpsInputs(), getDpsInits());
}

//===----------------------------------------------------------------------===//
// TemplateOps
//===----------------------------------------------------------------------===//
static ParseResult
parseNamedStructuredOpResults(OpAsmParser &parser,
                              SmallVectorImpl<Type> &resultTypes) {
  if (parser.parseOptionalArrowTypeList(resultTypes))
    return failure();
  return success();
}
/// Common parsing used for both named structured ops created by ods-gen and by
/// manually defined C++ ops. Does not handle regions.
static ParseResult
parseCommonStructuredOpParts(OpAsmParser &parser, OperationState &result,
                             SmallVectorImpl<Type> &inputTypes,
                             SmallVectorImpl<Type> &outputTypes,
                             bool addOperandSegmentSizes = true) {
  SMLoc attrsLoc, inputsOperandsLoc, outputsOperandsLoc;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> inputsOperands,
      outputsOperands;

  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseAttribute(result.propertiesAttr) || parser.parseGreater())
      return failure();
  }
  attrsLoc = parser.getCurrentLocation();
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("ins"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseLParen() || parser.parseOperandList(outputsOperands) ||
        parser.parseColonTypeList(outputTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands) ||
      parser.resolveOperands(outputsOperands, outputTypes, outputsOperandsLoc,
                             result.operands))
    return failure();

  if (addOperandSegmentSizes) {
    // This is a bit complex because we're trying to be backward compatible with
    // operation syntax that mix the inherent attributes and the discardable
    // ones in the same dictionary. If the properties are used, we append the
    // operandSegmentSizes there directly. Otherwise we append it to the
    // discardable attributes dictionary where it is handled by the generic
    // Operation::create(...) method.
    if (result.propertiesAttr) {
      NamedAttrList attrs = llvm::cast<DictionaryAttr>(result.propertiesAttr);
      attrs.append("operandSegmentSizes",
                   parser.getBuilder().getDenseI32ArrayAttr(
                       {static_cast<int32_t>(inputsOperands.size()),
                        static_cast<int32_t>(outputsOperands.size())}));
      result.propertiesAttr = attrs.getDictionary(parser.getContext());
    } else {
      result.addAttribute("operandSegmentSizes",
                          parser.getBuilder().getDenseI32ArrayAttr(
                              {static_cast<int32_t>(inputsOperands.size()),
                               static_cast<int32_t>(outputsOperands.size())}));
    }
  }
  if (!result.propertiesAttr) {
    std::optional<RegisteredOperationName> info =
        result.name.getRegisteredInfo();
    if (info) {
      if (failed(info->verifyInherentAttrs(result.attributes, [&]() {
            return parser.emitError(attrsLoc)
                   << "'" << result.name.getStringRef() << "' op ";
          })))
        return failure();
    }
  }
  return success();
}

static void printNamedStructuredOpResults(OpAsmPrinter &p,
                                          TypeRange resultTypes) {
  if (resultTypes.empty())
    return;
  p.printOptionalArrowTypeList(resultTypes);
}

static void printCommonStructuredOpParts(OpAsmPrinter &p, ValueRange inputs,
                                         ValueRange outputs) {
  if (!inputs.empty())
    p << " ins(" << inputs << " : " << inputs.getTypes() << ")";
  if (!outputs.empty())
    p << " outs(" << outputs << " : " << outputs.getTypes() << ")";
}

void TemplateOp::getAsmBlockArgumentNames(Region &region,
                                        OpAsmSetValueNameFn setNameFn) {
    return;
}

void TemplateOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayAttr indexingMaps,
    ArrayAttr iteratorTypes, StringAttr doc, StringAttr libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall);
  result.addAttributes(attributes);
  if (!bodyBuild)
    return;

  SmallVector<Type, 4> blockArgTypes;
  SmallVector<Location, 4> blockArgLocs;
  for (ValueRange container : {inputs, outputs}) {
    for (Value v : container) {
      blockArgTypes.push_back(getElementTypeOrSelf(v));
      blockArgLocs.push_back(v.getLoc());
    }
  }

  OpBuilder::InsertionGuard guard(builder);
  auto &region = *result.regions.front();
  Block *bodyBlock =
      builder.createBlock(&region, region.end(), blockArgTypes, blockArgLocs);
  bodyBuild(builder, result.location, bodyBlock->getArguments());
}

void TemplateOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs,
        builder.getAffineMapArrayAttr(indexingMaps),
        // builder.getStrArrayAttr(iteratorTypes),
        builder.getArrayAttr(llvm::to_vector(llvm::map_range(
            iteratorTypes,
            [&](utils::IteratorType iter) -> mlir::Attribute {
              return IteratorTypeAttr::get(builder.getContext(), iter);
            }))),
        doc.empty() ? StringAttr() : builder.getStringAttr(doc),
        libraryCall.empty() ? StringAttr() : builder.getStringAttr(libraryCall),
        bodyBuild, attributes);
}

void TemplateOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes, StringRef doc, StringRef libraryCall,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, TypeRange{}, inputs, outputs, indexingMaps,
        iteratorTypes, doc, libraryCall, bodyBuild, attributes);
}

void TemplateOp::build(
    OpBuilder &builder, OperationState &result, ValueRange inputs,
    ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, inputs, outputs, indexingMaps, iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void TemplateOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTensorTypes,
    ValueRange inputs, ValueRange outputs, ArrayRef<AffineMap> indexingMaps,
    ArrayRef<utils::IteratorType> iteratorTypes,
    function_ref<void(OpBuilder &, Location, ValueRange)> bodyBuild,
    ArrayRef<NamedAttribute> attributes) {
  build(builder, result, resultTensorTypes, inputs, outputs, indexingMaps,
        iteratorTypes,
        /*doc=*/"",
        /*libraryCall=*/"", bodyBuild, attributes);
}

void TemplateOp::print(OpAsmPrinter &p) {
  p << " ";

  // Print extra attributes.
  auto genericAttrNames = linalgTraitAttrNames();

  llvm::StringSet<> genericAttrNamesSet;
  genericAttrNamesSet.insert(genericAttrNames.begin(), genericAttrNames.end());
  SmallVector<NamedAttribute, 8> genericAttrs;
  for (auto attr : (*this)->getAttrs()) {
    if (attr.getName() == getIteratorTypesAttrName()) {
      auto iteratorTypes =
          attr.getValue()
              .cast<ArrayAttr>()
              .getAsValueRange<IteratorTypeAttr, utils::IteratorType>();
      // Convert IteratorType enums into the string representation. This is
      // needed, because tests still use the old format when 'iterator_types'
      // attribute is represented as an array of strings.
      // TODO: Remove this conversion once tests are fixed.
      SmallVector<Attribute> iteratorTypeNames =
          llvm::to_vector(llvm::map_range(
              iteratorTypes, [&](utils::IteratorType t) -> Attribute {
                return StringAttr::get(getContext(), stringifyIteratorType(t));
              }));

      genericAttrs.emplace_back(
          getIteratorTypesAttrName(),
          ArrayAttr::get(getContext(), iteratorTypeNames));
    } else if (genericAttrNamesSet.count(attr.getName().strref()) > 0) {
      genericAttrs.push_back(attr);
    }
  }
  if (!genericAttrs.empty()) {
    auto genericDictAttr = DictionaryAttr::get(getContext(), genericAttrs);
    p << genericDictAttr;
  }

  // Printing is shared with named ops, except for the region and attributes
  printCommonStructuredOpParts(p, getInputs(), getOutputs());

  genericAttrNames.push_back("operand_segment_sizes");
  genericAttrNamesSet.insert(genericAttrNames.back());

  bool hasExtraAttrs = false;
  for (NamedAttribute n : (*this)->getAttrs()) {
    if ((hasExtraAttrs = !genericAttrNamesSet.contains(n.getName().strref())))
      break;
  }
  if (hasExtraAttrs) {
    p << " attrs = ";
    p.printOptionalAttrDict((*this)->getAttrs(),
                            /*elidedAttrs=*/genericAttrNames);
  }

  // Print region.
  if (!getRegion().empty()) {
    p << ' ';
    p.printRegion(getRegion());
  }

  // Print results.
  printNamedStructuredOpResults(p, getResultTensors().getTypes());
}

ParseResult TemplateOp::parse(OpAsmParser &parser, OperationState &result) {
  DictionaryAttr dictAttr;
  // Parse the core linalg traits that must check into a dictAttr.
  // The name is unimportant as we will overwrite result.attributes.
  // The core linalg traits must contain the information necessary to pass the
  // verifier.
  if (parser.parseAttribute(dictAttr, "_", result.attributes))
    return failure();
  result.attributes.assign(dictAttr.getValue().begin(),
                           dictAttr.getValue().end());

  // Convert array of string into an array of IteratyType enums. This is needed,
  // because tests still use the old format when 'iterator_types' attribute is
  // represented as an array of strings.
  // TODO: Remove this conversion once tests are fixed.
  ArrayAttr iteratorTypes =
      result.attributes.get(getIteratorTypesAttrName(result.name))
          .cast<ArrayAttr>();

  SmallVector<Attribute> iteratorTypeAttrs;

  for (StringRef s : iteratorTypes.getAsValueRange<StringAttr>()) {
    auto maybeIteratorType = utils::symbolizeIteratorType(s);
    if (!maybeIteratorType.has_value())
      return parser.emitError(parser.getCurrentLocation())
             << "unexpected iterator_type (" << s << ")";
 
    iteratorTypeAttrs.push_back(
        IteratorTypeAttr::get(parser.getContext(), maybeIteratorType.value()));
  }
  result.attributes.set(getIteratorTypesAttrName(result.name),
                        parser.getBuilder().getArrayAttr(iteratorTypeAttrs));

  // Parsing is shared with named ops, except for the region.
  SmallVector<Type, 1> inputTypes, outputTypes;
  if (parseCommonStructuredOpParts(parser, result, inputTypes, outputTypes))
    return failure();

  // Optional attributes may be added.
  if (succeeded(parser.parseOptionalKeyword("attrs")))
    if (failed(parser.parseEqual()) ||
        failed(parser.parseOptionalAttrDict(result.attributes)))
      return failure();
 
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, {}))
    return failure();
  result.addRegion(std::move(region));

  // Generic ops may specify that a subset of its outputs are tensors. Such
  // outputs are specified in the result type.
  // TODO: may need to move output parsing before region parsing.
  // Need to wait for declarative assembly resolution to decide.
  SmallVector<Type, 1> outputTensorsTypes;
  if (parseNamedStructuredOpResults(parser, outputTensorsTypes))
    return failure();
  result.addTypes(outputTensorsTypes);

  return success();
}

void TemplateOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  return;
}

#define GET_OP_CLASSES
#include "akg/Dialect/Linalg/IR/LinalgExtOps.cpp.inc"
