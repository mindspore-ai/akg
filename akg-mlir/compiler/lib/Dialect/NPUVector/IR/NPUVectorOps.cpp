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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"

using namespace mlir;  // NOLINT(build/namespaces)
using namespace mlir::npuvector;  // NOLINT(build/namespaces)

//===----------------------------------------------------------------------===//
// TransferReadOp
//===----------------------------------------------------------------------===//

// Helper function to extract vector type information
static std::tuple<unsigned, ArrayRef<int64_t>, Type, VectorType>
getVectorTypeInfo(Type vectorTypeRaw) {
  VectorType vectorType = llvm::dyn_cast<VectorType>(vectorTypeRaw);
  NPUVectorType npuVectorType = llvm::dyn_cast<NPUVectorType>(vectorTypeRaw);
  unsigned rank = vectorType ? vectorType.getRank() : npuVectorType.getRank();
  ArrayRef<int64_t> shape = vectorType ? vectorType.getShape() : npuVectorType.getShape();
  Type elementType = vectorType ? vectorType.getElementType() : npuVectorType.getElementType();
  VectorType tempVectorType = vectorType ? vectorType : VectorType::get(shape, elementType);
  return std::make_tuple(rank, shape, elementType, tempVectorType);
}

// Helper function to setup permutation map attributes
static LogicalResult setupPermutationMapAttrs(OpAsmParser &parser, OperationState &result,
                                               ShapedType shapedType, VectorType tempVectorType,
                                               StringRef permMapAttrName, StringRef inBoundsAttrName,
                                               AffineMap &permMap) {
  auto &builder = parser.getBuilder();
  Attribute permMapAttr = result.attributes.get(permMapAttrName);
  if (!permMapAttr) {
    permMap = vector::getTransferMinorIdentityMap(shapedType, tempVectorType);
    result.attributes.set(permMapAttrName, AffineMapAttr::get(permMap));
  } else {
    permMap = llvm::cast<AffineMapAttr>(permMapAttr).getValue();
  }
  Attribute inBoundsAttr = result.attributes.get(inBoundsAttrName);
  if (!inBoundsAttr) {
    result.addAttribute(inBoundsAttrName,
                        builder.getBoolArrayAttr(SmallVector<bool>(permMap.getNumResults(), false)));
  }
  return success();
}

// Helper function to resolve mask operand
static LogicalResult resolveMaskOperand(OpAsmParser &parser, OpAsmParser::UnresolvedOperand &maskInfo,
                                        ShapedType shapedType, unsigned vectorRank, AffineMap permMap,
                                        VectorType tempVectorType, OperationState &result, SMLoc typesLoc) {
  if (llvm::dyn_cast<VectorType>(shapedType.getElementType()))
    return parser.emitError(maskInfo.location, "does not support masks with vector element type");
  if (vectorRank != permMap.getNumResults()) {
    return parser.emitError(typesLoc,
                            "expected the same rank for the vector and the "
                            "results of the permutation map");
  }
  auto maskType = vector::inferTransferOpMaskType(tempVectorType, permMap);
  return parser.resolveOperand(maskInfo, maskType, result.operands);
}

// Helper function to parse TransferRead operands syntax
static ParseResult parseTransferReadOperands(OpAsmParser &parser,
                                             OpAsmParser::UnresolvedOperand &sourceInfo,
                                             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indexInfo,
                                             SmallVectorImpl<OpAsmParser::UnresolvedOperand> &dynamicSizesInfo,
                                             OpAsmParser::UnresolvedOperand &paddingInfo,
                                             OpAsmParser::UnresolvedOperand &maskInfo,
                                             OpAsmParser::UnresolvedOperand &maxSizeInfo,
                                             ParseResult &hasMask,
                                             ParseResult &hasMaxSize) {
  if (parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square))
    return failure();
  if (succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(dynamicSizesInfo) || parser.parseRSquare())
      return failure();
  }
  hasMaxSize = parser.parseOptionalLSquare();
  if (hasMaxSize.succeeded()) {
    if (parser.parseOperand(maxSizeInfo) || parser.parseRSquare())
      return failure();
  }
  if (parser.parseComma() || parser.parseOperand(paddingInfo))
    return failure();
  hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded() && parser.parseOperand(maskInfo))
    return failure();
  return success();
}

// Helper function to validate transfer op types
static LogicalResult validateTransferTypes(OpAsmParser &parser, SMLoc typesLoc,
                                           ArrayRef<Type> types, ShapedType &shapedType,
                                           Type &vectorTypeRaw, bool isRead) {
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  Type type0 = isRead ? types[0] : types[1];
  Type type1 = isRead ? types[1] : types[0];

  shapedType = llvm::dyn_cast<ShapedType>(type0);
  if (!shapedType || !llvm::isa<MemRefType, RankedTensorType>(shapedType))
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");

  vectorTypeRaw = type1;
  if (!llvm::isa<VectorType, NPUVectorType>(vectorTypeRaw))
    return parser.emitError(typesLoc, "requires vector or npuvector type");

  return success();
}

// Helper function to resolve TransferRead operands
static LogicalResult resolveTransferReadOperands(OpAsmParser &parser, OperationState &result,
                                                 OpAsmParser::UnresolvedOperand &sourceInfo,
                                                 ArrayRef<OpAsmParser::UnresolvedOperand> indexInfo,
                                                 OpAsmParser::UnresolvedOperand &paddingInfo,
                                                 ArrayRef<OpAsmParser::UnresolvedOperand> dynamicSizesInfo,
                                                 OpAsmParser::UnresolvedOperand &maxSizeInfo,
                                                 bool hasMaxSize,
                                                 ShapedType shapedType, Type indexType) {
  if (parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands) ||
      parser.resolveOperand(paddingInfo, shapedType.getElementType(), result.operands))
    return failure();
  if (parser.resolveOperands(dynamicSizesInfo, indexType, result.operands))
    return failure();
  if (hasMaxSize && parser.resolveOperand(maxSizeInfo, indexType, result.operands))
    return failure();
  return success();
}

// Helper function to parse TransferWrite operands syntax
static ParseResult parseTransferWriteOperands(OpAsmParser &parser,
                                              OpAsmParser::UnresolvedOperand &vectorInfo,
                                              OpAsmParser::UnresolvedOperand &sourceInfo,
                                              SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indexInfo,
                                              OpAsmParser::UnresolvedOperand &maskInfo,
                                              ParseResult &hasMask) {
  if (parser.parseOperand(vectorInfo) || parser.parseComma() || parser.parseOperand(sourceInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square))
    return failure();
  hasMask = parser.parseOptionalComma();
  if (hasMask.succeeded() && parser.parseOperand(maskInfo))
    return failure();
  return success();
}

// Helper function to resolve TransferWrite operands
static LogicalResult resolveTransferWriteOperands(OpAsmParser &parser, OperationState &result,
                                                  OpAsmParser::UnresolvedOperand &vectorInfo,
                                                  OpAsmParser::UnresolvedOperand &sourceInfo,
                                                  ArrayRef<OpAsmParser::UnresolvedOperand> indexInfo,
                                                  Type vectorTypeRaw, ShapedType shapedType, Type indexType) {
  return parser.resolveOperand(vectorInfo, vectorTypeRaw, result.operands) ||
         parser.resolveOperand(sourceInfo, shapedType, result.operands) ||
         parser.resolveOperands(indexInfo, indexType, result.operands) ? failure() : success();
}

static void printTransferAttrs(OpAsmPrinter &p, VectorTransferOpInterface op) {
  SmallVector<StringRef, 3> elidedAttrs;
  // Both TransferReadOp and TransferWriteOp use the same attribute name
  elidedAttrs.push_back("operandSegmentSizes");
  if (op.getPermutationMap().isMinorIdentity()) {
    elidedAttrs.push_back(op.getPermutationMapAttrName());
  }
  // Elide in_bounds attribute if all dims are out-of-bounds.
  if (llvm::none_of(op.getInBoundsValues(), [](bool b) { return b; })) {
    elidedAttrs.push_back(op.getInBoundsAttrName());
  }
  p.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
}

void TransferReadOp::print(OpAsmPrinter &p) {
  p << " " << getSource() << "[" << getIndices() << "]";
  // Print dynamicSizes if present
  if (!getDynamicSizes().empty()) {
    p << "[" << getDynamicSizes() << "]";
  }
  if (getMaxSize()) {
    p << "[" << getMaxSize() << "]";
  }
  p << ", " << getPadding();
  if (getMask()) p << ", " << getMask();
  printTransferAttrs(p, *this);
  p << " : " << getShapedType() << ", " << getResult().getType();
}

ParseResult TransferReadOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SMLoc typesLoc;
  OpAsmParser::UnresolvedOperand sourceInfo, paddingInfo, maskInfo, maxSizeInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> dynamicSizesInfo;
  SmallVector<Type, 2> types;
  ParseResult hasMask, hasMaxSize;

  // Parse operands syntax
  if (failed(parseTransferReadOperands(parser, sourceInfo, indexInfo, dynamicSizesInfo,
                                       paddingInfo, maskInfo, maxSizeInfo, hasMask, hasMaxSize)))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes) || parser.getCurrentLocation(&typesLoc) ||
      parser.parseColonTypeList(types))
    return failure();

  // Validate types
  ShapedType shapedType;
  Type vectorTypeRaw;
  if (failed(validateTransferTypes(parser, typesLoc, types, shapedType, vectorTypeRaw, true)))
    return failure();

  // Setup attributes
  auto [vectorRank, vectorShape, elementType, tempVectorType] = getVectorTypeInfo(vectorTypeRaw);
  AffineMap permMap;
  if (failed(setupPermutationMapAttrs(parser, result, shapedType, tempVectorType,
                                      TransferReadOp::getPermutationMapAttrName(result.name),
                                      TransferReadOp::getInBoundsAttrName(result.name), permMap)))
    return failure();

  // Resolve operands
  auto indexType = builder.getIndexType();
  if (failed(resolveTransferReadOperands(parser, result, sourceInfo, indexInfo, paddingInfo,
                                         dynamicSizesInfo, maxSizeInfo, hasMaxSize.succeeded(),
                                         shapedType, indexType)))
    return failure();

  if (hasMask.succeeded() && failed(resolveMaskOperand(parser, maskInfo, shapedType, vectorRank,
                                                        permMap, tempVectorType, result, typesLoc)))
    return failure();

  result.addAttribute(TransferReadOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, static_cast<int32_t>(indexInfo.size()), 1,
                                                    static_cast<int32_t>(hasMask.succeeded()),
                                                    static_cast<int32_t>(dynamicSizesInfo.size()),
                                                    static_cast<int32_t>(hasMaxSize.succeeded())}));
  return parser.addTypeToList(vectorTypeRaw, result.types);
}

LogicalResult TransferReadOp::verify() {
  // Consistency of elemental types in source and vector.
  ShapedType shapedType = getShapedType();
  ShapedType vectorShapedType = getVectorType();  // Use ShapedType for result
  auto paddingType = getPadding().getType();
  auto permutationMap = getPermutationMap();
  auto sourceElementType = shapedType.getElementType();

  if (static_cast<int64_t>(getIndices().size()) != shapedType.getRank())
    return emitOpError("requires ") << shapedType.getRank() << " indices";

  // Verify padding type matches element type
  if (paddingType != sourceElementType) {
    return emitOpError("padding type must match source element type");
  }

  // Check that the permutation map has the correct number of results
  if (permutationMap.getNumResults() != static_cast<unsigned>(vectorShapedType.getRank())) {
    return emitOpError("permutation map result count must match vector rank");
  }

  // Verify in_bounds attribute length matches vector rank
  auto inBounds = getInBounds();
  if (inBounds.size() != static_cast<unsigned>(vectorShapedType.getRank())) {
    return emitOpError("in_bounds array length must match vector rank");
  }

  return success();
}

void TransferReadOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable(), SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// TransferWriteOp
//===----------------------------------------------------------------------===//

void TransferWriteOp::print(OpAsmPrinter &p) {
  p << " " << getVector() << ", " << getSource() << "[" << getIndices() << "]";
  if (getMask()) p << ", " << getMask();
  printTransferAttrs(p, *this);
  p << " : " << getVector().getType() << ", " << getShapedType();
  if (getResult()) p << " -> " << getResult().getType();
}

ParseResult TransferWriteOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  SMLoc typesLoc;
  OpAsmParser::UnresolvedOperand vectorInfo, sourceInfo, maskInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  SmallVector<Type, 2> types;
  ParseResult hasMask;

  // Parse operands syntax
  if (failed(parseTransferWriteOperands(parser, vectorInfo, sourceInfo, indexInfo, maskInfo, hasMask)))
    return failure();

  if (parser.parseOptionalAttrDict(result.attributes) || parser.getCurrentLocation(&typesLoc) ||
      parser.parseColonTypeList(types))
    return failure();

  // Validate types
  ShapedType shapedType;
  Type vectorTypeRaw;
  if (failed(validateTransferTypes(parser, typesLoc, types, shapedType, vectorTypeRaw, false)))
    return failure();

  // Setup attributes
  auto [vectorRank, vectorShape, elementType, tempVectorType] = getVectorTypeInfo(vectorTypeRaw);
  AffineMap permMap;
  if (failed(setupPermutationMapAttrs(parser, result, shapedType, tempVectorType,
                                      TransferWriteOp::getPermutationMapAttrName(result.name),
                                      TransferWriteOp::getInBoundsAttrName(result.name), permMap)))
    return failure();

  // Resolve operands
  auto indexType = builder.getIndexType();
  if (failed(resolveTransferWriteOperands(parser, result, vectorInfo, sourceInfo, indexInfo,
                                          vectorTypeRaw, shapedType, indexType)))
    return failure();

  if (hasMask.succeeded() && failed(resolveMaskOperand(parser, maskInfo, shapedType, vectorRank,
                                                        permMap, tempVectorType, result, typesLoc)))
    return failure();

  result.addAttribute(TransferWriteOp::getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr({1, 1, static_cast<int32_t>(indexInfo.size()),
                                                    static_cast<int32_t>(hasMask.succeeded())}));
  return failure(llvm::isa<RankedTensorType>(shapedType) && parser.addTypeToList(shapedType, result.types));
}

LogicalResult TransferWriteOp::verify() {
  // Consistency of elemental types in shape and vector.
  ShapedType shapedType = getShapedType();
  Type vectorTypeRaw = getVector().getType();
  VectorType vectorType = llvm::dyn_cast<VectorType>(vectorTypeRaw);
  NPUVectorType npuVectorType = llvm::dyn_cast<NPUVectorType>(vectorTypeRaw);

  // Get rank - support both VectorType and NPUVectorType
  unsigned vectorRank = vectorType ? vectorType.getRank() : npuVectorType.getRank();

  auto permutationMap = getPermutationMap();

  if (static_cast<int64_t>(getIndices().size()) != shapedType.getRank())
    return emitOpError("requires ") << shapedType.getRank() << " indices";

  // Check that the permutation map has the correct number of results
  if (permutationMap.getNumResults() != vectorRank) {
    return emitOpError("permutation map result count must match vector rank");
  }

  // Verify in_bounds attribute length matches vector rank
  auto inBounds = getInBounds();
  if (inBounds.size() != vectorRank) {
    return emitOpError("in_bounds array length must match vector rank");
  }

  return success();
}

void TransferWriteOp::getEffects(SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects) {
  if (llvm::isa<MemRefType>(getShapedType()))
    effects.emplace_back(MemoryEffects::Write::get(), &getSourceMutable(), SideEffects::DefaultResource::get());
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

LogicalResult ReductionOp::verify() {
  auto vectorType = getSourceVectorType();

  // Verify that the result type matches the element type
  auto elementType = vectorType.getElementType();
  if (getDest().getType() != elementType) {
    return emitOpError("result type must match vector element type");
  }

  // Verify accumulator type if present
  Value acc = getAcc();
  if (acc) {
    if (acc.getType() != elementType) {
      return emitOpError("accumulator type must match vector element type");
    }
  }

  return success();
}

void TransferReadOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // No canonicalization patterns for now
}

OpFoldResult TransferReadOp::fold(FoldAdaptor adaptor) {
  // No folding for now
  return OpFoldResult();
}

void TransferWriteOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // No canonicalization patterns for now
}

LogicalResult TransferWriteOp::fold(FoldAdaptor adaptor, SmallVectorImpl<OpFoldResult> &results) {
  // No folding for now
  return failure();
}

void ReductionOp::getCanonicalizationPatterns(RewritePatternSet &results, MLIRContext *context) {
  // No canonicalization patterns for now
}

//===----------------------------------------------------------------------===//
// VectorTransferOpInterface implementation
//===----------------------------------------------------------------------===//

VectorType TransferReadOp::getVectorType() {
  // Override to handle both VectorType and NPUVectorType
  Type resultType = getResult().getType();
  if (auto vectorType = llvm::dyn_cast<VectorType>(resultType)) return vectorType;
  if (auto npuVectorType = llvm::dyn_cast<NPUVectorType>(resultType)) {
    // Convert NPUVectorType to VectorType for interface compatibility
    return VectorType::get(npuVectorType.getShape(), npuVectorType.getElementType());
  }
  llvm_unreachable("expected VectorType or NPUVectorType");
}

VectorType TransferWriteOp::getVectorType() {
  // Override to handle both VectorType and NPUVectorType
  Type vectorTypeRaw = getVector().getType();
  if (auto vectorType = llvm::dyn_cast<VectorType>(vectorTypeRaw)) return vectorType;
  if (auto npuVectorType = llvm::dyn_cast<NPUVectorType>(vectorTypeRaw)) {
    // Convert NPUVectorType to VectorType for interface compatibility
    return VectorType::get(npuVectorType.getShape(), npuVectorType.getElementType());
  }
  llvm_unreachable("expected VectorType or NPUVectorType");
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.cpp.inc"

// Include builder implementations
#define GET_OP_IMPL
// NOLINTNEXTLINE(build/include)
#include "akg/Dialect/NPUVector/IR/NPUVectorOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Manual builder implementations
//===----------------------------------------------------------------------===//

// Manual implementation of TransferWriteOp::build without mask parameter
// This builder is declared in the generated header but not implemented
// because TableGen adds mask parameter automatically
void npuvector::TransferWriteOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState,
                                       ::mlir::Value vector, ::mlir::Value source, ::mlir::ValueRange indices,
                                       ::mlir::AffineMapAttr permutationMapAttr, ::mlir::ArrayAttr inBoundsAttr) {
  // Call the generated builder with mask = Value() (empty)
  build(odsBuilder, odsState, /*result=*/::mlir::Type(), vector, source, indices, permutationMapAttr,
        /*mask=*/::mlir::Value(), inBoundsAttr);
}

//===----------------------------------------------------------------------===//
// BroadcastOp
//===----------------------------------------------------------------------===//

LogicalResult BroadcastOp::verify() {
  // Get source type safely
  Value source = getSource();
  if (!source) {
    return emitOpError("source operand is null");
  }

  Type sourceType = source.getType();
  if (!sourceType) {
    return emitOpError("source type is null");
  }

  // Get result type safely
  Value result = getResult();
  if (!result) {
    return emitOpError("result is null");
  }

  Type resultTypeRaw = result.getType();
  if (!resultTypeRaw) {
    return emitOpError("result type is null");
  }

  // Try to cast to NPUVectorType
  auto resultType = resultTypeRaw.dyn_cast<NPUVectorType>();
  if (!resultType) {
    return emitOpError("result type must be NPUVectorType, got ") << resultTypeRaw;
  }

  // Get element types
  Type sourceElemType;
  if (sourceType.isa<IntegerType, FloatType, IndexType>()) {
    // Scalar type
    sourceElemType = sourceType;
  } else if (auto shapedType = sourceType.dyn_cast<ShapedType>()) {
    // Shaped type (vector, tensor, memref)
    sourceElemType = shapedType.getElementType();
  } else {
    return emitOpError("source type must be scalar or shaped type, got ") << sourceType;
  }

  Type resultElemType = resultType.getElementType();

  // Check element type compatibility
  if (sourceElemType != resultElemType) {
    return emitOpError("source element type must match result element type, got ")
           << sourceElemType << " and " << resultElemType;
  }

  return success();
}

OpFoldResult BroadcastOp::fold(FoldAdaptor adaptor) {
  // If source is a constant attribute, fold to dense constant
  if (auto attr = adaptor.getSource().dyn_cast_or_null<Attribute>()) {
    auto resultType = getResult().getType().cast<NPUVectorType>();

    // Broadcast scalar constant
    if (attr.isa<IntegerAttr, FloatAttr>()) {
      // DenseElementsAttr::get() accepts ShapedType
      // NPUVectorType implements ShapedTypeInterface, so this works
      return DenseElementsAttr::get(resultType, attr);
    }

    // Broadcasting dense constant vectors is not implemented yet
    // TODO(username): implement vector-to-vector broadcast folding
    (void)attr;  // Suppress unused variable warning
  }

  // If source itself is a broadcast, merge them
  if (auto sourceBroadcast = getSource().getDefiningOp<BroadcastOp>()) {
    // broadcast(broadcast(x)) → broadcast(x)
    getSourceMutable().assign(sourceBroadcast.getSource());
    return getResult();
  }

  return OpFoldResult();
}

void BroadcastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  // Canonicalization patterns can be added here if needed
}

//===----------------------------------------------------------------------===//
// DimOp
//===----------------------------------------------------------------------===//

OpFoldResult DimOp::fold(FoldAdaptor adaptor) {
  auto vecType = getSource().getType().cast<NPUVectorType>();

  // Get the dimension index being queried
  std::optional<int64_t> dimIndex = getConstantIntValue(getIndex());
  if (!dimIndex) return {};

  // Check if index is out of bounds
  if (*dimIndex < 0 || *dimIndex >= static_cast<int64_t>(vecType.getRank())) {
    return {};
  }

  // Case 1: Static dimension - return constant
  ArrayRef<int64_t> shape = vecType.getShape();
  if (!ShapedType::isDynamic(shape[*dimIndex])) {
    int64_t staticSize = shape[*dimIndex];
    return IntegerAttr::get(IndexType::get(getContext()), staticSize);
  }

  // Case 2: Dynamic dimension - get from defining operation
  Operation *defOp = getSource().getDefiningOp();
  if (!defOp) return {};

  // Get from transfer_read
  if (auto transferRead = dyn_cast<TransferReadOp>(defOp)) {
    ValueRange dynamicSizes = transferRead.getDynamicSizes();
    if (dynamicSizes.empty()) return {};

    // Calculate which dynamic dimension this is
    unsigned dynamicIdx = 0;
    for (unsigned i = 0; i < *dimIndex; ++i) {
      if (ShapedType::isDynamic(shape[i])) {
        dynamicIdx++;
      }
    }

    if (dynamicIdx < dynamicSizes.size()) {
      return dynamicSizes[dynamicIdx];
    }
  }

  // Get from broadcast
  if (auto broadcast = dyn_cast<BroadcastOp>(defOp)) {
    ValueRange dynamicSizes = broadcast.getDynamicSizes();
    if (dynamicSizes.empty()) return {};

    // Calculate which dynamic dimension this is
    unsigned dynamicIdx = 0;
    for (unsigned i = 0; i < *dimIndex; ++i) {
      if (ShapedType::isDynamic(shape[i])) {
        dynamicIdx++;
      }
    }

    if (dynamicIdx < dynamicSizes.size()) {
      return dynamicSizes[dynamicIdx];
    }
  }

  // Cannot fold in other cases
  return {};
}
