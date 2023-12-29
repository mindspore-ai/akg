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

#include "akg/Dialect/Fusion/IR/Fusion.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;
using namespace mlir::fusion;

namespace saturated_arith {
struct Wrapper {
  static Wrapper stride(int64_t v) { return (ShapedType::isDynamic(v)) ? Wrapper{true, 0} : Wrapper{false, v}; }
  static Wrapper offset(int64_t v) { return (ShapedType::isDynamic(v)) ? Wrapper{true, 0} : Wrapper{false, v}; }
  static Wrapper size(int64_t v) { return (ShapedType::isDynamic(v)) ? Wrapper{true, 0} : Wrapper{false, v}; }
  int64_t asOffset() const { return saturated ? ShapedType::kDynamic : v; }
  int64_t asSize() const { return saturated ? ShapedType::kDynamic : v; }
  int64_t asStride() const { return saturated ? ShapedType::kDynamic : v; }
  bool operator==(const Wrapper &other) const {
    return (saturated && other.saturated) || (!saturated && !other.saturated && v == other.v);
  }
  bool operator!=(const Wrapper &other) const { return !(*this == other); }
  Wrapper operator+(const Wrapper &other) const {
    if (saturated || other.saturated) {
      return Wrapper{true, 0};
    }
    return Wrapper{false, other.v + v};
  }
  Wrapper operator*(const Wrapper &other) const {
    if (saturated || other.saturated) {
      return Wrapper{true, 0};
    }
    return Wrapper{false, other.v * v};
  }
  bool saturated;
  int64_t v;
};
}  // namespace saturated_arith

template <typename OpTy>
static LogicalResult produceSubViewErrorMsg(SliceVerificationResult result, OpTy op, Type expectedType) {
  auto memrefType = expectedType.cast<ShapedType>();
  switch (result) {
    case SliceVerificationResult::Success:
      return success();
    case SliceVerificationResult::RankTooLarge:
      return op.emitError("expected result rank to be smaller or equal to ") << "the source rank. ";
    case SliceVerificationResult::SizeMismatch:
      return op.emitError("expected result type to be ")
             << expectedType << " or a rank-reduced version. (mismatch of result sizes) ";
    case SliceVerificationResult::ElemTypeMismatch:
      return op.emitError("expected result element type to be ") << memrefType.getElementType();
    case SliceVerificationResult::MemSpaceMismatch:
      return op.emitError("expected result and source memory spaces to match.");
    case SliceVerificationResult::LayoutMismatch:
      return op.emitError("expected result type to be ")
             << expectedType << " or a rank-reduced version. (mismatch of result layout) ";
    default:
      llvm_unreachable("unexpected subview verification result");
  }
  return success();
}

/// Return a map with key being elements in `vals` and data being number of
/// occurences of it. Use std::map, since the `vals` here are strides and the
/// dynamic stride value is the same as the tombstone value for
/// `DenseMap<int64_t>`.
static std::map<int64_t, unsigned> getNumOccurences(ArrayRef<int64_t> vals) {
  std::map<int64_t, unsigned> numOccurences;
  for (auto val : vals) {
    numOccurences[val]++;
  }
  return numOccurences;
}

/// Given the `originalType` and a `candidateReducedType` whose shape is assumed
/// to be a subset of `originalType` with some `1` entries erased, return the
/// set of indices that specifies which of the entries of `originalShape` are
/// dropped to obtain `reducedShape`.
/// This accounts for cases where there are multiple unit-dims, but only a
/// subset of those are dropped. For MemRefTypes these can be disambiguated
/// using the strides. If a dimension is dropped the stride must be dropped too.
static std::optional<llvm::SmallBitVector> computeMemRefRankReductionMask(const MemRefType originalType,
                                                                          const MemRefType reducedType,
                                                                          ArrayRef<OpFoldResult> sizes) {
  llvm::SmallBitVector unusedDims(originalType.getRank());
  if (originalType.getRank() == reducedType.getRank()) {
    return unusedDims;
  }

  for (const auto &dim : llvm::enumerate(sizes)) {
    if (auto attr = dim.value().dyn_cast<Attribute>()) {
      (void)unusedDims.set((unsigned int)dim.index());
    }
  }
  // Early exit for the case where the number of unused dims matches the number
  // of ranks reduced.
  if (static_cast<int64_t>(unusedDims.count()) + reducedType.getRank() == originalType.getRank()) {
    return unusedDims;
  }

  SmallVector<int64_t> originalStrides;
  SmallVector<int64_t> candidateStrides;
  int64_t originalOffset;
  int64_t candidateOffset;
  if (failed(getStridesAndOffset(originalType, originalStrides, originalOffset)) ||
      failed(getStridesAndOffset(reducedType, candidateStrides, candidateOffset))) {
    return std::nullopt;
  }

  // For memrefs, a dimension is truly dropped if its corresponding stride is
  // also dropped. This is particularly important when more than one of the dims
  // is 1. Track the number of occurences of the strides in the original type
  // and the candidate type. For each unused dim that stride should not be
  // present in the candidate type. Note that there could be multiple dimensions
  // that have the same size. We dont need to exactly figure out which dim
  // corresponds to which stride, we just need to verify that the number of
  // reptitions of a stride in the original + number of unused dims with that
  // stride == number of repititions of a stride in the candidate.
  std::map<int64_t, unsigned> currUnaccountedStrides = getNumOccurences(originalStrides);
  std::map<int64_t, unsigned> candidateStridesNumOccurences = getNumOccurences(candidateStrides);
  for (size_t dim = 0, e = unusedDims.size(); dim != e; ++dim) {
    if (!unusedDims.test((unsigned int)dim)) {
      continue;
    }
    int64_t originalStride = originalStrides[dim];
    if (currUnaccountedStrides[originalStride] > candidateStridesNumOccurences[originalStride]) {
      // This dim can be treated as dropped.
      currUnaccountedStrides[originalStride]--;
      continue;
    }
    if (currUnaccountedStrides[originalStride] == candidateStridesNumOccurences[originalStride]) {
      // The stride for this is not dropped. Keep as is.
      (void)unusedDims.reset(dim);
      continue;
    }
    if (currUnaccountedStrides[originalStride] < candidateStridesNumOccurences[originalStride]) {
      // This should never happen. Cant have a stride in the reduced rank type
      // that wasnt in the original one.
      return std::nullopt;
    }
  }

  if ((int64_t)unusedDims.count() + reducedType.getRank() != originalType.getRank()) {
    return std::nullopt;
  }
  return unusedDims;
}

/// Return true if t1 and t2 have equal offsets (both dynamic or of same
/// static value).
static bool haveCompatibleOffsets(MemRefType t1, MemRefType t2) {
  int64_t t1Offset;
  int64_t t2Offset;
  SmallVector<int64_t> t1Strides;
  SmallVector<int64_t> t2Strides;
  auto res1 = getStridesAndOffset(t1, t1Strides, t1Offset);
  auto res2 = getStridesAndOffset(t2, t2Strides, t2Offset);
  return succeeded(res1) && succeeded(res2) && t1Offset == t2Offset;
}

/// Checks if `original` Type type can be rank reduced to `reduced` type.
/// This function is slight variant of `is subsequence` algorithm where
/// not matching dimension must be 1.
static SliceVerificationResult isRankReducedMemRefType(const MemRefType originalType,
                                                       const MemRefType &candidateRankReducedType,
                                                       ArrayRef<OpFoldResult> sizes) {
  auto partialRes = isRankReducedType(originalType, candidateRankReducedType);
  if (partialRes != SliceVerificationResult::Success) {
    return partialRes;
  }

  auto optionalUnusedDimsMask = computeMemRefRankReductionMask(originalType, candidateRankReducedType, sizes);
  // Sizes cannot be matched in case empty vector is returned.
  if (!optionalUnusedDimsMask) {
    return SliceVerificationResult::LayoutMismatch;
  }

  if (originalType.getMemorySpace() != candidateRankReducedType.getMemorySpace()) {
    return SliceVerificationResult::MemSpaceMismatch;
  }

  // No amount of stride dropping can reconcile incompatible offsets.
  if (!haveCompatibleOffsets(originalType, candidateRankReducedType)) {
    return SliceVerificationResult::LayoutMismatch;
  }

  return SliceVerificationResult::Success;
}

// ===----------------------------------------------------------------------=== //
// LoadOp
// ===----------------------------------------------------------------------=== //
::mlir::ParseResult LoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  // parse memref and indices
  OpAsmParser::UnresolvedOperand memrefInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  if (parser.parseOperand(memrefInfo) || parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square)) {
    return failure();
  }

  // parse optional padding
  OpAsmParser::UnresolvedOperand paddingInfo;
  ParseResult hasPadding = parser.parseOptionalComma();
  if (hasPadding.succeeded()) {
    if (parser.parseOperand(paddingInfo)) {
      return failure();
    }
  }

  // parse optional attr dict and types
  SMLoc typesLoc;
  SmallVector<Type, 2> types;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.getCurrentLocation(&typesLoc) ||
      parser.parseColonTypeList(types)) {
    return failure();
  }

  // check validity of type
  if (types.size() != 2) {
    return parser.emitError(typesLoc, "requires two types");
  }

  auto memrefType = types[0].dyn_cast<MemRefType>();
  if (!memrefType) {
    return parser.emitError(typesLoc, "requires memref type");
  }

  if (!types[1].isa<VectorType>() && !types[1].isIntOrFloat()) {
    return parser.emitError(typesLoc, "requires scalar type or vector type");
  }

  // resolve operands
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  if (parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands)) {
    return failure();
  }

  if (hasPadding.succeeded()) {
    if (parser.resolveOperand(paddingInfo, memrefType.getElementType(), result.operands)) {
      return failure();
    }
  }

  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(

                        {1, static_cast<int32_t>(indexInfo.size()), static_cast<int32_t>(hasPadding.succeeded())}));

  return parser.addTypeToList(types[1], result.types);
}

void LoadOp::print(::mlir::OpAsmPrinter &p) {
  p << " " << getMemref() << "[" << getIndices() << "]";
  if (getPadding()) {
    p << ", " << getPadding() << " ";
  }
  p.printOptionalAttrDict(this->getOperation()->getAttrs(), {LoadOp::getOperandSegmentSizeAttr()});
  p << " : " << getMemRefType() << ", " << getResult().getType();
}

LogicalResult LoadOp::verify() {
  if (getIndices().size() != (size_t)getMemRefType().getRank()) {
    return emitOpError("incorrect number of indices for load");
  }
  return success();
}

OpFoldResult LoadOp::fold(FoldAdaptor) {
  /// load(memrefcast) -> load
  if (succeeded(memref::foldMemRefCast(*this))) {
    return getResult();
  }
  return OpFoldResult();
}

// ===----------------------------------------------------------------------=== //
// MultiLoadOp
// ===----------------------------------------------------------------------=== //
::mlir::ParseResult MultiLoadOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  // parse memref and indices
  OpAsmParser::UnresolvedOperand memrefInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  if (parser.parseOperand(memrefInfo) || parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square)) {
    return failure();
  }

  // parse optional padding
  OpAsmParser::UnresolvedOperand paddingInfo;
  ParseResult hasPadding = parser.parseOptionalComma();
  if (hasPadding.succeeded()) {
    if (parser.parseOperand(paddingInfo)) {
      return failure();
    }
  }

  // parse optional attr dict and types
  SMLoc typesLoc;
  SmallVector<Type, 2> types;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.getCurrentLocation(&typesLoc) ||
      parser.parseColonTypeList(types)) {
    return failure();
  }

  // check validity of type
  if (types.size() != 2) {
    return parser.emitError(typesLoc, "requires two types");
  }

  auto memrefType = types[0].dyn_cast<MemRefType>();
  if (!memrefType) {
    return parser.emitError(typesLoc, "requires memref type");
  }

  if (!types[1].isa<VectorType>() && !types[1].isIntOrFloat()) {
    return parser.emitError(typesLoc, "requires scalar type or vector type");
  }

  // resolve operands
  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  if (parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands))
    return failure();

  if (hasPadding.succeeded()) {
    if (parser.resolveOperand(paddingInfo, memrefType.getElementType(), result.operands)) {
      return failure();
    }
  }

  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                        {1, static_cast<int32_t>(indexInfo.size()), static_cast<int32_t>(hasPadding.succeeded())}));
  return parser.addTypeToList(types[1], result.types);
}

void MultiLoadOp::print(::mlir::OpAsmPrinter &p) {
  p << " " << getMemref() << "[" << getIndices() << "]";
  if (getPadding()) {
    p << ", " << getPadding() << "\n";
  }
  p.printOptionalAttrDict(this->getOperation()->getAttrs(), {LoadOp::getOperandSegmentSizeAttr()});
  p << " : " << getMemRefType() << ", " << getResult().getType();
}

LogicalResult MultiLoadOp::verify() {
  if (getIndices().size() != (size_t)getMemRefType().getRank()) {
    return emitOpError("incorrect number of indices for load");
  }
  return success();
}

OpFoldResult MultiLoadOp::fold(FoldAdaptor) {
  /// load(memrefcast) -> load
  if (succeeded(memref::foldMemRefCast(*this))) {
    return getResult();
  }
  return OpFoldResult();
}

// ===----------------------------------------------------------------------=== //
// SubViewOp
// ===----------------------------------------------------------------------=== //
/// Verifier for SubViewOp.
LogicalResult SubViewOp::verify() {
  MemRefType baseType = getSourceType();
  MemRefType subViewType = getType();
  // The base memref and the view memref should be in the same memory space.
  if (baseType.getMemorySpace() != subViewType.getMemorySpace()) {
    return emitError(
             "different memory spaces specified for base memref "
             "type ")
           << baseType << " and subview memref type " << subViewType;
  }

  // Verify that the base memref type has a strided layout map.
  if (!isStrided(baseType)) {
    return emitError("base type ") << baseType << " is not strided";
  }

  // Verify result type against inferred type.
  auto expectedType = SubViewOp::inferResultType(baseType, getStaticOffsets(), getStaticSizes(), getStaticStrides());

  auto result = isRankReducedMemRefType(expectedType.cast<MemRefType>(), subViewType, getMixedSizes());
  return produceSubViewErrorMsg(result, *this, expectedType);
}

void SubViewOp::getAsmResultNames(function_ref<void(Value, StringRef)> setNameFn) { setNameFn(getResult(), "subview"); }

/// For ViewLikeOpInterface.
Value SubViewOp::getViewSource() { return getSource(); }

// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void SubViewOp::build(OpBuilder &b, OperationState &result, MemRefType resultType, Value source,
                      ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
                      ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicOffsets;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceMemRefType = source.getType().cast<MemRefType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType =
      SubViewOp::inferResultType(sourceMemRefType, staticOffsets, staticSizes, staticStrides).cast<MemRefType>();
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes, dynamicStrides,
        b.getDenseI64ArrayAttr(staticOffsets), b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

/// A subview result type can be fully inferred from the source type and the
/// static representation of offsets, sizes and strides. Special sentinels
/// encode the dynamic case.
Type SubViewOp::inferResultType(MemRefType sourceMemRefType, ArrayRef<int64_t> staticOffsets,
                                ArrayRef<int64_t> staticSizes, ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceMemRefType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");

  // Extract source offset and strides.
  auto [sourceStrides, sourceOffset] = getStridesAndOffset(sourceMemRefType);

  // Compute target offset whose value is:
  //   `sourceOffset + sum_i(staticOffset_i * sourceStrides_i)`.
  int64_t targetOffset = sourceOffset;
  for (auto it : llvm::zip(staticOffsets, sourceStrides)) {
    auto staticOffset = std::get<0>(it), targetStride = std::get<1>(it);
    using saturated_arith::Wrapper;
    targetOffset =
      (Wrapper::offset(targetOffset) + Wrapper::offset(staticOffset) * Wrapper::stride(targetStride)).asOffset();
  }

  // Compute target stride whose value is:
  //   `sourceStrides_i * staticStrides_i`.
  SmallVector<int64_t, 4> targetStrides;
  targetStrides.reserve(staticOffsets.size());
  for (auto it : llvm::zip(sourceStrides, staticStrides)) {
    auto sourceStride = std::get<0>(it), staticStride = std::get<1>(it);
    using saturated_arith::Wrapper;
    targetStrides.push_back((Wrapper::stride(sourceStride) * Wrapper::stride(staticStride)).asStride());
  }

  // The type is now known.
  return MemRefType::get(staticSizes, sourceMemRefType.getElementType(),
                         StridedLayoutAttr::get(sourceMemRefType.getContext(), targetOffset, targetStrides),
                         sourceMemRefType.getMemorySpace());
}

Type SubViewOp::inferResultType(MemRefType sourceMemRefType, ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides) {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<Value> dynamicOffsets;
  SmallVector<Value> dynamicSizes;
  SmallVector<Value> dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubViewOp::inferResultType(sourceMemRefType, staticOffsets, staticSizes, staticStrides);
}

// ===----------------------------------------------------------------------=== //
// BroadcastOp
// ===----------------------------------------------------------------------=== //
LogicalResult BroadcastOp::verify() {
  std::pair<int, int> mismatchingDims;
  vector::BroadcastableToResult res = vector::isBroadcastableTo(getSourceType(), getVectorType(), &mismatchingDims);
  if (res == vector::BroadcastableToResult::Success) {
    return success();
  }
  if (res == vector::BroadcastableToResult::SourceRankHigher) {
    return emitOpError("source rank higher than destination rank");
  }
  if (res == vector::BroadcastableToResult::DimensionMismatch) {
    return emitOpError("dimension mismatch (") << mismatchingDims.first << " vs. " << mismatchingDims.second << ")";
  }

  if (res == vector::BroadcastableToResult::SourceTypeNotAVector) {
    return emitOpError("source type is not a vector");
  }
  llvm_unreachable("unexpected vector.broadcast op error");
}

// ===----------------------------------------------------------------------=== //
// TransposeOp
// ===----------------------------------------------------------------------=== //
LogicalResult TransposeOp::verify() {
  VectorType vectorType = getVectorType();
  VectorType resultType = getResultType();
  int64_t rank = resultType.getRank();
  if (vectorType.getRank() != rank) {
    return emitOpError("vector result rank mismatch: ") << rank;
  }
  // Verify transposition array.
  auto transpAttr = getTransp().getValue();
  int64_t size = (int64_t)transpAttr.size();
  if (rank != size) {
    return emitOpError("transposition length mismatch: ") << size;
  }
  SmallVector<bool, 8> seen(rank, false);
  for (const auto &ta : llvm::enumerate(transpAttr)) {
    int64_t i = ta.value().cast<IntegerAttr>().getInt();
    if (i < 0 || i >= rank) {
      return emitOpError("transposition index out of range: ") << i;
    }
    if (seen[(unsigned long)i]) {
      return emitOpError("duplicate position index: ") << i;
    }
    seen[i] = true;
    if (resultType.getDimSize(ta.index()) != vectorType.getDimSize((unsigned int)i)) {
      return emitOpError("dimension size mismatch at: ") << i;
    }
  }
  return success();
}

// ===----------------------------------------------------------------------=== //
// StoreOp
// ===----------------------------------------------------------------------=== //
::mlir::ParseResult StoreOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  OpAsmParser::UnresolvedOperand valueInfo, memrefInfo;
  SmallVector<OpAsmParser::UnresolvedOperand, 8> indexInfo;
  if (parser.parseOperand(valueInfo) || parser.parseComma() || parser.parseOperand(memrefInfo) ||
      parser.parseOperandList(indexInfo, OpAsmParser::Delimiter::Square)) {
    return failure();
  }

  SMLoc typesLoc;
  SmallVector<Type, 2> types;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.getCurrentLocation(&typesLoc) ||
      parser.parseColonTypeList(types)) {
    return failure();
  }

  if (types.size() != 2) {
    return parser.emitError(typesLoc, "requires two types");
  }

  if (!types[0].isa<VectorType>() && types[0].isa<ShapedType>()) {
    return parser.emitError(typesLoc, "require vector type or scalar type");
  }

  auto memrefType = types[1].dyn_cast<MemRefType>();
  if (!memrefType) {
    return parser.emitError(typesLoc, "requires memref or ranked tensor type");
  }

  auto &builder = parser.getBuilder();
  auto indexType = builder.getIndexType();
  if (parser.resolveOperand(valueInfo, types[0], result.operands) ||
      parser.resolveOperand(memrefInfo, memrefType, result.operands) ||
      parser.resolveOperands(indexInfo, indexType, result.operands)) {
    return failure();
  }

  return success();
}

void StoreOp::print(::mlir::OpAsmPrinter &p) {
  p << " " << getValueToStore() << ", " << getMemref() << "[" << getIndices() << "]";
  p.printOptionalAttrDict(this->getOperation()->getAttrs());
  p << " : " << getValueToStore().getType() << ", " << getMemRefType();
}

LogicalResult StoreOp::verify() {
  uint64_t opndIdxThreshold = 2;
  if ((uint64_t)getNumOperands() != opndIdxThreshold + (uint64_t)getMemRefType().getRank()) {
    return emitOpError("store index operand count not equal to memref rank");
  }

  return success();
}

LogicalResult StoreOp::fold(FoldAdaptor, SmallVectorImpl<OpFoldResult> &) {
  return memref::foldMemRefCast(*this, getValueToStore());
}

// ===----------------------------------------------------------------------=== //
// TableGen'd op method definitions
// ===----------------------------------------------------------------------=== //

#define GET_OP_CLASSES
#include "akg/Dialect/Fusion/IR/FusionOps.cpp.inc"
