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

#include "akg/Dialect/Pypto/IR/PyptoOps.h"

#include <algorithm>
#include <numeric>
#include <optional>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpImplementation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;         // NOLINT(build/namespaces)
using namespace mlir::pypto;  // NOLINT(build/namespaces)

namespace {

// Returns the static shape carried by a tile or another shaped type, or
// std::nullopt for plain scalars and opaque emitpy tensors. The lowering
// guards the opaque case, so the verifiers skip it.
std::optional<SmallVector<int64_t>> getShapeIfKnown(Type type) {
  if (auto tile = dyn_cast<TileType>(type)) {
    return SmallVector<int64_t>(tile.getShape());
  }
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return SmallVector<int64_t>(shaped.getShape());
  }
  return std::nullopt;
}

// Returns the element type of a tile or another shaped type, or
// std::nullopt for plain scalars and opaque emitpy tensors.
std::optional<Type> getElementTypeIfKnown(Type type) {
  if (auto tile = dyn_cast<TileType>(type)) {
    return tile.getElementType();
  }
  if (auto shaped = dyn_cast<ShapedType>(type)) {
    return shaped.getElementType();
  }
  return std::nullopt;
}

std::string shapeToString(ArrayRef<int64_t> shape) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "[";
  llvm::interleave(shape, os, ",");
  os << "]";
  os.flush();
  return out;
}

}  // namespace

//===----------------------------------------------------------------------===//
// A. Constants and allocation
//===----------------------------------------------------------------------===//

LogicalResult ConstantOp::verify() {
  if (getValue().empty()) {
    return emitOpError("value must be a non-empty Python literal");
  }
  Type type = getResult().getType();
  if (!TileType::isValidElementType(type)) {
    return emitOpError("result must carry a scalar PyPTO dtype, got ") << type;
  }
  return success();
}

LogicalResult TileFullOp::verify() {
  if (getValue().empty()) {
    return emitOpError("value must be a non-empty Python literal");
  }
  // `extents` names the runtime bound of every dynamic dimension and is
  // empty for statically shaped results. During the incremental lowering
  // the result may carry the source shaped type instead of a !pypto.tile.
  if (getExtents().empty()) {
    return success();
  }
  std::optional<SmallVector<int64_t>> shape = getShapeIfKnown(getResult().getType());
  if (!shape || shape->size() != getExtents().size()) {
    return emitOpError("extents (") << getExtents().size() << ") must be empty or match the result rank ("
                                    << (shape ? shape->size() : 0) << ")";
  }
  return success();
}

LogicalResult CreateTileOp::verify() {
  if (!getValidExtents().empty() && (getResult().getType().getRank() != 2 || getValidExtents().size() != 2)) {
    return emitOpError("valid extents require a 2D tile and two index operands");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// B. Tile views
//===----------------------------------------------------------------------===//

LogicalResult SliceOp::verify() {
  ArrayRef<int64_t> window = getWindow();
  auto nonPositive = std::find_if(window.begin(), window.end(), [](int64_t dim) { return dim <= 0; });
  if (nonPositive != window.end()) {
    return emitOpError("window dims must be positive, got ") << *nonPositive;
  }
  if (window.size() != getOffsets().size()) {
    return emitOpError("window rank (") << window.size() << ") must match the number of offsets ("
                                        << getOffsets().size() << ")";
  }

  // Unit window dims mark the axes a rank-reducing view drops; they must be
  // listed in drop_dims and must sit inside the window.
  llvm::SmallBitVector droppedAxes(window.size());
  if (auto dropDims = getDropDims()) {
    ArrayRef<int64_t> droppedList = *dropDims;
    for (int64_t axis : droppedList) {
      if (axis < 0 || axis >= static_cast<int64_t>(window.size())) {
        return emitOpError("drop_dims axis ") << axis << " is out of range for window rank " << window.size();
      }
      if (window[axis] != 1) {
        return emitOpError("dropped axis ") << axis << " must have a unit window dim, got " << window[axis];
      }
      droppedAxes.set(axis);
    }
  }

  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  if (srcShape && srcShape->size() != window.size()) {
    return emitOpError("window rank (") << window.size() << ") must match the source rank (" << srcShape->size() << ")";
  }

  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(getResult().getType());
  if (!dstShape) {
    return success();
  }
  if (dstShape->size() + droppedAxes.count() != window.size()) {
    return emitOpError("result rank (") << dstShape->size() << ") plus the dropped axes (" << droppedAxes.count()
                                        << ") must match the window rank (" << window.size() << ")";
  }

  // Non-dropped window dims must agree with the result dims.
  size_t resultDim = 0;
  for (auto [axis, dim] : llvm::enumerate(window)) {
    if (droppedAxes.test(axis)) {
      continue;
    }
    int64_t actual = (*dstShape)[resultDim++];
    if (!ShapedType::isDynamic(actual) && dim != actual) {
      return emitOpError("window dim ") << dim << " on axis " << axis << " must match the result dim " << actual;
    }
  }
  return success();
}

LogicalResult ReshapeOp::verify() {
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(getResult().getType());
  // `extents` names the runtime bound of every result dimension, mirroring
  // `pypto.tile_full`; it is empty for statically shaped results. Dynamic
  // reshapes carry it so the emitpy lowering can spell the `pl.reshape`
  // shape list.
  if (!getExtents().empty()) {
    if (!dstShape || dstShape->size() != getExtents().size()) {
      return emitOpError("extents (") << getExtents().size() << ") must be empty or match the result rank ("
                                      << (dstShape ? dstShape->size() : 0) << ")";
    }
  }
  if (!srcShape || !dstShape) {
    return success();
  }
  return success();
}

LogicalResult TransposeOp::verify() {
  int64_t i = getI();
  int64_t j = getJ();
  if (i == j) {
    return emitOpError("identity swaps fold away at conversion time; i (") << i << ") must differ from j (" << j << ")";
  }
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  if (srcShape) {
    if (i < 0 || i >= static_cast<int64_t>(srcShape->size()) || j < 0 || j >= static_cast<int64_t>(srcShape->size())) {
      return emitOpError("axes ") << i << " and " << j << " are out of range for source rank " << srcShape->size();
    }
  }
  return success();
}

LogicalResult ReinterpretViewOp::verify() {
  std::optional<Type> srcElem = getElementTypeIfKnown(getSource().getType());
  std::optional<Type> dstElem = getElementTypeIfKnown(getResult().getType());
  if (srcElem && dstElem && srcElem->getIntOrFloatBitWidth() != dstElem->getIntOrFloatBitWidth()) {
    return emitOpError("element types must share the bit width: ") << *srcElem << " vs " << *dstElem;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// C. Conversion, comparison and selection
//===----------------------------------------------------------------------===//

LogicalResult CastOp::verify() {
  if (auto mode = getMode()) {
    if (!llvm::is_contained(ArrayRef<StringRef>{"round", "floor", "ceil"}, *mode)) {
      return emitOpError("unknown float-to-int rounding mode: ") << *mode;
    }
  }
  return success();
}

LogicalResult CmpOp::verify() {
  std::optional<Type> elem = getElementTypeIfKnown(getResult().getType());
  if (elem && !elem->isInteger(1)) {
    return emitOpError("result must hold i1 elements, got ") << *elem;
  }
  return success();
}

LogicalResult SelectOp::verify() {
  std::optional<Type> condElem = getElementTypeIfKnown(getCondition().getType());
  if (condElem && !condElem->isInteger(1)) {
    return emitOpError("condition must hold i1 elements, got ") << *condElem;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// D. Elementwise binary ops
//===----------------------------------------------------------------------===//

// Physical shape relations are checked by PyptoAlignShape after propagation.
LogicalResult AddOp::verify() { return success(); }
LogicalResult SubOp::verify() { return success(); }
LogicalResult MulOp::verify() { return success(); }
LogicalResult DivOp::verify() { return success(); }
LogicalResult FmodOp::verify() { return success(); }
LogicalResult MaximumOp::verify() { return success(); }
LogicalResult MinimumOp::verify() { return success(); }

// Shared verifier for the tile-scalar clamps: the lhs and result share one
// tile shape while the rhs must stay a plain scalar (the runtime's TMAXS/
// TMINS intrinsics take the scalar directly, so a tile rhs would miss the
// point of the op family).
template <typename OpTy>
LogicalResult verifyTileScalarClampShape(OpTy op) {
  if (getShapeIfKnown(op.getRhs().getType())) {
    return op->emitOpError("tile-scalar clamp needs a scalar rhs, got ") << op.getRhs().getType();
  }
  return success();
}

LogicalResult MaximumsOp::verify() { return verifyTileScalarClampShape(*this); }
LogicalResult MinimumsOp::verify() { return verifyTileScalarClampShape(*this); }
LogicalResult AndOp::verify() { return success(); }
LogicalResult OrOp::verify() { return success(); }
LogicalResult XorOp::verify() { return success(); }
LogicalResult ShlOp::verify() { return success(); }
LogicalResult ShrOp::verify() { return success(); }

// Shared verifier for the row_expand family, mirroring the runtime's
// DeduceTileRowExpandType contract: both operands carry at least two
// dimensions (the last two drive the broadcast), the row vector's last
// dimension is one, the row counts agree, and the result matches the tile.
// Checks that depend on dynamic (`?`) extents are skipped, matching the
// runtime's provable-mismatch-only policy.
template <typename OpTy>
static LogicalResult verifyRowExpand(OpTy op) {
  std::optional<SmallVector<int64_t>> tileShape = getShapeIfKnown(op.getTile().getType());
  std::optional<SmallVector<int64_t>> rowShape = getShapeIfKnown(op.getRowVec().getType());
  if (tileShape && tileShape->size() < 2) {
    return op.emitOpError("tile must have at least 2 dimensions, got ") << shapeToString(*tileShape);
  }
  if (rowShape && rowShape->size() < 2) {
    return op.emitOpError("row vector must have at least 2 dimensions, got ") << shapeToString(*rowShape);
  }
  if (rowShape && !ShapedType::isDynamic(rowShape->back()) && rowShape->back() != 1) {
    return op.emitOpError("row vector's last dimension must be 1, got ") << shapeToString(*rowShape);
  }

  std::optional<Type> tileElem = getElementTypeIfKnown(op.getTile().getType());
  std::optional<Type> rowElem = getElementTypeIfKnown(op.getRowVec().getType());
  if (tileElem && rowElem && *tileElem != *rowElem) {
    return op.emitOpError("tile and row vector must share one element type");
  }
  return success();
}

LogicalResult RowExpandAddOp::verify() { return verifyRowExpand(*this); }
LogicalResult RowExpandSubOp::verify() { return verifyRowExpand(*this); }
LogicalResult RowExpandMulOp::verify() { return verifyRowExpand(*this); }
LogicalResult RowExpandDivOp::verify() { return verifyRowExpand(*this); }

// Shared verifier for the col_expand family, mirroring the runtime's
// DeduceTileColExpandType contract: both operands carry at least two
// dimensions (the last two drive the broadcast), the column vector's
// second-to-last dimension is one, the column counts agree, and the result
// matches the tile. Checks that depend on dynamic (`?`) extents are skipped,
// matching the runtime's provable-mismatch-only policy.
template <typename OpTy>
static LogicalResult verifyColExpand(OpTy op) {
  std::optional<SmallVector<int64_t>> tileShape = getShapeIfKnown(op.getTile().getType());
  std::optional<SmallVector<int64_t>> colShape = getShapeIfKnown(op.getColVec().getType());
  if (tileShape && tileShape->size() < 2) {
    return op.emitOpError("tile must have at least 2 dimensions, got ") << shapeToString(*tileShape);
  }
  if (colShape && colShape->size() < 2) {
    return op.emitOpError("column vector must have at least 2 dimensions, got ") << shapeToString(*colShape);
  }
  if (colShape && !ShapedType::isDynamic((*colShape)[colShape->size() - 2]) && (*colShape)[colShape->size() - 2] != 1) {
    return op.emitOpError("column vector's second-to-last dimension must be 1, got ") << shapeToString(*colShape);
  }

  std::optional<Type> tileElem = getElementTypeIfKnown(op.getTile().getType());
  std::optional<Type> colElem = getElementTypeIfKnown(op.getColVec().getType());
  if (tileElem && colElem && *tileElem != *colElem) {
    return op.emitOpError("tile and column vector must share one element type");
  }
  return success();
}

LogicalResult ColExpandAddOp::verify() { return verifyColExpand(*this); }
LogicalResult ColExpandSubOp::verify() { return verifyColExpand(*this); }
LogicalResult ColExpandMulOp::verify() { return verifyColExpand(*this); }
LogicalResult ColExpandDivOp::verify() { return verifyColExpand(*this); }

//===----------------------------------------------------------------------===//
// E. Elementwise unary ops
//===----------------------------------------------------------------------===//

LogicalResult NegOp::verify() { return success(); }
LogicalResult AbsOp::verify() { return success(); }
LogicalResult ExpOp::verify() { return success(); }
LogicalResult LogOp::verify() { return success(); }
LogicalResult SqrtOp::verify() { return success(); }
LogicalResult RsqrtOp::verify() { return success(); }
LogicalResult RecipOp::verify() { return success(); }
LogicalResult SinOp::verify() { return success(); }
LogicalResult CosOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// F. Scalar combines
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::verify() { return success(); }
LogicalResult MinOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// G. Reduction and scalar IO
//===----------------------------------------------------------------------===//

LogicalResult FillPadExpandOp::verify() {
  if (getPadValue() != "zero" && getPadValue() != "min" && getPadValue() != "max") {
    return emitOpError("padding mode must be zero, min or max");
  }
  auto sourceShape = getShapeIfKnown(getSource().getType());
  auto resultShape = getShapeIfKnown(getResult().getType());
  if (!resultShape || resultShape->size() != 2 || llvm::any_of(*resultShape, [](int64_t dim) { return dim <= 0; })) {
    return emitOpError("result must be a statically shaped positive 2D tile");
  }
  if (sourceShape) {
    if (sourceShape->size() != 2) return emitOpError("source must be a 2D tile");
    for (auto [sourceDim, resultDim] : llvm::zip(*sourceShape, *resultShape)) {
      if (!ShapedType::isDynamic(sourceDim) && sourceDim > resultDim) {
        return emitOpError("result cannot shrink a source dimension");
      }
    }
  }
  auto sourceElement = getElementTypeIfKnown(getSource().getType());
  auto resultElement = getElementTypeIfKnown(getResult().getType());
  if (sourceElement && resultElement && *sourceElement != *resultElement) {
    return emitOpError("source and result element types must match");
  }
  return success();
}

LogicalResult ReductionOp::verify() {
  auto shape = getShapeIfKnown(getVector().getType());
  if (shape && shape->empty()) return emitOpError("cannot reduce a rank-0 tile");
  return success();
}

// Index coordinates come from `arith.constant`; matching the operation by
// name and reading its value attribute keeps this IR library free of the
// arith dependency.
static std::optional<int64_t> constantIndexValue(Value index) {
  Operation *def = index.getDefiningOp();
  if (!def || def->getName().getStringRef() != "arith.constant") {
    return std::nullopt;
  }
  auto attr = dyn_cast<IntegerAttr>(def->getAttr("value"));
  if (!attr) {
    return std::nullopt;
  }
  return attr.getInt();
}

LogicalResult ReadOp::verify() {
  std::optional<SmallVector<int64_t>> shape = getShapeIfKnown(getSource().getType());
  if (!shape) {
    return success();
  }
  if (getIndices().size() != shape->size()) {
    return emitOpError("number of indices (")
           << getIndices().size() << ") must match the source rank (" << shape->size() << ")";
  }
  for (auto [index, dim] : llvm::zip(getIndices(), *shape)) {
    auto constant = constantIndexValue(index);
    if (!constant || ShapedType::isDynamic(dim)) {
      // Runtime coordinates (loop-dependent reads) are checked at runtime.
      continue;
    }
    if (*constant < 0 || *constant >= dim) {
      return emitOpError("index ") << *constant << " is out of range for dim " << dim;
    }
  }
  return success();
}

LogicalResult WriteOp::verify() {
  std::optional<SmallVector<int64_t>> shape = getShapeIfKnown(getDest().getType());
  if (shape && getOffsets().size() != shape->size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match the destination rank (" << shape->size() << ")";
  }
  if (getElementTypeIfKnown(getValue().getType())) {
    return emitOpError("value must be a scalar, got ") << getValue().getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Tile movement between tensor memory and cores
//===----------------------------------------------------------------------===//

LogicalResult LoadOp::verify() {
  // The result may be a !pypto.tile or, during the incremental lowering,
  // the source shaped type (vector/npuvector) it was derived from. Plain
  // scalars count as rank zero. Offsets address the source, so they match
  // the source's rank: rank-reduced loads (source rank > tile rank, as
  // produced by vector.transfer_read lowering) are legal. Sources that
  // carry no shape (e.g. !emitpy.opaque) fall back to the tile's rank.
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  std::optional<SmallVector<int64_t>> tileShape = getShapeIfKnown(getResult().getType());
  size_t srcRank = srcShape ? srcShape->size() : tileShape ? tileShape->size() : 0;
  if (getOffsets().size() != srcRank) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match the source rank (" << srcRank << ")";
  }
  size_t padded = getPaddedShapes().size();
  size_t valid = getValidExtents().size();
  if (padded != valid) {
    return emitOpError("padded shapes (") << padded << ") and valid extents (" << valid << ") must be given together";
  }
  if (padded != 0 && padded != srcRank) {
    return emitOpError("number of padded shapes (") << padded << ") must match the source rank (" << srcRank << ")";
  }
  return success();
}

LogicalResult StoreOp::verify() {
  // See LoadOp::verify: the value may carry the source shaped type
  // instead of a !pypto.tile, and offsets match the destination's rank;
  // destinations without a shape (e.g. !emitpy.opaque) fall back to the
  // value's rank.
  std::optional<SmallVector<int64_t>> destShape = getShapeIfKnown(getDest().getType());
  std::optional<SmallVector<int64_t>> valShape = getShapeIfKnown(getValue().getType());
  size_t rank = destShape ? destShape->size() : valShape ? valShape->size() : 0;
  if (getOffsets().size() != rank) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match the destination rank (" << rank << ")";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// H. SPMD and loop structure
//===----------------------------------------------------------------------===//

LogicalResult SpmdOp::verify() {
  if (getCoreNum() <= 0) {
    return emitOpError() << "expects a positive core count";
  }
  Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !isa<IndexType>(body.getArgument(0).getType())) {
    return emitOpError() << "expects exactly one index-typed block index argument";
  }
  return success();
}

ParseResult SpmdOp::parse(OpAsmParser &parser, OperationState &result) {
  // pypto.spmd (%blk) in (<core-num>) { ... }
  SmallVector<OpAsmParser::Argument, 1> blockArgs;
  if (parser.parseArgumentList(blockArgs, AsmParser::Delimiter::Paren)) {
    return failure();
  }
  // The block index is always index-typed.
  for (auto &arg : blockArgs) {
    arg.type = parser.getBuilder().getIndexType();
  }
  IntegerAttr coreNum;
  if (parser.parseKeyword("in") || parser.parseLParen() ||
      parser.parseAttribute(coreNum, "core_num", result.attributes) || parser.parseRParen()) {
    return failure();
  }
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, blockArgs)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

void SpmdOp::print(OpAsmPrinter &p) {
  p << " (" << getBody().front().getArgument(0) << ") in (" << getCoreNum() << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict((*this)->getAttrs(), {"core_num"});
}

LogicalResult RangeOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != 1 || !isa<IndexType>(body.getArgument(0).getType())) {
    return emitOpError() << "expects exactly one index-typed induction variable";
  }
  return success();
}

ParseResult RangeOp::parse(OpAsmParser &parser, OperationState &result) {
  // pypto.range %i = %lb to %ub step %step { ... }
  OpAsmParser::Argument iv;
  Type boundType = parser.getBuilder().getIndexType();
  OpAsmParser::UnresolvedOperand lb, ub, step;
  if (parser.parseArgument(iv) || parser.parseEqual() || parser.parseOperand(lb) || parser.parseKeyword("to") ||
      parser.parseOperand(ub) || parser.parseKeyword("step") || parser.parseOperand(step) ||
      parser.resolveOperand(lb, boundType, result.operands) || parser.resolveOperand(ub, boundType, result.operands) ||
      parser.resolveOperand(step, boundType, result.operands)) {
    return failure();
  }
  iv.type = boundType;
  Region *body = result.addRegion();
  if (parser.parseRegion(*body, iv)) {
    return failure();
  }
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }
  return success();
}

void RangeOp::print(OpAsmPrinter &p) {
  p << " " << getBody().front().getArgument(0) << " = " << getLowerBound() << " to " << getUpperBound() << " step "
    << getStep() << " ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict((*this)->getAttrs());
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions.
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "akg/Dialect/Pypto/IR/PyptoOps.cpp.inc"
