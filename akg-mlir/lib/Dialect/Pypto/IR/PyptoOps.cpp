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
#include <optional>
#include <string>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
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

// Returns the total element count, or std::nullopt when any dimension is
// dynamic.
std::optional<int64_t> getStaticNumElements(ArrayRef<int64_t> shape) {
  int64_t num = 1;
  for (int64_t dim : shape) {
    if (ShapedType::isDynamic(dim)) {
      return std::nullopt;
    }
    num *= dim;
  }
  return num;
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

// Checks that every tile-shaped value in `values` shares one static shape.
// Plain scalars (which broadcast) and opaque tensors (whose shape the
// lowering guards) are exempt.
LogicalResult verifySameShape(Operation *op, ArrayRef<Value> values) {
  std::optional<SmallVector<int64_t>> expected;
  for (Value value : values) {
    std::optional<SmallVector<int64_t>> shape = getShapeIfKnown(value.getType());
    if (!shape) {
      continue;
    }
    if (!expected) {
      expected = shape;
    } else if (*shape != *expected) {
      return op->emitOpError("tile operands/results must share one shape, got ")
             << shapeToString(*expected) << " and " << shapeToString(*shape);
    }
  }
  return success();
}

// Shared verifier for the elementwise binary ops.
template <typename OpTy>
LogicalResult verifyBinaryShape(OpTy op) {
  return verifySameShape(op.getOperation(), {op.getLhs(), op.getRhs(), op.getResult()});
}

// Shared verifier for the elementwise unary ops.
template <typename OpTy>
LogicalResult verifyUnaryShape(OpTy op) {
  return verifySameShape(op.getOperation(), {op.getInput(), op.getResult()});
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
  std::optional<int64_t> srcElems = getStaticNumElements(*srcShape);
  std::optional<int64_t> dstElems = getStaticNumElements(*dstShape);
  if (srcElems && dstElems && *srcElems != *dstElems) {
    return emitOpError("reshape must preserve the element count: source ")
           << shapeToString(*srcShape) << " has " << *srcElems << " elements, result " << shapeToString(*dstShape)
           << " has " << *dstElems;
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
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(getResult().getType());
  if (srcShape && dstShape) {
    SmallVector<int64_t> expected = *srcShape;
    std::swap(expected[i], expected[j]);
    if (expected != *dstShape) {
      return emitOpError("expected result shape ")
             << shapeToString(expected) << " for axes " << i << " and " << j << ", got " << shapeToString(*dstShape);
    }
  }
  return success();
}

LogicalResult ReinterpretViewOp::verify() {
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(getResult().getType());
  if (srcShape && dstShape && *srcShape != *dstShape) {
    return emitOpError("reinterpret_view must preserve the shape: ")
           << shapeToString(*srcShape) << " vs " << shapeToString(*dstShape);
  }
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
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(getSource().getType());
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(getResult().getType());
  if (srcShape && dstShape && *srcShape != *dstShape) {
    return emitOpError("cast must preserve the shape: ")
           << shapeToString(*srcShape) << " vs " << shapeToString(*dstShape);
  }
  return success();
}

LogicalResult CmpOp::verify() {
  if (failed(verifySameShape(getOperation(), {getLhs(), getRhs(), getResult()}))) {
    return failure();
  }
  std::optional<Type> elem = getElementTypeIfKnown(getResult().getType());
  if (elem && !elem->isInteger(1)) {
    return emitOpError("result must hold i1 elements, got ") << *elem;
  }
  return success();
}

LogicalResult SelectOp::verify() {
  if (failed(verifySameShape(getOperation(), {getCondition(), getTrueValue(), getFalseValue(), getResult()}))) {
    return failure();
  }
  std::optional<Type> condElem = getElementTypeIfKnown(getCondition().getType());
  if (condElem && !condElem->isInteger(1)) {
    return emitOpError("condition must hold i1 elements, got ") << *condElem;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// D. Elementwise binary ops
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() { return verifyBinaryShape(*this); }
LogicalResult SubOp::verify() { return verifyBinaryShape(*this); }
LogicalResult MulOp::verify() { return verifyBinaryShape(*this); }
LogicalResult DivOp::verify() { return verifyBinaryShape(*this); }
LogicalResult FmodOp::verify() { return verifyBinaryShape(*this); }
LogicalResult MaximumOp::verify() { return verifyBinaryShape(*this); }
LogicalResult MinimumOp::verify() { return verifyBinaryShape(*this); }
LogicalResult AndOp::verify() { return verifyBinaryShape(*this); }
LogicalResult OrOp::verify() { return verifyBinaryShape(*this); }
LogicalResult XorOp::verify() { return verifyBinaryShape(*this); }
LogicalResult ShlOp::verify() { return verifyBinaryShape(*this); }
LogicalResult ShrOp::verify() { return verifyBinaryShape(*this); }

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
  if (tileShape && rowShape) {
    int64_t tileRows = (*tileShape)[tileShape->size() - 2];
    int64_t rowRows = (*rowShape)[rowShape->size() - 2];
    if (!ShapedType::isDynamic(tileRows) && !ShapedType::isDynamic(rowRows) && tileRows != rowRows) {
      return op.emitOpError("row counts must match: tile has ") << tileRows << " rows, row vector has " << rowRows;
    }
  }
  if (failed(verifySameShape(op.getOperation(), {op.getTile(), op.getResult()}))) {
    return failure();
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

//===----------------------------------------------------------------------===//
// E. Elementwise unary ops
//===----------------------------------------------------------------------===//

LogicalResult NegOp::verify() { return verifyUnaryShape(*this); }
LogicalResult AbsOp::verify() { return verifyUnaryShape(*this); }
LogicalResult ExpOp::verify() { return verifyUnaryShape(*this); }
LogicalResult LogOp::verify() { return verifyUnaryShape(*this); }
LogicalResult SqrtOp::verify() { return verifyUnaryShape(*this); }
LogicalResult RsqrtOp::verify() { return verifyUnaryShape(*this); }
LogicalResult SinOp::verify() { return verifyUnaryShape(*this); }
LogicalResult CosOp::verify() { return verifyUnaryShape(*this); }

//===----------------------------------------------------------------------===//
// F. Scalar combines
//===----------------------------------------------------------------------===//

LogicalResult MaxOp::verify() { return verifyBinaryShape(*this); }
LogicalResult MinOp::verify() { return verifyBinaryShape(*this); }

//===----------------------------------------------------------------------===//
// G. Reduction and scalar IO
//===----------------------------------------------------------------------===//

LogicalResult ReductionOp::verify() {
  // The vector may be a !pypto.tile or, during the incremental lowering,
  // the source shaped type (vector/npuvector) it was derived from. The
  // scratch may also be an opaque `pl.Tile` produced mid-lowering; shape
  // checks only run when both sides carry a static shape.
  std::optional<SmallVector<int64_t>> vectorShape = getShapeIfKnown(getVector().getType());
  if (!vectorShape) {
    return success();
  }
  if (vectorShape->empty()) {
    return emitOpError("cannot reduce a rank-0 tile");
  }
  std::optional<SmallVector<int64_t>> scratchShape = getShapeIfKnown(getScratch().getType());
  if (scratchShape && *scratchShape != *vectorShape) {
    return emitOpError("scratch shape ") << shapeToString(*scratchShape) << " must match the vector shape "
                                         << shapeToString(*vectorShape);
  }

  std::optional<SmallVector<int64_t>> resultShape = getShapeIfKnown(getResult().getType());
  if (!resultShape) {
    return success();
  }
  SmallVector<int64_t> expected(*vectorShape);
  size_t folded = getAxis() == ReduceAxis::row ? expected.size() - 1 : 0;
  expected[folded] = 1;
  if (*resultShape != expected) {
    // During the incremental lowering a partial reduction keeps the source
    // lower-rank type instead (the folded axis is dropped); the folded
    // extent is 1, so col-major flat indexing stays equivalent.
    SmallVector<int64_t> lowered(*vectorShape);
    lowered.erase(lowered.begin() + folded);
    if (*resultShape == lowered) {
      return success();
    }
    return emitOpError("the reduction is keepdim: expected result shape ")
           << shapeToString(expected) << ", got " << shapeToString(*resultShape);
  }
  return success();
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
    if (ShapedType::isDynamic(dim)) {
      continue;
    }
    if (index < 0 || index >= dim) {
      return emitOpError("index ") << index << " is out of range for dim " << dim;
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
