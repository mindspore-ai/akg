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

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include "mlir/IR/Matchers.h"
#include "llvm/Support/raw_ostream.h"

#include "akg/Dialect/Pypto/Passes.h"
#include "akg/Dialect/Pypto/IR/PyptoOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
#define GEN_PASS_DEF_PYPTOALIGNSHAPE
#include "akg/Dialect/Pypto/Passes.h.inc"
namespace {
using Shape = SmallVector<int64_t>;
using Shapes = DenseMap<Value, Shape>;

static bool isSameShapeOp(Operation *op) {
  return isa<pypto::AddOp, pypto::SubOp, pypto::MulOp, pypto::DivOp, pypto::FmodOp, pypto::AndOp, pypto::OrOp,
             pypto::XorOp, pypto::ShlOp, pypto::ShrOp, pypto::MaximumOp, pypto::MinimumOp, pypto::MaximumsOp,
             pypto::MinimumsOp, pypto::CastOp, pypto::AbsOp, pypto::ExpOp, pypto::LogOp, pypto::SqrtOp, pypto::RsqrtOp,
             pypto::RecipOp, pypto::NegOp, pypto::SinOp, pypto::CosOp, pypto::CmpOp, pypto::SelectOp>(op);
}

using pypto::ReduceAxis;
using pypto::TileType;

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

// Check explicit allocation bounds, not the logical shape of views or
// comparison masks. Runtime bounds are checked by the PyPTO backend.
LogicalResult verifyAllocationAlignment(Operation *op, Type type, ValueRange extents = {}) {
  std::optional<Type> elementType = getElementTypeIfKnown(type);
  if (!elementType || !TileType::isValidElementType(*elementType)) {
    return success();
  }
  int64_t trailing;
  if (!extents.empty()) {
    APInt bound;
    if (!matchPattern(extents.back(), m_ConstantInt(&bound))) {
      return success();
    }
    trailing = bound.getSExtValue();
  } else {
    std::optional<SmallVector<int64_t>> shape = getShapeIfKnown(type);
    if (!shape || shape->empty() || ShapedType::isDynamic(shape->back())) {
      return success();
    }
    trailing = shape->back();
  }
  // Explicit BOOL allocations use one byte; index lowers to INT64.
  unsigned elementBits = isa<IndexType>(*elementType) ? 64 : elementType->getIntOrFloatBitWidth();
  unsigned elementBytes = (elementBits + 7) / 8;
  unsigned alignedElements = 32 / std::gcd(32u, elementBytes);
  if (trailing % alignedElements != 0) {
    return op->emitOpError("allocation trailing dimension must be a multiple of ")
           << alignedElements << " for 32-byte alignment, got " << trailing << " for " << *elementType;
  }
  return success();
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

LogicalResult verifyAlignedTileFull(pypto::TileFullOp op) {
  return verifyAllocationAlignment(op, op.getResult().getType(), op.getExtents());
}

LogicalResult verifyAlignedCreateTile(pypto::CreateTileOp op) {
  return verifyAllocationAlignment(op, op.getResult().getType());
}

LogicalResult verifyAlignedReshape(pypto::ReshapeOp op) {
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(op.getSource().getType());
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(op.getResult().getType());
  // `extents` names the runtime bound of every result dimension, mirroring
  // `pypto.tile_full`; it is empty for statically shaped results. Dynamic
  // reshapes carry it so the emitpy lowering can spell the `pl.reshape`
  // shape list.
  if (!op.getExtents().empty()) {
    if (!dstShape || dstShape->size() != op.getExtents().size()) {
      return op.emitOpError("extents (") << op.getExtents().size() << ") must be empty or match the result rank ("
                                         << (dstShape ? dstShape->size() : 0) << ")";
    }
  }
  if (!srcShape || !dstShape) {
    return success();
  }
  std::optional<int64_t> srcElems = getStaticNumElements(*srcShape);
  std::optional<int64_t> dstElems = getStaticNumElements(*dstShape);
  if (srcElems && dstElems && *srcElems != *dstElems) {
    return op.emitOpError("reshape must preserve the element count: source ")
           << shapeToString(*srcShape) << " has " << *srcElems << " elements, result " << shapeToString(*dstShape)
           << " has " << *dstElems;
  }
  return success();
}

LogicalResult verifyAlignedTranspose(pypto::TransposeOp op) {
  int64_t i = op.getI();
  int64_t j = op.getJ();
  if (i == j) {
    return op.emitOpError("identity swaps fold away at conversion time; i (")
           << i << ") must differ from j (" << j << ")";
  }
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(op.getSource().getType());
  if (srcShape) {
    if (i < 0 || i >= static_cast<int64_t>(srcShape->size()) || j < 0 || j >= static_cast<int64_t>(srcShape->size())) {
      return op.emitOpError("axes ") << i << " and " << j << " are out of range for source rank " << srcShape->size();
    }
  }
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(op.getResult().getType());
  if (srcShape && dstShape) {
    SmallVector<int64_t> expected = *srcShape;
    std::swap(expected[i], expected[j]);
    if (expected != *dstShape) {
      return op.emitOpError("expected result shape ")
             << shapeToString(expected) << " for axes " << i << " and " << j << ", got " << shapeToString(*dstShape);
    }
  }
  return success();
}

LogicalResult verifyAlignedReinterpretView(pypto::ReinterpretViewOp op) {
  std::optional<SmallVector<int64_t>> srcShape = getShapeIfKnown(op.getSource().getType());
  std::optional<SmallVector<int64_t>> dstShape = getShapeIfKnown(op.getResult().getType());
  if (srcShape && dstShape && *srcShape != *dstShape) {
    return op.emitOpError("reinterpret_view must preserve the shape: ")
           << shapeToString(*srcShape) << " vs " << shapeToString(*dstShape);
  }
  std::optional<Type> srcElem = getElementTypeIfKnown(op.getSource().getType());
  std::optional<Type> dstElem = getElementTypeIfKnown(op.getResult().getType());
  if (srcElem && dstElem && srcElem->getIntOrFloatBitWidth() != dstElem->getIntOrFloatBitWidth()) {
    return op.emitOpError("element types must share the bit width: ") << *srcElem << " vs " << *dstElem;
  }
  return success();
}

LogicalResult verifyAlignedReduction(pypto::ReductionOp op) {
  // The vector may be a !pypto.tile or, during the incremental lowering,
  // the source shaped type (vector/npuvector) it was derived from. The
  // scratch may also be an opaque `pl.Tile` produced mid-lowering; shape
  // checks only run when both sides carry a static shape.
  std::optional<SmallVector<int64_t>> vectorShape = getShapeIfKnown(op.getVector().getType());
  if (!vectorShape) {
    return success();
  }
  if (vectorShape->empty()) {
    return op.emitOpError("cannot reduce a rank-0 tile");
  }
  std::optional<SmallVector<int64_t>> scratchShape = getShapeIfKnown(op.getScratch().getType());
  if (scratchShape) {
    if (scratchShape->size() != vectorShape->size()) {
      return op.emitOpError("scratch shape ")
             << shapeToString(*scratchShape) << " must have the vector's rank " << shapeToString(*vectorShape);
    }
    // The runtime requires the scratch to cover the vector in every
    // dimension: a static vector dim must be matched exactly, while a
    // dynamic dim names no element count of its own and the scratch
    // materializes its allocation bound statically.
    for (auto [scratchDim, vectorDim] : llvm::zip(*scratchShape, *vectorShape)) {
      if (ShapedType::isDynamic(vectorDim)) {
        if (ShapedType::isDynamic(scratchDim)) {
          return op.emitOpError("scratch shape ")
                 << shapeToString(*scratchShape) << " must materialize a static bound for the dynamic vector shape "
                 << shapeToString(*vectorShape);
        }
      } else if (ShapedType::isDynamic(scratchDim) || scratchDim < vectorDim) {
        return op.emitOpError("scratch shape ")
               << shapeToString(*scratchShape) << " must cover the vector shape " << shapeToString(*vectorShape);
      }
    }
  }

  std::optional<SmallVector<int64_t>> resultShape = getShapeIfKnown(op.getResult().getType());
  if (!resultShape) {
    return success();
  }
  SmallVector<int64_t> expected(*vectorShape);
  size_t folded = op.getAxis() == ReduceAxis::row ? expected.size() - 1 : 0;
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
    return op.emitOpError("the reduction is keepdim: expected result shape ")
           << shapeToString(expected) << ", got " << shapeToString(*resultShape);
  }
  return success();
}

LogicalResult verifyAlignedLoad(pypto::LoadOp op) {
  return verifyAllocationAlignment(op, op.getResult().getType(), op.getPaddedShapes());
}

template <typename OpTy>
static LogicalResult verifyAlignedRowExpand(OpTy op) {
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

template <typename OpTy>
static LogicalResult verifyAlignedColExpand(OpTy op) {
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
  if (tileShape && colShape) {
    int64_t tileCols = tileShape->back();
    int64_t vecCols = colShape->back();
    if (!ShapedType::isDynamic(tileCols) && !ShapedType::isDynamic(vecCols) && tileCols != vecCols) {
      return op.emitOpError("column counts must match: tile has ")
             << tileCols << " columns, column vector has " << vecCols;
    }
  }
  if (failed(verifySameShape(op.getOperation(), {op.getTile(), op.getResult()}))) {
    return failure();
  }
  std::optional<Type> tileElem = getElementTypeIfKnown(op.getTile().getType());
  std::optional<Type> colElem = getElementTypeIfKnown(op.getColVec().getType());
  if (tileElem && colElem && *tileElem != *colElem) {
    return op.emitOpError("tile and column vector must share one element type");
  }
  return success();
}

LogicalResult verifyAlignedOperation(Operation *operation) {
  if (auto op = dyn_cast<pypto::TileFullOp>(operation)) return verifyAlignedTileFull(op);
  if (auto op = dyn_cast<pypto::CreateTileOp>(operation)) return verifyAlignedCreateTile(op);
  if (auto op = dyn_cast<pypto::ReshapeOp>(operation)) return verifyAlignedReshape(op);
  if (auto op = dyn_cast<pypto::TransposeOp>(operation)) return verifyAlignedTranspose(op);
  if (auto op = dyn_cast<pypto::ReinterpretViewOp>(operation)) return verifyAlignedReinterpretView(op);
  if (auto op = dyn_cast<pypto::ReductionOp>(operation)) return verifyAlignedReduction(op);
  if (auto op = dyn_cast<pypto::LoadOp>(operation)) return verifyAlignedLoad(op);
  if (isSameShapeOp(operation) || isa<pypto::MaxOp, pypto::MinOp>(operation)) {
    SmallVector<Value> values(operation->getOperands());
    llvm::append_range(values, operation->getResults());
    return verifySameShape(operation, values);
  }
  if (auto op = dyn_cast<pypto::RowExpandAddOp>(operation)) return verifyAlignedRowExpand(op);
  if (auto op = dyn_cast<pypto::RowExpandSubOp>(operation)) return verifyAlignedRowExpand(op);
  if (auto op = dyn_cast<pypto::RowExpandMulOp>(operation)) return verifyAlignedRowExpand(op);
  if (auto op = dyn_cast<pypto::RowExpandDivOp>(operation)) return verifyAlignedRowExpand(op);
  if (auto op = dyn_cast<pypto::ColExpandAddOp>(operation)) return verifyAlignedColExpand(op);
  if (auto op = dyn_cast<pypto::ColExpandSubOp>(operation)) return verifyAlignedColExpand(op);
  if (auto op = dyn_cast<pypto::ColExpandMulOp>(operation)) return verifyAlignedColExpand(op);
  if (auto op = dyn_cast<pypto::ColExpandDivOp>(operation)) return verifyAlignedColExpand(op);
  return success();
}

struct PyptoAlignShapePass : impl::PyptoAlignShapeBase<PyptoAlignShapePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pypto::PyptoDialect, arith::ArithDialect>();
  }
  Shapes shapes;
  DenseMap<Value, SmallVector<unsigned>> axes;
  SmallVector<unsigned> parents;
  SmallVector<int64_t> bounds;

  unsigned root(unsigned axis) {
    while (parents[axis] != axis) {
      parents[axis] = parents[parents[axis]];
      axis = parents[axis];
    }
    return axis;
  }

  LogicalResult join(unsigned a, unsigned b) {
    a = root(a);
    b = root(b);
    if (a == b) return success();
    if ((bounds[a] < 0 || bounds[b] < 0) && bounds[a] != bounds[b]) return failure();
    parents[b] = a;
    bounds[a] = std::max(bounds[a], bounds[b]);
    return success();
  }

  LogicalResult equalShapes(ValueRange values) {
    Value first;
    for (Value value : values) {
      if (!axes.count(value)) continue;
      if (!first) {
        first = value;
        continue;
      }
      if (axes[first].size() != axes[value].size()) return failure();
      for (auto [a, b] : llvm::zip(axes[first], axes[value]))
        if (failed(join(a, b))) return failure();
    }
    return success();
  }

  LogicalResult mapReshapeShape(pypto::ReshapeOp op) {
    Value src = op->getOperand(0), dst = op->getResult(0);
    if (!axes.count(src) || !axes.count(dst)) return success();
    auto a = cast<pypto::TileType>(src.getType()).getShape();
    auto b = cast<pypto::TileType>(dst.getType()).getShape();
    if (a == b) return equalShapes(ValueRange{src, dst});
    SmallVector<unsigned> ai, bi;
    for (auto [i, dim] : llvm::enumerate(a))
      if (dim != 1) ai.push_back(i);
    for (auto [i, dim] : llvm::enumerate(b))
      if (dim != 1) bi.push_back(i);
    if (ai.size() != bi.size()) return success();
    for (auto [i, j] : llvm::zip(ai, bi))
      if (a[i] != b[j]) return success();
    for (auto [i, j] : llvm::zip(ai, bi))
      if (failed(join(axes[src][i], axes[dst][j]))) return failure();
    return success();
  }

  LogicalResult collectConstraints(Operation *op) {
    if (auto transpose = dyn_cast<pypto::TransposeOp>(op)) {
      Value src = transpose.getSource(), dst = transpose.getResult();
      if (!axes.count(src) || !axes.count(dst)) return success();
      if (axes[src].size() != axes[dst].size()) return failure();
      // Output padding belongs to the corresponding input axis, not the
      // same numbered axis. Keep the valid window on the original load.
      auto mapped = axes[src];
      std::swap(mapped[transpose.getI()], mapped[transpose.getJ()]);
      for (auto [a, b] : llvm::zip(mapped, axes[dst]))
        if (failed(join(a, b))) return failure();
      return success();
    }
    // These families require all tile operands/results to share one shape.
    // Broadcast vectors instead constrain only their non-broadcast axis.
    if (isSameShapeOp(op)) {
      SmallVector<Value> values(op->getOperands());
      llvm::append_range(values, op->getResults());
      return equalShapes(values);
    }
    if (auto view = dyn_cast<pypto::ReshapeOp>(op)) return mapReshapeShape(view);
    StringRef name = op->getName().getStringRef();
    if (!name.starts_with("pypto.row_expand_") && !name.starts_with("pypto.col_expand_")) return success();
    Value carrier = op->getOperand(0), vector = op->getOperand(1), result = op->getResult(0);
    if (failed(equalShapes(ValueRange{carrier, result}))) return failure();
    if (!axes.count(vector) || !axes.count(result)) return success();
    if (axes[vector].size() != 2 || axes[result].size() != 2) return failure();
    unsigned axis = name.starts_with("pypto.row_") ? 0 : 1;
    return join(axes[vector][axis], axes[result][axis]);
  }

  void solveShapes(ArrayRef<Operation *> ops) {
    // Supported element widths give power-of-two alignment units, so their
    // maximum satisfies every allocation sharing an axis. Only activate a
    // rounding requirement when its allocation grows, including growth on
    // another axis. A worklist propagates newly activated requirements.
    SmallVector<int64_t> alignment(parents.size(), 1);
    SmallVector<SmallVector<unsigned>> dependents(parents.size());
    SmallVector<unsigned> pending;
    for (Operation *op : ops) {
      if (!isa<pypto::LoadOp, pypto::TileFullOp>(op)) continue;
      Value value = op->getResult(0);
      if (!axes.count(value) || axes[value].empty()) continue;
      auto type = cast<pypto::TileType>(value.getType());
      unsigned bits = type.getElementType().getIntOrFloatBitWidth();
      if (bits < 8) continue;
      unsigned last = root(axes[value].back());
      alignment[last] = std::max(alignment[last], int64_t(256 / bits));
      for (auto [axis, old] : llvm::zip(axes[value], type.getShape())) {
        unsigned group = root(axis);
        dependents[group].push_back(last);
        if (bounds[group] != old) pending.push_back(last);
      }
    }
    while (!pending.empty()) {
      unsigned group = pending.pop_back_val();
      int64_t old = bounds[group], unit = alignment[group];
      if (old <= 0) continue;
      int64_t next = ((old + unit - 1) / unit) * unit;
      if (old == next) continue;
      bounds[group] = next;
      llvm::append_range(pending, dependents[group]);
    }
    for (auto &entry : shapes)
      for (auto [dim, axis] : llvm::zip(entry.second, axes[entry.first])) dim = bounds[root(axis)];
  }

  LogicalResult checkReshapeUsers(Value result, ArrayRef<int64_t> src) {
    for (Operation *user : result.getUsers()) {
      if (auto view = dyn_cast<pypto::ReshapeOp>(user)) {
        // A generated valid-window view can already carry the padded shape.
        if (shapes[result] == shapes[view->getResult(0)]) continue;
        auto dst = cast<pypto::TileType>(view->getResult(0).getType()).getShape();
        // Generated unflatten views can already describe the padded physical
        // shape while their source still carries its logical trailing extent.
        // Only permit regrouping an unchanged contiguous prefix here.
        auto prefixSize = [](ArrayRef<int64_t> dims) {
          int64_t size = 1;
          for (int64_t dim : dims.drop_back()) {
            if (dim <= 0 || size > std::numeric_limits<int64_t>::max() / dim) return int64_t(-1);
            size *= dim;
          }
          return size;
        };
        auto planned = ArrayRef<int64_t>(shapes[result]);
        if (!src.empty() && !dst.empty() && !view.getValidExtents().empty() && prefixSize(src) > 0 &&
            prefixSize(src) == prefixSize(dst) && llvm::equal(src.drop_back(), planned.drop_back()) &&
            planned.back() == dst.back())
          continue;
        SmallVector<int64_t> a, b;
        auto nonUnit = [](int64_t dim) { return dim != 1; };
        llvm::copy_if(src, std::back_inserter(a), nonUnit);
        llvm::copy_if(dst, std::back_inserter(b), nonUnit);
        if (a != b) return view.emitError("cannot propagate padding through a non-unit reshape");
      }
    }
    return success();
  }

  LogicalResult rewriteAllocation(Value result, pypto::TileType type, ArrayRef<int64_t> shape) {
    Operation *op = result.getDefiningOp();
    OpBuilder builder(op);
    auto constants = [&](ArrayRef<int64_t> dims) {
      SmallVector<Value> values;
      llvm::transform(dims, std::back_inserter(values),
                      [&](int64_t dim) -> Value { return builder.create<arith::ConstantIndexOp>(op->getLoc(), dim); });
      return values;
    };
    if (auto load = dyn_cast<pypto::LoadOp>(op)) {
      auto valid = load.getValidExtents();
      auto source = cast<ShapedType>(load.getSource().getType());
      if (source.getRank() != static_cast<int64_t>(shape.size()))
        return load.emitError("rank-reduced padding requires an explicit dimension mapping");
      SmallVector<Value> kept(valid.begin(), valid.end());
      if (kept.empty()) kept = constants(type.getShape());
      load.getPaddedShapesMutable().assign(constants(shape));
      load.getValidExtentsMutable().assign(kept);
    } else if (auto full = dyn_cast<pypto::TileFullOp>(op)) {
      // Broadcast carriers already have a valid re-view. Other full tiles
      // (notably mask-unpack branches) must retain their old valid window.
      full->setOperands(constants(shape));
      SmallVector<OpOperand *> uses;
      for (OpOperand &use : result.getUses()) {
        if (isa<pypto::ReshapeOp>(use.getOwner()) ||
            use.getOwner()->getName().getStringRef().starts_with("pypto.col_expand_"))
          continue;
        uses.push_back(&use);
      }
      if (!uses.empty()) {
        builder.setInsertionPointAfter(op);
        auto physical = constants(shape);
        auto valid = constants(type.getShape());
        auto viewType = pypto::TileType::get(&getContext(), type.getElementType(), shape);
        Value view =
          builder.create<pypto::ReshapeOp>(op->getLoc(), viewType, result, ValueRange(physical), ValueRange(valid));
        for (OpOperand *use : uses) use->set(view);
      }
    } else if (auto view = dyn_cast<pypto::ReshapeOp>(op)) {
      if (view.getValidExtents().empty()) view.getValidExtentsMutable().assign(constants(type.getShape()));
      view.getExtentsMutable().assign(constants(shape));
    }
    return success();
  }

  LogicalResult applyShapePlan(Operation *op) {
    for (Value result : op->getResults()) {
      auto it = shapes.find(result);
      if (it == shapes.end()) continue;
      auto type = cast<pypto::TileType>(result.getType());
      if (type.getShape() == ArrayRef<int64_t>(it->second)) continue;
      if (llvm::any_of(it->second, [](int64_t dim) { return dim < 0; }))
        return op->emitError("shape alignment requires static allocation bounds");
      if (!isSameShapeOp(op) && !isa<pypto::LoadOp, pypto::TileFullOp, pypto::ReshapeOp, pypto::TransposeOp>(op) &&
          !op->getName().getStringRef().contains("_expand_"))
        return op->emitError("shape alignment reached an unsupported producer");
      if (failed(checkReshapeUsers(result, type.getShape()))) return failure();
      if (failed(rewriteAllocation(result, type, it->second))) return failure();
      result.setType(pypto::TileType::get(&getContext(), type.getElementType(), it->second));
    }
    return success();
  }

  void runOnOperation() override {
    shapes.clear();
    axes.clear();
    parents.clear();
    bounds.clear();
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      ops.push_back(op);
      for (Value value : op->getResults()) {
        auto type = dyn_cast<pypto::TileType>(value.getType());
        if (!type) continue;
        shapes.try_emplace(value, type.getShape().begin(), type.getShape().end());
        for (int64_t dim : type.getShape()) {
          unsigned id = parents.size();
          axes[value].push_back(id);
          parents.push_back(id);
          bounds.push_back(dim);
        }
      }
    });
    auto incompatible = llvm::find_if(ops, [&](Operation *op) { return failed(collectConstraints(op)); });
    if (incompatible != ops.end()) {
      (*incompatible)->emitError("incompatible tile shape constraints");
      signalPassFailure();
      return;
    }
    solveShapes(ops);
    if (llvm::any_of(ops, [&](Operation *op) { return failed(applyShapePlan(op)); })) {
      signalPassFailure();
      return;
    }
    if (getOperation()
          .walk([](Operation *op) {
            return failed(verifyAlignedOperation(op)) ? WalkResult::interrupt() : WalkResult::advance();
          })
          .wasInterrupted())
      signalPassFailure();
  }
};
}  // namespace
std::unique_ptr<Pass> createPyptoAlignShapePass() { return std::make_unique<PyptoAlignShapePass>(); }
}  // namespace mlir
