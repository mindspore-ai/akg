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
      if (!isSameShapeOp(op) && !isa<pypto::LoadOp, pypto::TileFullOp, pypto::ReshapeOp>(op) &&
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
    }
  }
};
}  // namespace
std::unique_ptr<Pass> createPyptoAlignShapePass() { return std::make_unique<PyptoAlignShapePass>(); }
}  // namespace mlir
