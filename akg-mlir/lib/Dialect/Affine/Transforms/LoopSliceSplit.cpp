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

#include "akg/Dialect/Affine/Transforms/LoopSliceSplit.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>
#include <utility>

#include "akg/Utils/AnalysisCommon.hpp"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"

namespace mlir {
#ifndef GEN_PASS_DEF_LOOPSLICESPLIT
#define GEN_PASS_DEF_LOOPSLICESPLIT
#ifndef GEN_PASS_DECL_LOOPSLICESPLIT
#define GEN_PASS_DECL_LOOPSLICESPLIT
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
#endif
}  // namespace mlir

#define DEBUG_TYPE "loop-slice-split"

namespace mlir {
namespace {

struct Interval {
  int64_t lo, hi;
  [[nodiscard]] bool valid() const { return lo < hi; }
  [[nodiscard]] bool equals(const Interval &o) const { return lo == o.lo && hi == o.hi; }
  [[nodiscard]] bool contains(const Interval &o) const { return o.lo >= lo && o.hi <= hi; }
};

struct AccessDim {
  // sum_i (stride_i * iv[axis_i]) + offset. Empty contribs = pure constant.
  SmallVector<std::pair<int, int64_t>, 2> contribs;
  int64_t offset = 0;
  [[nodiscard]] bool isConst() const { return contribs.empty(); }
  // Single contributing axis iff the dim is exactly `iv[axis] + offset`; else -1.
  [[nodiscard]] int affineAxis() const {
    return (contribs.size() == 1 && contribs[0].second == 1) ? contribs[0].first : -1;
  }
};

struct PerfectNest {
  SmallVector<affine::AffineForOp, 4> loops;
  SmallVector<int64_t, 4> lo, hi;
  [[nodiscard]] int axisOf(Value v) const {
    for (size_t i = 0; i < loops.size(); ++i) {
      affine::AffineForOp f = loops[i];
      if (f.getInductionVar() == v) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }
};

struct AccessInfo {
  Operation *op = nullptr;
  Value memref;
  SmallVector<AccessDim, 4> dims;
};

struct SplitBoundary {
  Value memref;
  size_t dim;
  int64_t absolutePoint;
};

struct LoopInfo {
  affine::AffineForOp root;
  PerfectNest nest;
  SmallVector<AccessInfo, 4> stores, loads;
  bool ok = true;
};

static std::optional<PerfectNest> capturePerfectNest(affine::AffineForOp root) {
  SmallVector<affine::AffineForOp, 4> band;
  affine::getPerfectlyNestedLoops(band, root);
  if (band.empty()) {
    return std::nullopt;
  }
  for (Operation &op : band.back().getBody()->without_terminator()) {
    if (isa<affine::AffineForOp>(&op)) {
      return std::nullopt;
    }
  }
  PerfectNest nest;
  for (affine::AffineForOp f : band) {
    if (!f.hasConstantLowerBound() || !f.hasConstantUpperBound() || f.getStepAsInt() != 1 || !f.getResults().empty()) {
      return std::nullopt;
    }
    nest.loops.push_back(f);
    nest.lo.push_back(f.getConstantLowerBound());
    nest.hi.push_back(f.getConstantUpperBound());
  }
  return nest;
}

// `analyzeAccess` pre-normalizes via `getUnifiedAffineAccess`, so symbols are
// already demoted to dims — `e` is always an AffineDimExpr into the flat list.
static bool addAxisTerm(AffineExpr e, ValueRange operands, const PerfectNest &nest, AccessDim &out,
                        int64_t multiplier) {
  auto dim = dyn_cast<AffineDimExpr>(e);
  if (!dim || dim.getPosition() >= operands.size()) {
    return false;
  }
  int a = nest.axisOf(operands[dim.getPosition()]);
  if (a == -1) {
    return false;
  }
  out.contribs.push_back({a, multiplier});
  return true;
}

static bool parseExpr(AffineExpr e, ValueRange operands, const PerfectNest &nest, AccessDim &out) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) {
    out.offset += c.getValue();
    return true;
  }
  if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
    if (bin.getKind() == AffineExprKind::Add) {
      return parseExpr(bin.getLHS(), operands, nest, out) && parseExpr(bin.getRHS(), operands, nest, out);
    }
    if (bin.getKind() == AffineExprKind::Mul) {
      auto lhsC = dyn_cast<AffineConstantExpr>(bin.getLHS());
      auto rhsC = dyn_cast<AffineConstantExpr>(bin.getRHS());
      AffineExpr varExpr;
      int64_t k = 0;
      if (lhsC) {
        k = lhsC.getValue();
        varExpr = bin.getRHS();
      } else if (rhsC) {
        k = rhsC.getValue();
        varExpr = bin.getLHS();
      } else {
        return false;
      }
      return addAxisTerm(varExpr, operands, nest, out, k);
    }
    return false;
  }
  return addAxisTerm(e, operands, nest, out, 1);
}

template <typename OpT>
static std::optional<AccessInfo> analyzeAccess(OpT op, const PerfectNest &nest) {
  // Compose feeding affine.apply chains and demote symbols to dims so the
  // result map only contains AffineDimExpr.
  AffineMap map;
  SmallVector<Value, 4> operands;
  CommonUtils::getUnifiedAffineAccess(op.getOperation(), map, operands);
  if (!map) {
    return std::nullopt;
  }
  AccessInfo info;
  info.op = op.getOperation();
  info.memref = op.getMemRef();
  for (AffineExpr e : map.getResults()) {
    AccessDim d;
    if (!parseExpr(e, operands, nest, d)) {
      return std::nullopt;
    }
    info.dims.push_back(d);
  }
  return info;
}

static LoopInfo gatherLoopInfo(affine::AffineForOp root) {
  LoopInfo info;
  info.root = root;
  auto maybeNest = capturePerfectNest(root);
  if (!maybeNest) {
    info.ok = false;
    return info;
  }
  info.nest = std::move(*maybeNest);
  for (Operation &op : info.nest.loops.back().getBody()->without_terminator()) {
    if (auto s = dyn_cast<affine::AffineStoreOp>(&op)) {
      auto a = analyzeAccess(s, info.nest);
      if (!a) {
        info.ok = false;
        return info;
      }
      info.stores.push_back(*a);
    } else if (auto l = dyn_cast<affine::AffineLoadOp>(&op)) {
      auto a = analyzeAccess(l, info.nest);
      if (!a) {
        info.ok = false;
        return info;
      }
      info.loads.push_back(*a);
    }
  }
  return info;
}

static Interval intervalFor(const AccessDim &d, const PerfectNest &nest) {
  if (d.isConst()) {
    return {d.offset, d.offset + 1};
  }
  int64_t lo = d.offset;
  int64_t hi = d.offset + 1;
  for (auto &c : d.contribs) {
    int a = c.first;
    int64_t s = c.second;
    if (s >= 0) {
      lo += nest.lo[a] * s;
      hi += (nest.hi[a] - 1) * s;
    } else {
      lo += (nest.hi[a] - 1) * s;
      hi += nest.lo[a] * s;
    }
  }
  return {lo, hi};
}

static bool accessTouches(const AccessInfo &a, Value memref, ArrayRef<Interval> ref, const PerfectNest &nest) {
  if (a.memref != memref || a.dims.size() != ref.size()) {
    return false;
  }
  for (size_t d = 0; d < a.dims.size(); ++d) {
    Interval ia = intervalFor(a.dims[d], nest);
    if (ia.lo >= ref[d].hi || ref[d].lo >= ia.hi) {
      return false;
    }
  }
  return true;
}

// Conservative fallback for loops whose access analysis failed: returns true
// iff any op inside `root` uses `memref` as an operand. Lets producer/consumer
// scans skip unanalyzable loops that don't reference the memref of interest.
bool loopHasAccessTo(affine::AffineForOp root, Value memref) {
  WalkResult res = root.walk([&](Operation *op) -> WalkResult {
    if (llvm::is_contained(op->getOperands(), memref)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res.wasInterrupted();
}

static affine::AffineForOp getNestedAt(affine::AffineForOp root, int axis) {
  affine::AffineForOp cur = root;
  for (int a = 0; a < axis; ++a) {
    for (Operation &op : cur.getBody()->without_terminator()) {
      if (auto f = dyn_cast<affine::AffineForOp>(&op)) {
        cur = f;
        break;
      }
    }
  }
  return cur;
}

static void splitLoopAtPoints(LoopInfo &L, int splitAxis, ArrayRef<int64_t> splitPoints) {
  OpBuilder b(L.root);
  b.setInsertionPoint(L.root);
  for (size_t s = 0; s + 1 < splitPoints.size(); ++s) {
    affine::AffineForOp clone = cast<affine::AffineForOp>(b.clone(*L.root.getOperation()));
    affine::AffineForOp target = getNestedAt(clone, splitAxis);
    target.setConstantLowerBound(splitPoints[s]);
    target.setConstantUpperBound(splitPoints[s + 1]);
  }
  L.root.erase();
  L.ok = false;
}

// Replace `iv` in `user`'s affine map; `make` receives the dim/symbol expr
// currently denoting `iv` and returns the substitution. Returns false when
// `user` isn't a rewritable affine op (callers may fall back to SSA replace).
static bool substituteIVInAffineOp(Operation *user, Value iv, llvm::function_ref<AffineExpr(AffineExpr)> make) {
  AffineMap map;
  SmallVector<Value> operands;
  enum { Load, Store, Apply, Other } kind = Other;

  if (auto load = dyn_cast<affine::AffineLoadOp>(user)) {
    map = load.getAffineMap();
    operands.assign(load.getMapOperands().begin(), load.getMapOperands().end());
    kind = Load;
  } else if (auto store = dyn_cast<affine::AffineStoreOp>(user)) {
    map = store.getAffineMap();
    operands.assign(store.getMapOperands().begin(), store.getMapOperands().end());
    kind = Store;
  } else if (auto apply = dyn_cast<affine::AffineApplyOp>(user)) {
    map = apply.getAffineMap();
    operands.assign(user->getOperands().begin(), user->getOperands().end());
    kind = Apply;
  } else {
    return false;
  }

  int idx = -1;
  for (unsigned i = 0; i < operands.size(); ++i) {
    if (operands[i] == iv) {
      idx = static_cast<int>(i);
      break;
    }
  }
  if (idx == -1) {
    return true;
  }

  unsigned numDims = map.getNumDims();
  unsigned numSyms = map.getNumSymbols();
  MLIRContext *ctx = user->getContext();

  AffineExpr ivExpr =
    (idx < static_cast<int>(numDims)) ? getAffineDimExpr(idx, ctx) : getAffineSymbolExpr(idx - numDims, ctx);
  AffineExpr substExpr = make(ivExpr);

  SmallVector<AffineExpr, 4> dimRepls;
  for (unsigned k = 0; k < numDims; ++k) {
    dimRepls.push_back(getAffineDimExpr(k, ctx));
  }
  SmallVector<AffineExpr, 4> symRepls;
  for (unsigned k = 0; k < numSyms; ++k) {
    symRepls.push_back(getAffineSymbolExpr(k, ctx));
  }
  if (idx < static_cast<int>(numDims)) {
    dimRepls[idx] = substExpr;
  } else {
    symRepls[idx - numDims] = substExpr;
  }
  AffineMap newMap = map.replaceDimsAndSymbols(dimRepls, symRepls, numDims, numSyms);
  affine::canonicalizeMapAndOperands(&newMap, &operands);

  if (kind == Load) {
    auto load = cast<affine::AffineLoadOp>(user);
    SmallVector<Value> allOps;
    allOps.push_back(load.getMemRef());
    allOps.append(operands.begin(), operands.end());
    load.setMap(newMap);
    user->setOperands(allOps);
  } else if (kind == Store) {
    auto store = cast<affine::AffineStoreOp>(user);
    SmallVector<Value> allOps;
    allOps.push_back(store.getValueToStore());
    allOps.push_back(store.getMemRef());
    allOps.append(operands.begin(), operands.end());
    store.setMap(newMap);
    user->setOperands(allOps);
  } else if (kind == Apply) {
    auto apply = cast<affine::AffineApplyOp>(user);
    apply.setMap(newMap);
    user->setOperands(operands);
  }
  return true;
}

// ===---------------------------------------------------------------------=== //
// WAW shrink
// ===---------------------------------------------------------------------=== //

struct WawShrink {
  int axis;
  int diffDim;
  int64_t newBound;
  bool shrinkLowerBound;
};

static std::optional<WawShrink> detectWawShrink(const AccessInfo &sa, const PerfectNest &nestA, const AccessInfo &sb,
                                                const PerfectNest &nestB) {
  if (sa.memref != sb.memref || sa.dims.size() != sb.dims.size()) {
    return std::nullopt;
  }
  int diffDim = -1;
  for (size_t k = 0; k < sa.dims.size(); ++k) {
    Interval ra = intervalFor(sa.dims[k], nestA);
    Interval rb = intervalFor(sb.dims[k], nestB);
    if (ra.equals(rb)) {
      continue;
    }
    if (diffDim != -1 || !ra.contains(rb)) {
      return std::nullopt;
    }
    diffDim = static_cast<int>(k);
  }
  if (diffDim == -1) {
    return std::nullopt;
  }
  int axis = sa.dims[diffDim].affineAxis();
  if (axis == -1) {
    return std::nullopt;
  }
  int64_t off = sa.dims[diffDim].offset;
  int64_t aLo = nestA.lo[axis], aHi = nestA.hi[axis];
  Interval dead = intervalFor(sb.dims[diffDim], nestB);
  int64_t deadLo = dead.lo - off;
  int64_t deadHi = dead.hi - off;
  if (deadLo == aLo && deadHi < aHi) {
    return WawShrink{axis, diffDim, deadHi, true};
  }
  if (deadHi == aHi && deadLo > aLo) {
    return WawShrink{axis, diffDim, deadLo, false};
  }
  return std::nullopt;
}

static SmallVector<Interval, 4> buildDeadRegion(const AccessInfo &sa, const PerfectNest &nestA, const AccessInfo &sb,
                                                const PerfectNest &nestB, int diffDim) {
  SmallVector<Interval, 4> dead;
  for (size_t k = 0; k < sa.dims.size(); ++k) {
    dead.push_back(static_cast<int>(k) == diffDim ? intervalFor(sb.dims[k], nestB) : intervalFor(sa.dims[k], nestA));
  }
  return dead;
}

// ===---------------------------------------------------------------------=== //
// Interval-based axis split (consumer slice + producer mirror)
// ===---------------------------------------------------------------------=== //
//
// Both transforms: pick the first axis whose collector yields non-empty split
// points and call `splitLoopAtPoints`. They differ only in the collector —
// the consumer-slice variant finds producer write boundaries falling inside
// a consumer's load interval; the producer-mirror variant maps the
// consumer-side boundaries back onto the producer's IV via the store offset.

static bool dimsCompatible(const AccessInfo &st, const AccessInfo &load, size_t dimIdx, const PerfectNest &stNest,
                           const PerfectNest &consumerNest) {
  if (st.memref != load.memref || st.dims.size() != load.dims.size()) {
    return false;
  }
  for (size_t dd = 0; dd < load.dims.size(); ++dd) {
    if (dd == dimIdx) {
      continue;
    }
    if (!intervalFor(st.dims[dd], stNest).contains(intervalFor(load.dims[dd], consumerNest))) {
      return false;
    }
  }
  return true;
}

static bool applyIntersection(Interval inter, Interval &remaining) {
  if (inter.lo == remaining.lo && inter.hi <= remaining.hi) {
    remaining.lo = inter.hi;
  } else if (inter.hi == remaining.hi && inter.lo >= remaining.lo) {
    remaining.hi = inter.lo;
  } else {
    remaining = {0, 0};
    return true;
  }
  return false;
}

static void collectProducerSplits(const AccessInfo &load, size_t dimIdx, const PerfectNest &consumerNest,
                                  ArrayRef<LoopInfo> earlier, SmallVectorImpl<int64_t> &pts,
                                  SmallVectorImpl<SplitBoundary> &boundaries) {
  const AccessDim &d = load.dims[dimIdx];
  int axis = d.affineAxis();
  if (axis == -1) {
    return;
  }
  int64_t c = d.offset;
  Interval absRange{consumerNest.lo[axis] + c, consumerNest.hi[axis] + c};
  Interval remaining = absRange;

  auto record = [&](int64_t p) {
    if (p > absRange.lo && p < absRange.hi) {
      pts.push_back(p - c);
      boundaries.push_back({load.memref, dimIdx, p});
    }
  };

  for (int k = static_cast<int>(earlier.size()) - 1; k >= 0 && remaining.valid(); --k) {
    if (!earlier[k].ok) {
      continue;
    }
    for (const AccessInfo &st : earlier[k].stores) {
      if (!dimsCompatible(st, load, dimIdx, earlier[k].nest, consumerNest)) {
        continue;
      }
      Interval rs = intervalFor(st.dims[dimIdx], earlier[k].nest);
      Interval inter{std::max(rs.lo, remaining.lo), std::min(rs.hi, remaining.hi)};
      if (!inter.valid()) {
        continue;
      }
      record(inter.lo);
      record(inter.hi);
      if (applyIntersection(inter, remaining)) {
        break;
      }
    }
  }
}

static SmallVector<int64_t, 4> collectAxisSplitPoints(const LoopInfo &L, size_t axis, ArrayRef<LoopInfo> earlier,
                                                      SmallVectorImpl<SplitBoundary> &boundaries) {
  SmallVector<int64_t, 4> pts;
  for (const AccessInfo &load : L.loads) {
    for (size_t d = 0; d < load.dims.size(); ++d) {
      if (load.dims[d].affineAxis() == static_cast<int>(axis)) {
        collectProducerSplits(load, d, L.nest, earlier, pts, boundaries);
      }
    }
  }
  llvm::sort(pts);
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
  SmallVector<int64_t, 4> filtered;
  std::copy_if(pts.begin(), pts.end(), std::back_inserter(filtered),
               [lo = L.nest.lo[axis], hi = L.nest.hi[axis]](int64_t p) { return p > lo && p < hi; });
  return filtered;
}

static SmallVector<int64_t, 4> collectMirrorSplitPoints(const LoopInfo &L, size_t axis,
                                                        ArrayRef<SplitBoundary> consumerBds) {
  SmallVector<int64_t, 4> ivPts;
  for (const AccessInfo &st : L.stores) {
    for (size_t d = 0; d < st.dims.size(); ++d) {
      if (st.dims[d].affineAxis() != static_cast<int>(axis)) {
        continue;
      }
      int64_t off = st.dims[d].offset;
      for (const SplitBoundary &b : consumerBds) {
        if (b.memref != st.memref || b.dim != d) {
          continue;
        }
        int64_t p = b.absolutePoint - off;
        if (p > L.nest.lo[axis] && p < L.nest.hi[axis]) {
          ivPts.push_back(p);
        }
      }
    }
  }
  llvm::sort(ivPts);
  ivPts.erase(std::unique(ivPts.begin(), ivPts.end()), ivPts.end());
  return ivPts;
}

// Shared driver: split at the first axis where `collect` yields non-empty
// points. Side data is accumulated via captures in the callback.
template <typename Collect>
static bool runAxisSplit(SmallVectorImpl<LoopInfo> &loops, Collect collect) {
  bool changed = false;
  for (size_t i = 0; i < loops.size(); ++i) {
    if (!loops[i].ok) {
      continue;
    }
    LoopInfo &L = loops[i];
    for (size_t axis = 0; axis < L.nest.loops.size(); ++axis) {
      auto pts = collect(L, axis, i);
      if (pts.empty()) {
        continue;
      }
      SmallVector<int64_t, 4> splitPoints;
      splitPoints.push_back(L.nest.lo[axis]);
      splitPoints.append(pts.begin(), pts.end());
      splitPoints.push_back(L.nest.hi[axis]);
      splitLoopAtPoints(L, static_cast<int>(axis), splitPoints);
      changed = true;
      break;
    }
  }
  return changed;
}

// ===---------------------------------------------------------------------=== //
// Producer shrink by consumer use
// ===---------------------------------------------------------------------=== //
//
// For each single-store producer (address `iv + offset`), shrink its bound on
// that axis to the union bbox of subsequent loads. Bail if the live region
// can't be bounded from visible reads: later loop unanalyzable, re-writes the
// memref, or has a partial overlap on a non-`d` dim.

enum class DimRel { Contained, Disjoint, Partial };

// Classify a consumer load's bbox vs a producer store's bbox on every dim
// except `skipDim`. Shared by both consumer-use scan and stride scan.
static DimRel classifyOtherDims(const AccessInfo &st, const AccessInfo &ld, size_t skipDim, const PerfectNest &stNest,
                                const PerfectNest &ldNest) {
  for (size_t dd = 0; dd < ld.dims.size(); ++dd) {
    if (dd == skipDim) {
      continue;
    }
    Interval rst = intervalFor(st.dims[dd], stNest);
    Interval rld = intervalFor(ld.dims[dd], ldNest);
    if (rst.contains(rld)) {
      continue;
    }
    if (rld.lo >= rst.hi || rst.lo >= rld.hi) {
      return DimRel::Disjoint;
    }
    return DimRel::Partial;
  }
  return DimRel::Contained;
}

// ===---------------------------------------------------------------------=== //
// Strided-consumer producer split
// ===---------------------------------------------------------------------=== //
//
// If a stride-1 producer writes `[0, N)` on some axis and every consumer
// reads via the same stride `s > 1` with offsets mod s covering {0..s-1},
// clone the producer into `s` copies each iterating `[0, N/s)` and writing
// residue-`k` via `iv*s + k`. Producer iteration now matches consumers' —
// enables vertical fusion. Pre-conditions are checked inline below.

struct StrideScan {
  int64_t stride = 0;
  llvm::SmallSetVector<int64_t, 4> offsetClasses;
};

// True iff offsetClasses == {0..stride-1} with stride > 1.
static bool stridedSplitResiduesCover(const StrideScan &sc) {
  if (sc.stride <= 1 || sc.offsetClasses.size() != static_cast<size_t>(sc.stride)) {
    return false;
  }
  for (int64_t k = 0; k < sc.stride; ++k) {
    if (!sc.offsetClasses.contains(k)) {
      return false;
    }
  }
  return true;
}

// Every axis-IV user must be an affine op `substituteIVInAffineOp` can rewrite.
static bool stridedSplitFoldable(affine::AffineForOp loop) {
  for (Operation *u : loop.getInductionVar().getUsers()) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineApplyOp>(u)) {
      return false;
    }
  }
  return true;
}

// Clone the producer `stride` times; each clone iterates `[0, halfSize)` and
// rewrites IV uses to `iv*stride + k` so the k-th copy writes residue-k.
static void stridedSplitTransform(LoopInfo &L, int axis, int64_t stride, int64_t halfSize) {
  affine::AffineForOp root = L.root;
  OpBuilder b(root);
  b.setInsertionPoint(root);

  for (int64_t k = 0; k < stride; ++k) {
    affine::AffineForOp clone = cast<affine::AffineForOp>(b.clone(*root.getOperation()));
    affine::AffineForOp targetLoop = getNestedAt(clone, axis);
    targetLoop.setConstantLowerBound(0);
    targetLoop.setConstantUpperBound(halfSize);

    Value newIv = targetLoop.getInductionVar();
    llvm::SmallSetVector<Operation *, 4> users;
    for (Operation *u : newIv.getUsers()) {
      users.insert(u);
    }
    for (Operation *u : users) {
      substituteIVInAffineOp(u, newIv, [stride, k](AffineExpr ivExpr) { return ivExpr * stride + k; });
    }
  }

  root.erase();
  L.ok = false;
}

// ===---------------------------------------------------------------------=== //
// Post-split cleanup
// ===---------------------------------------------------------------------=== //
//
// Two kinds of post-split residue: loops with non-zero lower bound, and
// loops with trip count 1. `normalizeNonZeroLb` folds `iv + lb` into user
// maps (avoiding the `affine.apply` MLIR's stock normalize would emit);
// `promoteSingletonLoop` substitutes the IV with `lb` and inlines the body.

static bool promoteSingletonLoop(affine::AffineForOp forOp) {
  if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
    return false;
  }
  int64_t lo = forOp.getConstantLowerBound();
  int64_t hi = forOp.getConstantUpperBound();
  int64_t step = forOp.getStepAsInt();
  if (step <= 0 || (hi - lo) > step) {
    return false;
  }

  Value iv = forOp.getInductionVar();
  AffineExpr loConst = getAffineConstantExpr(lo, forOp.getContext());

  llvm::SmallSetVector<Operation *, 4> users;
  for (Operation *u : iv.getUsers()) {
    users.insert(u);
  }
  for (Operation *u : users) {
    if (!substituteIVInAffineOp(u, iv, [&](AffineExpr) { return loConst; })) {
      OpBuilder b(u);
      auto cOp = b.create<arith::ConstantIndexOp>(forOp.getLoc(), lo);
      u->replaceUsesOfWith(iv, cOp.getResult());
    }
  }

  Block *body = forOp.getBody();
  for (Operation &op : llvm::make_early_inc_range(*body)) {
    if (isa<affine::AffineYieldOp>(&op)) {
      continue;
    }
    op.moveBefore(forOp);
  }
  forOp.erase();
  return true;
}

static bool normalizeNonZeroLb(affine::AffineForOp forOp) {
  if (!forOp.hasConstantLowerBound() || !forOp.hasConstantUpperBound()) {
    return false;
  }
  int64_t lb = forOp.getConstantLowerBound();
  int64_t ub = forOp.getConstantUpperBound();
  int64_t step = forOp.getStepAsInt();
  if (lb == 0 || step <= 0 || (ub - lb) <= step) {
    return false;
  }

  Value iv = forOp.getInductionVar();
  llvm::SmallSetVector<Operation *, 4> users;
  for (Operation *u : iv.getUsers()) {
    users.insert(u);
  }
  for (Operation *u : users) {
    if (!isa<affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineApplyOp>(u)) {
      return false;
    }
  }

  AffineExpr lbExpr = getAffineConstantExpr(lb, forOp.getContext());
  for (Operation *u : users) {
    substituteIVInAffineOp(u, iv, [&](AffineExpr ivExpr) { return ivExpr + lbExpr; });
  }

  forOp.setConstantLowerBound(0);
  forOp.setConstantUpperBound(ub - lb);
  return true;
}

// Rewrites top-level affine loop nests so producer/consumer pairs become
// structurally fusable. Each top-level loop becomes a `LoopInfo` (perfect
// nest + analyzed loads/stores); the stages below run in order:
//
//   WAW shrink:              drop stores fully overwritten later.
//   Consumer slice split:    split a consumer at producer write boundaries.
//   Producer mirror split:   mirror those boundaries onto the producer.
//   Producer shrink:         shrink producer bound to union bbox of later loads.
//   Strided producer split:  clone producer into `s` copies for stride-`s` consumers.
//   Post-split cleanup:      fold non-zero lower bounds, promote singleton loops.
//
// Each transforming stage re-`gather()`s a fresh `loops` snapshot before
// running. The cleanup stages walk the IR directly.

struct LoopSliceSplit : impl::LoopSliceSplitBase<LoopSliceSplit> {
  LoopSliceSplit() = default;

  void runOnOperation() override {
    func = getOperation();
    if (func.isExternal() || func.getBody().empty()) {
      return;
    }

    runWawShrink();
    runSliceSplit();
    runProducerMirrorSplit();
    runProducerShrinkByConsumerUse();
    runProducerStridedSplit();
    normalizeLowerBounds();
    promoteSingletons();
  }

 private:
  func::FuncOp func;
  SmallVector<LoopInfo, 16> loops;
  SmallVector<SplitBoundary, 8> boundaries;

  // Refresh `loops` from the current IR. Each stage that depends on `loops`
  // calls this first, so transformations that erased or created loops in
  // earlier stages don't leave us with stale handles.
  void gather() {
    loops.clear();
    for (Operation &op : func.getBody().front()) {
      if (auto f = dyn_cast<affine::AffineForOp>(&op)) {
        loops.push_back(gatherLoopInfo(f));
      }
    }
  }

  // WAW shrink
  void runWawShrink() {
    gather();
    for (size_t i = 0; i < loops.size(); ++i) {
      if (!loops[i].ok || loops[i].stores.size() != 1) {
        continue;
      }
      const AccessInfo &sa = loops[i].stores.front();
      for (size_t j = i + 1; j < loops.size(); ++j) {
        if (!loops[j].ok) {
          continue;
        }
        for (const AccessInfo &sb : loops[j].stores) {
          auto spec = detectWawShrink(sa, loops[i].nest, sb, loops[j].nest);
          if (!spec) {
            continue;
          }
          auto dead = buildDeadRegion(sa, loops[i].nest, sb, loops[j].nest, spec->diffDim);
          if (isWawDeadRegionBlocked(i, j, sa.memref, dead)) {
            continue;
          }
          affine::AffineForOp &axisLoop = loops[i].nest.loops[spec->axis];
          if (spec->shrinkLowerBound) {
            axisLoop.setConstantLowerBound(spec->newBound);
            loops[i].nest.lo[spec->axis] = spec->newBound;
          } else {
            axisLoop.setConstantUpperBound(spec->newBound);
            loops[i].nest.hi[spec->axis] = spec->newBound;
          }
        }
      }
    }
  }

  [[nodiscard]] bool isWawDeadRegionBlocked(size_t i, size_t j, Value memref, ArrayRef<Interval> dead) const {
    for (size_t k = i + 1; k < j; ++k) {
      if (!loops[k].ok) {
        if (loopHasAccessTo(loops[k].root, memref)) {
          return true;
        }
        continue;
      }
      if (std::any_of(loops[k].loads.begin(), loops[k].loads.end(),
                      [&](const AccessInfo &x) { return accessTouches(x, memref, dead, loops[k].nest); })) {
        return true;
      }
      if (std::any_of(loops[k].stores.begin(), loops[k].stores.end(),
                      [&](const AccessInfo &x) { return accessTouches(x, memref, dead, loops[k].nest); })) {
        return true;
      }
    }
    return false;
  }

  // Interval-based axis split
  void runSliceSplit() {
    gather();
    runAxisSplit(loops, [&](const LoopInfo &L, size_t axis, size_t i) {
      SmallVector<SplitBoundary, 4> bds;
      auto pts = collectAxisSplitPoints(L, axis, ArrayRef<LoopInfo>(loops).take_front(i), bds);
      if (!pts.empty()) {
        boundaries.append(bds.begin(), bds.end());
      }
      return pts;
    });
  }

  void runProducerMirrorSplit() {
    if (boundaries.empty()) {
      return;
    }
    gather();
    runAxisSplit(loops, [&](const LoopInfo &L, size_t axis, size_t /*i*/) {
      return collectMirrorSplitPoints(L, axis, boundaries);
    });
  }

  // Producer shrink by consumer use
  void runProducerShrinkByConsumerUse() {
    gather();
    for (size_t i = 0; i < loops.size(); ++i) {
      if (!loops[i].ok || loops[i].stores.size() != 1) {
        continue;
      }
      LoopInfo &L = loops[i];
      const AccessInfo &st = L.stores.front();

      for (size_t d = 0; d < st.dims.size(); ++d) {
        int axis = st.dims[d].affineAxis();
        if (axis == -1) {
          continue;
        }
        int64_t off = st.dims[d].offset;
        int64_t writeLo = L.nest.lo[axis] + off;
        int64_t writeHi = L.nest.hi[axis] + off;

        bool anyUse = false;
        auto useRange = scanConsumerUseRange(i, st, d, L.nest, anyUse);
        if (!useRange || !anyUse) {
          continue;
        }
        int64_t useLo = std::max(useRange->lo, writeLo);
        int64_t useHi = std::min(useRange->hi, writeHi);
        if (useLo >= useHi || (useLo == writeLo && useHi == writeHi)) {
          continue;
        }

        affine::AffineForOp &axisLoop = L.nest.loops[axis];
        if (useLo > writeLo) {
          axisLoop.setConstantLowerBound(useLo - off);
          L.nest.lo[axis] = useLo - off;
        }
        if (useHi < writeHi) {
          axisLoop.setConstantUpperBound(useHi - off);
          L.nest.hi[axis] = useHi - off;
        }
        break;
      }
    }
  }

  // Union of dim-`d` ranges of consumer loads of `st`'s memref in loops after
  // `i`. Returns nullopt on disqualifying state (see producer-shrink section).
  // `anyUse` is set iff at least one consuming load was found.
  std::optional<Interval> scanConsumerUseRange(size_t i, const AccessInfo &st, size_t d, const PerfectNest &stNest,
                                               bool &anyUse) const {
    int64_t useLo = std::numeric_limits<int64_t>::max();
    int64_t useHi = std::numeric_limits<int64_t>::min();
    anyUse = false;
    for (size_t j = i + 1; j < loops.size(); ++j) {
      if (!loops[j].ok) {
        if (loopHasAccessTo(loops[j].root, st.memref)) {
          return std::nullopt;
        }
        continue;
      }
      if (std::any_of(loops[j].stores.begin(), loops[j].stores.end(),
                      [&](const AccessInfo &midSt) { return midSt.memref == st.memref; })) {
        return std::nullopt;
      }
      for (const AccessInfo &ld : loops[j].loads) {
        if (ld.memref != st.memref || ld.dims.size() != st.dims.size()) {
          continue;
        }
        DimRel rel = classifyOtherDims(st, ld, d, stNest, loops[j].nest);
        if (rel == DimRel::Partial) {
          return std::nullopt;
        }
        if (rel == DimRel::Disjoint) {
          continue;
        }
        Interval ldRange = intervalFor(ld.dims[d], loops[j].nest);
        useLo = std::min(useLo, ldRange.lo);
        useHi = std::max(useHi, ldRange.hi);
        anyUse = true;
      }
    }
    return Interval{useLo, useHi};
  }

  // Strided-consumer producer split
  void runProducerStridedSplit() {
    gather();
    for (size_t i = 0; i < loops.size(); ++i) {
      if (!loops[i].ok || loops[i].stores.size() != 1) {
        continue;
      }
      LoopInfo &L = loops[i];
      const AccessInfo &st = L.stores.front();

      for (size_t d = 0; d < st.dims.size(); ++d) {
        int axis = st.dims[d].affineAxis();
        if (axis == -1) {
          continue;
        }
        int64_t off = st.dims[d].offset;
        int64_t writeLo = L.nest.lo[axis] + off;
        int64_t writeHi = L.nest.hi[axis] + off;
        if (writeLo != 0 || (writeHi - writeLo) <= 1) {
          continue;
        }
        int64_t writeSize = writeHi - writeLo;

        auto sc = scanConsumerStride(i, st, d, L.nest);
        if (!sc || !stridedSplitResiduesCover(*sc)) {
          continue;
        }
        if (writeSize % sc->stride != 0) {
          continue;
        }
        if (!stridedSplitFoldable(L.nest.loops[axis])) {
          continue;
        }

        stridedSplitTransform(L, axis, sc->stride, writeSize / sc->stride);
        break;
      }
    }
  }

  // Collect stride / offset-mod classes of consumer loads on dim `d` in loops
  // after `i`. Returns nullopt if any consumer disqualifies the split.
  [[nodiscard]] std::optional<StrideScan> scanConsumerStride(size_t i, const AccessInfo &st, size_t d,
                                                             const PerfectNest &stNest) const {
    StrideScan sc;
    for (size_t j = i + 1; j < loops.size(); ++j) {
      if (!loops[j].ok) {
        if (loopHasAccessTo(loops[j].root, st.memref)) {
          return std::nullopt;
        }
        continue;
      }
      if (std::any_of(loops[j].stores.begin(), loops[j].stores.end(),
                      [&](const AccessInfo &midSt) { return midSt.memref == st.memref; })) {
        return std::nullopt;
      }
      for (const AccessInfo &ld : loops[j].loads) {
        if (ld.memref != st.memref || ld.dims.size() != st.dims.size()) {
          continue;
        }
        DimRel rel = classifyOtherDims(st, ld, d, stNest, loops[j].nest);
        if (rel == DimRel::Partial) {
          return std::nullopt;
        }
        if (rel == DimRel::Disjoint) {
          continue;
        }
        const AccessDim &ldDim = ld.dims[d];
        if (ldDim.contribs.size() != 1) {
          return std::nullopt;
        }
        int64_t s = ldDim.contribs[0].second;
        if (s <= 0) {
          return std::nullopt;
        }
        if (sc.stride == 0) {
          sc.stride = s;
        } else if (sc.stride != s) {
          return std::nullopt;
        }
        sc.offsetClasses.insert(((ldDim.offset % s) + s) % s);
      }
    }
    return sc;
  }

  // Post-split cleanup
  // Fold `(iv + lb)` into each user's affine map; falls back to stock normalize
  // (which emits `affine.apply`) when a user isn't a load/store/apply.
  void normalizeLowerBounds() {
    SmallVector<affine::AffineForOp, 8> targets;
    func.walk([&](affine::AffineForOp f) {
      if (!f.hasConstantLowerBound() || !f.hasConstantUpperBound()) {
        return;
      }
      int64_t lb = f.getConstantLowerBound();
      int64_t ub = f.getConstantUpperBound();
      int64_t step = f.getStepAsInt();
      if (lb != 0 && step > 0 && (ub - lb) > step) {
        targets.push_back(f);
      }
    });
    for (auto f : targets) {
      if (!normalizeNonZeroLb(f)) {
        (void)affine::normalizeAffineFor(f);
      }
    }
  }

  // Promote trip-count-1 loops to a fixed point; inner singletons exposed by
  // removing an outer one are also picked up.
  void promoteSingletons() {
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<affine::AffineForOp, 8> singletons;
      func.walk([&](affine::AffineForOp f) {
        if (!f.hasConstantLowerBound() || !f.hasConstantUpperBound()) {
          return;
        }
        int64_t lo = f.getConstantLowerBound();
        int64_t hi = f.getConstantUpperBound();
        int64_t step = f.getStepAsInt();
        if (step > 0 && (hi - lo) <= step) {
          singletons.push_back(f);
        }
      });
      if (std::any_of(singletons.begin(), singletons.end(),
                      [](affine::AffineForOp f) { return promoteSingletonLoop(f); })) {
        changed = true;
      }
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLoopSliceSplitPass() { return std::make_unique<LoopSliceSplit>(); }

}  // namespace mlir
