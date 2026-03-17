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

#include "akg/Dialect/Linalg/Transforms/ShapeNormalization.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <memory>
#include <type_traits>
#include <variant>

#include "akg/Analysis/SymbolicShapeAnalysis.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
#define GEN_PASS_DECL_SHAPENORMALIZATION
#define GEN_PASS_DEF_SHAPENORMALIZATION
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using mlir::linalg::GenericOp;

namespace mlir {
namespace {

struct GlobalTargetInfo {
  SmallVector<std::string> symShape;
  SmallVector<int64_t> shape;
};

struct ShapeNormalState;

struct OpAdapter {
  virtual ~OpAdapter() = default;
  virtual bool match(Operation *op) const = 0;
  virtual void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const = 0;
  virtual void assignLabels(Operation *op, ShapeNormalState &state) const = 0;
  virtual void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const = 0;
};

struct OpAdapterRegistry {
  SmallVector<std::unique_ptr<OpAdapter>> adapters;
  void registerAdapter(std::unique_ptr<OpAdapter> adapter) { adapters.push_back(std::move(adapter)); }
  const OpAdapter *get(Operation *op) const {
    if (!op) return nullptr;
    for (auto &ptr : adapters)
      if (ptr && ptr->match(op)) return ptr.get();
    return nullptr;
  }
};

static const OpAdapterRegistry &getOpAdapterRegistry();

using ShapeDimValue = std::variant<Value, std::string, int64_t>;

struct ShapeNormalState {
  SymbolicShapeAnalysis &manager;
  bool isSupported = true;
  llvm::StringMap<int64_t> axisSizes;
  llvm::StringMap<std::pair<Value, int64_t>> DynSymtoArgDim;
  DenseMap<Value, DenseMap<int64_t, Value>> shapeStoreMap;
  DenseSet<Value> ValuesinFunc;
  DenseMap<Operation *, SmallVector<SmallVector<std::string>>> returnOpShapes;
  DenseMap<Operation *, SmallVector<std::string>> affineMapSymDims;
  DenseMap<Operation *, SmallVector<int64_t>> UpdatedAxes;
  DenseMap<Value, Value> ssaMap;
  DenseSet<Operation *> entryMaterializedOps;
  DenseSet<Operation *> toDeleteOps;
  llvm::StringMap<SmallVector<std::string>> decompositions;
  llvm::StringMap<GlobalTargetInfo> globalTargets;

  ShapeNormalState() : manager(SymbolicShapeAnalysis::getInstance()) { axisSizes["1"] = 1; }

  SmallVector<std::string> getDecomposition(const std::string &axis) const {
    auto it = decompositions.find(axis);
    if (it != decompositions.end()) {
      return it->second;
    }
    return {};
  }

  SmallVector<int64_t> getAxesSizes(const SmallVector<std::string> &axes) {
    SmallVector<int64_t> sizes;
    std::transform(axes.begin(), axes.end(), std::back_inserter(sizes),
                   [this](const std::string &axis) { return axisSizes[axis]; });
    return sizes;
  }

  SmallVector<std::string> expandGroupedAxes(const SmallVector<std::string> &groupedAxes) {
    SmallVector<std::string> result;
    for (const std::string &axis : groupedAxes) {
      SmallVector<std::string> expandedAxes = expandAxisRecursively(axis);
      result.append(expandedAxes);
    }

    SmallVector<std::string> newResult;
    std::copy_if(result.begin(), result.end(), std::back_inserter(newResult),
                 [](const std::string &axis) { return axis != "1"; });
    return newResult;
  }

  SmallVector<std::string> expandAxisRecursively(const std::string &axis) {
    auto decomposition = getDecomposition(axis);
    if (decomposition.empty()) {
      return {axis};
    }

    SmallVector<std::string> result;
    for (const std::string &childAxis : decomposition) {
      SmallVector<std::string> childExpansion = expandAxisRecursively(childAxis);
      result.append(childExpansion);
    }
    return result;
  }

  struct FinestDecompositionResult {
    bool success = true;
    SmallVector<std::string> finestAxes;
  };

  struct FinestSlices {
    SmallVector<SmallVector<std::string>> groupedAxes1;
    SmallVector<SmallVector<std::string>> groupedAxes2;
  };

  FinestSlices unifyAxesGroupsToSlices(const SmallVector<std::string> &axesgroup1,
                                       const SmallVector<std::string> &axesgroup2) {
    FinestSlices result;
    SmallVector<int64_t> sizes1 = getAxesSizes(axesgroup1);
    SmallVector<int64_t> sizes2 = getAxesSizes(axesgroup2);

    SmallVector<SmallVector<std::string>> groupedAxes1;
    SmallVector<SmallVector<std::string>> groupedAxes2;

    SmallVector<std::string> tempAxes1;
    SmallVector<int64_t> tempSizes1;
    SmallVector<std::string> tempAxes2;
    SmallVector<int64_t> tempSizes2;

    size_t idx1 = 1, idx2 = 1;
    int64_t accum1 = sizes1[0], accum2 = sizes2[0];
    tempAxes1.push_back(axesgroup1[0]);
    tempSizes1.push_back(sizes1[0]);
    tempAxes2.push_back(axesgroup2[0]);
    tempSizes2.push_back(sizes2[0]);

    while (idx1 < axesgroup1.size() || idx2 < axesgroup2.size()) {
      if (accum1 == accum2) {
        groupedAxes1.push_back(tempAxes1);
        groupedAxes2.push_back(tempAxes2);

        tempAxes1.clear();
        tempSizes1.clear();
        tempAxes2.clear();
        tempSizes2.clear();
        accum1 = sizes1[idx1];
        tempAxes1.push_back(axesgroup1[idx1]);
        tempSizes1.push_back(sizes1[idx1]);
        accum2 = sizes2[idx2];
        tempAxes2.push_back(axesgroup2[idx2]);
        tempSizes2.push_back(sizes2[idx2]);

        idx1++;
        idx2++;
      } else if (accum1 < accum2) {
        accum1 *= sizes1[idx1];
        tempAxes1.push_back(axesgroup1[idx1]);
        tempSizes1.push_back(sizes1[idx1]);
        idx1++;
      } else if (accum1 > accum2) {
        accum2 *= sizes2[idx2];
        tempAxes2.push_back(axesgroup2[idx2]);
        tempSizes2.push_back(sizes2[idx2]);
        idx2++;
      }
    }

    if (idx1 == axesgroup1.size() || idx2 == axesgroup2.size()) {
      while (idx1 < axesgroup1.size()) {
        tempAxes1.push_back(axesgroup1[idx1]);
        tempSizes1.push_back(sizes1[idx1]);
        idx1++;
      }
      while (idx2 < axesgroup2.size()) {
        tempAxes2.push_back(axesgroup2[idx2]);
        tempSizes2.push_back(sizes2[idx2]);
        idx2++;
      }
      groupedAxes1.push_back(tempAxes1);
      groupedAxes2.push_back(tempAxes2);
    }
    result.groupedAxes1 = groupedAxes1;
    result.groupedAxes2 = groupedAxes2;
    return result;
  }

  std::optional<FinestDecompositionResult> tryUnifySingleAxisCases(const SmallVector<std::string> &axes1,
                                                                   const SmallVector<std::string> &axes2) {
    if (axes1.size() == 1 && axes2.size() == 1) {
      FinestDecompositionResult r;
      r.finestAxes.push_back(globalReplaceAxis(axes1[0], axes2[0]));
      return r;
    }
    if (axes1.size() == 1) {
      FinestDecompositionResult r;
      updateDecomposition(axes1[0], axes2);
      r.finestAxes.append(axes2);
      return r;
    }
    if (axes2.size() == 1) {
      FinestDecompositionResult r;
      updateDecomposition(axes2[0], axes1);
      r.finestAxes.append(axes1);
      return r;
    }
    return std::nullopt;
  }

  void skipLeadingSizeOne(const SmallVector<std::string> &axes, const SmallVector<int64_t> &sizes, std::string &axis,
                          int64_t &size, int64_t &idx) {
    while (size == 1 && idx < static_cast<int64_t>(axes.size())) {
      axis = axes[idx];
      size = sizes[idx];
      idx++;
    }
  }

  bool mergeSmallerAxis(const SmallVector<std::string> &axes1, const SmallVector<int64_t> &sizes1,
                        const SmallVector<std::string> &axes2, const SmallVector<int64_t> &sizes2, std::string &axis1,
                        int64_t &size1, int64_t &idx1, std::string &axis2, int64_t &size2, int64_t &idx2,
                        FinestDecompositionResult &result) {
    const bool smallFirst = size1 < size2;
    int64_t smallSize = smallFirst ? size1 : size2;
    int64_t largeSize = smallFirst ? size2 : size1;
    const auto &smallAxes = smallFirst ? axes1 : axes2;
    const auto &smallSizes = smallFirst ? sizes1 : sizes2;
    int64_t &smallIdx = smallFirst ? idx1 : idx2;
    std::string &smallAxis = smallFirst ? axis1 : axis2;
    std::string &largeAxis = smallFirst ? axis2 : axis1;

    int64_t tmpSize = smallSize;
    SmallVector<std::string> tmpAxis = {smallAxis};
    while (smallIdx < static_cast<int64_t>(smallAxes.size())) {
      if (tmpSize * smallSizes[smallIdx] >= largeSize) break;
      tmpSize *= smallSizes[smallIdx];
      tmpAxis.push_back(smallAxes[smallIdx]);
      smallIdx++;
    }
    int64_t remainder = largeSize % tmpSize;
    result.finestAxes.push_back(axis1);
    result.finestAxes.append(tmpAxis);

    if (remainder != 0) {
      result.finestAxes.clear();
      result.success = false;
      result.finestAxes.append(sizes1[0] < sizes2[0] ? axes1 : axes2);
      return false;
    }
    int64_t newdimSize = largeSize / tmpSize;
    std::string newdim = createNewSymbolicDim(newdimSize);
    tmpAxis.push_back(newdim);
    updateDecomposition(largeAxis, tmpAxis);
    largeAxis = newdim;
    smallAxis = newdim;
    if (smallFirst) {
      size2 = newdimSize;
      if (smallIdx < static_cast<int64_t>(axes1.size())) {
        axis1 = axes1[smallIdx];
        size1 = sizes1[smallIdx];
        idx1 = smallIdx + 1;
      } else {
        axis1 = newdim;
        size1 = newdimSize;
      }
    } else {
      size1 = newdimSize;
      if (smallIdx < static_cast<int64_t>(axes2.size())) {
        axis2 = axes2[smallIdx];
        size2 = sizes2[smallIdx];
        idx2 = smallIdx + 1;
      } else {
        axis2 = newdim;
        size2 = newdimSize;
      }
    }
    return true;
  }

  FinestDecompositionResult unifySlicesToFinest(const SmallVector<std::string> &axes1,
                                                const SmallVector<std::string> &axes2) {
    SmallVector<int64_t> sizes1 = getAxesSizes(axes1);
    SmallVector<int64_t> sizes2 = getAxesSizes(axes2);
    FinestDecompositionResult result;
    result.success = true;

    if (auto trivial = tryUnifySingleAxisCases(axes1, axes2)) return *trivial;

    std::string axis1 = axes1[0], axis2 = axes2[0];
    int64_t size1 = sizes1[0], size2 = sizes2[0];
    int64_t idx1 = 1, idx2 = 1;
    skipLeadingSizeOne(axes1, sizes1, axis1, size1, idx1);
    skipLeadingSizeOne(axes2, sizes2, axis2, size2, idx2);

    while (idx1 < static_cast<int64_t>(axes1.size()) - 1 && idx2 < static_cast<int64_t>(axes2.size()) - 1) {
      if (!mergeSmallerAxis(axes1, sizes1, axes2, sizes2, axis1, size1, idx1, axis2, size2, idx2, result))
        return result;
    }

    SmallVector<std::string> tail;
    std::string parentAxis;
    if (size1 > size2) {
      tail.push_back(axis2);
      parentAxis = axis1;
    } else {
      tail.push_back(axis1);
      parentAxis = axis2;
    }
    for (; idx1 < static_cast<int64_t>(axes1.size()); idx1++) tail.push_back(axes1[idx1]);
    for (; idx2 < static_cast<int64_t>(axes2.size()); idx2++) tail.push_back(axes2[idx2]);
    result.finestAxes.append(tail);
    updateDecomposition(parentAxis, tail);
    return result;
  }

  FinestDecompositionResult unifyAxesGroupsToFinest(const SmallVector<std::string> &axesgroup1,
                                                    const SmallVector<std::string> &axesgroup2) {
    FinestDecompositionResult result;

    if (axesgroup1.empty() && axesgroup2.empty()) {
      return result;
    }
    SmallVector<std::string> expandedaxesgroup1 = expandGroupedAxes(axesgroup1);
    SmallVector<std::string> expandedaxesgroup2 = expandGroupedAxes(axesgroup2);

    // TODO(akg): This judgment may need to be improved in the future by the expression of dynamic axis length
    SmallVector<std::string> dynAxes1, dynAxes2;
    for (const auto &axis : expandedaxesgroup1) {
      if (axisSizes.lookup(axis) == ShapedType::kDynamic) dynAxes1.push_back(axis);
    }
    for (const auto &axis : expandedaxesgroup2) {
      if (axisSizes.lookup(axis) == ShapedType::kDynamic) dynAxes2.push_back(axis);
    }
    if (dynAxes1 != dynAxes2) {
      // TODO(akg): static axes being broken by dynamic axes should also return false.
      result.success = false;
      result.finestAxes = expandedaxesgroup2;
      return result;
    }

    FinestSlices Slices = unifyAxesGroupsToSlices(expandedaxesgroup1, expandedaxesgroup2);
    SmallVector<std::string> finestAxes;
    for (size_t i = 0; i < Slices.groupedAxes1.size(); i++) {
      FinestDecompositionResult tmpresult = unifySlicesToFinest(Slices.groupedAxes1[i], Slices.groupedAxes2[i]);
      if (!tmpresult.success) {
        result.success = false;
      }
      finestAxes.append(tmpresult.finestAxes);
    }
    result.finestAxes = finestAxes;
    return result;
  }

  void updateDecomposition(const std::string &parentAxis, const SmallVector<std::string> &childAxes) {
    SmallVector<std::string> newChildAxes;
    std::copy_if(childAxes.begin(), childAxes.end(), std::back_inserter(newChildAxes),
                 [](const std::string &axis) { return axis != "1"; });
    if (newChildAxes[0] == parentAxis) {
      return;
    }
    if (newChildAxes.size() == 1) {
      globalReplaceAxis(parentAxis, newChildAxes[0]);
      return;
    }
    auto existingIt = decompositions.find(parentAxis);
    if (existingIt != decompositions.end()) {
      FinestDecompositionResult result = unifyAxesGroupsToFinest(existingIt->second, newChildAxes);
      decompositions[parentAxis] = result.finestAxes;
      return;
    }
    decompositions[parentAxis] = newChildAxes;
  }

  std::string createNewSymbolicDim(int64_t dimSize) {
    if (dimSize == 1) {
      return "1";
    }
    std::string axis = manager.newSymbolicDim();
    axisSizes[axis] = dimSize;
    return axis;
  }

  Type updateValueSymbolicShape(Value v, const SmallVector<std::string> &newShape) {
    Type newType = manager.updateSymbolicShape(v.getType(), newShape);
    v.setType(newType);
    return newType;
  }

  Type removeValueSymbolicShape(Value v) {
    Type newType = manager.removeSymbolicShape(v.getType());
    v.setType(newType);
    return newType;
  }

  SmallVector<std::string> getValueSymbolicShape(Value v) const {
    return manager.getSymbolicShapeAutoComplete(v.getType());
  }

  std::string globalReplaceAxis(const std::string &Axis1, const std::string &Axis2) {
    if (Axis1 == "1") {
      return "1";
    }
    std::string oldAxis, newAxis;

    if (std::stoi(Axis1.substr(1)) < std::stoi(Axis2.substr(1))) {
      newAxis = Axis1;
      oldAxis = Axis2;
    } else if (std::stoi(Axis1.substr(1)) > std::stoi(Axis2.substr(1))) {
      newAxis = Axis2;
      oldAxis = Axis1;
    } else {
      return Axis1;
    }

    for (Value v : ValuesinFunc) {
      bool needUpdate = false;
      SmallVector<std::string> newSymShape;
      auto symShape = manager.getSymbolicShapeAutoComplete(v.getType());
      for (const std::string &sym : symShape) {
        if (sym == oldAxis) {
          newSymShape.push_back(newAxis);
          needUpdate = true;
        } else {
          newSymShape.push_back(sym);
        }
      }
      if (needUpdate) {
        Type newType = manager.updateSymbolicShape(v.getType(), newSymShape);
        v.setType(newType);
      }
    }

    auto oldkeyIt = decompositions.find(oldAxis);
    if (oldkeyIt != decompositions.end()) {
      updateDecomposition(newAxis, oldkeyIt->second);
      decompositions.erase(oldkeyIt);
    }

    for (auto &entry : decompositions) {
      for (size_t i = 0; i < entry.second.size(); i++) {
        if (entry.second[i] == oldAxis) {
          entry.second[i] = newAxis;
        }
      }
    }

    for (auto &entry : affineMapSymDims) {
      for (size_t i = 0; i < entry.second.size(); i++) {
        if (entry.second[i] == oldAxis) {
          entry.second[i] = newAxis;
        }
      }
    }

    if (DynSymtoArgDim.find(oldAxis) != DynSymtoArgDim.end()) {
      if (DynSymtoArgDim[oldAxis].second == -1 && isa<BlockArgument>(DynSymtoArgDim[oldAxis].first)) {
        DynSymtoArgDim[newAxis] = DynSymtoArgDim[oldAxis];
      }
    }

    return newAxis;
  }

  void globalReplaceAxes(const SmallVector<std::string> &oldAxes, const SmallVector<std::string> &newAxes) {
    if (oldAxes.size() != newAxes.size()) return;
    for (size_t i = 0; i < oldAxes.size(); ++i) {
      globalReplaceAxis(oldAxes[i], newAxes[i]);
    }
  }

  std::string getOrCreateSymDimNameAndSize(const SmallVector<std::string> &rawGroup) {
    bool isAllOne = !std::any_of(rawGroup.begin(), rawGroup.end(), [](const std::string &axis) { return axis != "1"; });
    if (isAllOne) {
      return "1";
    }

    SmallVector<std::string> group;
    std::copy_if(rawGroup.begin(), rawGroup.end(), std::back_inserter(group),
                 [](const std::string &axis) { return axis != "1"; });
    int64_t groupSize = 1;
    std::string groupSymShape;

    for (const std::string &axis : group) {
      int64_t axisSize = axisSizes[axis];
      if (axisSize == -1) {
        groupSize = ShapedType::kDynamic;
      }
      groupSize *= axisSize;
    }

    if (group.size() == 1) {
      groupSymShape = group[0];
    } else {
      auto it = std::find_if(decompositions.begin(), decompositions.end(),
                             [&group](const auto &decomp) { return decomp.second == group; });
      if (it != decompositions.end()) {
        groupSymShape = it->getKey().str();
      } else {
        groupSymShape = createNewSymbolicDim(groupSize);
        axisSizes[groupSymShape] = groupSize;
        manager.assignLabel(groupSymShape, manager.getLabel(group[0]));
        decompositions[groupSymShape] = group;
      }
    }
    return groupSymShape;
  }

  SmallVector<std::string> computeTargetSymShape(Value value) {
    auto symShape = getValueSymbolicShape(value);
    if (symShape.empty()) {
      return {};
    }
    return computeTargetSymShape(symShape);
  }

  SmallVector<std::string> computeTargetSymShape(const SmallVector<std::string> &symShape) {
    return computeTargetSymShapeWithMapping(symShape).first;
  }

  std::pair<SmallVector<std::string>, SmallVector<std::pair<int64_t, int64_t>>> computeTargetSymShapeWithMapping(
    const SmallVector<std::string> &symShape) {
    SmallVector<std::string> flattenedAxes = expandGroupedAxes(symShape);
    if (flattenedAxes.size() == 0) return {{}, {}};

    SmallVector<int64_t> flattenedAxisToSymShapeIndex(flattenedAxes.size(), -1);
    int64_t currentPos = 0;
    for (size_t j = 0; j < symShape.size(); ++j) {
      SmallVector<std::string> expanded = expandAxisRecursively(symShape[j]);
      for (const std::string &a : expanded) {
        if (a != "1") {
          flattenedAxisToSymShapeIndex[currentPos++] = (int64_t)j;
        }
      }
    }

    llvm::StringMap<int> nameCount;
    for (const std::string &axis : flattenedAxes) {
      nameCount[axis]++;
    }

    SmallVector<SmallVector<std::string>> targetDimGroups;
    SmallVector<std::string> currentGroup;
    int64_t currentLabel = manager.getLabel(flattenedAxes[0]);
    int64_t nextGroupIdx = 0;
    SmallVector<std::pair<int64_t, int64_t>> duplicateAxisMapping;
    for (size_t i = 0; i < flattenedAxes.size(); ++i) {
      const std::string &axis = flattenedAxes[i];
      bool isDuplicateName = nameCount[axis] > 1;
      int64_t label = manager.getLabel(axis);

      if (isDuplicateName) {
        if (!currentGroup.empty()) {
          targetDimGroups.push_back(currentGroup);
          nextGroupIdx++;
          currentGroup.clear();
        }
        targetDimGroups.push_back({axis});
        duplicateAxisMapping.push_back(std::make_pair(flattenedAxisToSymShapeIndex[i], nextGroupIdx));
        nextGroupIdx++;
        if (i + 1 < flattenedAxes.size()) {
          currentLabel = manager.getLabel(flattenedAxes[i + 1]);
        }
      } else {
        if (label != currentLabel) {
          targetDimGroups.push_back(currentGroup);
          nextGroupIdx++;
          currentGroup.clear();
          currentLabel = label;
        }
        currentGroup.push_back(axis);
      }
    }
    if (!currentGroup.empty()) {
      targetDimGroups.push_back(currentGroup);
    }

    SmallVector<std::string> targetSymShape;
    for (const SmallVector<std::string> &group : targetDimGroups) {
      std::string targetSymDim = getOrCreateSymDimNameAndSize(group);
      targetSymShape.push_back(targetSymDim);
    }
    return {targetSymShape, duplicateAxisMapping};
  }

  Value CastToTargetSymShape(Value v, const SmallVector<std::string> &targetSymShape, PatternRewriter &rewriter,
                             Location loc) {
    auto symShape = getValueSymbolicShape(v);
    if (symShape == targetSymShape) return v;
    Type targetTy = manager.updateSymbolicShape(v.getType(), targetSymShape);
    auto castOp = rewriter.create<memref::MemorySpaceCastOp>(loc, targetTy, v);
    entryMaterializedOps.insert(castOp);
    return castOp.getResult();
  }

  Type getMemRefTypeWithInheritedStrides(ArrayRef<int64_t> targetShape, Type elementType, MemRefType inputType,
                                         const SmallVector<ReassociationIndices> &reassociation, bool isExpand) {
    // Only support collapsing into a single axis
    if (isExpand) {
      return MemRefType::get(targetShape, elementType);
    }
    auto stridedLayout = dyn_cast<StridedLayoutAttr>(inputType.getLayout());
    if (!stridedLayout) return MemRefType::get(targetShape, elementType);
    ArrayRef<int64_t> inputStrides = stridedLayout.getStrides();
    int64_t inputOffset = stridedLayout.getOffset();
    SmallVector<int64_t> targetStrides;
    for (const ReassociationIndices &group : reassociation) {
      if (!group.empty()) {
        int64_t s = inputStrides[group.back()];
        targetStrides.push_back(s);
      }
    }
    if (targetStrides.size() != targetShape.size() ||
        llvm::any_of(targetStrides, [](int64_t s) { return s == ShapedType::kDynamic; }) ||
        inputOffset == ShapedType::kDynamic) {
      return MemRefType::get(targetShape, elementType);
    }
    auto newLayout = StridedLayoutAttr::get(inputType.getContext(), inputOffset, targetStrides);
    return MemRefType::get(targetShape, elementType, newLayout, inputType.getMemorySpace());
  }

  SmallVector<ReassociationIndices> buildReassociation(const SmallVector<std::string> &coarseShape,
                                                       const SmallVector<std::string> &fineShape) {
    SmallVector<ReassociationIndices> reassociation;
    int64_t tmpTargetSize = 1;
    size_t fineInd = 0;
    for (size_t coarseInd = 0; coarseInd < coarseShape.size(); ++coarseInd) {
      ReassociationIndices group;
      int64_t coarseSize = axisSizes[coarseShape[coarseInd]];
      if (coarseSize == ShapedType::kDynamic) {
        SmallVector<std::string> targetGroupFinestShape = expandAxisRecursively(coarseShape[coarseInd]);
        SmallVector<std::string> tmpGroupFinestShape;
        while (tmpGroupFinestShape != targetGroupFinestShape) {
          if (axisSizes[fineShape[fineInd]] == 1) {
            group.push_back(static_cast<int64_t>(fineInd++));
            continue;
          }
          tmpGroupFinestShape.append(expandAxisRecursively(fineShape[fineInd]));
          group.push_back(static_cast<int64_t>(fineInd));
          fineInd++;
        }
      }
      while (tmpTargetSize < coarseSize) {
        group.push_back(static_cast<int64_t>(fineInd));
        tmpTargetSize *= axisSizes[fineShape[fineInd]];
        fineInd++;
      }

      while (coarseInd == coarseShape.size() - 1 && fineInd < fineShape.size()) {
        group.push_back(static_cast<int64_t>(fineInd));
        fineInd++;
      }
      reassociation.push_back(group);
      tmpTargetSize = 1;
    }
    return reassociation;
  }

  void stripOnesFromShape(const SmallVector<std::string> &symShape, SmallVector<std::string> &stripped,
                          int64_t &oneCount) {
    oneCount = 0;
    for (const auto &axis : symShape) {
      if (axis != "1")
        stripped.push_back(axis);
      else
        oneCount++;
    }
  }

  SmallVector<Operation *> reshapeWithMemref(Value newRet, const SmallVector<std::string> &inputSymshape,
                                             const SmallVector<std::string> &targetSymShape, Type elementType,
                                             PatternRewriter &rewriter, Location loc) {
    SmallVector<Operation *> ops;
    SmallVector<int64_t> targetSizes = getAxesSizes(targetSymShape);
    int64_t rank = static_cast<int64_t>(targetSymShape.size());
    // TODO(akg): Unify the shape value to i64 for now, modify if needed later
    auto i64Ty = rewriter.getI64Type();
    auto shapeMemrefTy = MemRefType::get({rank}, i64Ty);
    Value shapeMemref = rewriter.create<memref::AllocOp>(loc, shapeMemrefTy);
    entryMaterializedOps.insert(shapeMemref.getDefiningOp());
    for (int64_t i = 0; i < rank; ++i) {
      Value dimVal;
      if (targetSizes[i] != ShapedType::kDynamic) {
        dimVal = rewriter.create<arith::ConstantIndexOp>(loc, targetSizes[i]).getResult();
      } else {
        auto [src, idx] = DynSymtoArgDim[targetSymShape[i]];
        if (idx == -1) {
          if (src.getType() != i64Ty) {
            dimVal = rewriter.create<arith::IndexCastOp>(loc, i64Ty, src).getResult();
          } else {
            dimVal = src;
          }
        } else {
          dimVal = rewriter.create<memref::DimOp>(loc, src, idx).getResult();
          if (dimVal.getType() != i64Ty) {
            dimVal = rewriter.create<arith::IndexCastOp>(loc, i64Ty, dimVal).getResult();
          }
        }
      }
      entryMaterializedOps.insert(dimVal.getDefiningOp());
      rewriter.create<memref::StoreOp>(loc, dimVal, shapeMemref,
                                       ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, i)});
    }
    Type targetRetType = manager.updateSymbolicShape(MemRefType::get(targetSizes, elementType), inputSymshape);
    auto reshapeOp = rewriter.create<memref::ReshapeOp>(loc, targetRetType, newRet, shapeMemref);
    entryMaterializedOps.insert(reshapeOp);
    ops.push_back(reshapeOp);
    Value newOut = CastToTargetSymShape(reshapeOp.getResult(), targetSymShape, rewriter, loc);
    ops.push_back(newOut.getDefiningOp());
    return ops;
  }

  SmallVector<Operation *> reshapeValueToSymShape(Value newRet, const SmallVector<std::string> &targetSymShape,
                                                  PatternRewriter &rewriter, Location loc) {
    SmallVector<Operation *> ops;
    SmallVector<std::string> inputSymshape = getValueSymbolicShape(newRet);
    auto elementType = cast<MemRefType>(newRet.getType()).getElementType();

    if (inputSymshape == targetSymShape) return ops;

    Type newType =
      manager.updateSymbolicShape(MemRefType::get(getAxesSizes(targetSymShape), elementType), inputSymshape);

    if (inputSymshape.empty()) {
      SmallVector<ReassociationIndices> reassociation;
      auto expandOp = rewriter.create<memref::ExpandShapeOp>(loc, newType, newRet, reassociation);
      ops.push_back(expandOp);
      entryMaterializedOps.insert(expandOp);
      ops.push_back(CastToTargetSymShape(expandOp.getResult(), targetSymShape, rewriter, loc).getDefiningOp());
      return ops;
    }
    if (targetSymShape.empty()) {
      SmallVector<ReassociationIndices> reassociation;
      auto collapseOp = rewriter.create<memref::CollapseShapeOp>(loc, newType, newRet, reassociation);
      ops.push_back(collapseOp);
      entryMaterializedOps.insert(collapseOp);
      ops.push_back(CastToTargetSymShape(collapseOp.getResult(), targetSymShape, rewriter, loc).getDefiningOp());
      return ops;
    }

    auto unifyResult = unifyAxesGroupsToFinest(inputSymshape, targetSymShape);
    if (!unifyResult.success) {
      return reshapeWithMemref(newRet, inputSymshape, targetSymShape, elementType, rewriter, loc);
    }

    SmallVector<std::string> finestAxes = unifyResult.finestAxes;
    SmallVector<std::string> inputStripped;
    SmallVector<std::string> targetStripped;
    int64_t input1Num = 0;
    int64_t target1Num = 0;
    stripOnesFromShape(inputSymshape, inputStripped, input1Num);
    stripOnesFromShape(targetSymShape, targetStripped, target1Num);

    if (input1Num == 0 && finestAxes == targetStripped) {
      auto expandOp =
        rewriter.create<memref::ExpandShapeOp>(loc, newType, newRet, buildReassociation(inputSymshape, targetSymShape));
      ops.push_back(expandOp);
      entryMaterializedOps.insert(expandOp);
      ops.push_back(CastToTargetSymShape(expandOp.getResult(), targetSymShape, rewriter, loc).getDefiningOp());
      return ops;
    }
    if ((target1Num == 0 && finestAxes == inputStripped) || (inputStripped == targetSymShape)) {
      auto reassociation = buildReassociation(targetSymShape, inputSymshape);
      auto inputMemRefTy = cast<MemRefType>(newRet.getType());
      Type collapseTargetType = getMemRefTypeWithInheritedStrides(getAxesSizes(targetSymShape), elementType,
                                                                  inputMemRefTy, reassociation, false);
      collapseTargetType = manager.updateSymbolicShape(collapseTargetType, inputSymshape);
      auto collapseOp = rewriter.create<memref::CollapseShapeOp>(loc, collapseTargetType, newRet, reassociation);
      ops.push_back(collapseOp);
      entryMaterializedOps.insert(collapseOp);
      ops.push_back(CastToTargetSymShape(collapseOp.getResult(), targetSymShape, rewriter, loc).getDefiningOp());
      return ops;
    }

    if (input1Num > 0) {
      Type midType =
        manager.updateSymbolicShape(MemRefType::get(getAxesSizes(inputStripped), elementType), inputSymshape);
      auto collapseOp = rewriter.create<memref::CollapseShapeOp>(loc, midType, newRet,
                                                                 buildReassociation(inputStripped, inputSymshape));
      ops.push_back(collapseOp);
      newRet = collapseOp.getResult();
      entryMaterializedOps.insert(collapseOp);
    }
    if (inputStripped != finestAxes) {
      Type midType = manager.updateSymbolicShape(MemRefType::get(getAxesSizes(finestAxes), elementType), inputSymshape);
      auto expandOp =
        rewriter.create<memref::ExpandShapeOp>(loc, midType, newRet, buildReassociation(inputStripped, finestAxes));
      ops.push_back(expandOp);
      newRet = expandOp.getResult();
      entryMaterializedOps.insert(expandOp);
    }
    if (finestAxes != targetStripped) {
      Type midType =
        manager.updateSymbolicShape(MemRefType::get(getAxesSizes(targetStripped), elementType), inputSymshape);
      auto collapseOp =
        rewriter.create<memref::CollapseShapeOp>(loc, midType, newRet, buildReassociation(targetStripped, finestAxes));
      ops.push_back(collapseOp);
      newRet = collapseOp.getResult();
      entryMaterializedOps.insert(collapseOp);
    }
    if (targetStripped != targetSymShape) {
      Type midType =
        manager.updateSymbolicShape(MemRefType::get(getAxesSizes(targetSymShape), elementType), inputSymshape);
      auto expandOp = rewriter.create<memref::ExpandShapeOp>(loc, midType, newRet,
                                                             buildReassociation(targetStripped, targetSymShape));
      ops.push_back(expandOp);
      newRet = expandOp.getResult();
      entryMaterializedOps.insert(expandOp);
    }
    ops.push_back(CastToTargetSymShape(newRet, targetSymShape, rewriter, loc).getDefiningOp());
    return ops;
  }

  bool isSupportedOp(Operation *op) const { return getOpAdapterRegistry().get(op) != nullptr; }
  void unifyToFinestAxes(Operation *op) {
    if (const OpAdapter *a = getOpAdapterRegistry().get(op)) a->unifyToFinestAxes(op, *this);
  }
  void assignLabels(Operation *op) {
    if (const OpAdapter *a = getOpAdapterRegistry().get(op)) a->assignLabels(op, *this);
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) {
    if (const OpAdapter *a = getOpAdapterRegistry().get(op)) a->rewrite(op, *this, rewriter);
  }
};

struct AllocAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::AllocOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto alloc = cast<memref::AllocOp>(op);
    Value result = alloc.getResult();
    auto resultType = dyn_cast<MemRefType>(result.getType());
    if (!resultType) return;
    auto dynSizes = alloc.getDynamicSizes();
    size_t dynIdx = 0;
    SmallVector<std::string> symShape;
    for (int64_t i = 0; i < resultType.getRank(); ++i) {
      int64_t dimSize = resultType.getDimSize(i);
      std::string sym = state.createNewSymbolicDim(dimSize);
      symShape.push_back(sym);
      if (dimSize == ShapedType::kDynamic && dynIdx < dynSizes.size() &&
          state.DynSymtoArgDim.find(sym) == state.DynSymtoArgDim.end()) {
        if (auto dimOp = dyn_cast<memref::DimOp>(dynSizes[dynIdx].getDefiningOp())) {
          auto cstOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
          assert(cstOp && "dim index must be constant");
          int64_t idxVal = cast<IntegerAttr>(cstOp.getValue()).getValue().getSExtValue();
          state.DynSymtoArgDim[sym] = std::make_pair(dimOp.getSource(), idxVal);
        } else {
          state.DynSymtoArgDim[sym] = std::make_pair(dynSizes[dynIdx], static_cast<int64_t>(-1));
        }
        dynIdx++;
      }
    }
    state.updateValueSymbolicShape(result, symShape);
    state.ValuesinFunc.insert(result);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto alloc = cast<memref::AllocOp>(op);
    Location loc = op->getLoc();
    Value oldResult = alloc.getResult();
    SmallVector<std::string> targetSymShape = state.computeTargetSymShape(oldResult);
    SmallVector<int64_t> targetShape = state.getAxesSizes(targetSymShape);
    auto oldResultTy = cast<MemRefType>(oldResult.getType());
    MemRefType newResultType = MemRefType::get(targetShape, oldResultTy.getElementType());
    newResultType = cast<MemRefType>(state.manager.updateSymbolicShape(newResultType, targetSymShape));
    rewriter.setInsertionPoint(op);
    SmallVector<Value> dynamicSizes;
    llvm::transform(alloc.getDynamicSizes(), std::back_inserter(dynamicSizes),
                   [&state](Value v) { return state.ssaMap.lookup(v) ? state.ssaMap.lookup(v) : v; });
    Value newAlloc = rewriter.create<memref::AllocOp>(loc, newResultType, dynamicSizes, alloc.getSymbolOperands(),
                                                      alloc.getAlignmentAttr());
    state.ssaMap[oldResult] = newAlloc;
    oldResult.replaceAllUsesWith(newAlloc);
    state.toDeleteOps.insert(op);
  }
};

struct ConstantAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<arith::ConstantOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto constantOp = cast<arith::ConstantOp>(op);
    Value cstValue = constantOp.getResult();
    auto rankedType = dyn_cast<MemRefType>(cstValue.getType());
    if (!rankedType) return;
    SmallVector<std::string> symShape;
    for (int64_t i = 0; i < rankedType.getRank(); ++i)
      symShape.push_back(state.createNewSymbolicDim(rankedType.getDimSize(i)));
    state.updateValueSymbolicShape(cstValue, symShape);
    state.ValuesinFunc.insert(cstValue);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto constantOp = cast<arith::ConstantOp>(op);
    auto cstValue = constantOp.getResult();
    auto rankedType = dyn_cast<MemRefType>(cstValue.getType());
    if (!rankedType) return;
    Type newType = cstValue.getType();
    auto oldAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
    auto newShapedTy = mlir::dyn_cast<ShapedType>(newType);
    if (!newShapedTy) return;
    DenseElementsAttr newAttr = oldAttr.reshape(newShapedTy);
    rewriter.setInsertionPoint(constantOp);
    auto newCstOp = rewriter.create<arith::ConstantOp>(constantOp.getLoc(), newType, newAttr);
    Value newCst = newCstOp.getResult();
    SmallVector<std::string> targetSymShape = state.computeTargetSymShape(newCst);
    SmallVector<int64_t> targetShapeInt = state.getAxesSizes(targetSymShape);
    MemRefType cstShape = dyn_cast<MemRefType>(newCst.getType());
    MemRefType targetShape = MemRefType::get(targetShapeInt, cstShape.getElementType());
    Value finalValue = newCst;
    if (cstShape.getShape() != targetShape.getShape()) {
      Location loc = newCstOp->getLoc();
      rewriter.setInsertionPointAfter(newCstOp);
      auto reshapeOps = state.reshapeValueToSymShape(newCst, targetSymShape, rewriter, loc);
      if (!reshapeOps.empty()) {
        finalValue = reshapeOps.back()->getResult(0);
        state.entryMaterializedOps.insert(reshapeOps.back());
      }
    }
    state.ssaMap[cstValue] = finalValue;
    cstValue.replaceAllUsesWith(finalValue);
    state.toDeleteOps.insert(constantOp);
  }
};

struct ArithOpAdapter final : OpAdapter {
  bool match(Operation *op) const override {
    return op->getDialect() && op->getDialect()->getNamespace() == "arith" && !isa<arith::ConstantOp>(op);
  }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {}
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    IRMapping mapping;
    bool hasMappedOperand = false;
    for (Value operand : op->getOperands()) {
      if (Value newVal = state.ssaMap.lookup(operand)) {
        mapping.map(operand, newVal);
        hasMappedOperand = true;
      }
    }
    if (!hasMappedOperand) return;
    rewriter.setInsertionPoint(op);
    Operation *newOp = rewriter.clone(*op, mapping);
    for (OpResult oldResult : op->getResults()) {
      Value newResult = newOp->getResult(oldResult.getResultNumber());
      state.ssaMap[oldResult] = newResult;
      oldResult.replaceAllUsesWith(newResult);
    }
    state.toDeleteOps.insert(op);
  }
};

struct AffineApplyAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<affine::AffineApplyOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {}
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    IRMapping mapping;
    bool hasMappedOperand = false;
    for (Value operand : op->getOperands()) {
      if (Value newVal = state.ssaMap.lookup(operand)) {
        mapping.map(operand, newVal);
        hasMappedOperand = true;
      }
    }
    if (!hasMappedOperand) return;
    rewriter.setInsertionPoint(op);
    Operation *newOp = rewriter.clone(*op, mapping);
    for (OpResult oldResult : op->getResults()) {
      Value newResult = newOp->getResult(oldResult.getResultNumber());
      state.ssaMap[oldResult] = newResult;
      oldResult.replaceAllUsesWith(newResult);
    }
    state.toDeleteOps.insert(op);
  }
};

struct StoreOpAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::StoreOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto storeOp = cast<memref::StoreOp>(op);
    Value memref = storeOp.getMemRef();
    Value value = storeOp.getValueToStore();
    auto indices = storeOp.getIndices();
    // TODO(akg): Future implementation of extract requires handling multi-dimensional indices
    if (indices.size() != 1) return;
    auto IndexOp = indices[0].getDefiningOp<arith::ConstantOp>();
    auto indAttr = dyn_cast<IntegerAttr>(IndexOp.getValue());
    std::optional<int64_t> idxOpt = indAttr.getValue().getSExtValue();
    state.shapeStoreMap[memref][*idxOpt] = value;
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {}
};

struct DimOpAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::DimOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {}
  void assignLabels(Operation *op, ShapeNormalState &state) const override {
    auto dimOp = cast<memref::DimOp>(op);
    Value source = dimOp.getSource();
    Value indexVal = dimOp.getIndex();
    std::optional<int64_t> idxOpt;
    if (auto cstOp = indexVal.getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(cstOp.getValue())) idxOpt = intAttr.getValue().getSExtValue();
    }
    if (!idxOpt.has_value()) return;
    int64_t idx = *idxOpt;
    SmallVector<std::string> symShape = state.getValueSymbolicShape(source);
    std::string symdim = symShape[idx];
    state.manager.assignLabel(symdim);
    state.affineMapSymDims[op] = {symdim};
  }
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    if (state.affineMapSymDims[op].size() < 1) return;
    std::string symdim = state.affineMapSymDims[op][0];
    auto dynIt = state.DynSymtoArgDim.find(symdim);
    if (dynIt == state.DynSymtoArgDim.end()) return;
    auto [arg, idx] = dynIt->second;
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    Value newResult;
    if (idx == -1) {
      if (arg.getType() != rewriter.getIndexType())
        newResult = rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), arg);
      else
        newResult = arg;
    } else {
      newResult = rewriter.create<memref::DimOp>(loc, arg, idx);
    }
    state.ssaMap[op->getResult(0)] = newResult;
    op->getResult(0).replaceAllUsesWith(newResult);
    state.toDeleteOps.insert(op);
  }
};

struct GlobalAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::GlobalOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto globalOp = cast<memref::GlobalOp>(op);
    ModuleOp module = globalOp->getParentOfType<ModuleOp>();
    StringRef symName = globalOp.getSymName();
    module.walk([&](memref::GetGlobalOp getGlobal) {
      if (getGlobal.getNameAttr().getLeafReference() != symName) return;
      Value result = getGlobal.getResult();
      auto resultType = dyn_cast<MemRefType>(result.getType());
      if (!resultType) return;
      SmallVector<std::string> symShape;
      for (int64_t i = 0; i < resultType.getRank(); ++i)
        symShape.push_back(state.createNewSymbolicDim(resultType.getDimSize(i)));
      state.updateValueSymbolicShape(result, symShape);
      state.ValuesinFunc.insert(result);
    });
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto globalOp = cast<memref::GlobalOp>(op);
    auto it = state.globalTargets.find(globalOp.getSymName().str());
    if (it == state.globalTargets.end()) return;
    const GlobalTargetInfo &target = it->second;
    MemRefType oldType = cast<MemRefType>(globalOp.getType());
    MemRefType baseNewType =
      MemRefType::get(target.shape, oldType.getElementType(), oldType.getLayout(), oldType.getMemorySpace());
    MemRefType newMemRefType = cast<MemRefType>(state.manager.updateSymbolicShape(baseNewType, target.symShape));
    if (oldType == newMemRefType) return;
    ModuleOp module = globalOp->getParentOfType<ModuleOp>();
    StringRef symName = globalOp.getSymName();
    SmallVector<memref::GetGlobalOp> getGlobalOpsToUpdate;
    module.walk([&](memref::GetGlobalOp getGlobal) {
      if (getGlobal.getNameAttr().getLeafReference() == symName) getGlobalOpsToUpdate.push_back(getGlobal);
    });
    Attribute initValue = globalOp.getConstantInitValue();
    if (auto denseAttr = dyn_cast_or_null<DenseElementsAttr>(initValue)) {
      Type newTensorType = memref::getTensorTypeFromMemRefType(newMemRefType);
      initValue = denseAttr.reshape(cast<ShapedType>(newTensorType));
    } else if (globalOp.getConstant()) {
      return;
    }
    SymbolTable symbolTable(module);
    Location loc = globalOp.getLoc();
    rewriter.setInsertionPoint(globalOp);
    StringAttr visibility = globalOp.getSymVisibilityAttr();
    if (!visibility) visibility = rewriter.getStringAttr("private");
    bool isConstant = static_cast<bool>(globalOp.getConstant());
    memref::GlobalOp newOp = rewriter.create<memref::GlobalOp>(loc, globalOp.getSymName(), visibility, newMemRefType,
                                                               initValue, isConstant, globalOp.getAlignmentAttr());
    symbolTable.erase(globalOp);
    (void)symbolTable.insert(newOp);
    newOp->moveBefore(&module.front());
    for (memref::GetGlobalOp oldGetGlobal : getGlobalOpsToUpdate) {
      Location getGlobalLoc = oldGetGlobal.getLoc();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(oldGetGlobal);
      auto newGetGlobal = rewriter.create<memref::GetGlobalOp>(getGlobalLoc, newMemRefType, oldGetGlobal.getNameAttr());
      oldGetGlobal.replaceAllUsesWith(newGetGlobal.getResult());
      oldGetGlobal.erase();
    }
  }
};

struct ExpandShapeAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::ExpandShapeOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto expandOp = cast<memref::ExpandShapeOp>(op);
    Value src = expandOp.getSrc();
    Value result = expandOp.getResult();
    SmallVector<std::string> srcSymShape = state.getValueSymbolicShape(src);
    auto reassociationVec = expandOp.getReassociationIndices();
    SmallVector<ArrayRef<int64_t>> reassociation;
    llvm::copy(reassociationVec, std::back_inserter(reassociation));

    int64_t outputRank = expandOp.getResultType().getRank();
    SmallVector<std::string> resultAxes(outputRank);
    for (size_t reassocIdx = 0; reassocIdx < reassociation.size(); ++reassocIdx) {
      const auto &indices = reassociation[reassocIdx];
      std::string srcAxis = srcSymShape[reassocIdx];
      if (indices.size() == 1) {
        int64_t outDim = indices[0];
        if (outDim < outputRank) resultAxes[outDim] = srcAxis;
      } else {
        SmallVector<std::string> childAxes;
        for (int64_t outDim : indices) {
          int64_t dimSize = expandOp.getResultType().getDimSize(outDim);
          std::string newAxis = state.createNewSymbolicDim(dimSize);
          resultAxes[outDim] = newAxis;
          childAxes.push_back(newAxis);
        }
        SmallVector<std::string> childAxesWithoutOne;
        std::copy_if(childAxes.begin(), childAxes.end(), std::back_inserter(childAxesWithoutOne),
                     [](const std::string &axis) { return axis != "1"; });
        if (childAxesWithoutOne.size() == 1) {
          auto it = std::find(childAxes.begin(), childAxes.end(), childAxesWithoutOne[0]);
          if (it != childAxes.end()) resultAxes[indices[std::distance(childAxes.begin(), it)]] = srcAxis;
        }
        state.updateDecomposition(srcAxis, childAxes);
      }
    }
    size_t dynIdx = 0;
    for (size_t outDim = 0; outDim < resultAxes.size(); ++outDim) {
      if (resultAxes[outDim].empty())
        resultAxes[outDim] = state.createNewSymbolicDim(expandOp.getResultType().getDimSize(outDim));
      if (expandOp.getResultType().getDimSize(outDim) == ShapedType::kDynamic &&
          state.DynSymtoArgDim.find(resultAxes[outDim]) == state.DynSymtoArgDim.end()) {
        Value dynValue = expandOp.getOutputShape()[dynIdx++];
        if (auto dimOp = dyn_cast<memref::DimOp>(dynValue.getDefiningOp())) {
          auto cstOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
          assert(cstOp && "dim index must be constant");
          int64_t idxVal = cast<IntegerAttr>(cstOp.getValue()).getValue().getSExtValue();
          state.DynSymtoArgDim[resultAxes[outDim]] = std::make_pair(dimOp.getSource(), idxVal);
        } else {
          state.DynSymtoArgDim[resultAxes[outDim]] = std::make_pair(dynValue, static_cast<int64_t>(-1));
        }
      }
    }
    state.updateValueSymbolicShape(result, resultAxes);
    state.ValuesinFunc.insert(result);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto expandOp = cast<memref::ExpandShapeOp>(op);
    Location loc = op->getLoc();
    Value oldSrc = expandOp.getSrc();
    Value oldResult = expandOp.getResult();
    Value newSrc = state.ssaMap.lookup(oldSrc) ? state.ssaMap.lookup(oldSrc) : oldSrc;
    SmallVector<std::string> newSrcSymShape = state.getValueSymbolicShape(newSrc);
    SmallVector<std::string> targetResultSymShape = state.computeTargetSymShape(oldResult);
    SmallVector<int64_t> targetShapeInt = state.getAxesSizes(targetResultSymShape);
    auto resultShape = dyn_cast<MemRefType>(oldResult.getType());
    if (!resultShape) return;
    Type targetShape = MemRefType::get(targetShapeInt, resultShape.getElementType());
    targetShape = state.manager.updateSymbolicShape(targetShape, targetResultSymShape);
    if (targetResultSymShape == newSrcSymShape) {
      state.ssaMap[oldResult] = newSrc;
      oldResult.replaceAllUsesWith(newSrc);
      state.toDeleteOps.insert(op);
      return;
    }
    auto reshapeOps = state.reshapeValueToSymShape(newSrc, targetResultSymShape, rewriter, loc);
    if (!reshapeOps.empty()) {
      Value result = reshapeOps.back()->getResult(0);
      state.ssaMap[oldResult] = result;
      oldResult.replaceAllUsesWith(result);
      state.toDeleteOps.insert(op);
    }
  }
};

struct CollapseShapeAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::CollapseShapeOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto collapseOp = cast<memref::CollapseShapeOp>(op);
    Value src = collapseOp.getSrc();
    Value result = collapseOp.getResult();
    SmallVector<std::string> srcSymShape = state.getValueSymbolicShape(src);
    auto reassociationVec = collapseOp.getReassociationIndices();
    SmallVector<ArrayRef<int64_t>> reassociation;
    llvm::copy(reassociationVec, std::back_inserter(reassociation));
    SmallVector<std::string> resultSymShape;
    for (size_t reassocIdx = 0; reassocIdx < reassociation.size(); ++reassocIdx) {
      const auto &indices = reassociation[reassocIdx];
      SmallVector<std::string> groupAxes;
      std::transform(indices.begin(), indices.end(), std::back_inserter(groupAxes),
                     [&srcSymShape](int64_t inputDim) { return srcSymShape[inputDim]; });
      if (groupAxes.size() == 1) {
        resultSymShape.push_back(groupAxes[0]);
      } else {
        std::string newAxis = state.getOrCreateSymDimNameAndSize(groupAxes);
        resultSymShape.push_back(newAxis);
        state.updateDecomposition(newAxis, groupAxes);
      }
    }
    state.updateValueSymbolicShape(result, resultSymShape);
    state.ValuesinFunc.insert(result);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto collapseOp = cast<memref::CollapseShapeOp>(op);
    Location loc = op->getLoc();
    Value oldSrc = collapseOp.getSrc();
    Value oldResult = collapseOp.getResult();
    Value newSrc = state.ssaMap.lookup(oldSrc) ? state.ssaMap.lookup(oldSrc) : oldSrc;
    SmallVector<std::string> newSrcSymShape = state.getValueSymbolicShape(newSrc);
    SmallVector<std::string> targetResultSymShape = state.computeTargetSymShape(oldResult);
    if (targetResultSymShape.empty()) {
      if (state.ssaMap.lookup(oldSrc)) op->setOperand(0, state.ssaMap.lookup(oldSrc));
      return;
    }
    if (targetResultSymShape == newSrcSymShape) {
      state.ssaMap[oldResult] = newSrc;
      oldResult.replaceAllUsesWith(newSrc);
      state.toDeleteOps.insert(op);
      return;
    }
    auto reshapeOps = state.reshapeValueToSymShape(newSrc, targetResultSymShape, rewriter, loc);
    if (!reshapeOps.empty()) {
      Value result = reshapeOps.back()->getResult(0);
      state.ssaMap[oldResult] = result;
      oldResult.replaceAllUsesWith(result);
      state.toDeleteOps.insert(op);
    }
  }
};

struct ReshapeAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::ReshapeOp>(op); }

 private:
  static ShapeDimValue getShapeDimValueFromValue(Value v, const ShapeNormalState &state) {
    if (auto cstOp = v.getDefiningOp<arith::ConstantOp>()) {
      return static_cast<int64_t>(cast<IntegerAttr>(cstOp.getValue()).getValue().getSExtValue());
    }
    if (auto dimOp = v.getDefiningOp<memref::DimOp>()) {
      auto idxCstOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
      assert(idxCstOp && "dim index must be constant");
      int64_t dimIdx = cast<IntegerAttr>(idxCstOp.getValue()).getValue().getSExtValue();
      return state.getValueSymbolicShape(dimOp.getSource())[dimIdx];
    }
    return v;
  }
  static SmallVector<ShapeDimValue> computeSrcShapeValues(const SmallVector<std::string> &srcSymShape,
                                                          const ShapeNormalState &state) {
    SmallVector<ShapeDimValue> srcShapeValues;
    for (size_t i = 0; i < srcSymShape.size(); ++i) {
      auto dynIt = state.DynSymtoArgDim.find(srcSymShape[i]);
      if (dynIt != state.DynSymtoArgDim.end()) {
        const auto &entry = dynIt->second;
        if (entry.second == -1) {
          if (auto cst = entry.first.getDefiningOp<arith::ConstantOp>()) {
            srcShapeValues.push_back(cast<IntegerAttr>(cst.getValue()).getValue().getSExtValue());
          } else {
            srcShapeValues.push_back(entry.first);
          }
        } else {
          srcShapeValues.push_back(srcSymShape[i]);
        }
      } else {
        auto axisIt = state.axisSizes.find(srcSymShape[i]);
        srcShapeValues.push_back(axisIt != state.axisSizes.end() ? axisIt->second : 0);
      }
    }
    return srcShapeValues;
  }
  static void matchDstShapeToSrcShape(const SmallVector<ShapeDimValue> &resultShapeValues,
                                      const SmallVector<ShapeDimValue> &srcShapeValues,
                                      const SmallVector<std::string> &srcSymShape,
                                      SmallVector<std::string> &resultSymShape) {
    size_t resIdx = 0;
    size_t srcIdx = 0;
    while (srcIdx < srcShapeValues.size() && resIdx < resultShapeValues.size()) {
      if (srcShapeValues[srcIdx] == resultShapeValues[resIdx]) {
        resultSymShape[resIdx] = srcSymShape[srcIdx];
        resIdx++;
        srcIdx++;
        continue;
      }
      size_t srcStart = srcIdx;
      int64_t tmpSrcProduct = 1;
      while (srcIdx < srcShapeValues.size() && std::get_if<int64_t>(&srcShapeValues[srcIdx])) {
        tmpSrcProduct *= std::get<int64_t>(srcShapeValues[srcIdx]);
        srcIdx++;
      }
      size_t resStart = resIdx;
      int64_t tmpResProduct = 1;
      while (resIdx < resultShapeValues.size() && std::get_if<int64_t>(&resultShapeValues[resIdx])) {
        tmpResProduct *= std::get<int64_t>(resultShapeValues[resIdx]);
        resIdx++;
      }
      // TODO(akg): support dynamic axis x/2 * 2
      if (tmpSrcProduct != tmpResProduct) break;
      if (srcIdx == srcStart && resIdx == resStart) break;
    }
  }
  static void fillResultSymShape(SmallVector<std::string> &resultSymShape, MemRefType resultType,
                                 const SmallVector<std::pair<int64_t, Value>> &sortedVals,
                                 ShapeNormalState &state) {
    for (size_t i = 0; i < resultSymShape.size(); ++i) {
      if (resultSymShape[i].empty()) {
        resultSymShape[i] = state.createNewSymbolicDim(resultType.getShape()[i]);
      }
      if (resultType.getShape()[i] != state.axisSizes[resultSymShape[i]]) {
        resultSymShape[i] = state.createNewSymbolicDim(resultType.getShape()[i]);
      }
      if (resultType.getShape()[i] == ShapedType::kDynamic &&
          state.DynSymtoArgDim.find(resultSymShape[i]) == state.DynSymtoArgDim.end()) {
        Value val = sortedVals[i].second;
        if (auto dimOp = val.getDefiningOp<memref::DimOp>()) {
          auto idxCstOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>();
          assert(idxCstOp && "dim index must be constant");
          int64_t dimIdx = cast<IntegerAttr>(idxCstOp.getValue()).getValue().getSExtValue();
          state.DynSymtoArgDim[resultSymShape[i]] = std::make_pair(dimOp.getSource(), dimIdx);
        } else if (val) {
          state.DynSymtoArgDim[resultSymShape[i]] = std::make_pair(val, static_cast<int64_t>(-1));
        }
      }
    }
  }

 public:
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto reshapeOp = cast<memref::ReshapeOp>(op);
    Value src = reshapeOp.getSource();
    Value shapeOperand = reshapeOp.getShape();
    Value result = reshapeOp.getResult();
    auto resultType = dyn_cast<MemRefType>(result.getType());
    if (!resultType) return;

    auto &shapeVals = state.shapeStoreMap[shapeOperand];
    SmallVector<std::pair<int64_t, Value>> sortedVals(shapeVals.begin(), shapeVals.end());
    llvm::sort(sortedVals, [](const auto &a, const auto &b) { return a.first < b.first; });
    SmallVector<ShapeDimValue> resultShapeValues;
    for (auto &[idx, val] : sortedVals) {
      (void)idx;
      resultShapeValues.push_back(getShapeDimValueFromValue(val, state));
    }

    SmallVector<std::string> srcSymShape = state.getValueSymbolicShape(src);
    SmallVector<ShapeDimValue> srcShapeValues = computeSrcShapeValues(srcSymShape, state);
    SmallVector<std::string> resultSymShape(resultShapeValues.size());
    matchDstShapeToSrcShape(resultShapeValues, srcShapeValues, srcSymShape, resultSymShape);
    // TODO(akg): consider match again from the end

    fillResultSymShape(resultSymShape, resultType, sortedVals, state);
    state.updateValueSymbolicShape(result, resultSymShape);
    state.ValuesinFunc.insert(result);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {}
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto reshapeOp = cast<memref::ReshapeOp>(op);
    Location loc = op->getLoc();
    Value oldSrc = reshapeOp.getSource();
    Value oldResult = reshapeOp.getResult();
    Value newSrc = state.ssaMap.lookup(oldSrc) ? state.ssaMap.lookup(oldSrc) : oldSrc;
    SmallVector<std::string> newSrcSymShape = state.getValueSymbolicShape(newSrc);
    SmallVector<std::string> targetResultSymShape = state.computeTargetSymShape(oldResult);
    if (targetResultSymShape.empty()) {
      if (state.ssaMap.lookup(oldSrc)) op->setOperand(0, state.ssaMap.lookup(oldSrc));
      return;
    }
    if (targetResultSymShape == newSrcSymShape) {
      state.ssaMap[oldResult] = newSrc;
      oldResult.replaceAllUsesWith(newSrc);
      state.toDeleteOps.insert(op);
      return;
    }
    auto reshapeOps = state.reshapeValueToSymShape(newSrc, targetResultSymShape, rewriter, loc);
    if (!reshapeOps.empty()) {
      Value result = reshapeOps.back()->getResult(0);
      state.ssaMap[oldResult] = result;
      oldResult.replaceAllUsesWith(result);
      state.toDeleteOps.insert(op);
    }
  }
};

struct GenericAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<linalg::GenericOp>(op); }
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto generic = cast<linalg::GenericOp>(op);
    auto indexingMaps = generic.getIndexingMapsArray();
    auto iteratorTypes = generic.getIteratorTypesArray();
    unsigned numInputs = generic.getNumDpsInputs();
    unsigned numOutputs = generic.getNumDpsInits();
    int64_t NumLoops = iteratorTypes.size();
    SmallVector<std::string> unifiedAxesNames(NumLoops);
    for (int64_t inputIndex = 0; inputIndex < numInputs + numOutputs; ++inputIndex) {
      SmallVector<std::string> SymShape;
      if (inputIndex < numInputs)
        SymShape = state.getValueSymbolicShape(generic.getDpsInputs()[inputIndex]);
      else
        SymShape = state.getValueSymbolicShape(generic.getDpsInits()[inputIndex - numInputs]);
      AffineMap inputIndexingMap = indexingMaps[inputIndex];
      SmallVector<int64_t> inputDimMap;
      for (unsigned i = 0; i < inputIndexingMap.getNumResults(); ++i) {
        if (inputIndexingMap.getResult(i) == 0) {
          inputDimMap.push_back(-1);
          continue;
        }
        auto de = dyn_cast<AffineDimExpr>(inputIndexingMap.getResult(i));
        inputDimMap.push_back(de.getPosition());
      }
      for (size_t i = 0; i < SymShape.size(); ++i) {
        if (inputDimMap[i] == -1) continue;
        if (unifiedAxesNames[inputDimMap[i]].empty()) {
          unifiedAxesNames[inputDimMap[i]] = SymShape[i];
        } else {
          unifiedAxesNames[inputDimMap[i]] = state.globalReplaceAxis(SymShape[i], unifiedAxesNames[inputDimMap[i]]);
        }
      }
    }
    for (unsigned outputIndex = 0; outputIndex < numOutputs; ++outputIndex) {
      Value output = generic.getDpsInits()[outputIndex];
      Value result =
        generic.getNumResults() > outputIndex ? generic.getResults()[outputIndex] : generic.getDpsInits()[outputIndex];
      state.updateValueSymbolicShape(result, state.getValueSymbolicShape(output));
      state.ValuesinFunc.insert(result);
    }
    for (size_t i = 0; i < unifiedAxesNames.size(); ++i) {
      if (unifiedAxesNames[i].empty()) unifiedAxesNames[i] = "1";
    }
    state.affineMapSymDims[op] = unifiedAxesNames;
  }

  static bool getOldBreakpoint(size_t pos1, size_t pos2, bool forward, size_t breakpointIdx,
                               const SmallVector<int64_t> &preLabels, SmallVector<bool> &breakpoints) {
    size_t neighborPos = forward ? pos1 + 1 : pos1 - 1;
    bool needBreak = (pos2 != neighborPos && preLabels[pos1] == preLabels[neighborPos]);
    if (needBreak) breakpoints[breakpointIdx] = true;
    return needBreak;
  }

  static bool updateBreakpoints(const SmallVector<std::string> &symShape,
                                const SmallVector<std::string> &unifiedAxesNames, const SmallVector<int64_t> &preLabels,
                                SmallVector<bool> &breakpoints) {
    bool needAssign = false;
    for (size_t i = 0; i < symShape.size() - 1; ++i) {
      size_t pos1 = std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape[i]) - unifiedAxesNames.begin();
      size_t pos2 =
        std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape[i + 1]) - unifiedAxesNames.begin();
      assert(pos1 < unifiedAxesNames.size() && pos2 < unifiedAxesNames.size() && "unsupported generic form");
      if (pos1 < unifiedAxesNames.size() - 1) {
        needAssign |= getOldBreakpoint(pos1, pos2, true, pos1, preLabels, breakpoints);
      }
    }
    for (size_t i = symShape.size() - 1; i > 0; --i) {
      size_t pos1 = std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape[i]) - unifiedAxesNames.begin();
      size_t pos2 =
        std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape[i - 1]) - unifiedAxesNames.begin();
      assert(pos1 < unifiedAxesNames.size() && pos2 < unifiedAxesNames.size() && "unsupported generic form");
      if (pos1 > 0) {
        needAssign |= getOldBreakpoint(pos1, pos2, false, pos1 - 1, preLabels, breakpoints);
      }
    }
    size_t firstPos =
      std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape[0]) - unifiedAxesNames.begin();
    if (firstPos != 0 && !breakpoints[firstPos - 1]) {
      breakpoints[firstPos - 1] = true;
      needAssign = true;
    }
    size_t lastPos =
      std::find(unifiedAxesNames.begin(), unifiedAxesNames.end(), symShape.back()) - unifiedAxesNames.begin();
    if (lastPos != unifiedAxesNames.size() - 1 && !breakpoints[lastPos]) {
      breakpoints[lastPos] = true;
      needAssign = true;
    }
    return needAssign;
  }

  static int findAxisIndex(const SmallVector<std::string> &axes, const std::string &a) {
    auto it = std::find(axes.begin(), axes.end(), a);
    return it != axes.end() ? (int)(it - axes.begin()) : -1;
  }

  void assignLabels(Operation *op, ShapeNormalState &state) const override {
    auto generic = cast<linalg::GenericOp>(op);
    unsigned numInputs = generic.getNumDpsInputs();
    unsigned numOutputs = generic.getNumDpsInits();
    SmallVector<std::string> unifiedAxesNames = state.expandGroupedAxes(state.affineMapSymDims[op]);
    if (unifiedAxesNames.empty()) return;
    SmallVector<int64_t> preLabels;
    llvm::transform(unifiedAxesNames, std::back_inserter(preLabels),
                    [&state](const std::string &a) { return state.manager.getLabel(a); });
    SmallVector<bool> breakpoints(unifiedAxesNames.size() - 1, false);
    for (size_t i = 0; i < breakpoints.size(); ++i) breakpoints[i] = (preLabels[i] != preLabels[i + 1]);
    bool needAssignLabel = false;
    for (unsigned ind = 0; ind < numInputs + numOutputs; ++ind) {
      SmallVector<std::string> symShape;
      if (ind < numInputs) {
        symShape = state.expandGroupedAxes(state.getValueSymbolicShape(generic.getDpsInputs()[ind]));
      } else {
        Value outputVal = (generic.getNumResults() > ind - numInputs) ? generic.getResults()[ind - numInputs]
                                                                      : generic.getDpsInits()[ind - numInputs];
        symShape = state.expandGroupedAxes(state.getValueSymbolicShape(outputVal));
      }
      if (symShape.empty()) continue;
      needAssignLabel |= updateBreakpoints(symShape, unifiedAxesNames, preLabels, breakpoints);
    }
    if (!needAssignLabel) return;
    int64_t currentLabel = state.manager.getLabel(unifiedAxesNames[0]);
    for (size_t i = 0; i < breakpoints.size(); ++i) {
      if (breakpoints[i]) {
        currentLabel = state.manager.assignLabel(unifiedAxesNames[i + 1]);
      } else {
        state.manager.assignLabel(unifiedAxesNames[i + 1], currentLabel);
      }
    }
  }

  static SmallVector<int64_t> getDuplicateAxisTargetIndicesInAffineMap(
    AffineMap oldMap, const SmallVector<std::pair<int64_t, int64_t>> &duplicateAxisMapping) {
    SmallVector<int64_t> indices;
    for (unsigned i = 0; i < oldMap.getNumResults(); ++i) {
      if (oldMap.getResult(i) == 0) continue;
      int64_t oldInd = (int64_t)dyn_cast<AffineDimExpr>(oldMap.getResult(i)).getPosition();
      for (const auto &[oldIdx, targetIdx] : duplicateAxisMapping) {
        if (oldIdx == oldInd) indices.push_back(targetIdx);
      }
    }
    return indices;
  }

  static SmallVector<AffineMap> buildGenericNewIndexingMaps(
    GenericOp generic, ArrayRef<Value> newInputs, ArrayRef<Value> newInitOperands,
    const SmallVector<std::string> &newUnifiedAxesNames,
    const SmallVector<std::pair<int64_t, int64_t>> &duplicateAxisMapping, const ShapeNormalState &state,
    PatternRewriter &rewriter) {
    llvm::StringMap<int> axisNameCount;
    for (const std::string &axis : newUnifiedAxesNames) axisNameCount[axis]++;
    SmallVector<AffineMap> newIndexingMaps;
    unsigned numInputs = generic.getNumDpsInputs();
    int64_t newNumLoops = newUnifiedAxesNames.size();
    auto oldIndexingMaps = generic.getIndexingMapsArray();
    for (int64_t ind = 0; ind < numInputs + generic.getNumDpsInits(); ++ind) {
      SmallVector<std::string> newSymShape = (ind < numInputs)
                                               ? state.getValueSymbolicShape(newInputs[ind])
                                               : state.getValueSymbolicShape(newInitOperands[ind - numInputs]);
      AffineMap oldMap = oldIndexingMaps[ind];
      SmallVector<int64_t> dupIndices = getDuplicateAxisTargetIndicesInAffineMap(oldMap, duplicateAxisMapping);
      SmallVector<AffineExpr> exprsNew(newSymShape.size());
      int64_t dupInd = 0;
      for (size_t k = 0; k < newSymShape.size(); ++k) {
        int64_t pos = (newSymShape[k] == "1") ? 0
                                              : (axisNameCount.lookup(newSymShape[k]) > 1
                                                   ? dupIndices[dupInd++]
                                                   : findAxisIndex(newUnifiedAxesNames, newSymShape[k]));
        exprsNew[k] = rewriter.getAffineDimExpr((unsigned)pos);
      }
      newIndexingMaps.push_back(AffineMap::get(newNumLoops, 0, exprsNew, rewriter.getContext()));
    }
    return newIndexingMaps;
  }

  static SmallVector<utils::IteratorType> buildGenericNewIteratorTypes(
    GenericOp generic, const SmallVector<std::string> &newUnifiedAxesNames,
    const SmallVector<std::string> &affineMapSymDims, const ShapeNormalState &state) {
    SmallVector<int64_t> productionOfUnifiedAxesNames;
    int64_t production = 1, lastValidProduction = -1;
    for (const std::string &axis : affineMapSymDims) {
      if (state.axisSizes.lookup(axis) == ShapedType::kDynamic) {
        productionOfUnifiedAxesNames.push_back(lastValidProduction);
      } else {
        production *= state.axisSizes.lookup(axis);
        productionOfUnifiedAxesNames.push_back(production);
        lastValidProduction = production;
      }
    }
    SmallVector<utils::IteratorType> newIteratorTypes(newUnifiedAxesNames.size(), utils::IteratorType::parallel);
    int64_t accumProduction = 1;
    for (int64_t k = 0; k < static_cast<int64_t>(newUnifiedAxesNames.size()); ++k) {
      std::string axis = newUnifiedAxesNames[k];
      int64_t pos;
      if (state.axisSizes.lookup(axis) == ShapedType::kDynamic) {
        pos = findAxisIndex(affineMapSymDims, axis);
      } else {
        accumProduction *= state.axisSizes.lookup(axis);
        pos =
          std::lower_bound(productionOfUnifiedAxesNames.begin(), productionOfUnifiedAxesNames.end(), accumProduction) -
          productionOfUnifiedAxesNames.begin();
      }
      assert(pos != -1 && "dynamic axis not found");
      newIteratorTypes[k] = generic.getIteratorTypesArray()[pos];
    }
    return newIteratorTypes;
  }

  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    auto generic = cast<linalg::GenericOp>(op);
    Location loc = op->getLoc();
    rewriter.setInsertionPoint(op);
    SmallVector<Value> newInputs;
    llvm::transform(generic.getDpsInputs(), std::back_inserter(newInputs),
                    [&state](Value v) { return state.ssaMap.lookup(v) ? state.ssaMap.lookup(v) : v; });
    SmallVector<Value> newInitOperands;
    llvm::transform(generic.getDpsInits(), std::back_inserter(newInitOperands),
                    [&state](Value v) { return state.ssaMap.lookup(v) ? state.ssaMap.lookup(v) : v; });
    auto [newUnifiedAxesNames, duplicateAxisMapping] =
      state.computeTargetSymShapeWithMapping(state.affineMapSymDims[op]);
    SmallVector<AffineMap> newIndexingMaps = buildGenericNewIndexingMaps(
      generic, newInputs, newInitOperands, newUnifiedAxesNames, duplicateAxisMapping, state, rewriter);
    SmallVector<utils::IteratorType> newIteratorTypes =
      buildGenericNewIteratorTypes(generic, newUnifiedAxesNames, state.affineMapSymDims[op], state);
    auto newGeneric = rewriter.create<GenericOp>(loc, SmallVector<Type>(), newInputs, newInitOperands, newIndexingMaps,
                                                 newIteratorTypes, nullptr);
    rewriter.cloneRegionBefore(generic.getRegion(), newGeneric.getRegion(), newGeneric.getRegion().begin());
    for (unsigned i = 0; i < generic.getNumDpsInits(); ++i) state.ssaMap[generic.getDpsInits()[i]] = newInitOperands[i];
    state.toDeleteOps.insert(op);
  }
};

struct SubviewAdapter final : OpAdapter {
  bool match(Operation *op) const override { return isa<memref::SubViewOp>(op); }

 private:
  static std::string tryGetSymdimFromDynamicSize(OpFoldResult mixedSize, const std::string &srcSymShapeI,
                                                 Value result, size_t i, ShapeNormalState &state) {
    auto getConstInt = [](Value v) -> std::optional<int64_t> {
      if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
        if (auto a = dyn_cast<IntegerAttr>(c.getValue())) return a.getValue().getSExtValue();
      }
      return std::nullopt;
    };
    if (!mixedSize.is<Value>()) return "";
    Value sizeVal = mixedSize.get<Value>();
    auto it = state.DynSymtoArgDim.find(srcSymShapeI);
    if (it == state.DynSymtoArgDim.end()) return "";
    auto [srcArg, srcIdx] = it->second;
    if (auto dimOp = sizeVal.getDefiningOp<memref::DimOp>()) {
      if (srcIdx < 0) return "";
      std::optional<int64_t> dimIdxOpt;
      if (auto cstOp = dimOp.getIndex().getDefiningOp<arith::ConstantOp>()) {
        if (auto a = dyn_cast<IntegerAttr>(cstOp.getValue())) dimIdxOpt = a.getValue().getSExtValue();
      } else if (auto idxOp = dimOp.getIndex().getDefiningOp<arith::ConstantIndexOp>()) {
        dimIdxOpt = cast<IntegerAttr>(idxOp.getValue()).getValue().getSExtValue();
      }
      if (!dimIdxOpt) return "";
      int64_t dimIdx = *dimIdxOpt;
      SmallVector<std::string> sourceSymShape = state.getValueSymbolicShape(srcArg);
      SmallVector<std::string> sizeSymShape = state.getValueSymbolicShape(dimOp.getSource());
      if (static_cast<size_t>(dimIdx) < sizeSymShape.size() && static_cast<size_t>(srcIdx) < sourceSymShape.size() &&
          sizeSymShape[dimIdx] == sourceSymShape[srcIdx])
        return srcSymShapeI;
      return "";
    }
    if (srcIdx == -1) {
      auto sizeConst = getConstInt(sizeVal);
      auto srcConst = getConstInt(srcArg);
      if (sizeConst && srcConst && *sizeConst == *srcConst) return srcSymShapeI;
    }
    return "";
  }

 public:
  void unifyToFinestAxes(Operation *op, ShapeNormalState &state) const override {
    auto subviewOp = cast<memref::SubViewOp>(op);
    Value src = subviewOp.getSource();
    Value result = subviewOp.getResult();
    ArrayRef<int64_t> offsets = subviewOp.getStaticOffsets();
    ArrayRef<int64_t> sizes = subviewOp.getStaticSizes();
    ArrayRef<int64_t> strides = subviewOp.getStaticStrides();
    SmallVector<std::string> srcSymShape = state.getValueSymbolicShape(src);
    SmallVector<std::string> resultSymShape;
    SmallVector<OpFoldResult> mixedSizes = subviewOp.getMixedSizes();
    for (size_t i = 0; i < srcSymShape.size(); ++i) {
      if (ShapedType::isDynamic(sizes[i])) {
        std::string symdimToUse = tryGetSymdimFromDynamicSize(mixedSizes[i], srcSymShape[i], result, i, state);
        if (symdimToUse.empty()) {
          symdimToUse = state.createNewSymbolicDim(ShapedType::kDynamic);
          state.DynSymtoArgDim[symdimToUse] = std::make_pair(result, static_cast<int64_t>(i));
        }
        resultSymShape.push_back(symdimToUse);
        state.UpdatedAxes[op].push_back(i);
        continue;
      }
      if (offsets[i] == 0 && strides[i] == 1 && sizes[i] == state.axisSizes[srcSymShape[i]]) {
        resultSymShape.push_back(srcSymShape[i]);
      } else {
        assert(sizes[i] > 0 && "sizes[i] must be greater than 0 in memref.subview op");
        resultSymShape.push_back(state.createNewSymbolicDim(sizes[i]));
        state.UpdatedAxes[op].push_back(i);
      }
    }
    state.updateValueSymbolicShape(result, resultSymShape);
    state.ValuesinFunc.insert(result);
  }
  void assignLabels(Operation *op, ShapeNormalState &state) const override {
    auto subviewOp = cast<memref::SubViewOp>(op);
    Value src = subviewOp.getSource();
    Value result = subviewOp.getResult();
    SmallVector<int64_t> updatedAxes = state.UpdatedAxes[op];
    auto processLabels = [&](std::string symDim) {
      SmallVector<std::string> flatSym = state.expandAxisRecursively(symDim);
      if (flatSym.size() < 1) return;
      if (flatSym.size() == 1) {
        state.manager.assignLabel(flatSym[0]);
        return;
      }
      for (size_t i = 1; i < flatSym.size(); ++i) {
        if (state.manager.getLabel(flatSym[i]) != state.manager.getLabel(flatSym[i - 1])) {
          state.isSupported = false;
          // disconnected expanded symshape of slice axis is not supported
          return;
        }
      }
      int64_t newLabel = state.manager.assignLabel(flatSym[0]);
      for (size_t i = 1; i < flatSym.size(); ++i) state.manager.assignLabel(flatSym[i], newLabel);
    };
    for (int64_t SliceInd : updatedAxes) {
      processLabels(state.getValueSymbolicShape(result)[SliceInd]);
      processLabels(state.getValueSymbolicShape(src)[SliceInd]);
    }
  }
  void rewrite(Operation *op, ShapeNormalState &state, PatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto subviewOp = cast<memref::SubViewOp>(op);
    rewriter.setInsertionPoint(op);
    Value src = subviewOp.getSource();
    Value oldResult = subviewOp.getResult();
    Value newSrc = state.ssaMap.lookup(src) ? state.ssaMap.lookup(src) : src;
    SmallVector<std::string> oldResultSymShape = state.getValueSymbolicShape(oldResult);
    SmallVector<int64_t> updatedAxes = state.UpdatedAxes[op];
    SmallVector<OpFoldResult> oldOffsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> oldSizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> oldStrides = subviewOp.getMixedStrides();
    SmallVector<std::string> targetResultSymShape;
    SmallVector<std::string> targetForSubView;
    SmallVector<std::string> targetTemp;
    for (size_t i = 0; i < oldResultSymShape.size(); ++i) {
      bool slicedTo1 = llvm::is_contained(updatedAxes, static_cast<int64_t>(i)) && oldResultSymShape[i] == "1";
      if (slicedTo1) {
        targetForSubView.append(state.computeTargetSymShape(targetTemp));
        targetResultSymShape.append(state.computeTargetSymShape(targetTemp));
        targetTemp.clear();
        targetForSubView.push_back("1");
      } else {
        targetTemp.push_back(oldResultSymShape[i]);
      }
    }
    targetForSubView.append(state.computeTargetSymShape(targetTemp));
    targetResultSymShape.append(state.computeTargetSymShape(targetTemp));
    SmallVector<int64_t> targetForSubViewShapeInt = state.getAxesSizes(targetForSubView);
    auto srcMemRefTy = cast<MemRefType>(newSrc.getType());
    SmallVector<OpFoldResult> offsets(targetForSubView.size());
    SmallVector<OpFoldResult> sizes(targetForSubView.size());
    SmallVector<OpFoldResult> strides(targetForSubView.size());
    SmallVector<bool> offsetFilled(targetForSubView.size(), false);
    SmallVector<int64_t> OneDimIndsInTarget;
    for (size_t i = 0; i < targetForSubView.size(); ++i) {
      if (targetForSubView[i] == "1") OneDimIndsInTarget.push_back(i);
    }
    size_t OneDimInd = 0;
    DenseSet<int64_t> DynDimIndsInTarget;
    for (int64_t i : updatedAxes) {
      size_t pos;
      if (oldResultSymShape[i] == "1") {
        pos = OneDimIndsInTarget[OneDimInd];
        OneDimInd++;
      } else {
        pos =
          std::find(targetForSubView.begin(), targetForSubView.end(), oldResultSymShape[i]) - targetForSubView.begin();
      }
      assert(pos != targetForSubView.size() && "sliced axis should not be expanded in newsrc and targetForSubView");
      offsets[pos] = oldOffsets[i];
      sizes[pos] = oldSizes[i];
      strides[pos] = oldStrides[i];
      offsetFilled[pos] = true;
      if (oldSizes[i].is<Value>()) DynDimIndsInTarget.insert(pos);
    }
    for (size_t i = 0; i < targetForSubView.size(); ++i) {
      if (!offsetFilled[i]) {
        offsets[i] = rewriter.getIndexAttr(0);
        sizes[i] = rewriter.getIndexAttr(targetForSubViewShapeInt[i]);
        strides[i] = rewriter.getIndexAttr(1);
      }
    }
    Type inferredResultType = memref::SubViewOp::inferResultType(srcMemRefTy, offsets, sizes, strides);
    SmallVector<std::string> newSrcSymShape = state.getValueSymbolicShape(newSrc);
    Type subviewOutputType = state.manager.updateSymbolicShape(inferredResultType, newSrcSymShape);
    Value subviewResult =
      rewriter.create<memref::SubViewOp>(loc, cast<MemRefType>(subviewOutputType), newSrc, offsets, sizes, strides);
    Value newResult = state.CastToTargetSymShape(subviewResult, targetForSubView, rewriter, loc);
    for (int64_t dynInd : DynDimIndsInTarget) {
      state.DynSymtoArgDim[targetResultSymShape[dynInd]] = std::make_pair(newResult, dynInd);
    }
    if (targetForSubView != targetResultSymShape) {
      auto reshapeOps = state.reshapeValueToSymShape(newResult, targetResultSymShape, rewriter, loc);
      if (!reshapeOps.empty()) newResult = reshapeOps.back()->getResult(0);
    }
    state.ssaMap[oldResult] = newResult;
    oldResult.replaceAllUsesWith(newResult);
    state.toDeleteOps.insert(op);
  }
};

static const OpAdapterRegistry &getOpAdapterRegistry() {
  static OpAdapterRegistry R;
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    R.registerAdapter(std::make_unique<AllocAdapter>());
    R.registerAdapter(std::make_unique<ConstantAdapter>());
    R.registerAdapter(std::make_unique<ArithOpAdapter>());
    R.registerAdapter(std::make_unique<AffineApplyAdapter>());
    R.registerAdapter(std::make_unique<StoreOpAdapter>());
    R.registerAdapter(std::make_unique<DimOpAdapter>());
    R.registerAdapter(std::make_unique<GlobalAdapter>());
    R.registerAdapter(std::make_unique<ExpandShapeAdapter>());
    R.registerAdapter(std::make_unique<CollapseShapeAdapter>());
    R.registerAdapter(std::make_unique<ReshapeAdapter>());
    R.registerAdapter(std::make_unique<GenericAdapter>());
    R.registerAdapter(std::make_unique<SubviewAdapter>());
  }
  return R;
}

void initFinestAxes(func::FuncOp func, ShapeNormalState &state, PatternRewriter &rewriter) {
  for (BlockArgument arg : func.getArguments()) {
    state.ValuesinFunc.insert(arg);
    auto memrefTy = dyn_cast<MemRefType>(arg.getType());
    if (!memrefTy) continue;
    SmallVector<std::string> initialSymbolicShape;
    for (int64_t i = 0; i < memrefTy.getRank(); ++i) {
      int64_t dimSize = memrefTy.getDimSize(i);
      std::string axis = state.createNewSymbolicDim(dimSize);
      if (dimSize == ShapedType::kDynamic) state.DynSymtoArgDim[axis] = std::make_pair(arg, i);
      initialSymbolicShape.push_back(axis);
    }
    state.updateValueSymbolicShape(arg, initialSymbolicShape);
  }
}

void unifyToFinestAxes(ModuleOp module, ShapeNormalState &state) {
  module.walk([&](Operation *op) {
    if (!state.isSupportedOp(op)) return;
    state.unifyToFinestAxes(op);
  });

  module.walk([&](func::FuncOp func) {
    SmallVector<Type> newArgTypes;
    llvm::transform(func.getArguments(), std::back_inserter(newArgTypes),
                    [](BlockArgument arg) { return arg.getType(); });
    auto oldFuncType = func.getFunctionType();
    auto newFuncType = FunctionType::get(func.getContext(), newArgTypes, oldFuncType.getResults());
    func.setFunctionType(newFuncType);
  });
}

void collectGlobalTargetShapes(ModuleOp module, ShapeNormalState &state) {
  module.walk([&](memref::GetGlobalOp getGlobal) {
    Value result = getGlobal.getResult();
    if (!state.ValuesinFunc.contains(result)) return;
    SmallVector<std::string> targetSymShape = state.computeTargetSymShape(result);
    SmallVector<int64_t> targetShape = state.getAxesSizes(targetSymShape);
    StringRef name = getGlobal.getNameAttr().getLeafReference();
    state.globalTargets[name] = {targetSymShape, targetShape};
  });
}

void assignLabels(ModuleOp module, ShapeNormalState &state) {
  state.manager.clearLabels();

  for (auto &axisSize : state.axisSizes) {
    if (axisSize.second == ShapedType::kDynamic) {
      state.manager.assignLabel(axisSize.getKey());
    } else {
      state.manager.assignLabel(axisSize.getKey(), 0);
    }
  }

  module.walk([&](Operation *op) {
    if (!state.isSupportedOp(op)) return;
    state.assignLabels(op);
  });

  module.walk([&](func::ReturnOp returnOp) {
    SmallVector<SmallVector<std::string>> shapes;
    llvm::transform(returnOp.getOperands(), std::back_inserter(shapes),
                    [&state](Value v) { return state.getValueSymbolicShape(v); });
    state.returnOpShapes[returnOp] = std::move(shapes);
  });
}

void solveEntryTargetLayouts(func::FuncOp func, ShapeNormalState &state, PatternRewriter &rewriter) {
  for (int i = func.getArguments().size() - 1; i >= 0; --i) {
    BlockArgument arg = func.getArguments()[i];
    SmallVector<std::string> targetSymShape = state.computeTargetSymShape(arg);
    SmallVector<int64_t> targetShapeInt = state.getAxesSizes(targetSymShape);
    auto shapedTy = dyn_cast<ShapedType>(arg.getType());
    if (!shapedTy) continue;
    if (shapedTy.getShape() == ArrayRef<int64_t>(targetShapeInt)) {
      state.ssaMap[arg] = arg;
      continue;
    }
    Location loc = func.getLoc();
    rewriter.setInsertionPointToStart(&func.getBody().front());
    auto reshapeOps = state.reshapeValueToSymShape(arg, targetSymShape, rewriter, loc);
    if (!reshapeOps.empty()) {
      state.ssaMap[arg] = reshapeOps.back()->getResult(0);
      state.entryMaterializedOps.insert(reshapeOps.back());
    }
  }
}

void rewriteAllOps(ModuleOp module, ShapeNormalState &state, PatternRewriter &rewriter) {
  module.walk([&](Operation *op) {
    if (state.entryMaterializedOps.contains(op)) return;
    if (!state.isSupportedOp(op)) return;
    state.rewrite(op, rewriter);
  });
}

void materializeReturnReshapes(ModuleOp module, ShapeNormalState &state, PatternRewriter &rewriter) {
  module.walk([&](func::ReturnOp returnOp) {
    Location loc = returnOp.getLoc();
    SmallVector<Value> newOperands;
    func::FuncOp func = returnOp->getParentOfType<func::FuncOp>();

    auto it = state.returnOpShapes.find(returnOp);
    if (it == state.returnOpShapes.end()) return;
    const SmallVector<SmallVector<std::string>> &targetSymShapes = it->second;
    for (size_t i = 0; i < returnOp.getNumOperands(); ++i) {
      const SmallVector<std::string> &targetSymShape = targetSymShapes[i];

      Value oldRet = returnOp.getOperand(i);
      Value newRet = state.ssaMap.lookup(oldRet) ? state.ssaMap.lookup(oldRet) : oldRet;
      rewriter.setInsertionPoint(returnOp);
      auto reshapeOps = state.reshapeValueToSymShape(newRet, targetSymShape, rewriter, loc);
      if (!reshapeOps.empty()) {
        newOperands.push_back(reshapeOps.back()->getResult(0));
      } else {
        newOperands.push_back(newRet);
      }
    }

    bool hasChange = false;
    for (size_t i = 0; i < newOperands.size(); ++i) {
      if (newOperands[i] != returnOp.getOperand(i)) {
        hasChange = true;
        break;
      }
    }

    if (hasChange) {
      rewriter.setInsertionPoint(returnOp);
      rewriter.create<func::ReturnOp>(loc, newOperands);
      rewriter.eraseOp(returnOp);
    }

    SmallVector<Type> newResultTypes;
    std::transform(newOperands.begin(), newOperands.end(), std::back_inserter(newResultTypes),
                   [](Value ret) { return ret.getType(); });
    auto newFuncType = FunctionType::get(func.getContext(), func.getFunctionType().getInputs(), newResultTypes);
    func.setFunctionType(newFuncType);
  });
}

static bool analyzeShapeNormalization(ModuleOp module, ShapeNormalState &state, PatternRewriter &rewriter) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    for (Operation &op : func.getOps()) {
      if (!state.isSupportedOp(&op) && !isa<func::ReturnOp, memref::GetGlobalOp>(op)) return false;
    }
  }
  module.walk([&](func::FuncOp func) { initFinestAxes(func, state, rewriter); });
  unifyToFinestAxes(module, state);
  for (const auto &decomp : state.decompositions) {
    SmallVector<std::string> expandedAxes = state.expandGroupedAxes(decomp.second);
    state.decompositions[decomp.first()] = expandedAxes;
  }
  assignLabels(module, state);
  collectGlobalTargetShapes(module, state);

  if (!state.isSupported) {
    for (const auto &value : state.ValuesinFunc) {
      state.removeValueSymbolicShape(value);
    }
    module.walk([&](func::FuncOp func) {
      SmallVector<Type> newArgTypes;
      llvm::transform(func.getArguments(), std::back_inserter(newArgTypes),
                      [](BlockArgument arg) { return arg.getType(); });
      auto oldFuncType = func.getFunctionType();
      auto newFuncType = FunctionType::get(func.getContext(), newArgTypes, oldFuncType.getResults());
      func.setFunctionType(newFuncType);
    });
    return false;
  }
  return true;
}

static void rewriteShapeNormalization(ModuleOp module, ShapeNormalState &state, PatternRewriter &rewriter) {
  module.walk([&](func::FuncOp func) { solveEntryTargetLayouts(func, state, rewriter); });
  rewriteAllOps(module, state, rewriter);
  materializeReturnReshapes(module, state, rewriter);
  for (Operation *op : state.toDeleteOps) rewriter.eraseOp(op);

  auto isDeadStore = [](memref::StoreOp store) {
    Value memref = store.getMemRef();
    if (isa<BlockArgument>(memref)) return false;
    for (Operation *user : memref.getUsers())
      if (!isa<memref::StoreOp>(user)) return false;
    return true;
  };
  auto canErase = [&](Operation *op) {
    if (auto store = dyn_cast<memref::StoreOp>(op)) return isDeadStore(store);
    for (Value r : op->getResults())
      if (!r.use_empty()) return false;
    return true;
  };
  for (bool changed = true; changed;) {
    changed = false;
    SmallVector<Operation *> ops;
    module.walk([&](Operation *op) {
      if (isa<memref::StoreOp, memref::DimOp, memref::AllocOp, arith::ConstantOp>(op)) ops.push_back(op);
    });
    for (Operation *op : llvm::reverse(ops)) {
      if (canErase(op)) {
        rewriter.eraseOp(op);
        changed = true;
      }
    }
  }
}

struct ShapeNormalization : public impl::ShapeNormalizationBase<ShapeNormalization> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    PatternRewriter rewriter(&getContext());

    ShapeNormalState state;
    if (!analyzeShapeNormalization(module, state, rewriter)) return;

    rewriteShapeNormalization(module, state, rewriter);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createShapeNormalizationPass() {
  return std::make_unique<ShapeNormalization>();
}
}  // namespace mlir
