/**
 * Copyright 2023-2026 Huawei Technologies Co., Ltd
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

// ===- UnifyShape.cc - Remove expand/collapse shape ops -------------------=== //
// This file implements the transformation passes to remove expand/collapse
// shapes operations.
// ===----------------------------------------------------------------------=== //

#include "akg/Dialect/Affine/Transforms/UnifyShape.h"

#include <algorithm>
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "akg/Utils/Constants.h"
namespace mlir {
#ifndef GEN_PASS_DEF_UNIFYSHAPE
#define GEN_PASS_DEF_UNIFYSHAPE
#include "akg/Dialect/Affine/Passes.h.inc"

#endif
}  // namespace mlir
#define DEBUG_TYPE "unify-shape"

namespace mlir {

static constexpr const int kVectorSizeFour = 4;
static constexpr const int kVectorSizeEight = 8;

struct UnifyShapeInfos {
  bool skip = false;
  bool dynamicReshape = false;
  AffineMap newAccessMap;
  SmallVector<int64_t, kVectorSizeFour> deletedDim;
  // newShapeMap for generate new alloc Operation, associated with the dynamic dim if required
  llvm::MapVector<AffineMap, SmallVector<Value, kVectorSizeFour>> newShapeMap;
};

struct MapProcessContext {
  Operation *referenceOp;
  MemRefType referenceType;
  MLIRContext *context;
  unsigned nbSymbolExpr = 0;
  SmallVector<AffineExpr, kVectorSizeFour> newAccesExpr;
  UnifyShapeInfos result;
};

struct DimAccumState {
  AffineExpr accessRes;
  AffineExpr shapeRes;
  SmallVector<Value, kVectorSizeFour> symbolicValue;
};

class UnifyShape : public mlir::impl::UnifyShapeBase<UnifyShape> {
 public:
  UnifyShape() = default;
  UnifyShape(const UnifyShape &pass) = default;
  UnifyShape &operator=(const UnifyShape &) = default;
  UnifyShape(const bool allowNonPolyhedralAccess, const bool keepArg) {
    this->allowNonPolyhedralAccess = allowNonPolyhedralAccess;
    this->keepArgsShape = keepArg;
  }
  explicit UnifyShape(const UnifyShapeOptions &options) : UnifyShapeBase(options) {}

  void argsFindAndEraseCascadeShapeOps(Value arg, Operation *userOp) {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - work on operation:\n";
      userOp->dump();
    });

    // 1. Retrieve the new memRefType for arg
    MemRefType newmemrefType;
    SmallVector<AffineMap, kVectorSizeFour> reassociationMap;
    if (auto csop = dyn_cast<mlir::memref::CollapseShapeOp>(userOp)) {
      newmemrefType = cast<MemRefType>(csop.getType());
      reassociationMap = csop.getReassociationMaps();
    } else if (auto esop = dyn_cast<mlir::memref::ExpandShapeOp>(userOp)) {
      newmemrefType = cast<MemRefType>(esop.getType());
      reassociationMap = esop.getReassociationMaps();
    } else {
      llvm::errs() << DEBUG_TYPE << " - This case may never happen.\n"
                   << "Enter argsFindAndEraseCascadeShapeOps when op is not Collapse/Expand\n"
                   << userOp;
    }

    // 2. Update the Type of arg with memRefType
    arg.setType(newmemrefType);

    // 3. Update all the usage of the reshape value by arg
    for (Value resultValue : userOp->getResults()) {
      // need to study how to skip the modification of the shape if it impact a memref.dim
      for (Operation *uop : resultValue.getUsers()) {
        if (mlir::memref::DimOp dimOp = dyn_cast<mlir::memref::DimOp>(uop)) {
          SmallVector<int64_t, kVectorSizeFour> deletedDim = getDeletedDim(reassociationMap);
          updateDimOp(dimOp, deletedDim);
        }
      }
      resultValue.replaceAllUsesWith(arg);
    }

    // 4. Remove the reshape op
    userOp->erase();

    // 5. Recurse
    for (Operation *newUserOp : arg.getUsers()) {
      if (!isa<mlir::memref::CollapseShapeOp>(newUserOp) && !isa<mlir::memref::ExpandShapeOp>(newUserOp)) {
        continue;
      }

      argsFindAndEraseCascadeShapeOps(arg, newUserOp);
    }
  }

  void argsFindAndEraseExpandAndCollapseShapeOps(const Value &arg) {
    if (arg.hasOneUse()) {
      for (Operation *userOp : arg.getUsers()) {
        if (!isa<mlir::memref::CollapseShapeOp>(userOp) && !isa<mlir::memref::ExpandShapeOp>(userOp)) {
          continue;
        }

        // We seek to handle "cascade" expand/collapse, that is, for instance:
        // %collapse_shape = memref.collapse_shape %arg0 [[0, 1], [2]] : memref<X> into memref<Y>
        // %expand_shape = memref.expand_shape %collapse_shape [[0, 1], [2, 3]] : memref<Y> into memref<Z>
        // We could think of calling argsFindAndEraseExpandAndCollapseShapeOps recursively but this poses
        // some issues:
        // * At this point, userOp is NOT yet deleted (and we can't do it yet otherwise it will cause issues).
        // Therefore, it still appears in the list of uses and a recursive call of this function will
        // cause to fall into the else section. However, for now it is not yet clear
        // what other cases of multiple uses can occur, other than cases of "cascade". Therefore
        // it is not clear whether cascade must be handled in the else condition of this function.
        // For now, we therefore use a call a separate function that is similar but
        // that does not check whether there is only one use
        argsFindAndEraseCascadeShapeOps(arg, userOp);
      }
    } else {
      for (Operation *userOp : arg.getUsers()) {
        if (!isa<mlir::memref::CollapseShapeOp>(userOp) && !isa<mlir::memref::ExpandShapeOp>(userOp)) {
          continue;
        }

        // have to manage when arg are used more than 1 time in presence of
        // CollapseShapeOp or ExpandShapeOp
        LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << " Do not manage many usage of an argument that include a reshape, on "
                                << arg << "\n");
      }
    }
  }

  //  void eraseOpsUsingBlockArguments(mlir::func::FuncOp &fop) {
  void eraseOpsUsingBlockArguments(mlir::ModuleOp &m) {
    m.walk([this](mlir::func::FuncOp fop) {
      FunctionType functionType = fop.getFunctionType();
      SmallVector<Type, kVectorSizeEight> newArgTypes;
      SmallVector<Type, kVectorSizeFour> resultTypes;
      FunctionType newFuncType;
      resultTypes = llvm::to_vector<kSmallVectorSizeFour>(functionType.getResults());

      assert(resultTypes.empty() &&
             "Function result must be empty due to the call of "
             "-buffer-results-to-out-params pass");

      for (BlockArgument &bbArg : fop.getArguments()) {
        LLVM_DEBUG({
          llvm::dbgs() << "\n" << DEBUG_TYPE << " - work with BlockArgument:\n";
          bbArg.dump();
        });
        argsFindAndEraseExpandAndCollapseShapeOps(bbArg);
        newArgTypes.push_back(bbArg.getType());
      }

      newFuncType = FunctionType::get(&getContext(), newArgTypes, resultTypes);
      fop.setType(newFuncType);
    });
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - After eraseOpsUsingBlockArguments:\n";
      m.dump();
    });
  }

  /// updateDimOp, OK to update dimOp, if the dim that are fused are not dynamic
  ///   check if the fused dim are dynamic, return false in this case? Is it possible?
  void updateDimOp(mlir::memref::DimOp dimOp, const SmallVector<int64_t, kVectorSizeFour> &deletedDim) const {
    Value idx = dimOp.getIndex();

    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - updateDimOp work on:\n";
      dimOp.dump();
      idx.dump();
    });

    if (auto cop = dyn_cast<mlir::arith::ConstantOp>(idx.getDefiningOp())) {
      if (auto attr = dyn_cast<IntegerAttr>(cop.getValue())) {
        llvm::APInt indexValue = attr.getValue();
        llvm::APInt newIndex = indexValue;
        // 1. Compute new index to access
        for (uint64_t i : deletedDim) {
          // Note: differ from SimplifyShape on this test
          if (indexValue.uge(i)) {
            --newIndex;
          }
        }

        // 2. if new index to access is diff from original one, update it
        if (indexValue != newIndex) {
          Type type = attr.getType();
          mlir::OpBuilder builder(cop);
          auto newcstOp = builder.create<mlir::arith::ConstantOp>(cop.getLoc(), type, IntegerAttr::get(type, newIndex));
          llvm::dbgs() << DEBUG_TYPE
                       << " - BECAREFULL: memref.dim will be updated, may cause a wrong code? (if fused dim "
                          "are dynamic)\nFrom\n";
          dimOp.dump();
          dimOp->replaceUsesOfWith(idx, newcstOp.getResult());
          llvm::dbgs() << "To\n";
          dimOp.dump();
        }
      } else {
        llvm::errs() << DEBUG_TYPE << " - updateDimOp cannot update access for:\n"
                     << dimOp << "Unknown kind of attribute for" << cop << "May result to a wrong running code...\n";
      }
    } else {
      llvm::errs() << DEBUG_TYPE << " - updateDimOp cannot update access for:\n"
                   << dimOp << "Do not come from a mlir::arith::ConstantOp\n"
                   << "May result to a wrong running code...\n";
    }
    LLVM_DEBUG({
      llvm::dbgs() << " new DimOp:\n";
      dimOp.dump();
    });
  }

  AffineMap getUpdatedAffineMap(AffineMap &aff, const AffineMap &newMapAccess) const {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - getUpdatedAffineMap:\n";
      llvm::dbgs() << " Map to update:\n";
      aff.dump();
      llvm::dbgs() << " with Map to apply:\n";
      newMapAccess.dump();
    });
    aff = newMapAccess.compose(aff);
    LLVM_DEBUG({
      llvm::dbgs() << " new map:\n";
      aff.dump();
    });
    return aff;
  }

  [[nodiscard]] SmallVector<int64_t, kVectorSizeFour> getDeletedDim(
    const SmallVector<AffineMap, kVectorSizeFour> &reshapeMap) const {
    SmallVector<int64_t, kVectorSizeFour> deletedDim;
    for (auto map : reshapeMap) {
      for (unsigned j = 1; j < map.getNumResults(); ++j) {
        int64_t dim = map.getDimPosition(j);
        deletedDim.push_back(dim);
      }
    }
    return deletedDim;
  }

  /// \param shape of the mlir Value to study
  /// \param reshapeMap reshaping map for instance [[0, 1], [2]]
  /// \return true if there is a reshape and one of the element is dynamic,
  ///         eg. true if reshapeMap=[[0, 1], [2]] and shape[0] or shape[1] is dynamic
  ///         eg. false if reshapeMap=[[0, 1], [2]] and only shape[2] is dynamic
  [[nodiscard]] bool isDynamicReshape(const ArrayRef<int64_t> &shape,
                                      const SmallVector<AffineMap, kVectorSizeFour> &reshapeMap) const {
    for (auto m : reshapeMap) {
      if (m.getNumResults() > 1) {
        for (unsigned j = 0; j < m.getNumResults(); ++j) {
          int64_t dim = m.getDimPosition(j);
          auto upperbound = shape[dim];
          if (::mlir::ShapedType::isDynamic(upperbound)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  bool processDynamicDim(unsigned j, int64_t dim, DimAccumState &accum, MapProcessContext &ctx) {
    if (j > 0 && !this->allowNonPolyhedralAccess) {
      LLVM_DEBUG({
        llvm::dbgs() << DEBUG_TYPE
                     << " allow-non-polyhedral-access=false prevent removing an operation that will imply "
                        "potential non polyhedral access function\n";
      });
      ctx.result.skip = true;
      return false;
    }
    if (auto allocOp = dyn_cast<memref::AllocOp>(ctx.referenceOp)) {
      unsigned dynIndex = ctx.referenceType.getDynamicDimIndex(dim);
      auto dynamicVar = allocOp.getDynamicSizes()[dynIndex];
      accum.symbolicValue.push_back(dynamicVar);
      if (j > 0) {
        accum.accessRes = accum.accessRes * mlir::getAffineSymbolExpr(ctx.nbSymbolExpr, ctx.context);
      }
      accum.shapeRes = accum.shapeRes * mlir::getAffineSymbolExpr(ctx.nbSymbolExpr, ctx.context);
      ctx.nbSymbolExpr++;
    } else if (isa<memref::ExpandShapeOp>(ctx.referenceOp)) {
      if (j > 0) {
        llvm::errs() << DEBUG_TYPE
                     << " WARNING -- ExpandShapeOp - cannot compute new AffineMap Access function "
                        "because of "
                        "unknown dynamic/symbolic shape\n";
        ctx.result.skip = true;
        return false;
      }
    } else {
      llvm::errs() << DEBUG_TYPE << " getNewAffineMapResults- WARNING!!! This case may never happen\n";
      llvm::dbgs() << DEBUG_TYPE
                   << " WARNING -- cannot compute new AffineMap Access function because of unknown "
                      "dynamic/symbolic shape\n";
      ctx.result.skip = true;
      return false;
    }
    return true;
  }

  bool processMapEntry(AffineMap map, MapProcessContext &ctx) {
    DimAccumState accum;
    accum.accessRes = map.getResult(0);
    accum.shapeRes = mlir::getAffineConstantExpr(1, ctx.context);
    auto referenceShape = ctx.referenceType.getShape();

    for (unsigned j = 0; j < map.getNumResults(); ++j) {
      int64_t dim = map.getDimPosition(j);
      if (j > 0) {
        ctx.result.deletedDim.push_back(dim);
      }
      auto upperbound = referenceShape[dim];
      if (::mlir::ShapedType::isDynamic(upperbound)) {
        if (!processDynamicDim(j, dim, accum, ctx)) {
          return false;
        }
      } else {
        if (j > 0) {
          accum.accessRes = accum.accessRes * upperbound;
        }
        accum.shapeRes = accum.shapeRes * upperbound;
      }
      if (j > 0) {
        accum.accessRes = accum.accessRes + map.getResult(j);
      }
    }
    ctx.newAccesExpr.push_back(accum.accessRes);

    SmallVector<AffineMap, kSmallVectorSizeOne> newShapeMap =
      AffineMap::inferFromExprList({accum.shapeRes}, ctx.context);
    ctx.result.newShapeMap[newShapeMap[0]] = accum.symbolicValue;
    return true;
  }

  /// Compute the new access function + new shape with corresponding dynamic dim if necessary
  /// \param reshapeMap reshaping map for instance [[0, 1], [2]]
  /// \param referenceValue Value with the shape that will be modified
  ///                       if isCollapse, referenceValue.getDefiningOp() will be an memref::AllocOp,
  ///                       else an memref::ExpandShapeOp
  /// \param isCollapse if true treat for a collapse, else treat for a expand
  UnifyShapeInfos getNewAffineMapResults(SmallVector<AffineMap, kVectorSizeFour> reshapeMap, Value referenceValue) {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - getNewAffineMapResults:\n";
      llvm::dbgs() << " Initial mapping:\n";
      for (auto m : reshapeMap) {
        m.dump();
      }
      llvm::dbgs() << " reference Value:\n";
      referenceValue.dump();
    });
    MapProcessContext ctx;
    ctx.referenceType = cast<MemRefType>(referenceValue.getType());
    ctx.context = referenceValue.getContext();
    ctx.referenceOp = referenceValue.getDefiningOp();
    auto referenceShape = ctx.referenceType.getShape();

    ctx.result.dynamicReshape = isDynamicReshape(referenceShape, reshapeMap);

    if (!std::all_of(reshapeMap.begin(), reshapeMap.end(),
                     [this, &ctx](AffineMap map) { return processMapEntry(map, ctx); })) {
      return ctx.result;
    }

    if (reshapeMap.empty()) {
      if (isa<memref::ExpandShapeOp>(ctx.referenceOp)) {
        ctx.newAccesExpr.push_back(getAffineConstantExpr(0, ctx.context));
      }
    }

    SmallVector<AffineMap, kSmallVectorSizeOne> newAccessMap =
      AffineMap::inferFromExprList(ctx.newAccesExpr, ctx.context);
    assert(newAccessMap.size() == 1 && "Generation from a list of AffineExpr must result in only one AffineMap");
    LLVM_DEBUG({
      llvm::dbgs() << " new access map function:\n";
      newAccessMap[0].dump();
      llvm::dbgs() << " new shape map:\n";
      for (auto shape : ctx.result.newShapeMap) {
        llvm::dbgs() << shape.first << "  with \n";
        for (auto dim : shape.second) {
          dim.dump();
        }
      }
    });
    ctx.result.newAccessMap = newAccessMap[0];
    return ctx.result;
  }

  /// createnewAllocOp
  [[nodiscard]] mlir::memref::AllocOp createNewAllocOp(mlir::memref::AllocOp allocOp, MemRefType allocType,
                                                       const UnifyShapeInfos &shapeInfo) const {
    mlir::OpBuilder builder(allocOp);
    if (!shapeInfo.dynamicReshape) {
      auto newAlloc = builder.create<mlir::memref::AllocOp>(allocOp.getLoc(), allocType, allocOp.getDynamicSizes(),
                                                            allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());
      return newAlloc;
    }
    // 1. Create/compute the right dynamic size from original dynamic variable and new shape mapping
    SmallVector<Value, kVectorSizeFour> dynamicValue;
    for (auto shapeMap : shapeInfo.newShapeMap) {
      mlir::affine::AffineApplyOp applyOp =
        builder.create<mlir::affine::AffineApplyOp>(allocOp.getLoc(), shapeMap.first, shapeMap.second);
      dynamicValue.push_back(applyOp.getResult());
    }
    // 2. Create the new Alloc Op with the new dynamic size
    // NB: don't know to what correspond getSymbolOperands()...
    mlir::memref::AllocOp newAlloc = builder.create<mlir::memref::AllocOp>(
      allocOp.getLoc(), allocType, dynamicValue, allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());

    return newAlloc;
  }

  /// updateAffineOps update the access functions
  void updateAffineOps(Operation *uop, const UnifyShapeInfos &shapeInfo) const {
    if (!shapeInfo.dynamicReshape) {
      AffineMap aff = AffineMap::get(uop->getContext());
      StringRef attrStrName;

      if (isa<mlir::affine::AffineLoadOp>(uop)) {
        mlir::affine::AffineLoadOp alop = cast<mlir::affine::AffineLoadOp>(uop);
        aff = alop.getAffineMap();
        attrStrName = affine::AffineLoadOp::getMapAttrStrName();
      } else if (isa<mlir::affine::AffineStoreOp>(uop)) {
        mlir::affine::AffineStoreOp asop = cast<mlir::affine::AffineStoreOp>(uop);
        aff = asop.getAffineMap();
        attrStrName = affine::AffineStoreOp::getMapAttrStrName();
      }

      if (!aff.isEmpty()) {
        aff = getUpdatedAffineMap(aff, shapeInfo.newAccessMap);
        uop->setAttr(attrStrName, AffineMapAttr::get(aff));
      }
    } else {
      mlir::OpBuilder builder(uop);
      // improvement: if operation already have some dynamic access,
      //   just add new value at the end of the list may not be correct
      //   it may result with too many operand :(
      if (isa<mlir::affine::AffineLoadOp>(uop)) {
        mlir::affine::AffineLoadOp alop = cast<mlir::affine::AffineLoadOp>(uop);
        SmallVector<Value, kVectorSizeFour> operands = uop->getOperands();
        for (auto shape : shapeInfo.newShapeMap) {
          std::copy(shape.second.begin(), shape.second.end(), std::back_inserter(operands));
        }
        AffineMap aff = alop.getAffineMap();
        auto newOp = builder.create<mlir::affine::AffineLoadOp>(
          uop->getLoc(), getUpdatedAffineMap(aff, shapeInfo.newAccessMap), operands);
        alop.getResult().replaceAllUsesWith(newOp);
        uop->erase();
      } else if (isa<mlir::affine::AffineStoreOp>(uop)) {
        mlir::affine::AffineStoreOp asop = cast<mlir::affine::AffineStoreOp>(uop);
        SmallVector<Value, kVectorSizeFour> operands = asop.getIndices();
        for (auto shape : shapeInfo.newShapeMap) {
          std::copy(shape.second.begin(), shape.second.end(), std::back_inserter(operands));
        }
        AffineMap aff = asop.getAffineMap();
        (void)builder.create<mlir::affine::AffineStoreOp>(uop->getLoc(), asop.getValue(), asop.getMemref(),
                                                          getUpdatedAffineMap(aff, shapeInfo.newAccessMap), operands);
        uop->erase();
      }
    }
  }

  void updateFuncArgTypeForBlockArg(BlockArgument barg, Value target, const MemRefType &newmemreftype) {
    Block *owner = barg.getOwner();
    Operation *pop = owner->getParentOp();
    if (mlir::func::FuncOp fop = dyn_cast<mlir::func::FuncOp>(pop)) {
      FunctionType functionType = fop.getFunctionType();
      SmallVector<Type, kVectorSizeEight> newArgTypes;
      SmallVector<Type, kVectorSizeFour> resultTypes;
      FunctionType newFuncType;
      resultTypes = llvm::to_vector<kSmallVectorSizeFour>(functionType.getResults());

      assert(resultTypes.empty() &&
             "Function result must be empty due to the call of "
             "-buffer-results-to-out-params pass");

      for (BlockArgument &bbarg : fop.getArguments()) {
        if (bbarg != barg) {
          newArgTypes.push_back(bbarg.getType());
          continue;
        }
        target.setType(newmemreftype);
        newArgTypes.push_back(target.getType());
      }

      newFuncType = FunctionType::get(&getContext(), newArgTypes, resultTypes);
      fop.setType(newFuncType);
    }
  }

  /// In case an copy op is imply in a reshape,
  /// check if the reshape concern the source of the copy
  /// if so, try to update the target shape of the copy
  ///        (If the target of the copy is used many time, do not update)
  /// if the reshape concern the target of the copy,
  ///    it is not normal and may not occur.
  ///    Because it means that the variable was not declare with the right shape size
  bool updateCopyOps(Operation *uop, const Operation *reshapesop, const MemRefType &newmemreftype,
                     const UnifyShapeInfos &unifyShapeInfos) {
    if (!isa<mlir::memref::CopyOp>(uop)) {
      return true;
    }

    mlir::memref::CopyOp cop = cast<mlir::memref::CopyOp>(uop);
    Value source = cop.getSource();
    Value target = cop.getTarget();
    Operation *sourceOp = source.getDefiningOp();
    Operation *targetOp = target.getDefiningOp();
    // If the source is an expandshape, it means at this point, it is
    // ready to be erased
    if (sourceOp == reshapesop) {
      if (!target.hasOneUse()) {
        llvm::errs() << DEBUG_TYPE << " WARNING -- Unsafe operation: target has multiple uses.\n";
        return false;
      }

      if (BlockArgument barg = dyn_cast<BlockArgument>(target)) {
        if (keepArgsShape) {
          llvm::dbgs() << DEBUG_TYPE << " WARNING -- keepArgsShape=true, cannot update argument function shape\n";
          return false;
        }
        updateFuncArgTypeForBlockArg(barg, target, newmemreftype);
      } else if (mlir::memref::AllocOp oldAllocOp = dyn_cast_or_null<mlir::memref::AllocOp>(targetOp)) {
        mlir::memref::AllocOp newAlloc = createNewAllocOp(oldAllocOp, newmemreftype, unifyShapeInfos);
        // There is only one use which is for the concerned memcopy.
        // No need to do further updates.
        target.replaceAllUsesWith(newAlloc);
        targetOp->erase();
      }
    } else if (targetOp == reshapesop) {
      llvm::errs() << DEBUG_TYPE
                   << " WARNING -- This operation may be inconsistent. Please declare source with appropriate "
                      "type directly\n";
      return false;
    } else {
      LLVM_DEBUG({
        llvm::dbgs() << DEBUG_TYPE
                     << " WARNING -- unexpected case. Collapse shape use alloc Operation instead of reshape "
                        "one? Retry\n";
      });
      return false;
    }
    return true;
  }

  /// Search for every ExpandShapeOp to remove them.
  /// The remove of Expand Shape will consist of update the user of the expandShape
  /// and keep the initial/original shape.
  /// So the access functions to modify are the one that are after the expandShapeOp
  //  void eraseExpandShapeOps(mlir::func::FuncOp &fop) {
  void eraseExpandShapeOps(mlir::ModuleOp &m) {
    // a = alloc() : memref<a0 x type>
    // for i, 0 a0
    //     a[i] = c[i]
    // b = expand_shape a [[0, 1]]: memref<a00 x a01 x type>
    // for i, 0 a00
    //     for j, 0 a01
    //         d[i, j] = b[i, j]
    // -->
    // a = alloc() : memref<a0 x type>
    // for i, 0 a0
    //     a[i] = c[i]
    // for i, 0 a00
    //     for j, 0 a01
    //         d[i, j] = a[i*a01 + j]
    (void)m.walk([this](mlir::memref::ExpandShapeOp esop) {
      LLVM_DEBUG({
        llvm::dbgs() << "\n" << DEBUG_TYPE << " - work on operation:\n";
        esop.dump();
      });
      Value initValue = esop.getOperands()[0];
      // Cases where a BlockArgument is the operand should not be
      // handled in this function
      if (!isa<BlockArgument>(initValue)) {
        // 1. Get the shape of the operand, which is the one we want to keep
        // Also get the expand shape reassociation information:
        // we need it to update concerned access functions
        Value resultValue = esop.getResult();
        SmallVector<AffineMap, kVectorSizeFour> esopMap = esop.getReassociationMaps();
        auto unifyShapeInfos = getNewAffineMapResults(esopMap, resultValue);
        if (unifyShapeInfos.skip) {
          LLVM_DEBUG({
            llvm::dbgs() << DEBUG_TYPE << " - Cannot remove the following Operation:\n";
            esop.dump();
          });
          return WalkResult::skip();
        }
        if (unifyShapeInfos.dynamicReshape) {
          for (Operation *uop : resultValue.getUsers()) {
            if (isa<mlir::memref::DimOp>(uop)) {
              LLVM_DEBUG({
                llvm::dbgs() << DEBUG_TYPE
                             << " - Cannot remove the following Operation because of dynamic shape "
                                "that was fused and "
                                "new dim used:\n";
                esop.dump();
                uop->dump();
              });
              return WalkResult::skip();
            }
          }
        }

        // 2. Find users of resultValue to update their access function
        // NB: if there is many users with many copy, and that it is the last was that fail,
        //       we may result in an inconsistent code... :(
        for (Operation *uop : resultValue.getUsers()) {
          // If updateCopyOps returns false, there is a problem with
          // the attempt to erase this ExpandShapeOp. Skip.
          if (!updateCopyOps(uop, esop, cast<MemRefType>(initValue.getType()), unifyShapeInfos)) {
            llvm::dbgs() << DEBUG_TYPE
                         << " WARNING -- failed to update corresponding copy Op, a expandShapeOp will be kept\n";
            return WalkResult::skip();
          }
          updateAffineOps(uop, unifyShapeInfos);
          if (mlir::memref::DimOp dimOp = dyn_cast<mlir::memref::DimOp>(uop)) {
            updateDimOp(dimOp, unifyShapeInfos.deletedDim);
          }
        }

        // 3. Replace all the usage of the resultValue of csop by the newAlloc Value
        resultValue.replaceAllUsesWith(initValue);
        // NB. if 3. exec before 2. dimension issue can occur

        // 4. Remove the ExpandShapeOp esop
        esop.erase();
      }
      return WalkResult::advance();
    });
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - After eraseExpandShapeOps:\n";
      m.dump();
    });
  }

  /// Search for every CollapseShapeOp to remove them.
  /// The remove of Collapse Shape will consist of update the declaration
  /// and keep the resulting shape.
  /// So the access functions to modify are the one that are before the collapseShapeOp
  // void eraseCollapseShapeOps(mlir::func::FuncOp &fop) {
  void eraseCollapseShapeOps(mlir::ModuleOp &m) {
    // a = alloc() : memref<a00 x a01 x type>
    // for i, 0 a00
    //     for j, 0 a01
    //         a[i, j] = c[i, j]
    // b = collapse_shape a [[0, 1]]: memref<a0 x type>
    // for i, 0 a0
    //     d[i] = b[i]
    // -->
    // a = alloc() : memref<a0 x type>
    // for i, 0 a00
    //     for j, 0 a01
    //         a[i*a01 + j] = c[i, j]
    // for i, 0 a0
    //     d[i] = a[i]
    (void)m.walk([this](mlir::memref::CollapseShapeOp csop) {
      LLVM_DEBUG({
        llvm::dbgs() << "\n" << DEBUG_TYPE << " - work on operation:\n";
        csop.dump();
      });
      Value initValue = csop.getOperand();
      // Cases where a BlockArgument is the operand should not be
      // handled in this function
      if (!isa<BlockArgument>(initValue)) {
        MemRefType resultType = cast<MemRefType>(csop.getType());
        // 1. Get the shape of the operand, which is the one we want to keep
        // Also get the collapse shape reassociation information:
        // we need it to update concerned access functions
        SmallVector<AffineMap, kVectorSizeFour> csopMap = csop.getReassociationMaps();
        auto unifyShapeInfos = getNewAffineMapResults(csopMap, initValue);
        if (unifyShapeInfos.skip) {
          LLVM_DEBUG({
            llvm::dbgs() << DEBUG_TYPE << " - Cannot remove the following Operation:\n";
            csop.dump();
          });
          return WalkResult::skip();
        }
        if (unifyShapeInfos.dynamicReshape) {
          for (Operation *uop : initValue.getUsers()) {
            if (isa<mlir::memref::DimOp>(uop)) {
              LLVM_DEBUG({
                llvm::dbgs() << DEBUG_TYPE
                             << " - Cannot remove the following Operation because of dynamic shape "
                                "that was fused and "
                                "new dim used:\n";
                csop.dump();
                uop->dump();
              });
              return WalkResult::skip();
            }
          }
        }

        // 2. If operand is a memref alloc, replace it with a new alloc using result type
        Operation *initAlloc = initValue.getDefiningOp();
        if (mlir::memref::AllocOp allocOp = dyn_cast_or_null<mlir::memref::AllocOp>(initAlloc)) {
          mlir::memref::AllocOp newAlloc = createNewAllocOp(allocOp, resultType, unifyShapeInfos);
          // 3. Find users of initValue to update their access function
          // NB: if there is many users with many copy, and that it is the last was that fail,
          //       we may result in an inconsistent code... :(
          for (Operation *uop : initValue.getUsers()) {
            // If updateCopyOps returns false, there is a problem with
            // the attempt to erase this CollapseShapeOp. Skip.
            if (!updateCopyOps(uop, csop, resultType, unifyShapeInfos)) {
              if (!updateCopyOps(uop, allocOp, resultType, unifyShapeInfos)) {
                llvm::dbgs() << DEBUG_TYPE
                             << " WARNING -- failed to update corresponding copy Op, a collapseShapeOp will be kept\n";
                return WalkResult::skip();
              }
            }
            updateAffineOps(uop, unifyShapeInfos);
            if (mlir::memref::DimOp dimOp = dyn_cast<mlir::memref::DimOp>(uop)) {
              updateDimOp(dimOp, unifyShapeInfos.deletedDim);
            }
          }

          // 4. Replace all the usage of the allocOp (initValue) by the newAlloc Value
          // and remove the allocOp Operator that is no more used
          initValue.replaceAllUsesWith(newAlloc);
          allocOp->erase();

          // 5. Replace all the usage of the resultValue of csop by the newAlloc Value
          Value resultValue = csop.getResult();
          resultValue.replaceAllUsesWith(newAlloc);

          // 6. Remove the CollapseShapeOp csop
          csop.erase();
        } else {
          llvm::errs() << DEBUG_TYPE << " - WARNING -- collapseShapeOp will be kept - Untreat DefiningOp:\n";
          initAlloc->dump();
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - After eraseCollapseShapeOps:\n";
      m.dump();
    });
  }

  /// runOnOperation is on moduleOp and not on FuncOp because we can modify
  /// the argument of the function
  /// WARNING: It can modify the function argument
  ///          BUT do not try to search for his caller to update them !!!
  /// option keepArgsShape = true not well manage and really guarantee to keep them
  /// Except for the BlockArgment, the strategy of UnifyShape will be to keep the
  /// smaller number shape and update the access function.
  /// The new access functions will become something like [a * sizeof(b) + b]
  /// This strategy is decided because we do not want to modify the affine.for operation
  /// create new affine.for operation by explicitly stripmining them can result to an
  /// unknown decision.
  /// If 1 variable is expand and collapse many time with really different value,
  /// it may result to an undecidable decision
  /// Moreover if modify the affine.for loop (even just the bound), will result to modify
  /// by side effect the access function of all other variable present in the loop
  /// Another strategy can be to keep the bigger possible shape and still update the access function.
  /// The new access function will then become something like [a floordiv sizeof(b), a mod sizeof(b)]
  /// This kind of access function is not really affine friendly, and it is recommended to avoid it
  /// For Dynamic shape,
  /// Some restriction will be required
  ///   relation with taking dim, will remove function updateDim???
  void runOnOperation() override {
    // mlir::func::FuncOp fop = getOperation();
    mlir::ModuleOp m = getOperation();

    // 1. handle expand/collapse using a BlockArgument as operand
    if (!keepArgsShape) {
      /// for each arguments, it will search there usage.
      /// if there is only 1 use and that use is a collapse/expand
      ///    update the shape of the BlockArgument and remove the collapse/expand
      ///    to directly use the Blockargument
      /// if there is many use do not modify the BlockArgument (even if only 1 collapse/expand)
      ///    because cannot make a decision on which shape will be the best one
      eraseOpsUsingBlockArguments(m);
    }

    // 2. Handle expandShapeOp and collapseShapeOp
    /// Search for every ExpandShapeOp to remove them.
    /// The remove of Expand Shape will consist of update the user of the expandShape
    /// and keep the initial/original shape. (smaller number of shape)
    /// So the access functions to modify are the one that are after the expandShapeOp
    eraseExpandShapeOps(m);
    /// Search for every CollapseShapeOp to remove them.
    /// The remove of Collapse Shape will consist of update the declaration
    /// and keep the resulting shape.
    /// So the access functions to modify are the one that are before the collapseShapeOp
    eraseCollapseShapeOps(m);
  }
};

}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createUnifyShapePass() {
  return std::make_unique<mlir::UnifyShape>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createUnifyShapePass(bool allowNonPolyhedralAccess,
                                                                                bool keepArg = false) {
  return std::make_unique<mlir::UnifyShape>(allowNonPolyhedralAccess, keepArg);
}
