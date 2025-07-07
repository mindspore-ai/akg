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

// ===- UnifyShape.cc - Remove expand/collapse shape ops -------------------=== //
//
// This file implements the transformation passes to remove expand/collapse
// shapes operations.
//
// ===----------------------------------------------------------------------=== //

#include "akg/Dialect/Affine/Transforms/UnifyShape.h"

namespace mlir {
#ifndef GEN_PASS_DEF_UNIFYSHAPE
#define GEN_PASS_DEF_UNIFYSHAPE
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
}  // namespace mlir

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

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

class UnifyShapePass : public mlir::impl::UnifyShapeBase<UnifyShapePass> {
 public:
  UnifyShapePass() = default;
  UnifyShapePass(const UnifyShapePass &pass) = default;
  UnifyShapePass(const bool allowNonPolyhedralAccess, const bool keepArg) {
    this->allowNonPolyhedralAccess = allowNonPolyhedralAccess;
    this->keepArgsShape = keepArg;
  }
  explicit UnifyShapePass(const UnifyShapeOptions &options) : UnifyShapeBase(options) {}

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
        //
        // We could think of calling argsFindAndEraseExpandAndCollapseShapeOps recursively but this poses
        // some issues:
        // * At this point, userOp is NOT yet deleted (and we can't do it yet otherwise it will cause issues).
        // Therefore, it still appears in the list of uses and a recursive call of this function will
        // cause to fall into the else section. However, for now it is not yet clear
        // what other cases of multiple uses can occur, other than cases of "cascade". Therefore
        // it is not clear whether cascade must be handled in the else condition of this function.
        //
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
    m.walk([&](mlir::func::FuncOp fop) {
      FunctionType functionType = fop.getFunctionType();
      SmallVector<Type, kVectorSizeEight> newArgTypes;
      SmallVector<Type, kVectorSizeFour> resultTypes;
      FunctionType newFuncType;
      resultTypes = llvm::to_vector<4>(functionType.getResults());

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

        // 2. if new index to acces is diff from original one, update it
        if (indexValue != newIndex) {
          Type type = attr.getType();
          mlir::OpBuilder builder(cop);
          mlir::arith::ConstantOp newcstOp =
            builder.create<mlir::arith::ConstantOp>(cop.getLoc(), type, IntegerAttr::get(type, newIndex));
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
                     << dimOp << "Unkown kind of attribut for" << cop << "May result to a wrong running code...\n";
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

  SmallVector<int64_t, kVectorSizeFour> getDeletedDim(const SmallVector<AffineMap, kVectorSizeFour> &reshapeMap) const {
    SmallVector<int64_t, kVectorSizeFour> deletedDim;
    for (unsigned i = 0; i < reshapeMap.size(); ++i) {
      AffineMap map = reshapeMap[i];
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
  bool isDynamicReshape(const ArrayRef<int64_t> &shape,
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

  /// Compute the new access function + new shape with corresponding dynamic dim if necessary
  /// \param reshapeMap reshaping map for instance [[0, 1], [2]]
  /// \param referenceValue Value with the shape that will be modified
  ///                       if isCollapse, referenceValue.getDefiningOp() will be an memref::AllocOp,
  ///                       else an memref::ExpandShapeOp
  /// \param isCollapse if true treat for a collapse, else treat for a expand
  UnifyShapeInfos getNewAffineMapResults(SmallVector<AffineMap, kVectorSizeFour> reshapeMap, Value referenceValue) {
    // We may be able to already prepare the new map
    // Each map returns a tuple of dim (d0, ..., dN)
    // we want to resturn instead (d0 * M_d1 + d1 * ... + dN-1 * M_dN + dN)
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - getNewAffineMapResults:\n";
      llvm::dbgs() << " Initial mapping:\n";
      for (auto m : reshapeMap) {
        m.dump();
      }
      llvm::dbgs() << " reference Value:\n";
      referenceValue.dump();
    });
    UnifyShapeInfos result;
    MemRefType referenceType = cast<MemRefType>(referenceValue.getType());
    MLIRContext *context = referenceValue.getContext();
    Operation *referenceOp = referenceValue.getDefiningOp();
    auto referenceShape = referenceType.getShape();
    unsigned nbSymbolExpr = 0;
    SmallVector<AffineExpr, kVectorSizeFour> newAccesExpr;

    result.dynamicReshape = isDynamicReshape(referenceShape, reshapeMap);

    for (unsigned i = 0; i < reshapeMap.size(); ++i) {
      SmallVector<Value, kVectorSizeFour> symbolicValue;
      AffineMap map = reshapeMap[i];
      AffineExpr accessRes = map.getResult(0);
      AffineExpr shapeRes = mlir::getAffineConstantExpr(1, context);

      for (unsigned j = 0; j < map.getNumResults(); ++j) {
        int64_t dim = map.getDimPosition(j);
        if (j > 0) {
          result.deletedDim.push_back(dim);
        }
        auto upperbound = referenceShape[dim];
        if (::mlir::ShapedType::isDynamic(upperbound)) {
          if (j > 0 && !this->allowNonPolyhedralAccess) {
            LLVM_DEBUG({
              llvm::dbgs() << DEBUG_TYPE
                           << " allow-non-polyhedral-access=false prevent removing an operation that will imply "
                              "potential non polyhedral access function\n";
            });
            result.skip = true;
            return result;
          }
          // Note: we cannot create a memref::DimOp, to retrieve the dynamic/symbolic shape
          // Because there the value of this index will be updated by a bigger value
          // Since the dimension are expand/collapse... (and only keep the smaller number of dimension)
          if (memref::AllocOp allocOp = dyn_cast<memref::AllocOp>(referenceOp)) {
            // For memory referenceType is the MemRefType of allocOp
            unsigned dynIndex = referenceType.getDynamicDimIndex(dim);
            auto dynamicVar = allocOp.getDynamicSizes()[dynIndex];
            symbolicValue.push_back(dynamicVar);

            if (j > 0) {
              accessRes = accessRes * mlir::getAffineSymbolExpr(nbSymbolExpr, context);
            }
            shapeRes = shapeRes * mlir::getAffineSymbolExpr(nbSymbolExpr, context);
            nbSymbolExpr++;
          } else if (memref::ExpandShapeOp ExpandOp = dyn_cast<memref::ExpandShapeOp>(referenceOp)) {
            if (j > 0) {
              // If there is dynamic dimension that are not the outtermost dim to expand not ok
              // example:
              // NOT OK memref.expand_shape [[0, 1]] : memref<?xf16> into memref<1024x?xf16>
              // OK memref.expand_shape [[0, 1]] : memref<?xf16> into memref<?x1024xf16>
              // OK memref.expand_shape [[0, 1], 2] : memref<?x?xf16> into memref<?x1024x?xf16>
              llvm::errs() << DEBUG_TYPE
                           << " WARNING -- ExpandShapeOp - cannot compute new AffineMap Access function "
                              "because of "
                              "unknown dynamic/symbolic shape\n";
              result.skip = true;
              return result;
            }
          } else {  // referenceOp is not a  AllocOp or ExpandShapeOp
            llvm::errs() << DEBUG_TYPE << " getNewAffineMapResults- WARNING!!! This case may never happen\n";
            llvm::dbgs() << DEBUG_TYPE
                         << " WARNING -- cannot compute new AffineMap Access function because of unknown "
                            "dynamic/symbolic shape\n";
            result.skip = true;
            return result;
          }
        } else {  // if (::mlir::ShapedType::isDynamic(upperbound))
          if (j > 0) {
            accessRes = accessRes * upperbound;
          }
          shapeRes = shapeRes * upperbound;
        }
        if (j > 0) {
          accessRes = accessRes + map.getResult(j);
        }
      }
      newAccesExpr.push_back(accessRes);

      SmallVector<AffineMap, 1> newShapeMap = AffineMap::inferFromExprList({shapeRes}, context);
      result.newShapeMap[newShapeMap[0]] = symbolicValue;
    }

    // special case, with memref.expand_shape %alloc [] : memref<f32> into memref<1xf32>
    // need to have similar stuff for collapse_shape???
    if (reshapeMap.size() == 0) {
      if (isa<memref::ExpandShapeOp>(referenceOp)) {
        newAccesExpr.push_back(getAffineConstantExpr(0, context));
      }
    }

    SmallVector<AffineMap, 1> newAccessMap = AffineMap::inferFromExprList(newAccesExpr, context);
    assert(newAccessMap.size() == 1 && "Generation from a list of AffineExpr must result in only one AffineMap");
    LLVM_DEBUG({
      llvm::dbgs() << " new access map function:\n";
      newAccessMap[0].dump();
      llvm::dbgs() << " new shape map:\n";
      for (auto shape : result.newShapeMap) {
        llvm::dbgs() << shape.first << "  with \n";
        for (auto dim : shape.second) {
          dim.dump();
        }
      }
    });
    result.newAccessMap = newAccessMap[0];
    return result;
  }

  /// createnewAllocOp
  mlir::memref::AllocOp createNewAllocOp(mlir::memref::AllocOp allocOp, MemRefType allocType,
                                         const UnifyShapeInfos &shapeInfo) const {
    mlir::OpBuilder builder(allocOp);
    if (!shapeInfo.dynamicReshape) {
      mlir::memref::AllocOp newAlloc =
        builder.create<mlir::memref::AllocOp>(allocOp.getLoc(), allocType, allocOp.getDynamicSizes(),
                                              allocOp.getSymbolOperands(), allocOp.getAlignmentAttr());
      return newAlloc;
    } else {
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
      // improvment: if operation already have some dynamic access,
      //   just add new value at the end of the list may not be correct
      //   it may result with too many operand :(
      if (isa<mlir::affine::AffineLoadOp>(uop)) {
        mlir::affine::AffineLoadOp alop = cast<mlir::affine::AffineLoadOp>(uop);
        SmallVector<Value, kVectorSizeFour> operands = uop->getOperands();
        for (auto shape : shapeInfo.newShapeMap) {
          for (auto value : shape.second) {
            operands.push_back(value);
          }
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
          for (auto value : shape.second) {
            operands.push_back(value);
          }
        }
        AffineMap aff = asop.getAffineMap();
        (void)builder.create<mlir::affine::AffineStoreOp>(uop->getLoc(), asop.getValue(), asop.getMemref(),
                                                          getUpdatedAffineMap(aff, shapeInfo.newAccessMap), operands);
        uop->erase();
      }
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

      // Handle BlockArguments
      if (BlockArgument barg = dyn_cast<BlockArgument>(target)) {
        if (keepArgsShape) {
          llvm::dbgs() << DEBUG_TYPE << " WARNING -- keepArgsShape=true, cannot update argument function shape\n";
          return false;
        }

        Block *owner = barg.getOwner();
        Operation *pop = owner->getParentOp();
        if (mlir::func::FuncOp fop = dyn_cast<mlir::func::FuncOp>(pop)) {
          FunctionType functionType = fop.getFunctionType();
          SmallVector<Type, kVectorSizeEight> newArgTypes;
          SmallVector<Type, kVectorSizeFour> resultTypes;
          FunctionType newFuncType;
          resultTypes = llvm::to_vector<4>(functionType.getResults());

          assert(resultTypes.empty() &&
                 "Function result must be empty due to the call of "
                 "-buffer-results-to-out-params pass");

          for (BlockArgument &bbarg : fop.getArguments()) {
            if (bbarg == barg) {
              target.setType(newmemreftype);
              newArgTypes.push_back(target.getType());
            } else {
              newArgTypes.push_back(bbarg.getType());
            }
          }

          newFuncType = FunctionType::get(&getContext(), newArgTypes, resultTypes);
          fop.setType(newFuncType);
        }
      }
      // Handle AllocOps
      else if (mlir::memref::AllocOp oldAllocOp = dyn_cast_or_null<mlir::memref::AllocOp>(targetOp)) {
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
    //
    // -->
    // a = alloc() : memref<a0 x type>
    // for i, 0 a0
    //     a[i] = c[i]
    // for i, 0 a00
    //     for j, 0 a01
    //         d[i, j] = a[i*a01 + j]
    (void)m.walk([&](mlir::memref::ExpandShapeOp esop) {
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
        //       we may result in an inconsistant code... :(
        for (Operation *uop : resultValue.getUsers()) {
          // If updateCopyOps returns false, there is a problem with
          // the attempt to erase this ExpandShapeOp. Skip.
          if (!updateCopyOps(uop, esop, cast<MemRefType>(initValue.getType()), unifyShapeInfos)) {
            llvm::dbgs() << DEBUG_TYPE
                         << " WARNING -- failed to update corresponding copy Op, a expandShapeOp will be keept\n";
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
    //
    // -->
    // a = alloc() : memref<a0 x type>
    // for i, 0 a00
    //     for j, 0 a01
    //         a[i*a01 + j] = c[i, j]
    // for i, 0 a0
    //     d[i] = a[i]
    (void)m.walk([&](mlir::memref::CollapseShapeOp csop) {
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
          //       we may result in an inconsistant code... :(
          for (Operation *uop : initValue.getUsers()) {
            // If updateCopyOps returns false, there is a problem with
            // the attempt to erase this CollapseShapeOp. Skip.
            if (!updateCopyOps(uop, csop, resultType, unifyShapeInfos)) {
              if (!updateCopyOps(uop, allocOp, resultType, unifyShapeInfos)) {
                llvm::dbgs() << DEBUG_TYPE
                             << " WARNING -- failed to update corresponding copy Op, a collapseShapeOp "
                                "will be keept\n";
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
          llvm::errs() << DEBUG_TYPE << " - WARNING -- collapseShapeOp will be keept - Untreat DefiningOp:\n";
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
  /// option keepArgsShape = true not well manage and really garanty to keep them
  ///
  /// Except for the BlockArgment, the strategy of UnifyShape will be to keep the
  /// smaller number shape and update the access function.
  /// The new access functions will become something like [a * sizeof(b) + b]
  /// This strategy is decided because we do not want to modify the affine.for operation
  /// create new affine.for operation by explicitly stripmining them can result to an
  /// unknown decision.
  /// If 1 variable is expand and collapse many time with really different value,
  /// it may result to an undecidable decision
  /// Moreover if modify the affine.for loop (even just the bound), will result to modify
  /// by side effect the access funciton of all other variable present in the loop
  ///
  /// Another strategy can be to keep the bigger possible shape and still update the access function.
  /// The new access function will then become something like [a floordiv sizeof(b), a mod sizeof(b)]
  /// This kind of access function is not really affine friendly, and it is recommended to avoid it
  ///
  /// For Dynamic shape,
  /// Some restriction will be required
  ///   relation with taking dim, will remove function updateDim???
  void runOnOperation() override {
    // mlir::func::FuncOp fop = getOperation();
    mlir::ModuleOp m = getOperation();

    // 1. handle expand/collapse using a BlockArgument as operand
    if (!keepArgsShape) {
      /// for each argments, it will search there usage.
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
  return std::make_unique<mlir::UnifyShapePass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> mlir::createUnifyShapePass(bool allowNonPolyhedralAccess,
                                                                                bool keepArg = false) {
  return std::make_unique<mlir::UnifyShapePass>(allowNonPolyhedralAccess, keepArg);
}
