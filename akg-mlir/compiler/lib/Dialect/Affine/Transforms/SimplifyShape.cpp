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

// ===- SimplifyShape.cpp - Simplify dimension 1 shape ---------------------=== //
//

// This file implements the transformation passes to remove dimension 1 from the shapes.
//
// ===----------------------------------------------------------------------=== //

#include "akg/Dialect/Affine/Transforms/SimplifyShape.h"
#include "akg/Utils/AKGGlobalVars.hpp"
using akgglobal::ShapeAlignTool;

namespace mlir {
#ifndef GEN_PASS_DEF_SIMPLIFYSHAPE
#define GEN_PASS_DEF_SIMPLIFYSHAPE
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
}  // namespace mlir

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#define DEBUG_TYPE "simplify-shape"

namespace mlir {

static constexpr const int kVectorSizeFour = 4;
static constexpr const int kVectorSizeTwo = 2;
static constexpr const int kVectorSizeEight = 8;

using SimplifiedShapeInfos = std::pair<MemRefType, SmallVector<int64_t, kVectorSizeFour>>;

struct SimplifyShapePass : public mlir::impl::SimplifyShapeBase<SimplifyShapePass> {
 public:
  SimplifyShapePass() = default;
  SimplifyShapePass(const SimplifyShapePass &pass) = default;
  explicit SimplifyShapePass(const bool keepArg) { this->keepArgsShape = keepArg; }
  explicit SimplifyShapePass(const SimplifyShapeOptions &options) : SimplifyShapeBase(options) {}

  SimplifiedShapeInfos getSimplifiedShapeInfos(const MemRefType &mrtype) const {
    MemRefType newmrtype = mrtype;
    ArrayRef<int64_t> shape = mrtype.getShape();
    SmallVector<int64_t, kVectorSizeFour> todelete;
    SmallVector<int64_t, kVectorSizeFour> newShape;

    if (!shape.empty()) {
      for (unsigned i = 0; i < shape.size(); ++i) {
        if (shape[i] == 1) {
          todelete.push_back(i);
        } else {
          newShape.push_back(shape[i]);
        }
      }
    }

    ArrayRef<int64_t> newShapeRef(newShape);
    if (newShapeRef != shape) {
      newmrtype = MemRefType::get(newShapeRef, mrtype.getElementType());
    }

    return std::make_pair(newmrtype, todelete);
  }

  void updateDimOp(mlir::memref::DimOp dimOp, const SmallVector<int64_t, kVectorSizeFour> &todelete) const {
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
        for (uint64_t i : todelete) {
          assert(indexValue != i && "index dimension try to retrieve a index that is remove... (know to be 1)");
          if (indexValue.ugt(i)) {
            --newIndex;
          }
        }

        // 2. if new index to acces is diff from original one, update it
        if (indexValue != newIndex) {
          Type type = attr.getType();
          mlir::OpBuilder builder(cop);
          mlir::arith::ConstantOp newcstOp =
            builder.create<mlir::arith::ConstantOp>(cop.getLoc(), type, IntegerAttr::get(type, newIndex));
          dimOp->replaceUsesOfWith(idx, newcstOp.getResult());
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
  }

  template <typename T>
  T updateReassociationMaps(T shapeOp, const SmallVector<int64_t, kVectorSizeFour> &todelete, MemRefType resultShape,
                            Value operand) const {
    SmallVector<SmallVector<int64_t, kVectorSizeTwo>, kVectorSizeFour> oldShapeOpIndices =
      shapeOp.getReassociationIndices();
    SmallVector<SmallVector<int64_t, kVectorSizeTwo>, kVectorSizeFour> newShapeOpIndices;

    // We recreate a newShapeOpIndices where all indexes are
    // consecutive and the associations correspond to that
    // which match the simplified shape of the corresponding memref.
    // We check whether an index should be kept
    // or not (if should be deleted instead). We increment
    // the newIndex value, only if we know that an index will
    // be kept.
    int newIndex = 0;
    for (auto association : oldShapeOpIndices) {
      mlir::ReassociationIndices new_association;
      for (auto index : association) {
        if (std::find(todelete.begin(), todelete.end(), index) == todelete.end()) {
          new_association.push_back(newIndex);
          newIndex++;
        }
      }
      if (!new_association.empty()) {
        newShapeOpIndices.push_back(new_association);
      }
    }

    ArrayRef<mlir::ReassociationIndices> newOnes(newShapeOpIndices);
    mlir::OpBuilder builder(shapeOp);
    auto loc = shapeOp.getLoc();
    T newShapeOp = builder.create<T>(loc, resultShape, operand, newOnes);

    return newShapeOp;
  }

  template <typename T>
  void simplifyMemrefReshapeOp(T reshapeOp, const Value &initValue, const SimplifiedShapeInfos &initSimplifyInfos) {
    MemRefType initSimplifyType = initSimplifyInfos.first;

    Value resultValue = reshapeOp.getResult();
    MemRefType resultType = resultValue.getType().cast<mlir::MemRefType>();
    const SimplifiedShapeInfos resultSimplifyInfos = getSimplifiedShapeInfos(resultType);
    MemRefType resultSimplifyType = resultSimplifyInfos.first;

    simplifyValue(resultValue, resultType, resultSimplifyInfos);

    // If the simplified initValue type == the symplified result type
    //   Just replace the usage
    // Else create a new one
    if (initSimplifyType == resultSimplifyType) {
      resultValue.replaceAllUsesWith(initValue);
    } else {
      const SmallVector<int64_t, kVectorSizeFour> todelete =
        isa<mlir::memref::CollapseShapeOp>(reshapeOp) ? initSimplifyInfos.second : resultSimplifyInfos.second;
      T newShapeOp = updateReassociationMaps<T>(reshapeOp, todelete, resultSimplifyType, reshapeOp.getOperand());
      reshapeOp->replaceAllUsesWith(newShapeOp);
    }
    reshapeOp.erase();
  }

  AffineMap getSimplifiedAffineMap(AffineMap am, const SmallVector<int64_t, kVectorSizeFour> &todelete) const {
    AffineMap updatedAffineMap = am;

    if (!todelete.empty()) {
      updatedAffineMap = am.dropResults(todelete);
    }
    return updatedAffineMap;
  }

  template <typename T>
  void simplifyAffineOperation(T o, const SmallVector<int64_t, kVectorSizeFour> &todelete) const {
    auto initialAffineMap = o.getAffineMap();
    AffineMap newAffineMap = getSimplifiedAffineMap(initialAffineMap, todelete);
    if (newAffineMap != initialAffineMap) {
      AffineMapAttr simplifiedAffineMapAttr = AffineMapAttr::get(newAffineMap);
      o->setAttr(T::getMapAttrStrName(), simplifiedAffineMapAttr);
    }
  }

  void simplifyAffineOps(Operation *op, SmallVector<int64_t, kVectorSizeFour> todelete) {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - simplifyAffineOps START:\n";
      op->dump();
      for (auto d : todelete) {
        llvm::dbgs() << d << " ";
      }
      llvm::dbgs() << "\n";
    });
    if (mlir::AffineStoreOp asop = dyn_cast<mlir::AffineStoreOp>(op)) {
      simplifyAffineOperation<mlir::AffineStoreOp>(asop, todelete);
    } else if (mlir::AffineLoadOp alop = dyn_cast<mlir::AffineLoadOp>(op)) {
      simplifyAffineOperation<mlir::AffineLoadOp>(alop, todelete);
    }
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - simplifyAffineOps END:\n";
      op->dump();
    });
  }

  void simplifyValue(Value initValue, const MemRefType &initType, const SimplifiedShapeInfos &initSimplifyInfos) {
    MemRefType initSimplifyType = initSimplifyInfos.first;
    SmallVector<int64_t, kVectorSizeFour> todelete = initSimplifyInfos.second;
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - simplifyValue\n";
      initValue.dump();
      initType.dump();
      llvm::dbgs() << "\n";
      initSimplifyType.dump();
      llvm::dbgs() << "\n";
      for (auto d : todelete) {
        llvm::dbgs() << d << " ";
      }
      llvm::dbgs() << "\n";
    });

    for (Operation *userOp : initValue.getUsers()) {
      // In cases of AffineLoad/StoreOps, just update
      // those which will have a simplified initValue type
      // as operand
      if (initType != initSimplifyType && !todelete.empty()) {
        simplifyAffineOps(userOp, todelete);
        if (mlir::memref::DimOp dimOp = dyn_cast<mlir::memref::DimOp>(userOp)) {
          updateDimOp(dimOp, todelete);
        }
      }

      // We seek to remove expands and collapses those which
      // full simplifcation lead to identity operations
      if (mlir::memref::CollapseShapeOp collapseOp = dyn_cast<mlir::memref::CollapseShapeOp>(userOp)) {
        simplifyMemrefReshapeOp<mlir::memref::CollapseShapeOp>(collapseOp, initValue, initSimplifyInfos);
      } else if (mlir::memref::ExpandShapeOp expandOp = dyn_cast<mlir::memref::ExpandShapeOp>(userOp)) {
        simplifyMemrefReshapeOp<mlir::memref::ExpandShapeOp>(expandOp, initValue, initSimplifyInfos);
      }
    }
  }

  void simplifyOpsUsingBlockArguments(mlir::ModuleOp &m) {
    ShapeAlignTool &tool = ShapeAlignTool::getInstance();
    m.walk([&](mlir::func::FuncOp fop) {
      FunctionType functionType = fop.getFunctionType();
      SmallVector<Type, kVectorSizeEight> newArgTypes;
      SmallVector<Type, kVectorSizeFour> resultTypes;
      FunctionType newFuncType;
      resultTypes = llvm::to_vector<4>(functionType.getResults());

      assert(resultTypes.empty() &&
             "Function result must be empty due to the call of "
             "-buffer-results-to-out-params pass");

      size_t argIdx = 0;
      for (BlockArgument &bbArg : fop.getArguments()) {
        MemRefType argType = bbArg.getType().cast<MemRefType>();
        const SimplifiedShapeInfos argSimplifiedInfos = getSimplifiedShapeInfos(argType);
        MemRefType argSimplifyType = argSimplifiedInfos.first;
        tool.alignStaticShapeReconstruct(argIdx, argType.dyn_cast<Type>(), argSimplifyType.dyn_cast<Type>());
        simplifyValue(bbArg, argType, argSimplifiedInfos);
        bbArg.setType(argSimplifyType);
        // update the type of arg
        newArgTypes.push_back(argSimplifyType);
        argIdx++;
      }

      newFuncType = FunctionType::get(&getContext(), newArgTypes, resultTypes);
      fop.setType(newFuncType);
    });
  }

  void simplifyDefiningOp(Operation *oldOp) {
    Value result = oldOp->getResult(0);
    MemRefType resultType = result.getType().cast<MemRefType>();
    const SimplifiedShapeInfos resultSimplifiedInfos = getSimplifiedShapeInfos(resultType);
    MemRefType resultSimplifyType = resultSimplifiedInfos.first;
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - simplifyDefiningOp START\n";
      result.dump();
      resultType.dump();
      llvm::dbgs() << "\n";
      resultSimplifyType.dump();
      llvm::dbgs() << "\n";
      for (auto d : resultSimplifiedInfos.second) {
        llvm::dbgs() << d << " ";
      }
      llvm::dbgs() << "\n";
    });

    // Note: if simplifyValue call inside the following if
    //       issue can occur because not fully explore the use-def chains
    //       especially expand_shape that can introduce new shape of size 1
    simplifyValue(result, resultType, resultSimplifiedInfos);
    if (resultType != resultSimplifyType) {
      auto loc = oldOp->getLoc();
      mlir::OpBuilder builder(oldOp);
      if (mlir::memref::AllocOp allocop = dyn_cast<mlir::memref::AllocOp>(oldOp)) {
        mlir::memref::AllocOp newalloc = builder.create<mlir::memref::AllocOp>(
          loc, resultSimplifyType, allocop.getDynamicSizes(), allocop.getSymbolOperands(), allocop.getAlignmentAttr());
        allocop->replaceAllUsesWith(newalloc);
        allocop.erase();
      }
      if (mlir::memref::GetGlobalOp getglobalop = dyn_cast<mlir::memref::GetGlobalOp>(oldOp)) {
        mlir::memref::GetGlobalOp newop =
          builder.create<mlir::memref::GetGlobalOp>(loc, resultSimplifyType, getglobalop.getName());
        getglobalop->replaceAllUsesWith(newop);
        getglobalop.erase();
      }
    }
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - simplifyDefiningOp END\n";
    });
  }

  void simplifyAllocOpShape(mlir::ModuleOp m) {
    m.walk([&](mlir::memref::AllocOp allocop) { simplifyDefiningOp(allocop); });
  }

  void simplifyGlobalOps(mlir::ModuleOp m) {
    // First simplify the globalops definition in the symbol table
    m.walk([&](mlir::memref::GlobalOp globalop) {
      MemRefType resultType = globalop.getType().cast<MemRefType>();
      const SimplifiedShapeInfos resultSimplifiedInfos = getSimplifiedShapeInfos(resultType);
      MemRefType resultSimplifyType = resultSimplifiedInfos.first;

      if (resultType != resultSimplifyType) {
        Attribute initValue = globalop.getConstantInitValue();
        DenseElementsAttr elementsAttr = initValue.dyn_cast_or_null<DenseElementsAttr>();
        // Check if the global op is a constant
        if (elementsAttr) {
          Type simplifiedTensorType = mlir::memref::getTensorTypeFromMemRefType(resultSimplifyType);
          DenseElementsAttr reshapedElementsAttr = elementsAttr.reshape(simplifiedTensorType);
          SymbolTable symbolTable(m);

          auto loc = globalop.getLoc();
          mlir::OpBuilder builder(globalop);
          mlir::memref::GlobalOp newop =
            builder.create<mlir::memref::GlobalOp>(loc, globalop.getSymName(), builder.getStringAttr("private"),
                                                   resultSimplifyType, reshapedElementsAttr, true, IntegerAttr());

          symbolTable.erase(globalop);
          (void)symbolTable.insert(newop);
          newop->moveBefore(&m.front());
        } else {
          llvm::errs() << DEBUG_TYPE << " - Unkown initValue type cannot replace the GlobalOp";
        }
      }
    });

    // Now simplify uses of globalops within the function
    m.walk([&](mlir::memref::GetGlobalOp getglobalop) { simplifyDefiningOp(getglobalop); });
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    auto walkResult = m.walk([&](mlir::memref::ReshapeOp op) {
      LLVM_DEBUG({
        llvm::dbgs() << DEBUG_TYPE << " - DISABLE --simplify-shape pass. Don't treat memref.reshape op\n";
        op.dump();
      });
      return WalkResult::interrupt();
    });
    if (walkResult.wasInterrupted()) {
      return;
    }

    walkResult = m.walk([&](mlir::memref::SubViewOp op) {
      LLVM_DEBUG({
        llvm::dbgs() << DEBUG_TYPE << " - DISABLE --simplify-shape pass. Don't treat memref.subview op\n";
        op.dump();
      });
      return WalkResult::interrupt();
    });
    if (walkResult.wasInterrupted()) {
      return;
    }

    // Handle BlockArgument
    if (!keepArgsShape) {
      simplifyOpsUsingBlockArguments(m);
    } else {
      llvm::errs()
        << DEBUG_TYPE
        << " - BEAWARE: keepArgsShape not well manage especially when it implies a copy or interprocedural update\n";
    }

    // Handle GlobalOps
    simplifyGlobalOps(m);

    // Handle AllocOp
    simplifyAllocOpShape(m);
  }
};

}  // namespace mlir

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createSimplifyShapePass() {
  return std::make_unique<mlir::SimplifyShapePass>();
}

std::unique_ptr<OperationPass<mlir::ModuleOp>> mlir::createSimplifyShapePass(bool keepArg) {
  return std::make_unique<mlir::SimplifyShapePass>(keepArg);
}
