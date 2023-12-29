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

// ===- WorkaroundFixReduceInitialization.cpp -          -------------------=== //
//
// Workaround to move the initialization of the reduce variable, near the usage and the reduction of it.
// The initialization must be inside a 'if'.
// This pass is present because else we will have an issue during the memory-promotion pass that fail
// when the initialization is not near the usage.
// When fix memory-promotion, this pass may disapear. But it may require more work to do that this pass.
//
// ===----------------------------------------------------------------------=== //

#include "akg/Dialect/Affine/Transforms/WorkaroundFixReduceInitialization.h"

namespace mlir {
#ifndef GEN_PASS_DEF_WORKAROUNDFIXREDUCEINITIALIZATION
#define GEN_PASS_DEF_WORKAROUNDFIXREDUCEINITIALIZATION
#include "akg/Dialect/Affine/Passes.h.inc"
#endif
}  // namespace mlir

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/Support/Debug.h"

#include "akg/Utils/AnalysisCommon.hpp"

using namespace mlir;

#define DEBUG_TYPE "fix-reduce-initialization"

namespace mlir {

static constexpr const int kExpectedNbStore = 2;
static constexpr const int kExpectedNBUsage = 3;

class WorkaroundFixReduceInitializationPass
    : public mlir::impl::WorkaroundFixReduceInitializationBase<WorkaroundFixReduceInitializationPass> {
 public:
  WorkaroundFixReduceInitializationPass() = default;
  WorkaroundFixReduceInitializationPass(const WorkaroundFixReduceInitializationPass &pass) = default;

 private:
  void findReduceValue(func::FuncOp fop);
  void moveInitializationOp();

 public:
  void runOnOperation() override;

 private:
  /// The first Operation of the vector is expected to be store for initialization
  /// The second Operation of the vector is expected to be store for reduction
  /// The third Operation of the vector is expected to be load for reduction
  llvm::MapVector<mlir::Value, SmallVector<mlir::Operation *, kExpectedNBUsage>> reduceValue;
};

/// If the ReduceValue has not exactly 3 usage, 2 store and 1 load,
/// findReduceValue will not find it as a ReduceValue...
void WorkaroundFixReduceInitializationPass::findReduceValue(func::FuncOp funcOp) {
  llvm::MapVector<mlir::Value, SmallVector<mlir::Operation *, kExpectedNBUsage>> reduceCandidate;

  funcOp.walk([&](AffineStoreOp storeOp) {
    mlir::Value candidate = storeOp.getMemref();
    /// if (reduceCandidate.contains(candidate)) { // contains doesn't exist in LLVM 16.0.6
    if (reduceCandidate.count(candidate) != 0) {
      reduceCandidate[candidate].push_back(storeOp);
    } else {
      reduceCandidate[candidate] = {storeOp};
    }
  });

  LLVM_DEBUG({
    llvm::dbgs() << DEBUG_TYPE << " - reduceCandidate Value:\n";
    for (auto c : reduceCandidate) {
      c.first.dump();
      for (auto o : c.second) {
        o->dump();
      }
      llvm::dbgs() << "\n";
    }
  });

  for (auto [candidateValue, candidateOps] : reduceCandidate) {
    if (candidateOps.size() == kExpectedNbStore) {
      int nbUser = 0;
      if (isa<mlir::AffineIfOp>(candidateOps[0]->getParentOp())) {
        this->reduceValue[candidateValue] = candidateOps;
        for (Operation *userOp : candidateValue.getUsers()) {
          nbUser++;
          if (isa<mlir::AffineLoadOp>(userOp)) {
            this->reduceValue[candidateValue].push_back(userOp);
          } else if (isa<mlir::memref::CopyOp>(userOp)) {
            // Do not count memref.copy statement...
            nbUser--;
          }
        }
      }
      if (this->reduceValue[candidateValue].size() != kExpectedNBUsage || nbUser != kExpectedNBUsage) {
        LLVM_DEBUG({
          llvm::dbgs() << DEBUG_TYPE
                       << " - Remove candidate Value from ReduceValue because cannot find LoadOp or find too many "
                          "LoadOp to determine the right one:\n";
          candidateValue.dump();
        });
        (void) this->reduceValue.erase(candidateValue);
      }
    }
  }

  LLVM_DEBUG({
    llvm::dbgs() << DEBUG_TYPE << " - reduceValue:\n";
    for (auto c : this->reduceValue) {
      c.first.dump();
      for (auto o : c.second) {
        o->dump();
      }
    }
  });
}

static constexpr const int kDestination = 2;

void WorkaroundFixReduceInitializationPass::moveInitializationOp() {
  for (auto r : this->reduceValue) {
    mlir::Operation *initOp = r.second[0]->getParentOp();
    initOp->moveBefore(r.second[kDestination]);
  }
}

void WorkaroundFixReduceInitializationPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp->hasAttr(kOperatorTypeStr) && funcOp->getAttr(kOperatorTypeStr).dyn_cast<StringAttr>() == kReduceStr) {
    LLVM_DEBUG({
      llvm::dbgs() << DEBUG_TYPE << " - reduce FuncOp\n";
    });
    findReduceValue(funcOp);
    moveInitializationOp();
  }
  return;
}

}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createWorkaroundFixReduceInitializationPass() {
  return std::make_unique<mlir::WorkaroundFixReduceInitializationPass>();
}
