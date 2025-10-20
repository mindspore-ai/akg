/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "akg/Dialect/Linalg/Transforms/MemrefCopyToLoops.h"
#include "akg/Utils/AnalysisCommon.hpp"

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
#define GEN_PASS_DECL_MEMREFCOPYTOLOOPS
#define GEN_PASS_DEF_MEMREFCOPYTOLOOPS
#include "akg/Dialect/Linalg/Passes.h.inc"
}  // namespace mlir

using namespace mlir;

namespace mlir {
namespace linalg {
namespace {

struct MemrefCopyToLoops : public impl::MemrefCopyToLoopsBase<MemrefCopyToLoops> {
  public:
    MemrefCopyToLoops() = default;

    void runOnOperation() override{
        func::FuncOp funcOp = getOperation();

        SmallVector<memref::CopyOp> needConvert;
        (void)funcOp->walk([&](memref::CopyOp copyOp) {
          auto srcOp = copyOp.getSource().getDefiningOp();
          if (!srcOp){
            return;
          }

          if (auto tomem = dyn_cast<bufferization::ToMemrefOp>(srcOp)){
            auto mem = tomem.getMemref();
            auto totensor = tomem.getTensor().getDefiningOp();
            if (!totensor){
                return;
            }
            if (auto tt = dyn_cast<bufferization::ToTensorOp>(totensor)) {
              auto tensormem = tt.getMemref().getDefiningOp();
              if (!tensormem){
                return;
              }
              if (isa<memref::ExpandShapeOp, memref::CollapseShapeOp>(tensormem)) {
                needConvert.emplace_back(copyOp);
              }
            }
          }

        });
        OpBuilder builder(funcOp);

        for (auto copyOp : needConvert) {
          builder.setInsertionPoint(copyOp);
          auto newCopyOp = makeMemRefCopyOp(builder, copyOp->getLoc(), copyOp.getSource(), copyOp.getTarget());
          copyOp.getOperation()->replaceAllUsesWith(newCopyOp.getOperation());
          copyOp.erase();
        }
    }
};
}      // namespace
}      // namespace linalg
}      // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> mlir::createMemrefCopyToLoopsPass() {
    return std::make_unique<linalg::MemrefCopyToLoops>();
}


