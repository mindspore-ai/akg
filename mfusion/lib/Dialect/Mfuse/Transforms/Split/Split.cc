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

#include "mfusion/Dialect/Mfuse/Transforms/Split/Split.h"

#include "mfusion/Analysis/Split/SplitModel.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/FuseOpSplitter.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Dialect/Dvm/Dvm.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "split"

namespace mlir {
namespace mfuse {

#define GEN_PASS_DEF_SPLIT
#include "mfusion/Dialect/Mfuse/Transforms/Passes.h.inc"

struct SplitPass : public impl::SplitBase<SplitPass> {
  using SplitBase::SplitBase;

  void runOnOperation() override {
    func::FuncOp func_op = getOperation();
    SmallVector<FusedOp> fuseOps;
    func_op.walk([&](FusedOp fuseOp) { fuseOps.push_back(fuseOp); });

    FuseOpSplitter fuse_op_splitter;
    for (FusedOp fuseOp : fuseOps) {
      if (fuse_op_splitter.trySplit(fuseOp)) {
        LLVM_DEBUG(llvm::dbgs() << "Split fuseOp: " << fuseOp << "\n");
      }
    }
  }
};

}  // namespace mfuse

std::unique_ptr<Pass> createSplitPass() { return std::make_unique<mfuse::SplitPass>(); }

}  // namespace mlir
