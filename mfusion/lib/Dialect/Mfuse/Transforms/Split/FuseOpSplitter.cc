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

#include "mfusion/Dialect/Mfuse/Transforms/Split/FuseOpSplitter.h"

#include "mfusion/Analysis/Split/Area.h"
#include "mfusion/Analysis/Split/SplitModel.h"
#include "mfusion/Analysis/Split/SplitModelFactory.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/FuseOpRebuilder.h"
#include "mfusion/Dialect/Mfuse/Transforms/Split/SplitSchemer.h"
#include "mfusion/Dialect/Mfuse/Mfuse.h"
#include "mfusion/Dialect/Mfuse/MfuseDialect.h"
#include "mfusion/Dialect/Mfuse/Utils/OpConstants.h"
#include "mfusion/Dialect/Dvm/Dvm.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace mfuse {
namespace split {

// CppCostModelSplitSchemer uses cost model to split operations
class CppCostModelSplitSchemer : public SplitSchemer {
 public:
  explicit CppCostModelSplitSchemer(const std::string &processor) : processor_(processor) {}
  ~CppCostModelSplitSchemer() = default;

  bool split(Block *block) override {
    if (!splitByCostModel(block)) {
      return false;
    }
    return true;
  }

 protected:
  bool splitByCostModel(Block *block) {
    // Map from Operation* to its index in the block for sorting
    mlir::DenseMap<mlir::Operation *, size_t> op_idx_map;
    mlir::SmallVector<mlir::Operation *> ops;
    size_t idx = 0;
    for (auto &op : block->getOperations()) {
      op_idx_map[&op] = idx++;
      ops.push_back(&op);
    }

    // Create split model
    auto model = SplitModelFactory::Instance().createSplitModel(processor_);
    // Run the model on the operations
    model->run(block);
    // Get areas from the model
    auto &areas = model->areas();
    LLVM_DEBUG(llvm::dbgs() << "Total areas size: " << areas.size() << "\n");
    // Process each area to build the split plan
    for (auto &area : areas) {
      mlir::SmallVector<mlir::Operation *> area_ops;
      LLVM_DEBUG(llvm::dbgs() << "Current area ops size: " << area->ops().size() << "\n");
      for (auto &op : area->ops()) {
        area_ops.push_back(op);
        LLVM_DEBUG(llvm::dbgs() << "Current op: " << *op << "\n");
      }
      // Sort operations by their original index in the block
      std::sort(area_ops.begin(), area_ops.end(),
                [&op_idx_map](mlir::Operation *a, mlir::Operation *b) { return op_idx_map[a] < op_idx_map[b]; });
      // Add to split plan
      split_plan_.push_back(std::move(area_ops));
      // Determine if this area needs inline
      need_inline_.push_back((area->mode() == AreaMode::BASIC ? 1 : 0));
    }

    // Check if splitting is needed
    return split_plan_.size() > 1 || (split_plan_.size() == 1 && needInline(0));
  }

  std::string processor_;
};

// Splitter splits graph kernel operations for Ascend processor
class Splitter {
 public:
  using SplitterPtr = std::shared_ptr<Splitter>;

  // Split the graph
  bool split() {
    genParamMap();
    Region &region = fuseOp_->getRegion(0);
    Block *block = &region.front();
    if (!splitSchemer_->split(block)) {
      return false;
    }
    return rebuildGraph();
  }

  // Create a splitter
  static SplitterPtr makeSplitter(mlir::mfuse::FusedOp fuseOp, const SplitSchemerPtr &splitSchemer) {
    if (!fuseOp || !splitSchemer) {
      return nullptr;
    }
    return std::make_shared<Splitter>(fuseOp, splitSchemer);
  }

  // Constructor
  Splitter(mlir::mfuse::FusedOp fuseOp, const SplitSchemerPtr &splitSchemer)
      : fuseOp_(fuseOp), oldSubgraphOp_(fuseOp.getOperation()), splitSchemer_(splitSchemer) {}

  ~Splitter() = default;

 private:
  bool rebuildGraph() {
    Rebuilder rebuilder(fuseOp_, splitSchemer_, paramToMainGraphValueMap_);
    rebuilder.rebuild();
    return true;
  }

  // Generate parameter map, reference MindSpore GraphKernel::GenParamMap
  void genParamMap() {
    if (!fuseOp_) {
      return;
    }

    Block *body = &fuseOp_.getBodyBlock();
    for (auto it : llvm::enumerate(body->getArguments())) {
      BlockArgument arg = it.value();
      size_t idx = it.index();
      if (idx < fuseOp_.getNumOperands()) {
        paramToMainGraphValueMap_[arg] = fuseOp_.getOperand(idx);
      }
    }
  }

  mlir::mfuse::FusedOp fuseOp_;
  Operation *oldSubgraphOp_;
  SplitSchemerPtr splitSchemer_;
  DenseMap<Value, Value> paramToMainGraphValueMap_;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

mlir::mfuse::split::SplitSchemerPtr FuseOpSplitter::getSplitSchema(const std::string &processor) {
  // Default split schemer using CppCostModel
  return std::make_shared<mlir::mfuse::split::CppCostModelSplitSchemer>(processor);
}

bool FuseOpSplitter::trySplit(mlir::mfuse::FusedOp op) {
  // Only support DVM processor for now
  auto schm = getSplitSchema(mlir::mfuse::kProcessorDVM);
  auto splitter = mlir::mfuse::split::Splitter::makeSplitter(op, schm);
  bool result = splitter->split();
  return result;
}
