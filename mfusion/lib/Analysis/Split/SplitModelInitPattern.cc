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

#include "mfusion/Analysis/Split/SplitModelInitPattern.h"

#include <algorithm>
#include <numeric>
#include "llvm/Support/Debug.h"
#include "mfusion/Analysis/Split/FusePattern.h"
#include "mfusion/Analysis/Split/FuseTagBarrier.h"
#include "mfusion/Dialect/Mfuse/IR/Mfuse.h"

#define DEBUG_TYPE "split"

namespace mlir {
namespace mfuse {
namespace split {
namespace ascend {
constexpr size_t kReduceFusionDepth = 20;
constexpr size_t kBroadcastFusionDepth = 6;
constexpr size_t kReduceBwdMaxAreaSize = 10;
constexpr size_t kReduceBwdMaxAreaOutputs = 3;

// Fuse pattern for slice operations
class FuseSlice : public FusePattern {
 public:
  FuseSlice() : FusePattern("slice", FuseDirection::BACKWARD) {}
  ~FuseSlice() = default;

 protected:
  bool check(const AreaPtr &area) override {
    // Check if operation is Slice or StridedSlice
    if (!area->dom() || !area->dom()->op()) {
      return false;
    }
    auto op = area->dom()->op();
    return op->getName().getStringRef() == "mfuse.slice" || op->getName().getStringRef() == "mfuse.strided_slice";
  }
  bool match(const AreaPtr &area) override {
    for (const auto &[a, r] : area->usersWithRelation()) {
      if (a->pattern() < NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !hasCircle(area, a)) {
        fused_areas_.push_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

// Fuse pattern for element-wise operations
class FuseElemAny : public FusePattern {
 public:
  FuseElemAny() : FusePattern("elemany_addn") {}
  ~FuseElemAny() = default;

 protected:
  bool check(const AreaPtr &area) override {
    // Check if operation is ElemAny
    if (!area->dom() || !area->dom()->op()) {
      return false;
    }
    auto op = area->dom()->op();
    return op->getName().getStringRef() == "mfuse.elem_any";
  }
  bool match(const AreaPtr &area) override {
    for (const auto &[a, r] : area->inputsWithRelation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !hasCircle(area, a)) {
        fused_areas_.push_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};
}  // namespace ascend

namespace dvm {
// Fuse pattern for reduction forward pass
class FuseReduceFwd : public FusePattern {
 public:
  FuseReduceFwd(FuseType fuseType, size_t sizeLimit)
      : FusePattern("reduce_fwd", FuseDirection::FORWARD), fuseType_(fuseType), sizeLimit_(sizeLimit) {
    name_ += (fuseType == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseReduceFwd() = default;

  static std::shared_ptr<FuseReduceFwd> createDepthMatcher(size_t sizeLimit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kDepth, sizeLimit);
  }
  static std::shared_ptr<FuseReduceFwd> createWidthMatcher(size_t sizeLimit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kWidth, sizeLimit);
  }

 protected:
  bool check(const AreaPtr &area) override {
    if (area->pattern() != NodePattern::REDUCE) {
      return false;
    }
    return fuseType_ == FuseType::kWidth || area->inputNum() == 1;
  }
  bool match(const AreaPtr &area) override {
    for (const auto &[a, r] : area->inputsWithRelation()) {
      if (fuseType_ == FuseType::kDepth && a->userNum() != 1) {
        continue;
      }
      if (a->size() > sizeLimit_) {
        continue;
      }
      if (a->pattern() <= NodePattern::BROADCAST) {
        if (r != EdgeRelation::INJECTIVE && (a->userNum() != 1 || a->isOutput())) {
          continue;
        }
        if (fuseType_ == FuseType::kWidth && hasCircle(area, a)) {
          continue;
        }
        fused_areas_.emplace_back(a);
      }
    }
    return !fused_areas_.empty();
  }

 private:
  FuseType fuseType_;
  size_t sizeLimit_;
};

// Fuse pattern for reduction backward pass
class FuseReduceBwd : public FusePattern {
 public:
  FuseReduceBwd() : FusePattern("reduce_bwd", FuseDirection::BACKWARD) {}
  ~FuseReduceBwd() = default;

 protected:
  static bool CheckReduceArea(const AreaPtr &area) {
    if (!area || !area->isAlive() || area->pattern() != NodePattern::REDUCE || !area->dom() || !area->dom()->op()) {
      return false;
    }

    auto reduce = mlir::dyn_cast<mlir::mfuse::ReduceSumOp>(area->dom()->op());
    if (!reduce) {
      return false;
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(reduce.getInput().getType());
    if (!inputType || !inputType.hasRank()) {
      return false;
    }

    auto dimensions = reduce.getDimensions();
    if (dimensions.empty()) {
      return false;
    }

    const int64_t rank = inputType.getRank();
    std::vector<int64_t> dims;
    dims.reserve(dimensions.size());
    for (auto dimAttr : dimensions.getValue()) {
      auto dim = mlir::cast<mlir::IntegerAttr>(dimAttr).getValue().getSExtValue();
      if (dim < 0 || dim >= rank) {
        return false;
      }
      dims.push_back(dim);
    }

    std::sort(dims.begin(), dims.end());
    return std::adjacent_find(dims.begin(), dims.end()) == dims.end();
  }

  bool check(const AreaPtr &area) override {
    // Match a single-user reduce neighborhood so the reduce can sink
    // into its post-reduce pointwise/broadcast area.
    return area->userNum() == 1 && CheckReduceArea(area);
  }

  static bool CheckPostReduceArea(const AreaPtr &area, EdgeRelation relation) {
    if (!area || !area->isAlive()) {
      return false;
    }
    if (relation != EdgeRelation::INJECTIVE && relation != EdgeRelation::BROADCAST) {
      return false;
    }
    if (area->pattern() != NodePattern::RESHAPE && area->pattern() != NodePattern::ELEMWISE &&
        area->pattern() != NodePattern::BROADCAST) {
      return false;
    }
    if (area->areaOutputs().size() > ascend::kReduceBwdMaxAreaOutputs) {
      return false;
    }
    return area->size() <= ascend::kReduceBwdMaxAreaSize;
  }

  // Check if the area is a terminal single reshape area. If this area is fused, an extra copy would be inserted by DVM.
  static bool IsTerminalSingleReshapeArea(const AreaPtr &area) {
    static constexpr llvm::StringLiteral kReshapeOpName = "mfuse.reshape";
    return area && area->size() == 1 && area->userNum() == 0 && area->dom() && area->dom()->op() &&
           area->dom()->op()->getName().getStringRef() == kReshapeOpName;
  }

  bool match(const AreaPtr &area) override {
    for (const auto &[a, r] : area->usersWithRelation()) {
      if (IsTerminalSingleReshapeArea(a) || hasCircle(area, a) || !CheckPostReduceArea(a, r)) {
        continue;
      }
      fused_areas_.push_back(a);
    }
    return fused_areas_.size() == 1;
  }
};

// Fuse pattern for matmul operations
class FuseMatMul : public FusePattern {
 public:
  FuseMatMul() : FusePattern("matmul_depth", FuseDirection::BACKWARD) {}
  ~FuseMatMul() = default;

 protected:
  bool check(const AreaPtr &area) override {
    if (area->size() == 1) {
      if (!area->dom() || !area->dom()->op()) {
        return false;
      }
      auto op = area->dom()->op();
      auto opName = op->getName().getStringRef();
      return opName == "mfuse.matmul";
    }
    // To Check if operation is GroupedMatmul.
    return false;
  }

  bool isSameShapeSize(int64_t size, const std::vector<Node *> &output_nodes) {
    for (auto &node : output_nodes) {
      if (std::accumulate(mlir::cast<mlir::ShapedType>(node->op()->getResult(0).getType()).getShape().begin(),
                          mlir::cast<mlir::ShapedType>(node->op()->getResult(0).getType()).getShape().end(),
                          static_cast<int64_t>(1), std::multiplies<int64_t>()) != size) {
        return false;
      }
    }
    return true;
  }

  bool match(const AreaPtr &dom) override {
    constexpr size_t MAX_FUSE_NUM = 5;
    size_t current_size = 0;
    if (dom->nodes().empty()) {
      return false;
    }
    auto output_op = dom->nodes().back()->op();
    if (!output_op) {
      return false;
    }
    auto output_shape = mlir::cast<mlir::ShapedType>(output_op->getResult(0).getType()).getShape();
    int64_t matmul_output_size =
      std::accumulate(output_shape.begin(), output_shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
    if (output_shape.back() == 1) {
      return false;
    }
    auto opName = output_op->getName().getStringRef();

    auto users = dom->users();
    std::sort(users.begin(), users.end(),
              [](const AreaPtr &a, const AreaPtr &b) { return a->areaOutputs().size() < b->areaOutputs().size(); });
    for (auto &a : users) {
      if (current_size + a->areaOutputs().size() > MAX_FUSE_NUM) {
        break;
      }
      if (a->size() == 1 && a->dom()->op()->getName().getStringRef() == "mfuse.reshape") {
        continue;
      }
      bool fuse_flag = (opName == "mfuse.matmul" || opName == "mfuse.grouped_matmul");
      if (std::any_of(a->nodes().begin(), a->nodes().end(),
                      [](const Node *node) { return node->op()->getName().getStringRef() == "mfuse.reshape"; })) {
        fuse_flag = fuse_flag && (a->pattern() < NodePattern::BROADCAST);
      } else {
        fuse_flag = fuse_flag && (a->pattern() <= NodePattern::BROADCAST);
      }
      if (fuse_flag && !hasCircle(dom, a) && isSameShapeSize(matmul_output_size, a->areaOutputs())) {
        fused_areas_.push_back(a);
        current_size += a->areaOutputs().size();
      }
    }
    return !fused_areas_.empty();
  }
};

// Fuse pattern for AllReduce forward pass
class FuseAllReduceFwd : public FusePattern {
 public:
  FuseAllReduceFwd() : FusePattern("allreduce_fwd", FuseDirection::FORWARD) {}
  ~FuseAllReduceFwd() = default;

 protected:
  bool check(const AreaPtr &area) override {
    if (!area->dom() || !area->dom()->op()) {
      return false;
    }
    auto op = area->dom()->op();
    return area->size() == 1 && op->getName().getStringRef() == "mfuse.allreduce";
  }

  bool match(const AreaPtr &area) override {
    for (const auto &[a, r] : area->inputsWithRelation()) {
      if (a->userNum() != 1) {
        continue;
      }
      auto op = a->dom()->op();
      if (!hasCircle(a, area) && r == EdgeRelation::INJECTIVE && a->size() == 1 && op &&
          op->getName().getStringRef() == "mfuse.matmul") {
        fused_areas_.push_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

// Fuse pattern for AllReduce backward pass
class FuseAllReduceBwd : public FusePattern {
 public:
  FuseAllReduceBwd() : FusePattern("allreduce_bwd", FuseDirection::BACKWARD) {}
  ~FuseAllReduceBwd() = default;

 protected:
  bool check(const AreaPtr &area) override {
    auto nodes = area->nodes();
    return std::any_of(nodes.begin(), nodes.end(), [](const auto &node) {
      return node->op() && node->op()->getName().getStringRef() == "mfuse.allreduce";
    });
  }

  bool match(const AreaPtr &area) override {
    for (const auto &a : area->users()) {
      if (a->pattern() < NodePattern::BROADCAST && !hasCircle(area, a)) {
        fused_areas_.push_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

// Fuse pattern for virtual nodes
class FuseVirtualNode : public FusePattern {
 public:
  FuseVirtualNode() : FusePattern("fuse_virtual_node", FuseDirection::FORWARD) {}
  ~FuseVirtualNode() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->pattern() == NodePattern::VIRTUAL; }
  bool match(const AreaPtr &area) override {
    for (const auto &inp : area->inputs()) {
      if (inp->dom() && inp->dom()->op() && inp->dom()->op()->getName().getStringRef() != "mfuse.permute") {
        fused_areas_.push_back(inp);
      }
    }
    return !fused_areas_.empty();
  }
};

}  // namespace dvm

// Initialize fuse patterns for DVM
void DVMSplitModel::initFusePatterns() {
  // Add DVM-specific fuse patterns
  addPattern(std::make_shared<dvm::FuseVirtualNode>(), true);
  addPattern(std::make_shared<FuseReshape>(), true);
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
  addPattern(dvm::FuseReduceFwd::createDepthMatcher(ascend::kReduceFusionDepth), true);
  addPattern(dvm::FuseReduceFwd::createWidthMatcher(ascend::kReduceFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createDepthMatcher(ascend::kBroadcastFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createWidthMatcher(ascend::kBroadcastFusionDepth), true);
  addPattern(std::make_shared<dvm::FuseReduceBwd>(), true);
  addPattern(createDvmFuseGroupTagBarrier(FuseDirection::FORWARD), true);
  addPattern(createDvmFuseGroupTagBarrier(FuseDirection::BACKWARD), true);
  addPattern(std::make_shared<ascend::FuseElemAny>(), true);
  addPattern(std::make_shared<ascend::FuseSlice>(), true);
  // To be enabled by config in the future.
  addPattern(std::make_shared<dvm::FuseMatMul>(), true);
  addPattern(std::make_shared<dvm::FuseAllReduceFwd>(), true);
  addPattern(std::make_shared<dvm::FuseAllReduceBwd>(), true);
}

void AKGSplitModel::initFusePatterns() {
  addPattern(std::make_shared<dvm::FuseVirtualNode>(), true);
  addPattern(std::make_shared<FuseReshape>(), true);
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
  addPattern(FuseElemwiseBroadcastBwd::createDepthMatcher(ascend::kBroadcastFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createWidthMatcher(ascend::kBroadcastFusionDepth), true);
}

void BishengSplitModel::initFusePatterns() {
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
}

}  // namespace split
}  // namespace mfuse
}  // namespace mlir
