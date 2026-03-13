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

#define DEBUG_TYPE "split"

namespace mlir {
namespace mfuse {
namespace split {
namespace ascend {
constexpr size_t kReduceFusionDepth = 10;
constexpr size_t kBroadcastFusionDepth = 6;
// Fuse pattern for reduction backward pass
class FuseReduceBwd : public FusePattern {
 public:
  FuseReduceBwd() : FusePattern("reduce_bwd", FuseDirection::BACKWARD) {}
  ~FuseReduceBwd() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->isAlive() && area->pattern() == NodePattern::REDUCE; }
  bool match(const AreaPtr &area) override {
    // Check if operation has reduce_output_fuse attribute
    if (!area->dom() || !area->dom()->op()) {
      return false;
    }
    Operation *op = area->dom()->op();
    if (!op->hasAttr("reduce_output_fuse")) {
      return false;
    }
    // Continue with existing matching logic
    for (const auto &[a, r] : area->usersWithRelation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !hasCircle(area, a)) {
        fused_areas_.push_back(a);
      }
    }
    return !fused_areas_.empty();
  }
};

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
      return opName == "mfuse.matmul" || opName == "mfuse.batch_matmul";
    }
    // TODO: Check if operation is GroupedMatmul
    return false;
  }

  bool isSameShapeSize(int64_t size, const std::vector<Node *> &output_nodes) {
    for (auto &node : output_nodes) {
      if (std::accumulate(node->op()->getResult(0).getType().cast<mlir::ShapedType>().getShape().begin(),
                          node->op()->getResult(0).getType().cast<mlir::ShapedType>().getShape().end(),
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
    auto output_shape = output_op->getResult(0).getType().cast<mlir::ShapedType>().getShape();
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
      bool fuse_flag = (opName == "mfuse.matmul" || opName == "mfuse.batch_matmul" || opName == "mfuse.grouped_matmul");
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
    static constexpr llvm::StringLiteral kAllReduceOpName = "mfuse.allreduce";
    auto nodes = area->nodes();
    return std::any_of(nodes.begin(), nodes.end(), [](const auto &node) {
      return node->op() && node->op()->getName().getStringRef() == kAllReduceOpName;
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
  addPattern(std::make_shared<ascend::FuseElemAny>(), true);
  addPattern(std::make_shared<ascend::FuseSlice>(), true);
  // TODO: Need enabled by config
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
