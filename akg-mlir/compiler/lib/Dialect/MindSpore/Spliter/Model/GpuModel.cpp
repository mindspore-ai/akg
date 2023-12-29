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
#include "akg/Dialect/MindSpore/Spliter/GpuModel.h"

#include <memory>
#include <string>

namespace mlir::spliter {
namespace gpu {
constexpr size_t kReduceFusionDepth = 10;
constexpr size_t kBroadcastFusionDepth = 6;
constexpr size_t kReduceLineNums = 12;

class FuseReduceBwd : public FusePattern {
 public:
  FuseReduceBwd() : FusePattern("reduce_bwd") { direction = FuseDirection::BACKWARD; }
  ~FuseReduceBwd() = default;

 protected:
  bool check(const AreaPtr &dom) override { return dom->isAlive() && dom->pattern() == NodePattern::REDUCE; }
  bool match(const AreaPtr &dom) override {
    auto opAttrs = dom->dom()->getAttrs();
    if (!opAttrs.get("reduce_output_fuse")) {
      return false;
    }
    bool isAllReduce = (!dom->dom()->hasSymbolicShape() && (dom->dom()->tensorSize() == 1));
    if (isAllReduce && dom->getInput(0)->computeSize() > ReduceSizeLine * (int32_t)kReduceLineNums) {
      return false;
    }
    for (auto &[a, r] : dom->usersWithRelation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r <= EdgeRelation::BROADCAST && !hasCircle(dom, a)) {
        (void)fusedAreas.emplace_back(a);
      }
    }
    return !fusedAreas.empty();
  }
  const int32_t ReduceSizeLine = 1024;
};
}  // namespace gpu

void GpuModel::initFusePatterns() {
  addPattern(std::make_shared<FuseVirtualNode>(), true);
  addPattern(std::make_shared<FuseReshape>(), true);
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
  addPattern(FuseReduceFwd::createDepthMatcher(gpu::kReduceFusionDepth), true);
  addPattern(FuseReduceFwd::createWidthMatcher(gpu::kReduceFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createDepthMatcher(gpu::kBroadcastFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createWidthMatcher(gpu::kBroadcastFusionDepth), true);
  addPattern(std::make_shared<gpu::FuseReduceBwd>(), true);
}

AreaMode GpuModel::getDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr && node->getOp() == kReshapeOpName) {
    return AreaMode::BASIC;
  }
  return AreaMode::COMPOSITE;
}
}  // namespace mlir::spliter
