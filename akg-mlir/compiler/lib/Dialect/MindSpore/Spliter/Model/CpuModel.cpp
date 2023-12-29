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
#include "akg/Dialect/MindSpore/Spliter/CpuModel.h"

#include <memory>

namespace mlir::spliter {

void initCpuModel() {
  static bool initAllModel = false;
  if (!initAllModel) {
    initAllModel = true;
  }
}

constexpr size_t kReduceFusionDepth = 20;
constexpr size_t kBroadcastFusionDepth = 20;

class FuseElemwiseFwd : public FusePattern {
 public:
  explicit FuseElemwiseFwd(FuseType newFuseType) : FusePattern("cpu_elemwise_fwd"), fuseType(newFuseType) {
    direction = FuseDirection::FORWARD;
    name += (fuseType == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseFwd() = default;
  static FusePatternPtr createDepthMatcher() { return std::make_shared<FuseElemwiseFwd>(FuseType::kDepth); }
  static FusePatternPtr createWidthMatcher() { return std::make_shared<FuseElemwiseFwd>(FuseType::kWidth); }

 protected:
  bool check(const AreaPtr &dom) override {
    if (dom->pattern() != NodePattern::ELEMWISE) {
      return false;
    }
    return fuseType == FuseType::kWidth || dom->inputNum() == 1;
  }
  bool match(const AreaPtr &dom) override {
    for (auto &[a, r] : dom->getInputsWithRelation()) {
      // depth match only support one to one pattern
      if (fuseType == FuseType::kDepth && a->userNum() != 1) {
        continue;
      }
      if (a->pattern() <= NodePattern::ELEMWISE && r == EdgeRelation::INJECTIVE) {
        // it's unnecessary to check circle for depth match
        if (fuseType == FuseType::kWidth && hasCircle(a, dom)) {
          continue;
        }

        if (a->hasSameComputeSize(dom)) {
          (void)fusedAreas.emplace_back(a);
        }
      }
    }
    return !fusedAreas.empty();
  }

  FuseType fuseType;
};

class FuseConv : public FusePattern {
 public:
  FuseConv() : FusePattern("conv") { direction = FuseDirection::BACKWARD; }
  ~FuseConv() = default;

 protected:
  bool check(const AreaPtr &dom) override {
    if (dom->dom()->getOp() != "Conv2D") {
      return false;
    }
    return true;
  }
  bool match(const AreaPtr &dom) override {
    for (auto d : dom->usersWithRelation()) {
      auto a = d.first;
      if (hasCircle(dom, a)) {
        continue;
      }
      if (a->pattern() < NodePattern::BROADCAST ||
          (a->pattern() == NodePattern::BROADCAST && dom->dom()->hasSameShape(a->dom()))) {
        (void)fusedAreas.emplace_back(a);
      }
    }
    return !fusedAreas.empty();
  }
};

void CpuModel::initFusePatterns() {
  addPattern(std::make_shared<FuseVirtualNode>(), true);
  addPattern(std::make_shared<FuseReshape>(), true);
  addPattern(FuseElemwiseFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseFwd::createWidthMatcher(), true);
  addPattern(std::make_shared<FuseConv>(), true);
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
  addPattern(FuseReduceFwd::createDepthMatcher(kReduceFusionDepth), true);
  addPattern(FuseReduceFwd::createWidthMatcher(kReduceFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createDepthMatcher(kBroadcastFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createWidthMatcher(kBroadcastFusionDepth), true);
  addPattern(std::make_shared<FuseIsolateReshape>(), true);
}

AreaMode CpuModel::getDefaultAreaMode(const PrimOpPtr &) const { return AreaMode::COMPOSITE; }
}  // namespace mlir::spliter
