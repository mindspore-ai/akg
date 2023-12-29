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
#include "akg/Dialect/MindSpore/Spliter/AscendModel.h"

#include <memory>
#include <string>

namespace mlir::spliter {
namespace ascend {
constexpr size_t kReduceFusionDepth = 10;
constexpr size_t kBroadcastFusionDepth = 6;

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
    for (auto &[a, r] : dom->usersWithRelation()) {
      if (a->pattern() <= NodePattern::BROADCAST && r == EdgeRelation::INJECTIVE && !hasCircle(dom, a)) {
        (void)fusedAreas.emplace_back(a);
      }
    }
    return !fusedAreas.empty();
  }
};

class FuseMatMul : public FusePattern {
 public:
  FuseMatMul() : FusePattern("matmul_depth") {}
  ~FuseMatMul() = default;

 protected:
  bool check(const AreaPtr &dom) override {
    return dom->isAlive() && (dom->dom()->getOp() == kMatMulOpName || dom->dom()->getOp() == kBatchMatMulOpName);
  }
  bool match(const AreaPtr &dom) override {
    auto domName = dom->dom()->getOp();
    for (auto &a : dom->getUsers()) {
      if (!a->isAlive()) {
        continue;
      }
      auto userName = a->dom()->getOp();
      if (((domName == kMatMulOpName &&
            (userName == kAddNOpName || userName == kTensorAddOpName || userName == kCastOpName)) ||
           (domName == kBatchMatMulOpName && a->pattern() == NodePattern::ELEMWISE)) &&
          !hasCircle(dom, a)) {
        (void)fusedAreas.emplace_back(a);
      }
    }
    direction = FuseDirection::BACKWARD;
    return !fusedAreas.empty();
  }
};

class FuseTransdata : public FusePattern {
 public:
  FuseTransdata() : FusePattern("transdata") {}
  ~FuseTransdata() = default;

 protected:
  bool check(const AreaPtr &dom) override { return dom->isAlive() && dom->dom()->getOp() == kTransDataOpName; }
  bool match(const AreaPtr &dom) override {
    for (auto &a : dom->getInputs()) {
      if (a->isAlive() && supported(dom, a) && !hasCircle(a, dom)) {
        (void)fusedAreas.emplace_back(a);
      }
    }
    return !fusedAreas.empty();
  }

 private:
  bool needPad(const DShape &inShape, const DShape &outShape) const {
    const size_t minRank = 2;
    const int64_t blockSz = 16;
    return !(inShape.size() >= minRank && outShape.size() >= minRank && inShape[inShape.size() - 1] == blockSz &&
             inShape[inShape.size() - 1] == blockSz && outShape[outShape.size() - 2] == blockSz &&
             outShape[outShape.size() - 2] == blockSz);
  }
  bool supported(const AreaPtr &dom, const AreaPtr &a) const {
    if (dom->size() != 1 || dom->dom()->getInputs().empty() ||
        needPad(dom->dom()->getInput(0)->shape, dom->dom()->shape)) {
      return false;
    }
    if (a->dom()->getOp() == kMatMulOpName) {
      return true;
    }
    if (a->pattern() > NodePattern::BROADCAST) {
      return false;
    }
    auto opAttrs = dom->dom()->getAttrs();
    if (opAttrs.get(kAttrSrcFormat) || opAttrs.get(kAttrDstFormat)) {
      llvm::errs() << "For '" << dom->dom()->getOp() << "', can not find the attr '" << kAttrSrcFormat << "' or '"
                   << kAttrDstFormat << "'\n";
      return false;
    }
    auto srcFormat = opAttrs.get(kAttrSrcFormat).cast<StringAttr>().str();
    auto dstFormat = opAttrs.get(kAttrDstFormat).cast<StringAttr>().str();
    if (srcFormat == kOpFormat_FRAC_NZ && (dstFormat == kOpFormat_DEFAULT || dstFormat == kOpFormat_NCHW)) {
      return true;
    }
    return (srcFormat == kOpFormat_DEFAULT || srcFormat == kOpFormat_NCHW) && dstFormat == kOpFormat_FRAC_NZ &&
           a->size() == 1 && a->dom()->getOp() == kCastOpName && !a->OutputJudge();
  }
};
}  // namespace ascend

void AscendModel::initFusePatterns() {
  addPattern(std::make_shared<FuseVirtualNode>(), true);
  addPattern(std::make_shared<FuseReshape>(), true);
  addPattern(FuseElemwiseBroadcastFwd::createDepthMatcher(), true);
  addPattern(FuseElemwiseBroadcastFwd::createWidthMatcher(), true);
  addPattern(FuseReduceFwd::createDepthMatcher(ascend::kReduceFusionDepth), true);
  addPattern(FuseReduceFwd::createWidthMatcher(ascend::kReduceFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createDepthMatcher(ascend::kBroadcastFusionDepth), true);
  addPattern(FuseElemwiseBroadcastBwd::createWidthMatcher(ascend::kBroadcastFusionDepth), true);
  addPattern(std::make_shared<ascend::FuseMatMul>(), true);
  addPattern(std::make_shared<ascend::FuseReduceBwd>(), true);
  addPattern(std::make_shared<ascend::FuseTransdata>(), true);
}

AreaMode AscendModel::getDefaultAreaMode(const PrimOpPtr &node) const {
  if (node != nullptr && node->getOp() == kReshapeOpName) {
    return AreaMode::BASIC;
  }
  return AreaMode::COMPOSITE;
}
}  // namespace mlir::spliter
