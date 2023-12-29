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
#ifndef COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_FUSEPATTERN_H_
#define COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_FUSEPATTERN_H_

#include <memory>
#include <string>
#include <vector>
#include "akg/Dialect/MindSpore/Spliter/Area.h"

namespace mlir::spliter {
class CircleChecker {
 public:
  // whether it will form a circle if the two areas are fused.
  virtual bool hasCircle(const AreaPtr &a, const AreaPtr &b) const = 0;
  virtual ~CircleChecker() = default;
};
using CircleCheckerPtr = std::shared_ptr<CircleChecker>;

enum class FuseDirection {
  FORWARD,  // fuse with inputs
  BACKWARD  // fuse with outputs
};

// the base class of fusion patterns
class FusePattern {
 public:
  explicit FusePattern(const std::string &patterName) : name(patterName) {}
  virtual ~FusePattern() = default;
  // run the pattern
  bool run(const AreaPtr &dom) {
    reset();
    return check(dom) && match(dom);
  }
  std::string toString() const;
  // Bind the circle checker
  void setCircleChecker(const CircleCheckerPtr &c) { circleChecker = c; }

  std::string getName() const { return name; }
  FuseDirection getDirection() const { return direction; }
  std::vector<AreaPtr> fusedAreas;

 protected:
  void reset() { fusedAreas.clear(); }
  // Check whether the pattern can handle this area
  virtual bool check(const AreaPtr &) { return true; }
  // Match the ADJACENT areas of `dom`
  virtual bool match(const AreaPtr &dom) = 0;
  // whether it will form a circle if the two areas are fused.
  bool hasCircle(const AreaPtr &a, const AreaPtr &b) const {
    assert(circleChecker != nullptr);
    return circleChecker->hasCircle(a, b);
  }

  std::string name;
  FuseDirection direction{FuseDirection::FORWARD};
  CircleCheckerPtr circleChecker{nullptr};
};
using FusePatternPtr = std::shared_ptr<FusePattern>;

/* some common patterns are defined below */
enum class FuseType { kWidth, kDepth };
class FuseReshape : public FusePattern {
 public:
  FuseReshape() : FusePattern("reshape") {}
  ~FuseReshape() = default;

 protected:
  bool check(const AreaPtr &dom) override { return dom->pattern() == NodePattern::RESHAPE; }
  bool match(const AreaPtr &dom) override;
  void keepMinimumArea(const AreaPtr &a, FuseDirection dir);
  AreaPtr minArea;
};

class FuseIsolateReshape : public FusePattern {
 public:
  FuseIsolateReshape() : FusePattern("isolate_reshape") {}
  ~FuseIsolateReshape() = default;

 protected:
  bool check(const AreaPtr &dom) override { return dom->pattern() == NodePattern::RESHAPE && dom->size() == 1; }
  bool match(const AreaPtr &dom) override;
};

class FuseElemwiseBroadcastFwd : public FusePattern {
 public:
  explicit FuseElemwiseBroadcastFwd(FuseType fuseType) : FusePattern("elemwise_broadcast_fwd"), fuseType(fuseType) {
    direction = FuseDirection::FORWARD;
    name += (fuseType == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastFwd() = default;
  static FusePatternPtr createDepthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kDepth); }
  static FusePatternPtr createWidthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kWidth); }

 protected:
  bool check(const AreaPtr &dom) override;
  bool match(const AreaPtr &dom) override;
  FuseType fuseType;
};

class FuseReduceFwd : public FusePattern {
 public:
  FuseReduceFwd(FuseType newFuseType, size_t newSizelimit)
      : FusePattern("reduce_fwd"), fuseType(newFuseType), sizeLimit(newSizelimit) {
    direction = FuseDirection::FORWARD;
    name += (fuseType == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseReduceFwd() = default;
  static FusePatternPtr createDepthMatcher(size_t newSizelimit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kDepth, newSizelimit);
  }
  static FusePatternPtr createWidthMatcher(size_t newSizelimit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kWidth, newSizelimit);
  }

 protected:
  bool check(const AreaPtr &dom) override;
  bool match(const AreaPtr &dom) override;
  FuseType fuseType;
  size_t sizeLimit;
};

class FuseElemwiseBroadcastBwd : public FusePattern {
 public:
  FuseElemwiseBroadcastBwd(FuseType newFuseType, size_t newSizeLimit)
      : FusePattern("elemwise_broadcast_bwd"), fuseType(newFuseType), sizeLimit(newSizeLimit) {
    direction = FuseDirection::BACKWARD;
    name += (fuseType == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastBwd() = default;
  static FusePatternPtr createDepthMatcher(size_t newSizeLimit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kDepth, newSizeLimit);
  }
  static FusePatternPtr createWidthMatcher(size_t newSizeLimit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kWidth, newSizeLimit);
  }

 protected:
  bool check(const AreaPtr &dom) override;
  bool match(const AreaPtr &dom) override;
  FuseType fuseType;
  size_t sizeLimit;
};

// bind the virtual nodes to their inputs
class FuseVirtualNode : public FusePattern {
 public:
  FuseVirtualNode() : FusePattern("bind_virtual_node") { direction = FuseDirection::FORWARD; }
  ~FuseVirtualNode() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->pattern() == NodePattern::VIRTUAL; }
  bool match(const AreaPtr &area) override;
};
}  // namespace mlir::spliter
#endif  // COMPILER_INCLUDE_AKG_DIALECT_MINDSPORE_SPLITER_FUSEPATTERN_H_
