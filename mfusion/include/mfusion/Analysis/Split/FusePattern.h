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

#ifndef MFUSION_ANALYSIS_SPLIT_FUSEPATTERN_H
#define MFUSION_ANALYSIS_SPLIT_FUSEPATTERN_H

#include <vector>
#include <memory>
#include <string>
#include "mfusion/Analysis/Split/Area.h"

namespace mlir {
namespace mfuse {
namespace split {

// Forward declarations
class ReachTable;

/// FuseDirection indicates the direction of fusion
enum class FuseDirection {
  FORWARD,  // Fuse from input to output
  BACKWARD  // Fuse from output to input
};

/// FuseType indicates the type of fusion
enum class FuseType {
  kWidth,  // Fuse multiple inputs of the same node
  kDepth   // Fuse nodes along the execution path
};

/// FusePattern is the base class for fusion patterns
class FusePattern {
 public:
  explicit FusePattern(const std::string &name, FuseDirection direction = FuseDirection::FORWARD)
      : name_(name), direction_(direction) {}
  virtual ~FusePattern() = default;

  /// Run this pattern on the given area
  bool run(const AreaPtr &area);

  /// Convert to string representation
  std::string toString() const;

  /// Get the name of this pattern
  const std::string &name() const { return name_; }

  /// Get the direction of this pattern
  FuseDirection direction() const { return direction_; }

  /// Set the circle checker
  void setCircleChecker(std::shared_ptr<ReachTable> checker);

  /// Get the fused areas
  std::vector<AreaPtr> &fused_areas() { return fused_areas_; }

 protected:
  void reset();
  /// Check if the area is applicable for this pattern
  virtual bool check(const AreaPtr &area) { return false; }

  /// Match and collect fused areas
  virtual bool match(const AreaPtr &area) { return false; }

  /// Check if the fusion of `area` and `fuse_area` will form a circle
  bool hasCircle(const AreaPtr &area, const AreaPtr &fuse_area);

  std::string name_;
  FuseDirection direction_;
  std::shared_ptr<ReachTable> circle_checker_{nullptr};
  std::vector<AreaPtr> fused_areas_;
};

using FusePatternPtr = std::shared_ptr<FusePattern>;

/// FuseReshape fuses reshape operations with their neighbors
class FuseReshape : public FusePattern {
 public:
  FuseReshape() : FusePattern("reshape", FuseDirection::FORWARD) {}
  ~FuseReshape() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->pattern() == NodePattern::RESHAPE; }
  bool match(const AreaPtr &area) override;
  void keepMinimumArea(const AreaPtr &a, FuseDirection dir);
  AreaPtr min_area_;
};

/// FuseIsolateReshape fuses isolated reshape operations
class FuseIsolateReshape : public FusePattern {
 public:
  FuseIsolateReshape() : FusePattern("isolate_reshape", FuseDirection::FORWARD) {}
  ~FuseIsolateReshape() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->pattern() == NodePattern::RESHAPE && area->userNum() == 1; }
  bool match(const AreaPtr &area) override;
};

/// FuseElemwiseBroadcastFwd fuses elemwise and broadcast operations in forward direction
class FuseElemwiseBroadcastFwd : public FusePattern {
 public:
  explicit FuseElemwiseBroadcastFwd(FuseType fuse_type)
      : FusePattern("elemwise_broadcast_fwd", FuseDirection::FORWARD), fuse_type_(fuse_type) {
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastFwd() = default;
  static FusePatternPtr createDepthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kDepth); }
  static FusePatternPtr createWidthMatcher() { return std::make_shared<FuseElemwiseBroadcastFwd>(FuseType::kWidth); }

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;
  FuseType fuse_type_;
};

/// FuseDynElemwiseBroadcastFwd fuses dynamic elemwise and broadcast operations in forward direction
class FuseDynElemwiseBroadcastFwd : public FusePattern {
 public:
  explicit FuseDynElemwiseBroadcastFwd(FuseType fuse_type)
      : FusePattern("elemwise_broadcast_fwd", FuseDirection::FORWARD), fuse_type_(fuse_type) {
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseDynElemwiseBroadcastFwd() = default;

  static FusePatternPtr createDepthMatcher() { return std::make_shared<FuseDynElemwiseBroadcastFwd>(FuseType::kDepth); }

  static FusePatternPtr createWidthMatcher() { return std::make_shared<FuseDynElemwiseBroadcastFwd>(FuseType::kWidth); }

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;
  FuseType fuse_type_;
};

/// FuseReduceFwd fuses reduce operations in forward direction
class FuseReduceFwd : public FusePattern {
 public:
  FuseReduceFwd(FuseType fuse_type, size_t size_limit)
      : FusePattern("reduce_fwd", FuseDirection::FORWARD), fuse_type_(fuse_type), size_limit_(size_limit) {
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseReduceFwd() = default;

  static FusePatternPtr createDepthMatcher(size_t size_limit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kDepth, size_limit);
  }

  static FusePatternPtr createWidthMatcher(size_t size_limit) {
    return std::make_shared<FuseReduceFwd>(FuseType::kWidth, size_limit);
  }

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;
  FuseType fuse_type_;
  size_t size_limit_;
};

/// FuseDynReduceFwd fuses dynamic reduce operations in forward direction
class FuseDynReduceFwd : public FusePattern {
 public:
  FuseDynReduceFwd(FuseType fuse_type, size_t size_limit)
      : FusePattern("reduce_fwd", FuseDirection::FORWARD), fuse_type_(fuse_type), size_limit_(size_limit) {
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseDynReduceFwd() = default;

  static FusePatternPtr createDepthMatcher(size_t size_limit) {
    return std::make_shared<FuseDynReduceFwd>(FuseType::kDepth, size_limit);
  }

  static FusePatternPtr createWidthMatcher(size_t size_limit) {
    return std::make_shared<FuseDynReduceFwd>(FuseType::kWidth, size_limit);
  }

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;
  FuseType fuse_type_;
  size_t size_limit_;
};

/// FuseElemwiseBroadcastBwd fuses elemwise and broadcast operations in backward direction
class FuseElemwiseBroadcastBwd : public FusePattern {
 public:
  FuseElemwiseBroadcastBwd(FuseType fuse_type, size_t size_limit)
      : FusePattern("elemwise_broadcast_bwd", FuseDirection::BACKWARD), fuse_type_(fuse_type), size_limit_(size_limit) {
    name_ += (fuse_type == FuseType::kWidth ? "_width" : "_depth");
  }
  ~FuseElemwiseBroadcastBwd() = default;

  static FusePatternPtr createDepthMatcher(size_t size_limit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kDepth, size_limit);
  }

  static FusePatternPtr createWidthMatcher(size_t size_limit) {
    return std::make_shared<FuseElemwiseBroadcastBwd>(FuseType::kWidth, size_limit);
  }

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;
  FuseType fuse_type_;
  size_t size_limit_;
};

/// FuseVirtualNode fuses virtual nodes with their inputs
class FuseVirtualNode : public FusePattern {
 public:
  FuseVirtualNode() : FusePattern("FuseVirtualNode", FuseDirection::FORWARD) {}
  ~FuseVirtualNode() = default;

 protected:
  bool check(const AreaPtr &area) override { return area->pattern() == NodePattern::VIRTUAL; }
  bool match(const AreaPtr &area) override;
};

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_FUSEPATTERN_H
