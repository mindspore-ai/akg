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

#ifndef MFUSION_ANALYSIS_SPLIT_FUSETAGBARRIER_H
#define MFUSION_ANALYSIS_SPLIT_FUSETAGBARRIER_H

#include <string>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "mfusion/Analysis/Split/FusePattern.h"

namespace mlir {
namespace mfuse {
namespace split {

/// Merge adjacent split areas when any node carries one of the given unit attrs (legacy).
class FuseTagBarrier : public FusePattern {
 public:
  FuseTagBarrier(std::string name, llvm::ArrayRef<llvm::StringRef> mergeAttrs, FuseDirection direction);

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;

 private:
  bool areaHasMergeAttr(const AreaPtr &area) const;

  std::vector<std::string> merge_attrs_storage_;
  llvm::SmallVector<llvm::StringRef, 4> merge_attrs_;
};

/// Merge adjacent areas that share at least one mfusion.dvm_fuse_group (member role only).
class FuseTagBarrierByGroupId : public FusePattern {
 public:
  explicit FuseTagBarrierByGroupId(std::string name, FuseDirection direction);

 protected:
  bool check(const AreaPtr &area) override;
  bool match(const AreaPtr &area) override;

 private:
  void collectAreaGroups(const AreaPtr &area, llvm::StringSet<> &groups) const;
  static bool groupsIntersect(const llvm::StringSet<> &lhs, const llvm::StringSet<> &rhs);
};

FusePatternPtr createDvmFuseGroupTagBarrier(FuseDirection direction);

/// Deprecated: use createDvmFuseGroupTagBarrier.
FusePatternPtr createLayerNormDvmTagBarrier(FuseDirection direction);

}  // namespace split
}  // namespace mfuse
}  // namespace mlir

#endif  // MFUSION_ANALYSIS_SPLIT_FUSETAGBARRIER_H
