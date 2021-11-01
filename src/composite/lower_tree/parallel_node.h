/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_PARALLEL_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_PARALLEL_NODE_H_
#include <sstream>
#include <vector>
#include <pass/utils.h>
#include "picojson.h"
#include "build_module.h"
#include "codegen/lower.h"
#include "codegen/pass_mgr.h"
#include "codegen/stage_lower.h"
#include "composite/utils/dimension_peeling.h"
#include "composite/utils/dump.h"
#include "composite/utils/util.h"
#include "composite/lower_tree/base_node.h"
#include "composite/lower_tree/multichild_node.h"

namespace akg {
namespace lower {
constexpr auto kParallel = "Parallel";

class CudaParallelLowerNode : public MultiChildLowerNode {
 public:
  CudaParallelLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                        const Array<NodeRef> &kernel_outputs)
      : MultiChildLowerNode(target, kernel_inputs, kernel_outputs) {
    CHECK(target_ == kCuda);
    entrance_stage_ = StageType::BeforeLowerFunc;
    name_ = __FUNCTION__;
  }
  ~CudaParallelLowerNode() override = default;
  void ExcuteImpl(StageType to) override;

 private:
  void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) override;
  Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) override;
};

class AscendParallelLowerNode : public MultiChildLowerNode {
 public:
  AscendParallelLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                          const Array<NodeRef> &kernel_outputs)
      : MultiChildLowerNode(target, kernel_inputs, kernel_outputs) {
    CHECK(target_ == kCce);
    entrance_stage_ = StageType::BeforeLowerFunc;
    name_ = __FUNCTION__;
  }
  ~AscendParallelLowerNode() override = default;
  void ExcuteImpl(StageType to) override;

 private:
  Stmt AddPeelInfoAndBlockAttr(Stmt &s, LowerData &data, PeelInfo &peel_info,
                               std::unordered_map<std::string, NodeRef> &outputs2args, int block);
  Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &block_irs) override;
  void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) override;
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_PARALLEL_NODE_H_
