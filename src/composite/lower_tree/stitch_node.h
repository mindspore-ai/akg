/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef AKG_SRC_COMPOSITE_LOWER_TREE_STITCH_NODE_H_
#define AKG_SRC_COMPOSITE_LOWER_TREE_STITCH_NODE_H_
#include <sstream>
#include <vector>
#include <pass/utils.h>
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
constexpr auto kStitch = "Stitch";
constexpr auto kAllocMap = "alloc_map";
constexpr auto kReuseMap = "reuse_map";
constexpr auto kCleanOpMap = "clean_op_map";

class CudaStitchLowerNode : public MultiChildLowerNode {
 public:
  CudaStitchLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                      const Array<NodeRef> &kernel_outputs, Map<std::string, Array<NodeRef>> alloc_map,
                      Map<std::string, Array<NodeRef>> reuse_map, Map<std::string, Array<NodeRef>> clean_op_map)
      : MultiChildLowerNode(target, kernel_inputs, kernel_outputs),
        alloc_map_(alloc_map),
        reuse_map_(reuse_map),
        clean_op_map_(clean_op_map) {
    CHECK(target_ == kCuda);
    entrance_stage_ = StageType::BeforeLowerFunc;
    name_ = __FUNCTION__;
  }
  ~CudaStitchLowerNode() override = default;
  void Lower(StageType to) override;

 private:
  Map<std::string, NodeRef> GetStitchForwardInfo(Expr child_json, const Map<std::string, NodeRef> &child_attrs,
                                                 size_t i, bool fold_dim, std::vector<GridBlockDims> &dim_array,
                                                 std::vector<StitchOpType> &ir_type_array,
                                                 std::vector<size_t> &split_index);
  Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &stitch_irs) override;
  void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &) override;

  Map<std::string, Array<NodeRef>> alloc_map_;
  Map<std::string, Array<NodeRef>> reuse_map_;
  Map<std::string, Array<NodeRef>> clean_op_map_;

  std::vector<StitchOpType> ir_type_array_;
};

class AscendStitchLowerNode : public MultiChildLowerNode {
 public:
  AscendStitchLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                        const Array<NodeRef> &kernel_outputs, const std::string &stitch_origin_json,
                        Map<std::string, Array<NodeRef>> alloc_map)
      : MultiChildLowerNode(target, kernel_inputs, kernel_outputs),
        stitch_origin_json_(stitch_origin_json),
        alloc_map_(alloc_map) {
    CHECK(target_ == kCce);
    entrance_stage_ = StageType::BeforeFlattern;
    name_ = __FUNCTION__;
  }
  ~AscendStitchLowerNode() override = default;
  void Lower(StageType to) override;

 private:
  Map<std::string, NodeRef> GetStitchForwardInfo(const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim);
  Stmt AddPeelInfoForLoopAndData(Stmt &s, LowerData &data, Map<std::string, NodeRef> &attrs);
  std::unordered_map<std::string, NodeRef> GetStitchBuffer(const Map<std::string, Array<NodeRef>> &alloc_map);
  Map<Tensor, Buffer> FixBinds(const Map<Tensor, Buffer> &origin_binds, const Array<NodeRef> &ordered_args);
  Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &stitch_irs) override;
  void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &node_ref) override;

  std::string stitch_origin_json_;
  Map<std::string, Array<NodeRef>> alloc_map_;
  Map<Tensor, Buffer> workspace_binds_;
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_STITCH_NODE_H_
