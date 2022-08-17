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
#include "composite/lower_tree/base_node.h"
#include "composite/lower_tree/multichild_node.h"

namespace akg {
namespace lower {
constexpr auto kStitch = "Stitch";
constexpr auto kAllocMap = "alloc_map";
constexpr auto kReuseMap = "reuse_map";
constexpr auto kCleanOpMap = "clean_op_map";

class StitchLowerNode : public MultiChildLowerNode {
 public:
  explicit StitchLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                           const Array<NodeRef> &kernel_outputs, Map<std::string, Array<NodeRef>> alloc_map)
      : MultiChildLowerNode(target, kernel_inputs, kernel_outputs), alloc_map_(alloc_map){};
  ~StitchLowerNode() override = default;
  void Lower(StageType to) override;

 protected:
  Stmt MergeStmts(const LowerData &data, std::vector<Stmt> &stitch_irs) override;
  Map<std::string, NodeRef> GetStitchForwardInfo(const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim,
                                                 Expr child_json);
  void PostUpdateDataAndNodeRef(LowerData &data, NodeRef &node_ref) override;

  virtual Map<std::string, NodeRef> GetNewAttr(Map<std::string, NodeRef> &forward_infos,
                                               const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim,
                                               Expr child_json) = 0;
  virtual void GetStitchForwardInfoArgs(){};
  virtual void GetBufferManager(std::vector<Stmt> &stitch_irs){};
  virtual void MergeIRAndTryDump(DumpManager &dump_mng, Stmt &merged_ir, std::vector<Stmt> &stitch_irs,
                                 const LowerData &data){};
  virtual void FixLowerDataForStitch(LowerData &data, NodeRef &node_ref){};

  Map<std::string, Array<NodeRef>> alloc_map_;
};

class CudaStitchLowerNode : public StitchLowerNode {
 public:
  CudaStitchLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                      const Array<NodeRef> &kernel_outputs, Map<std::string, Array<NodeRef>> alloc_map,
                      Map<std::string, Array<NodeRef>> reuse_map, Map<std::string, Array<NodeRef>> clean_op_map)
      : StitchLowerNode(target, kernel_inputs, kernel_outputs, alloc_map), reuse_map_(reuse_map) {
    CHECK(target_ == kCuda);
    entrance_stage_ = StageType::BeforeFlattern;
    name_ = __FUNCTION__;
  }
  ~CudaStitchLowerNode() override = default;

 private:
  Map<std::string, NodeRef> GetNewAttr(Map<std::string, NodeRef> &forward_infos,
                                       const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim,
                                       Expr child_json) override;
  void GetStitchForwardInfoArgs() override;
  void GetBufferManager(std::vector<Stmt> &stitch_irs) override;
  void MergeIRAndTryDump(DumpManager &dump_mng, Stmt &merged_ir, std::vector<Stmt> &stitch_irs,
                         const LowerData &data) override;
  void FixLowerDataForStitch(LowerData &data, NodeRef &) override;
  std::unordered_map<std::string, NodeRef> GetStitchBuffer(const Map<std::string, Array<NodeRef>> &alloc_map);
  Map<std::string, Array<NodeRef>> reuse_map_;
  std::vector<StitchOpType> ir_type_array_;
  std::vector<size_t> split_index;
  std::vector<GridBlockDims> dim_array;
  StitchAttrInfo stitch_attr;
  std::unordered_map<std::string, NodeRef> stitch_buffer_;
  Map<Tensor, Buffer> workspace_binds_;
  std::unordered_map<std::string, Region> buf_region_map_;
  int total_block_{1};
};

class AscendStitchLowerNode : public StitchLowerNode {
 public:
  AscendStitchLowerNode(const std::string &target, const Array<NodeRef> &kernel_inputs,
                        const Array<NodeRef> &kernel_outputs, const std::string &stitch_origin_json,
                        Map<std::string, Array<NodeRef>> alloc_map)
      : StitchLowerNode(target, kernel_inputs, kernel_outputs, alloc_map), stitch_origin_json_(stitch_origin_json) {
    CHECK(target_ == kCce);
    entrance_stage_ = StageType::BeforeFlattern;
    name_ = __FUNCTION__;
  }
  ~AscendStitchLowerNode() override = default;

 private:
  Map<std::string, NodeRef> GetNewAttr(Map<std::string, NodeRef> &forward_infos,
                                       const Map<std::string, NodeRef> &child_attrs, size_t i, bool fold_dim,
                                       Expr child_json) override;
  void GetStitchForwardInfoArgs() override{};
  void GetBufferManager(std::vector<Stmt> &stitch_irs) override;
  void MergeIRAndTryDump(DumpManager &dump_mng, Stmt &merged_ir, std::vector<Stmt> &stitch_irs,
                         const LowerData &data) override;
  void FixLowerDataForStitch(LowerData &data, NodeRef &node_ref) override;

  Stmt AddPeelInfoForLoopAndData(Stmt &s, LowerData &data, Map<std::string, NodeRef> &attrs);
  std::unordered_map<std::string, NodeRef> GetStitchBuffer(const Map<std::string, Array<NodeRef>> &alloc_map);
  Map<Tensor, Buffer> FixBinds(const Map<Tensor, Buffer> &origin_binds, const Array<NodeRef> &ordered_args);

  std::string stitch_origin_json_;
  Map<Tensor, Buffer> workspace_binds_;
  std::unordered_map<std::string, NodeRef> stitch_buffer;
};
}  // namespace lower
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_LOWER_TREE_STITCH_NODE_H_
