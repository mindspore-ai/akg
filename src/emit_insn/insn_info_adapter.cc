/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include <tvm/base.h>
#include <tvm/api_registry.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>

#include "ir_pass.h"
#include "pass/utils.h"
#include "insn_info.h"

namespace akg {
using ktvm::runtime::TVMArgs;
using ktvm::runtime::TVMRetValue;

class TestInfoNode : public Node {
 public:
  Array<VarExpr> ifvar;
  Array<Stmt> ifop;
  Array<VarExpr> forvar;
  Array<Stmt> forop;
  Array<NodeRef> stores;
  Array<NodeRef> loads;
  Array<StmtStoreInfo> com_info_list;
  Array<StmtStoreInfo> scalar_info_list;
  Array<StmtStoreInfo> dst_info_list;
  Array<StmtStoreInfo> src_info_list;
  Array<Buffer> dst_buffer_id_list;
  Array<Buffer> src_buffer_id_list;
  Map<std::string, Expr> arg_info_map;
  Map<std::string, Expr> ub_copy_pre;
  Map<std::string, Expr> ub_copy_post;
  std::string dma_mode;
  std::string intrin_name;
  std::string mode;
  ArgInfo arg_info;
  Array<Var> vars;
  Array<Expr> shapes;
  Array<Expr> strides;
  bool is_mask_set = false;
  Stmt result_stmt;

  static constexpr const char *_type_key = "TestInfo";
  TVM_DECLARE_NODE_TYPE_INFO(TestInfoNode, Node);

  void VisitAttrs(ktvm::AttrVisitor *v) {
    v->Visit("ifvar", &ifvar);
    v->Visit("ifop", &ifop);
    v->Visit("forvar", &forvar);
    v->Visit("forop", &forop);
    v->Visit("stores", &stores);
    v->Visit("loads", &loads);
    v->Visit("comInfoList", &com_info_list);
    v->Visit("dstInfoList", &dst_info_list);
    v->Visit("srcInfoList", &src_info_list);
    v->Visit("scalarInfoList", &scalar_info_list);
    v->Visit("dstBufferIdList", &dst_buffer_id_list);
    v->Visit("srcBufferIdList", &src_buffer_id_list);
    v->Visit("argInfoMap", &arg_info_map);
    v->Visit("ubCopyPre", &ub_copy_pre);
    v->Visit("ubCopyPost", &ub_copy_post);
    v->Visit("dmaMode", &dma_mode);
    v->Visit("intrinName", &intrin_name);
    v->Visit("mode", &mode);
    v->Visit("argInfo", &arg_info);
    v->Visit("isMaskSet", &is_mask_set);
    v->Visit("resultStmt", &result_stmt);
  }
};

class TestInfo : public NodeRef {
 public:
  TestInfo() = default;
  explicit TestInfo(const ktvm::NodePtr<TestInfoNode> &n) : NodeRef(n), node_(n) {}
  ~TestInfo() = default;

  inline TestInfoNode *GetNode() const {
    return static_cast<TestInfoNode *>(node_.get());
  }

  inline const TestInfoNode *operator->() const {
    return static_cast<const TestInfoNode *>(node_.get());
  }

  void SetIfAndFor(const StmtInfo &if_info, const StmtInfo &for_info) {
    this->GetNode()->ifvar = if_info.vars_;
    this->GetNode()->ifop = if_info.ops_;
    this->GetNode()->forvar = for_info.vars_;
    this->GetNode()->forop = for_info.ops_;
  }

  ktvm::NodePtr<TestInfoNode> node_;
};

TVM_REGISTER_NODE_TYPE(TestInfoNode);

/// Helper function to create computation instance
/// \param strides array of Int
/// \param shape array of Int
/// \param var array of Var
/// \param scope scope of this store
/// \param name name of store
/// \param index index of store
/// \param elem_offset offset of store, used to get access ptr
/// \param insn_offset offset of insn
/// \param dtype type of store
/// \param data_alignment length of continuous data
/// \param data variable of store
/// \return computation instance
StmtStoreInfo CreateStoreInfo(const Array<Expr> &strides, const Array<Expr> &shape, const Array<Var> &var,
                              const std::string &scope, const std::string &name, const Expr &index,
                              const Expr &elem_offset, const Expr &insn_offset, const Type &dtype,
                              const int &data_alignment, const Var &data) {
  auto info = StmtStoreInfo(make_node<StmtStoreInfoNode>());
  auto t = info.GetNode();
  t->strides_ = strides;
  t->shape_ = shape;
  t->var_ = var;
  t->scope_ = scope;
  t->name_ = name;
  t->index_ = index;
  t->elem_offset_ = elem_offset;
  t->insn_offset_ = insn_offset;
  t->dtype_ = dtype;
  t->data_alignment_ = data_alignment;
  t->data_ = data;

  return info;
}

TestInfo GetCompactComputationInfo(const Stmt stmt, bool same_dtype) {
  auto info = TestInfo(make_node<TestInfoNode>());
  auto t = info.GetNode();

  StmtInfo if_info = StmtInfo();
  StmtInfo for_info = StmtInfo();

  GetCompactComputationInfo(stmt, t->dst_info_list, t->src_info_list, if_info, for_info, same_dtype);
  info.SetIfAndFor(if_info, for_info);

  return info;
}

TVM_REGISTER_API("cce_util.create_storeinfo").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret =
    CreateStoreInfo(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10]);
});

TVM_REGISTER_API("cce_util.GetCompactComputationInfo").set_body([](const TVMArgs args, TVMRetValue *ret) {
  *ret = GetCompactComputationInfo(args[0], args[1]);
});
}  // namespace akg
