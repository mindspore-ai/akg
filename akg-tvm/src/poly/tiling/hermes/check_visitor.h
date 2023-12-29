/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_HERMES_CHECK_VISITOR_H_
#define POLY_TILING_HERMES_CHECK_VISITOR_H_

#include <tvm.h>
#include <tvm/ir_visitor.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "poly/tiling/hermes/node.h"
#include "poly/tiling/hermes/op.h"

namespace akg {
namespace ir {
namespace poly {
class CheckVisitor : public IRVisitor {
 public:
  void Visit_(const Provide *op) override;
  void Visit_(const For *op) override;
  void Visit_(const Call *op) override;
  void Visit_(const Realize *op) override;

  // BinaryOpNode
  void Visit_(const Add *op) override;
  void Visit_(const Sub *op) override;
  void Visit_(const Mul *op) override;
  void Visit_(const Div *op) override;
  void Visit_(const FloorDiv *op) override;
  void Visit_(const Min *op) override;
  void Visit_(const Max *op) override;

  // CmpOpNode
  void Visit_(const EQ *op) override;
  void Visit_(const NE *op) override;

  // ExprNode
  void Visit_(const Cast *op) override;
  void Visit_(const Select *op) override;

  static void Clear();
  static void PrintBuildGraphInfo();

  static std::vector<std::shared_ptr<Node>> nodes_;

 private:
  void SetOperatorType(Op::OpType op_type);
  void DefineCallOpType(const Call *op);
  void SetInputNode(const Call *op);
  void UpdateCurrentNode(const air::ir::FunctionRef &func, int value_index, const std::shared_ptr<Node> &node);
  static std::string GetNewNodeName(const std::shared_ptr<Node> &node, const air::Array<air::Expr> &op_args);
  std::shared_ptr<Tensor> GetTensor(const air::Array<air::Expr> &args, const std::string &name);
  static std::shared_ptr<Tensor> GetTensor(const air::ir::FunctionRef &func, int value_index);
  static std::string GetDatatypeString(air::DataType op_dtype);
  static std::vector<std::string> GetVarNamesFromExpr(const Expr &expr);

  std::shared_ptr<Node> cur_node_;
  Node realize_node_;
  std::string realize_dtype_;
  std::string loop_var_;
  std::map<int, std::string> merge_map_;
  std::pair<bool, std::string> dtype_change_ = std::make_pair(false, "");
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_HERMES_CHECK_VISITOR_H_
