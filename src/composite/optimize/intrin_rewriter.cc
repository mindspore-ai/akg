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

#include "composite/optimize/intrin_rewriter.h"
#include "pass/utils.h"

namespace akg {
// rewrite intrinsic that broadcast the last dimension
// For example:
// output_0_6(384, 32, 512, 16) = Sub(input_0(384, 32, 512, 16), output_0_3(384, 1, 512, 1))
//
// To ===>
//
// output_0_3_trans(384, 1, 1, 512) = Transpose(output_0_3(384, 1, 512, 1))
// output_0_3_trans_broad(384, 1, 16, 512) = BroadcastTo(output_0_3_trans(384, 1, 1, 512))
// output_0_3_trans_broad_trans(384, 1, 512, 16) = Transpose(output_0_3_trans_broad(384, 1, 16, 512))
// output_0_6(384, 32, 512, 16) = Sub(input_0(384, 32, 512, 16), output_0_3_trans_broad_trans(384, 1, 512, 16))
class BroadcastRewriterMutator : public IRMutator {
 public:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const auto *provide = op->body.as<Provide>();
      CHECK(provide);
      auto value_call = provide->value.as<Call>();
      CHECK(value_call);
      auto it = broadcast_ops_.find(value_call->name);
      if (it != broadcast_ops_.end()) {
        for (size_t i = 0; i < value_call->args.size(); ++i) {
          auto call = value_call->args[i].as<Call>();
          if (call && call->func.as<OperationNode>()) {
            auto rewrite_shape = GetBroadRewriteShape(call->type, call->args, provide->args);
            if (rewrite_shape.empty()) {
              continue;
            }
            auto rewrite_stmts = BroadcastRewrite(call, rewrite_shape);
            auto args = value_call->args;
            args.Set(i, Call::make(rewrite_out_tensor_->dtype, rewrite_out_tensor_->op->name,
                                   rewrite_out_tensor_->shape, Call::CallType::Halide, rewrite_out_tensor_->op));
            auto rewrite_value_call = Call::make(value_call->type, value_call->name, args, value_call->call_type);
            auto op_body = Provide::make(provide->func, provide->value_index, rewrite_value_call, provide->args);
            auto op_stmt = AttrStmt::make(op->node, op->attr_key, op->value, op_body);
            rewrite_stmts.push_back(op_stmt);
            return Block::make(rewrite_stmts);
          }
        }
      }
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  Array<Expr> GetBroadRewriteShape(Type type, const Array<Expr> &input_shape, const Array<Expr> &output_shape) {
    Array<Expr> broad_rewrite_shape;
    auto rewrite_type_bits = 16;
    auto rewrite_align_size = 16;
    if (type.bits() != rewrite_type_bits) {
      return broad_rewrite_shape;
    }
    std::vector<int> input_shape_const = GetExprArrayConst(input_shape);
    std::vector<int> output_shape_const = GetExprArrayConst(output_shape);
    if (input_shape_const.size() < 2 || output_shape_const.size() < 2) {
      return broad_rewrite_shape;
    }
    auto shape_len = input_shape_const.size();
    if (input_shape_const[shape_len - 1] == 1 && output_shape_const[shape_len - 1] == rewrite_align_size &&
        input_shape_const[shape_len - 2] % rewrite_align_size == 0) {
      input_shape_const[shape_len - 1] = input_shape_const[shape_len - 2];
      input_shape_const[shape_len - 2] = output_shape_const[shape_len - 1];
      for (auto s : input_shape_const) {
        broad_rewrite_shape.push_back(s);
      }
      return broad_rewrite_shape;
    }
    return broad_rewrite_shape;
  }

  std::vector<Stmt> BroadcastRewrite(const Call *call, const Array<Expr> &broad_rewrite_shape) {
    auto shape_size = call->args.size();
    std::vector<int> transpose_axes;
    for (size_t i = 0; i < shape_size - 2; ++i) {
      transpose_axes.push_back(i);
    }
    transpose_axes.push_back(shape_size - 1);
    transpose_axes.push_back(shape_size - 2);
    // transpose
    auto trans1_shape = TransposeArray(call->args, transpose_axes);
    auto trans1_tensor = placeholder(trans1_shape, call->type, call->name + "_trans");
    auto trans1_stmt =
      GetTransposeStmt(Downcast<Operation>(call->func).output(0), call->args, trans1_tensor, transpose_axes);
    // broadcast
    auto broad_tensor = placeholder(broad_rewrite_shape, trans1_tensor->dtype, trans1_tensor->op->name + "_broad");
    auto broad_stmt = GetBroadcastStmt(trans1_tensor, broad_tensor, broad_rewrite_shape);
    // transpose
    auto trans2_shape = TransposeArray(broad_tensor->shape, transpose_axes);
    auto trans2_tensor = placeholder(trans2_shape, broad_tensor->dtype, broad_tensor->op->name + "_trans");
    auto trans2_stmt = GetTransposeStmt(broad_tensor, broad_tensor->shape, trans2_tensor, transpose_axes);
    std::vector<Stmt> rewrite_stmt = {trans1_stmt, broad_stmt, trans2_stmt};
    rewrite_out_tensor_ = trans2_tensor;
    return rewrite_stmt;
  }

  Stmt GetTransposeStmt(const Tensor &input_tensor, const Array<Expr> &input_args, const Tensor &output_tensor,
                        const std::vector<int> &axes) {
    auto input_call =
      Call::make(input_tensor->dtype, input_tensor->op->name, input_args, Call::CallType::Halide, input_tensor->op);
    auto trans_body = Provide::make(
      output_tensor->op, 0, Call::make(input_tensor->dtype, "Transpose", {input_call}, Call::CallType::PureIntrinsic),
      output_tensor->shape);
    Map<std::string, NodeRef> trans_attrs;
    Array<Integer> axes_array(axes.begin(), axes.end());
    trans_attrs.Set("perm", axes_array);
    auto trans_stmt = AttrStmt::make(trans_attrs, "attrs", Expr(1), trans_body);
    return trans_stmt;
  }

  Stmt GetBroadcastStmt(const Tensor &input_tensor, const Tensor &output_tensor, const Array<Expr> &shape) {
    auto input_call = Call::make(input_tensor->dtype, input_tensor->op->name, input_tensor->shape,
                                 Call::CallType::Halide, input_tensor->op);
    auto broad_body = Provide::make(
      output_tensor->op, 0, Call::make(input_tensor->dtype, "BroadcastTo", {input_call}, Call::CallType::PureIntrinsic),
      output_tensor->shape);
    Map<std::string, NodeRef> broad_attrs;
    broad_attrs.Set("shape", shape);
    auto broad_stmt = AttrStmt::make(broad_attrs, "attrs", Expr(1), broad_body);
    return broad_stmt;
  }

  Array<Expr> TransposeArray(const Array<Expr> exprs, std::vector<int> perm) {
    CHECK_EQ(exprs.size(), perm.size());
    Array<Expr> res;
    for (auto i : perm) {
      res.push_back(exprs[i]);
    }
    return res;
  }

  std::vector<int> GetExprArrayConst(const Array<Expr> &exprs) {
    std::vector<int> exprs_const;
    for (auto e : exprs) {
      if (!(e->IsInstance<IntImm>() || e->IsInstance<UIntImm>())) {
        LOG(INFO) << "There are non-const expr:" << exprs;
        exprs_const.clear();
        return exprs_const;
      }
      exprs_const.push_back(ir::GetInt32Const(e));
    }
    return exprs_const;
  }

 private:
  std::unordered_set<std::string> broadcast_ops_ = {"BroadcastTo", "TensorAdd", "Add",     "RealDiv",
                                                    "Mul",         "Minimum",   "Maximum", "Sub"};
  Tensor rewrite_out_tensor_;
};

Stmt IntrinRewriter::Run(const Stmt &stmt) { return BroadcastRewriterMutator().Mutate(stmt); }
}  // namespace akg
