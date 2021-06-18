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

#include "composite/optimize/transdata_rewriter.h"
#include "pass/utils.h"

namespace akg {
// Since scenes with inconsistent shape dimensions are not easy to handle for poly,
// this pass tries to rewrite the Transdata to the Reshape and the Transpose,
// and then Reshape can be eliminated by subsequent passes(elim_reshape),
// which eliminates the inconsistency of dimensions and facilitates poly processing.
// Some cases of this pass as follows:
//
// Case 1:
// // attr [{"output_shape": [16, 1, 64, 64, 16, 16], "dst_format": "FRACTAL_NZ", "src_format": "NCHW",
//           "output_0_2_format": "DefaultFormat"}] attrs = 1
// output_0_3(16, 1, 64, 64, 16, 16) = TransData(output_0_2(16, 1, 1024, 1024)):int32:PI
//
//  ===>
//
// // attr [{"shape": [16, 1, 64, 16, 64, 16]}] attrs = 1
// output_0_2_re(16, 1, 64, 16, 64, 16) = Reshape(output_0_2(16, 1, 1024, 1024)):float32:PI
// // attr [{"perm": [0, 1, 4, 2, 3, 5]}] attrs = 1
// output_0_3(16, 1, 64, 64, 16, 16) = Transpose(output_0_2_re(16, 1, 64, 16, 64, 16)):float32:PI
//
//
// Case 2:
// // attr [{"output_shape": [12288, 1024], "output_0_1_format": "FRACTAL_NZ", "dst_format": "NCHW",
//           "src_format": "FRACTAL_NZ"}] attrs = 1
// output_0_2(12288, 1024) = TransData(output_0_1(64, 768, 16, 16)):int32:PI
//
// ===>
//
// // attr [{"perm": [1, 2, 0, 3]}] attrs = 1
// output_0_1_tr(768, 16, 64, 16) = Transpose(output_0_1(64, 768, 16, 16)):float16:PI
// //attr [{"shape": [12288, 1024]}] attrs = 1
// output_0_2(12288, 1024) = Reshape(output_0_1_tr(768, 16, 64, 16)):float16:PI
class TransDataRewrite : public IRMutator {
 private:
  Stmt Mutate_(const AttrStmt *op, const Stmt &s) override {
    if (op->attr_key == "attrs" && op->body.as<Provide>()) {
      const auto *provide = op->body.as<Provide>();
      CHECK(provide);
      auto value_call = provide->value.as<Call>();
      CHECK(value_call);
      if (value_call->name != "TransData") {
        return s;
      }
      auto attrs = Downcast<Map<std::string, NodeRef>>(op->node);
      auto src_format = GetFormatValue(attrs, "src_format");
      auto dst_format = GetFormatValue(attrs, "dst_format");
      // input and output for rewriter
      auto input_call = value_call->args[0].as<Call>();
      CHECK(input_call);
      auto input_shape = input_call->args;
      auto output_func = provide->func;
      auto output_args = provide->args;
      // has pad do not support
      if (HasPad(input_shape, output_args)) {
        return s;
      }
      if ((src_format == "DefaultFormat" || src_format == "NCHW") && dst_format == "FRACTAL_NZ") {
        // reshape
        auto reshape_shape = GetReshapeShape(input_shape, dst_format);
        auto reshape_out = placeholder(reshape_shape, input_call->type, input_call->name + "_re");
        auto reshape_stmt =
          GetReshapeStmt(GetRef<Expr>(input_call), reshape_out->op, reshape_out->shape, reshape_shape);
        // transpose
        auto transpose_perm = GetTransposePerm(reshape_out->shape.size(), dst_format);
        auto transpose_stmt =
          GetTransposeStmt(GetCallFromTensor(reshape_out), output_func, output_args, transpose_perm);
        return Block::make(reshape_stmt, transpose_stmt);
      } else if (src_format == "FRACTAL_NZ" && (dst_format == "DefaultFormat" || dst_format == "NCHW")) {
        // transpose
        auto transpose_perm = GetTransposePerm(input_shape.size(), dst_format);
        auto transpose_shape = TransposeArray(input_shape, transpose_perm);
        auto transpose_out = placeholder(transpose_shape, input_call->type, input_call->name + "_tr");
        auto transpose_stmt =
          GetTransposeStmt(GetRef<Expr>(input_call), transpose_out->op, transpose_out->shape, transpose_perm);
        // reshape
        auto reshape_shape = GetReshapeShape(transpose_out->shape, dst_format);
        auto reshape_stmt = GetReshapeStmt(GetCallFromTensor(transpose_out), output_func, output_args, reshape_shape);
        return Block::make(transpose_stmt, reshape_stmt);
      } else {
        LOG(FATAL) << "TransData for src_format " << src_format << " and " << dst_format
                   << " is not currently supported!";
      }
    }
    return IRMutator::Mutate_(op, s);
  }

  bool HasPad(const Array<Expr> &input_shape, const Array<Expr> &output_shape) {
    CHECK_GE(input_shape.size(), 2);
    CHECK_GE(output_shape.size(), 2);
    Array<Expr> check_shape;
    if (input_shape.size() < output_shape.size()) {
      check_shape = input_shape;
    } else {
      check_shape = output_shape;
    }
    for (auto i = check_shape.size() - 2; i < check_shape.size(); ++i) {
      auto dim = check_shape[i];
      if (!is_zero(Simplify(truncmod(Simplify(dim), fractal_size)))) {
        return true;
      }
    }
    return false;
  }

  Array<Expr> GetReshapeShape(Array<Expr> shape, std::string dst_format) {
    Array<Expr> NewShape;
    if (dst_format == "FRACTAL_NZ") {
      CHECK_GE(shape.size(), 2);
      for (size_t i = 0; i < shape.size() - 2; ++i) {
        NewShape.push_back(shape[i]);
      }
      // (m, n) ==> (m1, m0, n1, n0)
      for (size_t i = shape.size() - 2; i < shape.size(); ++i) {
        auto dim1 = Simplify(truncdiv(shape[i], fractal_size));
        auto dim2 = Expr(fractal_size);
        NewShape.push_back(dim1);
        NewShape.push_back(dim2);
        CHECK(Equal(Simplify(dim1 * dim2), shape[i])) << "The shape" << shape << "do not support rewrite for TransData";
      }
    } else {
      CHECK_GE(shape.size(), 4);
      for (size_t i = 0; i < shape.size() - 4; ++i) {
        NewShape.push_back(shape[i]);
      }
      // (m1, m0, n1, n0) ==> (m, n)
      NewShape.push_back(Simplify(shape[shape.size() - 4] * shape[shape.size() - 3]));
      NewShape.push_back(Simplify(shape[shape.size() - 2] * shape[shape.size() - 1]));
    }
    return NewShape;
  }

  std::string GetFormatValue(const Map<std::string, NodeRef> &attrs, const std::string &key) {
    if (attrs.find(key) != attrs.end()) {
      auto value = attrs[key];
      CHECK(value.as<StringImm>());
      return value.as<StringImm>()->value;
    }
    LOG(FATAL) << "fail to find " << key << " in attrs " << attrs;
    return "";
  }

  Stmt GetReshapeStmt(const Expr &input_call, const FunctionRef &output_func, const Array<Expr> &output_args,
                      const Array<Expr> &reshape_shape) {
    auto reshape_call = Call::make(input_call.type(), "Reshape", {input_call}, Call::CallType::PureIntrinsic);
    auto provide = Provide::make(output_func, 0, reshape_call, output_args);
    Map<std::string, NodeRef> attrs;
    attrs.Set("shape", reshape_shape);
    return AttrStmt::make(attrs, "attrs", Expr(1), provide);
  }

  std::vector<int> GetTransposePerm(int axes_num, std::string dst_format) {
    std::vector<int> tranpose_perm;
    CHECK_GE(axes_num, 4);
    int i = 0;
    for (; i < axes_num - 4; ++i) {
      tranpose_perm.push_back(i);
    }
    if (dst_format == "FRACTAL_NZ") {
      // (m1, m0, n1, n0) ==> (n1, m1, m0, n0)
      tranpose_perm.push_back(i + 2);
      tranpose_perm.push_back(i);
      tranpose_perm.push_back(i + 1);
      tranpose_perm.push_back(i + 3);
    } else {
      // (n1, m1, m0, n0) ==> (m1, m0, n1, n0)
      tranpose_perm.push_back(i + 1);
      tranpose_perm.push_back(i + 2);
      tranpose_perm.push_back(i);
      tranpose_perm.push_back(i + 3);
    }
    return tranpose_perm;
  }

  Stmt GetTransposeStmt(const Expr &input_call, const FunctionRef &output_func, const Array<Expr> &output_args,
                        const std::vector<int> &transpose_perm) {
    auto transpose_call = Call::make(input_call.type(), "Transpose", {input_call}, Call::CallType::PureIntrinsic);
    auto provide = Provide::make(output_func, 0, transpose_call, output_args);
    auto perm = Array<Integer>(transpose_perm.begin(), transpose_perm.end());
    Map<std::string, NodeRef> attrs;
    attrs.Set("perm", perm);
    return AttrStmt::make(attrs, "attrs", Expr(1), provide);
  }

  Array<Expr> TransposeArray(const Array<Expr> exprs, std::vector<int> perm) {
    CHECK_EQ(exprs.size(), perm.size());
    Array<Expr> res;
    for (auto i : perm) {
      res.push_back(exprs[i]);
    }
    return res;
  }

  Expr GetCallFromTensor(const Tensor &tensor) {
    return Call::make(tensor->dtype, tensor->op->name, tensor->shape, Call::CallType::Halide, tensor->op);
  }

  int fractal_size = 16;
};

Stmt TransDataRewriter::Run(const Stmt &stmt) { return TransDataRewrite().Mutate(stmt); }
}  // namespace akg
