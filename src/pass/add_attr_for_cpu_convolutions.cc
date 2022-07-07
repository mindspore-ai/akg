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
#include <tvm/ir_pass.h>
#include <tvm/ir_mutator.h>
#include <tvm.h>
#include <pass/ir_util.h>
#include <pass/storage_access.h>
#include <pass/utils.h>
#include "ir_pass.h"

namespace akg {
namespace ir {
/*
Add attributes for cpu-convolutions operators.
*/
class ScanConvLoopOp : public IRVisitor {
  void Visit_(const For *op) override {
    _axes_names.insert(op->loop_var->name_hint);
    IRVisitor::Visit(op->body);
  }

 public:
  std::string ConcatSetToString() {
    std::string tmp = "|";
    for (auto name : _axes_names) {
      tmp += name + "|";
    }
    return tmp;
  }

  std::set<std::string> _axes_names;
  std::string _conv_type = "conv2d";  // can be conv2d, depthwise_conv2d ...
};

Stmt AddAttrForConvolutionsOp(const Stmt stmt) {
  auto scan_conv_loop_op = ScanConvLoopOp();
  scan_conv_loop_op.Visit(stmt);
  auto axes_names = scan_conv_loop_op.ConcatSetToString();
  auto new_stmt = AttrStmt::make(Expr("INFO"), AKG_CONVOLUTION_AXES, Expr(axes_names), stmt);
  new_stmt = AttrStmt::make(Expr("INFO"), AKG_CONVOLUTION_TPYE, Expr(scan_conv_loop_op._conv_type), new_stmt);

  return new_stmt;
}
}  // namespace ir
}  // namespace akg
