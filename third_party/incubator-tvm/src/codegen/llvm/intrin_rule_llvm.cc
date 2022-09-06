/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file intrin_rule_llvm.cc
 *
 * 2021.12.21
 *   Fixed prefetch intrinsic
 * 2022.4.16
 *   Optimize log intrinsic
 * 2022.9.6
 *   Optimize tanh intrinsic
 */
#ifdef TVM_LLVM_VERSION

#include "intrin_rule_llvm.h"

namespace air {
namespace codegen {
namespace llvm {

static constexpr auto COEF_0 = 28.f;
static constexpr auto COEF_1 = 378.0f;
static constexpr auto COEF_2 = 3150.0f;
static constexpr auto COEF_3 = 17325.0f;
static constexpr auto COEF_4 = 62370.0f;
static constexpr auto COEF_5 = 135135.0f;

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.prefetch")
.set_body(DispatchLLVMIntrin<::llvm::Intrinsic::prefetch, 4>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.exp")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const auto type = x.type();
  if (type.is_float() && type.bits() == 32) {
    *rv = e;
  } else {
    DispatchLLVMPureIntrin<::llvm::Intrinsic::exp, 1>(targs, rv);
  }
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fma")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fmuladd, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.log")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  const auto type = x.type();
  if (type.is_float() && type.bits() == 32) {
    *rv = e;
  } else {
    DispatchLLVMPureIntrin<::llvm::Intrinsic::log, 1>(targs, rv);
  }
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sqrt")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sqrt, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.floor")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::floor, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.ceil")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ceil, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.trunc")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::trunc, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.fabs")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::fabs, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.round")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::round, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.nearbyint")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::nearbyint, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.tanh")
.set_body([](const TVMArgs& targs, TVMRetValue* rv) {
  // Efficient tanh computation using Lambert's continued fraction
  // tanh(x) = ((((x^2 + 378) * x^2 + 17325) * x^2 + 135135) * x) / (((28 * x^2 + 3150) * x^2 + 62370) * x^2 + 135135)
  Expr e = targs[0];
  const ir::Call* call = e.as<ir::Call>();
  CHECK(call != nullptr);
  const Expr& x = call->args[0];
  Expr square = x * x;
  Expr a = (((square + COEF_1) * square + COEF_3) * square + COEF_5) * x;
  Expr b = ((COEF_0 * square + COEF_2) * square + COEF_4) * square + COEF_5;
  *rv = ir::Div::make(a, b);
});

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.pow")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::pow, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.popcount")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::ctpop, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.cos")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::cos, 1>);

TVM_REGISTER_GLOBAL("tvm.intrin.rule.llvm.sin")
.set_body(DispatchLLVMPureIntrin<::llvm::Intrinsic::sin, 1>);

}  // namespace llvm
}  // namespace codegen
}  // namespace air

#endif  // LLVM_VERSION
