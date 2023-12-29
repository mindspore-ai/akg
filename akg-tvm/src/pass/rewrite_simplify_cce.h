/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#ifndef PASS_REWRITE_SIMPLIFY_CCE_H_
#define PASS_REWRITE_SIMPLIFY_CCE_H_
#define CHECK_IMPL_CONDITION false

#include "tvm.h"

namespace akg {
namespace ir {
/*!
 * \brief Simplify expr using custom cce simplifiers.
 *
 * \param expr The expression to be simplified.
 * \return The result.
 *
 * \note Analyzer will call into sub-analyzers to get the result.
 */
Expr Simplify_cce(Expr expr, const Map<Var, Range> &vrange = Map<Var, Range>());
/*!
 * \brief Simplify stmt using custom cce simplifiers.
 *
 * \param expr The statement to be simplified.
 * \return The result.
 *
 * \note Analyzer will call into sub-analyzers to get the result.
 */
Stmt Simplify_cce(const Stmt &stmt, const Map<Var, Range> &vrange = Map<Var, Range>());
}  // namespace ir
}  // namespace akg

#endif  // PASS_REWRITE_SIMPLIFY_CCE_H_
