/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef POLY_SCOP_BUILDER_H_
#define POLY_SCOP_BUILDER_H_

#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <tvm/ir_visitor.h>

#include "poly/isl.h"
#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {
static const char kStatementLabel[] = "S_";

isl::space CreateParamsSpace(const isl::ctx &ctx);

isl::space CreateParamsSpace(const isl::ctx &ctx, const std::unordered_map<std::string, air::Var> &params);

isl::schedule MakeScheduleTree(const isl::space &param_space, isl::set paramSet, const Stmt &stmt, ScopInfo &scop_info);

std::vector<isl::aff> Expr2AffBounds(const isl::space &space, const Expr &e, bool allow_min, bool allow_max,
                                     bool ignore_error = true);

std::vector<isl::aff> Expr2AffChecked(const isl::space &space, const Expr &e, bool allow_min, bool allow_max);

isl::aff Expr2Aff(const isl::space &space, const Expr &e);
isl::aff Int2Aff(const isl::space &s, int64_t v);
isl::multi_id CollectTensorCoordinate(const isl::space &pspace, const isl::id &Id, size_t dim);

isl::map AddSuffix4Accesses(AccessMap &accesses, const isl::map &in_map, const Node *op, const isl::ctx &ctx);

void ParseStmtOpCall(const isl::id &id, const Call *call, AnalysisResult &result, const FunctionRef &func);

void ParseStmtOps(const isl::id &id, const Expr &val, AnalysisResult &result, const FunctionRef &func);

void ParseStmtOps(const isl::id &id, const Evaluate *stmt, AnalysisResult &result, const isl::union_map &new_reads,
                  const isl::union_map &new_writes);

void ParseStmtOps(const isl::id &id, const Provide *stmt, AnalysisResult &result, const isl::union_map &new_reads,
                  const isl::union_map &new_writes);

VarNames VisitVarNames(const air::Expr &arg, VarNames var_names, bool add_num = true, bool visit_tot = false);

bool IsNum(const std::string &name);

isl::set CutSet(std::vector<Expr> cond_vec, const isl::set &set, bool is_else, bool is_or);

isl::schedule MakeScheduleTreeHelper(const NodeRef &s, ScopInfo &scop_info, const isl::set &set,
                                     const isl::id_list &outer, ssize_t macro_stmt);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SCOP_BUILDER_H_
