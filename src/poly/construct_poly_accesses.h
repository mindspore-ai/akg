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

#ifndef POLY_CONSTRUCT_POLY_ACCESSES_H_
#define POLY_CONSTRUCT_POLY_ACCESSES_H_

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>

#include <tuple>

#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {

std::pair<isl::map, isl::map> ConstructPolyAccess(const OperatorDomainSpace &domain, const Node *op,
                                                  const std::string &tensor, const Array<Expr> &dimensions,
                                                  AccessMap &accesses);

std::tuple<isl::union_map, isl::union_map, isl::union_map> ConstructPolyAccesses(const OperatorDomainSpace &domain,
                                                                                 const Stmt &s, AccessMap &accesses);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_CONSTRCUT_POLY_ACCESSES_H_