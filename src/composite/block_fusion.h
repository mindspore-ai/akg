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
#ifndef AKG_SRC_COMPOSITE_BLOCK_FUSION_H_
#define AKG_SRC_COMPOSITE_BLOCK_FUSION_H_
#include <string>
#include <vector>
#include "ir_pass.h"

namespace akg {
namespace ir {
std::vector<Stmt> PipelineFusion(const std::vector<Stmt> &stmts, const Array<Array<NodeRef>> &pipeline_groups,
                                 const std::string &target);
Stmt BlockFusion(const std::vector<Stmt> &stmts, const std::string &target);
}  // namespace ir
}  // namespace akg
#endif  // AKG_SRC_COMPOSITE_BLOCK_FUSION_H_
