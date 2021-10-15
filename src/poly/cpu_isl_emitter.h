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
#ifndef POLY_CPU_ISL_EMITTER_H_
#define POLY_CPU_ISL_EMITTER_H_

#include "poly/isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {

class CpuIslEmitter : public IslEmitter {
 public:
  CpuIslEmitter(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : IslEmitter(info, n, i) {}
  ~CpuIslEmitter() override = default;
  Stmt Emit(const isl::ast_node &node);

 private:
  Stmt EmitBlock(const isl::ast_node_block &block_node) override;
  Stmt EmitUserStmt(const isl::ast_node_user &node) override;
  Stmt EmitCall(const isl::ast_node_user &node) override;
  Stmt EmitRealizeForGlobalTensor(const Stmt &stmt);
  Stmt InsertRealize(const Stmt &stmt, const isl::id &var);
  Stmt EmitReduce(const std::vector<std::string> &args);
  Stmt EmitMatrixTranspose(const std::vector<std::string> &names);
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_CPU_ISL_EMITTER_H_
