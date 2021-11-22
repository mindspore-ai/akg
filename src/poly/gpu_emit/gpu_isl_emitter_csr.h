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
#ifndef POLY_GPU_ISL_EMITTER_CSR_H_
#define POLY_GPU_ISL_EMITTER_CSR_H_

#include "ir_pass.h"
#include "gpu_isl_emitter.h"
#include "gpu_isl_emitter_reduce.h"

namespace akg {
namespace ir {
namespace poly {

class GpuIslEmitterCsr : virtual public GpuIslEmitter {
 public:
  GpuIslEmitterCsr(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : GpuIslEmitter(info, n, i) {}
  ~GpuIslEmitterCsr() override = default;

 private:
  Stmt EmitAccessNodeCall(const Node *node, const VarMap &var_map_tmp, BufferedFootPrintInfo &buffer_fp_info) final;
  Stmt EmitAccessNodeFromPromoteAcsCall(isl::id var, const Node *node, Array<Expr> &args) final;
  Stmt EmitFor(const isl::ast_node_for &node) final;
  Stmt SubstituteTensorStmt(const Stmt &s, Tensor origin, Tensor replaced) final;
  Stmt EmitTensorOfTensorStmt(const Stmt &s) final;

  bool csr_dynamic_scope_{false};
};

class GpuIslEmitterCsrReduce : public GpuIslEmitterCsr, public GpuIslEmitterReduce {
 public:
  GpuIslEmitterCsrReduce(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) :
    GpuIslEmitter(info, n, i), GpuIslEmitterCsr(info, n, i), GpuIslEmitterReduce(info, n, i) {}
  ~GpuIslEmitterCsrReduce() override = default;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_GPU_ISL_EMITTER_CSR_H_