/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#ifndef POLY_GPU_ISL_EMITTER_REDUCE_H_
#define POLY_GPU_ISL_EMITTER_REDUCE_H_

#include "ir_pass.h"
#include "gpu_isl_emitter.h"
#include "poly/poly_util.h"

namespace akg {
namespace ir {
namespace poly {

// example:
// red_init_SumOp_S_1_0
constexpr auto REDUCE_FLAG_SIZE = 6;
constexpr auto REDUCE_FLAG_TYPE_POS = 2;
constexpr auto REDUCE_FLAG_STMT_PREFIX_POS = 3;
constexpr auto REDUCE_FLAG_STMT_NUM_POS = 4;
constexpr auto REDUCE_FLAG_REDUCE_INDEX = 5;

constexpr auto DEFAULT_TENSOR_INDEX = "[0]";

constexpr auto USELESS_INDEX = "0";
constexpr auto USELESS_SHAPE_SIZE = "1";

constexpr auto SHARED_MEMORY_PREFIX = "__shared__";

constexpr auto MEM_TYPE_SHARED = "shared";
constexpr auto MEM_TYPE_LOCAL = "local";

class GpuIslEmitterReduce : virtual public GpuIslEmitter {
 public:
  GpuIslEmitterReduce(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : GpuIslEmitter(info, n, i) {}
  ~GpuIslEmitterReduce() override = default;

  Stmt EmitterPostProcess(Stmt &s) override;
  Stmt EmitUserStmt(const isl::ast_node_user &node);

 private:
  Stmt EmitMark(const isl::ast_node_mark &node_id) final;
  Stmt EmitStmt(const isl::ast_node_user &node) final;
  Stmt EmitFilter(std::string name);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_GPU_ISL_EMITTER_REDUCE_H_