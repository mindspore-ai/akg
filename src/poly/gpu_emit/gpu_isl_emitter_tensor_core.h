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
#ifndef POLY_GPU_ISL_EMITTER_TENSOR_CORE_H_
#define POLY_GPU_ISL_EMITTER_TENSOR_CORE_H_

#include "ir_pass.h"
#include "gpu_isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {

// add for tensor core
constexpr auto MMA_A = "matrix_a";
constexpr auto MMA_B = "matrix_b";
constexpr auto MMA_C = "accumulator";
constexpr auto MMA_SYNC = "matrix_sync";
constexpr auto MMA_PREFIX = "matrix_";
constexpr auto MMA_FILL_STMT_SERIAL = 2;
constexpr auto MMA_SYNC_STMT_SERIAL = 1;
constexpr auto CAST_FLAG = "CAST";
constexpr auto CAST_MODE_1 = "mode1";
constexpr auto GMREAD_FLAG = "GMRead";
constexpr auto FRAGMENT_A = "fragment_a";
constexpr auto FRAGMENT_B = "fragment_b";
constexpr auto FRAGMENT_C = "fragment_c";

constexpr auto FOR_INFO_COLLECT_DEPTH = 3;
constexpr auto LOCAL_INDEX_POS = 4;
constexpr auto TENSOR_CORE_MODE_ONE = "1";
constexpr auto TENSOR_CORE_MODE_TWO = "2";
constexpr auto WARP_MARKER = "warp_marker";

constexpr auto DATA_LOAD_STORE_FOR_DEPTH = 2;
constexpr auto DATA_COMPUTE_FOR_DEPTH = 3;
constexpr auto CONV_OUTPUT_DIMENSION = 4;
constexpr auto CONV_MATRIXA_DIMENSION = 4;

struct Tile {
  int m{-1};
  int n{-1};
  int k{-1};
};

class TensorCoreInfo {
 public:
  Tile warp_tile_;

  std::unordered_map<std::string, std::string> matrix_major_;
  std::unordered_map<std::string, std::string> matrix_abc_;
  std::unordered_map<air::ir::TensorKey, Region> bounds_;
  std::unordered_map<std::string, Array<Expr>> strides_;
  std::set<std::string> frag_reg_;
  std::unordered_set<std::string> cast_tensors_;
  std::unordered_map<std::string, Array<Expr>> min_bounds_;
  std::string wmma_scope_;
};

class GpuIslEmitterTensorCore : public GpuIslEmitter {
 public:
  GpuIslEmitterTensorCore(ScopInfo &info, const NodeInfoRepo &n, const isl::id_list &i) : GpuIslEmitter(info, n, i) {}
  ~GpuIslEmitterTensorCore() override = default;

  Stmt Emit(const isl::ast_node &node) final;

 private:
  Stmt EmitMark(const isl::ast_node_mark &node_id) final;
  isl::multi_aff TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &subscripts, const isl::id &stmt_id);
  TensorCoreInfo tensor_core_info_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_GPU_ISL_EMITTER_TENSOR_CORE_H_