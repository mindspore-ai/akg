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

#include "emit_pass.h"
#include "gpu_isl_emitter_reduce.h"

namespace akg {
namespace ir {
namespace poly {
Stmt GpuIslEmitterTensorCore::Emit(const isl::ast_node &node) {
  Stmt stmt = GpuIslEmitter::Emit(node);

  if (info_.user_config_.GetEnableTensorCoreUsePoly() && info_.user_config_.GetEnableEmitCore()) {
    stmt = EmitForTensorCore(stmt, tensor_core_info_, info_);
  } else {
    tensor_core_info_.cast_tensors_ = info_.analysis_result_.GetCastTensors();
    stmt = EmitForTensorCoreDesignOne(stmt, tensor_core_info_);
  }

  return stmt;
}

Stmt GpuIslEmitterTensorCore::EmitStmt(const isl::ast_node_user &node) {
  CHECK(node.get_expr().isa<isl::ast_expr_op>());
  isl::ast_expr_op usr_expr = node.get_expr().as<isl::ast_expr_op>();
  CHECK(usr_expr);
  auto stmt_id = usr_expr.get_arg(0).as<isl::ast_expr_id>().get_id();
  auto node_id = node.get_annotation();

  if (info_.IsGMWrite(stmt_id) || info_.IsGMLWrite(stmt_id)) {
    auto iterator_map = node_info_map_.at(node_id).iterator_map;
    auto original = iterator_map.range_factor_domain().range_factor_range();
    auto srcid = original.get_tuple_id(isl_dim_out);
    bool no_need_to_emit = GpuIslEmitter::NoNeedToEmitForTempTensor(srcid);
    if (no_need_to_emit) return Stmt();
  }
  return GpuIslEmitter::EmitStmt(node);
}

Stmt GpuIslEmitterTensorCore::EmitMark(const isl::ast_node_mark &node) {
  std::string mark = node.get_id().get_name();
  // add for tensor core
  if (mark == WARP_MARKER || mark == CONV_KHKW_OUTER) {
    Stmt stmt = EmitAst(node.get_node());
    if (!stmt.defined()) {
      return Stmt();
    }
    return AttrStmt::make(Expr("INFO"), mark, StringImm::make(mark), stmt);
  }
  return GpuIslEmitter::EmitMark(node);
}

void GetNameWithoutShared(isl::id &tensor_id, ScopInfo &info) {
  if (info.user_config_.GetEnableMatmul()) {
    size_t pos = tensor_id.get_name().find(SHARE_SUFFIX);
    std::string substr = tensor_id.get_name().substr(0, pos);
    if (pos != 0) tensor_id = isl::id(tensor_id.ctx(), substr);
  }
}

isl::multi_aff GpuIslEmitterTensorCore::TensorAccessMultAff(isl::id &tensor_id, const Array<Expr> &tensor_index,
                                                            const isl::id &node_id) {
  GetNameWithoutShared(tensor_id, info_);
  return IslEmitter::TensorAccessMultAff(tensor_id, tensor_index, node_id);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
