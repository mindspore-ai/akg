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
#include "poly/scop.h"

#include <fstream>

#include "poly/scop_builder.h"
#include "poly/poly_util.h"
#include "poly/isl_emitter_csr.h"
#include "poly/cpu_isl_emitter.h"
#include "poly/npu_isl_emitter.h"
#include "poly/gpu_emit/gpu_isl_emitter.h"
#include "poly/gpu_emit/gpu_isl_emitter_reduce.h"
#include "poly/gpu_emit/gpu_isl_emitter_tensor_core.h"
#include "poly/dsa_mgr_strategy.h"
#include "poly/gpu_mgr_strategy.h"
#include "poly/cpu_mgr_strategy.h"
#include "poly/schedule_pass_mgr.h"
#include "build_module.h"

namespace akg {
namespace ir {
namespace poly {
void Scop::ParseUserConfig(std::string target, const Map<Tensor, Buffer> &extern_buffer,
                           const Map<std::string, NodeRef> &spec_gemm_attrs, bool is_tuning, bool is_dynamic,
                           const Schedule &sch) {
  info_.user_config_.SetTarget(target);
  if (info_.user_config_.GetTarget() == TARGET_CCE) {
    info_.user_config_.SetEnableRestart(true);
  }
  if (spec_gemm_attrs.empty()) {
    info_.user_config_.SetAttrs(g_attrs);
    info_.mmu_info_.SetAttrs(g_attrs);
  } else {
    info_.user_config_.SetAttrs(spec_gemm_attrs);
    info_.mmu_info_.SetAttrs(spec_gemm_attrs);
    info_.mmu_info_.SetSpecGemm(true);
    info_.mmu_info_.SetConvAttrInfo(spec_gemm_attrs);
  }
  
  info_.user_config_.SetBind(extern_buffer);
  info_.user_config_.SetOriginBind(extern_buffer);
  info_.user_config_.SetIsTuning(is_tuning);
  info_.user_config_.SetDynamic(is_dynamic);
  info_.user_config_.SetScheduleInfo(sch);
  if (g_attrs.GetBool("is_csr", false)) {
    info_.analysis_result_.SetCsr(true);
  }
}

isl::set CreateParamsSet(ScopInfo &info) {
  auto space = CreateParamsSpace(info.GetCtx(), info.user_config_.GetParams());
  auto context = isl::set::universe(space);
  auto dynamic_shape = info.user_config_.GetDynamicShape();
  auto params = info.user_config_.GetParams();
  for (const auto &param : params) {
    isl::aff aff(isl::aff::param_on_domain(space, isl::id(info.GetCtx(), param.second->name_hint)));
    context = context & (aff > 0);
    if (dynamic_shape.empty()) {
      continue;
    }
    for (const auto &ds : dynamic_shape) {
      if (auto dsn = ds.as<air::DynamicShapeNode>()) {
        if (dsn->tensor_name == param.second->name_hint) {
          context = context & (aff < dsn->poly_upper_bound);
        }
      }
    }
  }
  return context;
}

isl::schedule Scop::GenIsl() {
  auto outer_let_stmts = info_.user_config_.GetOuterLetStmts();
  body_ = PeelOuterLetStmt(body_, outer_let_stmts);
  info_.user_config_.SetOuterLetStmts(outer_let_stmts);
  info_.user_config_.CollectParams();
  auto params = info_.user_config_.GetParams();
  if (!params.empty()) {
    auto mutator = ConsolidateExprMutator(params);
    body_ = mutator.Mutate(body_);

    Binds new_binds;
    auto binds = info_.user_config_.GetBind();
    for (auto &it : binds) {
      Array<Expr> shape = it.first->shape;
      for (size_t i = 0; i < shape.size(); ++i) {
        if (!is_const(shape[i])) {
          shape.Set(i, mutator.Mutate(shape[i]));
        }
      }
      Tensor t = TensorNode::make(shape, it.first->dtype, it.first->op, it.first->value_index);

      shape = it.second->shape;
      for (size_t i = 0; i < shape.size(); ++i) {
        if (!is_const(shape[i])) {
          shape.Set(i, mutator.Mutate(shape[i]));
        }
      }
      Buffer b = BufferNode::make(it.second->data, it.second->dtype, shape, it.second->strides, it.second->elem_offset,
                                  it.second->name, it.second->scope, it.second->data_alignment,
                                  it.second->offset_factor, it.second->buffer_type);

      new_binds.Set(t, b);
    }
    info_.user_config_.SetBind(new_binds);
  } else if (!g_csr.empty()) {
    for (const auto& it: g_csr) {
      if (auto var = it.first.as<Variable>()) {
        params.emplace(var->name_hint, air::Downcast<Var>(it.first));
      }
    }
  }

  isl::space param_space = CreateParamsSpace(ctx_, params);
  isl::set param_set = CreateParamsSet(info_);

  info_.user_config_.SetBody(body_);
  Stmt stmt = body_;
  // Make schedule
  isl::schedule schedule_tmp = MakeScheduleTree(param_space, param_set, stmt, info_);
  info_.CreateDataFlow();
  info_.mmu_info_.UpdateComputeAttrInfo();
  info_.mmu_info_.ComputeByPassL1();
  return schedule_tmp;
}

isl::schedule Scop::Transform(const isl::schedule &input_schedule) {
  auto final_schedule = input_schedule;
  SchedulePassMgr mgr(info_);
  std::shared_ptr<PassMgrStrategy> pass_stra(nullptr);
  info_.user_config_.SetConsiderCoincidence(true);

  if (info_.user_config_.GetTarget() == TARGET_CCE) {
    pass_stra.reset(new DsaMgrStrategy(info_));
  } else if (info_.user_config_.GetTarget() == TARGET_CUDA) {
    pass_stra.reset(new GPUMgrStrategy(info_));
  } else if (info_.user_config_.GetTarget() == TARGET_CPU) {
    pass_stra.reset(new CPUMgrStrategy(info_));
  }
  final_schedule = mgr.Run(input_schedule, pass_stra);
  info_.DumpTransform("dsa_transfrom.log", pass_stra->pass_info_);

  // We offer a restart mechanism for scalar stmt that cannot tile: do not consider coincidence
  // and re-compute/re-tile to generate final schedule.
  if (mgr.need_restart_ && info_.user_config_.GetEnableRestart()) {
    info_.user_config_.SetConsiderCoincidence(false);
    if (info_.user_config_.GetTarget() == TARGET_CCE) {
      pass_stra.reset(new DsaMgrStrategy(info_));
    } else if (info_.user_config_.GetTarget() == TARGET_CUDA) {
      pass_stra.reset(new GPUMgrStrategy(info_));
    } else if (info_.user_config_.GetTarget() == TARGET_CPU) {
      pass_stra.reset(new CPUMgrStrategy(info_));
    }
    if ((info_.user_config_.GetTarget() == TARGET_CUDA) && (info_.analysis_result_.GetEnabledAutoTiling())) {
      auto block_cfg = info_.user_config_.GetBlockConfig();
      if (block_cfg) {
        block_cfg->Reset();
      }
      auto thread_cfg = info_.user_config_.GetThreadConfig();
      if (thread_cfg) {
        thread_cfg->Reset();
      }
    }
    final_schedule = mgr.Run(input_schedule, pass_stra);
    info_.DumpTransform("scalar_transform.log", pass_stra->pass_info_);
  }

  return final_schedule;
}  // namespace poly

isl::id_list CreateIteratorList(const isl::schedule &schedule_iter, const std::string &prefix) {
  int depth = 0;
  auto root = schedule_iter.root();
  auto fn = [&depth](const isl::schedule_node &node) -> isl::schedule_node {
    if (node.as<isl::schedule_node_band>()) {
      auto schedule_depth = static_cast<int>(node.schedule_depth());
      schedule_depth = schedule_depth + static_cast<int>(node.as<isl::schedule_node_band>().n_member());
      depth = schedule_depth > depth ? schedule_depth : depth;
    }
    return node;
  };
  root = root.map_descendant_bottom_up(fn);
  isl::id_list res(root.ctx(), depth);

  for (int i = 0; i < depth; ++i) {
    std::stringstream ss;
    ss << prefix << i;
    res = res.add(isl::id(root.ctx(), ss.str()));
  }
  return res;
}

size_t &AstNodeNum() {
  static thread_local size_t n = 0;
  return n;
}
constexpr auto AST_NODE_ID_PREFIX = "__node_";
Stmt GenHalide(ScopInfo &info, const isl::schedule &sch, bool used_for_tile_out_band) {
  if (sch.get()) info.analysis_result_.SetTransformedSchedule(sch);
  if (!used_for_tile_out_band) {
    // we should check the return value to be isl_stat_ok, but it returns isl_stat_error, so we skip this check.
    static_cast<void>(isl_options_set_ast_build_group_coscheduled(sch.ctx().get(), isl_bool_true));
    if (info.mmu_info_.IsConv()) info.mmu_info_.CreateConvModel();
  }

  NodeInfoRepo node_info_repo;
  auto gather = [&node_info_repo](const isl::ast_node &node, const isl::ast_build &build) -> isl::ast_node {
    auto schedule_map = isl::map::from(build.get_schedule());

    auto node_id = isl::id(node.ctx(), std::string(AST_NODE_ID_PREFIX) + std::to_string(AstNodeNum()++));
    CHECK_EQ(0u, node_info_repo.count(node_id)) << "node already exists: " << node_id;

    auto &node_info = node_info_repo[node_id];
    node_info.iterator_map = isl::pw_multi_aff(schedule_map.reverse());
    node_info.build = build;
    return node.set_annotation(node_id);
  };

  // set up ast builder
  auto builder = isl::ast_build(sch.ctx());
  if (info.user_config_.GetTarget() == TARGET_CPU) {
    builder = builder.set_eliminate_for(false);
  }
  // Keep the for whose extent is 1, and the subsequent processing flow will eliminate the for.
  if (info.user_config_.GetTarget() == TARGET_CCE && !info.user_config_.GetIsDynamic()) {
    builder = builder.set_eliminate_for(false);
  }
  builder = builder.set_at_each_domain(gather);

  auto iter_prefix = info.user_config_.GetIterPrefix(info.mmu_info_.IsSpecGemm());
  isl::id_list iters = CreateIteratorList(sch, iter_prefix);
  builder = builder.set_iterators(iters);

  // build processing
  std::chrono::high_resolution_clock::time_point timer_start;
  TIMER_START;
  auto ast_node = builder.node_from(sch);
  TIMER_SHOW("NodeFrom", std::string(info.mmu_info_.IsSpecGemm() ? "_specgemm" : ""));

  ast_node = CanonicalizeBlockInAst(ast_node);

  if (PRINT_EMITTER) {
    PrintHeader("FINAL SCHEDULE");
    std::cout << PrettyPrintSchTree(sch) << std::endl;
    PrintHeader("FINAL ASTNODE");
    std::cout << FormatMupaStr(ast_node.to_str(), false) << std::endl << std::endl;
    PrintHeader("FINAL ASTNODE TO C");
    std::cout << ast_node.to_C_str() << std::endl;
  }
  TIMER_START;
  Stmt stmt;
  if (PRINT_ISL_EMITTER) {
    if (used_for_tile_out_band) {
      if (info.user_config_.GetTarget() == TARGET_CCE) {
        PrintHeader("NPUIslEmitter");
        stmt = NPUIslEmitter(info, node_info_repo, iters).Emit(ast_node);
      } else if (info.user_config_.GetTarget() == TARGET_CUDA) {
        PrintHeader("GpuIslEmitter");
        if (info.analysis_result_.GetUseGpuReduceLib()) {
          stmt = GpuIslEmitterReduce(info, node_info_repo, iters).Emit(ast_node);
        } else if (info.user_config_.GetEnableTensorCore()) {
          stmt = GpuIslEmitterTensorCore(info, node_info_repo, iters).Emit(ast_node);
        } else {
          stmt = GpuIslEmitter(info, node_info_repo, iters).Emit(ast_node);
        }
      } else if (info.user_config_.GetTarget() == TARGET_CPU) {
        PrintHeader("CpuIslEmitter");
        stmt = CpuIslEmitter(info, node_info_repo, iters).Emit(ast_node);
      }
    } else {
      PrintHeader("IslEmitter");
      stmt = IslEmitter(info, node_info_repo, iters).Emit(ast_node);
    }
  } else {
    if (info.user_config_.GetTarget() == TARGET_CCE) {
      stmt = NPUIslEmitter(info, node_info_repo, iters).Emit(ast_node);
    } else if (info.user_config_.GetTarget() == TARGET_CUDA) {
      if (info.analysis_result_.GetUseGpuReduceLib() && info.analysis_result_.GetCsr()) {
        stmt = GpuIslEmitterCsrReduce(info, node_info_repo, iters).Emit(ast_node);
      } else if (info.analysis_result_.GetUseGpuReduceLib()) {
        stmt = GpuIslEmitterReduce(info, node_info_repo, iters).Emit(ast_node);
      } else if (info.user_config_.GetEnableTensorCore()) {
        stmt = GpuIslEmitterTensorCore(info, node_info_repo, iters).Emit(ast_node);
      } else if (info.analysis_result_.GetCsr()) {
        stmt = GpuIslEmitterCsr(info, node_info_repo, iters).Emit(ast_node);
      } else {
        stmt = GpuIslEmitter(info, node_info_repo, iters).Emit(ast_node);
      }
    } else if (info.user_config_.GetTarget() == TARGET_CPU) {
      if (info.analysis_result_.GetCsr()) {
        stmt = CpuIslEmitterCsr(info, node_info_repo, iters).Emit(ast_node);
      } else {
        stmt = CpuIslEmitter(info, node_info_repo, iters).Emit(ast_node);
      }
    }
  }

  TIMER_SHOW("IslEmitter", std::string(info.mmu_info_.IsSpecGemm() ? "_specgemm" : ""));

  if (PRINT_EMITTER) {
    PrintHeader("FINAL STMT");
    std::cout << stmt;
  }
  return stmt;
}

Stmt Scop::GenHalide(const isl::schedule &sch) { return poly::GenHalide(info_, sch, false); }

}  // namespace poly
}  // namespace ir
}  // namespace akg