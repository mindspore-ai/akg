/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "poly/scop_builder.h"
#include "poly/transform.h"
#include "poly/cce_isl_emitter.h"

namespace akg {
namespace ir {
namespace poly {
Scop::Scop(Stmt body, const Binds &binds, isl::ctx ctx, bool is_spec_gemm)
    : body_(std::move(body)),
      binds_(binds),
      binds_orig_(binds),
      ctx_(ctx),
      is_spec_gemm_(is_spec_gemm),
      isolated_(false),
      isolated_idx_(0) {
  if (is_spec_gemm) {
    iter_prefix_ = kGemmIterNamePrefix;
  } else {
    iter_prefix_ = kIterNamePrefix;
  }
}

Scop::~Scop() {
  if (model_ != nullptr) {
    delete model_;
    model_ = nullptr;
  }
}

isl::set Scop::CreateParamsSet() const {
  auto space = CreateParamsSpace(ctx_, params_);
  auto context = isl::set::universe(space);

  for (const auto &param : params_) {
    isl::aff aff(isl::aff::param_on_domain(space, isl::id(ctx_, param.second->name_hint)));
    context = context & (aff > 0);
    if (!dynamic_shape_.empty()) {
      for (const auto &ds : dynamic_shape_) {
        if (auto dsn = ds.as<air::DynamicShapeNode>()) {
          if (dsn->tensor_name == param.second->name_hint) {
            context = context & (aff < dsn->poly_upper_bound);
          }
        }
      }
    }
  }
  return context;
}

isl::schedule Scop::GenIsl() {
  body_ = PeelOuterLetStmt(body_, outer_let_stmts_);

  GetParams();
  if (!params_.empty()) {
    auto mutator = ConsolidateExprMutator(params_);
    body_ = mutator.Mutate(body_);

    Binds new_binds;
    for (auto &it : binds_) {
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
    binds_ = new_binds;
  }

  isl::space param_space = CreateParamsSpace(ctx_, params_);
  isl::set param_set = CreateParamsSet();

  // Make schedule
  Stmt stmt = body_;
  isl::schedule schedule_tmp = MakeScheduleTree(param_space, param_set, stmt, *this);
  return schedule_tmp;
}

isl::schedule Scop::Transform(isl::schedule sched, bool coincident, bool tuning) {
  auto timer_start = std::chrono::high_resolution_clock::now();
  CreateDataFlowInfo();
  DumpSchTree("00_before_group" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  bool has_group = false;
  isl::schedule sch = sched;
  if (!disable_group_) {
    sch = GroupStatements(sched, has_group);
    DumpSchTree("01_after_group" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sch);
  }

  // perform polyhedral transformation ongoing, gradually
  poly::Transform transform(sch, data_, *this, has_group);

  TIMER_START;
  data_.copyin = transform.ComputeCopyIn();
  TIMER_SHOW("computeCopyIn", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  CheckAndRemoveUninitializedCopyin(data_.copyin, binds_orig_);
  sch = transform.Initialize(coincident);

  if (outer_band_need_split_ && !is_spec_gemm_) {
    sch = SplitOuterBand(sch);
    DumpSchTree("06_splitOuterBand" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sch);
  }

  TIMER_START;
  data_.inter_band_dependency = transform.ComputeFakeCopyin(sch).subtract(data_.copyin);
  TIMER_SHOW("computeFakeCopyin", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  if (!is_spec_gemm_ && (IsConv() || IsGemm())) {
    this->sch_ = sch.get_map();
    isl::union_map fake_copyin = transform.ComputeFakeCopyin(sch);
    ComputeTransferCopyin(fake_copyin);
  }

  isl::schedule tiling_sch = sch;
  if (!is_spec_gemm_ && !data_.transfer_stmt.is_empty()) {
    TransferStmt(tiling_sch);
  }

  // get compute attr for conv and load3d op
  UpdateComputeAttrInfo();
  if (PRINT_SCHEDULE_INFO) LOG(INFO) << GenHalide(sch);

  // 4. tiling, an initial strategy, pending optimization
  Tiles tiles;
  if (tuning) {
    spaces_ = GenerateTilingSpace(this, sch, dump_tuning_level_, custom_tiling_, dynamic_shape_);
    return sch;
  }

  TIMER_START;
  InitDimensionInfo(tiling_sch);
  MergeTilingInfo(tiles);
  TIMER_SHOW("AutoTiling", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  if (IsConv()) CreateConvModel(is_dynamic_);

  TIMER_START;
  isl::schedule tmp_schedule = transform.TileOuterBand(tiles, sch);
  is_tiled_ = true;
  TIMER_SHOW("tileOuterBand", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  // for scalar stmt, keep going when coincident = false
  if (tmp_schedule.get_map().is_equal(sch.get_map()) && coincident) {
    LOG(WARNING) << "same schedule";
    return sched;
  }

  if (sch.plain_is_equal(tmp_schedule)) {
    tmp_schedule = transform.TryMarkScalarStmts(sch.get_root()).get_schedule();
  }

  sched = tmp_schedule;
  DumpSchTree("07_tileOuterBand" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);

  if (transform.HasInvariantDependence()) {
    sched = transform.ReorderInvariantSetSchedule(sched);
    DumpSchTree("07_01_reorderAfterTileOuterBand" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  }

  sched = ResetCoincidenceOfReduceAxis(sched, data_.reduce_stmts);
  if (pragma_set_all_coincident_) {
    sched = transform.SetAllCoincident(sched);
  }
  // 5. apply intra tile rescheduling
  if (!is_dynamic_ || !IsConv()) {
    TIMER_START;
    transform.IntraTileReschedule(sched, tile_inner_band_, is_spec_gemm_);
    TIMER_SHOW("IntraTileRescheduling", std::string(is_spec_gemm_ ? "_specgemm" : ""));
    DumpSchTree("08_0_reschedule" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  }

  sched = ReorderInnerBandLoops(sched);
  DumpSchTree("08_1_reorderInnerBandLoops" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  sched = ChangeMarkNodePosition(sched);
  DumpSchTree("08_2_changeMarkNodePos" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);
  sched = LabelRealizeOutPosition(sched);
  DumpSchTree("08_3_labelAlloc" + std::string(is_spec_gemm_ ? "_specgemm" : ""), sched);

  sched = InsertNodeForAllocC(sched);

  std::vector<std::vector<int>> partition_info = AddTileInfo(transform.getPartitionInfo());
  AddPartitionInfoToData(partition_info);

  ComputeByPassL1();

  this->schedule_ = sched;

  TIMER_START;
  AddStateTensorsDataFlow();
  ReorderBufferedDefInfos();
  RecordAllTensorBufferFootprintToExtension();
  if (enable_hoist_cond_write_) {
    FindConditionalWritePromotions();
  }
  TIMER_SHOW("MemoryPromotion", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  if (!is_spec_gemm_ && !data_.transfer_stmt.is_empty()) {
    TransferStmt(this->schedule_);
  }
  DumpSchTree("09_mem_promote" + std::string(is_spec_gemm_ ? "_specgemm" : ""), this->schedule_);

  this->schedule_ = ReorderMarkNodes(this->schedule_);
  DumpSchTree("10_reorderMarkNodes" + std::string(is_spec_gemm_ ? "_specgemm" : ""), schedule_);
  this->schedule_ = MarkFuseOp(this->schedule_);
  DumpSchTree("11_markFuseOp" + std::string(is_spec_gemm_ ? "_specgemm" : ""), this->schedule_);

  // if coincidence constraints are disabled (due to reschedule), we cannot determine multicore axis reliably
  bool can_use_multiCore = !is_spec_gemm_ && coincident;
  if (can_use_multiCore || enable_mark_multi_core_) {
    this->schedule_ = MarkOuterMost(this->schedule_);
    DumpSchTree("12_markOuterMost" + std::string(is_spec_gemm_ ? "_specgemm" : ""), this->schedule_);
  }
  return this->schedule_;
}

isl::id_list Scop::CreateIteratorList(const isl::schedule &schedule_iter, const std::string &prefix) {
  auto root = schedule_iter.root();
  auto fn = [this](const isl::schedule_node &node) -> isl::schedule_node {
    if (node.as<isl::schedule_node_band>()) {
      auto depth = static_cast<int>(node.schedule_depth());
      depth = depth + static_cast<int>(node.as<isl::schedule_node_band>().n_member());
      this->depth_ = depth > this->depth_ ? depth : this->depth_;
    }
    return node;
  };
  root = root.map_descendant_bottom_up(fn);
  isl::id_list res(root.ctx(), depth_);

  for (int i = 0; i < depth_; ++i) {
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
Stmt Scop::GenHalide(const isl::schedule &schedule_gen) {
  // we should check the return value to be isl_stat_ok, but it returns isl_stat_error, so we skip this check.
  static_cast<void>(isl_options_set_ast_build_group_coscheduled(schedule_.ctx().get(), isl_bool_true));

  NodeInfoRepo node_info_repo;
  auto gather = [&node_info_repo](const isl::ast_node &node, const isl::ast_build &build) -> isl::ast_node {
    auto fillUpRepo = [](const isl::ast_node &node, const isl::ast_build &build,
                         NodeInfoRepo *node_info_repo) -> isl::ast_node {
      CHECK(node_info_repo != nullptr);
      auto schedule_map = isl::map::from(build.get_schedule());

      auto node_id = isl::id(node.ctx(), std::string(AST_NODE_ID_PREFIX) + std::to_string(AstNodeNum()++));
      CHECK_EQ(0u, node_info_repo->count(node_id)) << "node already exists: " << node_id;

      auto &node_info = (*node_info_repo)[node_id];
      node_info.iterator_map = isl::pw_multi_aff(schedule_map.reverse());
      node_info.build = build;
      return node.set_annotation(node_id);
    };

    return fillUpRepo(node, build, &node_info_repo);
  };

  // set up ast builder
  auto builder = isl::ast_build(schedule_gen.ctx());
  builder = builder.set_at_each_domain(gather);

  isl::id_list iters = CreateIteratorList(schedule_gen, iter_prefix_);
  builder = builder.set_iterators(iters);

  // build processing
  std::chrono::high_resolution_clock::time_point timer_start;
  TIMER_START;
  auto ast_node = builder.node_from(schedule_gen);
  TIMER_SHOW("NodeFrom", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  ast_node = CanonicalizeBlockInAst(ast_node);

  TIMER_START;
  Stmt stmt = CCEIslEmitter(*this, node_info_repo, iters).Emit(ast_node);
  TIMER_SHOW("CCEIslEmitter", std::string(is_spec_gemm_ ? "_specgemm" : ""));

  if (is_dynamic_) {
    stmt = RestoreCombinedParams(stmt);
  }
  return stmt;
}

Stmt OptimizeHalide(const Stmt &s, bool dynamic_shape) { return OptimizeCce(s, dynamic_shape); }
}  // namespace poly
}  // namespace ir
}  // namespace akg
