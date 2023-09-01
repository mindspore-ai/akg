/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include "poly/tune_info_adapter.h"
#include "poly/tiling/hermes/check_visitor.h"

namespace akg {
namespace ir {
class LoopMinFixer : public IRMutator {
 public:
  LoopMinFixer() {}
  ~LoopMinFixer() override = default;

  Stmt Mutate_(const AttrStmt *op, const Stmt &s) final {
    if (op->attr_key == "pragma_reschedule") {
      in_reschedule_ = true;
      auto stmt = IRMutator::Mutate(op->body);
      in_reschedule_ = false;
      return stmt;
    }
    return IRMutator::Mutate_(op, s);
  }

  Stmt Mutate_(const For *op, const Stmt &s) final {
    Stmt stmt;
    if (!in_reschedule_) {
      return IRMutator::Mutate_(op, s);
    }
    if (op->min.as<IntImm>() && op->min.as<IntImm>()->value == 0) {
      stmt = IRMutator::Mutate_(op, s);
    } else {
      Expr extent = Substitute(op->extent, {{op->loop_var, Simplify(op->loop_var + op->min)}});
      Stmt body = Substitute(op->body, {{op->loop_var, Simplify(op->loop_var + op->min)}});
      body = CanonicalSimplify(body);
      body = Mutate(body);
      stmt = For::make(op->loop_var, 0, extent, op->for_type, op->device_api, body);
    }
    return stmt;
  }

 private:
  bool in_reschedule_{false};
};

Stmt FixLoopMin(Stmt stmt) {
  stmt = LoopMinFixer().Mutate(stmt);
  return stmt;
}

/*!
 * \brief Poly entry
 */
class Poly {
 public:
  Poly() : isl_ctx_(isl::ctx(isl_ctx_alloc())) {}

  ~Poly() noexcept {
    scop_->info_.user_config_.FreeReplaceConfig();
    scop_.reset();
    // scop must be deconstructed before isl_ctx is deconstructed
    isl_ctx_free(isl_ctx_.get());
  }

  void Run(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, std::string target,
           const Map<std::string, NodeRef> &spec_gemm_attrs, bool is_tuning, bool is_dynamic,
           const Schedule &origin_sch) {
    stmt_ = stmt;
    scop_.reset(new poly::Scop(Simplify_cce(stmt_), isl_ctx_));
    CHECK(scop_ != nullptr);
    scop_->ParseUserConfig(target, extern_buffer, spec_gemm_attrs, is_tuning, is_dynamic, origin_sch);
    bool is_spec_gemm = !spec_gemm_attrs.empty();

    if (scop_->info_.user_config_.IsSymbolicTiling(stmt)) {
      poly::CheckVisitor check_visit;
      check_visit.Clear();
      check_visit.Visit(stmt);
    }

    std::chrono::high_resolution_clock::time_point timer_start;
    // generate isl schedule from Halide
    TIMER_START;
    isl::schedule sch = scop_->GenIsl();
    TIMER_SHOW("GenIsl", std::string(is_spec_gemm ? "_specgemm" : ""));

    // isl schedule transform
    TIMER_START;
    isl::schedule sched = scop_->Transform(sch);
    TIMER_SHOW("Transform", std::string(is_spec_gemm ? "_specgemm" : ""));

    // generate Halide from isl schedule
    TIMER_START;
    if (is_tuning) {
      stmt_ = GenHalide(scop_->info_, sched, true);
    } else {
      stmt_ = scop_->GenHalide(sched);
    }
    TIMER_SHOW("GenHalide", std::string(is_spec_gemm ? "_specgemm" : ""));
    if (scop_->info_.user_config_.GetTarget() == TARGET_CCE) {
      stmt_ = FixLoopMin(stmt_);
    }

    if (is_dynamic) stmt_ = RestoreCombinedParams(stmt_, scop_->info_);

    if (is_tuning) {
      if (scop_->info_.user_config_.GetUseNewSpace()) {
        auto tune_info = GenerateTuningInfo(sched, &scop_->info_, stmt_);
        spaces_ = GenerateTuningSpace(tune_info.get(), scop_->info_.user_config_.GetDumpTuningLevel());
      } else {
        spaces_ = GenerateTilingSpace(sched, scop_->info_, stmt_, scop_->info_.user_config_.GetDumpTuningLevel());
      }
      return;
    }

    // optimize post poly Halide IR
    if (scop_->info_.user_config_.GetEnableFeatureLib() || scop_->info_.user_config_.GetOptimizeForNPU()) {
      stmt_ = poly::DsaHalideOptimizer(stmt_, !scop_->info_.user_config_.GetParams().empty());
    }
    if (scop_->info_.user_config_.GetTarget() == TARGET_CCE && scop_->info_.user_config_.GetFrontendLower()) {
      stmt_ = poly::DsaHalideFixer(stmt_, scop_->info_.analysis_result_.mmu_bias_init_c_ != nullptr);
    }
    gen_empty_tiling = scop_->info_.analysis_result_.GetIsTiled();
  }

  Stmt GetStmt() { return stmt_; }

  NodeRef GetSpaces() { return spaces_; }

  Array<Var> GetTilingParams() {
    CHECK(scop_ != nullptr);
    Array<Var> tiling_params_array;
    if (gen_empty_tiling) return tiling_params_array;
    std::unordered_set<Var, NodeHash, NodeEqual> tiling_params;
    auto param_tiling_map = scop_->info_.user_config_.GetParamTilingMap();
    for (const auto &kv : param_tiling_map) {
      GatherVars(kv.second, &tiling_params);
    }
    for (const auto &param : tiling_params) tiling_params_array.push_back(param);
    return tiling_params_array;
  }

  void GatherVars(const Expr expr, std::unordered_set<Var, air::NodeHash, air::NodeEqual> *vset) {
    PostOrderVisit(expr, [&vset](const NodeRef &node) {
      if (node.as<Variable>()) {
        vset->insert(Downcast<Var>(node));
      }
    });
  }

 private:
  std::unique_ptr<poly::Scop> scop_{nullptr};
  // define isl_ctx outside scop because there are a lot of isl objects in the members of scop class,
  // and we need to ensure that they are deconstructed before the isl_ctx is freed.
  isl::ctx isl_ctx_;
  Stmt stmt_;
  NodeRef spaces_;
  bool gen_empty_tiling{false};
};

/// Interface for lower pass
Array<NodeRef> AutoPoly(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, std::string target,
                        const bool is_dynamic, const Map<std::string, NodeRef> &spec_gemm_attrs, Schedule sch) {
  Poly poly;
  poly.Run(stmt, extern_buffer, target, spec_gemm_attrs, false, is_dynamic, sch);
  return Array<NodeRef>({poly.GetStmt(), poly.GetTilingParams()});
}

NodeRef GenTuningSpace(const Stmt &stmt, std::string target, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &spec_gemm_attrs, Schedule sch) {
  Poly poly;
  poly.Run(stmt, extern_buffer, target, spec_gemm_attrs, true, false, sch);
  return poly.GetSpaces();
}
}  // namespace ir
}  // namespace akg
