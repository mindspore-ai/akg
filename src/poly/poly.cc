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
namespace akg {
namespace ir {
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
           const Map<std::string, NodeRef> &attrs, const bool is_spec_gemm, bool is_tuning, bool is_dynamic,
           const Schedule &origin_sch) {
    stmt_ = stmt;
    scop_.reset(new poly::Scop(Simplify_cce(stmt_), isl_ctx_));
    CHECK(scop_ != nullptr);
    scop_->ParseUserConfig(target, attrs, extern_buffer, is_spec_gemm, is_tuning, is_dynamic, origin_sch);

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
    stmt_ = scop_->GenHalide(sched);
    TIMER_SHOW("GenHalide", std::string(is_spec_gemm ? "_specgemm" : ""));

    if (is_dynamic) stmt_ = RestoreCombinedParams(stmt_, scop_->info_);

    if (is_tuning) {
      if (scop_->info_.user_config_.GetUseNewSpace()) {
        spaces_ = GenerateTuningSpace(sched, scop_->info_, stmt_, scop_->info_.user_config_.GetDumpTuningLevel()); 
      } else {
        spaces_ = GenerateTilingSpace(sched, scop_->info_, stmt_, scop_->info_.user_config_.GetDumpTuningLevel());
      }
      return;
    }

    // optimize post poly Halide IR
    if (scop_->info_.user_config_.GetEnableFeatureLib() || scop_->info_.user_config_.GetOptimizeForNPU()) {
      stmt_ = poly::DsaHalideOptimizer(stmt_, !scop_->info_.user_config_.GetParams().empty());
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
                        const Map<std::string, NodeRef> &attrs, const bool is_specgemm, const bool is_dynamic,
                        Schedule sch) {
  Poly poly;
  poly.Run(stmt, extern_buffer, target, attrs, is_specgemm, false, is_dynamic, sch);
  return Array<NodeRef>({poly.GetStmt(), poly.GetTilingParams()});
}

NodeRef GenTuningSpace(const Stmt &stmt, std::string target, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &attrs, const bool is_specgemm, Schedule sch) {
  Poly poly;
  poly.Run(stmt, extern_buffer, target, attrs, is_specgemm, true, false, sch);
  return poly.GetSpaces();
}
}  // namespace ir
}  // namespace akg
