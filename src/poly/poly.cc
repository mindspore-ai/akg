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
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

#include <memory>

#include "ir_pass.h"
#include "poly/scop.h"
#include "pass/utils.h"

namespace akg {
namespace ir {
/*!
 * \brief Poly entry
 */
class Poly {
 public:
  Poly() : isl_ctx_(isl::ctx(isl_ctx_alloc())) {}

  void Run(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer, const Map<std::string, NodeRef> &attrs,
           const bool is_spec_gemm, bool is_tuning, bool is_dynamic) {
    stmt_ = stmt;
    scop_.reset(new poly::Scop(Simplify_cce(stmt_), extern_buffer, isl_ctx_, is_spec_gemm));
    CHECK(scop_ != nullptr);

    scop_->SetAttrs(attrs);
    scop_->is_dynamic_ = is_dynamic;

    // generate isl schedule from Halide
    std::chrono::high_resolution_clock::time_point timer_start;
    TIMER_START;
    isl::schedule sch = scop_->GenIsl();
    TIMER_SHOW("GenIsl", std::string(is_spec_gemm ? "_specgemm" : ""));

    // transform isl schedule with coincidence constraints
    isl::schedule scht = scop_->Transform(sch, true, is_tuning);
    if (is_tuning) return;

    if (scht.get() == sch.get()) {
      // transform failed, redo transform without coincidence constraints
      scht = scop_->Transform(sch, false);
    }

    // generate Halide from isl schedule
    stmt_ = scop_->GenHalide(scht);

    // optimize post poly Halide IR for Davinci
    if (scop_->enable_feature_library_ || scop_->optimize_for_davinci_) {
      stmt_ = poly::OptimizeHalide(stmt_, !scop_->params_.empty());
    }
    gen_empty_tiling = scop_->is_tiled_;
  }

  ~Poly() noexcept {
    scop_.reset();
    // scop must be deconstructed before isl_ctx is deconstructed
    isl_ctx_free(isl_ctx_.get());
  }

  Stmt getstmt() { return stmt_; }
  bool gen_empty_tiling{false};
  Array<Var> getTilingParams() {
    CHECK(scop_ != nullptr);
    Array<Var> tiling_params_array;
    if (gen_empty_tiling) return tiling_params_array;
    std::unordered_set<Var, NodeHash, NodeEqual> tiling_params;
    for (const auto &kv : scop_->param_tiling_map_) {
      GatherVars(kv.second, &tiling_params);
    }
    for (const auto &param : tiling_params) tiling_params_array.push_back(param);
    return tiling_params_array;
  }

  NodeRef getspaces() {
    CHECK(scop_ != nullptr);
    return scop_->spaces_;
  }

 private:
  std::unique_ptr<poly::Scop> scop_{nullptr};
  // define isl_ctx outside scop because there are a lot of isl objects in the members of scop class,
  // and we need to ensure that they are deconstructed before the isl_ctx is freed.
  isl::ctx isl_ctx_;
  Stmt stmt_;
};

/// Interface for lower pass
Array<NodeRef> AutoPoly(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer,
                        const Map<std::string, NodeRef> &attrs, const bool is_specgemm, const bool is_dynamic) {
  Poly poly;
  poly.Run(stmt, extern_buffer, attrs, is_specgemm, false, is_dynamic);
  return Array<NodeRef>({poly.getstmt(), poly.getTilingParams()});
}

NodeRef GenTuningSpace(const Stmt &stmt, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &attrs, const bool is_specgemm) {
  Poly poly;
  poly.Run(stmt, extern_buffer, attrs, is_specgemm, true, false);
  return poly.getspaces();
}
}  // namespace ir
}  // namespace akg
