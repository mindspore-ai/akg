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
#ifndef POLY_SCOP_H_
#define POLY_SCOP_H_

#include "poly/scop_info.h"
#include "poly/pass_info.h"
#include "auto_tune/tune_info.h"

namespace akg {
namespace ir {
namespace poly {
class Scop {
 public:
  Scop(Stmt body, isl::ctx ctx) : info_(ScopInfo(ctx)), body_(std::move(body)), ctx_(ctx) {}
  ~Scop() = default;

  void ParseUserConfig(std::string target, const Map<Tensor, Buffer> &extern_buffer,
                       const Map<std::string, NodeRef> &spec_gemm_attrs, bool is_tuning, bool is_dynamic,
                       const Schedule &sch);
  isl::schedule GenIsl();
  isl::schedule Transform(const isl::schedule &input_schedule);
  Stmt GenHalide(const isl::schedule &sch);

  ScopInfo info_;

 private:
  void ResetConfig();
  Stmt body_;
  isl::ctx ctx_;
  const int kBit32 = 32;
};

Stmt GenHalide(ScopInfo &info, const isl::schedule &, bool used_for_tile_out_band = false);
Stmt DsaHalideOptimizer(const Stmt &s, bool dynamic_shape = false);
Stmt RestoreCombinedParams(Stmt stmt, ScopInfo &info);
std::pair<TileSizes, std::deque<ParamInfo>> GenerateTiling(const isl::schedule &sch, ScopInfo &scop_info, Stmt body);
NodeRef GenerateTilingSpace(const isl::schedule &sch, ScopInfo &scop_info, Stmt body, int dump_level);
NodeRef GenerateTuningSpace(TuneInfo *tune_info, int dump_level);
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SCOP_H_
