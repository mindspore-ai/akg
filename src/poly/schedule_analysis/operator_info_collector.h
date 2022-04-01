/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef OPERATOR_INFO_COLLECTOR_H
#define OPERATOR_INFO_COLLECTOR_H

#include "poly/scop_info.h"

namespace akg {
namespace ir {
namespace poly {

struct ToTTensor {
  std::string name;
  std::set<std::string> loop_vars;
  std::vector<int64_t> indices;
};

class OpTypeCollector : public IRVisitor {
 public:
  OpTypeCollector(ScopInfo &scop_info, const Stmt &stmt) : scop_info_(scop_info), stmt_(stmt) {
    target_ = scop_info.user_config_.GetTarget();
  }
  ~OpTypeCollector() override = default;

  void Run();
  void Collect();

  // Provides stmt after analysis.
  std::unordered_map<const For *, std::vector<ProvideEntry>> provides_ana_;
  TensorEntry count_op_tensor_;

 private:
  void WriteToScopInfo();
  void Dump();

  void Visit_(const AttrStmt *op) final;
  void Visit_(const Realize *op) final;
  void Visit_(const Provide *op) final;
  void Visit_(const For *op) final;
  void Visit_(const IfThenElse *op) final;
  void Visit_(const Evaluate *op) final;

  void AnalyzeProvide(const Provide *op);
  TensorEntry MatchLoopByName(TensorEntry tensor);
  std::string GetSingleOpType(const TensorEntry &dst, const TensorEntry &srcs);
  std::string GetBasicOpType(const TensorEntry &dst, const std::vector<TensorEntry> &srcs);
  TensorEntry MakeTensorEntry(const ToTTensor &tot);
  size_t CountUniqueLoopName(std::vector<VarNames> &var_names, bool count_num = false);
  bool IsTranspose(std::vector<VarNames> &dst_vars, std::vector<VarNames> &src_vars);
  std::string GetFusedCaseType(const TensorEntry &d, const TensorEntry &s);
  TensorEntry GetDstTensor(const Provide *op);
  std::vector<TensorEntry> GetSourceTensors(const Provide *op);
  std::string InitBasicOpType(const Provide *op);

  std::string target_;
  ScopInfo &scop_info_;
  const Stmt stmt_;
  const For *cur_loop_{nullptr};
  const AttrStmt *cur_attr_{nullptr};
  const IfThenElse *cur_if_{nullptr};
  std::vector<const air::ir::For *> cur_band_;
  int loop_count_ = 0;
  size_t band_count_ = 0;
  std::unordered_set<std::string> local_buf_;
  air::arith::Analyzer arith_ana_;
};

}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // OPERATOR_INFO_COLLECTOR_H