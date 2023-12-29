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
#ifndef COMPOSITE_OPTIMIZE_ELIM_RESHAPE_H_
#define COMPOSITE_OPTIMIZE_ELIM_RESHAPE_H_
#include "composite/optimize/optimize.h"

namespace akg {
class ElimReshapeAnalysis {
 public:
  ElimReshapeAnalysis(Graph &g, BuildOpt &opt, AnalysisResult &result, bool forward)
      : g_(g), opt_(opt), result_(result), forward_(forward){};
  bool Run();

 private:
  void AnalysisForward();
  void AnalysisBackkward();
  void AnalysisTransform(const FunctionRef &output);
  bool AnalysisElemwise(const FunctionRef &output);
  bool AnalysisElemwiseBackward(const FunctionRef &output);
  bool AnalysisElemwiseForward(const FunctionRef &output);
  void AnalysisOthers(const FunctionRef &output);
  void AnalysisInplaceAssign(const FunctionRef &output);
  void AnalysisInner(const FunctionRef &output);
  bool AnalysisElimValid();
  int ElimForwardEasier();
  bool ForwardHasOtherOp(const FuncRefList &funcs, FuncBoolMap &cache_res);

 private:
  Graph &g_;
  BuildOpt &opt_;
  AnalysisResult &result_;
  bool forward_{false};
};

class ElimReshapeOpChecker : public IRVisitor {
 public:
  ElimReshapeOpChecker() = default;
  void Visit_(const Provide *op) override {
    auto op_name = GetOpName(op);
    if (!(IsTransform(op_name) || IsInplaceAssign(op_name) || IsAssign(op_name))) {
      can_elim = true;
    }
  }
  bool can_elim{false};
};

}  // namespace akg
#endif  // COMPOSITE_OPTIMIZE_ELIM_RESHAPE_H_
