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

#ifndef POLY_SPACE_ANALYZER_H_
#define POLY_SPACE_ANALYZER_H_

#include <tvm/ir.h>

#include <vector>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "pass/common.h"
#include "poly/tiling_analyzer.h"
#include "poly/custom_tiling.h"

namespace akg {
namespace ir {
namespace poly {
class SpaceAnalyzer {
 public:
  explicit SpaceAnalyzer(TilingAnalyzer *analyzer) : analyzer_(analyzer) {}
  ~SpaceAnalyzer() {}
  using VarNames = TilingAnalyzer::VarNames;

  // represent a tensor
  // e.g. for(cc0,0,8){input_red(cc0)=0;}
  // => Tensor{name:input_red,loops:[cc0]}
  struct Tensor {
    std::string name;
    Array<Expr> args;
    std::vector<VarNames> var_names;
    std::unordered_map<size_t, std::vector<const For *>> loops;
    size_t band_index{0};
    int type_byte{1};
  };
  // represent a provide stmt
  struct ProvideEntry {
    std::string basic_op_type;
    std::unordered_set<int> pipe;
    std::vector<Tensor> src;
    Tensor dst;
    size_t band_index{0};
    const Provide *op{nullptr};
    const IfThenElse *cond{nullptr};
  };

  void AnalyzeSpecialAxes();

 private:
  TilingAnalyzer *analyzer_{nullptr};
  // Provides stmt after analysis.
  std::unordered_map<const For *, std::vector<ProvideEntry>> provides_ana_;

  const For *GetBufferInnerAxis(Tensor t, int offset = 1);
  // generalized cases
  void IdentifyInsnType();
  void IdentifyDmaUnderCondition();
  void IdentifySpecialAxes();
  void IdentifySharedAxes() const;
  void IdentifyVectorizedAxes();
  void IdentifyAlignAxes();
  void IdentifyReduceAxes();
  void IdentifyCastAxes();
  void IdentifyModAxes();
  std::vector<Expr> FindModConstraint(const Expr &arg, std::vector<Expr> constraints);
  // customized cases
  void IdentifyDynamicShape();
  void IdentifyCustomTiling();
  void SetAttrForAxis(int tile_band, int tile_axis, const std::string &attr_key, const std::string &attr_value);
  void SetAttrForTensor(const std::string &tensor_name, int pos, const std::string &attr_key,
                        const std::string &attr_value);
  std::string ParseAllTypeExpr(const Expr constraint);
};
}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_SPACE_ANALYZER_H_
