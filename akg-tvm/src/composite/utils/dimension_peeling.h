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

#ifndef COMPOSITE_UTILS_DIMENSION_PEELING_H_
#define COMPOSITE_UTILS_DIMENSION_PEELING_H_
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "composite/optimize/optimize.h"

namespace akg {

class AffinityAnalyzer : public IRVisitor {
 public:
  enum {
    AF_NONE = 0,
    AF_ELEMWISE,
    AF_BROADCAST,
    AF_REDUCE,
    AF_RESHAPE,
  };

  struct Tensor;
  struct Dim {
    Tensor *tensor{nullptr};
    int index{0};
    int64_t size{0};
    std::vector<std::pair<Dim *, int>> prod;
    std::vector<std::pair<Dim *, int>> cons;
  };
  struct Tensor {
    FunctionRef ref;
    std::string op;
    std::vector<Tensor *> prod;
    std::vector<Tensor *> cons;
    std::vector<std::unique_ptr<Dim>> dims;
  };

  AffinityAnalyzer() = default;
  ~AffinityAnalyzer() = default;

  void Analyze(Stmt stmt);

  void VisitProd(Dim *dim, std::function<bool(Dim *, Dim *, int)> fun);
  void VisitCons(Dim *dim, std::function<bool(Dim *, Dim *, int)> fun);

  void Dump(std::ostringstream &os);

  void Visit_(const AttrStmt *op) override;
  void Visit_(const Provide *op) override;

  std::vector<std::unique_ptr<Tensor>> tensors_;

 private:
  std::unordered_map<FunctionRef, Tensor *, NodeHash, NodeEqual> tensor_map_;
  Map<std::string, NodeRef> attrs_;

  void AddElemBroadRelation(Tensor *input, Tensor *output);
  void AddReduceRelation(Tensor *input, Tensor *output);
  void AddTransposeRelation(Tensor *input, Tensor *output);

  Tensor *NewTensor(FunctionRef ref, std::string op, Array<Expr> shape);
};

class DimensionPeeler {
 public:
  using Peeling = std::vector<std::pair<int, int64_t>>;  // dim, split_val
  using Tensor = AffinityAnalyzer::Tensor;
  using Dim = AffinityAnalyzer::Dim;

  DimensionPeeler() = default;
  ~DimensionPeeler() = default;

  void Analyze(Stmt s);
  std::vector<int64_t> GetAxisSpace();
  std::vector<Peeling> GetPeelSpace(int limit_depth = 0, std::unordered_set<int> *limit_range = nullptr);
  Stmt GetPeelBody(std::unordered_map<std::string, Peeling> config);
  Stmt GetPeelBody(const Peeling &peeling);
  Peeling GetPeelDims(const std::string &tensor, const Peeling &peeling);
  std::unordered_map<std::string, Peeling> GetPeelTensors(const Peeling &peeling);

 private:
  struct Axis {
    int size{0};
    std::vector<int64_t> peel_val;
  };

  Stmt stmt_;
  std::vector<std::unique_ptr<Axis>> axis_space_;
  // axis_idx -> dim_idx
  std::unordered_map<std::string, std::vector<int>> dim_map_;

  std::vector<int64_t> GetDivisors(int64_t n);
  Tensor *BuildAxisSpace(AffinityAnalyzer &aff);
  void MapDimToSpace(AffinityAnalyzer &aff, Dim *dom_dim, int axis_idx);
  bool Propagation(int axis_idx, Dim *from, Dim *to, int affinity);
  void AddDimMap(Dim *dim, int axis_idx);
};

DimensionPeeler::Peeling Str2Peeling(const std::string &peeling);
Expr Peeling2Str(const DimensionPeeler::Peeling &peeling);
}  // namespace akg
#endif  // COMPOSITE_UTILS_DIMENSION_PEELING_H_
