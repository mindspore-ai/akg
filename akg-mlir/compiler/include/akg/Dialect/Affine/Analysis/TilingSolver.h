/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_TILINGSOLVER_H_
#define COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_TILINGSOLVER_H_
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "akg/Dialect/Affine/Analysis/Axis.h"
#include "akg/Dialect/Affine/Analysis/Config.h"
#include "akg/Dialect/Affine/Analysis/Model.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace mlir {
namespace func {
class FuncOp;
}  // namespace func
namespace akg {
namespace autotiling {

using Rule = std::function<std::deque<int>(const AxisPtr &, std::vector<int> &)>;
class SolveTarget {
 public:
  explicit SolveTarget(const std::string &name) : name(name) {}
  void addRule(const Rule &rule) { rules.push_back(rule); }
  std::string name;
  std::vector<Rule> rules;
};
using SolveTargetPtr = std::shared_ptr<SolveTarget>;

class TilingSolver {
 public:
  explicit TilingSolver(const ModelGraphPtr modelGraph) : modelGraph(modelGraph) {}
  virtual ~TilingSolver() = default;

  void initMinSize();

  std::vector<AxisPtr> sortAxis(size_t bandIdx);

  void solve(const AxisPtr a);

  int getTileSize(const AxisPtr a, std::vector<int> candidates);

  virtual bool genSolveTarget() { return false; }
  virtual std::vector<ConfigPtr> sortSolveTask(const AxisPtr &axis) = 0;
  ModelGraphPtr modelGraph;

  std::set<AxisPtr> solved;
  SolveTargetPtr target;
};
using TilingSolverPtr = std::shared_ptr<TilingSolver>;

class HeuristicTilingSolver : public TilingSolver {
 public:
  explicit HeuristicTilingSolver(const ModelGraphPtr modelGraph) : TilingSolver(modelGraph) {}
  virtual ~HeuristicTilingSolver() = default;

  bool genSolveTarget() override;
  std::vector<ConfigPtr> sortSolveTask(const AxisPtr &axis) override;
};

class GlobalConfigSolver {
 public:
  explicit GlobalConfigSolver(const TilingSolverPtr tilingSolver)
      : modelGraph(tilingSolver->modelGraph), tilingSolver(tilingSolver) {}
  virtual ~GlobalConfigSolver() = default;

  void solve(func::FuncOp funcOp);
  void UpdateGlobalInfo(func::FuncOp funcOp);
  void setEnableVectorize();

  ModelGraphPtr modelGraph;
  std::vector<std::pair<std::string, int>> solveMapResource(const AxisPtr axis);
  std::map<AxisPtr, std::vector<std::pair<std::string, int>>> globalAlloc();

 private:
  TilingSolverPtr tilingSolver;
};

}  // namespace autotiling
}  // namespace akg
}  // namespace mlir

#endif  // COMPILER_INCLUDE_AKG_DIALECT_AFFINE_ANALYSIS_TILINGSOLVER_H_

