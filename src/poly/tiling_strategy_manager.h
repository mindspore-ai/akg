/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef POLY_TILING_STRATEGY_MANAGER_H_
#define POLY_TILING_STRATEGY_MANAGER_H_

#include <iostream>

#include "poly/tiling_analyzer.h"

namespace akg {
namespace ir {
namespace poly {
class TilingStrategy {
 public:
  explicit TilingStrategy(const TilingAnalyzer *a) : analyzer_(a) {}
  ~TilingStrategy() {}
  virtual void AddConstraint(){};
  std::string interested_attr_key;

 protected:
  const TilingAnalyzer *analyzer_;

  std::unordered_map<TileAxis *, std::vector<AttrInfo>> GetInterestedInfo(const std::string &attr_key,
                                                                          bool match_whole_word = true);
};

class TilingStrategyManager {
 public:
  ~TilingStrategyManager() {}

  static TilingStrategyManager &GetInstance() {
    static TilingStrategyManager strategy_manager_;
    return strategy_manager_;
  }

  void SetStrategies(std::vector<TilingStrategy *> strategies) {
    this->strategies_.assign(strategies.begin(), strategies.end());
  }
  std::vector<TilingStrategy *> GetStrategies() { return this->strategies_; }

  void Execute() {
    for (auto strategy : this->strategies_) {
      strategy->AddConstraint();
    }
  }

 private:
  TilingStrategyManager() {}
  std::vector<TilingStrategy *> strategies_;
};

class CustomTilingStrategy : public TilingStrategy {
 public:
  explicit CustomTilingStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~CustomTilingStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "CUSTOM";
};

class ConflictTreeRangeStrategy : public TilingStrategy {
 public:
  explicit ConflictTreeRangeStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ConflictTreeRangeStrategy() {}
  void AddConstraint();
};

class ModStrategy : public TilingStrategy {
 public:
  explicit ModStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ModStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "MOD";
};

// These strategies aim to deal with special insn in Davinci core.
class CastStrategy : public TilingStrategy {
 public:
  explicit CastStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~CastStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "CAST";
};

class ReduceStrategy : public TilingStrategy {
 public:
  explicit ReduceStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ReduceStrategy() {}
  void AddConstraint();
};

class VectorizedStrategy : public TilingStrategy {
 public:
  explicit VectorizedStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~VectorizedStrategy() {}
  void AddConstraint();
};

class TensorOfTensorStrategy : public TilingStrategy {
 public:
  explicit TensorOfTensorStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~TensorOfTensorStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "CAST";
};

class PassDownAttrStrategy : public TilingStrategy {
 public:
  explicit PassDownAttrStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~PassDownAttrStrategy() {}
  void AddConstraint();
};

class DynamicShapeLimitStrategy : public TilingStrategy {
 public:
  explicit DynamicShapeLimitStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~DynamicShapeLimitStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "DYN_SHAPE_LIMIT";
};

class ShiftAxisStrategy : public TilingStrategy {
 public:
  explicit ShiftAxisStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ShiftAxisStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "SHIFT";
};

class ModShiftAxisStrategy : public TilingStrategy {
 public:
  explicit ModShiftAxisStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ModShiftAxisStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "MODSHIFT";
};

class DynamicBoundStrategy : public TilingStrategy {
 public:
  explicit DynamicBoundStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~DynamicBoundStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "DYNAMIC_BOUND";
};

class ConvStrategy : public TilingStrategy {
 public:
  explicit ConvStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~ConvStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "CONV";

  std::unordered_map<std::string, Expr> conv_info_{};
  ktvm::arith::Analyzer arith_ana_;

  void RestrainH(TileAxis *axis);
  void RestrainW(TileAxis *axis);
};

class GemmStrategy : public TilingStrategy {
 public:
  explicit GemmStrategy(const TilingAnalyzer *a) : TilingStrategy(a) {}
  ~GemmStrategy() {}
  void AddConstraint();

  std::string interested_attr_key = "GEMM";
};

class MulticoreStrategy {
 public:
  MulticoreStrategy(TileCandidate &cand, const std::string log_file)
      : cand_(cand), logger_(TileLogger::GetInstance(log_file)) {}
  ~MulticoreStrategy() {}
  std::pair<int, int> GetProposalRangeForFullMulticore(TileAxis *axis);
  int64_t AdjustTilingAccordingToMulticoreConstraint(TileAxis *axis, int64_t tiling_factor);

 private:
  TileCandidate &cand_;
  TileLogger &logger_;
};
}  // namespace poly
}  // namespace ir
}  // namespace akg
#endif  // POLY_TILING_STRATEGY_MANAGER_H_
