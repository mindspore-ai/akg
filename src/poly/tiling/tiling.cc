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
#include <tvm/ir_visitor.h>

#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>

#include "poly/scop_info.h"
#include "poly/poly_util.h"
#include "poly/tiling/tiling_analyzer.h"
#include "poly/tiling/tiling_algorithm.h"
#include "poly/tiling/tiling_strategy_manager.h"
#include "poly/tiling/tiling_solver.h"

namespace akg {
namespace ir {
namespace poly {
class TilingGenerator {
 public:
  explicit TilingGenerator(TilingAnalyzer &analyzer) : analyzer_(analyzer) {}
  ~TilingGenerator() = default;
  using DynamicMemInfo = TileCandidate::DynamicMemInfo;

  struct TileInfo {
    TileInfo(TileAxis *a, TileLevel l, int b) : axis(a), level(l), band(b) {}
    TileAxis *axis;
    TileLevel level;
    int band;
    int64_t min_tile = 0;
    int64_t deviation = 0;
  };

  struct ParamReplacement {
    std::vector<int64_t> mul_tile;
    std::vector<int64_t> mod_tile;
    std::vector<int64_t> l0_tile;
  };

  TileSizes Generate() {
    TraverseSolver solver(analyzer_);
    this->cand_ = solver.Solve();
    return ConvertToDims();
  }

  TileSizes GenerateQuickly() {
    InequalitySolver solver(analyzer_);
    this->cand_ = solver.Solve();
    ConvertVarTilesToDims();
    return dims_;
  }

  std::pair<TileSizes, std::deque<ParamInfo>> GenerateDynamic() {
    DynamicShapeSolver solver(analyzer_);
    this->cand_ = solver.Solve();
    param_info_ = solver.GetParamInfo();
    param_replacement_ = CreateVarTileReplaceMap();
    size_t before = param_replacement_.l0_tile.size();
    ConvertVarTilesToDims();
    size_t after = param_replacement_.l0_tile.size();
    if (before > after) {
      LOG(INFO) << "========= Test Tiling Strategy ============";
      Stmt stmt = Evaluate::make(0);
      for (auto info = param_info_.rbegin(); info != param_info_.rend(); ++info) {
        if (info->type_key == "AttrStmt") {
          auto attr_key = info->key.as<StringImm>();
          CHECK(attr_key);
          stmt = AttrStmt::make(make_zero(Int(32)), attr_key->value, info->value, stmt);
        } else if (info->type_key == "LetStmt") {
          stmt = LetStmt::make(air::Downcast<Var>(info->key), info->value, stmt);
        } else {
          analyzer_.logger_.LogFatalAndSaveLog("Unsupported type_key for now: " + info->type_key);
        }
      }
      LOG(INFO) << stmt;
    } else {
      param_info_.clear();
    }
    return std::make_pair(dims_, param_info_);
  }

 private:
  TileSizes ConvertToDims() {
    TileSizes dims;

    auto Convert = [this, &dims](TileAxis *axis) {
      if (axis->index < 0) return;
      if (axis->is_inner && !axis->is_pragma) return;
      Expr l1_val = 1;
      Expr l0_val = 1;
      DimensionInfo dimInfo;
      dimInfo.index = axis->index;
      if (axis->axis_type_.empty())
        dimInfo.axis = std::to_string(axis->dim_axis);
      else
        dimInfo.axis = axis->axis_type_;
      std::tie(l1_val, l0_val) = this->cand_->GetTileVal(axis);
      l1_val = CanonicalSimplify(l1_val);
      l0_val = CanonicalSimplify(l0_val);
      const auto l1 = l1_val.as<IntImm>();
      const auto l0 = l0_val.as<IntImm>();
      CHECK(l1 && l0);
      // Make sure tile size is positive.
      auto l1_pos_tile_size = l1->value <= 0 ? MIN_TILE : l1->value;
      auto l0_pos_tile_size = l0->value <= 0 ? l1_pos_tile_size : l0->value;
      dimInfo.l1_tiling_size = l1_pos_tile_size;
      dimInfo.l0_tiling_size = l0_pos_tile_size;
      dimInfo.dim_seq = axis->seq_index;
      dims.push_back(dimInfo);
    };
    analyzer_.ForEachAxisTopDown(Convert);
    return dims;
  }

  bool IsConflictPrime(const int64_t prime, const ParamReplacement &prev) {
    // Conflict exists in three cases:
    // 1. A = B
    // 2. A = B - 1 (when B - 1 is used in min/max)
    // 3. A = B + 1 (when A - 1 is used in min/max)
    std::unordered_set<int64_t> chosen;
    auto InsertMinMax = [&chosen](int64_t value) {
      chosen.insert(value);
      chosen.insert(value - 1);
      chosen.insert(value + 1);
    };

    for (const auto &it : this->cand_->tile_val_) {
      const auto tile_l1 = it.second.tile_l1.as<IntImm>();
      const auto tile_l0 = it.second.tile_l0.as<IntImm>();
      if (tile_l1) {
        InsertMinMax(tile_l1->value);
      }
      if (tile_l0) {
        InsertMinMax(tile_l0->value);
      }
      if (tile_l1 && tile_l0) {
        int64_t floordiv = tile_l1->value / tile_l0->value;
        int64_t mod = tile_l1->value - floordiv * tile_l0->value;
        InsertMinMax(floordiv);
        InsertMinMax(mod);
      }
    }
    for (const auto &d : this->dims_) {
      InsertMinMax(d.l1_tiling_size);
      InsertMinMax(d.l0_tiling_size);
    }
    if (chosen.count(prime) != 0) return true;
    for (auto mul : prev.mul_tile)
      if (prime == mul + 1) return true;
    return false;
  }

  ParamReplacement CreateVarTileReplaceMap() {
    ParamReplacement param_replacement;
    std::vector<int64_t> var_tile_replace;
    int64_t base_num = 37;
    auto IsPrime = [&base_num]() -> bool {
      for (auto i = 2; i < static_cast<int>(sqrt(base_num)); ++i) {
        if (base_num % i == 0) return false;
      }
      return true;
    };

    auto Finish = [&param_replacement]() -> bool {
      return (param_replacement.mod_tile.size() == GEN_PRIME_NUM) &&
             (param_replacement.mod_tile.size() == param_replacement.l0_tile.size());
    };
    while (!Finish()) {
      if (!IsConflictPrime(base_num, param_replacement)) {
        if (IsPrime()) {
          if (param_replacement.mod_tile.empty() ||
              param_replacement.mod_tile.size() <= param_replacement.l0_tile.size()) {
            param_replacement.mod_tile.insert(param_replacement.mod_tile.begin(), base_num);
          } else {
            int64_t mod = param_replacement.mod_tile.back();
            if (base_num > mod * 2) {
              param_replacement.l0_tile.insert(param_replacement.l0_tile.begin(), base_num);
            } else if (param_replacement.mod_tile.size() < GEN_PRIME_NUM) {
              param_replacement.mod_tile.insert(param_replacement.mod_tile.begin(), base_num);
            }
          }
        } else if (param_replacement.mul_tile.size() < GEN_PRIME_NUM) {
          param_replacement.mul_tile.insert(param_replacement.mul_tile.begin(), base_num);
        }
      }
      base_num += 1;
    }
    return param_replacement;
  }

  int64_t CalL1VarTiling(int64_t l0_tiling, TileAxis *axis) {
    auto GetCand = [this, l0_tiling]() -> int64_t {
      if (analyzer_.op_type_ == VECTOR_OP) {
        if (param_replacement_.l0_tile.empty())
          analyzer_.logger_.LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                               std::to_string(GEN_PRIME_NUM) + ")");
        int64_t c = param_replacement_.l0_tile.back();
        param_replacement_.l0_tile.pop_back();
        return c;
      } else {
        if (param_replacement_.mod_tile.empty() || param_replacement_.mul_tile.empty())
          analyzer_.logger_.LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                               std::to_string(GEN_PRIME_NUM) + ")");
        int64_t c = param_replacement_.mul_tile.back() * l0_tiling + param_replacement_.mod_tile.back();
        param_replacement_.mul_tile.pop_back();
        param_replacement_.mod_tile.pop_back();
        return c;
      }
    };
    int64_t cand = GetCand();
    if (analyzer_.op_type_ == GEMM_OP || analyzer_.op_type_ == CONV_OP) {
      bool found = false;
      while (!found && !param_replacement_.mul_tile.empty() && !param_replacement_.mod_tile.empty()) {
        found = true;
        for (auto p : prev_tiling_) {
          if (cand == p || cand == p - 1) found = false;
        }
        cand = GetCand();
      }
      if (!found) LOG(INFO) << "Use conflict prime " << cand << " for var replacement, may raise problem.";
    } else {
      const auto bound = axis->l1_constraints.tile_mod_.as<IntImm>();
      if (bound != nullptr && bound->value != -1) {
        CHECK_NE(bound->value, 0);
        CHECK_GT(cand, 0);
        // When shift exist, it is better to choose prime number that is divisible by bound
        // to generate less complicate Halide IR.
        while ((cand < bound->value) && (bound->value % cand != 0 || IsConflictPrime(cand, param_replacement_)))
          cand += 1;
      }
    }
    return cand;
  }

  static DimensionInfo ConvertDefaultInfo(TileAxis *axis) {
    DimensionInfo dimInfo;
    dimInfo.index = axis->index;
    dimInfo.dim_seq = axis->seq_index;
    dimInfo.is_inner = axis->is_inner;
    if (axis->axis_type_.empty())
      dimInfo.axis = std::to_string(axis->dim_axis);
    else
      dimInfo.axis = axis->axis_type_;
    return dimInfo;
  }

  void ConvertVarTilesToDims() {
    Map<Var, Expr> var_to_prime_record;
    auto Convert = [this, &var_to_prime_record](TileAxis *axis) {
      if (axis->index < 0) return;
      if (axis->is_pragma) return;
      axis->DumpAxis();
      Expr l1_val;
      Expr l0_val;
      DimensionInfo dimInfo = ConvertDefaultInfo(axis);
      std::tie(l1_val, l0_val) = this->cand_->GetTileVal(axis);
      l1_val = CanonicalSimplify(l1_val);
      l0_val = CanonicalSimplify(l0_val);
      const auto l1 = l1_val.as<IntImm>();
      const auto l0 = l0_val.as<IntImm>();
      if (l0 != nullptr && l0->value != TileVarId::UNDEFINE) {
        if (l0->value == TileVarId::VAR)
          analyzer_.logger_.LogFatalAndSaveLog("L0 value of axis " + std::to_string(axis->index) + "_" +
                                               std::to_string(axis->dim_axis) + " has not been tiled.");
        dimInfo.l0_tiling_size = l0->value;
      } else {
        if (analyzer_.op_type_ == GEMM_OP) {
          if (param_replacement_.l0_tile.empty())
            analyzer_.logger_.LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                                 std::to_string(GEN_PRIME_NUM) + ")");
          dimInfo.l0_tiling_size = param_replacement_.l0_tile.back();
          param_replacement_.l0_tile.pop_back();
          prev_tiling_.emplace_back(dimInfo.l0_tiling_size);
          dimInfo.l0_var = l0_val;
          if (l0_val.as<Variable>()) {
            auto v = air::Downcast<Var>(l0_val);
            var_to_prime_record.Set(v, make_const(v->type, dimInfo.l0_tiling_size));
          }
        } else if (analyzer_.op_type_ == CONV_OP) {
          dimInfo.l0_tiling_size = 65535;
        } else {
          dimInfo.l0_tiling_size = 1;
        }
      }
      if (l1 != nullptr && l1->value != TileVarId::UNDEFINE) {
        if (l1->value == TileVarId::VAR)
          analyzer_.logger_.LogFatalAndSaveLog("L1 value of axis " + std::to_string(axis->index) + "_" +
                                               std::to_string(axis->dim_axis) + " has not been tiled.");
        dimInfo.l1_tiling_size = l1->value;
      } else {
        int64_t l1_base = analyzer_.op_type_ == CONV_OP ? 1 : dimInfo.l0_tiling_size;
        dimInfo.l1_tiling_size = CalL1VarTiling(l1_base, axis);
        prev_tiling_.emplace_back(dimInfo.l1_tiling_size);
        dimInfo.l1_var = l1_val;
        if (l1_val.as<Variable>()) {
          auto v = air::Downcast<Var>(l1_val);
          var_to_prime_record.Set(v, make_const(v->type, dimInfo.l1_tiling_size));
        }
      }
      this->dims_.push_back(dimInfo);
    };
    analyzer_.ForEachAxisTopDown(Convert);
    if (analyzer_.op_type_ == CONV_OP) ConvertPragmaToDims(var_to_prime_record);
    ConvertShiftBoundToDims();
  }

  void ConvertShiftBoundToDims() {
    // dim.l1_size -> dynamic bound value set in attr
    // dim.l1_val  -> corresponding dynamic bound's variable
    // dim.l0_size -> tile prime chosen for this axis
    // dim.l0_val  -> corresponding tile variable
    // dim.prgama  -> shift time (IntImm)
    auto Convert = [this](TileAxis *axis) {
      std::vector<std::string> bound_value = axis->GetAttrValue("DYNAMIC_BOUND");
      if (!bound_value.empty()) {
        CHECK_EQ(bound_value.size(), 1U);
        CHECK_NE(bound_value[0], "");
        auto bound = static_cast<int>(std::strtol(bound_value[0].c_str(), nullptr, 10));
        DimensionInfo bound_info = ConvertDefaultInfo(axis);
        bound_info.l1_tiling_size = bound;
        bound_info.l1_var = axis->range_extent;
        for (const auto &d : this->dims_) {
          if (d.dim_seq != bound_info.dim_seq) continue;
          bound_info.l0_tiling_size = d.l1_tiling_size;
          bound_info.l0_var = d.l1_var;
        }
        std::vector<std::string> shift_value = axis->GetAttrValue("DYNAMIC_SHIFT");
        CHECK_EQ(shift_value.size(), 1U) << "Empty shift_time for dynamic bound " << bound;
        CHECK_NE(shift_value[0], "");
        auto shift = static_cast<int>(std::strtol(shift_value[0].c_str(), nullptr, 10));
        bound_info.pragma = shift;
        CHECK_NE(bound_info.l0_tiling_size, -1);
        this->dims_.push_back(bound_info);
      }
    };
    analyzer_.ForEachAxisTopDown(Convert);
  }

  void ConvertPragmaToDims(Map<Var, Expr> var_to_prime_record) {
    auto ConvertPragma = [this, &var_to_prime_record](TileAxis *axis) {
      if (!axis->is_pragma) return;
      Expr l1_val;
      Expr l0_val;
      DimensionInfo dimInfo = ConvertDefaultInfo(axis);
      std::tie(l1_val, l0_val) = this->cand_->GetTileVal(axis);
      const auto l1 = l1_val.as<IntImm>();
      const auto l0 = l0_val.as<IntImm>();
      if (l0 != nullptr && l0->value != TileVarId::UNDEFINE) {
        dimInfo.l0_tiling_size = l0->value;
      } else {
        CHECK(!param_replacement_.l0_tile.empty())
          << "Number of axis to tile exceeds maximal var replace limit (" << GEN_PRIME_NUM << ")";
        dimInfo.l0_tiling_size = param_replacement_.l0_tile.back();
        param_replacement_.l0_tile.pop_back();
        param_replacement_.l0_tile.pop_back();
        prev_tiling_.emplace_back(dimInfo.l0_tiling_size);
        dimInfo.l0_var = l0_val;
      }
      if (l1 != nullptr && l1->value != TileVarId::UNDEFINE) {
        dimInfo.l1_tiling_size = l1->value;
      } else {
        dimInfo.l1_tiling_size = dimInfo.l0_tiling_size;
      }
      dimInfo.l1_var = l1_val;
      dimInfo.l0_var = l0_val;
      dimInfo.pragma = l0_val;
      // Use same prime of Conv L1 for specgemm.
      for (const auto &d : dims_) {
        if (!d.l1_var.defined() || !l1_val.defined()) continue;
        Expr sub = CanonicalSimplify(Substitute(l1_val, var_to_prime_record));
        if (analyzer_.arith_ana_.CanProve(l1_val == d.l1_var) || analyzer_.arith_ana_.CanProve(sub == d.l1_var)) {
          dimInfo.l1_tiling_size = d.l1_tiling_size;
          dimInfo.l1_var = d.l1_var;
        } else if (const auto imm = sub.as<IntImm>()) {
          dimInfo.l1_tiling_size = imm->value;
        }
      }
      dims_.push_back(dimInfo);
    };
    analyzer_.ForEachAxisTopDown(ConvertPragma);
  }

  TilingAnalyzer &analyzer_;
  TileCandidate *cand_{nullptr};
  int64_t mem_limit_[MEM_SCOPE_BULK]{0};
  std::deque<ParamInfo> param_info_;
  ParamReplacement param_replacement_;
  std::vector<int64_t> prev_tiling_;
  TileSizes dims_;
  bool tile_success_{true};
};

// Map result to required format.
TileSizes NullTiling() {
  TileSizes dims;
  DimensionInfo dimInfo;
  dimInfo.index = 0;
  dimInfo.axis = "0";
  dimInfo.l1_tiling_size = MIN_TILE;
  dimInfo.l0_tiling_size = MIN_TILE;
  dimInfo.dim_seq = 0;
  dims.push_back(dimInfo);
  return dims;
}

std::pair<TileSizes, std::deque<ParamInfo>> GenerateTiling(const isl::schedule &sch, ScopInfo &scop_info, Stmt body) {
  scop_info.analysis_result_.SetIsTiled(false);
  TileSizes dims = NullTiling();
  std::deque<ParamInfo> param_info;
  TilingAnalyzer analyzer(sch, scop_info, body);
  bool need_tiling = analyzer.Prepare();

  std::stringstream ss;
  ss << body;
  analyzer.logger_.AppendLog(DO_TILING, ss);
  if (!need_tiling) {
    LOG(INFO) << "No need for tiling, exit.";
    if (!analyzer.logger_.DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
    return std::make_pair(dims, param_info);
  }
  TilingGenerator generator(analyzer);
  if (analyzer.is_dynamic_) {
    std::tie(dims, param_info) = generator.GenerateDynamic();
  } else if (scop_info.user_config_.GetPragmaSpeedUpTiling() && analyzer.op_type_ == VECTOR_OP) {
    dims = generator.GenerateQuickly();
  } else {
    dims = generator.Generate();
  }

  LOG(INFO) << "This dim is generated by auto tiling";
  if (!analyzer.logger_.DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
  return std::make_pair(dims, param_info);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
