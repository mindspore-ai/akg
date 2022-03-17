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

#include "poly/tiling/tiling.h"

namespace akg {
namespace ir {
namespace poly {

TileSizes TilingGenerator::Generate() {
  TraverseSolver solver(analyzer_);
  this->cand_ = solver.Solve();
  return ConvertToDims();
}

TileSizes TilingGenerator::GenerateQuickly() {
  if (analyzer_.scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    GpuSolver solver(analyzer_);
    this->cand_ = solver.Solve();
    return ConvertToDims();
  } else {
    InequalitySolver solver(analyzer_);
    this->cand_ = solver.Solve();
    this->memory_constraints_ = solver.GetMemoryConstraints();
    ConvertVarTilesToDims();
    return dims_;
  }
}

std::pair<TileSizes, std::deque<ParamInfo>> TilingGenerator::GenerateDynamic() {
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
        analyzer_.GetTileLogger().LogFatalAndSaveLog("Unsupported type_key for now: " + info->type_key);
      }
    }
    LOG(INFO) << stmt;
  } else {
    param_info_.clear();
  }
  return std::make_pair(dims_, param_info_);
}

TileSizes TilingGenerator::ConvertToDims() {
  TileSizes dims;

  auto Convert = [this, &dims](TileAxis *axis) {
    if (axis->index < 0) return;
    if (axis->is_inner && !axis->is_pragma) return;
    Expr c1_val = 1;
    Expr c0_val = 1;
    DimensionInfo dimInfo;
    dimInfo.index = axis->index;
    if (axis->axis_type_.empty())
      dimInfo.axis = std::to_string(axis->dim_axis);
    else
      dimInfo.axis = axis->axis_type_;
    std::tie(c1_val, c0_val) = this->cand_->GetTileVal(axis);
    c1_val = CanonicalSimplify(c1_val);
    c0_val = CanonicalSimplify(c0_val);
    int64_t c1_pos_tile_size = 1;
    int64_t c0_pos_tile_size = 1;
    if (!analyzer_.scop_info_.analysis_result_.GetIsGpuDmaAnalysed()) {
      const auto c1 = c1_val.as<IntImm>();
      const auto c0 = c0_val.as<IntImm>();
      CHECK(c1 && c0);
      // Make sure tile size is positive.
      c1_pos_tile_size = c1->value <= 0 ? MIN_TILE : c1->value;
      c0_pos_tile_size = c0->value <= 0 ? c1_pos_tile_size : c0->value;
    }
    dimInfo.c1_tiling_size = c1_pos_tile_size;
    dimInfo.c0_tiling_size = c0_pos_tile_size;
    dimInfo.dim_seq = axis->seq_index;
    dims.push_back(dimInfo);
  };
  analyzer_.ForEachAxisTopDown(Convert);
  return dims;
}

bool TilingGenerator::IsConflictPrime(const int64_t prime, const ParamReplacement &prev) {
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
    const auto tile_c1 = it.second.tile_c1.as<IntImm>();
    const auto tile_c0 = it.second.tile_c0.as<IntImm>();
    if (tile_c1) {
      InsertMinMax(tile_c1->value);
    }
    if (tile_c0) {
      InsertMinMax(tile_c0->value);
    }
    if (tile_c1 && tile_c0) {
      int64_t floordiv = tile_c1->value / tile_c0->value;
      int64_t mod = tile_c1->value - floordiv * tile_c0->value;
      InsertMinMax(floordiv);
      InsertMinMax(mod);
    }
  }
  for (const auto &d : this->dims_) {
    InsertMinMax(d.c1_tiling_size);
    InsertMinMax(d.c0_tiling_size);
  }
  if (chosen.count(prime) != 0) return true;
  for (auto mul : prev.mul_tile)
    if (prime == mul + 1) return true;
  return false;
}

TilingGenerator::ParamReplacement TilingGenerator::CreateVarTileReplaceMap() {
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

int64_t TilingGenerator::CalL1VarTiling(int64_t l0_tiling, TileAxis *axis) {
  auto GetCand = [this, l0_tiling]() -> int64_t {
    if (analyzer_.op_type_ == TileOpType::VECTOR_OP) {
      if (param_replacement_.l0_tile.empty())
        analyzer_.GetTileLogger().LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                                     std::to_string(GEN_PRIME_NUM) + ")");
      int64_t c = param_replacement_.l0_tile.back();
      param_replacement_.l0_tile.pop_back();
      return c;
    } else {
      if (param_replacement_.mod_tile.empty() || param_replacement_.mul_tile.empty())
        analyzer_.GetTileLogger().LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                                     std::to_string(GEN_PRIME_NUM) + ")");
      int64_t c = param_replacement_.mul_tile.back() * l0_tiling + param_replacement_.mod_tile.back();
      param_replacement_.mul_tile.pop_back();
      param_replacement_.mod_tile.pop_back();
      return c;
    }
  };
  int64_t cand = GetCand();
  if (analyzer_.op_type_ == TileOpType::GEMM_OP || analyzer_.op_type_ == TileOpType::CONV_OP) {
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
    const auto bound = axis->c1_constraints.tile_mod_.as<IntImm>();
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

DimensionInfo TilingGenerator::ConvertDefaultInfo(TileAxis *axis) {
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

void TilingGenerator::ConvertVarTilesToDims() {
  Map<Var, Expr> var_to_prime_record;
  auto Convert = [this, &var_to_prime_record](TileAxis *axis) {
    if (axis->index < 0) return;
    if (axis->is_pragma) return;
    axis->DumpAxis();
    Expr c1_val;
    Expr c0_val;
    DimensionInfo dimInfo = ConvertDefaultInfo(axis);
    std::tie(c1_val, c0_val) = this->cand_->GetTileVal(axis);
    c1_val = CanonicalSimplify(c1_val);
    c0_val = CanonicalSimplify(c0_val);
    const auto c1 = c1_val.as<IntImm>();
    const auto c0 = c0_val.as<IntImm>();
    if (c0 != nullptr && c0->value != TileVarId::UNDEFINE) {
      if (c0->value == TileVarId::VAR)
        analyzer_.GetTileLogger().LogFatalAndSaveLog("c0 value of axis " + std::to_string(axis->index) + "_" +
                                                     std::to_string(axis->dim_axis) + " has not been tiled.");
      dimInfo.c0_tiling_size = c0->value;
    } else {
      if (analyzer_.op_type_ == TileOpType::GEMM_OP) {
        if (param_replacement_.l0_tile.empty())
          analyzer_.GetTileLogger().LogFatalAndSaveLog("Axis index exceed maximal var replace limit (" +
                                                       std::to_string(GEN_PRIME_NUM) + ")");
        dimInfo.c0_tiling_size = param_replacement_.l0_tile.back();
        param_replacement_.l0_tile.pop_back();
        prev_tiling_.emplace_back(dimInfo.c0_tiling_size);
        dimInfo.c0_var = c0_val;
        if (c0_val.as<Variable>()) {
          auto v = air::Downcast<Var>(c0_val);
          var_to_prime_record.Set(v, make_const(v->type, dimInfo.c0_tiling_size));
        }
      } else if (analyzer_.op_type_ == TileOpType::CONV_OP) {
        dimInfo.c0_tiling_size = 65535;
      } else {
        dimInfo.c0_tiling_size = 1;
      }
    }
    if (c1 != nullptr && c1->value != TileVarId::UNDEFINE) {
      if (c1->value == TileVarId::VAR)
        analyzer_.GetTileLogger().LogFatalAndSaveLog("c1 value of axis " + std::to_string(axis->index) + "_" +
                                                     std::to_string(axis->dim_axis) + " has not been tiled.");
      dimInfo.c1_tiling_size = c1->value;
    } else {
      int64_t l1_base = analyzer_.op_type_ == TileOpType::CONV_OP ? 1 : dimInfo.c0_tiling_size;
      if (analyzer_.scop_info_.analysis_result_.IsCsrDynamicExtent(axis->range_extent)) {
        dimInfo.c1_tiling_size = analyzer_.scop_info_.user_config_.GetCsrThreadNum();
      } else {
        dimInfo.c1_tiling_size = CalL1VarTiling(l1_base, axis);
      }
      prev_tiling_.emplace_back(dimInfo.c1_tiling_size);
      dimInfo.c1_var = c1_val;
      if (c1_val.as<Variable>()) {
        auto v = air::Downcast<Var>(c1_val);
        var_to_prime_record.Set(v, make_const(v->type, dimInfo.c1_tiling_size));
      }
    }
    this->dims_.push_back(dimInfo);
  };
  analyzer_.ForEachAxisTopDown(Convert);
  if (analyzer_.op_type_ == TileOpType::CONV_OP) ConvertPragmaToDims(var_to_prime_record);
  ConvertShiftBoundToDims();
}

void TilingGenerator::ConvertShiftBoundToDims() {
  // dim.l1_size -> dynamic bound value set in attr
  // dim.c1_val  -> corresponding dynamic bound's variable
  // dim.l0_size -> tile prime chosen for this axis
  // dim.c0_val  -> corresponding tile variable
  // dim.prgama  -> shift time (IntImm)
  auto Convert = [this](TileAxis *axis) {
    std::vector<std::string> bound_value = axis->GetAttrValue(AT_DYNAMIC_BOUND);
    if (!bound_value.empty()) {
      CHECK_EQ(bound_value.size(), 1U);
      CHECK_NE(bound_value[0], "");
      auto bound = StrToDecimalInt(bound_value[0]);
      DimensionInfo bound_info = ConvertDefaultInfo(axis);
      bound_info.c1_tiling_size = bound;
      bound_info.c1_var = axis->range_extent;
      for (const auto &d : this->dims_) {
        if (d.dim_seq != bound_info.dim_seq) {
          continue;
        }
        bound_info.c0_tiling_size = d.c1_tiling_size;
        bound_info.c0_var = d.c1_var;
      }
      std::vector<std::string> shift_value = axis->GetAttrValue(AT_DYNAMIC_SHIFT);
      CHECK_EQ(shift_value.size(), 1U) << "Empty shift_time for dynamic bound " << bound;
      CHECK_NE(shift_value[0], "");
      auto shift = StrToDecimalInt(shift_value[0]);
      bound_info.pragma = shift;
      CHECK_NE(bound_info.c0_tiling_size, -1);
      this->dims_.push_back(bound_info);
    }
  };
  analyzer_.ForEachAxisTopDown(Convert);
}

void TilingGenerator::ConvertPragmaToDims(Map<Var, Expr> var_to_prime_record) {
  auto ConvertPragma = [this, &var_to_prime_record](TileAxis *axis) {
    if (!axis->is_pragma) return;
    Expr c1_val;
    Expr c0_val;
    DimensionInfo dimInfo = ConvertDefaultInfo(axis);
    std::tie(c1_val, c0_val) = this->cand_->GetTileVal(axis);
    const auto c1 = c1_val.as<IntImm>();
    const auto c0 = c0_val.as<IntImm>();
    if (c0 != nullptr && c0->value != TileVarId::UNDEFINE) {
      dimInfo.c0_tiling_size = c0->value;
    } else {
      CHECK(!param_replacement_.l0_tile.empty())
        << "Number of axis to tile exceeds maximal var replace limit (" << GEN_PRIME_NUM << ")";
      dimInfo.c0_tiling_size = param_replacement_.l0_tile.back();
      param_replacement_.l0_tile.pop_back();
      param_replacement_.l0_tile.pop_back();
      prev_tiling_.emplace_back(dimInfo.c0_tiling_size);
      dimInfo.c0_var = c0_val;
    }
    if (c1 != nullptr && c1->value != TileVarId::UNDEFINE) {
      dimInfo.c1_tiling_size = c1->value;
    } else {
      dimInfo.c1_tiling_size = dimInfo.c0_tiling_size;
    }
    dimInfo.c1_var = c1_val;
    dimInfo.c0_var = c0_val;
    dimInfo.pragma = c0_val;
    // Use same prime of Conv c1 for specgemm.
    for (const auto &d : dims_) {
      if (!d.c1_var.defined() || !c1_val.defined()) continue;
      Expr sub = CanonicalSimplify(Substitute(c1_val, var_to_prime_record));
      if (analyzer_.arith_ana_.CanProve(c1_val == d.c1_var) || analyzer_.arith_ana_.CanProve(sub == d.c1_var)) {
        dimInfo.c1_tiling_size = d.c1_tiling_size;
        dimInfo.c1_var = d.c1_var;
      } else if (const auto imm = sub.as<IntImm>()) {
        dimInfo.c1_tiling_size = imm->value;
      }
    }
    dims_.push_back(dimInfo);
  };
  analyzer_.ForEachAxisTopDown(ConvertPragma);
}

// Map result to required format.
TileSizes NullTiling() {
  TileSizes dims;
  DimensionInfo dimInfo;
  dimInfo.index = 0;
  dimInfo.axis = "0";
  dimInfo.c1_tiling_size = MIN_TILE;
  dimInfo.c0_tiling_size = MIN_TILE;
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
  analyzer.GetTileLogger().AppendLog(DO_TILING, ss);
  if (!need_tiling) {
    LOG(INFO) << "No need for tiling, exit.";
    if (!analyzer.GetTileLogger().DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
    return std::make_pair(dims, param_info);
  }
  TilingGenerator generator(analyzer);
  if (analyzer.scop_info_.user_config_.GetIsDynamic()) {
    std::tie(dims, param_info) = generator.GenerateDynamic();
  } else if ((scop_info.user_config_.GetPragmaSpeedUpTiling() && analyzer.op_type_ == TileOpType::VECTOR_OP) ||
             !g_attrs.GetStr(kErrorInfo, "").empty() || analyzer.scop_info_.user_config_.GetTarget() == TARGET_CUDA ||
             analyzer.scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    dims = generator.GenerateQuickly();
  } else {
    dims = generator.Generate();
  }

  LOG(INFO) << "This dim is generated by auto tiling";
  if (!analyzer.GetTileLogger().DumpLogFile()) LOG(WARNING) << "Write tiling log fail.";
  return std::make_pair(dims, param_info);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
