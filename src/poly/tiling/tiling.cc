/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <unordered_set>

#include "poly/tiling/tiling.h"
#include "poly/tiling/hermes/check_visitor.h"
#include "poly/tiling/hermes/hardware.h"
#include "poly/tiling/hermes/model_graph.h"
#include "poly/tiling/hermes/tiling_algo.h"
#include "poly/tiling/hermes/tiling_ir_survey.h"

namespace akg {
namespace ir {
namespace poly {

TileSizes TilingGenerator::Generate() {
  TraverseSolver solver(analyzer_);
  this->cand_ = solver.Solve();
  if (IsSymbolicTiling()) {
    CollectAxis();
  }
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
    if (IsSymbolicTiling()) {
      CollectAxis();
    }
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

bool TilingGenerator::IsSymbolicTiling() {
  bool is_symbolic_enabled = analyzer_.scop_info_.user_config_.GetIsSymbolicTiling();
  if (is_symbolic_enabled) {
    bool is_force_symbolic_tiling = analyzer_.scop_info_.user_config_.GetIsForceSymbolicTiling();
    if (is_force_symbolic_tiling) {
      return is_symbolic_enabled;
    }
    if (HasSymbolicStatusChanged(analyzer_.Halide())) {
      analyzer_.scop_info_.user_config_.SetIsSymbolicTiling(false);
      return false;
    }
  }
  return is_symbolic_enabled;
}

void TilingGenerator::CollectAxis() {
  ModelGraph::global_axis_vec_.clear();
  auto Collect = [](TileAxis *axis) {
    if (axis->index < 0) {
      return;
    }
    Axis detected_axis = Axis();
    detected_axis.index_ = axis->index;
    detected_axis.dim_axis_ = axis->dim_axis;
    detected_axis.range_ = axis->GetConstExtent();
    detected_axis.is_reduce_axis_ = axis->HasAttr(AT_REDUCE_AXIS);
    detected_axis.is_reduce_src_last_ = axis->HasAttr(AT_REDUCE_SRC_LAST);
    detected_axis.is_innermost_ =
      axis->HasAttr(AT_BROADCAST_INNERMOST_AXIS) || axis->HasAttr(AT_TRANSPOSE_INNERMOST_AXIS);
    if (axis->is_inner && !axis->is_pragma) {
      detected_axis.is_inner_ = true;
    }
    for (auto attr : axis->attrs) {
      if (attr.attr_key.find(AT_GEMM) != std::string::npos) {
        detected_axis.gemm_axis_ = attr.attr_value;
      }
    }
    bool has_axis = false;
    for (auto const &global_axis : ModelGraph::global_axis_vec_) {
      if (global_axis.dim_axis_ == detected_axis.dim_axis_ && global_axis.range_ == detected_axis.range_ &&
          global_axis.index_ == detected_axis.index_) {
        has_axis = true;
        break;
      }
    }
    if (!has_axis) {
      ModelGraph::global_axis_vec_.push_back(detected_axis);
    }
  };
  analyzer_.ForEachAxisTopDown(Collect);
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
  auto Finish = [&param_replacement]() -> bool {
    return (param_replacement.mod_tile.size() == GEN_PRIME_NUM) &&
           (param_replacement.mod_tile.size() == param_replacement.l0_tile.size());
  };
  while (!Finish()) {
    if (!IsConflictPrime(base_num, param_replacement)) {
      if (analyzer_.IsPrime(base_num)) {
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

std::vector<Axis> TilingGenerator::GetAxisDimFromGlobal(std::vector<Axis> &axis_of_node) {
  for (auto &axis : axis_of_node) {
    bool has_axis = false;
    for (auto global_axis : ModelGraph::global_axis_vec_) {
      if (axis.name_ == global_axis.name_) {
        axis.dim_axis_ = global_axis.dim_axis_;
        has_axis = true;
        break;
      }
    }
    if (!has_axis) {
      for (auto it = ModelGraph::name_dim_set_.begin(); it != ModelGraph::name_dim_set_.end(); it++) {
        if (axis.name_ == it->first && ModelGraph::global_axis_vec_.size() > it->second) {
          axis.name_ = ModelGraph::global_axis_vec_[it->second].name_;
          axis.dim_axis_ = ModelGraph::global_axis_vec_[it->second].dim_axis_;
          break;
        }
      }
    }
  }
  return axis_of_node;
}

TileSizes TilingGenerator::HermesTiling(TileSizes dims) {
  std::unique_ptr<InitGraph> init_graph = std::make_unique<InitGraph>(CheckVisitor::nodes_);
  for (auto node : CheckVisitor::nodes_) {
    for (auto node_init : init_graph->nodes_) {
      if (node_init->name_ == node->name_) {
        node_init->axis_of_node_ = GetAxisDimFromGlobal(node->axis_of_node_);
      }
    }
  }

  bool disable_db_and_bo = true;
  for (auto node : init_graph->nodes_) {
    if (node->op_.op_type_ == Op::OpType::MatMul || node->op_.op_type_ == Op::OpType::BatchMatMul) {
      disable_db_and_bo = false;
      break;
    }
  }
  if (disable_db_and_bo) {
    g_attrs.Set(kEnableDoubleBuffer, air::make_const(Int(kBit32), false));
    g_attrs.Set(kEnableBisectOptimize, air::make_const(Int(kBit32), false));
  }

  std::unique_ptr<ModelGraph> model_graph = std::make_unique<ModelGraph>(*init_graph);
  model_graph->is_activated_double_buffer_ = g_attrs.GetBool(kEnableDoubleBuffer, true);

  Hardware hardware(kNumCore, kMemVCSize, kMemC1Size, kMemC0Size, kMemVCAlign, kMemC1Align, kVBlockNum, kVBlockSize);

  GetTilingSize(*model_graph, hardware);

  size_t idx_global_axis_vec = 0;
  for (size_t i = 0; i < std::min(dims.size(), model_graph->global_axis_vec_.size()); ++i) {
    if (model_graph->global_axis_vec_[idx_global_axis_vec].is_inner_) {
      idx_global_axis_vec++;
      i--;
      continue;
    }
    int64_t c0_tiling = model_graph->global_axis_vec_[idx_global_axis_vec].c0_tiling_;
    int64_t c1_tiling = std::max(model_graph->global_axis_vec_[idx_global_axis_vec].c0_tiling_,
                                 model_graph->global_axis_vec_[idx_global_axis_vec].c1_tiling_);
    dims[i].c0_tiling_size = c0_tiling;
    dims[i].c1_tiling_size = c1_tiling;
    idx_global_axis_vec++;
  }

  std::stringstream tensor_dims_str_stream;
  tensor_dims_str_stream << "tensor dims = [" << model_graph->global_axis_vec_[0].range_;
  for (size_t i = 1; i < model_graph->global_axis_vec_.size(); ++i) {
    tensor_dims_str_stream << ";" << model_graph->global_axis_vec_[i].range_;
  }
  tensor_dims_str_stream << "]";
  LOG(INFO) << tensor_dims_str_stream.str();

  LOG(INFO) << "categorie = " << StringOfCategory(model_graph->OperatorCategory()) << std::endl;

  return dims;
}

void TilingGenerator::ExtractAxisInfoFromScheduler(const isl::schedule &sch) {
  ModelGraph::name_dim_set_.clear();
  auto map_list = sch.get_map().get_map_list();
  auto map_list_size = map_list.size();
  for (unsigned i = 0; i < map_list_size; i++) {
    auto set = map_list.get_at(i).domain();
    auto set_size = set.get_space().dim(isl_dim_out);
    for (unsigned j = 0; j < set_size; j++) {
      auto dim = isl::manage(isl_set_get_dim_id(set.get(), isl_dim_out, j));
      ModelGraph::name_dim_set_.insert(std::make_pair(dim.to_str(), j));
    }
  }

  for (auto &global_axis : ModelGraph::global_axis_vec_) {
    for (auto it = ModelGraph::name_dim_set_.begin(); it != ModelGraph::name_dim_set_.end(); it++) {
      if (global_axis.dim_axis_ == static_cast<int>(it->second)) {
        global_axis.name_ = it->first;
        break;
      }
    }
  }
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

  if (analyzer.scop_info_.user_config_.GetIsSymbolicTiling() && Hardware::HasVCFail(g_attrs.GetStr(kErrorInfo, ""))) {
    Hardware::AddVCFailCounter();
    g_attrs.Set(kErrorInfo, StringImm::make(""));
  } else if (!g_attrs.GetBool(kIsPolyConfigReset, false)) {
    Hardware::ResetVCFailCounter();
  }

  if (analyzer.scop_info_.user_config_.GetIsDynamic()) {
    std::tie(dims, param_info) = generator.GenerateDynamic();
  } else if ((scop_info.user_config_.GetPragmaSpeedUpTiling() && analyzer.op_type_ == TileOpType::VECTOR_OP) ||
             !g_attrs.GetStr(kErrorInfo, "").empty() || analyzer.scop_info_.user_config_.GetTarget() == TARGET_CUDA ||
             analyzer.scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    dims = generator.GenerateQuickly();
  } else {
    dims = generator.Generate();
  }

  // Hermes call
  if (analyzer.scop_info_.user_config_.GetIsSymbolicTiling()) {
    generator.ExtractAxisInfoFromScheduler(sch);
    dims = generator.HermesTiling(dims);
    LOG(INFO) << "This dim is generated by symbolic tiling";
  } else {
    LOG(INFO) << "This dim is generated by auto tiling";
  }
  if (!analyzer.GetTileLogger().DumpLogFile()) {
    LOG(WARNING) << "Write tiling log fail.";
  }

  return std::make_pair(dims, param_info);
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
