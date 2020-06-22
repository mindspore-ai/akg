/**
 *
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

#include "poly/tiling_solver.h"

namespace akg {
namespace ir {
namespace poly {
void TilingSolver::CollectMemoryLimit() {
  double percentage = ALLOCATION_PERCENTAGE;
  for (auto attr : analyzer_.RootAxis()->attrs) {
    if (attr.attr_key != "MEM_RATIO") continue;
    CHECK_NE(attr.attr_value, "");
    percentage = std::strtod(attr.attr_value.c_str(), nullptr);
    break;
  }

  DavinciInfo &d_info = DavinciInfo::GetInstance();
  for (auto i = 0; i < MEM_SCOPE_BULK; ++i) {
    this->mem_limit_[i] = d_info.GetMemoryLimitInScope(i) * percentage;
  }
}

void TilingSolver::CollectTileAxisTopDown() {
  auto CollectTileAxis = [this](TileAxis *a) {
    if (a == analyzer_.RootAxis() || a->index != this->tiling_band_) {
      return;
    }
    this->cand_.InsertAxisBack(a);
  };

  this->cand_.ResetTileAxis();
  this->analyzer_.ForEachAxisTopDown(CollectTileAxis);
  this->cand_.SortByPriority();
}

void InequalitySolver::InitTileAxis(TileLevel level) {
  tiling_mem_info_ = std::unique_ptr<TilingMemInfo>(new (std::nothrow) TilingMemInfo());
  CHECK(tiling_mem_info_) << "memory alloc fail";

  auto UpdateLevelTile = [this, level](TileAxis *axis, Expr tile) {
    if (level == LEVEL1) {
      this->cand_.UpdateL1Tile(axis, tile);
    } else {
      this->cand_.UpdateL0Tile(axis, tile);
    }
  };

  for (auto axis : this->cand_.GetTileAxis()) {
    // Step 1: Create unique tile var for each axis.
    std::string var_name = level == LEVEL1 ? "T1_" : "T0_";
    var_name += std::to_string(axis->index) + "_";
    var_name += axis->axis_type_.empty() ? std::to_string(axis->dim_axis) : axis->axis_type_;
    Var tile_var;

    // ensure unique address
    if (tiling_mem_info_->tile_var_map.find(var_name) == tiling_mem_info_->tile_var_map.end()) {
      tile_var = Var(var_name, Int(32));
      tiling_mem_info_->tile_var_map[var_name] = tile_var;
    } else {
      tile_var = tiling_mem_info_->tile_var_map[var_name];
    }
    UpdateLevelTile(axis, tile_var);

    // Step 2: Update for axes with determined tiling factor.
    TileAxis::Constraint cons = axis->GetConstConstraint(level);

    // These are two cases when tiling factor is fixed for axis with static shape:
    // 1. if tile_min == tile_extent ==> tile factor = tile_extent
    // 2. contains only one tile candidate ==> tile factor = this candidate
    if (cons.tile_extent_.as<IntImm>()->value > 0 &&
        cons.tile_min_.as<IntImm>()->value == cons.tile_extent_.as<IntImm>()->value) {
      UpdateLevelTile(axis, CastInt64ToExpr(cons.tile_extent_.as<IntImm>()->value));
    } else if (cons.cand_factor.size() == 1U) {
      UpdateLevelTile(axis, CastInt64ToExpr(cons.cand_factor[0].as<IntImm>()->value));
    }
  }
}

TileCandidate *InequalitySolver::Solve() {
  CollectMemoryLimit();

  auto tile_band_size = static_cast<int>(analyzer_.RootAxis()->children.size());
  for (auto band = 0; band < tile_band_size; ++band) {
    tiling_band_ = band;
    CollectTileAxisTopDown();

    InitTileAxis(LEVEL1);
    if (analyzer_.op_type_ != VECTOR_OP) {
      InitTileAxis(LEVEL0);
    }

    if (analyzer_.scop_->pragma_analyze_reuse_buffer_) {
      UpdateMemInfoWithBufReuse();
    } else {
      UpdateMemInfo();
    }

    Array<Expr> memory_constraints = CollectMemoryConstraints();

    auto tile_axes = cand_.GetTileAxis();
    for (auto i = static_cast<int>(tile_axes.size()) - 1; i >= 0; --i) {
      TileAxis *axis = tile_axes[i];
      DetermineTileFactor(axis, LEVEL1, memory_constraints);
    }
    if (analyzer_.op_type_ != VECTOR_OP) {
      for (auto i = static_cast<int>(tile_axes.size()) - 1; i >= 0; --i) {
        TileAxis *axis = tile_axes[i];
        DetermineTileFactor(axis, LEVEL0, memory_constraints);
      }
    }
  }
  return &cand_;
}

Expr InequalitySolver::GetSubstitutedExpr(const NodeRef &op) {
  const auto v = op.as<Variable>();
  auto var = ktvm::Downcast<Var>(op);
  Expr ret;
  if (defined_vars_.find(var) == defined_vars_.end()) {
    bool is_tile_var = false;
    for (auto it : this->cand_.tile_val_) {
      if ((v == it.second.tile_l1.as<Variable>()) || (v == it.second.tile_l0.as<Variable>())) {
        is_tile_var = true;
        break;
      }
    }
    if (!is_tile_var) {
      return ret;
    }

    ret = make_const(var.type(), 1);
    auto ScanTileVal = [this, &ret, &var](TileAxis *axis) {
      const auto l1_var = this->cand_.GetTileVal(axis).first.as<Variable>();
      const auto l0_var = this->cand_.GetTileVal(axis).second.as<Variable>();
      if (l1_var != nullptr && l1_var->name_hint == var->name_hint) {
        ret = axis->l1_constraints.tile_min_;
      } else if (l0_var != nullptr && l0_var->name_hint == var->name_hint) {
        ret = axis->l0_constraints.tile_min_;
      }
      if (ret.type() != var.type()) {
        if (ret.as<IntImm>()) {
          ret = make_const(var.type(), ret.as<IntImm>()->value);
        } else {
          ret = Cast::make(var.type(), ret);
        }
      }
    };
    this->analyzer_.ForEachAxisTopDown(ScanTileVal);
  } else if (defined_vars_[var].as<IntImm>()) {
    ret = defined_vars_[var];
  }
  return ret;
}

Expr InequalitySolver::SolveMemoryConstraint(const Array<Expr> &memory_constraints, const Var tiling_var) {
  Expr result;
  Array<Expr> cons_on_var;
  std::stringstream ss;
  ss << "Start to solve tiling_var " << tiling_var;
  analyzer_.logger_.AppendLog(DO_TILING, ss);

  for (auto mc : memory_constraints) {
    // All memory constraints are in `{Const, Var} op {Const, Var} <= Const` form,
    // e.g. 256 * T1_0_0 + 64 * floordiv((T1_0_1 + 15), 16) * 16 + 96 <= 131072.
    const auto le = mc.as<LE>();
    if (le == nullptr || !ContainVar(le->a, tiling_var)) {
      continue;
    }

    ss << "[Memory constraint]: " << mc;
    analyzer_.logger_.AppendLog(DO_TILING, ss);

    Map<Var, Expr> var_max;
    auto SubstituteOtherVar = [this, &var_max, tiling_var](const NodeRef &op) {
      const auto v = op.as<Variable>();
      if (v == nullptr || v->name_hint == tiling_var->name_hint) {
        return;
      }
      auto var = ktvm::Downcast<Var>(op);
      Expr value = GetSubstitutedExpr(op);
      if (value.defined()) {
        var_max.Set(var, value);
      }
    };
    ktvm::ir::PostOrderVisit(mc, SubstituteOtherVar);
    mc = Substitute(mc, var_max);
    cons_on_var.push_back(CanonicalSimplify(mc));
  }

  if (!analyzer_.is_dynamic_ && cons_on_var.size() == 1U && ContainVar(cons_on_var[0], tiling_var)) {
    result = ExprSimplifier().ReduceInequality(cons_on_var[0], tiling_var, true, false);
    ss << "ReduceInequality Result: " << result;
    // When result of reduce is not like form `var <= something`, use inferbound instead.
    if (result.as<LE>() != nullptr && (result.as<LE>()->a.as<Variable>() == nullptr ||
                                       result.as<LE>()->a.as<Variable>()->name_hint != tiling_var->name_hint)) {
      result = SolveByInferBound(cons_on_var, tiling_var);
    }
  } else if (!cons_on_var.empty()) {
    result = SolveByInferBound(cons_on_var, tiling_var);
  } else {
    ss << "No constraint on tiling_var " << tiling_var;
  }
  analyzer_.logger_.AppendLog(DO_TILING, ss);
  return result;
}

Expr InequalitySolver::SolveByInferBound(const Array<Expr> &cons_on_var, const Var tiling_var) {
  std::stringstream ss;
  auto new_constraints = cons_on_var;
  analyzer_.ForEachAxisTopDown([&](TileAxis *axis) {
    if (axis == analyzer_.RootAxis()) {
      return;
    }

    new_constraints.push_back(axis->range_extent >= CastInt64ToExpr(1));
    if (axis->HasAttr("DYN_SHAPE_LIMIT")) {
      auto res = axis->GetAttrValue("DYN_SHAPE_LIMIT");
      CHECK_EQ(res.size(), 1U);
      auto range_limit = static_cast<int>(std::strtol(res[0].c_str(), nullptr, 10));
      new_constraints.push_back(axis->range_extent <= CastIntToExpr(range_limit));
    }
  });

  Expr infer_res = (tiling_var <= InferBoundOfExprWithCond(tiling_var, new_constraints).max);
  ss << "Use inferbound to solve instread. Result: " << infer_res;
  return infer_res;
}

std::deque<Scop::ParamInfo> DynamicShapeSolver::GetParamInfo() { return this->solver_.param_info_; }

void InequalitySolver::DetermineTileFactor(TileAxis *axis, TileLevel level, const Array<Expr> &memory_constraints) {
  if (axis->is_pragma && level == LEVEL1) {
    return;
  }

  std::stringstream ss;
  Expr l1_expr = CanonicalSimplify(cand_.GetTileVal(axis).first);
  Expr l0_expr = CanonicalSimplify(cand_.GetTileVal(axis).second);
  Expr to_tile = level == LEVEL1 ? l1_expr : l0_expr;
  TileAxis::Constraint cons = level == LEVEL1 ? axis->l1_constraints : axis->l0_constraints;

  if (axis->HasAttr("DYN_SHAPE_LIMIT")) {
    AppendShapeLimitConstraint(axis, to_tile);
  }

  if (to_tile.as<Variable>()) {
    Expr res = SolveMemoryConstraint(memory_constraints, ktvm::Downcast<Var>(to_tile));
    if (!res.defined()) {
      ss << "No memory constraint on " << to_tile << " for now, use maximal tile " << cons.tile_extent_;
      analyzer_.logger_.AppendLog(DO_TILING, ss);
      res = (to_tile <= cons.tile_extent_);
    }
    res = RemoveCast(Substitute(res, defined_vars_));
    ss << "Result after substitute defined vars: " << res;
    analyzer_.logger_.AppendLog(DO_TILING, ss);

    const auto le = res.as<LE>();
    CHECK(le) << "Cannot define tile range for axis " << axis->index << "_" << axis->dim_axis;

    Expr mem_constraint = CanonicalSimplify(le->b);
    Expr tile_min;
    Expr tile_range;
    Expr shape_range;
    if (level == LEVEL1) {
      shape_range = axis->range_extent;
      tile_min = axis->l1_constraints.tile_min_;
      tile_range = CanonicalSimplify(Min::make(axis->l1_constraints.tile_extent_, shape_range));
    } else {
      shape_range = l1_expr;
      tile_min = axis->l0_constraints.tile_min_;
      tile_range = CanonicalSimplify(Min::make(axis->l0_constraints.tile_extent_, shape_range));
    }

    if (analyzer_.arith_ana_.CanProve(mem_constraint <= 0)) {
      ss << "Memory limit should be positive, but get " << mem_constraint << ", use minimal tile " << tile_min;
      analyzer_.logger_.AppendLog(DO_TILING, ss);
      mem_constraint = tile_min;
    }

    Expr final_factor_expr;
    bool is_static_shape = tile_range.as<IntImm>() != nullptr;
    if (is_static_shape) {
      if (mem_constraint.as<IntImm>() == nullptr) {
        tile_success_ = false;
        analyzer_.logger_.AppendLine(DO_TILING,
                                     "[Warning] Static shape's memory limit is not const, use static tiling instead.");
        return;
      }
      int64_t final_factor = DetermineTileForStatic(axis, mem_constraint, tile_range, level);
      ss << "[Static shape final factor]: " << to_tile << " -> " << final_factor;
      analyzer_.logger_.AppendLog(DO_TILING, ss);
      final_factor_expr = CastInt64ToExpr(final_factor);
    } else {
      if (analyzer_.arith_ana_.CanProve(tile_min == tile_range)) {
        param_info_.push_front(Scop::ParamInfo{"LetStmt", Expr(to_tile), tile_range});
        AppendShapeLimitConstraint(axis, to_tile);
        defined_vars_.Set(ktvm::Downcast<Var>(to_tile), tile_range);
        return;
      }

      param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[MemoryLimit_UB]"), to_tile <= shape_range});
      final_factor_expr = DetermineTileForDynamic(axis, mem_constraint, to_tile, shape_range, tile_range, level);
      param_info_.push_front(Scop::ParamInfo{"LetStmt", to_tile, final_factor_expr});
      param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[MemoryLimit_UB]"), to_tile <= final_factor_expr});
      ss << "[Dynamic shape final factor]: " << to_tile << " -> " << final_factor_expr;
      analyzer_.logger_.AppendLog(DO_TILING, ss);
    }

    CHECK(final_factor_expr.defined());
    defined_vars_.Set(ktvm::Downcast<Var>(to_tile), final_factor_expr);
    // We can only update const tiling factor to final dim as we will replace those var factor with prime number.
    if (const auto imm = final_factor_expr.as<IntImm>()) {
      if (level == LEVEL1) {
        cand_.UpdateL1Tile(axis, imm->value);
      } else {
        cand_.UpdateL0Tile(axis, imm->value);
      }
    }
  } else if (to_tile.as<IntImm>() == nullptr) {
    LOG(INFO) << "Tile var should be either IntImm or Variable, but found " << to_tile;
  }
}

Expr InequalitySolver::DetermineTileForDynamic(TileAxis *axis, const Expr &mem_constraint, const Expr &to_tile,
                                               const Expr &shape_range, const Expr &tile_range, TileLevel level) {
  Expr final_factor;
  std::stringstream ss;
  auto tile = ktvm::Downcast<Var>(to_tile);
  auto new_mem_constraint = mem_constraint;
  TileAxis::Constraint cons = level == LEVEL1 ? axis->l1_constraints : axis->l0_constraints;

  bool infer_bound_fail =
    new_mem_constraint.as<Variable>() && new_mem_constraint.as<Variable>()->name_hint == tile->name_hint;

  if (analyzer_.op_type_ != CONV_OP && infer_bound_fail) {
    LOG(WARNING) << "Result of infer max bound for var " << to_tile << " fail, apply minimal tile " << cons.tile_min_;
    final_factor = cons.tile_min_;
  } else {
    bool need_adjust_mem =
      ((analyzer_.arith_ana_.CanProve(cons.tile_mod_ > 1)) &&
       (analyzer_.arith_ana_.CanProve(new_mem_constraint % cons.tile_mod_ != 0)) && (!axis->HasAttr("DYNAMIC_SHIFT")));

    // Reduce memory limit so that mem_constraint % tile_mod == 0.
    if (need_adjust_mem) {
      if (!analyzer_.arith_ana_.CanProve(new_mem_constraint >= cons.tile_mod_)) {
        LOG(WARNING) << "Maximal memory for axis " << to_tile << " is " << new_mem_constraint << ", constraint \""
                     << new_mem_constraint << " % " << cons.tile_mod_ << " == 0\""
                     << " is invalid, final factor may not be aligned.";
      } else {
        ss << "reduce memory limit from " << new_mem_constraint;
        while (analyzer_.arith_ana_.CanProve(new_mem_constraint % cons.tile_mod_ != 0))
          new_mem_constraint = CanonicalSimplify(new_mem_constraint - 1);
        ss << " to " << new_mem_constraint << " according to mod constraint " << cons.tile_mod_;
        analyzer_.logger_.AppendLog(DO_TILING, ss);
      }
    }

    param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[MemoryLimit_UB]"), to_tile <= new_mem_constraint});

    if (!cons.cand_factor.empty()) {
      // If candidate factors are provided, final factor is set to `max(min(c1, shape), ..., min(cn, shape))`
      // where c1, ..., cn are n candidate factors.
      std::vector<Expr> min_set;
      for (auto c : cons.cand_factor) {
        min_set.emplace_back(Min::make(c, shape_range));
      }
      final_factor = min_set.back();
      min_set.pop_back();
      while (!min_set.empty()) {
        final_factor = Max::make(final_factor, min_set.back());
        min_set.pop_back();
      }
    } else {
      final_factor = CanonicalSimplify(Min::make(new_mem_constraint, tile_range));
    }
  }

  // Add forbid isolation constraint to final factor by custom cce call `FindDivisibleTilingFactor`.
  if (level == LEVEL1 && axis->forbid_iso) {
    auto max_final_factor = InferBoundOfExprWithCond(final_factor, {tile > 0, tile <= axis->range_extent}).max;
    bool need_constraint = !(max_final_factor.as<IntImm>() && max_final_factor.as<IntImm>()->value == 1);
    if (axis->HasAttr("DYN_SHAPE_LIMIT")) {
      auto shape_limit = axis->GetAttrValue("DYN_SHAPE_LIMIT");
      CHECK_EQ(shape_limit.size(), 1U);
      auto range_limit = static_cast<int>(std::strtol(shape_limit[0].c_str(), nullptr, 10));
      if (analyzer_.arith_ana_.CanProve(range_limit <= GetConstIntUpBound(max_final_factor))) {
        final_factor = axis->range_extent;
      } else {
        final_factor = Call::make(tile->type, tiling_algorithm::intrinsic::FL_find_divisible_tiling_factor,
                                  {max_final_factor, axis->range_extent}, Call::Extern);
      }
    } else if (need_constraint) {
      final_factor = Call::make(tile->type, tiling_algorithm::intrinsic::FL_find_divisible_tiling_factor,
                                {max_final_factor, axis->range_extent}, Call::Extern);
    }
  }
  return final_factor;
}

void InequalitySolver::AppendShapeLimitConstraint(TileAxis *axis, Expr to_tile) {
  if (axis->dyn_shape_limit == -1) {
    LOG(WARNING) << "It is better to set dynamic shape limit for full tile axis " << axis->range_extent;
  } else {
    param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[MemoryLimit_UB]"),
                                          axis->range_extent <= CastIntToExpr(axis->dyn_shape_limit)});
  }
}

int64_t InequalitySolver::DetermineTileForStatic(TileAxis *axis, const Expr &mem_constraint, const Expr &tile_range,
                                                 TileLevel level) {
  std::stringstream ss;
  auto final_factor = MIN_TILE;
  auto static_shape = tile_range.as<IntImm>()->value;
  auto static_mem_constraint = mem_constraint.as<IntImm>()->value;
  TileAxis::Constraint cons = level == LEVEL1 ? axis->l1_constraints : axis->l0_constraints;

  if (!cons.cand_factor.empty()) {
    for (auto i = static_cast<int>(cons.cand_factor.size()) - 1; i >= 0; --i) {
      auto max_cand = cons.cand_factor[i];

      if (max_cand.as<IntImm>() == nullptr) {
        ss << "Static shape should have const candidate factor, while got " << max_cand;
        analyzer_.logger_.LogFatalAndSaveLog(ss.str());
      }

      if (max_cand.as<IntImm>()->value <= static_mem_constraint) {
        final_factor = max_cand.as<IntImm>()->value;
        ss << "--> Candidate factor " << final_factor;
        break;
      }
    }
  } else {
    if (static_mem_constraint >= static_shape) {
      final_factor = static_shape;
    } else {
      if (cons.tile_min_.as<IntImm>() == nullptr) {
        ss << "Static shape should have const tile min, while got " << cons.tile_min_;
        analyzer_.logger_.LogFatalAndSaveLog(ss.str());
      }

      final_factor = std::max(cons.tile_min_.as<IntImm>()->value, static_mem_constraint);
      ss << "--> Init factor " << final_factor;

      auto mod_value = cons.tile_mod_.as<IntImm>() ? cons.tile_mod_.as<IntImm>()->value : 1;
      if (static_shape >= mod_value && final_factor % mod_value != 0) {
        final_factor = std::max(static_cast<int>(final_factor / mod_value * mod_value), 1);
        ss << "--> Mod value " << mod_value << " --> Align to mod " << final_factor;
      }

      auto tail = static_shape - (static_shape / final_factor) * final_factor;
      ss << "--> Tail " << tail;

      // When tiling factor generating tail, we need to check whether it is valid (only for vector op).
      if (level == LEVEL1 && tail > 0) {
        if (axis->forbid_iso) {
          // We use conservative strategy here to choose final factor, i.e. use divisible factor that is smaller
          // than memory limit; In the future, we may consider to choose from larger-divisible factor and
          // smaller-divisible factor;
          while (static_shape % final_factor != 0) --final_factor;
          ss << "--> Forbid isolate " << final_factor;
        } else if (final_factor % GetMaxAlignBytes(axis->data_size) != 0) {
          if (final_factor < GetMaxAlignBytes(axis->data_size)) {
            final_factor =
              GetMaxAlignBytes(axis->data_size) > static_mem_constraint ? MIN_TILE : GetMaxAlignBytes(axis->data_size);
          } else {
            while (final_factor % GetMaxAlignBytes(axis->data_size) != 0) {
              --final_factor;
            }
          }
          ss << "--> Align to (" << GetMaxAlignBytes(axis->data_size) << ") bytes " << final_factor;
        }
      }
    }

    if (analyzer_.scop_->pragma_analyze_multicore_ && !analyzer_.is_dynamic_ && analyzer_.op_type_ == VECTOR_OP) {
      MulticoreStrategy mc_strategy_ = MulticoreStrategy(cand_, analyzer_.logger_.GetDumpDir());
      final_factor = mc_strategy_.AdjustTilingAccordingToMulticoreConstraint(axis, final_factor);
    }
  }
  return final_factor;
}

void InequalitySolver::CalculateMemoryInBuffer(const TilingAnalyzer::BufferEntry *buf, TilingMemInfo *mem_info) {
  std::stringstream ss;
  bool this_band_buf = (buf->scope == MEM_SCOPE_GM);
  Expr buf_shape = CastInt64ToExpr(buf->size * buf->expand_size);
  bool is_l0_buf = buf->scope > MEM_SCOPE_L1;

  if (buf->scope != MEM_SCOPE_GM) {
    for (auto &axis : *(buf->tile_axis)) {
      if (axis == this->analyzer_.RootAxis() || axis->index != tiling_band_) {
        continue;
      }
      this_band_buf = true;

      // Multiply var's shape to get buffer tile shape.
      Expr tile_var = is_l0_buf ? this->cand_.tile_val_[axis].tile_l0 : this->cand_.tile_val_[axis].tile_l1;
      CHECK(tile_var.defined()) << "Tile var not defined.";

      // Use original extent for shifted axes.
      if (analyzer_.arith_ana_.CanProve(tile_var > axis->range_extent)) tile_var = axis->range_extent;

      // Make tile var align to 32 Bytes.
      tile_var = EstimateAlignment(buf, axis, tile_var);

      buf_shape *= tile_var;
    }
  }

  if (!this_band_buf) {
    return;
  }

  mem_info->live_buf[buf] = buf_shape;

  if (mem_info->live_size[buf->scope].defined()) {
    mem_info->live_size[buf->scope] = CanonicalSimplify(mem_info->live_size[buf->scope] + buf_shape);
  } else {
    mem_info->live_size[buf->scope] = buf_shape;
  }

  if (mem_info->max_live_size[buf->scope].defined()) {
    bool current_is_larger =
      ExprSimplifier().CanProveWithPosParam(mem_info->live_size[buf->scope] >= mem_info->max_live_size[buf->scope]);
    bool current_is_smaller =
      ExprSimplifier().CanProveWithPosParam(mem_info->live_size[buf->scope] < mem_info->max_live_size[buf->scope]);

    if (current_is_larger) {
      ss << "Can prove current live size" << mem_info->live_size[buf->scope] << " greater than maximal size "
         << mem_info->max_live_size[buf->scope];
      mem_info->max_live_size[buf->scope] = mem_info->live_size[buf->scope];
    } else if (!current_is_smaller) {
      ss << "Can not compare current live size" << mem_info->live_size[buf->scope] << " with maximal size "
         << mem_info->max_live_size[buf->scope];
      mem_info->max_live_size[buf->scope] = CanonicalSimplify(mem_info->max_live_size[buf->scope] + buf_shape);
    }

    analyzer_.logger_.AppendLog(DO_TILING, ss);

  } else {
    mem_info->max_live_size[buf->scope] = mem_info->live_size[buf->scope];
  }
}

Expr InequalitySolver::EstimateAlignment(const TilingAnalyzer::BufferEntry *buf, TileAxis *axis, Expr tile) const {
  if (analyzer_.op_type_ != VECTOR_OP) {
    return tile;
  }

  auto GetAlignType = [axis, buf]() -> std::string {
    std::string align_type;
    for (const auto &attr : axis->attrs) {
      if (attr.attr_key.find("ALIGN") == std::string::npos) continue;
      std::string local_name = attr.attr_value + "_local_UB";
      if (buf->name.find(local_name) != std::string::npos) {
        std::vector<std::string> res = akg::common::Split(attr.attr_key, ":");
        if (res.size() == 2U) align_type = res[1];
        return align_type;
      }
    }
    return align_type;
  };

  std::string align_type = GetAlignType();
  Expr block_size = CastInt64ToExpr(GetAlignBytes(buf->align_size));
  if (align_type.find("TRANSPOSE") != std::string::npos) {
    return CanonicalSimplify(tile * block_size);
  } else if (!align_type.empty() || axis == buf->tile_axis.get()->back()) {
    return CanonicalSimplify(floordiv((tile - 1 + block_size), block_size) * block_size);
  } else {
    return tile;
  }
}

void InequalitySolver::UpdateMemInfo() {
  auto mem_info = tiling_mem_info_.get();
  CHECK(mem_info);

  auto &linear_seq = analyzer_.linear_seq_;
  for (int idx = static_cast<int>(linear_seq.size()) - 1; idx >= 0; idx--) {
    int scope_pair_offset = linear_seq[idx].scope_pair_offset;
    auto &e = linear_seq[scope_pair_offset >= 0 ? idx : idx + scope_pair_offset];

    if (e.def != nullptr && mem_info->live_buf.count(e.def) == 0) {
      CalculateMemoryInBuffer(e.def, mem_info);
    }

    for (auto ref : e.ref) {
      if (mem_info->live_buf.count(ref) > 0) {
        continue;
      }
      CalculateMemoryInBuffer(ref, mem_info);
    }

    if (scope_pair_offset >= 0) {
      for (auto alloc : e.alloc) {
        if (mem_info->live_size[alloc->scope].defined() && mem_info->live_buf[alloc].defined()) {
          mem_info->live_size[alloc->scope] -= mem_info->live_buf[alloc];
        }
        mem_info->live_buf.erase(alloc);
      }
    }
  }
}

void InequalitySolver::UpdateMemInfoWithBufReuse() {
  auto mem_info = tiling_mem_info_.get();
  CHECK(mem_info);

  for (auto cur_time = 0; cur_time <= static_cast<int>(analyzer_.buffer_usage_timetable_.size() - 1); ++cur_time) {
    // Step 1: Release not used buffer.
    for (auto it : analyzer_.buffer_usage_timetable_) {
      auto last_use_time = it.second.second;
      if (last_use_time >= cur_time) {
        continue;
      }
      if (mem_info->live_size[it.first->scope].defined() && mem_info->live_buf[it.first].defined()) {
        mem_info->live_size[it.first->scope] -= mem_info->live_buf[it.first];
      }
      mem_info->live_buf.erase(it.first);
    }
    // Step 2: Update memory for new buffer.
    for (auto it : analyzer_.buffer_usage_timetable_) {
      auto alloc_time = it.second.first;
      if (mem_info->live_buf.count(it.first) != 0) {
        continue;
      }
      if (alloc_time == cur_time) {
        CalculateMemoryInBuffer(it.first, mem_info);
      }
    }
  }
}

Array<Expr> InequalitySolver::CollectMemoryConstraints() {
  std::unordered_map<int, std::string> memory_map = {{1, "UB"}, {2, "L1"}, {3, "L0A"}, {4, "L0B"}, {5, "L0C"}};
  auto mem_info = tiling_mem_info_.get();
  Array<Expr> memory_constraints;
  for (int i = 1; i < MEM_SCOPE_BULK; ++i) {
    if (!mem_info->max_live_size[i].defined()) {
      continue;
    }

    Expr constraint = ktvm::ir::CanonicalSimplify(mem_info->max_live_size[i] <= CastInt64ToExpr(mem_limit_[i]));
    if (analyzer_.arith_ana_.CanProve(constraint == 0)) {
      LOG(WARNING) << "Memory " << i << " exceed limit, " << mem_info->max_live_size[i] << " vs " << mem_limit_[i];
      continue;
    } else if (analyzer_.arith_ana_.CanProve(constraint == 1)) {
      continue;
    }

    memory_constraints.push_back(constraint);
    param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[MemoryLimit_" + memory_map[i] + "]"), constraint});
  }
  return memory_constraints;
}

bool InequalitySolver::ContainVar(Expr expr, Var var) {
  if (const auto v = expr.as<Variable>()) {
    return (v->name_hint == var->name_hint);
  } else if (const auto a = expr.as<Add>()) {
    return (ContainVar(a->a, var) || ContainVar(a->b, var));
  } else if (const auto s = expr.as<Sub>()) {
    return (ContainVar(s->a, var) || ContainVar(s->b, var));
  } else if (const auto m = expr.as<Mul>()) {
    return (ContainVar(m->a, var) || ContainVar(m->b, var));
  } else if (const auto d = expr.as<Div>()) {
    return (ContainVar(d->a, var) || ContainVar(d->b, var));
  } else if (const auto fd = expr.as<FloorDiv>()) {
    return (ContainVar(fd->a, var) || ContainVar(fd->b, var));
  } else if (const auto c = expr.as<Cast>()) {
    return (ContainVar(c->value, var));
  } else if (const auto le = expr.as<LE>()) {
    return (ContainVar(le->a, var) || ContainVar(le->b, var));
  }
  return false;
}

///////////////////////////////////////////////////////////

TileCandidate *DynamicShapeSolver::Solve() {
  auto result = this->solver_.Solve();
  auto tile_band_size = static_cast<int>(analyzer_.RootAxis()->children.size());
  for (auto band = 0; band < tile_band_size; ++band) {
    tiling_band_ = band;
    AppendTileConstraintInIR(result, TileLevel::LEVEL1);
    if (analyzer_.op_type_ == GEMM_OP) {
      AppendTileConstraintInIR(result, TileLevel::LEVEL0);
    }
  }
  return result;
}

void DynamicShapeSolver::AppendTileConstraintInIR(TileCandidate *cand, TileLevel level) {
  auto Append = [this, level, cand](TileAxis *axis) {
    if (axis->parent == nullptr || axis->index != this->tiling_band_) {
      return;
    }

    TileAxis::Constraint cons = level == LEVEL1 ? axis->l1_constraints : axis->l0_constraints;
    Expr tile_var = level == LEVEL1 ? cand->tile_val_[axis].tile_l1 : cand->tile_val_[axis].tile_l0;
    CHECK(tile_var.defined());
    if (analyzer_.arith_ana_.CanProve(tile_var == axis->range_extent) || tile_var.as<IntImm>() != nullptr) {
      return;
    }

    // add mod constraint attr
    if (!analyzer_.arith_ana_.CanProve(cons.tile_mod_ == 1)) {
      Expr mod_cons = (floormod(tile_var, cons.tile_mod_) == 0);
      this->solver_.param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[ModConstraint]"), mod_cons});
    }

    // add forbid isolate constraint attr
    if (axis->forbid_iso) {
      Expr iso_cons = (floormod(axis->range_extent, tile_var) == 0);
      this->solver_.param_info_.push_back(Scop::ParamInfo{"AttrStmt", Expr("[IsolateConstraint]"), iso_cons});
    }
  };
  analyzer_.ForEachAxisTopDown(Append);
}

///////////////////////////////////////////////////////////

TileCandidate *TraverseSolver::Solve() {
  CollectMemoryLimit();
  auto tile_band_size = static_cast<int>(analyzer_.RootAxis()->children.size());
  for (auto band = 0; band < tile_band_size; ++band) {
    tiling_band_ = band;
    CollectTileAxisTopDown();

    // tile all axis top down
    for (TileAxis *axis : cand_.GetTileAxis()) {
      std::unique_ptr<TileInfo> info(new (std::nothrow) TileInfo(axis, LEVEL1, band));
      CHECK(info) << "memory alloc fail";
      if (IsTilable(info.get())) {
        if (DoTiling(info.get())) break;
      }
    }

    if (analyzer_.op_type_ == GEMM_OP) {
      for (TileAxis *axis : cand_.GetTileAxis()) {
        std::unique_ptr<TileInfo> info(new (std::nothrow) TileInfo(axis, LEVEL0, band));
        CHECK(info) << "memory alloc fail";
        if (IsTilable(info.get())) {
          if (DoTiling(info.get())) break;
        }
      }

      std::vector<TileAxis *> ko_axes = this->analyzer_.GetAxesOfAttr(AttrInfo{"GEMM", "ko"});
      std::vector<TileAxis *> mo_axes = this->analyzer_.GetAxesOfAttr(AttrInfo{"GEMM", "mo"});
      std::vector<TileAxis *> no_axes = this->analyzer_.GetAxesOfAttr(AttrInfo{"GEMM", "no"});

      auto MakeL1L0Consistency = [this](const std::vector<TileAxis *> &axes) {
        if (axes.size() == 1U) {
          cand_.UpdateConstTile(axes[0], this->cand_.GetConstTileVal(axes[0]).second);
        }
      };

      MakeL1L0Consistency(ko_axes);
      MakeL1L0Consistency(mo_axes);
      MakeL1L0Consistency(no_axes);
    }
  }

  if (analyzer_.op_type_ == CONV_OP) {
    if (analyzer_.scop_->IsConvBackpropFilter()) {
      AppendConvBackpropPragma();
    } else {
      AppendConvPragma();
    }
  }
  return &cand_;
}

bool TraverseSolver::IsTilable(TileInfo *info) {
  TileAxis *axis = info->axis;
  TileLevel level = info->level;
  int64_t deviation = EXCEED_MEM_CODE;

  // Step 1: Probe by min tile, to verify memory.
  int min_tile;
  TileAxis::Constraint cons = axis->GetConstConstraint(level);
  int const_extent = axis->GetConstExtent();
  if (const_extent == -1) {
    return false;
  }

  if (level == LEVEL1) {
    min_tile = cons.tile_mod_.as<IntImm>()->value;

    if ((info->axis->forbid_iso && const_extent % min_tile != 0) || (cons.tile_min_.as<IntImm>()->value > min_tile) ||
        (cons.tile_min_.as<IntImm>()->value == MIN_TILE)) {
      min_tile = cons.tile_min_.as<IntImm>()->value;
    }
    if (axis->range_min > min_tile) {
      min_tile = axis->range_min;
    }

    cand_.UpdateConstTile(axis, min_tile);
  } else {
    if (cand_.GetConstTileVal(info->axis).first == TileVarId::UNDEFINE) {
      analyzer_.logger_.LogFatalAndSaveLog("Should tile L1 first!");
    }

    min_tile = cons.tile_min_.as<IntImm>()->value;

    if (min_tile < cons.tile_mod_.as<IntImm>()->value) {
      min_tile = cons.tile_mod_.as<IntImm>()->value;
    }

    cand_.UpdateConstTile(axis, cand_.GetConstTileVal(axis).first, min_tile);
  }
  info->min_tile = min_tile;

  // Step 2: Set all fix axis before verify memory.
  cand_.UpdateFixTileAxis(level);

  bool mem_ok = MemoryVerify(level, info->band, &deviation);
  std::stringstream ss;
  ss << "Begin ::: mem ok = " << mem_ok << " dev " << deviation;
  analyzer_.logger_.AppendLog(DO_TILING, ss);
  info->deviation = deviation;
  return mem_ok;
}

bool TraverseSolver::MemoryVerify(TileLevel level, int band, int64_t *deviation) {
  std::vector<int64_t> original_size;
  std::vector<int64_t> expanded_size;
  int dev = 0;
  for (int i = 0; i < MEM_SCOPE_BULK; ++i) {
    auto scope = static_cast<DavinciMemScope>(i);
    std::pair<int64_t, int64_t> mem_pair = cand_.MemInfer(scope, band);
    int64_t origin = mem_pair.first;
    int64_t expand = mem_pair.second;
    int dev_a = EXCEED_MEM_CODE;
    if (origin <= mem_limit_[scope]) {
      dev_a = mem_limit_[scope] - origin;
    }
    if (level == LEVEL0 && i > MEM_SCOPE_UB) {
      if (dev_a != EXCEED_MEM_CODE) dev += dev_a;
    } else if (scope == MEM_SCOPE_UB) {
      dev += dev_a;
    }
    original_size.emplace_back(origin);
    expanded_size.emplace_back(expand);
  }
  if (deviation) {
    *deviation = dev;
  }

  bool L1_valid = (expanded_size[MEM_SCOPE_L1] <= mem_limit_[MEM_SCOPE_L1]);
  bool UB_valid = (expanded_size[MEM_SCOPE_UB] <= mem_limit_[MEM_SCOPE_UB]);
  bool L0A_valid = (expanded_size[MEM_SCOPE_L0A] <= mem_limit_[MEM_SCOPE_L0A]);
  bool L0B_valid = (expanded_size[MEM_SCOPE_L0B] <= mem_limit_[MEM_SCOPE_L0B]);
  bool L0C_valid = (expanded_size[MEM_SCOPE_L0C] <= mem_limit_[MEM_SCOPE_L0C]);
  bool cut_reduce = analyzer_.scop_->IsConvBackpropFilter();

  std::vector<TileAxis *> batch_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "N"});
  std::vector<TileAxis *> h_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "H"});
  std::vector<TileAxis *> w_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "W"});

  if (cut_reduce) {
    cut_reduce = ((batch_axes.size() == 1U && batch_axes[0]->GetConstExtent() > 1) ||
                  (h_axes.size() == 1U && h_axes[0]->GetConstExtent() > 1) ||
                  (w_axes.size() == 1U && w_axes[0]->GetConstExtent() > 1));
  }
  if ((!cut_reduce && level == LEVEL1 && (!L1_valid || (!UB_valid && analyzer_.op_type_ == VECTOR_OP))) ||
      ((cut_reduce || level == LEVEL0) && (!L0A_valid || !L0B_valid || !L0C_valid))) {
    return false;
  }
  return true;
}

bool TraverseSolver::DoTiling(const TileInfo *info) {
  bool success = false;
  TileAxis *axis = info->axis;
  int64_t deviation = info->deviation;
  int64_t best_val = TileVarId::UNDEFINE;
  int64_t best_no_iso_val = TileVarId::UNDEFINE;

  if (cand_.SpaceVerify(axis, info->level, info->band)) {
    best_val = info->min_tile;
    best_no_iso_val = info->min_tile;
    cand_.UpdateConstTile(axis, info->min_tile);
  }

  int64_t best_devs = deviation;
  int64_t best_no_iso_devs = deviation;
  int64_t balance_factor = analyzer_.scop_->pragma_allow_tail_tiling_ ? 1 : GetMaxAlignBytes(axis->data_size);

  TileAxis::Constraint cons = axis->GetConstConstraint(info->level);
  CHECK_GT(cons.tile_extent_.as<IntImm>()->value, 0) << "Static shape's L1 max factor should be positive integer";
  int64_t init = info->min_tile;
  int64_t dst = info->level == LEVEL1 ? cons.tile_extent_.as<IntImm>()->value : this->cand_.GetConstTileVal(axis).first;

  int64_t mod = cons.tile_mod_.as<IntImm>()->value;
  bool check_mod = dst >= mod;
  if (axis->forbid_iso) check_mod = (dst % mod == 0);

  std::stringstream ss;
  ss << "start to tile from " << init << " to " << dst;
  analyzer_.logger_.AppendLog(DO_TILING, ss);
  for (int64_t t = init; t <= dst; ++t) {
    if ((axis->forbid_iso && dst % t != 0) || (check_mod && t % mod != 0)) {
      continue;
    }
    if (info->level == LEVEL1) {
      cand_.UpdateConstTile(axis, t);
    } else {
      cand_.UpdateConstTile(axis, cand_.GetConstTileVal(axis).first, t);
    }

    if (!cand_.SpaceVerify(axis, info->level, info->band)) continue;
    bool mem_ok = MemoryVerify(info->level, info->band, &deviation);

    if (deviation < 0) {
      ss << "factor " << t << " exceed memory, exit";
      analyzer_.logger_.AppendLog(DO_TILING, ss);
      break;
    }

    if (!mem_ok) continue;
    success = true;
    auto tail = dst % t;
    if (tail == 0) {
      if (deviation > best_no_iso_devs) continue;
      ss << "factor " << t << " has " << deviation << " deviation, update to no isolate factor";
      best_no_iso_val = t;
      best_no_iso_devs = deviation;
    } else {
      if (deviation > best_devs) continue;
      if (analyzer_.scop_->pragma_allow_tail_tiling_ && tail < GetMaxAlignBytes(axis->data_size)) {
        ss << "factor " << t << " has " << tail << " tail that may disable multicore, skip.";
        continue;
      }
      ss << "factor " << t << " has " << deviation << " deviation, update to isolate factor";
      best_val = t;
      best_devs = deviation;
    }
    analyzer_.logger_.AppendLog(DO_TILING, ss);
  }

  int64_t final_factor = (axis->forbid_iso || best_no_iso_val * balance_factor > best_val) ? best_no_iso_val : best_val;
  final_factor = PostprocessFinalFactor(final_factor, axis);
  if (info->level == LEVEL1) {
    cand_.UpdateConstTile(axis, final_factor);
  } else {
    cand_.UpdateConstTile(axis, cand_.GetConstTileVal(axis).first, final_factor);
  }
  return success;
}

int64_t TraverseSolver::PostprocessFinalFactor(int64_t final_factor, TileAxis *axis) {
  auto processed = final_factor;
  if (processed == TileVarId::UNDEFINE) {
    processed = MIN_TILE;
  }

  if (analyzer_.scop_->pragma_analyze_multicore_ && !analyzer_.is_dynamic_ && analyzer_.op_type_ == VECTOR_OP) {
    MulticoreStrategy mc_strategy_ = MulticoreStrategy(cand_, analyzer_.logger_.GetDumpDir());
    processed = mc_strategy_.AdjustTilingAccordingToMulticoreConstraint(axis, processed);
  }
  std::stringstream ss;
  ss << "final factor " << processed;
  analyzer_.logger_.AppendLog(DO_TILING, ss);
  return processed;
}

void TraverseSolver::AppendConvPragma() {
  Expr no = CastIntToExpr(1);
  Expr M = CastIntToExpr(1);
  Expr ko = CastIntToExpr(1);
  Expr c_cut = CastIntToExpr(16);
  Expr kh_cut = CastIntToExpr(1);
  Expr kw_cut = CastIntToExpr(1);
  std::vector<TileAxis *> c_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "C1"});
  if (c_axes.size() == 1U) {
    c_cut *= cand_.GetTileVal(c_axes[0]).first;
    no *= cand_.GetTileVal(c_axes[0]).first;
  } else {
    c_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "C1_in_out"});
    if (c_axes.size() == 1U) {
      c_cut *= cand_.GetTileVal(c_axes[0]).first;
      no *= cand_.GetTileVal(c_axes[0]).first;
      ko *= cand_.GetTileVal(c_axes[0]).first;
    }
  }
  Expr tile_out_h = 1;
  std::vector<TileAxis *> h_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "H"});
  if (h_axes.size() == 1U) {
    tile_out_h *= cand_.GetTileVal(h_axes[0]).first;
    M *= cand_.GetTileVal(h_axes[0]).first;
  }
  Expr tile_out_w = 1;
  std::vector<TileAxis *> w_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "W"});
  if (w_axes.size() == 1U) {
    tile_out_w *= cand_.GetTileVal(w_axes[0]).first;
    M *= cand_.GetTileVal(w_axes[0]).first;
  }
  std::vector<TileAxis *> kc_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "C1_in"});
  if (kc_axes.size() == 1U) {
    ko *= cand_.GetTileVal(kc_axes[0]).first;
  }
  std::vector<TileAxis *> kh_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "kh"});
  if (kh_axes.size() == 1U) {
    ko *= cand_.GetTileVal(kh_axes[0]).first;
    kh_cut *= cand_.GetTileVal(kh_axes[0]).first;
  }
  std::vector<TileAxis *> kw_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "kw"});
  if (kw_axes.size() == 1U) {
    ko *= cand_.GetTileVal(kw_axes[0]).first;
    kw_cut *= cand_.GetTileVal(kw_axes[0]).first;
  }
  CHECK(M.defined());
  M = CanonicalSimplify((floordiv((M - 1 + CUBE_UNIT), CUBE_UNIT)) * CUBE_UNIT);
  Expr mo = CanonicalSimplify(floordiv(M, CUBE_UNIT));
  CreateSpecgemmTileAxis(mo, no, ko, false);
  this->cand_.SetBatchAxis(spec_tile_axis_);
  if (analyzer_.is_dynamic_) {
    cand_.InitTileAxis(LEVEL0);
  } else {
    for (TileAxis *axis : this->cand_.GetTileAxis()) {
      std::unique_ptr<TileInfo> info(new (std::nothrow) TileInfo(axis, LEVEL0, 0));
      CHECK(info) << "memory alloc fail";
      if (IsTilable(info.get())) {
        static_cast<void>(DoTiling(info.get()));
      }
    }
  }
  Expr cin_cut;
  Expr batch_cut;
  CreateConvPragma(c_cut, tile_out_h, tile_out_w, kh_cut, kw_cut, cin_cut, batch_cut);
}

void TraverseSolver::AppendConvBackpropPragma() {
  Expr no = 1;
  Expr mo = 1;
  Expr ko = 1;
  Expr cin_cut = 16;
  Expr co_cut = 16;
  Expr batch_cut = 1;
  Expr kh_cut = 1;
  Expr kw_cut = 1;
  bool cut_reduce = false;
  ktvm::arith::Analyzer arith_ana;
  std::vector<TileAxis *> batch_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "N"});
  if (batch_axes.size() == 1U) {
    batch_cut *= cand_.GetTileVal(batch_axes[0]).first;
    cut_reduce = cut_reduce || arith_ana.CanProve(batch_cut < batch_axes[0]->range_extent);
    ko *= cand_.GetTileVal(batch_axes[0]).first;
  }
  Expr tile_out_h = 1;
  std::vector<TileAxis *> h_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "H"});
  if (h_axes.size() == 1U) {
    tile_out_h *= cand_.GetTileVal(h_axes[0]).first;
    cut_reduce = cut_reduce || arith_ana.CanProve(tile_out_h < h_axes[0]->range_extent);
    ko *= cand_.GetTileVal(h_axes[0]).first;
  }
  Expr tile_out_w = 1;
  std::vector<TileAxis *> w_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "W"});
  if (w_axes.size() == 1U) {
    tile_out_w *= cand_.GetTileVal(w_axes[0]).first;
    cut_reduce = cut_reduce || arith_ana.CanProve(tile_out_w < h_axes[0]->range_extent);
    ko *= cand_.GetTileVal(w_axes[0]).first;
  }
  std::vector<TileAxis *> kc_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "C1_in"});
  if (kc_axes.size() == 1U) {
    co_cut *= cand_.GetTileVal(kc_axes[0]).first;
    mo *= cand_.GetTileVal(kc_axes[0]).first;
  }
  std::vector<TileAxis *> kh_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "kh"});
  if (kh_axes.size() == 1U) {
    ko *= cand_.GetTileVal(kh_axes[0]).first;
    no *= cand_.GetTileVal(kh_axes[0]).first;
    kh_cut *= cand_.GetTileVal(kh_axes[0]).first;
  }
  std::vector<TileAxis *> kw_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "kw"});
  if (kw_axes.size() == 1U) {
    ko *= cand_.GetTileVal(kw_axes[0]).first;
    no *= cand_.GetTileVal(kw_axes[0]).first;
    kw_cut *= cand_.GetTileVal(kw_axes[0]).first;
  }
  std::vector<TileAxis *> co_axes = analyzer_.GetAxesOfAttr(AttrInfo{"CONV", "C1_out"});
  if (co_axes.size() == 1U) {
    cin_cut *= cand_.GetTileVal(co_axes[0]).first;
    no *= cand_.GetTileVal(co_axes[0]).first;
  }

  CreateSpecgemmTileAxis(mo, no, ko, cut_reduce);
  this->cand_.SetBatchAxis(spec_tile_axis_);
  if (analyzer_.is_dynamic_) {
    cand_.InitTileAxis(LEVEL0);
  } else {
    for (TileAxis *axis : this->cand_.GetTileAxis()) {
      std::unique_ptr<TileInfo> info(new (std::nothrow) TileInfo(axis, LEVEL0, 0));
      CHECK(info) << "memory alloc fail";
      if (IsTilable(info.get())) {
        static_cast<void>(DoTiling(info.get()));
      }
    }
  }
  CreateConvPragma(co_cut, tile_out_h, tile_out_w, kh_cut, kw_cut, cin_cut, batch_cut);
}

void TraverseSolver::RestrainConvBackInputTileK(TileAxis *k_axis) const {
  std::unordered_map<std::string, Expr> conv_info = analyzer_.scop_->GetConvInfoForTiling();
  CHECK(conv_info.find(ATTR_CONV_KERNEL_H) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_KERNEL_W) != conv_info.end());
  Expr k_w = conv_info[ATTR_CONV_KERNEL_W];
  Expr k_h = conv_info[ATTR_CONV_KERNEL_H];
  Expr k_mod = k_h * k_w;
  k_axis->TileRestrainMod(k_mod, LEVEL0);
}

void TraverseSolver::CreateSpecgemmTileAxis(Expr mo, Expr no, Expr ko, bool cut_reduce) {
  TileAxis *mo_axis = GeneratePragmaAxes(std::move(mo), ATTR_CONV_TILE_M, false);
  TileAxis *no_axis = GeneratePragmaAxes(std::move(no), ATTR_CONV_TILE_N, false);
  TileAxis *ko_axis = GeneratePragmaAxes(std::move(ko), ATTR_CONV_TILE_K, false);
  TileAxis *mi_axis = GeneratePragmaAxes(CUBE_UNIT, ATTR_CONV_M_INNER, true);
  TileAxis *ni_axis = GeneratePragmaAxes(CUBE_UNIT, ATTR_CONV_N_INNER, true);
  TileAxis *ki_axis = GeneratePragmaAxes(CUBE_UNIT, ATTR_CONV_K_INNER, true);
  if (cut_reduce) {
    mo_axis->TileRestrainEntire(LEVEL0);
    no_axis->TileRestrainEntire(LEVEL0);
  }
  if (analyzer_.scop_->IsConvBackpropInput()) {
    RestrainConvBackInputTileK(ko_axis);
  }
  // Append axes to corresponding buffers.
  std::unordered_map<std::string, std::vector<TileAxis *>> spec_map = {
    {"L0A", {mo_axis, mi_axis, ko_axis, ki_axis}},
    {"L0B", {no_axis, ni_axis, ko_axis, ki_axis}},
    {"L0C", {mo_axis, mi_axis, no_axis, ni_axis}},
  };
  auto append_axis = [&spec_map](TilingAnalyzer::BufferEntry *buf) {
    if (buf == nullptr) return;
    for (const auto &it : spec_map) {
      std::string key = it.first;
      if (buf->name.find(key) != std::string::npos) {
        std::vector<TileAxis *> axes = it.second;
        Expr shape;
        for (auto a : axes) {
          CHECK(a);
          buf->tile_axis->emplace_back(a);
          if (shape.defined())
            shape *= a->range_extent;
          else
            shape = a->range_extent;
        }
        buf->shape = shape;
      }
    }
  };
  std::unordered_set<TilingAnalyzer::BufferEntry *> L0Buffer;
  auto process = [&L0Buffer](TilingAnalyzer::BufferEntry *buf) {
    if (buf == nullptr || buf->name.find("L0") == std::string::npos) return;
    buf->tile_axis->clear();
    buf->shape = 1;
    L0Buffer.insert(buf);
  };
  for (const auto &stmt : analyzer_.linear_seq_) {
    process(stmt.def);
    for (auto b : stmt.ref) process(b);
    for (auto b : stmt.alloc) process(b);
  }
  for (auto buf : L0Buffer) append_axis(buf);
}

void TraverseSolver::CreateConvPragma(const Expr &co_cut, Expr tile_out_h, Expr tile_out_w, Expr kh_cut, Expr kw_cut,
                                      Expr ci_cut, const Expr &batch_cut) {
  std::unordered_map<std::string, Expr> conv_info = analyzer_.scop_->GetConvInfoForTiling();
  CHECK(conv_info.find(ATTR_CONV_STRIDE_H) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_DILATION_H) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_KERNEL_H) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_STRIDE_W) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_DILATION_W) != conv_info.end());
  CHECK(conv_info.find(ATTR_CONV_KERNEL_W) != conv_info.end());

  Expr s_h = conv_info[ATTR_CONV_STRIDE_H];
  Expr s_w = conv_info[ATTR_CONV_STRIDE_W];
  Expr k_h = conv_info[ATTR_CONV_KERNEL_H];
  Expr k_w = conv_info[ATTR_CONV_KERNEL_W];
  Expr d_h = conv_info[ATTR_CONV_DILATION_H];
  Expr d_w = conv_info[ATTR_CONV_DILATION_W];
  Expr k_h_d = (k_h - 1) * d_h + 1;
  Expr k_w_d = (k_w - 1) * d_w + 1;
  Expr h_cut = (tile_out_h - 1) * s_h + k_h_d;
  Expr w_cut = (tile_out_w - 1) * s_w + k_w_d;

  TileAxis *pragma_cout = GeneratePragmaAxes(co_cut, ATTR_CONV_TILE_CO, true);
  TileAxis *pragma_h = GeneratePragmaAxes(h_cut, ATTR_CONV_TILE_H, true);
  TileAxis *pragma_w = GeneratePragmaAxes(w_cut, ATTR_CONV_TILE_W, true);
  TileAxis *pragma_kh = GeneratePragmaAxes(kh_cut, ATTR_CONV_TILE_KH, true);
  TileAxis *pragma_kw = GeneratePragmaAxes(kw_cut, ATTR_CONV_TILE_KW, true);

  cand_.UpdateTile(pragma_cout, co_cut, co_cut);
  cand_.UpdateTile(pragma_h, h_cut, h_cut);
  cand_.UpdateTile(pragma_w, w_cut, w_cut);
  cand_.UpdateTile(pragma_kh, kh_cut, kh_cut);
  cand_.UpdateTile(pragma_kw, kw_cut, kw_cut);

  // Channel-in cut and batch cut pragma are used in conv backprop filter.
  if (ci_cut.defined()) {
    TileAxis *pragma_cin = GeneratePragmaAxes(ci_cut, ATTR_CONV_TILE_CIN, true);
    cand_.UpdateTile(pragma_cin, ci_cut, ci_cut);
  }
  if (batch_cut.defined()) {
    TileAxis *pragma_b = GeneratePragmaAxes(batch_cut, ATTR_CONV_TILE_B, true);
    cand_.UpdateTile(pragma_b, batch_cut, batch_cut);
  }
}

TileAxis *TraverseSolver::GeneratePragmaAxes(const Expr &size, const std::string &type, bool is_pragma) {
  std::unique_ptr<TileAxis> axis(new (std::nothrow) TileAxis(size, size, type, &this->analyzer_, is_pragma));
  CHECK(axis) << "memory alloc fail";
  analyzer_.RootAxis()->children.emplace_back(std::move(axis));
  TileAxis *a = analyzer_.RootAxis()->children.back().get();
  spec_tile_axis_.emplace_back(a);
  this->cand_.InsertAxisBack(a);
  return a;
}
std::vector<TileAxis *> TraverseSolver::GetSpecTileAxis() { return this->spec_tile_axis_; }

}  // namespace poly
}  // namespace ir
}  // namespace akg
