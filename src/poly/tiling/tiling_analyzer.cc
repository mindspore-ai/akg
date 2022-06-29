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
#include "tiling_analyzer.h"

#include <tvm/ir.h>
#include <tvm/ir_visitor.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

#include "poly/tiling/schtree_analyzer.h"
#include "poly/tiling/space_analyzer.h"
#include "poly/tiling/tiling_strategy_manager.h"
#include "poly/dsa_utils.h"

namespace akg {
namespace ir {
namespace poly {
TileAxis::TileAxis(TileAxis *p, int i, int da, bool mc, const std::pair<std::string, int> &ds, bool inner,
                   TilingAnalyzer *ta)
    : parent(p),
      index(i),
      dim_axis(da),
      mc_sup(mc),
      range_min(0),
      range_extent(MIN_TILE),
      extent_val(MIN_TILE),
      forbid_iso(false),
      is_inner(inner),
      analyzer_(ta) {
  data_size[ds.first].emplace_back(ds.second);
  c1_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
  c1_constraints.tile_extent_ = CastIntToExpr(MIN_TILE);

  c0_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
  c0_constraints.tile_extent_ = CastIntToExpr(MIN_TILE);
  if (is_inner) {
    this->TileRestrainEntire(CACHE1);
    this->TileRestrainEntire(CACHE0);
  }
}

TileAxis::TileAxis(const Expr &l1_size, const Expr &l0_size, const std::string &at, TilingAnalyzer *ta, bool inner)
    : forbid_iso(false), is_inner(inner), axis_type_(std::move(at)), analyzer_(ta) {
  is_pragma = true;
  range_min = MIN_TILE;
  range_extent = l1_size;
  extent_val = MIN_TILE;

  c1_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
  c0_constraints.tile_min_ = CastIntToExpr(MIN_TILE);
  c1_constraints.tile_extent_ = l1_size;
  c0_constraints.tile_extent_ = std::move(l0_size);
  if (is_inner) {
    this->TileRestrainEntire(CACHE1);
    this->TileRestrainEntire(CACHE0);
  }
}

int64_t GetExtentVal(Expr &range_extent, TilingAnalyzer *analyzer) {
  if (analyzer->scop_info_.analysis_result_.IsCsrDynamicExtent(range_extent)) {
    return std::max(1, analyzer->scop_info_.user_config_.GetCsrThreadNum());
  }
  auto extent = range_extent.as<IntImm>();
  if (extent != nullptr) {
    return extent->value;
  }
  return MIN_TILE;
}

void TileAxis::LinkToLoop(const For *loop) {
  CHECK(loop) << "Link to nullptr, please check";
  const auto offset = loop->min.as<IntImm>();
  CHECK(offset) << "Loop's offset contains Expr, please check";
  int64_t offset_int = offset->value;
  if (this->loops.empty()) {
    this->range_min = offset_int;
    this->range_extent = loop->extent.as<IntImm>() ? CanonicalSimplify(loop->min + loop->extent) : loop->extent;
    this->extent_val = GetExtentVal(this->range_extent, analyzer_);
  } else if (std::count(this->loops.begin(), this->loops.end(), loop) == 0) {
    if (this->range_extent.as<IntImm>()) {
      if (analyzer_->arith_ana_.CanProve(this->range_extent != loop->extent)) {
        if (this->range_min > offset_int) {
          this->range_min = offset_int;
        }
        if (analyzer_->arith_ana_.CanProve(this->range_extent < (loop->min + loop->extent))) {
          this->range_extent = CanonicalSimplify(loop->min + loop->extent);
          this->extent_val = GetExtentVal(this->range_extent, analyzer_);
        }
      }
    }
  } else {
    return;
  }
  this->loops.emplace_back(loop);

  this->c1_constraints.tile_min_ = this->range_min == 0 ? CastIntToExpr(MIN_TILE) : CastIntToExpr(this->range_min);
  this->c1_constraints.tile_extent_ = this->range_extent;
  this->c0_constraints.tile_min_ = this->c1_constraints.tile_min_;
  this->c0_constraints.tile_extent_ = this->c1_constraints.tile_extent_;
}

void TileAxis::MarkWithAttr(const AttrInfo &attr) {
  for (const auto &old_attr : this->attrs) {
    if (old_attr.attr_key == attr.attr_key && old_attr.attr_value == attr.attr_value) {
      return;
    }
  }
  this->attrs.emplace_back(attr);
}

std::vector<std::string> TileAxis::GetAttrValue(const std::string &attr_key) const {
  std::vector<std::string> match;
  for (const auto &attr : this->attrs) {
    if (attr.attr_key == attr_key) {
      match.emplace_back(attr.attr_value);
    }
  }
  return match;
}

void TileAxis::InsertC1CandFactor(const Expr &f) {
  size_t i = 0;
  while (i < this->c1_constraints.cand_factor.size()) {
    if (Equal(this->c1_constraints.cand_factor[i], f)) {
      return;
    } else if (analyzer_->arith_ana_.CanProve(this->c1_constraints.cand_factor[i] < f)) {
      break;
    }
    ++i;
  }
  this->c1_constraints.cand_factor.insert(this->c1_constraints.cand_factor.begin() + i, f);
}

void TileAxis::InsertC0CandFactor(const Expr &f) {
  size_t i = 0;
  while (i < this->c0_constraints.cand_factor.size()) {
    if (Equal(this->c0_constraints.cand_factor[i], f)) {
      return;
    } else if (analyzer_->arith_ana_.CanProve(this->c0_constraints.cand_factor[i] < f)) {
      break;
    }
    ++i;
  }
  this->c0_constraints.cand_factor.insert(this->c0_constraints.cand_factor.begin() + i, f);
}

void TileAxis::DumpAxis(bool on_screen) {
  std::stringstream ss;
  std::string tag = this->is_pragma ? this->axis_type_ : std::to_string(this->dim_axis);
  ss << "| Axis (" << this << ") " << this->index << "_" << tag << "| Parent " << this->parent << " | Is inner "
     << this->is_inner << "| Range [" << this->range_min << "," << this->range_extent << "]"
     << "| L1 Tile [" << this->c1_constraints.tile_min_ << "," << this->c1_constraints.tile_extent_ << "]"
     << "| Data size {";
  for (const auto &it : this->data_size) {
    ss << it.first << ":";
    for (const auto &sz : it.second) {
      ss << sz << ", ";
    }
  }
  ss << "} | Align to = " << this->c1_constraints.tile_mod_ << "| L0 Tile [" << this->c0_constraints.tile_min_ << ","
     << this->c0_constraints.tile_extent_ << "] "
     << "| Thread mapping constraints: [" << this->thread_constraints.map_min_ << ", "
     << this->thread_constraints.map_extent_ << "]"
     << "| Block mapping constraints: [" << this->block_constraints.map_min_ << ", "
     << this->block_constraints.map_extent_ << "]"
     << "| Align to = " << this->c0_constraints.tile_mod_ << "| Forbid isolate = " << this->forbid_iso
     << "| Multi-core support = " << this->mc_sup << "| Priority = " << this->priority << "| Loops : {";
  for (auto loop : this->loops) {
    ss << loop->loop_var.get()->name_hint << ",";
  }
  ss << "} |";
  if (on_screen) LOG(INFO) << ss.str();
  analyzer_->GetTileLogger().AppendLog(ANA_TILING_SPACE, ss);
  if (!this->attrs.empty()) {
    ss << "| Attrs:{";
    int line_sep = 7;
    for (unsigned i = 0; i < this->attrs.size(); ++i) {
      auto attr = this->attrs[i];
      ss << "(" << attr.attr_key << ":" << attr.attr_value << "),";
      if (i > 0 && i % line_sep == 0) {
        if (on_screen) LOG(INFO) << ss.str();
        analyzer_->GetTileLogger().AppendLog(ANA_TILING_SPACE, ss);
      }
    }
    ss << "} |";
    if (on_screen) LOG(INFO) << ss.str();
    analyzer_->GetTileLogger().AppendLog(ANA_TILING_SPACE, ss);
  }
  if (!this->c1_constraints.cand_factor.empty()) {
    ss << "| L1 Cand_factors:{";
    bool full_dump = this->c1_constraints.cand_factor.size() <= 10;
    if (full_dump) {
      for (const auto &f : this->c1_constraints.cand_factor) {
        ss << f << ",";
      }
    } else {
      ss << this->c1_constraints.cand_factor[0] << " ... " << this->c1_constraints.cand_factor.back();
    }
    ss << "} |";
    if (on_screen) LOG(INFO) << ss.str();
    analyzer_->GetTileLogger().AppendLog(ANA_TILING_SPACE, ss);
  }
  if (!this->c0_constraints.cand_factor.empty()) {
    ss << "| L0 Cand_factors:{";
    bool full_dump = this->c0_constraints.cand_factor.size() <= 10;
    if (full_dump) {
      for (const auto &f : this->c0_constraints.cand_factor) {
        ss << f << ",";
      }
    } else {
      ss << this->c0_constraints.cand_factor[0] << " ... " << this->c0_constraints.cand_factor.back();
    }
    ss << "} |";
    if (on_screen) LOG(INFO) << ss.str();
    analyzer_->GetTileLogger().AppendLog(ANA_TILING_SPACE, ss);
  }
}

void TileAxis::TileRestrainMod(const Expr &mod, TileLevel level) {
  CHECK(analyzer_->arith_ana_.CanProve(mod != 0));
  auto &constraint = level == CACHE1 ? this->c1_constraints : this->c0_constraints;
  Expr ori_mod = constraint.tile_mod_;
  Expr gcd = analyzer_->expr_ac_.Gcd(mod, ori_mod);
  CHECK(analyzer_->arith_ana_.CanProve(gcd != 0));
  Expr lcm = CanonicalSimplify(floordiv(mod * ori_mod, gcd));
  constraint.tile_mod_ = lcm;
}

void TileAxis::TileRestrainUpper(const Expr &value, TileLevel level) {
  auto &constraint = level == CACHE1 ? this->c1_constraints : this->c0_constraints;
  auto old_upper = constraint.tile_extent_;
  auto new_value = value.type() == old_upper.type() ? value : Cast::make(old_upper.type(), value);
  auto new_upper = CanonicalSimplify(Min::make(old_upper, new_value));
  new_upper = CanonicalSimplify(Max::make(constraint.tile_min_, new_upper));
  constraint.tile_extent_ = new_upper;
}

void TileAxis::TileRestrainLower(const Expr &value, TileLevel level) {
  auto &constraint = level == CACHE1 ? this->c1_constraints : this->c0_constraints;
  auto old_lower = constraint.tile_min_;
  auto new_value = value.type() == old_lower.type() ? value : Cast::make(old_lower.type(), value);
  auto new_lower = CanonicalSimplify(Max::make(old_lower, new_value));
  new_lower = CanonicalSimplify(Min::make(constraint.tile_extent_, new_lower));
  constraint.tile_min_ = new_lower;
}

void TileAxis::TileRestrainToSingleValue(const Expr &value, TileLevel level) {
  auto &constraint = level == CACHE1 ? this->c1_constraints : this->c0_constraints;
  constraint.tile_min_ = value;
  constraint.tile_extent_ = value;
}

void TileAxis::TileRestrainEntire(TileLevel level) {
  if (level == CACHE1) {
    Expr extent;
    if (analyzer_->scop_info_.analysis_result_.IsCsrDynamicExtent(range_extent)) {
      extent = this->extent_val;
    } else {
      extent = range_extent;
    }
    if (this->HasAttr(AT_SHIFT)) extent = this->c1_constraints.tile_extent_;
    this->c1_constraints.tile_min_ = extent;
    this->c1_constraints.tile_extent_ = extent;
  } else {
    this->c0_constraints.tile_min_ = this->c1_constraints.tile_extent_;
    this->c0_constraints.tile_extent_ = this->c1_constraints.tile_extent_;
  }
}

void TileCandidate::SetBatchAxis(const std::vector<TileAxis *> &axis) { this->tile_axis_ = axis; }

void TileCandidate::InitTileAxis(TileLevel level) {
  dynamic_mem_info_ = std::unique_ptr<DynamicMemInfo>(new (std::nothrow) DynamicMemInfo());
  CHECK(dynamic_mem_info_) << "memory alloc fail";
  for (auto axis : tile_axis_) {
    TileAxis::Constraint cons = axis->GetConstConstraint(level);
    auto Update = [this, level, axis](const Expr &tile) {
      if (level == CACHE1) {
        this->UpdateTile(axis, tile);
      } else {
        this->UpdateTile(axis, this->GetTileVal(axis).first, tile);
      }
    };

    // For axis with dynamic shape, simply create tile var and store them
    // generated var.
    std::string var_name = level == CACHE1 ? "T1_" : "T0_";
    var_name += std::to_string(axis->index) + "_";
    var_name += axis->axis_type_.empty() ? std::to_string(axis->dim_axis) : axis->axis_type_;
    // unify var address
    Var tile_var;
    if (dynamic_mem_info_->tile_var_map.find(var_name) == dynamic_mem_info_->tile_var_map.end()) {
      tile_var = Var(var_name, Int(32));
      dynamic_mem_info_->tile_var_map[var_name] = tile_var;
    } else {
      tile_var = dynamic_mem_info_->tile_var_map[var_name];
    }
    axis->var_names[var_name] = tile_var;
    Update(tile_var);

    if (cons.tile_extent_.as<IntImm>()->value != -1) {
      // These are two cases when tiling factor is fixed for axis with static shape:
      // 1. tile_min == tile_extent ==> tile factor = tile_extent
      // 2. contains only one tile candidate ==> tile factor = this candidate
      if (cons.tile_min_.as<IntImm>()->value == cons.tile_extent_.as<IntImm>()->value) {
        Update(CastInt64ToExpr(cons.tile_extent_.as<IntImm>()->value));
      } else if (cons.cand_factor.size() == 1U) {
        Update(CastInt64ToExpr(cons.cand_factor[0].as<IntImm>()->value));
      }
    }
  }
}

void TileCandidate::UpdateFixTileAxis(TileLevel level) {
  for (auto fix_axis : tile_axis_) {
    TileAxis::Constraint cons = fix_axis->GetConstConstraint(level);
    if (level == CACHE1) {
      if (cons.tile_min_.as<IntImm>()->value == cons.tile_extent_.as<IntImm>()->value) {
        this->UpdateConstTile(fix_axis, cons.tile_extent_.as<IntImm>()->value);
      } else if (cons.cand_factor.size() == 1U) {
        this->UpdateConstTile(fix_axis, cons.cand_factor[0].as<IntImm>()->value);
      }
    } else {
      if (this->GetConstTileVal(fix_axis).first == TileVarId::UNDEFINE) {
        continue;
      }
      if (cons.tile_min_.as<IntImm>()->value == cons.tile_extent_.as<IntImm>()->value) {
        this->UpdateConstTile(fix_axis, this->GetConstTileVal(fix_axis).first, cons.tile_extent_.as<IntImm>()->value);
      } else if (cons.cand_factor.size() == 1U) {
        this->UpdateConstTile(fix_axis, this->GetConstTileVal(fix_axis).first, cons.cand_factor[0].as<IntImm>()->value);
      }
    }
  }
}

bool TileCandidate::SpaceVerify(const TileAxis *axis, TileLevel level, const int band_idx) {
  if (axis->index != band_idx) return true;

  TileVal tile_val = this->tile_val_[axis];
  auto CheckCandfactor = [level, tile_val](const TileAxis *axis) -> bool {
    Expr tile_expr = level == CACHE1 ? tile_val.tile_c1 : tile_val.tile_c0;
    const auto tile_imm = tile_expr.as<IntImm>();
    if (tile_imm == nullptr) {
      return true;
    }
    auto tile = tile_imm->value;
    std::vector<Expr> cand = axis->GetConstConstraint(level).cand_factor;
    for (const auto &f : cand) {
      auto imm = f.as<IntImm>()->value;
      if (tile == imm) {
        return true;
      }
    }
    return false;
  };

  if (level == CACHE1) {
    if (!axis->c1_constraints.cand_factor.empty()) {
      // Reshape axis's tiling factor must chosen from a set of candidate factors.
      return CheckCandfactor(axis);
    }
  } else {
    if (!axis->c0_constraints.cand_factor.empty()) {
      // Reshape axis's tiling factor must chosen from a set of candidate factors.
      return CheckCandfactor(axis);
    }
  }
  return true;
}

std::pair<int64_t, int64_t> TileCandidate::MemInfer(TilingMemScope scope, int band_idx) {
  tiling_band_ = band_idx;
  if (!is_update_) {
    DoMemInfer();
    is_update_ = true;
  }
  return std::make_pair(mem_infer_[scope], align_mem_infer_[scope]);
}

void TileCandidate::UpdateConstTile(const TileAxis *a, int64_t c1_val, int64_t c0_val) {
  TileVal &val = this->tile_val_[a];
  val.tile_c1 = c1_val;
  val.tile_c0 = c0_val == -1 ? c1_val : c0_val;
  is_update_ = false;
}

void TileCandidate::UpdateC1Tile(const TileAxis *a, const Expr &c1_val) {
  TileVal &val = this->tile_val_[a];
  val.tile_c1 = c1_val;
  is_update_ = false;
}

void TileCandidate::UpdateC0Tile(const TileAxis *a, const Expr &c0_val) {
  TileVal &val = this->tile_val_[a];
  val.tile_c0 = c0_val;
  is_update_ = false;
}

void TileCandidate::UpdateTile(const TileAxis *a, const Expr &c1_val, const Expr &c0_val) {
  TileVal &val = this->tile_val_[a];
  val.tile_c1 = c1_val;
  if (c0_val.defined()) {
    val.tile_c0 = c0_val;
  }
  is_update_ = false;
}

std::pair<Expr, Expr> TileCandidate::GetTileVal(const TileAxis *a) {
  if (this->tile_val_.find(a) != this->tile_val_.end()) {
    TileVal &val = this->tile_val_[a];
    return {val.tile_c1, val.tile_c0};
  }
  return std::make_pair(CastIntToExpr(TileVarId::UNDEFINE), CastIntToExpr(TileVarId::UNDEFINE));
}

std::pair<int64_t, int64_t> TileCandidate::GetConstTileVal(const TileAxis *a) {
  Expr c1_expr;
  Expr c0_expr;
  std::tie(c1_expr, c0_expr) = GetTileVal(a);
  int64_t c1 = a->range_extent.as<IntImm>() ? TileVarId::UNDEFINE : TileVarId::VAR;
  int64_t c0 = a->range_extent.as<IntImm>() ? TileVarId::UNDEFINE : TileVarId::VAR;
  if (const auto c1_imm = c1_expr.as<IntImm>()) c1 = c1_imm->value;
  if (const auto c0_imm = c0_expr.as<IntImm>()) c0 = c0_imm->value;
  return std::make_pair(c1, c0);
}

int64_t TileCandidate::CalActualTile(const CalAlignInfo *align_info) {
  CHECK(align_info);
  int64_t actual_tile = align_info->tile;
  int64_t split = (align_info->divisor + align_info->tile - 1) / align_info->tile;
  auto GetAlignType = [align_info]() -> std::string {
    std::string align_type = "";
    for (const auto &attr : align_info->a->attrs) {
      if (attr.attr_key.find(AT_ALIGN) == std::string::npos) {
        continue;
      }
      std::string local_name = attr.attr_value + LOCAL_BUF;
      if (align_info->buf->name.find(local_name) == std::string::npos) {
        continue;
      }
      std::vector<std::string> res = akg::common::Split(attr.attr_key, ":");
      if (res.size() == 2U) {
        align_type = res[1];
      }
      return align_type;
    }
    return align_type;
  };
  if (this->analyzer_->scop_info_.user_config_.GetTarget() != TARGET_CCE ||
      this->analyzer_->op_type_ != TileOpType::VECTOR_OP) {
    return actual_tile;
  }
  std::string align_type = GetAlignType();
  if (align_type.find(AT_TRANSPOSE) != std::string::npos) {
    int64_t block_size = GetAlignBytes(align_info->buf->align_size);
    actual_tile = align_info->tile * block_size;
  } else if (align_type.find(AT_DMA) != std::string::npos) {
    int64_t block_size = GetAlignBytes(align_info->buf->align_size);
    int64_t gcd = air::ir::gcd(align_info->tile, block_size);
    CHECK_NE(gcd, 0);
    actual_tile = align_info->tile * block_size / gcd;
  } else if (align_type != "" || align_info->a == align_info->buf->tile_axis.get()->back()) {
    int64_t isolate_block = align_info->divisor - (split - 1) * align_info->tile;
    int64_t gcd = air::ir::gcd(align_info->tile, isolate_block);
    int64_t block_size = GetAlignBytes(align_info->buf->align_size);
    CHECK_NE(isolate_block, 0);
    CHECK_NE(gcd, 0);
    if (align_info->tile % isolate_block == 0 || gcd > block_size) {
      // When no isolate or gcd of full-tiled and isolate block is greater than block size,
      // actual tile is aligned to block size directly.
      while (actual_tile % block_size != 0) actual_tile++;
    } else {
      // When gcd of full-tiled and isolate block is smaller than block size,
      // alignment will be smaller than block size, which causes terrible expansion.
      auto expansion = static_cast<int64_t>((block_size - 1 + gcd) / gcd);
      actual_tile *= expansion;
    }
  }
  return actual_tile;
}

void TileCandidate::UpdateMemoryAfterBuffer(const BufferEntry *buf, MemInferInfo *mem_infer_info) {
  CHECK(buf);
  CHECK(mem_infer_info);
  const auto fix_size = buf->shape.as<IntImm>();
  if (fix_size == nullptr) {
    std::stringstream ss;
    ss << "Buffer " << buf->name << " contains dynamic shape " << buf->shape << ", skip.";
    analyzer_->GetTileLogger().AppendLog(DO_TILING, ss);
    return;
  }
  int64_t buf_size = buf->size * buf->expand_size * fix_size->value;
  CHECK_GT(buf_size, 0) << "Buffer size must be positive.";
  int64_t act_buf_size = buf_size;
  TilingMemScope scope = buf->scope;
  bool this_band_buf = (scope == MEM_SCOPE_GM);
  auto FindPartialMatch = [](const std::string &full_name, const std::unordered_set<std::string> name_set) -> bool {
    for (const auto &part_name : name_set) {
      if (full_name.find(part_name) != std::string::npos) {
        return true;
      }
    }
    return false;
  };
  bool is_elem = FindPartialMatch(buf->name, elem_align_buf_);
  bool is_bcast = FindPartialMatch(buf->name, broadcast_align_buf_);
  int64_t f_mul = 1;
  std::unique_ptr<BufSizeInfo> buf_size_info(new (std::nothrow)
                                               BufSizeInfo{buf_size, act_buf_size, f_mul, is_elem, is_bcast});
  CHECK(buf_size_info) << "memory alloc fail";
  if (scope != MEM_SCOPE_GM) {
    this_band_buf = GetActualBufSize(buf, buf_size_info.get());
  }
  GetElemwiseActualBufSize(buf, buf_size_info.get());

  if (this_band_buf) {
    mem_infer_info->live_buf[buf] = buf_size_info->buf_size;
    mem_infer_info->live_size[scope] += buf_size_info->buf_size;
    mem_infer_info->actual_live_size[scope] += buf_size_info->act_buf_size;
  }
  if (mem_infer_info->live_size[scope] > mem_infer_info->max_live_size[scope]) {
    mem_infer_info->max_live_size[scope] = mem_infer_info->live_size[scope];
  }
  if (mem_infer_info->actual_live_size[scope] > mem_infer_info->max_act_live_size[scope]) {
    mem_infer_info->max_act_live_size[scope] = mem_infer_info->actual_live_size[scope];
  }
}

bool TileCandidate::GetActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info) {
  bool this_band_buf = false;
  static const bool is_l0_tile[MEM_SCOPE_BULK] = {false, false, false, true, true, true, false, false};
  for (auto &it : *(buf->tile_axis)) {
    TileAxis *a = it;
    if (a == analyzer_->RootAxis()) {
      continue;
    }
    CHECK(a);
    if (a->index != tiling_band_) {
      continue;
    }
    this_band_buf = true;
    bool is_tiling = (std::count(this->tile_axis_.begin(), this->tile_axis_.end(), a) != 0);
    int64_t tile = 1;
    int64_t divisor = a->GetConstExtent();
    if (divisor == -1) {
      continue;
    }
    CHECK_GT(divisor, 0) << "Axis range must be positive.";
    if (is_tiling) {
      Expr tile_expr = is_l0_tile[buf->scope] ? this->tile_val_[a].tile_c0 : this->tile_val_[a].tile_c1;
      if (const auto tile_imm = tile_expr.as<IntImm>()) tile = tile_imm->value;
    }
    if (tile >= divisor) {
      tile = divisor;
    }
    CHECK_GT(tile, 0) << "Tile factor must be positive";
    auto split = divisor / tile;
    std::unique_ptr<CalAlignInfo> align_info(
      new (std::nothrow) CalAlignInfo{tile, divisor, a, buf, buf_size_info->is_elem, buf_size_info->is_bcast});
    CHECK(align_info) << "memory alloc fail";
    int64_t actual_tile = CalActualTile(align_info.get());
    CHECK_GT(actual_tile, 0);
    buf_size_info->f_mul *= actual_tile;
    CHECK_GT(split, 0);
    buf_size_info->buf_size = (buf_size_info->buf_size + split - 1) / split;
    if (actual_tile != tile) {
      CHECK_GT(actual_tile, 0);
      double act_split = static_cast<double>(divisor) / static_cast<double>(actual_tile);
      CHECK_NE(act_split, 0);
      if (act_split > buf_size_info->act_buf_size) {
        buf_size_info->act_buf_size = 1;
      } else {
        buf_size_info->act_buf_size =
          static_cast<int64_t>(static_cast<double>(buf_size_info->act_buf_size) / act_split);
      }
      std::stringstream ss;
      ss << "Divisor: " << divisor << " Tile: " << tile << " Bufsize: " << buf_size_info->buf_size
         << " ActTile: " << actual_tile << " ActBufSize: " << buf_size_info->act_buf_size;

      analyzer_->GetTileLogger().AppendLog(DO_TILING, ss);
    } else {
      buf_size_info->act_buf_size = buf_size_info->buf_size;
    }
  }
  return this_band_buf;
}

void TileCandidate::GetElemwiseActualBufSize(const BufferEntry *buf, BufSizeInfo *buf_size_info) {
  if (analyzer_->scop_info_.user_config_.GetTarget() != TARGET_CCE || !buf_size_info->is_elem) {
    return;
  }
  if (buf_size_info->is_bcast) {
    // Elemwise and bcast buffer cannot be reused.
    buf_size_info->act_buf_size *= 2;
    if (buf->tile_axis != nullptr && !buf->tile_axis->empty()) {
      TileAxis *bc_last = buf->tile_axis->back();
      int64_t const_extent = bc_last->GetConstExtent();
      if (const_extent != -1) {
        int64_t block_size = GetMaxAlignBytes(bc_last->data_size);
        int64_t l1_size = this->GetConstTileVal(bc_last).first;
        if (l1_size == TileVarId::UNDEFINE) {
          l1_size = const_extent;
        }
        if (l1_size < block_size) {
          CHECK_GT(l1_size, 0);
          buf_size_info->act_buf_size *= (block_size - 1 + l1_size) / l1_size;
        }
      }
    }
  } else {
    int64_t align = GetAlignBytes(buf->size);
    if (buf_size_info->f_mul < align || (align != 0 && buf_size_info->f_mul % align != 0)) {
      CHECK_GT(buf_size_info->act_buf_size, 0);
      int64_t align_m = buf_size_info->f_mul;
      while (align_m % align != 0) {
        align_m += 1;
      }
      double exp = static_cast<double>(align_m) / static_cast<double>(buf_size_info->f_mul);
      buf_size_info->act_buf_size = static_cast<int64_t>(static_cast<double>(buf_size_info->act_buf_size) * exp);
    }
  }
}

void TileCandidate::DoMemInfer() {
  if (buffer_usage_ == nullptr) {
    buffer_usage_ = std::make_unique<BufferUsage>(BufferUsage());
    buffer_usage_->Build(analyzer_->buffer_usage_timetable_);
  }
  std::unique_ptr<MemInferInfo> mem_infer_info(new (std::nothrow) MemInferInfo());
  CHECK(mem_infer_info) << "memory alloc fail";

  for (auto cur_time = 0; cur_time <= buffer_usage_->max_time; ++cur_time) {
    auto releases = buffer_usage_->buf_release_time.find(cur_time);
    if (releases != buffer_usage_->buf_release_time.end()) {
      for (auto buf : releases->second) {
        mem_infer_info->live_size[buf->scope] -= mem_infer_info->live_buf[buf];
        mem_infer_info->live_buf.erase(buf);
      }
    }
    auto allocates = buffer_usage_->buf_alloc_time.find(cur_time);
    if (allocates != buffer_usage_->buf_alloc_time.end()) {
      for (auto buf : allocates->second) {
        if (mem_infer_info->live_buf.count(buf) != 0) {
          continue;
        }
        UpdateMemoryAfterBuffer(buf, mem_infer_info.get());
      }
    }
  }

  for (int i = 0; i < MEM_SCOPE_BULK; ++i) {
    mem_infer_[i] = mem_infer_info->max_live_size[i];
    align_mem_infer_[i] = mem_infer_info->max_act_live_size[i];
  }
}

/*
 * This function returns current data size moved from local buffer
 * to main memory within target axis.
 *  e.g.1: target is not inner-most axis
 * Input ir:
 *  for (cc0) <--- axis, dtype = float16
 *    for (cc1)  <--- tile factor 1024, dtype = float16
 *      GM_BUF1[cc0, cc1] = UB_BUF1[cc0, cc1]
 *  for (cc0) <--- axis
 *    for (cc2)  <--- tile factor 1024, dtype = float32
 *      GM_BUF2[cc0, cc2] = UB_BUF2[cc0, cc2]
 * Return:
 *  min(1024 * 2(fp16), 1024 * 4(fp32)) = 1024 * 2
 *
 * e.g.2: target is inner-most axis
 * Input ir:
 *  for (cc0) <--- axis, dtype = float16
 *    GM_BUF1[cc0] = UB_BUF1[cc0]
 * Return:
 *  32(ALIGN_BYTES) / 2(fp16) = 16
 */
int TileCandidate::GetDmaCopySizeWithinAxis(TileAxis *target_axis) {
  std::stringstream ss;
  int min_data_each_core = -1;
  bool before_this_axis = true;
  for (const auto &attr : analyzer_->RootAxis()->attrs) {
    if (attr.attr_key.find(AT_DMA3) == std::string::npos) {
      continue;
    }
    int64_t data_each_core = 1;
    int data_bytes = -1;
    bool need_record = true;
    std::string gm_buf_name = attr.attr_value;
    auto it = analyzer_->buf_info_.find(gm_buf_name);
    if (it == analyzer_->buf_info_.end()) {
      continue;
    }
    auto gm_buf = it->second.get();
    for (auto &gm_axis : *(gm_buf->tile_axis)) {
      if (gm_axis->index != target_axis->index || gm_axis->range_extent.as<IntImm>() == nullptr) {
        need_record = false;
        break;
      }
      if (gm_axis == target_axis) {
        before_this_axis = false;
        continue;
      }
      if (before_this_axis) {
        continue;
      }
      int64_t c1_val = MIN_TILE;
      std::tie(c1_val, std::ignore) = GetConstTileVal(gm_axis);
      if (c1_val == TileVarId::VAR) {
        need_record = false;
        break;
      }
      CHECK_NE(c1_val, 0) << "Inner axis " << gm_axis->dim_axis << " should be tile before axis "
                          << target_axis->dim_axis;
      if (gm_axis->HasAnyAttr({AT_REDUCE_AXIS, AT_TRANSPOSE, AT_TRANSFORM})) {
        ss << "axis " << gm_axis->index << "_" << gm_axis->dim_axis << " cannot be flatten. clear data each core.";
        analyzer_->GetTileLogger().AppendLog(DO_TILING, ss);
        data_each_core = 1;
        data_bytes = 1;
        continue;
      }
      ss << "axis " << gm_axis->index << "_" << gm_axis->dim_axis << " contains " << c1_val;
      data_each_core *= c1_val;
      auto min_bytes = static_cast<int>(ALIGN_BYTES / GetMaxAlignBytes(gm_axis->data_size));
      data_bytes = (data_bytes == -1 || min_bytes < data_bytes) ? min_bytes : data_bytes;
    }
    if (need_record && (min_data_each_core == -1 || data_bytes * data_each_core < min_data_each_core)) {
      min_data_each_core = data_bytes * data_each_core;
    }
  }
  ss << "[Data within axis " << target_axis->index << "_" << target_axis->dim_axis << "]: " << min_data_each_core;
  analyzer_->GetTileLogger().AppendLog(DO_TILING, ss);
  return min_data_each_core == -1 ? static_cast<int>(ALIGN_BYTES / GetMaxAlignBytes(target_axis->data_size))
                                  : min_data_each_core;
}

/*
 * This function returns the minimal tile size of axis that can enable multi-core function.
 * If inner-most data granularity of DMA from local buffer to main memory is less than align bytes,
 * it will disable multi-core function.
 */
int TileCandidate::GetMinFactorToEnableMulticore(TileAxis *axis) {
  return std::max(static_cast<int>(ALIGN_BYTES / GetDmaCopySizeWithinAxis(axis)), 1);
}

/*
 * This function returns the minimal tile size of axis that each core can have enough data granularity to process.
 * Minimal data granularity for each core is set to 256 bytes by default and if actual data granularity is less
 * than this value, the candidate tile sizes will be regarded as multi-core inefficient.
 */
int TileCandidate::GetMinFactorForMinDataGranularity(TileAxis *axis) {
  auto granularity = 1;
  for (auto a : this->tile_axis_) {
    if (a == axis) {
      continue;
    }
    if (!a->range_extent.as<IntImm>()) {
      continue;
    }
    int64_t c1_val = this->GetConstTileVal(a).first;
    if (c1_val == TileVarId::UNDEFINE || c1_val == TileVarId::VAR) {
      continue;
    }
    granularity *= c1_val;
  }
  return std::max(static_cast<int>(MIN_CORE_GRANULARITY / granularity), 1);
}

/*
 * This function returns the multiplies of loop extent of all the pending (not tiled) axes.
 */
int TileCandidate::GetMaximalPendingBlocks(TileAxis *excluded_axis) {
  int64_t blocks = 1;
  for (auto axis : this->tile_axis_) {
    if (axis == excluded_axis) {
      continue;
    }
    if (!axis->range_extent.as<IntImm>()) {
      continue;
    }
    int64_t c1_val = this->GetConstTileVal(axis).first;
    if (c1_val == TileVarId::UNDEFINE || c1_val == TileVarId::VAR) {
      blocks *= axis->range_extent.as<IntImm>()->value;
    }
  }
  return blocks;
}

class LinearAccessPatternBuilder : public IRVisitor {
 public:
  using StmtEntry = TilingAnalyzer::StmtEntry;
  using BufferEntry = TilingAnalyzer::BufferEntry;

  explicit LinearAccessPatternBuilder(TilingAnalyzer *a) : analyzer_(a) {}
  ~LinearAccessPatternBuilder() override = default;

  void Build(const Stmt &stmt) {
    CHECK(analyzer_ != nullptr);
    if (!analyzer_->scop_info_.user_config_.GetIsDynamic()) {
      CollectGlobalWritesBuf();
    }
    CollectAlignedBuf();
    CollectReduceBuf();
    CollectExpandedBuf();
    cur_axis_ = analyzer_->RootAxis();
    seq_.emplace_back(StmtEntry{cur_axis_, 0, nullptr});
    IRVisitor::Visit(stmt);
    auto end = static_cast<int>(seq_.size());
    CHECK(!seq_.empty());
    seq_[0].scope_pair_offset = end;
    seq_.emplace_back(StmtEntry{cur_axis_, -end, nullptr});
    if (analyzer_->scop_info_.user_config_.GetTarget() == TARGET_CCE) {
      CollectCastedBuf();
      UpdateBufferAlignSize();
    }
    BuildBufferUsageTimetable();
  }

  void BuildBufferUsageTimetable() {
    // Build usage timetable for each stmt, which will be used during tiling
    // Stmt in linear_seq matches pattern `buffer_def : {buffer_ref1, buffer_ref2...}`
    // E.g.
    // Stmt1 `input_1_local_UB: input_1`
    // Stmt2 `input_2_local_UB: input_2`
    // Stmt3 `output_0_local_UB: input_1_local_UB, input_2_local_UB`
    // Stmt4 `output_0: output_0_local_UB`
    // will build following timetable
    // ------------------------------------------------------
    // |        Buffer         | Alloc-time | Last-used-time |
    // | input_1_local_UB  |      1     |        3       |
    // | input_2_local_UB  |      2     |        3       |
    // | output_0_local_UB |      3     |        4       |
    // ------------------------------------------------------
    // During tiling, timestamp will be scanned from 0 and size of buffer whose `Alloc-time`
    // equal than current timestamp will be added to live size; and buffer whose `Last-used-time`
    // smaller than current timestamp will be removed from live size.

    int timestamp = 0;
    for (auto idx = 0; idx <= static_cast<int>(seq_.size()) - 1; ++idx) {
      auto &e = seq_[idx];
      if (e.def == nullptr) {
        continue;
      }
      // record earliest def time
      if (buffer_usage_timetable_.find(e.def) == buffer_usage_timetable_.end()) {
        buffer_usage_timetable_[e.def].first = timestamp;
      } else if (timestamp < buffer_usage_timetable_[e.def].first ||
                 timestamp < buffer_usage_timetable_[e.def].second) {
        buffer_usage_timetable_[e.def].first = timestamp;
      }
      // record latest ref time
      for (auto ref : e.ref) {
        if (buffer_usage_timetable_.find(ref) == buffer_usage_timetable_.end()) {
          buffer_usage_timetable_[ref].second = timestamp;
        } else if (timestamp > buffer_usage_timetable_[ref].first || timestamp > buffer_usage_timetable_[ref].second) {
          buffer_usage_timetable_[ref].second = timestamp;
        }
        // If it is gm->local_buf, local_buf's allocate might be lifted in invariant hoist, so make its ref time to
        // maximal.
        if (local_buf_.count(ref->name) == 0) buffer_usage_timetable_[e.def].second = seq_.size();
      }
      timestamp += 1;
    }
  }

  void Visit_(const Realize *op) final {
    auto buf_name = op->func->func_name();
    local_buf_.insert(buf_name);
    IRVisitor::Visit_(op);
  }

  void Visit_(const For *op) override {
    TileAxis *last_axis = cur_axis_;
    TileAxis *axis = analyzer_->Axis(op);
    if (axis != nullptr) cur_axis_ = axis;
    auto entry_idx = static_cast<int>(seq_.size());
    cur_axis_->seq_index = entry_idx;
    seq_.emplace_back(StmtEntry{cur_axis_, 0, nullptr});
    var_axis_[op->loop_var.get()] = cur_axis_;
    IRVisitor::Visit_(op);
    var_axis_.erase(op->loop_var.get());
    auto exit_idx = static_cast<int>(seq_.size());
    seq_.emplace_back(StmtEntry{cur_axis_, entry_idx - exit_idx, nullptr});
    CHECK_LT((uint)entry_idx, seq_.size());
    seq_[(uint)entry_idx].scope_pair_offset = exit_idx - entry_idx;
    cur_axis_ = last_axis;
  }

  void Visit_(const Provide *op) override {
    in_stmt_ = true;
    IRVisitor::Visit_(op);
    in_stmt_ = false;
    std::string buf = op->func->func_name();
    UpdateTileAxis(buf, op->args);
    StmtAppend(buf);
  }

  void Visit_(const Store *op) override {
    in_stmt_ = true;
    IRVisitor::Visit_(op);
    in_stmt_ = false;
    std::string buf = op->buffer_var->name_hint;
    UpdateTileAxis(buf, {op->index});
    StmtAppend(buf);
  }

  void Visit_(const Load *op) override {
    CHECK(in_stmt_);
    std::string buf = op->buffer_var->name_hint;
    UpdateTileAxis(buf, {op->index});
    cur_ref_.emplace(buf);
    IRVisitor::Visit_(op);
  }

  void Visit_(const Call *op) override {
    IRVisitor::Visit_(op);
    if (op->call_type == Call::Halide) {
      UpdateTileAxis(op->name, op->args);
      cur_ref_.emplace(op->name);
      if (!in_stmt_) {
        StmtAppend(op->name);
      }
    }
  }

  std::vector<StmtEntry> seq_;
  std::unordered_map<std::string, std::shared_ptr<BufferEntry>> buf_;
  std::unordered_map<TilingAnalyzer::BufferEntry *, std::pair<int, int>> buffer_usage_timetable_;

 private:
  void StmtAppend(const std::string &def) {
    std::vector<BufferEntry *> ref_buf;
    auto tensor_mem_flows = analyzer_->scop_info_.analysis_result_.GetTensorMemFlows();

    // Deal with referenced buffers
    for (const std::string &ref : cur_ref_) {
      MemFlow &mem_flow = tensor_mem_flows[ref];
      CHECK(!mem_flow.empty());
      if (global_writes_buf_.find(GetBuffer(ref, mem_flow[0])->name) != global_writes_buf_.end()) {
        // In tensor mem flow, the output tensor will have a reverse mem flow:
        // e.g. DDR -> UB for vector or DDR -> UB -> L0C for cube;
        // Therefore, we need to reverse mem flow to get the correct memory usage.
        std::reverse(mem_flow.begin(), mem_flow.end());
      }
      if (local_buf_.count(ref) && mem_flow.size() == 2U) {
        // If it is a buffer from gm to certain location, directly use destination as buffer's scope
        // in conv_backprop, multiple bands are merged and local buffer can have gm->buf->c1->c0 memflow.
        ref_buf.push_back(GetBuffer(ref, mem_flow.back()));
      } else {
        BufferEntry *from = GetBuffer(ref, mem_flow[0]);
        for (size_t i = 1; i < mem_flow.size(); ++i) {
          if (mem_flow[i] == C1_ && mem_flow[i] == mem_flow[i - 1]) {  // fractal_C1
            continue;
          }
          BufferEntry *to = GetBuffer(ref, mem_flow[i]);
          StmtAppend(to, {from});
          from = to;
        }
        ref_buf.push_back(from);
      }
    }

    // Deal with defined buffers
    MemFlow &def_flow = tensor_mem_flows[def];
    CHECK(!def_flow.empty());
    BufferEntry *def_buf = GetBuffer(def, def_flow.back());

    StmtAppend(def_buf, ref_buf);
    if (!local_buf_.count(def)) {
      BufferEntry *from = def_buf;
      for (int64_t i = def_flow.size() - 2; i >= 0; --i) {
        BufferEntry *to = GetBuffer(def, def_flow[i]);

        StmtAppend(to, {from});
        from = to;
      }
    }
    cur_ref_.clear();
  }

  std::string GetBufferName(const std::string &name, MemType type) {
    auto tensor_mem_flows = analyzer_->scop_info_.analysis_result_.GetTensorMemFlows();
    MemFlow &mem_flow = tensor_mem_flows[name];
    auto tensor_name_flows = analyzer_->scop_info_.analysis_result_.GetTensorNameFlows();
    std::vector<std::string> &name_flow = tensor_name_flows[name];
    for (size_t i = 0; i < mem_flow.size(); ++i) {
      if (mem_flow[i] == type) {
        return name_flow[i];
      }
    }
    CHECK(false) << "no buffer was found: " << name << ", " << type;
    return std::string();
  }

  BufferEntry *GetBuffer(const std::string &name, MemType type) {
    // If the global cache does not exist, create a new one, otherwise return to exist.
    std::string buf_name = GetBufferName(name, type);
    auto it = buf_.find(buf_name);
    if (it != buf_.end()) {
      return it->second.get();
    }
    bool is_reduce = false;
    for (auto n : reduce_dst_buf_) {
      if (buf_name.find(n) != std::string::npos) {
        is_reduce = true;
      }
    }
    if (!local_buf_.count(name) && type != DDR && !is_reduce) {  // global cache use idx
      int idx = 0;
      if (buf_idx_.count(buf_name) == 0) {
        buf_idx_[buf_name] = 1;
      } else {
        idx = buf_idx_[buf_name];
        buf_idx_[buf_name] = idx + 1;
      }
      buf_name = buf_name + "_" + std::to_string(idx);
    }
    auto buf = std::make_shared<BufferEntry>();
    buf->name = buf_name;
    buf->scope = mem_type_to_scope_[type];
    GetBufferSize(name, buf);
    buf->alloc_seq = -1;
    buf->tile_axis = buf_tile_axis_[name];
    buf_.emplace(buf_name, buf);
    return buf.get();
  }

  void StmtAppend(BufferEntry *def, const std::vector<BufferEntry *> &refs) {
    CHECK(def);
    seq_.emplace_back(StmtEntry{cur_axis_, 0, def});
    if (def->alloc_seq == -1) {
      def->alloc_seq = seq_.size() - 1;
      seq_.back().alloc.insert(def);
    }
    LivenessExtent(def);
    for (BufferEntry *ref : refs) {
      CHECK(ref);
      if (ref->alloc_seq == -1) {
        ref->alloc_seq = seq_.size() - 1;
      }
      seq_.back().ref.insert(ref);
      LivenessExtent(ref);
    }
  }

  void CollectGlobalWritesBuf() {
    analyzer_->scop_info_.analysis_result_.GetWrites().foreach_map([this](const isl::map &access) -> void {
      const isl::id &tensor_id = access.get_tuple_id(isl_dim_out);
      global_writes_buf_.insert(tensor_id.name());
    });
  }

  void CollectAlignedBuf() {
    for (const auto &attr : analyzer_->RootAxis()->attrs) {
      if (attr.attr_key != AT_TRANSFORM) continue;
      aligned_buf_.insert(attr.attr_value);
    }
  }

  void CollectReduceBuf() {
    for (const auto &attr : analyzer_->RootAxis()->attrs) {
      if (attr.attr_key != AT_REDUCE_FLOW) continue;
      std::vector<std::string> flow = akg::common::Split(attr.attr_value, "->");
      CHECK_EQ(flow.size(), 2U);
      reduce_src_buf_.insert(flow[0]);
      reduce_dst_buf_.insert(flow[1]);
    }
  }

  void CollectExpandedBuf() {
    for (const auto &attr : analyzer_->RootAxis()->attrs) {
      if (attr.attr_key != "EXPANSION") continue;
      std::vector<std::string> info = akg::common::Split(attr.attr_value, "->");
      CHECK_EQ(info.size(), 2U);
      std::string buffer = info[0];
      auto times = StrToDecimalInt(info[1]);
      expanded_buf_[buffer] = times;
    }
  }

  void CollectCastedBuf() {
    auto GetMinAlignSize = [this](const std::string &buf_name, int64_t ori_size) -> int64_t {
      auto it = this->buf_.find(buf_name);
      if (it != this->buf_.end()) {
        BufferEntry *buf = it->second.get();
        CHECK(buf);
        return std::min(buf->size, ori_size);
      }
      return ori_size;
    };
    auto CollectBuf = [GetMinAlignSize, this](TileAxis *axis) {
      for (const auto &attr : axis->attrs) {
        if (attr.attr_key != AT_CAST) continue;
        std::vector<std::string> buffer_names;

        std::vector<std::string> src_dst = akg::common::Split(attr.attr_value, "->");
        CHECK_EQ(src_dst.size(), 2U);

        std::vector<std::string> src_list = akg::common::Split(src_dst[0], ",");
        CHECK_GE(src_list.size(), 1U);
        for (const auto &src : src_list) {
          std::vector<std::string> src_info = akg::common::Split(src, ":");
          CHECK_EQ(src_info.size(), 2U);
          std::string src_buffer = src_info[0];
          buffer_names.emplace_back(src_buffer);
          buffer_names.emplace_back(src_buffer + LOCAL_BUF);
        }

        std::vector<std::string> dst_info = akg::common::Split(src_dst[1], ":");
        CHECK_EQ(dst_info.size(), 2U);
        CHECK_NE(dst_info[1], "");
        std::string dst_buffer = dst_info[0];
        auto cast_to_size = static_cast<int64_t>(std::strtol(dst_info[1].c_str(), nullptr, 10));
        buffer_names.emplace_back(dst_buffer);
        buffer_names.emplace_back(dst_buffer + LOCAL_BUF);

        for (const auto &bn : buffer_names) {
          cast_to_size = GetMinAlignSize(bn, cast_to_size);
        }
        for (const auto &bn : buffer_names) {
          this->casted_buf_[bn] = cast_to_size;
        }
      }
    };
    this->analyzer_->ForEachAxisTopDown(CollectBuf);
  }

  void UpdateBufferAlignSize() {
    for (const auto &it : buf_) {
      BufferEntry *buf = it.second.get();
      if (casted_buf_.find(buf->name) != casted_buf_.end()) {
        int64_t min_size = std::min(buf->size, casted_buf_[buf->name]);
        buf->align_size = min_size;
      } else {
        it.second->align_size = it.second->size;
      }
    }
  }

  void LivenessExtent(BufferEntry *buf) {
    CHECK(buf);
    if (buf->scope == MEM_SCOPE_GM) return;
    TileAxis *use_parent = seq_.back().parent;
    TileAxis *alloc_parent = nullptr;

    bool is_reduce = false;
    for (auto name : reduce_src_buf_) {
      if (buf->name.find(name) != std::string::npos) is_reduce = true;
    }
    // Use the outermost axis as alloc parent.
    for (auto &it : *(buf->tile_axis)) {
      TileAxis *axis = it;
      CHECK(axis);
      if (alloc_parent == nullptr) {
        alloc_parent = axis;
      } else if (axis->dim_axis < alloc_parent->dim_axis) {
        alloc_parent = axis;
      }
    }
    // Lift reduced buffer to the very beginning def of buf.
    if (is_reduce || alloc_parent == nullptr) {
      alloc_parent = seq_[buf->alloc_seq].parent;
    }

    CHECK(alloc_parent);
    seq_[buf->alloc_seq].alloc.erase(buf);
    seq_[alloc_parent->seq_index].alloc.insert(buf);
    buf->alloc_seq = alloc_parent->seq_index;

    if (use_parent == alloc_parent) {
      seq_[use_parent->seq_index].ref.insert(buf);
      return;
    }
    CHECK(use_parent);
    TileAxis *alloc = alloc_parent;
    TileAxis *use = use_parent;
    if (alloc_parent->dim_axis < use_parent->dim_axis) {
      while (use_parent != nullptr && alloc_parent->dim_axis < use_parent->dim_axis) {
        use = use_parent;
        use_parent = use_parent->parent;
      }
      if (use_parent == alloc_parent) {
        seq_[use->seq_index].ref.insert(buf);
        return;
      }
    } else {
      while (alloc_parent != nullptr && use_parent->dim_axis < alloc_parent->dim_axis) {
        alloc = alloc_parent;
        alloc_parent = alloc_parent->parent;
      }
    }
    while (alloc_parent != use_parent && use_parent != nullptr && alloc_parent != nullptr) {
      alloc = alloc_parent;
      alloc_parent = alloc_parent->parent;
      use = use_parent;
      use_parent = use_parent->parent;
    }
    CHECK_NE(use, alloc);
    CHECK(alloc);
    CHECK(use);
    seq_[buf->alloc_seq].alloc.erase(buf);
    seq_[alloc->seq_index].alloc.insert(buf);
    buf->alloc_seq = alloc->seq_index;
    seq_[use->seq_index].ref.insert(buf);
  }

  void UpdateTileAxis(const std::string &buf, const Array<Expr> &args) {
    if (buf_tile_axis_.count(buf) && local_buf_.count(buf)) return;
    auto tile_axis = std::make_shared<std::vector<TileAxis *>>();
    auto CollectAxis = [&tile_axis, this](const NodeRef &op) {
      const auto var = op.as<Variable>();
      if (var == nullptr) return;
      auto it = this->var_axis_.find(var);
      if (it != this->var_axis_.end()) {
        tile_axis->push_back(it->second);
      }
    };
    for (Expr e : args) {
      air::ir::PostOrderVisit(e, CollectAxis);
    }
    buf_tile_axis_[buf] = tile_axis;
  }

  void GetBufferSize(const std::string &name, const std::shared_ptr<BufferEntry> &buf) {
    int64_t dsize = 1;
    int64_t expand_size = 1;
    Expr shape = CastIntToExpr(1);
    CHECK(buf);
    auto binds = analyzer_->scop_info_.user_config_.GetBind();
    for (auto i : binds) {
      if (i.first->op->name != name) continue;
      dsize = static_cast<int64_t>(i.first->dtype.bytes());
      CHECK_GT(dsize, 0) << name << "'s data type error, bytes = 0";
      for (Expr dim : i.first->shape) shape *= dim;
      CHECK(shape.defined()) << "Buffer " << name << "'s shape not defined.";
      shape = CanonicalSimplify(shape);
      if (analyzer_->scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
        break;
      }
      if (!analyzer_->scop_info_.user_config_.GetIsDynamic() && reduce_src_buf_.find(name) != reduce_src_buf_.end()) {
        expand_size *= BISEC_REDUCE_MEM_EXPANSION;
      }
      if (expanded_buf_.find(name) != expanded_buf_.end()) {
        expand_size *= expanded_buf_[name];
      }
      if (aligned_buf_.find(name) != aligned_buf_.end()) {
        expand_size *= GetAlignBytes(dsize);
      }
      break;
    }
    buf->size = dsize;
    buf->shape = shape;
    buf->expand_size = expand_size;
  }

  TilingAnalyzer *analyzer_;
  TileAxis *cur_axis_{nullptr};
  bool in_stmt_{false};
  std::unordered_set<std::string> local_buf_;
  std::unordered_set<std::string> cur_ref_;
  std::unordered_map<std::string, int> buf_idx_;
  std::unordered_map<const Variable *, TileAxis *> var_axis_;
  std::unordered_map<std::string, std::shared_ptr<std::vector<TileAxis *>>> buf_tile_axis_;
  std::unordered_set<std::string> aligned_buf_;
  std::unordered_set<std::string> reduce_src_buf_;
  std::unordered_set<std::string> reduce_dst_buf_;
  std::unordered_set<std::string> global_writes_buf_;
  std::unordered_map<std::string, int> expanded_buf_;
  std::unordered_map<std::string, int64_t> casted_buf_;

  std::unordered_map<int, TilingMemScope> mem_type_to_scope_ = {
    {DDR, MEM_SCOPE_GM},         {C1_, MEM_SCOPE_CACHE1},    {BUF_, MEM_SCOPE_BUFFER},    {C0A_, MEM_SCOPE_CACHE0_A},
    {C0B_, MEM_SCOPE_CACHE0_B},  {C0C_, MEM_SCOPE_CACHE0_C}, {BUF_C0_, MEM_SCOPE_BUFFER}, {BUF_C1_, MEM_SCOPE_BUFFER},
    {SHARED_, MEM_SCOPE_SHARED}, {LOCAL_, MEM_SCOPE_LOCAL}};
};

std::vector<TileAxis *> TilingAnalyzer::GetAxesContainsAttr(const std::string &attr_key) const {
  std::vector<TileAxis *> axes;
  auto AddAxisWithAttr = [&attr_key, &axes](TileAxis *a) {
    for (const auto &attr : a->attrs) {
      if (attr.attr_key.find(attr_key) != std::string::npos) {
        axes.emplace_back(a);
        break;
      }
    }
  };
  this->ForEachAxisTopDown(AddAxisWithAttr);
  return axes;
}

std::vector<TileAxis *> TilingAnalyzer::GetAxesOfAttr(const std::string &attr_key, int band_index) const {
  std::vector<TileAxis *> axes;
  auto AddAxisWithAttr = [&attr_key, &axes, &band_index](TileAxis *a) {
    for (const auto &attr : a->attrs) {
      if (attr.attr_key == attr_key && (band_index == -1 || a->index == band_index)) {
        axes.emplace_back(a);
        break;
      }
    }
  };
  this->ForEachAxisTopDown(AddAxisWithAttr);
  return axes;
}

std::vector<TileAxis *> TilingAnalyzer::GetAxesOfAttr(const AttrInfo &attr_info, int band_index) const {
  std::vector<TileAxis *> axes;
  auto AddAxisWithAttr = [&attr_info, &axes, &band_index](TileAxis *a) {
    for (const auto &attr : a->attrs) {
      if (attr.attr_key == attr_info.attr_key && attr.attr_value == attr_info.attr_value &&
          (band_index == -1 || a->index == band_index)) {
        axes.emplace_back(a);
        break;
      }
    }
  };
  this->ForEachAxisTopDown(AddAxisWithAttr);
  return axes;
}

bool TileAxis::HasAttr(const std::string &attr_key, const bool partial_match) const {
  for (const auto &a : this->attrs) {
    if (partial_match) {
      if (a.attr_key.find(attr_key) != std::string::npos) {
        return true;
      }
    } else {
      if (a.attr_key == attr_key) {
        return true;
      }
    }
  }
  return false;
}

bool TileAxis::HasAttr(const AttrInfo &attr) const {
  for (const auto &a : this->attrs) {
    if (a.attr_key == attr.attr_key && a.attr_value == attr.attr_value) {
      return true;
    }
  }
  return false;
}

bool TileAxis::HasAnyAttr(const std::unordered_set<std::string> &attr_keys, const bool partial_match) const {
  for (const auto &key : attr_keys) {
    if (this->HasAttr(key, partial_match)) {
      return true;
    }
  }
  return false;
}

void TileAxis::RemoveAttr(const std::string &attr_key) {
  for (auto &a : this->attrs) {
    if (a.attr_key == attr_key) {
      a.attr_key = "";
    }
  }
}

void TileAxis::RemoveAttr(const AttrInfo &attr) {
  for (auto &a : this->attrs) {
    if (a.attr_key == attr.attr_key && a.attr_value == attr.attr_value) {
      a.attr_key = "";
    }
  }
}

int TilingAnalyzer::GetNumOfAxisInBand(int band_idx) const {
  int max = 0;
  auto UpdateMax = [&band_idx, &max](TileAxis *axis) {
    if (axis->index != band_idx) {
      return;
    }
    int dim = axis->dim_axis;
    if (dim > max) {
      max = dim;
    }
  };
  this->ForEachAxisTopDown(UpdateMax);
  return max + 1;
}

void TilingAnalyzer::AddPostTilingConstraints() {
  auto strategy_manager = std::unique_ptr<TilingStrategyManager>(new (std::nothrow) TilingStrategyManager());
  CHECK(strategy_manager) << "memory alloc fail.";
  std::vector<TilingStrategy *> actived_strategies;

  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    CountStrategy count_strategy(this);
    ReduceStrategy reduce_strategy(this);
    ModStrategy mod_strategy(this);
    ShiftAxisStrategy shift_strategy(this);
    GemmStrategy gemm_strategy(this);
    ConvStrategy conv_strategy(this);
    GpuDmaAnalysisStrategy dma_analysis_strategy(this);
    CustomTilingStrategy custom_strategy(this);
    CsrStrategy csr_strategy(this);
    VectorizedStrategy vectorized_strategy(this);
    TensorOfTensorStrategy tot_strategy(this);
    GpuStrategy gpu_strategy(this);
    if (scop_info_.analysis_result_.GetIsGpuDmaAnalysed()) {
      actived_strategies.push_back(&dma_analysis_strategy);
    } else {
      if (scop_info_.analysis_result_.GetOpTemplate() == Template::COUNT_OP) {
        actived_strategies.push_back(&count_strategy);
      }
      if (scop_info_.user_config_.GetIsTuning()) {
        actived_strategies.push_back(&custom_strategy);
      }
      actived_strategies.push_back(&reduce_strategy);
      actived_strategies.push_back(&mod_strategy);
      actived_strategies.push_back(&shift_strategy);
      actived_strategies.push_back(&gemm_strategy);
      actived_strategies.push_back(&conv_strategy);
      actived_strategies.push_back(&vectorized_strategy);
      actived_strategies.push_back(&tot_strategy);
      if (scop_info_.analysis_result_.GetCsr()) {
        actived_strategies.push_back(&csr_strategy);
      }
      actived_strategies.push_back(&gpu_strategy);
    }
    strategy_manager->SetStrategies(actived_strategies);
    strategy_manager->ExecuteGpu();
    if (scop_info_.user_config_.GetIsTuning()) {
      binding_spaces_.clear();
      for (auto i : gpu_strategy.thread_binding_spaces_) {
        UpdateBindingSpace(i);
      }
      for (auto i : gpu_strategy.block_binding_spaces_) {
        UpdateBindingSpace(i);
      }
    }
    return;
  } else if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    CpuStrategy cpu_strategy(this);
    actived_strategies.push_back(&cpu_strategy);
    strategy_manager->SetStrategies(actived_strategies);
    strategy_manager->ExecuteCpu();
    return;
  }
}

void TilingAnalyzer::AddTilingConstraints() {
  auto strategy_manager = std::unique_ptr<TilingStrategyManager>(new (std::nothrow) TilingStrategyManager());
  CHECK(strategy_manager) << "memory alloc fail.";
  std::vector<TilingStrategy *> actived_strategies;

  if (scop_info_.user_config_.GetTarget() == TARGET_CUDA) {
    CastStrategy cast_strategy(this);
    actived_strategies.push_back(&cast_strategy);

    strategy_manager->SetStrategies(actived_strategies);
    strategy_manager->ExecuteGpu();
    return;
  }

  if (scop_info_.user_config_.GetTarget() == TARGET_CPU) {
    CastStrategy cast_strategy(this);
    actived_strategies.push_back(&cast_strategy);
    strategy_manager->SetStrategies(actived_strategies);
    strategy_manager->ExecuteCpu();
    return;
  }

  // CCE strategies
  PassDownAttrStrategy pd_attr_strategy(this);
  actived_strategies.push_back(&pd_attr_strategy);

  CastStrategy cast_strategy(this);
  VectorizedStrategy vectorized_strategy(this);
  TensorOfTensorStrategy tot_strategy(this);
  actived_strategies.push_back(&cast_strategy);
  if (!scop_info_.user_config_.GetIsTuning()) {
    actived_strategies.push_back(&vectorized_strategy);
  }
  actived_strategies.push_back(&tot_strategy);

  ReduceStrategy reduce_strategy(this);
  DmaAlignStrategy dma_align_stratgey(this);

  if (!scop_info_.user_config_.GetIsTuning()) {
    actived_strategies.push_back(&reduce_strategy);
    actived_strategies.push_back(&dma_align_stratgey);
  }

  ModStrategy mod_strategy(this);
  actived_strategies.push_back(&mod_strategy);

  ConvStrategy conv_strategy(this);
  actived_strategies.push_back(&conv_strategy);

  GemmStrategy gemm_strategy(this);
  actived_strategies.push_back(&gemm_strategy);

  ConflictTreeRangeStrategy conflict_strategy(this);
  actived_strategies.push_back(&conflict_strategy);

  CustomTilingStrategy custom_strategy(this);
  actived_strategies.push_back(&custom_strategy);
  DynamicShapeLimitStrategy dyn_limit_strategy(this);
  actived_strategies.push_back(&dyn_limit_strategy);

  ShiftAxisStrategy shift_strategy(this);
  ModShiftAxisStrategy mod_shift_strategy(this);
  actived_strategies.push_back(&shift_strategy);
  actived_strategies.push_back(&mod_shift_strategy);

  DynamicBoundStrategy dyn_bound_strategy(this);
  actived_strategies.push_back(&dyn_bound_strategy);

  strategy_manager->SetStrategies(actived_strategies);
  strategy_manager->ExecuteNpu();
}

bool TilingAnalyzer::Prepare() {
  logger_ = std::unique_ptr<TileLogger>(new (std::nothrow) TileLogger(
    scop_info_.AddDumpDir("tiling.log"), !scop_info_.user_config_.GetDumpPolyDir().empty()));
  CHECK(logger_) << "memory alloc fail.";

  // Stage 1: Analyze schedule tree.
  ScheduleTreeAnalyzer sch_ana(this, this->sch_);
  root_axis_ = sch_ana.Build(this->Halide());
  if (root_axis_ == nullptr) {
    return false;
  }
  if (root_axis_->children.empty()) {
    return false;
  }
  auto BuildAxisMap = [this](const TileAxis *a) {
    for (auto loop : a->loops) {
      CHECK(loop) << "Tile axis has null ptr loop, check";
      this->tile_axis_[loop] = const_cast<TileAxis *>(a);
    }
  };
  this->ForEachAxisTopDown(BuildAxisMap);
  if (op_type_ != TileOpType::VECTOR_OP) {
    sch_ana.AnalyzeCubeInfo();
  }

  // Stage 2: Analyze Halide IR and add tiling constraints.
  SpaceAnalyzer space_analyzer(this);

  space_analyzer.AnalyzeSpecialAxes();

  AddTilingConstraints();

  // Stage 3: Analyze buffer footprint.
  LinearAccessPatternBuilder lap_bdr(this);
  lap_bdr.Build(body_);
  linear_seq_ = std::move(lap_bdr.seq_);
  buf_info_ = std::move(lap_bdr.buf_);
  buffer_usage_timetable_ = std::move(lap_bdr.buffer_usage_timetable_);

  AddPostTilingConstraints();

  // Stage 4: Set tiling priority based on previous analysis.
  TilingPriorityScorer scroer(*this);
  scroer.SetPriorityByScoring();

  // Logging
  logger_->AppendLine(ANA_TILING_SPACE, "After adding constraints =======>");
  auto PrintAttr = [&](TileAxis *a) -> void {
    if (a != nullptr) a->DumpAxis();
  };
  ForEachAxisTopDown(PrintAttr);
  logger_->AppendLine(ANA_TILING_SPACE, "<=============");
  DumpLinearSeq();

  return true;
}

void TilingAnalyzer::ForEachAxisTopDown(const std::function<void(TileAxis *)> &fn, TileAxis *top) const {
  std::vector<TileAxis *> stack;
  if (top == nullptr) {
    top = root_axis_.get();
    if (top == nullptr) {
      return;
    }
  }
  stack.push_back(top);
  while (!stack.empty()) {
    TileAxis *a = stack.back();
    CHECK(a);
    stack.pop_back();
    fn(a);
    for (auto &i : a->children) {
      stack.push_back(i.get());
    }
  }
}

void TilingAnalyzer::DumpLinearSeq() {
  auto PrintBufList = [](const std::unordered_set<BufferEntry *> &bufs, std::stringstream &ss) {
    size_t num = bufs.size();
    for (auto it : bufs) {
      CHECK(it);
      ss << it->name << " (" << it->size << " * " << it->shape << " * " << it->expand_size << ")";
      if (--num) ss << ",";
    }
  };
  auto PrintIndent = [](const int n, std::stringstream &ss) {
    for (int i = 0; i < n; ++i) ss << "  ";
  };
  DumpBufferInfo();
  for (size_t seq_idx = 0; seq_idx < linear_seq_.size(); ++seq_idx) {
    auto &e = linear_seq_[seq_idx];
    int layer = e.parent->dim_axis;
    std::stringstream ss;
    PrintIndent(layer, ss);
    if (e.scope_pair_offset > 0) {
      TileAxis *axis = e.parent;
      CHECK(axis);
      ss << "[Offset] " << e.scope_pair_offset;
      ss << "[entry]";
      if (!e.alloc.empty()) {
        ss << "  [alloc] {";
        PrintBufList(e.alloc, ss);
        ss << "}";
      }
      if (!e.ref.empty()) {
        ss << "  [ref] {";
        PrintBufList(e.ref, ss);
        ss << "}";
      }
      CHECK(e.def == nullptr);
      for (auto loop : axis->loops) {
        CHECK(loop);
        ss << " loop=" << loop->loop_var << ":" << loop->extent;
      }
    } else if (e.scope_pair_offset < 0) {
      auto &entry = linear_seq_[seq_idx + e.scope_pair_offset];
      ss << "[exit]";
      if (!entry.ref.empty()) {
        ss << "  [ref]";
        PrintBufList(entry.ref, ss);
      }
    } else {
      ss << "  " << (e.def ? e.def->name : "null") << ": ";
      PrintBufList(e.ref, ss);
    }
    logger_->AppendLog(ANA_BUF_LIVE_EXTENT, ss);
  }
  DumpBufferUsageTimeable();
}

void TilingAnalyzer::DumpBufferInfo() {
  logger_->AppendLine(ANA_BUF_LIVE_EXTENT, "[buffer]");
  for (auto &it : buf_info_) {
    BufferEntry *buf = it.second.get();
    CHECK(buf);
    std::stringstream ss;
    Expr buf_size = Expr(buf->size * buf->expand_size) * buf->shape;
    ss << "  " << buf->name << ": size=" << buf_size << ", scope=" << buf->scope << ", tile={";

    size_t num = buf->tile_axis->size();
    for (auto &it2 : *(buf->tile_axis)) {
      TileAxis *tile_axis = it2;
      CHECK(tile_axis);
      for (auto loop : tile_axis->loops) {
        CHECK(loop);
        ss << loop->loop_var << "(" << tile_axis->index << ")";
        if (--num) ss << ",";
      }
    }
    ss << "}";
    logger_->AppendLog(ANA_BUF_LIVE_EXTENT, ss);
  }
}

void TilingAnalyzer::DumpBufferUsageTimeable() {
  logger_->AppendLine(ANA_BUF_LIVE_EXTENT, "========= Buffer Usage Timetable =========");
  std::stringstream ss;
  std::unordered_set<std::string> lived_buf_name;
  for (auto cur_time = 0; cur_time <= static_cast<int>(buffer_usage_timetable_.size() - 1); ++cur_time) {
    for (auto it : buffer_usage_timetable_) {
      auto alloc_time = it.second.first;
      auto last_use_time = it.second.second;
      if (last_use_time < cur_time && lived_buf_name.find(it.first->name) != lived_buf_name.end()) {
        lived_buf_name.erase(it.first->name);
      }
      if (alloc_time != cur_time) {
        continue;
      }
      lived_buf_name.insert(it.first->name);
      ss << "Buffer " << it.first->name << " | Alloc time: " << alloc_time << " | Last use time : " << last_use_time
         << " | ";
      logger_->AppendLog(ANA_BUF_LIVE_EXTENT, ss);
    }
  }
}

int64_t TilingAnalyzer::FindDivisibleTilingFactor(int64_t limit, int64_t range) {
  CHECK(range > 0 && limit > 0) << "Need positive range and limit.";
  if (range <= limit) {
    return range;
  }
  int64_t exp = (range - 1 + limit) / limit;
  int64_t init = exp > 2 ? exp : 2;
  int64_t end = static_cast<int>(sqrt(range));
  end = end <= init ? range : end;
  for (auto div = init; div < end; ++div) {
    if (range % div == 0) {
      return (range / div);
    }
  }
  return 1;
}

std::vector<int> TilingAnalyzer::GetSortedBands() const {
  std::vector<int> sorted_bands;
  std::unordered_map<Template, std::vector<int>> templates_map;
  for (int i = 0; i < static_cast<int>(root_axis_->children.size()); ++i) {
    auto current_outer_bn = scop_info_.analysis_result_.GetOuterBandNode(i);
    auto template_type = current_outer_bn->template_type;
    templates_map[template_type].emplace_back(i);
  }
  auto InsertIndexOfBand = [&sorted_bands, &templates_map]() {
    for (int templates = Template::DEFAULT; templates <= Template::TEMPLATE_BULK; ++templates) {
      auto it = templates_map.find(Template(templates));
      if (it == templates_map.end()) continue;
      for (auto i : it->second) {
        sorted_bands.emplace_back(i);
      }
    }
  };
  InsertIndexOfBand();
  return sorted_bands;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
