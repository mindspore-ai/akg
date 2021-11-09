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

#include "schtree_analyzer.h"
#include "poly/tiling/tiling_analyzer.h"
#include "poly/schedule_pass.h"

namespace akg {
namespace ir {
namespace poly {
ScheduleTreeAnalyzer::ScheduleTreeAnalyzer(TilingAnalyzer *a, const isl::schedule &s) : analyzer_(a), sch_(s) {}

std::unique_ptr<TileAxis> ScheduleTreeAnalyzer::Build(const Stmt &stmt) {
  // Step 1: Get tile info (band, seq, range, coincident) from schedule tree.
  bool need_tiling = AnalyzeScheduleTree();
  if (!need_tiling) return std::move(root_);
  root_ = std::unique_ptr<TileAxis>(new (std::nothrow) TileAxis(nullptr, -1, -1, false, {}, false, analyzer_));
  CHECK(root_) << "memory alloc fail";

  // Step 2: Get detailed range info of loop from halide ir.
  AnalyzeHalide(stmt);

  // Step 3: Generate tile axis.
  CreateTileAxes();

  return std::move(root_);
}

void ScheduleTreeAnalyzer::GetDimRangeFromTree(const isl::schedule &schedule) {
  isl::union_set dom = schedule.get_domain();
  dom.foreach_set([&](const isl::set &s) -> void {
    auto n = static_cast<int>(s.n_dim());
    std::vector<std::pair<int64_t, Expr>> stmt_range;
    for (int i = 0; i < n; ++i) {
      isl::pw_aff pw_min = s.dim_min(i);
      isl::pw_aff pw_max = s.dim_max(i);
      Expr val_min;
      Expr val_max;
      Expr range;
      if ((pw_min.n_piece() != 1) || (pw_max.n_piece() != 1)) {
        stmt_range.emplace_back(std::make_pair(-1, Expr(-1)));
        continue;
      }
      pw_min.foreach_piece(
        [&](const isl::set &s, const isl::aff &a) -> void { val_min = a.get_constant_val().get_num_si(); });
      pw_max.foreach_piece([&](const isl::set &s, const isl::aff &a) -> void {
        std::string dim_name;
        int param_dim = isl_aff_dim(a.get(), isl_dim_param);
        CHECK_GE(param_dim, 0);
        for (auto j = 0; j < param_dim; ++j) {
          isl_val *coef_val = isl_aff_get_coefficient_val(a.get(), isl_dim_param, j);
          int coef = isl_val_get_num_si(coef_val);
          static_cast<void>(isl_val_free(coef_val));
          if (coef != 0) {
            dim_name = std::string(isl_aff_get_dim_name(a.get(), isl_dim_param, j));
            break;
          }
        }
        if (!dim_name.empty()) {
          range = dim_name;
        } else {
          val_max = a.get_constant_val().get_num_si();
          range = val_max - val_min + 1;
        }
      });
      const auto offset = val_min.as<IntImm>();
      CHECK(offset) << "Get expr offset in schedule tree: " << val_min;
      stmt_range.emplace_back(std::make_pair(offset->value, range));
    }
    dim_range_[s.get_tuple_name()] = stmt_range;
  });
}

bool ScheduleTreeAnalyzer::AnalyzeScheduleTree() {
  // Step 1: Init dim range in schedule tree and check loops.
  GetDimRangeFromTree(sch_);
  bool has_loop = false;
  for (const auto &it : dim_range_) {
    if (!it.second.empty()) {
      has_loop = true;
      break;
    }
  }
  if (!has_loop) return false;

  // Step 2: Construct tree nodes from each outer band.
  auto &band_nodes = analyzer_->scop_info_.analysis_result_.GetAllOuterBandNode();
  std::stringstream ss;
  for (auto &band_node : band_nodes) {
    auto *bn = band_node.get();
    isl::multi_union_pw_aff prefix_schedule = bn->node.get_partial_schedule();
    if (prefix_schedule.is_null()) {
      return false;
    }

    ss << "============ Band " << bn->index << " schedule tree ==========";
    analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);
    analyzer_->GetTileLogger().AppendLine(ANA_SCHETREE, prefix_schedule.to_str());

    ss << "=========== In total " << prefix_schedule.size() << " tileable axes ===========";
    analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);

    isl::union_pw_aff_list upa_list = prefix_schedule.get_union_pw_aff_list();
    candidates_.clear();
    tile_size_in_band_[bn->index] = upa_list.size();
    for (size_t i = 0; i < upa_list.size(); ++i) {
      isl::union_pw_aff upa = upa_list.get_at(static_cast<int>(i));
      isl::pw_aff_list pa_list = upa.get_pw_aff_list();
      GetCandidatesInSequence(i, pa_list, true, bn->node.member_get_coincident(static_cast<int>(i)));
    }

    std::vector<OuterBandNode *> stack;
    for (auto &ci : bn->children) {
      stack.emplace_back(ci.get());
    }

    while (!stack.empty()) {
      auto *child = stack.back();
      stack.pop_back();
      prefix_schedule = child->node.get_partial_schedule();
      if (prefix_schedule.is_null()) {
        return false;
      }

      ss << "============ Inner Band " << child->index << " schedule tree ==========";
      analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);
      analyzer_->GetTileLogger().AppendLine(ANA_SCHETREE, prefix_schedule.to_str());
      ss << "=========== In total " << prefix_schedule.size() << " tileable axes ===========";
      analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);

      upa_list = prefix_schedule.get_union_pw_aff_list();
      for (size_t i = 0; i < upa_list.size(); ++i) {  // partial schedule
        isl::union_pw_aff upa = upa_list.get_at(static_cast<int>(i));
        isl::pw_aff_list pa_list = upa.get_pw_aff_list();
        GetCandidatesInSequence(child->index + i, pa_list, false,
                                child->node.member_get_coincident(static_cast<int>(i)));
      }
      for (auto &ci : child->children) {
        stack.emplace_back(ci.get());
      }
    }

    ConstructTreePattern(static_cast<int>(bn->index));
  }
  return true;
}

void ScheduleTreeAnalyzer::GetCandidatesInSequence(size_t seq, const isl::pw_aff_list &pa_list, bool is_outer,
                                                   bool mc_sup) {
  auto FormatName = [&](std::string &var) {
    std::vector<char> remove_sign = {'{', '}', '[', ']', '(', ')', ' '};
    for (auto c : remove_sign) {
      var.erase(std::remove(var.begin(), var.end(), c), var.end());
    }
  };
  for (unsigned int j = 0; j < pa_list.size(); ++j) {
    isl::pw_aff pa = pa_list.get_at(j);
    auto in_dim_size = static_cast<int>(isl_pw_aff_dim(pa.get(), isl_dim_in));
    std::string pa_name = pa.domain().get_tuple_name();
    std::vector<std::string> all_var_names;
    for (int d = 0; d < in_dim_size; ++d) {
      const char *dim_name = isl_pw_aff_get_dim_name(pa.get(), isl_dim_in, d);
      CHECK(dim_name) << "Cannot read stmt name in schedule tree";
      std::string n = dim_name;
      all_var_names.emplace_back(n);
    }
    bool has_var = false;
    pa.foreach_piece([&](const isl::set &s, const isl::aff &a) -> void {
      has_var = has_var || pa.nonneg_set().max_val(a).is_infty();
    });
    if (!has_var) continue;

    std::vector<std::string> sp_var = akg::common::Split(pa.to_str(), "->");
    CHECK_GE(sp_var.size(), 2U) << "error, missing -> in schedule tree analyze";

    std::string var = sp_var.back();
    FormatName(var);

    for (size_t pos = 0; pos < all_var_names.size(); ++pos) {
      std::string n = all_var_names[pos];
      if (var.find(n) == std::string::npos) continue;
      auto dr = dim_range_.find(pa_name);
      CHECK(dr != dim_range_.end()) << "Cannot find " << pa_name << "'s dim range";
      std::vector<std::pair<int64_t, Expr>> ranges = dr->second;
      CHECK(!ranges.empty() && pos <= ranges.size() - 1) << "Cannot map " << pa_name << " 's range";
      if (const auto var_name = ranges[pos].second.as<StringImm>()) {
        if (var != n) continue;
        auto cit = candidates_.find(seq);
        if (cit == candidates_.end()) {
          candidates_[seq] = {
            TilePos{is_outer, pos, pa_name, var, n, ranges[pos].first, Expr(var_name->value), mc_sup}};
        } else {
          candidates_[seq].emplace_back(
            TilePos{is_outer, pos, pa_name, var, n, ranges[pos].first, Expr(var_name->value), mc_sup});
        }
      } else if (const auto mr_imm = ranges[pos].second.as<IntImm>()) {
        int mr = mr_imm->value;
        std::pair<int, int> trange(0, mr);
        if (var != n) {
          std::pair<int, int> new_range = trange;
          // substring of name matched, continue
          if (!GetPosShiftedTileRange(var, n, new_range) && !GetNegShiftedTileRange(var, n, new_range)) {
            continue;
          }
          trange = new_range;
        }
        auto cit = candidates_.find(seq);
        if (cit == candidates_.end()) {
          candidates_[seq] = {TilePos{is_outer, pos, pa_name, var, n, trange.first, Expr(trange.second), mc_sup}};
        } else {
          bool is_same = false;
          for (auto &tp : cit->second) {
            const auto tm = tp.max_range.as<IntImm>();
            if ((tp.min_range == trange.first && (tm && tm->value == trange.second)) ||
                (tp.var_name == var && tp.var_pos == pos)) {
              is_same = true;
              break;
            }
          }
          if (!is_same) {
            candidates_[seq].emplace_back(
              TilePos{is_outer, pos, pa_name, var, n, trange.first, Expr(trange.second), mc_sup});
          }
        }
      } else {
        LOG(FATAL) << "Unknown type " << ranges[pos].second;
      }
      break;
    }
  }
}

bool ScheduleTreeAnalyzer::GetPosShiftedTileRange(const std::string &vname, const std::string &actual_name,
                                                  std::pair<int, int> &old_ranges) {
  std::pair<int, int> ranges = old_ranges;
  std::vector<std::string> sp_add = akg::common::Split(vname, "+");
  if (sp_add.size() != 2U) return false;
  std::string pre = sp_add[0];
  std::string post = sp_add[1];
  if (pre.empty() || post.empty()) return false;
  if (pre == actual_name && post != actual_name) {
    auto add_range = static_cast<int>(std::strtol(post.c_str(), nullptr, 10));
    ranges.first += add_range;
    ranges.second += add_range;
    old_ranges = ranges;
    return true;
  } else if (post == actual_name && pre != actual_name) {
    auto add_range = static_cast<int>(std::strtol(pre.c_str(), nullptr, 10));
    ranges.first += add_range;
    ranges.second += add_range;
    old_ranges = ranges;
    return true;
  }
  return false;
}

bool ScheduleTreeAnalyzer::GetNegShiftedTileRange(const std::string &vname, const std::string &actual_name,
                                                  std::pair<int, int> &old_ranges) {
  std::pair<int, int> ranges = old_ranges;
  std::vector<std::string> sp_sub = akg::common::Split(vname, "-");
  if (sp_sub.size() != 2U) return false;
  std::string pre = sp_sub[0];
  std::string post = sp_sub[1];
  if (pre.empty() || post.empty()) return false;
  if (pre == actual_name && post != actual_name) {
    auto sub_range = static_cast<int>(std::strtol(post.c_str(), nullptr, 10));
    ranges.first -= sub_range;
    ranges.second -= sub_range;
    old_ranges = ranges;
    return true;
  } else if (post == actual_name && pre != actual_name) {
    auto sub_range = static_cast<int>(std::strtol(pre.c_str(), nullptr, 10));
    std::pair<int, int> res;
    res.second = sub_range - ranges.first;
    res.first = sub_range - ranges.second;
    if (res.first < 0) {
      res.second += (-res.first);
      res.first = 0;
    }
    old_ranges = res;
    return true;
  }
  return false;
}
void ScheduleTreeAnalyzer::ConstructTreePattern(int band_id) {
  for (size_t p = 0; p < candidates_.size(); ++p) {
    auto cit = candidates_.find(p);
    if (cit == candidates_.end()) continue;
    for (const auto &tp : cit->second) {
      tile_nodes_.emplace_back(TileNode{tp.is_outer, band_id, p, tp.min_range, tp.max_range, tp.var_pos, tp.mc_sup});
    }
  }
}

void ScheduleTreeAnalyzer::AnalyzeHalide(const Stmt &stmt) {
  class HalideVisitor : public IRVisitor {
   public:
    void Collect(const Stmt &stmt) { this->Visit(stmt); }
    void Visit_(const For *op) override {
      cur_loop_ = op;
      loop_count_ += 1;
      cur_band_.emplace_back(op);
      IRVisitor::Visit_(op);
      cur_loop_ = op;
      loop_count_ -= 1;
      if (loop_count_ == 0) {
        band_list_.emplace_back(cur_band_);
        cur_band_.clear();
      }
    }

    void Visit_(const Provide *op) final {
      auto it = provides_map_.find(cur_loop_);
      if (it == provides_map_.end()) {
        provides_map_[cur_loop_] = {op};
      } else {
        provides_map_[cur_loop_].emplace_back(op);
      }
      IRVisitor::Visit_(op);
    }

    void Visit_(const IfThenElse *op) final {
      auto it = ifs_map_.find(cur_loop_);
      if (it == ifs_map_.end()) {
        ifs_map_[cur_loop_] = {op};
      } else {
        ifs_map_[cur_loop_].emplace_back(op);
      }
      IRVisitor::Visit_(op);
    }
    std::unordered_map<const For *, std::vector<const Provide *>> provides_map_;
    std::unordered_map<const For *, std::vector<const IfThenElse *>> ifs_map_;
    std::vector<Band> band_list_;

   private:
    Band cur_band_;
    const For *cur_loop_{nullptr};
    int loop_count_ = 0;
  };
  // Step 1: Collect Provide and IfThenElse node.
  HalideVisitor visitor;
  visitor.Collect(stmt);
  provides_map_ = std::move(visitor.provides_map_);
  ifs_map_ = std::move(visitor.ifs_map_);
  band_list_ = std::move(visitor.band_list_);

  // Step 2: Separate tileable band and un-tileable band.
  auto CountUniqueLoopName = [&](const Band &band) -> size_t {
    std::unordered_set<std::string> names;
    for (auto loop : band) {
      names.insert(loop->loop_var.get()->name_hint);
    }
    return names.size();
  };

  for (const auto &cur_band : band_list_) {
    size_t cur_tb_size = tileable_band_.size();
    size_t cur_band_size = CountUniqueLoopName(cur_band);
    auto it = tile_size_in_band_.find(cur_tb_size);
    if (it != tile_size_in_band_.end() && it->second <= cur_band_size)
      tileable_band_.emplace_back(cur_band);
    else
      untileable_band_.emplace_back(cur_band);
  }

  // Step 3: Calculate loop's possible range (considering shift).
  AddLoopRangeFromBand();
  AddLoopRangeFromIfs();

  // Step 4: Mark loop with data size by tensor which use the index of loop.
  AddLoopDataSize();
}

void ScheduleTreeAnalyzer::AddLoopRangeFromBand() {
  for (const auto &band : band_list_) {
    for (auto loop : band) {
      const auto offset = loop->min.as<IntImm>();
      if (offset == nullptr) {
        continue;
      }
      auto GetReplacedName = [this, loop]() -> std::string {
        std::string var_name;
        Expr elim_shift = loop->extent;
        if (const auto add = loop->extent.as<Add>()) {
          if (add->b.as<IntImm>()) {
            elim_shift = add->a;
          }
        }
        auto params_rev_map = analyzer_->scop_info_.user_config_.GetParamsRevMap();
        for (const auto &it : params_rev_map) {
          if (Equal(loop->extent, it.second) || Equal(elim_shift, it.second)) {
            return it.first;
          }
        }
        return var_name;
      };
      std::string rep_name = GetReplacedName();
      loop_seq_.emplace_back(loop);
      if (!rep_name.empty()) {
        loop_dynamic_range_map_[loop] = {std::make_pair(offset->value, rep_name)};
      } else if (const auto extent = loop->extent.as<IntImm>()) {
        int min = static_cast<int>(offset->value);
        int max = static_cast<int>(offset->value + extent->value);
        loop_range_map_[loop] = {std::make_pair(min, max)};
      } else if (const auto var = loop->extent.as<Variable>()) {
        loop_dynamic_range_map_[loop] = {std::make_pair(offset->value, var->name_hint)};
      } else if (const auto mul = loop->extent.as<Mul>()) {
        if (mul->a.as<Variable>() && mul->b.as<IntImm>()) {
          loop_dynamic_range_map_[loop] = {std::make_pair(offset->value, mul->a.as<Variable>()->name_hint)};
        } else if (mul->b.as<Variable>() && mul->a.as<IntImm>()) {
          loop_dynamic_range_map_[loop] = {std::make_pair(offset->value, mul->b.as<Variable>()->name_hint)};
        } else {
          LOG(FATAL) << "Only support variable multiplies integer, but found " << loop->extent;
        }
      } else {
        LOG(WARNING) << "Match loop fail, cannot tile loop with extent " << loop->extent;
      }
    }
  }
}

void ScheduleTreeAnalyzer::AddLoopRangeFromIfs() {
  for (auto &it : ifs_map_) {
    std::vector<const IfThenElse *> ifs = it.second;
    for (auto cond : ifs) {
      std::deque<Expr> if_expr;
      auto Analyze = [&if_expr](const NodeRef &op) {
        if (const And *and_op = op.as<And>()) {
          if_expr.emplace_back(and_op->a);
          if_expr.emplace_back(and_op->b);
        }
      };
      air::ir::PostOrderVisit(cond->condition, Analyze);
      if (if_expr.empty()) if_expr.emplace_back(cond->condition);

      for (const auto &e : if_expr) {
        if (const auto ge = e.as<GE>()) {
          const auto var = ge->a.as<Variable>();
          const auto rshift = ge->b.as<IntImm>();
          if (var == nullptr || rshift == nullptr) {
            continue;
          }
          Band preloops = GetPreviousLoops(it.first);
          while (!preloops.empty()) {
            const For *l = preloops.back();
            preloops.pop_back();
            if (l->loop_var->name_hint == var->name_hint) {
              std::vector<std::pair<int64_t, int64_t>> new_ranges;
              for (auto r : loop_range_map_[l]) {
                new_ranges.emplace_back(std::make_pair(rshift->value, r.second));
              }
              loop_range_map_[l].insert(loop_range_map_[l].begin(), new_ranges.begin(), new_ranges.end());
              break;
            }
          }
        } else if (const auto le = e.as<LE>()) {
          const auto var = le->a.as<Variable>();
          const auto lshift = le->b.as<IntImm>();
          if (var && lshift) {
            Band preloops = GetPreviousLoops(it.first);
            while (!preloops.empty()) {
              const For *l = preloops.back();
              preloops.pop_back();
              if (l->loop_var->name_hint != var->name_hint) {
                continue;
              }
              std::vector<std::pair<int64_t, int64_t>> new_ranges;
              for (auto r : loop_range_map_[l]) {
                new_ranges.emplace_back(std::make_pair(r.first, lshift->value + 1));
              }
              loop_range_map_[l].insert(loop_range_map_[l].begin(), new_ranges.begin(), new_ranges.end());
              break;
            }
          }
        }
      }
    }
  }
}

void ScheduleTreeAnalyzer::AddLoopDataSize() {
  for (const auto &it : provides_map_) {
    if (it.first == nullptr) continue;
    std::vector<const Provide *> pros = it.second;
    for (const Provide *p : pros) {
      int data_size = analyzer_->scop_info_.user_config_.GetDataBytes(p->func->func_name());
      VarNames related_name;
      auto ExtractName = [this, &related_name](const NodeRef &op) {
        if (const Call *call = op.as<Call>()) {
          for (auto arg : call->args) {
            related_name = VisitVarNames(arg, related_name);
          }
        }
      };
      for (auto arg : p->args) {
        related_name = VisitVarNames(arg, related_name);
      }
      air::ir::PostOrderVisit(p->value, ExtractName);
      Band pre_loops = GetPreviousLoops(it.first);
      for (auto loop : pre_loops) {
        for (const auto &name : related_name) {
          if (name != loop->loop_var.get()->name_hint) continue;
          loop_data_size_map_[loop] = std::make_pair(p->func->func_name(), data_size);
          break;
        }
      }
    }
  }
}

int ScheduleTreeAnalyzer::GetLayerIndex(const std::string &var_name) {
  std::string layer_s;
  for (char i : var_name) {
    if (i >= '0' && i <= '9') {
      layer_s += i;
    }
  }
  return layer_s.empty() ? -1 : static_cast<int>(std::strtol(layer_s.c_str(), nullptr, 10));
}

bool ScheduleTreeAnalyzer::MatchNodeWithDynamicLoop(std::unordered_set<const For *> &matched, TileNode &node,
                                                    const For *loop) {
  if (matched.find(loop) != matched.end()) return false;
  auto it = loop_dynamic_range_map_.find(loop);
  if (it == loop_dynamic_range_map_.end()) return false;
  CHECK(loop);
  std::vector<std::pair<int64_t, std::string>> ranges = it->second;
  std::string var_name = loop->loop_var.get()->name_hint;
  int layer_index = GetLayerIndex(var_name);
  if (layer_index == -1) return false;
  auto InitNode = [this, &node, &matched, layer_index, loop]() {
    node.loop = loop;
    auto it1 = this->loop_data_size_map_.find(loop);
    if (it1 == this->loop_data_size_map_.end()) return;
    node.data_size = it1->second;
    matched.insert(loop);
    if (!node.is_outer) node.axis = layer_index;
  };
  for (const auto &r : ranges) {
    if (const auto nm = node.range_max.as<StringImm>()) {
      if (nm->value != r.second) continue;
    } else if (node.range_max.as<IntImm>() == nullptr) {
      LOG(INFO) << "Detect unknown type " << node.range_max;
      continue;
    }
    InitNode();
    return true;
  }
  return false;
}

bool ScheduleTreeAnalyzer::MatchNodeWithLoop(std::unordered_set<const For *> &matched, TileNode &node,
                                             const For *loop) {
  if (matched.find(loop) != matched.end()) return false;
  auto it = loop_range_map_.find(loop);
  if (it == loop_range_map_.end()) return false;
  std::vector<std::pair<int64_t, int64_t>> ranges = it->second;
  CHECK(loop);
  std::string var_name = loop->loop_var.get()->name_hint;
  int layer_index = GetLayerIndex(var_name);
  if (layer_index == -1) return false;
  if (node.is_outer && static_cast<int>(node.axis) != layer_index) return false;
  for (auto r : ranges) {
    const auto nm = node.range_max.as<IntImm>();
    if (nm == nullptr) continue;
    if ((node.range_min == r.first) && (nm->value == r.second)) {
      // strict match
      node.loop = loop;
      auto it1 = this->loop_data_size_map_.find(loop);
      if (it1 == this->loop_data_size_map_.end()) continue;
      node.data_size = it1->second;
      matched.insert(loop);
      if (!node.is_outer) node.axis = layer_index;
      return true;
    } else if ((node.range_min == r.second) || (nm->value == r.first) || (nm->value - 1 == r.first) ||
               ((node.range_min != 0 || r.first != 0) && ((node.range_min == r.first) || (nm->value == r.second)))) {
      // shift match has two cases:
      // 1) concat (A.max == B.min or A.min == B.max) e.g. A = [0, 37] B = [37, 2331]
      // 2) contain ((A.min != 0 or B.min != 0) and (A.max == B.max (Left contain) or A.min == B.min (Right contain)))
      // e.g. A = [0, 2331] B = [37, 2331]
      node.loop = loop;
      auto it2 = this->loop_data_size_map_.find(loop);
      if (it2 == this->loop_data_size_map_.end()) continue;
      node.data_size = it2->second;
      matched.insert(loop);
      if (!node.is_outer) node.axis = layer_index;
      return true;
    }
  }
  return false;
}

void ScheduleTreeAnalyzer::CreateTileAxes() {
  std::unordered_set<const For *> matched;
  std::vector<int> unmatched_pos;
  for (size_t i = 0; i < tile_nodes_.size(); ++i) {
    bool match = false;
    CHECK_LE(tile_nodes_[i].index + 1, tileable_band_.size());
    Band band = tileable_band_[tile_nodes_[i].index];
    for (auto loop : band) {
      match =
        MatchNodeWithLoop(matched, tile_nodes_[i], loop) || MatchNodeWithDynamicLoop(matched, tile_nodes_[i], loop);
      if (match) break;
    }
    if (!match) unmatched_pos.emplace_back(i);
  }

  for (int unmatched_po : unmatched_pos) {
    bool match = false;
    for (const auto &band : untileable_band_) {
      for (auto loop : band) {
        match = MatchNodeWithLoop(matched, tile_nodes_[unmatched_po], loop) ||
                MatchNodeWithDynamicLoop(matched, tile_nodes_[unmatched_po], loop);
        if (match) break;
      }
      if (match) break;
    }
  }

  TileAxis *last_axis = root_.get();
  auto SortNodes = [&](const TileNode &n1, const TileNode &n2) {
    if (n1.index != n2.index) {
      return n1.index < n2.index;
    } else {
      if (n1.axis != n2.axis) {
        return n1.axis < n2.axis;
      } else {
        if (n1.loop != nullptr && n2.loop != nullptr) {
          if (analyzer_->arith_ana_.CanProve(n1.loop->min == n2.loop->min)) {
            return !analyzer_->arith_ana_.CanProve(n1.loop->extent >= n2.loop->extent);
          } else if (analyzer_->arith_ana_.CanProve(n1.loop->extent == n2.loop->extent)) {
            return !analyzer_->arith_ana_.CanProve(n2.loop->min < n1.loop->min);
          } else {
            return true;
          }
        }
        return n1.loop != nullptr;
      }
    }
  };

  std::sort(tile_nodes_.begin(), tile_nodes_.end(), SortNodes);
  auto InsertDefinedLoop = [this](const TileNode &n) {
    if (n.range_max.as<IntImm>()) {
      defined_static_loop_.emplace_back(n.loop);
    } else {
      defined_dynamic_loop_.emplace_back(n.loop);
    }
  };
  for (const auto &node : tile_nodes_) {
    if (node.loop == nullptr) continue;
    if (node.index == last_axis->index + 1) last_axis = root_.get();
    if (static_cast<int>(node.axis) > last_axis->dim_axis) {
      std::unique_ptr<TileAxis> cur_axis(new (std::nothrow) TileAxis(
        last_axis, node.index, static_cast<int>(node.axis), node.mc_sup, node.data_size, !node.is_outer, analyzer_));
      if (node.loop != nullptr) {
        cur_axis->LinkToLoop(node.loop);
        InsertDefinedLoop(node);
        RecordTreeRanges(cur_axis.get(), node.loop);
      }
      last_axis->children.emplace_back(std::move(cur_axis));
      last_axis = last_axis->children.back().get();
    } else if (static_cast<int>(node.axis) == last_axis->dim_axis) {
      CHECK(!last_axis->loops.empty()) << "Error, empty loop seq";
      if (node.loop != nullptr) {
        last_axis->LinkToLoop(node.loop);
        InsertDefinedLoop(node);
        RecordTreeRanges(last_axis, node.loop);
      }
    }
  }
  CreateAxisForUndefinedLoop(last_axis);
}

void ScheduleTreeAnalyzer::CreateAxisForUndefinedLoop(TileAxis *last_axis) {
  auto GetSameNameLoop = [this](const For *l) -> const For * {
    CHECK(l);
    for (auto dl : defined_static_loop_) {
      if (dl->loop_var.get()->name_hint == l->loop_var.get()->name_hint) return dl;
    }
    return nullptr;
  };
  std::stringstream ss;
  for (auto loop : loop_seq_) {
    if (loop_range_map_.find(loop) != loop_range_map_.end() &&
        std::find(defined_static_loop_.begin(), defined_static_loop_.end(), loop) == defined_static_loop_.end()) {
      auto snl = GetSameNameLoop(loop);
      bool matched = false;
      if (snl != nullptr) {
        ss << "Same name loop " << loop << " with range " << loop->min << "," << loop->extent;
        std::vector<TileAxis *> stack;
        stack.emplace_back(last_axis);
        while (!matched && !stack.empty()) {
          TileAxis *cur = stack.back();
          stack.pop_back();
          for (auto i = 0u; i < cur->loops.size(); ++i) {
            if (cur->loops[i] == snl) {
              cur->LinkToLoop(loop);
              RecordTreeRanges(cur, loop);
              matched = true;
              break;
            }
          }
          if (cur->parent != nullptr) {
            if (cur->parent->index == last_axis->index) {
              stack.emplace_back(cur->parent);
              for (size_t i = 0; i < cur->parent->children.size(); ++i) {
                if (cur->parent->children[i].get() != cur && cur->parent->children[i].get()->index == last_axis->index)
                  stack.emplace_back(cur->parent->children[i].get());
              }
            }
          }
        }
      }
      if (!matched) {
        ss << "Undefined loop " << loop;
        std::unique_ptr<TileAxis> inner(new (std::nothrow) TileAxis(
          last_axis, last_axis->index, last_axis->dim_axis + 1, false, {}, true, analyzer_));
        CHECK(inner) << "memory alloc fail";
        inner->LinkToLoop(loop);
        RecordTreeRanges(last_axis, loop);
        last_axis->children.emplace_back(std::move(inner));
        last_axis = last_axis->children.back().get();
      }

      analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);
    } else if (loop_dynamic_range_map_.find(loop) != loop_dynamic_range_map_.end() &&
               std::find(defined_dynamic_loop_.begin(), defined_dynamic_loop_.end(), loop) ==
                 defined_dynamic_loop_.end()) {
      ss << "Undefined loop " << loop;
      auto p = last_axis;
      bool found = false;
      while (p != nullptr && !found) {
        for (auto i = 0u; i < p->loops.size(); ++i) {
          if (p->loops[i] == nullptr) continue;
          if (p->loops[i]->loop_var.get()->name_hint == loop->loop_var.get()->name_hint) {
            p->LinkToLoop(loop);
            RecordTreeRanges(p, loop);
            found = true;
          }
        }
        p = p->parent;
      }
      if (found) continue;
      std::unique_ptr<TileAxis> inner(
        new (std::nothrow) TileAxis(last_axis, last_axis->index, last_axis->dim_axis + 1, false, {}, true, analyzer_));
      CHECK(inner) << "memory alloc fail";
      inner->LinkToLoop(loop);
      RecordTreeRanges(last_axis, loop);
      last_axis->children.emplace_back(std::move(inner));
      last_axis = last_axis->children.back().get();
      analyzer_->GetTileLogger().AppendLog(ANA_SCHETREE, ss);
    }
  }
}
void ScheduleTreeAnalyzer::RecordTreeRanges(TileAxis *axis, const For *loop) {
  std::vector<std::pair<int64_t, Expr>> ranges;
  if (loop_range_map_.find(loop) != loop_range_map_.end()) {
    for (auto r : loop_range_map_[loop]) {
      std::pair<int64_t, Expr> var_range;
      var_range.first = r.first;
      var_range.second = Expr(r.second);
      ranges.emplace_back(var_range);
    }
  } else {
    CHECK(loop_dynamic_range_map_.find(loop) != loop_dynamic_range_map_.end()) << "Loop range not record, error";
    for (const auto &r : loop_dynamic_range_map_[loop]) {
      std::pair<int64_t, Expr> var_range;
      var_range.first = r.first;
      var_range.second = Expr(r.second);
      ranges.emplace_back(var_range);
    }
  }
  for (const auto &r : ranges) axis->tree_ranges.emplace_back(r);
}

void ScheduleTreeAnalyzer::AnalyzeCubeInfo() {
  std::string res = analyzer_->scop_info_.mmu_info_.GetCName();
  for (const auto &it : provides_map_) {
    std::vector<const Provide *> pros = it.second;
    std::vector<const Call *> op_list;
    auto GetDeepestCall = [this, &op_list](const NodeRef &op) {
      if (const Call *call = op.as<Call>()) {
        for (auto arg : call->args) {
          // call has inner call
          if (arg.as<Call>()) return;
        }
        if (call->name != analyzer_->scop_info_.mmu_info_.GetAName() &&
            call->name != analyzer_->scop_info_.mmu_info_.GetBName() &&
            call->name != analyzer_->scop_info_.mmu_info_.GetCName()) {
          return;
        }
        op_list.emplace_back(call);
      }
    };
    for (auto op : pros) {
      if (op->func->func_name() != res) continue;
      if (op->value.as<Call>()) {
        air::ir::PostOrderVisit(op->value, GetDeepestCall);
      }
    }
    auto SortMatrixInCBAOrder = [this](const Call *c1, const Call *c2) {
      if (c1->name == this->analyzer_->scop_info_.mmu_info_.GetCName() ||
          c2->name == this->analyzer_->scop_info_.mmu_info_.GetCName()) {
        return (c1->name == this->analyzer_->scop_info_.mmu_info_.GetCName());
      } else if (c1->name == this->analyzer_->scop_info_.mmu_info_.GetBName() ||
                 c2->name == this->analyzer_->scop_info_.mmu_info_.GetBName()) {
        return (c2->name == this->analyzer_->scop_info_.mmu_info_.GetBName());
      }
      return true;
    };
    std::sort(op_list.begin(), op_list.end(), SortMatrixInCBAOrder);
    if (op_list.size() != 3U) continue;
    if (analyzer_->op_type_ == GEMM_OP) {
      MatchGemmVarNames(op_list);
    } else if (analyzer_->op_type_ == CONV_OP) {
      for (auto call : op_list) {
        if (analyzer_->scop_info_.mmu_info_.IsConvBackpropFilter()) {
          MatchConvFilterVarNames(call);
        } else {
          MatchConvVarNames(call);
        }
      }
    }
    Band pre_loops = GetPreviousLoops(it.first);
    for (auto l : pre_loops) {
      for (const auto &it2 : mmu_var_map_) {
        std::string lname = it2.first;
        std::string type = it2.second;
        if (l->loop_var.get()->name_hint != lname) continue;
        TileAxis *axis = analyzer_->Axis(l);
        CHECK(axis) << "cannot find axis for " << l->loop_var.get()->name_hint;
        std::string key = analyzer_->op_type_ == CONV_OP ? AT_CONV : AT_GEMM;
        axis->attrs.emplace_back(AttrInfo{key, type});
        break;
      }
    }
    mmu_var_map_.clear();
  }
}

// Match in C -> A -> B sequence
void ScheduleTreeAnalyzer::MatchConvVarNames(const Call *call) {
  int count = -1;
  for (auto arg : call->args) {
    count += 1;
    VarNames var_names;
    var_names = VisitVarNames(arg, var_names, call->name == analyzer_->scop_info_.mmu_info_.GetCName());
    if (var_names.empty()) continue;
    if (var_names.size() == 1U) {
      std::string name = var_names[0];
      if (name == "0") {
        continue;
      } else {
        if (call->name == analyzer_->scop_info_.mmu_info_.GetCName()) {
          mmu_var_map_[name] = DsaNC1HWC0[count];
        } else if (call->name == analyzer_->scop_info_.mmu_info_.GetAName()) {
          if (ForwardFeaturemap[count] == "N") {
            CHECK(mmu_var_map_.find(name) != mmu_var_map_.end());
            CHECK_EQ(mmu_var_map_[name], "N");
          } else if (ForwardFeaturemap[count] == "H_in") {
            if (mmu_var_map_.find(name) == mmu_var_map_.end()) {  // H is 1
              mmu_var_map_[name] = "kh";
            }  // else kh is 1
          } else if (ForwardFeaturemap[count] == "W_in") {
            if (mmu_var_map_.find(name) == mmu_var_map_.end()) {  // W is 1
              mmu_var_map_[name] = "kw";
            }  // else kw is 1
          } else if (ForwardFeaturemap[count] == "C1_in") {
            if (mmu_var_map_.find(name) == mmu_var_map_.end()) {  // normal conv
              mmu_var_map_[name] = ForwardFeaturemap[count];
            } else {  // depthwise
              mmu_var_map_[name] = "C1_in_out";
            }
          } else {
            if (mmu_var_map_.find(name) == mmu_var_map_.end()) {
              mmu_var_map_[name] = ForwardFeaturemap[count];
            } else {
              CHECK(ForwardFeaturemap[count].find(mmu_var_map_[name]) != std::string::npos);
            }
          }
        }
      }
    } else {  // only H_in, W_in in FM and C1_in in FT
      CHECK(call->name != analyzer_->scop_info_.mmu_info_.GetCName());
      if (call->name == analyzer_->scop_info_.mmu_info_.GetAName()) {
        CHECK(ForwardFeaturemap[count] == "H_in" || ForwardFeaturemap[count] == "W_in");
        for (const auto &name : var_names) {
          if (mmu_var_map_.find(name) == mmu_var_map_.end()) {  // kh or kw
            if (ForwardFeaturemap[count] == "H_in") {
              mmu_var_map_[name] = "kh";
            } else if (ForwardFeaturemap[count] == "W_in") {
              mmu_var_map_[name] = "kw";
            }
          }
        }
      } else if (call->name == analyzer_->scop_info_.mmu_info_.GetBName()) {
        CHECK(ForwardFilter[count] == "C1_in" || BackpropFilter[count] == "C1_out");
        for (const auto &name : var_names) {
          CHECK(mmu_var_map_.find(name) != mmu_var_map_.end());
        }
      }
    }
  }
}

void ScheduleTreeAnalyzer::MatchConvFilterVarNames(const Call *call) {
  if (call->name != analyzer_->scop_info_.mmu_info_.GetAName() &&
      call->name != analyzer_->scop_info_.mmu_info_.GetCName())
    return;
  int count = -1;
  for (auto arg : call->args) {
    count += 1;
    VarNames var_names;
    var_names = VisitVarNames(arg, var_names, call->name == analyzer_->scop_info_.mmu_info_.GetCName());
    if (var_names.empty()) continue;
    if (var_names.size() == 1U) {
      std::string name = var_names[0];
      if (name == "0") {
        continue;
      } else {
        if (call->name == analyzer_->scop_info_.mmu_info_.GetCName()) {
          mmu_var_map_[name] = FilterOutput[count];
        } else {
          mmu_var_map_[name] = FilterInput[count];
        }
      }
    } else {
      CHECK(call->name == analyzer_->scop_info_.mmu_info_.GetAName());
      for (const auto &name : var_names) {
        if (mmu_var_map_.find(name) == mmu_var_map_.end()) {
          mmu_var_map_[name] = FilterInput[count];
          break;
        }
      }
    }
  }
}

void ScheduleTreeAnalyzer::MatchGemmVarNames(std::vector<const Call *> op_list) {
  std::vector<VarNames> var_name_list;
  VarNames mx_a, mx_b, mx_c;
  CHECK_GE(op_list.size(), 3);
  for (auto arg : op_list[0]->args) {
    mx_c = VisitVarNames(arg, mx_c, false);
  }
  for (auto arg : op_list[1]->args) {
    mx_a = VisitVarNames(arg, mx_a, false);
  }
  for (auto arg : op_list[2]->args) {
    mx_b = VisitVarNames(arg, mx_b, false);
  }
  var_name_list.emplace_back(mx_c);
  var_name_list.emplace_back(mx_a);
  var_name_list.emplace_back(mx_b);
  mmu_var_map_ = ExtractLoopIndicesFromMatrices(var_name_list);
}

Band ScheduleTreeAnalyzer::GetPreviousLoops(const For *loop) {
  Band pre_band;
  if (nullptr == loop) return pre_band;
  for (const auto &band : this->band_list_) {
    pre_band.clear();
    for (auto l : band) {
      pre_band.emplace_back(l);
      if (l == loop) return pre_band;
    }
  }

  std::stringstream ss;
  ss << "Loop " << loop->loop_var.get()->name_hint << " not found";
  analyzer_->GetTileLogger().LogFatalAndSaveLog(ss.str());
  return pre_band;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
