/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include "reorder_inner_band.h"

namespace akg {
namespace ir {
namespace poly {

std::vector<std::string> ExtractDimNames(const isl::aff &aff) {
  std::vector<std::string> dim_names;
  int dims = isl_aff_dim(aff.get(), isl_dim_in);
  CHECK_GE(dims, 0);
  for (int i = 0; i < dims; ++i) {
    isl_val *coef_val = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, i);
    int coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef != 0) {
      auto dim_name = std::string(isl_aff_get_dim_name(aff.get(), isl_dim_in, i));
      dim_names.push_back(dim_name);
    }
  }
  return dim_names;
}

isl::multi_union_pw_aff MergeTwoUPALs(const isl::multi_union_pw_aff &partial_schedule,
                                      const std::vector<isl::union_pw_aff> &dims_with_if,
                                      const std::vector<isl::union_pw_aff> &dims_without_if) {
  auto num_dims_with_if = dims_with_if.size();
  auto num_dims_without_if = dims_without_if.size();
  CHECK(partial_schedule.size() == num_dims_with_if + num_dims_without_if);
  auto new_schedule = partial_schedule;
  for (unsigned dim = 0; dim < num_dims_with_if; ++dim) {
    new_schedule = new_schedule.set_at(dim, dims_with_if[dim]);
  }
  for (unsigned dim = 0; dim < num_dims_without_if; ++dim) {
    new_schedule = new_schedule.set_at(dim + num_dims_with_if, dims_without_if[dim]);
  }
  return new_schedule;
}

void MergeTwoDimMaps(const std::vector<unsigned> &in1, const std::vector<unsigned> &in2, std::vector<unsigned> &out) {
  out.resize(in1.size() + in2.size());
  for (unsigned i = 0; i < in1.size(); ++i) {
    out[in1[i]] = i;
  }
  for (unsigned i = 0; i < in2.size(); ++i) {
    out[in2[i]] = i + in1.size();
  }
}

/*
 * Reorder the partial schedule such that isl::union_pw_aff with range var in cond_vars are
 * ordered before others.
 *
 * Example:
 * [{ S_0[j, k, l] -> [((k) mod 16)] },
 *  { S_0[j, k, l] -> [((l) mod 16)] },
 *  { S_0[j, k, l] -> [(0)] },
 *  { S_0[j, k, l] -> [((j) mod 32)] }]
 *
 * If "j" appears in the conditions, the partial schedule is transformed to:
 *
 * [{ S_0[j, k, l] -> [((j) mod 32)] },
 *  { S_0[j, k, l] -> [((k) mod 16)] },
 *  { S_0[j, k, l] -> [((l) mod 16)] },
 *  { S_0[j, k, l] -> [(0)] }]
 */
isl::multi_union_pw_aff ReorderLocalSchedule(const CondVarsMap &cond_vars,
                                             const isl::multi_union_pw_aff &partial_schedule,
                                             std::vector<unsigned> &dim_map, bool &need_update) {
  std::vector<isl::union_pw_aff> dims_with_if, dims_without_if;
  std::vector<unsigned> with_if_dim_map, without_if_dim_map;
  need_update = false;
  auto original_upal = partial_schedule.union_pw_aff_list();
  unsigned upal_size = original_upal.size();
  for (unsigned dim = 0; dim < upal_size; ++dim) {
    auto dim_schedule = original_upal.get_at(dim);
    bool found_dim_in_cond = false;
    dim_schedule.get_pw_aff_list().foreach([&cond_vars, &found_dim_in_cond](const isl::pw_aff &stmt_schedule) -> void {
      stmt_schedule.foreach_piece([&cond_vars, &found_dim_in_cond](const isl::set &set, const isl::aff &aff) -> void {
        isl::id stmt_id = set.get_tuple_id();
        if (cond_vars.count(stmt_id) == 0) return;
        const auto &cond_vars_in_stmt = cond_vars.at(stmt_id);
        auto dim_names = ExtractDimNames(aff);
        for (const auto &dim_name : dim_names) {
          if (cond_vars_in_stmt.count(dim_name)) found_dim_in_cond = true;
        }
      });
    });
    if (found_dim_in_cond) {
      with_if_dim_map.push_back(dim);
      dims_with_if.push_back(dim_schedule);
      need_update = true;
    } else {
      without_if_dim_map.push_back(dim);
      dims_without_if.push_back(dim_schedule);
    }
  }

  if (need_update) {
    MergeTwoDimMaps(with_if_dim_map, without_if_dim_map, dim_map);
    return MergeTwoUPALs(partial_schedule, dims_with_if, dims_without_if);
  } else {
    return partial_schedule;
  }
}

/*
 * isl::schedule_node_band does not provide an interface to update the partial schedule,
 * so we have to delete the band node, copy other attributes from the original node and
 * insert the new partial schedule.
 *
 * Note that the member coincident values needs to be updated according to the mapping
 * between original and new dims.
 */
isl::schedule_node setLocalSchedule(const isl::schedule_node_band &band,
                                    const isl::multi_union_pw_aff &partial_schedule,
                                    const std::vector<unsigned> &dim_map) {
  auto removed_band = band.del();
  auto new_band_obj = removed_band.insert_partial_schedule(partial_schedule);
  auto new_band = new_band_obj.copy();
  new_band = isl_schedule_node_band_set_permutable(new_band, band.get_permutable());
  auto ast_build_options = band.get_ast_build_options();
  new_band = isl_schedule_node_band_set_ast_build_options(new_band, ast_build_options.copy());
  unsigned n_member = band.n_member();
  CHECK(dim_map.size() == n_member);
  for (unsigned i = 0; i < n_member; ++i) {
    bool coincident = band.member_get_coincident(i);
    unsigned new_member = dim_map[i];
    new_band = isl_schedule_node_band_member_set_coincident(new_band, new_member, coincident);
  }
  return isl::manage(new_band);
}

isl::schedule_node RewriteLeafBandNode(const CondVarsMap &cond_vars, const isl::schedule_node_band &band) {
  auto partial_schedule = band.get_partial_schedule();
  std::vector<unsigned> dim_map;
  bool need_update = false;
  auto new_partial_schedule = ReorderLocalSchedule(cond_vars, partial_schedule, dim_map, need_update);
  if (!need_update)
    return band;
  else
    return setLocalSchedule(band, new_partial_schedule, dim_map);
}

isl::schedule ReorderInnerBand::Run(isl::schedule curr_schedule) {
  isl::schedule_node root = curr_schedule.get_root();
  auto cond_vars = cond_vars_;
  root = root.map_descendant_bottom_up([&cond_vars](const isl::schedule_node &node) -> isl::schedule_node {
    bool is_leaf_band = (node.as<isl::schedule_node_band>() && (node.n_children() == 1)
      && !node.is_subtree_anchored() && node.first_child().as<isl::schedule_node_leaf>());
    if (!is_leaf_band) return node;

    auto band = node.as<isl::schedule_node_band>();
    if (!band.get_permutable()) return node;
    return RewriteLeafBandNode(cond_vars, band);
  });
  return root.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
