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

#include "keep_outer_band_order.h"

namespace akg {
namespace ir {
namespace poly {

isl::schedule KeepOuterBandOrder::Run(isl::schedule sch) {
  auto outer_band_node = GetOuterBand(sch.get_root());
  if (!outer_band_node.isa<isl::schedule_node_band>()) {
    return sch;
  }
  auto outer_band = outer_band_node.as<isl::schedule_node_band>();
  if (!outer_band.get_permutable()) {
    return sch;
  }
  auto mupa = outer_band.get_partial_schedule();
  auto n_member = mupa.size();
  // rank outer band members according to input order
  std::multimap<size_t, size_t> axes_scores;  // map score to old axes
  for (auto i = 0u; i < n_member; ++i) {
    auto upa = mupa.get_at(i);
    size_t axis_score = 0;
    upa.get_pw_aff_list().foreach([&](const isl::pw_aff &pw_aff) {
      pw_aff.foreach_piece([&](const isl::set &set, const isl::aff &aff) {
        size_t n_dims = isl_aff_dim(aff.get(), isl_dim_in);
        CHECK_GE(n_dims, 0);
        // vector
        size_t min_dim_in_aff = 0;
        if (info_.mmu_info_.HasCube()) {
          // cube
          min_dim_in_aff = n_dims;
        }
        for (auto j = 0u; j < n_dims; ++j) {
          auto coef_val = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, j);
          if (isl_val_get_num_si(coef_val) != 0) {
            min_dim_in_aff = j;
            break;
          }
          static_cast<void>(isl_val_free(coef_val));
        }
        axis_score += min_dim_in_aff;
      });
    });
    axes_scores.insert(std::make_pair(axis_score, i));
  }

  std::vector<size_t> axes_map;  // new axes to old axes map
  for (auto it : axes_scores) {
    axes_map.push_back(it.second);
  }

  // construct new outer band according to the axes map
  isl::union_pw_aff_list new_upal;
  for (auto i = 0u; i < n_member; ++i) {
    if (i == 0) {
      new_upal = isl::union_pw_aff_list(mupa.get_at(axes_map[i]));
    } else {
      new_upal = new_upal.add(mupa.get_at(axes_map[i]));
    }
  }

  auto new_mupa = isl::multi_union_pw_aff(mupa.get_space(), new_upal);

  // save permutable and coincident of old node
  bool permutable = outer_band.get_permutable();
  std::vector<bool> coincident;
  for (auto i = 0u; i < n_member; ++i) {
    coincident.push_back(outer_band.member_get_coincident(axes_map[i]));
  }
  if (!info_.mmu_info_.HasCube()) {
    coincident[0] = true;
  }

  // delete old node
  outer_band_node = outer_band_node.del();

  // insert new node
  outer_band_node = outer_band_node.insert_partial_schedule(new_mupa);
  outer_band_node = outer_band_node.as<isl::schedule_node_band>().set_permutable(permutable);
  for (auto i = 0u; i < n_member; ++i) {
    outer_band_node = outer_band_node.as<isl::schedule_node_band>().member_set_coincident(i, coincident[i]);
  }

  return outer_band_node.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
