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
#include "sink_last_axis.h"

#include <isl/constraint.h>

namespace akg {
namespace ir {
namespace poly {

/*
 * Check whether the domain of the last axis is not larger than a threshold,
 * so that large last axes can still be tiled.
 */
static bool CheckLastAxisDomainSize(const isl::schedule_node_band &outer_band, const int dim_size_threshold) {
  // if the domain size of last axis is larger than the following threshold, it will not be sinked.
  bool is_valid = true;
  outer_band.domain().foreach_set([&](const isl::set &set) -> void {
    unsigned int dim = set.n_dim();
    if (dim == 0) return;
    auto simple_hull = set.simple_hull();
    auto dim_size = simple_hull.dim_max_val(dim - 1).get_num_si();
    if (dim_size > dim_size_threshold) is_valid = false;
  });
  return is_valid;
}

/*
 * Find the last axis in the outer band schedule, and check that
 * it is the last axis in the domain of each statement.
 *
 * e.g. { S_0[i0, i1, i2, i3, i4] -> [(i4)]; S_1[n, c1, h, iw, c0] -> [(c0)]; }
 */
static bool CheckLastAxisInOuterSchedule(const isl::schedule_node_band &outer_band) {
  auto outer_schedule = outer_band.get_partial_schedule();
  unsigned int num_outer_axes = outer_band.n_member();
  // if the outer band contains only one axis, do not sink it
  if (num_outer_axes <= 1) return false;

  auto last_axis_map = outer_schedule.get_at(num_outer_axes - 1);
  bool found_invalid = false;
  last_axis_map.get_pw_aff_list().foreach([&found_invalid](const isl::pw_aff &pw_aff) -> void {
    if (!pw_aff.isa_aff()) {
      found_invalid = true;
      return;
    }
    auto aff = pw_aff.as_aff();
    auto const_val = isl_aff_get_constant_val(aff.get());
    if (isl_val_get_num_si(const_val) != 0) found_invalid = true;
    static_cast<void>(isl_val_free(const_val));
    int num_dims = isl_aff_dim(aff.get(), isl_dim_in);
    for (int i = 0; i < num_dims; ++i) {
      auto coef_val = isl_aff_get_coefficient_val(aff.get(), isl_dim_in, i);
      if (i == num_dims - 1) {
        if (isl_val_get_num_si(coef_val) != 1) found_invalid = true;
      } else {
        if (isl_val_get_num_si(coef_val) != 0) found_invalid = true;
      }
      static_cast<void>(isl_val_free(coef_val));
    }
  });
  return !found_invalid;
}

static bool CheckLastAxisInConstaintEx(__isl_keep isl_constraint *c) {
  int n_dim_in = isl_constraint_dim(c, isl_dim_in);
  int n_dim_out = isl_constraint_dim(c, isl_dim_out);
  if (n_dim_in <= 0 || n_dim_out <= 0) return true;

  auto domain_last_coef_val = isl_constraint_get_coefficient_val(c, isl_dim_in, n_dim_in - 1);
  auto domain_last_coef = isl_val_get_num_si(domain_last_coef_val);
  static_cast<void>(isl_val_free(domain_last_coef_val));

  auto range_last_coef_val = isl_constraint_get_coefficient_val(c, isl_dim_out, n_dim_out - 1);
  auto range_last_coef = isl_val_get_num_si(range_last_coef_val);
  static_cast<void>(isl_val_free(range_last_coef_val));

  // the constraint does not involve the last dim
  if (domain_last_coef == 0 && range_last_coef == 0) return true;

  // check that other coefficients are zero
  for (auto i = 0; i < n_dim_in - 1; ++i) {
    auto coef_val = isl_constraint_get_coefficient_val(c, isl_dim_in, i);
    auto coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef != 0) return false;
  }
  for (auto i = 0; i < n_dim_out - 1; ++i) {
    auto coef_val = isl_constraint_get_coefficient_val(c, isl_dim_out, i);
    auto coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef != 0) return false;
  }

  // inequality constraint
  if (!isl_constraint_is_equality(c)) {
    // check that the inequality only involves domain or range
    return (domain_last_coef == 0 || range_last_coef == 0);
  }
  // check that the equality is in the form of c0 - c0' = 0
  if (domain_last_coef + range_last_coef != 0) return false;

  // check constant is zero
  auto constant_val = isl_constraint_get_constant_val(c);
  auto constant = isl_val_get_num_si(constant_val);
  static_cast<void>(isl_val_free(constant_val));
  if (constant != 0) return false;

  return true;
}

static isl_stat CheckLastAxisInConstraint(__isl_take isl_constraint *c, void *data) {
  auto found_invalid = reinterpret_cast<bool *>(data);
  CHECK(found_invalid != nullptr);
  if (!CheckLastAxisInConstaintEx(c)) *found_invalid = true;
  static_cast<void>(isl_constraint_free(c));
  return isl_stat_ok;
}

static std::unordered_set<isl::id, isl::IslIdIslHash> GetStatementsInBand(const isl::schedule_node &outer_band_node) {
  CHECK(outer_band_node.isa<isl::schedule_node_band>());
  auto outer_band = outer_band_node.as<isl::schedule_node_band>();
  auto partial_schedule = outer_band.get_partial_schedule_union_map();
  std::unordered_set<isl::id, isl::IslIdIslHash> statements_in_band;
  partial_schedule.foreach_map(
    [&](const isl::map &map) -> void { statements_in_band.insert(map.get_tuple_id(isl_dim_in)); });
  return statements_in_band;
}

/*
 * Check that all dependencies of the last axis are equality constraints
 * and all dependencies of the last axis do not appear in other non-last axes.
 *
 * e.g. S_1[n, c1, h, iw, c0] -> S_3[n' = n, c1' = c1, h' = h, iw' = iw, c0' = c0] :
 *      0 <= n <= 1 and 0 <= c1 <= 15 and 0 <= h <= 19 and 0 <= iw <= 23 and 0 <= c0 <= 15
 */
static bool CheckLastAxisDependence(const isl::schedule_node &outer_band_node, const isl::union_map &dependence) {
  auto statements_in_band = GetStatementsInBand(outer_band_node);
  bool found_invalid = false;
  dependence.foreach_map([&](const isl::map &map) -> void {
    if (statements_in_band.count(map.get_tuple_id(isl_dim_in)) == 0) return;
    if (statements_in_band.count(map.get_tuple_id(isl_dim_out)) == 0) return;
    map.foreach_basic_map([&](const isl::basic_map &bmap) -> void {
      static_cast<void>(isl_basic_map_foreach_constraint(bmap.get(), CheckLastAxisInConstraint,
                                                         reinterpret_cast<void *>(&found_invalid)));
    });
  });
  return !found_invalid;
}

/*
 * 1) Remove the C0 axis from the outer band schedule, and
 * 2) Add a partial schedule (C0) to each leaf filter node that contains the last axis.
 */
static isl::schedule_node MoveLastAxisToInnermostBand(const isl::schedule_node_band &outer_band) {
  unsigned int num_outer_axes = outer_band.n_member();
  CHECK_GE(num_outer_axes, 1);

  auto new_outer_band = outer_band.split(num_outer_axes - 1);
  auto last_axis_node = new_outer_band.child(0);
  CHECK(last_axis_node.isa<isl::schedule_node_band>());

  // We cannot use last_axis_node.sink() because the sink() method has incorrect return type.
  // The method returns isl::schedule_node_band, but the sinked node may not be a band node.
  isl::schedule_node sinked_node = isl::manage(isl_schedule_node_band_sink(last_axis_node.copy()));
  CHECK(!sinked_node.is_null());

  auto sinked_outer_band = sinked_node.parent();
  CHECK(sinked_outer_band.isa<isl::schedule_node_band>());
  return sinked_outer_band;
}

static isl::schedule_node SinkLastAxisFromBand(const isl::schedule_node &outer_band_node,
                                               const isl::union_map &dependences) {
  if (!outer_band_node.isa<isl::schedule_node_band>()) return outer_band_node;
  auto outer_band = outer_band_node.as<isl::schedule_node_band>();
  if (!CheckLastAxisInOuterSchedule(outer_band)) return outer_band_node;
  if (!CheckLastAxisDependence(outer_band_node, dependences)) return outer_band_node;

  const int dim_size_threshold = 16;
  if (!CheckLastAxisDomainSize(outer_band, dim_size_threshold)) {
    LOG(INFO) << "The last axis of outer band can be sinked, but it is larger than threshold " << dim_size_threshold
              << " , so it is not sinked.";
    return outer_band_node;
  }

  LOG(INFO) << "The last axis of outer band will be sinked to the innermost band.";
  return MoveLastAxisToInnermostBand(outer_band);
}

isl::schedule SinkLastAxis::Run(isl::schedule sch) {
  auto outer_band_node = sch.get_root();
  while (true) {
    if (outer_band_node.isa<isl::schedule_node_band>()) {
      outer_band_node = SinkLastAxisFromBand(outer_band_node, pass_info_.dependences_);
      break;
    }
    unsigned int n_children = outer_band_node.n_children();
    if (n_children == 0) {
      break;
    } else if (n_children == 1) {
      outer_band_node = outer_band_node.child(0);
      continue;
    } else if (outer_band_node.child(0).isa<isl::schedule_node_filter>()) {
      for (unsigned int i = 0; i < n_children; ++i) {
        if (outer_band_node.child(i).n_children() == 0) continue;
        outer_band_node = outer_band_node.child(i).child(0);
        if (outer_band_node.isa<isl::schedule_node_band>()) {
          outer_band_node = SinkLastAxisFromBand(outer_band_node, pass_info_.dependences_);
        }
        outer_band_node = outer_band_node.parent().parent();
      }
      break;
    } else {
      break;
    }
  }
  return outer_band_node.get_schedule();
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
