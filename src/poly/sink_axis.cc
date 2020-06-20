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

#include "poly/sink_axis.h"

#include <tvm/ir_visitor.h>
#include <tvm/operation.h>
#include <isl/constraint.h>

#include <climits>
#include <fstream>
#include <queue>
#include <cmath>

#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {

bool FindC0Schedule(const isl::pw_aff_list &paList) {
  for (unsigned int upaIdx = 0; upaIdx < paList.size(); ++upaIdx) {
    isl::pw_aff pa = paList.get_at(upaIdx);
    int64_t inDimSize = isl_pw_aff_dim(pa.get(), isl_dim_in);
    CHECK_NE(inDimSize, -1);
    const char *lastInDim = isl_pw_aff_get_dim_name(pa.get(), isl_dim_in, inDimSize - 1);
    if (lastInDim == nullptr) {
      continue;
    }
    std::string lastAxis = lastInDim;
    // pw_aff { S_4[n, c1, kh, oh, c0] -> [(n)] }
    // to do use isl api to mark schedule axis
    std::string pwAffStr = pa.to_str();
    std::size_t arrowPos = pwAffStr.find("->");
    if (arrowPos == std::string::npos) {
      continue;
    }
    std::string rang = pwAffStr.substr(arrowPos + 2, pwAffStr.size() - (arrowPos + 2));
    std::size_t leftBracket = rang.find("(");
    std::size_t rightBracket = rang.find(")");
    if ((leftBracket == std::string::npos) || (rightBracket == std::string::npos) ||
        (rightBracket <= leftBracket + 1)) {
      continue;
    }
    std::string scheduleAxis = rang.substr(leftBracket + 1, rightBracket - leftBracket - 1);
    if (lastAxis == scheduleAxis) {
      // lastIdxSchedule[i] = true;
      // findC0Schedule = true;
      // break;
      return true;
    }
  }
  return false;
}

void ExchangeCoincident(std::vector<int> &coincident, const isl::schedule_node &node,
                        const std::unordered_map<int, bool> lastIdxSchedule, const int &n) {
  // save coincident value for this band
  std::vector<int> coincidentOld;
  for (int i = 0; i < n; ++i) {
    coincidentOld.push_back(node.as<isl::schedule_node_band>().member_get_coincident(i));
  }

  // exchange last axis coincident to last position
  for (int i = 0; i < n; ++i) {
    if (lastIdxSchedule.count(i) > 0) {
      continue;
    }
    coincident.push_back(coincidentOld[i]);
  }

  for (auto item : lastIdxSchedule) {
    CHECK_GE(item.first, 0) << "index of coincident can not be negative: " << item.first;
    coincident.push_back(coincidentOld[item.first]);
  }
}

/* *****************************************************
 * Initialization part:
 * get partial_schedule info and union_pw_aff_list from band node
 * partial_schedule is a multi_union_pw_aff as follows:
 * [
    { S_4[n, c1, kh, oh, c0] -> [(n)]; S_3[n, c1, oh, ow, c0] -> [(n)]; S_5[n, c1, kh, oh, ow, c0] -> [(n)]; S_6[n,
c1, kh, kw, oh, ow, c0] -> [(n)] }, { S_4[n, c1, kh, oh, c0] -> [(c1)]; S_3[n, c1, oh, ow, c0] -> [(c1)]; S_5[n, c1, kh,
oh, ow, c0] -> [(c1)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c1)] }, { S_4[n, c1, kh, oh, c0] -> [(oh)]; S_3[n, c1, oh,
ow, c0] -> [(oh)]; S_5[n, c1, kh, oh, ow, c0] -> [(oh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(oh)] }, { S_4[n, c1, kh,
oh, c0] -> [(0)]; S_3[n, c1, oh, ow, c0] -> [(ow)]; S_5[n, c1, kh, oh, ow, c0] -> [(1 + ow)]; S_6[n, c1, kh, kw, oh, ow,
c0] -> [(ow)] }, { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] ->
[(c0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c0)] }, { S_4[n, c1, kh, oh, c0] -> [(kh)]; S_3[n, c1, oh, ow, c0] -> [(0)];
S_5[n, c1, kh, oh, ow, c0] -> [(kh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(kh)] }, { S_4[n, c1, kh, oh, c0] -> [(0)];
S_3[n, c1, oh, ow, c0] -> [(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(-kw)] }
   ]
 * Is union_pw_aff_list(upal) the other form of multi_union_pw_aff ? and it can not print in LOG(INFO)
 * but we need it during update, at least we make a new multi_union_pw_aff from union_pw_aff_list
 * and add it to the band node, shown in the following pseudo-code
 * isl::union_pw_aff_list upal = isl::union_pw_aff_list();
 * ... ...
 * update strategy of upal ...
 * ... ...
 * isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(partial_schedule.get_space(), upal);
 * node = node.del();
 * node = node.insert_partial_schedule(mupa);
 *
 * The update strategy of SinkC0 is moving the schedule of axis of C0 with every statement
 * to the end of the multi_union_pw_aff, the purpose result is shown in the following:
 *
[
{ S_4[n, c1, kh, oh, c0] -> [(n)]; S_3[n, c1, oh, ow, c0] -> [(n)]; S_5[n, c1, kh, oh, ow, c0] -> [(n)]; S_6[n, c1, kh,
kw, oh, ow, c0] -> [(n)] }, { S_4[n, c1, kh, oh, c0] -> [(c1)]; S_3[n, c1, oh, ow, c0] -> [(c1)]; S_5[n, c1, kh, oh, ow,
c0] -> [(c1)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c1)] }, { S_4[n, c1, kh, oh, c0] -> [(oh)]; S_3[n, c1, oh, ow, c0] ->
[(oh)]; S_5[n, c1, kh, oh, ow, c0] -> [(oh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(oh)] }, { S_4[n, c1, kh, oh, c0] ->
[(0)]; S_3[n, c1, oh, ow, c0] -> [(ow)]; S_5[n, c1, kh, oh, ow, c0] -> [(1 + ow)]; S_6[n, c1, kh, kw, oh, ow, c0] ->
[(ow)] }, del { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] ->
[(c0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(c0)] }, |  { S_4[n, c1, kh, oh, c0] -> [(kh)]; S_3[n, c1, oh, ow, c0] ->
[(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(kh)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(kh)] }, v  { S_4[n, c1, kh, oh, c0] ->
[(0)]; S_3[n, c1, oh, ow, c0] -> [(0)]; S_5[n, c1, kh, oh, ow, c0] -> [(0)]; S_6[n, c1, kh, kw, oh, ow, c0] -> [(-kw)] }
add { S_4[n, c1, kh, oh, c0] -> [(c0)]; S_3[n, c1, oh, ow, c0] -> [(c0)]; S_5[n, c1, kh, oh, ow, c0] -> [(c0)]; S_6[n,
c1, kh, kw, oh, ow, c0] -> [(c0)] },
]
 * This strategy is designed for Davinci architecture, for its five dimension data format.
 * We suppose two steps to achieve this strategy:
 * 1. find the last axis C0 schedule in the multi_union_pw_aff
 * 2. if find this schedule, move it to the end of the multi_union_pw_aff
 * 3. add the updated multi_union_pw_aff to the band node
 * *****************************************************/
isl::schedule_node Transform::SinkC0Schedule(isl::schedule_node &node) {
  if (!node.isa<isl::schedule_node_band>()) {
    return node;
  }
  auto schedule = node.as<isl::schedule_node_band>().get_partial_schedule();
  isl::union_pw_aff_list upal = isl::union_pw_aff_list();
  std::unordered_map<int, bool> lastIdxSchedule;

  // make new union pw aff list
  for (unsigned int i = 0; i < schedule.size(); ++i) {
    isl::union_pw_aff upa = schedule.get_union_pw_aff(i);
    isl::pw_aff_list paList = upa.get_pw_aff_list();
    bool findC0Schedule = FindC0Schedule(paList);
    if (findC0Schedule) {
      lastIdxSchedule[i] = true;
      continue;
    }
    if (upal.is_null()) {
      upal = isl::union_pw_aff_list(upa);
    } else {
      upal = upal.add(upa);
    }
  }

  // save permutable value for this band
  int permutable = node.as<isl::schedule_node_band>().get_permutable();
  if (!lastIdxSchedule.empty() && permutable == 1) {
    for (auto idx : lastIdxSchedule) {
      isl::union_pw_aff upa = schedule.get_union_pw_aff(idx.first);
      if (upal.is_null()) {
        upal = isl::union_pw_aff_list(upa);
      } else {
        upal = upal.add(upa);
      }
    }
  } else {
    return node;
  }

  std::vector<int> coincident;
  int n = node.as<isl::schedule_node_band>().n_member();
  ExchangeCoincident(coincident, node, lastIdxSchedule, n);

  // make multi_union_pw_aff
  isl::multi_union_pw_aff mupa = isl::multi_union_pw_aff(schedule.get_space(), upal);

  // delete old node
  node = node.del();

  // insert new node
  node = node.insert_partial_schedule(mupa);
  node = node.as<isl::schedule_node_band>().set_permutable(permutable);
  for (int i = 0; i < n; ++i) {
    node = node.as<isl::schedule_node_band>().member_set_coincident(i, coincident[i]);
  }
  return node;
}

isl::schedule Transform::SinkC0(const isl::schedule &sch) {
  auto fn = [&, this](isl::schedule_node node) -> isl::schedule_node {
    if (node.isa<isl::schedule_node_band>()) {
      node = SinkC0Schedule(node);
    }
    return node;
  };

  return sch.get_root().map_descendant_bottom_up(fn).get_schedule();
}

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

/*
 * Try to sink the last axis of outer band to the leaves of the schedule tree.
 *
 * The criteria that the last axis can be sinked:
 * 1) the axis is the last axis in the outer band schedule.
 * 2) the axis is the last axis in the domain of each statement.
 * 3) all dependencies of the last axis are equality constraints. (i.e. S_1[c0] -> S_2[c0' = c0])
 * 4) all dependencies of the last axis do not appear in other non-last axes.
 * 5) the domain of the last axis is not larger than a threshold (otherwise it still should be tiled).
 *
 * sinkLastAxis will:
 * 1) remove the C0 axis from the outer band schedule, and
 * 2) add a partial schedule (C0) to each leaf filter node that contains the last axis.
 */
isl::schedule Transform::SinkLastAxis(const isl::schedule &sch) {
  auto outer_band_node = sch.get_root();
  while (true) {
    if (outer_band_node.isa<isl::schedule_node_band>()) {
      outer_band_node = SinkLastAxisFromBand(outer_band_node, dependences_);
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
          outer_band_node = SinkLastAxisFromBand(outer_band_node, dependences_);
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
