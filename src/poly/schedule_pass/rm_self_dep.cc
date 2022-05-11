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

#include "rm_self_dep.h"

#include <tvm/ir_visitor.h>
#include <isl/constraint.h>

#include <fstream>
#include <queue>
#include <cmath>

#include "poly/schedule_pass.h"
#include "poly/dump_log.h"

namespace akg {
namespace ir {
namespace poly {

using ReferenceReduceInfo = std::unordered_map<std::string, ReferenceAxisInfo>;

using EqualityVars = std::vector<std::pair<std::string, std::string>>;

using EqualityVarsMap = std::unordered_map<std::string, std::string>;

/* key: is reduce statement
** value: reduce tensor name*/
using ReduceOp = std::pair<bool, std::string>;

/* key: is reduce statement
** value: reduce axis number*/
using ReduceAxisInfo = std::pair<bool, int>;

static bool IsConstantValNonZero(__isl_keep isl_constraint *c) {
  auto val_ptr = isl_constraint_get_constant_val(c);
  if (isl_val_get_num_si(val_ptr) != 0) {
    static_cast<void>(isl_val_free(val_ptr));
    return true;
  }
  static_cast<void>(isl_val_free(val_ptr));
  return false;
}

static void ExtractEqualityVarsInConstraint(__isl_keep isl_constraint *c, EqualityVars *equality_var_map) {
  CHECK(equality_var_map != nullptr);

  if (!isl_constraint_is_equality(c)) return;
  if (IsConstantValNonZero(c)) return;

  int n_dim_in = isl_constraint_dim(c, isl_dim_in);
  int n_dim_out = isl_constraint_dim(c, isl_dim_out);
  bool domain_found = false;
  bool range_found = false;
  const char *domain_name = nullptr;
  const char *range_name = nullptr;
  bool domain_is_positive = false;
  bool range_is_positive = false;

  for (int i = 0; i < n_dim_in; ++i) {
    auto coef_val = isl_constraint_get_coefficient_val(c, isl_dim_in, i);
    int64_t coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef == 0) {
      continue;
    } else if (coef == 1 || coef == -1) {
      if (domain_found) return;
      domain_found = true;
      domain_is_positive = (coef == 1);
      domain_name = isl_constraint_get_dim_name(c, isl_dim_in, i);
    } else {
      return;
    }
  }

  for (int i = 0; i < n_dim_out; ++i) {
    auto coef_val = isl_constraint_get_coefficient_val(c, isl_dim_out, i);
    int64_t coef = isl_val_get_num_si(coef_val);
    static_cast<void>(isl_val_free(coef_val));
    if (coef == 0) {
      continue;
    } else if (coef == 1 || coef == -1) {
      if (range_found) return;
      range_found = true;
      range_is_positive = (coef == 1);
      range_name = isl_constraint_get_dim_name(c, isl_dim_out, i);
    } else {
      return;
    }
  }

  bool is_one_positive_one_negative =
    (!domain_is_positive && range_is_positive) || (domain_is_positive && !range_is_positive);
  if (domain_found && range_found && is_one_positive_one_negative) {
    CHECK(domain_name != nullptr);
    CHECK(range_name != nullptr);
    equality_var_map->push_back(std::make_pair(domain_name, range_name));
  }
}

static isl_stat ExtractEqualityForeachConstraint(__isl_take isl_constraint *c, void *user) {
  auto equality_var_map = reinterpret_cast<EqualityVars *>(user);
  ExtractEqualityVarsInConstraint(c, equality_var_map);
  static_cast<void>(isl_constraint_free(c));
  return isl_stat_ok;
}

static EqualityVarsMap ExtractEqualityVarsMap(const isl::map &m, bool is_domain_to_range) {
  EqualityVars equality_var_pairs;
  EqualityVarsMap equality_var_map;
  m.foreach_basic_map([&](const isl::basic_map &basic_map) -> void {
    isl_stat status = isl_basic_map_foreach_constraint(basic_map.get(), ExtractEqualityForeachConstraint,
                                                       reinterpret_cast<void *>(&equality_var_pairs));
    CHECK(status == isl_stat_ok);
  });

  for (const auto &pair : equality_var_pairs) {
    if (is_domain_to_range) {
      equality_var_map[pair.first] = pair.second;
    } else {
      equality_var_map[pair.second] = pair.first;
    }
  }
  return equality_var_map;
}

static void ExtractReduceInfoFromRange(const isl::map &m, isl::local_space &domain_space, isl::local_space &range_space,
                                       ReduceInfo &reduce_info, const isl::id &ref_tuple_id) {
  EqualityVarsMap equality_vars_range_to_domain = ExtractEqualityVarsMap(m, false);
  bool is_ref_tuple_in_range = (ref_tuple_id == m.get_tuple_id(isl_dim_out));

  int num_range_dim = isl_local_space_dim(range_space.get(), isl_dim_out);
  for (int dim = 0; dim < num_range_dim; ++dim) {
    isl::id range_axis = isl::manage(isl_local_space_get_dim_id(range_space.get(), isl_dim_out, dim));
    std::string range_axis_name = range_axis.get_name();
    std::string domain_axis_name = range_axis_name;
    if (equality_vars_range_to_domain.count(range_axis_name)) {
      domain_axis_name = equality_vars_range_to_domain[range_axis_name];
    }

    AxisInfo axis_info;
    axis_info.iter = is_ref_tuple_in_range ? range_axis_name : domain_axis_name;

    bool is_outer_axis = false;
    int domain_space_dim = isl_local_space_find_dim_by_name(domain_space.get(), isl_dim_out, domain_axis_name.c_str());
    if (domain_space_dim != -1) {
      is_outer_axis = true;
      isl::aff get_dim_in_domain = isl::aff::var_on_domain(domain_space, isl_dim_out, domain_space_dim);
      axis_info.domain_min = static_cast<int>(m.domain().min_val(get_dim_in_domain).get_num_si());
      axis_info.domain_max = static_cast<int>(m.domain().max_val(get_dim_in_domain).get_num_si());
    }

    isl::aff get_dim_in_range = isl::aff::var_on_domain(range_space, isl_dim_out, dim);
    axis_info.range_min = static_cast<int>(m.range().min_val(get_dim_in_range).get_num_si());
    axis_info.range_max = static_cast<int>(m.range().max_val(get_dim_in_range).get_num_si());

    if (is_outer_axis) {
      reduce_info.outer_axis.push_back(axis_info);
    } else {
      reduce_info.reduce_axis.push_back(axis_info);
    }
  }
}

static void ExtractReduceInfoFromDomain(const isl::map &m, isl::local_space &domain_space,
                                        isl::local_space &range_space, ReduceInfo &reduce_info,
                                        const isl::id &ref_tuple_id) {
  EqualityVarsMap equality_vars_domain_to_range = ExtractEqualityVarsMap(m, true);
  bool is_ref_tuple_in_range = (ref_tuple_id == m.get_tuple_id(isl_dim_out));

  int num_domain_dim = isl_local_space_dim(domain_space.get(), isl_dim_out);
  for (int dim = 0; dim < num_domain_dim; ++dim) {
    const char *domain_axis_name = isl_local_space_get_dim_name(domain_space.get(), isl_dim_out, dim);
    if (domain_axis_name == nullptr) continue;
    const char *range_axis_name = domain_axis_name;
    if (equality_vars_domain_to_range.count(domain_axis_name)) {
      range_axis_name = equality_vars_domain_to_range[domain_axis_name].c_str();
    }

    int range_space_dim = isl_local_space_find_dim_by_name(range_space.get(), isl_dim_out, range_axis_name);
    if (range_space_dim == -1) {
      AxisInfo axis_info;
      axis_info.iter = is_ref_tuple_in_range ? range_axis_name : domain_axis_name;

      isl::aff get_dim_in_domain = isl::aff::var_on_domain(domain_space, isl_dim_out, dim);
      axis_info.domain_min = static_cast<int>(m.domain().min_val(get_dim_in_domain).get_num_si());
      axis_info.domain_max = static_cast<int>(m.domain().max_val(get_dim_in_domain).get_num_si());

      reduce_info.broadcast_axis.push_back(axis_info);
    }
  }
}

static ReduceInfo ExtractReduceInfo(const isl::map &m, const isl::id &ref_tuple_id) {
  CHECK_EQ(m.n_basic_map(), 1);
  isl::space domain_space_obj = m.domain().get_space();
  isl::space range_space_obj = m.range().get_space();
  // isl_local_space_from_space will manage the isl_space* parameter,
  // so we must copy the isl_space* to avoid double free
  isl::local_space domain_space = isl::manage(isl_local_space_from_space(isl_space_copy(domain_space_obj.get())));
  isl::local_space range_space = isl::manage(isl_local_space_from_space(isl_space_copy(range_space_obj.get())));

  ReduceInfo reduce_info;
  ExtractReduceInfoFromRange(m, domain_space, range_space, reduce_info, ref_tuple_id);
  ExtractReduceInfoFromDomain(m, domain_space, range_space, reduce_info, ref_tuple_id);
  return reduce_info;
}

static std::ostream &operator<<(std::ostream &os, const AxisInfo &axis) {
  os << axis.iter << ": domain=[" << axis.domain_min << ", " << axis.domain_max << "], "
     << "range=[" << axis.range_min << ", " << axis.range_max << "]";
  return os;
}

static std::ostream &operator<<(std::ostream &os, const ReduceInfo &info) {
  for (const auto &axis : info.outer_axis) {
    os << "OuterAxis " << axis << std::endl;
  }
  for (const auto &axis : info.reduce_axis) {
    os << "ReduceAxis " << axis << std::endl;
  }
  for (const auto &axis : info.broadcast_axis) {
    os << "BroadcastAxis " << axis << std::endl;
  }
  return os;
}

static std::ostream &operator<<(std::ostream &os, const ReferenceAxisInfo &axis) {
  os << "type=";
  if (axis.is_axis_type_known) {
    os << (axis.is_reduce_axis ? "reduce" : "outer");
  } else {
    os << "unknown";
  }
  os << ", min=";
  if (axis.min_defined) {
    os << axis.min;
  } else {
    os << "unknown";
  }
  os << ", max=";
  if (axis.max_defined) {
    os << axis.max;
  } else {
    os << "unknown";
  }
  return os;
}

static std::ostream &operator<<(std::ostream &os, const ReferenceReduceInfo &reduce_info) {
  for (const auto &axis : reduce_info) {
    os << axis.first << ": " << axis.second << std::endl;
  }
  return os;
}

static bool CompareAndUpdateReferenceReduceInfo(ReferenceReduceInfo &reference, const ReferenceReduceInfo &to_check) {
  if (reference.empty()) {
    reference = to_check;
    return true;
  }

  for (const auto &axis : to_check) {
    auto iter = axis.first;
    ReferenceAxisInfo &ref_axis = reference[iter];
    const ReferenceAxisInfo &to_check_axis = axis.second;
    if (to_check_axis.is_axis_type_known) {
      if (ref_axis.is_axis_type_known && ref_axis.is_reduce_axis != to_check_axis.is_reduce_axis) return false;
      ref_axis.is_axis_type_known = true;
      ref_axis.is_reduce_axis = to_check_axis.is_reduce_axis;
    }
    if (to_check_axis.min_defined) {
      if (!ref_axis.min_defined || ref_axis.min < to_check_axis.min) ref_axis.min = to_check_axis.min;
      ref_axis.min_defined = true;
    }
    if (to_check_axis.max_defined) {
      if (!ref_axis.max_defined || ref_axis.max < to_check_axis.max) ref_axis.max = to_check_axis.max;
      ref_axis.max_defined = true;
    }
  }
  return true;
}

/*
 *    S_x[o1, o2] -> S_1[o1, o2, min1, min2, min3]
 * or S_x[o3, o4] -> S_1[o1, o2, min1, min2, min3]
 *
 * Check the min value of initialization.
 */
static bool CheckInitializationDependence(ReferenceReduceInfo &reference, const ReduceInfo &to_check) {
  ReferenceReduceInfo to_check_ref;
  for (const auto &axis : to_check.reduce_axis) {
    if (axis.range_min != axis.range_max) return false;

    ReferenceAxisInfo ref;
    ref.is_axis_type_known = true;
    ref.is_reduce_axis = true;
    ref.min_defined = true;
    ref.min = axis.range_min;
    ref.max_defined = false;
    to_check_ref.emplace(axis.iter, ref);
  }

  for (const auto &axis : to_check.outer_axis) {
    if (axis.domain_min != axis.range_min) continue;
    if (axis.domain_max != axis.range_max) continue;

    ReferenceAxisInfo ref;
    ref.is_axis_type_known = true;
    ref.is_reduce_axis = false;
    ref.min_defined = true;
    ref.min = axis.range_min;
    ref.max_defined = false;
    to_check_ref.emplace(axis.iter, ref);
  }

  return CompareAndUpdateReferenceReduceInfo(reference, to_check_ref);
}

/*
 *  S_y[o1, o2, i1, i2, i3] -> S_1[o1, o2, i1, i2, i3]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2, min3 <= i3 <= max3
 *
 *  Do not check any axis because the dependency may be partial,
 *  e.g. S_y[i1] = select(i1 < 100, S_1[i1], 0);
 */
static bool CheckReducedTensorDependence(ReferenceReduceInfo &reference, const ReduceInfo &to_check) {
  ReferenceReduceInfo to_check_ref;
  for (const auto &axis : to_check.outer_axis) {
    ReferenceAxisInfo ref;
    ref.is_axis_type_known = false;
    ref.min_defined = false;
    ref.max_defined = false;
    to_check_ref.emplace(axis.iter, ref);
  }

  return CompareAndUpdateReferenceReduceInfo(reference, to_check_ref);
}

/*
 *    isl_basic_map may be one of the following:
 *
 *    S_1[o1, o2, i1, i2, i3] -> S_1[o1, o2, i1, i2, i3 + 1]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2, min3 <= i3 <= max3 - 1
 *    S_1[o1, o2, i1, i2, max3] -> S_1[o1, o2, i1, i2 + 1, min3]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2 - 1
 *    S_1[o1, o2, i1, max2, max3] -> S_1[o1, o2 + 1, i1, min2, min3]:
 *          min1 <= i1 <= max1 - 1
 *
 *    Check min and max, and at least one axis is the reduce axis.
 */
static bool CheckSelfDependence(ReferenceReduceInfo &reference, const ReduceInfo &to_check) {
  ReferenceReduceInfo to_check_ref;
  if (!to_check.broadcast_axis.empty()) return false;
  if (!to_check.reduce_axis.empty()) return false;

  bool reduce_axis_found = false;
  for (const auto &axis : to_check.outer_axis) {
    ReferenceAxisInfo ref;
    ref.is_axis_type_known = false;

    // type I: S_1[i1, ...] -> S_1[i1, ...]: min1 <= i1 <= max1
    if (axis.domain_min == axis.range_min && axis.domain_max == axis.range_max) {
      ref.max = axis.domain_max;
      ref.min = axis.domain_min;
    } else if (axis.domain_min + 1 == axis.range_min && axis.domain_max + 1 == axis.range_max) {
      // type II: S_1[..., i2, ...] -> S_1[..., i2 + 1, ...]: min2 <= i2 <= max2 - 1
      // only one type II (reduce) axis is allowed
      if (reduce_axis_found) return false;
      reduce_axis_found = true;
      ref.max = axis.range_max;
      ref.min = axis.domain_min;
      ref.is_axis_type_known = true;
      ref.is_reduce_axis = true;
    } else if (axis.domain_min == axis.domain_max && axis.range_min == axis.range_max) {
      // type III: S_1[..., max3] -> S_1[..., min3]
      ref.max = axis.domain_max;
      ref.min = axis.range_min;
      ref.is_axis_type_known = true;
      ref.is_reduce_axis = true;
    } else {
      return false;
    }

    ref.min_defined = true;
    ref.max_defined = true;
    to_check_ref.emplace(axis.iter, ref);
  }
  if (!reduce_axis_found) return false;

  return CompareAndUpdateReferenceReduceInfo(reference, to_check_ref);
}

/*
 *    S_1[o1, o2, max1, max2, max3] -> S_z[o1, o2]
 * or S_1[o1, o2, max1, max2, max3] -> S_z[o3, o4]
 *
 * Check the max value of getting reduce result.
 */
static bool CheckGetReduceResultDependence(ReferenceReduceInfo &reference, const ReduceInfo &to_check) {
  ReferenceReduceInfo to_check_ref;
  for (const auto &axis : to_check.outer_axis) {
    ReferenceAxisInfo ref;
    ref.is_axis_type_known = false;
    ref.min_defined = false;
    ref.max_defined = false;
    to_check_ref.emplace(axis.iter, ref);
  }

  for (const auto &axis : to_check.broadcast_axis) {
    ReferenceAxisInfo ref;
    if (axis.domain_min != axis.domain_max) {
      ref.is_axis_type_known = false;
      ref.min_defined = false;
      ref.max_defined = false;
      to_check_ref.emplace(axis.iter, ref);
    } else {
      ref.is_axis_type_known = true;
      ref.is_reduce_axis = true;
      ref.max_defined = true;
      ref.max = axis.domain_max;
      ref.min_defined = false;
      to_check_ref.emplace(axis.iter, ref);
    }
  }

  return CompareAndUpdateReferenceReduceInfo(reference, to_check_ref);
}

static std::vector<std::string> GetReduceAxisList(const ReferenceReduceInfo &ref_reduce_info) {
  std::vector<std::string> list;
  for (const auto &axis : ref_reduce_info) {
    if (axis.second.is_axis_type_known && axis.second.is_reduce_axis) {
      list.push_back(axis.first);
    }
  }
  return list;
}

static bool IsDependenceFromOtherStmt(const isl::map &m, const isl::id &tuple_id) {
  return m.range().get_tuple_id() == tuple_id && m.domain().get_tuple_id() != tuple_id;
}

static bool IsSelfDependence(const isl::map &m, const isl::id &tuple_id) {
  return m.range().get_tuple_id() == tuple_id && m.domain().get_tuple_id() == tuple_id;
}

static bool IsDependenceToOtherStmt(const isl::map &m, const isl::id &tuple_id) {
  return m.range().get_tuple_id() != tuple_id && m.domain().get_tuple_id() == tuple_id;
}

// extract the common elements of two vectors
// assume no duplicate in "a" and no duplicate in "b".
std::vector<std::string> ExtractCommonAxis(const std::vector<AxisInfo> &a, const std::vector<AxisInfo> &b) {
  std::vector<std::string> common;
  for (const auto &sa : a) {
    for (const auto &sb : b) {
      if (sa.iter == sb.iter) common.push_back(sa.iter);
    }
  }
  return common;
}

/* Check two accesses form a reduce op, and find the reduce axis list.
 *
 * Example:
 * { S_2[ax4, n0_n0_k0, n1_n2_k2, n2_n3_k3] -> reduce_tensor[arg0 = 0, arg1 = 0, arg2 = 0, arg3 = 0, arg4 = ax4] :
 *   0 <= ax4 <= 15 and 0 <= n0_n0_k0 <= 2047 and 0 <= n1_n2_k2 <= 4095 and 0 <= n2_n3_k3 <= 1 }
 *
 * ReduceInfo:
 * OuterAxis ax4: domain=[0, 15], range=[0, 15]
 * ReduceAxis arg0: domain=[0, 0], range=[0, 0]
 * ReduceAxis arg1: domain=[0, 0], range=[0, 0]
 * ReduceAxis arg2: domain=[0, 0], range=[0, 0]
 * ReduceAxis arg3: domain=[0, 0], range=[0, 0]
 * BroadcastAxis n0_n0_k0: domain=[0, 2047], range=[0, 0]
 * BroadcastAxis n1_n2_k2: domain=[0, 4095], range=[0, 0]
 * BroadcastAxis n2_n3_k3: domain=[0, 1], range=[0, 0]
 *
 * { S_2[ax4, n0_n0_k0, n1_n2_k2, n2_n3_k3] -> other_read_tensor[arg0 = n0_n0_k0, arg1 = 0, arg2 = n1_n2_k2,
 *   arg3 = n2_n3_k3, arg4 = ax4] : 0 <= ax4 <= 15 and 0 <= n0_n0_k0 <= 2047 and 0 <= n1_n2_k2 <= 4095
 *   and 0 <= n2_n3_k3 <= 1 }
 *
 * ReduceInfo:
 * OuterAxis n0_n0_k0: domain=[0, 2047], range=[0, 2047]
 * OuterAxis n1_n2_k2: domain=[0, 4095], range=[0, 4095]
 * OuterAxis n2_n3_k3: domain=[0, 1], range=[0, 1]
 * OuterAxis ax4: domain=[0, 15], range=[0, 15]
 * ReduceAxis arg1: domain=[0, 0], range=[0, 0]
 *
 * Reduce axes are: n0_n0_k0, n1_n2_k2, n2_n3_k3.
 */
bool CheckReduceAxis(const ReduceInfo &reduce_info, const ReduceInfo &other_info,
                     const std::vector<std::string> &reduce_axis_list) {
  if (reduce_axis_list.empty()) return false;
  for (const auto &reduce_axis : reduce_axis_list) {
    AxisInfo reduce_axis_info, other_axis_info;
    for (const auto &axis : reduce_info.broadcast_axis) {
      if (axis.iter == reduce_axis) reduce_axis_info = axis;
    }
    for (const auto &axis : other_info.outer_axis) {
      if (axis.iter == reduce_axis) other_axis_info = axis;
    }
    if (reduce_axis_info.range_max != reduce_axis_info.range_min) return false;
    if (reduce_axis_info.domain_max != other_axis_info.domain_max) return false;
    if (reduce_axis_info.domain_min != other_axis_info.domain_min) return false;
    if (other_axis_info.domain_max != other_axis_info.range_max) return false;
    if (other_axis_info.domain_min != other_axis_info.range_min) return false;
  }

  return true;
}

bool FindReduceAxis(const isl::map &reduce_access, const isl::map &other_access,
                    std::vector<std::string> &reduce_axis_list) {
  auto reduce_info = ExtractReduceInfo(reduce_access, reduce_access.get_tuple_id(isl_dim_in));
  auto other_info = ExtractReduceInfo(other_access, other_access.get_tuple_id(isl_dim_in));
  reduce_axis_list = ExtractCommonAxis(reduce_info.broadcast_axis, other_info.outer_axis);
  bool is_reduce_axis = CheckReduceAxis(reduce_info, other_info, reduce_axis_list);
  if (!is_reduce_axis) {
    LOG(INFO) << "Accesses of self dependence do not appear to be a reduce op, will check dependences later. "
              << "Access of reduce tensor: " << reduce_access;
    LOG(INFO) << "Access of the other tensor: " << other_access;
  }
  return is_reduce_axis;
}

/*
 * Criteria for S_1 to be a reduce op:
 *
 * 1. initialization:
 *    S_x[o1, o2] -> S_1[o1, o2, min1, min2, min3]
 *    (Initialization can be omitted when initialization is a const value.)
 *
 * 2. reduced tensor:
 *    S_y[o1, o2, i1, i2, i3] -> S_1[o1, o2, i1, i2, i3]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2, min3 <= i3 <= max3
 *    (This dependency can be omitted when reduced tensor is an input tensor.)
 *
 * 3. reduce self dependence (reduce axis one by one):
 *    S_1[o1, o2, i1, i2, i3] -> S_1[o1, o2, i1, i2, i3 + 1]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2, min3 <= i3 <= max3 - 1
 *    S_1[o1, o2, i1, i2, max3] -> S_1[o1, o2, i1, i2 + 1, min3]:
 *          min1 <= i1 <= max1, min2 <= i2 <= max2 - 1
 *    S_1[o1, o2, i1, max2, max3] -> S_1[o1, o2 + 1, i1, min2, min3]:
 *          min1 <= i1 <= max1 - 1
 *
 *    In these reduce statements, o1 and o2 must have identical domains.
 *
 * 4. get the reduce result:
 *    S_1[o1, o2, max1, max2, max3] -> S_z[o1, o2]
 *    i.e. Other statements can only depend on [max1, max2, max3].
 */
static bool CheckIsStmtReduceOp(const isl::union_map &dependences, const isl::id &tuple_id,
                                std::vector<std::string> &reduce_axis) {
  bool foundSelfDependence = false;
  bool foundInvalid = false;
  ReferenceReduceInfo ref_reduce_info;
  dependences.foreach_map([&](const isl::map &dependence_map) -> void {
    dependence_map.foreach_basic_map([&](const isl::basic_map &basic_map) -> void {
      if (foundInvalid) return;
      const isl::map m = basic_map;
      ReduceInfo reduce_info = ExtractReduceInfo(m, tuple_id);

      if (IsDependenceFromOtherStmt(m, tuple_id)) {
        bool isInitializationStmt = !reduce_info.reduce_axis.empty();
        if (isInitializationStmt) {
          if (!CheckInitializationDependence(ref_reduce_info, reduce_info)) foundInvalid = true;
        } else {
          if (!CheckReducedTensorDependence(ref_reduce_info, reduce_info)) foundInvalid = true;
        }
      } else if (IsSelfDependence(m, tuple_id)) {
        foundSelfDependence = true;
        if (!CheckSelfDependence(ref_reduce_info, reduce_info)) foundInvalid = true;
      } else if (IsDependenceToOtherStmt(m, tuple_id)) {
        if (!CheckGetReduceResultDependence(ref_reduce_info, reduce_info)) foundInvalid = true;
      }

      if (foundInvalid) {
        LOG(INFO) << "self dependence cannot be removed: " << FormatMupaStr(dependence_map);
        LOG(INFO) << "reduce info: " << reduce_info;
        LOG(INFO) << "reference reduce info: " << ref_reduce_info;
      }
    });
  });
  reduce_axis = GetReduceAxisList(ref_reduce_info);
  return foundSelfDependence && !foundInvalid;
}

/* Check whether stmt "tuple_id" is a reduce op according to read/write accesses.
 * A reduce stmt is in the form reduce_tensor[reduce_args] = reduce_tensor[reduce_args] + other_tensor[other_args].
 */
ReduceOp CheckIsStmtReduceOp(const isl::union_map &reads, const isl::union_map &writes, const isl::id &tuple_id,
                             std::vector<std::string> &reduce_axis_list) {
  ReduceOp false_op = std::make_pair(false, "");
  std::unordered_map<isl::id, isl::map, isl::IslIdIslHash> read_tensors, write_tensors;
  reads.foreach_map([&](const isl::map &map) -> void {
    if (map.domain().unwrap().get_tuple_id(isl_dim_in) == tuple_id) {
      read_tensors[map.get_tuple_id(isl_dim_out)] = map.domain_factor_domain();
    }
  });
  writes.foreach_map([&](const isl::map &map) -> void {
    if (map.domain().unwrap().get_tuple_id(isl_dim_in) == tuple_id) {
      write_tensors[map.get_tuple_id(isl_dim_out)] = map.domain_factor_domain();
    }
  });
  if (write_tensors.size() != 1) return false_op;
  // there may be more than 2 read tensors due to wrapped MAD stmt.
  if (read_tensors.size() < 2) return false_op;

  const isl::map &reduce_write_access = write_tensors.begin()->second;
  const isl::id &reduce_tensor = write_tensors.begin()->first;
  if (read_tensors.count(reduce_tensor) == 0) return false_op;
  const isl::map &reduce_read_access = read_tensors.at(reduce_tensor);
  if (!reduce_write_access.is_equal(reduce_read_access)) return false_op;

  // check that each other_tensor matches the reduce axis criteria and the reduce axes are the same
  std::vector<std::string> last_reduce_axis_list;
  for (const auto &it : read_tensors) {
    if (it.first != reduce_tensor) {
      const isl::map &other_read_access = it.second;
      if (!FindReduceAxis(reduce_read_access, other_read_access, reduce_axis_list)) return false_op;

      if (!last_reduce_axis_list.empty() && reduce_axis_list != last_reduce_axis_list) return false_op;
      last_reduce_axis_list = reduce_axis_list;
    }
  }
  return std::make_pair(true, reduce_tensor.get_name());
}

static ReduceAxisInfo IsMultiAxisSelfDependence(const isl::union_map &dependences, const isl::id &tuple_id) {
  ReduceAxisInfo found_multi_axis_self_dependence = std::make_pair(false, 0);
  dependences.foreach_map([&](const isl::map &m) -> void {
    if (m.domain().get_tuple_id() == m.range().get_tuple_id() && m.domain().get_tuple_id() == tuple_id) {
      found_multi_axis_self_dependence = std::make_pair(true, m.n_basic_map());
    }
  });
  return found_multi_axis_self_dependence;
}

/*
 * Removes self dependence of (multi-axis) reduce operations.
 */
isl::union_map RemoveReduceOpSelfDependence(ScopInfo &scop_info, PassInfo &pass_info) {
  isl::union_map preserved_dependences = isl::union_map::empty(pass_info.dependences_.get_space());
  /* *********************************************
   * key:   the reduce stmt id
   * value: reduce axis no. of this reduce stmt, if the no. >= 1, it is the reduce statement
   * *********************************************/
  std::unordered_map<std::string, int> is_tuple_reduce_op;
  pass_info.dependences_.foreach_map(
    [&scop_info, &pass_info, &preserved_dependences, &is_tuple_reduce_op](const isl::map &m) -> void {
      if (m.domain().get_tuple_id() != m.range().get_tuple_id()) {
        preserved_dependences = preserved_dependences.add_map(m);
      } else {  // self dependence
        isl::id tuple_id = m.domain().get_tuple_id();
        std::string tuple_id_key = tuple_id.get_name();
        if (is_tuple_reduce_op.count(tuple_id_key) == 0) {
          std::vector<std::string> reduce_axis_list;
          ReduceOp res = std::make_pair(false, "");
          ReduceAxisInfo reduce_axis_info = IsMultiAxisSelfDependence(pass_info.dependences_, tuple_id);
          is_tuple_reduce_op[tuple_id_key] = reduce_axis_info.second;
          if (reduce_axis_info.first) {
            res = CheckIsStmtReduceOp(scop_info.analysis_result_.GetReads(), scop_info.analysis_result_.GetWrites(),
                                      tuple_id, reduce_axis_list);
            if (!(res.first || CheckIsStmtReduceOp(pass_info.dependences_, tuple_id, reduce_axis_list))) {
              is_tuple_reduce_op[tuple_id_key] = 0;
            }
          }

          if (is_tuple_reduce_op[tuple_id_key] >= 2) {
            ReduceTensorInfo reduce_tensor_info;
            reduce_tensor_info.axis_vec = reduce_axis_list;
            reduce_tensor_info.stmt_map = isl::union_map::empty(isl::space(scop_info.ctx_, 0));
            scop_info.analysis_result_.RecordReduceTensorInfoMap(tuple_id, reduce_tensor_info);
          }

          /***************************************************
           * New flow of atomic add optimization on poly npu
           * will store the reduce tensor info for npu isl emitter.
           ****************************************************/
          if (is_tuple_reduce_op[tuple_id_key] >= 1 && scop_info.user_config_.GetEnableAtomicAdd() &&
              !res.second.empty()) {
            bool is_global = false;
            for (auto it : scop_info.user_config_.GetOriginBind()) {
              if (it.second->data->name_hint == res.second) {
                is_global = true;
                break;
              }
            }
            if (is_global) {
              scop_info.analysis_result_.RecordReduceOutTensors(res.second);
              scop_info.analysis_result_.RecordReduceOutStmtIdToTensor(tuple_id_key, res.second);
            }
          }
        }

        // for reduce axis number is smaller than one, keep the dependences relation
        if (is_tuple_reduce_op[tuple_id_key] <= 1) {
          preserved_dependences = preserved_dependences.add_map(m);
        }
      }
    });
  return preserved_dependences;
}

/*
 * Removes all self dependences in the program. Use with special care.
 * If tensor_name_map is not empty, only the self-dependency of tensor in tensor_name_map is deleted.
 */
isl::union_map RemoveSelfDependence(PassInfo &pass_info, std::map<std::string, std::string> tensor_name_map) {
  isl::union_map preserved = isl::union_map::empty(pass_info.dependences_.get_space());
  isl::union_map removed = isl::union_map::empty(pass_info.dependences_.get_space());
  pass_info.dependences_.foreach_map([&preserved, &removed, tensor_name_map](const isl::map &m) -> void {
    auto domian_id = m.domain().get_tuple_id();
    if (domian_id != m.range().get_tuple_id()) {
      preserved = preserved.add_map(m);
    } else {
      if (!tensor_name_map.empty() && tensor_name_map.count(domian_id.get_name()) == 0) {
        preserved = preserved.add_map(m);
      } else {
        removed = removed.add_map(m);
      }
    }
  });
  if (!removed.is_empty()) LOG(INFO) << "force remove self dependence: " << removed;
  return preserved;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
