/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <memory>

#include "poly/tiling/hermes/axis.h"
#include "poly/tiling/hermes/model_graph.h"

namespace akg {
namespace ir {
namespace poly {
void Axis::SetGlobalAxisName() {
  for (auto &g_axis : ModelGraph::global_axis_vec_) {
    for (auto const &name_dim_range : ModelGraph::name_dim_range_set_) {
      if (g_axis.dim_axis_ == std::get<1>(name_dim_range) && g_axis.range_ == std::get<2>(name_dim_range)) {
        g_axis.name_ = std::get<0>(name_dim_range);
      }
    }
  }
}

/// \brief Get axis information from the schedule tree
/// \param[in] tree Schedule tree to inspect
/// \return Axis information for each statement
void ScheduleAxisInfo::GetAxisInfo(const isl::schedule &sch) {
  std::map<std::string, AxisInfo> result;

  // First extract ranges
  GetAxisRanges(sch.get(), &result);
  // Then extract axis mappings
  GetAxisMapping(sch.get(), &result);

  ModelGraph::name_dim_range_set_.clear();

  for (const auto &[statement, statement_info] : result) {
    (void)statement;
    for (const auto &[axis_name, axis_info] : statement_info.mapping) {
      auto dim_idx = axis_info.schedule_dim;
      int64_t range = (statement_info.ranges[dim_idx].ub - statement_info.ranges[dim_idx].lb + 1) *
                      statement_info.ranges[dim_idx].stride;
      auto dim = axis_info.schedule_dim - axis_info.offset;
      ModelGraph::InsertToNameDimRangeSet(axis_name, dim, range);
    }
  }
}

/// \brief Get axis ranges from a schedule tree
/// \param[in] tree Schedule tree
/// \param[in,out] result Output
/// \note Currently supported trees are:
///       - top level outer band
///       - top level sequence (or set) of outer bands
void ScheduleAxisInfo::GetAxisRanges(__isl_keep isl_schedule *const tree, std::map<std::string, AxisInfo> *result) {
  std::unique_ptr<isl_union_set, decltype(&isl_union_set_free)> domain(isl_schedule_get_domain(tree),
                                                                       isl_union_set_free);
  std::unique_ptr<isl_union_map, decltype(&isl_union_map_free)> schedule(isl_schedule_get_map(tree),
                                                                         isl_union_map_free);
  GetAxisRanges(domain.get(), schedule.get(), result);
}

/// \brief Get axis ranges from (possibly a subpart of) a domain and a schedule
/// \param[in] domain Domain
/// \param[in] schedule Schedule
/// \param[in,out] result Output
void ScheduleAxisInfo::GetAxisRanges(__isl_keep isl_union_set *const full_domain,
                                     __isl_keep isl_union_map *const schedule,
                                     std::map<std::string, AxisInfo> *result) {
  // We iterate over each statement separately to better handle unfused dimensions
  std::unique_ptr<isl_set_list, decltype(&isl_set_list_free)> slist(isl_union_set_get_set_list(full_domain),
                                                                    isl_set_list_free);
  const isl_size statement_count = isl_set_list_n_set(slist.get());
  for (int i = 0; i < statement_count; ++i) {
    isl_set *const dset = isl_set_list_get_at(slist.get(), i);
    std::unique_ptr<isl_union_set, decltype(&isl_union_set_free)> domain(isl_union_set_from_set(dset),
                                                                         isl_union_set_free);

    // 1. Apply the schedule to the domain
    std::unique_ptr<isl_union_set, decltype(&isl_union_set_free)> uset(
      isl_union_set_apply(isl_union_set_copy(domain.get()), isl_union_map_copy(schedule)), isl_union_set_free);
    const bool isa_set = isl_union_set_isa_set(uset.get()) == isl_bool_true;
    if (!isa_set) {
      std::cerr << "get_axis_ranges(): !isa_set()" << std::endl;
      isl_union_set_dump(full_domain);
      isl_union_set_dump(domain.get());
      isl_union_map_dump(schedule);
      isl_union_set_dump(uset.get());
      return;
    }

    // 2. Compute the lexmin and lexmax
    std::unique_ptr<isl_set, decltype(&isl_set_free)> set(isl_set_from_union_set(isl_union_set_copy(uset.get())),
                                                          isl_set_free);
    std::unique_ptr<isl_set, decltype(&isl_set_free)> lexmin(isl_set_lexmin(isl_set_copy(set.get())), isl_set_free);
    std::unique_ptr<isl_set, decltype(&isl_set_free)> lexmax(isl_set_lexmax(isl_set_copy(set.get())), isl_set_free);

    const bool singletons =
      isl_set_is_singleton(lexmin.get()) == isl_bool_true && isl_set_is_singleton(lexmax.get()) == isl_bool_true;
    if (!singletons) {
      std::cerr << "get_axis_ranges(): !is_singleton()" << std::endl;
      return;
    }

    // 3. Get the involved statements
    const std::vector<std::string> statements = GetTupleNames(domain.get());
    if (statements.size() != 1) {
      std::cerr << "get_axis_ranges(): unexpected statement names count" << std::endl;
    }

    const std::string &statement = statements[0];
    const isl_size dims = isl_set_dim(set.get(), isl_dim_set);

    // 4. Initialize the information for the statements we will initialize
    (*result)[statement].name = statement;
    (*result)[statement].ranges.resize(dims);

    for (int dim = 0; dim < dims; ++dim) {
      // Extract lower and upper bounds
      std::unique_ptr<isl_val, decltype(&isl_val_free)> lb_val(
        isl_set_plain_get_val_if_fixed(lexmin.get(), isl_dim_set, dim), isl_val_free);
      std::unique_ptr<isl_val, decltype(&isl_val_free)> ub_val(
        isl_set_plain_get_val_if_fixed(lexmax.get(), isl_dim_set, dim), isl_val_free);
      int64_t lb = isl_val_get_num_si(lb_val.get());
      int64_t ub = isl_val_get_num_si(ub_val.get());

      // Extract the stride
      std::unique_ptr<isl_stride_info, decltype(&isl_stride_info_free)> info(isl_set_get_stride_info(set.get(), dim),
                                                                             isl_stride_info_free);
      std::unique_ptr<isl_val, decltype(&isl_val_free)> stride_val(isl_stride_info_get_stride(info.get()),
                                                                   isl_val_free);
      int64_t stride = isl_val_get_num_si(stride_val.get());

      // Record the values for the current scheduling dimension
      (*result)[statement].ranges[dim].ub = ub;
      (*result)[statement].ranges[dim].lb = lb;
      (*result)[statement].ranges[dim].stride = stride;
    }
  }
}

/// \brief Get axis mappings for all statements involved in the schedule tree
/// \param[in] tree Schedule tree to inspect
/// \param[in,out] result Output
/// \pre \a result was already partially initialized in a previous call of GetAxisRanges()
void ScheduleAxisInfo::GetAxisMapping(__isl_keep isl_schedule *const tree, std::map<std::string, AxisInfo> *result) {
  std::unique_ptr<isl_union_map, decltype(&isl_union_map_free)> schedule(isl_schedule_get_map(tree),
                                                                         isl_union_map_free);
  GetAxisMapping(schedule.get(), result);
}

/// \brief Get axis mappings for all statements involved in the scheduling relations
/// \param[in] umap union map to inspect
/// \param[in,out] result Output
/// \pre \a result was already partially initialized in a previous call of GetAxisRanges()
void ScheduleAxisInfo::GetAxisMapping(__isl_keep isl_union_map *const umap, std::map<std::string, AxisInfo> *result) {
  std::unique_ptr<isl_map_list, decltype(&isl_map_list_free)> mlist(isl_union_map_get_map_list(umap),
                                                                    isl_map_list_free);
  const isl_size map_count = isl_map_list_n_map(mlist.get());
  for (int i = 0; i < map_count; ++i) {
    std::unique_ptr<isl_map, decltype(&isl_map_free)> map(isl_map_list_get_at(mlist.get(), i), isl_map_free);
    GetAxisMapping(map.get(), result);
  }
}

/// \brief Get axis mappings for a given statement from a scheduling relation
/// \param[in] map map to inspect
/// \param[in,out] result Output
/// \pre \a result was already partially initialized in a previous call of GetAxisRanges()
void ScheduleAxisInfo::GetAxisMapping(__isl_keep isl_map *const map, std::map<std::string, AxisInfo> *result) {
  std::unique_ptr<isl_basic_map_list, decltype(&isl_basic_map_list_free)> blist(isl_map_get_basic_map_list(map),
                                                                                isl_basic_map_list_free);
  const isl_size bmap_count = isl_basic_map_list_n_basic_map(blist.get());
  if (bmap_count != 1) {
    std::cerr << "GetAxisMapping(): unexpected map?!" << std::endl;
    isl_map_dump(map);
    return;
  }

  for (int i = 0; i < bmap_count; ++i) {
    std::unique_ptr<isl_basic_map, decltype(&isl_basic_map_free)> bmap(isl_basic_map_list_get_at(blist.get(), i),
                                                                       isl_basic_map_free);
    GetAxisMapping(bmap.get(), result);
  }
}

/// \brief Get axis mappings for a given statement from a scheduling relation
/// \param[in] bmap basic map to inspect
/// \param[in,out] result Output
/// \pre \a result was already partially initialized in a previous call of GetAxisRanges()
void ScheduleAxisInfo::GetAxisMapping(__isl_keep isl_basic_map *const bmap, std::map<std::string, AxisInfo> *result) {
  const std::string statement(isl_basic_map_get_tuple_name(bmap, isl_dim_in));
  std::unique_ptr<isl_mat, decltype(&isl_mat_free)> eq(
    isl_basic_map_equalities_matrix(bmap, isl_dim_in, isl_dim_out, isl_dim_div, isl_dim_param, isl_dim_cst),
    isl_mat_free);
  const isl_size in_dims = isl_basic_map_dim(bmap, isl_dim_in);
  const isl_size out_dims = isl_basic_map_dim(bmap, isl_dim_out);
  const isl_size rows = isl_mat_rows(eq.get());

  size_t previous_scalar_dims = 0;
  for (isl_size scheduling_dim = 0; scheduling_dim < out_dims; ++scheduling_dim) {
    const isl_size offset = in_dims + scheduling_dim;
    bool is_scalar_dim = true;
    for (isl_size row = 0; row < rows; ++row) {
      std::unique_ptr<isl_val, decltype(&isl_val_free)> scheduling_dim_coeff(
        isl_mat_get_element_val(eq.get(), row, offset), isl_val_free);
      if (isl_val_is_zero(scheduling_dim_coeff.get()) == isl_bool_true) {
        continue;
      }

      for (isl_size statement_dim = 0; statement_dim < in_dims; ++statement_dim) {
        std::unique_ptr<isl_val, decltype(&isl_val_free)> statement_dim_coeff(
          isl_mat_get_element_val(eq.get(), row, statement_dim), isl_val_free);
        if (isl_val_is_zero(statement_dim_coeff.get()) == isl_bool_true) {
          continue;
        }

        const char *dim_name = isl_basic_map_get_dim_name(bmap, isl_dim_in, statement_dim);
        const std::string name(dim_name);
        const AxisDim axis_dim = {name, static_cast<int>(statement_dim), static_cast<int>(scheduling_dim),
                                  static_cast<int>(previous_scalar_dims)};
        (*result)[statement].mapping.try_emplace(name, axis_dim);

        is_scalar_dim = false;
      }
    }

    if (is_scalar_dim) {
      ++previous_scalar_dims;
    }
  }
}

// Misc. isl utilities
std::vector<std::string> ScheduleAxisInfo::GetTupleNames(__isl_keep isl_union_set *const uset) {
  std::vector<std::string> result;

  std::unique_ptr<isl_set_list, decltype(&isl_set_list_free)> slist(isl_union_set_get_set_list(uset),
                                                                    isl_set_list_free);
  const isl_size count = isl_set_list_n_set(slist.get());
  for (int i = 0; i < count; ++i) {
    std::unique_ptr<isl_set, decltype(&isl_set_free)> set(isl_set_list_get_at(slist.get(), i), isl_set_free);
    const char *const name = isl_set_get_tuple_name(set.get());
    result.emplace_back(name);
  }

  return result;
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
