/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "poly/isl_util.h"

#include "poly/log_util.h"

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// local types
////////////////////////////////////////////////////////////////////////////////

enum isl_schedule_node_fine_adjustment_type {
  FINE_ADJUSTMENT_NONE = 0,
  FINE_ADJUSTMENT_MOD,
  FINE_ADJUSTMENT_SCALE,
  FINE_ADJUSTMENT_SCALE_DOWN,
};

struct isl_schedule_node_fine_adjustment_data {
  isl_union_pw_aff *result;
  const char *name;
  isl_val *value;
  enum isl_schedule_node_fine_adjustment_type type;
};

////////////////////////////////////////////////////////////////////////////////
// local declarations
////////////////////////////////////////////////////////////////////////////////

static inline __isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_collapse(__isl_take isl_multi_union_pw_aff *aff,
                                                                                 int dim, __isl_take isl_val *val);
static inline __isl_give isl_val *isl_schedule_node_band_find_collapse_coeff(__isl_take isl_schedule_node *band,
                                                                             int dim);
static inline isl_stat isl_pw_aff_fine_adjustment(__isl_take isl_pw_aff *pa, void *user);

static inline isl::multi_union_pw_aff isl_multi_union_pw_aff_collapse(const isl::multi_union_pw_aff &aff, int dim,
                                                                      const isl::val &value) {
  return isl::manage(isl_multi_union_pw_aff_collapse(aff.copy(), dim, value.copy()));
}
static inline isl::val isl_schedule_node_band_find_collapse_coeff(const isl::schedule_node_band &band, int dim) {
  return isl::manage(isl_schedule_node_band_find_collapse_coeff(band.copy(), dim));
}

////////////////////////////////////////////////////////////////////////////////
// C++ wrappers for unexposed isl functions
////////////////////////////////////////////////////////////////////////////////

bool isl_aff_is_cst(const isl::aff &a) {
  isl_aff *const internal = a.get();
  return isl_aff_is_cst(internal);
}

std::string isl_set_get_dim_name(const isl::set &s, enum isl_dim_type type, unsigned int pos) {
  const isl::id &id = isl_set_get_dim_id(s, type, pos);
  return id.name();
}

isl::id isl_set_get_dim_id(const isl::set &s, enum isl_dim_type type, unsigned int pos) {
  isl_id *const id = isl_set_get_dim_id(s.get(), type, pos);
  return isl::manage(id);
}

int isl_set_find_dim_by_id(const isl::set &s, enum isl_dim_type type, const isl::id &id) {
  return isl_set_find_dim_by_id(s.get(), type, id.get());
}

int isl_set_find_dim_by_name(const isl::set &s, enum isl_dim_type type, const std::string &name) {
  return isl_set_find_dim_by_name(s.get(), type, name.c_str());
}

unsigned isl_set_dim(const isl::set &s, enum isl_dim_type type) {
  isl_set *const internal = s.get();
  const isl_size size = isl_set_dim(internal, type);
  return size;
}

long isl_set_plain_get_num_si(const isl::set &s, unsigned int pos) {
  const isl::val &value = isl_set_plain_get_val_if_fixed(s, isl_dim_set, pos);
  const long result = value.get_num_si();
  return result;
}

isl::val isl_set_plain_get_val_if_fixed(const isl::set &s, enum isl_dim_type type, unsigned int pos) {
  isl_set *const internal = s.get();
  isl_val *const value = isl_set_plain_get_val_if_fixed(internal, type, pos);
  return isl::manage(value);
}

isl::set isl_set_move_dims(const isl::set &s, enum isl_dim_type dst_type, unsigned int dst_pos,
                           enum isl_dim_type src_type, unsigned int src_pos, unsigned int n) {
  isl_set *const internal = isl_set_move_dims(s.copy(), dst_type, dst_pos, src_type, src_pos, n);
  return isl::manage(internal);
}

std::string isl_map_get_dim_name(const isl::map &m, enum isl_dim_type type, unsigned int pos) {
  const isl::id &id = isl_map_get_dim_id(m, type, pos);
  return id.name();
}

isl::id isl_map_get_dim_id(const isl::map &m, enum isl_dim_type type, unsigned int pos) {
  isl_map *const internal = m.get();
  isl_id *const id = isl_map_get_dim_id(internal, type, pos);
  return isl::manage(id);
}

bool isl_map_involves_dims(const isl::map &m, enum isl_dim_type type, unsigned int first, unsigned n) {
  isl_map *const internal = m.get();
  return isl_map_involves_dims(internal, type, first, n);
}

isl::map isl_map_drop_constraints_not_involving_dims(const isl::map &m, enum isl_dim_type type, unsigned int first,
                                                     unsigned n) {
  isl_map *const internal = isl_map_copy(m.get());
  isl_map *const drop = isl_map_drop_constraints_not_involving_dims(internal, type, first, n);
  return isl::manage(drop);
}

isl::union_map isl_union_map_align_params(const isl::union_map &map, const isl::space &space) {
  isl_union_map *const aligned = isl_union_map_align_params(map.copy(), space.copy());
  return isl::manage(aligned);
}

isl::union_pw_aff_list isl_union_pw_aff_list_insert(const isl::union_pw_aff_list &list, unsigned int pos,
                                                    const isl::union_pw_aff &aff) {
  isl_union_pw_aff *const element = aff.copy();

  isl_union_pw_aff_list *result = list.copy();
  result = isl_union_pw_aff_list_insert(result, pos, element);

  return isl::manage(result);
}

isl::space isl_space_set_alloc(isl::ctx ctx, unsigned int nparam, unsigned int dim) {
  isl_space *const internal = isl_space_set_alloc(ctx.get(), nparam, dim);
  return isl::manage(internal);
}

isl::id isl_space_get_dim_id(const isl::space &space, enum isl_dim_type type, unsigned int pos) {
  isl_space *const internal = space.get();
  isl_id *const id = isl_space_get_dim_id(internal, type, pos);
  return isl::manage(id);
}

isl::space isl_space_set_dim_id(const isl::space &space, enum isl_dim_type type, unsigned int pos, const isl::id &id) {
  isl_id *const internal_id = id.copy();
  isl_space *internal_space = space.copy();
  internal_space = isl_space_set_dim_id(internal_space, type, pos, internal_id);
  return isl::manage(internal_space);
}

////////////////////////////////////////////////////////////////////////////////
// Misc.
////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> isl_set_dim_names(const isl::set &set, enum isl_dim_type type) {
  std::vector<std::string> result;

  const unsigned int count = isl_set_dim(set, type);
  for (unsigned int i = 0; i < count; ++i) {
    const std::string &current = isl_set_get_dim_name(set, type, i);
    result.push_back(current);
  }

  return result;
}

std::vector<std::string> isl_set_all_names(const isl::set &set) {
  std::vector<std::string> result;

  const unsigned int set_dimensions = isl_set_dim(set, isl_dim_set);
  for (unsigned int i = 0; i < set_dimensions; ++i) {
    const std::string &current = isl_set_get_dim_name(set, isl_dim_set, i);
    result.push_back(current);
  }
  const unsigned int parameters = isl_set_dim(set, isl_dim_param);
  for (unsigned int i = 0; i < parameters; ++i) {
    const std::string &current = isl_set_get_dim_name(set, isl_dim_param, i);
    result.push_back(current);
  }

  return result;
}

std::vector<int> isl_set_lexmax_extract_upper_bounds(const isl::set &set, const std::vector<std::string> &names) {
  std::vector<int> result;

  const std::size_t count = names.size();
  for (std::size_t i = 0; i < count; ++i) {
    const int position = isl_set_find_dim_by_name(set, isl_dim_set, names[i]);
    if (position >= 0) {
      // The input set is supposed to be a lexmax
      const isl::val &lexmax = isl_set_plain_get_val_if_fixed(set, isl_dim_set, position);
      const isl::val &upper_bound = lexmax.add(1);
      const int value = static_cast<int>(upper_bound.get_num_si());
      result.push_back(value);
    }
  }

  return result;
}

std::vector<int> isl_set_lexmax_extract_upper_bounds(const isl::set &s, const std::vector<int> &dimensions) {
  std::vector<int> result;

  const int size = static_cast<int>(isl_set_dim(s, isl_dim_set));
  for (auto dimension : dimensions) {
    if (dimension < size) {
      // We need to add 1 because the input set is a lexmax
      const isl::val &lexmax = isl_set_plain_get_val_if_fixed(s, isl_dim_set, dimension);
      const isl::val &upper_bound = lexmax.add(1);
      const int value = static_cast<int>(upper_bound.get_num_si());
      result.push_back(value);
    } else {
      log::Warn("cannot retrieve size for dimension " + std::to_string(dimension));
    }
  }

  return result;
}

isl::space isl_space_copy_param_names(const isl::space &space, const isl::space &source) {
  isl::space result = space;

  const unsigned int params = source.dim(isl_dim_param);
  const unsigned int limit = std::min(params, result.dim(isl_dim_param));
  if (params > limit) {
    log::Warn("destination space is smaller than source space");
  }

  for (unsigned int i = 0; i < limit; ++i) {
    const isl::id &name = isl_space_get_dim_id(source, isl_dim_param, i);
    result = isl_space_set_dim_id(result, isl_dim_param, i, name);
  }

  return result;
}

isl::space isl_space_set_cat(const isl::space &left, const isl::space &right) {
  const unsigned int params = left.dim(isl_dim_param);
  const unsigned int out = left.dim(isl_dim_out) + right.dim(isl_dim_out);

  // No constructor that takes both isl_dim_params and isl_dim_out sizes in the C++ wrapper!
  isl_ctx *const ctx = left.ctx().get();
  isl_space *const result = isl_space_set_alloc(ctx, params, out);
  // Note: we currently do not need to extract dim names if they were named in the input spaces.

  return isl::manage(result);
}

isl::multi_union_pw_aff isl_multi_union_pw_aff_cat(const isl::multi_union_pw_aff &left,
                                                   const isl::multi_union_pw_aff &right) {
  const unsigned int left_count = left.size();
  const unsigned int right_count = right.size();

  if (!left_count) {
    return right;
  } else if (!right_count) {
    return left;
  }

  const isl::union_pw_aff_list &left_list = left.union_pw_aff_list();
  const isl::union_pw_aff_list &right_list = right.union_pw_aff_list();
  const isl::union_pw_aff_list &list = left_list.concat(right_list);

  const isl::space &left_space = left.space();
  const isl::space &right_space = right.space();
  const isl::space &space = isl_space_set_cat(left_space, right_space);

  const isl::multi_union_pw_aff &result = isl::multi_union_pw_aff(space, list);
  return result;
}

__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_cat(__isl_take isl_multi_union_pw_aff *left,
                                                              __isl_take isl_multi_union_pw_aff *right) {
  const isl::multi_union_pw_aff &wrapped = isl_multi_union_pw_aff_cat(isl::manage(left), isl::manage(right));
  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_copy(wrapped.get());
  return result;
}

isl::multi_union_pw_aff isl_multi_union_pw_aff_insert(const isl::multi_union_pw_aff &aff, unsigned pos,
                                                      const isl::union_pw_aff &el) {
  const isl::space &initial_space = aff.space();
  const isl::space &space = initial_space.add_dims(isl_dim_out, 1);

  const isl::union_pw_aff_list &initial_list = aff.union_pw_aff_list();
  const isl::union_pw_aff_list &list = isl_union_pw_aff_list_insert(initial_list, pos, el);

  const isl::multi_union_pw_aff &result = isl::multi_union_pw_aff(space, list);
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node utilities
////////////////////////////////////////////////////////////////////////////////

bool isl_schedule_node_is_band(const isl::schedule_node &node) { return node.isa<isl::schedule_node_band>(); }

bool isl_schedule_node_is_sequence(const isl::schedule_node &node) { return node.isa<isl::schedule_node_sequence>(); }

bool isl_schedule_node_has_single_child(const isl::schedule_node &node) { return node.n_children() == 1; }

bool isl_schedule_node_band_can_unsplit(const isl::schedule_node_band &band) {
  return isl_schedule_node_has_single_child(band) && band.child(0).isa<isl::schedule_node_band>();
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band utilities
////////////////////////////////////////////////////////////////////////////////

std::vector<bool> isl_schedule_node_band_get_coincidence(const isl::schedule_node_band &band) {
  std::vector<bool> result;

  const unsigned int members = band.n_member();
  if (!members) {
    return result;
  }

  for (unsigned int i = 0; i < members; ++i) {
    const bool coincident = band.member_get_coincident(static_cast<int>(i));
    result.push_back(coincident);
  }

  return result;
}

isl::schedule_node_band isl_schedule_node_band_set_coincidence(const isl::schedule_node_band &band,
                                                               const std::vector<bool> &coincidence) {
  const std::size_t members = static_cast<size_t>(band.n_member());
  const std::size_t size = coincidence.size();
  const std::size_t limit = std::min(members, size);
  if (members != size) {
    log::Warn("band size differs from coincidence vector!");
  }

  isl::schedule_node_band result = band;
  for (std::size_t i = 0; i < limit; ++i) {
    const int pos = static_cast<int>(i);
    const bool coincident = coincidence[i];
    result = result.member_set_coincident(pos, coincident);
  }

  return result;
}

isl::schedule_node_band isl_schedule_node_band_member_copy_properties(const isl::schedule_node_band &band, int pos,
                                                                      const isl::schedule_node_band &wrapped_original) {
  isl_schedule_node *const original = wrapped_original.get();
  const isl_bool coincident = isl_schedule_node_band_member_get_coincident(original, pos);
  const enum isl_ast_loop_type loop_type = isl_schedule_node_band_member_get_ast_loop_type(original, pos);
  const enum isl_ast_loop_type isolate_type = isl_schedule_node_band_member_get_isolate_ast_loop_type(original, pos);

  isl_schedule_node *internal = isl_schedule_node_copy(band.get());
  internal = isl_schedule_node_band_member_set_coincident(internal, pos, coincident);
  internal = isl_schedule_node_band_member_set_ast_loop_type(internal, pos, loop_type);
  internal = isl_schedule_node_band_member_set_isolate_ast_loop_type(internal, pos, isolate_type);

  const isl::schedule_node &node = isl::manage(internal);
  return node.as<isl::schedule_node_band>();
}

isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node &node,
                                                               const isl::schedule_node &original) {
  return isl_schedule_node_band_copy_properties(node.as<isl::schedule_node_band>(),
                                                original.as<isl::schedule_node_band>());
}

isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node &node,
                                                               const isl::schedule_node_band &original) {
  return isl_schedule_node_band_copy_properties(node.as<isl::schedule_node_band>(), original);
}

isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node_band &band,
                                                               const isl::schedule_node &original) {
  return isl_schedule_node_band_copy_properties(band, original.as<isl::schedule_node_band>());
}

isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node_band &band,
                                                               const isl::schedule_node_band &original) {
  isl::schedule_node_band result = band;

  const bool permutable = original.permutable();
  const isl::union_set &ast_build_options = original.ast_build_options();
  result = result.set_permutable(permutable);
  result = result.set_ast_build_options(ast_build_options);

  const int limit = static_cast<int>(std::min(band.n_member(), original.n_member()));
  for (int i = 0; i < limit; ++i) {
    result = isl_schedule_node_band_member_copy_properties(result, i, original);
  }

  return result;
}

isl::schedule_node_band isl_schedule_node_band_replace_partial_schedule(const isl::schedule_node &node,
                                                                        const isl::multi_union_pw_aff &partial,
                                                                        bool keep_properties) {
  return isl_schedule_node_band_replace_partial_schedule(node.as<isl::schedule_node_band>(), partial, keep_properties);
}

isl::schedule_node_band isl_schedule_node_band_replace_partial_schedule(const isl::schedule_node_band &band,
                                                                        const isl::multi_union_pw_aff &partial,
                                                                        bool keep_properties) {
  // The new schedule will be inserted above the current band
  isl::schedule_node_band result = band.insert_partial_schedule(partial).as<isl::schedule_node_band>();
  if (keep_properties) {
    const isl::schedule_node_band &original = result.child(0).as<isl::schedule_node_band>();
    result = isl_schedule_node_band_copy_properties(result, original);
  }
  // Do not forget to delete the previous band node and then move back to the "new" current position!
  result = result.child(0).del().parent().as<isl::schedule_node_band>();
  return result;
}

isl::set isl_schedule_node_band_lexmax(const isl::schedule_node &node) {
  return isl_schedule_node_band_lexmax(node.as<isl::schedule_node_band>());
}

isl::set isl_schedule_node_band_lexmax(const isl::schedule_node_band &band) {
  const isl::union_set &domain = band.domain();
  const isl::union_map &schedule = band.partial_schedule_union_map();
  const isl::union_set &application = domain.apply(schedule);
  const isl::union_set &lexmax = application.lexmax();
  return isl::set::from_union_set(lexmax);
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule_node_band isl_schedule_node_band_interchange(const isl::schedule_node_band &band, unsigned int first,
                                                           unsigned int second) {
  isl::multi_union_pw_aff partial = band.get_partial_schedule();
  const unsigned int dims = partial.size();
  if (first >= dims || second >= dims) {
    log::Warn(std::string(__func__) + ": target dimension out of bounds");
    return band;
  }

  const isl::union_pw_aff &aff_1 = partial.at(first);
  const isl::union_pw_aff &aff_2 = partial.at(second);
  partial = partial.set_at(first, aff_2);
  partial = partial.set_at(second, aff_1);
  // Save the properties
  const bool permutable = band.permutable();
  std::vector<bool> coincidence = isl_schedule_node_band_get_coincidence(band);
  // Interchange the coincidences
  const bool tmp = coincidence[first];
  coincidence[first] = coincidence[second];
  coincidence[second] = tmp;

  isl::schedule_node_band result = isl_schedule_node_band_replace_partial_schedule(band, partial, false);
  result = result.set_permutable(permutable);
  result = isl_schedule_node_band_set_coincidence(band, coincidence);

  return result;
}

isl::schedule_node_band isl_schedule_node_band_stripmine(const isl::schedule_node_band &band, unsigned int dim,
                                                         int value) {
  isl::ctx ctx = band.ctx();
  const isl::val &val = isl::val(ctx, value);
  return isl_schedule_node_band_stripmine(band, dim, val);
}

isl::schedule_node_band isl_schedule_node_band_stripmine(const isl::schedule_node_band &band, unsigned int dim,
                                                         const isl::val &value) {
  const unsigned int members = band.n_member();
  if (dim >= members) {
    log::Warn(std::string(__func__) + ": cannot stripmine out of bounds dimension");
    return band;
  }

  isl::multi_union_pw_aff schedule = band.partial_schedule();
  const isl::union_pw_aff &div = schedule.at(dim).scale_down(value);
  const isl::union_pw_aff &mod = schedule.at(dim).mod(value);

  schedule = schedule.set_at(dim, div);
  schedule = isl_multi_union_pw_aff_insert(schedule, dim + 1, mod);

  const bool permutable = band.permutable();
  std::vector<bool> coincidence = isl_schedule_node_band_get_coincidence(band);
  auto position = coincidence.begin() + dim + 1;
  coincidence.insert(position, coincidence[dim]);

  isl::schedule_node_band result = isl_schedule_node_band_replace_partial_schedule(band, schedule, true);
  result = result.set_permutable(permutable);
  result = isl_schedule_node_band_set_coincidence(result, coincidence);

  return result;
}

isl::schedule_node_band isl_schedule_node_band_collapse(const isl::schedule_node_band &band, unsigned int dim) {
  const unsigned int members = band.n_member();
  if (dim >= members) {
    return band;
  }

  const isl::val &coeff = isl_schedule_node_band_find_collapse_coeff(band, dim);
  isl::multi_union_pw_aff partial = band.partial_schedule();
  partial = isl_multi_union_pw_aff_collapse(partial, dim, coeff);

  const bool permutable = band.permutable();
  std::vector<bool> coincidence = isl_schedule_node_band_get_coincidence(band);
  const bool collapsed_coincidence = coincidence[dim] && coincidence[dim + 1];
  coincidence[dim] = collapsed_coincidence;
  auto position = coincidence.begin() + dim + 1;
  coincidence.erase(position);

  isl::schedule_node_band result = isl_schedule_node_band_replace_partial_schedule(band, partial, false);
  result = result.set_permutable(permutable);
  result = isl_schedule_node_band_set_coincidence(result, coincidence);

  return result;
}

isl::schedule_node_band isl_schedule_node_band_fine_adjustment(const isl::schedule_node_band &band,
                                                               enum isl_schedule_node_fine_adjustment_type type,
                                                               const std::string &name, unsigned int dimension,
                                                               const isl::val &value) {
  isl::multi_union_pw_aff partial = band.partial_schedule();
  const unsigned int dims = partial.size();
  if (dimension >= dims) {
    log::Warn(std::string(__func__) + ": target dimension out of bounds");
    return band;
  }

  // Save the properties
  const bool permutable = band.permutable();
  std::vector<bool> coincidence = isl_schedule_node_band_get_coincidence(band);

  const isl::union_pw_aff &target = partial.at(dimension);
  struct isl_schedule_node_fine_adjustment_data arg = {
    .result = NULL,
    .name = name.c_str(),
    .value = value.get(),
    .type = type,
  };
  isl_union_pw_aff_foreach_pw_aff(target.get(), isl_pw_aff_fine_adjustment, &arg);
  isl::union_pw_aff adjusted = isl::manage(arg.result);

  // Replace the target in the partial schedule
  partial = partial.set_at(dimension, adjusted);

  // Insert the new schedule and delete the previous one.
  isl::schedule_node_band result = isl_schedule_node_band_replace_partial_schedule(band, partial, false);
  result = result.set_permutable(permutable);
  result = isl_schedule_node_band_set_coincidence(result, coincidence);

  return result;
}

isl::schedule_node_band isl_schedule_node_band_fine_adjustment(const isl::schedule_node_band &band,
                                                               enum isl_schedule_node_fine_adjustment_type type,
                                                               const std::string &name, unsigned int dimension,
                                                               int value) {
  isl::ctx ctx = band.ctx();
  const isl::val &val = isl::val(ctx, value);
  return isl_schedule_node_band_fine_adjustment(band, type, name, dimension, val);
}

////////////////////////////////////////////////////////////////////////////////
// schedule tree transformations (on a schedule_node)
////////////////////////////////////////////////////////////////////////////////

isl::schedule_node_band isl_schedule_node_band_unsplit(const isl::schedule_node_band &band) {
  // We assume isl_schedule_node_band_can_unsplit() has been checked

  isl::schedule_node_band result = band;
  // Insert the new partial schedule at the current position
  {
    const isl::schedule_node_band &child = result.child(0).as<isl::schedule_node_band>();
    const isl::multi_union_pw_aff &top = result.partial_schedule();
    const isl::multi_union_pw_aff &bottom = child.partial_schedule();
    const isl::multi_union_pw_aff &partial = isl_multi_union_pw_aff_cat(top, bottom);
    result = result.insert_partial_schedule(partial).as<isl::schedule_node_band>();
  }
  // Set the band's properties
  // NOTE: we do not set AST loop type or AST build options!
  {
    const isl::schedule_node_band &first = result.child(0).as<isl::schedule_node_band>();
    const isl::schedule_node_band &second = first.child(0).as<isl::schedule_node_band>();

    const bool permutable = first.permutable() && second.permutable();
    result = result.set_permutable(permutable);

    const unsigned int first_size = first.n_member();
    for (unsigned int i = 0; i < first_size; ++i) {
      const bool coincident = first.member_get_coincident(i);
      result = result.member_set_coincident(i, coincident);
    }

    const unsigned int second_size = second.n_member();
    for (unsigned int i = 0; i < second_size; ++i) {
      const bool coincident = second.member_get_coincident(i);
      const unsigned int member = first_size + i;
      result = result.member_set_coincident(member, coincident);
    }
  }
  // Remove the two successive children we merged
  {
    isl::schedule_node node = result.child(0);
    node = node.del();
    node = node.del();
    result = node.parent().as<isl::schedule_node_band>();
  }

  return result;
}

isl::schedule_node_band isl_schedule_node_band_fully_unsplit(const isl::schedule_node_band &band) {
  isl::schedule_node_band result = band;
  while (isl_schedule_node_band_can_unsplit(result)) {
    result = isl_schedule_node_band_unsplit(result);
  }
  return result;
}

isl::schedule_node isl_schedule_node_band_fully_split(const isl::schedule_node_band &band) {
  isl::schedule_node_band result = band;
  while (result.n_member() > 1) {
    const int kept = result.n_member() - 1;
    result = result.split(kept);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule isl_schedule_collapse(const isl::schedule &schedule, unsigned int first, unsigned int last) {
  if (last < first) {
    return schedule;
  }

  const isl::schedule_node &root = schedule.root();
  isl::schedule_node_band band = root.child(0).as<isl::schedule_node_band>();
  for (unsigned int dim = last; dim-- > first;) {
    band = isl_schedule_node_band_collapse(band, dim);
  }

  const isl::schedule &result = band.schedule();
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// String conversions for wrapped isl objects
////////////////////////////////////////////////////////////////////////////////

std::string to_c_code_string(const isl::schedule &s) { return to_c_code_string(s.get()); }

std::string to_c_code_string(const isl::schedule_constraints &c) { return to_c_code_string(c.get()); }

std::string to_block_string(const isl::schedule &s) { return to_block_string(s.get()); }

std::string to_block_string(const isl::union_map &map) { return to_block_string(map.get()); }

std::string to_block_string(const isl::schedule_constraints &constraints) { return to_block_string(constraints.get()); }

////////////////////////////////////////////////////////////////////////////////
// "Simple" C code string
////////////////////////////////////////////////////////////////////////////////

std::string to_c_code_string(__isl_keep isl_schedule *const schedule) {
  isl_ctx *const ctx = isl_schedule_get_ctx(schedule);
  isl_ast_build *const build = isl_ast_build_alloc(ctx);
  isl_ast_node *const ast = isl_ast_build_node_from_schedule(build, isl_schedule_copy(schedule));

  isl_printer *printer = isl_printer_to_str(ctx);
  printer = isl_printer_set_output_format(printer, ISL_FORMAT_C);
  printer = isl_printer_print_ast_node(printer, ast);

  char *const s = isl_printer_get_str(printer);
  std::string result(s);

  isl_printer_free(printer);
  isl_ast_build_free(build);
  isl_ast_node_free(ast);
  free(s);

  return result;
}

std::string to_c_code_string(__isl_keep isl_schedule_constraints *const input) {
  isl_schedule_constraints *const constraints = isl_schedule_constraints_copy(input);
  isl_ctx *const ctx = isl_schedule_constraints_get_ctx(constraints);
  const int previous_behaviour = isl_options_get_on_error(ctx);
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  isl_schedule *const schedule = isl_schedule_constraints_compute_schedule(constraints);
  isl_options_set_on_error(ctx, previous_behaviour);

  const std::string result = to_c_code_string(schedule);
  isl_schedule_free(schedule);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// Strings
////////////////////////////////////////////////////////////////////////////////

template <typename T, __isl_keep isl_ctx *(*ctx_getter)(__isl_keep T),
          __isl_give isl_printer *(*printer_function)(__isl_give isl_printer *, __isl_keep T)>
std::string isl_to_block_str(T t) {
  isl_ctx *const ctx = ctx_getter(t);

  isl_printer *printer = isl_printer_to_str(ctx);
  printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  printer = printer_function(printer, t);

  char *const block = isl_printer_get_str(printer);
  const std::string result(block);

  isl_printer_free(printer);
  free(block);

  return result;
}

std::string to_block_string(__isl_keep isl_schedule *const schedule) {
  return isl_to_block_str<isl_schedule *, isl_schedule_get_ctx, isl_printer_print_schedule>(schedule);
}

static inline isl_stat isl_println_basic_map(__isl_take isl_basic_map *const map, void *user) {
  isl_printer **printer = (isl_printer **)user;
  *printer = isl_printer_print_basic_map(*printer, map);
  *printer = isl_printer_print_str(*printer, "\n");

  isl_basic_map_free(map);
  return isl_stat_ok;
}

static inline __isl_give isl_printer *isl_printer_print_map_as_block(__isl_give isl_printer *printer,
                                                                     __isl_keep isl_map *const map) {
  isl_map_foreach_basic_map(map, isl_println_basic_map, (void *)&printer);
  return printer;
}

static inline isl_stat isl_println_map(__isl_take isl_map *const map, void *user) {
  isl_printer **printer = (isl_printer **)user;
  *printer = isl_printer_print_map_as_block(*printer, map);

  isl_map_free(map);
  return isl_stat_ok;
}

static inline __isl_give isl_printer *isl_printer_print_union_map_as_block(__isl_give isl_printer *printer,
                                                                           __isl_keep isl_union_map *const map) {
  isl_union_map_foreach_map(map, isl_println_map, (void *)&printer);
  return printer;
}

std::string to_block_string(__isl_keep isl_union_map *const map) {
  return isl_to_block_str<isl_union_map *, isl_union_map_get_ctx, isl_printer_print_union_map_as_block>(map);
}

static inline __isl_give isl_printer *isl_printer_print_schedule_constraints_as_block(
  __isl_give isl_printer *printer, __isl_keep isl_schedule_constraints *const constraints) {
  isl_union_set *const domain = isl_schedule_constraints_get_domain(constraints);
  if (domain) {
    printer = isl_printer_print_str(printer, "domain:\n");
    printer = isl_printer_print_union_set(printer, domain);
    printer = isl_printer_print_str(printer, "\n");
    isl_union_set_free(domain);
  }

  isl_set *const context = isl_schedule_constraints_get_context(constraints);
  if (context) {
    if (!isl_set_plain_is_empty(context)) {
      printer = isl_printer_print_str(printer, "context:\n");
      printer = isl_printer_print_set(printer, context);
      printer = isl_printer_print_str(printer, "\n");
    }
    isl_set_free(context);
  }

#define _print_constraints_union_map(getter, title)                       \
  do {                                                                    \
    isl_union_map *const _target_map = getter(constraints);               \
    if (_target_map) {                                                    \
      if (!isl_union_map_plain_is_empty(_target_map)) {                   \
        const std::string &_target_string = to_block_string(_target_map); \
        printer = isl_printer_print_str(printer, title ":\n");            \
        printer = isl_printer_print_str(printer, _target_string.c_str()); \
      }                                                                   \
      isl_union_map_free(_target_map);                                    \
    }                                                                     \
  } while (0)

  _print_constraints_union_map(isl_schedule_constraints_get_validity, "validity");
  _print_constraints_union_map(isl_schedule_constraints_get_proximity, "proximity");
  _print_constraints_union_map(isl_schedule_constraints_get_coincidence, "coincidence");
  _print_constraints_union_map(isl_schedule_constraints_get_conditional_validity, "conditional validity");
  _print_constraints_union_map(isl_schedule_constraints_get_conditional_validity_condition,
                               "conditional validity condition");

#undef _print_constraints_union_map

  return printer;
}

std::string to_block_string(__isl_keep isl_schedule_constraints *const constraints) {
  const std::string result = isl_to_block_str<isl_schedule_constraints *, isl_schedule_constraints_get_ctx,
                                              isl_printer_print_schedule_constraints_as_block>(constraints);
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// local definitions
////////////////////////////////////////////////////////////////////////////////

__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_collapse(__isl_take isl_multi_union_pw_aff *aff, int dim,
                                                                   __isl_take isl_val *val) {
  if (!aff) {
    return aff;
  }

  const isl_size count = isl_multi_union_pw_aff_size(aff);
  if (!count || dim >= count - 1) {
    return aff;
  }

  isl_union_pw_aff *const target_1 = isl_multi_union_pw_aff_get_at(aff, dim);
  isl_union_pw_aff *const target_2 = isl_multi_union_pw_aff_get_at(aff, dim + 1);

  isl_union_pw_aff_dump(target_1);
  isl_union_pw_aff_dump(target_2);

  isl_union_pw_aff *const scaled = isl_union_pw_aff_scale_val(target_1, val);
  isl_union_pw_aff *const collapsed = isl_union_pw_aff_add(scaled, target_2);
  isl_union_pw_aff_dump(collapsed);

  const isl_size size = isl_multi_union_pw_aff_size(aff);
  isl_union_pw_aff_list *list = NULL;

  if (dim != 0) {
    isl_union_pw_aff *const first = isl_multi_union_pw_aff_get_at(aff, 0);
    list = isl_union_pw_aff_list_from_union_pw_aff(first);
    for (isl_size i = 1; i < dim; ++i) {
      isl_union_pw_aff *const current = isl_multi_union_pw_aff_get_at(aff, i);
      list = isl_union_pw_aff_list_add(list, current);
    }
    list = isl_union_pw_aff_list_add(list, collapsed);
  } else {
    list = isl_union_pw_aff_list_from_union_pw_aff(collapsed);
  }

  for (isl_size i = dim + 2; i < size; ++i) {
    isl_union_pw_aff *const current = isl_multi_union_pw_aff_get_at(aff, i);
    list = isl_union_pw_aff_list_add(list, current);
  }

  isl_ctx *const ctx = isl_multi_union_pw_aff_get_ctx(aff);
  isl_space *const original_space = isl_multi_union_pw_aff_get_space(aff);
  const isl_size params = isl_space_dim(original_space, isl_dim_param);
  const isl_size new_size = isl_union_pw_aff_list_size(list);
  isl_space *const space = isl_space_set_alloc(ctx, params, new_size);
  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_from_union_pw_aff_list(space, list);

  isl_space_free(original_space);
  isl_multi_union_pw_aff_free(aff);

  return result;
}

__isl_give isl_val *isl_schedule_node_band_find_collapse_coeff(__isl_take isl_schedule_node *band, int dim) {
  isl_union_set *domain = isl_schedule_node_get_domain(band);

  isl_multi_union_pw_aff *const prefix = isl_schedule_node_get_prefix_schedule_multi_union_pw_aff(band);
  const isl_size prefix_size = isl_multi_union_pw_aff_size(prefix);

  isl_multi_union_pw_aff *partial = isl_schedule_node_band_get_partial_schedule(band);
  isl_multi_union_pw_aff *const schedule = isl_multi_union_pw_aff_cat(prefix, partial);
  isl_union_map *const map = isl_union_map_from_multi_union_pw_aff(schedule);
  domain = isl_union_set_apply(domain, map);
  isl_union_set *const union_lexmax = isl_union_set_lexmax(domain);
  isl_set *const lexmax = isl_set_from_union_set(union_lexmax);

  const isl_size target = prefix_size + dim + 1;
  isl_val *coeff = isl_set_plain_get_val_if_fixed(lexmax, isl_dim_set, target);
  coeff = isl_val_add_ui(coeff, 1);

  isl_set_free(lexmax);

  return coeff;
}

isl_stat isl_pw_aff_fine_adjustment(__isl_take isl_pw_aff *pa, void *user) {
  struct isl_schedule_node_fine_adjustment_data *arg = (struct isl_schedule_node_fine_adjustment_data *)user;

  isl_id *const id = isl_pw_aff_get_tuple_id(pa, isl_dim_in);
  const char *const name = isl_id_get_name(id);

  if (!strcmp(name, arg->name)) {
    isl_val *const val = isl_val_copy(arg->value);
    if (arg->type == FINE_ADJUSTMENT_SCALE_DOWN) {
      pa = isl_pw_aff_scale_down_val(pa, val);
    } else if (arg->type == FINE_ADJUSTMENT_MOD) {
      pa = isl_pw_aff_mod_val(pa, val);
    } else if (arg->type == FINE_ADJUSTMENT_SCALE) {
      pa = isl_pw_aff_scale_val(pa, val);
    } else {
      fprintf(stderr, "%s: unknown adjustment operation!\n", __func__);
    }
  }
  isl_id_free(id);

  if (arg->result) {
    arg->result = isl_union_pw_aff_add_pw_aff(arg->result, pa);
  } else {
    arg->result = isl_union_pw_aff_from_pw_aff(pa);
  }

  return isl_stat_ok;
}

////////////////////////////////////////////////////////////////////////////////
// "special" definitions
////////////////////////////////////////////////////////////////////////////////

// clang-format off

#define _define_isl_schedule_node_band_fine(name, type)                                                           \
  isl::schedule_node_band isl_schedule_node_band_fine_##name(                                                     \
      const isl::schedule_node_band &band, const std::string &name, unsigned int dim, int value) {                \
    const isl::schedule_node_band &result = isl_schedule_node_band_fine_adjustment(band, type, name, dim, value); \
    return result;                                                                                                \
  }                                                                                                               \
                                                                                                                  \
  isl::schedule_node_band isl_schedule_node_band_fine_##name(                                                     \
      const isl::schedule_node_band &band, const std::string &name, unsigned int dim, const isl::val &value) {    \
    const isl::schedule_node_band &result = isl_schedule_node_band_fine_adjustment(band, type, name, dim, value); \
    return result;                                                                                                \
  }

_define_isl_schedule_node_band_fine(mod, FINE_ADJUSTMENT_MOD)
_define_isl_schedule_node_band_fine(scale, FINE_ADJUSTMENT_SCALE)
_define_isl_schedule_node_band_fine(scale_down, FINE_ADJUSTMENT_SCALE_DOWN)

#undef _define_isl_schedule_node_band_fine

}  // namespace poly
}  // namespace ir
}  // namespace akg
