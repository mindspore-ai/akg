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

////////////////////////////////////////////////////////////////////////////////
// Misc.
////////////////////////////////////////////////////////////////////////////////

long isl_set_plain_get_num_si(const isl::set &s, int dim) { return isl_set_plain_get_num_si(s.get(), dim); }

long isl_set_plain_get_num_si(__isl_keep isl_set *const set, int dim) {
  isl_val *value = isl_set_plain_get_val_if_fixed(set, isl_dim_set, dim);
  value = isl_val_add_ui(value, 1);

  const long result = isl_val_get_num_si(value);
  isl_val_free(value);

  return result;
}

std::vector<int> extract_upper_bounds(const isl::set &s, const std::vector<int> &dimensions) {
  return extract_upper_bounds(s.get(), dimensions);
}

std::vector<int> extract_upper_bounds(__isl_keep isl_set *const set, const std::vector<int> &dimensions) {
  std::vector<int> result;

  const isl_size size = isl_set_dim(set, isl_dim_set);
  for (auto dimension : dimensions) {
    if (dimension < size) {
      isl_val *value = isl_set_plain_get_val_if_fixed(set, isl_dim_set, dimension);
      // upper_bound = lexmax + 1
      value = isl_val_add_ui(value, 1);

      char *const string = isl_val_to_str(value);
      result.push_back(std::stoi(string));

      isl_val_free(value);
      free(string);
    } else {
      LOG(WARNING) << "cannot retrieve size for dimension " << dimension;
    }
  }

  return result;
}

isl::space isl_space_set_cat(const isl::space &left, const isl::space &right) {
  isl_space *const a = isl_space_copy(left.get());
  isl_space *const b = isl_space_copy(right.get());
  isl_space *const result = isl_space_set_cat(a, b);
  return isl::manage(result);
}

__isl_give isl_space *isl_space_set_cat(__isl_take isl_space *left, __isl_take isl_space *right) {
  if (!left) {
    return right;
  } else if (!right) {
    return left;
  }

  const isl_size params = isl_space_dim(left, isl_dim_param);
  const isl_size out = isl_space_dim(left, isl_dim_out) + isl_space_dim(right, isl_dim_out);
  isl_ctx *const ctx = isl_space_get_ctx(left);

  // Note: we do not extract dim names if they were named in the input space.
  isl_space *const result = isl_space_set_alloc(ctx, params, out);

  isl_space_free(left);
  isl_space_free(right);

  return result;
}

isl::multi_union_pw_aff isl_multi_union_pw_aff_cat(const isl::multi_union_pw_aff &left,
                                                   const isl::multi_union_pw_aff &right) {
  isl_multi_union_pw_aff *const a = isl_multi_union_pw_aff_copy(left.get());
  isl_multi_union_pw_aff *const b = isl_multi_union_pw_aff_copy(right.get());
  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_cat(a, b);
  return isl::manage(result);
}

__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_cat(__isl_take isl_multi_union_pw_aff *left,
                                                              __isl_take isl_multi_union_pw_aff *right) {
  // Pointer check
  if (!right) {
    return left;
  } else if (!left) {
    return right;
  }

  const isl_size left_count = isl_multi_union_pw_aff_size(left);
  const isl_size right_count = isl_multi_union_pw_aff_size(right);

  // Size check
  if (!left_count) {
    isl_multi_union_pw_aff_free(left);
    return right;
  } else if (!right_count) {
    isl_multi_union_pw_aff_free(right);
    return left;
  }

  // Cat the two multi_union_pw_aff in a union_pw_aff_list
  isl_union_pw_aff *const first = isl_multi_union_pw_aff_get_at(left, 0);
  isl_union_pw_aff_list *list = isl_union_pw_aff_list_from_union_pw_aff(first);
  for (isl_size i = 1; i < left_count; ++i) {
    isl_union_pw_aff *const current = isl_multi_union_pw_aff_get_at(left, i);
    list = isl_union_pw_aff_list_add(list, current);
  }
  for (isl_size i = 0; i < right_count; ++i) {
    isl_union_pw_aff *const current = isl_multi_union_pw_aff_get_at(right, i);
    list = isl_union_pw_aff_list_add(list, current);
  }

  // Cat the spaces
  isl_space *const left_space = isl_multi_union_pw_aff_get_space(left);
  isl_space *const right_space = isl_multi_union_pw_aff_get_space(right);
  isl_space *const space = isl_space_set_cat(left_space, right_space);

  // Convert the list back into a multi_union_pw_aff
  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_from_union_pw_aff_list(space, list);

  // Cleanup
  isl_multi_union_pw_aff_free(left);
  isl_multi_union_pw_aff_free(right);

  return result;
}

isl::multi_union_pw_aff isl_multi_union_pw_aff_insert(const isl::multi_union_pw_aff &aff, unsigned pos,
                                                      const isl::union_pw_aff &el) {
  isl_multi_union_pw_aff *const a = isl_multi_union_pw_aff_copy(aff.get());
  isl_union_pw_aff *const e = isl_union_pw_aff_copy(el.get());
  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_insert(a, pos, e);
  return isl::manage(result);
}

__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_insert(__isl_take isl_multi_union_pw_aff *aff, unsigned pos,
                                                                 __isl_take isl_union_pw_aff *el) {
  if (!aff) {
    return isl_multi_union_pw_aff_from_union_pw_aff(el);
  }

  const isl_size count = isl_multi_union_pw_aff_size(aff);
  if (!count) {
    isl_multi_union_pw_aff_free(aff);
    return isl_multi_union_pw_aff_from_union_pw_aff(el);
  }

  isl_union_pw_aff *const first = isl_multi_union_pw_aff_get_at(aff, 0);
  isl_union_pw_aff_list *list = isl_union_pw_aff_list_from_union_pw_aff(first);
  for (isl_size i = 1; i < count; ++i) {
    isl_union_pw_aff *const current = isl_multi_union_pw_aff_get_at(aff, i);
    list = isl_union_pw_aff_list_add(list, current);
  }
  list = isl_union_pw_aff_list_insert(list, pos, el);

  isl_space *const aff_space = isl_multi_union_pw_aff_get_space(aff);
  const isl_size params = isl_space_dim(aff_space, isl_dim_param);
  const isl_size out = isl_space_dim(aff_space, isl_dim_out);

  isl_ctx *const ctx = isl_space_get_ctx(aff_space);
  isl_space *const space = isl_space_set_alloc(ctx, params, out + 1);
  isl_space_free(aff_space);

  isl_multi_union_pw_aff *const result = isl_multi_union_pw_aff_from_union_pw_aff_list(space, list);
  isl_multi_union_pw_aff_free(aff);

  return result;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node utilities
////////////////////////////////////////////////////////////////////////////////

bool isl_schedule_node_is_band(const isl::schedule_node &node) {
  return isl_schedule_node_is_band(node.get()) == isl_bool_true;
}

bool isl_schedule_node_is_sequence(const isl::schedule_node &node) {
  return isl_schedule_node_is_sequence(node.get()) == isl_bool_true;
}

bool isl_schedule_node_has_single_child(const isl::schedule_node &node) {
  return isl_schedule_node_has_single_child(node.get()) == isl_bool_true;
}

bool isl_schedule_node_band_can_unsplit(const isl::schedule_node &node) {
  return isl_schedule_node_band_can_unsplit(node.get()) == isl_bool_true;
}

isl_bool isl_schedule_node_is_band(__isl_keep isl_schedule_node *const node) {
  const enum isl_schedule_node_type node_type = isl_schedule_node_get_type(node);
  return node_type == isl_schedule_node_band ? isl_bool_true : isl_bool_false;
}

isl_bool isl_schedule_node_is_sequence(__isl_keep isl_schedule_node *const node) {
  const enum isl_schedule_node_type node_type = isl_schedule_node_get_type(node);
  return node_type == isl_schedule_node_sequence ? isl_bool_true : isl_bool_false;
}

isl_bool isl_schedule_node_has_single_child(__isl_keep isl_schedule_node *const node) {
  const isl_size children = isl_schedule_node_n_children(node);
  return children == 1 ? isl_bool_true : isl_bool_false;
}

isl_bool isl_schedule_node_band_can_unsplit(__isl_keep isl_schedule_node *const band) {
  if (!band || !isl_schedule_node_is_band(band) || !isl_schedule_node_has_single_child(band)) return isl_bool_false;

  isl_schedule_node *const child = isl_schedule_node_get_child(band, 0);
  const isl_bool child_is_band = isl_schedule_node_is_band(child);
  isl_schedule_node_free(child);

  return child_is_band;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band utilities
////////////////////////////////////////////////////////////////////////////////

__isl_give isl_bool *isl_schedule_node_band_get_coincidence(__isl_keep isl_schedule_node *const band) {
  if (!band || !isl_schedule_node_is_band(band)) {
    return 0;
  }

  const isl_size members = isl_schedule_node_band_n_member(band);
  if (!members) {
    return 0;
  }

  isl_bool *const coincidence = (isl_bool *)malloc(members * sizeof *coincidence);
  if (!coincidence) return coincidence;

  for (isl_size i = 0; i < members; ++i) {
    coincidence[i] = isl_schedule_node_band_member_get_coincident(band, i);
  }

  return coincidence;
}

__isl_give isl_schedule_node *isl_schedule_node_band_set_coincidence(__isl_take isl_schedule_node *band,
                                                                     __isl_take isl_bool *const coincidence) {
  if (!coincidence) {
    return band;
  }

  if (!band || !isl_schedule_node_is_band(band)) {
    free(coincidence);
    return band;
  }

  const isl_size members = isl_schedule_node_band_n_member(band);
  for (isl_size i = 0; i < members; ++i) {
    band = isl_schedule_node_band_member_set_coincident(band, i, coincidence[i]);
  }
  free(coincidence);

  return band;
}

__isl_give isl_schedule_node *isl_schedule_node_band_copy_properties(__isl_take isl_schedule_node *band,
                                                                     __isl_keep isl_schedule_node *const original) {
  if (!band || !original) {
    return band;
  }

  const isl_bool permutable = isl_schedule_node_band_get_permutable(original);
  isl_union_set *const ast_build_options = isl_schedule_node_band_get_ast_build_options(original);

  band = isl_schedule_node_band_set_permutable(band, permutable);
  band = isl_schedule_node_band_set_ast_build_options(band, ast_build_options);

  const isl_size old_size = isl_schedule_node_band_n_member(original);
  const isl_size new_size = isl_schedule_node_band_n_member(band);
  const isl_size limit = new_size >= old_size ? old_size : new_size;
  for (isl_size i = 0; i < limit; ++i) {
    const isl_bool coincident = isl_schedule_node_band_member_get_coincident(original, i);
    const enum isl_ast_loop_type loop_type = isl_schedule_node_band_member_get_ast_loop_type(original, i);
    const enum isl_ast_loop_type isolate_type = isl_schedule_node_band_member_get_isolate_ast_loop_type(original, i);

    band = isl_schedule_node_band_member_set_coincident(band, i, coincident);
    band = isl_schedule_node_band_member_set_ast_loop_type(band, i, loop_type);
    band = isl_schedule_node_band_member_set_isolate_ast_loop_type(band, i, isolate_type);
  }

  return band;
}

__isl_give isl_schedule_node *isl_schedule_node_band_replace_partial_schedule(
  __isl_take isl_schedule_node *band, __isl_take isl_multi_union_pw_aff *schedule, isl_bool keep_properties) {
  if (!isl_schedule_node_is_band(band)) {
    fprintf(stderr, "%s: not an isl_schedule_node_band!\n", __func__);
    isl_multi_union_pw_aff_free(schedule);
    return band;
  }

  // Insert the new band
  band = isl_schedule_node_insert_partial_schedule(band, schedule);
  if (keep_properties) {
    // It is usually the caller's responsibility to know/decide the properties of
    // the new partial schedule.
    // For convenience, the caller can set keep_properties to isl_bool_true if they
    // are sure all properties will remain identical.
    isl_schedule_node *const original = isl_schedule_node_get_child(band, 0);
    band = isl_schedule_node_band_copy_properties(band, original);
    isl_schedule_node_free(original);
  }
  // Delete the previous node
  isl_schedule_node *result = isl_schedule_node_get_child(band, 0);
  result = isl_schedule_node_delete(result);
  // Move back to the target node
  result = isl_schedule_node_parent(result);
  // Do not forget to free the input schedule node.
  isl_schedule_node_free(band);

  return result;
}

__isl_give isl_set *isl_schedule_node_band_lexmax(__isl_keep isl_schedule_node *const band) {
  isl_union_set *const domain = isl_schedule_node_get_domain(band);
  isl_union_map *const schedule = isl_schedule_node_band_get_partial_schedule_union_map(band);
  isl_union_set *const application = isl_union_set_apply(domain, schedule);
  isl_union_set *const lexmax = isl_union_set_lexmax(application);
  if (!isl_union_set_isa_set(lexmax)) {
    return NULL;
  }

  isl_set *const result = isl_set_from_union_set(lexmax);
  return result;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule_node isl_schedule_node_band_interchange(const isl::schedule_node &band, int first, int second) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_interchange(b, first, second);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_interchange(__isl_take isl_schedule_node *band, int first,
                                                                 int second) {
  if (!band || !isl_schedule_node_is_band(band) || first == second) {
    return band;
  }

  isl_multi_union_pw_aff *partial = isl_schedule_node_band_get_partial_schedule(band);

  const isl_size dims = isl_multi_union_pw_aff_size(partial);
  if (first >= dims || second >= dims) {
    fprintf(stderr, "%s: target dimension out of bounds\n", __func__);
    isl_multi_union_pw_aff_free(partial);
    return band;
  }

  isl_union_pw_aff *const aff_1 = isl_multi_union_pw_aff_get_at(partial, first);
  isl_union_pw_aff *const aff_2 = isl_multi_union_pw_aff_get_at(partial, second);
  partial = isl_multi_union_pw_aff_set_at(partial, first, aff_2);
  partial = isl_multi_union_pw_aff_set_at(partial, second, aff_1);

  // Save the properties
  const isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  isl_bool *const coincidence = isl_schedule_node_band_get_coincidence(band);

  // Interchange the coincidences
  const isl_bool tmp = coincidence[first];
  coincidence[first] = coincidence[second];
  coincidence[second] = tmp;

  // Replace the partial schedule and set its properties
  band = isl_schedule_node_band_replace_partial_schedule(band, partial, isl_bool_false);
  band = isl_schedule_node_band_set_permutable(band, permutable);
  band = isl_schedule_node_band_set_coincidence(band, coincidence);

  return band;
}

isl::schedule_node isl_schedule_node_band_stripmine(const isl::schedule_node &band, int dim, int value) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_stripmine(b, dim, value);
  return isl::manage(result);
}

isl::schedule_node isl_schedule_node_band_stripmine(const isl::schedule_node &band, int dim, const isl::val &value) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_val *const v = isl_val_copy(value.get());
  isl_schedule_node *const result = isl_schedule_node_band_stripmine(b, dim, v);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_stripmine(__isl_take isl_schedule_node *band, int dimension,
                                                               __isl_take isl_val *value) {
  if (!band || !isl_schedule_node_is_band(band)) {
    isl_val_free(value);
    return band;
  }

  const isl_size members = isl_schedule_node_band_n_member(band);
  if (dimension >= members) {
    isl_val_free(value);
    return band;
  }

  isl_multi_union_pw_aff *schedule = isl_schedule_node_band_get_partial_schedule(band);
  isl_union_pw_aff *div = isl_multi_union_pw_aff_get_at(schedule, dimension);
  isl_union_pw_aff *mod = isl_multi_union_pw_aff_get_at(schedule, dimension);

  isl_val *const v1 = value;
  isl_val *const v2 = isl_val_copy(v1);
  div = isl_union_pw_aff_scale_down_val(div, v1);
  mod = isl_union_pw_aff_mod_val(mod, v2);

  schedule = isl_multi_union_pw_aff_set_at(schedule, dimension, div);
  schedule = isl_multi_union_pw_aff_insert(schedule, dimension + 1, mod);

  const isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  isl_bool *const coincidence = isl_schedule_node_band_get_coincidence(band);

  band = isl_schedule_node_band_replace_partial_schedule(band, schedule, isl_bool_true);

  // Set permutable
  band = isl_schedule_node_band_set_permutable(band, permutable);
  // We need to manually set the coincidence since we introduced a new dimension
  for (isl_size i = 0; i < dimension; ++i) {
    band = isl_schedule_node_band_member_set_coincident(band, i, coincidence[i]);
  }
  band = isl_schedule_node_band_member_set_coincident(band, dimension, coincidence[dimension]);
  band = isl_schedule_node_band_member_set_coincident(band, dimension + 1, coincidence[dimension]);
  for (isl_size i = dimension + 2; i < members + 1; ++i) {
    band = isl_schedule_node_band_member_set_coincident(band, i, coincidence[i - 1]);
  }

  free(coincidence);

  return band;
}

__isl_give isl_schedule_node *isl_schedule_node_band_stripmine(__isl_take isl_schedule_node *band, int dimension,
                                                               int value) {
  isl_ctx *const ctx = isl_schedule_node_get_ctx(band);
  isl_val *const val = isl_val_int_from_si(ctx, value);
  return isl_schedule_node_band_stripmine(band, dimension, val);
}

isl::schedule_node isl_schedule_node_band_collapse(const isl::schedule_node &band, int dim) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_collapse(b, dim);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_collapse(__isl_take isl_schedule_node *band, int dim) {
  if (!band || !isl_schedule_node_is_band(band)) {
    return band;
  }

  const isl_size members = isl_schedule_node_band_n_member(band);
  if (dim >= members) {
    return band;
  }

  isl_val *const coeff = isl_schedule_node_band_find_collapse_coeff(band, dim);
  isl_multi_union_pw_aff *partial = isl_schedule_node_band_get_partial_schedule(band);
  partial = isl_multi_union_pw_aff_collapse(partial, dim, coeff);

  const isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  isl_bool *const coincidence = isl_schedule_node_band_get_coincidence(band);

  band = isl_schedule_node_band_replace_partial_schedule(band, partial, isl_bool_false);
  band = isl_schedule_node_band_set_permutable(band, permutable);
  for (isl_size i = 0; i < dim; ++i) {
    const isl_bool current = coincidence[i];
    band = isl_schedule_node_band_member_set_coincident(band, i, current);
  }
  const isl_bool collapsed_coincidence = coincidence[dim] && coincidence[dim + 1] ? isl_bool_true : isl_bool_false;
  band = isl_schedule_node_band_member_set_coincident(band, dim, collapsed_coincidence);
  for (isl_size i = dim + 1; i < members - 1; ++i) {
    const isl_bool current = coincidence[i + 1];
    band = isl_schedule_node_band_member_set_coincident(band, i, current);
  }

  free(coincidence);

  return band;
}

__isl_give isl_schedule_node *isl_schedule_node_band_fine_adjustment(__isl_take isl_schedule_node *band,
                                                                     enum isl_schedule_node_fine_adjustment_type type,
                                                                     const char *name, int dimension,
                                                                     __isl_take isl_val *value) {
  const enum isl_schedule_node_type node_type = isl_schedule_node_get_type(band);
  if (node_type != isl_schedule_node_band) {
    fprintf(stderr, "%s: not an isl_schedule_node_band!\n", __func__);
    return band;
  }

  isl_multi_union_pw_aff *partial = isl_schedule_node_band_get_partial_schedule(band);

  const isl_size dims = isl_multi_union_pw_aff_size(partial);
  if (dimension >= dims) {
    isl_multi_union_pw_aff_dump(partial);
    isl_multi_union_pw_aff_free(partial);
    fprintf(stderr, "%s: target dimension out of bounds\n", __func__);
    return band;
  }

  // Save the properties
  const isl_bool permutable = isl_schedule_node_band_get_permutable(band);
  isl_bool *const coincidence = isl_schedule_node_band_get_coincidence(band);

  isl_union_pw_aff *target = isl_multi_union_pw_aff_get_at(partial, dimension);
  struct isl_schedule_node_fine_adjustment_data arg = {
    .result = NULL,
    .name = name,
    .value = value,
    .type = type,
  };
  isl_union_pw_aff_foreach_pw_aff(target, isl_pw_aff_fine_adjustment, &arg);
  isl_val_free(value);

  // Replace the target in the partial schedule
  partial = isl_multi_union_pw_aff_set_at(partial, dimension, arg.result);
  isl_union_pw_aff_free(target);

  // Insert the new schedule and delete the previous one.
  band = isl_schedule_node_band_replace_partial_schedule(band, partial, isl_bool_false);
  band = isl_schedule_node_band_set_permutable(band, permutable);
  band = isl_schedule_node_band_set_coincidence(band, coincidence);

  return band;
}

__isl_give isl_schedule_node *isl_schedule_node_band_fine_adjustment(__isl_take isl_schedule_node *band,
                                                                     enum isl_schedule_node_fine_adjustment_type type,
                                                                     const char *name, int dimension, int value) {
  isl_ctx *const ctx = isl_schedule_node_get_ctx(band);
  isl_val *const val = isl_val_int_from_si(ctx, value);
  return isl_schedule_node_band_fine_adjustment(band, type, name, dimension, val);
}

////////////////////////////////////////////////////////////////////////////////
// schedule tree transformations (on a schedule_node)
////////////////////////////////////////////////////////////////////////////////

isl::schedule_node isl_schedule_node_band_unsplit(const isl::schedule_node &band) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_unsplit(b);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_unsplit(__isl_take isl_schedule_node *band) {
  // We assume isl_schedule_node_band_can_unsplit() has been checked

  isl_schedule_node *const child = isl_schedule_node_get_child(band, 0);
  isl_multi_union_pw_aff *const top = isl_schedule_node_band_get_partial_schedule(band);
  isl_multi_union_pw_aff *const bottom = isl_schedule_node_band_get_partial_schedule(child);

  // Insert the new partial schedule at the current position
  isl_multi_union_pw_aff *const partial = isl_multi_union_pw_aff_cat(top, bottom);
  band = isl_schedule_node_insert_partial_schedule(band, partial);

  // Set the band's properties
  // NOTE: we do not set AST loop type or AST build options!
  {
    isl_schedule_node *const first = isl_schedule_node_get_child(band, 0);
    isl_schedule_node *const second = isl_schedule_node_get_child(first, 0);

    const isl_bool permutable =
      isl_schedule_node_band_get_permutable(first) && isl_schedule_node_band_get_permutable(second) ? isl_bool_true
                                                                                                    : isl_bool_false;
    band = isl_schedule_node_band_set_permutable(band, permutable);

    const isl_size first_size = isl_schedule_node_band_n_member(first);
    const isl_size second_size = isl_schedule_node_band_n_member(second);
    for (isl_size i = 0; i < first_size; ++i) {
      const isl_bool coincident = isl_schedule_node_band_member_get_coincident(first, i);
      band = isl_schedule_node_band_member_set_coincident(band, i, coincident);
    }
    for (isl_size i = 0; i < second_size; ++i) {
      const isl_bool coincident = isl_schedule_node_band_member_get_coincident(second, i);
      band = isl_schedule_node_band_member_set_coincident(band, first_size + i, coincident);
    }

    isl_schedule_node_free(first);
    isl_schedule_node_free(second);
  }

  // Remove the two successive children we merged
  isl_schedule_node *result = isl_schedule_node_get_child(band, 0);
  result = isl_schedule_node_delete(result);
  result = isl_schedule_node_delete(result);
  // Move to the original position in the tree
  result = isl_schedule_node_parent(result);

  isl_schedule_node_free(band);
  isl_schedule_node_free(child);

  return result;
}

isl::schedule_node isl_schedule_node_band_fully_unsplit(const isl::schedule_node &band) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_fully_unsplit(b);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_fully_unsplit(__isl_take isl_schedule_node *band) {
  while (isl_schedule_node_band_can_unsplit(band)) {
    band = isl_schedule_node_band_unsplit(band);
  }
  return band;
}

isl::schedule_node isl_schedule_node_band_fully_split(const isl::schedule_node &band) {
  isl_schedule_node *const b = isl_schedule_node_copy(band.get());
  isl_schedule_node *const result = isl_schedule_node_band_fully_split(b);
  return isl::manage(result);
}

__isl_give isl_schedule_node *isl_schedule_node_band_fully_split(__isl_take isl_schedule_node *band) {
  while (isl_schedule_node_band_n_member(band) > 1) {
    const isl_size kept = isl_schedule_node_band_n_member(band) - 1;
    band = isl_schedule_node_band_split(band, kept);
  }
  return band;
}

////////////////////////////////////////////////////////////////////////////////
// isl_schedule transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule isl_schedule_collapse(const isl::schedule &schedule, int first, int last) {
  isl_schedule *const s = isl_schedule_copy(schedule.get());
  isl_schedule *const result = isl_schedule_collapse(s, first, last);
  return isl::manage(result);
}

__isl_give isl_schedule *isl_schedule_collapse(__isl_take isl_schedule *schedule, int first, int last) {
  if (last < first) {
    return schedule;
  }

  isl_schedule_node *const root = isl_schedule_get_root(schedule);
  isl_schedule_node *band = isl_schedule_node_get_child(root, 0);
  isl_schedule_node_free(root);

  for (int dim = last; dim-- > first;) {
    band = isl_schedule_node_band_collapse(band, dim);
  }

  isl_schedule *const result = isl_schedule_node_get_schedule(band);
  isl_schedule_node_free(band);
  isl_schedule_free(schedule);

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

#define _define_isl_schedule_node_band_fine(name, type)                                                          \
  __isl_give isl_schedule_node *isl_schedule_node_band_fine_##name(                                              \
      __isl_take isl_schedule_node *band,  const char *name, int dimension, int value) {                         \
    return isl_schedule_node_band_fine_adjustment(band, (type), name, dimension, value);                         \
  }                                                                                                              \
                                                                                                                 \
  __isl_give isl_schedule_node *isl_schedule_node_band_fine_##name(                                              \
    __isl_take isl_schedule_node *band, const char *name, int dimension, __isl_take isl_val *value) {            \
    return isl_schedule_node_band_fine_adjustment(band, (type), name, dimension, value);                         \
  }                                                                                                              \
                                                                                                                 \
  isl::schedule_node isl_schedule_node_band_fine_##name(                                                         \
      const isl::schedule_node &band,  const std::string &name, int dimension, int value) {                      \
    isl_schedule_node *const b = isl_schedule_node_copy(band.get());                                             \
    const char *const name_str = name.c_str();                                                                   \
    isl_schedule_node *const result = isl_schedule_node_band_fine_##name(b, name_str, dimension, value);         \
    return isl::manage(result);                                                                                  \
  }                                                                                                              \
                                                                                                                 \
  isl::schedule_node isl_schedule_node_band_fine_##name(                                                         \
      const isl::schedule_node &band,  const std::string &name, int dimension, const isl::val &value) {          \
    isl_schedule_node *const b = isl_schedule_node_copy(band.get());                                             \
    const char *const name_str = name.c_str();                                                                   \
    isl_val *const v = isl_val_copy(value.get());                                                                \
    isl_schedule_node *const result = isl_schedule_node_band_fine_##name(b, name_str, dimension, v);             \
    return isl::manage(result);                                                                                  \
  }

_define_isl_schedule_node_band_fine(mod, FINE_ADJUSTMENT_MOD)
_define_isl_schedule_node_band_fine(scale, FINE_ADJUSTMENT_SCALE)
_define_isl_schedule_node_band_fine(scale_down, FINE_ADJUSTMENT_SCALE_DOWN)

#undef _define_isl_schedule_node_band_fine

}  // namespace poly
}  // namespace ir
}  // namespace akg
