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
#ifndef POLY_ISL_UTIL_H_
#define POLY_ISL_UTIL_H_

#include <vector>

#include "isl/cpp.h"

#include "poly/pass_info.h"
#include "poly/scop_info.h"

// Hardcore isl functions

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// Misc.
////////////////////////////////////////////////////////////////////////////////

long isl_set_plain_get_num_si(const isl::set &s, int dim);
long isl_set_plain_get_num_si(__isl_keep isl_set *const set, int dim);

std::vector<int> extract_upper_bounds(const isl::set &s, const std::vector<int> &dimensions);
std::vector<int> extract_upper_bounds(__isl_keep isl_set *const set, const std::vector<int> &dimensions);

// Combining isl_spaces
isl::space isl_space_set_cat(const isl::space &left, const isl::space &right);
__isl_give isl_space *isl_space_set_cat(__isl_take isl_space *left, __isl_take isl_space *right);

// Utilities for isl_multi_union_pw_aff*
isl::multi_union_pw_aff isl_multi_union_pw_aff_cat(const isl::multi_union_pw_aff &left,
                                                   const isl::multi_union_pw_aff &right);
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_cat(__isl_take isl_multi_union_pw_aff *left,
                                                              __isl_take isl_multi_union_pw_aff *right);

isl::multi_union_pw_aff isl_multi_union_pw_aff_insert(const isl::multi_union_pw_aff &aff, unsigned pos,
                                                      const isl::union_pw_aff &el);
__isl_give isl_multi_union_pw_aff *isl_multi_union_pw_aff_insert(__isl_take isl_multi_union_pw_aff *aff, unsigned pos,
                                                                 __isl_take isl_union_pw_aff *el);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node utilities
////////////////////////////////////////////////////////////////////////////////

bool isl_schedule_node_is_band(const isl::schedule_node &node);
bool isl_schedule_node_is_sequence(const isl::schedule_node &node);
bool isl_schedule_node_has_single_child(const isl::schedule_node &node);
bool isl_schedule_node_band_can_unsplit(const isl::schedule_node &node);

isl_bool isl_schedule_node_is_band(__isl_keep isl_schedule_node *const node);
isl_bool isl_schedule_node_is_sequence(__isl_keep isl_schedule_node *const node);
isl_bool isl_schedule_node_has_single_child(__isl_keep isl_schedule_node *const node);
isl_bool isl_schedule_node_band_can_unsplit(__isl_keep isl_schedule_node *const band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band utilities
////////////////////////////////////////////////////////////////////////////////

__isl_give isl_bool *isl_schedule_node_band_get_coincidence(__isl_keep isl_schedule_node *const band);
__isl_give isl_schedule_node *isl_schedule_node_band_set_coincidence(__isl_take isl_schedule_node *band,
                                                                     __isl_take isl_bool *const coincidence);
__isl_give isl_schedule_node *isl_schedule_node_band_copy_properties(__isl_take isl_schedule_node *band,
                                                                     __isl_keep isl_schedule_node *const original);
__isl_give isl_schedule_node *isl_schedule_node_band_replace_partial_schedule(
  __isl_take isl_schedule_node *band, __isl_take isl_multi_union_pw_aff *schedule, isl_bool keep_properties);
__isl_give isl_set *isl_schedule_node_band_lexmax(__isl_keep isl_schedule_node *const band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band transformations
////////////////////////////////////////////////////////////////////////////////

// These functions only preserve permutable/coincidence.
// AST options are not preserved.

// interchange dimensions 'first' and 'second'
isl::schedule_node isl_schedule_node_band_interchange(const isl::schedule_node &band, int first, int second);
__isl_give isl_schedule_node *isl_schedule_node_band_interchange(__isl_take isl_schedule_node *band, int first,
                                                                 int second);

// strip mine only dimension 'dim'
isl::schedule_node isl_schedule_node_band_stripmine(const isl::schedule_node &band, int dim, int value);
isl::schedule_node isl_schedule_node_band_stripmine(const isl::schedule_node &band, int dim, const isl::val &value);

__isl_give isl_schedule_node *isl_schedule_node_band_stripmine(__isl_take isl_schedule_node *band, int dim, int value);
__isl_give isl_schedule_node *isl_schedule_node_band_stripmine(__isl_take isl_schedule_node *band, int dim,
                                                               __isl_take isl_val *value);

// collapse dimensions 'dim' and 'dim + 1'
isl::schedule_node isl_schedule_node_band_collapse(const isl::schedule_node &band, int dim);
__isl_give isl_schedule_node *isl_schedule_node_band_collapse(__isl_take isl_schedule_node *band, int dim);

// modulo/scale/scale down a given dimension of a target statement
isl::schedule_node isl_schedule_node_band_fine_mod(const isl::schedule_node &band, const std::string &name,
                                                   int dimension, int value);
isl::schedule_node isl_schedule_node_band_fine_scale(const isl::schedule_node &band, const std::string &name,
                                                     int dimension, int value);
isl::schedule_node isl_schedule_node_band_fine_scale_down(const isl::schedule_node &band, const std::string &name,
                                                          int dimension, int value);
isl::schedule_node isl_schedule_node_band_fine_mod(const isl::schedule_node &band, const std::string &name,
                                                   int dimension, const isl::val &value);
isl::schedule_node isl_schedule_node_band_fine_scale(const isl::schedule_node &band, const std::string &name,
                                                     int dimension, const isl::val &value);
isl::schedule_node isl_schedule_node_band_fine_scale_down(const isl::schedule_node &band, const std::string &name,
                                                          int dimension, const isl::val &value);

__isl_give isl_schedule_node *isl_schedule_node_band_fine_mod(__isl_take isl_schedule_node *band, const char *name,
                                                              int dimension, int value);
__isl_give isl_schedule_node *isl_schedule_node_band_fine_scale(__isl_take isl_schedule_node *band, const char *name,
                                                                int dimension, int value);
__isl_give isl_schedule_node *isl_schedule_node_band_fine_scale_down(__isl_take isl_schedule_node *band,
                                                                     const char *name, int dimension, int value);
__isl_give isl_schedule_node *isl_schedule_node_band_fine_mod(__isl_take isl_schedule_node *band, const char *name,
                                                              int dimension, __isl_take isl_val *value);
__isl_give isl_schedule_node *isl_schedule_node_band_fine_scale(__isl_take isl_schedule_node *band, const char *name,
                                                                int dimension, __isl_take isl_val *value);
__isl_give isl_schedule_node *isl_schedule_node_band_fine_scale_down(__isl_take isl_schedule_node *band,
                                                                     const char *name, int dimension,
                                                                     __isl_take isl_val *value);

////////////////////////////////////////////////////////////////////////////////
// schedule tree transformations (on a schedule_node)
////////////////////////////////////////////////////////////////////////////////

// Merge two nested isl_schedule_node_band.
// Assuming the input schedule_node_band has only one child and this child is an isl_schedule_node_band.
isl::schedule_node isl_schedule_node_band_unsplit(const isl::schedule_node &band);
__isl_give isl_schedule_node *isl_schedule_node_band_unsplit(__isl_take isl_schedule_node *band);

// Call isl_schedule_node_band_unsplit() until it is not possible
isl::schedule_node isl_schedule_node_band_fully_unsplit(const isl::schedule_node &band);
__isl_give isl_schedule_node *isl_schedule_node_band_fully_unsplit(__isl_take isl_schedule_node *band);

// Call isl_schedule_node_band_split() until it is not possible
isl::schedule_node isl_schedule_node_band_fully_split(const isl::schedule_node &band);
__isl_give isl_schedule_node *isl_schedule_node_band_fully_split(__isl_take isl_schedule_node *band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule isl_schedule_collapse(const isl::schedule &schedule, int first, int last);
__isl_give isl_schedule *isl_schedule_collapse(__isl_take isl_schedule *schedule, int first, int last);

////////////////////////////////////////////////////////////////////////////////
// "Readable" strings conversions
////////////////////////////////////////////////////////////////////////////////

// Simple C code
std::string to_c_code_string(const isl::schedule &sch);
std::string to_c_code_string(__isl_keep isl_schedule *const schedule);
std::string to_c_code_string(const isl::schedule_constraints &c);
std::string to_c_code_string(__isl_keep isl_schedule_constraints *const constraints);

// isl_*_to_str functions return isl formatted strings!
std::string to_block_string(const isl::schedule &s);
std::string to_block_string(__isl_keep isl_schedule *const schedule);

std::string to_block_string(const isl::union_map &map);
std::string to_block_string(__isl_keep isl_union_map *const map);

std::string to_block_string(const isl::schedule_constraints &constraints);
std::string to_block_string(__isl_keep isl_schedule_constraints *const constraints);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_ISL_UTIL_H_
