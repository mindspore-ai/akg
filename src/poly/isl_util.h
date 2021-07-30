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
// C++ wrappers for some missing isl functions wrappers
////////////////////////////////////////////////////////////////////////////////

bool isl_aff_is_cst(const isl::aff &a);

std::string isl_set_get_dim_name(const isl::set &s, enum isl_dim_type type, unsigned int pos);
isl::id isl_set_get_dim_id(const isl::set &s, enum isl_dim_type type, unsigned int pos);
int isl_set_find_dim_by_id(const isl::set &s, enum isl_dim_type type, const isl::id &id);
int isl_set_find_dim_by_name(const isl::set &s, enum isl_dim_type type, const std::string &name);
unsigned isl_set_dim(const isl::set &s, enum isl_dim_type type);
long isl_set_plain_get_num_si(const isl::set &s, unsigned int pos);
isl::val isl_set_plain_get_val_if_fixed(const isl::set &s, enum isl_dim_type type, unsigned int pos);
isl::set isl_set_move_dims(const isl::set &s, enum isl_dim_type dst_type, unsigned int dst_pos,
                           enum isl_dim_type src_type, unsigned int src_pos, unsigned int n);

std::string isl_map_get_dim_name(const isl::map &m, enum isl_dim_type type, unsigned int pos);
isl::id isl_map_get_dim_id(const isl::map &m, enum isl_dim_type type, unsigned int pos);
bool isl_map_involves_dims(const isl::map &m, enum isl_dim_type type, unsigned int first, unsigned n);
isl::map isl_map_drop_constraints_not_involving_dims(const isl::map &m, enum isl_dim_type type, unsigned int first,
                                                     unsigned n);

isl::union_map isl_union_map_align_params(const isl::union_map &map, const isl::space &space);

isl::union_pw_aff_list isl_union_pw_aff_list_insert(const isl::union_pw_aff_list &list, unsigned int pos,
                                                    const isl::union_pw_aff &aff);

isl::space isl_space_set_alloc(isl::ctx ctx, unsigned int nparam, unsigned int dim);
isl::id isl_space_get_dim_id(const isl::space &space, enum isl_dim_type type, unsigned int pos);
isl::space isl_space_set_dim_id(const isl::space &space, enum isl_dim_type type, unsigned int pos, const isl::id &id);

////////////////////////////////////////////////////////////////////////////////
// Misc.
////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> isl_set_dim_names(const isl::set &set, enum isl_dim_type type);
std::vector<std::string> isl_set_all_names(const isl::set &set);

std::vector<int> isl_set_lexmax_extract_upper_bounds(const isl::set &set, const std::vector<std::string> &names);
std::vector<int> isl_set_lexmax_extract_upper_bounds(const isl::set &s, const std::vector<int> &dimensions);

isl::space isl_space_copy_param_names(const isl::space &space, const isl::space &source);

// Combining isl_spaces
isl::space isl_space_set_cat(const isl::space &left, const isl::space &right);

// Utilities for isl_multi_union_pw_aff*
isl::multi_union_pw_aff isl_multi_union_pw_aff_cat(const isl::multi_union_pw_aff &left,
                                                   const isl::multi_union_pw_aff &right);

isl::multi_union_pw_aff isl_multi_union_pw_aff_insert(const isl::multi_union_pw_aff &aff, unsigned pos,
                                                      const isl::union_pw_aff &el);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node utilities
////////////////////////////////////////////////////////////////////////////////

bool isl_schedule_node_is_band(const isl::schedule_node &node);
bool isl_schedule_node_is_sequence(const isl::schedule_node &node);
bool isl_schedule_node_has_single_child(const isl::schedule_node &node);
bool isl_schedule_node_band_can_unsplit(const isl::schedule_node_band &band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band utilities
////////////////////////////////////////////////////////////////////////////////

std::vector<bool> isl_schedule_node_band_get_coincidence(const isl::schedule_node_band &band);

isl::schedule_node_band isl_schedule_node_band_set_coincidence(const isl::schedule_node_band &band,
                                                               const std::vector<bool> &coincidence);

isl::schedule_node_band isl_schedule_node_band_member_copy_properties(const isl::schedule_node_band &band, int pos,
                                                                      const isl::schedule_node_band &wrapped_original);

isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node &node,
                                                               const isl::schedule_node &original);
isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node &node,
                                                               const isl::schedule_node_band &original);
isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node_band &band,
                                                               const isl::schedule_node &original);
isl::schedule_node_band isl_schedule_node_band_copy_properties(const isl::schedule_node_band &band,
                                                               const isl::schedule_node_band &original);

isl::schedule_node_band isl_schedule_node_band_replace_partial_schedule(const isl::schedule_node &node,
                                                                        const isl::multi_union_pw_aff &partial,
                                                                        bool keep_properties);
isl::schedule_node_band isl_schedule_node_band_replace_partial_schedule(const isl::schedule_node_band &band,
                                                                        const isl::multi_union_pw_aff &partial,
                                                                        bool keep_properties);

isl::set isl_schedule_node_band_lexmax(const isl::schedule_node &node);
isl::set isl_schedule_node_band_lexmax(const isl::schedule_node_band &band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule_node_band transformations
////////////////////////////////////////////////////////////////////////////////

// These functions only preserve permutable/coincidence.
// AST options are not preserved.

// interchange dimensions 'first' and 'second'
isl::schedule_node_band isl_schedule_node_band_interchange(const isl::schedule_node_band &band, unsigned int first,
                                                           unsigned int second);

// strip mine only dimension 'dim'
isl::schedule_node_band isl_schedule_node_band_stripmine(const isl::schedule_node_band &band, unsigned int dim,
                                                         int value);
isl::schedule_node_band isl_schedule_node_band_stripmine(const isl::schedule_node_band &band, unsigned int dim,
                                                         const isl::val &value);

// collapse dimensions 'dim' and 'dim + 1'
isl::schedule_node_band isl_schedule_node_band_collapse(const isl::schedule_node_band &band, unsigned int dim);

// modulo/scale/scale down a given dimension of a target statement
isl::schedule_node_band isl_schedule_node_band_fine_mod(const isl::schedule_node_band &band, const std::string &name,
                                                        unsigned int dimension, int value);
isl::schedule_node_band isl_schedule_node_band_fine_scale(const isl::schedule_node_band &band, const std::string &name,
                                                          unsigned int dimension, int value);
isl::schedule_node_band isl_schedule_node_band_fine_scale_down(const isl::schedule_node_band &band,
                                                               const std::string &name, unsigned int dimension,
                                                               int value);
isl::schedule_node_band isl_schedule_node_band_fine_mod(const isl::schedule_node_band &band, const std::string &name,
                                                        unsigned int dimension, const isl::val &value);
isl::schedule_node_band isl_schedule_node_band_fine_scale(const isl::schedule_node_band &band, const std::string &name,
                                                          unsigned int dimension, const isl::val &value);
isl::schedule_node_band isl_schedule_node_band_fine_scale_down(const isl::schedule_node_band &band,
                                                               const std::string &name, unsigned int dimension,
                                                               const isl::val &value);

////////////////////////////////////////////////////////////////////////////////
// schedule tree transformations (on a schedule_node)
////////////////////////////////////////////////////////////////////////////////

// Merge two nested isl_schedule_node_band.
// Assuming the input schedule_node_band has only one child and this child is an isl_schedule_node_band.
isl::schedule_node_band isl_schedule_node_band_unsplit(const isl::schedule_node_band &band);

// Call isl_schedule_node_band_unsplit() until it is not possible
isl::schedule_node_band isl_schedule_node_band_fully_unsplit(const isl::schedule_node_band &band);

// Call isl_schedule_node_band_split() until it is not possible
isl::schedule_node isl_schedule_node_band_fully_split(const isl::schedule_node_band &band);

////////////////////////////////////////////////////////////////////////////////
// isl_schedule transformations
////////////////////////////////////////////////////////////////////////////////

isl::schedule isl_schedule_collapse(const isl::schedule &schedule, unsigned int first, unsigned int last);

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
