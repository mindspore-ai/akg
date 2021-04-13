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
#include "poly/schedule_pass/constrain_schedule.h"

#include "poly/isl_util.h"
#include "poly/log_util.h"

namespace akg {
namespace ir {
namespace poly {

////////////////////////////////////////////////////////////////////////////////
// local declarations
////////////////////////////////////////////////////////////////////////////////

#ifndef _AKG_DUMMY_DATE_STRING
#define _AKG_DUMMY_DATE_STRING "_akg_dummy_date"
#endif

static const char *const _akg_dummy_date_string = _AKG_DUMMY_DATE_STRING;

// This should be changed later.
static int _verbosity = 2;

__isl_give isl_schedule *isl_schedule_constraints_silently_compute_schedule(
  __isl_take isl_schedule_constraints *constraints);
__isl_give isl_stat map_statement_to_dummy_date(__isl_take isl_map *map, void *user);
__isl_give isl_union_map *isl_map_domain_to_dummy_date(__isl_take isl_union_set *domain);
__isl_give isl_union_map *isl_schedule_get_temporal_accesses(__isl_keep isl_schedule *schedule);
__isl_give isl_union_map *isl_schedule_get_temporal_dependences(__isl_keep isl_schedule *schedule);
__isl_give isl_schedule *isl_schedule_compute_verifying_schedule(
  __isl_keep isl_schedule *const schedule, __isl_keep isl_schedule_constraints *const initial_constraints);
__isl_give isl_stat isl_schedule_check(__isl_keep isl_schedule *const schedule,
                                       __isl_keep isl_schedule_constraints *const initial_constraints);

////////////////////////////////////////////////////////////////////////////////
// Schedule checking functions in ConstrainSchedule public API
////////////////////////////////////////////////////////////////////////////////

bool ConstrainSchedule::CheckSchedule(const isl::schedule &sch) const {
  isl_schedule_constraints *const constraints = pass_info_.constraints_.get();
  isl_schedule *const schedule = sch.get();

  _verbosity = verbosity_;
  const isl_stat status = isl_schedule_check(schedule, constraints);

  return status == isl_stat_ok;
}

////////////////////////////////////////////////////////////////////////////////
// local definitions
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Silent wrapper for isl_schedule_constraints_compute_schedule().
 */
__isl_give isl_schedule *isl_schedule_constraints_silently_compute_schedule(
  __isl_take isl_schedule_constraints *const constraints) {
  akg::ir::poly::log::Info(log::Verbosity::high, "silent constraints:\n" + to_block_string(constraints));

  isl_ctx *const ctx = isl_schedule_constraints_get_ctx(constraints);
  const int previous_behaviour = isl_options_get_on_error(ctx);
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  isl_schedule *const result = isl_schedule_constraints_compute_schedule(constraints);
  isl_options_set_on_error(ctx, previous_behaviour);

  return result;
}

/**
 * \brief Callback for isl_map_list_foreach(); sets the out dim to _dummy_date.
 */
__isl_give isl_stat map_statement_to_dummy_date(__isl_take isl_map *map, void *const user) {
  // The output dimension in the input map is empty.
  // We want it to be '_akg_dummy_date[0]':
  map = isl_map_insert_dims(map, isl_dim_out, 0, 1);
  map = isl_map_fix_si(map, isl_dim_out, 0, 0);
  map = isl_map_set_tuple_name(map, isl_dim_out, _akg_dummy_date_string);

  isl_union_map **result = (isl_union_map **)user;
  if (!*result) {
    *result = isl_union_map_from_map(map);
  } else {
    *result = isl_union_map_add_map(*result, map);
  }

  return isl_stat_ok;
}

/**
 * \brief Map statements to dummy dates
 */
__isl_give isl_union_map *isl_map_domain_to_dummy_date(__isl_take isl_union_set *const domain) {
  isl_union_map *start = isl_union_map_from_domain(domain);
  isl_map_list *list = isl_union_map_get_map_list(start);
  isl_union_map *result = NULL;
  isl_map_list_foreach(list, map_statement_to_dummy_date, &result);

  isl_union_map_free(start);
  isl_map_list_free(list);

  return result;
}

__isl_give isl_union_map *isl_schedule_get_temporal_accesses(__isl_keep isl_schedule *const schedule) {
  isl_union_set *const domain = isl_schedule_get_domain(schedule);
  return isl_map_domain_to_dummy_date(domain);
}

__isl_give isl_union_map *isl_schedule_get_temporal_dependences(__isl_keep isl_schedule *const schedule) {
  /* Prepare inputs for the union_access_info. */
  isl_union_map *const sink = isl_schedule_get_temporal_accesses(schedule);
  isl_union_map *const source = isl_union_map_copy(sink);
  isl_schedule *const schedule_copy = isl_schedule_copy(schedule);

  /* Build the union_access_info. */
  isl_union_access_info *accesses = isl_union_access_info_from_sink(sink);
  accesses = isl_union_access_info_set_schedule(accesses, schedule_copy);
  accesses = isl_union_access_info_set_must_source(accesses, isl_union_map_copy(source));
  accesses = isl_union_access_info_set_kill(accesses, source);

  /* Compute the flow. */
  isl_union_flow *const flow = isl_union_access_info_compute_flow(accesses);

  /* Extract must dependences. */
  isl_union_map *const dependences = isl_union_flow_get_may_dependence(flow);
  isl_union_flow_free(flow);

  return dependences;

  // isl_union_map* const dependences = isl_union_flow_get_full_may_dependence(flow);
  // isl_union_flow_free(flow);

  // /*
  //  * At this point, our maps (for instance, with S0 and S1) will be in form:
  //  *   { S0[...] -> [S1[...] -> _akg_dummy_date[0]]: ...; ... }
  //  * We want maps in the form:
  //  *   { S0[...] -> S1[...]: ...; ... }
  //  * We need to:
  //  * - uncurry the nested relation of the "range" of the map
  //  *   (turning the "domain" of the map into a nested relation)
  //  * - unwrap the nested relation of the "domain" of the map
  //  */

  // /* Uncurry, ditch the "new" range and unwrap our dummy dependences. */
  // isl_union_map* const uncurried = isl_union_map_uncurry(dependences);
  // isl_union_set* const domain = isl_union_map_domain(uncurried);
  // isl_union_map* const unwrapped = isl_union_set_unwrap(domain);

  // return unwrapped;
}

/**
 * \brief Check a new schedule against previous constraints
 *
 * Bonus: can also be used to add permutable/coincident metadata and "reshape"
 * a schedule tree (initial_constraints may be a null pointer).
 */
__isl_give isl_schedule *isl_schedule_compute_verifying_schedule(
  __isl_keep isl_schedule *const schedule, __isl_keep isl_schedule_constraints *const initial_constraints) {
  /*
   * Principle:
   * 1. extract "date" constraints from the proposed schedule
   * 2. combine the date constraints with the initial constraints
   * 3. attempt to schedule:
   *    - a schedule can be computed from the combined constraints:
   *      the proposed schedule is valid
   *    - no schedule can be computed from the combined constraints:
   *      the proposed schedule violates the initial constraints
   */
  isl_union_map *const dates = isl_schedule_get_temporal_dependences(schedule);

  /* Some logging. */
  if (initial_constraints)
    akg::ir::poly::log::Info(log::Verbosity::high, "initial constraints:\n" + to_block_string(initial_constraints));
  else
    akg::ir::poly::log::Info(log::Verbosity::high, "initial constraints: none");
  akg::ir::poly::log::Info(log::Verbosity::high, "dates:\n" + to_block_string(dates));

  isl_schedule_constraints *constraints = NULL;
  if (!initial_constraints) {
    /*
     * Hack/Easter egg: the function will be used to "reshape" the schedule tree
     */

    /* Build new schedule constraints. */
    isl_union_set *const domain = isl_schedule_get_domain(schedule);
    constraints = isl_schedule_constraints_on_domain(domain);
    constraints = isl_schedule_constraints_set_validity(constraints, dates);
  } else {
    /* Combine the schedule constraints. */
    constraints = isl_schedule_constraints_copy(initial_constraints);
    isl_union_map *const validity = isl_schedule_constraints_get_validity(constraints);
    isl_space *const space = isl_union_map_get_space(dates);
    isl_union_map *const aligned = isl_union_map_align_params(validity, space);

    isl_union_map *const restricted = isl_union_map_union(aligned, dates);
    constraints = isl_schedule_constraints_set_validity(constraints, restricted);
  }

  /* Finally attempt to schedule. */
  isl_schedule *const result = isl_schedule_constraints_silently_compute_schedule(constraints);

  return result;
}

__isl_give isl_stat isl_schedule_check(__isl_keep isl_schedule *const schedule,
                                       __isl_keep isl_schedule_constraints *const initial_constraints) {
  isl_schedule *const result = isl_schedule_compute_verifying_schedule(schedule, initial_constraints);

  /* Check whether we managed to schedule something. */
  if (result) {
    akg::ir::poly::log::Info(log::Verbosity::high, text_blue "schedule seems valid\n" + to_block_string(result));
    isl_schedule_free(result);
    return isl_stat_ok;
  } else {
    akg::ir::poly::log::Warn(log::Verbosity::veryLow, "schedule is invalid");
    return isl_stat_error;
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
