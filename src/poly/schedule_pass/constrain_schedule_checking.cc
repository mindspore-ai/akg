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
#ifdef AKG_USE_POLYTOPS

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

isl::schedule isl_schedule_constraints_silently_compute_schedule(const isl::schedule_constraints &constraints);
__isl_give isl_stat map_statement_to_dummy_date(__isl_take isl_map *map, void *user);
isl::union_map isl_map_domain_to_dummy_date(const isl::union_set &domain);
isl::union_map isl_schedule_get_temporal_accesses(const isl::schedule &schedule);
isl::union_map isl_schedule_get_temporal_dependences(const isl::schedule &schedule);
isl::schedule isl_schedule_compute_verifying_schedule(const isl::schedule &schedule,
                                                      const isl::schedule_constraints &constraints);
static __isl_give isl_stat isl_schedule_check(const isl::schedule &schedule,
                                              const isl::schedule_constraints &constraints);

////////////////////////////////////////////////////////////////////////////////
// Schedule checking functions in ConstrainSchedule public API
////////////////////////////////////////////////////////////////////////////////

bool ConstrainSchedule::CheckSchedule(const isl::schedule &sch) const {
  _verbosity = verbosity_;
  const isl_stat status = isl_schedule_check(sch, pass_info_.constraints_);

  return status == isl_stat_ok;
}

////////////////////////////////////////////////////////////////////////////////
// local definitions
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Silent wrapper for isl_schedule_constraints_compute_schedule().
 */
isl::schedule isl_schedule_constraints_silently_compute_schedule(const isl::schedule_constraints &constraints) {
  log::Info(log::Verbosity::high, "silent constraints:\n" + to_block_string(constraints));

  isl_ctx *const ctx = constraints.ctx().get();
  const int previous_behaviour = isl_options_get_on_error(ctx);
  isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
  const isl::schedule &result = constraints.compute_schedule();
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
isl::union_map isl_map_domain_to_dummy_date(const isl::union_set &domain) {
  const isl::union_map &domain_map = isl::union_map::from_domain(domain);
  const isl::map_list &list = domain_map.map_list();

  isl_union_map *result = NULL;
  isl_map_list_foreach(list.get(), map_statement_to_dummy_date, &result);

  return isl::manage(result);
}

isl::union_map isl_schedule_get_temporal_accesses(const isl::schedule &schedule) {
  const isl::union_set &domain = schedule.domain();
  return isl_map_domain_to_dummy_date(domain);
}

isl::union_map isl_schedule_get_temporal_dependences(const isl::schedule &schedule) {
  /* Prepare inputs for the union_access_info. */
  const isl::union_map &sink = isl_schedule_get_temporal_accesses(schedule);
  const isl::union_map &source = sink;

  /* Build the union_access_info. */
  isl::union_access_info accesses = isl::union_access_info(sink);
  accesses = accesses.set_schedule(schedule);
  accesses = accesses.set_must_source(source);
  accesses = accesses.set_kill(source);

  const isl::union_flow &flow = accesses.compute_flow();
  const isl::union_map &dependences = flow.may_dependence();

  return dependences;
}

/**
 * \brief Check a new schedule against previous constraints
 *
 * Bonus: can also be used to add permutable/coincident metadata and "reshape"
 * a schedule tree (initial_constraints may be a null pointer).
 */
isl::schedule isl_schedule_compute_verifying_schedule(const isl::schedule &schedule,
                                                      const isl::schedule_constraints &initial) {
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
  const isl::union_map &dates = isl_schedule_get_temporal_dependences(schedule);

  /* Some logging. */
  if (initial)
    log::Info(log::Verbosity::high, "initial constraints:\n" + to_block_string(initial));
  else
    log::Info(log::Verbosity::high, "initial constraints: none");
  log::Info(log::Verbosity::high, "dates:\n" + to_block_string(dates));

  isl::schedule_constraints constraints;
  if (!initial) {
    /*
     * Hack/Easter egg: the function will be used to "reshape" the schedule tree
     */

    /* Build new schedule constraints. */
    const isl::union_set &domain = schedule.domain();
    constraints = isl::schedule_constraints::on_domain(domain);
    constraints = constraints.set_validity(dates);
  } else {
    /* Combine the schedule constraints. */
    constraints = initial;
    const isl::union_map &validity = constraints.validity();
    const isl::space &space = dates.space();
    const isl::union_map &aligned = isl_union_map_align_params(validity, space);
    const isl::union_map &restricted = aligned.unite(dates);
    constraints = constraints.set_validity(restricted);
  }

  /* Finally attempt to schedule. */
  const isl::schedule &result = isl_schedule_constraints_silently_compute_schedule(constraints);

  return result;
}

static __isl_give isl_stat isl_schedule_check(const isl::schedule &schedule,
                                              const isl::schedule_constraints &constraints) {
  const isl::schedule &result = isl_schedule_compute_verifying_schedule(schedule, constraints);

  /* Check whether we managed to schedule something. */
  if (result) {
    log::Info(log::Verbosity::high, text_blue "schedule seems valid\n" + to_block_string(result));
    return isl_stat_ok;
  } else {
    log::Warn(log::Verbosity::veryLow, "schedule is invalid");
    return isl_stat_error;
  }
}

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // AKG_USE_POLYTOPS
