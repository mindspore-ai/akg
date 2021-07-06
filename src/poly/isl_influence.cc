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
#include "poly/isl_influence.h"

// ISL
#include <isl/aff.h>
#include <isl/mat.h>
#include <isl/hash.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <isl/constraint.h>
#include <isl/map_to_basic_set.h>

// ISL private headers
#include <isl_int.h>
#include <isl_tab.h>
extern "C" {
#include <isl_aff_private.h>
#include <isl_mat_private.h>
#include <isl_schedule_constraints.h>
}

///////////////////////////////////////////////////////////////////////////
// CODE
///////////////////////////////////////////////////////////////////////////

namespace akg {
namespace ir {
namespace poly {

void akg_isl_influence_enable(isl_ctx *ctx) {
  isl_options_set_akg_print_debug(ctx, 1);
  isl_options_set_akg_influence_scheduler(ctx, 1);
  isl_influence_enabled = 1;
}

void akg_isl_influence_disable(isl_ctx *ctx) {
  isl_options_set_akg_print_debug(ctx, 0);
  isl_options_set_akg_influence_scheduler(ctx, 0);
  isl_influence_enabled = 0;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
