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
#ifndef POLY_ISL_INFLUENCE_H_
#define POLY_ISL_INFLUENCE_H_

// STL
#include <tuple>
#include <vector>

// ISL
#include <isl/cpp.h>

namespace akg {
namespace ir {
namespace poly {

///////////////////////////////////////////////////////////////////////////
// Typedefs for soft constraints
///////////////////////////////////////////////////////////////////////////

#define INF_BLOCK_SIZE 8
enum isl_influence_coeff_type { isl_cst, isl_param, isl_var };
struct isl_influence_equal {
  char *statement1;
  char *statement2;
  int sched_dim;
  int sched_dim1;
  int sched_dim2;
  isl_influence_coeff_type type;
  int coef_dim1;
  int coef_dim2;
  int scanned;
};
typedef struct isl_influence_equal isl_influence_equal;

struct isl_influence_equal_list {
  int size;
  int mem;
  struct isl_influence_equal *data;
};
typedef struct isl_influence_equal_list isl_influence_equal_list;

struct isl_influence {
  char *statement_name;
  isl_influence_coeff_type type;
  int sched_dim;
  int coef_dim;
  int val;
  int scanned;
};
typedef struct isl_influence isl_influence;

struct isl_influence_list {
  int size;
  int mem;
  struct isl_influence *data;
};
typedef struct isl_influence_list isl_influence_list;

struct isl_influence_sol {
  int size;
  int mem;
  int *data;
};
typedef struct isl_influence_sol isl_influence_sol;

struct isl_influence_sol_list {
  int size;
  int mem;
  isl_influence_sol *data;
};
typedef struct isl_influence_sol_list isl_influence_sol_list;

void akg_isl_influence_enable(isl_ctx *ctx);
void akg_isl_influence_disable(isl_ctx *ctx);

int akg_isl_influence_maxvar(struct isl_sched_graph *graph);

isl_basic_set *akg_isl_influence_set_coef(isl_ctx *ctx, struct isl_sched_graph *graph, isl_basic_set *bset);
isl_basic_set *akg_isl_influence_set_equal(isl_ctx *ctx, struct isl_sched_graph *graph, isl_basic_set *bset);

int akg_isl_influence_check_coincident(struct isl_sched_graph *graph, isl_vec *sol);

__isl_give isl_schedule *akg_isl_schedule_constraints_compute_schedule_influence(
  __isl_take isl_schedule_constraints *sc, isl_influence_list *inf_coef, isl_influence_equal_list *inf_equal);

void *isl_influence_list_free(isl_influence_list *inf_list);
void *isl_influence_equal_list_free(isl_influence_equal_list *inf_equal_list);
struct isl_sched_graph *akg_isl_influence_sol_list_free(struct isl_sched_graph *graph);
struct isl_sched_graph *akg_isl_influence_sol_add_elem(isl_vec *sol, struct isl_sched_graph *graph);
int akg_isl_influence_sol_get_elem(int sched, int pos, struct isl_sched_graph *graph);

}  // namespace poly
}  // namespace ir
}  // namespace akg

#endif  // POLY_ISL_INFLUENCE_H_
