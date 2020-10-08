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

#include "poly/log_util.h"

struct isl_sched_graph {
  isl_map_to_basic_set *intra_hmap;
  isl_map_to_basic_set *intra_hmap_param;
  isl_map_to_basic_set *inter_hmap;

  struct isl_sched_node *node;
  int n;
  int maxvar;
  int max_row;
  int n_row;

  int *sorted;

  int n_total_row;
  int band_start;

  struct isl_sched_graph *root;

  struct isl_sched_edge *edge;
  int n_edge;
  int max_edge[isl_edge_last + 1];
  struct isl_hash_table *edge_table[isl_edge_last + 1];

  struct isl_hash_table *node_table;
  struct isl_trivial_region *region;

  isl_basic_set *lp;

  int src_scc;
  int dst_scc;

  int scc;
  int weak;

  int max_weight;
  /* AKG isl_influence patch - start */
  akg::ir::poly::isl_influence_list *inf_list;
  akg::ir::poly::isl_influence_equal_list *inf_equal_list;
  akg::ir::poly::isl_influence_sol_list *inf_sol_list;
  /* AKG isl_influence patch - end */
};

///////////////////////////////////////////////////////////////////////////
// CODE
///////////////////////////////////////////////////////////////////////////

namespace akg {
namespace ir {
namespace poly {

static inline void akg_isl_influence_set_function_pointers(void) {
  isl_influence_set_coef = akg_isl_influence_set_coef;
  isl_influence_set_equal = akg_isl_influence_set_equal;
  isl_influence_maxvar = akg_isl_influence_maxvar;
  isl_influence_check_coincident = akg_isl_influence_check_coincident;
  isl_influence_sol_list_free = akg_isl_influence_sol_list_free;
  isl_influence_sol_add_elem = akg_isl_influence_sol_add_elem;
  isl_influence_sol_get_elem = akg_isl_influence_sol_get_elem;
}

static inline void akg_isl_influence_unset_function_pointers(void) {
  isl_influence_set_coef = 0;
  isl_influence_set_equal = 0;
  isl_influence_maxvar = 0;
  isl_influence_check_coincident = 0;
  isl_influence_sol_list_free = 0;
  isl_influence_sol_add_elem = 0;
  isl_influence_sol_get_elem = 0;
}

void akg_isl_influence_enable(isl_ctx *ctx) {
  isl_options_set_akg_print_debug(ctx, 1);
  isl_options_set_akg_influence_scheduler(ctx, 1);
  akg_isl_influence_set_function_pointers();
}

void akg_isl_influence_disable(isl_ctx *ctx) {
  isl_options_set_akg_print_debug(ctx, 0);
  isl_options_set_akg_influence_scheduler(ctx, 0);
  akg_isl_influence_unset_function_pointers();
}

static const char *inf_types_str[] = {
  [isl_cst] = "isl_cst",
  [isl_param] = "isl_param",
  [isl_var] = "isl_var",
};

struct isl_sched_graph *akg_isl_influence_sol_list_free(isl_sched_graph *graph) {
  log::Info(log::Verbosity::high, "Entering isl_influence_sol_list_free");

  isl_influence_sol_list *inf = graph->inf_sol_list;
  if (inf) {
    log::Info(log::Verbosity::high, "inf-mem: " + std::to_string(inf->mem));
    for (int i = 0; i < inf->mem; i++) {
      isl_influence_sol *p = &inf->data[i];
      p->mem = 0;
      p->size = 0;
      free(p->data);
      p->data = NULL;
    }
    free(inf->data);
    inf->data = NULL;
    inf->size = 0;
    inf->mem = 0;
    free(inf);
  }
  graph->inf_sol_list = NULL;
  log::Info(log::Verbosity::high, "Leaving isl_influence_sol_list_free");
  return graph;
}
struct isl_sched_graph *akg_isl_influence_sol_add_elem(isl_vec *sol, struct isl_sched_graph *graph) {
  log::Info(log::Verbosity::high, "Entering isl_influence_sol_add_elem");

  if (!sol) {
    return graph;
  }

  isl_influence_sol_list *inf = graph->inf_sol_list;
  if (!inf) {
    inf = (isl_influence_sol_list *)calloc(1, sizeof(isl_influence_sol_list));
    if (!inf) {
      log::Info(log::Verbosity::high,
                "MEM ERROR: isl_influence_sol_add_elem could not allocate memory for isl_influence_sol_list");
      return graph;
    }
    inf->mem = 0;
    inf->size = 0;
    inf->data = NULL;
  }

  if (inf->mem == 0) {
    inf->mem = INF_BLOCK_SIZE;
    inf->data = (isl_influence_sol *)calloc(inf->mem, sizeof(isl_influence_sol));
    if (inf->data == NULL) {
      log::Info(log::Verbosity::high,
                "MEM ERROR: isl_influence_sol_add_elem could not allocate memory for isl_influence_sol_list");
      return graph;
    }
  }

  if (inf->mem == inf->size) {
    inf->mem += INF_BLOCK_SIZE;
    inf->data = (isl_influence_sol *)realloc(inf->data, inf->mem * sizeof(isl_influence_sol));
    if (inf->data == NULL) {
      log::Info(log::Verbosity::high,
                "MEM ERROR: isl_influence_sol_add_elem could not reallocate memory isl_influence_sol");
      return graph;
    }
  }

  isl_influence_sol *p = &inf->data[inf->size];
  const int sol_size = isl_inf_vec_get_size(sol);
  p->data = (int *)calloc((size_t)sol_size, sizeof(int));

  if (p->data == NULL) {
    log::Info(log::Verbosity::high, "MEM ERROR: isl_influence_sol_add_elem could not allocate memory");
    return graph;
  }

  p->mem = sol_size;
  isl_ctx *ctx = isl_vec_get_ctx(sol);

  for (int i = 0; i < p->mem; i++) {
    isl_printer *printer = isl_printer_to_str(ctx);
    isl_val *const val = isl_influence_vec_get_elem(sol, i);

    printer = isl_printer_print_val(printer, val);
    char *const str = isl_printer_get_str(printer);

    const int value = atoi(isl_printer_get_str(printer));
    p->data[i] = value;
    p->size++;

    free(str);
    isl_printer_free(printer);
    isl_val_free(val);
  }

  inf->size++;

  log::Info(log::Verbosity::high, "inf->mem: " + std::to_string(inf->mem));
  log::Info(log::Verbosity::high, "inf->size: " + std::to_string(inf->size));

  log::Info(log::Verbosity::high, "Leaving isl_influence_sol_add_elem");
  graph->inf_sol_list = inf;
  return graph;
}

int akg_isl_influence_sol_get_elem(int dim, int pos, struct isl_sched_graph *graph) {
  isl_influence_sol_list *inf = graph->inf_sol_list;

  if (NULL == inf) {
    return -1;
  }

  isl_influence_sol *p = NULL;
  int retval;

  if (dim < inf->size) {
    p = &inf->data[dim];
  } else {
    std::string message = "ERROR: isl_influence_sol_get_elem index out of range isl_influence_sol_list size : ";
    message += std::to_string(inf->size) + " < dim: " + std::to_string(dim);
    log::Info(log::Verbosity::high, message);
    retval = -1;
  }

  if (1 + pos < p->size) {
    retval = p->data[1 + pos];
  } else {
    std::string message = "ERROR: isl_influence_sol_get_elem index out of range isl_ifnluence_sol size: ";
    message += std::to_string(p->size) + " < pos: " + std::to_string(pos);
    retval = -1;
  }

  return retval;
}

void print_basic_set(isl_basic_set *set, const char *str) {
  isl_ctx *const ctx = isl_basic_set_get_ctx(set);
  const bool print_debug = isl_options_get_akg_print_debug(ctx);
  if (print_debug) {
    isl_printer *printer = isl_printer_to_str(ctx);
    printer = isl_printer_print_basic_set(printer, set);
    char *const printed_str = isl_printer_get_str(printer);

    log::Info(log::Verbosity::high, str);
    log::Info(log::Verbosity::high, printed_str);

    isl_printer_free(printer);
    free(printed_str);
  }
}

void *isl_influence_equal_list_free(isl_influence_equal_list *inf_equal_list) {
  if (inf_equal_list->data) {
    for (int i = 0; i < inf_equal_list->size; ++i) {
      free(inf_equal_list->data[i].statement1);
      free(inf_equal_list->data[i].statement2);
    }
    free(inf_equal_list->data);
    inf_equal_list->data = NULL;
    inf_equal_list->mem = 0;
    inf_equal_list->size = 0;
  }
  free(inf_equal_list);
  return NULL;
}

void *isl_influence_list_free(isl_influence_list *inf_list) {
  if (inf_list->data) {
    for (int i = 0; i < inf_list->size; ++i) {
      free(inf_list->data[i].statement_name);
    }
    free(inf_list->data);
    inf_list->data = NULL;
    inf_list->mem = 0;
    inf_list->size = 0;
  }
  free(inf_list);
  return NULL;
}

isl_basic_set *hack_coefficients(isl_basic_set *coef, const char *msg, int pos, int lb, int ub) {
  isl_ctx *const ctx = isl_basic_set_get_ctx(coef);

  isl_val *const v_ub = isl_val_int_from_si(ctx, ub);
  isl_val *const v_lb = isl_val_int_from_si(ctx, lb);

  isl_size dim = isl_basic_set_n_dim(coef);
  log::Info(log::Verbosity::high, "pos: " + std::to_string(pos));

  if (pos < (int)dim) {
    const bool influence_schedule = isl_options_get_akg_influence_scheduler(ctx);
    if (influence_schedule) {
      coef = isl_basic_set_upper_bound_val(coef, isl_dim_set, pos, isl_val_copy(v_ub));
      coef = isl_basic_set_lower_bound_val(coef, isl_dim_set, pos, isl_val_copy(v_lb));

      std::string message = " -> i" + std::to_string(pos);
      message += " = (" + std::to_string(lb);
      message += ", " + std::to_string(ub) + ")";
      log::Info(log::Verbosity::high, message);
    }
  }

  isl_val_free(v_ub);
  isl_val_free(v_lb);

  return coef;
}

isl_basic_set *create_constraint(isl_basic_set *coef, const char *msg, int pos1, int pos2) {
  isl_ctx *ctx = isl_basic_set_get_ctx(coef);
  const bool influence_schedule = isl_options_get_akg_influence_scheduler(ctx);
  if (influence_schedule) {
    isl_local_space *ls = isl_basic_set_get_local_space(coef);
    isl_constraint *c = isl_constraint_alloc_equality(isl_local_space_copy(ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_set, pos1, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_set, pos2, -1);
    coef = isl_basic_set_add_constraint(coef, c);
    print_basic_set(coef, msg);

    log::Info(log::Verbosity::high, "--> i" + std::to_string(pos1) + " = i" + std::to_string(pos2));
    isl_local_space_free(ls);
  }

  return coef;
}

isl_basic_set *graph_find_basic_set_by_statement_name(struct isl_sched_graph *graph, const char *name,
                                                      int *node_index) {
  isl_basic_set *bset = NULL;

  for (int i = 0; i < graph->n && bset == NULL; ++i) {
    struct isl_sched_node *node = isl_sched_graph_get_node(graph, i);
    isl_map *ma = isl_sched_node_extract_schedule(node);
    const char *strstat = isl_map_get_tuple_name(ma, isl_dim_in);

    if (strcmp(strstat, name) == 0) {
      bset = graph->lp;
      *node_index = i;
    }

    isl_map_free(ma);
  }
  return bset;
}

int get_pos_from_bset(isl_basic_set *bset, struct isl_sched_node *node, isl_influence_equal *inf, int coef_dim) {
  int pos = -1;

  switch (inf->type) {
    case isl_cst:
      pos = isl_sched_node_cst_coef_offset(node);
      break;
    case isl_param:
      pos = isl_sched_node_par_coef_offset(node) - coef_dim;
      break;
    case isl_var:
      pos = isl_sched_node_par_coef_offset(node) - isl_sched_node_get_nparam(node) - coef_dim * 2 - 1;
      break;
    default:
      break;
  }

  log::Info(log::Verbosity::high,
            "coefficient position for coef_dim=" + std::to_string(coef_dim) + ": " + std::to_string(pos));

  return pos;
}

void print_influence(isl_influence_equal_list *equal, isl_influence_list *coef) {
  log::Info(log::Verbosity::high, "start printing isl_influence hard constraints");
  for (int i = 0; coef && i < coef->size; i++) {
    isl_influence *inf = &coef->data[i];
    log::Info(log::Verbosity::high,
              "hard constraint\t\t[+" + std::to_string(i + 1) + ":" + std::to_string(coef->size) + "]");
    log::Info(log::Verbosity::high, "statement:\t\t" + std::string(inf->statement_name));
    log::Info(log::Verbosity::high, "type:\t\t\t" + std::string(inf_types_str[(int)inf->type]));
    log::Info(log::Verbosity::high, "sched_dim:\t\t" + std::to_string(inf->sched_dim));
    log::Info(log::Verbosity::high, "coef_dim:\t\t" + std::to_string(inf->coef_dim));
    log::Info(log::Verbosity::high, "val:\t\t\t" + std::to_string(inf->val));
  }
  log::Info(log::Verbosity::high, "End printing isl_influence hard constraints");

  log::Info(log::Verbosity::high, "Start printing isl_influence soft constraints");
  for (int i = 0; equal && i < equal->size; i++) {
    isl_influence_equal *inf_equal = &equal->data[i];
    log::Info(log::Verbosity::high,
              "soft constraint\t\t[" + std::to_string(i + 1) + ":" + std::to_string(equal->size) + "]");
    log::Info(log::Verbosity::high, "statement1:\t\t" + std::string(inf_equal->statement1));
    log::Info(log::Verbosity::high, "statement2:\t\t" + std::string(inf_equal->statement2));
    log::Info(log::Verbosity::high, "sched_dim1:\t\t" + std::to_string(inf_equal->sched_dim1));
    log::Info(log::Verbosity::high, "sched_dim2:\t\t" + std::to_string(inf_equal->sched_dim2));
    log::Info(log::Verbosity::high, "type:\t\t\t" + std::string(inf_types_str[(int)inf_equal->type]));
    log::Info(log::Verbosity::high, "coef_dim1:\t\t" + std::to_string(inf_equal->coef_dim1));
    log::Info(log::Verbosity::high, "coef_dim2:\t\t" + std::to_string(inf_equal->coef_dim2));
  }
  log::Info(log::Verbosity::high, "End  printing isl_influence soft constraints");
}

int report_influence(isl_influence_equal_list *equal, isl_influence_list *coef, isl_influence_sol_list *sol,
                     int maxvar) {
  int bad_equal = 0;
  int bad_coef = 0;
  int result = 1;
  for (int i = 0; i < coef->size; i++) {
    isl_influence *inf = &coef->data[i];
    if (inf->scanned != 1) {
      log::Info(log::Verbosity::high,
                "warning: influence hard constraint [" + std::to_string(i) + "] was not processed");
      log::Info(log::Verbosity::high, "statement:\t\t" + std::string(inf->statement_name));
      log::Info(log::Verbosity::high, "type:\t\t\t" + std::string(inf_types_str[(int)inf->type]));
      log::Info(log::Verbosity::high, "sched_dim:\t\t" + std::to_string(inf->sched_dim));
      log::Info(log::Verbosity::high, "coef_dim:\t\t" + std::to_string(inf->coef_dim));
      log::Info(log::Verbosity::high, "val:\t\t\t" + std::to_string(inf->val));
      bad_coef++;
      result = 0;
    }
  }

  for (int i = 0; i < equal->size; i++) {
    isl_influence_equal *inf_equal = &equal->data[i];
    if (inf_equal->scanned != 1) {
      log::Info(log::Verbosity::high,
                "warning: influence soft constraint [" + std::to_string(i) + "] was not processed");
      log::Info(log::Verbosity::high, "statement1:\t\t" + std::string(inf_equal->statement1));
      log::Info(log::Verbosity::high, "statement2:\t\t" + std::string(inf_equal->statement2));
      log::Info(log::Verbosity::high, "sched_dim1:\t\t" + std::to_string(inf_equal->sched_dim1));
      log::Info(log::Verbosity::high, "sched_dim2:\t\t" + std::to_string(inf_equal->sched_dim2));
      log::Info(log::Verbosity::high, "type:\t\t\t" + std::string(inf_types_str[(int)inf_equal->type]));
      log::Info(log::Verbosity::high, "coef_dim1:\t\t" + std::to_string(inf_equal->coef_dim1));
      log::Info(log::Verbosity::high, "coef_dim2:\t\t" + std::to_string(inf_equal->coef_dim2));
      bad_equal++;
      result = 0;
    }
  }

  if (bad_coef == 0)
    log::Info(log::Verbosity::high, std::to_string(coef->size) + "influence hard coef constraints processed correctly");
  else
    log::Info(log::Verbosity::high, std::to_string(bad_coef) + "influence hard coef constraints were not processed");

  if (bad_equal == 0)
    log::Info(log::Verbosity::high, std::to_string(equal->size) + "influence equal constraints processed correctly");
  else
    log::Info(log::Verbosity::high, std::to_string(bad_equal) + "influence equal constraints were not processed");

  if (!sol || sol->size != maxvar) {
    log::Info(log::Verbosity::high, "isl influence could not find solution for all dimensions");
    result = 0;
  }

  return result;
}

int set_params(isl_influence_equal *inf, int *sched_from, int *sched_to, int *coef_from, int *coef_to, int actual_dim) {
  int retval = 0;

  if (actual_dim == 0 && inf->sched_dim1 != inf->sched_dim2) {
    return retval;
  } else if (actual_dim == inf->sched_dim1) {
    if (inf->sched_dim1 >= inf->sched_dim2) {
      *sched_from = inf->sched_dim2;
      *sched_to = inf->sched_dim1;
      *coef_from = inf->coef_dim2;
      *coef_to = inf->coef_dim1;
      retval = 1;

    } else {
      log::Info(log::Verbosity::high,
                "cannot set future coef for dimension: " + std::to_string(actual_dim) + " and inf_equal:");
      log::Info(log::Verbosity::high, "inf->sched_dim1: " + std::to_string(inf->sched_dim1));
      log::Info(log::Verbosity::high, "inf->sched_dim2: " + std::to_string(inf->sched_dim2));
    }

  } else if (actual_dim == inf->sched_dim2) {
    if (inf->sched_dim2 >= inf->sched_dim1) {
      *sched_from = inf->sched_dim1;
      *sched_to = inf->sched_dim2;
      *coef_from = inf->coef_dim1;
      *coef_to = inf->coef_dim2;
      retval = 1;
    } else {
      log::Info(log::Verbosity::high,
                "cannot set future coef for dimension: " + std::to_string(actual_dim) + " and inf_equal:");
      log::Info(log::Verbosity::high, "inf->sched_dim1: " + std::to_string(inf->sched_dim1));
      log::Info(log::Verbosity::high, "inf->sched_dim2: " + std::to_string(inf->sched_dim2));
    }
  }

  return retval;
}
isl_basic_set *akg_isl_influence_set_equal(isl_ctx *ctx, struct isl_sched_graph *graph, isl_basic_set *bset) {
  // loop over iinfluence list equal
  log::Info(log::Verbosity::high, "Enter isl_influence_set_equal for dimension --> " + std::to_string(graph->n_row));
  isl_influence_equal_list *inf_list = graph->inf_equal_list;
  for (int i = 0; i < inf_list->size; i++) {
    isl_influence_equal *inf_equal = &inf_list->data[i];

    if (inf_equal->sched_dim1 > inf_equal->sched_dim2) {
      std::string message0 = "ERROR: isl cannot compute soft constraint";
      message0 += "[" + std::to_string(i) + " from  dimension " + std::to_string(inf_equal->sched_dim2) +
                  " to dimension " + std::to_string(inf_equal->sched_dim1);
      std::string message1 = "Reason: destination coefficient unknown";
      log::Info(log::Verbosity::high, message0);
      log::Info(log::Verbosity::high, message1);
    }
    int sched_from;
    int sched_to;
    int coef_from;
    int coef_to;
    if (!set_params(inf_equal, &sched_from, &sched_to, &coef_from, &coef_to, graph->n_row)) continue;

    isl_basic_set *bset_from;
    isl_basic_set *bset_to;

    int node_index_from = -1;
    int node_index_to = -1;

    bset_from = graph_find_basic_set_by_statement_name(graph, inf_equal->statement1, &node_index_from);
    bset_to = graph_find_basic_set_by_statement_name(graph, inf_equal->statement2, &node_index_to);

    if (bset_from != NULL && bset_to != NULL && node_index_from != -1 && node_index_to != -1) {
      int pos_from;
      int pos_to;
      log::Info(log::Verbosity::high, "scanning equal constraint for influence equal constraint[" +
                                        std::to_string(i + 1) + ":" + std::to_string(inf_list->size) + "]");
      log::Info(log::Verbosity::high, "statement from:\t" + std::string(inf_equal->statement2));
      log::Info(log::Verbosity::high, "statement to:\t" + std::string(inf_equal->statement1));
      log::Info(log::Verbosity::high, "sched_dim from:\t" + std::to_string(sched_from));
      log::Info(log::Verbosity::high, "sched_dim to:\t" + std::to_string(sched_to));
      log::Info(log::Verbosity::high, "coef_dim from\t" + std::to_string(coef_from));
      log::Info(log::Verbosity::high, "coef_dim to\t" + std::to_string(coef_to));
      log::Info(log::Verbosity::high, "type:\t\t" + std::string(inf_types_str[(int)(inf_equal->type)]));
      log::Info(log::Verbosity::high, "isl_influence_set_equal: copying coef from " +
                                        std::string(inf_equal->statement2) + " to " +
                                        std::string(inf_equal->statement1));
      print_basic_set(bset_from, "bset from:");
      print_basic_set(bset_to, "bset to:");
      log::Info(log::Verbosity::high, "node from:\t" + std::to_string(node_index_from));
      log::Info(log::Verbosity::high, "node to:\t" + std::to_string(node_index_to));
      pos_from = get_pos_from_bset(bset_from, isl_sched_graph_get_node(graph, node_index_from), inf_equal, coef_from);
      pos_to = get_pos_from_bset(bset_to, isl_sched_graph_get_node(graph, node_index_to), inf_equal, coef_to);

      if (inf_equal->sched_dim2 == inf_equal->sched_dim1) {
        bset = create_constraint(bset, "constraint created", pos_from, pos_to);
        if (inf_equal->type == isl_var) bset = create_constraint(bset, "constraint created", pos_from - 1, pos_to - 1);
      } else {
        int val = isl_influence_sol_get_elem(sched_from, pos_from, graph);
        log::Info(log::Verbosity::high, "val=" + std::to_string(val));
        bset = hack_coefficients(bset, "isl_equal", pos_to, val, val);
        if (inf_equal->type == isl_var) {
          val = isl_influence_sol_get_elem(sched_from, pos_from - 1, graph);
          bset = hack_coefficients(bset, "is_equal", pos_to - 1, val, val);
        }
      }
      inf_equal->scanned = 1;
    }
  }
  log::Info(log::Verbosity::high, "Leave isl_influence_set_equal");
  return bset;
}

__isl_give isl_schedule *akg_isl_schedule_constraints_compute_schedule_influence(
  __isl_take isl_schedule_constraints *sc, isl_influence_list *inf_coef, isl_influence_equal_list *inf_equal) {
  isl_ctx *ctx = isl_schedule_constraints_get_ctx(sc);
  struct isl_sched_graph graph = {0};
  isl_schedule *sched;
  isl_schedule_node *node;

  isl_union_set *domain;
  isl_size n;

  log::Info(log::Verbosity::high, "isl_schedule_constraints_compute_schedule : start printing constraints");
  isl_printer *p;
  p = isl_printer_to_str(ctx);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_constraints(p, sc);
  char *log_string = isl_printer_get_str(p);

  log::Info(log::Verbosity::high, std::string(log_string));

  isl_printer_free(p);
  free(log_string);

  log::Info(log::Verbosity::high, "isl_schedule_constraints_compute_schedule : end printing constraints");

  print_influence(inf_equal, inf_coef);

  graph.inf_list = inf_coef;
  graph.inf_equal_list = inf_equal;

  sc = isl_schedule_constraints_align_params(sc);

  domain = isl_schedule_constraints_get_domain(sc);
  n = isl_union_set_n_set(domain);
  if (n == 0) {
    isl_schedule_constraints_free(sc);
    return isl_schedule_from_domain(domain);
  }

  if (n < 0 || isl_sched_graph_init(&graph, sc) < 0) {
    domain = isl_union_set_free(domain);
  }

  node = isl_schedule_node_from_domain(domain);
  node = isl_schedule_node_child(node, 0);

  if (graph.n > 0) {
    node = isl_schedule_node_compute_schedule(node, &graph);
  }
  sched = isl_schedule_node_get_schedule(node);
  int result = report_influence(inf_equal, inf_coef, graph.inf_sol_list, graph.maxvar);

  isl_schedule_node_free(node);
  isl_sched_graph_free(ctx, &graph);
  isl_schedule_constraints_free(sc);

  if (!result) {
    log::Info(log::Verbosity::high, "isl_influence failed, will fallback to default isl");
    isl_schedule_free(sched);
    sched = NULL;
  }
  return sched;
}

isl_basic_set *akg_isl_influence_set_coef(isl_ctx *ctx, struct isl_sched_graph *graph, isl_basic_set *bset) {
  log::Info(log::Verbosity::high, "Enter isl_influence_set_coef for dimension --> " + std::to_string(graph->n_row));

  isl_influence_list *inf_list = graph->inf_list;
  int dimension = graph->n_row;

  for (int i = 0; i < graph->n; ++i) {
    struct isl_sched_node *node = isl_sched_graph_get_node(graph, i);
    isl_map *ma = isl_sched_node_extract_schedule(node);
    log::Info(log::Verbosity::high, "statement:");
    if (ma != NULL) {
      isl_printer *p;
      p = isl_printer_to_str(ctx);
      p = isl_printer_print_map(p, ma);

      char *log_str = isl_printer_get_str(p);
      log::Info(log::Verbosity::high, std::string(log_str));

      isl_printer_free(p);
      free(log_str);
    }
    log::Info(log::Verbosity::high, "end statement");
    const char *strstat = isl_map_get_tuple_name(ma, isl_dim_in);
    isl_map_free(ma);

    for (int j = 0; j < inf_list->size; j++) {
      isl_influence *inf = &inf_list->data[j];

      if (inf->sched_dim == dimension && strcmp(strstat, inf->statement_name) == 0) {
        int pos;
        int ub = inf->val;
        int lb = inf->val;
        log::Info(log::Verbosity::high, "scanning isl coefficients for influence[" + std::to_string(j + 1) + ":" +
                                          std::to_string(inf_list->size) + "]:");
        // S_0,S_1,...,S_n-1,S_n
        log::Info(log::Verbosity::high, "statement_name:\t\t" + std::string(inf->statement_name));
        // i,j,k.... (only apply for type=isl_var_plus | isl_var_minus
        log::Info(log::Verbosity::high, "sched dim:\t\t" + std::to_string(inf->sched_dim));
        // coefficient index
        log::Info(log::Verbosity::high, "rank variable:\t\t" + std::to_string(inf->coef_dim));
        int nparam = isl_sched_node_get_nparam(node);
        int nvar = isl_sched_node_get_nvar(node);
        log::Info(log::Verbosity::high, "statement variables:\t" + std::to_string(nvar));
        // coefficient value
        log::Info(log::Verbosity::high, "coefficient value:\t" + std::to_string(inf->val));
        log::Info(log::Verbosity::high, "type:\t\t" + std::string(inf_types_str[(int)(inf->type)]));
        print_basic_set(bset, "lp  problem to influence:");
        switch (inf->type) {
          case isl_cst:
            pos = isl_sched_node_cst_coef_offset(node);
            bset = hack_coefficients(bset, "isl_cst", pos, ub, lb);
            inf->scanned = 1;
            break;
          case isl_param:
            log::Info(log::Verbosity::high, "Statement param (node->nparam): " + std::to_string(nparam));
            pos = isl_sched_node_cst_coef_offset(node) - (nparam - inf->coef_dim);
            bset = hack_coefficients(bset, "isl_param", pos, ub, lb);
            inf->scanned = 1;
            break;
          case isl_var:
            // dim_+ coefficient	;
            if (inf->coef_dim <= nvar) {
              pos = isl_sched_node_cst_coef_offset(node) - nparam - 1 - inf->coef_dim * 2;
              if (inf->val > 0) {
                bset = hack_coefficients(bset, "isl_var_+", pos, ub, lb);
                // dim_- coeffiecient
                pos--;
                bset = hack_coefficients(bset, "isl_var_-", pos, 0, 0);
                inf->scanned = 1;
              } else if (inf->val == 0) {
                // dim_+ coefficient
                bset = hack_coefficients(bset, "isl_var_+", pos, 0, 0);
                // dim_- coeffiecient
                pos--;
                bset = hack_coefficients(bset, "isl_var_-", pos, 0, 0);
                inf->scanned = 1;

              } else if (inf->val < 0) {
                bset = hack_coefficients(bset, "isl_var_+", pos, 0, 0);
                // dim_- coeffiecient
                pos--;
                bset = hack_coefficients(bset, "isl_var_-", pos, -ub, -lb);
                inf->scanned = 1;
              } else {
                log::Info(log::Verbosity::high, "invalid inf->val: " + std::to_string(inf->val));
              }
            } else {
              log::Info(log::Verbosity::high,
                        "Warning: dimension overflow -->  dimension required: " + std::to_string(inf->coef_dim) +
                          " max dimensions: " + std::to_string(nvar));
            }
            break;
          default:
            log::Info(log::Verbosity::high, "unknown influence coef type");
            break;
        }
        print_basic_set(bset, "lp influenced problem:");
      }
    }
  }
  log::Info(log::Verbosity::high, "Leave isl_influence_set_coef");
  return bset;
}

int akg_isl_influence_maxvar(struct isl_sched_graph *graph) {
  log::Info(log::Verbosity::high, "Entering akg_isl_influence_maxar");
  int maxvar = 0;
  int previous = maxvar;
  int var;
  isl_influence_list *inf_list = graph->inf_list;
  isl_influence_equal_list *inf_equal_list = graph->inf_equal_list;

  for (int i = 0; NULL != inf_list && i < inf_list->size; i++) {
    isl_influence *inf = &inf_list->data[i];
    var = inf->sched_dim + 1;
    if (maxvar < var) {
      maxvar = var;
    }
  }
  for (int i = 0; NULL != inf_equal_list && i < inf_equal_list->size; i++) {
    isl_influence_equal *inf_equal = &inf_equal_list->data[i];
    var = inf_equal->sched_dim1 + 1;

    if (maxvar < var) {
      maxvar = var;
    }

    var = inf_equal->sched_dim2 + 1;

    if (maxvar < var) {
      maxvar = var;
    }
  }

  log::Info(log::Verbosity::high, "isl_influence_maxvar : " + std::to_string(maxvar) +
                                    " (previous maxvar: " + std::to_string(previous) + ")");
  log::Info(log::Verbosity::high, "Leaving akg_isl_influence_maxvar");
  return maxvar;
}

int akg_isl_influence_check_coincident(struct isl_sched_graph *graph, isl_vec *sol) {
  int coincident = graph->n > 1 ? 1 : 0;
  int pos = 0;

  log::Info(log::Verbosity::high, "Entering isl_influene_check_coincidence for dimension " +
                                    std::to_string(graph->n_row) +
                                    ", number of nodes (statements): " + std::to_string(graph->n));
  int *vec = (int *)calloc(graph->n * 3, sizeof(int));

  for (int i = 0; i < graph->n; ++i) {
    isl_sched_node *node = isl_sched_graph_get_node(graph, i);
    int cst_offset = isl_sched_node_cst_coef_offset(node);
    int nparam = isl_sched_node_get_nparam(node);
    int nvar = isl_sched_node_get_nvar(node);

    log::Info(log::Verbosity::high, "graph node:\t" + std::to_string(i));
    log::Info(log::Verbosity::high, "cst_offset:\t" + std::to_string(cst_offset));
    log::Info(log::Verbosity::high, "nparam:\t\t" + std::to_string(nparam));
    log::Info(log::Verbosity::high, "var:\t\t" + std::to_string(nvar));

    vec[pos++] = cst_offset;
    vec[pos++] = nparam;
    vec[pos++] = nvar;
  }

  int is_equal;

  for (int i = 0; i < graph->n; ++i)
    for (int j = i; j < graph->n; ++j) {
      if (j == i) continue;

      log::Info(log::Verbosity::high,
                "calculating coincidence for statement " + std::to_string(i) + " and " + std::to_string(j));

      int cst_offset_S0 = vec[i * 3];
      int cst_offset_S1 = vec[j * 3];

      is_equal = isl_influence_int_eq(sol, 1 + cst_offset_S0, 1 + cst_offset_S1);

      log::Info(log::Verbosity::high, "S_" + std::to_string(i) + "_i" + std::to_string(cst_offset_S0) + ", S_" +
                                        std::to_string(j) + "_i" + std::to_string(cst_offset_S1) +
                                        ", cst coefficient equal: " + std::to_string(is_equal));

      if (is_equal == 0) {
        coincident = 0;
        break;
      }

      log::Info(log::Verbosity::high, "param coefficient(s) equal:");

      int nparam_S0 = vec[i * 3 + 1];
      int nparam_S1 = vec[j * 3 + 1];

      if (nparam_S0 != 0 && nparam_S1 != 0 && nparam_S0 == nparam_S1) {
        int nparam_pos_0 = cst_offset_S0 - nparam_S0;
        int nparam_pos_1 = cst_offset_S1 - nparam_S1;

        for (int k = nparam_pos_0, l = nparam_pos_1; k < nparam_pos_0 + nparam_S0; k++, l++) {
          is_equal = isl_influence_int_eq(sol, 1 + k, 1 + l);
          log::Info(log::Verbosity::high, "S_" + std::to_string(i) + "_i" + std::to_string(k) + ", S_" +
                                            std::to_string(j) + "_i" + std::to_string(l) +
                                            "equal: " + std::to_string(is_equal));
          if (is_equal == 0) {
            coincident = 0;
            break;
          }
        }
      } else {
        log::Info(log::Verbosity::high, " no parameteres found or number of parameters distinct for each statement.");
      }

      log::Info(log::Verbosity::high, "variable coefficients equal:");

      int nvar_S0 = vec[i * 3 + 2];
      int nvar_S1 = vec[j * 3 + 2];

      if (nvar_S0 != nvar_S1) {
        coincident = 0;
        break;
      }

      int nvar_pos_0 = cst_offset_S0 - nparam_S0 - 2 * nvar_S0;
      int nvar_pos_1 = cst_offset_S1 - nparam_S1 - 2 * nvar_S1;

      for (int k = nvar_pos_0, l = nvar_pos_1; k < nvar_pos_0 + 2 * nvar_S0; k++, l++) {
        is_equal = isl_influence_int_eq(sol, 1 + k, 1 + l);
        log::Info(log::Verbosity::high, "S_" + std::to_string(i) + "_i" + std::to_string(k) + ", S_" +
                                          std::to_string(j) + "_i" + std::to_string(l) +
                                          " equal: " + std::to_string(is_equal));
        if (is_equal == 0) {
          coincident = 0;
          break;
        }
      }
    }

  free(vec);

  log::Info(log::Verbosity::high, "Leaving isl_check_coindicent result: " + std::to_string(coincident));

  return coincident;
}

}  // namespace poly
}  // namespace ir
}  // namespace akg
