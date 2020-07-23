/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
#ifndef POLY_UTIL_H_
#define POLY_UTIL_H_
#include <tvm/ir.h>
#include <ir_pass.h>
#include <chrono>
#include "isl.h"

namespace akg {
namespace ir {
namespace poly {
/// IR dump and debug options
#define DUMP_IR true
#define PRETTY_PRINT_IR true
#define DUMP_SCOP_DATA true
#define DUMP_SCOP_DATA_PER_PASS false
#define DUMP_IN_CURRENT_DIR false

#define PRINT_SCHEDULE_INFO false
#define PRINT_ISL_EMMITER false
#define PRINT_CCE_ISL_EMMITER false
#define PRINT_EMMITER (PRINT_ISL_EMMITER || PRINT_CCE_ISL_EMMITER)
#define SPEC_GEMM true
#define DELETE_FRACTAL true

/// conv_backward options
#define SELECT_DOMAIN_OPT true

// timer records
#define TIMER_START timer_start = std::chrono::high_resolution_clock::now()
#define TIMER_DURATION                                                                                                \
  (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - timer_start) \
     .count()) *                                                                                                      \
    1000
#define TIMER_SHOW(NAME, SPEC_GEMM) \
  { LOG(INFO) << "[ Polyhedral exec time" << SPEC_GEMM << " ], " << NAME << " spent " << TIMER_DURATION << " ms"; }

unsigned int WrappedStrtol(const std::string &str);

bool IsEndsWith(const std::string &str, const std::string &suffix);

std::vector<std::string> Split(const std::string &str, const std::string &pattern);

std::vector<int> SplitString(const std::string &str, const std::string &separator);

isl::ast_node CanonicalizeBlockInAst(const isl::ast_node &astNode);

Expr RemoveCast(Expr e);

Stmt PeelOuterLetStmt(const Stmt &s, std::vector<Stmt> &outer_stmts);

isl::union_map ShortSchedule(const isl::schedule_node &node);
isl::union_map LocalSchedule(const isl::schedule_node &node);
void GetAffOffsetAndNumVars(const isl::aff &aff, int &offset, int &num_vars);
bool IsAffVarPlusOffset(const isl::aff &aff);
bool IsAffNonZeroConst(const isl::aff &aff);

class ConsolidateExprMutator : public IRMutator {
 public:
  explicit ConsolidateExprMutator(const std::unordered_map<std::string, Var> &params_) : params(params_) {}
  ~ConsolidateExprMutator() override = default;

 private:
  template <class T>
  Expr GenericMutate(const T *op, const Expr &e) {
    std::stringstream os;
    os << e;
    std::string expr_str = os.str();
    if (params.count(expr_str) > 0) {
      return params.at(expr_str);
    }
    return IRMutator::Mutate_(op, e);
  }

  // list operators that may appear in dynamic shape params
  Expr Mutate_(const Add *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Sub *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Mul *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const FloorDiv *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const FloorMod *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Div *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Mod *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Min *op, const Expr &e) override { return GenericMutate(op, e); }
  Expr Mutate_(const Max *op, const Expr &e) override { return GenericMutate(op, e); }

  const std::unordered_map<std::string, Var> &params;
};
}  // namespace poly
constexpr auto ATTR_CONV_FEATURE_NAME = "feature";
constexpr auto ATTR_CONV_FILTER_NAME = "filter";
constexpr auto ATTR_CONV_BIAS_NAME = "bias";
constexpr auto ATTR_CONV_RES_NAME = "res";
constexpr auto ATTR_CONV_FEATURE_N = "pragma_conv_fm_n";
constexpr auto ATTR_CONV_FEATURE_C = "pragma_conv_fm_c";
constexpr auto ATTR_CONV_FEATURE_H = "pragma_conv_fm_h";
constexpr auto ATTR_CONV_FEATURE_W = "pragma_conv_fm_w";
constexpr auto ATTR_CONV_KERNEL_N = "pragma_conv_kernel_n";
constexpr auto ATTR_CONV_KERNEL_H = "pragma_conv_kernel_h";
constexpr auto ATTR_CONV_KERNEL_W = "pragma_conv_kernel_w";
constexpr auto ATTR_CONV_STRIDE_H = "pragma_conv_stride_h";
constexpr auto ATTR_CONV_STRIDE_W = "pragma_conv_stride_w";
constexpr auto ATTR_CONV_DILATION_H = "pragma_conv_dilation_h";
constexpr auto ATTR_CONV_DILATION_W = "pragma_conv_dilation_w";
constexpr auto ATTR_CONV_PAD_LEFT = "pragma_conv_padding_left";
constexpr auto ATTR_CONV_PAD_RIGHT = "pragma_conv_padding_right";
constexpr auto ATTR_CONV_PAD_TOP = "pragma_conv_padding_top";
constexpr auto ATTR_CONV_PAD_BOTTOM = "pragma_conv_padding_bottom";
constexpr auto ATTR_CONV_BYPASS_L1 = "pragma_conv_bypass_l1";
constexpr auto ATTR_CONV_BACKPROP_INPUT = "pragma_conv_backprop_input";
constexpr auto ATTR_CONV_BACKPROP_FILTER = "pragma_conv_backprop_filter";
constexpr auto ATTR_CONV_SPECIAL_DMA = "pragma_conv_special_dma";
constexpr auto ATTR_CONV_TILE_B = "pragma_conv_batch_cut";
constexpr auto ATTR_CONV_TILE_H = "pragma_conv_h_cut";
constexpr auto ATTR_CONV_TILE_W = "pragma_conv_w_cut";
constexpr auto ATTR_CONV_TILE_KH = "pragma_conv_kh_cut";
constexpr auto ATTR_CONV_TILE_KW = "pragma_conv_kw_cut";
constexpr auto ATTR_CONV_TILE_CO = "pragma_conv_co_cut";
constexpr auto ATTR_CONV_TILE_CIN = "pragma_conv_cin_cut";
constexpr auto ATTR_CONV_TILE_M = "pragma_conv_m_cut";
constexpr auto ATTR_CONV_TILE_K = "pragma_conv_k_cut";
constexpr auto ATTR_CONV_TILE_N = "pragma_conv_n_cut";
constexpr auto ATTR_CONV_M_INNER = "pragma_conv_m_inner";
constexpr auto ATTR_CONV_N_INNER = "pragma_conv_n_inner";
constexpr auto ATTR_CONV_K_INNER = "pragma_conv_k_inner";
constexpr auto ATTR_CONV_M_CUT_SIZE = "pragma_conv_m_cut_size";
constexpr auto ATTR_PRAGMA_OUT_H = "pragma_out_h";
constexpr auto ATTR_PRAGMA_OUT_W = "pragma_out_w";

constexpr auto ATTR_SPEC_GEMM_BATCH = "pragma_spec_gemm_batch";
constexpr auto ATTR_SPEC_GEMM_M = "pragma_spec_gemm_m";
constexpr auto ATTR_SPEC_GEMM_K = "pragma_spec_gemm_k";
constexpr auto ATTR_SPEC_GEMM_N = "pragma_spec_gemm_n";
constexpr auto ATTR_SPEC_GEMM_M_ALIGN = "pragma_spec_gemm_m_align";
constexpr auto ATTR_SPEC_GEMM_K_ALIGN = "pragma_spec_gemm_k_align";
constexpr auto ATTR_SPEC_GEMM_N_ALIGN = "pragma_spec_gemm_n_align";
constexpr auto ATTR_SPEC_GEMM_M_INNER = "pragma_spec_gemm_m_inner";
constexpr auto ATTR_SPEC_GEMM_K_INNER = "pragma_spec_gemm_k_inner";
constexpr auto ATTR_SPEC_GEMM_N_INNER = "pragma_spec_gemm_n_inner";
constexpr auto ATTR_SPEC_GEMM_TILE_M = "pragma_spec_gemm_m_cut";
constexpr auto ATTR_SPEC_GEMM_TILE_K = "pragma_spec_gemm_k_cut";
constexpr auto ATTR_SPEC_GEMM_TILE_N = "pragma_spec_gemm_n_cut";

constexpr auto ATTR_CONV_BATCH = "pragma_conv_batch";
constexpr auto ATTR_CONV_GMM_FEATURE = "pragma_gemm_data";
constexpr auto ATTR_CONV_GMM_WEIGHT = "pragma_gemm_weight";
constexpr auto ATTR_CONV_GMM_RES = "pragma_gemm_res";
constexpr auto ATTR_CONV_GMM_M = "pragma_conv_gemm_m";
constexpr auto ATTR_CONV_GMM_K = "pragma_conv_gemm_k";
constexpr auto ATTR_GEMM_DATA_TRANSPOSE = "pragma_data_transpose";
constexpr auto ATTR_GEMM_DATA_TRANSPOSE_BLOCK = "pragma_data_transpose_block";
constexpr auto ATTR_GEMM_DATA_TRANSPOSE_BLOCK_INNER = "pragma_data_transpose_block_inner";
constexpr auto ATTR_GEMM_WEIGHT_TRANSPOSE = "pragma_weight_transpose";
constexpr auto ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK = "pragma_weight_transpose_block";
constexpr auto ATTR_GEMM_WEIGHT_TRANSPOSE_BLOCK_INNER = "pragma_weight_transpose_block_inner";

constexpr auto ATTR_ATOMIC_ADD = "atomic_add";
constexpr auto ATOMIC_COND_CLEAN = "atomic_cond_clean";

constexpr auto UBL0 = "UBL0";
constexpr auto REALIZE_ = "realize_";
/******************************************************
 * Following const is the mark tags for schedule tree
 ******************************************************/
constexpr auto REALIZE = "realize";
constexpr auto REALIZE_L1 = "realize_L1";
constexpr auto REALIZE_L0 = "realize_L0";
constexpr auto REALIZE_UB = "realize_UB";
constexpr auto REALIZE_UBL0 = "realize_UBL0";
constexpr auto REALIZE_UBL1 = "realize_UBL1";
constexpr auto REALIZE_L1UBL1 = "realize_L1UBL1";
constexpr auto CONV_GEMM = "conv_gemm";
constexpr auto FUSE_VECTOR = "fuse_vector";
constexpr auto MULTICORE_COINCIDENT = "multicore_coincident_";

constexpr auto ALLOC_C = "alloc_C";
constexpr auto ALLOC_REALIZE_OUT = "alloc_out";

constexpr auto CALL_IM2COL_UB = "cce_img2col_ub";
constexpr auto ATTR_IM2COL_KEY = "im2colKey";

const std::vector<std::string> ConvATTRList = {ATTR_CONV_FEATURE_W,  ATTR_CONV_KERNEL_H,   ATTR_CONV_KERNEL_W,
                                               ATTR_CONV_STRIDE_H,   ATTR_CONV_STRIDE_W,   ATTR_CONV_DILATION_H,
                                               ATTR_CONV_DILATION_W, ATTR_CONV_PAD_LEFT,   ATTR_CONV_PAD_RIGHT,
                                               ATTR_CONV_PAD_TOP,    ATTR_CONV_PAD_BOTTOM, ATTR_CONV_BYPASS_L1};

const std::vector<std::string> FastPoolingATTRList = {
  ATTR_CONV_FEATURE_H, ATTR_CONV_FEATURE_W,  ATTR_CONV_KERNEL_H,   ATTR_CONV_KERNEL_W, ATTR_CONV_STRIDE_H,
  ATTR_CONV_STRIDE_W,  ATTR_CONV_DILATION_H, ATTR_CONV_DILATION_W, ATTR_CONV_PAD_LEFT, ATTR_CONV_PAD_RIGHT,
  ATTR_CONV_PAD_TOP,   ATTR_CONV_PAD_BOTTOM, ATTR_CONV_TILE_H,     ATTR_CONV_TILE_W};

}  // namespace ir
}  // namespace akg
#endif  // POLY_UTIL_H_
