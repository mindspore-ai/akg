/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <limits.h>
#include <tvm/ir.h>
#include <ir_pass.h>
#include <chrono>
#include "isl.h"
#include <pass/utils.h>

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
#define PRINT_ISL_EMITTER false
#define PRINT_NPU_ISL_EMITTER false
#define PRINT_EMITTER (PRINT_ISL_EMITTER || PRINT_NPU_ISL_EMITTER)
#define SPEC_GEMM true
#define DELETE_FRACTAL true
#define USE_SIMPLE_EXTENSION true

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

#define FOREACH(X)                \
  X(elewise_single_VS_add)        \
  X(elewise_single_VS_mul)        \
  X(elewise_single_rec)           \
  X(elewise_single_log)           \
  X(elewise_single_exp)           \
  X(elewise_single_sin)           \
  X(elewise_single_cos)           \
  X(elewise_single_asin)          \
  X(elewise_single_acos)          \
  X(elewise_single_isinf)         \
  X(elewise_single_isfinite)      \
  X(elewise_single_isnan)         \
  X(elewise_single_sqrt)          \
  X(elewise_single_rsqrt)         \
  X(vec_single_cast)              \
  X(vec_single_floor)             \
  X(vec_single_round)             \
  X(elewise_single_ceil)          \
  X(vec_single_trunc)             \
  X(elewise_single_not)           \
  X(elewise_single_fabs)          \
  X(elewise_single_relu)          \
  X(broadcast)                    \
  X(pandora_select)               \
  X(pandora_cmp)                  \
  X(reg_mov)                      \
  X(mad)                          \
  X(elewise_binary_add)           \
  X(elewise_binary_sub)           \
  X(elewise_binary_mul)           \
  X(elewise_binary_div)           \
  X(elewise_binary_mod)           \
  X(elewise_binary_min)           \
  X(elewise_binary_max)           \
  X(elewise_binary_or)            \
  X(elewise_binary_and)           \
  X(elewise_binary_EQ)            \
  X(elewise_binary_NE)            \
  X(elewise_binary_GT)            \
  X(elewise_binary_GE)            \
  X(elewise_binary_LT)            \
  X(elewise_binary_LE)            \
  X(elewise_binary_scalar_axpy)   \
  X(four2five_nchw)               \
  X(vec_argmax)                   \
  X(elewise_binary_proposal_sort) \
  X(elewise_binary_topk_sort)     \
  X(elewise_binary_nms)           \
  X(with)                         \
  X(vec_argmin)                   \
  X(elewise_binary_dropout)       \
  X(elewise_binary_iou)           \
  X(elewise_binary_unknown)       \
  X(assignment)                   \
  X(im2col)                       \
  X(poly_op_type_max)             \
  X(vmadd)                        \
  X(vmaddrelu)                    \
  X(vaxpy)                        \
  X(vmla)                         \
  X(elewise_binary_bitwise_and)   \
  X(elewise_binary_bitwise_or)    \
  X(elewise_single_bitwise_not)   \
  X(tvm_if_then_else)             \
  X(reshape)                      \
  X(transpose)                    \
  X(divide_var)                   \
  X(sub_relu)                     \
  X(pow)                          \
  X(isnan)                        \
  X(load_im2col_c1_buf)           \
  X(tanh)                         \
  X(asinh)                        \
  X(acosh)                        \
  X(elewise_single_atan)          \
  X(elewise_binary_atan2)         \
  X(elewise_single_expm1)         \
  X(elewise_single_erf)           \
  X(elewise_single_tot)

#define GENERATE_ENUM(ENUM) ENUM,

enum class PolyOpType : int { FOREACH(GENERATE_ENUM) };

const std::map<std::string, PolyOpType> POLY_SUPPORTED_OPS = {
  {"sin", PolyOpType::elewise_single_sin},
  {"cos", PolyOpType::elewise_single_cos},
  {"asin", PolyOpType::elewise_single_asin},
  {"acos", PolyOpType::elewise_single_acos},
  {"isnan", PolyOpType::elewise_single_isnan},
  {"isinf", PolyOpType::elewise_single_isinf},
  {"isfinite", PolyOpType::elewise_single_isfinite},
  {"log", PolyOpType::elewise_single_log},
  {"exp", PolyOpType::elewise_single_exp},
  {"sqrt", PolyOpType::elewise_single_sqrt},
  {"rsqrt", PolyOpType::elewise_single_rsqrt},
  {"fabs", PolyOpType::elewise_single_fabs},
  {"rec", PolyOpType::elewise_single_rec},
  {"floor", PolyOpType::vec_single_floor},
  {"round", PolyOpType::vec_single_round},
  {"ceil", PolyOpType::elewise_single_ceil},
  {"trunc", PolyOpType::vec_single_trunc},
  {"not", PolyOpType::elewise_single_not},
  {"relu", PolyOpType::elewise_single_relu},
  {"EQ", PolyOpType::elewise_binary_EQ},
  {"NE", PolyOpType::elewise_binary_NE},
  {"GT", PolyOpType::elewise_binary_GT},
  {"GE", PolyOpType::elewise_binary_GE},
  {"LT", PolyOpType::elewise_binary_LT},
  {"LE", PolyOpType::elewise_binary_LE},
  {"fargmax", PolyOpType::vec_argmax},
  {"fargmin", PolyOpType::vec_argmin},
  {"four2five_nchw", PolyOpType::four2five_nchw},
  {"vand", PolyOpType::elewise_binary_and},
  {"bitwise_and", PolyOpType::elewise_binary_bitwise_and},
  {"bitwise_or", PolyOpType::elewise_binary_bitwise_or},
  {"bitwise_not", PolyOpType::elewise_single_bitwise_not},
  {"proposal_sort", PolyOpType::elewise_binary_proposal_sort},
  {"topk_sort", PolyOpType::elewise_binary_topk_sort},
  {"nms", PolyOpType::elewise_binary_nms},
  {"dropout", PolyOpType::elewise_binary_dropout},
  {"iou", PolyOpType::elewise_binary_iou},
  {"vmadd", PolyOpType::vmadd},
  {"vmaddrelu", PolyOpType::vmaddrelu},
  {"vaxpy", PolyOpType::vaxpy},
  {"vmla", PolyOpType::vmla},
  {"tvm_if_then_else", PolyOpType::tvm_if_then_else},
  {"reshape", PolyOpType::reshape},
  {"transpose", PolyOpType::transpose},
  {"divide_var", PolyOpType::divide_var},
  {"sub_relu", PolyOpType::sub_relu},
  {"pow", PolyOpType::pow},
  {"isnan", PolyOpType::isnan},
  {"load_im2col_c1_buf", PolyOpType::load_im2col_c1_buf},
  {"tanh", PolyOpType::tanh},
  {"asinh", PolyOpType::asinh},
  {"acosh", PolyOpType::acosh},
  {"atan", PolyOpType::elewise_single_atan},
  {"atan2", PolyOpType::elewise_binary_atan2},
  {"expm1", PolyOpType::elewise_single_expm1},
  {"erf", PolyOpType::elewise_single_erf},
  {"tot_op", PolyOpType::elewise_single_tot},
};

unsigned int WrappedStrtol(const std::string &str);

bool IsEndsWith(const std::string &str, const std::string &suffix);

bool IsStartsWith(const std::string &str, const std::string &prefix);

std::vector<std::string> Split(const std::string &str, const std::string &pattern);

std::vector<int> SplitString(const std::string &str, const std::string &separator);

isl::ast_node CanonicalizeBlockInAst(const isl::ast_node &astNode);

Expr RemoveCast(Expr e);

Stmt PeelOuterLetStmt(const Stmt &s, std::vector<Stmt> &outer_stmts);

isl::union_map ShortSchedule(const isl::schedule_node &node);
isl::union_map LocalSchedule(const isl::schedule_node &node);
isl::multi_union_pw_aff ShortScheduleMupa(const isl::schedule_node &root, const isl::schedule_node &tree);
isl::multi_union_pw_aff ShortScheduleMupaImpl(const isl::schedule_node &root, const isl::schedule_node &relative_root,
                                              const isl::schedule_node &node);
void GetAffOffsetAndNumVars(const isl::aff &aff, int &offset, int &num_vars);
bool IsAffVarPlusOffset(const isl::aff &aff);
bool IsAffNonZeroConst(const isl::aff &aff);
isl::union_map ScheduleTensorMapping(const isl::multi_union_pw_aff &outer_schedule,
                                     const isl::union_map &tensor_access);

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
class FindInnerRealize : public air::ir::IRMutator {
 public:
  explicit FindInnerRealize(std::string name) : name_(std::move(name)) {}
  ~FindInnerRealize() override = default;

 private:
  Stmt Mutate_(const Realize *op, const Stmt &s) final {
    if (op->func->func_name() == name_) {
      return this->Mutate(op->body);
    }
    return IRMutator::Mutate_(op, s);
  }

 private:
  std::string name_;
};

constexpr auto MMA_UNIT = 16;
constexpr auto MIN_TILE = 1;
constexpr auto AT_TEMPLATE = "TEMPLATE";
constexpr auto AT_THREAD_CFG = "THREAD_CONFIG";
constexpr auto AT_BLOCK_CFG = "BLOCK_CONFIG";

/*!
 * IslEmitter for GPU
 */
constexpr auto AKG_ALL_REDUCE = "akg_reduce::ALL_REDUCE";
constexpr auto AKG_X_REDUCE = "akg_reduce::REDUCE2D_X";
constexpr auto AKG_Y_REDUCE = "akg_reduce::REDUCE2D_Y";
constexpr auto SCALAR_TENSOR_PREFIX = "acc_";
constexpr auto SCALAR_KHT_PREFIX = "kahan_t";
constexpr auto SCALAR_KHY_PREFIX = "kahan_y";
constexpr auto SCALAR_KHC_PREFIX = "kahan_c";
constexpr auto SHARED_TENSOR_PREFIX = "red_buf";

enum TileLevel { CACHE0 = 0, CACHE1 };

enum ReduceDirection {
  UNKNOWN = 0,
  X,
  Y,
  ALL,
};

}  // namespace poly
constexpr auto TARGET_CCE = "cce";
constexpr auto TARGET_CUDA = "cuda";
constexpr auto TARGET_CPU = "cpu";

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
constexpr auto ATTR_ATOMIC_CLEAN_ZERO = "atomic_clean_zero";

constexpr auto UBL0 = "UBL0";
constexpr auto REALIZE_ = "realize_";
constexpr auto REALIZE_PREFIX = "REALIZE_";
constexpr auto REALIZE_PREFIX_LEN = 8;

constexpr auto BLOCK_STR = "b";
constexpr auto THREAD_STR = "t";
constexpr auto WARP_STR = "w";
constexpr auto B0 = "b0";
constexpr auto B1 = "b1";
constexpr auto B2 = "b2";
constexpr auto T0 = "t0";
constexpr auto T1 = "t1";
constexpr auto T2 = "t2";
constexpr auto TILE_WITH_C1 = "C1";
constexpr auto TILE_WITH_C0 = "C0";
constexpr auto TILE_WITH_C0_C1 = "C0_C1";
constexpr auto TILE_WITH_LAST_C1 = "LAST_C1";
constexpr auto TILE_WITH_LAST_C0 = "LAST_C0";
constexpr auto TILE_WITH_WARP_C1 = "WARP_C1";
constexpr auto REPLACE = "replace_";
constexpr auto COMPUTE = "compute";
constexpr auto REPEATED_MAPPING = "repeated_";
constexpr auto PROMOTE = "promote_";
constexpr auto WARP_COMPUTE = "warp_compute";

constexpr auto BLOCK_IDX_X = "blockIdx.x";
constexpr auto BLOCK_IDX_Y = "blockIdx.y";
constexpr auto BLOCK_IDX_Z = "blockIdx.z";
constexpr auto THREAD_IDX_X = "threadIdx.x";
constexpr auto THREAD_IDX_Y = "threadIdx.y";
constexpr auto THREAD_IDX_Z = "threadIdx.z";

constexpr auto SYNC_FLAG = "_sync_";
constexpr auto STORAGE_SYNC = "tvm_storage_sync";
constexpr auto REDUCE = "reduce";
constexpr auto SYNC_SCOP_WARP = "warp";
constexpr auto SYNC_SCOP_SHARED = "shared";
constexpr auto SYNC_SCOP_GLOBAL = "global";

constexpr auto ROW_MAJOR = "row_major";
constexpr auto COL_MAJOR = "col_major";
constexpr auto REDUCE_AREA_FLAG = "reduce_area";

/******************************************************
 * Following const is the mark tags for schedule tree
 ******************************************************/
constexpr auto REALIZE = "realize";
constexpr auto CONV_GEMM = "conv_gemm";
constexpr auto CONV_KHKW_OUTER = "conv_khkw_outer";
constexpr auto FUSE_VECTOR = "fuse_vector";
constexpr auto MULTICORE_COINCIDENT = "multicore_coincident_";

constexpr auto ALLOC_C = "alloc_C";
constexpr auto ALLOC_REALIZE_OUT = "alloc_out";

constexpr auto CALL_IM2COL_UB = "cce_img2col_ub";
constexpr auto ATTR_IM2COL_KEY = "im2colKey";

constexpr auto MAPPING_INVALID_WARP = INT_MAX;
// promote marker for poly
constexpr auto PROMOTE_GLOBAL_TO_SHARED_AB = "promote_global_to_shared_ab";
constexpr auto PROMOTE_GLOBAL_TO_SHARED_C = "promote_global_to_shared_c";
constexpr auto PROMOTE_SHARED_TO_REGISTER_AB = "promote_shared_to_register_ab";
constexpr auto PROMOTE_SHARED_TO_REGISTER_C = "promote_shared_to_register_c";
constexpr auto PROMOTE_GLOBAL_TO_REGISTER_C = "promote_global_to_register_c";
// promote marker for thread group
constexpr auto PROMOTE_GLOBAL_TO_SHARED = "promote_global_to_shared";
constexpr auto PROMOTE_REGISTER_TO_GLOBAL = "promote_register_to_global";
constexpr auto PROMOTE_REGISTER_TO_SHARED = "promote_register_to_shared";
constexpr auto PROMOTE_SHARED_TO_GLOBAL = "promote_shared_to_global";

// promote marker for ForType
constexpr auto FOR_SERIAL = "for_serial";
constexpr auto FOR_PARALLEL = "for_parallel";
constexpr auto FOR_VECTORIZED = "for_vectorized";
constexpr auto FOR_UNROLLED = "for_unrolled";
constexpr auto FOR_SWIZZLED = "for_swizzled";

constexpr auto PROMOTE_VECTORIZATION = "promote_vectorization";
constexpr auto PROMOTE_VECTORIZATION_BIT = 128;
constexpr auto SKIP_MARKER = "skip";
constexpr auto MAP_TO_WARP = "map_to_warp";
constexpr auto THREAD_MARKER = "thread_marker";
constexpr auto BLOCK_MARKER = "block_marker";
constexpr auto WARP_MARKER = "warp_marker";
constexpr auto KH_KW_MARKER = "kh_kw_marker";
constexpr auto VECTORIZATION_MARKER = "vectorization_marker";
constexpr auto REDUCE_MARKER = "reduce_marker_";
constexpr auto ATOMIC_MARKER = "atomic";
constexpr auto REDUCE_INIT = "red_init_";
constexpr auto REDUCE_UPDATE = "red_update_";
constexpr auto INSERT_SYNC = "insert_sync";

constexpr auto READ_ID_NAME = "GMread";
constexpr auto WRITE_ID_NAME = "GMwrite";
constexpr auto SHARED_READ_ID_NAME = "SHAREDread";
constexpr auto SHARED_WRITE_ID_NAME = "SHAREDwrite";
constexpr auto GML_READ_ID_NAME = "GMLread";
constexpr auto GML_WRITE_ID_NAME = "GMLwrite";

constexpr auto MATRIX_A = "matrix_a";
constexpr auto MATRIX_B = "matrix_b";
constexpr auto MATRIX_C = "matrix_c";
constexpr auto MATRIX_ELSE = "matrix_else";
constexpr auto FRAGMENT = "fragment_";
constexpr auto LOCAL_SUFFIX = "_local";
constexpr auto SHARE_SUFFIX = "_shared";
constexpr auto PROMOTION_INFIX = "_promotion_";

const std::unordered_set<std::string> AkgSupportedReduceOp = {AKG_REDUCE_SUM, AKG_REDUCE_MIN, AKG_REDUCE_MAX,
                                                              AKG_REDUCE_AND, AKG_REDUCE_OR,  AKG_REDUCE_PROD};

const std::unordered_set<std::string> AkgSupportedTotOp = {AKG_ATOMIC_TOT, AKG_TENSOR_OF_TENSOR, AKG_TENSOR_NOT_PROMOTE,
                                                           AKG_INNER_TENSOR};

const std::vector<std::string> ConvATTRList = {ATTR_CONV_FEATURE_W,  ATTR_CONV_KERNEL_H,   ATTR_CONV_KERNEL_W,
                                               ATTR_CONV_STRIDE_H,   ATTR_CONV_STRIDE_W,   ATTR_CONV_DILATION_H,
                                               ATTR_CONV_DILATION_W, ATTR_CONV_PAD_LEFT,   ATTR_CONV_PAD_RIGHT,
                                               ATTR_CONV_PAD_TOP,    ATTR_CONV_PAD_BOTTOM, ATTR_CONV_BYPASS_L1};

const std::vector<std::string> FastPoolingATTRList = {
  ATTR_CONV_FEATURE_H, ATTR_CONV_FEATURE_W,  ATTR_CONV_KERNEL_H,   ATTR_CONV_KERNEL_W, ATTR_CONV_STRIDE_H,
  ATTR_CONV_STRIDE_W,  ATTR_CONV_DILATION_H, ATTR_CONV_DILATION_W, ATTR_CONV_PAD_LEFT, ATTR_CONV_PAD_RIGHT,
  ATTR_CONV_PAD_TOP,   ATTR_CONV_PAD_BOTTOM, ATTR_CONV_TILE_H,     ATTR_CONV_TILE_W};

const std::unordered_map<std::string, air::ir::ForType> AkgSupportedForType = {
  {FOR_SERIAL, air::ir::ForType::Serial},
  {FOR_PARALLEL, air::ir::ForType::Parallel},
  {FOR_VECTORIZED, air::ir::ForType::Vectorized},
  {FOR_UNROLLED, air::ir::ForType::Unrolled},
  {FOR_SWIZZLED, air::ir::ForType::Swizzled}};
}  // namespace ir
}  // namespace akg
#endif  // POLY_UTIL_H_
