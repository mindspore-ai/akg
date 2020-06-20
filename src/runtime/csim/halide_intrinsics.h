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

#ifndef RUNTIME_CSIM_HALIDE_INTRINSICS_H_
#define RUNTIME_CSIM_HALIDE_INTRINSICS_H_

#include "aicore_fast_sim.h"

float vrelu(float a);
float relu(float a);
float rec(float a);
float rsqrt(float a);
float log(float a);
float exp(float a);

static int type_annotation() { return 0; }

template <typename T>
static T *tvm_access_ptr(int dummy_type_annotation, T *buffer, int begin_addr, int access_length, int read_or_write) {
  CHECK_GE(begin_addr, 0);
  CHECK_GE(access_length, 0);
  return buffer + begin_addr;
}

template <typename T_cond, typename T_then, typename T_else>
static T_then tvm_if_then_else(const T_cond &condition, const T_then &then_case, const T_else &else_case) {
  return condition ? then_case : (T_then)else_case;
}

template <typename T_red, typename T_inc>
static T_red mad(const T_red &reduce, const T_inc &increment) {
  return reduce + increment;
}

template <typename Ta, typename Tb>
static Ta bitwise_and(const Ta &a, const Tb &b) {
  return a & b;
}

template <typename Ta, typename Tb>
static Ta bitwise_or(const Ta &a, const Tb &b) {
  return a | b;
}

template <typename Ta, typename Tb>
static Ta vaxpy(const Ta &a, const Ta &b, const Tb &c) {
  return a * (Ta)c + b;
}

// ideally there should be a definition table of string constants, but for now we simply replace it by 0
static int tvm_cce_string_print(const char *s) { return 0; }

enum pipe_t {
  PIPE_S = 0,  // Scalar Pipe
  PIPE_V,      // Vector Pipe, including{VectorOP write UB,  L0C->UB write}
  PIPE_M,      // Matrix Pipe, including{}
  PIPE_MTE1,   // L1->L0{A,B}
  PIPE_MTE2,   // OUT ->{L1, L0{A,B}, UB}
  PIPE_MTE3,   // UB ->{OUT,L1}
  PIPE_ALL,
};

// Event Id
enum event_t {
  EVENT_ID0 = 0,
  EVENT_ID1,
  EVENT_ID2,
  EVENT_ID3,
  EVENT_ID_DUMMY,  // mark the end of event IDs
};

enum ConvRelu_t {
  CRMODE_NONE = 0,
  CRMODE_F32toF16_NONE = 1,
  CRMODE_F32toF16_RELU = 2,
  CRMODE_S32toF16_NONE = 3,
  CRMODE_F16toF32_NONE = 4,
  CRMODE_NONE_RELU = 5,
};

enum pad_t {
  PAD_NONE = 0,
  PAD_MODE1 = 1,
  PAD_MODE2 = 2,
  PAD_MODE3 = 3,
  PAD_MODE4 = 4,
  PAD_MODE5 = 5,
};

enum csize_t {
  CSIZE0 = 0,
  CSIZE1 = 1,
};

enum ub_addr8_t {
  VA0 = 0,
  VA1,
  VA2,
  VA3,
  VA4,
  VA5,
  VA6,
  VA7,
};

enum vtype_t {
  b8 = 0,
  b16 = 1,
  b32 = 2,
  s8 = 3,
  s32 = 4,
  f16 = 5,
  f32 = 6,
  fmix = 7,
};

int64_t abs(int64_t in);
int64_t min(int64_t in1, int64_t in2);
int64_t max(int64_t in1, int64_t in2);
int64_t sqrt(int64_t in);
void create_ca_matrix(__ca__ half *dst, uint8_t repeat, half value);
void create_cb_matrix(__cb__ half *dst, uint8_t repeat, half value);
void pipe_barrier(pipe_t pipe);
void set_flag(pipe_t pipe, pipe_t tpipe, event_t n);
void wait_flag(pipe_t pipe, pipe_t tpipe, event_t n);
void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride, pad_t pad_mode);
void copy_gm_to_ubuf(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride);
void copy_ubuf_to_cbuf(__cbuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride);
void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride);
void check_crmode(ConvRelu_t cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode);
void copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride);
void copy_cbuf_to_ubuf(__ubuf__ void *dst, __cbuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride);
void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
                     uint8_t sid, bool transpose);
void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
                     uint8_t sid, bool transpose);
void load_gm_generic(half *dst, half *src, uint64_t config);
void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint64_t config);
void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint64_t config);
void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint64_t config);
void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c);
void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c);
void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c);
void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c);
void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c);
void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c);
void broadcast_ub_to_cc(__cc__ half *dst, __ubuf__ half *xn, uint64_t config);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ uint32_t *xn, uint64_t config);
void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ half *xn, uint64_t config);
void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n, bool init_val_control_c);
void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n, bool init_val_control_c);
void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c);
void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c);
void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c);
void set_vector_mask(uint64_t m1, uint64_t m0);
void set_vector_mask_dup(uint64_t m);
void set_va_reg(ub_addr8_t addr, uint64_t *array);
void vector_dup(__ubuf__ uint16_t *dst, uint16_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vector_dup(__ubuf__ half *dst, half a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vector_dup(__ubuf__ uint32_t *dst, uint32_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vector_dup(__ubuf__ int32_t *dst, int32_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vector_dup(__ubuf__ float *dst, float a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1);
void scatter_vabs(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride);
void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void scatter_vexp(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride);
void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
float exp(float a);
float vrelu(float a);
float relu(float a);
void scatter_vrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride);
float rec(float a);
void scatter_vrec(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride);
void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void scatter_vln(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                 uint16_t src_stride);
void vln(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
         uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vln(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
         uint8_t dst_stride_m1, uint8_t src_stride_m1);
float log(float a);
float rsqrt(float a);
void scatter_vrsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                    uint16_t src_stride);
void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
            uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
            uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vnot(__ubuf__ void *dst, __ubuf__ void *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1);
void scatter_vadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride);
void vadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vadd(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void scatter_vsub(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride);
void vsub(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vsub(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vsub(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void scatter_vmul(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride);
void vmul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmul(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void scatter_vmax(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride);
void vmax(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmax(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmax(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void scatter_vmin(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride);
void vmin(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmin(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vmin(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void scatter_vmadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void vmadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void scatter_vmaddrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void vmaddrelu(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmaddrelu(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void scatter_vmla(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void vmla(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vmla(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint64_t config);
void vmla(__ubuf__ float *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vor(__ubuf__ void *dst, __ubuf__ void *src0, __ubuf__ void *src1, uint8_t repeat, uint8_t dst_stride_m0,
         uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
         uint8_t src1_stride_m1);
void vand(__ubuf__ void *dst, __ubuf__ void *src0, __ubuf__ void *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void set_cmpmask(__ubuf__ void *src);
void get_cmpmask(__ubuf__ void *dst);
void vcmp_eq(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_eq(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_ne(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_ne(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_lt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_lt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_le(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_le(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_gt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_gt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_ge(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmp_ge(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1);
void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1);
void scatter_vsel(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);
void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1);
void vextract(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vconcat(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vmergech(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void vmergech(__ubuf__ uint8_t *dst, __ubuf__ uint8_t *src, uint64_t config);
void vrpac(__ubuf__ half *dst, __ubuf__ half *src, uint64_t config);
void viou(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void vaadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t config);
void rpn_cor(__ubuf__ half *dst, __ubuf__ half *src0, uint64_t config);
void rpn_cor_diag(__ubuf__ half *dst, __ubuf__ half *src0);
void vcadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1);
void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1);
void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1);
void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1);
void vcgmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride);
void vcgmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride);
void vcgadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride);
void vtranspose(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride);
void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride);
void vadds(half *dst, half *src, half a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vadds(float *dst, float *src, float a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, half a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride);
void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, float a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride);
void vmuls(half *dst, half *src, half a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vmuls(float *dst, float *src, float a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f322f16(half *dst, float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162f32(float *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162s8(int8_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162u8(uint8_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_deq(half *dst, int32_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
               uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162s32f(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162s32c(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_f162s32r(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_u82f16(half *dst, uint8_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_s82f16(half *dst, int8_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1);
void vconv_s322f32(float *dst, int32_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1);
void set_deqscale(half config);
void set_padding(uint64_t config);
void set_l1_3d_size(uint64_t config);
void set_fmatrix(uint64_t config);
void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, csize_t arg16);
void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, int arg16);
void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint16_t arg2, uint16_t arg3, uint8_t arg4,
                        uint8_t arg5, uint8_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint16_t arg10,
                        uint16_t arg11, uint16_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, uint8_t arg16,
                        uint8_t arg17, uint8_t arg18, uint8_t arg19, uint8_t arg20, uint8_t arg21, csize_t arg22);
void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt, int c);
void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, csize_t arg16);
void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, int arg16);
void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint16_t arg2, uint16_t arg3, uint8_t arg4,
                        uint8_t arg5, uint8_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint16_t arg10,
                        uint16_t arg11, uint16_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, uint8_t arg16,
                        uint8_t arg17, uint8_t arg18, uint8_t arg19, uint8_t arg20, uint8_t arg21, csize_t arg22);
void img2col_cbuf_to_cb(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt, int c);
void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        int c);
void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride, int pad_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode);
void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode);
void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode);

#endif  // RUNTIME_CSIM_HALIDE_INTRINSICS_H_
