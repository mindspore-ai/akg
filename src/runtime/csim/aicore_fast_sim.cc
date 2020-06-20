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

#include "aicore_fast_sim.h"

#define UB_BLOCK_SIZE 32
#define L0_BLOCK_SIZE 512
#define UB_BLOCK_SIZE_BYTES (UB_BLOCK_SIZE * sizeof(uint8_t))
#define L0_BLOCK_SIZE_BYTES (L0_BLOCK_SIZE * sizeof(uint8_t))

#define NUM_VA_REGS 8
#define NUM_VA_BLOCKS 8
static uint64_t va_reg[NUM_VA_REGS][NUM_VA_BLOCKS] __attribute__((aligned(UB_BLOCK_SIZE_BYTES))) = {0};

#define BYTES_PER_REPEAT (256 * sizeof(uint8_t))
#define NUM_BLOCKS_PER_REPEAT 8

#define MAX_ELEM_PER_REPEAT 128
static bool vector_mask[MAX_ELEM_PER_REPEAT] __attribute__((aligned(UB_BLOCK_SIZE_BYTES))) = {false};

static half g_deqscale = half(1.0f);

static uint64_t g_padding = 0;

static uint64_t g_l1_3d_size = 0;

static uint64_t g_fmatrix_config = 0;

#define MAD_BLOCK_SIZE 16
static half g_mad_regs[MAD_BLOCK_SIZE][MAD_BLOCK_SIZE] __attribute__((aligned(L0_BLOCK_SIZE_BYTES)));

#define NUM_CMPMASK 128
static bool g_cmpmask[NUM_CMPMASK] __attribute__((aligned(UB_BLOCK_SIZE_BYTES))) = {0};

#define CHECK_ALIGN(addr, alignment)                                                          \
  do {                                                                                        \
    CHECK((size_t)(addr) % ((alignment) * sizeof(uint8_t)) == 0)                              \
      << "Alignment check failed: address " << addr << " is not " << alignment << " aligned"; \
  } while (0);
// templates

static uint64_t get_bits(uint64_t config, uint8_t high_bit, uint8_t low_bit) {
  CHECK_GE(high_bit, low_bit);
  CHECK_LE(high_bit, 63);
  uint64_t mask = ((uint64_t)1 << high_bit) - 1;
  return (config & mask) >> low_bit;
}

template <typename T>
static void generic_unary_va(ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride,
                             T (*UnaryOp)(const T &)) {
  CHECK(dst < NUM_VA_REGS);
  CHECK(src < NUM_VA_REGS);
  const int elem_per_block = BYTES_PER_REPEAT / sizeof(T) / NUM_VA_REGS;
  for (int repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    for (int block = 0; block < NUM_VA_BLOCKS; ++block) {
      T *dst_block = reinterpret_cast<T *>(va_reg[dst][block] + dst_stride * elem_per_block * repeat_it);
      T *src_block = reinterpret_cast<T *>(va_reg[src][block] + src_stride * elem_per_block * repeat_it);
      for (int elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = UnaryOp(src_block[elem]);
        }
      }
    }
  }
}

// Template cannot be a parameter of a function, so I use a C macro instead.
// Please feel free to refactor this code if you know a more elegant way.
#define generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, UnaryOp)    \
  do {                                                                                     \
    switch (type) {                                                                        \
      case f16:                                                                            \
        generic_unary_va<half>(dst, src, repeat, dst_stride, src_stride, UnaryOp<half>);   \
        break;                                                                             \
      case f32:                                                                            \
        generic_unary_va<float>(dst, src, repeat, dst_stride, src_stride, UnaryOp<float>); \
        break;                                                                             \
      default:                                                                             \
        CHECK(false) << "Unsupported data type " << type << " in VA instruction";          \
    }                                                                                      \
  } while (0)

template <typename T>
static void generic_binary_va_imm(ub_addr8_t dst, ub_addr8_t src, T imm, uint8_t repeat, uint16_t dst_stride,
                                  uint16_t src_stride, T (*BinaryOp)(const T &, const T &)) {
  CHECK(dst < NUM_VA_REGS);
  CHECK(src < NUM_VA_REGS);
  const int elem_per_block = BYTES_PER_REPEAT / sizeof(T) / NUM_VA_REGS;
  for (int repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    for (int block = 0; block < NUM_VA_BLOCKS; ++block) {
      T *dst_block = reinterpret_cast<T *>(va_reg[dst][block] + dst_stride * elem_per_block * repeat_it);
      T *src_block = reinterpret_cast<T *>(va_reg[src][block] + src_stride * elem_per_block * repeat_it);
      for (int elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = BinaryOp(src_block[elem], imm);
        }
      }
    }
  }
}

// Template cannot be a parameter of a function, so I use a C macro instead.
// Please feel free to refactor this code if you know a more elegant way.
#define generic_binary_va_imm_vtype(type, dst, src, imm, repeat, dst_stride, src_stride, BinaryOp)    \
  do {                                                                                                \
    switch (type) {                                                                                   \
      case f16:                                                                                       \
        generic_binary_va_imm<half>(dst, src, imm, repeat, dst_stride, src_stride, BinaryOp<half>);   \
        break;                                                                                        \
      case f32:                                                                                       \
        generic_binary_va_imm<float>(dst, src, imm, repeat, dst_stride, src_stride, BinaryOp<float>); \
        break;                                                                                        \
      default:                                                                                        \
        CHECK(false) << "Unsupported data type " << type << " in VA instruction";                     \
    }                                                                                                 \
  } while (0)

template <typename T_dst, typename T_src>
static void generic_unary_vec_2type(T_dst *dst, T_src *src, uint8_t repeat, uint16_t dst_stride_m0,
                                    uint16_t src_stride_m0, uint8_t dst_stride_m1, uint8_t src_stride_m1,
                                    T_dst (*UnaryOp)(const T_src &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  if (dst_stride_m0 == 0) {
    dst_stride_m0 = 1;
  }
  const int elem_size = sizeof(T_dst) > sizeof(T_src) ? sizeof(T_dst) : sizeof(T_src);
  const int bytes_per_block = BYTES_PER_REPEAT / NUM_BLOCKS_PER_REPEAT;
  const int elem_per_block = bytes_per_block / elem_size;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T_dst *dst_base = dst + dst_stride_m1 * repeat_it * bytes_per_block / sizeof(T_dst);
    T_src *src_base = src + src_stride_m1 * repeat_it * bytes_per_block / sizeof(T_src);
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T_dst *dst_block = dst_base + dst_stride_m0 * block * bytes_per_block / sizeof(T_dst);
      T_src *src_block = src_base + src_stride_m0 * block * bytes_per_block / sizeof(T_src);
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = UnaryOp(src_block[elem]);
        }
      }
    }
  }
}

template <typename T>
static void generic_unary_vec(T *dst, T *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                              uint8_t dst_stride_m1, uint8_t src_stride_m1, T (*UnaryOp)(const T &)) {
  generic_unary_vec_2type<T, T>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1, UnaryOp);
}

template <typename T>
static void generic_unary_vec_imm(T *dst, T src, uint8_t repeat, uint16_t dst_stride_m0, uint8_t dst_stride_m1,
                                  T (*UnaryOp)(const T &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  if (dst_stride_m0 == 0) {
    dst_stride_m0 = 1;
  }
  const int elem_size = sizeof(T);
  const int elem_per_block = BYTES_PER_REPEAT / elem_size / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T *dst_base = dst + dst_stride_m1 * repeat_it * elem_per_block;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T *dst_block = dst_base + dst_stride_m0 * block * elem_per_block;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = UnaryOp(src);
        }
      }
    }
  }
}

template <typename T>
static void generic_binary_vec_imm(T *dst, T *src, T imm, uint8_t repeat, uint16_t dst_stride_m0,
                                   uint16_t src_stride_m0, uint8_t dst_stride_m1, uint8_t src_stride_m1,
                                   T (*BinaryOp)(const T &, const T &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  if (dst_stride_m0 == 0) {
    dst_stride_m0 = 1;
  }
  const int elem_per_block = BYTES_PER_REPEAT / sizeof(T) / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T *dst_base = dst + dst_stride_m1 * repeat_it * elem_per_block;
    T *src_base = src + src_stride_m1 * repeat_it * elem_per_block;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T *dst_block = dst_base + dst_stride_m0 * block * elem_per_block;
      T *src_block = src_base + src_stride_m0 * block * elem_per_block;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = BinaryOp(src_block[elem], imm);
        }
      }
    }
  }
}

template <typename T_dst, typename T_src>
static void generic_reduce_2type(T_dst *dst, T_src *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
                                 uint16_t src_stride_m1, T_dst (*ReduceOp)(const T_dst &, const T_src &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  const int elem_size = sizeof(T_src);
  const int elem_per_block = BYTES_PER_REPEAT / elem_size / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T_src *src_base = src + src_stride_m1 * repeat_it * elem_per_block;
    T_dst reduce;
    bool is_first = true;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T_src *src_block = src_base + src_stride_m0 * block * elem_per_block;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          if (is_first) {
            reduce = src_block[elem];
            is_first = false;
          } else {
            reduce = ReduceOp(reduce, src_block[elem]);
          }
        }
      }
    }
    if (!is_first) {
      dst[dst_stride * repeat_it] = reduce;
    }
  }
}

template <typename T>
static void generic_reduce(T *dst, T *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
                           uint16_t src_stride_m1, T (*ReduceOp)(const T &, const T &)) {
  generic_reduce_2type<T, T>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, ReduceOp);
}

template <typename T_dst, typename T_src>
static void generic_reduce_group_2type(T_dst *dst, T_src *src, uint8_t repeat, uint16_t dst_stride,
                                       uint16_t src_stride_m0, uint16_t src_stride_m1,
                                       T_dst (*ReduceOp)(const T_dst &, const T_src &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src, UB_BLOCK_SIZE);

  const int elem_size = sizeof(T_src);
  const int elem_per_block = BYTES_PER_REPEAT / elem_size / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T_src *src_base = src + src_stride_m1 * repeat_it * elem_per_block;
    T_dst *dst_base = dst + dst_stride * repeat_it * elem_per_block;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T_src *src_block = src_base + src_stride_m0 * block * elem_per_block;
      bool is_first = true;
      T_dst reduce;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          if (is_first) {
            reduce = src_block[elem];
            is_first = false;
          } else {
            reduce = ReduceOp(reduce, src_block[elem]);
          }
        }
      }
      if (!is_first) {
        dst_base[block] = reduce;
      }
    }
  }
}

template <typename T>
static void generic_reduce_group(T *dst, T *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
                                 uint16_t src_stride_m1, T (*ReduceOp)(const T &, const T &)) {
  generic_reduce_group_2type<T, T>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, ReduceOp);
}

// AIcore function implementation
int64_t min(int64_t in1, int64_t in2) { return std::min(in1, in2); }

int64_t max(int64_t in1, int64_t in2) { return std::max(in1, in2); }

int64_t sqrt(int64_t in) { return std::sqrt(in); }

// tvm interface
static void create_matrix(half *dst, uint8_t repeat, const half &value) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  for (int i = 0; i < repeat; ++i) {
    for (int j = 0; j < MAD_BLOCK_SIZE; ++j) {
      for (int k = 0; k < MAD_BLOCK_SIZE; ++k) {
        dst[i * MAD_BLOCK_SIZE * MAD_BLOCK_SIZE + j * MAD_BLOCK_SIZE + k] = value;
      }
    }
  }
}

void create_ca_matrix(__ca__ half *dst, uint8_t repeat, const half &value) { create_matrix(dst, repeat, value); }

void create_cb_matrix(__cb__ half *dst, uint8_t repeat, const half &value) { create_matrix(dst, repeat, value); }

const pipe_t num_pipes = PIPE_ALL;
const event_t num_events = EVENT_ID_DUMMY;
static int is_flag_set[num_pipes][num_pipes][num_events] = {0};

inline pipe_t &operator++(pipe_t &p) {
  p = static_cast<pipe_t>(static_cast<int>(p) + 1);
  return p;
}

inline event_t &operator++(event_t &p) {
  p = static_cast<event_t>(static_cast<int>(p) + 1);
  return p;
}

inline int internal_get_flag(pipe_t pipe, pipe_t tpipe, event_t n) {
  return is_flag_set[static_cast<int>(pipe)][static_cast<int>(tpipe)][static_cast<int>(n)];
}

inline void internal_set_flag(pipe_t pipe, pipe_t tpipe, event_t n, int value) {
  is_flag_set[static_cast<int>(pipe)][static_cast<int>(tpipe)][static_cast<int>(n)] = value;
}

void pipe_barrier(pipe_t pipe) {
  if (pipe == PIPE_ALL) {
    for (pipe_t tpipe = static_cast<pipe_t>(0); tpipe < num_pipes; ++tpipe) {
      pipe_barrier(tpipe);
    }
    return;
  }
  CHECK(pipe < num_pipes);
  for (event_t event = static_cast<event_t>(0); event < num_events; ++event) {
    internal_set_flag(pipe, pipe, event, 0);
  }
}

void set_flag(pipe_t pipe, pipe_t tpipe, event_t n) {
  CHECK(n < num_events);
  if (tpipe == PIPE_ALL) {
    for (pipe_t p = static_cast<pipe_t>(0); p < num_pipes; ++p) {
      set_flag(pipe, p, n);
    }
    return;
  }
  if (pipe == PIPE_ALL) {
    for (pipe_t p = static_cast<pipe_t>(0); p < num_pipes; ++p) {
      set_flag(p, tpipe, n);
    }
    return;
  }
  CHECK(tpipe < num_pipes);
  if (pipe == tpipe) {
    return;
  }
  if (internal_get_flag(pipe, tpipe, n)) {
    LOG(WARNING) << "duplicate set flag: pipe " << pipe << " -> " << tpipe << " event_id " << n;
  }
  internal_set_flag(pipe, tpipe, n, 1);
}

void wait_flag(pipe_t pipe, pipe_t tpipe, event_t n) {
  CHECK(n < num_events);
  if (tpipe == PIPE_ALL) {
    for (pipe_t p = static_cast<pipe_t>(0); p < num_pipes; ++p) {
      wait_flag(pipe, p, n);
    }
    return;
  }
  if (pipe == PIPE_ALL) {
    for (pipe_t p = static_cast<pipe_t>(0); p < num_pipes; ++p) {
      wait_flag(p, tpipe, n);
    }
    return;
  }
  CHECK(tpipe < num_pipes);
  if (pipe == tpipe) {
    return;
  }
  CHECK(internal_get_flag(pipe, tpipe, n))
    << "possible deadlock: wait on flag " << pipe << " -> " << tpipe << " event_id " << n << " before it is set";
  internal_set_flag(pipe, tpipe, n, 0);
}

static inline void eltwise_copy(uint8_t *dst, uint8_t *src, size_t length) {
#ifdef ENABLE_CDIFF
  DisableUndefinedAssignCheck();
#endif
  for (size_t elem_in_burst = 0; elem_in_burst < length; ++elem_in_burst) {
    dst[elem_in_burst] = src[elem_in_burst];
  }
#ifdef ENABLE_CDIFF
  RestoreUndefinedAssignCheck();
#endif
}

static void generic_dma(void *dst, void *src, uint16_t n_burst, uint16_t len_burst, uint16_t burst_length_unit,
                        uint16_t src_stride, uint16_t src_gap_unit, uint16_t dst_stride, uint16_t dst_gap_unit,
                        bool src_need_align, bool dst_need_align) {
  if (src_need_align) {
    CHECK_ALIGN(src, burst_length_unit);
  }
  if (dst_need_align) {
    CHECK_ALIGN(dst, burst_length_unit);
  }
  CHECK(n_burst > 0) << "nBurst cannot be zero";
  uint8_t *dst_base = reinterpret_cast<uint8_t *>(dst);
  uint8_t *src_base = reinterpret_cast<uint8_t *>(src);
  for (size_t burst = 0; burst < n_burst; ++burst) {
    size_t burst_length = (size_t)len_burst * burst_length_unit;
    size_t dst_offset = burst * (burst_length + (size_t)dst_stride * dst_gap_unit);
    size_t src_offset = burst * (burst_length + (size_t)src_stride * src_gap_unit);
    eltwise_copy(dst_base + dst_offset, src_base + src_offset, burst_length);
  }
}

void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride, pad_t pad_mode) {
  if (pad_mode == 0) {
    generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE,
                false, true);
  } else {
    CHECK(false) << "pad not supported yet in copy_gm_to_cbuf";
  }
}

void copy_gm_to_ubuf(__ubuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride) {
  generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE, false,
              true);
}

void copy_ubuf_to_cbuf(__cbuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride) {
  generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE, true,
              true);
}

void copy_ubuf_to_gm(__gm__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride) {
  generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE, true,
              false);
}

void check_crmode(ConvRelu_t cr_mode) { CHECK(cr_mode == CRMODE_NONE) << "CRMODE not supported yet in copy_matrix"; }

template <typename T>
static void generic_copy_matrix(T *dst, T *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                                uint16_t burst_length_unit, uint16_t src_stride, uint16_t src_gap_unit,
                                uint16_t dst_stride, uint16_t dst_gap_unit, ConvRelu_t cr_mode) {
  check_crmode(cr_mode);
  CHECK_ALIGN(dst, dst_gap_unit);
  CHECK_ALIGN(src, src_gap_unit);

  for (int burst = 0; burst < n_burst; ++burst) {
    const size_t burst_size = (size_t)len_burst * burst_length_unit;
    eltwise_copy(reinterpret_cast<uint8_t *>(dst), reinterpret_cast<uint8_t *>(src), burst_size);
    src += (burst_size + (size_t)src_stride * src_gap_unit) * sizeof(uint8_t) / sizeof(T);
    dst += (burst_size + (size_t)dst_stride * dst_gap_unit) * sizeof(uint8_t) / sizeof(T);
  }
}

template <typename T_dst, typename T_src>
static void generic_copy_matrix_conv(T_dst *dst, T_src *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                                     uint16_t burst_length_unit, uint16_t src_stride, uint16_t src_gap_unit,
                                     uint16_t dst_stride, uint16_t dst_gap_unit, ConvRelu_t cr_mode) {
  CHECK_ALIGN(dst, dst_gap_unit);
  CHECK_ALIGN(src, src_gap_unit);

  for (int burst = 0; burst < n_burst; ++burst) {
    const size_t element_size = (sizeof(T_src) > sizeof(T_dst) ? sizeof(T_src) : sizeof(T_dst));
    const size_t num_elements = (size_t)len_burst * burst_length_unit / element_size;
    for (size_t i = 0; i < num_elements; ++i) {
      dst[i] = static_cast<T_dst>(src[i]);
    }

    const size_t src_burst_size = num_elements * sizeof(T_src);
    const size_t dst_burst_size = num_elements * sizeof(T_dst);
    src += (src_burst_size + (size_t)src_stride * src_gap_unit * sizeof(uint8_t)) / sizeof(T_src);
    dst += (dst_burst_size + (size_t)dst_stride * dst_gap_unit * sizeof(uint8_t)) / sizeof(T_dst);
  }
}

void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<half>(dst, src, sid, n_burst, len_burst, 512, src_stride, 32, dst_stride, 512, cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<uint32_t>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<int32_t>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<float>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<float, half>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 32, dst_stride, 1024,
                                        cr_mode);
}

void copy_vector_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<half>(dst, src, sid, n_burst, len_burst, 32, src_stride, 32, dst_stride, 512, cr_mode);
}

void copy_vector_ubuf_to_cc(__cc__ uint32_t *dst, __ubuf__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<uint32_t>(dst, src, sid, n_burst, len_burst, 64, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_vector_ubuf_to_cc(__cc__ int32_t *dst, __ubuf__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<int32_t>(dst, src, sid, n_burst, len_burst, 64, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<float>(dst, src, sid, n_burst, len_burst, 64, src_stride, 32, dst_stride, 1024, cr_mode);
}

void copy_vector_ubuf_to_cc(__cc__ float *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<float, half>(dst, src, sid, n_burst, len_burst, 64, src_stride, 32, dst_stride, 1024,
                                        cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<half>(dst, src, sid, n_burst, len_burst, 512, src_stride, 512, dst_stride, 32, cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<uint32_t>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<int32_t>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<float>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<half, float>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 1024, dst_stride, 32,
                                        cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<half, int32_t>(dst, src, sid, n_burst, len_burst, 1024, src_stride, 1024, dst_stride, 32,
                                          cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<half>(dst, src, sid, n_burst, len_burst, 32, src_stride, 512, dst_stride, 32, cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ uint32_t *dst, __cc__ uint32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<uint32_t>(dst, src, sid, n_burst, len_burst, 64, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ int32_t *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst,
                            uint16_t len_burst, uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<int32_t>(dst, src, sid, n_burst, len_burst, 64, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix<float>(dst, src, sid, n_burst, len_burst, 64, src_stride, 1024, dst_stride, 32, cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<half, float>(dst, src, sid, n_burst, len_burst, 64, src_stride, 1024, dst_stride, 32,
                                        cr_mode);
}

void copy_vector_cc_to_ubuf(__ubuf__ half *dst, __cc__ int32_t *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, ConvRelu_t cr_mode) {
  generic_copy_matrix_conv<half, int32_t>(dst, src, sid, n_burst, len_burst, 64, src_stride, 1024, dst_stride, 32,
                                          cr_mode);
}

void copy_ubuf_to_ubuf(__ubuf__ void *dst, __ubuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride) {
  generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE, true,
              true);
}

void copy_cbuf_to_ubuf(__ubuf__ void *dst, __cbuf__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                       uint16_t src_stride, uint16_t dst_stride) {
  generic_dma(dst, src, n_burst, len_burst, UB_BLOCK_SIZE, src_stride, UB_BLOCK_SIZE, dst_stride, UB_BLOCK_SIZE, true,
              true);
}

static void load_2d(half *dst, half *src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride, uint8_t sid,
                    bool transpose) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  CHECK_ALIGN(src, L0_BLOCK_SIZE);
  const size_t elem_per_block = L0_BLOCK_SIZE / sizeof(half);
  src += base_idx * elem_per_block;
  for (int block = 0; block < repeat; ++block) {
    if (!transpose) {
      for (int i = 0; i < MAD_BLOCK_SIZE; ++i) {
        for (int j = 0; j < MAD_BLOCK_SIZE; ++j) {
          dst[i * MAD_BLOCK_SIZE + j] = src[i * MAD_BLOCK_SIZE + j];
        }
      }
    } else {
      for (int i = 0; i < MAD_BLOCK_SIZE; ++i) {
        for (int j = 0; j < MAD_BLOCK_SIZE; ++j) {
          dst[i * MAD_BLOCK_SIZE + j] = src[j * MAD_BLOCK_SIZE + i];
        }
      }
    }

    src += src_stride * elem_per_block;
    dst += elem_per_block;
  }
}

void load_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
                     uint8_t sid, bool transpose) {
  load_2d(dst, src, base_idx, repeat, src_stride, sid, transpose);
}

void load_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint16_t base_idx, uint8_t repeat, uint16_t src_stride,
                     uint8_t sid, bool transpose) {
  load_2d(dst, src, base_idx, repeat, src_stride, sid, transpose);
}

void load_gm_generic(half *dst, half *src, uint64_t config) {
  uint16_t base_idx = get_bits(config, 15, 0);
  uint8_t repeat = get_bits(config, 23, 16);
  uint16_t src_stride = get_bits(config, 39, 24);
  uint8_t sid = get_bits(config, 43, 40);
  bool transpose = false;
  load_2d(dst, src, base_idx, repeat, src_stride, sid, transpose);
}

void load_gm_to_ca(__ca__ half *dst, __gm__ half *src, uint64_t config) { load_gm_generic(dst, src, config); }

void load_gm_to_cb(__cb__ half *dst, __gm__ half *src, uint64_t config) { load_gm_generic(dst, src, config); }

void load_gm_to_cbuf(__cbuf__ half *dst, __gm__ half *src, uint64_t config) { load_gm_generic(dst, src, config); }

/*
 * Note: img2col feature is under development. Correctness not guaranteed.
 */
class img2col_class {
 public:
  img2col_class() = default;
  ~img2col_class() = default;
  void img2col(half *in_dst, half *in_src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt, csize_t c) {
    dst_ = in_dst;
    src_ = in_src;
    parse_params(fmatrix_config, xm, xt);
    check_params();
    compute();
  }

 private:
  void parse_params(uint64_t fmatrix_config, uint64_t xm, uint64_t xt) {
    fmap_w_ = get_bits(fmatrix_config, 15, 0);
    fmap_h_ = get_bits(fmatrix_config, 31, 16);
    pad_left_ = get_bits(fmatrix_config, 39, 32);
    pad_right_ = get_bits(fmatrix_config, 47, 40);
    pad_top_ = get_bits(fmatrix_config, 55, 48);
    pad_bottom_ = get_bits(fmatrix_config, 63, 56);

    filter_fetch_w_ = get_bits(xm, 23, 16);
    filter_fetch_h_ = get_bits(xm, 31, 24);
    fmap_start_w_ = get_bits(xm, 47, 32);
    if (fmap_start_w_ >= 32768) {
      fmap_start_w_ = (int16_t)(uint16_t)fmap_start_w_;
    }
    fmap_start_h_ = get_bits(xm, 63, 48);
    if (fmap_start_h_ >= 32768) {
      fmap_start_h_ = (int16_t)(uint16_t)fmap_start_h_;
    }
    c_channel_pos_ = get_bits(xm, 11, 0);

    stride_w_ = get_bits(xt, 5, 0);
    stride_h_ = get_bits(xt, 11, 6);
    filter_w_ = get_bits(xt, 19, 12);
    filter_h_ = get_bits(xt, 27, 20);
    dilation_w_ = get_bits(xt, 35, 28);
    dilation_h_ = get_bits(xt, 43, 36);
    jump_offset_ = get_bits(xt, 51, 44);
    repeat_mode_ = get_bits(xt, 52, 52);
    repeat_time_ = get_bits(xt, 63, 56);
  }

  void check_params() {
    CHECK_ALIGN(src_, UB_BLOCK_SIZE);

    CHECK_GE(fmap_w_, 1);
    CHECK_LE(fmap_w_, 32768);
    CHECK_GE(fmap_h_, 1);
    CHECK_LE(fmap_h_, 32768);
    CHECK_LE(filter_fetch_w_, 254);
    CHECK_LE(filter_fetch_h_, 254);
    CHECK_GE(fmap_start_w_, -255);
    CHECK_LE(fmap_start_w_, 32767);
    CHECK_GE(fmap_start_h_, -255);
    CHECK_LE(fmap_start_h_, 32767);
    CHECK_GE(stride_w_, 1);
    CHECK_GE(stride_h_, 1);
    CHECK_GE(filter_w_, 1);
    CHECK_GE(filter_h_, 1);
    CHECK_GE(dilation_w_, 1);
    CHECK_GE(dilation_h_, 1);
    CHECK_GE(jump_offset_, 1);
    CHECK_LE(jump_offset_, 127);
  }

  void compute() {
    for (size_t block = 0; block < repeat_time_; ++block) {
      init_pointers();
      for (int i = 0; i < MAD_BLOCK_SIZE; ++i) {
        for (int c0 = 0; c0 < MAD_BLOCK_SIZE; ++c0) {
          copy_point(c0);
        }
        src_fmatrix_next_row();
        dst_fmatrix_next_row();
      }
      src_next_fmatrix();
      dst_next_fmatrix();
    }
  }

  void init_pointers() {
    curr_fmap_w_ = fmap_start_w_;
    curr_fmap_h_ = fmap_start_h_;
  }

  void copy_point(int c0) {
    const half PAD_VALUE = half(.0f);
    half src_value = PAD_VALUE;
    if (curr_fmap_h_ < fmap_h_ && curr_fmap_w_ < fmap_w_) {
      src_value = *(src_ + c_channel_pos_ * fmap_h_ * fmap_w_ * MAD_BLOCK_SIZE +
                    curr_fmap_h_ * fmap_w_ * MAD_BLOCK_SIZE + curr_fmap_w_ * MAD_BLOCK_SIZE + c0);
    }
    half *curr_dst = dst_ + c0;
    *curr_dst = src_value;
  }

  void src_fmatrix_next_row() {
    curr_fmap_w_ += dilation_w_;
    if (curr_fmap_w_ >= fmap_w_ + pad_right_) {
      curr_fmap_w_ = -pad_left_;

      curr_fmap_h_ += dilation_h_;
      if (curr_fmap_h_ >= fmap_h_ + pad_bottom_) {
        curr_fmap_h_ = -pad_top_;
      }
    }
  }

  void dst_fmatrix_next_row() { dst_ += MAD_BLOCK_SIZE; }

  void src_next_fmatrix() {
    if (repeat_mode_ == 0) {
      next_filter();
    } else {
      next_fmap();
    }
  }

  void next_filter() {
    fmap_start_w_ += stride_w_;
    filter_fetch_w_ += 1;

    if (filter_fetch_w_ >= filter_w_) {
      fmap_start_w_ -= filter_fetch_w_ * stride_w_;
      filter_fetch_w_ = 0;

      fmap_start_h_ += stride_h_;
      filter_fetch_h_ += 1;

      if (filter_fetch_h_ >= filter_h_) {
        fmap_start_h_ -= filter_fetch_h_ * stride_h_;
        filter_fetch_h_ = 0;

        c_channel_pos_ += 1;
      }
    }
  }

  void next_fmap() {
    fmap_start_w_ = curr_fmap_w_;
    fmap_start_h_ = curr_fmap_h_;
  }

  void dst_next_fmatrix() {
    if (repeat_mode_ == 1) {
      dst_ += (jump_offset_ - 1) * MAD_BLOCK_SIZE * MAD_BLOCK_SIZE;
    }
  }

 private:
  half *dst_{nullptr};
  half *src_{nullptr};
  size_t fmap_w_{0};
  size_t fmap_h_{0};
  size_t pad_left_{0};
  size_t pad_right_{0};
  size_t pad_top_{0};
  size_t pad_bottom_{0};
  size_t filter_fetch_w_{0};
  size_t filter_fetch_h_{0};
  int fmap_start_w_{0};
  int fmap_start_h_{0};
  size_t c_channel_pos_{0};
  size_t stride_w_{0};
  size_t stride_h_{0};
  size_t filter_w_{0};
  size_t filter_h_{0};
  size_t dilation_w_{0};
  size_t dilation_h_{0};
  size_t jump_offset_{0};
  bool repeat_mode_{false};
  size_t repeat_time_{0};

  size_t curr_fmap_w_{0};
  size_t curr_fmap_h_{0};
};

static void img2col(half *dst, half *src, uint64_t xm, uint64_t xt, csize_t c) {
  img2col_class().img2col(dst, src, g_fmatrix_config, xm, xt, c);
}

void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  set_fmatrix(fmatrix_config);
  img2col(dst, src, xm, xt, c);
}

void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  img2col(dst, src, xm, xt, c);
}

void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  set_fmatrix(fmatrix_config);
  img2col(dst, src, xm, xt, c);
}

void img2col_cbuf_to_cb(__cb__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c) {
  CHECK_ALIGN(dst, L0_BLOCK_SIZE);
  img2col(dst, src, xm, xt, c);
}

void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        csize_t c) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  set_fmatrix(fmatrix_config);
  img2col(dst, src, xm, xt, c);
}

void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t xm, uint64_t xt, csize_t c) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  img2col(dst, src, xm, xt, c);
}

template <typename T_dst, typename T_src>
static void generic_broadcast_ub_to_cc(T_dst *dst, T_src *xn, uint64_t config, uint64_t dst_gap_size_unit) {
  size_t num_burst = get_bits(config, 7, 0);
  size_t burst_len = get_bits(config, 15, 8);
  size_t src_gap = get_bits(config, 23, 16);
  size_t dst_gap = get_bits(config, 31, 24);

  CHECK_GE(num_burst, 1);
  CHECK_GE(burst_len, 1);
  size_t src_gap_bytes = src_gap * UB_BLOCK_SIZE * sizeof(uint8_t);
  size_t dst_gap_bytes = dst_gap * dst_gap_size_unit * sizeof(uint8_t);

  T_src *src = xn;
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  CHECK_ALIGN(dst, dst_gap_size_unit);

  const size_t block_size = UB_BLOCK_SIZE / sizeof(half);
  for (size_t burst = 0; burst < num_burst; ++burst) {
    for (size_t repeat = 0; repeat < burst_len; ++repeat) {
      for (size_t elem = 0; elem < block_size; ++elem) {
        dst[repeat * block_size + elem] = src[elem];
      }
    }

    src += block_size + src_gap_bytes / sizeof(T_src);
    dst += burst_len * block_size + dst_gap_bytes / sizeof(T_dst);
  }
}

void broadcast_ub_to_cc(__cc__ half *dst, __ubuf__ half *xn, uint64_t config) {
  generic_broadcast_ub_to_cc<half, half>(dst, xn, config, 512);
}

void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ uint32_t *xn, uint64_t config) {
  generic_broadcast_ub_to_cc<float, uint32_t>(dst, xn, config, 1024);
}

void broadcast_ub_to_cc(__cc__ float *dst, __ubuf__ half *xn, uint64_t config) {
  generic_broadcast_ub_to_cc<float, half>(dst, xn, config, 1024);
}

template <typename T_c, typename T_a, typename T_b>
void generic_mad(__cc__ T_c *c, __ca__ T_a *a, __cb__ T_b *b, uint16_t m, uint16_t k, uint16_t n,
                 bool init_val_control_c) {
  // variable alignment could be constraint at int range
  const int a_alignment = MAD_BLOCK_SIZE * MAD_BLOCK_SIZE * sizeof(T_a) / sizeof(uint8_t);
  const int b_alignment = MAD_BLOCK_SIZE * MAD_BLOCK_SIZE * sizeof(T_b) / sizeof(uint8_t);
  const int c_alignment = MAD_BLOCK_SIZE * MAD_BLOCK_SIZE * sizeof(T_c) / sizeof(uint8_t);
  CHECK_ALIGN(a, a_alignment);
  CHECK_ALIGN(b, b_alignment);
  CHECK_ALIGN(c, c_alignment);
  if (m == 0 || k == 0 || n == 0) {
    return;
  }

#define ceil_div(a, b) (((a) + (b)-1) / (b))
  const int ni_extent = MAD_BLOCK_SIZE;
  const int no_extent = ceil_div(n, ni_extent);
  const int mi_extent = MAD_BLOCK_SIZE;
  const int mo_extent = ceil_div(m, mi_extent);
  const int ki_extent = MAD_BLOCK_SIZE;
  const int ko_extent = ceil_div(k, ki_extent);
#undef ceil_div

  for (int no = 0; no < no_extent; ++no) {
    for (int mo = 0; mo < mo_extent; ++mo) {
      for (int mi = 0; mi < mi_extent && mi + mo * mo_extent < m; ++mi) {
        for (int ni = 0; ni < ni_extent && ni + no * no_extent < n; ++ni) {
#define addr(i1, i2, i3, i4) (((i1 * i2##_extent + i2) * i3##_extent + i3) * i4##_extent + i4)
          T_c reduce;
          if (init_val_control_c) {
            reduce = (T_c)0;
          } else {
            reduce = c[addr(no, mo, mi, ni)];
          }
          for (int ko = 0; ko < ko_extent; ++ko) {
            for (int ki = 0; ki < ki_extent && ki + ko * ki_extent < k; ++ki) {
              reduce = reduce + (T_c)a[addr(mo, ko, mi, ki)] * (T_c)b[addr(ko, no, ni, ki)];
            }
          }
          c[addr(no, mo, mi, ni)] = reduce;
#undef addr
        }
      }
    }
  }
}

void mad(__cc__ float *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n, bool init_val_control_c) {
  generic_mad<float, half, half>(c, a, b, m, k, n, init_val_control_c);
}

void mad(__cc__ half *c, __ca__ half *a, __cb__ half *b, uint16_t m, uint16_t k, uint16_t n, bool init_val_control_c) {
  generic_mad<half, half, half>(c, a, b, m, k, n, init_val_control_c);
}

void mad(__cc__ uint32_t *c, __ca__ uint8_t *a, __cb__ uint8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c) {
  generic_mad<uint32_t, uint8_t, uint8_t>(c, a, b, m, k, n, init_val_control_c);
}

void mad(__cc__ int32_t *c, __ca__ int8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c) {
  generic_mad<int32_t, int8_t, int8_t>(c, a, b, m, k, n, init_val_control_c);
}

void mad(__cc__ int32_t *c, __ca__ uint8_t *a, __cb__ int8_t *b, uint16_t m, uint16_t k, uint16_t n,
         bool init_val_control_c) {
  generic_mad<int32_t, uint8_t, int8_t>(c, a, b, m, k, n, init_val_control_c);
}

void set_vector_mask(uint64_t m1, uint64_t m0) {
  for (int i = 0; i < 64; ++i) {
    vector_mask[i] = (m0 >> i) & (uint64_t)1;
    vector_mask[i + 64] = (m1 >> i) & (uint64_t)1;
  }
}

void set_vector_mask_dup(uint64_t m) { set_vector_mask(m, m); }

void set_va_reg(ub_addr8_t addr, uint64_t *array) {
  CHECK(addr < NUM_VA_REGS);
  for (int i = 0; i < NUM_VA_BLOCKS; ++i) {
    CHECK_ALIGN(array[i], UB_BLOCK_SIZE);
    va_reg[addr][i] = array[i];
  }
}

template <typename T>
static T unary_assign(const T &a) {
  return a;
}

void vector_dup(__ubuf__ uint16_t *dst, uint16_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_imm<uint16_t>(dst, a, repeat, dst_stride_m0, dst_stride_m1, unary_assign<uint16_t>);
}

void vector_dup(__ubuf__ half *dst, const half &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_imm<half>(dst, a, repeat, dst_stride_m0, dst_stride_m1, unary_assign<half>);
}

void vector_dup(__ubuf__ uint32_t *dst, uint32_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_imm<uint32_t>(dst, a, repeat, dst_stride_m0, dst_stride_m1, unary_assign<uint32_t>);
}

void vector_dup(__ubuf__ int32_t *dst, int32_t a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_imm<int32_t>(dst, a, repeat, dst_stride_m0, dst_stride_m1, unary_assign<int32_t>);
}

void vector_dup(__ubuf__ float *dst, const float &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_imm<float>(dst, a, repeat, dst_stride_m0, dst_stride_m1, unary_assign<float>);
}

template <typename T>
static T unary_abs(const T &a) {
  return static_cast<T>(std::abs(static_cast<double>(a)));
}

void scatter_vabs(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride) {
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_abs);
}

void vabs(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                          unary_abs<half>);
}

void vabs(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1, unary_abs);
}

template <typename T>
static T unary_exp(const T &a) {
  return static_cast<T>(std::exp(static_cast<double>(a)));
}

void scatter_vexp(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride) {
  CHECK(type == f16);
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_exp);
}

void vexp(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                          unary_exp<half>);
}

void vexp(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                           unary_exp<float>);
}

float exp(float a) { return unary_exp(a); }

template <typename T>
static T unary_relu(const T &a) {
  return a > static_cast<T>(0) ? a : static_cast<T>(0);
}

float vrelu(float a) { return unary_relu(a); }

float relu(float a) { return unary_relu(a); }

void scatter_vrelu(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride) {
  CHECK_EQ(type, f16);
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_relu);
}

template <typename T>
static T unary_rec(const T &a) {
  CHECK_NE(a, static_cast<T>(0));
  return static_cast<T>(1) / a;
}

float rec(float a) { return unary_rec(a); }

void scatter_vrec(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src_stride) {
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_rec);
}

void vrec(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                          unary_rec<half>);
}

void vrec(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                           unary_rec<float>);
}

template <typename T>
static T unary_ln(const T &a) {
  return static_cast<T>(std::log(static_cast<double>(a)));
}

void scatter_vln(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                 uint16_t src_stride) {
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_ln);
}

void vln(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
         uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1, unary_ln<half>);
}

void vln(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
         uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                           unary_ln<float>);
}

float log(float a) { return unary_ln(a); }

template <typename T>
static T unary_rsqrt(const T &a) {
  return static_cast<T>(1.0 / std::sqrt(static_cast<double>(a)));
}

float rsqrt(float a) { return unary_rsqrt(a); }

void scatter_vrsqrt(vtype_t type, ub_addr8_t dst, ub_addr8_t src, uint8_t repeat, uint16_t dst_stride,
                    uint16_t src_stride) {
  generic_unary_va_vtype(type, dst, src, repeat, dst_stride, src_stride, unary_rsqrt);
}

void vrsqrt(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
            uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                          unary_rsqrt<half>);
}

void vrsqrt(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
            uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                           unary_rsqrt<float>);
}

template <typename T>
static T unary_not(const T &a) {
  return ~a;
}

void vnot(__ubuf__ void *dst, __ubuf__ void *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec<uint16_t>(reinterpret_cast<uint16_t *>(dst), reinterpret_cast<uint16_t *>(src), repeat,
                              dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1, unary_not<uint16_t>);
}

template <typename T>
static void generic_binary_va(ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                              uint16_t src0_stride, uint16_t src1_stride, T (*BinaryOp)(const T &, const T &)) {
  CHECK(dst < NUM_VA_REGS);
  CHECK(src0 < NUM_VA_REGS);
  CHECK(src1 < NUM_VA_REGS);

  const int elem_per_block = BYTES_PER_REPEAT / sizeof(T) / NUM_VA_REGS;
  for (int repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    for (int block = 0; block < NUM_VA_BLOCKS; ++block) {
      T *dst_block = reinterpret_cast<T *>(va_reg[dst][block] + dst_stride * elem_per_block * repeat_it);
      T *src0_block = reinterpret_cast<T *>(va_reg[src0][block] + src0_stride * elem_per_block * repeat_it);
      T *src1_block = reinterpret_cast<T *>(va_reg[src1][block] + src1_stride * elem_per_block * repeat_it);
      for (int elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = BinaryOp(src0_block[elem], src1_block[elem]);
        }
      }
    }
  }
}

// Template cannot be a parameter of a function, so I use a C macro instead.
// Please feel free to refactor this code if you know a more elegant way.
#define generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, BinaryOp)        \
  do {                                                                                                                \
    switch (type) {                                                                                                   \
      case b16:                                                                                                       \
        generic_binary_va<uint16_t>(dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride,                    \
                                    BinaryOp<uint16_t>);                                                              \
        break;                                                                                                        \
      case s32:                                                                                                       \
        generic_binary_va<int32_t>(dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, BinaryOp<int32_t>); \
        break;                                                                                                        \
      case f16:                                                                                                       \
        generic_binary_va<half>(dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, BinaryOp<half>);       \
        break;                                                                                                        \
      case f32:                                                                                                       \
        generic_binary_va<float>(dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, BinaryOp<float>);     \
        break;                                                                                                        \
      default:                                                                                                        \
        CHECK(false) << "Unsupported data type " << type << " in VA instruction";                                     \
    }                                                                                                                 \
  } while (0)

template <typename T_dst, typename T_src>
static void generic_binary_vec_2type(T_dst *dst, T_src *src0, T_src *src1, uint8_t repeat, uint8_t dst_stride_m0,
                                     uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1,
                                     uint8_t src0_stride_m1, uint8_t src1_stride_m1,
                                     T_dst (*BinaryOp)(const T_src &, const T_src &)) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src0, UB_BLOCK_SIZE);
  CHECK_ALIGN(src1, UB_BLOCK_SIZE);
  if (dst_stride_m0 == 0) {
    dst_stride_m0 = 1;
  }
  const int elem_size = sizeof(T_dst) > sizeof(T_src) ? sizeof(T_dst) : sizeof(T_src);
  const int elem_per_block = BYTES_PER_REPEAT / elem_size / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T_dst *dst_base = dst + dst_stride_m1 * repeat_it * elem_per_block;
    T_src *src0_base = src0 + src0_stride_m1 * repeat_it * elem_per_block;
    T_src *src1_base = src1 + src1_stride_m1 * repeat_it * elem_per_block;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T_dst *dst_block = dst_base + dst_stride_m0 * block * elem_per_block;
      T_src *src0_block = src0_base + src0_stride_m0 * block * elem_per_block;
      T_src *src1_block = src1_base + src1_stride_m0 * block * elem_per_block;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = BinaryOp(src0_block[elem], src1_block[elem]);
        }
      }
    }
  }
}

template <typename T>
static void generic_binary_vec(T *dst, T *src0, T *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
                               uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
                               uint8_t src1_stride_m1, T (*BinaryOp)(const T &, const T &)) {
  generic_binary_vec_2type<T, T>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                                 src0_stride_m1, src1_stride_m1, BinaryOp);
}

template <typename T>
static T binary_add(const T &a, const T &b) {
  return a + b;
}

void scatter_vadd(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride) {
  generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, binary_add);
}

void vadd(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                           src0_stride_m1, src1_stride_m1, binary_add<half>);
}

void vadd(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<int32_t>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                              src0_stride_m1, src1_stride_m1, binary_add<int32_t>);
}

void vadd(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                            src0_stride_m1, src1_stride_m1, binary_add<float>);
}

template <typename T>
static T binary_sub(const T &a, const T &b) {
  return a - b;
}

void scatter_vsub(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride) {
  generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, binary_sub);
}

void vsub(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                           src0_stride_m1, src1_stride_m1, binary_sub<half>);
}

void vsub(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<int32_t>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                              src0_stride_m1, src1_stride_m1, binary_sub<int32_t>);
}

void vsub(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                            src0_stride_m1, src1_stride_m1, binary_sub<float>);
}

template <typename T>
static T binary_mul(const T &a, const T &b) {
  return a * b;
}

void scatter_vmul(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride) {
  generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, binary_mul);
}

void vmul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                           src0_stride_m1, src1_stride_m1, binary_mul<half>);
}

void vmul(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<int32_t>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                              src0_stride_m1, src1_stride_m1, binary_mul<int32_t>);
}

void vmul(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                            src0_stride_m1, src1_stride_m1, binary_mul<float>);
}

template <typename T>
static T binary_max(const T &a, const T &b) {
  return std::max(a, b);
}

void scatter_vmax(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride) {
  generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, binary_max);
}

void vmax(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                           src0_stride_m1, src1_stride_m1, binary_max<half>);
}

void vmax(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<int32_t>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                              src0_stride_m1, src1_stride_m1, binary_max<int32_t>);
}

void vmax(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                            src0_stride_m1, src1_stride_m1, binary_max<float>);
}

template <typename T>
static T binary_min(const T &a, const T &b) {
  return std::min(a, b);
}

void scatter_vmin(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint8_t repeat, uint16_t dst_stride,
                  uint16_t src0_stride, uint16_t src1_stride) {
  generic_binary_va_vtype(type, dst, src0, src1, repeat, dst_stride, src0_stride, src1_stride, binary_min);
}

void vmin(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                           src0_stride_m1, src1_stride_m1, binary_min<half>);
}

void vmin(__ubuf__ int32_t *dst, __ubuf__ int32_t *src0, __ubuf__ int32_t *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<int32_t>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                              src0_stride_m1, src1_stride_m1, binary_min<int32_t>);
}

void vmin(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                            src0_stride_m1, src1_stride_m1, binary_min<float>);
}

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

template <typename T>
static T binary_or(const T &a, const T &b) {
  return a | b;
}

void vor(__ubuf__ void *dst, __ubuf__ void *src0, __ubuf__ void *src1, uint8_t repeat, uint8_t dst_stride_m0,
         uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
         uint8_t src1_stride_m1) {
  generic_binary_vec<uint16_t>(reinterpret_cast<uint16_t *>(dst), reinterpret_cast<uint16_t *>(src0),
                               reinterpret_cast<uint16_t *>(src1), repeat, dst_stride_m0, src0_stride_m0,
                               src1_stride_m0, dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_or<uint16_t>);
}

template <typename T>
static T binary_and(const T &a, const T &b) {
  return a & b;
}

void vand(__ubuf__ void *dst, __ubuf__ void *src0, __ubuf__ void *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_binary_vec<uint16_t>(reinterpret_cast<uint16_t *>(dst), reinterpret_cast<uint16_t *>(src0),
                               reinterpret_cast<uint16_t *>(src1), repeat, dst_stride_m0, src0_stride_m0,
                               src1_stride_m0, dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_and<uint16_t>);
}

void set_cmpmask(__ubuf__ void *src) {
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  __ubuf__ uint8_t *src_byte = reinterpret_cast<__ubuf__ uint8_t *>(src);
  for (int byte = 0; byte < NUM_CMPMASK / 8; ++byte) {
    for (int bit = 0; bit < 8; ++bit) {
      g_cmpmask[byte * 8 + bit] = src_byte[byte] & (1 << bit);
    }
  }
}

void get_cmpmask(__ubuf__ void *dst) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  __ubuf__ uint8_t *dst_byte = reinterpret_cast<__ubuf__ uint8_t *>(dst);
  for (int byte = 0; byte < NUM_CMPMASK / 8; ++byte) {
    dst_byte[byte] = 0;
    for (int bit = 0; bit < 8; ++bit) {
      dst_byte[byte] |= (g_cmpmask[byte * 8 + bit] * (1 << bit));
    }
  }
}

template <typename T_dst, typename T_src>
static T_dst binary_eq(const T_src &a, const T_src &b) {
  return a == b;
}

void vcmp_eq(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_eq<bool, half>);
}

void vcmp_eq(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_eq<bool, float>);
}

template <typename T_dst, typename T_src>
static T_dst binary_ne(const T_src &a, const T_src &b) {
  return a != b;
}

void vcmp_ne(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ne<bool, half>);
}

void vcmp_ne(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ne<bool, float>);
}

template <typename T_dst, typename T_src>
static T_dst binary_lt(const T_src &a, const T_src &b) {
  return a < b;
}

void vcmp_lt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_lt<bool, half>);
}

void vcmp_lt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_lt<bool, float>);
}

template <typename T_dst, typename T_src>
static T_dst binary_le(const T_src &a, const T_src &b) {
  return a <= b;
}

void vcmp_le(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_le<bool, half>);
}

void vcmp_le(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_le<bool, float>);
}

template <typename T_dst, typename T_src>
static T_dst binary_gt(const T_src &a, const T_src &b) {
  return a > b;
}

void vcmp_gt(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_gt<bool, half>);
}

void vcmp_gt(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_gt<bool, float>);
}

template <typename T_dst, typename T_src>
static T_dst binary_ge(const T_src &a, const T_src &b) {
  return a >= b;
}

void vcmp_ge(__ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, half>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                       dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ge<bool, half>);
}

void vcmp_ge(__ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
             uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  generic_binary_vec_2type<bool, float>(g_cmpmask, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                        dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ge<bool, float>);
}

void vcmpv_eq(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_eq<uint8_t, half>);
}

void vcmpv_ne(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ne<uint8_t, half>);
}

void vcmpv_lt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_lt<uint8_t, half>);
}

void vcmpv_le(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_le<uint8_t, half>);
}

void vcmpv_gt(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_gt<uint8_t, half>);
}

void vcmpv_ge(__ubuf__ uint8_t *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
              uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
              uint8_t src1_stride_m1) {
  generic_binary_vec_2type<uint8_t, half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0,
                                          dst_stride_m1, src0_stride_m1, src1_stride_m1, binary_ge<uint8_t, half>);
}

void scatter_vsel(vtype_t type, ub_addr8_t dst, ub_addr8_t src0, ub_addr8_t src1, uint64_t config);

template <typename T>
void generic_vsel(T *dst, T *src0, T *src1, uint8_t repeat, uint8_t dst_stride_m0, uint8_t src0_stride_m0,
                  uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1, uint8_t src1_stride_m1) {
  if (dst_stride_m0 == 0) {
    dst_stride_m0 = 1;
  }
  const int elem_size = sizeof(T);
  const int elem_per_block = BYTES_PER_REPEAT / elem_size / NUM_BLOCKS_PER_REPEAT;
  for (size_t repeat_it = 0; repeat_it < repeat; ++repeat_it) {
    T *dst_base = dst + dst_stride_m1 * repeat_it * elem_per_block;
    T *src0_base = src0 + src0_stride_m1 * repeat_it * elem_per_block;
    T *src1_base = src1 + src1_stride_m1 * repeat_it * elem_per_block;
    for (size_t block = 0; block < NUM_BLOCKS_PER_REPEAT; ++block) {
      T *dst_block = dst_base + dst_stride_m0 * block * elem_per_block;
      T *src0_block = src0_base + src0_stride_m0 * block * elem_per_block;
      T *src1_block = src1_base + src1_stride_m0 * block * elem_per_block;
      for (size_t elem = 0; elem < elem_per_block; ++elem) {
        if (vector_mask[block * elem_per_block + elem]) {
          dst_block[elem] = (g_cmpmask[block * elem_per_block + elem] ? src0_block[elem] : src1_block[elem]);
        }
      }
    }
  }
}

void vsel(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_vsel<half>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                     src0_stride_m1, src1_stride_m1);
}

void vsel(__ubuf__ float *dst, __ubuf__ float *src0, __ubuf__ float *src1, uint8_t repeat, uint8_t dst_stride_m0,
          uint8_t src0_stride_m0, uint8_t src1_stride_m0, uint8_t dst_stride_m1, uint8_t src0_stride_m1,
          uint8_t src1_stride_m1) {
  generic_vsel<float>(dst, src0, src1, repeat, dst_stride_m0, src0_stride_m0, src1_stride_m0, dst_stride_m1,
                      src0_stride_m1, src1_stride_m1);
}

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
           uint16_t src_stride_m1) {
  generic_reduce<half>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, binary_add<half>);
}

void vcadd(__ubuf__ float *dst, __ubuf__ float *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1) {
  generic_reduce<float>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, binary_add<float>);
}

void vcmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1) {
  generic_reduce<half>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, binary_max<half>);
}

void vcmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src_stride_m0,
           uint16_t src_stride_m1) {
  generic_reduce<half>(dst, src, repeat, dst_stride, src_stride_m0, src_stride_m1, binary_min<half>);
}

void vcgmax(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride) {
  generic_reduce_group<half>(dst, src, repeat, dst_stride, src0_stride, src1_stride, binary_max<half>);
}

void vcgmin(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride) {
  generic_reduce_group<half>(dst, src, repeat, dst_stride, src0_stride, src1_stride, binary_min<half>);
}

void vcgadd(__ubuf__ half *dst, __ubuf__ half *src, uint8_t repeat, uint16_t dst_stride, uint16_t src0_stride,
            uint16_t src1_stride) {
  generic_reduce_group<half>(dst, src, repeat, dst_stride, src0_stride, src1_stride, binary_add<half>);
}

void vtranspose(__ubuf__ uint16_t *dst, __ubuf__ uint16_t *src) {
  CHECK_ALIGN(dst, UB_BLOCK_SIZE);
  CHECK_ALIGN(src, UB_BLOCK_SIZE);
  const int matrix_size = UB_BLOCK_SIZE / sizeof(uint16_t);
  for (int i = 0; i < matrix_size; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      dst[j * matrix_size + i] = src[i * matrix_size + j];
    }
  }
}

void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, const half &a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride) {
  CHECK(type == f16);
  generic_binary_va_imm<half>(dst, src, a, repeat, dst_stride, src_stride, binary_add<half>);
}

void scatter_vadds(vtype_t type, ub_addr8_t dst, ub_addr8_t src, const float &a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride) {
  CHECK(type == f32);
  generic_binary_va_imm<float>(dst, src, a, repeat, dst_stride, src_stride, binary_add<float>);
}

void vadds(half *dst, half *src, const half &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_binary_vec_imm<half>(dst, src, a, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                               binary_add<half>);
}

void vadds(float *dst, float *src, const float &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_binary_vec_imm<float>(dst, src, a, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                                binary_add<float>);
}

void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, const half &a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride) {
  CHECK(type == f16);
  generic_binary_va_imm<half>(dst, src, a, repeat, dst_stride, src_stride, binary_mul<half>);
}

void scatter_vmuls(vtype_t type, ub_addr8_t dst, ub_addr8_t src, const float &a, uint8_t repeat, uint16_t dst_stride,
                   uint16_t src_stride) {
  CHECK(type == f32);
  generic_binary_va_imm<float>(dst, src, a, repeat, dst_stride, src_stride, binary_mul<float>);
}

void vmuls(half *dst, half *src, const half &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_binary_vec_imm<half>(dst, src, a, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                               binary_mul<half>);
}

void vmuls(float *dst, float *src, const float &a, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
           uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_binary_vec_imm<float>(dst, src, a, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                                binary_mul<float>);
}

template <typename T_dst, typename T_src>
static T_dst unary_conv(const T_src &a) {
  return (T_dst)a;
}

template <typename T_dst, typename T_src>
static void generic_vconv(T_dst *dst, T_src *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                          uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_unary_vec_2type<T_dst, T_src>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1,
                                        unary_conv<T_dst, T_src>);
}

void vconv_f322f16(half *dst, float *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<half, float>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162f32(float *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<float, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162s8(int8_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<int8_t, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162u8(uint8_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<uint8_t, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_deq(half *dst, int32_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
               uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<half, int32_t>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162s32f(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<int32_t, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162s32c(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<int32_t, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_f162s32r(int32_t *dst, half *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                    uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<int32_t, half>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_u82f16(half *dst, uint8_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<half, uint8_t>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_s82f16(half *dst, int8_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                  uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<half, int8_t>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void vconv_s322f32(float *dst, int32_t *src, uint8_t repeat, uint16_t dst_stride_m0, uint16_t src_stride_m0,
                   uint8_t dst_stride_m1, uint8_t src_stride_m1) {
  generic_vconv<float, int32_t>(dst, src, repeat, dst_stride_m0, src_stride_m0, dst_stride_m1, src_stride_m1);
}

void set_deqscale(const half &config) { g_deqscale = config; }

void set_padding(uint64_t config) { g_padding = config; }

void set_l1_3d_size(uint64_t config) { g_l1_3d_size = config; }
void set_fmatrix(uint64_t config) { g_fmatrix_config = config; }

void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, csize_t arg16) {
  img2col_cbuf_to_ca(arg0, arg1,
                     (((uint64_t)arg2 & 0xff) << 16 | ((uint64_t)arg3 & 0xff) << 24 | ((uint64_t)arg4 & 0xffff) << 32 |
                      ((uint64_t)arg5 & 0xffff) << 48 | ((uint64_t)arg6 & 0xfff) << 0),
                     (((uint64_t)arg7 & 63) << 0 | ((uint64_t)arg8 & 63) << 6 | ((uint64_t)arg9 & 0xff) << 12 |
                      ((uint64_t)arg10 & 0xff) << 20 | ((uint64_t)arg11 & 0xff) << 28 | ((uint64_t)arg12 & 0xff) << 36 |
                      ((uint64_t)arg13 & 0xff) << 44 | ((uint64_t)arg14 & 1) << 52 | ((uint64_t)arg15 & 0xff) << 56),
                     arg16);
}

void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, int arg16) {
  img2col_cbuf_to_ca(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14,
                     arg15, (csize_t)arg16);
}

void img2col_cbuf_to_ca(__ca__ half *arg0, __cbuf__ half *arg1, uint16_t arg2, uint16_t arg3, uint8_t arg4,
                        uint8_t arg5, uint8_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint16_t arg10,
                        uint16_t arg11, uint16_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, uint8_t arg16,
                        uint8_t arg17, uint8_t arg18, uint8_t arg19, uint8_t arg20, uint8_t arg21, csize_t arg22) {
  img2col_cbuf_to_ca(arg0, arg1,
                     (((uint64_t)arg2 & 0xffff) << 0 | ((uint64_t)arg3 & 0xffff) << 16 | ((uint64_t)arg4 & 0xff) << 32 |
                      ((uint64_t)arg5 & 0xff) << 40 | ((uint64_t)arg6 & 0xff) << 48 | ((uint64_t)arg7 & 0xff) << 56),
                     (((uint64_t)arg8 & 0xff) << 16 | ((uint64_t)arg9 & 0xff) << 24 | ((uint64_t)arg10 & 0xffff) << 32 |
                      ((uint64_t)arg11 & 0xffff) << 48 | ((uint64_t)arg12 & 0xfff) << 0),
                     (((uint64_t)arg13 & 63) << 0 | ((uint64_t)arg14 & 63) << 6 | ((uint64_t)arg15 & 0xff) << 12 |
                      ((uint64_t)arg16 & 0xff) << 20 | ((uint64_t)arg17 & 0xff) << 28 | ((uint64_t)arg18 & 0xff) << 36 |
                      ((uint64_t)arg19 & 0xff) << 44 | ((uint64_t)arg20 & 1) << 52 | ((uint64_t)arg21 & 0xff) << 56),
                     arg22);
}

void img2col_cbuf_to_ca(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        int c) {
  img2col_cbuf_to_ca(dst, src, fmatrix_config, xm, xt, (csize_t)c);
}

void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, csize_t arg16) {
  img2col_cbuf_to_cb(arg0, arg1,
                     (((uint64_t)arg2 & 0xff) << 16 | ((uint64_t)arg3 & 0xff) << 24 | ((uint64_t)arg4 & 0xffff) << 32 |
                      ((uint64_t)arg5 & 0xffff) << 48 | ((uint64_t)arg6 & 0xfff) << 0),
                     (((uint64_t)arg7 & 63) << 0 | ((uint64_t)arg8 & 63) << 6 | ((uint64_t)arg9 & 0xff) << 12 |
                      ((uint64_t)arg10 & 0xff) << 20 | ((uint64_t)arg11 & 0xff) << 28 | ((uint64_t)arg12 & 0xff) << 36 |
                      ((uint64_t)arg13 & 0xff) << 44 | ((uint64_t)arg14 & 1) << 52 | ((uint64_t)arg15 & 0xff) << 56),
                     arg16);
}

void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint8_t arg2, uint8_t arg3, uint16_t arg4,
                        uint16_t arg5, uint16_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint8_t arg10,
                        uint8_t arg11, uint8_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, int arg16) {
  img2col_cbuf_to_cb(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10, arg11, arg12, arg13, arg14,
                     arg15, (csize_t)arg16);
}

void img2col_cbuf_to_cb(__cb__ half *arg0, __cbuf__ half *arg1, uint16_t arg2, uint16_t arg3, uint8_t arg4,
                        uint8_t arg5, uint8_t arg6, uint8_t arg7, uint8_t arg8, uint8_t arg9, uint16_t arg10,
                        uint16_t arg11, uint16_t arg12, uint8_t arg13, uint8_t arg14, uint8_t arg15, uint8_t arg16,
                        uint8_t arg17, uint8_t arg18, uint8_t arg19, uint8_t arg20, uint8_t arg21, csize_t arg22) {
  img2col_cbuf_to_cb(arg0, arg1,
                     (((uint64_t)arg2 & 0xffff) << 0 | ((uint64_t)arg3 & 0xffff) << 16 | ((uint64_t)arg4 & 0xff) << 32 |
                      ((uint64_t)arg5 & 0xff) << 40 | ((uint64_t)arg6 & 0xff) << 48 | ((uint64_t)arg7 & 0xff) << 56),
                     (((uint64_t)arg8 & 0xff) << 16 | ((uint64_t)arg9 & 0xff) << 24 | ((uint64_t)arg10 & 0xffff) << 32 |
                      ((uint64_t)arg11 & 0xffff) << 48 | ((uint64_t)arg12 & 0xfff) << 0),
                     (((uint64_t)arg13 & 63) << 0 | ((uint64_t)arg14 & 63) << 6 | ((uint64_t)arg15 & 0xff) << 12 |
                      ((uint64_t)arg16 & 0xff) << 20 | ((uint64_t)arg17 & 0xff) << 28 | ((uint64_t)arg18 & 0xff) << 36 |
                      ((uint64_t)arg19 & 0xff) << 44 | ((uint64_t)arg20 & 1) << 52 | ((uint64_t)arg21 & 0xff) << 56),
                     arg22);
}

void img2col_cbuf_to_cb(__ca__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        int c) {
  img2col_cbuf_to_cb(dst, src, fmatrix_config, xm, xt, (csize_t)c);
}

void img2col_cbuf_to_ub(__ubuf__ half *dst, __cbuf__ half *src, uint64_t fmatrix_config, uint64_t xm, uint64_t xt,
                        int c) {
  img2col_cbuf_to_ub(dst, src, fmatrix_config, xm, xt, (csize_t)c);
}

void copy_gm_to_cbuf(__cbuf__ void *dst, __gm__ void *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                     uint16_t src_stride, uint16_t dst_stride, int pad_mode) {
  copy_gm_to_cbuf(dst, src, sid, n_burst, len_burst, src_stride, dst_stride, (pad_t)pad_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ half *dst, __cc__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode) {
  copy_matrix_cc_to_ubuf(dst, src, sid, n_burst, len_burst, src_stride, dst_stride, (ConvRelu_t)cr_mode);
}

void copy_matrix_cc_to_ubuf(__ubuf__ float *dst, __cc__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode) {
  copy_matrix_cc_to_ubuf(dst, src, sid, n_burst, len_burst, src_stride, dst_stride, (ConvRelu_t)cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ half *dst, __ubuf__ half *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode) {
  copy_matrix_ubuf_to_cc(dst, src, sid, n_burst, len_burst, src_stride, dst_stride, (ConvRelu_t)cr_mode);
}

void copy_matrix_ubuf_to_cc(__cc__ float *dst, __ubuf__ float *src, uint8_t sid, uint16_t n_burst, uint16_t len_burst,
                            uint16_t src_stride, uint16_t dst_stride, int cr_mode) {
  copy_matrix_ubuf_to_cc(dst, src, sid, n_burst, len_burst, src_stride, dst_stride, (ConvRelu_t)cr_mode);
}
