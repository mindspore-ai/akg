#include "kernel_operator.h"
using namespace AscendC;

#define __aicore__ [aicore]
inline __attribute__((always_inline)) __aicore__ void reduce_sum_op_impl_npu_0_kernel(__gm__ uint8_t *__restrict__ gm_input, __gm__ uint8_t *__restrict__ gm_output) {
  __ubuf__ half * ub_input_7 = (__ubuf__ half *)((uintptr_t)(0));
  __ubuf__ float * ub_input_float_8 = (__ubuf__ float *)((uintptr_t)(4096));
  __ubuf__ float * ub_output_float_10 = (__ubuf__ float *)((uintptr_t)(512));
  __ubuf__ float * ub_9 = (__ubuf__ float *)((uintptr_t)(0));
  __ubuf__ half * ub_output_11 = (__ubuf__ half *)((uintptr_t)(0));
  int32_t block_idx_0 = (int32_t)block_idx;
  pipe_barrier(PIPE_S);
  int32_t scalar_2 = float(block_idx_0) * float(16);
  set_flag(PIPE_S, PIPE_MTE2, (event_t)0);
  wait_flag(PIPE_S, PIPE_MTE2, (event_t)0);
  copy_gm_to_ubuf(ub_input_7, (__gm__ half *)gm_input + (scalar_2) * 128, 0, 1, 128, 896, 0);
  set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  vconv_f162f32(ub_input_float_8 + 0, ub_input_7 + 0, 32, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  vadd(ub_9 + 0, ub_input_float_8 + 0, ub_input_float_8 + 8 + 0, 2, 1, 16, 16, 8, 128, 128);
  for (int rep = 2; rep < 16; ++rep) {
    pipe_barrier(PIPE_V);
    vadd(ub_9 + 0, ub_9 + 0, ub_input_float_8 + rep * 8 + 0, 2, 1, 1, 16, 8, 8, 128);
  }
  pipe_barrier(PIPE_V);
  vcgadd(ub_output_float_10 + 0, ub_9 + 0, 2, 1, 1, 8);
  pipe_barrier(PIPE_V);
  set_mask_norm();
  set_vector_mask(0ULL, 4294967295ULL);
  vconv_f322f16(ub_output_11 + 0, ub_output_float_10 + 0, 1, 1, 1, 4, 8);
  set_mask_norm();
  set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
  set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  copy_ubuf_to_gm((__gm__ half *)gm_output + scalar_2, ub_output_11, 0, 1, 1, 0, 7);
}
extern "C" __global__ __aicore__ void reduce_sum_op_custom(GM_ADDR gm_input, GM_ADDR gm_output, GM_ADDR workspace, GM_ADDR tiling) {
  reduce_sum_op_impl_npu_0_kernel(gm_input, gm_output);
}

#ifndef __CCE_KT_TEST__
void reduce_sum_op_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t * gm_input, uint8_t * gm_output, uint8_t* workspace, uint8_t* tiling) {
  reduce_sum_op_custom<<<8, l2ctrl, stream>>>(gm_input, gm_output, workspace, tiling);
}
#endif
