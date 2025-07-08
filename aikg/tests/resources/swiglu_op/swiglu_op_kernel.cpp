#include "kernel_operator.h"
using namespace AscendC;

#define __aicore__ [aicore]
inline __attribute__((always_inline)) __aicore__ void swiglu_op_impl_npu_0_kernel(__gm__ uint8_t *__restrict__ gm_input, __gm__ uint8_t *__restrict__ gm_output) {
  __ubuf__ half * ub_input_half_9 = (__ubuf__ half *)((uintptr_t)(0));
  __ubuf__ half * ub_input1_float_16 = (__ubuf__ half *)((uintptr_t)(7680));
  __ubuf__ float * ub_input1_float_22 = (__ubuf__ float *)((uintptr_t)(8960));
  __ubuf__ half * ub_input0_float_21 = (__ubuf__ half *)((uintptr_t)(2560));
  __ubuf__ float * ub_input0_float_23 = (__ubuf__ float *)((uintptr_t)(0));
  __ubuf__ float * neg_tile_25 = (__ubuf__ float *)((uintptr_t)(2560));
  __ubuf__ float * exp_tile_26 = (__ubuf__ float *)((uintptr_t)(5120));
  __ubuf__ float * exp_tile_28 = (__ubuf__ float *)((uintptr_t)(2560));
  __ubuf__ float * sigmoid_tile_29 = (__ubuf__ float *)((uintptr_t)(5120));
  __ubuf__ float * result_tile_30 = (__ubuf__ float *)((uintptr_t)(0));
  __ubuf__ half * ub_output_half_31 = (__ubuf__ half *)((uintptr_t)(2560));
  int32_t block_idx_0 = (int32_t)block_idx;
  pipe_barrier(PIPE_S);
  int32_t scalar_2 = float(block_idx_0) * float(5);
  set_flag(PIPE_S, PIPE_MTE2, (event_t)0);
  wait_flag(PIPE_S, PIPE_MTE2, (event_t)0);
  copy_gm_to_ubuf(ub_input_half_9, (__gm__ half *)gm_input + (scalar_2) * 256, 0, 1, 80, 560, 0);
  set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  copy_ubuf_to_ubuf(ub_input1_float_16, ub_input_half_9 + 128, 0, 5, 8, 8, 0);
  pipe_barrier(PIPE_V);
  vconv_f162f32(ub_input1_float_22 + 0, ub_input1_float_16 + 0, 10, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  copy_ubuf_to_ubuf(ub_input0_float_21, ub_input_half_9 + 0, 0, 5, 8, 8, 0);
  pipe_barrier(PIPE_V);
  vconv_f162f32(ub_input0_float_23 + 0, ub_input0_float_21 + 0, 10, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);
  vmuls(neg_tile_25 + 0, ub_input0_float_23 + 0, (float)-1.0000000000, 10, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vexp(exp_tile_26 + 0, neg_tile_25 + 0, 10, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vadds(exp_tile_28 + 0, exp_tile_26 + 0, (float)1.0000000000, 10, 1, 1, 8, 8);
  pipe_barrier(PIPE_V);
  vdiv(sigmoid_tile_29 + 0, ub_input0_float_23 + 0, exp_tile_28 + 0, 10, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vmul(result_tile_30 + 0, sigmoid_tile_29 + 0, ub_input1_float_22 + 0, 10, 1, 1, 1, 8, 8, 8);
  pipe_barrier(PIPE_V);
  vconv_f322f16(ub_output_half_31 + 0, result_tile_30 + 0, 10, 1, 1, 4, 8);
  set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  copy_ubuf_to_gm((__gm__ half *)gm_output + (scalar_2) * 128, ub_output_half_31, 0, 1, 40, 0, 280);
}
extern "C" __global__ __aicore__ void swiglu_op_custom(GM_ADDR gm_input, GM_ADDR gm_output, GM_ADDR workspace, GM_ADDR tiling) {
  swiglu_op_impl_npu_0_kernel(gm_input, gm_output);
}

#ifndef __CCE_KT_TEST__
void swiglu_op_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t * gm_input, uint8_t * gm_output, uint8_t* workspace, uint8_t* tiling) {
  swiglu_op_custom<<<blockDim, l2ctrl, stream>>>(gm_input, gm_output, workspace, tiling);
}
#endif
