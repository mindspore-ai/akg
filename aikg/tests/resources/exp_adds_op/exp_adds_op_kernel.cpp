#include "kernel_operator.h"
using namespace AscendC;

#define __aicore__ [aicore]
inline __attribute__((always_inline)) __aicore__ void exp_adds_0_kernel(__gm__ uint8_t *__restrict__ gm_input, __gm__ uint8_t *__restrict__ gm_output) {
  __gm__ half * gm_input_1 = ((__gm__ half *)gm_input) + (int32_t)1280 * (int32_t)block_idx;
  __ubuf__ half * ub_tmp_2 = (__ubuf__ half *)((uintptr_t)(0));
  __ubuf__ half * ub_exp_3 = (__ubuf__ half *)((uintptr_t)(2560));
  __ubuf__ half * ub_add_5 = (__ubuf__ half *)((uintptr_t)(0));
  __gm__ half * gm_output_6 = ((__gm__ half *)gm_output) + (int32_t)1280 * (int32_t)block_idx;
  if ((int)block_idx * 5 < 36) {
    copy_gm_to_ubuf(ub_tmp_2, (__gm__ half *)gm_input_1, 0, 1, 80, 0, 0);
  }
  set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
  vexp(ub_exp_3 + 0, ub_tmp_2 + 0, (uint16_t)10, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);
  pipe_barrier(PIPE_V);
  vadds(ub_add_5 + 0, ub_exp_3 + 0, (half)1.0000000000, (uint8_t)10, (uint16_t)1, (uint16_t)1, (uint16_t)8, (uint16_t)8);
  set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
  if ((int)block_idx * 5 < 36) {
    copy_ubuf_to_gm((__gm__ half *)gm_output_6, ub_add_5, 0, 1, 80, 0, 0);
  }
}

extern "C" __global__ __aicore__ void exp_adds_op_custom(GM_ADDR gm_input, GM_ADDR gm_output, GM_ADDR workspace, GM_ADDR tiling)
{
    exp_adds_0_kernel(gm_input, gm_output);
}

#ifndef __CCE_KT_TEST__
void exp_adds_op_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, 
    uint8_t* workspace, uint8_t* tiling)
{
    exp_adds_op_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif