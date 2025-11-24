#include "kernel_operator.h"
using namespace AscendC;

#define __aicore__ [aicore]
inline __attribute__((always_inline)) __aicore__ void swiglu_quant_op_impl_npu_0_kernel(__gm__ uint8_t *__restrict__ gm_x, __gm__ uint8_t *__restrict__ gm_smooth_scales, __gm__ uint8_t *__restrict__ gm_output, __gm__ uint8_t *__restrict__ gm_scale, __gm__ uint8_t *__restrict__ gm_swiglu_out) {
  __ubuf__ half * x1_16 = (__ubuf__ half *)((uintptr_t)(5632));
  __ubuf__ float * x1_f32_21 = (__ubuf__ float *)((uintptr_t)(6144));
  __ubuf__ half * x0_20 = (__ubuf__ half *)((uintptr_t)(0));
  __ubuf__ float * x0_f32_22 = (__ubuf__ float *)((uintptr_t)(512));
  __ubuf__ float * x0_mul_24 = (__ubuf__ float *)((uintptr_t)(1536));
  __ubuf__ float * x0_exp_25 = (__ubuf__ float *)((uintptr_t)(2560));
  __ubuf__ float * x0_add_27 = (__ubuf__ float *)((uintptr_t)(3584));
  __ubuf__ float * sigmoid_t_28 = (__ubuf__ float *)((uintptr_t)(4608));
  __ubuf__ float * swiglu_t_29 = (__ubuf__ float *)((uintptr_t)(7168));
  __ubuf__ half * swiglu_t_fp16_30 = (__ubuf__ half *)((uintptr_t)(8192));
  __ubuf__ half * smooth_tile_f16_6 = (__ubuf__ half *)((uintptr_t)(8704));
  __ubuf__ float * smooth_tile_f32_9 = (__ubuf__ float *)((uintptr_t)(9216));
  __ubuf__ float * q_mul_31 = (__ubuf__ float *)((uintptr_t)(10240));
  __ubuf__ float * q_abs_36 = (__ubuf__ float *)((uintptr_t)(8704));
  __ubuf__ float * q_max_38 = (__ubuf__ float *)((uintptr_t)(9760));
  __ubuf__ float * ub_37 = (__ubuf__ float *)((uintptr_t)(9728));
  __ubuf__ float * q_max_mul_40 = (__ubuf__ float *)((uintptr_t)(9728));
  __ubuf__ float * q_max_brcb_42 = (__ubuf__ float *)((uintptr_t)(19456));
  __ubuf__ float * ub_41 = (__ubuf__ float *)((uintptr_t)(11264));
  __ubuf__ float * q_brcb_mul_45 = (__ubuf__ float *)((uintptr_t)(11264));
  __ubuf__ half * q_tile_half_46 = (__ubuf__ half *)((uintptr_t)(12288));
  __ubuf__ int8_t * q_tile_int8_47 = (__ubuf__ int8_t *)((uintptr_t)(9792));
  int32_t scalar_2 = float(512) / float(8);
  for (int dynamic_loop_var_0_8 = 0; dynamic_loop_var_0_8 < scalar_2; ++dynamic_loop_var_0_8) {
    int32_t block_idx_7 = (int32_t)block_idx;
    pipe_barrier(PIPE_S);
    int32_t scalar_10 = float(block_idx_7) * float(scalar_2);
    pipe_barrier(PIPE_S);
    int32_t scalar_11 = float(scalar_10) + float(dynamic_loop_var_0_8);
    set_flag(PIPE_S, PIPE_MTE2, (event_t)0);
    wait_flag(PIPE_S, PIPE_MTE2, (event_t)0);
    copy_gm_to_ubuf(x1_16, (__gm__ half *)gm_x + ((scalar_11) * 512) + 256, 0, 1, 16, 16, 0);
    set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    vconv_f162f32(x1_f32_21 + 0, x1_16 + 0, 4, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    copy_gm_to_ubuf(x0_20, (__gm__ half *)gm_x + ((scalar_11) * 512) + 0, 0, 1, 16, 16, 0);
    set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    vconv_f162f32(x0_f32_22 + 0, x0_20 + 0, 4, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    vmuls(x0_mul_24 + 0, x0_f32_22 + 0, (float)-1.0000000000, 4, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vexp(x0_exp_25 + 0, x0_mul_24 + 0, 4, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vadds(x0_add_27 + 0, x0_exp_25 + 0, (float)1.0000000000, 4, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vdiv(sigmoid_t_28 + 0, x0_f32_22 + 0, x0_add_27 + 0, 4, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vmul(swiglu_t_29 + 0, sigmoid_t_28 + 0, x1_f32_21 + 0, 4, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16(swiglu_t_fp16_30 + 0, swiglu_t_29 + 0, 4, 1, 1, 4, 8);
    set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
    wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
    copy_ubuf_to_gm((__gm__ half *)gm_swiglu_out + (scalar_11) * 256, swiglu_t_fp16_30, 0, 1, 16, 0, 8176);
    copy_gm_to_ubuf(smooth_tile_f16_6, (__gm__ half *)gm_smooth_scales + 0, 0, 1, 16, 0, 0);
    set_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    wait_flag(PIPE_MTE2, PIPE_V, (event_t)0);
    vconv_f162f32(smooth_tile_f32_9 + 0, smooth_tile_f16_6 + 0, 4, 1, 1, 8, 4);
    pipe_barrier(PIPE_V);
    vmul(q_mul_31 + 0, swiglu_t_29 + 0, smooth_tile_f32_9 + 0, 4, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vabs(q_abs_36 + 0, q_mul_31 + 0, 4, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0ULL, 255ULL);
    vmax(ub_37 + 0, q_abs_36 + 0, q_abs_36 + 8 + 0, 1, 1, 32, 32, 0, 0, 0);
    set_mask_norm();
    set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
    for (int rep = 2; rep < 32; ++rep) {
      pipe_barrier(PIPE_V);
      set_mask_norm();
      set_vector_mask(0ULL, 255ULL);
      vmax(ub_37 + 0, ub_37 + 0, q_abs_36 + rep * 8 + 0, 1, 1, 1, 32, 0, 0, 0);
      set_mask_norm();
      set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
    }
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0ULL, 255ULL);
    vcgmax(q_max_38 + 0, ub_37 + 0, 1, 1, 1, 8);
    set_mask_norm();
    set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
    pipe_barrier(PIPE_V);
    set_mask_norm();
    set_vector_mask(0ULL, 1ULL);
    vmuls(q_max_mul_40 + 0, q_max_38 + 0, (float)0.0078740157, 1, 1, 1, 8, 8);
    set_mask_norm();
    set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
    pipe_barrier(PIPE_V);
    for (int bs = 0; bs < 1; ++bs) {
      for (int rep = 0; rep < 256; ++rep) {
        set_mask_norm();
        set_vector_mask(0ULL, 1ULL);
        vadds(ub_41 + bs * 256 + rep * 1 + 0, q_max_mul_40 + bs * 1 + 0, (float)(0), 1, 1, 1, 8, 8);
        set_mask_norm();
        set_vector_mask(18446744073709551615ULL, 18446744073709551615ULL);
      }
    }
    pipe_barrier(PIPE_V);
    for (int bs = 0; bs < 1; ++bs) {
      v4dtrans((__ubuf__ uint32_t*)q_max_brcb_42 + bs * 256, (__ubuf__ uint32_t*)ub_41 + bs * 256, 256, 1, true);
    }
    pipe_barrier(PIPE_V);
    vdiv(q_brcb_mul_45 + 0, q_mul_31 + 0, q_max_brcb_42 + 0, 4, 1, 1, 1, 8, 8, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16(q_tile_half_46 + 0, q_brcb_mul_45 + 0, 4, 1, 1, 4, 8);
    pipe_barrier(PIPE_V);
    vconv_f162s8(q_tile_int8_47 + 0, q_tile_half_46 + 0, 2, 1, 1, 4, 8);
    set_flag(PIPE_V, PIPE_MTE3, (event_t)0);
    wait_flag(PIPE_V, PIPE_MTE3, (event_t)0);
    copy_ubuf_to_gm((__gm__ int8_t *)gm_output + (scalar_11) * 256, q_tile_int8_47, 0, 1, 8, 0, 4088);
    copy_ubuf_to_gm((__gm__ float *)gm_scale + scalar_11, q_max_mul_40, 0, 1, 1, 0, 63);
    pipe_barrier(PIPE_ALL);
  }
}
extern "C" __global__ __aicore__ void swiglu_quant_op_custom(GM_ADDR gm_x, GM_ADDR gm_smooth_scales, GM_ADDR gm_output, GM_ADDR gm_scale, GM_ADDR gm_swiglu_out, GM_ADDR workspace, GM_ADDR tiling) {
  swiglu_quant_op_impl_npu_0_kernel(gm_x, gm_smooth_scales, gm_output, gm_scale, gm_swiglu_out);
}

#ifndef __CCE_KT_TEST__
void swiglu_quant_op_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t * gm_x, uint8_t * gm_smooth_scales, uint8_t * gm_output, uint8_t * gm_scale, uint8_t * gm_swiglu_out, uint8_t* workspace, uint8_t* tiling) {
  swiglu_quant_op_custom<<<8, l2ctrl, stream>>>(gm_x, gm_smooth_scales, gm_output, gm_scale, gm_swiglu_out, workspace, tiling);
}
#endif
