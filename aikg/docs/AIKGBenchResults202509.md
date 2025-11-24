# AIKGBench Dynamic Shape Kernel Generation Success Rate Report

## Overview

AIKG's kernel generation success rate statistics on the AIKGBench Dynamic Shape benchmark (pass@4).

## Success Rate Statistics

- **Ascend**: 161/200 (80.5%)
- **CUDA**: Pending Testing

## Detailed Results

| No. | Kernel Name | Ascend | CUDA |
|-----|-------------|--------|------|
| 1 | attention_BNSD | ✗ |  |
| 2 | attention_BSH | ✗ |  |
| 3 | attention_flash_score | ✗ |  |
| 4 | attention_flash | ✗ |  |
| 5 | attention_flash_with_causal | ✗ |  |
| 6 | attention_flash_with_dropout | ✗ |  |
| 7 | attention_gqa | ✗ |  |
| 8 | attention_mqa | ✗ |  |
| 9 | attention_rope | ✗ |  |
| 10 | attention_scaled_dot_product_large | ✗ |  |
| 11 | attention_scaled_dot_product_small | ✗ |  |
| 12 | attention_scaled_dot_product | ✗ |  |
| 13 | attention_TH | ✗ |  |
| 14 | elemwise_abs | ✓ |  |
| 15 | elemwise_add_001 | ✓ |  |
| 16 | elemwise_add_002 | ✓ |  |
| 17 | elemwise_add_003 | ✓ |  |
| 18 | elemwise_addcdiv | ✓ |  |
| 19 | elemwise_addcmul | ✓ |  |
| 20 | elemwise_atan | ✓ |  |
| 21 | elemwise_atan2 | ✓ |  |
| 22 | elemwise_bitwise_and | ✓ |  |
| 23 | elemwise_bitwise_not | ✓ |  |
| 24 | elemwise_bitwise_or | ✓ |  |
| 25 | elemwise_bitwise_xor | ✓ |  |
| 26 | elemwise_broadcast | ✓ |  |
| 27 | elemwise_cast_full | ✓ |  |
| 28 | elemwise_cast | ✓ |  |
| 29 | elemwise_cdiv | ✓ |  |
| 30 | elemwise_ceil | ✓ |  |
| 31 | elemwise_clamp | ✓ |  |
| 32 | elemwise_cos | ✓ |  |
| 33 | elemwise_count_dim0 | ✓ |  |
| 34 | elemwise_count_dim1 | ✓ |  |
| 35 | elemwise_div_001 | ✓ |  |
| 36 | elemwise_equal | ✓ |  |
| 37 | elemwise_exp_001 | ✓ |  |
| 38 | elemwise_expand | ✓ |  |
| 39 | elemwise_floor_div | ✓ |  |
| 40 | elemwise_floor | ✓ |  |
| 41 | elemwise_gelu_001 | ✓ |  |
| 42 | elemwise_greater_equal | ✓ |  |
| 43 | elemwise_greater | ✓ |  |
| 44 | elemwise_hard_sigmoid_001 | ✓ |  |
| 45 | elemwise_hard_swish_001 | ✓ |  |
| 46 | elemwise_i2f | ✓ |  |
| 47 | elemwise_is_finite | ✓ |  |
| 48 | elemwise_is_inf | ✓ |  |
| 49 | elemwise_is_nan | ✓ |  |
| 50 | elemwise_leaky_relu_001 | ✓ |  |
| 51 | elemwise_less_equal | ✓ |  |
| 52 | elemwise_less | ✓ |  |
| 53 | elemwise_linspace | ✓ |  |
| 54 | elemwise_log | ✓ |  |
| 55 | elemwise_log2 | ✓ |  |
| 56 | elemwise_logical_and | ✓ |  |
| 57 | elemwise_logical_not | ✓ |  |
| 58 | elemwise_logical_or | ✓ |  |
| 59 | elemwise_maximum | ✓ |  |
| 60 | elemwise_minimum | ✓ |  |
| 61 | elemwise_mul_001 | ✓ |  |
| 62 | elemwise_ne | ✓ |  |
| 63 | elemwise_neg | ✓ |  |
| 64 | elemwise_pow_001 | ✓ |  |
| 65 | elemwise_pow_002 | ✓ |  |
| 66 | elemwise_precise_div | ✓ |  |
| 67 | elemwise_relu_001 | ✓ |  |
| 68 | elemwise_rope_001 | ✗ |  |
| 69 | elemwise_round | ✓ |  |
| 70 | elemwise_rsqrt | ✓ |  |
| 71 | elemwise_sigmoid_001 | ✓ |  |
| 72 | elemwise_sigmoid | ✓ |  |
| 73 | elemwise_silu_001 | ✓ |  |
| 74 | elemwise_silu | ✓ |  |
| 75 | elemwise_sin | ✓ |  |
| 76 | elemwise_softplus_001 | ✓ |  |
| 77 | elemwise_softsign_001 | ✓ |  |
| 78 | elemwise_sqrt | ✓ |  |
| 79 | elemwise_sub_001 | ✓ |  |
| 80 | elemwise_sub_002 | ✓ |  |
| 81 | elemwise_swish_001 | ✓ |  |
| 82 | elemwise_tan | ✓ |  |
| 83 | elemwise_tanh_001 | ✓ |  |
| 84 | fused_add_layer_norm | ✓ |  |
| 85 | fused_add_rms_norm | ✓ |  |
| 86 | fused_apply_rotary_pos_emb | ✗ |  |
| 87 | fused_ffn | ✗ |  |
| 88 | fused_gelu_and_mul | ✓ |  |
| 89 | fused_moe_gating_topk_softmax | ✗ |  |
| 90 | fused_silu_and_mul | ✓ |  |
| 91 | index_feeds_repeat | ✗ |  |
| 92 | index_gather | ✓ |  |
| 93 | index_masked_select | ✗ |  |
| 94 | index_scatter_add_with_sorted | ✓ |  |
| 95 | index_scatter_elements | ✗ |  |
| 96 | index_select | ✓ |  |
| 97 | index_topk | ✗ |  |
| 98 | norm_add_layer_quant | ✗ |  |
| 99 | norm_add_layer | ✓ |  |
| 100 | norm_add_rms_cast | ✓ |  |
| 101 | norm_add_rms_dynamic_quant | ✓ |  |
| 102 | norm_add_rms_quant | ✓ |  |
| 103 | norm_add_rms | ✓ |  |
| 104 | norm_batch | ✓ |  |
| 105 | norm_deep | ✓ |  |
| 106 | norm_dua_quantize_add_layer | ✓ |  |
| 107 | norm_group_silu | ✓ |  |
| 108 | norm_group_swish | ✓ |  |
| 109 | norm_group | ✓ |  |
| 110 | norm_inplace_add_layer | ✗ |  |
| 111 | norm_inplace_add_rms | ✗ |  |
| 112 | norm_instance | ✓ |  |
| 113 | norm_layer_001_large | ✓ |  |
| 114 | norm_layer_001_small | ✓ |  |
| 115 | norm_layer_001 | ✓ |  |
| 116 | norm_layer | ✓ |  |
| 117 | norm_quantize_add_layer | ✓ |  |
| 118 | norm_rms_001_large_std | ✓ |  |
| 119 | norm_rms_001_large | ✓ |  |
| 120 | norm_rms_001_small_std | ✓ |  |
| 121 | norm_rms_001_small | ✓ |  |
| 122 | norm_rms_001_std | ✓ |  |
| 123 | norm_rms_001 | ✓ |  |
| 124 | norm_rms | ✓ |  |
| 125 | reduction_amax | ✓ |  |
| 126 | reduction_amin | ✓ |  |
| 127 | reduction_argmax_001 | ✓ |  |
| 128 | reduction_argmax_002 | ✓ |  |
| 129 | reduction_argmax | ✓ |  |
| 130 | reduction_argmin_001 | ✓ |  |
| 131 | reduction_argmin_002 | ✓ |  |
| 132 | reduction_count_nonzero | ✓ |  |
| 133 | reduction_count_vector | ✓ |  |
| 134 | reduction_cumprod | ✓ |  |
| 135 | reduction_cumsum_001 | ✓ |  |
| 136 | reduction_cumsum_002 | ✓ |  |
| 137 | reduction_cumsum | ✓ |  |
| 138 | reduction_dot | ✓ |  |
| 139 | reduction_fast_softmax_001 | ✓ |  |
| 140 | reduction_fast_softmax_grad_001 | ✗ |  |
| 141 | reduction_max_001 | ✓ |  |
| 142 | reduction_max_002 | ✓ |  |
| 143 | reduction_max_003 | ✓ |  |
| 144 | reduction_max_large | ✓ |  |
| 145 | reduction_max_small | ✓ |  |
| 146 | reduction_max | ✓ |  |
| 147 | reduction_mean_001 | ✓ |  |
| 148 | reduction_mean_002 | ✓ |  |
| 149 | reduction_mean_003 | ✓ |  |
| 150 | reduction_mean_004 | ✓ |  |
| 151 | reduction_mean_005 | ✓ |  |
| 152 | reduction_mean_006 | ✓ |  |
| 153 | reduction_mean | ✓ |  |
| 154 | reduction_mean_vector | ✓ |  |
| 155 | reduction_min_001 | ✓ |  |
| 156 | reduction_min_002 | ✓ |  |
| 157 | reduction_min_003 | ✓ |  |
| 158 | reduction_min | ✓ |  |
| 159 | reduction_prod_001 | ✗ |  |
| 160 | reduction_prod_002 | ✓ |  |
| 161 | reduction_prod | ✓ |  |
| 162 | reduction_softmax_001 | ✓ |  |
| 163 | reduction_sum_001 | ✓ |  |
| 164 | reduction_sum_002 | ✓ |  |
| 165 | reduction_sum_005 | ✓ |  |
| 166 | reduction_sum_006 | ✓ |  |
| 167 | reduction_sum_dim1 | ✓ |  |
| 168 | reduction_sum | ✓ |  |
| 169 | reduction_top2_gating_argmax | ✗ |  |
| 170 | sorting_isin | ✗ |  |
| 171 | sorting_sort | ✗ |  |
| 172 | sorting_topk_001 | ✗ |  |
| 173 | sorting_topk_top_p_sampling_001 | ✗ |  |
| 174 | sorting_topk | ✗ |  |
| 175 | tensor_manipulation_cat_001 | ✓ |  |
| 176 | tensor_manipulation_cat_002 | ✓ |  |
| 177 | tensor_manipulation_cat_dim | ✗ |  |
| 178 | tensor_manipulation_cat | ✓ |  |
| 179 | tensor_manipulation_flip | ✓ |  |
| 180 | tensor_manipulation_gather | ✓ |  |
| 181 | tensor_manipulation_hstack | ✓ |  |
| 182 | tensor_manipulation_index_select | ✓ |  |
| 183 | tensor_manipulation_pad_001 | ✗ |  |
| 184 | tensor_manipulation_permute | ✗ |  |
| 185 | tensor_manipulation_reshape | ✓ |  |
| 186 | tensor_manipulation_scatter_add | ✗ |  |
| 187 | tensor_manipulation_scatter_multiply | ✗ |  |
| 188 | tensor_manipulation_scatter | ✗ |  |
| 189 | tensor_manipulation_split_001 | ✗ |  |
| 190 | tensor_manipulation_split | ✓ |  |
| 191 | tensor_manipulation_squeeze | ✓ |  |
| 192 | tensor_manipulation_stack_001 | ✓ |  |
| 193 | tensor_manipulation_stack_002 | ✓ |  |
| 194 | tensor_manipulation_stack | ✓ |  |
| 195 | tensor_manipulation_transpose_001 | ✓ |  |
| 196 | tensor_manipulation_transpose_2d | ✓ |  |
| 197 | tensor_manipulation_transpose | ✓ |  |
| 198 | tensor_manipulation_unpad_001 | ✓ |  |
| 199 | tensor_manipulation_unsqueeze_001 | ✓ |  |
| 200 | tensor_manipulation_unsqueeze | ✓ |  |

## Notes

- ✓ indicates successful generation
- ✗ indicates failed generation
- Last updated: September 26, 2025
- Test method: pass@4 (at least 1 success in 4 attempts)
