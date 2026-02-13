# KernelBench Level1 Kernel Generation Success Rate Report

## Overview

CUDA C, C++ and TileLang_CUDA DSL: AIKG operator generation success rate (pass@4) in KernelBench Level1 benchmark.

## Success Rate Statistics

- **CUDA C**: 59/100 (59%)
- **CPP**: 63/100 (63%)
- **TileLang_CUDA**: 44/100 (44%)

## Detailed Results


| No. | Kernel Name  | CUDA C | CPP | TileLang_CUDA |
|------|----------|--------|------|----------|
| 1 | Square_matrix_multiplication_ | ✓ | ✓ | ✓ |
| 2 | Standard_matrix_multiplication_ | ✓ | ✓ | ✓ |
| 3 | Batched_matrix_multiplication | ✓ | ✓ | ✓ |
| 4 | Matrix_vector_multiplication_ | ✓ | ✓ | ✗ |
| 5 | Matrix_scalar_multiplication | ✓ | ✓ | ✓ |
| 6 | Matmul_with_large_K_dimension_ | ✓ | ✓ | ✓ |
| 7 | Matmul_with_small_K_dimension_ | ✓ | ✓ | ✓ |
| 8 | Matmul_with_irregular_shapes_ | ✓ | ✓ | ✓ |
| 9 | Tall_skinny_matrix_multiplication_ | ✓ | ✓ | ✓ |
| 10 | 3D_tensor_matrix_multiplication | ✓ | ✓ | ✓ |
| 11 | 4D_tensor_matrix_multiplication | ✓ | ✓ | ✓ |
| 12 | Matmul_with_diagonal_matrices_ | ✗ | ✓ | ✓ |
| 13 | Matmul_for_symmetric_matrices | ✓ | ✓ | ✓ |
| 14 | Matmul_for_upper_triangular_matrices | ✓ | ✓ | ✓ |
| 15 | Matmul_for_lower_triangular_matrices | ✓ | ✓ | ✓ |
| 16 | Matmul_with_transposed_A | ✓ | ✓ | ✓ |
| 17 | Matmul_with_transposed_B | ✓ | ✓ | ✓ |
| 18 | Matmul_with_transposed_both | ✓ | ✗ | ✓ |
| 19 | ReLU | ✓ | ✓ | ✓ |
| 20 | LeakyReLU | ✓ | ✓ | ✓ |
| 21 | Sigmoid | ✓ | ✓ | ✓ |
| 22 | Tanh | ✓ | ✓ | ✓ |
| 23 | Softmax | ✓ | ✓ | ✗ |
| 24 | LogSoftmax | ✓ | ✓ | ✓ |
| 25 | Swish | ✓ | ✓ | ✓ |
| 26 | GELU_ | ✓ | ✓ | ✓ |
| 27 | SELU_ | ✓ | ✓ | ✓ |
| 28 | HardSigmoid | ✓ | ✓ | ✓ |
| 29 | Softplus | ✓ | ✓ | ✓ |
| 30 | Softsign | ✓ | ✓ | ✓ |
| 31 | ELU | ✓ | ✓ | ✓ |
| 32 | HardTanh | ✓ | ✓ | ✓ |
| 33 | BatchNorm | ✗ | ✗ | ✗ |
| 34 | InstanceNorm | ✓ | ✓ | ✗ |
| 35 | GroupNorm_ | ✗ | ✓ | ✗ |
| 36 | RMSNorm_ | ✗ | ✓ | ✓ |
| 37 | FrobeniusNorm_ | ✓ | ✗ | ✗ |
| 38 | L1Norm_ | ✓ | ✓ | ✓ |
| 39 | L2Norm_ | ✓ | ✓ | ✓ |
| 40 | LayerNorm | ✓ | ✗ | ✗ |
| 41 | Max_Pooling_1D | ✓ | ✓ | ✗ |
| 42 | Max_Pooling_2D | ✗ | ✓ | ✗ |
| 43 | Max_Pooling_3D | ✓ | ✓ | ✗ |
| 44 | Average_Pooling_1D | ✓ | ✓ | ✗ |
| 45 | Average_Pooling_2D | ✗ | ✓ | ✗ |
| 46 | Average_Pooling_3D | ✓ | ✗ | ✗ |
| 47 | Sum_reduction_over_a_dimension | ✓ | ✓ | ✓ |
| 48 | Mean_reduction_over_a_dimension | ✓ | ✓ | ✓ |
| 49 | Max_reduction_over_a_dimension | ✓ | ✓ | ✓ |
| 50 | Product_reduction_over_a_dimension | ✓ | ✓ | ✗ |
| 51 | Argmax_over_a_dimension | ✓ | ✓ | ✓ |
| 52 | Argmin_over_a_dimension | ✓ | ✓ | ✓ |
| 53 | Min_reduction_over_a_dimension | ✓ | ✓ | ✓ |
| 54 | conv_standard_3D__square_input__square_kernel | ✗ | ✗ | ✗ |
| 55 | conv_standard_2D__asymmetric_input__square_kernel | ✗ | ✗ | ✗ |
| 56 | conv_standard_2D__asymmetric_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 57 | conv_transposed_2D__square_input__square_kernel | ✗ | ✓ | ✗ |
| 58 | conv_transposed_3D__asymmetric_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 59 | conv_standard_3D__asymmetric_input__square_kernel | ✗ | ✓ | ✗ |
| 60 | conv_standard_3D__square_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 61 | conv_transposed_3D__square_input__square_kernel | ✗ | ✗ | ✗ |
| 62 | conv_standard_2D__square_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 63 | conv_standard_2D__square_input__square_kernel | ✗ | ✗ | ✗ |
| 64 | conv_transposed_1D | ✗ | ✗ | ✗ |
| 65 | conv_transposed_2D__square_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 66 | conv_standard_3D__asymmetric_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 67 | conv_standard_1D | ✗ | ✗ | ✗ |
| 68 | conv_transposed_3D__square_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 69 | conv_transposed_2D__asymmetric_input__asymmetric_kernel | ✗ | ✗ | ✗ |
| 70 | conv_transposed_3D__asymmetric_input__square_kernel | ✗ | ✗ | ✗ |
| 71 | conv_transposed_2D__asymmetric_input__square_kernel | ✗ | ✗ | ✗ |
| 72 | conv_transposed_3D_asymmetric_input_asymmetric_kernel___strided_padded_grouped_ | ✗ | ✗ | ✗ |
| 73 | conv_transposed_3D_asymmetric_input_square_kernel__strided_padded__grouped | ✗ | ✗ | ✗ |
| 74 | conv_transposed_1D_dilated | ✗ | ✗ | ✗ |
| 75 | conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__ | ✗ | ✗ | ✗ |
| 76 | conv_standard_1D_dilated_strided__ | ✗ | ✗ | ✗ |
| 77 | conv_transposed_3D_square_input_square_kernel___padded____dilated____strided__ | ✗ | ✗ | ✗ |
| 78 | conv_transposed_2D_asymmetric_input_asymmetric_kernel___padded__ | ✗ | ✗ | ✗ |
| 79 | conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__ | ✗ | ✗ | ✗ |
| 80 | conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__ | ✗ | ✓ | ✗ |
| 81 | conv_transposed_2D_asymmetric_input_square_kernel___dilated____padded____strided__ | ✗ | ✗ | ✗ |
| 82 | conv_depthwise_2D_square_input_square_kernel | ✗ | ✗ | ✗ |
| 83 | conv_depthwise_2D_square_input_asymmetric_kernel | ✗ | ✗ | ✗ |
| 84 | conv_depthwise_2D_asymmetric_input_square_kernel | ✗ | ✗ | ✗ |
| 85 | conv_depthwise_2D_asymmetric_input_asymmetric_kernel | ✗ | ✗ | ✗ |
| 86 | conv_depthwise_separable_2D | ✗ | ✗ | ✗ |
| 87 | conv_pointwise_2D | ✗ | ✗ | ✗ |
| 88 | MinGPTNewGelu | ✓ | ✓ | ✓ |
| 89 | cumsum | ✓ | ✓ | ✓ |
| 90 | cumprod | ✓ | ✓ | ✓ |
| 91 | cumsum_reverse | ✓ | ✓ | ✗ |
| 92 | cumsum_exclusive | ✗ | ✗ | ✗ |
| 93 | masked_cumsum | ✓ | ✓ | ✗ |
| 94 | MSELoss | ✓ | ✓ | ✓ |
| 95 | CrossEntropyLoss | ✓ | ✓ | ✗ |
| 96 | HuberLoss | ✓ | ✓ | ✓ |
| 97 | CosineSimilarityLoss | ✓ | ✓ | ✗ |
| 98 | KLDivLoss | ✓ | ✓ | ✗ |
| 99 | TripletMarginLoss | ✓ | ✓ | ✗ |
| 100 | HingeLoss | ✓ | ✓ | ✗ |

## Notes

- ✓ indicates successful generation
- ✗ indicates failed generation
- Update date: October 14, 2025
- Test method: pass@4 (at least 1 success in 4 attempts)
