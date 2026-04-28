# Triton Ascend 算子测试报告

**测试时间**: 2026-04-01

**测试环境**:
- Python 3.10.0
- torch_npu 2.7.1
- triton_ascend 3.2.0
- Backend: Ascend
- Arch: Ascend910B4

**总计**: 54 个算子 | 验证通过: 54 | 验证失败: 0 | 完成性能测试: 54

## 性能数据

| # | 算子 | Verify | Base (us) | Generation (us) | 加速比 |
| --- | --- | ------ | --------- | --------------- | ----------- |
| 1 | 1_Square_matrix_multiplication_ | PASS | 302.47 | 321.66 | 0.9403x |
| 2 | 2_Standard_matrix_multiplication_ | PASS | 380.50 | 386.03 | 0.9857x |
| 3 | 3_Batched_matrix_multiplication | PASS | 306.37 | 305.55 | 1.0027x |
| 4 | 4_Matrix_vector_multiplication_ | PASS | 1248.75 | 906.36 | 1.3778x |
| 5 | 5_Matrix_scalar_multiplication | PASS | 922.15 | 898.58 | 1.0262x |
| 6 | 6_Matmul_with_large_K_dimension_ | PASS | 1443.91 | 3052.00 | 0.4731x |
| 7 | 7_Matmul_with_small_K_dimension_ | PASS | 1533.08 | 1489.03 | 1.0296x |
| 8 | 8_Matmul_with_irregular_shapes_ | PASS | 8401.00 | 16763.43 | 0.5012x |
| 9 | 9_Tall_skinny_matrix_multiplication_ | PASS | 1512.68 | 1456.90 | 1.0383x |
| 10 | 10_3D_tensor_matrix_multiplication | PASS | 927.06 | 853.28 | 1.0865x |
| 11 | 12_Matmul_with_diagonal_matrices_ | PASS | 2390.04 | 293.71 | 8.1373x |
| 12 | 13_Matmul_for_symmetric_matrices | PASS | 2278.17 | 2268.36 | 1.0043x |
| 13 | 14_Matmul_for_upper_triangular_matrices | PASS | 2376.45 | 1542.08 | 1.5411x |
| 14 | 15_Matmul_for_lower_triangular_matrices | PASS | 2371.98 | 1554.44 | 1.5259x |
| 15 | 16_Matmul_with_transposed_A | PASS | 401.75 | 393.25 | 1.0216x |
| 16 | 17_Matmul_with_transposed_B | PASS | 411.19 | 389.61 | 1.0554x |
| 17 | 18_Matmul_with_transposed_both | PASS | 377.04 | 737.88 | 0.5110x |
| 18 | 19_ReLU | PASS | 6.54 | 7.25 | 0.9031x |
| 19 | 20_LeakyReLU | PASS | 6.48 | 6.90 | 0.9396x |
| 20 | 21_Sigmoid | PASS | 6.60 | 6.94 | 0.9505x |
| 21 | 22_Tanh | PASS | 7.75 | 6.99 | 1.1083x |
| 22 | 23_Softmax | PASS | 10.42 | 8.84 | 1.1779x |
| 23 | 24_LogSoftmax | PASS | 10.36 | 8.61 | 1.2030x |
| 24 | 25_Swish | PASS | 10.31 | 7.04 | 1.4649x |
| 25 | 26_GELU_ | PASS | 6.98 | 7.58 | 0.9207x |
| 26 | 27_SELU_ | PASS | 6.92 | 7.09 | 0.9772x |
| 27 | 28_HardSigmoid | PASS | 6.64 | 7.50 | 0.8850x |
| 28 | 29_Softplus | PASS | 9.39 | 6.44 | 1.4579x |
| 29 | 30_Softsign | PASS | 15.88 | 6.71 | 2.3647x |
| 30 | 31_ELU | PASS | 7.08 | 6.90 | 1.0262x |
| 31 | 32_HardTanh | PASS | 9.60 | 7.30 | 1.3162x |
| 32 | 34_InstanceNorm | PASS | 1323.41 | 1470.45 | 0.9000x |
| 33 | 35_GroupNorm_ | PASS | 923.01 | 1093.42 | 0.8442x |
| 34 | 36_RMSNorm_ | PASS | 2218.04 | 968.30 | 2.2907x |
| 35 | 38_L1Norm_ | PASS | 20.77 | 6.72 | 3.0885x |
| 36 | 39_L2Norm_ | PASS | 11.85 | 6.65 | 1.7829x |
| 37 | 40_LayerNorm | PASS | 3647.09 | 2578.09 | 1.4146x |
| 38 | 44_Average_Pooling_1D | PASS | 40.52 | 73.62 | 0.5504x |
| 39 | 45_Average_Pooling_2D | PASS | 1540.04 | 40465.32 | 0.0381x |
| 40 | 47_Sum_reduction_over_a_dimension | PASS | 16.93 | 16.28 | 1.0398x |
| 41 | 49_Max_reduction_over_a_dimension | PASS | 104.46 | 18.06 | 5.7840x |
| 42 | 50_Product_reduction_over_a_dimension | PASS | 16.62 | 17.63 | 0.9424x |
| 43 | 52_Argmin_over_a_dimension | PASS | 409.40 | 82.99 | 4.9332x |
| 44 | 53_Min_reduction_over_a_dimension | PASS | 104.92 | 15.47 | 6.7811x |
| 45 | 88_MinGPTNewGelu | PASS | 234.88 | 81.33 | 2.8879x |
| 46 | 89_cumsum | PASS | 225.28 | 264.90 | 0.8504x |
| 47 | 91_cumsum_reverse | PASS | 1384.29 | 264.86 | 5.2264x |
| 48 | 93_masked_cumsum | PASS | 247.76 | 266.57 | 0.9295x |
| 49 | 94_MSELoss | PASS | 36.06 | 21.21 | 1.6996x |
| 50 | 96_HuberLoss | PASS | 20.77 | 18.97 | 1.0949x |
| 51 | 97_CosineSimilarityLoss | PASS | 86.53 | 29.80 | 2.9037x |
| 52 | 98_KLDivLoss | PASS | 138.85 | 18.39 | 7.5487x |
| 53 | 99_TripletMarginLoss | PASS | 71.47 | 28.32 | 2.5241x |
| 54 | 100_HingeLoss | PASS | 6.45 | 4.17 | 1.5476x |

## 统计摘要

| 指标 | 值 |
| -------- | ----------- |
| 加速比几何平均 | 1.3155x |
| fast1 (>1x) | 36/54 (66.7%) |
| fast0.8 (>0.8x) | 49/54 (90.7%) |
