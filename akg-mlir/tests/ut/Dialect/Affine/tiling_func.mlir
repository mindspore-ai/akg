// RUN: akg-opt %s --tiling-func | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT: func.func @Fused_BiasAdd_10033593016906428850_single_outlined_0_0_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = "HOST"} {
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: return %c0_i64 : i64
// CHECK-NEXT: }
// CHECK-NEXT: func.func @Fused_BiasAdd_10033593016906428850_single_outlined_0_0_tiling_function(%arg0: tensor<28x3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: tensor<3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: tensor<28x3072xbf16> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> (i64 {hacc.arg_type = "tiling_key"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}, i64 {hacc.arg_type = "tiling_data"}) attributes {hacc.function_kind = "HOST", hacc.host_func_type = "tiling_function"} {
// CHECK-NEXT: %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT: %c12280_i64 = arith.constant 12280 : i64
// CHECK-NEXT: %c13_i64 = arith.constant 13 : i64
// CHECK-NEXT: %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT: %c49120_i64 = arith.constant 49120 : i64
// CHECK-NEXT: return %c0_i64, %c12280_i64, %c13_i64, %c1_i64, %c49120_i64 : i64, i64, i64, i64, i64
// CHECK-NEXT: }
// CHECK-NEXT: func.func @Fused_BiasAdd_10033593016906428850_single_outlined_0_0_0(%arg0: tensor<28x3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: tensor<3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: tensor<28x3072xbf16> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> tensor<28x3072xbf16> attributes {enable_auto_mark_buffer_size, hacc.block_dim = 40 : i64, hacc.entry, hacc.function_kind = "DEVICE", hacc.tiling_function = @Fused_BiasAdd_10033593016906428850_single_outlined_0_0_tiling_function, hfusion.fusion_kind = "PURE_ELEMWISE"} {
// CHECK-NEXT: %0 = tensor.empty() : tensor<28x3072xbf16>
// CHECK-NEXT: %1 = tensor.empty() : tensor<28x3072xbf16>
// CHECK-NEXT: %2:2 = affine.for %arg3 = 0 to 28 iter_args(%arg4 = %0, %arg5 = %arg2) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
// CHECK-NEXT: %3:2 = affine.for %arg6 = 0 to 3072 step 3072 iter_args(%arg7 = %arg4, %arg8 = %arg5) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
// CHECK-NEXT: %extracted_slice = tensor.extract_slice %arg1[0] [3072] [1] : tensor<3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT: %inserted_slice = tensor.insert_slice %extracted_slice into %arg7[%arg3, 0] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
// CHECK-NEXT: %extracted_slice_0 = tensor.extract_slice %arg0[%arg3, 0] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT: %extracted_slice_1 = tensor.extract_slice %0[%arg3, 0] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
// CHECK-NEXT: %4 = arith.addf %extracted_slice_0, %extracted_slice_1 : tensor<3072xbf16>
// CHECK-NEXT: %inserted_slice_2 = tensor.insert_slice %4 into %arg8[%arg3, 0] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
// CHECK-NEXT: affine.yield %inserted_slice, %inserted_slice_2 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
// CHECK-NEXT: }
// CHECK-NEXT: affine.yield %3#0, %3#1 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
// CHECK-NEXT: }
// CHECK-NEXT: return %2#1 : tensor<28x3072xbf16>
// CHECK-NEXT: }
// CHECK-NEXT: func.func @Fused_BiasAdd_10033593016906428850(%arg0: tensor<28x3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: tensor<3072xbf16> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: tensor<28x3072xbf16> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> tensor<28x3072xbf16> {
// CHECK-NEXT: %0 = call @Fused_BiasAdd_10033593016906428850_single_outlined_0_0_0(%arg0, %arg1, %arg2) : (tensor<28x3072xbf16>, tensor<3072xbf16>, tensor<28x3072xbf16>) -> tensor<28x3072xbf16>
// CHECK-NEXT: return %0 : tensor<28x3072xbf16>
// CHECK-NEXT: }
// CHECK-NEXT:}

module {
  func.func @Fused_BiasAdd_10033593016906428850(%arg0: tensor<28x3072xbf16>, %arg1: tensor<3072xbf16>) -> tensor<28x3072xbf16> {
    %0 = tensor.empty() : tensor<28x3072xbf16>
    %1 = tensor.empty() : tensor<28x3072xbf16>
    %2:2 = affine.for %arg2 = 0 to 28 iter_args(%arg3 = %0, %arg4 = %1) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
      %3:2 = affine.for %arg5 = 0 to 3072 step 3072 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (tensor<28x3072xbf16>, tensor<28x3072xbf16>) {
        %extracted_slice = tensor.extract_slice %arg1[0] [3072] [1] : tensor<3072xbf16> to tensor<3072xbf16>
          %inserted_slice = tensor.insert_slice %extracted_slice into %arg6[%arg2, 0] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
          %extracted_slice_0 = tensor.extract_slice %arg0[%arg2, 0] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
          %extracted_slice_1 = tensor.extract_slice %0[%arg2, 0] [1, 3072] [1, 1] : tensor<28x3072xbf16> to tensor<3072xbf16>
          %4 = arith.addf %extracted_slice_0, %extracted_slice_1 : tensor<3072xbf16>
          %inserted_slice_2 = tensor.insert_slice %4 into %arg7[%arg2, 0] [1, 3072] [1, 1] : tensor<3072xbf16> into tensor<28x3072xbf16>
          affine.yield %inserted_slice, %inserted_slice_2 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
      }
      affine.yield %3#0, %3#1 : tensor<28x3072xbf16>, tensor<28x3072xbf16>
    }
    return %2#1 : tensor<28x3072xbf16>
  }
}

