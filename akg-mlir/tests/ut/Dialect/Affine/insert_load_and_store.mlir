// RUN: akg-opt %s --insert-load-and-store | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 512)>
#map2 = affine_map<(d0, d1) -> (0)>
#map3 = affine_map<(d0, d1) -> (d0)>
#map4 = affine_map<(d0, d1) -> (d1)>
func.func @Fused_Sub_Add_fusion(%arg0: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 2 : i64}, %arg3: tensor<1x1666x1024xf32> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> tensor<1x1666x1024xf32> {
  %0 = tensor.empty() : tensor<1x1666x1024xf32>
  %1 = affine.for %arg4 = 0 to 1536 step 512 iter_args(%arg5 = %arg3) -> (tensor<1x1666x1024xf32>) {
    %3 = affine.for %arg6 = 0 to 1024 step 512 iter_args(%arg7 = %arg5) -> (tensor<1x1666x1024xf32>) {
      %4 = affine.for %arg8 = #map(%arg4) to #map1(%arg4) iter_args(%arg9 = %arg7) -> (tensor<1x1666x1024xf32>) {
        %5 = affine.for %arg10 = #map(%arg6) to #map1(%arg6) step 512 iter_args(%arg11 = %arg9) -> (tensor<1x1666x1024xf32>) {
          %6 = affine.apply #map2(%arg8, %arg10)
          %7 = affine.apply #map3(%arg8, %arg10)
          %8 = affine.apply #map4(%arg8, %arg10)
          %extracted_slice = tensor.extract_slice %arg0[%6, %7, %8] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
          %9 = affine.apply #map2(%arg8, %arg10)
          %10 = affine.apply #map3(%arg8, %arg10)
          %11 = affine.apply #map4(%arg8, %arg10)
          %extracted_slice_0 = tensor.extract_slice %arg1[%9, %10, %11] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
          %12 = arith.subf %extracted_slice, %extracted_slice_0 : tensor<512xf32>
          %13 = affine.apply #map2(%arg8, %arg10)
          %14 = affine.apply #map3(%arg8, %arg10)
          %15 = affine.apply #map4(%arg8, %arg10)
          %extracted_slice_1 = tensor.extract_slice %arg2[%13, %14, %15] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
          %16 = arith.addf %12, %extracted_slice_1 : tensor<512xf32>
          %17 = affine.apply #map2(%arg8, %arg10)
          %18 = affine.apply #map3(%arg8, %arg10)
          %19 = affine.apply #map4(%arg8, %arg10)
          %inserted_slice = tensor.insert_slice %16 into %arg11[%17, %18, %19] [1, 1, 512] [1, 1, 1] : tensor<512xf32> into tensor<1x1666x1024xf32>
          affine.yield %inserted_slice : tensor<1x1666x1024xf32>
        } {tensorize.liveout_indices = array<i64: 0>}
        affine.yield %5 : tensor<1x1666x1024xf32>
      } {tensorize.liveout_indices = array<i64: 0>}
      affine.yield %4 : tensor<1x1666x1024xf32>
    } {tensorize.liveout_indices = array<i64: 0>}
    affine.yield %3 : tensor<1x1666x1024xf32>
  } {tensorize.liveout_indices = array<i64: 0>}
  %2 = affine.for %arg4 = 0 to 1024 step 512 iter_args(%arg5 = %1) -> (tensor<1x1666x1024xf32>) {
    %3 = affine.for %arg6 = 1536 to 1666 iter_args(%arg7 = %arg5) -> (tensor<1x1666x1024xf32>) {
      %4 = affine.for %arg8 = #map(%arg4) to #map1(%arg4) step 512 iter_args(%arg9 = %arg7) -> (tensor<1x1666x1024xf32>) {
        %5 = affine.apply #map2(%arg6, %arg8)
        %6 = affine.apply #map3(%arg6, %arg8)
        %7 = affine.apply #map4(%arg6, %arg8)
        %extracted_slice = tensor.extract_slice %arg0[%5, %6, %7] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
        %8 = affine.apply #map2(%arg6, %arg8)
        %9 = affine.apply #map3(%arg6, %arg8)
        %10 = affine.apply #map4(%arg6, %arg8)
        %extracted_slice_0 = tensor.extract_slice %arg1[%8, %9, %10] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
        %11 = arith.subf %extracted_slice, %extracted_slice_0 : tensor<512xf32>
        %12 = affine.apply #map2(%arg6, %arg8)
        %13 = affine.apply #map3(%arg6, %arg8)
        %14 = affine.apply #map4(%arg6, %arg8)
        %extracted_slice_1 = tensor.extract_slice %arg2[%12, %13, %14] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
        %15 = arith.addf %11, %extracted_slice_1 : tensor<512xf32>
        %16 = affine.apply #map2(%arg6, %arg8)
        %17 = affine.apply #map3(%arg6, %arg8)
        %18 = affine.apply #map4(%arg6, %arg8)
        %inserted_slice = tensor.insert_slice %15 into %arg9[%16, %17, %18] [1, 1, 512] [1, 1, 1] : tensor<512xf32> into tensor<1x1666x1024xf32>
        affine.yield %inserted_slice : tensor<1x1666x1024xf32>
      } {tensorize.liveout_indices = array<i64: 0>}
      affine.yield %4 : tensor<1x1666x1024xf32>
    } {tensorize.liveout_indices = array<i64: 0>}
    affine.yield %3 : tensor<1x1666x1024xf32>
  } {tensorize.liveout_indices = array<i64: 0>}
  return %2 : tensor<1x1666x1024xf32>
}

// CHECK-LABEL:   #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:    #map1 = affine_map<(d0) -> (d0 + 512)>
// CHECK-NEXT:    #map2 = affine_map<(d0, d1) -> (0)>
// CHECK-NEXT:    #map3 = affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT:    #map4 = affine_map<(d0, d1) -> (d1)>
// CHECK-NEXT:    module {
// CHECK-NEXT:      func.func @Fused_Sub_Add_fusion(%arg0: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 0 : i64}, %arg1: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 1 : i64}, %arg2: tensor<1x1666x1024xf32> {hacc.arg_type = "input", hacc.input_idx = 2 : i64}, %arg3: tensor<1x1666x1024xf32> {hacc.arg_type = "output", hacc.output_idx = 0 : i64}) -> tensor<1x1666x1024xf32> {
// CHECK-NEXT:        %0 = tensor.empty() : tensor<1x1666x1024xf32>
// CHECK-NEXT:        %1 = affine.for %arg4 = 0 to 1536 step 512 iter_args(%arg5 = %arg3) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:          %3 = affine.for %arg6 = 0 to 1024 step 512 iter_args(%arg7 = %arg5) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:            %4 = affine.for %arg8 = #map(%arg4) to #map1(%arg4) iter_args(%arg9 = %arg7) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:              %5 = affine.for %arg10 = #map(%arg6) to #map1(%arg6) step 512 iter_args(%arg11 = %arg9) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:                %6 = affine.apply #map2(%arg8, %arg10)
// CHECK-NEXT:                %7 = affine.apply #map3(%arg8, %arg10)
// CHECK-NEXT:                %8 = affine.apply #map4(%arg8, %arg10)
// CHECK-NEXT:                %extracted_slice = tensor.extract_slice %arg0[%6, %7, %8] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:                %9 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:                %10 = hfusion.load ins(%extracted_slice : tensor<512xf32>) outs(%9 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:                annotation.mark %10 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:                %11 = affine.apply #map2(%arg8, %arg10)
// CHECK-NEXT:                %12 = affine.apply #map3(%arg8, %arg10)
// CHECK-NEXT:                %13 = affine.apply #map4(%arg8, %arg10)
// CHECK-NEXT:                %extracted_slice_0 = tensor.extract_slice %arg1[%11, %12, %13] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:                %14 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:                %15 = hfusion.load ins(%extracted_slice_0 : tensor<512xf32>) outs(%14 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:                annotation.mark %15 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:                %16 = arith.subf %10, %15 : tensor<512xf32>
// CHECK-NEXT:                %17 = affine.apply #map2(%arg8, %arg10)
// CHECK-NEXT:                %18 = affine.apply #map3(%arg8, %arg10)
// CHECK-NEXT:                %19 = affine.apply #map4(%arg8, %arg10)
// CHECK-NEXT:                %extracted_slice_1 = tensor.extract_slice %arg2[%17, %18, %19] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:                %20 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:                %21 = hfusion.load ins(%extracted_slice_1 : tensor<512xf32>) outs(%20 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:                annotation.mark %21 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:                %22 = arith.addf %16, %21 : tensor<512xf32>
// CHECK-NEXT:                %23 = affine.apply #map2(%arg8, %arg10)
// CHECK-NEXT:                %24 = affine.apply #map3(%arg8, %arg10)
// CHECK-NEXT:                %25 = affine.apply #map4(%arg8, %arg10)
// CHECK-NEXT:                annotation.mark %22 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:                %extracted_slice_2 = tensor.extract_slice %arg11[%23, %24, %25] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:                %26 = hfusion.store ins(%22 : tensor<512xf32>) outs(%extracted_slice_2 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:                %inserted_slice = tensor.insert_slice %26 into %arg11[%23, %24, %25] [1, 1, 512] [1, 1, 1] : tensor<512xf32> into tensor<1x1666x1024xf32>
// CHECK-NEXT:                affine.yield %inserted_slice : tensor<1x1666x1024xf32>
// CHECK-NEXT:              } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:              affine.yield %5 : tensor<1x1666x1024xf32>
// CHECK-NEXT:            } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:            affine.yield %4 : tensor<1x1666x1024xf32>
// CHECK-NEXT:          } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:          affine.yield %3 : tensor<1x1666x1024xf32>
// CHECK-NEXT:        } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:        %2 = affine.for %arg4 = 0 to 1024 step 512 iter_args(%arg5 = %1) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:          %3 = affine.for %arg6 = 1536 to 1666 iter_args(%arg7 = %arg5) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:            %4 = affine.for %arg8 = #map(%arg4) to #map1(%arg4) step 512 iter_args(%arg9 = %arg7) -> (tensor<1x1666x1024xf32>) {
// CHECK-NEXT:              %5 = affine.apply #map2(%arg6, %arg8)
// CHECK-NEXT:              %6 = affine.apply #map3(%arg6, %arg8)
// CHECK-NEXT:              %7 = affine.apply #map4(%arg6, %arg8)
// CHECK-NEXT:              %extracted_slice = tensor.extract_slice %arg0[%5, %6, %7] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:              %8 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:              %9 = hfusion.load ins(%extracted_slice : tensor<512xf32>) outs(%8 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:              annotation.mark %9 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:              %10 = affine.apply #map2(%arg6, %arg8)
// CHECK-NEXT:              %11 = affine.apply #map3(%arg6, %arg8)
// CHECK-NEXT:              %12 = affine.apply #map4(%arg6, %arg8)
// CHECK-NEXT:              %extracted_slice_0 = tensor.extract_slice %arg1[%10, %11, %12] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:              %13 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:              %14 = hfusion.load ins(%extracted_slice_0 : tensor<512xf32>) outs(%13 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:              annotation.mark %14 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:              %15 = arith.subf %9, %14 : tensor<512xf32>
// CHECK-NEXT:              %16 = affine.apply #map2(%arg6, %arg8)
// CHECK-NEXT:              %17 = affine.apply #map3(%arg6, %arg8)
// CHECK-NEXT:              %18 = affine.apply #map4(%arg6, %arg8)
// CHECK-NEXT:              %extracted_slice_1 = tensor.extract_slice %arg2[%16, %17, %18] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:              %19 = tensor.empty() : tensor<512xf32>
// CHECK-NEXT:              %20 = hfusion.load ins(%extracted_slice_1 : tensor<512xf32>) outs(%19 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:              annotation.mark %20 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:              %21 = arith.addf %15, %20 : tensor<512xf32>
// CHECK-NEXT:              %22 = affine.apply #map2(%arg6, %arg8)
// CHECK-NEXT:              %23 = affine.apply #map3(%arg6, %arg8)
// CHECK-NEXT:              %24 = affine.apply #map4(%arg6, %arg8)
// CHECK-NEXT:              annotation.mark %21 {buffer_size_in_byte = 16384 : i64} : tensor<512xf32>
// CHECK-NEXT:              %extracted_slice_2 = tensor.extract_slice %arg9[%22, %23, %24] [1, 1, 512] [1, 1, 1] : tensor<1x1666x1024xf32> to tensor<512xf32>
// CHECK-NEXT:              %25 = hfusion.store ins(%21 : tensor<512xf32>) outs(%extracted_slice_2 : tensor<512xf32>) -> tensor<512xf32>
// CHECK-NEXT:              %inserted_slice = tensor.insert_slice %25 into %arg9[%22, %23, %24] [1, 1, 512] [1, 1, 1] : tensor<512xf32> into tensor<1x1666x1024xf32>
// CHECK-NEXT:              affine.yield %inserted_slice : tensor<1x1666x1024xf32>
// CHECK-NEXT:            } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:            affine.yield %4 : tensor<1x1666x1024xf32>
// CHECK-NEXT:          } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:          affine.yield %3 : tensor<1x1666x1024xf32>
// CHECK-NEXT:        } {tensorize.liveout_indices = array<i64: 0>}
// CHECK-NEXT:        return %2 : tensor<1x1666x1024xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
