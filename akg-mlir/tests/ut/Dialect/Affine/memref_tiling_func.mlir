// RUN: akg-opt %s --memref-tiling-func | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449_00_get_tiling_struct_size_function() -> i64 attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<get_tiling_struct_size_function>} {
// CHECK-NEXT:    %c0_i64 = arith.constant 0 : i64
// CHECK-NEXT:    return %c0_i64 : i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449_single_outlined_0_0_tiling_function(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> (i64 {hacc.arg_type = #hacc.arg_type<tiling_key>}, i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}, i64 {hacc.arg_type = #hacc.arg_type<tiling_data>}) attributes {hacc.function_kind = #hacc.function_kind<HOST>, hacc.host_func_type = #hacc.host_func_type<tiling_function>} {
// CHECK-NEXT:    %c5_i64 = arith.constant 5 : i64
// CHECK-NEXT:    %c12280_i64 = arith.constant 12280 : i64
// CHECK-NEXT:    %c13_i64 = arith.constant 13 : i64
// CHECK-NEXT:    %c1_i64 = arith.constant 1 : i64
// CHECK-NEXT:    %c49120_i64 = arith.constant 49120 : i64
// CHECK-NEXT:    return %c5_i64, %c12280_i64, %c13_i64, %c1_i64, %c49120_i64 : i64, i64, i64, i64, i64
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449_00(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1x1666x1024xf32> attributes {enable_auto_mark_buffer_size, hacc.block_dim = 40 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hacc.tiling_function = #hacc.tiling_function<@Fused_Sub_Add_fusion_97330263758862449_single_outlined_0_0_tiling_function>, hfusion.fusion_kind = "PURE_ELEMWISE"} {
// CHECK-NEXT:    affine.for %arg4 = 0 to 1536 step 512 {
// CHECK-NEXT:      affine.for %arg5 = 0 to 1024 step 512 {
// CHECK-NEXT:        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
// CHECK-NEXT:          affine.for %arg7 = #map(%arg5) to #map1(%arg5) {
// CHECK-NEXT:            %0 = affine.load %arg0[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %1 = affine.load %arg1[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:            %3 = affine.load %arg2[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:            affine.store %4, %arg3[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.for %arg4 = 0 to 1024 step 512 {
// CHECK-NEXT:      affine.for %arg5 = 1536 to 1666 {
// CHECK-NEXT:        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
// CHECK-NEXT:          %0 = affine.load %arg0[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %1 = affine.load %arg1[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:          %3 = affine.load %arg2[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:          affine.store %4, %arg3[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:   }
// CHECK-NEXT:    return %arg3 : memref<1x1666x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:    %0 = call @Fused_Sub_Add_fusion_97330263758862449_00(%arg0, %arg1, %arg2, %arg3) : (memref<1x1666x1024xf32>, memref<1x1666x1024xf32>, memref<1x1666x1024xf32>, memref<1x1666x1024xf32>) -> memref<1x1666x1024xf32>
// CHECK-NEXT:    return %0 : memref<1x1666x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:}

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 512)>
module {
  func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    affine.for %arg4 = 0 to 1536 step 512 {
      affine.for %arg5 = 0 to 1024 step 512 {
        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
          affine.for %arg7 = #map(%arg5) to #map1(%arg5) {
            %0 = affine.load %arg0[0, %arg6, %arg7] : memref<1x1666x1024xf32>
            %1 = affine.load %arg1[0, %arg6, %arg7] : memref<1x1666x1024xf32>
            %2 = arith.subf %0, %1 : f32
            %3 = affine.load %arg2[0, %arg6, %arg7] : memref<1x1666x1024xf32>
            %4 = arith.addf %2, %3 : f32
            affine.store %4, %arg3[0, %arg6, %arg7] : memref<1x1666x1024xf32>
          }
        }
      }
    }
    affine.for %arg4 = 0 to 1024 step 512 {
      affine.for %arg5 = 1536 to 1666 {
        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
          %0 = affine.load %arg0[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %1 = affine.load %arg1[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %2 = arith.subf %0, %1 : f32
          %3 = affine.load %arg2[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %arg3[0, %arg5, %arg6] : memref<1x1666x1024xf32>
        }
      }
    }
    return %arg3 : memref<1x1666x1024xf32>
  }
}
