// RUN: akg-opt %s --add-out-parameter | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
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
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg3 : memref<1x1666x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:}


func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32>, %arg1: memref<1x1666x1024xf32>, %arg2: memref<1x1666x1024xf32>) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1666x1024xf32>
  affine.for %arg3 = 0 to 1536 step 512 {
    affine.for %arg4 = 0 to 1024 step 512 {
      affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 512)>(%arg3) {
        affine.for %arg6 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 512)>(%arg4) {
          %0 = affine.load %arg0[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %1 = affine.load %arg1[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %2 = arith.subf %0, %1 : f32
          %3 = affine.load %arg2[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %alloc[0, %arg5, %arg6] : memref<1x1666x1024xf32>
        }
      }
    }
  }
  affine.for %arg3 = 0 to 1024 step 512 {
    affine.for %arg4 = 1536 to 1666 {
      affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 512)>(%arg3) {
        %0 = affine.load %arg0[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %1 = affine.load %arg1[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %2 = arith.subf %0, %1 : f32
        %3 = affine.load %arg2[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %alloc[0, %arg4, %arg5] : memref<1x1666x1024xf32>
      }
    }
  }
  return %alloc : memref<1x1666x1024xf32>
}
