// RUN: akg-opt %s --remove-redundant-loops | FileCheck %s

// CHECK: #set = affine_set<(d0) : (d0 == 0)>
// CHECK-LABEL: func.func @Fused_ReduceSum_split_2679919397770605199(%arg0: memref<4954x4953x3xf32>, %arg1: memref<4954x3xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   affine.for %arg2 = 0 to 4954 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 3 {
// CHECK-NEXT:       affine.for %arg4 = 0 to 4953 {
// CHECK-NEXT:         %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-NEXT:         affine.if #set(%arg4) {
// CHECK-NEXT:           vector.transfer_write %cst_0, %arg1[%arg2, %arg3] : vector<4xf32>, memref<4954x3xf32>
// CHECK-NEXT:         }
// CHECK-NEXT:         %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %0 = vector.transfer_read %arg0[%arg2, %arg4, %arg3], %cst_1 : memref<4954x4953x3xf32>, vector<4xf32>
// CHECK-NEXT:         %cst_2 = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:         %1 = vector.transfer_read %arg1[%arg2, %arg3], %cst_2 : memref<4954x3xf32>, vector<4xf32>
// CHECK-NEXT:         %2 = arith.addf %0, %1 {reduction_axes = [2 : index], reduction_type = "y"} : vector<4xf32>
// CHECK-NEXT:         vector.transfer_write %2, %arg1[%arg2, %arg3] : vector<4xf32>, memref<4954x3xf32>
// CHECK-NEXT:       } {reduceLoop}
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#set = affine_set<(d0) : (d0 == 0)>
func.func @Fused_ReduceSum_split_2679919397770605199(%arg0: memref<4954x4953x3xf32>, %arg1: memref<4954x3xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg2 = 0 to 4954 {
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 4953 {
        affine.for %arg5 = #map(%arg2) to #map1(%arg2) {
          affine.for %arg6 = #map(%arg3) to #map1(%arg3) step 4 {
            %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
            affine.for %arg7 = #map(%arg4) to #map1(%arg4) {
              affine.if #set(%arg7) {
                vector.transfer_write %cst_0, %arg1[%arg5, %arg6] : vector<4xf32>, memref<4954x3xf32>
              }
              %cst_1 = arith.constant 0.000000e+00 : f32
              %0 = vector.transfer_read %arg0[%arg5, %arg7, %arg6], %cst_1 : memref<4954x4953x3xf32>, vector<4xf32>
              %cst_2 = arith.constant 0.000000e+00 : f32
              %1 = vector.transfer_read %arg1[%arg5, %arg6], %cst_2 : memref<4954x3xf32>, vector<4xf32>
              %2 = arith.addf %0, %1 {reduction_axes = [2 : index], reduction_type = "y"} : vector<4xf32>
              vector.transfer_write %2, %arg1[%arg5, %arg6] : vector<4xf32>, memref<4954x3xf32>
            }
          }
        }
      } {reduceLoop}
    }
  }
  return
}
