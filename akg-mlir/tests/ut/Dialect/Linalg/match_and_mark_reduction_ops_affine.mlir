// RUN: akg-opt %s -split-input-file --match-and-mark-reduction-ops="dialect=affine" | FileCheck %s

// CHECK-LABEL:  #map = affine_map<(d0) -> (d0)>
// CHECK-NEXT:   #map1 = affine_map<(d0) -> (d0 + 1)>
// CHECK-NEXT:   #map2 = affine_map<(d0) -> (d0 + 2)>
// CHECK-NEXT:   #set = affine_set<(d0) : (d0 == 0)>
// CHECK-NEXT:   #set1 = affine_set<(d0) : (-d0 + 1 == 0)>
// CHECK-NEXT:   module {
// CHECK-NEXT:     func.func @Fused_Sub_Exp_ReduceSum_Log_split_11941850547297653662(%arg0: memref<233008xf32>, %arg1: memref<233008x2xf32>, %arg2: memref<233008x2xf32>, %arg3: memref<233008xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, gpu_parallel_reduce = false, mindspore_kernel, process = "cuda", scop.ignored} {
// CHECK-NEXT:       %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:       %alloc = memref.alloc() {alignment = 64 : i64} : memref<233008xf32>
// CHECK-NEXT:       affine.for %arg4 = 0 to 233008 {
// CHECK-NEXT:         affine.for %arg5 = 0 to 2 step 2 {
// CHECK-NEXT:           affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
// CHECK-NEXT:             affine.for %arg7 = #map(%arg5) to #map2(%arg5) {
// CHECK-NEXT:               %0 = affine.load %arg1[%arg6, %arg7] : memref<233008x2xf32>
// CHECK-NEXT:               %1 = affine.load %arg0[%arg6] : memref<233008xf32>
// CHECK-NEXT:               %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:               affine.store %2, %arg2[%arg6, %arg7] : memref<233008x2xf32>
// CHECK-NEXT:               affine.if #set(%arg7) {
// CHECK-NEXT:                 affine.store %cst, %alloc[%arg6] : memref<233008xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:               %3 = affine.load %arg2[%arg6, %arg7] : memref<233008x2xf32>
// CHECK-NEXT:               %4 = math.exp %3 : f32
// CHECK-NEXT:               %5 = affine.load %alloc[%arg6] : memref<233008xf32>
// CHECK-NEXT:               %6 = arith.addf %4, %5 {enable_atomic_add = false, gpu_parallel_reduce = false, reduction_axes = [1 : index, 3 : index], reduction_type = "x"} : f32
// CHECK-NEXT:               affine.store %6, %alloc[%arg6] : memref<233008xf32>
// CHECK-NEXT:               affine.if #set1(%arg7) {
// CHECK-NEXT:                 %7 = affine.load %alloc[%arg6] : memref<233008xf32>
// CHECK-NEXT:                 %8 = math.log %7 : f32
// CHECK-NEXT:                 affine.store %8, %arg3[%arg6] : memref<233008xf32>
// CHECK-NEXT:               }
// CHECK-NEXT:             } {reduction_loop}
// CHECK-NEXT:           }
// CHECK-NEXT:         } {reduction_loop}
// CHECK-NEXT:       }
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }


#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#map2 = affine_map<(d0) -> (d0 + 2)>
#set = affine_set<(d0) : (d0 == 0)>
#set1 = affine_set<(d0) : (-d0 + 1 == 0)>
module {
  func.func @Fused_Sub_Exp_ReduceSum_Log_split_11941850547297653662(%arg0: memref<233008xf32>, %arg1: memref<233008x2xf32>, %arg2: memref<233008x2xf32>, %arg3: memref<233008xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, gpu_parallel_reduce = false, mindspore_kernel, process = "cuda", scop.ignored} {
    %cst = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<233008xf32>
    affine.for %arg4 = 0 to 233008 {
      affine.for %arg5 = 0 to 2 step 2 {
        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
          affine.for %arg7 = #map(%arg5) to #map2(%arg5) {
            %0 = affine.load %arg1[%arg6, %arg7] : memref<233008x2xf32>
            %1 = affine.load %arg0[%arg6] : memref<233008xf32>
            %2 = arith.subf %0, %1 : f32
            affine.store %2, %arg2[%arg6, %arg7] : memref<233008x2xf32>
            affine.if #set(%arg7) {
              affine.store %cst, %alloc[%arg6] : memref<233008xf32>
            }
            %3 = affine.load %arg2[%arg6, %arg7] : memref<233008x2xf32>
            %4 = math.exp %3 : f32
            %5 = affine.load %alloc[%arg6] : memref<233008xf32>
            %6 = arith.addf %4, %5 {enable_atomic_add = false, gpu_parallel_reduce = false, reduction_axes = [1 : index], reduction_type = "x"} : f32
            affine.store %6, %alloc[%arg6] : memref<233008xf32>
            affine.if #set1(%arg7) {
              %7 = affine.load %alloc[%arg6] : memref<233008xf32>
              %8 = math.log %7 : f32
              affine.store %8, %arg3[%arg6] : memref<233008xf32>
            }
          }
        }
      }
    }
    return
  }
}

