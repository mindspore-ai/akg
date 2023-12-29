// RUN: akg-opt %s -affine-load-removal | FileCheck %s

// CHECK-LABEL: func.func @ReduceMax_X(%arg0: memref<4096x7680xf32>, %arg1: memref<4096xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:  	%cst = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:  	%alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
// CHECK-NEXT:  	affine.for %arg2 = 0 to 4096 {
// CHECK-NEXT:  		%0 = affine.for %arg3 = 0 to 7680 iter_args(%arg4 = %cst) -> (f32) {
// CHECK-NEXT:  			%1 = affine.load %arg0[%arg2, %arg3] : memref<4096x7680xf32>
// CHECK-NEXT:  			%2 = arith.maxf %1, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:  			affine.yield %2 : f32
// CHECK-NEXT:  		}
// CHECK-NEXT:  		affine.store %0, %alloc[%arg2] : memref<4096xf32>
// CHECK-NEXT:  	}
// CHECK-NEXT:  	memref.copy %alloc, %arg1 : memref<4096xf32> to memref<4096xf32>
// CHECK-NEXT:  	return
// CHECK-NEXT:  }

func.func @ReduceMax_X(%arg0: memref<4096x7680xf32>, %arg1: memref<4096xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant -3.40282347E+38 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  affine.for %arg2 = 0 to 4096 {
    affine.store %cst, %alloc[%arg2] : memref<4096xf32>
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  memref.copy %alloc, %alloc_0 : memref<4096xf32> to memref<4096xf32>
  affine.for %arg2 = 0 to 4096 {
    affine.for %arg3 = 0 to 7680 {
      %0 = affine.load %arg0[%arg2, %arg3] : memref<4096x7680xf32>
      %1 = affine.load %alloc_0[%arg2] : memref<4096xf32>
      %2 = arith.maxf %0, %1 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %2, %alloc_0[%arg2] : memref<4096xf32>
    }
  }
  memref.copy %alloc_0, %arg1 : memref<4096xf32> to memref<4096xf32>
  return
}


// ---------------------


// CHECK-NEXT: func.func @ReduceSum_All_tile(%arg0: memref<31457280xf32>, %arg1: memref<1xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:   %0 = affine.for %arg2 = 0 to 31457280 step 7680 iter_args(%arg3 = %cst) -> (f32) {
// CHECK-NEXT:     %1 = affine.for %arg4 = #map(%arg2) to #map1(%arg2) iter_args(%arg5 = %cst) -> (f32) {
// CHECK-NEXT:       %3 = affine.load %arg0[%arg4] : memref<31457280xf32>
// CHECK-NEXT:       %4 = arith.addf %3, %arg5 {reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
// CHECK-NEXT:       affine.yield %4 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     %2 = arith.addf %1, %arg3 : f32
// CHECK-NEXT:     affine.yield %2 : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %alloc[] : memref<f32>
// CHECK-NEXT:   %expand_shape = memref.expand_shape %alloc [] : memref<f32> into memref<1xf32>
// CHECK-NEXT:   memref.copy %expand_shape, %arg1 : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 7680)>
func.func @ReduceSum_All_tile(%arg0: memref<31457280xf32>, %arg1: memref<1xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  affine.store %cst, %alloc[] : memref<f32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  memref.copy %alloc, %alloc_0 : memref<f32> to memref<f32>
  affine.for %arg2 = 0 to 31457280 step 7680 {
    affine.for %arg3 = #map(%arg2) to #map1(%arg2) {
      %0 = affine.load %arg0[%arg3] : memref<31457280xf32>
      %1 = affine.load %alloc_0[] : memref<f32>
      %2 = arith.addf %0, %1 {reduction_axes = [0 : index, 1 : index], reduction_type = "all"} : f32
      affine.store %2, %alloc_0[] : memref<f32>
    }
  }
  %expand_shape = memref.expand_shape %alloc_0 [] : memref<f32> into memref<1xf32>
  memref.copy %expand_shape, %arg1 : memref<1xf32> to memref<1xf32>
  return
}


// ---------------------


// CHECK-NEXT: func.func @ReduceSum_All(%arg0: memref<31457280xf32>, %arg1: memref<1xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
// CHECK-NEXT:   %0 = affine.for %arg2 = 0 to 31457280 iter_args(%arg3 = %cst) -> (f32) {
// CHECK-NEXT:     %1 = affine.load %arg0[%arg2] : memref<31457280xf32>
// CHECK-NEXT:     %2 = arith.addf %1, %arg3 {reduction_axes = [0 : index], reduction_type = "all"} : f32
// CHECK-NEXT:     affine.yield %2 : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   affine.store %0, %alloc[] : memref<f32>
// CHECK-NEXT:   %expand_shape = memref.expand_shape %alloc [] : memref<f32> into memref<1xf32>
// CHECK-NEXT:   memref.copy %expand_shape, %arg1 : memref<1xf32> to memref<1xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @ReduceSum_All(%arg0: memref<31457280xf32>, %arg1: memref<1xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<f32>
  affine.store %cst, %alloc[] : memref<f32>
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<f32>
  memref.copy %alloc, %alloc_0 : memref<f32> to memref<f32>
  affine.for %arg2 = 0 to 31457280 {
    %0 = affine.load %arg0[%arg2] : memref<31457280xf32>
    %1 = affine.load %alloc_0[] : memref<f32>
    %2 = arith.addf %0, %1 {reduction_axes = [0 : index], reduction_type = "all"} : f32
    affine.store %2, %alloc_0[] : memref<f32>
  }
  %expand_shape = memref.expand_shape %alloc_0 [] : memref<f32> into memref<1xf32>
  memref.copy %expand_shape, %arg1 : memref<1xf32> to memref<1xf32>
  return
}


// ---------------------


// CHECK-NEXT: func.func @ReduceSum_Y(%arg0: memref<4096x7680xf32>, %arg1: memref<7680xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<7680xf32>
// CHECK-NEXT:   affine.for %arg2 = 0 to 7680 {
// CHECK-NEXT:     affine.store %cst, %alloc[%arg2] : memref<7680xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<7680xf32>
// CHECK-NEXT:   memref.copy %alloc, %alloc_0 : memref<7680xf32> to memref<7680xf32>
// CHECK-NEXT:   affine.for %arg2 = 0 to 4096 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 7680 {
// CHECK-NEXT:       %0 = affine.load %arg0[%arg2, %arg3] : memref<4096x7680xf32>
// CHECK-NEXT:       %1 = affine.load %alloc_0[%arg3] : memref<7680xf32>
// CHECK-NEXT:       %2 = arith.addf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
// CHECK-NEXT:       affine.store %2, %alloc_0[%arg3] : memref<7680xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.copy %alloc_0, %arg1 : memref<7680xf32> to memref<7680xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @ReduceSum_Y(%arg0: memref<4096x7680xf32>, %arg1: memref<7680xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<7680xf32>
  affine.for %arg2 = 0 to 7680 {
    affine.store %cst, %alloc[%arg2] : memref<7680xf32>
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<7680xf32>
  memref.copy %alloc, %alloc_0 : memref<7680xf32> to memref<7680xf32>
  affine.for %arg2 = 0 to 4096 {
    affine.for %arg3 = 0 to 7680 {
      %0 = affine.load %arg0[%arg2, %arg3] : memref<4096x7680xf32>
      %1 = affine.load %alloc_0[%arg3] : memref<7680xf32>
      %2 = arith.addf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
      affine.store %2, %alloc_0[%arg3] : memref<7680xf32>
    }
  }
  memref.copy %alloc_0, %arg1 : memref<7680xf32> to memref<7680xf32>
  return
}


// ---------------------


// CHECK-NEXT: func.func @Fused_Add_ReduceMax(%arg0: memref<4096x7680xf32>, %arg1: memref<4096x7680xf32>, %arg2: memref<4096xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
// CHECK-NEXT:   affine.for %arg3 = 0 to 4096 {
// CHECK-NEXT:     %0 = affine.for %arg4 = 0 to 7680 iter_args(%arg5 = %cst) -> (f32) {
// CHECK-NEXT:       %1 = affine.load %arg0[%arg3, %arg4] : memref<4096x7680xf32>
// CHECK-NEXT:       %2 = affine.load %arg1[%arg3, %arg4] : memref<4096x7680xf32>
// CHECK-NEXT:       %3 = arith.addf %1, %2 : f32
// CHECK-NEXT:       %4 = arith.maxf %3, %arg5 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK-NEXT:       affine.yield %4 : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %0, %alloc[%arg3] : memref<4096xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.copy %alloc, %arg2 : memref<4096xf32> to memref<4096xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

func.func @Fused_Add_ReduceMax(%arg0: memref<4096x7680xf32>, %arg1: memref<4096x7680xf32>, %arg2: memref<4096xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant -3.40282347E+38 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  affine.for %arg3 = 0 to 4096 {
    affine.store %cst, %alloc[%arg3] : memref<4096xf32>
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<4096xf32>
  memref.copy %alloc, %alloc_0 : memref<4096xf32> to memref<4096xf32>
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 7680 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x7680xf32>
      %1 = affine.load %arg1[%arg3, %arg4] : memref<4096x7680xf32>
      %2 = affine.load %alloc_0[%arg3] : memref<4096xf32>
      %3 = arith.addf %0, %1 : f32
      %4 = arith.maxf %3, %2 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %4, %alloc_0[%arg3] : memref<4096xf32>
    }
  }
  memref.copy %alloc_0, %arg2 : memref<4096xf32> to memref<4096xf32>
  return
}


// ---------------------


// CHECK-NEXT: func.func @Fused_ReduceMax_fusion_2365464562326572(%arg0: memref<512x64x2048xf32>, %arg1: memref<512x64xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
// CHECK-NEXT:   %cst = arith.constant -3.40282347E+38 : f32
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x64xf32>
// CHECK-NEXT:   affine.for %arg2 = 0 to 512 {
// CHECK-NEXT:     affine.for %arg3 = 0 to 64 {
// CHECK-NEXT:       %0 = affine.for %arg4 = 0 to 2048 iter_args(%arg5 = %cst) -> (f32) {
// CHECK-NEXT:         %1 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<512x64x2048xf32>
// CHECK-NEXT:         %2 = arith.maxf %1, %arg5 {reduction_axes = [2 : index], reduction_type = "x"} : f32
// CHECK-NEXT:         affine.yield %2 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       affine.store %0, %alloc[%arg2, %arg3] : memref<512x64xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.copy %alloc, %arg1 : memref<512x64xf32> to memref<512x64xf32>
// CHECK-NEXT:  return
// CHECK-NEXT: }

func.func @Fused_ReduceMax_fusion_2365464562326572(%arg0: memref<512x64x2048xf32>, %arg1: memref<512x64xf32>) attributes {OperatorType = "Reduce", mindspore_kernel, scop.ignored} {
  %cst = arith.constant -3.40282347E+38 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x64xf32>
  affine.for %arg2 = 0 to 512 {
    affine.for %arg3 = 0 to 64 {
      affine.store %cst, %alloc[%arg2, %arg3] : memref<512x64xf32>
    }
  }
  %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<512x64xf32>
  memref.copy %alloc, %alloc_0 : memref<512x64xf32> to memref<512x64xf32>
  affine.for %arg2 = 0 to 512 {
    affine.for %arg3 = 0 to 64 {
      affine.for %arg4 = 0 to 2048 {
        %0 = affine.load %arg0[%arg2, %arg3, %arg4] : memref<512x64x2048xf32>
        %1 = affine.load %alloc_0[%arg2, %arg3] : memref<512x64xf32>
        %2 = arith.maxf %0, %1 {reduction_axes = [2 : index], reduction_type = "x"} : f32
        affine.store %2, %alloc_0[%arg2, %arg3] : memref<512x64xf32>
      }
    }
  }
  memref.copy %alloc_0, %arg1 : memref<512x64xf32> to memref<512x64xf32>
  return
}