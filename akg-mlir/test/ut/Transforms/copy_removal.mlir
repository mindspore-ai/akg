// RUN: akg-opt %s -copy-removal | FileCheck %s

// CHECK-LABEL: func.func @Fused_Add_fusion_215689222737559696(%arg0: memref<4096x7680xf32>, %arg1: memref<7680xf32>, %arg2: memref<4096x7680xf32>) attributes {mindspore_kernel} {
// CHECK-NEXT:  	affine.for %arg3 = 0 to 4096 {
// CHECK-NEXT:    	affine.for %arg4 = 0 to 7680 {
// CHECK-NEXT:      	%0 = affine.load %arg0[%arg3, %arg4] : memref<4096x7680xf32>
// CHECK-NEXT:      	%1 = affine.load %arg1[%arg4] : memref<7680xf32>
// CHECK-NEXT:       	%2 = arith.addf %0, %1 : f32
// CHECK-NEXT:      	affine.store %2, %arg2[%arg3, %arg4] : memref<4096x7680xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }

module {
  func.func @Fused_Add_fusion_215689222737559696(%arg0: memref<4096x7680xf32>, %arg1: memref<7680xf32>, %arg2: memref<4096x7680xf32>) attributes {mindspore_kernel} {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4096x7680xf32>
    affine.for %arg3 = 0 to 4096 {
      affine.for %arg4 = 0 to 7680 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x7680xf32>
        %1 = affine.load %arg1[%arg4] : memref<7680xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg3, %arg4] : memref<4096x7680xf32>
      }
    }
    memref.copy %alloc, %arg2 : memref<4096x7680xf32> to memref<4096x7680xf32>
    return
  }
}
