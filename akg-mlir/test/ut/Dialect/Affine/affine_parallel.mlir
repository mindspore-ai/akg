// RUN: akg-opt %s --akg-affine-loop-parallelize | FileCheck %s

// CHECK: func.func @Fused_Add_fusion_3594431651152944861(%arg0: memref<4096x2xf32>, %arg1: memref<4096x2xf32>, %arg2: memref<4096x2xf32>) attributes {OperatorType = "Elementwise", mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK: 	affine.for %arg3 = 0 to 4096 {
// CHECK: 		affine.for %arg4 = 0 to 2 {
// CHECK: 			%0 = affine.load %arg0[%arg3, %arg4] : memref<4096x2xf32>
// CHECK: 			%1 = affine.load %arg1[%arg3, %arg4] : memref<4096x2xf32>
// CHECK: 			%2 = arith.addf %0, %1 : f32
// CHECK: 			affine.store %2, %arg2[%arg3, %arg4] : memref<4096x2xf32>
// CHECK: 		}
// CHECK: 	}
// CHECK: 	return
// CHECK: }

func.func @Fused_Add_fusion_3594431651152944861(%arg0: memref<4096x2xf32>, %arg1: memref<4096x2xf32>, %arg2: memref<4096x2xf32>) attributes {OperatorType = "Elementwise", mindspore_kernel, process = "cpu", scop.ignored} {
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 2 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x2xf32>
      %1 = affine.load %arg1[%arg3, %arg4] : memref<4096x2xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg3, %arg4] : memref<4096x2xf32>
    }
  }
  return
}


// ---------------------


// CHECK: func.func @Fused_Add_fusion_3594431651152944862(%arg0: memref<4096x20xf32>, %arg1: memref<4096x20xf32>, %arg2: memref<4096x20xf32>) attributes {OperatorType = "Elementwise", mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK: 	affine.parallel (%arg3) = (0) to (4096) {
// CHECK: 		affine.for %arg4 = 0 to 20 {
// CHECK: 			%0 = affine.load %arg0[%arg3, %arg4] : memref<4096x20xf32>
// CHECK: 			%1 = affine.load %arg1[%arg3, %arg4] : memref<4096x20xf32>
// CHECK: 			%2 = arith.addf %0, %1 : f32
// CHECK: 			affine.store %2, %arg2[%arg3, %arg4] : memref<4096x20xf32>
// CHECK: 		}
// CHECK: 	}
// CHECK: 	return
// CHECK: }

func.func @Fused_Add_fusion_3594431651152944862(%arg0: memref<4096x20xf32>, %arg1: memref<4096x20xf32>, %arg2: memref<4096x20xf32>) attributes {OperatorType = "Elementwise", mindspore_kernel, process = "cpu", scop.ignored} {
  affine.for %arg3 = 0 to 4096 {
    affine.for %arg4 = 0 to 20 {
      %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x20xf32>
      %1 = affine.load %arg1[%arg3, %arg4] : memref<4096x20xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg3, %arg4] : memref<4096x20xf32>
    }
  }
  return
}
