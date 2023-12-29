// RUN: akg-opt %s -affine-tail-block-tiling | FileCheck %s

// CHECK:  func.func @Fused_Add_fusion_215689222737559696(%arg0: memref<4096x769xf32>, %arg1: memref<769xf32>, %arg2: memref<4096x769xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK:    affine.for %arg3 = 0 to 4096 {
// CHECK:      affine.for %arg4 = 0 to 768 {
// CHECK:        %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x769xf32>
// CHECK:        %1 = affine.load %arg1[%arg4] : memref<769xf32>
// CHECK:        %2 = arith.addf %0, %1 : f32
// CHECK:        affine.store %2, %arg2[%arg3, %arg4] : memref<4096x769xf32>
// CHECK:      }
// CHECK:      affine.for %arg4 = 768 to 769 {
// CHECK:        %0 = affine.load %arg0[%arg3, %arg4] : memref<4096x769xf32>
// CHECK:        %1 = affine.load %arg1[%arg4] : memref<769xf32>
// CHECK:        %2 = arith.addf %0, %1 : f32
// CHECK:        affine.store %2, %arg2[%arg3, %arg4] : memref<4096x769xf32>
// CHECK:      } {tailBlock}
// CHECK:    }
// CHECK:    return
// CHECK:  }

func.func @Fused_Add_fusion_215689222737559696(%arg0: memref<4096x769xf32>, %arg1: memref<769xf32>, %arg2: memref<4096x769xf32>) attributes {OperatorType = "Reshape", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
	affine.for %arg3 = 0 to 4096 {
		affine.for %arg4 = 0 to 769 {
			%0 = affine.load %arg0[%arg3, %arg4] : memref<4096x769xf32>
			%1 = affine.load %arg1[%arg4] : memref<769xf32>
			%2 = arith.addf %0, %1 : f32
			affine.store %2, %arg2[%arg3, %arg4] : memref<4096x769xf32>
		}
	}
	return
}


// ---------------------


// CHECK:  func.func @Fused_ReduceMaxY_fusion_2054638663855723468(%arg0: memref<4096x17xf32>, %arg1: memref<17xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK:    %cst = arith.constant -3.40282347E+38 : f32
// CHECK:    affine.for %arg2 = 0 to 16 {
// CHECK:      affine.store %cst, %arg1[%arg2] : memref<17xf32>
// CHECK:      affine.for %arg3 = 0 to 4096 {
// CHECK:        %0 = affine.load %arg0[%arg3, %arg2] : memref<4096x17xf32>
// CHECK:        %1 = affine.load %arg1[%arg2] : memref<17xf32>
// CHECK:        %2 = arith.maxf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
// CHECK:        affine.store %2, %arg1[%arg2] : memref<17xf32>
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 16 to 17 {
// CHECK:      affine.store %cst, %arg1[%arg2] : memref<17xf32>
// CHECK:      affine.for %arg3 = 0 to 4096 {
// CHECK:        %0 = affine.load %arg0[%arg3, %arg2] : memref<4096x17xf32>
// CHECK:        %1 = affine.load %arg1[%arg2] : memref<17xf32>
// CHECK:        %2 = arith.maxf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
// CHECK:        affine.store %2, %arg1[%arg2] : memref<17xf32>
// CHECK:      }
// CHECK:    } {tailBlock}
// CHECK:    return
// CHECK:  }

func.func @Fused_ReduceMaxY_fusion_2054638663855723468(%arg0: memref<4096x17xf32>, %arg1: memref<17xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
	%cst = arith.constant -3.40282347E+38 : f32
	affine.for %arg2 = 0 to 17 {
		affine.store %cst, %arg1[%arg2] : memref<17xf32>
		affine.for %arg3 = 0 to 4096 {
			%0 = affine.load %arg0[%arg3, %arg2] : memref<4096x17xf32>
			%1 = affine.load %arg1[%arg2] : memref<17xf32>
			%2 = arith.maxf %0, %1 {reduction_axes = [0 : index], reduction_type = "y"} : f32
			affine.store %2, %arg1[%arg2] : memref<17xf32>
		}
	}
	return
}


// ---------------------


// CHECK:  func.func @Fused_Add_ReduceMax_32094723897589235(%arg0: memref<4096x769xf32>, %arg1: memref<769xf32>, %arg2: memref<769xf32>, %arg3: memref<4096xf32>, %arg4: memref<4096x769xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
// CHECK:    %cst = arith.constant -3.40282347E+38 : f32
// CHECK:    affine.for %arg5 = 0 to 4096 {
// CHECK:      affine.for %arg6 = 0 to 768 {
// CHECK:        %0 = affine.load %arg0[%arg5, %arg6] : memref<4096x769xf32>
// CHECK:        %1 = affine.load %arg1[%arg6] : memref<769xf32>
// CHECK:        %2 = arith.addf %0, %1 : f32
// CHECK:        affine.if #set(%arg6) {
// CHECK:          affine.store %cst, %arg3[%arg5] : memref<4096xf32>
// CHECK:        }
// CHECK:        %3 = affine.load %arg3[%arg5] : memref<4096xf32>
// CHECK:        %4 = arith.maxf %2, %3 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK:        affine.store %4, %arg3[%arg5] : memref<4096xf32>
// CHECK:        %5 = affine.load %arg2[%arg6] : memref<769xf32>
// CHECK:        %6 = arith.subf %2, %5 : f32
// CHECK:        affine.store %6, %arg4[%arg5, %arg6] : memref<4096x769xf32>
// CHECK:      }
// CHECK:      affine.for %arg6 = 768 to 769 {
// CHECK:        %0 = affine.load %arg0[%arg5, %arg6] : memref<4096x769xf32>
// CHECK:        %1 = affine.load %arg1[%arg6] : memref<769xf32>
// CHECK:        %2 = arith.addf %0, %1 : f32
// CHECK:        affine.if #set(%arg6) {
// CHECK:          affine.store %cst, %arg3[%arg5] : memref<4096xf32>
// CHECK:        }
// CHECK:        %3 = affine.load %arg3[%arg5] : memref<4096xf32>
// CHECK:        %4 = arith.maxf %2, %3 {reduction_axes = [1 : index], reduction_type = "x"} : f32
// CHECK:        affine.store %4, %arg3[%arg5] : memref<4096xf32>
// CHECK:        %5 = affine.load %arg2[%arg6] : memref<769xf32>
// CHECK:        %6 = arith.subf %2, %5 : f32
// CHECK:        affine.store %6, %arg4[%arg5, %arg6] : memref<4096x769xf32>
// CHECK:      } {tailBlock}
// CHECK:    }
// CHECK:    return
// CHECK:  }

func.func @Fused_Add_ReduceMax_32094723897589235(%arg0: memref<4096x769xf32>, %arg1: memref<769xf32>, %arg2: memref<769xf32>, %arg3: memref<4096xf32>, %arg4: memref<4096x769xf32>) attributes {OperatorType = "Reduce", enable_atomic_add = false, mindspore_kernel, process = "cpu", scop.ignored} {
  %cst = arith.constant -3.40282347E+38 : f32
  affine.for %arg5 = 0 to 4096 {
    affine.for %arg6 = 0 to 769 {
      %0 = affine.load %arg0[%arg5, %arg6] : memref<4096x769xf32>
      %1 = affine.load %arg1[%arg6] : memref<769xf32>
      %2 = arith.addf %0, %1 : f32
      affine.if affine_set<(d0) : (d0 == 0)>(%arg6) {
        affine.store %cst, %arg3[%arg5] : memref<4096xf32>
      }
      %3 = affine.load %arg3[%arg5] : memref<4096xf32>
      %4 = arith.maxf %2, %3 {reduction_axes = [1 : index], reduction_type = "x"} : f32
      affine.store %4, %arg3[%arg5] : memref<4096xf32>
      %5 = affine.load %arg2[%arg6] : memref<769xf32>
      %6 = arith.subf %2, %5 : f32
      affine.store %6, %arg4[%arg5, %arg6] : memref<4096x769xf32>
    }
  }
  return
}