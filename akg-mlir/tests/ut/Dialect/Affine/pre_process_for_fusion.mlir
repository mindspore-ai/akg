// RUN: akg-opt %s --pre-process-for-fusion | FileCheck %s

// CHECK-LABEL:  module {
// CHECK:  func.func @akg_fused__npu_dtype_cast_add_masked_fill_rsub_1_auto_fallback(%arg0: memref<256x10xi32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<10x10xi1, {SymShapeAttr = ["s2", "s1"]}>) -> memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}> attributes {OperatorType = "Default", hacc.function_kind = #hacc.function_kind<HOST>} {
// CHECK:    %cst = arith.constant 1.000000e+00 : f32
// CHECK:    %cst_0 = arith.constant -3.40282347E+38 : f32
// CHECK:    %cst_1 = arith.constant 0.000000e+00 : f32
// CHECK:    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<f32, {SymShapeAttr = []}>
// CHECK:    %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32, {SymShapeAttr = []}>
// CHECK:    %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:    %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:    %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:    %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %expand_shape = memref.expand_shape %alloc_13 {{.*}} output_shape [256, 1, 10, 10] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}> into memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:    %memspacecast = memref.memory_space_cast %expand_shape : memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}> to memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}>
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %arg0[%arg2, %arg4] : memref<256x10xi32, {SymShapeAttr = ["s0", "s1"]}>
// CHECK:          affine.store %3, %alloc[%arg2, %arg3, %arg4] : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %4 = arith.sitofp %3 : i32 to f32
// CHECK:          affine.store %4, %alloc_2[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc_2[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %4 = arith.subf %cst, %3 : f32
// CHECK:          affine.store %4, %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %4 = arith.cmpf une, %3, %cst_1 : f32
// CHECK:          affine.store %4, %alloc_4[%arg2, %arg3, %arg4] : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.store %cst_0, %alloc_5[] : memref<f32, {SymShapeAttr = []}>
// CHECK:    %0 = affine.load %alloc_5[] : memref<f32, {SymShapeAttr = []}>
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          affine.store %0, %alloc_6[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc_4[%arg2, %arg3, %arg4] : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %4 = affine.load %alloc_6[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %5 = affine.load %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %6 = arith.select %3, %4, %5 : f32
// CHECK:          affine.store %6, %alloc_7[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.store %cst_1, %alloc_8[] : memref<f32, {SymShapeAttr = []}>
// CHECK:    %1 = affine.load %alloc_8[] : memref<f32, {SymShapeAttr = []}>
// CHECK:    affine.for %arg2 = 0 to 10 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.store %1, %alloc_9[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:      }
// CHECK:    }
// CHECK:    %2 = affine.load %alloc_5[] : memref<f32, {SymShapeAttr = []}>
// CHECK:    affine.for %arg2 = 0 to 10 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.store %2, %alloc_10[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 10 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        %3 = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:        %4 = affine.load %alloc_9[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:        %5 = affine.load %alloc_10[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:        %6 = arith.select %3, %4, %5 : f32
// CHECK:        affine.store %6, %alloc_11[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc_11[%arg3, %arg4] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
// CHECK:          affine.store %3, %alloc_12[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    affine.for %arg2 = 0 to 256 {
// CHECK:      affine.for %arg3 = 0 to 10 {
// CHECK:        affine.for %arg4 = 0 to 10 {
// CHECK:          %3 = affine.load %alloc_7[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %4 = affine.load %alloc_12[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:          %5 = arith.addf %3, %4 : f32
// CHECK:          affine.store %5, %alloc_13[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
// CHECK:        }
// CHECK:      }
// CHECK:    }
// CHECK:    return %memspacecast : memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}>
// CHECK:  }
// CHECK:}

func.func @akg_fused__npu_dtype_cast_add_masked_fill_rsub_1_auto_fallback(%arg0: memref<256x10xi32, {SymShapeAttr = ["s0", "s1"]}>, %arg1: memref<10x10xi1, {SymShapeAttr = ["s2", "s1"]}>) -> memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}> attributes {OperatorType = "Default", hacc.function_kind = #hacc.function_kind<HOST>} {
  %cst = arith.constant 1.000000e+00 : f32
  %cst_0 = arith.constant -3.40282347E+38 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %arg0[%arg2, %arg4] : memref<256x10xi32, {SymShapeAttr = ["s0", "s1"]}>
        affine.store %0, %alloc[%arg2, %arg3, %arg4] : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc[%arg2, %arg3, %arg4] : memref<256x10x10xi32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %1 = arith.sitofp %0 : i32 to f32
        affine.store %1, %alloc_2[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_2[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %1 = arith.subf %cst, %0 : f32
        affine.store %1, %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %1 = arith.cmpf une, %0, %cst_1 : f32
        affine.store %1, %alloc_4[%arg2, %arg3, %arg4] : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<f32, {SymShapeAttr = []}>
  affine.store %cst_0, %alloc_5[] : memref<f32, {SymShapeAttr = []}>
  %alloc_6 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_5[] : memref<f32, {SymShapeAttr = []}>
        affine.store %0, %alloc_6[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_7 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_4[%arg2, %arg3, %arg4] : memref<256x10x10xi1, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %1 = affine.load %alloc_6[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %2 = affine.load %alloc_3[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %3 = arith.select %0, %1, %2 : f32
        affine.store %3, %alloc_7[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_8 = memref.alloc() {alignment = 64 : i64} : memref<f32, {SymShapeAttr = []}>
  affine.store %cst_1, %alloc_8[] : memref<f32, {SymShapeAttr = []}>
  %alloc_9 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
  affine.for %arg2 = 0 to 10 {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %alloc_8[] : memref<f32, {SymShapeAttr = []}>
      affine.store %0, %alloc_9[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
    }
  }
  %alloc_10 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
  affine.for %arg2 = 0 to 10 {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %alloc_5[] : memref<f32, {SymShapeAttr = []}>
      affine.store %0, %alloc_10[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
    }
  }
  %alloc_11 = memref.alloc() {alignment = 64 : i64} : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
  affine.for %arg2 = 0 to 10 {
    affine.for %arg3 = 0 to 10 {
      %0 = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1, {SymShapeAttr = ["s2", "s1"]}>
      %1 = affine.load %alloc_9[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
      %2 = affine.load %alloc_10[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
      %3 = arith.select %0, %1, %2 : f32
      affine.store %3, %alloc_11[%arg2, %arg3] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
    }
  }
  %alloc_12 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_11[%arg3, %arg4] : memref<10x10xf32, {SymShapeAttr = ["s2", "s1"]}>
        affine.store %0, %alloc_12[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %alloc_13 = memref.alloc() {alignment = 64 : i64} : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  affine.for %arg2 = 0 to 256 {
    affine.for %arg3 = 0 to 10 {
      affine.for %arg4 = 0 to 10 {
        %0 = affine.load %alloc_7[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %1 = affine.load %alloc_12[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc_13[%arg2, %arg3, %arg4] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
      }
    }
  }
  %expand_shape = memref.expand_shape %alloc_13 [[0], [1, 2], [3]] output_shape [256, 1, 10, 10] : memref<256x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}> into memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}>
  %memspacecast = memref.memory_space_cast %expand_shape : memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "s2", "s1"]}> to memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}>
  return %memspacecast : memref<256x1x10x10xf32, {SymShapeAttr = ["s0", "1", "s2", "s1"]}>
}
