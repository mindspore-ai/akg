// RUN: akg-opt --alloc-buffer-shrink -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @akg_fused_clone_40_auto_fallback(%arg0: memref<4x2x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<4x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<4x2x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<4x2x116x14x14xf32> attributes {OperatorType = "Elementwise", enable_auto_mark_buffer_size, hacc.block_dim = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
// CHECK-NEXT:   %c4 = arith.constant 4 : index
// CHECK-NEXT:   %c116 = arith.constant 116 : index
// CHECK-NEXT:   %c2 = arith.constant 2 : index
// CHECK-NEXT:   %c196 = arith.constant 196 : index
// CHECK-NEXT:   %c4_0 = arith.constant 4 : index
// CHECK-NEXT:   %c116_1 = arith.constant 116 : index
// CHECK-NEXT:   %c2_2 = arith.constant 2 : index
// CHECK-NEXT:   %c196_3 = arith.constant 196 : index
// CHECK-NEXT:   %c196_4 = arith.constant 196 : index
// CHECK-NEXT:   %c2_5 = arith.constant 2 : index
// CHECK-NEXT:   %c116_6 = arith.constant 116 : index
// CHECK-NEXT:   %c4_7 = arith.constant 4 : index
// CHECK-NEXT:   %c0 = arith.constant 0 : index
// CHECK-NEXT:   %c1 = arith.constant 1 : index
// CHECK-NEXT:   %collapse_shape = memref.collapse_shape %arg0 {{.*}} : memref<4x2x116x14x14xf32> into memref<4x232x196xf32>
// CHECK-NEXT:   %expand_shape = memref.expand_shape %arg1 {{.*}} output_shape [4, 58, 2, 14, 14] : memref<4x116x14x14xf32> into memref<4x58x2x14x14xf32>
// CHECK-NEXT:   %collapse_shape_8 = memref.collapse_shape %expand_shape {{.*}} : memref<4x58x2x14x14xf32> into memref<4x58x2x196xf32>
// CHECK-NEXT:   %subview = memref.subview %collapse_shape[0, 0, 0] [4, 116, 196] [1, 1, 1] : memref<4x232x196xf32> to memref<4x116x196xf32, strided<[45472, 196, 1]>>
// CHECK-NEXT:   %expand_shape_9 = memref.expand_shape %subview {{.*}} output_shape [4, 58, 2, 14, 14] : memref<4x116x196xf32, strided<[45472, 196, 1]>> into memref<4x58x2x14x14xf32, strided<[45472, 392, 196, 14, 1]>>
// CHECK-NEXT:   %collapse_shape_10 = memref.collapse_shape %expand_shape_9 {{.*}} : memref<4x58x2x14x14xf32, strided<[45472, 392, 196, 14, 1]>> into memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
// CHECK-NEXT:   %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x116x2x196xf32>
// CHECK-NEXT:   %c0_11 = arith.constant 0 : index
// CHECK-NEXT:   %subview_12 = memref.subview %alloc[0, 0, 0, 0] [1, 58, 2, 196] [1, 1, 1, 1] : memref<1x116x2x196xf32> to memref<1x58x2x196xf32, strided<[45472, 392, 196, 1]>>
// CHECK-NEXT:   %subview_13 = memref.subview %alloc[0, 58, 0, 0] [1, 58, 2, 196] [1, 1, 1, 1] : memref<1x116x2x196xf32> to memref<1x58x2x196xf32, strided<[45472, 392, 196, 1], offset: 22736>>
// CHECK-NEXT:   %collapse_shape_14 = memref.collapse_shape %arg2 {{.*}} : memref<4x2x116x14x14xf32> into memref<4x2x116x196xf32>
// CHECK-NEXT:   %c0_15 = arith.constant 0 : index
// CHECK-NEXT:   %c1_16 = arith.constant 1 : index
// CHECK-NEXT:   scf.for %arg3 = %c0_15 to %c1_16 step %c1_16 {
// CHECK-NEXT:     scf.for %arg4 = %c0_15 to %c1_16 step %c1_16 {
// CHECK-NEXT:       %c40 = arith.constant 40 : index
// CHECK-NEXT:       %0 = affine.apply #map(%arg3, %arg4)
// CHECK-NEXT:       %c4_17 = arith.constant 4 : index
// CHECK-NEXT:       %1 = affine.apply #map1(%0)[%c4_0]
// CHECK-NEXT:       %2 = affine.min #map2(%1, %c4_17)
// CHECK-NEXT:       %c1_18 = arith.constant 1 : index
// CHECK-NEXT:       %3 = arith.addi %0, %c1_18 : index
// CHECK-NEXT:       %4 = affine.apply #map1(%3)[%c4_0]
// CHECK-NEXT:       %5 = affine.min #map2(%4, %c4_17)
// CHECK-NEXT:       scf.for %arg5 = %2 to %5 step %c1_16 {
// CHECK-NEXT:         %c0_19 = arith.constant 0 : index
// CHECK-NEXT:         %c1_20 = arith.constant 1 : index
// CHECK-NEXT:         scf.for %arg6 = %c0_19 to %c1_20 step %c1_20 {
// CHECK-NEXT:           scf.for %arg7 = %c0_19 to %c1_20 step %c1_20 {
// CHECK-NEXT:             %c40_21 = arith.constant 40 : index
// CHECK-NEXT:             %6 = affine.apply #map(%arg6, %arg7)
// CHECK-NEXT:             %c116_22 = arith.constant 116 : index
// CHECK-NEXT:             %7 = affine.apply #map1(%6)[%c116_1]
// CHECK-NEXT:             %8 = affine.min #map2(%7, %c116_22)
// CHECK-NEXT:             %c1_23 = arith.constant 1 : index
// CHECK-NEXT:             %9 = arith.addi %6, %c1_23 : index
// CHECK-NEXT:             %10 = affine.apply #map1(%9)[%c116_1]
// CHECK-NEXT:             %11 = affine.min #map2(%10, %c116_22)
// CHECK-NEXT:             scf.for %arg8 = %8 to %11 step %c1_20 {
// CHECK-NEXT:               %c0_24 = arith.constant 0 : index
// CHECK-NEXT:               %c1_25 = arith.constant 1 : index
// CHECK-NEXT:               scf.for %arg9 = %c0_24 to %c1_25 step %c1_25 {
// CHECK-NEXT:                 scf.for %arg10 = %c0_24 to %c1_25 step %c1_25 {
// CHECK-NEXT:                   %c40_26 = arith.constant 40 : index
// CHECK-NEXT:                   %12 = affine.apply #map(%arg9, %arg10)
// CHECK-NEXT:                   %c2_27 = arith.constant 2 : index
// CHECK-NEXT:                   %13 = affine.apply #map1(%12)[%c2_2]
// CHECK-NEXT:                   %14 = affine.min #map2(%13, %c2_27)
// CHECK-NEXT:                   %c1_28 = arith.constant 1 : index
// CHECK-NEXT:                   %15 = arith.addi %12, %c1_28 : index
// CHECK-NEXT:                   %16 = affine.apply #map1(%15)[%c2_2]
// CHECK-NEXT:                   %17 = affine.min #map2(%16, %c2_27)
// CHECK-NEXT:                   scf.for %arg11 = %14 to %17 step %c1_25 {
// CHECK-NEXT:                     %c0_29 = arith.constant 0 : index
// CHECK-NEXT:                     %c1_30 = arith.constant 1 : index
// CHECK-NEXT:                     scf.for %arg12 = %c0_29 to %c1_30 step %c1_30 {
// CHECK-NEXT:                       scf.for %arg13 = %c0_29 to %c1_30 step %c1_30 {
// CHECK-NEXT:                         %c40_31 = arith.constant 40 : index
// CHECK-NEXT:                         %18 = affine.apply #map(%arg12, %arg13)
// CHECK-NEXT:                         %c196_32 = arith.constant 196 : index
// CHECK-NEXT:                         %19 = affine.apply #map1(%18)[%c196_3]
// CHECK-NEXT:                         %20 = affine.min #map2(%19, %c196_32)
// CHECK-NEXT:                         %c1_33 = arith.constant 1 : index
// CHECK-NEXT:                         %21 = arith.addi %18, %c1_33 : index
// CHECK-NEXT:                         %22 = affine.apply #map1(%21)[%c196_3]
// CHECK-NEXT:                         %23 = affine.min #map2(%22, %c196_32)
// CHECK-NEXT:                         scf.for %arg14 = %20 to %23 step %c1_30 {
// CHECK-NEXT:                           %24 = affine.apply #map3(%arg8)
// CHECK-NEXT:                           %25 = arith.cmpi sge, %24, %c0 {skip_vectorize} : index
// CHECK-NEXT:                           scf.if %25 {
// CHECK-NEXT:                             %29 = memref.load %collapse_shape_10[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
// CHECK-NEXT:                             memref.store %29, %subview_12[%c0_11, %arg8, %arg11, %arg14] : memref<1x58x2x196xf32, strided<[45472, 392, 196, 1]>>
// CHECK-NEXT:                           }
// CHECK-NEXT:                           %26 = affine.apply #map3(%arg8)
// CHECK-NEXT:                           %27 = arith.cmpi sge, %26, %c0 {skip_vectorize} : index
// CHECK-NEXT:                           scf.if %27 {
// CHECK-NEXT:                             %29 = memref.load %collapse_shape_8[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32>
// CHECK-NEXT:                             memref.store %29, %subview_13[%c0_11, %arg8, %arg11, %arg14] : memref<1x58x2x196xf32, strided<[45472, 392, 196, 1], offset: 22736>>
// CHECK-NEXT:                           }
// CHECK-NEXT:                           %28 = memref.load %alloc[%c0_11, %arg8, %arg11, %arg14] : memref<1x116x2x196xf32>
// CHECK-NEXT:                           memref.store %28, %collapse_shape_14[%arg5, %arg11, %arg8, %arg14] : memref<4x2x116x196xf32>
// CHECK-NEXT:                         }
// CHECK-NEXT:                       }
// CHECK-NEXT:                     }
// CHECK-NEXT:                   }
// CHECK-NEXT:                 }
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           }
// CHECK-NEXT:         }
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   } {map_for_to_forall}
// CHECK-NEXT:   return %arg2 : memref<4x2x116x14x14xf32>
// CHECK-NEXT: }


#map = affine_map<(d0, d1) -> (d0 + d1)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0, d1) -> (d0, d1)>
#map3 = affine_map<(d0) -> (-d0 + 57)>
module {
  memref.global "private" @global_seed : memref<i64> = dense<0>
  func.func @akg_fused_clone_40_auto_fallback(%arg0: memref<4x2x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<4x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<4x2x116x14x14xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<4x2x116x14x14xf32> attributes {OperatorType = "Elementwise", enable_auto_mark_buffer_size, hacc.block_dim = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>} {
    %c4 = arith.constant 4 : index
    %c116 = arith.constant 116 : index
    %c2 = arith.constant 2 : index
    %c196 = arith.constant 196 : index
    %c4_0 = arith.constant 4 : index
    %c116_1 = arith.constant 116 : index
    %c2_2 = arith.constant 2 : index
    %c196_3 = arith.constant 196 : index
    %c196_4 = arith.constant 196 : index
    %c2_5 = arith.constant 2 : index
    %c116_6 = arith.constant 116 : index
    %c4_7 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0], [1, 2], [3, 4]] : memref<4x2x116x14x14xf32> into memref<4x232x196xf32>
    %expand_shape = memref.expand_shape %arg1 [[0], [1, 2], [3], [4]] output_shape [4, 58, 2, 14, 14] : memref<4x116x14x14xf32> into memref<4x58x2x14x14xf32>
    %collapse_shape_8 = memref.collapse_shape %expand_shape [[0], [1], [2], [3, 4]] : memref<4x58x2x14x14xf32> into memref<4x58x2x196xf32>
    %subview = memref.subview %collapse_shape[0, 0, 0] [4, 116, 196] [1, 1, 1] : memref<4x232x196xf32> to memref<4x116x196xf32, strided<[45472, 196, 1]>>
    %expand_shape_9 = memref.expand_shape %subview [[0], [1, 2], [3, 4]] output_shape [4, 58, 2, 14, 14] : memref<4x116x196xf32, strided<[45472, 196, 1]>> into memref<4x58x2x14x14xf32, strided<[45472, 392, 196, 14, 1]>>
    %collapse_shape_10 = memref.collapse_shape %expand_shape_9 [[0], [1], [2], [3, 4]] : memref<4x58x2x14x14xf32, strided<[45472, 392, 196, 14, 1]>> into memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<4x116x2x196xf32>
    %subview_11 = memref.subview %alloc[0, 0, 0, 0] [4, 58, 2, 196] [1, 1, 1, 1] : memref<4x116x2x196xf32> to memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
    %subview_12 = memref.subview %alloc[0, 58, 0, 0] [4, 58, 2, 196] [1, 1, 1, 1] : memref<4x116x2x196xf32> to memref<4x58x2x196xf32, strided<[45472, 392, 196, 1], offset: 22736>>
    %collapse_shape_13 = memref.collapse_shape %arg2 [[0], [1], [2], [3, 4]] : memref<4x2x116x14x14xf32> into memref<4x2x116x196xf32>
    %c0_14 = arith.constant 0 : index
    %c1_15 = arith.constant 1 : index
    scf.for %arg3 = %c0_14 to %c1_15 step %c1_15 {
      scf.for %arg4 = %c0_14 to %c1_15 step %c1_15 {
        %c40 = arith.constant 40 : index
        %0 = affine.apply #map(%arg3, %arg4)
        %c4_16 = arith.constant 4 : index
        %1 = affine.apply #map1(%0)[%c4_0]
        %2 = affine.min #map2(%1, %c4_16)
        %c1_17 = arith.constant 1 : index
        %3 = arith.addi %0, %c1_17 : index
        %4 = affine.apply #map1(%3)[%c4_0]
        %5 = affine.min #map2(%4, %c4_16)
        scf.for %arg5 = %2 to %5 step %c1_15 {
          %c0_18 = arith.constant 0 : index
          %c1_19 = arith.constant 1 : index
          scf.for %arg6 = %c0_18 to %c1_19 step %c1_19 {
            scf.for %arg7 = %c0_18 to %c1_19 step %c1_19 {
              %c40_20 = arith.constant 40 : index
              %6 = affine.apply #map(%arg6, %arg7)
              %c116_21 = arith.constant 116 : index
              %7 = affine.apply #map1(%6)[%c116_1]
              %8 = affine.min #map2(%7, %c116_21)
              %c1_22 = arith.constant 1 : index
              %9 = arith.addi %6, %c1_22 : index
              %10 = affine.apply #map1(%9)[%c116_1]
              %11 = affine.min #map2(%10, %c116_21)
              scf.for %arg8 = %8 to %11 step %c1_19 {
                %c0_23 = arith.constant 0 : index
                %c1_24 = arith.constant 1 : index
                scf.for %arg9 = %c0_23 to %c1_24 step %c1_24 {
                  scf.for %arg10 = %c0_23 to %c1_24 step %c1_24 {
                    %c40_25 = arith.constant 40 : index
                    %12 = affine.apply #map(%arg9, %arg10)
                    %c2_26 = arith.constant 2 : index
                    %13 = affine.apply #map1(%12)[%c2_2]
                    %14 = affine.min #map2(%13, %c2_26)
                    %c1_27 = arith.constant 1 : index
                    %15 = arith.addi %12, %c1_27 : index
                    %16 = affine.apply #map1(%15)[%c2_2]
                    %17 = affine.min #map2(%16, %c2_26)
                    scf.for %arg11 = %14 to %17 step %c1_24 {
                      %c0_28 = arith.constant 0 : index
                      %c1_29 = arith.constant 1 : index
                      scf.for %arg12 = %c0_28 to %c1_29 step %c1_29 {
                        scf.for %arg13 = %c0_28 to %c1_29 step %c1_29 {
                          %c40_30 = arith.constant 40 : index
                          %18 = affine.apply #map(%arg12, %arg13)
                          %c196_31 = arith.constant 196 : index
                          %19 = affine.apply #map1(%18)[%c196_3]
                          %20 = affine.min #map2(%19, %c196_31)
                          %c1_32 = arith.constant 1 : index
                          %21 = arith.addi %18, %c1_32 : index
                          %22 = affine.apply #map1(%21)[%c196_3]
                          %23 = affine.min #map2(%22, %c196_31)
                          scf.for %arg14 = %20 to %23 step %c1_29 {
                            %24 = affine.apply #map3(%arg8)
                            %25 = arith.cmpi sge, %24, %c0 {skip_vectorize} : index
                            scf.if %25 {
                              %29 = memref.load %collapse_shape_10[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
                              memref.store %29, %subview_11[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32, strided<[45472, 392, 196, 1]>>
                            }
                            %26 = affine.apply #map3(%arg8)
                            %27 = arith.cmpi sge, %26, %c0 {skip_vectorize} : index
                            scf.if %27 {
                              %29 = memref.load %collapse_shape_8[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32>
                              memref.store %29, %subview_12[%arg5, %arg8, %arg11, %arg14] : memref<4x58x2x196xf32, strided<[45472, 392, 196, 1], offset: 22736>>
                            }
                            %28 = memref.load %alloc[%arg5, %arg8, %arg11, %arg14] : memref<4x116x2x196xf32>
                            memref.store %28, %collapse_shape_13[%arg5, %arg11, %arg8, %arg14] : memref<4x2x116x196xf32>
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    } {map_for_to_forall}
    return %arg2 : memref<4x2x116x14x14xf32>
  }
}
