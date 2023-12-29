// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" --linalg-generalize-named-ops -linalg-fuse-template-ops | FileCheck %s

#map = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 2048)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map3 = affine_map<(d0) -> (d0 ceildiv 8)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0, 512)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#func_map10 = affine_map<(d0, d1, d2, d3) -> (d3)>
#func_map00 = affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>
#func_map01 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#func_map02 = affine_map<(d0, d1, d2) -> (d2)>
#func_map03 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#func_map1 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#func_map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4)>
#func_map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

module {
  func.func @template_batch_matmul_4d_transpose_b_f16_f16_f16(%targ0: tensor<?x?x?x?xf16>, %targ1: tensor<?x?x?x?xf16>, %targ2: tensor<?x?x?x?xf16> {bufferization.access = "write"}, %batch_outer: index, %batch_inner:index, %m: index, %n:index, %k:index) attributes {fusion.kind="identityWithBroadPerm"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?x?x?xf16>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?x?x?xf16>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?x?x?xf16>

    %cst = arith.constant 0.000000e+00 : f16
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    //%dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf16, #map>            // batch_outer
    //%dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf16, #map>          // batch_inner
    //%dim_1 = memref.dim %arg0, %c2 : memref<?x?x?x?xf16, #map>          // m
    //%dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf16, #map>          // k
    //%dim_3 = memref.dim %arg1, %c2 : memref<?x?x?x?xf16, #map>          // n
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256x1x1x1x8xf16>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<64x256x1x1x1x8xf16>
    scf.for %arg3 = %c0 to %batch_outer step %c1 {                        // batch_outer
      scf.for %arg4 = %c0 to %batch_inner step %c1 {                      // batch_inner
        scf.for %arg5 = %c0 to %m step %c2048 {                           // m o
          %0 = affine.min #map1(%arg5)[%m]
          scf.for %arg6 = %c0 to %k step %c256 {                          // k o
            %1 = affine.min #map2(%arg6)[%k]
            //%subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, %0, %1] [1, 1, 1, 1] : memref<?x?x?x?xf16, #map> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            scf.for %arg7 = %c0 to %0 step %c8 {                          // m io
              %2 = affine.apply #map3(%arg7)
              %3 = affine.min #map4(%arg7)[%0]
              scf.for %arg8 = %c0 to %1 step %c1 {                        // k io
                %100 = arith.addi %arg5, %arg7 : index
                %200 = arith.addi %arg6, %arg8 : index
                //%subview_5 = memref.subview %subview[%c0, %c0, %arg7, %arg8] [1, 1, %3, 1] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x1xf16, strided<[?, ?, ?, 1], offset: ?>>
                %44 = fusion.load %arg0[%c0, %c0, %100, %200], %cst {in_bounds = [true, true, false, true]} : memref<?x?x?x?xf16>, vector<1x1x8x1xf16>
                fusion.multi_load %arg0[%c0, %c0, %100, %200] : memref<?x?x?x?xf16>, vector<1x1x8x1xf16>
                //%4 = vector.transfer_read %subview_5[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, true]} : memref<1x1x?x1xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x8x1xf16>
                %444 = fusion.insert %arg0, %44 : memref<?x?x?x?xf16>, vector<1x1x8x1xf16> to vector<1x1x8x1xf16>
                %5 = vector.transpose %444, [0, 1, 3, 2] : vector<1x1x8x1xf16> to vector<1x1x1x8xf16>
                vector.transfer_write %5, %alloc[%2, %arg8, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<256x256x1x1x1x8xf16>
              }
            }
            scf.for %arg7 = %c0 to %n step %c512 {                        // n o
              %2 = affine.min #map5(%arg7)[%n]
              //%subview_5 = memref.subview %arg1[%arg3, %arg4, %arg7, %arg6] [1, 1, %2, %1] [1, 1, 1, 1] : memref<?x?x?x?xf16, #map> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              //%subview_6 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg7] [1, 1, %0, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16, #map> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              %subview_5 = memref.subview %arg1[%arg3, %arg4, %arg7, %arg6] [1, 1, %2, %1] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              %subview_6 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg7] [1, 1, %0, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              scf.for %arg8 = %c0 to %2 step %c8 {                        // n io
                %3 = affine.apply #map3(%arg8)
                %4 = affine.min #map4(%arg8)[%2]
                scf.for %arg9 = %c0 to %1 step %c1 {                      // k io
                  %subview_7 = memref.subview %subview_5[%c0, %c0, %arg8, %arg9] [1, 1, %4, 1] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x1xf16, strided<[?, ?, ?, 1], offset: ?>>
                  %5 = vector.transfer_read %subview_7[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, true]} : memref<1x1x?x1xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x8x1xf16>
                  %6 = vector.transpose %5, [0, 1, 3, 2] : vector<1x1x8x1xf16> to vector<1x1x1x8xf16>
                  vector.transfer_write %6, %alloc_4[%3, %arg9, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x1x8xf16>, memref<64x256x1x1x1x8xf16>
                }
              }
              scf.for %arg8 = %c0 to %0 step %c8 {                        // m io
                %3 = affine.min #map4(%arg8)[%0]
                %4 = affine.apply #map3(%arg8)
                scf.for %arg9 = %c0 to %2 step %c8 {                      // n io
                  %5 = affine.min #map4(%arg9)[%2]
                  %6 = affine.apply #map3(%arg9)
                  %subview_7 = memref.subview %subview_6[%c0, %c0, %arg8, %arg9] [1, 1, %3, %5] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                  %not_first = arith.cmpi ne, %arg6, %c0 : index
                  %777 = scf.if %not_first -> (vector<1x1x8x8xf16>) {
                    %7 = vector.transfer_read %subview_7[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, false]} : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x8x8xf16>
                    scf.yield %7 : vector<1x1x8x8xf16>
                  } else {
                    scf.yield %cst_0 : vector<1x1x8x8xf16>
                  }
                  %8 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %777) -> (vector<1x1x8x8xf16>) {  // k io
                    %9 = vector.transfer_read %alloc[%4, %arg10, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<256x256x1x1x1x8xf16>, vector<1x1x1x8xf16>
                    %10 = vector.transfer_read %alloc_4[%6, %arg10, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<64x256x1x1x1x8xf16>, vector<1x1x1x8xf16>
                    %11 = vector.contract {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %arg11 : vector<1x1x1x8xf16>, vector<1x1x1x8xf16> into vector<1x1x8x8xf16>
                    scf.yield %11 : vector<1x1x8x8xf16>
                  }
                  %300 = arith.addi %arg5, %arg8 : index
                  %400 = arith.addi %arg7, %arg9 : index
                  %last = arith.subi %k, %c256 : index
                  %is_last = arith.cmpi sge, %arg6, %last : index
                  %888 = scf.if %is_last -> (vector<1x1x8x8xf16>) {
                    fusion.multi_load %arg2[%arg3, %arg4, %300, %400], %cst : memref<?x?x?x?xf16>, vector<1x1x8x8xf16>
                    %88 = fusion.insert %arg2, %8 : memref<?x?x?x?xf16>, vector<1x1x8x8xf16> to vector<1x1x8x8xf16>
                    scf.yield %88 : vector<1x1x8x8xf16>
                  } else {
                    scf.yield %8 : vector<1x1x8x8xf16>
                  }
                  fusion.store %888, %arg2[%arg3, %arg4, %300, %400] : vector<1x1x8x8xf16>, memref<?x?x?x?xf16>
                  //memref.copy %subview_7, %subview_7 : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                }
              }
              //memref.copy %subview_6, %subview_6 : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x256x1x1x1x8xf16>
    memref.dealloc %alloc_4 : memref<64x256x1x1x1x8xf16>
    //return %arg2 : memref<?x?x?x?xf16, #map>
    return
  }

  func.func @fuse_multi_src_broadcast_vector_matmul(%arg0: tensor<8xf16>, %arg1: tensor<2x2x32x8xf16>, %arg2: tensor<2x2x16x32xf16>) -> tensor<2x2x16x32xf16> {
    %0 = tensor.empty() : tensor<2x2x16x8xf16>
    %res_0 = tensor.empty() : tensor<2x16x8xf16>
    %res_1 = tensor.empty() : tensor<2x16x8xf16>
    %cst_1 = arith.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf16>
    %cst_2 = arith.constant dense<[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]> : tensor<8xf16>
    %arg_0 = linalg.generic {indexing_maps = [#func_map02, #func_map02, #func_map03], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst_1 : tensor<8xf16>, tensor<8xf16>) outs(%res_0 : tensor<2x16x8xf16>) {
    ^bb0(%in0: f16, %in1: f16, %out: f16):
      %5 = arith.addf %in0, %in1 : f16
      linalg.yield %5 : f16
    } -> tensor<2x16x8xf16>
    %arg_1 = linalg.generic {indexing_maps = [#func_map02, #func_map02, #func_map03], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %cst_2 : tensor<8xf16>, tensor<8xf16>) outs(%res_1 : tensor<2x16x8xf16>) {
    ^bb0(%in0: f16, %in1: f16, %out: f16):
      %5 = arith.addf %in0, %in1 : f16
      linalg.yield %5 : f16
    } -> tensor<2x16x8xf16>
    %1 = linalg.generic {indexing_maps = [#func_map00, #func_map00, #func_map01], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg_0, %arg_1 : tensor<2x16x8xf16>, tensor<2x16x8xf16>) outs(%0 : tensor<2x2x16x8xf16>) {
    ^bb0(%in0: f16, %in1: f16, %out: f16):
      //%3 = arith.extf %in0 : f8 to f16
      //%4 = arith.extf %in1 : f8 to f16
      //%5 = arith.addf %3, %4 : f16
      %5 = arith.addf %in0, %in1 : f16
      linalg.yield %5 : f16
    } -> tensor<2x2x16x8xf16>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<2x2x16x8xf16>, tensor<2x2x32x8xf16>) outs(%arg2 : tensor<2x2x16x32xf16>) attrs =  {template_func = @template_batch_matmul_4d_transpose_b_f16_f16_f16} {
    ^bb0(%in: f16, %in_0: f16, %init: f16):
      %3 = arith.mulf %in, %in_0 : f16
      %4 = arith.addf %init, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<2x2x16x32xf16>
    return %2 : tensor<2x2x16x32xf16>
  }
}

// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0:.+]][%[[INDEX0:.+]]], %[[CST0:.+]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: %[[FUSION_BROADCAST0:.+]] = fusion.broadcast %[[FUSION_INSERT0]]

// CHECK: %[[FUSION_LOAD1:.+]] = fusion.load %[[MEM_ARG1:.+]][%[[INDEX0]]]
// CHECK: %[[FUSION_INSERT1:.+]] = fusion.insert %[[MEM_ARG1]], %[[FUSION_LOAD1]]
// CHECK: %[[FUSION_BROADCAST1:.+]] = fusion.broadcast %[[FUSION_INSERT1]]

// CHECK: %[[ADD_RESULT0:.+]] = arith.addf %[[FUSION_BROADCAST0]], %[[FUSION_BROADCAST1]]
// CHECK: %[[FUSION_BROADCAST_ADD0:.+]] = fusion.broadcast %[[ADD_RESULT0]]

// CHECK: %[[FUSION_LOAD2:.+]] = fusion.load %[[MEM_ARG2:.+]][%[[INDEX0]]]
// CHECK: %[[FUSION_INSERT2:.+]] = fusion.insert %[[MEM_ARG2]], %[[FUSION_LOAD2]]
// CHECK: %[[FUSION_BROADCAST2:.+]] = fusion.broadcast %[[FUSION_INSERT2]]

// CHECK: %[[FUSION_LOAD3:.+]] = fusion.load %[[MEM_ARG3:.+]][%[[INDEX0]]]
// CHECK: %[[FUSION_INSERT3:.+]] = fusion.insert %[[MEM_ARG3]], %[[FUSION_LOAD3]]
// CHECK: %[[FUSION_BROADCAST3:.+]] = fusion.broadcast %[[FUSION_INSERT3]]

// CHECK: %[[ADD_RESULT1:.+]] = arith.addf %[[FUSION_BROADCAST2]], %[[FUSION_BROADCAST3]]
// CHECK: %[[FUSION_BROADCAST_ADD1:.+]] = fusion.broadcast %[[ADD_RESULT1]]

// CHECK: %[[ADD_RESULT2:.+]] = arith.addf %[[FUSION_BROADCAST_ADD0]], %[[FUSION_BROADCAST_ADD1]]


// -----

#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 2048)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map5 = affine_map<(d0) -> (d0 ceildiv 8)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
#func_map00 = affine_map<(d0, d1) -> (d1, d0)>
#func_map01 = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ2: tensor<?x?xf32>, %targ3: tensor<?x?xf32>, %targ4: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identityWithBroadPerm"} {
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %arg3 = bufferization.to_memref %targ3 : memref<?x?xf32>
    %arg4 = bufferization.to_memref %targ4 : memref<?x?xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8x16xf32>
    %dim = memref.dim %arg4, %c0 : memref<?x?xf32>
    %dim_1 = memref.dim %arg4, %c1 : memref<?x?xf32>
    scf.for %arg5 = %c0 to %dim step %c8 {
      %0 = affine.min #map1(%arg5)[%dim]
      scf.for %arg6 = %c0 to %dim_1 step %c16 {
        %1 = affine.min #map2(%arg6)[%dim_1]
        %subview = memref.subview %arg4[%arg5, %arg6] [%0, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        vector.transfer_write %cst_0, %subview[%c0, %c0] : vector<8x16xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
      }
    }
    //%m = memref.dim %arg2, %c0 : memref<?x?xf32> // m
    //%k = memref.dim %arg2, %c1 : memref<?x?xf32> // k
    //%n = memref.dim %arg3, %c0 : memref<?x?xf32> // n
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256x1x8xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x256x1x8xf32>
    scf.for %arg5 = %c0 to %m step %c2048 {
      %0 = affine.min #map3(%arg5)[%m]
      scf.for %arg6 = %c0 to %k step %c256 {
        %1 = affine.min #map4(%arg6)[%k]
        %subview = memref.subview %arg2[%arg5, %arg6] [%0, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        scf.for %arg7 = %c0 to %0 step %c8 {
          %2 = affine.apply #map5(%arg7)
          %3 = affine.min #map1(%arg7)[%0]
          scf.for %arg8 = %c0 to %1 step %c1 {
            %index_n = arith.addi %arg5, %arg7 : index
            %index_k = arith.addi %arg6, %arg8 : index
            %4 = fusion.load %arg2[%index_n, %index_k], %cst {in_bounds = [false, true]} : memref<?x?xf32>, vector<8x1xf32>
            fusion.multi_load %arg2[%index_n, %index_k] : memref<?x?xf32>, vector<8x1xf32>
            %444 = fusion.insert %arg2, %4 : memref<?x?xf32>, vector<8x1xf32> to vector<8x1xf32>
            %5 = vector.transpose %444, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
            vector.transfer_write %5, %alloc[%2, %arg8, %c0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<256x256x1x8xf32>
          }
        }
        scf.for %arg7 = %c0 to %n step %c256 {
          %2 = affine.min #map4(%arg7)[%n]
          %subview_6 = memref.subview %arg3[%arg7, %arg6] [%2, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_7 = memref.subview %arg4[%arg5, %arg7] [%0, %2] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.for %arg8 = %c0 to %2 step %c8 {
            %3 = affine.apply #map5(%arg8)
            %4 = affine.min #map1(%arg8)[%2]
            scf.for %arg9 = %c0 to %1 step %c1 {
              %subview_8 = memref.subview %subview_6[%arg8, %arg9] [%4, 1] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x1xf32, strided<[?, 1], offset: ?>>
              %5 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x1xf32, strided<[?, 1], offset: ?>>, vector<8x1xf32>
              %6 = vector.transpose %5, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
              vector.transfer_write %6, %alloc_5[%3, %arg9, %c0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<32x256x1x8xf32>
            }
          }
          scf.for %arg8 = %c0 to %0 step %c8 {
            %3 = affine.min #map1(%arg8)[%0]
            %4 = affine.apply #map5(%arg8)
            scf.for %arg9 = %c0 to %2 step %c8 {
              %5 = affine.min #map1(%arg9)[%2]
              %6 = affine.apply #map5(%arg9)
              %subview_8 = memref.subview %subview_7[%arg8, %arg9] [%3, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %7 = vector.transfer_read %subview_8[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
              %8 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %7) -> (vector<8x8xf32>) {
                %9 = vector.transfer_read %alloc[%4, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<256x256x1x8xf32>, vector<1x8xf32>
                %10 = vector.transfer_read %alloc_5[%6, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<32x256x1x8xf32>, vector<1x8xf32>
                %11 = vector.contract {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %arg11 : vector<1x8xf32>, vector<1x8xf32> into vector<8x8xf32>
                scf.yield %11 : vector<8x8xf32>
              }
              vector.transfer_write %8, %subview_8[%c0, %c0] : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x256x1x8xf32>
    memref.dealloc %alloc_5 : memref<32x256x1x8xf32>
    return
  }

  func.func @fuse_multi_src_permutation_vector_matmul(%arg00: tensor<16x8xf32>, %arg01: tensor<8x16xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<8x16xf32>
    %1 = tensor.empty() : tensor<16x8xf32>
    %2 = linalg.generic {indexing_maps = [#func_map00, #func_map01, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%arg00, %arg01 : tensor<16x8xf32>, tensor<8x16xf32>) outs(%0 : tensor<8x16xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %5 = arith.subf %in0, %in1 : f32
      linalg.yield %5 : f32
    } -> tensor<8x16xf32>
    %3 = linalg.generic {indexing_maps = [#func_map00, #func_map00, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%2, %arg01 : tensor<8x16xf32>, tensor<8x16xf32>) outs(%1 : tensor<16x8xf32>) {
    ^bb0(%in0: f32, %in1: f32, %out: f32):
      %5 = arith.addf %in0, %in1 : f32
      linalg.yield %5 : f32
    } -> tensor<16x8xf32>
    %4 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%3, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %5 = arith.mulf %in, %in_0 : f32
      %6 = arith.addf %init, %5 : f32
      linalg.yield %6 : f32
    } -> tensor<16x32xf32>
    return %4 : tensor<16x32xf32>
  }
}

// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0:.+]][%[[INDEX0:.+]], %[[INDEX1:.+]]], %[[CST0:.+]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: %[[FUSION_TRANSPOSE0:.+]] = fusion.transpose %[[FUSION_INSERT0]]

// CHECK: %[[FUSION_LOAD1:.+]] = fusion.load %[[MEM_ARG1:.+]][%[[INDEX1:.+]], %[[INDEX0]]]
// CHECK: %[[FUSION_INSERT1:.+]] = fusion.insert %[[MEM_ARG1]], %[[FUSION_LOAD1]]

// CHECK: %[[SUB_RESULT0:.+]] = arith.subf %[[FUSION_TRANSPOSE0]], %[[FUSION_INSERT1]]
// CHECK: %[[FUSION_TRANSPOSE_RESULT0:.+]] = fusion.transpose %[[SUB_RESULT0]]

// CHECK: %[[FUSION_LOAD2:.+]] = fusion.load %[[MEM_ARG2:.+]][%[[INDEX1:.+]], %[[INDEX0]]]
// CHECK: %[[FUSION_INSERT2:.+]] = fusion.insert %[[MEM_ARG2]], %[[FUSION_LOAD2]]
// CHECK: %[[FUSION_TRANSPOSE2:.+]] = fusion.transpose %[[FUSION_INSERT2]]

// CHECK: %[[ADD_RESULT0:.+]] = arith.addf %[[FUSION_TRANSPOSE_RESULT0]], %[[FUSION_TRANSPOSE2]]

// -----
