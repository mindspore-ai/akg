#map = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3] -> (d0 * s1 + s0 + d1 * s2 + d2 * s3 + d3)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 2048)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 128)>
#map3 = affine_map<(d0) -> (d0 ceildiv 8)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map5 = affine_map<(d0) -> (d0 ceildiv 4)>
#map6 = affine_map<(d0)[s0] -> (-d0 + s0, 4)>
#map7 = affine_map<(d0)[s0] -> (-d0 + s0, 32)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
module {
  //func.func @batch_matmul_4d(%arg0: memref<?x?x?x?xf16, #map>, %arg1: memref<?x?x?x?xf16, #map>, %arg2: memref<?x?x?x?xf16, #map>) -> memref<?x?x?x?xf16, #map> {
  func.func @template_batch_matmul_4d_f16_f16_f16(%targ0: tensor<?x?x?x?xf16>, %targ1: tensor<?x?x?x?xf16>, %targ2: tensor<?x?x?x?xf16> {bufferization.access = "write"}, %batch_outer: index, %batch_inner:index, %m: index, %n:index, %k:index) attributes {fusion.kind="identityWithBroadPerm"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?x?x?xf16>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?x?x?xf16>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?x?x?xf16>
    %cst = arith.constant 0.000000e+00 : f16
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c128 = arith.constant 128 : index
    %c32 = arith.constant 32 : index
    %c2048 = arith.constant 2048 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    //%dim = memref.dim %arg0, %c0 : memref<?x?x?x?xf16, #map>            // batch_outer
    //%dim_0 = memref.dim %arg0, %c1 : memref<?x?x?x?xf16, #map>          // batch_inner
    //%dim_1 = memref.dim %arg0, %c2 : memref<?x?x?x?xf16, #map>          // m
    //%dim_2 = memref.dim %arg0, %c3 : memref<?x?x?x?xf16, #map>          // k
    //%dim_3 = memref.dim %arg1, %c3 : memref<?x?x?x?xf16, #map>          // n
    %cst_0 = arith.constant dense<0.000000e+00> : vector<1x1x8x8xf16>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x32x1x1x4x8xf16>
    %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<4x32x1x1x4x8xf16>
    scf.for %arg3 = %c0 to %batch_outer step %c1 {                        // batch_outer
      scf.for %arg4 = %c0 to %batch_inner step %c1 {                      // batch_inner
        scf.for %arg5 = %c0 to %m step %c2048 {                           // m o
          %0 = affine.min #map1(%arg5)[%m]
          scf.for %arg6 = %c0 to %k step %c128 {                          // k o
            %1 = affine.min #map2(%arg6)[%k]
            //%subview = memref.subview %arg0[%arg3, %arg4, %arg5, %arg6] [1, 1, %0, %1] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
            scf.for %arg7 = %c0 to %0 step %c8 {                          // m io
              %2 = affine.apply #map3(%arg7)
              %3 = affine.min #map4(%arg7)[%0]
              scf.for %arg8 = %c0 to %1 step %c4 {                        // k io
                %4 = affine.apply #map5(%arg8)
                %5 = affine.min #map6(%arg8)[%1]
                %100 = arith.addi %arg5, %arg7 : index
                %200 = arith.addi %arg6, %arg8 : index
                //%subview_5 = memref.subview %subview[0, 0, %arg7, %arg8] [1, 1, %3, %5] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                %66 = fusion.load %arg0[%c0, %c0, %100, %200], %cst {in_bounds = [true, true, false, false]} : memref<?x?x?x?xf16>, vector<1x1x8x4xf16>
                fusion.multi_load %arg0[%c0, %c0, %100, %200] : memref<?x?x?x?xf16>, vector<1x1x8x4xf16>
                //%6 = vector.transfer_read %subview_5[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, false]} : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x8x4xf16>
                %666 = fusion.insert %arg0, %66 : memref<?x?x?x?xf16>, vector<1x1x8x4xf16> to vector<1x1x8x4xf16>
                %7 = vector.transpose %666, [0, 1, 3, 2] : vector<1x1x8x4xf16> to vector<1x1x4x8xf16>
                vector.transfer_write %7, %alloc[%2, %4, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xf16>, memref<256x32x1x1x4x8xf16>
              }
            }
            scf.for %arg7 = %c0 to %n step %c32 {                        // n o
              %2 = affine.min #map7(%arg7)[%n]
              //%subview_5 = memref.subview %arg1[%arg3, %arg4, %arg6, %arg7] [1, 1, %1, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              //%subview_6 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg7] [1, 1, %0, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              %subview_5 = memref.subview %arg1[%arg3, %arg4, %arg6, %arg7] [1, 1, %1, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              %subview_6 = memref.subview %arg2[%arg3, %arg4, %arg5, %arg7] [1, 1, %0, %2] [1, 1, 1, 1] : memref<?x?x?x?xf16> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
              scf.for %arg8 = %c0 to %2 step %c8 {                        // n io
                %3 = affine.apply #map3(%arg8)
                %4 = affine.min #map4(%arg8)[%2]
                scf.for %arg9 = %c0 to %1 step %c4 {                      // k io
                  %5 = affine.apply #map5(%arg9)
                  %6 = affine.min #map6(%arg9)[%1]
                  %subview_7 = memref.subview %subview_5[0, 0, %arg9, %arg8] [1, 1, %6, %4] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                  %7 = vector.transfer_read %subview_7[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, false]} : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x4x8xf16>
                  vector.transfer_write %7, %alloc_4[%3, %5, %c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x8xf16>, memref<4x32x1x1x4x8xf16>
                }
              }
              scf.for %arg8 = %c0 to %0 step %c8 {                        // m io
                %3 = affine.min #map4(%arg8)[%0]
                %4 = affine.apply #map3(%arg8)
                scf.for %arg9 = %c0 to %2 step %c8 {                      // n io
                  %5 = affine.min #map4(%arg9)[%2]
                  %6 = affine.apply #map3(%arg9)
                  %subview_7 = memref.subview %subview_6[0, 0, %arg8, %arg9] [1, 1, %3, %5] [1, 1, 1, 1] : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>> to memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                  %not_first = arith.cmpi ne, %arg6, %c0 : index
                  %777 = scf.if %not_first -> (vector<1x1x8x8xf16>) {
                    %7 = vector.transfer_read %subview_7[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, false, false]} : memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>, vector<1x1x8x8xf16>
                    scf.yield %7 : vector<1x1x8x8xf16>
                  } else {
                    scf.yield %cst_0 : vector<1x1x8x8xf16>
                  }
                  %8 = scf.for %arg10 = %c0 to %1 step %c4 iter_args(%arg11 = %777) -> (vector<1x1x8x8xf16>) {  // k io
                    %9 = affine.apply #map5(%arg10)
                    %10 = vector.transfer_read %alloc[%4, %9, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<256x32x1x1x4x8xf16>, vector<1x1x4x8xf16>
                    %11 = vector.transfer_read %alloc_4[%6, %9, %c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<4x32x1x1x4x8xf16>, vector<1x1x4x8xf16>
                    %12 = vector.contract {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %10, %11, %arg11 : vector<1x1x4x8xf16>, vector<1x1x4x8xf16> into vector<1x1x8x8xf16>
                    scf.yield %12 : vector<1x1x8x8xf16>
                  }
                  %300 = arith.addi %arg5, %arg8 : index
                  %400 = arith.addi %arg7, %arg9 : index
                  %last = arith.subi %k, %c128 : index
                  %is_last = arith.cmpi sge, %arg6, %last : index
                  %888 = scf.if %is_last -> (vector<1x1x8x8xf16>) {
                    fusion.multi_load %arg2[%arg3, %arg4, %300, %400], %cst : memref<?x?x?x?xf16>, vector<1x1x8x8xf16>
                    %88 = fusion.insert %arg2, %8 : memref<?x?x?x?xf16>, vector<1x1x8x8xf16> to vector<1x1x8x8xf16>
                    scf.yield %88 : vector<1x1x8x8xf16>
                  } else {
                    scf.yield %8 : vector<1x1x8x8xf16>
                  }
                  fusion.store %888, %arg2[%arg3, %arg4, %300, %400] : vector<1x1x8x8xf16>, memref<?x?x?x?xf16>
                  //vector.transfer_write %8, %subview_7[%c0, %c0, %c0, %c0] {in_bounds = [true, true, false, false]} : vector<1x1x8x8xf16>, memref<1x1x?x?xf16, strided<[?, ?, ?, 1], offset: ?>>
                }
              }
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x32x1x1x4x8xf16>
    memref.dealloc %alloc_4 : memref<4x32x1x1x4x8xf16>
    //return %arg2 : memref<?x?x?x?xf16, #map>
    return
  }
}
