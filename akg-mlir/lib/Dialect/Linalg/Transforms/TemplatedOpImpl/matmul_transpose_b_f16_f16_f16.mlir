#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 2048)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map3 = affine_map<(d0) -> (d0 ceildiv 16)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map5 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @template_matmul_transpose_b_f16_f16_f16 {
  //func.func @gemm(%arg0: f16, %arg1: f16, %arg2: memref<?x?xf16, #map>, %arg3: memref<?x?xf16, #map>, %arg4: memref<?x?xf16, #map>) -> memref<?x?xf16, #map> {
  func.func @template_matmul_transpose_b_f16_f16_f16(%targ2: tensor<?x?xf16>, %targ3: tensor<?x?xf16>, %targ4: tensor<?x?xf16> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identityWithBroadPerm"} {
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf16>
    %arg3 = bufferization.to_memref %targ3 : memref<?x?xf16>
    %arg4 = bufferization.to_memref %targ4 : memref<?x?xf16>

    %cst = arith.constant 0.000000e+00 : f16
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c256 = arith.constant 256 : index
    %c2048 = arith.constant 2048 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x16xf16>
    //%dim = memref.dim %arg2, %c0 : memref<?x?xf16, #map>     // m
    //%dim_0 = memref.dim %arg2, %c1 : memref<?x?xf16, #map>   // k
    //%dim_1 = memref.dim %arg3, %c0 : memref<?x?xf16, #map>   // n

    scf.for %arg5 = %c0 to %m step %c16 {
      %0 = affine.min #map4(%arg5)[%m]
      scf.for %arg6 = %c0 to %n step %c16 {
        %1 = affine.min #map4(%arg6)[%n]
        %subview = memref.subview %arg4[%arg5, %arg6] [%0, %1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        vector.transfer_write %cst_0, %subview[%c0, %c0] : vector<16x16xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
      }
    }

    %alloc = memref.alloc() {alignment = 64 : i64} : memref<128x256x1x16xf16>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<256x1x8xf16>
    scf.for %arg5 = %c0 to %m step %c2048 {                    // m o
      %0 = affine.min #map1(%arg5)[%m]
      scf.for %arg6 = %c0 to %k step %c256 {                   // k o
        %1 = affine.min #map2(%arg6)[%k]
        //%subview = memref.subview %arg2[%arg5, %arg6] [%0, %1] [1, 1] : memref<?x?xf16, #map> to memref<?x?xf16, strided<[?, 1], offset: ?>>
        scf.for %arg7 = %c0 to %n step %c2048 {                // n o
          %2 = affine.min #map1(%arg7)[%n]
          //%subview_3 = memref.subview %arg3[%arg7, %arg6] [%2, %1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          //%subview_4 = memref.subview %arg4[%arg5, %arg7] [%0, %2] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_3 = memref.subview %arg3[%arg7, %arg6] [%2, %1] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          %subview_4 = memref.subview %arg4[%arg5, %arg7] [%0, %2] [1, 1] : memref<?x?xf16> to memref<?x?xf16, strided<[?, 1], offset: ?>>
          scf.for %arg8 = %c0 to %2 step %c16 {                // n io
            %3 = affine.apply #map3(%arg8)
            %4 = affine.min #map4(%arg8)[%2]
            scf.for %arg9 = %c0 to %1 step %c1 {               // k io
              %subview_5 = memref.subview %subview_3[%arg8, %arg9] [%4, 1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x1xf16, strided<[?, 1], offset: ?>>
              %5 = vector.transfer_read %subview_5[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x1xf16, strided<[?, 1], offset: ?>>, vector<16x1xf16>
              %6 = vector.transpose %5, [1, 0] : vector<16x1xf16> to vector<1x16xf16>
              vector.transfer_write %6, %alloc[%3, %arg9, %c0, %c0] {in_bounds = [true, true]} : vector<1x16xf16>, memref<128x256x1x16xf16>
            }
          }
          scf.for %arg8 = %c0 to %0 step %c8 {                // m io
            %3 = affine.min #map5(%arg8)[%0]
            %100 = arith.addi %arg5, %arg8 : index
            scf.for %arg9 = %c0 to %1 step %c1 {               // k io
              //%subview_5 = memref.subview %subview[%arg8, %arg9] [%3, 1] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x1xf16, strided<[?, 1], offset: ?>>
              //%4 = vector.transfer_read %subview_5[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x1xf16, strided<[?, 1], offset: ?>>, vector<8x1xf16>
              %200 = arith.addi %arg6, %arg9 : index
              %300 = fusion.load %arg2[%100, %200], %cst {in_bounds = [false, true]} : memref<?x?xf16>, vector<8x1xf16>
              fusion.multi_load %arg2[%100, %200] : memref<?x?xf16>, vector<8x1xf16>
              %400 = fusion.insert %arg2, %300 : memref<?x?xf16>, vector<8x1xf16> to vector<8x1xf16>
              %5 = vector.transpose %400, [1, 0] : vector<8x1xf16> to vector<1x8xf16>
              vector.transfer_write %5, %alloc_2[%arg9, %c0, %c0] {in_bounds = [true, true]} : vector<1x8xf16>, memref<256x1x8xf16>
            }
            scf.for %arg9 = %c0 to %2 step %c16 {                // n io
              %4 = affine.min #map4(%arg9)[%2]
              %5 = affine.apply #map3(%arg9)
              %subview_5 = memref.subview %subview_4[%arg8, %arg9] [%3, %4] [1, 1] : memref<?x?xf16, strided<[?, 1], offset: ?>> to memref<?x?xf16, strided<[?, 1], offset: ?>>
              %not_first = arith.cmpi ne, %arg6, %c0 : index
              //%666 = scf.if %not_first -> (vector<8x16xf16>) {
              %6 = vector.transfer_read %subview_5[%c0, %c0], %cst : memref<?x?xf16, strided<[?, 1], offset: ?>>, vector<8x16xf16>
              //  scf.yield %6 : vector<8x16xf16>
              //} else {
              //  scf.yield %cst_0 : vector<8x16xf16>
              //}
              %7 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %6) -> (vector<8x16xf16>) {      // k io
                %8 = vector.transfer_read %alloc_2[%arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<256x1x8xf16>, vector<1x8xf16>
                %9 = vector.transfer_read %alloc[%5, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<128x256x1x16xf16>, vector<1x16xf16>
                %10 = vector.contract {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %8, %9, %arg11 : vector<1x8xf16>, vector<1x16xf16> into vector<8x16xf16>
                scf.yield %10 : vector<8x16xf16>
              }
              %500 = arith.addi %arg7, %arg9 : index
              %last = arith.subi %k, %c256 : index
              %is_last = arith.cmpi sge, %arg6, %last : index
              %600 = scf.if %is_last -> (vector<8x16xf16>) {
                fusion.multi_load %arg4[%100, %500], %cst : memref<?x?xf16>, vector<8x16xf16>
                %700 = fusion.insert %arg4, %7 : memref<?x?xf16>, vector<8x16xf16> to vector<8x16xf16>
                scf.yield %700 : vector<8x16xf16>
              } else {
                scf.yield %7 : vector<8x16xf16>
              }
              fusion.store %600, %arg4[%100, %500] : vector<8x16xf16>, memref<?x?xf16>
              //vector.transfer_write %7, %subview_5[%c0, %c0] : vector<8x16xf16>, memref<?x?xf16, strided<[?, 1], offset: ?>>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<128x256x1x16xf16>
    memref.dealloc %alloc_2 : memref<256x1x8xf16>
    return
  }
}
