#map = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<(d0)[s0] -> (-d0 + s0, 8)>
#map2 = affine_map<(d0)[s0] -> (-d0 + s0, 16)>
#map3 = affine_map<(d0)[s0] -> (-d0 + s0, 2048)>
#map4 = affine_map<(d0)[s0] -> (-d0 + s0, 256)>
#map5 = affine_map<(d0) -> (d0 ceildiv 8)>
#map6 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map7 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map8 = affine_map<(d0, d1, d2) -> (d0, d1)>
module @template_matmul_transpose_b_f32_f32_f32 {
  func.func @template_matmul_transpose_b_f32_f32_f32(%targ2: tensor<?x?xf32>, %targ3: tensor<?x?xf32>, %targ4: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identity"} {
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
    %cst_0 = arith.constant dense<0.000000e+00> : vector<8x8xf32>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<256x256x1x8xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<32x256x1x8xf32>
    scf.for %arg5 = %c0 to %m step %c2048 {                // m o
      %0 = affine.min #map3(%arg5)[%m]
      scf.for %arg6 = %c0 to %k step %c256 {               // k o
        %1 = affine.min #map4(%arg6)[%k]
        %subview = memref.subview %arg2[%arg5, %arg6] [%0, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
        scf.for %arg7 = %c0 to %0 step %c8 {               // m io
          %2 = affine.apply #map5(%arg7)
          %3 = affine.min #map1(%arg7)[%0]
          scf.for %arg8 = %c0 to %1 step %c1 {             // k io
            %subview_6 = memref.subview %subview[%arg7, %arg8] [%3, 1] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x1xf32, strided<[?, 1], offset: ?>>
            %4 = fusion.load %arg2[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x?xf32>, vector<8x1xf32>
            fusion.multi_load %arg2[%c0, %c0] : memref<?x?xf32>, vector<8x1xf32>
            //%4 = fusion.load %subview_6[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x1xf32, strided<[?, 1], offset: ?>>, vector<8x1xf32>
            %444 = fusion.insert %arg2, %4 : memref<?x?xf32>, vector<8x1xf32> to vector<8x1xf32>
            %5 = vector.transpose %444, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
            vector.transfer_write %5, %alloc[%2, %arg8, %c0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<256x256x1x8xf32>
          }
        }
        scf.for %arg7 = %c0 to %n step %c256 {             // n o
          %2 = affine.min #map4(%arg7)[%n]
          %subview_6 = memref.subview %arg3[%arg7, %arg6] [%2, %1] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          %subview_7 = memref.subview %arg4[%arg5, %arg7] [%0, %2] [1, 1] : memref<?x?xf32> to memref<?x?xf32, strided<[?, 1], offset: ?>>
          scf.for %arg8 = %c0 to %2 step %c8 {             // n io
            %3 = affine.apply #map5(%arg8)
            %4 = affine.min #map1(%arg8)[%2]
            scf.for %arg9 = %c0 to %1 step %c1 {           // k io
              %subview_8 = memref.subview %subview_6[%arg8, %arg9] [%4, 1] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x1xf32, strided<[?, 1], offset: ?>>
              %5 = vector.transfer_read %subview_8[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x1xf32, strided<[?, 1], offset: ?>>, vector<8x1xf32>
              %6 = vector.transpose %5, [1, 0] : vector<8x1xf32> to vector<1x8xf32>
              vector.transfer_write %6, %alloc_5[%3, %arg9, %c0, %c0] {in_bounds = [true, true]} : vector<1x8xf32>, memref<32x256x1x8xf32>
            }
          }
          scf.for %arg8 = %c0 to %0 step %c8 {             // m io
            %3 = affine.min #map1(%arg8)[%0]
            %4 = affine.apply #map5(%arg8)
            scf.for %arg9 = %c0 to %2 step %c8 {           // n io
              %5 = affine.min #map1(%arg9)[%2]
              %6 = affine.apply #map5(%arg9)
              %subview_8 = memref.subview %subview_7[%arg8, %arg9] [%3, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              %not_first = arith.cmpi ne, %arg6, %c0 : index
              %777 = scf.if %not_first -> (vector<8x8xf32>) {
                %7 = vector.transfer_read %subview_8[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
                scf.yield %7 : vector<8x8xf32>
              } else {
                scf.yield %cst_0 : vector<8x8xf32>
              }
              %8 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %777) -> (vector<8x8xf32>) {    // k io
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
}