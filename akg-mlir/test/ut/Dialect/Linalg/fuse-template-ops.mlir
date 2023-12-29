// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../compiler/lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" --linalg-generalize-named-ops -linalg-fuse-template-ops | FileCheck %s

func.func @element_matmul_func(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>)->tensor<16x32xf32> {
  %A1 = tensor.empty() : tensor<16x8xf32>
  %A2 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                               ins(%A: tensor<16x8xf32>) outs(%A1: tensor<16x8xf32>) -> tensor<16x8xf32>

  %res = linalg.matmul ins(%A2, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  return %res: tensor<16x32xf32>
}


// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0]] : memref<?x?xf32>

// CHECK: %[[FUSION_LOAD:.+]] = fusion.load %[[MEM_ARG0]][%[[INDEX0:.+]], %[[INDEX1:.+]]] 
// CHECK: %[[FUSION_INSERT:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD]]
// CHECK: %[[FUSION_RES:.+]] = math.floor %[[FUSION_INSERT]]
// CHECK: arith.mulf %[[FUSION_RES]]

// CHECK: func @element_matmul_func


// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
module {
  func.func @add_transpose_batch_matmul(%arg0: tensor<1x4096x768xf32>, %arg1: tensor<1x4096x768xf32>, %arg2: tensor<1x768x768xf32>) -> tensor<1x4096x768xf32> {
    %1 = tensor.empty() : tensor<1x4096x768xf32>
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : tensor<1x4096x768xf32>, tensor<1x4096x768xf32>) outs(%1 : tensor<1x4096x768xf32>) {
    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
      %8 = arith.addf %arg3, %arg4 : f32
      linalg.yield %8 : f32
    } -> tensor<1x4096x768xf32>
    %3 = tensor.empty() : tensor<1x768x768xf32>
    %4 = linalg.generic {indexing_maps = [#map1, #map0], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg2 : tensor<1x768x768xf32>) outs(%3 : tensor<1x768x768xf32>) {
    ^bb0(%arg3: f32, %arg4: f32):
      linalg.yield %arg3 : f32
    } -> tensor<1x768x768xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %5 = tensor.empty() : tensor<1x4096x768xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<1x4096x768xf32>) -> tensor<1x4096x768xf32>
    %7 = linalg.batch_matmul ins(%2, %4 : tensor<1x4096x768xf32>, tensor<1x768x768xf32>) outs(%6 : tensor<1x4096x768xf32>) -> tensor<1x4096x768xf32>
    return %7 : tensor<1x4096x768xf32>
  }
}

// CHECK: func.func @template_batch_matmul_f32_f32_f32_0(%[[ARG0:.+]]: tensor<?x?x?xf32>, %[[ARG1:.+]]: tensor<1x4096x768xf32>, %[[ARG2:.+]]: tensor<?x?x?xf32>, %[[ARG3:.+]]: tensor<?x?x?xf32>
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]]

// CHECK: scf.for %[[BATCH:.+]] =
// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[BATCH]], %[[M]], %[[K]]]
// CHECK: %[[MEMREF_LOAD1:.+]] = fusion.load %[[MEM_ARG1]][%[[BATCH]], %[[M]], %[[K]]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG1]], %[[MEMREF_LOAD1]]
// CHECK: %[[FUSION_LOAD2:.+]] = fusion.load %[[MEM_ARG2]][%[[BATCH]], %[[N]], %[[K]]]
// CHECK: %[[FUSION_INSERT1:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: arith.addf %[[FUSION_INSERT1]], %[[FUSION_INSERT0]]

// CHECK: func @add_transpose_batch_matmul

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_cast_matmul(%A : tensor<16x8xf16>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>)->tensor<16x32xf32> {
  %A1 = tensor.empty() : tensor<16x8xf32>
  %A2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                       ins(%A: tensor<16x8xf16>) outs(%A1: tensor<16x8xf32>) {
   ^bb0(%arg10: f16, %arg11: f32):
         %153 = arith.extf %arg10 : f16 to f32
         linalg.yield %153 : f32
  } -> tensor<16x8xf32>

  %res = linalg.matmul ins(%A2, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  return %res: tensor<16x32xf32>
}

// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]

// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[M]], %[[K]]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: arith.extf %[[FUSION_INSERT0]]

// CHECK: func @fuse_cast_matmul
// CHECK: linalg.template
// CHECK: arith.extf

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
func.func @fuse_cast_matmul2(%A : tensor<16x8xf32>, %B: tensor<8x32xf16>, %C: tensor<16x32xf32>)->tensor<16x32xf32> {
  %B1 = tensor.empty() : tensor<8x32xf32>
  %B2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                       ins(%B: tensor<8x32xf16>) outs(%B1: tensor<8x32xf32>) {
   ^bb0(%arg10: f16, %arg11: f32):
         %153 = arith.extf %arg10 : f16 to f32
         linalg.yield %153 : f32
  } -> tensor<8x32xf32>

  %res = linalg.matmul ins(%A, %B2: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  return %res: tensor<16x32xf32>
}

// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]

// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD1:.+]] = fusion.load %[[MEM_ARG1]][%[[K]], %[[N]]]
// CHECK: %[[FUSION_INSERT1:.+]] = fusion.insert %[[MEM_ARG1]], %[[FUSION_LOAD1]]
// CHECK: arith.extf %[[FUSION_INSERT1]]

// CHECK: func @fuse_cast_matmul2
// CHECK: linalg.template
// CHECK: arith.extf

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
#func_map = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ2: tensor<?x?xf32>, %targ3: tensor<?x?xf32>, %targ4: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identity"} {
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
  func.func @fuse_cast_vector_matmul(%arg0: tensor<16x8xf16>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#func_map, #func_map, #func_map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<16x8xf16>, tensor<16x8xf16>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in0: f16, %in1: f16, %out: f32):
      %3 = arith.extf %in0 : f16 to f32
      %4 = arith.extf %in1 : f16 to f32
      %5 = arith.addf %3, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]]: tensor<?x?xf16>, %[[ARG1:.+]]: tensor<16x8xf16>
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0]] : memref<?x?xf16>
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[INDEX0:.+]], %[[INDEX1:.+]]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0:.+]], %[[FUSION_LOAD0:.+]]
// CHECK: arith.extf %[[FUSION_INSERT0]]

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
#func_map00 = affine_map<(d0, d1) -> (d1)>
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
  func.func @fuse_broadcast_vector_matmul(%arg0: tensor<8xf16>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#func_map00, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8xf16>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0:.+]], %[[FUSION_LOAD0:.+]]
// CHECK: %[[FUSION_BROADCAST0:.+]] = fusion.broadcast %[[FUSION_INSERT0]]
// CHECK: arith.extf %[[FUSION_BROADCAST0]]

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
#func_map00 = affine_map<(d0, d1) -> (d1)>
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

  func.func @fuse_multi_src_broadcast_vector_matmul(%arg0: tensor<8xf16>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#func_map00, #func_map00, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<8xf16>, tensor<8xf16>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in0: f16, %in1: f16, %out: f32):
      %3 = arith.extf %in0 : f16 to f32
      %4 = arith.extf %in1 : f16 to f32
      %5 = arith.addf %3, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]]: tensor<?xf16>,
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0]] : memref<?xf16>
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[INDEX0:.+]]], %[[CONST:.+]] {in_bounds = [false, true]} : memref<?xf16>, vector<1xf16>
// CHECK: %[[FUSION_MULTILOAD0:.+]] = fusion.multi_load %[[MEM_ARG0]][%[[INDEX0]]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: %[[FUSION_BROADCAST:.+]] = fusion.broadcast %[[FUSION_INSERT0]]
// CHECK: arith.extf %[[FUSION_BROADCAST]]

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
  func.func @fuse_permutation_vector_matmul(%arg0: tensor<8x16xf16>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#func_map00, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<8x16xf16>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in: f16, %out: f32):
      %3 = arith.extf %in : f16 to f32
      linalg.yield %3 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0:.+]], %[[FUSION_LOAD0:.+]]
// CHECK: %[[FUSION_TRANSPOSE0:.+]] = fusion.transpose %[[FUSION_INSERT0]]
// CHECK: arith.extf %[[FUSION_TRANSPOSE0]]

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

  func.func @fuse_multi_src_permutation_vector_matmul(%arg0: tensor<8x16xf16>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#func_map00, #func_map00, #func_map01], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg0 : tensor<8x16xf16>, tensor<8x16xf16>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in0: f16, %in1: f16, %out: f32):
      %3 = arith.extf %in0 : f16 to f32
      %4 = arith.extf %in1 : f16 to f32
      %5 = arith.addf %3, %4 : f32
      linalg.yield %5 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]]: tensor<?x?xf16>,
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0]] : memref<?x?xf16>
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0:.+]]
// CHECK: %[[FUSION_TRANSPOSE0:.+]] = fusion.transpose %[[FUSION_INSERT0]]
// CHECK: arith.extf %[[FUSION_TRANSPOSE0]]

// -----

func.func @matmul_element_func(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>)->tensor<16x32xf32> {
  %C1 = tensor.empty() : tensor<16x32xf32>
  %res = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  %C2 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                               ins(%res: tensor<16x32xf32>) outs(%C1: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %C2: tensor<16x32xf32>
}

// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<?x?xf32>
// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FLOOR_RES:.+]] = math.floor %[[DATA0:.+]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG2]], %[[FLOOR_RES]]
// CHECK: fusion.store %[[FUSION_INSERT1:.+]], %[[MEM_ARG2]][%[[M]], %[[N]]]

// -----

func.func @matmul_element_binary_func(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>, %D: tensor<16x32xf32>)->tensor<16x32xf32> {
  %C1 = tensor.empty() : tensor<16x32xf32>
  %res = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  %C2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                               ins(%res, %D: tensor<16x32xf32>, tensor<16x32xf32>) outs(%C1: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %C2: tensor<16x32xf32>
}

// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<16x32xf32>, %[[ARG3:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG3:.+]] = bufferization.to_memref %[[ARG3]] : memref<?x?xf32>
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<16x32xf32>
// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[DATA0:.+]] = arith.addf
// CHECK: %[[LOAD_DATA1:.+]] = fusion.load %[[MEM_ARG2]][%[[INDEX0:.+]], %[[INDEX1:.+]]] : memref<16x32xf32>, f32
// CHECK: %[[DATA1:.+]] = fusion.insert %[[MEM_ARG2]], %[[LOAD_DATA1]]
// CHECK: %[[FLOOR_RES:.+]] = arith.mulf %[[DATA0]], %[[DATA1]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG3]], %[[FLOOR_RES]]
// CHECK: fusion.store %[[FUSION_INSERT1:.+]], %[[MEM_ARG3]][%[[M]], %[[N]]]

// -----

func.func @matmul_element_binary_func2(%A : tensor<16x8xf32>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>, %D: tensor<16x32xf32>)->tensor<16x32xf32> {
  %C1 = tensor.empty() : tensor<16x32xf32>
  %res = linalg.matmul ins(%A, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                       outs(%C: tensor<16x32xf32>)-> tensor<16x32xf32>
  %C2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                               ins(%D, %res: tensor<16x32xf32>, tensor<16x32xf32>) outs(%C1: tensor<16x32xf32>) -> tensor<16x32xf32>
  return %C2: tensor<16x32xf32>
}

// CHECK: func.func @template_matmul_f32_f32_f32_0(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<16x32xf32>, %[[ARG3:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG3:.+]] = bufferization.to_memref %[[ARG3]] : memref<?x?xf32>
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<16x32xf32>
// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[DATA0:.+]] = arith.addf
// CHECK: %[[LOAD_DATA1:.+]] = fusion.load %[[MEM_ARG2]][%[[INDEX0:.+]], %[[INDEX1:.+]]] : memref<16x32xf32>, f32
// CHECK: %[[DATA1:.+]] = fusion.insert %[[MEM_ARG2]], %[[LOAD_DATA1]]
// CHECK: %[[FLOOR_RES:.+]] = arith.mulf %[[DATA1]], %[[DATA0]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG3]], %[[FLOOR_RES]]
// CHECK: fusion.store %[[FUSION_INSERT1:.+]], %[[MEM_ARG3]][%[[M]], %[[N]]]

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
#func_map = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ2: tensor<?x?xf32>, %targ3: tensor<?x?xf32>, %targ4: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identity"} {
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
    // %dim = memref.dim %arg4, %c0 : memref<?x?xf32>
    // %dim_1 = memref.dim %arg4, %c1 : memref<?x?xf32>
    scf.for %arg5 = %c0 to %m step %c8 {
      %0 = affine.min #map1(%arg5)[%m]
      scf.for %arg6 = %c0 to %n step %c16 {
        %1 = affine.min #map2(%arg6)[%n]
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
              //%subview_8 = memref.subview %subview_7[%arg8, %arg9] [%3, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              //%7 = vector.transfer_read %subview_8[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
              %index_m = arith.addi %arg5, %arg8 : index
              %index_n = arith.addi %arg7, %arg9 : index
              %7 = vector.transfer_read %arg4[%index_m, %index_n], %cst : memref<?x?xf32>, vector<8x8xf32>
              %8 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %7) -> (vector<8x8xf32>) {
                %9 = vector.transfer_read %alloc[%4, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<256x256x1x8xf32>, vector<1x8xf32>
                %10 = vector.transfer_read %alloc_5[%6, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<32x256x1x8xf32>, vector<1x8xf32>
                %11 = vector.contract {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %arg11 : vector<1x8xf32>, vector<1x8xf32> into vector<8x8xf32>
                %index_k = arith.addi %arg6, %arg10 : index
                %last = arith.subi %k, %c1 : index
                %is_last = arith.cmpi sge, %index_k, %last : index
                %12 = scf.if %is_last -> vector<8x8xf32> {
                  fusion.multi_load %arg4[%index_m, %index_n] : memref<?x?xf32>, vector<8x8xf32>
                  %111 = fusion.insert %arg4, %11 : memref<?x?xf32>, vector<8x8xf32> to vector<8x8xf32>
                  scf.yield %111 : vector<8x8xf32>
                } else {
                  scf.yield %11 : vector<8x8xf32>
                }
                scf.yield %12 : vector<8x8xf32>
              }
              //vector.transfer_write %8, %subview_8[%c0, %c0] : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              fusion.store %8, %arg4[%index_m, %index_n] : vector<8x8xf32>, memref<?x?xf32>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x256x1x8xf32>
    memref.dealloc %alloc_5 : memref<32x256x1x8xf32>
    return
  }
  func.func @fuse_matmul_element_vec(%arg0: tensor<16x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%0 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
        ^bb0(%in: f32, %in_0: f32, %init: f32):
          %3 = arith.mulf %in, %in_0 : f32
          %4 = arith.addf %init, %3 : f32
          linalg.yield %4 : f32
        } -> tensor<16x32xf32>
    %2 = linalg.elemwise_unary {fun = #linalg.unary_fn<floor>}
                                 ins(%1: tensor<16x32xf32>) outs(%arg2: tensor<16x32xf32>) -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<?x?xf32>
// CHECK: %[[DATA:.+]] = vector.transfer_read %[[MEM_ARG2]][%[[M:.+]], %[[N:.+]]]
// CHECK: %[[DATA0:.+]] = vector.contract
// CHECK: %[[FLOOR_RES:.+]] = math.floor %[[DATA0]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG2]], %[[FLOOR_RES]]
// CHECK: fusion.store %[[FUSION_INSERT:.+]], %[[MEM_ARG2]][%[[M]], %[[N]]]

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
#func_map = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ2: tensor<?x?xf32>, %targ3: tensor<?x?xf32>, %targ4: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="identity"} {
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
    // %dim = memref.dim %arg4, %c0 : memref<?x?xf32>
    // %dim_1 = memref.dim %arg4, %c1 : memref<?x?xf32>
    scf.for %arg5 = %c0 to %m step %c8 {
      %0 = affine.min #map1(%arg5)[%m]
      scf.for %arg6 = %c0 to %n step %c16 {
        %1 = affine.min #map2(%arg6)[%n]
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
              //%subview_8 = memref.subview %subview_7[%arg8, %arg9] [%3, %5] [1, 1] : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[?, 1], offset: ?>>
              //%7 = vector.transfer_read %subview_8[%c0, %c0], %cst : memref<?x?xf32, strided<[?, 1], offset: ?>>, vector<8x8xf32>
              %index_m = arith.addi %arg5, %arg8 : index
              %index_n = arith.addi %arg7, %arg9 : index
              %7 = vector.transfer_read %arg4[%index_m, %index_n], %cst : memref<?x?xf32>, vector<8x8xf32>
              %8 = scf.for %arg10 = %c0 to %1 step %c1 iter_args(%arg11 = %7) -> (vector<8x8xf32>) {
                %9 = vector.transfer_read %alloc[%4, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<256x256x1x8xf32>, vector<1x8xf32>
                %10 = vector.transfer_read %alloc_5[%6, %arg10, %c0, %c0], %cst {in_bounds = [true, true]} : memref<32x256x1x8xf32>, vector<1x8xf32>
                %11 = vector.contract {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %arg11 : vector<1x8xf32>, vector<1x8xf32> into vector<8x8xf32>
                %index_k = arith.addi %arg6, %arg10 : index
                %last = arith.subi %k, %c1 : index
                %is_last = arith.cmpi sge, %index_k, %last : index
                %12 = scf.if %is_last -> vector<8x8xf32> {
                  fusion.multi_load %arg4[%index_m, %index_n] : memref<?x?xf32>, vector<8x8xf32>
                  %111 = fusion.insert %arg4, %11 : memref<?x?xf32>, vector<8x8xf32> to vector<8x8xf32>
                  scf.yield %111 : vector<8x8xf32>
                } else {
                  scf.yield %11 : vector<8x8xf32>
                }
                scf.yield %12 : vector<8x8xf32>
              }
              //vector.transfer_write %8, %subview_8[%c0, %c0] : vector<8x8xf32>, memref<?x?xf32, strided<[?, 1], offset: ?>>
              fusion.store %8, %arg4[%index_m, %index_n] : vector<8x8xf32>, memref<?x?xf32>
            }
          }
        }
      }
    }
    memref.dealloc %alloc : memref<256x256x1x8xf32>
    memref.dealloc %alloc_5 : memref<32x256x1x8xf32>
    return
  }
  func.func @fuse_matmul_element_vec(%arg0: tensor<16x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>, %arg3: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%0 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
        ^bb0(%in: f32, %in_0: f32, %init: f32):
          %3 = arith.mulf %in, %in_0 : f32
          %4 = arith.addf %init, %3 : f32
          linalg.yield %4 : f32
        } -> tensor<16x32xf32>
    %2 = linalg.elemwise_binary {fun = #linalg.binary_fn<mul>}
                               ins(%1, %arg3: tensor<16x32xf32>, tensor<16x32xf32>) outs(%arg2: tensor<16x32xf32>) -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]]: tensor<?x?xf32>, %[[ARG1:.+]]: tensor<?x?xf32>, %[[ARG2:.+]]: tensor<16x32xf32>, %[[ARG3:.+]]: tensor<?x?xf32>
// CHECK: %[[MEM_ARG3:.+]] = bufferization.to_memref %[[ARG3]] : memref<?x?xf32>
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<16x32xf32>
// CHECK: %[[DATA:.+]] = vector.transfer_read %[[MEM_ARG3]][%[[M:.+]], %[[N:.+]]]
// CHECK: %[[DATA0:.+]] = vector.contract
// CHECK: %[[LOAD_DATA1:.+]] = fusion.load %[[MEM_ARG2]][%[[M]], %[[N]]]
// CHECK: %[[DATA1:.+]] = fusion.insert %[[MEM_ARG2]], %[[LOAD_DATA1]]
// CHECK: %[[MUL_RES:.+]] = arith.mulf %[[DATA0]], %[[DATA1]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG3]], %[[MUL_RES]]
// CHECK: fusion.store %[[FUSION_INSERT:.+]], %[[MEM_ARG3]][%[[M]], %[[N]]]

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @template_matmul_f32_f32_f32_0(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32> {bufferization.access = "write"}, %arg3: index, %arg4: index, %arg5: index) attributes {fusion.kind = "invertible"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %1 = bufferization.to_memref %arg1 : memref<?x?xf32>
    %2 = bufferization.to_memref %arg2 : memref<?x?xf32>
    scf.for %arg6 = %c0 to %arg3 step %c1 {
      scf.for %arg7 = %c0 to %arg4 step %c1 {
        scf.for %arg8 = %c0 to %arg5 step %c1 {
          %3 = fusion.load %0[%arg6, %arg8] : memref<?x?xf32>, f32
          %4 = fusion.multi_load %0[%arg6, %arg8] : memref<?x?xf32>, f32
          %5 = fusion.load %1[%arg8, %arg7] : memref<?x?xf32>, f32
          %6 = fusion.multi_load %1[%arg8, %arg7] : memref<?x?xf32>, f32
          %7 = memref.load %2[%arg6, %arg7] : memref<?x?xf32>
          %8 = fusion.insert %0, %3 : memref<?x?xf32>, f32 to f32
          %9 = fusion.insert %1, %5 : memref<?x?xf32>, f32 to f32
          %10 = arith.mulf %8, %9 : f32
          %11 = arith.addf %7, %10 : f32
          %12 = arith.subi %arg5, %c1 : index
          %13 = arith.cmpi eq, %arg8, %12 : index
          %14 = scf.if %13 -> (f32) {
            %15 = fusion.multi_load %2[%arg6, %arg7] : memref<?x?xf32>, f32
            %16 = fusion.insert %2, %11 : memref<?x?xf32>, f32 to f32
            scf.yield %16 : f32
          } else {
            scf.yield %11 : f32
          }
          fusion.store %14, %2[%arg6, %arg7] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @unfusionable_func(%arg0: tensor<16x8xf32>, %arg1: tensor<8x32xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x8xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"], fusion.flag = "unfusionable"} ins(%arg0 : tensor<16x8xf32>) outs(%0 : tensor<16x8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = math.floor %in : f32
      linalg.yield %3 : f32
    } -> tensor<16x8xf32>
    %2 = linalg.template {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%1, %arg1 : tensor<16x8xf32>, tensor<8x32xf32>) outs(%arg2 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32_0} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %2 : tensor<16x32xf32>
  }
}

// CHECK: linalg.generic
// CHECK: linalg.template

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @template_matmul_f16_f16_f16(%arg0: tensor<?x?xf16>, %arg1: tensor<?x?xf16>, %arg2: tensor<?x?xf16> {bufferization.access = "write"}, %arg3: index, %arg4: index, %arg5: index) attributes {fusion.kind = "invertible"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg0 : memref<?x?xf16>
    %1 = bufferization.to_memref %arg1 : memref<?x?xf16>
    %2 = bufferization.to_memref %arg2 : memref<?x?xf16>
    scf.for %arg6 = %c0 to %arg3 step %c1 {
      scf.for %arg7 = %c0 to %arg4 step %c1 {
        scf.for %arg8 = %c0 to %arg5 step %c1 {
          %3 = fusion.load %0[%arg6, %arg8] : memref<?x?xf16>, f16
          %4 = fusion.multi_load %0[%arg6, %arg8] : memref<?x?xf16>, f16
          %5 = memref.load %1[%arg8, %arg7] : memref<?x?xf16>
          %6 = memref.load %2[%arg6, %arg7] : memref<?x?xf16>
          %7 = fusion.insert %0, %3 : memref<?x?xf16>, f16 to f16
          %8 = arith.mulf %7, %5 : f16
          %9 = arith.addf %6, %8 : f16
          fusion.multi_load %2[%arg6, %arg7] : memref<?x?xf16>, f16
          %10 = fusion.insert %2, %9 : memref<?x?xf16>, f16 to f16
          fusion.store %10, %2[%arg6, %arg7] : f16, memref<?x?xf16>
        }
      }
    }
    return
  }
  func.func @fuse_scalar_template_and_generic_with_extern_cst(%arg0: tensor<4096x512xf16>, %arg1: tensor<512x4096xf16>, %arg2: tensor<4096x4096xf16>) -> tensor<4096x4096xf16> {
    %cst = arith.constant 0.000000e+00 : f16
    %0 = tensor.empty() : tensor<4096x4096xf16>
    %1 = linalg.template {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4096x512xf16>, tensor<512x4096xf16>) outs(%0 : tensor<4096x4096xf16>) attrs =  {template_func = @template_matmul_f16_f16_f16} {
    ^bb0(%in: f16, %in_0: f16, %init: f16):
      %3 = arith.mulf %in, %in_0 : f16
      %4 = arith.addf %init, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<4096x4096xf16>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<4096x4096xf16>) outs(%0 : tensor<4096x4096xf16>) {
    ^bb0(%in: f16, %out: f16):
      %3 = arith.cmpf ogt, %in, %cst : f16
      %4 = arith.select %3, %in, %cst : f16
      linalg.yield %4 : f16
    } -> tensor<4096x4096xf16>
    return %2 : tensor<4096x4096xf16>
  }
}

// CHECK: func.func @template_matmul_f16_f16_f16(%[[ARG0:.+]]: tensor<?x?xf16>, %[[ARG1:.+]]: tensor<?x?xf16>, %[[ARG2:.+]]: tensor<?x?xf16>
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2]] : memref<?x?xf16>
// CHECK: %[[DATA0:.+]] = arith.addf
// CHECK: %[[DATA1:.+]] = arith.cmpf ogt, %[[DATA0]], %[[CST]]
// CHECK: %[[DATA2:.+]] = arith.select %[[DATA1]], %[[DATA0]], %[[CST]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG2]], %[[DATA2]]
// CHECK: fusion.store %[[FUSION_INSERT:.+]], %[[MEM_ARG2]]

// -----

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @template_matmul_f32_f32_f32(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>, %arg2: tensor<?x?xf32> {bufferization.access = "write"}, %arg3: index, %arg4: index, %arg5: index) attributes {fusion.kind = "invertible"} {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_memref %arg0 : memref<?x?xf32>
    %1 = bufferization.to_memref %arg1 : memref<?x?xf32>
    %2 = bufferization.to_memref %arg2 : memref<?x?xf32>
    scf.for %arg6 = %c0 to %arg3 step %c1 {
      scf.for %arg7 = %c0 to %arg4 step %c1 {
        scf.for %arg8 = %c0 to %arg5 step %c1 {
          %3 = fusion.load %0[%arg6, %arg8] : memref<?x?xf32>, f32
          %4 = fusion.multi_load %0[%arg6, %arg8] : memref<?x?xf32>, f32
          %5 = memref.load %1[%arg8, %arg7] : memref<?x?xf32>
          %6 = memref.load %2[%arg6, %arg7] : memref<?x?xf32>
          %7 = fusion.insert %0, %3 : memref<?x?xf32>, f32 to f32
          %8 = arith.mulf %7, %5 : f32
          %9 = arith.addf %6, %8 : f32
          fusion.multi_load %2[%arg6, %arg7] : memref<?x?xf32>, f32
          %10 = fusion.insert %2, %9 : memref<?x?xf32>, f32 to f32
          fusion.store %10, %2[%arg6, %arg7] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }
  func.func @unfuse_scalar_template_and_generic_with_extern_var(%arg0: tensor<4096x512xf32>, %arg1: tensor<512x4096xf32>, %arg2: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    %cst = arith.constant 0.000000e+00 : f16
    %cst_0 = arith.extf %cst : f16 to f32
    %0 = tensor.empty() : tensor<4096x4096xf32>
    %1 = linalg.template {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4096x512xf32>, tensor<512x4096xf32>) outs(%0 : tensor<4096x4096xf32>) attrs =  {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<4096x4096xf32>
    %2 = linalg.generic {indexing_maps = [#map3, #map3], iterator_types = ["parallel", "parallel"]} ins(%1 : tensor<4096x4096xf32>) outs(%0 : tensor<4096x4096xf32>) {
    ^bb0(%in: f32, %out: f32):
      %3 = arith.cmpf ogt, %in, %cst_0 : f32
      %4 = arith.select %3, %in, %cst_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4096x4096xf32>
    return %2 : tensor<4096x4096xf32>
  }
}

// CHECK: linalg.template
// CHECK: linalg.generic

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @unfuse_vector_template_and_generic_with_cst(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
  %res = tensor.empty() : tensor<16x32xf16>

  %1 = linalg.matmul_transpose_b ins(%A, %B: tensor<16x8xf16>, tensor<32x8xf16>)
                       outs(%res: tensor<16x32xf16>)-> tensor<16x32xf16>
  %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                       ins(%1: tensor<16x32xf16>) outs(%C: tensor<16x32xf16>) {
   ^bb0(%arg10: f16, %arg11: f16):
         %cst = arith.constant 1.000000e+00 : f16
         %153 = arith.addf %arg10, %cst : f16
         linalg.yield %153 : f16
  } -> tensor<16x32xf16>
  return %2: tensor<16x32xf16>
}

// CHECK: linalg.template
// CHECK: linalg.generic

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>

func.func @unfuse_vector_template_and_generic_with_extern_cst(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
  %res = tensor.empty() : tensor<16x32xf16>
  %cst = arith.constant 1.000000e+00 : f16
  %1 = linalg.matmul_transpose_b ins(%A, %B: tensor<16x8xf16>, tensor<32x8xf16>)
                       outs(%res: tensor<16x32xf16>)-> tensor<16x32xf16>
  %2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                       ins(%1: tensor<16x32xf16>) outs(%C: tensor<16x32xf16>) {
   ^bb0(%arg10: f16, %arg11: f16):
         %153 = arith.addf %arg10, %cst : f16
         linalg.yield %153 : f16
  } -> tensor<16x32xf16>
  return %2: tensor<16x32xf16>
}

// CHECK: linalg.template
// CHECK: linalg.generic