// RUN: akg-opt %s -split-input-file -linalg-fuse-template-ops | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ0: tensor<?x?xf32>, %targ1: tensor<?x?xf32>, %targ2: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5], %cst : memref<?x?xf32>, f32
          fusion.multi_load %arg0[%arg3, %arg5], %cst : memref<?x?xf32>, f32
          %4 = fusion.load %arg1[%arg5, %arg4], %cst : memref<?x?xf32>, f32
          fusion.multi_load %arg1[%arg5, %arg4], %cst : memref<?x?xf32>, f32
          %5 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %30 = fusion.insert %arg0, %3 : memref<?x?xf32>, f32 to f32
          %40 = fusion.insert %arg1, %4 : memref<?x?xf32>, f32 to f32
          %6 = arith.mulf %30, %40 : f32
          %7 = arith.addf %5, %6 : f32
          %last = arith.subi %k, %c1 : index
          %is_last = arith.cmpi eq, %arg5, %last : index
          %700 = scf.if %is_last -> f32 {
            fusion.multi_load %arg2[%arg3, %arg4] : memref<?x?xf32>, f32
            %70 = fusion.insert %arg2, %7 : memref<?x?xf32>, f32 to f32
            scf.yield %70 : f32
          } else {
            scf.yield %7 : f32
          }
          fusion.store %700, %arg2[%arg3, %arg4] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }

  func.func @fuse_cast_matmul(%A : tensor<16x8xf16>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>)->tensor<16x32xf32> {
    %A1 = tensor.empty() : tensor<16x8xf32>
    %A2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                         ins(%A: tensor<16x8xf16>) outs(%A1: tensor<16x8xf32>) {
     ^bb0(%arg10: f16, %arg11: f32):
           %153 = arith.extf %arg10 : f16 to f32
           linalg.yield %153 : f32
    } -> tensor<16x8xf32>
  
    %res = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A2, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                         outs(%C: tensor<16x32xf32>) attrs = {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %res: tensor<16x32xf32>
  }
}
// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]

// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_MULTI_LOAD0:.+]] = fusion.multi_load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: arith.extf %[[FUSION_INSERT0]]

// CHECK: func @fuse_cast_matmul
// CHECK: linalg.template
// CHECK: arith.extf

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f16_f16_f16(%targ0: tensor<?x?xf16>, %targ1: tensor<?x?xf16>, %targ2: tensor<?x?xf16> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf16>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf16>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf16>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f16
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5], %cst : memref<?x?xf16>, f16
          fusion.multi_load %arg0[%arg3, %arg5], %cst : memref<?x?xf16>, f16
          %4 = fusion.load %arg1[%arg5, %arg4], %cst : memref<?x?xf16>, f16
          fusion.multi_load %arg1[%arg5, %arg4], %cst : memref<?x?xf16>, f16
          %5 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf16>
          %30 = fusion.insert %arg0, %3 : memref<?x?xf16>, f16 to f16
          %40 = fusion.insert %arg1, %4 : memref<?x?xf16>, f16 to f16
          %6 = arith.mulf %30, %40 : f16
          %7 = arith.addf %5, %6 : f16
          %last = arith.subi %k, %c1 : index
          %is_last = arith.cmpi eq, %arg5, %last : index
          %700 = scf.if %is_last -> f16 {
            fusion.multi_load %arg2[%arg3, %arg4] : memref<?x?xf16>, f16
            %70 = fusion.insert %arg2, %7 : memref<?x?xf16>, f16 to f16
            scf.yield %70 : f16
          } else {
            scf.yield %7 : f16
          }
          fusion.store %700, %arg2[%arg3, %arg4] : f16, memref<?x?xf16>
        }
      }
    }
    return
  }

  func.func @fuse_cast_matmul_2(%A : tensor<16x8xf32>, %B: tensor<8x32xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
    %A1 = tensor.empty() : tensor<16x8xf16>
    %A2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel"]}
                         ins(%A: tensor<16x8xf32>) outs(%A1: tensor<16x8xf16>) {
     ^bb0(%arg10: f32, %arg11: f16):
           %153 = arith.truncf %arg10 : f32 to f16
           linalg.yield %153 : f16
    } -> tensor<16x8xf16>
  
    %res = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A2, %B: tensor<16x8xf16>, tensor<8x32xf16>)
                         outs(%C: tensor<16x32xf16>) attrs = {template_func = @template_matmul_f16_f16_f16} {
    ^bb0(%in: f16, %in_0: f16, %init: f16):
      %3 = arith.mulf %in, %in_0 : f16
      %4 = arith.addf %init, %3 : f16
      linalg.yield %4 : f16
    } -> tensor<16x32xf16>
    return %res: tensor<16x32xf16>
  }
}
// CHECK: func.func @template_matmul_f16_f16_f16(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]

// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[CST_CAST0:.+]] = arith.extf %[[CST]] : f16 to f32
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST_CAST0]]
// CHECK: %[[CST_CAST1:.+]] = arith.extf %[[CST]] : f16 to f32
// CHECK: %[[FUSION_MULTI_LOAD:.+]] = fusion.multi_load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST_CAST1]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: arith.truncf %[[FUSION_INSERT0]]

// CHECK: func @fuse_cast_matmul_2
// CHECK: linalg.template
// CHECK: arith.truncf

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

module {
  func.func @template_matmul_f32_f32_f32(%targ0: tensor<?x?xf32>, %targ1: tensor<?x?xf32>, %targ2: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5], %cst : memref<?x?xf32>, f32
          fusion.multi_load %arg0[%arg3, %arg5], %cst : memref<?x?xf32>, f32
          %4 = fusion.load %arg1[%arg5, %arg4], %cst : memref<?x?xf32>, f32
          fusion.multi_load %arg1[%arg5, %arg4], %cst : memref<?x?xf32>, f32
          %5 = memref.load %arg2[%arg3, %arg4] : memref<?x?xf32>
          %30 = fusion.insert %arg0, %3 : memref<?x?xf32>, f32 to f32
          %40 = fusion.insert %arg1, %4 : memref<?x?xf32>, f32 to f32
          %6 = arith.mulf %30, %40 : f32
          %7 = arith.addf %5, %6 : f32
          %last = arith.subi %k, %c1 : index
          %is_last = arith.cmpi eq, %arg5, %last : index
          %700 = scf.if %is_last -> f32 {
            fusion.multi_load %arg2[%arg3, %arg4] : memref<?x?xf32>, f32
            %70 = fusion.insert %arg2, %7 : memref<?x?xf32>, f32 to f32
            scf.yield %70 : f32
          } else {
            scf.yield %7 : f32
          }
          fusion.store %700, %arg2[%arg3, %arg4] : f32, memref<?x?xf32>
        }
      }
    }
    return
  }

  func.func @fuse_cast_matmul_3(%A : tensor<16x8xf16>, %B: tensor<8x32xf32>, %C: tensor<16x32xf32>, %D: tensor<16x8xf16>)->tensor<16x32xf32> {
    %A1 = tensor.empty() : tensor<16x8xf32>
    %A2 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types=["parallel", "parallel"]}
                         ins(%A, %D: tensor<16x8xf16>, tensor<16x8xf16>) outs(%A1: tensor<16x8xf32>) {
     ^bb0(%arg10: f16, %arg11: f16, %arg12: f32):
           %153 = arith.extf %arg10 : f16 to f32
           %154 = arith.extf %arg11 : f16 to f32
           %155 = arith.addf %153, %154 : f32
           linalg.yield %155 : f32
    } -> tensor<16x8xf32>
  
    %res = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%A2, %B: tensor<16x8xf32>, tensor<8x32xf32>)
                         outs(%C: tensor<16x32xf32>) attrs = {template_func = @template_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x32xf32>
    return %res: tensor<16x32xf32>
  }
}
// CHECK: func.func @template_matmul_f32_f32_f32(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]
// CHECK: %[[MEM_ARG3:.+]] = bufferization.to_memref %[[ARG3:.+]]

// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_MULTI_LOAD0:.+]] = fusion.multi_load %[[MEM_ARG0]][%[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_LOAD1:.+]] = fusion.load %[[MEM_ARG1]][%[[M]], %[[K]]]
// CHECK: %[[FUSION_INSERT1:.+]] = fusion.insert %[[MEM_ARG1]], %[[FUSION_LOAD1]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: %[[EXT0:.+]] = arith.extf %[[FUSION_INSERT0]]
// CHECK: %[[EXT1:.+]] = arith.extf %[[FUSION_INSERT1]]
// CHECK: arith.addf %[[EXT0:.+]], %[[EXT1:.+]]

// CHECK: func @fuse_cast_matmul_3
// CHECK: linalg.template
// CHECK: arith.extf
// CHECK: arith.extf

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#func_map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#func_map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#func_map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

module {
  func.func @template_batch_matmul_f32_f32_f32(%targ0: tensor<?x?x?xf32>, %targ1: tensor<?x?x?xf32>, %targ2: tensor<?x?x?xf32> {bufferization.access = "write"}, %batch: index, %m: index, %n:index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.for %arg3 = %c0 to %batch step %c1 {
      scf.for %arg4 = %c0 to %m step %c1 {
        scf.for %arg5 = %c0 to %n step %c1 {
          scf.for %arg6 = %c0 to %k step %c1 {
            %4 = fusion.load %arg0[%arg3, %arg4, %arg6], %cst : memref<?x?x?xf32>, f32
            fusion.multi_load %arg0[%arg3, %arg4, %arg6], %cst : memref<?x?x?xf32>, f32
            %5 = fusion.load %arg1[%arg3, %arg6, %arg5], %cst : memref<?x?x?xf32>, f32
            fusion.multi_load %arg1[%arg3, %arg6, %arg5], %cst : memref<?x?x?xf32>, f32
            %6 = memref.load %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
            %40 = fusion.insert %arg0, %4 : memref<?x?x?xf32>, f32 to f32
            %50 = fusion.insert %arg1, %5 : memref<?x?x?xf32>, f32 to f32
            %7 = arith.mulf %40, %50 : f32
            %8 = arith.addf %6, %7 : f32
            memref.store %8, %arg2[%arg3, %arg4, %arg5] : memref<?x?x?xf32>
          }
        }
      }
    }
    return
  }

  func.func @fuse_cast_batch_matmul(%A : tensor<16x16x8xf16>, %B: tensor<16x8x32xf32>, %C: tensor<16x16x32xf32>)->tensor<16x16x32xf32> {
    %A1 = tensor.empty() : tensor<16x16x8xf32>
    %A2 = linalg.generic {indexing_maps = [#map0, #map0], iterator_types=["parallel", "parallel", "parallel"]}
                         ins(%A: tensor<16x16x8xf16>) outs(%A1: tensor<16x16x8xf32>) {
     ^bb0(%arg10: f16, %arg11: f32):
           %153 = arith.extf %arg10 : f16 to f32
           linalg.yield %153 : f32
    } -> tensor<16x16x8xf32>
  
    %res = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%A2, %B: tensor<16x16x8xf32>, tensor<16x8x32xf32>)
                         outs(%C: tensor<16x16x32xf32>) attrs = {template_func = @template_batch_matmul_f32_f32_f32} {
    ^bb0(%in: f32, %in_0: f32, %init: f32):
      %3 = arith.mulf %in, %in_0 : f32
      %4 = arith.addf %init, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x16x32xf32>
    return %res: tensor<16x16x32xf32>
  }
}
// CHECK: func.func @template_batch_matmul_f32_f32_f32(%[[ARG0:.+]], %[[ARG1:.+]], %[[ARG2:.+]], %[[ARG3:.+]])
// CHECK: %[[CST:.+]] = arith.constant 0.000000e+00 : f16
// CHECK: %[[MEM_ARG0:.+]] = bufferization.to_memref %[[ARG0:.+]]
// CHECK: %[[MEM_ARG1:.+]] = bufferization.to_memref %[[ARG1:.+]]
// CHECK: %[[MEM_ARG2:.+]] = bufferization.to_memref %[[ARG2:.+]]

// CHECK: scf.for %[[BATCH:.+]] =
// CHECK: scf.for %[[M:.+]] =
// CHECK: scf.for %[[N:.+]] =
// CHECK: scf.for %[[K:.+]] =
// CHECK: %[[FUSION_LOAD0:.+]] = fusion.load %[[MEM_ARG0]][%[[BATCH]], %[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_MULTI_LOAD0:.+]] = fusion.multi_load %[[MEM_ARG0]][%[[BATCH]], %[[M]], %[[K]]], %[[CST]]
// CHECK: %[[FUSION_INSERT0:.+]] = fusion.insert %[[MEM_ARG0]], %[[FUSION_LOAD0]]
// CHECK: arith.extf %[[FUSION_INSERT0]]

// CHECK: func @fuse_cast_batch_matmul
// CHECK: linalg.template
// CHECK: arith.extf
