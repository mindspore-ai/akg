// RUN: akg-opt %s -split-input-file --linalg-template-named-ops="template-path=%S/../../../../compiler/lib/Dialect/Linalg/Transforms/TemplatedOpImpl/" --linalg-generalize-named-ops -linalg-fuse-template-ops | FileCheck %s

#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @template_matmul_transpose_b_f32_f32_f32(%targ0: tensor<?x?xf32>, %targ1: tensor<?x?xf32>, %targ2: tensor<?x?xf32>, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          fusion.multi_load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          %4 = fusion.load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
          fusion.multi_load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
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

  func.func @test_template_producer_read_output(%arg0: tensor<16x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%0 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_transpose_b_f32_f32_f32} {
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
// CHECK: linalg.template
// CHECK: linalg.generic

// -----


#func_map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#func_map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#func_map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @template_matmul_transpose_b_f32_f32_f32(%targ0: tensor<?x?xf32>, %targ1: tensor<?x?xf32>, %targ2: tensor<?x?xf32> {bufferization.access = "write"}, %m: index, %n: index, %k:index) attributes {fusion.kind="invertible"} {
    %arg0 = bufferization.to_memref %targ0 : memref<?x?xf32>
    %arg1 = bufferization.to_memref %targ1 : memref<?x?xf32>
    %arg2 = bufferization.to_memref %targ2 : memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %m step %c1 {
      scf.for %arg4 = %c0 to %n step %c1 {
        scf.for %arg5 = %c0 to %k step %c1 {
          %3 = fusion.load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          fusion.multi_load %arg0[%arg3, %arg5] : memref<?x?xf32>, f32
          %4 = fusion.load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
          fusion.multi_load %arg1[%arg5, %arg4] : memref<?x?xf32>, f32
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

  func.func @test_template_producer_writeonly_output(%arg0: tensor<16x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %0 = tensor.empty() : tensor<16x32xf32>
    %1 = linalg.template {indexing_maps = [#func_map1, #func_map2, #func_map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<16x8xf32>, tensor<32x8xf32>) outs(%0 : tensor<16x32xf32>) attrs =  {template_func = @template_matmul_transpose_b_f32_f32_f32} {
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
// CHECK: linalg.template
// CHECK-NOT: linalg.generic
// CHECK: return

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @test_generic_producer_read_output(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
    %A1 = tensor.empty() : tensor<16x8xf16>
    %A2 = tensor.empty() : tensor<16x8xf16>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%A, %A1 : tensor<16x8xf16>, tensor<16x8xf16>) outs(%A2 : tensor<16x8xf16>) {
    ^bb0(%b0 : f16, %b1 : f16, %b2 : f16):
      %2 = arith.addf %b0, %b1 : f16
      %3 = arith.addf %2, %b2 : f16
      linalg.yield %3 : f16
    } -> tensor<16x8xf16>

    %res = linalg.matmul_transpose_b ins(%1, %B: tensor<16x8xf16>, tensor<32x8xf16>)
                         outs(%C: tensor<16x32xf16>)-> tensor<16x32xf16>
    return %res: tensor<16x32xf16>
  }
}


// CHECK: linalg.generic
// CHECK: linalg.template

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @test_generic_producer_writeonly_output(%A : tensor<16x8xf16>, %B: tensor<32x8xf16>, %C: tensor<16x32xf16>)->tensor<16x32xf16> {
    %A1 = tensor.empty() : tensor<16x8xf16>
    %A2 = tensor.empty() : tensor<16x8xf16>
    %1 = linalg.generic {indexing_maps = [#map0, #map0, #map0], iterator_types = ["parallel", "parallel"]} ins(%A, %A1 : tensor<16x8xf16>, tensor<16x8xf16>) outs(%A2 : tensor<16x8xf16>) {
    ^bb0(%b0 : f16, %b1 : f16, %b2 : f16):
      %2 = arith.addf %b0, %b1 : f16
      linalg.yield %2 : f16
    } -> tensor<16x8xf16>

    %res = linalg.matmul_transpose_b ins(%1, %B: tensor<16x8xf16>, tensor<32x8xf16>)
                         outs(%C: tensor<16x32xf16>)-> tensor<16x32xf16>
    return %res: tensor<16x32xf16>
  }
}


// CHECK: test_generic_producer_writeonly_output
// CHECK-NOT: linalg.generic
// CHECK: linalg.template
