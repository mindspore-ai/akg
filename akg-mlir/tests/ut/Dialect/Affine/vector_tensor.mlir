// RUN: akg-opt %s --vector-transfer-tensorize -allow-unregistered-dialect | FileCheck %s

//CHECK-LABEL: #map = affine_map<(d0) -> (d0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 + 512)>
//CHECK-NEXT: module {
//CHECK-NEXT:   func.func @Fused_ReduceSum_split_10316266540114966563(%arg0: tensor<5100x3072xf32>) -> tensor<5100xf32> attributes {OperatorType = "Reduce", compute_capability = "", mindspore_kernel, process = "aicore"} {
//CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:     %0 = tensor.empty() : tensor<5100xf32>
//CHECK-NEXT:     affine.for %arg1 = 0 to 4608 step 512 {
//CHECK-NEXT:       affine.for %arg2 = #map(%arg1) to #map1(%arg1) {
//CHECK-NEXT:         %cst_0 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
//CHECK-NEXT:         %1 = affine.for %arg3 = 0 to 3072 step 3072 iter_args(%arg4 = %cst_0) -> (tensor<3072xf32>) {
//CHECK-NEXT:           %extracted_slice = tensor.extract_slice %arg0[%arg2, %arg3] [1, 3072] [1, 1] : tensor<5100x3072xf32> to tensor<3072xf32>
//CHECK-NEXT:           %extracted_slice_2 = tensor.extract_slice %arg0[%arg2, %arg3] [1, 3072] [1, 1] : tensor<5100x3072xf32> to tensor<3072xf32>
//CHECK-NEXT:           %4 = arith.addf %extracted_slice_2, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : tensor<3072xf32>
//CHECK-NEXT:           affine.yield %4 : tensor<3072xf32>
//CHECK-NEXT:         }
//CHECK-NEXT:         %2 = tensor.empty() : tensor<f32>
//CHECK-NEXT:         %cst_1 = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:         %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<f32>) -> tensor<f32>
//CHECK-NEXT:         %reduced = linalg.reduce ins(%1 : tensor<3072xf32>) outs(%3 : tensor<f32>) dimensions = [0] 
//CHECK-NEXT:           (%in: f32, %init: f32) {
//CHECK-NEXT:             %4 = arith.addf %in, %init : f32
//CHECK-NEXT:             linalg.yield %4 : f32
//CHECK-NEXT:           }
//CHECK-NEXT:         %expanded = tensor.expand_shape %reduced [] output_shape [1] : tensor<f32> into tensor<1xf32>
//CHECK-NEXT:         %inserted_slice = tensor.insert_slice %expanded into %0[%arg2] [1] [1] : tensor<1xf32> into tensor<5100xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:     }
//CHECK-NEXT:     affine.for %arg1 = 4608 to 5100 {
//CHECK-NEXT:       %cst_0 = arith.constant dense<0.000000e+00> : tensor<3072xf32>
//CHECK-NEXT:       %1 = affine.for %arg2 = 0 to 3072 step 3072 iter_args(%arg3 = %cst_0) -> (tensor<3072xf32>) {
//CHECK-NEXT:         %extracted_slice = tensor.extract_slice %arg0[%arg1, %arg2] [1, 3072] [1, 1] : tensor<5100x3072xf32> to tensor<3072xf32>
//CHECK-NEXT:         %extracted_slice_2 = tensor.extract_slice %arg0[%arg1, %arg2] [1, 3072] [1, 1] : tensor<5100x3072xf32> to tensor<3072xf32>
//CHECK-NEXT:         %4 = arith.addf %extracted_slice_2, %arg3 {reduction_axes = [1 : index], reduction_type = "x"} : tensor<3072xf32>
//CHECK-NEXT:         affine.yield %4 : tensor<3072xf32>
//CHECK-NEXT:       }
//CHECK-NEXT:       %2 = tensor.empty() : tensor<f32>
//CHECK-NEXT:       %cst_1 = arith.constant 0.000000e+00 : f32
//CHECK-NEXT:       %3 = linalg.fill ins(%cst_1 : f32) outs(%2 : tensor<f32>) -> tensor<f32>
//CHECK-NEXT:       %reduced = linalg.reduce ins(%1 : tensor<3072xf32>) outs(%3 : tensor<f32>) dimensions = [0] 
//CHECK-NEXT:         (%in: f32, %init: f32) {
//CHECK-NEXT:           %4 = arith.addf %in, %init : f32
//CHECK-NEXT:           linalg.yield %4 : f32
//CHECK-NEXT:         }
//CHECK-NEXT:       %expanded = tensor.expand_shape %reduced [] output_shape [1] : tensor<f32> into tensor<1xf32>
//CHECK-NEXT:       %inserted_slice = tensor.insert_slice %expanded into %0[%arg1] [1] [1] : tensor<1xf32> into tensor<5100xf32>
//CHECK-NEXT:     }
//CHECK-NEXT:     return %0 : tensor<5100xf32>
//CHECK-NEXT:   }
//CHECK-NEXT: }

func.func @Fused_ReduceSum_split_10316266540114966563(%arg0: tensor<5100x3072xf32>) -> tensor<5100xf32> attributes {OperatorType = "Reduce", compute_capability = "", mindspore_kernel, process = "aicore"} {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = bufferization.to_memref %arg0 : memref<5100x3072xf32, strided<[?, ?], offset: ?>>
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<5100xf32>
  affine.for %arg1 = 0 to 4608 step 512 {
    affine.for %arg2 = affine_map<(d0) -> (d0)>(%arg1) to affine_map<(d0) -> (d0 + 512)>(%arg1) {
      %cst_0 = arith.constant dense<0.000000e+00> : vector<3072xf32>
      %2 = affine.for %arg3 = 0 to 3072 step 3072 iter_args(%arg4 = %cst_0) -> (vector<3072xf32>) {
        %cst_1 = arith.constant 0.000000e+00 : f32
        %4 = vector.transfer_read %0[%arg2, %arg3], %cst_1 : memref<5100x3072xf32, strided<[?, ?], offset: ?>>, vector<3072xf32>
        %5 = arith.addf %4, %arg4 {reduction_axes = [1 : index], reduction_type = "x"} : vector<3072xf32>
        affine.yield %5 : vector<3072xf32>
      }
      %3 = vector.reduction <add>, %2 : vector<3072xf32> into f32
      affine.store %3, %alloc[%arg2] : memref<5100xf32>
    }
  }
  affine.for %arg1 = 4608 to 5100 {
    %cst_0 = arith.constant dense<0.000000e+00> : vector<3072xf32>
    %2 = affine.for %arg2 = 0 to 3072 step 3072 iter_args(%arg3 = %cst_0) -> (vector<3072xf32>) {
      %cst_1 = arith.constant 0.000000e+00 : f32
      %4 = vector.transfer_read %0[%arg1, %arg2], %cst_1 : memref<5100x3072xf32, strided<[?, ?], offset: ?>>, vector<3072xf32>
      %5 = arith.addf %4, %arg3 {reduction_axes = [1 : index], reduction_type = "x"} : vector<3072xf32>
      affine.yield %5 : vector<3072xf32>
    }
    %3 = vector.reduction <add>, %2 : vector<3072xf32> into f32
    affine.store %3, %alloc[%arg1] : memref<5100xf32>
  }
  %1 = bufferization.to_tensor %alloc : memref<5100xf32>
  return %1 : tensor<5100xf32>
}