// RUN: akg-opt %s --copy-returned-func-args | FileCheck %s

// CHECK-LABEL: func @return_tensor_arg
// CHECK-SAME: (%[[ARG:.*]]: tensor<2x3xf32>)
func.func @return_tensor_arg(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
  // CHECK: %[[E:.*]] = tensor.empty() : tensor<2x3xf32>
  // CHECK: %[[C:.*]] = linalg.copy ins(%[[ARG]] : tensor<2x3xf32>) outs(%[[E]] : tensor<2x3xf32>) -> tensor<2x3xf32>
  // CHECK: return %[[C]] : tensor<2x3xf32>
  func.return %arg0 : tensor<2x3xf32>
}

// -----

// CHECK-LABEL: func @return_memref_arg
// CHECK-SAME: (%[[ARG:.*]]: memref<4xf32>)
func.func @return_memref_arg(%arg0: memref<4xf32>) -> memref<4xf32> {
  // CHECK: %[[A:.*]] = memref.alloc() : memref<4xf32>
  // CHECK: linalg.copy ins(%[[ARG]] : memref<4xf32>) outs(%[[A]] : memref<4xf32>)
  // CHECK: return %[[A]] : memref<4xf32>
  func.return %arg0 : memref<4xf32>
}
