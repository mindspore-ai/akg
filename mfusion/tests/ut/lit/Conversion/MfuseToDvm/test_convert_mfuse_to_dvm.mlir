// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // CHECK-LABEL: func @main_mul_fused_0
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.mul
  func.func @main_mul_fused_0(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_mul_fused_1
  // CHECK: mfuse.mul
  // CHECK-NOT: dvm.load
  // CHECK-NOT: dvm.binary
  // CHECK-NOT: dvm.store
  func.func @main_mul_fused_1(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "akg"} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main
  // CHECK: %[[CALL:.*]] = call @main_mul_fused_0(%[[ARG0:.*]], %[[ARG1:.*]])
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = call @main_mul_fused_0(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
