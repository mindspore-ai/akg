// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // CHECK-LABEL: func.func @main_bool_scalar
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<2xf32>)
  // CHECK: %[[LOAD:.*]] = dvm.load %[[ARG0]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[ADD:.*]] = dvm.binary_scalar Add %[[LOAD]], 1 : tensor<2xf32>, i32 -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.constant
  func.func @main_bool_scalar(%arg0: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<true> : tensor<i1, {is_scalar = ""}>
    %1 = mfuse.add %arg0, %0 : (tensor<2xf32>, tensor<i1, {is_scalar = ""}>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }
}
