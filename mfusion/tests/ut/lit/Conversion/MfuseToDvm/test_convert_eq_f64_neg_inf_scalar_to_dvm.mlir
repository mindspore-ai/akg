// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // Albert-style attention mask: eq(scores, -inf) with f32 tensor and f64 scalar.
  // CHECK-LABEL: func @main_eq_f64_neg_inf_scalar_fused
  // CHECK-SAME: (%[[SCORES:.*]]: tensor<4x12x512x512xf32>)
  // CHECK: %[[LOAD:.*]] = dvm.load %[[SCORES]] : tensor<4x12x512x512xf32> -> tensor<4x12x512x512xf32>
  // CHECK: %[[EQ:.*]] = dvm.binary_scalar Equal %[[LOAD]], {{.*}} : tensor<4x12x512x512xf32>, f32 -> tensor<4x12x512x512xi1>
  // CHECK: %[[STORE:.*]] = dvm.store %[[EQ]] : tensor<4x12x512x512xi1> -> tensor<4x12x512x512xi1>
  // CHECK: return %[[STORE]] : tensor<4x12x512x512xi1>
  // CHECK-NOT: dvm.binary
  // CHECK-NOT: mfuse.eq
  func.func @main_eq_f64_neg_inf_scalar_fused(%scores: tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xi1>
      attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %neg = mfuse.constant dense<0xFF800000> : tensor<f64, {is_scalar = ""}>
    %eq = mfuse.eq %scores, %neg : (tensor<4x12x512x512xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x12x512x512xi1>
    return %eq : tensor<4x12x512x512xi1>
  }
}
