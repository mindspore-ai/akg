// RUN: mfusion-opt %s --mfuse-promote-binary-ops | FileCheck %s

// CHECK-LABEL: func @add_mixed_types
// Test: Add with mixed f16 and f32 inputs, result is f32
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.add %[[CAST]], %{{.*}}
func.func @add_mixed_types(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<4x4xf32>):
      %1 = mfuse.add %x0, %x1 : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @sub_mixed_types
// Test: Sub with mixed f16 and f32 inputs, result is f32
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.sub %[[CAST]], %{{.*}}
func.func @sub_mixed_types(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<4x4xf32>):
      %1 = mfuse.sub %x0, %x1 : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @mul_mixed_types
// Test: Mul with mixed f16 and f32 inputs, result is f32
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.mul %[[CAST]], %{{.*}}
func.func @mul_mixed_types(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<4x4xf32>):
      %1 = mfuse.mul %x0, %x1 : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @eq_mixed_types
// Test: Eq (comparison) with mixed f16 and f32 inputs
// Should promote both inputs to same type (f32)
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.eq %[[CAST]], %{{.*}}
func.func @eq_mixed_types(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<4x4xf32>):
      %1 = mfuse.eq %x0, %x1 : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// CHECK-LABEL: func @gt_mixed_types
// Test: Gt (comparison) with mixed f16 and f32 inputs
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.gt %[[CAST]], %{{.*}}
func.func @gt_mixed_types(%arg0: tensor<4x4xf16>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<4x4xf32>):
      %1 = mfuse.gt %x0, %x1 : (tensor<4x4xf16>, tensor<4x4xf32>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// CHECK-LABEL: func @no_promotion_same_types
// Test: Should NOT promote when input types already match
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.add
func.func @no_promotion_same_types(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf32>, %x1: tensor<4x4xf32>):
      %1 = mfuse.add %x0, %x1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @add_scalar_right_valid
// Test: Add with lhs=f32 and rhs=scalar(f64), should NOT promote rhs (valid)
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.add
func.func @add_scalar_right_valid(%arg0: tensor<4x4xf32>, %arg1: tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf32>, %x1: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.add %x0, %x1 : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @add_scalar_right_lhs_not_valid
// Test: Add with lhs=f16 and rhs=scalar(f64), should promote lhs to f32
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.add %[[CAST]], %{{.*}}
func.func @add_scalar_right_lhs_not_valid(%arg0: tensor<4x4xf16>, %arg1: tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<4x4xf16>, %x1: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.add %x0, %x1 : (tensor<4x4xf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @eq_scalar_right_valid
// Test: Eq comparison with scalar right, same types should remain unchanged
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.eq
func.func @eq_scalar_right_valid(%arg0: tensor<4x4xi64>, %arg1: tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xi64>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<4x4xi64>, %x1: tensor<i64, {is_scalar = ""}>):
      %1 = mfuse.eq %x0, %x1 : (tensor<4x4xi64>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// CHECK-LABEL: func @eq_scalar_right_promote_rhs
// Test: Eq comparison with scalar right is left unchanged by this pass
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.eq
func.func @eq_scalar_right_promote_rhs(%arg0: tensor<4x4xi32>, %arg1: tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<4x4xi32>, %x1: tensor<i64, {is_scalar = ""}>):
      %1 = mfuse.eq %x0, %x1 : (tensor<4x4xi32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// CHECK-LABEL: func @gt_scalar_right_promote_rhs
// Test: Gt comparison with scalar right is left unchanged by this pass
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.gt
func.func @gt_scalar_right_promote_rhs(%arg0: tensor<4x4xf32>, %arg1: tensor<f64, {is_scalar = ""}>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<4x4xf32>, %x1: tensor<f64, {is_scalar = ""}>):
      %1 = mfuse.gt %x0, %x1 : (tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}

// CHECK-LABEL: func @sub_scalar_left_valid
// Test: Sub with lhs=scalar(i64) and rhs=f32 should NOT promote scalar or rhs
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.sub
func.func @sub_scalar_left_valid(%arg0: tensor<i64, {is_scalar = ""}>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<i64, {is_scalar = ""}>, %x1: tensor<4x4xf32>):
      %1 = mfuse.sub %x0, %x1 : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @sub_scalar_left_rhs_not_valid
// Test: Sub with lhs=scalar(i64), rhs=f16 and result=f32 should promote tensor rhs only
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.sub %{{.*}}, %[[CAST]]
func.func @sub_scalar_left_rhs_not_valid(%arg0: tensor<i64, {is_scalar = ""}>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf16>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<i64, {is_scalar = ""}>, %x1: tensor<4x4xf16>):
      %1 = mfuse.sub %x0, %x1 : (tensor<i64, {is_scalar = ""}>, tensor<4x4xf16>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @div_scalar_left_rhs_not_valid
// Test: Div with lhs=scalar(f64), rhs=f16 and result=f32 should promote tensor rhs only
// CHECK: %[[CAST:.*]] = mfuse.cast %{{.*}} : (tensor<4x4xf16>) -> tensor<4x4xf32>
// CHECK: mfuse.div %{{.*}}, %[[CAST]]
func.func @div_scalar_left_rhs_not_valid(%arg0: tensor<f64, {is_scalar = ""}>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<f64, {is_scalar = ""}>, tensor<4x4xf16>) -> tensor<4x4xf32> {
    ^bb0(%x0: tensor<f64, {is_scalar = ""}>, %x1: tensor<4x4xf16>):
      %1 = mfuse.div %x0, %x1 : (tensor<f64, {is_scalar = ""}>, tensor<4x4xf16>) -> tensor<4x4xf32>
      mfuse.yield %1 : tensor<4x4xf32>
  }
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @gt_scalar_left_promote_lhs
// Test: Gt comparison with scalar left is left unchanged by this pass
// CHECK-NOT: mfuse.cast
// CHECK: mfuse.gt
func.func @gt_scalar_left_promote_lhs(%arg0: tensor<f64, {is_scalar = ""}>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
  %0 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<f64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xi1> {
    ^bb0(%x0: tensor<f64, {is_scalar = ""}>, %x1: tensor<4x4xf32>):
      %1 = mfuse.gt %x0, %x1 : (tensor<f64, {is_scalar = ""}>, tensor<4x4xf32>) -> tensor<4x4xi1>
      mfuse.yield %1 : tensor<4x4xi1>
  }
  return %0 : tensor<4x4xi1>
}
