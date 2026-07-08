// RUN: mfusion-opt %s --fuse-logical-not-compare | FileCheck %s

module {
  // CHECK-LABEL: func @fold_logical_not_eq
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[NE:.*]] = mfuse.ne %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.eq
  // CHECK-NOT: mfuse.logical_not
  // CHECK: return %[[NE]]
  func.func @fold_logical_not_eq(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xi1> {
    %0 = mfuse.eq %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %1 = mfuse.logical_not %0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %1 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_logical_not_comparisons
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[EQ:.*]] = mfuse.eq %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT_GT:.*]] = mfuse.logical_not %[[GT]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: %[[GE:.*]] = mfuse.ge %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT_GE:.*]] = mfuse.logical_not %[[GE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: %[[LT:.*]] = mfuse.lt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT_LT:.*]] = mfuse.logical_not %[[LT]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: %[[LE:.*]] = mfuse.le %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT_LE:.*]] = mfuse.logical_not %[[LE]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: return %[[EQ]], %[[NOT_GT]], %[[NOT_GE]], %[[NOT_LT]], %[[NOT_LE]]
  func.func @fold_logical_not_comparisons(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) ->
      (tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>) {
    %0 = mfuse.ne %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %1 = mfuse.logical_not %0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %2 = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %3 = mfuse.logical_not %2 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %4 = mfuse.ge %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %5 = mfuse.logical_not %4 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %6 = mfuse.lt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %7 = mfuse.logical_not %6 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %8 = mfuse.le %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %9 = mfuse.logical_not %8 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %1, %3, %5, %7, %9 : tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_logical_not_ordered_integer_comparisons
  // CHECK-SAME: %arg0: tensor<4x4xsi32>
  // CHECK-SAME: %arg1: tensor<4x4xsi32>
  // CHECK: %[[LE:.*]] = mfuse.le %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
  // CHECK: %[[LT:.*]] = mfuse.lt %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
  // CHECK: %[[GE:.*]] = mfuse.ge %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
  // CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.logical_not
  // CHECK: return %[[LE]], %[[LT]], %[[GE]], %[[GT]]
  func.func @fold_logical_not_ordered_integer_comparisons(%arg0: tensor<4x4xsi32>, %arg1: tensor<4x4xsi32>) ->
      (tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>) {
    %0 = mfuse.gt %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
    %1 = mfuse.logical_not %0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %2 = mfuse.ge %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
    %3 = mfuse.logical_not %2 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %4 = mfuse.lt %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
    %5 = mfuse.logical_not %4 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    %6 = mfuse.le %arg0, %arg1 : (tensor<4x4xsi32>, tensor<4x4xsi32>) -> tensor<4x4xi1>
    %7 = mfuse.logical_not %6 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %1, %3, %5, %7 : tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>, tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_logical_not_ordered_non_nan_float_constants
  // CHECK: %[[C0:.*]] = mfuse.constant dense<1.000000e+00> : tensor<4x4xf32>
  // CHECK: %[[C1:.*]] = mfuse.constant dense<2.000000e+00> : tensor<4x4xf32>
  // CHECK: %[[LE:.*]] = mfuse.le %[[C0]], %[[C1]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.logical_not
  // CHECK: return %[[LE]]
  func.func @fold_logical_not_ordered_non_nan_float_constants() -> tensor<4x4xi1> {
    %0 = mfuse.constant dense<1.000000e+00> : tensor<4x4xf32>
    %1 = mfuse.constant dense<2.000000e+00> : tensor<4x4xf32>
    %2 = mfuse.gt %0, %1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %3 = mfuse.logical_not %2 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %3 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @keep_logical_not_ordered_nan_float_constants
  // CHECK: %[[NAN:.*]] = mfuse.constant dense<0x7FC00000> : tensor<4x4xf32>
  // CHECK: %[[ONE:.*]] = mfuse.constant dense<1.000000e+00> : tensor<4x4xf32>
  // CHECK: %[[GT:.*]] = mfuse.gt %[[NAN]], %[[ONE]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT:.*]] = mfuse.logical_not %[[GT]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.le
  // CHECK: return %[[NOT]]
  func.func @keep_logical_not_ordered_nan_float_constants() -> tensor<4x4xi1> {
    %0 = mfuse.constant dense<0x7FC00000> : tensor<4x4xf32>
    %1 = mfuse.constant dense<1.000000e+00> : tensor<4x4xf32>
    %2 = mfuse.gt %0, %1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %3 = mfuse.logical_not %2 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %3 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_logical_not_ordered_non_nan_float_cast_from_int
  // CHECK-SAME: %arg0: tensor<4x4xsi32>
  // CHECK-SAME: %arg1: tensor<4x4xsi32>
  // CHECK: %[[CAST0:.*]] = mfuse.cast %arg0 : (tensor<4x4xsi32>) -> tensor<4x4xf32>
  // CHECK: %[[CAST1:.*]] = mfuse.cast %arg1 : (tensor<4x4xsi32>) -> tensor<4x4xf32>
  // CHECK: %[[LE:.*]] = mfuse.le %[[CAST0]], %[[CAST1]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.logical_not
  // CHECK: return %[[LE]]
  func.func @fold_logical_not_ordered_non_nan_float_cast_from_int(%arg0: tensor<4x4xsi32>, %arg1: tensor<4x4xsi32>) ->
      tensor<4x4xi1> {
    %2 = mfuse.cast %arg0 : (tensor<4x4xsi32>) -> tensor<4x4xf32>
    %3 = mfuse.cast %arg1 : (tensor<4x4xsi32>) -> tensor<4x4xf32>
    %4 = mfuse.gt %2, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %5 = mfuse.logical_not %4 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %5 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @keep_logical_not_ordered_external_float_inputs
  // CHECK-SAME: %arg0: tensor<4x4xf32>
  // CHECK-SAME: %arg1: tensor<4x4xf32>
  // CHECK: %[[GT:.*]] = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT:.*]] = mfuse.logical_not %[[GT]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.le
  // CHECK: return %[[NOT]]
  func.func @keep_logical_not_ordered_external_float_inputs(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) ->
      tensor<4x4xi1> {
    %0 = mfuse.gt %arg0, %arg1 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %1 = mfuse.logical_not %0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %1 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_logical_not_ordered_non_nan_float_full
  // CHECK: %[[FILL0:.*]] = mfuse.constant dense<1.000000e+00> : tensor<f32, {is_scalar = ""}>
  // CHECK: %[[FILL1:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
  // CHECK: %[[FULL0:.*]] = mfuse.full %[[FILL0]] : (tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
  // CHECK: %[[FULL1:.*]] = mfuse.full %[[FILL1]] : (tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
  // CHECK: %[[LE:.*]] = mfuse.le %[[FULL0]], %[[FULL1]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK-NOT: mfuse.logical_not
  // CHECK: return %[[LE]]
  func.func @fold_logical_not_ordered_non_nan_float_full() -> tensor<4x4xi1> {
    %0 = mfuse.constant dense<1.000000e+00> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.constant dense<2.000000e+00> : tensor<f32, {is_scalar = ""}>
    %2 = mfuse.full %0 : (tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    %3 = mfuse.full %1 : (tensor<f32, {is_scalar = ""}>) -> tensor<4x4xf32>
    %4 = mfuse.gt %2, %3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
    %5 = mfuse.logical_not %4 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %5 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @keep_standalone_logical_not
  // CHECK: %[[NOT:.*]] = mfuse.logical_not %arg0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: return %[[NOT]]
  func.func @keep_standalone_logical_not(%arg0: tensor<4x4xi1>) -> tensor<4x4xi1> {
    %0 = mfuse.logical_not %arg0 : (tensor<4x4xi1>) -> tensor<4x4xi1>
    return %0 : tensor<4x4xi1>
  }

  // CHECK-LABEL: func @fold_inside_fused_region
  // CHECK: %[[FUSED:.*]]:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"}
  // CHECK: ^bb0(%[[IN0:.*]]: tensor<4x4xf32>, %[[IN1:.*]]: tensor<4x4xf32>):
  // CHECK: %[[NE:.*]] = mfuse.ne %[[IN0]], %[[IN1]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[GT:.*]] = mfuse.gt %[[IN0]], %[[IN1]] : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
  // CHECK: %[[NOT_GT:.*]] = mfuse.logical_not %[[GT]] : (tensor<4x4xi1>) -> tensor<4x4xi1>
  // CHECK: mfuse.yield %[[NE]], %[[NOT_GT]]
  func.func @fold_inside_fused_region(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) ->
      (tensor<4x4xi1>, tensor<4x4xi1>) {
    %0:2 = mfuse.fused %arg0, %arg1 {fusion_type = "dvm"} : (tensor<4x4xf32>, tensor<4x4xf32>) ->
        (tensor<4x4xi1>, tensor<4x4xi1>) {
    ^bb0(%arg2: tensor<4x4xf32>, %arg3: tensor<4x4xf32>):
      %1 = mfuse.eq %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %2 = mfuse.logical_not %1 : (tensor<4x4xi1>) -> tensor<4x4xi1>
      %3 = mfuse.gt %arg2, %arg3 : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
      %4 = mfuse.logical_not %3 : (tensor<4x4xi1>) -> tensor<4x4xi1>
      mfuse.yield %2, %4 : tensor<4x4xi1>, tensor<4x4xi1>
    }
    return %0#0, %0#1 : tensor<4x4xi1>, tensor<4x4xi1>
  }
}
