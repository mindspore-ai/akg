// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" --verify-each | FileCheck %s

// CHECK-LABEL: @aclnn_add_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @aclnn_add_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_test_with_f64_alpha
// CHECK: mfuse.add
func.func @aclnn_add_test_with_f64_alpha(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_with_f16_input
// CHECK: mfuse.add
func.func @aclnn_add_alpha_one_with_f16_input(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf16>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.add
// CHECK-NOT: mfuse.mul
func.func @aclnn_add_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_test
// CHECK: mfuse.mul {{.*}} -> tensor<4x4xf32>
func.func @aclnn_sub_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_alpha_one_test
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.sub
// CHECK-NOT: mfuse.mul
func.func @aclnn_sub_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_scalar_lhs_bf16_tensor_alpha_one_test
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.sub {{.*}} : (tensor<f32, {is_scalar = ""}>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK-NOT: mfuse.mul
func.func @aclnn_sub_scalar_lhs_bf16_tensor_alpha_one_test(%arg0: tensor<f32, {is_scalar = ""}>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<f32, {is_scalar = ""}>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

// CHECK-LABEL: @aclnn_add_i64_scalar_lhs_bool_tensor
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.add %arg0, {{.*}} : (tensor<4x4xi1>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi64>
func.func @aclnn_add_i64_scalar_lhs_bool_tensor(%arg0: tensor<4x4xi1>) -> tensor<4x4xi64> {
  %scalar = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %scalar, %arg0, %alpha : (tensor<i64, {is_scalar = ""}>, tensor<4x4xi1>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}

// CHECK-LABEL: @aclnn_add_i64_scalar_rhs_bool_tensor
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.add %arg0, {{.*}} : (tensor<4x4xi1>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xi64>
func.func @aclnn_add_i64_scalar_rhs_bool_tensor(%arg0: tensor<4x4xi1>) -> tensor<4x4xi64> {
  %scalar = mfuse.constant dense<2> : tensor<i64, {is_scalar = ""}>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %arg0, %scalar, %alpha : (tensor<4x4xi1>, tensor<i64, {is_scalar = ""}>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}

// CHECK-LABEL: @aclnn_sub_num_to_tensor_scalar_lhs_user_chain
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK-NOT: mfuse.num_to_tensor
// CHECK: %[[SUB:.*]] = mfuse.sub %[[SCALAR]], %arg0 : (tensor<f64, {is_scalar = ""}>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: %[[ADD:.*]] = mfuse.add %[[SUB]], %arg1 : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return %[[ADD]] : tensor<4x4xbf16>
func.func @aclnn_sub_num_to_tensor_scalar_lhs_user_chain(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %n, %arg0, %alpha : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  %1 = mfuse.aclnn.add %0, %arg1, %alpha : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %1 : tensor<4x4xbf16>
}

// CHECK-LABEL: @keep_aclnn_sub_when_scalar_inference_changes_result_type
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: %[[NUM:.*]] = mfuse.num_to_tensor %[[SCALAR]] : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
// CHECK: %[[SUB:.*]] = mfuse.aclnn.sub %[[NUM]], %arg0, {{.*}} : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
// CHECK: return %[[SUB]] : tensor<4x4xf32>
func.func @keep_aclnn_sub_when_scalar_inference_changes_result_type(%arg0: tensor<4x4xbf16>) -> tensor<4x4xf32> {
  %c = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %n, %arg0, %alpha : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_num_to_tensor_scalar_lhs
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK-NOT: mfuse.num_to_tensor
// CHECK: %[[ADD:.*]] = mfuse.add %arg0, %[[SCALAR]] : (tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
// CHECK: return %[[ADD]] : tensor<4x4xbf16>
func.func @aclnn_add_num_to_tensor_scalar_lhs(%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.add %n, %arg0, %alpha : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

// CHECK-LABEL: @aclnn_sub_num_to_tensor_scalar_lhs_non_unit_alpha
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK-NOT: mfuse.num_to_tensor
// CHECK: %[[MUL:.*]] = mfuse.mul %arg0, {{.*}} : (tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
// CHECK: %[[SUB:.*]] = mfuse.sub %[[SCALAR]], %[[MUL]] : (tensor<f64, {is_scalar = ""}>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return %[[SUB]] : tensor<4x4xbf16>
func.func @aclnn_sub_num_to_tensor_scalar_lhs_non_unit_alpha(%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %n, %arg0, %alpha : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

// CHECK-LABEL: @keep_aclnn_sub_num_to_tensor_scalar_rhs_non_unit_alpha
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: %[[NUM:.*]] = mfuse.num_to_tensor %[[SCALAR]] : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
// CHECK-NOT: mfuse.mul
// CHECK: %[[SUB:.*]] = mfuse.aclnn.sub %arg0, %[[NUM]], {{.*}} : (tensor<4x4xbf16>, tensor<f32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
// CHECK: return %[[SUB]] : tensor<4x4xbf16>
func.func @keep_aclnn_sub_num_to_tensor_scalar_rhs_non_unit_alpha(%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
  %c = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %arg0, %n, %alpha : (tensor<4x4xbf16>, tensor<f32>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %0 : tensor<4x4xbf16>
}

// CHECK-LABEL: @keep_shared_num_to_tensor
// CHECK: %[[SCALAR:.*]] = mfuse.constant dense<2.000000e+00> : tensor<f64, {is_scalar = ""}>
// CHECK: %[[NUM:.*]] = mfuse.num_to_tensor %[[SCALAR]] : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
// CHECK: %[[SUB:.*]] = mfuse.sub %[[SCALAR]], %arg0 : (tensor<f64, {is_scalar = ""}>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return %[[SUB]], %[[NUM]] : tensor<4x4xbf16>, tensor<f32>
func.func @keep_shared_num_to_tensor(%arg0: tensor<4x4xbf16>) -> (tensor<4x4xbf16>, tensor<f32>) {
  %c = mfuse.constant dense<2.0> : tensor<f64, {is_scalar = ""}>
  %n = mfuse.num_to_tensor %c : (tensor<f64, {is_scalar = ""}>) -> tensor<f32>
  %alpha = mfuse.constant dense<1.0> : tensor<f64, {is_scalar = ""}>
  %0 = mfuse.aclnn.sub %n, %arg0, %alpha : (tensor<f32>, tensor<4x4xbf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xbf16>
  return %0, %n : tensor<4x4xbf16>, tensor<f32>
}

// CHECK-LABEL: @aclnn_sub_alpha_with_cast
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.mul
// CHECK: mfuse.sub
func.func @aclnn_sub_alpha_with_cast(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf16>, %alpha: tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf16>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// decompose AclnnAdd with dynamic 2D shape torch tensor inputs (partially static)
// CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_2d
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
// CHECK: mfuse.add {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
func.func @test_aclnn_add_dynamic_shape_2d(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>, %arg2: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<f64, {is_scalar = ""}>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> to !torch.vtensor<[2,?],f32>
  return %3 : !torch.vtensor<[2,?],f32>
}

// decompose AclnnAdd with dynamic 1D shape torch tensor inputs
// CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_1d
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul {{.*}} -> tensor<?xf32, #mfuse.symshape<["s0"]>>
// CHECK: mfuse.add {{.*}} -> tensor<?xf32, #mfuse.symshape<["s0"]>>
func.func @test_aclnn_add_dynamic_shape_1d(%arg0: !torch.int, %arg1: !torch.vtensor<[?],f32>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?],f32> to tensor<?xf32, #mfuse.symshape<["s0"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?],f32> to tensor<?xf32, #mfuse.symshape<["s0"]>>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<f64, {is_scalar = ""}>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?xf32, #mfuse.symshape<["s0"]>> to !torch.vtensor<[?],f32>
  return %3 : !torch.vtensor<[?],f32>
}

// decompose AclnnAdd with fully dynamic 2D shape torch tensor inputs
// CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_2d_all_dynamic
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul {{.*}} -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
// CHECK: mfuse.add {{.*}} -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
func.func @test_aclnn_add_dynamic_shape_2d_all_dynamic(%arg0: !torch.int, %arg1: !torch.vtensor<[?,?],f32>, %arg2: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>, s1 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?,?],f32> to tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?,?],f32> to tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<f64, {is_scalar = ""}>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>> to !torch.vtensor<[?,?],f32>
  return %3 : !torch.vtensor<[?,?],f32>
}

// decompose AclnnAdd with broadcasted dynamic 2D shapes:
// a.shape = [s0, 1], b.shape = [s1, 8]
// CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_broadcast
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @test_aclnn_add_dynamic_shape_broadcast(%arg0: !torch.int, %arg1: !torch.vtensor<[?,1],f32>, %arg2: !torch.vtensor<[?,8],f32>) -> !torch.vtensor<[?,8],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>, s1 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?,1],f32> to tensor<?x1xf32, #mfuse.symshape<["s0", "1"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?,8],f32> to tensor<?x8xf32, #mfuse.symshape<["s1", "8"]>>
  %alpha = mfuse.constant dense<0.5> : tensor<f64, {is_scalar = ""}>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?x1xf32, #mfuse.symshape<["s0", "1"]>>, tensor<?x8xf32, #mfuse.symshape<["s1", "8"]>>, tensor<f64, {is_scalar = ""}>) -> tensor<?x8xf32, #mfuse.symshape<["s0", "8"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?x8xf32, #mfuse.symshape<["s0", "8"]>> to !torch.vtensor<[?,8],f32>
  return %3 : !torch.vtensor<[?,8],f32>
}

// CHECK-LABEL: @test_aclnn_add_with_scalar
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : !torch.int to tensor<i64, {is_scalar = ""}>
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : !torch.vtensor<[4,4],f32> to tensor<4x4xf32>
// CHECK: mfuse.add %{{.*}}, %{{.*}} : (tensor<4x4xf32>, tensor<i64, {is_scalar = ""}>) -> tensor<4x4xf32>
// CHECK: builtin.unrealized_conversion_cast %{{.*}} : tensor<4x4xf32> to !torch.vtensor<[4,4],f32>
func.func @test_aclnn_add_with_scalar(%arg0: !torch.vtensor<[4,4],f32>, %arg1: !torch.int) -> !torch.vtensor<[4,4],f32> {
  %0 = mfuse.constant dense<1.000000e+00> : tensor<f64, {is_scalar = ""}>
  %1 = builtin.unrealized_conversion_cast %arg1 : !torch.int to tensor<i64, {is_scalar = ""}>
  %2 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[4,4],f32> to tensor<4x4xf32>
  %3 = mfuse.aclnn.add %2, %1, %0 : (tensor<4x4xf32>, tensor<i64, {is_scalar = ""}>, tensor<f64, {is_scalar = ""}>) -> tensor<4x4xf32>
  %4 = builtin.unrealized_conversion_cast %3 : tensor<4x4xf32> to !torch.vtensor<[4,4],f32>
  return %4 : !torch.vtensor<[4,4],f32>
}
