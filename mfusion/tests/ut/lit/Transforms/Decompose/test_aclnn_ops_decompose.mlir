// RUN: mfusion-opt %s -decompose="pattern-type=BEFORE_MANUAL_FUSION" -allow-unregistered-dialect -mlir-print-ir-after-all | FileCheck %s

module {
// CHECK-LABEL: @aclnn_add_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.mul
// CHECK: mfuse.add
func.func @aclnn_add_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_test_with_f64_alpha
// CHECK: mfuse.aclnn.add
func.func @aclnn_add_test_with_f64_alpha(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f64>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_with_f64_input
// CHECK: mfuse.aclnn.add
func.func @aclnn_add_alpha_one_with_f64_input(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf64>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf64>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_add_alpha_one_test
// CHECK-NOT: mfuse.aclnn.add
// CHECK: mfuse.add
// CHECK-NOT: mfuse.mul
func.func @aclnn_add_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (1.0)
  %alpha = arith.constant dense<1.0> : tensor<f64>
  // AclnnAdd with alpha == 1
  %0 = mfuse.aclnn.add %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_test
// CHECK: mfuse.aclnn.sub
func.func @aclnn_sub_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // Create alpha constant (0.5)
  %alpha = arith.constant dense<0.5> : tensor<f64>
  // AclnnSub with alpha != 1
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_alpha_one_test
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.sub
// CHECK-NOT: mfuse.mul
func.func @aclnn_sub_alpha_one_test(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %alpha = arith.constant dense<1.0> : tensor<f64>
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<f64>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: @aclnn_sub_alpha_with_cast
// CHECK-NOT: mfuse.aclnn.sub
// CHECK: mfuse.cast
// CHECK: mfuse.mul
// CHECK: mfuse.sub
func.func @aclnn_sub_alpha_with_cast(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf16>, %alpha: tensor<f32>) -> tensor<4x4xf32> {
  %0 = mfuse.aclnn.sub %arg0, %arg1, %alpha : (tensor<4x4xf32>, tensor<4x4xf16>, tensor<f32>) -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}
}

// decompose AclnnAdd with dynamic 2D shape torch tensor inputs (partially static)
func.func @test_aclnn_add_dynamic_shape_2d(%arg0: !torch.int, %arg1: !torch.vtensor<[2,?],f32>, %arg2: !torch.vtensor<[2,?],f32>) -> !torch.vtensor<[2,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[2,?],f32> to tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>, tensor<f32>) -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>> to !torch.vtensor<[2,?],f32>
  return %3 : !torch.vtensor<[2,?],f32>

  // CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_2d
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.mul {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
  // CHECK: mfuse.add {{.*}} -> tensor<2x?xf32, #mfuse.symshape<["2", "s0"]>>
}

// decompose AclnnAdd with dynamic 1D shape torch tensor inputs
func.func @test_aclnn_add_dynamic_shape_1d(%arg0: !torch.int, %arg1: !torch.vtensor<[?],f32>, %arg2: !torch.vtensor<[?],f32>) -> !torch.vtensor<[?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?],f32> to tensor<?xf32, #mfuse.symshape<["s0"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?],f32> to tensor<?xf32, #mfuse.symshape<["s0"]>>
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<?xf32, #mfuse.symshape<["s0"]>>, tensor<f32>) -> tensor<?xf32, #mfuse.symshape<["s0"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?xf32, #mfuse.symshape<["s0"]>> to !torch.vtensor<[?],f32>
  return %3 : !torch.vtensor<[?],f32>

  // CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_1d
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.mul {{.*}} -> tensor<?xf32, #mfuse.symshape<["s0"]>>
  // CHECK: mfuse.add {{.*}} -> tensor<?xf32, #mfuse.symshape<["s0"]>>
}

// decompose AclnnAdd with fully dynamic 2D shape torch tensor inputs
func.func @test_aclnn_add_dynamic_shape_2d_all_dynamic(%arg0: !torch.int, %arg1: !torch.vtensor<[?,?],f32>, %arg2: !torch.vtensor<[?,?],f32>) -> !torch.vtensor<[?,?],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>, s1 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?,?],f32> to tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?,?],f32> to tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>, tensor<f32>) -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>> to !torch.vtensor<[?,?],f32>
  return %3 : !torch.vtensor<[?,?],f32>

  // CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_2d_all_dynamic
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.mul {{.*}} -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
  // CHECK: mfuse.add {{.*}} -> tensor<?x?xf32, #mfuse.symshape<["s0", "s1"]>>
}

// decompose AclnnAdd with broadcasted dynamic 2D shapes:
// a.shape = [s0, 1], b.shape = [s1, 8]
func.func @test_aclnn_add_dynamic_shape_broadcast(%arg0: !torch.int, %arg1: !torch.vtensor<[?,1],f32>, %arg2: !torch.vtensor<[?,8],f32>) -> !torch.vtensor<[?,8],f32> attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>, s1 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
  %0 = builtin.unrealized_conversion_cast %arg1 : !torch.vtensor<[?,1],f32> to tensor<?x1xf32, #mfuse.symshape<["s0", "1"]>>
  %1 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?,8],f32> to tensor<?x8xf32, #mfuse.symshape<["s1", "8"]>>
  %alpha = arith.constant dense<0.5> : tensor<f32>
  %2 = mfuse.aclnn.add %0, %1, %alpha : (tensor<?x1xf32, #mfuse.symshape<["s0", "1"]>>, tensor<?x8xf32, #mfuse.symshape<["s1", "8"]>>, tensor<f32>) -> tensor<?x8xf32, #mfuse.symshape<["s0", "8"]>>
  %3 = builtin.unrealized_conversion_cast %2 : tensor<?x8xf32, #mfuse.symshape<["s0", "8"]>> to !torch.vtensor<[?,8],f32>
  return %3 : !torch.vtensor<[?,8],f32>

  // CHECK-LABEL: func.func @test_aclnn_add_dynamic_shape_broadcast
  // CHECK-NOT: mfuse.aclnn.add
  // CHECK: mfuse.mul
  // CHECK: mfuse.add
}
