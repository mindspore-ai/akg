// RUN: mfusion-opt %s --split | FileCheck %s

module {
  // Matcher-fused safe-softmax islands (tag on mfuse.fused wrapper only) must not be split.
  // CHECK-LABEL: func @test_skip_safe_softmax_split
  // CHECK-COUNT-1: mfuse.fused
  // CHECK-SAME: mfusion.dvm_fuse_kind = "safe_softmax"
  // CHECK: mfuse.reduce_max
  // CHECK: mfuse.yield
  func.func @test_skip_safe_softmax_split(%mask: tensor<4x12x512x1xi1>, %scores: tensor<4x12x512x512xf32>)
      -> tensor<4x12x512x512xf32> {
    %fused:2 = mfuse.fused %mask, %scores {
      fusion_type = "dvm",
      mfusion.dvm_fuse_kind = "safe_softmax",
      mfusion.dvm_fuse_group = "safe_softmax#0",
      mfusion.dvm_fuse_role = "member"
    } : (tensor<4x12x512x1xi1>, tensor<4x12x512x512xf32>) -> (tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) {
    ^bb0(%arg0: tensor<4x12x512x1xi1>, %arg1: tensor<4x12x512x512xf32>):
      %cst = mfuse.constant dense<0> : tensor<i64, {is_scalar = ""}>
      %zero = mfuse.full %cst : (tensor<i64, {is_scalar = ""}>) -> tensor<4x12x512x512xf32>
      %not = mfuse.logical_not %arg0 : (tensor<4x12x512x1xi1>) -> tensor<4x12x512x1xi1>
      %amax = mfuse.reduce_max %arg1 {dimensions = [3], keepdim = true}
          : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x1xf32>
      %sub = mfuse.sub %arg1, %amax : (tensor<4x12x512x512xf32>, tensor<4x12x512x1xf32>) -> tensor<4x12x512x512xf32>
      %exp = mfuse.exp %sub : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
      %sum = mfuse.reduce_sum %exp {dimensions = [3], keepdim = true}
          : (tensor<4x12x512x512xf32>) -> tensor<4x12x512x1xf32>
      %div = mfuse.div %exp, %sum : (tensor<4x12x512x512xf32>, tensor<4x12x512x1xf32>) -> tensor<4x12x512x512xf32>
      %sel = mfuse.select %not, %zero, %div
          : (tensor<4x12x512x1xi1>, tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>) -> tensor<4x12x512x512xf32>
      mfuse.yield %zero, %sel : tensor<4x12x512x512xf32>, tensor<4x12x512x512xf32>
    }
    return %fused#1 : tensor<4x12x512x512xf32>
  }
}
