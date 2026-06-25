// RUN: mfusion-opt %s --mfuse-fold-redundant-matmul-transpose --canonicalize | FileCheck %s

module {
  // CHECK-LABEL: func.func @main
  func.func @main(%arg0: tensor<4x2xf32>, %arg1: tensor<2x3xf32>,
                  %arg2: tensor<4x2xf32>, %arg3: tensor<3x2xf32>, %arg4: tensor<3xf32>,
                  %arg5: tensor<5x4x2xf32>, %arg6: tensor<5x2x3xf32>,
                  %arg7: tensor<5x4x2xf32>, %arg8: tensor<5x3x2xf32>,
                  %arg9: tensor<4x2xf32>, %arg10: tensor<2x3xf32>,
                  %arg11: tensor<4x2xf32>, %arg12: tensor<3x2xf32>,
                  %arg13: tensor<2x4xf32>, %arg14: tensor<2x3xf32>)
      -> (tensor<4x3xf32>, tensor<4x3xf32>, tensor<5x4x3xf32>, tensor<5x4x3xf32>,
          tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>) {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %1 = mfuse.matmul %0, %arg1 {trans_x1 = true} : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>

    %2 = mfuse.permute %arg3, [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %3 = mfuse.matmul_with_bias %arg2, %2, %arg4 {trans_x2 = true} : (tensor<4x2xf32>, tensor<2x3xf32>, tensor<3xf32>) -> tensor<4x3xf32>

    %4 = mfuse.permute %arg5, [0, 2, 1] : (tensor<5x4x2xf32>) -> tensor<5x2x4xf32>
    %5 = mfuse.batch_matmul %4, %arg6 {transpose_a = true} : (tensor<5x2x4xf32>, tensor<5x2x3xf32>) -> tensor<5x4x3xf32>

    %6 = mfuse.permute %arg8, [0, 2, 1] : (tensor<5x3x2xf32>) -> tensor<5x2x3xf32>
    %7 = mfuse.aclnn.batch_matmul %arg7, %6 {trans_x2 = true} : (tensor<5x4x2xf32>, tensor<5x2x3xf32>) -> tensor<5x4x3xf32>

    %8 = mfuse.permute %arg9, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %9 = mfuse.aclnn.matmul %8, %arg10 {trans_x1 = true} : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>

    %10 = mfuse.permute %arg12, [1, 0] : (tensor<3x2xf32>) -> tensor<2x3xf32>
    %11 = mfuse.aclnn.mm %arg11, %10 {trans_x2 = true} : (tensor<4x2xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>

    %12 = mfuse.aclnn.mm %arg13, %arg14 {trans_x1 = true} : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
    return %1, %3, %5, %7, %9, %11, %12 : tensor<4x3xf32>, tensor<4x3xf32>, tensor<5x4x3xf32>, tensor<5x4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>, tensor<4x3xf32>
  }

  // CHECK-NOT: mfuse.permute
  // CHECK: %[[MM:.*]] = mfuse.matmul %arg0, %arg1 : (tensor<4x2xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[MWB:.*]] = mfuse.matmul_with_bias %arg2, %arg3, %arg4 : (tensor<4x2xf32>, tensor<3x2xf32>, tensor<3xf32>) -> tensor<4x3xf32>
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[BMM:.*]] = mfuse.batch_matmul %arg5, %arg6 : (tensor<5x4x2xf32>, tensor<5x2x3xf32>) -> tensor<5x4x3xf32>
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[ACLNN_BMM:.*]] = mfuse.aclnn.batch_matmul %arg7, %arg8 : (tensor<5x4x2xf32>, tensor<5x3x2xf32>) -> tensor<5x4x3xf32>
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[ACLNN_MATMUL:.*]] = mfuse.aclnn.matmul %arg9, %arg10 : (tensor<4x2xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
  // CHECK-NOT: mfuse.permute
  // CHECK: %[[ACLNN_MM:.*]] = mfuse.aclnn.mm %arg11, %arg12 : (tensor<4x2xf32>, tensor<3x2xf32>) -> tensor<4x3xf32>
  // CHECK: %[[RESIDUAL:.*]] = mfuse.aclnn.mm %arg13, %arg14 {trans_x1 = true} : (tensor<2x4xf32>, tensor<2x3xf32>) -> tensor<4x3xf32>
  // CHECK: return %[[MM]], %[[MWB]], %[[BMM]], %[[ACLNN_BMM]], %[[ACLNN_MATMUL]], %[[ACLNN_MM]], %[[RESIDUAL]]

  // CHECK-LABEL: func.func @main_fused_0_shape
  func.func @main_fused_0_shape(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>)
      -> tensor<4096x64xf32> attributes {mfusion.fusion_type = "dvm"} {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK: mfuse.permute
  // CHECK: mfuse.aclnn.mm {{.*}} {trans_x1 = true}

  // CHECK-LABEL: func.func private @outlined_dvm_shape
  func.func private @outlined_dvm_shape(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>)
      -> tensor<4096x64xf32> attributes {mfusion.fusion_type = "dvm", mfusion.outlined} {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK: mfuse.permute
  // CHECK: mfuse.aclnn.mm {{.*}} {trans_x1 = true}

  // CHECK-LABEL: func.func private @copied_akg_shape
  func.func private @copied_akg_shape(%arg0: tensor<4096x16xf32>, %arg1: tensor<16x64xf32>)
      -> tensor<4096x64xf32> attributes {mfusion.fusion_type = "akg"} {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4096x16xf32>) -> tensor<16x4096xf32>
    %1 = mfuse.aclnn.mm %0, %arg1 {trans_x1 = true} : (tensor<16x4096xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
    return %1 : tensor<4096x64xf32>
  }

  // CHECK-NOT: mfuse.permute
  // CHECK: mfuse.aclnn.mm %arg0, %arg1 : (tensor<4096x16xf32>, tensor<16x64xf32>) -> tensor<4096x64xf32>
}
