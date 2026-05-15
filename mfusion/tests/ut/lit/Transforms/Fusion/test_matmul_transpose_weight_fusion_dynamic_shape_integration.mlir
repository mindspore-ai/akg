// RUN: mfusion-opt %s --mfuse-fusion | FileCheck %s

// Regression: mfuse-fusion on dynamic-shape inputs must complete successfully
// after transpose-weight alignment. The specific alignment permute inserted by
// fuse-matmul-transpose-weight may be folded by later sub-passes, so this test
// only checks stable final-IR properties of the composite pipeline.

module {
  // CHECK-LABEL: func.func @main
  // CHECK: mfuse.matmul
  // CHECK: -> tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>
  // CHECK: mfuse.permute %{{.*}}, [1, 0] : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>) -> tensor<16x?xf16, #mfuse.symshape<["16", "s0"]>>
  func.func @main(%arg0: !torch.vtensor<[32,16],f16>, %arg1: !torch.int, %arg2: !torch.vtensor<[?,16],f16>, %arg3: !torch.vtensor<[16,32],f16>, %arg4: !torch.vtensor<[16,16],f16>) -> (!torch.vtensor<[?,16],f16>, !torch.vtensor<[?,32],f16>, !torch.vtensor<[?,16],i1>, !torch.vtensor<[16,?],f16>, !torch.vtensor<[16,32],f16>, !torch.int) attributes {mfuse.syminfo = {s0 = #mfuse.syminfo<range=[2, inf]>}, torch.assume_strict_symbolic_shapes} {
    %0 = mfuse.constant dense<0.000000e+00> : tensor<f16, {is_scalar = ""}>
    %1 = builtin.unrealized_conversion_cast %arg4 : !torch.vtensor<[16,16],f16> to tensor<16x16xf16>
    %2 = builtin.unrealized_conversion_cast %arg3 : !torch.vtensor<[16,32],f16> to tensor<16x32xf16>
    %3 = builtin.unrealized_conversion_cast %arg2 : !torch.vtensor<[?,16],f16> to tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>
    %4 = builtin.unrealized_conversion_cast %arg0 : !torch.vtensor<[32,16],f16> to tensor<32x16xf16>
    %5 = mfuse.permute %4, [1, 0] : (tensor<32x16xf16>) -> tensor<16x32xf16>
    %6 = mfuse.matmul %3, %5 : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>, tensor<16x32xf16>) -> tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>
    %7 = mfuse.relu %6 : (tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>) -> tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>
    %8 = builtin.unrealized_conversion_cast %7 : tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>> to !torch.vtensor<[?,32],f16>
    %9 = mfuse.permute %2, [1, 0] : (tensor<16x32xf16>) -> tensor<32x16xf16>
    %10 = mfuse.permute %1, [1, 0] : (tensor<16x16xf16>) -> tensor<16x16xf16>
    %11 = mfuse.matmul %3, %10 : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>, tensor<16x16xf16>) -> tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>
    %12 = mfuse.matmul_with_bias %7, %9, %11 : (tensor<?x32xf16, #mfuse.symshape<["s0", "32"]>>, tensor<32x16xf16>, tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>) -> tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>
    %13 = mfuse.relu %12 : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>) -> tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>
    %14 = builtin.unrealized_conversion_cast %13 : tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>> to !torch.vtensor<[?,16],f16>
    %15 = mfuse.le %13, %0 : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>, tensor<f16, {is_scalar = ""}>) -> tensor<?x16xi1, #mfuse.symshape<["s0", "16"]>>
    %16 = builtin.unrealized_conversion_cast %15 : tensor<?x16xi1, #mfuse.symshape<["s0", "16"]>> to !torch.vtensor<[?,16],i1>
    %17 = mfuse.permute %3, [1, 0] : (tensor<?x16xf16, #mfuse.symshape<["s0", "16"]>>) -> tensor<16x?xf16, #mfuse.symshape<["16", "s0"]>>
    %18 = builtin.unrealized_conversion_cast %17 : tensor<16x?xf16, #mfuse.symshape<["16", "s0"]>> to !torch.vtensor<[16,?],f16>
    return %14, %8, %16, %18, %arg3, %arg1 : !torch.vtensor<[?,16],f16>, !torch.vtensor<[?,32],f16>, !torch.vtensor<[?,16],i1>, !torch.vtensor<[16,?],f16>, !torch.vtensor<[16,32],f16>, !torch.int
  }
}
