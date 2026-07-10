// RUN: mfusion-opt %s --fuse-batch-matmul --canonicalize | FileCheck %s

module {
  // Mode 1: permute (swap last two dims) + matmul -> matmul with trans flag; permute eliminated.
  // CHECK-LABEL: func @transpose_elimination_matmul_one_permute
  func.func @transpose_elimination_matmul_one_permute(%arg0: tensor<4x2xf32>, %arg1: tensor<2x8xf32>) -> tensor<4x8xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %1 = mfuse.matmul %0, %arg1 : (tensor<2x4xf32>, tensor<2x8xf32>) -> tensor<4x8xf32>
    return %1 : tensor<4x8xf32>
    // After pass: use permute input and set trans_x1=true (trans_x2=false omitted when default).
    // CHECK-NOT: mfuse.permute
    // CHECK: mfuse.matmul {{.*}} {trans_x1 = true}
  }

  // Mode 1: both inputs from permute (swap last two) -> matmul with both trans set.
  // CHECK-LABEL: func @transpose_elimination_matmul_both_permute
  func.func @transpose_elimination_matmul_both_permute(%arg0: tensor<4x2xf32>, %arg1: tensor<8x2xf32>) -> tensor<4x8xf32> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<4x2xf32>) -> tensor<2x4xf32>
    %1 = mfuse.permute %arg1, [1, 0] : (tensor<8x2xf32>) -> tensor<2x8xf32>
    %2 = mfuse.matmul %0, %1 : (tensor<2x4xf32>, tensor<2x8xf32>) -> tensor<4x8xf32>
    return %2 : tensor<4x8xf32>
    // CHECK: mfuse.matmul {{.*}} {trans_x1 = true, trans_x2 = true}
  }

  // Mode 1: permute (swap last two dims) + matmul_with_bias -> matmul_with_bias with trans flag; permute eliminated.
  // CHECK-LABEL: func @transpose_elimination_matmul_with_bias_one_permute
  func.func @transpose_elimination_matmul_with_bias_one_permute(
      %arg0: tensor<640x640xbf16>, %arg1: tensor<4096x640xbf16>, %arg2: tensor<640xbf16>) -> tensor<4096x640xbf16> {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.matmul_with_bias %arg1, %0, %arg2
        : (tensor<4096x640xbf16>, tensor<640x640xbf16>, tensor<640xbf16>) -> tensor<4096x640xbf16>
    return %1 : tensor<4096x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: mfuse.matmul_with_bias {{.*}} {trans_x2 = true}
  }

  // Mode 1b: broadcast_to(permute(x)) on rhs -> broadcast_to(x) + trans_x2=true.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_rhs
  func.func @transpose_elimination_matmul_broadcast_permute_rhs(
      %arg0: tensor<1x4096x640xbf16>, %arg1: tensor<640x640xbf16>) -> tensor<1x4096x640xbf16> {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %2 = mfuse.matmul %arg0, %1 : (tensor<1x4096x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x4096x640xbf16>
    return %2 : tensor<1x4096x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg1 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    // CHECK: mfuse.matmul %arg0, %[[BCAST]] {trans_x2 = true}
  }

  // Mode 1b: broadcast_to(permute(x)) on rhs of matmul_with_bias -> broadcast_to(x) + trans_x2=true.
  // CHECK-LABEL: func @transpose_elimination_matmul_with_bias_broadcast_permute_rhs
  func.func @transpose_elimination_matmul_with_bias_broadcast_permute_rhs(
      %arg0: tensor<1x4096x640xbf16>, %arg1: tensor<640x640xbf16>, %arg2: tensor<640xbf16>) -> tensor<1x4096x640xbf16> {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %2 = mfuse.matmul_with_bias %arg0, %1, %arg2
        : (tensor<1x4096x640xbf16>, tensor<1x640x640xbf16>, tensor<640xbf16>) -> tensor<1x4096x640xbf16>
    return %2 : tensor<1x4096x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg1 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    // CHECK: mfuse.matmul_with_bias %arg0, %[[BCAST]], %arg2 {trans_x2 = true}
  }

  // Mode 1b: broadcast_to(permute(x)) with non-square weight (K!=N) - verify shape swap works correctly.
  // Static-shape counterpart of the symbolic-shape test:
  //   x (rhs) is [N, K] = [4096, 640]; permute swaps last two -> [K, N] = [640, 4096]
  //   broadcast adds leading batch dim -> [B, K, N] = [2, 640, 4096]; contracting dim K=640 matches lhs
  //   After optimization: broadcast original x directly to [B, N, K] = [2, 4096, 640] and set trans_x2=true.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_rhs_nonsquare
  func.func @transpose_elimination_matmul_broadcast_permute_rhs_nonsquare(
      %arg0: tensor<2x512x640xbf16>, %arg1: tensor<4096x640xbf16>) -> tensor<2x512x4096xbf16> {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<4096x640xbf16>) -> tensor<640x4096xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x4096xbf16>) -> tensor<2x640x4096xbf16>
    %2 = mfuse.matmul %arg0, %1 : (tensor<2x512x640xbf16>, tensor<2x640x4096xbf16>) -> tensor<2x512x4096xbf16>
    return %2 : tensor<2x512x4096xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg1 : (tensor<4096x640xbf16>) -> tensor<2x4096x640xbf16>
    // CHECK: mfuse.matmul %arg0, %[[BCAST]] {trans_x2 = true} : (tensor<2x512x640xbf16>, tensor<2x4096x640xbf16>) -> tensor<2x512x4096xbf16>
  }

  // Mode 1b + symbolic shape: rewritten broadcast_to must preserve swapped-back symshape.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_rhs_symbolic
  func.func @transpose_elimination_matmul_broadcast_permute_rhs_symbolic(
      %arg0: tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s1"]>>,
      %arg1: tensor<?x?xbf16, #mfuse.symshape<["s0", "s1"]>>) -> tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s0"]>> {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<?x?xbf16, #mfuse.symshape<["s0", "s1"]>>) -> tensor<?x?xbf16, #mfuse.symshape<["s1", "s0"]>>
    %1 = mfuse.broadcast_to %0 : (tensor<?x?xbf16, #mfuse.symshape<["s1", "s0"]>>) -> tensor<1x?x?xbf16, #mfuse.symshape<["1", "s1", "s0"]>>
    %2 = mfuse.matmul %arg0, %1
        : (tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s1"]>>, tensor<1x?x?xbf16, #mfuse.symshape<["1", "s1", "s0"]>>) -> tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s0"]>>
    return %2 : tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s0"]>>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg1 : (tensor<?x?xbf16, #mfuse.symshape<["s0", "s1"]>>) -> tensor<1x?x?xbf16, #mfuse.symshape<["1", "s0", "s1"]>>
    // CHECK: mfuse.matmul %arg0, %[[BCAST]] {trans_x2 = true} : (tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s1"]>>, tensor<1x?x?xbf16, #mfuse.symshape<["1", "s0", "s1"]>>) -> tensor<1x4096x?xbf16, #mfuse.symshape<["1", "4096", "s0"]>>
  }

  // Mode 1b: shared broadcast_to(permute(x)) used by multiple matmuls can still be folded.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_rhs_multi_matmul_users
  func.func @transpose_elimination_matmul_broadcast_permute_rhs_multi_matmul_users(
      %arg0: tensor<1x4096x640xbf16>, %arg1: tensor<1x512x640xbf16>, %arg2: tensor<640x640xbf16>)
      -> (tensor<1x4096x640xbf16>, tensor<1x512x640xbf16>) {
    %0 = mfuse.permute %arg2, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %2 = mfuse.matmul %arg0, %1 : (tensor<1x4096x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x4096x640xbf16>
    %3 = mfuse.matmul %arg1, %1 : (tensor<1x512x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x512x640xbf16>
    return %2, %3 : tensor<1x4096x640xbf16>, tensor<1x512x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg2 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    // CHECK-DAG: %[[MM0:.*]] = mfuse.matmul %arg0, %[[BCAST]] {trans_x2 = true}
    // CHECK-DAG: %[[MM1:.*]] = mfuse.matmul %arg1, %[[BCAST]] {trans_x2 = true}
    // CHECK: return %[[MM0]], %[[MM1]]
  }

  // Mixed operand positions: shared broadcast_to(permute(x)) can be rewritten once and feed lhs/rhs matmuls.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_multi_users_mixed_operands
  func.func @transpose_elimination_matmul_broadcast_permute_multi_users_mixed_operands(
      %arg0: tensor<640x640xbf16>, %arg1: tensor<1x640x256xbf16>, %arg2: tensor<1x128x640xbf16>)
      -> (tensor<1x640x256xbf16>, tensor<1x128x640xbf16>) {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %2 = mfuse.matmul %1, %arg1 : (tensor<1x640x640xbf16>, tensor<1x640x256xbf16>) -> tensor<1x640x256xbf16>
    %3 = mfuse.matmul %arg2, %1 : (tensor<1x128x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x128x640xbf16>
    return %2, %3 : tensor<1x640x256xbf16>, tensor<1x128x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    // CHECK-DAG: %[[MM0:.*]] = mfuse.matmul %[[BCAST]], %arg1 {trans_x1 = true}
    // CHECK-DAG: %[[MM1:.*]] = mfuse.matmul %arg2, %[[BCAST]] {trans_x2 = true}
    // CHECK: return %[[MM0]], %[[MM1]]
  }

  // Both operands: lhs/rhs shared broadcast_to(permute(x)) values can both be folded.
  // CHECK-LABEL: func @transpose_elimination_matmul_broadcast_permute_multi_users_both_operands
  func.func @transpose_elimination_matmul_broadcast_permute_multi_users_both_operands(
      %arg0: tensor<640x128xbf16>, %arg1: tensor<256x640xbf16>, %arg2: tensor<1x640x32xbf16>, %arg3: tensor<1x16x640xbf16>)
      -> (tensor<1x128x256xbf16>, tensor<1x128x32xbf16>, tensor<1x16x256xbf16>) {
    %0 = mfuse.permute %arg0, [1, 0] : (tensor<640x128xbf16>) -> tensor<128x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<128x640xbf16>) -> tensor<1x128x640xbf16>
    %2 = mfuse.permute %arg1, [1, 0] : (tensor<256x640xbf16>) -> tensor<640x256xbf16>
    %3 = mfuse.broadcast_to %2 : (tensor<640x256xbf16>) -> tensor<1x640x256xbf16>
    %4 = mfuse.matmul %1, %3 : (tensor<1x128x640xbf16>, tensor<1x640x256xbf16>) -> tensor<1x128x256xbf16>
    %5 = mfuse.matmul %1, %arg2 : (tensor<1x128x640xbf16>, tensor<1x640x32xbf16>) -> tensor<1x128x32xbf16>
    %6 = mfuse.matmul %arg3, %3 : (tensor<1x16x640xbf16>, tensor<1x640x256xbf16>) -> tensor<1x16x256xbf16>
    return %4, %5, %6 : tensor<1x128x256xbf16>, tensor<1x128x32xbf16>, tensor<1x16x256xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[RHS_BCAST:.*]] = mfuse.broadcast_to %arg1 : (tensor<256x640xbf16>) -> tensor<1x256x640xbf16>
    // CHECK: %[[LHS_BCAST:.*]] = mfuse.broadcast_to %arg0 : (tensor<640x128xbf16>) -> tensor<1x640x128xbf16>
    // CHECK-DAG: %[[MM0:.*]] = mfuse.matmul %[[LHS_BCAST]], %[[RHS_BCAST]] {trans_x1 = true, trans_x2 = true}
    // CHECK-DAG: %[[MM1:.*]] = mfuse.matmul %[[LHS_BCAST]], %arg2 {trans_x1 = true}
    // CHECK-DAG: %[[MM2:.*]] = mfuse.matmul %arg3, %[[RHS_BCAST]] {trans_x2 = true}
    // CHECK: return %[[MM0]], %[[MM1]], %[[MM2]]
  }

  // Matmul-like users: shared broadcast_to(permute(x)) can feed matmul and matmul_with_bias.
  // CHECK-LABEL: func @transpose_elimination_broadcast_permute_shared_matmul_and_matmul_with_bias
  func.func @transpose_elimination_broadcast_permute_shared_matmul_and_matmul_with_bias(
      %arg0: tensor<1x4096x640xbf16>, %arg1: tensor<1x512x640xbf16>, %arg2: tensor<640x640xbf16>, %arg3: tensor<640xbf16>)
      -> (tensor<1x4096x640xbf16>, tensor<1x512x640xbf16>) {
    %0 = mfuse.permute %arg2, [1, 0] : (tensor<640x640xbf16>) -> tensor<640x640xbf16>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    %2 = mfuse.matmul %arg0, %1 : (tensor<1x4096x640xbf16>, tensor<1x640x640xbf16>) -> tensor<1x4096x640xbf16>
    %3 = mfuse.matmul_with_bias %arg1, %1, %arg3
        : (tensor<1x512x640xbf16>, tensor<1x640x640xbf16>, tensor<640xbf16>) -> tensor<1x512x640xbf16>
    return %2, %3 : tensor<1x4096x640xbf16>, tensor<1x512x640xbf16>
    // CHECK-NOT: mfuse.permute
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %arg2 : (tensor<640x640xbf16>) -> tensor<1x640x640xbf16>
    // CHECK-DAG: %[[MM:.*]] = mfuse.matmul %arg0, %[[BCAST]] {trans_x2 = true}
    // CHECK-DAG: %[[MM_BIAS:.*]] = mfuse.matmul_with_bias %arg1, %[[BCAST]], %arg3 {trans_x2 = true}
    // CHECK: return %[[MM]], %[[MM_BIAS]]
  }

  // Mixed users: if broadcast_to(permute(x)) also feeds a non-matmul op, skip the fold conservatively.
  // CHECK-LABEL: func @do_not_fold_broadcast_permute_rhs_with_non_matmul_user
  func.func @do_not_fold_broadcast_permute_rhs_with_non_matmul_user(
      %arg0: tensor<1x4096x640xf32>, %arg1: tensor<640x640xf32>)
      -> (tensor<1x4096x640xf32>, tensor<1x640x640xf32>) {
    %0 = mfuse.permute %arg1, [1, 0] : (tensor<640x640xf32>) -> tensor<640x640xf32>
    %1 = mfuse.broadcast_to %0 : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    %2 = mfuse.matmul %arg0, %1 : (tensor<1x4096x640xf32>, tensor<1x640x640xf32>) -> tensor<1x4096x640xf32>
    %3 = mfuse.add %1, %1 : (tensor<1x640x640xf32>, tensor<1x640x640xf32>) -> tensor<1x640x640xf32>
    return %2, %3 : tensor<1x4096x640xf32>, tensor<1x640x640xf32>
    // CHECK: %[[PERM:.*]] = mfuse.permute %arg1, [1, 0] : (tensor<640x640xf32>) -> tensor<640x640xf32>
    // CHECK: %[[BCAST:.*]] = mfuse.broadcast_to %[[PERM]] : (tensor<640x640xf32>) -> tensor<1x640x640xf32>
    // CHECK: %[[MM:.*]] = mfuse.matmul %arg0, %[[BCAST]] : (tensor<1x4096x640xf32>, tensor<1x640x640xf32>) -> tensor<1x4096x640xf32>
    // CHECK: %[[ADD:.*]] = mfuse.add %[[BCAST]], %[[BCAST]] : (tensor<1x640x640xf32>, tensor<1x640x640xf32>) -> tensor<1x640x640xf32>
    // CHECK: return %[[MM]], %[[ADD]]
  }

}
