// RUN: akg-opt %s -split-input-file --npuvector-to-vector | FileCheck %s

// CHECK-LABEL: func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_0
// CHECK-SAME: (%arg0: memref<4096xf32>, %arg1: memref<4096xf32>)
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:         %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C64:.*]] = arith.constant 64 : index
// CHECK:         scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4096]] step %[[C64]] {
// CHECK:           %[[V1:.*]] = vector.transfer_read %arg0[%[[ARG2]]], %[[CST]] {in_bounds = [true]} : memref<4096xf32>, vector<64xf32>
// CHECK:           %[[V2:.*]] = vector.transfer_read %arg1[%[[ARG2]]], %[[CST_0]] {in_bounds = [true]} : memref<4096xf32>, vector<64xf32>
// CHECK:           %[[ADD:.*]] = arith.addf %[[V2]], %[[V1]] {reduction_axes = [0 : index], reduction_type = "all"} : vector<64xf32>
// CHECK:           vector.transfer_write %[[ADD]], %arg0[%[[ARG2]]] {in_bounds = [true]} : vector<64xf32>, memref<4096xf32>
// CHECK:         }
// CHECK:         return

// CHECK-LABEL: func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_1
// CHECK-SAME: (%arg0: memref<4096xf32>)
// CHECK:         %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C64:.*]] = arith.constant 64 : index
// CHECK:         scf.for %[[ARG1:.*]] = %[[C0]] to %[[C4096]] step %[[C64]] {
// CHECK:           %[[ZERO_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK:           vector.transfer_write %[[ZERO_VEC]], %arg0[%[[ARG1]]] {in_bounds = [true]} : vector<64xf32>, memref<4096xf32>
// CHECK:         }
// CHECK:         return

// CHECK-LABEL: func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_2
// CHECK-SAME: (%arg0: memref<4096xf32>, %arg1: memref<1xf32>)
// CHECK:         %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C64:.*]] = arith.constant 64 : index
// CHECK:         %[[C4096:.*]] = arith.constant 4096 : index
// CHECK:         %[[ZERO_VEC:.*]] = arith.constant dense<0.000000e+00> : vector<64xf32>
// CHECK:         %[[CST_1:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         memref.store %[[CST_1]], %arg1[%[[C0]]] : memref<1xf32>
// CHECK:         %[[ACC:.*]] = scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4096]] step %[[C64]] iter_args(%[[ARG3:.*]] = %[[ZERO_VEC]]) -> (vector<64xf32>) {
// CHECK:           %[[V:.*]] = vector.transfer_read %arg0[%[[ARG2]]], %[[CST]] {in_bounds = [true]} : memref<4096xf32>, vector<64xf32>
// CHECK:           %[[ADD:.*]] = arith.addf %[[ARG3]], %[[V]] : vector<64xf32>
// CHECK:           scf.yield %[[ADD]] : vector<64xf32>
// CHECK:         }
// CHECK:         %[[SCALAR_VEC:.*]] = vector.transfer_read %arg1[%[[C0]]], %[[CST_1]] : memref<1xf32>, vector<f32>
// CHECK:         %[[SCALAR:.*]] = vector.extractelement %[[SCALAR_VEC]][] : vector<f32>
// CHECK:         %[[REDUCED:.*]] = vector.multi_reduction <add>, %[[ACC]], %[[SCALAR]] [0] : vector<64xf32> to f32
// CHECK:         %[[BCAST:.*]] = vector.broadcast %[[REDUCED]] : f32 to vector<1xf32>
// CHECK:         %[[MASK1:.*]] = vector.create_mask %[[C1]] : vector<1xi1>
// CHECK:         vector.transfer_write %[[BCAST]], %arg1[%[[C0]]], %[[MASK1]] {in_bounds = [true]} : vector<1xf32>, memref<1xf32>
// CHECK:         return

module {
  func.func @Fused_ReduceSum_split_18237984148215155593(%arg0: memref<1x5222400xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Reduction", arch = "Ascend950PR_9599", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, mindspore_kernel, process = "aicore"} {
    %alloc = memref.alloc() : memref<4096xf32>
    %alloc_0 = memref.alloc() : memref<1xf32>
    %c5222400 = arith.constant 5222400 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x5222400xf32> into memref<5222400xf32>
    %collapse_shape_1 = memref.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>
    %c4096 = arith.constant 4096 : index
    %c4096_2 = arith.constant 4096 : index
    %cst_3 = arith.constant dense<0.000000e+00> : !npuvector<4096xf32>
    %c0_4 = arith.constant 0 : index
    %alloc_5 = memref.alloc() : memref<4096xf32>
    call @Fused_ReduceSum_split_18237984148215155593_outlined_vf_1(%alloc_5) {hivm.vector_function, no_inline} : (memref<4096xf32>) -> ()
    scf.for %arg2 = %c0 to %c5222400 step %c4096_2 {
      %c0_10 = arith.constant 0 : index
      %c4096_11 = arith.constant 4096 : index
      %c4096_12 = arith.constant 4096 : index
      %cst_13 = arith.constant 0.000000e+00 : f32
      %c0_14 = arith.constant 0 : index
      %c4096_15 = arith.constant 4096 : index
      %c4096_16 = arith.constant 4096 : index
      %c4096_17 = arith.constant 4096 : index
      %cst_18 = arith.constant 0.000000e+00 : f32
      %1 = npuvector.transfer_read %collapse_shape[%arg2] [%c4096_17] [%c4096_17], %cst_18 : memref<5222400xf32>, !npuvector<4096xf32>
      npuvector.transfer_write %1, %alloc[%c0_14] : !npuvector<4096xf32>, memref<4096xf32>
      func.call @Fused_ReduceSum_split_18237984148215155593_outlined_vf_0(%alloc_5, %alloc) {hivm.vector_function, no_inline} : (memref<4096xf32>, memref<4096xf32>) -> ()
    }
    %c4096_6 = arith.constant 4096 : index
    %c4096_7 = arith.constant 4096 : index
    %cst_8 = arith.constant 0.000000e+00 : f32
    call @Fused_ReduceSum_split_18237984148215155593_outlined_vf_2(%alloc_5, %alloc_0) {hivm.vector_function, no_inline} : (memref<4096xf32>, memref<1xf32>) -> ()
    %c0_9 = arith.constant 0 : index
    %0 = memref.load %alloc_0[%c0_9] : memref<1xf32>
    memref.store %0, %collapse_shape_1[] : memref<f32>
    return
  }
  func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_0(%arg0: memref<4096xf32>, %arg1: memref<4096xf32>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c4096_1 = arith.constant 4096 : index
    %c4096_2 = arith.constant 4096 : index
    %0 = npuvector.transfer_read %arg0[%c0_0] [%c4096] [%c4096_1], %cst : memref<4096xf32>, !npuvector<4096xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %c4096_4 = arith.constant 4096 : index
    %c4096_5 = arith.constant 4096 : index
    %1 = npuvector.transfer_read %arg1[%c0] [%c4096_4] [%c4096_5], %cst_3 : memref<4096xf32>, !npuvector<4096xf32>
    %2 = arith.addf %1, %0 {reduction_axes = [0 : index], reduction_type = "all"} : !npuvector<4096xf32>
    npuvector.transfer_write %2, %arg0[%c0_0] : !npuvector<4096xf32>, memref<4096xf32>
    return
  }
  func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_1(%arg0: memref<4096xf32>) attributes {hivm.vector_function, no_inline} {
    %cst = arith.constant dense<0.000000e+00> : !npuvector<4096xf32>
    %c0 = arith.constant 0 : index
    npuvector.transfer_write %cst, %arg0[%c0] : !npuvector<4096xf32>, memref<4096xf32>
    return
  }
  func.func private @Fused_ReduceSum_split_18237984148215155593_outlined_vf_2(%arg0: memref<4096xf32>, %arg1: memref<1xf32>) attributes {hivm.vector_function, no_inline} {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c4096 = arith.constant 4096 : index
    %c4096_0 = arith.constant 4096 : index
    %0 = npuvector.transfer_read %arg0[%c0] [%c4096] [%c4096_0], %cst : memref<4096xf32>, !npuvector<4096xf32>
    %1 = npuvector.reduction <add>, %0 : !npuvector<4096xf32> into f32
    %c0_1 = arith.constant 0 : index
    memref.store %1, %arg1[%c0_1] : memref<1xf32>
    return
  }
}
