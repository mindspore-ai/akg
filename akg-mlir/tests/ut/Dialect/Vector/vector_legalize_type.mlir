// RUN: akg-opt %s --vector-legalize-type | FileCheck %s

module attributes {hivm.module_core_type = #hivm.module_core_type<AIV>} {
  memref.global "private" constant @__constant_1xf32 : memref<f32> = dense<1.000000e+00> {alignment = 64 : i64}
  func.func @Fused_Cast_Sub_fusion_8883087981125364622(%arg0: memref<1xi8> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) attributes {OperatorType = "Default", arch = "Ascend950PR_9599", compute_capability = "", enable_auto_mark_buffer_size, hacc.block_dim = 1 : i64, hacc.entry, hacc.function_kind = #hacc.function_kind<DEVICE>, hivm.func_core_type = #hivm.func_core_type<AIV>, mindspore_kernel, process = "aicore"} {
    %alloc = memref.alloc() : memref<64xf32>
    %alloc_0 = memref.alloc() : memref<256xi8>
    %collapse_shape = memref.collapse_shape %arg0 [] : memref<1xi8> into memref<i8>
    %collapse_shape_1 = memref.collapse_shape %arg1 [] : memref<1xf32> into memref<f32>
    %0 = memref.load %collapse_shape[] : memref<i8>
    %subview = memref.subview %alloc_0[0] [1] [1] : memref<256xi8> to memref<1xi8, strided<[1]>>
    hivm.hir.vbrc ins(%0 : i8) outs(%subview : memref<1xi8, strided<[1]>>)
    call @Fused_Cast_Sub_fusion_8883087981125364622_outlined_vf_0(%alloc_0, %alloc) {hivm.vector_function, no_inline} : (memref<256xi8>, memref<64xf32>) -> ()
    %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1], strides: [1] : memref<64xf32> to memref<1xf32>
    %reinterpret_cast_2 = memref.reinterpret_cast %collapse_shape_1 to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32>
    hivm.hir.store ins(%reinterpret_cast : memref<1xf32>) outs(%reinterpret_cast_2 : memref<1xf32>)
    return
  }
  func.func private @Fused_Cast_Sub_fusion_8883087981125364622_outlined_vf_0(%arg0: memref<256xi8>, %arg1: memref<64xf32>) attributes {hivm.func_core_type = #hivm.func_core_type<AIV>, hivm.vector_function, no_inline} {
    %cst = arith.constant dense<1.000000e+00> : vector<64xf32>
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %0 = vector.constant_mask [1] : vector<64xi1>
    %1 = vector.transfer_read %arg0[%c0], %c0_i8, %0 {in_bounds = [true]} : memref<256xi8>, vector<64xi8>
    // CHECK-LABEL: @Fused_Cast_Sub_fusion_8883087981125364622_outlined_vf_0
    // CHECK: vector.transfer_read
    // CHECK-NEXT: arith.extui {{%.*}} : vector<64xi8> to vector<64xi16>
    // CHECK-NEXT: arith.uitofp {{%.*}} : vector<64xi16> to vector<64xf32>
    %2 = arith.uitofp %1 : vector<64xi8> to vector<64xf32>
    %3 = arith.subf %cst, %2 : vector<64xf32>
    vector.transfer_write %3, %arg1[%c0], %0 {in_bounds = [true]} : vector<64xf32>, memref<64xf32>
    return
  }
}
