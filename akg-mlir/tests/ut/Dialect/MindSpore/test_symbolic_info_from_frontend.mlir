// RUN: akg-opt %s --split-input-file --infer-symbolic-shapes | FileCheck %s
// CHECK: SymShapeAttr = ["s66", "1024"]
// CHECK: SymShapeAttr = ["s67", "1024"]
// CHECK: SymShapeAttr = ["s66", "1024"]
// CHECK：SymShapeAttr = ["32", "1", "s426", "s427"]
// CHECK：SymShapeAttr = ["32", "16", "s428", "s429"]
// CHECK：SymShapeAttr = ["32", "1", "s426", "s427"]
// CHECK-NOT：frontend_symbol

module {
  func.func @Fused_Add_Cast_fusion_10992272294568024314(%arg0: tensor<?x1024xf16>, %arg1: tensor<?x1024xf16>) -> tensor<?x1024xf32> attributes {frontend_symbol = {input_0 = ["s66", "1024"], input_1 = ["s67", "1024"], output_0 = ["s66", "1024"]}, mindspore_kernel, process = "cuda"} {
    %0 = "mindspore.add"(%arg0, %arg1) {frontend_symbol = {input_0 = ["s66", "1024"], input_1 = ["s67", "1024"], output_0 = ["s66", "1024"]}, ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Default/Add-op9196_651098"} : (tensor<?x1024xf16>, tensor<?x1024xf16>) -> tensor<?x1024xf16>
    %1 = "mindspore.cast"(%0) {frontend_symbol = {input_0 = ["s66", "1024"], output_0 = ["s66", "1024"]}, ms_attr = {DstT = "float32", SrcT = "float16", dst_type = "float32", input_is_dynamic_shape = true, is_backend_cast = false, output_is_dynamic_shape = true}, ptr_address = "Gradients/Default/network/network/transformer/projection/gradCast-expand/Cast-op8179_651099"} : (tensor<?x1024xf16>) -> tensor<?x1024xf32>
    return %1 : tensor<?x1024xf32>
  }
  func.func @Fused_Sub_Mul_Mul_Add_Cast_fusion_4041712636908909135(%arg0: tensor<32x1x?x?xf16>, %arg1: tensor<32x16x?x?xf16>) -> (tensor<32x1x?x?xf16>, tensor<32x16x?x?xf16>, tensor<32x16x?x?xf32>) attributes {frontend_symbol = {input_0 = ["32", "1", "s426", "s427"], input_1 = ["32", "16", "s428", "s429"], output_0 = ["32", "1", "s426", "s427"], output_1 = ["32", "16", "s428", "s429"], output_2 = ["32", "16", "s426", "s427"]}, mindspore_kernel, process = "cuda"} {
    %0 = "mindspore.const"() {value = dense<1.000000e+00> : tensor<1xf16>} : () -> tensor<1xf16>
    %1 = "mindspore.sub"(%0, %arg0) {frontend_symbol = {input_0 = ["1"], input_1 = ["32", "1", "s426", "s427"], output_0 = ["32", "1", "s426", "s427"]}, ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Default/network/network/transformer/tfm_encoder/layers/5/attention/attention/Sub-op4174_649203"} : (tensor<1xf16>, tensor<32x1x?x?xf16>) -> tensor<32x1x?x?xf16>
    %2 = "mindspore.const"() {value = dense<-1.000000e+04> : tensor<1xf16>} : () -> tensor<1xf16>
    %3 = "mindspore.mul"(%1, %2) {frontend_symbol = {input_0 = ["32", "1", "s426", "s427"], input_1 = ["1"], output_0 = ["32", "1", "s426", "s427"]}, ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Default/network/network/transformer/tfm_encoder/layers/5/attention/attention/Mul-op4175_649204"} : (tensor<32x1x?x?xf16>, tensor<1xf16>) -> tensor<32x1x?x?xf16>
    %4 = "mindspore.const"() {value = dense<1.250000e-01> : tensor<1xf16>} : () -> tensor<1xf16>
    %5 = "mindspore.mul"(%arg1, %4) {frontend_symbol = {input_0 = ["32", "16", "s428", "s429"], input_1 = ["1"], output_0 = ["32", "16", "s428", "s429"]}, ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Default/network/network/transformer/tfm_encoder/layers/0/attention/attention/Mul-op4219_649206"} : (tensor<32x16x?x?xf16>, tensor<1xf16>) -> tensor<32x16x?x?xf16>
    %6 = "mindspore.add"(%3, %5) {frontend_symbol = {input_0 = ["32", "1", "s426", "s427"], input_1 = ["32", "16", "s428", "s429"], output_0 = ["32", "16", "s426", "s427"]}, ms_attr = {input_is_dynamic_shape = true, output_is_dynamic_shape = true}, ptr_address = "Default/network/network/transformer/tfm_encoder/layers/0/attention/attention/Add-op4220_649207"} : (tensor<32x1x?x?xf16>, tensor<32x16x?x?xf16>) -> tensor<32x16x?x?xf16>
    %7 = "mindspore.cast"(%6) {frontend_symbol = {input_0 = ["32", "16", "s426", "s427"], output_0 = ["32", "16", "s426", "s427"]}, ms_attr = {DstT = "float32", SrcT = "float16", dst_type = "float32", input_is_dynamic_shape = true, is_backend_cast = false, output_is_dynamic_shape = true}, ptr_address = "Default/network/network/transformer/tfm_encoder/layers/0/attention/attention/Cast-op7812_649208"} : (tensor<32x16x?x?xf16>) -> tensor<32x16x?x?xf32>
    return %3, %5, %7 : tensor<32x1x?x?xf16>, tensor<32x16x?x?xf16>, tensor<32x16x?x?xf32>
  }
}