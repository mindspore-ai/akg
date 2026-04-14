// RUN: mfusion-opt %s --convert-mfuse-to-dvm | FileCheck %s

module {
  // CHECK-LABEL: func @main_mul_fused_0
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.mul
  func.func @main_mul_fused_0(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_mul_fused_1
  // CHECK: mfuse.mul
  // CHECK-NOT: dvm.load
  // CHECK-NOT: dvm.binary
  // CHECK-NOT: dvm.store
  func.func @main_mul_fused_1(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "akg"} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_add_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.add
  func.func @main_add_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.add %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_sub_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[SUB:.*]] = dvm.binary Sub %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[SUB]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.sub
  func.func @main_sub_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.sub %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_div_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[DIV:.*]] = dvm.binary Div %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[DIV]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.div
  func.func @main_div_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.div %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_abs_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[ABS:.*]] = dvm.unary Abs %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ABS]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.abs
  func.func @main_abs_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.abs %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_sqrt_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[SQRT:.*]] = dvm.unary Sqrt %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[SQRT]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.sqrt
  func.func @main_sqrt_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.sqrt %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_exp_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[EXP:.*]] = dvm.unary Exp %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[EXP]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.exp
  func.func @main_exp_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.exp %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_log_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOG:.*]] = dvm.unary Log %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[LOG]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.log
  func.func @main_log_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.log %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_constant_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<1.000000e+00> : tensor<f32>
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[LOADA]], %[[CONST]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.add
  func.func @main_constant_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<1.0> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.add %a, %0 : (tensor<2xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_constant_i32_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xi32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xi32> -> tensor<2xi32>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<42> : tensor<i32>
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[LOADA]], %[[CONST]] : tensor<2xi32>, tensor<2xi32> -> tensor<2xi32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]] : tensor<2xi32> -> tensor<2xi32>
  // CHECK: return %[[STORE]] : tensor<2xi32>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.add
  func.func @main_constant_i32_fused(%a: tensor<2xi32>) -> tensor<2xi32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<42> : tensor<i32, {is_scalar = ""}>
    %1 = mfuse.add %a, %0 : (tensor<2xi32>, tensor<i32, {is_scalar = ""}>) -> tensor<2xi32>
    return %1 : tensor<2xi32>
  }

  // CHECK-LABEL: func @main_constant_i64_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xi64>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xi64> -> tensor<2xi64>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<100> : tensor<i64>
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[CONST]] : tensor<2xi64>, tensor<2xi64> -> tensor<2xi64>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]] : tensor<2xi64> -> tensor<2xi64>
  // CHECK: return %[[STORE]] : tensor<2xi64>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.mul
  func.func @main_constant_i64_fused(%a: tensor<2xi64>) -> tensor<2xi64> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<100> : tensor<i64, {is_scalar = ""}>
    %1 = mfuse.mul %a, %0 : (tensor<2xi64>, tensor<i64, {is_scalar = ""}>) -> tensor<2xi64>
    return %1 : tensor<2xi64>
  }

  // CHECK-LABEL: func @main_constant_f16_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf16>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf16> -> tensor<2xf16>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<2.5> : tensor<f16>
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[CONST]] : tensor<2xf16>, tensor<2xf16> -> tensor<2xf16>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]] : tensor<2xf16> -> tensor<2xf16>
  // CHECK: return %[[STORE]] : tensor<2xf16>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.mul
  func.func @main_constant_f16_fused(%a: tensor<2xf16>) -> tensor<2xf16> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<2.5> : tensor<f16, {is_scalar = ""}>
    %1 = mfuse.mul %a, %0 : (tensor<2xf16>, tensor<f16, {is_scalar = ""}>) -> tensor<2xf16>
    return %1 : tensor<2xf16>
  }

  // CHECK-LABEL: func @main_constant_f64_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf64>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf64> -> tensor<2xf64>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<3.1415926535897931> : tensor<f64>
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[CONST]] : tensor<2xf64>, tensor<2xf64> -> tensor<2xf64>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]] : tensor<2xf64> -> tensor<2xf64>
  // CHECK: return %[[STORE]] : tensor<2xf64>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.mul
  func.func @main_constant_f64_fused(%a: tensor<2xf64>) -> tensor<2xf64> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<3.141592653589793> : tensor<f64, {is_scalar = ""}>
    %1 = mfuse.mul %a, %0 : (tensor<2xf64>, tensor<f64, {is_scalar = ""}>) -> tensor<2xf64>
    return %1 : tensor<2xf64>
  }

  // CHECK-LABEL: func @main_constant_negative_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<-5.500000e+00> : tensor<f32>
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[LOADA]], %[[CONST]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.constant
  // CHECK-NOT: mfuse.add
  func.func @main_constant_negative_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.constant dense<-5.5> : tensor<f32, {is_scalar = ""}>
    %1 = mfuse.add %a, %0 : (tensor<2xf32>, tensor<f32, {is_scalar = ""}>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_neg_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[NEG:.*]] = dvm.binary Sub %[[CONST]], %[[LOADA]] : tensor<f32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[NEG]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.neg
  func.func @main_neg_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.neg %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_rsqrt_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[SQRT:.*]] = dvm.unary Sqrt %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[RECIP:.*]] = dvm.unary Reciprocal %[[SQRT]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[RECIP]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.rsqrt
  func.func @main_rsqrt_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.rsqrt %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_reciprocal_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[RECIP:.*]] = dvm.unary Reciprocal %[[LOADA]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[RECIP]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.reciprocal
  func.func @main_reciprocal_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.reciprocal %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_cast_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[CAST:.*]] = dvm.cast %[[LOADA]] type Int32 : tensor<2xf32> -> tensor<2xi32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[CAST]] : tensor<2xi32> -> tensor<2xi32>
  // CHECK: return %[[STORE]] : tensor<2xi32>
  // CHECK-NOT: mfuse.cast
  func.func @main_cast_fused(%a: tensor<2xf32>) -> tensor<2xi32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.cast %a : (tensor<2xf32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }

  // CHECK-LABEL: func @main_broadcast_to_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[BCAST:.*]] = dvm.broadcast %[[LOADA]] shape [4, 2] : tensor<2xf32> -> tensor<4x2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[BCAST]] : tensor<4x2xf32> -> tensor<4x2xf32>
  // CHECK: return %[[STORE]] : tensor<4x2xf32>
  // CHECK-NOT: mfuse.broadcast_to
  func.func @main_broadcast_to_fused(%a: tensor<2xf32>) -> tensor<4x2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.broadcast_to %a {shape = array<i64: 4, 2>} : (tensor<2xf32>) -> tensor<4x2xf32>
    return %0 : tensor<4x2xf32>
  }

  // CHECK-LABEL: func @main_reshape_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<4x2xf32> -> tensor<4x2xf32>
  // CHECK: %[[RESHAPE:.*]] = dvm.reshape %[[LOADA]] shape [8] : tensor<4x2xf32> -> tensor<8xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[RESHAPE]] : tensor<8xf32> -> tensor<8xf32>
  // CHECK: return %[[STORE]] : tensor<8xf32>
  // CHECK-NOT: mfuse.reshape
  func.func @main_reshape_fused(%a: tensor<4x2xf32>) -> tensor<8xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.reshape %a : (tensor<4x2xf32>) -> tensor<8xf32>
    return %0 : tensor<8xf32>
  }

  // CHECK-LABEL: func @main_reduce_sum_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<4x2xf32> -> tensor<4x2xf32>
  // CHECK: %[[REDUCE:.*]] = dvm.reduce Sum %[[LOADA]] dims [1] keepdims false : tensor<4x2xf32> -> tensor<4xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[REDUCE]] : tensor<4xf32> -> tensor<4xf32>
  // CHECK: return %[[STORE]] : tensor<4xf32>
  // CHECK-NOT: mfuse.reduce_sum
  func.func @main_reduce_sum_fused(%a: tensor<4x2xf32>) -> tensor<4xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.reduce_sum %a {dimensions = [1], keepdim = false} : (tensor<4x2xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }

  // CHECK-LABEL: func @main_relu_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[CONST:.*]] = dvm.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RELU:.*]] = dvm.binary Maximum %[[LOADA]], %[[CONST]] : tensor<2xf32>, tensor<f32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[RELU]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.relu
  func.func @main_relu_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.relu %a : (tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_real_div_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[DIV:.*]] = dvm.binary Div %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[DIV]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.real_div
  func.func @main_real_div_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.real_div %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_select_fused
  // CHECK-SAME: (%[[COND:.*]]: tensor<2xi1>, %[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADCOND:.*]] = dvm.load %[[COND]] : tensor<2xi1> -> tensor<2xi1>
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[SELECT:.*]] = dvm.select %[[LOADCOND]], %[[LOADA]], %[[LOADB]] : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[SELECT]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.select
  func.func @main_select_fused(%cond: tensor<2xi1>, %a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.select %cond, %a, %b : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_matmul_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<4x8xf32> -> tensor<4x8xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<8x16xf32> -> tensor<8x16xf32>
  // CHECK: %[[MATMUL:.*]] = dvm.matmul %[[LOADA]], %[[LOADB]] trans_a false trans_b false : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MATMUL]] : tensor<4x16xf32> -> tensor<4x16xf32>
  // CHECK: return %[[STORE]] : tensor<4x16xf32>
  // CHECK-NOT: mfuse.matmul
  func.func @main_matmul_fused(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.matmul %a, %b {trans_x1 = false, trans_x2 = false} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  // CHECK-LABEL: func @main_maximum_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[MAX:.*]] = dvm.binary Maximum %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MAX]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.maximum
  func.func @main_maximum_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.maximum %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_minimum_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[MIN:.*]] = dvm.binary Minimum %[[LOADA]], %[[LOADB]] : tensor<2xf32>, tensor<2xf32> -> tensor<2xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MIN]] : tensor<2xf32> -> tensor<2xf32>
  // CHECK: return %[[STORE]] : tensor<2xf32>
  // CHECK-NOT: mfuse.minimum
  func.func @main_minimum_fused(%a: tensor<2xf32>, %b: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.minimum %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_batch_matmul_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2x4x8xf32>, %[[B:.*]]: tensor<2x8x16xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]] : tensor<2x4x8xf32> -> tensor<2x4x8xf32>
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]] : tensor<2x8x16xf32> -> tensor<2x8x16xf32>
  // CHECK: %[[MATMUL:.*]] = dvm.matmul %[[LOADA]], %[[LOADB]] trans_a false trans_b false : tensor<2x4x8xf32>, tensor<2x8x16xf32> -> tensor<2x4x16xf32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[MATMUL]] : tensor<2x4x16xf32> -> tensor<2x4x16xf32>
  // CHECK: return %[[STORE]] : tensor<2x4x16xf32>
  // CHECK-NOT: mfuse.batch_matmul
  func.func @main_batch_matmul_fused(%a: tensor<2x4x8xf32>, %b: tensor<2x8x16xf32>) -> tensor<2x4x16xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.batch_matmul %a, %b : (tensor<2x4x8xf32>, tensor<2x8x16xf32>) -> tensor<2x4x16xf32>
    return %0 : tensor<2x4x16xf32>
  }

  // CHECK-LABEL: func @main_mul_add_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>, %[[C:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[LOADC:.*]] = dvm.load %[[C]]
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[LOADB]]
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[MUL]], %[[LOADC]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]]
  // CHECK: return %[[STORE]]
  // CHECK-NOT: mfuse.mul
  // CHECK-NOT: mfuse.add
  func.func @main_mul_add_fused(%a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.mul %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = mfuse.add %0, %c : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_relu_matmul_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[CONST:.*]] = dvm.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[MATMUL:.*]] = dvm.matmul %[[LOADA]], %[[LOADB]]
  // CHECK: %[[RELU:.*]] = dvm.binary Maximum %[[MATMUL]], %[[CONST]] : tensor<4x16xf32>, tensor<f32>
  // CHECK: %[[STORE:.*]] = dvm.store %[[RELU]]
  // CHECK: return %[[STORE]]
  // CHECK-NOT: mfuse.matmul
  // CHECK-NOT: mfuse.relu
  func.func @main_relu_matmul_fused(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.matmul %a, %b {trans_x1 = false, trans_x2 = false} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %1 = mfuse.relu %0 : (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %1 : tensor<4x16xf32>
  }

  // CHECK-LABEL: func @main_sqrt_reciprocal_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[SQRT:.*]] = dvm.unary Sqrt %[[LOADA]]
  // CHECK: %[[RECIP:.*]] = dvm.unary Reciprocal %[[SQRT]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[RECIP]]
  // CHECK: return %[[STORE]]
  // CHECK-NOT: mfuse.sqrt
  // CHECK-NOT: mfuse.reciprocal
  func.func @main_sqrt_reciprocal_fused(%a: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.sqrt %a : (tensor<2xf32>) -> tensor<2xf32>
    %1 = mfuse.reciprocal %0 : (tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_cast_mul_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xi32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[CAST:.*]] = dvm.cast %[[LOADB]] type Int32
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[CAST]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]]
  // CHECK: return %[[STORE]]
  // CHECK-NOT: mfuse.cast
  // CHECK-NOT: mfuse.mul
  func.func @main_cast_mul_fused(%a: tensor<2xf32>, %b: tensor<2xi32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.cast %b : (tensor<2xi32>) -> tensor<2xf32>
    %1 = mfuse.mul %a, %0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_add_relu_maximum_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>, %[[C:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[LOADC:.*]] = dvm.load %[[C]]
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[LOADA]], %[[LOADB]]
  // CHECK: %[[CONST:.*]] = dvm.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RELU:.*]] = dvm.binary Maximum %[[ADD]], %[[CONST]] : tensor<2xf32>, tensor<f32>
  // CHECK: %[[MAX:.*]] = dvm.binary Maximum %[[RELU]], %[[LOADC]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[MAX]]
  // CHECK: return %[[STORE]]
  func.func @main_add_relu_maximum_fused(%a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.add %a, %b : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = mfuse.relu %0 : (tensor<2xf32>) -> tensor<2xf32>
    %2 = mfuse.maximum %1, %c : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %2 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_matmul_with_bias_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>, %[[BIAS:.*]]: tensor<4x16xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[LOADBIAS:.*]] = dvm.load %[[BIAS]]
  // CHECK: %[[MATMUL:.*]] = dvm.matmul %[[LOADA]], %[[LOADB]] trans_a false trans_b false bias %[[LOADBIAS]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[MATMUL]]
  // CHECK: return %[[STORE]]
  func.func @main_matmul_with_bias_fused(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %bias: tensor<4x16xf32>) -> tensor<4x16xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.matmul_with_bias %a, %b, %bias {trans_x1 = false, trans_x2 = false} : (tensor<4x8xf32>, tensor<8x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }

  // CHECK-LABEL: func @main_reduce_add_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x2xf32>, %[[B:.*]]: tensor<4xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[REDUCE:.*]] = dvm.reduce Sum %[[LOADA]] dims [1] keepdims false
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[REDUCE]], %[[LOADB]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[ADD]]
  // CHECK: return %[[STORE]]
  func.func @main_reduce_add_fused(%a: tensor<4x2xf32>, %b: tensor<4xf32>) -> tensor<4xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.reduce_sum %a {dimensions = [1], keepdim = false} : (tensor<4x2xf32>) -> tensor<4xf32>
    %1 = mfuse.add %0, %b : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
    return %1 : tensor<4xf32>
  }

  // CHECK-LABEL: func @main_select_mul_fused
  // CHECK-SAME: (%[[COND:.*]]: tensor<2xi1>, %[[A:.*]]: tensor<2xf32>, %[[B:.*]]: tensor<2xf32>, %[[C:.*]]: tensor<2xf32>)
  // CHECK: %[[LOADCOND:.*]] = dvm.load %[[COND]]
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[LOADC:.*]] = dvm.load %[[C]]
  // CHECK: %[[SELECT:.*]] = dvm.select %[[LOADCOND]], %[[LOADB]], %[[LOADC]]
  // CHECK: %[[MUL:.*]] = dvm.binary Mul %[[LOADA]], %[[SELECT]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[MUL]]
  // CHECK: return %[[STORE]]
  func.func @main_select_mul_fused(%cond: tensor<2xi1>, %a: tensor<2xf32>, %b: tensor<2xf32>, %c: tensor<2xf32>) -> tensor<2xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.select %cond, %b, %c : (tensor<2xi1>, tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    %1 = mfuse.mul %a, %0 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %1 : tensor<2xf32>
  }

  // CHECK-LABEL: func @main_complex_fused
  // CHECK-SAME: (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>, %[[C:.*]]: tensor<4x16xf32>, %[[D:.*]]: tensor<4x16xf32>)
  // CHECK: %[[LOADA:.*]] = dvm.load %[[A]]
  // CHECK: %[[LOADB:.*]] = dvm.load %[[B]]
  // CHECK: %[[LOADC:.*]] = dvm.load %[[C]]
  // CHECK: %[[LOADD:.*]] = dvm.load %[[D]]
  // CHECK: %[[MATMUL:.*]] = dvm.matmul %[[LOADA]], %[[LOADB]]
  // CHECK: %[[ADD:.*]] = dvm.binary Add %[[MATMUL]], %[[LOADC]]
  // CHECK: %[[SUB:.*]] = dvm.binary Sub %[[ADD]], %[[LOADD]]
  // CHECK: %[[CONST:.*]] = dvm.constant dense<0.000000e+00> : tensor<f32>
  // CHECK: %[[RELU:.*]] = dvm.binary Maximum %[[SUB]], %[[CONST]] : tensor<4x16xf32>, tensor<f32>
  // CHECK: %[[SQRT:.*]] = dvm.unary Sqrt %[[RELU]]
  // CHECK: %[[STORE:.*]] = dvm.store %[[SQRT]]
  // CHECK: return %[[STORE]]
  func.func @main_complex_fused(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<4x16xf32>, %d: tensor<4x16xf32>) -> tensor<4x16xf32> attributes {mfusion.outlined, mfusion.fusion_type = "dvm"} {
    %0 = mfuse.matmul %a, %b {trans_x1 = false, trans_x2 = false} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
    %1 = mfuse.add %0, %c : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    %2 = mfuse.sub %1, %d : (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
    %3 = mfuse.relu %2 : (tensor<4x16xf32>) -> tensor<4x16xf32>
    %4 = mfuse.sqrt %3 : (tensor<4x16xf32>) -> tensor<4x16xf32>
    return %4 : tensor<4x16xf32>
  }

  // CHECK-LABEL: func @main
  // CHECK: %[[CALL:.*]] = call @main_mul_fused_0(%[[ARG0:.*]], %[[ARG1:.*]])
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = call @main_mul_fused_0(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
