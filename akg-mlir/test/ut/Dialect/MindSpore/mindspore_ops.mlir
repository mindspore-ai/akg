module {
  func.func @Fused_Add_fusion_2159647779082411177(%arg0: tensor<1x3072xbf16>) -> tensor<1x3072xbf16> attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %0 = "mindspore.const"() <{value = dense<1.880160e+14> : tensor<1xbf16>}> : () -> tensor<1xbf16>
    %1 = "mindspore.add"(%0, %arg0){ptr_address = "recompute_Default/network-TrainStepForCogVideo/Add-op18_358914"}: (tensor<1xbf16>, tensor<1x3072xbf16>) -> tensor<1x3072xbf16>
    return %1 : tensor<1x3072xbf16>
  }
}

module {
  func.func @Fused_ArgMin_1(%arg0: tensor<1x3072xbf16>) -> tensor<1xi64>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %0 = "mindspore.argmin"(%arg0)
         {axis = array<i64: 1>, keepdims = false}
         : (tensor<1x3072xbf16>) -> tensor<1xi64>
    return %0 : tensor<1xi64>
  }
}

module {
  func.func @AdaptiveAvgPool2d_2x2(%arg0: tensor<1x3x5x7xbf16>) -> tensor<1x3x2x2xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %0 = "mindspore.adaptive_avg_pool2d"(%arg0)
         { output_size = array<i64: 2, 2> }
         : (tensor<1x3x5x7xbf16>) -> tensor<1x3x2x2xbf16>
    return %0 : tensor<1x3x2x2xbf16>
  }
}

module {
  func.func @AdaptiveAvgPool3d_3x3x4(%x: tensor<4x3x5x6x7xf32>) -> tensor<4x3x3x3x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.adaptive_avg_pool3d"(%x)
         { output_size = array<i64: 3, 3, 4> }
         : (tensor<4x3x5x6x7xf32>) -> tensor<4x3x3x3x4xf32>
    return %y : tensor<4x3x3x3x4xf32>
  }
}

module {
  func.func @LogSoftmax_DefaultAxis(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.log_softmax"(%x)
         : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}

module {
  func.func @Softmax_NegAxis(%x: tensor<2x5x7xf32>) -> tensor<2x5x7xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.softmax"(%x) {axis = -2 : i64}
         : (tensor<2x5x7xf32>) -> tensor<2x5x7xf32>
    return %y : tensor<2x5x7xf32>
  }
}

module {
  func.func @AdaptiveAvgPool1d_3(%a: tensor<1x1x9xf32>) -> tensor<1x1x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %b = "mindspore.adaptive_avg_pool1d"(%a)
         { output_size = 3 : i64 }
         : (tensor<1x1x9xf32>) -> tensor<1x1x3xf32>
    return %b : tensor<1x1x3xf32>
  }
}

module {
  func.func @Addmm_Default(%input: tensor<2x3xf32>,
                           %m1: tensor<2x4xf32>,
                           %m2: tensor<4x3xf32>) -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %out = "mindspore.addmm"(%input, %m1, %m2)
           : (tensor<2x3xf32>, tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}

module {
  func.func @Amax_WithWhereInitial(%x: tensor<2x3xf32>, %mask: tensor<2x1xi1>)
      -> tensor<2xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %m = "mindspore.amax"(%x, %mask)
         { axis = array<i64: 1>, initial = -1.0 : f64 }
         : (tensor<2x3xf32>, tensor<2x1xi1>) -> tensor<2xf32>
    return %m : tensor<2xf32>
  }
}

module {
  func.func @Amin_WithWhereInitial(%x: tensor<2x3xf32>, %mask: tensor<2x1xi1>)
      -> tensor<2xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %m = "mindspore.amin"(%x, %mask)
         { axis = array<i64: 1>, initial = 1.0e+10 : f64 }
         : (tensor<2x3xf32>, tensor<2x1xi1>) -> tensor<2xf32>
    return %m : tensor<2xf32>
  }
}

module { func.func @Fused_Any_Axis1_KeepDims(%arg0: tensor<1x3x5xi1>) -> tensor<1x1x5xi1> 
attributes {compute_capability = "", mindspore_kernel, process = "aicore"} { 
  %0 = "mindspore.any"(%arg0) {axis = array<i64: 1>, keep_dims = true} 
        : (tensor<1x3x5xi1>) -> tensor<1x1x5xi1> 
  return %0 : tensor<1x1x5xi1> } 
}

module {
  func.func @Atanh_BF16_4D(%t: tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.atanh"(%t)
         : (tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
    return %r : tensor<1x3x5x7xbf16>
  }
}

module {
  func.func @AvgPool1d_NoPadCount(%a: tensor<2x1x7xf32>) -> tensor<2x1x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %b = "mindspore.avg_pool1d"(%a)
         { kernel_size = 3 : i64,
           stride      = 2 : i64,
           padding     = 1 : i64,
           ceil_mode   = false,
           count_include_pad = false }
         : (tensor<2x1x7xf32>) -> tensor<2x1x4xf32>
    return %b : tensor<2x1x4xf32>
  }
}

module {
  func.func @AvgPool2d_Basic(%x: tensor<2x1x6x7xf32>) -> tensor<2x1x3x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.avg_pool2d"(%x)
         { kernel_size = 3 : i64,
           stride = 2 : i64,
           padding = 1 : i64,
           ceil_mode = false,
           count_include_pad = true,
           divisor_override = 0 : i64 }
         : (tensor<2x1x6x7xf32>) -> tensor<2x1x3x4xf32>
    return %y : tensor<2x1x3x4xf32>
  }
}

module {
  func.func @AvgPool3d_DivisorOverride(%a: tensor<2x2x6x6x6xf32>) -> tensor<2x2x3x3x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %b = "mindspore.avg_pool3d"(%a)
         { kernel_size = 2 : i64,
           stride = 2 : i64,
           padding = 0 : i64,
           ceil_mode = false,
           count_include_pad = false,
           divisor_override = 8 : i64 }
         : (tensor<2x2x6x6x6xf32>) -> tensor<2x2x3x3x3xf32>
    return %b : tensor<2x2x3x3x3xf32>
  }
}

module {
  func.func @Bmm_BF16(%x: tensor<1x5x7xbf16>,
                      %y: tensor<1x7x3xbf16>) -> tensor<1x5x3xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %out = "mindspore.bmm"(%x, %y)
           : (tensor<1x5x7xbf16>, tensor<1x7x3xbf16>) -> tensor<1x5x3xbf16>
    return %out : tensor<1x5x3xbf16>
  }
}

module {
  func.func @Cat_Axis1_NCHW(%a: tensor<1x2x3x4xf32>,
                            %b: tensor<1x1x3x4xf32>) -> tensor<1x3x3x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %out = "mindspore.cat"(%a, %b) {axis = 1 : i64}
           : (tensor<1x2x3x4xf32>, tensor<1x1x3x4xf32>) -> tensor<1x3x3x4xf32>
    return %out : tensor<1x3x3x4xf32>
  }
}

module {
  func.func @Clamp_Broadcast(%x: tensor<2x3xf32>,
                             %min: tensor<1x3xf32>,
                             %max: tensor<1x1xf32>) -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.clamp"(%x, %min, %max)
         { operand_segment_sizes = array<i32: 1, 1, 1> }
         : (tensor<2x3xf32>, tensor<1x3xf32>, tensor<1x1xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}

module {
  func.func @Col2Im_Static(%x: tensor<1x2x9x9xf32>) -> tensor<1x2x10x12xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %os = arith.constant dense<[10, 12]> : tensor<2xi64>
    %y = "mindspore.col2im"(%x, %os)
         { kernel_size   = array<i64: 3, 3>,
           dilation      = array<i64: 1, 1>,
           padding_value = array<i64: 0, 0>,
           stride        = array<i64: 2, 2> }
         : (tensor<1x2x9x9xf32>, tensor<2xi64>) -> tensor<1x2x10x12xf32>
    return %y : tensor<1x2x10x12xf32>
  }
}

module {
  func.func @Cosh_BF16_4D(%t: tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.cosh"(%t)
         : (tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
    return %r : tensor<1x3x5x7xbf16>
  }
}

module {
  func.func @Diagonal_2D_OffsetNeg1(%x: tensor<3x4xf32>) -> tensor<2xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.diagonal"(%x) {offset = -1 : i64}
         : (tensor<3x4xf32>) -> tensor<2xf32>
    return %y : tensor<2xf32>
  }
}

module {
  func.func @Elu_BF16(%x: tensor<2x2x3xbf16>) -> tensor<2x2x3xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.elu"(%x) {alpha = 1.0 : f64}
         : (tensor<2x2x3xbf16>) -> tensor<2x2x3xbf16>
    return %y : tensor<2x2x3xbf16>
  }
}

module {
  func.func @Embedding_Basic(%idx: tensor<3xi32>,
                             %weight: tensor<10x4xf32>) -> tensor<3x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %emb = "mindspore.embedding"(%idx, %weight)
           : (tensor<3xi32>, tensor<10x4xf32>) -> tensor<3x4xf32>
    return %emb : tensor<3x4xf32>
  }
}

module {
  func.func @Flip_Vector(%v: tensor<8xf32>) -> tensor<8xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.flip"(%v) {dims = array<i64: 0>}
         : (tensor<8xf32>) -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}

module {
  func.func @Full_F32_2x3() -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %t = "mindspore.full"()
         { size = array<i64: 2, 3>,
           fill_value = 0.0 : f64,
           dtype = f32 }
         : () -> tensor<2x3xf32>
    return %t : tensor<2x3xf32>
  }
}

module {
  func.func @FullLike_I32(%x: tensor<8xi32>) -> tensor<8xi32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.full_like"(%x)
         { fill_value = -1 : i64, dtype = i32 }
         : (tensor<8xi32>) -> tensor<8xi32>
    return %y : tensor<8xi32>
  }
}

module {
  func.func @Hardtanh_Default(%x: tensor<2x3xf32>) -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.hardtanh"(%x)
         : (tensor<2x3xf32>) -> tensor<2x3xf32>
    return %y : tensor<2x3xf32>
  }
}

module {
  func.func @IndexSelect_Axis1(%input: tensor<2x5xf32>, %idx: tensor<3xi64>)
      -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %out = "mindspore.index_select"(%input, %idx) { axis = 1 : i64 }
           : (tensor<2x5xf32>, tensor<3xi64>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}

module {
  func.func @LeakyRelu_Vector(%v: tensor<8xf32>) -> tensor<8xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.leaky_relu"(%v) { alpha = 5.0e-1 : f64 }
         : (tensor<8xf32>) -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}

module {
  func.func @Log10_Scalar(%s: tensor<f32>) -> tensor<f32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.log10"(%s) : (tensor<f32>) -> tensor<f32>
    return %o : tensor<f32>
  }
}

module {
  func.func @Log2_BF16_4D(%t: tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.log2"(%t)
         : (tensor<1x3x5x7xbf16>) -> tensor<1x3x5x7xbf16>
    return %r : tensor<1x3x5x7xbf16>
  }
}

module {
  func.func @LogicalXor_Broadcast_LastDim(%u: tensor<2x3x4xi1>, %v: tensor<1x4xi1>)
      -> tensor<2x3x4xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.logical_xor"(%u, %v)
         : (tensor<2x3x4xi1>, tensor<1x4xi1>) -> tensor<2x3x4xi1>
    return %r : tensor<2x3x4xi1>
  }
}

module {
  func.func @Mean_MultiAxis_KeepDims(%x: tensor<2x3x4x5xf32>) -> tensor<1x3x1x5xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.mean"(%x) {axis = array<i64: 0, 2>, keep_dims = true}
         : (tensor<2x3x4x5xf32>) -> tensor<1x3x1x5xf32>
    return %y : tensor<1x3x1x5xf32>
  }
}

module {
  func.func @Mm_BF16_5x7_7x3(%a: tensor<5x7xbf16>,
                             %b: tensor<7x3xbf16>) -> tensor<5x3xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %c = "mindspore.mm"(%a, %b)
         : (tensor<5x7xbf16>, tensor<7x3xbf16>) -> tensor<5x3xbf16>
    return %c : tensor<5x3xbf16>
  }
}

module {
  func.func @NonZero_Default(%x: tensor<2x3xi1>) -> tensor<?x2xi64>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %idx = "mindspore.nonzero"(%x)
           : (tensor<2x3xi1>) -> tensor<?x2xi64>
    return %idx : tensor<?x2xi64>
  }
}

module {
  func.func @Permute_2D_Swap(%x: tensor<3x4xf32>) -> tensor<4x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.permute"(%x) {axis = array<i64: 1, 0>}
         : (tensor<3x4xf32>) -> tensor<4x3xf32>
    return %y : tensor<4x3xf32>
  }
}

module {
  func.func @Prod_AllElements_F32(%x: tensor<2x3xf32>) -> tensor<f32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.prod"(%x)
         : (tensor<2x3xf32>) -> tensor<f32>
    return %y : tensor<f32>
  }
}

module {
  func.func @Rand_F64_4D() -> tensor<1x3x5x7xf64>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.rand"()
         { size = array<i64: 1, 3, 5, 7>,
           dtype = f64,
           seed = 42 : i64 }
         : () -> tensor<1x3x5x7xf64>
    return %y : tensor<1x3x5x7xf64>
  }
}

module {
  func.func @Randn_BF16_1D() -> tensor<4xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %v = "mindspore.randn"()
         { size = array<i64: 4>,
           dtype = bf16 }
         : () -> tensor<4xbf16>
    return %v : tensor<4xbf16>
  }
}

module {
  func.func @RandPerm_100_i32_seed() -> tensor<100xi32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.randperm"()
         { n = array<i64: 100>, seed = 123 : i64, dtype = i32 }
         : () -> tensor<100xi32>
    return %r : tensor<100xi32>
  }
}

module { 
  func.func @Reciprocal_2D(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> 
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} { 
  %0 = "mindspore.reciprocal"(%arg0) 
        : (tensor<2x3xf32>) -> tensor<2x3xf32> 
  return %0 : tensor<2x3xf32> 
  } 
}

module {
  func.func @ScatterAdd_2D(%x: tensor<5x3xf32>, %idx: tensor<2xi64>, %upd: tensor<2x3xf32>)
      -> tensor<5x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.scatter_add"(%x, %idx, %upd)
         : (tensor<5x3xf32>, tensor<2xi64>, tensor<2x3xf32>) -> tensor<5x3xf32>
    return %y : tensor<5x3xf32>
  }
}

module {
  func.func @SelectScatter_Axis1_Idx0(%input: tensor<2x3x4xf32>,
                                      %src:   tensor<2x4xf32>) -> tensor<2x3x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %out = "mindspore.select_scatter"(%input, %src)
           { axis = 1 : i64, index = 0 : i64 }
           : (tensor<2x3x4xf32>, tensor<2x4xf32>) -> tensor<2x3x4xf32>
    return %out : tensor<2x3x4xf32>
  }
}

module {
  func.func @Sinh_Vector(%v: tensor<8xf32>) -> tensor<8xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.sinh"(%v) : (tensor<8xf32>) -> tensor<8xf32>
    return %o : tensor<8xf32>
  }
}

module {
  func.func @SliceScatter_Step2_Axis1(%a: tensor<4x6xf32>,
                                      %b: tensor<4x3xf32>) -> tensor<4x6xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.slice_scatter"(%a, %b)
         { axis = 1 : i64, start = 0 : i64, end = 6 : i64, step = 2 : i64 }
         : (tensor<4x6xf32>, tensor<4x3xf32>) -> tensor<4x6xf32>
    return %o : tensor<4x6xf32>
  }
}

module { 
  func.func @Sort_3D_LastDim(%x: tensor<1x3x7xf32>) 
      -> (tensor<1x3x7xf32>, tensor<1x3x7xi64>) 
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} { 
  %sorted, %idx = "mindspore.sort"(%x) {axis = -1 : i64} 
                  : (tensor<1x3x7xf32>) -> (tensor<1x3x7xf32>, tensor<1x3x7xi64>) 
  return %sorted, %idx : tensor<1x3x7xf32>, tensor<1x3x7xi64> 
  } 
}

module {
  func.func @Tan_VectorF16(%v: tensor<8xf16>) -> tensor<8xf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.tan"(%v) : (tensor<8xf16>) -> tensor<8xf16>
    return %o : tensor<8xf16>
  }
}

module {
  func.func @TopK_Default(%x: tensor<2x5xf32>)
      -> (tensor<2x3xf32>, tensor<2x3xi64>)
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %values, %indices = "mindspore.topk"(%x)
                        { k = 3 : i64 }
                        : (tensor<2x5xf32>) -> (tensor<2x3xf32>, tensor<2x3xi64>)
    return %values, %indices : tensor<2x3xf32>, tensor<2x3xi64>
  }
}

module {
  func.func @Unsqueeze_Scalar(%s: tensor<bf16>) -> tensor<1xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.unsqueeze"(%s) { dim = 0 : i64 }
         : (tensor<bf16>) -> tensor<1xbf16>
    return %o : tensor<1xbf16>
  }
}

module {
  func.func @BitwiseAnd_Scalar(%sa: tensor<i32>, %sb: tensor<i32>) -> tensor<i32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.bitwise_and"(%sa, %sb) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %r : tensor<i32>
  }
}

module {
  func.func @BitwiseOr_I32_2D(%a: tensor<2x3xi32>, %b: tensor<2x3xi32>) -> tensor<2x3xi32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.bitwise_or"(%a, %b)
         : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    return %y : tensor<2x3xi32>
  }
}

module {
  func.func @BitwiseXor_Scalar(%sa: tensor<i64>, %sb: tensor<i64>) -> tensor<i64>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.bitwise_xor"(%sa, %sb) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    return %r : tensor<i64>
  }
}

module {
  func.func @Eq_Bool(%p: tensor<4x4xi1>, %q: tensor<1x4xi1>) -> tensor<4x4xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.eq"(%p, %q)
         : (tensor<4x4xi1>, tensor<1x4xi1>) -> tensor<4x4xi1>
    return %r : tensor<4x4xi1>
  }
}

module {
  func.func @Fmod_BF16_NCHW(%t: tensor<1x2x5x7xbf16>, %d: tensor<1x1x1x1xbf16>) -> tensor<1x2x5x7xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.fmod"(%t, %d)
         : (tensor<1x2x5x7xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x2x5x7xbf16>
    return %y : tensor<1x2x5x7xbf16>
  }
}

module {
  func.func @Ge_BF16_HighRank(%t: tensor<1x3x5x7xbf16>, %u: tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.ge"(%t, %u)
         : (tensor<1x3x5x7xbf16>, tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
    return %r : tensor<1x3x5x7xi1>
  }
}

module {
  func.func @Le_Bool(%p: tensor<4x4xi1>, %q: tensor<4x4xi1>) -> tensor<4x4xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.le"(%p, %q)
         : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
    return %y : tensor<4x4xi1>
  }
}

module {
  func.func @Lt_Bool(%p: tensor<4x4xi1>, %q: tensor<4x4xi1>) -> tensor<4x4xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.lt"(%p, %q)
         : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
    return %y : tensor<4x4xi1>
  }
}

module {
  func.func @Gt_BF16_HighRank(%t: tensor<1x3x5x7xbf16>, %u: tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.gt"(%t, %u)
         : (tensor<1x3x5x7xbf16>, tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
    return %r : tensor<1x3x5x7xi1>
  }
}

module {
  func.func @Ne_BF16_HighRank(%t: tensor<1x3x5x7xbf16>, %u: tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %r = "mindspore.ne"(%t, %u)
         : (tensor<1x3x5x7xbf16>, tensor<1x1x1x7xbf16>) -> tensor<1x3x5x7xi1>
    return %r : tensor<1x3x5x7xi1>
  }
}

module {
  func.func @Remainder_BF16_NCHW(%t: tensor<1x2x5x7xbf16>, %d: tensor<1x1x1x1xbf16>) -> tensor<1x2x5x7xbf16>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.remainder"(%t, %d)
         : (tensor<1x2x5x7xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x2x5x7xbf16>
    return %y : tensor<1x2x5x7xbf16>
  }
}

module {
  func.func @Arange_NegStep_I32() -> tensor<?xi32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.arange"()
         { start = 5 : i64, end = -1 : i64, step = -2 : i64, dtype = i32 }
         : () -> tensor<?xi32>
    return %y : tensor<?xi32>
  }
}

module {
  func.func @Fill_F32_2x3() -> tensor<2x3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %shape = arith.constant dense<[2, 3]> : tensor<2xi64>
    %val   = arith.constant dense<1.25>   : tensor<f32>
    %out = "mindspore.fill"(%shape, %val) { type = f32 }
           : (tensor<2xi64>, tensor<f32>) -> tensor<2x3xf32>
    return %out : tensor<2x3xf32>
  }
}

module {
  func.func @Squeeze_Axes_HeadTail(%x: tensor<1x5x1xf32>) -> tensor<5xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.squeeze"(%x) { axis = array<i64: 0, 2> }
         : (tensor<1x5x1xf32>) -> tensor<5xf32>
    return %y : tensor<5xf32>
  }
}

module {
  func.func @Var_Axes12_Defaults(%x: tensor<3x4x5xf32>) -> tensor<3xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %y = "mindspore.var"(%x) { axis = array<i64: 1, 2> }
         : (tensor<3x4x5xf32>) -> tensor<3xf32>
    return %y : tensor<3xf32>
  }
}

module {
  func.func @Where_MultiBroadcast(%cond: tensor<2x1xi1>,
                                  %inp: tensor<2x4xf32>,
                                  %oth: tensor<1x4xf32>) -> tensor<2x4xf32>
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %o = "mindspore.where"(%cond, %inp, %oth)
         : (tensor<2x1xi1>, tensor<2x4xf32>, tensor<1x4xf32>) -> tensor<2x4xf32>
    return %o : tensor<2x4xf32>
  }
}

module {
  func.func @Max_BF16_HighRank(%t: tensor<1x3x5x7xbf16>) -> (tensor<1x3x5xbf16>, tensor<1x3x5xi64>)
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %m, %i = "mindspore.max"(%t) { axis = -1 : i64, keepdims = true }
         : (tensor<1x3x5x7xbf16>) -> (tensor<1x3x5xbf16>, tensor<1x3x5xi64>)
    return %m, %i : tensor<1x3x5xbf16>, tensor<1x3x5xi64>
  }
}

module {
  func.func @Min_BF16_HighRank(%t: tensor<1x3x5x7xbf16>) -> (tensor<1x3x5xbf16>, tensor<1x3x5xi64>)
      attributes {compute_capability = "", mindspore_kernel, process = "aicore"} {
    %m, %i = "mindspore.min"(%t) { axis = -1 : i64, keepdims = true }
         : (tensor<1x3x5x7xbf16>) -> (tensor<1x3x5xbf16>, tensor<1x3x5xi64>)
    return %m, %i : tensor<1x3x5xbf16>, tensor<1x3x5xi64>
  }
}












