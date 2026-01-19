// RUN: akg-opt %s --add-out-parameter | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:  func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<2>}, %arg3: memref<1x1666x1024xf32> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:    affine.for %arg4 = 0 to 1536 step 512 {
// CHECK-NEXT:      affine.for %arg5 = 0 to 1024 step 512 {
// CHECK-NEXT:        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
// CHECK-NEXT:          affine.for %arg7 = #map(%arg5) to #map1(%arg5) {
// CHECK-NEXT:            %0 = affine.load %arg0[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %1 = affine.load %arg1[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:            %3 = affine.load %arg2[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:            %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:            affine.store %4, %arg3[0, %arg6, %arg7] : memref<1x1666x1024xf32>
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    affine.for %arg4 = 0 to 1024 step 512 {
// CHECK-NEXT:      affine.for %arg5 = 1536 to 1666 {
// CHECK-NEXT:        affine.for %arg6 = #map(%arg4) to #map1(%arg4) {
// CHECK-NEXT:          %0 = affine.load %arg0[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %1 = affine.load %arg1[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %2 = arith.subf %0, %1 : f32
// CHECK-NEXT:          %3 = affine.load %arg2[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:          %4 = arith.addf %2, %3 : f32
// CHECK-NEXT:          affine.store %4, %arg3[0, %arg5, %arg6] : memref<1x1666x1024xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return %arg3 : memref<1x1666x1024xf32>
// CHECK-NEXT:  }
// CHECK-NEXT: func.func @reshape_from_temp_buffer(%arg0: memref<1x512xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x512xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<512xbf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}) -> memref<512xbf16> attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT:  %c512 = arith.constant 512 : index
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %alloc = memref.alloc() : memref<2xindex>
// CHECK-NEXT:  %c1_0 = arith.constant 1 : index
// CHECK-NEXT:  %c0_1 = arith.constant 0 : index
// CHECK-NEXT:  memref.store %c1_0, %alloc[%c0_1] : memref<2xindex>
// CHECK-NEXT:  %c512_2 = arith.constant 512 : index
// CHECK-NEXT:  %c1_3 = arith.constant 1 : index
// CHECK-NEXT:  memref.store %c512_2, %alloc[%c1_3] : memref<2xindex>
// CHECK-NEXT:  %reshape = memref.reshape %arg2(%alloc) : (memref<512xbf16>, memref<2xindex>) -> memref<1x512xbf16>
// CHECK-NEXT:  scf.for %arg3 = %c0 to %c512 step %c1 {
// CHECK-NEXT:    %c0_6 = arith.constant 0 : index
// CHECK-NEXT:    %0 = memref.load %arg0[%c0_6, %arg3] : memref<1x512xbf16>
// CHECK-NEXT:    %1 = memref.load %arg1[%c0_6, %arg3] : memref<1x512xbf16>
// CHECK-NEXT:    %2 = arith.addf %0, %1 : bf16
// CHECK-NEXT:    memref.store %2, %reshape[%c0_6, %arg3] : memref<1x512xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  %c0_4 = arith.constant 0 : index
// CHECK-NEXT:  %c512_5 = arith.constant 512 : index
// CHECK-NEXT:  return %arg2 : memref<512xbf16>
// CHECK-NEXT: }
// CHECK-NEXT:  func.func @reshape_from_output_buffer(%arg0: memref<1x512xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<0>}, %arg1: memref<1x512xbf16> {hacc.arg_type = #hacc.arg_type<input>, hacc.input_idx = #hacc.input_idx<1>}, %arg2: memref<1x512xbf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<0>}, %arg3: memref<512xbf16> {hacc.arg_type = #hacc.arg_type<output>, hacc.output_idx = #hacc.output_idx<1>}) -> (memref<1x512xbf16>, memref<512xbf16>) attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
// CHECK-NEXT:  %alloc = memref.alloc() : memref<2xindex>
// CHECK-NEXT:  %c1 = arith.constant 1 : index
// CHECK-NEXT:  %c0 = arith.constant 0 : index
// CHECK-NEXT: memref.store %c1, %alloc[%c0] : memref<2xindex>
// CHECK-NEXT:  %c512 = arith.constant 512 : index
// CHECK-NEXT:  %c1_0 = arith.constant 1 : index
// CHECK-NEXT:  memref.store %c512, %alloc[%c1_0] : memref<2xindex>
// CHECK-NEXT:  %reshape = memref.reshape %arg3(%alloc) : (memref<512xbf16>, memref<2xindex>) -> memref<1x512xbf16>
// CHECK-NEXT:  %c0_1 = arith.constant 0 : index
// CHECK-NEXT:  %c512_2 = arith.constant 512 : index
// CHECK-NEXT:  %c1_3 = arith.constant 1 : index
// CHECK-NEXT:  scf.for %arg4 = %c0_1 to %c512_2 step %c1_3 {
// CHECK-NEXT:    %c0_6 = arith.constant 0 : index
// CHECK-NEXT:    %0 = memref.load %arg0[%c0_6, %arg4] : memref<1x512xbf16>
// CHECK-NEXT:    memref.store %0, %arg2[%c0_6, %arg4] : memref<1x512xbf16>
// CHECK-NEXT:    memref.store %0, %reshape[%c0_6, %arg4] : memref<1x512xbf16>
// CHECK-NEXT:  }
// CHECK-NEXT:  %c0_4 = arith.constant 0 : index
// CHECK-NEXT:  %c512_5 = arith.constant 512 : index
// CHECK-NEXT: return %arg2, %arg3 : memref<1x512xbf16>, memref<512xbf16>
// CHECK-NEXT: }
// CHECK-NEXT:}

func.func @Fused_Sub_Add_fusion_97330263758862449(%arg0: memref<1x1666x1024xf32>, %arg1: memref<1x1666x1024xf32>, %arg2: memref<1x1666x1024xf32>) -> memref<1x1666x1024xf32> attributes {OperatorType = "Elementwise", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1666x1024xf32>
  affine.for %arg3 = 0 to 1536 step 512 {
    affine.for %arg4 = 0 to 1024 step 512 {
      affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 512)>(%arg3) {
        affine.for %arg6 = affine_map<(d0) -> (d0)>(%arg4) to affine_map<(d0) -> (d0 + 512)>(%arg4) {
          %0 = affine.load %arg0[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %1 = affine.load %arg1[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %2 = arith.subf %0, %1 : f32
          %3 = affine.load %arg2[0, %arg5, %arg6] : memref<1x1666x1024xf32>
          %4 = arith.addf %2, %3 : f32
          affine.store %4, %alloc[0, %arg5, %arg6] : memref<1x1666x1024xf32>
        }
      }
    }
  }
  affine.for %arg3 = 0 to 1024 step 512 {
    affine.for %arg4 = 1536 to 1666 {
      affine.for %arg5 = affine_map<(d0) -> (d0)>(%arg3) to affine_map<(d0) -> (d0 + 512)>(%arg3) {
        %0 = affine.load %arg0[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %1 = affine.load %arg1[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %2 = arith.subf %0, %1 : f32
        %3 = affine.load %arg2[0, %arg4, %arg5] : memref<1x1666x1024xf32>
        %4 = arith.addf %2, %3 : f32
        affine.store %4, %alloc[0, %arg4, %arg5] : memref<1x1666x1024xf32>
      }
    }
  }
  return %alloc : memref<1x1666x1024xf32>
}

func.func @reshape_from_temp_buffer(%arg0: memref<1x512xbf16>, %arg1: memref<1x512xbf16>) -> (memref<512xbf16>) attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c1 = arith.constant 1 : index
  %tmp = memref.alloc() : memref<1x512xbf16>
  scf.for %i = %c0 to %c512 step %c1 {
    %c0_0 = arith.constant 0 : index
    %v0 = memref.load %arg0[%c0_0, %i] : memref<1x512xbf16>
    %v1 = memref.load %arg1[%c0_0, %i] : memref<1x512xbf16>
    %sum = arith.addf %v0, %v1 : bf16
    memref.store %sum, %tmp[%c0_0, %i] : memref<1x512xbf16>
  }
  %shape = memref.alloc() : memref<1xindex>
  %c0_index = arith.constant 0 : index
  %c512_index = arith.constant 512 : index
  memref.store %c512_index, %shape[%c0_index] : memref<1xindex>
  %reshape = memref.reshape %tmp(%shape): (memref<1x512xbf16>, memref<1xindex>) -> memref<512xbf16>
  return %reshape : memref<512xbf16>
}

func.func @reshape_from_output_buffer(%arg0: memref<1x512xbf16>, %arg1: memref<1x512xbf16>) -> (memref<1x512xbf16>, memref<512xbf16>) attributes {OperatorType = "Broadcast", compute_capability = "", hacc.function_kind = #hacc.function_kind<HOST>, mindspore_kernel, process = "aicore"} {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x512xbf16>
    scf.for %i = %c0 to %c512 step %c1 {
      %c0_0 = arith.constant 0 : index
      %v = memref.load %arg0[%c0_0, %i] : memref<1x512xbf16>
      memref.store %v, %alloc[%c0_0, %i] : memref<1x512xbf16>
    }
    %shape = memref.alloc() : memref<1xindex>
    %c0_index = arith.constant 0 : index
    %c512_index = arith.constant 512 : index
    memref.store %c512_index, %shape[%c0_index] : memref<1xindex>
    %reshape = memref.reshape %alloc(%shape): (memref<1x512xbf16>, memref<1xindex>) -> memref<512xbf16>
    return %alloc, %reshape : memref<1x512xbf16>, memref<512xbf16>
}
