// RUN: akg-opt %s -split-input-file --legalize-bool | FileCheck %s

// CHECK-LABEL:    func.func @Fused_Mul_Maximum_Select_Mul_Select_Assign_fusion(
// CHECK-SAME:       %arg0: memref<1xf32>, %arg1: memref<1xi8>, %arg2: memref<1xi8>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<1xf32>
// CHECK:            scf.for {{.*}} {
// CHECK:              [[LOAD0:%.*]] = memref.load %arg0{{.*}} : memref<1xf32>
// CHECK:              [[MUL0:%.*]] = arith.mulf [[LOAD0]], {{.*}} : f32
// CHECK:              [[MAX0:%.*]] = arith.maximumf [[MUL0]], {{.*}} : f32
// CHECK:              [[COND0_I8:%.*]] = memref.load %arg1{{.*}} : memref<1xi8>
// CHECK:              [[LOAD1:%.*]] = memref.load %arg0{{.*}} : memref<1xf32>
// CHECK:              [[ZERO0:%.*]] = arith.constant 0.000000e+00 : f16
// CHECK:              [[COND0_F16:%.*]] = arith.uitofp [[COND0_I8]] : i8 to f16
// CHECK:              [[COND0_I1:%.*]] = arith.cmpf one, [[COND0_F16]], [[ZERO0]] : f16
// CHECK:              [[SEL0:%.*]] = arith.select [[COND0_I1]], [[MAX0]], [[LOAD1]] : f32
// CHECK:              [[MUL1:%.*]] = arith.mulf [[SEL0]], {{.*}} : f32
// CHECK:              [[COND1_I8:%.*]] = memref.load %arg2{{.*}} : memref<1xi8>
// CHECK:              [[ZERO1:%.*]] = arith.constant 0.000000e+00 : f16
// CHECK:              [[COND1_F16:%.*]] = arith.uitofp [[COND1_I8]] : i8 to f16
// CHECK:              [[COND1_I1:%.*]] = arith.cmpf one, [[COND1_F16]], [[ZERO1]] : f16
// CHECK:              [[SEL1:%.*]] = arith.select [[COND1_I1]], [[MUL1]], [[SEL0]] : f32
// CHECK:              memref.store [[SEL1]], %arg4{{.*}} : memref<1xf32>
// CHECK:            } {vector = 4096 : i64}
// CHECK:            return %arg4 : memref<1xf32>

func.func @Fused_Mul_Maximum_Select_Mul_Select_Assign_fusion(%arg0: memref<1xf32>, %arg1: memref<1xi1>, %arg2: memref<1xi1>, %arg3: memref<1xf32>, %arg4: memref<1xf32>) -> memref<1xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c1_0 = arith.constant 1 : index
  scf.for %arg5 = %c0 to %c1 step %c1_0 {
    %cst = arith.constant 2.000000e+00 : f32
    %cst_1 = arith.constant 1.000000e+00 : f32
    %cst_2 = arith.constant 5.000000e-01 : f32
    %c0_3 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0_3] : memref<1xf32>
    %1 = arith.mulf %0, %cst_2 : f32
    %2 = arith.maximumf %1, %cst_1 : f32
    %c0_4 = arith.constant 0 : index
    %3 = memref.load %arg1[%c0_4] : memref<1xi1>
    %c0_5 = arith.constant 0 : index
    %4 = memref.load %arg0[%c0_5] : memref<1xf32>
    %5 = arith.select %3, %2, %4 : f32
    %6 = arith.mulf %5, %cst : f32
    %c0_6 = arith.constant 0 : index
    %7 = memref.load %arg2[%c0_6] : memref<1xi1>
    %8 = arith.select %7, %6, %5 : f32
    %c0_7 = arith.constant 0 : index
    memref.store %8, %arg4[%c0_7] : memref<1xf32>
  } {vector = 4096 : i64}
  return %arg4 : memref<1xf32>
}

// CHECK-LABEL:    func.func @Fused_Mul_IsFinite_split(
// CHECK-SAME:       %arg0: memref<16xbf16>, %arg1: memref<1xbf16>, %arg2: memref<16xbf16>, %arg3: memref<16xi8>) -> (memref<16xbf16>, memref<16xi8>)
// CHECK:            scf.for {{.*}} {
// CHECK:              [[BOOL:%.*]] = arith.xori {{.*}} : i1
// CHECK:              [[BOOL_F16:%.*]] = arith.uitofp [[BOOL]] : i1 to f16
// CHECK:              [[BOOL_I8:%.*]] = arith.fptoui [[BOOL_F16]] : f16 to i8
// CHECK:              memref.store [[BOOL_I8]], %arg3{{.*}} : memref<16xi8>
// CHECK:            }
// CHECK:            return %arg2, %arg3 : memref<16xbf16>, memref<16xi8>

func.func @Fused_Mul_IsFinite_split(%arg0: memref<16xbf16>, %arg1: memref<1xbf16>, %arg2: memref<16xbf16>, %arg3: memref<16xi1>) -> (memref<16xbf16>, memref<16xi1>) {
  %true = arith.constant true
  %c0_i16 = arith.constant 0 : i16
  %c32512_i16 = arith.constant 32512 : i16
  %c255_i16 = arith.constant 255 : i16
  %c32767_i16 = arith.constant 32767 : i16
  %collapse_shape = memref.collapse_shape %arg1 [] : memref<1xbf16> into memref<bf16>
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %c1 = arith.constant 1 : index
  scf.for %arg4 = %c0 to %c16 step %c1 {
    %0 = memref.load %collapse_shape[] : memref<bf16>
    %1 = memref.load %arg0[%arg4] : memref<16xbf16>
    %2 = arith.extf %1 : bf16 to f32
    %3 = arith.extf %0 : bf16 to f32
    %4 = arith.mulf %2, %3 : f32
    %5 = arith.truncf %4 : f32 to bf16
    memref.store %5, %arg2[%arg4] : memref<16xbf16>
    %6 = memref.load %arg2[%arg4] : memref<16xbf16>
    %7 = arith.bitcast %6 : bf16 to i16
    %8 = arith.andi %7, %c32767_i16 : i16
    %9 = arith.andi %8, %c255_i16 : i16
    %10 = arith.andi %8, %c32512_i16 : i16
    %11 = arith.cmpi eq, %10, %c32512_i16 : i16
    %12 = arith.cmpi eq, %9, %c0_i16 : i16
    %13 = arith.andi %11, %12 : i1
    %14 = arith.cmpi ne, %9, %c0_i16 : i16
    %15 = arith.andi %11, %14 : i1
    %16 = arith.ori %13, %15 : i1
    %17 = arith.xori %16, %true : i1
    memref.store %17, %arg3[%arg4] : memref<16xi1>
  }
  return %arg2, %arg3 : memref<16xbf16>, memref<16xi1>
}
