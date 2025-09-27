// RUN: mindspore-translate --json-to-mindspore %S/../../../test/ut/Mindspore-translate/Fused_Add_ReduceSum_Sub_Mul_Mul_Mul_Mul_Mul_split_540127469267638059.info | FileCheck %s

// CHECK: module
// CHECK: func.func
// CHECK: mindspore.add
// CHECK: mindspore.reduce_sum
// CHECK: mindspore.inplace_assign
// CHECK: mindspore.sub
// CHECK: mindspore.mul
// CHECK: mindspore.mul
// CHECK: mindspore.mul
// CHECK: mindspore.mul
// CHECK: mindspore.mul
// CHECK: return