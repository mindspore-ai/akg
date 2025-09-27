// RUN: mindspore-translate --json-to-mindspore %S/../../../test/ut/Mindspore-translate/symbolic_shape.info | FileCheck %s

// CHECK: module
// CHECK: func.func
// CHECK: frontend_symbol
// CHECK: mindspore.mul
// CHECK: frontend_symbol
// CHECK: return