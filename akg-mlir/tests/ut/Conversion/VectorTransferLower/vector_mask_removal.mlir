// RUN: akg-opt %s --vector-transfer-lower | FileCheck %s

// CHECK-LABEL: module {
// CHECK-NEXT:    func.func @Fused_Activation_77223251194711242(%arg0: memref<802816xf32>, %arg1: memref<802816xf32>) attributes {mindspore_kernel, scop.ignored} {
// CHECK-NEXT:      %cst = arith.constant 1.000000e+00 : f32
// CHECK-NEXT:      affine.for %arg2 = 0 to 802816 step 8 {
// CHECK-NEXT:        %cst_0 = arith.constant dense<1.000000e+00> : vector<8xf32>
// CHECK-NEXT:        %0 = vector.load %arg0[%arg2] : memref<802816xf32>, vector<8xf32>
// CHECK-NEXT:        %1 = arith.negf %0 : vector<8xf32>
// CHECK-NEXT:        %2 = math.exp %1 : vector<8xf32>
// CHECK-NEXT:        %3 = arith.addf %2, %cst_0 : vector<8xf32>
// CHECK-NEXT:        %4 = arith.divf %cst_0, %3 : vector<8xf32>
// CHECK-NEXT:        %5 = vector.load %arg0[%arg2] : memref<802816xf32>, vector<8xf32>
// CHECK-NEXT:        %6 = arith.mulf %5, %4 : vector<8xf32>
// CHECK-NEXT:        vector.store %6, %arg1[%arg2] : memref<802816xf32>, vector<8xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }


module {
  func.func @Fused_Activation_77223251194711242(%arg0: memref<802816xf32>, %arg1: memref<802816xf32>) attributes {mindspore_kernel, scop.ignored} {
    %cst = arith.constant 1.000000e+00 : f32
    affine.for %arg2 = 0 to 802816 step 8 {
      %cst_0 = arith.constant dense<1.000000e+00> : vector<8xf32>
      %cst_1 = arith.constant 0.000000e+00 : f32
      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<802816xf32>, vector<8xf32>
      %1 = arith.negf %0 : vector<8xf32>
      %2 = math.exp %1 : vector<8xf32>
      %3 = arith.addf %2, %cst_0 : vector<8xf32>
      %4 = arith.divf %cst_0, %3 : vector<8xf32>
      %cst_2 = arith.constant 0.000000e+00 : f32
      %5 = vector.transfer_read %arg0[%arg2], %cst_2 : memref<802816xf32>, vector<8xf32>
      %6 = arith.mulf %5, %4 : vector<8xf32>
      vector.transfer_write %6, %arg1[%arg2] : vector<8xf32>, memref<802816xf32>
    }
    return
  }
}