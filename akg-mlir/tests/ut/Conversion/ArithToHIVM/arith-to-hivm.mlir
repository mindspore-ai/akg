// RUN: akg-opt %s -convert-arith-to-hivm | FileCheck %s

func.func @test_addf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}>) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)
  
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.addf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_addi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)
  
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.addi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_mulf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}>) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vmul ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)
  
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.mulf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_muli(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmul ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.muli %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_subf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}>) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vsub ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.subf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_subi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vsub ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.subi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_divf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}>) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vdiv ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.divf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_divsi(%arg0 : memref<32x32xi64>, %arg1 : memref<32x32xi64>, %arg2 : memref<32x32xi64>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i64
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi64, {{.*}}>) outs(%[[UB0]] : memref<32x32xi64>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi64, {{.*}}>) outs(%[[UB1]] : memref<32x32xi64>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.vdiv ins(%[[UB0]], %[[UB1]] : memref<32x32xi64>, memref<32x32xi64>) outs(%[[UB_OUT]] : memref<32x32xi64>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi64>) outs(%[[SV_OUT]] : memref<32x32xi64, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i64
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi64>, vector<32x32xi64>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi64>, vector<32x32xi64>
  %0 = arith.divsi %v0, %v1 : vector<32x32xi64>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi64>, memref<32x32xi64>
  return
}

// -----

func.func @test_divui(%arg0 : memref<32x32xi64>, %arg1 : memref<32x32xi64>, %arg2 : memref<32x32xi64>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i64
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi64, {{.*}}>) outs(%[[UB0]] : memref<32x32xi64>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi64, {{.*}}>) outs(%[[UB1]] : memref<32x32xi64>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.vdiv ins(%[[UB0]], %[[UB1]] : memref<32x32xi64>, memref<32x32xi64>) outs(%[[UB_OUT]] : memref<32x32xi64>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi64>) outs(%[[SV_OUT]] : memref<32x32xi64, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i64
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi64>, vector<32x32xi64>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi64>, vector<32x32xi64>
  %0 = arith.divui %v0, %v1 : vector<32x32xi64>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi64>, memref<32x32xi64>
  return
}

// -----

func.func @test_maxsi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmax ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.maxsi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_maxui(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmax ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.maxui %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_minsi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmin ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.minsi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_minui(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}>) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmin ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.minui %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_truncf_f32_f16(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf16>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf16>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf16>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf16>) outs(%[[SV_OUT]] : memref<32x32xf16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.truncf %v0 : vector<32x32xf32> to vector<32x32xf16>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xf16>, memref<32x32xf16>
  return
}

// -----

func.func @test_truncf_f32_bf16(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xbf16>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xbf16>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xbf16>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xbf16>) outs(%[[SV_OUT]] : memref<32x32xbf16, {{.*}}>)
  
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.truncf %v0 : vector<32x32xf32> to vector<32x32xbf16>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xbf16>, memref<32x32xbf16>
  return
}

// -----

func.func @test_extf_f16_f32(%arg0 : memref<32x32xf16>, %arg1 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f16
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf16> to memref<32x32xf16, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf16>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf16, {{.*}}>) outs(%[[UB0]] : memref<32x32xf16>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xf16>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f16
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf16>, vector<32x32xf16>
  %0 = arith.extf %v0 : vector<32x32xf16> to vector<32x32xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_extf_bf16_f32(%arg0 : memref<32x32xbf16>, %arg1 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : bf16
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xbf16> to memref<32x32xbf16, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xbf16>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xbf16, {{.*}}>) outs(%[[UB0]] : memref<32x32xbf16>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xbf16>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xbf16>, vector<32x32xbf16>
  %0 = arith.extf %v0 : vector<32x32xbf16> to vector<32x32xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_extui_i8_i16(%arg0 : memref<32x32xi8>, %arg1 : memref<32x32xi16>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i8
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi8> to memref<32x32xi8, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi8>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi8, {{.*}}>) outs(%[[UB0]] : memref<32x32xi8>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi16>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xi8>) outs(%[[UB_OUT]] : memref<32x32xi16>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi16> to memref<32x32xi16, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi16>) outs(%[[SV_OUT]] : memref<32x32xi16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i8 
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi8>, vector<32x32xi8>
  %0 = arith.extui %v0 : vector<32x32xi8> to vector<32x32xi16>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xi16>, memref<32x32xi16>
  return
}

// -----

func.func @test_fptosi_f32_i32(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi32>) round_mode = <trunc>
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.fptosi %v0 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_fptoui_f32_i32(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi32>) round_mode = <trunc>
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.fptoui %v0 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_sitofp_i32_f32(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.sitofp %v0 : vector<32x32xi32> to vector<32x32xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_trunci_i64_i32(%arg0 : memref<32x32xi64>, %arg1 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i64
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi64> to memref<32x32xi64, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi64>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi64, {{.*}}>) outs(%[[UB0]] : memref<32x32xi64>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vcast ins(%[[UB0]] : memref<32x32xi64>) outs(%[[UB_OUT]] : memref<32x32xi32>) round_mode = <truncwithoverflow>
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i64
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi64>, vector<32x32xi64>
  %0 = arith.trunci %v0 : vector<32x32xi64> to vector<32x32xi32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_bitcast_f32_i32(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}>) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = hivm.hir.bitcast %[[UB0]] : memref<32x32xf32> -> memref<32x32xi32>
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %0 = arith.bitcast %v0 : vector<32x32xf32> to vector<32x32xi32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_bitcast_i32_f32(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}>) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = hivm.hir.bitcast %[[UB0]] : memref<32x32xi32> -> memref<32x32xf32>
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %0 = arith.bitcast %v0 : vector<32x32xi32> to vector<32x32xf32>
  vector.transfer_write %0, %arg1[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

// -----

func.func @test_cmpf_oeq(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>)
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf oeq, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_one(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ne>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf one, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ole(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <le>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ole, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_olt(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <lt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf olt, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_oge(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ge>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf oge, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ogt(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <gt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ogt, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ueq(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>)
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ueq, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_une(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ne>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf une, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ule(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <le>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ule, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ult(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <lt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ult, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_uge(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ge>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf uge, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpf_ugt(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <gt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xf32>, memref<32x32xf32>) outs(%[[SEL_RES]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xf32>) outs(%[[SV_OUT:.*]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.cmpf ugt, %v0, %v1 : vector<32x32xf32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xf32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_cmpi_eq(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_eq
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>)
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi eq, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_ne(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_ne
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ne>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi ne, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_slt(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_slt
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <lt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi slt, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_sgt(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_sgt
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <gt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi sgt, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_sle(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_sle
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <le>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi sle, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_sge(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_sge
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ge>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi sge, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_ult(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_ult
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <lt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi ult, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_ugt(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_ugt
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <gt>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi ugt, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_ule(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_ule
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <le>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi ule, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_cmpi_uge(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK-LABEL: func.func @test_cmpi_uge
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi1>
  // CHECK: hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi1>) compare_mode = <ge>
  // CHECK-FIXME: %[[SEL_RES:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK-FIXME: hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<32x32xi1>, memref<32x32xi32>, memref<32x32xi32>) outs(%[[SEL_RES]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%{{.*}} : memref<32x32xi32>) outs(%[[SV_OUT:.*]] : memref<32x32xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.cmpi uge, %v0, %v1 : vector<32x32xi32>
  %v3 = arith.select %res0, %v0, %v1 : vector<32x32xi1>, vector<32x32xi32>
  vector.transfer_write %v3, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_mulsi_extended(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT_0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: %[[UB_OUT_1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmulext ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT_0]], %[[UB_OUT_1]] : memref<32x32xi32>, memref<32x32xi32>)
  // CHECK: %[[SV_OUT_0:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_0]] : memref<32x32xi32>) outs(%[[SV_OUT_0]] : memref<32x32xi32, {{.*}})
  // CHECK: %[[SV_OUT_1:.*]] = memref.subview %arg3[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_1]] : memref<32x32xi32>) outs(%[[SV_OUT_1]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0, %res1 = arith.mulsi_extended %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  vector.transfer_write %res1, %arg3[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_mului_extended(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>, %arg3 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT_0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: %[[UB_OUT_1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmulext ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT_0]], %[[UB_OUT_1]] : memref<32x32xi32>, memref<32x32xi32>)
  // CHECK: %[[SV_OUT_0:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_0]] : memref<32x32xi32>) outs(%[[SV_OUT_0]] : memref<32x32xi32, {{.*}})
  // CHECK: %[[SV_OUT_1:.*]] = memref.subview %arg3[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT_1]] : memref<32x32xi32>) outs(%[[SV_OUT_1]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0, %res1 = arith.mului_extended %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  vector.transfer_write %res1, %arg3[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_andi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vand ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.andi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_ori(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vor ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.ori %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_xori(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vxor ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.xori %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_remsi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmod ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.remsi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_remui(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vmod ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.remui %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_minnumf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vmin ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.minnumf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_minimumf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vmin ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.minimumf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_maxnumf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vmax ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.maxnumf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_maximumf(%arg0 : memref<32x32xf32>, %arg1 : memref<32x32xf32>, %arg2 : memref<32x32xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xf32, {{.*}}) outs(%[[UB0]] : memref<32x32xf32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xf32, {{.*}}) outs(%[[UB1]] : memref<32x32xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vmax ins(%[[UB0]], %[[UB1]] : memref<32x32xf32>, memref<32x32xf32>) outs(%[[UB_OUT]] : memref<32x32xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xf32>) outs(%[[SV_OUT]] : memref<32x32xf32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xf32>, vector<32x32xf32>
  %res0 = arith.maximumf %v0, %v1 : vector<32x32xf32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}

// -----

func.func @test_shli(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vshl ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.shli %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_shrsi(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vshr ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>) round : true
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.shrsi %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_shrui(%arg0 : memref<32x32xi32>, %arg1 : memref<32x32xi32>, %arg2 : memref<32x32xi32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0 : i32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<32x32xi32, {{.*}}) outs(%[[UB0]] : memref<32x32xi32>)
  // CHECK: %[[SV1:.*]] = memref.subview %arg1[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.load ins(%[[SV1]] : memref<32x32xi32, {{.*}}) outs(%[[UB1]] : memref<32x32xi32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<32x32xi32>
  // CHECK: hivm.hir.vshr ins(%[[UB0]], %[[UB1]] : memref<32x32xi32>, memref<32x32xi32>) outs(%[[UB_OUT]] : memref<32x32xi32>) round : true
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg2[0, 0] [32, 32] [1, 1] : memref<32x32xi32> to memref<32x32xi32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<32x32xi32>) outs(%[[SV_OUT]] : memref<32x32xi32, {{.*}})

  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  %v0 = vector.transfer_read %arg0[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %v1 = vector.transfer_read %arg1[%c0, %c0], %cst : memref<32x32xi32>, vector<32x32xi32>
  %res0 = arith.shrui %v0, %v1 : vector<32x32xi32>
  vector.transfer_write %res0, %arg2[%c0, %c0] : vector<32x32xi32>, memref<32x32xi32>
  return
}

// -----

func.func @test_constant_broadcast(%arg0 : memref<32x32xf32>) {
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<32x32xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<32x32xf32>)
  // CHECK: %[[SV:.*]] = memref.subview %arg0[0, 0] [32, 32] [1, 1] : memref<32x32xf32> to memref<32x32xf32, strided<[32, 1]>>
  // CHECK: hivm.hir.store ins(%[[UB]] : memref<32x32xf32>) outs(%[[SV]] : memref<32x32xf32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %0 = arith.constant dense<1.0> : vector<32x32xf32>
  vector.transfer_write %0, %arg0[%c0, %c0] : vector<32x32xf32>, memref<32x32xf32>
  return
}
// -----

// CHECK: func.func @test_vector_reduction_xor_standard(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_xor_standard(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <xori> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <xor>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_add_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_add_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <add>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_mul_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_mul_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <prod> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <mul>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_and_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_and_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant -1 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <andi> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <and>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_or_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_or_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <ori> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <or>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_maxsi_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_maxsi_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <max> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <maxsi>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_minui_i32(%[[ARG0:.*]]: memref<16xi32>, %[[ARG1:.*]]: memref<1xi32>)
func.func @test_vector_reduction_minui_i32(%arg0: memref<16xi32>, %arg1: memref<1xi32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0 : i32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xi32> to memref<16xi32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xi32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xi32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xi32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xi32>
  // CHECK: hivm.hir.vreduce <min> ins(%[[ALLOC_READ]] : memref<16xi32>) outs(%[[ALLOC_RED]] : memref<1xi32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xi32> to memref<1xi32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xi32>) outs(%[[SUBVIEW_WRITE]] : memref<1xi32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xi32>, vector<16xi32>
  %1 = vector.reduction <minui>, %0 : vector<16xi32> into i32
  %2 = vector.broadcast %1 : i32 to vector<1xi32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xi32>, memref<1xi32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_add_f32(%[[ARG0:.*]]: memref<16xf32>, %[[ARG1:.*]]: memref<1xf32>)
func.func @test_vector_reduction_add_f32(%arg0: memref<16xf32>, %arg1: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xf32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[ALLOC_READ]] : memref<16xf32>) outs(%[[ALLOC_RED]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xf32>) outs(%[[SUBVIEW_WRITE]] : memref<1xf32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xf32>, vector<16xf32>
  %1 = vector.reduction <add>, %0 : vector<16xf32> into f32
  %2 = vector.broadcast %1 : f32 to vector<1xf32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_mul_f32(%[[ARG0:.*]]: memref<16xf32>, %[[ARG1:.*]]: memref<1xf32>)
func.func @test_vector_reduction_mul_f32(%arg0: memref<16xf32>, %arg1: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 1.0 : f32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xf32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <prod> ins(%[[ALLOC_READ]] : memref<16xf32>) outs(%[[ALLOC_RED]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xf32>) outs(%[[SUBVIEW_WRITE]] : memref<1xf32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xf32>, vector<16xf32>
  %1 = vector.reduction <mul>, %0 : vector<16xf32> into f32
  %2 = vector.broadcast %1 : f32 to vector<1xf32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  return
}

// -----

// CHECK: func.func @test_vector_reduction_maxnumf_f32(%[[ARG0:.*]]: memref<16xf32>, %[[ARG1:.*]]: memref<1xf32>)
func.func @test_vector_reduction_maxnumf_f32(%arg0: memref<16xf32>, %arg1: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.0 : f32
  // CHECK: %[[SUBVIEW_READ:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC_READ:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_READ]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC_READ]] : memref<16xf32>)
  // CHECK: %[[ALLOC_RED:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <max> ins(%[[ALLOC_READ]] : memref<16xf32>) outs(%[[ALLOC_RED]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[SUBVIEW_WRITE:.*]] = memref.subview %[[ARG1]][0] [1] [1] : memref<1xf32> to memref<1xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC_RED]] : memref<1xf32>) outs(%[[SUBVIEW_WRITE]] : memref<1xf32, strided<[1]>>)
  %0 = vector.transfer_read %arg0[%c0], %cst : memref<16xf32>, vector<16xf32>
  %1 = vector.reduction <maxnumf>, %0 : vector<16xf32> into f32
  %2 = vector.broadcast %1 : f32 to vector<1xf32>
  vector.transfer_write %2, %arg1[%c0] : vector<1xf32>, memref<1xf32>
  return
}

// -----

// CHECK: func.func @test_vector_transfer(%[[ARG0:.*]]: memref<16xf32>, %[[ARG1:.*]]: memref<16xf32>)
func.func @test_vector_transfer(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[ARG0]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<16xf32>)
  %0 = vector.transfer_read %arg0[%c0], %pad : memref<16xf32>, vector<16xf32>
  
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[ARG1]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<16xf32>) outs(%[[SUBVIEW_DST]] : memref<16xf32, strided<[1]>>)
  vector.transfer_write %0, %arg1[%c0] : vector<16xf32>, memref<16xf32>
  return
}
