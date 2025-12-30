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

func.func @test_vadd_npuvector_scalar(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[CST]] : memref<128xf32>, f32) outs(%[[UB_OUT]] : memref<128xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<128xf32>) outs(%[[SV_OUT]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %cst : memref<128xf32>, !npuvector<128xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b : f32 to !npuvector<128xf32>
  %0 = arith.addf %v0, %v1 : !npuvector<128xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

func.func @test_vadd_npuvector_scalar_commute(%arg0 : memref<128xf32>, %arg1 : memref<128xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %{{.*}} = arith.constant 0.000000e+00 : f32
  // CHECK: %[[SV0:.*]] = memref.subview %arg0[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<128xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %[[CST]] : memref<128xf32>, f32) outs(%[[UB_OUT]] : memref<128xf32>)
  // CHECK: %[[SV_OUT:.*]] = memref.subview %arg1[0] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<128xf32>) outs(%[[SV_OUT]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %v0 = npuvector.transfer_read %arg0[%c0], %cst : memref<128xf32>, !npuvector<128xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b : f32 to !npuvector<128xf32>
  %0 = arith.addf %v1, %v0 : !npuvector<128xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<128xf32>, memref<128xf32>
  return
}

// -----

func.func @test_vadd_npuvector_scalar_dynamic(%arg0 : memref<?xf32>, %arg1 : memref<?xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?xf32>
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg0[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: %[[UB0:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB0]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<?xf32, strided<[1]>>) outs(%[[UB0]] : memref<?xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK: %[[CST1:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB_OUT:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB_OUT]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB0]], %{{.*}}) outs(%[[UB_OUT]] : memref<?xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<?xf32>) outs(%[[SUBVIEW_DST]] : memref<?xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %v0 = npuvector.transfer_read %arg0[%c0][%dim][%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %b = arith.constant 1.000000e+00 : f32
  %v1 = npuvector.broadcast %b[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %0 = arith.addf %v0, %v1 : !npuvector<?xf32>
  npuvector.transfer_write %0, %arg1[%c0] : !npuvector<?xf32>, memref<?xf32>
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

// -----

func.func @test_minimal_scf_loop_npuvector() {
  // CHECK-LABEL: func.func @test_minimal_scf_loop
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C3072:.*]] = arith.constant 3072 : index
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<3072xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<3072xf32>)
  // CHECK: %[[RES:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C3072]] step %[[C3072]] iter_args(%[[ACC:.*]] = %[[UB]]) -> (memref<3072xf32>) {
  // CHECK:   scf.yield %[[ACC]] : memref<3072xf32>
  // CHECK: }
  
  %c0 = arith.constant 0 : index
  %c3072 = arith.constant 3072 : index
  %cst = arith.constant dense<1.0> : !npuvector<3072xf32>
  %0 = scf.for %i = %c0 to %c3072 step %c3072 iter_args(%acc = %cst) -> (!npuvector<3072xf32>) {
    scf.yield %acc : !npuvector<3072xf32>
  }
  return
}

// -----

func.func @test_minimal_scf_loop() {
  // CHECK-LABEL: func.func @test_minimal_scf_loop
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C3072:.*]] = arith.constant 3072 : index
  // CHECK: %[[CST:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[UB:.*]] = memref.alloc() : memref<3072xf32>
  // CHECK: hivm.hir.vbrc ins(%[[CST]] : f32) outs(%[[UB]] : memref<3072xf32>)
  // CHECK: %[[RES:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C3072]] step %[[C3072]] iter_args(%[[ACC:.*]] = %[[UB]]) -> (memref<3072xf32>) {
  // CHECK:   scf.yield %[[ACC]] : memref<3072xf32>
  // CHECK: }
  
  %c0 = arith.constant 0 : index
  %c3072 = arith.constant 3072 : index
  %cst = arith.constant dense<1.0> : vector<3072xf32>
  %0 = scf.for %i = %c0 to %c3072 step %c3072 iter_args(%acc = %cst) -> (vector<3072xf32>) {
    scf.yield %acc : vector<3072xf32>
  }
  return
}

// -----

func.func @static_npuvector_transfer_read_write(%in: memref<1024xf32>, %out: memref<1024xf32>) {
  // CHECK-LABEL: func.func @static_npuvector_transfer_read_write
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xf32>, %[[OUT:.*]]: memref<1024xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<1024xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<1024xf32>) outs(%[[SUBVIEW_DST]] : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  npuvector.transfer_write %vec, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32> 
  return
}

// -----

func.func @test_negf_npuvector(%in: memref<1024xf32>, %out: memref<1024xf32>) {
  // CHECK-LABEL: func.func @test_negf_npuvector
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xf32>, %[[OUT:.*]]: memref<1024xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: %[[IN_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xf32, strided<[1]>>) outs(%[[IN_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[ZERO_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%[[ZERO]] : f32) outs(%[[ZERO_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[RES_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vsub ins(%[[ZERO_ALLOC]], %[[IN_ALLOC]] : memref<1024xf32>, memref<1024xf32>) outs(%[[RES_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[RES_ALLOC]] : memref<1024xf32>) outs(%[[SUBVIEW_DST]] : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  %neg = arith.negf %vec : !npuvector<1024xf32>
  npuvector.transfer_write %neg, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @test_exp_npuvector(%in: memref<1024xf32>, %out: memref<1024xf32>) {
  // CHECK-LABEL: func.func @test_exp_npuvector
  // CHECK-SAME: (%[[IN:.*]]: memref<1024xf32>, %[[OUT:.*]]: memref<1024xf32>)
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: %[[IN_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<1024xf32, strided<[1]>>) outs(%[[IN_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[RES_ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vexp ins(%[[IN_ALLOC]] : memref<1024xf32>) outs(%[[RES_ALLOC]] : memref<1024xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[RES_ALLOC]] : memref<1024xf32>) outs(%[[SUBVIEW_DST]] : memref<1024xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %vec = npuvector.transfer_read %in[%c0], %padding : memref<1024xf32>, !npuvector<1024xf32>
  %res = math.exp %vec : !npuvector<1024xf32>
  npuvector.transfer_write %res, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @dynamic_npuvector_transfer_read_write(%in: memref<?xf32>, %out: memref<?xf32>) {
  // CHECK-LABEL: func.func @dynamic_npuvector_transfer_read_write
  // CHECK-SAME: (%[[IN:.*]]: memref<?xf32>, %[[OUT:.*]]: memref<?xf32>)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[DIM:.*]] = memref.dim %[[IN]], %[[C0]] : memref<?xf32>
  // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %[[IN]][0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<?xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<?xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %[[OUT]][0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
  // CHECK: hivm.hir.store ins(%[[ALLOC]] : memref<?xf32>) outs(%[[SUBVIEW_DST]] : memref<?xf32, strided<[1]>>)
  %c0 = arith.constant 0 : index
  %padding = arith.constant 0.0 : f32
  %dim = memref.dim %in, %c0 : memref<?xf32>
  %max_vl = arith.constant 4096 : index
  %vec = npuvector.transfer_read %in[%c0][%dim][%max_vl], %padding : memref<?xf32>, !npuvector<?xf32>
  npuvector.transfer_write %vec, %out[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @simple_broadcast_static(%arg0: f32, %out: memref<1024xf32>) {
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<1024xf32>)
  %c0 = arith.constant 0 : index
  %0 = npuvector.broadcast %arg0 : f32 to !npuvector<1024xf32>
  npuvector.transfer_write %0, %out[%c0] : !npuvector<1024xf32>, memref<1024xf32>
  return
}

// -----

func.func @simple_broadcast_dynamic(%mem: memref<?xf32>, %out: memref<?xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %[[C0]] : memref<?xf32>
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 16384 : index}
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC]] : memref<?xf32>)
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %dim = memref.dim %mem, %c0 : memref<?xf32>
  %0 = npuvector.broadcast %cst_0[%dim] [%c4096] : f32 to !npuvector<?xf32>
  npuvector.transfer_write %0, %out[%c0] : !npuvector<?xf32>, memref<?xf32>
  return
}

// -----

func.func @test_static_reduction(%arg0: memref<16xf32>, %arg1: memref<f32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ARG0:.*]][0] [16] [1] : memref<16xf32> to memref<16xf32, strided<[1]>>
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
    // CHECK: hivm.hir.load ins(%[[SUBVIEW]] : memref<16xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<16xf32>)
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xf32>
    // CHECK: hivm.hir.vreduce <sum> ins(%[[ALLOC]] : memref<16xf32>) outs(%[[UB_REDUCE]] : memref<1xf32>) reduce_dims = [0]
    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG1:.*]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32>
    // CHECK: hivm.hir.store ins(%[[UB_REDUCE]] : memref<1xf32>) outs(%[[CAST]] : memref<1xf32>)
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %0 = npuvector.transfer_read %arg0[%c0], %f0 : memref<16xf32>, !npuvector<16xf32>
    %1 = npuvector.reduction <add>, %0 : !npuvector<16xf32> into f32
    memref.store %1, %arg1[] : memref<f32>  
    return
}

// -----

func.func @test_dynamic_reduction(%input: memref<?xf32>, %output: memref<f32>) {
    // CHECK: %[[C0:.*]] = arith.constant 0 : index
    // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
    // CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[DIM:.*]] = memref.dim %[[INPUT:.*]], %[[C0]] : memref<?xf32>
    // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[INPUT]][0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
    // CHECK: hivm.hir.load ins(%[[SUBVIEW]] : memref<?xf32, strided<[1]>>) outs(%[[ALLOC]] : memref<?xf32>)
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xf32>
    // CHECK: hivm.hir.vreduce <sum> ins(%[[ALLOC]] : memref<?xf32>) outs(%[[UB_REDUCE]] : memref<1xf32>) reduce_dims = [0]
    // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[OUTPUT:.*]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32>
    // CHECK: hivm.hir.store ins(%[[UB_REDUCE]] : memref<1xf32>) outs(%[[CAST]] : memref<1xf32>)
    %c0 = arith.constant 0 : index
    %c4096 = arith.constant 4096 : index
    %f0 = arith.constant 0.0 : f32
    %dim = memref.dim %input, %c0 : memref<?xf32>
    %vec = npuvector.transfer_read %input[%c0][%dim][%c4096], %f0 : memref<?xf32>, !npuvector<?xf32>
    %res = npuvector.reduction <add>, %vec : !npuvector<?xf32> into f32
    memref.store %res, %output[] : memref<f32>
    return
}

// -----

func.func @test_npuvector_reduction_mul(%arg0: memref<16xf32>, %arg1: memref<f32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_mul
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xf32>
    // CHECK: hivm.hir.vreduce <prod> ins(%[[ALLOC]] : memref<16xf32>) outs(%[[UB_REDUCE]] : memref<1xf32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 1.0 : f32
    %0 = npuvector.transfer_read %arg0[%c0], %f0 : memref<16xf32>, !npuvector<16xf32>
    %1 = npuvector.reduction <mul>, %0 : !npuvector<16xf32> into f32
    memref.store %1, %arg1[] : memref<f32>
    return
}

// -----

func.func @test_npuvector_reduction_min(%arg0: memref<16xf32>, %arg1: memref<f32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_min
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xf32>
    // CHECK: hivm.hir.vreduce <min> ins(%[[ALLOC]] : memref<16xf32>) outs(%[[UB_REDUCE]] : memref<1xf32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %0 = npuvector.transfer_read %arg0[%c0], %f0 : memref<16xf32>, !npuvector<16xf32>
    %1 = npuvector.reduction <minnumf>, %0 : !npuvector<16xf32> into f32
    memref.store %1, %arg1[] : memref<f32>
    return
}

// -----

func.func @test_npuvector_reduction_max(%arg0: memref<16xf32>, %arg1: memref<f32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_max
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xf32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xf32>
    // CHECK: hivm.hir.vreduce <max> ins(%[[ALLOC]] : memref<16xf32>) outs(%[[UB_REDUCE]] : memref<1xf32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f32
    %0 = npuvector.transfer_read %arg0[%c0], %f0 : memref<16xf32>, !npuvector<16xf32>
    %1 = npuvector.reduction <maxnumf>, %0 : !npuvector<16xf32> into f32
    memref.store %1, %arg1[] : memref<f32>
    return
}

// -----

func.func @test_npuvector_reduction_and(%arg0: memref<16xi32>, %arg1: memref<i32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_and
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xi32>
    // CHECK: hivm.hir.vreduce <andi> ins(%[[ALLOC]] : memref<16xi32>) outs(%[[UB_REDUCE]] : memref<1xi32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %cst = arith.constant -1 : i32
    %0 = npuvector.transfer_read %arg0[%c0], %cst : memref<16xi32>, !npuvector<16xi32>
    %1 = npuvector.reduction <and>, %0 : !npuvector<16xi32> into i32
    memref.store %1, %arg1[] : memref<i32>
    return
}

// -----

func.func @test_npuvector_reduction_or(%arg0: memref<16xi32>, %arg1: memref<i32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_or
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xi32>
    // CHECK: hivm.hir.vreduce <ori> ins(%[[ALLOC]] : memref<16xi32>) outs(%[[UB_REDUCE]] : memref<1xi32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%c0], %cst : memref<16xi32>, !npuvector<16xi32>
    %1 = npuvector.reduction <or>, %0 : !npuvector<16xi32> into i32
    memref.store %1, %arg1[] : memref<i32>
    return
}

// -----

func.func @test_npuvector_reduction_xor(%arg0: memref<16xi32>, %arg1: memref<i32>) {
    // CHECK-LABEL: func.func @test_npuvector_reduction_xor
    // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<16xi32>
    // CHECK: %[[UB_REDUCE:.*]] = memref.alloc() : memref<1xi32>
    // CHECK: hivm.hir.vreduce <xori> ins(%[[ALLOC]] : memref<16xi32>) outs(%[[UB_REDUCE]] : memref<1xi32>) reduce_dims = [0]
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%c0], %cst : memref<16xi32>, !npuvector<16xi32>
    %1 = npuvector.reduction <xor>, %0 : !npuvector<16xi32> into i32
    memref.store %1, %arg1[] : memref<i32>
    return
}

// -----

func.func @test_elementwise_static(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %arg3: f32) {
  // CHECK-LABEL: func.func @test_elementwise_static
  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%arg3 : f32) outs(%[[ALLOC]] : memref<1024xf32>)
  // CHECK: scf.for %[[ARG4:.*]] = %c0 to %c1024 step %c1024_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %arg0[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<1024xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_1]] : memref<1024xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK:   %[[CST_2:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SUBVIEW_3:.*]] = memref.subview %arg1[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_4:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW_3]] : memref<1024xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_4]] : memref<1024xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK:   %[[ALLOC_5:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.vmul ins(%[[ALLOC]], %[[ALLOC_1]] : memref<1024xf32>, memref<1024xf32>) outs(%[[ALLOC_5]] : memref<1024xf32>)
  // CHECK:   %[[ALLOC_6:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ALLOC_5]], %[[ALLOC_4]] : memref<1024xf32>, memref<1024xf32>) outs(%[[ALLOC_6]] : memref<1024xf32>)
  // CHECK:   %[[SUBVIEW_7:.*]] = memref.subview %arg2[%[[ARG4]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[ALLOC_6]] : memref<1024xf32>) outs(%[[SUBVIEW_7]] : memref<1024xf32, strided<[1], offset: ?>>)
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c1024_0 = arith.constant 1024 : index
  %0 = npuvector.broadcast %arg3 : f32 to !npuvector<1024xf32>
  scf.for %arg4 = %c0 to %c1024 step %c1024_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %1 = npuvector.transfer_read %arg0[%arg4], %cst : memref<1024xf32>, !npuvector<1024xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %2 = npuvector.transfer_read %arg1[%arg4], %cst_1 : memref<1024xf32>, !npuvector<1024xf32>
    %3 = arith.mulf %0, %1 : !npuvector<1024xf32>
    %4 = arith.addf %3, %2 : !npuvector<1024xf32>
    npuvector.transfer_write %4, %arg2[%arg4] : !npuvector<1024xf32>, memref<1024xf32>
  }
  return
}

// -----

func.func @test_elementwise_dynamic(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: f32, %arg3: f32) {
  // CHECK-LABEL: func.func @test_elementwise_dynamic
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: f32, %[[ARG3:.*]]: f32)
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = memref.dim %[[ARG0]], %[[C0]] : memref<?xf32>
  // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK: %[[ALLOC:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vbrc ins(%[[ARG2]] : f32) outs(%[[ALLOC]] : memref<?xf32>)
  // CHECK: %[[ALLOC0:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC0]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vbrc ins(%[[ARG3]] : f32) outs(%[[ALLOC0]] : memref<?xf32>)
  // CHECK: scf.for %[[ARG4:.*]] = %[[C0]] to %[[DIM]] step %[[DIM]] {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %[[ARG0]][%[[ARG4]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_1]] : memref<?xf32>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK:   %[[ALLOC_4:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_4]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vmul ins(%[[ALLOC_1]], %[[ALLOC]] : memref<?xf32>, memref<?xf32>) outs(%[[ALLOC_4]] : memref<?xf32>)
  // CHECK:   %[[ALLOC_7:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_7]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ALLOC_4]], %[[ALLOC0]] : memref<?xf32>, memref<?xf32>) outs(%[[ALLOC_7]] : memref<?xf32>)
  // CHECK:   %[[SUBVIEW_10:.*]] = memref.subview %[[ARG1]][%[[ARG4]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[ALLOC_7]] : memref<?xf32>) outs(%[[SUBVIEW_10]] : memref<?xf32, strided<[1], offset: ?>>)
  // CHECK: }
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %c4096 = arith.constant 4096 : index
  %0 = npuvector.broadcast %arg2[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %1 = npuvector.broadcast %arg3[%dim] [%c4096] : f32 to !npuvector<?xf32>
  scf.for %arg4 = %c0 to %dim step %dim {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = npuvector.transfer_read %arg0[%arg4][%dim][%c4096], %cst : memref<?xf32>, !npuvector<?xf32>
    %3 = arith.mulf %2, %0 : !npuvector<?xf32>
    %4 = arith.addf %3, %1 : !npuvector<?xf32>
    npuvector.transfer_write %4, %arg1[%arg4] : !npuvector<?xf32>, memref<?xf32>
  }
  return
}

// -----

func.func @test_reduction_static(%arg0: memref<1024xf32>, %arg1: memref<f32>) {
  // CHECK-LABEL: func.func @test_reduction_static
  // CHECK: %[[ALLOC_INIT:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_INIT]] : memref<1024xf32>)
  // CHECK: %[[RES:.*]] = scf.for %[[ARG2:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG3:.*]] = %[[ALLOC_INIT]]) -> (memref<1024xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%[[ARG2]]] [1024] [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_READ:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<1024xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_READ]] : memref<1024xf32>)
  // CHECK:   %[[ALLOC_ADD:.*]] = memref.alloc() : memref<1024xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ARG3]], %[[ALLOC_READ]] : memref<1024xf32>, memref<1024xf32>) outs(%[[ALLOC_ADD]] : memref<1024xf32>)
  // CHECK:   scf.yield %[[ALLOC_ADD]] : memref<1024xf32>
  // CHECK: }
  // CHECK: %[[ALLOC_REDUCE:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[RES]] : memref<1024xf32>) outs(%[[ALLOC_REDUCE]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32>
  // CHECK: hivm.hir.store ins(%[[ALLOC_REDUCE]] : memref<1xf32>) outs(%[[CAST]] : memref<1xf32>)
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c1024_0 = arith.constant 1024 : index
  %cst_1 = arith.constant dense<0.000000e+00> : !npuvector<1024xf32>
  %0 = scf.for %arg2 = %c0 to %c1024 step %c1024_0 iter_args(%arg3 = %cst_1) -> (!npuvector<1024xf32>) {
    %cst_2 = arith.constant 0.000000e+00 : f32
    %2 = npuvector.transfer_read %arg0[%arg2], %cst_2 : memref<1024xf32>, !npuvector<1024xf32>
    %3 = arith.addf %arg3, %2 : !npuvector<1024xf32>
    scf.yield %3 : !npuvector<1024xf32>
  }
  %1 = npuvector.reduction <add>, %0 : !npuvector<1024xf32> into f32
  memref.store %1, %arg1[] : memref<f32>
  return
}

// -----

func.func @test_reduction_dynamic(%arg0: memref<?xf32>, %arg1: memref<f32>) {
  // CHECK-LABEL: func.func @test_reduction_dynamic
  // CHECK: %[[DIM:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: %[[ALLOC_INIT:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[ALLOC_INIT]] {buffer_size_in_byte = 16384 : index}
  // CHECK: hivm.hir.vbrc ins(%{{.*}} : f32) outs(%[[ALLOC_INIT]] : memref<?xf32>)
  // CHECK: %[[RES:.*]] = scf.for %[[ARG2:.*]] = %{{.*}} to %[[DIM]] step %[[DIM]] iter_args(%[[ARG3:.*]] = %[[ALLOC_INIT]]) -> (memref<?xf32>) {
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %{{.*}}[%[[ARG2]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   %[[ALLOC_READ:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[ALLOC_READ]] {buffer_size_in_byte = 16384 : index}
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<?xf32, strided<[1], offset: ?>>) outs(%[[ALLOC_READ]] : memref<?xf32>)
  // CHECK:   %[[ALLOC_ADD:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   hivm.hir.vadd ins(%[[ARG3]], %[[ALLOC_READ]] : memref<?xf32>, memref<?xf32>) outs(%[[ALLOC_ADD]] : memref<?xf32>)
  // CHECK:   scf.yield %[[ALLOC_ADD]] : memref<?xf32>
  // CHECK: }
  // CHECK: %[[ALLOC_REDUCE:.*]] = memref.alloc() : memref<1xf32>
  // CHECK: hivm.hir.vreduce <sum> ins(%[[RES]] : memref<?xf32>) outs(%[[ALLOC_REDUCE]] : memref<1xf32>) reduce_dims = [0]
  // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32>
  // CHECK: hivm.hir.store ins(%[[ALLOC_REDUCE]] : memref<1xf32>) outs(%[[CAST]] : memref<1xf32>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?xf32>
  %cst = arith.constant 0.000000e+00 : f32
  %c4096 = arith.constant 4096 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = npuvector.broadcast %cst_0[%dim] [%c4096] : f32 to !npuvector<?xf32>
  %1 = scf.for %arg2 = %c0 to %dim step %dim iter_args(%arg3 = %0) -> (!npuvector<?xf32>) {
    %cst_1 = arith.constant 0.000000e+00 : f32
    %3 = npuvector.transfer_read %arg0[%arg2][%dim][%c4096], %cst_1 : memref<?xf32>, !npuvector<?xf32>
    %4 = arith.addf %arg3, %3 : !npuvector<?xf32>
    scf.yield %4 : !npuvector<?xf32>
  }
  %2 = npuvector.reduction <add>, %1 : !npuvector<?xf32> into f32
  memref.store %2, %arg1[] : memref<f32>
  return
}

// -----

func.func @test_npuvector_extf_static(%arg0: memref<128xbf16>, %arg1: memref<128xf32>) {
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, {{.*}}>) outs(%[[UB0]] : memref<128xbf16>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xbf16>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xbf16>, !npuvector<128xbf16>
    %1 = npuvector.extf %0 : !npuvector<128xbf16> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_truncf_static(%arg0: memref<128xf32>, %arg1: memref<128xbf16>) {
  // CHECK-LABEL: func.func @test_npuvector_truncf_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xbf16>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xbf16>) outs(%[[SV1]] : memref<128xbf16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.truncf %0 : !npuvector<128xf32> to !npuvector<128xbf16>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xbf16>, memref<128xbf16>
  }
  return
}

// -----

func.func @test_npuvector_extsi_static(%arg0: memref<128xi8>, %arg1: memref<128xi32>) {
  // CHECK-LABEL: func.func @test_npuvector_extsi_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i8
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.extsi %0 : !npuvector<128xi8> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_extui_static(%arg0: memref<128xi8>, %arg1: memref<128xi32>) {
  // CHECK-LABEL: func.func @test_npuvector_extui_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i8
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi8, {{.*}}>) outs(%[[UB0]] : memref<128xi8>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi8>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i8 = arith.constant 0 : i8
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i8 : memref<128xi8>, !npuvector<128xi8>
    %1 = npuvector.extui %0 : !npuvector<128xi8> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_trunci_static(%arg0: memref<128xi32>, %arg1: memref<128xi8>) {
  // CHECK-LABEL: func.func @test_npuvector_trunci_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi8>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xi8>) round_mode = <truncwithoverflow>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi8> to memref<128xi8, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi8>) outs(%[[SV1]] : memref<128xi8, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.trunci %0 : !npuvector<128xi32> to !npuvector<128xi8>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi8>, memref<128xi8>
  }
  return
}

// -----

func.func @test_npuvector_sitofp_static(%arg0: memref<128xi32>, %arg1: memref<128xf32>) {
  // CHECK-LABEL: func.func @test_npuvector_sitofp_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.sitofp %0 : !npuvector<128xi32> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_uitofp_static(%arg0: memref<128xi32>, %arg1: memref<128xf32>) {
  // CHECK-LABEL: func.func @test_npuvector_uitofp_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xi32>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xf32>) outs(%[[SV1]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %c0_i32 = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg2], %c0_i32 : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.uitofp %0 : !npuvector<128xi32> to !npuvector<128xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_fptosi_static(%arg0: memref<128xf32>, %arg1: memref<128xi32>) {
  // CHECK-LABEL: func.func @test_npuvector_fptosi_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xi32>) round_mode = <trunc>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.fptosi %0 : !npuvector<128xf32> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_fptoui_static(%arg0: memref<128xf32>, %arg1: memref<128xi32>) {
  // CHECK-LABEL: func.func @test_npuvector_fptoui_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xf32>) outs(%[[UB1]] : memref<128xi32>) round_mode = <trunc>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi32>) outs(%[[SV1]] : memref<128xi32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.fptoui %0 : !npuvector<128xf32> to !npuvector<128xi32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_bitcast_static(%arg0: memref<128xbf16>, %arg1: memref<128xi16>) {
  // CHECK-LABEL: func.func @test_npuvector_bitcast_static
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, {{.*}}>) outs(%[[UB0]] : memref<128xbf16>)
  // CHECK:   %[[UB1:.*]] = hivm.hir.bitcast %[[UB0]] : memref<128xbf16> -> memref<128xi16>
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xi16> to memref<128xi16, {{.*}}>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<128xi16>) outs(%[[SV1]] : memref<128xi16, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2], %cst : memref<128xbf16>, !npuvector<128xbf16>
    %1 = npuvector.bitcast %0 : !npuvector<128xbf16> to !npuvector<128xi16>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<128xi16>, memref<128xi16>
  }
  return
}

// -----

func.func @test_npuvector_cmpi_static(%arg0: memref<128xi32>, %arg1: memref<128xi32>, %arg2: memref<128xi32>) {
  // CHECK-LABEL: func.func @test_npuvector_cmpi_static
  // CHECK: scf.for %[[ARG3:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0 : i32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xi32, {{.*}}>) outs(%[[UB0]] : memref<128xi32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.load ins(%[[SV1]] : memref<128xi32, {{.*}}>) outs(%[[UB1]] : memref<128xi32>)
  // CHECK:   %[[UB_OUT:.*]] = memref.alloc() : memref<128xi1>
  // CHECK:   hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<128xi32>, memref<128xi32>) outs(%[[UB_OUT]] : memref<128xi1>) compare_mode = <gt>
  // CHECK:   %[[SEL_RES:.*]] = memref.alloc() : memref<128xi32>
  // CHECK:   hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<128xi1>, memref<128xi32>, memref<128xi32>) outs(%[[SEL_RES]] : memref<128xi32>)
  // CHECK:   %[[SV2:.*]] = memref.subview %arg2[%[[ARG3]]] [128] [1] : memref<128xi32> to memref<128xi32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%{{.*}} : memref<128xi32>) outs(%[[SV2]] : memref<128xi32, {{.*}}>)

  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0 : i32
    %0 = npuvector.transfer_read %arg0[%arg3], %cst : memref<128xi32>, !npuvector<128xi32>
    %1 = npuvector.transfer_read %arg1[%arg3], %cst : memref<128xi32>, !npuvector<128xi32>
    %2 = npuvector.cmpi sgt, %0, %1 : !npuvector<128xi32> to !npuvector<128xi1>
    %3 = npuvector.select %2, %0, %1 : !npuvector<128xi1>, !npuvector<128xi32>
    npuvector.transfer_write %3, %arg2[%arg3] : !npuvector<128xi32>, memref<128xi32>
  }
  return
}

// -----

func.func @test_npuvector_cmpf_static(%arg0: memref<128xf32>, %arg1: memref<128xf32>, %arg2: memref<128xf32>) {
  // CHECK-LABEL: func.func @test_npuvector_cmpf_static
  // CHECK: scf.for %[[ARG3:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xf32, {{.*}}>) outs(%[[UB0]] : memref<128xf32>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.load ins(%[[SV1]] : memref<128xf32, {{.*}}>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[UB_OUT:.*]] = memref.alloc() : memref<128xi1>
  // CHECK:   hivm.hir.vcmp ins(%[[UB0]], %[[UB1]] : memref<128xf32>, memref<128xf32>) outs(%[[UB_OUT]] : memref<128xi1>) compare_mode = <gt>
  // CHECK:   %[[SEL_RES:.*]] = memref.alloc() : memref<128x{{.*}}>
  // CHECK:   hivm.hir.vsel ins(%[[UB_OUT]], %[[UB0]], %[[UB1]] : memref<128xi1>, memref<128x{{.*}}>, memref<128x{{.*}}>) outs(%[[SEL_RES]] : memref<128x{{.*}}>)
  // CHECK:   %[[SV2:.*]] = memref.subview %arg2[%[[ARG3]]] [128] [1] : memref<128xf32> to memref<128xf32, {{.*}}>
  // CHECK:   hivm.hir.store ins(%{{.*}} : memref<128xf32>) outs(%[[SV2]] : memref<128xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c128_0 = arith.constant 128 : index
  scf.for %arg3 = %c0 to %c128 step %c128_0 {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = npuvector.transfer_read %arg0[%arg3], %cst : memref<128xf32>, !npuvector<128xf32>
    %1 = npuvector.transfer_read %arg1[%arg3], %cst : memref<128xf32>, !npuvector<128xf32>
    %2 = npuvector.cmpf ogt, %0, %1 : !npuvector<128xf32> to !npuvector<128xi1>
    %3 = npuvector.select %2, %0, %1 : !npuvector<128xi1>, !npuvector<128xf32>
    npuvector.transfer_write %3, %arg2[%arg3] : !npuvector<128xf32>, memref<128xf32>
  }
  return
}

// -----

func.func @test_npuvector_mixed_type_conversion(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>) {
  // CHECK: %[[CST1:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK: scf.for %[[ARG2:.*]] = %c0 to %c128 step %c128_0 {
  // CHECK:   %[[CST0:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:   %[[SV0:.*]] = memref.subview %arg0[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, strided<[1], offset: ?>>
  // CHECK:   %[[UB0:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.load ins(%[[SV0]] : memref<128xbf16, strided<[1], offset: ?>>) outs(%[[UB0]] : memref<128xbf16>) init_out_buffer = false may_implicit_transpose_with_last_axis = false
  // CHECK:   %[[UB1:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<128xbf16>) outs(%[[UB1]] : memref<128xf32>)
  // CHECK:   %[[UB2:.*]] = memref.alloc() : memref<128xf32>
  // CHECK:   hivm.hir.vmul ins(%[[UB1]], %[[CST1]] : memref<128xf32>, f32) outs(%[[UB2]] : memref<128xf32>)
  // CHECK:   %[[UB3:.*]] = memref.alloc() : memref<128xbf16>
  // CHECK:   hivm.hir.vcast ins(%[[UB2]] : memref<128xf32>) outs(%[[UB3]] : memref<128xbf16>)
  // CHECK:   %[[SV1:.*]] = memref.subview %arg1[%[[ARG2]]] [128] [1] : memref<128xbf16> to memref<128xbf16, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[UB3]] : memref<128xbf16>) outs(%[[SV1]] : memref<128xbf16, strided<[1], offset: ?>>)
  %c0 = arith.constant 0 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 2.000000e+00 : f32
  %c128_0 = arith.constant 128 : index
  %0 = npuvector.broadcast %cst : f32 to !npuvector<128xf32>
  scf.for %arg2 = %c0 to %c128 step %c128_0 {
    %cst_1 = arith.constant 0.000000e+00 : bf16
    %1 = npuvector.transfer_read %arg0[%arg2], %cst_1 : memref<128xbf16>, !npuvector<128xbf16>
    %2 = npuvector.extf %1 : !npuvector<128xbf16> to !npuvector<128xf32>
    %3 = arith.mulf %2, %0 : !npuvector<128xf32>
    %4 = npuvector.truncf %3 : !npuvector<128xf32> to !npuvector<128xbf16>
    npuvector.transfer_write %4, %arg1[%arg2] : !npuvector<128xbf16>, memref<128xbf16>
  }
  return
}

// -----

func.func @test_npuvector_extf_dynamic(%arg0: memref<?xbf16>, %arg1: memref<?xf32>) {
  // CHECK-LABEL: func.func @test_npuvector_extf_dynamic
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg0, %[[C0]] : memref<?xbf16>
  // CHECK: %[[C4096:.*]] = arith.constant 4096 : index
  // CHECK: scf.for %[[ARG2:.*]] = %[[C0]] to %[[DIM]] step %[[DIM]] {
  // CHECK:   %[[CST:.*]] = arith.constant 0.000000e+00 : bf16
  // CHECK:   %[[SUBVIEW:.*]] = memref.subview %arg0[%[[ARG2]]] [%[[DIM]]] [1] : memref<?xbf16> to memref<?xbf16, strided<[1], offset: ?>>
  // CHECK:   %[[UB0:.*]] = memref.alloc(%[[DIM]]) : memref<?xbf16>
  // CHECK:   annotation.mark %[[UB0]] {buffer_size_in_byte = 8192 : index} : memref<?xbf16>
  // CHECK:   hivm.hir.load ins(%[[SUBVIEW]] : memref<?xbf16, strided<[1], offset: ?>>) outs(%[[UB0]] : memref<?xbf16>)
  // CHECK:   %[[UB1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK:   annotation.mark %[[UB1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK:   hivm.hir.vcast ins(%[[UB0]] : memref<?xbf16>) outs(%[[UB1]] : memref<?xf32>)
  // CHECK:   %[[SUBVIEW_OUT:.*]] = memref.subview %arg1[%[[ARG2]]] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: ?>>
  // CHECK:   hivm.hir.store ins(%[[UB1]] : memref<?xf32>) outs(%[[SUBVIEW_OUT]] : memref<?xf32, strided<[1], offset: ?>>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg0, %c0 : memref<?xbf16>
  %c4096 = arith.constant 4096 : index
  scf.for %arg2 = %c0 to %dim step %dim {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = npuvector.transfer_read %arg0[%arg2] [%dim] [%c4096], %cst : memref<?xbf16>, !npuvector<?xbf16>
    %1 = npuvector.extf %0 : !npuvector<?xbf16> to !npuvector<?xf32>
    npuvector.transfer_write %1, %arg1[%arg2] : !npuvector<?xf32>, memref<?xf32>
  }
  return
}

// -----

func.func @test_rank0_transfer_read_write_dynamic(%arg0: memref<f32>, %arg1: memref<1x?xf32>, %arg2: memref<?xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[DIM:.*]] = memref.dim %arg1, %[[C1]] : memref<1x?xf32>
  // CHECK: %[[SCALAR:.*]] = memref.load %arg0[] : memref<f32>
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg2[0] [%[[DIM]]] [1] : memref<?xf32> to memref<?xf32, {{.*}}>
  // CHECK: %[[UB1:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB1]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<?xf32, {{.*}}>) outs(%[[UB1]] : memref<?xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc(%[[DIM]]) : memref<?xf32>
  // CHECK: annotation.mark %[[UB_OUT]] {buffer_size_in_byte = 16384 : index} : memref<?xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB1]], %[[SCALAR]] : memref<?xf32>, f32) outs(%[[UB_OUT]] : memref<?xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0, 0] [1, %[[DIM]]] [1, 1] : memref<1x?xf32> to memref<?xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<?xf32>) outs(%[[SUBVIEW_DST]] : memref<?xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %dim = memref.dim %arg1, %c1 : memref<1x?xf32>
  %c4096 = arith.constant 4096 : index
  %pad = arith.constant 0.0 : f32
  %vec_scalar = npuvector.transfer_read %arg0[] [%dim] [%c4096], %pad : memref<f32>, !npuvector<?xf32>
  %vec_arg2 = npuvector.transfer_read %arg2[%c0] [%dim] [%c4096], %pad : memref<?xf32>, !npuvector<?xf32>
  %vec_add = arith.addf %vec_scalar, %vec_arg2 : !npuvector<?xf32>
  npuvector.transfer_write %vec_add, %arg1[%c0, %c0] : !npuvector<?xf32>, memref<1x?xf32>
  return
}

// -----

func.func @test_rank0_transfer_read_write_static(%arg0: memref<f32>, %arg1: memref<1x170xf32>,%arg2: memref<170xf32>) {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[SCALAR:.*]] = memref.load %arg0[] : memref<f32>
  // CHECK: %[[SUBVIEW_SRC:.*]] = memref.subview %arg2[0] [170] [1] : memref<170xf32> to memref<170xf32, {{.*}}>
  // CHECK: %[[UB1:.*]] = memref.alloc() : memref<170xf32>
  // CHECK: hivm.hir.load ins(%[[SUBVIEW_SRC]] : memref<170xf32, {{.*}}>) outs(%[[UB1]] : memref<170xf32>)
  // CHECK: %[[UB_OUT:.*]] = memref.alloc() : memref<170xf32>
  // CHECK: hivm.hir.vadd ins(%[[UB1]], %[[SCALAR]] : memref<170xf32>, f32) outs(%[[UB_OUT]] : memref<170xf32>)
  // CHECK: %[[SUBVIEW_DST:.*]] = memref.subview %arg1[0, 0] [1, 170] [1, 1] : memref<1x170xf32> to memref<170xf32, {{.*}}>
  // CHECK: hivm.hir.store ins(%[[UB_OUT]] : memref<170xf32>) outs(%[[SUBVIEW_DST]] : memref<170xf32, {{.*}}>)
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %vec_scalar = npuvector.transfer_read %arg0[], %pad : memref<f32>, !npuvector<170xf32>
  %vec_arg2 = npuvector.transfer_read %arg2[%c0], %pad : memref<170xf32>, !npuvector<170xf32>
  %vec_add = arith.addf %vec_scalar, %vec_arg2 : !npuvector<170xf32>
  npuvector.transfer_write %vec_add, %arg1[%c0, %c0] : !npuvector<170xf32>, memref<1x170xf32>
  return
}
