// RUN: akg-opt %s -split-input-file | akg-opt | FileCheck %s
// RUN: akg-opt %s --mlir-print-op-generic -split-input-file | akg-opt | FileCheck %s
 
// CHECK-DAG: #[[$BASE_MAP1:map[0-9]*]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: #[[$SUBVIEW_MAP1:map[0-9]*]] = affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>
// CHECK-DAG: #[[$SUBVIEW_MAP11:map[0-9]*]] = affine_map<() -> (4)>
// CHECK-DAG: #[[$SUBVIEW_MAP12:map[0-9]*]] = affine_map<()[s0] -> (s0)>
 
// CHECK-LABEL: func @fusion_subview(%arg0
func.func @fusion_subview(%arg0 : index, %arg1 : index, %arg2 : index) {
  // CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  // CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
 
  %0 = memref.alloc() : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>>
  // CHECK: subview %{{.*}}[%[[c0]], %[[c0]], %[[c0]]] [%{{.*}}, %{{.*}}, %{{.*}}] [%[[c1]], %[[c1]], %[[c1]]] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
  %1 = fusion.subview %0[%c0, %c0, %c0][%arg0, %arg1, %arg2][%c1, %c1, %c1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<?x?x?xf32, strided<[?, ?, ?], offset: ?>>
 
  %2 = memref.alloc()[%arg2] : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>>
  // CHECK: fusion.subview %{{.*}}[%[[c1]]] [%{{.*}}] [%[[c1]]] :
  // CHECK-SAME: memref<64xf32, #[[$BASE_MAP1]]>
  // CHECK-SAME: to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %3 = fusion.subview %2[%c1][%arg0][%c1]
    : memref<64xf32, affine_map<(d0)[s0] -> (d0 + s0)>> to
      memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
 
  %4 = memref.alloc() : memref<64x22xf32, strided<[22, 1]>>
  // CHECK: fusion.subview %{{.*}}[%[[c0]], %[[c1]]] [%{{.*}}, %{{.*}}] [%[[c1]], %[[c0]]] :
  // CHECK-SAME: memref<64x22xf32, strided<[22, 1]>>
  // CHECK-SAME: to memref<?x?xf32, strided<[?, ?], offset: ?>>
  %5 = fusion.subview %4[%c0, %c1][%arg0, %arg1][%c1, %c0]
    : memref<64x22xf32, strided<[22, 1], offset: 0>> to
      memref<?x?xf32, strided<[?, ?], offset: ?>>
 
  // CHECK: fusion.subview %{{.*}}[0, 2, 0] [4, 4, 4] [1, 1, 1] :
  // CHECK-SAME: memref<8x16x4xf32, strided<[64, 4, 1]>>
  // CHECK-SAME: to memref<4x4x4xf32, strided<[64, 4, 1], offset: 8>>
  %6 = fusion.subview %0[0, 2, 0][4, 4, 4][1, 1, 1]
    : memref<8x16x4xf32, strided<[64, 4, 1], offset: 0>> to
      memref<4x4x4xf32, strided<[64, 4, 1], offset: 8>>
 
  %7 = memref.alloc(%arg1, %arg2) : memref<?x?xf32>
  // CHECK: fusion.subview {{%.*}}[0, 0] [4, 4] [1, 1] :
  // CHECK-SAME: memref<?x?xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[?, 1], offset: ?>>
  %8 = fusion.subview %7[0, 0][4, 4][1, 1]
    : memref<?x?xf32> to memref<4x4xf32, strided<[?, 1], offset: ?>>
 
  %9 = memref.alloc() : memref<16x4xf32>
  // CHECK: fusion.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [{{%.*}}, {{%.*}}] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[?, ?], offset: ?>>
  %10 = fusion.subview %9[%arg1, %arg1][4, 4][%arg2, %arg2]
    : memref<16x4xf32> to memref<4x4xf32, strided<[?, ?], offset: ?>>
 
  // CHECK: fusion.subview {{%.*}}[{{%.*}}, {{%.*}}] [4, 4] [2, 2] :
  // CHECK-SAME: memref<16x4xf32>
  // CHECK-SAME: to memref<4x4xf32, strided<[8, 2], offset: ?>>
  %11 = fusion.subview %9[%arg1, %arg2][4, 4][2, 2]
    : memref<16x4xf32> to memref<4x4xf32, strided<[8, 2], offset: ?>>
 
  %12 = memref.alloc() : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>>
  // CHECK: fusion.subview
  // CHECK-SAME: [1, 9, 1, 4, 1]
  // CHECK-SAME: memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>> to memref<9x4xf32, strided<[?, ?], offset: ?>>
  %13 = fusion.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1], offset: 0>> to memref<9x4xf32, strided<[?, ?], offset: ?>>
  // CHECK: fusion.subview
  // CHECK-SAME: [1, 9, 1, 4, 1]
  // CHECK-SAME: memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1]>> to memref<1x9x4xf32, strided<[?, ?, ?], offset: ?>>
  %14 = fusion.subview %12[%arg1, %arg1, %arg1, %arg1, %arg1][1, 9, 1, 4, 1][%arg2, %arg2, %arg2, %arg2, %arg2] : memref<1x9x1x4x1xf32, strided<[36, 36, 4, 4, 1], offset: 0>> to memref<1x9x4xf32, strided<[?, ?, ?], offset: ?>>
 
  %15 = memref.alloc(%arg1, %arg2)[%c0, %c1, %arg1, %arg0, %arg0, %arg2, %arg2] : memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>>
  // CHECK: fusion.subview %{{.*}}[0, 0, 0, 0, 0, 0] [1, %{{.*}}, 5, 1, %{{.*}}, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?xf32, strided<[?, ?, ?], offset: ?>>
  %16 = fusion.subview %15[0, 0, 0, 0, 0, 0][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] : memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?xf32, strided<[?, ?, ?], offset: ?>>
  // CHECK: fusion.subview %{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [1, %{{.*}}, 5, 1, %{{.*}}, 1] [1, 1, 1, 1, 1, 1]  :
  // CHECK-SAME: memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?x1xf32, strided<[?, ?, ?, ?], offset: ?>>
  %17 = fusion.subview %15[%arg1, %arg1, %arg1, %arg1, %arg1, %arg1][1, %arg1, 5, 1, %arg2, 1][1, 1, 1, 1, 1, 1] :  memref<1x?x5x1x?x1xf32, strided<[?, ?, ?, ?, ?, ?], offset: ?>> to memref<?x5x?x1xf32, strided<[?, ?, ?, ?], offset: ?>>
 
  %18 = memref.alloc() : memref<1x8xf32>
  // CHECK: fusion.subview %{{.*}}[0, 0] [1, 8] [1, 1]  : memref<1x8xf32> to memref<8xf32>
  %19 = fusion.subview %18[0, 0][1, 8][1, 1] : memref<1x8xf32> to memref<8xf32>
 
  %20 = memref.alloc() : memref<8x16x4xf32>
  // CHECK: fusion.subview %{{.*}}[0, 0, 0] [1, 16, 4] [1, 1, 1]  : memref<8x16x4xf32> to memref<16x4xf32>
  %21 = fusion.subview %20[0, 0, 0][1, 16, 4][1, 1, 1] : memref<8x16x4xf32> to memref<16x4xf32>
 
  %22 = fusion.subview %20[3, 4, 2][1, 6, 3][1, 1, 1] : memref<8x16x4xf32> to memref<6x3xf32, strided<[4, 1], offset: 210>>
 
  %23 = memref.alloc() : memref<f32>
  %78 = fusion.subview %23[] [] []  : memref<f32> to memref<f32>
 
  /// Subview with only leading operands.
  %24 = memref.alloc() : memref<5x3xf32>
  // CHECK: fusion.subview %{{.*}}[2, 0] [3, 3] [1, 1] : memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>
  %25 = fusion.subview %24[2, 0][3, 3][1, 1]: memref<5x3xf32> to memref<3x3xf32, strided<[3, 1], offset: 6>>
 
  /// Rank-reducing subview with only leading operands.
  // CHECK: fusion.subview %{{.*}}[1, 0] [1, 3] [1, 1] : memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>
  %26 = fusion.subview %24[1, 0][1, 3][1, 1]: memref<5x3xf32> to memref<3xf32, strided<[1], offset: 3>>
 
  // Corner-case of 0-D rank-reducing subview with an offset.
  // CHECK: fusion.subview %{{.*}}[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP11]]>
  %27 = fusion.subview %24[1, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<() -> (4)>>
 
  // CHECK: fusion.subview %{{.*}}[%{{.*}}, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, #[[$SUBVIEW_MAP12]]>
  %28 = fusion.subview %24[%arg0, 1] [1, 1] [1, 1] : memref<5x3xf32> to memref<f32, affine_map<()[s0] -> (s0)>>
 
  // CHECK: fusion.subview %{{.*}}[0, %{{.*}}] [%{{.*}}, 1] [1, 1] : memref<?x?xf32> to memref<?xf32, #[[$SUBVIEW_MAP1]]>
  %a30 = memref.alloc(%arg0, %arg0) : memref<?x?xf32>
  %30 = fusion.subview %a30[0, %arg1][%arg2, 1][1, 1] : memref<?x?xf32> to memref<?xf32, affine_map<(d0)[s0, s1] -> (d0 * s1 + s0)>>
 
  %c8 = arith.constant 8 : index
  %a40 = memref.alloc() : memref<16x16xf32>
  // CHECK: fusion.subview
  %40 = fusion.subview %a40[%c8, 8][8, 8][1, 1]  :
    memref<16x16xf32> to memref<8x8xf32, affine_map<(d0, d1)[s0] -> (d0 * 16 + d1 + s0)>>
 
  return
}

// -----
 
// CHECK-LABEL: @fusion_broadcast
 
func.func @fusion_broadcast(%a: f32, %b: vector<f32>, %c: vector<16xf32>, %d: vector<1x16xf32>, %e: vector<8x1xf32>) -> vector<8x16xf32> {
  // CHECK: fusion.broadcast %{{.*}} : f32 to vector<f32>
  %0 = fusion.broadcast %a : f32 to vector<f32>
  // CHECK: fusion.broadcast %{{.*}} : vector<f32> to vector<4xf32>
  %1 = fusion.broadcast %b : vector<f32> to vector<4xf32>
  // CHECK: fusion.broadcast %{{.*}} : f32 to vector<16xf32>
  %2 = fusion.broadcast %a : f32 to vector<16xf32>
  // CHECK-NEXT: fusion.broadcast %{{.*}} : vector<16xf32> to vector<8x16xf32>
  %3 = fusion.broadcast %c : vector<16xf32> to vector<8x16xf32>
  // CHECK-NEXT: fusion.broadcast %{{.*}} : vector<1x16xf32> to vector<8x16xf32>
  %4 = fusion.broadcast %d : vector<1x16xf32> to vector<8x16xf32>
  // CHECK-NEXT: fusion.broadcast %{{.*}} : vector<8x1xf32> to vector<8x16xf32>
  %5 = fusion.broadcast %e : vector<8x1xf32> to vector<8x16xf32>
  return %4 : vector<8x16xf32>
}

// -----
 
// CHECK-LABEL: @transpose_fp
func.func @transpose_fp(%arg0: vector<3x7xf32>) -> vector<7x3xf32> {
  // CHECK: %[[X:.*]] = fusion.transpose %{{.*}}, [1, 0] : vector<3x7xf32> to vector<7x3xf32>
  %0 = fusion.transpose %arg0, [1, 0] : vector<3x7xf32> to vector<7x3xf32>
  // CHECK: return %[[X]] : vector<7x3xf32>
  return %0 : vector<7x3xf32>
}
 
// CHECK-LABEL: @transpose_int
func.func @transpose_int(%arg0: vector<11x7x3x2xi32>) -> vector<2x11x7x3xi32> {
  // CHECK: %[[X:.*]] = fusion.transpose %{{.*}}, [3, 0, 1, 2] : vector<11x7x3x2xi32> to vector<2x11x7x3xi32>
  %0 = fusion.transpose %arg0, [3, 0, 1, 2] : vector<11x7x3x2xi32> to vector<2x11x7x3xi32>
  // CHECK: return %[[X]] : vector<2x11x7x3xi32>
  return %0 : vector<2x11x7x3xi32>
}
 
// CHECK-LABEL: @transpose_fp_0d
func.func @transpose_fp_0d(%arg0: vector<f32>) -> vector<f32> {
  // CHECK: %[[X:.*]] = fusion.transpose %{{.*}}, [] : vector<f32> to vector<f32>
  %0 = fusion.transpose %arg0, [] : vector<f32> to vector<f32>
  // CHECK: return %[[X]] : vector<f32>
  return %0 : vector<f32>
}
 
// CHECK-LABEL: @transpose_int_0d
func.func @transpose_int_0d(%arg0: vector<i32>) -> vector<i32> {
  // CHECK: %[[X:.*]] = fusion.transpose %{{.*}}, [] : vector<i32> to vector<i32>
  %0 = fusion.transpose %arg0, [] : vector<i32> to vector<i32>
  // CHECK: return %[[X]] : vector<i32>
  return %0 : vector<i32>
}

// -----

// CHECK: func.func @multi_load_test(%[[ARG:.+]]: memref<?x?xf32>)
func.func @multi_load_test(%arg : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  // CHECK: fusion.multi_load %[[ARG]]
  %1 = fusion.multi_load %arg[%c0, %c0] : memref<?x?xf32>, f32
  // CHECK: fusion.multi_load %[[ARG]]
  %2 = fusion.multi_load %arg[%c0, %c0] : memref<?x?xf32>, vector<8x1xf32>
  return
}

// -----

func.func @fusion_store_op(%arg0 : memref<?x?xf32>) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %1 = fusion.load %arg0[%c0, %c0] : memref<?x?xf32>, f32
  %2 = fusion.load %arg0[%c0, %c0] : memref<?x?xf32>, vector<8x1xf32>
  %3 = fusion.load %arg0[%c0, %c0], %cst {in_bounds = [false, true]} : memref<?x?xf32>, vector<8x1xf32>
  // CHECK: fusion.store
  // CHECK-SAME: f32
  // CHECK-SAME: memref<?x?xf32>
  fusion.store %1, %arg0[%c0, %c0] : f32, memref<?x?xf32>
  // CHECK: fusion.store
  // CHECK-SAME: vector<8x1xf32>
  // CHECK-SAME: memref<?x?xf32>
  fusion.store %2, %arg0[%c0, %c0] : vector<8x1xf32>, memref<?x?xf32>
  // CHECK: fusion.store
  // CHECK-SAME: {in_bounds = [false, true]}
  // CHECK-SAME: vector<8x1xf32>
  // CHECK-SAME: memref<?x?xf32>
  fusion.store %3, %arg0[%c0, %c0] {in_bounds = [false, true]} : vector<8x1xf32>, memref<?x?xf32>
  return
}
