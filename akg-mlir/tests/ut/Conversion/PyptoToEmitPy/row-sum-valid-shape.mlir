// RUN: akg-opt %s -pypto-align-shape | FileCheck %s --check-prefix=SHAPE
// RUN: akg-opt %s -pypto-align-shape -convert-pypto-to-emitpy | akg-translate --allow-unregistered-dialect --mlir-to-pypto | FileCheck %s --check-prefix=PYTHON

// SHAPE-LABEL: func.func @row_sum_partial_result
// The data and scratch have physical [8, 512] and actual valid [3, 510].
// SHAPE: pypto.load {{.*}} padded[%c8, %c512] valid[%c3, %c510]
// SHAPE: pypto.create_tile valid[%c3, %c510] : <f32, [8, 512]>
// SHAPE: pypto.reduction {{.*}} -> !pypto.tile<f32, [8]>
// SHAPE: pypto.reshape %{{[0-9]+}} extents[{{.*}}] : !pypto.tile<f32, [8]> to !pypto.tile<f32, [1, 8]>

// PYTHON-LABEL: def row_sum_partial_result(
// PYTHON: pl.load({{.*}}, [8, 512], [3, 510])
// PYTHON: [[ALLOC:var_[0-9]+]] = pl.create_tile([8, 512]
// PYTHON-NEXT: [[SCRATCH:var_[0-9]+]] = pl.set_validshape([[ALLOC]], 3, 510)
// PYTHON-NEXT: pl.row_sum({{.*}}, [[SCRATCH]])
// PYTHON: [[RESULT:var_[0-9]+]] = pl.reshape({{.*}}, [1, 8])
// PYTHON-NEXT: pl.store([[RESULT]],

// SHAPE-LABEL: func.func @row_sum_transpose_input
// SHAPE: %[[TRANSPOSE:[0-9]+]] = pypto.transpose
// SHAPE-NEXT: %[[DATA:[0-9]+]] = pypto.reshape %[[TRANSPOSE]] valid[%c13, %c7]
// SHAPE-NEXT: %[[TMP:[0-9]+]] = pypto.create_tile valid[%c13, %c7] : <f32, [16, 8]>
// SHAPE-NEXT: pypto.reduction {{.*}} %[[DATA]], %[[TMP]]

// PYTHON-LABEL: def row_sum_transpose_input(
// PYTHON: pl.transpose(
// PYTHON: [[TILE:var_[0-9]+]] = pl.slice({{.*}}, [16, 8], [0, 0], valid_shape=[13, 7])
// PYTHON: [[TMP_ALLOC:var_[0-9]+]] = pl.create_tile([16, 8]
// PYTHON-NEXT: [[TMP:var_[0-9]+]] = pl.set_validshape([[TMP_ALLOC]], 13, 7)
// PYTHON-NEXT: pl.row_sum([[TILE]], [[TMP]])

module {
  emitpy.import import "pypto.language" as "pl"
  func.func @row_sum_partial_result(%src: memref<7x512xf32>, %dst: memref<1x8xf32>) attributes {arch = "Ascend910B3", emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c3 = arith.constant 3 : index
    %c510 = arith.constant 510 : index
    pypto.spmd (%block) in (1) {
      %input = pypto.load %src[%block, %c0] padded[%c7, %c510] valid[%c3, %c510] : memref<7x512xf32> -> !pypto.tile<f32, [7, 510]>
      %scratch = pypto.create_tile : <f32, [7, 510]>
      %result = pypto.reduction sum, row, %input, %scratch : !pypto.tile<f32, [7, 510]>, !pypto.tile<f32, [7, 510]> -> !pypto.tile<f32, [7]>
      %view = pypto.reshape %result : !pypto.tile<f32, [7]> to !pypto.tile<f32, [1, 7]>
      pypto.store %view, %dst[%block, %c0] : !pypto.tile<f32, [1, 7]>, memref<1x8xf32>
    }
    return
  }
  func.func @row_sum_transpose_input(%src: memref<8x16xf32>, %dst: memref<1x16xf32>) attributes {arch = "Ascend910B3", emitpy.out_args = array<i64: 1>, emitpy.pypto_kernel} {
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c13 = arith.constant 13 : index
    %c16 = arith.constant 16 : index
    pypto.spmd (%block) in (1) {
      %input = pypto.load %src[%block, %c0] padded[%c8, %c16] valid[%c7, %c13] : memref<8x16xf32> -> !pypto.tile<f32, [8, 16]>
      %transposed = pypto.transpose %input, 0, 1 : !pypto.tile<f32, [8, 16]> to !pypto.tile<f32, [16, 8]>
      %scratch = pypto.create_tile : <f32, [16, 8]>
      %result = pypto.reduction sum, row, %transposed, %scratch : !pypto.tile<f32, [16, 8]>, !pypto.tile<f32, [16, 8]> -> !pypto.tile<f32, [16]>
      %view = pypto.reshape %result : !pypto.tile<f32, [16]> to !pypto.tile<f32, [1, 16]>
      pypto.store %view, %dst[%block, %c0] : !pypto.tile<f32, [1, 16]>, memref<1x16xf32>
    }
    return
  }
}
