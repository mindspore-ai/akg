// RUN: akg-opt %s  --force-convert-affine-for-to-affine-parallel='match-op-type=gpu-reduction'
// UNSUPPORTED: true

func.func @before_force_parallel(%in: memref<2x3x4xf32>, %out: memref<3xf32>) {
  %c0 = arith.constant 0 : index
  affine.for %j = 0 to 3 {
    affine.for %i = 0 to 2 {
      affine.for %k = 0 to 4 {
        %ld = affine.load %in[%i, %j, %k] : memref<2x3x4xf32>
        %res = affine.load %out[%j] : memref<3xf32>
        %res_update = arith.addf %res, %ld {reduction_axes = [0 : index, 2 : index], reduction_type = "x"} : f32 
        affine.store %res_update, %out[%j] : memref<3xf32>
      } {reduction_loop = true}
    } {reduction_loop = true}
  }
  return
}
